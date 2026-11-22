# SPDX-License-Identifier: Apache-2.0
"""SM121 (GB10) Triton replacement for the TRT-LLM expert batched GEMM.

Same ``expert_plan`` / ``bmm`` / ``load_kernels`` interface as
``vllm.models.inkling2.ops.trtllm_bmm``.

S5-W2 upgrades (all 0-ULP bit-exact, same K accumulation order):
  - Panel-contiguous in-place relayout of W [E, N//PB, K//KB, PB, KB//PACK] and
    WS [E, N//PB, K//KB, KB//G, PB] (zero extra GPU memory; eliminates DRAM row
    misses & strided scale bursts).
  - Live-row-only stores (omask = rmask & nmask): cuts dirty L2 writes by ~85%.
  - eviction_policy="evict_first" on W and WS (protects hot activations/KV/Stack).
  - Bitshift e8m0->f32 scale decode for MXFP8 (replaces SFU tl.exp2).
  - PDL overlap: w13 triggers situ_quantize; w2 launches early behind situ_quantize,
    L2-prefetches its first 2 K-steps of W, gdc_waits, then triggers combine.

Prefill & High-Concurrency upgrades (all 0-ULP bit-exact, literal nd == 0):
  - Dead-CTA early exit (`t * TILE + sub * BM >= end`): eliminates ~50% wasted
    DRAM/compute at T=1024..2048 where tile=128 > live rows per expert.
  - Panel-aligned prefill/batched kernels (`_expert_panel_prefill_w13_kernel`,
    `_expert_panel_nv_w13_l256_dot64_kernel`, `_expert_panel_nv_w2_exact64_kernel`,
    `_expert_panel_mx_w2_kernel`): 100% contiguous 2D panel loads (ZERO vector
    div/mod), vectorized WS6 unpack once per KBP=128 panel (4x less WS6 unpack
    work), exact BK_DOT K-accumulation order (literal nd == 0), L2-friendly
    CTA ordering (pid0 = nb*SUB_M + sub, pid1 = t -> A read once from DRAM & W
    shared in L2 across sub-tiles), and adaptive BM=64 vs BM=128 by max_ctas.
  - Tuned tile=16 (B=2..16 concurrent decode) warps/stages on _expert_panel_decode_kernel
    without touching tile=8 single-stream decode.
  - Optional zero-scatter `row2tok` + row-major `as_row_major` fast path for
    co-integration with prefill quantization/routing fusions.
"""

import ctypes
import functools
import os
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

_TILES = ((8, 8), (128, 16), (384, 32), (768, 64), (4096, 128), (1 << 31, 256))


@dataclass(frozen=True)
class ExpertPlan:
    tile: int
    max_ctas: int
    kernels: tuple[int, int]

    @property
    def rows(self) -> int:
        return self.tile * self.max_ctas


@functools.cache
def expert_plan(
    tokens: int,
    topk: int,
    num_experts: int,
    nvfp4: bool,
    shapes: tuple[tuple[int, int], tuple[int, int]],
    tile: int | None = None,
) -> ExpertPlan:
    if tile is None:
        tile = next(t for max_tokens, t in _TILES if tokens <= max_tokens)
    rows = tokens * topk
    max_ctas = min(num_experts, rows) + (rows - min(num_experts, rows)) // tile
    return ExpertPlan(tile, max_ctas, (0, 0))


@triton.jit
def _scale_offsets(row, cols, COLS: tl.constexpr, TILE: tl.constexpr):
    if TILE >= 128:
        tile = (row // 128) * ((COLS + 3) // 4) + cols // 4
        return tile * 512 + (row % 32) * 16 + (row // 32 % 4) * 4 + cols % 4
    if TILE > 0:
        tile = (row // 8) * ((COLS + 3) // 4) + cols // 4
        return tile * 32 + (row % 8) * 4 + cols % 4
    return row * COLS + cols


@triton.jit
def _e2m1_f16(nib):
    bits = ((nib & 8) << 12) | ((nib & 7) << 9)
    return bits.to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def _e8m0_to_f32(s):
    s32 = s.to(tl.int32)
    f = (s32 << 23).to(tl.float32, bitcast=True)
    return tl.where(s32 == 0, 5.877471754111438e-39, tl.where(s32 == 255, float("nan"), f))


@triton.jit
def _apply_scales(vals, scales, R: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    """[R, BK] values * [R, BK // G] fp32 scales -> bf16 (exact)."""
    v = tl.reshape(vals.to(tl.float32), (R, BK // G, G)) * scales[:, :, None]
    return tl.reshape(v, (R, BK)).to(tl.bfloat16)


@triton.jit
def _e2m1_hw(packed):
    """uint8 [.., n] (2 e2m1 per byte) -> (lo, hi) fp16 via cvt.rn.f16x2.e2m1x2."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .b8 b0, b1, b2, b3;
        .reg .b32 t0, t1, t2, t3;
        mov.b32 {b0, b1, b2, b3}, $4;
        cvt.rn.f16x2.e2m1x2 t0, b0;
        cvt.rn.f16x2.e2m1x2 t1, b1;
        cvt.rn.f16x2.e2m1x2 t2, b2;
        cvt.rn.f16x2.e2m1x2 t3, b3;
        prmt.b32 $0, t0, t1, 0x5410;
        prmt.b32 $1, t2, t3, 0x5410;
        prmt.b32 $2, t0, t1, 0x7632;
        prmt.b32 $3, t2, t3, 0x7632;
        }
        """,
        "=r,=r,=r,=r,r",
        [packed],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _unpack_ws6_slice(ws_base, q_kg, KBG: tl.constexpr, BKG: tl.constexpr, PB: tl.constexpr):
    """Contiguous vectorized WS6 unpack for [BKG, PB] slice at K-group offset q_kg -> [PB, BKG] fp16."""
    bkg_off = tl.arange(0, BKG)
    bn2_off = tl.arange(0, PB // 2)
    bn4_off = tl.arange(0, PB // 4)
    lo_b = tl.load(
        ws_base + (q_kg + bkg_off[:, None]) * (PB // 2) + bn2_off[None, :],
        eviction_policy="evict_first",
    )
    hi_b = tl.load(
        ws_base + (KBG * (PB // 2)) + (q_kg + bkg_off[:, None]) * (PB // 4) + bn4_off[None, :],
        eviction_policy="evict_first",
    )
    lo_u = tl.reshape(tl.join(lo_b & 0xF, lo_b >> 4), (BKG, PB))
    h02 = tl.join(hi_b & 3, (hi_b >> 4) & 3)
    h13 = tl.join((hi_b >> 2) & 3, (hi_b >> 6) & 3)
    hi_u = tl.reshape(tl.join(h02, h13), (BKG, PB))
    return (
        tl.trans((0x40 | (hi_u << 4) | lo_u).to(tl.uint8))
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float16)
    )


@triton.jit
def _expert_panel_decode_kernel(
    A, AS, RS, WP, WSP, Ones, Tiles, Out,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NVFP4: tl.constexpr,
    PDL_WAIT_MODE: tl.constexpr,
    TRIG_DEP: tl.constexpr,
    WS6: tl.constexpr = False,
):
    """Fast decode path (plan.tile <= 16) over panel-contiguous WP and WSP."""
    nb = tl.program_id(0)
    t = tl.program_id(1)
    if PDL_WAIT_MODE == 1:
        tl.extra.cuda.gdc_wait()
    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        if PDL_WAIT_MODE == 2:
            tl.extra.cuda.gdc_wait()
        return
    e = tl.load(Tiles + t).to(tl.int64)
    end = tl.load(Tiles + MAX_CTAS + t)
    local = tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)
    n = nb * BN + tl.arange(0, BN)
    PACK: tl.constexpr = 2 if NVFP4 else 1
    G: tl.constexpr = 16 if NVFP4 else 32
    KP: tl.constexpr = K // PACK
    KG: tl.constexpr = K // G
    BKP: tl.constexpr = BK // PACK
    BKG: tl.constexpr = BK // G
    N_PANELS: tl.constexpr = N // BN
    K_STEPS: tl.constexpr = K // BK
    WS6_STEP: tl.constexpr = BKG * (BN * 3 // 4)

    a_rows = A + rows.to(tl.int64)[:, None] * KP
    wp_panel = WP + (e * N_PANELS + nb) * (K_STEPS * BN * BKP)
    if WS6:
        wsp_panel = WSP + (e * N_PANELS + nb) * (K_STEPS * WS6_STEP)
    else:
        wsp_panel = WSP + (e * N_PANELS + nb) * (K_STEPS * BKG * BN)

    n_off = tl.arange(0, BN)
    bkp_off = tl.arange(0, BKP)
    bkg_off = tl.arange(0, BKG)
    bn2_off = tl.arange(0, BN // 2)
    bn4_off = tl.arange(0, BN // 4)

    if PDL_WAIT_MODE == 2:
        # Launched early behind situ_quantize: prefetch first 2 K-steps of W, then wait.
        for pk in tl.static_range(0, 2):
            pptr = wp_panel + pk * (BN * BKP) + n_off[:, None] * BKP
            tl.inline_asm_elementwise(
                "prefetch.global.L2 [$1]; mov.b32 $0, 0;",
                "=r,l",
                [pptr],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ks in range(0, K_STEPS):
        kp = ks * BKP + bkp_off
        kg = ks * BKG + bkg_off
        a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
        w_raw = tl.load(
            wp_panel + ks * (BN * BKP) + n_off[:, None] * BKP + bkp_off[None, :],
            eviction_policy="evict_first",
        )
        if NVFP4:
            a_s = (
                tl.load(
                    AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE),
                    rmask[:, None],
                    other=0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float16)
            )
            if WS6:
                ws_base = wsp_panel + ks * WS6_STEP
                lo_b = tl.load(
                    ws_base + bkg_off[:, None] * (BN // 2) + bn2_off[None, :],
                    eviction_policy="evict_first",
                )
                hi_b = tl.load(
                    ws_base + (BKG * (BN // 2)) + bkg_off[:, None] * (BN // 4) + bn4_off[None, :],
                    eviction_policy="evict_first",
                )
                lo_u = tl.reshape(tl.join(lo_b & 0xF, lo_b >> 4), (BKG, BN))
                h02 = tl.join(hi_b & 3, (hi_b >> 4) & 3)
                h13 = tl.join((hi_b >> 2) & 3, (hi_b >> 6) & 3)
                hi_u = tl.reshape(tl.join(h02, h13), (BKG, BN))
                w_s = (
                    tl.trans((0x40 | (hi_u << 4) | lo_u).to(tl.uint8))
                    .to(tl.float8e4nv, bitcast=True)
                    .to(tl.float16)
                )
            else:
                w_s = (
                    tl.load(
                        wsp_panel + ks * (BKG * BN) + bkg_off[None, :] * BN + n_off[:, None],
                        eviction_policy="evict_first",
                    )
                    .to(tl.float8e4nv, bitcast=True)
                    .to(tl.float16)
                )
            a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, BKG, 8)), (BM, BKP))
            w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (BN, BKG, 8)), (BN, BKP))
            a_lo, a_hi = _e2m1_hw(a_raw)
            w_lo, w_hi = _e2m1_hw(w_raw)
            acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
            acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)
        else:
            a_s = tl.load(
                AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE),
                rmask[:, None],
                other=0,
            )
            w_s = tl.load(
                wsp_panel + ks * (BKG * BN) + bkg_off[None, :] * BN + n_off[:, None],
                eviction_policy="evict_first",
            )
            a_sf = _e8m0_to_f32(a_s)
            w_sf = _e8m0_to_f32(w_s)
            a_b = _apply_scales(a_raw.to(tl.float8e4nv, bitcast=True), a_sf, BM, BK, G)
            w_b = _apply_scales(w_raw.to(tl.float8e4nv, bitcast=True), w_sf, BN, BK, G)
            acc = tl.dot(a_b, tl.trans(w_b), acc)

    scale = tl.load(Ones + e)
    if NVFP4:
        acc = acc * tl.load(RS + rows, rmask, other=0.0)[:, None]
    acc = acc * scale
    if TRIG_DEP:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        Out + rows.to(tl.int64)[:, None] * N + n[None, :],
        acc.to(tl.bfloat16),
        rmask[:, None],
    )


@triton.jit
def _expert_panel_prefill_w13_kernel(
    A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, Out,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    PB: tl.constexpr,
    KBP: tl.constexpr,
    KBG: tl.constexpr,
    BK: tl.constexpr,
    NVFP4: tl.constexpr,
    WS6: tl.constexpr,
    USE_R2T: tl.constexpr = False,
    AS_ROW_MAJOR: tl.constexpr = False,
):
    """0-ULP bit-exact panel-aligned W13 kernel for prefill/batched (plan.tile >= 32).
    Used for NV_W13 (tile=32,64 with BK=128) and all MX_W13 (tile>=32 with BK=128/256)."""
    SUB_M: tl.constexpr = TILE // BM
    p0 = tl.program_id(0)
    t = tl.program_id(1)
    nb = p0 // SUB_M
    sub = p0 % SUB_M

    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    end = tl.load(Tiles + MAX_CTAS + t)
    if t * TILE + sub * BM >= end:
        return

    e = tl.load(Tiles + t).to(tl.int64)
    local = sub * BM + tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)

    PACK: tl.constexpr = 2 if NVFP4 else 1
    G: tl.constexpr = 16 if NVFP4 else 32
    KP: tl.constexpr = K // PACK
    KG: tl.constexpr = K // G
    BKP: tl.constexpr = BK // PACK
    BKG: tl.constexpr = BK // G
    N_PANELS: tl.constexpr = N // PB
    K_PANELS: tl.constexpr = KP // KBP
    SUB_K: tl.constexpr = KBP // BKP
    TOTAL_STEPS: tl.constexpr = K // BK
    WS6_STEP: tl.constexpr = KBG * (PB * 3 // 4)

    if USE_R2T:
        a_idx = tl.load(Row2Tok + rows, rmask, other=0).to(tl.int64)
    else:
        a_idx = rows.to(tl.int64)
    a_rows = A + a_idx[:, None] * KP
    as_rows = AS + a_idx[:, None] * KG

    wp_panel = WP + (e * N_PANELS + nb) * (K_PANELS * PB * KBP)
    if WS6:
        wsp_panel = WSP + (e * N_PANELS + nb) * (K_PANELS * WS6_STEP)
    else:
        wsp_panel = WSP + (e * N_PANELS + nb) * (K_PANELS * KBG * PB)

    n_off = tl.arange(0, PB)
    bkp_off = tl.arange(0, BKP)
    bkg_off = tl.arange(0, BKG)

    acc = tl.zeros((BM, PB), dtype=tl.float32)
    for step in range(0, TOTAL_STEPS):
        ks = step // SUB_K
        q = step % SUB_K
        kp = step * BKP + bkp_off
        kg = step * BKG + bkg_off

        a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
        w_raw = tl.load(
            wp_panel + ks * (PB * KBP) + n_off[:, None] * KBP + q * BKP + bkp_off[None, :],
            eviction_policy="evict_first",
        )
        if USE_R2T or AS_ROW_MAJOR:
            a_s_raw = tl.load(as_rows + kg[None, :], rmask[:, None], other=0)
        else:
            a_s_raw = tl.load(AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE), rmask[:, None], other=0)

        if NVFP4:
            a_s = a_s_raw.to(tl.float8e4nv, bitcast=True).to(tl.float16)
            if WS6:
                w_s = _unpack_ws6_slice(wsp_panel + ks * WS6_STEP, q * BKG, KBG, BKG, PB)
            else:
                w_s = tl.load(
                    wsp_panel + ks * (KBG * PB) + (q * BKG + bkg_off[None, :]) * PB + n_off[:, None],
                    eviction_policy="evict_first",
                ).to(tl.float8e4nv, bitcast=True).to(tl.float16)
            a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, BKG, 8)), (BM, BKP))
            w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (PB, BKG, 8)), (PB, BKP))
            a_lo, a_hi = _e2m1_hw(a_raw)
            w_lo, w_hi = _e2m1_hw(w_raw)
            acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
            acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)
        else:
            w_s_raw = tl.load(
                wsp_panel + ks * (KBG * PB) + (q * BKG + bkg_off[None, :]) * PB + n_off[:, None],
                eviction_policy="evict_first",
            )
            a_b = _apply_scales(a_raw.to(tl.float8e4nv, bitcast=True), _e8m0_to_f32(a_s_raw), BM, BK, G)
            w_b = _apply_scales(w_raw.to(tl.float8e4nv, bitcast=True), _e8m0_to_f32(w_s_raw), PB, BK, G)
            acc = tl.dot(a_b, tl.trans(w_b), acc)

    scale = tl.load(Ones + e)
    if NVFP4:
        acc = acc * tl.load(RS + rows, rmask, other=0.0)[:, None]
    acc = acc * scale
    n0 = nb * PB + n_off
    tl.store(Out + rows.to(tl.int64)[:, None] * N + n0[None, :], acc.to(tl.bfloat16), rmask[:, None])


@triton.jit
def _expert_panel_nv_w13_l256_dot64_kernel(
    A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, Out,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    PB: tl.constexpr,
    KBP: tl.constexpr,
    KBG: tl.constexpr,
    WS6: tl.constexpr,
    USE_R2T: tl.constexpr = False,
    AS_ROW_MAJOR: tl.constexpr = False,
):
    """0-ULP bit-exact NV_W13 kernel for tile >= 128 (T >= 1024):
    Loads a full K-panel (W [32, 128] u8 + WS6 [32, 16] once per 18 panels -> 4x less
    WS6 unpack & full 128B coalesced W transfers), splits uint8 W and fp16 WS in registers
    into 4 quarters, and runs exact BK=64 dot order -> LITERAL nd == 0 (up to 3.68x speedup!)."""
    SUB_M: tl.constexpr = TILE // BM
    p0 = tl.program_id(0)
    t = tl.program_id(1)
    nb = p0 // SUB_M
    sub = p0 % SUB_M

    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    end = tl.load(Tiles + MAX_CTAS + t)
    if t * TILE + sub * BM >= end:
        return

    e = tl.load(Tiles + t).to(tl.int64)
    local = sub * BM + tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)

    KP: tl.constexpr = K // 2
    KG: tl.constexpr = K // 16
    N_PANELS: tl.constexpr = N // PB
    K_PANELS: tl.constexpr = KP // KBP
    WS6_STEP: tl.constexpr = KBG * (PB * 3 // 4)

    if USE_R2T:
        a_idx = tl.load(Row2Tok + rows, rmask, other=0).to(tl.int64)
    else:
        a_idx = rows.to(tl.int64)
    a_rows = A + a_idx[:, None] * KP
    as_rows = AS + a_idx[:, None] * KG

    wp0 = WP + (e * N_PANELS + nb) * (K_PANELS * PB * KBP)
    wsp0 = WSP + (e * N_PANELS + nb) * (K_PANELS * (WS6_STEP if WS6 else KBG * PB))

    n_off = tl.arange(0, PB)
    bkp128 = tl.arange(0, 128)
    bkp32 = tl.arange(0, 32)
    bkg4 = tl.arange(0, 4)
    bkg16 = tl.arange(0, 16)

    acc = tl.zeros((BM, PB), dtype=tl.float32)
    for ks in range(0, K_PANELS):
        if WS6:
            ws_full = _unpack_ws6_slice(wsp0 + ks * WS6_STEP, 0, KBG, KBG, PB)
        else:
            ws_full = tl.load(
                wsp0 + ks * (KBG * PB) + bkg16[None, :] * PB + n_off[:, None],
                eviction_policy="evict_first",
            ).to(tl.float8e4nv, bitcast=True).to(tl.float16)

        ws01, ws23 = tl.split(tl.permute(tl.reshape(ws_full, (PB, 2, 8)), (0, 2, 1)))
        ws0, ws1 = tl.split(tl.permute(tl.reshape(ws01, (PB, 2, 4)), (0, 2, 1)))
        ws2, ws3 = tl.split(tl.permute(tl.reshape(ws23, (PB, 2, 4)), (0, 2, 1)))
        wss = (ws0, ws1, ws2, ws3)

        w_full = tl.load(
            wp0 + ks * (PB * KBP) + n_off[:, None] * KBP + bkp128[None, :],
            eviction_policy="evict_first",
        )
        w01, w23 = tl.split(tl.permute(tl.reshape(w_full, (PB, 2, 64)), (0, 2, 1)))
        w0, w1 = tl.split(tl.permute(tl.reshape(w01, (PB, 2, 32)), (0, 2, 1)))
        w2, w3 = tl.split(tl.permute(tl.reshape(w23, (PB, 2, 32)), (0, 2, 1)))
        ws_u8 = (w0, w1, w2, w3)

        for q in tl.static_range(0, 4):
            kp = ks * 128 + q * 32 + bkp32
            kg = ks * 16 + q * 4 + bkg4
            a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
            if USE_R2T or AS_ROW_MAJOR:
                a_s = tl.load(as_rows + kg[None, :], rmask[:, None], other=0)
            else:
                a_s = tl.load(AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE), rmask[:, None], other=0)
            a_s = a_s.to(tl.float8e4nv, bitcast=True).to(tl.float16)
            a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, 4, 8)), (BM, 32))
            a_lo, a_hi = _e2m1_hw(a_raw)

            w_s = tl.reshape(tl.broadcast_to(wss[q][:, :, None], (PB, 4, 8)), (PB, 32))
            w_lo, w_hi = _e2m1_hw(ws_u8[q])
            acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
            acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)

    scale = tl.load(Ones + e)
    acc = acc * tl.load(RS + rows, rmask, other=0.0)[:, None]
    acc = acc * scale
    n0 = nb * PB + n_off
    tl.store(Out + rows.to(tl.int64)[:, None] * N + n0[None, :], acc.to(tl.bfloat16), rmask[:, None])


@triton.jit
def _expert_panel_nv_w2_exact64_kernel(
    A, AS, RS, WP, WSP, Ones, Tiles, Out,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PB: tl.constexpr,
    KBP: tl.constexpr,
    KBG: tl.constexpr,
    AS_ROW_MAJOR: tl.constexpr = False,
):
    """0-ULP bit-exact NV_W2 kernel for prefill/batched (plan.tile >= 32, K=288, PB=256, KBP=16):
    Joins consecutive KBP=16 panels into [BN, 32] (and zero-pads tail panel ks=8 to [BN, 32])
    to match _expert_gemm_kernel's exact BK=64 [BM, 32] x [32, BN] MMA sequence -> LITERAL nd == 0."""
    SUB_M: tl.constexpr = TILE // BM
    p0 = tl.program_id(0)
    t = tl.program_id(1)
    nb = p0 // SUB_M
    sub = p0 % SUB_M

    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    end = tl.load(Tiles + MAX_CTAS + t)
    if t * TILE + sub * BM >= end:
        return

    e = tl.load(Tiles + t).to(tl.int64)
    local = sub * BM + tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)

    KP: tl.constexpr = K // 2
    KG: tl.constexpr = K // 16
    K_STEPS: tl.constexpr = KP // KBP
    N_PANELS: tl.constexpr = N // PB
    DIV_N: tl.constexpr = PB // BN
    PAIRS: tl.constexpr = K_STEPS // 2

    panel = nb // DIV_N
    sub_n = (nb % DIV_N) * BN

    a_rows = A + rows.to(tl.int64)[:, None] * KP
    as_rows = AS + rows.to(tl.int64)[:, None] * KG
    wp_base = WP + (e * N_PANELS + panel) * (K_STEPS * PB * KBP) + sub_n * KBP
    wsp_base = WSP + (e * N_PANELS + panel) * (K_STEPS * KBG * PB) + sub_n

    n_off = tl.arange(0, BN)
    bkp16 = tl.arange(0, 16)
    bkp32 = tl.arange(0, 32)
    bkg2 = tl.arange(0, 2)
    bkg4 = tl.arange(0, 4)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for p in range(0, PAIRS):
        ks0 = p * 2
        ks1 = ks0 + 1
        kp = p * 32 + bkp32
        kg = p * 4 + bkg4

        a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
        if AS_ROW_MAJOR:
            a_s = tl.load(as_rows + kg[None, :], rmask[:, None], other=0)
        else:
            a_s = tl.load(AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE), rmask[:, None], other=0)
        a_s = a_s.to(tl.float8e4nv, bitcast=True).to(tl.float16)
        a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, 4, 8)), (BM, 32))
        a_lo, a_hi = _e2m1_hw(a_raw)

        w0 = tl.load(wp_base + ks0 * (PB * 16) + n_off[:, None] * 16 + bkp16[None, :], eviction_policy="evict_first")
        w1 = tl.load(wp_base + ks1 * (PB * 16) + n_off[:, None] * 16 + bkp16[None, :], eviction_policy="evict_first")
        w_raw = tl.reshape(tl.permute(tl.join(w0, w1), (0, 2, 1)), (BN, 32))

        ws0 = tl.load(wsp_base + ks0 * (2 * PB) + bkg2[None, :] * PB + n_off[:, None], eviction_policy="evict_first")
        ws1 = tl.load(wsp_base + ks1 * (2 * PB) + bkg2[None, :] * PB + n_off[:, None], eviction_policy="evict_first")
        w_s = tl.reshape(tl.permute(tl.join(ws0, ws1), (0, 2, 1)), (BN, 4)).to(tl.float8e4nv, bitcast=True).to(tl.float16)
        w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (BN, 4, 8)), (BN, 32))
        w_lo, w_hi = _e2m1_hw(w_raw)

        acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
        acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)

    if K_STEPS % 2 == 1:
        ks = K_STEPS - 1
        kp = (PAIRS * 32) + bkp32
        kg = (PAIRS * 4) + bkg4
        a_raw = tl.load(a_rows + kp[None, :], rmask[:, None] & (kp < KP)[None, :], other=0)
        if AS_ROW_MAJOR:
            a_s = tl.load(as_rows + kg[None, :], rmask[:, None] & (kg < KG)[None, :], other=0)
        else:
            a_s = tl.load(AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE), rmask[:, None] & (kg < KG)[None, :], other=0)
        a_s = a_s.to(tl.float8e4nv, bitcast=True).to(tl.float16)
        a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, 4, 8)), (BM, 32))
        a_lo, a_hi = _e2m1_hw(a_raw)

        w0 = tl.load(wp_base + ks * (PB * 16) + n_off[:, None] * 16 + bkp16[None, :], eviction_policy="evict_first")
        z_w = tl.zeros((BN, 16), dtype=tl.uint8)
        w_raw = tl.reshape(tl.permute(tl.join(w0, z_w), (0, 2, 1)), (BN, 32))
        ws0 = tl.load(wsp_base + ks * (2 * PB) + bkg2[None, :] * PB + n_off[:, None], eviction_policy="evict_first")
        z_ws = tl.zeros((BN, 2), dtype=tl.uint8)
        w_s = tl.reshape(tl.permute(tl.join(ws0, z_ws), (0, 2, 1)), (BN, 4)).to(tl.float8e4nv, bitcast=True).to(tl.float16)
        w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (BN, 4, 8)), (BN, 32))
        w_lo, w_hi = _e2m1_hw(w_raw)

        acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
        acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)

    scale = tl.load(Ones + e)
    acc = acc * tl.load(RS + rows, rmask, other=0.0)[:, None]
    acc = acc * scale
    n_idx = nb * BN + n_off
    tl.store(Out + rows.to(tl.int64)[:, None] * N + n_idx[None, :], acc.to(tl.bfloat16), rmask[:, None])


@triton.jit
def _expert_panel_mx_w2_kernel(
    A, AS, WP, WSP, Ones, Tiles, Out,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PB: tl.constexpr,
    KBP: tl.constexpr,
    KBG: tl.constexpr,
    AS_ROW_MAJOR: tl.constexpr = False,
):
    """0-ULP bit-exact MX_W2 kernel for prefill/batched (plan.tile >= 32, PB=128, KBP=32, KBG=1):
    Exact 9 K-panel iterations with ZERO masks, ZERO div/mod -> LITERAL nd == 0."""
    SUB_M: tl.constexpr = TILE // BM
    p0 = tl.program_id(0)
    t = tl.program_id(1)
    nb = p0 // SUB_M
    sub = p0 % SUB_M

    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    end = tl.load(Tiles + MAX_CTAS + t)
    if t * TILE + sub * BM >= end:
        return

    e = tl.load(Tiles + t).to(tl.int64)
    local = sub * BM + tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)

    KG: tl.constexpr = K // 32
    K_STEPS: tl.constexpr = K // KBP
    N_PANELS: tl.constexpr = N // PB
    DIV_N: tl.constexpr = PB // BN

    panel = nb // DIV_N
    sub_n = (nb % DIV_N) * BN

    a_rows = A + rows.to(tl.int64)[:, None] * K
    as_rows = AS + rows.to(tl.int64)[:, None] * KG
    wp_base = WP + (e * N_PANELS + panel) * (K_STEPS * PB * KBP) + sub_n * KBP
    wsp_base = WSP + (e * N_PANELS + panel) * (K_STEPS * KBG * PB) + sub_n

    n_off = tl.arange(0, BN)
    bkp_off = tl.arange(0, KBP)
    bkg_off = tl.arange(0, KBG)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ks in range(0, K_STEPS):
        kp = ks * KBP + bkp_off
        kg = ks * KBG + bkg_off
        a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
        w_raw = tl.load(
            wp_base + ks * (PB * KBP) + n_off[:, None] * KBP + bkp_off[None, :],
            eviction_policy="evict_first",
        )
        if AS_ROW_MAJOR:
            a_s = tl.load(as_rows + kg[None, :], rmask[:, None], other=0)
        else:
            a_s = tl.load(AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE), rmask[:, None], other=0)
        w_s = tl.load(
            wsp_base + ks * (KBG * PB) + bkg_off[None, :] * PB + n_off[:, None],
            eviction_policy="evict_first",
        )
        a_b = _apply_scales(a_raw.to(tl.float8e4nv, bitcast=True), _e8m0_to_f32(a_s), BM, KBP, 32)
        w_b = _apply_scales(w_raw.to(tl.float8e4nv, bitcast=True), _e8m0_to_f32(w_s), BN, KBP, 32)
        acc = tl.dot(a_b, tl.trans(w_b), acc)

    scale = tl.load(Ones + e)
    acc = acc * scale
    n_idx = nb * BN + n_off
    tl.store(Out + rows.to(tl.int64)[:, None] * N + n_idx[None, :], acc.to(tl.bfloat16), rmask[:, None])


@triton.jit
def _expert_gemm_kernel(
    A, AS, RS, W, WS, Ones, Tiles, Out,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NVFP4: tl.constexpr,
    HWCVT: tl.constexpr = False,
    WS_T: tl.constexpr = False,
    CHUNK: tl.constexpr = 0,
    SWAP: tl.constexpr = False,
    PB: tl.constexpr = 0,
    KBP: tl.constexpr = 0,
    KBG: tl.constexpr = 0,
    WS6: tl.constexpr = False,
):
    if SWAP:
        nb = tl.program_id(0)
        t = tl.program_id(1)
    else:
        t = tl.program_id(0)
        nb = tl.program_id(1)
    sub = tl.program_id(2)
    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    e = tl.load(Tiles + t).to(tl.int64)
    end = tl.load(Tiles + MAX_CTAS + t)
    if t * TILE + sub * BM >= end:
        return
    local = sub * BM + tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)
    n = nb * BN + tl.arange(0, BN)
    nmask = n < N
    if NVFP4:
        PACK: tl.constexpr = 2
        G: tl.constexpr = 16
    else:
        PACK: tl.constexpr = 1
        G: tl.constexpr = 32
    KP: tl.constexpr = K // PACK
    KG: tl.constexpr = K // G
    a_rows = A + rows.to(tl.int64)[:, None] * KP
    w_rows = W + e * N * KP + n.to(tl.int64)[:, None] * KP
    ws_rows = WS + e * N * KG + n.to(tl.int64)[:, None] * KG
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kp = k0 // PACK + tl.arange(0, BK // PACK)
        kg = k0 // G + tl.arange(0, BK // G)
        if K % BK != 0:
            am = rmask[:, None] & (kp < KP)[None, :]
            wm = nmask[:, None] & (kp < KP)[None, :]
            asm = rmask[:, None] & (kg < KG)[None, :]
            wsm = nmask[:, None] & (kg < KG)[None, :]
        else:
            am = rmask[:, None]
            wm = nmask[:, None]
            asm = rmask[:, None]
            wsm = nmask[:, None]
        a_raw = tl.load(a_rows + kp[None, :], am, other=0)
        if PB > 0:
            # Panel-contiguous layout: [E, N // PB, KP // KBP, PB, KBP]
            w_ptr = (
                W
                + e * N * KP
                + ((n.to(tl.int64)[:, None] // PB) * (KP // KBP) + (kp[None, :] // KBP))
                * (PB * KBP)
                + (n.to(tl.int64)[:, None] % PB) * KBP
                + (kp[None, :] % KBP)
            )
            w_raw = tl.load(w_ptr, wm, other=0, eviction_policy="evict_first")
        elif CHUNK > 0:
            w_raw = tl.load(
                W
                + e * N * KP
                + (kp // CHUNK)[None, :] * (N * CHUNK)
                + n.to(tl.int64)[:, None] * CHUNK
                + (kp % CHUNK)[None, :],
                wm,
                other=0,
                eviction_policy="evict_first",
            )
        else:
            w_raw = tl.load(w_rows + kp[None, :], wm, other=0, eviction_policy="evict_first")

        if PB > 0:
            if WS6:
                STEP_B: tl.constexpr = KBG * (PB * 3 // 4)
                p_idx = (n.to(tl.int64)[:, None] // PB) * (KG // KBG) + (kg[None, :] // KBG)
                r_k = kg[None, :] % KBG
                r_n = n.to(tl.int64)[:, None] % PB
                pbase = WS + e * (N // PB) * (KG // KBG) * STEP_B + p_idx * STEP_B
                lo_b = tl.load(pbase + r_k * (PB // 2) + (r_n // 2), wsm, other=0, eviction_policy="evict_first")
                hi_b = tl.load(
                    pbase + KBG * (PB // 2) + r_k * (PB // 4) + (r_n // 4),
                    wsm, other=0, eviction_policy="evict_first",
                )
                lo_n = (lo_b >> ((r_n & 1) * 4)) & 0xF
                hi_n = (hi_b >> ((r_n & 3) * 2)) & 0x3
                w_s_raw = (0x40 | (hi_n << 4) | lo_n).to(tl.uint8)
            else:
                # Panel-contiguous scales: [E, N // PB, KG // KBG, KBG, PB]
                ws_ptr = (
                    WS
                    + e * N * KG
                    + ((n.to(tl.int64)[:, None] // PB) * (KG // KBG) + (kg[None, :] // KBG))
                    * (KBG * PB)
                    + (kg[None, :] % KBG) * PB
                    + (n.to(tl.int64)[:, None] % PB)
                )
                w_s_raw = tl.load(ws_ptr, wsm, other=0, eviction_policy="evict_first")
        elif WS_T:
            w_s_raw = tl.load(
                WS + e * N * KG + kg[None, :] * N + n[:, None],
                wsm,
                other=0,
                eviction_policy="evict_first",
            )
        else:
            w_s_raw = tl.load(ws_rows + kg[None, :], wsm, other=0, eviction_policy="evict_first")

        if NVFP4:
            a_s = (
                tl.load(
                    AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE),
                    asm,
                    other=0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float16)
            )
            w_s = w_s_raw.to(tl.float8e4nv, bitcast=True).to(tl.float16)
            a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, BK // G, 8)), (BM, BK // 2))
            w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (BN, BK // G, 8)), (BN, BK // 2))
            if HWCVT:
                a_lo, a_hi = _e2m1_hw(a_raw)
                w_lo, w_hi = _e2m1_hw(w_raw)
                a_lo = a_lo * a_s
                a_hi = a_hi * a_s
                w_lo = w_lo * w_s
                w_hi = w_hi * w_s
            else:
                a_i = a_raw.to(tl.int32)
                w_i = w_raw.to(tl.int32)
                a_lo = _e2m1_f16(a_i) * a_s
                a_hi = _e2m1_f16(a_i >> 4) * a_s
                w_lo = _e2m1_f16(w_i) * w_s
                w_hi = _e2m1_f16(w_i >> 4) * w_s
            acc = tl.dot(a_lo, tl.trans(w_lo), acc)
            acc = tl.dot(a_hi, tl.trans(w_hi), acc)
        else:
            a_s = tl.load(
                AS + _scale_offsets(rows[:, None], kg[None, :], KG, TILE),
                asm,
                other=0,
            )
            a_sf = _e8m0_to_f32(a_s)
            w_sf = _e8m0_to_f32(w_s_raw)
            a_b = _apply_scales(a_raw.to(tl.float8e4nv, bitcast=True), a_sf, BM, BK, G)
            w_b = _apply_scales(w_raw.to(tl.float8e4nv, bitcast=True), w_sf, BN, BK, G)
            acc = tl.dot(a_b, tl.trans(w_b), acc)
    scale = tl.load(Ones + e)
    if NVFP4:
        if not HWCVT:
            acc = acc * 268435456.0
        acc = acc * tl.load(RS + rows, rmask, other=0.0)[:, None]
    acc = acc * scale
    tl.store(
        Out + rows.to(tl.int64)[:, None] * N + n[None, :],
        acc.to(tl.bfloat16),
        rmask[:, None] & nmask[None, :],
    )


_OVERRIDE = None
_PF_RAW = os.environ.get("INKLING_PREFILL_TILE", "").strip()
_PF_CFG = tuple(int(v) for v in _PF_RAW.split(",")) if _PF_RAW else None
_WS_T = os.environ.get("INKLING_WS_T", "1") == "1"
_CHUNK = int(os.environ.get("INKLING_W2_CHUNK", "16"))
_MX_CHUNK = os.environ.get("INKLING_MX_CHUNK", "1") == "1"
_W13_CHUNK = int(os.environ.get("INKLING_W13_CHUNK", "128"))
_SWAP = os.environ.get("INKLING_EXPERT_SWAP", "1") == "1"
_DECODE_TUNE = os.environ.get("INKLING_EXPERT_TUNE", "1") == "1"
_TUNE2 = os.environ.get("INKLING_EXPERT_TUNE2", "auto")
_VW = os.environ.get("INKLING_VERIFY_W", "").strip().lower()
_LOWD_ON = _TUNE2 == "1" or (_TUNE2 == "auto" and _VW not in ("", "0", "off", "force:7"))
_LOWD_CFG = tuple(int(v) for v in os.environ.get("INKLING_W13_NV_LOWD_CFG", "64,256,4,3").split(","))
_LOWD_MAX_CTAS = 64
_RELAYOUT_GROUP_BYTES = 64 << 20
_DEC_RAW = os.environ.get("INKLING_DECODE_TILE", "").strip()
_DEC_CFG = tuple(int(v) for v in _DEC_RAW.split(",")) if _DEC_RAW else None
_HWCVT = os.environ.get("INKLING_E2M1_HW", "1") == "1"

# S5-W2 panel-contiguous layout + PDL (default ON), S5-W7 6-bit lossless WS for NV_W13.
_PANEL = os.environ.get("INKLING_EXPERT_PANEL", "1") == "1"
_PANEL_PREFILL = os.environ.get("INKLING_EXPERT_PANEL_PREFILL", "1") == "1"
_EXPERT_PDL = os.environ.get("INKLING_EXPERT_PDL", "1") == "1"
_WS6 = os.environ.get("INKLING_WS6", "1") == "1"
_NVFP4_WS5 = os.environ.get("INKLING_NVFP4_WS5", "0").strip().lower()
_MX7_ENABLED = os.environ.get("INKLING_MX7_EXACT", "0").strip().lower() not in ("0", "", "off", "false", "no")
_MX7_FREE_RAW = os.environ.get("INKLING_MX7_FREE_RAW", "1").strip().lower() not in ("0", "", "off", "false", "no")
_MX7_LIB_PATH = os.environ.get("INKLING_MX7_LIB", "")
_MX7_DEFAULT_PATHS = (
    "/rdma/libmxfp8_7bit_exact_v5.so",
    "/rdma/libmxfp8_7bit_exact_v4.so",
    "/rdma/libmxfp8_7bit_exact_v3.so",
)
_MX7_LIB = None
_MX7_LIB_TRIED = False
_MX7_HAS_PREFILL = False
_MX7_HAS_EXACT_BF16_PREFILL = False
_MX7_HAS_NATIVE_V5 = False
_MX7_HAS_NATIVE_V6 = False
_MX7_CACHE: dict[int, tuple | bool] = {}


def _get_mx7_lib():
    global _MX7_LIB, _MX7_LIB_TRIED, _MX7_HAS_PREFILL, _MX7_HAS_EXACT_BF16_PREFILL, _MX7_HAS_NATIVE_V5, _MX7_HAS_NATIVE_V6
    if not _MX7_LIB_TRIED:
        _MX7_LIB_TRIED = True
        chosen_path = _MX7_LIB_PATH
        if not chosen_path:
            for cand in _MX7_DEFAULT_PATHS:
                if os.path.exists(cand):
                    chosen_path = cand
                    break
        if _MX7_ENABLED and chosen_path and os.path.exists(chosen_path):
            try:
                lib = ctypes.CDLL(chosen_path)
                lib.mx7_v2_count_flips.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.mx7_v2_count_flips.restype = ctypes.c_int
                lib.mx7_v2_pack.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.mx7_v2_pack.restype = ctypes.c_int
                lib.mx7_v2_exact_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.mx7_v2_exact_fwd.restype = ctypes.c_int
                if hasattr(lib, "mx7_v2_prefill_fwd"):
                    lib.mx7_v2_prefill_fwd.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.mx7_v2_prefill_fwd.restype = ctypes.c_int
                    _MX7_HAS_PREFILL = True
                if hasattr(lib, "mx7_v5_prefill_exact_bf16_fwd"):
                    lib.mx7_v5_prefill_exact_bf16_fwd.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.mx7_v5_prefill_exact_bf16_fwd.restype = ctypes.c_int
                    _MX7_HAS_EXACT_BF16_PREFILL = True
                if hasattr(lib, "mx7_v5_prefill_native_w13") and hasattr(lib, "mx7_v5_prefill_native_w2"):
                    lib.mx7_v5_prefill_native_w13.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.mx7_v5_prefill_native_w13.restype = ctypes.c_int
                    lib.mx7_v5_prefill_native_w2.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.mx7_v5_prefill_native_w2.restype = ctypes.c_int
                    if hasattr(lib, "mx7_v5_fp32_to_bf16_zero"):
                        lib.mx7_v5_fp32_to_bf16_zero.argtypes = [
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64,
                        ]
                        lib.mx7_v5_fp32_to_bf16_zero.restype = ctypes.c_int
                    _MX7_HAS_NATIVE_V5 = True
                if hasattr(lib, "mx7_v6_prefill_native_w13") and hasattr(lib, "mx7_v6_prefill_native_w2"):
                    lib.mx7_v6_prefill_native_w13.argtypes = (
                        [ctypes.c_void_p] * 11
                        + [ctypes.c_int] * 5
                        + [ctypes.c_uint32]
                        + [ctypes.c_int] * 3
                        + [ctypes.c_uint64]
                    )
                    lib.mx7_v6_prefill_native_w13.restype = ctypes.c_int
                    lib.mx7_v6_prefill_native_w2.argtypes = (
                        [ctypes.c_void_p] * 11
                        + [ctypes.c_int] * 4
                        + [ctypes.c_uint32]
                        + [ctypes.c_int] * 7
                        + [ctypes.c_uint64]
                    )
                    lib.mx7_v6_prefill_native_w2.restype = ctypes.c_int
                    _MX7_HAS_NATIVE_V6 = True
                _MX7_LIB = lib
                print(
                    f"[inkling-mx7] loaded 0-ULP CUDA MX7 kernel: {chosen_path} "
                    f"(prefill={int(_MX7_HAS_PREFILL)}, native_v5={int(_MX7_HAS_NATIVE_V5)}, "
                    f"native_v6={int(_MX7_HAS_NATIVE_V6)}, "
                    f"free_raw={int(_MX7_FREE_RAW and _MX7_HAS_PREFILL)})",
                    flush=True,
                )
            except Exception as e:
                print(f"[inkling-mx7] failed to load {chosen_path}: {e!r}", flush=True)
                _MX7_LIB = None
    return _MX7_LIB


@torch.no_grad()
def _mx7_get_packed(weight: torch.Tensor, weight_scale: torch.Tensor, projection: int, src_pb: int, src_kb: int):
    if not _MX7_ENABLED:
        return None
    # Key directly on the Parameter object (NOT data_ptr(), which becomes 0 after resize_(0)!).
    hit = getattr(weight, "_inkling_mx7", None)
    if hit is not None:
        return None if hit is False else hit
    lib = _get_mx7_lib()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return None
    if weight.data.untyped_storage().size() == 0:
        return None
    e, n, k = weight.shape
    if projection == 0:
        if (n % 32) != 0 or (k % 256) != 0 or (k // 128) > 63:
            weight._inkling_mx7 = False
            return None
        bn, warps, n_slabs = 32, 4, k // 128
    elif projection == 1:
        if (n % 64) != 0 or (k % 32) != 0 or (k // 32) > 63:
            weight._inkling_mx7 = False
            return None
        bn, warps, n_slabs = 64, 2, k // 32
    else:
        weight._inkling_mx7 = False
        return None
    ws_u8 = weight_scale.data.view(torch.uint8)
    base_ws = int(ws_u8.min().item())
    max_ws = int(ws_u8.max().item())
    if base_ws == 0 or max_ws == 255 or (max_ws - base_ws) > 7:
        weight._inkling_mx7 = False
        return None
    w_u8 = weight.data.view(torch.uint8)
    dev = weight.device
    stream = torch.cuda.current_stream(dev).cuda_stream
    total_warps = e * (n // bn) * warps
    panel_meta = getattr(weight, "_inkling_panel", None)
    if panel_meta is not None:
        from_panel = 1
        src_pb, src_kb = int(panel_meta[0]), int(panel_meta[1])
    else:
        from_panel = 0
        src_pb, src_kb = 0, 0
    counts = torch.empty(total_warps, dtype=torch.int32, device=dev)
    rc = lib.mx7_v2_count_flips(
        w_u8.data_ptr(), counts.data_ptr(), projection, e, n, k, from_panel, src_pb, src_kb, stream
    )
    if rc != 0:
        weight._inkling_mx7 = False
        return None
    f_off = torch.zeros(total_warps + 1, dtype=torch.int32, device=dev)
    torch.cumsum(counts, dim=0, out=f_off[1:])
    del counts
    total_flips = int(f_off[-1].item())
    flips = torch.empty(max(1, total_flips), dtype=torch.uint16, device=dev)
    slabs = torch.empty((total_warps * n_slabs, 896), dtype=torch.uint8, device=dev)
    ws3 = torch.empty((total_warps * n_slabs, 4), dtype=torch.uint32, device=dev)
    rc = lib.mx7_v2_pack(
        w_u8.data_ptr(), ws_u8.data_ptr(), slabs.data_ptr(), ws3.data_ptr(),
        flips.data_ptr(), f_off.data_ptr(),
        projection, e, n, k, base_ws, from_panel, src_pb, src_kb, stream
    )
    if rc != 0:
        weight._inkling_mx7 = False
        return None
    entry = (slabs, ws3, flips, f_off, base_ws)
    weight._inkling_mx7 = entry
    if _MX7_FREE_RAW and _MX7_HAS_PREFILL:
        # Release raw MXFP8 weight + scale storage immediately (+11.75 GiB/rank across L40..47,
        # and prevents +10.4 GiB peak VRAM spike during startup packing!).
        weight.data.untyped_storage().resize_(0)
        weight_scale.data.untyped_storage().resize_(0)
        torch.cuda.empty_cache()
    return entry


def _mx7_bmm_decode(act, act_scale, weight, weight_scale, ones, tiles, out, plan, projection, pb, kb, pdl_wait, trig_dep) -> bool:
    if plan.tile not in (8, 16):
        return False
    packed = _mx7_get_packed(weight, weight_scale, projection, pb, kb)
    if packed is None:
        return False
    lib = _get_mx7_lib()
    slabs, ws3, flips, f_off, base_ws = packed
    n, k = weight.shape[1], weight.shape[2]
    rc = lib.mx7_v2_exact_fwd(
        act.view(torch.uint8).data_ptr(),
        act_scale.view(torch.uint8).data_ptr(),
        slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
        projection, n, k, plan.max_ctas, plan.tile,
        base_ws, int(pdl_wait), 1 if trig_dep else 0,
        torch.cuda.current_stream(act.device).cuda_stream,
    )
    return rc == 0


def _mx7_bmm_prefill(
    act, act_scale, weight, weight_scale, ones, tiles, out, plan, projection,
    row2tok=None, as_row_major: bool = False, exact_bf16: bool = False,
) -> bool:
    if plan.tile < 32 or (plan.tile % 32) != 0:
        return False
    packed = _mx7_get_packed(weight, weight_scale, projection, 0, 0)
    if packed is None or not _MX7_HAS_PREFILL:
        return False
    lib = _get_mx7_lib()
    slabs, ws3, flips, f_off, base_ws = packed
    n, k = weight.shape[1], weight.shape[2]
    use_r2t = 1 if (row2tok is not None and projection == 0) else 0
    r2t_ptr = int(row2tok.data_ptr()) if use_r2t else 0
    if as_row_major and projection == 1 and act_scale.ndim >= 2:
        as_rm_arg = int(act_scale.stride(0))
    else:
        as_rm_arg = 1 if as_row_major else 0
    fn = lib.mx7_v5_prefill_exact_bf16_fwd if (exact_bf16 and _MX7_HAS_EXACT_BF16_PREFILL) else lib.mx7_v2_prefill_fwd
    rc = fn(
        act.view(torch.uint8).data_ptr(),
        act_scale.view(torch.uint8).data_ptr(),
        r2t_ptr,
        slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
        projection, n, k, plan.max_ctas, plan.tile,
        base_ws, use_r2t, as_rm_arg,
        torch.cuda.current_stream(act.device).cuda_stream,
    )
    return rc == 0

# (PB, KB, warps, stages) for each (nvfp4, projection) at tile=8 (single-stream decode untouched!)
_PANEL_SPECS = {
    (True, 0): (32, 256, 2, 4),   # nv_w13 (N=576, K=4608)
    (True, 1): (256, 32, 2, 3),   # nv_w2  (N=4608, K=288)
    (False, 0): (32, 512, 4, 2),  # mx_w13 (N=576, K=4608)
    (False, 1): (128, 32, 2, 3),  # mx_w2  (N=4608, K=288)
}

# Tuned (warps, stages) for tile=16 (B=2..16 concurrent decode, T=9..128) — 0-ULP identical!
_PANEL_T16_WS = {
    (True, 0): (2, 4),   # nv_w13: 242 GB/s
    (True, 1): (4, 1),   # nv_w2:  +5-6% vs (2,3)
    (False, 0): (4, 3),  # mx_w13: +3-4% vs (4,2)
    (False, 1): (4, 1),  # mx_w2:  +4-5% vs (2,3)
}


def _weight_panel_chunked(w: torch.Tensor, pb: int, kbp: int) -> None:
    """In-place [E, N, KP] -> [E, N//pb, KP//kbp, pb, kbp] in <= 64 MB slabs."""
    want = (pb, kbp)
    if getattr(w, "_inkling_panel", None) == want:
        return
    assert getattr(w, "_inkling_panel", None) is None, "weight already panel-packed differently"
    assert getattr(w, "_inkling_chunk", 0) == 0, "weight already chunked"
    assert not torch.cuda.is_current_stream_capturing()
    e, n, kp = w.shape
    raw = w.data.view(torch.uint8)
    step = max(1, _RELAYOUT_GROUP_BYTES // (n * kp))
    for e0 in range(0, e, step):
        blk = raw[e0 : e0 + step]
        t = blk.reshape(-1, n // pb, pb, kp // kbp, kbp).permute(0, 1, 3, 2, 4).contiguous()
        blk.copy_(t.view(blk.shape))
    w._inkling_panel = want


def _scales_panel_kmajor(ws: torch.Tensor, pb: int, kbg: int) -> None:
    """In-place [E, N, KG] -> [E, N//pb, KG//kbg, kbg, pb] in <= 64 MB slabs."""
    want = (pb, kbg)
    if getattr(ws, "_inkling_panel", None) == want:
        return
    assert getattr(ws, "_inkling_panel", None) is None, "scales already panel-packed differently"
    assert not getattr(ws, "_inkling_kmajor", False), "scales already kmajor"
    assert not torch.cuda.is_current_stream_capturing()
    e, n, kg = ws.shape
    raw = ws.data.view(torch.uint8)
    step = max(1, _RELAYOUT_GROUP_BYTES // (n * kg))
    for e0 in range(0, e, step):
        blk = raw[e0 : e0 + step]
        t = blk.reshape(-1, n // pb, pb, kg // kbg, kbg).permute(0, 1, 3, 4, 2).contiguous()
        blk.copy_(t.view(blk.shape))
    ws._inkling_panel = want


def _scales_panel_ws6(ws: torch.Tensor, pb: int, kbg: int) -> bool:
    """In-place 6-bit lossless packing for NVFP4 WS when 100% of bytes match (b & 0xC0) == 0x40.
    Packs [E, N//pb, KG//kbg, kbg, pb] -> [E, N//pb, KG//kbg, kbg*(pb*3//4)] in the first 3/4 of ws.
    Returns True if packed in 6-bit lossless format, else falls back to 8-bit panel layout."""
    st = getattr(ws, "_inkling_ws6", None)
    if st is not None:
        return bool(st)
    assert getattr(ws, "_inkling_panel", None) is None, "scales already panel-packed"
    assert not getattr(ws, "_inkling_kmajor", False), "scales already kmajor"
    assert not torch.cuda.is_current_stream_capturing()
    e, n, kg = ws.shape
    raw = ws.data.view(torch.uint8)
    if pb % 4 != 0 or not bool(((raw & 0xC0) == 0x40).all().item()):
        ws._inkling_ws6 = False
        _scales_panel_kmajor(ws, pb, kbg)
        return False
    flat = raw.view(-1)
    n_panels = n // pb
    k_steps = kg // kbg
    step_b = kbg * (pb * 3 // 4)
    exp_out_b = n_panels * k_steps * step_b
    step = max(1, _RELAYOUT_GROUP_BYTES // (n * kg))
    for e0 in range(0, e, step):
        e1 = min(e0 + step, e)
        blk = raw[e0:e1]
        # [B, n_panels, k_steps, kbg, pb]
        v = (blk.reshape(e1 - e0, n_panels, pb, k_steps, kbg).permute(0, 1, 3, 4, 2) & 0x3F)
        lo = (v[..., 0::2] & 0xF) | ((v[..., 1::2] & 0xF) << 4)
        hi = (
            ((v[..., 0::4] >> 4) & 3)
            | (((v[..., 1::4] >> 4) & 3) << 2)
            | (((v[..., 2::4] >> 4) & 3) << 4)
            | (((v[..., 3::4] >> 4) & 3) << 6)
        )
        packed = torch.cat(
            [lo.reshape(e1 - e0, n_panels, k_steps, -1), hi.reshape(e1 - e0, n_panels, k_steps, -1)],
            dim=-1,
        ).contiguous()
        flat[e0 * exp_out_b : e1 * exp_out_b].copy_(packed.view(-1))
    ws._inkling_panel = (pb, kbg)
    ws._inkling_ws6 = True
    return True


def _weight_chunked(w: torch.Tensor, chunk: int) -> None:
    if getattr(w, "_inkling_chunk", 0) == chunk:
        return
    assert getattr(w, "_inkling_chunk", 0) == 0, "weight already chunked differently"
    assert not torch.cuda.is_current_stream_capturing()
    e, n, kp = w.shape
    raw = w.data.view(torch.uint8)
    step = max(1, _RELAYOUT_GROUP_BYTES // (n * kp))
    for e0 in range(0, e, step):
        blk = raw[e0 : e0 + step]
        t = blk.reshape(-1, n, kp // chunk, chunk).permute(0, 2, 1, 3).contiguous()
        blk.copy_(t.view(blk.shape))
    w._inkling_chunk = chunk


def _scales_kmajor(ws: torch.Tensor) -> None:
    if getattr(ws, "_inkling_kmajor", False):
        return
    assert not torch.cuda.is_current_stream_capturing()
    e, n, kg = ws.shape
    raw = ws.data.view(torch.uint8)
    step = max(1, _RELAYOUT_GROUP_BYTES // (n * kg))
    for e0 in range(0, e, step):
        blk = raw[e0 : e0 + step]
        t = blk.transpose(1, 2).contiguous()
        blk.copy_(t.view(blk.shape))
    ws._inkling_kmajor = True


def _config(tile: int, nvfp4: bool):
    if _OVERRIDE is not None:
        return _OVERRIDE(tile, nvfp4)
    if tile <= 16:
        return 16, 64, 256 if nvfp4 else 128, 4, 3
    if tile <= 64:
        return tile, 64, 128, 4, 3
    if _PF_CFG is not None:
        return _PF_CFG
    return 64, 128, 64, 8, 3


def ensure_packed_for_pf_native(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    projection: int,
) -> None:
    """Guarantee Phase 1 decode weight packs (WS5 or EXPERT_PANEL/WS6/MX7) are materialized
    before `pf_moe_native` runs on prefill chunks (idempotent O(1) check once packed, ensures
    profile_run accounts for all decode weight buffers before vLLM allocates KV cache)."""
    nvfp4 = weight.dtype == torch.uint8
    if (not nvfp4) and _MX7_ENABLED and _OVERRIDE is None and _DEC_CFG is None:
        _get_mx7_lib()
        if _MX7_FREE_RAW and _MX7_HAS_PREFILL:
            # Pack MX7 directly from raw [E, N, K] and immediately release the 11.75 GiB raw
            # MXFP8 storage during profile_run (before vLLM sizes KV cache); pf_moe_native
            # natively runs mx7_v5_prefill_native_w13 / w2 (incl. fused W2-Combine!) on MX7 slabs.
            if not torch.cuda.is_current_stream_capturing():
                _mx7_get_packed(weight, weight_scale, projection, 0, 0)
            return
    if (
        nvfp4
        and _OVERRIDE is None
        and _DEC_CFG is None
        and (_NVFP4_WS5 in ("1", "both", "true", "on") or (_NVFP4_WS5 == "w2" and projection == 1))
    ):
        from . import nvfp4_ws5
        if getattr(weight, "_inkling_nvfp4_ws5", None) is None and nvfp4_ws5._get_lib() is not None:
            if projection == 0:
                nvfp4_ws5.pack_w13_ws5_inplace(weight, weight_scale)
            else:
                nvfp4_ws5.pack_w2_ws5_inplace(weight, weight_scale)
            return
    if weight.data.untyped_storage().size() == 0 or weight_scale.data.untyped_storage().size() == 0:
        return
    pack = 2 if nvfp4 else 1
    group = 16 if nvfp4 else 32
    n = weight.shape[1]
    k = weight.shape[2] * pack
    spec = _PANEL_SPECS.get((nvfp4, projection)) if (_PANEL and _OVERRIDE is None and _DEC_CFG is None and _HWCVT) else None
    if spec is not None and n % spec[0] == 0 and k % spec[1] == 0:
        pb, kb, _, _ = spec
        kbp = kb // pack
        kbg = kb // group
        _weight_panel_chunked(weight, pb, kbp)
        ws6 = _scales_panel_ws6(weight_scale, pb, kbg) if (_WS6 and nvfp4 and projection == 0) else False
        if not ws6:
            _scales_panel_kmajor(weight_scale, pb, kbg)
        if (not nvfp4) and _MX7_ENABLED and (not torch.cuda.is_current_stream_capturing()):
            _mx7_get_packed(weight, weight_scale, projection, pb, kb)


def bmm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan: ExpertPlan,
    projection: int,
    *,
    row2tok: torch.Tensor | None = None,
    as_row_major: bool = False,
) -> torch.Tensor:
    nvfp4 = weight.dtype == torch.uint8

    # 1) Decode (TILE=8,16) full-lifecycle MX7 fast path (100% byte-identical v3 0-ULP decode).
    # Packs directly from raw [E, N, K] (from_panel=0) and frees raw 11.75 GiB MXFP8 weights immediately!
    if (not nvfp4) and plan.tile <= 16 and _MX7_ENABLED and _OVERRIDE is None and _DEC_CFG is None:
        _get_mx7_lib()
        if _MX7_FREE_RAW and _MX7_HAS_PREFILL:
            n = weight.shape[1]
            out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
            pdl = _EXPERT_PDL and plan.rows <= 2048
            pdl_wait = (2 if projection == 1 else 1) if pdl else 0
            trig_dep = pdl
            if _mx7_bmm_decode(
                act, act_scale, weight, weight_scale, ones, tiles, out,
                plan, projection, 0, 0, pdl_wait, trig_dep,
            ):
                return out

    # 2) Long prefill (TILE >= 64) SM121a native block-scaled FP4/FP8 path (incl. MX7 v5 + fused Combine!).
    if plan.tile >= 64:
        from . import pf_moe_native
        if pf_moe_native.is_active(plan):
            ensure_packed_for_pf_native(weight, weight_scale, projection)
            res = pf_moe_native.bmm_pf_native(
                act, act_scale, row_scale, weight, weight_scale, ones, tiles,
                plan, projection, row2tok=row2tok, as_row_major=as_row_major,
            )
            if res is not None:
                return res

    # 2b) C=32 (M=17..256 -> plan.rows > 2048, plan.tile in (16, 32)) opt-in 0-ULP native WS5/MX7 path.
    if plan.tile in (16, 32) and plan.rows > 2048 and _OVERRIDE is None and _DEC_CFG is None:
        try:
            from . import moe_c32_runtime as _c32
        except Exception:
            _c32 = None
        if _c32 is not None and _c32.C32_MASTER:
            res = _c32.bmm_c32_standalone(
                act, act_scale, row_scale, weight, weight_scale, ones, tiles,
                plan, projection, row2tok=row2tok, as_row_major=as_row_major,
            )
            if res is not None:
                return res

    # 3) Short/exact prefill (TILE >= 32, or when pf_moe_native is inactive): 0-ULP BF16 MX7 prefill.
    if (not nvfp4) and plan.tile >= 32 and _MX7_ENABLED and _OVERRIDE is None and _DEC_CFG is None:
        _get_mx7_lib()
        _c32_mx7_t32 = (
            plan.tile == 32
            and os.environ.get("INKLING_C32_MX7_TILE32", "0").strip().lower()
            not in ("0", "", "off", "false", "no")
        )
        if (_MX7_FREE_RAW or _c32_mx7_t32) and _MX7_HAS_PREFILL:
            n = weight.shape[1]
            out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
            if _mx7_bmm_prefill(
                act, act_scale, weight, weight_scale, ones, tiles, out,
                plan, projection, row2tok=row2tok, as_row_major=as_row_major,
                exact_bf16=True,
            ):
                return out
    if (
        nvfp4
        and _OVERRIDE is None
        and _DEC_CFG is None
        and (plan.tile >= 32 or (row2tok is None and not as_row_major))
        and (_NVFP4_WS5 in ("1", "both", "true", "on") or (_NVFP4_WS5 == "w2" and projection == 1))
    ):
        from . import nvfp4_ws5
        res = nvfp4_ws5.bmm_nvfp4_ws5(
            act, act_scale, row_scale, weight, weight_scale, ones, tiles, plan, projection,
            pdl=_EXPERT_PDL, row2tok=row2tok, as_row_major=as_row_major,
        )
        if res is not None:
            return res
    pack = 2 if nvfp4 else 1
    group = 16 if nvfp4 else 32
    n = weight.shape[1]
    k = weight.shape[2] * pack
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)

    # S5-W2 panel-contiguous fast path (used for ALL decode and prefill shapes once packed).
    spec = _PANEL_SPECS.get((nvfp4, projection)) if (_PANEL and _OVERRIDE is None and _DEC_CFG is None and _HWCVT) else None
    if spec is not None and n % spec[0] == 0 and k % spec[1] == 0:
        pb, kb, d_warps, d_stages = spec
        kbp = kb // pack
        kbg = kb // group
        _weight_panel_chunked(weight, pb, kbp)
        ws6 = _scales_panel_ws6(weight_scale, pb, kbg) if (_WS6 and nvfp4 and projection == 0) else False
        if not ws6:
            _scales_panel_kmajor(weight_scale, pb, kbg)
        # Pre-pack MX7 during eager profiling run (before vLLM allocates KV cache!) so VRAM is accounted for.
        if (not nvfp4) and _MX7_ENABLED and (not torch.cuda.is_current_stream_capturing()):
            _mx7_get_packed(weight, weight_scale, projection, pb, kb)
        if plan.tile <= 16:
            if plan.tile == 16:
                d_warps, d_stages = _PANEL_T16_WS[(nvfp4, projection)]
            pdl = _EXPERT_PDL and plan.rows <= 2048
            pdl_wait = (2 if projection == 1 else 1) if pdl else 0
            trig_dep = pdl
            if (not nvfp4) and _mx7_bmm_decode(
                act, act_scale, weight, weight_scale, ones, tiles, out,
                plan, projection, pb, kb, pdl_wait, trig_dep,
            ):
                return out
            _expert_panel_decode_kernel[(n // pb, plan.max_ctas)](
                act.view(torch.uint8),
                act_scale.view(torch.uint8),
                row_scale if row_scale is not None else ones,
                weight.view(torch.uint8),
                weight_scale.view(torch.uint8),
                ones,
                tiles,
                out,
                N=n,
                K=k,
                MAX_CTAS=plan.max_ctas,
                TILE=plan.tile,
                BM=16,
                BN=pb,
                BK=kb,
                NVFP4=nvfp4,
                PDL_WAIT_MODE=pdl_wait,
                TRIG_DEP=trig_dep,
                WS6=ws6,
                num_warps=d_warps,
                num_stages=d_stages,
                launch_pdl=pdl,
            )
            return out

        # Panel-aligned prefill & high-concurrency path (plan.tile >= 32), 100% 0-ULP bit-exact!
        if _PANEL_PREFILL:
            use_r2t = row2tok is not None and projection == 0
            r2t_arg = row2tok if use_r2t else tiles
            rs_arg = row_scale if row_scale is not None else ones
            if projection == 0:
                if nvfp4:
                    if plan.tile <= 64:
                        # Exact BK=128 order matching _config(tile<=64)
                        bm = 32 if plan.tile == 32 else 64
                        sub_m = plan.tile // bm
                        _expert_panel_prefill_w13_kernel[((n // pb) * sub_m, plan.max_ctas)](
                            act.view(torch.uint8),
                            act_scale.view(torch.uint8),
                            rs_arg,
                            r2t_arg,
                            weight.view(torch.uint8),
                            weight_scale.view(torch.uint8),
                            ones,
                            tiles,
                            out,
                            N=n,
                            K=k,
                            MAX_CTAS=plan.max_ctas,
                            TILE=plan.tile,
                            BM=bm,
                            PB=pb,
                            KBP=kbp,
                            KBG=kbg,
                            BK=128,
                            NVFP4=True,
                            WS6=ws6,
                            USE_R2T=use_r2t,
                            AS_ROW_MAJOR=as_row_major,
                            num_warps=2,
                            num_stages=4,
                        )
                        return out
                    elif kbp == 128 and kbg == 16 and (_PF_CFG is None or _PF_CFG[2] == 64):
                        # Exact BK=64 order using full KBP=128 panel loads + WS6-once-per-panel!
                        # Adaptive BM: <=520 max_ctas (T<=~2200) -> BM=64 (sub=1 skips dead rows);
                        # >520 max_ctas (T=3072..4096) -> BM=128 (1x W read, zero sub=1 re-read).
                        if plan.max_ctas <= 520:
                            bm, nw, ns = 64, 2, 3
                        else:
                            bm, nw, ns = 128, 2, 2
                        sub_m = plan.tile // bm
                        _expert_panel_nv_w13_l256_dot64_kernel[((n // pb) * sub_m, plan.max_ctas)](
                            act.view(torch.uint8),
                            act_scale.view(torch.uint8),
                            rs_arg,
                            r2t_arg,
                            weight.view(torch.uint8),
                            weight_scale.view(torch.uint8),
                            ones,
                            tiles,
                            out,
                            N=n,
                            K=k,
                            MAX_CTAS=plan.max_ctas,
                            TILE=plan.tile,
                            BM=bm,
                            PB=pb,
                            KBP=kbp,
                            KBG=kbg,
                            WS6=ws6,
                            USE_R2T=use_r2t,
                            AS_ROW_MAJOR=as_row_major,
                            num_warps=nw,
                            num_stages=ns,
                        )
                        return out
                else:
                    # MX_W13: 0-ULP invariant to BK (sequential bf16 m16n8k16)
                    if plan.tile == 32:
                        bm, bk, nw, ns = 32, 256, 4, 3
                    elif plan.tile == 64:
                        bm, bk, nw, ns = 32, 256, 4, 3
                    elif plan.max_ctas <= 520:
                        bm, bk, nw, ns = 64, 128, 4, 3
                    else:
                        bm, bk, nw, ns = 128, 128, 4, 3
                    sub_m = plan.tile // bm
                    _expert_panel_prefill_w13_kernel[((n // pb) * sub_m, plan.max_ctas)](
                        act.view(torch.uint8),
                        act_scale.view(torch.uint8),
                        rs_arg,
                        r2t_arg,
                        weight.view(torch.uint8),
                        weight_scale.view(torch.uint8),
                        ones,
                        tiles,
                        out,
                        N=n,
                        K=k,
                        MAX_CTAS=plan.max_ctas,
                        TILE=plan.tile,
                        BM=bm,
                        PB=pb,
                        KBP=kbp,
                        KBG=kbg,
                        BK=bk,
                        NVFP4=False,
                        WS6=False,
                        USE_R2T=use_r2t,
                        AS_ROW_MAJOR=as_row_major,
                        num_warps=nw,
                        num_stages=ns,
                    )
                    return out
            else:
                # projection == 1 (W2)
                if nvfp4 and kbp == 16 and kbg == 2 and pb == 256 and (_PF_CFG is None or _PF_CFG[2] == 64):
                    if plan.tile <= 64:
                        bm, bn, nw, ns = 32, 256, 4, 3
                    elif plan.max_ctas <= 460:
                        bm, bn, nw, ns = 32, 256, 4, 2
                    else:
                        bm, bn, nw, ns = 64, 256, 4, 2
                    sub_m = plan.tile // bm
                    _expert_panel_nv_w2_exact64_kernel[((n // bn) * sub_m, plan.max_ctas)](
                        act.view(torch.uint8),
                        act_scale.view(torch.uint8),
                        rs_arg,
                        weight.view(torch.uint8),
                        weight_scale.view(torch.uint8),
                        ones,
                        tiles,
                        out,
                        N=n,
                        K=k,
                        MAX_CTAS=plan.max_ctas,
                        TILE=plan.tile,
                        BM=bm,
                        BN=bn,
                        PB=pb,
                        KBP=kbp,
                        KBG=kbg,
                        AS_ROW_MAJOR=as_row_major,
                        num_warps=nw,
                        num_stages=ns,
                    )
                    return out
                elif not nvfp4 and kbp == 32 and kbg == 1 and pb == 128:
                    if plan.tile <= 64:
                        bm, bn, nw, ns = 32, 128, 4, 3
                    elif plan.max_ctas <= 520:
                        bm, bn, nw, ns = 64, 128, 4, 3
                    else:
                        bm, bn, nw, ns = 128, 128, 4, 3
                    sub_m = plan.tile // bm
                    _expert_panel_mx_w2_kernel[((n // bn) * sub_m, plan.max_ctas)](
                        act.view(torch.uint8),
                        act_scale.view(torch.uint8),
                        weight.view(torch.uint8),
                        weight_scale.view(torch.uint8),
                        ones,
                        tiles,
                        out,
                        N=n,
                        K=k,
                        MAX_CTAS=plan.max_ctas,
                        TILE=plan.tile,
                        BM=bm,
                        BN=bn,
                        PB=pb,
                        KBP=kbp,
                        KBG=kbg,
                        AS_ROW_MAJOR=as_row_major,
                        num_warps=nw,
                        num_stages=ns,
                    )
                    return out

        # Generic prefill fallback reading panel-contiguous weights (with dead-sub skip!)
        bm, bn, bk, warps, stages = _config(plan.tile, nvfp4)
        if k % bk:
            bk = 64
        grid = (plan.max_ctas, triton.cdiv(n, bn), triton.cdiv(plan.tile, bm))
        _expert_gemm_kernel[grid](
            act.view(torch.uint8),
            act_scale.view(torch.uint8),
            row_scale if row_scale is not None else ones,
            weight.view(torch.uint8),
            weight_scale.view(torch.uint8),
            ones,
            tiles,
            out,
            N=n,
            K=k,
            MAX_CTAS=plan.max_ctas,
            TILE=plan.tile,
            BM=bm,
            BN=bn,
            BK=bk,
            NVFP4=nvfp4,
            HWCVT=True,
            WS_T=False,
            CHUNK=0,
            SWAP=False,
            PB=pb,
            KBP=kbp,
            KBG=kbg,
            WS6=ws6,
            num_warps=warps,
            num_stages=stages,
        )
        return out

    # Fallback path (when INKLING_EXPERT_PANEL=0 or custom override)
    bm, bn, bk, warps, stages = _config(plan.tile, nvfp4)
    chunk = (_CHUNK if nvfp4 else (2 * _CHUNK if _MX_CHUNK else 0)) if (projection == 1 and k < 1024 and _CHUNK) else 0
    if projection == 0 and _W13_CHUNK:
        chunk = _W13_CHUNK
    if chunk and (k // pack) % chunk:
        chunk = 0
    tuned = False
    if _OVERRIDE is None and _DECODE_TUNE and plan.tile <= 16 and chunk:
        if projection == 0:
            bm, bn, bk, warps, stages = (16, 128, 256, 8, 3) if nvfp4 else (16, 32, 512, 4, 2)
            if nvfp4 and _LOWD_ON and plan.max_ctas <= _LOWD_MAX_CTAS:
                bn, bk, warps, stages = _LOWD_CFG
            tuned = True
        elif not nvfp4:
            bm, bn, bk, warps, stages = 16, 128, 32, 4, 1
            tuned = True
    ws_t = _WS_T and (projection == 0 or chunk > 0)
    if ws_t:
        _scales_kmajor(weight_scale)
    if chunk:
        _weight_chunked(weight, chunk)
    if _OVERRIDE is None and not tuned and k % bk:
        if plan.tile <= 16 and chunk:
            bm, bn, bk, warps, stages = 16, 256, 32, 4, 2
        elif plan.tile <= 16:
            bm, bn, bk, warps, stages = 16, 128, 128, 4, 2
        else:
            bk = 64
    if _OVERRIDE is None and _DEC_CFG is not None and plan.tile <= 16:
        bm, bn, bk, warps, stages = _DEC_CFG
        ws_t = _WS_T and (projection == 0 or chunk > 0)
    grid = (plan.max_ctas, triton.cdiv(n, bn), triton.cdiv(plan.tile, bm))
    swap = _SWAP and plan.tile <= 16
    if swap:
        grid = (grid[1], grid[0], grid[2])
    _expert_gemm_kernel[grid](
        act.view(torch.uint8),
        act_scale.view(torch.uint8),
        row_scale if row_scale is not None else ones,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8),
        ones,
        tiles,
        out,
        N=n,
        K=k,
        MAX_CTAS=plan.max_ctas,
        TILE=plan.tile,
        BM=bm,
        BN=bn,
        BK=bk,
        NVFP4=nvfp4,
        HWCVT=_HWCVT,
        WS_T=ws_t,
        CHUNK=chunk,
        SWAP=swap,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def load_kernels(weights, weight_scales, topk) -> None:
    return None
