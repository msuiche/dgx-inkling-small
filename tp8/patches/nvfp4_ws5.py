# SPDX-License-Identifier: Apache-2.0
"""Ready-to-integrate SM121 (GB10) 0-ULP Lossless WS5 NVFP4 MoE integration for trtllm_bmm.py.

Usage inside ``trtllm_bmm.py``:
  - Controlled by ``INKLING_NVFP4_WS5=1`` (or ``w2``) and ``INKLING_NVFP4_WS5_LIB=/rdma/libnvfp4_ws5_exact.so``
    (or co-located path).
  - Decode (``plan.tile <= 16``, ``nvfp4=True``): dispatches directly to CUDA C++
    ``nvfp4_w13_ws5_fwd`` (N=576, K=4608) and ``nvfp4_w2_ws5_fwd`` (N=4608, K=288).
  - Prefill / high-concurrency (``plan.tile > 16``): reads the EXACT SAME packed tensors via fast
    panel-aligned Triton kernels (``_nvfp4_ws5_prefill_w13_kernel`` and
    ``_nvfp4_ws5_prefill_w2_kernel``) with dead-CTA early exit, L2-friendly CTA ordering,
    128B-aligned contiguous slab/panel loads + register unswizzle, uniform CTA-level patch-skip
    fast path, zero N-padding waste, zero extra GPU memory, and literal 0-ULP bit-exactness
    matching stock ``BK=128`` (``tile<=64`` W13), ``BK=64`` (``tile>=128`` W13), and ``BK=64``
    with zero-padded ``K=288`` tail (all ``tile>=32`` W2).
  - Net GPU memory: -1.43 GB across L0..39 (saves 0.96 GB on W13 + 0.47 GB on W2 via WS5!).
"""

from __future__ import annotations

import ctypes
import os
import torch
from vllm.triton_utils import tl, triton

_LIB = None
_LIB_PATHS = (
    os.environ.get("INKLING_NVFP4_WS5_LIB", ""),
    "/rdma/libnvfp4_ws5_exact.so",
    "/tmp/swarm_nvfp4/libnvfp4_ws5_exact.so",
)


def _get_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    for p in _LIB_PATHS:
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            lib.nvfp4_w13_ws5_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.nvfp4_w13_ws5_fwd.restype = ctypes.c_int
            lib.nvfp4_w2_ws5_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.nvfp4_w2_ws5_fwd.restype = ctypes.c_int
            _LIB = lib
            return _LIB
    return None


def _compute_ws_base(ws_u8: torch.Tensor) -> torch.Tensor:
    E = ws_u8.shape[0]
    flat = ws_u8.reshape(E, -1).to(torch.int64)
    h = torch.zeros((E, 128), dtype=torch.int32, device=ws_u8.device)
    h.scatter_add_(1, flat.clamp(0, 127), torch.ones_like(flat, dtype=torch.int32))
    cs = torch.cat([torch.zeros((E, 1), dtype=torch.int32, device=ws_u8.device), h.cumsum(dim=1)], dim=1)
    return (cs[:, 32:] - cs[:, :-32]).argmax(dim=1).to(torch.uint8)


def pack_w13_ws5_inplace(weight: torch.Tensor, weight_scale: torch.Tensor) -> dict:
    """Pack raw W13 [E, 576, 2304] + WS [E, 576, 288] into single contiguous 8832B super-slabs.
    Reuses slab allocation & releases raw buffers (net -23.9 MB/layer = -0.96 GB across 40L!)."""
    meta = getattr(weight, "_inkling_nvfp4_ws5", None)
    if meta is not None:
        return meta
    assert not torch.cuda.is_current_stream_capturing()
    w_u8 = weight.data.view(torch.uint8)
    ws_u8 = weight_scale.data.view(torch.uint8)
    E, N, KP = w_u8.shape
    KG = ws_u8.shape[2]
    assert N == 576 and KP == 2304 and KG == 288
    n_panels = 18
    dev = w_u8.device

    ws_base = _compute_ws_base(ws_u8)
    slabs = torch.empty((E, n_panels, 9, 8832), dtype=torch.uint8, device=dev)

    # Chunk over experts (<= 64 MB peak scratch) to avoid OOM during startup packing
    step = max(1, (64 << 20) // (N * KP))
    shifts4 = torch.arange(8, device=dev, dtype=torch.int64) * 4
    shifts1 = torch.arange(8, device=dev, dtype=torch.int64)
    out_wuid_list, out_rec_list = [], []

    for e0 in range(0, E, step):
        e1 = min(e0 + step, E)
        B = e1 - e0
        w_blk = w_u8[e0:e1].reshape(B, n_panels, 4, 8, 9, 2, 2, 2, 2, 4, 2, 2)
        w_ss = w_blk.permute(0, 1, 4, 5, 2, 6, 3, 9, 7, 10, 8, 11).contiguous().reshape(B, n_panels, 9, 8192)
        slabs[e0:e1, :, :, :8192].copy_(w_ss)

        ws_r = ws_u8[e0:e1].reshape(B, n_panels, 4, 8, 9, 4, 8).permute(0, 1, 4, 2, 3, 5, 6).contiguous()
        diff = ws_r.to(torch.int32) - ws_base[e0:e1].to(torch.int32)[:, None, None, None, None, None, None]
        in_range = (diff >= 0) & (diff < 32)
        code = torch.where(in_range, diff, torch.zeros_like(diff)).to(torch.int64)
        ws_lo4 = ((code & 0xF) << shifts4).sum(-1).to(torch.uint32).view(torch.uint8).reshape(B, n_panels, 9, 512)
        ws_hi1 = (((code >> 4) & 1) << shifts1).sum(-1).to(torch.uint8).reshape(B, n_panels, 9, 128)
        slabs[e0:e1, :, :, 8192:8704].copy_(ws_lo4)
        slabs[e0:e1, :, :, 8704:8832].copy_(ws_hi1)

        outliers = (~in_range).nonzero(as_tuple=False)
        if outliers.numel() > 0:
            be, nb_i, ss_i, wp_i, g_i, lt_i, ii = outliers.unbind(dim=1)
            wuid = ((be + e0) * n_panels + nb_i) * 4 + wp_i
            kg_i = ss_i * 32 + lt_i * 8 + ii
            exact_b = ws_r[be, nb_i, ss_i, wp_i, g_i, lt_i, ii].to(torch.int64)
            recs = (kg_i << 16) | (g_i << 8) | exact_b
            out_wuid_list.append(wuid)
            out_rec_list.append(recs)

    num_w = E * n_panels * 4
    has_patches = bool(out_wuid_list)
    if not has_patches:
        p_off = torch.zeros(num_w + 1, dtype=torch.int32, device=dev)
        patches = torch.zeros(1, dtype=torch.uint32, device=dev)
    else:
        wuid_all = torch.cat(out_wuid_list)
        recs_all = torch.cat(out_rec_list)
        order = torch.argsort(wuid_all, stable=True)
        wuid_all = wuid_all[order]
        counts = torch.bincount(wuid_all, minlength=num_w).to(torch.int32)
        p_off = torch.zeros(num_w + 1, dtype=torch.int32, device=dev)
        p_off[1:] = counts.cumsum(0)
        patches = recs_all[order].to(torch.uint32).contiguous()
        del wuid_all, recs_all, order, counts
    del out_wuid_list, out_rec_list

    # Release original raw buffers now that slabs holds the 0-ULP lossless packed data
    # (net -23.9 MB/layer = -0.96 GB across 40 layers, instead of +21.96 GB!).
    weight.data.untyped_storage().resize_(0)
    weight_scale.data.untyped_storage().resize_(0)

    meta = {
        "proj": 0,
        "slabs": slabs,
        "ws_base": ws_base,
        "p_off": p_off,
        "patches": patches,
        "has_patches": has_patches,
    }
    weight._inkling_nvfp4_ws5 = meta
    return meta


def pack_w2_ws5_inplace(weight: torch.Tensor, weight_scale: torch.Tensor) -> dict:
    """Pack raw W2 [E, 4608, 144] IN-PLACE and WS [E, 4608, 18] IN-PLACE into first 5/8 of WS."""
    meta = getattr(weight, "_inkling_nvfp4_ws5", None)
    if meta is not None:
        return meta
    assert not torch.cuda.is_current_stream_capturing()
    w_u8 = weight.data.view(torch.uint8)
    ws_u8 = weight_scale.data.view(torch.uint8)
    E, N, KP = w_u8.shape
    KG = ws_u8.shape[2]
    assert N == 4608 and KP == 144 and KG == 18
    n_panels = 36
    dev = w_u8.device

    ws_base = _compute_ws_base(ws_u8)
    ws5_pack = torch.empty((E, n_panels, 360), dtype=torch.uint32, device=dev)

    step = max(1, (64 << 20) // (N * KP))
    shifts4 = torch.arange(8, device=dev, dtype=torch.int64) * 4
    shifts1 = torch.arange(32, device=dev, dtype=torch.int64)
    out_puid_list, out_rec_list = [], []

    for e0 in range(0, E, step):
        e1 = min(e0 + step, E)
        B = e1 - e0
        # In-place repack of W2
        blk = w_u8[e0:e1]
        w_p = blk.reshape(B, n_panels, 4, 4, 8, 9, 2, 4, 2).permute(0, 1, 2, 5, 4, 7, 3, 6, 8).contiguous()
        blk.copy_(w_p.view(blk.shape))

        ws_r = ws_u8[e0:e1].reshape(B, n_panels, 128 * 18).contiguous()
        diff = ws_r.to(torch.int32) - ws_base[e0:e1].to(torch.int32)[:, None, None]
        in_range = (diff >= 0) & (diff < 32)
        code = torch.where(in_range, diff, torch.zeros_like(diff)).to(torch.int64)
        ws_lo4 = ((code.reshape(B, n_panels, 288, 8) & 0xF) << shifts4).sum(-1).to(torch.uint32)
        ws_hi1 = (((code.reshape(B, n_panels, 72, 32) >> 4) & 1) << shifts1).sum(-1).to(torch.uint32)
        ws5_pack[e0:e1, :, :288].copy_(ws_lo4)
        ws5_pack[e0:e1, :, 288:360].copy_(ws_hi1)

        outliers = (~in_range).nonzero(as_tuple=False)
        if outliers.numel() > 0:
            be, nb_i, idx_i = outliers.unbind(dim=1)
            puid = (be + e0) * n_panels + nb_i
            exact_b = ws_r[be, nb_i, idx_i].to(torch.int64)
            recs = (idx_i << 8) | exact_b
            out_puid_list.append(puid)
            out_rec_list.append(recs)

    # Keep the compact 5/8 ws5_pack directly and release the full 8/8 weight_scale storage!
    # Reclaims 3/8 of w2_weight_scale = 11.39 MiB/layer = 455.6 MiB (0.445 GiB/rank) across 40L.
    ws5_view = ws5_pack
    weight_scale.data.untyped_storage().resize_(0)

    num_p = E * n_panels
    has_patches = bool(out_puid_list)
    if not has_patches:
        p_off = torch.zeros(num_p + 1, dtype=torch.int32, device=dev)
        patches = torch.zeros(1, dtype=torch.uint32, device=dev)
    else:
        puid_all = torch.cat(out_puid_list)
        recs_all = torch.cat(out_rec_list)
        order = torch.argsort(puid_all, stable=True)
        puid_all = puid_all[order]
        counts = torch.bincount(puid_all, minlength=num_p).to(torch.int32)
        p_off = torch.zeros(num_p + 1, dtype=torch.int32, device=dev)
        p_off[1:] = counts.cumsum(0)
        patches = recs_all[order].to(torch.uint32).contiguous()

    meta = {
        "proj": 1,
        "w_pack": w_u8,
        "ws5_pack": ws5_view,
        "ws_base": ws_base,
        "p_off": p_off,
        "patches": patches,
        "has_patches": has_patches,
    }
    weight._inkling_nvfp4_ws5 = meta
    return meta


@triton.jit
def _scale_offsets_pf(row, cols, COLS: tl.constexpr, TILE: tl.constexpr):
    if TILE >= 128:
        tile = (row // 128) * ((COLS + 3) // 4) + cols // 4
        return tile * 512 + (row % 32) * 16 + (row // 32 % 4) * 4 + cols % 4
    if TILE > 0:
        tile = (row // 8) * ((COLS + 3) // 4) + cols // 4
        return tile * 32 + (row % 8) * 4 + cols % 4
    return row * COLS + cols


@triton.jit
def _e2m1_hw_pf(packed):
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
def _w13_ws5_ks_step(
    acc, ks, a_rows, as_rows, AS, rows, rmask, panel_base, base_e,
    p0, p1, p2, p3, p4, PATCHES,
    n_off, r64, c64, c8, c2, b8, bkg16, bkp32, bkp64, bkg4, bkg8,
    KG: tl.constexpr, TILE: tl.constexpr, BM: tl.constexpr,
    BK_DOT: tl.constexpr, APPLY_PATCH: tl.constexpr,
    USE_R2T: tl.constexpr, AS_ROW_MAJOR: tl.constexpr,
):
    """Single 256-K (128 packed bytes) slab step for W13 SLABS [E, 18, 9, 8832].
    Loads 4096 B contiguous W + 320 B contiguous WS5, unswizzles in registers, and
    executes exact BK_DOT=128 (tile<=64) or BK_DOT=64 (tile>=128) MMA order (0-ULP)."""
    ss = ks >> 1
    ks2 = ks & 1
    ss_ptr = panel_base + ss * 8832

    # Unpack 16 K-groups of WS5 scales for all 32 rows of this panel (320 B total;
    # left L1/L2-cached so ks2=1 hits the same 512B/128B cache lines fetched at ks2=0).
    lo_i = tl.load(ss_ptr + 8192 + n_off[:, None] * 16 + (ks2 * 8 + c8[None, :])).to(tl.int32)
    hi_i = tl.load(ss_ptr + 8704 + n_off[:, None] * 4 + (ks2 * 2 + c2[None, :])).to(tl.int32)
    lo_u = tl.reshape(tl.join(lo_i & 0xF, (lo_i >> 4) & 0xF), (32, 16))
    hi_u = tl.reshape((hi_i[:, :, None] >> b8[None, None, :]) & 1, (32, 16))
    w_s_i = base_e + (lo_u | (hi_u << 4))

    if APPLY_PATCH:
        kg_lo = ks * 16
        for pi in range(p0, p4):
            rec = tl.load(PATCHES + pi).to(tl.int64)
            p_kg = (rec >> 16).to(tl.int32)
            p_wp = (
                tl.where(pi >= p1, 1, 0)
                + tl.where(pi >= p2, 1, 0)
                + tl.where(pi >= p3, 1, 0)
            )
            p_n = p_wp * 8 + ((rec >> 8) & 0xFF).to(tl.int32)
            p_val = (rec & 0xFF).to(tl.int32)
            hit = (n_off[:, None] == p_n) & (bkg16[None, :] == (p_kg - kg_lo))
            w_s_i = tl.where(hit, p_val, w_s_i)

    ws_full = w_s_i.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float16)

    # Single contiguous 128B-aligned 4096 B load: [wp=4, blk=2, g_n=8, lt=4, q2=2, mmap=2, half=2, b2=2]
    w_slab = tl.load(
        ss_ptr + ks2 * 4096 + r64[:, None] * 64 + c64[None, :],
        eviction_policy="evict_first",
    )
    wb0, wb1 = tl.split(tl.permute(tl.reshape(w_slab, (4, 2, 8, 64)), (0, 2, 3, 1)))

    if BK_DOT == 128:
        ws_b0, ws_b1 = tl.split(tl.permute(tl.reshape(ws_full, (32, 2, 8)), (0, 2, 1)))
        ws_halves = (ws_b0, ws_b1)
        w_halves = (wb0, wb1)
        for b in tl.static_range(0, 2):
            kp = ks * 128 + b * 64 + bkp64
            kg = ks * 16 + b * 8 + bkg8
            a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
            if USE_R2T or AS_ROW_MAJOR:
                a_s = tl.load(as_rows + kg[None, :], rmask[:, None], other=0)
            else:
                a_s = tl.load(
                    AS + _scale_offsets_pf(rows[:, None], kg[None, :], KG, TILE),
                    rmask[:, None],
                    other=0,
                )
            a_s = a_s.to(tl.float8e4nv, bitcast=True).to(tl.float16)
            a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, 8, 8)), (BM, 64))
            a_lo, a_hi = _e2m1_hw_pf(a_raw)

            wt = tl.permute(tl.reshape(w_halves[b], (32, 4, 2, 8)), (0, 2, 1, 3))
            wt = tl.permute(tl.reshape(wt, (64, 8, 2, 2)), (0, 2, 1, 3))
            w_raw = tl.reshape(wt, (32, 64))
            w_s = tl.reshape(tl.broadcast_to(ws_halves[b][:, :, None], (32, 8, 8)), (32, 64))
            w_lo, w_hi = _e2m1_hw_pf(w_raw)
            acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
            acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)
    else:
        ws01, ws23 = tl.split(tl.permute(tl.reshape(ws_full, (32, 2, 8)), (0, 2, 1)))
        ws0, ws1 = tl.split(tl.permute(tl.reshape(ws01, (32, 2, 4)), (0, 2, 1)))
        ws2, ws3 = tl.split(tl.permute(tl.reshape(ws23, (32, 2, 4)), (0, 2, 1)))
        wss = (ws0, ws1, ws2, ws3)

        w0, w1 = tl.split(tl.permute(tl.reshape(wb0, (32, 4, 2, 8)), (0, 1, 3, 2)))
        w2, w3 = tl.split(tl.permute(tl.reshape(wb1, (32, 4, 2, 8)), (0, 1, 3, 2)))
        wqs = (w0, w1, w2, w3)

        for q in tl.static_range(0, 4):
            kp = ks * 128 + q * 32 + bkp32
            kg = ks * 16 + q * 4 + bkg4
            a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
            if USE_R2T or AS_ROW_MAJOR:
                a_s = tl.load(as_rows + kg[None, :], rmask[:, None], other=0)
            else:
                a_s = tl.load(
                    AS + _scale_offsets_pf(rows[:, None], kg[None, :], KG, TILE),
                    rmask[:, None],
                    other=0,
                )
            a_s = a_s.to(tl.float8e4nv, bitcast=True).to(tl.float16)
            a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, 4, 8)), (BM, 32))
            a_lo, a_hi = _e2m1_hw_pf(a_raw)

            w_raw = tl.reshape(tl.permute(tl.reshape(wqs[q], (32, 8, 2, 2)), (0, 2, 1, 3)), (32, 32))
            w_s = tl.reshape(tl.broadcast_to(wss[q][:, :, None], (32, 4, 8)), (32, 32))
            w_lo, w_hi = _e2m1_hw_pf(w_raw)
            acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
            acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)

    return acc


@triton.jit
def _nvfp4_ws5_prefill_w13_kernel(
    A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BK_DOT: tl.constexpr,
    HAS_PATCHES: tl.constexpr,
    USE_R2T: tl.constexpr = False,
    AS_ROW_MAJOR: tl.constexpr = False,
):
    """Fast 0-ULP prefill/high-concurrency W13 kernel (N=576, K=4608, PB=32) reading WS5 SLABS.
    Eliminates dead CTAs, uses L2-friendly CTA ordering (pid0 = nb*SUB_M+sub, pid1 = t),
    loads 4096 B contiguous slabs + 320 B WS5 with zero scalar div/mod and zero N-padding,
    and runs a pure pipelined patch-free K-loop on the 99.4% of panels without outliers."""
    SUB_M: tl.constexpr = TILE // BM
    pid0 = tl.program_id(0)
    t = tl.program_id(1)
    nb = pid0 // SUB_M
    sub = pid0 % SUB_M

    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    end = tl.load(Tiles + MAX_CTAS + t)
    if t * TILE + sub * BM >= end:
        return

    e = tl.load(Tiles + t).to(tl.int64)
    local = sub * BM + tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)

    N: tl.constexpr = 576
    KP: tl.constexpr = 2304
    KG: tl.constexpr = 288

    if USE_R2T:
        a_idx = tl.load(Row2Tok + rows, rmask, other=0).to(tl.int64)
    else:
        a_idx = rows.to(tl.int64)
    a_rows = A + a_idx[:, None] * KP
    as_rows = AS + a_idx[:, None] * KG

    panel_base = SLABS + (e * 18 + nb) * (9 * 8832)
    base_e = tl.load(WS_BASE + e).to(tl.int32)

    n_off = tl.arange(0, 32)
    r64 = tl.arange(0, 64)
    c64 = tl.arange(0, 64)
    c8 = tl.arange(0, 8)
    c2 = tl.arange(0, 2)
    b8 = tl.arange(0, 8)
    bkg16 = tl.arange(0, 16)
    bkp32 = tl.arange(0, 32)
    bkp64 = tl.arange(0, 64)
    bkg4 = tl.arange(0, 4)
    bkg8 = tl.arange(0, 8)

    acc = tl.zeros((BM, 32), dtype=tl.float32)

    if not HAS_PATCHES:
        for ks in range(0, 18):
            acc = _w13_ws5_ks_step(
                acc, ks, a_rows, as_rows, AS, rows, rmask, panel_base, base_e,
                0, 0, 0, 0, 0, PATCHES,
                n_off, r64, c64, c8, c2, b8, bkg16, bkp32, bkp64, bkg4, bkg8,
                KG, TILE, BM, BK_DOT, False, USE_R2T, AS_ROW_MAJOR,
            )
    else:
        wuid0 = (e * 18 + nb) * 4
        p0 = tl.load(P_OFF + wuid0)
        p1 = tl.load(P_OFF + wuid0 + 1)
        p2 = tl.load(P_OFF + wuid0 + 2)
        p3 = tl.load(P_OFF + wuid0 + 3)
        p4 = tl.load(P_OFF + wuid0 + 4)
        if p0 == p4:
            for ks in range(0, 18):
                acc = _w13_ws5_ks_step(
                    acc, ks, a_rows, as_rows, AS, rows, rmask, panel_base, base_e,
                    p0, p1, p2, p3, p4, PATCHES,
                    n_off, r64, c64, c8, c2, b8, bkg16, bkp32, bkp64, bkg4, bkg8,
                    KG, TILE, BM, BK_DOT, False, USE_R2T, AS_ROW_MAJOR,
                )
        else:
            for ks in tl.range(0, 18, num_stages=1):
                acc = _w13_ws5_ks_step(
                    acc, ks, a_rows, as_rows, AS, rows, rmask, panel_base, base_e,
                    p0, p1, p2, p3, p4, PATCHES,
                    n_off, r64, c64, c8, c2, b8, bkg16, bkp32, bkp64, bkg4, bkg8,
                    KG, TILE, BM, BK_DOT, True, USE_R2T, AS_ROW_MAJOR,
                )

    scale = tl.load(Ones + e)
    acc = acc * tl.load(RS + rows, rmask, other=0.0)[:, None]
    acc = acc * scale
    n0 = nb * 32 + n_off
    tl.store(Out + rows.to(tl.int64)[:, None] * N + n0[None, :], acc.to(tl.bfloat16), rmask[:, None])


@triton.jit
def _w2_ws5_pair_step(
    acc, p, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
    base_e, p_lo, p_hi, PATCHES,
    wp4, b1024, n128, c2, bkp32, bkg4,
    KG: tl.constexpr, TILE: tl.constexpr, BM: tl.constexpr,
    APPLY_PATCH: tl.constexpr, AS_ROW_MAJOR: tl.constexpr,
):
    """Single 64-K (32 packed bytes) pair step for W2 W_PACK [E, 36, 4, 9, 32, 16] + WS5_PACK."""
    kp = p * 32 + bkp32
    kg = p * 4 + bkg4
    a_raw = tl.load(a_rows + kp[None, :], rmask[:, None], other=0)
    if AS_ROW_MAJOR:
        a_s = tl.load(as_rows + kg[None, :], rmask[:, None], other=0)
    else:
        a_s = tl.load(
            AS + _scale_offsets_pf(rows[:, None], kg[None, :], KG, TILE),
            rmask[:, None],
            other=0,
        )
    a_s = a_s.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, 4, 8)), (BM, 32))
    a_lo, a_hi = _e2m1_hw_pf(a_raw)

    # Contiguous 4096 B (4x1024) load covering ks = 2*p and 2*p+1 across all 4 warps (128B-aligned!)
    # Memory order: [wp=4, k2=2, g_n=8, lt=4, sub_n=4, half=2, b2=2]
    w_pair = tl.load(
        w_panel + wp4[:, None] * 4608 + p * 1024 + b1024[None, :],
        eviction_policy="evict_first",
    )
    w = tl.permute(tl.reshape(w_pair, (64, 4, 4, 4)), (0, 2, 1, 3))
    w = tl.permute(tl.reshape(w, (8, 8, 4, 16)), (0, 2, 1, 3))
    w = tl.permute(tl.reshape(w, (4, 2, 32, 16)), (0, 2, 1, 3))
    w = tl.permute(tl.reshape(w, (256, 4, 2, 2)), (0, 2, 1, 3))
    w_raw = tl.reshape(w, (128, 32))

    # WS5 unpack for kg = 4*p .. 4*p+3 (1440 B panel stays L1-hot across all 4 pairs + tail)
    lo_b = tl.load(ws_lo_panel + n128[:, None] * 9 + (p * 2 + c2[None, :])).to(tl.int32)
    lo_u = tl.reshape(tl.join(lo_b & 0xF, (lo_b >> 4) & 0xF), (128, 4))
    bit0 = n128 * 18 + p * 4
    by0 = bit0 >> 3
    sh0 = bit0 & 7
    h0 = tl.load(ws_hi_panel + by0).to(tl.int32)
    h1 = tl.load(ws_hi_panel + by0 + 1, mask=(by0 + 1) < 288, other=0).to(tl.int32)
    h16 = h0 | (h1 << 8)
    hi_u = (h16[:, None] >> (sh0[:, None] + bkg4[None, :])) & 1
    w_s_i = base_e + (lo_u | (hi_u << 4))

    if APPLY_PATCH:
        kg_lo = p * 4
        for pi in range(p_lo, p_hi):
            rec = tl.load(PATCHES + pi).to(tl.int64)
            p_idx = (rec >> 8).to(tl.int32)
            p_n = p_idx // 18
            p_kg = p_idx - p_n * 18
            p_val = (rec & 0xFF).to(tl.int32)
            hit = (n128[:, None] == p_n) & (bkg4[None, :] == (p_kg - kg_lo))
            w_s_i = tl.where(hit, p_val, w_s_i)

    w_s = w_s_i.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float16)
    w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (128, 4, 8)), (128, 32))
    w_lo, w_hi = _e2m1_hw_pf(w_raw)
    acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
    acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)
    return acc


@triton.jit
def _w2_ws5_tail_step(
    acc, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
    base_e, p_lo, p_hi, PATCHES,
    wp4, b512, n128, c2, bkp32, bkg4,
    KP: tl.constexpr, KG: tl.constexpr, TILE: tl.constexpr, BM: tl.constexpr,
    APPLY_PATCH: tl.constexpr, AS_ROW_MAJOR: tl.constexpr,
):
    """Exact BK=64 zero-padded tail step (ks=8, kp=128..143, kg=16..17 -> padded to 32/4) for 0-ULP."""
    kp = 128 + bkp32
    kg = 16 + bkg4
    a_raw = tl.load(a_rows + kp[None, :], rmask[:, None] & (kp < KP)[None, :], other=0)
    if AS_ROW_MAJOR:
        a_s = tl.load(as_rows + kg[None, :], rmask[:, None] & (kg < KG)[None, :], other=0)
    else:
        a_s = tl.load(
            AS + _scale_offsets_pf(rows[:, None], kg[None, :], KG, TILE),
            rmask[:, None] & (kg < KG)[None, :],
            other=0,
        )
    a_s = a_s.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    a_s = tl.reshape(tl.broadcast_to(a_s[:, :, None], (BM, 4, 8)), (BM, 32))
    a_lo, a_hi = _e2m1_hw_pf(a_raw)

    # Tail W: [4, 512] at offset 8 * 512 = 4096 inside each warp's 4608B block
    w_tail = tl.load(
        w_panel + wp4[:, None] * 4608 + 4096 + b512[None, :],
        eviction_policy="evict_first",
    )
    wt = tl.permute(tl.reshape(w_tail, (32, 4, 4, 4)), (0, 2, 1, 3))
    wt = tl.permute(tl.reshape(wt, (4, 8, 4, 16)), (0, 2, 1, 3))
    wt = tl.permute(tl.reshape(wt, (128, 4, 2, 2)), (0, 2, 1, 3))
    w0_16 = tl.reshape(wt, (128, 16))
    z_w = tl.zeros((128, 16), dtype=tl.uint8)
    w_raw = tl.reshape(tl.permute(tl.join(w0_16, z_w), (0, 2, 1)), (128, 32))

    # Tail WS5: kg = 16, 17 -> lo4 at col 8; hi1 bit0 = n*18 + 16 is always even so both bits lie in byte by0_t
    lo_bt = tl.load(ws_lo_panel + n128 * 9 + 8).to(tl.int32)
    lo_ut = tl.join(lo_bt & 0xF, (lo_bt >> 4) & 0xF)
    bit0_t = n128 * 18 + 16
    by0_t = bit0_t >> 3
    sh0_t = bit0_t & 7
    h0_t = tl.load(ws_hi_panel + by0_t).to(tl.int32)
    hi_ut = (h0_t[:, None] >> (sh0_t[:, None] + c2[None, :])) & 1
    w_st_i = base_e + (lo_ut | (hi_ut << 4))

    if APPLY_PATCH:
        for pi in range(p_lo, p_hi):
            rec = tl.load(PATCHES + pi).to(tl.int64)
            p_idx = (rec >> 8).to(tl.int32)
            p_n = p_idx // 18
            p_kg = p_idx - p_n * 18
            p_val = (rec & 0xFF).to(tl.int32)
            hit = (n128[:, None] == p_n) & (c2[None, :] == (p_kg - 16))
            w_st_i = tl.where(hit, p_val, w_st_i)

    ws0_2 = w_st_i.to(tl.uint8)
    z_ws = tl.zeros((128, 2), dtype=tl.uint8)
    w_s = (
        tl.reshape(tl.permute(tl.join(ws0_2, z_ws), (0, 2, 1)), (128, 4))
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float16)
    )
    w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (128, 4, 8)), (128, 32))
    w_lo, w_hi = _e2m1_hw_pf(w_raw)
    acc = tl.dot(a_lo * a_s, tl.trans(w_lo * w_s), acc)
    acc = tl.dot(a_hi * a_s, tl.trans(w_hi * w_s), acc)
    return acc


@triton.jit
def _nvfp4_ws5_prefill_w2_kernel(
    A, AS, RS, W_PACK, WS5_U8, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out,
    MAX_CTAS: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    HAS_PATCHES: tl.constexpr,
    AS_ROW_MAJOR: tl.constexpr = False,
):
    """Fast 0-ULP prefill/high-concurrency W2 kernel (N=4608, K=288, PB=128) reading WS5 W_PACK+WS5_PACK.
    Eliminates dead CTAs, uses L2-friendly CTA ordering (pid0 = nb*SUB_M+sub, pid1 = t),
    loads 4096 B contiguous pairs + 1440 B WS5 with zero scalar div/mod, zero-pads the K=288 tail
    to [BM, 32] x [32, 128] for literal nd == 0, and runs a pure pipelined fast path when patch-free."""
    SUB_M: tl.constexpr = TILE // BM
    pid0 = tl.program_id(0)
    t = tl.program_id(1)
    nb = pid0 // SUB_M
    sub = pid0 % SUB_M

    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    end = tl.load(Tiles + MAX_CTAS + t)
    if t * TILE + sub * BM >= end:
        return

    e = tl.load(Tiles + t).to(tl.int64)
    local = sub * BM + tl.arange(0, BM)
    rows = t * TILE + local
    rmask = (local < TILE) & (rows < end)

    N: tl.constexpr = 4608
    KP: tl.constexpr = 144
    KG: tl.constexpr = 18

    a_rows = A + rows.to(tl.int64)[:, None] * KP
    as_rows = AS + rows.to(tl.int64)[:, None] * KG

    puid = e * 36 + nb
    w_panel = W_PACK + puid * 18432
    ws_lo_panel = WS5_U8 + puid * 1440
    ws_hi_panel = ws_lo_panel + 1152
    base_e = tl.load(WS_BASE + e).to(tl.int32)

    wp4 = tl.arange(0, 4)
    b1024 = tl.arange(0, 1024)
    b512 = tl.arange(0, 512)
    n128 = tl.arange(0, 128)
    c2 = tl.arange(0, 2)
    bkp32 = tl.arange(0, 32)
    bkg4 = tl.arange(0, 4)

    acc = tl.zeros((BM, 128), dtype=tl.float32)

    if not HAS_PATCHES:
        for p in range(0, 4):
            acc = _w2_ws5_pair_step(
                acc, p, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
                base_e, 0, 0, PATCHES,
                wp4, b1024, n128, c2, bkp32, bkg4,
                KG, TILE, BM, False, AS_ROW_MAJOR,
            )
        acc = _w2_ws5_tail_step(
            acc, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
            base_e, 0, 0, PATCHES,
            wp4, b512, n128, c2, bkp32, bkg4,
            KP, KG, TILE, BM, False, AS_ROW_MAJOR,
        )
    else:
        p_lo = tl.load(P_OFF + puid)
        p_hi = tl.load(P_OFF + puid + 1)
        if p_lo == p_hi:
            for p in range(0, 4):
                acc = _w2_ws5_pair_step(
                    acc, p, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
                    base_e, p_lo, p_hi, PATCHES,
                    wp4, b1024, n128, c2, bkp32, bkg4,
                    KG, TILE, BM, False, AS_ROW_MAJOR,
                )
            acc = _w2_ws5_tail_step(
                acc, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
                base_e, p_lo, p_hi, PATCHES,
                wp4, b512, n128, c2, bkp32, bkg4,
                KP, KG, TILE, BM, False, AS_ROW_MAJOR,
            )
        else:
            for p in tl.range(0, 4, num_stages=1):
                acc = _w2_ws5_pair_step(
                    acc, p, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
                    base_e, p_lo, p_hi, PATCHES,
                    wp4, b1024, n128, c2, bkp32, bkg4,
                    KG, TILE, BM, True, AS_ROW_MAJOR,
                )
            acc = _w2_ws5_tail_step(
                acc, a_rows, as_rows, AS, rows, rmask, w_panel, ws_lo_panel, ws_hi_panel,
                base_e, p_lo, p_hi, PATCHES,
                wp4, b512, n128, c2, bkp32, bkg4,
                KP, KG, TILE, BM, True, AS_ROW_MAJOR,
            )

    scale = tl.load(Ones + e)
    acc = acc * tl.load(RS + rows, rmask, other=0.0)[:, None]
    acc = acc * scale
    n0 = nb * 128 + n128
    tl.store(Out + rows.to(tl.int64)[:, None] * N + n0[None, :], acc.to(tl.bfloat16), rmask[:, None])


def bmm_nvfp4_ws5(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan,
    projection: int,
    pdl: bool = True,
    *,
    row2tok: torch.Tensor | None = None,
    as_row_major: bool = False,
) -> torch.Tensor | None:
    """Drop-in fast path for trtllm_bmm.bmm() on L0..39 NVFP4 MoE (returns None if lib unavailable)."""
    lib = _get_lib()
    if lib is None or weight.dtype != torch.uint8:
        return None
    n = weight.shape[1]
    k = weight.shape[2] * 2
    if not ((projection == 0 and n == 576 and k == 4608) or (projection == 1 and n == 4608 and k == 288)):
        return None
    if plan.tile <= 16 and (row2tok is not None or as_row_major):
        return None

    meta = pack_w13_ws5_inplace(weight, weight_scale) if projection == 0 else pack_w2_ws5_inplace(weight, weight_scale)
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)

    if plan.tile <= 16:
        use_pdl = pdl and plan.rows <= 2048
        pdl_wait = (2 if projection == 1 else 1) if use_pdl else 0
        trig_dep = 1 if use_pdl else 0
        st = torch.cuda.current_stream(act.device).cuda_stream
        if projection == 0:
            rc = lib.nvfp4_w13_ws5_fwd(
                act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(),
                meta["slabs"].data_ptr(), meta["ws_base"].data_ptr(),
                meta["p_off"].data_ptr(), meta["patches"].data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                n, k, plan.max_ctas, plan.tile, pdl_wait, trig_dep, st,
            )
        else:
            rc = lib.nvfp4_w2_ws5_fwd(
                act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(),
                meta["w_pack"].data_ptr(), meta["ws5_pack"].data_ptr(), meta["ws_base"].data_ptr(),
                meta["p_off"].data_ptr(), meta["patches"].data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                n, k, plan.max_ctas, plan.tile, pdl_wait, trig_dep, st,
            )
        assert rc == 0, f"nvfp4_ws5 CUDA error {rc}"
        return out

    # Fast 0-ULP panel-aligned prefill / high-concurrency path (plan.tile >= 32):
    # reads the exact same packed WS5 tensors with dead-CTA exit & L2-friendly CTA order!
    has_patches = meta.get("has_patches", True)
    if projection == 0:
        use_r2t = row2tok is not None
        r2t_arg = row2tok if use_r2t else tiles
        if plan.tile <= 64:
            bk_dot = 128
            bm, warps, stages = (32, 2, 3) if plan.tile == 32 else (64, 2, 3)
        else:
            bk_dot = 64
            if plan.max_ctas <= 520:
                bm, warps, stages = 64, 2, 3
            else:
                bm, warps, stages = 128, 2, 2
        bm = min(bm, plan.tile)
        sub_m = plan.tile // bm
        grid = (18 * sub_m, plan.max_ctas)
        _nvfp4_ws5_prefill_w13_kernel[grid](
            act.view(torch.uint8), act_scale.view(torch.uint8), row_scale, r2t_arg,
            meta["slabs"], meta["ws_base"], meta["p_off"], meta["patches"],
            ones, tiles, out,
            MAX_CTAS=plan.max_ctas, TILE=plan.tile, BM=bm, BK_DOT=bk_dot,
            HAS_PATCHES=has_patches, USE_R2T=use_r2t, AS_ROW_MAJOR=as_row_major,
            num_warps=warps, num_stages=stages,
        )
        return out

    # projection == 1 (W2: N=4608, K=288, PB=128, 36 panels, exact BK=64 + padded tail)
    if plan.tile <= 64:
        bm, warps, stages = 32, 4, 3
    elif plan.max_ctas <= 460:
        bm, warps, stages = 32, 4, 3
    elif plan.max_ctas <= 520:
        bm, warps, stages = 64, 4, 3
    else:
        bm, warps, stages = 128, 8, 2
    bm = min(bm, plan.tile)
    sub_m = plan.tile // bm
    grid = (36 * sub_m, plan.max_ctas)
    _nvfp4_ws5_prefill_w2_kernel[grid](
        act.view(torch.uint8), act_scale.view(torch.uint8), row_scale,
        meta["w_pack"], meta["ws5_pack"].view(torch.uint8),
        meta["ws_base"], meta["p_off"], meta["patches"],
        ones, tiles, out,
        MAX_CTAS=plan.max_ctas, TILE=plan.tile, BM=bm,
        HAS_PATCHES=has_patches, AS_ROW_MAJOR=as_row_major,
        num_warps=warps, num_stages=stages,
    )
    return out
