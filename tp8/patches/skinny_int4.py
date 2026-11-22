# SPDX-License-Identifier: Apache-2.0
"""SM12x (GB10) 4-bit skinny GEMM for the DSpark drafter: out[M, N] = x[M, K] @ dequant(w4[N, K]).T.

Supports high-accuracy 4-bit panel & row-major formats:
  - ``"u4p32"`` / ``"u4p64"``: Panel-contiguous asymmetric uint4 (g=32: 5.00 bpw, g=64: 4.50 bpw).
  - ``"i4p32"`` / ``"i4p64"``: Panel-contiguous symmetric MSE-clipped int4 (q - 7.5) * sc with NO zero-point
                               tensor (g=32: 4.50 bpw, g=64: 4.25 bpw; -10% to -15% bytes vs u4p32,
                               hits ~228 GB/s on draft_head and ~208 GB/s on gate_up!).
  - ``"nvfp4"``: Group-16 NVFP4 (0.5625 B/w, hardware cvt.rn.f16x2.e2m1x2).
  - ``"uint4"``: Row-major group-32 asymmetric uint4 (0.625 B/w).

Fused options on panel kernels (M <= 64):
  - ``fuse_silu=True``: Interleaved [gate, up] panel weights -> fp32 silu(gate)*up directly in epilogue,
                        writing [M, N//2] and eliminating _silu_and_mul_kernel (-15.2 us/step).
  - ``x2=f_rnn`` (``fuse_rnn=True``): Reads x[m, k] for k < K//2 and broadcasts x2[0, k - K//2] for
                        k >= K//2, eliminating torch.cat + contiguous in MarkovHead.context (-5.8 us/step).

Used ONLY for the drafter (target model stays 100% bit-exact under greedy verification).
"""

import os

import torch

from vllm.triton_utils import tl, tldevice, triton

_U4_PANEL_MAX_M = int(os.environ.get("INKLING_U4_PANEL_MAX_M", "256") or 256)
# INKLING_U4_MT_BM64=1: opt-in BM=64 (8 warps) tile for _u4_panel_mt_kernel when M >= 64
# (halves redundant 4-bit weight dequant ALU & L2 weight fetches at B=12..32 with 0 extra VRAM!).
_U4_MT_BM64 = os.environ.get("INKLING_U4_MT_BM64", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_PDL = os.environ.get("INKLING_DRAFT_PDL", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# INKLING_DRAFT_U4_FUSE_REDUCE=1: fuse split-K reduction + optional bias directly into the
# last-arriving CTA of _u4_panel_kernel (M <= 32) via per-bank 2x ping-pong (PART, SEM) scratch,
# eliminating 10x _reduce_kernel + 1x _reduce_bias_kernel launches (bit-exact deterministic order).
_U4_FUSE_REDUCE = os.environ.get("INKLING_DRAFT_U4_FUSE_REDUCE", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)

# Per-(device, bank) 2x ping-pong persistent scratch for fused split-K last-CTA reduction.
# Bank 0 serves the main / CUDA-graph-capture stream (handles torch.cuda.graph's fresh capture
# stream transparently); registered side streams (e.g. drafter ctx_kv stream) get disjoint banks
# >= 1 so concurrent split-K kernels across streams never contend on PART or SEM.
_FUSE_PART_BUFS: dict[tuple[int, int], list[torch.Tensor]] = {}
_FUSE_SEM_BUFS: dict[tuple[int, int], list[torch.Tensor]] = {}
_FUSE_SCRATCH_IDX: dict[tuple[int, int], int] = {}
_SIDE_STREAM_BANK: dict[tuple[int, int], int] = {}
_NEXT_SIDE_BANK: dict[int, int] = {}
_MIN_FUSE_PART_ELEMS = 131072  # 512 KiB/slot (>= 32*3*32*32 and 24*4*32*32 = 98,304 floats)
_MIN_FUSE_SEM_TILES = 256


def _dev_idx(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _ensure_bank_scratch(
    device: torch.device, bank: int, min_part_elems: int, min_sem_tiles: int
) -> None:
    key = (_dev_idx(device), bank)
    need_p = max(_MIN_FUSE_PART_ELEMS, int(min_part_elems))
    need_s = max(_MIN_FUSE_SEM_TILES, int(min_sem_tiles))
    parts = _FUSE_PART_BUFS.get(key)
    if parts is None or parts[0].numel() < need_p:
        _FUSE_PART_BUFS[key] = [
            torch.empty(need_p, dtype=torch.float32, device=device),
            torch.empty(need_p, dtype=torch.float32, device=device),
        ]
    sems = _FUSE_SEM_BUFS.get(key)
    if sems is None or sems[0].numel() < need_s:
        _FUSE_SEM_BUFS[key] = [
            torch.zeros(need_s, dtype=torch.int32, device=device),
            torch.zeros(need_s, dtype=torch.int32, device=device),
        ]
    if key not in _FUSE_SCRATCH_IDX:
        _FUSE_SCRATCH_IDX[key] = 0


def prealloc_stream_scratch(
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
    min_part_elems: int = _MIN_FUSE_PART_ELEMS,
    min_sem_tiles: int = _MIN_FUSE_SEM_TILES,
) -> int:
    """Pre-allocate fused split-K (PART, SEM) scratch outside CUDA graph capture.
    Pass stream=None for the main/capture bank (bank 0); pass a dedicated side stream
    to register and pre-allocate a disjoint bank (>= 1) for that stream."""
    if not _U4_FUSE_REDUCE or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return 0
    d = _dev_idx(device)
    if stream is None:
        bank = 0
    else:
        skey = (d, int(stream.cuda_stream))
        bank = _SIDE_STREAM_BANK.get(skey)
        if bank is None:
            bank = _NEXT_SIDE_BANK.get(d, 1)
            _NEXT_SIDE_BANK[d] = bank + 1
            _SIDE_STREAM_BANK[skey] = bank
    _ensure_bank_scratch(device, bank, min_part_elems, min_sem_tiles)
    return bank


def _next_fuse_scratch(
    device: torch.device, need_part: int, need_sem: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    d = _dev_idx(device)
    cur_ptr = int(torch.cuda.current_stream(device).cuda_stream)
    bank = _SIDE_STREAM_BANK.get((d, cur_ptr), 0)
    key = (d, bank)
    parts = _FUSE_PART_BUFS.get(key)
    sems = _FUSE_SEM_BUFS.get(key)
    if parts is None or parts[0].numel() < need_part or sems is None or sems[0].numel() < need_sem:
        if torch.cuda.is_current_stream_capturing():
            return None  # fail closed: never allocate/zero SEM inside graph capture
        _ensure_bank_scratch(device, bank, need_part, need_sem)
        parts = _FUSE_PART_BUFS[key]
        sems = _FUSE_SEM_BUFS[key]
    idx = _FUSE_SCRATCH_IDX.get(key, 0)
    _FUSE_SCRATCH_IDX[key] = idx ^ 1
    return parts[idx], sems[idx]


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
def _u4_to_f16_pair(packed):
    """uint8 [.., n] (lo nibble = even/first, hi nibble = odd/second) -> (lo_f16, hi_f16) in [0.0..15.0].
    Uses exact fp16 bias trick: (0x6400 | nibble).bitcast(fp16) - 1024.0 == float16(nibble)."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .b32 lo0, lo1, hi0, hi1;
        .reg .b32 m0f, magic, bias;
        mov.b32 m0f, 0x000f000f;
        mov.b32 magic, 0x64006400;
        mov.b32 bias, 0xe400e400;
        prmt.b32 lo0, $4, 0, 0x4140;
        prmt.b32 lo1, $4, 0, 0x4342;
        and.b32 lo0, lo0, m0f;
        and.b32 lo1, lo1, m0f;
        or.b32 lo0, lo0, magic;
        or.b32 lo1, lo1, magic;
        add.f16x2 $0, lo0, bias;
        add.f16x2 $1, lo1, bias;
        shr.u32 hi0, $4, 4;
        prmt.b32 hi1, hi0, 0, 0x4342;
        prmt.b32 hi0, hi0, 0, 0x4140;
        and.b32 hi0, hi0, m0f;
        and.b32 hi1, hi1, m0f;
        or.b32 hi0, hi0, magic;
        or.b32 hi1, hi1, magic;
        add.f16x2 $2, hi0, bias;
        add.f16x2 $3, hi1, bias;
        }
        """,
        "=r,=r,=r,=r,r",
        [packed],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _w4_nvfp4_kernel(
    X, W, WS, RS, P, O, M, N, K, sx, sw, sws, so_m, KS,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    DIRECT: tl.constexpr,
):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    nm = n < N
    mm = m < M
    acc = tl.zeros([BM, BN], tl.float32)
    k_lo = ps * KS
    k_hi = tl.minimum(k_lo + KS, K)
    BKP: tl.constexpr = BK // 2
    BKG: tl.constexpr = BK // 16
    kp_lim = K // 2
    kg_lim = K // 16
    for k0 in range(k_lo, k_hi, BK):
        kp = k0 // 2 + tl.arange(0, BKP)
        kg = k0 // 16 + tl.arange(0, BKG)
        kk = k0 + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        x_lo, x_hi = tl.split(tl.reshape(x, (BM, BKP, 2)))
        w_raw = tl.load(
            W + n[:, None].to(tl.int64) * sw + kp[None, :],
            mask=nm[:, None] & (kp[None, :] < kp_lim),
            other=0,
            eviction_policy="evict_first",
        )
        w_s = (
            tl.load(
                WS + n[:, None].to(tl.int64) * sws + kg[None, :],
                mask=nm[:, None] & (kg[None, :] < kg_lim),
                other=0,
                eviction_policy="evict_first",
            )
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float16)
        )
        w_s = tl.reshape(tl.broadcast_to(w_s[:, :, None], (BN, BKG, 8)), (BN, BKP))
        w_lo, w_hi = _e2m1_hw(w_raw)
        acc = tl.dot(x_lo, tl.trans(w_lo * w_s), acc)
        acc = tl.dot(x_hi, tl.trans(w_hi * w_s), acc)
    acc = acc * tl.load(RS + n, mask=nm, other=0.0)[None, :]
    if DIRECT:
        tl.store(O + m[:, None] * so_m + n[None, :], acc.to(O.dtype.element_ty), mask=mm[:, None] & nm[None, :])
    else:
        tl.store(P + ps * M * N + m[:, None] * N + n[None, :], acc, mask=mm[:, None] & nm[None, :])


@triton.jit
def _w4_uint4_kernel(
    X, W, WSC, WZP, P, O, M, N, K, sx, sw, sws, so_m, KS,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    DIRECT: tl.constexpr,
):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    nm = n < N
    mm = m < M
    acc = tl.zeros([BM, BN], tl.float32)
    k_lo = ps * KS
    k_hi = tl.minimum(k_lo + KS, K)
    BKP: tl.constexpr = BK // 2
    BKG: tl.constexpr = BK // 32
    kp_lim = K // 2
    kg_lim = K // 32
    for k0 in range(k_lo, k_hi, BK):
        kp = k0 // 2 + tl.arange(0, BKP)
        kg = k0 // 32 + tl.arange(0, BKG)
        kk = k0 + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        x_lo, x_hi = tl.split(tl.reshape(x, (BM, BKP, 2)))
        w_raw = tl.load(
            W + n[:, None].to(tl.int64) * sw + kp[None, :],
            mask=nm[:, None] & (kp[None, :] < kp_lim),
            other=0,
            eviction_policy="evict_first",
        )
        sc = tl.load(
            WSC + n[:, None].to(tl.int64) * sws + kg[None, :],
            mask=nm[:, None] & (kg[None, :] < kg_lim),
            other=0.0,
            eviction_policy="evict_first",
        )
        zp = tl.load(
            WZP + n[:, None].to(tl.int64) * sws + kg[None, :],
            mask=nm[:, None] & (kg[None, :] < kg_lim),
            other=0.0,
            eviction_policy="evict_first",
        )
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BKG, 16)), (BN, BKP))
        zp = tl.reshape(tl.broadcast_to(zp[:, :, None], (BN, BKG, 16)), (BN, BKP))
        q_lo, q_hi = _u4_to_f16_pair(w_raw)
        w_lo = q_lo * sc + zp
        w_hi = q_hi * sc + zp
        acc = tl.dot(x_lo, tl.trans(w_lo), acc)
        acc = tl.dot(x_hi, tl.trans(w_hi), acc)
    if DIRECT:
        tl.store(O + m[:, None] * so_m + n[None, :], acc.to(O.dtype.element_ty), mask=mm[:, None] & nm[None, :])
    else:
        tl.store(P + ps * M * N + m[:, None] * N + n[None, :], acc, mask=mm[:, None] & nm[None, :])


@triton.jit
def _reduce_kernel(
    P, O, MN, so_m, N, S: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = i < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(S):
        acc += tl.load(P + s * MN + i, mask=msk, other=0.0)
    tl.store(O + (i // N) * so_m + i % N, acc.to(O.dtype.element_ty), mask=msk)


@triton.jit
def _reduce_bias_kernel(
    P, BIAS, O, MN, so_m, N, S: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = i < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(S):
        acc += tl.load(P + s * MN + i, mask=msk, other=0.0)
    col = i % N
    acc += tl.load(BIAS + col, mask=msk, other=0.0)
    tl.store(O + (i // N) * so_m + col, acc.to(O.dtype.element_ty), mask=msk)


@triton.jit
def _u4_panel_kernel(
    X, X2, WP, SCP, ZPP, BIAS, P, SEM, O, M, N, K, sx, sx2, so_m, KS, N_REQS, K_STEPS,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    DIRECT: tl.constexpr, HAS_BIAS: tl.constexpr,
    SYM_I4: tl.constexpr, FUSE_SILU: tl.constexpr, FUSE_RNN: tl.constexpr,
    SPLIT: tl.constexpr = 1, FUSE_SPLIT: tl.constexpr = False,
    PDL: tl.constexpr = False,
    FUSE_RNN_BATCH: tl.constexpr = False,
):
    """Unified panel-contiguous 4-bit skinny GEMM for u4p32, u4p64, i4p32, and i4p64.
    - SYM_I4=False: asymmetric u4 (w = q * sc + zp) reading WP, SCP, ZPP.
    - SYM_I4=True:  symmetric MSE-clipped i4 (w = (q - 7.5) * sc) reading ONLY WP, SCP (no ZPP!).
    - FUSE_SILU=True (requires DIRECT=True and interleaved [gate, up] rows):
      computes fp32 silu(gate)*up in epilogue and writes [M, N//2].
    - FUSE_RNN=True (for MarkovHead.context, num_reqs*steps rows, K=2*H):
      k < K//2 reads X[m, k], k >= K//2 reads X2[m // n_spec, k - K//2] (sx2 == 0 for R=1 broadcast!).
    """
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    nm = n < N
    mm = m < M
    acc = tl.zeros([BM, BN], tl.float32)
    BKP: tl.constexpr = BK // 2
    BKG: tl.constexpr = BK // G
    G_HALF: tl.constexpr = G // 2
    k_lo = ps * KS
    k_hi = tl.minimum(k_lo + KS, K)
    n_ksteps = K // BK
    half_K = K // 2
    wp_base = WP + pn.to(tl.int64) * (n_ksteps * BN * BKP)
    sp_base = SCP + pn.to(tl.int64) * (n_ksteps * BKG * BN)
    if not SYM_I4:
        zp_base = ZPP + pn.to(tl.int64) * (n_ksteps * BKG * BN)
    bn_off = tl.arange(0, BN)
    bkp_off = tl.arange(0, BKP)
    bkg_off = tl.arange(0, BKG)
    if PDL:
        tl.extra.cuda.gdc_wait()
    if FUSE_RNN_BATCH:
        m_req = m % N_REQS
        m_step = m // N_REQS
        m_x = m_req * K_STEPS + m_step
    for k0 in range(k_lo, k_hi, BK):
        ks = k0 // BK
        kk = k0 + tl.arange(0, BK)
        if FUSE_RNN_BATCH:
            if k0 < half_K:
                x = tl.load(X + m_x[:, None] * sx + kk[None, :], mask=mm[:, None], other=0.0).to(tl.float16)
            else:
                x = tl.load(X2 + m_req[:, None] * sx2 + (kk - half_K)[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        elif FUSE_RNN:
            if k0 < half_K:
                x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None], other=0.0).to(tl.float16)
            else:
                # For R=1, sx2=0 broadcasts X2[0, :] across all M rows; for R>1, caller passes expanded/strided X2.
                x = tl.load(X2 + m[:, None] * sx2 + (kk - half_K)[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        else:
            x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        x_lo, x_hi = tl.split(tl.reshape(x, (BM, BKP, 2)))
        w_raw = tl.load(wp_base + ks * (BN * BKP) + bn_off[:, None] * BKP + bkp_off[None, :], eviction_policy="evict_first")
        sc = tl.load(sp_base + ks * (BKG * BN) + bkg_off[None, :] * BN + bn_off[:, None], eviction_policy="evict_first")
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BKG, G_HALF)), (BN, BKP))
        if SYM_I4:
            zp = sc * (-7.5)
        else:
            zp = tl.load(zp_base + ks * (BKG * BN) + bkg_off[None, :] * BN + bn_off[:, None], eviction_policy="evict_first")
            zp = tl.reshape(tl.broadcast_to(zp[:, :, None], (BN, BKG, G_HALF)), (BN, BKP))
        q_lo, q_hi = _u4_to_f16_pair(w_raw)
        acc = tl.dot(x_lo, tl.trans(q_lo * sc + zp), acc)
        acc = tl.dot(x_hi, tl.trans(q_hi * sc + zp), acc)
    if DIRECT:
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        if FUSE_SILU:
            gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
            out_v = tl.div_rn(gate, 1.0 + tldevice.exp(-gate)) * up
            nh = pn * (BN // 2) + tl.arange(0, BN // 2)
            tl.store(O + m[:, None] * so_m + nh[None, :], out_v.to(O.dtype.element_ty), mask=mm[:, None] & (nh[None, :] < (N // 2)))
        else:
            if HAS_BIAS:
                acc = acc + tl.load(BIAS + n, mask=nm, other=0.0)[None, :]
            tl.store(O + m[:, None] * so_m + n[None, :], acc.to(O.dtype.element_ty), mask=mm[:, None] & nm[None, :])
    elif FUSE_SPLIT:
        tile_base = pn.to(tl.int64) * (SPLIT * BM * BN)
        off_mn = m[:, None] * BN + bn_off[None, :]
        tl.store(
            P + tile_base + ps.to(tl.int64) * (BM * BN) + off_mn,
            acc,
            mask=mm[:, None] & nm[None, :],
        )
        tl.debug_barrier()
        if tl.atomic_add(SEM + pn, 1, sem="acq_rel") == SPLIT - 1:
            tl.store(SEM + pn, 0)
            tot = tl.zeros([BM, BN], tl.float32)
            for s in tl.static_range(SPLIT):
                tot += tl.load(
                    P + tile_base + s * (BM * BN) + off_mn,
                    mask=mm[:, None] & nm[None, :],
                    other=0.0,
                    cache_modifier=".cg",
                )
            if HAS_BIAS:
                tot = tot + tl.load(BIAS + n, mask=nm, other=0.0)[None, :]
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(O + m[:, None] * so_m + n[None, :], tot.to(O.dtype.element_ty), mask=mm[:, None] & nm[None, :])
    else:
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(P + ps * M * N + m[:, None] * N + n[None, :], acc, mask=mm[:, None] & nm[None, :])


@triton.jit
def _u4_panel_mt_kernel(
    X, X2, WP, SCP, ZPP, BIAS, P, O, M, N, K, sx, sx2, so_m, KS, N_REQS, K_STEPS,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    DIRECT: tl.constexpr, HAS_BIAS: tl.constexpr,
    SYM_I4: tl.constexpr, FUSE_SILU: tl.constexpr, FUSE_RNN: tl.constexpr,
    CACHE_W: tl.constexpr,
    PDL: tl.constexpr = False,
    FUSE_RNN_BATCH: tl.constexpr = False,
):
    """M-tiled panel 4-bit kernel for high concurrency (M=33..128) supporting u4p32/u4p64/i4p32/i4p64,
    FUSE_SILU, and FUSE_RNN. pid0=m_tile varies fastest so all M-tiles of the same weight panel
    co-schedule on SMs and share it in L2 (eliminates BM=64 register spills and extends M up to 128)."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    ps = tl.program_id(2)
    n = pn * BN + tl.arange(0, BN)
    m = pm * BM + tl.arange(0, BM)
    nm = n < N
    mm = m < M
    acc = tl.zeros([BM, BN], tl.float32)
    BKP: tl.constexpr = BK // 2
    BKG: tl.constexpr = BK // G
    G_HALF: tl.constexpr = G // 2
    k_lo = ps * KS
    k_hi = tl.minimum(k_lo + KS, K)
    n_ksteps = K // BK
    half_K = K // 2
    wp_base = WP + pn.to(tl.int64) * (n_ksteps * BN * BKP)
    sp_base = SCP + pn.to(tl.int64) * (n_ksteps * BKG * BN)
    if not SYM_I4:
        zp_base = ZPP + pn.to(tl.int64) * (n_ksteps * BKG * BN)
    bn_off = tl.arange(0, BN)
    bkp_off = tl.arange(0, BKP)
    bkg_off = tl.arange(0, BKG)
    if PDL:
        tl.extra.cuda.gdc_wait()
    if FUSE_RNN_BATCH:
        m_req = m % N_REQS
        m_step = m // N_REQS
        m_x = m_req * K_STEPS + m_step
    for k0 in range(k_lo, k_hi, BK):
        ks = k0 // BK
        kk = k0 + tl.arange(0, BK)
        if FUSE_RNN_BATCH:
            if k0 < half_K:
                x = tl.load(X + m_x[:, None] * sx + kk[None, :], mask=mm[:, None], other=0.0).to(tl.float16)
            else:
                x = tl.load(X2 + m_req[:, None] * sx2 + (kk - half_K)[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        elif FUSE_RNN:
            if k0 < half_K:
                x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None], other=0.0).to(tl.float16)
            else:
                x = tl.load(X2 + m[:, None] * sx2 + (kk - half_K)[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        else:
            x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0).to(tl.float16)
        x_lo, x_hi = tl.split(tl.reshape(x, (BM, BKP, 2)))
        if CACHE_W:
            w_raw = tl.load(wp_base + ks * (BN * BKP) + bn_off[:, None] * BKP + bkp_off[None, :])
            sc = tl.load(sp_base + ks * (BKG * BN) + bkg_off[None, :] * BN + bn_off[:, None])
        else:
            w_raw = tl.load(wp_base + ks * (BN * BKP) + bn_off[:, None] * BKP + bkp_off[None, :], eviction_policy="evict_first")
            sc = tl.load(sp_base + ks * (BKG * BN) + bkg_off[None, :] * BN + bn_off[:, None], eviction_policy="evict_first")
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BKG, G_HALF)), (BN, BKP))
        if SYM_I4:
            zp = sc * (-7.5)
        else:
            if CACHE_W:
                zp = tl.load(zp_base + ks * (BKG * BN) + bkg_off[None, :] * BN + bn_off[:, None])
            else:
                zp = tl.load(zp_base + ks * (BKG * BN) + bkg_off[None, :] * BN + bn_off[:, None], eviction_policy="evict_first")
            zp = tl.reshape(tl.broadcast_to(zp[:, :, None], (BN, BKG, G_HALF)), (BN, BKP))
        q_lo, q_hi = _u4_to_f16_pair(w_raw)
        acc = tl.dot(x_lo, tl.trans(q_lo * sc + zp), acc)
        acc = tl.dot(x_hi, tl.trans(q_hi * sc + zp), acc)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if DIRECT:
        if FUSE_SILU:
            gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
            out_v = tl.div_rn(gate, 1.0 + tldevice.exp(-gate)) * up
            nh = pn * (BN // 2) + tl.arange(0, BN // 2)
            tl.store(O + m[:, None] * so_m + nh[None, :], out_v.to(O.dtype.element_ty), mask=mm[:, None] & (nh[None, :] < (N // 2)))
        else:
            if HAS_BIAS:
                acc = acc + tl.load(BIAS + n, mask=nm, other=0.0)[None, :]
            tl.store(O + m[:, None] * so_m + n[None, :], acc.to(O.dtype.element_ty), mask=mm[:, None] & nm[None, :])
    else:
        tl.store(P + ps * M * N + m[:, None] * N + n[None, :], acc, mask=mm[:, None] & nm[None, :])


@torch.no_grad()
def quantize_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16/fp32 w[N, K] -> (w_pack uint8 [N, K//2], ws uint8(e4m3) [N, K//16], row_scale fp32 [N])."""
    N, K = w.shape
    assert K % 16 == 0
    wf = w.float()
    row_amax = wf.abs().amax(dim=1).clamp_min(1e-12)
    row_scale = row_amax / (6.0 * 448.0)
    g = wf.view(N, K // 16, 16)
    g_amax = g.abs().amax(dim=-1)
    g_scale = (g_amax / (6.0 * row_scale[:, None])).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
    eff = (g_scale.float() * row_scale[:, None]).clamp_min(1e-30)[..., None]
    v = g / eff
    sign = (v < 0).to(torch.uint8)
    av = v.abs()
    mag = (
        (av >= 0.25).to(torch.uint8)
        + (av >= 0.75).to(torch.uint8)
        + (av >= 1.25).to(torch.uint8)
        + (av >= 1.75).to(torch.uint8)
        + (av >= 2.50).to(torch.uint8)
        + (av >= 3.50).to(torch.uint8)
        + (av >= 5.00).to(torch.uint8)
    )
    nib = ((sign << 3) | mag).view(N, K // 2, 2)
    w_pack = (nib[:, :, 0] | (nib[:, :, 1] << 4)).contiguous()
    ws = g_scale.view(torch.uint8).contiguous()
    return w_pack, ws, row_scale.contiguous()


@torch.no_grad()
def quantize_uint4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16/fp32 w[N, K] -> (w_pack uint8 [N, K//2], sc fp16 [N, K//32], zp fp16 [N, K//32])."""
    N, K = w.shape
    assert K % 32 == 0
    g = w.float().view(N, K // 32, 32)
    lo = g.amin(dim=-1)
    hi = g.amax(dim=-1)
    sc = ((hi - lo) / 15.0).clamp_min(1e-7).to(torch.float16)
    zp = lo.to(torch.float16)
    q = torch.round((g - zp.float()[..., None]) / sc.float()[..., None]).clamp(0, 15).to(torch.uint8)
    nib = q.view(N, K // 2, 2)
    w_pack = (nib[:, :, 0] | (nib[:, :, 1] << 4)).contiguous()
    return w_pack, sc.contiguous(), zp.contiguous()


_CFG16_U4 = {
    (4608, 4608):  (16, 32, 256, 3, 4, 3),
    (4608, 2304):  (16, 32, 256, 3, 4, 3),
    (1024, 4608):  (16, 32, 256, 6, 4, 3),
    (4608, 768):   (16, 32, 256, 1, 4, 3),
    (256,  4608):  (16, 16, 256, 9, 4, 3),
    (25128, 4608): (16, 32, 256, 1, 4, 3),
}
_NUM_SMS = None


def _config(M, N, K):
    global _NUM_SMS
    if M <= 16:
        c = _CFG16_U4.get((N, K))
        if c is not None:
            return c
        bm, bn, bk, nw, ns = 16, 64, 256, 4, 3
    else:
        bm, bn, bk, nw, ns = (32 if M <= 32 else 64), 64, 256, 4, 3
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    n_tiles = triton.cdiv(N, bn)
    split = max(1, min(triton.cdiv(4 * _NUM_SMS, n_tiles), max(1, K // bk)))
    return bm, bn, bk, split, nw, ns


def supported(x: torch.Tensor, w_pack: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and 0 < x.shape[0] <= 64
        and x.dtype == torch.bfloat16
        and w_pack.dtype == torch.uint8
        and x.stride(1) == 1
        and w_pack.stride(1) == 1
        and x.shape[1] == w_pack.shape[1] * 2
        and x.shape[1] >= 256
        and x.shape[1] % 32 == 0
    )


def skinny_nvfp4_linear(
    x: torch.Tensor, w_pack: torch.Tensor, ws: torch.Tensor, row_scale: torch.Tensor,
    out: torch.Tensor | None = None, cfg=None,
) -> torch.Tensor:
    M, K = x.shape
    N = w_pack.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    bm, bn, bk, split, nw, ns = cfg if cfg is not None else _config(M, N, K)
    ks = triton.cdiv(triton.cdiv(K, split), bk) * bk
    split = triton.cdiv(K, ks)
    direct = split == 1
    part = out if direct else torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    _w4_nvfp4_kernel[(triton.cdiv(N, bn), split)](
        x, w_pack, ws, row_scale, part, out, M, N, K,
        x.stride(0), w_pack.stride(0), ws.stride(0), out.stride(0), ks,
        BM=bm, BN=bn, BK=bk, DIRECT=direct, num_warps=nw, num_stages=ns,
    )
    if not direct:
        mn = M * N
        _reduce_kernel[(triton.cdiv(mn, 1024),)](part, out, mn, out.stride(0), N, S=split, BLOCK=1024, num_warps=4)
    return out


def skinny_uint4_linear(
    x: torch.Tensor, w_pack: torch.Tensor, sc: torch.Tensor, zp: torch.Tensor,
    out: torch.Tensor | None = None, cfg=None,
) -> torch.Tensor:
    M, K = x.shape
    N = w_pack.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    bm, bn, bk, split, nw, ns = cfg if cfg is not None else _config(M, N, K)
    ks = triton.cdiv(triton.cdiv(K, split), bk) * bk
    split = triton.cdiv(K, ks)
    direct = split == 1
    part = out if direct else torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    _w4_uint4_kernel[(triton.cdiv(N, bn), split)](
        x, w_pack, sc, zp, part, out, M, N, K,
        x.stride(0), w_pack.stride(0), sc.stride(0), out.stride(0), ks,
        BM=bm, BN=bn, BK=bk, DIRECT=direct, num_warps=nw, num_stages=ns,
    )
    if not direct:
        mn = M * N
        _reduce_kernel[(triton.cdiv(mn, 1024),)](part, out, mn, out.stride(0), N, S=split, BLOCK=1024, num_warps=4)
    return out


# Cold-cache GB10 SM121a tuned panel configs: (bm, bn, bk, split, warps, stages)
# dec160: Added balanced pruned draft_head shapes (12560, 8192, 4096, 2048) + tuned kv_proj_ctx
# split=3 (48 CTAs = 1 CTA/SM instead of 96 CTAs thrashing L0.qkv_proj under CTXKV_OVERLAP=1!).
_CFG16_U4P32 = {
    (25128, 4608): (16, 64, 256, 1, 4, 4),
    (12560, 4608): (16, 64, 256, 1, 4, 4),
    (8192,  4608): (16, 64, 256, 1, 4, 4),
    (4096,  4608): (16, 32, 256, 1, 4, 4),
    (2048,  4608): (16, 32, 256, 2, 2, 4),
    (4608,  4608): (16, 64, 256, 1, 4, 3),
    (4608,  2304): (16, 32, 256, 1, 2, 4),
    (1024,  4608): (16, 32, 256, 3, 2, 4),
    (4608,   768): (16, 32, 256, 1, 4, 3),
    (256,   4608): (16, 16, 256, 3, 2, 4),
    (768,   9216): (16, 32, 256, 4, 2, 4),
}

# Retuned on spark-one (fixes gate_up (4608,4608) & draft_head ns=4 -> -124.5 us/step vs u4p32!)
_CFG16_U4P64 = {
    (25128, 4608): (16, 64, 256, 1, 4, 4),
    (12560, 4608): (16, 64, 256, 1, 4, 4),
    (8192,  4608): (16, 64, 256, 1, 4, 4),
    (4096,  4608): (16, 32, 256, 1, 4, 4),
    (2048,  4608): (16, 32, 256, 2, 2, 4),
    (4608,  4608): (16, 64, 256, 1, 4, 3),
    (4608,  2304): (16, 64, 256, 1, 4, 3),
    (1024,  4608): (16, 32, 256, 2, 2, 4),
    (4608,   768): (16, 32, 256, 1, 4, 3),
    (256,   4608): (16, 16, 256, 3, 2, 4),
    (768,   9216): (16, 32, 256, 4, 2, 4),
}

_CFG16_I4P32 = {
    (25128, 4608): (16, 64, 256, 1, 4, 4),
    (12560, 4608): (16, 64, 256, 1, 4, 4),
    (8192,  4608): (16, 64, 256, 1, 4, 4),
    (4096,  4608): (16, 32, 256, 1, 4, 4),
    (2048,  4608): (16, 32, 256, 2, 2, 4),
    (4608,  4608): (16, 64, 256, 1, 4, 3),
    (4608,  2304): (16, 64, 256, 1, 4, 3),
    (1024,  4608): (16, 32, 256, 3, 2, 4),
    (4608,   768): (16, 32, 256, 1, 4, 3),
    (256,   4608): (16, 16, 256, 3, 2, 4),
    (768,   9216): (16, 32, 256, 4, 2, 4),
}

_CFG16_I4P64 = {
    (25128, 4608): (16, 64, 256, 1, 4, 4),
    (12560, 4608): (16, 64, 256, 1, 4, 4),
    (8192,  4608): (16, 64, 256, 1, 4, 4),
    (4096,  4608): (16, 32, 256, 1, 4, 4),
    (2048,  4608): (16, 32, 256, 2, 2, 4),
    (4608,  4608): (16, 64, 256, 1, 4, 3),
    (4608,  2304): (16, 64, 256, 1, 4, 3),
    (1024,  4608): (16, 32, 256, 3, 2, 4),
    (4608,   768): (16, 32, 256, 1, 4, 3),
    (256,   4608): (16, 16, 256, 3, 2, 4),
    (768,   9216): (16, 32, 256, 3, 2, 4),
}


@torch.no_grad()
def quantize_u4_panel(
    w: torch.Tensor,
    group_size: int = 32,
    symmetric: bool = False,
    interleave_silu: bool = False,
):
    """bf16/fp32 w[N, K] -> panel-contiguous (wp, scp, zpp_or_None, meta) for u4p32/u4p64/i4p32/i4p64.
    - symmetric=True (i4p32 / i4p64): per-group 5-alpha MSE-clipped symmetric (q - 7.5)*sc, zpp=None!
    - interleave_silu=True (gate_up_proj): interleaves [gate[i], up[i]] as adjacent cols for fused silu epilogue.
    """
    assert group_size in (32, 64)
    N, K = int(w.shape[0]), int(w.shape[1])
    if interleave_silu:
        assert N % 2 == 0
        N2 = N // 2
        w = torch.stack([w[:N2], w[N2:]], dim=1).reshape(N, K).contiguous()
    if symmetric:
        tbl = _CFG16_I4P32 if group_size == 32 else _CFG16_I4P64
    else:
        tbl = _CFG16_U4P32 if group_size == 32 else _CFG16_U4P64
    cfg = tbl.get((N, K))
    if cfg is None:
        bn, bk, split, nw, ns = 64, 256, 1, 4, 3
        if K % bk != 0:
            bk = 128
        assert K % bk == 0
    else:
        _, bn, bk, split, nw, ns = cfg
    if interleave_silu:
        # Fused silu requires DIRECT (split=1) and even BN
        assert split == 1 and bn % 2 == 0
    assert K % bk == 0 and bk % group_size == 0
    n_panels = triton.cdiv(N, bn)
    k_steps = K // bk
    bkp = bk // 2
    bkg = bk // group_size
    wp = torch.empty((n_panels, k_steps, bn, bkp), dtype=torch.uint8, device=w.device)
    scp = torch.empty((n_panels, k_steps, bkg, bn), dtype=torch.float16, device=w.device)
    zpp = None if symmetric else torch.empty((n_panels, k_steps, bkg, bn), dtype=torch.float16, device=w.device)

    p_step = max(1, 4096 // bn)
    for p0 in range(0, n_panels, p_step):
        p1 = min(p0 + p_step, n_panels)
        r0 = p0 * bn
        r1 = min(p1 * bn, N)
        rows = (p1 - p0) * bn
        wf = torch.zeros((rows, K), dtype=torch.float32, device=w.device)
        wf[: r1 - r0].copy_(w[r0:r1].float())
        g = wf.view(rows, K // group_size, group_size)
        if symmetric:
            amax = g.abs().amax(dim=-1).clamp_min(1e-7)
            best_err = torch.full_like(amax, 1e30)
            best_sc = (amax / 7.5).to(torch.float16)
            for alpha in (1.0, 0.94, 0.88, 0.82, 0.76):
                sct = (amax * alpha / 7.5).clamp_min(1e-7).to(torch.float16)
                qt = torch.round(g / sct.float()[..., None] + 7.5).clamp(0, 15)
                err = (g - (qt - 7.5) * sct.float()[..., None]).square().mean(dim=-1)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_sc = torch.where(better, sct, best_sc)
            sc = best_sc
            q = torch.round(g / sc.float()[..., None] + 7.5).clamp(0, 15).to(torch.uint8)
        else:
            lo = g.amin(dim=-1)
            hi = g.amax(dim=-1)
            sc = ((hi - lo) / 15.0).clamp_min(1e-7).to(torch.float16)
            zp = lo.to(torch.float16)
            q = torch.round((g - zp.float()[..., None]) / sc.float()[..., None]).clamp(0, 15).to(torch.uint8)
            zpp[p0:p1].copy_(zp.view(p1 - p0, bn, k_steps, bkg).permute(0, 2, 3, 1))
        nib = q.view(rows, K // 2, 2)
        w_pack = nib[:, :, 0] | (nib[:, :, 1] << 4)
        wp[p0:p1].copy_(w_pack.view(p1 - p0, bn, k_steps, bkp).permute(0, 2, 1, 3))
        scp[p0:p1].copy_(sc.view(p1 - p0, bn, k_steps, bkg).permute(0, 2, 3, 1))
    # meta: 10-tuple (N, K, bn, bk, group_size, split, nw, ns, symmetric, interleave_silu)
    meta = (N, K, bn, bk, group_size, split, nw, ns, symmetric, interleave_silu)
    return wp, scp, zpp, meta


def u4_panel_supported(x: torch.Tensor, meta: tuple, fuse_rnn: bool = False) -> bool:
    if not meta or len(meta) != 10:
        return False
    expected_k = meta[1] // 2 if fuse_rnn else meta[1]
    return (
        x.dim() == 2
        and 0 < x.shape[0] <= _U4_PANEL_MAX_M
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.shape[1] == expected_k
    )


def skinny_u4_panel_linear(
    x: torch.Tensor,
    wp: torch.Tensor,
    scp: torch.Tensor,
    zpp: torch.Tensor | None,
    meta: tuple,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    bias: torch.Tensor | None = None,
    x2: torch.Tensor | None = None,
    pdl: bool | None = None,
    n_reqs: int = 1,
    k_steps: int = 0,
) -> torch.Tensor:
    if len(meta) == 8:
        N, K, bn, bk, g, split, nw, ns = meta
        sym_i4, fuse_silu = False, False
    else:
        N, K, bn, bk, g, split, nw, ns, sym_i4, fuse_silu = meta
    M = x.shape[0]
    out_n = (N // 2) if fuse_silu else N
    if out is None:
        out = torch.empty((M, out_n), dtype=x.dtype if out_dtype is None else out_dtype, device=x.device)
    if M == 0:
        return out
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    has_bias = bias is not None
    fuse_rnn = x2 is not None
    fuse_rnn_batch = bool(fuse_rnn and n_reqs > 1 and k_steps > 0)
    sx2 = 0 if (not fuse_rnn or x2.shape[0] == 1) else x2.stride(0)
    fuse_split = False
    if M <= 32:
        bm = 16 if M <= 16 else 32
        if bm == 32:
            nw = max(nw, 4)
            ns = min(ns, 2)
        ks = triton.cdiv(triton.cdiv(K, split), bk) * bk
        split = triton.cdiv(K, ks)
        direct = split == 1
        if fuse_silu:
            assert direct, "fuse_silu requires split == 1"
        n_panels = triton.cdiv(N, bn)
        sem = out
        if direct:
            part = out
        else:
            scratch = (
                _next_fuse_scratch(x.device, n_panels * split * bm * bn, n_panels)
                if _U4_FUSE_REDUCE
                else None
            )
            if scratch is not None:
                part, sem = scratch
                fuse_split = True
            else:
                part = torch.empty((split, M, N), dtype=torch.float32, device=x.device)
        _u4_panel_kernel[(n_panels, split)](
            x, x2 if fuse_rnn else x, wp, scp, scp if sym_i4 else zpp,
            bias if has_bias else out, part, sem, out, M, N, K,
            x.stride(0), sx2, out.stride(0), ks, int(n_reqs), int(k_steps),
            BM=bm, BN=bn, BK=bk, G=g, DIRECT=direct, HAS_BIAS=has_bias,
            SYM_I4=sym_i4, FUSE_SILU=fuse_silu, FUSE_RNN=fuse_rnn,
            SPLIT=split, FUSE_SPLIT=fuse_split,
            PDL=use_pdl, FUSE_RNN_BATCH=fuse_rnn_batch,
            num_warps=nw, num_stages=ns, launch_pdl=use_pdl,
        )
    else:
        bm = 64 if (_U4_MT_BM64 and M >= 64) else 32
        m_tiles = triton.cdiv(M, bm)
        n_panels = triton.cdiv(N, bn)
        if split > 1 and n_panels * m_tiles >= 48:
            split = 1
        ks = triton.cdiv(triton.cdiv(K, split), bk) * bk
        split = triton.cdiv(K, ks)
        direct = split == 1
        if fuse_silu:
            assert direct, "fuse_silu requires split == 1"
        part = out if direct else torch.empty((split, M, N), dtype=torch.float32, device=x.device)
        _u4_panel_mt_kernel[(m_tiles, n_panels, split)](
            x, x2 if fuse_rnn else x, wp, scp, scp if sym_i4 else zpp,
            bias if has_bias else out, part, out, M, N, K,
            x.stride(0), sx2, out.stride(0), ks, int(n_reqs), int(k_steps),
            BM=bm, BN=bn, BK=bk, G=g, DIRECT=direct, HAS_BIAS=has_bias,
            SYM_I4=sym_i4, FUSE_SILU=fuse_silu, FUSE_RNN=fuse_rnn,
            CACHE_W=(m_tiles > 1), PDL=use_pdl, FUSE_RNN_BATCH=fuse_rnn_batch,
            num_warps=max(nw, 8 if bm == 64 else 4), num_stages=min(ns, 2), launch_pdl=use_pdl,
        )
    if not direct and not fuse_split:
        mn = M * N
        if has_bias:
            _reduce_bias_kernel[(triton.cdiv(mn, 1024),)](
                part, bias, out, mn, out.stride(0), N, S=split, BLOCK=1024,
                PDL=use_pdl, num_warps=4, launch_pdl=use_pdl,
            )
        else:
            _reduce_kernel[(triton.cdiv(mn, 1024),)](
                part, out, mn, out.stride(0), N, S=split, BLOCK=1024,
                PDL=use_pdl, num_warps=4, launch_pdl=use_pdl,
            )
    return out


# ---- dec160: Hadamard-64 + Bitplane-Split NF3 (3.25 / 3.50 bpw) for Drafter MLP ----
_H64_NF3_LIB_PATH = os.environ.get("INKLING_DRAFT_NF3_LIB", "/rdma/libdrafter_h64_nf3_sm121.so")
_H64_NF3_LIB = None
_H64_NF3_CHECKED = False
_H64_MAT_CACHE: dict[int, torch.Tensor] = {}

_NF3_POS = (0.1138916015625, 0.351318359375, 0.62451171875, 1.0)  # exact fp16 normalized Lloyd-Max centroids


def _load_h64_nf3_lib():
    global _H64_NF3_LIB, _H64_NF3_CHECKED
    if _H64_NF3_CHECKED:
        return _H64_NF3_LIB
    _H64_NF3_CHECKED = True
    try:
        import ctypes
        lib = ctypes.CDLL(_H64_NF3_LIB_PATH)
        lib.drafter_h64_nf3_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.drafter_h64_nf3_launch.restype = ctypes.c_int
        _H64_NF3_LIB = lib
    except Exception:
        _H64_NF3_LIB = None
    return _H64_NF3_LIB


def _get_h64(device: torch.device) -> torch.Tensor:
    key = -1 if device.type != "cuda" else _dev_idx(device)
    H = _H64_MAT_CACHE.get(key)
    if H is None:
        h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        while h.shape[0] < 64:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        H = (h / 8.0).contiguous()
        _H64_MAT_CACHE[key] = H
    return H


@torch.no_grad()
def quantize_h64_nf3_panel(
    w: torch.Tensor,
    group_size: int = 64,
    interleave_silu: bool = False,
):
    """bf16/fp32 w[N, K] -> Hadamard-64 rotated, MMA-lane-packed bitplane-split NF3
    (wlo2 [n_panels, K//64, 128, 2] uint32, whi1 [n_panels, K//64, 128] uint32, scp [n_panels, K//G, 64] fp16, meta).
    Exact match to drafter_h64_nf3_m8_kernel in drafter_h64_nf3_sm121.cu!"""
    assert group_size in (32, 64)
    N, K = int(w.shape[0]), int(w.shape[1])
    assert K % 64 == 0
    if interleave_silu:
        assert N % 2 == 0
        N2 = N // 2
        w = torch.stack([w[:N2], w[N2:]], dim=1).reshape(N, K).contiguous()
    bn = 64
    n_panels = triton.cdiv(N, bn)
    k_blocks = K // 64
    H64 = _get_h64(w.device)
    pos_vals = torch.tensor(_NF3_POS, dtype=torch.float32, device=w.device)
    pos_mids = 0.5 * (pos_vals[:-1] + pos_vals[1:])  # 3 midpoints

    wlo2 = torch.empty((n_panels, k_blocks, 128, 2), dtype=torch.uint32, device=w.device)
    whi1 = torch.empty((n_panels, k_blocks, 128), dtype=torch.uint32, device=w.device)
    scp = torch.empty((n_panels, K // group_size, bn), dtype=torch.float16, device=w.device)

    lanes = torch.arange(32, device=w.device)
    g_idx = lanes >> 2    # 0..7
    t_idx = lanes & 3     # 0..3
    k_offs = torch.stack([2 * t_idx, 2 * t_idx + 1, 2 * t_idx + 8, 2 * t_idx + 9], dim=-1)  # [32, 4]

    for p0 in range(n_panels):
        r0 = p0 * bn
        r1 = min(r0 + bn, N)
        wf = torch.zeros((bn, K), dtype=torch.float32, device=w.device)
        wf[: r1 - r0].copy_(w[r0:r1].float())
        # Apply orthonormal H_64 along K in 64-elem blocks
        w_rot = (wf.view(bn, k_blocks, 64) @ H64).view(bn, K)
        # MSE-optimal NF3 quantization per group_size
        gg = w_rot.view(bn, K // group_size, group_size)
        std = gg.std(dim=-1, unbiased=False).clamp_min(1e-7)
        amax = gg.abs().amax(dim=-1).clamp_min(1e-7)
        best_err = torch.full_like(amax, 1e30)
        best_sc = torch.minimum(amax, 2.15 * std).clamp_min(1e-7).half()
        for mult in (1.75, 1.95, 2.15, 2.35, 2.60):
            sct = torch.minimum(amax, mult * std).clamp_min(1e-7).half()
            v = (gg / sct.float()[..., None]).clamp(-1.0, 1.0)
            mag = torch.bucketize(v.abs(), pos_mids)
            sgn = torch.where(v < 0, -1.0, 1.0)
            err = (gg - sgn * pos_vals[mag] * sct.float()[..., None]).square().mean(dim=-1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_sc = torch.where(better, sct, best_sc)
        scp[p0].copy_(best_sc.transpose(0, 1))  # [K//G, 64]
        v = (gg / best_sc.float()[..., None]).clamp(-1.0, 1.0)
        mag_all = torch.bucketize(v.abs(), pos_mids).view(bn, k_blocks, 4, 16).permute(1, 2, 0, 3).to(torch.int64)
        sgn_all = (v < 0).view(bn, k_blocks, 4, 16).permute(1, 2, 0, 3).to(torch.int64)
        # mag_all, sgn_all: [k_blocks, 4, bn=64, 16] -> pack into MMA lane layout [k_blocks, 4 warps, 32 lanes]
        for warp in range(4):
            row0 = warp * 16 + g_idx      # [32]
            row1 = row0 + 8               # [32]
            lo_x = torch.zeros((k_blocks, 32), dtype=torch.int64, device=w.device)
            lo_y = torch.zeros((k_blocks, 32), dtype=torch.int64, device=w.device)
            hi_u = torch.zeros((k_blocks, 32), dtype=torch.int64, device=w.device)
            for s in range(4):
                # Slice [k_blocks, 64, 16] -> fancy index adjacent last two dims -> [k_blocks, 32, 4]
                m0 = mag_all[:, s, row0[:, None], k_offs]
                m1 = mag_all[:, s, row1[:, None], k_offs]
                s0 = sgn_all[:, s, row0[:, None], k_offs]
                s1 = sgn_all[:, s, row1[:, None], k_offs]
                b0 = m0[..., 0] | (m0[..., 1] << 2) | (m0[..., 2] << 4) | (m0[..., 3] << 6)
                b1 = m1[..., 0] | (m1[..., 1] << 2) | (m1[..., 2] << 4) | (m1[..., 3] << 6)
                n0 = s0[..., 0] | (s0[..., 1] << 1) | (s0[..., 2] << 2) | (s0[..., 3] << 3)
                n1 = s1[..., 0] | (s1[..., 1] << 1) | (s1[..., 2] << 2) | (s1[..., 3] << 3)
                lo_x |= (b0 << (s * 8))
                lo_y |= (b1 << (s * 8))
                hi_u |= (n0 << (s * 4)) | (n1 << (16 + s * 4))
            tid_slice = slice(warp * 32, (warp + 1) * 32)
            wlo2[p0, :, tid_slice, 0].copy_(lo_x.to(torch.uint32))
            wlo2[p0, :, tid_slice, 1].copy_(lo_y.to(torch.uint32))
            whi1[p0, :, tid_slice].copy_(hi_u.to(torch.uint32))
    meta = ("h64_nf3", N, K, group_size, interleave_silu)
    return wlo2.contiguous(), whi1.contiguous(), scp.contiguous(), meta


def h64_nf3_supported(x: torch.Tensor, meta: tuple) -> bool:
    return (
        len(meta) == 5
        and meta[0] == "h64_nf3"
        and x.dim() == 2
        and 0 < x.shape[0] <= 8
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.shape[1] == meta[2]
        and _load_h64_nf3_lib() is not None
    )


def skinny_h64_nf3_linear(
    x: torch.Tensor,
    wlo2: torch.Tensor,
    whi1: torch.Tensor,
    scp: torch.Tensor,
    meta: tuple,
    out: torch.Tensor | None = None,
    rotate_x: bool = True,
    pdl: bool | None = None,
) -> torch.Tensor:
    _, N, K, gs, fuse_silu = meta
    M = int(x.shape[0])
    out_n = (N // 2) if fuse_silu else N
    if out is None:
        out = torch.empty((M, out_n), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    lib = _load_h64_nf3_lib()
    assert lib is not None, "libdrafter_h64_nf3_sm121.so not loaded"
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    stream_u64 = int(torch.cuda.current_stream(x.device).cuda_stream)
    rc = lib.drafter_h64_nf3_launch(
        int(x.data_ptr()), int(wlo2.data_ptr()), int(whi1.data_ptr()), int(scp.data_ptr()),
        int(out.data_ptr()), M, N, K, int(x.stride(0)), int(out.stride(0)),
        int(gs), int(rotate_x), int(fuse_silu), int(use_pdl), stream_u64,
    )
    if rc != 0:
        raise RuntimeError(f"drafter_h64_nf3_launch failed with rc={rc}")
    return out
