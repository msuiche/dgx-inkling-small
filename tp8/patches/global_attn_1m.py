# SPDX-License-Identifier: Apache-2.0
"""Phase 2 (1M Context <= 120s) Global Full-Attention Prefill Kernels for Inkling-2-Small on GB10 SM121a.

Exploits the exact mathematical structure of Inkling2's 8 global full-attention layers
(L5, 11, 17, 23, 29, 35, 41, 47; H=6, Hkv=1, G=6, D=128, BS=256, E=1024, EW=1280):
  - For 99.902% of all causal QK tiles at S=1,048,576 (kpos <= qpos - 1024), rel_bias == 0
    and every (q, k) pair is causally valid.
  - 48-SM L2-Lockstep CTA Grid Swizzle: grid = (num_mt * H, num_reqs) with
    ``h = pid % H; mt = pid // H`` packs 6 heads x 8 adjacent Q-tiles = 48 CTAs per wave
    across all 48 GB10 SMs so every SM marches n0 = 0, BN, 2*BN, ... in lockstep, sharing
    K/V tiles 48x in GB10's 24 MB L2 (105.6 TB -> 2.2 TB KV DRAM reads!).
  - 100% int64 KV block indexing (`blk.to(tl.int64)`) & non-negative 64-bit BIAS offsets:
    safe across the full 35,312-block (4.31 GiB) 1M KV slab (fixes blk >= 32,768 overflow).

Supported modes (`INKLING_PF_GLOBAL_MODE`):
  - ``off``           : Disabled (stock sm12_attn._attn_prefill_full_kernel).
  - ``bf16_lockstep`` : Mode 1 — 100% 0-ULP bit-exact BF16 (f32 accum). Bias-free prefix
                        split + 48-SM L2-lockstep grid + coalesced D-contiguous K loads +
                        exact BF16 biased tail (same n0=0,64,... alignment -> nd == 0!).
  - ``bf16_pvskip``   : Mode 2 — Near-exact lossless BF16 + softmax PV-skip on bias-free
                        prefix tiles with max_row(tile_max - m_new) < -TAU (default 24.0,
                        2^-24 = 5.96e-8; skips ~90-95% of far V loads & P@V MMAs, <1e-6 diff).
  - ``fp8_sage2``     : Mode 3 (Triton) — Path 2A Dense <= 120s target. Bias-free prefix runs
                        SM121a FP8 E4M3 QK^T (f16 accum @ 314 TF with Q_MAX=K_MAX=22.0 zero
                        overflow, or f32 accum @ 157 TF) + lossless PV-skip (tau=24) + BF16 or
                        FP8-f16 P@V, handing (m_i, l_i, acc) into the 100% exact BF16 biased tail.
  - ``fp8_sage2_cuda``: Mode 3 (CUDA) — Native SM121a PTX mma.sync.m16n8k32.f16.e4m3.e4m3.f16
                        prefix kernel (`libglobal_attn_sage2_sm121.so`) + exact BF16 biased tail
                        merge kernel (auto-falls back to Triton `fp8_sage2` if .so unavailable).

Hard Invariant: Strictly gated to global prefill chunks (`window < 0`, `qlen >= MIN_QLEN`,
`past >= MIN_PAST` when known). C=1 decode (T <= 16) and local SWA layers are 100% untouched.
"""

from __future__ import annotations

import ctypes
import os

import torch

try:
    from vllm.triton_utils import tl, triton
except Exception:  # Standalone bench / local test without vllm
    import triton
    import triton.language as tl

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# ---- Env configuration (strictly opt-in; C=1 decode 100% untouched) ----
MODE = os.environ.get("INKLING_PF_GLOBAL_MODE", "off").strip().lower()
MIN_QLEN = int(os.environ.get("INKLING_PF_GLOBAL_MIN_QLEN", "1024") or 1024)
MIN_PAST = int(os.environ.get("INKLING_PF_GLOBAL_MIN_PAST", "4096") or 4096)
PV_SKIP_TAU = float(
    os.environ.get("INKLING_PF_GLOBAL_TAU", os.environ.get("INKLING_PF_PVSKIP_TAU", "24.0")) or 24.0
)
GLOBAL_TOPK = int(os.environ.get("INKLING_PF_GLOBAL_TOPK", "0") or 0)  # 0 = 100% DENSE default!
GLOBAL_SINK_BLKS = int(os.environ.get("INKLING_PF_GLOBAL_SINK_BLKS", "4") or 4)
GLOBAL_LOCAL_BLKS = int(os.environ.get("INKLING_PF_GLOBAL_LOCAL_BLKS", "16") or 16)
SAGE2_ACC = os.environ.get("INKLING_PF_SAGE2_ACC", "f16").strip().lower()  # "f16" or "f32"
SAGE2_PV = os.environ.get("INKLING_PF_SAGE2_PV", "bf16").strip().lower()  # "bf16" or "fp8_f16"

_ENV_BM = int(os.environ.get("INKLING_PF_GLOBAL_BM", "0") or 0)
_ENV_BN = int(os.environ.get("INKLING_PF_GLOBAL_BN", "0") or 0)
_ENV_WARPS = int(os.environ.get("INKLING_PF_GLOBAL_WARPS", "0") or 0)
_ENV_STAGES = int(os.environ.get("INKLING_PF_GLOBAL_STAGES", "0") or 0)
_ENV_MIN_CTAS = int(os.environ.get("INKLING_PF_GLOBAL_MIN_CTAS", "2") or 2)
_PF_GLOBAL_ZERO_SYNC = os.environ.get("INKLING_PF_GLOBAL_ZERO_SYNC", "0").strip().lower() in (
    "1", "on", "true", "yes",
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUDA_LIB_ENV = os.environ.get("INKLING_PF_GLOBAL_LIB", "").strip()
_CUDA_LIB_CANDIDATES = (
    _CUDA_LIB_ENV,
    "/rdma/libglobal_attn_sage2_sm121_v5.so",
    os.path.join(_HERE, "libglobal_attn_sage2_sm121_v5.so"),
    "/rdma/libglobal_attn_sage2_sm121_v4.so",
    os.path.join(_HERE, "libglobal_attn_sage2_sm121_v4.so"),
    "/rdma/libglobal_attn_sage2_sm121_v3.so",
    os.path.join(_HERE, "libglobal_attn_sage2_sm121_v3.so"),
    "/rdma/libglobal_attn_sage2_sm121.so",
    os.path.join(_HERE, "libglobal_attn_sage2_sm121.so"),
)
_CUDA_LIB = None
_CUDA_LIB_TRIED = False
_CUDA_LIB_CACHE: dict[str, ctypes.CDLL] = {}
_CUDA_SCRATCH: dict[tuple, tuple] = {}
_TOPK_SCORE_SCRATCH: dict[object, torch.Tensor] = {}


def _get_topk_scores_scratch(device, n_ctas: int, max_pblks: int) -> torch.Tensor:
    """Reusable high-water scratch for Quest exact-Kmax block scores (eliminates 16x/chunk
    6-12 MiB aten::empty allocations inside select_topk_blocks_exact_kmax)."""
    need = n_ctas * max_pblks
    buf = _TOPK_SCORE_SCRATCH.get(device)
    if buf is None or buf.numel() < need:
        cap = max(need, 4096 * 384)
        buf = torch.empty((cap,), dtype=torch.float32, device=device)
        _TOPK_SCORE_SCRATCH[device] = buf
    return buf[:need].view(n_ctas, max_pblks)


def _get_cuda_scratch(device, T: int, H: int, D: int):
    """Cached contiguous scratch (acc, m, l) for CUDA prefix -> Triton tail merge."""
    key = (device, H, D)
    entry = _CUDA_SCRATCH.get(key)
    if entry is None or entry[0].shape[0] < T:
        cap = max(T, 4096)
        acc = torch.empty((cap, H, D), dtype=torch.float32, device=device)
        m = torch.empty((cap, H), dtype=torch.float32, device=device)
        l = torch.empty((cap, H), dtype=torch.float32, device=device)
        entry = (acc, m, l)
        _CUDA_SCRATCH[key] = entry
    return entry[0][:T], entry[1][:T], entry[2][:T]


def get_cuda_lib(lib_path: str | None = None):
    """Lazily load compiled SM121a SageAttention2 prefix library via ctypes (cached by path)."""
    global _CUDA_LIB, _CUDA_LIB_TRIED
    if lib_path is None and _CUDA_LIB_TRIED:
        return _CUDA_LIB
    if lib_path is not None and lib_path in _CUDA_LIB_CACHE:
        return _CUDA_LIB_CACHE[lib_path]
    if lib_path is None:
        _CUDA_LIB_TRIED = True
    candidates = ([lib_path] if lib_path else []) + list(_CUDA_LIB_CANDIDATES)
    found = next((p for p in candidates if p and os.path.exists(p)), None)
    if found is None:
        return None
    if found in _CUDA_LIB_CACHE:
        lib = _CUDA_LIB_CACHE[found]
        if lib_path is None:
            _CUDA_LIB = lib
        return lib
    lib = ctypes.CDLL(found)
    print(f"[inkling-pf-global] loaded {found}", flush=True)
    lib.global_sage2_prefix_sm121_fwd.argtypes = [
        ctypes.c_void_p,  # Q [T, H, D] bf16
        ctypes.c_void_p,  # K [NBLK, BS, Hkv, D] bf16
        ctypes.c_void_p,  # V [NBLK, BS, Hkv, D] bf16
        ctypes.c_void_p,  # BT [R, max_blks] int32
        ctypes.c_void_p,  # SEQ [R] int32
        ctypes.c_void_p,  # CUQ [R+1] int32
        ctypes.c_void_p,  # ACC_OUT [T, H, D] fp32
        ctypes.c_void_p,  # M_OUT [T, H] fp32
        ctypes.c_void_p,  # L_OUT [T, H] fp32
        ctypes.c_int64, ctypes.c_int64,  # sq_t, sq_h
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sk_b, sk_s, sk_h
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sv_b, sv_s, sv_h
        ctypes.c_int64,  # sbt_r
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_reqs, max_seqlen_q, H, G
        ctypes.c_int, ctypes.c_int,  # BS, E
        ctypes.c_float, ctypes.c_float,  # scale_log2, pv_skip_tau
        ctypes.c_uint64,  # stream
    ]
    lib.global_sage2_prefix_sm121_fwd.restype = ctypes.c_int
    if hasattr(lib, "global_sage2_prefix_v2_sm121_fwd"):
        lib.global_sage2_prefix_v2_sm121_fwd.argtypes = [
            ctypes.c_void_p,  # Q [T, H, D] bf16
            ctypes.c_void_p,  # K_fp8 [NBLK, BS, Hkv, D] fp8_e4m3fn (uint8)
            ctypes.c_void_p,  # K_scale [NBLK, Hkv, BS] fp32
            ctypes.c_void_p,  # V [NBLK, BS, Hkv, D] bf16
            ctypes.c_void_p,  # BT [R, max_blks] int32
            ctypes.c_void_p,  # SEQ [R] int32
            ctypes.c_void_p,  # CUQ [R+1] int32
            ctypes.c_void_p,  # ACC_OUT [T, H, D] fp32
            ctypes.c_void_p,  # M_OUT [T, H] fp32
            ctypes.c_void_p,  # L_OUT [T, H] fp32
            ctypes.c_int64, ctypes.c_int64,  # sq_t, sq_h
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sk8_b, sk8_s, sk8_h
            ctypes.c_int64, ctypes.c_int64,  # sks_b, sks_h
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sv_b, sv_s, sv_h
            ctypes.c_int64,  # sbt_r
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_reqs, max_seqlen_q, H, G
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # BS, E, bn (64 or 128)
            ctypes.c_float, ctypes.c_float,  # scale_log2, pv_skip_tau
            ctypes.c_uint64,  # stream
        ]
        lib.global_sage2_prefix_v2_sm121_fwd.restype = ctypes.c_int
    if hasattr(lib, "global_sage2_prefix_v3_sm121_fwd"):
        lib.global_sage2_prefix_v3_sm121_fwd.argtypes = [
            ctypes.c_void_p,  # Q [T, H, D] bf16
            ctypes.c_void_p,  # K_fp8 [NBLK, BS, Hkv, D] fp8_e4m3fn (uint8)
            ctypes.c_void_p,  # KS64 [NBLK, Hkv, BS//64] fp32
            ctypes.c_void_p,  # V [NBLK, BS, Hkv, D] bf16
            ctypes.c_void_p,  # BT [R, max_blks] int32
            ctypes.c_void_p,  # SEQ [R] int32
            ctypes.c_void_p,  # CUQ [R+1] int32
            ctypes.c_void_p,  # SEL_BLOCKS [R*num_mt*H, sel_stride] int32 (or 0 for dense)
            ctypes.c_void_p,  # SEL_COUNTS [R*num_mt*H] int32 (or 0 for dense)
            ctypes.c_void_p,  # ACC_OUT [T, H, D] fp32
            ctypes.c_void_p,  # M_OUT [T, H] fp32
            ctypes.c_void_p,  # L_OUT [T, H] fp32
            ctypes.c_int64, ctypes.c_int64,  # sq_t, sq_h
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sk8_b, sk8_s, sk8_h
            ctypes.c_int64, ctypes.c_int64,  # sks64_b, sks64_h
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sv_b, sv_s, sv_h
            ctypes.c_int64,  # sbt_r
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_reqs, max_seqlen_q, H, G
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # BS, E, bn, min_ctas, sel_stride
            ctypes.c_float, ctypes.c_float,  # scale_log2, pv_skip_tau
            ctypes.c_uint64,  # stream
        ]
        lib.global_sage2_prefix_v3_sm121_fwd.restype = ctypes.c_int
    if hasattr(lib, "global_sage2_prefix_v5_sm121_fwd"):
        lib.global_sage2_prefix_v5_sm121_fwd.argtypes = [
            ctypes.c_void_p,  # Q [T, H, D] bf16
            ctypes.c_void_p,  # K_fp8 [CAP, BS, Hkv, D] fp8_e4m3fn (uint8)
            ctypes.c_void_p,  # KS64 [CAP, Hkv, BS//64] fp32
            ctypes.c_void_p,  # V [NBLK, BS, Hkv, D] bf16
            ctypes.c_void_p,  # BT [R, max_blks] int32 (V block table)
            ctypes.c_void_p,  # SEQ [R] int32
            ctypes.c_void_p,  # CUQ [R+1] int32
            ctypes.c_void_p,  # SEL_BLOCKS [R*num_mt*H, sel_stride] int32 (or 0 for dense)
            ctypes.c_void_p,  # SEL_COUNTS [R*num_mt*H] int32 (or 0 for dense)
            ctypes.c_void_p,  # ACC_OUT [T, H, D] fp32
            ctypes.c_void_p,  # M_OUT [T, H] fp32
            ctypes.c_void_p,  # L_OUT [T, H] fp32
            ctypes.c_int64, ctypes.c_int64,  # sq_t, sq_h
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sk8_b, sk8_s, sk8_h
            ctypes.c_int64, ctypes.c_int64,  # sks64_b, sks64_h
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # sv_b, sv_s, sv_h
            ctypes.c_int64,  # sbt_r
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_reqs, max_seqlen_q, H, G
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # BS, E, bn, min_ctas, sel_stride
            ctypes.c_float, ctypes.c_float,  # scale_log2, pv_skip_tau
            ctypes.c_void_p,  # K_BT [R, max_kblks] int32 (used when k_bt_mode == 2)
            ctypes.c_int64,   # skbt_r
            ctypes.c_int,     # k_bt_mode (0=shared BT, 1=direct logical idx, 2=separate K_BT)
            ctypes.c_uint64,  # stream
        ]
        lib.global_sage2_prefix_v5_sm121_fwd.restype = ctypes.c_int
    if hasattr(lib, "score_kmax_blocks_fp8_sm121_fwd"):
        lib.score_kmax_blocks_fp8_sm121_fwd.argtypes = [
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K_MID
            ctypes.c_void_p,  # K_RAD
            ctypes.c_void_p,  # BT
            ctypes.c_void_p,  # SEQ
            ctypes.c_void_p,  # CUQ
            ctypes.c_void_p,  # SCORES
            ctypes.c_void_p,  # SEL_COUNTS
            ctypes.c_int64, ctypes.c_int64,  # sq_t, sq_h
            ctypes.c_int64, ctypes.c_int64,  # sm_b, sm_h
            ctypes.c_int64,  # sbt_r
            ctypes.c_int64,  # ssc_cta
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_reqs, max_seqlen_q, max_pblks
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # H, G, BS, E
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # sink_blocks, local_blocks, k_sel
            ctypes.c_uint64,  # stream
        ]
        lib.score_kmax_blocks_fp8_sm121_fwd.restype = ctypes.c_int
    _CUDA_LIB_CACHE[found] = lib
    if lib_path:
        _CUDA_LIB_CACHE[lib_path] = lib
    if lib_path is None:
        _CUDA_LIB = lib
    return lib


# ==================================================================================================
# Pre-Quantized FP8 Shadow K Cache Builders (1.032 GiB total across all 8 global layers at S=1M!)
# ==================================================================================================
@triton.jit
def _quantize_k_blocks_fp8_kernel(
    K_IN, K_FP8_OUT, K_SCALE_OUT, BLK_IDS,
    sk_b, sk_s, sk_h,
    sk8_b, sk8_s, sk8_h,
    sks_b, sks_h,
    NUM_BLKS,
    HAS_IDS: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Quantize paged BF16 K blocks -> FP8 E4M3 (scaled to [-22.0, +22.0] for zero FP16-accum
    overflow in SM121a mma.sync.m16n8k32.f16.e4m3.e4m3.f16) + per-token FP32 scale."""
    pid = tl.program_id(0)
    kvh = tl.program_id(1)
    chunks_per_blk = BS // BLOCK_S
    b_idx = pid // chunks_per_blk
    s_chunk = pid % chunks_per_blk
    if b_idx >= NUM_BLKS:
        return
    blk = tl.load(BLK_IDS + b_idx).to(tl.int64) if HAS_IDS else b_idx.to(tl.int64)
    ss = s_chunk * BLOCK_S + tl.arange(0, BLOCK_S)
    dd = tl.arange(0, D)

    k = tl.load(K_IN + blk * sk_b + ss[:, None] * sk_s + kvh * sk_h + dd[None, :]).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(k), 1), 1.0e-12)
    k_fp8 = (k * (22.0 / amax)[:, None]).to(tl.float8e4nv)
    k_scale = amax * (1.0 / 22.0)

    tl.store(K_FP8_OUT + blk * sk8_b + ss[:, None] * sk8_s + kvh * sk8_h + dd[None, :], k_fp8)
    tl.store(K_SCALE_OUT + blk * sks_b + kvh * sks_h + ss, k_scale)


@triton.jit
def _quantize_k_slots_fp8_kernel(
    K_IN, K_FP8_OUT, K_SCALE_OUT, SLOTS,
    sk_b, sk_s, sk_h,
    sk8_b, sk8_s, sk8_h,
    sks_b, sks_h,
    N_SLOTS,
    BS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Quantize newly appended K tokens given vLLM `slot_mapping` (< 0.015 ms per 4K chunk!)."""
    pid = tl.program_id(0)
    kvh = tl.program_id(1)
    ti = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_ok = ti < N_SLOTS
    slot_flat = tl.load(SLOTS + ti, mask=t_ok, other=-1).to(tl.int64)
    valid = t_ok & (slot_flat >= 0)
    blk = tl.where(valid, slot_flat // BS, 0)
    slot = tl.where(valid, slot_flat % BS, 0)
    dd = tl.arange(0, D)

    k = tl.load(
        K_IN + blk[:, None] * sk_b + slot[:, None] * sk_s + kvh * sk_h + dd[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(k), 1), 1.0e-12)
    k_fp8 = (k * (22.0 / amax)[:, None]).to(tl.float8e4nv)
    k_scale = amax * (1.0 / 22.0)

    tl.store(
        K_FP8_OUT + blk[:, None] * sk8_b + slot[:, None] * sk8_s + kvh * sk8_h + dd[None, :],
        k_fp8,
        mask=valid[:, None],
    )
    tl.store(K_SCALE_OUT + blk * sks_b + kvh * sks_h + slot, k_scale, mask=valid)


def alloc_fp8_shadow_k(key_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate matching shadow FP8 E4M3 K cache & per-token FP32 scale buffer.
    Layout: k_fp8 [NBLK, BS, Hkv, D] float8_e4m3fn, k_scale [NBLK, Hkv, BS] float32.
    Memory at 1M tokens (4096 blocks x 256 x 1 x 128): 128 MiB + 4 MiB = 132 MiB per global layer!"""
    nblk, bs, hkv, d = key_cache.shape
    k_fp8 = torch.empty((nblk, bs, hkv, d), dtype=torch.float8_e4m3fn, device=key_cache.device)
    k_scale = torch.empty((nblk, hkv, bs), dtype=torch.float32, device=key_cache.device)
    return k_fp8, k_scale


def quantize_k_to_fp8_blocks(
    key_cache: torch.Tensor,
    k_fp8_cache: torch.Tensor | None = None,
    k_scale_cache: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize paged BF16 `key_cache` blocks into pre-quantized FP8 shadow K + per-token scale.
    If `block_ids` (1D int32/int64) is provided, quantizes ONLY those blocks (e.g., 16 new blocks
    for a 4K chunk in ~15 us!)."""
    if k_fp8_cache is None or k_scale_cache is None:
        k_fp8_cache, k_scale_cache = alloc_fp8_shadow_k(key_cache)
    nblk, bs, hkv, d = key_cache.shape
    has_ids = block_ids is not None
    if has_ids:
        ids = block_ids.reshape(-1).contiguous()
        num_blks = int(ids.numel())
    else:
        ids = key_cache  # unused dummy ptr
        num_blks = nblk
    if num_blks == 0:
        return k_fp8_cache, k_scale_cache
    block_s = 64 if bs >= 64 and bs % 64 == 0 else bs
    grid = (num_blks * (bs // block_s), hkv)
    _quantize_k_blocks_fp8_kernel[grid](
        key_cache, k_fp8_cache, k_scale_cache, ids,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1),
        num_blks,
        HAS_IDS=has_ids, BS=bs, D=d, BLOCK_S=block_s,
        num_warps=4, num_stages=2,
    )
    return k_fp8_cache, k_scale_cache


def quantize_k_to_fp8_slots(
    key_cache: torch.Tensor,
    k_fp8_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Quantize ONLY the newly appended slots in `slot_mapping` into shadow FP8 K cache."""
    slots = slot_mapping.reshape(-1)
    n_slots = int(slots.numel())
    if n_slots == 0:
        return
    _, bs, hkv, d = key_cache.shape
    block_t = 64
    grid = (triton.cdiv(n_slots, block_t), hkv)
    _quantize_k_slots_fp8_kernel[grid](
        key_cache, k_fp8_cache, k_scale_cache, slots,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1),
        n_slots,
        BS=bs, D=d, BLOCK_T=block_t,
        num_warps=4, num_stages=2,
    )


# ==================================================================================================
# v3 Per-64-Token FP8 Shadow K Builder (1.0005 GiB total @ 1M!) + Exact-Kmax Quest Bounding Box
# ==================================================================================================
@triton.jit
def _quantize_k_blocks_fp8_v3_kernel(
    K_IN, K_FP8_OUT, KS64_OUT, K_MID_OUT, K_RAD_OUT, BLK_IDS,
    sk_b, sk_s, sk_h,
    sk8_b, sk8_s, sk8_h,
    sks_b, sks_h,
    smid_b, smid_h,
    NUM_BLKS, OUT_OFFSET,
    HAS_IDS: tl.constexpr,
    COMPUTE_KMAX: tl.constexpr,
    COMPACT_OUT: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
):
    """Single-pass per-64-token FP8 E4M3 quantization ([-22, +22] zero FP16-accum overflow)
    plus optional exact elementwise block bounding box (k_mid, k_rad) for Quest exact-Kmax Top-K.
    When `COMPACT_OUT=True`, writes to logical compact slot `OUT_OFFSET + b_idx` instead of `blk`."""
    b_idx = tl.program_id(0)
    kvh = tl.program_id(1)
    if b_idx >= NUM_BLKS:
        return
    blk = tl.load(BLK_IDS + b_idx).to(tl.int64) if HAS_IDS else b_idx.to(tl.int64)
    out_blk = (OUT_OFFSET + b_idx).to(tl.int64) if COMPACT_OUT else blk
    dd = tl.arange(0, D)
    s_off = tl.arange(0, 64)

    if COMPUTE_KMAX:
        b_min = tl.full([D], float("inf"), tl.float32)
        b_max = tl.full([D], float("-inf"), tl.float32)

    for g in range(0, BS // 64):
        ss = g * 64 + s_off
        k = tl.load(K_IN + blk * sk_b + ss[:, None] * sk_s + kvh * sk_h + dd[None, :]).to(tl.float32)
        amax64 = tl.maximum(tl.max(tl.abs(k)), 1.0e-12)
        k_fp8 = (k * (22.0 / amax64)).to(tl.float8e4nv)
        tl.store(K_FP8_OUT + out_blk * sk8_b + ss[:, None] * sk8_s + kvh * sk8_h + dd[None, :], k_fp8)
        tl.store(KS64_OUT + out_blk * sks_b + kvh * sks_h + g, amax64 * (1.0 / 22.0))
        if COMPUTE_KMAX:
            b_min = tl.minimum(b_min, tl.min(k, axis=0))
            b_max = tl.maximum(b_max, tl.max(k, axis=0))

    if COMPUTE_KMAX:
        k_mid = (0.5 * (b_max + b_min)).to(tl.bfloat16)
        k_rad = (0.5 * (b_max - b_min)).to(tl.bfloat16)
        tl.store(K_MID_OUT + out_blk * smid_b + kvh * smid_h + dd, k_mid)
        tl.store(K_RAD_OUT + out_blk * smid_b + kvh * smid_h + dd, k_rad)


def alloc_fp8_shadow_k_v3(
    key_cache: torch.Tensor,
    with_kmax: bool = False,
    cap_blks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Allocate v3 FP8 shadow K + per-64-token FP32 scale (+ optional exact-Kmax bounding box).
    When `cap_blks` is provided (Compact Shadow K, e.g. 4096), allocates constant 1.016 GiB total
    across all 8 global layers regardless of total KV pool size (`nblk`)!"""
    nblk, bs, hkv, d = key_cache.shape
    nblk_alloc = min(nblk, int(cap_blks)) if (cap_blks is not None and cap_blks > 0) else nblk
    assert bs % 64 == 0, f"v3 requires BS % 64 == 0, got {bs}"
    k_fp8 = torch.empty((nblk_alloc, bs, hkv, d), dtype=torch.float8_e4m3fn, device=key_cache.device)
    k_scale64 = torch.empty((nblk_alloc, hkv, bs // 64), dtype=torch.float32, device=key_cache.device)
    if with_kmax:
        k_mid = torch.empty((nblk_alloc, hkv, d), dtype=torch.bfloat16, device=key_cache.device)
        k_rad = torch.empty((nblk_alloc, hkv, d), dtype=torch.bfloat16, device=key_cache.device)
    else:
        k_mid = k_rad = None
    return k_fp8, k_scale64, k_mid, k_rad


def quantize_k_to_fp8_blocks_v3(
    key_cache: torch.Tensor,
    k_fp8_cache: torch.Tensor | None = None,
    k_scale64_cache: torch.Tensor | None = None,
    k_mid_cache: torch.Tensor | None = None,
    k_rad_cache: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
    with_kmax: bool = False,
    out_offset: int = 0,
    compact_out: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Quantize paged BF16 `key_cache` blocks for v3 (per-64-token scale + optional Quest k_mid/k_rad)."""
    want_kmax = with_kmax or (k_mid_cache is not None and k_rad_cache is not None)
    if k_fp8_cache is None or k_scale64_cache is None:
        k_fp8_cache, k_scale64_cache, km_new, kr_new = alloc_fp8_shadow_k_v3(key_cache, with_kmax=want_kmax)
        if k_mid_cache is None:
            k_mid_cache = km_new
        if k_rad_cache is None:
            k_rad_cache = kr_new
    nblk, bs, hkv, d = key_cache.shape
    has_ids = block_ids is not None
    ids = block_ids.reshape(-1).contiguous() if has_ids else key_cache
    num_blks = int(ids.numel()) if has_ids else nblk
    if num_blks == 0:
        return k_fp8_cache, k_scale64_cache, k_mid_cache, k_rad_cache
    dummy_m = k_mid_cache if want_kmax else key_cache
    dummy_r = k_rad_cache if want_kmax else key_cache
    smid_b = k_mid_cache.stride(0) if want_kmax else 0
    smid_h = k_mid_cache.stride(1) if want_kmax else 0
    _quantize_k_blocks_fp8_v3_kernel[(num_blks, hkv)](
        key_cache, k_fp8_cache, k_scale64_cache, dummy_m, dummy_r, ids,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
        k_scale64_cache.stride(0), k_scale64_cache.stride(1),
        smid_b, smid_h,
        num_blks, int(out_offset),
        HAS_IDS=has_ids, COMPUTE_KMAX=want_kmax, COMPACT_OUT=bool(compact_out), BS=bs, D=d,
        num_warps=4, num_stages=2,
    )
    return k_fp8_cache, k_scale64_cache, k_mid_cache, k_rad_cache


# ==================================================================================================
# Sync-Free GPU Shadow K Quantizers (`INKLING_PF_GLOBAL_ZERO_SYNC=1`, ZERO DtoH sync!)
# ==================================================================================================
@triton.jit
def _update_bt_state_kernel(BT_ROW, BT_STATE, END_IDX):
    """Record `[BT_ROW[0], BT_ROW[END_IDX]]` in a 2-int32 GPU state tensor (0 host sync)."""
    tl.store(BT_STATE, tl.load(BT_ROW))
    tl.store(BT_STATE + 1, tl.load(BT_ROW + END_IDX))


@triton.jit
def _quantize_k_blocks_fp8_v3_syncfree_kernel(
    K_IN, K_FP8_OUT, KS64_OUT, K_MID_OUT, K_RAD_OUT, BT_ROW, BT_STATE,
    sk_b, sk_s, sk_h,
    sk8_b, sk8_s, sk8_h,
    sks_b, sks_h,
    smid_b, smid_h,
    NUM_BLKS, START_BLK, PREV_IDX,
    CHECK_CONTIG: tl.constexpr,
    COMPUTE_KMAX: tl.constexpr,
    COMPACT_OUT: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
):
    """Sync-free v3 shadow K quantizer: checks `[BT_ROW[0], BT_ROW[PREV_IDX]] == BT_STATE`
    entirely on GPU. Contiguous chunks early-exit CTAs `< START_BLK` in < 3 us; any new
    request or evicted block automatically re-quantizes all `0..NUM_BLKS-1` (0 stale blocks).
    When `COMPACT_OUT=True`, writes to logical compact slot `b_idx` (constant 1.016 GiB!)."""
    b_idx = tl.program_id(0)
    kvh = tl.program_id(1)
    if b_idx >= NUM_BLKS:
        return
    if CHECK_CONTIG:
        if b_idx < START_BLK:
            if tl.load(BT_ROW) == tl.load(BT_STATE):
                if tl.load(BT_ROW + PREV_IDX) == tl.load(BT_STATE + 1):
                    return
    blk = tl.load(BT_ROW + b_idx).to(tl.int64)
    out_blk = b_idx.to(tl.int64) if COMPACT_OUT else blk
    dd = tl.arange(0, D)
    s_off = tl.arange(0, 64)

    if COMPUTE_KMAX:
        b_min = tl.full([D], float("inf"), tl.float32)
        b_max = tl.full([D], float("-inf"), tl.float32)

    for g in range(0, BS // 64):
        ss = g * 64 + s_off
        k = tl.load(K_IN + blk * sk_b + ss[:, None] * sk_s + kvh * sk_h + dd[None, :]).to(tl.float32)
        amax64 = tl.maximum(tl.max(tl.abs(k)), 1.0e-12)
        k_fp8 = (k * (22.0 / amax64)).to(tl.float8e4nv)
        tl.store(K_FP8_OUT + out_blk * sk8_b + ss[:, None] * sk8_s + kvh * sk8_h + dd[None, :], k_fp8)
        tl.store(KS64_OUT + out_blk * sks_b + kvh * sks_h + g, amax64 * (1.0 / 22.0))
        if COMPUTE_KMAX:
            b_min = tl.minimum(b_min, tl.min(k, axis=0))
            b_max = tl.maximum(b_max, tl.max(k, axis=0))

    if COMPUTE_KMAX:
        k_mid = (0.5 * (b_max + b_min)).to(tl.bfloat16)
        k_rad = (0.5 * (b_max - b_min)).to(tl.bfloat16)
        tl.store(K_MID_OUT + out_blk * smid_b + kvh * smid_h + dd, k_mid)
        tl.store(K_RAD_OUT + out_blk * smid_b + kvh * smid_h + dd, k_rad)


@triton.jit
def _quantize_k_blocks_fp8_v2_syncfree_kernel(
    K_IN, K_FP8_OUT, K_SCALE_OUT, BT_ROW, BT_STATE,
    sk_b, sk_s, sk_h,
    sk8_b, sk8_s, sk8_h,
    sks_b, sks_h,
    NUM_BLKS, START_BLK, PREV_IDX,
    CHECK_CONTIG: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Sync-free v2 shadow K quantizer (per-token FP32 scale, 0 DtoH sync, 0 stale blocks)."""
    pid = tl.program_id(0)
    kvh = tl.program_id(1)
    chunks_per_blk = BS // BLOCK_S
    b_idx = pid // chunks_per_blk
    s_chunk = pid % chunks_per_blk
    if b_idx >= NUM_BLKS:
        return
    if CHECK_CONTIG:
        if b_idx < START_BLK:
            if tl.load(BT_ROW) == tl.load(BT_STATE):
                if tl.load(BT_ROW + PREV_IDX) == tl.load(BT_STATE + 1):
                    return
    blk = tl.load(BT_ROW + b_idx).to(tl.int64)
    ss = s_chunk * BLOCK_S + tl.arange(0, BLOCK_S)
    dd = tl.arange(0, D)

    k = tl.load(K_IN + blk * sk_b + ss[:, None] * sk_s + kvh * sk_h + dd[None, :]).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(k), 1), 1.0e-12)
    k_fp8 = (k * (22.0 / amax)[:, None]).to(tl.float8e4nv)
    k_scale = amax * (1.0 / 22.0)

    tl.store(K_FP8_OUT + blk * sk8_b + ss[:, None] * sk8_s + kvh * sk8_h + dd[None, :], k_fp8)
    tl.store(K_SCALE_OUT + blk * sks_b + kvh * sks_h + ss, k_scale)


def quantize_k_to_fp8_blocks_v3_syncfree(
    key_cache: torch.Tensor,
    k_fp8_cache: torch.Tensor,
    k_scale64_cache: torch.Tensor,
    k_mid_cache: torch.Tensor | None,
    k_rad_cache: torch.Tensor | None,
    bt_row: torch.Tensor,
    bt_state: torch.Tensor,
    nblks: int,
    start_blk: int,
    prev_idx: int,
    check_contig: bool,
    with_kmax: bool = False,
    compact_out: bool = False,
) -> None:
    """Zero-sync GPU shadow K update for v3 (updates `bt_state` on GPU after quantization)."""
    if nblks <= 0:
        return
    _, bs, hkv, d = key_cache.shape
    want_kmax = with_kmax and (k_mid_cache is not None) and (k_rad_cache is not None)
    dummy_m = k_mid_cache if want_kmax else key_cache
    dummy_r = k_rad_cache if want_kmax else key_cache
    smid_b = k_mid_cache.stride(0) if want_kmax else 0
    smid_h = k_mid_cache.stride(1) if want_kmax else 0
    _quantize_k_blocks_fp8_v3_syncfree_kernel[(nblks, hkv)](
        key_cache, k_fp8_cache, k_scale64_cache, dummy_m, dummy_r, bt_row, bt_state,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
        k_scale64_cache.stride(0), k_scale64_cache.stride(1),
        smid_b, smid_h,
        nblks, start_blk, prev_idx,
        CHECK_CONTIG=bool(check_contig), COMPUTE_KMAX=want_kmax, COMPACT_OUT=bool(compact_out),
        BS=bs, D=d,
        num_warps=4, num_stages=2,
    )
    _update_bt_state_kernel[(1,)](bt_row, bt_state, nblks - 1, num_warps=1)


def quantize_k_to_fp8_blocks_v2_syncfree(
    key_cache: torch.Tensor,
    k_fp8_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    bt_row: torch.Tensor,
    bt_state: torch.Tensor,
    nblks: int,
    start_blk: int,
    prev_idx: int,
    check_contig: bool,
) -> None:
    """Zero-sync GPU shadow K update for v2 (updates `bt_state` on GPU after quantization)."""
    if nblks <= 0:
        return
    _, bs, hkv, d = key_cache.shape
    block_s = 64 if bs >= 64 and bs % 64 == 0 else bs
    grid = (nblks * (bs // block_s), hkv)
    _quantize_k_blocks_fp8_v2_syncfree_kernel[grid](
        key_cache, k_fp8_cache, k_scale_cache, bt_row, bt_state,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1),
        nblks, start_blk, prev_idx,
        CHECK_CONTIG=bool(check_contig), BS=bs, D=d, BLOCK_S=block_s,
        num_warps=4, num_stages=2,
    )
    _update_bt_state_kernel[(1,)](bt_row, bt_state, nblks - 1, num_warps=1)


@triton.jit
def _score_kmax_blocks_kernel(
    Q, K_MID, K_RAD, BT, SEQ, CUQ, SCORES, SEL_COUNTS,
    sq_t, sq_h,
    sm_b, sm_h,
    sbt_r,
    ssc_cta,
    MAX_PBLKS,
    SINK_BLKS, LOCAL_BLKS, K_SEL,
    H: tl.constexpr, G: tl.constexpr, BS: tl.constexpr, E: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """Optimized FP8-f16 Quest upper-bound block scorer with coalesced [BLOCK_B, D] loads,
    fused sink/local +1e30 boost, >= n_pblks -1e30 mask, and SEL_COUNTS write (3 launches total!)."""
    pid = tl.program_id(0)
    r = tl.program_id(1)
    h = pid % H
    mt = pid // H
    kvh = h // G
    cta_idx = r * tl.num_programs(0) + pid

    q0 = tl.load(CUQ + r)
    qlen = tl.load(CUQ + r + 1) - q0
    seqlen = tl.load(SEQ + r)
    past = seqlen - qlen

    i_lo = mt * BLOCK_M
    bb = tl.arange(0, BLOCK_B)
    sc_base = SCORES + cta_idx.to(tl.int64) * ssc_cta

    if i_lo >= qlen:
        tl.store(SEL_COUNTS + cta_idx, 0)
        for b0 in range(0, MAX_PBLKS, BLOCK_B):
            lb = b0 + bb
            tl.store(sc_base + lb, tl.full([BLOCK_B], -1.0e30, tl.float32), mask=(lb < MAX_PBLKS))
        return

    q_min = past + i_lo
    k_bias_raw = tl.maximum(q_min - E + 1, 0) if E > 0 else tl.maximum(q_min, 0)
    n_pblks = tl.minimum(k_bias_raw // BS, MAX_PBLKS)
    tl.store(SEL_COUNTS + cta_idx, tl.minimum(n_pblks, K_SEL))

    i_loc = i_lo + tl.arange(0, BLOCK_M)
    row_ok = i_loc < qlen
    tok = q0 + i_loc
    dd = tl.arange(0, D)

    # Quantize Q and |Q| once into FP8 E4M3 ([-22, 22] -> zero FP16 overflow, 4x TC throughput!)
    q_f32 = tl.load(
        Q + tok.to(tl.int64)[:, None] * sq_t + h * sq_h + dd[None, :],
        mask=row_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    q_abs_f32 = tl.abs(q_f32)
    q_amax = tl.maximum(tl.max(q_abs_f32, 1), 1.0e-12)
    q_inv = (22.0 / q_amax)[:, None]
    q_fp8 = (q_f32 * q_inv).to(tl.float8e4nv)
    q_abs_fp8 = (q_abs_f32 * q_inv).to(tl.float8e4nv)
    q_sc = q_amax * (1.0 / 22.0)

    bt_row = BT + r.to(tl.int64) * sbt_r
    local_start = n_pblks - LOCAL_BLKS

    for b0 in range(0, MAX_PBLKS, BLOCK_B):
        lb = b0 + bb
        b_in = lb < MAX_PBLKS
        if b0 >= n_pblks:
            tl.store(sc_base + lb, tl.full([BLOCK_B], -1.0e30, tl.float32), mask=b_in)
        elif (tl.minimum(b0 + BLOCK_B, n_pblks) <= SINK_BLKS) or (b0 >= local_start):
            blk_sc = tl.where(lb < n_pblks, 1.0e30, -1.0e30)
            tl.store(sc_base + lb, blk_sc, mask=b_in)
        else:
            b_ok = lb < n_pblks
            pblk = tl.load(bt_row + lb, mask=b_ok, other=0).to(tl.int64)
            # Coalesced [BLOCK_B, D] loads (fixes strided [D, BLOCK_B] gather!)
            km = tl.load(
                K_MID + pblk[:, None] * sm_b + kvh * sm_h + dd[None, :],
                mask=b_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            kr = tl.load(
                K_RAD + pblk[:, None] * sm_b + kvh * sm_h + dd[None, :],
                mask=b_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            km_amax = tl.maximum(tl.max(tl.abs(km), 1), 1.0e-12)
            kr_amax = tl.maximum(tl.max(tl.abs(kr), 1), 1.0e-12)
            km_fp8 = (km * (22.0 / km_amax)[:, None]).to(tl.float8e4nv)
            kr_fp8 = (kr * (22.0 / kr_amax)[:, None]).to(tl.float8e4nv)
            km_sc = km_amax * (1.0 / 22.0)
            kr_sc = kr_amax * (1.0 / 22.0)

            s_mid = tl.dot(q_fp8, tl.trans(km_fp8), out_dtype=tl.float16).to(tl.float32) * km_sc[None, :]
            s_rad = tl.dot(q_abs_fp8, tl.trans(kr_fp8), out_dtype=tl.float16).to(tl.float32) * kr_sc[None, :]
            ub = (s_mid + s_rad) * q_sc[:, None]
            ub = tl.where(row_ok[:, None], ub, -1.0e30)
            blk_sc = tl.max(ub, axis=0)
            is_mand = b_ok & ((lb < SINK_BLKS) | (lb >= local_start))
            blk_sc = tl.where(is_mand, 1.0e30, tl.where(b_ok, blk_sc, -1.0e30))
            tl.store(sc_base + lb, blk_sc, mask=b_in)


def select_topk_blocks_exact_kmax(
    q: torch.Tensor,
    k_mid: torch.Tensor,
    k_rad: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    E: int,
    BS: int,
    H: int,
    G: int,
    D: int,
    num_reqs: int,
    topk_tokens: int,
    sink_blocks: int | None = None,
    local_blocks: int | None = None,
    max_seq: int | None = None,
    cuda_lib=None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Opt-in 2-level exact-Kmax block selector (guaranteed needle-safe Quest upper bound).
    Uses native SM121a FP8-f16 CUDA scorer (`score_kmax_blocks_fp8_sm121_fwd`) when available,
    else upgraded coalesced FP8-f16 Triton scorer. Only 3 GPU launches (score -> topk -> sort)
    and ZERO host sync when `max_seq` is provided!"""
    s_blks = GLOBAL_SINK_BLKS if sink_blocks is None else int(sink_blocks)
    l_blks = GLOBAL_LOCAL_BLKS if local_blocks is None else int(local_blocks)
    if max_seq is None:
        max_seq = int(cache_seqlens[:num_reqs].max().item()) if num_reqs > 0 else 0
    # Prefix blocks NEVER include the final E-token biased tail!
    max_prefix_toks = max(max_seq - E, 0) if E > 0 else max_seq
    max_pblks = min(int(block_table.shape[1]), max_prefix_toks // BS)
    topk_blks = max(1, topk_tokens // BS)
    budget = s_blks + l_blks + topk_blks
    if max_pblks <= budget:
        return None, None

    num_mt = triton.cdiv(max_seqlen_q, 128)
    n_ctas = num_reqs * num_mt * H
    k_sel = min(budget, max_pblks)
    if _PF_GLOBAL_ZERO_SYNC and not torch.cuda.is_current_stream_capturing():
        scores = _get_topk_scores_scratch(q.device, n_ctas, max_pblks)
    else:
        scores = torch.empty((n_ctas, max_pblks), dtype=torch.float32, device=q.device)
    sel_counts = torch.empty((n_ctas,), dtype=torch.int32, device=q.device)

    used_cuda = False
    if (
        cuda_lib is not None
        and hasattr(cuda_lib, "score_kmax_blocks_fp8_sm121_fwd")
        and D == 128
        and q.dtype == torch.bfloat16
        and k_mid.dtype == torch.bfloat16
        and k_rad.dtype == torch.bfloat16
        and q.stride(2) == 1
        and k_mid.stride(2) == 1
        and k_rad.stride(2) == 1
    ):
        stream = int(torch.cuda.current_stream(q.device).cuda_stream)
        rc = cuda_lib.score_kmax_blocks_fp8_sm121_fwd(
            q.data_ptr(), k_mid.data_ptr(), k_rad.data_ptr(),
            block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
            scores.data_ptr(), sel_counts.data_ptr(),
            q.stride(0), q.stride(1),
            k_mid.stride(0), k_mid.stride(1),
            block_table.stride(0),
            scores.stride(0),
            int(num_reqs), int(max_seqlen_q), int(max_pblks),
            int(H), int(G), int(BS), int(E),
            int(s_blks), int(l_blks), int(k_sel),
            stream,
        )
        used_cuda = (rc == 0)

    if not used_cuda:
        _score_kmax_blocks_kernel[(num_mt * H, num_reqs)](
            q, k_mid, k_rad, block_table, cache_seqlens, cu_seqlens_q, scores, sel_counts,
            q.stride(0), q.stride(1),
            k_mid.stride(0), k_mid.stride(1),
            block_table.stride(0),
            scores.stride(0),
            max_pblks, s_blks, l_blks, k_sel,
            H=H, G=G, BS=BS, E=E, D=D, BLOCK_M=128, BLOCK_B=64,
            num_warps=4, num_stages=2,
        )

    _, top_idx = torch.topk(scores, k=k_sel, dim=-1, largest=True, sorted=False)
    sel_blocks, _ = torch.sort(top_idx.to(torch.int32), dim=-1)
    return sel_blocks.contiguous(), sel_counts.contiguous()


@triton.jit
def _sigmoid(x):
    """Exact match to sm12_attn._sigmoid (PTX ex2.approx + rcp.approx)."""
    exp = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[-x * 1.4426950408889634],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[exp + 1.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# ==================================================================================================
# Mode 1 (bf16_lockstep — 100% 0-ULP Bit-Exact) & Mode 2 (bf16_pvskip — Lossless PV-Skip)
# ==================================================================================================
@triton.jit
def _global_prefill_bf16_kernel(
    Q, K, V, BT, SEQ, CUQ, BIAS, O, L, GATE, SINK, Y,
    sq_t, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    sbt_r,
    sb_t, sb_h,
    so_t, so_h,
    sl_h, sl_t,
    sg_t, ssk_t,
    scale_log2,
    pv_skip_tau,
    H: tl.constexpr,
    E: tl.constexpr, EW: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    FUSE_GS: tl.constexpr = True,
    PV_SKIP: tl.constexpr = False,
):
    """48-SM L2-Lockstep + Bias-Free Prefix Split Global Full-Attention Kernel.

    Grid: (num_mt * H, num_reqs).
      - pid0 maps ``h = pid0 % H; mt = pid0 // H`` so each 48-CTA hardware wave packs
        ALL 6 heads x 8 adjacent Q-tiles (= 1,024 tokens x 6 heads = 6,144 Q-rows)
        marching n0 = 0, BN, 2*BN, ... in lockstep (48x L2 reuse of K/V!).
      - Phase A (Bias-free prefix [0, k_bias_start)): ZERO bias/causal checks, ZERO k_ok
        masks, coalesced D-contiguous K loads (transposed in registers/SMEM), optional
        lossless softmax PV-skip when max_row(tile_max - m_new) < -pv_skip_tau.
      - Phase B (Biased tail [k_bias_start, k_end)): exact sm12_attn.py biased+causal loop
        with identical n0 alignment (0-ULP bit-exact when PV_SKIP=False and BLOCK_N=64!).
    """
    pid = tl.program_id(0)
    r = tl.program_id(1)
    h = pid % H
    mt = pid // H
    kvh = h // G

    q0 = tl.load(CUQ + r)
    qlen = tl.load(CUQ + r + 1) - q0
    seqlen = tl.load(SEQ + r)
    past = seqlen - qlen

    i_lo = mt * BLOCK_M
    if i_lo >= qlen:
        return
    i_hi = tl.minimum(i_lo + BLOCK_M - 1, qlen - 1)

    i_loc = i_lo + tl.arange(0, BLOCK_M)
    row_ok = i_loc < qlen
    tok = q0 + i_loc
    qpos = past + i_loc
    tile_end = tl.minimum((i_lo // 128 + 1) * 128, qlen)
    off = ((past + tile_end + 127) // 128 - 10) * 128

    dd = tl.arange(0, D)
    q = tl.load(
        Q + tok.to(tl.int64)[:, None] * sq_t + h * sq_h + dd[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )

    k_end = past + i_hi + 1
    q_min = past + i_lo
    # Exact bias-free prefix boundary: every tile n0 < k_bias_start satisfies
    # n0 + BLOCK_N <= q_min - E + 1 (if E > 0) and n0 + BLOCK_N <= q_min <= k_end!
    k_bias_raw = tl.maximum(q_min - E + 1, 0) if E > 0 else tl.maximum(q_min, 0)
    k_bias_start = (k_bias_raw // BLOCK_N) * BLOCK_N

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    nn = tl.arange(0, BLOCK_N)
    bt_row = BT + r.to(tl.int64) * sbt_r

    # ---- Phase A: 99.90% Bias-Free Rectangular Prefix [0, k_bias_start) ----
    # Exact [D, BLOCK_N] K layout + tl.dot(q, k) matches stock sm12_attn._attn_prefill_full_kernel
    # MMA & warpsPerCTA reduction order for literal 0-ULP (nd == 0) bit-exactness!
    for n0 in range(0, k_bias_start, BLOCK_N):
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            blk_s = tl.load(bt_row + n0 // BS).to(tl.int64)
            slot = (n0 % BS) + nn
            k = tl.load(K + blk_s * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None])
        else:
            kpos = n0 + nn
            blk = tl.load(bt_row + kpos // BS).to(tl.int64)
            slot = kpos % BS
            k = tl.load(K + blk[None, :] * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None])

        s = tl.dot(q, k) * scale_log2
        s = tl.where(row_ok[:, None], s, float("-inf"))
        tile_max = tl.max(s, 1)
        m_new = tl.maximum(m_i, tile_max)

        if PV_SKIP:
            # Mask padded rows so partial last Q-tile still achieves full PV-skip rate!
            rel = tl.where(row_ok, tile_max - m_new, -1.0e9)
            max_rel = tl.max(rel, 0)
            if max_rel >= -pv_skip_tau:
                m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.exp2(m_i - m_safe)
                p = tl.exp2(s - m_safe[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
                    v = tl.load(V + blk_s * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :])
                else:
                    v = tl.load(V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :])
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new
        else:
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
                v = tl.load(V + blk_s * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :])
            else:
                v = tl.load(V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    # ---- Phase B: 0.10% Exact Biased + Causal Tail [k_bias_start, k_end) ----
    b_base = tok.to(tl.int64) * sb_t + h * sb_h
    for n0 in range(k_bias_start, k_end, BLOCK_N):
        kpos = n0 + nn
        k_ok = kpos < k_end
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            blk_s = tl.load(bt_row + n0 // BS).to(tl.int64)
            slot = (n0 % BS) + nn
            k = tl.load(
                K + blk_s * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None],
                mask=k_ok[None, :],
                other=0.0,
            )
        else:
            blk = tl.load(bt_row + kpos // BS, mask=k_ok, other=0).to(tl.int64)
            slot = kpos % BS
            k = tl.load(
                K + blk[None, :] * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None],
                mask=k_ok[None, :],
                other=0.0,
            )

        s = tl.dot(q, k)
        d = qpos[:, None] - kpos[None, :]
        valid = row_ok[:, None] & k_ok[None, :] & (d >= 0)
        if E > 0:
            col = kpos[None, :] - off
            b_ok = valid & (d < E) & (col >= 0) & (col < EW)
            bias = tl.load(
                BIAS + b_base[:, None] + col.to(tl.int64),
                mask=b_ok,
                other=0.0,
            ).to(tl.float32)
            bias = tl.where(bias == bias, bias, 0.0)
            s = s * scale_log2 + bias * 1.4426950408889634
        else:
            s = s * scale_log2
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            v = tl.load(
                V + blk_s * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        else:
            v = tl.load(
                V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # ---- Epilogue (exact match to sm12_attn._attn_prefill_full_kernel) ----
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    if FUSE_GS:
        col_g = h * D + dd[None, :]
        gate = tl.load(GATE + tok.to(tl.int64)[:, None] * sg_t + col_g, mask=row_ok[:, None], other=0).to(tl.float32)
        sink = tl.load(SINK + tok.to(tl.int64) * ssk_t + h, mask=row_ok, other=0).to(tl.float32)
        res = out * _sigmoid(gate)
        res = res * _sigmoid(lse - sink)[:, None]
        tl.store(Y + tok.to(tl.int64)[:, None] * (H * D) + col_g, res, mask=row_ok[:, None])
    else:
        tl.store(O + tok.to(tl.int64)[:, None] * so_t + h * so_h + dd[None, :], out, mask=row_ok[:, None])
        tl.store(L + h * sl_h + tok.to(tl.int64) * sl_t, lse, mask=row_ok)


# ==================================================================================================
# Mode 3 (Triton): SM121a SageAttention2 FP8 E4M3 + Lossless PV-Skip + Exact BF16 Biased Tail
# ==================================================================================================
@triton.jit
def _global_prefill_fp8_sage2_kernel(
    Q, K, V, BT, SEQ, CUQ, BIAS, O, L, GATE, SINK, Y,
    sq_t, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    sbt_r,
    sb_t, sb_h,
    so_t, so_h,
    sl_h, sl_t,
    sg_t, ssk_t,
    scale_log2,
    pv_skip_tau,
    H: tl.constexpr,
    E: tl.constexpr, EW: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    FUSE_GS: tl.constexpr = True,
    PV_SKIP: tl.constexpr = True,
    FP16_ACC: tl.constexpr = True,
    PV_FP8: tl.constexpr = False,
):
    """Path 2A Dense Global Attention (SageAttention2 FP8 on 99.9% bias-free prefix +
    100% exact BF16 on the 0.1% biased tail).

    Zero-overflow FP16-accum invariant:
      - When FP16_ACC=True, Q and K are scaled to [-22.0, 22.0] before casting to e4m3fn,
        bounding every dot product by |q . k| <= 128 * 22^2 = 61,952 < 65,504 (FP16 max)!
        Triggers SM121a mma.sync.m16n8k32.f16.e4m3.e4m3.f16 (~314 TFLOPS peak!).
      - When FP16_ACC=False, Q and K use full [-448.0, 448.0] e4m3fn dynamic range with
        FP32 accumulation (~157 TFLOPS peak).
      - BF16 Q is reloaded ONCE after the prefix loop so it never occupies registers during
        the 16,384-tile prefix!
    """
    pid = tl.program_id(0)
    r = tl.program_id(1)
    h = pid % H
    mt = pid // H
    kvh = h // G

    q0 = tl.load(CUQ + r)
    qlen = tl.load(CUQ + r + 1) - q0
    seqlen = tl.load(SEQ + r)
    past = seqlen - qlen

    i_lo = mt * BLOCK_M
    if i_lo >= qlen:
        return
    i_hi = tl.minimum(i_lo + BLOCK_M - 1, qlen - 1)

    i_loc = i_lo + tl.arange(0, BLOCK_M)
    row_ok = i_loc < qlen
    tok = q0 + i_loc
    qpos = past + i_loc
    tile_end = tl.minimum((i_lo // 128 + 1) * 128, qlen)
    off = ((past + tile_end + 127) // 128 - 10) * 128

    dd = tl.arange(0, D)
    q_ptrs = Q + tok.to(tl.int64)[:, None] * sq_t + h * sq_h + dd[None, :]

    k_end = past + i_hi + 1
    q_min = past + i_lo
    k_bias_raw = tl.maximum(q_min - E + 1, 0) if E > 0 else tl.maximum(q_min, 0)
    k_bias_start = (k_bias_raw // BLOCK_N) * BLOCK_N

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    nn = tl.arange(0, BLOCK_N)
    bt_row = BT + r.to(tl.int64) * sbt_r

    # ---- Phase A: 99.90% Bias-Free Prefix in Native SM121a FP8 E4M3 ----
    if k_bias_start > 0:
        Q_MAX: tl.constexpr = 22.0 if FP16_ACC else 448.0
        INV_Q_MAX: tl.constexpr = (1.0 / 22.0) if FP16_ACC else (1.0 / 448.0)

        q_bf16 = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
        q_f32 = q_bf16.to(tl.float32)
        q_amax = tl.maximum(tl.max(tl.abs(q_f32), 1), 1.0e-12)
        q_fp8 = (q_f32 * (Q_MAX / q_amax)[:, None]).to(tl.float8e4nv)
        qs_log2 = (q_amax * (INV_Q_MAX * scale_log2))  # [BLOCK_M] fp32

        for n0 in range(0, k_bias_start, BLOCK_N):
            if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
                blk_s = tl.load(bt_row + n0 // BS).to(tl.int64)
                slot = (n0 % BS) + nn
                k_row = tl.load(K + blk_s * sk_b + slot[:, None] * sk_s + kvh * sk_h + dd[None, :])
            else:
                kpos = n0 + nn
                blk = tl.load(bt_row + kpos // BS).to(tl.int64)
                slot = kpos % BS
                k_row = tl.load(K + blk[:, None] * sk_b + slot[:, None] * sk_s + kvh * sk_h + dd[None, :])

            k_f32 = k_row.to(tl.float32)
            k_amax = tl.maximum(tl.max(tl.abs(k_f32), 1), 1.0e-12)  # [BLOCK_N]
            k_fp8 = (k_f32 * (Q_MAX / k_amax)[:, None]).to(tl.float8e4nv)
            k_scale = k_amax * INV_Q_MAX

            if FP16_ACC:
                s_raw = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float16).to(tl.float32)
            else:
                s_raw = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)
            s = s_raw * qs_log2[:, None] * k_scale[None, :]

            tile_max = tl.max(s, 1)
            m_new = tl.maximum(m_i, tile_max)

            do_pv = True
            if PV_SKIP:
                rel = tl.where(row_ok, tile_max - m_new, -1.0e9)
                do_pv = tl.max(rel, 0) >= -pv_skip_tau

            if do_pv:
                m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.exp2(m_i - m_safe)
                p = tl.exp2(s - m_safe[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
                    v = tl.load(V + blk_s * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :])
                else:
                    v = tl.load(V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :])
                if PV_FP8:
                    # Per-channel V scaling + P scaled to [0, 22] -> max dot <= BN * 484 <= 61,952 < 65,504!
                    v_f32 = v.to(tl.float32)
                    v_amax = tl.maximum(tl.max(tl.abs(v_f32), 0), 1.0e-12)  # [D]
                    v_fp8 = (v_f32 * (22.0 / v_amax)[None, :]).to(tl.float8e4nv)
                    p_fp8 = (p * 22.0).to(tl.float8e4nv)
                    if FP16_ACC:
                        pv = tl.dot(p_fp8, v_fp8, out_dtype=tl.float16).to(tl.float32)
                    else:
                        pv = tl.dot(p_fp8, v_fp8, out_dtype=tl.float32)
                    acc = acc * alpha[:, None] + pv * ((1.0 / (22.0 * 22.0)) * v_amax)[None, :]
                else:
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new

    # ---- Phase B: 0.10% Exact BF16 Biased + Causal Tail [k_bias_start, k_end) ----
    # Reload BF16 Q once so BF16 Q never consumes registers during the long FP8 prefix loop!
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
    b_base = tok.to(tl.int64) * sb_t + h * sb_h
    for n0 in range(k_bias_start, k_end, BLOCK_N):
        kpos = n0 + nn
        k_ok = kpos < k_end
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            blk_s = tl.load(bt_row + n0 // BS).to(tl.int64)
            slot = (n0 % BS) + nn
            k_row = tl.load(
                K + blk_s * sk_b + slot[:, None] * sk_s + kvh * sk_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        else:
            blk = tl.load(bt_row + kpos // BS, mask=k_ok, other=0).to(tl.int64)
            slot = kpos % BS
            k_row = tl.load(
                K + blk[:, None] * sk_b + slot[:, None] * sk_s + kvh * sk_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )

        s = tl.dot(q, tl.trans(k_row))
        d = qpos[:, None] - kpos[None, :]
        valid = row_ok[:, None] & k_ok[None, :] & (d >= 0)
        if E > 0:
            col = kpos[None, :] - off
            b_ok = valid & (d < E) & (col >= 0) & (col < EW)
            bias = tl.load(
                BIAS + b_base[:, None] + col.to(tl.int64),
                mask=b_ok,
                other=0.0,
            ).to(tl.float32)
            bias = tl.where(bias == bias, bias, 0.0)
            s = s * scale_log2 + bias * 1.4426950408889634
        else:
            s = s * scale_log2
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            v = tl.load(
                V + blk_s * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        else:
            v = tl.load(
                V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    if FUSE_GS:
        col_g = h * D + dd[None, :]
        gate = tl.load(GATE + tok.to(tl.int64)[:, None] * sg_t + col_g, mask=row_ok[:, None], other=0).to(tl.float32)
        sink = tl.load(SINK + tok.to(tl.int64) * ssk_t + h, mask=row_ok, other=0).to(tl.float32)
        res = out * _sigmoid(gate)
        res = res * _sigmoid(lse - sink)[:, None]
        tl.store(Y + tok.to(tl.int64)[:, None] * (H * D) + col_g, res, mask=row_ok[:, None])
    else:
        tl.store(O + tok.to(tl.int64)[:, None] * so_t + h * so_h + dd[None, :], out, mask=row_ok[:, None])
        tl.store(L + h * sl_h + tok.to(tl.int64) * sl_t, lse, mask=row_ok)


# ==================================================================================================
# Exact BF16 Biased-Tail Merge Kernel (paired with CUDA SM121a Sage2 Prefix Kernel)
# ==================================================================================================
@triton.jit
def _global_biased_tail_merge_kernel(
    Q, K, V, BT, SEQ, CUQ, BIAS, ACC_IN, M_IN, L_IN, O, L, GATE, SINK, Y,
    sq_t, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    sbt_r,
    sb_t, sb_h,
    so_t, so_h,
    sl_h, sl_t,
    sg_t, ssk_t,
    scale_log2,
    H: tl.constexpr,
    E: tl.constexpr, EW: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PREFIX_BN: tl.constexpr = 64,
    FUSE_GS: tl.constexpr = True,
):
    """Continues online softmax from CUDA Sage2 prefix state (ACC_IN, M_IN, L_IN) across
    ONLY the tiny [k_bias_start, k_end) biased tail (~16 tiles) in 100% exact BF16."""
    pid = tl.program_id(0)
    r = tl.program_id(1)
    h = pid % H
    mt = pid // H
    kvh = h // G

    q0 = tl.load(CUQ + r)
    qlen = tl.load(CUQ + r + 1) - q0
    seqlen = tl.load(SEQ + r)
    past = seqlen - qlen

    i_lo = mt * BLOCK_M
    if i_lo >= qlen:
        return
    i_hi = tl.minimum(i_lo + BLOCK_M - 1, qlen - 1)

    i_loc = i_lo + tl.arange(0, BLOCK_M)
    row_ok = i_loc < qlen
    tok = q0 + i_loc
    qpos = past + i_loc
    tile_end = tl.minimum((i_lo // 128 + 1) * 128, qlen)
    off = ((past + tile_end + 127) // 128 - 10) * 128

    dd = tl.arange(0, D)
    q = tl.load(
        Q + tok.to(tl.int64)[:, None] * sq_t + h * sq_h + dd[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )

    k_end = past + i_hi + 1
    q_min = past + i_lo
    k_bias_raw = tl.maximum(q_min - E + 1, 0) if E > 0 else tl.maximum(q_min, 0)
    k_bias_start = (k_bias_raw // PREFIX_BN) * PREFIX_BN

    if k_bias_start > 0:
        m_i = tl.load(M_IN + tok.to(tl.int64) * H + h, mask=row_ok, other=float("-inf"))
        l_i = tl.load(L_IN + tok.to(tl.int64) * H + h, mask=row_ok, other=0.0)
        acc = tl.load(
            ACC_IN + (tok.to(tl.int64) * H + h)[:, None] * D + dd[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

    nn = tl.arange(0, BLOCK_N)
    bt_row = BT + r.to(tl.int64) * sbt_r
    b_base = tok.to(tl.int64) * sb_t + h * sb_h

    for n0 in range(k_bias_start, k_end, BLOCK_N):
        kpos = n0 + nn
        k_ok = kpos < k_end
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            blk_s = tl.load(bt_row + n0 // BS).to(tl.int64)
            slot = (n0 % BS) + nn
            k_row = tl.load(
                K + blk_s * sk_b + slot[:, None] * sk_s + kvh * sk_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        else:
            blk = tl.load(bt_row + kpos // BS, mask=k_ok, other=0).to(tl.int64)
            slot = kpos % BS
            k_row = tl.load(
                K + blk[:, None] * sk_b + slot[:, None] * sk_s + kvh * sk_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )

        s = tl.dot(q, tl.trans(k_row))
        d = qpos[:, None] - kpos[None, :]
        valid = row_ok[:, None] & k_ok[None, :] & (d >= 0)
        if E > 0:
            col = kpos[None, :] - off
            b_ok = valid & (d < E) & (col >= 0) & (col < EW)
            bias = tl.load(
                BIAS + b_base[:, None] + col.to(tl.int64),
                mask=b_ok,
                other=0.0,
            ).to(tl.float32)
            bias = tl.where(bias == bias, bias, 0.0)
            s = s * scale_log2 + bias * 1.4426950408889634
        else:
            s = s * scale_log2
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            v = tl.load(
                V + blk_s * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        else:
            v = tl.load(
                V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                mask=k_ok[:, None],
                other=0.0,
            )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    if FUSE_GS:
        col_g = h * D + dd[None, :]
        gate = tl.load(GATE + tok.to(tl.int64)[:, None] * sg_t + col_g, mask=row_ok[:, None], other=0).to(tl.float32)
        sink = tl.load(SINK + tok.to(tl.int64) * ssk_t + h, mask=row_ok, other=0).to(tl.float32)
        res = out * _sigmoid(gate)
        res = res * _sigmoid(lse - sink)[:, None]
        tl.store(Y + tok.to(tl.int64)[:, None] * (H * D) + col_g, res, mask=row_ok[:, None])
    else:
        tl.store(O + tok.to(tl.int64)[:, None] * so_t + h * so_h + dd[None, :], out, mask=row_ok[:, None])
        tl.store(L + h * sl_h + tok.to(tl.int64) * sl_t, lse, mask=row_ok)


# ==================================================================================================
# Public Dispatch & Gating API (matches sm12_attn._launch_prefill_opt 20-tuple args!)
# ==================================================================================================
def should_use_global_1m(
    window: int,
    max_seqlen_q: int,
    past_est: int | None = None,
    mode: str | None = None,
) -> bool:
    """Strict gate: ONLY global prefill chunks (`window < 0` and `qlen >= MIN_QLEN` and
    `past >= MIN_PAST` when known). Guarantees C=1 decode (T <= 16) is 100% untouched."""
    m = MODE if mode is None else mode
    if m in ("", "0", "off", "false", "no"):
        return False
    if window >= 0:
        return False
    if max_seqlen_q < MIN_QLEN:
        return False
    if past_est is not None and past_est < MIN_PAST:
        return False
    return True


def _pick_cfg(mode: str, max_seqlen_q: int, bm=None, bn=None, warps=None, stages=None):
    # Mode 1 (0-ULP bit-exact) defaults to BM=128, BN=64 (matches stock n0=0,64,...!).
    # Mode 2/3 default to BM=128, BN=64, 4 warps (low reg pressure, 2 CTAs/SM -> 96 CTAs!).
    if mode == "bf16_lockstep":
        d_bm = 128 if max_seqlen_q >= 1024 else 64
        d_bn = 64
        d_w = 8 if d_bm == 128 else 4
        d_s = 2
    elif mode == "bf16_pvskip":
        d_bm = 128 if max_seqlen_q >= 1024 else 64
        d_bn = 64
        d_w = 4
        d_s = 2
    else:  # fp8_sage2
        d_bm = 128
        d_bn = 64
        d_w = 4
        d_s = 2
    c_bm = bm or _ENV_BM or d_bm
    c_bn = bn or _ENV_BN or d_bn
    c_w = warps or _ENV_WARPS or d_w
    c_s = stages or _ENV_STAGES or d_s
    # Sheared bias `off` formula assumes all rows in a CTA share one 128-token bias window.
    if c_bm > 128:
        c_bm = 128
    return c_bm, c_bn, c_w, c_s


def launch_global_sage2_prefix_v3_only(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    E: int,
    G: int,
    BS: int,
    H: int,
    D: int,
    num_reqs: int,
    *,
    pv_skip_tau: float | None = None,
    bn: int | None = None,
    min_ctas: int | None = None,
    topk_tokens: int | None = None,
    cuda_lib_path: str | None = None,
    k_fp8_cache: torch.Tensor | None = None,
    k_scale64_cache: torch.Tensor | None = None,
    k_mid_cache: torch.Tensor | None = None,
    k_rad_cache: torch.Tensor | None = None,
    max_seq: int | None = None,
    k_bt: torch.Tensor | None = None,
    k_bt_mode: int = 0,
):
    """Run ONLY the SM121a Sage2 v3/v4/v5 FP8 prefix kernel and return
    `(acc_buf, m_buf, l_buf, prefix_bm=128, tail_prefix_bn)` for direct handoff to
    `libglobal_tail_otf_rel_sm121.so` (ZERO sheared rel_logits, ZERO Triton tail merge!)."""
    lib = get_cuda_lib(cuda_lib_path)
    p_bn = bn or _ENV_BN or 64
    if p_bn not in (64, 128):
        p_bn = 64
    p_min_ctas = _ENV_MIN_CTAS if min_ctas is None else int(min_ctas)
    if p_min_ctas not in (1, 2):
        p_min_ctas = 2
    tau = PV_SKIP_TAU if pv_skip_tau is None else float(pv_skip_tau)
    need_v5 = int(k_bt_mode) != 0
    if not (
        lib is not None
        and (
            hasattr(lib, "global_sage2_prefix_v5_sm121_fwd")
            if need_v5
            else hasattr(lib, "global_sage2_prefix_v3_sm121_fwd")
        )
        and k_fp8_cache is not None
        and k_scale64_cache is not None
        and k_scale64_cache.shape[-1] == BS // 64
        and D == 128
        and BS >= p_bn
        and BS % p_bn == 0
        and q.dtype == torch.bfloat16
        and value_cache.dtype == torch.bfloat16
        and k_fp8_cache.element_size() == 1
        and k_scale64_cache.dtype == torch.float32
        and q.stride(2) == 1
        and k_fp8_cache.stride(3) == 1
        and k_scale64_cache.stride(2) == 1
        and value_cache.stride(3) == 1
    ):
        return None

    eff_topk = GLOBAL_TOPK if topk_tokens is None else int(topk_tokens)
    sel_blocks = sel_counts = None
    if eff_topk > 0 and k_mid_cache is not None and k_rad_cache is not None:
        score_bt = k_bt if (need_v5 and k_bt is not None) else block_table
        sel_blocks, sel_counts = select_topk_blocks_exact_kmax(
            q, k_mid_cache, k_rad_cache, score_bt, cache_seqlens, cu_seqlens_q,
            max_seqlen_q, E, BS, H, G, D, num_reqs, eff_topk,
            max_seq=max_seq, cuda_lib=lib,
        )
    use_sparse = sel_blocks is not None and sel_counts is not None
    T = q.shape[0]
    acc_buf, m_buf, l_buf = _get_cuda_scratch(q.device, T, H, D)
    stream = int(torch.cuda.current_stream(q.device).cuda_stream)
    if need_v5:
        rc = lib.global_sage2_prefix_v5_sm121_fwd(
            q.data_ptr(), k_fp8_cache.data_ptr(), k_scale64_cache.data_ptr(), value_cache.data_ptr(),
            block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
            sel_blocks.data_ptr() if use_sparse else 0,
            sel_counts.data_ptr() if use_sparse else 0,
            acc_buf.data_ptr(), m_buf.data_ptr(), l_buf.data_ptr(),
            q.stride(0), q.stride(1),
            k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
            k_scale64_cache.stride(0), k_scale64_cache.stride(1),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0),
            int(num_reqs), int(max_seqlen_q), int(H), int(G),
            int(BS), int(E), int(p_bn), int(p_min_ctas),
            int(sel_blocks.shape[1]) if use_sparse else 0,
            float(softmax_scale * LOG2E), float(tau),
            k_bt.data_ptr() if (int(k_bt_mode) == 2 and k_bt is not None) else 0,
            int(k_bt.stride(0)) if (int(k_bt_mode) == 2 and k_bt is not None) else 0,
            int(k_bt_mode),
            stream,
        )
    else:
        rc = lib.global_sage2_prefix_v3_sm121_fwd(
            q.data_ptr(), k_fp8_cache.data_ptr(), k_scale64_cache.data_ptr(), value_cache.data_ptr(),
            block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
            sel_blocks.data_ptr() if use_sparse else 0,
            sel_counts.data_ptr() if use_sparse else 0,
            acc_buf.data_ptr(), m_buf.data_ptr(), l_buf.data_ptr(),
            q.stride(0), q.stride(1),
            k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
            k_scale64_cache.stride(0), k_scale64_cache.stride(1),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0),
            int(num_reqs), int(max_seqlen_q), int(H), int(G),
            int(BS), int(E), int(p_bn), int(p_min_ctas),
            int(sel_blocks.shape[1]) if use_sparse else 0,
            float(softmax_scale * LOG2E), float(tau),
            stream,
        )
    if rc != 0:
        return None
    return acc_buf, m_buf, l_buf, 128, (BS if use_sparse else p_bn)


def launch_global_prefill(
    args,
    gate=None,
    sinks=None,
    result=None,
    *,
    mode: str | None = None,
    pv_skip_tau: float | None = None,
    sage2_acc: str | None = None,
    sage2_pv: str | None = None,
    bm: int | None = None,
    bn: int | None = None,
    warps: int | None = None,
    stages: int | None = None,
    cuda_lib_path: str | None = None,
    k_fp8_cache: torch.Tensor | None = None,
    k_scale_cache: torch.Tensor | None = None,
    k_scale64_cache: torch.Tensor | None = None,
    k_mid_cache: torch.Tensor | None = None,
    k_rad_cache: torch.Tensor | None = None,
    min_ctas: int | None = None,
    topk_tokens: int | None = None,
    max_seq: int | None = None,
    k_bt: torch.Tensor | None = None,
    k_bt_mode: int = 0,
) -> bool:
    """Drop-in replacement for the `window < 0` branch of `sm12_attn._launch_prefill_opt`.
    Returns True if executed, False if caller should fall back to stock `_attn_prefill_full_kernel`."""
    (
        q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
        sheared_bias, out, lse, max_seqlen_q, softmax_scale, window, E, EW, G, BS, H, Hkv, D, num_reqs,
    ) = args
    m = (MODE if mode is None else mode).strip().lower()
    if window >= 0 or m in ("", "0", "off", "false", "no"):
        return False

    fuse = gate is not None
    dummy_g = gate if fuse else q
    dummy_s = sinks if fuse else q
    dummy_y = result if fuse else q
    sg_t = gate.stride(0) if fuse else 0
    ssk_t = sinks.stride(0) if fuse else 0
    tau = PV_SKIP_TAU if pv_skip_tau is None else float(pv_skip_tau)

    # ---- Mode 3 v3/v5 (CUDA SM121a 2-CTA/SM 16-Warp + Per-64 Scale + Packed Half2 hmax2 + Optional Top-K!) ----
    if m == "fp8_sage2_cuda_v3":
        lib = get_cuda_lib(cuda_lib_path)
        p_bn = bn or _ENV_BN or 64
        if p_bn not in (64, 128):
            p_bn = 64
        p_min_ctas = _ENV_MIN_CTAS if min_ctas is None else int(min_ctas)
        if p_min_ctas not in (1, 2):
            p_min_ctas = 2
        ks64 = k_scale64_cache if k_scale64_cache is not None else k_scale_cache
        need_v5 = int(k_bt_mode) != 0
        if (
            lib is not None
            and (
                hasattr(lib, "global_sage2_prefix_v5_sm121_fwd")
                if need_v5
                else hasattr(lib, "global_sage2_prefix_v3_sm121_fwd")
            )
            and k_fp8_cache is not None
            and ks64 is not None
            and ks64.shape[-1] == BS // 64
            and D == 128
            and BS >= p_bn
            and BS % p_bn == 0
            and q.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and k_fp8_cache.element_size() == 1
            and ks64.dtype == torch.float32
            and q.stride(2) == 1
            and k_fp8_cache.stride(3) == 1
            and ks64.stride(2) == 1
            and value_cache.stride(3) == 1
        ):
            eff_topk = GLOBAL_TOPK if topk_tokens is None else int(topk_tokens)
            sel_blocks = sel_counts = None
            if eff_topk > 0 and k_mid_cache is not None and k_rad_cache is not None:
                score_bt = k_bt if (need_v5 and k_bt is not None) else block_table
                sel_blocks, sel_counts = select_topk_blocks_exact_kmax(
                    q, k_mid_cache, k_rad_cache, score_bt, cache_seqlens, cu_seqlens_q,
                    max_seqlen_q, E, BS, H, G, D, num_reqs, eff_topk,
                    max_seq=max_seq, cuda_lib=lib,
                )
            use_sparse = sel_blocks is not None and sel_counts is not None
            T = q.shape[0]
            acc_buf, m_buf, l_buf = _get_cuda_scratch(q.device, T, H, D)
            stream = int(torch.cuda.current_stream(q.device).cuda_stream)
            if need_v5:
                rc = lib.global_sage2_prefix_v5_sm121_fwd(
                    q.data_ptr(), k_fp8_cache.data_ptr(), ks64.data_ptr(), value_cache.data_ptr(),
                    block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
                    sel_blocks.data_ptr() if use_sparse else 0,
                    sel_counts.data_ptr() if use_sparse else 0,
                    acc_buf.data_ptr(), m_buf.data_ptr(), l_buf.data_ptr(),
                    q.stride(0), q.stride(1),
                    k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
                    ks64.stride(0), ks64.stride(1),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    block_table.stride(0),
                    int(num_reqs), int(max_seqlen_q), int(H), int(G),
                    int(BS), int(E), int(p_bn), int(p_min_ctas),
                    int(sel_blocks.shape[1]) if use_sparse else 0,
                    float(softmax_scale * LOG2E), float(tau),
                    k_bt.data_ptr() if (int(k_bt_mode) == 2 and k_bt is not None) else 0,
                    int(k_bt.stride(0)) if (int(k_bt_mode) == 2 and k_bt is not None) else 0,
                    int(k_bt_mode),
                    stream,
                )
            else:
                rc = lib.global_sage2_prefix_v3_sm121_fwd(
                    q.data_ptr(), k_fp8_cache.data_ptr(), ks64.data_ptr(), value_cache.data_ptr(),
                    block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
                    sel_blocks.data_ptr() if use_sparse else 0,
                    sel_counts.data_ptr() if use_sparse else 0,
                    acc_buf.data_ptr(), m_buf.data_ptr(), l_buf.data_ptr(),
                    q.stride(0), q.stride(1),
                    k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
                    ks64.stride(0), ks64.stride(1),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    block_table.stride(0),
                    int(num_reqs), int(max_seqlen_q), int(H), int(G),
                    int(BS), int(E), int(p_bn), int(p_min_ctas),
                    int(sel_blocks.shape[1]) if use_sparse else 0,
                    float(softmax_scale * LOG2E), float(tau),
                    stream,
                )
            if rc == 0:
                t_bm, t_bn = 128, 64
                tail_prefix_bn = BS if use_sparse else p_bn
                num_mt = triton.cdiv(max_seqlen_q, t_bm)
                grid = (num_mt * H, num_reqs)
                _global_biased_tail_merge_kernel[grid](
                    q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
                    sheared_bias, acc_buf, m_buf, l_buf, out, lse, dummy_g, dummy_s, dummy_y,
                    q.stride(0), q.stride(1),
                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    block_table.stride(0), sheared_bias.stride(0), sheared_bias.stride(1),
                    out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
                    sg_t, ssk_t, softmax_scale * LOG2E,
                    H=H, E=E, EW=EW, G=G, BS=BS, D=D,
                    BLOCK_M=t_bm, BLOCK_N=t_bn, PREFIX_BN=tail_prefix_bn, FUSE_GS=fuse,
                    num_warps=4, num_stages=2,
                )
                return True
        # Graceful fallback to v2 / v1 if v3 symbol or ks64 absent
        m = "fp8_sage2_cuda_v2"

    # ---- Mode 3 v2 (CUDA SM121a Pre-Quantized FP8 Shadow K + cp.async + ldmatrix.x4!) ----
    if m == "fp8_sage2_cuda_v2":
        lib = get_cuda_lib(cuda_lib_path)
        p_bn = bn or _ENV_BN or 64
        if p_bn not in (64, 128):
            p_bn = 64
        if (
            lib is not None
            and hasattr(lib, "global_sage2_prefix_v2_sm121_fwd")
            and k_fp8_cache is not None
            and k_scale_cache is not None
            and D == 128
            and BS >= p_bn
            and BS % p_bn == 0
            and q.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and k_fp8_cache.element_size() == 1
            and k_scale_cache.dtype == torch.float32
            and q.stride(2) == 1
            and k_fp8_cache.stride(3) == 1
            and k_scale_cache.stride(2) == 1
            and value_cache.stride(3) == 1
        ):
            T = q.shape[0]
            acc_buf, m_buf, l_buf = _get_cuda_scratch(q.device, T, H, D)
            stream = int(torch.cuda.current_stream(q.device).cuda_stream)
            rc = lib.global_sage2_prefix_v2_sm121_fwd(
                q.data_ptr(), k_fp8_cache.data_ptr(), k_scale_cache.data_ptr(), value_cache.data_ptr(),
                block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
                acc_buf.data_ptr(), m_buf.data_ptr(), l_buf.data_ptr(),
                q.stride(0), q.stride(1),
                k_fp8_cache.stride(0), k_fp8_cache.stride(1), k_fp8_cache.stride(2),
                k_scale_cache.stride(0), k_scale_cache.stride(1),
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                block_table.stride(0),
                int(num_reqs), int(max_seqlen_q), int(H), int(G),
                int(BS), int(E), int(p_bn),
                float(softmax_scale * LOG2E), float(tau),
                stream,
            )
            if rc == 0:
                t_bm, t_bn = 128, 64
                num_mt = triton.cdiv(max_seqlen_q, t_bm)
                grid = (num_mt * H, num_reqs)
                _global_biased_tail_merge_kernel[grid](
                    q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
                    sheared_bias, acc_buf, m_buf, l_buf, out, lse, dummy_g, dummy_s, dummy_y,
                    q.stride(0), q.stride(1),
                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    block_table.stride(0), sheared_bias.stride(0), sheared_bias.stride(1),
                    out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
                    sg_t, ssk_t, softmax_scale * LOG2E,
                    H=H, E=E, EW=EW, G=G, BS=BS, D=D,
                    BLOCK_M=t_bm, BLOCK_N=t_bn, PREFIX_BN=p_bn, FUSE_GS=fuse,
                    num_warps=4, num_stages=2,
                )
                return True
        # Graceful fallback to v1 CUDA (online K quant) if shadow cache or v2 symbol absent
        m = "fp8_sage2_cuda"

    # ---- Mode 3 v1 (CUDA SM121a Native mma.sync.m16n8k32.f16.e4m3.e4m3.f16, online K quant) ----
    if m == "fp8_sage2_cuda":
        lib = get_cuda_lib(cuda_lib_path)
        if (
            lib is not None
            and D == 128
            and BS >= 64
            and BS % 64 == 0
            and q.dtype == torch.bfloat16
            and key_cache.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and q.stride(2) == 1
            and key_cache.stride(3) == 1
            and value_cache.stride(3) == 1
        ):
            T = q.shape[0]
            acc_buf, m_buf, l_buf = _get_cuda_scratch(q.device, T, H, D)
            stream = int(torch.cuda.current_stream(q.device).cuda_stream)
            rc = lib.global_sage2_prefix_sm121_fwd(
                q.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
                block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
                acc_buf.data_ptr(), m_buf.data_ptr(), l_buf.data_ptr(),
                q.stride(0), q.stride(1),
                key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                block_table.stride(0),
                int(num_reqs), int(max_seqlen_q), int(H), int(G),
                int(BS), int(E),
                float(softmax_scale * LOG2E), float(tau),
                stream,
            )
            if rc == 0:
                t_bm, t_bn = 128, 64
                num_mt = triton.cdiv(max_seqlen_q, t_bm)
                grid = (num_mt * H, num_reqs)
                _global_biased_tail_merge_kernel[grid](
                    q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
                    sheared_bias, acc_buf, m_buf, l_buf, out, lse, dummy_g, dummy_s, dummy_y,
                    q.stride(0), q.stride(1),
                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    block_table.stride(0), sheared_bias.stride(0), sheared_bias.stride(1),
                    out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
                    sg_t, ssk_t, softmax_scale * LOG2E,
                    H=H, E=E, EW=EW, G=G, BS=BS, D=D,
                    BLOCK_M=t_bm, BLOCK_N=t_bn, FUSE_GS=fuse,
                    num_warps=4, num_stages=2,
                )
                return True
        # Graceful fallback to Triton fp8_sage2 if CUDA .so absent/ineligible
        m = "fp8_sage2"

    c_bm, c_bn, c_warps, c_stages = _pick_cfg(m, max_seqlen_q, bm, bn, warps, stages)
    num_mt = triton.cdiv(max_seqlen_q, c_bm)
    grid = (num_mt * H, num_reqs)

    if m in ("bf16_lockstep", "bf16_pvskip"):
        _global_prefill_bf16_kernel[grid](
            q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
            sheared_bias, out, lse, dummy_g, dummy_s, dummy_y,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0), sheared_bias.stride(0), sheared_bias.stride(1),
            out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
            sg_t, ssk_t, softmax_scale * LOG2E, tau,
            H=H, E=E, EW=EW, G=G, BS=BS, D=D,
            BLOCK_M=c_bm, BLOCK_N=c_bn, FUSE_GS=fuse, PV_SKIP=(m == "bf16_pvskip"),
            num_warps=c_warps, num_stages=c_stages,
        )
        return True

    if m == "fp8_sage2":
        acc_mode = (SAGE2_ACC if sage2_acc is None else sage2_acc).strip().lower()
        pv_mode = (SAGE2_PV if sage2_pv is None else sage2_pv).strip().lower()
        _global_prefill_fp8_sage2_kernel[grid](
            q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
            sheared_bias, out, lse, dummy_g, dummy_s, dummy_y,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0), sheared_bias.stride(0), sheared_bias.stride(1),
            out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
            sg_t, ssk_t, softmax_scale * LOG2E, tau,
            H=H, E=E, EW=EW, G=G, BS=BS, D=D,
            BLOCK_M=c_bm, BLOCK_N=c_bn, FUSE_GS=fuse,
            PV_SKIP=(tau > 0.0),
            FP16_ACC=(acc_mode != "f32"),
            PV_FP8=(pv_mode == "fp8_f16"),
            num_warps=c_warps, num_stages=c_stages,
        )
        return True

    return False
