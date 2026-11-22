# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP Bit-Exact Fused Target Decode Attention Wrapper.

Wraps `fused_decode_attn_exact_sm121.cu`:
  1. `attn_prologue_kvq_exact_fwd`: 24-CTA 1-wave K-conv + V-conv + Q-clip ONLY
     (eliminates all 16 RelProj CTAs & 2.09 MB/layer sheared `rel_logits` round-trip!)
  2. `fused_decode_attn_otf_mgs_fwd`:
     - Pre-`gdc_wait()` async past-KV prefetch for splits `0..S-2`
     - On-the-fly exact `_l2_clip(R)` + `r_clip @ rel_proj_T` hidden behind `cp.async.cg` of K
     - Double-buffered XOR-swizzled SMEM + `ldmatrix.x4` / `ldmatrix.x4.trans`
     - Cooperative last-split `MergeGateSink` with persistent L2-hot ping-pong `OP`/`LP` scratch
       (eliminates 48 MGS launches/step AND 11.0 MB/step dirty per-layer `OP` allocations!)

Recommended paired L2PF env retuning (since fused attn cuts attn duration ~21 -> ~11 us/layer):
  INKLING_L2PF_PACE_ATTN_GBPS=200
  INKLING_L2PF_MAX_ATTN_US=10
  INKLING_L2PF_WO_MB=2.5
"""
from __future__ import annotations

import ctypes
import math
import os
import threading

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

ENABLED = os.environ.get("INKLING_FUSED_DECODE_ATTN", "0").strip().lower() in ("1", "true", "on", "yes")
ENABLED_C32 = os.environ.get("INKLING_FUSED_DECODE_ATTN_C32", "0").strip().lower() in ("1", "true", "on", "yes")
MAX_TOKENS_C32 = int(os.environ.get("INKLING_FUSED_ATTN_C32_MAX_T", "320"))
_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_LIB", "/rdma/libfused_decode_attn_exact_sm121.so")
_TREE_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_TREE_LIB", "/rdma/libfused_decode_attn_tree_exact_sm121.so")
_C32_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_C32_LIB", "/rdma/libfused_decode_attn_exact_sm121_c32.so")
_C32_FUSE_MT = os.environ.get("INKLING_FUSED_ATTN_C32_FUSE_MT", "auto").strip().lower()
_C32_EVICT_FIRST = os.environ.get("INKLING_FUSED_ATTN_C32_EVICT_FIRST", "1").strip().lower() not in ("0", "", "off", "false", "no")
_C32_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_C32_SPLITS", "1"))
_TREE_MODE = int(os.environ.get("INKLING_FUSED_ATTN_TREE_MODE", "0"))  # 0, 1, or 2 (auto-calibrated for nd==0)
_USE_PDL = os.environ.get("INKLING_FUSED_ATTN_PDL", "1").strip().lower() not in ("0", "", "off", "false", "no")
_TRIG_WO = os.environ.get("INKLING_FUSED_ATTN_TRIG_WO", "1").strip().lower() not in ("0", "", "off", "false", "no")
_LOG2E = 1.4426950408889634

_LIB = None
_TREE_LIB = None
_TREE_LIB_TRIED = False
_HAS_C32_SYM = False
_LIB_LOCK = threading.Lock()

# 2x ping-pong persistent L2-hot scratch per device:
#   OP:  [16, 16, 6, 128] fp32 = 768 KiB/slot (1.5 MiB total, 100% L2-resident!)
#   LP:  [16, 16, 6]      fp32 =   6 KiB/slot
#   SEM: [64]            int32 = 256 B/slot (self-resetting on last split!)
_SCRATCH: dict[int, dict] = {}
_REL_PROJ_T_CACHE: dict[int, object] = {}


def _load_lib():
    global _LIB, _HAS_C32_SYM
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is None:
            if os.environ.get("INKLING_FUSED_ATTN_LIB"):
                cands = [_LIB_PATH, _C32_LIB_PATH]
            elif ENABLED_C32:
                cands = [_C32_LIB_PATH, _LIB_PATH]
            else:
                cands = [_LIB_PATH]
            path = next((p for p in cands if os.path.exists(p)), None)
            if path is None:
                return None
            lib = ctypes.CDLL(path)
            lib.attn_prologue_kvq_exact_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
                ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.attn_prologue_kvq_exact_fwd.restype = ctypes.c_int

            lib.fused_decode_attn_otf_mgs_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.fused_decode_attn_otf_mgs_fwd.restype = ctypes.c_int
            if hasattr(lib, "fused_decode_attn_otf_mgs_c32_fwd"):
                lib.fused_decode_attn_otf_mgs_c32_fwd.argtypes = (
                    list(lib.fused_decode_attn_otf_mgs_fwd.argtypes[:-1])
                    + [ctypes.c_int, ctypes.c_int, ctypes.c_uint64]
                )
                lib.fused_decode_attn_otf_mgs_c32_fwd.restype = ctypes.c_int
                _HAS_C32_SYM = True
            _LIB = lib
    return _LIB


def prepack_rel_proj_t(rel_proj_weight):
    """Cache contiguous `[E, 16]` BF16 transpose of `rel_logits_proj` (shape `[16, E]` or `[E, 16]`)."""
    assert torch is not None
    key = int(rel_proj_weight.data_ptr())
    hit = _REL_PROJ_T_CACHE.get(key)
    if hit is not None:
        return hit
    w = rel_proj_weight.detach()
    if w.shape[0] == 16 and w.shape[1] != 16:
        wt = w.transpose(0, 1).contiguous()
    else:
        wt = w.contiguous()
    _REL_PROJ_T_CACHE[key] = wt
    return wt


def ensure_scratch(device, max_splits: int = 16, max_t: int = 16, max_h: int = 6, max_sem: int = 64):
    assert torch is not None
    dev_idx = device.index if device.index is not None else 0
    s = _SCRATCH.get(dev_idx)
    if s is None or s["op"][0].shape[0] < max_splits or s["op"][0].shape[1] < max_t or s["op"][0].shape[2] < max_h:
        _SCRATCH[dev_idx] = {
            "op": [
                torch.empty((max_splits, max_t, max_h, 128), dtype=torch.float32, device=device),
                torch.empty((max_splits, max_t, max_h, 128), dtype=torch.float32, device=device),
            ],
            "lp": [
                torch.empty((max_splits, max_t, max_h), dtype=torch.float32, device=device),
                torch.empty((max_splits, max_t, max_h), dtype=torch.float32, device=device),
            ],
            "sem": [
                torch.zeros((max_sem,), dtype=torch.int32, device=device),
                torch.zeros((max_sem,), dtype=torch.int32, device=device),
            ],
            "idx": 0,
        }
    return _SCRATCH[dev_idx]


def _next_scratch(device, num_splits: int, T: int, H: int, need_sem: int):
    s = ensure_scratch(device, max(16, num_splits), max(16, T), max(6, H), max(64, need_sem))
    idx = s["idx"]
    s["idx"] = idx ^ 1
    return s["op"][idx], s["lp"][idx], s["sem"][idx]


def fused_target_decode_attn(
    q, r, k, v,
    q_scale, r_scale, tau, rel_proj_t,
    k_conv_w, v_conv_w, k_dyn, v_dyn,
    conv_cache, key_cache, value_cache,
    kv_slot_mapping, positions, conv_block_table, conv_seq_idx, conv_slot_mapping, conv_query_start,
    attn_block_table, attn_seq_lens, attn_cu_seqlens_q,
    gate, sinks, output,
    softmax_scale: float,
    window: int,
    rel_threshold: float,
    clip_threshold: float,
    clip_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_actual_tokens: int | None = None,
    n_live_buf=None,
    q_out_buf=None,
    tree_mode: int = _TREE_MODE,
    pdl: bool = _USE_PDL,
    trig_wo: bool = _TRIG_WO,
    max_seqlen_q: int | None = None,
) -> bool:
    """Execute the complete 2-kernel 0-ULP fused target decode attention pipeline:
      1. `attn_prologue_kvq_exact_fwd` (24 CTAs for T<=16, wave-balanced 96 CTAs for T=17..320; NO RelProj!)
      2. `fused_decode_attn_otf_mgs[_c32]_fwd` (OTF rel-bias + XOR ldmatrix + direct S=1 Gate+Sink / last-split MGS!)
    Returns True if executed, False if unsupported shape/config (caller falls back cleanly).
    """
    lib = _load_lib()
    if lib is None or torch is None:
        return False
    T = int(q.shape[0])
    if T <= 0:
        return False
    if T <= 16:
        if not ENABLED:
            return False
    else:
        if not (ENABLED_C32 and _HAS_C32_SYM and T <= MAX_TOKENS_C32):
            return False
    NT = T if num_actual_tokens is None else int(num_actual_tokens)
    num_reqs = int(attn_seq_lens.shape[0])
    H = int(q_scale.shape[0])
    Hkv = int(key_cache.shape[2])
    G = H // Hkv
    eff_max_q = int(max_seqlen_q) if max_seqlen_q is not None else T
    if eff_max_q * G > 64:
        return False
    if window >= 0 and window != 1023:
        return False
    if num_reqs <= 4:
        num_splits = 8 if window >= 0 else 16
    else:
        if not _HAS_C32_SYM:
            return False
        num_splits = _C32_SPLITS if _C32_SPLITS in (1, 2, 4) else 1
    if T > 16:
        # At M >= 17 wo_ud runs cuBLAS BF16 GEMM (not PDL skinny); never arm _WO_ARMED!
        pdl = False
        trig_wo = False

    q_out = q_out_buf if q_out_buf is not None else torch.empty((T, H, 128), dtype=torch.bfloat16, device=q.device)
    dynamic = (k_dyn is not None and v_dyn is not None)
    has_tau = (tau is not None)
    has_nlive = (n_live_buf is not None)
    # Exact Inkling2 W=4 short-conv scale matching patches-s6/sconv_ops.py:1191!
    width = 4
    sconv_scale = float((width * (2 if dynamic else 1) + 1) ** -0.5)
    # conv_cache shape is [num_blocks, 1, N_cache, C] -> stride(2) & shape[2] (matching sconv_ops.py:1182,1188)!
    c_n_dim = 2 if conv_cache.dim() == 4 else 1
    stream = int(torch.cuda.current_stream(q.device).cuda_stream)

    rc = lib.attn_prologue_kvq_exact_fwd(
        int(k.data_ptr()), int(v.data_ptr()),
        int(k_conv_w.data_ptr()), int(v_conv_w.data_ptr()),
        int(k_dyn.data_ptr()) if dynamic else 0,
        int(v_dyn.data_ptr()) if dynamic else 0,
        int(conv_cache.data_ptr()), int(key_cache.data_ptr()), int(value_cache.data_ptr()),
        int(kv_slot_mapping.data_ptr()), int(positions.data_ptr()),
        int(conv_seq_idx.data_ptr()), int(conv_slot_mapping.data_ptr()),
        int(conv_block_table.data_ptr()), int(conv_query_start.data_ptr()),
        int(q.data_ptr()), int(q_scale.data_ptr()),
        int(tau.data_ptr()) if has_tau else 0,
        int(q_out.data_ptr()),
        int(n_live_buf.data_ptr()) if has_nlive else 0,
        T, int(k.stride(0)), int(v.stride(0)),
        int(k_conv_w.stride(0)), int(k_conv_w.stride(1)),
        int(k_dyn.stride(0)) if dynamic else 0,
        int(v_dyn.stride(0)) if dynamic else 0,
        int(conv_cache.stride(0)), int(conv_cache.stride(c_n_dim)),
        int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.shape[1]),
        int(conv_block_table.stride(0)), int(conv_block_table.shape[1]), int(conv_cache.shape[c_n_dim]),
        int(q.stride(0)),
        sconv_scale, float(clip_threshold), float(clip_eps), float(norm_eps),
        1 if dynamic else 0, 1 if has_tau else 0, 1 if has_nlive else 0, 1 if pdl else 0,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"attn_prologue_kvq_exact_fwd failed: cudaError={rc}")

    mtiles = (eff_max_q * G + 15) // 16
    if num_splits == 1:
        op_ptr = lp_ptr = sem_ptr = 0
        so_s = so_t = so_h = sl_s = sl_t = sl_h = 0
    else:
        op_buf, lp_buf, sem_buf = _next_scratch(q.device, num_splits, T, H, num_reqs * mtiles * Hkv)
        op_ptr, lp_ptr, sem_ptr = int(op_buf.data_ptr()), int(lp_buf.data_ptr()), int(sem_buf.data_ptr())
        so_s, so_t, so_h = int(op_buf.stride(0)), int(op_buf.stride(1)), int(op_buf.stride(2))
        sl_s, sl_t, sl_h = int(lp_buf.stride(0)), int(lp_buf.stride(1)), int(lp_buf.stride(2))
    E = int(rel_proj_t.shape[0]) if rel_proj_t is not None else 0
    r_view = r.view(T, H, 16) if r.dim() == 2 else r

    use_fuse_mt = False
    if _HAS_C32_SYM and T > 16 and 1 <= mtiles <= 3 and num_splits in (1, 2, 4):
        if _C32_FUSE_MT == "1":
            use_fuse_mt = True
        elif _C32_FUSE_MT == "auto":
            use_fuse_mt = (num_reqs >= 16)

    common_args = (
        int(q_out.data_ptr()),
        int(r_view.data_ptr()),
        int(r_scale.data_ptr()),
        int(tau.data_ptr()) if has_tau else 0,
        int(rel_proj_t.data_ptr()) if E > 0 else 0,
        int(key_cache.data_ptr()),
        int(value_cache.data_ptr()),
        int(attn_block_table.data_ptr()),
        int(attn_seq_lens.data_ptr()),
        int(attn_cu_seqlens_q.data_ptr()),
        int(gate.data_ptr()),
        int(sinks.data_ptr()),
        op_ptr, lp_ptr, sem_ptr,
        int(output.data_ptr()),
        int(n_live_buf.data_ptr()) if has_nlive else 0,
        int(q_out.stride(0)), int(q_out.stride(1)),
        int(r_view.stride(0)), int(r_view.stride(1)),
        int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.stride(2)),
        int(value_cache.stride(0)), int(value_cache.stride(1)), int(value_cache.stride(2)),
        int(attn_block_table.stride(0)),
        int(gate.stride(0)), int(sinks.stride(0)), int(output.stride(0)),
        so_s, so_t, so_h, sl_s, sl_t, sl_h,
        float(softmax_scale * _LOG2E), float(rel_threshold), float(clip_eps),
        T, NT, mtiles, num_reqs, Hkv, int(key_cache.shape[1]), E, G,
        int(window), int(num_splits),
        1 if has_tau else 0, 1 if has_nlive else 0,
        1 if pdl else 0, 1 if trig_wo else 0, int(tree_mode),
    )
    if _HAS_C32_SYM and (use_fuse_mt or T > 16):
        rc = lib.fused_decode_attn_otf_mgs_c32_fwd(
            *common_args, 1 if use_fuse_mt else 0, 1 if _C32_EVICT_FIRST else 0, stream
        )
    else:
        rc = lib.fused_decode_attn_otf_mgs_fwd(*common_args, stream)
    if rc != 0:
        raise RuntimeError(f"fused_decode_attn_otf_mgs_fwd failed: cudaError={rc}")
    if NT < T:
        output[NT:].zero_()
    if T <= 16 and trig_wo and pdl:
        from . import sm12_attn as _sm12_attn
        _sm12_attn._WO_ARMED = True
    return True


def _bind_tree_symbols(lib):
    if not (hasattr(lib, "attn_prologue_kvq_tree_exact_fwd") and hasattr(lib, "fused_decode_attn_otf_mgs_tree_fwd")):
        return None
    lib.attn_prologue_kvq_tree_exact_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.attn_prologue_kvq_tree_exact_fwd.restype = ctypes.c_int

    lib.fused_decode_attn_otf_mgs_tree_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.fused_decode_attn_otf_mgs_tree_fwd.restype = ctypes.c_int
    return lib


def _load_tree_lib():
    global _TREE_LIB, _TREE_LIB_TRIED
    if _TREE_LIB_TRIED:
        return _TREE_LIB
    with _LIB_LOCK:
        if not _TREE_LIB_TRIED:
            _TREE_LIB_TRIED = True
            if os.path.exists(_TREE_LIB_PATH):
                _TREE_LIB = _bind_tree_symbols(ctypes.CDLL(_TREE_LIB_PATH))
            if _TREE_LIB is None:
                base = _load_lib()
                if base is not None:
                    _TREE_LIB = _bind_tree_symbols(base)
    return _TREE_LIB


def has_tree_lib() -> bool:
    return _load_tree_lib() is not None


def fused_target_decode_attn_tree(
    q, r, k, v,
    q_scale, r_scale, tau, rel_proj_t,
    k_conv_w, v_conv_w, k_dyn, v_dyn,
    conv_cache, key_cache, value_cache,
    kv_slot_mapping, positions, conv_block_table, conv_seq_idx, conv_slot_mapping, conv_query_start,
    attn_block_table, attn_seq_lens, attn_cu_seqlens_q,
    gate, sinks, output,
    *,
    tree_sconv_anc,
    tree_meta,
    softmax_scale: float,
    window: int,
    rel_threshold: float,
    clip_threshold: float,
    clip_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_actual_tokens: int | None = None,
    n_live_buf=None,
    q_out_buf=None,
    tree_mode: int = _TREE_MODE,
    pdl: bool = _USE_PDL,
    trig_wo: bool = _TRIG_WO,
) -> bool:
    """Execute 0-ULP 2-kernel fused SM121a two-branch tree decode attention (M <= 8, ws <= 2)."""
    lib = _load_tree_lib()
    if lib is None or torch is None or tree_sconv_anc is None or tree_meta is None:
        return False
    T = int(q.shape[0])
    num_reqs = int(attn_seq_lens.shape[0])
    H = int(q_scale.shape[0])
    Hkv = int(key_cache.shape[2])
    G = H // Hkv
    if T <= 0 or T > 8 or num_reqs != 1 or G != 6:
        return False
    if window >= 0 and window != 1023:
        return False
    NT = T if num_actual_tokens is None else int(num_actual_tokens)
    num_splits = 8 if window >= 0 else 16

    q_out = q_out_buf if q_out_buf is not None else torch.empty((T, H, 128), dtype=torch.bfloat16, device=q.device)
    dynamic = (k_dyn is not None and v_dyn is not None)
    has_tau = (tau is not None)
    has_nlive = (n_live_buf is not None)
    width = 4
    sconv_scale = float((width * (2 if dynamic else 1) + 1) ** -0.5)
    c_n_dim = 2 if conv_cache.dim() == 4 else 1
    stream = int(torch.cuda.current_stream(q.device).cuda_stream)

    rc = lib.attn_prologue_kvq_tree_exact_fwd(
        int(k.data_ptr()), int(v.data_ptr()),
        int(k_conv_w.data_ptr()), int(v_conv_w.data_ptr()),
        int(k_dyn.data_ptr()) if dynamic else 0,
        int(v_dyn.data_ptr()) if dynamic else 0,
        int(conv_cache.data_ptr()), int(key_cache.data_ptr()), int(value_cache.data_ptr()),
        int(kv_slot_mapping.data_ptr()), int(positions.data_ptr()),
        int(conv_seq_idx.data_ptr()), int(conv_slot_mapping.data_ptr()),
        int(conv_block_table.data_ptr()), int(conv_query_start.data_ptr()),
        int(q.data_ptr()), int(q_scale.data_ptr()),
        int(tau.data_ptr()) if has_tau else 0,
        int(q_out.data_ptr()),
        int(n_live_buf.data_ptr()) if has_nlive else 0,
        int(tree_sconv_anc.data_ptr()),
        T, int(k.stride(0)), int(v.stride(0)),
        int(k_conv_w.stride(0)), int(k_conv_w.stride(1)),
        int(k_dyn.stride(0)) if dynamic else 0,
        int(v_dyn.stride(0)) if dynamic else 0,
        int(conv_cache.stride(0)), int(conv_cache.stride(c_n_dim)),
        int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.shape[1]),
        int(conv_block_table.stride(0)), int(conv_block_table.shape[1]), int(conv_cache.shape[c_n_dim]),
        int(q.stride(0)),
        sconv_scale, float(clip_threshold), float(clip_eps), float(norm_eps),
        1 if dynamic else 0, 1 if has_tau else 0, 1 if has_nlive else 0, 1 if pdl else 0,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"attn_prologue_kvq_tree_exact_fwd failed: cudaError={rc}")

    mtiles = 4
    op_buf, lp_buf, sem_buf = _next_scratch(q.device, num_splits, T, H, mtiles * Hkv)
    E = int(rel_proj_t.shape[0]) if rel_proj_t is not None else 0
    r_view = r.view(T, H, 16) if r.dim() == 2 else r

    rc = lib.fused_decode_attn_otf_mgs_tree_fwd(
        int(q_out.data_ptr()),
        int(r_view.data_ptr()),
        int(r_scale.data_ptr()),
        int(tau.data_ptr()) if has_tau else 0,
        int(rel_proj_t.data_ptr()) if E > 0 else 0,
        int(key_cache.data_ptr()),
        int(value_cache.data_ptr()),
        int(attn_block_table.data_ptr()),
        int(attn_seq_lens.data_ptr()),
        int(attn_cu_seqlens_q.data_ptr()),
        int(gate.data_ptr()),
        int(sinks.data_ptr()),
        int(op_buf.data_ptr()),
        int(lp_buf.data_ptr()),
        int(sem_buf.data_ptr()),
        int(output.data_ptr()),
        int(n_live_buf.data_ptr()) if has_nlive else 0,
        int(tree_meta.data_ptr()),
        int(q_out.stride(0)), int(q_out.stride(1)),
        int(r_view.stride(0)), int(r_view.stride(1)),
        int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.stride(2)),
        int(value_cache.stride(0)), int(value_cache.stride(1)), int(value_cache.stride(2)),
        int(attn_block_table.stride(0)),
        int(gate.stride(0)), int(sinks.stride(0)), int(output.stride(0)),
        int(op_buf.stride(0)), int(op_buf.stride(1)), int(op_buf.stride(2)),
        int(lp_buf.stride(0)), int(lp_buf.stride(1)), int(lp_buf.stride(2)),
        float(softmax_scale * _LOG2E), float(rel_threshold), float(clip_eps),
        T, NT, mtiles, num_reqs, Hkv, int(key_cache.shape[1]), E, G,
        int(window), int(num_splits),
        1 if has_tau else 0, 1 if has_nlive else 0,
        1 if pdl else 0, 1 if trig_wo else 0, int(tree_mode),
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"fused_decode_attn_otf_mgs_tree_fwd failed: cudaError={rc}")
    if trig_wo and pdl:
        from . import sm12_attn as _sm12_attn
        _sm12_attn._WO_ARMED = True
    return True
