# SPDX-License-Identifier: Apache-2.0
"""GB10 L2 prefetch during TP all-reduce and attention waits (Inkling2 decode).

Supports two backends:
- Legacy Triton `.cg` burst (INKLING_L2PF_G2=0, original patches-s3 behavior).
- Hardware TMA `cp.async.bulk.prefetch.L2.global` paced/burst via
  `/rdma/libdsv41_l2pf_g2.so` (INKLING_L2PF_G2=1). Runs on a dedicated side
  stream joined ONLY at forward end (`join_end()`), so mid-forward `join()`
  never stalls main-stream kernels (`sconv`). One warp with `%globaltimer`
  pacing avoids contending with RDMA flag polls / staging.

Pure performance hint: no data is written, numerics are bit-identical.
"""
import ctypes
import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("INKLING_L2PF", "0") == "1"
G2 = os.environ.get("INKLING_L2PF_G2", "0") == "1"
MAX_BYTES = int(float(os.environ.get("INKLING_L2PF_MB", "10" if G2 else "6")) * (1 << 20))
GATE = os.environ.get("INKLING_L2PF_GATE", "1") == "1"
WO = os.environ.get("INKLING_L2PF_WO", "0") == "1"
DRAFT = os.environ.get("INKLING_L2PF_DRAFT", "0") == "1"
DRAFT_ATTN = os.environ.get("INKLING_L2PF_DRAFT_ATTN", "0") == "1"
HEAD = os.environ.get("INKLING_L2PF_HEAD", "1" if G2 else "0") == "1"
_STOP_RAW = os.environ.get("INKLING_L2PF_STOP", "1").strip().lower()
STOP_SOFT = _STOP_RAW in ("soft", "grace", "2")
STOP_FLAG = _STOP_RAW not in ("0", "", "off", "false", "no")
_GRACE_MOE_NS = int(float(os.environ.get("INKLING_L2PF_GRACE_MOE_US", "9.0")) * 1000)
_GRACE_GATE_NS = int(float(os.environ.get("INKLING_L2PF_GRACE_GATE_US", "6.0")) * 1000)
_GRACE_HEAD_NS = int(float(os.environ.get("INKLING_L2PF_GRACE_HEAD_US", "0.0")) * 1000)
_GRACE_DRAFT_NS = int(float(os.environ.get("INKLING_L2PF_GRACE_DRAFT_US", "0.0")) * 1000)
_POST_MOE_GBPS = float(os.environ.get("INKLING_L2PF_POST_MOE_GBPS", "105"))
_POST_GATE_GBPS = float(os.environ.get("INKLING_L2PF_POST_GATE_GBPS", "110"))
_POST_HEAD_GBPS = float(os.environ.get("INKLING_L2PF_POST_HEAD_GBPS", "120"))
_POST_DRAFT_GBPS = float(os.environ.get("INKLING_L2PF_POST_DRAFT_GBPS", "110"))

# Per-kind tuning for G2 hardware TMA prefetch:
# rate_gbps == 0 -> hardware TMA burst (dsv41_l2pf), > 0 -> 1-warp paced (dsv41_l2pf_paced[_stop]).
_CHUNK = max(16, int(os.environ.get("INKLING_L2PF_CHUNK", "16384")) & ~15)
_PACE_MOE_GBPS = float(os.environ.get("INKLING_L2PF_PACE_MOE_GBPS", "180"))
_PACE_GATE_GBPS = float(os.environ.get("INKLING_L2PF_PACE_GATE_GBPS", "140"))
_PACE_ATTN_GBPS = float(os.environ.get("INKLING_L2PF_PACE_ATTN_GBPS", "200"))
_PACE_DRAFT_GBPS = float(os.environ.get("INKLING_L2PF_PACE_DRAFT_GBPS", "150"))
_PACE_DRAFT_ATTN_GBPS = float(os.environ.get("INKLING_L2PF_PACE_DRAFT_ATTN_GBPS", "160"))
_PACE_HEAD_GBPS = float(os.environ.get("INKLING_L2PF_PACE_HEAD_GBPS", "200"))

_MAX_MOE_NS = int(float(os.environ.get("INKLING_L2PF_MAX_MOE_US", "55")) * 1000)
_MAX_GATE_NS = int(float(os.environ.get("INKLING_L2PF_MAX_GATE_US", "35")) * 1000)
_MAX_ATTN_NS = int(float(os.environ.get("INKLING_L2PF_MAX_ATTN_US", "30")) * 1000)
_MAX_DRAFT_NS = int(float(os.environ.get("INKLING_L2PF_MAX_DRAFT_US", "40")) * 1000)
_MAX_DRAFT_ATTN_NS = int(float(os.environ.get("INKLING_L2PF_MAX_DRAFT_ATTN_US", "5")) * 1000)
_MAX_HEAD_NS = int(float(os.environ.get("INKLING_L2PF_MAX_HEAD_US", "45")) * 1000)

_DELAY_NS = int(float(os.environ.get("INKLING_L2PF_DELAY_US", "0")) * 1000)
_DELAY_MOE_NS = int(float(os.environ.get("INKLING_L2PF_DELAY_MOE_US", os.environ.get("INKLING_L2PF_DELAY_US", "2.0"))) * 1000)
_DELAY_GATE_NS = int(float(os.environ.get("INKLING_L2PF_DELAY_GATE_US", os.environ.get("INKLING_L2PF_DELAY_US", "2.0"))) * 1000)
_DELAY_ATTN_NS = int(float(os.environ.get("INKLING_L2PF_DELAY_ATTN_US", "0")) * 1000)
_DELAY_DRAFT_NS = int(float(os.environ.get("INKLING_L2PF_DELAY_DRAFT_US", os.environ.get("INKLING_L2PF_DELAY_US", "2.0"))) * 1000)
_DELAY_DRAFT_ATTN_NS = int(float(os.environ.get("INKLING_L2PF_DELAY_DRAFT_ATTN_US", "0")) * 1000)
_DELAY_HEAD_NS = int(float(os.environ.get("INKLING_L2PF_DELAY_HEAD_US", os.environ.get("INKLING_L2PF_DELAY_US", "2.0"))) * 1000)
_ATTN_STREAM = os.environ.get("INKLING_L2PF_ATTN_STREAM", "1") == "1"
_WO_BYTES = int(float(os.environ.get("INKLING_L2PF_WO_MB", "3.5")) * (1 << 20))
_DRAFT_BYTES = int(float(os.environ.get("INKLING_L2PF_DRAFT_MB", "5.5")) * (1 << 20))
_DRAFT_ATTN_BYTES = int(float(os.environ.get("INKLING_L2PF_DRAFT_ATTN_MB", "1.0")) * (1 << 20))
_HEAD_BYTES = int(float(os.environ.get("INKLING_L2PF_HEAD_MB", "10.0")) * (1 << 20))
_LIB_PATH = os.environ.get("INKLING_L2PF_LIB", "/rdma/libdsv41_l2pf_g2.so")

# Opt-in C=32 (M = 17..256) AR-window L2 prefetching (0-ULP pure performance hint; default OFF).
# Exploits 45..300 us idle LPDDR5X DRAM windows during all 96 RDMA ARs at B=32 to prefetch up to
# 17.5 MiB into 24 MiB L2: next QKVR (47 MoE ARs), Gate + active MoE W13 slabs (48 Attn ARs),
# and CertHead/lm_head (L47 MoE AR), with ar_done_cnt cooperative soft-stop!
C32 = os.environ.get("INKLING_L2PF_C32", "0") == "1"
C32_MAX_M = max(16, int(os.environ.get("INKLING_L2PF_C32_MAX_M", "256") or 256))
C32_MOE_W13 = os.environ.get("INKLING_L2PF_C32_MOE_W13", "1") == "1"
C32_WO = os.environ.get("INKLING_L2PF_C32_WO", "0") == "1"
_C32_STOP_RAW = os.environ.get("INKLING_L2PF_C32_STOP", "soft").strip().lower()
C32_STOP_SOFT = _C32_STOP_RAW in ("soft", "grace", "2")
C32_STOP_FLAG = _C32_STOP_RAW not in ("0", "", "off", "false", "no")
_C32_MOE_BYTES = int(float(os.environ.get("INKLING_L2PF_C32_MOE_MB", "17.5")) * (1 << 20))
_C32_GATE_BYTES = int(float(os.environ.get("INKLING_L2PF_C32_GATE_MB", "17.5")) * (1 << 20))
_C32_HEAD_BYTES = int(float(os.environ.get("INKLING_L2PF_C32_HEAD_MB", "17.5")) * (1 << 20))
_C32_WO_BYTES = int(float(os.environ.get("INKLING_L2PF_C32_WO_MB", "7.1")) * (1 << 20))
_C32_PACE_GBPS = float(os.environ.get("INKLING_L2PF_C32_PACE_GBPS", "210"))
_C32_POST_GBPS = float(os.environ.get("INKLING_L2PF_C32_POST_GBPS", "110"))
_C32_GRACE_NS = int(float(os.environ.get("INKLING_L2PF_C32_GRACE_US", "8.0")) * 1000)
_C32_DELAY_NS = int(float(os.environ.get("INKLING_L2PF_C32_DELAY_US", "2.0")) * 1000)
_C32_HEAD_SCALES_CAP = int(os.environ.get("INKLING_L2PF_C32_HEAD_SCALES_KB", "1536")) * 1024

# Legacy Triton burst config
_BLOCK, _ITERS = 1024, 8
_CTAS = int(os.environ.get("INKLING_L2PF_CTAS", "8"))
_WARPS = int(os.environ.get("INKLING_L2PF_WARPS", "8"))

_side = None
_side_attn = None  # dedicated side stream for kind="attn" (prevents FIFO queueing with AR pacers)
_sink = None
_pending = False
_pending_end = False
_pending_cap = False
_pending_attn_end = False
_pending_attn_cap = False
_lib = None
_lib_checked = False
_has_paced_stop = False
_has_paced_softstop = False
_ar_done_ptr: int = 0
_ar_done_checked = False
_rdma_comm_ref = None
_seg_cache: dict[tuple, torch.Tensor] = {}
_head_ref = None  # optional lm_head module reference for Layer-47 cert_head/lm_head prefetch
_cuda_head_slabs = None  # optional live CUDA cert_head multi-seg target (or slabs Tensor) for 2 <= M <= 16
_cuda_head_slabs_c32 = None  # optional C=32 multi-seg head target (larger scales cap for 17.5 MiB window)
_CUDA_HEAD_SCALES_CAP = int(os.environ.get("INKLING_L2PF_HEAD_SCALES_KB", "768")) * 1024


def _cuda_head_segments(target, scales_cap: int = _CUDA_HEAD_SCALES_CAP):
    """Build multi-segment prefetch target (nv, rt, (scales, cap), slabs) from a CudaCertTable,
    or pass through an existing tuple/list/Tensor."""
    if target is None:
        return None
    if isinstance(target, (tuple, list)):
        if scales_cap != _CUDA_HEAD_SCALES_CAP and len(target) == 4 and isinstance(target[2], tuple):
            return (target[0], target[1], (target[2][0], scales_cap), target[3])
        return tuple(target)
    if hasattr(target, "slabs"):
        segs = []
        nv = getattr(target, "nv", None)
        if nv is not None:
            segs.append(nv)
        rt = getattr(target, "rt", None)
        if rt is not None:
            segs.append(rt)
        scales = getattr(target, "scales", None)
        if scales is not None and scales_cap >= 16:
            segs.append((scales, scales_cap))
        slabs = getattr(target, "slabs", None)
        if slabs is not None:
            segs.append(slabs)
        return tuple(segs) if segs else None
    return target


def set_cuda_head_slabs(slabs) -> None:
    """Register CUDA cert_head prefetch target for 2 <= M <= 16 (and C=32 when INKLING_L2PF_C32=1).

    Accepts a CudaCertTable (preferred: builds multi-seg (nv, rt, (scales, 768 KiB), slabs)
    covering first-wave cert_screen CTAs), a tuple of tensors/(tensor, max_bytes), or a
    single slabs Tensor. Auto-prewarms the segment descriptor cache outside graph capture.
    """
    global _cuda_head_slabs, _cuda_head_slabs_c32
    _cuda_head_slabs = _cuda_head_segments(slabs, _CUDA_HEAD_SCALES_CAP)
    _cuda_head_slabs_c32 = _cuda_head_segments(slabs, _C32_HEAD_SCALES_CAP)
    if _cuda_head_slabs is not None:
        prewarm(_cuda_head_slabs, kind="head")
    if C32 and _cuda_head_slabs_c32 is not None:
        for _m in (32, 64, 128, 256):
            if _m <= C32_MAX_M:
                prewarm(_cuda_head_slabs_c32, kind="head", num_tokens=_m)


def _get_ar_done_ptr() -> int:
    """Resolve static GPU-L2 DevState::ar_done_cnt pointer from rdma_ar_s7 (0 if unavailable)."""
    global _ar_done_ptr, _ar_done_checked, _rdma_comm_ref
    if _ar_done_checked:
        return _ar_done_ptr
    if not (STOP_FLAG or (C32 and C32_STOP_FLAG)) or not _has_paced_stop:
        _ar_done_checked = True
        return 0
    try:
        from vllm.distributed import get_tp_group
        comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
        _rdma_comm_ref = comm
        h = getattr(comm, "h", None) if comm is not None else None
        if h is not None and hasattr(h, "ar_done_ptr"):
            _ar_done_ptr = int(h.ar_done_ptr())
            _ar_done_checked = True
            print(f"[inkling-l2pf] cooperative AR stop-flag bound: ar_done_ptr={_ar_done_ptr:#x}")
            return _ar_done_ptr
    except Exception:
        pass
    return 0


def _rdma_handles_ar(num_tokens: int | None, hidden: int = 4608) -> bool:
    """True iff rdma_ar_ext handles an AR of num_tokens (and thus increments ar_done_cnt)."""
    if num_tokens is None or num_tokens <= 16:
        return True
    comm = _rdma_comm_ref
    if comm is None:
        try:
            from vllm.distributed import get_tp_group
            comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
        except Exception:
            comm = None
    if comm is None:
        return False
    max_b = getattr(comm, "max_reduce_bytes", None)
    if max_b is None:
        return True
    return (int(num_tokens) * hidden * 2) <= int(max_b)


def _load_g2():
    global _lib, _lib_checked, _has_paced_stop, _has_paced_softstop
    if _lib_checked:
        return _lib
    _lib_checked = True
    if not G2:
        return None
    try:
        lib = ctypes.CDLL(_LIB_PATH)
        lib.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_uint64]
        lib.dsv41_l2pf.restype = ctypes.c_int
        lib.dsv41_l2pf_paced.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
            ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.dsv41_l2pf_paced.restype = ctypes.c_int
        if hasattr(lib, "dsv41_l2pf_paced_stop"):
            lib.dsv41_l2pf_paced_stop.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
                ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
            ]
            lib.dsv41_l2pf_paced_stop.restype = ctypes.c_int
            _has_paced_stop = True
        if hasattr(lib, "dsv41_l2pf_paced_softstop"):
            lib.dsv41_l2pf_paced_softstop.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
                ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
                ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_uint64,
            ]
            lib.dsv41_l2pf_paced_softstop.restype = ctypes.c_int
            _has_paced_softstop = True
        _lib = lib
        print(f"[inkling-l2pf] G2 hardware TMA loaded from {_LIB_PATH} "
              f"(mb={MAX_BYTES/(1<<20):.1f} wo={int(WO)} draft={int(DRAFT)} draft_attn={int(DRAFT_ATTN)} "
              f"head={int(HEAD)} stop_flag={int(STOP_FLAG and _has_paced_stop)} "
              f"pace_gbps=moe:{_PACE_MOE_GBPS:.0f}/gate:{_PACE_GATE_GBPS:.0f}/attn:{_PACE_ATTN_GBPS:.0f}/draft:{_PACE_DRAFT_GBPS:.0f})")
    except Exception as e:
        print(f"[inkling-l2pf] G2 library {_LIB_PATH} unavailable ({e!r}); falling back to Triton")
        _lib = None
    return _lib


def set_head_ref(lm_head) -> None:
    global _head_ref
    _head_ref = lm_head


def get_head_weight(num_tokens: int | None = None):
    """Resolve Layer-47 prefetch target.
    - M == 1: raw lm_head.weight (stock head always).
    - 2 <= M <= 16: registered CUDA cert_head slabs (or Triton cert_head / raw lm_head).
    - 17 <= M <= C32_MAX_M (when INKLING_L2PF_C32=1): C=32 CUDA cert_head slabs if CertHead-C32
      is active for M, else exact active raw BF16 lm_head.weight!
    - M > 64 (when C32=0) or M > C32_MAX_M (prefill): None."""
    if not HEAD or _head_ref is None:
        return None
    max_head_m = C32_MAX_M if C32 else 64
    if num_tokens is not None and num_tokens > max_head_m:
        return None
    w = getattr(_head_ref, "weight", None)
    if w is None:
        return None
    if num_tokens is not None and num_tokens < 2:
        return w  # M=1 is never qualified for cert_head; stock bf16 lm_head runs
    if _cuda_head_slabs is not None and (num_tokens is None or num_tokens <= 16):
        return _cuda_head_slabs
    if num_tokens is not None and num_tokens > 16:
        if C32 and _cuda_head_slabs is not None:
            try:
                from . import cert_head_cuda as chc
                if getattr(chc, "C32_ENABLED", False) and getattr(chc, "_HAS_C32", False) and num_tokens <= getattr(chc, "C32_MAX_M", 16):
                    return _cuda_head_slabs_c32 if _cuda_head_slabs_c32 is not None else _cuda_head_slabs
            except Exception:
                pass
        if num_tokens <= 64:
            try:
                from . import cert_head
                t = cert_head._TABLES.get((int(w.data_ptr()), tuple(w.shape)))
                if t and hasattr(t, "q"):
                    segs = []
                    if hasattr(t, "nv") and t.nv is not None:
                        segs.append(t.nv)
                    if hasattr(t, "e") and t.e is not None:
                        e_rows = min(int(t.e.shape[0]), 2048)
                        segs.append(t.e[:e_rows])
                    segs.append(t.q)
                    return tuple(segs)
            except Exception:
                pass
        return w
    try:
        from . import cert_head
        t = cert_head._TABLES.get((int(w.data_ptr()), tuple(w.shape)))
        if t and hasattr(t, "q"):
            # Prefetch nv (0.1 MiB) + first rows of e and q matching cert_screen pid order!
            segs = []
            if hasattr(t, "nv") and t.nv is not None:
                segs.append(t.nv)
            if hasattr(t, "e") and t.e is not None:
                # Cap e to first 2048 rows (~288 KiB) so q gets >90% of budget
                e_rows = min(int(t.e.shape[0]), 2048)
                segs.append(t.e[:e_rows])
            segs.append(t.q)
            return tuple(segs)
    except Exception:
        pass
    return w


def _tensor_live_bytes(t: torch.Tensor | None) -> int:
    if t is None:
        return 0
    try:
        sb = int(t.untyped_storage().size())
    except Exception:
        sb = t.numel() * t.element_size()
    return min(t.numel() * t.element_size(), sb)


def resolve_dense_next(experts, num_tokens: int | None = None):
    """Resolve exact active next-layer QKVR weight tensor for MoE AR-window prefetching."""
    if num_tokens is None or num_tokens <= 16:
        return getattr(experts, "l2pf_next", None)
    raw = getattr(experts, "l2pf_next_raw", None)
    if raw is None:
        return None
    try:
        from . import skinny_bf11
        if hasattr(skinny_bf11, "will_use_bf11") and skinny_bf11.will_use_bf11(raw, num_tokens):
            return getattr(experts, "l2pf_next", raw)
    except Exception:
        pass
    return raw


def moe_w13_segments(experts, num_tokens: int) -> tuple:
    """Resolve exact active MoE W13 weight segments (nvfp4_ws5 / mx7 / raw) safely across resize_(0)."""
    if experts is None:
        return ()
    w13 = getattr(experts, "w13_weight", None)
    if w13 is None:
        return ()
    if getattr(experts, "nvfp4", False):
        meta = getattr(w13, "_inkling_nvfp4_ws5", None)
        if isinstance(meta, dict):
            segs = []
            ws_base = meta.get("ws_base")
            slabs = meta.get("slabs")
            if _tensor_live_bytes(ws_base) >= 16:
                segs.append(ws_base)
            if _tensor_live_bytes(slabs) >= 16:
                segs.append(slabs)
            if segs:
                return tuple(segs)
    else:
        mx7 = getattr(w13, "_inkling_mx7", None)
        if mx7 is not None:
            use_mx7 = num_tokens <= 128
            if not use_mx7:
                try:
                    from . import trtllm_bmm
                    use_mx7 = bool(getattr(trtllm_bmm, "_MX7_HAS_PREFILL", False))
                except Exception:
                    use_mx7 = False
            if use_mx7 and len(mx7) >= 4:
                segs = []
                if _tensor_live_bytes(mx7[3]) >= 16:
                    segs.append(mx7[3])
                if _tensor_live_bytes(mx7[1]) >= 16:
                    segs.append((mx7[1], 524288))
                if _tensor_live_bytes(mx7[0]) >= 16:
                    segs.append(mx7[0])
                if segs:
                    return tuple(segs)
    segs = []
    w13_s = getattr(experts, "w13_weight_scale", None)
    if _tensor_live_bytes(w13_s) >= 16:
        segs.append(w13_s)
    if _tensor_live_bytes(w13) >= 16:
        segs.append(w13)
    return tuple(segs)


def resolve_gate_target(layer, num_tokens: int | None = None):
    """Resolve Attn AR-window prefetch target:
    - M <= 16: exact C=1 gate tensor (BF12 slabs or BF16 weight).
    - 17 <= M <= C32_MAX_M (C32=1): active gate tensor (BF12 slabs when C32_ROUTER_MODE=bf12
      is active for M, else raw BF16 gate weight) + first ~14-15 MiB of active MoE W13 slabs!"""
    if num_tokens is None or num_tokens <= 16:
        return getattr(layer, "_l2pf_gate_w", layer.mlp.gate.weight)
    raw_gw = layer.mlp.gate.weight
    bf12_gw = getattr(layer, "_l2pf_gate_w", raw_gw)
    use_bf12 = False
    try:
        from . import moe_c32_runtime as _c32_moe
        if hasattr(_c32_moe, "will_use_bf12_gate"):
            use_bf12 = bool(_c32_moe.will_use_bf12_gate(num_tokens))
    except Exception:
        use_bf12 = False
    gw = bf12_gw if use_bf12 else raw_gw
    w13_segs = moe_w13_segments(layer.mlp.experts, num_tokens) if C32_MOE_W13 else ()
    target = (gw, *w13_segs) if w13_segs else gw
    # During eager startup prewarm, also cache the alternate gate target for this budget bucket
    # so FULL decode buckets on either side of C32_MOE_MIN_M never miss `_seg_cache` during capture.
    if bf12_gw is not raw_gw and not torch.cuda.is_current_stream_capturing():
        alt_gw = raw_gw if use_bf12 else bf12_gw
        prewarm((alt_gw, *w13_segs) if w13_segs else alt_gw, kind="gate", num_tokens=num_tokens)
    return target


def _align_seg(addr: int, nbytes: int) -> tuple[int, int] | None:
    rem = addr & 15
    if rem:
        adj = 16 - rem
        addr += adj
        nbytes = max(0, nbytes - adj)
    nbytes &= ~15
    return (addr, nbytes) if nbytes >= 16 else None


def _normalize_segments(target, budget: int) -> tuple[list[tuple[int, int]], torch.device | None]:
    """Accept a single Tensor, CudaCertTable, or sequence of Tensors/(Tensor, max_bytes) and build 16B-aligned segments."""
    if target is None or budget < 16:
        return [], None
    if not isinstance(target, (torch.Tensor, tuple, list)) and hasattr(target, "slabs"):
        target = _cuda_head_segments(target)
        if target is None:
            return [], None
    items = (target,) if isinstance(target, torch.Tensor) else tuple(target)
    segs: list[tuple[int, int]] = []
    dev = None
    left = budget
    for item in items:
        if left < 16:
            break
        if isinstance(item, tuple):
            t, cap = item
            live_b = _tensor_live_bytes(t)
            if live_b < 16:
                continue
            tb = min(live_b, int(cap), left)
        else:
            t = item
            live_b = _tensor_live_bytes(t)
            if live_b < 16:
                continue
            tb = min(live_b, left)
        seg = _align_seg(int(t.data_ptr()), tb)
        if seg is not None:
            segs.append(seg)
            left -= seg[1]
            if dev is None:
                dev = t.device
    return segs, dev


def _seg_tensor_multi(segs: list[tuple[int, int]], device: torch.device) -> torch.Tensor | None:
    if not segs:
        return None
    key = (int(device.index or 0), tuple(segs))
    t = _seg_cache.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # never allocate/H2D inside CUDA graph capture
        t = torch.tensor(segs, dtype=torch.int64, device=device)
        _seg_cache[key] = t
    return t


@triton.jit
def _l2pf_kernel(X, N, O, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64) * BLOCK * ITERS
    acc = tl.zeros([BLOCK], tl.int32)
    for base in range(pid * BLOCK * ITERS, N, step):
        for i in tl.static_range(ITERS):
            off = base + i * BLOCK + tl.arange(0, BLOCK)
            acc ^= tl.load(X + off, mask=off < N, other=0, cache_modifier=".cg")
    s = tl.sum(acc)
    if s == 0x13572468:
        tl.store(O, s)


def _kind_params(kind: str, max_bytes: int | None, num_tokens: int | None = None):
    if C32 and num_tokens is not None and 17 <= num_tokens <= C32_MAX_M:
        if num_tokens <= 32:
            win_mb, win_ns = 10 << 20, 48_000
        elif num_tokens <= 64:
            win_mb, win_ns = 14 << 20, 78_000
        elif num_tokens <= 128:
            win_mb, win_ns = (17 << 20) + (512 << 10), 125_000
        else:
            win_mb, win_ns = (17 << 20) + (512 << 10), 180_000
        if kind == "gate":
            b = min(_C32_GATE_BYTES, win_mb) if max_bytes is None else max_bytes
            return b, _C32_PACE_GBPS, win_ns, _C32_DELAY_NS
        if kind == "attn":
            b = _C32_WO_BYTES if max_bytes is None else max_bytes
            return b, _PACE_ATTN_GBPS, max(_MAX_ATTN_NS, 45_000), _DELAY_ATTN_NS
        if kind == "head":
            b = min(_C32_HEAD_BYTES, win_mb) if max_bytes is None else max_bytes
            return b, _C32_PACE_GBPS, win_ns, _C32_DELAY_NS
        if kind == "moe":
            b = min(_C32_MOE_BYTES, win_mb) if max_bytes is None else max_bytes
            return b, _C32_PACE_GBPS, win_ns, _C32_DELAY_NS
    if kind == "gate":
        return (MAX_BYTES if max_bytes is None else max_bytes), _PACE_GATE_GBPS, _MAX_GATE_NS, _DELAY_GATE_NS
    if kind == "attn":
        return (_WO_BYTES if max_bytes is None else max_bytes), _PACE_ATTN_GBPS, _MAX_ATTN_NS, _DELAY_ATTN_NS
    if kind == "draft_attn":
        return (_DRAFT_ATTN_BYTES if max_bytes is None else max_bytes), _PACE_DRAFT_ATTN_GBPS, _MAX_DRAFT_ATTN_NS, _DELAY_DRAFT_ATTN_NS
    if kind == "draft":
        return (_DRAFT_BYTES if max_bytes is None else max_bytes), _PACE_DRAFT_GBPS, _MAX_DRAFT_NS, _DELAY_DRAFT_NS
    if kind == "head":
        return (_HEAD_BYTES if max_bytes is None else max_bytes), _PACE_HEAD_GBPS, _MAX_HEAD_NS, _DELAY_HEAD_NS
    return (MAX_BYTES if max_bytes is None else max_bytes), _PACE_MOE_GBPS, _MAX_MOE_NS, _DELAY_MOE_NS


def _is_piecewise_capturing() -> bool:
    """True anywhere inside a PIECEWISE breakable CUDA graph capture (including inside its
    @eager_break_during_capture segments). Full decode captures and normal eager runs return False."""
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        cap = BreakableCUDAGraphCapture.current()
        if cap is None:
            return False
        if is_forward_context_available():
            if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
                return False
        return True
    except Exception:
        return False


def prewarm(weight, max_bytes: int | None = None, kind: str = "moe", num_tokens: int | None = None) -> None:
    """Pre-populate the device segment descriptor cache eagerly before CUDA graph capture."""
    if not ENABLED or not G2 or weight is None or torch.cuda.is_current_stream_capturing():
        return
    budget, _, _, _ = _kind_params(kind, max_bytes, num_tokens=num_tokens)
    segs, dev = _normalize_segments(weight, budget)
    if segs and dev is not None:
        _seg_tensor_multi(segs, dev)


def fork(
    weight,
    max_bytes: int | None = None,
    kind: str = "moe",
    stop: bool | None = None,
    num_tokens: int | None = None,
) -> None:
    global _side, _side_attn, _sink, _pending, _pending_end, _pending_cap, _pending_attn_end, _pending_attn_cap
    if not ENABLED or weight is None:
        return
    is_c32_call = C32 and num_tokens is not None and 17 <= num_tokens <= C32_MAX_M
    if kind == "attn" and not (WO or (is_c32_call and C32_WO)):
        return
    if kind == "draft_attn" and (not DRAFT or not DRAFT_ATTN):
        return
    if kind == "draft" and not DRAFT:
        return
    if kind == "head" and not HEAD:
        return
    if G2 and _is_piecewise_capturing():
        return
    budget, gbps, max_ns, delay_ns = _kind_params(kind, max_bytes, num_tokens=num_tokens)
    segs, dev = _normalize_segments(weight, budget)
    if not segs or dev is None:
        return
    if _side is None:
        _side = torch.cuda.Stream(device=dev)
        _sink = torch.zeros(1, dtype=torch.int32, device=dev)
    use_attn_stream = _ATTN_STREAM and kind in ("attn", "draft_attn")
    if use_attn_stream and _side_attn is None:
        _side_attn = torch.cuda.Stream(device=dev)
    cur = torch.cuda.current_stream()

    lib = _load_g2() if G2 else None
    if lib is not None:
        seg = _seg_tensor_multi(segs, dev)
        if seg is None:
            return
        st = _side_attn if use_attn_stream else _side
        st.wait_stream(cur)
        stream_ptr = int(st.cuda_stream)
        nseg = len(segs)
        if gbps > 0:
            period_ps = max(1, int(_CHUNK * 1000.0 / gbps))
            use_stop = (kind in ("moe", "gate", "head", "draft")) if stop is None else bool(stop)
            if is_c32_call:
                stop_ok = use_stop and C32_STOP_FLAG and _rdma_handles_ar(num_tokens)
                stop_ptr = _get_ar_done_ptr() if stop_ok else 0
                use_soft = C32_STOP_SOFT and _has_paced_softstop
            else:
                stop_ptr = _get_ar_done_ptr() if (use_stop and STOP_FLAG) else 0
                use_soft = STOP_SOFT and _has_paced_softstop
            if stop_ptr and use_soft:
                if is_c32_call:
                    g_ns, p_gbps = _C32_GRACE_NS, _C32_POST_GBPS
                elif kind == "gate":
                    g_ns, p_gbps = _GRACE_GATE_NS, _POST_GATE_GBPS
                elif kind == "head":
                    g_ns, p_gbps = _GRACE_HEAD_NS, _POST_HEAD_GBPS
                elif kind == "draft":
                    g_ns, p_gbps = _GRACE_DRAFT_NS, _POST_DRAFT_GBPS
                else:
                    g_ns, p_gbps = _GRACE_MOE_NS, _POST_MOE_GBPS
                post_period_ps = max(1, int(_CHUNK * 1000.0 / p_gbps)) if p_gbps > 0 else period_ps
                lib.dsv41_l2pf_paced_softstop(
                    ctypes.c_void_p(int(seg.data_ptr())), nseg, _CHUNK, period_ps,
                    delay_ns, max_ns, ctypes.c_void_p(stop_ptr),
                    g_ns, post_period_ps, None, stream_ptr,
                )
            elif stop_ptr:
                lib.dsv41_l2pf_paced_stop(
                    ctypes.c_void_p(int(seg.data_ptr())), nseg, _CHUNK, period_ps,
                    delay_ns, max_ns, ctypes.c_void_p(stop_ptr), None, stream_ptr,
                )
            else:
                lib.dsv41_l2pf_paced(
                    ctypes.c_void_p(int(seg.data_ptr())), nseg, _CHUNK, period_ps,
                    delay_ns, max_ns, None, stream_ptr,
                )
        else:
            lib.dsv41_l2pf(ctypes.c_void_p(int(seg.data_ptr())), nseg, _CHUNK, stream_ptr)
        cap = torch.cuda.is_current_stream_capturing()
        if use_attn_stream:
            _pending_attn_end = True
            _pending_attn_cap = cap
        else:
            _pending_end = True
            _pending_cap = cap
        return

    # Legacy Triton path (single tensor only, blocks at join())
    if not isinstance(weight, torch.Tensor):
        return
    n = segs[0][1] // 4
    if n <= 0:
        return
    flat = weight.view(-1).view(torch.int32)
    _side.wait_stream(cur)
    with torch.cuda.stream(_side):
        grid = (min(_CTAS, triton.cdiv(n, _BLOCK * _ITERS)),)
        _l2pf_kernel[grid](flat, n, _sink, BLOCK=_BLOCK, ITERS=_ITERS, num_warps=_WARPS)
    _pending = True


def join() -> None:
    """Mid-forward join: only active on legacy Triton path (no-op for G2 TMA)."""
    global _pending
    if _pending:
        torch.cuda.current_stream().wait_stream(_side)
        _pending = False


def join_end() -> None:
    """End-of-forward join: closes side stream(s) inside CUDA graph without stalling mid-forward."""
    global _pending, _pending_end, _pending_cap, _pending_attn_end, _pending_attn_cap
    capturing = torch.cuda.is_current_stream_capturing()
    if _pending or _pending_end:
        if not (capturing and not _pending_cap):
            torch.cuda.current_stream().wait_stream(_side)
        _pending = False
        _pending_end = False
        _pending_cap = False
    if _pending_attn_end:
        if not (capturing and not _pending_attn_cap):
            torch.cuda.current_stream().wait_stream(_side_attn)
        _pending_attn_end = False
        _pending_attn_cap = False
