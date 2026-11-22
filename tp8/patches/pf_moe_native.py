# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) Native Block-Scaled FP4/FP8 MoE Prefill + Fused W2-Combine Loader.

Wave 2A integration module for Inkling-2-Small TP8 (E=384, H=4608, inter=288, topk=8):
  - Strictly gated to prefill chunks (``INKLING_PF_MOE_NATIVE=1``, ``plan.tile >= 64``,
    and ``tokens >= INKLING_PF_MOE_NATIVE_MIN_T`` [default 1024]), leaving C=1 decode
    (T <= 16) 100% untouched and 0-ULP bit-exact.
  - Zero extra weight VRAM: directly consumes live packed ``_inkling_nvfp4_ws5``
    (SLABS / W_PACK+WS5_PACK) OR ``_inkling_panel`` (WP + WSP / WS6) tensors!
  - Eliminates the ``w2_out [plan.rows, 4608]`` allocation (-1.96 GiB at T=16K) and
    7.42 TB of 1M-context DRAM round-trip via nb-outer L2-resident fused W2-Combine.

Environment variables:
  INKLING_PF_MOE_NATIVE=0|1           Enable native SM121a prefill MoE (default: 0)
  INKLING_PF_MOE_NATIVE_LIB=<path>    Path to libpf_moe_native_sm121.so
  INKLING_PF_MOE_NATIVE_MIN_T=1024    Minimum tokens to activate (default: 1024)
  INKLING_PF_MOE_COMBINE_MODE=0|1|2   0=unfused, 1=fused FP32 atomic+cast (default),
                                      2=fused BF16x2 atomic directly into out
  INKLING_PF_MOE_NATIVE_BM=0|64|128   0=auto (64 if tile==64 or max_ctas<=520 else 128)

Exact integration snippets (for Wave 2B wiring into patches-s6/):
  1) In ``trtllm_bmm.py:bmm(...)`` (right after ``nvfp4 = weight.dtype == torch.uint8``):
        if plan.tile >= 64:
            from . import pf_moe_native
            if pf_moe_native.is_active(plan):
                res = pf_moe_native.bmm_pf_native(
                    act, act_scale, row_scale, weight, weight_scale, ones, tiles,
                    plan, projection, row2tok=row2tok, as_row_major=as_row_major,
                )
                if res is not None:
                    return res

  2) In ``moe.py:Inkling2Experts.forward(...)`` (replacing ``down = self._gemm(...w2)`` +
     ``combine(down, ...)`` when ``reduce=True and push is None and plan.tile >= 64``):
        if reduce and push is None and plan.tile >= 64:
            from .ops import pf_moe_native
            if pf_moe_native.is_active(plan, tokens=x.shape[0]) and pf_moe_native.COMBINE_MODE != 0:
                sbuf = None
                if self.ep_size > 1:
                    from vllm.distributed.device_communicators.rdma_all_reduce import get_send_buffer
                    sbuf = get_send_buffer((x.shape[0], self.hidden_size), x.dtype)
                r2t_w2 = getattr(quantized[0], "_inkling_row2tok", None)
                act_v, act_s, row_s = activated
                out = pf_moe_native.w2_combine_pf_native(
                    act_v, act_s, row_s, self.w2_weight, self.w2_weight_scale,
                    self.ones, tiles, plan, src2dst, self.topk, x.dtype,
                    out=sbuf, row2tok=r2t_w2,
                    as_row_major=bool(getattr(act_s, "_inkling_rowmaj", False)),
                )
                if out is not None:
                    if self.ep_size == 1:
                        return out
                    return tensor_model_parallel_all_reduce(out)
"""

from __future__ import annotations

import ctypes
import os
import torch

ENABLED = os.environ.get("INKLING_PF_MOE_NATIVE", "0").strip().lower() in ("1", "true", "on")
MIN_TOKENS = int(os.environ.get("INKLING_PF_MOE_NATIVE_MIN_T", "1024"))
COMBINE_MODE = int(os.environ.get("INKLING_PF_MOE_COMBINE_MODE", "1"))
BM_OVERRIDE = int(os.environ.get("INKLING_PF_MOE_NATIVE_BM", "0"))
USE_MX7 = os.environ.get("INKLING_PF_MOE_NATIVE_MX7", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# Zeroing path for COMBINE_MODE=2 out [T, 4608] bf16 buffer:
#   "memset" (default): cudaMemsetAsync via pf_memset_zero (bypasses L2 WB RFO)
#   "cs":               zero_bf16_cs_kernel (streaming st.global.cs.v4.u32)
#   "torch":            legacy PyTorch out.zero_() (FillFunctor)
ZERO_MODE = os.environ.get("INKLING_PF_MOE_ZERO_MODE", "memset").strip().lower()
# Async pre-zero out [T, 4608] bf16 on a dedicated priority=0 stream BEFORE
# route() + W13 + situ_quantize so zeroing overlaps route/W13 and never evicts
# situ_quantize's w2_act from L2 right before W2! (opt-in, default 0)
ASYNC_ZERO = os.environ.get("INKLING_PF_MOE_ASYNC_ZERO", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
# v3 weight L2 eviction policy: 0=ca (v2.1), 1=cs on last tile per expert (default), 2=cs all
W_EVICT_MODE = int(os.environ.get("INKLING_PF_MOE_W_EVICT", "1"))
# v3/v4 W2 panels/CTA for COMBINE_MODE=2: 2, 3, 4, 6; 0=auto
W2_PANELS = int(os.environ.get("INKLING_PF_MOE_W2_PANELS", "0"))
# v4 Expert-Persistent MoE Prefill (opt-in; eliminates 2-3x weight re-reads & Y_out zero/RFO!)
PERSISTENT = os.environ.get("INKLING_PF_MOE_PERSISTENT", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PERSISTENT_W13 = os.environ.get("INKLING_PF_MOE_PERSISTENT_W13", "1" if PERSISTENT else "0").strip().lower() in (
    "1", "true", "on", "yes",
)
# v4 W13 BM override: 0=auto (256 when tile>=256 else select_bm), 64, 128, or 256
V4_BM_OVERRIDE = int(os.environ.get("INKLING_PF_MOE_V4_BM", "0"))
# v4 weight eviction override: -1=auto (maps W_EVICT=2 -> 1 cs_last for persistent W13), 0=ca, 1=cs_last, 2=cs_all
V4_W_EVICT_OVERRIDE = int(os.environ.get("INKLING_PF_MOE_V4_W_EVICT", "-1"))
# v4 W2 per-slice in-L2 zeroing (COMBINE_MODE=2): eliminates 151 MB memset + 151 MB cold atomic R!
L2_ZERO_SLICE = os.environ.get("INKLING_PF_MOE_L2_ZERO_SLICE", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
# v4 1-block LPT expert plan builder (shared across W13+W2, ~0.6 us, zero host sync)
LPT_PLAN = os.environ.get("INKLING_PF_MOE_LPT_PLAN", "1").strip().lower() not in (
    "0", "", "off", "false", "no",
)
# v4 optional mega-expert split threshold (multiple of 256; 0=no split, 1024=split rare >1024-row experts)
SPLIT_M = int(os.environ.get("INKLING_PF_MOE_SPLIT_M", "1024"))

_LIB = None
_LIB_TRIED = False
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATHS = (
    os.environ.get("INKLING_PF_MOE_NATIVE_LIB", ""),
    "/rdma/libpf_moe_native_sm121_v4.so",
    "/rdma/libpf_moe_native_sm121_v3.so",
    "/rdma/libpf_moe_native_sm121_v21.so",
    "/rdma/libpf_moe_native_sm121_v2.so",
    "/rdma/libpf_moe_native_sm121.so",
    os.path.join(_HERE, "libpf_moe_native_sm121_v4.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121_v3.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121_v21.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121_v2.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121.so"),
)

# Reusable per-device scratch buffers (Y_fp32 is maintained permanently zeroed via
# pf_fp32_to_bf16_zero after every layer so zero extra cudaMemsetAsync is needed).
_Y_FP32_CACHE: dict[torch.device, torch.Tensor] = {}
_R2T_CACHE: dict[torch.device, torch.Tensor] = {}
_PLAN_CACHE: dict[torch.device, torch.Tensor] = {}
_ZERO_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _bind_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_vp = ctypes.c_void_p
    c_i = ctypes.c_int
    c_i32 = ctypes.c_int32
    c_i64 = ctypes.c_int64
    c_u64 = ctypes.c_uint64

    lib.pf_nvfp4_w13_ws5_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w13_ws5_native.restype = c_i

    lib.pf_nvfp4_w13_panel_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w13_panel_native.restype = c_i

    lib.pf_nvfp4_w2_ws5_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w2_ws5_native.restype = c_i

    lib.pf_nvfp4_w2_panel_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w2_panel_native.restype = c_i

    lib.pf_mxfp8_w13_native.argtypes = [
        c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_mxfp8_w13_native.restype = c_i

    lib.pf_mxfp8_w2_native.argtypes = [
        c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_mxfp8_w2_native.restype = c_i

    lib.pf_fp32_to_bf16_zero.argtypes = [c_vp, c_vp, c_i64, c_u64]
    lib.pf_fp32_to_bf16_zero.restype = c_i

    lib.pf_build_row2tok.argtypes = [c_vp, c_vp, c_i32, c_i32, c_u64]
    lib.pf_build_row2tok.restype = c_i

    # Optional v3 entrypoints (L2 weight eviction control, tunable W2 panels, memset/cs zeroing)
    if hasattr(lib, "pf_nvfp4_w13_ws5_native_v3"):
        lib.pf_nvfp4_w13_ws5_native_v3.argtypes = [c_vp] * 11 + [c_i] * 7 + [c_u64]
        lib.pf_nvfp4_w13_ws5_native_v3.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_ws5_native_v3"):
        lib.pf_nvfp4_w2_ws5_native_v3.argtypes = [c_vp] * 12 + [c_i] * 9 + [c_u64]
        lib.pf_nvfp4_w2_ws5_native_v3.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_panel_native_v3"):
        lib.pf_nvfp4_w2_panel_native_v3.argtypes = [c_vp] * 9 + [c_i] * 8 + [c_u64]
        lib.pf_nvfp4_w2_panel_native_v3.restype = c_i
    if hasattr(lib, "pf_mxfp8_w2_native_v3"):
        lib.pf_mxfp8_w2_native_v3.argtypes = [c_vp] * 8 + [c_i] * 8 + [c_u64]
        lib.pf_mxfp8_w2_native_v3.restype = c_i
    if hasattr(lib, "pf_memset_zero"):
        lib.pf_memset_zero.argtypes = [c_vp, c_i64, c_u64]
        lib.pf_memset_zero.restype = c_i
    if hasattr(lib, "pf_zero_cs_bf16"):
        lib.pf_zero_cs_bf16.argtypes = [c_vp, c_i64, c_u64]
        lib.pf_zero_cs_bf16.restype = c_i
    # Optional v4 Expert-Persistent entrypoints
    if hasattr(lib, "pf_build_expert_plan"):
        lib.pf_build_expert_plan.argtypes = [c_vp] * 2 + [c_i] * 5 + [c_u64]
        lib.pf_build_expert_plan.restype = c_i
    if hasattr(lib, "pf_nvfp4_w13_ws5_native_v4"):
        lib.pf_nvfp4_w13_ws5_native_v4.argtypes = [c_vp] * 12 + [c_i] * 9 + [c_u64]
        lib.pf_nvfp4_w13_ws5_native_v4.restype = c_i
    if hasattr(lib, "pf_nvfp4_w13_panel_native_v4"):
        lib.pf_nvfp4_w13_panel_native_v4.argtypes = [c_vp] * 10 + [c_i] * 9 + [c_u64]
        lib.pf_nvfp4_w13_panel_native_v4.restype = c_i
    if hasattr(lib, "pf_mxfp8_w13_native_v4"):
        lib.pf_mxfp8_w13_native_v4.argtypes = [c_vp] * 9 + [c_i] * 8 + [c_u64]
        lib.pf_mxfp8_w13_native_v4.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_ws5_native_v4"):
        lib.pf_nvfp4_w2_ws5_native_v4.argtypes = [c_vp] * 13 + [c_i] * 12 + [c_u64]
        lib.pf_nvfp4_w2_ws5_native_v4.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_panel_native_v4"):
        lib.pf_nvfp4_w2_panel_native_v4.argtypes = [c_vp] * 10 + [c_i] * 11 + [c_u64]
        lib.pf_nvfp4_w2_panel_native_v4.restype = c_i
    if hasattr(lib, "pf_mxfp8_w2_native_v4"):
        lib.pf_mxfp8_w2_native_v4.argtypes = [c_vp] * 9 + [c_i] * 11 + [c_u64]
        lib.pf_mxfp8_w2_native_v4.restype = c_i
    if hasattr(lib, "pf_zero_slice_wb_bf16"):
        lib.pf_zero_slice_wb_bf16.argtypes = [c_vp] + [c_i] * 3 + [c_u64]
        lib.pf_zero_slice_wb_bf16.restype = c_i
    return lib


def _get_mx7_v5_lib() -> ctypes.CDLL | None:
    if not USE_MX7:
        return None
    from . import trtllm_bmm
    lib = trtllm_bmm._get_mx7_lib()
    if lib is not None and getattr(trtllm_bmm, "_MX7_HAS_NATIVE_V5", False):
        return lib
    return None


def _get_mx7_v6_lib() -> ctypes.CDLL | None:
    if not USE_MX7:
        return None
    from . import trtllm_bmm
    lib = trtllm_bmm._get_mx7_lib()
    if lib is not None and getattr(trtllm_bmm, "_MX7_HAS_NATIVE_V6", False):
        return lib
    return None


def get_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB, _LIB_TRIED
    if path is not None:
        if os.path.isfile(path):
            _LIB = _bind_lib(ctypes.CDLL(path))
            _LIB_TRIED = True
            _v3 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v3"))
            _v4 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v4"))
            print(
                f"[inkling-pf-moe-native] loaded {path} "
                f"(v3={_v3}, v4={_v4}, persist={int(PERSISTENT)}, l2_zero={int(L2_ZERO_SLICE)}, "
                f"cm={COMBINE_MODE}, zero={ZERO_MODE}, async_zero={int(ASYNC_ZERO)}, "
                f"w_evict={W_EVICT_MODE}, w2_panels={W2_PANELS})",
                flush=True,
            )
            return _LIB
        return None
    if _LIB is not None:
        return _LIB
    if _LIB_TRIED:
        return None
    _LIB_TRIED = True
    for p in _LIB_PATHS:
        if p and os.path.isfile(p):
            _LIB = _bind_lib(ctypes.CDLL(p))
            _v3 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v3"))
            _v4 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v4"))
            print(
                f"[inkling-pf-moe-native] loaded {p} "
                f"(v3={_v3}, v4={_v4}, persist={int(PERSISTENT)}, l2_zero={int(L2_ZERO_SLICE)}, "
                f"cm={COMBINE_MODE}, zero={ZERO_MODE}, async_zero={int(ASYNC_ZERO)}, "
                f"w_evict={W_EVICT_MODE}, w2_panels={W2_PANELS})",
                flush=True,
            )
            return _LIB
    return None


def is_active(plan, tokens: int | None = None) -> bool:
    """Strictly gate native SM121a prefill MoE to long prefill chunks (never decode)."""
    if not ENABLED:
        return False
    if plan.tile < 64:
        return False
    if tokens is not None and tokens < MIN_TOKENS:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    # When tokens is omitted (inside trtllm_bmm.bmm), plan.tile >= 128 guarantees T >= 769;
    # enforce MIN_TOKENS via plan.max_ctas lower bound if MIN_TOKENS >= 1024.
    if tokens is None and MIN_TOKENS > 768 and plan.tile < 128:
        return False
    return get_lib() is not None


def select_bm(plan, bm: int | None = None) -> int:
    if bm is not None and bm in (64, 128) and plan.tile % bm == 0:
        return bm
    if BM_OVERRIDE in (64, 128) and plan.tile % BM_OVERRIDE == 0:
        return BM_OVERRIDE
    if plan.tile == 64 or plan.max_ctas <= 520:
        return 64
    return 128


def select_bm_v4(plan, bm: int | None = None) -> int:
    if bm is not None and bm in (64, 128, 256):
        return bm
    if V4_BM_OVERRIDE in (64, 128, 256):
        return V4_BM_OVERRIDE
    if BM_OVERRIDE in (64, 128):
        return BM_OVERRIDE
    if plan.tile >= 256:
        return 256
    if plan.tile == 64 or plan.max_ctas <= 520:
        return 64
    return 128


def select_v4_w_evict(w_evict_mode: int | None = None) -> int:
    if w_evict_mode is not None:
        return w_evict_mode
    if V4_W_EVICT_OVERRIDE >= 0:
        return V4_W_EVICT_OVERRIDE
    return 0 if W_EVICT_MODE == 0 else 1


def select_w2_panels(
    tokens: int,
    combine_mode: int,
    w2_panels: int | None = None,
    *,
    persistent: bool = False,
) -> int:
    if combine_mode != 2:
        return 2 if combine_mode == 1 else 6
    p = W2_PANELS if w2_panels is None else w2_panels
    if p in (2, 3, 4, 6):
        if p == 6 and not persistent:
            return 4
        return p
    if persistent:
        # Persistent W2: 4 panels @ T<=8192 (8 MiB Y_out + 10.6 MiB A = 18.6 MiB < 24 MiB L2),
        # 3 panels @ T>8192 (12 MiB Y_out + 5.3 MiB active A + serpentine = 17.3 MiB < 24 MiB L2).
        return 4 if tokens <= 8192 else 3
    # Auto (v3): 4 panels @ T<=8192 (8 MiB Y_out slice fits 24 MiB L2 cleanly),
    # 2 panels @ T>8192 (keeps Y_out slice <= 6 MiB @ T=12K inside 24 MiB L2).
    return 4 if tokens <= 8192 else 2


def _get_plan_buf(need_i32: int, device: torch.device) -> torch.Tensor:
    buf = _PLAN_CACHE.get(device)
    if buf is None or buf.numel() < need_i32:
        buf = torch.empty(need_i32, dtype=torch.int32, device=device)
        _PLAN_CACHE[device] = buf
    return buf[:need_i32]


def get_or_build_expert_plan(
    tiles: torch.Tensor,
    plan,
    num_experts: int,
    lib: ctypes.CDLL | None,
    stream: int,
    *,
    split_m: int | None = None,
) -> tuple[int, int]:
    """Build (once per layer) the 1-block LPT expert work plan for persistent W13 & W2.

    Returns ``(plan_buf_ptr, max_work)``; returns ``(0, num_experts)`` if disabled or unavailable.
    """
    if not LPT_PLAN or lib is None or not hasattr(lib, "pf_build_expert_plan") or num_experts > 1024:
        return 0, num_experts
    cached = getattr(tiles, "_inkling_exp_plan", None)
    if cached is not None and cached[2] == num_experts:
        return cached[0], cached[1]
    sm = SPLIT_M if split_m is None else split_m
    eff_split = ((sm + 255) // 256) * 256 if sm > 0 else 0
    max_work = (num_experts + (plan.max_ctas * plan.tile) // eff_split + 1) if eff_split > 0 else num_experts
    buf = _get_plan_buf(3 * max_work + 1, tiles.device)
    rc = lib.pf_build_expert_plan(
        tiles.data_ptr(), buf.data_ptr(),
        plan.max_ctas, plan.tile, num_experts, eff_split, max_work, stream,
    )
    if rc != 0:
        return 0, num_experts
    ptr = buf.data_ptr()
    try:
        tiles._inkling_exp_plan = (ptr, max_work, num_experts)  # type: ignore[attr-defined]
    except Exception:
        pass
    return ptr, max_work


def _get_zero_stream(device: torch.device) -> torch.cuda.Stream:
    zs = _ZERO_STREAMS.get(device)
    if zs is None:
        zs = torch.cuda.Stream(device=device, priority=0)
        _ZERO_STREAMS[device] = zs
    return zs


def zero_bf16_out(
    out: torch.Tensor,
    *,
    mode: str | None = None,
    lib: ctypes.CDLL | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Zero a contiguous `[T, H]` bf16 output tensor without polluting L2 cache."""
    m = ZERO_MODE if mode is None else mode
    lib = lib or get_lib()
    st_obj = stream if stream is not None else torch.cuda.current_stream(out.device)
    st = int(st_obj.cuda_stream)
    if m == "memset" and lib is not None and hasattr(lib, "pf_memset_zero"):
        rc = lib.pf_memset_zero(out.data_ptr(), out.numel() * out.element_size(), st)
        if rc == 0:
            return
    elif m == "cs" and lib is not None and hasattr(lib, "pf_zero_cs_bf16"):
        rc = lib.pf_zero_cs_bf16(out.data_ptr(), out.numel(), st)
        if rc == 0:
            return
    with torch.cuda.stream(st_obj):
        out.zero_()


def alloc_and_async_zero_out(
    tokens: int,
    hidden: int,
    device: torch.device,
    compute_stream: torch.cuda.Stream | None = None,
    lib: ctypes.CDLL | None = None,
) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Allocate `[tokens, hidden]` bf16 on `compute_stream` and asynchronously zero it on
    a dedicated priority=0 `zero_stream` so zeroing overlaps `route()` + `W13` and does
    not evict `situ_quantize_288`'s `w2_act` from L2 right before W2!"""
    cs = compute_stream if compute_stream is not None else torch.cuda.current_stream(device)
    zs = _get_zero_stream(device)
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    ev_alloc = torch.cuda.Event()
    ev_alloc.record(cs)
    zs.wait_event(ev_alloc)
    out.record_stream(zs)
    zero_bf16_out(out, lib=lib, stream=zs)
    ev_done = torch.cuda.Event()
    ev_done.record(zs)
    return out, ev_done


def _get_y_fp32(tokens: int, hidden: int, device: torch.device) -> torch.Tensor:
    need = tokens * hidden
    buf = _Y_FP32_CACHE.get(device)
    if buf is None or buf.numel() < need:
        buf = torch.zeros(need, dtype=torch.float32, device=device)
        _Y_FP32_CACHE[device] = buf
    return buf[:need].view(tokens, hidden)


def _get_r2t_buf(rows: int, device: torch.device) -> torch.Tensor:
    buf = _R2T_CACHE.get(device)
    if buf is None or buf.numel() < rows:
        buf = torch.empty(rows, dtype=torch.int32, device=device)
        _R2T_CACHE[device] = buf
    return buf[:rows]


def build_row2tok(
    src2dst: torch.Tensor,
    rows: int,
    topk: int = 8,
    out: torch.Tensor | None = None,
    lib: ctypes.CDLL | None = None,
) -> torch.Tensor:
    lib = lib or get_lib()
    assert lib is not None, "libpf_moe_native_sm121.so not loaded"
    tokens = src2dst.numel() // topk
    r2t = out if out is not None else _get_r2t_buf(rows, src2dst.device)
    st = int(torch.cuda.current_stream(src2dst.device).cuda_stream)
    rc = lib.pf_build_row2tok(src2dst.data_ptr(), r2t.data_ptr(), tokens, topk, st)
    assert rc == 0, f"pf_build_row2tok CUDA error {rc}"
    return r2t


def bmm_pf_native(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan,
    projection: int,
    *,
    row2tok: torch.Tensor | None = None,
    as_row_major: bool = False,
    bm: int | None = None,
    w_evict_mode: int | None = None,
    persistent: bool | None = None,
    out: torch.Tensor | None = None,
    lib: ctypes.CDLL | None = None,
) -> torch.Tensor | None:
    """Drop-in native SM121a prefill GEMM (unfused output [plan.rows, N] bf16).

    Returns None (so caller falls back cleanly) if weight layout or shape is unsupported.
    """
    lib = lib or get_lib()
    if lib is None or plan.tile < 64 or plan.tile % 64 != 0:
        return None

    nvfp4 = weight.dtype == torch.uint8
    pack = 2 if nvfp4 else 1
    num_experts = int(weight.shape[0])
    n = weight.shape[1]
    k = weight.shape[2] * pack
    if not ((projection == 0 and n == 576 and k == 4608) or (projection == 1 and n == 4608 and k == 288)):
        return None

    use_persist = (PERSISTENT_W13 if projection == 0 else PERSISTENT) if persistent is None else persistent
    bm_sel = select_bm(plan, bm)
    bm_v4 = select_bm_v4(plan, bm)
    w_evict = W_EVICT_MODE if w_evict_mode is None else w_evict_mode
    w_evict_v4 = select_v4_w_evict(w_evict_mode)
    if out is None:
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    st = int(torch.cuda.current_stream(act.device).cuda_stream)

    if nvfp4:
        assert row_scale is not None
        ws5 = getattr(weight, "_inkling_nvfp4_ws5", None)
        if projection == 0:
            use_r2t = 1 if row2tok is not None else 0
            r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
            if ws5 is not None and ws5.get("proj") == 0:
                if use_persist and hasattr(lib, "pf_nvfp4_w13_ws5_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w13_ws5_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, use_r2t,
                        1 if as_row_major else 0, 1 if ws5.get("has_patches", True) else 0,
                        w_evict_v4, st,
                    )
                elif hasattr(lib, "pf_nvfp4_w13_ws5_native_v3"):
                    rc = lib.pf_nvfp4_w13_ws5_native_v3(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, use_r2t,
                        1 if as_row_major else 0, 1 if ws5.get("has_patches", True) else 0,
                        w_evict, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w13_ws5_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, use_r2t,
                        1 if as_row_major else 0, 1 if ws5.get("has_patches", True) else 0, st,
                    )
            elif (
                getattr(weight, "_inkling_panel", None) == (32, 128)
                and getattr(weight_scale, "_inkling_panel", None) == (32, 16)
                and weight.data.untyped_storage().size() > 0
                and weight_scale.data.untyped_storage().size() > 0
            ):
                ws6 = 1 if bool(getattr(weight_scale, "_inkling_ws6", False)) else 0
                if use_persist and hasattr(lib, "pf_nvfp4_w13_panel_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w13_panel_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, use_r2t,
                        1 if as_row_major else 0, ws6, w_evict_v4, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w13_panel_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, use_r2t,
                        1 if as_row_major else 0, ws6, st,
                    )
            else:
                return None
        else:
            as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 18
            if ws5 is not None and ws5.get("proj") == 1:
                if use_persist and hasattr(lib, "pf_nvfp4_w2_ws5_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w2_ws5_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w,
                        1 if as_row_major else 0, as_stride,
                        1 if ws5.get("has_patches", True) else 0, 0, w_evict_v4, 6, 0, 0, st,
                    )
                elif hasattr(lib, "pf_nvfp4_w2_ws5_native_v3"):
                    rc = lib.pf_nvfp4_w2_ws5_native_v3(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        1 if ws5.get("has_patches", True) else 0, 0, w_evict, 6, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w2_ws5_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        1 if ws5.get("has_patches", True) else 0, 0, st,
                    )
            elif (
                getattr(weight, "_inkling_panel", None) == (256, 16)
                and getattr(weight_scale, "_inkling_panel", None) == (256, 2)
                and weight.data.untyped_storage().size() > 0
                and weight_scale.data.untyped_storage().size() > 0
            ):
                if use_persist and hasattr(lib, "pf_nvfp4_w2_panel_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w2_panel_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w,
                        1 if as_row_major else 0, as_stride,
                        0, w_evict_v4, 6, 0, 0, st,
                    )
                elif hasattr(lib, "pf_nvfp4_w2_panel_native_v3"):
                    rc = lib.pf_nvfp4_w2_panel_native_v3(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        0, w_evict, 6, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w2_panel_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        0, st,
                    )
            else:
                return None
        assert rc == 0, f"pf_nvfp4 native CUDA error {rc} (proj={projection})"
        return out

    # MXFP8 path: prefer v6 Expert-Persistent / v5 SM121a native block-scaled FP8 on MX7 slabs
    mx7 = getattr(weight, "_inkling_mx7", None)
    if isinstance(mx7, tuple):
        mx7_v6_lib = _get_mx7_v6_lib() if use_persist else None
        if mx7_v6_lib is not None:
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
            if projection == 0:
                use_r2t = 1 if row2tok is not None else 0
                r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
                rc = mx7_v6_lib.mx7_v6_prefill_native_w13(
                    act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, base_ws,
                    use_r2t, 1 if as_row_major else 0, w_evict_v4, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
                rc = mx7_v6_lib.mx7_v6_prefill_native_w2(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w, base_ws,
                    1 if as_row_major else 0, as_stride, 0, w_evict_v4, 6, 0, 0, st,
                )
            assert rc == 0, f"mx7_v6_prefill_native CUDA error {rc} (proj={projection})"
            return out
        mx7_lib = _get_mx7_v5_lib()
        if mx7_lib is not None:
            slabs, ws3, flips, f_off, base_ws = mx7
            if projection == 0:
                use_r2t = 1 if row2tok is not None else 0
                r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
                rc = mx7_lib.mx7_v5_prefill_native_w13(
                    act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                    plan.max_ctas, plan.tile, bm_sel, base_ws,
                    use_r2t, 1 if as_row_major else 0, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
                rc = mx7_lib.mx7_v5_prefill_native_w2(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                    plan.max_ctas, plan.tile, bm_sel, base_ws,
                    1 if as_row_major else 0, as_stride, 0, st,
                )
            assert rc == 0, f"mx7_v5_prefill_native CUDA error {rc} (proj={projection})"
            return out

    # Raw EXPERT_PANEL MXFP8 fallback (returns None cleanly if INKLING_MX7_FREE_RAW=1 freed raw storage)
    if weight.data.untyped_storage().size() == 0 or weight_scale.data.untyped_storage().size() == 0:
        return None
    if projection == 0:
        if getattr(weight, "_inkling_panel", None) != (32, 512) or getattr(weight_scale, "_inkling_panel", None) != (32, 16):
            return None
        use_r2t = 1 if row2tok is not None else 0
        r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
        if use_persist and hasattr(lib, "pf_mxfp8_w13_native_v4"):
            p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
            rc = lib.pf_mxfp8_w13_native_v4(
                act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                weight.data_ptr(), weight_scale.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, use_r2t,
                1 if as_row_major else 0, w_evict_v4, st,
            )
        else:
            rc = lib.pf_mxfp8_w13_native(
                act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                weight.data_ptr(), weight_scale.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                plan.max_ctas, plan.tile, bm_sel, use_r2t, 1 if as_row_major else 0, st,
            )
    else:
        if getattr(weight, "_inkling_panel", None) != (128, 32) or getattr(weight_scale, "_inkling_panel", None) != (128, 1):
            return None
        as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
        if use_persist and hasattr(lib, "pf_mxfp8_w2_native_v4"):
            p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
            rc = lib.pf_mxfp8_w2_native_v4(
                act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                weight.data_ptr(), weight_scale.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                plan.max_ctas, plan.tile, num_experts, max_w,
                1 if as_row_major else 0, as_stride, 0, w_evict_v4, 6, 0, 0, st,
            )
        elif hasattr(lib, "pf_mxfp8_w2_native_v3"):
            rc = lib.pf_mxfp8_w2_native_v3(
                act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                weight.data_ptr(), weight_scale.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                0, w_evict, 6, st,
            )
        else:
            rc = lib.pf_mxfp8_w2_native(
                act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                weight.data_ptr(), weight_scale.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                0, st,
            )
    assert rc == 0, f"pf_mxfp8 native CUDA error {rc} (proj={projection})"
    return out


def w2_combine_pf_native(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan,
    src2dst: torch.Tensor,
    topk: int = 8,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    out: torch.Tensor | None = None,
    out_prezeroed: bool = False,
    row2tok: torch.Tensor | None = None,
    as_row_major: bool = False,
    combine_mode: int | None = None,
    bm: int | None = None,
    w_evict_mode: int | None = None,
    w2_panels: int | None = None,
    persistent: bool | None = None,
    l2_zero_slice: bool | None = None,
    lib: ctypes.CDLL | None = None,
) -> torch.Tensor | None:
    """Fused SM121a W2 GEMM + L2-resident Combine epilogue -> [T, 4608] bf16.

    Eliminates ``w2_out [plan.rows, 4608]`` (-1.96 GiB VRAM at T=16K) and ``combine()``.
    - ``combine_mode == 1`` (default): nb-outer FP32 atomic into reusable pre-zeroed
      ``[T, 4608]`` fp32 scratch, followed by vectorized ``pf_fp32_to_bf16_zero``.
    - ``combine_mode == 2``: zeroes ``out [T, 4608]`` bf16 (or per-slice in-L2 zeroing
      when ``persistent=True`` and ``l2_zero_slice=True``) and runs nb-outer BF16x2
      atomic directly into ``out``.
    """
    lib = lib or get_lib()
    cm = COMBINE_MODE if combine_mode is None else combine_mode
    if lib is None or cm not in (1, 2) or out_dtype != torch.bfloat16:
        return None
    if plan.tile < 64 or plan.tile % 64 != 0:
        return None

    nvfp4 = weight.dtype == torch.uint8
    pack = 2 if nvfp4 else 1
    num_experts = int(weight.shape[0])
    n = weight.shape[1]
    k = weight.shape[2] * pack
    if not (n == 4608 and k == 288):
        return None

    use_persist = PERSISTENT if persistent is None else persistent
    use_l2_zero = L2_ZERO_SLICE if l2_zero_slice is None else l2_zero_slice
    tokens = src2dst.numel() // topk
    bm_sel = select_bm(plan, bm)
    w_evict = W_EVICT_MODE if w_evict_mode is None else w_evict_mode
    w_evict_v4 = select_v4_w_evict(w_evict_mode)
    device = act.device
    st = int(torch.cuda.current_stream(device).cuda_stream)

    # Determine upfront whether a v4/v6 persistent handler will run (so l2_zero_slice only skips
    # pre-zeroing when the selected kernel actually performs per-slice in-L2 zeroing!).
    ws5 = getattr(weight, "_inkling_nvfp4_ws5", None) if nvfp4 else None
    mx7 = getattr(weight, "_inkling_mx7", None) if not nvfp4 else None
    mx7_v6_lib = (_get_mx7_v6_lib() if (isinstance(mx7, tuple) and use_persist) else None)
    can_persist_handler = False
    if use_persist:
        if nvfp4:
            if ws5 is not None and ws5.get("proj") == 1:
                can_persist_handler = hasattr(lib, "pf_nvfp4_w2_ws5_native_v4")
            elif getattr(weight, "_inkling_panel", None) == (256, 16):
                can_persist_handler = hasattr(lib, "pf_nvfp4_w2_panel_native_v4")
        else:
            if mx7_v6_lib is not None:
                can_persist_handler = True
            elif not isinstance(mx7, tuple) and getattr(weight, "_inkling_panel", None) == (128, 32):
                can_persist_handler = hasattr(lib, "pf_mxfp8_w2_native_v4")

    w2_p = select_w2_panels(tokens, cm, w2_panels, persistent=can_persist_handler)
    do_l2_slice = int(cm == 2 and can_persist_handler and use_l2_zero and tokens > 0)

    if row2tok is None:
        row2tok = build_row2tok(src2dst, plan.rows, topk=topk, lib=lib)

    if out is None:
        out = torch.empty((tokens, n), dtype=torch.bfloat16, device=device)
        out_prezeroed = False

    if cm == 1:
        y_accum = _get_y_fp32(tokens, n, device)
        out_ptr = y_accum.data_ptr()
    else:
        if not out_prezeroed and not do_l2_slice:
            zero_bf16_out(out, lib=lib)
        y_accum = None
        out_ptr = out.data_ptr()

    as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else (18 if nvfp4 else 9)

    if nvfp4:
        assert row_scale is not None
        if ws5 is not None and ws5.get("proj") == 1:
            if can_persist_handler:
                p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                rc = lib.pf_nvfp4_w2_ws5_native_v4(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                    plan.max_ctas, plan.tile, num_experts, max_w,
                    1 if as_row_major else 0, as_stride,
                    1 if ws5.get("has_patches", True) else 0, cm, w_evict_v4, w2_p,
                    do_l2_slice, tokens, st,
                )
            elif hasattr(lib, "pf_nvfp4_w2_ws5_native_v3"):
                rc = lib.pf_nvfp4_w2_ws5_native_v3(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    1 if ws5.get("has_patches", True) else 0, cm, w_evict, w2_p, st,
                )
            else:
                rc = lib.pf_nvfp4_w2_ws5_native(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    1 if ws5.get("has_patches", True) else 0, cm, st,
                )
        elif (
            getattr(weight, "_inkling_panel", None) == (256, 16)
            and getattr(weight_scale, "_inkling_panel", None) == (256, 2)
            and weight.data.untyped_storage().size() > 0
            and weight_scale.data.untyped_storage().size() > 0
        ):
            if can_persist_handler:
                p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                rc = lib.pf_nvfp4_w2_panel_native_v4(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    weight.data_ptr(), weight_scale.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                    plan.max_ctas, plan.tile, num_experts, max_w,
                    1 if as_row_major else 0, as_stride,
                    cm, w_evict_v4, w2_p, do_l2_slice, tokens, st,
                )
            elif hasattr(lib, "pf_nvfp4_w2_panel_native_v3"):
                rc = lib.pf_nvfp4_w2_panel_native_v3(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    weight.data_ptr(), weight_scale.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    cm, w_evict, w2_p, st,
                )
            else:
                rc = lib.pf_nvfp4_w2_panel_native(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    weight.data_ptr(), weight_scale.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    cm, st,
                )
        else:
            return None
    else:
        if mx7_v6_lib is not None and isinstance(mx7, tuple):
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
            rc = mx7_v6_lib.mx7_v6_prefill_native_w2(
                act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                plan.max_ctas, plan.tile, num_experts, max_w, base_ws,
                1 if as_row_major else 0, as_stride,
                cm, w_evict_v4, w2_p, do_l2_slice, tokens, st,
            )
        else:
            mx7_lib = _get_mx7_v5_lib() if isinstance(mx7, tuple) else None
            if mx7_lib is not None:
                slabs, ws3, flips, f_off, base_ws = mx7
                rc = mx7_lib.mx7_v5_prefill_native_w2(
                    act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, base_ws,
                    1 if as_row_major else 0, as_stride, cm, st,
                )
            else:
                if (
                    weight.data.untyped_storage().size() == 0
                    or weight_scale.data.untyped_storage().size() == 0
                    or getattr(weight, "_inkling_panel", None) != (128, 32)
                    or getattr(weight_scale, "_inkling_panel", None) != (128, 1)
                ):
                    return None
                if can_persist_handler:
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_mxfp8_w2_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                        plan.max_ctas, plan.tile, num_experts, max_w,
                        1 if as_row_major else 0, as_stride,
                        cm, w_evict_v4, w2_p, do_l2_slice, tokens, st,
                    )
                elif hasattr(lib, "pf_mxfp8_w2_native_v3"):
                    rc = lib.pf_mxfp8_w2_native_v3(
                        act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out_ptr,
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        cm, w_evict, w2_p, st,
                    )
                else:
                    rc = lib.pf_mxfp8_w2_native(
                        act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out_ptr,
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        cm, st,
                    )
    assert rc == 0, f"w2_combine_pf_native CUDA error {rc} (nvfp4={nvfp4}, cm={cm})"

    if cm == 1:
        rc2 = lib.pf_fp32_to_bf16_zero(y_accum.data_ptr(), out.data_ptr(), tokens * n, st)
        assert rc2 == 0, f"pf_fp32_to_bf16_zero CUDA error {rc2}"

    return out
