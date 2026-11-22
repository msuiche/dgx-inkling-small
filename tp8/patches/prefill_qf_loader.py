# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP CUDA fast path loader for Prefill & High-Concurrency MoE scaffolding
and short-conv (`libprefill_moe_qf.so`).

Install path: vllm/models/inkling2/ops/prefill_qf.py
Env knobs:
  INKLING_PREFILL_QF=0              Enable 0-ULP CUDA scatter/situ/combine/sconv for T > 16 (opt-in, default 0)
  INKLING_PREFILL_QF_LIB=/rdma/libprefill_moe_qf.so
  INKLING_MOE_ROWMAJ_SCALES=0       Row-major QScales (TILE=0); requires matching trtllm_bmm.py
  INKLING_MOE_ZERO_SCATTER=0        Zero-scatter W13 (Row2Tok); requires matching trtllm_bmm.py
"""
from __future__ import annotations

import ctypes
import os

ENABLED = os.environ.get("INKLING_PREFILL_QF", "0").strip().lower() not in ("0", "", "off", "false", "no")
LIB_PATH = os.environ.get("INKLING_PREFILL_QF_LIB", "/rdma/libprefill_moe_qf.so")
ROWMAJ_SCALES = os.environ.get("INKLING_MOE_ROWMAJ_SCALES", "0").strip().lower() in ("1", "on", "true")
ZERO_SCATTER = os.environ.get("INKLING_MOE_ZERO_SCATTER", "0").strip().lower() in ("1", "on", "true")

_LIB = None
_TRIED = False


def get_lib():
    global _LIB, _TRIED
    if not ENABLED:
        return None
    if not _TRIED:
        _TRIED = True
        if os.path.exists(LIB_PATH):
            try:
                lib = ctypes.CDLL(LIB_PATH)
                vp, ci, ci64, cf, cu64 = (
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int64,
                    ctypes.c_float,
                    ctypes.c_uint64,
                )
                lib.launch_nvfp4_scatter.argtypes = [vp] * 14 + [ci] * 6 + [cu64]
                lib.launch_nvfp4_scatter.restype = ci
                lib.launch_mxfp8_scatter.argtypes = [vp] * 11 + [ci] * 6 + [cu64]
                lib.launch_mxfp8_scatter.restype = ci
                if hasattr(lib, "launch_nvfp4_scatter_fp8"):
                    lib.launch_nvfp4_scatter_fp8.argtypes = [vp] * 15 + [ci] * 6 + [cu64]
                    lib.launch_nvfp4_scatter_fp8.restype = ci
                if hasattr(lib, "launch_mxfp8_scatter_fp8"):
                    lib.launch_mxfp8_scatter_fp8.argtypes = [vp] * 12 + [ci] * 6 + [cu64]
                    lib.launch_mxfp8_scatter_fp8.restype = ci
                lib.launch_situ_quantize_288.argtypes = [vp] * 10 + [ci] * 5 + [cu64]
                lib.launch_situ_quantize_288.restype = ci
                lib.launch_combine_v2.argtypes = [vp] * 3 + [ci, ci, ci, cu64]
                lib.launch_combine_v2.restype = ci
                lib.launch_short_conv_v2.argtypes = (
                    [vp] * 11 + [ci] + [ci64] * 6 + [ci] * 4 + [cf, ci, ci, ci, cu64]
                )
                lib.launch_short_conv_v2.restype = ci
                _LIB = lib
                _fp8_dir = int(
                    hasattr(lib, "launch_nvfp4_scatter_fp8")
                    and hasattr(lib, "launch_mxfp8_scatter_fp8")
                )
                print(
                    f"[inkling-prefill-qf] loaded 0-ULP SM121a kernels: {LIB_PATH} "
                    f"(rowmaj={int(ROWMAJ_SCALES)}, zero_scatter={int(ZERO_SCATTER)}, fp8_direct={_fp8_dir})",
                    flush=True,
                )
            except Exception as exc:
                print(f"[inkling-prefill-qf] load failed ({exc!r}); using Triton", flush=True)
                _LIB = None
    return _LIB
