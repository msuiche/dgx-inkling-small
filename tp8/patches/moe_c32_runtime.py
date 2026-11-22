# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) C=32 (B=32, M=17..256) 0-ULP Bit-Exact Target 48L MoE Runtime.

Opt-in orchestration module for the C=32 3-library Target MoE stack (default OFF;
strictly guarded by `17 <= M <= 256` so C=1 `M <= 16` and `<120s` 1M prefill
`M > 256` are 100% untouched):

  1. `librouter_bf12_c32.so`:
     - `router_bf12_c32_gemm_fwd`: SMEM-tiled BF12 router GEMM reading existing
       `router_bf12` slabs (2.654 MB vs 3.539 MB BF16, -25.0% bytes, 0 extra VRAM).
     - `router_c32_quant_route_place_fwd`: 2-phase fused Zero-Scatter quantize +
       Top-8 + last-CTA 6-warp prefix scan + placement (`Row2Tok`, `TileCompactStart`,
       `Src2Compact`, pre-zeroed compact `RowAmaxW2`). Supports BOTH BF16 logits
       (from `k_group_partials` for literal `nd == 0` vs live `patches-s6` on
       `exact_ink.py` & `bench.py`) AND FP32 logits (from `router_bf12_c32_gemm_fwd`).
  2. `libnvfp4_ws5_c32.so`:
     - Native single-pass `BM=16` (`TILE=16`, `BK_DOT=256/32`) & `BM=32` (`TILE=32`,
       `BK_DOT=128/64` + tail zero-pad) 0-ULP WS5 NVFP4 W13/W2 kernels.
     - Dynamic `has_m8/has_m16/has_m24` row predication, `Row2Tok` Zero-Scatter,
       `WS_BASE` bit-7 `P_OFF` bypass, 0-way SMEM banking, Fused W13 SiTU -> compact
       `SiTU_Out` + W2 OTF strictly-unfused NVFP4 quant-dequant, and 6-panel-wave
       double-buffered compact L2-resident `DownPanel` combine -> `SendBuf`.
  3. `libmxfp8_mx7_c32.so`:
     - Native single-pass `BM=16` & `BM=32` 0-ULP BF16-MMA MX7 (7.28 bpw) W13/W2
       kernels (upgraded `BN=128, 4 warps` W2 with 1-to-1 zero-repack bijection
       onto existing MX7 slabs -> 2x less `sA` work, 4x more warps/SM than v4!).
     - Closes the live `INKLING_MX7_FREE_RAW=0` TILE=32 MX7 skip bug and shares the
       exact same 6-panel-wave L2-resident `DownPanel` combine -> `SendBuf`.

1-Knob Rollout Controls (ALL default OFF unless `INKLING_C32_MOE=1`):
  - `INKLING_C32_MOE=1`                      : master opt-in switch
  - `INKLING_C32_MOE_MIN_M=17`               : strict lower M guard (never touches M<=16)
  - `INKLING_C32_MOE_MAX_M=256`              : strict upper M guard (never touches prefill)
  - `INKLING_C32_ROUTER_MODE=route_only|bf12|off`
                                             : `route_only` (default, 0-ULP vs cuBLAS BF16)
                                               or `bf12` (-25% router weight bytes)
  - `INKLING_C32_ROUTER_BF12_CAPTURE_ONLY=1` : when `ROUTER_MODE=bf12`, keep eager prefill
                                               prompts on `route_only` (`exact_ink.py` 0-ULP!)
  - `INKLING_C32_FUSE_SITU=1`                : fuse W13 SiTU + W2 OTF quant (-48 launches)
  - `INKLING_C32_FUSE_COMBINE=1`             : 6-wave L2-resident fused combine (-48 launches)
  - `INKLING_C32_MX7_TILE32=1`               : enable MX7 at TILE=32 even when FREE_RAW=0
  - `INKLING_C32_PDL=2`                      : PDL mode (`0`, `1`, `2`)
"""

from __future__ import annotations

import ctypes
import os
import torch


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "", "off", "false", "no")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


C32_MASTER = _env_flag("INKLING_C32_MOE", False)
C32_MIN_M = max(17, _env_int("INKLING_C32_MOE_MIN_M", 17))
C32_MAX_M = min(256, _env_int("INKLING_C32_MOE_MAX_M", 256))

_raw_rmode = os.environ.get("INKLING_C32_ROUTER_MODE", "").strip().lower()
if _raw_rmode in ("route_only", "bf12", "off"):
    C32_ROUTER_MODE = _raw_rmode
else:
    C32_ROUTER_MODE = "route_only" if C32_MASTER else "off"

C32_ROUTER_BF12_CAPTURE_ONLY = _env_flag("INKLING_C32_ROUTER_BF12_CAPTURE_ONLY", True)
C32_FUSE_SITU = _env_flag("INKLING_C32_FUSE_SITU", C32_MASTER)
C32_FUSE_COMBINE = _env_flag("INKLING_C32_FUSE_COMBINE", C32_MASTER)
C32_MX7_TILE32 = _env_flag("INKLING_C32_MX7_TILE32", C32_MASTER)
C32_PDL = _env_int("INKLING_C32_PDL", 2)

_DIR = os.path.dirname(os.path.abspath(__file__))
_NVFP4_PATHS = (
    os.environ.get("INKLING_NVFP4_WS5_C32_LIB", ""),
    "/rdma/libnvfp4_ws5_c32.so",
    os.path.join(_DIR, "libnvfp4_ws5_c32.so"),
)
_MX7_PATHS = (
    os.environ.get("INKLING_MX7_C32_LIB", ""),
    "/rdma/libmxfp8_mx7_c32.so",
    os.path.join(_DIR, "libmxfp8_mx7_c32.so"),
)
_ROUTER_PATHS = (
    os.environ.get("INKLING_ROUTER_BF12_C32_LIB", ""),
    "/rdma/librouter_bf12_c32.so",
    os.path.join(_DIR, "librouter_bf12_c32.so"),
)

_LIB_NVFP4: ctypes.CDLL | None = None
_LIB_MX7: ctypes.CDLL | None = None
_LIB_ROUTER: ctypes.CDLL | None = None
_TRIED = {"nvfp4": False, "mx7": False, "router": False}

# Static per-device C32 workspace (7.62 MiB total across all 48L, CUDA-graph capture safe).
_WS_MAX_T = 256
_WS_MAX_ROUTED = _WS_MAX_T * 8   # 2048
_WS_MAX_CTAS = 512               # covers max_ctas <= 448 at M=256
_WS_MAX_ROWS = _WS_MAX_CTAS * 32  # 16384
_WS: dict[int, dict[str, torch.Tensor]] = {}


def _find_lib(paths: tuple[str, ...], symbol: str) -> ctypes.CDLL | None:
    for p in paths:
        if p and os.path.isfile(p):
            try:
                lib = ctypes.CDLL(p)
                if hasattr(lib, symbol):
                    return lib
            except OSError:
                continue
    return None


def get_nvfp4_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_NVFP4
    if _LIB_NVFP4 is not None and path is None:
        return _LIB_NVFP4
    if _TRIED["nvfp4"] and path is None:
        return None
    if path is None:
        _TRIED["nvfp4"] = True
    paths = (path,) if path else _NVFP4_PATHS
    lib = _find_lib(paths, "nvfp4_w13_ws5_c32_fwd")
    if lib is None:
        return None
    # Match nvfp4_ws5_c32.cu extern "C" signatures 1-to-1
    lib.nvfp4_w13_ws5_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # RS
        ctypes.c_void_p,  # Row2Tok
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # WS_BASE
        ctypes.c_void_p,  # P_OFF
        ctypes.c_void_p,  # PATCHES
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Gammas
        ctypes.c_void_p,  # RowAmaxW2
        ctypes.c_void_p,  # Out
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_int,     # bk_dot
        ctypes.c_int,     # zero_scatter
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # fuse_situ
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.nvfp4_w13_ws5_c32_fwd.restype = ctypes.c_int

    lib.nvfp4_w2_ws5_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # RS
        ctypes.c_void_p,  # RowAmaxW2
        ctypes.c_void_p,  # WAmax
        ctypes.c_void_p,  # WDenominator
        ctypes.c_void_p,  # W_PACK
        ctypes.c_void_p,  # WS5_PACK
        ctypes.c_void_p,  # WS_BASE
        ctypes.c_void_p,  # P_OFF
        ctypes.c_void_p,  # PATCHES
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Out
        ctypes.c_void_p,  # Src2Compact
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # PanelDone
        ctypes.c_void_p,  # BufGen
        ctypes.c_void_p,  # SendBuf
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_int,     # T
        ctypes.c_int,     # bk_dot
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # as_row_stride
        ctypes.c_int,     # down_row_stride
        ctypes.c_int,     # fuse_quant
        ctypes.c_int,     # fuse_combine
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.nvfp4_w2_ws5_c32_fwd.restype = ctypes.c_int
    if path is None:
        _LIB_NVFP4 = lib
    return lib


def get_mx7_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_MX7
    if _LIB_MX7 is not None and path is None:
        return _LIB_MX7
    if _TRIED["mx7"] and path is None:
        return None
    if path is None:
        _TRIED["mx7"] = True
    paths = (path,) if path else _MX7_PATHS
    lib = _find_lib(paths, "mx7_w13_c32_fwd")
    if lib is None:
        return None
    # Match mxfp8_mx7_c32.cu extern "C" signatures 1-to-1
    if hasattr(lib, "mx7_v2_count_flips"):
        lib.mx7_v2_count_flips.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v2_count_flips.restype = ctypes.c_int
    if hasattr(lib, "mx7_v2_pack"):
        lib.mx7_v2_pack.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v2_pack.restype = ctypes.c_int

    lib.mx7_w13_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # Row2Tok
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # WS3
        ctypes.c_void_p,  # FLIPS
        ctypes.c_void_p,  # F_OFF
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Gammas
        ctypes.c_void_p,  # Out
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_uint32,  # base_ws
        ctypes.c_int,     # zero_scatter
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # fuse_situ
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.mx7_w13_c32_fwd.restype = ctypes.c_int

    lib.mx7_w2_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # WS3
        ctypes.c_void_p,  # FLIPS
        ctypes.c_void_p,  # F_OFF
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Out
        ctypes.c_void_p,  # Src2Compact
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # PanelDone
        ctypes.c_void_p,  # BufGen
        ctypes.c_void_p,  # SendBuf
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_int,     # T
        ctypes.c_uint32,  # base_ws
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # as_row_stride
        ctypes.c_int,     # down_row_stride
        ctypes.c_int,     # fuse_quant
        ctypes.c_int,     # fuse_combine
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.mx7_w2_c32_fwd.restype = ctypes.c_int
    if path is None:
        _LIB_MX7 = lib
    return lib


def get_router_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_ROUTER
    if _LIB_ROUTER is not None and path is None:
        return _LIB_ROUTER
    if _TRIED["router"] and path is None:
        return None
    if path is None:
        _TRIED["router"] = True
    paths = (path,) if path else _ROUTER_PATHS
    lib = _find_lib(paths, "router_c32_quant_route_place_fwd")
    if lib is None:
        return None
    # Match router_bf12_c32.cu extern "C" signatures 1-to-1
    lib.router_bf12_c32_gemm_fwd.argtypes = [
        ctypes.c_void_p,  # X
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # OUTLIERS
        ctypes.c_void_p,  # O_OFF
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # Logits
        ctypes.c_int,     # T
        ctypes.c_uint32,  # base_exp
        ctypes.c_int,     # pdl
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.router_bf12_c32_gemm_fwd.restype = ctypes.c_int

    lib.router_c32_quant_route_place_fwd.argtypes = [
        ctypes.c_void_p,  # X
        ctypes.c_void_p,  # Logits
        ctypes.c_void_p,  # Bias
        ctypes.c_void_p,  # ExpertBounds
        ctypes.c_void_p,  # Padding
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # Counts
        ctypes.c_void_p,  # Counters
        ctypes.c_void_p,  # Ids
        ctypes.c_void_p,  # Ranks
        ctypes.c_void_p,  # Weights
        ctypes.c_void_p,  # Starts
        ctypes.c_void_p,  # StartsCompact
        ctypes.c_void_p,  # Src2Dst
        ctypes.c_void_p,  # Src2Compact
        ctypes.c_void_p,  # Gammas
        ctypes.c_void_p,  # Rows
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Row2Tok
        ctypes.c_void_p,  # RowAmaxW2
        ctypes.c_void_p,  # QValues
        ctypes.c_void_p,  # QScales
        ctypes.c_void_p,  # QRowScales
        ctypes.c_void_p,  # RowAmaxIn
        ctypes.c_void_p,  # WAmax
        ctypes.c_void_p,  # WDenominator
        ctypes.c_int,     # T
        ctypes.c_float,   # route_scale
        ctypes.c_int,     # num_local
        ctypes.c_int,     # tile_sz
        ctypes.c_int,     # max_ctas
        ctypes.c_int,     # nvfp4
        ctypes.c_int,     # logits_bf16
        ctypes.c_int,     # pdl
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.router_c32_quant_route_place_fwd.restype = ctypes.c_int
    if path is None:
        _LIB_ROUTER = lib
    return lib


def get_workspace(device: torch.device) -> dict[str, torch.Tensor]:
    """Return persistent CUDA-graph-safe C32 scratch buffers (~7.62 MiB total)."""
    idx = int(device.index or 0)
    ws = _WS.get(idx)
    if ws is None:
        ws = {
            "situ_out": torch.empty((_WS_MAX_ROUTED, 288), dtype=torch.bfloat16, device=device),
            "row_amax_w2": torch.zeros((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "down_panel": torch.empty((2 * 6 * _WS_MAX_ROUTED, 128), dtype=torch.bfloat16, device=device),
            "panel_done": torch.zeros((36,), dtype=torch.int32, device=device),
            "buf_gen": torch.zeros((12,), dtype=torch.int32, device=device),
            "logits_ws": torch.empty((_WS_MAX_T, 384), dtype=torch.float32, device=device),
            "counts": torch.zeros((384,), dtype=torch.int32, device=device),
            "counters": torch.zeros((64,), dtype=torch.int32, device=device),
            "ids": torch.empty((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "ranks": torch.empty((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "weights": torch.empty((_WS_MAX_ROUTED,), dtype=torch.float32, device=device),
            "starts": torch.empty((384,), dtype=torch.int32, device=device),
            "starts_compact": torch.empty((384,), dtype=torch.int32, device=device),
            "tile_compact_start": torch.empty((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
            "src2compact": torch.empty((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "row2tok": torch.empty((_WS_MAX_ROWS,), dtype=torch.int32, device=device),
            "row_amax_in": torch.empty((_WS_MAX_T,), dtype=torch.float32, device=device),
        }
        _WS[idx] = ws
    return ws


@torch.no_grad()
def get_ws_base_c32(meta: dict) -> torch.Tensor:
    """Return `ws_base` with bit 7 set (`| 0x80`) on experts with ZERO scale patches.

    Leaves `meta['ws_base']` untouched so Triton prefill kernels stay 100% unmodified.
    """
    tagged = meta.get("ws_base_c32")
    if tagged is None:
        tagged = meta.get("ws_base_s8")
    if tagged is not None:
        return tagged
    ws_base: torch.Tensor = meta["ws_base"]
    if torch.cuda.is_current_stream_capturing():
        return ws_base
    p_off: torch.Tensor = meta["p_off"]
    E = int(ws_base.shape[0])
    proj = int(meta["proj"])
    stride = (18 * 4) if proj == 0 else 36
    idx = torch.arange(E, device=ws_base.device, dtype=torch.long) * stride
    no_patch = p_off[idx + stride] == p_off[idx]
    assert int(ws_base.max().item()) < 128, "NVFP4 ws_base unexpectedly >= 128"
    tagged = torch.where(no_patch, ws_base | 0x80, ws_base).to(torch.uint8).contiguous()
    meta["ws_base_c32"] = tagged
    return tagged


@torch.no_grad()
def get_mx7_packed_c32(weight: torch.Tensor, weight_scale: torch.Tensor, projection: int):
    """Return `(slabs, ws3, flips, f_off, base_ws)` reusing `weight._inkling_mx7` if present."""
    cached = getattr(weight, "_inkling_mx7", None)
    if isinstance(cached, tuple):
        return cached
    if cached is False:
        return None
    # Prefer trtllm_bmm._mx7_get_packed so packing & FREE_RAW policy stay unified.
    try:
        from . import trtllm_bmm as _tbmm
        packed = _tbmm._mx7_get_packed(weight, weight_scale, projection, 0, 0)
        if isinstance(packed, tuple):
            return packed
    except Exception:
        pass
    if torch.cuda.is_current_stream_capturing():
        return None
    lib = get_mx7_lib()
    if lib is None or not (hasattr(lib, "mx7_v2_count_flips") and hasattr(lib, "mx7_v2_pack")):
        return None
    if weight.data.untyped_storage().size() == 0 or weight_scale.data.untyped_storage().size() == 0:
        return None
    e, n, k = int(weight.shape[0]), int(weight.shape[1]), int(weight.shape[2])
    if projection == 0 and (n % 32) == 0 and (k % 256) == 0:
        bn, warps, n_slabs = 32, 4, k // 128
    elif projection == 1 and (n % 64) == 0 and (k % 32) == 0:
        bn, warps, n_slabs = 64, 2, k // 32
    else:
        return None
    ws_u8 = weight_scale.data.view(torch.uint8)
    base_ws = int(ws_u8.min().item())
    max_ws = int(ws_u8.max().item())
    if base_ws == 0 or max_ws == 255 or (max_ws - base_ws) > 7:
        weight._inkling_mx7 = False
        return None
    w_u8 = weight.data.view(torch.uint8)
    dev = weight.device
    stream = int(torch.cuda.current_stream(dev).cuda_stream)
    total_warps = e * (n // bn) * warps
    panel_meta = getattr(weight, "_inkling_panel", None)
    if panel_meta is not None:
        from_panel, src_pb, src_kb = 1, int(panel_meta[0]), int(panel_meta[1])
    else:
        from_panel, src_pb, src_kb = 0, 0, 0
    counts = torch.empty(total_warps, dtype=torch.int32, device=dev)
    rc = lib.mx7_v2_count_flips(
        w_u8.data_ptr(), counts.data_ptr(), projection, e, n, k, from_panel, src_pb, src_kb, stream
    )
    if rc != 0:
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
        return None
    entry = (slabs, ws3, flips, f_off, base_ws)
    weight._inkling_mx7 = entry
    return entry


def c32_eligible(
    tokens: int,
    hidden: int,
    topk: int,
    plan,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> bool:
    return (
        C32_MASTER
        and C32_MIN_M <= tokens <= C32_MAX_M
        and hidden == 4608
        and topk == 8
        and plan.tile in (16, 32)
        and plan.max_ctas <= _WS_MAX_CTAS
        and plan.rows <= _WS_MAX_ROWS
        and gate_cap == 4.0
        and up_cap == 25.0
    )


def _is_full_decode_capturing() -> bool:
    """True ONLY inside FULL decode CUDA graph capture (never eager or PIECEWISE prefill!).

    vLLM runs in FULL_AND_PIECEWISE mode and captures PIECEWISE prefill graphs for M=32..256
    where `torch.cuda.is_current_stream_capturing()` is True. Running `router_bf12_c32_gemm_fwd`
    (FP32 5-K-group logits) inside PIECEWISE prefill diverges from stock cuBLAS BF16
    `k_group_partials` (deterministic `exact_ink` hash `1dcfb306` -> `9a860d12` when
    `INKLING_C32_MOE_MIN_M=32`). Restricting `bf12` to FULL decode graphs keeps all eager
    and PIECEWISE prefills 100% 0-ULP bit-exact while letting FULL decode graphs use `bf12`.
    """
    if not torch.cuda.is_current_stream_capturing():
        return False
    try:
        from . import l2pf
        if l2pf._is_piecewise_capturing():
            return False
    except Exception:
        pass
    try:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if is_forward_context_available():
            return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
    except Exception:
        pass
    return True


def will_use_bf12_gate(num_tokens: int) -> bool:
    """True iff FULL decode graphs at `num_tokens` execute `router_bf12_c32_gemm_fwd`."""
    return (
        C32_MASTER
        and C32_ROUTER_MODE == "bf12"
        and C32_MIN_M <= int(num_tokens) <= C32_MAX_M
    )


def route_c32(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    plan,
    weight_amax: torch.Tensor | None,
    weight_denominator: torch.Tensor | None,
    num_local: int,
    is_padding: torch.Tensor | None = None,
    nlive_buf: torch.Tensor | None = None,
):
    """C=32 (M=17..256) fused router + Zero-Scatter quantizer + compact metadata builder.

    Returns `((src2dst, gammas, routed, tiles), quantized)` or `None` if ineligible/disabled.
    Attaches `_inkling_row2tok` onto `quantized[0]`, `_inkling_rowmaj = True` onto `quantized[1]`,
    and `_inkling_tile_compact_start`, `_inkling_src2compact`, `_inkling_row_amax_w2` onto `tiles`.
    """
    if C32_ROUTER_MODE == "off":
        return None
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return None
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if not c32_eligible(tokens, hidden, topk, plan):
        return None
    if int(gate_weight.shape[0]) != 384 or num_local > 384:
        return None

    lib = get_router_lib()
    if lib is None:
        return None

    device = x.device
    ws = get_workspace(device)
    st = int(torch.cuda.current_stream(device).cuda_stream)
    nvfp4 = weight_amax is not None
    n = tokens * topk
    rows = int(plan.rows)

    # Determine whether to run BF12 router GEMM or stock cuBLAS BF16 `k_group_partials`.
    use_bf12 = (C32_ROUTER_MODE == "bf12")
    if use_bf12 and C32_ROUTER_BF12_CAPTURE_ONLY and not _is_full_decode_capturing():
        use_bf12 = False

    logits_bf16 = 1
    logits_ptr = 0
    if use_bf12:
        try:
            from . import router_bf12 as _rb12
            _rb12.prepack(gate_weight)
            hit = _rb12._CACHE.get((int(gate_weight.data_ptr()), 384, 4608))
            if hit is not None and hit[0]() is gate_weight:
                _, slabs, outliers, o_off, base_exp = hit
                rc = lib.router_bf12_c32_gemm_fwd(
                    x.data_ptr(),
                    slabs.data_ptr(),
                    outliers.data_ptr(),
                    o_off.data_ptr(),
                    0 if nlive_buf is None else nlive_buf.data_ptr(),
                    ws["logits_ws"].data_ptr(),
                    tokens,
                    int(base_exp),
                    1 if C32_PDL else 0,
                    1 if C32_PDL else 0,
                    st,
                )
                if rc != 0:
                    raise RuntimeError(f"router_bf12_c32_gemm_fwd failed: rc={rc}")
                logits_bf16 = 0
                logits_ptr = ws["logits_ws"].data_ptr()
        except Exception:
            logits_bf16 = 1
            logits_ptr = 0

    if logits_ptr == 0:
        from .linear import k_group_partials
        partials = k_group_partials(x, gate_weight, out_dtype=torch.bfloat16)
        logits_bf16 = 1
        logits_ptr = partials.data_ptr()

    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)

    # Zero-Scatter: QValues and row-major QScales are sized [tokens, ...], NEVER [plan.rows, ...]!
    if nvfp4:
        quantized = (
            torch.empty((tokens, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((tokens, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
        )
    else:
        quantized = (
            torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device),
            None,
        )

    tile_compact_start = ws["tile_compact_start"][: plan.max_ctas]
    src2compact = ws["src2compact"][:n]
    row2tok = ws["row2tok"][:rows]
    row_amax_w2 = ws["row_amax_w2"][:n] if (nvfp4 and C32_FUSE_SITU) else None

    rc = lib.router_c32_quant_route_place_fwd(
        x.data_ptr(),
        logits_ptr,
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        0 if is_padding is None else is_padding.data_ptr(),
        0 if nlive_buf is None else nlive_buf.data_ptr(),
        ws["counts"].data_ptr(),
        ws["counters"].data_ptr(),
        ws["ids"].data_ptr(),
        ws["ranks"].data_ptr(),
        ws["weights"].data_ptr(),
        ws["starts"].data_ptr(),
        ws["starts_compact"].data_ptr(),
        src2dst.data_ptr(),
        src2compact.data_ptr(),
        gammas.data_ptr(),
        routed.data_ptr(),
        tiles.data_ptr(),
        tile_compact_start.data_ptr(),
        row2tok.data_ptr(),
        0 if row_amax_w2 is None else row_amax_w2.data_ptr(),
        quantized[0].data_ptr(),
        quantized[1].data_ptr(),
        0 if quantized[2] is None else quantized[2].data_ptr(),
        ws["row_amax_in"].data_ptr(),
        0 if weight_amax is None else weight_amax.data_ptr(),
        0 if weight_denominator is None else weight_denominator.data_ptr(),
        tokens,
        float(route_scale),
        int(num_local),
        int(plan.tile),
        int(plan.max_ctas),
        1 if nvfp4 else 0,
        logits_bf16,
        C32_PDL,
        st,
    )
    if rc != 0:
        raise RuntimeError(f"router_c32_quant_route_place_fwd failed: rc={rc}")

    quantized[0]._inkling_row2tok = row2tok  # type: ignore[attr-defined]
    quantized[1]._inkling_rowmaj = True  # type: ignore[attr-defined]
    tiles._inkling_tile_compact_start = tile_compact_start  # type: ignore[attr-defined]
    tiles._inkling_src2compact = src2compact  # type: ignore[attr-defined]
    if row_amax_w2 is not None:
        tiles._inkling_row_amax_w2 = row_amax_w2  # type: ignore[attr-defined]

    return (src2dst, gammas, routed, tiles), quantized


def bmm_c32_standalone(
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
) -> torch.Tensor | None:
    """Standalone unfused C=32 GEMM dispatch for `trtllm_bmm.bmm()` at `plan.tile in (16, 32)`."""
    if not C32_MASTER or plan.tile not in (16, 32) or plan.rows <= 2048 or plan.max_ctas > _WS_MAX_CTAS:
        return None
    nvfp4 = weight.dtype == torch.uint8
    pack = 2 if nvfp4 else 1
    n = int(weight.shape[1])
    k = int(weight.shape[2]) * pack
    if not ((projection == 0 and n == 576 and k == 4608) or (projection == 1 and n == 4608 and k == 288)):
        return None

    device = act.device
    st = int(torch.cuda.current_stream(device).cuda_stream)
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=device)

    if nvfp4:
        lib = get_nvfp4_lib()
        if lib is None:
            return None
        from . import nvfp4_ws5
        meta = (
            nvfp4_ws5.pack_w13_ws5_inplace(weight, weight_scale)
            if projection == 0
            else nvfp4_ws5.pack_w2_ws5_inplace(weight, weight_scale)
        )
        ws_base = get_ws_base_c32(meta)
        if projection == 0:
            zs = 1 if row2tok is not None else 0
            rm = 1 if (zs or as_row_major) else 0
            bk_dot = 256 if plan.tile <= 16 else 128
            rc = lib.nvfp4_w13_ws5_c32_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                row_scale.data_ptr(),
                0 if row2tok is None else row2tok.data_ptr(),
                meta["slabs"].data_ptr(),
                ws_base.data_ptr(),
                meta["p_off"].data_ptr(),
                meta["patches"].data_ptr(),
                ones.data_ptr(),
                tiles.data_ptr(),
                0, 0, 0,
                out.data_ptr(),
                n, k, int(plan.max_ctas), int(plan.tile),
                bk_dot, zs, rm, 0,
                1 if C32_PDL else 0, 1 if C32_PDL else 0, st,
            )
        else:
            bk_dot = 32 if plan.tile <= 16 else 64
            as_stride = int(act_scale.stride(0)) if (as_row_major and act_scale.ndim >= 2) else 18
            rc = lib.nvfp4_w2_ws5_c32_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                row_scale.data_ptr(),
                0, 0, 0,
                meta["w_pack"].data_ptr(),
                meta["ws5_pack"].data_ptr(),
                ws_base.data_ptr(),
                meta["p_off"].data_ptr(),
                meta["patches"].data_ptr(),
                ones.data_ptr(),
                tiles.data_ptr(),
                0,
                out.data_ptr(),
                0, 0, 0, 0, 0,
                n, k, int(plan.max_ctas), int(plan.tile), 0,
                bk_dot, 1 if as_row_major else 0, as_stride, 0,
                0, 0,
                2 if C32_PDL else 0, 1 if C32_PDL else 0, st,
            )
        return out if rc == 0 else None

    # MXFP8 MX7 path
    if plan.tile == 32 and not C32_MX7_TILE32:
        return None
    lib = get_mx7_lib()
    if lib is None:
        return None
    packed = get_mx7_packed_c32(weight, weight_scale, projection)
    if packed is None:
        return None
    slabs, ws3, flips, f_off, base_ws = packed
    if projection == 0:
        zs = 1 if row2tok is not None else 0
        rm = 1 if (zs or as_row_major) else 0
        rc = lib.mx7_w13_c32_fwd(
            act.view(torch.uint8).data_ptr(),
            act_scale.view(torch.uint8).data_ptr(),
            0 if row2tok is None else row2tok.data_ptr(),
            slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
            ones.data_ptr(), tiles.data_ptr(),
            0, 0, out.data_ptr(),
            n, k, int(plan.max_ctas), int(plan.tile), int(base_ws),
            zs, rm, 0,
            1 if C32_PDL else 0, 1 if C32_PDL else 0, st,
        )
    else:
        as_stride = int(act_scale.stride(0)) if (as_row_major and act_scale.ndim >= 2) else 9
        rc = lib.mx7_w2_c32_fwd(
            act.view(torch.uint8).data_ptr(),
            act_scale.view(torch.uint8).data_ptr(),
            slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
            ones.data_ptr(), tiles.data_ptr(), 0, out.data_ptr(),
            0, 0, 0, 0, 0,
            n, k, int(plan.max_ctas), int(plan.tile), 0, int(base_ws),
            1 if as_row_major else 0, as_stride, 0,
            0, 0,
            2 if C32_PDL else 0, 1 if C32_PDL else 0, st,
        )
    return out if rc == 0 else None


def forward_experts_c32(
    experts,
    x: torch.Tensor,
    routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    plan,
    *,
    out_buf: torch.Tensor | None = None,
    nlive_buf: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Execute full C=32 0-ULP fused W13 (+SiTU) -> W2 (+OTF Quant + 6-Wave L2 Combine).

    Returns unreduced combined `[T, 4608]` bf16 (in `out_buf` if provided), or `None`
    to cleanly fall back to stock `Inkling2Experts.forward`.
    """
    tokens = int(x.shape[0])
    if not c32_eligible(
        tokens, int(experts.hidden_size), int(experts.topk), plan,
        float(experts.gate_cap), float(experts.up_cap),
    ):
        return None
    if int(experts.w13_weight.shape[1]) != 576 or int(experts.w2_weight.shape[1]) != 4608:
        return None

    device = x.device
    out_dtype = torch.bfloat16 if x.dtype == torch.float8_e4m3fn else x.dtype
    if out_dtype != torch.bfloat16:
        return None

    src2dst, gammas, rows, tiles = routing
    act, act_scale, row_scale = quantized
    r2t = getattr(act, "_inkling_row2tok", None)
    as_rm = bool(getattr(act_scale, "_inkling_rowmaj", False))
    tile_compact_start = getattr(tiles, "_inkling_tile_compact_start", None)
    src2compact = getattr(tiles, "_inkling_src2compact", None)
    row_amax_w2 = getattr(tiles, "_inkling_row_amax_w2", None)

    has_compact = (tile_compact_start is not None) and (src2compact is not None)
    use_fuse_situ = C32_FUSE_SITU and has_compact and ((not experts.nvfp4) or (row_amax_w2 is not None))
    use_fuse_combine = C32_FUSE_COMBINE and has_compact

    ws = get_workspace(device)
    st = int(torch.cuda.current_stream(device).cuda_stream)
    pdl_w13_wait = 1 if C32_PDL else 0
    pdl_w2_wait = 2 if C32_PDL else 0
    trig = 1 if C32_PDL else 0
    down_row_stride = tokens * 8  # compact rows in [0, 8*T)

    if experts.nvfp4:
        lib = get_nvfp4_lib()
        if lib is None:
            return None
        from . import nvfp4_ws5
        m13 = nvfp4_ws5.pack_w13_ws5_inplace(experts.w13_weight, experts.w13_weight_scale)
        m2 = nvfp4_ws5.pack_w2_ws5_inplace(experts.w2_weight, experts.w2_weight_scale)
        ws_base_13 = get_ws_base_c32(m13)
        ws_base_2 = get_ws_base_c32(m2)
        bk_w13 = 256 if plan.tile <= 16 else 128
        bk_w2 = 32 if plan.tile <= 16 else 64

        if use_fuse_situ:
            situ_out = ws["situ_out"][:down_row_stride]
            rc = lib.nvfp4_w13_ws5_c32_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                row_scale.data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                m13["slabs"].data_ptr(),
                ws_base_13.data_ptr(),
                m13["p_off"].data_ptr(),
                m13["patches"].data_ptr(),
                experts.ones.data_ptr(),
                tiles.data_ptr(),
                tile_compact_start.data_ptr(),
                gammas.data_ptr(),
                row_amax_w2.data_ptr(),
                situ_out.data_ptr(),
                576, 4608, int(plan.max_ctas), int(plan.tile),
                bk_w13, 1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, 1,
                pdl_w13_wait, trig, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w13_ws5_c32_fwd (fuse_situ=1) failed: rc={rc}")
            w2_a_ptr = situ_out.data_ptr()
            w2_as_ptr = 0
            w2_rs_ptr = 0
            w2_as_rm = 0
            w2_as_stride = 18
        else:
            gate_up = torch.empty((plan.rows, 576), dtype=torch.bfloat16, device=device)
            rc = lib.nvfp4_w13_ws5_c32_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                row_scale.data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                m13["slabs"].data_ptr(),
                ws_base_13.data_ptr(),
                m13["p_off"].data_ptr(),
                m13["patches"].data_ptr(),
                experts.ones.data_ptr(),
                tiles.data_ptr(),
                0, 0, 0,
                gate_up.data_ptr(),
                576, 4608, int(plan.max_ctas), int(plan.tile),
                bk_w13, 1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, 0,
                pdl_w13_wait, trig, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w13_ws5_c32_fwd (fuse_situ=0) failed: rc={rc}")
            from .activation import swiglu_quantize
            m = plan.max_ctas
            act2_v, act2_s, act2_rs = swiglu_quantize(
                gate_up, gammas, experts.gate_cap, experts.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, *experts.scales("w2"),
            )
            del gate_up
            w2_a_ptr = act2_v.view(torch.uint8).data_ptr()
            w2_as_ptr = act2_s.view(torch.uint8).data_ptr()
            w2_rs_ptr = act2_rs.data_ptr()
            w2_as_rm = 1 if bool(getattr(act2_s, "_inkling_rowmaj", False)) else 0
            w2_as_stride = int(act2_s.stride(0)) if (w2_as_rm and act2_s.ndim >= 2) else 18

        w2_amax, w2_denom = experts.scales("w2")
        if use_fuse_combine:
            out = out_buf if out_buf is not None else torch.empty(
                (tokens, 4608), dtype=torch.bfloat16, device=device
            )
            rc = lib.nvfp4_w2_ws5_c32_fwd(
                w2_a_ptr, w2_as_ptr, w2_rs_ptr,
                row_amax_w2.data_ptr() if use_fuse_situ else 0,
                w2_amax.data_ptr() if use_fuse_situ else 0,
                w2_denom.data_ptr() if use_fuse_situ else 0,
                m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(), tile_compact_start.data_ptr(),
                ws["down_panel"].data_ptr(),
                src2compact.data_ptr(),
                0 if nlive_buf is None else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr(),
                ws["buf_gen"].data_ptr(),
                out.data_ptr(),
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens,
                bk_w2, w2_as_rm, w2_as_stride, down_row_stride,
                1 if use_fuse_situ else 0, 1,
                pdl_w2_wait, trig, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w2_ws5_c32_fwd (fuse_combine=1) failed: rc={rc}")
            return out

        down = torch.empty((plan.rows, 4608), dtype=torch.bfloat16, device=device)
        rc = lib.nvfp4_w2_ws5_c32_fwd(
            w2_a_ptr, w2_as_ptr, w2_rs_ptr,
            row_amax_w2.data_ptr() if use_fuse_situ else 0,
            w2_amax.data_ptr() if use_fuse_situ else 0,
            w2_denom.data_ptr() if use_fuse_situ else 0,
            m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
            m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            tile_compact_start.data_ptr() if use_fuse_situ else 0,
            down.data_ptr(),
            0, 0, 0, 0, 0,
            4608, 288, int(plan.max_ctas), int(plan.tile), tokens,
            bk_w2, w2_as_rm, w2_as_stride, down_row_stride,
            1 if use_fuse_situ else 0, 0,
            pdl_w2_wait, trig, st,
        )
        if rc != 0:
            raise RuntimeError(f"nvfp4_w2_ws5_c32_fwd (fuse_combine=0) failed: rc={rc}")
        from .moe import combine
        return combine(down, src2dst, experts.topk, out_dtype, out=out_buf)

    # ---- MXFP8 MX7 (L40..47) C=32 path ----
    if plan.tile == 32 and not C32_MX7_TILE32:
        return None
    lib = get_mx7_lib()
    if lib is None:
        return None
    p13 = get_mx7_packed_c32(experts.w13_weight, experts.w13_weight_scale, 0)
    p2 = get_mx7_packed_c32(experts.w2_weight, experts.w2_weight_scale, 1)
    if p13 is None or p2 is None:
        return None
    slabs13, ws3_13, flips13, f_off13, base_ws13 = p13
    slabs2, ws3_2, flips2, f_off2, base_ws2 = p2

    if use_fuse_situ:
        situ_out = ws["situ_out"][:down_row_stride]
        rc = lib.mx7_w13_c32_fwd(
            act.view(torch.uint8).data_ptr(),
            act_scale.view(torch.uint8).data_ptr(),
            0 if r2t is None else r2t.data_ptr(),
            slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), f_off13.data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            tile_compact_start.data_ptr(), gammas.data_ptr(), situ_out.data_ptr(),
            576, 4608, int(plan.max_ctas), int(plan.tile), int(base_ws13),
            1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, 1,
            pdl_w13_wait, trig, st,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_w13_c32_fwd (fuse_situ=1) failed: rc={rc}")
        w2_a_ptr = situ_out.data_ptr()
        w2_as_ptr = 0
        w2_as_rm = 0
        w2_as_stride = 9
    else:
        gate_up = torch.empty((plan.rows, 576), dtype=torch.bfloat16, device=device)
        rc = lib.mx7_w13_c32_fwd(
            act.view(torch.uint8).data_ptr(),
            act_scale.view(torch.uint8).data_ptr(),
            0 if r2t is None else r2t.data_ptr(),
            slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), f_off13.data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            0, 0, gate_up.data_ptr(),
            576, 4608, int(plan.max_ctas), int(plan.tile), int(base_ws13),
            1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, 0,
            pdl_w13_wait, trig, st,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_w13_c32_fwd (fuse_situ=0) failed: rc={rc}")
        from .activation import swiglu_quantize
        m = plan.max_ctas
        act2_v, act2_s, _ = swiglu_quantize(
            gate_up, gammas, experts.gate_cap, experts.up_cap,
            (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, None, None,
        )
        del gate_up
        w2_a_ptr = act2_v.view(torch.uint8).data_ptr()
        w2_as_ptr = act2_s.view(torch.uint8).data_ptr()
        w2_as_rm = 1 if bool(getattr(act2_s, "_inkling_rowmaj", False)) else 0
        w2_as_stride = int(act2_s.stride(0)) if (w2_as_rm and act2_s.ndim >= 2) else 9

    if use_fuse_combine:
        out = out_buf if out_buf is not None else torch.empty(
            (tokens, 4608), dtype=torch.bfloat16, device=device
        )
        rc = lib.mx7_w2_c32_fwd(
            w2_a_ptr, w2_as_ptr,
            slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), f_off2.data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(), tile_compact_start.data_ptr(),
            ws["down_panel"].data_ptr(),
            src2compact.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            ws["panel_done"].data_ptr(),
            ws["buf_gen"].data_ptr(),
            out.data_ptr(),
            4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(base_ws2),
            w2_as_rm, w2_as_stride, down_row_stride,
            1 if use_fuse_situ else 0, 1,
            pdl_w2_wait, trig, st,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_w2_c32_fwd (fuse_combine=1) failed: rc={rc}")
        return out

    down = torch.empty((plan.rows, 4608), dtype=torch.bfloat16, device=device)
    rc = lib.mx7_w2_c32_fwd(
        w2_a_ptr, w2_as_ptr,
        slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), f_off2.data_ptr(),
        experts.ones.data_ptr(), tiles.data_ptr(),
        tile_compact_start.data_ptr() if use_fuse_situ else 0,
        down.data_ptr(),
        0, 0, 0, 0, 0,
        4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(base_ws2),
        w2_as_rm, w2_as_stride, down_row_stride,
        1 if use_fuse_situ else 0, 0,
        pdl_w2_wait, trig, st,
    )
    if rc != 0:
        raise RuntimeError(f"mx7_w2_c32_fwd (fuse_combine=0) failed: rc={rc}")
    from .moe import combine
    return combine(down, src2dst, experts.topk, out_dtype, out=out_buf)
