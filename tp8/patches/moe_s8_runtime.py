# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) Dec160 S8 0-ULP Bit-Exact Fused Target MoE Decode Runtime.

Drop-in orchestration module for the S8 3-library Target MoE decode stack:
  1. `librouter_bf12_s8.so`: 48-CTA zero-tail K1 + 1-CTA Zero-Scatter SMEM-Route K2
     (`TileMask` + row-major `QValues[T]`/`QScales[T]` + pre-zeroed `RowAmaxW2`).
  2. `libnvfp4_ws5_s8.so`: Bank-conflict-free (`SA_STRIDE=144`) WS5 NVFP4 W13/W2 with
     `fns.b32` Zero-Scatter, `WS_BASE` bit-7 `P_OFF` bypass, inter-tile L2 prefetch,
     Fused W13 SiTU + `RowAmaxW2` epilogue, Fused W2 OTF NVFP4 quant-dequant, and
     Fused W2 exact slot-0..7 combine directly into RDMA_AR `SendBuf` + PDL trigger.
  3. `libmxfp8_mx7_s8.so`: 0-way `LDS.64` (`SA_STRIDE=272/304`) MX7 W13/W2 with
     `fns.b32` Zero-Scatter, inter-tile L2 prefetch, Fused W13 SiTU epilogue,
     Fused W2 OTF MXFP8 quant-dequant, Fused W2 exact slot-0..7 combine -> `SendBuf`,
     while preserving 100% of v4 prefill + `INKLING_MX7_FREE_RAW=1` compatibility.

1-Knob Rollout Controls (all default off unless `INKLING_MOE_S8=1` or set individually):
  - `INKLING_MOE_S8=1`              : master switch enabling all 4 knobs below
  - `INKLING_MOE_S8_ROUTER=1`       : 48-CTA zero-tail K1 + 1-CTA K2 router
  - `INKLING_MOE_S8_ZERO_SCATTER=1` : TileMask + row-major per-token QValues/QScales
  - `INKLING_MOE_S8_FUSE_SITU=1`    : fuse SiTU into W13 + OTF quant into W2 (-48 launches/step)
  - `INKLING_MOE_S8_FUSE_COMBINE=1` : fuse exact slot-0..7 combine into W2 -> SendBuf (-48 launches/step)
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


S8_MASTER = _env_flag("INKLING_MOE_S8", False)
S8_ROUTER = _env_flag("INKLING_MOE_S8_ROUTER", S8_MASTER)
S8_ZERO_SCATTER = _env_flag("INKLING_MOE_S8_ZERO_SCATTER", S8_MASTER)
S8_FUSE_SITU = _env_flag("INKLING_MOE_S8_FUSE_SITU", S8_MASTER)
S8_FUSE_COMBINE = _env_flag("INKLING_MOE_S8_FUSE_COMBINE", S8_MASTER)

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROUTER_PATHS = (
    os.environ.get("INKLING_ROUTER_BF12_S8_LIB", ""),
    os.environ.get("INKLING_ROUTER_BF12_LIB", ""),
    "/rdma/librouter_bf12_s8.so",
    os.path.join(_DIR, "librouter_bf12_s8.so"),
)
_NVFP4_PATHS = (
    os.environ.get("INKLING_NVFP4_WS5_S8_LIB", ""),
    os.environ.get("INKLING_NVFP4_WS5_LIB", ""),
    "/rdma/libnvfp4_ws5_s8.so",
    os.path.join(_DIR, "libnvfp4_ws5_s8.so"),
)
_MX7_PATHS = (
    os.environ.get("INKLING_MX7_S8_LIB", ""),
    os.environ.get("INKLING_MX7_LIB", ""),
    "/rdma/libmxfp8_mx7_s8.so",
    os.path.join(_DIR, "libmxfp8_mx7_s8.so"),
)

_LIB_ROUTER = None
_LIB_NVFP4 = None
_LIB_MX7 = None
_TRIED = {"router": False, "nvfp4": False, "mx7": False}

# Static per-device S8 workspace (shared across all 48 layers, CUDA-graph capture safe).
# Sized for T <= 16 decode (plan.rows <= 2048, plan.max_ctas <= 256):
#   situ_out     [2048, 288]  bf16  = 1.125 MiB
#   row_amax_w2  [2048]       int32 = 8 KiB
#   tile_mask    [256]        int32 = 1 KiB
#   down_ws      [2048, 4608] bf16  = 18.0 MiB
#   panel_done   [72]         int32 = 288 B (zeroed once; self-reset by last panel CTA!)
#   logits_ws    [5, 16, 384] fp32  = 120 KiB
# Total VRAM across all 48 layers = 19.25 MiB (< 0.015% of GB10 128 GiB).
_WS_MAX_ROWS = 2048
_WS_MAX_CTAS = 256
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


def get_router_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_ROUTER
    if _LIB_ROUTER is not None and path is None:
        return _LIB_ROUTER
    if _TRIED["router"] and path is None:
        return None
    if path is None:
        _TRIED["router"] = True
    paths = (path,) if path else _ROUTER_PATHS
    lib = _find_lib(paths, "router_bf12_s8_zero_scatter_fwd")
    if lib is None:
        return None
    lib.router_bf12_s8_zero_scatter_fwd.argtypes = [
        ctypes.c_void_p,  # X
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # OUTLIERS
        ctypes.c_void_p,  # O_OFF
        ctypes.c_void_p,  # Bias
        ctypes.c_void_p,  # ExpertBounds
        ctypes.c_void_p,  # Padding
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # LogitsWS
        ctypes.c_void_p,  # Src2Dst
        ctypes.c_void_p,  # Gammas
        ctypes.c_void_p,  # Rows
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileMask
        ctypes.c_void_p,  # RowAmaxW2
        ctypes.c_void_p,  # QValues
        ctypes.c_void_p,  # QScales
        ctypes.c_void_p,  # QRowScales
        ctypes.c_void_p,  # WAmax
        ctypes.c_void_p,  # WDenominator
        ctypes.c_int,     # T
        ctypes.c_float,   # route_scale
        ctypes.c_uint32,  # base_exp
        ctypes.c_int,     # num_local
        ctypes.c_int,     # tile_sz
        ctypes.c_int,     # max_ctas
        ctypes.c_int,     # nvfp4
        ctypes.c_int,     # pdl
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.router_bf12_s8_zero_scatter_fwd.restype = ctypes.c_int
    if hasattr(lib, "router_bf12_exact_fwd"):
        lib.router_bf12_exact_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_float, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.router_bf12_exact_fwd.restype = ctypes.c_int
    if path is None:
        _LIB_ROUTER = lib
    return lib


def get_nvfp4_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_NVFP4
    if _LIB_NVFP4 is not None and path is None:
        return _LIB_NVFP4
    if _TRIED["nvfp4"] and path is None:
        return None
    if path is None:
        _TRIED["nvfp4"] = True
    paths = (path,) if path else _NVFP4_PATHS
    lib = _find_lib(paths, "nvfp4_w13_ws5_s8_fwd")
    if lib is None:
        return None
    lib.nvfp4_w13_ws5_s8_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ]
    lib.nvfp4_w13_ws5_s8_fwd.restype = ctypes.c_int
    lib.nvfp4_w2_ws5_s8_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ]
    lib.nvfp4_w2_ws5_s8_fwd.restype = ctypes.c_int
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
    lib = _find_lib(paths, "mx7_w13_s8_fwd")
    if lib is None:
        return None
    lib.mx7_w13_s8_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ]
    lib.mx7_w13_s8_fwd.restype = ctypes.c_int
    lib.mx7_w2_s8_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ]
    lib.mx7_w2_s8_fwd.restype = ctypes.c_int
    if path is None:
        _LIB_MX7 = lib
    return lib


def get_workspace(device: torch.device) -> dict[str, torch.Tensor]:
    """Return persistent CUDA-graph-safe S8 scratch buffers for `device`."""
    idx = int(device.index or 0)
    ws = _WS.get(idx)
    if ws is None:
        ws = {
            "situ_out": torch.empty((_WS_MAX_ROWS, 288), dtype=torch.bfloat16, device=device),
            "row_amax_w2": torch.zeros((_WS_MAX_ROWS,), dtype=torch.int32, device=device),
            "tile_mask": torch.empty((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
            "down_ws": torch.empty((_WS_MAX_ROWS, 4608), dtype=torch.bfloat16, device=device),
            "panel_done": torch.zeros((72,), dtype=torch.int32, device=device),
            "logits_ws": torch.empty((5, 16, 384), dtype=torch.float32, device=device),
        }
        _WS[idx] = ws
    return ws


@torch.no_grad()
def get_ws_base_s8(meta: dict) -> torch.Tensor:
    """Return `ws_base` with bit 7 set (`| 0x80`) on experts with ZERO scale patches.

    Leaves `meta['ws_base']` untouched so Triton prefill kernels stay 100% unmodified.
    Safe to call during CUDA graph capture once prewarmed during eager startup!
    """
    tagged = meta.get("ws_base_s8")
    if tagged is not None:
        return tagged
    ws_base: torch.Tensor = meta["ws_base"]
    if torch.cuda.is_current_stream_capturing():
        # Legacy untagged ws_base has bit 7 == 0 -> safe fallback (reads P_OFF normally).
        return ws_base
    p_off: torch.Tensor = meta["p_off"]
    E = int(ws_base.shape[0])
    proj = int(meta["proj"])
    stride = (18 * 4) if proj == 0 else 36  # warps/expert for W13, panels/expert for W2
    idx = torch.arange(E, device=ws_base.device, dtype=torch.long) * stride
    no_patch = p_off[idx + stride] == p_off[idx]
    # All valid NVFP4 e4m3 scales are non-negative (0..126 < 128), so bit 7 is guaranteed free!
    assert int(ws_base.max().item()) < 128, "NVFP4 ws_base unexpectedly >= 128"
    tagged = torch.where(no_patch, ws_base | 0x80, ws_base).to(torch.uint8).contiguous()
    meta["ws_base_s8"] = tagged
    return tagged


def s8_decode_eligible(
    tokens: int,
    hidden: int,
    topk: int,
    plan,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> bool:
    return (
        0 < tokens <= 16
        and hidden == 4608
        and topk == 8
        and plan.tile in (8, 16)
        and plan.tile >= tokens
        and plan.rows <= _WS_MAX_ROWS
        and plan.max_ctas <= _WS_MAX_CTAS
        and gate_cap == 4.0
        and up_cap == 25.0
    )


def route_bf12_s8(
    x: torch.Tensor,
    slabs: torch.Tensor,
    outliers: torch.Tensor,
    o_off: torch.Tensor,
    base_exp: int,
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
    pdl: int = 2,
    zero_scatter: bool | None = None,
    prezero_amax_w2: bool | None = None,
):
    """Execute S8 48-CTA zero-tail K1 + 1-CTA SMEM-Route K2.

    When `zero_scatter=True`:
      - Returns `QValues` `[T, 2304|4608]` & row-major `QScales` `[T, 288|144]`.
      - Attaches `_inkling_tile_mask` (`[max_ctas]`) and optional `_inkling_row_amax_w2`
        (`[rows]`, pre-zeroed by K2!) onto `tiles` so downstream W13/W2 consume them directly.
    When `zero_scatter=False`:
      - Returns standard `[rows, ...]` swizzled buffers compatible with stock S7 W13!
    """
    lib = get_router_lib()
    if lib is None:
        return None
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if not s8_decode_eligible(tokens, hidden, topk, plan):
        return None

    use_zs = S8_ZERO_SCATTER if zero_scatter is None else bool(zero_scatter)
    use_amax = (S8_FUSE_SITU and weight_amax is not None) if prezero_amax_w2 is None else bool(prezero_amax_w2)

    device = x.device
    ws = get_workspace(device)
    nvfp4 = weight_amax is not None
    n = tokens * topk
    rows = plan.rows

    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)

    qrows = tokens if use_zs else rows
    if nvfp4:
        quantized = (
            torch.empty((qrows, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((qrows, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
        )
    else:
        quantized = (
            torch.empty((qrows, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((qrows, hidden // 32), dtype=torch.uint8, device=device),
            None,
        )

    st = int(torch.cuda.current_stream(device).cuda_stream)
    tile_mask = ws["tile_mask"][: plan.max_ctas]
    row_amax_w2 = ws["row_amax_w2"][:rows] if use_amax else None

    if use_zs:
        rc = lib.router_bf12_s8_zero_scatter_fwd(
            x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
            bias.data_ptr(), expert_bounds.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            ws["logits_ws"].data_ptr(),
            src2dst.data_ptr(), gammas.data_ptr(), routed.data_ptr(), tiles.data_ptr(),
            tile_mask.data_ptr(),
            0 if row_amax_w2 is None else row_amax_w2.data_ptr(),
            quantized[0].data_ptr(), quantized[1].data_ptr(),
            0 if quantized[2] is None else quantized[2].data_ptr(),
            0 if weight_amax is None else weight_amax.data_ptr(),
            0 if weight_denominator is None else weight_denominator.data_ptr(),
            tokens, float(route_scale), int(base_exp), int(num_local),
            int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0, int(pdl), st,
        )
        if rc != 0:
            raise RuntimeError(f"router_bf12_s8_zero_scatter_fwd failed with cudaError={rc}")
        tiles._inkling_tile_mask = tile_mask
        if row_amax_w2 is not None:
            tiles._inkling_row_amax_w2 = row_amax_w2
    else:
        rc = lib.router_bf12_exact_fwd(
            x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
            bias.data_ptr(), expert_bounds.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            0, 0, ws["logits_ws"].data_ptr(), 0,
            src2dst.data_ptr(), gammas.data_ptr(), routed.data_ptr(), tiles.data_ptr(),
            quantized[0].data_ptr(), quantized[1].data_ptr(),
            0 if quantized[2] is None else quantized[2].data_ptr(),
            0 if weight_amax is None else weight_amax.data_ptr(),
            0 if weight_denominator is None else weight_denominator.data_ptr(),
            tokens, float(route_scale), int(base_exp), int(num_local),
            int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0, int(pdl), st,
        )
        if rc != 0:
            raise RuntimeError(f"router_bf12_exact_fwd (s8) failed with cudaError={rc}")
        if row_amax_w2 is not None:
            row_amax_w2.zero_()
            tiles._inkling_row_amax_w2 = row_amax_w2

    return (src2dst, gammas, routed, tiles), quantized


def forward_experts_s8(
    experts,
    x: torch.Tensor,
    routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    plan,
    *,
    out_buf: torch.Tensor | None,
    nlive_buf: torch.Tensor | None = None,
    fuse_situ: bool | None = None,
    fuse_combine: bool | None = None,
    pdl: bool = True,
) -> torch.Tensor | None:
    """Execute S8 fused decode MoE experts (W13 [+SiTU] -> [situ_quantize] -> W2 [+Combine]).

    Returns combined `[T, 4608]` bf16 tensor (written directly into `out_buf` if provided),
    or `None` to cleanly fall back to stock S7 path if any library/condition isn't met.
    """
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if not s8_decode_eligible(
        tokens, hidden, experts.topk, plan, experts.gate_cap, experts.up_cap
    ):
        return None
    if experts.intermediate_size != 288:
        return None

    use_fs = S8_FUSE_SITU if fuse_situ is None else bool(fuse_situ)
    use_fc = S8_FUSE_COMBINE if fuse_combine is None else bool(fuse_combine)
    src2dst, gammas, rows, tiles = routing
    tile_mask = getattr(tiles, "_inkling_tile_mask", None)
    use_zs = tile_mask is not None
    if not (use_zs or use_fs or use_fc):
        return None

    device = x.device
    ws = get_workspace(device)
    st = int(torch.cuda.current_stream(device).cuda_stream)
    use_pdl = pdl and plan.rows <= _WS_MAX_ROWS
    act, act_scale, row_scale = quantized

    if experts.nvfp4:
        lib = get_nvfp4_lib()
        if lib is None:
            return None
        from . import nvfp4_ws5
        m13 = nvfp4_ws5.pack_w13_ws5_inplace(experts.w13_weight, experts.w13_weight_scale)
        m2 = nvfp4_ws5.pack_w2_ws5_inplace(experts.w2_weight, experts.w2_weight_scale)
        ws_base_13 = get_ws_base_s8(m13)
        ws_base_2 = get_ws_base_s8(m2)

        row_amax_w2 = getattr(tiles, "_inkling_row_amax_w2", None)
        if use_fs and row_amax_w2 is None:
            row_amax_w2 = ws["row_amax_w2"][: plan.rows]
            row_amax_w2.zero_()

        w13_out = ws["situ_out"][: plan.rows] if use_fs else torch.empty(
            (plan.rows, 576), dtype=torch.bfloat16, device=device
        )
        rc = lib.nvfp4_w13_ws5_s8_fwd(
            act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(),
            m13["slabs"].data_ptr(), ws_base_13.data_ptr(),
            m13["p_off"].data_ptr(), m13["patches"].data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            0 if not use_zs else tile_mask.data_ptr(),
            gammas.data_ptr() if use_fs else 0,
            row_amax_w2.data_ptr() if use_fs else 0,
            w13_out.data_ptr(),
            576, 4608, plan.max_ctas, plan.tile,
            1 if use_zs else 0, 1 if use_fs else 0,
            1 if use_pdl else 0, 1 if use_pdl else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"nvfp4_w13_ws5_s8_fwd failed with cudaError={rc}")

        if not use_fs:
            from .activation import swiglu_quantize
            m = plan.max_ctas
            act2, act2_s, act2_rs = swiglu_quantize(
                w13_out, gammas, experts.gate_cap, experts.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, *experts.scales("w2"),
            )
        else:
            act2, act2_s, act2_rs = w13_out, None, None

        w2_amax, w2_denom = experts.scales("w2")
        down = ws["down_ws"][: plan.rows] if use_fc else torch.empty(
            (plan.rows, 4608), dtype=torch.bfloat16, device=device
        )
        if use_fc and out_buf is None:
            out_buf = torch.empty((tokens, 4608), dtype=x.dtype, device=device)

        rc = lib.nvfp4_w2_ws5_s8_fwd(
            act2.data_ptr(),
            0 if act2_s is None else act2_s.data_ptr(),
            0 if act2_rs is None else act2_rs.data_ptr(),
            row_amax_w2.data_ptr() if use_fs else 0,
            w2_amax.data_ptr() if use_fs else 0,
            w2_denom.data_ptr() if use_fs else 0,
            m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
            m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            down.data_ptr(),
            src2dst.data_ptr() if use_fc else 0,
            0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
            ws["panel_done"].data_ptr() if use_fc else 0,
            out_buf.data_ptr() if use_fc else 0,
            4608, 288, plan.max_ctas, plan.tile, tokens,
            1 if use_fs else 0, 1 if use_fc else 0,
            2 if use_pdl else 0, 1 if use_pdl else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"nvfp4_w2_ws5_s8_fwd failed with cudaError={rc}")
    else:
        lib = get_mx7_lib()
        if lib is None:
            return None
        from . import trtllm_bmm
        pk13 = trtllm_bmm._mx7_get_packed(experts.w13_weight, experts.w13_weight_scale, 0, 32, 256)
        pk2 = trtllm_bmm._mx7_get_packed(experts.w2_weight, experts.w2_weight_scale, 1, 128, 32)
        if pk13 is None or pk2 is None:
            return None
        slabs13, ws3_13, flips13, foff13, bws13 = pk13
        slabs2, ws3_2, flips2, foff2, bws2 = pk2

        w13_out = ws["situ_out"][: plan.rows] if use_fs else torch.empty(
            (plan.rows, 576), dtype=torch.bfloat16, device=device
        )
        rc = lib.mx7_w13_s8_fwd(
            act.data_ptr(), act_scale.data_ptr(),
            slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), foff13.data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            0 if not use_zs else tile_mask.data_ptr(),
            gammas.data_ptr() if use_fs else 0,
            w13_out.data_ptr(),
            576, 4608, plan.max_ctas, plan.tile, int(bws13),
            1 if use_zs else 0, 1 if use_fs else 0,
            1 if use_pdl else 0, 1 if use_pdl else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_w13_s8_fwd failed with cudaError={rc}")

        if not use_fs:
            from .activation import swiglu_quantize
            m = plan.max_ctas
            act2, act2_s, _ = swiglu_quantize(
                w13_out, gammas, experts.gate_cap, experts.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, None, None,
            )
        else:
            act2, act2_s = w13_out, None

        down = ws["down_ws"][: plan.rows] if use_fc else torch.empty(
            (plan.rows, 4608), dtype=torch.bfloat16, device=device
        )
        if use_fc and out_buf is None:
            out_buf = torch.empty((tokens, 4608), dtype=x.dtype, device=device)

        rc = lib.mx7_w2_s8_fwd(
            act2.data_ptr(),
            0 if act2_s is None else act2_s.data_ptr(),
            slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), foff2.data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            down.data_ptr(),
            src2dst.data_ptr() if use_fc else 0,
            0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
            ws["panel_done"].data_ptr() if use_fc else 0,
            out_buf.data_ptr() if use_fc else 0,
            4608, 288, plan.max_ctas, plan.tile, tokens, int(bws2),
            1 if use_fs else 0, 1 if use_fc else 0,
            2 if use_pdl else 0, 1 if use_pdl else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_w2_s8_fwd failed with cudaError={rc}")

    if use_fc:
        return out_buf
    from .moe import combine
    return combine(down, src2dst, experts.topk, x.dtype, out=out_buf)
