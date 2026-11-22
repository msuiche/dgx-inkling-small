# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP Bit-Exact BF12 Fused MoE Router + Input Quantization.

Packs `gate.weight` [384, 4608] bf16 (3.54 MB) into lossless BF12 slabs
(2.65 MB, -25.0%) + tiny outlier list (~255 outliers/layer = 2 KB), and
executes a 1-wave 48-CTA fused CUDA kernel (`librouter_bf12_exact.so`) that
matches `ops_moe._fused_route_kernel` to LITERAL 0 DIFFS across every output
tensor (src2dst, gammas, routed, tiles, QValues, QScales, QRowScales) and
intermediate router logits.
"""

from __future__ import annotations

import ctypes
import os
import weakref
import torch

ENABLED = os.environ.get("INKLING_ROUTER_BF12", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_LIB_PATH = os.environ.get("INKLING_ROUTER_BF12_LIB", "/rdma/librouter_bf12_exact.so")
_LIB = None
_LIB_TRIED = False

# (data_ptr, E, H) -> (weakref_w, slabs, outliers, o_off, base_exp)
_CACHE: dict[tuple[int, int, int], tuple] = {}
# Persistent per-device tiny workspace (partials [5, 16, 384] fp32 = 120 KB)
_WS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_lib(lib_path: str | None = None):
    global _LIB, _LIB_TRIED
    path = lib_path or _LIB_PATH
    if not os.path.exists(path) and lib_path is None:
        local_fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librouter_bf12_exact.so")
        if os.path.exists(local_fallback):
            path = local_fallback
    if _LIB is not None and lib_path is None:
        return _LIB
    if not _LIB_TRIED or lib_path is not None:
        _LIB_TRIED = True
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.router_bf12_exact_fwd.argtypes = [
                ctypes.c_void_p,  # X
                ctypes.c_void_p,  # SLABS
                ctypes.c_void_p,  # OUTLIERS
                ctypes.c_void_p,  # O_OFF
                ctypes.c_void_p,  # Bias
                ctypes.c_void_p,  # ExpertBounds
                ctypes.c_void_p,  # Padding
                ctypes.c_void_p,  # NLivePtr
                ctypes.c_void_p,  # Counters
                ctypes.c_void_p,  # Masks
                ctypes.c_void_p,  # LogitsWS
                ctypes.c_void_p,  # KeysWS
                ctypes.c_void_p,  # Src2Dst
                ctypes.c_void_p,  # Gammas
                ctypes.c_void_p,  # Rows
                ctypes.c_void_p,  # Tiles
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
            lib.router_bf12_exact_fwd.restype = ctypes.c_int
            _LIB = lib
    return _LIB


def _get_ws(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = device.index or 0
    ws = _WS.get(idx)
    if ws is None:
        partials_ws = torch.empty((5, 16, 384), dtype=torch.float32, device=device)
        unused_ws = torch.empty((1,), dtype=torch.uint64, device=device)
        ws = (partials_ws, unused_ws)
        _WS[idx] = ws
    return ws


@torch.no_grad()
def prepack(w: torch.Tensor) -> torch.Tensor | None:
    """Pack bf16 gate.weight [384, 4608] into 0-ULP BF12 router slabs (48 CTAs x 36 slabs).

    Returns the packed slabs tensor (2.654 MB, suitable for L2 prefetch), or None if ineligible.
    """
    if (
        not ENABLED
        or w is None
        or w.dtype != torch.bfloat16
        or tuple(w.shape) != (384, 4608)
        or not w.is_contiguous()
    ):
        return None
    key = (int(w.data_ptr()), 384, 4608)
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is w:
        return hit[1]
    if torch.cuda.is_current_stream_capturing():
        return None

    device = w.device
    u = w.view(torch.uint16).to(torch.int32)
    exp = (u >> 7) & 0xFF
    cnt = torch.bincount(exp.flatten(), minlength=256)
    base = max(range(15, 255), key=lambda b: int(cnt[b - 15 : b + 1].sum().item()))

    delta = base - exp
    in_win = (delta >= 0) & (delta <= 15)
    d4 = torch.where(in_win, delta, torch.zeros_like(delta)).to(torch.uint8)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)

    # Reshape [384, 4608] = [48 CTAs, 8 experts, 36 slabs of 128 cols]:
    # Slab layout inside each CTA bid:
    #   warp 0: slabs 0..7   (k = 0..1023)
    #   warp 1: slabs 8..15  (k = 1024..2047)
    #   warp 2: slabs 16..23 (k = 2048..3071)
    #   warp 3: slabs 24..31 (k = 3072..4095)
    #   warp 4: slabs 32..35 (k = 4096..4607)
    # Within each slab (8 experts x 128 cols = 1024 elems):
    #   [8 steps of 16 cols] x [2 halves of 8 cols] x [4 t-groups of 2 cols]
    #   MMA lane = g * 4 + t (g = expert 0..7, t = 0..3), j = half * 2 + pair (0..3)
    def to_slab_mma(x: torch.Tensor) -> torch.Tensor:
        # [48, 8(g), 36(slab), 8(s), 2(half), 4(t), 2(pair)]
        v = x.view(48, 8, 36, 8, 2, 4, 2)
        # -> [48, 36, 8(s), 8(g), 4(t), 2(half), 2(pair)]
        v = v.permute(0, 2, 3, 1, 5, 4, 6).contiguous()
        return v.view(48, 36, 8, 32, 4)

    sm_m = to_slab_mma(sm)
    d4_m = to_slab_mma(d4)
    inw_m = to_slab_mma(in_win)
    u_m = to_slab_mma(u)

    # sm_slab: [48, 36, 2(chunk4s), 32(lane), 4(s_in_chunk), 4(j)] -> 1024 B/slab
    sm_slab = (
        sm_m.view(48, 36, 2, 4, 32, 4)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(48, 36, 1024)
    )
    # e4_slab: [48, 36, 32(lane), 8(s), 2(packed nibbles)] -> 512 B/slab
    d4_lane = d4_m.permute(0, 1, 3, 2, 4).contiguous()
    e4_slab = (
        torch.stack(
            [
                d4_lane[..., 0] | (d4_lane[..., 1] << 4),
                d4_lane[..., 2] | (d4_lane[..., 3] << 4),
            ],
            dim=-1,
        )
        .contiguous()
        .view(48, 36, 512)
    )
    slabs = torch.cat([sm_slab, e4_slab], dim=2).contiguous().view(-1)

    # Build per-warp outlier table (240 warps = 48 CTAs * 5 warps)
    # Slab index sl_global in [0..35] maps to:
    #   warp = sl_global // 8  (0..4; warp 4 gets 32..35)
    #   sl_local = sl_global % 8 (0..7 for warps 0..3, 0..3 for warp 4)
    bad = (~inw_m).nonzero(as_tuple=False)
    wtot = 48 * 5
    if bad.numel() > 0:
        bid_i = bad[:, 0]
        sl_g = bad[:, 1]
        s_i = bad[:, 2]
        lane_i = bad[:, 3]
        j_i = bad[:, 4]
        warp_i = sl_g >> 3
        sl_loc = sl_g & 7
        widx = bid_i * 5 + warp_i
        step_id = (sl_loc << 3) | s_i
        lj = (lane_i << 2) | j_i
        uval = u_m[bid_i, sl_g, s_i, lane_i, j_i].to(torch.int64) & 0xFFFF
        order = torch.argsort((widx << 24) | (step_id << 8) | lj)
        widx, step_id, lj, uval = widx[order], step_id[order], lj[order], uval[order]
        outliers = ((step_id << 32) | (lj << 16) | uval).to(torch.uint64).contiguous()
        counts = torch.bincount(widx, minlength=wtot).to(torch.int32)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=device)
        o_off[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    else:
        outliers = torch.zeros(1, dtype=torch.uint64, device=device)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=device)

    _CACHE[key] = (weakref.ref(w), slabs, outliers, o_off, base)
    return slabs


def l2pf_tensor(w: torch.Tensor | None, num_tokens: int | None = None) -> torch.Tensor | None:
    if not ENABLED or w is None or tuple(w.shape) != (384, 4608):
        return w
    if num_tokens is not None and num_tokens > 16:
        return w
    hit = _CACHE.get((int(w.data_ptr()), 384, 4608))
    return hit[1] if (hit is not None and hit[0]() is w) else w


def route_bf12(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    counters: torch.Tensor,
    counts: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    plan,
    weight_amax: torch.Tensor | None,
    weight_denominator: torch.Tensor | None,
    is_padding: torch.Tensor | None = None,
    nlive_buf: torch.Tensor | None = None,
    pdl: int | bool = 2,
    lib_path: str | None = None,
):
    """0-ULP BF12 fused route + quantize for T <= 16, E = 384, H = 4608, topk = 8.

    pdl: 0 = off, 1 = internal PDL (K1 -> K2), 2 = internal PDL + K2 triggers next kernel (w13 PDL_WAIT_MODE=1).
    """
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if (
        not ENABLED
        or tokens <= 0
        or tokens > 16
        or hidden != 4608
        or topk != 8
        or tuple(gate_weight.shape) != (384, 4608)
        or plan.tile > 64
        or plan.tile < tokens
    ):
        return None
    lib = _get_lib(lib_path)
    if lib is None:
        return None
    entry = _CACHE.get((int(gate_weight.data_ptr()), 384, 4608))
    if entry is None or entry[0]() is not gate_weight:
        if torch.cuda.is_current_stream_capturing():
            return None
        prepack(gate_weight)
        entry = _CACHE.get((int(gate_weight.data_ptr()), 384, 4608))
        if entry is None or entry[0]() is not gate_weight:
            return None

    _, slabs, outliers, o_off, base_exp = entry
    device = x.device
    nvfp4 = weight_amax is not None
    num_local = counts.numel()
    # Dec160 S8 fast path: 48-CTA zero-tail K1 + 1-CTA Zero-Scatter SMEM-Route K2
    try:
        from . import moe_s8_runtime as _s8
        if _s8.S8_ROUTER or _s8.S8_ZERO_SCATTER or _s8.S8_FUSE_SITU:
            _pdl_i = int(pdl) if not isinstance(pdl, bool) else (2 if pdl else 0)
            _s8_res = _s8.route_bf12_s8(
                x, slabs, outliers, o_off, int(base_exp), bias, topk, float(route_scale),
                expert_bounds, plan, weight_amax, weight_denominator, int(num_local),
                is_padding=is_padding, nlive_buf=nlive_buf, pdl=_pdl_i,
            )
            if _s8_res is not None:
                return _s8_res
    except Exception:
        pass
    n = tokens * topk
    rows = plan.rows

    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)
    if nvfp4:
        quantized = (
            torch.empty((rows, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((rows, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
        )
    else:
        quantized = (
            torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((rows, hidden // 32), dtype=torch.uint8, device=device),
            None,
        )

    logits_ws, keys_ws = _get_ws(device)
    rc = lib.router_bf12_exact_fwd(
        x.data_ptr(),
        slabs.data_ptr(),
        outliers.data_ptr(),
        o_off.data_ptr(),
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        0 if is_padding is None else is_padding.data_ptr(),
        0 if nlive_buf is None else nlive_buf.data_ptr(),
        counters.data_ptr(),
        counts.data_ptr(),
        logits_ws.data_ptr(),
        keys_ws.data_ptr(),
        src2dst.data_ptr(),
        gammas.data_ptr(),
        routed.data_ptr(),
        tiles.data_ptr(),
        quantized[0].data_ptr(),
        quantized[1].data_ptr(),
        0 if quantized[2] is None else quantized[2].data_ptr(),
        0 if weight_amax is None else weight_amax.data_ptr(),
        0 if weight_denominator is None else weight_denominator.data_ptr(),
        tokens,
        float(route_scale),
        int(base_exp),
        int(num_local),
        int(plan.tile),
        int(plan.max_ctas),
        1 if nvfp4 else 0,
        int(pdl) if not isinstance(pdl, bool) else (2 if pdl else 0),
        int(torch.cuda.current_stream().cuda_stream),
    )
    if rc != 0:
        raise RuntimeError(f"router_bf12_exact_fwd failed with cudaError={rc}")
    return (src2dst, gammas, routed, tiles), quantized
