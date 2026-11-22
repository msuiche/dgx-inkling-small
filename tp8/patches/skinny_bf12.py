# SPDX-License-Identifier: Apache-2.0
"""SM12x (GB10) 0-ULP Lossless BF12_EXACT Skinny GEMM (CUDA C++ mma.sync.m16n8k16).

Exploits BF16 exponent entropy:
  - 99.99% of weights fall in 16 consecutive exponents [base_exp - 15 .. base_exp].
  - Packed as 1.500 B/weight (1024 B sm + 512 B e4 = 1536 B slabs, 100% 128-bit ld.global.cs.v4.u32).
  - Remaining ~0.01% outliers are patched directly into MMA registers BEFORE mma.sync.m16n8k16
    -> LITERAL 0-ULP BIT-EXACT (ndiff == 0) with skinny.py across all M=1..16!
  - Achieves 315.8 eff GB/s on qkvr_loc (1896x4608) on 244 GB/s GB10.

Enabled via INKLING_SKINNY_BF12=1. Pre-packed once before CUDA graph capture via prepack().
"""

import ctypes
import os
import torch

ENABLED = os.environ.get("INKLING_SKINNY_BF12", "0").strip().lower() not in ("0", "", "off", "false", "no")
MAX_M = int(os.environ.get("INKLING_BF12_MAX_M", "64") or 64)
_LIB_PATH = os.environ.get("INKLING_BF12_LIB", "/rdma/libbf12_exact_v3.so")
_LIB = None
_LIB_TRIED = False


def _get_lib():
    global _LIB, _LIB_TRIED
    if not _LIB_TRIED:
        _LIB_TRIED = True
        if os.path.exists(_LIB_PATH):
            try:
                lib = ctypes.CDLL(_LIB_PATH)
                lib.bf12_exact_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint32,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.bf12_exact_fwd.restype = ctypes.c_int
                _LIB = lib
            except Exception as e:
                print(f"[inkling-bf12] failed to load {_LIB_PATH}: {e!r}", flush=True)
                _LIB = None
    return _LIB


# cfg_id -> (warps, sub_n, bn)
_CFGS = {
    0: (4, 2, 64),
    1: (4, 1, 32),
    2: (2, 2, 32),
    3: (3, 2, 48),
    4: (2, 1, 16),
}

# Tuned (cfg_id, split) per (N, K) for Inkling2 dense projections (matching skinny.py split-K exactly!).
_SHAPES = {
    (1896, 4608): (1, 6),  # local qkvr:  BN=32 W=4 SUB=1 sp=6 (315.8 eff GB/s, -21.2 us)
    (1640, 4608): (0, 6),  # global q_side: BN=64 W=4 SUB=2 sp=6 (273.2 eff GB/s, -12.1 us)
    (4608, 768):  (1, 1),  # wo_ud:       BN=32 W=4 SUB=1 sp=1 (256.9 eff GB/s, -4.0 us)
    (128, 4608):  (2, 9),  # global k/v:  BN=32 W=2 SUB=2 sp=9 (0-ULP with skinny.py sp=9, -9.6 us)
}

# (data_ptr, N, K) -> (slabs, outliers, o_off, base, ks, n_active, cfg_id, split)
_CACHE: dict[tuple[int, int, int], tuple] = {}


@torch.no_grad()
def prepack(w: torch.Tensor) -> torch.Tensor | None:
    """Pack a bf16 [N, K] dense weight into 0-ULP BF12_EXACT slab format (outside graph capture).
    Returns the packed slabs tensor (for L2 prefetch), or None if not eligible."""
    if not ENABLED or w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.stride(1) != 1:
        return None
    if _get_lib() is None:
        return None
    N, K = int(w.shape[0]), int(w.shape[1])
    cfg = _SHAPES.get((N, K))
    if cfg is None:
        return None
    key = (int(w.data_ptr()), N, K)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit[0]
    if torch.cuda.is_current_stream_capturing():
        return None

    cfg_id, split = cfg
    warps, sub_n, bn = _CFGS[cfg_id]
    assert K % (split * 128) == 0
    ks = K // split
    n_chunks = ks // 128
    n_tiles = (N + bn - 1) // bn
    n_pad = n_tiles * bn

    nz_rows = (w != 0).any(dim=1).nonzero(as_tuple=False)
    n_active = int(nz_rows[-1].item()) + 1 if nz_rows.numel() > 0 else 0
    wp = torch.zeros((n_pad, K), dtype=torch.bfloat16, device=w.device)
    if n_active > 0:
        wp[:n_active].copy_(w[:n_active])
    u = wp.view(torch.uint16).to(torch.int32)
    exp = (u >> 7) & 0xFF
    cnt = torch.bincount(exp[:n_active].flatten(), minlength=256) if n_active > 0 else torch.ones(256, dtype=torch.int64, device=w.device)
    base = max(range(15, 255), key=lambda b: cnt[b - 15 : b + 1].sum().item())

    delta = base - exp
    in_win = (delta >= 0) & (delta <= 15)
    if n_active < n_pad:
        in_win[n_active:] = True
    delta = torch.where(in_win, delta, torch.zeros_like(delta))
    if n_active < n_pad:
        delta[n_active:] = 0

    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    d4 = delta.to(torch.uint8)

    def to_mma(x: torch.Tensor) -> torch.Tensor:
        v = x.view(n_tiles, warps, sub_n, 8, split, n_chunks, 8, 2, 4, 2)
        v = v.permute(4, 0, 1, 5, 2, 6, 3, 8, 7, 9).contiguous()
        return v.view(split * n_tiles * warps, n_chunks * sub_n, 8, 32, 4)

    wtot = split * n_tiles * warps
    spw = n_chunks * sub_n
    sm_m = to_mma(sm)
    d4_m = to_mma(d4)
    inw_m = to_mma(in_win)
    u_m = to_mma(u)

    sm_slab = (
        sm_m.view(wtot, spw, 2, 4, 32, 4)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(wtot, spw, 1024)
    )
    d4_lane = d4_m.permute(0, 1, 3, 2, 4).contiguous()
    e4_slab = (
        torch.stack(
            [d4_lane[..., 0] | (d4_lane[..., 1] << 4), d4_lane[..., 2] | (d4_lane[..., 3] << 4)],
            dim=-1,
        )
        .contiguous()
        .view(wtot, spw, 512)
    )
    slabs = torch.cat([sm_slab, e4_slab], dim=2).contiguous().view(-1)

    bad = (~inw_m).nonzero(as_tuple=False)
    if bad.numel() > 0:
        widx, sl_i, s_i, lane_i, j_i = bad[:, 0], bad[:, 1], bad[:, 2], bad[:, 3], bad[:, 4]
        step_id = (sl_i << 3) | s_i
        lj = (lane_i << 2) | j_i
        uval = u_m[widx, sl_i, s_i, lane_i, j_i].to(torch.int64) & 0xFFFF
        order = torch.argsort((widx << 24) | (step_id << 8) | lj)
        widx, step_id, lj, uval = widx[order], step_id[order], lj[order], uval[order]
        outliers = ((step_id << 32) | (lj << 16) | uval).to(torch.uint64).contiguous()
        counts = torch.bincount(widx, minlength=wtot).to(torch.int32)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=w.device)
        o_off[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    else:
        outliers = torch.zeros(1, dtype=torch.uint64, device=w.device)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=w.device)

    _CACHE[key] = (slabs, outliers, o_off, base, ks, n_active, cfg_id, split)
    return slabs


def l2pf_tensor(w: torch.Tensor | None, num_tokens: int | None = None) -> torch.Tensor | None:
    if not ENABLED or w is None or w.dim() != 2:
        return w
    if num_tokens is not None and num_tokens > MAX_M:
        return w
    hit = _CACHE.get((int(w.data_ptr()), int(w.shape[0]), int(w.shape[1])))
    return hit[0] if hit is not None else w


def get_packed(w: torch.Tensor):
    if not ENABLED or w.dim() != 2:
        return None
    return _CACHE.get((int(w.data_ptr()), int(w.shape[0]), int(w.shape[1])))


def bf12_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    pdl: bool = False,
    trig_reduce: bool = False,
) -> torch.Tensor | None:
    """Run 0-ULP BF12_EXACT v3 CUDA GEMM if w is prepacked and M <= MAX_M (default 64); else return None."""
    if not ENABLED or x.dim() != 2 or x.shape[0] > MAX_M or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        return None
    lib = _get_lib()
    if lib is None:
        return None
    entry = get_packed(w)
    if entry is None:
        return None
    slabs, outliers, o_off, base, ks, n_active, cfg_id, split = entry
    M, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    part = None if split == 1 else torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    rc = lib.bf12_exact_fwd(
        x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
        0 if part is None else part.data_ptr(), out.data_ptr(),
        M, N, n_active, K, int(x.stride(0)), ks, int(out.stride(0)),
        cfg_id, split, base, 1 if pdl else 0, 1 if trig_reduce else 0,
        int(torch.cuda.current_stream().cuda_stream),
    )
    if rc != 0:
        raise RuntimeError(f"bf12_exact_fwd failed with cudaError={rc}")
    return out
