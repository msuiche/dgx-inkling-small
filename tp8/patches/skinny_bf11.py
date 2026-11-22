# SPDX-License-Identifier: Apache-2.0
"""SM12x (GB10) 0-ULP Lossless BF11_FLIP_EXACT Skinny GEMM (CUDA C++ mma.sync.m16n8k16).

Compresses BF16 dense projections from 2.00 B/w -> ~1.408-1.423 B/w (-29.4% vs BF16, -6.1% vs BF12):
  - 98.6% of weights fall in [base_exp - 7 .. base_exp]   -> 3-bit e3 (b3 = 0)
  -  1.4% of weights fall in [base_exp - 15 .. base_exp-8] -> 3-bit e3 + 1-bit sparse flip (b3 = 1)
  -  0.01% true outliers outside [base_exp - 15 .. base_exp] -> exact 64-bit inline patch list
  - SLABS: 1408 B/slab (1024 B sm + 384 B e3, 11x128B aligned), FLIPS: ~28 B/slab (__ldg cached)
  - Single-kernel last-CTA semaphore finalize (eliminates separate reduce kernel launch & gap!)
  - Dead-warp skip, optional NLIVE dynamic row skip, PDL (`::: "memory"`), TRIG on DIRECT & split-K,
    and Fused Dual-KV (k_proj + v_proj in ONE launch for global layers!).
  - LITERAL 0-ULP BIT-EXACT (ndiff == 0) with skinny.py across all M=1..16!

Enabled via INKLING_SKINNY_BF11=1. Pre-packed once before CUDA graph capture via prepack().
"""

import ctypes
import os
import torch

ENABLED = os.environ.get("INKLING_SKINNY_BF11", "0").strip().lower() not in ("0", "", "off", "false", "no")
USE_NLIVE = os.environ.get("INKLING_BF11_NLIVE", "1").strip().lower() not in ("0", "", "off", "false", "no")
USE_DUAL_KV = os.environ.get("INKLING_BF11_DUAL_KV", "1").strip().lower() not in ("0", "", "off", "false", "no")
C32_ENABLED = os.environ.get("INKLING_BF11_C32", "0").strip().lower() not in ("0", "", "off", "false", "no")
C32_MAX_M = max(16, int(os.environ.get("INKLING_BF11_C32_MAX_M", "256") or 256))
C32_BM = int(os.environ.get("INKLING_BF11_C32_BM", "0") or 0)
C32_KORDER = int(os.environ.get("INKLING_BF11_C32_KORDER", "0") or 0)
C32_QUALIFY = os.environ.get("INKLING_BF11_C32_QUALIFY", "1").strip().lower() not in ("0", "", "off", "false", "no")
C32_EXACT_REF = os.environ.get("INKLING_BF11_C32_EXACT_REF", "torch").strip().lower()
C32_DUAL_KV = os.environ.get("INKLING_BF11_C32_DUAL_KV", "1").strip().lower() not in ("0", "", "off", "false", "no")
_LIB_PATH = os.environ.get(
    "INKLING_BF11_LIB",
    "/rdma/libbf11_flip_exact_c32.so" if C32_ENABLED else "/rdma/libbf11_flip_exact.so",
)
_LIB = None
_LIB_TRIED = False
_HAS_C32 = False
_C32_QUAL_OK: dict[tuple[int, int, int], bool] = {}  # (N, K, M) -> 0-ULP verified
_C32_DUAL_QUAL_OK: dict[tuple[int, int, int], bool] = {}  # (N_each, K, M) -> 0-ULP verified
_C32_CANONICAL_MS = (24, 32, 36, 48, 60, 64, 72, 96, 112, 128, 160, 192, 224, 256)


def _get_lib():
    global _LIB, _LIB_TRIED, _HAS_C32
    if not _LIB_TRIED:
        _LIB_TRIED = True
        if os.path.exists(_LIB_PATH):
            try:
                lib = ctypes.CDLL(_LIB_PATH)
                lib.bf11_flip_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.bf11_flip_fwd.restype = ctypes.c_int
                lib.bf11_flip_dual_kv_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.bf11_flip_dual_kv_fwd.restype = ctypes.c_int
                if hasattr(lib, "bf11_flip_c32_fwd") and hasattr(lib, "bf11_flip_dual_kv_c32_fwd"):
                    lib.bf11_flip_c32_fwd.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_c32_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_c32_fwd.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_dual_kv_c32_fwd.restype = ctypes.c_int
                    _HAS_C32 = True
                _LIB = lib
            except Exception as e:
                print(f"[inkling-bf11] failed to load {_LIB_PATH}: {e!r}", flush=True)
                _LIB = None
    return _LIB


def active_max_m() -> int:
    if not ENABLED:
        return 0
    if C32_ENABLED:
        _get_lib()
        if _HAS_C32:
            return C32_MAX_M
    return 16


# cfg_id -> (warps, sub_n, bn)
_CFGS = {
    0: (4, 2, 64),
    1: (4, 1, 32),
    2: (2, 2, 32),
    4: (2, 1, 16),
    5: (1, 1, 8),
}

# Tuned (cfg_id, split) per (N, K) matching skinny.py split-K exactly (0-ULP!).
# cfg_id=1 (BN=32 W=4 SUB=1) achieves 237-245 GB/s DRAM (331-345 eff GB/s) on BOTH M=8 and M=16.
_QKVR_CFG = int(os.environ.get("INKLING_BF11_QKVR_CFG", "1"))      # 4 (BN=16 W=2: 714 CTAs = 14.875/SM) or 1 (360 CTAs)
_Q_SIDE_CFG = int(os.environ.get("INKLING_BF11_QSIDE_CFG", "1"))   # 1 (45.1/46.4us M=8/16) or 4 (43.3us M=8)
_KV_CFG = int(os.environ.get("INKLING_BF11_KV_CFG", "4"))          # 4 (BN=16 W=2: 144 CTAs = 3.0 CTAs/SM, zero wave tail!)
_SHAPES = {
    (1896, 4608): (_QKVR_CFG, 6),   # local qkvr:  cfg=4 eliminates 7.5 CTAs/SM wave tail (0-ULP identical split=6)
    (1640, 4608): (_Q_SIDE_CFG, 6), # global q_side: 45.13/46.41 us (334.9 eff GB/s; cfg=4: 43.31 us)
    (4608, 768):  (1, 1),           # wo_ud:       21.06/21.35 us (336.0/331.4 eff GB/s)
    (128, 4608):  (_KV_CFG, 9),     # global k/v:  cfg=4 eliminates 1.5-wave dual-KV tail (0-ULP, identical split=9)
}

# (data_ptr, N, K) -> (slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn)
_CACHE: dict[tuple[int, int, int], tuple] = {}
# 2x ping-pong persistent L2-hot split-K partial & self-resetting semaphore buffers per device.
# Preallocated in prepack() before CUDA graph capture; deterministic flip on every split-K call
# guarantees adjacent split-K calls never collide (including under PDL overlap & graph replay).
_PART_BUFS: dict[int, list[torch.Tensor]] = {}
_SEM_BUFS: dict[int, list[torch.Tensor]] = {}
_SCRATCH_IDX: dict[int, int] = {}
_MIN_PART_ELEMS = 262144  # 1.0 MiB/slot (>= 60*6*16*32 = 184,320 elems for max M=16 shape)
_MIN_SEM_TILES = 1024


def _ensure_scratch(device: torch.device, min_part_elems: int, min_sem_tiles: int) -> None:
    dev_idx = device.index if device.index is not None else 0
    need_p = max(_MIN_PART_ELEMS, int(min_part_elems))
    need_s = max(_MIN_SEM_TILES, int(min_sem_tiles))
    parts = _PART_BUFS.get(dev_idx)
    if parts is None or parts[0].numel() < need_p:
        _PART_BUFS[dev_idx] = [
            torch.empty(need_p, dtype=torch.float32, device=device),
            torch.empty(need_p, dtype=torch.float32, device=device),
        ]
    sems = _SEM_BUFS.get(dev_idx)
    if sems is None or sems[0].numel() < need_s:
        _SEM_BUFS[dev_idx] = [
            torch.zeros(need_s, dtype=torch.int32, device=device),
            torch.zeros(need_s, dtype=torch.int32, device=device),
        ]
    if dev_idx not in _SCRATCH_IDX:
        _SCRATCH_IDX[dev_idx] = 0


def _next_scratch(device: torch.device, min_part_elems: int, min_sem_tiles: int) -> tuple[torch.Tensor, torch.Tensor]:
    dev_idx = device.index if device.index is not None else 0
    parts = _PART_BUFS.get(dev_idx)
    sems = _SEM_BUFS.get(dev_idx)
    if parts is None or parts[0].numel() < min_part_elems or sems is None or sems[0].numel() < min_sem_tiles:
        if torch.cuda.is_current_stream_capturing():
            # Fallback should never trigger when prepack() ran prior to graph capture.
            part = torch.empty(min_part_elems, dtype=torch.float32, device=device)
            if sems is not None and sems[0].numel() >= min_sem_tiles:
                idx = _SCRATCH_IDX.get(dev_idx, 0)
                _SCRATCH_IDX[dev_idx] = idx ^ 1
                return part, sems[idx]
            return part, torch.zeros(max(_MIN_SEM_TILES, min_sem_tiles), dtype=torch.int32, device=device)
        _ensure_scratch(device, min_part_elems, min_sem_tiles)
        parts = _PART_BUFS[dev_idx]
        sems = _SEM_BUFS[dev_idx]
    idx = _SCRATCH_IDX.get(dev_idx, 0)
    _SCRATCH_IDX[dev_idx] = idx ^ 1
    return parts[idx], sems[idx]


def _get_sem(device: torch.device, min_tiles: int) -> torch.Tensor:
    _ensure_scratch(device, _MIN_PART_ELEMS, min_tiles)
    dev_idx = device.index if device.index is not None else 0
    return _SEM_BUFS[dev_idx][0]


def _rowar_nlive_ptr(M: int) -> int:
    if not USE_NLIVE:
        return 0
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as rar
        if rar.ROWAR.get("scope") == M:
            buf = rar.ROWAR.get("buf")
            if buf is not None:
                return int(buf.data_ptr())
    except Exception:
        pass
    return 0


@torch.no_grad()
def prepack(w: torch.Tensor) -> torch.Tensor | None:
    """Pack a bf16 [N, K] dense weight into 0-ULP BF11_FLIP_EXACT format (outside graph capture).
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
    spw = n_chunks * sub_n
    assert spw <= 12, f"BF11_FLIP WARP_META supports spw <= 12, got {spw}"

    nz_rows = (w != 0).any(dim=1).nonzero(as_tuple=False)
    n_active = int(nz_rows[-1].item()) + 1 if nz_rows.numel() > 0 else 0
    wp = torch.zeros((n_pad, K), dtype=torch.bfloat16, device=w.device)
    if n_active > 0:
        wp[:n_active].copy_(w[:n_active])
    u = wp.view(torch.uint16).to(torch.int32)
    exp = (u >> 7) & 0xFF

    if n_active > 0:
        cnt = torch.bincount(exp[:n_active].flatten(), minlength=256)
        tot_n = int(cnt.sum().item())
        best_b, best_cost = 122, 1 << 60
        for b in range(15, 255):
            c8 = int(cnt[b - 7 : b + 1].sum().item())
            c16 = int(cnt[b - 15 : b + 1].sum().item())
            cost = 2 * (c16 - c8) + 16 * (tot_n - c16)
            if cost < best_cost:
                best_cost, best_b = cost, b
        base = best_b
    else:
        base = 122

    delta = base - exp
    in_win16 = (delta >= 0) & (delta <= 15)
    if n_active < n_pad:
        in_win16[n_active:] = True
    is_flip = in_win16 & (delta >= 8)
    if n_active < n_pad:
        is_flip[n_active:] = False
    is_out = ~in_win16
    d3 = torch.where(in_win16, delta & 7, torch.zeros_like(delta)).to(torch.int64)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)

    def to_mma(x: torch.Tensor) -> torch.Tensor:
        v = x.view(n_tiles, warps, sub_n, 8, split, n_chunks, 8, 2, 4, 2)
        v = v.permute(4, 0, 1, 5, 2, 6, 3, 8, 7, 9).contiguous()
        return v.view(split * n_tiles * warps, spw, 8, 32, 4)

    wtot = split * n_tiles * warps
    sm_m, d3_m, fl_m, out_m, u_m = to_mma(sm), to_mma(d3), to_mma(is_flip), to_mma(is_out), to_mma(u)

    sm_slab = sm_m.view(wtot, spw, 2, 4, 32, 4).permute(0, 1, 2, 4, 3, 5).contiguous().view(wtot, spw, 1024)
    d3_lane = d3_m.permute(0, 1, 3, 2, 4).reshape(wtot, spw, 32, 32)
    shifts_lo = torch.arange(0, 21, device=w.device, dtype=torch.int64) * 3
    lo63 = (d3_lane[..., :21] << shifts_lo).sum(dim=-1)
    v21 = d3_lane[..., 21]
    e0 = (lo63 & 0xFFFFFFFF).to(torch.int32)
    e1 = (((lo63 >> 32) & 0x7FFFFFFF) | ((v21 & 1) << 31)).to(torch.int32)
    shifts_hi = torch.arange(0, 10, device=w.device, dtype=torch.int64) * 3 + 2
    e2 = ((v21 >> 1) | (d3_lane[..., 22:] << shifts_hi).sum(dim=-1)).to(torch.int32)
    e3_slab = torch.stack([e0, e1, e2], dim=2).contiguous().view(torch.uint8).view(wtot, spw, 384)
    slabs = torch.cat([sm_slab, e3_slab], dim=2).contiguous().view(-1)

    f_idx = fl_m.nonzero(as_tuple=False)
    total_slabs_all = wtot * spw
    if f_idx.numel() > 0:
        widx, sl_i, s_i, lane_i, j_i = f_idx[:, 0], f_idx[:, 1], f_idx[:, 2], f_idx[:, 3], f_idx[:, 4]
        slab_uid = widx * spw + sl_i
        order = torch.argsort(slab_uid)
        widx, sl_i, s_i, lane_i, j_i, slab_uid = (
            widx[order], sl_i[order], s_i[order], lane_i[order], j_i[order], slab_uid[order]
        )
        counts_sl = torch.bincount(slab_uid, minlength=total_slabs_all)
        if int(counts_sl.max().item()) > 32:
            sl_starts = torch.zeros(total_slabs_all + 1, dtype=torch.int64, device=w.device)
            sl_starts[1:] = torch.cumsum(counts_sl, 0)
            rank_in_sl = torch.arange(slab_uid.numel(), device=w.device) - sl_starts[slab_uid]
            keep = rank_in_sl < 32
            exc = ~keep
            out_m[widx[exc], sl_i[exc], s_i[exc], lane_i[exc], j_i[exc]] = True
            widx, sl_i, s_i, lane_i, j_i, slab_uid = (
                widx[keep], sl_i[keep], s_i[keep], lane_i[keep], j_i[keep], slab_uid[keep]
            )
            counts_sl = torch.clamp(counts_sl, max=32)
        flips = ((lane_i << 5) | (s_i << 2) | j_i).to(torch.uint16).contiguous()
        meta = torch.zeros((wtot, 4), dtype=torch.int32, device=w.device)
        warp_counts = counts_sl.view(wtot, spw).sum(dim=1).to(torch.int32)
        warp_foff = torch.zeros(wtot + 1, dtype=torch.int32, device=w.device)
        warp_foff[1:] = torch.cumsum(warp_counts, 0, dtype=torch.int32)
        meta[:, 0] = warp_foff[:wtot]
        np_pad = torch.zeros((wtot, 12), dtype=torch.uint8, device=w.device)
        np_pad[:, :spw] = counts_sl.view(wtot, spw).to(torch.uint8)
        meta[:, 1:] = np_pad.view(torch.int32)
        warp_meta = meta.contiguous()
    else:
        flips = torch.zeros(1, dtype=torch.uint16, device=w.device)
        warp_meta = torch.zeros((wtot, 4), dtype=torch.int32, device=w.device)

    bad = out_m.nonzero(as_tuple=False)
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

    _ensure_scratch(
        w.device,
        max(_MIN_PART_ELEMS, 2 * n_tiles * split * 16 * bn),
        max(_MIN_SEM_TILES, 2 * n_tiles),
    )
    _CACHE[key] = (slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn)
    return slabs


_L2PF_MULTI = os.environ.get("INKLING_BF11_L2PF_MULTI", "0").strip().lower() not in ("0", "", "off", "false", "no")
_L2PF_FLIPS_BYTES = int(os.environ.get("INKLING_BF11_L2PF_FLIPS_KB", "128")) * 1024


def will_use_bf11(w: torch.Tensor | None, num_tokens: int | None = None) -> bool:
    """True iff bf11_linear will serve this weight for num_tokens (exact L2PF hygiene guard)."""
    if not ENABLED or w is None or w.dim() != 2:
        return False
    if _CACHE.get((int(w.data_ptr()), int(w.shape[0]), int(w.shape[1]))) is None:
        return False
    if num_tokens is None or num_tokens <= 16:
        return True
    if not C32_ENABLED or num_tokens > C32_MAX_M:
        return False
    _get_lib()
    if not _HAS_C32:
        return False
    if not C32_QUALIFY:
        return True
    ok = _C32_QUAL_OK.get((int(w.shape[0]), int(w.shape[1]), int(num_tokens)))
    return bool(ok) if ok is not None else False


def l2pf_tensor(w: torch.Tensor | None, num_tokens: int | None = None):
    if not ENABLED or w is None or w.dim() != 2:
        return w
    if num_tokens is not None and num_tokens > 16 and not will_use_bf11(w, num_tokens):
        return w
    hit = _CACHE.get((int(w.data_ptr()), int(w.shape[0]), int(w.shape[1])))
    if hit is None:
        return w
    if _L2PF_MULTI:
        return (hit[2], (hit[1], _L2PF_FLIPS_BYTES), hit[4], hit[0])
    return hit[0]


def get_packed(w: torch.Tensor):
    if not ENABLED or w.dim() != 2:
        return None
    return _CACHE.get((int(w.data_ptr()), int(w.shape[0]), int(w.shape[1])))


def _qualify_c32_one(w: torch.Tensor, M: int) -> bool:
    """Fail-closed 0-ULP qualification of bf11_flip_c32_fwd on (N, K, M) outside graph capture."""
    N, K = int(w.shape[0]), int(w.shape[1])
    qkey = (N, K, M)
    cached = _C32_QUAL_OK.get(qkey)
    if cached is not None:
        return cached
    if not C32_QUALIFY:
        _C32_QUAL_OK[qkey] = True
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    lib = _get_lib()
    entry = get_packed(w)
    if lib is None or not _HAS_C32 or entry is None:
        _C32_QUAL_OK[qkey] = False
        return False
    slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn = entry
    ok = True
    g = torch.Generator(device=w.device)
    g.manual_seed(7777 + N + M)
    for sc in (1.0, 0.25):
        x = (torch.randn((M, K), device=w.device, generator=g) * sc).to(torch.bfloat16)
        out = torch.empty((M, N), dtype=torch.bfloat16, device=w.device)
        rc = lib.bf11_flip_c32_fwd(
            x.data_ptr(), slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
            outliers.data_ptr(), o_off.data_ptr(), out.data_ptr(), 0,
            M, N, n_active, K, int(x.stride(0)), ks, split, int(out.stride(0)),
            cfg_id, C32_BM, C32_KORDER, base, 0, 0,
            int(torch.cuda.current_stream().cuda_stream),
        )
        if rc != 0:
            ok = False
            break
        if C32_EXACT_REF == "torch":
            ref = torch.mm(x, w.t())
            if not bool(torch.equal(out.view(torch.int16), ref.view(torch.int16))):
                ok = False
                break
        else:
            # Verify 0-ULP bit-identity across BM=32 vs configured C32_BM (and vs k_order=0 reference)
            ref = torch.empty_like(out)
            rc2 = lib.bf11_flip_c32_fwd(
                x.data_ptr(), slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
                outliers.data_ptr(), o_off.data_ptr(), ref.data_ptr(), 0,
                M, N, n_active, K, int(x.stride(0)), ks, split, int(ref.stride(0)),
                cfg_id, 32, C32_KORDER, base, 0, 0,
                int(torch.cuda.current_stream().cuda_stream),
            )
            if rc2 != 0 or not bool(torch.equal(out.view(torch.int16), ref.view(torch.int16))):
                ok = False
                break
    _C32_QUAL_OK[qkey] = ok
    return ok


def qualify_c32(weights: list[torch.Tensor]) -> dict[tuple[int, int, int], bool]:
    """Pre-qualify canonical C=32 M buckets on representative prepacked model weights."""
    if not ENABLED or not C32_ENABLED or torch.cuda.is_current_stream_capturing():
        return {}
    lib = _get_lib()
    if lib is None or not _HAS_C32:
        return {}
    seen_shapes: dict[tuple[int, int], torch.Tensor] = {}
    for w in weights:
        if w is not None and w.dim() == 2 and get_packed(w) is not None:
            sh = (int(w.shape[0]), int(w.shape[1]))
            if sh not in seen_shapes:
                seen_shapes[sh] = w
    ms = [m for m in _C32_CANONICAL_MS if 17 <= m <= C32_MAX_M]
    n_pass, n_fail = 0, 0
    for sh, w in seen_shapes.items():
        for m in ms:
            if _qualify_c32_one(w, m):
                n_pass += 1
            else:
                n_fail += 1
    if seen_shapes:
        print(
            f"[inkling-bf11-c32] qualified {n_pass}/{n_pass + n_fail} (shape, M) pairs "
            f"(ref={C32_EXACT_REF}, k_order={C32_KORDER}, bm={C32_BM}, max_m={C32_MAX_M})",
            flush=True,
        )
    return dict(_C32_QUAL_OK)


def bf11_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    pdl: bool = False,
    trig: bool = False,
) -> torch.Tensor | None:
    """Run 0-ULP BF11_FLIP_EXACT CUDA GEMM (M<=16 S7 path; opt-in M=17..C32_MAX_M C=32 path)."""
    if not ENABLED or x.dim() != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        return None
    M, K = int(x.shape[0]), int(x.shape[1])
    if M > 16:
        if not C32_ENABLED or M > C32_MAX_M:
            return None
        lib = _get_lib()
        if lib is None or not _HAS_C32:
            return None
        entry = get_packed(w)
        if entry is None:
            return None
        if not _qualify_c32_one(w, M):
            return None
        slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn = entry
        N = int(w.shape[0])
        if out is None:
            out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        nlive_ptr = _rowar_nlive_ptr(M)
        rc = lib.bf11_flip_c32_fwd(
            x.data_ptr(), slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
            outliers.data_ptr(), o_off.data_ptr(), out.data_ptr(), nlive_ptr,
            M, N, n_active, K, int(x.stride(0)), ks, split, int(out.stride(0)),
            cfg_id, C32_BM, C32_KORDER, base, 1 if pdl else 0, 1 if trig else 0,
            int(torch.cuda.current_stream().cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"bf11_flip_c32_fwd failed with cudaError={rc}")
        return out
    lib = _get_lib()
    if lib is None:
        return None
    entry = get_packed(w)
    if entry is None:
        return None
    slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn = entry
    N = int(w.shape[0])
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    if split == 1:
        part = None
        sem = None
    else:
        part, sem = _next_scratch(x.device, n_tiles * split * M * bn, n_tiles)
    nlive_ptr = _rowar_nlive_ptr(M)
    rc = lib.bf11_flip_fwd(
        x.data_ptr(), slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
        outliers.data_ptr(), o_off.data_ptr(),
        0 if part is None else part.data_ptr(),
        0 if sem is None else sem.data_ptr(),
        out.data_ptr(), nlive_ptr,
        M, N, n_active, K, int(x.stride(0)), ks, int(out.stride(0)),
        cfg_id, split, base, 1 if pdl else 0, 1 if trig else 0,
        int(torch.cuda.current_stream().cuda_stream),
    )
    if rc != 0:
        raise RuntimeError(f"bf11_flip_fwd failed with cudaError={rc}")
    return out


def bf11_dual_kv(
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    pdl: bool = False,
    trig: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused 0-ULP global-layer K+V projection in ONE single-kernel launch (M<=16; opt-in M=17..C32_MAX_M)."""
    if (
        not ENABLED or not USE_DUAL_KV
        or x_k.dim() != 2 or x_v.dim() != 2
        or x_k.shape != x_v.shape
        or x_k.dtype != torch.bfloat16 or x_v.dtype != torch.bfloat16
        or x_k.stride(1) != 1 or x_v.stride(1) != 1
    ):
        return None
    M, K = int(x_k.shape[0]), int(x_k.shape[1])
    if M > 16 and (not C32_ENABLED or not C32_DUAL_KV or M > C32_MAX_M):
        return None
    lib = _get_lib()
    if lib is None:
        return None
    ek = get_packed(w_k)
    ev = get_packed(w_v)
    if ek is None or ev is None:
        return None
    if (
        ek[8] != ev[8]
        or ek[9] != ev[9]
        or (M <= 16 and ek[9] != 9)
        or w_k.shape != w_v.shape
        or ek[7] != w_k.shape[0]
        or ev[7] != w_v.shape[0]
    ):
        return None
    sl0, fl0, wm0, ol0, oo0, b0, ks, _, cfg_id, split, n_tiles, bn = ek
    sl1, fl1, wm1, ol1, oo1, b1, _, _, _, _, _, _ = ev
    N_each = int(w_k.shape[0])
    if M > 16:
        if not _HAS_C32 or cfg_id not in (1, 4, 5):
            return None
        if not (_qualify_c32_one(w_k, M) and _qualify_c32_one(w_v, M)):
            return None
        out_k = torch.empty((M, N_each), dtype=x_k.dtype, device=x_k.device)
        out_v = torch.empty((M, N_each), dtype=x_v.dtype, device=x_v.device)
        nlive_ptr = _rowar_nlive_ptr(M)
        rc = lib.bf11_flip_dual_kv_c32_fwd(
            x_k.data_ptr(), x_v.data_ptr(),
            sl0.data_ptr(), sl1.data_ptr(),
            fl0.data_ptr(), fl1.data_ptr(),
            wm0.data_ptr(), wm1.data_ptr(),
            ol0.data_ptr(), ol1.data_ptr(),
            oo0.data_ptr(), oo1.data_ptr(),
            out_k.data_ptr(), out_v.data_ptr(), nlive_ptr,
            M, N_each, K, int(x_k.stride(0)), int(x_v.stride(0)), ks, split,
            int(out_k.stride(0)), int(out_v.stride(0)),
            cfg_id, C32_KORDER, b0, b1, 1 if pdl else 0, 1 if trig else 0,
            int(torch.cuda.current_stream().cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"bf11_flip_dual_kv_c32_fwd failed with cudaError={rc}")
        return out_k, out_v
    out_k = torch.empty((M, N_each), dtype=x_k.dtype, device=x_k.device)
    out_v = torch.empty((M, N_each), dtype=x_v.dtype, device=x_v.device)
    if M == 0:
        return out_k, out_v
    part, sem = _next_scratch(x_k.device, 2 * n_tiles * split * M * bn, 2 * n_tiles)
    nlive_ptr = _rowar_nlive_ptr(M)
    rc = lib.bf11_flip_dual_kv_fwd(
        x_k.data_ptr(), x_v.data_ptr(),
        sl0.data_ptr(), sl1.data_ptr(),
        fl0.data_ptr(), fl1.data_ptr(),
        wm0.data_ptr(), wm1.data_ptr(),
        ol0.data_ptr(), ol1.data_ptr(),
        oo0.data_ptr(), oo1.data_ptr(),
        part.data_ptr(), sem.data_ptr(), out_k.data_ptr(), out_v.data_ptr(), nlive_ptr,
        M, N_each, K, int(x_k.stride(0)), int(x_v.stride(0)), ks,
        int(out_k.stride(0)), int(out_v.stride(0)),
        cfg_id, split, b0, b1, 1 if pdl else 0, 1 if trig else 0,
        int(torch.cuda.current_stream().cuda_stream),
    )
    if rc != 0:
        raise RuntimeError(f"bf11_flip_dual_kv_fwd failed with cudaError={rc}")
    return out_k, out_v
