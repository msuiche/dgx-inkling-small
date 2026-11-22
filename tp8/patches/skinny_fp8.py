# SPDX-License-Identifier: Apache-2.0
"""SM12x (GB10) W8A16 skinny GEMM for the DSpark drafter: out[M, N] = x[M, K] @ dequant(w8[N, K]).T.

``w8`` is float8_e4m3fn with one fp32 scale per output row (``scale[n] = amax(row) / 448``).
The weight streams at ~half the bf16 bytes; each tile is converted to bf16 (exact) for the
tensor-core dot, the fp32 accumulator is multiplied by the row scale before the split-K partial
store, and the partials are summed in a fixed order (deterministic, same as skinny.py).
Used only behind INKLING_DRAFT_FP8 / INKLING_DRAFT_HEAD_FP8 (see dspark_model.py).
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _w8_kernel(X, W, S, P, M, N, K, sx, sw, KS,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EVEN_K: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    nm = n < N
    mm = m < M
    acc = tl.zeros([BM, BN], tl.float32)
    k_lo = ps * KS
    for k0 in range(k_lo, tl.minimum(k_lo + KS, K), BK):
        kk = k0 + tl.arange(0, BK)
        if EVEN_K:
            x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None], other=0.0)
            w = tl.load(W + n[:, None].to(tl.int64) * sw + kk[None, :], mask=nm[:, None],
                        other=0.0, eviction_policy="evict_first")
        else:
            km = kk < K
            x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None] & km[None, :], other=0.0)
            w = tl.load(W + n[:, None].to(tl.int64) * sw + kk[None, :], mask=nm[:, None] & km[None, :],
                        other=0.0, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc)
    acc = acc * tl.load(S + n, mask=nm, other=0.0)[None, :]
    tl.store(P + ps * M * N + m[:, None] * N + n[None, :], acc, mask=mm[:, None] & nm[None, :])


@triton.jit
def _reduce_kernel(P, O, MN, so_m, N, S: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = i < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(S):
        acc += tl.load(P + s * MN + i, mask=msk, other=0.0)
    tl.store(O + (i // N) * so_m + i % N, acc.to(O.dtype.element_ty), mask=msk)


# (bm, bn, bk, split, warps, stages) per (N, K) at M <= 16; from bench_fp8.py cold-L2 sweeps on GB10.
_CFG16 = {
    (4608, 4608): (16, 32, 512, 3, 4, 3),   # drafter gate_up
    (4608, 2304): (16, 16, 256, 6, 4, 3),   # drafter down
    (1024, 4608): (16, 64, 256, 2, 8, 4),   # drafter qkv
    (4608, 768): (16, 16, 256, 2, 4, 3),    # drafter o_proj
    (25128, 4608): (16, 16, 512, 1, 4, 4),  # lm_head shard
}
_OV = None  # (bm, bn, bk, split, warps, stages) override for tuning
_NUM_SMS = None


def _config(M, N, K):
    global _NUM_SMS
    if M <= 16:
        c = _CFG16.get((N, K))
        if c is not None:
            return c
        bm, bn, bk, nw, ns = 16, 32, 256, 4, 3
    else:  # 17..64 rows: smaller K tile keeps x tiles inside the 99 KB shared memory
        bm, bn, bk, nw, ns = (32 if M <= 32 else 64), 32, 128, 4, 3
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    n_tiles = triton.cdiv(N, bn)
    split = max(1, min(triton.cdiv(4 * _NUM_SMS, n_tiles), K // (2 * bk)))
    return bm, bn, bk, split, nw, ns


def supported(x: torch.Tensor, w8: torch.Tensor) -> bool:
    return (x.dim() == 2 and 0 < x.shape[0] <= 64 and x.dtype == torch.bfloat16
            and w8.dtype == torch.float8_e4m3fn and x.stride(1) == 1 and w8.stride(1) == 1
            and x.shape[1] == w8.shape[1] and x.shape[1] >= 256)


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 ``w[N, K]`` -> (float8_e4m3fn ``[N, K]``, fp32 scale ``[N]`` = amax / 448)."""
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
    return (wf / scale[:, None]).to(torch.float8_e4m3fn).contiguous(), scale


def dequantize_rows(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w8.float() * scale[:, None]


def skinny_fp8_linear(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor,
                      out: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ dequant(w8).T``: x bf16 ``[M, K]`` (M <= 64), w8 e4m3 ``[N, K]``, scale fp32 ``[N]``."""
    M, K = x.shape
    N = w8.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    bm, bn, bk, split, nw, ns = _OV if _OV is not None else _config(M, N, K)
    ks = triton.cdiv(triton.cdiv(K, split), bk) * bk
    split = triton.cdiv(K, ks)
    part = torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    _w8_kernel[(triton.cdiv(N, bn), split)](
        x, w8, scale, part, M, N, K, x.stride(0), w8.stride(0), ks,
        BM=bm, BN=bn, BK=bk, EVEN_K=(K % bk == 0), num_warps=nw, num_stages=ns)
    mn = M * N
    _reduce_kernel[(triton.cdiv(mn, 1024),)](part, out, mn, out.stride(0), N, S=split,
                                            BLOCK=1024, num_warps=4)
    return out
