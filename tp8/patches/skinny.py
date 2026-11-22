# SPDX-License-Identifier: Apache-2.0
"""SM12x (GB10) bf16 skinny GEMM: out[M, N] = x[M, K] @ w[N, K].T for M <= 32.

cuBLAS picks a 1-warp SM80 wmma kernel for these decode shapes (~190 GB/s);
this streams the weight once with wide CTAs and split-K. Partials are summed
in a fixed order (deterministic). fp32 accumulation, bf16 (or out's dtype)
result.
"""
import os

import torch

from vllm.triton_utils import tl, triton

# Programmatic dependent launch (all default OFF; the coordinator A/Bs them live):
#   INKLING_PDL_SKINNY=1 (qkvr, behind attn_res_row) and INKLING_WO_L2PF=1 (wo, behind
#                          the attention chain; needs INKLING_ATTN_FUSE_GS=1): the GEMM
#                          launches with PDL and L2-prefetches its own W tile before
#                          gdc_wait (call-site gates in linear.py)
#   INKLING_PDL_PF_COLS=N  K columns per CTA segment to prefetch (default 512)
#   INKLING_SKINNY_DIRECT=1  split==1 GEMMs (wo) store bf16 straight from the GEMM
#                          kernel; no _reduce_kernel launch (bit-exact, see _skinny_kernel)
_PF_COLS = int(os.environ.get("INKLING_PDL_PF_COLS", "512"))
_DIRECT = os.environ.get("INKLING_SKINNY_DIRECT", "0") == "1"


@triton.jit
def _skinny_kernel(X, W, P, M, N, K, sx, sw, KS, O, so_m,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   PDL: tl.constexpr = False, PF: tl.constexpr = 0,
                   DIRECT: tl.constexpr = False, TRIG: tl.constexpr = False):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    nm = n < N
    mm = m < M
    acc = tl.zeros([BM, BN], tl.float32)
    k_lo = ps * KS
    if PDL:
        # Launched early behind a latency-bound predecessor: pull our own first PF
        # columns of W (weights, never written by the predecessor) into L2, then wait.
        # One 128-B line per 64 bf16; the prefetch instruction does not block.
        k_hi = tl.minimum(k_lo + KS, K)
        for k0 in range(k_lo, tl.minimum(k_lo + PF, k_hi), BK):
            kp = k0 + tl.arange(0, BK // 64) * 64
            pm = nm[:, None] & (kp[None, :] < k_hi)
            tl.inline_asm_elementwise(
                "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p prefetch.global.L2 [$1]; mov.b32 $0, 0; }",
                "=r,l,r", [W + n[:, None].to(tl.int64) * sw + kp[None, :], pm.to(tl.int32)],
                dtype=tl.int32, is_pure=False, pack=1)
        tl.extra.cuda.gdc_wait()
    for k0 in range(k_lo, tl.minimum(k_lo + KS, K), BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < K
        x = tl.load(X + m[:, None] * sx + kk[None, :], mask=mm[:, None] & km[None, :], other=0.0)
        w = tl.load(W + n[:, None].to(tl.int64) * sw + kk[None, :], mask=nm[:, None] & km[None, :],
                    other=0.0, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
    if TRIG:
        # the (tiny) reduce kernel may start scheduling while the last CTAs finish;
        # it gdc_waits for our completion before reading the partials
        tl.extra.cuda.gdc_launch_dependents()
    if DIRECT:
        # split == 1: _reduce_kernel computes (0.0 + p) -> out dtype; "+ 0.0" keeps the
        # -0.0 -> +0.0 behaviour so the result is bit-identical.
        tl.store(O + m[:, None] * so_m + n[None, :], (acc + 0.0).to(O.dtype.element_ty),
                 mask=mm[:, None] & nm[None, :])
    else:
        tl.store(P + ps * M * N + m[:, None] * N + n[None, :], acc, mask=mm[:, None] & nm[None, :])


@triton.jit
def _reduce_kernel(P, O, MN, so_m, N, S: tl.constexpr, BLOCK: tl.constexpr,
                   PDL: tl.constexpr = False, TRIG: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        if TRIG:
            tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = i < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(S):
        acc += tl.load(P + s * MN + i, mask=msk, other=0.0)
    tl.store(O + (i // N) * so_m + i % N, acc.to(O.dtype.element_ty), mask=msk)


_NUM_SMS = None
_CFG = {}
_OV = None  # (bm, bn, bk, split, warps, stages) override for tuning


def _config(M, N, K):
    global _NUM_SMS
    key = (M, N, K)
    c = _CFG.get(key)
    if c is None:
        if _NUM_SMS is None:
            _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
        bm = 16 if M <= 16 else 32
        bn, bk = 64, 256  # bench_skinny cold-cache sweep (GB10)
        n_tiles = triton.cdiv(N, bn)
        # aim for ~4 CTAs per SM, split-K in BK multiples
        split = max(1, min(triton.cdiv(4 * _NUM_SMS, n_tiles), K // (2 * bk)))
        ks = triton.cdiv(triton.cdiv(K, split), bk) * bk
        split = triton.cdiv(K, ks)
        c = _CFG[key] = (bm, bn, bk, split, ks)
    return c


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (x.dim() == 2 and x.shape[0] <= 32 and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16 and x.stride(1) == 1
            and weight.stride(1) == 1 and x.shape[1] == weight.shape[1]
            and x.shape[1] >= 256)


def skinny_linear(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None,
                  pdl: bool = False, pf_cols: int | None = None, trig_reduce: bool = False):
    """pdl: this GEMM follows a latency-bound kernel that calls gdc_launch_dependents
    early (attn_res_row, the attention chain): launch with PDL + own-slice L2 prefetch.
    Only enable per call site; behind a kernel that never triggers early it is a
    ~1.5 us loss.  trig_reduce: let the reduce kernel's dependents (the next PDL
    kernel) start early."""
    M, K = x.shape
    N = weight.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    if _OV is not None:
        bm, bn, bk, split, nw, ns = _OV
        ks = triton.cdiv(triton.cdiv(K, split), bk) * bk
        split = triton.cdiv(K, ks)
    else:
        bm, bn, bk, split, ks = _config(M, N, K)
        nw, ns = 4, 3
    pf = (_PF_COLS if pf_cols is None else pf_cols) if pdl else 0
    direct = _DIRECT and split == 1
    if direct:
        part = out
    else:
        part = torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    _skinny_kernel[(triton.cdiv(N, bn), split)](
        x, weight, part, M, N, K, x.stride(0), weight.stride(0), ks, out, out.stride(0),
        BM=bm, BN=bn, BK=bk, PDL=pdl, PF=pf, DIRECT=direct, TRIG=pdl and (not direct or trig_reduce),
        num_warps=nw, num_stages=ns, launch_pdl=pdl)
    if direct:
        return out
    mn = M * N
    _reduce_kernel[(triton.cdiv(mn, 1024),)](part, out, mn, out.stride(0), N, S=split,
                                            BLOCK=1024, PDL=pdl, TRIG=pdl and trig_reduce,
                                            num_warps=4, launch_pdl=pdl)
    return out
