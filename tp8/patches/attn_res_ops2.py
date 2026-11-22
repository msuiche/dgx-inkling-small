# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block attention residual: softmax-weighted mix of residual stream blocks.

For each token, sources ``x_n`` (stored blocks, then the partial block plus an
optional addend, all in fp32) are scored as ``(q . x_n) * rsqrt(mean(x_n^2) +
eps)`` per pseudo-query ``q``; the output is ``sum_n softmax_n * x_n``, with
``x_n`` RMS-normalized when ``normalize_values`` is set. The fp32 mix then
goes through the following sublayer's RMSNorm in the same kernel.
"""

import ctypes
import os

import torch

from vllm.models.inkling2.ops.norm import rsqrt
from vllm.triton_utils import tl, tldevice, triton


# INKLING_PDL_SKINNY=1: trigger the dependent launch at kernel entry (before gdc_wait) so the next
# PDL kernel (qkvr skinny GEMM on the attention side, the fused route kernel on the MLP
# side) is resident and prefetching its weights while this latency-bound kernel runs.
_EARLY = os.environ.get("INKLING_PDL_SKINNY", "0") == "1"
_CUDA_EXACT = os.environ.get("INKLING_ATTN_RES_CUDA", "0").strip().lower() not in ("0", "", "off", "false", "no")
_CUDA_NLIVE = os.environ.get("INKLING_ATTN_RES_NLIVE", "0").strip().lower() not in ("0", "", "off", "false", "no")
_AUDIT = os.environ.get("INKLING_ATTN_RES_AUDIT", "0").strip().lower() not in ("0", "", "off", "false", "no")
_AUDIT_STATE = {"eager_logs": 0, "g_calls": 0, "g_buf": None, "g_meta": [], "g_dumped": 0}
_AR_LIB_PATH = os.environ.get("INKLING_ATTN_RES_LIB", "/rdma/libattn_res_exact_v3.so")
_AR_LIB = None
_AR_LIB_TRIED = False
_PREFILL_CUDA = os.environ.get("INKLING_ATTN_RES_PREFILL_CUDA", "1").strip().lower() not in ("0", "", "off", "false", "no")
_AR_PREFILL_LIB_PATH = os.environ.get("INKLING_PREFILL_AR_LIB", "/rdma/libfused_sconv_ar_prefill.so")
_AR_PREFILL_LIB = None
_AR_PREFILL_LIB_TRIED = False


def _get_ar_prefill_lib():
    global _AR_PREFILL_LIB, _AR_PREFILL_LIB_TRIED
    if not _AR_PREFILL_LIB_TRIED:
        _AR_PREFILL_LIB_TRIED = True
        if os.path.exists(_AR_PREFILL_LIB_PATH):
            try:
                lib = ctypes.CDLL(_AR_PREFILL_LIB_PATH)
                lib.attn_res_prefill_v2_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.attn_res_prefill_v2_fwd.restype = ctypes.c_int
                _AR_PREFILL_LIB = lib
                print(f"[inkling-attn-res] loaded 0-ULP prefill CUDA kernel: {_AR_PREFILL_LIB_PATH}", flush=True)
            except Exception as e:
                print(f"[inkling-attn-res] prefill load failed ({e!r}); using Triton", flush=True)
                _AR_PREFILL_LIB = None
    return _AR_PREFILL_LIB


def _get_ar_lib():
    global _AR_LIB, _AR_LIB_TRIED
    if not _AR_LIB_TRIED:
        _AR_LIB_TRIED = True
        if os.path.exists(_AR_LIB_PATH):
            try:
                lib = ctypes.CDLL(_AR_LIB_PATH)
                lib.attn_res_exact_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.attn_res_exact_fwd.restype = ctypes.c_int
                _AR_LIB = lib
                print(f"[inkling-attn-res] loaded 0-ULP CUDA exact kernel: {_AR_LIB_PATH}", flush=True)
            except Exception as e:
                print(f"[inkling-attn-res] load failed ({e!r}); using Triton", flush=True)
                _AR_LIB = None
    return _AR_LIB


@triton.jit
def _load_source(
    Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
):
    if n < N_STACK:
        x = tl.load(Stack + n * stride_s + row_off + offs, mask, other=0)
        x = x.to(tl.float32)
    else:
        x = tl.load(Last + row_off + offs, mask, other=0).to(tl.float32)
        if HAS_ADDEND:
            x = x + tl.load(Addend + row_off + offs, mask, other=0).to(tl.float32)
    return x


@triton.jit
def _mix(
    Stack,
    stride_s,
    Last,
    Addend,
    row_off,
    offs,
    mask,
    probs,
    n_idx,
    N_STACK,
    HAS_ADDEND,
    BLOCK_D,
):
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        x = _load_source(
            Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
        )
        acc = tl.fma(tl.sum(tl.where(n_idx == n, probs, 0.0)), x, acc)
    return acc


@triton.jit
def _attn_res_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EARLY: tl.constexpr = False,
):
    if EARLY:
        tl.extra.cuda.gdc_launch_dependents()  # before the wait; dependents gdc_wait for our end
    tl.extra.cuda.gdc_wait()
    # One program per (token, query).
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    n_idx = tl.arange(0, BLOCK_N)
    d_idx = tl.arange(0, BLOCK_D)

    # Pass 1: per-source inverse RMS and query dot products.
    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        sum_sq = tl.zeros((BLOCK_D,), tl.float32)
        dots = tl.zeros((BLOCK_D,), tl.float32)
        for d0 in tl.range(0, D, BLOCK_D):
            offs = d0 + d_idx
            mask = offs < D
            x = _load_source(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                mask,
                n,
                N_STACK,
                HAS_ADDEND,
            )
            q = tl.load(Queries + q_off + offs, mask, other=0)
            sum_sq = tl.fma(x, x, sum_sq)
            dots = tl.fma(q, x, dots)
        rstd = rsqrt(tl.fma(tl.sum(sum_sq), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        logits = tl.where(n_idx == n, tl.sum(dots) * rstd, logits)

    probs = tldevice.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    # Pass 2: RMS of the probability-weighted sum; pass 3: normalized sum.
    sum_sq = 0.0
    for d0 in tl.range(0, D, BLOCK_D):
        offs = d0 + d_idx
        acc = (
            _mix(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                offs < D,
                probs,
                n_idx,
                N_STACK,
                HAS_ADDEND,
                BLOCK_D,
            )
            * inv_sum
        )
        sum_sq += tl.sum(acc * acc)
    out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    for d0 in tl.range(0, D, BLOCK_D):
        offs = d0 + d_idx
        mask = offs < D
        acc = (
            _mix(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                mask,
                probs,
                n_idx,
                N_STACK,
                HAS_ADDEND,
                BLOCK_D,
            )
            * inv_sum
        )
        w = tl.load(NormW + q_off + offs, mask, other=0).to(tl.float32)
        acc = acc * out_rstd * w
        tl.store(out + offs, acc.to(Out.dtype.element_ty), mask)


@triton.jit
def _attn_res_row_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    EARLY: tl.constexpr = False,
):
    if EARLY:
        tl.extra.cuda.gdc_launch_dependents()  # before the wait; dependents gdc_wait for our end
    tl.extra.cuda.gdc_wait()
    # Small batches: one program per (token, query) holds the row in registers
    # as a power-of-2 head [0, BLOCK_A) and a masked tail [BLOCK_A, D), so the
    # sources are read twice instead of three times.
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    da = tl.arange(0, BLOCK_A)
    db = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = da < D
    mask_b = db < D
    n_idx = tl.arange(0, BLOCK_N)
    qa = tl.load(Queries + q_off + da, mask_a, other=0)
    qb = tl.load(Queries + q_off + db, mask_b, other=0)
    src = (Stack, stride_s, Last, Addend, row_off)

    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        rstd = rsqrt(tl.fma(tl.sum(xa * xa) + tl.sum(xb * xb), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        dot = tl.sum(qa * xa) + tl.sum(qb * xb)
        logits = tl.where(n_idx == n, dot * rstd, logits)

    probs = tldevice.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    acc_a = tl.zeros((BLOCK_A,), tl.float32)
    acc_b = tl.zeros((BLOCK_B,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        p = tl.sum(tl.where(n_idx == n, probs, 0.0))
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        acc_a = tl.fma(p, xa, acc_a)
        acc_b = tl.fma(p, xb, acc_b)
    acc_a = acc_a * inv_sum
    acc_b = acc_b * inv_sum
    sum_sq = tl.sum(acc_a * acc_a) + tl.sum(acc_b * acc_b)
    out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    wa = tl.load(NormW + q_off + da, mask_a, other=0).to(tl.float32)
    wb = tl.load(NormW + q_off + db, mask_b, other=0).to(tl.float32)
    tl.store(out + da, (acc_a * out_rstd * wa).to(Out.dtype.element_ty), mask_a)
    tl.store(out + db, (acc_b * out_rstd * wb).to(Out.dtype.element_ty), mask_b)


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mix ``[*stack, last (+ addend)]`` with each pseudo-query.

    ``queries`` are fp32 ``(Q, D)``, already scaled by ``D**-0.5``; each fp32
    mix is RMS-normalized with its row of ``norm_weight`` ``(Q, D)``. Returns
    ``(Q, T, D)`` in ``last.dtype``.
    """
    global _CUDA_EXACT
    num_tokens, d = last.shape
    num_stack = stack.shape[0]
    num_q = queries.shape[0]
    assert last.is_contiguous() and queries.is_contiguous()
    assert queries.dtype == torch.float32
    assert stack.stride(1) == d and stack.stride(2) == 1
    if addend is not None:
        assert addend.is_contiguous() and addend.shape == last.shape
    assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    out = torch.empty((num_q, num_tokens, d), dtype=last.dtype, device=last.device)
    if (
        _CUDA_EXACT
        and d == 4608
        and 1 <= num_stack <= 8
        and normalize_values
        and num_tokens * num_q <= 256
        and last.dtype == torch.bfloat16
        and stack.dtype == torch.bfloat16
        and norm_weight.dtype == torch.bfloat16
        and (addend is None or addend.dtype == torch.bfloat16)
    ):
        lib = _get_ar_lib()
        if lib is not None:
            nlive_ptr = 0
            if _CUDA_NLIVE and not _AUDIT:
                try:
                    from vllm.distributed.device_communicators import rdma_all_reduce as _rar
                    if _rar.ROWAR.get("scope") == num_tokens and _rar.ROWAR.get("buf") is not None:
                        nlive_ptr = int(_rar.ROWAR["buf"].data_ptr())
                except Exception:
                    nlive_ptr = 0
            rc = lib.attn_res_exact_fwd(
                stack.data_ptr(),
                last.data_ptr(),
                0 if addend is None else addend.data_ptr(),
                queries.data_ptr(),
                norm_weight.data_ptr(),
                out.data_ptr(),
                nlive_ptr,
                int(stack.stride(0)),
                num_q,
                num_tokens,
                num_stack,
                float(eps),
                float(norm_eps),
                1 if addend is not None else 0,
                1 if normalize_values else 0,
                1 if _EARLY else 0,
                1 if nlive_ptr != 0 else 0,
                1,
                int(torch.cuda.current_stream().cuda_stream),
            )
            if rc != 0:
                raise RuntimeError(f"attn_res_exact_fwd failed with cudaError={rc}")
            if not _AUDIT:
                return out
            # Audit mode: also run Triton reference and compare!
            ref = torch.empty_like(out)
            _args = (stack, last, addend, queries, norm_weight, ref, stack.stride(0), num_tokens, eps, norm_eps)
            _meta = dict(D=d, N_STACK=num_stack, HAS_ADDEND=addend is not None, NORMALIZE=normalize_values,
                         BLOCK_N=triton.next_power_of_2(num_stack + 1), enable_fp_fusion=False,
                         launch_pdl=True, EARLY=_EARLY)
            _ba = triton.next_power_of_2(d + 1) // 2
            _bb = max(triton.next_power_of_2(d - _ba), 16)
            _attn_res_row_kernel[(num_q, num_tokens)](*_args, **_meta, BLOCK_A=_ba, BLOCK_B=_bb, num_warps=8)
            if torch.cuda.is_current_stream_capturing():
                if _AUDIT_STATE["g_buf"] is None:
                    _AUDIT_STATE["g_buf"] = torch.zeros(8192, dtype=torch.int32, device=out.device)
                idx = _AUDIT_STATE["g_calls"]
                if idx < 8192:
                    _AUDIT_STATE["g_calls"] = idx + 1
                    _AUDIT_STATE["g_meta"].append((idx, num_q, num_tokens, num_stack, addend is not None, normalize_values, int(stack.stride(0)), eps, norm_eps))
                    _AUDIT_STATE["g_buf"][idx].add_((out.view(torch.uint16) != ref.view(torch.uint16)).any().to(torch.int32))
            else:
                if _AUDIT_STATE["g_buf"] is not None and _AUDIT_STATE["g_dumped"] < 4:
                    gb = _AUDIT_STATE["g_buf"][:_AUDIT_STATE["g_calls"]].cpu()
                    nz = torch.nonzero(gb, as_tuple=False).flatten().tolist()
                    if nz or _AUDIT_STATE["g_dumped"] == 0:
                        _AUDIT_STATE["g_dumped"] += 1
                        print(f"[attn-res-audit] GRAPH CHECK #{_AUDIT_STATE['g_dumped']} g_calls={_AUDIT_STATE['g_calls']} total_nz={len(nz)} first10={[(i, int(gb[i]), _AUDIT_STATE['g_meta'][i]) for i in nz[:10]]}", flush=True)
                nd = int((out.view(torch.uint16) != ref.view(torch.uint16)).sum().item())
                if nd > 0 and _AUDIT_STATE["eager_logs"] < 20:
                    _AUDIT_STATE["eager_logs"] += 1
                    mx = float((out.float() - ref.float()).abs().max().item())
                    diff_idx = torch.nonzero(out.view(torch.uint16) != ref.view(torch.uint16), as_tuple=False)[0].tolist()
                    print(f"[attn-res-audit] EAGER DIFF #{_AUDIT_STATE['eager_logs']}: nq={num_q} T={num_tokens} ns={num_stack} "
                          f"add={addend is not None} norm={normalize_values} ss={int(stack.stride(0))} eps={eps} neps={norm_eps} "
                          f"nd={nd} max_abs={mx:.3e} first={diff_idx} got={out[tuple(diff_idx)].item():.6g} ref={ref[tuple(diff_idx)].item():.6g}", flush=True)
                    if _AUDIT_STATE["eager_logs"] == 1 and os.environ.get("INKLING_ATTN_RES_SAVE_DIFF", "0") == "1":
                        t0 = diff_idx[1]
                        try:
                            torch.save({
                                "stack": stack[:, t0:t0+1].detach().cpu(),
                                "last": last[t0:t0+1].detach().cpu(),
                                "addend": None if addend is None else addend[t0:t0+1].detach().cpu(),
                                "queries": queries.detach().cpu(),
                                "norm_weight": norm_weight.detach().cpu(),
                                "eps": float(eps), "norm_eps": float(norm_eps),
                                "normalize_values": bool(normalize_values),
                                "diff_idx": diff_idx,
                                "got_row": out[:, t0:t0+1].detach().cpu(),
                                "ref_row": ref[:, t0:t0+1].detach().cpu(),
                            }, "/prof/ar_diff1.pt")
                            print("[attn-res-audit] SAVED /prof/ar_diff1.pt", flush=True)
                        except Exception as _se:
                            print(f"[attn-res-audit] save failed: {_se!r}", flush=True)
                        # Disable CUDA exact after saving 1st diff so the rest of the profiling run matches S5-W5C!
                        _CUDA_EXACT = False
            return ref if os.environ.get("INKLING_ATTN_RES_SAVE_DIFF", "0") == "1" else out
    args = (
        stack,
        last,
        addend,
        queries,
        norm_weight,
        out,
        stack.stride(0),
        num_tokens,
        eps,
        norm_eps,
    )
    meta = dict(
        D=d,
        N_STACK=num_stack,
        HAS_ADDEND=addend is not None,
        NORMALIZE=normalize_values,
        BLOCK_N=triton.next_power_of_2(num_stack + 1),
        enable_fp_fusion=False,
        launch_pdl=(num_tokens <= 16),
        EARLY=(_EARLY and num_tokens <= 16),
    )
    # The queries of a token run back to back and share its sources in L2.
    grid = (num_q, num_tokens)
    if num_tokens * num_q <= 256:
        block_a = triton.next_power_of_2(d + 1) // 2
        block_b = max(triton.next_power_of_2(d - block_a), 16)
        _attn_res_row_kernel[grid](
            *args, **meta, BLOCK_A=block_a, BLOCK_B=block_b, num_warps=8
        )
    else:
        if (
            _PREFILL_CUDA
            and d == 4608
            and 1 <= num_stack <= 8
            and num_q in (1, 3)
            and normalize_values
            and last.dtype == torch.bfloat16
            and stack.dtype == torch.bfloat16
            and norm_weight.dtype == torch.bfloat16
            and (addend is None or addend.dtype == torch.bfloat16)
        ):
            plib = _get_ar_prefill_lib()
            if plib is not None:
                rc = plib.attn_res_prefill_v2_fwd(
                    stack.data_ptr(),
                    last.data_ptr(),
                    0 if addend is None else addend.data_ptr(),
                    queries.data_ptr(),
                    norm_weight.data_ptr(),
                    out.data_ptr(),
                    int(stack.stride(0)),
                    num_q,
                    num_tokens,
                    num_stack,
                    float(eps),
                    float(norm_eps),
                    1 if addend is not None else 0,
                    1,
                    int(torch.cuda.current_stream().cuda_stream),
                )
                if rc == 0:
                    return out
        _attn_res_kernel[grid](*args, **meta, BLOCK_D=1024, num_warps=4)
    return out
