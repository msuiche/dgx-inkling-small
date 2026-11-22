# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 linears: small batches on FlashInfer's CuTe DSL low-M kernels, optional
prefill-only dynamic FP8 dense path (INKLING_PF_DENSE_FP8=1, M >= 1024) on SM121,
and the reference ``linear`` with fp32 accumulation in ordered 1024-wide K groups."""

import os
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import is_flashinfer_autotuning

# Up to this many rows, FlashInfer's CuTe DSL low-M kernels run near HBM
# bandwidth (cuBLAS is up to 2x slower); larger batches use cuBLAS.
LOW_M_MAX_TOKENS = 32
# sm121: FlashInfer's CuTe DSL mm_bf16 has no SM12x build.
_SM12 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
# INKLING_SKINNY=1: decode-size bf16 GEMMs via a split-K Triton kernel that
# streams the weight at ~230-249 GB/s (cuBLAS picks 1-warp SM80 wmma, 207-230).
_SKINNY = _SM12 and os.environ.get("INKLING_SKINNY", "0") == "1"


# INKLING_PDL_SKINNY=1: qkvr (out is None, K=4608, N in _PDL_QKVR_N, behind attn_res_row)
# INKLING_WO_L2PF=1:    wo_ud (out given, weight [_PDL_WO_N, _PDL_WO_K], behind the
#                       sm12 attention + merge/gate_sink chain)
# launch their skinny GEMM with PDL + own-slice L2 prefetch (skinny.py).  Gated by
# shape so every other skinny call (drafter, k/v projections, lm_head) keeps the plain
# launch.  Both default OFF; the predecessors' early gdc_launch_dependents are behind
# the same flags (attn_res.py, sm12_attn.py).
_PDL_QKVR = _SM12 and os.environ.get("INKLING_PDL_SKINNY", "0") == "1"
_PDL_WO = (_SM12 and os.environ.get("INKLING_WO_L2PF", "0") == "1"
           and os.environ.get("INKLING_ATTN_FUSE_GS", "0") == "1")
_PDL = _PDL_QKVR or _PDL_WO
_PDL_QKVR_N = tuple(int(v) for v in os.environ.get("INKLING_PDL_QKVR_N", "1896,1640,128").split(","))
_PDL_WO_N, _PDL_WO_K = (int(v) for v in os.environ.get("INKLING_PDL_WO_NK", "4608,768").split(","))

# ---------------------------------------------------------------------------
# Prefill-only Dynamic FP8 Dense Linear (INKLING_PF_DENSE_FP8=1, M >= 1024)
# ---------------------------------------------------------------------------
# Strictly gated to large prefill chunks outside CUDA graph capture so decode
# (M <= 16) and exact_ink.py (T <= 120) remain 100% untouched & 0-ULP bit-exact!
_PF_DENSE_FP8 = _SM12 and os.environ.get("INKLING_PF_DENSE_FP8", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_PF_DENSE_FP8_MIN_M = int(os.environ.get("INKLING_PF_DENSE_FP8_MIN_M", "1024"))
# Break-even N on GB10 (~660): quantizing x[M, K] costs ~3*M*K/BW, so N < 512
# (e.g. global k_proj/v_proj N=128, K=4608) stays faster in BF16 cuBLAS!
_PF_DENSE_FP8_MIN_N = int(os.environ.get("INKLING_PF_DENSE_FP8_MIN_N", "512"))
_PF_DENSE_FP8_MODE = os.environ.get("INKLING_PF_DENSE_FP8_MODE", "rowwise").strip().lower()
# Prefill-only FP8 Router GEMM (INKLING_PF_ROUTER_FP8=1, M >= 1024):
# Feeds pre-quantized (x_fp8, x_scale) directly into torch._scaled_mm inside
# k_group_partials (or dynamically quantizes BF16 x), bypassing BF16 cuBLAS
# router GEMM (~5.0s -> ~1.5s @ 1M). Strictly gated off for decode / CUDA graphs.
_PF_ROUTER_FP8 = _SM12 and os.environ.get("INKLING_PF_ROUTER_FP8", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_PF_ROUTER_FP8_MIN_M = int(os.environ.get("INKLING_PF_ROUTER_FP8_MIN_M", "1024"))
_FP8_MAX = 448.0
_FP8_EPS = 1e-12

# Keyed by (weight.data_ptr(), N, K, stride0, stride1, mode) so ephemeral slices
# like global attention q_side_w / w_k / w_v cache cleanly without re-quantizing!
_FP8_W_CACHE: dict[tuple[int, int, int, int, int, str], tuple[torch.Tensor, torch.Tensor, int]] = {}
# Resolved backend mode after one-time SM121a probe: ("rowwise" | "tensorwise" | "off", supports_out_kwarg)
_FP8_CAP: tuple[str, bool] | None = None


@triton.jit
def _quant_fp8_rowwise_ht_kernel(
    X,
    Y,
    S,
    M,
    stride_xm,
    stride_ym,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Single-pass fused BF16 -> FP8 E4M3FN per-row quantizer for K = HEAD + TAIL
    where HEAD and TAIL are powers of 2 (covers K=4608=4096+512, 768=512+256,
    6144=4096+2048, and all power-of-2 K with zero DRAM re-reads!)."""
    row = tl.program_id(0)
    if row >= M:
        # Zero-pad rows M..M_pad-1 for cuBLASLt 16-row alignment.
        oh = tl.arange(0, HEAD)
        z = tl.zeros((HEAD,), dtype=tl.float8e4nv)
        tl.store(Y + row.to(tl.int64) * stride_ym + oh, z)
        if TAIL > 0:
            ot = tl.arange(0, TAIL)
            zt = tl.zeros((TAIL,), dtype=tl.float8e4nv)
            tl.store(Y + row.to(tl.int64) * stride_ym + HEAD + ot, zt)
        tl.store(S + row, 1.0)
        return

    x_row = X + row.to(tl.int64) * stride_xm
    y_row = Y + row.to(tl.int64) * stride_ym
    oh = tl.arange(0, HEAD)
    xh = tl.load(x_row + oh).to(tl.float32)
    amax = tl.max(tl.abs(xh), axis=0)
    if TAIL > 0:
        ot = tl.arange(0, TAIL)
        xt = tl.load(x_row + HEAD + ot).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(xt), axis=0))
    scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    inv_scale = 1.0 / scale
    tl.store(S + row, scale)
    yh = tl.clamp(xh * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(y_row + oh, yh)
    if TAIL > 0:
        yt = tl.clamp(xt * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_row + HEAD + ot, yt)


@triton.jit
def _quant_fp8_rowwise_generic_kernel(
    X,
    Y,
    S,
    M,
    K,
    stride_xm,
    stride_ym,
    BLOCK_K: tl.constexpr,
):
    """Generic per-row BF16 -> FP8 E4M3FN quantizer for arbitrary K % 16 == 0."""
    row = tl.program_id(0)
    y_row = Y + row.to(tl.int64) * stride_ym
    if row >= M:
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            tl.store(y_row + offs, tl.zeros((BLOCK_K,), dtype=tl.float8e4nv), mask=offs < K)
        tl.store(S + row, 1.0)
        return

    x_row = X + row.to(tl.int64) * stride_xm
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(xv), axis=0))
    scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    inv_scale = 1.0 / scale
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        yv = tl.clamp(xv * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_row + offs, yv, mask=offs < K)


def quantize_fp8_rowwise(x: torch.Tensor, pad_m16: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-row BF16 [M, K] -> (x_fp8 [M_pad, K] e4m3fn, x_scale [M_pad, 1] fp32)."""
    m, k = x.shape
    m_pad = ((m + 15) & ~15) if pad_m16 else m
    x_fp8 = torch.empty((m_pad, k), dtype=torch.float8_e4m3fn, device=x.device)
    x_scale = torch.empty((m_pad, 1), dtype=torch.float32, device=x.device)
    head = 1 << (k.bit_length() - 1)
    tail = k - head
    if head <= 8192 and (tail == 0 or (tail & (tail - 1)) == 0):
        warps = 8 if k >= 2048 else (4 if k >= 512 else 2)
        _quant_fp8_rowwise_ht_kernel[(m_pad,)](
            x, x_fp8, x_scale, m, x.stride(0), x_fp8.stride(0),
            HEAD=head, TAIL=tail, num_warps=warps,
        )
    else:
        _quant_fp8_rowwise_generic_kernel[(m_pad,)](
            x, x_fp8, x_scale, m, k, x.stride(0), x_fp8.stride(0),
            BLOCK_K=1024, num_warps=4,
        )
    return x_fp8, x_scale


@triton.jit
def _dequant_fp8_rowwise_ht_kernel(
    X,
    S,
    Y,
    stride_xm,
    stride_ym,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Single-pass fused FP8 E4M3FN + per-row fp32 scale -> BF16 dequantizer for K = HEAD + TAIL."""
    row = tl.program_id(0)
    scale = tl.load(S + row).to(tl.float32)
    x_row = X + row.to(tl.int64) * stride_xm
    y_row = Y + row.to(tl.int64) * stride_ym
    oh = tl.arange(0, HEAD)
    xh = tl.load(x_row + oh).to(tl.float32)
    tl.store(y_row + oh, (xh * scale).to(tl.bfloat16))
    if TAIL > 0:
        ot = tl.arange(0, TAIL)
        xt = tl.load(x_row + HEAD + ot).to(tl.float32)
        tl.store(y_row + HEAD + ot, (xt * scale).to(tl.bfloat16))


def dequantize_fp8_rowwise(x_fp8: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 E4M3FN [M, K] + per-row FP32 scale [M, 1] -> contiguous BF16 [M, K]."""
    m, k = int(x_fp8.shape[0]), int(x_fp8.shape[1])
    head = 1 << (k.bit_length() - 1)
    tail = k - head
    if (
        x_fp8.is_cuda
        and x_fp8.stride(1) == 1
        and head <= 8192
        and (tail == 0 or (tail & (tail - 1)) == 0)
    ):
        y = torch.empty((m, k), dtype=torch.bfloat16, device=x_fp8.device)
        warps = 8 if k >= 2048 else (4 if k >= 512 else 2)
        _dequant_fp8_rowwise_ht_kernel[(m,)](
            x_fp8, x_scale, y, x_fp8.stride(0), y.stride(0),
            HEAD=head, TAIL=tail, num_warps=warps,
        )
        return y
    return (x_fp8.to(torch.float32) * x_scale).to(torch.bfloat16)


def pf_dense_fp8_linear_prequant(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    min_n: int | None = None,
) -> torch.Tensor | None:
    """Direct `_scaled_mm` from pre-quantized rowwise FP8 `(x_fp8 [M, K], x_scale [M, 1])`
    (skips `_quant_fp8_rowwise_ht_kernel` completely when fed from FP8 AllGather!)."""
    if (
        not x_fp8.is_cuda
        or x_fp8.dtype != torch.float8_e4m3fn
        or weight.dtype != torch.bfloat16
        or x_fp8.ndim != 2
        or weight.ndim != 2
        or x_fp8.stride(1) != 1
        or weight.stride(1) != 1
    ):
        return None
    m, k = int(x_fp8.shape[0]), int(x_fp8.shape[1])
    n = int(weight.shape[0])
    n_floor = _PF_DENSE_FP8_MIN_N if min_n is None else min_n
    if m <= 0 or k < 256 or (k & 15) != 0 or n < n_floor:
        return None
    if out is not None and out.dtype != torch.bfloat16:
        return None
    mode, supports_out = _probe_fp8_cap(x_fp8.device)
    if mode != "rowwise":
        return None
    w_fp8_t, w_scale, n_pad = _get_cached_w_fp8(weight, "rowwise")
    m_pad = (m + 15) & ~15
    if m_pad != m:
        x_in = torch.zeros((m_pad, k), dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_in[:m].copy_(x_fp8)
        s_in = torch.ones((m_pad, 1), dtype=torch.float32, device=x_fp8.device)
        s_in[:m].copy_(x_scale.reshape(m, 1))
    else:
        x_in = x_fp8
        s_in = (
            x_scale
            if (
                x_scale.ndim == 2
                and x_scale.shape[1] == 1
                and x_scale.dtype == torch.float32
                and x_scale.is_contiguous()
            )
            else x_scale.to(torch.float32).reshape(m, 1).contiguous()
        )
    can_direct_out = (
        out is not None
        and supports_out
        and m_pad == m
        and n_pad == n
        and out.is_contiguous()
    )
    try:
        if can_direct_out:
            torch._scaled_mm(
                x_in, w_fp8_t, scale_a=s_in, scale_b=w_scale,
                out_dtype=torch.bfloat16, use_fast_accum=False, out=out,
            )
            return out
        res = torch._scaled_mm(
            x_in, w_fp8_t, scale_a=s_in, scale_b=w_scale,
            out_dtype=torch.bfloat16, use_fast_accum=False,
        )
    except Exception:
        return None
    if m_pad != m or n_pad != n:
        res = res[:m, :n]
    if out is not None:
        out.copy_(res)
        return out
    return res


def quantize_fp8_tensorwise(x: torch.Tensor, pad_m16: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor BF16 [M, K] -> (x_fp8 [M_pad, K] e4m3fn, x_scale [1] fp32)."""
    m, k = x.shape
    m_pad = ((m + 15) & ~15) if pad_m16 else m
    amax = x.abs().amax().to(torch.float32)
    scale = torch.clamp(amax * (1.0 / _FP8_MAX), min=_FP8_EPS).reshape(1)
    inv = (1.0 / scale).to(x.dtype)
    q = torch.clamp(x * inv, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    if m_pad == m:
        return q.contiguous(), scale
    out = torch.zeros((m_pad, k), dtype=torch.float8_e4m3fn, device=x.device)
    out[:m].copy_(q)
    return out, scale


def _probe_fp8_cap(device: torch.device) -> tuple[str, bool]:
    """One-time SM121a capability probe for torch._scaled_mm (rowwise vs tensorwise, out=)."""
    global _FP8_CAP
    if _FP8_CAP is not None:
        return _FP8_CAP
    a = torch.zeros((16, 64), dtype=torch.float8_e4m3fn, device=device)
    b = torch.zeros((64, 64), dtype=torch.float8_e4m3fn, device=device).t()
    out_probe = torch.empty((16, 64), dtype=torch.bfloat16, device=device)
    order = ("rowwise", "tensorwise") if _PF_DENSE_FP8_MODE != "tensorwise" else ("tensorwise", "rowwise")
    for mode in order:
        try:
            if mode == "rowwise":
                sa = torch.ones((16, 1), dtype=torch.float32, device=device)
                sb = torch.ones((1, 64), dtype=torch.float32, device=device)
                _ = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=False)
                supports_out = False
                try:
                    torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=False, out=out_probe)
                    supports_out = True
                except Exception:
                    supports_out = False
                _FP8_CAP = ("rowwise", supports_out)
                return _FP8_CAP
            else:
                sa = torch.ones((1,), dtype=torch.float32, device=device)
                sb = torch.ones((1,), dtype=torch.float32, device=device)
                _ = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True)
                supports_out = False
                try:
                    torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True, out=out_probe)
                    supports_out = True
                except Exception:
                    supports_out = False
                _FP8_CAP = ("tensorwise", supports_out)
                return _FP8_CAP
        except Exception:
            continue
    _FP8_CAP = ("off", False)
    return _FP8_CAP


def _get_cached_w_fp8(weight: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Lazily quantize & cache BF16 weight [N, K] -> (w_fp8_T [K, N_pad], w_scale, N_pad)."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    key = (int(weight.data_ptr()), n, k, int(weight.stride(0)), int(weight.stride(1)), mode)
    cached = _FP8_W_CACHE.get(key)
    if cached is not None:
        return cached
    n_pad = (n + 15) & ~15
    w_f32 = weight.float()
    if mode == "rowwise":
        amax = w_f32.abs().amax(dim=1)  # [N]
        sc = torch.clamp(amax * (1.0 / _FP8_MAX), min=_FP8_EPS)  # [N]
        w_q = torch.clamp(w_f32 / sc[:, None], -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        if n_pad != n:
            w_pad = torch.zeros((n_pad, k), dtype=torch.float8_e4m3fn, device=weight.device)
            w_pad[:n].copy_(w_q)
            sc_pad = torch.ones((1, n_pad), dtype=torch.float32, device=weight.device)
            sc_pad[0, :n].copy_(sc)
            w_fp8, w_scale = w_pad, sc_pad
        else:
            w_fp8 = w_q.contiguous()
            w_scale = sc.reshape(1, n).contiguous()
    else:
        amax = w_f32.abs().amax()
        sc = torch.clamp(amax * (1.0 / _FP8_MAX), min=_FP8_EPS).reshape(1)
        w_q = torch.clamp(w_f32 / sc, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        if n_pad != n:
            w_pad = torch.zeros((n_pad, k), dtype=torch.float8_e4m3fn, device=weight.device)
            w_pad[:n].copy_(w_q)
            w_fp8 = w_pad
        else:
            w_fp8 = w_q.contiguous()
        w_scale = sc.contiguous()
    # Keep transpose view [K, N_pad] (col-major contiguous!) cached to save view overhead.
    entry = (w_fp8.t(), w_scale, n_pad)
    _FP8_W_CACHE[key] = entry
    return entry


def pf_dense_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    mode_override: str | None = None,
    min_n: int | None = None,
) -> torch.Tensor | None:
    """Dynamic FP8 dense linear for prefill chunks (M >= 1024) on SM121a.

    Returns None to fall back cleanly to BF16 cuBLAS if unsupported.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.ndim != 2
        or x.stride(1) != 1
        or weight.stride(1) != 1
    ):
        return None
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(weight.shape[0])
    n_floor = _PF_DENSE_FP8_MIN_N if min_n is None else min_n
    if k < 256 or (k & 15) != 0 or n < n_floor:
        return None
    if out is not None and out.dtype != torch.bfloat16:
        return None

    if mode_override is not None:
        mode = mode_override
        supports_out = True
    else:
        mode, supports_out = _probe_fp8_cap(x.device)
    if mode == "off":
        return None

    w_fp8_t, w_scale, n_pad = _get_cached_w_fp8(weight, mode)
    if mode == "rowwise":
        x_fp8, x_scale = quantize_fp8_rowwise(x, pad_m16=True)
        fast_accum = False
    else:
        x_fp8, x_scale = quantize_fp8_tensorwise(x, pad_m16=True)
        fast_accum = True

    m_pad = int(x_fp8.shape[0])
    can_direct_out = (
        out is not None
        and supports_out
        and m_pad == m
        and n_pad == n
        and out.is_contiguous()
    )
    try:
        if can_direct_out:
            torch._scaled_mm(
                x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale,
                out_dtype=torch.bfloat16, use_fast_accum=fast_accum, out=out,
            )
            return out
        res = torch._scaled_mm(
            x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale,
            out_dtype=torch.bfloat16, use_fast_accum=fast_accum,
        )
    except Exception:
        return None

    if m_pad != m or n_pad != n:
        res = res[:m, :n]
    if out is not None:
        out.copy_(res)
        return out
    return res


def _pdl_site(x, weight, out) -> bool:
    if not _PDL:
        return False
    n, k = weight.shape
    if _PDL_WO and n == _PDL_WO_N and k == _PDL_WO_K:
        # wo_ud in TP decode on GB10 (out=None or out=sbuf). Only when the preceding
        # merge_gate_sink was the PDL trigger (sm12_attn._WO_ARMED, set at trace/capture
        # time); the drafter o_proj has the same shape but no triggering predecessor.
        from . import sm12_attn
        return sm12_attn._WO_ARMED
    if out is None:
        return _PDL_QKVR and k == 4608 and n in _PDL_QKVR_N
    return False


def linear(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 ``x @ weight.T`` with fp32 accumulation (into ``out`` if given)."""
    m = x.shape[0]
    if x.dtype == torch.float8_e4m3fn:
        x_scale = getattr(x, "_inkling_fp8_scale", None)
        if x_scale is not None:
            if _PF_DENSE_FP8 and not torch.cuda.is_current_stream_capturing():
                res = pf_dense_fp8_linear_prequant(x, x_scale, weight, out)
                if res is not None:
                    return res
            x = dequantize_fp8_rowwise(x, x_scale)
    if _SKINNY and m <= 16:
        from . import skinny_bf11, skinny_bf12
        from .skinny import skinny_linear, supported
        if supported(x, weight) and (out is None or out.stride(1) == 1):
            pdl = _pdl_site(x, weight, out)
            if _PDL_WO and tuple(weight.shape) == (_PDL_WO_N, _PDL_WO_K):
                from . import sm12_attn
                sm12_attn._WO_ARMED = False  # one-shot: consumed by this wo_ud launch
            if skinny_bf11.ENABLED:
                trig = pdl and (out is not None or weight.shape[1] == 4608)
                res = skinny_bf11.bf11_linear(x, weight, out, pdl=pdl, trig=trig)
                if res is not None:
                    return res
            if skinny_bf12.ENABLED:
                res = skinny_bf12.bf12_linear(x, weight, out, pdl=pdl, trig_reduce=pdl)
                if res is not None:
                    return res
            return skinny_linear(x, weight, out, pdl=pdl, trig_reduce=pdl)
    elif _SKINNY and m > 16:
        from . import skinny_bf11
        if (
            skinny_bf11.ENABLED
            and skinny_bf11.C32_ENABLED
            and m <= skinny_bf11.C32_MAX_M
            and x.dim() == 2
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.stride(1) == 1
            and weight.stride(1) == 1
            and (out is None or out.stride(1) == 1)
        ):
            pdl = _pdl_site(x, weight, out)
            if _PDL_WO and tuple(weight.shape) == (_PDL_WO_N, _PDL_WO_K):
                from . import sm12_attn
                sm12_attn._WO_ARMED = False
            trig = pdl and (out is not None or weight.shape[1] == 4608)
            res = skinny_bf11.bf11_linear(x, weight, out, pdl=pdl, trig=trig)
            if res is not None:
                return res
    if (
        _PF_DENSE_FP8
        and m >= _PF_DENSE_FP8_MIN_M
        and not torch.cuda.is_current_stream_capturing()
    ):
        res = pf_dense_fp8_linear(x, weight, out)
        if res is not None:
            return res
    if out is not None:
        return torch.mm(x, weight.t(), out=out)
    # Autotuning would profile the CuTe DSL kernels past their 32-row limit.
    if (x.shape[0] > LOW_M_MAX_TOKENS or is_flashinfer_autotuning()
            or _SM12):
        return torch.nn.functional.linear(x, weight)
    from flashinfer import mm_bf16

    return mm_bf16(x, weight.t(), pdl=True, backend="cute-dsl")


@triton.jit
def _k_group_tile(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    pid_n,
    pid_m,
    pid_k,
):
    # Row tiles of at most 64 keep tl.dot on the same mma.sync instruction
    # sequence as the reference's 16x32x32 tiling. With SPLIT, each program computes one
    # K group's partial; otherwise it sums the groups in order.
    cols = pid_n * BN + tl.arange(0, BN)
    rows = pid_m * BM + tl.arange(0, BM)
    ks = tl.arange(0, 128)
    first = pid_k if SPLIT else 0
    NUM_GROUPS: tl.constexpr = 1 if SPLIT else (K + 1023) // 1024
    total = tl.zeros((BM, BN), tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        group = first + g
        partial = tl.zeros((BM, BN), tl.float32)
        for tile in range(8):
            k = group * 1024 + tile * 128 + ks
            x = tl.load(
                X + rows[:, None] * K + k[None, :],
                (rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                W + cols[None, :] * K + k[:, None],
                (cols[None, :] < N) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        total = partial if g == 0 else total + partial
    tl.store(
        Y + (first * M + rows[:, None]) * N + cols[None, :],
        total.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _prefetch_k_group_tile(
    W, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, pid_n, pid_k
):
    # Pulls one K group's weight tile into L2, one 128-byte line per element.
    cols = pid_n * BN + tl.arange(0, BN)
    k = pid_k * 1024 + tl.arange(0, 16) * 64
    mask = (cols[:, None] < N) & (k[None, :] < K)
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; "
        "@p prefetch.global.L2 [$1]; mov.b32 $0, 0; }",
        "=r,l,r",
        [W + cols[:, None] * K + k[None, :], mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _k_group_kernel(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    if SPLIT and tl.program_id(1) == 0:
        _prefetch_k_group_tile(W, N, K, BN, tl.program_id(0), tl.program_id(2))
    tl.extra.cuda.gdc_wait()
    _k_group_tile(
        X,
        W,
        Y,
        M,
        N,
        K,
        BM,
        BN,
        SPLIT,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


@triton.jit
def _k_group_sum_kernel(Partials, Y, MN, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


@triton.jit
def _k_group_tma_kernel(
    XDesc,
    WDesc,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    # One output tile over all of K: each group's partial accumulates on the
    # tensor cores and is added to the running sum in order.
    TILES: tl.constexpr = (K + BK - 1) // BK
    GROUP_TILES: tl.constexpr = 1024 // BK
    rows = tl.program_id(1) * BM
    cols = tl.program_id(0) * BN
    tl.extra.cuda.gdc_wait()
    if LAUNCH_DEPENDENTS:
        tl.extra.cuda.gdc_launch_dependents()
    total = tl.zeros((BM, BN), tl.float32)
    partial = tl.zeros((BM, BN), tl.float32)
    for tile in tl.range(TILES, warp_specialize=True):
        x = XDesc.load([rows, tile * BK])
        w = WDesc.load([cols, tile * BK])
        partial = tl.dot(x, w.T, partial)
        if tile % GROUP_TILES == GROUP_TILES - 1 or tile == TILES - 1:
            total = total + partial
            partial = tl.zeros((BM, BN), tl.float32)
    r = rows + tl.arange(0, BM)[:, None]
    c = cols + tl.arange(0, BN)[None, :]
    tl.store(Y + r * N + c, total.to(Y.dtype.element_ty), (r < M) & (c < N))


# From this many rows, 128-row TMA tiles over all of K fill the GPU.
TMA_MIN_TOKENS = 2 ** 31  # sm121: TMA tiles exceed the 99KB shared-memory limit


def k_group_partials(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
    launch_dependents: bool = False,
) -> torch.Tensor:
    """sm121: the Triton k-group and TMA kernels both exceed the 99 KB
    shared-memory limit of GB10. Compute the same fp32 partials (G=1,
    fully summed) with a tensor-core matmul: bf16 inputs accumulate in
    fp32, matching the reference's precision up to summation order.

    When ``INKLING_PF_ROUTER_FP8=1`` and ``M >= INKLING_PF_ROUTER_FP8_MIN_M``
    (prefill only, never inside CUDA graph capture), routes through SM121a
    FP8 ``torch._scaled_mm`` — consuming pre-quantized ``(x_fp8, x_scale)``
    directly with ZERO dequant/re-quant when fed from FP8-Direct SP AllGather!"""
    m = int(x.shape[0])
    target_dtype = torch.bfloat16 if out_dtype == torch.float8_e4m3fn else out_dtype
    if x.dtype == torch.float8_e4m3fn:
        x_scale = getattr(x, "_inkling_fp8_scale", None)
        if x_scale is None:
            raise ValueError("FP8 tensor passed to k_group_partials without _inkling_fp8_scale")
        if (
            _PF_ROUTER_FP8
            and m >= _PF_ROUTER_FP8_MIN_M
            and not torch.cuda.is_current_stream_capturing()
        ):
            res = pf_dense_fp8_linear_prequant(x, x_scale, weight, min_n=16)
            if res is not None:
                if res.dtype != target_dtype:
                    res = res.to(target_dtype)
                return res.unsqueeze(0)
        x = dequantize_fp8_rowwise(x, x_scale)
    elif (
        _PF_ROUTER_FP8
        and x.dtype == torch.bfloat16
        and m >= _PF_ROUTER_FP8_MIN_M
        and not torch.cuda.is_current_stream_capturing()
    ):
        res = pf_dense_fp8_linear(x, weight, min_n=16)
        if res is not None:
            if res.dtype != target_dtype:
                res = res.to(target_dtype)
            return res.unsqueeze(0)
    out = torch.matmul(x, weight.t())
    if out.dtype != target_dtype:
        out = out.to(target_dtype)
    return out.unsqueeze(0)


def _k_group_partials_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
    launch_dependents: bool = False,
) -> torch.Tensor:
    """fp32 partials [G, T, N] of ``x @ weight.T`` whose in-order sum over G is
    the reference ``linear``. G is 1 (already summed, in ``out_dtype``) when
    there are enough rows to fill the GPU without splitting K. With
    ``launch_dependents``, the next kernel may start (and must wait) while
    the TMA kernel runs."""
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    n = weight.shape[0]
    if tokens >= TMA_MIN_TOKENS:
        from triton.tools.tensor_descriptor import TensorDescriptor

        out = torch.empty((1, tokens, n), dtype=out_dtype, device=x.device)
        block_n = 64 if tokens < 4096 else 128
        _k_group_tma_kernel[(triton.cdiv(n, block_n), triton.cdiv(tokens, 128))](
            TensorDescriptor.from_tensor(x, [128, 128]),
            TensorDescriptor.from_tensor(weight, [block_n, 128]),
            out,
            tokens,
            n,
            hidden,
            128,
            block_n,
            128,
            launch_dependents,
            num_warps=4 if block_n == 64 else 8,
            num_stages=4 if block_n == 64 else 3,
            enable_fp_fusion=False,
            launch_pdl=True,
        )
        return out
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    groups = triton.cdiv(hidden, 1024)
    split = groups > 1 and tokens < 4096
    num_warps = 4 if block_n <= 64 else 8
    num_stages = 4 if not split else 3 if block_n <= 64 else 2
    out = torch.empty(
        (groups if split else 1, tokens, n),
        dtype=torch.float32 if split else out_dtype,
        device=x.device,
    )
    grid = (
        triton.cdiv(n, block_n),
        triton.cdiv(tokens, block_m),
        groups if split else 1,
    )
    _k_group_kernel[grid](
        x,
        weight,
        out,
        tokens,
        n,
        hidden,
        block_m,
        block_n,
        split,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return out


def linear_k_groups(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [T, K] and ``weight`` [N, K].

    Each 1024-wide K group is accumulated in fp32 and the partials are summed in
    order, as in the reference ``linear``.
    """
    partials = k_group_partials(x, weight, out_dtype)
    if partials.shape[0] == 1:
        return partials[0]
    out = torch.empty(partials.shape[1:], dtype=out_dtype, device=x.device)
    _k_group_sum_kernel[(triton.cdiv(out.numel(), 1024),)](
        partials, out, out.numel(), partials.shape[0], 1024, launch_pdl=True
    )
    return out
