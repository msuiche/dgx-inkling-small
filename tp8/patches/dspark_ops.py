# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels of the Inkling2 DSpark draft, matching the reference implementation."""

import os

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, tldevice, triton

from .allreduce import _multicast_memory

# INKLING_DRAFT_PDL=1: opt-in PDL across drafter Triton kernels (default 0 for safe A/B).
# Invariant: ALWAYS gdc_wait() BEFORE reading any dynamic input, and gdc_launch_dependents() AFTER gdc_wait().
_DRAFT_PDL = os.environ.get("INKLING_DRAFT_PDL", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)


@triton.jit
def _mix_rmsnorm_kernel(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    mixed = tl.full((BLOCK,), 0, tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT + row * HIDDEN + h, mixed * inv_rms * weight, h < HIDDEN)


def mix_rmsnorm(
    targets: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    pdl: bool | None = None,
) -> torch.Tensor:
    """RMSNorm of the ``alpha``-weighted sum of ``targets[:, i]`` (fp32 math).

    ``targets`` is ``(rows, taps, hidden)``; the output has ``targets.dtype``.
    """
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    out = targets.new_empty((rows, hidden))
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if rows:
        _mix_rmsnorm_kernel[(rows,)](
            targets,
            alpha,
            weight,
            out,
            *targets.stride(),
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            PDL=use_pdl,
            num_warps=8,
            launch_pdl=use_pdl,
        )
    return out


@triton.jit
def _fused_context_anchor_mix_rmsnorm_kernel(
    HID,
    QSL,
    NUM_REJ,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Fuses prepare_context_anchor (ends = qsl[1:] - nr; rows = hidden[clamp(ends-1,0)];
    f_rnn[:R] = mix_rmsnorm(rows)) from 7 separate kernels (incl. 46 KB index_select copy
    + memcpy32_post!) into 1 single Triton kernel (~3.5 us, 0 temp alloc, 0 L2PF contention)."""
    req = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
    end_pos = tl.load(QSL + req + 1).to(tl.int64) - tl.load(NUM_REJ + req).to(tl.int64)
    src_row = tl.maximum(end_pos - 1, 0)
    mixed = tl.full((BLOCK,), 0.0, tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            HID + src_row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0.0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT + req * OUT_ROW_STRIDE + h, (mixed * inv_rms * weight).to(OUT.dtype.element_ty), h < HIDDEN)


def context_anchor_mix_rmsnorm(
    hidden_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_rejected: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    num_reqs: int,
    taps: int,
    hidden: int,
    eps: float,
    pdl: bool | None = None,
) -> None:
    """In-place fused prepare_context_anchor -> out[:num_reqs] (replaces 7 kernels with 1)."""
    if num_reqs <= 0:
        return
    targets = hidden_states.view(hidden_states.shape[0], taps, hidden)
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    _fused_context_anchor_mix_rmsnorm_kernel[(num_reqs,)](
        targets,
        query_start_loc,
        num_rejected,
        alpha,
        weight,
        out,
        targets.stride(0),
        targets.stride(1),
        targets.stride(2),
        out.stride(0),
        TAPS=taps,
        HIDDEN=hidden,
        EPS=eps,
        BLOCK=triton.next_power_of_2(hidden),
        PDL=use_pdl,
        num_warps=8,
        launch_pdl=use_pdl,
    )


@triton.jit
def _embed_block_ln_kernel(
    IDS,
    EMB_W,
    EMB_NW,
    MASK_E,
    LN_W,
    RES,
    OUT,
    stride_emb,
    full_v,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EMB_EPS: tl.constexpr,
    EPS: tl.constexpr,
    B_H: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Fuses embed_tokens(anchor) + embed_norm + mask_embed fill + L0.input_layernorm.
    Replaces 11 eager/graph PyTorch kernels (~32 us) with 1 Triton kernel (~8.4 us, max_abs=0).
    Handles OOB token IDs (< 0 or >= full_v) identically to _repl_forward (zeroed before norm)."""
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, B_H)
    m = h < H
    is_anchor = (row % BLOCK_SIZE) == 0
    if PDL:
        tl.extra.cuda.gdc_wait()
    if is_anchor:
        raw = tl.load(IDS + row).to(tl.int64)
        oob = (raw < 0) | (raw >= full_v)
        tok = tl.minimum(tl.maximum(raw, 0), full_v - 1)
        x = tl.load(EMB_W + tok * stride_emb + h, mask=m & (~oob), other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / H + EMB_EPS)
        nw = tl.load(EMB_NW + h, mask=m, other=0.0).to(tl.float32)
        x = (x * inv * nw).to(tl.bfloat16)
    else:
        x = tl.load(MASK_E + h, mask=m, other=0.0).to(tl.bfloat16)
    xf = x.to(tl.float32)
    inv2 = tl.rsqrt(tl.sum(xf * xf, 0) / H + EPS)
    lw = tl.load(LN_W + h, mask=m, other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(RES + row * H + h, x, mask=m)
    tl.store(OUT + row * H + h, (xf * inv2 * lw).to(tl.bfloat16), mask=m)


def embed_block_ln_fused(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    embed_norm_weight: torch.Tensor,
    mask_embed: torch.Tensor,
    ln_weight: torch.Tensor,
    block_size: int,
    eps: float,
    emb_eps: float | None = None,
    pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``embed_block`` + ``layers[0].input_layernorm`` returning ``(x_normed, residual)``."""
    num_rows = int(input_ids.numel())
    hidden = int(mask_embed.shape[0])
    emb_nw = embed_norm_weight.reshape(-1)
    if not emb_nw.is_contiguous():
        emb_nw = emb_nw.contiguous()
    residual = mask_embed.new_empty((num_rows, hidden))
    out = mask_embed.new_empty((num_rows, hidden))
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if num_rows:
        _embed_block_ln_kernel[(num_rows,)](
            input_ids.reshape(-1),
            embed_weight,
            emb_nw,
            mask_embed,
            ln_weight,
            residual,
            out,
            embed_weight.stride(0),
            int(embed_weight.shape[0]),
            H=hidden,
            BLOCK_SIZE=block_size,
            EMB_EPS=eps if emb_eps is None else float(emb_eps),
            EPS=eps,
            B_H=triton.next_power_of_2(hidden),
            PDL=use_pdl,
            num_warps=8,
            launch_pdl=use_pdl,
        )
    return out, residual


@triton.jit
def _qk_norm_rope_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POS,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    PDL: tl.constexpr = False,
    OTF_ROPE: tl.constexpr = False,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    row = X + t * row_stride + col0
    if PDL:
        tl.extra.cuda.gdc_wait()
    pos = tl.load(POS + t).to(tl.int64)
    if OTF_ROPE:
        angles = pos.to(tl.float32) * tl.load(COS_SIN + half_ar).to(tl.float32)
        cos = tldevice.cos(angles)
        sin = tldevice.sin(angles)
    else:
        cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
        sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)

    if NORM:
        w_ptr = tl.where(is_q, Q_WEIGHT.to(tl.int64), K_WEIGHT.to(tl.int64)).to(
            tl.pointer_type(tl.bfloat16)
        )
        x = tl.load(row + d_ar).to(tl.float32)
        ms = tl.sum(x * x, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w1 = tl.load(w_ptr + half_ar).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
        x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
        x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
    else:
        x1 = tl.load(row + half_ar).to(tl.float32)
        x2 = tl.load(row + HALF + half_ar).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(row + half_ar, o1.to(tl.bfloat16))
    tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))


@triton.jit
def _qk_norm_rope_write_kv_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POS,
    KC,
    VC,
    SLOT_MAP,
    row_stride,
    q_size,
    sk_b,
    sk_s,
    sk_h,
    sv_b,
    sv_s,
    sv_h,
    slot_stride,
    kv_block_size,
    NHQ: tl.constexpr,
    NHK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    PDL: tl.constexpr = False,
    OTF_ROPE: tl.constexpr = False,
):
    """Fused Q/K RMSNorm (bf16-rounded) + neox RoPE + direct paged KV-cache write.
    - h < NHQ: normalizes & rotates Q head h in-place in X.
    - h >= NHQ (kv_h = h - NHQ): normalizes & rotates K head directly from registers into
      KC[slot // BS, slot % BS, kv_h, :] and copies V head from X[..., q_size + (NHK + kv_h)*D]
      into VC[slot // BS, slot % BS, kv_h, :] (skipping slot < 0).
    Eliminates 10x reshape_and_cache_flash launches (5 trunk + 5 ctxkv); 0-ULP bit-exact."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    if PDL:
        tl.extra.cuda.gdc_wait()
    pos = tl.load(POS + t).to(tl.int64)
    if OTF_ROPE:
        angles = pos.to(tl.float32) * tl.load(COS_SIN + half_ar).to(tl.float32)
        cos = tldevice.cos(angles)
        sin = tldevice.sin(angles)
    else:
        cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
        sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)

    if NHQ > 0:
        if h < NHQ:
            row = X + t * row_stride + h.to(tl.int64) * D
            x = tl.load(row + d_ar).to(tl.float32)
            ms = tl.sum(x * x, 0) / D
            inv = 1.0 / tl.sqrt(ms + EPS)
            w1 = tl.load(Q_WEIGHT + half_ar).to(tl.float32)
            w2 = tl.load(Q_WEIGHT + HALF + half_ar).to(tl.float32)
            x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
            x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
            x1 = x1.to(tl.bfloat16).to(tl.float32)
            x2 = x2.to(tl.bfloat16).to(tl.float32)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(row + half_ar, o1.to(tl.bfloat16))
            tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))
            return

    kv_h = (h - NHQ).to(tl.int64)
    row_k = X + t * row_stride + q_size + kv_h * D
    row_v = row_k + NHK * D
    x = tl.load(row_k + d_ar).to(tl.float32)
    ms = tl.sum(x * x, 0) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    w1 = tl.load(K_WEIGHT + half_ar).to(tl.float32)
    w2 = tl.load(K_WEIGHT + HALF + half_ar).to(tl.float32)
    x1 = tl.load(row_k + half_ar).to(tl.float32) * inv * w1
    x2 = tl.load(row_k + HALF + half_ar).to(tl.float32) * inv * w2
    x1 = x1.to(tl.bfloat16).to(tl.float32)
    x2 = x2.to(tl.bfloat16).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    slot = tl.load(SLOT_MAP + t * slot_stride).to(tl.int64)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if slot >= 0:
        blk = slot // kv_block_size
        off = slot % kv_block_size
        kc_row = KC + blk * sk_b + off * sk_s + kv_h * sk_h
        vc_row = VC + blk * sv_b + off * sv_s + kv_h * sv_h
        v1 = tl.load(row_v + half_ar)
        v2 = tl.load(row_v + HALF + half_ar)
        tl.store(kc_row + half_ar, o1.to(KC.dtype.element_ty))
        tl.store(kc_row + HALF + half_ar, o2.to(KC.dtype.element_ty))
        tl.store(vc_row + half_ar, v1.to(VC.dtype.element_ty))
        tl.store(vc_row + HALF + half_ar, v2.to(VC.dtype.element_ty))


def qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    eps: float,
    pdl: bool | None = None,
) -> None:
    """In-place per-head Q/K RMSNorm (rounded to bf16) + neox RoPE on the
    leading ``num_q_heads + num_k_heads`` heads of the fused ``qkv``."""
    otf_rope = cos_sin_cache.ndim == 1
    head_dim = cos_sin_cache.shape[0] * 2 if otf_rope else cos_sin_cache.shape[-1]
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if qkv.shape[0]:
        _qk_norm_rope_kernel[(qkv.shape[0], num_q_heads + num_k_heads)](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            qkv.stride(0),
            num_q_heads * head_dim,
            NHQ=num_q_heads,
            D=head_dim,
            EPS=eps,
            NORM=True,
            PDL=use_pdl,
            OTF_ROPE=otf_rope,
            launch_pdl=use_pdl,
        )


def qk_norm_rope_write_kv_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    eps: float,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pdl: bool | None = None,
) -> None:
    """Fused Q/K RMSNorm + neox RoPE (Q in-place in ``qkv``) + paged KV cache write.
    ``key_cache`` and ``value_cache`` have shape ``(num_blocks, block_size, num_k_heads, head_dim)``
    with contiguous last dimension."""
    otf_rope = cos_sin_cache.ndim == 1
    head_dim = cos_sin_cache.shape[0] * 2 if otf_rope else cos_sin_cache.shape[-1]
    assert qkv.stride(1) == 1
    assert key_cache.stride(-1) == 1 and value_cache.stride(-1) == 1
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if qkv.shape[0]:
        _qk_norm_rope_write_kv_kernel[(qkv.shape[0], num_q_heads + num_k_heads)](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            key_cache,
            value_cache,
            slot_mapping,
            qkv.stride(0),
            num_q_heads * head_dim,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            slot_mapping.stride(0),
            key_cache.shape[1],
            NHQ=num_q_heads,
            NHK=num_k_heads,
            D=head_dim,
            EPS=eps,
            PDL=use_pdl,
            OTF_ROPE=otf_rope,
            launch_pdl=use_pdl,
        )


def rope_(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pdl: bool | None = None,
):
    """In-place neox RoPE on ``x`` of shape ``(T, H, D)`` (fp32 math)."""
    num_tokens, num_heads, head_dim = x.shape
    assert x.stride(1) == head_dim and x.stride(2) == 1
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    otf_rope = cos_sin_cache.ndim == 1
    if num_tokens:
        _qk_norm_rope_kernel[(num_tokens, num_heads)](
            x,
            x,
            x,
            cos_sin_cache,
            positions,
            x.stride(0),
            0,
            NHQ=0,
            D=head_dim,
            EPS=0.0,
            NORM=False,
            PDL=use_pdl,
            OTF_ROPE=otf_rope,
            launch_pdl=use_pdl,
        )


@triton.jit
def _silu_and_mul_kernel(X, Y, stride_x, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask, other=0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + tldevice.exp(-gate))
    tl.store(Y + row * N + cols, (silu * up).to(Y.dtype.element_ty), mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` in fp32 (precise exp/div), rounded once."""
    num_tokens, width = x.shape
    n = width // 2
    out = x.new_empty((num_tokens, n))
    if num_tokens:
        _silu_and_mul_kernel[(num_tokens, triton.cdiv(n, 1024))](
            x, out, x.stride(0), n, 1024, num_warps=4
        )
    return out


@triton.jit
def _add_softcap_kernel(A, B, OUT, stride_a, stride_b, softcap, N, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask, other=0).to(tl.float32)
    x = (a + b).to(tl.bfloat16)
    x = x / softcap
    x = tldevice.tanh(x)
    x = x * softcap
    tl.store(OUT + row * N + cols, x.to(tl.bfloat16), mask)


def add_softcap(a: torch.Tensor, b: torch.Tensor, softcap: float) -> torch.Tensor:
    """bf16 ``softcap * tanh(bf16(a + b) / softcap)`` of two bf16 2D tensors."""
    rows, n = a.shape
    assert a.stride(1) == 1 and b.stride(1) == 1
    out = a.new_empty((rows, n))
    if rows:
        _add_softcap_kernel[(triton.cdiv(n, 1024), rows)](
            a, b, out, a.stride(0), b.stride(0), softcap, n, 1024
        )
    return out


@triton.jit
def _round_bf16(x):
    # fp32 -> bf16 -> fp32 (round to nearest even) on the bits: the compiler
    # folds an extf(truncf(x)) pair, and with it fuses mul + add into an FMA.
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _markov_gate_kernel(RAW, STATE, OUT, LATENT, RANK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    raw = RAW + row * 3 * RANK + c
    gate = _round_bf16(tl.load(raw))
    cand = _round_bf16(tl.load(raw + RANK))
    out = _round_bf16(tl.load(raw + 2 * RANK))
    state = tl.load(STATE + row * RANK + c).to(tl.float32)
    # Each torch op of the reference rounds to bf16; its sigmoid is
    # 1 / (1 + expf(-x)) with IEEE division.
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate)))
    cand = _round_bf16(tldevice.tanh(cand))
    keep = _round_bf16(gate * state)
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    tl.store(OUT + row * RANK + c, (keep + update).to(tl.bfloat16))
    tl.store(LATENT + row * RANK + c, tldevice.tanh(out).to(tl.bfloat16))


def markov_gate(
    raw: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrent head's gating of the fp32 ``rnn_proj`` output ``raw``
    (rounded to bf16 first), as the reference's bf16 torch ops: returns
    ``(gate * state + (1 - gate) * tanh(cand), tanh(out))``."""
    rows, rank = state.shape
    assert raw.shape == (rows, 3 * rank) and raw.is_contiguous()
    assert state.is_contiguous() and rank & (rank - 1) == 0
    out, latent = torch.empty_like(state), torch.empty_like(state)
    if rows:
        _markov_gate_kernel[(rows,)](raw, state, out, latent, rank)
    return out, latent


class Inkling2DSparkArgmax:
    """Lamport slots of ``dspark_argmax``: each row's per-rank, per-block
    (max, index) pairs, multicast to every rank, in two buffers that a row
    alternates between; and each row's arrival count and buffer parity."""

    def __init__(self, group: dist.ProcessGroup, max_rows: int, num_cols: int):
        self.rank, self.world_size = dist.get_rank(group), group.size()
        self.num_cols = num_cols
        num_blocks = triton.cdiv(num_cols, 1024)
        num_bytes = 2 * max_rows * self.world_size * num_blocks * 8
        device = torch.accelerator.current_device_index()
        self.fallback = False
        if self.world_size > 1:
            cap = torch.cuda.get_device_capability(device)
            self.fallback = cap[0] == 12  # GB10: no NVLink multicast
        if self.world_size == 1 or self.fallback:
            # SM12x: dspark_argmax all-gathers per-rank (max, index) pairs.
            self.buffer = torch.full((num_bytes // 4,), -0.0, device=device)
            self.multicast_ptr = self.local_ptr = self.buffer.data_ptr()
        else:
            from flashinfer.comm.mnnvl import is_multicast_supported

            assert is_multicast_supported(device)
            self.buffer = _multicast_memory(num_bytes, group)
            self.multicast_ptr = self.buffer.get_multicast_ptr()
            assert self.multicast_ptr, "Inkling2 DSpark requires NVLink multicast"
            self.local_ptr = self.buffer.get_unicast_ptr(self.rank)
            self.buffer.lamport_initialize(self.rank, torch.float32)
        self.row_state = torch.zeros(max_rows, dtype=torch.int32, device=device)
        torch.accelerator.synchronize()
        dist.barrier(group=group)


# INKLING_ARGMAX_FUSED (default 1): the GB10 (no multicast) Markov argmax is one fused
# softcap+argmax(+sumexp) kernel per vocab slice, a 16 B-aligned RDMA all-gather of the
# slice records and one select kernel. Bit-exact to the legacy 11-kernel + NCCL path
# (selected token); 0 restores the legacy path.
_FUSED = os.environ.get("INKLING_ARGMAX_FUSED", "1") != "0"
_SLICES = int(os.environ.get("INKLING_ARGMAX_SLICES", "16"))  # column slices per row (grid y)


class _ConfSink:
    """Where ``dspark_argmax`` leaves the drafter's top-1 probability of each Markov step
    (INKLING_VERIFY_W=conf / INKLING_CONF_LOG). ``buf`` is a persistent fp32 ``[max_rows, C]``;
    step i of a proposal writes column i. ``col`` is trace-time Python state: the speculator
    resets it before the Markov loop (also while the CUDA graph is captured, so the graph
    bakes the column)."""

    def __init__(self, buf: torch.Tensor):
        self.buf = buf
        self.col = 0


_CONF_SINK: "_ConfSink | None" = None


def set_conf_sink(buf: "torch.Tensor | None") -> None:
    global _CONF_SINK
    _CONF_SINK = None if buf is None else _ConfSink(buf)


def conf_sink_begin() -> bool:
    """Call at the start of one Markov loop; True if the sink is armed."""
    if _CONF_SINK is None:
        return False
    _CONF_SINK.col = 0
    return True


def dspark_argmax(
    workspace: Inkling2DSparkArgmax,
    base: torch.Tensor,
    bias: torch.Tensor,
    softcap: float,
    valid: int,
    offset: int,
) -> torch.Tensor:
    """First argmax (NaN as -inf) of every rank's ``add_softcap(base, bias,
    softcap)[:, :valid]`` (its vocabulary shard at ``offset``), gathered."""
    assert base.dtype == bias.dtype == torch.bfloat16 and base.shape == bias.shape
    assert base.shape[1] == workspace.num_cols
    if workspace.fallback:
        if _FUSED:
            sink = _CONF_SINK
            if sink is not None and sink.col < sink.buf.shape[1] and base.shape[0] <= sink.buf.shape[0]:
                # Top-1 probability of this Markov step into sink.buf[:rows, col]
                # (identical on every rank: computed from the gathered records).
                col = sink.col
                sink.col += 1
                return gathered_argmax_fused(
                    base, bias, softcap, valid, offset, conf_out=sink.buf[: base.shape[0], col]
                )
            return gathered_argmax_fused(base, bias, softcap, valid, offset)
        return _gathered_argmax(base, bias, softcap, valid, offset)
    out = torch.empty(base.shape[0], dtype=torch.int64, device=base.device)
    torch.ops._C.inkling2_dspark_argmax(
        base,
        bias,
        softcap,
        valid,
        offset,
        workspace.multicast_ptr,
        workspace.local_ptr,
        workspace.row_state,
        workspace.rank,
        workspace.world_size,
        out,
    )
    return out


def _gathered_argmax(base, bias, softcap, valid, offset):
    """Same result as the multicast kernel: global first argmax, NaN as -inf.
    Ranks hold consecutive vocab shards, so on ties the lowest rank wins."""
    from vllm.distributed import tensor_model_parallel_all_gather
    logits = add_softcap(base, bias, softcap)[:, :valid].float()
    logits = torch.nan_to_num(logits, nan=float("-inf"))
    m = logits.max(dim=-1).values
    i = (logits == m[:, None]).int().argmax(dim=-1)  # first max index
    pair = torch.stack((m, (i + offset).float()), dim=-1)  # vocab < 2**24: exact
    g = tensor_model_parallel_all_gather(pair.unsqueeze(1).contiguous(), dim=1)
    r = (g[..., 0] == g[..., 0].max(dim=1, keepdim=True).values).int().argmax(dim=1)
    return g.gather(1, r[:, None, None].expand(-1, 1, 2))[:, 0, 1].long()


@triton.jit
def _argmax_record_kernel(
    A, B, REC, stride_a, stride_b, softcap, valid, offset, CH, BLOCK: tl.constexpr
):
    """Per (row, column slice): the slice's max of ``add_softcap(a, b, softcap)``
    (bit for bit the ``_add_softcap_kernel`` arithmetic, NaN and columns >= valid
    as -inf), the first column attaining it (+ offset, as fp32) and
    ``sum(exp(x - max))``. One 16 B record per (row, slice)."""
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    lo = s * CH
    cols = lo + tl.arange(0, BLOCK)
    mask = (cols < valid) & (cols < lo + CH)
    a = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask, other=0).to(tl.float32)
    x = (a + b).to(tl.bfloat16)
    x = x / softcap
    x = tldevice.tanh(x)
    x = x * softcap
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask & (x == x), x, -float("inf"))
    m = tl.max(x, 0)
    idx = tl.min(tl.where(x == m, cols, 2147483647), 0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m0))
    se = tl.sum(e, 0)
    o = tl.arange(0, 4)
    val = tl.where(
        o == 0, m, tl.where(o == 1, (idx + offset).to(tl.float32), tl.where(o == 2, se, 0.0))
    )
    tl.store(REC + (row * S + s) * 4 + o, val)


@triton.jit
def _argmax_select_kernel(
    G, OUT, CONF, rows, S, NS, OSTRIDE, CSTRIDE,
    NP2: tl.constexpr, WITH_CONF: tl.constexpr, PDL: tl.constexpr = False,
):
    """Global first max over the gathered records ``G[rank * rows + row, slice, 4]``
    (rank-major, slice-minor == vocabulary order, so the lowest index wins ties)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, NP2)
    ok = r < NS
    base = (((r // S).to(tl.int64) * rows + row) * S + r % S) * 4
    v = tl.load(G + base, ok, other=-float("inf"))
    i = tl.load(G + base + 1, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), r, NP2), 0)
    tl.store(OUT + row * OSTRIDE, tl.sum(tl.where(r == first, i, 0.0), 0).to(tl.int64))
    if WITH_CONF:
        se = tl.load(G + base + 2, ok, other=0.0)
        m0 = tl.where(m == -float("inf"), 0.0, m)
        w = tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0))
        tot = tl.sum(w, 0)
        tl.store(CONF + row * CSTRIDE, tl.where(tot > 0, 1.0 / tot, 0.0))


def _argmax_records(base, bias, softcap, valid, offset, slices):
    rows = base.shape[0]
    assert base.stride(1) == 1 and bias.stride(1) == 1
    ch = triton.cdiv(triton.cdiv(valid, slices), 8) * 8
    rec = torch.empty((rows, slices, 4), dtype=torch.float32, device=base.device)
    _argmax_record_kernel[(rows, slices)](
        base, bias, rec, base.stride(0), bias.stride(0), softcap, valid, offset, ch,
        BLOCK=triton.next_power_of_2(ch), num_warps=4,
    )
    return rec


def argmax_select(
    g: torch.Tensor,
    rows: int,
    with_conf: bool = False,
    conf_out: "torch.Tensor | None" = None,
    out_tok: "torch.Tensor | None" = None,
    pdl: bool | None = None,
):
    """Tokens (int64 ``[rows]``) from gathered records ``g`` ``[world * rows, S, 4]``
    (and, with ``with_conf``, the top-1 probability ``[rows]`` fp32 of the
    softcapped full-vocabulary distribution: 1 / sum(exp(x - max))). With ``conf_out``
    (fp32 view of >= ``rows`` elements, any row stride) the probability is written there
    instead and only the tokens are returned. With ``out_tok`` (int64 view of >= ``rows``
    elements, any row stride), tokens are written directly into ``out_tok`` (zero memcpy)."""
    ws, s = g.shape[0] // rows, g.shape[1]
    if out_tok is not None:
        assert out_tok.dtype == torch.int64 and out_tok.shape[0] >= rows
        out, ostride = out_tok, out_tok.stride(0)
    else:
        out, ostride = torch.empty(rows, dtype=torch.int64, device=g.device), 1
    if conf_out is not None:
        assert conf_out.dtype == torch.float32 and conf_out.shape[0] >= rows
        conf, cstride, with_conf, ret_conf = conf_out, conf_out.stride(0), True, False
    else:
        conf = torch.empty(rows, dtype=torch.float32, device=g.device) if with_conf else out
        cstride, ret_conf = 1, with_conf
    np2 = triton.next_power_of_2(ws * s)
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    _argmax_select_kernel[(rows,)](
        g, out, conf, rows, s, ws * s, ostride, cstride, np2, with_conf,
        PDL=use_pdl, num_warps=4 if np2 >= 512 else 1, launch_pdl=use_pdl,
    )
    return (out, conf) if ret_conf else out


def gathered_argmax_fused(base, bias, softcap, valid, offset, with_conf=False, conf_out=None):
    """Same tokens as ``_gathered_argmax``: global first argmax, NaN as -inf.
    The 16 B-multiple records take the RDMA all-gather (not NCCL)."""
    from vllm.distributed import tensor_model_parallel_all_gather

    rows = base.shape[0]
    rec = _argmax_records(base, bias, softcap, valid, offset, _SLICES)
    g = tensor_model_parallel_all_gather(rec, dim=0)  # [world * rows, S, 4]
    return argmax_select(g, rows, with_conf, conf_out)


# ---- S5 Drop B + S6 Drafter-Opt: Fused Markov RNN step + Tuned Fold+Argmax (288x4w) + Direct out_tok ----
_MARKOV_FUSED = os.environ.get("INKLING_MARKOV_FUSED", "0").strip().lower() not in ("0", "", "off", "false", "no")
# 288 slices (6 CTAs/SM across 48 GB10 SMs) x 4 warps cuts fold_argmax from 42.8/14.6 us -> 34.7/10.3 us (0-ULP same FP8!)
_MARKOV_FOLD_SLICES = int(os.environ.get("INKLING_MARKOV_FOLD_SLICES", "288"))
_MARKOV_FOLD_TC_SLICES = int(os.environ.get("INKLING_MARKOV_FOLD_TC_SLICES", "96"))
_MARKOV_FOLD_WARPS = int(os.environ.get("INKLING_MARKOV_FOLD_WARPS", "4"))
_MARKOV_INLINE_AG = os.environ.get("INKLING_MARKOV_INLINE_AG", "1").strip().lower() not in ("0", "", "off", "false", "no")
# INKLING_MARKOV_FUSE_REDUCE=1: fuse _argmax_local_reduce_kernel into the last-arriving CTA of
# _fold_argmax_record_kernel (rows == 1, inline AG) + skip zero-state FillFunctor & half RNN
# weights on Markov step 0 (0-ULP bit-exact to separate kernels; saves ~23 us/step).
_MARKOV_FUSE_REDUCE = os.environ.get("INKLING_MARKOV_FUSE_REDUCE", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# C=32 Batched Markov Optimizations (strictly opt-in, default OFF, rows >= 2 ONLY -> C=1 untouched!):
# - INKLING_MARKOV_RNN_TC=1: Tensor-Core batched _markov_rnn_step_tc_kernel (8 CTAs vs 1024 scalar CTAs @ B=32, 32x less L2!)
# - INKLING_MARKOV_TC_FUSE_REDUCE=1: fuse _argmax_local_reduce_kernel into last-CTA of _fold_argmax_tc_kernel (2 <= rows <= 32)
_MARKOV_RNN_TC = os.environ.get("INKLING_MARKOV_RNN_TC", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_MARKOV_TC_FUSE_REDUCE = os.environ.get("INKLING_MARKOV_TC_FUSE_REDUCE", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)

# Per-device 2x ping-pong self-resetting semaphores for fused _fold_argmax_record_kernel reduce.
_FOLD_SEM_BUFS: dict[int, list[torch.Tensor]] = {}
_FOLD_SEM_IDX: dict[int, int] = {}
_MIN_FOLD_SEM_ROWS = 64


def prealloc_fold_sem(device: torch.device, min_rows: int = _MIN_FOLD_SEM_ROWS) -> None:
    """Pre-allocate zeroed semaphores for fused Markov fold+reduce outside CUDA graph capture."""
    if not (_MARKOV_FUSE_REDUCE or _MARKOV_TC_FUSE_REDUCE) or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    need = max(_MIN_FOLD_SEM_ROWS, int(min_rows))
    sems = _FOLD_SEM_BUFS.get(dev_idx)
    if sems is None or sems[0].numel() < need:
        _FOLD_SEM_BUFS[dev_idx] = [
            torch.zeros(need, dtype=torch.int32, device=device),
            torch.zeros(need, dtype=torch.int32, device=device),
        ]
    if dev_idx not in _FOLD_SEM_IDX:
        _FOLD_SEM_IDX[dev_idx] = 0


def _next_fold_sem(device: torch.device, rows: int, force_enable: bool = False) -> torch.Tensor | None:
    if (not force_enable and not _MARKOV_FUSE_REDUCE) or device.type != "cuda":
        return None
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    sems = _FOLD_SEM_BUFS.get(dev_idx)
    if sems is None or sems[0].numel() < rows:
        if torch.cuda.is_current_stream_capturing():
            return None  # fail closed: never allocate/zero SEM inside graph capture
        prealloc_fold_sem(device, rows)
        sems = _FOLD_SEM_BUFS.get(dev_idx)
        if sems is None:
            return None
    idx = _FOLD_SEM_IDX.get(dev_idx, 0)
    _FOLD_SEM_IDX[dev_idx] = idx ^ 1
    return sems[idx]


@triton.jit
def _argmax_local_reduce_kernel(
    REC_IN, REC_OUT, SKIP_PTR,
    S, NP2: tl.constexpr, HAS_SKIP: tl.constexpr, PDL: tl.constexpr = False,
):
    """Collapse [rows, S, 4] slice records to [rows, 1, 4] (16B/row -> inline RDMA AG!)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            return
    row = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, NP2)
    ok = s < S
    base = (row * S + s) * 4
    v = tl.load(REC_IN + base, ok, other=-float("inf"))
    idx = tl.load(REC_IN + base + 1, ok, other=0.0)
    se = tl.load(REC_IN + base + 2, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), s, NP2), 0)
    best_i = tl.sum(tl.where(s == first, idx, 0.0), 0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    tot_se = tl.sum(tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0)), 0)
    o = tl.arange(0, 4)
    val = tl.where(o == 0, m, tl.where(o == 1, best_i, tl.where(o == 2, tot_se, 0.0)))
    tl.store(REC_OUT + row * 4 + o, val)


@triton.jit
def _markov_rnn_step_kernel(
    STATE, PREV, CTX, W1, RNN_W, OUT_STATE, LATENT, SKIP_PTR,
    prev_stride,
    RANK: tl.constexpr, BC: tl.constexpr, HAS_SKIP: tl.constexpr,
    INIT_ZERO: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    """Fuses markov_w1(prev) + cat([state, emb]) + addmm(ctx, x, rnn_w.T) + markov_gate.
    When INIT_ZERO=True (Markov step 0), state is identically 0: skips STATE & state-weight loads.
    Grid: (R, RANK // BC). Each CTA owns BC output dims of gate, cand, and out."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    row = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    c = cb * BC + tl.arange(0, BC)
    k = tl.arange(0, RANK)
    tok = tl.maximum(tl.load(PREV + row * prev_stride).to(tl.int64), 0)
    emb = tl.load(W1 + tok * RANK + k).to(tl.float32)

    stride_w = 2 * RANK
    if INIT_ZERO:
        wg1 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_g = tl.load(CTX + row * (3 * RANK) + c) + tl.sum(wg1 * emb[None, :], axis=1)

        wc1 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_c = tl.load(CTX + row * (3 * RANK) + RANK + c) + tl.sum(wc1 * emb[None, :], axis=1)

        wo1 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_o = tl.load(CTX + row * (3 * RANK) + 2 * RANK + c) + tl.sum(wo1 * emb[None, :], axis=1)
    else:
        st = tl.load(STATE + row * RANK + k).to(tl.float32)
        # Part 0: gate
        wg0 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + k[None, :]).to(tl.float32)
        acc_g = tl.sum(wg0 * st[None, :], axis=1)
        wg1 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_g = tl.load(CTX + row * (3 * RANK) + c) + acc_g + tl.sum(wg1 * emb[None, :], axis=1)

        # Part 1: cand
        wc0 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + k[None, :]).to(tl.float32)
        acc_c = tl.sum(wc0 * st[None, :], axis=1)
        wc1 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_c = tl.load(CTX + row * (3 * RANK) + RANK + c) + acc_c + tl.sum(wc1 * emb[None, :], axis=1)

        # Part 2: out
        wo0 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + k[None, :]).to(tl.float32)
        acc_o = tl.sum(wo0 * st[None, :], axis=1)
        wo1 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_o = tl.load(CTX + row * (3 * RANK) + 2 * RANK + c) + acc_o + tl.sum(wo1 * emb[None, :], axis=1)

    gate = _round_bf16(raw_g)
    cand = _round_bf16(raw_c)
    outv = _round_bf16(raw_o)
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate)))
    cand = _round_bf16(tldevice.tanh(cand))
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    if INIT_ZERO:
        new_st = update
    else:
        st_c = tl.load(STATE + row * RANK + c).to(tl.float32)
        keep = _round_bf16(gate * st_c)
        new_st = keep + update
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT_STATE + row * RANK + c, new_st.to(tl.bfloat16))
    tl.store(LATENT + row * RANK + c, tldevice.tanh(outv).to(tl.bfloat16))


@triton.jit
def _markov_rnn_step_tc_kernel(
    STATE, PREV, CTX, W1, RNN_W, OUT_STATE, LATENT,
    rows, prev_stride,
    RANK: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
    INIT_ZERO: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    """C=32 Batched Tensor-Core Markov RNN step (rows >= 2, opt-in via INKLING_MARKOV_RNN_TC=1).
    Reads rnn_w_rec [3*RANK, 2*RANK] ONCE per BM rows (16 or 32) instead of 32x per-row scalar
    CTAs (8 CTAs vs 1,024 CTAs @ B=32, 32x less L2 weight traffic!). Preserves exact _round_bf16
    recurrence gate semantics."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pm = tl.program_id(0)
    cb = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM)
    mm = m < rows
    c = cb * BC + tl.arange(0, BC)
    m64 = m.to(tl.int64)
    tok = tl.maximum(tl.load(PREV + m64 * prev_stride, mask=mm, other=0).to(tl.int64), 0)

    stride_w = 2 * RANK
    acc_g = tl.zeros((BM, BC), dtype=tl.float32)
    acc_c = tl.zeros((BM, BC), dtype=tl.float32)
    acc_o = tl.zeros((BM, BC), dtype=tl.float32)

    for k0 in tl.static_range(0, RANK, BK):
        kk = k0 + tl.arange(0, BK)
        emb = tl.load(W1 + tok[:, None] * RANK + kk[None, :], mask=mm[:, None], other=0.0)
        wg1 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + (RANK + kk)[None, :])
        acc_g = tl.dot(emb, tl.trans(wg1), acc_g)
        wc1 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + kk)[None, :])
        acc_c = tl.dot(emb, tl.trans(wc1), acc_c)
        wo1 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + kk)[None, :])
        acc_o = tl.dot(emb, tl.trans(wo1), acc_o)
        if not INIT_ZERO:
            st = tl.load(STATE + m64[:, None] * RANK + kk[None, :], mask=mm[:, None], other=0.0)
            wg0 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + kk[None, :])
            acc_g = tl.dot(st, tl.trans(wg0), acc_g)
            wc0 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + kk[None, :])
            acc_c = tl.dot(st, tl.trans(wc0), acc_c)
            wo0 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + kk[None, :])
            acc_o = tl.dot(st, tl.trans(wo0), acc_o)

    ctx_base = m64[:, None] * (3 * RANK)
    raw_g = tl.load(CTX + ctx_base + c[None, :], mask=mm[:, None], other=0.0) + acc_g
    raw_c = tl.load(CTX + ctx_base + (RANK + c)[None, :], mask=mm[:, None], other=0.0) + acc_c
    raw_o = tl.load(CTX + ctx_base + (2 * RANK + c)[None, :], mask=mm[:, None], other=0.0) + acc_o

    gate = _round_bf16(raw_g)
    cand = _round_bf16(raw_c)
    outv = _round_bf16(raw_o)
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate)))
    cand = _round_bf16(tldevice.tanh(cand))
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    if INIT_ZERO:
        new_st = update
    else:
        st_c = tl.load(STATE + m64[:, None] * RANK + c[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        keep = _round_bf16(gate * st_c)
        new_st = keep + update
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT_STATE + m64[:, None] * RANK + c[None, :], new_st.to(tl.bfloat16), mask=mm[:, None])
    tl.store(LATENT + m64[:, None] * RANK + c[None, :], tldevice.tanh(outv).to(tl.bfloat16), mask=mm[:, None])


def markov_rnn_step_fused(
    state: "torch.Tensor | None",
    prev_tokens: torch.Tensor,
    context: torch.Tensor,
    w1_weight: torch.Tensor,
    rnn_w_rec: torch.Tensor,
    skip_dev: "torch.Tensor | None" = None,
    pdl: bool | None = None,
    init_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state is None:
        init_zero = True
        rows = int(context.shape[0])
        rank = int(rnn_w_rec.shape[0] // 3)
        out_state = torch.empty((rows, rank), dtype=w1_weight.dtype, device=context.device)
        latent = torch.empty((rows, rank), dtype=w1_weight.dtype, device=context.device)
        state_arg = out_state
    else:
        rows, rank = state.shape
        out_state = torch.empty_like(state)
        latent = torch.empty_like(state)
        state_arg = state
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if rows:
        if _MARKOV_RNN_TC and rows >= 2 and skip_dev is None and (rank % 64 == 0):
            bm = 16 if rows <= 16 else 32
            bc = 32
            _markov_rnn_step_tc_kernel[(triton.cdiv(rows, bm), rank // bc)](
                state_arg, prev_tokens, context, w1_weight, rnn_w_rec, out_state, latent,
                rows, prev_tokens.stride(0),
                RANK=rank, BM=bm, BC=bc, BK=64, INIT_ZERO=bool(init_zero),
                PDL=use_pdl,
                num_warps=4, launch_pdl=use_pdl,
            )
        else:
            bc = 8
            _markov_rnn_step_kernel[(rows, rank // bc)](
                state_arg, prev_tokens, context, w1_weight, rnn_w_rec, out_state, latent,
                skip_dev if skip_dev is not None else prev_tokens,
                prev_tokens.stride(0),
                RANK=rank, BC=bc, HAS_SKIP=skip_dev is not None, INIT_ZERO=bool(init_zero),
                PDL=use_pdl,
                num_warps=4, launch_pdl=use_pdl,
            )
    return out_state, latent


@triton.jit
def _fold_argmax_record_kernel(
    BASE, LAT, FW, FS, FB, REC, REC_OUT, SEM, SKIP_PTR, ID_MAP,
    stride_base, softcap, valid, offset, CH,
    RANK: tl.constexpr, BC: tl.constexpr, USE_FP8: tl.constexpr, HAS_SKIP: tl.constexpr,
    FUSE_REDUCE: tl.constexpr = False, NP2_S: tl.constexpr = 512,
    PDL: tl.constexpr = False,
    HAS_ID_MAP: tl.constexpr = False,
):
    """Fuses fold_bias + latent @ fold_weight.T + base_logits + softcap + argmax_record
    (and optionally last-CTA local reduction across all S slices into REC_OUT[row, 4]).
    Zero intermediate [R, V] bias buffer; supports bf16 or per-row e4m3 fold_weight.
    Pins FW in L2 across steps 0..6 via eviction_policy='evict_last'."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    HALF: tl.constexpr = RANK // 2
    k0 = tl.arange(0, HALF)
    lat0 = tl.load(LAT + row * RANK + k0).to(tl.float32)
    lat1 = tl.load(LAT + row * RANK + HALF + k0).to(tl.float32)

    lo = s * CH
    hi = tl.minimum(lo + CH, valid)
    m_best = -float("inf")
    idx_best = 2147483647
    se_best = 0.0

    for c0 in range(lo, hi, BC):
        cols = c0 + tl.arange(0, BC)
        cmask = cols < hi
        c64 = cols.to(tl.int64)[:, None] * RANK
        fw0 = tl.load(FW + c64 + k0[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = tl.sum(fw0 * lat0[None, :], axis=1)
        fw1 = tl.load(FW + c64 + (HALF + k0)[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = dot + tl.sum(fw1 * lat1[None, :], axis=1)
        if USE_FP8:
            dot = dot * tl.load(FS + cols, mask=cmask, other=0.0)
        fb = tl.load(FB + cols, mask=cmask, other=0.0).to(tl.float32)
        b = (dot + fb).to(tl.bfloat16).to(tl.float32)
        a = tl.load(BASE + row * stride_base + cols, mask=cmask, other=0).to(tl.float32)
        x = (a + b).to(tl.bfloat16)
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
        x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.where(cmask & (x == x), x, -float("inf"))

        cm = tl.max(x, 0)
        cidx = tl.min(tl.where(x == cm, cols, 2147483647), 0)
        m_new = tl.maximum(m_best, cm)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_safe))
        sc_old = tl.where(m_best == -float("inf"), 0.0, tl.exp(m_best - m_safe))
        se_best = se_best * sc_old + tl.sum(e, 0)
        idx_best = tl.where(cm > m_best, cidx, tl.where(cm == m_best, tl.minimum(idx_best, cidx), idx_best))
        m_best = m_new

    o = tl.arange(0, 4)
    if HAS_ID_MAP:
        safe_idx = tl.where(idx_best < valid, idx_best, 0)
        out_id = tl.load(ID_MAP + safe_idx).to(tl.float32)
    else:
        out_id = (idx_best + offset).to(tl.float32)
    val = tl.where(
        o == 0, m_best, tl.where(o == 1, out_id, tl.where(o == 2, se_best, 0.0))
    )
    if not FUSE_REDUCE:
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(REC + (row * S + s) * 4 + o, val)
    else:
        tl.store(REC + (row * S + s) * 4 + o, val)
        tl.debug_barrier()
        if tl.atomic_add(SEM + row, 1, sem="acq_rel") == S - 1:
            tl.store(SEM + row, 0)
            ss = tl.arange(0, NP2_S)
            ok = ss < S
            rbase = (row * S + ss) * 4
            rv = tl.load(REC + rbase, ok, other=-float("inf"), cache_modifier=".cg")
            ridx = tl.load(REC + rbase + 1, ok, other=0.0, cache_modifier=".cg")
            rse = tl.load(REC + rbase + 2, ok, other=0.0, cache_modifier=".cg")
            rm = tl.max(rv, 0)
            rfirst = tl.min(tl.where(ok & (rv == rm), ss, NP2_S), 0)
            rbest_i = tl.sum(tl.where(ss == rfirst, ridx, 0.0), 0)
            rm0 = tl.where(rm == -float("inf"), 0.0, rm)
            rtot_se = tl.sum(tl.where(rv == -float("inf"), 0.0, rse * tl.exp(rv - rm0)), 0)
            rval = tl.where(o == 0, rm, tl.where(o == 1, rbest_i, tl.where(o == 2, rtot_se, 0.0)))
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(REC_OUT + row * 4 + o, rval)


@triton.jit
def _fold_argmax_top2_record_kernel(
    BASE, LAT, FW, FS, FB, REC, REC_OUT, SEM, SKIP_PTR,
    stride_base, softcap, valid, offset, CH,
    RANK: tl.constexpr, BC: tl.constexpr, USE_FP8: tl.constexpr, HAS_SKIP: tl.constexpr,
    FUSE_REDUCE: tl.constexpr = False, NP2_S: tl.constexpr = 512,
    PDL: tl.constexpr = False,
):
    """Zero-extra-DRAM-read top-2 upgrade of `_fold_argmax_record_kernel` for Tree Speculation.
    Tracks (m1, idx1, se, m2, idx2) in the exact same single L2-pinned pass over fold_weight [V, 256].
    Writes 8 fp32 values/row (32 B/row inline RDMA AG) returning exact (tok1, p1, tok2, p2)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    HALF: tl.constexpr = RANK // 2
    k0 = tl.arange(0, HALF)
    lat0 = tl.load(LAT + row * RANK + k0).to(tl.float32)
    lat1 = tl.load(LAT + row * RANK + HALF + k0).to(tl.float32)

    lo = s * CH
    hi = tl.minimum(lo + CH, valid)
    m1_best = -float("inf")
    i1_best = 2147483647
    m2_best = -float("inf")
    i2_best = 2147483647
    se_best = 0.0

    for c0 in range(lo, hi, BC):
        cols = c0 + tl.arange(0, BC)
        cmask = cols < hi
        c64 = cols.to(tl.int64)[:, None] * RANK
        fw0 = tl.load(FW + c64 + k0[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = tl.sum(fw0 * lat0[None, :], axis=1)
        fw1 = tl.load(FW + c64 + (HALF + k0)[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = dot + tl.sum(fw1 * lat1[None, :], axis=1)
        if USE_FP8:
            dot = dot * tl.load(FS + cols, mask=cmask, other=0.0)
        fb = tl.load(FB + cols, mask=cmask, other=0.0).to(tl.float32)
        b = (dot + fb).to(tl.bfloat16).to(tl.float32)
        a = tl.load(BASE + row * stride_base + cols, mask=cmask, other=0).to(tl.float32)
        x = (a + b).to(tl.bfloat16)
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
        x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.where(cmask & (x == x), x, -float("inf"))

        cm1 = tl.max(x, 0)
        ci1 = tl.min(tl.where(x == cm1, cols, 2147483647), 0)
        x_no1 = tl.where(cols == ci1, -float("inf"), x)
        cm2 = tl.max(x_no1, 0)
        ci2 = tl.min(tl.where(x_no1 == cm2, cols, 2147483647), 0)

        m_new = tl.maximum(m1_best, cm1)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_safe))
        sc_old = tl.where(m1_best == -float("inf"), 0.0, tl.exp(m1_best - m_safe))
        se_best = se_best * sc_old + tl.sum(e, 0)

        beat1 = (cm1 > m1_best) | ((cm1 == m1_best) & (ci1 < i1_best))
        cand2_m = tl.where(beat1, tl.maximum(m1_best, cm2), tl.maximum(m2_best, cm1))
        cand2_i = tl.where(
            beat1,
            tl.where(cm2 > m1_best, ci2, tl.where(cm2 == m1_best, tl.minimum(i1_best, ci2), i1_best)),
            tl.where(cm1 > m2_best, ci1, tl.where(cm1 == m2_best, tl.minimum(i2_best, ci1), i2_best)),
        )
        i1_best = tl.where(beat1, ci1, i1_best)
        m1_best = tl.where(beat1, cm1, m1_best)
        m2_best = cand2_m
        i2_best = cand2_i

    o = tl.arange(0, 8)
    val = tl.where(
        o == 0, m1_best,
        tl.where(
            o == 1, (i1_best + offset).to(tl.float32),
            tl.where(
                o == 2, se_best,
                tl.where(
                    o == 3, m2_best,
                    tl.where(o == 4, (i2_best + offset).to(tl.float32), 0.0),
                ),
            ),
        ),
    )
    tl.store(REC + (row * S + s) * 8 + o, val)
    if FUSE_REDUCE:
        tl.debug_barrier()
        if tl.atomic_add(SEM + row, 1, sem="acq_rel") == S - 1:
            tl.store(SEM + row, 0)
            ss = tl.arange(0, NP2_S)
            ok = ss < S
            rb = (row * S + ss) * 8
            v1 = tl.load(REC + rb + 0, ok, other=-float("inf"), cache_modifier=".cg")
            k1 = tl.load(REC + rb + 1, ok, other=0.0, cache_modifier=".cg")
            se = tl.load(REC + rb + 2, ok, other=0.0, cache_modifier=".cg")
            v2 = tl.load(REC + rb + 3, ok, other=-float("inf"), cache_modifier=".cg")
            k2 = tl.load(REC + rb + 4, ok, other=0.0, cache_modifier=".cg")
            gm1 = tl.max(v1, 0)
            w1 = tl.min(tl.where(ok & (v1 == gm1), ss, NP2_S), 0)
            gk1 = tl.sum(tl.where(ss == w1, k1, 0.0), 0)
            v_pool = tl.where(ss == w1, v2, v1)
            k_pool = tl.where(ss == w1, k2, k1)
            gm2 = tl.max(v_pool, 0)
            w2 = tl.min(tl.where(ok & (v_pool == gm2), ss, NP2_S), 0)
            gk2 = tl.sum(tl.where(ss == w2, k_pool, 0.0), 0)
            gm0 = tl.where(gm1 == -float("inf"), 0.0, gm1)
            gtot = tl.sum(tl.where(v1 == -float("inf"), 0.0, se * tl.exp(v1 - gm0)), 0)
            rval = tl.where(
                o == 0, gm1,
                tl.where(
                    o == 1, gk1,
                    tl.where(o == 2, gtot, tl.where(o == 3, gm2, tl.where(o == 4, gk2, 0.0))),
                ),
            )
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(REC_OUT + row * 8 + o, rval)


@triton.jit
def _es_init_kernel(
    IDX_PTR, TEMP_PTR, STEP_PTR,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTV_PTR, GREEDY_PTR, SLOT_PTR,
    DUMMY: tl.constexpr,
):
    slot = tl.load(IDX_PTR).to(tl.int64)
    temp = tl.load(TEMP_PTR + slot)
    s0 = tl.load(STEP_PTR)
    tl.store(SKIP_PTR, 0)
    tl.store(SURV_PTR, 1.0)
    tl.store(TOK_PTR, 1.0)
    tl.store(BESTV_PTR, 1.0 / s0)
    greedy = 0 if DUMMY else tl.where(temp == 0.0, 1, 0)
    tl.store(GREEDY_PTR, greedy)
    tl.store(SLOT_PTR, slot.to(tl.int32))


@triton.jit
def _argmax_select_es_kernel(
    G, OUT, CONF,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTV_PTR, GREEDY_PTR, SLOT_PTR,
    EMA_PTR, CAL_PTR, STEP_PTR,
    S, NS,
    STEP_IDX: tl.constexpr, K: tl.constexpr, NP2: tl.constexpr,
    USE_HIST: tl.constexpr, PRIOR_LOGIT: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Single-request (R=1) argmax_select with exact replicated GPU-side early-stop decision."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if STEP_IDX > 0:
        if tl.load(SKIP_PTR) != 0:
            tl.store(OUT, -1)
            tl.store(CONF, 0.0)
            return
    r = tl.arange(0, NP2)
    ok = r < NS
    base = r.to(tl.int64) * 4
    v = tl.load(G + base, ok, other=-float("inf"))
    i = tl.load(G + base + 1, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), r, NP2), 0)
    tok = tl.sum(tl.where(r == first, i, 0.0), 0).to(tl.int64)
    tl.store(OUT, tok)

    se = tl.load(G + base + 2, ok, other=0.0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    w = tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0))
    tot = tl.sum(w, 0)
    p = tl.where(tot > 0, 1.0 / tot, 0.0)
    tl.store(CONF, p)

    if STEP_IDX < K - 1:
        if tl.load(GREEDY_PTR) != 0:
            pc = tl.minimum(tl.maximum(p, 1e-6), 1.0 - 1e-6)
            a = tl.load(CAL_PTR + STEP_IDX)
            b = tl.load(CAL_PTR + 8 + STEP_IDX)
            beta = tl.load(CAL_PTR + 16)
            tau = tl.load(CAL_PTR + 17)
            z = a * (tl.log(pc) - tl.log(1.0 - pc)) + b
            if USE_HIST:
                slot = tl.load(SLOT_PTR).to(tl.int64)
                h = tl.load(EMA_PTR + slot * K + STEP_IDX)
                hc = tl.minimum(tl.maximum(h, 1e-4), 1.0 - 1e-4)
                z = z + beta * (tl.log(hc) - tl.log(1.0 - hc) - PRIOR_LOGIT)
            c = 1.0 / (1.0 + tl.exp(-z))
            surv = tl.load(SURV_PTR) * c
            tl.store(SURV_PTR, surv)
            if tau > 0.0:
                if surv < tau:
                    tl.store(SKIP_PTR, 1)
            else:
                tokens = tl.load(TOK_PTR) + surv
                tl.store(TOK_PTR, tokens)
                v_cur = tokens / tl.load(STEP_PTR + STEP_IDX + 1)
                best_v = tl.maximum(tl.load(BESTV_PTR), v_cur)
                tl.store(BESTV_PTR, best_v)
                can_beat = 0
                for ww in tl.static_range(STEP_IDX + 2, K + 1):
                    v_ub = (tokens + (ww - STEP_IDX - 1) * surv) / tl.load(STEP_PTR + ww)
                    can_beat = tl.where(v_ub > best_v, 1, can_beat)
                if can_beat == 0:
                    tl.store(SKIP_PTR, 1)


@triton.jit
def _es_init_dink_kernel(
    IDX_PTR, TEMP_PTR, STEP_PTR, CAL_PTR, DB_PTR, ETOK_PTR, EMS_PTR,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTU_PTR, GREEDY_PTR, SLOT_PTR, DB_DEV_PTR, LAM_DEV_PTR,
    DUMMY: tl.constexpr,
):
    """Initialize coupled Dinkelbach early-stop GPU state for R=1 before draft graph replay."""
    slot = tl.load(IDX_PTR).to(tl.int64)
    temp = tl.load(TEMP_PTR + slot)
    s0 = tl.load(STEP_PTR)
    static_lam = tl.load(CAL_PTR + 18)
    gamma = tl.load(CAL_PTR + 19)
    db = tl.load(DB_PTR + slot)
    etok = tl.load(ETOK_PTR + slot)
    ems = tl.maximum(tl.load(EMS_PTR + slot), 1.0)
    lam = tl.where(static_lam > 0.0, static_lam, gamma * (etok / ems))
    tl.store(SKIP_PTR, 0)
    tl.store(SURV_PTR, 1.0)
    tl.store(TOK_PTR, 1.0)
    tl.store(BESTU_PTR, 1.0 - lam * s0)
    greedy = 0 if DUMMY else tl.where(temp == 0.0, 1, 0)
    tl.store(GREEDY_PTR, greedy)
    tl.store(SLOT_PTR, slot.to(tl.int32))
    tl.store(DB_DEV_PTR, db)
    tl.store(LAM_DEV_PTR, lam)


@triton.jit
def _argmax_select_es_dink_kernel(
    G, OUT, CONF,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTU_PTR, GREEDY_PTR, DB_DEV_PTR, LAM_DEV_PTR,
    CAL_PTR, STEP_PTR,
    S, NS,
    STEP_IDX: tl.constexpr, K: tl.constexpr, NP2: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Single-request (R=1) argmax_select with exact lossless coupled Dinkelbach early-stop."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if STEP_IDX > 0:
        if tl.load(SKIP_PTR) != 0:
            tl.store(OUT, -1)
            tl.store(CONF, 0.0)
            return
    r = tl.arange(0, NP2)
    ok = r < NS
    base = r.to(tl.int64) * 4
    v = tl.load(G + base, ok, other=-float("inf"))
    i = tl.load(G + base + 1, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), r, NP2), 0)
    tok = tl.sum(tl.where(r == first, i, 0.0), 0).to(tl.int64)
    tl.store(OUT, tok)

    se = tl.load(G + base + 2, ok, other=0.0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    w = tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0))
    tot = tl.sum(w, 0)
    p = tl.where(tot > 0, 1.0 / tot, 0.0)
    tl.store(CONF, p)

    if STEP_IDX < K - 1:
        if tl.load(GREEDY_PTR) != 0:
            pc = tl.minimum(tl.maximum(p, 1e-6), 1.0 - 1e-6)
            a = tl.load(CAL_PTR + STEP_IDX)
            b = tl.load(CAL_PTR + 8 + STEP_IDX)
            mk_ms = tl.load(CAL_PTR + 20)
            db = tl.load(DB_DEV_PTR)
            lam = tl.load(LAM_DEV_PTR)
            z = a * (tl.log(pc) - tl.log(1.0 - pc)) + b + db
            c = 1.0 / (1.0 + tl.exp(-z))
            surv = tl.load(SURV_PTR) * c
            tokens = tl.load(TOK_PTR) + surv
            tl.store(SURV_PTR, surv)
            tl.store(TOK_PTR, tokens)
            u_cur = tokens - lam * tl.load(STEP_PTR + STEP_IDX + 1)
            best_u = tl.maximum(tl.load(BESTU_PTR), u_cur)
            tl.store(BESTU_PTR, best_u)
            can_beat = 0
            thresh = best_u + lam * mk_ms
            for ww in tl.static_range(STEP_IDX + 2, K + 1):
                u_ub = (tokens + (ww - STEP_IDX - 1) * surv) - lam * tl.load(STEP_PTR + ww)
                can_beat = tl.where(u_ub > thresh, 1, can_beat)
            if can_beat == 0:
                tl.store(SKIP_PTR, 1)


def es_init(idx, temp, step_ms, state_bufs, dummy: bool):
    """Initialize early-stop GPU state for R=1 before draft graph replay."""
    skip_dev, surv_dev, tok_dev, bestv_dev, greedy_dev, slot_dev = state_bufs[:6]
    _es_init_kernel[(1,)](
        idx, temp, step_ms,
        skip_dev, surv_dev, tok_dev, bestv_dev, greedy_dev, slot_dev,
        DUMMY=dummy, num_warps=1,
    )


def es_init_dink(idx, temp, step_eff, cal_ptr, db_ptr, etok_ptr, ems_ptr, state_bufs, dummy: bool):
    """Initialize coupled Dinkelbach early-stop GPU state for R=1 before draft graph replay."""
    skip_d, surv_d, tok_d, bestu_d, greedy_d, slot_d, db_d, lam_d = state_bufs
    _es_init_dink_kernel[(1,)](
        idx, temp, step_eff, cal_ptr, db_ptr, etok_ptr, ems_ptr,
        skip_d, surv_d, tok_d, bestu_d, greedy_d, slot_d, db_d, lam_d,
        DUMMY=dummy, num_warps=1,
    )


@triton.jit
def _fold_argmax_tc_kernel(
    BASE, LAT, FW, FS, FB, REC, REC_OUT, SEM, ID_MAP,
    stride_base, rows, softcap, valid, offset, CH,
    RANK: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, USE_FP8: tl.constexpr,
    FUSE_REDUCE: tl.constexpr = False, NP2_S: tl.constexpr = 128,
    PDL: tl.constexpr = False,
    HAS_ID_MAP: tl.constexpr = False,
):
    """High-concurrency (rows >= 2) batched Tensor-Core fold + softcap + argmax_record.
    Reads fold_weight [V, RANK] ONCE per BM rows (16 or 32) instead of per-row scalar loops
    (3.7x faster at B=32: 2.22 ms -> 0.60 ms across 7 Markov steps)."""
    pm = tl.program_id(0)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    m = pm * BM + tl.arange(0, BM)
    mm = m < rows
    k = tl.arange(0, RANK)
    if PDL:
        tl.extra.cuda.gdc_wait()
    lat = tl.load(LAT + m.to(tl.int64)[:, None] * RANK + k[None, :], mask=mm[:, None], other=0.0)

    lo = s * CH
    hi = tl.minimum(lo + CH, valid)
    m_best = tl.full((BM,), -float("inf"), tl.float32)
    idx_best = tl.full((BM,), 2147483647, tl.int32)
    se_best = tl.zeros((BM,), tl.float32)

    for c0 in range(lo, hi, BC):
        cols = c0 + tl.arange(0, BC)
        cmask = cols < hi
        c64 = cols.to(tl.int64)[:, None] * RANK
        fw = tl.load(FW + c64 + k[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last")
        if USE_FP8:
            fw = fw.to(tl.bfloat16)
        dot = tl.dot(lat, tl.trans(fw))
        if USE_FP8:
            fs = tl.load(FS + cols, mask=cmask, other=0.0)
            dot = dot * fs[None, :]
        fb = tl.load(FB + cols, mask=cmask, other=0.0).to(tl.float32)
        b = (dot + fb[None, :]).to(tl.bfloat16).to(tl.float32)
        a = tl.load(
            BASE + m.to(tl.int64)[:, None] * stride_base + cols[None, :],
            mask=mm[:, None] & cmask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = (a + b).to(tl.bfloat16)
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
        x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.where(mm[:, None] & cmask[None, :] & (x == x), x, -float("inf"))

        cm = tl.max(x, 1)
        cidx = tl.min(tl.where(x == cm[:, None], cols[None, :], 2147483647), 1)
        m_new = tl.maximum(m_best, cm)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_safe[:, None]))
        sc_old = tl.where(m_best == -float("inf"), 0.0, tl.exp(m_best - m_safe))
        se_best = se_best * sc_old + tl.sum(e, 1)
        idx_best = tl.where(cm > m_best, cidx, tl.where(cm == m_best, tl.minimum(idx_best, cidx), idx_best))
        m_best = m_new

    rec_base = (m.to(tl.int64) * S + s) * 4
    if HAS_ID_MAP:
        safe_idx = tl.where(idx_best < valid, idx_best, 0)
        out_id = tl.load(ID_MAP + safe_idx, mask=mm, other=0).to(tl.float32)
    else:
        out_id = (idx_best + offset).to(tl.float32)
    if not FUSE_REDUCE:
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(REC + rec_base + 0, m_best, mask=mm)
        tl.store(REC + rec_base + 1, out_id, mask=mm)
        tl.store(REC + rec_base + 2, se_best, mask=mm)
        tl.store(REC + rec_base + 3, 0.0, mask=mm)
    else:
        tl.store(REC + rec_base + 0, m_best, mask=mm)
        tl.store(REC + rec_base + 1, out_id, mask=mm)
        tl.store(REC + rec_base + 2, se_best, mask=mm)
        tl.store(REC + rec_base + 3, 0.0, mask=mm)
        tl.debug_barrier()
        if tl.atomic_add(SEM, 1, sem="acq_rel") == S - 1:
            tl.store(SEM, 0)
            ss = tl.arange(0, NP2_S)
            ok_s = ss < S
            o4 = tl.arange(0, 4)
            for r_i in range(0, rows):
                rbase = (r_i * S + ss).to(tl.int64) * 4
                rv = tl.load(REC + rbase, ok_s, other=-float("inf"), cache_modifier=".cg")
                ridx = tl.load(REC + rbase + 1, ok_s, other=0.0, cache_modifier=".cg")
                rse = tl.load(REC + rbase + 2, ok_s, other=0.0, cache_modifier=".cg")
                rm = tl.max(rv, 0)
                rfirst = tl.min(tl.where(ok_s & (rv == rm), ss, NP2_S), 0)
                rbest_i = tl.sum(tl.where(ss == rfirst, ridx, 0.0), 0)
                rm0 = tl.where(rm == -float("inf"), 0.0, rm)
                rtot_se = tl.sum(tl.where(rv == -float("inf"), 0.0, rse * tl.exp(rv - rm0)), 0)
                rval = tl.where(o4 == 0, rm, tl.where(o4 == 1, rbest_i, tl.where(o4 == 2, rtot_se, 0.0)))
                tl.store(REC_OUT + (r_i * 4 + o4).to(tl.int64), rval)
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()


def gathered_fold_argmax_fused(
    base, latent, fold_w, fold_b, softcap, valid, offset,
    fold_w8=None, fold_s8=None,
    conf_out=None,
    es_ctx=None,  # (step_idx, K, state_bufs, ema_ptr, cal_ptr, step_ptr, use_hist, prior_logit)
    out_tok=None,
    pdl: bool | None = None,
    id_map: "torch.Tensor | None" = None,
):
    """Fused fold + softcap + argmax_record + (conditional) RDMA AG + (early-stop) argmax_select."""
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather

    rows = base.shape[0]
    rank = latent.shape[1]
    base_slices = _MARKOV_FOLD_TC_SLICES if rows >= 2 else _MARKOV_FOLD_SLICES
    # Dynamically clamp slices for pruned vocabularies so we never launch empty CTAs or pad NP2=512.
    slices = min(base_slices, max(48, triton.cdiv(valid, 32)))
    ch = triton.cdiv(triton.cdiv(valid, slices), 8) * 8
    rec = torch.empty((rows, slices, 4), dtype=torch.float32, device=base.device)
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)

    skip_dev = None
    if es_ctx is not None and rows == 1:
        step_idx = es_ctx[0]
        if step_idx > 0:
            skip_dev = es_ctx[2][0]

    use_fp8 = fold_w8 is not None and fold_s8 is not None
    has_id_map = id_map is not None
    id_map_arg = id_map if has_id_map else rec
    fused_local = False
    rec1 = None
    if rows >= 2:
        bm = 16 if rows <= 16 else 32
        sem = (
            _next_fold_sem(base.device, 1, force_enable=True)
            if (_MARKOV_TC_FUSE_REDUCE and _MARKOV_INLINE_AG and slices > 1 and rows <= bm)
            else None
        )
        fused_local = sem is not None
        np2_s = triton.next_power_of_2(slices) if fused_local else 128
        if fused_local:
            rec1 = torch.empty((rows, 1, 4), dtype=torch.float32, device=base.device)
        _fold_argmax_tc_kernel[(triton.cdiv(rows, bm), slices)](
            base, latent,
            fold_w8 if use_fp8 else fold_w,
            fold_s8 if use_fp8 else fold_b,
            fold_b, rec,
            rec1 if fused_local else rec,
            sem if fused_local else rec,
            id_map_arg,
            base.stride(0), rows, softcap, valid, offset, ch,
            RANK=rank, BM=bm, BC=32, USE_FP8=use_fp8,
            FUSE_REDUCE=fused_local, NP2_S=np2_s, PDL=use_pdl,
            HAS_ID_MAP=has_id_map,
            num_warps=4, launch_pdl=use_pdl,
        )
    else:
        sem = _next_fold_sem(base.device, rows) if (_MARKOV_INLINE_AG and slices > 1) else None
        fused_local = sem is not None
        np2_s = triton.next_power_of_2(slices) if fused_local else 512
        if fused_local:
            rec1 = torch.empty((rows, 1, 4), dtype=torch.float32, device=base.device)
        _fold_argmax_record_kernel[(rows, slices)](
            base, latent,
            fold_w8 if use_fp8 else fold_w,
            fold_s8 if use_fp8 else fold_b,
            fold_b, rec,
            rec1 if fused_local else rec,
            sem if fused_local else rec,
            skip_dev if skip_dev is not None else rec,
            id_map_arg,
            base.stride(0), softcap, valid, offset, ch,
            RANK=rank, BC=32, USE_FP8=use_fp8, HAS_SKIP=skip_dev is not None,
            FUSE_REDUCE=fused_local, NP2_S=np2_s, PDL=use_pdl,
            HAS_ID_MAP=has_id_map,
            num_warps=_MARKOV_FOLD_WARPS, launch_pdl=use_pdl,
        )

    if fused_local:
        ag_in = rec1
        ag_slices = 1
    elif _MARKOV_INLINE_AG and slices > 1:
        rec1 = torch.empty((rows, 1, 4), dtype=torch.float32, device=base.device)
        np2_s = triton.next_power_of_2(slices)
        _argmax_local_reduce_kernel[(rows,)](
            rec, rec1, skip_dev if skip_dev is not None else rec1,
            slices, NP2=np2_s, HAS_SKIP=skip_dev is not None, PDL=use_pdl,
            num_warps=4 if np2_s >= 128 else 1, launch_pdl=use_pdl,
        )
        ag_in = rec1
        ag_slices = 1
    else:
        ag_in = rec
        ag_slices = slices

    comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
    if skip_dev is not None and comm is not None and comm.should_gather(ag_in):
        g = comm.all_gather_stacked_cond(ag_in, skip_dev).flatten(0, 1)  # [world * rows, ag_slices, 4]
    else:
        g = tensor_model_parallel_all_gather(ag_in, dim=0)

    if es_ctx is not None and rows == 1 and conf_out is not None:
        ws = g.shape[0]
        np2 = triton.next_power_of_2(ws * ag_slices)
        out = out_tok if (out_tok is not None and out_tok.shape[0] >= 1) else torch.empty(1, dtype=torch.int64, device=g.device)
        if len(es_ctx) == 9 and es_ctx[8] == "dink":
            step_idx, K, state_bufs, _ema_ptr, cal_ptr, step_ptr, _use_hist, _prior_logit, _mode = es_ctx
            skip_d, surv_d, tok_d, bestu_d, greedy_d, _slot_d, db_d, lam_d = state_bufs
            _argmax_select_es_dink_kernel[(1,)](
                g, out, conf_out,
                skip_d, surv_d, tok_d, bestu_d, greedy_d, db_d, lam_d,
                cal_ptr, step_ptr,
                ag_slices, ws * ag_slices,
                STEP_IDX=step_idx, K=K, NP2=np2, PDL=use_pdl,
                num_warps=4 if np2 >= 512 else 1, launch_pdl=use_pdl,
            )
            return out
        step_idx, K, state_bufs, ema_ptr, cal_ptr, step_ptr, use_hist, prior_logit = es_ctx[:8]
        skip_d, surv_d, tok_d, bestv_d, greedy_d, slot_d = state_bufs[:6]
        _argmax_select_es_kernel[(1,)](
            g, out, conf_out,
            skip_d, surv_d, tok_d, bestv_d, greedy_d, slot_d,
            ema_ptr, cal_ptr, step_ptr,
            ag_slices, ws * ag_slices,
            STEP_IDX=step_idx, K=K, NP2=np2,
            USE_HIST=use_hist, PRIOR_LOGIT=prior_logit, PDL=use_pdl,
            num_warps=4 if np2 >= 512 else 1, launch_pdl=use_pdl,
        )
        return out
    return argmax_select(g, rows, with_conf=False, conf_out=conf_out, out_tok=out_tok, pdl=use_pdl)


@triton.jit
def _argmax_select_top2_kernel(
    G, TOK1_OUT, P1_OUT, TOK2_OUT, P2_OUT,
    NS, NP2: tl.constexpr, PDL: tl.constexpr = False,
):
    """Reduce TP-gathered [TP, rows, 8] fp32 records -> exact global (tok1, p1, tok2, p2)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, NP2)
    ok = r < NS
    base = (r * tl.num_programs(0) + row) * 8
    v1 = tl.load(G + base + 0, ok, other=-float("inf"))
    k1 = tl.load(G + base + 1, ok, other=0.0)
    se = tl.load(G + base + 2, ok, other=0.0)
    v2 = tl.load(G + base + 3, ok, other=-float("inf"))
    k2 = tl.load(G + base + 4, ok, other=0.0)

    m1 = tl.max(v1, 0)
    w1 = tl.min(tl.where(ok & (v1 == m1), r, NP2), 0)
    t1 = tl.sum(tl.where(r == w1, k1, 0.0), 0).to(tl.int64)

    v_pool = tl.where(r == w1, v2, v1)
    k_pool = tl.where(r == w1, k2, k1)
    m2 = tl.max(v_pool, 0)
    w2 = tl.min(tl.where(ok & (v_pool == m2), r, NP2), 0)
    t2 = tl.sum(tl.where(r == w2, k_pool, 0.0), 0).to(tl.int64)

    m0 = tl.where(m1 == -float("inf"), 0.0, m1)
    tot = tl.sum(tl.where(v1 == -float("inf"), 0.0, se * tl.exp(v1 - m0)), 0)
    p1 = tl.where(tot > 0.0, 1.0 / tot, 0.0)
    p2 = tl.where((tot > 0.0) & (m2 > -float("inf")), tl.exp(m2 - m0) / tot, 0.0)

    tl.store(TOK1_OUT + row, t1)
    tl.store(P1_OUT + row, p1)
    tl.store(TOK2_OUT + row, t2)
    tl.store(P2_OUT + row, p2)


def gathered_fold_argmax_top2_fused(
    base, latent, fold_w, fold_b, softcap, valid, offset,
    fold_w8=None, fold_s8=None, pdl: bool | None = None,
):
    """Single-pass top-2 Markov fold + 32 B/row inline RDMA AG returning (tok1, p1, tok2, p2)."""
    from vllm.distributed import tensor_model_parallel_all_gather

    rows, rank = int(base.shape[0]), int(latent.shape[1])
    slices = _MARKOV_FOLD_SLICES
    ch = triton.cdiv(triton.cdiv(valid, slices), 8) * 8
    rec = torch.empty((rows, slices, 8), dtype=torch.float32, device=base.device)
    rec1 = torch.empty((rows, 1, 8), dtype=torch.float32, device=base.device)
    sem = _next_fold_sem(base.device, rows)
    use_fp8 = fold_w8 is not None and fold_s8 is not None
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    _fold_argmax_top2_record_kernel[(rows, slices)](
        base, latent,
        fold_w8 if use_fp8 else fold_w,
        fold_s8 if use_fp8 else fold_b,
        fold_b, rec, rec1, sem, rec,
        base.stride(0), softcap, valid, offset, ch,
        RANK=rank, BC=32, USE_FP8=use_fp8, HAS_SKIP=False,
        FUSE_REDUCE=True, NP2_S=triton.next_power_of_2(slices), PDL=use_pdl,
        num_warps=_MARKOV_FOLD_WARPS, launch_pdl=use_pdl,
    )
    g = tensor_model_parallel_all_gather(rec1.squeeze(1), dim=0)
    tp = g.shape[0] // rows
    tok1 = torch.empty((rows,), dtype=torch.int64, device=base.device)
    p1 = torch.empty((rows,), dtype=torch.float32, device=base.device)
    tok2 = torch.empty((rows,), dtype=torch.int64, device=base.device)
    p2 = torch.empty((rows,), dtype=torch.float32, device=base.device)
    _argmax_select_top2_kernel[(rows,)](
        g, tok1, p1, tok2, p2,
        tp, NP2=triton.next_power_of_2(tp), PDL=use_pdl,
        num_warps=1, launch_pdl=use_pdl,
    )
    return tok1, p1, tok2, p2
