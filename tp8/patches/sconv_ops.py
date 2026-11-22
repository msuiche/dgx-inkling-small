# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short convolution over a paged sliding-window conv-state cache.

The cache (``[num_blocks, H, N, D]``, see ``InklingConvState`` in
``vllm/models/inkling2/sconv.py``) stores each stream's conv input at every
absolute position. Per token, one launch writes the current input to its slot
and convolves the ``W`` taps ending at its position: taps inside this forward
are read from ``x``, earlier taps from the paged cache, and taps before
position 0 are zero.

Static convolution (fp32 accumulate, oldest tap first)::

    y = (sum_k w[c, k] * x[t - (W - 1) + k] + x[t]) * (W + 1) ** -0.5

Dynamic convolution (newest tap first, per-token per-head tap weights)::

    y = (sum_l (static_w[l, c] + dyn[t, head(c), l]) * x[t - l] + x[t])
        * (2 * W + 1) ** -0.5

``y`` is rounded to the input dtype. The K/V variant then L2-clips K and
RMS-normalizes V per head and writes both to the paged KV cache.
"""

import ctypes
import os

import torch

from vllm.models.inkling2.ops.norm import _l2_clip, qr_clip_rel_tokens, rsqrt
from vllm.triton_utils import tl, tldevice, triton

_TARGET_ATTN_LIB_PATH = os.environ.get(
    "INKLING_TARGET_ATTN_LIB", "/rdma/libinkling_target_attn_exact.so"
)
_SCONV_CUDA = os.environ.get("INKLING_SCONV_CUDA", "0") == "1"
_SCONV_V2_C32 = os.environ.get("INKLING_SCONV_V2_C32", "0").strip().lower() in ("1", "true", "on", "yes")
_SCONV_NLIVE = os.environ.get("INKLING_SCONV_NLIVE", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_SCONV_PDL = int(os.environ.get("INKLING_SCONV_PDL", "1"))
_PROLOGUE_CUDA = os.environ.get("INKLING_PROLOGUE_CUDA", "0") == "1"
_TARGET_ATTN_LIB = None
_TARGET_ATTN_LIB_TRIED = False
_TREE_SPEC = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)


def _get_tree_runtime():
    if not _TREE_SPEC:
        return None
    try:
        from . import tree_spec_runtime as _tsr
        return _tsr
    except Exception:
        return None


def _get_tree_target_lib():
    _tsr = _get_tree_runtime()
    return _tsr.get_tree_target_lib() if _tsr is not None else None


def _get_target_attn_lib():
    global _TARGET_ATTN_LIB, _TARGET_ATTN_LIB_TRIED
    if _TARGET_ATTN_LIB_TRIED:
        return _TARGET_ATTN_LIB
    _TARGET_ATTN_LIB_TRIED = True
    if not os.path.exists(_TARGET_ATTN_LIB_PATH):
        return None
    lib = ctypes.CDLL(_TARGET_ATTN_LIB_PATH)
    lib.sconv_exact_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.sconv_exact_fwd.restype = ctypes.c_int
    lib.attn_prologue_exact_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.attn_prologue_exact_fwd.restype = ctypes.c_int
    _TARGET_ATTN_LIB = lib
    return lib


@triton.jit
def _cached_taps(
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    c_mask,
    block_table_ptr,
    stride_bt,
    max_blocks,
    req,
    src,
    cached,
):
    safe_src = tl.maximum(src, 0)
    lblk = tl.minimum(safe_src // N, max_blocks - 1)
    blk = tl.load(block_table_ptr + req * stride_bt + lblk, mask=cached, other=0)
    return tl.load(
        cache_ptr
        + blk.to(tl.int64)[:, None] * stride_c_blk
        + (safe_src % N)[:, None] * stride_c_n
        + cache_col[None, :],
        mask=c_mask[None, :] & cached[:, None],
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _cached_tap(
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    c_mask,
    block_table_ptr,
    stride_bt,
    max_blocks,
    req,
    pos,
    toff,
    qstart,
    valid,
    lag: tl.constexpr,
):
    src = pos - lag
    cached = valid & (src >= 0) & (toff - lag < qstart)
    return _cached_taps(
        cache_ptr,
        cache_col,
        stride_c_blk,
        stride_c_n,
        N,
        c_mask,
        block_table_ptr,
        stride_bt,
        max_blocks,
        req,
        src,
        cached,
    )


@triton.jit
def _tap_sum(
    x_ptr,
    stride_x,
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    weight_ptr,
    stride_w0,
    stride_w1,
    block_table_ptr,
    stride_bt,
    max_blocks,
    toff,
    coff,
    c_mask,
    tc_mask,
    pos,
    req,
    qstart,
    valid,
    cur,
    W: tl.constexpr,
    dyn_ptr,
    stride_dyn,
    HEAD_DIM: tl.constexpr,
    preloaded=None,
    READ_CACHE: tl.constexpr = False,
    tree_anc_ptr=None,
    HAS_TREE: tl.constexpr = False,
):
    """Weighted sum of the W taps; earlier steps' taps come from
    ``preloaded`` or, with READ_CACHE, from the cache (else they are zero)."""
    dtype = cache_ptr.dtype.element_ty
    acc = tl.zeros(cur.shape, dtype=tl.float32)
    for k in tl.static_range(W):
        lag = k if dyn_ptr is not None else W - 1 - k
        src = pos - lag
        in_win = valid & (src >= 0)
        if HAS_TREE:
            row = tl.load(tree_anc_ptr + toff * W + lag, mask=valid, other=-1)
            intra = in_win & (row >= 0)
            cached = in_win & (row < 0)
        else:
            row = toff - lag
            intra = in_win & (row >= qstart)
            cached = in_win & (row < qstart)
        xt = (
            tl.load(
                x_ptr + tl.maximum(row, 0)[:, None] * stride_x + coff[None, :],
                mask=c_mask[None, :] & intra[:, None],
                other=0.0,
            )
            .to(dtype)
            .to(tl.float32)
        )
        if preloaded is not None:
            xt += preloaded[k]
        elif READ_CACHE:
            xt += _cached_taps(
                cache_ptr,
                cache_col,
                stride_c_blk,
                stride_c_n,
                N,
                c_mask,
                block_table_ptr,
                stride_bt,
                max_blocks,
                req,
                src,
                cached,
            )
        value = xt
        if dyn_ptr is not None:
            w = tl.load(
                weight_ptr + lag * stride_w0 + coff * stride_w1, mask=c_mask, other=0
            ).to(tl.float32)
            dyn = tl.load(
                dyn_ptr
                + toff[:, None] * stride_dyn
                + (coff // HEAD_DIM)[None, :] * W
                + lag,
                mask=tc_mask,
                other=0,
            ).to(tl.float32)
            w = w[None, :] + dyn
        else:
            w = tl.load(
                weight_ptr + coff * stride_w0 + k * stride_w1, mask=c_mask, other=0
            )
            w = w.to(dtype).to(tl.float32)[None, :]
        acc = value * w if dyn_ptr is not None and k == 0 else tl.fma(value, w, acc)
    return acc


@triton.jit
def _conv(
    x_ptr,  # [T, C] current-token inputs (also the residual)
    stride_x,
    cache_ptr,  # [num_blocks, H, N, D] paged conv-state cache
    cache_col,  # [C] offset of each channel within a cache block row
    stride_c_blk,
    stride_c_n,
    N,
    weight_ptr,  # static: [C, W]; dynamic: [W, C]
    stride_w0,
    stride_w1,
    pos_ptr,  # [T] absolute positions
    seq_idx_ptr,  # [T] request of each token
    slot_ptr,  # [T] flat cache slot (< 0: padding)
    block_table_ptr,  # [num_reqs, max_blocks]
    stride_bt,
    max_blocks,
    qstart_ptr,  # [T] first row of each token's request
    scale,
    toff,
    coff,
    t_mask,
    c_mask,
    W: tl.constexpr,
    dyn_ptr=None,  # dynamic: [T, C // HEAD_DIM, W]
    stride_dyn=0,
    HEAD_DIM: tl.constexpr = 1,
    SKIP_IDLE_CACHE: tl.constexpr = False,
    CACHE_FIRST: tl.constexpr = False,
    tile_reads_cache=True,
    tree_anc_ptr=None,
    HAS_TREE: tl.constexpr = False,
):
    """Store the current inputs to the cache; return ``y`` in fp32.

    With ``CACHE_FIRST`` (static W=4 conv), the taps from earlier steps are
    loaded before waiting on the producer of ``x`` (PDL).
    """
    tc_mask = t_mask[:, None] & c_mask[None, :]
    slot = tl.load(slot_ptr + toff, mask=t_mask, other=-1)
    valid = slot >= 0
    pos = tl.load(pos_ptr + toff, mask=t_mask, other=0)
    req = tl.load(seq_idx_ptr + toff, mask=t_mask, other=0)
    qstart = tl.load(qstart_ptr + toff, mask=t_mask, other=0)
    if CACHE_FIRST:
        tl.static_assert(W == 4 and dyn_ptr is None)
        taps = (
            cache_ptr,
            cache_col,
            stride_c_blk,
            stride_c_n,
            N,
            c_mask,
            block_table_ptr,
            stride_bt,
            max_blocks,
            req,
            pos,
            toff,
            qstart,
            valid,
        )
        cached0 = _cached_tap(*taps, 3)
        cached1 = _cached_tap(*taps, 2)
        cached2 = _cached_tap(*taps, 1)
        cached3 = _cached_tap(*taps, 0)
        tl.extra.cuda.gdc_wait()

    dtype = cache_ptr.dtype.element_ty
    cur = tl.load(x_ptr + toff[:, None] * stride_x + coff[None, :], mask=tc_mask)
    cur = cur.to(dtype)
    safe_slot = tl.maximum(slot, 0)
    dst = (
        cache_ptr
        + (safe_slot // N)[:, None] * stride_c_blk
        + (safe_slot % N)[:, None] * stride_c_n
        + cache_col[None, :]
    )
    tl.store(dst, cur, mask=tc_mask & valid[:, None])

    tap_args = (
        x_ptr,
        stride_x,
        cache_ptr,
        cache_col,
        stride_c_blk,
        stride_c_n,
        N,
        weight_ptr,
        stride_w0,
        stride_w1,
        block_table_ptr,
        stride_bt,
        max_blocks,
        toff,
        coff,
        c_mask,
        tc_mask,
        pos,
        req,
        qstart,
        valid,
        cur,
    )
    # Only tiles with a token among the first W - 1 of a request read taps
    # from the cache; SKIP_IDLE_CACHE branches around the others' loads once.
    if CACHE_FIRST:
        preloaded = (cached0, cached1, cached2, cached3)
        acc = _tap_sum(*tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM, preloaded)
    elif not SKIP_IDLE_CACHE or tile_reads_cache:
        acc = _tap_sum(
            *tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM,
            READ_CACHE=True, tree_anc_ptr=tree_anc_ptr, HAS_TREE=HAS_TREE,
        )
    else:
        acc = _tap_sum(
            *tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM,
            tree_anc_ptr=tree_anc_ptr, HAS_TREE=HAS_TREE,
        )
    return (acc + cur.to(tl.float32)) * scale


@triton.jit
def _sconv_kernel(
    x_ptr,
    cache_ptr,
    weight_ptr,
    out_ptr,  # [T, C]
    res_ptr,  # [T, C] optional residual base, added before addend
    addend_ptr,  # [T, C] optional
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_x,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_w0,
    stride_w1,
    stride_bt,
    max_blocks,
    N,
    scale,
    W: tl.constexpr,
    OFF: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    HAS_RES: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SKIP_IDLE_CACHE: tl.constexpr,
    tree_anc_ptr=None,
    HAS_TREE: tl.constexpr = False,
):
    first = tl.program_id(0) * BT
    tile_reads_cache = True
    if SKIP_IDLE_CACHE:
        tl.extra.cuda.gdc_wait()
        # Request starts are non-decreasing: no token of the tile is among the
        # first W - 1 of its request unless the last token's request starts
        # after first - (W - 1).
        last = tl.minimum(first + BT, T) - 1
        tile_reads_cache = tl.load(qstart_ptr + last) > first - (W - 1)
    toff = first + tl.arange(0, BT)
    coff = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    C = H * WS
    t_mask = toff < T
    c_mask = coff < C
    y = _conv(
        x_ptr,
        stride_x,
        cache_ptr,
        tl.minimum(coff // WS, H - 1) * stride_c_h + OFF + coff % WS,
        stride_c_blk,
        stride_c_n,
        N,
        weight_ptr,
        stride_w0,
        stride_w1,
        pos_ptr,
        seq_idx_ptr,
        slot_ptr,
        block_table_ptr,
        stride_bt,
        max_blocks,
        qstart_ptr,
        scale,
        toff,
        coff,
        t_mask,
        c_mask,
        W,
        SKIP_IDLE_CACHE=SKIP_IDLE_CACHE,
        CACHE_FIRST=(not SKIP_IDLE_CACHE) and (not HAS_TREE),
        tile_reads_cache=tile_reads_cache,
        tree_anc_ptr=tree_anc_ptr,
        HAS_TREE=HAS_TREE,
    ).to(out_ptr.dtype.element_ty)
    offs = toff[:, None] * C + coff[None, :]
    mask = t_mask[:, None] & c_mask[None, :]
    if HAS_ADDEND:
        # Residual update in the reference order: (res + addend) + y in fp32.
        acc = tl.load(addend_ptr + offs, mask, other=0).to(tl.float32)
        if HAS_RES:
            acc = tl.load(res_ptr + offs, mask, other=0).to(tl.float32) + acc
        y = (acc + y.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    addend: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    aux_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Insert ``x`` (rounded to the cache dtype) into the conv-state cache and
    return the static convolution with ``(C, W)`` taps ``weight``.

    With ``addend``, returns ``(residual + addend) + y`` accumulated in fp32
    (``residual`` may alias ``out``). Optional ``aux_out`` receives a fused
    copy of ``out`` (eliminating ``partial.clone()`` aux copies).
    """
    num_tokens, channels = x.shape
    if out is None:
        out = torch.empty((num_tokens, channels), dtype=cache.dtype, device=x.device)
    assert x.stride(1) == 1 and cache.stride(3) == 1 and out.is_contiguous()
    assert residual is None or addend is not None
    for t in (addend, residual, aux_out):
        assert t is None or (t.is_contiguous() and t.shape == out.shape)
    num_heads = cache.shape[1]
    assert channels == num_heads * stream_width
    width = weight.shape[1]
    _tsr = _get_tree_runtime()
    tree_st = _tsr.get_active_tree_state(num_tokens) if _tsr is not None else None
    if (
        tree_st is not None
        and 0 < num_tokens <= 32
        and channels == 4608
        and width == 4
        and num_heads == 1
        and x.dtype == torch.bfloat16
        and cache.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and (aux_out is None or aux_out.dtype == torch.bfloat16)
    ):
        tlib = _get_tree_target_lib()
        if tlib is not None:
            nl_buf = _rowar_nlive_buf(num_tokens) if _SCONV_NLIVE else None
            # sconv_tree_exact_fwd instantiates (addend, res, aux) = (1,1,1),(1,1,0),(1,0,0),(0,0,0)
            # paired with (has_nlive==use_pdl). Layer 40 (addend=1, res=0, aux=1) copies aux after.
            fused_aux = (aux_out is not None) and not (addend is not None and residual is None)
            use_pdl_t = 1 if nl_buf is not None else 0
            rc = tlib.sconv_tree_exact_fwd(
                x.data_ptr(),
                cache.data_ptr(),
                weight.data_ptr(),
                out.data_ptr(),
                aux_out.data_ptr() if fused_aux else 0,
                residual.data_ptr() if residual is not None else 0,
                addend.data_ptr() if addend is not None else 0,
                positions.data_ptr(),
                seq_idx.data_ptr(),
                slot_mapping.data_ptr(),
                block_table.data_ptr(),
                query_start.data_ptr(),
                nl_buf.data_ptr() if nl_buf is not None else 0,
                tree_st.tree_sconv_anc.data_ptr(),
                num_tokens,
                channels,
                x.stride(0),
                cache.stride(0),
                cache.stride(2),
                block_table.stride(0),
                block_table.shape[1],
                cache.shape[2],
                stream_offset,
                (width + 1) ** -0.5,
                1 if addend is not None else 0,
                1 if residual is not None else 0,
                1 if fused_aux else 0,
                1 if nl_buf is not None else 0,
                1,
                use_pdl_t,
                int(torch.cuda.current_stream(x.device).cuda_stream),
            )
            if rc == 0:
                if aux_out is not None and not fused_aux:
                    aux_out.copy_(out)
                return out
    if (
        tree_st is None
        and _SCONV_CUDA
        and 0 < num_tokens <= 32
        and channels == 4608
        and width == 4
        and num_heads == 1
        and x.dtype == torch.bfloat16
        and cache.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and (aux_out is None or aux_out.dtype == torch.bfloat16)
    ):
        lib = _get_target_attn_lib()
        if lib is not None:
            nl_buf = _rowar_nlive_buf(num_tokens) if _SCONV_NLIVE else None
            rc = lib.sconv_exact_fwd(
                x.data_ptr(),
                cache.data_ptr(),
                weight.data_ptr(),
                out.data_ptr(),
                aux_out.data_ptr() if aux_out is not None else 0,
                residual.data_ptr() if residual is not None else 0,
                addend.data_ptr() if addend is not None else 0,
                positions.data_ptr(),
                seq_idx.data_ptr(),
                slot_mapping.data_ptr(),
                block_table.data_ptr(),
                query_start.data_ptr(),
                nl_buf.data_ptr() if nl_buf is not None else 0,
                num_tokens,
                channels,
                x.stride(0),
                cache.stride(0),
                cache.stride(2),
                block_table.stride(0),
                block_table.shape[1],
                cache.shape[2],
                stream_offset,
                (width + 1) ** -0.5,
                1 if addend is not None else 0,
                1 if residual is not None else 0,
                1 if aux_out is not None else 0,
                1 if nl_buf is not None else 0,
                (2 if (_SCONV_PDL == 2 and addend is None) or _SCONV_PDL >= 3 else _SCONV_PDL),
                int(torch.cuda.current_stream(x.device).cuda_stream),
            )
            if rc == 0:
                return out
    # Prefill tiles skip the cache reads they never need.
    prefill = num_tokens > 256
    if num_tokens <= 8:
        block_t, block_c = 1, 1024
    else:
        block_t, block_c = (4, 256) if prefill else (2, 512)
    block_c = min(triton.next_power_of_2(channels), block_c)
    if (
        (num_tokens >= 1024 or (192 <= num_tokens <= 384) or (_SCONV_V2_C32 and 33 <= num_tokens <= 384))
        and width == 4
        and num_heads == 1
        and channels % 512 == 0
        and x.dtype == torch.bfloat16
        and cache.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and (addend is None or addend.dtype == torch.bfloat16)
        and (residual is None or residual.dtype == torch.bfloat16)
        and stream_offset % 8 == 0
    ):
        try:
            from . import prefill_qf as _pqf
            _lib = _pqf.get_lib()
        except Exception:
            _lib = None
        if _lib is not None:
            _bt = 4 if num_tokens >= 1024 else (8 if num_tokens <= 128 else 16)
            _rc = _lib.launch_short_conv_v2(
                x.data_ptr(),
                cache.data_ptr(),
                weight.data_ptr(),
                out.data_ptr(),
                0 if residual is None else residual.data_ptr(),
                0 if addend is None else addend.data_ptr(),
                positions.data_ptr(),
                seq_idx.data_ptr(),
                slot_mapping.data_ptr(),
                block_table.data_ptr(),
                query_start.data_ptr(),
                num_tokens,
                int(x.stride(0)),
                int(cache.stride(0)),
                int(cache.stride(2)),
                int(weight.stride(0)),
                int(weight.stride(1)),
                int(block_table.stride(0)),
                int(block_table.shape[1]),
                int(cache.shape[2]),
                int(stream_offset),
                int(channels),
                float((width + 1) ** -0.5),
                1 if addend is not None else 0,
                1 if residual is not None else 0,
                _bt,
                int(torch.cuda.current_stream().cuda_stream),
            )
            if _rc == 0:
                if aux_out is not None:
                    aux_out.copy_(out)
                return out
    grid = (triton.cdiv(num_tokens, block_t), triton.cdiv(channels, block_c))
    _sconv_kernel[grid](
        x,
        cache,
        weight,
        out,
        residual,
        addend,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        x.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        weight.stride(0),
        weight.stride(1),
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        (width + 1) ** -0.5,
        W=width,
        OFF=stream_offset,
        WS=stream_width,
        H=num_heads,
        HAS_RES=residual is not None,
        HAS_ADDEND=addend is not None,
        BT=block_t,
        BLOCK_C=block_c,
        SKIP_IDLE_CACHE=prefill and (tree_st is None),
        tree_anc_ptr=tree_st.tree_sconv_anc if tree_st is not None else seq_idx,
        HAS_TREE=tree_st is not None,
        num_warps=4,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    if aux_out is not None:
        aux_out.copy_(out)
    return out


# Columns of shift_rel_proj beyond the extent, 256 on either side.
_SHIFT_PAD = 512


def shift_rel_proj(rel_proj: torch.Tensor, is_local: bool) -> torch.Tensor:
    """Return the ``(8, d_rel + 1, extent + 512)`` table ``attention_prologue``
    reads the ``(d_rel, extent)`` relative projection from, so that every
    sheared output column block is an aligned slice of one of 8 shifts.

    ``out[s, :d_rel, y]`` is ``rel_proj[:, d]`` for distance ``d = s + extent
    + 256 - y``, or zero out of range; row ``d_rel`` is the padding added to
    the logits: -inf for negative distances and, if ``is_local``, those past
    the extent, else zero.
    """
    d_rel, extent = rel_proj.shape
    shift = torch.arange(8, device=rel_proj.device)[:, None]
    y = torch.arange(extent + _SHIFT_PAD, device=rel_proj.device)
    dist = shift + extent + 256 - y
    outside = (dist < 0) | (dist >= extent)
    proj = rel_proj[:, dist.clamp(0, extent - 1)].transpose(0, 1)
    pad = (dist >= extent) if is_local else torch.zeros_like(outside)
    pad = (dist < 0) | pad
    pad = torch.zeros_like(proj[:, :1]).masked_fill(pad[:, None], float("-inf"))
    return torch.cat([proj.masked_fill(outside[:, None], 0), pad], 1).contiguous()


@triton.jit
def _attn_prologue_kernel(
    k_ptr,  # [T, H * D] K projection
    v_ptr,  # [T, H * D] V projection
    wk_ptr,  # static: [H * D, W]; dynamic: [W, H * D]
    wv_ptr,
    dk_ptr,  # dynamic: [T, H, W] per-token K tap deltas
    dv_ptr,
    cache_ptr,
    key_cache_ptr,  # [num_blocks, block_size, H, D]
    value_cache_ptr,
    kv_slot_ptr,  # [T] KV cache slot (< 0: skip)
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_k,
    stride_v,
    stride_w0,
    stride_w1,
    stride_dk,
    stride_dv,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_kv_blk,
    stride_kv_n,
    stride_kv_h,
    kv_block_size,
    stride_bt,
    max_blocks,
    N,
    scale,
    clip_threshold,
    clip_eps,
    norm_eps,
    q_ptr,  # [T, HQ, D] query projection (row-strided)
    r_ptr,  # [T, HQ, D_REL] relative-query projection (row-strided)
    q_scale_ptr,
    r_scale_ptr,
    tau_ptr,
    rel_proj_ptr,
    q_out_ptr,
    rel_out_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    stride_q,
    stride_r,
    rel_threshold,
    W: tl.constexpr,
    D: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BT: tl.constexpr,
    HQ: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    SHIFT_W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_E: tl.constexpr,
    QR_TOKENS: tl.constexpr,
    TILE: tl.constexpr,
    NLivePtr=None,
    HAS_NLIVE: tl.constexpr = False,
):
    tl.extra.cuda.gdc_wait()
    # The decode attention kernel prefetches cached keys meanwhile.
    tl.extra.cuda.gdc_launch_dependents()
    if HAS_NLIVE:
        if tl.program_id(0) * BT >= tl.load(NLivePtr):
            return
    # Programs (tile, head, 0 | 1) run the K | V conv of one KV head; programs
    # (tile, head, 2) clip q of QR_TOKENS of the tile's tokens and programs
    # (tile, head, 3 + c) their r and column block c of the relative logits.
    if tl.program_id(2) >= 2:
        if tl.program_id(1) * QR_TOKENS < BT:
            qr_clip_rel_tokens(
                tl.program_id(0) * BT + tl.program_id(1) * QR_TOKENS,
                QR_TOKENS,
                tl.program_id(2) - 3,
                T,
                q_ptr,
                r_ptr,
                q_scale_ptr,
                r_scale_ptr,
                tau_ptr,
                rel_proj_ptr,
                q_out_ptr,
                rel_out_ptr,
                seq_idx_ptr,
                cu_seqlens_q_ptr,
                seq_lens_ptr,
                stride_q,
                stride_r,
                clip_threshold,
                rel_threshold,
                clip_eps,
                HQ,
                D,
                D_REL,
                E,
                SHIFT_W,
                HAS_TAU,
                BLOCK_HQ,
                BLOCK_E,
            )
    else:
        _kv_conv_tile(
            k_ptr,
            v_ptr,
            wk_ptr,
            wv_ptr,
            dk_ptr,
            dv_ptr,
            cache_ptr,
            key_cache_ptr,
            value_cache_ptr,
            kv_slot_ptr,
            pos_ptr,
            seq_idx_ptr,
            slot_ptr,
            block_table_ptr,
            qstart_ptr,
            T,
            stride_k,
            stride_v,
            stride_w0,
            stride_w1,
            stride_dk,
            stride_dv,
            stride_c_blk,
            stride_c_h,
            stride_c_n,
            stride_kv_blk,
            stride_kv_n,
            stride_kv_h,
            kv_block_size,
            stride_bt,
            max_blocks,
            N,
            scale,
            clip_threshold,
            clip_eps,
            norm_eps,
            W,
            D,
            DYNAMIC,
            BT,
            TILE,
        )


@triton.jit
def _kv_conv_tile(
    k_ptr,
    v_ptr,
    wk_ptr,
    wv_ptr,
    dk_ptr,
    dv_ptr,
    cache_ptr,
    key_cache_ptr,
    value_cache_ptr,
    kv_slot_ptr,
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_k,
    stride_v,
    stride_w0,
    stride_w1,
    stride_dk,
    stride_dv,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_kv_blk,
    stride_kv_n,
    stride_kv_h,
    kv_block_size,
    stride_bt,
    max_blocks,
    N,
    scale,
    clip_threshold,
    clip_eps,
    norm_eps,
    W: tl.constexpr,
    D: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BT: tl.constexpr,
    TILE: tl.constexpr,
):
    # TILE rows are always computed (so every row reduction keeps the stock
    # [4, D] layout and rounding order); only the first BT rows are live.
    rows = tl.arange(0, TILE)
    toff = tl.program_id(0) * BT + rows
    head = tl.program_id(1)
    is_v = tl.program_id(2) == 1
    d_idx = tl.arange(0, D)
    coff = head * D + d_idx
    t_mask = (toff < T) & (rows < BT)
    if is_v:
        x_ptr = v_ptr
        w_ptr = wv_ptr
        dyn_ptr = dv_ptr
        stride_x = stride_v
        stride_dyn = stride_dv
    else:
        x_ptr = k_ptr
        w_ptr = wk_ptr
        dyn_ptr = dk_ptr
        stride_x = stride_k
        stride_dyn = stride_dk
    y = _conv(
        x_ptr,
        stride_x,
        cache_ptr,
        head * stride_c_h + tl.program_id(2) * D + d_idx,
        stride_c_blk,
        stride_c_n,
        N,
        w_ptr,
        stride_w0,
        stride_w1,
        pos_ptr,
        seq_idx_ptr,
        slot_ptr,
        block_table_ptr,
        stride_bt,
        max_blocks,
        qstart_ptr,
        scale,
        toff,
        coff,
        t_mask,
        d_idx < D,
        W,
        dyn_ptr if DYNAMIC else None,
        stride_dyn,
        D,
    )
    y = y.to(k_ptr.dtype.element_ty).to(tl.float32)
    sum_sq = tl.sum(y * y, axis=1)
    if is_v:
        y = y * rsqrt(tl.div_rn(sum_sq, D) + norm_eps)[:, None]
        cache = value_cache_ptr
    else:
        norm = tl.where(sum_sq > 0, sum_sq * rsqrt(sum_sq), 0.0)
        y = y * tl.minimum(tl.div_rn(clip_threshold, norm + clip_eps), 1.0)[:, None]
        cache = key_cache_ptr
    kv_slot = tl.load(kv_slot_ptr + toff, mask=t_mask, other=-1)
    safe = tl.maximum(kv_slot, 0)
    dst = (
        cache
        + (safe // kv_block_size)[:, None] * stride_kv_blk
        + (safe % kv_block_size)[:, None] * stride_kv_n
        + head * stride_kv_h
        + d_idx[None, :]
    )
    tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])


# INKLING_PROLOGUE_SPLIT=1: launch the decode-sized attention prologue with
# INKLING_PROLOGUE_SPLIT_BT (default 1) tokens per tile instead of 4.
_PROLOGUE_SPLIT = os.environ.get("INKLING_PROLOGUE_SPLIT", "0") == "1"
_PROLOGUE_PREFILL_OPT = os.environ.get("INKLING_PROLOGUE_PREFILL_OPT", "1").strip().lower() not in ("0", "", "off", "false", "no")


@triton.jit
def _rel_prefill_kernel(
    R, RScale, Tau, RelProj, RelOut, SeqIdx, CuSeqlensQ, SeqLens,
    T, stride_r, r_threshold, eps,
    H: tl.constexpr, D_REL: tl.constexpr, E: tl.constexpr, W: tl.constexpr,
    HAS_TAU: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr,
    BT: tl.constexpr,
):
    """0-ULP prefill relative-logits kernel: clips r ONCE per token across all
    (E+256)//BLOCK_E chunks and streams RelOut with cache_modifier='.cs'."""
    t0 = tl.program_id(0) * BT
    E_PAD: tl.constexpr = E + 256
    N_CHUNKS: tl.constexpr = E_PAD // BLOCK_E
    heads = tl.arange(0, BLOCK_H)[:, None]
    h_mask = heads < H
    r_cols = tl.arange(0, D_REL)[None, :]
    e_idx = tl.arange(0, BLOCK_E)[None, :]

    for i in tl.static_range(BT):
        t = (t0 + i).to(tl.int64)
        if t < T:
            seq = tl.load(SeqIdx + t)
            q_start = tl.load(CuSeqlensQ + seq)
            q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
            q_idx = t - q_start
            pos = tl.load(SeqLens + seq) - q_len + q_idx
            tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
            diag = pos + E_PAD - tl.cdiv(pos - q_idx + tile_end, 128) * 128
            diag = tl.minimum(tl.maximum(diag, E + 1), E_PAD - 1)
            tau = tl.load(Tau + t) if HAS_TAU else 1.0
            r = _l2_clip(
                R + t * stride_r, RScale, heads, r_cols, h_mask,
                r_threshold, eps, tau, HAS_TAU,
            )
            diag8 = diag // 8 * 8
            table = RelProj + (diag - diag8) * (D_REL + 1) * W + E + 256 - diag8
            out_row = RelOut + t * H * E_PAD + heads * E_PAD
            for ch in tl.static_range(N_CHUNKS):
                col = ch * BLOCK_E + e_idx
                proj = tl.load(table + tl.trans(r_cols) * W + col)
                pad = tl.load(table + D_REL * W + col)
                rel = tl.dot(r, proj) + pad.to(tl.float32)
                tl.store(out_row + col, rel.to(RelOut.dtype.element_ty), h_mask, cache_modifier=".cs")


@triton.jit
def _kvq_prefill_kernel(
    k_ptr, v_ptr, wk_ptr, wv_ptr, dk_ptr, dv_ptr, cache_ptr,
    key_cache_ptr, value_cache_ptr, kv_slot_ptr,
    pos_ptr, seq_idx_ptr, slot_ptr, block_table_ptr, qstart_ptr,
    T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
    stride_c_blk, stride_c_h, stride_c_n,
    stride_kv_blk, stride_kv_n, stride_kv_h, kv_block_size,
    stride_bt, max_blocks, N, scale, clip_threshold, clip_eps, norm_eps,
    q_ptr, q_scale_ptr, tau_ptr, q_out_ptr, stride_q,
    W: tl.constexpr, D: tl.constexpr, DYNAMIC: tl.constexpr, BT: tl.constexpr,
    HQ: tl.constexpr, HAS_TAU: tl.constexpr, BLOCK_HQ: tl.constexpr,
):
    pid2 = tl.program_id(2)
    if pid2 == 2:
        if tl.program_id(1) == 0:
            t0 = tl.program_id(0) * BT
            heads = tl.arange(0, BLOCK_HQ)[:, None]
            h_mask = heads < HQ
            q_cols = tl.arange(0, D)[None, :]
            for i in tl.static_range(BT):
                t = (t0 + i).to(tl.int64)
                if t < T:
                    tau = tl.load(tau_ptr + t) if HAS_TAU else 1.0
                    q = _l2_clip(q_ptr + t * stride_q, q_scale_ptr, heads, q_cols, h_mask,
                                 clip_threshold, clip_eps, tau, HAS_TAU)
                    tl.store(q_out_ptr + t * HQ * D + heads * D + q_cols, q, h_mask)
    else:
        _kv_conv_tile(
            k_ptr, v_ptr, wk_ptr, wv_ptr, dk_ptr, dv_ptr, cache_ptr,
            key_cache_ptr, value_cache_ptr, kv_slot_ptr,
            pos_ptr, seq_idx_ptr, slot_ptr, block_table_ptr, qstart_ptr,
            T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
            stride_c_blk, stride_c_h, stride_c_n,
            stride_kv_blk, stride_kv_n, stride_kv_h, kv_block_size,
            stride_bt, max_blocks, N, scale, clip_threshold, clip_eps, norm_eps,
            W, D, DYNAMIC, BT, 4,
        )
_PROLOGUE_SPLIT_BT = int(os.environ.get("INKLING_PROLOGUE_SPLIT_BT", "1"))
_PROLOGUE_SPLIT_MAXT = int(os.environ.get("INKLING_PROLOGUE_SPLIT_MAXT", "32"))
_ATTN_NLIVE = os.environ.get("INKLING_ATTN_NLIVE", "1").strip().lower() not in ("0", "", "off", "false", "no")


def _rowar_nlive_buf(num_tokens: int):
    if not _ATTN_NLIVE:
        return None
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as rar
        if rar.ROWAR.get("scope") == num_tokens:
            return rar.ROWAR.get("buf")
    except Exception:
        pass
    return None


def attention_prologue(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    rel_proj: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    q_out: torch.Tensor,
    rel_out: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    clip_threshold: float,
    rel_threshold: float,
    norm_eps: float,
    clip_eps: float = 1e-6,
    rel_proj_rev: torch.Tensor | None = None,
    skip_rel: bool = False,
) -> None:
    """Attention input processing in one launch.

    Writes the L2-clipped ``q * q_scale`` to ``q_out`` and the relative logits
    of the clipped ``r * r_scale`` to ``rel_out`` (with ``tau``, clipped values
    are rounded, then scaled by the fp32 per-token ``tau``). ``rel_proj`` is
    the ``shift_rel_proj`` table; the logits come out sheared as tml_fa4 reads
    them for a varlen launch over all tokens (``(T + 128, HQ, extent +
    256)``) from the per-request ``cu_seqlens_q`` and ``seq_lens``. Runs the
    K/V short convolutions, clips K, RMS-normalizes V and inserts both into
    the KV cache (tokens with ``kv_slot_mapping < 0`` skip the write).

    ``q`` ``(T, HQ, D)``, ``r`` ``(T, HQ, D_rel)`` and ``k``/``v``
    ``(T, H * D)`` may be row-strided; K and V occupy ``[0, D)`` and
    ``[D, 2 * D)`` of each conv-state head. Conv weights are static
    ``(H * D, W)`` taps, or with ``(T, H * W)`` per-token deltas
    ``k/v_dynamic`` the ``(W, H * D)`` static part.
    """
    num_tokens = kv_slot_mapping.shape[0]
    num_heads, head_dim = key_cache.shape[2:]
    num_q_heads, d_rel = r.shape[1:]
    extent = rel_proj.shape[2] - _SHIFT_PAD
    assert rel_proj.shape[:2] == (8, d_rel + 1)
    assert k.stride(1) == v.stride(1) == 1 and cache.stride(3) == 1
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert k_weight.stride() == v_weight.stride()
    assert q.shape[1:] == (num_q_heads, head_dim)
    assert q_out.shape == q.shape
    if not skip_rel:
        assert rel_out is not None
        assert rel_out.shape == (q.shape[0] + 128, num_q_heads, extent + 256)
    assert q.stride(2) == 1 and q.stride(1) == head_dim
    assert r.stride(2) == 1 and r.stride(1) == d_rel
    assert q_out.is_contiguous() and (skip_rel or (rel_out is not None and rel_out.is_contiguous()))
    assert q_scale.is_contiguous() and r_scale.is_contiguous()
    assert rel_proj.is_contiguous() and d_rel >= 16
    block_e = min(extent, 256)
    for size in (head_dim, d_rel, block_e):
        assert size & (size - 1) == 0, "tile sizes must be powers of 2"
    assert extent % block_e == 0
    dynamic = k_dynamic is not None
    if dynamic:
        assert k_dynamic is not None and v_dynamic is not None
        width = k_weight.shape[0]
        assert k_dynamic.stride(1) == v_dynamic.stride(1) == 1
        dyn_strides = (k_dynamic.stride(0), v_dynamic.stride(0))
    else:
        width = k_weight.shape[1]
        dyn_strides = (0, 0)
    _tsr = _get_tree_runtime()
    tree_st = _tsr.get_active_tree_state(num_tokens) if _tsr is not None else None
    _pro_geom_ok = (
        not skip_rel
        and rel_out is not None
        and rel_proj_rev is not None
        and 0 < num_tokens <= 16
        and num_q_heads == 6
        and num_heads == 1
        and head_dim == 128
        and d_rel == 16
        and extent == 1024
        and width == 4
        and cache.shape[1] == 1
        and rel_proj_rev.shape == (16, 1024)
        and rel_proj_rev.is_contiguous()
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and k_weight.dtype == torch.bfloat16
        and v_weight.dtype == torch.bfloat16
        and q_scale.dtype == torch.bfloat16
        and r_scale.dtype == torch.bfloat16
        and rel_proj_rev.dtype == torch.bfloat16
        and cache.dtype == torch.bfloat16
        and key_cache.dtype == torch.bfloat16
        and q_out.dtype == torch.bfloat16
        and rel_out.dtype == torch.bfloat16
        and (tau is None or (tau.dtype == torch.float32 and tau.is_contiguous()))
        and (
            not dynamic
            or (k_dynamic.dtype == torch.bfloat16 and v_dynamic.dtype == torch.bfloat16)
        )
    )
    # INKLING_TREE_SPEC=1: dispatch 0-ULP tree-aware CUDA prologue BEFORE linear prologue!
    if tree_st is not None and _pro_geom_ok:
        tlib = _get_tree_target_lib()
        if tlib is not None:
            nl_buf = _rowar_nlive_buf(num_tokens)
            rc = tlib.attn_prologue_tree_exact_fwd(
                k.data_ptr(),
                v.data_ptr(),
                k_weight.data_ptr(),
                v_weight.data_ptr(),
                k_dynamic.data_ptr() if dynamic else 0,
                v_dynamic.data_ptr() if dynamic else 0,
                cache.data_ptr(),
                key_cache.data_ptr(),
                value_cache.data_ptr(),
                kv_slot_mapping.data_ptr(),
                positions.data_ptr(),
                seq_idx.data_ptr(),
                slot_mapping.data_ptr(),
                block_table.data_ptr(),
                query_start.data_ptr(),
                q.data_ptr(),
                r.data_ptr(),
                q_scale.data_ptr(),
                r_scale.data_ptr(),
                tau.data_ptr() if tau is not None else 0,
                rel_proj_rev.data_ptr(),
                q_out.data_ptr(),
                rel_out.data_ptr(),
                cu_seqlens_q.data_ptr(),
                seq_lens.data_ptr(),
                nl_buf.data_ptr() if nl_buf is not None else 0,
                tree_st.tree_sconv_anc.data_ptr(),
                num_tokens,
                seq_lens.shape[0],
                k.stride(0),
                v.stride(0),
                k_weight.stride(0),
                k_weight.stride(1),
                dyn_strides[0],
                dyn_strides[1],
                cache.stride(0),
                cache.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.shape[1],
                block_table.stride(0),
                block_table.shape[1],
                cache.shape[2],
                q.stride(0),
                r.stride(0),
                (width * (2 if dynamic else 1) + 1) ** -0.5,
                clip_threshold,
                rel_threshold,
                clip_eps,
                norm_eps,
                1 if dynamic else 0,
                1 if tau is not None else 0,
                1 if nl_buf is not None else 0,
                1,
                1,
                int(torch.cuda.current_stream(q.device).cuda_stream),
            )
            if rc == 0:
                return
    if tree_st is None and _PROLOGUE_CUDA and _pro_geom_ok:
        lib = _get_target_attn_lib()
        if lib is not None:
            nl_buf = _rowar_nlive_buf(num_tokens)
            rc = lib.attn_prologue_exact_fwd(
                k.data_ptr(),
                v.data_ptr(),
                k_weight.data_ptr(),
                v_weight.data_ptr(),
                k_dynamic.data_ptr() if dynamic else 0,
                v_dynamic.data_ptr() if dynamic else 0,
                cache.data_ptr(),
                key_cache.data_ptr(),
                value_cache.data_ptr(),
                kv_slot_mapping.data_ptr(),
                positions.data_ptr(),
                seq_idx.data_ptr(),
                slot_mapping.data_ptr(),
                block_table.data_ptr(),
                query_start.data_ptr(),
                q.data_ptr(),
                r.data_ptr(),
                q_scale.data_ptr(),
                r_scale.data_ptr(),
                tau.data_ptr() if tau is not None else 0,
                rel_proj_rev.data_ptr(),
                q_out.data_ptr(),
                rel_out.data_ptr(),
                cu_seqlens_q.data_ptr(),
                seq_lens.data_ptr(),
                nl_buf.data_ptr() if nl_buf is not None else 0,
                num_tokens,
                seq_lens.shape[0],
                k.stride(0),
                v.stride(0),
                k_weight.stride(0),
                k_weight.stride(1),
                dyn_strides[0],
                dyn_strides[1],
                cache.stride(0),
                cache.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.shape[1],
                block_table.stride(0),
                block_table.shape[1],
                cache.shape[2],
                q.stride(0),
                r.stride(0),
                (width * (2 if dynamic else 1) + 1) ** -0.5,
                clip_threshold,
                rel_threshold,
                clip_eps,
                norm_eps,
                1 if dynamic else 0,
                1 if tau is not None else 0,
                1 if nl_buf is not None else 0,
                1,
                int(torch.cuda.current_stream(q.device).cuda_stream),
            )
            if rc == 0:
                return
    if _PROLOGUE_PREFILL_OPT and num_tokens >= 256 and num_heads == 1:
        _kvq_prefill_kernel[(triton.cdiv(num_tokens, 4), num_heads, 3)](
            k, v, k_weight, v_weight,
            k_dynamic if dynamic else k_weight, v_dynamic if dynamic else v_weight,
            cache, key_cache, value_cache, kv_slot_mapping,
            positions, seq_idx, slot_mapping, block_table, query_start,
            num_tokens, k.stride(0), v.stride(0), k_weight.stride(0), k_weight.stride(1), *dyn_strides,
            cache.stride(0), cache.stride(1), cache.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.shape[1],
            block_table.stride(0), block_table.shape[1], cache.shape[2],
            (width * (2 if dynamic else 1) + 1) ** -0.5, clip_threshold, clip_eps, norm_eps,
            q, q_scale, tau if tau is not None else q, q_out, q.stride(0),
            W=width, D=head_dim, DYNAMIC=dynamic, BT=4, HQ=num_q_heads,
            HAS_TAU=tau is not None, BLOCK_HQ=max(triton.next_power_of_2(num_q_heads), 16),
            num_warps=4, enable_fp_fusion=False,
        )
        if skip_rel:
            return
        bt_r = 1 if (num_tokens >= 4096 or num_tokens < 1024) else (8 if num_tokens >= 2048 else 4)
        be_r = 128 if (1024 <= num_tokens < 2048) else block_e
        _rel_prefill_kernel[(triton.cdiv(num_tokens, bt_r),)](
            r, r_scale, tau if tau is not None else r, rel_proj, rel_out,
            seq_idx, cu_seqlens_q, seq_lens,
            num_tokens, r.stride(0), rel_threshold, clip_eps,
            H=num_q_heads, D_REL=d_rel, E=extent, W=rel_proj.shape[2],
            HAS_TAU=tau is not None, BLOCK_H=max(triton.next_power_of_2(num_q_heads), 16),
            BLOCK_E=be_r, BT=bt_r,
            num_warps=4, enable_fp_fusion=False,
        )
        return
    block_t = 4
    if _PROLOGUE_SPLIT and num_tokens <= _PROLOGUE_SPLIT_MAXT:
        # One token per program column: 4x the CTAs on 48 SMs for the
        # latency-bound decode launch. The K/V tile keeps its [4, D] shape
        # (TILE) so the row-sum rounding order is the stock one.
        block_t = _PROLOGUE_SPLIT_BT
    grid = (triton.cdiv(num_tokens, block_t), num_heads, 3 + (extent + 256) // block_e)
    nl_buf = _rowar_nlive_buf(num_tokens) if block_t == 1 else None
    _attn_prologue_kernel[grid](
        k,
        v,
        k_weight,
        v_weight,
        # Unused placeholders for static convs keep both branches typed alike.
        k_dynamic if dynamic else k_weight,
        v_dynamic if dynamic else v_weight,
        cache,
        key_cache,
        value_cache,
        kv_slot_mapping,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        k.stride(0),
        v.stride(0),
        k_weight.stride(0),
        k_weight.stride(1),
        *dyn_strides,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.shape[1],
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        (width * (2 if dynamic else 1) + 1) ** -0.5,
        clip_threshold,
        clip_eps,
        norm_eps,
        q,
        r,
        q_scale,
        r_scale,
        tau,
        rel_proj,
        q_out,
        rel_out,
        cu_seqlens_q,
        seq_lens,
        q.stride(0),
        r.stride(0),
        rel_threshold,
        W=width,
        D=head_dim,
        DYNAMIC=dynamic,
        BT=block_t,
        HQ=num_q_heads,
        D_REL=d_rel,
        E=extent,
        SHIFT_W=rel_proj.shape[2],
        HAS_TAU=tau is not None,
        BLOCK_HQ=max(triton.next_power_of_2(num_q_heads), 16),
        BLOCK_E=block_e,
        QR_TOKENS=triton.cdiv(block_t, num_heads),
        TILE=4,
        NLivePtr=nl_buf if nl_buf is not None else seq_idx,
        HAS_NLIVE=nl_buf is not None,
        num_warps=4,
        enable_fp_fusion=False,
        launch_pdl=True,
    )


# ---- S5-W4: 0-ULP Fused Triton short_conv + attn_res + RMSNorm (decode T <= 8, D == 4608) ----
@triton.jit
def _sconv_1tok_half(
    x_ptr, stride_x, cache_ptr, cache_off, stride_c_blk, stride_c_n, N_CACHE,
    w_ptr, stride_w0, stride_w1,
    bt_ptr, stride_bt, max_blocks,
    t, pos, req, qstart, valid,
    coff, c_mask, cur_bf16, scale,
    W: tl.constexpr,
):
    dtype = cache_ptr.dtype.element_ty
    acc = tl.zeros(coff.shape, dtype=tl.float32)
    for k in tl.static_range(W):
        lag = W - 1 - k
        src = pos - lag
        row = t - lag
        in_win = valid & (src >= 0)
        intra = in_win & (row >= qstart)
        cached = in_win & (row < qstart)
        xt = tl.load(
            x_ptr + tl.maximum(row, 0) * stride_x + coff,
            mask=c_mask & intra, other=0.0,
        ).to(dtype).to(tl.float32)
        safe_src = tl.maximum(src, 0)
        lblk = tl.minimum(safe_src // N_CACHE, max_blocks - 1)
        blk = tl.load(bt_ptr + req * stride_bt + lblk, mask=cached, other=0)
        xt += tl.load(
            cache_ptr + blk.to(tl.int64) * stride_c_blk + (safe_src % N_CACHE) * stride_c_n + cache_off + coff,
            mask=c_mask & cached, other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + coff * stride_w0 + k * stride_w1, mask=c_mask, other=0).to(dtype).to(tl.float32)
        acc = tl.fma(xt, w, acc)
    return ((acc + cur_bf16.to(tl.float32)) * scale).to(dtype)


@triton.jit
def _fused_sconv_ar_triton_kernel(
    X, Cache, CW, ConvOut, ConvRes, ConvAddend,
    Pos, SeqIdx, Slot, BT, QStart,
    Stack, MixLast, Queries, NormW, Out, NLivePtr,
    T, stride_x, stride_c_blk, stride_c_n, stride_w0, stride_w1, stride_bt, max_blocks, N_CACHE,
    stride_s, scale, eps, norm_eps,
    OFF: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
    NQ: tl.constexpr, N_STACK: tl.constexpr,
    HAS_CONV_RES: tl.constexpr, HAS_CONV_ADDEND: tl.constexpr, HAS_MIX_LAST: tl.constexpr,
    NORMALIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr,
    EARLY: tl.constexpr, HAS_NLIVE: tl.constexpr,
):
    if EARLY:
        tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0).to(tl.int64)
    if HAS_NLIVE:
        if t >= tl.load(NLivePtr):
            return
    da = tl.arange(0, BLOCK_A)
    db = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = da < D
    mask_b = db < D

    slot = tl.load(Slot + t)
    valid = slot >= 0
    pos = tl.load(Pos + t)
    req = tl.load(SeqIdx + t)
    qstart = tl.load(QStart + t)
    dtype = Cache.dtype.element_ty

    cur_a = tl.load(X + t * stride_x + da, mask=mask_a, other=0).to(dtype)
    cur_b = tl.load(X + t * stride_x + db, mask=mask_b, other=0).to(dtype)
    safe_slot = tl.maximum(slot, 0)
    dst_base = Cache + (safe_slot // N_CACHE) * stride_c_blk + (safe_slot % N_CACHE) * stride_c_n + OFF
    tl.store(dst_base + da, cur_a, mask=mask_a & valid)
    tl.store(dst_base + db, cur_b, mask=mask_b & valid)

    ya = _sconv_1tok_half(
        X, stride_x, Cache, OFF, stride_c_blk, stride_c_n, N_CACHE,
        CW, stride_w0, stride_w1, BT, stride_bt, max_blocks,
        t, pos, req, qstart, valid, da, mask_a, cur_a, scale, W,
    )
    yb = _sconv_1tok_half(
        X, stride_x, Cache, OFF, stride_c_blk, stride_c_n, N_CACHE,
        CW, stride_w0, stride_w1, BT, stride_bt, max_blocks,
        t, pos, req, qstart, valid, db, mask_b, cur_b, scale, W,
    )

    row_off = t * D
    if HAS_CONV_ADDEND:
        aa = tl.load(ConvAddend + row_off + da, mask=mask_a, other=0).to(tl.float32)
        ab = tl.load(ConvAddend + row_off + db, mask=mask_b, other=0).to(tl.float32)
        if HAS_CONV_RES:
            aa = tl.load(ConvRes + row_off + da, mask=mask_a, other=0).to(tl.float32) + aa
            ab = tl.load(ConvRes + row_off + db, mask=mask_b, other=0).to(tl.float32) + ab
        ya = (aa + ya.to(tl.float32)).to(dtype)
        yb = (ab + yb.to(tl.float32)).to(dtype)
    tl.store(ConvOut + row_off + da, ya, mask=mask_a)
    tl.store(ConvOut + row_off + db, yb, mask=mask_b)

    if HAS_MIX_LAST:
        la = tl.load(MixLast + row_off + da, mask=mask_a, other=0).to(tl.float32) + ya.to(tl.float32)
        lb = tl.load(MixLast + row_off + db, mask=mask_b, other=0).to(tl.float32) + yb.to(tl.float32)
    else:
        la = ya.to(tl.float32)
        lb = yb.to(tl.float32)

    n_idx = tl.arange(0, BLOCK_N)
    rstd_last = rsqrt(tl.fma(tl.sum(la * la) + tl.sum(lb * lb), 1.0 / D, eps))

    for q_idx in tl.static_range(NQ):
        q_off = q_idx * D
        qa = tl.load(Queries + q_off + da, mask=mask_a, other=0)
        qb = tl.load(Queries + q_off + db, mask=mask_b, other=0)
        logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        inv_rms = tl.zeros((BLOCK_N,), tl.float32)
        for n in tl.static_range(N_STACK):
            xa = tl.load(Stack + n * stride_s + row_off + da, mask=mask_a, other=0).to(tl.float32)
            xb = tl.load(Stack + n * stride_s + row_off + db, mask=mask_b, other=0).to(tl.float32)
            rstd = rsqrt(tl.fma(tl.sum(xa * xa) + tl.sum(xb * xb), 1.0 / D, eps))
            inv_rms = tl.where(n_idx == n, rstd, inv_rms)
            dot = tl.sum(qa * xa) + tl.sum(qb * xb)
            logits = tl.where(n_idx == n, dot * rstd, logits)
        inv_rms = tl.where(n_idx == N_STACK, rstd_last, inv_rms)
        dot_last = tl.sum(qa * la) + tl.sum(qb * lb)
        logits = tl.where(n_idx == N_STACK, dot_last * rstd_last, logits)

        probs = tldevice.exp(logits - tl.max(logits))
        inv_sum = tl.div_rn(1.0, tl.sum(probs))
        if NORMALIZE:
            probs = probs * inv_rms

        acc_a = tl.zeros((BLOCK_A,), tl.float32)
        acc_b = tl.zeros((BLOCK_B,), tl.float32)
        for n in tl.static_range(N_STACK):
            p = tl.sum(tl.where(n_idx == n, probs, 0.0))
            xa = tl.load(Stack + n * stride_s + row_off + da, mask=mask_a, other=0).to(tl.float32)
            xb = tl.load(Stack + n * stride_s + row_off + db, mask=mask_b, other=0).to(tl.float32)
            acc_a = tl.fma(p, xa, acc_a)
            acc_b = tl.fma(p, xb, acc_b)
        p_last = tl.sum(tl.where(n_idx == N_STACK, probs, 0.0))
        acc_a = tl.fma(p_last, la, acc_a)
        acc_b = tl.fma(p_last, lb, acc_b)

        acc_a = acc_a * inv_sum
        acc_b = acc_b * inv_sum
        sum_sq = tl.sum(acc_a * acc_a) + tl.sum(acc_b * acc_b)
        out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
        out_p = Out + q_idx * T * D + row_off
        wa = tl.load(NormW + q_off + da, mask=mask_a, other=0).to(tl.float32)
        wb = tl.load(NormW + q_off + db, mask=mask_b, other=0).to(tl.float32)
        tl.store(out_p + da, (acc_a * out_rstd * wa).to(Out.dtype.element_ty), mask=mask_a)
        tl.store(out_p + db, (acc_b * out_rstd * wb).to(Out.dtype.element_ty), mask=mask_b)


def fused_sconv_attn_res_triton(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    conv_out: torch.Tensor,
    conv_res: torch.Tensor | None,
    conv_addend: torch.Tensor | None,
    stack: torch.Tensor,
    mix_last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    norm_eps: float,
    normalize_values: bool,
    early_pdl: bool,
    nl_buf: torch.Tensor | None,
) -> torch.Tensor:
    T, D = x.shape
    nq = queries.shape[0]
    n_stack = stack.shape[0]
    out = torch.empty((nq, T, D), dtype=x.dtype, device=x.device)
    block_a = triton.next_power_of_2(D + 1) // 2
    block_b = max(triton.next_power_of_2(D - block_a), 16)
    _fused_sconv_ar_triton_kernel[(T,)](
        x, cache, conv_weight, conv_out,
        conv_res if conv_res is not None else x,
        conv_addend if conv_addend is not None else x,
        positions, seq_idx, slot_mapping, block_table, query_start,
        stack, mix_last if mix_last is not None else x, queries, norm_weight, out,
        nl_buf if nl_buf is not None else seq_idx,
        T, x.stride(0), cache.stride(0), cache.stride(2),
        conv_weight.stride(0), conv_weight.stride(1),
        block_table.stride(0), block_table.shape[1], cache.shape[2],
        stack.stride(0), (conv_weight.shape[1] + 1) ** -0.5, eps, norm_eps,
        OFF=stream_offset, D=D, W=conv_weight.shape[1],
        NQ=nq, N_STACK=n_stack,
        HAS_CONV_RES=conv_res is not None,
        HAS_CONV_ADDEND=conv_addend is not None,
        HAS_MIX_LAST=mix_last is not None,
        NORMALIZE=normalize_values,
        BLOCK_N=triton.next_power_of_2(n_stack + 1),
        BLOCK_A=block_a, BLOCK_B=block_b,
        EARLY=early_pdl, HAS_NLIVE=nl_buf is not None,
        num_warps=8, enable_fp_fusion=False, launch_pdl=True,
    )
    return out
