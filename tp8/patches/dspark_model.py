# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft (``mtp.safetensors``): an anchor plus mask block run
through a small dense transformer over DFlare-mixed target-feature K/V, then a
recurrent (``rnn_hidden``) Markov head. Numerics follow the reference
implementation's draft (flashinfer RMSNorms, bf16 logits)."""

import os
from collections.abc import Iterable
from typing import cast

import torch
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

from .configs import Inkling2DSparkConfig
from .ops.attention import block_decode_attention, plan_decode_attention
from .ops import dspark as dspark_ops
from .ops.dspark import (
    Inkling2DSparkArgmax,
    add_softcap,
    dspark_argmax,
    embed_block_ln_fused,
    gathered_fold_argmax_fused,
    markov_gate,
    markov_rnn_step_fused,
    mix_rmsnorm,
    qk_norm_rope_,
    qk_norm_rope_write_kv_,
    rope_,
    silu_and_mul,
)
from .ops import l2pf
from .ops.linear import linear
from .ops.skinny_fp8 import quantize_rows, skinny_fp8_linear
from .ops.skinny_fp8 import supported as fp8_supported
from .ops.skinny_int4 import (
    h64_nf3_supported,
    prealloc_stream_scratch,
    quantize_h64_nf3_panel,
    quantize_nvfp4,
    quantize_u4_panel,
    quantize_uint4,
    skinny_h64_nf3_linear,
    skinny_nvfp4_linear,
    skinny_u4_panel_linear,
    skinny_uint4_linear,
    supported as int4_supported,
    u4_panel_supported,
)
from .speculator_config import Inkling2DSparkSpeculatorConfig


# INKLING_DRAFT_FP8=1: the drafter's dense weights (qkv, o_proj, gate_up, down of every layer) are
# re-quantized at load to float8_e4m3fn with one fp32 scale per output row and read by the W8A16
# skinny kernel (half the bytes). The bf16 copies are kept unless INKLING_DRAFT_FP8_FREE=1. Target outputs are unchanged (greedy
# verification); only draft quality, hence acceptance length, can move.
# INKLING_DRAFT_HEAD_FP8=1: the drafter's base logits (shared lm_head GEMM) read a private fp8 copy
# of the lm_head shard; the Markov fold keeps the bf16 head, and the target's head is untouched.
_DRAFT_FP8 = os.environ.get("INKLING_DRAFT_FP8", "0") not in ("0", "", "off")
_DRAFT_HEAD_FP8 = os.environ.get("INKLING_DRAFT_HEAD_FP8", "0") not in ("0", "", "off")
_PANEL_4BIT_MODES = ("u4p32", "u4p64", "i4p32", "i4p64")
_NF3_MODES = ("h64_nf3p32", "h64_nf3p64")
_ALL_4BIT_MODES = ("off", "0", "", "uint4", "nvfp4") + _PANEL_4BIT_MODES + _NF3_MODES
_DRAFT_4BIT = os.environ.get("INKLING_DRAFT_4BIT", "off").strip().lower()
if _DRAFT_4BIT not in _ALL_4BIT_MODES:
    raise ValueError(f"INKLING_DRAFT_4BIT={_DRAFT_4BIT!r}: expected off, uint4, nvfp4, u4p32, u4p64, i4p32, i4p64, h64_nf3p32 or h64_nf3p64")
if _DRAFT_4BIT in ("0", ""):
    _DRAFT_4BIT = "off"
_DRAFT_MLP_4BIT = os.environ.get("INKLING_DRAFT_MLP_4BIT", _DRAFT_4BIT).strip().lower()
if _DRAFT_MLP_4BIT in ("0", ""):
    _DRAFT_MLP_4BIT = "off"
if _DRAFT_MLP_4BIT not in _ALL_4BIT_MODES:
    raise ValueError(f"INKLING_DRAFT_MLP_4BIT={_DRAFT_MLP_4BIT!r}: invalid mode")
_DRAFT_HEAD_4BIT = os.environ.get("INKLING_DRAFT_HEAD_4BIT", "off").strip().lower()
if _DRAFT_HEAD_4BIT not in _ALL_4BIT_MODES:
    raise ValueError(f"INKLING_DRAFT_HEAD_4BIT={_DRAFT_HEAD_4BIT!r}: expected off, uint4, nvfp4, u4p32, u4p64, i4p32 or i4p64")
if _DRAFT_HEAD_4BIT in ("0", ""):
    _DRAFT_HEAD_4BIT = "off"
_DRAFT_RNN_4BIT = os.environ.get(
    "INKLING_DRAFT_RNN_4BIT",
    _DRAFT_4BIT if _DRAFT_4BIT in _PANEL_4BIT_MODES else "off",
).strip().lower()
if _DRAFT_RNN_4BIT in ("0", ""):
    _DRAFT_RNN_4BIT = "off"
# Balanced 8-rank draft vocabulary pruning (0 = disabled; e.g. 65536 = 8192/rank -> -224 us/step!).
# Target verification stays over full 201,024 vocab -> 100% 0-ULP bit-exact!
_DRAFT_VOCAB_HOT = int(os.environ.get("INKLING_DRAFT_VOCAB_HOT", "0") or "0")
_DRAFT_VOCAB_FILE = os.environ.get("INKLING_DRAFT_VOCAB_FILE", "").strip()
_DRAFT_FUSE_SILU = os.environ.get("INKLING_DRAFT_FUSE_SILU", "1").strip().lower() not in ("0", "", "off", "false", "no")
_DRAFT_FUSE_RNN = os.environ.get("INKLING_DRAFT_FUSE_RNN", "1").strip().lower() not in ("0", "", "off", "false", "no")
_DRAFT_FUSE_EMBED = os.environ.get("INKLING_DRAFT_FUSE_EMBED", "1").strip().lower() not in ("0", "", "off", "false", "no")
_DRAFT_FUSE_KV = os.environ.get("INKLING_DRAFT_FUSE_KV", "0").strip().lower() not in ("0", "", "off", "false", "no")
_DRAFT_PDL = os.environ.get("INKLING_DRAFT_PDL", "0").strip().lower() not in ("0", "", "off", "false", "no")
_DRAFT_KVCTX_FP8 = os.environ.get("INKLING_DRAFT_KVCTX_FP8", "1" if _DRAFT_FP8 else "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_MARKOV_FOLD_FP8 = os.environ.get("INKLING_MARKOV_FOLD_FP8", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_ROPE_OTF = os.environ.get("INKLING_DRAFT_ROPE_OTF", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_4BIT_FREE = os.environ.get("INKLING_DRAFT_4BIT_FREE", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_MARKOV_FREE = os.environ.get("INKLING_DRAFT_MARKOV_FREE", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# INKLING_DRAFT_HEAD=mxint8: the same base logits read the MXINT8 table (int8 + 2^e per 32, 0.87% weight error)
# built for the certified target head (ops/cert_head.py, 123 MB/rank, ONE copy shared with INKLING_CERT_HEAD);
# it replaces the private e4m3 copy of INKLING_DRAFT_HEAD_FP8 (which is then not built).
_DRAFT_HEAD_MXINT8 = os.environ.get("INKLING_DRAFT_HEAD", "").strip().lower() in ("mxint8", "int8", "1", "on", "true")
# INKLING_DRAFT_GLOBAL_WINDOW: convert the drafter's global attention layer ([2])
# into a sliding-window layer with this window size (default: 1024, matching the
# other 4 drafter layers so all 5 drafter layers collapse into 1 single KV cache
# group and eliminate 512 MiB/rank of 1M global KV + the group_size=9 padding tax).
# Set to "0" or "off" to keep drafter layer [2] as full 1M global attention.
# Target model outputs remain 100% bit-exact regardless of drafter windowing.
_DRAFT_GLOBAL_WINDOW_RAW = os.environ.get("INKLING_DRAFT_GLOBAL_WINDOW", "1024").strip().lower()
if _DRAFT_GLOBAL_WINDOW_RAW in ("0", "", "off", "false", "no", "none", "global"):
    _DRAFT_GLOBAL_WINDOW = 0
else:
    _DRAFT_GLOBAL_WINDOW = int(_DRAFT_GLOBAL_WINDOW_RAW)
    if _DRAFT_GLOBAL_WINDOW < 0:
        raise ValueError(
            f"INKLING_DRAFT_GLOBAL_WINDOW={_DRAFT_GLOBAL_WINDOW_RAW!r}: expected >= 0 or 'off'"
        )
_FP8_MAX_ROWS = 64  # skinny_fp8 handles up to 64 rows; larger drafter batches keep bf16 weights
# C=32 High-M FP8 Tensor-Core Drafter Path (opt-in, default OFF):
# At M < _DRAFT_FP8_HIGH_M_MIN (incl. C=1 M=8 and B<=8), drafter stays 100% on 4-bit u4p32/i4p32.
# At M >= _DRAFT_FP8_HIGH_M_MIN (default 96, i.e. B=12..32), switches dense+lm_head GEMMs from
# software-dequant _u4_panel_mt_kernel (7.22 ms @ M=256!) to SM121a rowwise FP8 torch._scaled_mm (~2.1 ms!).
_DRAFT_FP8_HIGH_M = os.environ.get("INKLING_DRAFT_FP8_HIGH_M", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_FP8_HIGH_M_MIN = int(os.environ.get("INKLING_DRAFT_FP8_HIGH_M_MIN", "96") or "96")
_DRAFT_FP8_HIGH_M_HEAD = os.environ.get("INKLING_DRAFT_FP8_HIGH_M_HEAD", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_L2PF_DRAFT_MAX_M = int(os.environ.get("INKLING_L2PF_DRAFT_MAX_M", "32") or "32")
_L2PF_DRAFT_ATTN_MAX_M = int(os.environ.get("INKLING_L2PF_DRAFT_ATTN_MAX_M", "16") or "16")
_DRAFT_FP8_HI_CAP: tuple[str, bool] | None = None


@torch.no_grad()
def _quantize_weight_fp8_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize BF16 weight [N, K] -> (w_fp8 [N_pad, K] e4m3fn, w_scale [1, N_pad] fp32, n_orig)."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    n_pad = (n + 15) & ~15
    w_fp8 = torch.zeros((n_pad, k), dtype=torch.float8_e4m3fn, device=weight.device)
    w_scale = torch.ones((1, n_pad), dtype=torch.float32, device=weight.device)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        wf = weight[start:end].float()
        amax = wf.abs().amax(dim=1)
        sc = torch.clamp(amax * (1.0 / 448.0), min=1e-12)
        w_fp8[start:end].copy_(torch.clamp(wf / sc[:, None], -448.0, 448.0).to(torch.float8_e4m3fn))
        w_scale[0, start:end].copy_(sc)
    return w_fp8.contiguous(), w_scale.contiguous(), n


def _ensure_draft_fp8_hi_cap(device: torch.device) -> tuple[str, bool]:
    global _DRAFT_FP8_HI_CAP
    if _DRAFT_FP8_HI_CAP is not None:
        return _DRAFT_FP8_HI_CAP
    from .ops.linear import _probe_fp8_cap

    _DRAFT_FP8_HI_CAP = _probe_fp8_cap(device)
    return _DRAFT_FP8_HI_CAP


def _can_fp8_high_m(x: torch.Tensor, module: nn.Module) -> bool:
    return (
        _DRAFT_FP8_HIGH_M
        and getattr(module, "fp8_hi_wt", None) is not None
        and x.dim() == 2
        and x.shape[0] >= _DRAFT_FP8_HIGH_M_MIN
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and (x.shape[1] & 15) == 0
        and x.is_cuda
    )


def _fp8_high_m_linear(
    x: torch.Tensor,
    w_fp8_t: torch.Tensor,
    w_scale: torch.Tensor,
    n_orig: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    cap = _DRAFT_FP8_HI_CAP
    if cap is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        cap = _ensure_draft_fp8_hi_cap(x.device)
    mode, supports_out = cap
    if mode != "rowwise":
        return None
    from .ops.linear import quantize_fp8_rowwise

    m = int(x.shape[0])
    m_pad = (m + 15) & ~15
    n_pad = int(w_fp8_t.shape[1])
    x_fp8, x_scale = quantize_fp8_rowwise(x, pad_m16=True)
    can_direct_out = (
        out is not None
        and supports_out
        and m_pad == m
        and n_pad == n_orig
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
    )
    if can_direct_out:
        torch._scaled_mm(
            x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale,
            out_dtype=torch.bfloat16, use_fast_accum=False, out=out,
        )
        return out
    res = torch._scaled_mm(
        x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale,
        out_dtype=torch.bfloat16, use_fast_accum=False,
    )
    if m_pad != m or n_pad != n_orig:
        res = res[:m, :n_orig]
    if out is not None:
        out.copy_(res)
        return out
    return res


def _draft_sbuf(shape, dtype: torch.dtype):
    from vllm.distributed.device_communicators.rdma_all_reduce import get_send_buffer
    try:
        return get_send_buffer(shape, dtype, domain=1)
    except TypeError:
        return get_send_buffer(shape, dtype)


def _dlinear(x: torch.Tensor, module: nn.Module, out: torch.Tensor | None = None, rotate_x: bool = True) -> torch.Tensor:
    """``x @ module.weight.T`` through high-M FP8 TC (M >= 96), 3-bit NF3, 4-bit or fp8 copy when quantized."""
    if _can_fp8_high_m(x, module):
        y = _fp8_high_m_linear(x, module.fp8_hi_wt, module.fp8_hi_sc, module.fp8_hi_n, out=out)
        if y is not None:
            return y
    w4_meta = getattr(module, "w4_meta", None)
    if w4_meta is not None:
        if h64_nf3_supported(x, w4_meta):
            return skinny_h64_nf3_linear(
                x, module.w4, module.w4_s1, module.w4_s2, w4_meta,
                out=out, rotate_x=rotate_x, pdl=_DRAFT_PDL,
            )
        if u4_panel_supported(x, w4_meta):
            return skinny_u4_panel_linear(
                x, module.w4, module.w4_s1, module.w4_s2, w4_meta, out=out, pdl=_DRAFT_PDL
            )
    w4 = getattr(module, "w4", None)
    if w4 is not None and int4_supported(x, w4):
        if getattr(module, "w4_mode", "") == "nvfp4":
            return skinny_nvfp4_linear(x, w4, module.w4_s1, module.w4_s2, out=out)
        return skinny_uint4_linear(x, w4, module.w4_s1, module.w4_s2, out=out)
    w8 = getattr(module, "w8", None)
    if w8 is not None and fp8_supported(x, w8):
        return skinny_fp8_linear(x, w8, module.w8_scale, out=out)
    return linear(x, module.weight, out=out)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
        # In place: residual += x; x = norm(residual), from the unrounded sum.
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x


_DRAFT_FUSED_AR_NORM = os.environ.get("INKLING_DRAFT_FUSED_AR_NORM", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_AR_INT8 = os.environ.get("INKLING_DRAFT_AR_INT8", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# dec160: INKLING_DRAFT_AR_MODE=bf16|int8|h64_q8|h64_q6|h64_q4 (overrides INKLING_DRAFT_AR_INT8)
_DRAFT_AR_MODE = os.environ.get(
    "INKLING_DRAFT_AR_MODE", "int8" if _DRAFT_AR_INT8 else "bf16"
).strip().lower()
_DRAFT_AR_C32_MODE = os.environ.get("INKLING_DRAFT_AR_C32_MODE", "off").strip().lower()


def all_reduce_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, norm: _RMSNorm, out_h64: bool = False,
) -> torch.Tensor:
    """In place: ``residual += all_reduce(x); x = norm(residual)``."""
    if _DRAFT_FUSED_AR_NORM or (_DRAFT_AR_C32_MODE not in ("", "0", "off", "false", "no", "bf16") and x.shape[0] >= 16):
        rdma_comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
        if rdma_comm is not None and hasattr(rdma_comm, "all_reduce_add_rmsnorm"):
            try:
                out = rdma_comm.all_reduce_add_rmsnorm(
                    x, residual, norm.weight, norm.eps,
                    use_int8=(_DRAFT_AR_MODE == "int8"),
                    ar_mode=_DRAFT_AR_MODE,
                    out_h64=out_h64,
                    c32_mode=_DRAFT_AR_C32_MODE,
                )
            except TypeError:
                out = (
                    None
                    if out_h64
                    else rdma_comm.all_reduce_add_rmsnorm(
                        x, residual, norm.weight, norm.eps, use_int8=_DRAFT_AR_INT8,
                    )
                )
            if out is not None:
                return out
    fi_ar = getattr(get_tp_group().device_communicator, "fi_ar_comm", None)
    if fi_ar is None or not fi_ar.should_use_fi_ar(x):
        return norm(tensor_model_parallel_all_reduce(x), residual)
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        flashinfer_comm,
        flashinfer_trtllm_fused_allreduce_norm,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )

    workspace = get_fi_ar_workspace(
        fi_ar.world_size,
        fi_ar.rank,
        fi_ar.max_num_tokens,
        x.shape[1],
        x.dtype,
        fi_ar.group,
    )
    assert flashinfer_comm is not None and workspace is not None
    assert workspace.backend == "mnnvl"
    assert x.dtype == residual.dtype == norm.weight.dtype == torch.bfloat16
    assert x.shape == residual.shape and residual.is_contiguous()
    flashinfer_trtllm_fused_allreduce_norm(
        allreduce_in=x,
        residual=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.eps,
        world_size=fi_ar.world_size,
        launch_with_pdl=True,
        fp32_acc=True,
        max_token_num=fi_ar.max_num_tokens,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
    )
    return x


class Inkling2DraftSelfAttention(nn.Module):
    """Bidirectional block attention over context K/V (from target features)
    and the block itself; local layers see ``sliding_window_size`` positions
    before the block start."""

    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= config.num_key_value_heads
        hidden, self.head_dim = config.hidden_size, config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        self.block = config.draft_block_size
        self.is_local = is_local
        effective_window = (
            config.sliding_window_size
            if is_local
            else (_DRAFT_GLOBAL_WINDOW if _DRAFT_GLOBAL_WINDOW > 0 else None)
        )
        self._effective_is_local = effective_window is not None
        self.qkv_proj = QKVParallelLinear(
            hidden,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.kv_proj_ctx = MergedColumnParallelLinear(
            hidden,
            [config.num_key_value_heads * self.head_dim] * 2,
            bias=False,
            prefix=f"{prefix}.kv_proj_ctx",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            hidden,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm_ctx = _RMSNorm(self.head_dim, self.eps)
        # Owns the paged KV cache only; the same spec as the target's layers
        # (the scheduler keeps the last sliding_window_size positions before
        # the block, which is what the draft reads).
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=effective_window,
            prefix=f"{prefix}.attn",
            attn_backend=FlashAttentionBackend,
        )
        # Keys a block sees: itself and, if windowed, the window before it.
        self.window = (
            effective_window + self.block if effective_window is not None else 0
        )
        vllm_config = get_current_vllm_config()
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.register_buffer(
            "cu_seqlens_q",
            torch.arange(max_num_seqs + 1, dtype=torch.int32) * self.block,
            persistent=False,
        )

    def _metadata(self) -> FlashAttentionMetadata | None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or self.attn.kv_cache.numel() == 0:
            return None
        return cast(FlashAttentionMetadata, attn_metadata[self.attn.layer_name])

    def _write_kv(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)

    def _can_fuse_kv(
        self, x: torch.Tensor, slot_mapping: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not _DRAFT_FUSE_KV
            or slot_mapping is None
            or self.attn.kv_cache.numel() == 0
            or x.stride(1) != 1
            or slot_mapping.shape[0] < x.shape[0]
        ):
            return None
        key_cache, value_cache = self.attn.kv_cache.transpose(1, 2).split(
            self.head_dim, dim=-1
        )
        if (
            key_cache.ndim != 4
            or key_cache.dtype != x.dtype
            or value_cache.dtype != x.dtype
            or key_cache.stride(-1) != 1
            or value_cache.stride(-1) != 1
            or key_cache.shape[-2] != self.num_kv_heads
            or key_cache.shape[-1] != self.head_dim
        ):
            return None
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_idx: int = -1,
        ctx_sched=None,
    ) -> torch.Tensor:
        qkv = _dlinear(x, self.qkv_proj)
        md = self._metadata()
        kv_fused = (
            self._can_fuse_kv(qkv, md.slot_mapping[: qkv.shape[0]])
            if md is not None
            else None
        )
        if kv_fused is not None:
            key_cache, value_cache = kv_fused
            qk_norm_rope_write_kv_(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                self.num_heads,
                self.num_kv_heads,
                self.eps,
                key_cache,
                value_cache,
                md.slot_mapping[: qkv.shape[0]],
                pdl=_DRAFT_PDL,
            )
            q = qkv.narrow(-1, 0, self.q_size)
            out = q.new_empty((q.shape[0], self.q_size))
        else:
            qk_norm_rope_(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                self.num_heads,
                self.num_kv_heads,
                self.eps,
                pdl=_DRAFT_PDL,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            if md is not None:
                out = q.new_empty((q.shape[0], self.q_size))
                self._write_kv(
                    k.view(-1, self.num_kv_heads, self.head_dim),
                    v.view(-1, self.num_kv_heads, self.head_dim),
                    md.slot_mapping[: q.shape[0]],
                )
                key_cache, value_cache = self.attn.kv_cache.transpose(1, 2).split(
                    self.head_dim, dim=-1
                )
                key_cache = canonicalize_singleton_dim_strides(key_cache)
                value_cache = canonicalize_singleton_dim_strides(value_cache)
        if md is not None:
            num_reqs = min(q.shape[0] // self.block, md.seq_lens.shape[0])
            num_tokens = num_reqs * self.block
            if num_tokens < q.shape[0]:
                out[num_tokens:].zero_()
            if l2pf.DRAFT and 0 < num_tokens <= _L2PF_DRAFT_ATTN_MAX_M:
                l2pf.fork(_draft_pf_weight(self.o_proj, num_tokens), kind="draft_attn")
            # Ensure this layer's overlapped context K/V write is visible before attention reads KV cache.
            if ctx_sched is not None and layer_idx >= 0:
                ctx_sched.join_layer(layer_idx)
            # The plan's 0 (FA4, for very large target batches) is one split
            # here.
            num_splits = max(
                1,
                plan_decode_attention(
                    self._effective_is_local,
                    num_reqs,
                    self.block,
                    self.num_heads,
                    self.num_kv_heads,
                    self.window,
                ),
            )
            block_decode_attention(
                q[:num_tokens].view(num_tokens, self.num_heads, self.head_dim),
                key_cache,
                value_cache,
                out[:num_tokens],
                block_table=md.block_table[:num_reqs],
                cache_seqlens=md.seq_lens[:num_reqs],
                cu_seqlens_q=self.cu_seqlens_q[: num_reqs + 1],
                block=self.block,
                window=self.window,
                num_splits=num_splits,
            )
            l2pf.join()
        else:
            if ctx_sched is not None and layer_idx >= 0:
                ctx_sched.join_layer(layer_idx)
            out = q.new_zeros((q.shape[0], self.q_size))
        sbuf = _draft_sbuf((out.shape[0], self.o_proj.weight.shape[0]), out.dtype)
        return _dlinear(out, self.o_proj, out=sbuf)

    def store_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        kv = _dlinear(context, self.kv_proj_ctx)
        kv_fused = self._can_fuse_kv(kv, slot_mapping)
        if kv_fused is not None:
            key_cache, value_cache = kv_fused
            qk_norm_rope_write_kv_(
                kv,
                positions,
                self.k_norm_ctx.weight,
                self.k_norm_ctx.weight,
                cos_sin_cache,
                0,
                self.num_kv_heads,
                self.eps,
                key_cache,
                value_cache,
                slot_mapping,
                pdl=_DRAFT_PDL,
            )
            return
        # Fuse k_norm_ctx + neox RoPE in-place on the leading num_kv_heads of kv (eliminates 5x RMSNorm launches!)
        qk_norm_rope_(
            kv,
            positions,
            self.k_norm_ctx.weight,
            self.k_norm_ctx.weight,
            cos_sin_cache,
            0,
            self.num_kv_heads,
            self.eps,
            pdl=_DRAFT_PDL,
        )
        if slot_mapping is not None and self.attn.kv_cache.numel() > 0:
            k, v = kv.split([self.kv_size, self.kv_size], -1)
            self._write_kv(
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                slot_mapping,
            )


class Inkling2DraftMLP(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w4_meta = getattr(self.gate_up_proj, "w4_meta", None)
        use_fp8_hi = _can_fp8_high_m(x, self.gate_up_proj)
        if not use_fp8_hi and w4_meta is not None and h64_nf3_supported(x, w4_meta) and w4_meta[4]:
            x = skinny_h64_nf3_linear(
                x, self.gate_up_proj.w4, self.gate_up_proj.w4_s1, self.gate_up_proj.w4_s2,
                w4_meta, rotate_x=True, pdl=_DRAFT_PDL,
            )
        elif (
            not use_fp8_hi
            and w4_meta is not None
            and len(w4_meta) >= 10
            and w4_meta[9]
            and u4_panel_supported(x, w4_meta)
        ):
            # Fused gate_up + fp32 silu_and_mul epilogue -> [M, intermediate // tp] directly!
            x = skinny_u4_panel_linear(
                x, self.gate_up_proj.w4, self.gate_up_proj.w4_s1, self.gate_up_proj.w4_s2, w4_meta,
                pdl=_DRAFT_PDL,
            )
        else:
            x = silu_and_mul(_dlinear(x, self.gate_up_proj))
        sbuf = _draft_sbuf((x.shape[0], self.down_proj.weight.shape[0]), x.dtype)
        return _dlinear(x, self.down_proj, out=sbuf)


def _draft_pf_weight(module: nn.Module, m: int = 0):
    """Best weight tensor(s) to L2-prefetch for a drafter linear module (high-M FP8 or panel-matched w4)."""
    if _DRAFT_FP8_HIGH_M and m >= _DRAFT_FP8_HIGH_M_MIN:
        w_hi = getattr(module, "fp8_hi_w", None)
        if w_hi is not None:
            return w_hi
    w = getattr(module, "w4", None)
    if w is not None:
        if l2pf.G2 and getattr(module, "w4_meta", None) is not None:
            s1 = getattr(module, "w4_s1", None)
            s2 = getattr(module, "w4_s2", None)
            if s1 is not None:
                # For large matrices (e.g. gate_up 10.6 MB), cap each scale segment at 768 KB (~58% of
                # panels = ~6.1 MB w4) so scales stay in exact lockstep with the leading w4 panels.
                w_bytes = w.numel() * w.element_size()
                sc_cap = 786432 if w_bytes > (4 << 20) else None
                segs = [(s1, sc_cap) if sc_cap is not None else s1]
                if s2 is not None:
                    segs.append((s2, sc_cap) if sc_cap is not None else s2)
                segs.append(w)
                return tuple(segs)
        return w
    w = getattr(module, "w8", None)
    if w is not None:
        return w
    return getattr(module, "weight", None)


class Inkling2DraftLayer(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _RMSNorm(hidden, eps)
        self.post_attention_layernorm = _RMSNorm(hidden, eps)
        self.dflare_norm = _RMSNorm(hidden, eps)
        self.self_attn = Inkling2DraftSelfAttention(
            config, is_local, f"{prefix}.self_attn"
        )
        self.mlp = Inkling2DraftMLP(config, f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
        pre_normed: bool = False,
        layer_idx: int = -1,
        ctx_sched=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns this rank's partial MLP output and the residual."""
        if pre_normed:
            assert residual is not None
        elif residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            if l2pf.DRAFT and x.shape[0] <= _L2PF_DRAFT_MAX_M:
                l2pf.fork(_draft_pf_weight(self.self_attn.qkv_proj, x.shape[0]), kind="draft")
            # Overlap L{i}.ctx_kv (~17 us, 0.62 MB) on ctx_stream behind L{i-1}.down_proj AR (~35 us)
            # + L{i}.qkv_proj (~16 us); joined right before L{i}.block_decode_attention.
            if ctx_sched is not None and layer_idx >= 0:
                ctx_sched.fork_layer(layer_idx)
            x = all_reduce_add_rmsnorm(x, residual, self.input_layernorm)
            l2pf.join()
        x = self.self_attn(
            positions, x, cos_sin_cache, layer_idx=layer_idx, ctx_sched=ctx_sched
        )
        if l2pf.DRAFT and x.shape[0] <= _L2PF_DRAFT_MAX_M:
            l2pf.fork(_draft_pf_weight(self.mlp.gate_up_proj, x.shape[0]), kind="draft")
        x = all_reduce_add_rmsnorm(x, residual, self.post_attention_layernorm)
        l2pf.join()
        return self.mlp(x), residual


class Inkling2MarkovHead(nn.Module):
    """Recurrent (``rnn_hidden``) correction of the draft's base logits.

    Its rank-``r`` latent goes through ``hidden_out`` and the shared LM head,
    folded once into a ``(vocab, r)`` projection. The ``[hidden, f_rnn]``
    columns of ``rnn_proj`` (and its bias) are applied to all draft steps at
    once in fp32; each step adds the ``[state, previous]`` columns and rounds.
    """

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size,
            rank,
            _weight=torch.empty(config.padded_vocab_size, rank),
            _freeze=True,
        )
        self.rnn_proj = ReplicatedLinear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = ReplicatedLinear(rank, hidden, bias=True)
        self.dflare_norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.fold_weight: torch.Tensor | None = None
        self.fold_bias: torch.Tensor | None = None
        self.rnn_bias: torch.Tensor | None = None
        self.rnn_w_rec: torch.Tensor | None = None
        self.fold_w8: torch.Tensor | None = None
        self.fold_s8: torch.Tensor | None = None
        self.rnn_w4: torch.Tensor | None = None
        self.rnn_s1: torch.Tensor | None = None
        self.rnn_s2: torch.Tensor | None = None
        self.rnn_meta: tuple | None = None

    @torch.no_grad()
    def fold(self, lm_head_weight: torch.Tensor) -> None:
        """Fold ``hidden_out`` into the (TP-local) LM head rows in fp64."""
        projection = self.hidden_out.weight.double()
        bias = self.hidden_out.bias.double()
        num_rows = lm_head_weight.shape[0]
        self.fold_weight = lm_head_weight.new_empty((num_rows, self.markov_rank))
        self.fold_bias = lm_head_weight.new_empty((num_rows,))
        for start in range(0, num_rows, 4096):
            rows = slice(start, start + 4096)
            head = lm_head_weight[rows].double()
            self.fold_weight[rows].copy_(head @ projection)
            self.fold_bias[rows].copy_(head @ bias)
        self.rnn_w_rec = self.rnn_proj.weight[:, : 2 * self.markov_rank].contiguous()
        if _MARKOV_FOLD_FP8:
            w8 = torch.empty(self.fold_weight.shape, dtype=torch.float8_e4m3fn, device=self.fold_weight.device)
            s8 = torch.empty(num_rows, dtype=torch.float32, device=self.fold_weight.device)
            for start in range(0, num_rows, 4096):
                rows = slice(start, start + 4096)
                w8[rows], s8[rows] = quantize_rows(self.fold_weight[rows])
            self.fold_w8, self.fold_s8 = w8, s8
            print(f"[inkling-markov] fold_weight -> e4m3 per-row {tuple(w8.shape)} ({w8.numel() / 2**20:.1f} MiB)")

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """fp32 ``rnn_proj`` bias plus its ``[hidden, f_rnn]`` term for the
        ``(req, step)``-ordered ``hidden``; returns ``(step, req, 3 * r)``."""
        assert self.rnn_bias is not None
        num_reqs, width = f_rnn.shape
        if (
            self.rnn_meta is not None
            and _DRAFT_FUSE_RNN
            and num_reqs == 1
            and u4_panel_supported(hidden, self.rnn_meta, fuse_rnn=True)
        ):
            # Zero-copy broadcast of f_rnn[0, :] inside K-loop (eliminates torch.cat + contiguous!)
            out = skinny_u4_panel_linear(
                hidden, self.rnn_w4, self.rnn_s1, self.rnn_s2, self.rnn_meta,
                out_dtype=torch.float32, bias=self.rnn_bias, x2=f_rnn, pdl=_DRAFT_PDL,
            )
            return out.view(hidden.shape[0], 1, -1)
        if (
            self.rnn_meta is not None
            and _DRAFT_FUSE_RNN
            and num_reqs >= 2
            and hidden.shape[0] % num_reqs == 0
            and u4_panel_supported(hidden, self.rnn_meta, fuse_rnn=True)
        ):
            # C=32 zero-copy batched FUSE_RNN_BATCH: reads (req, step) hidden & f_rnn[req] directly
            # into (step, req, 3*R) output — eliminates transpose + 4.13 MB torch.cat + contiguous!
            k_steps = hidden.shape[0] // num_reqs
            out = skinny_u4_panel_linear(
                hidden, self.rnn_w4, self.rnn_s1, self.rnn_s2, self.rnn_meta,
                out_dtype=torch.float32, bias=self.rnn_bias, x2=f_rnn, pdl=_DRAFT_PDL,
                n_reqs=num_reqs, k_steps=k_steps,
            )
            return out.view(k_steps, num_reqs, -1)
        hidden = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], dim=-1)
        if self.rnn_meta is not None:
            inp2 = inputs.flatten(0, 1).contiguous()
            if u4_panel_supported(inp2, self.rnn_meta):
                out = skinny_u4_panel_linear(
                    inp2, self.rnn_w4, self.rnn_s1, self.rnn_s2, self.rnn_meta,
                    out_dtype=torch.float32, bias=self.rnn_bias, pdl=_DRAFT_PDL,
                )
                return out.view(hidden.shape[0], num_reqs, -1)
        weight = self.rnn_proj.weight[:, 2 * self.markov_rank :]
        out = torch.addmm(
            self.rnn_bias, inputs.flatten(0, 1), weight.T, out_dtype=torch.float32
        )
        return out.view(hidden.shape[0], num_reqs, -1)

    def step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step; returns the new state and the TP-local bias."""
        assert self.fold_weight is not None and self.fold_bias is not None
        inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
        weight = (
            self.rnn_w_rec
            if self.rnn_w_rec is not None
            else self.rnn_proj.weight[:, : 2 * self.markov_rank]
        )
        raw = torch.addmm(context, inputs, weight.T, out_dtype=torch.float32)
        state, latent = markov_gate(raw, state)
        if (
            self.fold_weight.untyped_storage().size() == 0
            and self.fold_w8 is not None
            and self.fold_s8 is not None
        ):
            fw = (self.fold_w8.float() * self.fold_s8[:, None]).to(latent.dtype)
            return state, torch.addmm(self.fold_bias, latent, fw.T)
        return state, torch.addmm(self.fold_bias, latent, self.fold_weight.T)


class _CtxOverlapSchedule:
    """Staggered side-stream schedule for drafter ctx_kv (L0..L4) + f_rnn inside a FULL draft graph.

    All CUDA streams/events are pre-allocated outside capture by Inkling2DSparkSpeculator.capture().
    Every side-stream output lands in a persistent buffer (paged KV cache / speculator.f_rnn),
    so no temporary tensor crosses streams and all forks/joins live inside one captured graph.
    Exact same kernels on exact same inputs -> 0-ULP target & drafter identical."""

    __slots__ = (
        "decoder",
        "context_states",
        "context_positions",
        "context_slots",
        "frnn_fn",
        "stream",
        "ev_in",
        "ev_out",
        "n_layers",
        "forked",
        "joined",
    )

    def __init__(
        self,
        decoder: "Inkling2DSparkDecoder",
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slots: list[torch.Tensor | None],
        frnn_fn,
        stream: torch.cuda.Stream,
        ev_in: list[torch.cuda.Event],
        ev_out: list[torch.cuda.Event],
    ) -> None:
        self.decoder = decoder
        self.context_states = context_states
        self.context_positions = context_positions
        self.context_slots = context_slots
        self.frnn_fn = frnn_fn
        self.stream = stream
        self.ev_in = ev_in
        self.ev_out = ev_out
        self.n_layers = len(decoder.layers)
        self.forked = [False] * (self.n_layers + 1)
        self.joined = [False] * (self.n_layers + 1)

    def fork_layer(self, idx: int) -> None:
        if self.forked[idx]:
            return
        self.forked[idx] = True
        cur = torch.cuda.current_stream(self.context_states.device)
        self.ev_in[idx].record(cur)
        self.stream.wait_event(self.ev_in[idx])
        with torch.cuda.stream(self.stream):
            layer = self.decoder.layers[idx]
            context = self.decoder.mix(self.context_states, idx, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context,
                self.context_positions,
                self.decoder.cos_sin_cache,
                self.context_slots[idx],
            )
            self.ev_out[idx].record(self.stream)

    def join_layer(self, idx: int) -> None:
        if not self.forked[idx]:
            self.fork_layer(idx)
        if self.joined[idx]:
            return
        self.joined[idx] = True
        torch.cuda.current_stream(self.context_states.device).wait_event(self.ev_out[idx])

    def fork_frnn(self) -> None:
        idx = self.n_layers
        if self.forked[idx]:
            return
        self.forked[idx] = True
        cur = torch.cuda.current_stream(self.context_states.device)
        self.ev_in[idx].record(cur)
        self.stream.wait_event(self.ev_in[idx])
        with torch.cuda.stream(self.stream):
            self.frnn_fn()
            self.ev_out[idx].record(self.stream)

    def join_frnn(self) -> None:
        idx = self.n_layers
        if not self.forked[idx]:
            self.fork_frnn()
        if self.joined[idx]:
            return
        self.joined[idx] = True
        torch.cuda.current_stream(self.context_states.device).wait_event(self.ev_out[idx])

    def join_all(self) -> None:
        """Fallback join guaranteeing every forked layer/f_rnn is complete & closed in the graph."""
        for idx in range(self.n_layers):
            if not self.joined[idx]:
                self.join_layer(idx)
        if not self.joined[self.n_layers]:
            self.join_frnn()


class Inkling2DSparkDecoder(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, max_model_len: int, prefix: str
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(
                config, i in config.local_layer_ids, f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.markov_head = Inkling2MarkovHead(config)
        self.fuse_weight = nn.Parameter(
            torch.zeros(config.num_hidden_layers + 1, len(config.target_layer_ids)),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.register_buffer("fuse_alpha", torch.empty_like(self.fuse_weight))
        # On-the-fly RoPE inv_freq (256 B) vs 512 MiB [1M, 128] fp32 table (+0.500 GiB/rank).
        if _DRAFT_ROPE_OTF:
            inv_freq = 1.0 / (
                float(config.rope_theta)
                ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
            )
            self.register_buffer("cos_sin_cache", inv_freq, persistent=False)
        else:
            self.cos_sin_cache = get_rope(
                config.head_dim,
                max_position=max_model_len,
                rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
                dtype=torch.float32,
            ).cos_sin_cache
        # Shared with the target model (see Inkling2DSparkModel.attach_target).
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def mix(self, features: torch.Tensor, row: int, norm: _RMSNorm) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.config.hidden_size)
        return mix_rmsnorm(targets, self.fuse_alpha[row], norm.weight, norm.eps, pdl=_DRAFT_PDL)

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normalized target embedding of each block's anchor, then masks."""
        assert self.embed_tokens is not None and self.embed_norm is not None
        block = self.config.draft_block_size
        ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        x[:, 0] = self.embed_norm(self.embed_tokens(ids[:, 0].contiguous()))
        return x.view(-1, self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] > 16 and torch.cuda.is_current_stream_capturing():
            from vllm.distributed.device_communicators import rdma_all_reduce as _rar_mod
            _reset_ring = getattr(_rar_mod, "reset_send_buffer_ring", None)
            if _reset_ring is not None:
                _reset_ring(1)
        sched = getattr(self, "_ctx_sched", None)
        if sched is not None and len(self.layers) > 0:
            # Overlap L0.ctx_kv (~17.6 us) behind embed_block_ln_fused + L0.qkv_proj (~29.4 us).
            sched.fork_layer(0)
        # Fused embed_block + embed_norm + L0.input_layernorm (replaces 11 kernels / ~32 us -> ~8.4 us!)
        emb_w = getattr(self.embed_tokens, "_repl_weight", None)
        if emb_w is None:
            emb_w = getattr(self.embed_tokens, "weight", None)
        if (
            _DRAFT_FUSE_EMBED
            and emb_w is not None
            and self.embed_norm is not None
            and emb_w.shape[0] >= self.config.vocab_size
            and emb_w.dtype == torch.bfloat16
            and emb_w.stride(1) == 1
            and len(self.layers) > 0
        ):
            emb_nw = self.embed_norm.weight.view(-1)
            emb_eps = float(getattr(self.embed_norm, "eps", self.config.rms_norm_eps))
            x, residual = embed_block_ln_fused(
                input_ids,
                emb_w,
                emb_nw,
                self.mask_embed,
                self.layers[0].input_layernorm.weight,
                self.config.draft_block_size,
                self.config.rms_norm_eps,
                emb_eps=emb_eps,
                pdl=_DRAFT_PDL,
            )
            for idx, layer in enumerate(self.layers):
                x, residual = layer(
                    positions,
                    x,
                    residual,
                    self.cos_sin_cache,
                    pre_normed=(idx == 0),
                    layer_idx=idx,
                    ctx_sched=sched,
                )
        else:
            x = self.embed_block(input_ids)
            residual = None
            for idx, layer in enumerate(self.layers):
                x, residual = layer(
                    positions,
                    x,
                    residual,
                    self.cos_sin_cache,
                    layer_idx=idx,
                    ctx_sched=sched,
                )
        assert residual is not None
        if l2pf.DRAFT and x.shape[0] <= _L2PF_DRAFT_MAX_M:
            hw = (
                getattr(self, "_l2pf_head_fp8", None)
                if (_DRAFT_FP8_HIGH_M and x.shape[0] >= _DRAFT_FP8_HIGH_M_MIN)
                else None
            )
            if hw is None:
                hw = getattr(self, "_l2pf_head", None)
            if hw is not None:
                # Keep kind="draft" (NOT kind="head") so draft_head L2PF matches drafter AR timing
                # and does not collide with target Layer-47 MAX_HEAD_US.
                l2pf.fork(hw, kind="draft")
        if sched is not None:
            # Overlap prepare_context_anchor / f_rnn (~19 us) behind final AR (~31 us) + base_logits (~302 us).
            sched.fork_frnn()
        out = all_reduce_add_rmsnorm(x, residual, self.norm)
        l2pf.join_end()
        return out


class Inkling2DSparkModel(nn.Module):
    """DSpark draft for Inkling2; plugged in by ``Inkling2DSparkSpeculator``."""

    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = Inkling2DSparkSpeculatorConfig
    allow_patterns_overrides = ["mtp.safetensors"]
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    _stacked = (
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
        (".k_proj_ctx.", ".kv_proj_ctx.", 0),
        (".v_proj_ctx.", ".kv_proj_ctx.", 1),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        assert isinstance(config, Inkling2DSparkConfig)
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        sam_hybrid_on = os.environ.get("INKLING_SAM_HYBRID", "0").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        if num_spec != config.draft_block_size - 1 and not (
            sam_hybrid_on and num_spec in (11, 15) and config.draft_block_size == 8
        ):
            raise ValueError(
                "Inkling2 DSpark requires num_speculative_tokens = "
                f"{config.draft_block_size - 1} (the checkpoint's block size - 1)"
            )
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(
                "Inkling2 DSpark drafts up to max_position_embeddings="
                f"{config.max_position_embeddings} positions; set --max-model-len "
                "to at most that"
            )
        self.config = config
        self.model = Inkling2DSparkDecoder(
            config, vllm_config.model_config.max_model_len, f"{prefix}model"
        )
        self.lm_head: nn.Module | None = None
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.head_w8: torch.Tensor | None = None
        self.head_w8_scale: torch.Tensor | None = None
        self.head_w4: torch.Tensor | None = None
        self.head_w4_s1: torch.Tensor | None = None
        self.head_w4_s2: torch.Tensor | None = None
        self.head_w4_mode: str = "off"
        self.head_w4_meta: tuple | None = None
        self.head_fp8_hi_w: torch.Tensor | None = None
        self.head_fp8_hi_wt: torch.Tensor | None = None
        self.head_fp8_hi_sc: torch.Tensor | None = None
        self.head_fp8_hi_n: int = 0
        self.head_mx = None  # cert_head.CertTable (shared MXINT8 copy of the lm_head shard)
        self.draft_id_map: torch.Tensor | None = None  # [V_local] int32 global IDs when vocab-pruned
        self._hot_ids_global: torch.Tensor | None = None  # [V_hot] int64 sorted global IDs (for temp>0 scatter)

    @torch.no_grad()
    def _build_hot_vocab_local(self) -> torch.Tensor | None:
        """Build balanced TP8 hot-vocab shard W_hot_local [V_local, H] when INKLING_DRAFT_VOCAB_HOT > 0.
        Never mutates target self.lm_head.weight; target verification remains 100% 0-ULP bit-exact!"""
        if _DRAFT_VOCAB_HOT <= 0:
            return None
        assert self.lm_head is not None
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        full_local = int(self.lm_head.weight.shape[0])
        full_vocab = int(self.config.vocab_size)
        dev = self.lm_head.weight.device

        # Load explicit hot IDs from JSON file if provided, else special tail + lowest BPE merge ranks
        if _DRAFT_VOCAB_FILE:
            import json
            with open(_DRAFT_VOCAB_FILE) as f:
                ids_list = sorted(set(int(i) for i in json.load(f) if 0 <= int(i) < full_vocab))
            n_align = (len(ids_list) // (tp_size * 64)) * (tp_size * 64)
            hot_ids = torch.tensor(ids_list[:n_align], dtype=torch.int64, device="cpu")
        else:
            v_hot = min(_DRAFT_VOCAB_HOT, full_vocab)
            v_hot = (v_hot // (tp_size * 64)) * (tp_size * 64)
            if v_hot <= 0 or v_hot >= full_vocab:
                return None
            # Reserve top 256 IDs (special/control/chat tokens) + lowest (v_hot - 256) BPE merge ranks
            n_spec = min(256, v_hot // 16)
            spec_ids = list(range(full_vocab - n_spec, full_vocab))
            base_ids = list(range(0, v_hot - n_spec))
            hot_ids = torch.tensor(sorted(base_ids + spec_ids), dtype=torch.int64, device="cpu")

        v_local = int(hot_ids.numel()) // tp_size
        my_hot = hot_ids[tp_rank * v_local : (tp_rank + 1) * v_local].to(dev)
        w_hot_local = torch.empty((v_local, self.lm_head.weight.shape[1]), dtype=torch.bfloat16, device=dev)

        # Chunked broadcast from each rank owning any needed global IDs (<35 ms, <75 MiB peak)
        slab = 4096
        buf = torch.empty((slab, self.lm_head.weight.shape[1]), dtype=torch.bfloat16, device=dev)
        tp_grp = get_tp_group().device_group
        for src in range(tp_size):
            lo_g, hi_g = src * full_local, (src + 1) * full_local
            if not bool(((hot_ids >= lo_g) & (hot_ids < hi_g)).any()):
                continue
            src_mine = (my_hot >= lo_g) & (my_hot < hi_g)
            has_mine = bool(src_mine.any())
            for off in range(0, full_local, slab):
                n = min(slab, full_local - off)
                if tp_rank == src:
                    buf[:n].copy_(self.lm_head.weight[off : off + n])
                torch.distributed.broadcast(buf[:n], src=src, group=tp_grp)
                if has_mine:
                    s_lo, s_hi = lo_g + off, lo_g + off + n
                    hit = (my_hot >= s_lo) & (my_hot < s_hi)
                    if bool(hit.any()):
                        w_hot_local[hit] = buf[my_hot[hit] - s_lo]
        del buf
        self.draft_id_map = my_hot.to(torch.int32).contiguous()
        self._hot_ids_global = hot_ids.to(dev)
        if tp_rank == 0:
            print(
                f"[inkling-draft-vocab] balanced TP{tp_size} pruned draft vocab: "
                f"{hot_ids.numel()}/{full_vocab} ({100.0 * hot_ids.numel() / full_vocab:.1f}%, "
                f"{v_local} rows/rank, source={'file:' + _DRAFT_VOCAB_FILE if _DRAFT_VOCAB_FILE else 'bpe_prefix+special'})"
            )
        return w_hot_local

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """Finish sharing the target's normalized embeddings and LM head (its
        ``embed_tokens`` and ``lm_head`` are aliased by ``load_dspark_model``)
        and derive the weights that depend on them."""
        assert self.lm_head is not None and self.lm_head.weight.dtype == torch.bfloat16
        self.model.embed_norm = target_model.model.embed_norm
        head = self.model.markov_head
        w_draft_head = self._build_hot_vocab_local()
        head_src = w_draft_head if w_draft_head is not None else self.lm_head.weight
        head.fold(head_src)
        head.rnn_bias = head.rnn_proj.bias.float()
        num_cols = head_src.shape[0]
        if self.draft_id_map is not None:
            self.vocab_offset = 0
            self.valid_vocab = num_cols
        else:
            self.vocab_offset = get_tensor_model_parallel_rank() * num_cols
            self.valid_vocab = min(num_cols, self.config.vocab_size - self.vocab_offset)
        assert self.valid_vocab > 0
        self.argmax_workspace = Inkling2DSparkArgmax(
            get_tp_group().cpu_group, self.max_num_reqs, num_cols
        )
        self.model.fuse_alpha.copy_(
            torch.softmax(self.model.fuse_weight.float(), dim=-1)
        )
        if _DRAFT_4BIT in ("uint4", "nvfp4") + _PANEL_4BIT_MODES + _NF3_MODES or _DRAFT_MLP_4BIT in _PANEL_4BIT_MODES + _NF3_MODES:
            self._quantize_draft_dense_4bit(_DRAFT_4BIT)
        elif _DRAFT_FP8:
            self._quantize_draft_dense()
        if _DRAFT_HEAD_4BIT in ("uint4", "nvfp4") + _PANEL_4BIT_MODES:
            self._quantize_draft_head_4bit(_DRAFT_HEAD_4BIT, weight_override=w_draft_head)
        elif _DRAFT_HEAD_MXINT8 and w_draft_head is None:
            self._attach_head_mxint8()
        elif _DRAFT_HEAD_FP8:
            self._quantize_draft_head(weight_override=w_draft_head)
        if w_draft_head is not None:
            del w_draft_head
            torch.cuda.empty_cache()
        if _DRAFT_RNN_4BIT in _PANEL_4BIT_MODES:
            gs = 32 if _DRAFT_RNN_4BIT.endswith("32") else 64
            sym = _DRAFT_RNN_4BIT.startswith("i4")
            w_ctx = head.rnn_proj.weight[:, 2 * head.markov_rank :].contiguous()
            head.rnn_w4, head.rnn_s1, head.rnn_s2, head.rnn_meta = quantize_u4_panel(
                w_ctx, group_size=gs, symmetric=sym
            )
            s2_b = 0 if head.rnn_s2 is None else head.rnn_s2.numel() * 2
            mb = (head.rnn_w4.numel() + head.rnn_s1.numel() * 2 + s2_b) / 2**20
            print(f"[inkling-draft-4bit] markov rnn_proj context {tuple(w_ctx.shape)} -> {_DRAFT_RNN_4BIT} ({mb:.1f} MiB)")
        if _DRAFT_MARKOV_FREE:
            # hidden_out is folded once into fold_weight/fold_bias and never read again (+2.26 MiB).
            freed_b = head.hidden_out.weight.numel() * head.hidden_out.weight.element_size()
            head.hidden_out.weight.data.untyped_storage().resize_(0)
            if head.hidden_out.bias is not None:
                freed_b += head.hidden_out.bias.numel() * head.hidden_out.bias.element_size()
                head.hidden_out.bias.data.untyped_storage().resize_(0)
            from .ops.skinny_int4 import _U4_PANEL_MAX_M
            if (
                head.rnn_meta is not None
                and head.rnn_w_rec is not None
                and self.max_num_reqs * self.config.draft_block_size <= _U4_PANEL_MAX_M
            ):
                freed_b += head.rnn_proj.weight.numel() * head.rnn_proj.weight.element_size()
                head.rnn_proj.weight.data.untyped_storage().resize_(0)
            if head.fold_w8 is not None and head.fold_s8 is not None and head.fold_weight is not None:
                freed_b += head.fold_weight.numel() * head.fold_weight.element_size()
                head.fold_weight.untyped_storage().resize_(0)
            if freed_b > 0:
                torch.cuda.empty_cache()
                print(f"[inkling-vram-reclaim] freed dead markov BF16 copies ({freed_b / 2**20:.1f} MiB)")
        # Link drafter lm_head weight for Layer-4 AR L2 prefetch and prewarm G2 descriptors.
        if self.head_w4 is not None:
            if l2pf.G2 and self.head_w4_meta is not None:
                # Lockstep scale prefix (~768 KB each = ~42 panels = ~6.2 MB of 57.9 MB head_w4).
                segs = []
                if self.head_w4_s1 is not None:
                    segs.append((self.head_w4_s1, 786432))
                if self.head_w4_s2 is not None:
                    segs.append((self.head_w4_s2, 786432))
                segs.append(self.head_w4)
                self.model._l2pf_head = tuple(segs) if len(segs) > 1 else self.head_w4
            else:
                self.model._l2pf_head = self.head_w4
        elif self.head_mx is not None:
            self.model._l2pf_head = self.head_mx.q
        elif self.head_w8 is not None:
            self.model._l2pf_head = self.head_w8
        else:
            self.model._l2pf_head = self.lm_head.weight
        for layer in self.model.layers:
            l2pf.prewarm(_draft_pf_weight(layer.self_attn.qkv_proj), kind="draft")
            # Bug fix (dec160): must prewarm with kind="draft_attn" (matching fork() in self_attn.forward!)
            # so _seg_cache hits inside CUDA graph capture!
            l2pf.prewarm(_draft_pf_weight(layer.self_attn.o_proj), kind="draft_attn")
            l2pf.prewarm(_draft_pf_weight(layer.mlp.gate_up_proj), kind="draft")
            if _DRAFT_FP8_HIGH_M:
                for _m_hi in (layer.self_attn.qkv_proj, layer.mlp.gate_up_proj):
                    if getattr(_m_hi, "fp8_hi_w", None) is not None:
                        l2pf.prewarm(_m_hi.fp8_hi_w, kind="draft")
                if getattr(layer.self_attn.o_proj, "fp8_hi_w", None) is not None:
                    l2pf.prewarm(layer.self_attn.o_proj.fp8_hi_w, kind="draft_attn")
        if self.head_fp8_hi_w is not None:
            self.model._l2pf_head_fp8 = self.head_fp8_hi_w
            l2pf.prewarm(self.head_fp8_hi_w, kind="draft")
        l2pf.prewarm(self.model._l2pf_head, kind="draft")
        fold_pf = head.fold_w8 if head.fold_w8 is not None else head.fold_weight
        if fold_pf is not None:
            l2pf.prewarm(fold_pf, kind="draft")
        if self.lm_head.weight.is_cuda:
            dev = self.lm_head.weight.device
            prealloc_stream_scratch(dev)
            dspark_ops.prealloc_fold_sem(dev)

    @torch.no_grad()
    def _quantize_draft_dense_4bit(self, mode: str) -> None:
        """INKLING_DRAFT_4BIT / INKLING_DRAFT_MLP_4BIT: 4-bit (u4/i4) or 3-bit Hadamard-NF3 copies of drafter dense weights."""
        from .ops.skinny_int4 import _U4_PANEL_MAX_M
        n_bytes = 0
        freed_bf16 = 0
        for layer in self.model.layers:
            mods = [
                (layer.self_attn.qkv_proj, mode),
                (layer.self_attn.o_proj, mode),
                (layer.mlp.gate_up_proj, _DRAFT_MLP_4BIT),
                (layer.mlp.down_proj, _DRAFT_MLP_4BIT),
            ]
            if _DRAFT_KVCTX_FP8:
                mods.append((layer.self_attn.kv_proj_ctx, mode))
            for mod, m_mode in mods:
                if m_mode == "off":
                    continue
                assert mod.weight.dtype == torch.bfloat16
                is_gu = (mod is layer.mlp.gate_up_proj) and _DRAFT_FUSE_SILU
                if m_mode in _NF3_MODES:
                    gs = 32 if m_mode.endswith("32") else 64
                    w4, s1, s2, meta = quantize_h64_nf3_panel(
                        mod.weight.data, group_size=gs, interleave_silu=is_gu
                    )
                    mod.w4_meta = meta
                    can_free = False  # h64_nf3 supports M <= 8; keep BF16 for M > 8 fallback
                elif m_mode in _PANEL_4BIT_MODES:
                    gs = 32 if m_mode.endswith("32") else 64
                    sym = m_mode.startswith("i4")
                    w4, s1, s2, meta = quantize_u4_panel(
                        mod.weight.data, group_size=gs, symmetric=sym, interleave_silu=is_gu
                    )
                    mod.w4_meta = meta
                    can_free = _DRAFT_4BIT_FREE and self.max_num_reqs * self.config.draft_block_size <= _U4_PANEL_MAX_M
                else:
                    qfn = quantize_nvfp4 if m_mode == "nvfp4" else quantize_uint4
                    w4, s1, s2 = qfn(mod.weight.data)
                    can_free = False
                mod.register_buffer("w4", w4, persistent=False)
                mod.register_buffer("w4_s1", s1, persistent=False)
                mod.register_buffer("w4_s2", s2, persistent=False)
                mod.w4_mode = m_mode
                s2_b = 0 if s2 is None else s2.numel() * s2.element_size()
                n_bytes += mod.weight.numel() * 2 - (
                    w4.numel() * w4.element_size() + s1.numel() * s1.element_size() + s2_b
                )
                # C=32 High-M FP8 Tensor-Core copy (built BEFORE optional BF16 storage reclaim!).
                if (
                    _DRAFT_FP8_HIGH_M
                    and mod.weight.shape[0] >= 256
                    and (mod.weight.shape[1] & 15) == 0
                ):
                    w_hi, sc_hi, n_hi = _quantize_weight_fp8_rowwise(mod.weight.data)
                    mod.register_buffer("fp8_hi_w", w_hi, persistent=False)
                    mod.register_buffer("fp8_hi_sc", sc_hi, persistent=False)
                    mod.fp8_hi_wt = mod.fp8_hi_w.t()
                    mod.fp8_hi_n = n_hi
                # Keep kv_proj_ctx BF16 (11.25 MiB) for T=4096 prefill; free the 4 trunk projections
                # (382.5 MiB) using storage resize_(0) so mod.weight.shape[0] stays valid!
                if can_free and mod is not layer.self_attn.kv_proj_ctx:
                    freed_bf16 += mod.weight.numel() * 2
                    mod.weight.data.untyped_storage().resize_(0)
        if _DRAFT_FP8_HIGH_M and self.model.layers and self.model.layers[0].self_attn.qkv_proj.weight.is_cuda:
            _ensure_draft_fp8_hi_cap(self.model.layers[0].self_attn.qkv_proj.weight.device)
        torch.cuda.empty_cache()
        print(
            f"[inkling-draft-4bit] drafter dense weights -> attn={mode}, mlp={_DRAFT_MLP_4BIT} "
            f"(fuse_silu={int(_DRAFT_FUSE_SILU)}, bf16_freed={freed_bf16 / 2**20:.1f} MiB), "
            f"{n_bytes / 2**20:.0f} MiB fewer bytes read"
        )

    @torch.no_grad()
    def _quantize_draft_head_4bit(self, mode: str, weight_override: torch.Tensor | None = None) -> None:
        """INKLING_DRAFT_HEAD_4BIT=uint4|nvfp4|u4p32|u4p64|i4p32|i4p64: 4-bit copy of this rank's lm_head shard."""
        assert self.lm_head is not None
        weight = weight_override if weight_override is not None else self.lm_head.weight
        if mode in _PANEL_4BIT_MODES:
            gs = 32 if mode.endswith("32") else 64
            sym = mode.startswith("i4")
            w4, s1, s2, meta = quantize_u4_panel(weight, group_size=gs, symmetric=sym)
            self.head_w4, self.head_w4_s1, self.head_w4_s2, self.head_w4_mode, self.head_w4_meta = (
                w4, s1, s2, mode, meta
            )
            if _DRAFT_FP8_HIGH_M and _DRAFT_FP8_HIGH_M_HEAD and (weight.shape[1] & 15) == 0:
                w_hi, sc_hi, n_hi = _quantize_weight_fp8_rowwise(weight)
                self.head_fp8_hi_w = w_hi
                self.head_fp8_hi_wt = w_hi.t()
                self.head_fp8_hi_sc = sc_hi
                self.head_fp8_hi_n = n_hi
                if weight.is_cuda:
                    _ensure_draft_fp8_hi_cap(weight.device)
            torch.cuda.empty_cache()
            s2_b = 0 if s2 is None else s2.numel() * s2.element_size()
            mb = (w4.numel() * w4.element_size() + s1.numel() * s1.element_size() + s2_b) / 2**20
            print(f"[inkling-draft-4bit] drafter lm_head copy {tuple(weight.shape)} -> {mode} ({mb:.1f} MiB)")
            return
        N, K = weight.shape
        qfn = quantize_nvfp4 if mode == "nvfp4" else quantize_uint4
        w4 = torch.empty((N, K // 2), dtype=torch.uint8, device=weight.device)
        if mode == "nvfp4":
            s1 = torch.empty((N, K // 16), dtype=torch.uint8, device=weight.device)
            s2 = torch.empty((N,), dtype=torch.float32, device=weight.device)
        else:
            s1 = torch.empty((N, K // 32), dtype=torch.float16, device=weight.device)
            s2 = torch.empty((N, K // 32), dtype=torch.float16, device=weight.device)
        for start in range(0, N, 4096):
            rows = slice(start, start + 4096)
            w4[rows], s1[rows], s2[rows] = qfn(weight[rows])
        self.head_w4, self.head_w4_s1, self.head_w4_s2, self.head_w4_mode = w4, s1, s2, mode
        torch.cuda.empty_cache()
        mb = (w4.numel() * w4.element_size() + s1.numel() * s1.element_size() + s2.numel() * s2.element_size()) / 2**20
        print(f"[inkling-draft-4bit] drafter lm_head copy {tuple(weight.shape)} -> {mode} ({mb:.1f} MiB)")

    @torch.no_grad()
    def _attach_head_mxint8(self) -> None:
        """INKLING_DRAFT_HEAD=mxint8: shared MXINT8 table (built once per lm_head weight; the target's certified
        head reuses it). Falls back to the bf16 head on any failure."""
        try:
            from .ops import cert_head
            t = cert_head.get_table(self.lm_head.weight, None)
            if t is None:
                raise ValueError("table build failed")
            self.head_mx = t
            print(f"[inkling-draft-head] drafter base logits read the MXINT8 table ({t.bytes / 2**20:.0f} MiB, "
                  f"rel eps {t.eps * 100:.3f}%); no private fp8 copy")
        except Exception as e:  # noqa: BLE001
            print(f"[inkling-draft-head] MXINT8 head unavailable ({e!r}); bf16 head")

    @torch.no_grad()
    def _quantize_draft_dense(self) -> None:
        """INKLING_DRAFT_FP8: fp8 (e4m3, per-output-row scale) copies of the drafter's dense
        weights; the bf16 copies are freed when every batch the drafter sees fits skinny_fp8."""
        # Freeing the bf16 copies (~190 MiB/rank) is opt-in: any call with more than 64 rows
        # (profile / warmup runs) would then have no weights to fall back on.
        free = (
            os.environ.get("INKLING_DRAFT_FP8_FREE", "0") == "1"
            and self.max_num_reqs * self.config.draft_block_size <= _FP8_MAX_ROWS
        )
        n_bytes = 0
        for layer in self.model.layers:
            mods = [
                layer.self_attn.qkv_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_up_proj,
                layer.mlp.down_proj,
            ]
            if _DRAFT_KVCTX_FP8:
                mods.append(layer.self_attn.kv_proj_ctx)
            for mod in mods:
                assert mod.weight.dtype == torch.bfloat16
                w8, scale = quantize_rows(mod.weight.data)
                mod.register_buffer("w8", w8, persistent=False)
                mod.register_buffer("w8_scale", scale, persistent=False)
                n_bytes += mod.weight.numel() * 2 - w8.numel() - scale.numel() * 4
                if free and mod is not layer.self_attn.kv_proj_ctx:
                    mod.weight.data = mod.weight.data.new_empty(0)
        torch.cuda.empty_cache()
        print(
            f"[inkling-draft-fp8] drafter dense weights -> e4m3 per-row, bf16 "
            f"{'freed' if free else 'kept'}, {n_bytes / 2**20:.0f} MiB fewer bytes read"
        )

    @torch.no_grad()
    def _quantize_draft_head(self, weight_override: torch.Tensor | None = None) -> None:
        """INKLING_DRAFT_HEAD_FP8: private fp8 (e4m3, per-row scale) copy of this rank's lm_head
        shard for the drafter's base logits (the Markov fold keeps the bf16 head)."""
        assert self.lm_head is not None
        weight = weight_override if weight_override is not None else self.lm_head.weight
        w8 = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
        scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
        for start in range(0, weight.shape[0], 4096):
            rows = slice(start, start + 4096)
            w8[rows], scale[rows] = quantize_rows(weight[rows])
        self.head_w8, self.head_w8_scale = w8, scale
        torch.cuda.empty_cache()
        print(f"[inkling-draft-fp8] drafter lm_head copy {tuple(w8.shape)} e4m3 per-row ({w8.numel() / 2**20:.0f} MiB)")

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each draft layer mixes the raw target features itself.
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        for i, layer in enumerate(self.model.layers):
            context = self.model.mix(context_states, i, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context,
                context_positions,
                self.model.cos_sin_cache,
                None if context_slot_mappings is None else context_slot_mappings[i],
            )

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        """Recurrent-head context from the last committed token's features."""
        return self.model.mix(features, -1, self.model.markov_head.dflare_norm)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """TP-local bf16 base logits (high-M FP8 TC at M >= 96 when opted in, else 4-bit/bf16)."""
        assert self.lm_head is not None
        if (
            _DRAFT_FP8_HIGH_M
            and self.head_fp8_hi_wt is not None
            and hidden_states.dim() == 2
            and hidden_states.shape[0] >= _DRAFT_FP8_HIGH_M_MIN
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.stride(1) == 1
            and hidden_states.is_cuda
        ):
            y = _fp8_high_m_linear(
                hidden_states, self.head_fp8_hi_wt, self.head_fp8_hi_sc, self.head_fp8_hi_n
            )
            if y is not None:
                return y
        if self.head_w4_meta is not None and u4_panel_supported(hidden_states, self.head_w4_meta):
            return skinny_u4_panel_linear(
                hidden_states, self.head_w4, self.head_w4_s1, self.head_w4_s2, self.head_w4_meta,
                pdl=_DRAFT_PDL,
            )
        if self.head_w4 is not None and int4_supported(hidden_states, self.head_w4):
            if self.head_w4_mode == "nvfp4":
                return skinny_nvfp4_linear(hidden_states, self.head_w4, self.head_w4_s1, self.head_w4_s2)
            return skinny_uint4_linear(hidden_states, self.head_w4, self.head_w4_s1, self.head_w4_s2)
        if self.head_mx is not None:
            from .ops import cert_head
            if cert_head.dq_supported(hidden_states, self.head_mx):
                return cert_head.dq_logits(self.head_mx, hidden_states)
        if self.head_w8 is not None and fp8_supported(hidden_states, self.head_w8):
            return skinny_fp8_linear(hidden_states, self.head_w8, self.head_w8_scale)
        return linear(hidden_states, self.lm_head.weight)

    def markov_step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the recurrent head; return the state and TP-local bias."""
        return self.model.markov_head.step(state, prev_tokens, context)

    def markov_step_and_argmax(
        self,
        state: torch.Tensor | None,
        prev_tokens: torch.Tensor,
        context: torch.Tensor,
        base_logits: torch.Tensor,
        es_ctx=None,
        out_tok: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused Markov RNN step + fused fold+softcap+argmax_record + (cond) RDMA AG + select."""
        head = self.model.markov_head
        assert head.fold_weight is not None and head.fold_bias is not None and head.rnn_w_rec is not None
        rows = base_logits.shape[0]
        skip_dev = es_ctx[2][0] if (es_ctx is not None and rows == 1 and es_ctx[0] > 0) else None
        state, latent = markov_rnn_step_fused(
            state, prev_tokens, context, head.markov_w1.weight, head.rnn_w_rec,
            skip_dev=skip_dev, pdl=_DRAFT_PDL, init_zero=(state is None),
        )
        sink = dspark_ops._CONF_SINK
        conf_out = None
        if sink is not None and sink.col < sink.buf.shape[1] and rows <= sink.buf.shape[0]:
            col = sink.col
            sink.col += 1
            conf_out = sink.buf[:rows, col]
        tok = gathered_fold_argmax_fused(
            base_logits,
            latent,
            head.fold_weight,
            head.fold_bias,
            self.config.final_logit_softcapping,
            self.valid_vocab,
            self.vocab_offset,
            fold_w8=head.fold_w8,
            fold_s8=head.fold_s8,
            conf_out=conf_out,
            es_ctx=es_ctx,
            out_tok=out_tok,
            pdl=_DRAFT_PDL,
            id_map=self.draft_id_map,
        )
        return state, tok

    def markov_logits(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Full-vocab softcapped logits (scatters into -inf when draft vocab is pruned,
        keeping stochastic temp>0 rejection sampling mathematically exact to target!)."""
        logits = add_softcap(base_logits, bias, self.config.final_logit_softcapping)
        logits = tensor_model_parallel_all_gather(logits)
        if self._hot_ids_global is not None:
            full = torch.full(
                (logits.shape[0], self.config.vocab_size),
                float("-inf"),
                dtype=logits.dtype,
                device=logits.device,
            )
            full[:, self._hot_ids_global] = logits
            return full
        return logits[:, : self.config.vocab_size]

    def markov_argmax(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """``markov_logits(...).argmax(-1)``, NaN as -inf."""
        return dspark_argmax(
            self.argmax_workspace,
            base_logits,
            bias,
            self.config.final_logit_softcapping,
            self.valid_vocab,
            self.vocab_offset,
        )

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            name, shard = f"model.{name}", None
            for old, new, shard_id in self._stacked:
                if old in name:
                    name, shard = name.replace(old, new), shard_id
                    break
            param = params[name]
            if weight.dtype != param.dtype:
                raise ValueError(f"{name}: expected {param.dtype}, got {weight.dtype}")
            if shard is None:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            else:
                param.weight_loader(param, weight, shard)
            loaded.add(name)
        return loaded
