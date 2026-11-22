# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention: relative-bias GQA with short-conv K/V, L2-clipped
q/k/r, gated output and per-token attention sinks."""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import cast

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import bucket_max_seqlen_q
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingConvState
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .configs import Inkling2ModelConfig
from .lora import add_linear_lora
from .ops.attention import (
    _INKLING2_FA4_REL_ATTENTION_KERNEL,
    gate_sink,
    plan_attention,
    plan_decode_attention,
    rel_decode_attention,
)
from .ops import l2pf
from .ops.linear import linear
from .ops.sconv import attention_prologue, shift_rel_proj
from .sconv import K_STREAM, V_STREAM, Inkling2ShortConv, conv_metadata

# INKLING_GLOBAL_DYN_OVERLAP=1: overlap global-layer k_sconv/v_sconv dynamic_taps
# (~10.4 us/layer, 83 us/step) on a reusable side stream behind q_side + dual_kv
# (~52 us/layer), restoring the direct q_side -> dual_kv -> attn_prologue PDL chain.
# Strictly for decode (M <= 16); exact same kernels & inputs -> 0-ULP bit-identical.
_GLOBAL_DYN_OVERLAP = os.environ.get(
    "INKLING_GLOBAL_DYN_OVERLAP", "1"
).strip().lower() not in ("0", "", "off", "false", "no")
_GLOBAL_DYN_OVERLAP_MAX_M = int(os.environ.get("INKLING_GLOBAL_DYN_OVERLAP_MAX_M", "16"))
_FUSED_ATTN_SM12_FIX = os.environ.get(
    "INKLING_FUSED_DECODE_ATTN_SM12_FIX", "0"
).strip().lower() in ("1", "true", "on", "yes")
_TREE_SPEC = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DYN_STREAMS: dict[int, torch.cuda.Stream] = {}


def _get_dyn_stream(device: torch.device) -> torch.cuda.Stream | None:
    """Reusable per-device side stream for global k_dyn/v_dyn overlap.
    Created & cuBLAS-warmed strictly outside CUDA graph capture (fail-closed inside capture)."""
    if device.type != "cuda":
        return None
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _DYN_STREAMS.get(dev_idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        st = torch.cuda.Stream(device=device)
        # Materialize per-stream cuBLAS handle & workspace outside CUDA graph capture.
        with torch.cuda.stream(st):
            _z = torch.empty((1, 16), dtype=torch.bfloat16, device=device)
            _ = torch.mm(_z, _z.t())
        st.synchronize()
        _DYN_STREAMS[dev_idx] = st
    return st


@dataclass
class Inkling2AttentionMetadata(FlashAttentionMetadata):
    # Host copies for planning FA4 splits (exact for prefill requests).
    query_lens_cpu: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    seq_lens_cpu: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))


class Inkling2AttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Orders decode/verify requests first and records the split, so that a
    mixed batch runs its decode part on ``rel_decode_attention``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2AttentionMetadata:
        cm = common_attn_metadata
        assert cm.seq_lens_cpu_upper_bound is not None
        md = Inkling2AttentionMetadata(
            **vars(super().build(common_prefix_len, cm, fast_build)),
            query_lens_cpu=cm.query_start_loc_cpu.diff().numpy(),
            seq_lens_cpu=cm.seq_lens_cpu_upper_bound.numpy(),
        )
        assert self.reorder_batch_threshold is not None
        (
            md.num_decode_reqs,
            md.num_prefill_reqs,
            md.num_decode_tokens,
            md.num_prefill_tokens,
        ) = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.reorder_batch_threshold
        )
        return md


class Inkling2AttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer.

    Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from ``x_q``, k from ``x_k`` and v from
    ``x_v``, and use dynamic K/V convolutions and log-scaled q/r.
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.window_size = (config.sliding_window_size - 1, 0) if is_local else (-1, -1)
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        # Full layers project the query side alone; rounding its N up to 8
        # keeps the output rows 16-byte aligned for cuBLAS (the extra columns,
        # from the K rows, are discarded).
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        self.max_decode_query_len = max_decode_query_len(vllm_config)
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        self.kv_cache = torch.tensor([])
        self._dyn_ev_in: torch.cuda.Event | None = None
        self._dyn_ev_out: torch.cuda.Event | None = None
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self
        if vllm_config.kernel_config.enable_jit_warmup:
            _INKLING2_FA4_REL_ATTENTION_KERNEL.register_warmup()

    def _ensure_dyn_events(
        self, device: torch.device
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if self._dyn_ev_in is not None and self._dyn_ev_out is not None:
            return self._dyn_ev_in, self._dyn_ev_out
        if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return None
        ev_in = torch.cuda.Event()
        ev_out = torch.cuda.Event()
        # Materialize underlying cudaEvent_t eagerly outside graph capture.
        cur = torch.cuda.current_stream(device)
        ev_in.record(cur)
        ev_out.record(cur)
        self._dyn_ev_in = ev_in
        self._dyn_ev_out = ev_out
        return ev_in, ev_out

    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)
        self.rel_proj_rev = self.rel_logits_proj.flip(-1).contiguous()
        self.rel_proj_t = self.rel_logits_proj.t().contiguous()
        if not self.is_local and _GLOBAL_DYN_OVERLAP and self.rel_logits_proj.is_cuda:
            dev = self.rel_logits_proj.device
            _get_dyn_stream(dev)
            self._ensure_dyn_events(dev)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if self.is_local:
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=vllm_config.model_config.dtype,
                sliding_window=self.sliding_window,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=vllm_config.model_config.dtype,
        )

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        # ``linear`` on the weight bypasses the layer (and a LoRA wrapper's
        # forward), so the LoRA deltas are added here (no-op without LoRA).
        weight = self.qkvr.weight
        if self.is_local:
            projected = linear(x_q, weight)
            add_linear_lora(self.qkvr, projected, x_q, range(6))
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = linear(x_q, weight[: sum(self.q_side_splits)])
        kv_pair = None
        if x_k.shape[0] <= 16:
            from .ops import skinny_bf11
            if skinny_bf11.ENABLED:
                from .ops import linear as _lin_mod
                kv_pair = skinny_bf11.bf11_dual_kv(
                    x_k, x_v, weight[q_side:k_end], weight[k_end : k_end + kv_size],
                    pdl=_lin_mod._PDL_QKVR, trig=False,
                )
        else:
            from .ops import skinny_bf11
            if skinny_bf11.ENABLED and skinny_bf11.C32_ENABLED and x_k.shape[0] <= skinny_bf11.C32_MAX_M:
                kv_pair = skinny_bf11.bf11_dual_kv(
                    x_k, x_v, weight[q_side:k_end], weight[k_end : k_end + kv_size],
                    pdl=False, trig=False,
                )
        if kv_pair is not None:
            k, v = kv_pair
        else:
            k = linear(x_k, weight[q_side:k_end])
            v = linear(x_v, weight[k_end : k_end + kv_size])
        add_linear_lora(self.qkvr, projected, x_q, range(4))
        add_linear_lora(self.qkvr, k, x_k, [4])
        add_linear_lora(self.qkvr, v, x_v, [5])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _project_global_with_dyn(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # Global-layer decode projection:
        #   - With INKLING_GLOBAL_DYN_OVERLAP=1 (M <= 16): fork k_dyn + v_dyn onto a
        #     reusable side stream right after attn_res while main stream runs the
        #     direct PDL chain q_side(pdl=1,trig=1) -> dual_kv(pdl=1,trig=1), joining
        #     the side stream right before _attention/attention_prologue (0-ULP exact).
        #   - Otherwise (prefill M > 16 or flag off): serial fallback keeping
        #     dual_kv(pdl=0,trig=1) -> attn_prologue(pdl=1).
        weight = self.qkvr.weight
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        q_side_w = weight[: sum(self.q_side_splits)]
        if not isinstance(x_k, torch.Tensor):
            # INKLING_PREFILL_SP_Q3=repl_kv: pre-projected [T, kv_dim] K/V + [T, W] dyn taps
            # from sharded x_k/x_v + all_to_all (strictly prefill-only, 100% BF16 exact!).
            projected = linear(x_q, q_side_w)
            add_linear_lora(self.qkvr, projected, x_q, range(4))
            q, r, gate, sinks = projected.split(self.q_side_splits, dim=-1)[:4]
            return q, r, gate, sinks, x_k.k, x_k.v, x_k.k_dyn, x_k.v_dyn
        w_k = weight[q_side:k_end]
        w_v = weight[k_end : k_end + kv_size]

        use_overlap = (
            _GLOBAL_DYN_OVERLAP
            and x_q.shape[0] <= _GLOBAL_DYN_OVERLAP_MAX_M
            and x_k.is_cuda
            and self.k_sconv.dynamic
            and self.v_sconv.dynamic
            and not l2pf._is_piecewise_capturing()
        )
        dyn_stream = None
        dyn_evs = None
        if use_overlap:
            dyn_stream = _get_dyn_stream(x_k.device)
            if dyn_stream is not None:
                dyn_evs = self._ensure_dyn_events(x_k.device)
            if dyn_stream is None or dyn_evs is None:
                use_overlap = False

        if use_overlap:
            assert dyn_stream is not None and dyn_evs is not None
            ev_in, ev_out = dyn_evs
            cur = torch.cuda.current_stream(x_k.device)
            # Dynamic taps depend solely on x_k / x_v (already produced by attn_res).
            ev_in.record(cur)
            dyn_stream.wait_event(ev_in)
            with torch.cuda.stream(dyn_stream):
                k_dyn = self.k_sconv.dynamic_taps(x_k)
                v_dyn = self.v_sconv.dynamic_taps(x_v)
                ev_out.record(dyn_stream)
            try:
                projected = None
                from .ops import linear as _lin_mod
                from .ops import skinny_bf11

                can_bf11_kv = (
                    skinny_bf11.ENABLED
                    and skinny_bf11.USE_DUAL_KV
                    and skinny_bf11.get_packed(w_k) is not None
                    and skinny_bf11.get_packed(w_v) is not None
                )
                if x_q.shape[0] <= 16 and _lin_mod._SKINNY and skinny_bf11.ENABLED:
                    projected = skinny_bf11.bf11_linear(
                        x_q,
                        q_side_w,
                        pdl=_lin_mod._pdl_site(x_q, q_side_w, None),
                        trig=(_lin_mod._PDL_QKVR and can_bf11_kv),
                    )
                if projected is None:
                    projected = linear(x_q, q_side_w)
                add_linear_lora(self.qkvr, projected, x_q, range(4))

                kv_pair = None
                if x_k.shape[0] <= 16 and skinny_bf11.ENABLED:
                    kv_pair = skinny_bf11.bf11_dual_kv(
                        x_k,
                        x_v,
                        w_k,
                        w_v,
                        pdl=_lin_mod._PDL_QKVR,
                        trig=_lin_mod._PDL_QKVR,
                    )
                if kv_pair is not None:
                    k, v = kv_pair
                else:
                    k = linear(x_k, w_k)
                    v = linear(x_v, w_v)
                add_linear_lora(self.qkvr, k, x_k, [4])
                add_linear_lora(self.qkvr, v, x_v, [5])
            finally:
                cur.wait_event(ev_out)
            if not torch.cuda.is_current_stream_capturing():
                x_k.record_stream(dyn_stream)
                x_v.record_stream(dyn_stream)
                if k_dyn is not None:
                    k_dyn.record_stream(cur)
                if v_dyn is not None:
                    v_dyn.record_stream(cur)
            q, r, gate, sinks = projected.split(self.q_side_splits, dim=-1)[:4]
            return q, r, gate, sinks, k, v, k_dyn, v_dyn

        projected = None
        if x_q.shape[0] <= 16:
            from .ops import linear as _lin_mod
            from .ops import skinny_bf11
            if _lin_mod._SKINNY and skinny_bf11.ENABLED:
                projected = skinny_bf11.bf11_linear(
                    x_q,
                    q_side_w,
                    pdl=_lin_mod._pdl_site(x_q, q_side_w, None),
                    trig=False,
                )
        else:
            from .ops import linear as _lin_mod
            from .ops import skinny_bf11
            if (
                _lin_mod._SKINNY
                and skinny_bf11.ENABLED
                and skinny_bf11.C32_ENABLED
                and x_q.shape[0] <= skinny_bf11.C32_MAX_M
            ):
                projected = skinny_bf11.bf11_linear(
                    x_q,
                    q_side_w,
                    pdl=False,
                    trig=False,
                )
        if projected is None:
            projected = linear(x_q, q_side_w)
        add_linear_lora(self.qkvr, projected, x_q, range(4))

        # Dynamic taps depend solely on x_k / x_v (already produced by attn_res).
        k_dyn = self.k_sconv.dynamic_taps(x_k)
        v_dyn = self.v_sconv.dynamic_taps(x_v)

        kv_pair = None
        if x_k.shape[0] <= 16:
            from .ops import skinny_bf11
            if skinny_bf11.ENABLED:
                from .ops import linear as _lin_mod
                kv_pair = skinny_bf11.bf11_dual_kv(
                    x_k,
                    x_v,
                    w_k,
                    w_v,
                    pdl=False,
                    trig=_lin_mod._PDL_QKVR,
                )
        else:
            from .ops import skinny_bf11
            if skinny_bf11.ENABLED and skinny_bf11.C32_ENABLED and x_k.shape[0] <= skinny_bf11.C32_MAX_M:
                kv_pair = skinny_bf11.bf11_dual_kv(
                    x_k,
                    x_v,
                    w_k,
                    w_v,
                    pdl=False,
                    trig=False,
                )
        if kv_pair is not None:
            k, v = kv_pair
        else:
            k = linear(x_k, w_k)
            v = linear(x_v, w_v)
        add_linear_lora(self.qkvr, k, x_k, [4])
        add_linear_lora(self.qkvr, v, x_v, [5])
        q, r, gate, sinks = projected.split(self.q_side_splits, dim=-1)[:4]
        return q, r, gate, sinks, k, v, k_dyn, v_dyn

    def _split_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.kv_cache.transpose(1, 2).split(
            self.head_dim, dim=-1
        )
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return this rank's bf16 ``(T, hidden)`` partial output (to be
        all-reduced over TP), written to ``out`` if given.

        Local layers read q/k/v from ``x_q``; full layers take ``x_k``, ``x_v`` and
        the fp32 log-scaling ``tau`` for q/r.
        """
        if self.is_local:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        if self.is_local:
            q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
            k_dyn = self.k_sconv.dynamic_taps(x_k)
            v_dyn = self.v_sconv.dynamic_taps(x_v)
        else:
            q, r, gate, sinks, k, v, k_dyn, v_dyn = self._project_global_with_dyn(
                x_q, x_k, x_v
            )
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            k_dyn,
            v_dyn,
            tau,
            output,
        )
        out = linear(output, self.wo_ud.weight, out)
        add_linear_lora(self.wo_ud, out, output)
        return out

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        """Write the gated attention output to ``output``.

        The kernel plan and the prologue's layout follow the batch's
        decode/prefill split, so this runs eagerly in piecewise CUDA graphs,
        which are captured on a dummy batch and replayed for any batch of the
        same token count.
        """
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        attn_metadata = get_forward_context().attn_metadata
        md = None
        if isinstance(attn_metadata, dict) and self.kv_cache.numel() > 0:
            md = cast(Inkling2AttentionMetadata, attn_metadata[self.prefix])
        num_decodes, num_decode_tokens, decode_splits = 0, 0, 0
        if md is not None:
            num_decodes, num_decode_tokens, decode_splits = self._plan_decode(
                md, num_tokens
            )
        clipped_q = q.new_empty(q.shape)
        from .ops import sm12_attn as _sm12_attn

        if (
            self.is_local
            and md is not None
            and num_decode_tokens == 0
            and decode_splits == 0
            and num_tokens >= _sm12_attn._PF_LOCAL_OTF_REL_MIN_T
            and _sm12_attn._PF_LOCAL_OTF_REL
            and self.head_dim == 128
            and self.d_rel == 16
            and self.num_kv_heads == 1
            and getattr(self, "rel_proj_t", None) is not None
            and not torch.cuda.is_current_stream_capturing()
            and _sm12_attn.get_local_otf_rel_lib() is not None
        ):
            conv_md = conv_metadata(self.conv_state)
            assert conv_md is not None
            key_cache, value_cache = self._split_kv_cache()
            r_view = r.view(num_tokens, self.num_heads, self.d_rel)
            attention_prologue(
                q,
                r_view,
                k,
                v,
                self.q_scale,
                self.r_scale,
                tau,
                self.rel_proj_shifted,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                self.conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                key_cache,
                value_cache,
                md.slot_mapping,
                clipped_q,
                None,
                md.query_start_loc,
                md.seq_lens,
                clip_threshold=self.qk_clip,
                rel_threshold=self.rel_clip,
                norm_eps=self.eps,
                rel_proj_rev=getattr(self, "rel_proj_rev", None),
                skip_rel=True,
            )
            nt = md.num_actual_tokens
            if _sm12_attn.local_swa_otf_rel_prefill(
                clipped_q[:nt],
                r_view[:nt],
                self.r_scale,
                self.rel_proj_t,
                key_cache,
                value_cache,
                md.block_table,
                md.seq_lens,
                md.query_start_loc,
                gate[:nt],
                sinks[:nt],
                output[:nt],
                max_seqlen_q=md.max_query_len,
                softmax_scale=self.scaling,
                window=self.window_size[0],
                rel_threshold=self.rel_clip,
            ):
                if nt < num_tokens:
                    output[nt:].zero_()
                return
        elif (
            not self.is_local
            and md is not None
            and num_decode_tokens == 0
            and decode_splits == 0
            and num_tokens >= _sm12_attn._PF_GLOBAL_OTF_REL_MIN_T
            and _sm12_attn._PF_GLOBAL_OTF_REL
            and self.head_dim == 128
            and self.d_rel == 16
            and self.num_kv_heads == 1
            and getattr(self, "rel_proj_t", None) is not None
            and not torch.cuda.is_current_stream_capturing()
            and _sm12_attn.get_global_tail_otf_rel_lib() is not None
        ):
            from .ops import sconv as _sconv_mod

            if getattr(_sconv_mod, "_PROLOGUE_PREFILL_OPT", True):
                conv_md = conv_metadata(self.conv_state)
                assert conv_md is not None
                key_cache, value_cache = self._split_kv_cache()
                r_view = r.view(num_tokens, self.num_heads, self.d_rel)
                if _sm12_attn.can_global_otf_rel_prefill(
                    clipped_q,
                    r_view,
                    tau,
                    self.rel_proj_t,
                    key_cache,
                    value_cache,
                    gate,
                    sinks,
                    output,
                ):
                    attention_prologue(
                        q,
                        r_view,
                        k,
                        v,
                        self.q_scale,
                        self.r_scale,
                        tau,
                        self.rel_proj_shifted,
                        self.k_sconv.conv_weight,
                        self.v_sconv.conv_weight,
                        k_dynamic,
                        v_dynamic,
                        self.conv_state.kv_cache,
                        positions,
                        conv_md.block_table,
                        conv_md.seq_idx,
                        conv_md.slot_mapping,
                        conv_md.query_start,
                        key_cache,
                        value_cache,
                        md.slot_mapping,
                        clipped_q,
                        None,
                        md.query_start_loc,
                        md.seq_lens,
                        clip_threshold=self.qk_clip,
                        rel_threshold=self.rel_clip,
                        norm_eps=self.eps,
                        rel_proj_rev=getattr(self, "rel_proj_rev", None),
                        skip_rel=True,
                    )
                    nt = md.num_actual_tokens
                    tau_nt = tau[:nt] if tau is not None else None
                    if _sm12_attn.global_otf_rel_prefill(
                        clipped_q[:nt],
                        r_view[:nt],
                        self.r_scale,
                        tau_nt,
                        self.rel_proj_t,
                        key_cache,
                        value_cache,
                        md.block_table,
                        md.seq_lens,
                        md.query_start_loc,
                        gate[:nt],
                        sinks[:nt],
                        output[:nt],
                        max_seqlen_q=md.max_query_len,
                        softmax_scale=self.scaling,
                        rel_threshold=self.rel_clip,
                        seq_lens_cpu=md.seq_lens_cpu,
                        query_lens_cpu=md.query_lens_cpu,
                    ):
                        if nt < num_tokens:
                            output[nt:].zero_()
                        return
        # S8 / C=32 Fused SM121a Decode Attention fast-path:
        #   - C=1 (`M <= 16`): `decode_splits` OR opt-in `_FUSED_ATTN_SM12_FIX` (fixes SM12x `decode_splits==0`).
        #   - C=32 (`17 <= M <= MAX_TOKENS_C32`): opt-in `INKLING_FUSED_DECODE_ATTN_C32=1` when `max_q*G <= 64`
        #     (strictly decode/verify; never fires on prefill!).
        _active_tree_st = None
        if _TREE_SPEC and num_tokens <= 16:
            from .ops import tree_spec_runtime as _tsr
            _active_tree_st = _tsr.get_active_tree_state(num_tokens)
        if md is not None and self.head_dim == 128 and self.d_rel == 16 and self.num_kv_heads == 1:
            from .ops import sconv as _sconv_mod, sm12_attn_fused
            _max_q_bkt = bucket_max_seqlen_q(md.max_query_len)
            _g_ratio = self.num_heads // self.num_kv_heads
            _can_tree_fused = (
                _active_tree_st is not None
                and num_tokens <= 8
                and getattr(_active_tree_st, "ws_max", 4) <= 2
                and getattr(_active_tree_st, "tree_meta", None) is not None
                and sm12_attn_fused.has_tree_lib()
            )
            _try_fused_c1 = (
                num_tokens <= 16
                and sm12_attn_fused.ENABLED
                and (_active_tree_st is None or _can_tree_fused)
                and (bool(decode_splits) or (_FUSED_ATTN_SM12_FIX and _max_q_bkt * _g_ratio <= 64))
            )
            _try_fused_c32 = (
                _active_tree_st is None
                and 17 <= num_tokens <= sm12_attn_fused.MAX_TOKENS_C32
                and sm12_attn_fused.ENABLED_C32
                and decode_splits == 0
                and _max_q_bkt * _g_ratio <= 64
            )
            if _try_fused_c1 or _try_fused_c32:
                if l2pf.WO and num_tokens <= 16:
                    from .ops import skinny_bf11, skinny_bf12
                    _wt = skinny_bf11.l2pf_tensor(self.wo_ud.weight) if skinny_bf11.ENABLED else skinny_bf12.l2pf_tensor(self.wo_ud.weight)
                    l2pf.fork(_wt, kind="attn")
                conv_md = conv_metadata(self.conv_state)
                assert conv_md is not None
                key_cache, value_cache = self._split_kv_cache()
                rel_proj_t = getattr(self, "rel_proj_t", None)
                if rel_proj_t is None and self.rel_extent > 0 and not torch.cuda.is_current_stream_capturing():
                    rel_proj_t = sm12_attn_fused.prepack_rel_proj_t(self.rel_logits_proj)
                    self.rel_proj_t = rel_proj_t
                if self.rel_extent == 0 or rel_proj_t is not None:
                    if _can_tree_fused:
                        if sm12_attn_fused.fused_target_decode_attn_tree(
                            q, r, k, v,
                            self.q_scale, self.r_scale, tau, rel_proj_t,
                            self.k_sconv.conv_weight, self.v_sconv.conv_weight, k_dynamic, v_dynamic,
                            self.conv_state.kv_cache, key_cache, value_cache,
                            md.slot_mapping, positions, conv_md.block_table, conv_md.seq_idx,
                            conv_md.slot_mapping, conv_md.query_start,
                            md.block_table, md.seq_lens, md.query_start_loc,
                            gate, sinks, output,
                            tree_sconv_anc=_active_tree_st.tree_sconv_anc,
                            tree_meta=_active_tree_st.tree_meta,
                            softmax_scale=self.scaling,
                            window=self.window_size[0] if self.is_local else -1,
                            rel_threshold=self.rel_clip,
                            clip_threshold=self.qk_clip,
                            norm_eps=self.eps,
                            num_actual_tokens=md.num_actual_tokens,
                            n_live_buf=_sconv_mod._rowar_nlive_buf(num_tokens),
                            q_out_buf=clipped_q,
                        ):
                            return
                    elif sm12_attn_fused.fused_target_decode_attn(
                        q, r, k, v,
                        self.q_scale, self.r_scale, tau, rel_proj_t,
                        self.k_sconv.conv_weight, self.v_sconv.conv_weight, k_dynamic, v_dynamic,
                        self.conv_state.kv_cache, key_cache, value_cache,
                        md.slot_mapping, positions, conv_md.block_table, conv_md.seq_idx,
                        conv_md.slot_mapping, conv_md.query_start,
                        md.block_table, md.seq_lens, md.query_start_loc,
                        gate, sinks, output,
                        softmax_scale=self.scaling,
                        window=self.window_size[0] if self.is_local else -1,
                        rel_threshold=self.rel_clip,
                        clip_threshold=self.qk_clip,
                        norm_eps=self.eps,
                        num_actual_tokens=md.num_actual_tokens,
                        n_live_buf=_sconv_mod._rowar_nlive_buf(num_tokens),
                        q_out_buf=clipped_q,
                        max_seqlen_q=_max_q_bkt,
                    ):
                        return
        # Sheared for tml_fa4 (see attention_prologue); the decode kernels
        # read the same layout.
        rel_logits = q.new_empty(
            (num_tokens + 128, self.num_heads, self.rel_extent + 256)
        )
        if md is not None:
            if l2pf.WO and num_tokens <= 16:
                from .ops import skinny_bf11, skinny_bf12
                _wt = skinny_bf11.l2pf_tensor(self.wo_ud.weight) if skinny_bf11.ENABLED else skinny_bf12.l2pf_tensor(self.wo_ud.weight)
                l2pf.fork(_wt, kind="attn")
            elif (
                (l2pf.WO and num_tokens <= 32)
                or (getattr(l2pf, "C32", False) and getattr(l2pf, "C32_WO", False) and 17 <= num_tokens <= getattr(l2pf, "C32_MAX_M", 256))
            ):
                # L2PF hygiene fix: at M > 16 linear.py never calls skinny_bf12; prefetch the
                # exact active tensor (BF11_C32 slabs if active for num_tokens, else raw BF16).
                from .ops import skinny_bf11
                _wt = skinny_bf11.l2pf_tensor(self.wo_ud.weight, num_tokens=num_tokens) if skinny_bf11.ENABLED else self.wo_ud.weight
                if getattr(l2pf, "C32", False):
                    l2pf.fork(_wt, kind="attn", num_tokens=num_tokens)
                else:
                    l2pf.fork(_wt, kind="attn")
            conv_md = conv_metadata(self.conv_state)
            assert conv_md is not None
            key_cache, value_cache = self._split_kv_cache()
            attention_prologue(
                q,
                r.view(num_tokens, self.num_heads, self.d_rel),
                k,
                v,
                self.q_scale,
                self.r_scale,
                tau,
                self.rel_proj_shifted,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                self.conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                key_cache,
                value_cache,
                md.slot_mapping,
                clipped_q,
                rel_logits,
                md.query_start_loc,
                md.seq_lens,
                clip_threshold=self.qk_clip,
                rel_threshold=self.rel_clip,
                norm_eps=self.eps,
                rel_proj_rev=getattr(self, "rel_proj_rev", None),
            )
        if decode_splits:
            assert md is not None
            self._decode_attention(
                md,
                clipped_q[:num_decode_tokens],
                rel_logits,
                gate[:num_decode_tokens],
                sinks[:num_decode_tokens],
                output[:num_decode_tokens],
                decode_splits,
                num_decodes,
            )
        if num_decode_tokens < num_tokens:
            out = q.new_empty(
                (num_tokens - num_decode_tokens, self.num_heads, self.head_dim),
                dtype=torch.float32,
            )
            lse = q.new_empty((self.num_heads, out.shape[0]), dtype=torch.float32)
            if md is not None:
                num_splits, use_scheduler = plan_attention(
                    self.is_local,
                    md.query_lens_cpu[num_decodes:],
                    md.seq_lens_cpu[num_decodes:],
                    self.num_heads,
                    self.num_kv_heads,
                )
                self._fa4_attention(
                    md,
                    clipped_q[num_decode_tokens:],
                    rel_logits[num_decode_tokens:],
                    out,
                    lse,
                    num_splits,
                    use_scheduler,
                    num_decodes,
                )
            else:
                out.zero_()
                lse.zero_()
            gate_sink(
                out,
                gate[num_decode_tokens:],
                lse.t(),
                sinks[num_decode_tokens:],
                output[num_decode_tokens:],
            )

    def _plan_decode(
        self, md: FlashAttentionMetadata, num_tokens: int
    ) -> tuple[int, int, int]:
        """Return ``(num_decodes, num_decode_tokens, num_splits)`` for
        ``rel_decode_attention``: the whole batch, or in a mixed batch the
        leading decode/verify requests."""
        num_reqs = md.seq_lens.shape[0]
        plan = partial(
            plan_decode_attention,
            self.is_local,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rel_extent=self.rel_extent,
        )
        num_splits = plan(num_reqs=num_reqs, max_query_len=md.max_query_len)
        if num_splits:
            return num_reqs, num_tokens, num_splits
        num_decodes, num_decode_tokens = md.num_decode_reqs, md.num_decode_tokens
        if not 0 < num_decodes < num_reqs:
            return 0, 0, 0
        max_query_len = min(
            self.max_decode_query_len, num_decode_tokens - num_decodes + 1
        )
        num_splits = plan(num_reqs=num_decodes, max_query_len=max_query_len)
        if not num_splits:
            return 0, 0, 0
        return num_decodes, num_decode_tokens, num_splits

    def _decode_attention(
        self,
        md: FlashAttentionMetadata,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        output: torch.Tensor,
        num_splits: int,
        num_reqs: int,
    ) -> None:
        key_cache, value_cache = self._split_kv_cache()
        mixed = num_reqs < md.seq_lens.shape[0]
        num_tokens = md.num_decode_tokens if mixed else md.num_actual_tokens
        max_query_len = md.max_query_len
        if mixed:
            max_query_len = min(self.max_decode_query_len, num_tokens - num_reqs + 1)
        rel_decode_attention(
            q,
            key_cache,
            value_cache,
            rel_logits,
            gate,
            sinks,
            output,
            block_table=md.block_table[:num_reqs],
            cache_seqlens=md.seq_lens[:num_reqs],
            cu_seqlens_q=md.query_start_loc[: num_reqs + 1],
            max_query_len=max_query_len,
            num_actual_tokens=num_tokens,
            softmax_scale=self.scaling,
            is_local=self.is_local,
            num_splits=num_splits,
        )

    def _fa4_attention(
        self,
        md: FlashAttentionMetadata,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        num_splits: int,
        use_scheduler: bool,
        num_decodes: int,
    ) -> None:
        """FA4 for the requests after the first ``num_decodes``."""
        num_decode_tokens = md.num_decode_tokens if num_decodes else 0
        nt = md.num_actual_tokens - num_decode_tokens
        cu_seqlens_q = md.query_start_loc[num_decodes:]
        if num_decodes:
            cu_seqlens_q = cu_seqlens_q - num_decode_tokens
        key_cache, value_cache = self._split_kv_cache()
        from .ops import sm12_attn as _sm12_attn

        _sl_cpu = getattr(md, "seq_lens_cpu", None)
        _ql_cpu = getattr(md, "query_lens_cpu", None)
        _sm12_attn.set_prefill_cpu_lens(
            _sl_cpu[num_decodes:] if _sl_cpu is not None else None,
            _ql_cpu[num_decodes:] if _ql_cpu is not None else None,
        )
        try:
            _INKLING2_FA4_REL_ATTENTION_KERNEL(
                q[:nt],
                key_cache,
                value_cache,
                block_table=md.block_table[num_decodes:],
                cache_seqlens=md.seq_lens[num_decodes:],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=bucket_max_seqlen_q(md.max_query_len),
                softmax_scale=self.scaling,
                window_size=self.window_size,
                sheared_bias=rel_logits[: nt + 128],
                num_splits=num_splits,
                use_scheduler=use_scheduler,
                out=out[:nt],
                lse=lse[:, :nt],
            )
        finally:
            _sm12_attn.set_prefill_cpu_lens(None, None)
        out[nt:].zero_()
        lse[:, nt:].zero_()
