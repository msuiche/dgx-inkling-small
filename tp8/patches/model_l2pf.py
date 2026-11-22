# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 model (NVIDIA).

Decoder layers are tied together by block attention residuals: instead of a
single residual stream, each sublayer input is a learned softmax mix over the
outputs of previous blocks of ``n_layers_per_attn_res_block`` layers plus the
running partial block. Residual blocks are stored in bf16; mixes are fp32.
"""

import os
import threading
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

import os

import regex as re
import torch
from torch import nn

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsLoRA,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
)
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingConvState
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .attention import Inkling2Attention
from .configs import Inkling2MMConfig, Inkling2ModelConfig
from .lora import (
    LORA_NAME_PREFIXES,
    LORA_NAME_SUBSTRINGS,
    PACKED_MODULES_MAPPING,
)
from .lora import register as register_lora
from .moe import Inkling2MoE, QuantFormat
from .ops import l2pf
from .ops.allreduce import Inkling2AllReduceWorkspace
from .ops.attention import compute_log_scaling_tau
from .ops.attn_res import attn_res
from .ops.moe import combine as moe_combine
from .ops.norm import rms_norm
from .ops.seq_parallel import SeqParallel
from .prefill_sp import (
    AR_OVERLAP_ENABLED,
    DBO_MAX_INFLIGHT_LAYERS,
    DBO_PIPE_SP,
    HALO_ROWS,
    SP_ENABLED,
    SP_FP8_AG,
    SP_FP8_AG_KV,
    SP_FP8_AG_MOE,
    SP_FP8_AG_MOE_MODE,
    SP_Q3_MODE,
    SP_ZERO_SYNC,
    DBO_INTERLEAVE,
    DeferredConvSyncPlan,
    GlobalKVReplWeights,
    PrefillAROverlap,
    PrefillSeqParallel,
    _tp_all_gather_dim0,
    build_global_kv_repl_weights,
    pad_prefill_sp_inputs,
    patch_forward_attn_metadata,
    patch_prefill_pad_context,
    run_sharded_sconv_attn_res,
    scatter_remote_conv_cache,
    sync_deferred_conv_caches,
)
from .processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from .quant_config import Inkling2QuantConfig
from .sconv import ATTN_STREAM, MLP_STREAM, Inkling2ShortConv, conv_metadata
from .vision import Inkling2VisionEncoder, VisionMetadata

_SYNC_FREE = os.environ.get("INKLING_SYNC_FREE", "off").strip().lower() in (
    "1", "on", "all", "true", "argmax", "audit"
)
_FUSED_EMBED_NORM = os.environ.get("INKLING_FUSED_EMBED_NORM", "0").strip().lower() in (
    "1", "on", "true", "yes"
)
_FUSED_EMBED_KERNEL = None
_TREE_SPEC = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_TREE_POS_KERNEL = None
_DIRECT_EMBED_OUT_TLS = threading.local()


def _get_tree_pos_kernel():
    global _TREE_POS_KERNEL
    if _TREE_POS_KERNEL is not None:
        return _TREE_POS_KERNEL
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _tree_apply_positions_kernel(
        pos_ptr,
        tree_active_ptr,
        tree_depth_ptr,
        n_live_ptr,
        MAX_M: tl.constexpr,
    ):
        """1-warp capture-safe post-prepare_attn tree position override:
        sets positions[i] = positions[0] + tree_depth[i] for i < n_live when tree_active[0] != 0
        (0-cost immediate return when tree_active[0] == 0)."""
        if tl.load(tree_active_ptr) == 0:
            return
        base_pos = tl.load(pos_ptr + 0)
        n_live = tl.minimum(tl.load(n_live_ptr), MAX_M)
        i = tl.arange(0, MAX_M)
        depth = tl.load(tree_depth_ptr + i).to(tl.int64)
        new_pos = tl.where(i < n_live, base_pos + depth, base_pos + i.to(tl.int64))
        tl.store(pos_ptr + i, new_pos)

    _TREE_POS_KERNEL = _tree_apply_positions_kernel
    return _TREE_POS_KERNEL


def set_direct_embed_out(buf: torch.Tensor | None) -> None:
    """One-shot thread-local destination for Inkling2ForConditionalGeneration.embed_input_ids
    (eliminates the 73 KB Memcpy DtoD inside model_state.prepare_inputs_embeds on decode)."""
    _DIRECT_EMBED_OUT_TLS.out = buf


def _get_fused_embed_kernel():
    global _FUSED_EMBED_KERNEL
    if _FUSED_EMBED_KERNEL is not None:
        return _FUSED_EMBED_KERNEL
    from vllm.triton_utils import tl, triton
    from .ops.norm import rsqrt

    @triton.jit
    def _fused_vw_repl_embed_norm_kernel(
        IDS,
        EMB_W,
        NORM_W,
        VW_MASK,
        OUT,
        stride_w,
        full_v,
        N: tl.constexpr,
        eps,
        HAS_VW: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """0-ULP fused replacement for _vw_mask + clamp + _repl_forward + embed_norm.
        Uses identical rsqrt(tl.div_rn(tl.sum(x * x), N) + eps) reduction as ops/norm.py."""
        row = tl.program_id(0).to(tl.int64)
        tok = tl.load(IDS + row).to(tl.int64)
        if HAS_VW:
            tl.store(VW_MASK + row, tok < 0)
            tok = tl.maximum(tok, 0)
        oob = (tok < 0) | (tok >= full_v)
        safe_tok = tl.minimum(tl.maximum(tok, 0), full_v - 1)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(
            EMB_W + safe_tok * stride_w + cols,
            mask=mask & (~oob),
            other=0.0,
        ).to(tl.float32)
        rstd = rsqrt(tl.div_rn(tl.sum(x * x), N) + eps)
        y = x * rstd
        y = y * tl.load(NORM_W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + row * N + cols, y.to(OUT.dtype.element_ty), mask=mask)

    _FUSED_EMBED_KERNEL = (triton, _fused_vw_repl_embed_norm_kernel)
    return _FUSED_EMBED_KERNEL


def _fused_vw_repl_embed_norm(
    input_ids: torch.Tensor,
    repl_w: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    vw_mask: torch.Tensor | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    triton, kernel = _get_fused_embed_kernel()
    t = input_ids.shape[0]
    n = repl_w.shape[1]
    if (
        out is not None
        and out.dim() == 2
        and out.shape == (t, n)
        and out.dtype == repl_w.dtype
        and out.device == input_ids.device
        and out.is_contiguous()
    ):
        out_buf = out
    else:
        out_buf = torch.empty((t, n), dtype=repl_w.dtype, device=input_ids.device)
    block = triton.next_power_of_2(n)
    kernel[(t,)](
        input_ids,
        repl_w,
        norm_w,
        vw_mask if vw_mask is not None else input_ids,
        out_buf,
        repl_w.stride(0),
        int(repl_w.shape[0]),
        n,
        eps,
        HAS_VW=vw_mask is not None,
        BLOCK=block,
        num_warps=min(max(block // 512, 1), 16),
    )
    return out_buf



class Inkling2RMSNorm(nn.Module):
    """RMSNorm; ``num_inputs > 1`` stacks the weights of per-input norms (the
    block AttnRes mixes apply them in the same kernel)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.bfloat16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size), requires_grad=False
        )
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2AttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), pre-scaled by
    ``hidden_size**-0.5`` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_local
            else config.num_key_value_heads
        )
        self.conv_state = InklingConvState(
            num_kv_heads=num_kv_heads,
            head_dim=config.swa_head_dim if self.is_local else config.head_dim,
            hidden_size=config.hidden_size,
            kernel_size=config.sconv_kernel_size,
            prefix=f"{prefix}.conv_state",
            replicate_hidden=True,
        )
        self.attn = Inkling2Attention(
            config, self.is_local, self.conv_state, prefix=f"{prefix}.attn"
        )
        eps = config.rms_norm_eps
        # Local layers: attn_norm; full layers: q/k/v_input_norm, stacked.
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2RMSNorm(
            config.hidden_size, eps, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2RMSNorm(config.hidden_size, eps)
        expert_format: QuantFormat = (
            "mxfp8"
            if quant_config is not None and quant_config.is_mxfp8_layer(layer_id)
            else "nvfp4"
        )
        self.mlp = Inkling2MoE(config, expert_format)
        self.attn_sconv, self.mlp_sconv = (
            Inkling2ShortConv(
                config.hidden_size, config.sconv_kernel_size, self.conv_state, stream
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )

    def attention(
        self,
        positions: torch.Tensor,
        inputs: Sequence[torch.Tensor],
        tau: torch.Tensor | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's partial attention output on the normalized ``inputs``."""
        if self.is_local:
            return self.attn(positions, inputs[0], out=out)
        return self.attn(positions, inputs[0], inputs[1], inputs[2], tau, out)


_ROW_AR = os.environ.get("INKLING_ROW_AR", "").strip().lower() in ("1", "on", "true", "yes")


def _rowar_mod():
    from vllm.distributed.device_communicators import rdma_all_reduce

    return rdma_all_reduce


class Inkling2Model(nn.Module):
    def _l2pf_link(self) -> None:
        from .ops import router_bf12, skinny_bf11, skinny_bf12
        if skinny_bf11.ENABLED and not torch.cuda.is_current_stream_capturing():
            tot_b = 0
            n_pk = 0
            for layer in self.layers:
                wq = layer.attn.qkvr.weight if layer.is_local else layer.attn.qkvr.weight[: sum(layer.attn.q_side_splits)]
                ws = [wq, layer.attn.wo_ud.weight]
                if not layer.is_local:
                    q_side, kv_size = sum(layer.attn.split_sizes[:4]), layer.attn.split_sizes[4]
                    k_end = q_side + kv_size
                    ws.append(layer.attn.qkvr.weight[q_side:k_end])
                    ws.append(layer.attn.qkvr.weight[k_end : k_end + kv_size])
                for w in ws:
                    p = skinny_bf11.prepack(w)
                    if p is not None:
                        n_pk += 1
                        tot_b += p.numel()
            if n_pk:
                print(f"[inkling-bf11] prepacked {n_pk} dense projections ({tot_b / 2**20:.1f} MiB)", flush=True)
            if getattr(skinny_bf11, "C32_ENABLED", False) and hasattr(skinny_bf11, "qualify_c32"):
                _q_ws = []
                for layer in self.layers:
                    _wq = layer.attn.qkvr.weight if layer.is_local else layer.attn.qkvr.weight[: sum(layer.attn.q_side_splits)]
                    _q_ws.extend([_wq, layer.attn.wo_ud.weight])
                    if not layer.is_local:
                        _qs, _kvs = sum(layer.attn.split_sizes[:4]), layer.attn.split_sizes[4]
                        _q_ws.append(layer.attn.qkvr.weight[_qs : _qs + _kvs])
                skinny_bf11.qualify_c32(_q_ws)
        if skinny_bf12.ENABLED and (not skinny_bf11.ENABLED or os.environ.get("INKLING_BF12_CONC_PREPACK", "0") == "1") and not torch.cuda.is_current_stream_capturing():
            tot_b = 0
            tot_out = 0
            n_pk = 0
            for layer in self.layers:
                wq = layer.attn.qkvr.weight if layer.is_local else layer.attn.qkvr.weight[: sum(layer.attn.q_side_splits)]
                ws = [wq, layer.attn.wo_ud.weight]
                if not layer.is_local:
                    q_side, kv_size = sum(layer.attn.split_sizes[:4]), layer.attn.split_sizes[4]
                    k_end = q_side + kv_size
                    ws.append(layer.attn.qkvr.weight[q_side:k_end])
                    ws.append(layer.attn.qkvr.weight[k_end : k_end + kv_size])
                for w in ws:
                    p = skinny_bf12.prepack(w)
                    if p is not None:
                        n_pk += 1
                        tot_b += p.numel()
                        e = skinny_bf12.get_packed(w)
                        if e is not None:
                            tot_out += int(e[2][-1].item())
            if n_pk:
                print(f"[inkling-bf12] prepacked {n_pk} dense projections ({tot_b / 2**20:.1f} MiB, {tot_out} exact inline outliers)", flush=True)
        if router_bf12.ENABLED and not torch.cuda.is_current_stream_capturing():
            n_rg = 0
            for layer in self.layers:
                if router_bf12.prepack(layer.mlp.gate.weight) is not None:
                    n_rg += 1
            if n_rg:
                print(f"[inkling-router-bf12] prepacked {n_rg} gate weights (2.65 MB/layer)", flush=True)
        # Layer i's MoE all-reduce prefetches layer i+1's live qkvr weight; last layer prefetches cert_head/lm_head.
        _dense_pf = skinny_bf11.l2pf_tensor if skinny_bf11.ENABLED else skinny_bf12.l2pf_tensor
        for a, b in zip(self.layers[:-1], self.layers[1:]):
            wq = b.attn.qkvr.weight if b.is_local else b.attn.qkvr.weight[: sum(b.attn.q_side_splits)]
            pf_w = _dense_pf(wq)
            a.mlp.experts.l2pf_next = pf_w
            a.mlp.experts.l2pf_next_raw = wq
            l2pf.prewarm(pf_w, kind="moe")
            if l2pf.C32:
                for _m in (32, 64, 128, 256):
                    if _m <= l2pf.C32_MAX_M:
                        l2pf.prewarm(l2pf.resolve_dense_next(a.mlp.experts, _m), kind="moe", num_tokens=_m)
        self.layers[-1].mlp.experts.l2pf_last = True
        for layer in self.layers:
            gw_pf = router_bf12.l2pf_tensor(layer.mlp.gate.weight) if router_bf12.ENABLED else layer.mlp.gate.weight
            layer._l2pf_gate_w = gw_pf
            l2pf.prewarm(gw_pf, kind="gate")
            l2pf.prewarm(_dense_pf(layer.attn.wo_ud.weight), kind="attn")
            if l2pf.C32:
                for _m in (32, 64, 128, 256):
                    if _m <= l2pf.C32_MAX_M:
                        l2pf.prewarm(l2pf.resolve_gate_target(layer, _m), kind="gate", num_tokens=_m)
                        if l2pf.C32_WO:
                            _wo_m = (
                                skinny_bf11.l2pf_tensor(layer.attn.wo_ud.weight, num_tokens=_m)
                                if skinny_bf11.ENABLED
                                else layer.attn.wo_ud.weight
                            )
                            l2pf.prewarm(_wo_m, kind="attn", num_tokens=_m)
        l2pf.prewarm(l2pf.get_head_weight(num_tokens=8), kind="head")
        l2pf.prewarm(l2pf.get_head_weight(num_tokens=1), kind="head")
        if l2pf.C32:
            for _m in (32, 64, 128, 256):
                if _m <= l2pf.C32_MAX_M:
                    l2pf.prewarm(l2pf.get_head_weight(num_tokens=_m), kind="head", num_tokens=_m)
        from . import sconv as _sc_mod
        _sc_mod._get_fused_sconv_ar()
        try:
            from .ops import moe_s8_runtime as _s8
            if (
                (_s8.S8_ROUTER or _s8.S8_ZERO_SCATTER or _s8.S8_FUSE_SITU or _s8.S8_FUSE_COMBINE)
                and not torch.cuda.is_current_stream_capturing()
            ):
                _s8.get_workspace(self.layers[0].mlp.gate.weight.device)
                _n_tagged = 0
                for _layer in self.layers:
                    _exp = _layer.mlp.experts
                    if _exp.nvfp4:
                        for _w, _ws in (
                            (_exp.w13_weight, _exp.w13_weight_scale),
                            (_exp.w2_weight, _exp.w2_weight_scale),
                        ):
                            _m = getattr(_w, "_inkling_nvfp4_ws5", None)
                            if _m is not None:
                                _s8.get_ws_base_s8(_m)
                                _n_tagged += 1
                print(f"[inkling-moe-s8] prewarmed S8 decode workspace (19.3 MiB) + {_n_tagged} WS_BASE bit-7 tables", flush=True)
        except Exception:
            pass
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.empty_cache()
        self._l2pf_linked = True

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = Inkling2RMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=None
        )
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, i, quant_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        # INKLING_VERIFY_W (adaptive verify width): the speculator marks the
        # unverified tail drafts with token id -1. embed_input_ids records
        # those rows in this mask; forward() ORs it into the forward
        # context's is_padding (MoE routes padding rows to no expert).
        self._vw = os.environ.get("INKLING_VERIFY_W", "").strip().lower() not in (
            "",
            "0",
            "off",
        )
        if self._vw:
            self.register_buffer(
                "_vw_mask",
                torch.zeros(
                    get_current_vllm_config().scheduler_config.max_num_batched_tokens,
                    dtype=torch.bool,
                ),
                persistent=False,
            )
        # Large batches keep a token shard of the hidden stream on each rank.
        self.allreduce_workspace = None
        self.seq_parallel = None
        self.prefill_sp: PrefillSeqParallel | None = None
        self.prefill_dbo: PrefillAROverlap | None = None
        if get_tensor_model_parallel_world_size() > 1:
            assert self.normalize_values
            group = get_tp_group().cpu_group
            self.allreduce_workspace = Inkling2AllReduceWorkspace.create(
                group, config.hidden_size
            )
            vllm_config = get_current_vllm_config()
            spec = vllm_config.speculative_config
            num_spec = 0 if spec is None else spec.num_speculative_tokens
            tail_rows = max(config.sconv_kernel_size - 1, num_spec + 1)
            if self.allreduce_workspace is not None:
                self.seq_parallel = SeqParallel(
                    group,
                    config.hidden_size,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    tail_rows=tail_rows,
                )
            elif SP_ENABLED:
                self.prefill_sp = PrefillSeqParallel(
                    config.hidden_size,
                    tail_rows=tail_rows,
                )
            if self.allreduce_workspace is None and AR_OVERLAP_ENABLED:
                self.prefill_dbo = PrefillAROverlap(torch.device("cuda"))
        # Layers whose input (the partial block) the DSpark draft reads; layer
        # id num_hidden_layers stands for the final-norm output.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(
        self, input_ids: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        repl_w = getattr(self.embed_tokens, "_repl_weight", None)
        if (
            repl_w is not None
            and _FUSED_EMBED_NORM
            and input_ids.dim() == 1
            and input_ids.is_contiguous()
            and input_ids.shape[0] > 0
            and (not self._vw or input_ids.shape[0] <= self._vw_mask.shape[0])
        ):
            return _fused_vw_repl_embed_norm(
                input_ids,
                repl_w,
                self.embed_norm.weight[0],
                self.embed_norm.eps,
                self._vw_mask if self._vw else None,
                out=out,
            )
        if self._vw and input_ids.shape[0] <= self._vw_mask.shape[0]:
            masked = input_ids < 0  # unverified tail drafts (speculator.py)
            self._vw_mask[: input_ids.shape[0]].copy_(masked)
            input_ids = input_ids.clamp(min=0)
        return self.embed_norm(self.embed_tokens(input_ids))

    def _mix(
        self,
        stack: torch.Tensor,
        last: torch.Tensor,
        query: torch.Tensor,
        norm: Inkling2RMSNorm,
        addend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return attn_res(
            stack, last, addend, query, norm.weight, norm.eps, self.normalize_values
        )

    def _ensure_tree_compact_groups(self, tree_st: Any, attn_metadata: dict[str, Any]) -> None:
        """Build (once, outside CUDA graph capture) per-KV/Conv-group device pointer tables on
        `tree_st.compact_groups` and pre-JIT `_tree_apply_positions_kernel`."""
        groups = tree_st.compact_groups
        if groups is not None:
            for grp in groups:
                a_md = attn_metadata.get(grp["rep_attn_prefix"])
                c_md = attn_metadata.get(grp["rep_conv_prefix"])
                if a_md is not None and a_md.slot_mapping.data_ptr() != grp["kv_slot_map"].data_ptr():
                    grp["kv_slot_map"] = a_md.slot_mapping
                if c_md is not None and c_md.slot_mapping.data_ptr() != grp["conv_slot_map"].data_ptr():
                    grp["conv_slot_map"] = c_md.slot_mapping
            return
        by_key: dict[tuple, dict[str, Any]] = {}
        order: list[tuple] = []
        for layer in self.layers:
            if layer.attn.kv_cache.numel() == 0 or layer.conv_state.kv_cache.numel() == 0:
                return
            a_md = attn_metadata.get(layer.attn.prefix)
            c_md = attn_metadata.get(layer.conv_state.prefix)
            if a_md is None or c_md is None:
                return
            kc, vc = layer.attn._split_kv_cache()
            cc = layer.conv_state.kv_cache
            sr = layer.conv_state.stream_ranges
            conv_elems = int(sr[-1][0] + sr[-1][1])
            key = (
                int(a_md.slot_mapping.data_ptr()),
                int(c_md.slot_mapping.data_ptr()),
                int(kc.stride(0)),
                int(kc.stride(1)),
                int(kc.shape[1]),
                int(kc.shape[2] * kc.shape[3]),
                int(cc.stride(0)),
                int(cc.stride(2)),
                int(cc.shape[2]),
                conv_elems,
            )
            entry = by_key.get(key)
            if entry is None:
                entry = {
                    "kc_list": [],
                    "vc_list": [],
                    "cc_list": [],
                    "rep_attn_prefix": layer.attn.prefix,
                    "rep_conv_prefix": layer.conv_state.prefix,
                    "kv_slot_map": a_md.slot_mapping,
                    "conv_slot_map": c_md.slot_mapping,
                    "stride_kv_blk": key[2],
                    "stride_kv_n": key[3],
                    "kv_bs": key[4],
                    "kv_elems": key[5],
                    "stride_c_blk": key[6],
                    "stride_c_n": key[7],
                    "c_bs": key[8],
                    "conv_elems": key[9],
                }
                by_key[key] = entry
                order.append(key)
            entry["kc_list"].append(int(kc.data_ptr()))
            entry["vc_list"].append(int(vc.data_ptr()))
            entry["cc_list"].append(int(cc.data_ptr()))
        built: list[dict[str, Any]] = []
        dev = tree_st.device
        for key in order:
            e = by_key[key]
            e["kc_ptrs"] = torch.tensor(e.pop("kc_list"), dtype=torch.int64, device=dev)
            e["vc_ptrs"] = torch.tensor(e.pop("vc_list"), dtype=torch.int64, device=dev)
            e["cc_ptrs"] = torch.tensor(e.pop("cc_list"), dtype=torch.int64, device=dev)
            e["num_layers"] = int(e["kc_ptrs"].shape[0])
            built.append(e)
        tree_st.compact_groups = built

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if not _ROW_AR and not _TREE_SPEC:
            return self._forward_impl(positions, inputs_embeds)
        suppress_tree = False
        tsr_mod = None
        if _TREE_SPEC:
            from .ops import tree_spec_runtime as tsr_mod
            if tsr_mod.TREE_STATE.get("scope") == inputs_embeds.shape[0]:
                is_decode_b1 = False
                if is_forward_context_available():
                    _amd = get_forward_context().attn_metadata
                    if isinstance(_amd, dict) and len(self.layers) > 0:
                        _md0 = _amd.get(self.layers[0].attn.prefix)
                        if (
                            _md0 is not None
                            and getattr(_md0, "num_prefill_tokens", 0) == 0
                            and getattr(_md0, "num_decode_reqs", 0) == 1
                        ):
                            is_decode_b1 = True
                if not is_decode_b1:
                    suppress_tree = True
                    tsr_mod.TREE_STATE["suppressed"] = True
        # INKLING_ROW_AR: the row scope (set in _forward_impl) must not leak into the drafter's or any later forward.
        try:
            return self._forward_impl(positions, inputs_embeds)
        finally:
            if suppress_tree and tsr_mod is not None:
                tsr_mod.TREE_STATE["suppressed"] = False
            if _ROW_AR:
                _rowar_mod().rowar_end()

    def _forward_impl(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens, hidden = inputs_embeds.shape
        if num_tokens > 16 and torch.cuda.is_current_stream_capturing():
            from vllm.distributed.device_communicators import rdma_all_reduce as _rar_mod
            _reset_ring = getattr(_rar_mod, 'reset_send_buffer_ring', None)
            if _reset_ring is not None:
                _reset_ring(0)
        if _ROW_AR:
            _rowar_mod().ROWAR["last_live"] = None
        if self._vw:
            # First in-graph op: rows embed_input_ids flagged become padding
            # (no expert rows). The mask is consumed (zeroed) so a forward
            # without a fresh embed_input_ids (dummy / capture runs) never
            # sees a stale one. The runner re-fills is_padding every step.
            vw_mask = self._vw_mask[:num_tokens]
            pad = None
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                _p = get_forward_context().is_padding
                if _p is not None and _p.shape[0] == num_tokens:
                    pad = _p
            fused_done = False
            if _ROW_AR:
                # Target graph entry 3->1 fusion: pad.logical_or_(vw_mask) + _n_live_kernel + vw_mask.zero_()
                _rm = _rowar_mod()
                if getattr(_rm, "_ROWAR_FUSED_ENTRY", False):
                    fused_done = bool(_rm.rowar_begin(num_tokens, vw_mask, pad=pad, zero_mask=True))
                else:
                    if pad is not None:
                        pad.logical_or_(vw_mask)
                    _rm.rowar_begin(num_tokens, vw_mask)
                    vw_mask.zero_()
                    fused_done = True
            if not fused_done:
                if pad is not None:
                    pad.logical_or_(vw_mask)
                vw_mask.zero_()
        aux_hidden_states = []
        num_layers = len(self.layers)
        # Wave 1A GB10 eager prefill fast paths (strictly opt-in via env vars,
        # never active during CUDA graph capture/replay or decode T < MIN_T):
        #   1. INKLING_PREFILL_AR_OVERLAP=1: Dual-Microbatch (T/2+T/2) AR overlap
        #      (optionally composed with INKLING_PREFILL_SP=1).
        #   2. INKLING_PREFILL_SP=1: Sequence-parallel sharded residual stream
        #      (8x smaller stacked RAM + 8x faster sharded sconv/AttnRes).
        if self.allreduce_workspace is None:
            tp_size = get_tensor_model_parallel_world_size()
            fc_avail = is_forward_context_available()
            fc_obj = get_forward_context() if fc_avail else None
            fc_meta = fc_obj.attn_metadata if fc_obj is not None else None
            first_attn_prefix = self.layers[0].attn.prefix

            def _slice_padded_out(res):
                if isinstance(res, tuple):
                    out_t, aux_list = res
                    return out_t[:num_tokens].contiguous(), [a[:num_tokens].contiguous() for a in aux_list]
                return res[:num_tokens].contiguous()

            if (
                self.prefill_dbo is not None
                and fc_avail
                and PrefillAROverlap.eligible(
                    num_tokens,
                    tp_size,
                    fc_meta,
                    first_attn_prefix,
                )
            ):
                return self._forward_prefill_dbo(positions, inputs_embeds)
            if self.prefill_dbo is not None and fc_avail:
                t_pad_dbo = PrefillAROverlap.pad_target(
                    num_tokens, tp_size, fc_meta, first_attn_prefix
                )
                if t_pad_dbo > 0:
                    pos_p, emb_p, meta_p, pad_mask = pad_prefill_sp_inputs(
                        positions,
                        inputs_embeds,
                        fc_meta,
                        getattr(fc_obj, "is_padding", None),
                        t_pad_dbo,
                    )
                    with patch_prefill_pad_context(meta_p, pad_mask):
                        res_p = self._forward_prefill_dbo(
                            pos_p, emb_p, actual_num_tokens=num_tokens
                        )
                    return _slice_padded_out(res_p)
            if self.prefill_sp is not None and self.prefill_sp.eligible(num_tokens):
                return self._forward_prefill_sp(positions, inputs_embeds)
            if self.prefill_sp is not None:
                t_pad_sp = self.prefill_sp.pad_target(
                    num_tokens, fc_meta, first_attn_prefix
                )
                if t_pad_sp > 0:
                    pos_p, emb_p, meta_p, pad_mask = pad_prefill_sp_inputs(
                        positions,
                        inputs_embeds,
                        fc_meta,
                        getattr(fc_obj, "is_padding", None) if fc_obj is not None else None,
                        t_pad_sp,
                    )
                    if fc_avail:
                        with patch_prefill_pad_context(meta_p, pad_mask):
                            res_p = self._forward_prefill_sp(pos_p, emb_p)
                    else:
                        res_p = self._forward_prefill_sp(pos_p, emb_p)
                    return _slice_padded_out(res_p)
        # With a conv-state cache (TP > 1), the hidden-stream conv runs fused
        # with the next AttnRes mix: small batches also fuse each sublayer's
        # all-reduce, larger ones run it on the all-reduced input. Large
        # batches keep a token shard of the hidden stream on each rank
        # (reduce-scatter, then `conv_mix`), also in profile runs (no
        # conv-state cache) for their memory use.
        workspace = self.allreduce_workspace
        plan = [(False, False)] * 2
        has_cache = conv_metadata(self.layers[0].conv_state) is not None
        sp = self.seq_parallel
        if sp is not None and num_tokens < sp.min_tokens:
            sp = None
        conv_mix = workspace is not None and has_cache
        if conv_mix and sp is None:
            assert workspace is not None
            plan = workspace.plan(num_tokens)
        (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
        # stack[:num_blocks] holds completed blocks; stack[num_blocks] is the
        # running partial block (of this rank's shard of rows[:capacity]).
        max_blocks = -(-num_layers // self.block_size)
        begin, end, capacity = 0, num_tokens, num_tokens
        prefix_align = 0
        if sp is not None:
            begin, end, capacity = sp.shard(num_tokens)
            prefix_align = self._prefix_align() if has_cache else 0
        stacked = inputs_embeds.new_empty((max_blocks + 1, capacity, hidden))
        stack = stacked[:, : end - begin]
        stack[0].copy_(inputs_embeds[begin:end])
        if _TREE_SPEC:
            from .ops import tree_spec_runtime as _tsr

            _tree_st = _tsr.get_active_tree_state(num_tokens)
            if (
                _tree_st is not None
                and positions.dim() == 1
                and positions.stride(0) == 1
                and positions.shape[0] >= num_tokens
            ):
                if has_cache and is_forward_context_available() and not torch.cuda.is_current_stream_capturing():
                    _amd = get_forward_context().attn_metadata
                    if isinstance(_amd, dict):
                        self._ensure_tree_compact_groups(_tree_st, _amd)
                _get_tree_pos_kernel()[(1,)](
                    positions,
                    _tree_st.tree_active,
                    _tree_st.tree_depth,
                    _tree_st.n_live,
                    MAX_M=_tree_st.max_m,
                    num_warps=1,
                )
        num_blocks = 0
        tau = None
        if self.config.log_scaling_n_floor is not None:
            tau = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )
        # The fused MoE all-reduce also produces the next layer's inputs.
        inputs = None
        pending_aux = None
        for i, layer in enumerate(self.layers):
            partial = stack[num_blocks]
            if i in self.aux_hidden_state_layers:
                if sp is None:
                    if pending_aux is not None:
                        aux_hidden_states.append(pending_aux)
                        pending_aux = None
                    elif i % self.block_size == 0:
                        aux_hidden_states.append(partial)
                    else:
                        aux_hidden_states.append(partial.clone())
                else:
                    aux_hidden_states.append(
                        tensor_model_parallel_all_gather(stacked[num_blocks], 0)[
                            :num_tokens
                        ]
                    )
            if inputs is None and num_blocks == 0:
                # The shard of the first block holds this rank's rows only.
                first_in = partial if sp is None else inputs_embeds
                inputs = [
                    layer.attn_norm(first_in, j) for j in range(layer.num_attn_inputs)
                ]
            elif inputs is None:
                query = self.attn_res.attn_queries[i - 1]
                inputs = self._mix(stack[:num_blocks], partial, query, layer.attn_norm)
            attn_out_buf = (
                sp.partial(num_tokens)
                if sp is not None
                else (
                    _rowar_mod().get_send_buffer((num_tokens, hidden), inputs_embeds.dtype)
                    if not conv_mix
                    else None
                )
            )
            attn_partial = layer.attention(
                positions,
                inputs,
                tau,
                out=attn_out_buf,
            )

            mlp_query = self.attn_res.mlp_queries[i]
            # The MLP input mixes the blocks with (partial +) attn_out.
            first = i % self.block_size == 0
            if sp is not None or conv_mix:
                attn_out = torch.empty_like(partial)
                mix_args = (
                    positions,
                    attn_out,
                    stack[: num_blocks + first],
                    None if first else partial,
                    mlp_query,
                    layer.mlp_norm,
                )
                if sp is not None:
                    shard, halo = sp.reduce_scatter(num_tokens)
                    mlp_in = layer.attn_sconv.conv_mix(
                        shard, *mix_args, sp=sp, halo=halo, prefix_align=prefix_align
                    )
                elif attn_fused:
                    assert workspace is not None
                    mlp_in = layer.attn_sconv.all_reduce_mix(
                        workspace, *mix_args, x=attn_partial, two_shot=attn_two_shot
                    )
                else:
                    attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
                    mlp_in = layer.attn_sconv.conv_mix(attn_reduced, *mix_args)
            else:
                if not getattr(self, "_l2pf_linked", False):
                    self._l2pf_link()
                if l2pf.GATE and num_tokens <= 16:
                    l2pf.fork(getattr(layer, "_l2pf_gate_w", layer.mlp.gate.weight), kind="gate")
                elif l2pf.C32 and l2pf.GATE and 17 <= num_tokens <= l2pf.C32_MAX_M:
                    l2pf.fork(l2pf.resolve_gate_target(layer, num_tokens), kind="gate", num_tokens=num_tokens)
                attn_out = torch.empty_like(partial)
                wo_stack = stack[: num_blocks + 1] if first else stack[:num_blocks]
                wo_mix_last = None if first else partial
                mlp_in = layer.attn_sconv.try_fused_ar_sconv_attn_res(
                    attn_partial,
                    positions,
                    attn_out,
                    wo_stack,
                    wo_mix_last,
                    mlp_query,
                    layer.mlp_norm,
                    self.normalize_values,
                )
                if mlp_in is not None:
                    l2pf.join()
                else:
                    attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
                    l2pf.join()
                    mlp_in = layer.attn_sconv.try_fused_sconv_attn_res(
                        attn_reduced,
                        positions,
                        attn_out,
                        wo_stack,
                        wo_mix_last,
                        mlp_query,
                        layer.mlp_norm,
                        self.normalize_values,
                    )
                    if mlp_in is None:
                        attn_out = layer.attn_sconv(attn_reduced, positions, out=attn_out)
                        if first:
                            mlp_in = self._mix(
                                stack[: num_blocks + 1], attn_out, mlp_query, layer.mlp_norm
                            )
                        else:
                            mlp_in = self._mix(
                                stack[:num_blocks], partial, mlp_query, layer.mlp_norm, attn_out
                            )
            # The MLP short conv also applies the residual update.
            residual = None if first else partial
            if residual is None:
                num_blocks += 1
            inputs = None
            if sp is None and not conv_mix:
                if i + 1 < num_layers:
                    nxt_query = self.attn_res.attn_queries[i]
                    nxt_norm = self.layers[i + 1].attn_norm
                else:
                    nxt_query, nxt_norm = self.attn_res.output_query, self.norm
                if layer.mlp_sconv.can_fused_ar_sconv_attn_res(
                    num_tokens, hidden, stack[:num_blocks], nxt_query, nxt_norm, self.normalize_values
                ):
                    moe_partial = layer.mlp(mlp_in[0], reduce=False, combine_only=True, use_sendbuf=True)
                    _exp = layer.mlp.experts
                    if num_tokens <= 16:
                        _nxt_pf = getattr(_exp, "l2pf_next", None)
                        if _nxt_pf is not None:
                            l2pf.fork(_nxt_pf, kind="moe")
                        elif getattr(_exp, "l2pf_last", False):
                            l2pf.fork(l2pf.get_head_weight(num_tokens), kind="head")
                    elif l2pf.C32 and num_tokens <= l2pf.C32_MAX_M:
                        _nxt_pf = l2pf.resolve_dense_next(_exp, num_tokens)
                        if _nxt_pf is not None:
                            l2pf.fork(_nxt_pf, kind="moe", num_tokens=num_tokens)
                        elif getattr(_exp, "l2pf_last", False):
                            l2pf.fork(l2pf.get_head_weight(num_tokens), kind="head", num_tokens=num_tokens)
                    nxt_aux = None
                    if (
                        (i + 1) in self.aux_hidden_state_layers
                        and (i + 1) % self.block_size != 0
                    ):
                        nxt_aux = torch.empty_like(stack[num_blocks])
                    inputs = layer.mlp_sconv.try_fused_ar_sconv_attn_res(
                        moe_partial,
                        positions,
                        stack[num_blocks],
                        stack[:num_blocks],
                        None,
                        nxt_query,
                        nxt_norm,
                        self.normalize_values,
                        conv_addend=attn_out,
                        conv_res=residual,
                        aux_out=nxt_aux,
                    )
                    l2pf.join()
                    if inputs is not None:
                        if nxt_aux is not None:
                            pending_aux = nxt_aux
                        continue
                    moe_reduced = tensor_model_parallel_all_reduce(moe_partial)
                else:
                    moe_reduced = layer.mlp(mlp_in[0])
                need_aux = (
                    (i + 1) in self.aux_hidden_state_layers
                    and (i + 1) % self.block_size != 0
                )
                nxt_aux = torch.empty_like(stack[num_blocks]) if (need_aux and num_tokens >= 17) else None
                inputs = layer.mlp_sconv.try_fused_sconv_attn_res(
                    moe_reduced,
                    positions,
                    stack[num_blocks],
                    stack[:num_blocks],
                    None,
                    nxt_query,
                    nxt_norm,
                    self.normalize_values,
                    conv_addend=attn_out,
                    conv_res=residual,
                    aux_out=nxt_aux,
                )
                if inputs is not None:
                    if nxt_aux is not None:
                        pending_aux = nxt_aux
                else:
                    if need_aux and nxt_aux is None:
                        nxt_aux = torch.empty_like(stack[num_blocks])
                    if nxt_aux is not None:
                        pending_aux = nxt_aux
                    layer.mlp_sconv(
                        moe_reduced,
                        positions,
                        addend=attn_out,
                        residual=residual,
                        out=stack[num_blocks],
                        aux_out=nxt_aux,
                    )
                continue
            if i + 1 < num_layers:
                query = self.attn_res.attn_queries[i]
                norm = self.layers[i + 1].attn_norm
            else:
                query, norm = self.attn_res.output_query, self.norm
            mix_args = (
                positions,
                stack[num_blocks],
                stack[:num_blocks],
                None,
                query,
                norm,
            )
            if sp is not None:
                layer.mlp(mlp_in[0], reduce=False, push=sp)
                shard, halo = sp.pushed(num_tokens)
                inputs = layer.mlp_sconv.conv_mix(
                    shard,
                    *mix_args,
                    addend=attn_out,
                    residual=residual,
                    sp=sp,
                    halo=halo,
                    prefix_align=prefix_align,
                )
            elif moe_fused:
                assert workspace is not None
                inputs = layer.mlp_sconv.all_reduce_mix(
                    workspace,
                    *mix_args,
                    combine=layer.mlp(mlp_in[0], reduce=False),
                    addend=attn_out,
                    residual=residual,
                    two_shot=moe_two_shot,
                )
            else:
                # The MoE output is all-reduced (in bf16) inside layer.mlp.
                moe_reduced = layer.mlp(mlp_in[0])
                inputs = layer.mlp_sconv.conv_mix(
                    moe_reduced, *mix_args, addend=attn_out, residual=residual
                )

        if inputs is not None:
            # The all-gather buffer is reused by the next forward.
            out = inputs[0] if sp is None else inputs[0].clone()
        else:
            out = self._mix(
                stack[:num_blocks],
                stack[num_blocks],
                self.attn_res.output_query,
                self.norm,
            )[0]
        l2pf.join_end()
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out

    def _prefix_align(self) -> int:
        """A token multiple of every prefix-cache hit (a KV cache block size
        of the full-attention layers)."""
        layer = next(layer for layer in self.layers if not layer.is_local)
        return layer.attn.kv_cache.shape[2]

    @torch.no_grad()
    def _ensure_global_kv_repl(self) -> None:
        """Materialize replicated `[W*(kv+w), H]` global K/V + dyn weights once across TP
        when `INKLING_PREFILL_SP_Q3=repl_kv` (~148.5 MiB total across 8 global layers)."""
        if getattr(self, "_global_kv_repl_ready", False):
            return
        self._global_kv_repl: dict[int, GlobalKVReplWeights] = {}
        self._global_kv_repl_ready = True
        tp = get_tensor_model_parallel_world_size()
        if not (SP_ENABLED and SP_Q3_MODE == "repl_kv" and tp > 1):
            return
        if torch.cuda.is_current_stream_capturing():
            self._global_kv_repl_ready = False
            return
        tot_b = 0
        for idx, layer in enumerate(self.layers):
            if not layer.is_local:
                rw = build_global_kv_repl_weights(layer.attn, tp)
                if rw is not None:
                    self._global_kv_repl[idx] = rw
                    tot_b += (rw.wkd_all.numel() + rw.wvd_all.numel()) * rw.wkd_all.element_size()
        if self._global_kv_repl:
            print(
                f"[inkling-prefill-sp] repl_kv replicated {len(self._global_kv_repl)} global layers' "
                f"K/V+dyn weights ({tot_b / 2**20:.1f} MiB) across TP={tp}; zero x_k/x_v AllGather!",
                flush=True,
            )

    def _get_global_kv_repl(self, layer_idx: int) -> GlobalKVReplWeights | None:
        repl = getattr(self, "_global_kv_repl", None)
        return None if repl is None else repl.get(layer_idx)

    @staticmethod
    def _moe_partial(layer: Inkling2DecoderLayer, x: torch.Tensor) -> torch.Tensor:
        """Compute this rank's unreduced bf16 `[T, H]` MoE partial sum.
        Uses fused W2-Combine automatically when Phase 2 `moe.py` (`combine_only=True`) and
        `INKLING_PF_MOE_NATIVE=1` are active; cleanly falls back to stock `(down, src2dst)`."""
        try:
            return layer.mlp(x, reduce=False, combine_only=True)
        except TypeError:
            down, src2dst = layer.mlp(x, reduce=False)
            out_dtype = torch.bfloat16 if x.dtype == torch.float8_e4m3fn else x.dtype
            return moe_combine(down, src2dst, layer.mlp.experts.topk, out_dtype)

    def _run_unsharded_sconv_ar(
        self,
        sconv_mod: Inkling2ShortConv,
        x_reduced: torch.Tensor,
        positions: torch.Tensor,
        conv_out: torch.Tensor,
        stack: torch.Tensor,
        mix_last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: Inkling2RMSNorm,
        conv_addend: torch.Tensor | None = None,
        conv_res: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unsharded fused (or unfused fallback) short_conv + AttnRes + RMSNorm -> `[Q, T, H]`."""
        out = sconv_mod.try_fused_sconv_attn_res(
            x_reduced,
            positions,
            conv_out,
            stack,
            mix_last,
            queries,
            norm,
            self.normalize_values,
            conv_addend=conv_addend,
            conv_res=conv_res,
        )
        if out is not None:
            return out
        sconv_mod(
            x_reduced,
            positions,
            addend=conv_addend,
            residual=conv_res,
            out=conv_out,
        )
        if mix_last is not None:
            return self._mix(stack, mix_last, queries, norm, conv_out)
        return self._mix(stack, conv_out, queries, norm)

    def _forward_prefill_sp(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Wave 1A GB10 Sequence-Parallel eager prefill (`INKLING_PREFILL_SP=1|nccl|exact|rdma`).

        Keeps the AttnRes residual stack sharded `[max_blocks + 1, T // 8, H]` (8x less RAM
        and 8x less sconv + AttnRes DRAM traffic across all 96 sublayers) while keeping
        `conv_state.kv_cache` 100% bit-identical across all ranks via a single tiny RDMA
        halo+boundary+tail all-gather per sublayer.
        """
        pf_sp = self.prefill_sp
        assert pf_sp is not None
        num_tokens, hidden = inputs_embeds.shape
        num_layers = len(self.layers)
        if not getattr(self, "_l2pf_linked", False):
            self._l2pf_link()
        if SP_Q3_MODE == "repl_kv":
            self._ensure_global_kv_repl()

        conv_md = conv_metadata(self.layers[0].conv_state)
        has_cache = conv_md is not None
        n_cache = int(self.layers[0].conv_state.kv_cache.shape[2]) if has_cache else 4
        prefix_align = int(self._prefix_align()) if has_cache else 0
        plan = pf_sp.build_plan(
            num_tokens=num_tokens,
            positions=positions,
            conv_md=conv_md,
            n_cache=n_cache,
            prefix_align=prefix_align,
        )
        def_plan = None if SP_ZERO_SYNC else DeferredConvSyncPlan.from_plans([plan])

        max_blocks = -(-num_layers // self.block_size)
        stack = inputs_embeds.new_empty((max_blocks + 1, plan.shard_size, hidden))
        stack[0].copy_(inputs_embeds[plan.begin : plan.end])
        num_blocks = 0

        tau = None
        if self.config.log_scaling_n_floor is not None:
            tau = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )

        aux_hidden_states: list[torch.Tensor] = []
        inputs: list[torch.Tensor] = [
            self.layers[0].attn_norm(inputs_embeds, j)
            for j in range(self.layers[0].num_attn_inputs)
        ]

        for i, layer in enumerate(self.layers):
            partial = stack[num_blocks]
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(_tp_all_gather_dim0(partial))

            attn_partial = layer.attention(positions, inputs, tau)
            first = i % self.block_size == 0
            attn_out = torch.empty_like(partial)
            mlp_in = pf_sp.sublayer_step(
                partial=attn_partial,
                sconv_mod=layer.attn_sconv,
                plan=plan,
                conv_out=attn_out,
                stack=stack[: num_blocks + 1] if first else stack[:num_blocks],
                mix_last=None if first else partial,
                queries=self.attn_res.mlp_queries[i],
                norm=layer.mlp_norm,
                normalize_values=self.normalize_values,
                fp8_ag_modes=(SP_FP8_AG_MOE_MODE,) if (SP_FP8_AG and SP_FP8_AG_MOE) else None,
            )

            residual = None if first else partial
            if residual is None:
                num_blocks += 1

            moe_partial = self._moe_partial(layer, mlp_in[0])
            if i + 1 < num_layers:
                nxt_query = self.attn_res.attn_queries[i]
                nxt_norm = self.layers[i + 1].attn_norm
                nxt_repl = self._get_global_kv_repl(i + 1)
                if SP_FP8_AG:
                    kv_m = "dequant" if SP_FP8_AG_KV else "bf16"
                    nxt_ag_modes: tuple[str, ...] | None = (
                        ("direct",) if self.layers[i + 1].is_local else ("direct", kv_m, kv_m)
                    )
                else:
                    nxt_ag_modes = None
            else:
                nxt_query, nxt_norm = self.attn_res.output_query, self.norm
                nxt_repl = None
                nxt_ag_modes = None

            inputs = pf_sp.sublayer_step(
                partial=moe_partial,
                sconv_mod=layer.mlp_sconv,
                plan=plan,
                conv_out=stack[num_blocks],
                stack=stack[:num_blocks],
                mix_last=None,
                queries=nxt_query,
                norm=nxt_norm,
                normalize_values=self.normalize_values,
                conv_addend=attn_out,
                conv_res=residual,
                next_global_repl=nxt_repl,
                fp8_ag_modes=nxt_ag_modes,
            )

        if def_plan is None and SP_ZERO_SYNC:
            def_plan = DeferredConvSyncPlan.from_plans([plan])
        sync_deferred_conv_caches(self.layers, def_plan)
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out

    def _forward_prefill_dbo(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        actual_num_tokens: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Wave 1A Dual-Microbatch (`T/2 + T/2`) Overlapped Prefill All-Reduce
        (`INKLING_PREFILL_AR_OVERLAP=1`), optionally composed with `INKLING_PREFILL_SP=1`.

        Pipelines micro-batch A=[0:T/2) and B=[T/2:T) compute on `compute_stream` against
        TP collectives on a single dedicated high-priority `comm_stream` (strictly serialized
        NCCL queue -> zero concurrent `ncclComm_t` usage, deadlock-free; explicit CUDA events
        and `.record_stream(...)` on every cross-stream tensor).
        """
        dbo = self.prefill_dbo
        assert dbo is not None
        num_tokens, hidden = inputs_embeds.shape
        t0 = num_tokens // 2
        t1 = num_tokens - t0
        num_layers = len(self.layers)
        if not getattr(self, "_l2pf_linked", False):
            self._l2pf_link()

        compute_stream = torch.cuda.current_stream(inputs_embeds.device)
        orig_meta = get_forward_context().attn_metadata
        assert isinstance(orig_meta, dict)
        meta_a, meta_b = PrefillAROverlap.split_solo_metadata(
            orig_meta, num_tokens, t0, inputs_embeds.device, actual_num_tokens=actual_num_tokens
        )
        pos_a = positions[:t0].contiguous()
        pos_b = positions[t0:].contiguous()
        slice_a = slice(0, t0)
        slice_b = slice(t0, num_tokens)

        tau_a = tau_b = None
        if self.config.log_scaling_n_floor is not None:
            tau = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )
            tau_a = tau[:t0].contiguous()
            tau_b = tau[t0:].contiguous()

        pf_sp = self.prefill_sp
        use_sp = pf_sp is not None and pf_sp.eligible(t0)
        plan_a = plan_b = None
        def_plan = None
        if use_sp:
            assert pf_sp is not None
            if SP_Q3_MODE == "repl_kv":
                self._ensure_global_kv_repl()
            n_cache = int(self.layers[0].conv_state.kv_cache.shape[2])
            prefix_align = int(self._prefix_align())
            conv_key = self.layers[0].conv_state.prefix
            # Micro-batch A is always followed by prefill micro-batch B (never decode/verify rollback),
            # so A only needs tail_rows=HALO_ROWS (3) instead of 8 (saves 5 wasted rows/rank!).
            plan_a = pf_sp.build_plan(
                t0, pos_a, meta_a.get(conv_key), n_cache, prefix_align,
                tail_rows_override=HALO_ROWS,
            )
            plan_b = pf_sp.build_plan(t1, pos_b, meta_b.get(conv_key), n_cache, prefix_align)
            def_plan = None if SP_ZERO_SYNC else DeferredConvSyncPlan.from_plans([plan_a, plan_b])
            row_a, row_b = plan_a.shard_size, plan_b.shard_size
        else:
            row_a, row_b = t0, t1

        max_blocks = -(-num_layers // self.block_size)
        stack_a = inputs_embeds.new_empty((max_blocks + 1, row_a, hidden))
        stack_b = inputs_embeds.new_empty((max_blocks + 1, row_b, hidden))
        if use_sp:
            assert plan_a is not None and plan_b is not None
            stack_a[0].copy_(inputs_embeds[plan_a.begin : plan_a.end])
            stack_b[0].copy_(inputs_embeds[t0 + plan_b.begin : t0 + plan_b.end])
        else:
            stack_a[0].copy_(inputs_embeds[:t0])
            stack_b[0].copy_(inputs_embeds[t0:])
        num_blocks = 0

        aux_hidden_states: list[torch.Tensor] = []
        if 0 in self.aux_hidden_state_layers:
            aux_hidden_states.append(inputs_embeds.clone())

        inputs_a: list[torch.Tensor] = [
            self.layers[0].attn_norm(inputs_embeds[:t0], j)
            for j in range(self.layers[0].num_attn_inputs)
        ]
        inputs_b: list[torch.Tensor] = [
            self.layers[0].attn_norm(inputs_embeds[t0:], j)
            for j in range(self.layers[0].num_attn_inputs)
        ]

        if use_sp and DBO_PIPE_SP:
            # Fully pipelined DBO+SP: entire `RS -> halo+cache_sync -> sharded sconv+AttnRes -> AG`
            # runs sequentially on `comm_stream`, overlapping 100% of BOTH RS and AG behind
            # `Attn` and `MoE` compute on `compute_stream` (zero AG wait bubble!).
            assert pf_sp is not None and plan_a is not None and plan_b is not None
            cs = dbo.comm_stream
            ev_init = torch.cuda.Event()
            ev_init.record(compute_stream)
            cs.wait_event(ev_init)
            stack_a.record_stream(cs)
            stack_b.record_stream(cs)
            plan_a.record_stream(cs)
            plan_b.record_stream(cs)
            with torch.cuda.stream(cs):
                attn_out_a = torch.empty_like(stack_a[0])
                attn_out_b = torch.empty_like(stack_b[0])
            pending_b_moe_interleave: tuple[Any, ...] | None = None

            pending_b_moe_handle = None
            pending_aux_a: torch.Tensor | None = None
            moe_ag_modes = (SP_FP8_AG_MOE_MODE,) if (SP_FP8_AG and SP_FP8_AG_MOE) else None
            kv_ag_mode = "dequant" if SP_FP8_AG_KV else "bf16"
            _layer_ev_ring: list[torch.cuda.Event] = []

            for i, layer in enumerate(self.layers):
                if DBO_MAX_INFLIGHT_LAYERS > 0 and len(_layer_ev_ring) >= DBO_MAX_INFLIGHT_LAYERS:
                    # Zero-bubble depth-K pacing: waits on layer i-K's comm_stream completion
                    # (K layers behind GPU frontier!) so CachingAllocator reclaims cross-stream
                    # record_stream limbo (<= K * 453 MB instead of 48 * 453 MB = 21.75 GB!).
                    _layer_ev_ring.pop(0).synchronize()
                first = i % self.block_size == 0
                partial_a = stack_a[num_blocks]
                partial_b = stack_b[num_blocks]

                # 1. Layer i Attn(A) on compute_stream (overlaps Layer_{i-1} MoE(B) RS/SP chain!)
                with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                    attn_partial_a = layer.attention(pos_a, inputs_a, tau_a)

                if DBO_INTERLEAVE and pending_b_moe_interleave is not None:
                    # Hybrid NIC-First Interleaved Half 1 (layers 1..47):
                    #   comm_stream:    | RS(MoE_{i-1},B) | RS(Attn_i,A) | AG(MoE_{i-1},B) | AG(Attn_i,A) |
                    #   compute_stream: | Attn_i(A)       | SC(MoE_B)    | SC(Attn_A)      | Attn_i(B)    |
                    # Keeps NICs 100% back-to-back with ZERO sconv+AttnRes stall on comm_stream!
                    (
                        h_prev_b_rs,
                        prev_b_sconv,
                        prev_b_conv_out,
                        prev_b_stack,
                        prev_b_query,
                        prev_b_norm,
                        prev_b_conv_addend,
                        prev_b_conv_res,
                        prev_b_need_aux,
                        prev_b_ag_modes,
                        prev_b_repl,
                    ) = pending_b_moe_interleave
                    pending_b_moe_interleave = None

                    # 2a. Enqueue RS+halo(Attn_i, A) immediately behind RS+halo(MoE_{i-1}, B)
                    with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                        h_attn_a_rs = dbo.launch_sp_rs_halo_only(
                            pf_sp, attn_partial_a, layer.attn_sconv, plan_a, compute_stream
                        )

                    # 2b. Run sconv+AttnRes(MoE_{i-1}, B) on compute_stream (overlaps RS(Attn_i, A)!)
                    with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                        x_prev_b_halo = h_prev_b_rs.wait(compute_stream)
                        out_shard_prev_b = run_sharded_sconv_attn_res(
                            sconv_mod=prev_b_sconv,
                            x_with_halo=x_prev_b_halo,
                            plan=plan_b,
                            conv_out=prev_b_conv_out,
                            stack=prev_b_stack,
                            mix_last=None,
                            queries=prev_b_query,
                            norm=prev_b_norm,
                            normalize_values=self.normalize_values,
                            conv_addend=prev_b_conv_addend,
                            conv_res=prev_b_conv_res,
                        )
                        h_prev_b_ag = dbo.launch_sp_gather_only(
                            pf_sp,
                            out_shard_prev_b,
                            compute_stream,
                            need_aux_shard=prev_b_need_aux,
                            fp8_ag_modes=prev_b_ag_modes,
                            next_global_repl=prev_b_repl,
                        )

                    # 2c. Run sconv+AttnRes(Attn_i, A) on compute_stream (overlaps AG(MoE_{i-1}, B)!)
                    with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                        x_attn_a_halo = h_attn_a_rs.wait(compute_stream)
                        out_shard_attn_a = run_sharded_sconv_attn_res(
                            sconv_mod=layer.attn_sconv,
                            x_with_halo=x_attn_a_halo,
                            plan=plan_a,
                            conv_out=attn_out_a,
                            stack=stack_a[: num_blocks + 1] if first else stack_a[:num_blocks],
                            mix_last=None if first else partial_a,
                            queries=self.attn_res.mlp_queries[i],
                            norm=layer.mlp_norm,
                            normalize_values=self.normalize_values,
                        )
                        h_attn_a = dbo.launch_sp_gather_only(
                            pf_sp,
                            out_shard_attn_a,
                            compute_stream,
                            fp8_ag_modes=moe_ag_modes,
                        )

                    # 3. Collect Layer_{i-1} MoE(B) AG outputs -> run Attn(B) (overlaps AG(Attn_i, A)!)
                    inputs_b, aux_b = h_prev_b_ag.wait(compute_stream)
                    if i in self.aux_hidden_state_layers:
                        assert pending_aux_a is not None and aux_b is not None
                        aux_hidden_states.append(
                            torch.cat([pending_aux_a, aux_b], dim=0)
                        )
                        pending_aux_a = None
                else:
                    # 2. Enqueue Attn(A) RS->sconv->AG on comm_stream (behind Layer_{i-1} MoE(B))
                    with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                        h_attn_a = dbo.launch_sp_sublayer(
                            pf_sp,
                            attn_partial_a,
                            layer.attn_sconv,
                            plan_a,
                            attn_out_a,
                            stack_a[: num_blocks + 1] if first else stack_a[:num_blocks],
                            None if first else partial_a,
                            self.attn_res.mlp_queries[i],
                            layer.mlp_norm,
                            self.normalize_values,
                            compute_stream,
                            fp8_ag_modes=moe_ag_modes,
                        )

                    # 3. Collect Layer_{i-1} MoE(B) outputs (already ready!)
                    if pending_b_moe_handle is not None:
                        inputs_b, aux_b = pending_b_moe_handle.wait(compute_stream)
                        if i in self.aux_hidden_state_layers:
                            assert pending_aux_a is not None and aux_b is not None
                            aux_hidden_states.append(
                                torch.cat([pending_aux_a, aux_b], dim=0)
                            )
                            pending_aux_a = None

                # 4. Layer i Attn(B) on compute_stream (overlaps Attn(A) SP chain / AG!)
                with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                    attn_partial_b = layer.attention(pos_b, inputs_b, tau_b)

                # 5. Enqueue Attn(B) RS->sconv->AG on comm_stream
                with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                    h_attn_b = dbo.launch_sp_sublayer(
                        pf_sp,
                        attn_partial_b,
                        layer.attn_sconv,
                        plan_b,
                        attn_out_b,
                        stack_b[: num_blocks + 1] if first else stack_b[:num_blocks],
                        None if first else partial_b,
                        self.attn_res.mlp_queries[i],
                        layer.mlp_norm,
                        self.normalize_values,
                        compute_stream,
                        fp8_ag_modes=moe_ag_modes,
                    )

                # 6. Wait Attn(A) SP chain -> run MoE(A) (overlaps Attn(B) SP chain!)
                mlp_in_a_list, _ = h_attn_a.wait(compute_stream)
                with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                    moe_partial_a = self._moe_partial(layer, mlp_in_a_list[0])

                res_a = None if first else partial_a
                res_b = None if first else partial_b
                if first:
                    num_blocks += 1

                if i + 1 < num_layers:
                    nxt_query = self.attn_res.attn_queries[i]
                    nxt_norm = self.layers[i + 1].attn_norm
                    nxt_repl = self._get_global_kv_repl(i + 1)
                    if SP_FP8_AG:
                        nxt_ag_modes: tuple[str, ...] | None = (
                            ("direct",)
                            if self.layers[i + 1].is_local
                            else ("direct", kv_ag_mode, kv_ag_mode)
                        )
                    else:
                        nxt_ag_modes = None
                else:
                    nxt_query, nxt_norm = self.attn_res.output_query, self.norm
                    nxt_repl = None
                    nxt_ag_modes = None

                need_aux = (i + 1) < num_layers and (
                    (i + 1) in self.aux_hidden_state_layers
                )

                # 7. Enqueue MoE(A) RS->sconv->AG (or repl_kv A2A) on comm_stream
                with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                    h_moe_a = dbo.launch_sp_sublayer(
                        pf_sp,
                        moe_partial_a,
                        layer.mlp_sconv,
                        plan_a,
                        stack_a[num_blocks],
                        stack_a[:num_blocks],
                        None,
                        nxt_query,
                        nxt_norm,
                        self.normalize_values,
                        compute_stream,
                        conv_addend=attn_out_a,
                        conv_res=res_a,
                        need_aux_shard=stack_a[num_blocks] if need_aux else None,
                        fp8_ag_modes=nxt_ag_modes,
                        next_global_repl=nxt_repl,
                    )

                # 8. Wait Attn(B) SP chain -> run MoE(B) (overlaps MoE(A) SP chain!)
                mlp_in_b_list, _ = h_attn_b.wait(compute_stream)
                with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                    moe_partial_b = self._moe_partial(layer, mlp_in_b_list[0])

                # 9. Enqueue MoE(B) RS->halo (interleaved) or full SP chain (default / final layer)
                if DBO_INTERLEAVE and (i + 1 < num_layers):
                    with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                        h_b_rs = dbo.launch_sp_rs_halo_only(
                            pf_sp, moe_partial_b, layer.mlp_sconv, plan_b, compute_stream
                        )
                    pending_b_moe_interleave = (
                        h_b_rs,
                        layer.mlp_sconv,
                        stack_b[num_blocks],
                        stack_b[:num_blocks],
                        nxt_query,
                        nxt_norm,
                        attn_out_b,
                        res_b,
                        stack_b[num_blocks] if need_aux else None,
                        nxt_ag_modes,
                        nxt_repl,
                    )
                    pending_b_moe_handle = None
                else:
                    with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                        pending_b_moe_handle = dbo.launch_sp_sublayer(
                            pf_sp,
                            moe_partial_b,
                            layer.mlp_sconv,
                            plan_b,
                            stack_b[num_blocks],
                            stack_b[:num_blocks],
                            None,
                            nxt_query,
                            nxt_norm,
                            self.normalize_values,
                            compute_stream,
                            conv_addend=attn_out_b,
                            conv_res=res_b,
                            need_aux_shard=stack_b[num_blocks] if need_aux else None,
                            fp8_ag_modes=nxt_ag_modes,
                            next_global_repl=nxt_repl,
                        )

                # 10. Wait MoE(A) SP chain -> ready for Layer_{i+1} Attn(A)
                inputs_a, pending_aux_a = h_moe_a.wait(compute_stream)
                if DBO_MAX_INFLIGHT_LAYERS > 0:
                    _ev_layer = torch.cuda.Event()
                    _ev_layer.record(cs)
                    _layer_ev_ring.append(_ev_layer)

            assert pending_b_moe_handle is not None
            # Build deferred sync plan while comm_stream finishes layer 47 MoE(B) (overlapped!)
            if def_plan is None and SP_ZERO_SYNC:
                def_plan = DeferredConvSyncPlan.from_plans([plan_a, plan_b])
            out_b_list, _ = pending_b_moe_handle.wait(compute_stream)
            sync_deferred_conv_caches(self.layers, def_plan, comm_stream=dbo.comm_stream)
            out = torch.cat([inputs_a[0], out_b_list[0]], dim=0)
            if self.aux_hidden_state_layers:
                return out, [*aux_hidden_states, out]
            return out

        # Pending B-side MoE completion carried cleanly across the layer i-1 -> layer i boundary
        # so AR/RS(MoE_{i-1}, B) overlaps with Layer_i.Attn(A)!
        pending_b_moe: tuple[Any, ...] | None = None
        pending_aux_a: torch.Tensor | None = None

        def _finish_sublayer_compute(
            sconv_mod: Inkling2ShortConv,
            handle: Any,
            pos_mb: torch.Tensor,
            plan_mb: Any,
            conv_out: torch.Tensor,
            stk: torch.Tensor,
            mix_last: torch.Tensor | None,
            queries: torch.Tensor,
            norm: Inkling2RMSNorm,
            conv_addend: torch.Tensor | None = None,
            conv_res: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Wait `handle` on `compute_stream`, scatter remote SP cache if needed,
            and execute sconv + AttnRes (returns `[Q, row_mb, H]`)."""
            x_data = handle.wait(compute_stream)
            if use_sp:
                assert plan_mb is not None
                if handle.gathered_sync is not None and (
                    plan_mb.remote_gather_idx is not None
                    or plan_mb.rank0_last3_slots is not None
                ):
                    off, w = sconv_mod.owner.stream_ranges[sconv_mod.stream]
                    scatter_remote_conv_cache(
                        handle.gathered_sync, plan_mb, sconv_mod.owner.kv_cache, off, w
                    )
                return run_sharded_sconv_attn_res(
                    sconv_mod=sconv_mod,
                    x_with_halo=x_data,
                    plan=plan_mb,
                    conv_out=conv_out,
                    stack=stk,
                    mix_last=mix_last,
                    queries=queries,
                    norm=norm,
                    normalize_values=self.normalize_values,
                    conv_addend=conv_addend,
                    conv_res=conv_res,
                )
            return self._run_unsharded_sconv_ar(
                sconv_mod=sconv_mod,
                x_reduced=x_data,
                positions=pos_mb,
                conv_out=conv_out,
                stack=stk,
                mix_last=mix_last,
                queries=queries,
                norm=norm,
                conv_addend=conv_addend,
                conv_res=conv_res,
            )

        def _launch_reduce(partial: torch.Tensor, plan_mb: Any) -> Any:
            if use_sp:
                assert pf_sp is not None and plan_mb is not None
                return dbo.launch_sp_rs_halo(pf_sp, partial, plan_mb, compute_stream)
            return dbo.launch_all_reduce(partial, compute_stream)

        _layer_ev_ring_np: list[torch.cuda.Event] = []
        for i, layer in enumerate(self.layers):
            if DBO_MAX_INFLIGHT_LAYERS > 0 and len(_layer_ev_ring_np) >= DBO_MAX_INFLIGHT_LAYERS:
                _layer_ev_ring_np.pop(0).synchronize()
            # 1. Layer i Attention(A) on compute_stream (OVERLAPS with pending_b_moe on comm_stream!)
            with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                attn_partial_a = layer.attention(pos_a, inputs_a, tau_a)

            # 2. Finish Layer i-1's B-side MoE sconv+AttnRes (comm already finished during Attn_i(A)!)
            #    Notice: if use_sp, we enqueue AG(inputs_b) BEFORE h_attn_a on comm_stream so
            #    comm_stream executes AG(inputs_b) -> h_attn_a without blocking Attn_i(B)!
            if pending_b_moe is not None:
                (
                    prev_layer,
                    h_prev_moe_b,
                    prev_attn_out_b,
                    prev_res_b,
                    prev_nb,
                    prev_q,
                    prev_norm,
                    prev_repl,
                ) = pending_b_moe
                with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                    out_b = _finish_sublayer_compute(
                        prev_layer.mlp_sconv,
                        h_prev_moe_b,
                        pos_b,
                        plan_b,
                        stack_b[prev_nb],
                        stack_b[:prev_nb],
                        None,
                        prev_q,
                        prev_norm,
                        conv_addend=prev_attn_out_b,
                        conv_res=prev_res_b,
                    )
                if use_sp:
                    ag_handles_b = dbo.launch_sp_all_gather(
                        out_b, compute_stream, next_global_repl=prev_repl
                    )
                else:
                    ag_handles_b = None
                    inputs_b = [out_b[q] for q in range(out_b.shape[0])]
                if i in self.aux_hidden_state_layers:
                    assert pending_aux_a is not None
                    if use_sp:
                        aux_b = dbo.all_gather_tensor_on_comm(stack_b[prev_nb], compute_stream)
                    else:
                        aux_b = stack_b[prev_nb].clone()
                    aux_hidden_states.append(torch.cat([pending_aux_a, aux_b], dim=0))
                    pending_aux_a = None
            else:
                ag_handles_b = None

            # 3. Launch AR/RS(Attn_i, A) on comm_stream (behind AG(inputs_b) if any)
            h_attn_a = _launch_reduce(attn_partial_a, plan_a)
            if ag_handles_b is not None:
                inputs_b = [h.wait(compute_stream) for h in ag_handles_b]

            # 4. Layer i Attention(B) on compute_stream (OVERLAPS with h_attn_a on comm_stream!)
            with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                attn_partial_b = layer.attention(pos_b, inputs_b, tau_b)

            first = i % self.block_size == 0
            partial_a = stack_a[num_blocks]
            partial_b = stack_b[num_blocks]
            attn_out_a = torch.empty_like(partial_a)
            attn_out_b = torch.empty_like(partial_b)

            # 5-6. Wait h_attn_a -> attn_sconv+AttnRes(A) -> queue AG(mlp_in_a) BEFORE h_attn_b
            #      so waiting on AG(mlp_in_a) NEVER stalls on h_attn_b, letting MoE_i(A) overlap
            #      100% with h_attn_b on comm_stream!
            if not use_sp:
                h_attn_b = _launch_reduce(attn_partial_b, plan_b)
            with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                mlp_in_a_raw = _finish_sublayer_compute(
                    layer.attn_sconv,
                    h_attn_a,
                    pos_a,
                    plan_a,
                    attn_out_a,
                    stack_a[: num_blocks + 1] if first else stack_a[:num_blocks],
                    None if first else partial_a,
                    self.attn_res.mlp_queries[i],
                    layer.mlp_norm,
                )
                if use_sp:
                    ag_a_h = dbo.launch_sp_all_gather(mlp_in_a_raw, compute_stream)[0]
                    h_attn_b = _launch_reduce(attn_partial_b, plan_b)
                    mlp_in_a = ag_a_h.wait(compute_stream)
                else:
                    mlp_in_a = mlp_in_a_raw[0]
                moe_partial_a = self._moe_partial(layer, mlp_in_a)

            # 7-8. Wait h_attn_b -> attn_sconv+AttnRes(B) -> queue AG(mlp_in_b) BEFORE h_moe_a
            #      so MoE_i(B) overlaps 100% with h_moe_a on comm_stream! (Note attn_sconv(A)
            #      already populated A[-3:] in conv_state.kv_cache on compute_stream.)
            if not use_sp:
                h_moe_a = _launch_reduce(moe_partial_a, plan_a)
            with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
                mlp_in_b_raw = _finish_sublayer_compute(
                    layer.attn_sconv,
                    h_attn_b,
                    pos_b,
                    plan_b,
                    attn_out_b,
                    stack_b[: num_blocks + 1] if first else stack_b[:num_blocks],
                    None if first else partial_b,
                    self.attn_res.mlp_queries[i],
                    layer.mlp_norm,
                )
                if use_sp:
                    ag_b_h = dbo.launch_sp_all_gather(mlp_in_b_raw, compute_stream)[0]
                    h_moe_a = _launch_reduce(moe_partial_a, plan_a)
                    mlp_in_b = ag_b_h.wait(compute_stream)
                else:
                    mlp_in_b = mlp_in_b_raw[0]
                moe_partial_b = self._moe_partial(layer, mlp_in_b)

            res_a = None if first else partial_a
            res_b = None if first else partial_b
            if first:
                num_blocks += 1

            if i + 1 < num_layers:
                nxt_query = self.attn_res.attn_queries[i]
                nxt_norm = self.layers[i + 1].attn_norm
                nxt_repl = self._get_global_kv_repl(i + 1) if use_sp else None
            else:
                nxt_query, nxt_norm = self.attn_res.output_query, self.norm
                nxt_repl = None

            # 9-10. Wait h_moe_a -> mlp_sconv+AttnRes(A) -> queue AG(inputs_a) BEFORE h_moe_b
            #       so Layer_{i+1}.Attn(A) overlaps 100% with h_moe_b on comm_stream!
            if not use_sp:
                h_moe_b = _launch_reduce(moe_partial_b, plan_b)
            with patch_forward_attn_metadata(meta_a, slice_a, num_tokens):
                out_a = _finish_sublayer_compute(
                    layer.mlp_sconv,
                    h_moe_a,
                    pos_a,
                    plan_a,
                    stack_a[num_blocks],
                    stack_a[:num_blocks],
                    None,
                    nxt_query,
                    nxt_norm,
                    conv_addend=attn_out_a,
                    conv_res=res_a,
                )
            if use_sp:
                ag_handles_a = dbo.launch_sp_all_gather(
                    out_a, compute_stream, next_global_repl=nxt_repl
                )
                if (i + 1) in self.aux_hidden_state_layers:
                    aux_a_h = dbo.launch_sp_all_gather(
                        stack_a[num_blocks].unsqueeze(0), compute_stream
                    )[0]
                else:
                    aux_a_h = None
                h_moe_b = _launch_reduce(moe_partial_b, plan_b)
                inputs_a = [h.wait(compute_stream) for h in ag_handles_a]
                if aux_a_h is not None:
                    pending_aux_a = aux_a_h.wait(compute_stream)
            else:
                inputs_a = [out_a[q] for q in range(out_a.shape[0])]
                if (i + 1) in self.aux_hidden_state_layers:
                    pending_aux_a = stack_a[num_blocks].clone()

            pending_b_moe = (
                layer,
                h_moe_b,
                attn_out_b,
                res_b,
                num_blocks,
                nxt_query,
                nxt_norm,
                nxt_repl,
            )
            if DBO_MAX_INFLIGHT_LAYERS > 0:
                _ev_layer_np = torch.cuda.Event()
                _ev_layer_np.record(dbo.comm_stream)
                _layer_ev_ring_np.append(_ev_layer_np)

        # Drain final Layer 47 B-side MoE completion
        assert pending_b_moe is not None
        (
            last_layer,
            h_last_moe_b,
            last_attn_out_b,
            last_res_b,
            last_nb,
            last_q,
            last_norm,
            _last_repl,
        ) = pending_b_moe
        with patch_forward_attn_metadata(meta_b, slice_b, num_tokens):
            out_b_final = _finish_sublayer_compute(
                last_layer.mlp_sconv,
                h_last_moe_b,
                pos_b,
                plan_b,
                stack_b[last_nb],
                stack_b[:last_nb],
                None,
                last_q,
                last_norm,
                conv_addend=last_attn_out_b,
                conv_res=last_res_b,
            )
        if use_sp:
            out_b_t = dbo.all_gather_tensor_on_comm(out_b_final[0], compute_stream)
            if def_plan is None and SP_ZERO_SYNC and plan_a is not None and plan_b is not None:
                def_plan = DeferredConvSyncPlan.from_plans([plan_a, plan_b])
            sync_deferred_conv_caches(self.layers, def_plan, comm_stream=dbo.comm_stream)
        else:
            out_b_t = out_b_final[0]
        out = torch.cat([inputs_a[0], out_b_t], dim=0)
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


register_lora()


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsLoRA,
):
    # LoRA (``lora.py``): adapters use the checkpoint's module names.
    packed_modules_mapping = PACKED_MODULES_MAPPING
    embedding_modules: dict[str, str] = {}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr=LORA_NAME_SUBSTRINGS,
        orig_to_new_prefix=LORA_NAME_PREFIXES,
    )

    # Weights mirror the checkpoint names after these substitutions, except for
    # the fused per-layer weights in ``_shards``.
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert quant_config is None or isinstance(quant_config, Inkling2QuantConfig)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        if vllm_config.model_config.enable_prompt_embeds:
            # The reference applies embed_norm to raw input embeddings, which
            # cannot be told apart from already-normalized ones here.
            raise ValueError("Inkling2 does not support prompt embeddings")
        self.config = config
        self.text_config = text_config = config.text_config
        self.visual = (
            Inkling2VisionEncoder(config.vision_config, text_config.hidden_size)
            if config.vision_config is not None
            and vllm_config.model_config.multimodal_config is not None
            and vllm_config.model_config.multimodal_config.get_limit_per_prompt("image")
            else None
        )
        self.model = Inkling2Model(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size, text_config.hidden_size
        )
        l2pf.set_head_ref(self.lm_head)
        self.logits_processor = LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )

    def get_language_model(self) -> nn.Module:
        return self

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches = kwargs.get("vision_pixels")
        grid_thw = kwargs.get("vision_grid_thw")
        if patches is None or grid_thw is None:
            return ()
        assert isinstance(patches, torch.Tensor)
        assert isinstance(grid_thw, torch.Tensor)
        assert self.visual is not None
        embeds = self.visual(patches, grid_thw)
        return embeds.split(grid_thw.prod(-1).tolist())

    # Encoder CUDA graphs (``compilation_config.cudagraph_mm_encoder``): one
    # graph per output-token budget; padded rows are dead (zero-length
    # sequences, never pooled).
    _graph_metadata_keys = ("meta", "patch_rows", "pixel_offsets", "pool_rows")

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        from vllm.model_executor.models.qwen2_5_vl import (
            _pad_cumulative_seqlens_buffer,
        )

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "vision_pixels",
                *self._graph_metadata_keys,
                "cu_seqlens",
            ],
            padding_logics={"cu_seqlens": _pad_cumulative_seqlens_buffer},
            out_hidden_size=self.text_config.hidden_size,
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        # 64 tokens: a 320x320 image.
        return 64, min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        return [
            EncoderItemSpec(input_size=h * w, output_tokens=h * w)
            for _, h, w in mm_kwargs["vision_grid_thw"].tolist()
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thw = mm_kwargs["vision_grid_thw"]
        pixels = mm_kwargs["vision_pixels"]
        patches = pixels.split(grid_thw.prod(-1).tolist())
        return {
            # Empty under data-parallel sharding when a rank gets no images.
            "vision_pixels": torch.cat([pixels[:0], *(patches[i] for i in indices)]),
            "vision_grid_thw": grid_thw[indices],
        }

    def _graph_values(
        self, patches: torch.Tensor, grid: list[list[int]]
    ) -> dict[str, torch.Tensor]:
        assert self.visual is not None
        metadata = self.visual.metadata(grid, patches.device)
        return {
            "vision_pixels": patches,
            **{key: getattr(metadata, key) for key in self._graph_metadata_keys},
            "cu_seqlens": metadata.cu_seqlens,
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        assert self.visual is not None
        side = self.visual.patch_size * self.visual.merge_size
        per_item = -(-token_budget // max_batch_size)
        pixels = torch.zeros(
            (max_batch_size * per_item, side, side, 3),
            dtype=torch.bfloat16,
            device=device,
        )
        grid = [[1, 1, per_item]] * max_batch_size
        return EncoderCudaGraphCaptureInputs(values=self._graph_values(pixels, grid))

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        return EncoderCudaGraphReplayBuffers(
            values=self._graph_values(
                mm_kwargs["vision_pixels"],
                mm_kwargs["vision_grid_thw"].tolist(),
            )
        )

    def encoder_cudagraph_forward(
        self, values: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        assert self.visual is not None
        meta = values["meta"]
        metadata = VisionMetadata(
            **{key: values[key] for key in self._graph_metadata_keys},
            cu_seqlens=values["cu_seqlens"],
            # Bounds every sequence a replay can pack into the buffers.
            max_seqlen=meta.shape[0],
        )
        return self.visual.encode(values["vision_pixels"], metadata)

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        assert self.visual is not None
        return self.visual(mm_kwargs["vision_pixels"], mm_kwargs["vision_grid_thw"])

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        direct_out = getattr(_DIRECT_EMBED_OUT_TLS, "out", None)
        if direct_out is not None:
            _DIRECT_EMBED_OUT_TLS.out = None
        if not multimodal_embeddings:
            return self.model.embed_input_ids(input_ids, out=direct_out)
        inputs_embeds = self.model.embed_input_ids(input_ids)
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if _ROW_AR:  # w3 rowar review: eager once per step -> periodic cross-rank stats line (INKLING_ROW_AR_STATS=1)
            _rowar_mod().rowar_stats_tick()
        if _SYNC_FREE:
            from .ops import sync_free

            out = sync_free.target_logits(self, hidden_states)
            if out is not None:
                return out
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        # Truncated models (e.g. fewer layers via hf_overrides) skip the rest.
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        visual_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((name.removeprefix("model.visual."), weight))
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(
                    r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name
                )
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if param.dtype != weight.dtype:
                raise ValueError(
                    f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}"
                )
            loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        if self.visual is not None:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_weights)}
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        if os.environ.get("INKLING_REPL_EMBED", "0") == "1" and get_tensor_model_parallel_world_size() > 1:
            self._replicate_embeddings()
        if SP_ENABLED and SP_Q3_MODE == "repl_kv" and get_tensor_model_parallel_world_size() > 1:
            self.model._ensure_global_kv_repl()
        try:
            from .ops import gb10_load as _gb10_load
            _gb10_load.release_staging()
        except Exception:
            pass
        torch.cuda.empty_cache()
        return loaded

    @torch.no_grad()
    def _replicate_embeddings(self) -> None:
        """INKLING_REPL_EMBED=1: all-gather embed_tokens once at load time so both target and
        drafter embedding lookups run locally with zero RDMA all-reduce (~50 us/step saved, bit-exact)."""
        emb = self.model.embed_tokens
        si = getattr(emb, "shard_indices", None)
        tp = get_tensor_model_parallel_world_size()
        if si is None or int(si.num_added_elements) != 0 or int(si.num_org_vocab_padding) != 0:
            print("[inkling-repl-embed] unsupported VocabParallelEmbedding layout; keeping sharded AR")
            return
        w = emb.weight.data.contiguous()
        full_w = tensor_model_parallel_all_gather(w, dim=0).contiguous()
        # Normalize -0.0 -> +0.0 to match fp32 AR sum (+0.0 + w) bitwise.
        full_w.copy_(torch.where(full_w == 0, torch.zeros_like(full_w), full_w))
        full_v = int(full_w.shape[0])
        emb.register_buffer("_repl_weight", full_w, persistent=False)

        def _repl_forward(input_: torch.Tensor) -> torch.Tensor:
            ids = input_.long()
            oob = (ids < 0) | (ids >= full_v)
            out = torch.nn.functional.embedding(ids.clamp(0, full_v - 1), emb._repl_weight)
            return out.masked_fill_(oob.unsqueeze(-1), 0)

        emb.forward = _repl_forward
        freed_mb = 0.0
        if os.environ.get("INKLING_REPL_EMBED_FREE", "1") != "0":
            # Original 1/8 shard is dead once _repl_weight + _repl_forward are installed (+220.85 MiB/rank).
            freed_mb = emb.weight.numel() * emb.weight.element_size() / 2**20
            emb.weight.data.untyped_storage().resize_(0)
        print(f"[inkling-repl-embed] replicated embed_tokens {tuple(full_w.shape)} {full_w.dtype} "
              f"({full_w.numel() * full_w.element_size() / 2**20:.0f} MiB) across TP={tp}; "
              f"zero embed AR (freed dead shard {freed_mb:.1f} MiB)")


__all__ = ["Inkling2ForConditionalGeneration"]
