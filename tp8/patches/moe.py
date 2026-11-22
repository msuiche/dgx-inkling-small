# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts layer (routed experts only, NVFP4 or MXFP8).

Experts are sharded contiguously over the TP group (EP size = TP size); each
rank returns the full all-reduced bf16 output. Checkpoint names under
``model.llm.layers.N.mlp`` map 1:1 to parameters.
"""

from typing import Literal, Protocol

import os

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from .ops.activation import swiglu_quantize
from .ops.block_scales import shuffle_rows, swizzle_block_scales
from .ops.gb10_load import copy_from_checkpoint_
from .ops.moe import NUM_COUNTERS, combine, combine_push, route
from .ops.seq_parallel import SeqParallel
from .ops.trtllm_bmm import bmm, ensure_packed_for_pf_native, expert_plan, load_kernels
from .ops import l2pf

QuantFormat = Literal["nvfp4", "mxfp8"]


class ExpertsLoRA(Protocol):
    """Adds the routed experts' LoRA deltas (``lora.Inkling2MoEWithLoRA``),
    for an experts implementation over expert-sorted rows: row ``r`` is token
    ``perm[r] // topk``, and this rank's experts own rows
    ``[offsets[0], offsets[-1])``. The NVFP4 / MXFP8 experts here (tile-padded
    rows, quantized activations) do not support it; the BF16 experts a model
    may plug in instead (SkyRL's ``Inkling2BF16Experts``) do.
    """

    def add_w13(
        self,
        gate_up: torch.Tensor,
        x: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...

    def add_w2(
        self,
        down: torch.Tensor,
        activated: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...


class Inkling2Gate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})
        # Completion counters of the routing kernels (reset by the kernels).
        self.register_buffer(
            "counters", torch.zeros(NUM_COUNTERS, dtype=torch.int32), persistent=False
        )


class Inkling2Experts(nn.Module):
    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        # INKLING_MOE_TP=1: every rank holds all experts with 1/TP of the
        # intermediate dim (checkpoint pre-sliced by tp_slice_send.py), so the
        # per-rank work is balanced; the reduce is the same partial-sum AR.
        self.moe_tp = os.environ.get("INKLING_MOE_TP", "0") == "1"
        self.inter_div = self.ep_size if self.moe_tp else 1
        if self.moe_tp:
            self.ep_rank, self.num_experts = 0, config.n_routed_experts
        else:
            assert config.n_routed_experts % self.ep_size == 0
            self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size // self.inter_div
        self.hidden_size, self.intermediate_size = hidden, inter

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.shapes = ((2 * inter, hidden), (hidden, inter))
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            # Rows are shuffled and block scales swizzled (``block_scales``).
            self._register(
                f"{name}_weight",
                (n, k // pack),
                weight_dtype,
                loader=self._load_weight,
            )
            self._register(
                f"{name}_weight_scale",
                (n, k // group),
                scale_dtype,
                loader=self._load_block_scales,
            )
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

        start = self.ep_rank * self.num_experts
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        # Rows per local expert of the routing kernels (reset by the kernels).
        self.register_buffer(
            "counts", torch.zeros(self.num_experts, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "ones", torch.ones(self.num_experts, dtype=torch.float32), persistent=False
        )
        load_kernels(
            (self.w13_weight, self.w2_weight),
            (self.w13_weight_scale, self.w2_weight_scale),
            self.topk,
        )

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
        loader=None,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        if loader is None:
            loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    # sm121: raw layout (no row shuffle / scale swizzle), the torch bmm reads
    # it directly.
    def _load_weight(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)

    def _load_block_scales(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)

    @staticmethod
    def _load_full(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        # GB10: ``loaded`` is a view into the shard's mmap; a direct copy_
        # makes the driver take its page faults (0.5-2 GiB per call here).
        copy_from_checkpoint_(param.data, loaded)

    def plan(self, tokens: int):
        return expert_plan(tokens, self.topk, self.num_experts, self.nvfp4, self.shapes)

    def scales(self, name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Outer scale amax and denominator of a projection's NVFP4 weights."""
        if not self.nvfp4:
            return None, None
        return (
            getattr(self, f"{name}_weight_amax"),
            getattr(self, f"{name}_weight_outer_scale_denominator"),
        )

    def _gemm(self, quantized, name: str, tiles, plan, projection: int):
        act, act_scale, row_scale = quantized
        r2t = getattr(act, "_inkling_row2tok", None) if projection == 0 else None
        as_rm = bool(getattr(act_scale, "_inkling_rowmaj", False))
        if r2t is not None or as_rm:
            return bmm(
                act,
                act_scale,
                row_scale,
                getattr(self, f"{name}_weight"),
                getattr(self, f"{name}_weight_scale"),
                self.ones,
                tiles,
                plan,
                projection,
                row2tok=r2t,
                as_row_major=as_rm,
            )
        return bmm(
            act,
            act_scale,
            row_scale,
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_weight_scale"),
            self.ones,
            tiles,
            plan,
            projection,
        )

    def forward(
        self,
        x: torch.Tensor,
        routing,
        quantized,
        plan,
        reduce: bool = True,
        push: SeqParallel | None = None,
        combine_only: bool = False,
        prezeroed_out: torch.Tensor | None = None,
        zero_done_event: torch.cuda.Event | None = None,
        use_sendbuf: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """``routing`` and ``quantized`` (``x`` in the expert layout) are from
        ``ops.moe.route``. The routing weight is applied inside the SwiGLU,
        before the down projection. Without ``reduce``, returns the local
        expert rows and ``src2dst`` for the consumer to combine and reduce
        (or combined unreduced ``[T, H]`` when ``combine_only=True``),
        or pushes their combine to the token shard owners."""
        out_dtype = torch.bfloat16 if x.dtype == torch.float8_e4m3fn else x.dtype
        src2dst, gammas, rows, tiles = routing
        # Dec160 S8 fused decode MoE fast path (Zero-Scatter + Fused W13 SiTU + W2 OTF Quant + Fused Combine)
        if (reduce or combine_only) and push is None and plan.tile <= 16:
            try:
                from .ops import moe_s8_runtime as _s8
                _s8_active = (
                    _s8.S8_ZERO_SCATTER
                    or _s8.S8_FUSE_SITU
                    or _s8.S8_FUSE_COMBINE
                    or getattr(tiles, "_inkling_tile_mask", None) is not None
                )
            except Exception:
                _s8 = None
                _s8_active = False
            if _s8 is not None and _s8_active:
                _sbuf = None
                _nlive = None
                if (reduce or (combine_only and use_sendbuf)) and self.ep_size > 1:
                    from vllm.distributed.device_communicators import rdma_all_reduce as _rar
                    _sbuf = _rar.get_send_buffer((x.shape[0], self.hidden_size), out_dtype)
                    if _rar.ROWAR.get("scope") == x.shape[0]:
                        _nlive = _rar.ROWAR.get("buf")
                _out = _s8.forward_experts_s8(
                    self, x, routing, quantized, plan, out_buf=_sbuf, nlive_buf=_nlive,
                )
                if _out is not None:
                    if combine_only or self.ep_size == 1:
                        return _out
                    _nt = x.shape[0]
                    if _nt <= 16:
                        nxt = getattr(self, "l2pf_next", None)
                        if nxt is not None:
                            l2pf.fork(nxt, kind="moe")
                        elif getattr(self, "l2pf_last", False):
                            l2pf.fork(l2pf.get_head_weight(_nt), kind="head")
                    elif l2pf.C32 and _nt <= l2pf.C32_MAX_M:
                        nxt = l2pf.resolve_dense_next(self, _nt)
                        if nxt is not None:
                            l2pf.fork(nxt, kind="moe", num_tokens=_nt)
                        elif getattr(self, "l2pf_last", False):
                            l2pf.fork(l2pf.get_head_weight(_nt), kind="head", num_tokens=_nt)
                    _out = tensor_model_parallel_all_reduce(_out)
                    l2pf.join()
                    return _out
        # C=32 (B=32, M=17..256) opt-in 0-ULP fused MoE fast path (Zero-Scatter + Fused SiTU + 6-Wave L2 Combine)
        if (
            (reduce or combine_only)
            and push is None
            and plan.tile in (16, 32)
            and 17 <= x.shape[0] <= 256
        ):
            try:
                from .ops import moe_c32_runtime as _c32
            except Exception:
                _c32 = None
            if _c32 is not None and _c32.C32_MASTER:
                _sbuf = None
                _nlive = None
                if (reduce or (combine_only and use_sendbuf)) and self.ep_size > 1:
                    from vllm.distributed.device_communicators import rdma_all_reduce as _rar
                    _sbuf = _rar.get_send_buffer((x.shape[0], self.hidden_size), out_dtype)
                    if _rar.ROWAR.get("scope") == x.shape[0]:
                        _nlive = _rar.ROWAR.get("buf")
                _out = _c32.forward_experts_c32(
                    self, x, routing, quantized, plan, out_buf=_sbuf, nlive_buf=_nlive,
                )
                if _out is not None:
                    if combine_only or self.ep_size == 1:
                        return _out
                    _nt = x.shape[0]
                    if l2pf.C32 and _nt <= l2pf.C32_MAX_M:
                        nxt = l2pf.resolve_dense_next(self, _nt)
                        if nxt is not None:
                            l2pf.fork(nxt, kind="moe", num_tokens=_nt)
                        elif getattr(self, "l2pf_last", False):
                            l2pf.fork(l2pf.get_head_weight(_nt), kind="head", num_tokens=_nt)
                    elif _nt <= 64 and getattr(self, "l2pf_last", False):
                        l2pf.fork(l2pf.get_head_weight(_nt), kind="head")
                    _out = tensor_model_parallel_all_reduce(_out)
                    l2pf.join()
                    return _out
        gate_up = self._gemm(quantized, "w13", tiles, plan, 0)
        m = plan.max_ctas
        activated = swiglu_quantize(
            gate_up,
            gammas,
            self.gate_cap,
            self.up_cap,
            (rows, tiles[2 * m + 2 :], tiles[:m]),
            plan.tile,
            *self.scales("w2"),
        )
        del gate_up
        if (reduce or combine_only) and push is None and plan.tile >= 64:
            from .ops import pf_moe_native
            if pf_moe_native.is_active(plan, tokens=x.shape[0]) and pf_moe_native.COMBINE_MODE != 0:
                ensure_packed_for_pf_native(self.w2_weight, self.w2_weight_scale, 1)
                sbuf = None
                if reduce and self.ep_size > 1:
                    from vllm.distributed.device_communicators.rdma_all_reduce import get_send_buffer
                    sbuf = get_send_buffer((x.shape[0], self.hidden_size), out_dtype)
                use_prezero = (
                    sbuf is None
                    and prezeroed_out is not None
                    and pf_moe_native.COMBINE_MODE == 2
                )
                if use_prezero:
                    if zero_done_event is not None:
                        torch.cuda.current_stream(x.device).wait_event(zero_done_event)
                    out_target = prezeroed_out
                else:
                    out_target = sbuf
                r2t_w2 = getattr(quantized[0], "_inkling_row2tok", None)
                act_v, act_s, row_s = activated
                out = pf_moe_native.w2_combine_pf_native(
                    act_v, act_s, row_s, self.w2_weight, self.w2_weight_scale,
                    self.ones, tiles, plan, src2dst, self.topk, out_dtype,
                    out=out_target, out_prezeroed=use_prezero, row2tok=r2t_w2,
                    as_row_major=bool(getattr(act_s, "_inkling_rowmaj", False)),
                )
                if out is not None:
                    if combine_only or self.ep_size == 1:
                        return out
                    return tensor_model_parallel_all_reduce(out)
        down = self._gemm(activated, "w2", tiles, plan, 1)
        if push is not None:
            # Each rank's fp32 combine is rounded once to bf16.
            combine_push(down, src2dst, self.topk, push)
            return None
        if not reduce:
            if combine_only:
                sbuf = None
                if use_sendbuf and self.ep_size > 1:
                    from vllm.distributed.device_communicators.rdma_all_reduce import get_send_buffer
                    sbuf = get_send_buffer((src2dst.numel() // self.topk, down.shape[1]), out_dtype)
                return combine(down, src2dst, self.topk, out_dtype, out=sbuf)
            return down, src2dst
        # Each rank's fp32 combine is rounded once to bf16 before the all-reduce.
        sbuf = None
        if self.ep_size > 1:
            from vllm.distributed.device_communicators.rdma_all_reduce import get_send_buffer
            sbuf = get_send_buffer((src2dst.numel() // self.topk, down.shape[1]), out_dtype)
        out = combine(down, src2dst, self.topk, out_dtype, out=sbuf)
        if self.ep_size == 1:
            return out
        # GB10: stream the next layer's BF11/BF12 qkvr weight (or cert_head on last layer) into L2 while the AR waits.
        _nt = x.shape[0]
        if _nt <= 16:
            nxt = getattr(self, "l2pf_next", None)
            if nxt is not None:
                l2pf.fork(nxt, kind="moe")
            elif getattr(self, "l2pf_last", False):
                l2pf.fork(l2pf.get_head_weight(_nt), kind="head")
        elif l2pf.C32 and _nt <= l2pf.C32_MAX_M:
            nxt = l2pf.resolve_dense_next(self, _nt)
            if nxt is not None:
                l2pf.fork(nxt, kind="moe", num_tokens=_nt)
            elif getattr(self, "l2pf_last", False):
                l2pf.fork(l2pf.get_head_weight(_nt), kind="head", num_tokens=_nt)
        elif _nt <= 64 and getattr(self, "l2pf_last", False):
            l2pf.fork(l2pf.get_head_weight(_nt), kind="head")
        out = tensor_model_parallel_all_reduce(out)  # EP or TP partials
        l2pf.join()
        return out


class Inkling2MoE(nn.Module):
    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format)

    def forward(
        self,
        x: torch.Tensor,
        reduce: bool = True,
        push: SeqParallel | None = None,
        lora: ExpertsLoRA | None = None,
        combine_only: bool = False,
        use_sendbuf: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """bf16 [T, H] -> bf16 [T, H], reduced over the expert-parallel group;
        without ``reduce``, the unreduced ``(down, src2dst)`` of
        ``Inkling2Experts`` (or unreduced combined ``[T, H]`` when
        ``combine_only=True``), or (with ``push``) pushes this rank's partial sums
        to the token shard owners.
        Padding tokens (``is_padding`` of the forward context) are routed to
        no expert and come out as 0. ``lora`` (``Inkling2MoEWithLoRA``) adds
        per-expert LoRA deltas inside the expert GEMMs, before any reduce or
        push; the NVFP4 / MXFP8 experts here do not support it."""
        if lora is not None:
            raise NotImplementedError(
                "Inkling2 expert LoRA needs BF16 experts; the NVFP4 / MXFP8 "
                "experts do not support it"
            )
        if not x.is_contiguous():
            _fp8_sc = getattr(x, "_inkling_fp8_scale", None)
            x = x.contiguous()
            if _fp8_sc is not None:
                x._inkling_fp8_scale = _fp8_sc  # type: ignore[attr-defined]
        gate, experts = self.gate, self.experts
        plan = experts.plan(x.shape[0])
        prezeroed_out = None
        zero_done_event = None
        if (
            (combine_only or experts.ep_size == 1)
            and push is None
            and plan.tile >= 64
            and not torch.cuda.is_current_stream_capturing()
        ):
            from .ops import pf_moe_native
            if (
                pf_moe_native.ASYNC_ZERO
                and pf_moe_native.COMBINE_MODE == 2
                and not (pf_moe_native.PERSISTENT and pf_moe_native.L2_ZERO_SLICE)
                and pf_moe_native.is_active(plan, tokens=x.shape[0])
            ):
                prezeroed_out, zero_done_event = pf_moe_native.alloc_and_async_zero_out(
                    x.shape[0], experts.hidden_size, x.device
                )
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
        routing, quantized = route(
            x,
            gate.weight,
            gate.bias,
            gate.counters,
            experts.counts,
            gate.topk,
            gate.route_scale,
            experts.expert_bounds,
            plan,
            *experts.scales("w13"),
            is_padding,
        )
        return experts(
            x,
            routing,
            quantized,
            plan,
            reduce,
            push,
            combine_only=combine_only,
            prezeroed_out=prezeroed_out,
            zero_done_event=zero_done_event,
            use_sendbuf=use_sendbuf,
        )
