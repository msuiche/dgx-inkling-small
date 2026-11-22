# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short convolutions with their state kept in a sliding-window cache.

Each decoder layer owns one ``InklingConvState`` (shared with Inkling v1)
holding its four conv streams (K, V, attn-out, mlp-out) packed head-major. K/V
are TP-sharded with the KV heads; the attn-/mlp-output streams are replicated
full-width on every rank (``replicate_hidden=True``).
"""

from typing import cast
import importlib.util
import os

import torch
from torch import nn

_FUSED_SCONV_AR = os.environ.get("INKLING_FUSED_SCONV_AR", "0").strip().lower() not in ("0", "", "off", "false", "no")
_FUSED_AR_SCONV = os.environ.get("INKLING_FUSED_AR_SCONV", "0").strip().lower() not in ("0", "", "off", "false", "no")
_FUSED_SCONV_AR_TRITON = os.environ.get("INKLING_FUSED_SCONV_AR_TRITON", "0").strip().lower() not in ("0", "", "off", "false", "no")
_TREE_SPEC = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in ("0", "", "off", "false", "no")


def _tree_active_for(num_tokens: int) -> bool:
    if not _TREE_SPEC:
        return False
    try:
        from .ops import tree_spec_runtime as _tsr
        return _tsr.get_active_tree_state(num_tokens) is not None
    except Exception:
        return False
_FUSED_SCONV_AR_PATH = os.environ.get("INKLING_SCONV_AR_EXT", "/rdma/inkling_sconv_ar_ext.so")
_FUSED_SCONV_AR_EXT = None
_FUSED_SCONV_AR_TRIED = False
_FUSED_SCONV_AR_PREFILL = os.environ.get("INKLING_FUSED_SCONV_AR_PREFILL", "1").strip().lower() not in ("0", "", "off", "false", "no")
_FUSED_PREFILL_MIN_T = int(os.environ.get("INKLING_FUSED_SCONV_AR_PREFILL_MIN_T", "1024"))
_FUSED_PREFILL_LIB_PATH = os.environ.get("INKLING_PREFILL_AR_LIB", "/rdma/libfused_sconv_ar_prefill.so")
_FUSED_PREFILL_LIB = None
_FUSED_PREFILL_LIB_TRIED = False
_FUSED_SCONV_AR_C32 = os.environ.get("INKLING_FUSED_SCONV_AR_C32", "0").strip().lower() in ("1", "true", "on", "yes")
_FUSED_C32_MIN_T = int(os.environ.get("INKLING_FUSED_SCONV_AR_C32_MIN_T", "17"))
_FUSED_C32_MAX_T = int(os.environ.get("INKLING_FUSED_SCONV_AR_C32_MAX_T", "512"))
_FUSED_C32_LIB_PATH = os.environ.get("INKLING_FUSED_SCONV_AR_C32_LIB", "/rdma/libfused_sconv_ar_c32_exact.so")
_FUSED_C32_LIB = None
_FUSED_C32_LIB_TRIED = False


def _get_fused_c32_lib():
    global _FUSED_C32_LIB, _FUSED_C32_LIB_TRIED
    if not _FUSED_SCONV_AR_C32:
        return None
    if not _FUSED_C32_LIB_TRIED:
        if torch.cuda.is_current_stream_capturing():
            return None
        _FUSED_C32_LIB_TRIED = True
        if os.path.exists(_FUSED_C32_LIB_PATH):
            try:
                import ctypes
                lib = ctypes.CDLL(_FUSED_C32_LIB_PATH)
                lib.fused_sconv_ar_c32_init_smem.argtypes = []
                lib.fused_sconv_ar_c32_init_smem.restype = ctypes.c_int
                argt = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.fused_sconv_ar_c32_row_fwd.argtypes = argt
                lib.fused_sconv_ar_c32_row_fwd.restype = ctypes.c_int
                lib.fused_sconv_ar_c32_blockd_fwd.argtypes = argt
                lib.fused_sconv_ar_c32_blockd_fwd.restype = ctypes.c_int
                rc = lib.fused_sconv_ar_c32_init_smem()
                if rc != 0:
                    raise RuntimeError(f"fused_sconv_ar_c32_init_smem failed: cudaError={rc}")
                _FUSED_C32_LIB = lib
                print(f"[inkling-sconv-ar-c32] loaded + init_smem OK: {_FUSED_C32_LIB_PATH}", flush=True)
            except Exception as e:
                print(f"[inkling-sconv-ar-c32] load failed ({e!r}); using unfused path", flush=True)
                _FUSED_C32_LIB = None
    return _FUSED_C32_LIB


def _get_fused_prefill_lib():
    global _FUSED_PREFILL_LIB, _FUSED_PREFILL_LIB_TRIED
    if not _FUSED_SCONV_AR_PREFILL:
        return None
    if not _FUSED_PREFILL_LIB_TRIED:
        if torch.cuda.is_current_stream_capturing():
            return None
        _FUSED_PREFILL_LIB_TRIED = True
        if os.path.exists(_FUSED_PREFILL_LIB_PATH):
            try:
                import ctypes
                lib = ctypes.CDLL(_FUSED_PREFILL_LIB_PATH)
                lib.fused_sconv_ar_prefill_exact_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.fused_sconv_ar_prefill_exact_fwd.restype = ctypes.c_int
                if hasattr(lib, "fused_sconv_ar_prefill_exact_v2_fwd"):
                    lib.fused_sconv_ar_prefill_exact_v2_fwd.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p,  # DiscardBlkPtr
                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_float, ctypes.c_float, ctypes.c_float,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_uint64,
                    ]
                    lib.fused_sconv_ar_prefill_exact_v2_fwd.restype = ctypes.c_int
                if hasattr(lib, "fused_sconv_ar_prefill_v2_init_smem"):
                    lib.fused_sconv_ar_prefill_v2_init_smem.argtypes = []
                    lib.fused_sconv_ar_prefill_v2_init_smem.restype = ctypes.c_int
                    rc = lib.fused_sconv_ar_prefill_v2_init_smem()
                    if rc != 0:
                        raise RuntimeError(f"fused_sconv_ar_prefill_v2_init_smem failed: cudaError={rc}")
                _FUSED_PREFILL_LIB = lib
                _tag = "v2 MWG+DiscardSkip" if hasattr(lib, "fused_sconv_ar_prefill_exact_v2_fwd") else "v1"
                print(f"[inkling-sconv-ar-prefill] loaded 0-ULP prefill fused kernel ({_tag}): {_FUSED_PREFILL_LIB_PATH}", flush=True)
            except Exception as e:
                print(f"[inkling-sconv-ar-prefill] load failed ({e!r}); using unfused path", flush=True)
                _FUSED_PREFILL_LIB = None
    return _FUSED_PREFILL_LIB


def _get_fused_sconv_ar():
    global _FUSED_SCONV_AR_EXT, _FUSED_SCONV_AR_TRIED
    if not _FUSED_SCONV_AR:
        return None
    if not _FUSED_SCONV_AR_TRIED:
        if torch.cuda.is_current_stream_capturing():
            return None
        _FUSED_SCONV_AR_TRIED = True
        try:
            if os.path.exists(_FUSED_SCONV_AR_PATH):
                spec = importlib.util.spec_from_file_location("inkling_sconv_ar_ext", _FUSED_SCONV_AR_PATH)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.init_smem()
                _FUSED_SCONV_AR_EXT = mod
                print(f"[inkling-sconv-ar] loaded + init_smem OK: {_FUSED_SCONV_AR_PATH}", flush=True)
        except Exception as e:
            print(f"[inkling-sconv-ar] load failed ({e!r}); using Triton fallback", flush=True)
            _FUSED_SCONV_AR_EXT = None
    return _FUSED_SCONV_AR_EXT


def _rowar_nlive_buf(num_tokens: int):
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as rar
        if rar.ROWAR.get("scope") == num_tokens:
            return rar.ROWAR.get("buf")
    except Exception:
        pass
    return None

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling.nvidia.sconv_swa_attn import (
    InklingConvState,
    InklingSconvMetadata,
    InklingSconvMetadataBuilder,
)

# INKLING_SCONV_META_DEDUP=1: all 6 InklingSconvMetadataBuilder instances compute the
# exact same (seq_idx, query_start) from common_attn_metadata.query_start_loc. Share one
# zero-initialized pair of buffers per device: 0 kernel launches at num_reqs==1 (all zeros!),
# and 1 launch instead of 6 at num_reqs>1 (~25 us/step saved at batch=1, 100% bit-exact).
_SCONV_META_DEDUP = os.environ.get("INKLING_SCONV_META_DEDUP", "0").strip().lower() in (
    "1", "on", "true", "yes",
)


def _install_sconv_meta_dedup() -> None:
    if not _SCONV_META_DEDUP or getattr(InklingSconvMetadataBuilder, "_inkling_dedup", False):
        return
    from vllm.models.inkling.nvidia.ops.sconv import sconv_seq_metadata

    _shared: dict = {}
    _orig_init = InklingSconvMetadataBuilder.__init__

    def _dedup_init(self, kv_cache_spec, layer_names, vllm_config, device):
        _orig_init(self, kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        key = (str(device), int(max_tokens))
        st = _shared.get(key)
        if st is None:
            st = {
                "seq": torch.zeros(max_tokens, dtype=torch.int32, device=device),
                "qs": torch.zeros(max_tokens, dtype=torch.int32, device=device),
                "dirty": False,
                "dirty_len": 0,
            }
            _shared[key] = st
            self._inkling_is_primary = True
        else:
            self._inkling_is_primary = False
        self._inkling_shared = st
        self.seq_idx_buffer = st["seq"]
        self.query_start_buffer = st["qs"]

    def _dedup_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens
        st = self._inkling_shared
        if num_reqs == 1:
            if st["dirty"]:
                clr = max(st["dirty_len"], num_padded_tokens)
                st["seq"][:clr].zero_()
                st["qs"][:clr].zero_()
                st["dirty"] = False
                st["dirty_len"] = 0
        else:
            if self._inkling_is_primary or not getattr(
                common_attn_metadata, "_inkling_sconv_meta_built", False
            ):
                sconv_seq_metadata(
                    common_attn_metadata.query_start_loc,
                    num_reqs,
                    num_actual_tokens,
                    self.seq_idx_buffer,
                    self.query_start_buffer,
                    num_padded_tokens,
                )
                st["dirty"] = True
                if num_padded_tokens > st["dirty_len"]:
                    st["dirty_len"] = num_padded_tokens
                try:
                    common_attn_metadata._inkling_sconv_meta_built = True
                except (AttributeError, TypeError):
                    pass
        return InklingSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )

    InklingSconvMetadataBuilder.__init__ = _dedup_init
    InklingSconvMetadataBuilder.build = _dedup_build
    InklingSconvMetadataBuilder._inkling_dedup = True


_install_sconv_meta_dedup()

from .ops.allreduce import Inkling2AllReduceWorkspace, all_reduce_conv_mix
from .ops.conv_mix import conv_mix as fused_conv_mix
from .ops.linear import linear_k_groups
from .ops.sconv import short_conv
from .ops.seq_parallel import SeqParallel

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)


def conv_metadata(owner: InklingConvState) -> InklingSconvMetadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    return cast(InklingSconvMetadata | None, attn_metadata.get(owner.prefix))


class Inkling2ShortConv(nn.Module):
    """Causal depthwise conv (+ residual, scaled) over one conv-state stream.

    Static: ``weight`` of shape ``(dim, 1, W)``. Dynamic: per-token taps
    ``static_w (W, dim) + weight_proj(dynamic_input)`` per head. The K/V convs
    only hold weights; the attention prologue kernel runs both.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        owner: InklingConvState,
        stream: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.stream = stream
        self.dynamic = dynamic_input_size is not None
        self.tp_rank = get_tensor_model_parallel_rank()
        if dynamic_input_size is not None:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim), requires_grad=False
            )
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size),
                requires_grad=False,
            )
            set_weight_attrs(self.static_w, {"weight_loader": self._load_dim1})
            set_weight_attrs(self.weight_proj, {"weight_loader": self._load_dim0})
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size), requires_grad=False
            )
            set_weight_attrs(self.weight, {"weight_loader": self._load_dim0})

    def _load(self, param: nn.Parameter, loaded: torch.Tensor, dim: int) -> None:
        size = param.shape[dim]
        if loaded.shape[dim] != size:
            loaded = loaded.narrow(dim, self.tp_rank * size, size)
        # sm121: staged H2D (gb10_load) instead of copy_ from the mmap view
        from .ops.gb10_load import copy_from_checkpoint_
        copy_from_checkpoint_(param.data, loaded)

    def _load_dim0(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 0)

    def _load_dim1(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 1)

    @property
    def conv_weight(self) -> torch.Tensor:
        """Static ``(dim, W)`` taps, or ``(W, dim)`` static part of dynamic taps."""
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        """Per-token tap deltas ``(T, num_heads * W)`` projected from ``x``."""
        if not self.dynamic:
            return None
        # Same dispatch as the reference ``linear``: fp32 1024-wide K groups.
        if self.weight_proj.shape[1] >= 2560 and self.weight_proj.shape[0] not in (
            1,
            2,
            4,
        ):
            return linear_k_groups(x.contiguous(), self.weight_proj, x.dtype)
        return nn.functional.linear(x, self.weight_proj)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        aux_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Static conv of ``x``; with ``addend``, returns ``(residual + addend)
        + conv(x)`` accumulated in fp32 (written to ``out`` if given)."""
        assert not self.dynamic
        m = conv_metadata(self.owner)
        if m is None:
            # No conv state (profile run): skip the conv, keep the residual sum.
            if addend is not None:
                acc = addend.float() if residual is None else residual + addend.float()
                x = (acc + x.to(addend.dtype).float()).to(addend.dtype)
            res = x if out is None else out.copy_(x)
            if aux_out is not None:
                aux_out.copy_(res)
            return res
        offset, width = self.owner.stream_ranges[self.stream]
        return short_conv(
            x,
            self.conv_weight,
            self.owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            addend,
            residual,
            out,
            aux_out=aux_out,
        )

    def can_fused_ar_sconv_attn_res(
        self,
        num_tokens: int,
        hidden: int,
        stack: torch.Tensor,
        queries: torch.Tensor,
        norm: nn.Module,
        normalize_values: bool,
    ) -> bool:
        if not _FUSED_AR_SCONV or self.dynamic or not normalize_values:
            return False
        if _TREE_SPEC and _tree_active_for(num_tokens):
            return False
        if not (1 <= num_tokens <= 8 and hidden == 4608):
            return False
        m = conv_metadata(self.owner)
        if m is None:
            return False
        nq = int(queries.shape[0])
        n_stack = int(stack.shape[0])
        offset, width = self.owner.stream_ranges[self.stream]
        c = self.owner.kv_cache
        cw = self.conv_weight
        if not (
            nq in (1, 3)
            and 1 <= n_stack <= 8
            and c.shape[1] == 1
            and width == 4608
            and offset % 8 == 0
            and c.stride(0) % 8 == 0
            and c.stride(2) % 8 == 0
            and c.stride(3) == 1
            and c.dtype == torch.bfloat16
            and cw.shape == (4608, 4)
            and cw.dtype == torch.bfloat16
            and cw.is_contiguous()
            and stack.dtype == torch.bfloat16
            and stack.stride(1) == 4608
            and stack.stride(2) == 1
            and queries.dtype == torch.float32
            and queries.is_contiguous()
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.is_contiguous()
            and m.slot_mapping.dtype == torch.int64
            and m.seq_idx.dtype == torch.int32
            and m.block_table.dtype == torch.int32
            and m.query_start.dtype == torch.int32
        ):
            return False
        try:
            from vllm.distributed import get_tp_group
            comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
        except Exception:
            return False
        return comm is not None and comm.can_fused_sconv_attnres(num_tokens, hidden)

    def try_fused_ar_sconv_attn_res(
        self,
        x_partial: torch.Tensor,
        positions: torch.Tensor,
        conv_out: torch.Tensor,
        stack: torch.Tensor,
        mix_last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: nn.Module,
        normalize_values: bool,
        conv_addend: torch.Tensor | None = None,
        conv_res: torch.Tensor | None = None,
        aux_out: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor | None:
        """s8: 0-ULP fused RDMA AllReduce + SConv + AttnRes + RMSNorm (eliminates 96x sconv + 96x attn_res!)."""
        if x_partial.dim() != 2 or x_partial.dtype != torch.bfloat16 or not x_partial.is_contiguous():
            return None
        if not self.can_fused_ar_sconv_attn_res(
            int(x_partial.shape[0]), int(x_partial.shape[1]), stack, queries, norm, normalize_values
        ):
            return None
        if not (
            positions.dtype == torch.int64
            and conv_out.is_contiguous()
            and conv_out.dtype == torch.bfloat16
            and (aux_out is None or (aux_out.is_contiguous() and aux_out.dtype == torch.bfloat16))
            and (conv_res is None or (conv_res.is_contiguous() and conv_res.dtype == torch.bfloat16))
            and (conv_addend is None or (conv_addend.is_contiguous() and conv_addend.dtype == torch.bfloat16))
            and (mix_last is None or (mix_last.is_contiguous() and mix_last.dtype == torch.bfloat16))
            and not (conv_res is not None and conv_addend is None)
            and not (conv_addend is not None and mix_last is not None)
        ):
            return None
        m = conv_metadata(self.owner)
        if m is None:
            return None
        offset, _ = self.owner.stream_ranges[self.stream]
        from vllm.distributed import get_tp_group
        comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
        if comm is None:
            return None
        return comm.all_reduce_sconv_attnres(
            x_partial,
            self.conv_weight,
            self.owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            conv_out,
            aux_out,
            conv_res,
            conv_addend,
            stack,
            mix_last,
            queries,
            norm.weight,
            eps=eps,
            norm_eps=norm.eps,
        )

    def try_fused_sconv_attn_res(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        conv_out: torch.Tensor,
        stack: torch.Tensor,
        mix_last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: nn.Module,
        normalize_values: bool,
        conv_addend: torch.Tensor | None = None,
        conv_res: torch.Tensor | None = None,
        eps: float = 1e-6,
        aux_out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """SM121 0-ULP fused sconv + attn_res + RMSNorm (+ optional aux_out).
        Returns (Q, T, D) bf16 output on success, or None to fall back to unfused path."""
        if self.dynamic:
            return None
        if _TREE_SPEC and _tree_active_for(int(x.shape[0])):
            return None
        # C=32 0-ULP 1-pass fused short_conv + AttnRes + RMSNorm (+ AuxOut) across ALL 96 layers (M=17..512):
        #   - T * nq <= 256 -> Regime A row_fwd (exact 256-thr BLOCK_A=4096, BLOCK_B=512 tree!)
        #   - T * nq >  256 -> Regime B blockd_fwd (exact 128-thr BLOCK_D=1024 Order-A FFMA tree!)
        if _FUSED_SCONV_AR_C32 and _FUSED_C32_MIN_T <= x.shape[0] <= _FUSED_C32_MAX_T:
            clib = _get_fused_c32_lib()
            m = conv_metadata(self.owner) if clib is not None else None
            if clib is not None and m is not None:
                T, D = x.shape
                nq = queries.shape[0]
                n_stack = stack.shape[0]
                offset, width = self.owner.stream_ranges[self.stream]
                c = self.owner.kv_cache
                cw = self.conv_weight
                if (
                    D == 4608
                    and nq in (1, 3)
                    and 1 <= n_stack <= 8
                    and normalize_values
                    and c.shape[1] == 1
                    and width == D
                    and offset % 8 == 0
                    and c.stride(0) % 8 == 0
                    and c.stride(2) % 8 == 0
                    and cw.shape == (D, 4)
                    and cw.is_contiguous()
                    and x.dtype == torch.bfloat16
                    and c.dtype == torch.bfloat16
                    and cw.dtype == torch.bfloat16
                    and norm.weight.dtype == torch.bfloat16
                    and stack.dtype == torch.bfloat16
                    and queries.dtype == torch.float32
                    and positions.dtype == torch.int64
                    and m.slot_mapping.dtype == torch.int64
                    and m.seq_idx.dtype == torch.int32
                    and m.block_table.dtype == torch.int32
                    and m.query_start.dtype == torch.int32
                    and x.stride(1) == 1
                    and x.stride(0) % 8 == 0
                    and c.stride(3) == 1
                    and conv_out.is_contiguous()
                    and conv_out.dtype == torch.bfloat16
                    and queries.is_contiguous()
                    and norm.weight.is_contiguous()
                    and stack.stride(1) == D
                    and stack.stride(2) == 1
                    and (aux_out is None or (aux_out.is_contiguous() and aux_out.dtype == torch.bfloat16))
                    and (conv_res is None or (conv_res.is_contiguous() and conv_res.dtype == torch.bfloat16))
                    and (conv_addend is None or (conv_addend.is_contiguous() and conv_addend.dtype == torch.bfloat16))
                    and (mix_last is None or (mix_last.is_contiguous() and mix_last.dtype == torch.bfloat16))
                    and not (conv_res is not None and conv_addend is None)
                    and not (conv_addend is not None and mix_last is not None)
                ):
                    out = torch.empty((nq, T, D), dtype=torch.bfloat16, device=x.device)
                    fn = clib.fused_sconv_ar_c32_row_fwd if (T * nq <= 256) else clib.fused_sconv_ar_c32_blockd_fwd
                    rc = fn(
                        x.data_ptr(), cw.data_ptr(), c.data_ptr(),
                        conv_out.data_ptr(),
                        0 if conv_res is None else conv_res.data_ptr(),
                        0 if conv_addend is None else conv_addend.data_ptr(),
                        0 if aux_out is None else aux_out.data_ptr(),
                        positions.data_ptr(), m.seq_idx.data_ptr(), m.slot_mapping.data_ptr(),
                        m.block_table.data_ptr(), m.query_start.data_ptr(),
                        stack.data_ptr(), 0 if mix_last is None else mix_last.data_ptr(),
                        queries.data_ptr(), norm.weight.data_ptr(), out.data_ptr(),
                        int(x.stride(0)), int(c.stride(0)), int(c.stride(2)), int(m.block_table.stride(0)),
                        int(m.block_table.shape[1]), int(c.shape[2]), int(offset), int(stack.stride(0)),
                        nq, T, n_stack, float(5**-0.5), float(eps), float(norm.eps),
                        1 if conv_res is not None else 0,
                        1 if conv_addend is not None else 0,
                        1 if mix_last is not None else 0,
                        int(torch.cuda.current_stream().cuda_stream),
                    )
                    if rc == 0:
                        return out
        # 0-ULP 1-pass prefill fused short_conv + AttnRes + RMSNorm (T >= 1024; below 1024 conv_out fits L2)
        if _FUSED_SCONV_AR_PREFILL and x.shape[0] >= _FUSED_PREFILL_MIN_T:
            plib = _get_fused_prefill_lib()
            m = conv_metadata(self.owner) if plib is not None else None
            if plib is not None and m is not None:
                T, D = x.shape
                nq = queries.shape[0]
                n_stack = stack.shape[0]
                offset, width = self.owner.stream_ranges[self.stream]
                c = self.owner.kv_cache
                cw = self.conv_weight
                if (
                    D == 4608
                    and nq in (1, 3)
                    and 1 <= n_stack <= 8
                    and normalize_values
                    and T * nq > 256
                    and c.shape[1] == 1
                    and width == D
                    and cw.shape == (D, 4)
                    and cw.is_contiguous()
                    and x.dtype == torch.bfloat16
                    and c.dtype == torch.bfloat16
                    and cw.dtype == torch.bfloat16
                    and norm.weight.dtype == torch.bfloat16
                    and stack.dtype == torch.bfloat16
                    and queries.dtype == torch.float32
                    and positions.dtype == torch.int64
                    and m.slot_mapping.dtype == torch.int64
                    and m.seq_idx.dtype == torch.int32
                    and m.block_table.dtype == torch.int32
                    and m.query_start.dtype == torch.int32
                    and x.stride(1) == 1
                    and c.stride(3) == 1
                    and conv_out.is_contiguous()
                    and queries.is_contiguous()
                    and norm.weight.is_contiguous()
                    and stack.stride(1) == D
                    and stack.stride(2) == 1
                    and (conv_res is None or (conv_res.is_contiguous() and conv_res.dtype == torch.bfloat16))
                    and (conv_addend is None or (conv_addend.is_contiguous() and conv_addend.dtype == torch.bfloat16))
                    and (mix_last is None or (mix_last.is_contiguous() and mix_last.dtype == torch.bfloat16))
                    and not (conv_res is not None and conv_addend is None)
                ):
                    out = torch.empty((nq, T, D), dtype=torch.bfloat16, device=x.device)
                    rc = plib.fused_sconv_ar_prefill_exact_fwd(
                        x.data_ptr(), cw.data_ptr(), c.data_ptr(),
                        conv_out.data_ptr(),
                        0 if conv_res is None else conv_res.data_ptr(),
                        0 if conv_addend is None else conv_addend.data_ptr(),
                        positions.data_ptr(), m.seq_idx.data_ptr(), m.slot_mapping.data_ptr(),
                        m.block_table.data_ptr(), m.query_start.data_ptr(),
                        stack.data_ptr(), 0 if mix_last is None else mix_last.data_ptr(),
                        queries.data_ptr(), norm.weight.data_ptr(), out.data_ptr(),
                        int(x.stride(0)), int(c.stride(0)), int(c.stride(2)), int(m.block_table.stride(0)),
                        int(m.block_table.shape[1]), int(c.shape[2]), int(offset), int(stack.stride(0)),
                        nq, T, n_stack, float(5**-0.5), float(eps), float(norm.eps),
                        1 if conv_res is not None else 0,
                        1 if conv_addend is not None else 0,
                        1 if mix_last is not None else 0,
                        int(torch.cuda.current_stream().cuda_stream),
                    )
                    if rc == 0:
                        if aux_out is not None:
                            aux_out.copy_(conv_out)
                        return out
        if _FUSED_SCONV_AR_TRITON:
            m = conv_metadata(self.owner)
            if m is None:
                return None
            T, D = x.shape
            nq = queries.shape[0]
            n_stack = stack.shape[0]
            offset, width = self.owner.stream_ranges[self.stream]
            c = self.owner.kv_cache
            cw = self.conv_weight
            if (
                0 < T <= 8
                and D == 4608
                and nq in (1, 3)
                and 1 <= n_stack <= 10
                and c.shape[1] == 1
                and width == D
                and cw.shape[1] == 4
                and x.stride(1) == 1
                and c.stride(3) == 1
                and conv_out.is_contiguous()
                and queries.is_contiguous()
                and norm.weight.is_contiguous()
                and stack.stride(1) == D
                and stack.stride(2) == 1
            ):
                from .ops.sconv import fused_sconv_attn_res_triton
                early_pdl = os.environ.get("INKLING_PDL_SKINNY", "0") == "1"
                use_nlive = os.environ.get("INKLING_SCONV_NLIVE", "0") == "1"
                nl_buf = _rowar_nlive_buf(T) if use_nlive else None
                return fused_sconv_attn_res_triton(
                    x, cw, c, positions, m.block_table, m.seq_idx, m.slot_mapping, m.query_start,
                    offset, conv_out, conv_res, conv_addend, stack, mix_last, queries, norm.weight,
                    eps, norm.eps, normalize_values, early_pdl, nl_buf,
                )
            return None
        ext = _get_fused_sconv_ar()
        if ext is None:
            return None
        m = conv_metadata(self.owner)
        if m is None:
            return None
        T, D = x.shape
        nq = queries.shape[0]
        n_stack = stack.shape[0]
        if (
            T > 8
            or T == 0
            or D != 4608
            or nq not in (1, 3)
            or n_stack < 1
            or n_stack > 10
            or x.dtype != torch.bfloat16
            or self.owner.kv_cache.dtype != torch.bfloat16
            or norm.weight.dtype != torch.bfloat16
            or queries.dtype != torch.float32
            or positions.dtype != torch.int64
            or m.slot_mapping.dtype != torch.int64
            or m.seq_idx.dtype != torch.int32
            or m.block_table.dtype != torch.int32
            or m.query_start.dtype != torch.int32
            or x.stride(1) != 1
            or not conv_out.is_contiguous()
            or not queries.is_contiguous()
            or not norm.weight.is_contiguous()
            or stack.stride(1) != D
            or stack.stride(2) != 1
        ):
            return None
        offset, width = self.owner.stream_ranges[self.stream]
        c = self.owner.kv_cache
        if (
            width % 8 != 0
            or offset % 8 != 0
            or c.stride(0) % 8 != 0
            or c.stride(1) % 8 != 0
            or c.stride(2) % 8 != 0
            or c.stride(3) != 1
        ):
            return None
        cw = self.conv_weight
        if cw.dtype != torch.bfloat16 or not cw.is_contiguous() or cw.shape[1] != 4:
            return None
        out = torch.empty((nq, T, D), dtype=torch.bfloat16, device=x.device)
        pdl_on = os.environ.get("INKLING_SCONV_AR_PDL", "1").strip().lower() not in ("0", "", "off", "false", "no")
        early_pdl = pdl_on and os.environ.get("INKLING_SCONV_AR_EARLY", os.environ.get("INKLING_PDL_SKINNY", "0")) == "1"
        use_nlive = os.environ.get("INKLING_SCONV_NLIVE", "0") == "1"
        nl_buf = _rowar_nlive_buf(T) if use_nlive else None
        if (
            os.environ.get("INKLING_SCONV_AR_AUDIT", "0") == "1"
            and not torch.cuda.is_current_stream_capturing()
        ):
            from .ops.attn_res import attn_res as _ref_attn_res
            _c = self.owner.kv_cache
            _cn = int(_c.shape[2])
            _slots = m.slot_mapping.tolist()
            _saved = []
            for _sl in set(_slots):
                if _sl >= 0:
                    _b, _s = _sl // _cn, _sl % _cn
                    _saved.append((_b, _s, _c[_b, :, _s, offset : offset + width].clone()))
            _co_ref = conv_out.clone()
            _cr_ref = conv_res.clone() if conv_res is not None else None
            _ml_ref = mix_last.clone() if mix_last is not None else None
            _st_ref = stack.clone()
            short_conv(
                x, cw, _c, positions, m.block_table, m.seq_idx,
                m.slot_mapping, m.query_start, offset, width,
                conv_addend, _cr_ref, _co_ref,
            )
            if _ml_ref is not None:
                _ref = _ref_attn_res(_st_ref, _ml_ref, _co_ref, queries, norm.weight, norm.eps, normalize_values, eps)
            else:
                _ref = _ref_attn_res(_st_ref, _co_ref, None, queries, norm.weight, norm.eps, normalize_values, eps)
            for _b, _s, _v in _saved:
                _c[_b, :, _s, offset : offset + width].copy_(_v)
            ext.sconv_attn_res_fwd(
                x, cw, _c, positions, m.block_table, m.seq_idx,
                m.slot_mapping, m.query_start, offset, width,
                conv_out, conv_res, conv_addend, stack, mix_last, queries, norm.weight, out,
                eps, norm.eps, normalize_values, early_pdl, pdl_on, nl_buf,
            )
            _dc = (conv_out.view(torch.int16) != _co_ref.view(torch.int16)).sum().item()
            _do = (out.view(torch.int16) != _ref.view(torch.int16)).sum().item()
            if _dc or _do:
                _mc = (conv_out.float() - _co_ref.float()).abs().max().item()
                _mo = (out.float() - _ref.float()).abs().max().item()
                print(
                    f"[SCONV-AR-AUDIT] DIFF prefix={self.owner.prefix} stream={self.stream} "
                    f"T={T} nq={nq} ns={n_stack} dc={_dc}({_mc:.2e}) do={_do}({_mo:.2e}) "
                    f"ca={conv_addend is not None} cr={conv_res is not None} ma={mix_last is not None} "
                    f"alias_cr={conv_res is not None and conv_res.data_ptr()==conv_out.data_ptr()} "
                    f"slots={_slots} pos={positions.tolist()} qs={m.query_start.tolist()} "
                    f"bt0={m.block_table[0, :8].tolist()} cn={_cn} off={offset} w={width}",
                    flush=True,
                )
                if not os.path.exists("/prof/sconv_fail.pt") and get_tensor_model_parallel_rank() == 0:
                    torch.save({
                        "prefix": self.owner.prefix, "stream": self.stream,
                        "st": _st_ref.cpu(), "co": _co_ref.cpu(),
                        "ml": _ml_ref.cpu() if _ml_ref is not None else None,
                        "q": queries.cpu(), "nw": norm.weight.cpu(),
                        "eps": eps, "norm_eps": norm.eps, "norm_v": normalize_values,
                        "ref": _ref.cpu(), "new": out.cpu(),
                    }, "/prof/sconv_fail.pt")
                    print("[SCONV-AR-AUDIT] SAVED /prof/sconv_fail.pt", flush=True)
            return out
        ext.sconv_attn_res_fwd(
            x,
            cw,
            self.owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            conv_out,
            conv_res,
            conv_addend,
            stack,
            mix_last,
            queries,
            norm.weight,
            out,
            eps,
            norm.eps,
            normalize_values,
            early_pdl,
            pdl_on,
            nl_buf,
        )
        return out

    def conv_mix(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        stack: torch.Tensor,
        last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: nn.Module,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        sp: SeqParallel | None = None,
        halo: int = 0,
        prefix_align: int = 0,
    ) -> torch.Tensor:
        """Convolve the reduced ``x`` into ``out`` like ``forward`` and return
        the AttnRes mixes of ``[*stack, (last +) out]`` with ``queries``
        through ``norm``. With ``sp``, the hidden stream is this rank's token
        shard, ``x`` its reduce-scatter (after ``halo`` rows) and the mixes
        are all-gathered; profile runs (no conv-state cache) need ``sp``."""
        assert not self.dynamic
        m = conv_metadata(self.owner)
        cache = self.owner.kv_cache
        if m is None:
            assert sp is not None
            m, cache, prefix_align = self._profile_metadata(positions)
        offset, width = self.owner.stream_ranges[self.stream]
        return fused_conv_mix(
            x,
            self.conv_weight,
            cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            out,
            residual,
            addend,
            stack,
            last,
            queries,
            norm.weight,
            norm.eps,
            sp=sp,
            halo=halo,
            prefix_align=prefix_align,
        )

    def all_reduce_mix(
        self,
        workspace: Inkling2AllReduceWorkspace,
        positions: torch.Tensor,
        out: torch.Tensor,
        stack: torch.Tensor,
        last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: nn.Module,
        x: torch.Tensor | None = None,
        combine: tuple[torch.Tensor, torch.Tensor] | None = None,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        two_shot: bool = False,
    ) -> torch.Tensor:
        """All-reduce this rank's partial sum (``x``, or the MoE ``combine``)
        in one or ``two_shot``s, convolve it into ``out`` like ``forward``,
        and return the AttnRes mixes of ``[*stack, (last +) out]`` with
        ``queries`` through ``norm``."""
        m = conv_metadata(self.owner)
        assert m is not None and not self.dynamic
        offset, width = self.owner.stream_ranges[self.stream]
        return all_reduce_conv_mix(
            workspace,
            x,
            combine,
            self.conv_weight,
            self.owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            out,
            residual,
            addend,
            stack,
            last,
            queries,
            norm.weight,
            norm.eps,
            two_shot=two_shot,
        )

    def _profile_metadata(
        self, positions: torch.Tensor
    ) -> tuple[InklingSconvMetadata, torch.Tensor, int]:
        """Metadata of a batch without conv-state cache slots, a one-block
        cache and a prefix alignment for it."""
        owner = self.owner
        zeros = torch.zeros_like(positions, dtype=torch.int32)
        metadata = InklingSconvMetadata(
            block_table=zeros[:1].view(1, 1),
            slot_mapping=torch.full_like(positions, -1),
            seq_idx=zeros,
            query_start=zeros,
        )
        cache = torch.empty(
            (1, owner.num_kv_heads, owner.block_size, owner.head_size),
            dtype=torch.bfloat16,
            device=positions.device,
        )
        return metadata, cache, owner.block_size
