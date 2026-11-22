# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Small per-step CUDA graph capture for eager decode glue (INKLING_DTGRAPH).

Ported and upgraded from DSv4.1 climb-r1/r1-eager (`dsv41_dtgraph.py`) for Inkling-2-Small TP8:
  * Phase 4 (`dt` region, 4_dft2tgt_glue): captures GPUModelRunner.prepare_inputs +
    prepare_attn + model_state.prepare_attn + model_state.prepare_inputs_embeds (including
    _vw_mask, INKLING_REPL_EMBED lookup, embed_norm, and inputs_embeds DtoD copy) up to
    cudagraph_manager.run_fullgraph.
  * Phase 2B (`td` region, 2_tgt2dft_glue post-AsyncOutput): captures
    GPUModelRunner.postprocess_sampled + speculator.prepare_propose + speculator.propose
    pre-graph work (es_init, prepare_dflash_inputs, _store_hidden_states, _num_rejected copy)
    up to speculator.query_cudagraph_manager.run_fullgraph.
    NOTE: Phase 2A (cert_head + sync_free AG + direct_reject + AsyncOutput s20 copy) stays
    eager because cert_head is 595 us (few ops), contains the RDMA AG collective, and
    sync_free/cert_head gate on is_current_stream_capturing(). Phase 2B starts strictly AFTER
    AsyncOutput.__init__ returns to the main stream and has zero collectives and zero
    is_current_stream_capturing() branches.

Key upgrades over DSv4.1 r1-eager:
  1. Zero edits to vLLM's model_runner.py: installs cleanly via lazy hooks on GPUModelRunner
     and CUDAGraphManager / DFlashCudaGraphManager (same pattern as patches-s5/sync_free.py).
  2. `cudaGraphExecUpdate` execs are keyed by `(region_name, shape_key, seg, slot)` so
     alternating batch sizes (T=8, T=16, batch>1) hit ExecUpdate 100% without topology churn.
  3. Full support for PARITY=1 (IMAGES=16, SEQS=32, 1M ctx): multimodal prefill / encoder
     steps are skipped; decode steps' pinned `is_mm_embed` allocations are served from the
     persistent pinned Arena (zero CachingHostAllocator pinned-memory leak).
  4. 100% compatible with dynamic W (`INKLING_VERIFY_W=conf|conf+ema|ema|force:K`),
     `INKLING_REPL_EMBED=1` (or 0, via RDMA hazard rail), `INKLING_MARKOV_EARLY_STOP=1`.

Env:
  INKLING_DTGRAPH=off|0|dt|td|both|update|warm=N|seg=N|max_reqs=N|log=N
  Recommended: INKLING_DTGRAPH=both,update  (or INKLING_DTGRAPH=dt,update)
  Default: off (zero imports, zero hooks, zero overhead).
"""
from __future__ import annotations

import dataclasses
import os
import threading
import time
import warnings
from collections import Counter, OrderedDict, deque

import numpy as np
import torch

TAG = "inkling-dtgraph"


def parse_flag(raw: str | None) -> dict:
    cfg = {
        "regions": set(),
        "update": False,
        "warm": 2,
        "seg": 4,
        "max_reqs": 0,  # 0 = unlimited (all FULL-cudagraph batch sizes)
        "max_execs": max(0, int(os.environ.get("INKLING_DTGRAPH_MAX_EXECS", "128") or "128")),
        "log": 5000,
    }
    for tok in (raw or "").replace("+", ",").split(","):
        tok = tok.strip().lower()
        if tok in ("", "0", "off", "false", "no"):
            continue
        if tok in ("1", "on", "true", "dt"):
            cfg["regions"].add("dt")
        elif tok == "td":
            cfg["regions"].add("td")
        elif tok in ("both", "all"):
            cfg["regions"].update(("dt", "td"))
        elif tok == "update":
            cfg["update"] = True
        elif "=" in tok and tok.split("=", 1)[0] in ("warm", "seg", "max_reqs", "max_execs", "log"):
            k, v = tok.split("=", 1)
            cfg[k] = max(1 if k == "seg" else 0, int(v))
        else:
            raise ValueError(
                f"INKLING_DTGRAPH: unknown token {tok!r} "
                "(use dt, td, both, update, warm=N, seg=N, max_reqs=N, max_execs=N, log=N)"
            )
    if cfg["update"] and not cfg["regions"]:
        cfg["regions"].update(("dt", "td"))
    return cfg


CFG = parse_flag(os.environ.get("INKLING_DTGRAPH", ""))
ON = bool(CFG["regions"])

_OFF = frozenset(("", "0", "off", "false", "no"))
# S7 Phase-1 step-boundary zero-math-change accelerators (all 0-ULP bit-exact, default ON):
#   INKLING_FUSED_DT_PREP=1       : 1-CTA Triton prepare_inputs for B=1 pure-decode (zero H2D DMA, -18..21 us/step)
#   INKLING_FUSED_DT_PREP_MAX_B=1 : opt-in B=2..32 uniform pure-decode fused prepare_inputs (default 1 = B=1 only)
#   INKLING_FUSED_EMBED_DIRECT=1  : write _fused_vw_repl_embed_norm directly into inputs_embeds (zero 73KB DtoD, -2.5..6 us/step)
#   INKLING_FUSED_DRAFT_STORE=1   : fold req_states.draft_tokens scatter into _vw_conf + skip readback (-10.5..11.5 us/step)
#   INKLING_PREFILL_PREWARM=0     : opt-in post-capture warmup of lazy prefill .so libs, FP8 weight cache & multi-req JITs
FUSED_DT_PREP = os.environ.get("INKLING_FUSED_DT_PREP", "1").strip().lower() not in _OFF
FUSED_DT_PREP_MAX_B = max(1, int(os.environ.get("INKLING_FUSED_DT_PREP_MAX_B", "1") or "1"))
FUSED_EMBED_DIRECT = os.environ.get("INKLING_FUSED_EMBED_DIRECT", "1").strip().lower() not in _OFF
FUSED_DRAFT_STORE = os.environ.get("INKLING_FUSED_DRAFT_STORE", "1").strip().lower() not in _OFF
PREFILL_PREWARM = os.environ.get("INKLING_PREFILL_PREWARM", "0").strip().lower() not in _OFF

PIN_ARENA = True  # unit tests on CPU-only torch set this False
SLOT_BYTES = 64 * 1024
NSLOT = 4
ALIGN = 256

# ----------------------------------------------------------------------------------------------
# Op classification (pure; unit-tested on CPU with "meta" standing in for CUDA)
# ----------------------------------------------------------------------------------------------
PASS, PASS_CPU, STAGE, HAZ, PIN = "pass", "pass_cpu", "stage", "haz", "pin"

_SYNC_OPS = frozenset({
    "_local_scalar_dense", "item", "nonzero", "nonzero_numpy", "argwhere", "masked_select",
    "_unique", "_unique2", "unique_dim", "unique_consecutive", "unique_dim_consecutive",
    "repeat_interleave", "bincount", "histc", "equal", "is_nonzero", "allclose", "isin_sync",
    "_assert_scalar", "_print", "segment_reduce", "_nested_tensor_from_mask",
})
_RNG_OPS = frozenset({
    "uniform", "uniform_", "normal", "normal_", "exponential", "exponential_", "bernoulli",
    "bernoulli_", "random", "random_", "multinomial", "rand", "rand_like", "randn", "randn_like",
    "randint", "randint_like", "randperm", "geometric", "geometric_", "cauchy", "cauchy_",
    "log_normal", "log_normal_", "rrelu", "rrelu_with_noise", "dropout", "native_dropout",
    "poisson", "_standard_gamma", "_sample_dirichlet", "binomial", "feature_dropout",
    "alpha_dropout", "_fused_dropout",
})
_COLL_WORDS = (
    "all_reduce", "allreduce", "all_gather", "allgather", "reduce_scatter", "broadcast",
    "all_to_all", "alltoall", "send", "recv", "barrier", "gather",
)
_COLL_NS = frozenset({"c10d", "_c10d_functional", "_c10d_functional_autograd", "_dtensor"})
_INDEX_OPS = frozenset({
    "index", "index_put", "index_put_", "_index_put_impl_", "_unsafe_index", "_unsafe_index_put",
})
_VIEW_OPS = frozenset({
    "view", "_unsafe_view", "slice", "select", "as_strided", "alias", "t", "transpose", "permute",
    "unsqueeze", "squeeze", "expand", "narrow", "detach", "reshape", "_reshape_alias", "flatten",
    "unflatten", "view_as", "split", "split_with_sizes", "chunk", "unbind", "diagonal", "numpy_T",
    "lift_fresh", "dim", "size", "stride", "sym_size", "sym_stride", "sym_numel", "is_pinned",
})
_META_FACTORY_OPS = frozenset({
    "new_empty", "new_empty_strided", "new_zeros", "new_ones", "new_full",
    "empty_like", "zeros_like", "ones_like", "full_like",
})
_INT_BOOL_DTYPES = frozenset({
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
})


def _default_is_dev(x) -> bool:
    return getattr(x, "is_cuda", False)


def _tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _tensors(o, out)
    return out


def _is_dev_obj(d, is_dev) -> bool:
    if d is None:
        return False
    if isinstance(d, str):
        d = torch.device(d)
    if isinstance(d, torch.device):
        return is_dev(torch.empty(0, device="meta")) if d.type == "meta" else d.type == "cuda"
    return False


def _is_cpu_obj(d) -> bool:
    if d is None:
        return False
    if isinstance(d, str):
        d = torch.device(d)
    return isinstance(d, torch.device) and d.type == "cpu"


def _dev_kw(kwargs, is_dev) -> bool:
    return _is_dev_obj(kwargs.get("device"), is_dev)


def _is_cpu(t) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "cpu"


def _dtype_size(dtype) -> int:
    sz = getattr(dtype, "itemsize", None)
    return int(sz) if sz is not None else torch.empty(0, dtype=dtype).element_size()


def op_name(func) -> tuple[str, str]:
    try:
        full = func._schema.name
    except AttributeError:
        return "", str(func)
    ns, _, name = full.partition("::")
    return ns, name


def _op_overload(func) -> str:
    try:
        return getattr(func._schema, "overload_name", "") or ""
    except AttributeError:
        return ""


def _parse_to(func, args, kwargs, is_dev):
    """Parse all PyTorch aten::to overloads (dtype_layout, device, other, dtype, prim_Device).
    Returns (ov, src, to_dev, to_cpu, target_dtype, pin_memory, nb, bad_layout)."""
    src = args[0]
    ov = _op_overload(func)
    if not ov:
        if (len(args) >= 2 and isinstance(args[1], torch.Tensor)) or isinstance(
            kwargs.get("other"), torch.Tensor
        ):
            ov = "other"
        elif len(args) >= 2 and isinstance(args[1], (torch.device, str)):
            ov = "device"
        elif len(args) >= 2 and isinstance(args[1], torch.dtype):
            ov = "dtype"
        else:
            ov = "dtype_layout"

    layout = kwargs.get("layout")
    mem_fmt = kwargs.get("memory_format")
    pin_memory = bool(kwargs.get("pin_memory") or False)
    if ov == "other":
        other = args[1] if len(args) >= 2 else kwargs.get("other")
        to_dev = is_dev(other) if isinstance(other, torch.Tensor) else False
        to_cpu = _is_cpu(other)
        dt = other.dtype if isinstance(other, torch.Tensor) else None
        nb = bool(args[2]) if len(args) > 2 else bool(kwargs.get("non_blocking", False))
        if len(args) > 4 and mem_fmt is None:
            mem_fmt = args[4]
    elif ov in ("device", "prim_Device"):
        d = args[1] if len(args) >= 2 else kwargs.get("device")
        to_dev = _is_dev_obj(d, is_dev)
        to_cpu = _is_cpu_obj(d)
        dt = args[2] if len(args) > 2 else kwargs.get("dtype")
        nb = bool(args[3]) if len(args) > 3 else bool(kwargs.get("non_blocking", False))
        if len(args) > 5 and mem_fmt is None:
            mem_fmt = args[5]
    elif ov == "dtype":
        to_dev = False
        to_cpu = False
        dt = args[1] if len(args) >= 2 else kwargs.get("dtype")
        nb = bool(args[2]) if len(args) > 2 else bool(kwargs.get("non_blocking", False))
        if len(args) > 4 and mem_fmt is None:
            mem_fmt = args[4]
    else:  # dtype_layout
        d = kwargs.get("device")
        to_dev = _is_dev_obj(d, is_dev)
        to_cpu = _is_cpu_obj(d)
        dt = kwargs.get("dtype")
        nb = bool(kwargs.get("non_blocking", False))

    bad_layout = (
        layout not in (None, torch.strided)
        or mem_fmt not in (None, torch.preserve_format, torch.contiguous_format)
    )
    return ov, src, to_dev, to_cpu, dt, pin_memory, nb, bad_layout


def classify(func, args, kwargs, is_dev=_default_is_dev, owned=lambda t: False) -> tuple[str, str]:
    kwargs = kwargs or {}
    ns, name = op_name(func)
    if ns == "aten" and name in _VIEW_OPS:
        return PASS, ""
    if name in ("empty", "empty_strided", "zeros") and kwargs.get("pin_memory"):
        return PIN, "pin_alloc"
    if name == "_pin_memory":
        return PIN, "pin_copy"
    ts = _tensors(args, [])
    _tensors(kwargs, ts)
    dev = [t for t in ts if is_dev(t)]
    devkw = _dev_kw(kwargs, is_dev)
    if ns in _COLL_NS or (ns == "vllm" and any(w in name for w in _COLL_WORDS)):
        return HAZ, f"collective:{ns}::{name}"
    if name in _RNG_OPS:
        return (HAZ, f"rng:{name}") if (dev or devkw) else (PASS_CPU, "")
    if name == "record_stream":
        return HAZ, "stream:record_stream"
    if ns == "aten" and name in _META_FACTORY_OPS and not kwargs.get("pin_memory"):
        return (PASS, "") if (dev or devkw) else (PASS_CPU, "")
    if name == "copy_" and (len(args) >= 2 or (len(args) == 1 and "src" in kwargs)):
        dst = args[0]
        src = args[1] if len(args) >= 2 else kwargs["src"]
        nb = bool(args[2]) if len(args) > 2 else bool(kwargs.get("non_blocking", False))
        if is_dev(dst) and _is_cpu(src):
            if not nb:
                return HAZ, "h2d_blocking"
            if src.dtype != dst.dtype:
                return HAZ, "h2d_cast"
            if owned(src):
                return PASS, ""
            if not src.is_contiguous():
                return HAZ, "h2d_noncontig"
            return STAGE, "h2d"
        if _is_cpu(dst) and is_dev(src):
            return HAZ, "d2h"
        return (PASS, "") if dev else (PASS_CPU, "")
    if name == "to" and args and isinstance(args[0], torch.Tensor):
        ov, src, to_dev, to_cpu, dt, pin_mem, nb, bad_layout = _parse_to(func, args, kwargs, is_dev)
        if _is_cpu(src) and to_dev:
            if bad_layout:
                return HAZ, "h2d_to_layout"
            if dt is not None and dt != src.dtype:
                if not (
                    isinstance(dt, torch.dtype)
                    and src.dtype in _INT_BOOL_DTYPES
                    and dt in _INT_BOOL_DTYPES
                ):
                    return HAZ, f"h2d_to_cast:{src.dtype}->{dt}"
                return STAGE, "h2d_to"
            if src.numel() == 0 and nb:
                return PASS, ""
            if owned(src) and nb and src.is_contiguous():
                return PASS, ""
            # Stage into persistent pinned Arena and upgrade non_blocking=True (0-ULP raw copy,
            # safe against caller CPU mutation and capture-illegal cudaStreamSynchronize).
            return STAGE, "h2d_to"
        if is_dev(src) and to_cpu:
            return HAZ, "d2h"
        if _is_cpu(src) and pin_mem and not to_dev:
            if bad_layout:
                return HAZ, "pin_layout"
            if dt is not None and dt != src.dtype and not (
                isinstance(dt, torch.dtype)
                and src.dtype in _INT_BOOL_DTYPES
                and dt in _INT_BOOL_DTYPES
            ):
                return HAZ, "pin_cast"
            return PIN, "pin_copy"
        return (PASS, "") if (is_dev(src) or to_dev) else (PASS_CPU, "")
    if name == "_to_copy" and args:
        src = args[0]
        to_dev = devkw
        to_cpu = _is_cpu_obj(kwargs.get("device"))
        nb = bool(kwargs.get("non_blocking", False))
        if _is_cpu(src) and to_dev:
            if not nb:
                return HAZ, "h2d_blocking"
            dt = kwargs.get("dtype")
            if dt is not None and dt != src.dtype:
                return HAZ, "h2d_cast"
            if owned(src):
                return PASS, ""
            if not src.is_contiguous():
                return HAZ, "h2d_noncontig"
            return STAGE, "h2d"
        if is_dev(src) and to_cpu:
            return HAZ, "d2h"
        return (PASS, "") if (dev or to_dev) else (PASS_CPU, "")
    if not dev and not devkw:
        return PASS_CPU, ""
    if name in _SYNC_OPS or (name == "where" and len(args) == 1):
        return HAZ, f"sync:{name}"
    if name in _INDEX_OPS:
        idx = args[1] if len(args) > 1 else kwargs.get("indices")
        for t in _tensors(idx, []):
            if t.dtype in (torch.bool, torch.uint8):
                return HAZ, "mask_index"
    for t in ts:
        if _is_cpu(t) and t.dim() > 0:
            return HAZ, f"cpu_arg:{ns}::{name}"
    return PASS, ""


# ----------------------------------------------------------------------------------------------
# Pinned staging arena
# ----------------------------------------------------------------------------------------------
class Arena:
    def __init__(self, slot_bytes: int = SLOT_BYTES, nslot: int = NSLOT, pin: bool | None = None):
        self.slot_bytes, self.nslot = slot_bytes, nslot
        pin = PIN_ARENA if pin is None else pin
        self.buf = torch.zeros(slot_bytes * nslot, dtype=torch.uint8, pin_memory=pin)
        self.slot, self.off = 0, 0

    def start(self, slot: int) -> None:
        self.slot, self.off = slot, 0

    def owns(self, t) -> bool:
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            return False
        lo = self.buf.data_ptr()
        return lo <= t.data_ptr() < lo + self.buf.numel()

    def alloc(self, shape, dtype) -> torch.Tensor | None:
        n = 1
        for d in shape:
            n *= int(d)
        nb = n * _dtype_size(dtype)
        a = (self.off + ALIGN - 1) // ALIGN * ALIGN
        if a + max(nb, 1) > self.slot_bytes:
            return None
        base = self.slot * self.slot_bytes + a
        self.off = a + nb
        return self.buf[base : base + nb].view(dtype).view(tuple(int(d) for d in shape))

    def take_cast(self, src: torch.Tensor, dtype=None) -> torch.Tensor | None:
        dt = src.dtype if dtype is None else dtype
        n = src.numel() * _dtype_size(dt)
        a = (self.off + ALIGN - 1) // ALIGN * ALIGN
        if a + max(n, 1) > self.slot_bytes:
            return None
        base = self.slot * self.slot_bytes + a
        view = self.buf[base : base + n].view(dt).view(src.shape)
        if n > 0:
            view.copy_(src)
        self.off = a + n
        return view

    def take(self, src: torch.Tensor) -> torch.Tensor | None:
        return self.take_cast(src)


def _contig_strides(shape) -> list[int]:
    st, acc = [], 1
    for d in reversed([int(x) for x in shape]):
        st.append(acc)
        acc *= max(d, 1)
    return st[::-1]


def pin_result(func, args, kwargs, arena: Arena):
    name = op_name(func)[1]
    if name == "_pin_memory":
        src = args[0]
        if not src.is_contiguous():
            return None
        return arena.take(src)
    if name == "to":
        _, src, _, _, dt, _, _, bad_layout = _parse_to(func, args, kwargs, _default_is_dev)
        if bad_layout:
            return None
        return arena.take_cast(src, dt if isinstance(dt, torch.dtype) else None)
    dtype = kwargs.get("dtype") or torch.get_default_dtype()
    if kwargs.get("layout") not in (None, torch.strided):
        return None
    dev = kwargs.get("device")
    if dev is not None and torch.device(dev).type != "cpu":
        return None
    shape = args[0]
    if name == "empty_strided" and list(args[1]) != _contig_strides(shape):
        return None
    out = arena.alloc(shape, dtype)
    if out is not None and name == "zeros":
        out.zero_()
    return out


def stage_args(func, args, kwargs, arena: Arena):
    name = op_name(func)[1]
    if name == "copy_":
        if len(args) >= 2:
            v = arena.take(args[1])
            return None if v is None else ((args[0], v, *args[2:]), kwargs)
        if "src" in kwargs:
            v = arena.take(kwargs["src"])
            if v is None:
                return None
            kw = dict(kwargs)
            kw["src"] = v
            return (args, kw)
    if name == "to":
        ov, src, _, _, dt, _, _, _ = _parse_to(func, args, kwargs, _default_is_dev)
        v = arena.take_cast(src, dt if isinstance(dt, torch.dtype) else None)
        if v is None:
            return None
        kw = dict(kwargs)
        if ov == "other" and len(args) > 2:
            kw.pop("non_blocking", None)
            return ((v, args[1], True, *args[3:]), kw)
        if ov in ("device", "prim_Device") and len(args) > 3:
            kw.pop("non_blocking", None)
            return ((v, args[1], args[2], True, *args[4:]), kw)
        kw["non_blocking"] = True
        return ((v, *args[1:]), kw)
    v = arena.take(args[0])
    return None if v is None else ((v, *args[1:]), kwargs)


# ----------------------------------------------------------------------------------------------
# Eligibility (pure)
# ----------------------------------------------------------------------------------------------
def dt_ineligible(
    *,
    cg_full: bool,
    num_ubatches: int,
    has_prefill: bool,
    pcp: bool,
    dcp_size: int,
    lora: bool,
    encoder_inputs: bool,
    encoder_decoder: bool,
    pp_single: bool,
    num_reqs: int = 1,
    max_reqs: int = 0,
) -> str | None:
    if not cg_full:
        return "not_full_graph"
    if num_ubatches > 1:
        return "ubatch"
    if has_prefill:
        return "prefill"
    if pcp:
        return "pcp"
    if dcp_size > 1:
        return "dcp"
    if lora:
        return "lora"
    if encoder_inputs or encoder_decoder:
        return "encoder"
    if not pp_single:
        return "pp"
    if max_reqs > 0 and num_reqs > max_reqs:
        return "max_reqs"
    return None


def td_ineligible(
    *,
    has_speculator: bool,
    has_full_draft_cg: bool,
    has_prefill: bool,
    pcp: bool,
    dcp_size: int,
    mm_inputs: bool,
    num_spec_tokens: int,
    context_kv_only: bool,
    conf_log: bool,
    vw_stats: bool,
    num_reqs: int = 1,
    max_reqs: int = 0,
) -> str | None:
    if not has_speculator:
        return "no_spec"
    if not has_full_draft_cg:
        return "no_draft_cg"
    if has_prefill:
        return "prefill"
    if pcp:
        return "pcp"
    if dcp_size > 1:
        return "dcp"
    if mm_inputs:
        return "mm_inputs"
    if num_spec_tokens <= 0 or context_kv_only:
        return "no_spec_tokens"
    if conf_log or vw_stats:
        return "logging_sync"
    if max_reqs > 0 and num_reqs > max_reqs:
        return "max_reqs"
    return None


# ----------------------------------------------------------------------------------------------
# Graph backend (upgraded: exec cache keyed by (region_name, shape_key, seg, slot))
# ----------------------------------------------------------------------------------------------
class TorchBackend:
    """torch.cuda.CUDAGraph per segment, private pool, thread-local capture, ExecUpdate."""

    def __init__(self, update: bool, max_execs: int = 128):
        self.update = update
        self.max_execs = max(0, int(max_execs))
        self.pool = torch.cuda.graph_pool_handle()
        self.keep: deque = deque(maxlen=16)
        self.execs: OrderedDict = OrderedDict()  # (region_name, shape_key, seg, slot) -> owner CUDAGraph
        self.events = [torch.cuda.Event() for _ in range(NSLOT)]
        self.recorded = [False] * NSLOT
        self.cudart = None
        self.update_hits = 0
        self.update_fail = 0
        self.evictions = 0
        if update:
            try:
                from cuda.bindings import runtime as cudart  # noqa: PLC0415

                self.cudart = cudart
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"INKLING_DTGRAPH update: cuda.bindings unavailable ({e}); instantiate mode"
                )
                self.update = False

    def stream(self):
        return torch.cuda.current_stream()

    def is_default(self, s) -> bool:
        return (
            s.cuda_stream == torch.cuda.default_stream(s.device).cuda_stream
            or s.cuda_stream in (0, 1, 2)
        )

    def wait_slot(self, slot: int) -> bool:
        ev = self.events[slot]
        if self.recorded[slot] and not _ORIG["Event.query"](ev):
            _ORIG["Event.synchronize"](ev)
            return True
        return False

    def begin(self):
        g = torch.cuda.CUDAGraph(keep_graph=self.update)
        g.capture_begin(pool=self.pool, capture_error_mode="thread_local")
        return g

    def fence(self, slot: int, stream) -> None:
        _ORIG["Event.record"](self.events[slot], stream)
        self.recorded[slot] = True

    def end_and_launch(self, g, key: tuple, stream) -> None:
        g.capture_end()
        self.keep.append(g)
        replay = _ORIG.get("CUDAGraph.replay", torch.cuda.CUDAGraph.replay)
        if not self.update:
            replay(g)
            return
        if self._update_launch(g, key, stream):
            self.update_hits += 1
            return
        g.instantiate()
        replay(g)
        self.execs[key] = g
        self.execs.move_to_end(key)
        if self.max_execs > 0:
            while len(self.execs) > self.max_execs:
                self.execs.popitem(last=False)
                self.evictions += 1

    def _update_launch(self, g, key: tuple, stream) -> bool:
        owner = self.execs.get(key)
        if owner is None or self.cudart is None:
            return False
        rt = self.cudart
        try:
            ex = rt.cudaGraphExec_t(init_value=owner.raw_cuda_graph_exec())
            gr = rt.cudaGraph_t(init_value=g.raw_cuda_graph())
            res = rt.cudaGraphExecUpdate(ex, gr)
            err = res[0] if isinstance(res, tuple) else res
            if err != rt.cudaError_t.cudaSuccess:
                self.update_fail += 1
                return False
            res = rt.cudaGraphLaunch(ex, rt.cudaStream_t(init_value=stream.cuda_stream))
            err = res[0] if isinstance(res, tuple) else res
            if err != rt.cudaError_t.cudaSuccess:
                self.update_fail += 1
                return False
            self.execs.move_to_end(key)
            return True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"INKLING_DTGRAPH update path disabled ({type(e).__name__}: {e})")
            self.cudart = None
            return False


# ----------------------------------------------------------------------------------------------
# Region state machine
# ----------------------------------------------------------------------------------------------
_TLS = threading.local()
_ORIG: dict = {}


class Region:
    """Capture region state machine shared by DT (Phase 4) and TD (Phase 2B)."""

    def __init__(
        self, backend, *, max_seg: int = 4, arena: Arena | None = None, is_dev=_default_is_dev
    ):
        self.b = backend
        self.max_seg = max_seg
        self.arena = arena if arena is not None else Arena()
        self.is_dev = is_dev
        self.active = False
        self.capturing = False
        self.region_name = ""
        self.shape_key: tuple = ()
        self.g = None
        self.seg = 0
        self.latch = False
        self.want = False
        self.slot = 0
        self.stream = None
        self.n = 0
        self.stats: Counter = Counter()

    def begin(self, stream, region_name: str = "dt", shape_key: tuple = ()) -> bool:
        if self.active:
            raise RuntimeError(
                f"INKLING_DTGRAPH: region {self.region_name!r} still open when starting "
                f"{region_name!r}; refusing to continue"
            )
        self.slot = self.n % NSLOT
        if self.b.wait_slot(self.slot):
            self.stats["slot_wait"] += 1
        self.arena.start(self.slot)
        self.stream = stream
        self.region_name = region_name
        self.shape_key = shape_key
        self.seg, self.latch, self.want = 0, False, False
        self.g = self.b.begin()
        self.capturing = True
        self.active = True
        _TLS.region = self
        self.stats[f"regions:{region_name}"] += 1
        return True

    def end(self) -> None:
        if not self.active:
            return
        try:
            if self.capturing:
                self._close()
            self.b.fence(self.slot, self.stream)
        finally:
            self.active = False
            self.capturing = False
            self.want = False
            self.latch = False
            self.region_name = ""
            _TLS.region = None
            self.n += 1

    def abort(self) -> None:
        if self.capturing:
            try:
                self.g.capture_end()
            except Exception:  # noqa: BLE001
                pass
        self.active = False
        self.capturing = False
        self.want = False
        self.latch = False
        self.region_name = ""
        _TLS.region = None

    def _close(self) -> None:
        g, self.g = self.g, None
        self.capturing = False
        key = (self.region_name, self.shape_key, self.seg, self.slot)
        self.b.end_and_launch(g, key, self.stream)
        self.stats[f"segments:{self.region_name}"] += 1
        self.seg += 1

    def hazard(self, why: str, latch: bool = False) -> None:
        self.stats[f"haz:{self.region_name}:{why}"] += 1
        if latch:
            self.latch = True
        if self.capturing:
            self._close()
        self.want = not self.latch

    def maybe_resume(self) -> None:
        if (
            self.want
            and not self.latch
            and not self.capturing
            and self.seg < self.max_seg
            and self.b.stream().cuda_stream == self.stream.cuda_stream
        ):
            self.want = False
            self.g = self.b.begin()
            self.capturing = True

    def dispatch(self, func, args, kwargs):
        if not self.active:
            return func(*args, **kwargs)
        if not self.capturing:
            if self.want:
                self.maybe_resume()
            if not self.capturing:
                return func(*args, **kwargs)
        # Elide identical-buffer self-copy (e.g. inputs_embeds[:n].copy_(embeds) after direct embed write).
        if op_name(func)[1] == "copy_" and len(args) >= 2:
            dst, src = args[0], args[1]
            if (
                isinstance(dst, torch.Tensor)
                and isinstance(src, torch.Tensor)
                and dst.device == src.device
                and dst.dtype == src.dtype
                and dst.shape == src.shape
                and dst.stride() == src.stride()
                and dst.data_ptr() == src.data_ptr()
            ):
                self.stats["elided_self_copy"] += 1
                return dst
        act, why = classify(func, args, kwargs, self.is_dev, self.arena.owns)
        if act in (PASS, PASS_CPU):
            return func(*args, **kwargs)
        if act == PIN:
            out = pin_result(func, args, kwargs, self.arena)
            if out is not None:
                self.stats["pinned"] += 1
                return out
            why = "arena_full"
        if act == STAGE:
            st = stage_args(func, args, kwargs, self.arena)
            if st is not None:
                self.stats["staged"] += 1
                return func(*st[0], **st[1])
            why = "arena_full"
        self.hazard(why)
        return func(*args, **kwargs)


def current_region():
    return getattr(_TLS, "region", None)


# ----------------------------------------------------------------------------------------------
# Python-level hazard hooks (installed once, only when INKLING_DTGRAPH is on)
# ----------------------------------------------------------------------------------------------
def _hz(orig, why, latch):
    def w(*a, **k):
        r = getattr(_TLS, "region", None)
        if r is not None:
            r.hazard(why, latch=latch)
        return orig(*a, **k)

    w.__wrapped__ = orig
    w._inkling_dtg = True
    return w


def _install_hooks() -> None:
    if _ORIG.get("_installed"):
        return
    E, S = torch.cuda.Event, torch.cuda.Stream
    for cls, cname, names in (
        (E, "Event", ("record", "wait", "synchronize", "query")),
        (S, "Stream", ("wait_stream", "wait_event", "record_event", "synchronize", "query")),
    ):
        for n in names:
            orig = getattr(cls, n)
            _ORIG[f"{cname}.{n}"] = orig
            setattr(cls, n, _hz(orig, f"{cname.lower()}:{n}", True))
    _ORIG["synchronize"] = torch.cuda.synchronize
    torch.cuda.synchronize = _hz(torch.cuda.synchronize, "sync:device", True)
    orig_set = torch.cuda.set_stream
    _ORIG["set_stream"] = orig_set

    def set_stream(stream):
        r = getattr(_TLS, "region", None)
        if r is not None and stream is not None and stream.cuda_stream != r.stream.cuda_stream:
            r.hazard("stream:set", latch=True)
        return orig_set(stream)

    set_stream._inkling_dtg = True
    torch.cuda.set_stream = set_stream
    _ORIG["CUDAGraph.replay"] = torch.cuda.CUDAGraph.replay
    torch.cuda.CUDAGraph.replay = _hz(torch.cuda.CUDAGraph.replay, "graph_replay", True)
    try:
        from vllm.distributed.parallel_state import GroupCoordinator  # noqa: PLC0415

        for n in (
            "all_reduce", "all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv",
            "reduce_scatterv_into_output", "gather", "broadcast", "broadcast_tensor_dict",
            "send", "recv", "send_tensor_dict", "recv_tensor_dict", "barrier",
        ):
            if hasattr(GroupCoordinator, n):
                setattr(GroupCoordinator, n, _hz(getattr(GroupCoordinator, n), "collective:" + n, False))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"INKLING_DTGRAPH: GroupCoordinator hooks not installed ({e})")
    try:
        from triton import knobs  # noqa: PLC0415

        prev = knobs.runtime.jit_cache_hook

        def cache_hook(*a, **k):
            r = getattr(_TLS, "region", None)
            if r is not None:
                r.hazard("triton_compile", latch=False)
            return prev(*a, **k) if prev is not None else None

        knobs.runtime.jit_cache_hook = cache_hook

        def load_hook(*a, **k):
            r = getattr(_TLS, "region", None)
            if r is not None:
                r.hazard("triton_load", latch=False)

        hc = knobs.runtime.kernel_load_start_hook
        if hasattr(hc, "add"):
            hc.add(load_hook)
        elif hc is None:
            knobs.runtime.kernel_load_start_hook = load_hook
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"INKLING_DTGRAPH: Triton hooks not installed ({e})")
    warnings.filterwarnings("ignore", message=".*The CUDA Graph is empty.*")
    _ORIG["_installed"] = True


def _hook_rdma() -> None:
    if _ORIG.get("_rdma"):
        return
    import sys  # noqa: PLC0415

    m = sys.modules.get("vllm.distributed.device_communicators.rdma_all_reduce")
    if m is None:
        return
    cls = getattr(m, "RdmaAllReduce", None) or next(
        (v for v in vars(m).values() if isinstance(v, type) and hasattr(v, "all_gather_stacked")),
        None,
    )
    if cls is not None:
        for n in (
            "all_reduce",
            "all_reduce_add_rmsnorm",
            "all_gather_stacked",
            "all_gather_stacked_cond",
            "all_gather_cols",
        ):
            if hasattr(cls, n):
                setattr(cls, n, _hz(getattr(cls, n), "collective:rdma_" + n, False))
    _ORIG["_rdma"] = True


# ----------------------------------------------------------------------------------------------
# Lazy GPUModelRunner + CUDAGraphManager integration (ZERO edits to model_runner.py!)
# ----------------------------------------------------------------------------------------------
class _Mode:
    cls = None

    @classmethod
    def make(cls, region):
        if cls.cls is None:
            from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC0415

            class _DTGMode(TorchDispatchMode):
                def __init__(self, r):
                    super().__init__()
                    self.r = r

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    return self.r.dispatch(func, args, kwargs or {})

            cls.cls = _DTGMode
        return cls.cls(region)


class _RunnerState:
    def __init__(self):
        self.region: Region | None = None
        self.mode = None
        self.seen: Counter = Counter()
        self.skips: Counter = Counter()
        self.disabled = False
        self.rank0 = True
        self.cur_sample_ctx: dict | None = None
        # Lever A (INKLING_FUSED_DT_PREP=1): self-probed dtypes + persistent B=1 / B=2..N GPU buffers
        self.b1_dtypes: tuple | None = None
        self.b1_bufs: dict[str, torch.Tensor] | None = None
        self.b1_compiled: set[tuple] = set()
        self.bn_bufs: dict[str, torch.Tensor] | None = None
        self.bn_compiled: set[tuple] = set()
        self.b1_failed = False
        self.b1_ib_cls = None
        self.b1_ib_fields: set[str] | None = None
        # Lever B (INKLING_FUSED_EMBED_DIRECT=1): discovered static inputs_embeds buffer
        self.inputs_embeds_buf: torch.Tensor | None = None
        # Lever C (INKLING_FUSED_DRAFT_STORE=1): real req_states.draft_tokens saved across propose->set_draft_tokens
        self.draft_store_real: torch.Tensor | None = None
        self.draft_store_active = False
        try:
            from vllm.distributed import get_tensor_model_parallel_rank  # noqa: PLC0415

            self.rank0 = get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------
# Lever A: Fused 1-CTA Triton B=1 pure-decode prepare_inputs (zero H2D DMA, 0-ULP bit-exact)
# ----------------------------------------------------------------------------------------------
_FUSED_DT_PREP_KERNEL = None


def _get_fused_dt_prep_kernel():
    global _FUSED_DT_PREP_KERNEL
    if _FUSED_DT_PREP_KERNEL is not None:
        return _FUSED_DT_PREP_KERNEL
    from vllm.triton_utils import tl, triton  # noqa: PLC0415

    @triton.jit(
        do_not_specialize=[
            "slot0",
            "num_tokens",
            "num_tokens_padded",
            "qsl_len",
            "sl_len",
            "n_logits",
            "draft_stride",
            "last_sampled_stride",
        ]
    )
    def _fused_dt_prep_b1_kernel(
        is_padding_ptr,
        idx_mapping_ptr,
        cu_num_logits_ptr,
        query_start_loc_ptr,
        expanded_idx_ptr,
        expanded_local_pos_ptr,
        positions_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        logits_indices_ptr,
        num_computed_ptr,
        last_sampled_ptr,
        draft_tokens_ptr,
        slot0,
        num_tokens,
        num_tokens_padded,
        qsl_len,
        sl_len,
        n_logits,
        draft_stride,
        last_sampled_stride,
        WRITE_IS_PADDING: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """1-CTA fused replacement for is_padding.fill_ + 3x async_tensor_h2d +
        expand_idx_mapping + prepare_pos_seq_lens + combine_sampled_and_draft_tokens
        on B=1 pure-decode steps (zero DMA HtoD transitions, 100% bit-exact)."""
        s0 = slot0.to(tl.int64)
        nt = num_tokens.to(tl.int32)
        num_computed = tl.load(num_computed_ptr + s0).to(tl.int64)
        last_tok = tl.load(last_sampled_ptr + s0 * last_sampled_stride.to(tl.int64)).to(tl.int64)

        tl.store(idx_mapping_ptr, s0.to(idx_mapping_ptr.dtype.element_ty))
        tl.store(cu_num_logits_ptr, tl.zeros((), dtype=cu_num_logits_ptr.dtype.element_ty))
        tl.store(cu_num_logits_ptr + 1, n_logits.to(cu_num_logits_ptr.dtype.element_ty))

        seq_len0 = (num_computed + nt).to(seq_lens_ptr.dtype.element_ty)
        for off in range(0, qsl_len, BLOCK_R):
            r = off + tl.arange(0, BLOCK_R)
            tl.store(
                query_start_loc_ptr + r,
                tl.where(r == 0, 0, nt).to(query_start_loc_ptr.dtype.element_ty),
                mask=r < qsl_len,
            )
        for off in range(0, sl_len, BLOCK_R):
            r = off + tl.arange(0, BLOCK_R)
            tl.store(
                seq_lens_ptr + r,
                tl.where(r == 0, seq_len0, 0).to(seq_lens_ptr.dtype.element_ty),
                mask=r < sl_len,
            )

        t = tl.arange(0, BLOCK_T)
        if WRITE_IS_PADDING:
            tl.store(is_padding_ptr + t, t >= num_tokens, mask=t < num_tokens_padded)
        m_tok = t < num_tokens
        tl.store(
            positions_ptr + t,
            (num_computed + t).to(positions_ptr.dtype.element_ty),
            mask=m_tok,
        )
        tl.store(
            expanded_idx_ptr + t,
            (tl.zeros((BLOCK_T,), dtype=tl.int64) + s0).to(expanded_idx_ptr.dtype.element_ty),
            mask=m_tok,
        )
        tl.store(
            expanded_local_pos_ptr + t,
            t.to(expanded_local_pos_ptr.dtype.element_ty),
            mask=m_tok,
        )
        tl.store(
            logits_indices_ptr + t,
            t.to(logits_indices_ptr.dtype.element_ty),
            mask=m_tok,
        )
        d_idx = tl.maximum(t - 1, 0).to(tl.int64)
        draft_tok = tl.load(
            draft_tokens_ptr + s0 * draft_stride.to(tl.int64) + d_idx,
            mask=(t >= 1) & m_tok,
            other=0,
        ).to(tl.int64)
        inp_tok = tl.where(t == 0, last_tok, draft_tok)
        tl.store(
            input_ids_ptr + t,
            inp_tok.to(input_ids_ptr.dtype.element_ty),
            mask=m_tok,
        )

    _FUSED_DT_PREP_KERNEL = (triton, _fused_dt_prep_b1_kernel)
    return _FUSED_DT_PREP_KERNEL


def _b1_prep_eligible(runner, scheduler_output, batch_req_state, batch_desc) -> bool:
    if not FUSED_DT_PREP:
        return False
    if len(batch_req_state.req_ids) != 1 or bool(batch_req_state.has_prefill):
        return False
    if not getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        return False
    if getattr(runner.model_state, "num_new_sampled_tokens_per_step", 0) != 1:
        return False
    if getattr(runner, "adaptive_verification", None) is not None:
        return False
    if getattr(runner, "fast_prefill", None) is not None:
        return False
    if getattr(runner.model_config, "rswa_window", None) is not None:
        return False
    if getattr(runner, "pcp_manager", None) is not None:
        return False
    num_tokens = int(batch_req_state.num_tokens)
    if num_tokens <= 0:
        return False
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if num_tokens_after_padding > 256:
        return False
    nst_np = batch_req_state.num_scheduled_tokens
    if int(nst_np[0]) != num_tokens:
        return False
    nd_np = batch_req_state.num_draft_tokens_np
    if nd_np is None or int(nd_np[0]) + 1 != num_tokens:
        return False
    # Guarantee decode past prefill boundary (so combine_sampled_and_draft_tokens always reads last_sampled + drafts).
    if int(batch_req_state.num_computed_prefill_tokens_np[0]) < int(batch_req_state.prefill_len_np[0]):
        return False
    return getattr(runner, "device", None) is not None and runner.device.type == "cuda"


def _ensure_b1_bufs(st: _RunnerState, device: torch.device, min_tok: int) -> bool:
    if st.b1_dtypes is None:
        return False
    need = max(64, int(min_tok))
    if st.b1_bufs is not None and st.b1_bufs["expanded_idx"].shape[0] >= need:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    dt_idx, dt_cu, dt_eidx, dt_epos, dt_li = st.b1_dtypes
    st.b1_bufs = {
        "idx": torch.empty((1,), dtype=dt_idx, device=device),
        "cu": torch.empty((2,), dtype=dt_cu, device=device),
        "expanded_idx": torch.empty((need,), dtype=dt_eidx, device=device),
        "expanded_pos": torch.empty((need,), dtype=dt_epos, device=device),
        "logits_indices": torch.empty((need,), dtype=dt_li, device=device),
    }
    return True


def _probe_b1_from_ib(st: _RunnerState, ib) -> None:
    if st.b1_dtypes is not None or ib is None:
        return
    try:
        cls = type(ib)
        if not dataclasses.is_dataclass(cls):
            st.b1_failed = True
            return
        fields = {f.name for f in dataclasses.fields(cls)}
        required = {
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }
        known = {
            "req_ids", "num_reqs", "num_reqs_after_padding", "idx_mapping", "idx_mapping_np",
            "expanded_idx_mapping", "expanded_local_pos", "num_scheduled_tokens", "num_tokens",
            "num_tokens_after_padding", "num_draft_tokens", "num_draft_tokens_per_req",
            "num_admitted_draft_tokens", "query_start_loc", "query_start_loc_np", "seq_lens",
            "seq_lens_cpu_upper_bound", "dcp_local_seq_lens", "num_computed_tokens_np",
            "prefill_len_np", "num_computed_prefill_tokens_np", "is_prefilling_np", "has_prefill",
            "prefill_runs_as_decode_np", "decode_graph_eligible", "max_seq_len_np", "input_ids",
            "positions", "is_padding", "logits_indices", "cu_num_logits", "cu_num_logits_np",
            "has_structured_output_reqs", "prompt_lens", "fast_prefill", "max_query_len",
        }
        if not required.issubset(known):
            st.b1_failed = True
            return
        st.b1_ib_cls = cls
        st.b1_ib_fields = fields
        st.b1_dtypes = (
            ib.idx_mapping.dtype,
            ib.cu_num_logits.dtype,
            ib.expanded_idx_mapping.dtype,
            ib.expanded_local_pos.dtype,
            ib.logits_indices.dtype,
        )
    except Exception:  # noqa: BLE001
        st.b1_failed = True


def _try_fused_prepare_inputs_b1(
    runner, st: _RunnerState, scheduler_output, batch_req_state, batch_desc
):
    if st.b1_failed or st.b1_dtypes is None or st.b1_ib_cls is None or st.b1_ib_fields is None:
        return None
    if not _b1_prep_eligible(runner, scheduler_output, batch_req_state, batch_desc):
        return None
    num_tokens = int(batch_req_state.num_tokens)
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if not _ensure_b1_bufs(st, runner.device, num_tokens_after_padding):
        return None
    from vllm import envs  # noqa: PLC0415

    write_pad = bool(envs.VLLM_MOE_SKIP_PADDING)
    triton, kernel = _get_fused_dt_prep_kernel()
    block_r = 256
    block_t = max(16, triton.next_power_of_2(num_tokens_after_padding))
    ckey = (write_pad, block_r, block_t)
    if ckey not in st.b1_compiled and torch.cuda.is_current_stream_capturing():
        return None

    req_ids = batch_req_state.req_ids
    num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
    idx_mapping_np = batch_req_state.idx_mapping_np
    slot0 = int(idx_mapping_np[0])
    num_draft_tokens_per_req = batch_req_state.num_draft_tokens_np
    total_num_draft_tokens = int(num_draft_tokens_per_req[0])
    total_num_logits = total_num_draft_tokens + 1
    num_reqs_padded = int(batch_desc.num_reqs or 1)

    bufs = st.b1_bufs
    assert bufs is not None
    idx_mapping = bufs["idx"]
    cu_num_logits = bufs["cu"]
    expanded_idx_mapping = bufs["expanded_idx"][:total_num_logits]
    expanded_local_pos = bufs["expanded_pos"][:total_num_logits]
    logits_indices = bufs["logits_indices"][:total_num_logits]

    qsl_full = runner.input_buffers.query_start_loc
    sl_full = runner.input_buffers.seq_lens
    is_padding = runner.input_buffers.is_padding[:num_tokens_after_padding]
    last_sampled = runner.req_states.last_sampled_tokens
    draft_tokens_buf = runner.req_states.draft_tokens
    if (
        not isinstance(draft_tokens_buf, torch.Tensor)
        or draft_tokens_buf.dim() != 2
        or draft_tokens_buf.stride(1) != 1
    ):
        return None

    kernel[(1,)](
        is_padding,
        idx_mapping,
        cu_num_logits,
        qsl_full,
        expanded_idx_mapping,
        expanded_local_pos,
        runner.input_buffers.positions,
        sl_full,
        runner.input_buffers.input_ids,
        logits_indices,
        runner.req_states.num_computed_tokens.gpu,
        last_sampled,
        draft_tokens_buf,
        slot0,
        num_tokens,
        num_tokens_after_padding,
        int(qsl_full.shape[0]),
        int(sl_full.shape[0]),
        total_num_logits,
        int(draft_tokens_buf.stride(0)),
        int(last_sampled.stride(0)),
        WRITE_IS_PADDING=write_pad,
        BLOCK_R=block_r,
        BLOCK_T=block_t,
        num_warps=4,
    )
    st.b1_compiled.add(ckey)

    cu_num_logits_np = np.array([0, total_num_logits], dtype=np.int32)
    query_start_loc_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
    query_start_loc_np[0] = 0
    query_start_loc = qsl_full[: num_reqs_padded + 1]
    seq_lens = sl_full[:num_reqs_padded]

    num_computed_tokens_np = runner.req_states.num_computed_tokens_np[idx_mapping_np]
    seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
    seq_lens_cpu_upper_bound_np[0] = int(num_computed_tokens_np[0]) + num_tokens
    seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

    all_kwargs = {
        "req_ids": req_ids,
        "num_reqs": 1,
        "num_reqs_after_padding": num_reqs_padded,
        "idx_mapping": idx_mapping,
        "idx_mapping_np": idx_mapping_np,
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
        "num_scheduled_tokens": num_scheduled_tokens_np,
        "num_tokens": num_tokens,
        "num_tokens_after_padding": num_tokens_after_padding,
        "num_draft_tokens": total_num_draft_tokens,
        "num_draft_tokens_per_req": num_draft_tokens_per_req,
        "num_admitted_draft_tokens": None,
        "query_start_loc": query_start_loc,
        "query_start_loc_np": query_start_loc_np,
        "seq_lens": seq_lens,
        "seq_lens_cpu_upper_bound": seq_lens_cpu_upper_bound,
        "dcp_local_seq_lens": None,
        "num_computed_tokens_np": num_computed_tokens_np,
        "prefill_len_np": batch_req_state.prefill_len_np,
        "num_computed_prefill_tokens_np": batch_req_state.num_computed_prefill_tokens_np,
        "is_prefilling_np": batch_req_state.is_prefilling_np,
        "has_prefill": False,
        "prefill_runs_as_decode_np": batch_req_state.prefill_runs_as_decode_np,
        "decode_graph_eligible": batch_req_state.decode_graph_eligible,
        "max_seq_len_np": batch_req_state.max_seq_len_np,
        "input_ids": runner.input_buffers.input_ids[:num_tokens_after_padding],
        "positions": runner.input_buffers.positions[:num_tokens_after_padding],
        "is_padding": is_padding,
        "logits_indices": logits_indices,
        "cu_num_logits": cu_num_logits,
        "cu_num_logits_np": cu_num_logits_np,
        "has_structured_output_reqs": scheduler_output.has_structured_output_requests,
        "prompt_lens": None,
        "fast_prefill": None,
        "max_query_len": None,
    }
    filtered = {k: v for k, v in all_kwargs.items() if k in st.b1_ib_fields}
    if st.region is not None:
        st.region.stats["b1_fused_prep"] += 1
    return st.b1_ib_cls(**filtered)


# ----------------------------------------------------------------------------------------------
# Lever A2 (opt-in C=32): Fused B=2..32 uniform pure-decode prepare_inputs
# (1 tiny B*8 B H2D of idx_mapping + 1 (B+1)-CTA Triton kernel replacing 2 H2Ds + 6 kernels!)
# ----------------------------------------------------------------------------------------------
_FUSED_DT_PREP_BN_KERNEL = None


def _get_fused_dt_prep_bn_kernel():
    global _FUSED_DT_PREP_BN_KERNEL
    if _FUSED_DT_PREP_BN_KERNEL is not None:
        return _FUSED_DT_PREP_BN_KERNEL
    from vllm.triton_utils import tl, triton  # noqa: PLC0415

    @triton.jit(
        do_not_specialize=[
            "num_reqs",
            "q_len",
            "num_tokens",
            "num_tokens_padded",
            "qsl_len",
            "sl_len",
            "draft_stride",
            "last_sampled_stride",
        ]
    )
    def _fused_dt_prep_bn_kernel(
        is_padding_ptr,
        idx_mapping_ptr,
        cu_num_logits_ptr,
        query_start_loc_ptr,
        expanded_idx_ptr,
        expanded_local_pos_ptr,
        positions_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        logits_indices_ptr,
        num_computed_ptr,
        last_sampled_ptr,
        draft_tokens_ptr,
        num_reqs,
        q_len,
        num_tokens,
        num_tokens_padded,
        qsl_len,
        sl_len,
        draft_stride,
        last_sampled_stride,
        WRITE_IS_PADDING: tl.constexpr,
        HAS_DRAFTS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """(B+1)-CTA fused replacement for 2x is_padding.fill_ + 2x H2D + expand_idx_mapping +
        prepare_pos_seq_lens + combine_sampled_and_draft_tokens on uniform pure-decode B=2..32."""
        pid = tl.program_id(0)
        if pid < num_reqs:
            r = pid.to(tl.int64)
            slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
            num_computed = tl.load(num_computed_ptr + slot).to(tl.int64)
            last_tok = tl.load(
                last_sampled_ptr + slot * last_sampled_stride.to(tl.int64)
            ).to(tl.int64)
            ql = q_len.to(tl.int32)
            base = (pid * ql).to(tl.int32)

            tl.store(
                query_start_loc_ptr + r,
                base.to(query_start_loc_ptr.dtype.element_ty),
            )
            tl.store(
                cu_num_logits_ptr + r,
                base.to(cu_num_logits_ptr.dtype.element_ty),
            )
            tl.store(
                seq_lens_ptr + r,
                (num_computed + ql).to(seq_lens_ptr.dtype.element_ty),
            )

            q = tl.arange(0, BLOCK_Q)
            mq = q < ql
            pos_out = base + q
            tl.store(
                positions_ptr + pos_out,
                (num_computed + q).to(positions_ptr.dtype.element_ty),
                mask=mq,
            )
            tl.store(
                expanded_idx_ptr + pos_out,
                (tl.zeros((BLOCK_Q,), dtype=tl.int64) + slot).to(
                    expanded_idx_ptr.dtype.element_ty
                ),
                mask=mq,
            )
            tl.store(
                expanded_local_pos_ptr + pos_out,
                q.to(expanded_local_pos_ptr.dtype.element_ty),
                mask=mq,
            )
            tl.store(
                logits_indices_ptr + pos_out,
                pos_out.to(logits_indices_ptr.dtype.element_ty),
                mask=mq,
            )
            if HAS_DRAFTS:
                d_idx = tl.maximum(q - 1, 0).to(tl.int64)
                draft_tok = tl.load(
                    draft_tokens_ptr + slot * draft_stride.to(tl.int64) + d_idx,
                    mask=(q >= 1) & mq,
                    other=0,
                ).to(tl.int64)
                inp_tok = tl.where(q == 0, last_tok, draft_tok)
            else:
                inp_tok = tl.zeros((BLOCK_Q,), dtype=tl.int64) + last_tok
            tl.store(
                input_ids_ptr + pos_out,
                inp_tok.to(input_ids_ptr.dtype.element_ty),
                mask=mq,
            )
        else:
            nt = num_tokens.to(tl.int32)
            tl.store(
                cu_num_logits_ptr + num_reqs,
                nt.to(cu_num_logits_ptr.dtype.element_ty),
            )
            for off in range(0, qsl_len, BLOCK_R):
                rr = off + tl.arange(0, BLOCK_R)
                tl.store(
                    query_start_loc_ptr + rr,
                    nt.to(query_start_loc_ptr.dtype.element_ty),
                    mask=(rr >= num_reqs) & (rr < qsl_len),
                )
            for off in range(0, sl_len, BLOCK_R):
                rr = off + tl.arange(0, BLOCK_R)
                tl.store(
                    seq_lens_ptr + rr,
                    tl.zeros((BLOCK_R,), dtype=seq_lens_ptr.dtype.element_ty),
                    mask=(rr >= num_reqs) & (rr < sl_len),
                )
            if WRITE_IS_PADDING:
                for off in range(0, num_tokens_padded, BLOCK_T):
                    t = off + tl.arange(0, BLOCK_T)
                    tl.store(
                        is_padding_ptr + t,
                        t >= num_tokens,
                        mask=t < num_tokens_padded,
                    )

    _FUSED_DT_PREP_BN_KERNEL = (triton, _fused_dt_prep_bn_kernel)
    return _FUSED_DT_PREP_BN_KERNEL


def _bn_prep_eligible(runner, scheduler_output, batch_req_state, batch_desc) -> bool:
    if not FUSED_DT_PREP or FUSED_DT_PREP_MAX_B < 2:
        return False
    num_reqs = len(batch_req_state.req_ids)
    if num_reqs < 2 or num_reqs > FUSED_DT_PREP_MAX_B or bool(batch_req_state.has_prefill):
        return False
    if getattr(runner.model_state, "num_new_sampled_tokens_per_step", 0) != 1:
        return False
    if getattr(runner, "adaptive_verification", None) is not None:
        return False
    if getattr(runner, "fast_prefill", None) is not None:
        return False
    if getattr(runner.model_config, "rswa_window", None) is not None:
        return False
    if getattr(runner, "pcp_manager", None) is not None:
        return False
    num_tokens = int(batch_req_state.num_tokens)
    if num_tokens <= 0 or (num_tokens % num_reqs) != 0:
        return False
    q_len = num_tokens // num_reqs
    if q_len < 1 or q_len > 16:
        return False
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if num_tokens_after_padding > 512:
        return False
    nst_np = batch_req_state.num_scheduled_tokens
    if not bool(np.all(nst_np == q_len)):
        return False
    has_drafts = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
    if has_drafts:
        nd_np = batch_req_state.num_draft_tokens_np
        if nd_np is None or not bool(np.all(nd_np == (q_len - 1))):
            return False
    elif q_len != 1:
        return False
    if not bool(
        np.all(
            batch_req_state.num_computed_prefill_tokens_np
            >= batch_req_state.prefill_len_np
        )
    ):
        return False
    return getattr(runner, "device", None) is not None and runner.device.type == "cuda"


def _ensure_bn_bufs(st: _RunnerState, device: torch.device, min_reqs: int, min_tok: int) -> bool:
    if st.b1_dtypes is None:
        return False
    need_r = max(64, int(min_reqs))
    need_t = max(256, int(min_tok))
    if (
        st.bn_bufs is not None
        and st.bn_bufs["idx"].shape[0] >= need_r
        and st.bn_bufs["expanded_idx"].shape[0] >= need_t
    ):
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    dt_idx, dt_cu, dt_eidx, dt_epos, dt_li = st.b1_dtypes
    st.bn_bufs = {
        "idx": torch.empty((need_r,), dtype=dt_idx, device=device),
        "cu": torch.empty((need_r + 1,), dtype=dt_cu, device=device),
        "expanded_idx": torch.empty((need_t,), dtype=dt_eidx, device=device),
        "expanded_pos": torch.empty((need_t,), dtype=dt_epos, device=device),
        "logits_indices": torch.empty((need_t,), dtype=dt_li, device=device),
    }
    return True


def _try_fused_prepare_inputs_bn(
    runner, st: _RunnerState, scheduler_output, batch_req_state, batch_desc
):
    if st.b1_failed or st.b1_dtypes is None or st.b1_ib_cls is None or st.b1_ib_fields is None:
        return None
    if not _bn_prep_eligible(runner, scheduler_output, batch_req_state, batch_desc):
        return None
    req_ids = batch_req_state.req_ids
    num_reqs = len(req_ids)
    num_tokens = int(batch_req_state.num_tokens)
    q_len = num_tokens // num_reqs
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if not _ensure_bn_bufs(st, runner.device, num_reqs, num_tokens_after_padding):
        return None
    from vllm import envs  # noqa: PLC0415

    write_pad = bool(envs.VLLM_MOE_SKIP_PADDING)
    has_drafts = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
    triton, kernel = _get_fused_dt_prep_bn_kernel()
    block_r = 256
    block_q = max(16, triton.next_power_of_2(q_len))
    block_t = 256
    ckey = (write_pad, has_drafts, block_r, block_q, block_t)
    if ckey not in st.bn_compiled and torch.cuda.is_current_stream_capturing():
        return None

    bufs = st.bn_bufs
    assert bufs is not None
    idx_mapping_np = batch_req_state.idx_mapping_np
    idx_mapping = bufs["idx"][:num_reqs]
    if idx_mapping.dtype == torch.int64:
        idx_cpu = torch.from_numpy(np.ascontiguousarray(idx_mapping_np, dtype=np.int64))
    else:
        idx_cpu = torch.from_numpy(np.ascontiguousarray(idx_mapping_np)).to(dtype=idx_mapping.dtype)
    idx_mapping.copy_(idx_cpu, non_blocking=True)

    cu_num_logits = bufs["cu"][: num_reqs + 1]
    expanded_idx_mapping = bufs["expanded_idx"][:num_tokens]
    expanded_local_pos = bufs["expanded_pos"][:num_tokens]
    logits_indices = bufs["logits_indices"][:num_tokens]

    qsl_full = runner.input_buffers.query_start_loc
    sl_full = runner.input_buffers.seq_lens
    is_padding = runner.input_buffers.is_padding[:num_tokens_after_padding]
    last_sampled = runner.req_states.last_sampled_tokens
    draft_tokens_buf = getattr(runner.req_states, "draft_tokens", None)
    if has_drafts:
        if (
            not isinstance(draft_tokens_buf, torch.Tensor)
            or draft_tokens_buf.dim() != 2
            or draft_tokens_buf.stride(1) != 1
        ):
            return None
        draft_ptr = draft_tokens_buf
        draft_stride = int(draft_tokens_buf.stride(0))
    else:
        draft_ptr = last_sampled
        draft_stride = int(last_sampled.stride(0))

    kernel[(num_reqs + 1,)](
        is_padding,
        idx_mapping,
        cu_num_logits,
        qsl_full,
        expanded_idx_mapping,
        expanded_local_pos,
        runner.input_buffers.positions,
        sl_full,
        runner.input_buffers.input_ids,
        logits_indices,
        runner.req_states.num_computed_tokens.gpu,
        last_sampled,
        draft_ptr,
        num_reqs,
        q_len,
        num_tokens,
        num_tokens_after_padding,
        int(qsl_full.shape[0]),
        int(sl_full.shape[0]),
        draft_stride,
        int(last_sampled.stride(0)),
        WRITE_IS_PADDING=write_pad,
        HAS_DRAFTS=has_drafts,
        BLOCK_R=block_r,
        BLOCK_Q=block_q,
        BLOCK_T=block_t,
        num_warps=4,
    )
    st.bn_compiled.add(ckey)

    num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
    num_reqs_padded = int(batch_desc.num_reqs or num_reqs)
    cu_num_logits_np = (np.arange(num_reqs + 1, dtype=np.int32) * q_len)
    query_start_loc_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
    query_start_loc_np[: num_reqs + 1] = cu_num_logits_np
    query_start_loc = qsl_full[: num_reqs_padded + 1]
    seq_lens = sl_full[:num_reqs_padded]

    num_computed_tokens_np = runner.req_states.num_computed_tokens_np[idx_mapping_np]
    seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
    np.add(
        num_computed_tokens_np,
        num_scheduled_tokens_np,
        out=seq_lens_cpu_upper_bound_np[:num_reqs],
    )
    seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

    num_draft_tokens_per_req = batch_req_state.num_draft_tokens_np if has_drafts else None
    total_num_draft_tokens = num_reqs * (q_len - 1) if has_drafts else 0

    all_kwargs = {
        "req_ids": req_ids,
        "num_reqs": num_reqs,
        "num_reqs_after_padding": num_reqs_padded,
        "idx_mapping": idx_mapping,
        "idx_mapping_np": idx_mapping_np,
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
        "num_scheduled_tokens": num_scheduled_tokens_np,
        "num_tokens": num_tokens,
        "num_tokens_after_padding": num_tokens_after_padding,
        "num_draft_tokens": total_num_draft_tokens,
        "num_draft_tokens_per_req": num_draft_tokens_per_req,
        "num_admitted_draft_tokens": None,
        "query_start_loc": query_start_loc,
        "query_start_loc_np": query_start_loc_np,
        "seq_lens": seq_lens,
        "seq_lens_cpu_upper_bound": seq_lens_cpu_upper_bound,
        "dcp_local_seq_lens": None,
        "num_computed_tokens_np": num_computed_tokens_np,
        "prefill_len_np": batch_req_state.prefill_len_np,
        "num_computed_prefill_tokens_np": batch_req_state.num_computed_prefill_tokens_np,
        "is_prefilling_np": batch_req_state.is_prefilling_np,
        "has_prefill": False,
        "prefill_runs_as_decode_np": batch_req_state.prefill_runs_as_decode_np,
        "decode_graph_eligible": batch_req_state.decode_graph_eligible,
        "max_seq_len_np": batch_req_state.max_seq_len_np,
        "input_ids": runner.input_buffers.input_ids[:num_tokens_after_padding],
        "positions": runner.input_buffers.positions[:num_tokens_after_padding],
        "is_padding": is_padding,
        "logits_indices": logits_indices,
        "cu_num_logits": cu_num_logits,
        "cu_num_logits_np": cu_num_logits_np,
        "has_structured_output_reqs": scheduler_output.has_structured_output_requests,
        "prompt_lens": None,
        "fast_prefill": None,
        "max_query_len": None,
    }
    filtered = {k: v for k, v in all_kwargs.items() if k in st.b1_ib_fields}
    if st.region is not None:
        st.region.stats["bn_fused_prep"] += 1
    return st.b1_ib_cls(**filtered)


def prewarm_prefill_c32(runner) -> dict[str, int | float]:
    """Opt-in (`INKLING_PREFILL_PREWARM=1`): eagerly load lazy prefill .so libs, warm FP8 dense
    weight cache, and pre-JIT multi-req Triton glue kernels right after capture_model() so the
    first C=32 ramp-up prefill pays 0 ms of cold-load / JIT stall. 100% 0-ULP bit-exact."""
    t0 = time.perf_counter()
    summary: dict[str, int | float] = {"libs": 0, "fp8_weights": 0, "jits": 0, "ms": 0.0}
    device = getattr(runner, "device", None)
    if device is None or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return summary

    # 1. Eagerly load lazy prefill CUDA .so libraries
    for mod_path, fn_name in (
        ("vllm.models.inkling2.sconv", "_get_fused_prefill_lib"),
        ("vllm.models.inkling2.ops.sm12_attn", "get_local_otf_rel_lib"),
        ("vllm.models.inkling2.ops.sm12_attn", "get_global_tail_otf_rel_lib"),
        ("vllm.models.inkling2.ops.pf_moe_native", "get_lib"),
        ("vllm.models.inkling2.ops.prefill_qf", "get_lib"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is not None and fn() is not None:
                summary["libs"] += 1
        except Exception:  # noqa: BLE001
            pass

    # 2. Pre-warm FP8 dense & router weight caches (_FP8_W_CACHE) for all 48 target layers
    try:
        from vllm.models.inkling2.ops import linear as _lin  # noqa: PLC0415

        want_dense = bool(getattr(_lin, "_PF_DENSE_FP8", False))
        want_router = bool(getattr(_lin, "_PF_ROUTER_FP8", False))
        if want_dense or want_router:
            mode, _ = _lin._probe_fp8_cap(device)
            if mode != "off":
                model = getattr(runner, "model", None)
                lm = model.get_language_model() if hasattr(model, "get_language_model") else model
                inner = getattr(lm, "model", lm)
                layers = getattr(inner, "layers", None) or ()
                for layer in layers:
                    if want_dense:
                        attn = getattr(layer, "attn", None)
                        if attn is not None:
                            is_local = bool(getattr(layer, "is_local", True))
                            qkvr = getattr(attn, "qkvr", None)
                            wq = getattr(qkvr, "weight", None)
                            if (
                                isinstance(wq, torch.Tensor)
                                and wq.is_cuda
                                and wq.dtype == torch.bfloat16
                                and wq.ndim == 2
                                and wq.untyped_storage().size() > 0
                            ):
                                if not is_local and hasattr(attn, "q_side_splits"):
                                    wq_use = wq[: sum(attn.q_side_splits)]
                                else:
                                    wq_use = wq
                                _lin._get_cached_w_fp8(wq_use, mode)
                                summary["fp8_weights"] += 1
                            wo = getattr(attn, "wo_ud", None)
                            wwo = getattr(wo, "weight", None)
                            if (
                                isinstance(wwo, torch.Tensor)
                                and wwo.is_cuda
                                and wwo.dtype == torch.bfloat16
                                and wwo.ndim == 2
                                and wwo.untyped_storage().size() > 0
                            ):
                                _lin._get_cached_w_fp8(wwo, mode)
                                summary["fp8_weights"] += 1
                    if want_router:
                        mlp = getattr(layer, "mlp", None)
                        gate = getattr(mlp, "gate", None) if mlp is not None else None
                        wg = getattr(gate, "weight", None)
                        if (
                            isinstance(wg, torch.Tensor)
                            and wg.is_cuda
                            and wg.dtype == torch.bfloat16
                            and wg.ndim == 2
                            and wg.untyped_storage().size() > 0
                        ):
                            _lin._get_cached_w_fp8(wg, mode)
                            summary["fp8_weights"] += 1
    except Exception:  # noqa: BLE001
        pass

    # 3. Pre-JIT multi-req sconv_seq_metadata (skipped at B=1 by INKLING_SCONV_META_DEDUP=1!)
    try:
        from vllm.models.inkling.nvidia.ops.sconv import sconv_seq_metadata  # noqa: PLC0415

        qsl = torch.arange(0, 33, dtype=torch.int32, device=device) * 8
        seq = torch.zeros(256, dtype=torch.int32, device=device)
        qs = torch.zeros(256, dtype=torch.int32, device=device)
        sconv_seq_metadata(qsl, 32, 256, seq, qs, 256)
        summary["jits"] += 1
    except Exception:  # noqa: BLE001
        pass

    # 4. Sync warmup work cleanly before serving traffic
    try:
        torch.cuda.synchronize(device)
    except Exception:  # noqa: BLE001
        pass
    summary["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    try:
        from vllm.distributed import get_tensor_model_parallel_rank  # noqa: PLC0415

        is_r0 = get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001
        is_r0 = True
    if is_r0:
        _logger().info(
            "INKLING_PREFILL_PREWARM: loaded %d prefill libs, warmed %d FP8 weights, "
            "JIT'd %d multi-req kernels in %.1f ms",
            summary["libs"],
            summary["fp8_weights"],
            summary["jits"],
            summary["ms"],
        )
    return summary


# ----------------------------------------------------------------------------------------------
# Lever C: Short-lived req_states.draft_tokens proxy across propose() -> set_draft_tokens()
# ----------------------------------------------------------------------------------------------
class _DraftSkipProxy:
    """Eliminates both eager index_elementwise_kernels in sample_tokens after _vw_conf_kernel
    already scatter-wrote req_states.draft_tokens[idx_mapping]. Active ONLY between
    speculator.propose() return and draft_tokens_handler.set_draft_tokens() entry."""

    __slots__ = ("_real", "_idx", "_drafts", "_stats")

    def __init__(self, real: torch.Tensor, idx: torch.Tensor, drafts: torch.Tensor, stats=None):
        self._real = real
        self._idx = idx
        self._drafts = drafts
        self._stats = stats

    def __setitem__(self, key, value):
        if key is self._idx and value is self._drafts:
            if self._stats is not None:
                self._stats["elided_draft_put"] += 1
            return
        self._real[key] = value

    def __getitem__(self, key):
        if key is self._idx:
            if self._stats is not None:
                self._stats["elided_draft_get"] += 1
            return self._drafts
        if isinstance(key, tuple) and len(key) == 2 and key[0] is self._idx:
            if self._stats is not None:
                self._stats["elided_draft_get"] += 1
            return self._drafts[:, key[1]]
        return self._real[key]

    def __getattr__(self, name):
        return getattr(self._real, name)


def _get_model_set_direct_embed():
    try:
        from vllm.models.inkling2 import model as _m  # noqa: PLC0415

        return getattr(_m, "set_direct_embed_out", None)
    except Exception:  # noqa: BLE001
        return None


def _get_spec_draft_store_tls():
    try:
        from vllm.models.inkling2 import speculator as _sp  # noqa: PLC0415

        return getattr(_sp, "_DRAFT_STORE_TLS", None)
    except Exception:  # noqa: BLE001
        return None


def _logger():
    try:
        from vllm.logger import init_logger  # noqa: PLC0415

        return init_logger(__name__)
    except Exception:  # noqa: BLE001
        import logging

        return logging.getLogger(TAG)


def _get_state(runner) -> _RunnerState:
    st = getattr(runner, "_inkling_dtg", None)
    if st is None:
        st = runner._inkling_dtg = _RunnerState()
    return st


def _ensure_region(st: _RunnerState):
    if st.region is None:
        _install_hooks()
        st.region = Region(
            TorchBackend(CFG["update"], max_execs=CFG["max_execs"]),
            max_seg=CFG["seg"],
        )
        if st.rank0:
            _logger().info(
                "INKLING_DTGRAPH armed: regions=%s mode=%s warm=%d max_seg=%d max_reqs=%d max_execs=%d",
                sorted(CFG["regions"]),
                "update" if CFG["update"] else "instantiate",
                CFG["warm"],
                CFG["seg"],
                CFG["max_reqs"],
                CFG["max_execs"],
            )
    _hook_rdma()
    stream = torch.cuda.current_stream()
    if st.region.b.is_default(stream):
        raise RuntimeError("current stream is the default stream")
    return stream


def _open_region(st: _RunnerState, region_name: str, shape_key: tuple) -> bool:
    st.seen[(region_name, shape_key)] += 1
    if st.seen[(region_name, shape_key)] <= CFG["warm"]:
        st.skips[f"{region_name}:warm"] += 1
        return False
    try:
        stream = _ensure_region(st)
    except Exception as e:  # noqa: BLE001
        st.disabled = True
        _logger().warning("INKLING_DTGRAPH disabled before capture: %s", e)
        return False
    try:
        st.region.begin(stream, region_name=region_name, shape_key=shape_key)
    except Exception as e:  # noqa: BLE001
        st.region.active = False
        _TLS.region = None
        st.disabled = True
        _logger().warning("INKLING_DTGRAPH disabled (capture_begin failed: %s)", e)
        return False
    st.mode = _Mode.make(st.region)
    st.mode.__enter__()
    return True


def close_region(
    runner, expected_name: str | None = None, *, keep_mode: bool = False
) -> None:
    st = getattr(runner, "_inkling_dtg", None)
    if st is None:
        return
    r = st.region
    has_active = r is not None and r.active
    if has_active:
        if expected_name is not None and r.region_name != expected_name:
            return
        if not keep_mode:
            mode, st.mode = st.mode, None
            if mode is not None:
                mode.__exit__(None, None, None)
        r.end()
        every = CFG["log"]
        if every and r.n % every == 0 and st.rank0:
            s = dict(r.stats)
            _logger().info(
                "INKLING_DTGRAPH stats after %d regions: stats=%s skips=%s update_hits=%d update_fail=%d evictions=%d",
                r.n,
                s,
                dict(st.skips),
                getattr(r.b, "update_hits", 0),
                getattr(r.b, "update_fail", 0),
                getattr(r.b, "evictions", 0),
            )
    elif not keep_mode and st.mode is not None:
        # Finish deferred TorchDispatchMode.__exit__() strictly outside nested
        # @torch.inference_mode() guards (e.g. DFlashSpeculator.propose) so PyTorch's
        # ForceDispatchKeyGuard never restores DispatchKey::Python onto an empty mode stack!
        mode, st.mode = st.mode, None
        mode.__exit__(None, None, None)


def abort_region(runner) -> None:
    st = getattr(runner, "_inkling_dtg", None)
    if st is None:
        return
    has_active = st.region is not None and st.region.active
    mode, st.mode = st.mode, None
    if mode is not None:
        try:
            mode.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
    if not has_active:
        return
    st.region.abort()
    st.disabled = True
    _logger().error("INKLING_DTGRAPH: exception inside glue region; capture dropped, lever disabled")


_INSTALLED = False


def install() -> bool:
    """Install lazy hooks on GPUModelRunner & graph managers when INKLING_DTGRAPH is enabled.
    Called once from Inkling2DSparkSpeculator.__init__ (or Inkling2Model). Safe & idempotent."""
    global _INSTALLED
    if not ON or _INSTALLED:
        return False
    _INSTALLED = True
    try:
        from vllm.v1.worker.gpu import model_runner as mr  # noqa: PLC0415

        cls = mr.GPUModelRunner
        if getattr(cls, "_inkling_dtg_installed", False):
            return True

        # 1. Wrap execute_model: abort on exception, ensure DT closes before non-FULL return,
        # and hook model_state.prepare_inputs_embeds for direct zero-copy embed output.
        orig_exec = cls.execute_model
        set_direct_embed = _get_model_set_direct_embed() if FUSED_EMBED_DIRECT else None
        draft_store_tls = _get_spec_draft_store_tls() if FUSED_DRAFT_STORE else None

        def wrapped_execute_model(self, *args, **kwargs):
            st = _get_state(self)
            # Ensure target cudagraph_manager.run_fullgraph closes DT right before replay.
            cgm = getattr(self, "cudagraph_manager", None)
            if cgm is not None and not getattr(cgm, "_inkling_dtg_cgm", False):
                orig_rfg = cgm.run_fullgraph

                def _rfg(batch_desc, _orig=orig_rfg, _runner=self):
                    close_region(_runner, expected_name="dt")
                    return _orig(batch_desc)

                cgm.run_fullgraph = _rfg
                cgm._inkling_dtg_cgm = True
            # Lever B: Hook model_state.prepare_inputs_embeds once to write directly into inputs_embeds.
            ms = getattr(self, "model_state", None)
            if (
                set_direct_embed is not None
                and ms is not None
                and hasattr(ms, "prepare_inputs_embeds")
                and not getattr(ms, "_inkling_dtg_pie", False)
            ):
                orig_pie = ms.prepare_inputs_embeds

                def _wrapped_pie(
                    scheduled_encoder_inputs,
                    input_batch,
                    req_states,
                    _orig=orig_pie,
                    _ms=ms,
                    _st=st,
                    _set_out=set_direct_embed,
                ):
                    direct_set = False
                    if (
                        not scheduled_encoder_inputs
                        and not getattr(input_batch, "has_prefill", True)
                        and int(getattr(input_batch, "num_tokens", 0) or 0) > 0
                    ):
                        buf = _st.inputs_embeds_buf
                        if buf is None:
                            cand = getattr(_ms, "inputs_embeds", None)
                            if cand is None:
                                er = getattr(_ms, "encoder_runner", None)
                                cand = getattr(er, "inputs_embeds", None) if er is not None else None
                            if (
                                isinstance(cand, torch.Tensor)
                                and cand.dim() == 2
                                and cand.is_contiguous()
                            ):
                                buf = _st.inputs_embeds_buf = cand
                        nt = int(input_batch.num_tokens)
                        if (
                            buf is not None
                            and buf.dim() == 2
                            and buf.is_contiguous()
                            and buf.shape[0] >= nt
                        ):
                            _set_out(buf[:nt])
                            direct_set = True
                    try:
                        res = _orig(scheduled_encoder_inputs, input_batch, req_states)
                    finally:
                        if direct_set:
                            _set_out(None)
                    if (
                        isinstance(res, torch.Tensor)
                        and res.dim() == 2
                        and res.is_contiguous()
                        and (
                            _st.inputs_embeds_buf is None
                            or res.data_ptr() != _st.inputs_embeds_buf.data_ptr()
                            or res.shape[0] > _st.inputs_embeds_buf.shape[0]
                        )
                    ):
                        _st.inputs_embeds_buf = res
                    return res

                ms.prepare_inputs_embeds = _wrapped_pie
                ms._inkling_dtg_pie = True
            try:
                out = orig_exec(self, *args, **kwargs)
                close_region(self, expected_name="dt")
                return out
            except BaseException:
                abort_region(self)
                raise

        # 2. Wrap prepare_inputs: open Phase 4 ("dt") region right before prepare_inputs,
        # and run Lever A (1-CTA Triton B=1 fused prepare_inputs, zero H2D DMA) when eligible.
        orig_prep = cls.prepare_inputs

        def wrapped_prepare_inputs(
            self, scheduler_output, batch_req_state, batch_desc, num_active_loras
        ):
            st = _get_state(self)
            b1_elig = (
                FUSED_DT_PREP
                and not st.b1_failed
                and _b1_prep_eligible(self, scheduler_output, batch_req_state, batch_desc)
            )
            bn_elig = (
                (not b1_elig)
                and FUSED_DT_PREP
                and FUSED_DT_PREP_MAX_B >= 2
                and not st.b1_failed
                and _bn_prep_eligible(self, scheduler_output, batch_req_state, batch_desc)
            )
            need_b1_warm = False
            if b1_elig:
                nt_pad = max(int(batch_req_state.num_tokens), int(batch_desc.num_tokens))
                if st.b1_dtypes is None:
                    need_b1_warm = True
                else:
                    _ensure_b1_bufs(st, self.device, nt_pad)
                    from vllm import envs  # noqa: PLC0415

                    bt = max(16, 1 << max(0, nt_pad - 1).bit_length())
                    if (bool(envs.VLLM_MOE_SKIP_PADDING), 256, bt) not in st.b1_compiled:
                        need_b1_warm = True
            elif bn_elig:
                nr = len(batch_req_state.req_ids)
                nt = int(batch_req_state.num_tokens)
                nt_pad = max(nt, int(batch_desc.num_tokens))
                ql = nt // nr
                if st.b1_dtypes is None:
                    need_b1_warm = True
                else:
                    _ensure_bn_bufs(st, self.device, nr, nt_pad)
                    from vllm import envs  # noqa: PLC0415

                    has_d = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
                    bq = max(16, 1 << max(0, ql - 1).bit_length())
                    if (bool(envs.VLLM_MOE_SKIP_PADDING), has_d, 256, bq, 256) not in st.bn_compiled:
                        need_b1_warm = True

            if not st.disabled and "dt" in CFG["regions"] and not need_b1_warm:
                num_reqs = len(batch_req_state.req_ids)
                why = dt_ineligible(
                    cg_full=getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode)) == "FULL",
                    num_ubatches=int(getattr(batch_desc, "num_ubatches", 1) or 1),
                    has_prefill=bool(batch_req_state.has_prefill),
                    pcp=getattr(self, "pcp_manager", None) is not None,
                    dcp_size=int(getattr(self, "dcp_size", 1) or 1),
                    lora=bool(getattr(self, "lora_config", None)) and num_active_loras > 0,
                    encoder_inputs=bool(getattr(scheduler_output, "scheduled_encoder_inputs", None)),
                    encoder_decoder=bool(getattr(self, "is_encoder_decoder", False)),
                    pp_single=bool(
                        getattr(self, "is_first_pp_rank", True)
                        and getattr(self, "is_last_pp_rank", True)
                    ),
                    num_reqs=num_reqs,
                    max_reqs=CFG["max_reqs"],
                )
                if why is not None:
                    st.skips[f"dt:{why}"] += 1
                else:
                    shape_key = (
                        num_reqs,
                        int(batch_req_state.num_tokens),
                        int(batch_desc.num_tokens),
                    )
                    _open_region(st, "dt", shape_key)
            if b1_elig:
                if st.b1_dtypes is None:
                    ib = orig_prep(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
                    _probe_b1_from_ib(st, ib)
                    if st.b1_dtypes is not None and not st.b1_failed:
                        _ensure_b1_bufs(
                            st,
                            self.device,
                            max(int(batch_req_state.num_tokens), int(batch_desc.num_tokens)),
                        )
                    return ib
                ib = _try_fused_prepare_inputs_b1(
                    self, st, scheduler_output, batch_req_state, batch_desc
                )
                if ib is not None:
                    return ib
            elif bn_elig:
                if st.b1_dtypes is None:
                    ib = orig_prep(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
                    _probe_b1_from_ib(st, ib)
                    if st.b1_dtypes is not None and not st.b1_failed:
                        _ensure_bn_bufs(
                            st,
                            self.device,
                            len(batch_req_state.req_ids),
                            max(int(batch_req_state.num_tokens), int(batch_desc.num_tokens)),
                        )
                    return ib
                ib = _try_fused_prepare_inputs_bn(
                    self, st, scheduler_output, batch_req_state, batch_desc
                )
                if ib is not None:
                    return ib
            return orig_prep(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)

        # 3. Wrap sample_tokens + postprocess_sampled for Phase 2B ("td", post-AsyncOutput)
        # and Lever C (direct req_states.draft_tokens scatter in _vw_conf + skip readback).
        orig_sample_tokens = cls.sample_tokens
        orig_postprocess = cls.postprocess_sampled

        def wrapped_sample_tokens(self, grammar_output):
            st = _get_state(self)
            ems = getattr(self, "execute_model_state", None)
            spec = getattr(self, "speculator", None)
            # Hook draft query_cudagraph_manager.run_fullgraph once to close TD capture right before
            # replay, and hook spec._generate_draft once for eager draft fallback.
            # CRITICAL PyTorch LIFO invariant: DFlashSpeculator.propose() is decorated with
            # @torch.inference_mode() (ForceDispatchKeyGuard snapshots TLS DispatchKey::Python on
            # entry and restores it on exit!). Therefore close_region inside propose() uses
            # keep_mode=True (ends CUDA graph capture immediately; r.dispatch becomes a 1-line no-op),
            # and _prop's finally block exits TorchDispatchMode strictly AFTER propose() returns!
            qcgm = getattr(spec, "query_cudagraph_manager", None) if spec is not None else None
            if qcgm is not None and not getattr(qcgm, "_inkling_dtg_qcgm", False):
                orig_qrfg = qcgm.run_fullgraph

                def _qrfg(batch_desc, _orig=orig_qrfg, _runner=self):
                    close_region(_runner, expected_name="td", keep_mode=True)
                    return _orig(batch_desc)

                qcgm.run_fullgraph = _qrfg
                qcgm._inkling_dtg_qcgm = True
            if spec is not None and hasattr(spec, "_generate_draft") and not getattr(spec, "_inkling_dtg_gd", False):
                orig_gd = spec._generate_draft

                def _gd(*a, _orig=orig_gd, _runner=self, **kw):
                    close_region(_runner, expected_name="td", keep_mode=True)
                    return _orig(*a, **kw)

                spec._generate_draft = _gd
                spec._inkling_dtg_gd = True
            # Always hook spec.propose (even when FUSED_DRAFT_STORE=0) so deferred
            # TorchDispatchMode.__exit__() ALWAYS executes strictly outside propose()'s
            # @torch.inference_mode() guard (plus Lever C _DraftSkipProxy when active).
            if (
                spec is not None
                and hasattr(spec, "propose")
                and not getattr(spec, "_inkling_dtg_prop", False)
            ):
                orig_prop = spec.propose

                def _prop(*a, _orig=orig_prop, _runner=self, _st=st, _tls=draft_store_tls, **kw):
                    try:
                        drafts = _orig(*a, **kw)
                    finally:
                        close_region(_runner, expected_name="td", keep_mode=False)
                    if (
                        _tls is not None
                        and _st.draft_store_active
                        and _st.draft_store_real is not None
                        and getattr(_tls, "written_drafts", None) is drafts
                        and getattr(_tls, "written_idx", None) is not None
                    ):
                        _runner.req_states.draft_tokens = _DraftSkipProxy(
                            _st.draft_store_real,
                            _tls.written_idx,
                            drafts,
                            _st.region.stats if _st.region is not None else None,
                        )
                    return drafts

                spec.propose = _prop
                spec._inkling_dtg_prop = True
            dth = getattr(self, "draft_tokens_handler", None)
            if (
                draft_store_tls is not None
                and dth is not None
                and hasattr(dth, "set_draft_tokens")
                and not getattr(dth, "_inkling_dtg_sdt", False)
            ):
                orig_sdt = dth.set_draft_tokens

                def _sdt(*a, _orig=orig_sdt, _runner=self, _st=st, **kw):
                    if _st.draft_store_real is not None and isinstance(
                        getattr(_runner.req_states, "draft_tokens", None), _DraftSkipProxy
                    ):
                        _runner.req_states.draft_tokens = _st.draft_store_real
                    return _orig(*a, **kw)

                dth.set_draft_tokens = _sdt
                dth._inkling_dtg_sdt = True

            if ems is not None and not st.disabled and "td" in CFG["regions"]:
                ib = ems.input_batch
                st.cur_sample_ctx = {
                    "ib": ib,
                    "num_spec": int(getattr(ems, "num_spec_tokens_to_schedule", 0) or 0),
                }
            else:
                st.cur_sample_ctx = None

            st.draft_store_active = False
            st.draft_store_real = None
            if (
                draft_store_tls is not None
                and ems is not None
                and spec is not None
                and getattr(self, "pcp_manager", None) is None
                and getattr(self, "pp_handler", None) is None
                and getattr(self, "adaptive_verification", None) is None
                and getattr(self, "num_speculative_steps", 0) > 0
                and int(getattr(ems, "num_spec_tokens_to_schedule", 0) or 0)
                == int(self.num_speculative_steps)
                and not getattr(ems.input_batch, "has_structured_output_reqs", False)
                and hasattr(self, "req_states")
                and isinstance(getattr(self.req_states, "draft_tokens", None), torch.Tensor)
                and dth is not None
            ):
                st.draft_store_real = self.req_states.draft_tokens
                st.draft_store_active = True
                draft_store_tls.req_drafts = st.draft_store_real
                draft_store_tls.written_drafts = None
                draft_store_tls.written_idx = None
            try:
                out = orig_sample_tokens(self, grammar_output)
                close_region(self, expected_name="td")
                return out
            except BaseException:
                abort_region(self)
                raise
            finally:
                if st.draft_store_real is not None and isinstance(
                    getattr(self.req_states, "draft_tokens", None), _DraftSkipProxy
                ):
                    self.req_states.draft_tokens = st.draft_store_real
                st.draft_store_active = False
                st.draft_store_real = None
                if draft_store_tls is not None:
                    draft_store_tls.req_drafts = None
                    draft_store_tls.written_drafts = None
                    draft_store_tls.written_idx = None
                st.cur_sample_ctx = None

        def wrapped_postprocess(self, *args, **kwargs):
            st = _get_state(self)
            ctx = st.cur_sample_ctx
            if ctx is not None and not st.disabled and "td" in CFG["regions"]:
                ib = ctx["ib"]
                spec = getattr(self, "speculator", None)
                qcgm = getattr(spec, "query_cudagraph_manager", None) if spec is not None else None
                why = td_ineligible(
                    has_speculator=spec is not None,
                    has_full_draft_cg=qcgm is not None,
                    has_prefill=bool(getattr(ib, "has_prefill", False)),
                    pcp=getattr(self, "pcp_manager", None) is not None,
                    dcp_size=int(getattr(self, "dcp_size", 1) or 1),
                    mm_inputs=bool(
                        spec is not None
                        and getattr(spec, "supports_mm_inputs", False)
                        and getattr(self.model_state, "supports_mm_inputs", False)
                    ),
                    num_spec_tokens=ctx["num_spec"],
                    context_kv_only=bool(getattr(spec, "context_kv_only", False)) if spec else True,
                    conf_log=bool(getattr(spec, "_cl_on", False)) if spec else False,
                    vw_stats=bool(getattr(spec, "_vw_stats_on", False)) if spec else False,
                    num_reqs=int(getattr(ib, "num_reqs", 1)),
                    max_reqs=CFG["max_reqs"],
                )
                if why is not None:
                    st.skips[f"td:{why}"] += 1
                else:
                    shape_key = (int(ib.num_reqs), int(ib.num_tokens), int(ctx["num_spec"]))
                    _open_region(st, "td", shape_key)
            return orig_postprocess(self, *args, **kwargs)

        # 4. Register static req_states.draft_tokens buffer before speculator.capture()
        #    so in-graph _vw_conf_kernel can write directly into req_states.draft_tokens,
        #    and optionally run opt-in INKLING_PREFILL_PREWARM right after capture completes.
        if (FUSED_DRAFT_STORE or PREFILL_PREWARM) and hasattr(cls, "capture_model"):
            orig_cap = cls.capture_model

            def wrapped_capture_model(self, *args, _orig=orig_cap, **kwargs):
                if FUSED_DRAFT_STORE:
                    spec = getattr(self, "speculator", None)
                    rs = getattr(self, "req_states", None)
                    if (
                        spec is not None
                        and rs is not None
                        and hasattr(spec, "set_req_drafts_buffer")
                        and isinstance(getattr(rs, "draft_tokens", None), torch.Tensor)
                    ):
                        spec.set_req_drafts_buffer(rs.draft_tokens)
                res = _orig(self, *args, **kwargs)
                if PREFILL_PREWARM:
                    prewarm_prefill_c32(self)
                return res

            cls.capture_model = wrapped_capture_model

        cls.execute_model = wrapped_execute_model
        cls.prepare_inputs = wrapped_prepare_inputs
        cls.sample_tokens = wrapped_sample_tokens
        cls.postprocess_sampled = wrapped_postprocess
        cls._inkling_dtg_installed = True
        _logger().info(
            "INKLING_DTGRAPH hooks installed on GPUModelRunner "
            "(regions=%s, update=%s, max_execs=%d, fused_dt_prep=%d[max_b=%d], "
            "direct_embed=%d, fused_draft_store=%d, prefill_prewarm=%d)",
            sorted(CFG["regions"]),
            CFG["update"],
            CFG["max_execs"],
            int(FUSED_DT_PREP),
            FUSED_DT_PREP_MAX_B,
            int(FUSED_EMBED_DIRECT),
            int(FUSED_DRAFT_STORE),
            int(PREFILL_PREWARM),
        )
        return True
    except Exception as e:  # noqa: BLE001
        _logger().warning("INKLING_DTGRAPH install failed (%r); staying eager", e)
        return False
