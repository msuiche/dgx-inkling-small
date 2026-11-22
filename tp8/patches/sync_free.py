# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Small greedy "sync-free" target head (output-exact; port of DSv4.1 DSV41_SYNC_FREE).

conc-sched concurrency upgrades (100% bit-exact, zero B=1 change):
  1. INKLING_SYNC_FREE_MAX_M (default 256, previously hardcoded 128): prevents the
     catastrophic C24/C32 cliff where M=192..256 disarms sync_free and falls back
     to a 154-206 MB fp32 full-vocab all-gather over RoCEv2!
  2. INKLING_SYNC_FREE_K0=1 (default 1): hooks Sampler.__call__ in addition to
     RejectionSampler.__call__ so greedy K=0 steps (Dynamic SD at B>=16 and
     greedy prefill final chunks) stay armed with cert_head + 64 B/row RDMA AG
     instead of falling back to full [M, 201024] fp32 all-gather + stock head.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

try:  # CPU unit tests import this file without vllm
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger("inkling_sync_free")

_OFF = ("", "0", "off", "false", "no")


def parse_mode(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in _OFF:
        return "off"
    if v in ("1", "on", "all", "true", "argmax"):
        return "argmax"
    if v == "audit":
        return "audit"
    logger.warning("INKLING_SYNC_FREE=%r not recognised (expected off|argmax|audit); feature OFF", raw)
    return "off"


MODE = parse_mode(os.environ.get("INKLING_SYNC_FREE", "off"))
AUDIT_EVERY = int(os.environ.get("INKLING_SYNC_FREE_AUDIT_EVERY", "2000") or 2000)
SPLIT = int(os.environ.get("INKLING_SYNC_FREE_SPLIT", "4") or 4)
AGREE = os.environ.get("INKLING_SYNC_FREE_AGREE", "0").strip().lower() not in _OFF
FUSE = os.environ.get("INKLING_SYNC_FREE_FUSE", "1").strip().lower() not in _OFF
SF_BLOCK = 2048  # local kernel column block
OH_BLOCK = 4096  # one-hot kernel column block
MAX_M = max(1, int(os.environ.get("INKLING_SYNC_FREE_MAX_M", "256") or 256))
ALLOW_K0 = os.environ.get("INKLING_SYNC_FREE_K0", "1").strip().lower() not in _OFF
CERT = os.environ.get("INKLING_CERT_HEAD", "off").strip().lower()
CERT = "off" if CERT in _OFF else ("1" if CERT in ("1", "on", "true", "argmax") else ("check" if CERT in ("check", "audit") else "off"))
DIRECT = os.environ.get("INKLING_SYNC_FREE_DIRECT", "1").strip().lower() not in _OFF
FUSE_REJECT = os.environ.get("INKLING_SYNC_FREE_FUSE_REJECT", "1").strip().lower() not in _OFF
ZERO_COPY_SLICE = os.environ.get("INKLING_SYNC_FREE_SLICE", "1").strip().lower() not in _OFF
FP32_FUSE = os.environ.get("INKLING_SYNC_FREE_FP32_FUSE", "0").strip().lower() not in _OFF
TREE_SPEC = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in _OFF

_S: dict = {
    "armed": False,
    "armed_k0": False,
    "num_tokens": 0,
    "installed": False,
    "runner_hooked": False,
    "tok": None,
    "gathered": None,
    "rs_hooked": False,
    "sp_hooked": False,
}


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# =====================================================================================================
# Kernels
# =====================================================================================================
_K: dict = {}


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as tldevice

    @triton.jit
    def _rn_bf16(x):
        bits = x.to(tl.uint32, bitcast=True)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _chain(x, inv_cap, cap):
        y = _rn_bf16(x * inv_cap)
        t = tldevice.tanh(y)
        y = _rn_bf16(tl.where(tl.abs(y) < 1.1754943508222875e-38, y, t))
        return _rn_bf16(y * cap)

    @triton.jit(do_not_specialize=["n"])
    def _chain_kernel(X, Y, n, inv_cap, cap, BLOCK: tl.constexpr):
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = cols < n
        x = tl.load(X + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + cols, _chain(x, inv_cap, cap).to(tl.bfloat16), mask=m)

    @triton.jit(do_not_specialize=["sl", "valid", "vstart", "seg"])
    def _local_kernel(LG, sl, OUT, valid, vstart, seg, inv_cap, cap,
                      CAP: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr,
                      FP32_CHAIN: tl.constexpr = False, DIVMODE: tl.constexpr = 0):
        row = tl.program_id(0)
        sp = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        lo = sp * seg
        hi = tl.minimum(lo + seg, valid)
        best_v = tl.full((), float("-inf"), tl.float32)
        best_i = lo
        for v0 in range(lo, hi, BLOCK):
            c = v0 + cols
            m = c < hi
            x = tl.load(LG + row.to(tl.int64) * sl + c, mask=m, other=0.0).to(tl.float32)
            bad = (x != x) | (~m)
            if CAP:
                if FP32_CHAIN:
                    y = x * inv_cap if DIVMODE == 0 else tldevice.div_rn(x, cap)
                    t = tldevice.tanh(y)
                    t = tl.where(tl.abs(y) < 1.1754943508222875e-38, y, t)
                    x = t * cap
                else:
                    x = _chain(x, inv_cap, cap)
            x = tl.where(bad, float("-inf"), x)
            v, i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
            upd = v > best_v
            best_i = tl.where(upd, v0 + i, best_i)
            best_v = tl.where(upd, v, best_v)
        base = OUT + (row * SPLIT + sp) * 4
        tl.store(base, best_v)
        tl.store(base + 1, (vstart + best_i).to(tl.float32))
        tl.store(base + 2, 0.0)
        tl.store(base + 3, 0.0)

    @triton.jit(do_not_specialize=["M", "so", "V"])
    def _onehot_kernel(G, M, TOK, OUT, so, V, TP: tl.constexpr, SPLIT: tl.constexpr, NP: tl.constexpr,
                       BLOCK: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        gi = ((r * M + row) * SPLIT + sidx) * 4
        v = tl.load(G + gi, mask=ok, other=float("-inf"))
        _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
        best_t = tl.load(G + ((w // SPLIT) * M + row) * SPLIT * 4 + (w % SPLIT) * 4 + 1)
        tok = best_t.to(tl.int32)
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        val = tl.where(cols == tok, 0.0, float("-inf")).to(OUT.dtype.element_ty)
        tl.store(OUT + row * so + cols, val, mask=cols < V)
        if blk == 0:
            tl.store(TOK + row, tok.to(tl.int64))

    @triton.jit(do_not_specialize=["M"])
    def _tok_only_kernel(
        G, M, TOK, TP: tl.constexpr, SPLIT: tl.constexpr, NP: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        row = tl.program_id(0)
        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        gi = ((r * M + row) * SPLIT + sidx) * 4
        v = tl.load(G + gi, mask=ok, other=float("-inf"))
        _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
        best_t = tl.load(G + ((w // SPLIT) * M + row) * SPLIT * 4 + (w % SPLIT) * 4 + 1)
        tl.store(TOK + row, best_t.to(tl.int32).to(tl.int64))

    @triton.jit
    def _direct_greedy_reject_kernel(
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        tok_ptr,
        draft_sampled_ptr,
        cu_num_logits_ptr,
    ):
        req_idx = tl.program_id(0)
        start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
        num_draft_tokens = end_idx - start_idx - 1
        accepted_length = tl.zeros((), tl.int64)
        verifying = True
        for i in range(num_draft_tokens):
            logit_idx = start_idx + i
            draft_tok = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            is_valid_draft = draft_tok >= 0
            if verifying:
                target_tok = tl.load(tok_ptr + logit_idx).to(tl.int64)
                accepted = (target_tok == draft_tok) & is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, draft_tok, target_tok),
                )
        if accepted_length == num_draft_tokens:
            bonus_tok = tl.load(tok_ptr + start_idx + num_draft_tokens).to(tl.int64)
            tl.store(sampled_ptr + req_idx * sampled_stride + num_draft_tokens, bonus_tok)
        tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))

    @triton.jit(do_not_specialize=["M"])
    def _fused_tok_gather_reject_kernel(
        G,
        M,
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        num_rejected_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        positions_ptr,
        logits_indices_ptr,
        idx_mapping_ptr,
        prefill_len_ptr,
        cu_num_logits_ptr,
        TP: tl.constexpr,
        SPLIT: tl.constexpr,
        NP: tl.constexpr,
        PDL: tl.constexpr = True,
    ):
        """Phase 2A 4->1 fusion: _tok_only + gather_draft_sampled + _direct_greedy_reject +
        _get_num_sampled_and_rejected. PDL-chained behind cert_head RDMA AG (gdc_wait before
        reading G), computes global argmax inline ONLY for verified rows (exact same
        tie-break-left as _tok_only_kernel), reads draft tokens directly via
        input_ids[logits_indices[logit_idx + 1]] with prefill_len guard, and writes sampled,
        num_sampled, and num_rejected directly (0-ULP bit-exact, eliminates 3 eager launches)."""
        if PDL:
            tl.extra.cuda.gdc_wait()
        req_idx = tl.program_id(0).to(tl.int64)
        start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
        num_draft_tokens = end_idx - start_idx - 1
        req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
        req_prefill_len = tl.load(prefill_len_ptr + req_state_idx).to(tl.int64)
        seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int64)

        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        m64 = M.to(tl.int64)

        accepted_length = tl.zeros((), tl.int64)
        verifying = True
        for i in range(num_draft_tokens):
            logit_idx = start_idx + i
            tok_idx = tl.load(logits_indices_ptr + logit_idx + 1).to(tl.int64)
            draft_tok = tl.load(input_ids_ptr + tok_idx).to(tl.int64)
            draft_pos = tl.load(positions_ptr + tok_idx).to(tl.int64)
            is_valid_draft = (draft_tok >= 0) & (draft_pos >= req_prefill_len)
            if verifying:
                gi = ((r.to(tl.int64) * m64 + logit_idx) * SPLIT + sidx) * 4
                v = tl.load(G + gi, mask=ok, other=float("-inf"))
                _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
                best_t = tl.load(
                    G + ((w // SPLIT).to(tl.int64) * m64 + logit_idx) * (SPLIT * 4) + (w % SPLIT) * 4 + 1
                )
                target_tok = best_t.to(tl.int32).to(tl.int64)
                accepted = (target_tok == draft_tok) & is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, draft_tok, target_tok),
                )
        if accepted_length == num_draft_tokens:
            bonus_row = start_idx + num_draft_tokens
            gi = ((r.to(tl.int64) * m64 + bonus_row) * SPLIT + sidx) * 4
            v = tl.load(G + gi, mask=ok, other=float("-inf"))
            _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
            best_t = tl.load(
                G + ((w // SPLIT).to(tl.int64) * m64 + bonus_row) * (SPLIT * 4) + (w % SPLIT) * 4 + 1
            )
            bonus_tok = best_t.to(tl.int32).to(tl.int64)
            tl.store(sampled_ptr + req_idx * sampled_stride + num_draft_tokens, bonus_tok)
        incomplete = seq_len < req_prefill_len
        ns_out = tl.where(incomplete, 0, accepted_length + 1).to(tl.int32)
        nr_out = tl.where(incomplete, 0, num_draft_tokens - accepted_length).to(tl.int32)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(num_sampled_ptr + req_idx, ns_out)
        tl.store(num_rejected_ptr + req_idx, nr_out)

    @triton.jit(do_not_specialize=["M"])
    def _fused_tree_reject_kernel(
        G,
        M,
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        num_rejected_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        positions_ptr,
        logits_indices_ptr,
        idx_mapping_ptr,
        prefill_len_ptr,
        cu_num_logits_ptr,
        tree_parent_ptr,
        n_live_ptr,
        accepted_path_ptr,
        accepted_len_ptr,
        needs_compact_ptr,
        TP: tl.constexpr,
        SPLIT: tl.constexpr,
        NP: tl.constexpr,
        MAX_M: tl.constexpr = 8,
        PDL: tl.constexpr = True,
    ):
        """0-ULP Tree Greedy Rejection Sampler (C=1).
        Walks from root cur=0, computes global argmax ONLY for visited nodes along the accepted
        path, matches live children c with TreeParent[c]==cur against target_tok, and emits
        (sampled, num_sampled, num_rejected, accepted_path, accepted_len, needs_compact)
        directly into PDL-chained `tree_compact_caches_fwd`!"""
        if PDL:
            tl.extra.cuda.gdc_wait()
        req_idx = tl.program_id(0).to(tl.int64)
        start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
        num_draft_tokens = end_idx - start_idx - 1
        req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
        req_prefill_len = tl.load(prefill_len_ptr + req_state_idx).to(tl.int64)
        seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int64)
        n_live = tl.minimum(tl.load(n_live_ptr).to(tl.int64), MAX_M)

        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        m64 = M.to(tl.int64)

        cur = tl.zeros((), tl.int64)
        depth = tl.zeros((), tl.int64)
        needs_cmp = tl.zeros((), tl.int32)
        tl.store(accepted_path_ptr + 0, 0)
        walking = True

        for _step in tl.static_range(MAX_M):
            if walking:
                logit_row = start_idx + cur
                gi = ((r.to(tl.int64) * m64 + logit_row) * SPLIT + sidx) * 4
                v = tl.load(G + gi, mask=ok, other=float("-inf"))
                _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
                best_t = tl.load(
                    G + ((w // SPLIT).to(tl.int64) * m64 + logit_row) * (SPLIT * 4) + (w % SPLIT) * 4 + 1
                )
                target_tok = best_t.to(tl.int32).to(tl.int64)
                tl.store(sampled_ptr + req_idx * sampled_stride + depth, target_tok)

                matched_c = tl.full((), -1, tl.int64)
                for c in tl.static_range(1, MAX_M):
                    if (c < n_live) & (matched_c < 0):
                        par = tl.load(tree_parent_ptr + c).to(tl.int64)
                        if par == cur:
                            tok_idx = tl.load(logits_indices_ptr + start_idx + c).to(tl.int64)
                            dtok = tl.load(input_ids_ptr + tok_idx).to(tl.int64)
                            dpos = tl.load(positions_ptr + tok_idx).to(tl.int64)
                            if (dtok == target_tok) & (dtok >= 0) & (dpos >= req_prefill_len):
                                matched_c = tl.full((), c, tl.int64)
                if matched_c >= 0:
                    depth = depth + 1
                    tl.store(accepted_path_ptr + depth, matched_c.to(tl.int32))
                    needs_cmp = tl.where(matched_c != depth, 1, needs_cmp)
                    cur = matched_c
                else:
                    walking = False

        incomplete = seq_len < req_prefill_len
        ns_out = tl.where(incomplete, 0, depth + 1).to(tl.int32)
        nr_out = tl.where(incomplete, 0, num_draft_tokens - depth).to(tl.int32)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(num_sampled_ptr + req_idx, ns_out)
        tl.store(num_rejected_ptr + req_idx, nr_out)
        tl.store(accepted_len_ptr, ns_out)
        tl.store(needs_compact_ptr, tl.where(incomplete, 0, needs_cmp))


    _K.update(triton=triton, chain=_chain_kernel, local=_local_kernel, onehot=_onehot_kernel,
              tok_only=_tok_only_kernel, direct_reject=_direct_greedy_reject_kernel,
              fused_reject=_fused_tok_gather_reject_kernel,
              fused_tree_reject=_fused_tree_reject_kernel)
    return _K


def _f32(x: float) -> float:
    return float(np.float32(x))


def inv_cap_of(cap: float) -> float:
    return float(np.float32(1.0) / np.float32(cap))


def chain_selftest(cap: float, device) -> tuple[bool, int]:
    k = _kernels()
    bits = torch.arange(0, 65536, dtype=torch.int32, device=device).to(torch.int16)
    x = bits.view(torch.bfloat16)
    ref = torch.tanh(x / cap) * cap
    y = torch.empty_like(x)
    k["chain"][(k["triton"].cdiv(x.numel(), 1024),)](x, y, x.numel(), inv_cap_of(cap), _f32(cap), BLOCK=1024)
    ok_mask = ~torch.isnan(x)
    diff = (ref.view(torch.int16) != y.view(torch.int16)) & ok_mask
    return int(diff.sum().item()) == 0, int(diff.sum().item())


def torch_chain(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    if cap is None:
        return x
    return torch.tanh(x / cap) * cap


def sf_local_pairs(local: torch.Tensor, valid: int, vstart: int, cap: float | None = None,
                   split: int = 1, fuse: bool = True, fp32_divmode: int = -1) -> torch.Tensor:
    assert local.dim() == 2 and local.stride(1) == 1 and local.dtype in (torch.bfloat16, torch.float32)
    fp32_chain = False
    if local.dtype != torch.bfloat16:
        if FP32_FUSE and cap is not None and fp32_divmode in (0, 1):
            fp32_chain = True
            fuse = True
        else:
            fuse = False
    assert 0 < valid <= local.shape[1] and split >= 1
    k = _kernels()
    M = local.shape[0]
    if cap is not None and not fuse:
        local = torch_chain(local, cap).contiguous()
        cap = None
    seg = SF_BLOCK * k["triton"].cdiv(k["triton"].cdiv(valid, SF_BLOCK), split)
    pair = torch.empty((M * split, 4), dtype=torch.float32, device=local.device)
    k["local"][(M, split)](local, local.stride(0), pair, valid, vstart, seg,
                           inv_cap_of(cap) if cap is not None else 1.0, _f32(cap) if cap is not None else 1.0,
                           CAP=cap is not None, SPLIT=split, BLOCK=SF_BLOCK,
                           FP32_CHAIN=fp32_chain, DIVMODE=fp32_divmode if fp32_chain else 0,
                           num_warps=8)
    return pair


def triton_np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def sf_tok_only(
    gathered: torch.Tensor, M: int, tp: int, split: int = 1, pdl: bool = False
) -> torch.Tensor:
    k = _kernels()
    assert gathered.shape == (tp * M * split, 4) and gathered.is_contiguous() and gathered.dtype == torch.float32
    tok = torch.empty((M,), dtype=torch.int64, device=gathered.device)
    k["tok_only"][(M,)](
        gathered, M, tok, TP=tp, SPLIT=split, NP=triton_np2(tp * split),
        PDL=pdl, num_warps=1, launch_pdl=pdl,
    )
    return tok


def sf_onehot(gathered: torch.Tensor, M: int, tp: int, V: int, split: int = 1, dtype=torch.bfloat16):
    k = _kernels()
    assert gathered.shape == (tp * M * split, 4) and gathered.is_contiguous() and gathered.dtype == torch.float32
    out = torch.empty((M, V), dtype=dtype, device=gathered.device)
    tok = torch.empty((M,), dtype=torch.int64, device=gathered.device)
    k["onehot"][(M, k["triton"].cdiv(V, OH_BLOCK))](gathered, M, tok, out, out.stride(0), V,
                                                     TP=tp, SPLIT=split, NP=triton_np2(tp * split), BLOCK=OH_BLOCK, num_warps=4)
    return out, tok


# =====================================================================================================
# Engine integration
# =====================================================================================================
def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def _tp_min(vals: list[int]) -> list[int]:
    from vllm.distributed import get_tp_group

    grp = get_tp_group()
    if grp.world_size == 1:
        return vals
    t = torch.tensor(vals, dtype=torch.int64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp.cpu_group)
    return [int(v) for v in t.tolist()]


def step_armed(runner, input_batch, grammar_output) -> tuple[bool, bool]:
    """Returns (armed, is_k0).
    True only when the only consumer of this step's target logits is a greedy argmax.
    Supports both speculative verification (num_draft_tokens > 0) and non-speculative /
    Dynamic-SD K=0 greedy steps (num_draft_tokens == 0, when ALLOW_K0 is enabled)."""
    try:
        if getattr(runner, "batch_sharder", None) is not None:
            return False, False
        if grammar_output is not None or input_batch is None:
            return False, False
        rs, sp = getattr(runner, "rejection_sampler", None), getattr(runner, "sampler", None)
        if sp is None:
            return False, False
        num_draft = int(getattr(input_batch, "num_draft_tokens", 0))
        is_k0 = (num_draft <= 0) or (rs is None)
        if is_k0:
            if not ALLOW_K0:
                return False, False
            # Require exactly 1 logit per request (excludes prompt_logprobs / expanded logits)
            cu_np = getattr(input_batch, "cu_num_logits_np", None)
            if cu_np is None or len(cu_np) == 0 or int(cu_np[-1]) != int(input_batch.num_reqs):
                return False, False
            if getattr(sp, "watermark_key", None) is not None:
                return False, False
        else:
            if getattr(rs, "watermark_key", None) is not None or getattr(rs, "synthetic_conditional_rates", None) is not None:
                return False, False
            if getattr(rs, "use_block_verification", False) or getattr(rs, "fly_window_size", 0):
                return False, False
            if getattr(rs, "enable_adaptive_verification", False):
                return False, False
        if getattr(sp, "compute_nans", False) or getattr(sp, "trace_replay_state", None) is not None:
            return False, False
        if getattr(sp, "return_sampling_mask", False):
            return False, False
        idx = input_batch.idx_mapping_np
        if idx is None or len(idx) == 0:
            return False, False
        ss = sp.sampling_states
        if not bool(np.all(ss.temperature.np[idx] == 0.0)):
            return False, False
        if bool(np.any(sp.needs_logits_processing[idx])):
            return False, False
        from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

        if ss.max_num_logprobs(idx) != NO_LOGPROBS:
            return False, False
        lts = getattr(sp, "logprob_token_ids_state", None)
        if lts is not None and lts.max_num_token_ids(idx) != 0:
            return False, False
        return True, is_k0
    except Exception as e:  # noqa: BLE001 - fail closed
        if not _S.get("arm_warned"):
            _S["arm_warned"] = True
            logger.warning("INKLING_SYNC_FREE: arming check failed (%r); stock head", e)
        return False, False


def _install_runner_hook() -> None:
    if _S["runner_hooked"]:
        return
    _S["runner_hooked"] = True
    try:
        from vllm.v1.worker.gpu import model_runner as mr

        cls = mr.GPUModelRunner
        if getattr(cls, "_inkling_sync_free", False):
            return
        orig = cls.sample

        def sample(self, *args, **kwargs):
            hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            ib = kwargs.get("input_batch", args[1] if len(args) > 1 else None)
            go = kwargs.get("grammar_output", args[2] if len(args) > 2 else None)
            armed, is_k0 = step_armed(self, ib, go)
            _S["armed"] = armed
            _S["armed_k0"] = is_k0
            # Only pass num_tokens for pure-decode, unpadded-token-count batches so cert_head
            # nlive trimming safely falls back to M on prefill, mixed, or padded K=0 batches.
            nt = 0
            nt_raw = 0
            if ib is not None and not getattr(ib, "has_prefill", False):
                nt_pad = int(getattr(ib, "num_tokens_after_padding", 0) or 0)
                nt_raw = int(getattr(ib, "num_tokens", 0) or 0)
                if nt_pad > 0 and nt_raw == nt_pad:
                    nt = nt_pad
            _S["num_tokens"] = nt
            _S["tok"] = None
            _S["gathered"] = None
            try:
                # Zero-copy slice fast path on pure-decode batches where every token produces a logit
                # (sum(num_logits) == num_tokens -> logits_indices == [0, 1, ..., num_tokens - 1]):
                # eliminates the eager vectorized_gather_kernel (~6.2 us/step, 0-ULP bit-exact).
                if (
                    ZERO_COPY_SLICE
                    and armed
                    and go is None
                    and ib is not None
                    and not getattr(ib, "has_prefill", True)
                    and getattr(self, "batch_sharder", None) is None
                    and getattr(self, "adaptive_verification", None) is None
                    and getattr(ib, "dsv41_dead_drafts", None) is None
                    and isinstance(hs, torch.Tensor)
                    and hs.dim() == 2
                    and nt_raw > 0
                    and nt_raw <= int(hs.shape[0])
                    and int(getattr(ib, "num_reqs", 0) or 0) > 0
                ):
                    cu_np = getattr(ib, "cu_num_logits_np", None)
                    nreqs = int(ib.num_reqs)
                    if (
                        cu_np is not None
                        and len(cu_np) == nreqs + 1
                        and int(cu_np[-1]) == nt_raw
                        and (
                            (is_k0 and getattr(self, "sampler", None) is not None)
                            or (
                                not is_k0
                                and getattr(self, "rejection_sampler", None) is not None
                                and getattr(self, "speculator", None) is not None
                            )
                        )
                    ):
                        sample_hs = hs[:nt_raw]
                        ms = getattr(self, "model_state", None)
                        if ms is not None and hasattr(ms, "compute_sample_logits"):
                            logits = ms.compute_sample_logits(sample_hs)
                        else:
                            logits = self.model.compute_logits(sample_hs)
                        if is_k0:
                            sampler_output = self.sampler(logits, ib)
                        else:
                            sampler_output = self.rejection_sampler(
                                logits, ib, self.speculator.draft_logits
                            )
                        return (
                            sampler_output,
                            sampler_output.num_sampled,
                            sampler_output.num_rejected,
                        )
                return orig(self, *args, **kwargs)
            finally:
                _S["armed"] = False
                _S["armed_k0"] = False
                _S["num_tokens"] = 0
                _S["tok"] = None
                _S["gathered"] = None

        cls.sample = sample
        cls._inkling_sync_free = True
        logger.info("INKLING_SYNC_FREE: GPUModelRunner.sample arming hook installed (MAX_M=%d, K0=%d)", MAX_M, int(ALLOW_K0))
        if DIRECT and MODE == "argmax":
            if not _S.get("rs_hooked"):
                _install_rejection_hook()
            if ALLOW_K0 and not _S.get("sp_hooked"):
                _install_sampler_hook()
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_SYNC_FREE: could not hook GPUModelRunner.sample (%r): never armed", e)


def _install_sampler_hook() -> None:
    """Direct greedy bypass for Sampler.__call__ (K=0 Dynamic SD & greedy prefill final chunks)."""
    if _S.get("sp_hooked"):
        return
    try:
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample import sampler as sp_mod
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        cls = sp_mod.Sampler
        if getattr(cls, "_inkling_sf_direct", False):
            _S["sp_hooked"] = True
            return
        orig_call = cls.__call__

        def _direct_sampler_call(self, logits, input_batch):
            tok = _S.get("tok")
            g_info = _S.get("gathered")
            if tok is None and g_info is not None and _S.get("armed_k0", False):
                _S["gathered"] = None
                gathered, M_g, tp_g, split_g = g_info
                if M_g == logits.shape[0]:
                    tok = sf_tok_only(gathered, M_g, tp_g, split_g, pdl=True)
            if tok is not None and _S.get("armed_k0", False):
                _S["tok"] = None
                _S["gathered"] = None
                if (
                    tok.shape[0] == logits.shape[0]
                    and tok.shape[0] == input_batch.num_reqs
                    and input_batch.num_reqs > 0
                ):
                    num_sampled, num_rejected = get_num_sampled_and_rejected(
                        input_batch.seq_lens.new_ones(input_batch.num_reqs),
                        input_batch.seq_lens,
                        input_batch.cu_num_logits,
                        input_batch.idx_mapping,
                        self.req_states.prefill_len.gpu,
                    )
                    return SamplerOutput(
                        sampled_token_ids=tok.view(-1, 1),
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                        num_rejected=num_rejected,
                        sampling_mask_tensors=None,
                    )
            return orig_call(self, logits, input_batch)

        cls.__call__ = _direct_sampler_call
        cls._inkling_sf_direct = True
        _S["sp_hooked"] = True
        logger.info("INKLING_SYNC_FREE: direct greedy Sampler bypass installed (K=0 / non-spec zero [M,V] one-hot)")
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_SYNC_FREE: direct Sampler hook failed (%r); K=0 falls back to one-hot", e)


def _install_rejection_hook() -> None:
    if _S.get("rs_hooked"):
        return
    try:
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample.output import SamplerOutput
        from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_mod

        gather_draft_sampled = rs_mod.gather_draft_sampled

        cls = rs_mod.RejectionSampler
        if getattr(cls, "_inkling_sf_direct", False):
            _S["rs_hooked"] = True
            return
        orig_call = cls.__call__

        def _direct_call(self, logits, input_batch, draft_logits=None):
            g_info = _S.get("gathered")
            tok = _S.get("tok")
            if g_info is not None:
                _S["gathered"] = None
                _S["tok"] = None
                gathered, M_g, tp_g, split_g = g_info
                num_reqs = input_batch.num_reqs
                if M_g == logits.shape[0] and num_reqs > 0:
                    if (
                        FUSE_REJECT
                        and not getattr(input_batch, "has_prefill", True)
                        and input_batch.positions.dim() == 1
                        and input_batch.logits_indices.shape[0] == M_g
                        and input_batch.input_ids.stride(0) == 1
                        and input_batch.positions.stride(0) == 1
                        and input_batch.logits_indices.stride(0) == 1
                    ):
                        sampled = torch.empty(
                            (num_reqs, self.num_speculative_steps + 1),
                            dtype=torch.int64,
                            device=gathered.device,
                        )
                        num_sampled = torch.empty(
                            (num_reqs,), dtype=torch.int32, device=gathered.device
                        )
                        num_rejected = torch.empty(
                            (num_reqs,), dtype=torch.int32, device=gathered.device
                        )
                        k = _kernels()
                        tree_st = None
                        tsr_mod = None
                        if TREE_SPEC and num_reqs == 1:
                            try:
                                from . import tree_spec_runtime as tsr_mod
                            except ImportError:
                                import tree_spec_runtime as tsr_mod  # type: ignore[no-redef]
                            tree_st = tsr_mod.get_active_tree_state(M_g)
                        if tree_st is not None and tsr_mod is not None:
                            k["fused_tree_reject"][(1,)](
                                gathered,
                                M_g,
                                sampled,
                                sampled.stride(0),
                                num_sampled,
                                num_rejected,
                                input_batch.seq_lens,
                                input_batch.input_ids,
                                input_batch.positions,
                                input_batch.logits_indices,
                                input_batch.idx_mapping,
                                self.sampler.req_states.prefill_len.gpu,
                                input_batch.cu_num_logits,
                                tree_st.tree_parent,
                                tree_st.n_live,
                                tree_st.accepted_path,
                                tree_st.accepted_len,
                                tree_st.needs_compact,
                                TP=tp_g,
                                SPLIT=split_g,
                                NP=triton_np2(tp_g * split_g),
                                MAX_M=tree_st.max_m,
                                PDL=True,
                                num_warps=1,
                                launch_pdl=True,
                            )
                            tsr_mod.compact_tree_kv_conv(
                                tree_st,
                                input_batch.positions,
                                use_pdl=True,
                            )
                        else:
                            k["fused_reject"][(num_reqs,)](
                                gathered,
                                M_g,
                                sampled,
                                sampled.stride(0),
                                num_sampled,
                                num_rejected,
                                input_batch.seq_lens,
                                input_batch.input_ids,
                                input_batch.positions,
                                input_batch.logits_indices,
                                input_batch.idx_mapping,
                                self.sampler.req_states.prefill_len.gpu,
                                input_batch.cu_num_logits,
                                TP=tp_g,
                                SPLIT=split_g,
                                NP=triton_np2(tp_g * split_g),
                                PDL=True,
                                num_warps=1,
                                launch_pdl=True,
                            )
                        return SamplerOutput(
                            sampled_token_ids=sampled,
                            logprobs_tensors=None,
                            num_nans=None,
                            num_sampled=num_sampled,
                            num_rejected=num_rejected,
                            sampling_mask_tensors=None,
                        )
                    tok = sf_tok_only(gathered, M_g, tp_g, split_g, pdl=True)
            if tok is not None:
                _S["tok"] = None
                if tok.shape[0] == logits.shape[0] and input_batch.num_reqs > 0:
                    draft_sampled, _pos = gather_draft_sampled(
                        input_batch.input_ids,
                        input_batch.positions,
                        input_batch.logits_indices,
                        input_batch.expanded_idx_mapping,
                        input_batch.expanded_local_pos,
                        self.sampler.req_states.prefill_len.gpu,
                    )
                    num_reqs = input_batch.num_reqs
                    sampled = draft_sampled.new_empty(
                        (num_reqs, self.num_speculative_steps + 1), dtype=torch.int64
                    )
                    num_sampled = sampled.new_empty((num_reqs,), dtype=torch.int32)
                    k = _kernels()
                    k["direct_reject"][(num_reqs,)](
                        sampled,
                        sampled.stride(0),
                        num_sampled,
                        tok,
                        draft_sampled,
                        input_batch.cu_num_logits,
                        num_warps=1,
                    )
                    num_sampled, num_rejected = get_num_sampled_and_rejected(
                        num_sampled,
                        input_batch.seq_lens,
                        input_batch.cu_num_logits,
                        input_batch.idx_mapping,
                        self.sampler.req_states.prefill_len.gpu,
                    )
                    return SamplerOutput(
                        sampled_token_ids=sampled,
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                        num_rejected=num_rejected,
                        sampling_mask_tensors=None,
                    )
            return orig_call(self, logits, input_batch, draft_logits)

        cls.__call__ = _direct_call
        cls._inkling_sf_direct = True
        _S["rs_hooked"] = True
        logger.info("INKLING_SYNC_FREE: direct greedy RejectionSampler bypass installed (zero [M,V] one-hot)")
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_SYNC_FREE: direct RejectionSampler hook failed (%r); falling back to one-hot", e)


def _layout_ok(lp, head, V: int) -> tuple[bool, str]:
    try:
        si = head.shard_indices
        rank, tp = _tp()
        org = int(lp.org_vocab_size)
        if getattr(lp, "logits_as_input", False):
            return False, "logits_as_input"
        if lp.scale != 1.0:
            return False, "logits scale != 1"
        if int(head.tp_size) != tp or tp < 2:
            return False, f"head tp_size {head.tp_size} vs TP {tp}"
        if int(si.num_added_elements) != 0:
            return False, "added (LoRA) vocab"
        if int(si.num_org_vocab_padding) != 0 or int(si.num_org_elements) != V:
            return False, f"per-shard padding (num_org_elements {si.num_org_elements}, V {V})"
        if int(si.org_vocab_start_index) != rank * V:
            return False, f"org_vocab_start_index {si.org_vocab_start_index} != rank*V"
        if not ((tp - 1) * V < org <= tp * V):
            return False, f"org_vocab {org} not inside the last shard of {tp} x {V}"
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"layout probe failed {e!r}"


def _state(model, hidden_states: torch.Tensor):
    st = getattr(model, "_inkling_sf", None)
    if st is not None:
        return st or None
    if _capturing():
        return None
    lp, head = model.logits_processor, model.lm_head
    rank, tp = _tp()
    W = getattr(head, "weight", None)
    V = int(W.shape[0]) if W is not None and W.dim() == 2 else 0
    ok, why = _layout_ok(lp, head, V) if V else (False, "no 2-D lm_head.weight")
    cap = None if lp.soft_cap is None else float(lp.soft_cap)
    fuse = FUSE and cap is not None
    chain_ok = True
    nbad = 0
    if ok and fuse:
        chain_ok, nbad = chain_selftest(cap, hidden_states.device)
        if not chain_ok:
            logger.warning("INKLING_SYNC_FREE: fused soft-cap chain differs from torch on %d bf16 patterns; "
                           "using the torch-op chain on the shard", nbad)
    hd = getattr(lp, "head_dtype", None)
    dt = torch.float32 if hd == torch.float32 else torch.bfloat16
    if ok and (W.dtype != torch.bfloat16 or hidden_states.dtype != torch.bfloat16
               or hd not in (None, torch.bfloat16, torch.float32)):
        ok, why = False, f"head dtype {W.dtype} / head_dtype {hd}"
    agreed = _tp_min([int(ok)])[0] == 1
    st = False
    if agreed:
        org = int(lp.org_vocab_size)
        split = max(1, SPLIT)
        valid = max(1, min(V, org - rank * V))
        st = dict(V=V, tp=tp, rank=rank, org=org, cap=cap, valid=valid, vstart=rank * V,
                  fuse=fuse and chain_ok and dt == torch.bfloat16, split=split, dt=dt)
        st["cs"] = _cert_state(lp, head, W, valid, cap, dt, hidden_states.device) if CERT != "off" and MODE != "off" else None
        if FP32_FUSE and dt == torch.float32:
            if st["cs"] is not None:
                st["fp32_divmode"] = int(st["cs"].divmode)
            elif cap is not None:
                try:
                    from . import cert_head as _ch_dm
                except Exception:  # noqa: BLE001
                    import cert_head as _ch_dm  # type: ignore
                _dm, _, _ = _ch_dm.pick_chain_mode(cap, hidden_states.device)
                st["fp32_divmode"] = int(_dm) if _dm >= 0 else -1
            else:
                st["fp32_divmode"] = -1
        else:
            st["fp32_divmode"] = -1
        dummy = torch.zeros((2, V), dtype=dt, device=hidden_states.device)
        p = sf_local_pairs(dummy, valid, rank * V, cap, split, st["fuse"], fp32_divmode=st["fp32_divmode"])
        g_dum = p.repeat(tp, 1).contiguous()
        sf_onehot(g_dum, 2, tp, org, split, dt)
        sf_tok_only(g_dum, 2, tp, split)
        sf_tok_only(g_dum, 2, tp, split, pdl=True)
        if DIRECT and MODE == "argmax":
            k = _kernels()
            s_dum = torch.empty((1, 8), dtype=torch.int64, device=hidden_states.device)
            ns_dum = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
            t_dum = torch.zeros((2,), dtype=torch.int64, device=hidden_states.device)
            d_dum = torch.zeros((2,), dtype=torch.int32, device=hidden_states.device)
            c_dum = torch.tensor([0, 2], dtype=torch.int32, device=hidden_states.device)
            k["direct_reject"][(1,)](s_dum, 8, ns_dum, t_dum, d_dum, c_dum, num_warps=1)
            if FUSE_REJECT:
                nr_dum = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
                sl_dum = torch.tensor([2], dtype=torch.int32, device=hidden_states.device)
                idx_dum = torch.tensor([0, 1], dtype=torch.int32, device=hidden_states.device)
                pos_dum = torch.tensor([1, 2], dtype=torch.int64, device=hidden_states.device)
                map_dum = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
                pfl_dum = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
                for _sp in sorted({1, split}):
                    _g_sp = torch.zeros((tp * 2 * _sp, 4), dtype=torch.float32, device=hidden_states.device)
                    k["fused_reject"][(1,)](
                        _g_sp, 2, s_dum, 8, ns_dum, nr_dum, sl_dum, d_dum, pos_dum, idx_dum, map_dum, pfl_dum, c_dum,
                        TP=tp, SPLIT=_sp, NP=triton_np2(tp * _sp), PDL=True,
                        num_warps=1, launch_pdl=True,
                    )
                    if TREE_SPEC:
                        try:
                            from . import tree_spec_runtime as _tsr_w
                        except ImportError:
                            import tree_spec_runtime as _tsr_w  # type: ignore[no-redef]
                        _tsr_w.get_tree_target_lib()
                        _tp_dum = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32, device=hidden_states.device)
                        _nl_dum = torch.tensor([2], dtype=torch.int32, device=hidden_states.device)
                        _ap_dum = torch.zeros((8,), dtype=torch.int32, device=hidden_states.device)
                        _al_dum = torch.ones((1,), dtype=torch.int32, device=hidden_states.device)
                        _nc_dum = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
                        k["fused_tree_reject"][(1,)](
                            _g_sp, 2, s_dum, 8, ns_dum, nr_dum, sl_dum, d_dum, pos_dum, idx_dum, map_dum, pfl_dum, c_dum,
                            _tp_dum, _nl_dum, _ap_dum, _al_dum, _nc_dum,
                            TP=tp, SPLIT=_sp, NP=triton_np2(tp * _sp), MAX_M=8, PDL=True,
                            num_warps=1, launch_pdl=True,
                        )
        torch.cuda.synchronize()
        logger.info("INKLING_SYNC_FREE=%s agreed on every rank: V/shard %d, org_vocab %d, rank %d valid %d, "
                    "cap %s, head dtype %s, fused chain %s (selftest: %d/65536 bf16 patterns differ), split %d, MAX_M %d, K0 %d",
                    MODE, V, org, rank, valid, cap, dt, st["fuse"], nbad, split, MAX_M, int(ALLOW_K0))
    else:
        logger.warning("INKLING_SYNC_FREE=%s DISABLED on every rank (this rank: %s)", MODE, why or "ok")
    model._inkling_sf = st
    return st or None


def _cert_state(lp, head, W: torch.Tensor, valid: int, cap, dt, device):
    try:
        from . import cert_head as ch
    except Exception:  # noqa: BLE001
        import cert_head as ch  # type: ignore
    try:
        if dt != torch.float32:
            raise ValueError("the certified head needs the fp32 head (head_dtype float32); this model runs a bf16 head")
        if cap is None:
            raise ValueError("no soft cap")
        mode, nd, dip = ch.pick_chain_mode(cap, device)
        if mode < 0:
            raise ValueError(f"fp32 soft-cap chain not bit-equal to torch / not near-monotone ({nd} values differ, dip {dip})")
        # INKLING_CERT_HEAD_CUDA_ONLY=1 (default ON when CUDA cert_head is enabled and not in check mode):
        # build ONLY the 105.3 MiB CUDA int7_g32 table and skip the redundant 117.4 MiB Triton MXINT8 table
        # (+0.115 GiB/rank saved and faster startup; M==1 or M>16 fall back to exact stock BF16 head).
        cuda_only = (
            os.environ.get("INKLING_CERT_HEAD_CUDA_ONLY", "1").strip().lower() not in ("0", "", "off", "false", "no")
            and CERT != "check"
        )
        cs_cu = None
        ct_cu = None
        try:
            from . import cert_head_cuda_v7 as ch7
            if ch7.ENABLED and ch7._load_lib() is not None:
                rank = _tp()[0]
                ct_v7 = ch7.build_cuda_table_v7(W)
                cand_v7 = ch7.CudaCertStateV7Adapter(ct_v7, W, vreal=valid, vstart=rank * W.shape[0], cap=cap, dip=dip)
                bad_v7 = cand_v7.qualify(16)
                if not bad_v7:
                    cs_cu = cand_v7
                    try:
                        from . import l2pf
                        _head_segs = (
                            (ct_v7.nv1, ct_v7.rt1, (ct_v7.big_s, 131072), (ct_v7.ms_u4, 524288), ct_v7.slabs_u4)
                            if getattr(l2pf, "G2", False)
                            else ct_v7.slabs_u4
                        )
                        l2pf.set_cuda_head_slabs(_head_segs)
                        l2pf.prewarm(_head_segs, kind="head")
                    except Exception:
                        pass
                    logger.info("INKLING_CERT_HEAD_V7=1 ready: S1=%.1f MB (0/15 M=2..16 diffs)",
                                ct_v7.slabs_u4.numel() * ct_v7.slabs_u4.element_size() / 1e6)
                else:
                    logger.warning("INKLING_CERT_HEAD_V7 qualify failed on M=%s; falling back to v6", bad_v7)
        except Exception as _ce7:
            logger.warning("INKLING_CERT_HEAD_V7 init failed (%r); falling back to v6", _ce7)

        if cs_cu is None:
            try:
                from . import cert_head_cuda as chc
                if chc.ENABLED and chc._get_lib() is not None:
                    rank = _tp()[0]
                    ct_cu = chc.build_cuda_table(W, valid)
                    cand_cu = chc.CudaCertState(ct_cu, W, vstart=rank * W.shape[0], cap=cap, dip=dip)
                    _qmax = cand_cu.active_max_m() if hasattr(cand_cu, "active_max_m") else 16
                    bad_cu = cand_cu.qualify(_qmax)
                    bad_le16 = [m for m in bad_cu if m <= 16]
                    if not bad_le16:
                        cs_cu = cand_cu
                        try:
                            from . import l2pf
                            # Wave 0-2 lockstep: nv (100KB) + rt (100KB) + scales[:640KB] (~142 CTAs) + slabs (~7-8MB)
                            _head_segs = (
                                (ct_cu.nv, ct_cu.rt, (ct_cu.scales, 655360), ct_cu.slabs)
                                if getattr(l2pf, "G2", False)
                                else ct_cu.slabs
                            )
                            l2pf.set_cuda_head_slabs(_head_segs)
                            l2pf.prewarm(_head_segs, kind="head")
                            if getattr(l2pf, "C32", False) and _qmax > 16:
                                _head_segs_c32 = (ct_cu.nv, ct_cu.rt, (ct_cu.scales, 1572864), ct_cu.slabs)
                                l2pf.prewarm(_head_segs_c32, kind="head", num_tokens=min(256, _qmax))
                        except Exception:
                            pass
                        logger.info("INKLING_CERT_HEAD_CUDA=1 ready: fmt=%s slabs=%.1f MB max_m=%d (bad_gt16=%s)",
                                    chc.DEFAULT_FMT, ct_cu.slabs.numel() * ct_cu.slabs.element_size() / 1e6,
                                    _qmax, [m for m in bad_cu if m > 16])
                    else:
                        logger.warning("INKLING_CERT_HEAD_CUDA qualify failed on M=%s; falling back to Triton cert_head", bad_le16)
            except Exception as _ce:
                logger.warning("INKLING_CERT_HEAD_CUDA init failed (%r); falling back to Triton cert_head", _ce)

        if cuda_only and cs_cu is not None:
            cs = ch.CertState(None, W, valid, cap, mode, dip)
            cs.ch = ch
            cs.cuda_state = cs_cu
            torch.cuda.empty_cache()
            logger.info("INKLING_CERT_HEAD_CUDA_ONLY=1: skipped 117.4 MiB Triton MXINT8 table; "
                        "CUDA int7_g32 serves M=2..16, stock fp32 head serves M=1 & M>16")
            return cs

        t = ch.get_table(W, valid)
        if t is None:
            raise ValueError("table build failed")
        cs = ch.CertState(t, W, valid, cap, mode, dip)
        cs.ch = ch
        if cs_cu is not None:
            cs.cuda_state = cs_cu
        _prequalify(cs, lp, head, device)
        nq = sum(1 for v in cs.m_ok.values() if v)
        logger.info("INKLING_CERT_HEAD=%s: certified head ready (table %.1f MB, MAXM %d, chain mode %d, chain dip %.3g, "
                    "%d row counts bit-equal to the stock head)", CERT, t.bytes / 1e6, ch.MAXM, mode, dip, nq)
        return cs
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_CERT_HEAD=%s DISABLED on this rank (%r): stock head", CERT, e)
        return None


def _winners(pairs: torch.Tensor, M: int, split: int, vlen: int) -> torch.Tensor:
    return sf_onehot(pairs.contiguous(), M, 1, vlen, pairs.shape[0] // M, torch.float32)[1]


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def _qualify(cs, lp, head, M: int) -> bool:
    ch = cs.ch
    ok = 2 <= M <= ch.MAXM
    if ok:
        g = torch.Generator(device=cs.W.device)
        g.manual_seed(1234 + M)
        for sc in (1.0, 0.3, 4.0):
            x = (torch.randn((M, cs.W.shape[1]), device=cs.W.device, generator=g) * sc).to(torch.bfloat16)
            ref = lp._apply_head(head, x, None)
            if ref.dtype != torch.float32 or ref.shape != (M, cs.W.shape[0]) or \
                    not bool(torch.equal(_bits(ch.exact_all(cs, x)), _bits(ref))):
                ok = False
                break
    cs.m_ok[M] = ok
    return ok


def _prequalify(cs, lp, head, device) -> None:
    ch = cs.ch
    t0 = time.time()
    bad = [M for M in range(2, ch.MAXM + 1) if not _qualify(cs, lp, head, M)]
    torch.cuda.synchronize(device)
    logger.info("INKLING_CERT_HEAD: refine vs torch.mm(out_dtype=fp32) bit-equality for M=2..%d in %.1fs; "
                "NOT bit-equal (stock head): M=1 (always)%s", ch.MAXM, time.time() - t0,
                (", " + ",".join(map(str, bad))) if bad else "")


def _head_local(model, st, lp, head, hidden_states: torch.Tensor):
    cs = st.get("cs")
    M = int(hidden_states.shape[0])
    if cs is None or cs.disabled or M > cs.ch.MAXM or not cs.m_ok.get(M, False):
        return lp._apply_head(head, hidden_states, None)
    try:
        cert = cs.ch.certified_local(cs, hidden_states, num_tokens=_S.get("num_tokens", 0))
    except TypeError:
        cert = cs.ch.certified_local(cs, hidden_states)
    if CERT == "check":
        stock = lp._apply_head(head, hidden_states, None)
        pc = sf_local_pairs(cert, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"])
        ps = sf_local_pairs(stock, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"])
        ctr = cs.check
        if ctr is None:
            ctr = cs.check = torch.zeros((4,), dtype=torch.int64, device=stock.device)
        comp = ~torch.isnan(cert)
        vlen = st["valid"] + st["vstart"]
        ctr.add_(torch.stack([
            torch.ones((), dtype=torch.int64, device=ctr.device),
            torch.full((), M, dtype=torch.int64, device=ctr.device),
            (_winners(pc, M, st["split"], vlen) != _winners(ps, M, st["split"], vlen)).sum(),
            (_bits(cert) != _bits(stock))[comp].sum()]))
        if (cs.calls + 1) % cs.ch.PERIOD == 0 and cs.ch.LOG_EVERY > 0:
            c = [int(v) for v in ctr.tolist()]
            msg = (f"INKLING_CERT_HEAD check rank {st['rank']}: calls {c[0]} rows {c[1]} winner mismatches {c[2]} "
                   f"computed-logit bit mismatches {c[3]}")
            (logger.warning if (c[2] or c[3]) else logger.info)(msg)
        cs.ch.tick(cs, st["rank"])
        return stock
    cs.ch.tick(cs, st["rank"])
    return cert


def _stats_tick(armed: bool) -> None:
    _S["calls"] = _S.get("calls", 0) + 1
    _S["armed_calls"] = _S.get("armed_calls", 0) + int(armed)
    if _S["calls"] % 20000 == 0:
        try:
            rank = _tp()[0]
        except Exception:  # noqa: BLE001
            rank = 0
        if rank == 0:
            logger.info("INKLING_SYNC_FREE: %d head calls, %d armed", _S["calls"], _S["armed_calls"])


def compact_logits(lp, head, st, local: torch.Tensor):
    from vllm.distributed import tensor_model_parallel_all_gather

    M = int(local.shape[0])
    pair = sf_local_pairs(
        local, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"],
        fp32_divmode=st.get("fp32_divmode", -1),
    )
    gathered = tensor_model_parallel_all_gather(pair, dim=0)
    if not _S.get("logged"):
        _S["logged"] = True
        logger.info("INKLING_SYNC_FREE: first armed step (M=%d): %d B/row all-gather replaces the "
                    "%d x %d bf16 logits all-gather + 4 full-vocab elementwise kernels",
                    M, 16 * st["split"], M, local.shape[1])
    return sf_onehot(gathered.contiguous(), M, st["tp"], st["org"], st["split"], st.get("dt", torch.bfloat16))


def _audit(model, st, local: torch.Tensor, hidden_states: torch.Tensor):
    from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax

    full = model.logits_processor(model.lm_head, hidden_states)
    _out, tok = compact_logits(model.logits_processor, model.lm_head, st, local)
    ref = greedy_argmax(full)
    fnan = torch.isnan(full).any(dim=-1)
    ties = (full == full.max(dim=-1, keepdim=True).values).sum(dim=-1) > 1
    ctr = _S.get("ctr")
    if ctr is None:
        ctr = _S["ctr"] = torch.zeros((5,), dtype=torch.int64, device=local.device)
    n = torch.full((), ref.shape[0], dtype=torch.int64, device=ctr.device)
    bad = (ref != tok) & ~fnan
    ctr.add_(torch.stack([torch.ones((), dtype=torch.int64, device=ctr.device), n,
                          bad.sum(), fnan.sum(), ties.sum()]))
    _S["audit_calls"] = _S.get("audit_calls", 0) + 1
    if _S["audit_calls"] % AUDIT_EVERY == 0:
        c = [int(v) for v in ctr.tolist()]
        rank = st["rank"]
        msg = (f"INKLING_SYNC_FREE audit rank {rank}: calls {c[0]} rows {c[1]} winner mismatches {c[2]} "
               f"(NaN rows excluded: {c[3]}; rows with an exact max tie: {c[4]})")
        if c[2]:
            logger.warning(msg + "  MISMATCH: keep the stock gather")
        elif rank == 0:
            logger.info(msg)
    return full


def target_logits(model, hidden_states: torch.Tensor):
    """Hook for Inkling2ForCausalLM.compute_logits. None -> the caller runs the stock path."""
    if MODE == "off":
        return None
    if not _S["installed"]:
        _S["installed"] = True
        _install_runner_hook()
    if _capturing() or not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        return None
    M = int(hidden_states.shape[0])
    lp, head = model.logits_processor, model.lm_head
    armed = bool(_S["armed"])
    is_k0 = bool(_S.get("armed_k0", False))
    st = getattr(model, "_inkling_sf", None)
    if st is None:
        _state(model, hidden_states)
        return None
    if M == 0 or M > MAX_M or hidden_states.dtype != torch.bfloat16:
        return None
    if st is False or not (armed or _S.get("force")):
        _stats_tick(False)
        return None
    if MODE == "audit" or AGREE:
        armed = _tp_min([int(armed)])[0] == 1
        if not armed:
            _stats_tick(False)
            return None
    direct_ok = _S.get("sp_hooked") if is_k0 else _S.get("rs_hooked")
    cs = st.get("cs")
    cs_cu = getattr(cs, "cuda_state", None) if (cs is not None and not cs.disabled) else None
    _cu_max_m = cs_cu.active_max_m() if (cs_cu is not None and hasattr(cs_cu, "active_max_m")) else 16
    if (
        cs_cu is not None
        and DIRECT
        and direct_ok
        and armed
        and not _S.get("force")
        and MODE == "argmax"
        and 2 <= M <= _cu_max_m
    ):
        nlive = None
        if M <= 16:
            try:
                nlive = cs.ch._nlive_for(M, hidden_states.device, num_tokens=_S.get("num_tokens", 0))
            except TypeError:
                nlive = cs.ch._nlive_for(M, hidden_states.device)
        if getattr(cs_cu, "is_v7", False):
            pair = cs_cu.local_pairs(hidden_states, nlive=nlive)
        else:
            from . import cert_head_cuda as chc
            pair = chc.certified_local_pairs(cs_cu, hidden_states, nlive=nlive)
        if pair is not None:
            from vllm.distributed import tensor_model_parallel_all_gather
            _stats_tick(True)
            gathered = tensor_model_parallel_all_gather(pair, dim=0)
            if FUSE_REJECT and not is_k0:
                _S["gathered"] = (gathered.contiguous(), M, st["tp"], 1)
                _S["tok"] = None
            else:
                tok = sf_tok_only(gathered.contiguous(), M, st["tp"], split=1, pdl=True)
                _S["tok"] = tok
                _S["gathered"] = None
            if not _S.get("logged"):
                _S["logged"] = True
                logger.info("INKLING_SYNC_FREE: first armed CUDA cert_head step (M=%d, k0=%d, fuse_reject=%d): 16 B/row (SPLIT=1) AG + direct greedy",
                            M, int(is_k0), int(FUSE_REJECT and not is_k0))
            return torch.empty((M, 1), dtype=st.get("dt", torch.bfloat16), device=hidden_states.device)
    local = _head_local(model, st, lp, head, hidden_states)
    if local.dtype != st.get("dt", torch.bfloat16) or local.stride(1) != 1 or local.shape[1] != st["V"]:
        return None
    _stats_tick(True)
    if MODE == "audit":
        return _audit(model, st, local, hidden_states)
    direct_ok = _S.get("sp_hooked") if is_k0 else _S.get("rs_hooked")
    if DIRECT and direct_ok and armed and not _S.get("force"):
        from vllm.distributed import tensor_model_parallel_all_gather
        M_l = int(local.shape[0])
        pair = sf_local_pairs(
            local, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"],
            fp32_divmode=st.get("fp32_divmode", -1),
        )
        gathered = tensor_model_parallel_all_gather(pair, dim=0)
        if FUSE_REJECT and not is_k0:
            _S["gathered"] = (gathered.contiguous(), M_l, st["tp"], st["split"])
            _S["tok"] = None
        else:
            tok = sf_tok_only(gathered.contiguous(), M_l, st["tp"], st["split"], pdl=True)
            _S["tok"] = tok
            _S["gathered"] = None
        if not _S.get("logged"):
            _S["logged"] = True
            logger.info("INKLING_SYNC_FREE: first armed direct step (M=%d, k0=%d, fuse_reject=%d): %d B/row all-gather + direct greedy %s (no [M,V] one-hot)",
                        M_l, int(is_k0), int(FUSE_REJECT and not is_k0), 16 * st["split"], "sample" if is_k0 else "reject")
        return torch.empty((M_l, 1), dtype=st.get("dt", torch.bfloat16), device=local.device)
    out, _tok = compact_logits(lp, head, st, local)
    return out
