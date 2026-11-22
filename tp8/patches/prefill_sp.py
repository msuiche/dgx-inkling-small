# SPDX-License-Identifier: Apache-2.0
"""Wave 1A: Prefill Sequence Parallelism (INKLING_PREFILL_SP) and Dual-Microbatch
All-Reduce Overlap (INKLING_PREFILL_AR_OVERLAP) for Inkling-2-Small TP8 on GB10 (SM121a).

Design invariants:
  1. 100% opt-in (default OFF) and strictly gated on eager prefill (never fires during
     CUDA graph capture/replay, decode, or small concurrency batches T < MIN_T).
     C=1 decode >= 140 tok/s bit-exact is 100% untouched.
  2. Zero CUDA recompile required for sharded sconv + AttnRes: passes `X = x_with_halo[halo:]`
     and `QStart = query_start[begin:end] - begin` directly into the existing 0-ULP
     `libfused_sconv_ar_prefill.so` (`fused_sconv_ar_prefill_exact_fwd`), whose signed
     `int row = t - lag; tap_ptrs[k] = X + (int64_t)row * stride_x` natively reads the
     3-row halo at negative offsets `-3, -2, -1`! Also ships an exact Triton/PyTorch
     halo fallback when the .so is absent or D != 4608.
  3. Unified Halo + Conv-State Cache Sync: ONE deterministic index plan built once per
     forward pass packs each rank's 3 shard-end halo rows + `prefix_align=256` block-end
     rows + `tail_rows=8` request-end rows into a tiny `[K, H]` tensor (`K=11` -> 101 KiB
     at T=4096, routed automatically over 4-HCA RDMA AG in ~28 us!), giving every rank
     exact 3-row halos AND 100% bit-identical `conv_state.kv_cache` for future chunks,
     decode, verify rollbacks, and prefix-cache hits.
  4. Three SP transport modes:
       - `INKLING_PREFILL_SP=nccl` (or `1`): 8-ch NCCL reduce_scatter + all_gather
         (same 1.75*S wire as AR on all 88 Q=1 sublayers; ~0-1 ULP vs NCCL AR).
       - `INKLING_PREFILL_SP=exact`: stock all_reduce + local shard slice + sharded
         sconv/attn_res + all_gather (0-ULP bit-exact verification canary!).
       - `INKLING_PREFILL_SP=rdma`: fixed-rank-order RDMA reduce_scatter hook (for Wave 1B
         `rdma_ar_ext_s7.so`; gracefully falls back to `nccl` if unavailable).
  5. Two Q=3 global-next handling modes (`INKLING_PREFILL_SP_Q3`):
       - `ag3` (default): keeps `stacked` 100% sharded `[9, T/8, H]` (zero extra weight RAM),
         runs sharded Q=3 sconv+attn_res (8x faster), gathers `[x_q, x_k, x_v]` via 3
         contiguous `all_gather(dim=0)` calls.
       - `repl_kv`: optional helper (155 MiB replicated global `wk_dv/wv_dv/k_dyn/v_dyn`)
         that gathers only `x_q` and exchanges `[S, 132]` K/V projections via `all_to_all`
         (+0.05*S instead of +1.75*S -> effectively 0% extra wire across all 96 sublayers!).
  6. Dual-Microbatch (`T/2 + T/2`) AR Overlap (`INKLING_PREFILL_AR_OVERLAP=1`):
     Splits solo prefill chunks (`num_reqs == 1`, `T >= 8192`, `T % 256 == 0`) into two
     aligned micro-batches A=[0:T/2) and B=[T/2:T) and pipelines compute on the default
     stream with TP collectives on a dedicated high-priority `comm_stream` (strict CUDA
     event + `record_stream` allocator safety; composes cleanly with `INKLING_PREFILL_SP`).
"""

from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

HALO_ROWS = 3

# ---------------------------------------------------------------------------
# Environment configuration (strictly opt-in, default OFF)
# ---------------------------------------------------------------------------
_SP_RAW = os.environ.get("INKLING_PREFILL_SP", "0").strip().lower()
SP_MODE = {
    "1": "nccl",
    "on": "nccl",
    "true": "nccl",
    "yes": "nccl",
    "nccl": "nccl",
    "exact": "exact",
    "rdma": "rdma",
}.get(_SP_RAW, "off")
SP_ENABLED = SP_MODE != "off"
SP_MIN_TOKENS = max(int(os.environ.get("INKLING_PREFILL_SP_MIN_T", "1024") or 1024), 64)
SP_Q3_MODE = os.environ.get("INKLING_PREFILL_SP_Q3", "ag3").strip().lower()
if SP_Q3_MODE not in ("ag3", "repl_kv"):
    SP_Q3_MODE = "ag3"
# INKLING_PREFILL_SP_PREFIX_SYNC:
#   - eager (default): sync halo + prefix_align + tail_rows on every sublayer (k_sync=17 @ 8K)
#   - deferred: sync ONLY 3 halo rows per sublayer (k_sync=3 -> 5.7x less wire, 1-CTA kR3S1 or
#               40x less wire via ring_shift!) + 1 single bulk sync at end of forward (100% 0-ULP
#               prefix-cache exact!)
#   - off: sync ONLY 3 halo rows per sublayer + 1 single end-of-forward rank-7 tail broadcast
#          (for --prefix-cache-retention-interval=0 / cold-prefill; ~0.27 ms bulk sync!)
SP_PREFIX_SYNC = os.environ.get("INKLING_PREFILL_SP_PREFIX_SYNC", "eager").strip().lower()
if SP_PREFIX_SYNC not in ("eager", "deferred", "off"):
    SP_PREFIX_SYNC = "eager"
# INKLING_PREFILL_SP_RS_MODE (prefill-only ReduceScatter transport, default nccl):
#   nccl | rhd_int8 | rhd_fp8 | fp8_a2a | int8_a2a | int8_col | bf12 | rdma_bf16 | rdma_int8_col | rdma_bf12
SP_RS_MODE = os.environ.get("INKLING_PREFILL_SP_RS_MODE", "nccl").strip().lower()
if SP_RS_MODE not in (
    "nccl",
    "rhd_int8",
    "rhd_fp8",
    "fp8_a2a",
    "int8_a2a",
    "int8_col",
    "bf12",
    "rdma_bf16",
    "rdma_int8_col",
    "rdma_bf12",
):
    SP_RS_MODE = "nccl"
SP_RS_MIN_T = max(int(os.environ.get("INKLING_PREFILL_SP_RS_MIN_T", "2048") or 2048), 1024)
# INKLING_PREFILL_SP_AG_MODE (prefill-only AllGather transport, default nccl):
#   nccl | rhd_int8 | rhd_fp8 (zero-incast 3-step Recursive Doubling AllGather, T >= 2048!)
SP_AG_MODE = os.environ.get("INKLING_PREFILL_SP_AG_MODE", "nccl").strip().lower()
if SP_AG_MODE not in ("nccl", "rhd_int8", "rhd_fp8"):
    SP_AG_MODE = "nccl"
SP_AG_MIN_T = max(int(os.environ.get("INKLING_PREFILL_SP_AG_MIN_T", "2048") or 2048), 1024)
# Opt-in tail prefill chunk SP/DBO padding (default OFF; T >= 2048 with T % 8 != 0 -> pad to SP/DBO!)
SP_PAD_TAIL = os.environ.get("INKLING_PREFILL_SP_PAD_TAIL", "0").strip().lower() in (
    "1", "on", "true", "yes",
)
SP_PAD_MIN_T = max(int(os.environ.get("INKLING_PREFILL_SP_PAD_MIN_T", "2048") or 2048), 1024)
# Opt-in zero-sync SP plan builder + sparse discard-block write skip (default OFF)
SP_ZERO_SYNC = os.environ.get("INKLING_PREFILL_SP_ZERO_SYNC", "0").strip().lower() in (
    "1", "on", "true", "yes",
)
SP_DEBUG = os.environ.get("INKLING_PREFILL_SP_DEBUG", "0").strip().lower() in (
    "1", "on", "true", "yes",
)

_DBO_RAW = os.environ.get("INKLING_PREFILL_AR_OVERLAP", "0").strip().lower()
AR_OVERLAP_ENABLED = _DBO_RAW in ("1", "on", "true", "yes", "exact")
AR_OVERLAP_MIN_TOKENS = max(
    int(os.environ.get("INKLING_PREFILL_AR_OVERLAP_MIN_T", "8192") or 8192), 256
)
DBO_PIPE_SP = os.environ.get("INKLING_PREFILL_DBO_PIPE_SP", "1").strip().lower() in (
    "1", "on", "true", "yes",
)
# Opt-in bounded cross-stream inflight layer pacing for DBO (default 0 = off; set 2 when
# INKLING_PF_GLOBAL_ZERO_SYNC=1 + INKLING_PREFILL_SP_ZERO_SYNC=1 to cap record_stream limbo
# at <= 0.9 GB instead of 21.75 GB across all 48 layers, zero GPU bubble!).
DBO_MAX_INFLIGHT_LAYERS = max(
    int(os.environ.get("INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS", "0") or 0), 0
)
SP_FP8_AG = os.environ.get("INKLING_PREFILL_SP_FP8_AG", "0").strip().lower() in (
    "1", "on", "true", "yes",
)
_SP_FP8_AG_MOE_RAW = os.environ.get("INKLING_PREFILL_SP_FP8_AG_MOE", "1").strip().lower()
SP_FP8_AG_MOE_MODE = (
    "direct"
    if _SP_FP8_AG_MOE_RAW == "direct"
    else ("dequant" if _SP_FP8_AG_MOE_RAW in ("1", "on", "true", "yes", "dequant") else "bf16")
)
SP_FP8_AG_MOE = SP_FP8_AG_MOE_MODE != "bf16"
SP_FP8_AG_KV = os.environ.get("INKLING_PREFILL_SP_FP8_AG_KV", "1").strip().lower() in (
    "1", "on", "true", "yes",
)
# Opt-in Hybrid NIC-First Interleaved DBO+SP (default OFF):
# Moves Attn-half `sconv+AttnRes` compute off `comm_stream` onto idle `compute_stream`
# so NICs stay 100% back-to-back (`RS(MoE_B)->RS(Attn_A)->AG(MoE_B)->AG(Attn_A)`),
# saving ~1.8-2.2 ms/layer (~4.6-6.5 s @ 1M, 0-ULP identical, stacks with RHD!).
DBO_INTERLEAVE = os.environ.get("INKLING_PREFILL_DBO_INTERLEAVE", "0").strip().lower() in (
    "1", "on", "true", "yes",
)

_LOGGED_ONCE: set[str] = set()


def _log_once(key: str, msg: str) -> None:
    if key not in _LOGGED_ONCE:
        _LOGGED_ONCE.add(key)
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pure-math sync-mask & plan construction (shared with CPU unit tests!)
# ---------------------------------------------------------------------------
def _real_slot_mask(slot_mapping: torch.Tensor, n_cache: int) -> torch.Tensor:
    """Return `[T]` bool mask: True where `slot >= 0` AND physical block `slot // n_cache`
    is NOT the shared `sparse_prefill_discard` write-only block (which maps `> n_cache`
    distinct tokens in the chunk and is never read by any future chunk or prefix hit)."""
    valid = slot_mapping >= 0
    if not bool(valid.any().item()):
        return valid
    v_blks = (slot_mapping[valid] // n_cache).to(torch.int64)
    _, inv, counts = torch.unique(v_blks, return_inverse=True, return_counts=True)
    real = torch.zeros_like(valid)
    real[valid] = counts[inv] <= n_cache
    return real


def compute_sp_sync_mask(
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_idx: torch.Tensor,
    query_start: torch.Tensor,
    world_size: int,
    prefix_align: int,
    tail_rows: int,
    halo_rows: int = HALO_ROWS,
    real_slot_mask: torch.Tensor | None = None,
    include_halo: bool = True,
) -> torch.Tensor:
    """Deterministic [T] bool mask (identical on all TP ranks, zero comms!) marking:
      1. Shard-end halo source rows (when `include_halo=True`): local idx >= shard_size - halo_rows.
      2. Prefix-cache boundary rows: real slot & (pos % prefix_align >= prefix_align - halo_rows).
      3. Request-tail rows: last `tail_rows` live tokens of every request in the batch
         (so next chunk, decode, verify rollback, and prefix-cache hits see identical
         `conv_state.kv_cache` on all ranks).
    """
    num_tokens = int(positions.shape[0])
    assert num_tokens % world_size == 0 and num_tokens > 0
    shard_size = num_tokens // world_size
    assert shard_size >= halo_rows
    device = positions.device

    # 1. Shard-end halo sources: last `halo_rows` of every rank's shard.
    if include_halo:
        local_pos = torch.arange(num_tokens, device=device, dtype=torch.int64) % shard_size
        is_halo_src = local_pos >= (shard_size - halo_rows)
    else:
        is_halo_src = torch.zeros(num_tokens, dtype=torch.bool, device=device)

    valid_slot = slot_mapping >= 0
    cache_slot = valid_slot if real_slot_mask is None else (valid_slot & real_slot_mask)

    # 2. Prefix-cache hit boundary tails: last `halo_rows` before each `prefix_align` boundary.
    if prefix_align > 0:
        is_prefix_tail = cache_slot & (
            (positions % prefix_align) >= (prefix_align - halo_rows)
        )
    else:
        is_prefix_tail = torch.zeros(num_tokens, dtype=torch.bool, device=device)

    # 3. Request tails: last `tail_rows` live tokens of every request (handles multi-req & pad tails).
    req_end = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    req_end[-1] = valid_slot[-1]
    if num_tokens > 1:
        boundary = (
            (seq_idx[1:] != seq_idx[:-1])
            | (query_start[1:] != query_start[:-1])
            | (valid_slot[:-1] & (~valid_slot[1:]))
        )
        req_end[:-1] = boundary & valid_slot[:-1]

    is_req_tail = req_end.clone()
    max_d = min(int(tail_rows), num_tokens)
    for d in range(1, max_d):
        same_req = (
            (seq_idx[:-d] == seq_idx[d:])
            & (query_start[:-d] == query_start[d:])
            & valid_slot[:-d]
        )
        is_req_tail[:-d] |= req_end[d:] & same_req
    is_req_tail &= cache_slot

    return is_halo_src | is_prefix_tail | is_req_tail


@dataclass
class PrefillSPPlan:
    """Precomputed once per forward pass and reused across all 96 sublayers."""

    num_tokens: int
    world_size: int
    rank: int
    shard_size: int
    begin: int
    end: int
    halo: int  # 0 for rank 0, HALO_ROWS (3) for ranks 1..world_size-1
    has_cache: bool
    n_cache: int  # conv_state.kv_cache.shape[2] (4)
    # Sharded metadata views (length = shard_size)
    positions_s: torch.Tensor
    seq_idx_s: torch.Tensor | None
    slot_mapping_s: torch.Tensor | None
    qstart_local: torch.Tensor | None  # int32 [shard_size] = query_start[begin:end] - begin
    # Unified halo + conv_state sync indices (identical K across all ranks)
    k_sync: int
    my_pack_idx: torch.Tensor | None  # [k_sync] int64 local indices in [0, shard_size)
    halo_gather_idx: torch.Tensor | None  # [halo] int64 indices into [world_size * k_sync]
    remote_gather_idx: torch.Tensor | None  # [M] int64 indices into [world_size * k_sync]
    remote_dst_blk: torch.Tensor | None  # [M] int64 block IDs in conv_state.kv_cache
    remote_dst_off: torch.Tensor | None  # [M] int64 slot-in-block offsets in [0, n_cache)
    # Deferred/halo-only sync fields (`INKLING_PREFILL_SP_PREFIX_SYNC=deferred|off`):
    halo_only: bool = False  # True => k_sync == 3; rank r only needs rank (r-1)%W's 3 rows!
    deferred_slots: torch.Tensor | None = None  # [M_def] int64 real slots in chronological order
    deferred_ranks: torch.Tensor | None = None  # [M_def] int64 owning rank 0..W-1 per deferred slot
    real_written_slots: torch.Tensor | None = None  # [M_wr] int64 real slots written by this plan
    # Zero-sync GPU SP fields (`INKLING_PREFILL_SP_ZERO_SYNC=1`, zero DtoH sync at chunk entry!):
    discard_blk_dev: torch.Tensor | None = None  # [1] int64 GPU scalar (-1 if dense, else discard_blk)
    rank0_last3_slots: torch.Tensor | None = None  # [3] int64 GPU slots for rank-0 halo scatter
    lazy_def_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None = None

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for t in (
            self.positions_s,
            self.seq_idx_s,
            self.slot_mapping_s,
            self.qstart_local,
            self.my_pack_idx,
            self.halo_gather_idx,
            self.remote_gather_idx,
            self.remote_dst_blk,
            self.remote_dst_off,
            self.discard_blk_dev,
            self.rank0_last3_slots,
        ):
            if t is not None:
                t.record_stream(stream)

    @classmethod
    def build(
        cls,
        num_tokens: int,
        world_size: int,
        rank: int,
        positions: torch.Tensor,
        conv_md: Any | None,
        n_cache: int,
        prefix_align: int,
        tail_rows: int,
        prefix_sync_mode: str = "eager",
    ) -> PrefillSPPlan:
        assert num_tokens % world_size == 0
        shard_size = num_tokens // world_size
        begin = rank * shard_size
        end = begin + shard_size
        halo = 0 if rank == 0 else HALO_ROWS
        positions_s = positions[begin:end].contiguous()

        if conv_md is None:
            return cls(
                num_tokens=num_tokens,
                world_size=world_size,
                rank=rank,
                shard_size=shard_size,
                begin=begin,
                end=end,
                halo=halo,
                has_cache=False,
                n_cache=n_cache,
                positions_s=positions_s,
                seq_idx_s=None,
                slot_mapping_s=None,
                qstart_local=None,
                k_sync=0,
                my_pack_idx=None,
                halo_gather_idx=None,
                remote_gather_idx=None,
                remote_dst_blk=None,
                remote_dst_off=None,
                halo_only=False,
                deferred_slots=None,
                deferred_ranks=None,
                real_written_slots=None,
            )

        device = positions.device
        slot_mapping = conv_md.slot_mapping[:num_tokens]
        seq_idx = conv_md.seq_idx[:num_tokens]
        query_start = conv_md.query_start[:num_tokens]

        seq_idx_s = seq_idx[begin:end].contiguous()
        slot_mapping_s = slot_mapping[begin:end].contiguous()
        qstart_local = (query_start[begin:end] - begin).to(torch.int32).contiguous()

        halo_only = prefix_sync_mode in ("deferred", "off")
        eff_prefix_align = 0 if prefix_sync_mode == "off" else prefix_align
        if halo_only and SP_ZERO_SYNC and positions.is_cuda:
            # Zero-sync GPU plan build (0 `.item()`, 0 `torch.unique`, 0 `torch.nonzero`!):
            # Detects sparse _discard_block strictly on GPU, sets rank-0 last-3 slots on GPU,
            # and defers DeferredConvSyncPlan building until end-of-forward (1 single batched D2H!).
            from . import prefill_sp_rs as _sp_rs

            discard_blk_dev = _sp_rs.detect_discard_block_gpu(slot_mapping, n_cache)
            k_sync = HALO_ROWS
            my_pack_idx = torch.arange(
                shard_size - HALO_ROWS, shard_size, dtype=torch.int64, device=device
            ).contiguous()
            halo_gather_idx = None
            if halo > 0:
                prev_base = (rank - 1) * HALO_ROWS
                halo_gather_idx = torch.arange(
                    prev_base, prev_base + HALO_ROWS, dtype=torch.int64, device=device
                )
            rank0_last3_slots = None
            if rank == 0:
                last_start = (world_size - 1) * shard_size + (shard_size - HALO_ROWS)
                rank0_last3_slots = (
                    slot_mapping[last_start : last_start + HALO_ROWS].to(torch.int64).contiguous()
                )
            return cls(
                num_tokens=num_tokens,
                world_size=world_size,
                rank=rank,
                shard_size=shard_size,
                begin=begin,
                end=end,
                halo=halo,
                has_cache=True,
                n_cache=n_cache,
                positions_s=positions_s,
                seq_idx_s=seq_idx_s,
                slot_mapping_s=slot_mapping_s,
                qstart_local=qstart_local,
                k_sync=k_sync,
                my_pack_idx=my_pack_idx,
                halo_gather_idx=halo_gather_idx,
                remote_gather_idx=None,
                remote_dst_blk=None,
                remote_dst_off=None,
                halo_only=True,
                deferred_slots=None,
                deferred_ranks=None,
                real_written_slots=None,
                discard_blk_dev=discard_blk_dev,
                rank0_last3_slots=rank0_last3_slots,
                lazy_def_meta=(
                    positions[:num_tokens],
                    slot_mapping,
                    seq_idx,
                    query_start,
                    eff_prefix_align,
                    tail_rows,
                ),
            )

        real_slots = _real_slot_mask(slot_mapping, n_cache)
        full_sync_mask = compute_sp_sync_mask(
            positions[:num_tokens],
            slot_mapping,
            seq_idx,
            query_start,
            world_size=world_size,
            prefix_align=eff_prefix_align,
            tail_rows=tail_rows,
            halo_rows=HALO_ROWS,
            real_slot_mask=real_slots,
        )

        if halo_only:
            # Per-sublayer sync carries ONLY the 3 shard-end halo rows (k_sync=3 -> 27 KiB,
            # triggers 1-CTA kR3S1 or 1-to-1 ring_shift!). Rank 0 scatters rank W-1's last 3
            # real rows (e.g. DBO A[-3:] for B[0..2]) immediately; all remaining real prefix
            # boundaries & request tails are synced ONCE at end of forward!
            k_sync = HALO_ROWS
            my_pack_idx = torch.arange(
                shard_size - HALO_ROWS, shard_size, dtype=torch.int64, device=device
            ).contiguous()
            halo_gather_idx = None
            if halo > 0:
                prev_base = (rank - 1) * HALO_ROWS
                halo_gather_idx = torch.arange(
                    prev_base, prev_base + HALO_ROWS, dtype=torch.int64, device=device
                )

            remote_gather_idx = None
            remote_dst_blk = None
            remote_dst_off = None
            if rank == 0:
                last_start = (world_size - 1) * shard_size + (shard_size - HALO_ROWS)
                last_slots = slot_mapping[last_start : last_start + HALO_ROWS]
                last_real = real_slots[last_start : last_start + HALO_ROWS]
                if bool(last_real.any().item()):
                    r_base = (world_size - 1) * HALO_ROWS
                    remote_gather_idx = (
                        r_base + torch.arange(HALO_ROWS, dtype=torch.int64, device=device)[last_real]
                    ).contiguous()
                    r_slots = last_slots[last_real].to(torch.int64)
                    remote_dst_blk = (r_slots // n_cache).contiguous()
                    remote_dst_off = (r_slots % n_cache).contiguous()

            # Deferred rows: strictly real (non-discard) prefix-boundary & request-tail slots
            # in chronological order, with ZERO spurious shard-end halo rows when S % align != 0.
            def_mask = compute_sp_sync_mask(
                positions[:num_tokens],
                slot_mapping,
                seq_idx,
                query_start,
                world_size=world_size,
                prefix_align=eff_prefix_align,
                tail_rows=tail_rows,
                halo_rows=HALO_ROWS,
                real_slot_mask=real_slots,
                include_halo=False,
            )
            def_idx = torch.nonzero(def_mask, as_tuple=False).flatten()
            deferred_slots = slot_mapping[def_idx].to(torch.int64).contiguous()
            deferred_ranks = (def_idx // shard_size).to(torch.int64).contiguous()
            real_written_slots = slot_mapping[real_slots].to(torch.int64).contiguous()

            return cls(
                num_tokens=num_tokens,
                world_size=world_size,
                rank=rank,
                shard_size=shard_size,
                begin=begin,
                end=end,
                halo=halo,
                has_cache=True,
                n_cache=n_cache,
                positions_s=positions_s,
                seq_idx_s=seq_idx_s,
                slot_mapping_s=slot_mapping_s,
                qstart_local=qstart_local,
                k_sync=k_sync,
                my_pack_idx=my_pack_idx,
                halo_gather_idx=halo_gather_idx,
                remote_gather_idx=remote_gather_idx,
                remote_dst_blk=remote_dst_blk,
                remote_dst_off=remote_dst_off,
                halo_only=True,
                deferred_slots=deferred_slots,
                deferred_ranks=deferred_ranks,
                real_written_slots=real_written_slots,
            )

        sync_2d = full_sync_mask.view(world_size, shard_size)
        counts = sync_2d.sum(dim=1).tolist()
        k_sync = max(int(c) for c in counts)
        assert k_sync >= HALO_ROWS

        # Build deterministic per-rank packed index list (padded to k_sync with shard_size-1).
        # Because local rows [shard_size - 3, shard_size) are always True in sync_2d[k],
        # the last 3 valid entries of rank k's ascending nonzero list are ALWAYS
        # (shard_size - 3, shard_size - 2, shard_size - 1)!
        per_rank_local: list[torch.Tensor] = []
        remote_src_chunks: list[torch.Tensor] = []
        remote_slots_chunks: list[torch.Tensor] = []

        for k in range(world_size):
            nz = torch.nonzero(sync_2d[k], as_tuple=False).flatten()  # ascending int64
            cnt = int(nz.numel())
            if cnt < k_sync:
                pad = torch.full(
                    (k_sync - cnt,), shard_size - 1, dtype=torch.int64, device=device
                )
                packed_k = torch.cat([nz, pad], dim=0)
            else:
                packed_k = nz
            per_rank_local.append(packed_k)

            if k != rank and cnt > 0:
                global_idx = k * shard_size + nz
                slots_k = slot_mapping[global_idx]
                live_k = real_slots[global_idx]
                if bool(live_k.any().item()):
                    gather_pos = (
                        k * k_sync
                        + torch.arange(cnt, dtype=torch.int64, device=device)[live_k]
                    )
                    remote_src_chunks.append(gather_pos)
                    remote_slots_chunks.append(slots_k[live_k])

        my_pack_idx = per_rank_local[rank].contiguous()

        halo_gather_idx = None
        if halo > 0:
            prev_cnt = int(counts[rank - 1])
            prev_base = (rank - 1) * k_sync + (prev_cnt - halo)
            halo_gather_idx = torch.arange(
                prev_base, prev_base + halo, dtype=torch.int64, device=device
            )

        if remote_src_chunks:
            remote_gather_idx = torch.cat(remote_src_chunks, dim=0).contiguous()
            remote_slots = torch.cat(remote_slots_chunks, dim=0).to(torch.int64)
            remote_dst_blk = (remote_slots // n_cache).contiguous()
            remote_dst_off = (remote_slots % n_cache).contiguous()
        else:
            remote_gather_idx = None
            remote_dst_blk = None
            remote_dst_off = None

        return cls(
            num_tokens=num_tokens,
            world_size=world_size,
            rank=rank,
            shard_size=shard_size,
            begin=begin,
            end=end,
            halo=halo,
            has_cache=True,
            n_cache=n_cache,
            positions_s=positions_s,
            seq_idx_s=seq_idx_s,
            slot_mapping_s=slot_mapping_s,
            qstart_local=qstart_local,
            k_sync=k_sync,
            my_pack_idx=my_pack_idx,
            halo_gather_idx=halo_gather_idx,
            remote_gather_idx=remote_gather_idx,
            remote_dst_blk=remote_dst_blk,
            remote_dst_off=remote_dst_off,
            halo_only=False,
            deferred_slots=None,
            deferred_ranks=None,
            real_written_slots=None,
        )


@dataclass
class DeferredConvSyncPlan:
    """Single end-of-forward bulk sync plan for all 48 layers' real prefix-boundary &
    request-tail `conv_state.kv_cache` rows (used when `INKLING_PREFILL_SP_PREFIX_SYNC`
    is `deferred` or `off`, allowing per-sublayer `k_sync` to shrink from 17 -> 3!)."""

    world_size: int
    rank: int
    k_def: int
    single_sender_rank: int  # >= 0 when only 1 rank has deferred rows (uses 1x broadcast!)
    my_src_blk: torch.Tensor  # [k_def] int64
    my_src_off: torch.Tensor  # [k_def] int64
    remote_gather_idx: torch.Tensor | None  # [M] int64 into [W * k_def] (or [k_def] for broadcast)
    remote_dst_blk: torch.Tensor | None  # [M] int64
    remote_dst_off: torch.Tensor | None  # [M] int64

    @classmethod
    def _from_lazy_plans_np(cls, lazy_plans: list[PrefillSPPlan]) -> DeferredConvSyncPlan | None:
        """Build `DeferredConvSyncPlan` at end-of-forward with ONE single batched D2H copy
        and fast NumPy (eliminates 14 chunk-entry `cudaStreamSynchronize` stalls!)."""
        p0 = lazy_plans[0]
        W, rank, n_cache = p0.world_size, p0.rank, p0.n_cache
        device = p0.positions_s.device

        # Pack all plans' [positions, slot_mapping, seq_idx, query_start] into 1 contiguous D2H transfer!
        cols: list[torch.Tensor] = []
        lengths: list[int] = []
        for p in lazy_plans:
            assert p.lazy_def_meta is not None
            pos_t, sm_t, seq_t, qs_t, _pa, _tr = p.lazy_def_meta
            T_i = int(pos_t.shape[0])
            lengths.append(T_i)
            cols.append(
                torch.stack(
                    (
                        pos_t.to(torch.int64),
                        sm_t.to(torch.int64),
                        seq_t.to(torch.int64),
                        qs_t.to(torch.int64),
                    ),
                    dim=0,
                )
            )
        packed_cpu = torch.cat(cols, dim=1).cpu().numpy()  # [4, sum(T)] int64

        per_plan_def: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        off = 0
        for p, T_i in zip(lazy_plans, lengths):
            assert p.lazy_def_meta is not None
            _pa = p.lazy_def_meta[4]
            _tr = p.lazy_def_meta[5]
            pos = packed_cpu[0, off : off + T_i]
            sm = packed_cpu[1, off : off + T_i]
            seq = packed_cpu[2, off : off + T_i]
            qs = packed_cpu[3, off : off + T_i]
            off += T_i

            valid = sm >= 0
            real = np.zeros(T_i, dtype=bool)
            if valid.any():
                v_blks = sm[valid] // n_cache
                _, inv, counts = np.unique(v_blks, return_inverse=True, return_counts=True)
                real[valid] = counts[inv] <= n_cache
            cache_slot = valid & real
            if _pa > 0:
                is_prefix = cache_slot & ((pos % _pa) >= (_pa - HALO_ROWS))
            else:
                is_prefix = np.zeros(T_i, dtype=bool)

            req_end = np.zeros(T_i, dtype=bool)
            if T_i > 0:
                req_end[-1] = valid[-1]
            if T_i > 1:
                boundary = (
                    (seq[1:] != seq[:-1])
                    | (qs[1:] != qs[:-1])
                    | (valid[:-1] & (~valid[1:]))
                )
                req_end[:-1] = boundary & valid[:-1]
            is_req_tail = req_end.copy()
            for d in range(1, min(int(_tr), T_i)):
                same_req = (seq[:-d] == seq[d:]) & (qs[:-d] == qs[d:]) & valid[:-d]
                is_req_tail[:-d] |= req_end[d:] & same_req
            is_req_tail &= cache_slot

            def_idx = np.flatnonzero(is_prefix | is_req_tail)
            s_i = sm[def_idx]
            r_i = def_idx // p.shard_size
            real_wr = sm[real]
            per_plan_def.append((s_i, r_i, real_wr))

        last_owner: dict[int, int] = {}
        n_plans = len(per_plan_def)
        for pi, (s_i, r_i, _rw) in enumerate(per_plan_def):
            if s_i.size == 0:
                continue
            if pi + 1 < n_plans:
                keep = np.ones(s_i.shape[0], dtype=bool)
                for pj in range(pi + 1, n_plans):
                    later = per_plan_def[pj][2]
                    if later.size > 0:
                        keep &= ~np.isin(s_i, later)
                s_i = s_i[keep]
                r_i = r_i[keep]
            for s_val, r_val in zip(s_i.tolist(), r_i.tolist()):
                last_owner[int(s_val)] = int(r_val)

        if not last_owner:
            return None

        buckets: list[list[int]] = [[] for _ in range(W)]
        for s_val, r_val in last_owner.items():
            buckets[r_val].append(s_val)

        counts = [len(b) for b in buckets]
        k_def = max(counts)
        if k_def == 0:
            return None

        active_ranks = [k for k, c in enumerate(counts) if c > 0]
        single_sender = active_ranks[0] if len(active_ranks) == 1 else -1

        my_list = list(buckets[rank])
        if len(my_list) < k_def:
            pad_val = my_list[-1] if my_list else 0
            my_list.extend([pad_val] * (k_def - len(my_list)))
        my_padded = torch.tensor(my_list, dtype=torch.int64, device=device)
        my_src_blk = (my_padded // n_cache).contiguous()
        my_src_off = (my_padded % n_cache).contiguous()

        remote_src_list: list[int] = []
        remote_dst_list: list[int] = []
        for k in range(W):
            if k == rank or counts[k] == 0:
                continue
            base = 0 if single_sender >= 0 else k * k_def
            remote_src_list.extend(range(base, base + counts[k]))
            remote_dst_list.extend(buckets[k])

        if remote_src_list:
            remote_gather_idx = torch.tensor(remote_src_list, dtype=torch.int64, device=device)
            r_slots = torch.tensor(remote_dst_list, dtype=torch.int64, device=device)
            remote_dst_blk = (r_slots // n_cache).contiguous()
            remote_dst_off = (r_slots % n_cache).contiguous()
        else:
            remote_gather_idx = remote_dst_blk = remote_dst_off = None

        return cls(
            world_size=W,
            rank=rank,
            k_def=k_def,
            single_sender_rank=single_sender,
            my_src_blk=my_src_blk,
            my_src_off=my_src_off,
            remote_gather_idx=remote_gather_idx,
            remote_dst_blk=remote_dst_blk,
            remote_dst_off=remote_dst_off,
        )

    @classmethod
    def from_plans(cls, plans: list[PrefillSPPlan]) -> DeferredConvSyncPlan | None:
        lazy_plans = [p for p in plans if p.has_cache and p.lazy_def_meta is not None]
        if lazy_plans:
            return cls._from_lazy_plans_np(lazy_plans)
        valid_plans = [
            p
            for p in plans
            if p.has_cache and p.deferred_slots is not None and p.deferred_ranks is not None
        ]
        if not valid_plans:
            return None
        p0 = valid_plans[0]
        W, rank, n_cache = p0.world_size, p0.rank, p0.n_cache
        device = p0.positions_s.device

        # Chronologically merge deferred writes across `[plan_a, plan_b]`:
        # 1) Drop any deferred slot in an earlier plan `p_i` that was overwritten by ANY
        #    real write in a later plan `p_j` (`j > i`) (handles `T_a % 256 != 0` mid-block splits!).
        # 2) Deduplicate repeated writes to the same `(blk, off)` (e.g. `tail_rows=8` on `n_cache=4`)
        #    by keeping ONLY the latest writer rank so every rank holds the final value with zero
        #    write-order hazard!
        last_owner: dict[int, int] = {}
        n_plans = len(valid_plans)
        for pi, p in enumerate(valid_plans):
            s_i = p.deferred_slots
            r_i = p.deferred_ranks
            assert s_i is not None and r_i is not None
            if s_i.numel() == 0:
                continue
            if pi + 1 < n_plans:
                keep = torch.ones_like(s_i, dtype=torch.bool)
                for pj in range(pi + 1, n_plans):
                    later = valid_plans[pj].real_written_slots
                    if later is not None and later.numel() > 0:
                        keep &= ~torch.isin(s_i, later)
                s_i = s_i[keep]
                r_i = r_i[keep]
            if s_i.numel() > 0:
                for s_val, r_val in zip(s_i.tolist(), r_i.tolist()):
                    last_owner[int(s_val)] = int(r_val)

        if not last_owner:
            return None

        buckets: list[list[int]] = [[] for _ in range(W)]
        for s_val, r_val in last_owner.items():
            buckets[r_val].append(s_val)

        per_rank_slots: list[torch.Tensor] = [
            torch.tensor(b, dtype=torch.int64, device=device) for b in buckets
        ]
        counts: list[int] = [len(b) for b in buckets]
        k_def = max(counts)
        if k_def == 0:
            return None

        active_ranks = [k for k, c in enumerate(counts) if c > 0]
        single_sender = active_ranks[0] if len(active_ranks) == 1 else -1

        my_slots = per_rank_slots[rank]
        if my_slots.numel() < k_def:
            pad_val = int(my_slots[-1].item()) if my_slots.numel() > 0 else 0
            pad = torch.full((k_def - my_slots.numel(),), pad_val, dtype=torch.int64, device=device)
            my_padded = torch.cat([my_slots, pad], dim=0)
        else:
            my_padded = my_slots
        my_src_blk = (my_padded // n_cache).contiguous()
        my_src_off = (my_padded % n_cache).contiguous()

        remote_src_chunks: list[torch.Tensor] = []
        remote_dst_chunks: list[torch.Tensor] = []
        for k in range(W):
            if k == rank or counts[k] == 0:
                continue
            base = 0 if single_sender >= 0 else k * k_def
            remote_src_chunks.append(base + torch.arange(counts[k], dtype=torch.int64, device=device))
            remote_dst_chunks.append(per_rank_slots[k])

        if remote_src_chunks:
            remote_gather_idx = torch.cat(remote_src_chunks, dim=0).contiguous()
            r_slots = torch.cat(remote_dst_chunks, dim=0)
            remote_dst_blk = (r_slots // n_cache).contiguous()
            remote_dst_off = (r_slots % n_cache).contiguous()
        else:
            remote_gather_idx = remote_dst_blk = remote_dst_off = None

        return cls(
            world_size=W,
            rank=rank,
            k_def=k_def,
            single_sender_rank=single_sender,
            my_src_blk=my_src_blk,
            my_src_off=my_src_off,
            remote_gather_idx=remote_gather_idx,
            remote_dst_blk=remote_dst_blk,
            remote_dst_off=remote_dst_off,
        )


def sync_deferred_conv_caches(
    layers: Any,
    def_plan: DeferredConvSyncPlan | None,
    comm_stream: torch.cuda.Stream | None = None,
) -> None:
    """Execute ONE single batched bulk sync across all 48 layers' `ATTN_STREAM` + `MLP_STREAM`
    real deferred `conv_state.kv_cache` rows (uses 1x broadcast if only 1 rank has deferred
    rows, else 1x AllGather across all 48 layers simultaneously!)."""
    if def_plan is None:
        return
    from .sconv import ATTN_STREAM, MLP_STREAM

    num_layers = len(layers)
    c0 = layers[0].conv_state.kv_cache
    if c0.numel() == 0:
        return
    a_off, a_w = layers[0].conv_state.stream_ranges[ATTN_STREAM]
    m_off, m_w = layers[0].conv_state.stream_ranges[MLP_STREAM]
    adjacent = (m_off == a_off + a_w) and (a_w == m_w)
    width_both = a_w + m_w
    k_def = def_plan.k_def

    def _run() -> None:
        packed = c0.new_empty((num_layers, k_def, width_both))
        if def_plan.single_sender_rank < 0 or def_plan.rank == def_plan.single_sender_rank:
            for li, layer in enumerate(layers):
                c = layer.conv_state.kv_cache
                if adjacent:
                    packed[li] = c[def_plan.my_src_blk, 0, def_plan.my_src_off, a_off : a_off + width_both]
                else:
                    la, lw = layer.conv_state.stream_ranges[ATTN_STREAM]
                    lm, mw = layer.conv_state.stream_ranges[MLP_STREAM]
                    packed[li, :, :lw] = c[def_plan.my_src_blk, 0, def_plan.my_src_off, la : la + lw]
                    packed[li, :, lw:] = c[def_plan.my_src_blk, 0, def_plan.my_src_off, lm : lm + mw]

        if def_plan.single_sender_rank >= 0:
            # 1x broadcast from the single rank with deferred tail rows (e.g. rank 7 -> 0.27 ms!)
            gathered = _tp_broadcast(packed, src=def_plan.single_sender_rank)  # [L, k_def, 2H]
            if def_plan.remote_gather_idx is None:
                return
            assert def_plan.remote_dst_blk is not None and def_plan.remote_dst_off is not None
            for li, layer in enumerate(layers):
                c = layer.conv_state.kv_cache
                rows = gathered[li].index_select(0, def_plan.remote_gather_idx)
                if adjacent:
                    c[def_plan.remote_dst_blk, 0, def_plan.remote_dst_off, a_off : a_off + width_both] = rows
                else:
                    la, lw = layer.conv_state.stream_ranges[ATTN_STREAM]
                    lm, mw = layer.conv_state.stream_ranges[MLP_STREAM]
                    c[def_plan.remote_dst_blk, 0, def_plan.remote_dst_off, la : la + lw] = rows[:, :lw]
                    c[def_plan.remote_dst_blk, 0, def_plan.remote_dst_off, lm : lm + mw] = rows[:, lw:]
            return

        # Multi-rank deferred prefix boundaries: ONE single AllGather across all 48 layers!
        flat_g = _tp_all_gather_dim0(packed.reshape(1, -1))  # [W, L * k_def * 2H]
        if def_plan.remote_gather_idx is None:
            return
        assert def_plan.remote_dst_blk is not None and def_plan.remote_dst_off is not None
        g_4d = flat_g.view(def_plan.world_size, num_layers, k_def, width_both)
        for li, layer in enumerate(layers):
            c = layer.conv_state.kv_cache
            rows = g_4d[:, li, :, :].reshape(def_plan.world_size * k_def, width_both).index_select(
                0, def_plan.remote_gather_idx
            )
            if adjacent:
                c[def_plan.remote_dst_blk, 0, def_plan.remote_dst_off, a_off : a_off + width_both] = rows
            else:
                la, lw = layer.conv_state.stream_ranges[ATTN_STREAM]
                lm, mw = layer.conv_state.stream_ranges[MLP_STREAM]
                c[def_plan.remote_dst_blk, 0, def_plan.remote_dst_off, la : la + lw] = rows[:, :lw]
                c[def_plan.remote_dst_blk, 0, def_plan.remote_dst_off, lm : lm + mw] = rows[:, lw:]

    if comm_stream is not None:
        cur = torch.cuda.current_stream(c0.device)
        ev_in = torch.cuda.Event()
        ev_in.record(cur)
        comm_stream.wait_event(ev_in)
        for _t in (
            def_plan.my_src_blk,
            def_plan.my_src_off,
            def_plan.remote_gather_idx,
            def_plan.remote_dst_blk,
            def_plan.remote_dst_off,
        ):
            if _t is not None:
                _t.record_stream(comm_stream)
        with torch.cuda.stream(comm_stream):
            _run()
            ev_out = torch.cuda.Event()
            ev_out.record(comm_stream)
        cur.wait_event(ev_out)
    else:
        _run()


# ---------------------------------------------------------------------------
# Collective primitives (NCCL RS/AG/A2A/Bcast, Exact canary, Compressed RS, RDMA hooks)
# ---------------------------------------------------------------------------
def _tp_reduce_scatter_into(
    partial: torch.Tensor,
    out_shard: torch.Tensor,
    mode: str,
    begin: int,
    end: int,
) -> None:
    """Reduce-scatter `partial [T, H]` along dim=0 directly into `out_shard [S, H]`."""
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce

    if mode == "exact":
        full = tensor_model_parallel_all_reduce(partial)
        out_shard.copy_(full[begin:end])
        return

    tp_group = get_tp_group()
    comm = getattr(tp_group, "device_communicator", None)
    rs_mode = SP_RS_MODE
    if mode == "rdma" and rs_mode == "nccl":
        rs_mode = "rdma_bf16"

    # 1. Optional custom 4-HCA pipelined RDMA ReduceScatter (rdma_rs_s8_ext.so)
    if rs_mode.startswith("rdma_") and comm is not None:
        rdma_comm = getattr(comm, "rdma_comm", None)
        rs_fn = getattr(rdma_comm, "reduce_scatter_into", None) if rdma_comm is not None else None
        if rs_fn is not None:
            sub_mode = rs_mode[5:]  # "bf16" | "int8_col" | "bf12"
            if rs_fn(out_shard, partial.contiguous(), mode=sub_mode):
                return
        _log_once(
            f"rdma_rs_fallback_{rs_mode}",
            f"[inkling-prefill-sp] RS mode={rs_mode} requested but rdma_rs_s8 sidecar not "
            "loaded (set RDMA_RS_EXT=/rdma/rdma_rs_s8_ext.so); falling back cleanly.",
        )
        # If compressed RDMA requested without .so, fall back to Triton+PyNccl A2A of same codec!
        if rs_mode in ("rdma_int8_col", "rdma_bf12"):
            rs_mode = rs_mode[5:]

    # 2. Optional compressed ReduceScatter (`rhd_int8` | `rhd_fp8` | `fp8_a2a` | `int8_a2a` | `int8_col` | `bf12`, T >= SP_RS_MIN_T)
    if (
        rs_mode in ("rhd_int8", "rhd_fp8", "fp8_a2a", "int8_a2a", "int8_col", "bf12")
        and int(partial.shape[0]) >= SP_RS_MIN_T
    ):
        try:
            from . import prefill_sp_rs as _sp_rs

            if _sp_rs.compressed_reduce_scatter_into(
                partial.contiguous(),
                out_shard,
                mode=rs_mode,
                world_size=tp_group.world_size,
                rank=tp_group.rank_in_group,
                min_tokens=SP_RS_MIN_T,
            ):
                return
        except Exception as _e:
            _log_once(
                f"rs_comp_fallback_{rs_mode}",
                f"[inkling-prefill-sp] compressed RS ({rs_mode}) unavailable ({_e!r}); "
                "falling back to 8-ch PyNccl BF16 reduce_scatter.",
            )

    # 3. Zero-copy PyNccl reduce_scatter directly into preallocated `out_shard` (= x_with_halo[halo:])
    pynccl = getattr(comm, "pynccl_comm", None) if comm is not None else None
    if pynccl is not None and not getattr(pynccl, "disabled", False):
        pynccl.reduce_scatter(out_shard, partial.contiguous())
        return

    # Fallback via vLLM group reduce_scatter
    res = tp_group.reduce_scatter(partial.contiguous(), dim=0)
    out_shard.copy_(res)


def _tp_all_to_all_dim0(send_buf: torch.Tensor) -> torch.Tensor:
    """Shift-rotated 1-hop AllToAll of contiguous `[world_size, ...]` along dim=0 on the
    current CUDA stream (stays strictly FIFO on `comm_stream` inside DBO!).
    Output `recv_buf[src]` == `send_buf_of_rank_src[my_rank]`.
    """
    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    world_size = tp_group.world_size
    rank = tp_group.rank_in_group
    send_c = send_buf.contiguous()
    recv_buf = torch.empty_like(send_c)
    comm = getattr(tp_group, "device_communicator", None)
    pynccl = getattr(comm, "pynccl_comm", None) if comm is not None else None
    if pynccl is not None and not getattr(pynccl, "disabled", False):
        recv_buf[rank].copy_(send_c[rank])
        pynccl.group_start()
        for d in range(1, world_size):
            dst = (rank + d) % world_size
            src = (rank - d) % world_size
            pynccl.send(send_c[dst], dst)
            pynccl.recv(recv_buf[src], src)
        pynccl.group_end()
        return recv_buf
    torch.distributed.all_to_all_single(recv_buf, send_c, group=tp_group.device_group)
    return recv_buf


def _tp_broadcast(tensor: torch.Tensor, src: int) -> torch.Tensor:
    """In-place broadcast of contiguous `tensor` from `src` on the current CUDA stream."""
    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    comm = getattr(tp_group, "device_communicator", None)
    pynccl = getattr(comm, "pynccl_comm", None) if comm is not None else None
    if pynccl is not None and not getattr(pynccl, "disabled", False):
        pynccl.broadcast(tensor, src)
        return tensor
    torch.distributed.broadcast(tensor, src=tp_group.ranks[src], group=tp_group.device_group)
    return tensor


def _tp_ring_shift_into(out_prev: torch.Tensor, in_mine: torch.Tensor) -> bool:
    """1-to-1 ring shift (`rank r -> rank (r + 1) % W`) via custom RDMA sidecar when loaded
    (40x less wire than k_sync=17 AG!). Returns False if sidecar is absent."""
    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    comm = getattr(tp_group, "device_communicator", None)
    rdma_comm = getattr(comm, "rdma_comm", None) if comm is not None else None
    fn = getattr(rdma_comm, "ring_shift_into", None) if rdma_comm is not None else None
    if fn is not None:
        return bool(fn(out_prev, in_mine.contiguous()))
    return False


def _tp_all_gather_dim0(shard: torch.Tensor) -> torch.Tensor:
    """All-gather contiguous `[S, H]` along dim=0 -> `[T, H]`."""
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(shard.contiguous(), dim=0)


def _tp_all_gather_mode(shard: torch.Tensor, mode: str) -> torch.Tensor:
    """All-gather contiguous `[S, H]` BF16 shard along dim=0 with optional zero-incast
    3-step RHD (`INKLING_PREFILL_SP_AG_MODE=rhd_int8|rhd_fp8`) or FP8 Ring compression:
      - `bf16`: standard BF16 AllGather (2 B/elem) — unless `SP_AG_MODE != nccl` and
        `mode == "rhd"` (used when `SP_FP8_AG=0` + `SP_AG_MODE=rhd_int8|rhd_fp8`).
      - `direct`: quantize `[S, H]` -> FP8 E4M3 (1 B/elem) + `[S, 1]` FP32 scale, AllGather
        (single packed RHD collective when `SP_AG_MODE=rhd_fp8`, else 2x NCCL Ring AG),
        returning FP8 tensor carrying `._inkling_fp8_scale` for `linear.py` `torch._scaled_mm`.
      - `dequant`: 2x-smaller compressed AllGather + fused dequant to BF16 on current stream.
    """
    if (
        SP_AG_MODE in ("rhd_int8", "rhd_fp8")
        and mode != "bf16"
        and shard.dtype == torch.bfloat16
        and shard.ndim == 2
        and not torch.cuda.is_current_stream_capturing()
    ):
        try:
            from . import prefill_sp_rs as _sp_rs

            want_direct = (mode == "direct") and (SP_AG_MODE == "rhd_fp8")
            rhd_out = _sp_rs.compressed_all_gather_dim0(
                shard,
                mode=SP_AG_MODE,
                fp8_direct=want_direct,
                min_tokens=SP_AG_MIN_T,
            )
            if rhd_out is not None:
                return rhd_out
        except Exception as _e:
            _log_once(
                f"ag_rhd_fallback_{SP_AG_MODE}",
                f"[inkling-prefill-sp] RHD AG ({SP_AG_MODE}) unavailable ({_e!r}); "
                "falling back to standard NCCL AllGather.",
            )

    if mode in ("bf16", "rhd") or not SP_FP8_AG or shard.dtype != torch.bfloat16 or shard.ndim != 2:
        return _tp_all_gather_dim0(shard)
    from .ops.linear import dequantize_fp8_rowwise, quantize_fp8_rowwise

    x_fp8_s, x_scale_s = quantize_fp8_rowwise(shard, pad_m16=False)
    x_fp8_full = _tp_all_gather_dim0(x_fp8_s.view(torch.uint8)).view(torch.float8_e4m3fn)
    x_scale_full = _tp_all_gather_dim0(x_scale_s)
    if mode == "direct":
        x_fp8_full._inkling_fp8_scale = x_scale_full  # type: ignore[attr-defined]
        return x_fp8_full
    return dequantize_fp8_rowwise(x_fp8_full, x_scale_full)


# ---------------------------------------------------------------------------
# Exact Halo-Aware Sharded short_conv + AttnRes Dispatcher (0-ULP!)
# ---------------------------------------------------------------------------
def _sharded_short_conv_halo_torch(
    x_with_halo: torch.Tensor,  # [halo + S, D] bf16
    halo: int,
    conv_weight: torch.Tensor,  # [D, 4] bf16
    cache: torch.Tensor,  # [num_blocks, 1, N_cache, head_size] bf16
    positions_s: torch.Tensor,  # [S] int64
    block_table: torch.Tensor,  # [num_reqs, max_blocks] int32
    seq_idx_s: torch.Tensor,  # [S] int32
    slot_mapping_s: torch.Tensor,  # [S] int64
    qstart_local: torch.Tensor,  # [S] int32 (= query_start[begin:end] - begin)
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,  # [S, D] bf16
    conv_addend: torch.Tensor | None = None,
    conv_res: torch.Tensor | None = None,
) -> torch.Tensor:
    """0-ULP exact PyTorch fallback for sharded static W=4 short_conv with negative-offset halo.

    Matches `_sconv_kernel` (`sconv_ops.py`) and `fused_sconv_ar_prefill_exact.cu`:
      1. Stores `x_cur[t]` into `cache[slot // N, 0, slot % N, offset:offset+D]` where `slot >= 0`.
      2. Computes `y = (sum_{k=0..3} w[:, k] * tap(t - (3 - k)) + x_cur[t]) * 5**-0.5` in fp32,
         rounded to bf16.
      3. Optional residual update: `((conv_res + conv_addend) + y)` in fp32, rounded to bf16.
    """
    S, D = conv_out.shape
    W = int(conv_weight.shape[1])
    assert W == 4 and stream_width == D
    n_cache = int(cache.shape[2])
    max_blocks = int(block_table.shape[1])
    x_cur = x_with_halo[halo:]  # [S, D]

    # 1. Write current shard inputs to local conv_state.kv_cache (64-bit block indexing!)
    valid = slot_mapping_s >= 0
    if bool(valid.any().item()):
        v_slots = slot_mapping_s[valid].to(torch.int64)
        cache[v_slots // n_cache, 0, v_slots % n_cache, stream_offset : stream_offset + D] = (
            x_cur[valid].to(cache.dtype)
        )

    # 2. Gather W=4 causal taps (k=0 -> lag=3, k=1 -> lag=2, k=2 -> lag=1, k=3 -> lag=0)
    t_idx = torch.arange(S, dtype=torch.int64, device=x_with_halo.device)
    qstart64 = qstart_local.to(torch.int64)
    w_f32 = conv_weight.to(cache.dtype).float()  # [D, 4]
    acc = torch.zeros((S, D), dtype=torch.float32, device=x_with_halo.device)

    for k in range(W):
        lag = W - 1 - k
        src = positions_s - lag
        row = t_idx - lag
        in_win = valid & (src >= 0)
        intra = in_win & (row >= qstart64)
        cached = in_win & (row < qstart64)

        tap = torch.zeros((S, D), dtype=torch.float32, device=x_with_halo.device)
        if bool(intra.any().item()):
            # row + halo is guaranteed in [0, halo + S - 1]!
            xh_idx = (row[intra] + halo).clamp(0, halo + S - 1)
            tap[intra] = x_with_halo[xh_idx].to(cache.dtype).float()
        if bool(cached.any().item()):
            c_src = src[cached].clamp(min=0)
            c_req = seq_idx_s[cached].to(torch.int64)
            c_lblk = (c_src // n_cache).clamp(max=max_blocks - 1)
            c_blk = block_table[c_req, c_lblk].to(torch.int64)
            tap[cached] = cache[
                c_blk, 0, c_src % n_cache, stream_offset : stream_offset + D
            ].float()
        acc = acc + tap * w_f32[:, k].unsqueeze(0)

    scale = float((W + 1) ** -0.5)
    y_bf16 = ((acc + x_cur.to(cache.dtype).float()) * scale).to(conv_out.dtype)

    if conv_addend is not None:
        ca = conv_addend.float()
        if conv_res is not None:
            ca = conv_res.float() + ca
        y_bf16 = (ca + y_bf16.float()).to(conv_out.dtype)

    conv_out.copy_(y_bf16)
    return conv_out


def scatter_remote_conv_cache(
    gathered_sync: torch.Tensor,
    plan: PrefillSPPlan,
    cache: torch.Tensor,
    stream_offset: int,
    stream_width: int,
) -> None:
    """Scatter remote ranks' prefix-boundary & request-tail rows into this rank's
    `conv_state.kv_cache` so all 8 ranks hold 100% bit-identical cache state for
    future chunks, decode, verify rollbacks, and prefix-cache hits."""
    if plan.rank0_last3_slots is not None:
        from . import prefill_sp_rs as _sp_rs

        if int(gathered_sync.shape[0]) == HALO_ROWS:
            src_rows = gathered_sync
        else:
            r_base = (plan.world_size - 1) * HALO_ROWS
            src_rows = gathered_sync[r_base : r_base + HALO_ROWS]
        if _sp_rs.scatter_halo_last3_gpu(
            src_rows,
            plan.rank0_last3_slots,
            plan.discard_blk_dev,
            cache,
            stream_offset,
            stream_width,
            plan.n_cache,
        ):
            return
    if plan.remote_gather_idx is None:
        return
    assert plan.remote_dst_blk is not None and plan.remote_dst_off is not None
    if plan.halo_only and int(gathered_sync.shape[0]) == HALO_ROWS:
        idx = plan.remote_gather_idx - (plan.world_size - 1) * HALO_ROWS
        rows = gathered_sync.index_select(0, idx).to(cache.dtype)
    else:
        rows = gathered_sync.index_select(0, plan.remote_gather_idx).to(cache.dtype)
    cache[
        plan.remote_dst_blk,
        0,
        plan.remote_dst_off,
        stream_offset : stream_offset + stream_width,
    ] = rows


def run_sharded_sconv_attn_res(
    sconv_mod: Any,  # Inkling2ShortConv
    x_with_halo: torch.Tensor,  # [halo + S, D] bf16
    plan: PrefillSPPlan,
    conv_out: torch.Tensor,  # [S, D] bf16
    stack: torch.Tensor,  # [n_stack, S, D] bf16
    mix_last: torch.Tensor | None,  # [S, D] bf16 or None
    queries: torch.Tensor,  # [nq, D] fp32
    norm: Any,  # Inkling2RMSNorm
    normalize_values: bool,
    conv_addend: torch.Tensor | None = None,
    conv_res: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run sharded short_conv + AttnRes + RMSNorm on this rank's `[S, D]` slice.

    Returns `[nq, S, D]` bf16.
    Fast path: calls `libfused_sconv_ar_prefill.so` (`fused_sconv_ar_prefill_exact_fwd`)
    directly with `X = x_with_halo[halo:]` and `QStart = plan.qstart_local` (negative-offset
    halo, 0-ULP bit-exact, zero CUDA recompile!).
    Fallback: exact halo-aware PyTorch short_conv + AttnRes matching `_attn_res_kernel`.
    """
    S, D = conv_out.shape
    nq = int(queries.shape[0])
    n_stack = int(stack.shape[0])
    halo = plan.halo
    x_cur = x_with_halo[halo:]  # [S, D] contiguous; x_cur[-halo:] is valid halo memory!

    if not plan.has_cache:
        # Profile run (no conv_state cache): mirror Inkling2ShortConv.forward profile branch.
        if conv_addend is not None:
            acc = conv_addend.float() if conv_res is None else conv_res.float() + conv_addend.float()
            res = (acc + x_cur.to(conv_addend.dtype).float()).to(conv_addend.dtype)
        else:
            res = x_cur
        conv_out.copy_(res)
        from .ops.attn_res import attn_res as _attn_res

        if mix_last is not None:
            return _attn_res(
                stack, mix_last, conv_out, queries, norm.weight, norm.eps, normalize_values, eps
            )
        return _attn_res(
            stack, conv_out, None, queries, norm.weight, norm.eps, normalize_values, eps
        )

    from .sconv import _get_fused_prefill_lib, conv_metadata

    m = conv_metadata(sconv_mod.owner)
    assert m is not None
    offset, width = sconv_mod.owner.stream_ranges[sconv_mod.stream]
    c = sconv_mod.owner.kv_cache
    cw = sconv_mod.conv_weight

    # Fast 0-ULP 1-pass CUDA path: works for ANY shard size S >= 1 when logical T >= 1024!
    plib = _get_fused_prefill_lib()
    if (
        plib is not None
        and D == 4608
        and nq in (1, 3)
        and 1 <= n_stack <= 8
        and normalize_values
        and c.shape[1] == 1
        and width == D
        and cw.shape == (D, 4)
        and cw.is_contiguous()
        and x_with_halo.dtype == torch.bfloat16
        and x_with_halo.is_contiguous()
        and c.dtype == torch.bfloat16
        and cw.dtype == torch.bfloat16
        and norm.weight.dtype == torch.bfloat16
        and stack.dtype == torch.bfloat16
        and queries.dtype == torch.float32
        and plan.positions_s.dtype == torch.int64
        and plan.slot_mapping_s is not None
        and plan.slot_mapping_s.dtype == torch.int64
        and plan.seq_idx_s is not None
        and plan.seq_idx_s.dtype == torch.int32
        and m.block_table.dtype == torch.int32
        and plan.qstart_local is not None
        and plan.qstart_local.dtype == torch.int32
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
        out = torch.empty((nq, S, D), dtype=torch.bfloat16, device=x_with_halo.device)
        v2_fn = getattr(plib, "fused_sconv_ar_prefill_exact_v2_fwd", None)
        if v2_fn is not None:
            discard_ptr = 0 if plan.discard_blk_dev is None else int(plan.discard_blk_dev.data_ptr())
            rc = v2_fn(
                x_cur.data_ptr(),
                cw.data_ptr(),
                c.data_ptr(),
                conv_out.data_ptr(),
                0 if conv_res is None else conv_res.data_ptr(),
                0 if conv_addend is None else conv_addend.data_ptr(),
                plan.positions_s.data_ptr(),
                plan.seq_idx_s.data_ptr(),
                plan.slot_mapping_s.data_ptr(),
                m.block_table.data_ptr(),
                plan.qstart_local.data_ptr(),
                stack.data_ptr(),
                0 if mix_last is None else mix_last.data_ptr(),
                queries.data_ptr(),
                norm.weight.data_ptr(),
                out.data_ptr(),
                discard_ptr,
                int(x_cur.stride(0)),
                int(c.stride(0)),
                int(c.stride(2)),
                int(m.block_table.stride(0)),
                int(m.block_table.shape[1]),
                int(c.shape[2]),
                int(offset),
                int(stack.stride(0)),
                nq,
                S,
                n_stack,
                float(5**-0.5),
                float(eps),
                float(norm.eps),
                1 if conv_res is not None else 0,
                1 if conv_addend is not None else 0,
                1 if mix_last is not None else 0,
                int(torch.cuda.current_stream().cuda_stream),
            )
        else:
            rc = plib.fused_sconv_ar_prefill_exact_fwd(
                x_cur.data_ptr(),
                cw.data_ptr(),
                c.data_ptr(),
                conv_out.data_ptr(),
                0 if conv_res is None else conv_res.data_ptr(),
                0 if conv_addend is None else conv_addend.data_ptr(),
                plan.positions_s.data_ptr(),
                plan.seq_idx_s.data_ptr(),
                plan.slot_mapping_s.data_ptr(),
                m.block_table.data_ptr(),
                plan.qstart_local.data_ptr(),
                stack.data_ptr(),
                0 if mix_last is None else mix_last.data_ptr(),
                queries.data_ptr(),
                norm.weight.data_ptr(),
                out.data_ptr(),
                int(x_cur.stride(0)),
                int(c.stride(0)),
                int(c.stride(2)),
                int(m.block_table.stride(0)),
                int(m.block_table.shape[1]),
                int(c.shape[2]),
                int(offset),
                int(stack.stride(0)),
                nq,
                S,
                n_stack,
                float(5**-0.5),
                float(eps),
                float(norm.eps),
                1 if conv_res is not None else 0,
                1 if conv_addend is not None else 0,
                1 if mix_last is not None else 0,
                int(torch.cuda.current_stream().cuda_stream),
            )
        if rc == 0:
            return out

    # Fallback path: exact halo-aware short_conv + AttnRes (force BLOCK_D=1024 prefill kernel
    # via attn_res_prefill_v2_fwd when available to keep literal 0-ULP parity with T_full > 256!)
    assert plan.seq_idx_s is not None and plan.slot_mapping_s is not None and plan.qstart_local is not None
    _sharded_short_conv_halo_torch(
        x_with_halo,
        halo,
        cw,
        c,
        plan.positions_s,
        m.block_table,
        plan.seq_idx_s,
        plan.slot_mapping_s,
        plan.qstart_local,
        offset,
        width,
        conv_out,
        conv_addend=conv_addend,
        conv_res=conv_res,
    )
    from .ops import attn_res as _ar_mod

    # Prefer attn_res_prefill_v2_fwd directly (matches logical T_full > 256 even if S <= 256)
    get_ar_lib = getattr(_ar_mod, "_get_ar_prefill_lib", None)
    ar_lib = get_ar_lib() if get_ar_lib is not None else None
    last_t = mix_last if mix_last is not None else conv_out
    add_t = conv_out if mix_last is not None else None
    if ar_lib is not None and D == 4608 and 1 <= n_stack <= 8 and nq in (1, 3) and normalize_values:
        out = torch.empty((nq, S, D), dtype=torch.bfloat16, device=x_with_halo.device)
        rc = ar_lib.attn_res_prefill_v2_fwd(
            stack.data_ptr(),
            last_t.data_ptr(),
            0 if add_t is None else add_t.data_ptr(),
            queries.data_ptr(),
            norm.weight.data_ptr(),
            out.data_ptr(),
            int(stack.stride(0)),
            nq,
            S,
            n_stack,
            float(eps),
            float(norm.eps),
            1 if add_t is not None else 0,
            1,
            int(torch.cuda.current_stream().cuda_stream),
        )
        if rc == 0:
            return out
    return _ar_mod.attn_res(
        stack, last_t, add_t, queries, norm.weight, norm.eps, normalize_values, eps
    )


# ---------------------------------------------------------------------------
# PrefillSeqParallel: High-level GB10 Prefill SP Controller
# ---------------------------------------------------------------------------
class PrefillSeqParallel:
    """Sequence-parallel residual stream for GB10 eager prefill (INKLING_PREFILL_SP)."""

    def __init__(self, hidden_size: int, tail_rows: int) -> None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        self.world_size = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        self.hidden_size = hidden_size
        self.tail_rows = max(int(tail_rows), HALO_ROWS)
        self.mode = SP_MODE
        self.min_tokens = SP_MIN_TOKENS
        self.q3_mode = SP_Q3_MODE

    def eligible(self, num_tokens: int) -> bool:
        if not SP_ENABLED or self.world_size <= 1:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if num_tokens < self.min_tokens:
            return False
        if num_tokens % self.world_size != 0:
            return False
        return (num_tokens // self.world_size) >= HALO_ROWS

    def pad_target(
        self,
        num_tokens: int,
        attn_metadata: Any = None,
        first_attn_prefix: str = "",
    ) -> int:
        """Return aligned padded token count (> num_tokens) for unaligned tail prefill chunks
        (`INKLING_PREFILL_SP_PAD_TAIL=1`, `T >= SP_PAD_MIN_T`, `T % world_size != 0`), or 0."""
        if not (SP_ENABLED and SP_PAD_TAIL) or self.world_size <= 1:
            return 0
        if torch.cuda.is_current_stream_capturing():
            return 0
        if num_tokens < max(self.min_tokens, SP_PAD_MIN_T):
            return 0
        if num_tokens % self.world_size == 0:
            return 0
        if isinstance(attn_metadata, dict) and first_attn_prefix:
            md = attn_metadata.get(first_attn_prefix)
            if md is None:
                return 0
            if getattr(md, "num_decode_reqs", 1) != 0 or getattr(md, "num_prefill_reqs", 0) < 1:
                return 0
            if getattr(md, "num_actual_tokens", -1) != num_tokens:
                return 0
        align = max(self.world_size, 16)
        t_pad = ((num_tokens + align - 1) // align) * align
        if (t_pad // self.world_size) < HALO_ROWS:
            return 0
        return t_pad

    def build_plan(
        self,
        num_tokens: int,
        positions: torch.Tensor,
        conv_md: Any | None,
        n_cache: int,
        prefix_align: int,
        tail_rows_override: int | None = None,
        prefix_sync_mode: str | None = None,
    ) -> PrefillSPPlan:
        eff_tail = self.tail_rows if tail_rows_override is None else max(int(tail_rows_override), HALO_ROWS)
        eff_psync = SP_PREFIX_SYNC if prefix_sync_mode is None else prefix_sync_mode
        plan = PrefillSPPlan.build(
            num_tokens=num_tokens,
            world_size=self.world_size,
            rank=self.rank,
            positions=positions,
            conv_md=conv_md,
            n_cache=n_cache,
            prefix_align=prefix_align,
            tail_rows=eff_tail,
            prefix_sync_mode=eff_psync,
        )
        if SP_DEBUG:
            _log_once(
                f"sp_active_{num_tokens}_{plan.has_cache}_{eff_tail}_{eff_psync}",
                f"[inkling-prefill-sp] ACTIVE rank={self.rank}/{self.world_size} "
                f"mode={self.mode} rs={SP_RS_MODE} ag={SP_AG_MODE} q3={self.q3_mode} psync={eff_psync} "
                f"T={num_tokens} S={plan.shard_size} k_sync={plan.k_sync} has_cache={plan.has_cache}",
            )
        return plan

    def reduce_scatter_and_halo(
        self,
        partial: torch.Tensor,
        plan: PrefillSPPlan,
        sconv_mod: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """1. Reduce-scatter `partial [T, H]` directly into `x_with_halo[halo:]`.
        2. Populate the 3-row halo `x_with_halo[:halo]` + immediate `conv_state.kv_cache` sync rows:
           - In `halo_only` mode (`deferred`/`off`, `k_sync == 3`): tries 1-to-1 RDMA `ring_shift_into`
             (`27 KiB`, 40x less wire!) first, falling back to `k_sync=3` RDMA AG (`1-CTA kR3S1`).
           - In `eager` mode: unified `k_sync`-row RDMA AG.
        Returns `(x_with_halo, gathered_sync)`.
        """
        halo = plan.halo
        x_with_halo = partial.new_empty((halo + plan.shard_size, self.hidden_size))
        x_shard = x_with_halo[halo:]
        _tp_reduce_scatter_into(partial, x_shard, self.mode, plan.begin, plan.end)

        gathered_sync = None
        if plan.has_cache and plan.my_pack_idx is not None:
            has_remote = (
                plan.remote_gather_idx is not None
                or plan.rank0_last3_slots is not None
            )
            if plan.halo_only:
                # Last 3 rows are already contiguous at x_shard[-HALO_ROWS:] -> zero index_select!
                my_last3 = x_shard[-HALO_ROWS:]
                recv_prev = x_with_halo[:HALO_ROWS] if halo > 0 else partial.new_empty((HALO_ROWS, self.hidden_size))
                if _tp_ring_shift_into(recv_prev, my_last3):
                    gathered_sync = recv_prev if halo == 0 else None
                    if sconv_mod is not None and has_remote and gathered_sync is not None:
                        offset, width = sconv_mod.owner.stream_ranges[sconv_mod.stream]
                        scatter_remote_conv_cache(
                            gathered_sync, plan, sconv_mod.owner.kv_cache, offset, width
                        )
                    return x_with_halo, gathered_sync
                packed = my_last3.contiguous()
            else:
                packed = x_shard.index_select(0, plan.my_pack_idx).contiguous()
            gathered_sync = _tp_all_gather_dim0(packed)
            if halo > 0 and plan.halo_gather_idx is not None:
                x_with_halo[:halo].copy_(gathered_sync.index_select(0, plan.halo_gather_idx))
            if sconv_mod is not None and has_remote:
                offset, width = sconv_mod.owner.stream_ranges[sconv_mod.stream]
                scatter_remote_conv_cache(
                    gathered_sync, plan, sconv_mod.owner.kv_cache, offset, width
                )
        return x_with_halo, gathered_sync

    def sublayer_step(
        self,
        partial: torch.Tensor,
        sconv_mod: Any,
        plan: PrefillSPPlan,
        conv_out: torch.Tensor,
        stack: torch.Tensor,
        mix_last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: Any,
        normalize_values: bool,
        conv_addend: torch.Tensor | None = None,
        conv_res: torch.Tensor | None = None,
        next_global_repl: GlobalKVReplWeights | None = None,
        fp8_ag_modes: tuple[str, ...] | None = None,
    ) -> list[Any]:
        """Full SP sublayer: RS -> Halo+CacheSync -> Sharded Fused sconv+AttnRes -> AG (or repl_kv A2A).
        Returns list of `Q` full `[T, H]` tensors (`Q=1` for 88 sublayers, `Q=3` for 8 global-next),
        or `[x_q_full, PreprojectedGlobalKV, PreprojectedGlobalKV]` when `repl_kv` is active."""
        x_with_halo, _ = self.reduce_scatter_and_halo(partial, plan, sconv_mod=sconv_mod)
        out_shard = run_sharded_sconv_attn_res(
            sconv_mod=sconv_mod,
            x_with_halo=x_with_halo,
            plan=plan,
            conv_out=conv_out,
            stack=stack,
            mix_last=mix_last,
            queries=queries,
            norm=norm,
            normalize_values=normalize_values,
            conv_addend=conv_addend,
            conv_res=conv_res,
        )
        return self.gather_queries(
            out_shard, next_global_repl=next_global_repl, fp8_ag_modes=fp8_ag_modes
        )

    def gather_queries(
        self,
        out_shard: torch.Tensor,
        next_global_repl: GlobalKVReplWeights | None = None,
        fp8_ag_modes: tuple[str, ...] | None = None,
    ) -> list[Any]:
        """All-gather `[Q, S, H]` along dim=0 per query -> list of `Q` contiguous `[T, H]` tensors.
        When `next_global_repl` is provided and `Q == 3` (`INKLING_PREFILL_SP_Q3=repl_kv`),
        AllGathers ONLY `x_q` and projects sharded `x_k, x_v` + 1 tiny `all_to_all` (20x less wire,
        100% BF16 exact!)."""
        nq = int(out_shard.shape[0])
        _default_ag_mode = "rhd" if SP_AG_MODE != "nccl" else "bf16"
        if next_global_repl is not None and nq == 3:
            q_mode = fp8_ag_modes[0] if (SP_FP8_AG and fp8_ag_modes is not None) else _default_ag_mode
            x_q_full = _tp_all_gather_mode(out_shard[0], q_mode)
            kv_preproj = global_kv_proj_fused_a2a(
                out_shard[1], out_shard[2], next_global_repl, _tp_all_to_all_dim0
            )
            return [x_q_full, kv_preproj, kv_preproj]
        if SP_FP8_AG and fp8_ag_modes is not None and len(fp8_ag_modes) == nq:
            return [_tp_all_gather_mode(out_shard[q], fp8_ag_modes[q]) for q in range(nq)]
        if SP_AG_MODE != "nccl":
            return [_tp_all_gather_mode(out_shard[q], "rhd") for q in range(nq)]
        return [_tp_all_gather_dim0(out_shard[q]) for q in range(nq)]


# ---------------------------------------------------------------------------
# Zero-Extra-Wire Q=3 Global KV Projection (`INKLING_PREFILL_SP_Q3=repl_kv`, 148.5 MiB)
# ---------------------------------------------------------------------------
@dataclass
class PreprojectedGlobalKV:
    """Pre-projected `[T, kv_dim]` K/V and `[T, W]` dynamic sconv taps produced from
    sharded `x_k`/`x_v` via replicated global weights + 1 tiny `all_to_all` (20x less wire,
    100% BF16 exact!). Consumed directly by `Inkling2Attention._project_global_with_dyn`."""

    k: torch.Tensor       # [T, kv_dim] bf16 (stride(1) == 1, 16B-aligned rows)
    v: torch.Tensor       # [T, kv_dim] bf16 (stride(1) == 1, 16B-aligned rows)
    k_dyn: torch.Tensor   # [T, w_dim] bf16 (stride(1) == 1)
    v_dyn: torch.Tensor   # [T, w_dim] bf16 (stride(1) == 1)
    _backing: torch.Tensor  # single contiguous AllToAll output buffer owning storage

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self._backing.record_stream(stream)


@dataclass
class GlobalKVReplWeights:
    """Replicated `[world_size * (kv_dim + w_dim), H]` K+k_dyn and V+v_dyn weights for one
    global attention layer (~18.56 MiB/layer, ~148.5 MiB total across all 8 global layers)."""

    wkd_all: torch.Tensor  # [world_size * (kv_dim + w_dim), H] bf16
    wvd_all: torch.Tensor  # [world_size * (kv_dim + w_dim), H] bf16
    kv_dim: int
    w_dim: int
    world_size: int


@torch.no_grad()
def build_global_kv_repl_weights(attn: Any, world_size: int) -> GlobalKVReplWeights | None:
    """All-gather `w_k, k_dyn_w` and `w_v, v_dyn_w` once across TP for a global attention layer.
    Returns None cleanly for local layers, non-dynamic convs, or LoRA-wrapped `qkvr`."""
    if attn.is_local or world_size <= 1:
        return None
    if not (attn.k_sconv.dynamic and attn.v_sconv.dynamic):
        return None
    if getattr(attn.qkvr, "punica_wrapper", None) is not None or hasattr(attn.qkvr, "lora_a_stacked"):
        return None
    weight = attn.qkvr.weight.data
    q_side = sum(attn.split_sizes[:4])
    kv_dim = int(attn.split_sizes[4])
    k_end = q_side + kv_dim
    w_k = weight[q_side:k_end]
    w_v = weight[k_end : k_end + kv_dim]
    k_dyn_w = attn.k_sconv.weight_proj.data
    v_dyn_w = attn.v_sconv.weight_proj.data
    w_dim = int(k_dyn_w.shape[0])
    if int(v_dyn_w.shape[0]) != w_dim:
        return None
    local_kd = torch.cat([w_k, k_dyn_w], dim=0).contiguous()  # [kv_dim + w_dim, H]
    local_vd = torch.cat([w_v, v_dyn_w], dim=0).contiguous()  # [kv_dim + w_dim, H]
    wkd_all = _tp_all_gather_dim0(local_kd).contiguous()      # [W * (kv_dim + w_dim), H]
    wvd_all = _tp_all_gather_dim0(local_vd).contiguous()      # [W * (kv_dim + w_dim), H]
    return GlobalKVReplWeights(
        wkd_all=wkd_all,
        wvd_all=wvd_all,
        kv_dim=kv_dim,
        w_dim=w_dim,
        world_size=world_size,
    )


def global_kv_proj_fused_a2a(
    x_k_shard: torch.Tensor,  # [S, H] bf16
    x_v_shard: torch.Tensor,  # [S, H] bf16
    repl_w: GlobalKVReplWeights,
    all_to_all_fn: Any,
) -> PreprojectedGlobalKV:
    """Fused 2-GEMM + 1-AllToAll zero-extra-wire Q=3 projection (`INKLING_PREFILL_SP_Q3=repl_kv`).

    Runs 2 BF16 GEMMs (`[S, H] @ [H, W*(kv+w)]`, exact same total FLOPs as standard TP,
    100% BF16 exact — NO FP8 quantization of `x_k`/`x_v`!), packs `[k, v, k_dyn, v_dyn]`
    in 16B-aligned row order `[128, 128, 4, 4]`, runs 1 shift-rotated `all_to_all`
    (`3.78 MB` vs `66.06 MB` BF16 / `33.03 MB` FP8 AG!), and returns zero-copy row-strided
    views (`stride(1)==1`, directly consumed by `_kvq_prefill_kernel` in `sconv_ops.py`!).
    """
    S = int(x_k_shard.shape[0])
    W = repl_w.world_size
    kv = repl_w.kv_dim
    wd = repl_w.w_dim
    stride = kv + wd
    T = S * W

    k_kd = torch.nn.functional.linear(x_k_shard, repl_w.wkd_all).view(S, W, stride)
    v_vd = torch.nn.functional.linear(x_v_shard, repl_w.wvd_all).view(S, W, stride)
    # Order [k(128), v(128), k_dyn(4), v_dyn(4)] -> stride 264 bf16 = 528B (% 16 == 0) so both k & v are 16B-aligned!
    packed_in = (
        torch.cat(
            [k_kd[:, :, :kv], v_vd[:, :, :kv], k_kd[:, :, kv:], v_vd[:, :, kv:]],
            dim=-1,
        )
        .transpose(0, 1)
        .contiguous()
    )
    packed_out = all_to_all_fn(packed_in)  # [W, S, 2*(kv + wd)]
    flat = packed_out.reshape(T, 2 * stride)
    k_full, v_full, k_dyn, v_dyn = flat.split([kv, kv, wd, wd], dim=-1)
    return PreprojectedGlobalKV(
        k=k_full, v=v_full, k_dyn=k_dyn, v_dyn=v_dyn, _backing=packed_out
    )


def global_kv_proj_sharded_a2a(
    x_k_shard: torch.Tensor,  # [S, H]
    x_v_shard: torch.Tensor,  # [S, H]
    wk_all: torch.Tensor,  # [world_size * kv_dim, H] (replicated across ranks)
    wv_all: torch.Tensor,  # [world_size * kv_dim, H]
    k_dyn_w_all: torch.Tensor,  # [world_size * W, H]
    v_dyn_w_all: torch.Tensor,  # [world_size * W, H]
    world_size: int,
    all_to_all_fn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-extra-wire Q=3 projection for the 8 global layers (`INKLING_PREFILL_SP_Q3=repl_kv`).

    Instead of all-gathering `x_k` and `x_v` (`2 * 0.875 * S_bytes = 1.75 * S_bytes`),
    each rank projects its own `[S, H]` shard against all 8 ranks' replicated `wk/wv/dyn`
    weights (`155 MiB` total across 8 global layers; EXACT same total FLOPs as standard TP!)
    and exchanges ONLY `[S, 132]` per K/V via `all_to_all` (`0.05 * S_bytes` = 97% less wire!).

    Returns `(k_full [T, kv_dim], v_full [T, kv_dim], k_dyn [T, W], v_dyn [T, W])`,
    0-ULP bit-identical to projecting gathered `x_k, x_v` against local `wk_rank, wv_rank`!
    """
    S = int(x_k_shard.shape[0])
    kv_dim = int(wk_all.shape[0]) // world_size
    w_dim = int(k_dyn_w_all.shape[0]) // world_size
    T = S * world_size

    # Local GEMMs on sharded [S, H] (exact same K=H dot products as unsharded!)
    k_all = torch.nn.functional.linear(x_k_shard, wk_all).view(S, world_size, kv_dim)
    v_all = torch.nn.functional.linear(x_v_shard, wv_all).view(S, world_size, kv_dim)
    kd_all = torch.nn.functional.linear(x_k_shard, k_dyn_w_all).view(S, world_size, w_dim)
    vd_all = torch.nn.functional.linear(x_v_shard, v_dyn_w_all).view(S, world_size, w_dim)

    # Pack into single contiguous [world_size, S, 2*(kv_dim + w_dim)] buffer for 1 all_to_all!
    packed_in = torch.cat([k_all, v_all, kd_all, vd_all], dim=-1).transpose(0, 1).contiguous()
    packed_out = all_to_all_fn(packed_in)  # [world_size, S, 2*(kv_dim + w_dim)]
    flat = packed_out.reshape(T, 2 * (kv_dim + w_dim))
    k_full, v_full, k_dyn, v_dyn = flat.split([kv_dim, kv_dim, w_dim, w_dim], dim=-1)
    return k_full.contiguous(), v_full.contiguous(), k_dyn.contiguous(), v_dyn.contiguous()


# ---------------------------------------------------------------------------
# Dual-Microbatch (T/2 + T/2) Overlapped Prefill All-Reduce Engine
# ---------------------------------------------------------------------------
@dataclass
class _AsyncHandle:
    out: Any
    done_event: torch.cuda.Event
    comm_stream: torch.cuda.Stream
    gathered_sync: torch.Tensor | None = None

    def wait(self, compute_stream: torch.cuda.Stream) -> Any:
        compute_stream.wait_event(self.done_event)
        self.out.record_stream(compute_stream)
        sc = getattr(self.out, "_inkling_fp8_scale", None)
        if sc is not None:
            sc.record_stream(compute_stream)
        if self.gathered_sync is not None:
            self.gathered_sync.record_stream(compute_stream)
        return self.out


@dataclass
class _AsyncSPSublayerHandle:
    """Async handle for a full SP sublayer pipelined on `comm_stream`:
    `reduce_scatter -> halo/cache_sync -> sharded sconv+AttnRes -> all_gather(queries [+ aux])`.
    """

    outs: list[Any]
    done_event: torch.cuda.Event
    comm_stream: torch.cuda.Stream
    aux: torch.Tensor | None = None

    def wait(
        self, compute_stream: torch.cuda.Stream
    ) -> tuple[list[Any], torch.Tensor | None]:
        compute_stream.wait_event(self.done_event)
        seen: set[int] = set()
        for t in self.outs:
            tid = id(t)
            if tid in seen:
                continue
            seen.add(tid)
            t.record_stream(compute_stream)
            sc = getattr(t, "_inkling_fp8_scale", None)
            if sc is not None:
                sc.record_stream(compute_stream)
        if self.aux is not None:
            self.aux.record_stream(compute_stream)
        return self.outs, self.aux


class PrefillAROverlap:
    """Dual-Microbatch (`T/2 + T/2`) Overlapped Prefill All-Reduce controller.

    Pipelines compute stream (Attn / SConv+AttnRes / MoE) with a dedicated high-priority
    `comm_stream` running TP collectives (AR, or SP RS+AG) for the other micro-batch:

      Compute: | Attn_i(A) |  Attn_i(B)  | SC+MoE_i(A) | SC+MoE_i(B) | SC+Attn_{i+1}(A) |
      Comm:    |   idle    |  AR(Attn,A) | AR(Attn,B)  |  AR(MoE,A)  |    AR(MoE,B)     |

    Strict invariants:
      - Only activates on eager solo prefill (`num_reqs == 1`, `num_decode_reqs == 0`,
        `T >= AR_OVERLAP_MIN_TOKENS`, `T % 256 == 0`, `T % (2 * world_size) == 0`).
      - `T0 = T // 2` is guaranteed a multiple of `128`, keeping `sm12_attn.py` relative
        bias `off` and `BLOCK_N=64` softmax chunks 0-ULP aligned with unsplit `T`!
      - Every rank runs the exact same FIFO collective order on a single `comm_stream`
        (zero concurrent collectives on the same `ncclComm_t` -> deadlock-free).
      - Every cross-stream tensor calls `.record_stream(...)` to prevent PyTorch
        CachingAllocator premature reuse.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._comm_stream: torch.cuda.Stream | None = None

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        if self._comm_stream is None:
            st = torch.cuda.Stream(device=self.device, priority=-1)
            # Warm up per-stream cuBLAS handle eagerly outside capture (used by repl_kv GEMMs!)
            if self.device.type == "cuda" and not torch.cuda.is_current_stream_capturing():
                with torch.cuda.stream(st):
                    _z = torch.empty((1, 16), dtype=torch.bfloat16, device=self.device)
                    _ = torch.mm(_z, _z.t())
                st.synchronize()
            self._comm_stream = st
        return self._comm_stream

    @staticmethod
    def eligible(
        num_tokens: int,
        world_size: int,
        attn_metadata: Any,
        first_attn_prefix: str,
    ) -> bool:
        if not AR_OVERLAP_ENABLED or world_size <= 1:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if num_tokens < AR_OVERLAP_MIN_TOKENS:
            return False
        # T0 = T // 2 must be divisible by 128 (relative-bias tile alignment) and world_size!
        if num_tokens % 256 != 0 or num_tokens % (2 * world_size) != 0:
            return False
        if not isinstance(attn_metadata, dict):
            return False
        md = attn_metadata.get(first_attn_prefix)
        if md is None:
            return False
        if getattr(md, "num_decode_reqs", 1) != 0 or getattr(md, "num_prefill_reqs", 0) != 1:
            return False
        seq_lens = getattr(md, "seq_lens", None)
        if seq_lens is None or seq_lens.shape[0] != 1:
            return False
        q_cpu = getattr(md, "query_lens_cpu", None)
        if q_cpu is None or len(q_cpu) != 1 or int(q_cpu[0]) != num_tokens:
            return False
        if getattr(md, "num_actual_tokens", -1) != num_tokens:
            return False
        return True

    @staticmethod
    def pad_target(
        num_tokens: int,
        world_size: int,
        attn_metadata: Any,
        first_attn_prefix: str,
    ) -> int:
        """Return aligned padded token count (> num_tokens) so unaligned solo tail prefill
        chunks (e.g. T=15,131 -> 15,360) stay inside DBO+SP instead of falling back to
        10.1 ms BF16 AllReduce + 563 ms Triton JIT stall (`INKLING_PREFILL_SP_PAD_TAIL=1`)."""
        if not (AR_OVERLAP_ENABLED and SP_PAD_TAIL) or world_size <= 1:
            return 0
        if torch.cuda.is_current_stream_capturing():
            return 0
        if num_tokens < max(AR_OVERLAP_MIN_TOKENS, SP_PAD_MIN_T):
            return 0
        align = max(256, 2 * world_size)
        if num_tokens % align == 0:
            return 0
        if not isinstance(attn_metadata, dict):
            return 0
        md = attn_metadata.get(first_attn_prefix)
        if md is None:
            return 0
        if getattr(md, "num_decode_reqs", 1) != 0 or getattr(md, "num_prefill_reqs", 0) != 1:
            return 0
        seq_lens = getattr(md, "seq_lens", None)
        if seq_lens is None or seq_lens.shape[0] != 1:
            return 0
        q_cpu = getattr(md, "query_lens_cpu", None)
        if q_cpu is None or len(q_cpu) != 1 or int(q_cpu[0]) != num_tokens:
            return 0
        if getattr(md, "num_actual_tokens", -1) != num_tokens:
            return 0
        t_pad = ((num_tokens + align - 1) // align) * align
        # Micro-batch A [0:t_pad//2) must contain strictly real tokens!
        if (t_pad // 2) >= num_tokens:
            return 0
        return t_pad

    _SOLO_META_CACHE: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def split_solo_metadata(
        attn_metadata: dict[str, Any],
        num_tokens: int,
        t0: int,
        device: torch.device,
        actual_num_tokens: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Slice vLLM's `attn_metadata` dict (all 48 Inkling2AttentionMetadata +
        all 48 InklingSconvMetadata) cleanly into micro-batch A=[0:t0) and B=[t0:T).

        Supports padded tail chunks (`actual_num_tokens < num_tokens` with `t0 < actual_num_tokens`):
          - Micro-batch A=[0:t0) contains `t0` real tokens.
          - Micro-batch B=[t0:num_tokens) contains `t1_real = actual_num_tokens - t0` real tokens
            followed by `num_tokens - actual_num_tokens` dummy slots (`slot_mapping == -1`,
            `is_padding == True`), keeping attention & KV/conv caches 0-ULP exact!
        """
        t1 = num_tokens - t0
        real_tokens = num_tokens if actual_num_tokens is None else int(actual_num_tokens)
        assert t0 <= real_tokens <= num_tokens
        t1_real = real_tokens - t0
        meta_a: dict[str, Any] = {}
        meta_b: dict[str, Any] = {}
        ckey = (str(device), t0, t1, t1_real)
        cached = PrefillAROverlap._SOLO_META_CACHE.get(ckey)
        if cached is None:
            cu_a = torch.tensor([0, t0], dtype=torch.int32, device=device)
            cu_b = torch.tensor([0, t1_real], dtype=torch.int32, device=device)
            zeros_b = torch.zeros((t1,), dtype=torch.int32, device=device)
            if len(PrefillAROverlap._SOLO_META_CACHE) < 16:
                PrefillAROverlap._SOLO_META_CACHE[ckey] = (cu_a, cu_b, zeros_b)
        else:
            cu_a, cu_b, zeros_b = cached
        qlen_a_np = np.array([t0], dtype=np.int64)
        qlen_b_np = np.array([t1_real], dtype=np.int64)

        # Deduplicate identical metadata objects (many layers share builders/tensors)
        seen_a: dict[int, Any] = {}
        seen_b: dict[int, Any] = {}

        for key, md in attn_metadata.items():
            mid = id(md)
            if mid in seen_a:
                meta_a[key] = seen_a[mid]
                meta_b[key] = seen_b[mid]
                continue

            if hasattr(md, "query_start") and hasattr(md, "seq_idx"):
                # InklingSconvMetadata
                from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvMetadata

                sa = InklingSconvMetadata(
                    block_table=md.block_table,
                    slot_mapping=md.slot_mapping[:t0],
                    seq_idx=md.seq_idx[:t0],
                    query_start=md.query_start[:t0],
                )
                sb = InklingSconvMetadata(
                    block_table=md.block_table,
                    slot_mapping=md.slot_mapping[t0:num_tokens],
                    seq_idx=md.seq_idx[t0:num_tokens],
                    query_start=zeros_b,
                )
                seen_a[mid], seen_b[mid] = sa, sb
                meta_a[key], meta_b[key] = sa, sb
            elif hasattr(md, "query_start_loc") and hasattr(md, "seq_lens_cpu"):
                # Inkling2AttentionMetadata
                full_seq = int(md.seq_lens_cpu[0])
                seq_a_val = full_seq - t1_real
                seq_a_t = torch.tensor([seq_a_val], dtype=md.seq_lens.dtype, device=device)
                seq_b_t = md.seq_lens

                ma = copy.copy(md)
                ma.num_actual_tokens = t0
                ma.num_prefill_tokens = t0
                ma.max_query_len = t0
                ma.max_seq_len = seq_a_val
                ma.query_start_loc = cu_a
                if hasattr(ma, "cu_seqlens_q"):
                    ma.cu_seqlens_q = cu_a
                ma.seq_lens = seq_a_t
                ma.slot_mapping = md.slot_mapping[:t0]
                ma.query_lens_cpu = qlen_a_np
                ma.seq_lens_cpu = np.array([seq_a_val], dtype=np.int64)

                mb = copy.copy(md)
                mb.num_actual_tokens = t1_real
                mb.num_prefill_tokens = t1_real
                mb.max_query_len = t1_real
                mb.max_seq_len = full_seq
                mb.query_start_loc = cu_b
                if hasattr(mb, "cu_seqlens_q"):
                    mb.cu_seqlens_q = cu_b
                mb.seq_lens = seq_b_t
                mb.slot_mapping = md.slot_mapping[t0:num_tokens]
                mb.query_lens_cpu = qlen_b_np
                mb.seq_lens_cpu = md.seq_lens_cpu

                seen_a[mid], seen_b[mid] = ma, mb
                meta_a[key], meta_b[key] = ma, mb
            else:
                # Non-Inkling metadata (e.g., draft layers untouched during target forward)
                seen_a[mid], seen_b[mid] = md, md
                meta_a[key], meta_b[key] = md, md

        return meta_a, meta_b

    def launch_all_reduce(
        self,
        partial: torch.Tensor,
        compute_stream: torch.cuda.Stream,
    ) -> _AsyncHandle:
        """Issue `all_reduce(partial)` asynchronously on `comm_stream`."""
        from vllm.distributed import tensor_model_parallel_all_reduce

        ev_ready = torch.cuda.Event()
        ev_ready.record(compute_stream)
        cs = self.comm_stream
        cs.wait_event(ev_ready)
        partial.record_stream(cs)
        with torch.cuda.stream(cs):
            out = tensor_model_parallel_all_reduce(partial)
            ev_done = torch.cuda.Event()
            ev_done.record(cs)
        return _AsyncHandle(out=out, done_event=ev_done, comm_stream=cs)

    def launch_sp_rs_halo(
        self,
        sp: PrefillSeqParallel,
        partial: torch.Tensor,
        plan: PrefillSPPlan,
        compute_stream: torch.cuda.Stream,
    ) -> _AsyncHandle:
        """Issue SP `reduce_scatter + halo/sync RDMA AG` asynchronously on `comm_stream`.
        Note: remote `conv_state.kv_cache` scatter runs on `compute_stream` right after
        `wait()` so `kv_cache` writes stay ordered cleanly on `compute_stream`."""
        ev_ready = torch.cuda.Event()
        ev_ready.record(compute_stream)
        cs = self.comm_stream
        cs.wait_event(ev_ready)
        partial.record_stream(cs)
        with torch.cuda.stream(cs):
            x_with_halo, gathered_sync = sp.reduce_scatter_and_halo(
                partial, plan, sconv_mod=None
            )
            ev_done = torch.cuda.Event()
            ev_done.record(cs)
        return _AsyncHandle(
            out=x_with_halo,
            done_event=ev_done,
            comm_stream=cs,
            gathered_sync=gathered_sync,
        )

    def launch_sp_all_gather(
        self,
        out_shard: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        next_global_repl: GlobalKVReplWeights | None = None,
        fp8_ag_modes: tuple[str, ...] | None = None,
    ) -> list[_AsyncHandle]:
        """Issue SP `all_gather` (or `repl_kv` fused K/V proj + A2A for Q=3 global layers)
        asynchronously on `comm_stream`."""
        ev_ready = torch.cuda.Event()
        ev_ready.record(compute_stream)
        cs = self.comm_stream
        cs.wait_event(ev_ready)
        out_shard.record_stream(cs)
        nq = int(out_shard.shape[0])
        handles: list[_AsyncHandle] = []
        _default_ag_mode = "rhd" if SP_AG_MODE != "nccl" else "bf16"
        with torch.cuda.stream(cs):
            if next_global_repl is not None and nq == 3:
                q_mode = fp8_ag_modes[0] if (SP_FP8_AG and fp8_ag_modes is not None) else _default_ag_mode
                g_q = _tp_all_gather_mode(out_shard[0], q_mode)
                ev_q = torch.cuda.Event()
                ev_q.record(cs)
                handles.append(_AsyncHandle(out=g_q, done_event=ev_q, comm_stream=cs))
                kv_pre = global_kv_proj_fused_a2a(
                    out_shard[1], out_shard[2], next_global_repl, _tp_all_to_all_dim0
                )
                ev_kv = torch.cuda.Event()
                ev_kv.record(cs)
                h_kv = _AsyncHandle(out=kv_pre, done_event=ev_kv, comm_stream=cs)
                handles.append(h_kv)
                handles.append(h_kv)
                return handles
            for q in range(nq):
                if SP_FP8_AG and fp8_ag_modes is not None and len(fp8_ag_modes) == nq:
                    g = _tp_all_gather_mode(out_shard[q], fp8_ag_modes[q])
                elif SP_AG_MODE != "nccl":
                    g = _tp_all_gather_mode(out_shard[q], "rhd")
                else:
                    g = _tp_all_gather_dim0(out_shard[q])
                ev_done = torch.cuda.Event()
                ev_done.record(cs)
                handles.append(_AsyncHandle(out=g, done_event=ev_done, comm_stream=cs))
        return handles

    def all_gather_tensor_on_comm(
        self,
        tensor: torch.Tensor,
        compute_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Run a single `all_gather(dim=0)` on `comm_stream` and synchronize `compute_stream`
        on its completion event (keeps ALL NCCL collectives strictly on `comm_stream`!)."""
        ev_ready = torch.cuda.Event()
        ev_ready.record(compute_stream)
        cs = self.comm_stream
        cs.wait_event(ev_ready)
        tensor.record_stream(cs)
        with torch.cuda.stream(cs):
            out = _tp_all_gather_dim0(tensor)
            ev_done = torch.cuda.Event()
            ev_done.record(cs)
        compute_stream.wait_event(ev_done)
        out.record_stream(compute_stream)
        return out

    def launch_sp_sublayer(
        self,
        sp: PrefillSeqParallel,
        partial: torch.Tensor,
        sconv_mod: Any,
        plan: PrefillSPPlan,
        conv_out: torch.Tensor,
        stack: torch.Tensor,
        mix_last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: Any,
        normalize_values: bool,
        compute_stream: torch.cuda.Stream,
        conv_addend: torch.Tensor | None = None,
        conv_res: torch.Tensor | None = None,
        need_aux_shard: torch.Tensor | None = None,
        fp8_ag_modes: tuple[str, ...] | None = None,
        next_global_repl: GlobalKVReplWeights | None = None,
    ) -> _AsyncSPSublayerHandle:
        """Execute an ENTIRE SP sublayer (`RS -> halo+cache_sync -> sharded sconv+AttnRes -> AG`)
        asynchronously on `comm_stream` so BOTH `RS` AND `AG` overlap 100% with the other
        micro-batch's `Attn`/`MoE` compute on `compute_stream`!

        Safety invariants:
          - `conv_state.kv_cache`, `stack_a/b`, `attn_out_a/b` are exclusively written/read
            by this method on the single FIFO `comm_stream` (strictly ordered A -> B across
            every sublayer; zero races with `compute_stream`'s Attention/MoE).
          - `ev_ready` ensures `partial` from `compute_stream` is ready before RS reads it.
          - `ev_done` ensures all outputs (and `conv_state.kv_cache` on final drain) are
            complete before `compute_stream` consumes them.
        """
        ev_ready = torch.cuda.Event()
        ev_ready.record(compute_stream)
        cs = self.comm_stream
        cs.wait_event(ev_ready)
        partial.record_stream(cs)
        with torch.cuda.stream(cs):
            x_with_halo, _ = sp.reduce_scatter_and_halo(
                partial, plan, sconv_mod=sconv_mod
            )
            out_shard = run_sharded_sconv_attn_res(
                sconv_mod=sconv_mod,
                x_with_halo=x_with_halo,
                plan=plan,
                conv_out=conv_out,
                stack=stack,
                mix_last=mix_last,
                queries=queries,
                norm=norm,
                normalize_values=normalize_values,
                conv_addend=conv_addend,
                conv_res=conv_res,
            )
            outs = sp.gather_queries(
                out_shard,
                next_global_repl=next_global_repl,
                fp8_ag_modes=fp8_ag_modes,
            )
            aux_g = (
                _tp_all_gather_dim0(need_aux_shard)
                if need_aux_shard is not None
                else None
            )
            ev_done = torch.cuda.Event()
            ev_done.record(cs)
        return _AsyncSPSublayerHandle(
            outs=outs, done_event=ev_done, comm_stream=cs, aux=aux_g
        )

    def launch_sp_rs_halo_only(
        self,
        sp: PrefillSeqParallel,
        partial: torch.Tensor,
        sconv_mod: Any,
        plan: PrefillSPPlan,
        compute_stream: torch.cuda.Stream,
    ) -> _AsyncHandle:
        """Stage 1 of Hybrid NIC-First Interleaved DBO+SP (`INKLING_PREFILL_DBO_INTERLEAVE=1`):
        runs ONLY `reduce_scatter + halo/cache_sync` on `comm_stream` and returns an
        `_AsyncHandle` whose `.wait(compute_stream)` delivers `x_with_halo` to `compute_stream`
        so `run_sharded_sconv_attn_res` overlaps behind the next collective on `comm_stream`!"""
        ev_ready = torch.cuda.Event()
        ev_ready.record(compute_stream)
        cs = self.comm_stream
        cs.wait_event(ev_ready)
        partial.record_stream(cs)
        with torch.cuda.stream(cs):
            x_with_halo, gathered_sync = sp.reduce_scatter_and_halo(
                partial, plan, sconv_mod=sconv_mod
            )
            ev_done = torch.cuda.Event()
            ev_done.record(cs)
        return _AsyncHandle(
            out=x_with_halo,
            done_event=ev_done,
            comm_stream=cs,
            gathered_sync=gathered_sync,
        )

    def launch_sp_gather_only(
        self,
        sp: PrefillSeqParallel,
        out_shard: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        need_aux_shard: torch.Tensor | None = None,
        fp8_ag_modes: tuple[str, ...] | None = None,
        next_global_repl: GlobalKVReplWeights | None = None,
    ) -> _AsyncSPSublayerHandle:
        """Stage 2 of Hybrid NIC-First Interleaved DBO+SP (`INKLING_PREFILL_DBO_INTERLEAVE=1`):
        runs ONLY `gather_queries (+ optional aux AG)` on `comm_stream` after `compute_stream`
        produces `out_shard` (and optional `need_aux_shard`)."""
        ev_ready = torch.cuda.Event()
        ev_ready.record(compute_stream)
        cs = self.comm_stream
        cs.wait_event(ev_ready)
        out_shard.record_stream(cs)
        if need_aux_shard is not None:
            need_aux_shard.record_stream(cs)
        with torch.cuda.stream(cs):
            outs = sp.gather_queries(
                out_shard,
                next_global_repl=next_global_repl,
                fp8_ag_modes=fp8_ag_modes,
            )
            aux_g = (
                _tp_all_gather_dim0(need_aux_shard)
                if need_aux_shard is not None
                else None
            )
            ev_done = torch.cuda.Event()
            ev_done.record(cs)
        return _AsyncSPSublayerHandle(
            outs=outs, done_event=ev_done, comm_stream=cs, aux=aux_g
        )


@contextlib.contextmanager
def patch_forward_attn_metadata(
    new_metadata: dict[str, Any],
    pad_slice: slice | None = None,
    full_num_tokens: int = 0,
):
    """Temporarily swap `get_forward_context().attn_metadata` (and slice `is_padding`
    when present) for a micro-batch, restoring both unconditionally on exit."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    orig_meta = fc.attn_metadata
    orig_pad = getattr(fc, "is_padding", None)
    fc.attn_metadata = new_metadata
    if (
        pad_slice is not None
        and orig_pad is not None
        and full_num_tokens > 0
        and orig_pad.shape[0] == full_num_tokens
    ):
        fc.is_padding = orig_pad[pad_slice]
    try:
        yield
    finally:
        fc.attn_metadata = orig_meta
        if orig_pad is not None:
            fc.is_padding = orig_pad


def pad_prefill_sp_inputs(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attn_metadata: dict[str, Any] | None,
    is_padding: torch.Tensor | None,
    target_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None, torch.Tensor]:
    """Pad an unaligned tail prefill chunk `[T, H]` -> `[target_tokens, H]` with dummy
    `slot_mapping = -1` and `is_padding[T:] = True` so it executes inside SP/DBO with
    0-ULP exactness on `[0:T)` and zero KV/conv-cache pollution."""
    orig_tokens = int(inputs_embeds.shape[0])
    pad_n = target_tokens - orig_tokens
    assert pad_n > 0
    device = inputs_embeds.device

    pos_pad = torch.cat(
        [positions[:orig_tokens], torch.zeros((pad_n,), dtype=positions.dtype, device=device)],
        dim=0,
    )
    emb_pad = torch.cat(
        [inputs_embeds, inputs_embeds.new_zeros((pad_n, inputs_embeds.shape[1]))],
        dim=0,
    )
    if is_padding is not None and int(is_padding.shape[0]) == orig_tokens:
        pad_mask = torch.cat(
            [is_padding, torch.ones((pad_n,), dtype=is_padding.dtype, device=is_padding.device)],
            dim=0,
        )
    else:
        pad_mask = torch.zeros((target_tokens,), dtype=torch.bool, device=device)
        pad_mask[orig_tokens:] = True

    if not isinstance(attn_metadata, dict):
        return pos_pad, emb_pad, attn_metadata, pad_mask

    padded_meta: dict[str, Any] = {}
    seen: dict[int, Any] = {}
    neg1_cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
    zero_cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}

    def _neg1(t: torch.Tensor) -> torch.Tensor:
        k = (t.dtype, t.device)
        v = neg1_cache.get(k)
        if v is None:
            v = torch.full((pad_n,), -1, dtype=t.dtype, device=t.device)
            neg1_cache[k] = v
        return v

    def _zero(t: torch.Tensor) -> torch.Tensor:
        k = (t.dtype, t.device)
        v = zero_cache.get(k)
        if v is None:
            v = torch.zeros((pad_n,), dtype=t.dtype, device=t.device)
            zero_cache[k] = v
        return v

    for key, md in attn_metadata.items():
        mid = id(md)
        if mid in seen:
            padded_meta[key] = seen[mid]
            continue
        if hasattr(md, "query_start") and hasattr(md, "seq_idx"):
            from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvMetadata

            sm = md.slot_mapping[:orig_tokens]
            si = md.seq_idx[:orig_tokens]
            qs = md.query_start[:orig_tokens]
            sp_md = InklingSconvMetadata(
                block_table=md.block_table,
                slot_mapping=torch.cat([sm, _neg1(sm)], dim=0),
                seq_idx=torch.cat([si, _zero(si)], dim=0),
                query_start=torch.cat([qs, _zero(qs)], dim=0),
            )
            seen[mid] = sp_md
            padded_meta[key] = sp_md
        elif hasattr(md, "query_start_loc") and hasattr(md, "seq_lens_cpu"):
            mp = copy.copy(md)
            sm = md.slot_mapping[:orig_tokens]
            mp.slot_mapping = torch.cat([sm, _neg1(sm)], dim=0)
            seen[mid] = mp
            padded_meta[key] = mp
        else:
            seen[mid] = md
            padded_meta[key] = md

    return pos_pad, emb_pad, padded_meta, pad_mask


@contextlib.contextmanager
def patch_prefill_pad_context(
    padded_metadata: dict[str, Any] | None,
    padded_is_padding: torch.Tensor,
):
    """Temporarily install padded `attn_metadata` and `is_padding` on `ForwardContext`,
    restoring both unconditionally on exit (even if `is_padding` was originally None)."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    orig_meta = fc.attn_metadata
    has_orig_pad = hasattr(fc, "is_padding")
    orig_pad = getattr(fc, "is_padding", None)
    if padded_metadata is not None:
        fc.attn_metadata = padded_metadata
    fc.is_padding = padded_is_padding
    try:
        yield
    finally:
        fc.attn_metadata = orig_meta
        if has_orig_pad:
            fc.is_padding = orig_pad
        elif hasattr(fc, "is_padding"):
            delattr(fc, "is_padding")
