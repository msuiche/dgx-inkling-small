# SPDX-License-Identifier: Apache-2.0
"""Runtime Tree Speculation Engine for Inkling-2-Small C=1 Bit-Exact Decode.

Owns:
  1. `TreeSpecState`: CUDA-graph-safe persistent device buffers (mirroring `ROWAR["buf"]`)
     holding `tree_active`, `tree_depth[M]`, `tree_parent[M]`, `tree_sconv_anc[M, 4]`,
     `tree_path[M, M]`, `tree_accepted_path[M]`, `tree_needs_compact[1]`.
     - Zero new CUDA graphs needed for M <= 8 (reuses existing M=8 decode graph!).
     - Linear mode simply keeps identity tables (`depth[i]=i, parent[i]=i-1, anc[i,l]=i-l, path[i,d]=d`).
  2. `build_tree_tables(parents, max_m)`: computes exact topological `depth`, `sconv_anc[M, 4]`
     (for W=4 ShortConv: `attn_sconv`, `mlp_sconv`, `k_sconv`, `v_sconv`), and canonical-lane
     `tree_path[M, M]` (for 0-ULP `_attn_kernel` tail split).
  3. `ExactDinkelbachTreeSelector`: exact rooted subtree knapsack DP maximizing
     `E[accepted_tokens(T)] - lam * step_ms(n_eff(T))` over DSpark beam-2/3 + SAM candidates.
  4. `apply_tree_positions_inplace(positions, base_pos, tree_depth, n_live)`:
     updates `positions[i] = P + tree_depth[i]` AFTER `prepare_attn` has assigned disjoint
     linear physical slots `phys(P + 0) .. phys(P + M - 1)` so siblings NEVER collide in
     `KVCache` or `ConvCache`!
"""
from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

TREE_SPEC_ENABLED = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
TREE_SPEC_MAX_M = int(os.environ.get("INKLING_TREE_SPEC_MAX_M", "8") or "8")
TREE_WS_MAX = max(1, min(4, int(os.environ.get("INKLING_TREE_WS_MAX", "2") or "2")))

# Global registry (mirroring rdma_all_reduce.ROWAR) so target ops (`sconv_ops.py`, `sm12_attn.py`,
# `sync_free.py`) read persistent CUDA-graph-captured device pointers with zero signature churn.
TREE_STATE: dict[str, object] = {
    "scope": 0,           # active padded batch size M (e.g., 8 or 16)
    "state": None,        # TreeSpecState instance
    "suppressed": False,  # True during prefill / B>1 / profile forwards
}

_TREE_TARGET_LIB_PATH = os.environ.get(
    "INKLING_TREE_TARGET_LIB", "/rdma/libinkling_tree_target_exact.so"
)
_TREE_TARGET_LIB: Any = None
_TREE_TARGET_LIB_TRIED = False


def get_active_tree_state(num_tokens: int) -> "TreeSpecState | None":
    if not TREE_SPEC_ENABLED or TREE_STATE.get("suppressed", False):
        return None
    if TREE_STATE.get("scope") == num_tokens:
        return TREE_STATE.get("state")  # type: ignore[return-value]
    return None


def get_tree_target_lib():
    """Load `/rdma/libinkling_tree_target_exact.so` once with exact verified C ABI argtypes."""
    global _TREE_TARGET_LIB, _TREE_TARGET_LIB_TRIED
    if _TREE_TARGET_LIB_TRIED:
        return _TREE_TARGET_LIB
    _TREE_TARGET_LIB_TRIED = True
    if not os.path.exists(_TREE_TARGET_LIB_PATH):
        return None
    lib = ctypes.CDLL(_TREE_TARGET_LIB_PATH)
    # 1. sconv_tree_exact_fwd (31 args: 14 ptr, 2 int, 4 i64, 2 int, 1 i64, 1 f32, 6 int, 1 u64)
    lib.sconv_tree_exact_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.sconv_tree_exact_fwd.restype = ctypes.c_int
    # 2. attn_prologue_tree_exact_fwd (56 args: 27 ptr, 2 int, 10 i64, 1 int, 1 i64, 2 int, 2 i64, 5 f32, 5 int, 1 u64)
    lib.attn_prologue_tree_exact_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.attn_prologue_tree_exact_fwd.restype = ctypes.c_int
    # 3. tree_compact_caches_fwd (25 args: 11 ptr, 2 int, 2 i64, 2 int, 2 i64, 2 int, 1 i64, 2 int, 1 u64)
    lib.tree_compact_caches_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int,
        ctypes.c_int, ctypes.c_uint64,
    ]
    lib.tree_compact_caches_fwd.restype = ctypes.c_int
    _TREE_TARGET_LIB = lib
    return lib


def build_tree_tables(
    parents: Sequence[int],
    max_m: int = 8,
    conv_w: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact target-verifier tree lookup tables from topological `parents` array.

    Invariants enforced:
      - `parents[0] == -1` (root = anchor token at logical position `P`, depth 0).
      - For every live node `i in 1..n_live-1`: `0 <= parents[i] < i` (strict topological order).
        This topological order guarantees `accepted_path[d] >= d` for every `d`, making
        post-rejection ascending-depth in-place cache compaction 100% race-free!
      - Unused padded slots `i in n_live..max_m-1` get `parent=-1, depth=0`, and safe
        sconv/path entries (never executed thanks to `nlive_buf` / `slot=-1`).

    Returns:
      - `parent_arr`    [max_m]          int32: parent node index (-1 for root/pad)
      - `depth_arr`     [max_m]          int32: depth from root (`0` for root)
      - `sconv_anc`     [max_m, conv_w]  int32: `sconv_anc[i, lag]` = intra-batch ancestor
        index at `lag in {0, 1, 2, 3}`, or `-1` if that lag reaches into the committed
        prefix `ConvCache` at logical position `Pos[i] - lag` (which is strictly `< P`!).
      - `tree_path`     [max_m, max_m]   int32: `tree_path[i, d]` = node index at depth `d`
        along the root->i path (for `0 <= d <= depth[i]`, else `-1`). Used by `sm12_attn`
        to load ancestor keys/values into their CANONICAL `kpos = P + d` lanes for 0-ULP parity!
    """
    n_live = len(parents)
    assert 1 <= n_live <= max_m, f"n_live={n_live} must be in 1..{max_m}"
    assert parents[0] == -1, f"root parent must be -1, got {parents[0]}"

    parent_arr = np.full((max_m,), -1, dtype=np.int32)
    depth_arr = np.zeros((max_m,), dtype=np.int32)
    sconv_anc = np.full((max_m, conv_w), -1, dtype=np.int32)
    tree_path = np.full((max_m, max_m), -1, dtype=np.int32)

    for i in range(n_live):
        p = int(parents[i])
        if i == 0:
            parent_arr[0] = -1
            depth_arr[0] = 0
            tree_path[0, 0] = 0
        else:
            assert 0 <= p < i, f"Tree must be topologically ordered: parents[{i}]={p} >= {i}"
            parent_arr[i] = p
            d = int(depth_arr[p]) + 1
            assert d < max_m, f"Tree depth {d} >= max_m {max_m}"
            depth_arr[i] = d
            tree_path[i, :d] = tree_path[p, :d]
            tree_path[i, d] = i

        # Populate W=4 ShortConv intra-batch ancestor pointers for lag = 0..conv_w-1
        cur = i
        for lag in range(conv_w):
            sconv_anc[i, lag] = cur
            if cur >= 0:
                cur = int(parent_arr[cur])

    # Safe identity defaults for unused tail rows (never read when nlive_buf == n_live)
    for i in range(n_live, max_m):
        parent_arr[i] = -1
        depth_arr[i] = 0
        sconv_anc[i, 0] = i
        tree_path[i, 0] = 0

    return parent_arr, depth_arr, sconv_anc, tree_path


class TreeSpecState:
    """Persistent device-side metadata buffers for CUDA-graph-safe 0-ULP tree verification."""

    def __init__(self, max_m: int = 8, device: torch.device | str = "cpu"):
        self.max_m = max_m
        self.device = torch.device(device)
        self.ws_max = TREE_WS_MAX
        # Compact [4] int32 metadata for 4-tile BM=16 two-branch attention (Triton & fused SM121a):
        #   [tree_active, best_wp, best_ws, n_live]
        self.tree_meta = torch.tensor([0, max_m - 1, 0, max_m], dtype=torch.int32, device=self.device)
        # Flag [1] int32: 0 = linear chain (fast path, 0 overhead), 1 = active tree
        self.tree_active = torch.zeros((1,), dtype=torch.int32, device=self.device)
        # Number of live tree nodes [1] int32 (synchronized with ROWAR["buf"])
        self.n_live = torch.tensor([max_m], dtype=torch.int32, device=self.device)
        # Tree structure tensors
        self.tree_parent = torch.full((max_m,), -1, dtype=torch.int32, device=self.device)
        self.tree_depth = torch.arange(max_m, dtype=torch.int32, device=self.device)
        self.tree_sconv_anc = torch.full((max_m, 4), -1, dtype=torch.int32, device=self.device)
        self.tree_path = torch.full((max_m, max_m), -1, dtype=torch.int32, device=self.device)
        # Post-rejection compaction scratch (written by _fused_tree_reject_kernel)
        self.accepted_path = torch.arange(max_m, dtype=torch.int32, device=self.device)
        self.accepted_len = torch.ones((1,), dtype=torch.int32, device=self.device)  # = depth + 1 = num_sampled
        self.needs_compact = torch.zeros((1,), dtype=torch.int32, device=self.device)
        # Registered per-KV/Conv-group cache pointer tables populated lazily by Inkling2Model
        self.compact_groups: list[dict[str, Any]] | None = None
        self._dummy_pos = torch.zeros((max_m,), dtype=torch.int64, device=self.device)
        self.reset_linear(max_m)

    def reset_linear(self, n_live: int | None = None) -> None:
        nl = self.max_m if n_live is None else int(n_live)
        parents = [-1] + list(range(nl - 1))
        self.set_tree(parents, is_tree=False)

    def set_tree(self, parents: Sequence[int], is_tree: bool | None = None) -> None:
        n_live = len(parents)
        p_arr, d_arr, anc_arr, path_arr = build_tree_tables(parents, max_m=self.max_m)
        if is_tree is None:
            is_tree = any(int(parents[i]) != i - 1 for i in range(1, n_live))
        self.tree_active.fill_(1 if is_tree else 0)
        self.n_live.fill_(n_live)
        self.tree_parent.copy_(torch.from_numpy(p_arr), non_blocking=True)
        self.tree_depth.copy_(torch.from_numpy(d_arr), non_blocking=True)
        self.tree_sconv_anc.copy_(torch.from_numpy(anc_arr), non_blocking=True)
        self.tree_path.copy_(torch.from_numpy(path_arr), non_blocking=True)
        best_wp = 0
        for i in range(1, n_live):
            if int(parents[i]) == i - 1:
                best_wp = i
            else:
                break
        rem = n_live - 1 - best_wp
        is_2b = (rem == 0) or (
            int(parents[best_wp + 1]) == 0
            and all(int(parents[best_wp + 1 + j]) == best_wp + j for j in range(1, rem))
        )
        best_ws = rem if is_2b else 0
        meta_arr = np.array(
            [1 if (is_tree and is_2b and best_ws > 0) else 0, best_wp, best_ws, n_live],
            dtype=np.int32,
        )
        self.tree_meta.copy_(torch.from_numpy(meta_arr), non_blocking=True)

    def bind_global(self) -> None:
        TREE_STATE["scope"] = self.max_m
        TREE_STATE["state"] = self


def compact_tree_kv_conv(
    tree_st: TreeSpecState,
    positions: torch.Tensor,
    *,
    use_pdl: bool = True,
) -> int:
    """Compact all 48 layers' KV + Conv caches across every KV/Conv cache group and restore
    contiguous `Positions[0..acc_len-1] = P + d`. 1.5 us/group no-op when `needs_compact == 0`!"""
    groups = tree_st.compact_groups
    if not groups:
        return 0
    lib = get_tree_target_lib()
    if lib is None:
        return -1
    stream_u64 = int(torch.cuda.current_stream(positions.device).cuda_stream) if positions.is_cuda else 0
    pdl_i = 1 if use_pdl else 0
    last_rc = 0
    for grp in groups:
        rc = lib.tree_compact_caches_fwd(
            grp["kc_ptrs"].data_ptr(),
            grp["vc_ptrs"].data_ptr(),
            grp["cc_ptrs"].data_ptr(),
            0,
            0,
            positions.data_ptr(),
            grp["kv_slot_map"].data_ptr(),
            grp["conv_slot_map"].data_ptr(),
            tree_st.accepted_path.data_ptr(),
            tree_st.accepted_len.data_ptr(),
            tree_st.needs_compact.data_ptr(),
            grp["num_layers"],
            0,
            grp["stride_kv_blk"],
            grp["stride_kv_n"],
            grp["kv_bs"],
            grp["kv_elems"],
            grp["stride_c_blk"],
            grp["stride_c_n"],
            grp["c_bs"],
            grp["conv_elems"],
            0,
            0,
            pdl_i,
            stream_u64,
        )
        if rc != 0:
            last_rc = rc
    return last_rc


def compact_tree_packed_aux(
    tree_st: TreeSpecState,
    packed_aux: torch.Tensor,
    num_aux: int,
    aux_dim: int,
    *,
    use_pdl: bool = False,
) -> int:
    """Compact `speculator.hidden_states` (`[T, num_aux * aux_dim]`) in-place after `torch.cat`
    when a secondary branch won (`needs_compact == 1`); ~1.5 us no-op when `needs_compact == 0`!"""
    if num_aux <= 0 or aux_dim <= 0:
        return 0
    lib = get_tree_target_lib()
    if lib is None:
        return -1
    stream_u64 = int(torch.cuda.current_stream(packed_aux.device).cuda_stream) if packed_aux.is_cuda else 0
    return int(
        lib.tree_compact_caches_fwd(
            0,
            0,
            0,
            0,
            packed_aux.data_ptr(),
            tree_st._dummy_pos.data_ptr(),
            0,
            0,
            tree_st.accepted_path.data_ptr(),
            tree_st.accepted_len.data_ptr(),
            tree_st.needs_compact.data_ptr(),
            0,
            int(num_aux),
            0,
            0,
            16,
            128,
            0,
            0,
            4,
            9472,
            int(packed_aux.stride(0)),
            int(aux_dim),
            1 if use_pdl else 0,
            stream_u64,
        )
    )


@dataclass
class CandidateNode:
    """Candidate token node in the DSpark beam + SAM proposal pool."""
    token_id: int
    parent_idx: int      # index into candidate list (-1 for anchor root)
    cond_prob: float     # P(node matches target | parent matched target)
    is_sibling: bool     # True if secondary/SAM sibling sharing parent with primary node
    source: str = "dspark_p1"  # "root", "dspark_p1", "dspark_p2", "sam_branch", "sam_ext"


@dataclass
class SelectedTree:
    tokens: list[int]          # [n_live] draft token IDs (tokens[0] = anchor)
    parents: list[int]         # [n_live] topological parent indices (parents[0] = -1)
    depths: list[int]          # [n_live] depths (depths[0] = 0)
    expected_tokens: float     # E[num_sampled]
    dinkelbach_obj: float      # E[num_sampled] - lam * step_ms(n_eff)
    is_branched: bool          # True if any non-linear branch was selected


class ExactDinkelbachTreeSelector:
    """Exact rooted subtree knapsack DP maximizing renewal-reward Dinkelbach objective:
        argmax_{T rooted at 0, |T| <= max_m}  (sum_{u in T} P(path to u)) - lam * step_ms(n_eff(T))
    where `n_eff(T) = sum_{u in T} (1 - sib_disc if u.is_sibling else 1.0)`.

    Complexity: O(|C| * max_m^2) where |C| <= 20, max_m <= 16 -> < 3 us in Python / < 0.2 us on GPU!
    """

    def __init__(
        self,
        step_ms: Sequence[float],
        max_m: int = 8,
        sib_disc: float = 0.25,
    ):
        self.step_ms = np.asarray(step_ms, dtype=np.float64)
        self.max_m = max_m
        self.sib_disc = sib_disc

    def _interp_ms(self, n_eff: float) -> float:
        w = max(0.0, min(float(len(self.step_ms) - 1), n_eff - 1.0))
        lo = int(math.floor(w))
        hi = min(len(self.step_ms) - 1, lo + 1)
        f = w - lo
        return (1.0 - f) * float(self.step_ms[lo]) + f * float(self.step_ms[hi])

    def select(self, candidates: Sequence[CandidateNode], lam: float) -> SelectedTree:
        n = len(candidates)
        assert n >= 1 and candidates[0].parent_idx == -1
        children: list[list[int]] = [[] for _ in range(n)]
        path_prob = [0.0] * n
        path_prob[0] = 1.0
        for i in range(1, n):
            p = candidates[i].parent_idx
            assert 0 <= p < i, "Candidates must be topologically ordered"
            children[p].append(i)
            path_prob[i] = path_prob[p] * max(0.0, min(1.0, candidates[i].cond_prob))

        cap = min(self.max_m, n)
        NEG = -1e18
        dp_val: list[list[float]] = [[NEG] * (cap + 1) for _ in range(n)]
        dp_set: list[list[list[int] | None]] = [[None] * (cap + 1) for _ in range(n)]

        def dfs(u: int) -> int:
            dp_val[u][1] = path_prob[u]
            dp_set[u][1] = [u]
            cur_sz = 1
            for v in children[u]:
                v_sz = dfs(v)
                new_sz = min(cap, cur_sz + v_sz)
                nv = [NEG] * (cap + 1)
                ns: list[list[int] | None] = [None] * (cap + 1)
                for k1 in range(1, cur_sz + 1):
                    if dp_set[u][k1] is None:
                        continue
                    if dp_val[u][k1] > nv[k1]:
                        nv[k1] = dp_val[u][k1]
                        ns[k1] = dp_set[u][k1]
                    for k2 in range(1, min(v_sz, cap - k1) + 1):
                        if dp_set[v][k2] is None:
                            continue
                        val = dp_val[u][k1] + dp_val[v][k2]
                        if val > nv[k1 + k2]:
                            nv[k1 + k2] = val
                            ns[k1 + k2] = dp_set[u][k1] + dp_set[v][k2]
                dp_val[u] = nv
                dp_set[u] = ns
                cur_sz = new_sz
            return cur_sz

        dfs(0)
        best_obj = -1e18
        best_nodes = [0]
        best_exp = 1.0
        for s in range(1, cap + 1):
            nodes = dp_set[0][s]
            if nodes is None:
                continue
            n_eff = sum(
                (1.0 - self.sib_disc) if candidates[u].is_sibling else 1.0
                for u in nodes
            )
            obj = dp_val[0][s] - lam * self._interp_ms(n_eff)
            if obj > best_obj:
                best_obj = obj
                best_nodes = nodes
                best_exp = dp_val[0][s]

        # Canonical packing: primary chain nodes FIRST (so when primary wins, accepted_path[d] == d
        # and needs_compact == 0 -> 1.5 us no-op!), followed by secondary nodes in topological order!
        prim_nodes = sorted(u for u in best_nodes if candidates[u].source in ("root", "dspark_p1", "sam_ext"))
        sec_nodes = sorted(u for u in best_nodes if u not in set(prim_nodes))
        ordered = prim_nodes + sec_nodes
        old_to_new = {old: new for new, old in enumerate(ordered)}
        tokens = [candidates[u].token_id for u in ordered]
        parents = [-1 if candidates[u].parent_idx < 0 else old_to_new[candidates[u].parent_idx] for u in ordered]
        depths = [0] * len(ordered)
        for i in range(1, len(ordered)):
            depths[i] = depths[parents[i]] + 1
        is_branched = any(parents[i] != i - 1 for i in range(1, len(ordered)))

        return SelectedTree(
            tokens=tokens,
            parents=parents,
            depths=depths,
            expected_tokens=best_exp,
            dinkelbach_obj=best_obj,
            is_branched=is_branched,
        )


def tree_greedy_reject_reference(
    target_argmax_tokens: Sequence[int],  # [n_live] argmax(target_logits[node])
    draft_tokens: Sequence[int],          # [n_live] input_ids[node] (draft_tokens[0] = anchor)
    parents: Sequence[int],               # [n_live] parents[0] = -1, parents[i] < i
    n_scheduled_drafts: int,              # = num_draft_tokens in vLLM scheduler (= max_m - 1)
) -> tuple[list[int], int, int, list[int], bool]:
    """Exact 0-ULP Tree Greedy Rejection Sampler.

    Walks from root `cur = 0`:
      - Reads `t_tok = target_argmax_tokens[cur]` (the exact target greedy prediction at `cur`).
      - Searches live children `c in cur+1 .. n_live-1` with `parents[c] == cur` and
        `draft_tokens[c] >= 0` for `draft_tokens[c] == t_tok`.
      - If found: appends `t_tok` to `sampled`, advances `cur = c`, appends `c` to `accepted_path`,
        and repeats!
      - If not found (or `cur` is a leaf): appends `t_tok` as the final correction/bonus token and stops!

    Returns:
      - `sampled`: list of emitted tokens `[len = depth + 1]`
      - `num_sampled`: `depth + 1`
      - `num_rejected`: `n_scheduled_drafts - depth`
      - `accepted_path`: node indices `[0, c_1, ..., c_depth]` (strictly increasing, `accepted_path[d] >= d`)
      - `needs_compact`: True iff `any(accepted_path[d] != d for d in range(len(accepted_path)))`
    """
    n_live = len(parents)
    cur = 0
    accepted_path = [0]
    sampled: list[int] = []

    while True:
        t_tok = int(target_argmax_tokens[cur])
        sampled.append(t_tok)
        matched_child = -1
        for c in range(cur + 1, n_live):
            if parents[c] == cur and draft_tokens[c] >= 0 and draft_tokens[c] == t_tok:
                matched_child = c
                break
        if matched_child >= 0:
            accepted_path.append(matched_child)
            cur = matched_child
        else:
            break

    acc_depth = len(accepted_path) - 1
    num_sampled = acc_depth + 1
    num_rejected = n_scheduled_drafts - acc_depth
    needs_compact = any(accepted_path[d] != d for d in range(len(accepted_path)))
    return sampled, num_sampled, num_rejected, accepted_path, needs_compact
