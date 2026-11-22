#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GB10 (SM121a) Stage-by-Stage 0-ULP Diagnostic, Verifier & 48L Cold-L2 Benchmark
for `fused_decode_attn_exact_sm121_v2.cu` (`libfused_decode_attn_exact_sm121_v2.so`).

Run on GB10 (e.g. inside `inkling-tp8-r6` container):
    python3 verify_c1_attn_v2_gb10.py --build --verify --bench

Key Design Properties:
  1. Sets `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` AND live `best.env`
     flags (`INKLING_ATTN_FUSE_GS=1`, `INKLING_MGS_CUDA=1`) strictly BEFORE `import torch`!
  2. Fixes the `REPORT.md §9.5` harness artifact: when `_FUSE_GS=1`, `sm12_attn.rel_attention`
     writes per-split FP32 outputs into `sm12_attn._PENDING[2:4]` (`op`, `lp`) and leaves
     `out`/`lse` uninitialized! This harness captures Triton's ACTUAL `op`/`lp` from `_PENDING`
     before `gate_sink()` consumes it, and verifies BOTH bitwise `op`/`lp` AND final BF16 `y`
     against BOTH live `ns_mgs` (`merge_gate_sink_exact_fwd`) AND Triton `_merge_gate_sink_kernel`!
  3. Covers ALL THREE target decode paths:
       - Suite A: Linear C=1 (`B=1`, `M in {1, 4, 8}`, `nl in {1, 4, 8}`, `seq in {64, 150, 504, 505, 1024, 1536, 2048}`,
                  Local `S=8` & Global `S=16`) + 2x determinism check!
       - Suite B: `S8-P6v2` Fast 4-Tile `BM=16` Beam-2 Tree (`B=1, M=8`, `(wp, ws) in {(7,0),(6,1),(5,2),(4,2),(3,2)}`,
                  `past in {0, 64, 504, 505, 1024, 2048}`, Local `S=8` & Global `S=16`) vs live Triton
                  `HAS_TREE_2B=True` + `ns_mgs`, PLUS primary-chain `[0..wp]` 0-ULP parity vs linear!
       - Suite C: Batched C=32 (`S=1/2/4`, `FUSE_MT=0/1`, `evict_first=0/1`, `M=32..256`, `seq=150/1024/4096`,
                  hetero qlen 1..8, tail `NT<T`, zero-seqlen reqs)!
  4. Built-in Live PTX Inspector + Orthogonal Zero-Recompile Sweeper:
       On `--dump-ptx` or any non-zero diff, dumps the COMPLETE softmax inner loop of live
       `_attn_kernel.ptx` (every `fma.rn`, `add.rn`, `mul.rn`, `shfl.sync`, `ex2.approx`,
       `div.full`, `log2`) AND sweeps `(sum_mode, fma_mode, pv_mode, mgs_ftz)` without recompiling!
  5. Suite D (`--bench`): 48-layer (`40` local + `8` global) cold-L2 (`64 MB` flush) CUDA-graph
     timing benchmark for C=1 Linear, `S8-P6v2` 4-tile Beam-2 Tree, and C=32!
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CRITICAL: Match live `best.env` Triton/ptxas/MGS environment BEFORE importing torch!
# ---------------------------------------------------------------------------
if os.path.exists("/usr/local/cuda/bin/ptxas"):
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("INKLING_ATTN_FUSE_GS", "1")
os.environ.setdefault("INKLING_MGS_CUDA", "1")
os.environ.setdefault("INKLING_ATTN_BM16", "1")
os.environ.setdefault("INKLING_ATTN_BM16_STAGES", "1")
for _s7_cand in (
    os.environ.get("INKLING_TARGET_ATTN_LIB", ""),
    "/rdma/libinkling_target_attn_exact_s7v2.so",
    "/rdma/libinkling_target_attn_exact_s7.so",
    "/rdma/libinkling_target_attn_exact.so",
):
    if _s7_cand and os.path.exists(_s7_cand):
        os.environ.setdefault("INKLING_TARGET_ATTN_LIB", _s7_cand)
        break

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_V2_SO = "/rdma/libfused_decode_attn_exact_sm121_v2.so"
LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Library Loader (all 5 C ABI symbols matching patches-s6/sm12_attn_fused.py)
# ---------------------------------------------------------------------------
def load_v2_lib(path: str):
    lib = ctypes.CDLL(path)

    pro_base = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    pro_tail = [
        ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.attn_prologue_kvq_exact_fwd.argtypes = pro_base + pro_tail
    lib.attn_prologue_kvq_exact_fwd.restype = ctypes.c_int
    lib.attn_prologue_kvq_tree_exact_fwd.argtypes = pro_base + [ctypes.c_void_p] + pro_tail
    lib.attn_prologue_kvq_tree_exact_fwd.restype = ctypes.c_int

    attn_ptrs = [ctypes.c_void_p] * 17
    attn_tail = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.fused_decode_attn_otf_mgs_fwd.argtypes = attn_ptrs + attn_tail + [ctypes.c_uint64]
    lib.fused_decode_attn_otf_mgs_fwd.restype = ctypes.c_int

    lib.fused_decode_attn_otf_mgs_c32_fwd.argtypes = (
        attn_ptrs + attn_tail + [ctypes.c_int, ctypes.c_int, ctypes.c_uint64]
    )
    lib.fused_decode_attn_otf_mgs_c32_fwd.restype = ctypes.c_int

    lib.fused_decode_attn_otf_mgs_tree_fwd.argtypes = (
        attn_ptrs + [ctypes.c_void_p] + attn_tail + [ctypes.c_uint64]
    )
    lib.fused_decode_attn_otf_mgs_tree_fwd.restype = ctypes.c_int
    return lib


# ---------------------------------------------------------------------------
# FP32 / BF16 Bitwise Diff Utilities
# ---------------------------------------------------------------------------
def bf16_diff_count(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.view(torch.uint16) != b.view(torch.uint16)).sum().item())


def f32_ulp_stats(got: torch.Tensor, ref: torch.Tensor, valid_mask: torch.Tensor | None = None):
    """Return (n_bit_diffs, max_ulp, first_idx, got_val, ref_val, got_hex, ref_hex)."""
    ga = got.contiguous().view(torch.int32)
    rb = ref.contiguous().view(torch.int32)
    diff = ga != rb
    if valid_mask is not None:
        diff = diff & valid_mask
    n_diff = int(diff.sum().item())
    if n_diff == 0:
        return 0, 0, None, 0.0, 0.0, 0, 0
    # Signed magnitude -> lexicographical int32 for exact IEEE-754 ULP distance
    def _to_ord(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0, (0x80000000 - x.to(torch.int64)), x.to(torch.int64))
    ulps = (_to_ord(ga) - _to_ord(rb)).abs()
    ulps = torch.where(diff, ulps, torch.zeros_like(ulps))
    max_ulp = int(ulps.max().item())
    nz = torch.nonzero(diff)[0].tolist()
    idx = tuple(nz)
    gv = float(got[idx].item())
    rv = float(ref[idx].item())
    gh = int(got.contiguous().view(torch.uint32)[idx].item())
    rh = int(ref.contiguous().view(torch.uint32)[idx].item())
    return n_diff, max_ulp, nz, gv, rv, gh, rh


def compare_op_lp(v2_op, v2_lp, tri_op, tri_lp, nl: int | None = None):
    """Compare per-split FP32 `op` and `lp` safely (ignoring empty splits where lp == -inf)."""
    if v2_op is None or v2_lp is None or tri_op is None or tri_lp is None:
        return None
    if nl is not None:
        v2_op, v2_lp = v2_op[:, :nl], v2_lp[:, :nl]
        tri_op, tri_lp = tri_op[:, :nl], tri_lp[:, :nl]
    v2_neg = torch.isneginf(v2_lp)
    tri_neg = torch.isneginf(tri_lp)
    inf_mismatch = int((v2_neg != tri_neg).sum().item())
    fin = (~v2_neg) & (~tri_neg)
    d_lp, ulp_lp, idx_lp, glp, rlp, glh, rlh = f32_ulp_stats(v2_lp, tri_lp, valid_mask=fin)
    d_op, ulp_op, idx_op, gop, rop, goh, roh = f32_ulp_stats(
        v2_op, tri_op, valid_mask=fin.unsqueeze(-1).expand_as(v2_op)
    )
    return {
        "inf_mismatch": inf_mismatch,
        "d_lp": d_lp, "ulp_lp": ulp_lp, "idx_lp": idx_lp,
        "glp": glp, "rlp": rlp, "glh": glh, "rlh": rlh,
        "d_op": d_op, "ulp_op": ulp_op, "idx_op": idx_op,
        "gop": gop, "rop": rop, "goh": goh, "roh": roh,
    }


# ---------------------------------------------------------------------------
# Live Triton `_attn_kernel.ptx` Full Softmax Loop Dumper
# ---------------------------------------------------------------------------
def dump_live_attn_ptx(max_lines: int = 180):
    cache_root = os.environ.get("TRITON_CACHE_DIR", "/root/.triton/cache")
    ptxs = sorted(
        glob.glob(f"{cache_root}/**/_attn_kernel.ptx", recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    if not ptxs:
        print(f"  [PTX] No _attn_kernel.ptx found under {cache_root}")
        return
    if len(ptxs) > 1:
        print(f"\n  [PTX-CATALOG] Newest {min(3, len(ptxs))} _attn_kernel.ptx files:")
        for ci, cp in enumerate(ptxs[:3]):
            cl = Path(cp).read_text().splitlines()
            n_fma = sum(1 for ln in cl if "fma.rn.f32x2" in ln)
            n_shf = sum(1 for ln in cl if "shfl.sync" in ln)
            n_ex2 = sum(1 for ln in cl if "ex2.approx" in ln)
            n_lg2 = sum(1 for ln in cl if ("log2" in ln or "lg2." in ln))
            print(f"    [{ci}] {cp} ({len(cl)}L: fma2={n_fma} shfl={n_shf} ex2={n_ex2} log2={n_lg2})")
    p = Path(ptxs[0])
    lines = p.read_text().splitlines()
    print(f"\n  [PTX-DUMP] {p} ({len(lines)} lines)")
    # Find first fma.rn.f32x2 / ex2.approx.ftz.f32 cluster (the softmax inner loop!)
    ex2_idx = [i for i, ln in enumerate(lines) if "ex2.approx" in ln]
    fma_idx = [i for i, ln in enumerate(lines) if "fma.rn.f32x2" in ln]
    shfl_idx = [i for i, ln in enumerate(lines) if "shfl.sync" in ln]
    log2_idx = [i for i, ln in enumerate(lines) if ("log2" in ln or "lg2." in ln)]
    div_idx = [i for i, ln in enumerate(lines) if ("div.full" in ln or "div.rn" in ln)]
    print(
        f"    Counts: fma.rn.f32x2={len(fma_idx)}, shfl.sync={len(shfl_idx)}, "
        f"ex2.approx={len(ex2_idx)}, div={len(div_idx)}, log2={len(log2_idx)}"
    )
    if fma_idx:
        # Also print where %rd constants used by fma.rn.f32x2 are loaded!
        fma_line = lines[fma_idx[0]].strip()
        print(f"    First fma.rn.f32x2 at L{fma_idx[0]+1}: {fma_line}")
        parts = [tok.strip(",;") for tok in fma_line.split()]
        if len(parts) >= 3:
            reg = parts[2]
            defs = [f"L{i+1}: {ln.strip()}" for i, ln in enumerate(lines[:fma_idx[0]]) if reg in ln]
            print(f"    Defs of operand {reg}:")
            for d in defs[-6:]:
                print(f"      {d}")
    if ex2_idx:
        # Print COMPLETE softmax window: score+bias+max AND full ex2 -> tl.sum(p, 1) -> l_i!
        start = max(0, (fma_idx[0] if fma_idx else ex2_idx[0]) - 25)
        end = min(len(lines), ex2_idx[-1] + 240)
        print(f"    --- Softmax Slice 1 (bias+score+max, L{start+1}..L{min(end, start+85)}) ---")
        for i in range(start, min(end, start + 85)):
            s = lines[i].strip()
            if s and not s.startswith("//") and not s.startswith(".loc"):
                print(f"      L{i+1:4d}: {s}")
        s2 = max(start + 85, ex2_idx[0] - 4)
        print(f"    --- Softmax Slice 2 (ex2+sum_p+l_i, L{s2+1}..L{end}) ---")
        for i in range(s2, end):
            s = lines[i].strip()
            if s and not s.startswith("//") and not s.startswith(".loc"):
                print(f"      L{i+1:4d}: {s}")
    if log2_idx or div_idx:
        e_start = max(0, min((log2_idx[0] if log2_idx else len(lines)), (div_idx[0] if div_idx else len(lines))) - 12)
        e_end = min(len(lines), e_start + 90)
        print(f"    --- Softmax Slice 3 (epilogue log2/div, L{e_start+1}..L{e_end}) ---")
        for i in range(e_start, e_end):
            s = lines[i].strip()
            if s and not s.startswith("//") and not s.startswith(".loc"):
                print(f"      L{i+1:4d}: {s}")


# ---------------------------------------------------------------------------
# Diagnostic Mode Encoder/Decoder (matches fused_decode_attn_exact_sm121_v2.cu)
# ---------------------------------------------------------------------------
SUM_MODE_NAMES = {
    0: "B0(<2xf32>_tree)",
    1: "B1(4-2-1_eo)",
    2: "B2(adj_pair)",
    3: "B3(stride2_pair)",
    4: "B4(seq_0..7)",
    12: "B5(<2xf32>_seq)",
    13: "B6(s4_pair_half)",
    8: "B0_cross01",
    9: "B1_cross01",
    10: "B0_crossSeq",
    14: "BL1_2x2([1,1][1,32][2,2])",
    15: "BL2_2x2([1,2][2,16][2,2])",
    16: "BL1_w02([1,1][1,32][4,1])",
    30: "BL1_w01([1,1][1,32][4,1])",
    31: "BL1_seq([1,1][1,32][4,1])",
    17: "BL2_0([1,2][2,16][4,1])",
    18: "BL2_1([1,2][2,16][4,1])",
    19: "BL2_2([1,2][2,16][4,1])",
    20: "BL4_0([1,4][4,8][4,1])",
    21: "BL4_1([1,4][4,8][4,1])",
    22: "BL4_2([1,4][4,8][4,1])",
    23: "BL4_3([1,4][4,8][4,1])",
    5: "M0_s21_w02",
    6: "M1_s21_w02",
    7: "M2_s21_w02",
    11: "M0_s12_w01",
    24: "M0_s21_w01",
    25: "M1_s21_w01",
    26: "M2_s21_w01",
    27: "M1_s12_w01",
    28: "M0_s21_wSeq",
    29: "M1_s21_wSeq",
}


def encode_diag_mode(sum_mode: int = 0, fma_mode: int = 0, pv_mode: int = 0, mgs_ftz: int = 0, force_diag: bool = True) -> int:
    v = (sum_mode & 0x1F) | ((fma_mode & 3) << 5) | ((pv_mode & 1) << 7) | ((mgs_ftz & 1) << 8)
    if force_diag:
        v |= 0x200  # bit 9 forces TREE_MODE == -1 (including FP32 OP/LP dump at S=1!)
    return v


def run_orthogonal_sweep(run_fn, y_ref: torch.Tensor, tri_op: torch.Tensor, tri_lp: torch.Tensor, nl: int | None = None):
    """Zero-recompile sweep across ALL 32 sum_modes (Blocked [1,1]/[1,2]/[1,4]/[1,8] + complete MMA), fma_modes, pv_modes, mgs_ftz."""
    print("    [SWEEP] Orthogonal zero-recompile diagnostic sweep (32 sum_modes):")
    best = None
    sm_order = (
        14, 15,              # Blocked([1,1],[1,32],[2,2]) & Blocked([1,2],[2,16],[2,2])
        16, 30, 31,          # Blocked([1,1],[1,32],[4,1])
        17, 18, 19,          # Blocked([1,2],[2,16],[4,1])
        20, 21, 22, 23,      # Blocked([1,4],[4,8],[4,1])
        0, 1, 2, 3, 4, 12, 13, 8, 9, 10,  # Blocked([1,8],[4,8],[4,1])
        5, 6, 7, 24, 25, 26, 11, 27, 28, 29,  # Complete MMA(v2,warps=[1,4]) matrix
    )
    sm_res = []
    for sm in sm_order:
        code = encode_diag_mode(sum_mode=sm, fma_mode=0, pv_mode=0, mgs_ftz=0, force_diag=True)
        yt, op_t, lp_t = run_fn(code)
        dy = bf16_diff_count(yt[:nl] if nl else yt, y_ref[:nl] if nl else y_ref)
        st = compare_op_lp(op_t, lp_t, tri_op, tri_lp, nl=nl)
        dlp = st["d_lp"] if st else -1
        dop = st["d_op"] if st else -1
        sm_res.append((sm, dy, dlp, dop, code))
        if best is None or (dlp, dop, dy) < best[0]:
            best = ((dlp, dop, dy), f"sm={sm}({SUM_MODE_NAMES[sm]}),fma=0,pv=0,ftz=0 [tree_mode={code}]")
    # Print compactly in 3 family groups
    for grp_lbl, grp_sms in (
        ("BL1/BL2/BL4", sm_res[:12]),
        ("BL8(B0..B6)", sm_res[12:22]),
        ("MMA(M0..M2)", sm_res[22:]),
    ):
        print(f"      {grp_lbl:11s}: " + ", ".join(
            f"{SUM_MODE_NAMES[sm]}:dy={dy}/lp={dlp}/op={dop}" for sm, dy, dlp, dop, _ in grp_sms
        ))

    # 2. Sweep fma_mode, pv_mode, mgs_ftz on top-3 sum_modes (ranked by lp_u32, op_u32, dy)
    top_sms = sorted(sm_res, key=lambda x: (x[2], x[3], x[1]))[:3]
    for sm, _, _, _, _ in top_sms:
        extras = []
        for fma_m in (0, 1, 2):
            for pv_m in (0, 1):
                for ftz_m in (0, 1):
                    if fma_m == 0 and pv_m == 0 and ftz_m == 0:
                        continue
                    code = encode_diag_mode(sum_mode=sm, fma_mode=fma_m, pv_mode=pv_m, mgs_ftz=ftz_m, force_diag=True)
                    yt, op_t, lp_t = run_fn(code)
                    dy = bf16_diff_count(yt[:nl] if nl else yt, y_ref[:nl] if nl else y_ref)
                    st = compare_op_lp(op_t, lp_t, tri_op, tri_lp, nl=nl)
                    dlp = st["d_lp"] if st else -1
                    dop = st["d_op"] if st else -1
                    if (dlp == 0 and dop == 0) or (best and (dlp, dop, dy) < best[0]):
                        extras.append(f"(fma={fma_m},pv={pv_m},ftz={ftz_m})->dy={dy}/lp={dlp}/op={dop}[tm={code}]")
                    if best is None or (dlp, dop, dy) < best[0]:
                        best = ((dlp, dop, dy), f"sm={sm}({SUM_MODE_NAMES[sm]}),fma={fma_m},pv={pv_m},ftz={ftz_m} [tree_mode={code}]")
        if extras:
            print(f"      extra hits for {SUM_MODE_NAMES[sm]}: " + "; ".join(extras[:6]))
    print(f"      BEST config: {best[1]} -> d_lp_u32={best[0][0]}, d_op_u32={best[0][1]}, dy={best[0][2]}")


# ---------------------------------------------------------------------------
# Reference Runner Capturing ACTUAL Triton `op, lp`, `y_ns_mgs`, & `y_tri_mgs`
# ---------------------------------------------------------------------------
def run_triton_mgs_direct(sm12_attn, op, lp, gate, sinks, nt, nl_buf=None):
    """Run Triton's `_merge_gate_sink_kernel` directly on `(op, lp)` for exact oracle cross-check."""
    T = gate.shape[0]
    S, _, H, D = op.shape
    y_tri = torch.empty((T, H * D), dtype=torch.bfloat16, device=gate.device)
    sm12_attn._merge_gate_sink_kernel[(T,)](
        op, lp, gate, sinks, y_tri,
        op.stride(0), op.stride(1), op.stride(2),
        lp.stride(0), lp.stride(1), lp.stride(2),
        gate.stride(0), sinks.stride(0), nt,
        H=H, HP=8, D=D, S=S, TRIG=False,
        NLivePtr=nl_buf if nl_buf is not None else op,
        HAS_NLIVE=nl_buf is not None,
        num_warps=4, launch_pdl=False,
    )
    return y_tri


# ---------------------------------------------------------------------------
# Tree Metadata Helpers for S8-P6v2 (4-Tile BM=16 Two-Branch Tree, M <= 8, ws <= 2)
# ---------------------------------------------------------------------------
def make_s8p6v2_tree_metadata(past: int, M: int, wp: int, ws: int, blocks_per_req: int, BS: int, C_N: int, device):
    """Construct exact `S8-P6v2` 4-tile BM=16 Beam-2 Tree metadata (`wp + ws <= M - 1, ws <= 2`):
      - Primary chain tokens `0..wp` at depths `0..wp` (logical positions `past..past+wp`)
      - Secondary branch tokens `wp+1..wp+ws` at depths `1..ws` (logical positions `past+1..past+ws`),
        with physical KV slots `past + wp + 1 .. past + wp + ws`!
      - `TreeMeta = [1 if ws > 0 else 0, wp, ws, 1 + wp + ws]`
      - `TreeSconvAnc[t, lag]` for `lag in 0..3`: intra-step ancestor token index or `-1` for cached!
    """
    nl = 1 + wp + ws
    assert 1 <= nl <= M <= 8 and 0 <= ws <= 2
    # Ancestor chain per token:
    #   Primary t in 0..wp: parent(t) = t-1 (or -1 if t==0)
    #   Secondary t = wp+1+j (j in 0..ws-1): parent(wp+1) = 0, parent(wp+2) = wp+1
    parent = [-1] * M
    depth = [0] * M
    for i in range(1, wp + 1):
        parent[i] = i - 1
        depth[i] = i
    for j in range(ws):
        t = 1 + wp + j
        parent[t] = 0 if j == 0 else (t - 1)
        depth[t] = 1 + j

    anc = [[-1, -1, -1, -1] for _ in range(M)]
    for t in range(M):
        if t < nl:
            cur = t
            for lag in range(4):
                anc[t][lag] = cur
                cur = parent[cur] if cur >= 0 else -1

    pos_list, kv_slot_list, c_slot_list = [], [], []
    for t in range(M):
        if t < nl:
            pos_list.append(past + depth[t])
            kv_slot_list.append(past + t)  # primary at past..past+wp, secondary at past+wp+1..past+wp+ws!
            c_slot_list.append((past + t) % (4 * C_N))
        else:
            pos_list.append(0)
            kv_slot_list.append(-1)
            c_slot_list.append(-1)

    tree_meta = torch.tensor([1 if ws > 0 else 0, wp, ws, nl], dtype=torch.int32, device=device)
    tree_anc = torch.tensor(anc, dtype=torch.int32, device=device)
    pos = torch.tensor(pos_list, dtype=torch.int64, device=device)
    kv_slot = torch.tensor(kv_slot_list, dtype=torch.int64, device=device)
    c_slot = torch.tensor(c_slot_list, dtype=torch.int64, device=device)
    seq_idx = torch.zeros((M,), dtype=torch.int32, device=device)
    qstart = torch.zeros((M,), dtype=torch.int32, device=device)
    cuq = torch.tensor([0, M], dtype=torch.int32, device=device)
    seql = torch.tensor([past + M], dtype=torch.int32, device=device)
    return tree_meta, tree_anc, pos, kv_slot, c_slot, seq_idx, qstart, cuq, seql, nl


# ---------------------------------------------------------------------------
# SUITE A: C=1 Linear (`M <= 8`, Local `S=8` & Global `S=16`, `nl in {1, 4, 8}`)
# ---------------------------------------------------------------------------
def verify_suite_a_c1_linear(lib, tree_mode: int, continue_on_fail: bool, dump_ptx_flag: bool):
    from vllm.models.inkling2.ops import attention as attn_ops
    from vllm.models.inkling2.ops import sconv as sconv_ops
    from vllm.models.inkling2.ops import sm12_attn

    device = torch.device("cuda:0")
    torch.manual_seed(202601)
    H, Hkv, G, D, D_REL, E, BS, C_N = 6, 1, 6, 128, 16, 1024, 64, 16
    clip_thr, rel_thr, norm_eps = 22.6274, 8.0, 1e-6
    scale = D ** -0.5

    blocks_per_req = 40
    kc_init = torch.randn((blocks_per_req, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.25
    vc_init = torch.randn((blocks_per_req, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.25
    cc_init = torch.randn((4, 1, C_N, 16384), dtype=torch.bfloat16, device=device) * 0.25

    rel_proj = torch.randn((D_REL, E), dtype=torch.bfloat16, device=device) * 0.12
    rel_proj_t = rel_proj.t().contiguous()
    rel_proj_rev = rel_proj.flip(-1).contiguous()
    rp_loc = sconv_ops.shift_rel_proj(rel_proj, True)
    rp_ful = sconv_ops.shift_rel_proj(rel_proj, False)
    q_sc = torch.randn((H, D), dtype=torch.bfloat16, device=device) * 0.4 + 1.0
    r_sc = torch.randn((H, D_REL), dtype=torch.bfloat16, device=device) * 0.4 + 1.0
    kw_s = torch.randn((D, 4), dtype=torch.bfloat16, device=device) * 0.25
    vw_s = torch.randn((D, 4), dtype=torch.bfloat16, device=device) * 0.25
    kw_d = torch.randn((4, D), dtype=torch.bfloat16, device=device) * 0.25
    vw_d = torch.randn((4, D), dtype=torch.bfloat16, device=device) * 0.25

    # Verify MGS oracle status explicitly
    mgs_lib = sm12_attn._get_mgs_lib()
    print(
        f"\n=== SUITE A: C=1 LINEAR 0-ULP VERIFICATION "
        f"(tree_mode={tree_mode}, FUSE_GS={sm12_attn._FUSE_GS}, MGS_CUDA={sm12_attn._MGS_CUDA}, "
        f"mgs_lib={'LOADED' if mgs_lib else 'NONE'}, ptxas={os.environ.get('TRITON_PTXAS_BLACKWELL_PATH')}) ==="
    )

    cases = [
        ("C1 M=1 nl=1 seq=1024", 1, 1, 1024),
        ("C1 M=4 nl=4 seq=1024", 4, 4, 1024),
        ("C1 M=8 nl=8 seq=64",   8, 8, 64),
        ("C1 M=8 nl=8 seq=150",  8, 8, 150),
        ("C1 M=8 nl=8 seq=504",  8, 8, 504),
        ("C1 M=8 nl=8 seq=505",  8, 8, 505),
        ("C1 M=8 nl=8 seq=1024", 8, 8, 1024),
        ("C1 M=8 nl=4 seq=1024", 8, 4, 1024),
        ("C1 M=8 nl=1 seq=1024", 8, 1, 1024),
        ("C1 M=8 nl=8 seq=1536", 8, 8, 1536),
        ("C1 M=8 nl=8 seq=2048", 8, 8, 2048),
    ]

    bt_kv = torch.randperm(blocks_per_req, dtype=torch.int32, device=device).view(1, blocks_per_req)
    bt_c = torch.arange(4, dtype=torch.int32, device=device).view(1, 4)

    n_fail = 0
    dumped_ptx = False
    for label, M, nl, seqlen in cases:
        past = seqlen - M
        cuq = torch.tensor([0, M], dtype=torch.int32, device=device)
        seql = torch.tensor([seqlen], dtype=torch.int32, device=device)
        pos = torch.arange(past, past + M, dtype=torch.int64, device=device)
        seq_idx = torch.zeros((M,), dtype=torch.int32, device=device)
        qstart = torch.zeros((M,), dtype=torch.int32, device=device)
        kv_slot = torch.tensor([int(bt_kv[0, (past + j) // BS]) * BS + ((past + j) % BS) for j in range(M)],
                               dtype=torch.int64, device=device)
        c_slot = torch.tensor([(past + j) % (4 * C_N) for j in range(M)], dtype=torch.int64, device=device)
        nl_buf = torch.tensor([nl], dtype=torch.int32, device=device) if nl < M else None

        q_in = torch.randn((M, H, D), dtype=torch.bfloat16, device=device) * 0.3
        r_in = torch.randn((M, H, D_REL), dtype=torch.bfloat16, device=device) * 0.3
        k_in = torch.randn((M, D), dtype=torch.bfloat16, device=device) * 0.3
        v_in = torch.randn((M, D), dtype=torch.bfloat16, device=device) * 0.3
        kd = torch.randn((M, 4), dtype=torch.bfloat16, device=device) * 0.2
        vd = torch.randn((M, 4), dtype=torch.bfloat16, device=device) * 0.2
        tau = torch.randn((M,), dtype=torch.float32, device=device) * 0.1 + 1.0
        gate = torch.randn((M, H * D), dtype=torch.bfloat16, device=device) * 0.3
        sinks = torch.randn((M, H), dtype=torch.bfloat16, device=device) * 0.3

        for is_local in (True, False):
            win = (1023, 0) if is_local else (-1, -1)
            win_i = 1023 if is_local else -1
            dyn = not is_local
            ns = 8 if is_local else 16

            # --- Reference: live attention_prologue + sm12_attn + ns_mgs (+ Triton MGS cross-check) ---
            kc_r, vc_r, cc_r = kc_init.clone(), vc_init.clone(), cc_init.clone()
            qo_r = torch.empty_like(q_in)
            rel_r = torch.empty((M + 128, H, E + 256), dtype=torch.bfloat16, device=device)
            sconv_ops.attention_prologue(
                q_in, r_in, k_in, v_in, q_sc, r_sc, None if is_local else tau,
                rp_loc if is_local else rp_ful,
                kw_s if is_local else kw_d, vw_s if is_local else vw_d,
                None if is_local else kd, None if is_local else vd,
                cc_r, pos, bt_c, seq_idx, c_slot, qstart, kc_r, vc_r, kv_slot,
                qo_r, rel_r, cuq, seql,
                clip_threshold=clip_thr, rel_threshold=rel_thr, norm_eps=norm_eps,
                rel_proj_rev=rel_proj_rev,
            )
            out_r = torch.empty((M, H, D), dtype=torch.float32, device=device)
            lse_r = torch.empty((H, M), dtype=torch.float32, device=device)
            y_ns = torch.empty((M, H * D), dtype=torch.bfloat16, device=device)
            # Note: if nl_buf is set, mirror ROWAR by running with nl_buf via direct _attn_kernel or nl==M
            if nl_buf is None:
                sm12_attn.rel_attention(
                    qo_r, kc_r, vc_r, block_table=bt_kv, cache_seqlens=seql,
                    cu_seqlens_q=cuq, max_seqlen_q=M, softmax_scale=scale,
                    window_size=win, sheared_bias=rel_r,
                    out=out_r, lse=lse_r, num_splits=ns,
                )
                assert sm12_attn._PENDING is not None, "Expected _FUSE_GS=1 deferred _PENDING!"
                tri_op = sm12_attn._PENDING[2].clone()
                tri_lp = sm12_attn._PENDING[3].clone()
                y_tri_mgs = run_triton_mgs_direct(sm12_attn, tri_op, tri_lp, gate, sinks, M, nl_buf=None)
                attn_ops.gate_sink(out_r, gate, lse_r.t(), sinks, y_ns)
            else:
                # Invoke live _attn_kernel directly with HAS_NLIVE=True + ns_mgs
                tri_op = torch.empty((ns, M, H, D), dtype=torch.float32, device=device)
                tri_lp = torch.empty((ns, M, H), dtype=torch.float32, device=device)
                mtiles = (M * G + 15) // 16
                sm12_attn._attn_kernel[(mtiles, 1, Hkv * ns)](
                    qo_r, kc_r, vc_r, bt_kv, seql, cuq, rel_r, tri_op, tri_lp,
                    qo_r.stride(0), qo_r.stride(1),
                    kc_r.stride(0), kc_r.stride(1), kc_r.stride(2),
                    vc_r.stride(0), vc_r.stride(1), vc_r.stride(2),
                    bt_kv.stride(0), rel_r.stride(0), rel_r.stride(1),
                    tri_op.stride(0), tri_op.stride(1), tri_op.stride(2),
                    tri_lp.stride(0), tri_lp.stride(1), tri_lp.stride(2),
                    scale * LOG2E,
                    WINDOW=win_i, E=E, EW=E + 256, G=G, BS=BS, D=D,
                    BLOCK_M=16, BLOCK_N=64, NUM_SPLITS=ns, DRAFT=False,
                    REQ_BOUNDS=(mtiles > 1), PDL=False,
                    NLivePtr=nl_buf, HAS_NLIVE=True,
                    TreeDepthPtr=cuq, TreePathPtr=cuq, TreeMetaPtr=cuq,
                    HAS_TREE=False, HAS_TREE_2B=False, TREE_MAX_M=8,
                    num_warps=4, num_stages=1, launch_pdl=False,
                )
                y_tri_mgs = run_triton_mgs_direct(sm12_attn, tri_op, tri_lp, gate, sinks, M, nl_buf=nl_buf)
                if mgs_lib is not None:
                    rc_m = mgs_lib.merge_gate_sink_exact_fwd(
                        tri_op.data_ptr(), tri_lp.data_ptr(), gate.data_ptr(), sinks.data_ptr(), y_ns.data_ptr(),
                        nl_buf.data_ptr(),
                        tri_op.stride(0), tri_op.stride(1), tri_op.stride(2),
                        tri_lp.stride(0), tri_lp.stride(1), tri_lp.stride(2),
                        gate.stride(0), sinks.stride(0),
                        M, M, H, ns, 1, 0, 1,
                        int(torch.cuda.current_stream(device).cuda_stream),
                    )
                    assert rc_m == 0
                else:
                    y_ns.copy_(y_tri_mgs)

            # Determinism sanity check on M=8 seq=1024 (proves §9.5 was a harness artifact!)
            if label == "C1 M=8 nl=8 seq=1024":
                out_r2 = torch.empty_like(out_r)
                lse_r2 = torch.empty_like(lse_r)
                sm12_attn.rel_attention(
                    qo_r, kc_r, vc_r, block_table=bt_kv, cache_seqlens=seql,
                    cu_seqlens_q=cuq, max_seqlen_q=M, softmax_scale=scale,
                    window_size=win, sheared_bias=rel_r,
                    out=out_r2, lse=lse_r2, num_splits=ns,
                )
                tri_op2 = sm12_attn._PENDING[2].clone()
                tri_lp2 = sm12_attn._PENDING[3].clone()
                sm12_attn._PENDING = None
                det_st = compare_op_lp(tri_op2, tri_lp2, tri_op, tri_lp, nl=nl)
                assert det_st["d_op"] == 0 and det_st["d_lp"] == 0, f"Triton op/lp nondet: {det_st}"

            # --- Test: v2.so (prologue_kvq + fused_decode_attn_otf_mgs_fwd) ---
            kc_g, vc_g, cc_g = kc_init.clone(), vc_init.clone(), cc_init.clone()
            qo_g = torch.empty_like(q_in)
            stream = int(torch.cuda.current_stream(device).cuda_stream)
            sconv_scale = float((4 * (2 if dyn else 1) + 1) ** -0.5)
            kw, vw = (kw_d, vw_d) if dyn else (kw_s, vw_s)
            rc1 = lib.attn_prologue_kvq_exact_fwd(
                k_in.data_ptr(), v_in.data_ptr(), kw.data_ptr(), vw.data_ptr(),
                kd.data_ptr() if dyn else 0, vd.data_ptr() if dyn else 0,
                cc_g.data_ptr(), kc_g.data_ptr(), vc_g.data_ptr(),
                kv_slot.data_ptr(), pos.data_ptr(), seq_idx.data_ptr(),
                c_slot.data_ptr(), bt_c.data_ptr(), qstart.data_ptr(),
                q_in.data_ptr(), q_sc.data_ptr(), tau.data_ptr() if dyn else 0,
                qo_g.data_ptr(), nl_buf.data_ptr() if nl_buf is not None else 0,
                M, k_in.stride(0), v_in.stride(0), kw.stride(0), kw.stride(1),
                kd.stride(0) if dyn else 0, vd.stride(0) if dyn else 0,
                cc_g.stride(0), cc_g.stride(2),
                kc_g.stride(0), kc_g.stride(1), BS,
                bt_c.stride(0), bt_c.shape[1], C_N, q_in.stride(0),
                sconv_scale, clip_thr, 1e-6, norm_eps,
                int(dyn), int(dyn), int(nl_buf is not None), 0, stream,
            )
            assert rc1 == 0, f"v2 prologue rc={rc1}"
            mtiles = (M * G + 15) // 16

            def _run_v2(tm: int):
                op_b = torch.empty((ns, M, H, D), dtype=torch.float32, device=device)
                lp_b = torch.empty((ns, M, H), dtype=torch.float32, device=device)
                sem_b = torch.zeros((mtiles * Hkv,), dtype=torch.int32, device=device)
                y_g = torch.empty((M, H * D), dtype=torch.bfloat16, device=device)
                rc2 = lib.fused_decode_attn_otf_mgs_fwd(
                    qo_g.data_ptr(), r_in.data_ptr(), r_sc.data_ptr(),
                    tau.data_ptr() if dyn else 0, rel_proj_t.data_ptr(),
                    kc_g.data_ptr(), vc_g.data_ptr(), bt_kv.data_ptr(), seql.data_ptr(), cuq.data_ptr(),
                    gate.data_ptr(), sinks.data_ptr(),
                    op_b.data_ptr(), lp_b.data_ptr(), sem_b.data_ptr(), y_g.data_ptr(),
                    nl_buf.data_ptr() if nl_buf is not None else 0,
                    qo_g.stride(0), qo_g.stride(1), r_in.stride(0), r_in.stride(1),
                    kc_g.stride(0), kc_g.stride(1), kc_g.stride(2),
                    vc_g.stride(0), vc_g.stride(1), vc_g.stride(2),
                    bt_kv.stride(0), gate.stride(0), sinks.stride(0), y_g.stride(0),
                    op_b.stride(0), op_b.stride(1), op_b.stride(2),
                    lp_b.stride(0), lp_b.stride(1), lp_b.stride(2),
                    float(scale * LOG2E), rel_thr, 1e-6,
                    M, M, mtiles, 1, Hkv, BS, E, G, win_i, ns,
                    int(dyn), int(nl_buf is not None), 0, 0, int(tm), stream,
                )
                assert rc2 == 0, f"v2 attn rc={rc2}"
                return y_g, op_b, lp_b

            y_g, v2_op, v2_lp = _run_v2(tree_mode)
            dq = bf16_diff_count(qo_r[:nl], qo_g[:nl])
            dk = bf16_diff_count(kc_r, kc_g) if nl == M else 0
            dv = bf16_diff_count(vc_r, vc_g) if nl == M else 0
            dc = bf16_diff_count(cc_r, cc_g) if nl == M else 0
            dy_ns = bf16_diff_count(y_ns[:nl], y_g[:nl])
            dy_tri = bf16_diff_count(y_tri_mgs[:nl], y_g[:nl])
            d_oracle = bf16_diff_count(y_ns[:nl], y_tri_mgs[:nl])
            st = compare_op_lp(v2_op, v2_lp, tri_op, tri_lp, nl=nl)
            kind = "local " if is_local else "global"
            print(
                f"  [{kind}] {label:22s}: dq={dq} dk={dk} dv={dv} dc={dc} | "
                f"lp_u32={st['d_lp']}(ulp={st['ulp_lp']},inf={st['inf_mismatch']}) "
                f"op_u32={st['d_op']}(ulp={st['ulp_op']}) | "
                f"dy_ns={dy_ns} dy_tri={dy_tri} (ns_vs_tri={d_oracle})"
            )
            if dump_ptx_flag and not dumped_ptx:
                dumped_ptx = True
                dump_live_attn_ptx()
            has_f32_diff = (st["d_lp"] > 0 or st["d_op"] > 0 or st["inf_mismatch"] > 0)
            if dq or dk or dv or dc or dy_ns or dy_tri or has_f32_diff:
                if dq or dk or dv or dc or dy_ns or dy_tri:
                    n_fail += 1
                if st["d_lp"] > 0:
                    print(
                        f"    [LP-DIFF] first={st['idx_lp']} got={st['glp']:.8g}(0x{st['glh']:08x}) "
                        f"ref={st['rlp']:.8g}(0x{st['rlh']:08x}) max_ulp={st['ulp_lp']}"
                    )
                if st["d_op"] > 0:
                    print(
                        f"    [OP-DIFF] first={st['idx_op']} got={st['gop']:.8g}(0x{st['goh']:08x}) "
                        f"ref={st['rop']:.8g}(0x{st['roh']:08x}) max_ulp={st['ulp_op']}"
                    )
                if not dumped_ptx:
                    # Zero-bias isolation test (rel_bias == 0 identically on both Triton & v2!)
                    zero_rel = torch.zeros_like(rel_r)
                    zero_rpt = torch.zeros_like(rel_proj_t)
                    z_tri_op = torch.empty((ns, M, H, D), dtype=torch.float32, device=device)
                    z_tri_lp = torch.empty((ns, M, H), dtype=torch.float32, device=device)
                    sm12_attn._attn_kernel[(mtiles, 1, Hkv * ns)](
                        qo_r, kc_r, vc_r, bt_kv, seql, cuq, zero_rel, z_tri_op, z_tri_lp,
                        qo_r.stride(0), qo_r.stride(1),
                        kc_r.stride(0), kc_r.stride(1), kc_r.stride(2),
                        vc_r.stride(0), vc_r.stride(1), vc_r.stride(2),
                        bt_kv.stride(0), zero_rel.stride(0), zero_rel.stride(1),
                        z_tri_op.stride(0), z_tri_op.stride(1), z_tri_op.stride(2),
                        z_tri_lp.stride(0), z_tri_lp.stride(1), z_tri_lp.stride(2),
                        scale * LOG2E,
                        WINDOW=win_i, E=E, EW=E + 256, G=G, BS=BS, D=D,
                        BLOCK_M=16, BLOCK_N=64, NUM_SPLITS=ns, DRAFT=False,
                        REQ_BOUNDS=(mtiles > 1), PDL=False,
                        NLivePtr=nl_buf if nl_buf is not None else cuq, HAS_NLIVE=(nl_buf is not None),
                        TreeDepthPtr=cuq, TreePathPtr=cuq, TreeMetaPtr=cuq,
                        HAS_TREE=False, HAS_TREE_2B=False, TREE_MAX_M=8,
                        num_warps=4, num_stages=1, launch_pdl=False,
                    )
                    z_op_b = torch.empty_like(z_tri_op)
                    z_lp_b = torch.empty_like(z_tri_lp)
                    z_sem = torch.zeros((mtiles * Hkv,), dtype=torch.int32, device=device)
                    z_y = torch.empty((M, H * D), dtype=torch.bfloat16, device=device)
                    lib.fused_decode_attn_otf_mgs_fwd(
                        qo_g.data_ptr(), r_in.data_ptr(), r_sc.data_ptr(),
                        tau.data_ptr() if dyn else 0, zero_rpt.data_ptr(),
                        kc_g.data_ptr(), vc_g.data_ptr(), bt_kv.data_ptr(), seql.data_ptr(), cuq.data_ptr(),
                        gate.data_ptr(), sinks.data_ptr(),
                        z_op_b.data_ptr(), z_lp_b.data_ptr(), z_sem.data_ptr(), z_y.data_ptr(),
                        nl_buf.data_ptr() if nl_buf is not None else 0,
                        qo_g.stride(0), qo_g.stride(1), r_in.stride(0), r_in.stride(1),
                        kc_g.stride(0), kc_g.stride(1), kc_g.stride(2),
                        vc_g.stride(0), vc_g.stride(1), vc_g.stride(2),
                        bt_kv.stride(0), gate.stride(0), sinks.stride(0), z_y.stride(0),
                        z_op_b.stride(0), z_op_b.stride(1), z_op_b.stride(2),
                        z_lp_b.stride(0), z_lp_b.stride(1), z_lp_b.stride(2),
                        float(scale * LOG2E), rel_thr, 1e-6,
                        M, M, mtiles, 1, Hkv, BS, E, G, win_i, ns,
                        int(dyn), int(nl_buf is not None), 0, 0, int(tree_mode), stream,
                    )
                    z_st = compare_op_lp(z_op_b, z_lp_b, z_tri_op, z_tri_lp, nl=nl)
                    print(
                        f"    [ZERO-BIAS ISOLATION ({kind})] lp_u32={z_st['d_lp']}(ulp={z_st['ulp_lp']}) "
                        f"op_u32={z_st['d_op']}(ulp={z_st['ulp_op']})"
                    )
                    run_orthogonal_sweep(_run_v2, y_ns, tri_op, tri_lp, nl=nl)
                    dumped_ptx = True
                    dump_live_attn_ptx()
                if (dq or dk or dv or dc or dy_ns or dy_tri) and not continue_on_fail:
                    raise AssertionError(f"Suite A FAIL on {label} ({kind}): dy_ns={dy_ns} dy_tri={dy_tri}")
    if n_fail > 0:
        print(f"  [SUMMARY] Suite A had {n_fail} failing BF16 case(s)!")
        if not continue_on_fail:
            raise AssertionError(f"Suite A had {n_fail} failing case(s)!")
        return n_fail
    print("  -> SUITE A (C=1 LINEAR) ALL 0-ULP CHECKS PASSED (dy_ns == 0 & dy_tri == 0)!")
    return 0


# ---------------------------------------------------------------------------
# SUITE B: S8-P6v2 Fast 4-Tile BM=16 Beam-2 Tree (`M=8, ws <= 2`, Local & Global)
# ---------------------------------------------------------------------------
def verify_suite_b_tree_2b(lib, tree_mode: int, continue_on_fail: bool):
    from vllm.models.inkling2.ops import sconv as sconv_ops
    from vllm.models.inkling2.ops import sm12_attn

    device = torch.device("cuda:0")
    torch.manual_seed(202602)
    H, Hkv, G, D, D_REL, E, BS, C_N = 6, 1, 6, 128, 16, 1024, 64, 16
    clip_thr, rel_thr, norm_eps = 22.6274, 8.0, 1e-6
    scale = D ** -0.5
    M = 8

    blocks_per_req = 40
    kc_init = torch.randn((blocks_per_req, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.25
    vc_init = torch.randn((blocks_per_req, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.25
    cc_init = torch.randn((4, 1, C_N, 16384), dtype=torch.bfloat16, device=device) * 0.25

    rel_proj = torch.randn((D_REL, E), dtype=torch.bfloat16, device=device) * 0.12
    rel_proj_t = rel_proj.t().contiguous()
    rp_loc = sconv_ops.shift_rel_proj(rel_proj, True)
    rp_ful = sconv_ops.shift_rel_proj(rel_proj, False)
    q_sc = torch.randn((H, D), dtype=torch.bfloat16, device=device) * 0.4 + 1.0
    r_sc = torch.randn((H, D_REL), dtype=torch.bfloat16, device=device) * 0.4 + 1.0
    kw_s = torch.randn((D, 4), dtype=torch.bfloat16, device=device) * 0.25
    vw_s = torch.randn((D, 4), dtype=torch.bfloat16, device=device) * 0.25
    kw_d = torch.randn((4, D), dtype=torch.bfloat16, device=device) * 0.25
    vw_d = torch.randn((4, D), dtype=torch.bfloat16, device=device) * 0.25

    mgs_lib = sm12_attn._get_mgs_lib()
    bt_kv = torch.randperm(blocks_per_req, dtype=torch.int32, device=device).view(1, blocks_per_req)
    bt_c = torch.arange(4, dtype=torch.int32, device=device).view(1, 4)

    print(f"\n=== SUITE B: S8-P6v2 4-TILE BM=16 BEAM-2 TREE 0-ULP VERIFICATION (tree_mode={tree_mode}) ===")
    configs = [
        (0,    5, 2),
        (64,   5, 2),
        (504,  5, 2),
        (505,  5, 2),
        (1024, 7, 0),  # t_act=0 linear fallback!
        (1024, 6, 1),
        (1024, 5, 2),
        (1024, 4, 2),
        (1024, 3, 2),
        (2048, 5, 2),
    ]

    n_fail = 0
    for past, wp, ws in configs:
        tree_meta, tree_anc, pos, kv_slot_log, c_slot, seq_idx, qstart, cuq, seql, nl = make_s8p6v2_tree_metadata(
            past, M, wp, ws, blocks_per_req, BS, C_N, device
        )
        # Map logical KV slots through scrambled bt_kv
        kv_slot_list = []
        for s in kv_slot_log.tolist():
            if s < 0:
                kv_slot_list.append(-1)
            else:
                kv_slot_list.append(int(bt_kv[0, s // BS]) * BS + (s % BS))
        kv_slot = torch.tensor(kv_slot_list, dtype=torch.int64, device=device)
        nl_buf = torch.tensor([nl], dtype=torch.int32, device=device)

        q_in = torch.randn((M, H, D), dtype=torch.bfloat16, device=device) * 0.3
        r_in = torch.randn((M, H, D_REL), dtype=torch.bfloat16, device=device) * 0.3
        k_in = torch.randn((M, D), dtype=torch.bfloat16, device=device) * 0.3
        v_in = torch.randn((M, D), dtype=torch.bfloat16, device=device) * 0.3
        kd = torch.randn((M, 4), dtype=torch.bfloat16, device=device) * 0.2
        vd = torch.randn((M, 4), dtype=torch.bfloat16, device=device) * 0.2
        tau = torch.randn((M,), dtype=torch.float32, device=device) * 0.1 + 1.0
        gate = torch.randn((M, H * D), dtype=torch.bfloat16, device=device) * 0.3
        sinks = torch.randn((M, H), dtype=torch.bfloat16, device=device) * 0.3

        for is_local in (True, False):
            win_i = 1023 if is_local else -1
            dyn = not is_local
            ns = 8 if is_local else 16
            stream = int(torch.cuda.current_stream(device).cuda_stream)
            sconv_scale = float((4 * (2 if dyn else 1) + 1) ** -0.5)
            kw, vw = (kw_d, vw_d) if dyn else (kw_s, vw_s)

            # 1. Run v2 tree prologue -> kc_g, vc_g, cc_g, qo_g
            kc_g, vc_g, cc_g = kc_init.clone(), vc_init.clone(), cc_init.clone()
            qo_g = torch.empty_like(q_in)
            rc1 = lib.attn_prologue_kvq_tree_exact_fwd(
                k_in.data_ptr(), v_in.data_ptr(), kw.data_ptr(), vw.data_ptr(),
                kd.data_ptr() if dyn else 0, vd.data_ptr() if dyn else 0,
                cc_g.data_ptr(), kc_g.data_ptr(), vc_g.data_ptr(),
                kv_slot.data_ptr(), pos.data_ptr(), seq_idx.data_ptr(),
                c_slot.data_ptr(), bt_c.data_ptr(), qstart.data_ptr(),
                q_in.data_ptr(), q_sc.data_ptr(), tau.data_ptr() if dyn else 0,
                qo_g.data_ptr(), nl_buf.data_ptr(), tree_anc.data_ptr(),
                M, k_in.stride(0), v_in.stride(0), kw.stride(0), kw.stride(1),
                kd.stride(0) if dyn else 0, vd.stride(0) if dyn else 0,
                cc_g.stride(0), cc_g.stride(2),
                kc_g.stride(0), kc_g.stride(1), BS,
                bt_c.stride(0), bt_c.shape[1], C_N, q_in.stride(0),
                sconv_scale, clip_thr, 1e-6, norm_eps,
                int(dyn), int(dyn), 1, 0, stream,
            )
            assert rc1 == 0, f"v2 tree prologue rc={rc1}"

            # 2. Build 100% EXACT live-Triton sheared `rel_r` AND verify tree K/V/Q prologue!
            #    - Primary tokens `0..wp`: run live `sconv_ops.attention_prologue` linearly on `0..M-1`
            #    - Secondary tokens `1+wp..wp+ws`: pack `[tok0, sec0, sec1, ...]` into an M=8 linear call
            #      so `sec_j` sits at index `1+j`, position `past+1+j`, with IDENTICAL `off` shear column!
            #      This gives exact live Triton `_l2_clip` + `mma` sheared `rel_r` AND bit-exact K/V/Q oracle!
            pos_lin = torch.arange(past, past + M, dtype=torch.int64, device=device)
            kv_slot_lin = torch.tensor(
                [int(bt_kv[0, (past + j) // BS]) * BS + ((past + j) % BS) for j in range(M)],
                dtype=torch.int64, device=device,
            )
            c_slot_lin = torch.tensor([(past + j) % (4 * C_N) for j in range(M)], dtype=torch.int64, device=device)
            rel_proj_rev = rel_proj.flip(-1).contiguous()

            kc_lin, vc_lin, cc_lin = kc_init.clone(), vc_init.clone(), cc_init.clone()
            qo_lin = torch.empty_like(q_in)
            rel_r = torch.empty((M + 128, H, E + 256), dtype=torch.bfloat16, device=device)
            sconv_ops.attention_prologue(
                q_in, r_in, k_in, v_in, q_sc, r_sc, None if is_local else tau,
                rp_loc if is_local else rp_ful,
                kw_s if is_local else kw_d, vw_s if is_local else vw_d,
                None if is_local else kd, None if is_local else vd,
                cc_lin, pos_lin, bt_c, seq_idx, c_slot_lin, qstart, kc_lin, vc_lin, kv_slot_lin,
                qo_lin, rel_r, cuq, seql,
                clip_threshold=clip_thr, rel_threshold=rel_thr, norm_eps=norm_eps,
                rel_proj_rev=rel_proj_rev,
            )
            dq_p = bf16_diff_count(qo_g[: 1 + wp], qo_lin[: 1 + wp])
            dk_p = 0
            dv_p = 0
            for i in range(1 + wp):
                b_i = int(bt_kv[0, (past + i) // BS])
                s_i = (past + i) % BS
                dk_p += bf16_diff_count(kc_g[b_i, s_i], kc_lin[b_i, s_i])
                dv_p += bf16_diff_count(vc_g[b_i, s_i], vc_lin[b_i, s_i])

            dq_s = dk_s = dv_s = 0
            if ws > 0:
                sec_idx_map = [0] + [1 + wp + j for j in range(ws)] + [0] * (M - 1 - ws)
                idx_t = torch.tensor(sec_idx_map, dtype=torch.int64, device=device)
                kc_s, vc_s, cc_s = kc_init.clone(), vc_init.clone(), cc_init.clone()
                qo_s = torch.empty_like(q_in)
                rel_s = torch.empty_like(rel_r)
                sconv_ops.attention_prologue(
                    q_in[idx_t].contiguous(), r_in[idx_t].contiguous(),
                    k_in[idx_t].contiguous(), v_in[idx_t].contiguous(),
                    q_sc, r_sc, None if is_local else tau[idx_t].contiguous(),
                    rp_loc if is_local else rp_ful,
                    kw_s if is_local else kw_d, vw_s if is_local else vw_d,
                    None if is_local else kd[idx_t].contiguous(),
                    None if is_local else vd[idx_t].contiguous(),
                    cc_s, pos_lin, bt_c, seq_idx, c_slot_lin, qstart, kc_s, vc_s, kv_slot_lin,
                    qo_s, rel_s, cuq, seql,
                    clip_threshold=clip_thr, rel_threshold=rel_thr, norm_eps=norm_eps,
                    rel_proj_rev=rel_proj_rev,
                )
                for j in range(ws):
                    rel_r[1 + wp + j].copy_(rel_s[1 + j])
                    dq_s += bf16_diff_count(qo_g[1 + wp + j], qo_s[1 + j])
                    b_g = int(bt_kv[0, (past + 1 + wp + j) // BS])
                    s_g = (past + 1 + wp + j) % BS
                    b_r = int(bt_kv[0, (past + 1 + j) // BS])
                    s_r = (past + 1 + j) % BS
                    dk_s += bf16_diff_count(kc_g[b_g, s_g], kc_s[b_r, s_r])
                    dv_s += bf16_diff_count(vc_g[b_g, s_g], vc_s[b_r, s_r])

            nl_p_buf = torch.tensor([1 + wp], dtype=torch.int32, device=device)

            # 3. Live Triton 4-tile HAS_TREE_2B + ns_mgs reference (on identical kc_g, vc_g, qo_g!)
            tri_op = torch.empty((ns, M, H, D), dtype=torch.float32, device=device)
            tri_lp = torch.empty((ns, M, H), dtype=torch.float32, device=device)
            sm12_attn._attn_kernel[(4, 1, Hkv * ns)](
                qo_g, kc_g, vc_g, bt_kv, seql, cuq, rel_r, tri_op, tri_lp,
                qo_g.stride(0), qo_g.stride(1),
                kc_g.stride(0), kc_g.stride(1), kc_g.stride(2),
                vc_g.stride(0), vc_g.stride(1), vc_g.stride(2),
                bt_kv.stride(0), rel_r.stride(0), rel_r.stride(1),
                tri_op.stride(0), tri_op.stride(1), tri_op.stride(2),
                tri_lp.stride(0), tri_lp.stride(1), tri_lp.stride(2),
                scale * LOG2E,
                WINDOW=win_i, E=E, EW=E + 256, G=G, BS=BS, D=D,
                BLOCK_M=16, BLOCK_N=64, NUM_SPLITS=ns, DRAFT=False,
                REQ_BOUNDS=True, PDL=False,
                NLivePtr=nl_buf, HAS_NLIVE=True,
                TreeDepthPtr=cuq, TreePathPtr=cuq, TreeMetaPtr=tree_meta,
                HAS_TREE=False, HAS_TREE_2B=True, TREE_MAX_M=8,
                num_warps=4, num_stages=1, launch_pdl=False,
            )
            y_tri_mgs = run_triton_mgs_direct(sm12_attn, tri_op, tri_lp, gate, sinks, M, nl_buf=nl_buf)
            y_ns = torch.empty((M, H * D), dtype=torch.bfloat16, device=device)
            if mgs_lib is not None:
                rc_m = mgs_lib.merge_gate_sink_exact_fwd(
                    tri_op.data_ptr(), tri_lp.data_ptr(), gate.data_ptr(), sinks.data_ptr(), y_ns.data_ptr(),
                    nl_buf.data_ptr(),
                    tri_op.stride(0), tri_op.stride(1), tri_op.stride(2),
                    tri_lp.stride(0), tri_lp.stride(1), tri_lp.stride(2),
                    gate.stride(0), sinks.stride(0),
                    M, M, H, ns, 1, 0, 1, stream,
                )
                assert rc_m == 0
            else:
                y_ns.copy_(y_tri_mgs)

            # 4. Run v2 fused_decode_attn_otf_mgs_tree_fwd
            def _run_v2_tree(tm: int):
                op_b = torch.empty((ns, M, H, D), dtype=torch.float32, device=device)
                lp_b = torch.empty((ns, M, H), dtype=torch.float32, device=device)
                sem_b = torch.zeros((4 * Hkv,), dtype=torch.int32, device=device)
                y_g = torch.empty((M, H * D), dtype=torch.bfloat16, device=device)
                rc2 = lib.fused_decode_attn_otf_mgs_tree_fwd(
                    qo_g.data_ptr(), r_in.data_ptr(), r_sc.data_ptr(),
                    tau.data_ptr() if dyn else 0, rel_proj_t.data_ptr(),
                    kc_g.data_ptr(), vc_g.data_ptr(), bt_kv.data_ptr(), seql.data_ptr(), cuq.data_ptr(),
                    gate.data_ptr(), sinks.data_ptr(),
                    op_b.data_ptr(), lp_b.data_ptr(), sem_b.data_ptr(), y_g.data_ptr(),
                    nl_buf.data_ptr(), tree_meta.data_ptr(),
                    qo_g.stride(0), qo_g.stride(1), r_in.stride(0), r_in.stride(1),
                    kc_g.stride(0), kc_g.stride(1), kc_g.stride(2),
                    vc_g.stride(0), vc_g.stride(1), vc_g.stride(2),
                    bt_kv.stride(0), gate.stride(0), sinks.stride(0), y_g.stride(0),
                    op_b.stride(0), op_b.stride(1), op_b.stride(2),
                    lp_b.stride(0), lp_b.stride(1), lp_b.stride(2),
                    float(scale * LOG2E), rel_thr, 1e-6,
                    M, M, 4, 1, Hkv, BS, E, G, win_i, ns,
                    int(dyn), 1, 0, 0, int(tm), stream,
                )
                assert rc2 == 0, f"v2 tree attn rc={rc2}"
                return y_g, op_b, lp_b

            y_g, v2_op, v2_lp = _run_v2_tree(tree_mode)

            # 5. Also verify primary-chain [0..wp] 0-ULP parity vs linear v2!
            op_l = torch.empty((ns, M, H, D), dtype=torch.float32, device=device)
            lp_l = torch.empty((ns, M, H), dtype=torch.float32, device=device)
            sem_l = torch.zeros((3 * Hkv,), dtype=torch.int32, device=device)
            y_lin = torch.empty((M, H * D), dtype=torch.bfloat16, device=device)
            rc_la = lib.fused_decode_attn_otf_mgs_fwd(
                qo_lin.data_ptr(), r_in.data_ptr(), r_sc.data_ptr(),
                tau.data_ptr() if dyn else 0, rel_proj_t.data_ptr(),
                kc_lin.data_ptr(), vc_lin.data_ptr(), bt_kv.data_ptr(), seql.data_ptr(), cuq.data_ptr(),
                gate.data_ptr(), sinks.data_ptr(),
                op_l.data_ptr(), lp_l.data_ptr(), sem_l.data_ptr(), y_lin.data_ptr(),
                nl_p_buf.data_ptr(),
                qo_lin.stride(0), qo_lin.stride(1), r_in.stride(0), r_in.stride(1),
                kc_lin.stride(0), kc_lin.stride(1), kc_lin.stride(2),
                vc_lin.stride(0), vc_lin.stride(1), vc_lin.stride(2),
                bt_kv.stride(0), gate.stride(0), sinks.stride(0), y_lin.stride(0),
                op_l.stride(0), op_l.stride(1), op_l.stride(2),
                lp_l.stride(0), lp_l.stride(1), lp_l.stride(2),
                float(scale * LOG2E), rel_thr, 1e-6,
                M, M, 3, 1, Hkv, BS, E, G, win_i, ns,
                int(dyn), 1, 0, 0, int(tree_mode), stream,
            )
            assert rc_la == 0
            dy_p_vs_lin = bf16_diff_count(y_g[: 1 + wp], y_lin[: 1 + wp])
            dy_ns = bf16_diff_count(y_ns[:nl], y_g[:nl])
            dy_tri = bf16_diff_count(y_tri_mgs[:nl], y_g[:nl])
            st = compare_op_lp(v2_op, v2_lp, tri_op, tri_lp, nl=nl)
            kind = "local " if is_local else "global"
            d_pro = dq_p + dk_p + dv_p + dq_s + dk_s + dv_s
            print(
                f"  [{kind}] past={past:4d} wp={wp} ws={ws} nl={nl}: "
                f"pro_p(q/k/v)={dq_p}/{dk_p}/{dv_p} pro_s(q/k/v)={dq_s}/{dk_s}/{dv_s} "
                f"dy_p_vs_lin={dy_p_vs_lin} | "
                f"lp_u32={st['d_lp']}(ulp={st['ulp_lp']}) op_u32={st['d_op']}(ulp={st['ulp_op']}) | "
                f"dy_ns={dy_ns} dy_tri={dy_tri}"
            )
            if d_pro or dy_p_vs_lin or dy_ns or dy_tri:
                n_fail += 1
                if n_fail == 1:
                    run_orthogonal_sweep(_run_v2_tree, y_ns, tri_op, tri_lp, nl=nl)
                if not continue_on_fail:
                    raise AssertionError(
                        f"Suite B FAIL past={past} wp={wp} ws={ws} ({kind}): "
                        f"d_pro={d_pro} dy_p_vs_lin={dy_p_vs_lin} dy_ns={dy_ns} dy_tri={dy_tri}"
                    )
    if n_fail > 0:
        print(f"  [SUMMARY] Suite B had {n_fail} failing BF16 case(s)!")
        if not continue_on_fail:
            raise AssertionError(f"Suite B had {n_fail} failing case(s)!")
        return n_fail
    print("  -> SUITE B (S8-P6v2 4-TILE BEAM-2 TREE) ALL 0-ULP CHECKS PASSED (dy == 0)!")
    return 0


# ---------------------------------------------------------------------------
# SUITE C: Batched C=32 (`S=1`, `FUSE_MT=0/1`, `M=32..256`, `seq=150/1024/4096`)
# ---------------------------------------------------------------------------
def verify_suite_c_c32(lib, tree_mode: int, continue_on_fail: bool):
    from vllm.models.inkling2.ops import attention as attn_ops
    from vllm.models.inkling2.ops import sconv as sconv_ops
    from vllm.models.inkling2.ops import sm12_attn

    device = torch.device("cuda:0")
    torch.manual_seed(202603)
    H, Hkv, G, D, D_REL, E, BS, C_N = 6, 1, 6, 128, 16, 1024, 64, 16
    clip_thr, rel_thr, norm_eps = 22.6274, 8.0, 1e-6
    scale = D ** -0.5

    blocks_per_req = 72
    kc_init = torch.randn((32 * blocks_per_req, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.25
    vc_init = torch.randn((32 * blocks_per_req, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.25
    cc_init = torch.randn((32 * 4, 1, C_N, 16384), dtype=torch.bfloat16, device=device) * 0.25

    rel_proj = torch.randn((D_REL, E), dtype=torch.bfloat16, device=device) * 0.12
    rel_proj_t = rel_proj.t().contiguous()
    rp_loc = sconv_ops.shift_rel_proj(rel_proj, True)
    rp_ful = sconv_ops.shift_rel_proj(rel_proj, False)
    q_sc = torch.randn((H, D), dtype=torch.bfloat16, device=device) * 0.4 + 1.0
    r_sc = torch.randn((H, D_REL), dtype=torch.bfloat16, device=device) * 0.4 + 1.0
    kw_s = torch.randn((D, 4), dtype=torch.bfloat16, device=device) * 0.25
    vw_s = torch.randn((D, 4), dtype=torch.bfloat16, device=device) * 0.25
    kw_d = torch.randn((4, D), dtype=torch.bfloat16, device=device) * 0.25
    vw_d = torch.randn((4, D), dtype=torch.bfloat16, device=device) * 0.25

    cases = [
        ("C32 B=32 M=32 K=0 seq=1024",          [1] * 32, [1024] * 32, 0, 1, 1, 1),
        ("C32 B=32 M=64 K=1 seq=1024",          [2] * 32, [1024] * 32, 0, 1, 1, 1),
        ("C32 B=32 M=128 K=3 seq=1024",         [4] * 32, [1024] * 32, 0, 1, 1, 1),
        ("C32 B=32 M=160 K=4 seq=1024",         [5] * 32, [1024] * 32, 0, 1, 1, 1),
        ("C32 B=32 M=224 K=6 seq=1024",         [7] * 32, [1024] * 32, 0, 1, 1, 1),
        ("C32 B=32 M=256 K=7 seq=150",          [8] * 32, [150] * 32,  0, 1, 1, 1),
        ("C32 B=32 M=256 K=7 seq=1024 FMT=0",   [8] * 32, [1024] * 32, 0, 0, 0, 1),
        ("C32 B=32 M=256 K=7 seq=4096 FMT=1",   [8] * 32, [4096] * 32, 0, 1, 1, 1),
        ("C32 B=32 hetero qlen 1..8",           [(i % 8) + 1 for i in range(32)], [512 + i * 64 for i in range(32)], 0, 1, 1, 1),
        ("Tail B=11 pad->16 M=128 (NT<T)",      [8] * 11, [1200] * 11, 40, 1, 1, 1),
        ("Tail B=11+5 zero-seqlen reqs",        [8] * 16, [1200] * 11 + [0] * 5, 0, 1, 1, 1),
    ]

    print(f"\n=== SUITE C: BATCHED C=32 0-ULP VERIFICATION (tree_mode={tree_mode}) ===")
    n_fail = 0
    for label, q_lens, seq_lens, pad_tail, fuse_mt, evict_first, ns in cases:
        B = len(q_lens)
        NT = sum(q_lens)
        T = NT + pad_tail
        max_q = max(q_lens)
        cuq = torch.tensor([0] + [sum(q_lens[: i + 1]) for i in range(B)], dtype=torch.int32, device=device)
        seql = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        bt_kv = torch.arange(B * blocks_per_req, dtype=torch.int32, device=device).view(B, blocks_per_req)
        bt_c = torch.arange(B * 4, dtype=torch.int32, device=device).view(B, 4)

        pos_list, seq_idx_list, qs_list, kv_slot_list, c_slot_list = [], [], [], [], []
        off_t = 0
        for r_i, (ql, sl) in enumerate(zip(q_lens, seq_lens)):
            past = max(sl - ql, 0)
            for j in range(ql):
                p = past + j
                pos_list.append(p)
                seq_idx_list.append(r_i)
                qs_list.append(off_t)
                if sl > 0:
                    kv_slot_list.append(r_i * blocks_per_req * BS + p)
                    c_slot_list.append((r_i * 4 * C_N) + (p % (4 * C_N)))
                else:
                    kv_slot_list.append(-1)
                    c_slot_list.append(-1)
            off_t += ql
        for _ in range(pad_tail):
            pos_list.append(0); seq_idx_list.append(0); qs_list.append(0)
            kv_slot_list.append(-1); c_slot_list.append(-1)

        pos = torch.tensor(pos_list, dtype=torch.int64, device=device)
        seq_idx = torch.tensor(seq_idx_list, dtype=torch.int32, device=device)
        qstart = torch.tensor(qs_list, dtype=torch.int32, device=device)
        kv_slot = torch.tensor(kv_slot_list, dtype=torch.int64, device=device)
        c_slot = torch.tensor(c_slot_list, dtype=torch.int64, device=device)

        q_in = torch.randn((T, H, D), dtype=torch.bfloat16, device=device) * 0.3
        r_in = torch.randn((T, H, D_REL), dtype=torch.bfloat16, device=device) * 0.3
        k_in = torch.randn((T, D), dtype=torch.bfloat16, device=device) * 0.3
        v_in = torch.randn((T, D), dtype=torch.bfloat16, device=device) * 0.3
        kd = torch.randn((T, 4), dtype=torch.bfloat16, device=device) * 0.2
        vd = torch.randn((T, 4), dtype=torch.bfloat16, device=device) * 0.2
        tau = torch.randn((T,), dtype=torch.float32, device=device) * 0.1 + 1.0
        gate = torch.randn((T, H * D), dtype=torch.bfloat16, device=device) * 0.3
        sinks = torch.randn((T, H), dtype=torch.bfloat16, device=device) * 0.3

        for is_local in (True, False):
            win = (1023, 0) if is_local else (-1, -1)
            win_i = 1023 if is_local else -1
            dyn = not is_local

            kc_r, vc_r, cc_r = kc_init.clone(), vc_init.clone(), cc_init.clone()
            qo_r = torch.empty_like(q_in)
            rel_r = torch.empty((T + 128, H, E + 256), dtype=torch.bfloat16, device=device)
            sconv_ops.attention_prologue(
                q_in, r_in, k_in, v_in, q_sc, r_sc, None if is_local else tau,
                rp_loc if is_local else rp_ful,
                kw_s if is_local else kw_d, vw_s if is_local else vw_d,
                None if is_local else kd, None if is_local else vd,
                cc_r, pos, bt_c, seq_idx, c_slot, qstart, kc_r, vc_r, kv_slot,
                qo_r, rel_r, cuq, seql,
                clip_threshold=clip_thr, rel_threshold=rel_thr, norm_eps=norm_eps,
            )
            out_r = torch.empty((T, H, D), dtype=torch.float32, device=device)
            lse_r = torch.empty((H, T), dtype=torch.float32, device=device)
            y_r = torch.empty((T, H * D), dtype=torch.bfloat16, device=device)
            sm12_attn.rel_attention(
                qo_r[:NT], kc_r, vc_r, block_table=bt_kv, cache_seqlens=seql,
                cu_seqlens_q=cuq, max_seqlen_q=max_q, softmax_scale=scale,
                window_size=win, sheared_bias=rel_r[: NT + 128],
                out=out_r[:NT], lse=lse_r[:, :NT], num_splits=ns,
            )
            if NT < T:
                out_r[NT:].zero_()
                lse_r[:, NT:].zero_()
            tri_op = out_r[:NT].unsqueeze(0).clone()
            tri_lp = lse_r[:, :NT].t().unsqueeze(0).clone()
            attn_ops.gate_sink(out_r, gate, lse_r.t(), sinks, y_r)

            kc_g, vc_g, cc_g = kc_init.clone(), vc_init.clone(), cc_init.clone()
            qo_g = torch.empty_like(q_in)
            stream = int(torch.cuda.current_stream(device).cuda_stream)
            sconv_scale = float((4 * (2 if dyn else 1) + 1) ** -0.5)
            kw, vw = (kw_d, vw_d) if dyn else (kw_s, vw_s)
            rc1 = lib.attn_prologue_kvq_exact_fwd(
                k_in.data_ptr(), v_in.data_ptr(), kw.data_ptr(), vw.data_ptr(),
                kd.data_ptr() if dyn else 0, vd.data_ptr() if dyn else 0,
                cc_g.data_ptr(), kc_g.data_ptr(), vc_g.data_ptr(),
                kv_slot.data_ptr(), pos.data_ptr(), seq_idx.data_ptr(),
                c_slot.data_ptr(), bt_c.data_ptr(), qstart.data_ptr(),
                q_in.data_ptr(), q_sc.data_ptr(), tau.data_ptr() if dyn else 0,
                qo_g.data_ptr(), 0,
                T, k_in.stride(0), v_in.stride(0), kw.stride(0), kw.stride(1),
                kd.stride(0) if dyn else 0, vd.stride(0) if dyn else 0,
                cc_g.stride(0), cc_g.stride(2),
                kc_g.stride(0), kc_g.stride(1), BS,
                bt_c.stride(0), bt_c.shape[1], C_N, q_in.stride(0),
                sconv_scale, clip_thr, 1e-6, norm_eps,
                int(dyn), int(dyn), 0, 0, stream,
            )
            assert rc1 == 0
            mtiles = (max_q * G + 15) // 16

            def _run_c32(tm: int):
                want_dump = (tm != 0) or (ns > 1)
                op_b = torch.empty((ns, T, H, D), dtype=torch.float32, device=device) if want_dump else None
                lp_b = torch.empty((ns, T, H), dtype=torch.float32, device=device) if want_dump else None
                sem_b = torch.zeros((B * mtiles * Hkv,), dtype=torch.int32, device=device) if ns > 1 else None
                y_g = torch.empty((T, H * D), dtype=torch.bfloat16, device=device)
                rc2 = lib.fused_decode_attn_otf_mgs_c32_fwd(
                    qo_g.data_ptr(), r_in.data_ptr(), r_sc.data_ptr(),
                    tau.data_ptr() if dyn else 0, rel_proj_t.data_ptr(),
                    kc_g.data_ptr(), vc_g.data_ptr(), bt_kv.data_ptr(), seql.data_ptr(), cuq.data_ptr(),
                    gate.data_ptr(), sinks.data_ptr(),
                    op_b.data_ptr() if op_b is not None else 0,
                    lp_b.data_ptr() if lp_b is not None else 0,
                    sem_b.data_ptr() if sem_b is not None else 0,
                    y_g.data_ptr(), 0,
                    qo_g.stride(0), qo_g.stride(1), r_in.stride(0), r_in.stride(1),
                    kc_g.stride(0), kc_g.stride(1), kc_g.stride(2),
                    vc_g.stride(0), vc_g.stride(1), vc_g.stride(2),
                    bt_kv.stride(0), gate.stride(0), sinks.stride(0), y_g.stride(0),
                    op_b.stride(0) if op_b is not None else 0,
                    op_b.stride(1) if op_b is not None else 0,
                    op_b.stride(2) if op_b is not None else 0,
                    lp_b.stride(0) if lp_b is not None else 0,
                    lp_b.stride(1) if lp_b is not None else 0,
                    lp_b.stride(2) if lp_b is not None else 0,
                    float(scale * LOG2E), rel_thr, 1e-6,
                    T, NT, mtiles, B, Hkv, BS, E, G, win_i, ns,
                    int(dyn), 0, 0, 0, int(tm), fuse_mt, evict_first, stream,
                )
                assert rc2 == 0, f"c32 attn rc={rc2} (label={label}, mt={mtiles}, fuse_mt={fuse_mt}, win={win_i})"
                if NT < T:
                    y_g[NT:].zero_()
                return y_g, (op_b[:, :NT] if op_b is not None else None), (lp_b[:, :NT] if lp_b is not None else None)

            y_g, _, _ = _run_c32(tree_mode)
            dq = bf16_diff_count(qo_r, qo_g)
            dk = bf16_diff_count(kc_r, kc_g)
            dv = bf16_diff_count(vc_r, vc_g)
            dc = bf16_diff_count(cc_r, cc_g)
            dy = bf16_diff_count(y_r, y_g)
            kind = "local " if is_local else "global"
            print(f"  [{kind}] {label:34s}: dq={dq} dk={dk} dv={dv} dc={dc} dy={dy}")
            if dq or dk or dv or dc or dy:
                n_fail += 1
                if dy > 0 and n_fail <= 2:
                    # Dump FP32 OP/LP via 0x200 diag bit + orthogonal sweep
                    def_sm = 14 if is_local else 0
                    _, d_op, d_lp = _run_c32(encode_diag_mode(def_sm, 0, 0, 0, force_diag=True))
                    st = compare_op_lp(d_op, d_lp, tri_op, tri_lp)
                    if st:
                        print(f"    [STAGE-DIAG] lp_u32={st['d_lp']}(ulp={st['ulp_lp']}) op_u32={st['d_op']}(ulp={st['ulp_op']})")
                    run_orthogonal_sweep(_run_c32, y_r[:NT], tri_op, tri_lp)
                if not continue_on_fail:
                    raise AssertionError(f"Suite C FAIL {label} ({kind}): dy={dy}")
    if n_fail > 0:
        print(f"  [SUMMARY] Suite C had {n_fail} failing BF16 case(s)!")
        if not continue_on_fail:
            raise AssertionError(f"Suite C had {n_fail} failing case(s)!")
        return n_fail
    print("  -> SUITE C (BATCHED C=32) ALL 0-ULP CHECKS PASSED (dy == 0)!")
    return 0


# ---------------------------------------------------------------------------
# SUITE D: Cold-L2 48-Layer CUDA-Graph Timing Benchmark (C=1 Linear + Tree + C=32)
# ---------------------------------------------------------------------------
def bench_graph_cold_us(capture_fn, warmup: int = 3, iters: int = 15) -> float:
    """Capture `capture_fn` in a CUDA graph and measure cold-L2 median time in microseconds."""
    for _ in range(warmup):
        capture_fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        capture_fn()
    torch.cuda.synchronize()
    l2 = torch.empty((16 * 1024 * 1024,), dtype=torch.int32, device="cuda")  # 64 MB > 24 MB GB10 L2
    ts = []
    for _ in range(iters):
        l2.zero_()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def bench_48l_cold_l2(lib, tree_mode: int):
    from vllm.models.inkling2.ops import attention as attn_ops
    from vllm.models.inkling2.ops import sconv as sconv_ops
    from vllm.models.inkling2.ops import sm12_attn

    device = torch.device("cuda:0")
    H, Hkv, G, D, D_REL, E, BS, C_N = 6, 1, 6, 128, 16, 1024, 64, 16
    clip_thr, rel_thr, norm_eps = 22.6274, 8.0, 1e-6
    scale = D ** -0.5

    # Allocate 48 distinct layer KV/conv/weight pools so L2 behavior is 100% realistic (no artificial reuse!)
    n_layers = 48
    n_blks = 40
    print("\n=== SUITE D: 48-LAYER COLD-L2 CUDA-GRAPH BENCHMARK (40 local + 8 global) ===")

    # 1. C=1 Linear (`M=8`, `nl in {1, 4, 8}`, `seq in {256, 512, 1024, 2048}`)
    M = 8
    kc = torch.randn((n_layers, n_blks, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.2
    vc = torch.randn((n_layers, n_blks, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.2
    cc = torch.randn((n_layers, 4, 1, C_N, 256), dtype=torch.bfloat16, device=device) * 0.2
    rel_proj = torch.randn((n_layers, D_REL, E), dtype=torch.bfloat16, device=device) * 0.1
    rel_proj_t = rel_proj.transpose(1, 2).contiguous()
    rel_proj_rev = rel_proj.flip(-1).contiguous()
    rp_loc = [sconv_ops.shift_rel_proj(rel_proj[l], True) for l in range(n_layers)]
    rp_ful = [sconv_ops.shift_rel_proj(rel_proj[l], False) for l in range(n_layers)]
    q_sc = torch.randn((n_layers, H, D), dtype=torch.bfloat16, device=device) * 0.3 + 1.0
    r_sc = torch.randn((n_layers, H, D_REL), dtype=torch.bfloat16, device=device) * 0.3 + 1.0
    kw_s = torch.randn((n_layers, D, 4), dtype=torch.bfloat16, device=device) * 0.2
    vw_s = torch.randn((n_layers, D, 4), dtype=torch.bfloat16, device=device) * 0.2
    kw_d = torch.randn((n_layers, 4, D), dtype=torch.bfloat16, device=device) * 0.2
    vw_d = torch.randn((n_layers, 4, D), dtype=torch.bfloat16, device=device) * 0.2

    q_in = torch.randn((M, H, D), dtype=torch.bfloat16, device=device) * 0.3
    r_in = torch.randn((M, H, D_REL), dtype=torch.bfloat16, device=device) * 0.3
    k_in = torch.randn((M, D), dtype=torch.bfloat16, device=device) * 0.3
    v_in = torch.randn((M, D), dtype=torch.bfloat16, device=device) * 0.3
    kd = torch.randn((M, 4), dtype=torch.bfloat16, device=device) * 0.2
    vd = torch.randn((M, 4), dtype=torch.bfloat16, device=device) * 0.2
    tau = torch.randn((M,), dtype=torch.float32, device=device) * 0.1 + 1.0
    gate = torch.randn((M, H * D), dtype=torch.bfloat16, device=device) * 0.3
    sinks = torch.randn((M, H), dtype=torch.bfloat16, device=device) * 0.3
    qo = torch.empty_like(q_in)
    rel_buf = torch.empty((M + 128, H, E + 256), dtype=torch.bfloat16, device=device)
    out_f = torch.empty((M, H, D), dtype=torch.float32, device=device)
    lse_f = torch.empty((H, M), dtype=torch.float32, device=device)
    y_out = torch.empty((M, H * D), dtype=torch.bfloat16, device=device)
    op_pp = [torch.empty((16, M, H, D), dtype=torch.float32, device=device) for _ in range(2)]
    lp_pp = [torch.empty((16, M, H), dtype=torch.float32, device=device) for _ in range(2)]
    sem_pp = [torch.zeros((8,), dtype=torch.int32, device=device) for _ in range(2)]

    bt_kv = torch.arange(n_blks, dtype=torch.int32, device=device).view(1, n_blks)
    bt_c = torch.arange(4, dtype=torch.int32, device=device).view(1, 4)
    cuq = torch.tensor([0, M], dtype=torch.int32, device=device)
    seq_idx = torch.zeros((M,), dtype=torch.int32, device=device)
    qstart = torch.zeros((M,), dtype=torch.int32, device=device)

    print("  [C=1 Linear 48L (M=8, PDL=1)]")
    for seqlen in (256, 512, 1024, 2048):
        past = seqlen - M
        seql = torch.tensor([seqlen], dtype=torch.int32, device=device)
        pos = torch.arange(past, past + M, dtype=torch.int64, device=device)
        kv_slot = torch.arange(past, past + M, dtype=torch.int64, device=device)
        c_slot = torch.tensor([(past + j) % (4 * C_N) for j in range(M)], dtype=torch.int64, device=device)

        def _base_48l():
            for l in range(n_layers):
                is_loc = (l % 6) != 5
                sconv_ops.attention_prologue(
                    q_in, r_in, k_in, v_in, q_sc[l], r_sc[l], None if is_loc else tau,
                    rp_loc[l] if is_loc else rp_ful[l],
                    kw_s[l] if is_loc else kw_d[l], vw_s[l] if is_loc else vw_d[l],
                    None if is_loc else kd, None if is_loc else vd,
                    cc[l], pos, bt_c, seq_idx, c_slot, qstart, kc[l], vc[l], kv_slot,
                    qo, rel_buf, cuq, seql,
                    clip_threshold=clip_thr, rel_threshold=rel_thr, norm_eps=norm_eps,
                    rel_proj_rev=rel_proj_rev[l],
                )
                sm12_attn.rel_attention(
                    qo, kc[l], vc[l], block_table=bt_kv, cache_seqlens=seql,
                    cu_seqlens_q=cuq, max_seqlen_q=M, softmax_scale=scale,
                    window_size=(1023, 0) if is_loc else (-1, -1),
                    sheared_bias=rel_buf, out=out_f, lse=lse_f,
                    num_splits=8 if is_loc else 16,
                )
                attn_ops.gate_sink(out_f, gate, lse_f.t(), sinks, y_out)

        def _v2_48l():
            st = int(torch.cuda.current_stream(device).cuda_stream)
            for l in range(n_layers):
                is_loc = (l % 6) != 5
                dyn = not is_loc
                ns = 8 if is_loc else 16
                sconv_scale = float((4 * (2 if dyn else 1) + 1) ** -0.5)
                kw, vw = (kw_d[l], vw_d[l]) if dyn else (kw_s[l], vw_s[l])
                lib.attn_prologue_kvq_exact_fwd(
                    k_in.data_ptr(), v_in.data_ptr(), kw.data_ptr(), vw.data_ptr(),
                    kd.data_ptr() if dyn else 0, vd.data_ptr() if dyn else 0,
                    cc[l].data_ptr(), kc[l].data_ptr(), vc[l].data_ptr(),
                    kv_slot.data_ptr(), pos.data_ptr(), seq_idx.data_ptr(),
                    c_slot.data_ptr(), bt_c.data_ptr(), qstart.data_ptr(),
                    q_in.data_ptr(), q_sc[l].data_ptr(), tau.data_ptr() if dyn else 0,
                    qo.data_ptr(), 0,
                    M, k_in.stride(0), v_in.stride(0), kw.stride(0), kw.stride(1),
                    kd.stride(0) if dyn else 0, vd.stride(0) if dyn else 0,
                    cc[l].stride(0), cc[l].stride(2),
                    kc[l].stride(0), kc[l].stride(1), BS,
                    bt_c.stride(0), bt_c.shape[1], C_N, q_in.stride(0),
                    sconv_scale, clip_thr, 1e-6, norm_eps,
                    int(dyn), int(dyn), 0, 1, st,
                )
                pp = l & 1
                lib.fused_decode_attn_otf_mgs_fwd(
                    qo.data_ptr(), r_in.data_ptr(), r_sc[l].data_ptr(),
                    tau.data_ptr() if dyn else 0, rel_proj_t[l].data_ptr(),
                    kc[l].data_ptr(), vc[l].data_ptr(), bt_kv.data_ptr(), seql.data_ptr(), cuq.data_ptr(),
                    gate.data_ptr(), sinks.data_ptr(),
                    op_pp[pp].data_ptr(), lp_pp[pp].data_ptr(), sem_pp[pp].data_ptr(), y_out.data_ptr(), 0,
                    qo.stride(0), qo.stride(1), r_in.stride(0), r_in.stride(1),
                    kc[l].stride(0), kc[l].stride(1), kc[l].stride(2),
                    vc[l].stride(0), vc[l].stride(1), vc[l].stride(2),
                    bt_kv.stride(0), gate.stride(0), sinks.stride(0), y_out.stride(0),
                    op_pp[pp].stride(0), op_pp[pp].stride(1), op_pp[pp].stride(2),
                    lp_pp[pp].stride(0), lp_pp[pp].stride(1), lp_pp[pp].stride(2),
                    float(scale * LOG2E), rel_thr, 1e-6,
                    M, M, 3, 1, Hkv, BS, E, G, 1023 if is_loc else -1, ns,
                    int(dyn), 0, 1, 1, int(tree_mode), st,
                )

        t_base = bench_graph_cold_us(_base_48l)
        t_v2 = bench_graph_cold_us(_v2_48l)
        print(
            f"    seq={seqlen:4d}: live_best={t_base:7.1f} us ({t_base/48:5.2f} us/L) | "
            f"v2_fused={t_v2:7.1f} us ({t_v2/48:5.2f} us/L) | "
            f"delta={t_v2 - t_base:+7.1f} us ({t_base / t_v2:.2f}x)"
        )

    # 2. S8-P6v2 4-Tile BM=16 Beam-2 Tree (`M=8, wp=5, ws=2`)
    print("  [S8-P6v2 4-Tile Beam-2 Tree 48L (M=8, wp=5, ws=2, PDL=1)]")
    for seqlen in (256, 512, 1024, 2048):
        past = seqlen - M
        tree_meta, tree_anc, pos_t, kv_slot_t, c_slot_t, seq_idx_t, qstart_t, cuq_t, seql_t, nl_t = (
            make_s8p6v2_tree_metadata(past, M, 5, 2, n_blks, BS, C_N, device)
        )
        nl_buf = torch.tensor([nl_t], dtype=torch.int32, device=device)
        tri_op = torch.empty((16, M, H, D), dtype=torch.float32, device=device)
        tri_lp = torch.empty((16, M, H), dtype=torch.float32, device=device)
        mgs_lib = sm12_attn._get_mgs_lib()

        def _base_tree_48l():
            st = int(torch.cuda.current_stream(device).cuda_stream)
            for l in range(n_layers):
                is_loc = (l % 6) != 5
                ns = 8 if is_loc else 16
                sconv_ops.attention_prologue(
                    q_in, r_in, k_in, v_in, q_sc[l], r_sc[l], None if is_loc else tau,
                    rp_loc[l] if is_loc else rp_ful[l],
                    kw_s[l] if is_loc else kw_d[l], vw_s[l] if is_loc else vw_d[l],
                    None if is_loc else kd, None if is_loc else vd,
                    cc[l], pos_t, bt_c, seq_idx_t, c_slot_t, qstart_t, kc[l], vc[l], kv_slot_t,
                    qo, rel_buf, cuq_t, seql_t,
                    clip_threshold=clip_thr, rel_threshold=rel_thr, norm_eps=norm_eps,
                    rel_proj_rev=rel_proj_rev[l],
                )
                sm12_attn._attn_kernel[(4, 1, Hkv * ns)](
                    qo, kc[l], vc[l], bt_kv, seql_t, cuq_t, rel_buf, tri_op[:ns], tri_lp[:ns],
                    qo.stride(0), qo.stride(1),
                    kc[l].stride(0), kc[l].stride(1), kc[l].stride(2),
                    vc[l].stride(0), vc[l].stride(1), vc[l].stride(2),
                    bt_kv.stride(0), rel_buf.stride(0), rel_buf.stride(1),
                    tri_op.stride(0), tri_op.stride(1), tri_op.stride(2),
                    tri_lp.stride(0), tri_lp.stride(1), tri_lp.stride(2),
                    scale * LOG2E,
                    WINDOW=1023 if is_loc else -1, E=E, EW=E + 256, G=G, BS=BS, D=D,
                    BLOCK_M=16, BLOCK_N=64, NUM_SPLITS=ns, DRAFT=False,
                    REQ_BOUNDS=True, PDL=True,
                    NLivePtr=nl_buf, HAS_NLIVE=True,
                    TreeDepthPtr=cuq_t, TreePathPtr=cuq_t, TreeMetaPtr=tree_meta,
                    HAS_TREE=False, HAS_TREE_2B=True, TREE_MAX_M=8,
                    num_warps=4, num_stages=1, launch_pdl=True,
                )
                if mgs_lib is not None:
                    mgs_lib.merge_gate_sink_exact_fwd(
                        tri_op.data_ptr(), tri_lp.data_ptr(), gate.data_ptr(), sinks.data_ptr(), y_out.data_ptr(),
                        nl_buf.data_ptr(),
                        tri_op.stride(0), tri_op.stride(1), tri_op.stride(2),
                        tri_lp.stride(0), tri_lp.stride(1), tri_lp.stride(2),
                        gate.stride(0), sinks.stride(0),
                        M, M, H, ns, 1, 1, 1, st,
                    )

        def _v2_tree_48l():
            st = int(torch.cuda.current_stream(device).cuda_stream)
            for l in range(n_layers):
                is_loc = (l % 6) != 5
                dyn = not is_loc
                ns = 8 if is_loc else 16
                sconv_scale = float((4 * (2 if dyn else 1) + 1) ** -0.5)
                kw, vw = (kw_d[l], vw_d[l]) if dyn else (kw_s[l], vw_s[l])
                lib.attn_prologue_kvq_tree_exact_fwd(
                    k_in.data_ptr(), v_in.data_ptr(), kw.data_ptr(), vw.data_ptr(),
                    kd.data_ptr() if dyn else 0, vd.data_ptr() if dyn else 0,
                    cc[l].data_ptr(), kc[l].data_ptr(), vc[l].data_ptr(),
                    kv_slot_t.data_ptr(), pos_t.data_ptr(), seq_idx_t.data_ptr(),
                    c_slot_t.data_ptr(), bt_c.data_ptr(), qstart_t.data_ptr(),
                    q_in.data_ptr(), q_sc[l].data_ptr(), tau.data_ptr() if dyn else 0,
                    qo.data_ptr(), nl_buf.data_ptr(), tree_anc.data_ptr(),
                    M, k_in.stride(0), v_in.stride(0), kw.stride(0), kw.stride(1),
                    kd.stride(0) if dyn else 0, vd.stride(0) if dyn else 0,
                    cc[l].stride(0), cc[l].stride(2),
                    kc[l].stride(0), kc[l].stride(1), BS,
                    bt_c.stride(0), bt_c.shape[1], C_N, q_in.stride(0),
                    sconv_scale, clip_thr, 1e-6, norm_eps,
                    int(dyn), int(dyn), 1, 1, st,
                )
                pp = l & 1
                lib.fused_decode_attn_otf_mgs_tree_fwd(
                    qo.data_ptr(), r_in.data_ptr(), r_sc[l].data_ptr(),
                    tau.data_ptr() if dyn else 0, rel_proj_t[l].data_ptr(),
                    kc[l].data_ptr(), vc[l].data_ptr(), bt_kv.data_ptr(), seql_t.data_ptr(), cuq_t.data_ptr(),
                    gate.data_ptr(), sinks.data_ptr(),
                    op_pp[pp].data_ptr(), lp_pp[pp].data_ptr(), sem_pp[pp].data_ptr(), y_out.data_ptr(),
                    nl_buf.data_ptr(), tree_meta.data_ptr(),
                    qo.stride(0), qo.stride(1), r_in.stride(0), r_in.stride(1),
                    kc[l].stride(0), kc[l].stride(1), kc[l].stride(2),
                    vc[l].stride(0), vc[l].stride(1), vc[l].stride(2),
                    bt_kv.stride(0), gate.stride(0), sinks.stride(0), y_out.stride(0),
                    op_pp[pp].stride(0), op_pp[pp].stride(1), op_pp[pp].stride(2),
                    lp_pp[pp].stride(0), lp_pp[pp].stride(1), lp_pp[pp].stride(2),
                    float(scale * LOG2E), rel_thr, 1e-6,
                    M, M, 4, 1, Hkv, BS, E, G, 1023 if is_loc else -1, ns,
                    int(dyn), 1, 1, 1, int(tree_mode), st,
                )

        t_base = bench_graph_cold_us(_base_tree_48l)
        t_v2 = bench_graph_cold_us(_v2_tree_48l)
        print(
            f"    seq={seqlen:4d}: triton_tree={t_base:7.1f} us ({t_base/48:5.2f} us/L) | "
            f"v2_tree={t_v2:7.1f} us ({t_v2/48:5.2f} us/L) | "
            f"delta={t_v2 - t_base:+7.1f} us ({t_base / t_v2:.2f}x)"
        )

    # 3. Batched C=32 (`B=32`, `M in {128, 256}`, `seq=1024`, 48L cold-L2)
    print("  [Batched C=32 48L (B=32, seq=1024, S=1, FUSE_MT=1 + L2::evict_first)]")
    del kc, vc, cc
    torch.cuda.empty_cache()
    B_c32 = 32
    blks_per_req_c32 = 18
    kc_c32 = torch.randn((n_layers, B_c32 * blks_per_req_c32, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.2
    vc_c32 = torch.randn((n_layers, B_c32 * blks_per_req_c32, BS, Hkv, D), dtype=torch.bfloat16, device=device) * 0.2
    cc_c32 = torch.randn((n_layers, B_c32 * 4, 1, C_N, 256), dtype=torch.bfloat16, device=device) * 0.2
    bt_kv_c32 = torch.arange(B_c32 * blks_per_req_c32, dtype=torch.int32, device=device).view(B_c32, blks_per_req_c32)
    bt_c_c32 = torch.arange(B_c32 * 4, dtype=torch.int32, device=device).view(B_c32, 4)
    seql_c32 = torch.full((B_c32,), 1024, dtype=torch.int32, device=device)

    for qlen_c32 in (4, 8):
        T_c32 = B_c32 * qlen_c32
        mt_c32 = (qlen_c32 * G + 15) // 16
        cuq_c32 = torch.arange(0, T_c32 + 1, qlen_c32, dtype=torch.int32, device=device)
        pos_c32 = torch.tensor([1024 - qlen_c32 + j for _ in range(B_c32) for j in range(qlen_c32)], dtype=torch.int64, device=device)
        seq_idx_c32 = torch.tensor([r for r in range(B_c32) for _ in range(qlen_c32)], dtype=torch.int32, device=device)
        qs_c32 = torch.tensor([r * qlen_c32 for r in range(B_c32) for _ in range(qlen_c32)], dtype=torch.int32, device=device)
        kv_slot_c32 = torch.tensor(
            [r * blks_per_req_c32 * BS + (1024 - qlen_c32 + j) for r in range(B_c32) for j in range(qlen_c32)],
            dtype=torch.int64, device=device,
        )
        c_slot_c32 = torch.tensor(
            [r * 4 * C_N + ((1024 - qlen_c32 + j) % (4 * C_N)) for r in range(B_c32) for j in range(qlen_c32)],
            dtype=torch.int64, device=device,
        )
        q_c = torch.randn((T_c32, H, D), dtype=torch.bfloat16, device=device) * 0.3
        r_c = torch.randn((T_c32, H, D_REL), dtype=torch.bfloat16, device=device) * 0.3
        k_c = torch.randn((T_c32, D), dtype=torch.bfloat16, device=device) * 0.3
        v_c = torch.randn((T_c32, D), dtype=torch.bfloat16, device=device) * 0.3
        kd_c = torch.randn((T_c32, 4), dtype=torch.bfloat16, device=device) * 0.2
        vd_c = torch.randn((T_c32, 4), dtype=torch.bfloat16, device=device) * 0.2
        tau_c = torch.randn((T_c32,), dtype=torch.float32, device=device) * 0.1 + 1.0
        gate_c = torch.randn((T_c32, H * D), dtype=torch.bfloat16, device=device) * 0.3
        sink_c = torch.randn((T_c32, H), dtype=torch.bfloat16, device=device) * 0.3
        qo_c = torch.empty_like(q_c)
        rel_c = torch.empty((T_c32 + 128, H, E + 256), dtype=torch.bfloat16, device=device)
        out_c = torch.empty((T_c32, H, D), dtype=torch.float32, device=device)
        lse_c = torch.empty((H, T_c32), dtype=torch.float32, device=device)
        y_c = torch.empty((T_c32, H * D), dtype=torch.bfloat16, device=device)

        def _base_c32_48l():
            for l in range(n_layers):
                is_loc = (l % 6) != 5
                sconv_ops.attention_prologue(
                    q_c, r_c, k_c, v_c, q_sc[l], r_sc[l], None if is_loc else tau_c,
                    rp_loc[l] if is_loc else rp_ful[l],
                    kw_s[l] if is_loc else kw_d[l], vw_s[l] if is_loc else vw_d[l],
                    None if is_loc else kd_c, None if is_loc else vd_c,
                    cc_c32[l], pos_c32, bt_c_c32, seq_idx_c32, c_slot_c32, qs_c32,
                    kc_c32[l], vc_c32[l], kv_slot_c32,
                    qo_c, rel_c, cuq_c32, seql_c32,
                    clip_threshold=clip_thr, rel_threshold=rel_thr, norm_eps=norm_eps,
                )
                sm12_attn.rel_attention(
                    qo_c, kc_c32[l], vc_c32[l], block_table=bt_kv_c32, cache_seqlens=seql_c32,
                    cu_seqlens_q=cuq_c32, max_seqlen_q=qlen_c32, softmax_scale=scale,
                    window_size=(1023, 0) if is_loc else (-1, -1),
                    sheared_bias=rel_c, out=out_c, lse=lse_c, num_splits=1,
                )
                attn_ops.gate_sink(out_c, gate_c, lse_c.t(), sink_c, y_c)

        def _v2_c32_48l():
            st = int(torch.cuda.current_stream(device).cuda_stream)
            for l in range(n_layers):
                is_loc = (l % 6) != 5
                dyn = not is_loc
                sconv_scale = float((4 * (2 if dyn else 1) + 1) ** -0.5)
                kw, vw = (kw_d[l], vw_d[l]) if dyn else (kw_s[l], vw_s[l])
                lib.attn_prologue_kvq_exact_fwd(
                    k_c.data_ptr(), v_c.data_ptr(), kw.data_ptr(), vw.data_ptr(),
                    kd_c.data_ptr() if dyn else 0, vd_c.data_ptr() if dyn else 0,
                    cc_c32[l].data_ptr(), kc_c32[l].data_ptr(), vc_c32[l].data_ptr(),
                    kv_slot_c32.data_ptr(), pos_c32.data_ptr(), seq_idx_c32.data_ptr(),
                    c_slot_c32.data_ptr(), bt_c_c32.data_ptr(), qs_c32.data_ptr(),
                    q_c.data_ptr(), q_sc[l].data_ptr(), tau_c.data_ptr() if dyn else 0,
                    qo_c.data_ptr(), 0,
                    T_c32, k_c.stride(0), v_c.stride(0), kw.stride(0), kw.stride(1),
                    kd_c.stride(0) if dyn else 0, vd_c.stride(0) if dyn else 0,
                    cc_c32[l].stride(0), cc_c32[l].stride(2),
                    kc_c32[l].stride(0), kc_c32[l].stride(1), BS,
                    bt_c_c32.stride(0), bt_c_c32.shape[1], C_N, q_c.stride(0),
                    sconv_scale, clip_thr, 1e-6, norm_eps,
                    int(dyn), int(dyn), 0, 0, st,
                )
                lib.fused_decode_attn_otf_mgs_c32_fwd(
                    qo_c.data_ptr(), r_c.data_ptr(), r_sc[l].data_ptr(),
                    tau_c.data_ptr() if dyn else 0, rel_proj_t[l].data_ptr(),
                    kc_c32[l].data_ptr(), vc_c32[l].data_ptr(), bt_kv_c32.data_ptr(),
                    seql_c32.data_ptr(), cuq_c32.data_ptr(),
                    gate_c.data_ptr(), sink_c.data_ptr(), 0, 0, 0, y_c.data_ptr(), 0,
                    qo_c.stride(0), qo_c.stride(1), r_c.stride(0), r_c.stride(1),
                    kc_c32[l].stride(0), kc_c32[l].stride(1), kc_c32[l].stride(2),
                    vc_c32[l].stride(0), vc_c32[l].stride(1), vc_c32[l].stride(2),
                    bt_kv_c32.stride(0), gate_c.stride(0), sink_c.stride(0), y_c.stride(0),
                    0, 0, 0, 0, 0, 0,
                    float(scale * LOG2E), rel_thr, 1e-6,
                    T_c32, T_c32, mt_c32, B_c32, Hkv, BS, E, G, 1023 if is_loc else -1, 1,
                    int(dyn), 0, 0, 0, int(tree_mode), 1, 1, st,
                )

        t_b = bench_graph_cold_us(_base_c32_48l)
        t_v = bench_graph_cold_us(_v2_c32_48l)
        print(
            f"    M={T_c32:3d} (qlen={qlen_c32}): triton={t_b:7.1f} us ({t_b/48:5.2f} us/L) | "
            f"v2_c32={t_v:7.1f} us ({t_v/48:5.2f} us/L) | "
            f"delta={t_v - t_b:+7.1f} us ({t_b / t_v:.2f}x)"
        )


def main():
    ap = argparse.ArgumentParser(description="GB10 0-ULP Verifier & 48L Cold-L2 Benchmark for v2 Fused Decode Attention")
    ap.add_argument("--lib", default=DEFAULT_V2_SO, help="Path to libfused_decode_attn_exact_sm121_v2.so")
    ap.add_argument("--build", action="store_true", help="Build .so via build_c1_attn_v2.sh before running")
    ap.add_argument("--verify", action="store_true", help="Run Suites A (C1 Linear), B (S8-P6v2 Tree), C (C32)")
    ap.add_argument("--suites", default="ABC", help="Subset of verification suites to run (default: ABC)")
    ap.add_argument("--bench", action="store_true", help="Run Suite D (48L cold-L2 CUDA-graph timing)")
    ap.add_argument("--tree-mode", type=int, default=0, help="Diagnostic tree_mode bitfield (0=production fast path)")
    ap.add_argument("--dump-ptx", action="store_true", help="Always dump live _attn_kernel.ptx softmax loop")
    ap.add_argument("--continue-on-fail", action="store_true", help="Run all cases even after first diff")
    args = ap.parse_args()

    if not (args.build or args.verify or args.bench):
        args.verify = True

    if args.build:
        build_sh = SCRIPT_DIR / "build_c1_attn_v2.sh"
        subprocess.run(["bash", str(build_sh), args.lib], check=True)

    lib = load_v2_lib(args.lib)

    tot_fail = 0
    if args.verify:
        if "A" in args.suites.upper():
            tot_fail += verify_suite_a_c1_linear(lib, args.tree_mode, args.continue_on_fail, args.dump_ptx)
        if "B" in args.suites.upper():
            tot_fail += verify_suite_b_tree_2b(lib, args.tree_mode, args.continue_on_fail)
        if "C" in args.suites.upper():
            tot_fail += verify_suite_c_c32(lib, args.tree_mode, args.continue_on_fail)

    if args.bench:
        bench_48l_cold_l2(lib, args.tree_mode)

    if tot_fail > 0:
        print(f"\n[OVERALL] Verification finished with {tot_fail} failing BF16 case(s) across suites {args.suites}.")
        sys.exit(1)
    elif args.verify:
        print(f"\n[OVERALL] ALL VERIFICATION SUITES ({args.suites}) PASSED WITH LITERAL 0-ULP!")


if __name__ == "__main__":
    main()
