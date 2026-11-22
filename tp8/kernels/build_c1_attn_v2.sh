#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build SM121a (GB10) Unified 0-ULP Bit-Exact Fused Target Decode Attention v2:
#   - Linear C=1 (M <= 16, Local S=8 & Global S=16)
#   - S8-P6v2 Fast 4-Tile BM=16 Beam-2 Tree (M <= 8, ws <= 2, Local S=8 & Global S=16)
#   - Batched C=32 (M = 17..512, S in {1, 2, 4}, FUSE_MT=0/1, L2::evict_first)
#
# STRICT 0-ULP COMPILER RULES:
#   - ALWAYS pass `--fmad=false` (prevents unauthorized FMA contraction!)
#   - NEVER pass `--use_fast_math` or `--ftz=true`
#   - Enforces 0 stack frame / 0 register spills across ALL compiled kernels via ptxas -v!
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-/rdma}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_121a}"

SRC="${SCRIPT_DIR}/fused_decode_attn_exact_sm121_v2.cu"
OUT_SO="${1:-${OUT_DIR}/libfused_decode_attn_exact_sm121_v2.so}"
PTXAS_LOG="${SCRIPT_DIR}/build_c1_attn_v2_ptxas.log"

mkdir -p "$(dirname "${OUT_SO}")"

echo "[build-v2] Compiling ${SRC} -> ${OUT_SO} (arch=${ARCH}, --fmad=false, 0-spill gate)..."
"${NVCC}" -O3 -std=c++17 -arch="${ARCH}" --fmad=false --shared -Xcompiler -fPIC \
    --ptxas-options=-v \
    "${SRC}" -o "${OUT_SO}" 2>&1 | tee "${PTXAS_LOG}"

echo ""
echo "[build-v2] Checking ptxas register/SMEM summary and enforcing 0 spills..."
python3 - "${PTXAS_LOG}" <<'PY'
import re, sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
kernels = re.findall(r"Compiling entry function '([^']+)'", log)
spills = re.findall(r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads", log)
regs = [int(x) for x in re.findall(r"Used\s+(\d+)\s+registers", log)]
smem = [int(x) for x in re.findall(r"(\d+)\s+bytes smem", log)]

bad = [(int(sf), int(ss), int(sl)) for sf, ss, sl in spills if int(sf) > 0 or int(ss) > 0 or int(sl) > 0]
print(f"  Compiled kernels : {len(kernels)}")
if regs:
    print(f"  Registers        : min={min(regs)}, max={max(regs)}")
if smem:
    print(f"  Static SMEM      : max={max(smem)} B")
print(f"  Spill entries    : {len(bad)} non-zero out of {len(spills)} reported")
if bad:
    print(f"[FAIL] Non-zero stack frame or register spills detected: {bad[:5]}", file=sys.stderr)
    sys.exit(1)
print("  [PASS] 0 stack frames, 0 spill stores, 0 spill loads across ALL kernels!")
PY

echo ""
echo "[build-v2] Verifying all 5 exported C ABI symbols..."
REQUIRED_SYMS=(
    "attn_prologue_kvq_exact_fwd"
    "fused_decode_attn_otf_mgs_fwd"
    "fused_decode_attn_otf_mgs_c32_fwd"
    "attn_prologue_kvq_tree_exact_fwd"
    "fused_decode_attn_otf_mgs_tree_fwd"
)
NM_OUT="$(nm -D --defined-only "${OUT_SO}")"
for sym in "${REQUIRED_SYMS[@]}"; do
    if ! grep -qw "${sym}" <<< "${NM_OUT}"; then
        echo "[FAIL] Missing exported symbol: ${sym}" >&2
        exit 1
    fi
    echo "  [OK] ${sym}"
done

echo ""
echo "[build-v2] DONE: ${OUT_SO}"
