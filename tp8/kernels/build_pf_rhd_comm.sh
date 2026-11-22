#!/usr/bin/env bash
# Build SM121a (DGX Spark GB10) shared library for Zero-Incast 3-Step Recursive
# Halving-Doubling (RHD / XOR Butterfly) FP8/INT8 ReduceScatter & AllGather.
#
# Run on ONE quiet GB10 node (NEVER while a live benchmark runs!) and distribute
# /rdma/libpf_rhd_comm_sm121.so to all 8 nodes (spark-one..eight).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/rdma}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${CUDA_ARCH:-sm_121a}"
OUT_SO="$OUT_DIR/libpf_rhd_comm_sm121.so"
BUILD_LOG="$SCRIPT_DIR/build_pf_rhd_comm.log"

mkdir -p "$OUT_DIR"

echo "=== Building libpf_rhd_comm_sm121.so (${ARCH}) ==="
"$NVCC" -O3 -std=c++17 -arch="$ARCH" --shared -Xcompiler -fPIC \
    --ptxas-options=-v \
    "$SCRIPT_DIR/pf_rhd_comm_sm121.cu" \
    -o "$OUT_SO" 2>&1 | tee "$BUILD_LOG"

echo ""
echo "=== Gate 1: Register spill check (must be 0 bytes spill stores/loads!) ==="
if grep -E "spill stores|spill loads" "$BUILD_LOG" | grep -v "0 bytes spill"; then
    echo "FAIL: Non-zero register spills detected!" >&2
    exit 1
else
    echo "PASS: 0 bytes spill stores/loads across all RHD kernels!"
fi

echo ""
echo "=== Gate 2: Exported C ABI symbol check ==="
REQUIRED_SYMS=(
    rhd_quant_slice_int8_g64_fwd
    rhd_quant_slice_fp8_g64_fwd
    rhd_dequant_add_split_quant_int8_g64_fwd
    rhd_dequant_add_split_quant_fp8_g64_fwd
    rhd_dequant_add_quant_int8_g64_fwd
    rhd_dequant_add_quant_fp8_g64_fwd
    rhd_dequant_add_bf16_int8_g64_fwd
    rhd_dequant_add_bf16_fp8_g64_fwd
    rhd_ag_dequant_int8_g64_fwd
    rhd_ag_dequant_fp8_g64_fwd
    rhd_ag_quant_fp8_rowwise_fwd
    rhd_ag_unpack_fp8_rowwise_fwd
)
EXPORTS="$(nm -D --defined-only "$OUT_SO")"
for sym in "${REQUIRED_SYMS[@]}"; do
    if ! grep -qw "$sym" <<< "$EXPORTS"; then
        echo "FAIL: Missing exported symbol: $sym" >&2
        exit 1
    fi
done
echo "PASS: All ${#REQUIRED_SYMS[@]} C ABI symbols exported!"

echo ""
ls -lh "$OUT_SO"
