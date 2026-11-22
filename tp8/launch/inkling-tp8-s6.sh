#!/usr/bin/env bash
# Inkling-2-Small TP8 S6 Launcher (8 DGX Sparks: Big-Core X925 Pinning, 8-ch NCCL, 4 MiB RDMA, PARITY=1)
set -euo pipefail
RANK="${1:?usage: inkling-tp8-s6.sh <rank 0..7>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${INKLING_IMAGE:-vllm/vllm-openai:inkling-2-small}"
if [ "${INKLING_MOE_TP:-0}" = "1" ]; then
  MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling2-small-tp8r$RANK}"
  [ -d "$MODEL_DIR" ] || { echo "no TP slice $MODEL_DIR"; exit 1; }
fi
MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling2-small-ep8r$RANK}"
[ -d "$MODEL_DIR" ] || MODEL_DIR="$HOME/models/inkling2-small"
IP="10.77.0.$((RANK + 1))1"

if [ "${INKLING_FASTSTART:-1}" = "1" ]; then
  if [ "${SERVING:-0}" = "1" ]; then MAX_BATCHED="${MAX_BATCHED:-4096}"; else MAX_BATCHED="${MAX_BATCHED:-2048}"; fi
  [ -z "${VLLM_EXTRA:-}" ] && VLLM_EXTRA="--cudagraph-capture-sizes 1 2 4 8 16 24 32 48"
fi

if [ "${PARITY:-0}" = "1" ]; then
  GMU="${GMU:-0.80}" MAXLEN="${MAXLEN:-1048576}" SEQS="${SEQS:-32}" MAX_BATCHED="${MAX_BATCHED:-8192}"
else
  GMU="${GMU:-0.80}" MAXLEN="${MAXLEN:-32768}" SEQS="${SEQS:-4}" MAX_BATCHED="${MAX_BATCHED:-8192}"
fi

ARGS=(/model --served-model-name inkling-2-small --trust-remote-code
  --quantization "${QUANT:-inkling2}"
  --tensor-parallel-size 8 --nnodes 8 --node-rank "$RANK"
  --master-addr 10.77.0.11 --master-port "${MASTER_PORT:-29511}"
  --distributed-executor-backend mp
  --distributed-timeout-seconds 7200 --cpu-distributed-timeout-seconds 7200
  --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED"
  --kv-cache-dtype "${KV_DTYPE:-auto}" --enable-prefix-caching --enable-chunked-prefill
  --disable-uvicorn-access-log
  --host 0.0.0.0 --port "${PORT:-8010}")
[ "${REASONING:-1}" = "1" ] && ARGS+=(--reasoning-parser inkling)
[ "$RANK" != "0" ] && ARGS+=(--headless)
if [ "${PARITY:-0}" = "1" ]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser inkling --enable-prompt-tokens-details)
  if [ "${IMAGES:-16}" -gt 0 ]; then
    ARGS+=(--limit-mm-per-prompt "{\"image\":${IMAGES:-16}}" --mm-processor-cache-gb 0.5 --allowed-media-domains media.invalid)
  else
    ARGS+=(--limit-mm-per-prompt '{"image":0}' --mm-processor-cache-gb 0)
  fi
else
  ARGS+=(--limit-mm-per-prompt '{"image":0}' --mm-processor-cache-gb 0)
fi
if [ -n "${LONG_PREFILL_THRESHOLD:-}" ] && [ "${LONG_PREFILL_THRESHOLD}" -gt 0 ] 2>/dev/null; then
  ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL_THRESHOLD" --long-prefill-token-threshold-adaptive)
fi
if [ -n "${STREAM_INTERVAL:-}" ] && [ "${STREAM_INTERVAL}" -gt 1 ] 2>/dev/null; then
  ARGS+=(--stream-interval "$STREAM_INTERVAL")
fi
if [ -n "${PREFILL_SCHEDULE_INTERVAL:-}" ] && [ "${PREFILL_SCHEDULE_INTERVAL}" -gt 1 ] 2>/dev/null; then
  ARGS+=(--prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL")
fi
if [ -n "${CUDAGRAPH_REQ_SIZES:-}" ]; then
  case "${VLLM_EXTRA:-}" in
    *compilation-config*) ;;
    *) ARGS+=("--compilation-config={\"decode_cudagraph_capture_request_sizes\":[${CUDAGRAPH_REQ_SIZES}]}") ;;
  esac
fi
if [ "${DSPARK:-0}" = "1" ]; then
  SPEC_JSON="{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_K:-7},\"draft_sample_method\":\"${DRAFT_SAMPLE:-greedy}\"${ADAPTIVE:+,\"enable_adaptive_verification\":true}${SPEC_BATCH_SCHEDULE:+,\"num_speculative_tokens_per_batch_size\":${SPEC_BATCH_SCHEDULE}}}"
  ARGS+=(--speculative-config "$SPEC_JSON")
fi
# shellcheck disable=SC2206
[ -n "${VLLM_EXTRA:-}" ] && ARGS+=($VLLM_EXTRA)

# NCCL across all 4 ConnectX-7 200GbE RoCEv2 ports (GID 5, PCIe-alternating order).
# Export defaults BEFORE forwarding caller env so ANY NCCL_* tuning knob in .env always wins!
export NCCL_SOCKET_IFNAME="enp1s0f0np0"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1}"
export NCCL_IB_GID_INDEX="5"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [ "${SERVING:-0}" = "1" ]; then
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
  export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
else
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
  export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
fi
ENVS=(-e GLOO_SOCKET_IFNAME=enp1s0f0np0
  -e VLLM_HOST_IP="$IP" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
for k in INKLING_LOAD_STAGED INKLING_LOAD_CHUNK_MB INKLING_LOAD_DROP_CACHE PYTORCH_CUDA_ALLOC_CONF VLLM_LOGGING_LEVEL VLLM_TORCH_PROFILER_DIR VLLM_PROFILER_MAX_ITERS INKLING_MOE_TP INKLING_L2PF INKLING_L2PF_MB INKLING_L2PF_GATE INKLING_L2PF_CTAS INKLING_L2PF_WARPS INKLING_SKINNY INKLING_EXTRA_ENV; do
  [ -n "${!k:-}" ] && ENVS+=(-e "$k=${!k}")
done
export INKLING_CERT_HEAD_CACHE="${INKLING_CERT_HEAD_CACHE:-/root/.cache/certhead}"

# Default RDMA knobs BEFORE forwarding caller env so caller env ALWAYS wins!
if [ "${RDMA:-0}" = "1" ]; then
  MOUNTS_RDMA=(-v "$HERE/rdma:/rdma:ro")
  export DSV41_RDMA_AR="${DSV41_RDMA_AR:-1}"
  export RDMA_AR_EXT="${RDMA_AR_EXT:-/rdma/rdma_ar_ext_s6.so}"
  export RDMA_AR_MAX_BYTES="${RDMA_AR_MAX_BYTES:-4194304}"
  export RDMA_AR_MAX_REDUCE_BYTES="${RDMA_AR_MAX_REDUCE_BYTES:-4194304}"
  export RDMA_AR_MAX_GATHER_BYTES="${RDMA_AR_MAX_GATHER_BYTES:-1048576}"
  export DSV41_RDMA_PARLOAD="${DSV41_RDMA_PARLOAD:-1}"
  export RDMA_AR_XSTREAM="${RDMA_AR_XSTREAM:-wait}"
  export RDMA_AR_FINGERPRINT="${RDMA_AR_FINGERPRINT:-1}"
  export RDMA_AR_SPLIT_POST="${RDMA_AR_SPLIT_POST:-1}"
  export RDMA_AR_PUB_V1="${RDMA_AR_PUB_V1:-1}"
  export RDMA_AR_PUB_V2="${RDMA_AR_PUB_V2:-1}"
  export RDMA_AR_HYG_PROXY="${RDMA_AR_HYG_PROXY:-1}"
  export RDMA_AR_HYG_POLL="${RDMA_AR_HYG_POLL:-1}"
  export DSV41_RDMA_TRIM="${DSV41_RDMA_TRIM:-${RDMA_TRIM:-1}}"
  export RDMA_AR_HCAS="${RDMA_AR_HCAS:-rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1}"
else
  MOUNTS_RDMA=()
fi

# Forward every NCCL_*, INKLING_*, VLLM_*, DSV41_*, RDMA_*, and OMP/BLAS/TORCH thread env var cleanly.
while IFS= read -r kv; do
  ENVS+=(-e "$kv")
done < <(env | grep -E '^(NCCL_[A-Z0-9_]+|INKLING_[A-Z0-9_]+|VLLM_[A-Z0-9_]+|DSV41_[A-Z0-9_]+|RDMA_AR_[A-Z0-9_]+|RDMA_[A-Z0-9_]+|OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|TORCH_NUM_THREADS|MALLOC_[A-Z0-9_]+)=' | grep -v '^VLLM_EXTRA=' || true)

MOUNTS=(-v "$MODEL_DIR:/model:ro" -v "$HOME/.cache/inkling-jit:/root/.cache"
  -v "$HOME/.triton:/root/.triton"
  -v "$HOME/.cache/certhead:/root/.cache/certhead"
  -v "$HOME/inkling-prof:/prof"
  "${MOUNTS_RDMA[@]}")
[ -n "${INKLING_MX_NVFP4:-}" ] && MOUNTS+=(-v "${INKLING_MX_NVFP4_HOST:-$HOME/ink-work/lossy/sidecar}:${INKLING_MX_NVFP4}:ro")

PATCH_DIR="${PATCH_DIR:-$HERE/patches-s6}"
while read -r f dest; do
  [ -z "$f" ] || [ "${f:0:1}" = "#" ] && continue
  MOUNTS+=(-v "$PATCH_DIR/$f:$dest:ro")
done < "$PATCH_DIR/MANIFEST"

mkdir -p "$HOME/.cache/inkling-jit" "$HOME/.cache/certhead" "$HOME/inkling-prof"
docker rm -f "inkling-tp8-r$RANK" >/dev/null 2>&1 || true

CPUSET_ARGS=()
[ -n "${INKLING_CPUSET:-}" ] && CPUSET_ARGS=(--cpuset-cpus "$INKLING_CPUSET")
echo "[rank $RANK] $MODEL_DIR ip=$IP gmu=$GMU maxlen=$MAXLEN dspark=${DSPARK:-0} cpuset=${INKLING_CPUSET:-all} nccl_ch=$NCCL_MAX_NCHANNELS rdma_ext=${RDMA_AR_EXT:-none}"
exec docker run -d --name "inkling-tp8-r$RANK" --gpus all --privileged --ipc=host --network host \
  --ulimit memlock=-1 --shm-size 16g "${CPUSET_ARGS[@]}" "${ENVS[@]}" "${MOUNTS[@]}" "$IMAGE" "${ARGS[@]}"
