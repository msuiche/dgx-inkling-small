#!/usr/bin/env bash
# inkling2-tp4.sh <rank> — Inkling-2-Small TP4 (+EP4 experts) across 4 DGX Sparks.
#
# Launch order: workers first (3, 2, 1), then rank 0 on the head. Rank 0
# waits for API readiness and prints the bench hint.
#
# Expects .env next to this script's repo root (see config/cluster.env.example).
set -euo pipefail

RANK="${1:?usage: inkling2-tp4.sh <rank 0..3>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.env"

: "${MASTER_ADDR:?set in .env}" "${MASTER_PORT:=29500}"
MODEL="${MODEL:-thinkingmachines/Inkling-2-Small}"
SERVED="${SERVED_MODEL_NAME:-inkling-2-small}"
IMAGE="${INKLING_IMAGE:-vllm/vllm-openai:inkling-2-small}"
GMU="${GMU:-0.85}"
MAXLEN="${MAXLEN:-262144}"
SEQS="${SEQS:-4}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
KV_DTYPE="${KV_DTYPE:-auto}"
MM_LIMIT="${MM_LIMIT:-2}"
PORT="${VLLM_PORT:-8000}"

ARGS=(
  "$MODEL"
  --served-model-name "$SERVED"
  # config.json advertises modelopt/NVFP4, but the model class asserts on
  # the inkling2 per-tensor metadata in hf_quant_config.json — force it.
  --quantization "${QUANT:-inkling2}"
  --tensor-parallel-size 4
  --nnodes 4
  --node-rank "$RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  # The post-load rendezvous uses torch gloo (30-min default): with a ~2h
  # first load on the head, every other rank times out and dies first.
  --distributed-timeout-seconds "${DIST_TIMEOUT:-21600}"
  --cpu-distributed-timeout-seconds "${DIST_TIMEOUT:-21600}"
  --gpu-memory-utilization "$GMU"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --max-num-batched-tokens "$MAX_BATCHED"
  --kv-cache-dtype "$KV_DTYPE"
  --enable-prefix-caching
  --enable-chunked-prefill
  --limit-mm-per-prompt "{\"image\":$MM_LIMIT}"
  --mm-processor-cache-gb 0
  --host 0.0.0.0
  --port "$PORT"
)
[ "${REASONING:-1}" = "1" ] && ARGS+=(--reasoning-parser inkling)
# Followers are worker-only: a full `vllm serve` on every node makes each
# follower's engine core run init collectives (get_kv_cache_specs, encoder
# measurement) that only the driver's executor may call -> assert. The dsv41
# lane's worker containers run --headless for the same reason.
[ "$RANK" != "0" ] && ARGS+=(--headless)
[ "${TOOLS:-0}" = "1" ] && ARGS+=(--tool-call-parser inkling --enable-auto-tool-choice)
if [ "${DSPARK:-1}" = "1" ]; then
  ARGS+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}')
fi
# shellcheck disable=SC2206
[ -n "${VLLM_EXTRA:-}" ] && ARGS+=($VLLM_EXTRA)

# The fabric netdev names flap between enp1s0* and rocep1s0* on this ring
# (driver re-enumeration); resolve by the 10.100.x address, not by name.
FABRIC_IF="${NCCL_SOCKET_IFNAME:-}"
if [ ! -d "/sys/class/net/$FABRIC_IF" ]; then
  FABRIC_IF=$(ip -br addr show | awk '/10\.100\./{print $1; exit}')
fi
[ -n "$FABRIC_IF" ] || { echo "no 10.100.x fabric interface found" >&2; exit 1; }

ENV_PAIRS=(
  -e "NCCL_SOCKET_IFNAME=$FABRIC_IF"
  # The executor's gloo control-plane derives the rank's advertise address
  # from the default interface; each node's hostname maps to 127.0.0.1 in
  # /etc/hosts, so without this the full-mesh connect dies on localhost.
  -e "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-wlP9s9}"
  -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
  -e "VLLM_HOST_IP=${VLLM_HOST_IP:?set per-node in .env}"
  -e "HF_HUB_OFFLINE=0"
)
# Optional torch allocator knob (workaround for driver livelock on GB10 at
# high occupancy: expandable segments' VA remapping spins forever there).
[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && ENV_PAIRS+=(-e "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF")
# Staged weight fill knobs (patches/gb10_load.py). Unset = staged, which is
# the fix; INKLING_LOAD_STAGED=0 restores the stock mmap copy for an A/B.
for knob in INKLING_LOAD_STAGED INKLING_LOAD_CHUNK_MB INKLING_LOAD_DROP_CACHE; do
  if [ -n "${!knob:-}" ]; then ENV_PAIRS+=(-e "$knob=${!knob}"); fi
done
if [ "${NCCL_IB_DISABLE:-0}" = "1" ]; then
  # Socket collectives (TCP over the routed fabric). On this switchless ring,
  # RoCE full-mesh for EP all-to-all needs cross-subnet QPs, which stall in
  # verbs; sockets through the routed fabric work everywhere.
  ENV_PAIRS+=(-e "NCCL_IB_DISABLE=1")
else
  ENV_PAIRS+=(
    -e "NCCL_IB_HCA=${NCCL_IB_HCA:?set in .env}"
    -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}"
    -e "NCCL_IB_AUTO_DETECT=0"
  )
fi
# shellcheck disable=SC2206
[ -n "${NCCL_EXTRA:-}" ] && ENV_PAIRS+=($NCCL_EXTRA)

echo "[rank $RANK] $SERVED <- $IMAGE  maxlen=$MAXLEN seqs=$SEQS gmu=$GMU kv=$KV_DTYPE"
# -d: daemonize at docker. Remote ssh+nohup launches stall on the inherited
# session; `docker logs -f inkling2-tp4-rank$RANK` follows the boot.
# no --rm: a crashed rank must keep its docker logs for diagnosis
MOUNTS=(-v "${HF_CACHE:?set in .env}:/root/.cache/huggingface")
# SM121 (GB10) patches over the stock image: each entry is host:container.
if [ -f "$ROOT/patches/MANIFEST" ]; then
  while read -r host_file dest; do
    [ -f "$ROOT/patches/$host_file" ] && MOUNTS+=(-v "$ROOT/patches/$host_file:$dest:ro")
  done < "$ROOT/patches/MANIFEST"
fi

exec docker run -d --name "inkling2-tp4-rank$RANK" \
  --gpus all --privileged --ipc=host --network host \
  "${ENV_PAIRS[@]}" \
  "${MOUNTS[@]}" \
  "$IMAGE" "${ARGS[@]}"
