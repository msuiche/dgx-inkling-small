# Inkling-2-Small on 8× NVIDIA DGX Spark (TP8, SM121a)

Production-grade, **100% 0-ULP bit-exact** inference stack for
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B total / 17B active MoE, 384 routed experts top-8, 40 NVFP4 local SWA-1024 layers + 8 MXFP8 global layers, 1M context, 5-layer DSpark speculator)
on **8× NVIDIA DGX Spark (GB10, SM121a)** over **4× 200GbE ConnectX-7 RoCEv2**.

> **Status (2026-11-22): LIVE & 5-GATE QUALIFIED on 8/8 nodes (`tp8/config/best.env`).**
> This repository started as a fork of `msuiche/dgx-inkling-small` (which documented the 4-Spark TP4 driver livelock — preserved under `launch/inkling2-tp4.sh`, `repro/`, and `nvidia-bug-report.md`). On 8× DGX Spark (`TP8`), the model runs cleanly with **17–23 GiB/node** free UMA headroom, **0-ULP deterministic decode**, **518.7 tok/s C=32**, **120.9 s 1M cold prefill**, and **4× simultaneous 1M-token KV sessions**.

---

## Qualified Checkpoint Results (`EXP-6B` → `tp8/config/best.env`)

All numbers below were measured on **one unified live server** (`GMU=0.72`, `MAX_BATCHED=16384`, port `8010`, stock 2000 MHz GB10 clocks, `best.env` sha256 `a4a8aa6235d0…`).

| Metric | Starting Baseline | Prior `best.env` | **Current (`EXP-6B`)** | How Measured |
|---|---:|---:|---:|---|
| **Bit-Exact Decode (0-ULP)** | `1dcf…f8c3` | `1dcf…f8c3` | **`1dcf…f8c3` (3/3 PASS)** | `exact_ink.py` (10 prompts, `temp=0`, 384 tok, serial). Full SHA-256: `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` |
| **C=32 Throughput** (`conc 32, 512 tok`) | `420.94 tok/s` (`S8-P3`) | `508.2 tok/s` | **`518.71 tok/s` median** (`514.2–524.7`, peak **`528.07`** in `EXP-6`) | `bench.py conc --conc 32 --max-tokens 512 --temp 0`, `n=6`, **0 outliers** (`mean 518.99 tok/s`, `wall 31.23–31.86 s`, steady-state `Running: 32 > 615 tok/s`) |
| **C=1 Decode** (`G1 GM, 1024 tok`) | `~110.9 → 126.1 tok/s` | `143.6 tok/s`* | **`139.5–140.3 tok/s`** (`131.6 / 102.9 / 203.8`) | `bench.py single --repeats 3` (9 samples). \*See note below: Patch 02 fixed a silent FP32 prefill leak in `math` (`T=125→M=128`), restoring true 0-ULP BF16 prefill while *lowering* step ITL (`26.12 → 25.75 ms`). |
| **1M Cold Prefill** (FP8/TopK, warm server) | `~250+ s` | `121.8 s` | **`120.90 s` / `121.44 s`** (`8,670 tok/s`, warm TTFT `2.95 s`) | `bench_1m_e2e.py --ctx 1048500 --warm --max-tokens 256`, **3/3 needles** (prefill-only FP8/TopK; decode remains 100% 0-ULP BF16). Historical `S7-U10f` peak: `118.26 s`. |
| **Multi-1M Simultaneous KV Residency** | `~2.5×` | `2× 1M` (3rd OOM) | **`4× 1M` simultaneous** (`4.19M` cached tok) | `bench_multi_1m.py --sessions 4`: **all 4 sessions `all_warm_ok=True`**, oldest-first Turn-2 TTFT **`2.93–2.99 s`** (`42.5×` speedup), **12/12 needles**, `17.1–22.7 GiB` `MemAvailable` free on 8/8 nodes. |
| **WikiText-2 Quality** (`lm-eval 0.4.13`) | — | — | **`word_ppl = 5.8052`** · **`BPB = 0.4745`** (`0.4936` detok) | `EleutherAI/wikitext_document_level` (`wikitext-2-raw-v1`, test, 62 docs, `max_length=2048`, `<\|begin_of_text\|>=200028`). |

Full machine-readable archive: [`tp8/results/checkpoint_2026-11-22.json`](tp8/results/checkpoint_2026-11-22.json).

---

## Hardware Topology (8× DGX Spark)

- **GPU:** NVIDIA GB10 (`SM121a`), 48 SMs, 99 KiB SMEM/SM, **244 GB/s** sustained UMA LPDDR5X, 24 MiB L2, 121.63 GiB usable UMA/node (keep `MemAvailable >= 6 GiB` on all 8 nodes to avoid the 4 GiB UMA swap cliff).
- **CPU:** 20-core Grace ARM (`0..4, 10..14` Cortex-A725 LITTLE; `5..9, 15..19` Cortex-X925 BIG). Container pinned to `--cpuset-cpus=5-9,15-19` with `SCHED_FIFO` RDMA proxies on CPUs `9` & `19` (`tp8/patches/host_pin.py`).
- **Fabric:** 4× 200GbE ConnectX-7 RoCEv2 (`~25.8 GB/s` aggregate wire, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`, `NCCL_IB_QPS_PER_CONNECTION=2`, `NCCL_IB_PCI_RELAXED_ORDERING=1`).

---

## What We Built & Proven Wins

### 1. Custom 0-ULP SM121a Decode Stack (`C=1` & `C=32`)
All target decode kernels are compiled with **`-gencode arch=compute_121a,code=sm_121a --fmad=false`** (no `--ftz=true`, no `--use_fast_math`, verified `0` register spills) and gated by 3-pass `exact_ink.py`:
- **Lossless RDMA AllReduce (`tp8/kernels/rdma_ar_c32_nlive.cu`):** Lossless `BF12-G128-OVF` wire encoding (`0.7579×` bytes, exhaustive 65,536 BF16 `nd==0`) + `Pipe2` + `Dual-QP` + `PCIe RO` + `SENDBUF_RING` (`18.8 MiB` vs `1.14 GiB`). Enabling `RDMA_AR_LEVEL_DB=1` unlocked `RDMA_AR_C32_PIPE=1` for **`+4.45 tok/s`** on C=32 while keeping `RDMA_AR_ROWS_P2=1` (optimal for C=1).
- **0-ULP C=32 MoE & Router (`nvfp4_ws5_c32.cu`, `mxfp8_mx7_c32.cu`, `router_bf12_c32.cu`):** 102/102 `nd==0` on GB10. Fixed the PIECEWISE prefill graph capture leak via `_is_full_decode_capturing()` (`Patch 02`), safely unlocking `INKLING_C32_MOE_MIN_M=17` (covering `B=12, K=1 → M=24` tail drain) with **3/3 `1dcf…f8c3` PASS**.
- **0-ULP Fused ShortConv+AttnRes (`fused_sconv_ar_c32_exact.cu`):** 372.3M/372.3M literal `nd==0` across `T=32..256` including in-place alias (`-0.84 ms` @ `M=128`).
- **0-ULP Lossless BF11 Dense GEMMs (`bf11_flip_exact_c32.cu`):** Revived dead `BF11_C32` path on `M=24..64` (pinned `INKLING_BF11_LIB=/rdma/libbf11_flip_exact_c32.so` and added missing canonical capture buckets `24, 36, 48, 60` in `skinny_bf11.py`), beating `torch.mm` by **`-0.78 to -1.32 ms/step`** across `M=24..64` while keeping `torch.mm` for `M >= 72`.
- **L2 Prefetch (`l2pf.py`):** Cooperative `ar_done_cnt` soft-stop (`L2PF_C32=1`) + fixed C=32 BF12 gate prefetch bug (previously prefetched 170 MB/step of unread BF16 `gate.weight` instead of 2.65 MB BF12 slabs).
- **Speculative Decode Policy:** `SPEC_BATCH_SCHEDULE='[[1,1,7],[2,4,7],[5,8,5],[9,12,1],[13,24,2],[25,32,3]]'` + confidence pruning `tau=0.30` at `B=32` (prunes hopeless tokens from MoE routing, cutting step time `-9.2 ms/step`) + `TREE_SPEC=0` (`+2.1 tok/s` C=1).

### 2. 1M Cold Prefill & 4× Simultaneous Multi-1M KV Residency
- **Compact Logical-Block FP8 Shadow K (`tp8/kernels/global_attn_sage2_sm121_v5.cu`):** Root-caused why `3× 1M` OOM'd at `GMU=0.72` — startup warmup ran `past=0 < 4096`, so PyTorch locked its memory fraction *before* `alloc_fp8_shadow_k_v3` lazily allocated `6.22–9.57 GiB` sized to the full KV block pool. Decoupling Shadow K to a constant **`1.016 GiB`** (`cap_blks=4096`, `0.0000000` bit-exact match across solo, multi-req, and session switches) reclaimed **`5.20–8.55 GiB/rank`** and doubled simultaneous 1M residency from **`2× → 4× 1M`** with `17–23 GiB` free UMA headroom.
- **Persistent NVFP4 W2 Prefill Kernel (`tp8/kernels/pf_moe_native_sm121_v4.cu`):** `1.24–1.28×` faster standalone on GB10 (`3.335 → 2.613 ms` at `T=8192`; `6.847 → 5.519 ms` at `T=16384`, `W2_PANELS=2`, `L2_ZERO_SLICE=1`), paired with `v3` W13 (`GMU=0.72`, `MAX_BATCHED=16384`).
- **Root-Caused DBO `record_stream` Allocator Limbo:** Discovered that disabling global syncs (`PF_GLOBAL_ZERO_SYNC=1`) without bounded layer depth exploded cross-stream `record_stream` limbo 8× (`2.72 GB → 21.75 GB`), triggering UMA page-fault thrash (`+5 to +52 s`). Adding a depth-2 layer-event ring (`INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS=2`) caps limbo at `0.90 GB`.

### 3. Verified R&D Artifacts Included (Not Active in `best.env`)
- **Zero-Incast 3-Step RHD INT8 ReduceScatter (`tp8/kernels/pf_rhd_comm_sm121.cu`, `tp8/bench/bench_rhd_8rank.py`):** First INT8 ReduceScatter to beat NCCL BF16 Ring on 8× DGX Spark 4×200GbE RoCEv2 — **1.09× faster at `T=8192`** (`2.628` vs `2.872 ms`, `25.13 GB/s` EqBusBW, `42.1 dB SQNR`) and **1.08× at `T=16384`** (`4.949` vs `5.354 ms`) by using 3 pairwise XOR hops (`r^4 → r^2 → r^1`) with zero 7-to-1 incast. End-to-end inside DBO (`EXP-6`), NCCL BF16 Ring still overlapped slightly better (`120.9–121.4 s` vs `124.5 s`), so `best.env` keeps `RS_MODE=nccl` while preserving `rhd_int8` opt-in.
- **Unified 0-ULP Fused Decode & Beam-2 Tree Attention (`tp8/kernels/fused_decode_attn_exact_sm121_v2.cu`, `tp8/bench/verify_c1_attn_v2_gb10.py`):** Passes **100% literal 0-ULP** across Suite A (Linear C=1 local+global), Suite B (`S8-P6v2` 4-tile Beam-2 Tree), and Suite C (all 22 C=32 cases including `M=224`, hetero `qlen`, zero-seqlen) by matching Triton's `BL1_2x2` warp-butterfly reduction and `fma.rn.f32x2` + `ex2.approx.ftz.f32` PTX. Suite D cold-L2 benchmarking proved live Triton + `ns_mgs` is already faster (`469–794 us` vs `690–997 us` across 48 layers), so we kept Triton live and archived `v2.cu`.

---

## Dead Ends Proven & Closed (Saved Future Effort)

| Idea Tested | Measured Outcome | Why |
|---|---|---|
| **7-peer P2P INT8/FP8 All-to-All RS** (`int8_a2a`, `int8_col`) | `130.4–138.6 s` vs `121.8 s` | 7-to-1 RoCEv2 incast / 1-channel-per-peer throttling vs 1-neighbor Ring |
| **C=32 `K=4 / 5 / 7` & Tier-2 Live Compaction** | `K=4` `-2.57%`, `K=7` `-47 tok/s`, Compact `≈ 0` | Extra verify rows (`M 128→160`) & prose stragglers outweigh AL gain; `K=3` is optimal |
| **BF11 Dense GEMMs above `M=64`** (`M=72..256`) | `+1.7 to +7.4 ms/step` slower | Above `M=64`, cuBLAS BF16 Tensor Cores beat custom BF11 unpack on GB10 |
| **`RDMA_AR_WIRE_BF12=1` / `RDMA_AR_ROWS_P2=3` on C=1** | `-1.3` to `-3.2 tok/s` | Extra pack/wire latency hurts C=1 (`M=8`) |
| **Fused Router `librouter_bf12_fused_s8.so`** | `+0.42–0.65 ms/step` slower | 52/52 0-ULP exact, but slower cold-L2 on GB10 |
| **Drafter `nf3+h64_q8`** | `-4.6%` C=1 | NF3 degraded draft acceptance length |

---

## Repository Layout

```text
tp8/
  config/best.env                       # Promoted EXP-6B TP8 serving config (GMU=0.72)
  launch/inkling-tp8-s6.sh              # Per-rank Docker launcher (CPU pinning + RoCEv2 + RDMA)
  patches/                              # vLLM Inkling-2 Python overlays + MANIFEST (py_compile green)
  kernels/                              # SM121a (GB10) CUDA kernels (.cu) + build scripts
  bench/                                # Multi-1M, 8-rank RHD comm, and 0-ULP attention verifiers
  results/checkpoint_2026-11-22.json    # Full verified benchmark & quality metrics archive

# Preserved 4-Spark TP4 archive (2026-11-16):
launch/inkling2-tp4.sh · config/cluster.env.example · patches/ · repro/ · nvidia-bug-report.md
```

## Quickstart (8× DGX Spark TP8)

1. Build the SM121a `.so` libraries on one quiet GB10 node using `-gencode arch=compute_121a,code=sm_121a` (using `--fmad=false` on all decode kernels) and distribute to `/home/<user>/inkling-tp8/rdma/` on all 8 nodes.
2. Sync `tp8/patches/` to `/home/<user>/inkling-tp8/patches-s6/` and `tp8/config/best.env` + `tp8/launch/inkling-tp8-s6.sh` to all 8 nodes.
3. Launch ranks `7..0` via `tp8/launch/inkling-tp8-s6.sh` (`vllm/vllm-openai:inkling-2-small` image) and verify:
   - Health: `curl -sf http://127.0.0.1:8010/health`
   - Bit-exactness: run 3 serial passes of `exact_ink.py` (passes 1–2 warm Triton/CUDA graphs; pass 3+ must match `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3`).
   - Multi-1M residency: `python3 tp8/bench/bench_multi_1m.py --sessions 4`
   - 8-rank RHD comm microbenchmark: `torchrun --nnodes=8 --nproc_per_node=1 ... tp8/bench/bench_rhd_8rank.py --lib /rdma/libpf_rhd_comm_sm121.so`

See [`STATUS.md`](STATUS.md) for the full wave-by-wave engineering log, operational rules, and remaining optimization roadmap.
