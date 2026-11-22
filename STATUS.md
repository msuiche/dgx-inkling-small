# Status & Engineering Log — Inkling-2-Small on DGX Spark

**Current Status (2026-11-22): LIVE, HEALTHY & 5-GATE QUALIFIED on 8× DGX Spark (TP8, port `8010`)**
**Promoted Config:** `tp8/config/best.env` (`EXP-6B`, sha256 `a4a8aa6235d000b0a6fa4eb38e4b3d34f7acc8a3413b0496ed80788ee7692cb5`)

---

## 1. Qualified Checkpoint Snapshot (`EXP-6B`)

- **Bit-exact decode (`exact_ink.py`, 10 prompts, `temp=0`, 384 tok, serial):**
  `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` (**3/3 PASS from pass 1**).
- **C=32 throughput (`conc 32, 512 tok, temp=0`, `n=6`):**
  **`518.71 tok/s` median** (`mean 518.99`, raw `524.68 / 515.37 / 514.20 / 522.06 / 523.22 / 514.43`, `wall 31.23–31.86 s`, **0 outliers**; peak **`528.07 tok/s`** observed on `EXP-6`).
- **C=1 decode (`single --repeats 3`, 1024 tok, `temp=0`):**
  **`139.51–140.25 tok/s` G1 GM** (`code 131.1–131.6`, `prose 102.2–102.9`, `math 202.8–203.8` at `25.75–25.82 ms/step` ITL after fixing the silent `router_bf12_c32` prefill leak in `math`).
- **1M cold prefill (`bench_1m_e2e.py --ctx 1048500 --warm --max-tokens 256`):**
  **`120.90 s` (`8,670.3 tok/s`) & `121.44 s` (`8,632.1 tok/s`)**, warm TTFT **`2.95 s`** (`41×` speedup), **3/3 needles**.
- **Multi-1M simultaneous KV residency (`bench_multi_1m.py --sessions 4`):**
  **4× simultaneous 1M sessions** (`4,192,256` cached tokens), all 4 warm Turn-2 queries **`2.93–2.99 s`** (`42.5×` speedup), **12/12 needles**, `all_warm_ok=True`.
- **Node UMA headroom (all 8 ranks live):** `17.1 GiB` free on rank 0, `22.3–22.7 GiB` free on ranks 1–7, zero swap.
- **WikiText-2 perplexity (`lm-eval 0.4.13`, `wikitext-2-raw-v1`, test, `max_length=2048`):**
  `word_perplexity = 5.8052`, `byte_perplexity = 1.3894`, `bits_per_byte = 0.4745` (`0.4936` on detokenized UTF-8 bytes; total NLL `424,448.04 nats`).

---

## 2. Wave-by-Wave Progression (8× DGX Spark TP8)

| Milestone | `exact_ink` | C=1 G1 GM | C=32 (`conc 32, 512t`) | 1M Cold Prefill | Multi-1M KV | Key Changes |
|---|---:|---:|---:|---:|---:|---|
| Starting TP8 baseline | PASS | `~110.9–126.1` | `~360–420.9` (`S8-P3`) | `~250+ s` | `~2.5×` | Stock vLLM TP8 + initial RoCE bring-up |
| `S7-U10f` | PASS | `141.82` | — | **`118.26 s`** | — | Native prefill MoE v3, Sage2 v4, OTF SWA/Global, DBO SP (`GMU=0.72`) |
| `C32-W4` (`s8_p9`) | PASS | `142.42` | `458.34` | — | — | `rdma_ar_ext_c32` `BF12-G128-OVF` + `c32-moe` + `L2PF_C32` |
| `C32-W9` (`best.env` v1) | PASS | `143.6` (`GMU=0.72`) | `508.2` median (`511.3` paired) | `121.8 s` | `2× 1M` (3rd OOM) | `fused_sconv_ar_c32_exact` (372.3M `nd==0`) + `K=3, tau=0.30` + `GMU 0.78→0.72` (`-4.8 s` 1M) |
| `EXP-5A` (v2 C=32 stack) | PASS | `140.3` (true 0-ULP math) | `517.31` median (`526.5` peak) | `126.8 s` (limbo thrash) | `2× 1M` (`+1.1 GiB`) | `RDMA_AR_LEVEL_DB=1` (`+4.45 tok/s`), `_is_full_decode_capturing()` fixing `MIN_M=32`, BF12 gate L2PF fix |
| `EXP-6` (v3 RHD + Compact Shadow K) | PASS (4/4) | `140.25` | `517.37` median (**`528.07` peak**) | `124.49 s` (9/9 runs ±0.2%) | **`4× 1M`** (`21.3 GiB` free) | Added Compact Shadow K (`sage2_v5`, `-5.2..8.5 GiB`), `MIN_M=17`, `BF11_C32` (`M<=64`), `rhd_int8` RS, DBO interleave + depth-2 limbo cap |
| **`EXP-6B` (Promoted `best.env`)** | **PASS (3/3)** | **`139.51`** | **`518.71` median** (`514.2–524.7`, 0 outliers) | **`120.90 / 121.44 s`** | **`4× 1M`** | Kept Compact Shadow K + `v4` W2 persistent (`P=2`) + full C=32 stack on clean NCCL BF16 Ring |

---

## 3. Hard-Won Operational Rules (GB10 / SM121a)

1. **Always compile SM121a kernels with `-gencode arch=compute_121a,code=sm_121a`** (not `-arch=sm_121a`). On CUDA 13.0, `-arch=sm_121a` also emits virtual `compute_121`, doubling compile time and rejecting `cvt.e2m1x2` PTX instructions.
2. **Always set `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` before `import torch`** in standalone 0-ULP tests (`vllm serve` uses CUDA 13.0 `ptxas`, whereas standalone Python defaults to Triton's bundled 13.3 `ptxas-blackwell`).
3. **Always run 3 passes of `exact_ink.py` on a fresh boot** (passes 1–2 can warm Triton/CUDAGraphs; pass 3+ must converge to `1dcf…f8c3`). A *stable* non-`1dcf` digest on a warm server is a real 0-ULP failure.
4. **Never run `nvcc` or standalone GPU tests on any sibling node while benchmarking.** A single sibling build adds `+25 to +56 s` to 1M prefill wall time.
5. **Measure 1M cold prefill warm** and compare small (`<1.5%`) C=1/C=32 deltas only on paired same-boot or `n >= 6` medians.
6. **Guard `PF_GLOBAL_ZERO_SYNC=1` with bounded inflight DBO depth (`INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS=2`):** Unbounded Python enqueue across 48 DBO layers expands cross-stream `record_stream` limbo from `2.72 GB → 21.75 GB`, causing severe UMA allocator thrash.

---

## 4. Next-Session Optimization Roadmap (If Continuing)

1. **C=1 Target MoE (`58.6%` of C=1 step = `14.04 ms`) & Native SM121a Drafter (`1.55 ms`):**
   - `router_bf12_v3` (0 spills, `nd==0`), `nvfp4_ws5_v3` (`nd==0`), and `mxfp8_mx7_v3` (`nd==0`) are already standalone-verified on GB10; fixing the remaining tiny `BN=32` 4–12 B spill + `cvta.to` PTX constraint in `drafter_i4_sm121.cu` targets **`-1.6 to -2.9 ms/step`** (`150–165 tok/s` C=1 bit-exact).
2. **C=32 Tail Drain (`M=24..48`) `FUSE_COMBINE=2`:**
   - Standalone GB10 testing showed panel-major `FUSE_COMBINE=2` saves **`-5.67 to -10.46 ms/step` with literal `nd==0`** at `M=24..48` (`B=12..16`), while `M >= 60` stays on unfused `FUSE_COMBINE=0`. Gating `FUSE_COMBINE=2` strictly to `M <= 48` can lift aggregate C=32 throughput toward **`535–550 tok/s`**.
3. **1M Prefill `<110 s` + `7×` Multi-1M Residency:**
   - Now that Compact Shadow K (`sage2_v5`) frees `8.55 GiB/rank` at `GMU=0.78`, re-testing `GMU=0.78` with Compact Shadow K active should unlock **`6×–7×` simultaneous 1M sessions** without hitting the PyTorch memory-fraction OOM wall that previously forced `GMU=0.72`.
   - Integrating `rhd_int8` RS (`1.09×` standalone over NCCL BF16 Ring) cleanly inside DBO without losing compute overlap remains a promising `-3 to -6 s` collective target.

---

## 5. Historical Note (2026-11-16 — 4-Spark TP4 Bring-Up)

The earlier 4-Spark TP4 bring-up (`launch/inkling2-tp4.sh`, `repro/gb10-livelock-repro.py`, `nvidia-bug-report.md`) hit a GB10 CUDA driver livelock near `94.5 GiB` per node when fitting the ~380 GB mixed NVFP4/MXFP8 checkpoint on 4× 121.6 GiB Sparks. Moving to **8× DGX Spark (TP8)** halves per-rank weight residency (~56 GiB consumed per node including buffers & graphs), completely avoiding the high-UMA livelock regime and leaving `17–23 GiB/node` free headroom alongside 4× simultaneous 1M contexts.
