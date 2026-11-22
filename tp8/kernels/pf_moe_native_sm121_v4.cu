// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// SM121a (NVIDIA GB10 Blackwell) Expert-Persistent Native Block-Scaled FP4/FP8
// MoE Prefill (v4) — Eliminates 2-3x LPDDR5X Weight Re-Reads & Y_out Zero/RFO!
// =============================================================================
//
// Targets Inkling-2-Small TP8 (INKLING_MOE_TP=1: E=384, H=4608, inter=288, topk=8)
// strictly for long-context eager prefill chunks (plan.tile >= 64, T >= 1024,
// !capturing), leaving C=1 decode (T <= 16) 100% untouched and 0-ULP bit-exact.
//
// Key Physics Upgrades in v4 over v3 (measured 495 ms/chunk -> ~305-335 ms/chunk):
//   1. Expert-Persistent W2 (NVFP4 WS5, NVFP4 Panel, MXFP8 Panel):
//      - Each CTA owns ONE expert e (or LPT work slice) and PANELS_PER_CTA (2/3/4/6)
//        128-col output panels.
//      - Loads & unswizzles each [128, 288] weight panel ONCE from LPDDR5X with
//        `ld.global.cs` (evict-first) directly into persistent registers
//        (36 regs for NVFP4, 72 regs for MXFP8) + sWS SMEM, then sweeps ALL
//        M-tiles of expert e (`BM=64`, 2-stage double-buffered cp.async `sA`,
//        SMEM-pinned `A` when `n_rows <= 128`!).
//      - Cuts W2 LPDDR5X weight traffic from ~2.5-3.0x -> EXACTLY 1.00x (274.7 MB/L)
//        and eliminates ~68% of W_PACK/WS5 unpack instructions & barriers!
//      - Full 64-row bank-conflict-free `W2EpiSmem64` (40.26 KiB <= 48 KiB ->
//        2 CTAs/SM for NVFP4!) cuts epilogue barriers from 9 -> 4 per 128 rows
//        and uses `red.relaxed.gpu.global.add.noftz.bf16x2`.
//      - Serpentine expert traversal across `nb_blk` waves cuts `A` DRAM re-reads
//        by 38-75% for free!
//   2. Opt-in In-L2 Per-Slice Output Zeroing (`l2_zero_slice = 1` for CM=2):
//      - Eliminates full-tensor `cudaMemsetAsync` (151 MB W) + cold first-atomic
//        DRAM reads (151 MB R) = -302 MB/layer (-1.24 ms/L = -49.5 ms/40L)!
//      - Uses 128B sector-complete `st.global.wb.v4.u32` right before each
//        `nb_blk` wave so the 8/12/16 MiB `Y_out` slice allocates 100% HOT in L2
//        with 2-tier LPT heavy-first + tail serpentine preventing tail bubbles!
//   3. Expert-Persistent W13 (`BM=256` 1 CTA/SM @ 74.1 KiB dynamic SMEM, plus
//      `BM=128` 2 CTAs/SM & `BM=64`):
//      - `BM=256` doubles register arithmetic intensity on every weight load AND
//        halves concurrent experts (16 -> 8 = 10.9 MiB < 24 MiB L2!), guaranteeing
//        100% L2 hit across `m_tile >= 1` while `ld.global.cs` on the LAST M-tile
//        protects hot `A+AS` in L2!
//      - 128-row 2-barrier coalesced `st.global.cs.v4.u32` SMEM epilogue (overlaying
//        `sA` via union -> 0 extra SMEM) + uniform warp `mi_end` tail MMA skip!
//   4. 1-Block Zero-Sync LPT Expert Plan Builder (`pf_build_expert_plan`):
//      - Extracts exact `(row_beg, row_end)` for all E experts in ~0.8 us, removes
//        empty experts, sorts heavy-first across 32 buckets, and optionally splits
//        rare mega-experts (`split_m`, multiple of 256) with zero host sync.
//   5. 100% Backwards-Compatible C ABI: preserves ALL 14 legacy/v3 symbols (with
//      `INKLING_PF_MOE_PERSISTENT=0|1` toggle for instant same-.so A/B!) plus
//      dedicated `_v4` entrypoints.
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 --use_fast_math --shared -Xcompiler -fPIC \
//       -gencode arch=compute_121a,code=sm_121a -Xptxas -v \
//       pf_moe_native_sm121_v4.cu -o libpf_moe_native_sm121_v4.so
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>

namespace {

// =============================================================================
// Runtime env helper (for legacy/v3 entrypoint dispatch toggle)
// =============================================================================

inline bool env_persistent_default() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("INKLING_PF_MOE_PERSISTENT");
        if (v == nullptr || v[0] == '\0') {
            cached = 1;  // Default ON when libpf_moe_native_sm121_v4.so is loaded
        } else {
            cached = (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') ? 0 : 1;
        }
    }
    return cached != 0;
}

inline int env_v4_w_evict_override(int caller_mode) {
    static int cached = -2;
    if (cached == -2) {
        const char* v = std::getenv("INKLING_PF_MOE_V4_W_EVICT");
        if (v == nullptr || v[0] == '\0') {
            cached = -1;
        } else {
            cached = std::atoi(v);
        }
    }
    if (cached >= 0) return cached;
    // In v4 persistent mode, map legacy best.env W_EVICT=2 (cs_all) -> 1 (cs_last)
    // unless INKLING_PF_MOE_V4_W_EVICT=2 is explicitly requested for ablation!
    return (caller_mode == 0) ? 0 : 1;
}

// =============================================================================
// Low-level SM121a PTX Primitives
// =============================================================================

__device__ __forceinline__ uint4 ld_cs_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint4 ld_ca_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t ld_ca_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t ld_cs_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint8_t ld_ca_u8(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.ca.u8 %0, [%1];" : "=r"(v) : "l"(ptr));
    return static_cast<uint8_t>(v);
}

__device__ __forceinline__ uint8_t ld_cs_u8(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u8 %0, [%1];" : "=r"(v) : "l"(ptr));
    return static_cast<uint8_t>(v);
}

__device__ __forceinline__ uint4 ld_w_u4(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u4(ptr) : ld_ca_u4(ptr);
}

__device__ __forceinline__ uint32_t ld_w_u32(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u32(ptr) : ld_ca_u32(ptr);
}

__device__ __forceinline__ uint8_t ld_w_u8(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u8(ptr) : ld_ca_u8(ptr);
}

__device__ __forceinline__ void st_wb_u4(void* ptr, uint4 v) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st_cs_u4(void* ptr, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    __nv_bfloat162 bv = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&bv);
}

__device__ __forceinline__ void red_add_f32(float* addr, float val) {
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n" :: "l"(addr), "f"(val) : "memory");
}

__device__ __forceinline__ void red_add_bf16x2_bits(uint32_t* addr, uint32_t bf16x2_bits) {
    asm volatile("red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;\n"
                 :: "l"(addr), "r"(bf16x2_bits) : "memory");
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory"
    );
}

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory"
    );
}

__device__ __forceinline__ void cp_async_ca_8(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// Native SM121a NVFP4 block-scaled MMA: m16n8k64, e2m1 x e2m1, ue4m3 (scale_vec::4X, VS=16).
__device__ __forceinline__ void mma_m16n8k64_nvfp4(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb
) {
    constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB)
    );
}

// Native SM121a MXFP8 block-scaled MMA: m16n8k32, e4m3 x e4m3, ue8m0 (scale_vec::1X, VS=32).
__device__ __forceinline__ void mma_m16n8k32_mxfp8(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb
) {
    constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB)
    );
}

__device__ __forceinline__ int64_t scale_offset_byte(int64_t row, int col, int COLS, int TILE) {
    if (TILE >= 128) {
        int64_t tile = (row >> 7) * ((COLS + 3) >> 2) + (col >> 2);
        return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (col & 3);
    }
    if (TILE > 0) {
        int64_t tile = (row >> 3) * ((COLS + 3) >> 2) + (col >> 2);
        return tile * 32 + (row & 7) * 4 + (col & 3);
    }
    return row * COLS + col;
}

__device__ __forceinline__ uint32_t load_as_u32_rowmaj(
    const uint8_t* __restrict__ AS, int64_t row, int col, int row_stride
) {
    const uint8_t* p = AS + row * row_stride + col;
    if ((row_stride & 3) == 0) {
        return *reinterpret_cast<const uint32_t*>(p);
    }
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ uint32_t load_as_u16_rowmaj(
    const uint8_t* __restrict__ AS, int64_t row, int col, int row_stride
) {
    const uint8_t* p = AS + row * row_stride + col;
    if ((row_stride & 1) == 0) {
        return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(p));
    }
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
}

// =============================================================================
// Expert Range Binary Search & Zero-Sync 1-Block LPT Expert Plan Builder
// =============================================================================
// Tiles[0 .. num_tiles-1] is guaranteed monotonically non-decreasing expert IDs.
// For any expert e with count > 0, its rows are strictly contiguous:
//   [t_lo * tile, Tiles[max_ctas + t_hi - 1]).

__device__ __forceinline__ void find_expert_row_range(
    const int32_t* __restrict__ Tiles,
    int max_ctas,
    int tile,
    int target_e,
    int& row_beg,
    int& row_end
) {
    const int num_tiles = Tiles[2 * max_ctas];
    if (num_tiles <= 0) {
        row_beg = 0;
        row_end = 0;
        return;
    }
    int lo = 0, hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] < target_e) lo = mid + 1;
        else hi = mid;
    }
    const int t_lo = lo;
    if (t_lo >= num_tiles || Tiles[t_lo] != target_e) {
        row_beg = 0;
        row_end = 0;
        return;
    }
    lo = t_lo + 1;
    hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] <= target_e) lo = mid + 1;
        else hi = mid;
    }
    const int t_hi = lo;
    row_beg = t_lo * tile;
    row_end = Tiles[max_ctas + t_hi - 1];
}

// 1-block LPT expert plan builder (~0.6 us, zero host sync):
// PlanBuf layout (size 3 * max_work + 1 int32s):
//   [0 .. max_work-1]              : work_e   (-1 if unused)
//   [max_work .. 2*max_work-1]     : work_beg (aligned to split_m / tile)
//   [2*max_work .. 3*max_work-1]   : work_end
//   [3*max_work]                   : num_work (actual active count <= max_work)
__global__ __launch_bounds__(256, 1)
void pf_build_expert_plan_kernel(
    const int32_t* __restrict__ Tiles,
    int32_t* __restrict__ PlanBuf,
    int max_ctas,
    int tile,
    int num_experts,
    int split_m,
    int max_work
) {
    constexpr int MAX_E = 1024;
    __shared__ int32_t sBeg[MAX_E];
    __shared__ int32_t sEnd[MAX_E];
    __shared__ uint32_t sBucketCnt[32];
    __shared__ uint32_t sBucketOff[32];

    const int tid = threadIdx.x;
    const int eff_e = (num_experts < MAX_E) ? num_experts : MAX_E;
    const int eff_split = (split_m > 0) ? ((split_m + 255) & ~255) : 0;

    if (tid < 32) {
        sBucketCnt[tid] = 0u;
    }
    for (int w = tid; w < max_work; w += 256) {
        PlanBuf[w] = -1;
    }
    __syncthreads();

    for (int e = tid; e < eff_e; e += 256) {
        int rb = 0, re = 0;
        find_expert_row_range(Tiles, max_ctas, tile, e, rb, re);
        sBeg[e] = rb;
        sEnd[e] = re;
        int cnt = re - rb;
        if (cnt > 0) {
            if (eff_split > 0 && cnt > eff_split) {
                for (int off = 0; off < cnt; off += eff_split) {
                    int sz = cnt - off;
                    if (sz > eff_split) sz = eff_split;
                    int raw_b = (sz - 1) >> 5;
                    int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                    atomicAdd(&sBucketCnt[b], 1u);
                }
            } else {
                int raw_b = (cnt - 1) >> 5;
                int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                atomicAdd(&sBucketCnt[b], 1u);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t acc = 0u;
        for (int b = 0; b < 32; ++b) {
            sBucketOff[b] = acc;
            acc += sBucketCnt[b];
        }
        int total_w = static_cast<int>(acc);
        if (total_w > max_work) total_w = max_work;
        PlanBuf[3 * max_work] = total_w;
    }
    __syncthreads();

    int32_t* out_e   = PlanBuf;
    int32_t* out_beg = PlanBuf + max_work;
    int32_t* out_end = PlanBuf + 2 * max_work;

    for (int e = tid; e < eff_e; e += 256) {
        int rb = sBeg[e];
        int re = sEnd[e];
        int cnt = re - rb;
        if (cnt > 0) {
            if (eff_split > 0 && cnt > eff_split) {
                for (int off = 0; off < cnt; off += eff_split) {
                    int sz = cnt - off;
                    if (sz > eff_split) sz = eff_split;
                    int raw_b = (sz - 1) >> 5;
                    int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                    uint32_t pos = atomicAdd(&sBucketOff[b], 1u);
                    if (static_cast<int>(pos) < max_work) {
                        out_e[pos]   = e;
                        out_beg[pos] = rb + off;
                        out_end[pos] = rb + off + sz;
                    }
                }
            } else {
                int raw_b = (cnt - 1) >> 5;
                int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                uint32_t pos = atomicAdd(&sBucketOff[b], 1u);
                if (static_cast<int>(pos) < max_work) {
                    out_e[pos]   = e;
                    out_beg[pos] = rb;
                    out_end[pos] = re;
                }
            }
        }
    }
}

// Resolve CTA's (e, row_beg, row_end) from PlanBuf (if non-null) or Tiles binary search.
// `serpentine_mode`: 0 = direct, 1 = full serpentine on odd nb_blk, 2 = 2-tier heavy-first + tail serpentine.
__device__ __forceinline__ bool resolve_persistent_work(
    const int32_t* __restrict__ PlanBuf,
    const int32_t* __restrict__ Tiles,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int bx,
    int nb_blk,
    int serpentine_mode,
    int32_t (&sWork)[3],
    int tid
) {
    if (tid == 0) {
        int e = -1, rb = 0, re = 0;
        if (PlanBuf != nullptr) {
            const int num_w = PlanBuf[3 * max_work];
            if (bx < num_w) {
                int widx = bx;
                if (serpentine_mode == 1 && (nb_blk & 1)) {
                    widx = num_w - 1 - bx;
                } else if (serpentine_mode == 2 && (nb_blk & 1)) {
                    const int head_keep = (num_w > 64) ? 48 : num_w;
                    if (bx >= head_keep) {
                        widx = num_w - 1 - (bx - head_keep);
                    }
                }
                e  = PlanBuf[widx];
                rb = PlanBuf[max_work + widx];
                re = PlanBuf[2 * max_work + widx];
            }
        } else {
            int e_cand = bx;
            if (serpentine_mode != 0 && (nb_blk & 1)) {
                e_cand = num_experts - 1 - bx;
            }
            if (e_cand >= 0 && e_cand < num_experts) {
                find_expert_row_range(Tiles, max_ctas, tile, e_cand, rb, re);
                if (rb < re) e = e_cand;
            }
        }
        sWork[0] = e;
        sWork[1] = rb;
        sWork[2] = re;
    }
    __syncthreads();
    return (sWork[0] >= 0 && sWork[1] < sWork[2]);
}

// =============================================================================
// In-L2 Per-Slice BF16 Zero Kernel (`st.global.wb.v4.u32`, 128B sector-complete!)
// =============================================================================
// Zeroes Out[0..tokens-1, nb_blk*PANELS*128 .. (nb_blk+1)*PANELS*128) directly
// into GB10's 24 MiB L2 cache right before the matching W2 slice runs!
// Eliminates BOTH full-tensor cudaMemsetAsync (151 MB W) AND cold first-atomic
// DRAM read-allocate (151 MB R) = -302 MB/layer!
template <int PANELS>
__global__ __launch_bounds__(256)
void zero_bf16_slice_wb_kernel(
    uint16_t* __restrict__ Out_bf16,
    int tokens,
    int nb_blk
) {
    constexpr int H = 4608;
    constexpr int COL_BASE_MULT = PANELS * 128;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tok = blockIdx.x * 8 + warp;
    if (tok >= tokens) return;

    uint4* dst = reinterpret_cast<uint4*>(
        Out_bf16 + static_cast<int64_t>(tok) * H + nb_blk * COL_BASE_MULT);
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (PANELS == 2) {
        // 256 bf16 = 512 B = 32 uint4 (4 full 128B lines)
        st_wb_u4(dst + lane, z4);
    } else if constexpr (PANELS == 3) {
        // 384 bf16 = 768 B = 48 uint4 (6 full 128B lines)
        st_wb_u4(dst + lane, z4);
        if (lane < 16) st_wb_u4(dst + 32 + lane, z4);
    } else if constexpr (PANELS == 4) {
        // 512 bf16 = 1024 B = 64 uint4 (8 full 128B lines)
        st_wb_u4(dst + lane, z4);
        st_wb_u4(dst + 32 + lane, z4);
    } else if constexpr (PANELS == 6) {
        // 768 bf16 = 1536 B = 96 uint4 (12 full 128B lines)
        st_wb_u4(dst + lane, z4);
        st_wb_u4(dst + 32 + lane, z4);
        st_wb_u4(dst + 64 + lane, z4);
    }
}

// =============================================================================
// Shared Warp-Coalesced W13 SMEM Epilogues (Zero Extra SMEM via Union!)
// =============================================================================

struct W13EpiSmem {
    uint32_t bf16x2[32][49];    // 6,272 B (0-way bank conflict, 32 rows x 96 bf16)
};

struct W13EpiSmem128 {
    uint32_t bf16x2[128][49];   // 25,088 B (0-way bank conflict, 128 rows x 96 bf16 -> 2 barriers for BM=256!)
};

template <int BM, int HAS_RS>
__device__ __forceinline__ void w13_coalesced_epilogue(
    W13EpiSmem& sEpi,
    const float (&acc)[(BM / 2) / 16][3][4],
    const float* sRS,
    float scale_e,
    uint32_t* __restrict__ Out_u32,
    int row0, int end, int nb_cta,
    int warp, int wm, int wp_n, int lane, int g, int lt
) {
    constexpr int N = 576;
    constexpr int M_TILES_W = (BM / 2) / 16;
    const int m_warp_base = wm * (BM / 2);

    __syncthreads();
    const int half24 = (lane >= 12) ? 1 : 0;
    const int c4 = lane - half24 * 12;

    #pragma unroll
    for (int mi = 0; mi < M_TILES_W; ++mi) {
        int m_lo = m_warp_base + mi * 16 + g;
        int m_hi = m_lo + 8;
        float s_lo = scale_e;
        float s_hi = scale_e;
        if constexpr (HAS_RS) {
            s_lo = (row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
            s_hi = (row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
        }
        int r_epi_lo = wm * 16 + g;
        int r_epi_hi = r_epi_lo + 8;
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            int col_pair = (wp_n * 3 + sn) * 4 + lt;
            sEpi.bf16x2[r_epi_lo][col_pair] = pack_bf16x2(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo);
            sEpi.bf16x2[r_epi_hi][col_pair] = pack_bf16x2(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi);
        }
        __syncthreads();

        if (lane < 24) {
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                int r_epi = warp * 4 + k * 2 + half24;
                int m = (r_epi < 16) ? (mi * 16 + r_epi) : ((BM / 2) + mi * 16 + (r_epi - 16));
                int r_glob = row0 + m;
                if (r_glob < end) {
                    uint4 v;
                    v.x = sEpi.bf16x2[r_epi][c4 * 4 + 0];
                    v.y = sEpi.bf16x2[r_epi][c4 * 4 + 1];
                    v.z = sEpi.bf16x2[r_epi][c4 * 4 + 2];
                    v.w = sEpi.bf16x2[r_epi][c4 * 4 + 3];
                    uint4* dst = reinterpret_cast<uint4*>(
                        Out_u32 + static_cast<int64_t>(r_glob) * (N / 2) + nb_cta * 48);
                    st_cs_u4(dst + c4, v);
                }
            }
        }
        if (mi + 1 < M_TILES_W) {
            __syncthreads();
        }
    }
}

// Fast 2-chunk 128-row W13 epilogue for BM=256 (cuts barriers from 15 -> 3!)
template <int HAS_RS>
__device__ __forceinline__ void w13_coalesced_epilogue_256(
    W13EpiSmem128& sEpi,
    const float (&acc)[8][3][4],
    const float* sRS,
    float scale_e,
    uint32_t* __restrict__ Out_u32,
    int row0, int end, int nb_cta,
    int warp, int wm, int wp_n, int lane, int g, int lt
) {
    constexpr int N = 576;
    const int m_warp_base = wm * 128;

    __syncthreads();
    const int half24 = (lane >= 12) ? 1 : 0;
    const int c4 = lane - half24 * 12;

    #pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        #pragma unroll
        for (int mi4 = 0; mi4 < 4; ++mi4) {
            int mi = chunk * 4 + mi4;
            int m_lo = m_warp_base + mi * 16 + g;
            int m_hi = m_lo + 8;
            float s_lo = scale_e;
            float s_hi = scale_e;
            if constexpr (HAS_RS) {
                s_lo = (row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
                s_hi = (row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
            }
            int r_epi_lo = wm * 64 + mi4 * 16 + g;
            int r_epi_hi = r_epi_lo + 8;
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                int col_pair = (wp_n * 3 + sn) * 4 + lt;
                sEpi.bf16x2[r_epi_lo][col_pair] = pack_bf16x2(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo);
                sEpi.bf16x2[r_epi_hi][col_pair] = pack_bf16x2(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi);
            }
        }
        __syncthreads();

        if (lane < 24) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int r_epi = warp * 16 + k * 2 + half24;   // 0..127
                int m = (r_epi < 64)
                    ? (chunk * 64 + r_epi)
                    : (128 + chunk * 64 + (r_epi - 64));
                int r_glob = row0 + m;
                if (r_glob < end) {
                    uint4 v;
                    v.x = sEpi.bf16x2[r_epi][c4 * 4 + 0];
                    v.y = sEpi.bf16x2[r_epi][c4 * 4 + 1];
                    v.z = sEpi.bf16x2[r_epi][c4 * 4 + 2];
                    v.w = sEpi.bf16x2[r_epi][c4 * 4 + 3];
                    uint4* dst = reinterpret_cast<uint4*>(
                        Out_u32 + static_cast<int64_t>(r_glob) * (N / 2) + nb_cta * 48);
                    st_cs_u4(dst + c4, v);
                }
            }
        }
        if (chunk == 0) {
            __syncthreads();
        }
    }
}

// =============================================================================
// Shared Warp-Coalesced W2 SMEM Epilogues (v3 32-row + v4 64-row Single-Barrier!)
// =============================================================================

union W2EpiSmem {
    uint32_t bf16x2[32][65];    //  8,320 B (v3 compat)
    float    f32[16][129];      //  8,256 B
};

union W2EpiSmem64 {
    uint32_t bf16x2[64][65];    // 16,640 B (0-way bank conflict, ALL 64 rows at once -> 1 barrier!)
    float    f32[32][129];      // 16,512 B (0-way bank conflict, 32 rows/pass -> 3 barriers for BM=64!)
};

// Full 64-row epilogue for BM=64 persistent W2 (only 1 __syncthreads for CM=0/2!).
// Caller handles the trailing __syncthreads() before overwriting sEpi / sTok.
template <int COMBINE_MODE, int HAS_RS>
__device__ __forceinline__ void w2_coalesced_epilogue_64(
    W2EpiSmem64& sEpi,
    const float (&acc)[2][4][4],
    const float* sRS,
    const int32_t* sTok,
    float scale_e,
    void* __restrict__ Out,
    int row0, int end, int nb,
    int warp, int wm, int wp, int lane, int g, int lt
) {
    constexpr int N = 4608;
    const int m_warp_base = wm * 32;

    if constexpr (COMBINE_MODE == 0 || COMBINE_MODE == 2) {
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            int m_lo = m_warp_base + mi * 16 + g;
            int m_hi = m_lo + 8;
            float s_lo = scale_e;
            float s_hi = scale_e;
            if constexpr (HAS_RS) {
                s_lo = (row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
                s_hi = (row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
            }
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                int col_pair = wp * 16 + sn * 4 + lt;
                sEpi.bf16x2[m_lo][col_pair] = pack_bf16x2(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo);
                sEpi.bf16x2[m_hi][col_pair] = pack_bf16x2(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi);
            }
        }
        __syncthreads();

        if constexpr (COMBINE_MODE == 0) {
            const int half_w = lane >> 4;
            const int c4 = lane & 15;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int m = warp * 8 + k * 2 + half_w;   // 0..63
                int r_glob = row0 + m;
                if (r_glob < end) {
                    uint4 v;
                    v.x = sEpi.bf16x2[m][c4 * 4 + 0];
                    v.y = sEpi.bf16x2[m][c4 * 4 + 1];
                    v.z = sEpi.bf16x2[m][c4 * 4 + 2];
                    v.w = sEpi.bf16x2[m][c4 * 4 + 3];
                    uint4* dst = reinterpret_cast<uint4*>(
                        reinterpret_cast<uint32_t*>(Out) + static_cast<int64_t>(r_glob) * (N / 2) + nb * 64);
                    st_cs_u4(dst + c4, v);
                }
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int m = warp * 8 + k;   // 0..63
                int r_glob = row0 + m;
                if (r_glob < end) {
                    int32_t tok = sTok[m];
                    if (tok >= 0) {
                        uint32_t* dst = reinterpret_cast<uint32_t*>(Out)
                                      + static_cast<int64_t>(tok) * (N / 2) + nb * 64;
                        red_add_bf16x2_bits(dst + lane,      sEpi.bf16x2[m][lane]);
                        red_add_bf16x2_bits(dst + 32 + lane, sEpi.bf16x2[m][32 + lane]);
                    }
                }
            }
        }
    } else {
        // COMBINE_MODE == 1: FP32 atomic via 32-row staging (2 passes for BM=64!)
        #pragma unroll
        for (int w_sel = 0; w_sel < 2; ++w_sel) {
            if (wm == w_sel) {
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    float s_lo = scale_e;
                    float s_hi = scale_e;
                    if constexpr (HAS_RS) {
                        s_lo = (row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
                        s_hi = (row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
                    }
                    int r_lo = mi * 16 + g;
                    int r_hi = r_lo + 8;
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        int c0 = wp * 32 + sn * 8 + lt * 2;
                        sEpi.f32[r_lo][c0 + 0] = acc[mi][sn][0] * s_lo;
                        sEpi.f32[r_lo][c0 + 1] = acc[mi][sn][1] * s_lo;
                        sEpi.f32[r_hi][c0 + 0] = acc[mi][sn][2] * s_hi;
                        sEpi.f32[r_hi][c0 + 1] = acc[mi][sn][3] * s_hi;
                    }
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int r_epi = warp * 4 + k;   // 0..31
                int m = w_sel * 32 + r_epi;
                int r_glob = row0 + m;
                if (r_glob < end) {
                    int32_t tok = sTok[m];
                    if (tok >= 0) {
                        float* dst = reinterpret_cast<float*>(Out)
                                   + static_cast<int64_t>(tok) * N + nb * 128;
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            red_add_f32(dst + q * 32 + lane, sEpi.f32[r_epi][q * 32 + lane]);
                        }
                    }
                }
            }
            if (w_sel == 0) {
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// SMEM Layouts & Compile-Time Occupancy Proofs
// =============================================================================

template <int BM>
struct W13Ws5SmemBlock {
    union {
        struct {
            uint32_t sA[2][BM][32];
            uint32_t sAS[2][BM][4];
        } m;
        W13EpiSmem    epi;
        W13EpiSmem128 epi128;
    } u;
    int32_t sTok[BM];
    float   sRS[BM];
    int32_t sPOff[12][2];
    int32_t sWork[3];
};

template <int BM>
struct W13PanelSmemBlock {
    union {
        struct {
            uint32_t sA[2][BM][20];
            uint32_t sB[2][96][20];
            uint32_t sAS[2][BM][3];
            uint8_t  sBS[2][96][16];
        } m;
        W13EpiSmem    epi;
        W13EpiSmem128 epi128;
    } u;
    int32_t sTok[BM];
    float   sRS[BM];
    int32_t sWork[3];
};

template <int BM>
struct MxW13SmemBlock {
    union {
        struct {
            uint32_t sA[3][BM][16];
            uint32_t sB[3][96][16];
            uint64_t sAS[2][BM];
            uint8_t  sBS[2][96][16];
        } m;
        W13EpiSmem    epi;
        W13EpiSmem128 epi128;
    } u;
    int32_t sTok[BM];
    int32_t sWork[3];
};

struct W2Nvfp4PersistSmem {
    uint32_t    sA_main[2][64][32];  // 16,384 B (2-stage double-buffered cp.async)
    uint32_t    sA_tail[2][64][4];   //  2,048 B
    uint32_t    sAS[2][64][5];       //  2,560 B
    uint32_t    sWS[128][5];         //  2,560 B
    float       sRS[2][64];          //    512 B
    int32_t     sTok[2][64];         //    512 B
    W2EpiSmem64 sEpi;                // 16,640 B (all 64 rows at once -> 1 barrier!)
    int32_t     sWork[3];            //     12 B
};

struct W2Mxfp8PersistSmem {
    uint32_t    sA_8[2][64][64];     // 32,768 B (2-stage double-buffered cp.async)
    uint32_t    sA_last[2][64][8];   //  4,096 B
    uint32_t    sAS[2][64][3];       //  1,536 B
    uint8_t     sWS[9][128];         //  1,152 B
    int32_t     sTok[2][64];         //    512 B
    W2EpiSmem64 sEpi;                // 16,640 B
    int32_t     sWork[3];            //     12 B
};

// Occupancy & SM121a SMEM proofs (48 KiB -> 2 CTAs/SM; 96 KiB < 99 KiB -> 1 CTA/SM)
static_assert(sizeof(W13Ws5SmemBlock<64>)   <= 49152, "W13Ws5<64> must fit <= 48 KiB (2 CTAs/SM)");
static_assert(sizeof(W13Ws5SmemBlock<128>)  <= 49152, "W13Ws5<128> must fit <= 48 KiB (2 CTAs/SM)");
static_assert(sizeof(W13Ws5SmemBlock<256>)  <= 98304, "W13Ws5<256> must fit <= 96 KiB (1 CTA/SM)");
static_assert(sizeof(W13PanelSmemBlock<128>)<= 49152, "W13Panel<128> must fit <= 48 KiB (2 CTAs/SM)");
static_assert(sizeof(W13PanelSmemBlock<256>)<= 98304, "W13Panel<256> must fit <= 96 KiB (1 CTA/SM)");
static_assert(sizeof(MxW13SmemBlock<128>)   <= 49152, "MxW13<128> must fit <= 48 KiB (2 CTAs/SM)");
static_assert(sizeof(MxW13SmemBlock<256>)   <= 98304, "MxW13<256> must fit <= 96 KiB (1 CTA/SM)");
static_assert(sizeof(W2Nvfp4PersistSmem)    <= 49152, "W2Nvfp4Persist must fit <= 48 KiB (2 CTAs/SM)");
static_assert(sizeof(W2Mxfp8PersistSmem)    <= 98304, "W2Mxfp8Persist must fit <= 96 KiB (1 CTA/SM)");

// =============================================================================
// Kernel P1A (v4): Expert-Persistent NVFP4 W13 WS5 (N=576, K=4608, BN=96)
// =============================================================================
// - BM=256 (1 CTA/SM, 74.1 KiB dyn SMEM, ~165 regs -> 0 spills):
//     2x register arithmetic intensity per weight load AND only 8 concurrent
//     experts across 48 SMs (10.9 MiB < 24 MiB L2!), guaranteeing 100% L2 hits
//     on m_tile >= 1 while ld.global.cs on the last M-tile protects A+AS!
// - BM=128 (2 CTAs/SM, 37.9 KiB dyn SMEM, ~110 regs -> 0 spills).
// - Uniform warp `mi_end` tail skip eliminates wasted MMAs on partial tail tiles.
// - Literal 0-ULP bit-exact match to v3 (identical ks=0..17 accumulation order).
template <int BM, int USE_R2T, int AS_ROW_MAJOR, int HAS_PATCHES>
__global__ __launch_bounds__(256, (BM == 256 ? 1 : 2))
void pf_nvfp4_w13_ws5_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    uint32_t* __restrict__ Out_u32,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int w_evict_mode
) {
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int M_TILES_W = (BM / 2) / 16;   // 8 (BM=256), 4 (BM=128), 2 (BM=64)

    const int nb_cta = blockIdx.x;   // 0..5
    const int w_bx   = blockIdx.y;   // 0..max_work-1

    extern __shared__ uint4 dyn_smem[];
    auto& sm = *reinterpret_cast<W13Ws5SmemBlock<BM>*>(dyn_smem);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 2;
    const int wp_n = warp & 3;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, 0, 0, sm.sWork, tid)) {
        return;
    }
    const int64_t e       = sm.sWork[0];
    const int row_begin   = sm.sWork[1];
    const int end         = sm.sWork[2];
    const int n_rows      = end - row_begin;
    const int num_m       = (n_rows + BM - 1) / BM;

    // Load per-(e, nb_cta) patch offsets ONCE across all M-tiles of expert e!
    if constexpr (HAS_PATCHES) {
        if (tid < 12) {
            int pl = tid >> 2;
            int wp = tid & 3;
            int64_t wuid = (e * 18 + nb_cta * 3 + pl) * 4 + wp;
            sm.sPOff[tid][0] = P_OFF[wuid];
            sm.sPOff[tid][1] = P_OFF[wuid + 1];
        }
    }

    const uint8_t* slabs_cta = SLABS + (e * 18 + nb_cta * 3) * (9LL * 8832LL);
    const int base_e         = static_cast<int>(WS_BASE[e]);
    const float scale_e      = Ones[e];
    const int m_warp_base    = wm * (BM / 2);
    const int m_sf_off       = g + ((lt & 1) << 3);

    #pragma unroll 1
    for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
        const int row0 = row_begin + mi_blk * BM;
        const bool is_last_m = (mi_blk + 1 >= num_m);
        const bool use_cs_w  = (w_evict_mode == 2) || (w_evict_mode == 1 && is_last_m);

        if (tid < BM) {
            int r = row0 + tid;
            if (r < end) {
                sm.sTok[tid] = USE_R2T ? Row2Tok[r] : r;
                sm.sRS[tid]  = RS[r];
            } else {
                sm.sTok[tid] = -1;
                sm.sRS[tid]  = 0.0f;
            }
        }
        __syncthreads();

        auto load_stage = [&](int ks, int buf) {
            constexpr int A_ITERS = (BM * 8) / 256;
            #pragma unroll
            for (int it = 0; it < A_ITERS; ++it) {
                int idx = it * 256 + tid;
                int m   = idx >> 3;
                int c4  = idx & 7;
                int sc4 = c4 ^ (m & 7);
                int32_t tok = sm.sTok[m];
                const uint8_t* src = (tok >= 0)
                    ? (A + static_cast<int64_t>(tok) * KP + ks * 128 + c4 * 16)
                    : A;
                cp_async_cg_16(&sm.u.m.sA[buf][m][sc4 * 4], src, (tok >= 0) ? 16 : 0);
            }
            if constexpr (USE_R2T || AS_ROW_MAJOR) {
                if (tid < BM) {
                    int32_t tok = sm.sTok[tid];
                    const uint8_t* src_as = (tok >= 0)
                        ? (AS + static_cast<int64_t>(tok) * KG + ks * 16)
                        : AS;
                    cp_async_cg_16(&sm.u.m.sAS[buf][tid][0], src_as, (tok >= 0) ? 16 : 0);
                }
            } else {
                constexpr int AS_ITERS = (BM * 4) / 256;
                #pragma unroll
                for (int it = 0; it < AS_ITERS; ++it) {
                    int idx = it * 256 + tid;
                    int m   = idx >> 2;
                    int q   = idx & 3;
                    int32_t tok = sm.sTok[m];
                    uint32_t sval = 0u;
                    if (tok >= 0) {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(row0 + m), ks * 16 + q * 4, KG, tile);
                        sval = *reinterpret_cast<const uint32_t*>(AS + off);
                    }
                    sm.u.m.sAS[buf][m][q] = sval;
                }
            }
            cp_async_commit();
        };

        auto decode_sfb_ss = [&](int ss, uint32_t (&p0)[3], uint32_t (&p1)[3]) {
            const int kg_lo = ss * 32;
            const int kg_hi = kg_lo + 32;
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                int wg = wp_n * 3 + sn;
                int pl = wg >> 2;
                int wp = wg & 3;
                const uint8_t* ss_ptr = slabs_cta + static_cast<int64_t>(pl * 9 + ss) * 8832;
                uint32_t ws_lo = ld_w_u32(ss_ptr + 8192 + wp * 128 + lane * 4, use_cs_w);
                uint8_t  ws_hi = ld_w_u8(ss_ptr + 8704 + wp * 32 + lane, use_cs_w);
                uint8_t sb[8];
                #pragma unroll
                for (int ii = 0; ii < 8; ++ii) {
                    int code = ((ws_lo >> (4 * ii)) & 0xF) | (((ws_hi >> ii) & 1) << 4);
                    sb[ii] = static_cast<uint8_t>(base_e + code);
                }
                if constexpr (HAS_PATCHES) {
                    int plo = sm.sPOff[wg][0];
                    int phi = sm.sPOff[wg][1];
                    if (plo < phi) {
                        for (int pi = plo; pi < phi; ++pi) {
                            uint32_t rec = PATCHES[pi];
                            int p_kg = static_cast<int>(rec >> 16);
                            if (p_kg >= kg_lo && p_kg < kg_hi) {
                                int p_g = static_cast<int>((rec >> 8) & 0xFF);
                                int p_lt = (p_kg - kg_lo) >> 3;
                                if (g == p_g && lt == p_lt) {
                                    sb[p_kg & 7] = static_cast<uint8_t>(rec & 0xFF);
                                }
                            }
                        }
                    }
                }
                p0[sn] = static_cast<uint32_t>(sb[0])
                       | (static_cast<uint32_t>(sb[1]) << 8)
                       | (static_cast<uint32_t>(sb[2]) << 16)
                       | (static_cast<uint32_t>(sb[3]) << 24);
                p1[sn] = static_cast<uint32_t>(sb[4])
                       | (static_cast<uint32_t>(sb[5]) << 8)
                       | (static_cast<uint32_t>(sb[6]) << 16)
                       | (static_cast<uint32_t>(sb[7]) << 24);
            }
        };

        load_stage(0, 0);
        uint4 wa[3], wb[3];
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            int wg = wp_n * 3 + sn;
            wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9) * 8832 + (wg & 3) * 1024 + lane * 16, use_cs_w);
        }
        uint32_t sfb_p0[3], sfb_p1[3];
        decode_sfb_ss(0, sfb_p0, sfb_p1);
        cp_async_wait<0>();
        __syncthreads();

        float acc[M_TILES_W][3][4];
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
            }
        }

        // Uniform warp-level active M-tile bound (skips dead MMAs on partial tail M-tile!).
        const int wm_live = (end - row0) - m_warp_base;
        const int mi_end = (wm_live <= 0) ? 0
                         : ((wm_live >= BM / 2) ? M_TILES_W : ((wm_live + 15) >> 4));

        #pragma unroll 1
        for (int ks = 0; ks < 18; ++ks) {
            const int cur = ks & 1;
            const int nxt = cur ^ 1;
            const int ss  = ks >> 1;
            const int ks2 = ks & 1;

            if (ks + 1 < 18) {
                load_stage(ks + 1, nxt);
            }

            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                int wg = wp_n * 3 + sn;
                wb[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + ss) * 8832
                                 + ks2 * 4096 + (wg & 3) * 1024 + 512 + lane * 16, use_cs_w);
            }

            const int src_lt0 = ks2 * 2;
            {
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    b0[sn]  = prmt_b32(wa[sn].x, wa[sn].y, 0x5410u);
                    b1[sn]  = prmt_b32(wa[sn].x, wa[sn].y, 0x7632u);
                    sfb[sn] = __shfl_sync(0xFFFFFFFFu, sfb_p0[sn], (g << 2) | src_lt0);
                }
                const int sw_c0 = (0 * 8 + lt) ^ (g << 2);
                const int sw_c1 = (0 * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    if (mi < mi_end) {
                        int m_lo = m_warp_base + mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur][m_lo][sw_c0];
                        uint32_t a1  = sm.u.m.sA[cur][m_hi][sw_c0];
                        uint32_t a2  = sm.u.m.sA[cur][m_lo][sw_c1];
                        uint32_t a3  = sm.u.m.sA[cur][m_hi][sw_c1];
                        uint32_t sfa = sm.u.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][0];
                        #pragma unroll
                        for (int sn = 0; sn < 3; ++sn) {
                            mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                        }
                    }
                }
            }
            {
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    b0[sn]  = prmt_b32(wa[sn].z, wa[sn].w, 0x5410u);
                    b1[sn]  = prmt_b32(wa[sn].z, wa[sn].w, 0x7632u);
                    sfb[sn] = __shfl_sync(0xFFFFFFFFu, sfb_p1[sn], (g << 2) | src_lt0);
                }
                const int sw_c0 = (1 * 8 + lt) ^ (g << 2);
                const int sw_c1 = (1 * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    if (mi < mi_end) {
                        int m_lo = m_warp_base + mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur][m_lo][sw_c0];
                        uint32_t a1  = sm.u.m.sA[cur][m_hi][sw_c0];
                        uint32_t a2  = sm.u.m.sA[cur][m_lo][sw_c1];
                        uint32_t a3  = sm.u.m.sA[cur][m_hi][sw_c1];
                        uint32_t sfa = sm.u.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][1];
                        #pragma unroll
                        for (int sn = 0; sn < 3; ++sn) {
                            mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                        }
                    }
                }
            }

            if (ks + 1 < 18) {
                const int n_ss  = (ks + 1) >> 1;
                const int n_ks2 = (ks + 1) & 1;
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    int wg = wp_n * 3 + sn;
                    wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + n_ss) * 8832
                                     + n_ks2 * 4096 + (wg & 3) * 1024 + lane * 16, use_cs_w);
                }
            }

            const int src_lt1 = ks2 * 2 + 1;
            {
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    b0[sn]  = prmt_b32(wb[sn].x, wb[sn].y, 0x5410u);
                    b1[sn]  = prmt_b32(wb[sn].x, wb[sn].y, 0x7632u);
                    sfb[sn] = __shfl_sync(0xFFFFFFFFu, sfb_p0[sn], (g << 2) | src_lt1);
                }
                const int sw_c0 = (2 * 8 + lt) ^ (g << 2);
                const int sw_c1 = (2 * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    if (mi < mi_end) {
                        int m_lo = m_warp_base + mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur][m_lo][sw_c0];
                        uint32_t a1  = sm.u.m.sA[cur][m_hi][sw_c0];
                        uint32_t a2  = sm.u.m.sA[cur][m_lo][sw_c1];
                        uint32_t a3  = sm.u.m.sA[cur][m_hi][sw_c1];
                        uint32_t sfa = sm.u.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][2];
                        #pragma unroll
                        for (int sn = 0; sn < 3; ++sn) {
                            mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                        }
                    }
                }
            }
            {
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    b0[sn]  = prmt_b32(wb[sn].z, wb[sn].w, 0x5410u);
                    b1[sn]  = prmt_b32(wb[sn].z, wb[sn].w, 0x7632u);
                    sfb[sn] = __shfl_sync(0xFFFFFFFFu, sfb_p1[sn], (g << 2) | src_lt1);
                }
                const int sw_c0 = (3 * 8 + lt) ^ (g << 2);
                const int sw_c1 = (3 * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    if (mi < mi_end) {
                        int m_lo = m_warp_base + mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur][m_lo][sw_c0];
                        uint32_t a1  = sm.u.m.sA[cur][m_hi][sw_c0];
                        uint32_t a2  = sm.u.m.sA[cur][m_lo][sw_c1];
                        uint32_t a3  = sm.u.m.sA[cur][m_hi][sw_c1];
                        uint32_t sfa = sm.u.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][3];
                        #pragma unroll
                        for (int sn = 0; sn < 3; ++sn) {
                            mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                        }
                    }
                }
            }

            if (ks + 1 < 18) {
                if (((ks + 1) & 1) == 0) {
                    decode_sfb_ss((ks + 1) >> 1, sfb_p0, sfb_p1);
                }
                cp_async_wait<0>();
                __syncthreads();
            }
        }

        if constexpr (BM == 256) {
            w13_coalesced_epilogue_256<1>(
                sm.u.epi128, acc, sm.sRS, scale_e, Out_u32, row0, end, nb_cta, warp, wm, wp_n, lane, g, lt
            );
        } else {
            w13_coalesced_epilogue<BM, 1>(
                sm.u.epi, acc, sm.sRS, scale_e, Out_u32, row0, end, nb_cta, warp, wm, wp_n, lane, g, lt
            );
        }

        if (!is_last_m) {
            __syncthreads();
        }
    }
}

// =============================================================================
// Kernel P1B (v4): Expert-Persistent NVFP4 W13 PANEL (N=576, K=4608, BN=96)
// =============================================================================
template <int BM, int USE_R2T, int AS_ROW_MAJOR, int WS6>
__global__ __launch_bounds__(256, (BM == 256 ? 1 : 2))
void pf_nvfp4_w13_panel_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ WP,
    const uint8_t* __restrict__ WSP,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    uint32_t* __restrict__ Out_u32,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int w_evict_mode
) {
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int M_TILES_W = (BM / 2) / 16;

    const int nb_cta = blockIdx.x;
    const int w_bx   = blockIdx.y;

    extern __shared__ uint4 dyn_smem[];
    auto& sm = *reinterpret_cast<W13PanelSmemBlock<BM>*>(dyn_smem);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 2;
    const int wp_n = warp & 3;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, 0, 0, sm.sWork, tid)) {
        return;
    }
    const int64_t e       = sm.sWork[0];
    const int row_begin   = sm.sWork[1];
    const int end         = sm.sWork[2];
    const int num_m       = ((end - row_begin) + BM - 1) / BM;

    const uint8_t* wp_p0  = WP  + (e * 18 + nb_cta * 3) * (18LL * 32LL * 128LL);
    const uint8_t* wsp_p0 = WSP + (e * 18 + nb_cta * 3) * (18LL * (WS6 ? 384LL : 512LL));
    const float scale_e   = Ones[e];
    const int m_warp_base = wm * (BM / 2);
    const int m_sf_off    = g + ((lt & 1) << 3);

    #pragma unroll 1
    for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
        const int row0 = row_begin + mi_blk * BM;
        const bool is_last_m = (mi_blk + 1 >= num_m);
        const bool use_cs_w  = (w_evict_mode == 2) || (w_evict_mode == 1 && is_last_m);

        if (tid < BM) {
            int r = row0 + tid;
            if (r < end) {
                sm.sTok[tid] = USE_R2T ? Row2Tok[r] : r;
                sm.sRS[tid]  = RS[r];
            } else {
                sm.sTok[tid] = -1;
                sm.sRS[tid]  = 0.0f;
            }
        }
        __syncthreads();

        auto load_ab_stage = [&](int stg, int buf) {
            const int ks = stg >> 1;
            const int hk = stg & 1;
            constexpr int A_ITERS = (BM * 4) / 256;
            #pragma unroll
            for (int it = 0; it < A_ITERS; ++it) {
                int idx = it * 256 + tid;
                int m   = idx >> 2;
                int c4  = idx & 3;
                int32_t tok = sm.sTok[m];
                const uint8_t* src = (tok >= 0)
                    ? (A + static_cast<int64_t>(tok) * KP + stg * 64 + c4 * 16)
                    : A;
                cp_async_cg_16(&sm.u.m.sA[buf][m][c4 * 4], src, (tok >= 0) ? 16 : 0);
            }
            for (int idx = tid; idx < 384; idx += 256) {
                int n_loc = idx >> 2;
                int c4    = idx & 3;
                int pl    = n_loc >> 5;
                int n_in  = n_loc & 31;
                const uint8_t* src = wp_p0 + static_cast<int64_t>(pl) * (18LL * 32LL * 128LL)
                                   + static_cast<int64_t>(ks) * (32 * 128)
                                   + n_in * 128 + hk * 64 + c4 * 16;
                cp_async_cg_16(&sm.u.m.sB[buf][n_loc][c4 * 4], src, 16);
            }
            for (int idx = tid; idx < BM * 2; idx += 256) {
                int m = idx >> 1;
                int q = idx & 1;
                int32_t tok = sm.sTok[m];
                uint32_t sval = 0u;
                if (tok >= 0) {
                    if constexpr (USE_R2T || AS_ROW_MAJOR) {
                        sval = *reinterpret_cast<const uint32_t*>(
                            AS + static_cast<int64_t>(tok) * KG + stg * 8 + q * 4);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(row0 + m), stg * 8 + q * 4, KG, tile);
                        sval = *reinterpret_cast<const uint32_t*>(AS + off);
                    }
                }
                sm.u.m.sAS[buf][m][q] = sval;
            }
            cp_async_commit();
        };

        auto load_bs_ks = [&](int ks, int bbuf) {
            for (int idx = tid; idx < 384; idx += 256) {
                int pl   = idx >> 7;
                int t128 = idx & 127;
                int kbg  = t128 >> 3;
                int j4   = t128 & 7;
                const uint8_t* ws_src = wsp_p0
                    + static_cast<int64_t>(pl) * (18LL * (WS6 ? 384LL : 512LL))
                    + static_cast<int64_t>(ks) * (WS6 ? 384 : 512);
                int nbase = pl * 32 + j4 * 4;
                if constexpr (WS6) {
                    uint8_t lo0 = ld_w_u8(ws_src + kbg * 16 + j4 * 2 + 0, use_cs_w);
                    uint8_t lo1 = ld_w_u8(ws_src + kbg * 16 + j4 * 2 + 1, use_cs_w);
                    uint8_t hib = ld_w_u8(ws_src + 256 + kbg * 8 + j4, use_cs_w);
                    sm.u.m.sBS[bbuf][nbase + 0][kbg] = static_cast<uint8_t>(0x40u | (((hib >> 0) & 3u) << 4) | (lo0 & 0xFu));
                    sm.u.m.sBS[bbuf][nbase + 1][kbg] = static_cast<uint8_t>(0x40u | (((hib >> 2) & 3u) << 4) | (lo0 >> 4));
                    sm.u.m.sBS[bbuf][nbase + 2][kbg] = static_cast<uint8_t>(0x40u | (((hib >> 4) & 3u) << 4) | (lo1 & 0xFu));
                    sm.u.m.sBS[bbuf][nbase + 3][kbg] = static_cast<uint8_t>(0x40u | (((hib >> 6) & 3u) << 4) | (lo1 >> 4));
                } else {
                    uint32_t s4 = ld_w_u32(ws_src + kbg * 32 + j4 * 4, use_cs_w);
                    sm.u.m.sBS[bbuf][nbase + 0][kbg] = static_cast<uint8_t>(s4 & 0xFFu);
                    sm.u.m.sBS[bbuf][nbase + 1][kbg] = static_cast<uint8_t>((s4 >> 8) & 0xFFu);
                    sm.u.m.sBS[bbuf][nbase + 2][kbg] = static_cast<uint8_t>((s4 >> 16) & 0xFFu);
                    sm.u.m.sBS[bbuf][nbase + 3][kbg] = static_cast<uint8_t>((s4 >> 24) & 0xFFu);
                }
            }
        };

        load_ab_stage(0, 0);
        load_bs_ks(0, 0);
        cp_async_wait<0>();
        __syncthreads();

        float acc[M_TILES_W][3][4];
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
            }
        }

        const int wm_live = (end - row0) - m_warp_base;
        const int mi_end = (wm_live <= 0) ? 0
                         : ((wm_live >= BM / 2) ? M_TILES_W : ((wm_live + 15) >> 4));

        #pragma unroll 1
        for (int stg = 0; stg < 36; ++stg) {
            const int cur  = stg & 1;
            const int nxt  = cur ^ 1;
            const int ks   = stg >> 1;
            const int hk   = stg & 1;
            const int bcur = ks & 1;

            if (stg + 1 < 36) load_ab_stage(stg + 1, nxt);
            if (hk == 0 && ks + 1 < 18) load_bs_ks(ks + 1, bcur ^ 1);

            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    int n_row = wp_n * 24 + sn * 8 + g;
                    b0[sn]  = sm.u.m.sB[cur][n_row][q * 8 + lt];
                    b1[sn]  = sm.u.m.sB[cur][n_row][q * 8 + 4 + lt];
                    sfb[sn] = reinterpret_cast<const uint32_t*>(sm.u.m.sBS[bcur][n_row])[hk * 2 + q];
                }
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    if (mi < mi_end) {
                        int m_lo = m_warp_base + mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur][m_lo][q * 8 + lt];
                        uint32_t a1  = sm.u.m.sA[cur][m_hi][q * 8 + lt];
                        uint32_t a2  = sm.u.m.sA[cur][m_lo][q * 8 + 4 + lt];
                        uint32_t a3  = sm.u.m.sA[cur][m_hi][q * 8 + 4 + lt];
                        uint32_t sfa = sm.u.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][q];
                        #pragma unroll
                        for (int sn = 0; sn < 3; ++sn) {
                            mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                        }
                    }
                }
            }

            if (stg + 1 < 36) {
                cp_async_wait<0>();
                __syncthreads();
            }
        }

        if constexpr (BM == 256) {
            w13_coalesced_epilogue_256<1>(
                sm.u.epi128, acc, sm.sRS, scale_e, Out_u32, row0, end, nb_cta, warp, wm, wp_n, lane, g, lt
            );
        } else {
            w13_coalesced_epilogue<BM, 1>(
                sm.u.epi, acc, sm.sRS, scale_e, Out_u32, row0, end, nb_cta, warp, wm, wp_n, lane, g, lt
            );
        }

        if (!is_last_m) __syncthreads();
    }
}

// =============================================================================
// Kernel P3 (v4): Expert-Persistent Raw MXFP8 W13 (N=576, K=4608, BN=96)
// =============================================================================
template <int BM, int USE_R2T, int AS_ROW_MAJOR>
__global__ __launch_bounds__(256, (BM == 256 ? 1 : 2))
void pf_mxfp8_w13_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ WP,
    const uint8_t* __restrict__ WSP,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    uint32_t* __restrict__ Out_u32,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int w_evict_mode
) {
    constexpr int KP = 4608;
    constexpr int KG = 144;
    constexpr int M_TILES_W = (BM / 2) / 16;

    const int nb_cta = blockIdx.x;
    const int w_bx   = blockIdx.y;

    extern __shared__ uint4 dyn_smem[];
    auto& sm = *reinterpret_cast<MxW13SmemBlock<BM>*>(dyn_smem);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 2;
    const int wp_n = warp & 3;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, 0, 0, sm.sWork, tid)) {
        return;
    }
    const int64_t e       = sm.sWork[0];
    const int row_begin   = sm.sWork[1];
    const int end         = sm.sWork[2];
    const int num_m       = ((end - row_begin) + BM - 1) / BM;

    const uint8_t* wp_p0  = WP  + (e * 18 + nb_cta * 3) * (9LL * 32LL * 512LL);
    const uint8_t* wsp_p0 = WSP + (e * 18 + nb_cta * 3) * (9LL * 16LL * 32LL);
    const float scale_e   = Ones[e];
    const int m_warp_base = wm * (BM / 2);
    const int m_sf_off    = g + ((lt & 1) << 3);
    const int sw_mask     = (g & 3) << 2;

    #pragma unroll 1
    for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
        const int row0 = row_begin + mi_blk * BM;
        const bool is_last_m = (mi_blk + 1 >= num_m);
        const bool use_cs_w  = (w_evict_mode == 2) || (w_evict_mode == 1 && is_last_m);

        if (tid < BM) {
            int r = row0 + tid;
            sm.sTok[tid] = (r < end) ? (USE_R2T ? Row2Tok[r] : r) : -1;
        }
        __syncthreads();

        auto load_ab_stage = [&](int stg, int buf) {
            const int kp_idx = stg >> 3;
            const int sub_k  = stg & 7;
            constexpr int A_ITERS = (BM * 4) / 256;
            #pragma unroll
            for (int it = 0; it < A_ITERS; ++it) {
                int idx = it * 256 + tid;
                int m   = idx >> 2;
                int c4  = idx & 3;
                int sc4 = c4 ^ (m & 3);
                int32_t tok = sm.sTok[m];
                const uint8_t* src = (tok >= 0)
                    ? (A + static_cast<int64_t>(tok) * KP + stg * 64 + c4 * 16)
                    : A;
                cp_async_cg_16(&sm.u.m.sA[buf][m][sc4 * 4], src, (tok >= 0) ? 16 : 0);
            }
            for (int idx = tid; idx < 384; idx += 256) {
                int n_loc = idx >> 2;
                int c4    = idx & 3;
                int sc4   = c4 ^ (n_loc & 3);
                int pl    = n_loc >> 5;
                int n_in  = n_loc & 31;
                const uint8_t* src = wp_p0 + static_cast<int64_t>(pl) * (9LL * 32LL * 512LL)
                                   + static_cast<int64_t>(kp_idx) * (32 * 512)
                                   + n_in * 512 + sub_k * 64 + c4 * 16;
                cp_async_cg_16(&sm.u.m.sB[buf][n_loc][sc4 * 4], src, 16);
            }
            if ((stg & 3) == 0 && tid < BM) {
                const int chunk = stg >> 2;
                const int abuf  = chunk & 1;
                int32_t tok = sm.sTok[tid];
                if constexpr (USE_R2T || AS_ROW_MAJOR) {
                    const uint8_t* src_as = (tok >= 0)
                        ? (AS + static_cast<int64_t>(tok) * KG + chunk * 8)
                        : AS;
                    cp_async_ca_8(&sm.u.m.sAS[abuf][tid], src_as, (tok >= 0) ? 8 : 0);
                } else {
                    uint64_t sval = 0ull;
                    if (tok >= 0) {
                        int64_t off0 = scale_offset_byte(static_cast<int64_t>(row0 + tid), chunk * 8 + 0, KG, tile);
                        int64_t off1 = scale_offset_byte(static_cast<int64_t>(row0 + tid), chunk * 8 + 4, KG, tile);
                        uint32_t w0 = *reinterpret_cast<const uint32_t*>(AS + off0);
                        uint32_t w1 = *reinterpret_cast<const uint32_t*>(AS + off1);
                        sval = static_cast<uint64_t>(w0) | (static_cast<uint64_t>(w1) << 32);
                    }
                    sm.u.m.sAS[abuf][tid] = sval;
                }
            }
            cp_async_commit();
        };

        auto load_bs_panel = [&](int kp_idx, int bbuf) {
            for (int idx = tid; idx < 384; idx += 256) {
                int pl   = idx >> 7;
                int t128 = idx & 127;
                int kbg  = t128 >> 3;
                int n4   = (t128 & 7) * 4;
                const uint8_t* ws_src = wsp_p0
                    + static_cast<int64_t>(pl) * (9LL * 16LL * 32LL)
                    + static_cast<int64_t>(kp_idx) * 512;
                uint32_t s4 = ld_w_u32(ws_src + kbg * 32 + n4, use_cs_w);
                int nbase = pl * 32 + n4;
                sm.u.m.sBS[bbuf][nbase + 0][kbg] = static_cast<uint8_t>(s4 & 0xFFu);
                sm.u.m.sBS[bbuf][nbase + 1][kbg] = static_cast<uint8_t>((s4 >> 8) & 0xFFu);
                sm.u.m.sBS[bbuf][nbase + 2][kbg] = static_cast<uint8_t>((s4 >> 16) & 0xFFu);
                sm.u.m.sBS[bbuf][nbase + 3][kbg] = static_cast<uint8_t>((s4 >> 24) & 0xFFu);
            }
        };

        load_ab_stage(0, 0);
        load_ab_stage(1, 1);
        load_bs_panel(0, 0);
        cp_async_wait<1>();
        __syncthreads();

        float acc[M_TILES_W][3][4];
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
            }
        }

        const int wm_live = (end - row0) - m_warp_base;
        const int mi_end = (wm_live <= 0) ? 0
                         : ((wm_live >= BM / 2) ? M_TILES_W : ((wm_live + 15) >> 4));

        int cur = 0;
        #pragma unroll 1
        for (int stg = 0; stg < 72; ++stg) {
            const int nxt2   = (cur == 0) ? 2 : (cur - 1);
            const int kp_idx = stg >> 3;
            const int sub_k  = stg & 7;
            const int bcur   = kp_idx & 1;
            const int acur   = (stg >> 2) & 1;
            const int sub4   = stg & 3;

            if (stg + 2 < 72) load_ab_stage(stg + 2, nxt2);
            if (sub_k == 0 && kp_idx + 1 < 9) load_bs_panel(kp_idx + 1, bcur ^ 1);

            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int sw_c0 = (q * 8 + lt) ^ sw_mask;
                const int sw_c1 = (q * 8 + 4 + lt) ^ sw_mask;
                const int as_sh = (sub4 * 2 + q) * 8;
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    int n_row = wp_n * 24 + sn * 8 + g;
                    b0[sn]  = sm.u.m.sB[cur][n_row][sw_c0];
                    b1[sn]  = sm.u.m.sB[cur][n_row][sw_c1];
                    sfb[sn] = static_cast<uint32_t>(sm.u.m.sBS[bcur][n_row][sub_k * 2 + q]);
                }
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    if (mi < mi_end) {
                        int m_lo = m_warp_base + mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur][m_lo][sw_c0];
                        uint32_t a1  = sm.u.m.sA[cur][m_hi][sw_c0];
                        uint32_t a2  = sm.u.m.sA[cur][m_lo][sw_c1];
                        uint32_t a3  = sm.u.m.sA[cur][m_hi][sw_c1];
                        uint32_t sfa = static_cast<uint32_t>(
                            (sm.u.m.sAS[acur][m_warp_base + mi * 16 + m_sf_off] >> as_sh) & 0xFFu);
                        #pragma unroll
                        for (int sn = 0; sn < 3; ++sn) {
                            mma_m16n8k32_mxfp8(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                        }
                    }
                }
            }

            if (stg + 2 < 72) {
                cp_async_wait<1>();
                __syncthreads();
            } else if (stg + 1 < 72) {
                cp_async_wait<0>();
                __syncthreads();
            }
            cur = (cur == 2) ? 0 : (cur + 1);
        }

        if constexpr (BM == 256) {
            w13_coalesced_epilogue_256<0>(
                sm.u.epi128, acc, nullptr, scale_e, Out_u32, row0, end, nb_cta, warp, wm, wp_n, lane, g, lt
            );
        } else {
            w13_coalesced_epilogue<BM, 0>(
                sm.u.epi, acc, nullptr, scale_e, Out_u32, row0, end, nb_cta, warp, wm, wp_n, lane, g, lt
            );
        }

        if (!is_last_m) __syncthreads();
    }
}

// =============================================================================
// Kernel P2A (v4): Expert-Persistent NVFP4 W2 WS5 (N=4608, K=288)
// =============================================================================
// - BM=64 double-buffered sA (SMEM-pinned when n_rows <= 128!) + 36-reg
//   register-resident W_PACK + sWS[128][5] loaded EXACTLY ONCE per panel with
//   ld.global.cs (1.00x exact weight DRAM read, ZERO L2 weight pollution!).
// - Full 64-row W2EpiSmem64 (40.26 KiB <= 48 KiB -> 2 CTAs/SM!) cuts epilogue
//   barriers by >2x and uses `red.relaxed.gpu.global.add.noftz.bf16x2`.
// - Serpentine expert ordering across `nb_blk` cuts `A` DRAM traffic by 38-75%.
// - Supports both single-grid multi-wave (`slice_nb_blk < 0`) and per-slice
//   in-L2 zeroed waves (`slice_nb_blk >= 0`, `l2_zero_slice = 1`).
template <int AS_ROW_MAJOR, int HAS_PATCHES, int COMBINE_MODE, int PANELS_PER_CTA>
__global__ __launch_bounds__(256, 2)
void pf_nvfp4_w2_ws5_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ W_PACK,
    const uint32_t* __restrict__ WS5_PACK,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int as_row_stride,
    int w_evict_mode,
    int slice_nb_blk
) {
    constexpr int BM = 64;
    constexpr int KP = 144;
    constexpr int KG = 18;

    int nb_blk, w_bx, serp_mode;
    if constexpr (COMBINE_MODE == 0) {
        nb_blk    = blockIdx.x;
        w_bx      = blockIdx.y;
        serp_mode = 0;
    } else {
        w_bx      = blockIdx.x;
        nb_blk    = (slice_nb_blk >= 0) ? slice_nb_blk : static_cast<int>(blockIdx.y);
        serp_mode = (slice_nb_blk >= 0) ? 2 : 1;
    }

    __shared__ alignas(16) W2Nvfp4PersistSmem sm;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 2;
    const int wp   = warp & 3;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;

    // Zero upper bytes 18..19 of sWS[0..127][4] BEFORE resolve_persistent_work's __syncthreads()!
    if (tid < 128) sm.sWS[tid][4] = 0u;

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, nb_blk, serp_mode, sm.sWork, tid)) {
        return;
    }
    const int64_t e       = sm.sWork[0];
    const int row_begin   = sm.sWork[1];
    const int end         = sm.sWork[2];
    const int n_rows      = end - row_begin;
    const int num_m       = (n_rows + BM - 1) / BM;
    const bool a_pinned   = (num_m <= 2);
    // Persistent W2 loads each weight panel ONCE per expert -> always use cs unless w_evict_mode==0!
    const bool use_cs_w   = (w_evict_mode != 0);

    const int base_e      = static_cast<int>(WS_BASE[e]);
    const float scale_e   = Ones[e];
    const int src0        = (g << 2) | ((lt & 1) << 1);
    const int src1        = src0 | 1;
    const uint32_t prmt_sel = (lt < 2) ? 0x5410u : 0x7632u;
    const int m_warp_base = wm * (BM / 2);
    const int m_sf_off    = g + ((lt & 1) << 3);

    auto load_a_tile = [&](int mi_blk, int buf, bool cache_l1) {
        const int r0 = row_begin + mi_blk * BM;
        constexpr int A_ITERS = (BM * 8) / 256;   // 2 for BM=64
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * 256 + tid;
            int m   = idx >> 3;
            int c4  = idx & 7;
            int sc4 = c4 ^ (m & 7);
            int r   = r0 + m;
            const uint8_t* src = (r < end) ? (A + static_cast<int64_t>(r) * KP + c4 * 16) : A;
            if (cache_l1) {
                cp_async_ca_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
            } else {
                cp_async_cg_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
            }
        }
        if (tid < BM) {
            int r = r0 + tid;
            const uint8_t* src_t = (r < end) ? (A + static_cast<int64_t>(r) * KP + 128) : A;
            if (cache_l1) {
                cp_async_ca_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
            } else {
                cp_async_cg_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
            }
        }
        cp_async_commit();

        if (tid < BM) {
            int r = r0 + tid;
            if (r < end) {
                sm.sRS[buf][tid] = RS[r];
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = Row2Tok[r];
            } else {
                sm.sRS[buf][tid] = 0.0f;
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = -1;
            }
        }
        for (int idx = tid; idx < BM * 5; idx += 256) {
            int m = (idx * 13108) >> 16;   // exact idx / 5 for idx in [0, 319]
            int p = idx - m * 5;
            int r = r0 + m;
            uint32_t sval = 0u;
            if (r < end) {
                if (p < 4) {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u32_rowmaj(AS, static_cast<int64_t>(r), p * 4, as_row_stride);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile);
                        sval = *reinterpret_cast<const uint32_t*>(AS + off);
                    }
                } else {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u16_rowmaj(AS, static_cast<int64_t>(r), 16, as_row_stride);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), 16, KG, tile);
                        sval = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(AS + off));
                    }
                }
            }
            sm.sAS[buf][m][p] = sval;
        }
    };

    // If n_rows <= 128 (num_m <= 2), preload ALL A tiles into SMEM ONCE across all dp panels!
    if (a_pinned) {
        load_a_tile(0, 0, false);
        if (num_m == 2) load_a_tile(1, 1, false);
    }

    #pragma unroll 1
    for (int dp = 0; dp < PANELS_PER_CTA; ++dp) {
        const int nb = nb_blk * PANELS_PER_CTA + dp;
        const int64_t puid = e * 36 + nb;
        const bool keep_a_l1 = (!a_pinned) && (dp + 1 < PANELS_PER_CTA);

        // Start loading mi_blk=0 of A overlapped with W_PACK & WS5_PACK loads when num_m >= 3!
        if (!a_pinned) {
            load_a_tile(0, 0, keep_a_l1);
        }

        // 1. Load & unpack WS5_PACK ONCE per (e, nb) panel with ld.global.cs!
        const uint32_t* ws5_panel = WS5_PACK + puid * 360LL;
        uint8_t* sWS_u8 = reinterpret_cast<uint8_t*>(sm.sWS);
        for (int u = tid; u < 288; u += 256) {
            uint32_t lo = ld_w_u32(ws5_panel + u, use_cs_w);
            uint32_t hi = ld_w_u32(ws5_panel + 288 + (u >> 2), use_cs_w);
            int hi_sh = (u & 3) * 8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int idx  = u * 8 + j;
                int n    = (idx * 3641) >> 16;
                int kg   = idx - n * 18;
                int code = ((lo >> (4 * j)) & 0xF) | (((hi >> (hi_sh + j)) & 1) << 4);
                sWS_u8[n * 20 + kg] = static_cast<uint8_t>(base_e + code);
            }
        }
        if constexpr (HAS_PATCHES) {
            int p_lo = P_OFF[puid];
            int p_hi = P_OFF[puid + 1];
            if (p_lo < p_hi) {
                __syncthreads();
                for (int pi = p_lo + tid; pi < p_hi; pi += 256) {
                    uint32_t rec = PATCHES[pi];
                    int idx = static_cast<int>(rec >> 8);
                    int n   = (idx * 3641) >> 16;
                    int kg  = idx - n * 18;
                    sWS_u8[n * 20 + kg] = static_cast<uint8_t>(rec & 0xFFu);
                }
            }
        }

        // 2. Load & unswizzle ENTIRE [128, 288] W_PACK panel ONCE into 36 registers!
        const uint8_t* wp_base = W_PACK + (puid * 4 + wp) * (9LL * 512LL);
        uint32_t b0_reg[4][4];
        uint32_t b1_reg[4][4];
        uint32_t b0_tail[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            uint4 w0 = ld_w_u4(wp_base + (2 * p + 0) * 512 + lane * 16, use_cs_w);
            uint4 w1 = ld_w_u4(wp_base + (2 * p + 1) * 512 + lane * 16, use_cs_w);
            uint32_t u0[4] = {w0.x, w0.y, w0.z, w0.w};
            uint32_t u1[4] = {w1.x, w1.y, w1.z, w1.w};
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                b0_reg[p][sn] = prmt_b32(__shfl_sync(0xFFFFFFFFu, u0[sn], src0),
                                         __shfl_sync(0xFFFFFFFFu, u0[sn], src1), prmt_sel);
                b1_reg[p][sn] = prmt_b32(__shfl_sync(0xFFFFFFFFu, u1[sn], src0),
                                         __shfl_sync(0xFFFFFFFFu, u1[sn], src1), prmt_sel);
            }
        }
        {
            uint4 wt = ld_w_u4(wp_base + 8 * 512 + lane * 16, use_cs_w);
            uint32_t u0[4] = {wt.x, wt.y, wt.z, wt.w};
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                b0_tail[sn] = prmt_b32(__shfl_sync(0xFFFFFFFFu, u0[sn], src0),
                                       __shfl_sync(0xFFFFFFFFu, u0[sn], src1), prmt_sel);
            }
        }

        if (!a_pinned || dp == 0) {
            cp_async_wait<0>();
        }
        __syncthreads();

        // 3. Sweep ALL M-tiles of expert e with ZERO global weight loads & ZERO unpack!
        #pragma unroll 1
        for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
            const int cur  = mi_blk & 1;
            const int nxt  = cur ^ 1;
            const int row0 = row_begin + mi_blk * BM;

            if (!a_pinned && mi_blk + 1 < num_m) {
                load_a_tile(mi_blk + 1, nxt, keep_a_l1);
            }

            float acc[2][4][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                    acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
                }
            }

            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint32_t sfb[4];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    sfb[sn] = sm.sWS[wp * 32 + sn * 8 + g][p];
                }
                const int sw_c0 = (p * 8 + lt) ^ (g << 2);
                const int sw_c1 = (p * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    uint32_t a0  = sm.sA_main[cur][m_lo][sw_c0];
                    uint32_t a1  = sm.sA_main[cur][m_hi][sw_c0];
                    uint32_t a2  = sm.sA_main[cur][m_lo][sw_c1];
                    uint32_t a3  = sm.sA_main[cur][m_hi][sw_c1];
                    uint32_t sfa = sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][p];
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3,
                                           b0_reg[p][sn], b1_reg[p][sn], sfa, sfb[sn]);
                    }
                }
            }

            // Tail p = 4 (k = 256..287)
            {
                uint32_t sfb[4];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    sfb[sn] = sm.sWS[wp * 32 + sn * 8 + g][4];
                }
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    uint32_t a0  = sm.sA_tail[cur][m_lo][lt];
                    uint32_t a1  = sm.sA_tail[cur][m_hi][lt];
                    uint32_t sfa = sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][4];
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, 0u, 0u,
                                           b0_tail[sn], 0u, sfa, sfb[sn]);
                    }
                }
            }

            w2_coalesced_epilogue_64<COMBINE_MODE, 1>(
                sm.sEpi, acc, sm.sRS[cur], sm.sTok[cur], scale_e,
                Out, row0, end, nb, warp, wm, wp, lane, g, lt
            );

            if (mi_blk + 1 < num_m) {
                if (!a_pinned) cp_async_wait<0>();
                __syncthreads();
            } else if (dp + 1 < PANELS_PER_CTA) {
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// Kernel P2B (v4): Expert-Persistent NVFP4 W2 PANEL (N=4608, K=288)
// =============================================================================
template <int AS_ROW_MAJOR, int COMBINE_MODE, int PANELS_PER_CTA>
__global__ __launch_bounds__(256, 2)
void pf_nvfp4_w2_panel_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ WP,
    const uint8_t* __restrict__ WSP,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int as_row_stride,
    int w_evict_mode,
    int slice_nb_blk
) {
    constexpr int BM = 64;
    constexpr int KP = 144;
    constexpr int KG = 18;

    int nb_blk, w_bx, serp_mode;
    if constexpr (COMBINE_MODE == 0) {
        nb_blk    = blockIdx.x;
        w_bx      = blockIdx.y;
        serp_mode = 0;
    } else {
        w_bx      = blockIdx.x;
        nb_blk    = (slice_nb_blk >= 0) ? slice_nb_blk : static_cast<int>(blockIdx.y);
        serp_mode = (slice_nb_blk >= 0) ? 2 : 1;
    }

    __shared__ alignas(16) W2Nvfp4PersistSmem sm;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 2;
    const int wp   = warp & 3;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;

    if (tid < 128) sm.sWS[tid][4] = 0u;

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, nb_blk, serp_mode, sm.sWork, tid)) {
        return;
    }
    const int64_t e       = sm.sWork[0];
    const int row_begin   = sm.sWork[1];
    const int end         = sm.sWork[2];
    const int num_m       = ((end - row_begin) + BM - 1) / BM;
    const bool a_pinned   = (num_m <= 2);
    const bool use_cs_w   = (w_evict_mode != 0);

    const float scale_e   = Ones[e];
    const int m_warp_base = wm * (BM / 2);
    const int m_sf_off    = g + ((lt & 1) << 3);

    auto load_a_tile = [&](int mi_blk, int buf, bool cache_l1) {
        const int r0 = row_begin + mi_blk * BM;
        constexpr int A_ITERS = (BM * 8) / 256;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * 256 + tid;
            int m   = idx >> 3;
            int c4  = idx & 7;
            int sc4 = c4 ^ (m & 7);
            int r   = r0 + m;
            const uint8_t* src = (r < end) ? (A + static_cast<int64_t>(r) * KP + c4 * 16) : A;
            if (cache_l1) cp_async_ca_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
        }
        if (tid < BM) {
            int r = r0 + tid;
            const uint8_t* src_t = (r < end) ? (A + static_cast<int64_t>(r) * KP + 128) : A;
            if (cache_l1) cp_async_ca_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
        }
        cp_async_commit();

        if (tid < BM) {
            int r = r0 + tid;
            if (r < end) {
                sm.sRS[buf][tid] = RS[r];
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = Row2Tok[r];
            } else {
                sm.sRS[buf][tid] = 0.0f;
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = -1;
            }
        }
        for (int idx = tid; idx < BM * 5; idx += 256) {
            int m = (idx * 13108) >> 16;
            int p = idx - m * 5;
            int r = r0 + m;
            uint32_t sval = 0u;
            if (r < end) {
                if (p < 4) {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u32_rowmaj(AS, static_cast<int64_t>(r), p * 4, as_row_stride);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile);
                        sval = *reinterpret_cast<const uint32_t*>(AS + off);
                    }
                } else {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u16_rowmaj(AS, static_cast<int64_t>(r), 16, as_row_stride);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), 16, KG, tile);
                        sval = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(AS + off));
                    }
                }
            }
            sm.sAS[buf][m][p] = sval;
        }
    };

    if (a_pinned) {
        load_a_tile(0, 0, false);
        if (num_m == 2) load_a_tile(1, 1, false);
    }

    #pragma unroll 1
    for (int dp = 0; dp < PANELS_PER_CTA; ++dp) {
        const int nb    = nb_blk * PANELS_PER_CTA + dp;
        const int panel = nb >> 1;
        const int sub_n = (nb & 1) * 128;
        const bool keep_a_l1 = (!a_pinned) && (dp + 1 < PANELS_PER_CTA);

        if (!a_pinned) load_a_tile(0, 0, keep_a_l1);

        const uint8_t* wsp_base = WSP + (e * 18 + panel) * (9LL * 2LL * 256LL) + sub_n;
        uint8_t* sWS_u8 = reinterpret_cast<uint8_t*>(sm.sWS);
        for (int idx = tid; idx < 18 * 32; idx += 256) {
            int kg = idx >> 5;
            int n4 = (idx & 31) * 4;
            int ks = kg >> 1;
            int k2 = kg & 1;
            uint32_t s4 = ld_w_u32(wsp_base + ks * 512 + k2 * 256 + n4, use_cs_w);
            sWS_u8[(n4 + 0) * 20 + kg] = static_cast<uint8_t>(s4 & 0xFFu);
            sWS_u8[(n4 + 1) * 20 + kg] = static_cast<uint8_t>((s4 >> 8) & 0xFFu);
            sWS_u8[(n4 + 2) * 20 + kg] = static_cast<uint8_t>((s4 >> 16) & 0xFFu);
            sWS_u8[(n4 + 3) * 20 + kg] = static_cast<uint8_t>((s4 >> 24) & 0xFFu);
        }

        const uint8_t* wp_base = WP + (e * 18 + panel) * (9LL * 256LL * 16LL)
                               + static_cast<int64_t>(sub_n + wp * 32 + g) * 16 + lt * 4;
        uint32_t b0_reg[4][4], b1_reg[4][4], b0_tail[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                b0_reg[p][sn] = ld_w_u32(wp_base + (2 * p + 0) * (256 * 16) + sn * (8 * 16), use_cs_w);
                b1_reg[p][sn] = ld_w_u32(wp_base + (2 * p + 1) * (256 * 16) + sn * (8 * 16), use_cs_w);
            }
        }
        #pragma unroll
        for (int sn = 0; sn < 4; ++sn) {
            b0_tail[sn] = ld_w_u32(wp_base + 8 * (256 * 16) + sn * (8 * 16), use_cs_w);
        }

        if (!a_pinned || dp == 0) cp_async_wait<0>();
        __syncthreads();

        #pragma unroll 1
        for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
            const int cur  = mi_blk & 1;
            const int nxt  = cur ^ 1;
            const int row0 = row_begin + mi_blk * BM;

            if (!a_pinned && mi_blk + 1 < num_m) {
                load_a_tile(mi_blk + 1, nxt, keep_a_l1);
            }

            float acc[2][4][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                    acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
                }
            }

            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint32_t sfb[4];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) sfb[sn] = sm.sWS[wp * 32 + sn * 8 + g][p];
                const int sw_c0 = (p * 8 + lt) ^ (g << 2);
                const int sw_c1 = (p * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    uint32_t a0  = sm.sA_main[cur][m_lo][sw_c0];
                    uint32_t a1  = sm.sA_main[cur][m_hi][sw_c0];
                    uint32_t a2  = sm.sA_main[cur][m_lo][sw_c1];
                    uint32_t a3  = sm.sA_main[cur][m_hi][sw_c1];
                    uint32_t sfa = sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][p];
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3,
                                           b0_reg[p][sn], b1_reg[p][sn], sfa, sfb[sn]);
                    }
                }
            }

            {
                uint32_t sfb[4];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) sfb[sn] = sm.sWS[wp * 32 + sn * 8 + g][4];
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    uint32_t a0  = sm.sA_tail[cur][m_lo][lt];
                    uint32_t a1  = sm.sA_tail[cur][m_hi][lt];
                    uint32_t sfa = sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][4];
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, 0u, 0u,
                                           b0_tail[sn], 0u, sfa, sfb[sn]);
                    }
                }
            }

            w2_coalesced_epilogue_64<COMBINE_MODE, 1>(
                sm.sEpi, acc, sm.sRS[cur], sm.sTok[cur], scale_e,
                Out, row0, end, nb, warp, wm, wp, lane, g, lt
            );

            if (mi_blk + 1 < num_m) {
                if (!a_pinned) cp_async_wait<0>();
                __syncthreads();
            } else if (dp + 1 < PANELS_PER_CTA) {
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// Kernel P4 (v4): Expert-Persistent Raw MXFP8 W2 (N=4608, K=288)
// =============================================================================
// 1 CTA/SM (55.3 KiB dyn SMEM, ~128 regs -> 0 spills): loads entire [128, 288]
// panel ONCE into 72 regs (`b0_reg[9][4], b1_reg[9][4]`) + `sWS[9][128]` with
// `ld.global.cs`, sweeps ALL M-tiles with double-buffered BM=64 sA!
template <int AS_ROW_MAJOR, int COMBINE_MODE, int PANELS_PER_CTA>
__global__ __launch_bounds__(256, 1)
void pf_mxfp8_w2_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ WP,
    const uint8_t* __restrict__ WSP,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int as_row_stride,
    int w_evict_mode,
    int slice_nb_blk
) {
    constexpr int BM = 64;
    constexpr int KP = 288;
    constexpr int KG = 9;

    int nb_blk, w_bx, serp_mode;
    if constexpr (COMBINE_MODE == 0) {
        nb_blk    = blockIdx.x;
        w_bx      = blockIdx.y;
        serp_mode = 0;
    } else {
        w_bx      = blockIdx.x;
        nb_blk    = (slice_nb_blk >= 0) ? slice_nb_blk : static_cast<int>(blockIdx.y);
        serp_mode = (slice_nb_blk >= 0) ? 2 : 1;
    }

    extern __shared__ uint4 dyn_smem[];
    auto& sm = *reinterpret_cast<W2Mxfp8PersistSmem*>(dyn_smem);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 2;
    const int wp   = warp & 3;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, nb_blk, serp_mode, sm.sWork, tid)) {
        return;
    }
    const int64_t e       = sm.sWork[0];
    const int row_begin   = sm.sWork[1];
    const int end         = sm.sWork[2];
    const int num_m       = ((end - row_begin) + BM - 1) / BM;
    const bool a_pinned   = (num_m <= 2);
    const bool use_cs_w   = (w_evict_mode != 0);

    const float scale_e   = Ones[e];
    const int m_warp_base = wm * (BM / 2);
    const int m_sf_off    = g + ((lt & 1) << 3);

    auto load_a_tile = [&](int mi_blk, int buf, bool cache_l1) {
        const int r0 = row_begin + mi_blk * BM;
        constexpr int A_ITERS = (BM * 16) / 256;   // 4 for BM=64
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * 256 + tid;
            int m   = idx >> 4;
            int c4  = idx & 15;
            int sc4 = c4 ^ (m & 7);
            int r   = r0 + m;
            const uint8_t* src = (r < end) ? (A + static_cast<int64_t>(r) * KP + c4 * 16) : A;
            if (cache_l1) cp_async_ca_16(&sm.sA_8[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_8[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
        }
        if (tid < BM * 2) {
            int m = tid >> 1;
            int h = tid & 1;
            int r = r0 + m;
            const uint8_t* src = (r < end) ? (A + static_cast<int64_t>(r) * KP + 256 + h * 16) : A;
            if (cache_l1) cp_async_ca_16(&sm.sA_last[buf][m][h * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_last[buf][m][h * 4], src, (r < end) ? 16 : 0);
        }
        cp_async_commit();

        if (tid < BM) {
            int r = r0 + tid;
            if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = (r < end) ? Row2Tok[r] : -1;
        }
        if (tid < BM * 3) {
            int m = (tid * 21846) >> 16;   // exact tid / 3 for tid in [0, 191]
            int p = tid - m * 3;
            int r = r0 + m;
            uint32_t sval = 0u;
            if (r < end) {
                if (p < 2) {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u32_rowmaj(AS, static_cast<int64_t>(r), p * 4, as_row_stride);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile);
                        sval = *reinterpret_cast<const uint32_t*>(AS + off);
                    }
                } else {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = static_cast<uint32_t>(AS[static_cast<int64_t>(r) * as_row_stride + 8]);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), 8, KG, tile);
                        sval = static_cast<uint32_t>(AS[off]);
                    }
                }
            }
            sm.sAS[buf][m][p] = sval;
        }
    };

    if (a_pinned) {
        load_a_tile(0, 0, false);
        if (num_m == 2) load_a_tile(1, 1, false);
    }

    #pragma unroll 1
    for (int dp = 0; dp < PANELS_PER_CTA; ++dp) {
        const int nb = nb_blk * PANELS_PER_CTA + dp;
        const bool keep_a_l1 = (!a_pinned) && (dp + 1 < PANELS_PER_CTA);

        if (!a_pinned) load_a_tile(0, 0, keep_a_l1);

        const uint32_t* wsp_u32 = reinterpret_cast<const uint32_t*>(WSP + (e * 36 + nb) * (9LL * 128LL));
        uint32_t* sWS_u32 = reinterpret_cast<uint32_t*>(sm.sWS);
        for (int idx = tid; idx < 9 * 32; idx += 256) {
            sWS_u32[idx] = ld_w_u32(wsp_u32 + idx, use_cs_w);
        }

        const uint8_t* wp_base = WP + (e * 36 + nb) * (9LL * 128LL * 32LL)
                               + static_cast<int64_t>(wp * 32 + g) * 32;
        uint32_t b0_reg[9][4], b1_reg[9][4];
        #pragma unroll
        for (int ks = 0; ks < 9; ++ks) {
            const uint8_t* wk = wp_base + ks * (128 * 32);
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                b0_reg[ks][sn] = ld_w_u32(wk + sn * (8 * 32) + lt * 4, use_cs_w);
                b1_reg[ks][sn] = ld_w_u32(wk + sn * (8 * 32) + 16 + lt * 4, use_cs_w);
            }
        }

        if (!a_pinned || dp == 0) cp_async_wait<0>();
        __syncthreads();

        #pragma unroll 1
        for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
            const int cur  = mi_blk & 1;
            const int nxt  = cur ^ 1;
            const int row0 = row_begin + mi_blk * BM;

            if (!a_pinned && mi_blk + 1 < num_m) {
                load_a_tile(mi_blk + 1, nxt, keep_a_l1);
            }

            float acc[2][4][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                    acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
                }
            }

            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                uint32_t sfb[4];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) sfb[sn] = static_cast<uint32_t>(sm.sWS[ks][wp * 32 + sn * 8 + g]);
                const int sw_c0 = (ks * 8 + lt) ^ (g << 2);
                const int sw_c1 = (ks * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    uint32_t a0  = sm.sA_8[cur][m_lo][sw_c0];
                    uint32_t a1  = sm.sA_8[cur][m_hi][sw_c0];
                    uint32_t a2  = sm.sA_8[cur][m_lo][sw_c1];
                    uint32_t a3  = sm.sA_8[cur][m_hi][sw_c1];
                    uint32_t sfa = static_cast<uint32_t>(
                        reinterpret_cast<const uint8_t*>(sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off])[ks]);
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        mma_m16n8k32_mxfp8(acc[mi][sn], a0, a1, a2, a3,
                                           b0_reg[ks][sn], b1_reg[ks][sn], sfa, sfb[sn]);
                    }
                }
            }

            // ks = 8 (k = 256..287)
            {
                uint32_t sfb[4];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) sfb[sn] = static_cast<uint32_t>(sm.sWS[8][wp * 32 + sn * 8 + g]);
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    uint32_t a0  = sm.sA_last[cur][m_lo][lt];
                    uint32_t a1  = sm.sA_last[cur][m_hi][lt];
                    uint32_t a2  = sm.sA_last[cur][m_lo][4 + lt];
                    uint32_t a3  = sm.sA_last[cur][m_hi][4 + lt];
                    uint32_t sfa = static_cast<uint32_t>(
                        reinterpret_cast<const uint8_t*>(sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off])[8]);
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        mma_m16n8k32_mxfp8(acc[mi][sn], a0, a1, a2, a3,
                                           b0_reg[8][sn], b1_reg[8][sn], sfa, sfb[sn]);
                    }
                }
            }

            w2_coalesced_epilogue_64<COMBINE_MODE, 0>(
                sm.sEpi, acc, nullptr, sm.sTok[cur], scale_e,
                Out, row0, end, nb, warp, wm, wp, lane, g, lt
            );

            if (mi_blk + 1 < num_m) {
                if (!a_pinned) cp_async_wait<0>();
                __syncthreads();
            } else if (dp + 1 < PANELS_PER_CTA) {
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// Helper Kernels (100% preserved from v3: streaming zero, FP32->BF16, row2tok)
// =============================================================================

__global__ __launch_bounds__(256)
void zero_bf16_cs_kernel(uint16_t* __restrict__ Out_bf16, int64_t numel) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base = idx * 32;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    if (base + 32 <= numel) {
        uint4* p = reinterpret_cast<uint4*>(Out_bf16 + base);
        st_cs_u4(p + 0, z4);
        st_cs_u4(p + 1, z4);
        st_cs_u4(p + 2, z4);
        st_cs_u4(p + 3, z4);
    } else {
        for (int64_t i = base; i < numel && i < base + 32; ++i) {
            Out_bf16[i] = 0u;
        }
    }
}

__global__ __launch_bounds__(256)
void fp32_to_bf16_and_zero_kernel(
    float* __restrict__ Y_fp32,
    uint16_t* __restrict__ Out_bf16,
    int64_t numel
) {
    const int64_t idx8 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base = idx8 * 8;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    if (base + 8 <= numel) {
        uint4* y_u4 = reinterpret_cast<uint4*>(Y_fp32 + base);
        uint4 a_u = ld_cs_u4(y_u4 + 0);
        uint4 b_u = ld_cs_u4(y_u4 + 1);
        const float4 a = *reinterpret_cast<const float4*>(&a_u);
        const float4 b = *reinterpret_cast<const float4*>(&b_u);
        uint4 out_v;
        out_v.x = pack_bf16x2(a.x, a.y);
        out_v.y = pack_bf16x2(a.z, a.w);
        out_v.z = pack_bf16x2(b.x, b.y);
        out_v.w = pack_bf16x2(b.z, b.w);
        st_wb_u4(Out_bf16 + base, out_v);
        st_cs_u4(y_u4 + 0, z4);
        st_cs_u4(y_u4 + 1, z4);
    } else {
        for (int64_t i = base; i < numel; ++i) {
            float v = Y_fp32[i];
            __nv_bfloat16 bv = __float2bfloat16_rn(v);
            Out_bf16[i] = *reinterpret_cast<const uint16_t*>(&bv);
            Y_fp32[i] = 0.0f;
        }
    }
}

__global__ __launch_bounds__(256)
void build_row2tok_kernel(
    const int32_t* __restrict__ Src2Dst,
    int32_t* __restrict__ Row2Tok,
    int32_t total_slots,
    int32_t topk
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_slots) {
        int32_t dst = Src2Dst[idx];
        if (dst >= 0) {
            Row2Tok[dst] = (topk == 8) ? (idx >> 3) : (idx / topk);
        }
    }
}

// =============================================================================
// v3 Non-Persistent Compat Kernels (NVFP4 WS5 W13 + W2 for same-.so A/B toggle)
// =============================================================================

template <int BM, int COMBINE_MODE, int HAS_RS>
__device__ __forceinline__ void w2_coalesced_epilogue_v3(
    W2EpiSmem& sEpi,
    const float (&acc)[(BM / 2) / 16][4][4],
    const float* sRS,
    const int32_t* sTok,
    float scale_e,
    void* __restrict__ Out,
    int row0, int end, int nb,
    int warp, int wm, int wp, int lane, int g, int lt
) {
    constexpr int N = 4608;
    constexpr int M_TILES_W = (BM / 2) / 16;
    const int m_warp_base = wm * (BM / 2);

    if constexpr (COMBINE_MODE == 0 || COMBINE_MODE == 2) {
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            int m_lo = m_warp_base + mi * 16 + g;
            int m_hi = m_lo + 8;
            float s_lo = scale_e;
            float s_hi = scale_e;
            if constexpr (HAS_RS) {
                s_lo = (row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
                s_hi = (row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
            }
            int r_epi_lo = wm * 16 + g;
            int r_epi_hi = r_epi_lo + 8;
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                int col_pair = wp * 16 + sn * 4 + lt;
                sEpi.bf16x2[r_epi_lo][col_pair] = pack_bf16x2(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo);
                sEpi.bf16x2[r_epi_hi][col_pair] = pack_bf16x2(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi);
            }
            __syncthreads();

            if constexpr (COMBINE_MODE == 0) {
                const int half_w = lane >> 4;
                const int c4 = lane & 15;
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    int r_epi = warp * 4 + k * 2 + half_w;
                    int m = (r_epi < 16) ? (mi * 16 + r_epi) : ((BM / 2) + mi * 16 + (r_epi - 16));
                    int r_glob = row0 + m;
                    if (r_glob < end) {
                        uint4 v;
                        v.x = sEpi.bf16x2[r_epi][c4 * 4 + 0];
                        v.y = sEpi.bf16x2[r_epi][c4 * 4 + 1];
                        v.z = sEpi.bf16x2[r_epi][c4 * 4 + 2];
                        v.w = sEpi.bf16x2[r_epi][c4 * 4 + 3];
                        uint4* dst = reinterpret_cast<uint4*>(
                            reinterpret_cast<uint32_t*>(Out) + static_cast<int64_t>(r_glob) * (N / 2) + nb * 64);
                        st_cs_u4(dst + c4, v);
                    }
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int r_epi = warp * 4 + k;
                    int m = (r_epi < 16) ? (mi * 16 + r_epi) : ((BM / 2) + mi * 16 + (r_epi - 16));
                    int r_glob = row0 + m;
                    if (r_glob < end) {
                        int32_t tok = sTok[m];
                        if (tok >= 0) {
                            uint32_t* dst = reinterpret_cast<uint32_t*>(Out)
                                          + static_cast<int64_t>(tok) * (N / 2) + nb * 64;
                            red_add_bf16x2_bits(dst + lane,      sEpi.bf16x2[r_epi][lane]);
                            red_add_bf16x2_bits(dst + 32 + lane, sEpi.bf16x2[r_epi][32 + lane]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    } else {
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            #pragma unroll
            for (int w_sel = 0; w_sel < 2; ++w_sel) {
                if (wm == w_sel) {
                    int m_lo = m_warp_base + mi * 16 + g;
                    int m_hi = m_lo + 8;
                    float s_lo = scale_e;
                    float s_hi = scale_e;
                    if constexpr (HAS_RS) {
                        s_lo = (row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
                        s_hi = (row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
                    }
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        int c0 = wp * 32 + sn * 8 + lt * 2;
                        sEpi.f32[g][c0 + 0]     = acc[mi][sn][0] * s_lo;
                        sEpi.f32[g][c0 + 1]     = acc[mi][sn][1] * s_lo;
                        sEpi.f32[g + 8][c0 + 0] = acc[mi][sn][2] * s_hi;
                        sEpi.f32[g + 8][c0 + 1] = acc[mi][sn][3] * s_hi;
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    int r_epi = warp * 2 + k;
                    int m = w_sel * (BM / 2) + mi * 16 + r_epi;
                    int r_glob = row0 + m;
                    if (r_glob < end) {
                        int32_t tok = sTok[m];
                        if (tok >= 0) {
                            float* dst = reinterpret_cast<float*>(Out)
                                       + static_cast<int64_t>(tok) * N + nb * 128;
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                red_add_f32(dst + q * 32 + lane, sEpi.f32[r_epi][q * 32 + lane]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

template <int BM, int USE_R2T, int AS_ROW_MAJOR, int HAS_PATCHES>
__global__ __launch_bounds__(256, 2)
void pf_nvfp4_w13_ws5_v3_compat_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    uint32_t* __restrict__ Out_u32,
    int max_ctas,
    int tile,
    int w_evict_mode
) {
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int M_TILES_W = (BM / 2) / 16;

    const int sub_m = tile / BM;
    const int pid0 = blockIdx.x;
    const int t = blockIdx.y;
    const int nb_cta = pid0 / sub_m;
    const int sub = pid0 - nb_cta * sub_m;

    const int num_tiles = Tiles[2 * max_ctas];
    if (t >= num_tiles) return;
    const int end = Tiles[max_ctas + t];
    const int row0 = t * tile + sub * BM;
    if (row0 >= end) return;

    const int64_t e = Tiles[t];
    const bool is_last_e = (row0 + BM >= end) && (t + 1 >= num_tiles || Tiles[t + 1] != static_cast<int32_t>(e));
    const bool use_cs_w = (w_evict_mode == 2) || (w_evict_mode == 1 && is_last_e);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wp_n = warp & 3;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;

    __shared__ alignas(16) union {
        struct { uint32_t sA[2][BM][32]; uint32_t sAS[2][BM][4]; } m;
        W13EpiSmem epi;
    } sm;
    __shared__ alignas(16) int32_t sTok[BM];
    __shared__ alignas(16) float   sRS[BM];
    __shared__ alignas(16) int32_t sPOff[12][2];

    if (tid < BM) {
        int r = row0 + tid;
        if (r < end) { sTok[tid] = USE_R2T ? Row2Tok[r] : r; sRS[tid] = RS[r]; }
        else { sTok[tid] = -1; sRS[tid] = 0.0f; }
    }
    if constexpr (HAS_PATCHES) {
        if (tid < 12) {
            int64_t wuid = (e * 18 + nb_cta * 3 + (tid >> 2)) * 4 + (tid & 3);
            sPOff[tid][0] = P_OFF[wuid];
            sPOff[tid][1] = P_OFF[wuid + 1];
        }
    }
    __syncthreads();

    auto load_stage = [&](int ks, int buf) {
        constexpr int A_ITERS = (BM * 8) / 256;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * 256 + tid;
            int m = idx >> 3; int c4 = idx & 7; int sc4 = c4 ^ (m & 7);
            int32_t tok = sTok[m];
            const uint8_t* src = (tok >= 0) ? (A + static_cast<int64_t>(tok) * KP + ks * 128 + c4 * 16) : A;
            cp_async_cg_16(&sm.m.sA[buf][m][sc4 * 4], src, (tok >= 0) ? 16 : 0);
        }
        if constexpr (USE_R2T || AS_ROW_MAJOR) {
            if (tid < BM) {
                int32_t tok = sTok[tid];
                const uint8_t* src_as = (tok >= 0) ? (AS + static_cast<int64_t>(tok) * KG + ks * 16) : AS;
                cp_async_cg_16(&sm.m.sAS[buf][tid][0], src_as, (tok >= 0) ? 16 : 0);
            }
        } else {
            constexpr int AS_ITERS = (BM * 4) / 256;
            #pragma unroll
            for (int it = 0; it < AS_ITERS; ++it) {
                int idx = it * 256 + tid; int m = idx >> 2; int q = idx & 3;
                int32_t tok = sTok[m];
                uint32_t sval = 0u;
                if (tok >= 0) {
                    int64_t off = scale_offset_byte(static_cast<int64_t>(row0 + m), ks * 16 + q * 4, KG, tile);
                    sval = *reinterpret_cast<const uint32_t*>(AS + off);
                }
                sm.m.sAS[buf][m][q] = sval;
            }
        }
        cp_async_commit();
    };

    const uint8_t* slabs_cta = SLABS + (e * 18 + nb_cta * 3) * (9LL * 8832LL);
    const int base_e = static_cast<int>(WS_BASE[e]);

    auto decode_sfb_ss = [&](int ss, uint32_t (&p0)[3], uint32_t (&p1)[3]) {
        const int kg_lo = ss * 32;
        const int kg_hi = kg_lo + 32;
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            int wg = wp_n * 3 + sn; int pl = wg >> 2; int wp = wg & 3;
            const uint8_t* ss_ptr = slabs_cta + static_cast<int64_t>(pl * 9 + ss) * 8832;
            uint32_t ws_lo = ld_w_u32(ss_ptr + 8192 + wp * 128 + lane * 4, use_cs_w);
            uint8_t  ws_hi = ld_w_u8(ss_ptr + 8704 + wp * 32 + lane, use_cs_w);
            uint8_t sb[8];
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                int code = ((ws_lo >> (4 * ii)) & 0xF) | (((ws_hi >> ii) & 1) << 4);
                sb[ii] = static_cast<uint8_t>(base_e + code);
            }
            if constexpr (HAS_PATCHES) {
                int plo = sPOff[wg][0], phi = sPOff[wg][1];
                for (int pi = plo; pi < phi; ++pi) {
                    uint32_t rec = PATCHES[pi];
                    int p_kg = static_cast<int>(rec >> 16);
                    if (p_kg >= kg_lo && p_kg < kg_hi) {
                        if (g == static_cast<int>((rec >> 8) & 0xFF) && lt == ((p_kg - kg_lo) >> 3)) {
                            sb[p_kg & 7] = static_cast<uint8_t>(rec & 0xFF);
                        }
                    }
                }
            }
            p0[sn] = static_cast<uint32_t>(sb[0]) | (static_cast<uint32_t>(sb[1]) << 8)
                   | (static_cast<uint32_t>(sb[2]) << 16) | (static_cast<uint32_t>(sb[3]) << 24);
            p1[sn] = static_cast<uint32_t>(sb[4]) | (static_cast<uint32_t>(sb[5]) << 8)
                   | (static_cast<uint32_t>(sb[6]) << 16) | (static_cast<uint32_t>(sb[7]) << 24);
        }
    };

    load_stage(0, 0);
    uint4 wa[3], wb[3];
    #pragma unroll
    for (int sn = 0; sn < 3; ++sn) {
        int wg = wp_n * 3 + sn;
        wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9) * 8832 + (wg & 3) * 1024 + lane * 16, use_cs_w);
    }
    uint32_t sfb_p0[3], sfb_p1[3];
    decode_sfb_ss(0, sfb_p0, sfb_p1);
    cp_async_wait<0>();
    __syncthreads();

    float acc[M_TILES_W][3][4];
    #pragma unroll
    for (int mi = 0; mi < M_TILES_W; ++mi)
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
            acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
        }

    const int m_warp_base = wm * (BM / 2);
    const int m_sf_off = g + ((lt & 1) << 3);

    #pragma unroll 1
    for (int ks = 0; ks < 18; ++ks) {
        const int cur = ks & 1, nxt = cur ^ 1, ss = ks >> 1, ks2 = ks & 1;
        if (ks + 1 < 18) load_stage(ks + 1, nxt);
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            int wg = wp_n * 3 + sn;
            wb[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + ss) * 8832
                             + ks2 * 4096 + (wg & 3) * 1024 + 512 + lane * 16, use_cs_w);
        }
        const int src_lt0 = ks2 * 2;
        for (int q2 = 0; q2 < 2; ++q2) {
            uint32_t b0[3], b1[3], sfb[3];
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                uint32_t lx = (q2 == 0) ? wa[sn].x : wa[sn].z;
                uint32_t ly = (q2 == 0) ? wa[sn].y : wa[sn].w;
                b0[sn] = prmt_b32(lx, ly, 0x5410u);
                b1[sn] = prmt_b32(lx, ly, 0x7632u);
                sfb[sn] = __shfl_sync(0xFFFFFFFFu, (q2 == 0) ? sfb_p0[sn] : sfb_p1[sn], (g << 2) | src_lt0);
            }
            const int sw_c0 = (q2 * 8 + lt) ^ (g << 2);
            const int sw_c1 = (q2 * 8 + 4 + lt) ^ (g << 2);
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                uint32_t a0 = sm.m.sA[cur][m_lo][sw_c0], a1 = sm.m.sA[cur][m_hi][sw_c0];
                uint32_t a2 = sm.m.sA[cur][m_lo][sw_c1], a3 = sm.m.sA[cur][m_hi][sw_c1];
                uint32_t sfa = sm.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][q2];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn)
                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
            }
        }
        if (ks + 1 < 18) {
            const int n_ss = (ks + 1) >> 1, n_ks2 = (ks + 1) & 1;
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                int wg = wp_n * 3 + sn;
                wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + n_ss) * 8832
                                 + n_ks2 * 4096 + (wg & 3) * 1024 + lane * 16, use_cs_w);
            }
        }
        const int src_lt1 = ks2 * 2 + 1;
        for (int q2 = 0; q2 < 2; ++q2) {
            uint32_t b0[3], b1[3], sfb[3];
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                uint32_t lx = (q2 == 0) ? wb[sn].x : wb[sn].z;
                uint32_t ly = (q2 == 0) ? wb[sn].y : wb[sn].w;
                b0[sn] = prmt_b32(lx, ly, 0x5410u);
                b1[sn] = prmt_b32(lx, ly, 0x7632u);
                sfb[sn] = __shfl_sync(0xFFFFFFFFu, (q2 == 0) ? sfb_p0[sn] : sfb_p1[sn], (g << 2) | src_lt1);
            }
            const int sw_c0 = ((2 + q2) * 8 + lt) ^ (g << 2);
            const int sw_c1 = ((2 + q2) * 8 + 4 + lt) ^ (g << 2);
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                uint32_t a0 = sm.m.sA[cur][m_lo][sw_c0], a1 = sm.m.sA[cur][m_hi][sw_c0];
                uint32_t a2 = sm.m.sA[cur][m_lo][sw_c1], a3 = sm.m.sA[cur][m_hi][sw_c1];
                uint32_t sfa = sm.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][2 + q2];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn)
                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
            }
        }
        if (ks + 1 < 18) {
            if (((ks + 1) & 1) == 0) decode_sfb_ss((ks + 1) >> 1, sfb_p0, sfb_p1);
            cp_async_wait<0>();
            __syncthreads();
        }
    }
    w13_coalesced_epilogue<BM, 1>(
        sm.epi, acc, sRS, Ones[e], Out_u32, row0, end, nb_cta, warp, wm, wp_n, lane, g, lt
    );
}

template <int BM, int AS_ROW_MAJOR, int HAS_PATCHES, int COMBINE_MODE, int PANELS_PER_CTA>
__global__ __launch_bounds__(256, 2)
void pf_nvfp4_w2_ws5_v3_compat_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ W_PACK,
    const uint32_t* __restrict__ WS5_PACK,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int as_row_stride,
    int w_evict_mode
) {
    constexpr int KP = 144;
    constexpr int KG = 18;
    constexpr int M_TILES_W = (BM / 2) / 16;

    const int sub_m = tile / BM;
    int nb_blk, t, sub;
    if constexpr (COMBINE_MODE == 0) {
        const int pid0 = blockIdx.x; t = blockIdx.y; nb_blk = pid0 / sub_m; sub = pid0 - nb_blk * sub_m;
    } else {
        const int pid0 = blockIdx.x; nb_blk = blockIdx.y; t = pid0 / sub_m; sub = pid0 - t * sub_m;
    }
    const int num_tiles = Tiles[2 * max_ctas];
    if (t >= num_tiles) return;
    const int end = Tiles[max_ctas + t];
    const int row0 = t * tile + sub * BM;
    if (row0 >= end) return;

    const int64_t e = Tiles[t];
    const bool is_last_e = (row0 + BM >= end) && (t + 1 >= num_tiles || Tiles[t + 1] != static_cast<int32_t>(e));
    const bool use_cs_w = (w_evict_mode == 2) || (w_evict_mode == 1 && is_last_e);

    const int tid = threadIdx.x, warp = tid >> 5, wm = warp >> 2, wp = warp & 3;
    const int lane = tid & 31, g = lane >> 2, lt = lane & 3;

    __shared__ alignas(16) uint32_t sA_main[BM][32];
    __shared__ alignas(16) uint32_t sA_tail[BM][4];
    __shared__ alignas(16) uint32_t sAS[BM][5];
    __shared__ alignas(16) uint32_t sWS[128][5];
    __shared__ alignas(16) float    sRS[BM];
    __shared__ alignas(16) int32_t  sTok[BM];
    __shared__ alignas(16) W2EpiSmem sEpi;

    {
        constexpr int A_ITERS = (BM * 8) / 256;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * 256 + tid; int m = idx >> 3; int c4 = idx & 7; int sc4 = c4 ^ (m & 7);
            int r = row0 + m;
            cp_async_cg_16(&sA_main[m][sc4 * 4], (r < end) ? (A + static_cast<int64_t>(r) * KP + c4 * 16) : A, (r < end) ? 16 : 0);
        }
        if (tid < BM) {
            int r = row0 + tid;
            cp_async_cg_16(&sA_tail[tid][0], (r < end) ? (A + static_cast<int64_t>(r) * KP + 128) : A, (r < end) ? 16 : 0);
        }
        cp_async_commit();
    }
    if (tid < BM) {
        int r = row0 + tid;
        if (r < end) { sRS[tid] = RS[r]; if constexpr (COMBINE_MODE != 0) sTok[tid] = Row2Tok[r]; }
        else { sRS[tid] = 0.0f; if constexpr (COMBINE_MODE != 0) sTok[tid] = -1; }
    }
    if (tid < 128) sWS[tid][4] = 0u;
    for (int idx = tid; idx < BM * 5; idx += 256) {
        int m = (idx * 13108) >> 16; int p = idx - m * 5; int r = row0 + m;
        uint32_t sval = 0u;
        if (r < end) {
            if (p < 4) {
                if constexpr (AS_ROW_MAJOR) sval = load_as_u32_rowmaj(AS, static_cast<int64_t>(r), p * 4, as_row_stride);
                else sval = *reinterpret_cast<const uint32_t*>(AS + scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile));
            } else {
                if constexpr (AS_ROW_MAJOR) sval = load_as_u16_rowmaj(AS, static_cast<int64_t>(r), 16, as_row_stride);
                else sval = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(AS + scale_offset_byte(static_cast<int64_t>(r), 16, KG, tile)));
            }
        }
        sAS[m][p] = sval;
    }
    __syncthreads();

    const int base_e = static_cast<int>(WS_BASE[e]);
    const float scale_e = Ones[e];
    const int src0 = (g << 2) | ((lt & 1) << 1), src1 = src0 | 1;
    const uint32_t prmt_sel = (lt < 2) ? 0x5410u : 0x7632u;
    const int m_warp_base = wm * (BM / 2), m_sf_off = g + ((lt & 1) << 3);

    #pragma unroll 1
    for (int dp = 0; dp < PANELS_PER_CTA; ++dp) {
        const int nb = nb_blk * PANELS_PER_CTA + dp;
        const int64_t puid = e * 36 + nb;
        const uint32_t* ws5_panel = WS5_PACK + puid * 360LL;
        uint8_t* sWS_u8 = reinterpret_cast<uint8_t*>(sWS);
        for (int u = tid; u < 288; u += 256) {
            uint32_t lo = ld_w_u32(ws5_panel + u, use_cs_w);
            uint32_t hi = ld_w_u32(ws5_panel + 288 + (u >> 2), use_cs_w);
            int hi_sh = (u & 3) * 8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int idx = u * 8 + j; int n = (idx * 3641) >> 16; int kg = idx - n * 18;
                int code = ((lo >> (4 * j)) & 0xF) | (((hi >> (hi_sh + j)) & 1) << 4);
                sWS_u8[n * 20 + kg] = static_cast<uint8_t>(base_e + code);
            }
        }
        if constexpr (HAS_PATCHES) {
            int p_lo = P_OFF[puid], p_hi = P_OFF[puid + 1];
            if (p_lo < p_hi) {
                __syncthreads();
                for (int pi = p_lo + tid; pi < p_hi; pi += 256) {
                    uint32_t rec = PATCHES[pi];
                    int idx = static_cast<int>(rec >> 8); int n = (idx * 3641) >> 16; int kg = idx - n * 18;
                    sWS_u8[n * 20 + kg] = static_cast<uint8_t>(rec & 0xFFu);
                }
            }
        }
        if (dp == 0) cp_async_wait<0>();
        __syncthreads();

        const uint8_t* wp_base = W_PACK + (puid * 4 + wp) * (9LL * 512LL);
        uint4 w0_cur = ld_w_u4(wp_base + 0 * 512 + lane * 16, use_cs_w);
        uint4 w1_cur = ld_w_u4(wp_base + 1 * 512 + lane * 16, use_cs_w);

        float acc[M_TILES_W][4][4];
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi)
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
            }

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            uint32_t u0[4] = {w0_cur.x, w0_cur.y, w0_cur.z, w0_cur.w};
            uint32_t u1[4] = {w1_cur.x, w1_cur.y, w1_cur.z, w1_cur.w};
            uint32_t b0[4], b1[4], sfb[4];
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                b0[sn] = prmt_b32(__shfl_sync(0xFFFFFFFFu, u0[sn], src0), __shfl_sync(0xFFFFFFFFu, u0[sn], src1), prmt_sel);
                b1[sn] = prmt_b32(__shfl_sync(0xFFFFFFFFu, u1[sn], src0), __shfl_sync(0xFFFFFFFFu, u1[sn], src1), prmt_sel);
                sfb[sn] = sWS[wp * 32 + sn * 8 + g][p];
            }
            if (p < 3) {
                w0_cur = ld_w_u4(wp_base + (2 * p + 2) * 512 + lane * 16, use_cs_w);
                w1_cur = ld_w_u4(wp_base + (2 * p + 3) * 512 + lane * 16, use_cs_w);
            } else {
                w0_cur = ld_w_u4(wp_base + 8 * 512 + lane * 16, use_cs_w);
            }
            const int sw_c0 = (p * 8 + lt) ^ (g << 2), sw_c1 = (p * 8 + 4 + lt) ^ (g << 2);
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                uint32_t a0 = sA_main[m_lo][sw_c0], a1 = sA_main[m_hi][sw_c0];
                uint32_t a2 = sA_main[m_lo][sw_c1], a3 = sA_main[m_hi][sw_c1];
                uint32_t sfa = sAS[m_warp_base + mi * 16 + m_sf_off][p];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn)
                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
            }
        }
        {
            uint32_t u0[4] = {w0_cur.x, w0_cur.y, w0_cur.z, w0_cur.w};
            uint32_t b0[4], sfb[4];
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                b0[sn] = prmt_b32(__shfl_sync(0xFFFFFFFFu, u0[sn], src0), __shfl_sync(0xFFFFFFFFu, u0[sn], src1), prmt_sel);
                sfb[sn] = sWS[wp * 32 + sn * 8 + g][4];
            }
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                uint32_t a0 = sA_tail[m_lo][lt], a1 = sA_tail[m_hi][lt];
                uint32_t sfa = sAS[m_warp_base + mi * 16 + m_sf_off][4];
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn)
                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, 0u, 0u, b0[sn], 0u, sfa, sfb[sn]);
            }
        }
        w2_coalesced_epilogue_v3<BM, COMBINE_MODE, 1>(
            sEpi, acc, sRS, sTok, scale_e, Out, row0, end, nb, warp, wm, wp, lane, g, lt
        );
    }
}

// =============================================================================
// Template Dispatchers (Dynamic SMEM Carveout + Optional Per-Slice L2 Zeroing)
// =============================================================================

template <typename KernelFn>
inline void configure_dyn_smem(KernelFn fn, size_t smem_bytes) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

template <int BM>
int launch_nvfp4_w13_ws5_persist_bm(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int use_r2t, int as_row_major, int has_patches, int w_evict_mode,
    cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    dim3 grid(6, eff_work);
    dim3 block(256);
    constexpr size_t SMEM = sizeof(W13Ws5SmemBlock<BM>);
    const int key = ((use_r2t ? 1 : 0) << 2) | ((as_row_major ? 1 : 0) << 1) | (has_patches ? 1 : 0);
    #define DISPATCH_P_W13_WS5(R2T, RM, HP) \
        do { \
            static bool _cfg = false; \
            if (!_cfg) { \
                configure_dyn_smem(pf_nvfp4_w13_ws5_persistent_kernel<BM, R2T, RM, HP>, SMEM); \
                _cfg = true; \
            } \
            pf_nvfp4_w13_ws5_persistent_kernel<BM, R2T, RM, HP><<<grid, block, SMEM, stream>>>( \
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint8_t*>(WS_BASE), \
                reinterpret_cast<const int32_t*>(P_OFF), reinterpret_cast<const uint32_t*>(PATCHES), \
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                reinterpret_cast<const int32_t*>(PlanBuf), reinterpret_cast<uint32_t*>(Out), \
                max_ctas, tile, num_experts, eff_work, w_evict_mode); \
        } while (0)
    switch (key) {
        case 0: DISPATCH_P_W13_WS5(0, 0, 0); break;
        case 1: DISPATCH_P_W13_WS5(0, 0, 1); break;
        case 2: DISPATCH_P_W13_WS5(0, 1, 0); break;
        case 3: DISPATCH_P_W13_WS5(0, 1, 1); break;
        case 4: DISPATCH_P_W13_WS5(1, 0, 0); break;
        case 5: DISPATCH_P_W13_WS5(1, 0, 1); break;
        case 6: DISPATCH_P_W13_WS5(1, 1, 0); break;
        case 7: DISPATCH_P_W13_WS5(1, 1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_P_W13_WS5
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int BM>
int launch_nvfp4_w13_panel_persist_bm(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int use_r2t, int as_row_major, int ws6, int w_evict_mode,
    cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    dim3 grid(6, eff_work);
    dim3 block(256);
    constexpr size_t SMEM = sizeof(W13PanelSmemBlock<BM>);
    const int key = ((use_r2t ? 1 : 0) << 2) | ((as_row_major ? 1 : 0) << 1) | (ws6 ? 1 : 0);
    #define DISPATCH_P_W13_PAN(R2T, RM, W6) \
        do { \
            static bool _cfg = false; \
            if (!_cfg) { \
                configure_dyn_smem(pf_nvfp4_w13_panel_persistent_kernel<BM, R2T, RM, W6>, SMEM); \
                _cfg = true; \
            } \
            pf_nvfp4_w13_panel_persistent_kernel<BM, R2T, RM, W6><<<grid, block, SMEM, stream>>>( \
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                reinterpret_cast<const uint8_t*>(WP), reinterpret_cast<const uint8_t*>(WSP), \
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                reinterpret_cast<const int32_t*>(PlanBuf), reinterpret_cast<uint32_t*>(Out), \
                max_ctas, tile, num_experts, eff_work, w_evict_mode); \
        } while (0)
    switch (key) {
        case 0: DISPATCH_P_W13_PAN(0, 0, 0); break;
        case 1: DISPATCH_P_W13_PAN(0, 0, 1); break;
        case 2: DISPATCH_P_W13_PAN(0, 1, 0); break;
        case 3: DISPATCH_P_W13_PAN(0, 1, 1); break;
        case 4: DISPATCH_P_W13_PAN(1, 0, 0); break;
        case 5: DISPATCH_P_W13_PAN(1, 0, 1); break;
        case 6: DISPATCH_P_W13_PAN(1, 1, 0); break;
        case 7: DISPATCH_P_W13_PAN(1, 1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_P_W13_PAN
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int BM>
int launch_mxfp8_w13_persist_bm(
    const void* A, const void* AS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int use_r2t, int as_row_major, int w_evict_mode,
    cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    dim3 grid(6, eff_work);
    dim3 block(256);
    constexpr size_t SMEM = sizeof(MxW13SmemBlock<BM>);
    const int key = ((use_r2t ? 1 : 0) << 1) | (as_row_major ? 1 : 0);
    #define DISPATCH_P_MX_W13(R2T, RM) \
        do { \
            static bool _cfg = false; \
            if (!_cfg) { \
                configure_dyn_smem(pf_mxfp8_w13_persistent_kernel<BM, R2T, RM>, SMEM); \
                _cfg = true; \
            } \
            pf_mxfp8_w13_persistent_kernel<BM, R2T, RM><<<grid, block, SMEM, stream>>>( \
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                reinterpret_cast<const int32_t*>(Row2Tok), \
                reinterpret_cast<const uint8_t*>(WP), reinterpret_cast<const uint8_t*>(WSP), \
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                reinterpret_cast<const int32_t*>(PlanBuf), reinterpret_cast<uint32_t*>(Out), \
                max_ctas, tile, num_experts, eff_work, w_evict_mode); \
        } while (0)
    switch (key) {
        case 0: DISPATCH_P_MX_W13(0, 0); break;
        case 1: DISPATCH_P_MX_W13(0, 1); break;
        case 2: DISPATCH_P_MX_W13(1, 0); break;
        case 3: DISPATCH_P_MX_W13(1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_P_MX_W13
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int COMBINE_MODE, int PANELS>
int launch_nvfp4_w2_ws5_persist_cm_p(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride, int has_patches,
    int w_evict_mode, int l2_zero_slice, int tokens,
    cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    constexpr int N_BLKS = 36 / PANELS;
    dim3 block(256);
    const bool do_slice = (COMBINE_MODE == 2 && l2_zero_slice != 0 && tokens > 0);
    const int key = ((as_row_major ? 1 : 0) << 1) | (has_patches ? 1 : 0);

    #define DISPATCH_P_W2_WS5(RM, HP) \
        do { \
            static bool _carved = false; \
            if (!_carved) { \
                cudaFuncSetAttribute(pf_nvfp4_w2_ws5_persistent_kernel<RM, HP, COMBINE_MODE, PANELS>, \
                                     cudaFuncAttributePreferredSharedMemoryCarveout, \
                                     cudaSharedmemCarveoutMaxShared); \
                _carved = true; \
            } \
            if (do_slice) { \
                dim3 zgrid((tokens + 7) / 8); \
                dim3 wgrid(eff_work, 1); \
                for (int nb_blk = 0; nb_blk < N_BLKS; ++nb_blk) { \
                    zero_bf16_slice_wb_kernel<PANELS><<<zgrid, block, 0, stream>>>( \
                        reinterpret_cast<uint16_t*>(Out), tokens, nb_blk); \
                    pf_nvfp4_w2_ws5_persistent_kernel<RM, HP, COMBINE_MODE, PANELS><<<wgrid, block, 0, stream>>>( \
                        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                        reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                        reinterpret_cast<const uint8_t*>(W_PACK), reinterpret_cast<const uint32_t*>(WS5_PACK), \
                        reinterpret_cast<const uint8_t*>(WS_BASE), reinterpret_cast<const int32_t*>(P_OFF), \
                        reinterpret_cast<const uint32_t*>(PATCHES), reinterpret_cast<const float*>(Ones), \
                        reinterpret_cast<const int32_t*>(Tiles), reinterpret_cast<const int32_t*>(PlanBuf), \
                        Out, max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, nb_blk); \
                } \
            } else { \
                dim3 grid = (COMBINE_MODE == 0) ? dim3(N_BLKS, eff_work) : dim3(eff_work, N_BLKS); \
                pf_nvfp4_w2_ws5_persistent_kernel<RM, HP, COMBINE_MODE, PANELS><<<grid, block, 0, stream>>>( \
                    reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                    reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                    reinterpret_cast<const uint8_t*>(W_PACK), reinterpret_cast<const uint32_t*>(WS5_PACK), \
                    reinterpret_cast<const uint8_t*>(WS_BASE), reinterpret_cast<const int32_t*>(P_OFF), \
                    reinterpret_cast<const uint32_t*>(PATCHES), reinterpret_cast<const float*>(Ones), \
                    reinterpret_cast<const int32_t*>(Tiles), reinterpret_cast<const int32_t*>(PlanBuf), \
                    Out, max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, -1); \
            } \
        } while (0)

    switch (key) {
        case 0: DISPATCH_P_W2_WS5(0, 0); break;
        case 1: DISPATCH_P_W2_WS5(0, 1); break;
        case 2: DISPATCH_P_W2_WS5(1, 0); break;
        case 3: DISPATCH_P_W2_WS5(1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_P_W2_WS5
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int COMBINE_MODE, int PANELS>
int launch_nvfp4_w2_panel_persist_cm_p(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride,
    int w_evict_mode, int l2_zero_slice, int tokens,
    cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    constexpr int N_BLKS = 36 / PANELS;
    dim3 block(256);
    const bool do_slice = (COMBINE_MODE == 2 && l2_zero_slice != 0 && tokens > 0);

    #define DISPATCH_P_W2_PAN(RM) \
        do { \
            static bool _carved = false; \
            if (!_carved) { \
                cudaFuncSetAttribute(pf_nvfp4_w2_panel_persistent_kernel<RM, COMBINE_MODE, PANELS>, \
                                     cudaFuncAttributePreferredSharedMemoryCarveout, \
                                     cudaSharedmemCarveoutMaxShared); \
                _carved = true; \
            } \
            if (do_slice) { \
                dim3 zgrid((tokens + 7) / 8); \
                dim3 wgrid(eff_work, 1); \
                for (int nb_blk = 0; nb_blk < N_BLKS; ++nb_blk) { \
                    zero_bf16_slice_wb_kernel<PANELS><<<zgrid, block, 0, stream>>>( \
                        reinterpret_cast<uint16_t*>(Out), tokens, nb_blk); \
                    pf_nvfp4_w2_panel_persistent_kernel<RM, COMBINE_MODE, PANELS><<<wgrid, block, 0, stream>>>( \
                        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                        reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                        reinterpret_cast<const uint8_t*>(WP), reinterpret_cast<const uint8_t*>(WSP), \
                        reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                        reinterpret_cast<const int32_t*>(PlanBuf), Out, \
                        max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, nb_blk); \
                } \
            } else { \
                dim3 grid = (COMBINE_MODE == 0) ? dim3(N_BLKS, eff_work) : dim3(eff_work, N_BLKS); \
                pf_nvfp4_w2_panel_persistent_kernel<RM, COMBINE_MODE, PANELS><<<grid, block, 0, stream>>>( \
                    reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                    reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                    reinterpret_cast<const uint8_t*>(WP), reinterpret_cast<const uint8_t*>(WSP), \
                    reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                    reinterpret_cast<const int32_t*>(PlanBuf), Out, \
                    max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, -1); \
            } \
        } while (0)

    if (as_row_major) { DISPATCH_P_W2_PAN(1); }
    else              { DISPATCH_P_W2_PAN(0); }
    #undef DISPATCH_P_W2_PAN
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int COMBINE_MODE, int PANELS>
int launch_mxfp8_w2_persist_cm_p(
    const void* A, const void* AS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride,
    int w_evict_mode, int l2_zero_slice, int tokens,
    cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    constexpr int N_BLKS = 36 / PANELS;
    constexpr size_t SMEM = sizeof(W2Mxfp8PersistSmem);
    dim3 block(256);
    const bool do_slice = (COMBINE_MODE == 2 && l2_zero_slice != 0 && tokens > 0);

    #define DISPATCH_P_MX_W2(RM) \
        do { \
            static bool _cfg = false; \
            if (!_cfg) { \
                configure_dyn_smem(pf_mxfp8_w2_persistent_kernel<RM, COMBINE_MODE, PANELS>, SMEM); \
                _cfg = true; \
            } \
            if (do_slice) { \
                dim3 zgrid((tokens + 7) / 8); \
                dim3 wgrid(eff_work, 1); \
                for (int nb_blk = 0; nb_blk < N_BLKS; ++nb_blk) { \
                    zero_bf16_slice_wb_kernel<PANELS><<<zgrid, block, 0, stream>>>( \
                        reinterpret_cast<uint16_t*>(Out), tokens, nb_blk); \
                    pf_mxfp8_w2_persistent_kernel<RM, COMBINE_MODE, PANELS><<<wgrid, block, SMEM, stream>>>( \
                        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                        reinterpret_cast<const int32_t*>(Row2Tok), \
                        reinterpret_cast<const uint8_t*>(WP), reinterpret_cast<const uint8_t*>(WSP), \
                        reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                        reinterpret_cast<const int32_t*>(PlanBuf), Out, \
                        max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, nb_blk); \
                } \
            } else { \
                dim3 grid = (COMBINE_MODE == 0) ? dim3(N_BLKS, eff_work) : dim3(eff_work, N_BLKS); \
                pf_mxfp8_w2_persistent_kernel<RM, COMBINE_MODE, PANELS><<<grid, block, SMEM, stream>>>( \
                    reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                    reinterpret_cast<const int32_t*>(Row2Tok), \
                    reinterpret_cast<const uint8_t*>(WP), reinterpret_cast<const uint8_t*>(WSP), \
                    reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                    reinterpret_cast<const int32_t*>(PlanBuf), Out, \
                    max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, -1); \
            } \
        } while (0)

    if (as_row_major) { DISPATCH_P_MX_W2(1); }
    else              { DISPATCH_P_MX_W2(0); }
    #undef DISPATCH_P_MX_W2
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int BM>
int launch_nvfp4_w13_ws5_v3_compat_bm(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, cudaStream_t stream
) {
    if (tile % BM != 0) return -2;
    const int sub_m = tile / BM;
    dim3 grid(6 * sub_m, max_ctas);
    dim3 block(256);
    const int key = ((use_r2t ? 1 : 0) << 2) | ((as_row_major ? 1 : 0) << 1) | (has_patches ? 1 : 0);
    #define DISPATCH_C_W13(R2T, RM, HP) \
        do { \
            static bool _carved = false; \
            if (!_carved) { \
                cudaFuncSetAttribute(pf_nvfp4_w13_ws5_v3_compat_kernel<BM, R2T, RM, HP>, \
                                     cudaFuncAttributePreferredSharedMemoryCarveout, \
                                     cudaSharedmemCarveoutMaxShared); \
                _carved = true; \
            } \
            pf_nvfp4_w13_ws5_v3_compat_kernel<BM, R2T, RM, HP><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint8_t*>(WS_BASE), \
                reinterpret_cast<const int32_t*>(P_OFF), reinterpret_cast<const uint32_t*>(PATCHES), \
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                reinterpret_cast<uint32_t*>(Out), max_ctas, tile, w_evict_mode); \
        } while (0)
    switch (key) {
        case 0: DISPATCH_C_W13(0, 0, 0); break;
        case 1: DISPATCH_C_W13(0, 0, 1); break;
        case 2: DISPATCH_C_W13(0, 1, 0); break;
        case 3: DISPATCH_C_W13(0, 1, 1); break;
        case 4: DISPATCH_C_W13(1, 0, 0); break;
        case 5: DISPATCH_C_W13(1, 0, 1); break;
        case 6: DISPATCH_C_W13(1, 1, 0); break;
        case 7: DISPATCH_C_W13(1, 1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_C_W13
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int BM, int COMBINE_MODE, int PANELS>
int launch_nvfp4_w2_ws5_v3_compat_bm_cm_p(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int as_row_major, int as_row_stride, int has_patches,
    int w_evict_mode, cudaStream_t stream
) {
    if (tile % BM != 0) return -2;
    const int sub_m = tile / BM;
    constexpr int N_BLKS = 36 / PANELS;
    dim3 grid = (COMBINE_MODE == 0) ? dim3(N_BLKS * sub_m, max_ctas) : dim3(max_ctas * sub_m, N_BLKS);
    dim3 block(256);
    const int key = ((as_row_major ? 1 : 0) << 1) | (has_patches ? 1 : 0);
    #define DISPATCH_C_W2(RM, HP) \
        do { \
            static bool _carved = false; \
            if (!_carved) { \
                cudaFuncSetAttribute(pf_nvfp4_w2_ws5_v3_compat_kernel<BM, RM, HP, COMBINE_MODE, PANELS>, \
                                     cudaFuncAttributePreferredSharedMemoryCarveout, \
                                     cudaSharedmemCarveoutMaxShared); \
                _carved = true; \
            } \
            pf_nvfp4_w2_ws5_v3_compat_kernel<BM, RM, HP, COMBINE_MODE, PANELS><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                reinterpret_cast<const uint8_t*>(W_PACK), reinterpret_cast<const uint32_t*>(WS5_PACK), \
                reinterpret_cast<const uint8_t*>(WS_BASE), reinterpret_cast<const int32_t*>(P_OFF), \
                reinterpret_cast<const uint32_t*>(PATCHES), reinterpret_cast<const float*>(Ones), \
                reinterpret_cast<const int32_t*>(Tiles), Out, max_ctas, tile, as_row_stride, w_evict_mode); \
        } while (0)
    switch (key) {
        case 0: DISPATCH_C_W2(0, 0); break;
        case 1: DISPATCH_C_W2(0, 1); break;
        case 2: DISPATCH_C_W2(1, 0); break;
        case 3: DISPATCH_C_W2(1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_C_W2
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // namespace

// =============================================================================
// C ABI Entrypoints (All 14 legacy/v3 symbols + 8 new v4 Expert-Persistent APIs)
// =============================================================================

extern "C" {

int pf_build_expert_plan(
    const void* Tiles,
    void* PlanBuf,
    int max_ctas,
    int tile,
    int num_experts,
    int split_m,
    int max_work,
    uint64_t stream_u64
) {
    if (num_experts <= 0 || max_work <= 0) return -2;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    pf_build_expert_plan_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const int32_t*>(Tiles),
        reinterpret_cast<int32_t*>(PlanBuf),
        max_ctas, tile, num_experts, split_m, max_work
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int pf_nvfp4_w13_ws5_native_v4(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int bm, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (bm == 256) {
        return launch_nvfp4_w13_ws5_persist_bm<256>(
            A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, has_patches, w_evict_mode, st);
    }
    if (bm == 128) {
        return launch_nvfp4_w13_ws5_persist_bm<128>(
            A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, has_patches, w_evict_mode, st);
    }
    if (bm == 64) {
        return launch_nvfp4_w13_ws5_persist_bm<64>(
            A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, has_patches, w_evict_mode, st);
    }
    return -4;
}

int pf_nvfp4_w13_panel_native_v4(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int bm, int use_r2t, int as_row_major, int ws6,
    int w_evict_mode, uint64_t stream_u64
) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (bm == 256) {
        return launch_nvfp4_w13_panel_persist_bm<256>(
            A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, ws6, w_evict_mode, st);
    }
    if (bm == 128) {
        return launch_nvfp4_w13_panel_persist_bm<128>(
            A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, ws6, w_evict_mode, st);
    }
    if (bm == 64) {
        return launch_nvfp4_w13_panel_persist_bm<64>(
            A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, ws6, w_evict_mode, st);
    }
    return -4;
}

int pf_mxfp8_w13_native_v4(
    const void* A, const void* AS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int bm, int use_r2t, int as_row_major, int w_evict_mode,
    uint64_t stream_u64
) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (bm == 256) {
        return launch_mxfp8_w13_persist_bm<256>(
            A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, w_evict_mode, st);
    }
    if (bm == 128) {
        return launch_mxfp8_w13_persist_bm<128>(
            A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, w_evict_mode, st);
    }
    if (bm == 64) {
        return launch_mxfp8_w13_persist_bm<64>(
            A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, use_r2t, as_row_major, w_evict_mode, st);
    }
    return -4;
}

int pf_nvfp4_w2_ws5_native_v4(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride, int has_patches,
    int combine_mode, int w_evict_mode, int w2_panels,
    int l2_zero_slice, int tokens, uint64_t stream_u64
) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (combine_mode == 0) {
        return launch_nvfp4_w2_ws5_persist_cm_p<0, 6>(
            A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 1) {
        return launch_nvfp4_w2_ws5_persist_cm_p<1, 2>(
            A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 2) {
        if (w2_panels == 2) return launch_nvfp4_w2_ws5_persist_cm_p<2, 2>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches, w_evict_mode, l2_zero_slice, tokens, st);
        if (w2_panels == 3) return launch_nvfp4_w2_ws5_persist_cm_p<2, 3>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches, w_evict_mode, l2_zero_slice, tokens, st);
        if (w2_panels == 6) return launch_nvfp4_w2_ws5_persist_cm_p<2, 6>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches, w_evict_mode, l2_zero_slice, tokens, st);
        return launch_nvfp4_w2_ws5_persist_cm_p<2, 4>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches, w_evict_mode, l2_zero_slice, tokens, st);
    }
    return -4;
}

int pf_nvfp4_w2_panel_native_v4(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride,
    int combine_mode, int w_evict_mode, int w2_panels,
    int l2_zero_slice, int tokens, uint64_t stream_u64
) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (combine_mode == 0) {
        return launch_nvfp4_w2_panel_persist_cm_p<0, 6>(
            A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 1) {
        return launch_nvfp4_w2_panel_persist_cm_p<1, 2>(
            A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 2) {
        if (w2_panels == 2) return launch_nvfp4_w2_panel_persist_cm_p<2, 2>(A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
        if (w2_panels == 3) return launch_nvfp4_w2_panel_persist_cm_p<2, 3>(A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
        if (w2_panels == 6) return launch_nvfp4_w2_panel_persist_cm_p<2, 6>(A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
        return launch_nvfp4_w2_panel_persist_cm_p<2, 4>(A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
    }
    return -4;
}

int pf_mxfp8_w2_native_v4(
    const void* A, const void* AS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride,
    int combine_mode, int w_evict_mode, int w2_panels,
    int l2_zero_slice, int tokens, uint64_t stream_u64
) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (combine_mode == 0) {
        return launch_mxfp8_w2_persist_cm_p<0, 6>(
            A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 1) {
        return launch_mxfp8_w2_persist_cm_p<1, 2>(
            A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 2) {
        if (w2_panels == 2) return launch_mxfp8_w2_persist_cm_p<2, 2>(A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
        if (w2_panels == 3) return launch_mxfp8_w2_persist_cm_p<2, 3>(A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
        if (w2_panels == 6) return launch_mxfp8_w2_persist_cm_p<2, 6>(A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
        return launch_mxfp8_w2_persist_cm_p<2, 4>(A, AS, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, l2_zero_slice, tokens, st);
    }
    return -4;
}

int pf_zero_slice_wb_bf16(
    void* Out_bf16, int tokens, int nb_blk, int panels, uint64_t stream_u64
) {
    if (tokens <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    dim3 grid((tokens + 7) / 8);
    dim3 block(256);
    if (panels == 2)      zero_bf16_slice_wb_kernel<2><<<grid, block, 0, st>>>(reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    else if (panels == 3) zero_bf16_slice_wb_kernel<3><<<grid, block, 0, st>>>(reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    else if (panels == 4) zero_bf16_slice_wb_kernel<4><<<grid, block, 0, st>>>(reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    else if (panels == 6) zero_bf16_slice_wb_kernel<6><<<grid, block, 0, st>>>(reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    else return -4;
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// =============================================================================
// Backwards-Compatible v3 / Legacy C ABI Entrypoints (14 symbols)
// =============================================================================
// When INKLING_PF_MOE_PERSISTENT != 0 (default ON in v4), transparently upgrades
// zero-patch callers to Expert-Persistent v4 (E=384, BM=256 W13 when tile>=256,
// W_EVICT=1 cs_last, register-resident W2)! Set INKLING_PF_MOE_PERSISTENT=0 for
// exact v3 non-persistent baseline on the same .so!

int pf_nvfp4_w13_ws5_native_v3(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    if (env_persistent_default()) {
        const int eff_bm = (tile >= 256 && bm == 128) ? 256 : bm;
        const int eff_ev = env_v4_w_evict_override(w_evict_mode);
        return pf_nvfp4_w13_ws5_native_v4(
            A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, nullptr, Out,
            max_ctas, tile, 384, 384, eff_bm, use_r2t, as_row_major, has_patches, eff_ev, stream_u64);
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (bm == 128) {
        return launch_nvfp4_w13_ws5_v3_compat_bm<128>(
            A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out,
            max_ctas, tile, use_r2t, as_row_major, has_patches, w_evict_mode, st);
    }
    if (bm == 64) {
        return launch_nvfp4_w13_ws5_v3_compat_bm<64>(
            A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out,
            max_ctas, tile, use_r2t, as_row_major, has_patches, w_evict_mode, st);
    }
    return -4;
}

int pf_nvfp4_w13_ws5_native(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int use_r2t, int as_row_major, int has_patches,
    uint64_t stream_u64
) {
    return pf_nvfp4_w13_ws5_native_v3(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out,
        max_ctas, tile, bm, use_r2t, as_row_major, has_patches, 1, stream_u64);
}

int pf_nvfp4_w13_panel_native(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int use_r2t, int as_row_major, int ws6,
    uint64_t stream_u64
) {
    const int eff_bm = (tile >= 256 && bm == 128) ? 256 : bm;
    return pf_nvfp4_w13_panel_native_v4(
        A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, nullptr, Out,
        max_ctas, tile, 384, 384, eff_bm, use_r2t, as_row_major, ws6, 1, stream_u64);
}

int pf_nvfp4_w2_ws5_native_v3(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int as_row_major, int as_row_stride,
    int has_patches, int combine_mode, int w_evict_mode, int w2_panels,
    uint64_t stream_u64
) {
    if (env_persistent_default()) {
        const int eff_ev = env_v4_w_evict_override(w_evict_mode);
        return pf_nvfp4_w2_ws5_native_v4(
            A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, nullptr, Out,
            max_ctas, tile, 384, 384, as_row_major, as_row_stride, has_patches,
            combine_mode, eff_ev, w2_panels, 0, 0, stream_u64);
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (bm == 128) {
        if (combine_mode == 0) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<128, 0, 6>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
        if (combine_mode == 1) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<128, 1, 2>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
        if (combine_mode == 2) {
            if (w2_panels == 2) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<128, 2, 2>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
            if (w2_panels == 3) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<128, 2, 3>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
            return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<128, 2, 4>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
        }
    } else if (bm == 64) {
        if (combine_mode == 0) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<64, 0, 6>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
        if (combine_mode == 1) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<64, 1, 2>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
        if (combine_mode == 2) {
            if (w2_panels == 2) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<64, 2, 2>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
            if (w2_panels == 3) return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<64, 2, 3>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
            return launch_nvfp4_w2_ws5_v3_compat_bm_cm_p<64, 2, 4>(A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out, max_ctas, tile, as_row_major, as_row_stride, has_patches, w_evict_mode, st);
        }
    }
    return -4;
}

int pf_nvfp4_w2_ws5_native(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int as_row_major, int as_row_stride,
    int has_patches, int combine_mode, uint64_t stream_u64
) {
    return pf_nvfp4_w2_ws5_native_v3(
        A, AS, RS, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, Out,
        max_ctas, tile, bm, as_row_major, as_row_stride, has_patches, combine_mode, 1, 4, stream_u64);
}

int pf_nvfp4_w2_panel_native_v3(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int as_row_major, int as_row_stride,
    int combine_mode, int w_evict_mode, int w2_panels, uint64_t stream_u64
) {
    (void)bm;
    const int eff_ev = env_v4_w_evict_override(w_evict_mode);
    return pf_nvfp4_w2_panel_native_v4(
        A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, nullptr, Out,
        max_ctas, tile, 384, 384, as_row_major, as_row_stride,
        combine_mode, eff_ev, w2_panels, 0, 0, stream_u64);
}

int pf_nvfp4_w2_panel_native(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int as_row_major, int as_row_stride,
    int combine_mode, uint64_t stream_u64
) {
    return pf_nvfp4_w2_panel_native_v3(
        A, AS, RS, Row2Tok, WP, WSP, Ones, Tiles, Out,
        max_ctas, tile, bm, as_row_major, as_row_stride, combine_mode, 1, 4, stream_u64);
}

int pf_mxfp8_w13_native(
    const void* A, const void* AS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int use_r2t, int as_row_major,
    uint64_t stream_u64
) {
    const int eff_bm = (tile >= 256 && bm == 128) ? 256 : bm;
    return pf_mxfp8_w13_native_v4(
        A, AS, Row2Tok, WP, WSP, Ones, Tiles, nullptr, Out,
        max_ctas, tile, 384, 384, eff_bm, use_r2t, as_row_major, 1, stream_u64);
}

int pf_mxfp8_w2_native_v3(
    const void* A, const void* AS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int as_row_major, int as_row_stride,
    int combine_mode, int w_evict_mode, int w2_panels, uint64_t stream_u64
) {
    (void)bm;
    const int eff_ev = env_v4_w_evict_override(w_evict_mode);
    return pf_mxfp8_w2_native_v4(
        A, AS, Row2Tok, WP, WSP, Ones, Tiles, nullptr, Out,
        max_ctas, tile, 384, 384, as_row_major, as_row_stride,
        combine_mode, eff_ev, w2_panels, 0, 0, stream_u64);
}

int pf_mxfp8_w2_native(
    const void* A, const void* AS, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, void* Out,
    int max_ctas, int tile, int bm, int as_row_major, int as_row_stride,
    int combine_mode, uint64_t stream_u64
) {
    return pf_mxfp8_w2_native_v3(
        A, AS, Row2Tok, WP, WSP, Ones, Tiles, Out,
        max_ctas, tile, bm, as_row_major, as_row_stride, combine_mode, 1, 4, stream_u64);
}

int pf_memset_zero(void* ptr, int64_t nbytes, uint64_t stream_u64) {
    if (nbytes <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    cudaError_t err = cudaMemsetAsync(ptr, 0, static_cast<size_t>(nbytes), st);
    return (err == cudaSuccess) ? 0 : -1;
}

int pf_zero_cs_bf16(void* Out_bf16, int64_t numel, uint64_t stream_u64) {
    if (numel <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    int64_t threads = (numel + 31) / 32;
    int blocks = static_cast<int>((threads + 255) / 256);
    zero_bf16_cs_kernel<<<blocks, 256, 0, st>>>(reinterpret_cast<uint16_t*>(Out_bf16), numel);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int pf_fp32_to_bf16_zero(void* Y_fp32, void* Out_bf16, int64_t numel, uint64_t stream_u64) {
    if (numel <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    int64_t threads = (numel + 7) / 8;
    int blocks = static_cast<int>((threads + 255) / 256);
    fp32_to_bf16_and_zero_kernel<<<blocks, 256, 0, st>>>(
        reinterpret_cast<float*>(Y_fp32), reinterpret_cast<uint16_t*>(Out_bf16), numel);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int pf_build_row2tok(const void* Src2Dst, void* Row2Tok, int32_t tokens, int32_t topk, uint64_t stream_u64) {
    int32_t total = tokens * topk;
    if (total <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    int blocks = (total + 255) / 256;
    build_row2tok_kernel<<<blocks, 256, 0, st>>>(
        reinterpret_cast<const int32_t*>(Src2Dst), reinterpret_cast<int32_t*>(Row2Tok), total, topk);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // extern "C"
