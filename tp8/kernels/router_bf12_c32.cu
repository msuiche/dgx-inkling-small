// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) C=32 (B=32, M=17..256) 0-ULP Bit-Exact Router + Zero-Scatter Quantizer
//
// Targets Inkling-2-Small TP8 48L MoE Router + Input Quantization (E=384, K=4608, topk=8):
//   1. `router_bf12_gemm_c32_kernel`:
//      - Reads the EXACT existing `router_bf12` 12-bit lossless slabs (2.654 MB vs 3.539 MB BF16,
//        -25.0% DRAM bytes = -42.5 MB/step across 48L, ZERO extra VRAM!).
//      - SMEM-tiled `sX[5][32*136]` (42.5 KiB, 2 CTAs/SM, 0-way bank-conflict-free LDS.32) with
//        `EW_PER_CTA = 1 (M<=32, 48 CTAs)`, `2 (M<=64, 48 CTAs)`, `4 (M<=256, 36..96 CTAs)`:
//        cuts L2 `X` traffic 4x at M=256 and uses `ld.ca` for multi-M-block weight L2 reuse!
//      - Exact ascending `pk=0..4` FP32 `add_rn` reduction (literal 0-ULP vs `router_bf12_s8`!).
//   2. `c32_quant_topk_kernel` (Phase A, Grid=(T,1,1), Block=(288,1,1) = 9 warps):
//      - Pre-`gdc_wait` (100% hidden behind Router GEMM via PDL!):
//        Single-pass `X` load into registers -> exact row amax -> 1 thread per NVFP4 16-elem block
//        (288 threads = exact 1 iter, explicit `fmaf` in MAE matching live `_scatter_kernel` SASS!)
//        or 144 threads per MXFP8 32-elem block -> writes compact Zero-Scatter `QValues[tok]` +
//        row-major `QScales[tok]` + `RowAmaxIn[tok]` (eliminates 8x scatter at M=17..128!).
//      - Post-`gdc_wait`: supports BOTH BF16 logits (`LOGITS_BF16=true`, from `k_group_partials`
//        for literal 0-ULP parity with live `patches-s6` on `exact_ink.py` & `bench.py`!) AND
//        FP32 logits (`LOGITS_BF16=false`, from `router_bf12_gemm_c32`!).
//      - Warp-0 8x `redux.sync.max/min.u32` Top-8 + exact `_log_sigmoid` + `_sum8` softmax +
//        global `Counts` atomic + last-CTA 6-warp SMEM prefix scan writing `Starts`,
//        `StartsCompact`, `Tiles`, `TileCompactStart`, and self-resetting `Counts` & `Counters`.
//   3. `c32_place_rows_kernel` (Phase B, Grid=(ceil(T*8/256),1,1), Block=(256,1,1), ~1.2 us):
//      - Writes `Src2Dst`, `Src2Compact`, `Gammas`, `Rows`, `Row2Tok`, `QRowScales`, and
//        pre-zeroes compact `RowAmaxW2[cdst] = 0u` for Fused W13 SiTU -> triggers PDL into W13!
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC router_bf12_c32.cu -o librouter_bf12_c32.so
//   (CRITICAL: --fmad=false required; OMIT --ftz=true to match Triton ftz=False!)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace {

static constexpr int E_FIXED = 384;
static constexpr int K_FIXED = 4608;
static constexpr int TOPK_FIXED = 8;
static constexpr int SLAB_BYTES = 1536;  // 1024 B sm + 512 B e4 (12 x 128 B)
static constexpr float FP32_MAX_VAL = 3.4028234663852886e38f;
static constexpr float INV_6 = 1.0f / 6.0f;
static constexpr float INV_448 = 1.0f / 448.0f;
static constexpr float INV_1536 = 1.0f / 1536.0f;

// ---- PTX Memory & Math Helpers ----

static __device__ __forceinline__ uint4 ld_cs_u4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_ca_u4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint16_t ld_cg_u16(const uint16_t* p) {
    uint16_t v;
    asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

static __device__ __forceinline__ float add_rn(float a, float b) {
    float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float sub_rn(float a, float b) {
    float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float mul_rn(float a, float b) {
    float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float div_rn(float a, float b) {
    float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float rcp_rn(float a) {
    float r; asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
static __device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}
static __device__ __forceinline__ uint32_t redux_max_u32(uint32_t v) {
    uint32_t r; asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v)); return r;
}
static __device__ __forceinline__ uint32_t redux_min_u32(uint32_t v) {
    uint32_t r; asm("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v)); return r;
}

// ---- BF12 Slab Load & Unpack ----

struct Slab12 { uint4 sm0, sm1, e4; };

template <bool CACHE_WEIGHTS>
static __device__ __forceinline__ Slab12 load_slab12(const uint8_t* sp, int lane) {
    const uint4* p = reinterpret_cast<const uint4*>(sp);
    Slab12 r;
    if constexpr (CACHE_WEIGHTS) {
        r.sm0 = ld_ca_u4(p + lane);
        r.sm1 = ld_ca_u4(p + 32 + lane);
        r.e4  = ld_ca_u4(p + 64 + lane);
    } else {
        r.sm0 = ld_cs_u4(p + lane);
        r.sm1 = ld_cs_u4(p + 32 + lane);
        r.e4  = ld_cs_u4(p + 64 + lane);
    }
    return r;
}

static __device__ __forceinline__ void unpack4_bf12(
    uint32_t sm4, uint32_t e16, uint32_t base_pair,
    uint32_t& b0, uint32_t& b1
) {
    uint32_t d0 = (e16 & 0x000Fu) | ((e16 & 0x00F0u) << 12);
    uint32_t d1 = ((e16 >> 8) & 0x000Fu) | ((e16 & 0xF000u) << 4);
    uint32_t exp0 = base_pair - d0;
    uint32_t exp1 = base_pair - d1;
    uint32_t s0 = __byte_perm(sm4, 0u, 0x4140u);
    uint32_t s1 = __byte_perm(sm4, 0u, 0x4342u);
    b0 = ((s0 & 0x00800080u) << 8) | (exp0 << 7) | (s0 & 0x007F007Fu);
    b1 = ((s1 & 0x00800080u) << 8) | (exp1 << 7) | (s1 & 0x007F007Fu);
}

// =========================================================================
// Kernel 1 (C32): SMEM-Tiled BF12 Router GEMM (M=17..256, 2.654 MB, -25% bytes)
//   - Grid (48 / EW_PER_CTA, ceil(T / 32), 1), Block (160, 1, 1) (5 warps = pk 0..4)
//   - EW_PER_CTA = 1 (M<=32, 48 CTAs), 2 (M<=64, 48 CTAs), 4 (M<=256, 36..96 CTAs)
//   - Union SMEM = 43,520 B (42.5 KiB -> 2 CTAs/SM, 0-way bank conflicts on LDS.32!)
//   - Exact ascending pk=0..4 FP32 `add_rn` reduction -> literal 0-ULP vs router_bf12_s8!
// =========================================================================
template <int EW_PER_CTA, bool CACHE_WEIGHTS, bool PDL, bool TRIG>
__global__ __launch_bounds__(160, 2)
void router_bf12_gemm_c32_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    const int32_t* __restrict__ NLivePtr,
    float* __restrict__ Logits,  // [T, 384]
    int T,
    uint32_t base_exp
) {
    constexpr int BM_CTA = 32;
    constexpr int SX_STRIDE = 136;  // 136 bf16 = 272 B = 68 banks === 4 mod 32 -> 0-way LDS.32 & 16B aligned!

    const int e_warp_base = static_cast<int>(blockIdx.x) * EW_PER_CTA;
    const int m_base = static_cast<int>(blockIdx.y) * BM_CTA;
    const int tid = threadIdx.x;
    const int pk = tid >> 5;     // 0..4
    const int lane = tid & 31;
    const int g = lane >> 2;     // 0..7
    const int lt = lane & 3;     // 0..3

    const int n_slabs = (pk < 4) ? 8 : 4;
    const int k_lo = pk * 1024;
    const uint32_t base_pair = base_exp | (base_exp << 16);

    int32_t o_cur[EW_PER_CTA];
    int32_t o_end[EW_PER_CTA];
    #pragma unroll
    for (int ew = 0; ew < EW_PER_CTA; ++ew) {
        const int wuid = (e_warp_base + ew) * 5 + pk;
        o_cur[ew] = __ldg(O_OFF + wuid);
        o_end[ew] = __ldg(O_OFF + wuid + 1);
    }

    // Prefetch first slab of ew=0 BEFORE gdc_wait!
    const uint8_t* slab0_ew0 = SLABS + static_cast<int64_t>(e_warp_base * 36 + pk * 8) * SLAB_BYTES;
    if constexpr (PDL) {
        if (lane < 12) prefetch_l2(slab0_ew0 + lane * 128);
        asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (TRIG) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }

    int n_live = T;
    if (NLivePtr != nullptr) {
        int nl = *NLivePtr;
        if (nl < 0) nl = 0;
        if (nl < n_live) n_live = nl;
    }
    int m_live = n_live - m_base;
    if (m_live > BM_CTA) m_live = BM_CTA;
    if (m_live <= 0) return;

    const bool has_m8  = (m_live > 8);
    const bool has_m16 = (m_live > 16);
    const bool has_m24 = (m_live > 24);

    union alignas(16) SmemUnion {
        __nv_bfloat16 sX[5][BM_CTA * SX_STRIDE];
        float         sP[5][BM_CTA][9];
    };
    __shared__ SmemUnion smem;
    __nv_bfloat16* sX_warp = smem.sX[pk];

    float acc[EW_PER_CTA][2][4];
    #pragma unroll
    for (int ew = 0; ew < EW_PER_CTA; ++ew) {
        #pragma unroll
        for (int im = 0; im < 2; ++im) {
            acc[ew][im][0] = 0.0f; acc[ew][im][1] = 0.0f;
            acc[ew][im][2] = 0.0f; acc[ew][im][3] = 0.0f;
        }
    }

    const int x0_0 = g * SX_STRIDE + 2 * lt;
    const int x0_1 = (g + 8) * SX_STRIDE + 2 * lt;
    const int x1_0 = (g + 16) * SX_STRIDE + 2 * lt;
    const int x1_1 = (g + 24) * SX_STRIDE + 2 * lt;

    const int pad_rows = (m_live <= 8) ? 8 : ((m_live <= 16) ? 16 : ((m_live <= 24) ? 24 : 32));
    const int total_u4 = pad_rows * 16;  // 16 uint4s = 128 bf16 per row

    for (int sl = 0; sl < n_slabs; ++sl) {
        // Cooperatively load X[m_base..m_base+pad_rows-1, k_lo + sl*128 .. +127] ONCE into sX_warp
        const int k_sl = k_lo + sl * 128;
        for (int qi = lane; qi < total_u4; qi += 32) {
            const int lm = qi >> 4;
            const int qk = qi & 15;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (lm < m_live) {
                const uint4* xp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(m_base + lm) * K_FIXED + k_sl + (qk << 3)
                );
                v = ld_ca_u4(xp);
            }
            *reinterpret_cast<uint4*>(sX_warp + lm * SX_STRIDE + (qk << 3)) = v;
        }
        __syncwarp();

        #pragma unroll
        for (int ew = 0; ew < EW_PER_CTA; ++ew) {
            const int slab_idx = (e_warp_base + ew) * 36 + pk * 8 + sl;
            const uint8_t* sp = SLABS + static_cast<int64_t>(slab_idx) * SLAB_BYTES;
            Slab12 cur_slab = load_slab12<CACHE_WEIGHTS>(sp, lane);

            int32_t oc = o_cur[ew];
            const int32_t oe = o_end[ew];
            uint64_t o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = static_cast<uint32_t>(o_rec >> 32);

            const uint32_t sm_words[8] = {
                cur_slab.sm0.x, cur_slab.sm0.y, cur_slab.sm0.z, cur_slab.sm0.w,
                cur_slab.sm1.x, cur_slab.sm1.y, cur_slab.sm1.z, cur_slab.sm1.w
            };
            const uint32_t e16_arr[8] = {
                cur_slab.e4.x & 0xFFFFu, cur_slab.e4.x >> 16,
                cur_slab.e4.y & 0xFFFFu, cur_slab.e4.y >> 16,
                cur_slab.e4.z & 0xFFFFu, cur_slab.e4.z >> 16,
                cur_slab.e4.w & 0xFFFFu, cur_slab.e4.w >> 16
            };

            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                uint32_t b0, b1;
                unpack4_bf12(sm_words[s], e16_arr[s], base_pair, b0, b1);

                const uint32_t cur_step = static_cast<uint32_t>((sl << 3) | s);
                if (__builtin_expect(o_next_step == cur_step, 0)) {
                    while (oc < oe && o_next_step == cur_step) {
                        uint32_t lj = static_cast<uint32_t>(o_rec >> 16) & 0x7Fu;
                        if ((lj >> 2) == static_cast<uint32_t>(lane)) {
                            uint32_t val = static_cast<uint32_t>(o_rec & 0xFFFFu);
                            uint32_t j = lj & 3u;
                            if (j == 0u)      b0 = (b0 & 0xFFFF0000u) | val;
                            else if (j == 1u) b0 = (b0 & 0x0000FFFFu) | (val << 16);
                            else if (j == 2u) b1 = (b1 & 0xFFFF0000u) | val;
                            else              b1 = (b1 & 0x0000FFFFu) | (val << 16);
                        }
                        ++oc;
                        o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
                        o_next_step = static_cast<uint32_t>(o_rec >> 32);
                    }
                }

                const int s_off = s * 16;
                {
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX_warp + x0_0 + s_off);
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX_warp + x0_0 + s_off + 8);
                    const uint32_t a1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_warp + x0_1 + s_off) : 0u;
                    const uint32_t a3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_warp + x0_1 + s_off + 8) : 0u;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[ew][0][0]), "+f"(acc[ew][0][1]), "+f"(acc[ew][0][2]), "+f"(acc[ew][0][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                    );
                }
                if (has_m16) {
                    const uint32_t c0 = *reinterpret_cast<const uint32_t*>(sX_warp + x1_0 + s_off);
                    const uint32_t c2 = *reinterpret_cast<const uint32_t*>(sX_warp + x1_0 + s_off + 8);
                    const uint32_t c1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_warp + x1_1 + s_off) : 0u;
                    const uint32_t c3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_warp + x1_1 + s_off + 8) : 0u;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[ew][1][0]), "+f"(acc[ew][1][1]), "+f"(acc[ew][1][2]), "+f"(acc[ew][1][3])
                        : "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(b0), "r"(b1)
                    );
                }
            }
            o_cur[ew] = oc;
        }
        __syncwarp();
    }

    // All 5 warps finished reading sX; synchronize before reusing union as sP!
    __syncthreads();

    const int n_out = m_live * 8;
    #pragma unroll
    for (int ew = 0; ew < EW_PER_CTA; ++ew) {
        smem.sP[pk][g][2 * lt]     = acc[ew][0][0];
        smem.sP[pk][g][2 * lt + 1] = acc[ew][0][1];
        if (has_m8) {
            smem.sP[pk][g + 8][2 * lt]     = acc[ew][0][2];
            smem.sP[pk][g + 8][2 * lt + 1] = acc[ew][0][3];
        }
        if (has_m16) {
            smem.sP[pk][g + 16][2 * lt]     = acc[ew][1][0];
            smem.sP[pk][g + 16][2 * lt + 1] = acc[ew][1][1];
        }
        if (has_m24) {
            smem.sP[pk][g + 24][2 * lt]     = acc[ew][1][2];
            smem.sP[pk][g + 24][2 * lt + 1] = acc[ew][1][3];
        }
        __syncthreads();

        const int e_col_base = (e_warp_base + ew) * 8;
        for (int oidx = tid; oidx < n_out; oidx += 160) {
            const int lm = oidx >> 3;
            const int e_loc = oidx & 7;
            float l = add_rn(add_rn(add_rn(add_rn(
                        smem.sP[0][lm][e_loc],
                        smem.sP[1][lm][e_loc]),
                        smem.sP[2][lm][e_loc]),
                        smem.sP[3][lm][e_loc]),
                        smem.sP[4][lm][e_loc]);
            Logits[static_cast<int64_t>(m_base + lm) * E_FIXED + e_col_base + e_loc] = l;
        }
        if (ew + 1 < EW_PER_CTA) __syncthreads();
    }
}

// =========================================================================
// Exact 0-ULP Input Quantization Helpers (Matches live _scatter_kernel SASS)
// =========================================================================

static __device__ __forceinline__ void quant_nvfp4_block16_fma(
    const uint4 v0, const uint4 v1,
    float encoder_scale, float decoder_scale, float outer_scale,
    uint64_t& out_packed, uint8_t& out_scale
) {
    float x[16];
    float bmax = 0.0f;
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
        float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
        x[2 * j]     = a;
        x[2 * j + 1] = b;
        bmax = fmaxf(bmax, fmaxf(fabsf(a), fabsf(b)));
    }

    float s6_raw = fminf(mul_rn(bmax, mul_rn(encoder_scale, INV_6)), 448.0f);
    float s4_raw = fminf(mul_rn(bmax, mul_rn(encoder_scale, 0.25f)), 448.0f);

    uint16_t s_e4m3x2;
    uint32_t s_f16x2;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %3, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, %0;"
        : "=&h"(s_e4m3x2), "=r"(s_f16x2)
        : "f"(s6_raw), "f"(s4_raw));

    float scale6 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 & 0xFFFFu)));
    float scale4 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 >> 16)));
    float inv6 = fminf(div_rn(1.0f, mul_rn(scale6, decoder_scale)), FP32_MAX_VAL);
    float inv4 = fminf(div_rn(1.0f, mul_rn(scale4, decoder_scale)), FP32_MAX_VAL);

    uint64_t p6 = 0ULL, p4 = 0ULL;
    float e6_pair[8], e4_pair[8];

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float x0 = x[2 * k];
        float x1 = x[2 * k + 1];
        {
            uint32_t qbyte, h;
            float a = mul_rn(x0, inv6), b = mul_rn(x1, inv6);
            asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
                "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
                : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
            p6 |= (static_cast<uint64_t>(qbyte)) << (8 * k);
            float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
            float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
            // Explicit fmaf matches ptxas 13.0 SLP-vectorized _scatter_kernel & prefill_moe_qf_s8!
            e6_pair[k] = add_rn(fabsf(fmaf(mul_rn(r0, scale6), outer_scale, -x0)),
                                fabsf(fmaf(mul_rn(r1, scale6), outer_scale, -x1)));
        }
        {
            uint32_t qbyte, h;
            float a = mul_rn(x0, inv4), b = mul_rn(x1, inv4);
            asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
                "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
                : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
            p4 |= (static_cast<uint64_t>(qbyte)) << (8 * k);
            float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
            float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
            e4_pair[k] = add_rn(fabsf(fmaf(mul_rn(r0, scale4), outer_scale, -x0)),
                                fabsf(fmaf(mul_rn(r1, scale4), outer_scale, -x1)));
        }
    }

    float err6 = add_rn(add_rn(add_rn(e6_pair[0], e6_pair[1]), add_rn(e6_pair[2], e6_pair[3])),
                        add_rn(add_rn(e6_pair[4], e6_pair[5]), add_rn(e6_pair[6], e6_pair[7])));
    float err4 = add_rn(add_rn(add_rn(e4_pair[0], e4_pair[1]), add_rn(e4_pair[2], e4_pair[3])),
                        add_rn(add_rn(e4_pair[4], e4_pair[5]), add_rn(e4_pair[6], e4_pair[7])));

    bool choose4 = (err4 < err6);
    out_packed = choose4 ? p4 : p6;
    out_scale  = choose4 ? static_cast<uint8_t>(s_e4m3x2 >> 8) : static_cast<uint8_t>(s_e4m3x2 & 0xFFu);
}

static __device__ __forceinline__ void quant_mxfp8_block32(
    const uint4 v[4], uint8_t* dst_vals, uint8_t& out_exp
) {
    float x[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
            float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
            x[i * 8 + j * 2]     = a;
            x[i * 8 + j * 2 + 1] = b;
            amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        }
    }

    float raw_scale = mul_rn(amax, INV_448);
    uint32_t rs_u = __float_as_uint(raw_scale);
    uint32_t exp_u;
    if (raw_scale != raw_scale) exp_u = 255u;
    else if (rs_u <= 0x00400000u) exp_u = 0u;
    else {
        exp_u = ((rs_u + 0x007FFFFFu) & 0x7F800000u) >> 23;
        if (exp_u > 254u) exp_u = 254u;
    }

    float inv;
    if (exp_u == 255u) inv = 0.0f;
    else if (exp_u == 0u) inv = __uint_as_float(0x7F000000u);
    else inv = rcp_rn(__uint_as_float(exp_u << 23));

    uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst_vals);
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        float a = mul_rn(x[2 * k], inv);
        float b = mul_rn(x[2 * k + 1], inv);
        if (a == a) { if (a < -448.0f) a = -448.0f; if (a > 448.0f) a = 448.0f; }
        if (b == b) { if (b < -448.0f) b = -448.0f; if (b > 448.0f) b = 448.0f; }
        uint16_t e4;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e4) : "f"(a), "f"(b));
        dst16[k] = e4;
    }
    out_exp = static_cast<uint8_t>(exp_u);
}

static __device__ __forceinline__ uint32_t selection_bits(float logit, float bias) {
    float sig = div_rn(1.0f, add_rn(1.0f, expf(-logit)));
    float sel = add_rn(sig, bias);
    uint32_t bits = __float_as_uint(sel);
    bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits;
}

// =========================================================================
// Kernel 2A (C32 Phase A): Zero-Scatter Quantize + Top-8 + Last-CTA Prefix Scan
//   - Grid = (T, 1, 1), Block = (288, 1, 1) (9 warps -> exact 288 NVFP4 / 144 MXFP8 blocks!)
//   - Quantizes X[tok] ONCE BEFORE gdc_wait (100% hidden behind Router GEMM!)
//   - Consumes either BF16 (`LOGITS_BF16=true`) or FP32 (`LOGITS_BF16=false`) logits
//   - Last CTA runs 6-warp SMEM prefix scan over 384 experts -> writes Starts, StartsCompact,
//     Tiles, TileCompactStart, and self-resets Counts & Counters!
// =========================================================================
template <bool NVFP4, bool LOGITS_BF16, bool PDL, bool TRIG_NEXT>
__global__ __launch_bounds__(288, 3)
void c32_quant_topk_kernel(
    const __nv_bfloat16* __restrict__ X,
    const void* __restrict__ Logits,         // [T, 384] bf16 if LOGITS_BF16 else fp32
    const float* __restrict__ Bias,          // [384]
    const int32_t* __restrict__ ExpertBounds,
    const bool* __restrict__ Padding,
    const int32_t* __restrict__ NLivePtr,
    int32_t* __restrict__ Counts,            // [num_local] zeroed; self-reset by last CTA
    uint32_t* __restrict__ Counters,         // [>=1] zeroed; self-reset by last CTA
    int32_t* __restrict__ Ids,               // [T * 8]
    int32_t* __restrict__ Ranks,             // [T * 8]
    float* __restrict__ Weights,             // [T * 8]
    int32_t* __restrict__ Starts,            // [num_local] layout start row
    int32_t* __restrict__ StartsCompact,     // [num_local] compact start row
    int32_t* __restrict__ Tiles,             // [2 * max_ctas + 3]
    int32_t* __restrict__ TileCompactStart,  // [max_ctas]
    uint8_t* __restrict__ QValues,           // [T, 2304] NVFP4 or [T, 4608] MXFP8
    uint8_t* __restrict__ QScales,           // [T, 288]  NVFP4 or [T, 144]  MXFP8 (row-major)
    float* __restrict__ RowAmaxIn,           // [T] (NVFP4 only)
    int T,
    float route_scale,
    int num_local,
    int tile_sz,
    int max_ctas
) {
    const int tok = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;  // 0..287
    const int warp = tid >> 5;    // 0..8
    const int lane = tid & 31;    // 0..31

    __shared__ float    sWarpMax[9];
    __shared__ int32_t  sIsLastCta;
    __shared__ int32_t  sWarpT[8];
    __shared__ int32_t  sWarpC[8];

    int n_live = T;
    if (NLivePtr != nullptr) {
        int nl = *NLivePtr;
        if (nl < 0) nl = 0;
        if (nl < n_live) n_live = nl;
    }
    const bool is_pad = (tok >= n_live) || ((Padding != nullptr) ? Padding[tok] : false);
    const int lo = ExpertBounds[0];
    const int hi = ExpertBounds[1];

    // ---- Step 1 (Pre-gdc_wait): Single-pass Zero-Scatter Quantization of X[tok] ----
    if (!is_pad) {
        if constexpr (NVFP4) {
            const uint4* bp = reinterpret_cast<const uint4*>(
                X + static_cast<int64_t>(tok) * K_FIXED + tid * 16
            );
            const uint4 v0 = ld_ca_u4(bp);
            const uint4 v1 = ld_ca_u4(bp + 1);

            float bmax = 0.0f;
            const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
                float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
                bmax = fmaxf(bmax, fmaxf(fabsf(a), fabsf(b)));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFFu, bmax, off));
            }
            if (lane == 0) sWarpMax[warp] = bmax;
            __syncthreads();

            float tok_amax = sWarpMax[0];
            #pragma unroll
            for (int wi = 1; wi < 9; ++wi) tok_amax = fmaxf(tok_amax, sWarpMax[wi]);
            if (tid == 0) RowAmaxIn[tok] = tok_amax;

            const float enc_s = fminf(div_rn(1536.0f, tok_amax), FP32_MAX_VAL);
            const float dec_s = div_rn(1.0f, enc_s);
            const float out_s = mul_rn(tok_amax, INV_1536);

            uint64_t packed;
            uint8_t sc;
            quant_nvfp4_block16_fma(v0, v1, enc_s, dec_s, out_s, packed, sc);
            reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(tok) * 288 + tid] = packed;
            QScales[static_cast<int64_t>(tok) * 288 + tid] = sc;
        } else {
            if (tid < 144) {
                const uint4* bp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(tok) * K_FIXED + tid * 32
                );
                uint4 v[4] = {ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3)};
                uint8_t sc;
                quant_mxfp8_block32(v, QValues + static_cast<int64_t>(tok) * K_FIXED + tid * 32, sc);
                QScales[static_cast<int64_t>(tok) * 144 + tid] = sc;
            }
        }
    }

    // Preload Bias for warp 0 BEFORE gdc_wait!
    float bias_reg[12];
    if (warp == 0 && !is_pad) {
        #pragma unroll
        for (int k = 0; k < 12; ++k) {
            bias_reg[k] = __ldg(Bias + k * 32 + lane);
        }
    }

    // ---- Step 2: Wait for Router GEMM Logits[T, 384] ----
    if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    // ---- Step 3: Warp-0 Exact Top-8 Selection + Softmax + Atomic Counts ----
    if (warp == 0) {
        if (!is_pad) {
            uint32_t b_val[12];
            uint32_t alive_mask = 0xFFFu;
            const int64_t row_off = static_cast<int64_t>(tok) * E_FIXED;

            #pragma unroll
            for (int k = 0; k < 12; ++k) {
                const int e = k * 32 + lane;
                float l;
                if constexpr (LOGITS_BF16) {
                    const uint16_t* p16 = static_cast<const uint16_t*>(Logits) + row_off;
                    l = bf16_bits_to_f32(ld_cg_u16(p16 + e));
                } else {
                    const float* p32 = static_cast<const float*>(Logits) + row_off;
                    l = ld_cg_f32(p32 + e);
                }
                b_val[k] = selection_bits(l, bias_reg[k]);
            }

            int32_t my_id = 0;
            #pragma unroll
            for (int r = 0; r < TOPK_FIXED; ++r) {
                uint32_t best_b = 0u;
                int best_k = -1;
                #pragma unroll
                for (int k = 0; k < 12; ++k) {
                    if ((alive_mask & (1u << k)) && (best_k < 0 || b_val[k] > best_b)) {
                        best_b = b_val[k];
                        best_k = k;
                    }
                }
                uint32_t mb = redux_max_u32(best_k >= 0 ? best_b : 0u);
                uint32_t cand_id = (best_k >= 0 && best_b == mb)
                    ? static_cast<uint32_t>(best_k * 32 + lane) : 0xFFFFFFFFu;
                uint32_t win_id = redux_min_u32(cand_id);
                if (lane == r) my_id = static_cast<int32_t>(win_id);
                if (static_cast<uint32_t>(lane) == (win_id & 31u)) {
                    alive_mask &= ~(1u << (win_id >> 5));
                }
            }

            // Exact softmax routing weights (identical _log_sigmoid + _sum8 order!)
            float lp = -INFINITY;
            if (lane < TOPK_FIXED) {
                float my_logit;
                if constexpr (LOGITS_BF16) {
                    const uint16_t* p16 = static_cast<const uint16_t*>(Logits) + row_off;
                    my_logit = bf16_bits_to_f32(ld_cg_u16(p16 + my_id));
                } else {
                    const float* p32 = static_cast<const float*>(Logits) + row_off;
                    my_logit = ld_cg_f32(p32 + my_id);
                }
                lp = sub_rn(fminf(my_logit, 0.0f), log1pf(expf(-fabsf(my_logit))));
            }
            float lp0 = __shfl_sync(0xFFFFFFFFu, lp, 0), lp1 = __shfl_sync(0xFFFFFFFFu, lp, 1);
            float lp2 = __shfl_sync(0xFFFFFFFFu, lp, 2), lp3 = __shfl_sync(0xFFFFFFFFu, lp, 3);
            float lp4 = __shfl_sync(0xFFFFFFFFu, lp, 4), lp5 = __shfl_sync(0xFFFFFFFFu, lp, 5);
            float lp6 = __shfl_sync(0xFFFFFFFFu, lp, 6), lp7 = __shfl_sync(0xFFFFFFFFu, lp, 7);
            float max_lp = fmaxf(fmaxf(fmaxf(lp0, lp1), fmaxf(lp2, lp3)),
                                 fmaxf(fmaxf(lp4, lp5), fmaxf(lp6, lp7)));
            float sh = (lane < TOPK_FIXED) ? expf(sub_rn(lp, max_lp)) : 0.0f;
            float sh0 = __shfl_sync(0xFFFFFFFFu, sh, 0), sh1 = __shfl_sync(0xFFFFFFFFu, sh, 1);
            float sh2 = __shfl_sync(0xFFFFFFFFu, sh, 2), sh3 = __shfl_sync(0xFFFFFFFFu, sh, 3);
            float sh4 = __shfl_sync(0xFFFFFFFFu, sh, 4), sh5 = __shfl_sync(0xFFFFFFFFu, sh, 5);
            float sh6 = __shfl_sync(0xFFFFFFFFu, sh, 6), sh7 = __shfl_sync(0xFFFFFFFFu, sh, 7);

            float total = add_rn(add_rn(add_rn(sh0, sh2), add_rn(sh1, sh3)),
                                 add_rn(add_rn(sh4, sh6), add_rn(sh5, sh7)));
            float lse = add_rn(max_lp, logf(total));

            if (lane < TOPK_FIXED) {
                const int32_t my_loc = (my_id >= lo && my_id < hi) ? (my_id - lo) : -1;
                const int32_t my_rank = (my_loc >= 0) ? atomicAdd(&Counts[my_loc], 1) : 0;
                const float my_wt = mul_rn(expf(sub_rn(lp, lse)), route_scale);
                const int slot = tok * TOPK_FIXED + lane;
                Ids[slot] = my_loc;
                Ranks[slot] = my_rank;
                Weights[slot] = my_wt;
            }
        } else {
            if (lane < TOPK_FIXED) {
                const int slot = tok * TOPK_FIXED + lane;
                Ids[slot] = -1;
                Ranks[slot] = 0;
                Weights[slot] = 0.0f;
            }
        }
    }

    // Ensure all global stores (QValues/QScales/RowAmaxIn/Ids/Ranks/Weights/Counts) are L2-visible before barrier!
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        uint32_t prev = atom_acq_rel_gpu_add(&Counters[0], 1u);
        sIsLastCta = (prev == static_cast<uint32_t>(T - 1)) ? 1 : 0;
    }
    __syncthreads();

    if (!sIsLastCta) {
        if constexpr (PDL && TRIG_NEXT) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    // ---- Step 4 (Last CTA Only): 6-Warp Prefix Scan over all num_local <= 384 Experts ----
    int e0 = warp * 64 + lane;
    int e1 = e0 + 32;
    int c0 = 0, c1 = 0, t0 = 0, t1 = 0;
    int exc_t0 = 0, exc_c0 = 0, exc_t1 = 0, exc_c1 = 0;

    if (warp < 6) {
        c0 = (e0 < num_local) ? static_cast<int>(ld_cg_u32(Counts + e0)) : 0;
        c1 = (e1 < num_local) ? static_cast<int>(ld_cg_u32(Counts + e1)) : 0;
        if (e0 < num_local) Counts[e0] = 0;
        if (e1 < num_local) Counts[e1] = 0;

        t0 = (c0 + tile_sz - 1) / tile_sz;
        t1 = (c1 + tile_sz - 1) / tile_sz;

        int it0 = t0, ic0 = c0, it1 = t1, ic1 = c1;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int nt0 = __shfl_up_sync(0xFFFFFFFFu, it0, off), nc0 = __shfl_up_sync(0xFFFFFFFFu, ic0, off);
            int nt1 = __shfl_up_sync(0xFFFFFFFFu, it1, off), nc1 = __shfl_up_sync(0xFFFFFFFFu, ic1, off);
            if (lane >= off) { it0 += nt0; ic0 += nc0; it1 += nt1; ic1 += nc1; }
        }
        int sum0_t = __shfl_sync(0xFFFFFFFFu, it0, 31);
        int sum0_c = __shfl_sync(0xFFFFFFFFu, ic0, 31);
        if (lane == 31) {
            sWarpT[warp] = sum0_t + it1;
            sWarpC[warp] = sum0_c + ic1;
        }
        exc_t0 = it0 - t0;
        exc_c0 = ic0 - c0;
        exc_t1 = sum0_t + it1 - t1;
        exc_c1 = sum0_c + ic1 - c1;
    }
    __syncthreads();

    if (warp == 0) {
        int wt = (lane < 6) ? sWarpT[lane] : 0;
        int wc = (lane < 6) ? sWarpC[lane] : 0;
        int iwt = wt, iwc = wc;
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1) {
            int nt = __shfl_up_sync(0xFFFFFFFFu, iwt, off);
            int nc = __shfl_up_sync(0xFFFFFFFFu, iwc, off);
            if (lane >= off) { iwt += nt; iwc += nc; }
        }
        if (lane < 6) {
            sWarpT[lane] = iwt - wt;
            sWarpC[lane] = iwc - wc;
        }
        if (lane == 5) {
            Tiles[2 * max_ctas]     = iwt;
            Tiles[2 * max_ctas + 1] = iwt * tile_sz;
            Tiles[2 * max_ctas + 2] = iwc;
            Counters[0] = 0u;
        }
    }
    __syncthreads();

    if (warp < 6) {
        const int base_t = sWarpT[warp];
        const int base_c = sWarpC[warp];
        if (e0 < num_local) {
            const int ts0 = base_t + exc_t0;
            const int cs0 = base_c + exc_c0;
            Starts[e0] = ts0 * tile_sz;
            StartsCompact[e0] = cs0;
            for (int k = 0; k < t0; ++k) {
                const int ti = ts0 + k;
                const int end_r = (k + 1 < t0) ? ((ti + 1) * tile_sz) : (ts0 * tile_sz + c0);
                Tiles[ti] = e0;
                Tiles[max_ctas + ti] = end_r;
                TileCompactStart[ti] = cs0 + k * tile_sz;
            }
        }
        if (e1 < num_local) {
            const int ts1 = base_t + exc_t1;
            const int cs1 = base_c + exc_c1;
            Starts[e1] = ts1 * tile_sz;
            StartsCompact[e1] = cs1;
            for (int k = 0; k < t1; ++k) {
                const int ti = ts1 + k;
                const int end_r = (k + 1 < t1) ? ((ti + 1) * tile_sz) : (ts1 * tile_sz + c1);
                Tiles[ti] = e1;
                Tiles[max_ctas + ti] = end_r;
                TileCompactStart[ti] = cs1 + k * tile_sz;
            }
        }
    }

    if constexpr (PDL && TRIG_NEXT) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 2B (C32 Phase B): Fast Placement + Row2Tok + Compact RowAmaxW2 Pre-Zero
//   - Grid = (ceil(T*8 / 256), 1, 1) = 1..8 CTAs, Block = (256, 1, 1) (~1.2 us!)
//   - Writes Src2Dst, Src2Compact, Gammas, Rows, Row2Tok, QRowScales, RowAmaxW2[cdst]=0u
// =========================================================================
template <bool NVFP4, bool PDL, bool TRIG_NEXT>
__global__ __launch_bounds__(256, 4)
void c32_place_rows_kernel(
    const int32_t* __restrict__ Ids,
    const int32_t* __restrict__ Ranks,
    const float* __restrict__ Weights,
    const int32_t* __restrict__ Starts,
    const int32_t* __restrict__ StartsCompact,
    const float* __restrict__ RowAmaxIn,
    const float* __restrict__ WAmax,
    const float* __restrict__ WDenominator,
    int32_t* __restrict__ Src2Dst,
    int32_t* __restrict__ Src2Compact,
    float* __restrict__ Gammas,
    int32_t* __restrict__ Rows,
    int32_t* __restrict__ Row2Tok,
    float* __restrict__ QRowScales,
    uint32_t* __restrict__ RowAmaxW2,
    int total_slots
) {
    if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int slot = static_cast<int>(blockIdx.x) * 256 + threadIdx.x;
    if (slot < total_slots) {
        const int tok = slot >> 3;
        const int32_t loc = Ids[slot];
        if (loc >= 0) {
            const int32_t rank = Ranks[slot];
            const int32_t dst  = Starts[loc] + rank;
            const int32_t cdst = StartsCompact[loc] + rank;
            Src2Dst[slot]     = dst;
            Src2Compact[slot] = cdst;
            Gammas[dst]       = Weights[slot];
            Rows[cdst]        = dst;
            Row2Tok[dst]      = tok;
            if constexpr (NVFP4) {
                const float rcp_denom = rcp_rn(mul_rn(1536.0f, __ldg(WDenominator)));
                QRowScales[dst] = mul_rn(mul_rn(RowAmaxIn[tok], __ldg(WAmax + loc)), rcp_denom);
            }
            if (RowAmaxW2 != nullptr) {
                RowAmaxW2[cdst] = 0u;
            }
        } else {
            Src2Dst[slot]     = -1;
            Src2Compact[slot] = -1;
        }
    }

    if constexpr (PDL && TRIG_NEXT) {
        __syncthreads();
        if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_router_c32(
    KernelT kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args
) {
    if (!pdl) {
        kernel<<<grid, block, 0, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

// =========================================================================
// Extern "C" Entry Points
// =========================================================================

extern "C" int router_bf12_c32_gemm_fwd(
    const void* X,
    const void* SLABS,
    const void* OUTLIERS,
    const void* O_OFF,
    const void* NLivePtr,
    void* Logits,
    int T,
    uint32_t base_exp,
    int pdl,
    int trig_dep,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T > 256) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int m_blks = (T + 31) / 32;
    const bool use_pdl = (pdl != 0);
    const bool trig = use_pdl && (trig_dep != 0);
    cudaError_t err = cudaSuccess;

    #define LAUNCH_G(EW, CW, P, TR) err = launch_router_c32( \
        router_bf12_gemm_c32_kernel<EW, CW, P, TR>, \
        dim3(48 / (EW), m_blks, 1), dim3(160, 1, 1), stream, (P), \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<const int32_t*>(NLivePtr), static_cast<float*>(Logits), T, base_exp)

    if (T <= 32) {
        if (use_pdl && trig)       { LAUNCH_G(1, false, true,  true);  }
        else if (use_pdl && !trig) { LAUNCH_G(1, false, true,  false); }
        else                       { LAUNCH_G(1, false, false, false); }
    } else if (T <= 64) {
        if (use_pdl && trig)       { LAUNCH_G(2, true,  true,  true);  }
        else if (use_pdl && !trig) { LAUNCH_G(2, true,  true,  false); }
        else                       { LAUNCH_G(2, true,  false, false); }
    } else {
        if (use_pdl && trig)       { LAUNCH_G(4, true,  true,  true);  }
        else if (use_pdl && !trig) { LAUNCH_G(4, true,  true,  false); }
        else                       { LAUNCH_G(4, true,  false, false); }
    }
    #undef LAUNCH_G
    return static_cast<int>(err);
}

extern "C" int router_c32_quant_route_place_fwd(
    const void* X,
    const void* Logits,
    const void* Bias,
    const void* ExpertBounds,
    const void* Padding,
    const void* NLivePtr,
    void* Counts,
    void* Counters,
    void* Ids,
    void* Ranks,
    void* Weights,
    void* Starts,
    void* StartsCompact,
    void* Src2Dst,
    void* Src2Compact,
    void* Gammas,
    void* Rows,
    void* Tiles,
    void* TileCompactStart,
    void* Row2Tok,
    void* RowAmaxW2,
    void* QValues,
    void* QScales,
    void* QRowScales,
    void* RowAmaxIn,
    const void* WAmax,
    const void* WDenominator,
    int T,
    float route_scale,
    int num_local,
    int tile_sz,
    int max_ctas,
    int nvfp4,
    int logits_bf16,
    int pdl,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T > 256 || num_local <= 0 || num_local > E_FIXED) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl = (pdl != 0);
    const bool trig_w13 = (pdl >= 2);
    const bool is_nv = (nvfp4 != 0);
    const bool is_bf16 = (logits_bf16 != 0);
    cudaError_t err = cudaSuccess;

    dim3 grid_a(T, 1, 1);
    dim3 block_a(288, 1, 1);
    #define LAUNCH_A(NV, LB, P, TR) err = launch_router_c32( \
        c32_quant_topk_kernel<NV, LB, P, TR>, grid_a, block_a, stream, (P), \
        static_cast<const __nv_bfloat16*>(X), Logits, \
        static_cast<const float*>(Bias), static_cast<const int32_t*>(ExpertBounds), \
        static_cast<const bool*>(Padding), static_cast<const int32_t*>(NLivePtr), \
        static_cast<int32_t*>(Counts), static_cast<uint32_t*>(Counters), \
        static_cast<int32_t*>(Ids), static_cast<int32_t*>(Ranks), static_cast<float*>(Weights), \
        static_cast<int32_t*>(Starts), static_cast<int32_t*>(StartsCompact), \
        static_cast<int32_t*>(Tiles), static_cast<int32_t*>(TileCompactStart), \
        static_cast<uint8_t*>(QValues), static_cast<uint8_t*>(QScales), \
        static_cast<float*>(RowAmaxIn), T, route_scale, num_local, tile_sz, max_ctas)

    if (is_nv) {
        if (is_bf16) {
            if (use_pdl) { LAUNCH_A(true,  true,  true,  true);  }
            else         { LAUNCH_A(true,  true,  false, false); }
        } else {
            if (use_pdl) { LAUNCH_A(true,  false, true,  true);  }
            else         { LAUNCH_A(true,  false, false, false); }
        }
    } else {
        if (is_bf16) {
            if (use_pdl) { LAUNCH_A(false, true,  true,  true);  }
            else         { LAUNCH_A(false, true,  false, false); }
        } else {
            if (use_pdl) { LAUNCH_A(false, false, true,  true);  }
            else         { LAUNCH_A(false, false, false, false); }
        }
    }
    #undef LAUNCH_A
    if (err != cudaSuccess) return static_cast<int>(err);

    const int total_slots = T * TOPK_FIXED;
    dim3 grid_b((total_slots + 255) / 256, 1, 1);
    dim3 block_b(256, 1, 1);
    #define LAUNCH_B(NV, P, TR) err = launch_router_c32( \
        c32_place_rows_kernel<NV, P, TR>, grid_b, block_b, stream, (P), \
        static_cast<const int32_t*>(Ids), static_cast<const int32_t*>(Ranks), \
        static_cast<const float*>(Weights), static_cast<const int32_t*>(Starts), \
        static_cast<const int32_t*>(StartsCompact), static_cast<const float*>(RowAmaxIn), \
        static_cast<const float*>(WAmax), static_cast<const float*>(WDenominator), \
        static_cast<int32_t*>(Src2Dst), static_cast<int32_t*>(Src2Compact), \
        static_cast<float*>(Gammas), static_cast<int32_t*>(Rows), \
        static_cast<int32_t*>(Row2Tok), static_cast<float*>(QRowScales), \
        static_cast<uint32_t*>(RowAmaxW2), total_slots)

    if (is_nv) {
        if (use_pdl && trig_w13)       { LAUNCH_B(true,  true,  true);  }
        else if (use_pdl && !trig_w13) { LAUNCH_B(true,  true,  false); }
        else                           { LAUNCH_B(true,  false, false); }
    } else {
        if (use_pdl && trig_w13)       { LAUNCH_B(false, true,  true);  }
        else if (use_pdl && !trig_w13) { LAUNCH_B(false, true,  false); }
        else                           { LAUNCH_B(false, false, false); }
    }
    #undef LAUNCH_B
    return static_cast<int>(err);
}
