// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) 0-ULP Bit-Exact Fused ShortConv (W=4) + AttnRes + RMSNorm (+ AuxOut) for C=32 (M=17..512).
//
// Covers 100% of all 96/96 Inkling-2-Small calls/step (88x nq=1 + 8x nq=3) across M = 17..512:
//   1. `fused_sconv_ar_c32_row_fwd` (T * nq <= 256: all 88 nq=1 calls @ M=17..256 + 8 nq=3 @ M=17..85):
//      - Exact 256-thread (BLOCK_A=4096, BLOCK_B=512, num_warps=8) reduction tree + CUDA 13.0 ptxas
//        Order-A/B FFMA quirks (P1_LAST_MODE, P2_MODE) matching `attn_res_exact_v4` / Triton
//        `_attn_res_row_kernel` to LITERAL 0-ULP (`nd == 0`)!
//      - Bank-conflict-free SMEM-staged `sStack` (9.2..73.7 KiB) cuts register usage from ~160 -> ~64
//        regs/thr (2-4 CTAs/SM for N_STACK=1..5, 0 spills across N_STACK=1..8!).
//      - Loops q=0..NQ-1 inside 1 CTA (1x Stack DRAM read instead of 3x on nq=3 @ M=32..64!).
//   2. `fused_sconv_ar_c32_blockd_fwd` (T * nq > 256: 8x nq=3 global calls @ M=96..256 + T > 256):
//      - Exact 128-thread (BLOCK_D=1024, 5 chunks, num_warps=4, Order-A FFMA Pass 2) reduction tree
//        matching `attn_res_prefill_v2_fwd` / Triton `_attn_res_kernel` to LITERAL 0-ULP (`nd == 0`)!
//   3. Shared C=32 wins in BOTH kernels:
//      - Eliminates 96x `ConvOut` DRAM re-reads (226.5 MB/step @ M=256) + 96 kernel launches/step!
//      - Eliminates all 7x `partial.clone()` launches (16.5 MB/step readback) via fused `AuxOut` store!
//      - Eliminates the M=64..160 standalone Triton `_sconv_kernel` fallback hole!
//      - L2-aware cache hints: `.cs` streaming on `Stack`/`ConvCache`/`ConvOut`/`AuxOut`/`ConvRes`/`ConvAdd`/`MixLast`,
//        `.ca` on L2-hot `X`, `.nc` on `CW`/`Queries`/`NormW`, and normal `.wb` on `Out` (keeps `mlp_in`
//        & `x_q/x_k/x_v` 100% L2-hot for immediately following `router`/`scatter` & `qkvr` GEMM!).
//      - Safe `ConvRes == ConvOut` in-place aliasing + pow2 bitshift `divmod` on `N_CACHE`.
//
// Build on GB10 (SM121a — MUST pass --fmad=false, OMIT --use_fast_math & --ftz=true!):
//   /usr/local/cuda/bin/nvcc -O3 -arch=sm_121a --fmad=false --shared -Xcompiler -fPIC \
//       --ptxas-options=-v fused_sconv_ar_c32_exact.cu \
//       -o /rdma/libfused_sconv_ar_c32_exact.so

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

static __device__ __forceinline__ float rsqrt_approx(float x) {
    float y;
    asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float tldevice_exp_sub(float a, float b) {
    float y;
    asm volatile(
        "{\n\t"
        ".reg .f32 diff, t0, t1, t2, t3, t4, t5, t6, t7, f0;\n\t"
        ".reg .b32 b2, s0;\n\t"
        "sub.rn.f32 diff, %1, %2;\n\t"
        "fma.rn.ftz.f32 t0, diff, 0f3BBB989D, 0f3F000000;\n\t"
        "cvt.ftz.sat.f32.f32 t1, t0;\n\t"
        "fma.rm.ftz.f32 t2, t1, 0f437C0000, 0f4B400001;\n\t"
        "add.rn.f32 t3, t2, 0fCB40007F;\n\t"
        "neg.f32 t4, t3;\n\t"
        "fma.rn.ftz.f32 t5, diff, 0f3FB8AA3B, t4;\n\t"
        "fma.rn.ftz.f32 t6, diff, 0f32A57060, t5;\n\t"
        "mov.b32 b2, t2;\n\t"
        "shl.b32 s0, b2, 23;\n\t"
        "mov.b32 f0, s0;\n\t"
        "ex2.approx.ftz.f32 t7, t6;\n\t"
        "mul.rn.f32 %0, t7, f0;\n\t"
        "}\n\t"
        : "=f"(y) : "f"(a), "f"(b)
    );
    return y;
}

static __device__ __forceinline__ void divmod_cache(
    int64_t val, int n_cache, int n_shift, int64_t& q, int64_t& r
) {
    if (n_shift >= 0) {
        q = val >> n_shift;
        r = val & ((1LL << n_shift) - 1);
    } else {
        q = val / n_cache;
        r = val % n_cache;
    }
}

static __device__ __forceinline__ uint4 ld_global_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_global_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_global_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_global_ca_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_const_u4(const void* p) {
    uint4 v;
    asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_const_u32(const void* p) {
    uint32_t v;
    asm("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ float4 ld_const_f4(const float* p) {
    float4 v;
    asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p));
    return v;
}

static __device__ __forceinline__ float2 ld_const_f2(const float* p) {
    float2 v;
    asm("ld.global.nc.v2.f32 {%0,%1}, [%2];"
        : "=f"(v.x), "=f"(v.y)
        : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cs_u4(void* p, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ void st_global_cs_u32(void* p, uint32_t v) {
    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ void unpack_bf16x8(uint4 raw, float out[8]) {
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(h2[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

static __device__ __forceinline__ void unpack_bf16x2(uint32_t raw, float out[2]) {
    float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
    out[0] = f.x;
    out[1] = f.y;
}

static __device__ __forceinline__ uint4 pack_f32x8_to_bf16_u4(const float in[8]) {
    uint4 raw;
    __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        h2[i] = __halves2bfloat162(__float2bfloat16_rn(in[2 * i]), __float2bfloat16_rn(in[2 * i + 1]));
    }
    return raw;
}

static __device__ __forceinline__ uint32_t pack_f32x2_to_bf16_u32(const float in[2]) {
    __nv_bfloat162 h = __halves2bfloat162(__float2bfloat16_rn(in[0]), __float2bfloat16_rn(in[1]));
    return *reinterpret_cast<const uint32_t*>(&h);
}

static __device__ __forceinline__ void load_f32x8_const(const float* p, float out[8]) {
    float4 v0 = ld_const_f4(p);
    float4 v1 = ld_const_f4(p + 4);
    out[0] = v0.x; out[1] = v0.y; out[2] = v0.z; out[3] = v0.w;
    out[4] = v1.x; out[5] = v1.y; out[6] = v1.z; out[7] = v1.w;
}

// Load W=4 conv weights for 8 consecutive channels (CW is [D, 4] contiguous bf16 -> 4 uint4s).
static __device__ __forceinline__ void load_cw4x8_const(const __nv_bfloat16* cw_base, float w[4][8]) {
    uint4 r0 = ld_const_u4(cw_base);
    uint4 r1 = ld_const_u4(cw_base + 8);
    uint4 r2 = ld_const_u4(cw_base + 16);
    uint4 r3 = ld_const_u4(cw_base + 24);
    auto unpack2 = [&](uint4 r, int c0) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
        float2 a0 = __bfloat1622float2(h[0]);
        float2 a1 = __bfloat1622float2(h[1]);
        float2 b0 = __bfloat1622float2(h[2]);
        float2 b1 = __bfloat1622float2(h[3]);
        w[0][c0] = a0.x; w[1][c0] = a0.y; w[2][c0] = a1.x; w[3][c0] = a1.y;
        w[0][c0+1] = b0.x; w[1][c0+1] = b0.y; w[2][c0+1] = b1.x; w[3][c0+1] = b1.y;
    };
    unpack2(r0, 0);
    unpack2(r1, 2);
    unpack2(r2, 4);
    unpack2(r3, 6);
}

// Load W=4 conv weights for 2 consecutive channels (8 bf16 = 1 uint4).
static __device__ __forceinline__ void load_cw4x2_const(const __nv_bfloat16* cw_base, float w[4][2]) {
    uint4 r = ld_const_u4(cw_base);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
    float2 a0 = __bfloat1622float2(h[0]);
    float2 a1 = __bfloat1622float2(h[1]);
    float2 b0 = __bfloat1622float2(h[2]);
    float2 b1 = __bfloat1622float2(h[3]);
    w[0][0] = a0.x; w[1][0] = a0.y; w[2][0] = a1.x; w[3][0] = a1.y;
    w[0][1] = b0.x; w[1][1] = b0.y; w[2][1] = b1.x; w[3][1] = b1.y;
}

// =========================================================================
// Exact reduction trees matching BOTH Triton `_attn_res_row_kernel` (Regime A)
// AND Triton `_attn_res_kernel` (Regime B) under live CUDA 13.0 ptxas!
// =========================================================================
static __device__ __forceinline__ float tree16_sum(const float a0[8], const float a1[8]) {
    float r0_e = __fadd_rn(__fadd_rn(a0[0], a0[2]), __fadd_rn(a0[4], a0[6]));
    float r0_o = __fadd_rn(__fadd_rn(a0[1], a0[3]), __fadd_rn(a0[5], a0[7]));
    float r1_e = __fadd_rn(__fadd_rn(a1[0], a1[2]), __fadd_rn(a1[4], a1[6]));
    float r1_o = __fadd_rn(__fadd_rn(a1[1], a1[3]), __fadd_rn(a1[5], a1[7]));
    return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
}

template <int MODE>
static __device__ __forceinline__ float tree16_sum_sq(const float a0[8], const float a1[8]) {
    if constexpr (MODE == 0) {
        float r0_e = __fadd_rn(__fadd_rn(__fmul_rn(a0[0], a0[0]), __fmul_rn(a0[2], a0[2])),
                               __fadd_rn(__fmul_rn(a0[4], a0[4]), __fmul_rn(a0[6], a0[6])));
        float r0_o = __fadd_rn(__fadd_rn(__fmul_rn(a0[1], a0[1]), __fmul_rn(a0[3], a0[3])),
                               __fadd_rn(__fmul_rn(a0[5], a0[5]), __fmul_rn(a0[7], a0[7])));
        float r1_e = __fadd_rn(__fadd_rn(__fmul_rn(a1[0], a1[0]), __fmul_rn(a1[2], a1[2])),
                               __fadd_rn(__fmul_rn(a1[4], a1[4]), __fmul_rn(a1[6], a1[6])));
        float r1_o = __fadd_rn(__fadd_rn(__fmul_rn(a1[1], a1[1]), __fmul_rn(a1[3], a1[3])),
                               __fadd_rn(__fmul_rn(a1[5], a1[5]), __fmul_rn(a1[7], a1[7])));
        return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
    } else if constexpr (MODE == 1) {
        float r0_e = __fadd_rn(fmaf(a0[0], a0[0], __fmul_rn(a0[2], a0[2])),
                               fmaf(a0[4], a0[4], __fmul_rn(a0[6], a0[6])));
        float r0_o = __fadd_rn(fmaf(a0[1], a0[1], __fmul_rn(a0[3], a0[3])),
                               fmaf(a0[5], a0[5], __fmul_rn(a0[7], a0[7])));
        float r1_e = __fadd_rn(fmaf(a1[0], a1[0], __fmul_rn(a1[2], a1[2])),
                               fmaf(a1[4], a1[4], __fmul_rn(a1[6], a1[6])));
        float r1_o = __fadd_rn(fmaf(a1[1], a1[1], __fmul_rn(a1[3], a1[3])),
                               fmaf(a1[5], a1[5], __fmul_rn(a1[7], a1[7])));
        return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
    } else {
        // MODE == 3 (N_STACK == 8 Pass 2 quirk in CUDA 13.0 ptxas)
        float r0_e = __fadd_rn(fmaf(a0[2], a0[2], __fmul_rn(a0[0], a0[0])),
                               fmaf(a0[4], a0[4], __fmul_rn(a0[6], a0[6])));
        float r0_o = __fadd_rn(fmaf(a0[3], a0[3], __fmul_rn(a0[1], a0[1])),
                               fmaf(a0[5], a0[5], __fmul_rn(a0[7], a0[7])));
        float r1_e = __fadd_rn(fmaf(a1[0], a1[0], __fmul_rn(a1[2], a1[2])),
                               fmaf(a1[4], a1[4], __fmul_rn(a1[6], a1[6])));
        float r1_o = __fadd_rn(fmaf(a1[1], a1[1], __fmul_rn(a1[3], a1[3])),
                               fmaf(a1[5], a1[5], __fmul_rn(a1[7], a1[7])));
        return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
    }
}

static __device__ __forceinline__ float tree8_sum(const float v[8]) {
    float e0 = __fadd_rn(__fadd_rn(v[0], v[2]), __fadd_rn(v[4], v[6]));
    float o0 = __fadd_rn(__fadd_rn(v[1], v[3]), __fadd_rn(v[5], v[7]));
    return __fadd_rn(e0, o0);
}

static __device__ __forceinline__ float warp_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ void warp_sum_bfly2(float& a, float& b) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        a = __fadd_rn(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
        b = __fadd_rn(b, __shfl_xor_sync(0xFFFFFFFFu, b, off));
    }
}

static __device__ __forceinline__ float warp8_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ float warp4_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

// =========================================================================
// KERNEL A: Fused ShortConv + AttnRes (BLOCK_A=4096, BLOCK_B=512, 256 thr)
//   Exact 0-ULP match to short_conv + attn_res_exact_v4 / _attn_res_row_kernel!
//   Used for T * nq <= 256 (all 88 nq=1 calls @ M=17..256 + 8 nq=3 @ M=17..85).
//   NOTE: ConvOut & ConvRes intentionally omit __restrict__ (in-place aliasing!).
// =========================================================================
template <int N_STACK, int NQ, bool HAS_CONV_RES, bool HAS_CONV_ADDEND, bool HAS_MIX_LAST>
__global__ void __launch_bounds__(256, 2)
fused_sconv_ar_c32_row_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ CW,       // [4608, 4] contiguous
    __nv_bfloat16* __restrict__ Cache,          // [num_blocks, 1, N_CACHE, cache_D]
    __nv_bfloat16* ConvOut,                     // [T, 4608] (may alias ConvRes!)
    const __nv_bfloat16* ConvRes,               // [T, 4608] optional
    const __nv_bfloat16* __restrict__ ConvAdd,  // [T, 4608] optional
    __nv_bfloat16* __restrict__ AuxOut,         // [T, 4608] optional (eliminates 7x partial.clone!)
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ QStart,
    const __nv_bfloat16* __restrict__ Stack,    // [N_STACK, stride_s]
    const __nv_bfloat16* __restrict__ MixLast,  // [T, 4608] optional
    const float* __restrict__ Queries,          // [NQ, 4608]
    const __nv_bfloat16* __restrict__ NormW,    // [NQ, 4608]
    __nv_bfloat16* __restrict__ Out,            // [NQ, T, 4608]
    int64_t stride_x,
    int64_t stride_c_blk,
    int64_t stride_c_n,
    int64_t stride_bt,
    int max_blocks,
    int N_CACHE,
    int N_SHIFT,
    int stream_offset,
    int64_t stride_s,
    int T,
    float scale,
    float eps,
    float norm_eps
) {
    constexpr int D = 4608;
    constexpr int NS = N_STACK + 1;
    constexpr float INV_D = 1.0f / (float)D;
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t row_off = static_cast<int64_t>(t) * D;
    const int off_a0 = tid * 8;
    const int off_a1 = 2048 + tid * 8;
    const int off_b  = 4096 + tid * 2;

    // Bank-conflict-free dynamic SMEM for Stack[0..N_STACK-1]:
    //   sStack_a0 : N_STACK * 256 uint4    (N_STACK * 4096 B)
    //   sStack_a1 : N_STACK * 256 uint4    (N_STACK * 4096 B)
    //   sStack_b  : N_STACK * 256 uint32_t (N_STACK * 1024 B)
    //   Total     = N_STACK * 9216 B (9.2 KiB .. 73.7 KiB <= 101.4 KiB!)
    extern __shared__ uint4 dyn_raw_u4[];
    uint8_t*  dyn_raw   = reinterpret_cast<uint8_t*>(dyn_raw_u4);
    uint4*    sStack_a0 = dyn_raw_u4;
    uint4*    sStack_a1 = reinterpret_cast<uint4*>(dyn_raw + static_cast<size_t>(N_STACK) * 4096);
    uint32_t* sStack_b  = reinterpret_cast<uint32_t*>(dyn_raw + static_cast<size_t>(N_STACK) * 8192);

    alignas(16) __shared__ float sRedSq[NS][2][8];        // [source][a/b][warp]
    alignas(16) __shared__ float sRedDt[NQ][NS][2][8];    // [q][source][a/b][warp]
    alignas(16) __shared__ float sSq2[2][8];              // Pass 2 [a/b][warp]

    // ------------------------------------------------------------------
    // Step 1: ShortConv for token t across all 3 segments (off_a0, off_a1, off_b)
    // ------------------------------------------------------------------
    const int64_t slot = Slot[t];
    const bool valid = (slot >= 0);
    const int64_t pos = Pos[t];
    const int32_t req = SeqIdx[t];
    const int32_t qstart = QStart[t];
    const bool lag0_ok = valid && (pos >= 0);

    const __nv_bfloat16* tap_ptrs[3];
    bool tap_valid[3];
    bool tap_is_cache[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int lag = 3 - k;
        const int64_t src = pos - lag;
        const int row = t - lag;
        const bool in_win = valid && (src >= 0);
        tap_valid[k] = in_win;
        if (in_win) {
            if (row >= qstart) {
                tap_ptrs[k] = X + static_cast<int64_t>(row) * stride_x;
                tap_is_cache[k] = false;
            } else {
                int64_t lblk, rem;
                divmod_cache(src, N_CACHE, N_SHIFT, lblk, rem);
                if (lblk >= max_blocks) lblk = max_blocks - 1;
                const int64_t blk = static_cast<int64_t>(BT[static_cast<int64_t>(req) * stride_bt + lblk]);
                tap_ptrs[k] = Cache + blk * stride_c_blk + rem * stride_c_n + stream_offset;
                tap_is_cache[k] = true;
            }
        } else {
            tap_ptrs[k] = X;
            tap_is_cache[k] = false;
        }
    }
    __nv_bfloat16* cache_dst = nullptr;
    if (valid) {
        int64_t dblk, drem;
        divmod_cache(slot, N_CACHE, N_SHIFT, dblk, drem);
        cache_dst = Cache + dblk * stride_c_blk + drem * stride_c_n + stream_offset;
    }

    const __nv_bfloat16* x_cur_p = X + static_cast<int64_t>(t) * stride_x;

    // Helper for 8-element segment (a0 or a1)
    auto run_sconv_seg8 = [&](int off) -> uint4 {
        uint4 raw_cur = ld_global_ca_u4(x_cur_p + off);
        uint4 raw_t0 = tap_valid[0] ? (tap_is_cache[0] ? ld_global_cs_u4(tap_ptrs[0] + off) : ld_global_ca_u4(tap_ptrs[0] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t1 = tap_valid[1] ? (tap_is_cache[1] ? ld_global_cs_u4(tap_ptrs[1] + off) : ld_global_ca_u4(tap_ptrs[1] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t2 = tap_valid[2] ? (tap_is_cache[2] ? ld_global_cs_u4(tap_ptrs[2] + off) : ld_global_ca_u4(tap_ptrs[2] + off)) : make_uint4(0u, 0u, 0u, 0u);
        if (valid) st_global_cs_u4(cache_dst + off, raw_cur);

        float w[4][8];
        load_cw4x8_const(CW + off * 4, w);
        float cur_f[8], t0_f[8], t1_f[8], t2_f[8];
        unpack_bf16x8(raw_cur, cur_f);
        unpack_bf16x8(raw_t0, t0_f);
        unpack_bf16x8(raw_t1, t1_f);
        unpack_bf16x8(raw_t2, t2_f);

        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float a = 0.0f;
            a = fmaf(tap_valid[0] ? t0_f[i] : 0.0f, w[0][i], a);
            a = fmaf(tap_valid[1] ? t1_f[i] : 0.0f, w[1][i], a);
            a = fmaf(tap_valid[2] ? t2_f[i] : 0.0f, w[2][i], a);
            a = fmaf(lag0_ok      ? cur_f[i] : 0.0f, w[3][i], a);
            acc[i] = __fmul_rn(__fadd_rn(a, cur_f[i]), scale);
        }
        uint4 raw_y = pack_f32x8_to_bf16_u4(acc);

        if constexpr (HAS_CONV_ADDEND) {
            // Load ConvRes STRICTLY BEFORE storing ConvOut (safe when ConvRes == ConvOut!)
            uint4 raw_ca = ld_global_cs_u4(ConvAdd + row_off + off);
            float ca[8], yf[8];
            unpack_bf16x8(raw_ca, ca);
            unpack_bf16x8(raw_y, yf);
            if constexpr (HAS_CONV_RES) {
                uint4 raw_cr = ld_global_cs_u4(ConvRes + row_off + off);
                float cr[8];
                unpack_bf16x8(raw_cr, cr);
                #pragma unroll
                for (int i = 0; i < 8; ++i) ca[i] = __fadd_rn(cr[i], ca[i]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(ca[i], yf[i]);
            raw_y = pack_f32x8_to_bf16_u4(acc);
        }
        st_global_cs_u4(ConvOut + row_off + off, raw_y);
        if (AuxOut != nullptr) st_global_cs_u4(AuxOut + row_off + off, raw_y);
        return raw_y;
    };

    // Helper for 2-element tail segment (off_b)
    auto run_sconv_seg2 = [&](int off) -> uint32_t {
        uint32_t raw_cur = ld_global_ca_u32(x_cur_p + off);
        uint32_t raw_t0 = tap_valid[0] ? (tap_is_cache[0] ? ld_global_cs_u32(tap_ptrs[0] + off) : ld_global_ca_u32(tap_ptrs[0] + off)) : 0u;
        uint32_t raw_t1 = tap_valid[1] ? (tap_is_cache[1] ? ld_global_cs_u32(tap_ptrs[1] + off) : ld_global_ca_u32(tap_ptrs[1] + off)) : 0u;
        uint32_t raw_t2 = tap_valid[2] ? (tap_is_cache[2] ? ld_global_cs_u32(tap_ptrs[2] + off) : ld_global_ca_u32(tap_ptrs[2] + off)) : 0u;
        if (valid) st_global_cs_u32(cache_dst + off, raw_cur);

        float w[4][2];
        load_cw4x2_const(CW + off * 4, w);
        float cur_f[2], t0_f[2], t1_f[2], t2_f[2];
        unpack_bf16x2(raw_cur, cur_f);
        unpack_bf16x2(raw_t0, t0_f);
        unpack_bf16x2(raw_t1, t1_f);
        unpack_bf16x2(raw_t2, t2_f);

        float acc[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            float a = 0.0f;
            a = fmaf(tap_valid[0] ? t0_f[i] : 0.0f, w[0][i], a);
            a = fmaf(tap_valid[1] ? t1_f[i] : 0.0f, w[1][i], a);
            a = fmaf(tap_valid[2] ? t2_f[i] : 0.0f, w[2][i], a);
            a = fmaf(lag0_ok      ? cur_f[i] : 0.0f, w[3][i], a);
            acc[i] = __fmul_rn(__fadd_rn(a, cur_f[i]), scale);
        }
        uint32_t raw_y = pack_f32x2_to_bf16_u32(acc);

        if constexpr (HAS_CONV_ADDEND) {
            uint32_t raw_ca = ld_global_cs_u32(ConvAdd + row_off + off);
            float ca[2], yf[2];
            unpack_bf16x2(raw_ca, ca);
            unpack_bf16x2(raw_y, yf);
            if constexpr (HAS_CONV_RES) {
                uint32_t raw_cr = ld_global_cs_u32(ConvRes + row_off + off);
                float cr[2];
                unpack_bf16x2(raw_cr, cr);
                ca[0] = __fadd_rn(cr[0], ca[0]);
                ca[1] = __fadd_rn(cr[1], ca[1]);
            }
            acc[0] = __fadd_rn(ca[0], yf[0]);
            acc[1] = __fadd_rn(ca[1], yf[1]);
            raw_y = pack_f32x2_to_bf16_u32(acc);
        }
        st_global_cs_u32(ConvOut + row_off + off, raw_y);
        if (AuxOut != nullptr) st_global_cs_u32(AuxOut + row_off + off, raw_y);
        return raw_y;
    };

    // Execute ShortConv -> keep ConvOut directly in registers (ZERO DRAM re-read!)
    uint4 last_r0 = run_sconv_seg8(off_a0);
    uint4 last_r1 = run_sconv_seg8(off_a1);
    uint32_t last_rb = run_sconv_seg2(off_b);

    // ------------------------------------------------------------------
    // Step 2: AttnRes Pass 1 on Last (= MixLast + ConvOut if HAS_MIX_LAST, else ConvOut)
    // ------------------------------------------------------------------
    float last_f0[8], last_f1[8], last_fb[2];
    {
        float lx0[8], lx1[8], lxb[2];
        unpack_bf16x8(last_r0, lx0);
        unpack_bf16x8(last_r1, lx1);
        unpack_bf16x2(last_rb, lxb);
        if constexpr (HAS_MIX_LAST) {
            uint4 ml_r0 = ld_global_cs_u4(MixLast + row_off + off_a0);
            uint4 ml_r1 = ld_global_cs_u4(MixLast + row_off + off_a1);
            uint32_t ml_rb = ld_global_cs_u32(MixLast + row_off + off_b);
            float m0[8], m1[8], mb[2];
            unpack_bf16x8(ml_r0, m0);
            unpack_bf16x8(ml_r1, m1);
            unpack_bf16x2(ml_rb, mb);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                // Exact order matching attn_res(stack, partial, attn_out): partial + attn_out!
                lx0[i] = __fadd_rn(m0[i], lx0[i]);
                lx1[i] = __fadd_rn(m1[i], lx1[i]);
                last_f0[i] = lx0[i];
                last_f1[i] = lx1[i];
            }
            lxb[0] = __fadd_rn(mb[0], lxb[0]);
            lxb[1] = __fadd_rn(mb[1], lxb[1]);
            last_fb[0] = lxb[0];
            last_fb[1] = lxb[1];
        }
#ifdef INKLING_PTXAS133_STRICT
        constexpr int P1_LAST_MODE = 0;
#else
        // Matches live vllm serve Triton (TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas, CUDA 13.0)
        // and libattn_res_exact_v4.so: ptxas 13.0 fuses mul.rn.f32x2+add.rn.f32x2 into Order-A FFMA when HAS_MIX_LAST.
        constexpr int P1_LAST_MODE = HAS_MIX_LAST ? 1 : 0;
#endif
        float sqa = tree16_sum_sq<P1_LAST_MODE>(lx0, lx1);
        float sqb = __fadd_rn(__fmul_rn(lxb[0], lxb[0]), __fmul_rn(lxb[1], lxb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[N_STACK][0][warp] = sqa;
            sRedSq[N_STACK][1][warp] = sqb;
        }
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float* qp = Queries + static_cast<int64_t>(q) * D;
            float qa0[8], qa1[8];
            load_f32x8_const(qp + off_a0, qa0);
            load_f32x8_const(qp + off_a1, qa1);
            float2 qbf = ld_const_f2(qp + off_b);
            float dt0[8], dt1[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dt0[i] = __fmul_rn(qa0[i], lx0[i]);
                dt1[i] = __fmul_rn(qa1[i], lx1[i]);
            }
            float dta = tree16_sum(dt0, dt1);
            float dtb = __fadd_rn(__fmul_rn(qbf.x, lxb[0]), __fmul_rn(qbf.y, lxb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[q][N_STACK][0][warp] = dta;
                sRedDt[q][N_STACK][1][warp] = dtb;
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Stream Stack[0..N_STACK-1] ONCE (.cs) into bank-conflict-free SMEM + Pass 1!
    // ------------------------------------------------------------------
    #pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
        const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
        uint4 r0 = ld_global_cs_u4(sp + off_a0);
        uint4 r1 = ld_global_cs_u4(sp + off_a1);
        uint32_t rb = ld_global_cs_u32(sp + off_b);
        sStack_a0[n * 256 + tid] = r0;
        sStack_a1[n * 256 + tid] = r1;
        sStack_b[n * 256 + tid]  = rb;

        float xa0[8], xa1[8], xb[2];
        unpack_bf16x8(r0, xa0);
        unpack_bf16x8(r1, xa1);
        unpack_bf16x2(rb, xb);

        float sqa = tree16_sum_sq<0>(xa0, xa1);
        float sqb = __fadd_rn(__fmul_rn(xb[0], xb[0]), __fmul_rn(xb[1], xb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[n][0][warp] = sqa;
            sRedSq[n][1][warp] = sqb;
        }
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float* qp = Queries + static_cast<int64_t>(q) * D;
            float qa0[8], qa1[8];
            load_f32x8_const(qp + off_a0, qa0);
            load_f32x8_const(qp + off_a1, qa1);
            float2 qbf = ld_const_f2(qp + off_b);
            float dt0[8], dt1[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dt0[i] = __fmul_rn(qa0[i], xa0[i]);
                dt1[i] = __fmul_rn(qa1[i], xa1[i]);
            }
            float dta = tree16_sum(dt0, dt1);
            float dtb = __fadd_rn(__fmul_rn(qbf.x, xb[0]), __fmul_rn(qbf.y, xb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[q][n][0][warp] = dta;
                sRedDt[q][n][1][warp] = dtb;
            }
        }
    }

    __syncthreads();

    // Compute per-source rstd ONCE across all NQ queries!
    const int l8 = lane & 7;
    const int half = (lane >> 3) & 1;  // 0 -> a, 1 -> b
    float rstd_arr[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float v = warp8_sum_bfly(sRedSq[n][half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v, 8);
        rstd_arr[n] = rsqrt_approx(fmaf(__fadd_rn(sa, sb), INV_D, eps));
    }

    // ------------------------------------------------------------------
    // Step 4: Pass 2 (ZERO global reads of Stack/Last!) + RMSNorm + L2-hot Out store
    // ------------------------------------------------------------------
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float my_lg = -INFINITY;
        float my_ir = 0.0f;
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float v = warp8_sum_bfly(sRedDt[q][n][half][l8]);
            float da = __shfl_sync(0xFFFFFFFFu, v, 0);
            float db = __shfl_sync(0xFFFFFFFFu, v, 8);
            if (lane == n) {
                my_lg = __fmul_rn(__fadd_rn(da, db), rstd_arr[n]);
                my_ir = rstd_arr[n];
            }
        }
        float mx = my_lg;
        if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
        if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        float p = (lane < NS) ? tldevice_exp_sub(my_lg, mx) : 0.0f;
        float sp = p;
        if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
        if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
        if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
        if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
        float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
        p = __fmul_rn(p, my_ir);  // normalize_values == true
        float probs[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) probs[n] = __shfl_sync(0xFFFFFFFFu, p, n);

        const int64_t q_off = static_cast<int64_t>(q) * D;
        uint4 nw_r0 = ld_const_u4(NormW + q_off + off_a0);
        uint4 nw_r1 = ld_const_u4(NormW + q_off + off_a1);
        uint32_t nw_rb = ld_const_u32(NormW + q_off + off_b);

        float acc0[8] = {0,0,0,0,0,0,0,0};
        float acc1[8] = {0,0,0,0,0,0,0,0};
        float accb[2] = {0,0};

        #pragma unroll
        for (int n = 0; n < N_STACK; ++n) {
            const float pn = probs[n];
            uint4 r0 = sStack_a0[n * 256 + tid];
            uint4 r1 = sStack_a1[n * 256 + tid];
            uint32_t rb = sStack_b[n * 256 + tid];
            const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
            const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f0 = __bfloat1622float2(h0[i]);
                acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                float2 f1 = __bfloat1622float2(h1[i]);
                acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
            }
            float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rb));
            accb[0] = fmaf(pn, fb.x, accb[0]);
            accb[1] = fmaf(pn, fb.y, accb[1]);
        }
        {
            const float pn = probs[N_STACK];
            if constexpr (HAS_MIX_LAST) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc0[i] = fmaf(pn, last_f0[i], acc0[i]);
                    acc1[i] = fmaf(pn, last_f1[i], acc1[i]);
                }
                accb[0] = fmaf(pn, last_fb[0], accb[0]);
                accb[1] = fmaf(pn, last_fb[1], accb[1]);
            } else {
                const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&last_r0);
                const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&last_r1);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float2 f0 = __bfloat1622float2(h0[i]);
                    acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                    acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                    float2 f1 = __bfloat1622float2(h1[i]);
                    acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                    acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
                }
                float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&last_rb));
                accb[0] = fmaf(pn, fb.x, accb[0]);
                accb[1] = fmaf(pn, fb.y, accb[1]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(acc0[i], inv_sum);
            acc1[i] = __fmul_rn(acc1[i], inv_sum);
        }
        accb[0] = __fmul_rn(accb[0], inv_sum);
        accb[1] = __fmul_rn(accb[1], inv_sum);

#ifdef INKLING_PTXAS133_STRICT
        constexpr int P2_MODE = 0;
#else
        constexpr int P2_MODE = (N_STACK == 8) ? 3 : (HAS_MIX_LAST ? 1 : 0);
#endif
        float sqa = tree16_sum_sq<P2_MODE>(acc0, acc1);
        float sqb = __fadd_rn(__fmul_rn(accb[0], accb[0]), __fmul_rn(accb[1], accb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sSq2[0][warp] = sqa;
            sSq2[1][warp] = sqb;
        }
        __syncthreads();

        float v2 = warp8_sum_bfly(sSq2[half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v2, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v2, 8);
        float out_rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(__fadd_rn(sa, sb), (float)D), norm_eps));

        float wa0[8], wa1[8], wb[2];
        unpack_bf16x8(nw_r0, wa0);
        unpack_bf16x8(nw_r1, wa1);
        unpack_bf16x2(nw_rb, wb);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(__fmul_rn(acc0[i], out_rstd), wa0[i]);
            acc1[i] = __fmul_rn(__fmul_rn(acc1[i], out_rstd), wa1[i]);
        }
        accb[0] = __fmul_rn(__fmul_rn(accb[0], out_rstd), wb[0]);
        accb[1] = __fmul_rn(__fmul_rn(accb[1], out_rstd), wb[1]);

        // Normal write-back store so Out (mlp_in / x_q,x_k,x_v) stays 100% L2-hot for next GEMM/scatter!
        __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
        *reinterpret_cast<uint4*>(out_p + off_a0)   = pack_f32x8_to_bf16_u4(acc0);
        *reinterpret_cast<uint4*>(out_p + off_a1)   = pack_f32x8_to_bf16_u4(acc1);
        *reinterpret_cast<uint32_t*>(out_p + off_b) = pack_f32x2_to_bf16_u32(accb);

        if constexpr (NQ > 1) {
            if (q + 1 < NQ) __syncthreads();
        }
    }
}

// =========================================================================
// KERNEL B: Fused ShortConv + AttnRes (BLOCK_D=1024, 128 thr, 5 chunks)
//   Exact 0-ULP match to short_conv + attn_res_prefill_v2_fwd / _attn_res_kernel!
//   Used for T * nq > 256 (all 8x nq=3 global calls @ M=96..256 + prefill!).
//   Adds fused AuxOut (kills all 7x partial.clone()!), pow2 divmod, & L2-hot Out for T<=512!
// =========================================================================
template <int N_STACK, int NQ, bool HAS_CONV_RES, bool HAS_CONV_ADDEND, bool HAS_MIX_LAST>
__global__ void __launch_bounds__(128)
fused_sconv_ar_c32_blockd_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ CW,
    __nv_bfloat16* __restrict__ Cache,
    __nv_bfloat16* ConvOut,
    const __nv_bfloat16* ConvRes,
    const __nv_bfloat16* __restrict__ ConvAdd,
    __nv_bfloat16* __restrict__ AuxOut,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ QStart,
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ MixLast,
    const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW,
    __nv_bfloat16* __restrict__ Out,
    int64_t stride_x,
    int64_t stride_c_blk,
    int64_t stride_c_n,
    int64_t stride_bt,
    int max_blocks,
    int N_CACHE,
    int N_SHIFT,
    int stream_offset,
    int64_t stride_s,
    int T,
    float scale,
    float eps,
    float norm_eps
) {
    constexpr int D = 4608;
    constexpr int NS = N_STACK + 1;
    constexpr int U4_PER_ROW = D / 8;
    constexpr float INV_D = 1.0f / (float)D;
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t row_off = static_cast<int64_t>(t) * D;

    extern __shared__ uint4 dyn_smem[];
    uint4* sStack = dyn_smem;
    uint4* sLastU4 = sStack + N_STACK * U4_PER_ROW;
    float* sLastF32 = reinterpret_cast<float*>(sLastU4);
    alignas(16) __shared__ float sRed[NS][1 + NQ][4];
    alignas(16) __shared__ float sSq2[5][4];

    const int64_t slot = Slot[t];
    const bool valid = (slot >= 0);
    const int64_t pos = Pos[t];
    const int32_t req = SeqIdx[t];
    const int32_t qstart = QStart[t];
    const bool lag0_ok = valid && (pos >= 0);

    const __nv_bfloat16* tap_ptrs[3];
    bool tap_valid[3];
    bool tap_is_cache[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int lag = 3 - k;
        const int64_t src = pos - lag;
        const int row = t - lag;
        const bool in_win = valid && (src >= 0);
        tap_valid[k] = in_win;
        if (in_win) {
            if (row >= qstart) {
                tap_ptrs[k] = X + static_cast<int64_t>(row) * stride_x;
                tap_is_cache[k] = false;
            } else {
                int64_t lblk, rem;
                divmod_cache(src, N_CACHE, N_SHIFT, lblk, rem);
                if (lblk >= max_blocks) lblk = max_blocks - 1;
                const int64_t blk = static_cast<int64_t>(BT[static_cast<int64_t>(req) * stride_bt + lblk]);
                tap_ptrs[k] = Cache + blk * stride_c_blk + rem * stride_c_n + stream_offset;
                tap_is_cache[k] = true;
            }
        } else {
            tap_ptrs[k] = X;
            tap_is_cache[k] = false;
        }
    }
    __nv_bfloat16* cache_dst = nullptr;
    if (valid) {
        int64_t dblk, drem;
        divmod_cache(slot, N_CACHE, N_SHIFT, dblk, drem);
        cache_dst = Cache + dblk * stride_c_blk + drem * stride_c_n + stream_offset;
    }

    // Step 1: ShortConv + Pass 1 on Last
    {
        const __nv_bfloat16* x_cur_p = X + static_cast<int64_t>(t) * stride_x;
        float sum_sq[8] = {0,0,0,0,0,0,0,0};
        float dots[NQ][8];
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) dots[q][i] = 0.0f;
        }

        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int d_base = c * 1024 + tid * 8;
            if (c < 4 || tid < 64) {
                uint4 raw_cur = ld_global_ca_u4(x_cur_p + d_base);
                uint4 raw_t0 = tap_valid[0] ? (tap_is_cache[0] ? ld_global_cs_u4(tap_ptrs[0] + d_base) : ld_global_ca_u4(tap_ptrs[0] + d_base)) : make_uint4(0u,0u,0u,0u);
                uint4 raw_t1 = tap_valid[1] ? (tap_is_cache[1] ? ld_global_cs_u4(tap_ptrs[1] + d_base) : ld_global_ca_u4(tap_ptrs[1] + d_base)) : make_uint4(0u,0u,0u,0u);
                uint4 raw_t2 = tap_valid[2] ? (tap_is_cache[2] ? ld_global_cs_u4(tap_ptrs[2] + d_base) : ld_global_ca_u4(tap_ptrs[2] + d_base)) : make_uint4(0u,0u,0u,0u);
                if (valid) st_global_cs_u4(cache_dst + d_base, raw_cur);

                float w[4][8];
                load_cw4x8_const(CW + d_base * 4, w);
                float cur_f[8], t0_f[8], t1_f[8], t2_f[8];
                unpack_bf16x8(raw_cur, cur_f);
                unpack_bf16x8(raw_t0, t0_f);
                unpack_bf16x8(raw_t1, t1_f);
                unpack_bf16x8(raw_t2, t2_f);

                float acc[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float a = 0.0f;
                    a = fmaf(tap_valid[0] ? t0_f[i] : 0.0f, w[0][i], a);
                    a = fmaf(tap_valid[1] ? t1_f[i] : 0.0f, w[1][i], a);
                    a = fmaf(tap_valid[2] ? t2_f[i] : 0.0f, w[2][i], a);
                    a = fmaf(lag0_ok      ? cur_f[i] : 0.0f, w[3][i], a);
                    acc[i] = __fmul_rn(__fadd_rn(a, cur_f[i]), scale);
                }
                uint4 raw_y = pack_f32x8_to_bf16_u4(acc);

                if constexpr (HAS_CONV_ADDEND) {
                    uint4 raw_ca = ld_global_cs_u4(ConvAdd + row_off + d_base);
                    float ca[8], y_f[8];
                    unpack_bf16x8(raw_ca, ca);
                    unpack_bf16x8(raw_y, y_f);
                    if constexpr (HAS_CONV_RES) {
                        uint4 raw_cr = ld_global_cs_u4(ConvRes + row_off + d_base);
                        float cr[8];
                        unpack_bf16x8(raw_cr, cr);
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) ca[i] = __fadd_rn(cr[i], ca[i]);
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(ca[i], y_f[i]);
                    raw_y = pack_f32x8_to_bf16_u4(acc);
                }
                st_global_cs_u4(ConvOut + row_off + d_base, raw_y);
                if (AuxOut != nullptr) st_global_cs_u4(AuxOut + row_off + d_base, raw_y);

                float x[8];
                unpack_bf16x8(raw_y, x);
                if constexpr (HAS_MIX_LAST) {
                    uint4 raw_ml = ld_global_cs_u4(MixLast + row_off + d_base);
                    float ml[8];
                    unpack_bf16x8(raw_ml, ml);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) x[i] = __fadd_rn(ml[i], x[i]);
                    *reinterpret_cast<float4*>(sLastF32 + d_base)     = make_float4(x[0], x[1], x[2], x[3]);
                    *reinterpret_cast<float4*>(sLastF32 + d_base + 4) = make_float4(x[4], x[5], x[6], x[7]);
                } else {
                    sLastU4[d_base >> 3] = raw_y;
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum_sq[i] = fmaf(x[i], x[i], sum_sq[i]);
                #pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float qv[8];
                    load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dots[q][i] = fmaf(qv[i], x[i], dots[q][i]);
                }
            }
        }
        float w_sq = warp_sum_bfly(tree8_sum(sum_sq));
        if (lane == 0) sRed[N_STACK][0][warp] = w_sq;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float w_dt = warp_sum_bfly(tree8_sum(dots[q]));
            if (lane == 0) sRed[N_STACK][1 + q][warp] = w_dt;
        }
    }

    // Step 2: Stream Stack[0..N_STACK-1] ONCE into SMEM (5x pipelined loads!)
    #pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
        const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
        uint4* dst_s = sStack + n * U4_PER_ROW;
        uint4 raws[5];
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            if (c < 4 || tid < 64) raws[c] = ld_global_cs_u4(sp + c * 1024 + tid * 8);
        }
        float sum_sq[8] = {0,0,0,0,0,0,0,0};
        float dots[NQ][8];
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) dots[q][i] = 0.0f;
        }
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int d_base = c * 1024 + tid * 8;
            if (c < 4 || tid < 64) {
                dst_s[d_base >> 3] = raws[c];
                float x[8];
                unpack_bf16x8(raws[c], x);
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum_sq[i] = fmaf(x[i], x[i], sum_sq[i]);
                #pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float qv[8];
                    load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dots[q][i] = fmaf(qv[i], x[i], dots[q][i]);
                }
            }
        }
        float w_sq = warp_sum_bfly(tree8_sum(sum_sq));
        if (lane == 0) sRed[n][0][warp] = w_sq;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float w_dt = warp_sum_bfly(tree8_sum(dots[q]));
            if (lane == 0) sRed[n][1 + q][warp] = w_dt;
        }
    }

    __syncthreads();

    // Step 3: Exact Pass 2 + Pass 3 from SMEM (Order-A FFMA Pass 2 matching live CUDA 13.0 ptxas!)
    const int l4 = lane & 3;
    float rstd_arr[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float tot_sq = warp4_sum_bfly(sRed[n][0][l4]);
        rstd_arr[n] = rsqrt_approx(fmaf(tot_sq, INV_D, eps));
    }

    const bool keep_out_l2 = (T <= 512);
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float my_lg = -INFINITY;
        float my_ir = 0.0f;
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float tot_dt = warp4_sum_bfly(sRed[n][1 + q][l4]);
            if (lane == n) {
                my_lg = __fmul_rn(tot_dt, rstd_arr[n]);
                my_ir = rstd_arr[n];
            }
        }
        float mx = my_lg;
        if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
        if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        float p = (lane < NS) ? tldevice_exp_sub(my_lg, mx) : 0.0f;
        float sp = p;
        if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
        if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
        if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
        if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
        float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
        p = __fmul_rn(p, my_ir);
        float probs[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) probs[n] = __shfl_sync(0xFFFFFFFFu, p, n);

        float acc[5][8];
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int d_base = c * 1024 + tid * 8;
            float sq_sum = 0.0f;
            if (c < 4 || tid < 64) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc[c][i] = 0.0f;
                const int u4_idx = d_base >> 3;
                #pragma unroll
                for (int n = 0; n < N_STACK; ++n) {
                    const float pn = probs[n];
                    float x[8];
                    unpack_bf16x8(sStack[n * U4_PER_ROW + u4_idx], x);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = fmaf(pn, x[i], acc[c][i]);
                }
                {
                    const float pn = probs[N_STACK];
                    float x[8];
                    if constexpr (HAS_MIX_LAST) {
                        float4 v0 = *reinterpret_cast<const float4*>(sLastF32 + d_base);
                        float4 v1 = *reinterpret_cast<const float4*>(sLastF32 + d_base + 4);
                        x[0]=v0.x; x[1]=v0.y; x[2]=v0.z; x[3]=v0.w;
                        x[4]=v1.x; x[5]=v1.y; x[6]=v1.z; x[7]=v1.w;
                    } else {
                        unpack_bf16x8(sLastU4[u4_idx], x);
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = fmaf(pn, x[i], acc[c][i]);
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc[c][i] = __fmul_rn(acc[c][i], inv_sum);
#ifdef INKLING_PTXAS133_STRICT
                float sq_v[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) sq_v[i] = __fmul_rn(acc[c][i], acc[c][i]);
                sq_sum = tree8_sum(sq_v);
#else
                // Order-A FFMA matching live CUDA 13.0 ptxas lowering of Triton _attn_res_kernel!
                float e01 = fmaf(acc[c][0], acc[c][0], __fmul_rn(acc[c][2], acc[c][2]));
                float o01 = fmaf(acc[c][1], acc[c][1], __fmul_rn(acc[c][3], acc[c][3]));
                float e23 = fmaf(acc[c][4], acc[c][4], __fmul_rn(acc[c][6], acc[c][6]));
                float o23 = fmaf(acc[c][5], acc[c][5], __fmul_rn(acc[c][7], acc[c][7]));
                sq_sum = __fadd_rn(__fadd_rn(e01, e23), __fadd_rn(o01, o23));
#endif
            }
            float w_sq = warp_sum_bfly(sq_sum);
            if (lane == 0) sSq2[c][warp] = w_sq;
        }

        __syncthreads();

        float tot_sum_sq = 0.0f;
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            tot_sum_sq = __fadd_rn(tot_sum_sq, warp4_sum_bfly(sSq2[c][l4]));
        }
        float out_rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(tot_sum_sq, (float)D), norm_eps));

        __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
        const __nv_bfloat16* nw_p = NormW + static_cast<int64_t>(q) * D;
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int d_base = c * 1024 + tid * 8;
            if (c < 4 || tid < 64) {
                float w[8];
                unpack_bf16x8(ld_const_u4(nw_p + d_base), w);
                float res[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) res[i] = __fmul_rn(__fmul_rn(acc[c][i], out_rstd), w[i]);
                uint4 out_u4 = pack_f32x8_to_bf16_u4(res);
                if (keep_out_l2) *reinterpret_cast<uint4*>(out_p + d_base) = out_u4;
                else             st_global_cs_u4(out_p + d_base, out_u4);
            }
        }
        if constexpr (NQ > 1) {
            if (q + 1 < NQ) __syncthreads();
        }
    }
}

static bool g_c32_smem_inited = false;

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t init_row_smem_one() {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    if constexpr (smem_bytes > 49152) {
        return cudaFuncSetAttribute(
            fused_sconv_ar_c32_row_kernel<N_STACK, NQ, HCR, HCA, HML>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
    }
    return cudaSuccess;
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t init_blockd_smem_one() {
    constexpr int D = 4608;
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * D * 2 + (HML ? static_cast<size_t>(D) * 4 : static_cast<size_t>(D) * 2);
    if constexpr (smem_bytes > 49152) {
        return cudaFuncSetAttribute(
            fused_sconv_ar_c32_blockd_kernel<N_STACK, NQ, HCR, HCA, HML>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
    }
    return cudaSuccess;
}

template <int N_STACK>
static inline cudaError_t init_smem_ns() {
    cudaError_t e = cudaSuccess;
    #define TRY_SMEM(fn) do { e = (fn); if (e != cudaSuccess) return e; } while (0)
    TRY_SMEM((init_row_smem_one<N_STACK, 1, false, false, false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 1, false, false, true>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 1, false, true,  false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 1, true,  true,  false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 3, false, true,  false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 3, true,  true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, false, false, false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, false, false, true>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, false, true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, true,  true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 3, false, true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 3, true,  true,  false>()));
    #undef TRY_SMEM
    return cudaSuccess;
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t launch_row_one(
    int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps
) {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    static bool optin_done = false;
    auto k = fused_sconv_ar_c32_row_kernel<N_STACK, NQ, HCR, HCA, HML>;
    if (!g_c32_smem_inited && !optin_done && smem_bytes > 49152) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        if (e != cudaSuccess) return e;
        optin_done = true;
    }
    k<<<dim3(T, 1, 1), dim3(256, 1, 1), smem_bytes, stream>>>(
        X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
        Pos, SeqIdx, Slot, BT, QStart,
        Stack, MixLast, Queries, NormW, Out,
        stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s,
        T, scale, eps, norm_eps
    );
    return cudaGetLastError();
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t launch_blockd_one(
    int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps
) {
    constexpr int D = 4608;
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * D * 2 + (HML ? static_cast<size_t>(D) * 4 : static_cast<size_t>(D) * 2);
    static bool optin_done = false;
    auto k = fused_sconv_ar_c32_blockd_kernel<N_STACK, NQ, HCR, HCA, HML>;
    if (!g_c32_smem_inited && !optin_done && smem_bytes > 49152) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        if (e != cudaSuccess) return e;
        optin_done = true;
    }
    k<<<dim3(T, 1, 1), dim3(128, 1, 1), smem_bytes, stream>>>(
        X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
        Pos, SeqIdx, Slot, BT, QStart,
        Stack, MixLast, Queries, NormW, Out,
        stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s,
        T, scale, eps, norm_eps
    );
    return cudaGetLastError();
}

template <int N_STACK, bool IS_ROW>
static cudaError_t dispatch_ns(
    int nq, int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps,
    bool hcr, bool hca, bool hml
) {
    #define GO(NQ_V, HCR_V, HCA_V, HML_V) \
        if constexpr (IS_ROW) { \
            return launch_row_one<N_STACK, NQ_V, HCR_V, HCA_V, HML_V>(T, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut, \
                Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, Out, \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s, scale, eps, norm_eps); \
        } else { \
            return launch_blockd_one<N_STACK, NQ_V, HCR_V, HCA_V, HML_V>(T, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut, \
                Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, Out, \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s, scale, eps, norm_eps); \
        }
    if (nq == 1) {
        if (!hca && !hcr && !hml) { GO(1, false, false, false); }
        if (!hca && !hcr &&  hml) { GO(1, false, false, true);  }
        if ( hca && !hcr && !hml) { GO(1, false, true,  false); }
        if ( hca &&  hcr && !hml) { GO(1, true,  true,  false); }
    } else if (nq == 3) {
        if ( hca && !hcr && !hml) { GO(3, false, true,  false); }
        if ( hca &&  hcr && !hml) { GO(3, true,  true,  false); }
    }
    #undef GO
    return cudaErrorInvalidValue;
}

}  // namespace

// Call once outside CUDA graph capture to opt-in >48 KiB dynamic SMEM for N_STACK=4..8!
extern "C" int fused_sconv_ar_c32_init_smem() {
    if (g_c32_smem_inited) return 0;
    cudaError_t e = cudaSuccess;
    if ((e = init_smem_ns<4>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_smem_ns<5>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_smem_ns<6>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_smem_ns<7>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_smem_ns<8>()) != cudaSuccess) return static_cast<int>(e);
    g_c32_smem_inited = true;
    return 0;
}

// Regime A entrypoint: exact 0-ULP match to short_conv + _attn_res_row_kernel (T * nq <= 256)
extern "C" int fused_sconv_ar_c32_row_fwd(
    const void* X, const void* CW, void* Cache,
    void* ConvOut, const void* ConvRes, const void* ConvAdd, void* AuxOut,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* Stack, const void* MixLast,
    const void* Queries, const void* NormW, void* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack,
    float scale, float eps, float norm_eps,
    int has_conv_res, int has_conv_add, int has_mix_last,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T * nq > 256) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int n_shift = (N_CACHE > 0 && (N_CACHE & (N_CACHE - 1)) == 0) ? __builtin_ctz(N_CACHE) : -1;
    #define CASE_R(N) case N: return (int)dispatch_ns<N, true>(nq, T, stream, \
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)CW, (__nv_bfloat16*)Cache, \
        (__nv_bfloat16*)ConvOut, (const __nv_bfloat16*)ConvRes, (const __nv_bfloat16*)ConvAdd, (__nv_bfloat16*)AuxOut, \
        (const int64_t*)Pos, (const int32_t*)SeqIdx, (const int64_t*)Slot, (const int32_t*)BT, (const int32_t*)QStart, \
        (const __nv_bfloat16*)Stack, (const __nv_bfloat16*)MixLast, \
        (const float*)Queries, (const __nv_bfloat16*)NormW, (__nv_bfloat16*)Out, \
        stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, n_shift, stream_offset, stride_s, \
        scale, eps, norm_eps, has_conv_res != 0, has_conv_add != 0, has_mix_last != 0)
    switch (n_stack) {
        CASE_R(1); CASE_R(2); CASE_R(3); CASE_R(4);
        CASE_R(5); CASE_R(6); CASE_R(7); CASE_R(8);
        default: return -1;
    }
    #undef CASE_R
}

// Regime B entrypoint: exact 0-ULP match to short_conv + _attn_res_kernel (T * nq > 256)
extern "C" int fused_sconv_ar_c32_blockd_fwd(
    const void* X, const void* CW, void* Cache,
    void* ConvOut, const void* ConvRes, const void* ConvAdd, void* AuxOut,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* Stack, const void* MixLast,
    const void* Queries, const void* NormW, void* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack,
    float scale, float eps, float norm_eps,
    int has_conv_res, int has_conv_add, int has_mix_last,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T * nq <= 256) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int n_shift = (N_CACHE > 0 && (N_CACHE & (N_CACHE - 1)) == 0) ? __builtin_ctz(N_CACHE) : -1;
    #define CASE_B(N) case N: return (int)dispatch_ns<N, false>(nq, T, stream, \
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)CW, (__nv_bfloat16*)Cache, \
        (__nv_bfloat16*)ConvOut, (const __nv_bfloat16*)ConvRes, (const __nv_bfloat16*)ConvAdd, (__nv_bfloat16*)AuxOut, \
        (const int64_t*)Pos, (const int32_t*)SeqIdx, (const int64_t*)Slot, (const int32_t*)BT, (const int32_t*)QStart, \
        (const __nv_bfloat16*)Stack, (const __nv_bfloat16*)MixLast, \
        (const float*)Queries, (const __nv_bfloat16*)NormW, (__nv_bfloat16*)Out, \
        stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, n_shift, stream_offset, stride_s, \
        scale, eps, norm_eps, has_conv_res != 0, has_conv_add != 0, has_mix_last != 0)
    switch (n_stack) {
        CASE_B(1); CASE_B(2); CASE_B(3); CASE_B(4);
        CASE_B(5); CASE_B(6); CASE_B(7); CASE_B(8);
        default: return -1;
    }
    #undef CASE_B
}
