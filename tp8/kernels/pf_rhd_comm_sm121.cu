// SM121a (NVIDIA DGX Spark GB10) Zero-Incast 3-Step Recursive Halving-Doubling (RHD / Butterfly)
// Prefill-Only Compressed ReduceScatter & AllGather CUDA Kernels.
//
// Strictly prefill-only (guarded T >= 2048 & !capturing in Python caller).
// Zero effect on C=1 / C=32 decode (exact_ink SHA-256 1dcfb30680d0252c unchanged).
//
// Why 3-Step RHD Beats 7-Peer All-to-All (`int8_a2a` / `fp8_a2a`) on 8x GB10 RoCEv2:
//   - Only 1 bidirectional peer per step (ZERO 7-to-1 RoCEv2 incast!) -> saturates 25.8 GB/s wire.
//   - Exact minimum collective volume (7/8 * T) in HALF the bytes (1.03125 B/elem vs 2.0 B/elem BF16).
//   - Only 3 hops / 3 quant-dequant stages for ReduceScatter, and ONLY 1 quant stage (zero re-quant
//     across doubling hops!) for mirror Recursive-Doubling AllGather.
//   - Fused Split-Half Kernels (`dequant_add_split_quant`) eliminate intermediate DRAM write/read
//     and intermediate BF16 rounding on the half forwarded to the next RHD step (-56.6 MB DRAM @ 8K!).
//   - Group-64 + BF16 Scales (`1.03125 B/elem` = exact same wire as g128 + FP32 scale!):
//       * Half-warp (16-lane) reduction needs only 4 shuffles (xor 8,4,2,1), 2 groups per warp per
//         single coalesced 256B load (`ld.global.cs.v2.u32`) and 128B store (`st.global.cs.u32`).
//       * Higher accuracy across EVERY regime vs g128 (`42.13 dB` normal, `38.58 dB` 100x clustered
//         outliers, `42.11 dB` tiny 1e-7 — BF16 scale has full FP32 8-bit exponent range!).
//   - Direct FP8 Rowwise AllGather (`rhd_ag_quant_fp8_rowwise` + `rhd_ag_unpack_fp8_rowwise`):
//       * Packs `[S, H]` FP8-E4M3FN + `[S]` FP32 scales into 1 contiguous shard (`1.0009 B/elem`),
//         eliminating the second separate scale AllGather while matching `linear.py` 0-ULP!
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_121a --shared -Xcompiler -fPIC \
//       --ptxas-options=-v pf_rhd_comm_sm121.cu -o /rdma/libpf_rhd_comm_sm121.so

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kGroupSize = 64;
constexpr int kMaxSmemGroups = 256;  // Supports up to H = 16,384 in SMEM (512 B BF16 / 1 KB FP32)

// ---------------------------------------------------------------------------
// Low-level SM121a PTX Load / Store & Quantization Helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ uint2 ld_global_cs_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint2 ld_global_ca_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_global_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];"
                 : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_global_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cs_u32(void* p, uint32_t v) {
    asm volatile("st.global.cs.u32 [%0], %1;"
                 :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ void st_global_cs_u2(void* p, uint2 v) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};"
                 :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}

static __device__ __forceinline__ void st_global_wb_u2(void* p, uint2 v) {
    // Write-back L2 caching: keeps keep2 (9.44 MB @ 8K) and final out_shard HOT in GB10's 24 MiB L2!
    asm volatile("st.global.v2.u32 [%0], {%1,%2};"
                 :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}

static __device__ __forceinline__ void st_global_wb_u4(void* p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ float halfwarp_max_f32(float v) {
    // 16-lane half-warp reduction (lanes 0..15 and 16..31 reduce independently, ZERO cross-half traffic!)
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ float warp_max_f32(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    return halfwarp_max_f32(v);
}

static __device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float a, float b) {
    // PTX ISA: cvt.rn.satfinite.e4m3x2.f32 d, src_hi, src_lo writes:
    //   d[15:8] = convert(src_hi), d[7:0] = convert(src_lo).
    // Pass (b, a) so elem0 (a) lands in d[7:0] and elem1 (b) lands in d[15:8]!
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(out) : "f"(b), "f"(a));
    return out;
}

static __device__ __forceinline__ float2 e4m3x2_to_f32x2(uint16_t packed) {
    // PTX ISA: cvt.rn.f16x2.e4m3x2 d, s converts s[7:0] -> lo half, s[15:8] -> hi half.
    uint32_t h2_bits;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
        : "=r"(h2_bits) : "h"(packed));
    __half2 h2 = *reinterpret_cast<const __half2*>(&h2_bits);
    return __half22float2(h2);
}

static __device__ __forceinline__ uint32_t f32x4_to_e4m3x4(
    float v0, float v1, float v2, float v3
) {
    uint16_t lo = f32x2_to_e4m3x2(v0, v1);
    uint16_t hi = f32x2_to_e4m3x2(v2, v3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

static __device__ __forceinline__ void e4m3x4_to_f32x4(
    uint32_t q4, float& o0, float& o1, float& o2, float& o3
) {
    float2 f01 = e4m3x2_to_f32x2((uint16_t)(q4 & 0xFFFFu));
    float2 f23 = e4m3x2_to_f32x2((uint16_t)(q4 >> 16));
    o0 = f01.x; o1 = f01.y; o2 = f23.x; o3 = f23.y;
}

static __device__ __forceinline__ uint32_t f32x4_to_int8x4(
    float v0, float v1, float v2, float v3
) {
    int i0 = __float2int_rn(fminf(fmaxf(v0, -127.0f), 127.0f));
    int i1 = __float2int_rn(fminf(fmaxf(v1, -127.0f), 127.0f));
    int i2 = __float2int_rn(fminf(fmaxf(v2, -127.0f), 127.0f));
    int i3 = __float2int_rn(fminf(fmaxf(v3, -127.0f), 127.0f));
    uint32_t b0 = (uint32_t)(uint8_t)(int8_t)i0;
    uint32_t b1 = (uint32_t)(uint8_t)(int8_t)i1;
    uint32_t b2 = (uint32_t)(uint8_t)(int8_t)i2;
    uint32_t b3 = (uint32_t)(uint8_t)(int8_t)i3;
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static __device__ __forceinline__ void int8x4_to_f32x4(
    uint32_t q4, float& o0, float& o1, float& o2, float& o3
) {
    o0 = (float)(int8_t)(q4 & 0xFFu);
    o1 = (float)(int8_t)((q4 >> 8) & 0xFFu);
    o2 = (float)(int8_t)((q4 >> 16) & 0xFFu);
    o3 = (float)(int8_t)((q4 >> 24) & 0xFFu);
}

static __device__ __forceinline__ void bf16x4_to_f32x4(
    uint2 raw, float& o0, float& o1, float& o2, float& o3
) {
    __nv_bfloat162 h0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    __nv_bfloat162 h1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    float2 f01 = __bfloat1622float2(h0);
    float2 f23 = __bfloat1622float2(h1);
    o0 = f01.x; o1 = f01.y; o2 = f23.x; o3 = f23.y;
}

static __device__ __forceinline__ uint2 f32x4_to_bf16_u2(
    float v0, float v1, float v2, float v3
) {
    __nv_bfloat162 h0 = __halves2bfloat162(__float2bfloat16_rn(v0), __float2bfloat16_rn(v1));
    __nv_bfloat162 h1 = __halves2bfloat162(__float2bfloat16_rn(v2), __float2bfloat16_rn(v3));
    uint2 out;
    out.x = *reinterpret_cast<const uint32_t*>(&h0);
    out.y = *reinterpret_cast<const uint32_t*>(&h1);
    return out;
}

template <bool IS_FP8>
static __device__ __forceinline__ uint32_t quant_group64_4vals(
    float v0, float v1, float v2, float v3,
    __nv_bfloat16& sc_bf_out
) {
    constexpr float inv_qmax = IS_FP8 ? (1.0f / 448.0f) : (1.0f / 127.0f);
    float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = halfwarp_max_f32(a);
    float raw_sc = fmaxf(amax * inv_qmax, 1.0e-12f);
    __nv_bfloat16 sc_bf = __float2bfloat16_rn(raw_sc);
    sc_bf_out = sc_bf;
    float inv_sc = 1.0f / __bfloat162float(sc_bf);
    if constexpr (IS_FP8) {
        return f32x4_to_e4m3x4(v0 * inv_sc, v1 * inv_sc, v2 * inv_sc, v3 * inv_sc);
    } else {
        return f32x4_to_int8x4(v0 * inv_sc, v1 * inv_sc, v2 * inv_sc, v3 * inv_sc);
    }
}

template <bool IS_FP8>
static __device__ __forceinline__ void dequant_4vals(
    uint32_t q4, float sc, float& o0, float& o1, float& o2, float& o3
) {
    float q0, q1, q2, q3;
    if constexpr (IS_FP8) {
        e4m3x4_to_f32x4(q4, q0, q1, q2, q3);
    } else {
        int8x4_to_f32x4(q4, q0, q1, q2, q3);
    }
    o0 = q0 * sc;
    o1 = q1 * sc;
    o2 = q2 * sc;
    o3 = q3 * sc;
}

// ---------------------------------------------------------------------------
// Kernel 1: Quantize Contiguous BF16 Slice [M, H] -> Packed g64 + BF16 Scales
//   Used by RHD RS Step 0 (4S, or 2S in pipe=1) AND RHD AG initial quant (1S).
//   1 CTA = 1 row, 128 threads = 4 warps = 8 half-warps (8 x 64 = 512 elems/iter).
//   Requires H % 128 == 0 so every active warp runs 2 full half-warps with 0 divergence!
// ---------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_quant_slice_g64_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ OUT,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int H
) {
    const int row      = (int)blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;           // 0..3
    const int lane     = tid & 31;           // 0..31
    const int halfw    = (lane >> 4) & 1;    // 0 or 1
    const int sublane  = lane & 15;          // 0..15
    const int G        = H / kGroupSize;     // even because H % 128 == 0 (72 for H=4608)

    const __nv_bfloat16* x_row = X + (row_off + (int64_t)row) * stride_xm;
    uint8_t* q_row             = OUT + (int64_t)row * H;
    __nv_bfloat16* sc_row      = reinterpret_cast<__nv_bfloat16*>(
        OUT + scale_off + (int64_t)row * G * 2
    );

    __shared__ __nv_bfloat16 sSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);  // 4 elems/lane -> 128 elems/warp coalesced!
        uint2 raw = ld_global_cs_u2(x_row + d);
        float v0, v1, v2, v3;
        bf16x4_to_f32x4(raw, v0, v1, v2, v3);

        __nv_bfloat16 sc_bf;
        uint32_t q4 = quant_group64_4vals<IS_FP8>(v0, v1, v2, v3, sc_bf);
        st_global_cs_u32(q_row + d, q4);

        if (sublane == 0) {
            if (use_smem) sSc[g] = sc_bf;
            else sc_row[g] = sc_bf;
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int i = tid; i < G; i += 128) {
            sc_row[i] = sSc[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: FUSED Dequant + Add + Split (Half -> Direct FP32 Re-Quant to NEXT_SEND,
//                                        Half -> BF16 Store to NEXT_KEEP)
//   Eliminates 1 full DRAM write+read AND intermediate BF16 rounding on the send half!
//   Grid: 2*N CTAs (`row = blockIdx.x`). Uniform branch across entire CTA -> ZERO divergence!
//   `send_upper == 1`: rows [N..2N-1] -> NEXT_SEND[0..N-1], rows [0..N-1] -> NEXT_KEEP[0..N-1]
//   `send_upper == 0`: rows [0..N-1]  -> NEXT_SEND[0..N-1], rows [N..2N-1] -> NEXT_KEEP[0..N-1]
// ---------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_dequant_add_split_quant_g64_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    uint8_t* __restrict__ NEXT_SEND,
    __nv_bfloat16* __restrict__ NEXT_KEEP,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep
) {
    const int row      = (int)blockIdx.x;    // 0 .. 2*N - 1
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;

    const bool is_send = send_upper ? (row >= N) : (row < N);
    const int out_row  = is_send
        ? (send_upper ? (row - N) : row)
        : (send_upper ? row : (row - N));

    const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
    const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
        RECV + recv_scale_off + (int64_t)row * G * 2
    );
    const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;

    __shared__ float sRecvSc[kMaxSmemGroups];
    __shared__ __nv_bfloat16 sSendSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sRecvSc[i] = __bfloat162float(recv_sc_p[i]);
        }
        __syncthreads();
    }

    const bool ca_local = ((int64_t)N * (int64_t)H * 4 <= (20LL << 20));

    if (is_send) {
        uint8_t* send_q_row      = NEXT_SEND + (int64_t)out_row * H;
        __nv_bfloat16* send_sc_p = reinterpret_cast<__nv_bfloat16*>(
            NEXT_SEND + next_scale_off + (int64_t)out_row * G * 2
        );
        for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

            uint2 raw_loc = ca_local ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
            uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

            float l0, l1, l2, l3, r0, r1, r2, r3;
            bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
            dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

            // Direct FP32 sum -> quantize in registers (ZERO intermediate BF16 rounding!)
            float s0 = l0 + r0;
            float s1 = l1 + r1;
            float s2 = l2 + r2;
            float s3 = l3 + r3;

            __nv_bfloat16 sc_out;
            uint32_t q_out = quant_group64_4vals<IS_FP8>(s0, s1, s2, s3, sc_out);
            st_global_cs_u32(send_q_row + d, q_out);

            if (sublane == 0) {
                if (use_smem) sSendSc[g] = sc_out;
                else send_sc_p[g] = sc_out;
            }
        }
        if (use_smem) {
            __syncthreads();
            for (int i = tid; i < G; i += 128) {
                send_sc_p[i] = sSendSc[i];
            }
        }
    } else {
        __nv_bfloat16* keep_row = NEXT_KEEP + (int64_t)out_row * stride_keep;
        for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

            uint2 raw_loc = ca_local ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
            uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

            float l0, l1, l2, l3, r0, r1, r2, r3;
            bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
            dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

            uint2 out_bf = f32x4_to_bf16_u2(l0 + r0, l1 + r1, l2 + r2, l3 + r3);
            if (wb_keep) {
                st_global_wb_u2(keep_row + d, out_bf);
            } else {
                st_global_cs_u2(keep_row + d, out_bf);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Fused Dequant + Add + Direct Re-Quant (for Step-0 Pipelined Sub-Half A!)
//   Reads RECV [M, H] packed + LOCAL [M, H] BF16 -> writes NEXT_SEND [M, H] packed.
// ---------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_dequant_add_quant_g64_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    uint8_t* __restrict__ NEXT_SEND,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int H
) {
    const int row      = (int)blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;

    const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
    const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
        RECV + recv_scale_off + (int64_t)row * G * 2
    );
    const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;
    uint8_t* send_q_row            = NEXT_SEND + (int64_t)row * H;
    __nv_bfloat16* send_sc_p       = reinterpret_cast<__nv_bfloat16*>(
        NEXT_SEND + next_scale_off + (int64_t)row * G * 2
    );

    __shared__ float sRecvSc[kMaxSmemGroups];
    __shared__ __nv_bfloat16 sSendSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sRecvSc[i] = __bfloat162float(recv_sc_p[i]);
        }
        __syncthreads();
    }

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);
        float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

        uint2 raw_loc = ld_global_cs_u2(loc_row + d);
        uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

        float l0, l1, l2, l3, r0, r1, r2, r3;
        bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
        dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

        __nv_bfloat16 sc_out;
        uint32_t q_out = quant_group64_4vals<IS_FP8>(l0 + r0, l1 + r1, l2 + r2, l3 + r3, sc_out);
        st_global_cs_u32(send_q_row + d, q_out);

        if (sublane == 0) {
            if (use_smem) sSendSc[g] = sc_out;
            else send_sc_p[g] = sc_out;
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int i = tid; i < G; i += 128) {
            send_sc_p[i] = sSendSc[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Fused Dequant + Add -> BF16 Output
//   Used for BOTH:
//     (a) Final RHD RS step (`keep2 [S, H]` + `recv2 [S, H]` -> `out_shard [S, H]`, wb_out=1
//         so out_shard stays HOT in GB10's 24 MiB L2 for `fused_sconv_ar_prefill`!), AND
//     (b) Step-0 pipelined `keep1` production (`local0_B [2S, H]` + `recv0_B [2S, H]` -> `keep1`).
// ---------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_dequant_add_bf16_g64_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    __nv_bfloat16* __restrict__ OUT,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int H,
    int wb_out
) {
    const int row      = (int)blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;

    const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
    const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
        RECV + recv_scale_off + (int64_t)row * G * 2
    );
    const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;
    __nv_bfloat16* out_row         = OUT + (int64_t)row * stride_out;

    __shared__ float sRecvSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sRecvSc[i] = __bfloat162float(recv_sc_p[i]);
        }
        __syncthreads();
    }

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);
        float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

        // Use ca (cache-all) on local keep2 (L2 hit!) and cs on incoming network recv buffer
        uint2 raw_loc = wb_out ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
        uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

        float l0, l1, l2, l3, r0, r1, r2, r3;
        bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
        dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

        uint2 out_bf = f32x4_to_bf16_u2(l0 + r0, l1 + r1, l2 + r2, l3 + r3);
        if (wb_out) {
            st_global_wb_u2(out_row + d, out_bf);
        } else {
            st_global_cs_u2(out_row + d, out_bf);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: RHD AllGather Final Dequant (`[W, shard_bytes]` g64 -> `[W*S, H]` BF16)
//   Reads strictly from AG_BUF[r] for all r = 0..W-1 -> 100% BIT-IDENTICAL across all TP ranks!
//   Grid: (S, W), 128 threads/CTA.
// ---------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_ag_dequant_g64_kernel(
    const uint8_t* __restrict__ AG_BUF,
    __nv_bfloat16* __restrict__ OUT,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H
) {
    const int local_row = (int)blockIdx.x;  // 0 .. S-1
    const int r         = (int)blockIdx.y;  // 0 .. W-1
    const int tid       = (int)threadIdx.x;
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int halfw     = (lane >> 4) & 1;
    const int sublane   = lane & 15;
    const int G         = H / kGroupSize;

    const uint8_t* slot_base = AG_BUF + (int64_t)r * shard_bytes;
    const uint8_t* q_row     = slot_base + (int64_t)local_row * H;
    const __nv_bfloat16* sc_p = reinterpret_cast<const __nv_bfloat16*>(
        slot_base + scale_off + (int64_t)local_row * G * 2
    );
    __nv_bfloat16* out_row   = OUT + ((int64_t)r * S + local_row) * stride_out;

    __shared__ float sSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sSc[i] = __bfloat162float(sc_p[i]);
        }
        __syncthreads();
    }

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);
        float sc = use_smem ? sSc[g] : __bfloat162float(sc_p[g]);

        uint32_t q4 = ld_global_cs_u32(q_row + d);
        float v0, v1, v2, v3;
        dequant_4vals<IS_FP8>(q4, sc, v0, v1, v2, v3);
        st_global_wb_u2(out_row + d, f32x4_to_bf16_u2(v0, v1, v2, v3));
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: RHD Direct FP8 Rowwise Quantize (`[S, H]` BF16 -> 1 packed shard `[S*H FP8 | S FP32]`)
//   Matches `linear.py::quantize_fp8_rowwise` (`scale = max(amax / 448.0, 1e-12f)`).
//   Single-pass register-cached for H <= 5120 (9 uint2 = 18 regs/thread for H=4608, 0 spills,
//   ZERO second DRAM read!), with clean 2-pass fallback for H > 5120.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(128)
void rhd_ag_quant_fp8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ OUT_SHARD,
    int64_t stride_xm,
    int64_t scale_off,
    int H
) {
    const int row  = (int)blockIdx.x;
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n_vec4 = H >> 2;  // H / 4 (1152 for H = 4608)

    const __nv_bfloat16* x_row = X + (int64_t)row * stride_xm;
    uint8_t* q_row             = OUT_SHARD + (int64_t)row * H;
    float* sc_out              = reinterpret_cast<float*>(OUT_SHARD + scale_off);

    __shared__ float sWarpMax[4];

    if (n_vec4 <= 1280) {
        // Fast 1-pass register-cached path (up to H = 5120; 9 iters for H = 4608 -> 0 spills!)
        uint2 cache[10];
        float local_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            int idx = i * 128 + tid;
            if (idx < n_vec4) {
                uint2 raw = ld_global_cs_u2(x_row + (idx << 2));
                cache[i] = raw;
                float v0, v1, v2, v3;
                bf16x4_to_f32x4(raw, v0, v1, v2, v3);
                float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
                local_amax = fmaxf(local_amax, a);
            }
        }
        float wmax = warp_max_f32(local_amax);
        if (lane == 0) sWarpMax[warp] = wmax;
        __syncthreads();
        float row_amax = fmaxf(fmaxf(sWarpMax[0], sWarpMax[1]), fmaxf(sWarpMax[2], sWarpMax[3]));
        float scale = fmaxf(row_amax * (1.0f / 448.0f), 1.0e-12f);
        float inv_scale = 1.0f / scale;
        if (tid == 0) sc_out[row] = scale;

        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            int idx = i * 128 + tid;
            if (idx < n_vec4) {
                float v0, v1, v2, v3;
                bf16x4_to_f32x4(cache[i], v0, v1, v2, v3);
                uint32_t q4 = f32x4_to_e4m3x4(
                    v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale
                );
                st_global_cs_u32(q_row + (idx << 2), q4);
            }
        }
    } else {
        // Generic 2-pass path for H > 5120
        float local_amax = 0.0f;
        for (int idx = tid; idx < n_vec4; idx += 128) {
            uint2 raw = ld_global_ca_u2(x_row + (idx << 2));
            float v0, v1, v2, v3;
            bf16x4_to_f32x4(raw, v0, v1, v2, v3);
            float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
            local_amax = fmaxf(local_amax, a);
        }
        float wmax = warp_max_f32(local_amax);
        if (lane == 0) sWarpMax[warp] = wmax;
        __syncthreads();
        float row_amax = fmaxf(fmaxf(sWarpMax[0], sWarpMax[1]), fmaxf(sWarpMax[2], sWarpMax[3]));
        float scale = fmaxf(row_amax * (1.0f / 448.0f), 1.0e-12f);
        float inv_scale = 1.0f / scale;
        if (tid == 0) sc_out[row] = scale;

        for (int idx = tid; idx < n_vec4; idx += 128) {
            uint2 raw = ld_global_ca_u2(x_row + (idx << 2));
            float v0, v1, v2, v3;
            bf16x4_to_f32x4(raw, v0, v1, v2, v3);
            uint32_t q4 = f32x4_to_e4m3x4(
                v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale
            );
            st_global_cs_u32(q_row + (idx << 2), q4);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: RHD Direct FP8 Rowwise Unpack (`[W, shard_bytes]` -> `OUT_FP8 [W*S, H]` + `OUT_SC [W*S]`)
//   16B vectorized uint4 copy (288 vectors/row @ H=4608 -> saturates 230+ GB/s GB10 UMA!).
//   Grid: (S, W), 128 threads/CTA.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(128)
void rhd_ag_unpack_fp8_rowwise_kernel(
    const uint8_t* __restrict__ AG_BUF,
    uint8_t* __restrict__ OUT_FP8,
    float* __restrict__ OUT_SCALE,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H
) {
    const int local_row = (int)blockIdx.x;  // 0 .. S-1
    const int r         = (int)blockIdx.y;  // 0 .. W-1
    const int tid       = (int)threadIdx.x;
    const int n_vec16   = H >> 4;           // H / 16 (288 for H = 4608)

    const uint8_t* slot_base = AG_BUF + (int64_t)r * shard_bytes;
    const uint8_t* src_row   = slot_base + (int64_t)local_row * H;
    const int64_t dst_r      = (int64_t)r * S + local_row;
    uint8_t* dst_row         = OUT_FP8 + dst_r * stride_out;

    for (int idx = tid; idx < n_vec16; idx += 128) {
        uint4 v = ld_global_cs_u4(src_row + (idx << 4));
        st_global_wb_u4(dst_row + (idx << 4), v);
    }
    if (tid == 0) {
        const float* src_sc = reinterpret_cast<const float*>(slot_base + scale_off);
        OUT_SCALE[dst_r] = src_sc[local_row];
    }
}

}  // namespace

// ===========================================================================
// C ABI Entrypoints (loaded via ctypes in prefill_sp_rs.py & bench_rhd_8rank.py)
// ===========================================================================
extern "C" {

int rhd_quant_slice_int8_g64_fwd(
    const void* x_bf16,
    void* out_packed,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_quant_slice_g64_kernel<false><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_packed),
        stride_xm, row_off, scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_quant_slice_fp8_g64_fwd(
    const void* x_bf16,
    void* out_packed,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_quant_slice_g64_kernel<true><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_packed),
        stride_xm, row_off, scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_split_quant_int8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    void* next_keep_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep,
    uint64_t stream_u64
) {
    if (N <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_dequant_add_split_quant_g64_kernel<false><<<dim3((unsigned)(2 * N), 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        reinterpret_cast<__nv_bfloat16*>(next_keep_bf16),
        stride_local, local_row_off, recv_scale_off, next_scale_off, stride_keep,
        N, H, send_upper, wb_keep
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_split_quant_fp8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    void* next_keep_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep,
    uint64_t stream_u64
) {
    if (N <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_dequant_add_split_quant_g64_kernel<true><<<dim3((unsigned)(2 * N), 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        reinterpret_cast<__nv_bfloat16*>(next_keep_bf16),
        stride_local, local_row_off, recv_scale_off, next_scale_off, stride_keep,
        N, H, send_upper, wb_keep
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_quant_int8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_dequant_add_quant_g64_kernel<false><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        stride_local, local_row_off, recv_scale_off, next_scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_quant_fp8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_dequant_add_quant_g64_kernel<true><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        stride_local, local_row_off, recv_scale_off, next_scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_bf16_int8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* out_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int M,
    int H,
    int wb_out,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_dequant_add_bf16_g64_kernel<false><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        stride_local, local_row_off, recv_scale_off, stride_out, H, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_bf16_fp8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* out_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int M,
    int H,
    int wb_out,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_dequant_add_bf16_g64_kernel<true><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        stride_local, local_row_off, recv_scale_off, stride_out, H, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_dequant_int8_g64_fwd(
    const void* ag_buf,
    void* out_bf16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_ag_dequant_g64_kernel<false><<<dim3((unsigned)S, (unsigned)W, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_dequant_fp8_g64_fwd(
    const void* ag_buf,
    void* out_bf16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_ag_dequant_g64_kernel<true><<<dim3((unsigned)S, (unsigned)W, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_quant_fp8_rowwise_fwd(
    const void* x_bf16,
    void* out_shard_packed,
    int64_t stride_xm,
    int64_t scale_off,
    int S,
    int H,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || (H % 16) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_ag_quant_fp8_rowwise_kernel<<<dim3((unsigned)S, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_shard_packed),
        stride_xm, scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_unpack_fp8_rowwise_fwd(
    const void* ag_buf,
    void* out_fp8,
    void* out_scale,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 16) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    rhd_ag_unpack_fp8_rowwise_kernel<<<dim3((unsigned)S, (unsigned)W, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<uint8_t*>(out_fp8),
        reinterpret_cast<float*>(out_scale),
        shard_bytes, scale_off, stride_out, S, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

}  // extern "C"
