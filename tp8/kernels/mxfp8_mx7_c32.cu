// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) C=32 (B=32, M=17..256) 0-ULP Bit-Exact MX7 (7.28 bpw) MoE Kernels
// Targets Inkling-2-Small TP8 L40..47 MXFP8 MoE (E=384, H=4608, inter=288, topk=8):
//   - MX_W13 (N=576,  K=4608): BN=32  (4 warps, 18 panels), 896 B 128B-aligned MX7 slabs
//   - MX_W2  (N=4608, K=288):  BN=128 (4 warps, 36 panels, SUB_N=4), ZERO-repack 1-to-1 bijection
//                              onto existing MX7 W2 slabs (`packed_wuid = e*144 + nb*4 + warp`)!
//
// Key C=32 (M=17..256) Architecture (100% 0-ULP bit-exact BF16 MMA, literal nd == 0):
//   1. Eliminates the live C=32 M=129..256 (TILE=32) MX7 skip bug (where `INKLING_MX7_FREE_RAW=0`
//      caused `trtllm_bmm.py:1391` to fall back to Triton raw 8.25 bpw FP8: 7.72 + 4.07 = 11.79 ms
//      in live trace #96!) AND beats `mx7_v4_prefill` (which ran non-persistent 2D grids with 4 warps/SM!).
//   2. Native single-pass BM=32 at TILE=32 (ZERO weight re-reads or re-unpacks across live_m in [17,32]!):
//        - MX_W13 TILE=16: BM=16, BK_SMEM=256, SA_STRIDE=272 -> 18.0 KiB -> 5 CTAs/SM = 240 CTAs (0-way LDS.64)
//        - MX_W13 TILE=32: BM=32, BK_SMEM=128, SA_STRIDE=144 -> 19.1 KiB -> 5 CTAs/SM = 240 CTAs (0-way LDS.64)
//        - MX_W2  TILE=16: BM=16, BN=128, SA_STRIDE=304      -> 10.3 KiB -> 6 CTAs/SM = 288 CTAs (6x48 wave)
//        - MX_W2  TILE=32: BM=32, BN=128, SA_STRIDE=304      -> 20.0 KiB -> 4 CTAs/SM = 192 CTAs (6x32 wave)
//   3. Tight Variable-Count Row Predication: dynamic `has_m8`, `has_m16`, `has_m24` flags skip ALL dead
//      8-row SMEM loads and skip ALL `im=1` Tensor Core MMAs when `live_m <= 16`.
//   4. Zero-Scatter `Row2Tok` (`sTok[BM]` SMEM cache + row-major `AS`), inter-tile slab-0 L2 prefetch,
//      `ld.global.cs` L2 eviction on weights, exact BF16 `m16n8k16` ascending K-order (literal `nd == 0`).
//   5. Fused W13 SiTU -> compact dense `SiTU_Out[compact_row, 288]` (<=1.125 MiB, ZERO atomics/barriers!)
//      + Fused W2 register-compact OTF MXFP8 quant-dequant (`<=22` regs, overlapped with slab-0 L2 prefetch).
//   6. Shares the EXACT same 36-panel 6-Panel-Wave Double-Buffered Compact L2-Resident `DownPanel`
//      (`0.75..6.0 MiB` TOTAL -> 100% L2-resident!) + `PanelDone[36]` + `BufGen[12]` combine as `NV_W2`,
//      summing in exact slot `0..7` FP32 order directly into `SendBuf`!
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC mxfp8_mx7_c32.cu -o libmxfp8_mx7_c32.so
//   (CRITICAL: --fmad=false required; OMIT --ftz=true to match Triton ftz=False!)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

static constexpr int MX7_SLAB_BYTES = 896;  // 512 sm4 + 256 e21 + 128 e0 = 7 * 128 B (128B aligned)
static constexpr float FP32_MAX_VAL = 3.4028234663852886e38f;
static constexpr float INV_25 = 1.0f / 25.0f;
static constexpr float INV_448 = 1.0f / 448.0f;

// ---- PTX Memory, Sync & Math Helpers ----

static __device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint2 ld_cs_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint2 ld_cg_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_cg_u32(void* p, uint32_t v) {
    asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

static __device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

static __device__ __forceinline__ uint32_t ld_acq_gpu_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ void st_rel_gpu_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ float add_rn(float a, float b) {
    float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float mul_rn(float a, float b) {
    float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float rcp_rn(float a) {
    float r; asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
static __device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}
static __device__ __forceinline__ uint16_t f32_to_bf16_rn(float f) {
    uint16_t b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(f));
    return b;
}
static __device__ __forceinline__ uint32_t pack_bf16x2(float f0, float f1) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1));
    return out;
}

static __device__ __forceinline__ float e8m0_to_f32(uint8_t s) {
    if (s == 0) return 5.877471754111438e-39f;
    if (s == 255) return __int_as_float(0x7FC00000);
    return __uint_as_float(static_cast<uint32_t>(s) << 23);
}

static __device__ __forceinline__ int64_t scale_offset_decode(int64_t row, int col, int cols) {
    int64_t tile = (row >> 3) * ((cols + 3) >> 2) + (col >> 2);
    return (tile << 5) + ((row & 7) << 2) + (col & 3);
}

struct SlabMX7 {
    uint4 sm;     // 16 B/lane (512 B)
    uint2 e21;    //  8 B/lane (256 B)
    uint32_t e0;  //  4 B/lane (128 B)
    uint4 ws3;    // 16 B broadcast from separate 16B-aligned WS3 buffer
};

static __device__ __forceinline__ SlabMX7 load_slab_mx7(
    const uint8_t* sp, const uint4* wsp, int lane
) {
    SlabMX7 r;
    r.sm  = ld_cs_u4(sp + lane * 16);
    r.e21 = ld_cs_u2(sp + 512 + lane * 8);
    r.e0  = ld_cs_u32(sp + 768 + lane * 4);
    r.ws3 = ld_cs_u4(wsp);
    return r;
}

static __device__ __forceinline__ uint32_t build_e4_pack_flip(
    uint32_t sm16, uint32_t e21_8, uint32_t e0_4, uint32_t flip4
) {
    uint32_t s_lo = (sm16 & 0x000Fu)
                  | ((sm16 & 0x00F0u) << 4)
                  | ((sm16 & 0x0F00u) << 8)
                  | ((sm16 & 0xF000u) << 12);
    uint32_t e21_b = ((e21_8 & 0x03u) << 4)
                   | ((e21_8 & 0x0Cu) << 10)
                   | ((e21_8 & 0x30u) << 16)
                   | ((e21_8 & 0xC0u) << 22);
    uint32_t e0_b = ((e0_4 & 1u) << 3)
                  | ((e0_4 & 2u) << 10)
                  | ((e0_4 & 4u) << 17)
                  | ((e0_4 & 8u) << 24);
    uint32_t e3_flip = ((flip4 & 1u) << 6)
                     | ((flip4 & 2u) << 13)
                     | ((flip4 & 4u) << 20)
                     | ((flip4 & 8u) << 27);
    return (0x40404040u ^ e3_flip) | ((s_lo & 0x08080808u) << 4) | e21_b | e0_b | (s_lo & 0x07070707u);
}

// Exact match to Triton: cvt.rn.f16x2.e4m3x2 -> cvt.f32.f16 -> mul.rn.f32 -> cvt.rn.bf16x2.f32
static __device__ __forceinline__ void e4m3x4_scale_to_bf16x4(
    uint32_t e4_pack, float sc, uint32_t& b0, uint32_t& b1
) {
    uint32_t h0, h1;
    asm volatile(
        "{\n\t"
        ".reg .b16 p0, p1;\n\t"
        "mov.b32 {p0, p1}, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %0, p0;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, p1;\n\t"
        "}\n\t"
        : "=r"(h0), "=r"(h1) : "r"(e4_pack)
    );
    float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
    float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
    float x0 = mul_rn(f0.x, sc), y0 = mul_rn(f0.y, sc);
    float x1 = mul_rn(f1.x, sc), y1 = mul_rn(f1.y, sc);
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "f"(x0), "f"(y0));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "f"(x1), "f"(y1));
}

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&acc)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

static __device__ __forceinline__ uint32_t finish_slab_flip_mask(
    uint32_t rec, const uint16_t* __restrict__ FLIPS, int32_t& f_cur, int32_t f_end,
    uint32_t sl, int lane, uint32_t* sFlipWarp
) {
    bool in_slab = (rec != 0xFFFFu) && ((rec >> 10) == sl);
    uint32_t m = __ballot_sync(0xFFFFFFFFu, in_slab);
    if (__builtin_expect(m == 0u, 0)) return 0u;
    sFlipWarp[lane] = 0u;
    __syncwarp();
    if (in_slab) {
        const uint32_t s  = (rec >> 7) & 7u;
        const uint32_t tl = (rec >> 2) & 31u;
        const uint32_t j  = rec & 3u;
        atomicOr(&sFlipWarp[tl], 1u << ((s << 2) | j));
    }
    int cnt = __popc(m);
    f_cur += cnt;
    while (__builtin_expect(cnt == 32 && f_cur < f_end, 0)) {
        const int32_t idx = f_cur + lane;
        const uint32_t r2 = (idx < f_end) ? static_cast<uint32_t>(__ldg(FLIPS + idx)) : 0xFFFFu;
        const bool in2 = (r2 != 0xFFFFu) && ((r2 >> 10) == sl);
        const uint32_t m2 = __ballot_sync(0xFFFFFFFFu, in2);
        if (m2 == 0u) break;
        if (in2) {
            const uint32_t s  = (r2 >> 7) & 7u;
            const uint32_t tl = (r2 >> 2) & 31u;
            const uint32_t j  = r2 & 3u;
            atomicOr(&sFlipWarp[tl], 1u << ((s << 2) | j));
        }
        cnt = __popc(m2);
        f_cur += cnt;
    }
    __syncwarp();
    return sFlipWarp[lane];
}

// ---- Exact 0-ULP SiTU Helpers (GATE_CAP=4, UP_CAP=25) ----

static __device__ __forceinline__ float sigmoid_approx_exact(float x) {
    float r;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }"
        : "=f"(r) : "f"(x));
    return r;
}

static __device__ __forceinline__ float softcap_4(float x) {
    float q = mul_rn(x, 0.25f);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(4.0f, t);
}

static __device__ __forceinline__ float softcap_25(float x) {
    float q = mul_rn(x, INV_25);
    float res = fmaxf(fmaf(q, 25.0f, -x), -FP32_MAX_VAL);
    q = fmaf(-res, INV_25, q);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(25.0f, t);
}

static __device__ __forceinline__ uint16_t situ_pair_bf16_exact(uint32_t pair, float gamma) {
    float gate = bf16_bits_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
    float up   = bf16_bits_to_f32(static_cast<uint16_t>(pair >> 16));
    float sig  = sigmoid_approx_exact(gate);
    gate = softcap_4(gate);
    up   = softcap_25(up);
    float y = mul_rn(mul_rn(mul_rn(gate, sig), up), gamma);
    return f32_to_bf16_rn(y);
}

// ---- Exact 0-ULP Register-Compact MXFP8 32-elem Quant-Dequant Directly Into sA_all ----
static __device__ __forceinline__ void quant_dequant_mxfp8_32_to_sA(
    const uint4 v[4], __nv_bfloat16* dst
) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
            float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
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

    const float a_sc = e8m0_to_f32(static_cast<uint8_t>(exp_u));

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t u[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t u0 = u[2 * h + 0];
            const uint32_t u1 = u[2 * h + 1];
            float f0 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u0 & 0xFFFFu)), inv);
            float f1 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u0 >> 16)),      inv);
            float f2 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u1 & 0xFFFFu)), inv);
            float f3 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u1 >> 16)),      inv);
            if (f0 == f0) { if (f0 < -448.0f) f0 = -448.0f; if (f0 > 448.0f) f0 = 448.0f; }
            if (f1 == f1) { if (f1 < -448.0f) f1 = -448.0f; if (f1 > 448.0f) f1 = 448.0f; }
            if (f2 == f2) { if (f2 < -448.0f) f2 = -448.0f; if (f2 > 448.0f) f2 = 448.0f; }
            if (f3 == f3) { if (f3 < -448.0f) f3 = -448.0f; if (f3 > 448.0f) f3 = 448.0f; }
            uint16_t e_lo, e_hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e_lo) : "f"(f0), "f"(f1));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e_hi) : "f"(f2), "f"(f3));
            uint32_t a4 = static_cast<uint32_t>(e_lo) | (static_cast<uint32_t>(e_hi) << 16);
            uint32_t b0, b1;
            e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
            *reinterpret_cast<uint2*>(dst + i * 8 + h * 4) = make_uint2(b0, b1);
        }
    }
}

// =========================================================================
// Self-Contained GPU Packer (100% ABI identical to v4/v5)
// =========================================================================
template <bool FROM_PANEL>
static __device__ __forceinline__ uint8_t read_w_elem(
    const uint8_t* W, int e, int row, int kcol, int N, int K, int pb, int kb
) {
    if constexpr (!FROM_PANEL) {
        return W[(static_cast<int64_t>(e) * N + row) * K + kcol];
    } else {
        int n_panels = N / pb;
        int k_steps  = K / kb;
        int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kcol / kb);
        return W[(p_idx * pb + (row % pb)) * kb + (kcol % kb)];
    }
}

template <bool FROM_PANEL>
static __device__ __forceinline__ uint8_t read_ws_elem(
    const uint8_t* WS, int e, int row, int kg_idx, int N, int KG, int pb, int kbg
) {
    if constexpr (!FROM_PANEL) {
        return WS[(static_cast<int64_t>(e) * N + row) * KG + kg_idx];
    } else {
        int n_panels = N / pb;
        int k_steps  = KG / kbg;
        int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kg_idx / kbg);
        return WS[(p_idx * kbg + (kg_idx % kbg)) * pb + (row % pb)];
    }
}

template <int PROJ, bool FROM_PANEL>
__global__ void mx7_c32_count_flips_kernel(
    const uint8_t* __restrict__ W,
    int32_t* __restrict__ counts,
    int E, int N, int K, int src_pb, int src_kb
) {
    constexpr int WARPS = (PROJ == 0) ? 4 : 2;
    constexpr int SUB_N = (PROJ == 0) ? 1 : 4;
    constexpr int BN    = WARPS * SUB_N * 8;
    constexpr int SLAB_K = (PROJ == 0) ? 128 : 32;

    const int n_panels = N / BN;
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= static_cast<int64_t>(E) * n_panels * WARPS) return;

    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int warp = static_cast<int>(warp_uid % WARPS);
    const int64_t panel_uid = warp_uid / WARPS;
    const int nb = static_cast<int>(panel_uid % n_panels);
    const int e = static_cast<int>(panel_uid / n_panels);
    const int n_slabs = K / SLAB_K;

    int c = 0;
    for (int sl = 0; sl < n_slabs; ++sl) {
        for (int s = 0; s < 8; ++s) {
            const int ks2 = s & 1;
            const int n_base = (PROJ == 0)
                ? (nb * BN + warp * 8)
                : (nb * BN + (warp * SUB_N + (s >> 1)) * 8);
            const int k32_base = (PROJ == 0) ? (sl * 128 + (s >> 1) * 32) : (sl * 32);
            const int row = n_base + g;
            const int k0 = k32_base + 4 * lt + ks2 * 2;
            uint8_t v0 = read_w_elem<FROM_PANEL>(W, e, row, k0,      N, K, src_pb, src_kb);
            uint8_t v1 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 1,  N, K, src_pb, src_kb);
            uint8_t v2 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 16, N, K, src_pb, src_kb);
            uint8_t v3 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 17, N, K, src_pb, src_kb);
            c += ((v0 & 0x40) == 0) + ((v1 & 0x40) == 0) + ((v2 & 0x40) == 0) + ((v3 & 0x40) == 0);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        c += __shfl_down_sync(0xFFFFFFFFu, c, off);
    }
    if (lane == 0) counts[warp_uid] = c;
}

template <int PROJ, bool FROM_PANEL>
__global__ void mx7_c32_pack_kernel(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ WS,
    uint8_t* __restrict__ SLABS,
    uint4* __restrict__ WS3,
    uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    int E, int N, int K, uint32_t base_ws, int src_pb, int src_kb
) {
    constexpr int WARPS  = (PROJ == 0) ? 4 : 2;
    constexpr int SUB_N  = (PROJ == 0) ? 1 : 4;
    constexpr int BN     = WARPS * SUB_N * 8;
    constexpr int SLAB_K = (PROJ == 0) ? 128 : 32;

    const int n_panels = N / BN;
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= static_cast<int64_t>(E) * n_panels * WARPS) return;

    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int warp = static_cast<int>(warp_uid % WARPS);
    const int64_t panel_uid = warp_uid / WARPS;
    const int nb = static_cast<int>(panel_uid % n_panels);
    const int e = static_cast<int>(panel_uid / n_panels);
    const int KG = K >> 5;
    const int n_slabs = K / SLAB_K;
    const int src_kbg = src_kb >> 5;

    int32_t f_pos = F_OFF[warp_uid];

    for (int sl = 0; sl < n_slabs; ++sl) {
        uint16_t sm_arr[8];
        uint8_t  e21_arr[8];
        uint32_t e0_reg = 0u;
        uint32_t ws_words[4] = {0u, 0u, 0u, 0u};

        for (int s = 0; s < 8; ++s) {
            const int ks2 = s & 1;
            const int n_base = (PROJ == 0)
                ? (nb * BN + warp * 8)
                : (nb * BN + (warp * SUB_N + (s >> 1)) * 8);
            const int k32_base = (PROJ == 0) ? (sl * 128 + (s >> 1) * 32) : (sl * 32);

            if (ks2 == 0 && lane == 0) {
                const int kg_idx = k32_base >> 5;
                uint32_t w_word = 0u;
                for (int gg = 0; gg < 8; ++gg) {
                    uint32_t v = static_cast<uint32_t>(
                        read_ws_elem<FROM_PANEL>(WS, e, n_base + gg, kg_idx, N, KG, src_pb, src_kbg)
                    ) - base_ws;
                    w_word |= (v & 7u) << (gg * 3);
                }
                ws_words[s >> 1] = w_word;
            }

            const int row = n_base + g;
            const int k0 = k32_base + 4 * lt + ks2 * 2;
            const uint8_t vals[4] = {
                read_w_elem<FROM_PANEL>(W, e, row, k0,      N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 1,  N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 16, N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 17, N, K, src_pb, src_kb),
            };

            uint32_t sm16 = 0u, e21_8 = 0u, e0_4 = 0u;
            uint32_t flip_mask = 0u;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b = vals[j];
                if ((b & 0x40u) == 0u) flip_mask |= (1u << j);
                uint32_t sm4 = (((b >> 7) & 1u) << 3) | (b & 7u);
                sm16  |= sm4 << (j * 4);
                e21_8 |= ((b >> 4) & 3u) << (j * 2);
                e0_4  |= ((b >> 3) & 1u) << j;
            }
            sm_arr[s] = static_cast<uint16_t>(sm16);
            e21_arr[s] = static_cast<uint8_t>(e21_8);
            e0_reg |= (e0_4 << (s * 4));

            const uint32_t step_id = static_cast<uint32_t>((sl << 3) | s);
            const uint32_t any_flips = __ballot_sync(0xFFFFFFFFu, flip_mask != 0u);
            if (any_flips != 0u) {
                int my_cnt = __popc(flip_mask);
                int prefix = my_cnt;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    int n_val = __shfl_up_sync(0xFFFFFFFFu, prefix, off);
                    if (lane >= off) prefix += n_val;
                }
                int total_step_flips = __shfl_sync(0xFFFFFFFFu, prefix, 31);
                int my_off = f_pos + (prefix - my_cnt);
                if (flip_mask != 0u) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (flip_mask & (1u << j)) {
                            FLIPS[my_off++] = static_cast<uint16_t>((step_id << 7) | (lane << 2) | j);
                        }
                    }
                }
                f_pos += total_step_flips;
            }
        }

        const int64_t slab_idx = warp_uid * n_slabs + sl;
        uint8_t* sp = SLABS + slab_idx * MX7_SLAB_BYTES;
        uint4 sm_v;
        sm_v.x = static_cast<uint32_t>(sm_arr[0]) | (static_cast<uint32_t>(sm_arr[1]) << 16);
        sm_v.y = static_cast<uint32_t>(sm_arr[2]) | (static_cast<uint32_t>(sm_arr[3]) << 16);
        sm_v.z = static_cast<uint32_t>(sm_arr[4]) | (static_cast<uint32_t>(sm_arr[5]) << 16);
        sm_v.w = static_cast<uint32_t>(sm_arr[6]) | (static_cast<uint32_t>(sm_arr[7]) << 16);
        *reinterpret_cast<uint4*>(sp + lane * 16) = sm_v;

        uint2 e21_v;
        e21_v.x = static_cast<uint32_t>(e21_arr[0]) | (static_cast<uint32_t>(e21_arr[1]) << 8)
                | (static_cast<uint32_t>(e21_arr[2]) << 16) | (static_cast<uint32_t>(e21_arr[3]) << 24);
        e21_v.y = static_cast<uint32_t>(e21_arr[4]) | (static_cast<uint32_t>(e21_arr[5]) << 8)
                | (static_cast<uint32_t>(e21_arr[6]) << 16) | (static_cast<uint32_t>(e21_arr[7]) << 24);
        *reinterpret_cast<uint2*>(sp + 512 + lane * 8) = e21_v;
        *reinterpret_cast<uint32_t*>(sp + 768 + lane * 4) = e0_reg;

        if (lane == 0) {
            WS3[slab_idx] = make_uint4(ws_words[0], ws_words[1], ws_words[2], ws_words[3]);
        }
    }
}

// =========================================================================
// Kernel 1 (C32): Native Single-Pass BM=TILE MX_W13 (N=576, K=4608, BN=32, 4 warps)
//   - TILE=8:  BM=8,  BK_SMEM=256, SA_STRIDE=272 ->  9.2 KiB -> 6 CTAs/SM = 288 CTAs
//   - TILE=16: BM=16, BK_SMEM=256, SA_STRIDE=272 -> 18.0 KiB -> 5 CTAs/SM = 240 CTAs
//   - TILE=32: BM=32, BK_SMEM=128, SA_STRIDE=144 -> 19.1 KiB -> 5 CTAs/SM = 240 CTAs (20 warps/SM!)
//   - Single MX7 slab load + flip decode + unpack reused across im=0,1 at TILE=32!
//   - Dynamic has_m8/has_m16/has_m24 skips ALL dead rows (`im=1` skipped when live_m <= 16)!
//   - Row2Tok Zero-Scatter + optional FUSE_SITU (compact SiTU_Out, ZERO atomics/barriers!)
// =========================================================================
template <int TILE, int MIN_CTA, int PDL_WAIT_MODE, bool TRIG_DEP, bool ZERO_SCATTER, bool AS_ROW_MAJOR, bool FUSE_SITU>
__global__ __launch_bounds__(128, MIN_CTA)
void mx7_w13_c32_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    __nv_bfloat16* __restrict__ Out,
    int N, int K, int MAX_CTAS, uint32_t base_ws
) {
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int THREADS = 128;
    constexpr int BN = 32;
    constexpr int SLAB_K = 128;
    constexpr int BK_SMEM = (TILE <= 16) ? 256 : 128;
    constexpr int SLABS_PER_BLK = BK_SMEM / SLAB_K;  // 2 for TILE<=16, 1 for TILE=32
    constexpr int M_ROWS = TILE;
    constexpr int M_STEPS = (TILE <= 16) ? 1 : 2;
    // SA_STRIDE = 272 (136 banks === 8 mod 32) or 144 (72 banks === 8 mod 32) -> 0-WAY LDS.64!
    constexpr int SA_STRIDE = BK_SMEM + 16;

    const int n_panels = N / BN;     // 18
    const int n_active = Tiles[2 * MAX_CTAS];
    const int n_total = n_active * n_panels;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx = static_cast<int>(blockIdx.x);

    if (bidx >= n_total) {
        if constexpr (PDL_WAIT_MODE == 2) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if constexpr (TRIG_DEP) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_slabs = K / SLAB_K;   // 36
    const int n_blks = K / BK_SMEM;   // 18 (TILE<=16) or 36 (TILE=32)
    const int KG = K >> 5;            // 144

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;
    const int x1_0 = (g + 16) * SA_STRIDE + 4 * lt;
    const int x1_1 = (g + 24) * SA_STRIDE + 4 * lt;

    alignas(16) __shared__ uint32_t sFlip[WARPS][32];
    alignas(16) __shared__ int32_t  sTok[M_ROWS];
    alignas(16) __shared__ __nv_bfloat16 sA[2][M_ROWS * SA_STRIDE];

    bool trig_fired = false;

    for (int idx = bidx; idx < n_total; idx += stride) {
        const int t  = idx / n_panels;
        const int nb = idx - t * n_panels;
        const bool is_first_tile = (idx == bidx);
        const bool is_last_tile  = (idx + stride >= n_total);

        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int64_t warp_uid = (e * n_panels + nb) * WARPS + warp;
        const uint8_t* slab_base = SLABS + warp_uid * n_slabs * MX7_SLAB_BYTES;
        const uint4* ws_base = WS3 + warp_uid * n_slabs;
        int32_t f_cur = F_OFF[warp_uid];
        const int32_t f_end = F_OFF[warp_uid + 1];

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_first_tile) {
                if (lane < 7) prefetch_l2(slab_base + lane * 128);
                if (lane == 7) prefetch_l2(ws_base);
                if (lane == 8 && f_cur < f_end) prefetch_l2(FLIPS + f_cur);
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }
        if (live_m == 0) continue;

        if constexpr (ZERO_SCATTER) {
            if (tid < live_m) sTok[tid] = Row2Tok[row_base + tid];
        }

        // Zero only padding rows inside active 8-row groups (prevents uninit NaNs/subnormals)
        const int pad_limit = (live_m <= 8) ? 8
            : ((live_m <= 16) ? 16 : ((live_m <= 24) ? 24 : M_ROWS));
        if (live_m < pad_limit) {
            constexpr int Q_PER_ROW = BK_SMEM >> 2;
            const int zero_q = (pad_limit - live_m) * Q_PER_ROW;
            for (int zidx = tid; zidx < zero_q; zidx += THREADS) {
                const int dlm = zidx / Q_PER_ROW;
                const int qk  = zidx - dlm * Q_PER_ROW;
                const int off = (live_m + dlm) * SA_STRIDE + (qk << 2);
                *reinterpret_cast<uint2*>(sA[0] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[1] + off) = make_uint2(0u, 0u);
            }
        }

        auto load_a_blk = [&](int blk, __nv_bfloat16* dst, bool use_smem_tok) {
            constexpr int Q_PER_ROW = BK_SMEM >> 2;  // 64 (TILE<=16) or 32 (TILE=32)
            constexpr int Q_SHIFT   = (BK_SMEM == 256) ? 6 : 5;
            constexpr int Q_MASK    = Q_PER_ROW - 1;
            const int total_q = live_m * Q_PER_ROW;
            const int k0 = blk * BK_SMEM;
            for (int aidx = tid; aidx < total_q; aidx += THREADS) {
                const int lm = aidx >> Q_SHIFT;
                const int qk = aidx & Q_MASK;
                const int64_t row = static_cast<int64_t>(row_base + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                    : row;
                const int kcol = k0 + (qk << 2);
                const int kg_col = kcol >> 5;
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + a_row * K + kcol);
                uint8_t as_b;
                if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                    as_b = AS[a_row * KG + kg_col];
                } else {
                    as_b = AS[scale_offset_decode(row, kg_col, KG)];
                }
                const float a_sc = e8m0_to_f32(as_b);
                uint32_t b0, b1;
                e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
                *reinterpret_cast<uint2*>(dst + lm * SA_STRIDE + (qk << 2)) = make_uint2(b0, b1);
            }
        };

        uint32_t rec0 = (f_cur + lane < f_end) ? static_cast<uint32_t>(__ldg(FLIPS + f_cur + lane)) : 0xFFFFu;
        if (lane == 0 && f_cur + 64 < f_end) prefetch_l2(FLIPS + f_cur + 64);
        SlabMX7 cur_slab = load_slab_mx7(slab_base, ws_base, lane);
        load_a_blk(0, sA[0], false);
        uint32_t cur_fmask = finish_slab_flip_mask(rec0, FLIPS, f_cur, f_end, 0u, lane, sFlip[warp]);
        __syncthreads();

        float acc[M_STEPS][4];
        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            acc[im][0] = 0.0f; acc[im][1] = 0.0f; acc[im][2] = 0.0f; acc[im][3] = 0.0f;
        }
        const bool has_m8  = (TILE > 8)  && (live_m > 8);
        const bool has_m16 = (TILE > 16) && (live_m > 16);
        const bool has_m24 = (TILE > 16) && (live_m > 24);

        for (int blk = 0; blk < n_blks; ++blk) {
            const __nv_bfloat16* sAp = sA[blk & 1];
            __nv_bfloat16* sAnext    = sA[(blk & 1) ^ 1];
            if (blk + 1 < n_blks) load_a_blk(blk + 1, sAnext, true);

            #pragma unroll 1
            for (int is = 0; is < SLABS_PER_BLK; ++is) {
                const int sl = blk * SLABS_PER_BLK + is;
                SlabMX7 nxt_slab;
                uint32_t nxt_rec = 0xFFFFu;
                if (sl + 1 < n_slabs) {
                    nxt_slab = load_slab_mx7(slab_base + static_cast<int64_t>(sl + 1) * MX7_SLAB_BYTES, ws_base + sl + 1, lane);
                    nxt_rec = (f_cur + lane < f_end) ? static_cast<uint32_t>(__ldg(FLIPS + f_cur + lane)) : 0xFFFFu;
                    if (lane == 0 && ((sl & 3) == 0) && (f_cur + 64 < f_end)) prefetch_l2(FLIPS + f_cur + 64);
                } else if (!is_last_tile) {
                    const int nidx = idx + stride;
                    const int nt = nidx / n_panels;
                    const int nnb = nidx - nt * n_panels;
                    const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                    const int64_t nwuid = (ne * n_panels + nnb) * WARPS + warp;
                    const uint8_t* nsp = SLABS + nwuid * n_slabs * MX7_SLAB_BYTES;
                    if (lane < 7) prefetch_l2(nsp + lane * 128);
                    if (lane == 7) prefetch_l2(WS3 + nwuid * n_slabs);
                }

                const uint32_t sm_arr[8] = {
                    cur_slab.sm.x & 0xFFFFu, cur_slab.sm.x >> 16,
                    cur_slab.sm.y & 0xFFFFu, cur_slab.sm.y >> 16,
                    cur_slab.sm.z & 0xFFFFu, cur_slab.sm.z >> 16,
                    cur_slab.sm.w & 0xFFFFu, cur_slab.sm.w >> 16
                };
                const uint32_t e21_arr[8] = {
                    cur_slab.e21.x & 0xFFu, (cur_slab.e21.x >> 8) & 0xFFu,
                    (cur_slab.e21.x >> 16) & 0xFFu, (cur_slab.e21.x >> 24) & 0xFFu,
                    cur_slab.e21.y & 0xFFu, (cur_slab.e21.y >> 8) & 0xFFu,
                    (cur_slab.e21.y >> 16) & 0xFFu, (cur_slab.e21.y >> 24) & 0xFFu
                };
                const uint32_t ws_arr[4] = {cur_slab.ws3.x, cur_slab.ws3.y, cur_slab.ws3.z, cur_slab.ws3.w};
                const int blk_k_off = is * SLAB_K;

                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int kq_off = blk_k_off + q * 32;
                    const uint32_t ws_byte = base_ws + ((ws_arr[q] >> (g * 3)) & 7u);
                    const float w_sc = __uint_as_float(ws_byte << 23);

                    const int s0 = q << 1;
                    const int s1 = s0 | 1;
                    uint32_t b0_0, b1_0, b0_1, b1_1;
                    e4m3x4_scale_to_bf16x4(
                        build_e4_pack_flip(sm_arr[s0], e21_arr[s0], (cur_slab.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu),
                        w_sc, b0_0, b1_0);
                    e4m3x4_scale_to_bf16x4(
                        build_e4_pack_flip(sm_arr[s1], e21_arr[s1], (cur_slab.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu),
                        w_sc, b0_1, b1_1);

                    // Paired 0-way bank-conflict-free LDS.64 loads (reuses unpacked b0/b1 across im=0,1!)
                    {
                        const uint2 a0 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off);
                        const uint2 a2 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off + 16);
                        uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                        if constexpr (TILE > 8) {
                            if (has_m8) {
                                a1 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off);
                                a3 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off + 16);
                            }
                        }
                        mma_m16n8k16_bf16(acc[0], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                        mma_m16n8k16_bf16(acc[0], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                    }
                    if constexpr (M_STEPS == 2) {
                        if (has_m16) {
                            const uint2 c0 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off);
                            const uint2 c2 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off + 16);
                            uint2 c1 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
                            if (has_m24) {
                                c1 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off);
                                c3 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off + 16);
                            }
                            mma_m16n8k16_bf16(acc[1], c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                            mma_m16n8k16_bf16(acc[1], c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                        }
                    }
                }

                if constexpr (TRIG_DEP) {
                    if (is_last_tile && sl == n_slabs - 2) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }

                if (sl + 1 < n_slabs) {
                    cur_slab = nxt_slab;
                    cur_fmask = finish_slab_flip_mask(nxt_rec, FLIPS, f_cur, f_end, static_cast<uint32_t>(sl + 1), lane, sFlip[warp]);
                }
            }
            if (blk + 1 < n_blks) __syncthreads();
        }

        const float scale = Ones[e];
        const int compact_base = FUSE_SITU ? TileCompactStart[t] : 0;

        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            if (im == 1 && !has_m16) break;
            const int lm0 = im * 16 + g;
            const int lm1 = lm0 + 8;
            const int r0 = row_base + lm0;
            const int r1 = row_base + lm1;
            const float o0 = mul_rn(acc[im][0], scale);
            const float o1 = mul_rn(acc[im][1], scale);
            const float o2 = mul_rn(acc[im][2], scale);
            const float o3 = mul_rn(acc[im][3], scale);

            if constexpr (FUSE_SITU) {
                // MXFP8 has NO RowAmaxW2 -> ZERO atomics, ZERO extra barriers!
                const int inter = N >> 1;  // 288
                const int c_warp = nb * 16 + warp * 4;
                const int c0 = compact_base + lm0;
                const int c1 = compact_base + lm1;

                uint16_t y0_b = 0u;
                if (lm0 < live_m) {
                    y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
                }
                uint32_t p0 = static_cast<uint32_t>(y0_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
                uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
                if (lt == 0 && lm0 < live_m) {
                    *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * inter + c_warp) = make_uint2(p0, p1);
                }
                if constexpr (TILE > 8) {
                    uint16_t y1_b = 0u;
                    if (lm1 < live_m) {
                        y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                    }
                    uint32_t q0 = static_cast<uint32_t>(y1_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                    uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                    if (lt == 0 && lm1 < live_m) {
                        *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * inter + c_warp) = make_uint2(q0, q1);
                    }
                }
            } else {
                const int n0 = nb * BN + warp * 8 + 2 * lt;
                if (lm0 < live_m) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
                }
                if constexpr (TILE > 8) {
                    if (lm1 < live_m) {
                        *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
                    }
                }
            }
        }

        if (!is_last_tile) __syncthreads();
    }

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 2 (C32): Native Single-Pass BM=TILE, BN=128 (4-Warp) MX_W2 (N=4608, K=288)
//   - 1-to-1 ZERO-REPACK BIJECTION onto packed BN=64/2-warp MX7 slabs:
//       packed_warp_uid = (e * 72 + (nb*2 + (warp>>1))) * 2 + (warp&1)
//                       = (e * 36 + nb) * 4 + warp
//   - TILE=8,16: BM=8/16, SA_STRIDE=304 -> 10.3 KiB -> 6 CTAs/SM = 288 CTAs (6x48 panel-wave)
//   - TILE=32:   BM=32,   SA_STRIDE=304 -> 20.0 KiB -> 4 CTAs/SM = 192 CTAs (6x32 panel-wave)
//   - Optional FUSE_QUANT (register-compact OTF MXFP8 quant-dequant from compact SiTU_Out)
//   - Optional FUSE_COMBINE (exact same 6-panel-wave double-buffered compact DownPanel as NV_W2!)
// =========================================================================
template <int TILE, int MIN_CTA, int CTAS_PER_PANEL, int PDL_WAIT_MODE, bool TRIG_DEP, bool AS_ROW_MAJOR, bool FUSE_QUANT, bool FUSE_COMBINE>
__global__ __launch_bounds__(128, MIN_CTA)
void mx7_w2_c32_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ Src2Compact,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ PanelDone,
    uint32_t* __restrict__ BufGen,
    __nv_bfloat16* __restrict__ SendBuf,
    int N, int K, int MAX_CTAS, int T, uint32_t base_ws, int as_row_stride, int down_row_stride
) {
    (void)K;
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int THREADS = 128;
    constexpr int SUB_N = 4;
    constexpr int BN = 128;              // 4 warps * 4 sub * 8 = 128
    constexpr int N_PANELS = 36;         // 4608 / 128
    constexpr int P_GROUP = 6;
    constexpr int N_WAVES = N_PANELS / P_GROUP;  // 6 waves
    constexpr int M_ROWS = TILE;
    constexpr int M_STEPS = (TILE <= 16) ? 1 : 2;
    constexpr int KG = 9;                // 288 / 32
    constexpr int SA_STRIDE = 304;       // 288 + 16 = 304 bf16 = 608 B === -8 mod 32 -> 0-way LDS.64!

    const int n_active = Tiles[2 * MAX_CTAS];
    const int bidx = static_cast<int>(blockIdx.x);
    const int stride = static_cast<int>(gridDim.x);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;
    const int x1_0 = (g + 16) * SA_STRIDE + 4 * lt;
    const int x1_1 = (g + 24) * SA_STRIDE + 4 * lt;

    if constexpr (FUSE_COMBINE) {
        const int p_slot = bidx / CTAS_PER_PANEL;
        const int p_rank = bidx - p_slot * CTAS_PER_PANEL;
        if (n_active == 0) {
            if constexpr (PDL_WAIT_MODE == 2) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if (p_rank == 0) {
                int n_live_tok = T;
                if (NLivePtr != nullptr) {
                    int nl = *NLivePtr;
                    if (nl < 0) nl = 0;
                    if (nl < n_live_tok) n_live_tok = nl;
                }
                for (int w = 0; w < N_WAVES; ++w) {
                    const int nb = w * P_GROUP + p_slot;
                    const int col0 = nb * BN + lane * 4;
                    for (int m = warp; m < n_live_tok; m += WARPS) {
                        *reinterpret_cast<uint2*>(SendBuf + static_cast<int64_t>(m) * N + col0) = make_uint2(0u, 0u);
                    }
                }
            }
            if constexpr (TRIG_DEP) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
        const int active_in_panel = (n_active < CTAS_PER_PANEL) ? n_active : CTAS_PER_PANEL;
        if (p_rank >= active_in_panel) {
            if constexpr (PDL_WAIT_MODE == 2) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if constexpr (TRIG_DEP) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    } else {
        const int n_total = n_active * N_PANELS;
        if (bidx >= n_total) {
            if constexpr (PDL_WAIT_MODE == 2) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if constexpr (TRIG_DEP) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    }

    alignas(16) __shared__ uint32_t sFlip[WARPS][32];
    alignas(16) __shared__ __nv_bfloat16 sA_all[M_ROWS * SA_STRIDE];
    __shared__ int32_t  sIsLastPanel;

    bool waited_pdl = false;
    bool trig_fired = false;

    auto run_tile = [&](int t, int nb, bool is_very_first, bool is_very_last,
                        int next_t, int next_nb, bool has_next_tile,
                        __nv_bfloat16* down_panel_buf) {
        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        // Exact 1-to-1 bijection onto packed BN=64/2-warp MX7 slabs!
        const int64_t warp_uid = (e * N_PANELS + nb) * WARPS + warp;
        const uint8_t* slab_base = SLABS + warp_uid * KG * MX7_SLAB_BYTES;
        const uint4* ws_base = WS3 + warp_uid * KG;
        int32_t f_cur = F_OFF[warp_uid];
        const int32_t f_end = F_OFF[warp_uid + 1];

        // Prefetch slab 0 & 1 + FLIPS into L2 before PDL wait / OTF quant (uses 0 extra live regs!)
        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_very_first) {
                if (lane < 7) prefetch_l2(slab_base + lane * 128);
                if (lane == 7) prefetch_l2(ws_base);
                if (lane == 8 && f_cur < f_end) prefetch_l2(FLIPS + f_cur);
                if (lane >= 9 && lane < 16) prefetch_l2(slab_base + MX7_SLAB_BYTES + (lane - 9) * 128);
            }
        }

        // Zero padding rows inside active 8-row groups
        const int pad_limit = (live_m <= 8) ? 8
            : ((live_m <= 16) ? 16 : ((live_m <= 24) ? 24 : M_ROWS));
        if (live_m < pad_limit) {
            constexpr int Q_PER_ROW = 288 >> 2;  // 72
            const int zero_q = (pad_limit - live_m) * Q_PER_ROW;
            for (int zidx = tid; zidx < zero_q; zidx += THREADS) {
                const int dlm = zidx / Q_PER_ROW;
                const int qk  = zidx - dlm * Q_PER_ROW;
                *reinterpret_cast<uint2*>(sA_all + (live_m + dlm) * SA_STRIDE + (qk << 2)) = make_uint2(0u, 0u);
            }
        }

        if constexpr (PDL_WAIT_MODE == 2) {
            if (!waited_pdl) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                waited_pdl = true;
            }
        }
        if (live_m == 0) return;

        // Overlap slab-0 L2 prefetch with OTF quant-dequant (0 extra regs!)
        if constexpr (FUSE_QUANT) {
            if (lane < 7) prefetch_l2(slab_base + lane * 128);
            if (lane == 7) prefetch_l2(ws_base);
            if (lane == 8 && f_cur < f_end) prefetch_l2(FLIPS + f_cur);
        }

        const int compact_tile_base = (FUSE_QUANT || FUSE_COMBINE) ? TileCompactStart[t] : 0;

        // Scoped activation load / OTF MXFP8 quant-dequant BEFORE cur_slab & acc (0 spills!)
        {
            if constexpr (FUSE_QUANT) {
                const __nv_bfloat16* situ_in = reinterpret_cast<const __nv_bfloat16*>(A);
                const int total_blks = live_m * KG;  // live_m * 9
                for (int bq = tid; bq < total_blks; bq += THREADS) {
                    const int lm = bq / KG;
                    const int blk = bq - lm * KG;
                    const int crow = compact_tile_base + lm;
                    const uint4* bp = reinterpret_cast<const uint4*>(situ_in + static_cast<int64_t>(crow) * 288 + blk * 32);
                    uint4 v[4] = {ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3)};
                    quant_dequant_mxfp8_32_to_sA(v, sA_all + lm * SA_STRIDE + blk * 32);
                }
            } else {
                constexpr int Q_PER_ROW = 72;  // 288 / 4
                const int total_q = live_m * Q_PER_ROW;
                for (int aidx = tid; aidx < total_q; aidx += THREADS) {
                    const int lm = aidx / Q_PER_ROW;
                    const int qk = aidx - lm * Q_PER_ROW;
                    const int64_t row = static_cast<int64_t>(row_base + lm);
                    const int kcol = qk << 2;
                    const int kg_col = kcol >> 5;
                    const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + row * 288 + kcol);
                    uint8_t as_b;
                    if constexpr (AS_ROW_MAJOR) {
                        as_b = AS[row * as_row_stride + kg_col];
                    } else {
                        as_b = AS[scale_offset_decode(row, kg_col, KG)];
                    }
                    const float a_sc = e8m0_to_f32(as_b);
                    uint32_t b0, b1;
                    e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
                    *reinterpret_cast<uint2*>(sA_all + lm * SA_STRIDE + kcol) = make_uint2(b0, b1);
                }
            }
        }

        uint32_t rec0 = (f_cur + lane < f_end) ? static_cast<uint32_t>(__ldg(FLIPS + f_cur + lane)) : 0xFFFFu;
        if (lane == 0 && f_cur + 64 < f_end) prefetch_l2(FLIPS + f_cur + 64);
        SlabMX7 cur_slab = load_slab_mx7(slab_base, ws_base, lane);
        uint32_t cur_fmask = finish_slab_flip_mask(rec0, FLIPS, f_cur, f_end, 0u, lane, sFlip[warp]);
        __syncthreads();

        float acc[M_STEPS][SUB_N][4];
        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                acc[im][sub][0] = 0.0f; acc[im][sub][1] = 0.0f;
                acc[im][sub][2] = 0.0f; acc[im][sub][3] = 0.0f;
            }
        }
        const bool has_m8  = (TILE > 8)  && (live_m > 8);
        const bool has_m16 = (TILE > 16) && (live_m > 16);
        const bool has_m24 = (TILE > 16) && (live_m > 24);

        #pragma unroll 1
        for (int kg = 0; kg < KG; ++kg) {
            SlabMX7 nxt_slab;
            uint32_t nxt_rec = 0xFFFFu;
            if (kg + 1 < KG) {
                nxt_slab = load_slab_mx7(slab_base + static_cast<int64_t>(kg + 1) * MX7_SLAB_BYTES, ws_base + kg + 1, lane);
                nxt_rec = (f_cur + lane < f_end) ? static_cast<uint32_t>(__ldg(FLIPS + f_cur + lane)) : 0xFFFFu;
            } else if (has_next_tile) {
                const int64_t ne = static_cast<int64_t>(Tiles[next_t]);
                const int64_t nwuid = (ne * N_PANELS + next_nb) * WARPS + warp;
                const uint8_t* nsp = SLABS + nwuid * KG * MX7_SLAB_BYTES;
                if (lane < 7) prefetch_l2(nsp + lane * 128);
                if (lane == 7) prefetch_l2(WS3 + nwuid * KG);
            }

            // Paired 0-way bank-conflict-free LDS.64 loads (reused across all 4 SUB_N!)
            const int k0 = kg * 32;
            const uint2 a0 = *reinterpret_cast<const uint2*>(sA_all + x0_0 + k0);
            const uint2 a2 = *reinterpret_cast<const uint2*>(sA_all + x0_0 + k0 + 16);
            uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
            if constexpr (TILE > 8) {
                if (has_m8) {
                    a1 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k0);
                    a3 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k0 + 16);
                }
            }
            uint2 c0 = make_uint2(0u, 0u), c1 = make_uint2(0u, 0u);
            uint2 c2 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
            if constexpr (M_STEPS == 2) {
                if (has_m16) {
                    c0 = *reinterpret_cast<const uint2*>(sA_all + x1_0 + k0);
                    c2 = *reinterpret_cast<const uint2*>(sA_all + x1_0 + k0 + 16);
                    if (has_m24) {
                        c1 = *reinterpret_cast<const uint2*>(sA_all + x1_1 + k0);
                        c3 = *reinterpret_cast<const uint2*>(sA_all + x1_1 + k0 + 16);
                    }
                }
            }

            const uint32_t sm_arr[8] = {
                cur_slab.sm.x & 0xFFFFu, cur_slab.sm.x >> 16,
                cur_slab.sm.y & 0xFFFFu, cur_slab.sm.y >> 16,
                cur_slab.sm.z & 0xFFFFu, cur_slab.sm.z >> 16,
                cur_slab.sm.w & 0xFFFFu, cur_slab.sm.w >> 16
            };
            const uint32_t e21_arr[8] = {
                cur_slab.e21.x & 0xFFu, (cur_slab.e21.x >> 8) & 0xFFu,
                (cur_slab.e21.x >> 16) & 0xFFu, (cur_slab.e21.x >> 24) & 0xFFu,
                cur_slab.e21.y & 0xFFu, (cur_slab.e21.y >> 8) & 0xFFu,
                (cur_slab.e21.y >> 16) & 0xFFu, (cur_slab.e21.y >> 24) & 0xFFu
            };
            const uint32_t ws_arr[4] = {cur_slab.ws3.x, cur_slab.ws3.y, cur_slab.ws3.z, cur_slab.ws3.w};

            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                const uint32_t ws_byte = base_ws + ((ws_arr[sub] >> (g * 3)) & 7u);
                const float w_sc = __uint_as_float(ws_byte << 23);
                const int s0 = sub << 1;
                const int s1 = s0 | 1;
                uint32_t b0_0, b1_0, b0_1, b1_1;
                e4m3x4_scale_to_bf16x4(
                    build_e4_pack_flip(sm_arr[s0], e21_arr[s0], (cur_slab.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu),
                    w_sc, b0_0, b1_0);
                e4m3x4_scale_to_bf16x4(
                    build_e4_pack_flip(sm_arr[s1], e21_arr[s1], (cur_slab.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu),
                    w_sc, b0_1, b1_1);

                mma_m16n8k16_bf16(acc[0][sub], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                mma_m16n8k16_bf16(acc[0][sub], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);

                if constexpr (M_STEPS == 2) {
                    if (has_m16) {
                        mma_m16n8k16_bf16(acc[1][sub], c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                        mma_m16n8k16_bf16(acc[1][sub], c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                    }
                }
            }

            if constexpr (TRIG_DEP) {
                if (is_very_last && kg == KG - 2) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (kg + 1 < KG) {
                cur_slab = nxt_slab;
                cur_fmask = finish_slab_flip_mask(nxt_rec, FLIPS, f_cur, f_end, static_cast<uint32_t>(kg + 1), lane, sFlip[warp]);
            }
        }

        const float scale = Ones[e];
        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            if (im == 1 && !has_m16) break;
            const int lm0 = im * 16 + g;
            const int lm1 = lm0 + 8;
            const int r0 = row_base + lm0;
            const int r1 = row_base + lm1;
            const int c0 = compact_tile_base + lm0;
            const int c1 = compact_tile_base + lm1;

            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                const int col_local = (warp * SUB_N + sub) * 8 + 2 * lt;
                if constexpr (FUSE_COMBINE) {
                    if (lm0 < live_m) {
                        st_cg_u32(
                            down_panel_buf + static_cast<int64_t>(c0) * BN + col_local,
                            pack_bf16x2(mul_rn(acc[im][sub][0], scale), mul_rn(acc[im][sub][1], scale))
                        );
                    }
                    if constexpr (TILE > 8) {
                        if (lm1 < live_m) {
                            st_cg_u32(
                                down_panel_buf + static_cast<int64_t>(c1) * BN + col_local,
                                pack_bf16x2(mul_rn(acc[im][sub][2], scale), mul_rn(acc[im][sub][3], scale))
                            );
                        }
                    }
                } else {
                    const int n0 = nb * BN + col_local;
                    if (lm0 < live_m) {
                        *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) =
                            pack_bf16x2(mul_rn(acc[im][sub][0], scale), mul_rn(acc[im][sub][1], scale));
                    }
                    if constexpr (TILE > 8) {
                        if (lm1 < live_m) {
                            *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) =
                                pack_bf16x2(mul_rn(acc[im][sub][2], scale), mul_rn(acc[im][sub][3], scale));
                        }
                    }
                }
            }
        }
    };

    if constexpr (FUSE_COMBINE) {
        const int p_slot = bidx / CTAS_PER_PANEL;
        const int p_rank = bidx - p_slot * CTAS_PER_PANEL;
        const int active_in_panel = (n_active < CTAS_PER_PANEL) ? n_active : CTAS_PER_PANEL;

        for (int w = 0; w < N_WAVES; ++w) {
            const int nb = w * P_GROUP + p_slot;
            const int buf = w & 1;
            __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>((buf * P_GROUP + p_slot) * down_row_stride) * BN;

            if (w >= 2) {
                if (tid == 0) {
                    const uint32_t need = static_cast<uint32_t>(w >> 1);
                    while (ld_acq_gpu_u32(&BufGen[p_slot * 2 + buf]) < need) {}
                }
                __syncthreads();
            }

            for (int t = p_rank; t < n_active; t += CTAS_PER_PANEL) {
                const bool is_very_first = (w == 0 && t == p_rank);
                const bool is_last_in_wave = (t + CTAS_PER_PANEL >= n_active);
                const bool is_very_last = (w == N_WAVES - 1 && is_last_in_wave);
                const int next_t = is_last_in_wave ? p_rank : (t + CTAS_PER_PANEL);
                const int next_nb = is_last_in_wave ? ((w + 1) * P_GROUP + p_slot) : nb;
                run_tile(t, nb, is_very_first, is_very_last, next_t, next_nb, !is_very_last, down_panel_buf);
                if (!is_last_in_wave) __syncthreads();
            }

            // Flush all threads' DownPanel stores to L2 BEFORE PanelDone release!
            __threadfence();
            __syncthreads();
            if (active_in_panel == 1) {
                if (tid == 0) sIsLastPanel = 1;
            } else {
                if (tid == 0) {
                    uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], 1u);
                    if (prev == static_cast<uint32_t>(active_in_panel - 1)) {
                        st_rel_gpu_u32(&PanelDone[nb], 0u);
                        sIsLastPanel = 1;
                    } else {
                        sIsLastPanel = 0;
                    }
                }
            }
            __syncthreads();

            if (sIsLastPanel) {
                int n_live_tok = T;
                if (NLivePtr != nullptr) {
                    int nl = *NLivePtr;
                    if (nl < 0) nl = 0;
                    if (nl < n_live_tok) n_live_tok = nl;
                }
                const int col_local = lane * 4;
                const int col_glob  = nb * BN + col_local;
                for (int m = warp; m < n_live_tok; m += WARPS) {
                    const int32_t* s2c = Src2Compact + m * 8;
                    int64_t r[8];
                    #pragma unroll
                    for (int s = 0; s < 8; ++s) r[s] = static_cast<int64_t>(__ldg(s2c + s));

                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    #pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        const uint2 v = (r[s] >= 0)
                            ? ld_cg_u2(down_panel_buf + r[s] * BN + col_local)
                            : make_uint2(0u, 0u);
                        a0 = add_rn(a0, bf16_bits_to_f32(static_cast<uint16_t>(v.x & 0xFFFFu)));
                        a1 = add_rn(a1, bf16_bits_to_f32(static_cast<uint16_t>(v.x >> 16)));
                        a2 = add_rn(a2, bf16_bits_to_f32(static_cast<uint16_t>(v.y & 0xFFFFu)));
                        a3 = add_rn(a3, bf16_bits_to_f32(static_cast<uint16_t>(v.y >> 16)));
                    }
                    *reinterpret_cast<uint2*>(SendBuf + static_cast<int64_t>(m) * N + col_glob) =
                        make_uint2(pack_bf16x2(a0, a1), pack_bf16x2(a2, a3));
                }
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    const uint32_t next_gen = (w < N_WAVES - 2) ? static_cast<uint32_t>((w >> 1) + 1) : 0u;
                    st_rel_gpu_u32(&BufGen[p_slot * 2 + buf], next_gen);
                }
            }
            if (w + 1 < N_WAVES) __syncthreads();
        }
    } else {
        const int n_total = n_active * N_PANELS;
        for (int idx = bidx; idx < n_total; idx += stride) {
            const int t  = idx / N_PANELS;
            const int nb = idx - t * N_PANELS;
            const bool is_very_first = (idx == bidx);
            const bool is_very_last  = (idx + stride >= n_total);
            const int nidx = idx + stride;
            const int nt  = is_very_last ? 0 : (nidx / N_PANELS);
            const int nnb = is_very_last ? 0 : (nidx - nt * N_PANELS);
            run_tile(t, nb, is_very_first, is_very_last, nt, nnb, !is_very_last, nullptr);
            if (!is_very_last) __syncthreads();
        }
    }

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_mx7_c32(
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

template <int TILE, int MIN_CTA, bool ZS, bool RM, bool FS>
static int dispatch_mx7_w13(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const int32_t* Row2Tok,
    const uint8_t* SLABS, const uint4* WS3,
    const uint16_t* FLIPS, const int32_t* F_OFF,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileCompactStart, const float* Gammas, __nv_bfloat16* Out,
    int N, int K, int MAX_CTAS, uint32_t base_ws
) {
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define M13(PW, TR) err = launch_mx7_c32( \
        mx7_w13_c32_kernel<TILE, MIN_CTA, PW, TR, ZS, RM, FS>, \
        grid, block, stream, use_pdl, \
        A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, \
        TileCompactStart, Gammas, Out, N, K, MAX_CTAS, base_ws)
    if      (pdl_wait_mode == 1 && trig_dep)  { M13(1, true);  }
    else if (pdl_wait_mode == 2 && trig_dep)  { M13(2, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { M13(0, true);  }
    else if (pdl_wait_mode == 1 && !trig_dep) { M13(1, false); }
    else if (pdl_wait_mode == 2 && !trig_dep) { M13(2, false); }
    else                                      { M13(0, false); }
    #undef M13
    return static_cast<int>(err);
}

template <int TILE, int MIN_CTA, int CPP, bool RM, bool FQ, bool FC>
static int dispatch_mx7_w2(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS,
    const uint8_t* SLABS, const uint4* WS3,
    const uint16_t* FLIPS, const int32_t* F_OFF,
    const float* Ones, const int32_t* Tiles, const int32_t* TileCompactStart,
    __nv_bfloat16* Out,
    const int32_t* Src2Compact, const int32_t* NLivePtr,
    uint32_t* PanelDone, uint32_t* BufGen, __nv_bfloat16* SendBuf,
    int N, int K, int MAX_CTAS, int T, uint32_t base_ws, int as_row_stride, int down_row_stride
) {
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define M2(PW, TR) err = launch_mx7_c32( \
        mx7_w2_c32_kernel<TILE, MIN_CTA, CPP, PW, TR, RM, FQ, FC>, \
        grid, block, stream, use_pdl, \
        A, AS, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, TileCompactStart, \
        Out, Src2Compact, NLivePtr, PanelDone, BufGen, SendBuf, \
        N, K, MAX_CTAS, T, base_ws, as_row_stride, down_row_stride)
    if      (pdl_wait_mode == 2 && trig_dep)  { M2(2, true);  }
    else if (pdl_wait_mode == 1 && trig_dep)  { M2(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { M2(0, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { M2(2, false); }
    else if (pdl_wait_mode == 1 && !trig_dep) { M2(1, false); }
    else                                      { M2(0, false); }
    #undef M2
    return static_cast<int>(err);
}

}  // namespace

// =========================================================================
// Extern "C" Packer & C32 Entry Points
// =========================================================================

extern "C" int mx7_v2_count_flips(
    const void* W, void* counts, int proj, int E, int N, int K,
    int from_panel, int src_pb, int src_kb, uintptr_t stream_ptr
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int bn = (proj == 0) ? 32 : 64;
    const int warps = (proj == 0) ? 4 : 2;
    const int total_warps = E * (N / bn) * warps;
    if (proj == 0) {
        if (from_panel) mx7_c32_count_flips_kernel<0, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, src_pb, src_kb);
        else mx7_c32_count_flips_kernel<0, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, 0, 0);
    } else if (proj == 1) {
        if (from_panel) mx7_c32_count_flips_kernel<1, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, src_pb, src_kb);
        else mx7_c32_count_flips_kernel<1, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, 0, 0);
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx7_v2_pack(
    const void* W, const void* WS, void* SLABS, void* WS3, void* FLIPS, const void* F_OFF,
    int proj, int E, int N, int K, uint32_t base_ws,
    int from_panel, int src_pb, int src_kb, uintptr_t stream_ptr
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int bn = (proj == 0) ? 32 : 64;
    const int warps = (proj == 0) ? 4 : 2;
    const int total_warps = E * (N / bn) * warps;
    if (proj == 0) {
        if (from_panel) mx7_c32_pack_kernel<0, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, src_pb, src_kb);
        else mx7_c32_pack_kernel<0, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, 0, 0);
    } else if (proj == 1) {
        if (from_panel) mx7_c32_pack_kernel<1, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, src_pb, src_kb);
        else mx7_c32_pack_kernel<1, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, 0, 0);
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx7_w13_c32_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    if ((N % 32) != 0 || (K % 256) != 0 || (K / 128) > 63) return -3;
    const int total_tiles = (N / 32) * MAX_CTAS;
    if (total_tiles <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int persist_cap = (TILE <= 8) ? 288 : 240;  // 6/SM (TILE=8) or 5/SM (TILE=16,32)
    dim3 grid((total_tiles < persist_cap) ? total_tiles : persist_cap, 1, 1);
    dim3 block(128, 1, 1);

    const bool zs = (zero_scatter != 0) && (Row2Tok != nullptr);
    const bool rm = zs || (as_row_major != 0);
    const bool fs = (fuse_situ != 0);

    #define C13(TV, MC, Z, R, F) return dispatch_mx7_w13<TV, MC, Z, R, F>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const int32_t*>(Row2Tok), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint4*>(WS3), static_cast<const uint16_t*>(FLIPS), \
        static_cast<const int32_t*>(F_OFF), static_cast<const float*>(Ones), \
        static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileCompactStart), \
        static_cast<const float*>(Gammas), static_cast<__nv_bfloat16*>(Out), \
        N, K, MAX_CTAS, base_ws)

    if (TILE == 8) {
        if (!zs && !rm && !fs) { C13(8, 6, false, false, false); }
        return -4;
    } else if (TILE == 16) {
        if      (zs && fs)          { C13(16, 5, true,  true,  true);  }
        else if (zs && !fs)         { C13(16, 5, true,  true,  false); }
        else if (!zs && rm && fs)   { C13(16, 5, false, true,  true);  }
        else if (!zs && rm && !fs)  { C13(16, 5, false, true,  false); }
        else if (!zs && !rm && fs)  { C13(16, 5, false, false, true);  }
        else                        { C13(16, 5, false, false, false); }
    } else if (TILE == 32) {
        if      (zs && fs)          { C13(32, 5, true,  true,  true);  }
        else if (zs && !fs)         { C13(32, 5, true,  true,  false); }
        else if (!zs && rm && fs)   { C13(32, 5, false, true,  true);  }
        else if (!zs && rm && !fs)  { C13(32, 5, false, true,  false); }
        else if (!zs && !rm && fs)  { C13(32, 5, false, false, true);  }
        else                        { C13(32, 5, false, false, false); }
    }
    #undef C13
    return -1;
}

extern "C" int mx7_w2_c32_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    if (N != 4608 || K != 288) return -2;
    const int total_tiles = (N / 128) * MAX_CTAS;
    const bool fc = (fuse_combine != 0);
    if (total_tiles <= 0 && !fc) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool fq = (fuse_quant != 0);
    const bool rm = (!fq) && (as_row_major != 0);
    const int eff_as_stride = (as_row_stride > 0) ? as_row_stride : 9;
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);

    const int persist_cap = (TILE <= 16) ? 288 : 192;
    const int grid_x = fc ? persist_cap : ((total_tiles < persist_cap) ? total_tiles : persist_cap);
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);

    #define C2(TV, MC, CPP, R, Q, C) return dispatch_mx7_w2<TV, MC, CPP, R, Q, C>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint4*>(WS3), \
        static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF), \
        static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), static_cast<__nv_bfloat16*>(Out), \
        static_cast<const int32_t*>(Src2Compact), static_cast<const int32_t*>(NLivePtr), \
        static_cast<uint32_t*>(PanelDone), static_cast<uint32_t*>(BufGen), \
        static_cast<__nv_bfloat16*>(SendBuf), \
        N, K, MAX_CTAS, T, base_ws, eff_as_stride, eff_down_stride)

    if (TILE == 8) {
        if (!fq && !fc && !rm) { C2(8, 6, 48, false, false, false); }
        return -4;
    } else if (TILE == 16) {
        if      (fq && fc)          { C2(16, 6, 48, false, true,  true);  }
        else if (fq && !fc)         { C2(16, 6, 48, false, true,  false); }
        else if (!fq && fc && rm)   { C2(16, 6, 48, true,  false, true);  }
        else if (!fq && fc && !rm)  { C2(16, 6, 48, false, false, true);  }
        else if (!fq && !fc && rm)  { C2(16, 6, 48, true,  false, false); }
        else                        { C2(16, 6, 48, false, false, false); }
    } else if (TILE == 32) {
        if      (fq && fc)          { C2(32, 4, 32, false, true,  true);  }
        else if (fq && !fc)         { C2(32, 4, 32, false, true,  false); }
        else if (!fq && fc && rm)   { C2(32, 4, 32, true,  false, true);  }
        else if (!fq && fc && !rm)  { C2(32, 4, 32, false, false, true);  }
        else if (!fq && !fc && rm)  { C2(32, 4, 32, true,  false, false); }
        else                        { C2(32, 4, 32, false, false, false); }
    }
    #undef C2
    return -1;
}
