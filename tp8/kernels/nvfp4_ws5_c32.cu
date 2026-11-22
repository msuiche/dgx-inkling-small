// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) C=32 (B=32, M=17..256) 0-ULP Bit-Exact WS5 NVFP4 MoE Kernels
// Targets Inkling-2-Small TP8 L0..39 NVFP4 MoE (E=384, H=4608, inter=288, topk=8):
//   - NV_W13 (N=576,  K=4608): BN=32 (4 warps), 8,832 B 128B-aligned contiguous super-slabs
//   - NV_W2  (N=4608, K=288):  BN=128 (4 warps, SUB_N=4), unified 1,440 B WS5 + 128B-aligned W_PACK
//
// Key C=32 (M=17..256) Architecture (100% 0-ULP bit-exact, literal nd == 0):
//   1. Closes the M=129..256 (TILE=32) Triton fallback dead zone (-26.23 + -16.73 = 42.96 ms/step
//      in live GB10 C=32 trace #96!):
//        - TILE=16 (M=17..128):  BM=16, BK_DOT=256 (W13) / BK_DOT=32 (W2) -> nd == 0 vs decode!
//        - TILE=32 (M=129..256): Native single-pass BM=32 with ZERO weight re-reads or re-unpacks!
//          * W13 uses half-slab double-buffered SMEM (BKP_HALF=64, SA_STRIDE=80 -> 20.6 KiB -> 4 CTAs/SM
//            = 192 persistent CTAs) matching BK_DOT=128 exact MMA order -> literal nd == 0!
//          * W2 uses single-pass BM=32 (21.3 KiB -> 4 CTAs/SM = 192 persistent CTAs = 6x32 panel-wave)
//            matching BK_DOT=64 + zero-padded K=288 tail MMA order -> literal nd == 0!
//   2. Tight Variable-Count Row Predication (ZERO wasted zero-padded rows):
//      Dynamic `has_m8`, `has_m16`, `has_m24` flags skip ALL dead 8-row SMEM loads and dead 16-row
//      Tensor Core MMAs (`im=1` completely skipped when `live_m <= 16`!).
//   3. Zero-Scatter `Row2Tok` (`sTok[BM]` SMEM cache + row-major `AS`), 0-way SMEM banking
//      (`SA_STRIDE=144/80` W13, `152` W2), `WS_BASE` bit-7 `P_OFF` bypass (skips cold DRAM `P_OFF`
//      on 86.7% W13 / 85.7% W2 experts), `ld.global.cs` L2 eviction on weights, inter-tile L2 prefetch.
//   4. Fused W13 SiTU -> compact dense `SiTU_Out[compact_row, 288]` (<=1.125 MiB) + `RowAmaxW2`
//      + Fused W2 OTF strictly-unfused NVFP4 quant-dequant (overlapped with `prefetch_l2(wp_base)`).
//   5. 6-Panel-Wave Double-Buffered Compact L2-Resident Fused W2 Combine (`FUSE_COMBINE=true`):
//      Writes `DownPanel[2][6][max_routed][128]` (0.75..6.0 MiB TOTAL -> 100% L2-resident!),
//      mathematically deadlock-free `PanelDone` + `BufGen` self-resetting release/acquire sync,
//      exact slot 0..7 FP32 summation directly into `SendBuf` + PDL trigger -> eliminates
//      `18.87 MB/L x 2` Down DRAM spill!
//   6. Drop-in S7 wrappers (`nvfp4_w13_ws5_fwd`, `nvfp4_w2_ws5_fwd`) preserved untouched for M<=16.
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC nvfp4_ws5_c32.cu -o libnvfp4_ws5_c32.so
//   (CRITICAL: --fmad=false required; OMIT --ftz=true to match Triton ftz=False!)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

static constexpr int W13_SS_BYTES = 8832;  // 8192 W + 512 WS5_LO4 + 128 WS5_HI1 = 69 * 128 B
static constexpr float FP32_MAX_VAL = 3.4028234663852886e38f;
static constexpr float INV_6 = 1.0f / 6.0f;
static constexpr float INV_25 = 1.0f / 25.0f;
static constexpr float INV_1536 = 1.0f / 1536.0f;

// ---- PTX Memory & Synchronization Helpers ----

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

static __device__ __forceinline__ uint32_t ld_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint8_t ld_cs_u8(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u8 %0, [%1];" : "=r"(v) : "l"(p));
    return static_cast<uint8_t>(v);
}

static __device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
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

// ---- Exact IEEE-754 & Conversion Helpers ----

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
static __device__ __forceinline__ uint16_t f32_to_bf16_rn(float f) {
    uint16_t b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(f));
    return b;
}

static __device__ __forceinline__ int64_t scale_offset_decode(int64_t row, int col, int cols) {
    int64_t tile = (row >> 3) * ((cols + 3) >> 2) + (col >> 2);
    return (tile << 5) + ((row & 7) << 2) + (col & 3);
}

static __device__ __forceinline__ uint32_t cvt_e4m3x2_to_f16x2(uint16_t e4x2) {
    uint32_t h2;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(e4x2));
    return h2;
}

static __device__ __forceinline__ void unpack_e2m1x8(
    uint32_t packed4, uint32_t& lo01, uint32_t& lo23, uint32_t& hi01, uint32_t& hi23
) {
    asm volatile(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        ".reg .b32 t0, t1, t2, t3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 t0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 t1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 t2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t3, b3;\n\t"
        "prmt.b32 %0, t0, t1, 0x5410;\n\t"
        "prmt.b32 %1, t2, t3, 0x5410;\n\t"
        "prmt.b32 %2, t0, t1, 0x7632;\n\t"
        "prmt.b32 %3, t2, t3, 0x7632;\n\t"
        "}\n\t"
        : "=r"(lo01), "=r"(lo23), "=r"(hi01), "=r"(hi23)
        : "r"(packed4)
    );
}

static __device__ __forceinline__ void unpack_e2m1x8_lo(uint32_t packed4, uint32_t& lo01, uint32_t& lo23) {
    asm volatile(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        ".reg .b32 t0, t1, t2, t3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 t1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 t2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t3, b3;\n\t"
        "prmt.b32 %0, t0, t1, 0x5410;\n\t"
        "prmt.b32 %1, t2, t3, 0x5410;\n\t"
        "}\n\t"
        : "=r"(lo01), "=r"(lo23)
        : "r"(packed4)
    );
}

static __device__ __forceinline__ void unpack_e2m1x8_hi(uint32_t packed4, uint32_t& hi01, uint32_t& hi23) {
    asm volatile(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        ".reg .b32 t0, t1, t2, t3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 t1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 t2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t3, b3;\n\t"
        "prmt.b32 %0, t0, t1, 0x7632;\n\t"
        "prmt.b32 %1, t2, t3, 0x7632;\n\t"
        "}\n\t"
        : "=r"(hi01), "=r"(hi23)
        : "r"(packed4)
    );
}

static __device__ __forceinline__ uint32_t mul_f16x2(uint32_t a, uint32_t b) {
    uint32_t out;
    asm volatile("mul.f16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

static __device__ __forceinline__ uint32_t splat_lo_f16(uint32_t f16x2) {
    uint32_t out;
    asm volatile("prmt.b32 %0, %1, %1, 0x1010;" : "=r"(out) : "r"(f16x2));
    return out;
}

static __device__ __forceinline__ uint32_t splat_hi_f16(uint32_t f16x2) {
    uint32_t out;
    asm volatile("prmt.b32 %0, %1, %1, 0x3232;" : "=r"(out) : "r"(f16x2));
    return out;
}

static __device__ __forceinline__ void mma_m16n8k16_f16(
    float (&acc)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

static __device__ __forceinline__ uint32_t pack_bf16x2(float f0, float f1) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1));
    return out;
}

static __device__ __forceinline__ void decode_ws5_x8(
    uint32_t lo4, uint8_t hi1, uint32_t base_e, uint8_t (&s_bytes)[8]
) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t c = ((lo4 >> (i * 4)) & 0xFu) | ((static_cast<uint32_t>(hi1 >> i) & 1u) << 4);
        s_bytes[i] = static_cast<uint8_t>(base_e + c);
    }
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

// ---- Exact 0-ULP Register-Compact NVFP4 Quant-Dequant into sA ----
// Strictly UNFUSED sub_rn(mul_rn(mul_rn(...))) matches prefill_moe_qf_s8<0> & Triton _situ_quantize_kernel!
static __device__ __forceinline__ void eval_nvfp4_half4_unfused(
    uint4 v,
    float inv6, float scale6,
    float inv4, float scale4,
    float outer_scale,
    uint32_t& pk6_w, uint32_t& pk4_w,
    float& err6_h, float& err4_h
) {
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
    uint32_t p6 = 0u, p4 = 0u;
    float e6_0 = 0.0f, e6_1 = 0.0f, e6_2 = 0.0f, e6_3 = 0.0f;
    float e4_0 = 0.0f, e4_1 = 0.0f, e4_2 = 0.0f, e4_3 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x0 = bf16_bits_to_f32(static_cast<uint16_t>(words[k] & 0xFFFFu));
        const float x1 = bf16_bits_to_f32(static_cast<uint16_t>(words[k] >> 16));
        float pe6, pe4;
        {
            uint32_t qbyte, h;
            float a = mul_rn(x0, inv6), b = mul_rn(x1, inv6);
            asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
                "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
                : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
            p6 |= (qbyte << (8 * k));
            float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
            float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
            pe6 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale6), outer_scale), x0)),
                         fabsf(sub_rn(mul_rn(mul_rn(r1, scale6), outer_scale), x1)));
        }
        {
            uint32_t qbyte, h;
            float a = mul_rn(x0, inv4), b = mul_rn(x1, inv4);
            asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
                "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
                : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
            p4 |= (qbyte << (8 * k));
            float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
            float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
            pe4 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale4), outer_scale), x0)),
                         fabsf(sub_rn(mul_rn(mul_rn(r1, scale4), outer_scale), x1)));
        }
        if (k == 0)      { e6_0 = pe6; e4_0 = pe4; }
        else if (k == 1) { e6_1 = pe6; e4_1 = pe4; }
        else if (k == 2) { e6_2 = pe6; e4_2 = pe4; }
        else             { e6_3 = pe6; e4_3 = pe4; }
    }
    pk6_w = p6;
    pk4_w = p4;
    err6_h = add_rn(add_rn(e6_0, e6_1), add_rn(e6_2, e6_3));
    err4_h = add_rn(add_rn(e4_0, e4_1), add_rn(e4_2, e4_3));
}

static __device__ __forceinline__ void quant_dequant_nvfp4_16_to_sA(
    const uint4 v0, const uint4 v1,
    float enc_s, float dec_s, float out_s,
    uint16_t* dst_lo, uint16_t* dst_hi
) {
    float bmax = 0.0f;
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
        float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
        bmax = fmaxf(bmax, fmaxf(fabsf(a), fabsf(b)));
    }

    float s6_raw = fminf(mul_rn(bmax, mul_rn(enc_s, INV_6)), 448.0f);
    float s4_raw = fminf(mul_rn(bmax, mul_rn(enc_s, 0.25f)), 448.0f);

    uint16_t s_e4m3x2;
    uint32_t s_f16x2;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %3, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, %0;"
        : "=&h"(s_e4m3x2), "=r"(s_f16x2)
        : "f"(s6_raw), "f"(s4_raw));

    float scale6 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 & 0xFFFFu)));
    float scale4 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 >> 16)));
    float inv6 = fminf(div_rn(1.0f, mul_rn(scale6, dec_s)), FP32_MAX_VAL);
    float inv4 = fminf(div_rn(1.0f, mul_rn(scale4, dec_s)), FP32_MAX_VAL);

    uint32_t pk6_lo, pk4_lo, pk6_hi, pk4_hi;
    float e6_lo, e4_lo, e6_hi, e4_hi;
    eval_nvfp4_half4_unfused(v0, inv6, scale6, inv4, scale4, out_s, pk6_lo, pk4_lo, e6_lo, e4_lo);
    eval_nvfp4_half4_unfused(v1, inv6, scale6, inv4, scale4, out_s, pk6_hi, pk4_hi, e6_hi, e4_hi);

    const bool choose4 = (add_rn(e4_lo, e4_hi) < add_rn(e6_lo, e6_hi));
    const uint32_t a4_0 = choose4 ? pk4_lo : pk6_lo;
    const uint32_t a4_1 = choose4 ? pk4_hi : pk6_hi;
    const uint32_t as_f16x2 = choose4 ? splat_hi_f16(s_f16x2) : splat_lo_f16(s_f16x2);

    uint32_t lo01, lo23, hi01, hi23;
    unpack_e2m1x8(a4_0, lo01, lo23, hi01, hi23);
    *reinterpret_cast<uint2*>(dst_lo)     = make_uint2(mul_f16x2(lo01, as_f16x2), mul_f16x2(lo23, as_f16x2));
    *reinterpret_cast<uint2*>(dst_hi)     = make_uint2(mul_f16x2(hi01, as_f16x2), mul_f16x2(hi23, as_f16x2));
    unpack_e2m1x8(a4_1, lo01, lo23, hi01, hi23);
    *reinterpret_cast<uint2*>(dst_lo + 4) = make_uint2(mul_f16x2(lo01, as_f16x2), mul_f16x2(lo23, as_f16x2));
    *reinterpret_cast<uint2*>(dst_hi + 4) = make_uint2(mul_f16x2(hi01, as_f16x2), mul_f16x2(hi23, as_f16x2));
}

// =========================================================================
// Kernel 1A (TILE=8,16): NV_W13 (N=576, K=4608), BM=TILE, BK_DOT=256 (0-ULP)
//   - Persistent 1D grid: 288 CTAs (TILE=8, 6/SM) or 240 CTAs (TILE=16, 5/SM)
//   - SA_STRIDE = 144 (0-way SMEM bank conflicts on LDS.64)
//   - Zero-Scatter Row2Tok + WS_BASE bit-7 P_OFF bypass + optional FUSE_SITU
// =========================================================================
template <int TILE, int MIN_CTA, int PDL_WAIT_MODE, bool TRIG_DEP, bool ZERO_SCATTER, bool AS_ROW_MAJOR, bool FUSE_SITU>
__global__ __launch_bounds__(128, MIN_CTA)
void nvfp4_w13_ws5_t16_kernel(
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
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    uint32_t* __restrict__ RowAmaxW2,
    __nv_bfloat16* __restrict__ Out,
    int N, int K, int MAX_CTAS
) {
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int BN = 32;
    constexpr int BKP = 128;
    constexpr int M_ROWS = (TILE <= 8) ? 8 : 16;
    constexpr int SA_STRIDE = BKP + 16;  // 144 halfs -> 0-way LDS.64
    constexpr int K_STEPS = 18;          // 4608 / 256

    const int n_panels = N / BN;
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

    const int KP = K >> 1;
    const int KG = K >> 4;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int w_off = warp * 1024 + lane * 16;
    const int ws_lo_off = 8192 + warp * 128 + lane * 4;
    const int ws_hi_off = 8704 + warp * 32 + lane;
    const int x_off0 = g * SA_STRIDE + 4 * lt;
    const int x_off1 = (g + 8) * SA_STRIDE + 4 * lt;

    alignas(16) __shared__ uint16_t sA[2][2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ int32_t  sTok[M_ROWS];
    alignas(16) __shared__ float    sPanelAmax[WARPS][M_ROWS];

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

        const int64_t panel_uid = e * n_panels + nb;
        const int64_t warp_uid = panel_uid * WARPS + warp;
        const uint8_t* panel_base = SLABS + panel_uid * static_cast<int64_t>(9 * W13_SS_BYTES);

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_first_tile) {
                if ((lane & 7) == 0) {
                    prefetch_l2(panel_base + w_off);
                    prefetch_l2(panel_base + w_off + 512);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }
        if (live_m == 0) continue;

        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e]);
        const uint32_t base_e = raw_base & 0x7Fu;
        const bool exp_no_patch = (raw_base & 0x80u) != 0u;

        uint4 cur_wa = ld_cs_u4(panel_base + w_off);
        uint4 cur_wb = ld_cs_u4(panel_base + w_off + 512);
        uint32_t cur_ws_lo = ld_cs_u32(panel_base + ws_lo_off);
        uint8_t  cur_ws_hi = ld_cs_u8(panel_base + ws_hi_off);

        int32_t p_cur = 0, p_end = 0;
        if (!exp_no_patch) {
            p_cur = P_OFF[warp_uid];
            p_end = P_OFF[warp_uid + 1];
        }
        const bool has_patches = (p_cur < p_end);

        if constexpr (ZERO_SCATTER) {
            if (tid < live_m) sTok[tid] = Row2Tok[row_base + tid];
        }

        if (live_m < M_ROWS) {
            const int zero_q = (M_ROWS - live_m) * 32;
            for (int zidx = tid; zidx < zero_q; zidx += 128) {
                const int lm = live_m + (zidx >> 5);
                const int qk = zidx & 31;
                const int off = lm * SA_STRIDE + (qk << 2);
                *reinterpret_cast<uint2*>(sA[0][0] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[0][1] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[1][0] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[1][1] + off) = make_uint2(0u, 0u);
            }
        }

        auto load_a_slab = [&](int ks, int buf, bool use_smem_tok) {
            const int total_q = live_m * 32;
            const int kp0 = ks * BKP;
            const int kg0 = ks * 16;
            uint16_t* dst_lo = sA[buf][0];
            uint16_t* dst_hi = sA[buf][1];
            for (int aidx = tid; aidx < total_q; aidx += 128) {
                const int lm = aidx >> 5;
                const int qk = aidx & 31;
                const int64_t row = static_cast<int64_t>(row_base + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                    : row;
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + a_row * KP + kp0 + (qk << 2));
                const int kg = kg0 + (qk >> 1);
                uint8_t as_b;
                if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                    as_b = AS[a_row * KG + kg];
                } else {
                    as_b = AS[scale_offset_decode(row, kg, KG)];
                }
                const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                uint32_t lo01, lo23, hi01, hi23;
                unpack_e2m1x8(a4, lo01, lo23, hi01, hi23);
                const int off = lm * SA_STRIDE + (qk << 2);
                *reinterpret_cast<uint2*>(dst_lo + off) = make_uint2(mul_f16x2(lo01, as_f16x2), mul_f16x2(lo23, as_f16x2));
                *reinterpret_cast<uint2*>(dst_hi + off) = make_uint2(mul_f16x2(hi01, as_f16x2), mul_f16x2(hi23, as_f16x2));
            }
        };

        load_a_slab(0, 0, false);
        __syncthreads();

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t ws_h2[4];
        const bool load_hi = (TILE > 8) && (live_m > 8);

        for (int ks = 0; ks < K_STEPS; ++ks) {
            const int cur_buf = ks & 1;
            const int nxt_buf = cur_buf ^ 1;
            const int ss = ks >> 1;

            if ((ks & 1) == 0) {
                uint8_t sb[8];
                decode_ws5_x8(cur_ws_lo, cur_ws_hi, base_e, sb);
                if (__builtin_expect(has_patches, 0)) {
                    const int kg_lo = ss * 32 + lt * 8;
                    const int kg_hi = kg_lo + 8;
                    for (int32_t pi = p_cur; pi < p_end; ++pi) {
                        const uint32_t rec = PATCHES[pi];
                        const int p_kg = static_cast<int>(rec >> 16);
                        const int p_g  = static_cast<int>((rec >> 8) & 0xFFu);
                        if (p_g == g && p_kg >= kg_lo && p_kg < kg_hi) {
                            sb[p_kg - kg_lo] = static_cast<uint8_t>(rec & 0xFFu);
                        }
                    }
                }
                ws_h2[0] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[0]) | (static_cast<uint16_t>(sb[2]) << 8));
                ws_h2[1] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[1]) | (static_cast<uint16_t>(sb[3]) << 8));
                ws_h2[2] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[4]) | (static_cast<uint16_t>(sb[6]) << 8));
                ws_h2[3] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[5]) | (static_cast<uint16_t>(sb[7]) << 8));
            }

            uint4 nxt_wa, nxt_wb;
            uint32_t nxt_ws_lo = cur_ws_lo;
            uint8_t  nxt_ws_hi = cur_ws_hi;
            if (ks + 1 < K_STEPS) {
                const int nxt_ks = ks + 1;
                const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                const int ks_off_b = (nxt_ks & 1) * 4096;
                nxt_wa = ld_cs_u4(nxt_ss + ks_off_b + w_off);
                nxt_wb = ld_cs_u4(nxt_ss + ks_off_b + w_off + 512);
                if ((nxt_ks & 1) == 0) {
                    nxt_ws_lo = ld_cs_u32(nxt_ss + ws_lo_off);
                    nxt_ws_hi = ld_cs_u8(nxt_ss + ws_hi_off);
                }
                load_a_slab(nxt_ks, nxt_buf, true);
            } else if (!is_last_tile) {
                const int nidx = idx + stride;
                const int nt = nidx / n_panels;
                const int nnb = nidx - nt * n_panels;
                const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                const uint8_t* nxt_pbase = SLABS + (ne * n_panels + nnb) * static_cast<int64_t>(9 * W13_SS_BYTES);
                if ((lane & 7) == 0) {
                    prefetch_l2(nxt_pbase + w_off);
                    prefetch_l2(nxt_pbase + w_off + 512);
                }
            }

            const int src0 = (g << 2) | ((ks & 1) * 2);
            const int src1 = src0 | 1;
            const uint32_t s0e = __shfl_sync(0xFFFFFFFFu, ws_h2[0], src0);
            const uint32_t s0o = __shfl_sync(0xFFFFFFFFu, ws_h2[1], src0);
            const uint32_t s1e = __shfl_sync(0xFFFFFFFFu, ws_h2[2], src0);
            const uint32_t s1o = __shfl_sync(0xFFFFFFFFu, ws_h2[3], src0);
            const uint32_t s2e = __shfl_sync(0xFFFFFFFFu, ws_h2[0], src1);
            const uint32_t s2o = __shfl_sync(0xFFFFFFFFu, ws_h2[1], src1);
            const uint32_t s3e = __shfl_sync(0xFFFFFFFFu, ws_h2[2], src1);
            const uint32_t s3o = __shfl_sync(0xFFFFFFFFu, ws_h2[3], src1);
            const uint32_t sc_q[4] = {
                (lt < 2) ? s0e : s0o, (lt < 2) ? s1e : s1o,
                (lt < 2) ? s2e : s2o, (lt < 2) ? s3e : s3o,
            };

            const uint32_t w_arr[8] = {
                cur_wa.x, cur_wa.y, cur_wa.z, cur_wa.w,
                cur_wb.x, cur_wb.y, cur_wb.z, cur_wb.w
            };

            const uint16_t* sA_lo = sA[cur_buf][0];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t sc0 = splat_lo_f16(sc_q[q]);
                const uint32_t sc1 = splat_hi_f16(sc_q[q]);
                const int q_off = q * 32;
                const uint2 a0_01 = *reinterpret_cast<const uint2*>(sA_lo + x_off0 + q_off);
                const uint2 a2_01 = *reinterpret_cast<const uint2*>(sA_lo + x_off0 + q_off + 16);
                uint2 a1_01 = make_uint2(0u, 0u), a3_01 = make_uint2(0u, 0u);
                if constexpr (TILE > 8) {
                    if (load_hi) {
                        a1_01 = *reinterpret_cast<const uint2*>(sA_lo + x_off1 + q_off);
                        a3_01 = *reinterpret_cast<const uint2*>(sA_lo + x_off1 + q_off + 16);
                    }
                }
                uint32_t lo01, lo23;
                unpack_e2m1x8_lo(w_arr[2 * q + 0], lo01, lo23);
                mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x,
                                 mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1));
                unpack_e2m1x8_lo(w_arr[2 * q + 1], lo01, lo23);
                mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y,
                                 mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1));
            }

            const uint16_t* sA_hi = sA[cur_buf][1];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t sc0 = splat_lo_f16(sc_q[q]);
                const uint32_t sc1 = splat_hi_f16(sc_q[q]);
                const int q_off = q * 32;
                const uint2 a0_01 = *reinterpret_cast<const uint2*>(sA_hi + x_off0 + q_off);
                const uint2 a2_01 = *reinterpret_cast<const uint2*>(sA_hi + x_off0 + q_off + 16);
                uint2 a1_01 = make_uint2(0u, 0u), a3_01 = make_uint2(0u, 0u);
                if constexpr (TILE > 8) {
                    if (load_hi) {
                        a1_01 = *reinterpret_cast<const uint2*>(sA_hi + x_off1 + q_off);
                        a3_01 = *reinterpret_cast<const uint2*>(sA_hi + x_off1 + q_off + 16);
                    }
                }
                uint32_t hi01, hi23;
                unpack_e2m1x8_hi(w_arr[2 * q + 0], hi01, hi23);
                mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x,
                                 mul_f16x2(hi01, sc0), mul_f16x2(hi23, sc1));
                unpack_e2m1x8_hi(w_arr[2 * q + 1], hi01, hi23);
                mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y,
                                 mul_f16x2(hi01, sc0), mul_f16x2(hi23, sc1));
            }

            if constexpr (TRIG_DEP) {
                if (is_last_tile && ks == K_STEPS - 2) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (ks + 1 < K_STEPS) {
                cur_wa = nxt_wa;
                cur_wb = nxt_wb;
                cur_ws_lo = nxt_ws_lo;
                cur_ws_hi = nxt_ws_hi;
                __syncthreads();
            }
        }

        const float scale = Ones[e];
        const int r0 = row_base + g;
        const int r1 = row_base + g + 8;
        const float rs0 = (g < live_m) ? RS[r0] : 0.0f;
        float o0 = mul_rn(mul_rn(acc[0], rs0), scale);
        float o1 = mul_rn(mul_rn(acc[1], rs0), scale);
        float o2 = 0.0f, o3 = 0.0f;
        if constexpr (TILE > 8) {
            const float rs1 = (g + 8 < live_m) ? RS[r1] : 0.0f;
            o2 = mul_rn(mul_rn(acc[2], rs1), scale);
            o3 = mul_rn(mul_rn(acc[3], rs1), scale);
        }

        if constexpr (FUSE_SITU) {
            const int inter = N >> 1;
            const int c_warp = nb * 16 + warp * 4;
            const int compact_base = TileCompactStart[t];
            const int c0 = compact_base + g;
            const int c1 = compact_base + g + 8;

            uint16_t y0_b = 0u;
            if (g < live_m) y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
            uint32_t p0 = static_cast<uint32_t>(y0_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
            uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
            if (lt == 0 && g < live_m) {
                *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * inter + c_warp) = make_uint2(p0, p1);
            }
            float a0 = (g < live_m) ? fabsf(bf16_bits_to_f32(y0_b)) : 0.0f;
            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
            if (lt == 0) sPanelAmax[warp][g] = a0;

            if constexpr (TILE > 8) {
                uint16_t y1_b = 0u;
                if (g + 8 < live_m) y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                uint32_t q0 = static_cast<uint32_t>(y1_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                if (lt == 0 && g + 8 < live_m) {
                    *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * inter + c_warp) = make_uint2(q0, q1);
                }
                float a1 = (g + 8 < live_m) ? fabsf(bf16_bits_to_f32(y1_b)) : 0.0f;
                a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 1));
                a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 2));
                if (lt == 0) sPanelAmax[warp][g + 8] = a1;
            }

            __syncthreads();
            if (tid < live_m) {
                float rmax = fmaxf(
                    fmaxf(sPanelAmax[0][tid], sPanelAmax[1][tid]),
                    fmaxf(sPanelAmax[2][tid], sPanelAmax[3][tid])
                );
                if (rmax > 0.0f) {
                    atomicMax(&RowAmaxW2[compact_base + tid], __float_as_uint(rmax));
                }
            }
        } else {
            const int n0 = nb * BN + warp * 8 + 2 * lt;
            if (g < live_m) {
                *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
            }
            if constexpr (TILE > 8) {
                if (g + 8 < live_m) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
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
// Kernel 1B (TILE=32): Native Single-Pass BM=32 NV_W13 (N=576, K=4608)
//   - Half-slab double-buffered SMEM: BKP_HALF=64, SA_STRIDE=80 (20.6 KiB -> 4 CTAs/SM = 192 CTAs!)
//   - Exact BK_DOT=128 order: 36 half-slabs of (4 lo MMAs + 4 hi MMAs) -> literal nd == 0!
//   - Single weight load (`ld.global.cs`) & single unpack across ALL 32 rows (`im=0, 1`)!
//   - Dynamic `has_m8/has_m16/has_m24` skips ALL dead rows (`im=1` skipped when `live_m <= 16`)!
// =========================================================================
template <int PDL_WAIT_MODE, bool TRIG_DEP, bool ZERO_SCATTER, bool AS_ROW_MAJOR, bool FUSE_SITU>
__global__ __launch_bounds__(128, 4)
void nvfp4_w13_ws5_t32_kernel(
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
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    uint32_t* __restrict__ RowAmaxW2,
    __nv_bfloat16* __restrict__ Out,
    int N, int K, int MAX_CTAS
) {
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int TILE = 32;
    constexpr int WARPS = 4;
    constexpr int BN = 32;
    constexpr int BKP_HALF = 64;
    constexpr int SA_STRIDE = BKP_HALF + 16;  // 80 halfs = 160 B = 40 banks === 8 mod 32 -> 0-way LDS.64!
    constexpr int K_HALF_STEPS = 36;          // 4608 / 128

    const int n_panels = N / BN;  // 18
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

    const int KP = K >> 1;  // 2304
    const int KG = K >> 4;  // 288
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int w_off = warp * 1024 + lane * 16;
    const int ws_lo_off = 8192 + warp * 128 + lane * 4;
    const int ws_hi_off = 8704 + warp * 32 + lane;

    alignas(16) __shared__ uint16_t sA[2][2][TILE * SA_STRIDE];  // 20,480 B
    alignas(16) __shared__ int32_t  sTok[TILE];                  //    128 B
    alignas(16) __shared__ float    sPanelAmax[WARPS][TILE];     //    512 B -> 21,120 B total (4 CTAs/SM!)

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

        const int64_t panel_uid = e * n_panels + nb;
        const int64_t warp_uid = panel_uid * WARPS + warp;
        const uint8_t* panel_base = SLABS + panel_uid * static_cast<int64_t>(9 * W13_SS_BYTES);

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_first_tile) {
                if ((lane & 7) == 0) {
                    prefetch_l2(panel_base + w_off);
                    prefetch_l2(panel_base + w_off + 512);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }
        if (live_m == 0) continue;

        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e]);
        const uint32_t base_e = raw_base & 0x7Fu;
        const bool exp_no_patch = (raw_base & 0x80u) != 0u;

        // Load first half-slab (wa) + first super-slab WS5 immediately!
        uint4 cur_w = ld_cs_u4(panel_base + w_off);
        uint32_t cur_ws_lo = ld_cs_u32(panel_base + ws_lo_off);
        uint8_t  cur_ws_hi = ld_cs_u8(panel_base + ws_hi_off);

        int32_t p_cur = 0, p_end = 0;
        if (!exp_no_patch) {
            p_cur = P_OFF[warp_uid];
            p_end = P_OFF[warp_uid + 1];
        }
        const bool has_patches = (p_cur < p_end);

        if constexpr (ZERO_SCATTER) {
            if (tid < live_m) sTok[tid] = Row2Tok[row_base + tid];
        }

        // Only zero padding rows inside active 16-row groups (rows >= 16 never read if live_m <= 16!)
        const int pad_limit = (live_m <= 8) ? 8 : ((live_m <= 16) ? 16 : ((live_m <= 24) ? 24 : 32));
        if (live_m < pad_limit) {
            const int zero_q = (pad_limit - live_m) * 16;
            for (int zidx = tid; zidx < zero_q; zidx += 128) {
                const int lm = live_m + (zidx >> 4);
                const int qk = zidx & 15;
                const int off = lm * SA_STRIDE + (qk << 2);
                *reinterpret_cast<uint2*>(sA[0][0] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[0][1] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[1][0] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[1][1] + off) = make_uint2(0u, 0u);
            }
        }

        auto load_a_half = [&](int h_step, int buf, bool use_smem_tok) {
            const int total_q = live_m * 16;  // 16 uint32s = 64 packed bytes per row
            const int kp0 = h_step * BKP_HALF;
            const int kg0 = h_step * 8;
            uint16_t* dst_lo = sA[buf][0];
            uint16_t* dst_hi = sA[buf][1];
            for (int aidx = tid; aidx < total_q; aidx += 128) {
                const int lm = aidx >> 4;
                const int qk = aidx & 15;
                const int64_t row = static_cast<int64_t>(row_base + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                    : row;
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + a_row * KP + kp0 + (qk << 2));
                const int kg = kg0 + (qk >> 1);
                uint8_t as_b;
                if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                    as_b = AS[a_row * KG + kg];
                } else {
                    as_b = AS[scale_offset_decode(row, kg, KG)];
                }
                const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                uint32_t lo01, lo23, hi01, hi23;
                unpack_e2m1x8(a4, lo01, lo23, hi01, hi23);
                const int off = lm * SA_STRIDE + (qk << 2);
                *reinterpret_cast<uint2*>(dst_lo + off) = make_uint2(mul_f16x2(lo01, as_f16x2), mul_f16x2(lo23, as_f16x2));
                *reinterpret_cast<uint2*>(dst_hi + off) = make_uint2(mul_f16x2(hi01, as_f16x2), mul_f16x2(hi23, as_f16x2));
            }
        };

        load_a_half(0, 0, false);
        __syncthreads();

        float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        uint32_t ws_h2[4];
        const bool has_m8  = (live_m > 8);
        const bool has_m16 = (live_m > 16);
        const bool has_m24 = (live_m > 24);

        const int x0_0 = g * SA_STRIDE + 4 * lt;
        const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;
        const int x1_0 = (g + 16) * SA_STRIDE + 4 * lt;
        const int x1_1 = (g + 24) * SA_STRIDE + 4 * lt;

        for (int hs = 0; hs < K_HALF_STEPS; ++hs) {
            const int cur_buf = hs & 1;
            const int nxt_buf = cur_buf ^ 1;
            const int ks = hs >> 1;      // 0..17
            const int half = hs & 1;     // 0 or 1
            const int ss = ks >> 1;      // 0..8

            // Decode WS5 once per super-slab (every 4 half-steps: hs = 0, 4, 8, ..., 32)
            if ((hs & 3) == 0) {
                uint8_t sb[8];
                decode_ws5_x8(cur_ws_lo, cur_ws_hi, base_e, sb);
                if (__builtin_expect(has_patches, 0)) {
                    const int kg_lo = ss * 32 + lt * 8;
                    const int kg_hi = kg_lo + 8;
                    for (int32_t pi = p_cur; pi < p_end; ++pi) {
                        const uint32_t rec = PATCHES[pi];
                        const int p_kg = static_cast<int>(rec >> 16);
                        const int p_g  = static_cast<int>((rec >> 8) & 0xFFu);
                        if (p_g == g && p_kg >= kg_lo && p_kg < kg_hi) {
                            sb[p_kg - kg_lo] = static_cast<uint8_t>(rec & 0xFFu);
                        }
                    }
                }
                ws_h2[0] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[0]) | (static_cast<uint16_t>(sb[2]) << 8));
                ws_h2[1] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[1]) | (static_cast<uint16_t>(sb[3]) << 8));
                ws_h2[2] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[4]) | (static_cast<uint16_t>(sb[6]) << 8));
                ws_h2[3] = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(sb[5]) | (static_cast<uint16_t>(sb[7]) << 8));
            }

            // Issue next half-slab weight load BEFORE load_a_half
            uint4 nxt_w;
            uint32_t nxt_ws_lo = cur_ws_lo;
            uint8_t  nxt_ws_hi = cur_ws_hi;
            if (hs + 1 < K_HALF_STEPS) {
                const int nhs = hs + 1;
                const int nks = nhs >> 1;
                const int nhalf = nhs & 1;
                const uint8_t* nss = panel_base + static_cast<int64_t>(nks >> 1) * W13_SS_BYTES;
                const int ks_off_b = (nks & 1) * 4096 + nhalf * 512;
                nxt_w = ld_cs_u4(nss + ks_off_b + w_off);
                if ((nhs & 3) == 0) {
                    nxt_ws_lo = ld_cs_u32(nss + ws_lo_off);
                    nxt_ws_hi = ld_cs_u8(nss + ws_hi_off);
                }
                load_a_half(nhs, nxt_buf, true);
            } else if (!is_last_tile) {
                const int nidx = idx + stride;
                const int nt = nidx / n_panels;
                const int nnb = nidx - nt * n_panels;
                const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                const uint8_t* nxt_pbase = SLABS + (ne * n_panels + nnb) * static_cast<int64_t>(9 * W13_SS_BYTES);
                if ((lane & 7) == 0) {
                    prefetch_l2(nxt_pbase + w_off);
                    prefetch_l2(nxt_pbase + w_off + 512);
                }
            }

            // Select the 2 scale-quarters for this half-slab (`half==0`: src0; `half==1`: src1)
            const int src = (g << 2) | ((ks & 1) * 2) | half;
            const uint32_t se0 = __shfl_sync(0xFFFFFFFFu, ws_h2[0], src);
            const uint32_t so0 = __shfl_sync(0xFFFFFFFFu, ws_h2[1], src);
            const uint32_t se1 = __shfl_sync(0xFFFFFFFFu, ws_h2[2], src);
            const uint32_t so1 = __shfl_sync(0xFFFFFFFFu, ws_h2[3], src);
            const uint32_t sc_h[2] = {
                (lt < 2) ? se0 : so0,
                (lt < 2) ? se1 : so1,
            };
            const uint32_t w_h[4] = {cur_w.x, cur_w.y, cur_w.z, cur_w.w};

            // Pass 1: 4 lo MMAs per half-slab (weight unpacked ONCE and reused across im=0,1!)
            const uint16_t* sA_lo = sA[cur_buf][0];
            #pragma unroll
            for (int dq = 0; dq < 2; ++dq) {
                const uint32_t sc0 = splat_lo_f16(sc_h[dq]);
                const uint32_t sc1 = splat_hi_f16(sc_h[dq]);
                const int q_off = dq * 32;

                uint32_t b0_0, b1_0, b0_1, b1_1;
                unpack_e2m1x8_lo(w_h[2 * dq + 0], b0_0, b1_0);
                b0_0 = mul_f16x2(b0_0, sc0); b1_0 = mul_f16x2(b1_0, sc1);
                unpack_e2m1x8_lo(w_h[2 * dq + 1], b0_1, b1_1);
                b0_1 = mul_f16x2(b0_1, sc0); b1_1 = mul_f16x2(b1_1, sc1);

                const uint2 a0 = *reinterpret_cast<const uint2*>(sA_lo + x0_0 + q_off);
                const uint2 a2 = *reinterpret_cast<const uint2*>(sA_lo + x0_0 + q_off + 16);
                uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                if (has_m8) {
                    a1 = *reinterpret_cast<const uint2*>(sA_lo + x0_1 + q_off);
                    a3 = *reinterpret_cast<const uint2*>(sA_lo + x0_1 + q_off + 16);
                }
                mma_m16n8k16_f16(acc[0], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                mma_m16n8k16_f16(acc[0], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);

                if (has_m16) {
                    const uint2 c0 = *reinterpret_cast<const uint2*>(sA_lo + x1_0 + q_off);
                    const uint2 c2 = *reinterpret_cast<const uint2*>(sA_lo + x1_0 + q_off + 16);
                    uint2 c1 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
                    if (has_m24) {
                        c1 = *reinterpret_cast<const uint2*>(sA_lo + x1_1 + q_off);
                        c3 = *reinterpret_cast<const uint2*>(sA_lo + x1_1 + q_off + 16);
                    }
                    mma_m16n8k16_f16(acc[1], c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                    mma_m16n8k16_f16(acc[1], c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                }
            }

            // Pass 2: 4 hi MMAs per half-slab
            const uint16_t* sA_hi = sA[cur_buf][1];
            #pragma unroll
            for (int dq = 0; dq < 2; ++dq) {
                const uint32_t sc0 = splat_lo_f16(sc_h[dq]);
                const uint32_t sc1 = splat_hi_f16(sc_h[dq]);
                const int q_off = dq * 32;

                uint32_t b0_0, b1_0, b0_1, b1_1;
                unpack_e2m1x8_hi(w_h[2 * dq + 0], b0_0, b1_0);
                b0_0 = mul_f16x2(b0_0, sc0); b1_0 = mul_f16x2(b1_0, sc1);
                unpack_e2m1x8_hi(w_h[2 * dq + 1], b0_1, b1_1);
                b0_1 = mul_f16x2(b0_1, sc0); b1_1 = mul_f16x2(b1_1, sc1);

                const uint2 a0 = *reinterpret_cast<const uint2*>(sA_hi + x0_0 + q_off);
                const uint2 a2 = *reinterpret_cast<const uint2*>(sA_hi + x0_0 + q_off + 16);
                uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                if (has_m8) {
                    a1 = *reinterpret_cast<const uint2*>(sA_hi + x0_1 + q_off);
                    a3 = *reinterpret_cast<const uint2*>(sA_hi + x0_1 + q_off + 16);
                }
                mma_m16n8k16_f16(acc[0], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                mma_m16n8k16_f16(acc[0], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);

                if (has_m16) {
                    const uint2 c0 = *reinterpret_cast<const uint2*>(sA_hi + x1_0 + q_off);
                    const uint2 c2 = *reinterpret_cast<const uint2*>(sA_hi + x1_0 + q_off + 16);
                    uint2 c1 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
                    if (has_m24) {
                        c1 = *reinterpret_cast<const uint2*>(sA_hi + x1_1 + q_off);
                        c3 = *reinterpret_cast<const uint2*>(sA_hi + x1_1 + q_off + 16);
                    }
                    mma_m16n8k16_f16(acc[1], c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                    mma_m16n8k16_f16(acc[1], c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                }
            }

            if constexpr (TRIG_DEP) {
                if (is_last_tile && hs == K_HALF_STEPS - 2) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (hs + 1 < K_HALF_STEPS) {
                cur_w = nxt_w;
                cur_ws_lo = nxt_ws_lo;
                cur_ws_hi = nxt_ws_hi;
                __syncthreads();
            }
        }

        const float scale = Ones[e];
        const int compact_base = FUSE_SITU ? TileCompactStart[t] : 0;

        #pragma unroll
        for (int im = 0; im < 2; ++im) {
            if (im == 1 && !has_m16) break;
            const int lm0 = im * 16 + g;
            const int lm1 = lm0 + 8;
            const int r0 = row_base + lm0;
            const int r1 = row_base + lm1;
            const float rs0 = (lm0 < live_m) ? RS[r0] : 0.0f;
            const float rs1 = (lm1 < live_m) ? RS[r1] : 0.0f;
            const float o0 = mul_rn(mul_rn(acc[im][0], rs0), scale);
            const float o1 = mul_rn(mul_rn(acc[im][1], rs0), scale);
            const float o2 = mul_rn(mul_rn(acc[im][2], rs1), scale);
            const float o3 = mul_rn(mul_rn(acc[im][3], rs1), scale);

            if constexpr (FUSE_SITU) {
                const int inter = N >> 1;
                const int c_warp = nb * 16 + warp * 4;
                const int c0 = compact_base + lm0;
                const int c1 = compact_base + lm1;

                uint16_t y0_b = 0u;
                if (lm0 < live_m) y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
                uint32_t p0 = static_cast<uint32_t>(y0_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
                uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
                if (lt == 0 && lm0 < live_m) {
                    *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * inter + c_warp) = make_uint2(p0, p1);
                }
                float a0 = (lm0 < live_m) ? fabsf(bf16_bits_to_f32(y0_b)) : 0.0f;
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
                if (lt == 0) sPanelAmax[warp][lm0] = a0;

                uint16_t y1_b = 0u;
                if (lm1 < live_m) y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                uint32_t q0 = static_cast<uint32_t>(y1_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                if (lt == 0 && lm1 < live_m) {
                    *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * inter + c_warp) = make_uint2(q0, q1);
                }
                float a1 = (lm1 < live_m) ? fabsf(bf16_bits_to_f32(y1_b)) : 0.0f;
                a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 1));
                a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 2));
                if (lt == 0) sPanelAmax[warp][lm1] = a1;
            } else {
                const int n0 = nb * BN + warp * 8 + 2 * lt;
                if (lm0 < live_m) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
                }
                if (lm1 < live_m) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
                }
            }
        }

        if constexpr (FUSE_SITU) {
            __syncthreads();
            if (tid < live_m) {
                float rmax = fmaxf(
                    fmaxf(sPanelAmax[0][tid], sPanelAmax[1][tid]),
                    fmaxf(sPanelAmax[2][tid], sPanelAmax[3][tid])
                );
                if (rmax > 0.0f) {
                    atomicMax(&RowAmaxW2[compact_base + tid], __float_as_uint(rmax));
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
// Kernel 2 (C32): Native Single-Pass BM=TILE NV_W2 (N=4608, K=288, BN=128)
//   - TILE=8,16: BM=8/16, BK_DOT=32, 288 CTAs (6 CTAs/SM = 6 panel-slots x 48 CTAs/panel)
//   - TILE=32:   BM=32,   BK_DOT=64 (+ exact tail zero-pad MMA), 192 CTAs (4 CTAs/SM = 6 x 32)
//   - Single weight load (`ld.global.cs`) & single unpack across ALL live_m rows!
//   - Dynamic `has_m8/has_m16/has_m24` skips ALL dead rows (`im=1` skipped when `live_m <= 16`)!
//   - Optional FUSE_QUANT (OTF strictly-unfused NVFP4 quant-dequant from compact SiTU_Out)
//   - Optional FUSE_COMBINE (6-Panel-Wave Double-Buffered Compact L2-Resident Combine -> SendBuf)
// =========================================================================
template <int TILE, int BK_DOT, int MIN_CTA, int CTAS_PER_PANEL, int PDL_WAIT_MODE, bool TRIG_DEP, bool AS_ROW_MAJOR, bool FUSE_QUANT, bool FUSE_COMBINE>
__global__ __launch_bounds__(128, MIN_CTA)
void nvfp4_w2_ws5_c32_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const uint32_t* __restrict__ RowAmaxW2,
    const float* __restrict__ WAmax,
    const float* __restrict__ WDenominator,
    const uint8_t* __restrict__ W_PACK,
    const uint32_t* __restrict__ WS5_PACK,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ Src2Compact,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ PanelDone,
    uint32_t* __restrict__ BufGen,
    __nv_bfloat16* __restrict__ SendBuf,
    int N, int K, int MAX_CTAS, int T, int as_row_stride, int down_row_stride
) {
    (void)K;
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int SUB_N = 4;
    constexpr int BN = 128;
    constexpr int KP = 144;
    constexpr int KG = 18;
    constexpr int K_STEPS = 9;
    constexpr int M_ROWS = TILE;
    constexpr int M_STEPS = (TILE <= 16) ? 1 : 2;
    constexpr int SA_STRIDE = KP + 8;  // 152 halfs = 304 B -> 0-way LDS.32!
    constexpr int N_PANELS = 36;
    constexpr int P_GROUP = 6;
    constexpr int N_WAVES = N_PANELS / P_GROUP;  // 6 waves

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int ws_row_base = (warp * SUB_N) * 8 + g;

    const int n_active = Tiles[2 * MAX_CTAS];
    const int bidx = static_cast<int>(blockIdx.x);
    const int stride = static_cast<int>(gridDim.x);

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

    alignas(16) __shared__ uint16_t sA[2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ uint8_t  sWS[2304];
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

        const int64_t panel_uid = e * N_PANELS + nb;
        const int64_t warp_uid = panel_uid * WARPS + warp;
        const uint8_t* wp_base = W_PACK + warp_uid * static_cast<int64_t>(K_STEPS) * 512;

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_very_first) {
                if ((lane & 7) == 0) {
                    prefetch_l2(wp_base + lane * 16);
                    prefetch_l2(wp_base + 512 + lane * 16);
                }
            }
        }

        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e]);
        const uint32_t base_e = raw_base & 0x7Fu;
        const bool exp_no_patch = (raw_base & 0x80u) != 0u;

        const uint32_t* ws_lo = WS5_PACK + panel_uid * 360;
        const uint32_t* ws_hi = ws_lo + 288;
        for (int widx = tid; widx < 288; widx += 128) {
            const uint32_t lo_w = ld_cs_u32(ws_lo + widx);
            const uint32_t hi_w = ld_cs_u32(ws_hi + (widx >> 2));
            const uint32_t hi_b = (hi_w >> ((widx & 3) * 8)) & 0xFFu;
            uint8_t* dst = sWS + (widx << 3);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t c = ((lo_w >> (i * 4)) & 0xFu) | (((hi_b >> i) & 1u) << 4);
                dst[i] = static_cast<uint8_t>(base_e + c);
            }
        }

        int32_t p_cur = 0, p_end = 0;
        if (!exp_no_patch) {
            p_cur = P_OFF[panel_uid];
            p_end = P_OFF[panel_uid + 1];
            if constexpr (PDL_WAIT_MODE == 2) {
                if (is_very_first && tid == 0 && p_cur < p_end) {
                    prefetch_l2(PATCHES + p_cur);
                }
            }
        }

        // Only zero padding rows inside active 8-row groups
        const int pad_limit = (live_m <= 8) ? 8
            : ((live_m <= 16) ? 16 : ((live_m <= 24) ? 24 : M_ROWS));
        if (live_m < pad_limit) {
            const int zero_q = (pad_limit - live_m) * 36;
            for (int zidx = tid; zidx < zero_q; zidx += 128) {
                const int dlm = zidx / 36;
                const int qk = zidx - dlm * 36;
                const int off = (live_m + dlm) * SA_STRIDE + (qk << 2);
                *reinterpret_cast<uint2*>(sA[0] + off) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(sA[1] + off) = make_uint2(0u, 0u);
            }
        }

        if constexpr (PDL_WAIT_MODE == 2) {
            if (!waited_pdl) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                waited_pdl = true;
            }
        }
        if (live_m == 0) return;

        // Overlap wp_base DRAM fetch with OTF quant-dequant (0 extra regs!)
        if constexpr (FUSE_QUANT) {
            if ((lane & 7) == 0) {
                prefetch_l2(wp_base + lane * 16);
                prefetch_l2(wp_base + 512 + lane * 16);
            }
        }

        const int compact_tile_base = (FUSE_QUANT || FUSE_COMBINE) ? TileCompactStart[t] : 0;

        // Scoped activation load / OTF quant-dequant BEFORE cur_w & acc (keeps regs compact, 0 spills!)
        {
            if constexpr (FUSE_QUANT) {
                const __nv_bfloat16* situ_in = reinterpret_cast<const __nv_bfloat16*>(A);
                const int total_blks = live_m * KG;
                for (int bq = tid; bq < total_blks; bq += 128) {
                    const int lm = bq / KG;
                    const int blk = bq - lm * KG;
                    const int crow = compact_tile_base + lm;
                    const float amax = __uint_as_float(ld_cg_u32(RowAmaxW2 + crow));
                    const float enc_s = fminf(div_rn(1536.0f, amax), FP32_MAX_VAL);
                    const float dec_s = div_rn(1.0f, enc_s);
                    const float out_s = mul_rn(amax, INV_1536);
                    const uint4* bp = reinterpret_cast<const uint4*>(situ_in + static_cast<int64_t>(crow) * 288 + blk * 16);
                    const int off0 = lm * SA_STRIDE + (blk << 3);
                    quant_dequant_nvfp4_16_to_sA(
                        ld_ca_u4(bp), ld_ca_u4(bp + 1),
                        enc_s, dec_s, out_s,
                        sA[0] + off0, sA[1] + off0
                    );
                }
            } else {
                const int total_q = live_m * 36;
                for (int aidx = tid; aidx < total_q; aidx += 128) {
                    const int lm = aidx / 36;
                    const int qk = aidx - lm * 36;
                    const int64_t row = static_cast<int64_t>(row_base + lm);
                    const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + row * KP + (qk << 2));
                    const int kg = qk >> 1;
                    uint8_t as_b;
                    if constexpr (AS_ROW_MAJOR) {
                        as_b = AS[row * as_row_stride + kg];
                    } else {
                        as_b = AS[scale_offset_decode(row, kg, KG)];
                    }
                    const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                    uint32_t lo01, lo23, hi01, hi23;
                    unpack_e2m1x8(a4, lo01, lo23, hi01, hi23);
                    const int off = lm * SA_STRIDE + (qk << 2);
                    *reinterpret_cast<uint2*>(sA[0] + off) = make_uint2(mul_f16x2(lo01, as_f16x2), mul_f16x2(lo23, as_f16x2));
                    *reinterpret_cast<uint2*>(sA[1] + off) = make_uint2(mul_f16x2(hi01, as_f16x2), mul_f16x2(hi23, as_f16x2));
                }
            }
        }

        uint4 cur_w0 = ld_cs_u4(wp_base + lane * 16);
        uint4 cur_w1;
        if constexpr (BK_DOT == 64) {
            cur_w1 = ld_cs_u4(wp_base + 512 + lane * 16);
        }
        __syncthreads();

        if (__builtin_expect(p_cur < p_end, 0)) {
            for (int32_t pi = p_cur + tid; pi < p_end; pi += 128) {
                const uint32_t rec = PATCHES[pi];
                sWS[rec >> 8] = static_cast<uint8_t>(rec & 0xFFu);
            }
            __syncthreads();
        }

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
        const int x0_0 = g * SA_STRIDE + 2 * lt;
        const int x0_1 = (g + 8) * SA_STRIDE + 2 * lt;
        const int x1_0 = (g + 16) * SA_STRIDE + 2 * lt;
        const int x1_1 = (g + 24) * SA_STRIDE + 2 * lt;

        if constexpr (BK_DOT == 32) {
            // Exact TILE<=16 order: 9 K-steps of (1 lo MMA + 1 hi MMA)
            #pragma unroll
            for (int ks = 0; ks < K_STEPS; ++ks) {
                uint4 nxt_w;
                if (ks + 1 < K_STEPS) {
                    nxt_w = ld_cs_u4(wp_base + static_cast<int64_t>(ks + 1) * 512 + lane * 16);
                } else if (has_next_tile) {
                    const int64_t ne = static_cast<int64_t>(Tiles[next_t]);
                    const int64_t npuid = ne * N_PANELS + next_nb;
                    const uint8_t* nwp = W_PACK + (npuid * WARPS + warp) * static_cast<int64_t>(K_STEPS) * 512;
                    if ((lane & 7) == 0) {
                        prefetch_l2(nwp + lane * 16);
                        if (warp == 0) prefetch_l2(WS5_PACK + npuid * 360 + (lane >> 3) * 32);
                    }
                }

                const int ks_off = ks * 16;
                const uint32_t w_sub[4] = {cur_w0.x, cur_w0.y, cur_w0.z, cur_w0.w};

                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const int s_idx = (ws_row_base + sub * 8) * 18 + ks * 2;
                    const uint16_t ws_pair = *reinterpret_cast<const uint16_t*>(sWS + s_idx);
                    const uint32_t ws_f16x2 = cvt_e4m3x2_to_f16x2(ws_pair);
                    const uint32_t sc0 = splat_lo_f16(ws_f16x2);
                    const uint32_t sc1 = splat_hi_f16(ws_f16x2);

                    uint32_t lo01, lo23, hi01, hi23;
                    unpack_e2m1x8(w_sub[sub], lo01, lo23, hi01, hi23);
                    const uint32_t bl0 = mul_f16x2(lo01, sc0), bl1 = mul_f16x2(lo23, sc1);
                    const uint32_t bh0 = mul_f16x2(hi01, sc0), bh1 = mul_f16x2(hi23, sc1);

                    {
                        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + ks_off);
                        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + ks_off + 8);
                        const uint32_t a1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + ks_off) : 0u;
                        const uint32_t a3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + ks_off + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bl0, bl1);
                    }
                    {
                        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + ks_off);
                        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + ks_off + 8);
                        const uint32_t a1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + ks_off) : 0u;
                        const uint32_t a3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + ks_off + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bh0, bh1);
                    }
                    if constexpr (M_STEPS == 2) {
                        if (has_m16) {
                            const uint32_t c0 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + ks_off);
                            const uint32_t c2 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + ks_off + 8);
                            const uint32_t c1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + ks_off) : 0u;
                            const uint32_t c3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + ks_off + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, bl0, bl1);

                            const uint32_t d0 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + ks_off);
                            const uint32_t d2 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + ks_off + 8);
                            const uint32_t d1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + ks_off) : 0u;
                            const uint32_t d3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + ks_off + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, bh0, bh1);
                        }
                    }
                }

                if constexpr (TRIG_DEP) {
                    if (is_very_last && ks == K_STEPS - 2) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }
                cur_w0 = nxt_w;
            }
        } else {
            // Exact TILE==32 order (BK_DOT=64): 4 pairs of (lo0, lo1, hi0, hi1) + zero-padded tail ks=8 -> nd == 0!
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int ks0 = p * 2;
                const int ks1 = ks0 + 1;
                uint4 nxt_w0 = ld_cs_u4(wp_base + static_cast<int64_t>(ks0 + 2) * 512 + lane * 16);

                const int o0 = ks0 * 16;
                const int o1 = ks1 * 16;
                const uint32_t w0_sub[4] = {cur_w0.x, cur_w0.y, cur_w0.z, cur_w0.w};
                const uint32_t w1_sub[4] = {cur_w1.x, cur_w1.y, cur_w1.z, cur_w1.w};

                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const int s_idx0 = (ws_row_base + sub * 8) * 18 + ks0 * 2;
                    const uint16_t ws_p0 = *reinterpret_cast<const uint16_t*>(sWS + s_idx0);
                    const uint16_t ws_p1 = *reinterpret_cast<const uint16_t*>(sWS + s_idx0 + 2);
                    const uint32_t wf0 = cvt_e4m3x2_to_f16x2(ws_p0);
                    const uint32_t wf1 = cvt_e4m3x2_to_f16x2(ws_p1);
                    const uint32_t s00 = splat_lo_f16(wf0), s01 = splat_hi_f16(wf0);
                    const uint32_t s10 = splat_lo_f16(wf1), s11 = splat_hi_f16(wf1);

                    // lo0 + lo1 (weight unpacked ONCE and reused across im=0, 1!)
                    {
                        uint32_t l0a, l0b, l1a, l1b;
                        unpack_e2m1x8_lo(w0_sub[sub], l0a, l0b);
                        l0a = mul_f16x2(l0a, s00); l0b = mul_f16x2(l0b, s01);
                        unpack_e2m1x8_lo(w1_sub[sub], l1a, l1b);
                        l1a = mul_f16x2(l1a, s10); l1b = mul_f16x2(l1b, s11);

                        {
                            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + o0);
                            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + o0 + 8);
                            const uint32_t a1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + o0) : 0u;
                            const uint32_t a3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + o0 + 8) : 0u;
                            mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, l0a, l0b);

                            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + o1);
                            const uint32_t b2 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + o1 + 8);
                            const uint32_t b1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + o1) : 0u;
                            const uint32_t b3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + o1 + 8) : 0u;
                            mma_m16n8k16_f16(acc[0][sub], b0, b1, b2, b3, l1a, l1b);
                        }
                        if constexpr (M_STEPS == 2) {
                            if (has_m16) {
                                const uint32_t c0 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + o0);
                                const uint32_t c2 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + o0 + 8);
                                const uint32_t c1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + o0) : 0u;
                                const uint32_t c3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + o0 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, l0a, l0b);

                                const uint32_t d0 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + o1);
                                const uint32_t d2 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + o1 + 8);
                                const uint32_t d1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + o1) : 0u;
                                const uint32_t d3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + o1 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, l1a, l1b);
                            }
                        }
                    }

                    // hi0 + hi1
                    {
                        uint32_t h0a, h0b, h1a, h1b;
                        unpack_e2m1x8_hi(w0_sub[sub], h0a, h0b);
                        h0a = mul_f16x2(h0a, s00); h0b = mul_f16x2(h0b, s01);
                        unpack_e2m1x8_hi(w1_sub[sub], h1a, h1b);
                        h1a = mul_f16x2(h1a, s10); h1b = mul_f16x2(h1b, s11);

                        {
                            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + o0);
                            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + o0 + 8);
                            const uint32_t a1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + o0) : 0u;
                            const uint32_t a3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + o0 + 8) : 0u;
                            mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, h0a, h0b);

                            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + o1);
                            const uint32_t b2 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + o1 + 8);
                            const uint32_t b1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + o1) : 0u;
                            const uint32_t b3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + o1 + 8) : 0u;
                            mma_m16n8k16_f16(acc[0][sub], b0, b1, b2, b3, h1a, h1b);
                        }
                        if constexpr (M_STEPS == 2) {
                            if (has_m16) {
                                const uint32_t c0 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + o0);
                                const uint32_t c2 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + o0 + 8);
                                const uint32_t c1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + o0) : 0u;
                                const uint32_t c3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + o0 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, h0a, h0b);

                                const uint32_t d0 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + o1);
                                const uint32_t d2 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + o1 + 8);
                                const uint32_t d1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + o1) : 0u;
                                const uint32_t d3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + o1 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, h1a, h1b);
                            }
                        }
                    }
                }

                cur_w0 = nxt_w0;
                if (p < 3) {
                    cur_w1 = ld_cs_u4(wp_base + static_cast<int64_t>(ks0 + 3) * 512 + lane * 16);
                }
                if constexpr (TRIG_DEP) {
                    if (is_very_last && p == 3) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }
            }

            // Tail ks = 8 (k = 256..287) with exact zero-pad 2nd m16n8k16 MMA for -0.0 parity
            {
                if (has_next_tile) {
                    const int64_t ne = static_cast<int64_t>(Tiles[next_t]);
                    const int64_t npuid = ne * N_PANELS + next_nb;
                    const uint8_t* nwp = W_PACK + (npuid * WARPS + warp) * static_cast<int64_t>(K_STEPS) * 512;
                    if ((lane & 7) == 0) {
                        prefetch_l2(nwp + lane * 16);
                        if (warp == 0) prefetch_l2(WS5_PACK + npuid * 360 + (lane >> 3) * 32);
                    }
                }
                const int o8 = 8 * 16;
                const uint32_t w_sub[4] = {cur_w0.x, cur_w0.y, cur_w0.z, cur_w0.w};

                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const int s_idx = (ws_row_base + sub * 8) * 18 + 16;
                    const uint16_t ws_pair = *reinterpret_cast<const uint16_t*>(sWS + s_idx);
                    const uint32_t ws_f16x2 = cvt_e4m3x2_to_f16x2(ws_pair);
                    const uint32_t sc0 = splat_lo_f16(ws_f16x2);
                    const uint32_t sc1 = splat_hi_f16(ws_f16x2);

                    uint32_t lo01, lo23, hi01, hi23;
                    unpack_e2m1x8(w_sub[sub], lo01, lo23, hi01, hi23);
                    const uint32_t bl0 = mul_f16x2(lo01, sc0), bl1 = mul_f16x2(lo23, sc1);
                    const uint32_t bh0 = mul_f16x2(hi01, sc0), bh1 = mul_f16x2(hi23, sc1);

                    {
                        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + o8);
                        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sA[0] + x0_0 + o8 + 8);
                        const uint32_t a1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + o8) : 0u;
                        const uint32_t a3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[0] + x0_1 + o8 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bl0, bl1);
                        mma_m16n8k16_f16(acc[0][sub], 0u, 0u, 0u, 0u, 0u, 0u);
                    }
                    {
                        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + o8);
                        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sA[1] + x0_0 + o8 + 8);
                        const uint32_t a1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + o8) : 0u;
                        const uint32_t a3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sA[1] + x0_1 + o8 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bh0, bh1);
                        mma_m16n8k16_f16(acc[0][sub], 0u, 0u, 0u, 0u, 0u, 0u);
                    }
                    if constexpr (M_STEPS == 2) {
                        if (has_m16) {
                            const uint32_t c0 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + o8);
                            const uint32_t c2 = *reinterpret_cast<const uint32_t*>(sA[0] + x1_0 + o8 + 8);
                            const uint32_t c1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + o8) : 0u;
                            const uint32_t c3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[0] + x1_1 + o8 + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, bl0, bl1);
                            mma_m16n8k16_f16(acc[1][sub], 0u, 0u, 0u, 0u, 0u, 0u);

                            const uint32_t d0 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + o8);
                            const uint32_t d2 = *reinterpret_cast<const uint32_t*>(sA[1] + x1_0 + o8 + 8);
                            const uint32_t d1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + o8) : 0u;
                            const uint32_t d3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sA[1] + x1_1 + o8 + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, bh0, bh1);
                            mma_m16n8k16_f16(acc[1][sub], 0u, 0u, 0u, 0u, 0u, 0u);
                        }
                    }
                }
            }
        }

        const float scale = Ones[e];
        const float w_amax_e = FUSE_QUANT ? __ldg(WAmax + e) : 0.0f;
        const float rcp_denom = FUSE_QUANT ? rcp_rn(mul_rn(1536.0f, __ldg(WDenominator))) : 0.0f;

        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            if (im == 1 && !has_m16) break;
            const int lm0 = im * 16 + g;
            const int lm1 = lm0 + 8;
            const int r0 = row_base + lm0;
            const int r1 = row_base + lm1;
            const int c0 = compact_tile_base + lm0;
            const int c1 = compact_tile_base + lm1;

            float rs0 = 0.0f, rs1 = 0.0f;
            if constexpr (FUSE_QUANT) {
                if (lm0 < live_m) {
                    const float am0 = __uint_as_float(ld_cg_u32(RowAmaxW2 + c0));
                    rs0 = mul_rn(mul_rn(am0, w_amax_e), rcp_denom);
                }
                if constexpr (TILE > 8) {
                    if (lm1 < live_m) {
                        const float am1 = __uint_as_float(ld_cg_u32(RowAmaxW2 + c1));
                        rs1 = mul_rn(mul_rn(am1, w_amax_e), rcp_denom);
                    }
                }
            } else {
                rs0 = (lm0 < live_m) ? RS[r0] : 0.0f;
                rs1 = (TILE > 8 && lm1 < live_m) ? RS[r1] : 0.0f;
            }

            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                const int col_local = (warp * SUB_N + sub) * 8 + 2 * lt;
                if constexpr (FUSE_COMBINE) {
                    if (lm0 < live_m) {
                        float o0 = mul_rn(mul_rn(acc[im][sub][0], rs0), scale);
                        float o1 = mul_rn(mul_rn(acc[im][sub][1], rs0), scale);
                        st_cg_u32(down_panel_buf + static_cast<int64_t>(c0) * BN + col_local, pack_bf16x2(o0, o1));
                    }
                    if constexpr (TILE > 8) {
                        if (lm1 < live_m) {
                            float o2 = mul_rn(mul_rn(acc[im][sub][2], rs1), scale);
                            float o3 = mul_rn(mul_rn(acc[im][sub][3], rs1), scale);
                            st_cg_u32(down_panel_buf + static_cast<int64_t>(c1) * BN + col_local, pack_bf16x2(o2, o3));
                        }
                    }
                } else {
                    const int n0 = nb * BN + col_local;
                    if (lm0 < live_m) {
                        float o0 = mul_rn(mul_rn(acc[im][sub][0], rs0), scale);
                        float o1 = mul_rn(mul_rn(acc[im][sub][1], rs0), scale);
                        *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
                    }
                    if constexpr (TILE > 8) {
                        if (lm1 < live_m) {
                            float o2 = mul_rn(mul_rn(acc[im][sub][2], rs1), scale);
                            float o3 = mul_rn(mul_rn(acc[im][sub][3], rs1), scale);
                            *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
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
static inline cudaError_t launch_nvfp4(
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
static int dispatch_w13_t16(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS, const int32_t* Row2Tok,
    const uint8_t* SLABS, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileCompactStart, const float* Gammas, uint32_t* RowAmaxW2,
    __nv_bfloat16* Out, int N, int K, int MAX_CTAS
) {
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define R16(PW, TR) err = launch_nvfp4( \
        nvfp4_w13_ws5_t16_kernel<TILE, MIN_CTA, PW, TR, ZS, RM, FS>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, \
        TileCompactStart, Gammas, RowAmaxW2, Out, N, K, MAX_CTAS)
    if      (pdl_wait_mode == 1 && trig_dep)  { R16(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { R16(0, true);  }
    else if (pdl_wait_mode == 1 && !trig_dep) { R16(1, false); }
    else if (pdl_wait_mode == 2 && trig_dep)  { R16(2, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { R16(2, false); }
    else                                      { R16(0, false); }
    #undef R16
    return static_cast<int>(err);
}

template <bool ZS, bool RM, bool FS>
static int dispatch_w13_t32(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS, const int32_t* Row2Tok,
    const uint8_t* SLABS, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileCompactStart, const float* Gammas, uint32_t* RowAmaxW2,
    __nv_bfloat16* Out, int N, int K, int MAX_CTAS
) {
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define R32(PW, TR) err = launch_nvfp4( \
        nvfp4_w13_ws5_t32_kernel<PW, TR, ZS, RM, FS>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, \
        TileCompactStart, Gammas, RowAmaxW2, Out, N, K, MAX_CTAS)
    if      (pdl_wait_mode == 1 && trig_dep)  { R32(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { R32(0, true);  }
    else if (pdl_wait_mode == 1 && !trig_dep) { R32(1, false); }
    else if (pdl_wait_mode == 2 && trig_dep)  { R32(2, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { R32(2, false); }
    else                                      { R32(0, false); }
    #undef R32
    return static_cast<int>(err);
}

template <int TILE, int BK_DOT, int MIN_CTA, int CPP, bool RM, bool FQ, bool FC>
static int dispatch_w2_pdl(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS,
    const uint32_t* RowAmaxW2, const float* WAmax, const float* WDenominator,
    const uint8_t* W_PACK, const uint32_t* WS5_PACK, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles, const int32_t* TileCompactStart,
    __nv_bfloat16* Out,
    const int32_t* Src2Compact, const int32_t* NLivePtr,
    uint32_t* PanelDone, uint32_t* BufGen, __nv_bfloat16* SendBuf,
    int N, int K, int MAX_CTAS, int T, int as_row_stride, int down_row_stride
) {
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define RUN2(PW, TR) err = launch_nvfp4( \
        nvfp4_w2_ws5_c32_kernel<TILE, BK_DOT, MIN_CTA, CPP, PW, TR, RM, FQ, FC>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, RowAmaxW2, WAmax, WDenominator, \
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, TileCompactStart, \
        Out, Src2Compact, NLivePtr, PanelDone, BufGen, SendBuf, \
        N, K, MAX_CTAS, T, as_row_stride, down_row_stride)
    if      (pdl_wait_mode == 2 && trig_dep)  { RUN2(2, true);  }
    else if (pdl_wait_mode == 1 && trig_dep)  { RUN2(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { RUN2(0, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { RUN2(2, false); }
    else if (pdl_wait_mode == 1 && !trig_dep) { RUN2(1, false); }
    else                                      { RUN2(0, false); }
    #undef RUN2
    return static_cast<int>(err);
}

}  // namespace

// =========================================================================
// C32 Entry Points
// =========================================================================

extern "C" int nvfp4_w13_ws5_c32_fwd(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* RowAmaxW2,
    void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int bk_dot, int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    (void)bk_dot;
    if (N != 576 || K != 4608) return -2;
    const int total_tiles = (N / 32) * MAX_CTAS;
    if (total_tiles <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int persist_cap = (TILE <= 8) ? 288 : ((TILE <= 16) ? 240 : 192);
    dim3 grid((total_tiles < persist_cap) ? total_tiles : persist_cap, 1, 1);
    dim3 block(128, 1, 1);

    const bool zs = (zero_scatter != 0) && (Row2Tok != nullptr);
    const bool rm = zs || (as_row_major != 0);
    const bool fs = (fuse_situ != 0);

    #define C16(TV, MC, Z, R, F) return dispatch_w13_t16<TV, MC, Z, R, F>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const int32_t*>(Row2Tok), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint8_t*>(WS_BASE), \
        static_cast<const int32_t*>(P_OFF), static_cast<const uint32_t*>(PATCHES), \
        static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
        static_cast<uint32_t*>(RowAmaxW2), static_cast<__nv_bfloat16*>(Out), N, K, MAX_CTAS)
    #define C32(Z, R, F) return dispatch_w13_t32<Z, R, F>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const int32_t*>(Row2Tok), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint8_t*>(WS_BASE), \
        static_cast<const int32_t*>(P_OFF), static_cast<const uint32_t*>(PATCHES), \
        static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
        static_cast<uint32_t*>(RowAmaxW2), static_cast<__nv_bfloat16*>(Out), N, K, MAX_CTAS)

    if (TILE == 8) {
        if (!zs && !rm && !fs) { C16(8, 6, false, false, false); }
        return -4;
    } else if (TILE == 16) {
        if      (zs && fs)          { C16(16, 5, true,  true,  true);  }
        else if (zs && !fs)         { C16(16, 5, true,  true,  false); }
        else if (!zs && rm && fs)   { C16(16, 5, false, true,  true);  }
        else if (!zs && rm && !fs)  { C16(16, 5, false, true,  false); }
        else if (!zs && !rm && fs)  { C16(16, 5, false, false, true);  }
        else                        { C16(16, 5, false, false, false); }
    } else if (TILE == 32) {
        if      (zs && fs)          { C32(true,  true,  true);  }
        else if (zs && !fs)         { C32(true,  true,  false); }
        else if (!zs && rm && fs)   { C32(false, true,  true);  }
        else if (!zs && rm && !fs)  { C32(false, true,  false); }
        else if (!zs && !rm && fs)  { C32(false, false, true);  }
        else                        { C32(false, false, false); }
    }
    #undef C16
    #undef C32
    return -1;
}

extern "C" int nvfp4_w2_ws5_c32_fwd(
    const void* A, const void* AS, const void* RS,
    const void* RowAmaxW2, const void* WAmax, const void* WDenominator,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T,
    int bk_dot, int as_row_major, int as_row_stride, int down_row_stride,
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
    const int eff_bk = (bk_dot == 64 || (bk_dot == 0 && TILE >= 32)) ? 64 : 32;
    const int eff_as_stride = (as_row_stride > 0) ? as_row_stride : 18;
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);

    const int persist_cap = (TILE <= 16) ? 288 : 192;
    const int grid_x = fc ? persist_cap : ((total_tiles < persist_cap) ? total_tiles : persist_cap);
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);

    #define CALL2(TV, BK, MC, CPP, R, Q, C) return dispatch_w2_pdl<TV, BK, MC, CPP, R, Q, C>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const uint32_t*>(RowAmaxW2), \
        static_cast<const float*>(WAmax), static_cast<const float*>(WDenominator), \
        static_cast<const uint8_t*>(W_PACK), static_cast<const uint32_t*>(WS5_PACK), \
        static_cast<const uint8_t*>(WS_BASE), static_cast<const int32_t*>(P_OFF), \
        static_cast<const uint32_t*>(PATCHES), static_cast<const float*>(Ones), \
        static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileCompactStart), \
        static_cast<__nv_bfloat16*>(Out), static_cast<const int32_t*>(Src2Compact), \
        static_cast<const int32_t*>(NLivePtr), static_cast<uint32_t*>(PanelDone), \
        static_cast<uint32_t*>(BufGen), static_cast<__nv_bfloat16*>(SendBuf), \
        N, K, MAX_CTAS, T, eff_as_stride, eff_down_stride)

    if (TILE == 8) {
        if (!fq && !fc && !rm) { CALL2(8, 32, 6, 48, false, false, false); }
        return -4;
    } else if (TILE == 16) {
        if      (fq && fc)          { CALL2(16, 32, 6, 48, false, true,  true);  }
        else if (fq && !fc)         { CALL2(16, 32, 6, 48, false, true,  false); }
        else if (!fq && fc && rm)   { CALL2(16, 32, 6, 48, true,  false, true);  }
        else if (!fq && fc && !rm)  { CALL2(16, 32, 6, 48, false, false, true);  }
        else if (!fq && !fc && rm)  { CALL2(16, 32, 6, 48, true,  false, false); }
        else                        { CALL2(16, 32, 6, 48, false, false, false); }
    } else if (TILE == 32) {
        if (eff_bk == 64) {
            if      (fq && fc)          { CALL2(32, 64, 4, 32, false, true,  true);  }
            else if (fq && !fc)         { CALL2(32, 64, 4, 32, false, true,  false); }
            else if (!fq && fc && rm)   { CALL2(32, 64, 4, 32, true,  false, true);  }
            else if (!fq && fc && !rm)  { CALL2(32, 64, 4, 32, false, false, true);  }
            else if (!fq && !fc && rm)  { CALL2(32, 64, 4, 32, true,  false, false); }
            else                        { CALL2(32, 64, 4, 32, false, false, false); }
        } else {
            if      (fq && fc)          { CALL2(32, 32, 4, 32, false, true,  true);  }
            else if (fq && !fc)         { CALL2(32, 32, 4, 32, false, true,  false); }
            else if (!fq && fc && !rm)  { CALL2(32, 32, 4, 32, false, false, true);  }
            else if (!fq && !fc && !rm) { CALL2(32, 32, 4, 32, false, false, false); }
            else                        { CALL2(32, 32, 4, 32, true,  false, false); }
        }
    }
    #undef CALL2
    return -1;
}

// Drop-in S7 backward-compatible ABI wrappers (M <= 16 untouched!)
extern "C" int nvfp4_w13_ws5_fwd(
    const void* A, const void* AS, const void* RS,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w13_ws5_c32_fwd(
        A, AS, RS, nullptr, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        nullptr, nullptr, nullptr, Out,
        N, K, MAX_CTAS, TILE, 256, 0, 0, 0, pdl_wait_mode, trig_dep, stream_ptr
    );
}

extern "C" int nvfp4_w2_ws5_fwd(
    const void* A, const void* AS, const void* RS,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w2_ws5_c32_fwd(
        A, AS, RS, nullptr, nullptr, nullptr,
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, nullptr, Out,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        N, K, MAX_CTAS, TILE, 0, 32, 0, 18, 0, 0, 0, pdl_wait_mode, trig_dep, stream_ptr
    );
}
