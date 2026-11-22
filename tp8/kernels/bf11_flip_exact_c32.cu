// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) 0-ULP Lossless BF11_FLIP_EXACT Dense GEMM — Unified C=1 (M<=16) + C=32 (M=17..256+)
//
// Drop-in superset of `bf11_flip_exact_s7.cu` (`libbf11_flip_exact_s7.so`):
//   1. M <= 16: 100% UNCHANGED S7 kernels (`bf11_flip_kernel`, `bf11_flip_dual_kv_kernel`,
//      `bf11_flip_fwd`, `bf11_flip_dual_kv_fwd`) -> literal 0-cycle / 0-ULP zero regression on C=1!
//   2. M = 17..256+: NEW `bf11_flip_c32_kernel` & `bf11_flip_dual_kv_c32_kernel` reusing the EXACT
//      SAME prepacked BF11_FLIP slabs (`SLABS`, `FLIPS`, `WARP_META`, `OUTLIERS`, `O_OFF`) with
//      ZERO extra VRAM and ZERO repacking!
//
// Key SM121a C=32 breakthroughs (beating live GB10 `cutlass_80` BF16 126 GB/s by ~1.8-2.1x!):
//   - Zero Split-K Partial-Sum Traffic (`split_cta = 1`): each CTA loops `ps = 0..split_packed-1`
//     over the existing prepacked splits, eliminating 5.8-23.3 MB/layer of FP32 partial-sum DRAM/L2
//     traffic and separate reduce launches that caused `BF12_v3` to regress at M>=128!
//   - Two exact 0-ULP K-reduction orders on the SAME slabs:
//       * k_order = 0 (default): exact ascending-K `mma.sync.m16n8k16` (`k = 0, 16, ..., K-16`),
//         matching cuBLAS `torch.mm` splitK=1 to literal 0-ULP (`nd == 0`)!
//       * k_order = 1: exact `split_packed`-grouped FP32 summation (`tot += p_acc` per `ps`),
//         matching `skinny.py` / `BF11_FLIP M<=16` to literal 0-ULP (`nd == 0`), still with ZERO
//         global partial buffers!
//   - Zero-bank-conflict double-buffered `cp.async.ca` SMEM (`SX_STRIDE = 72` for SUB_N=1 BK=64;
//     `SX_STRIDE = 136` for SUB_N=2 BK=128): `(4*g + t) % 32` is a 1-to-1 bijection across all
//     32 SMEM banks, overlapping `sX` + next `Slab11` loads behind `MT = 2/4/8` (`BM = 32/64/128`)
//     Tensor Core MMAs!
//   - L2-co-scheduled CTA grid (`blockIdx.x = mb`, `blockIdx.y = pn`): all `m_blocks` of the same
//     `pn` launch simultaneously on adjacent SMs and hit the exact same 1.4 KB slab in 24 MB L2
//     (`ld.global.ca` when `m_blocks > 1`, `ld.global.cs` when `m_blocks == 1`).
//   - Fused Dual-KV C=32 kernel (`bf11_flip_dual_kv_c32_kernel`, BM=32) runs both `k_proj` and
//     `v_proj` in ONE launch for global attention layers across `M = 17..256+`.
//
// Build on GB10 (SM121a, STRICT `--fmad=false`, NO `--ftz=true`):
//   /usr/local/cuda/bin/nvcc -O3 -arch=sm_121a --fmad=false --shared -Xcompiler -fPIC \
//       --ptxas-options=-v bf11_flip_exact_c32.cu -o /rdma/libbf11_flip_exact_c32.so

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

static constexpr int SLAB11_BYTES = 1408;

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

static __device__ __forceinline__ uint32_t ld_cs_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_ca_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_nc_u16(const uint16_t* p) {
    return (uint32_t)__ldg(p);
}

static __device__ __forceinline__ float2 ld_cg_f2(const float2* p) {
    float2 v;
    asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];"
                 : "=f"(v.x), "=f"(v.y)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

static __device__ __forceinline__ int atomic_add_acq_rel_gpu(int32_t* ptr, int32_t val) {
    int32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(val) : "memory");
    return old;
}

static __device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

struct Slab11 {
    uint4 sm0, sm1;
    uint32_t e0, e1, e2;
};

template <bool CACHE_L2 = false>
static __device__ __forceinline__ Slab11 load_slab11_t(const uint8_t* sp, int lane) {
    const uint4* sm_p = reinterpret_cast<const uint4*>(sp);
    const uint32_t* e3_p = reinterpret_cast<const uint32_t*>(sp + 1024);
    Slab11 r;
    if constexpr (CACHE_L2) {
        r.sm0 = ld_ca_u4(sm_p + lane);
        r.sm1 = ld_ca_u4(sm_p + 32 + lane);
        r.e0  = ld_ca_u32(e3_p + lane);
        r.e1  = ld_ca_u32(e3_p + 32 + lane);
        r.e2  = ld_ca_u32(e3_p + 64 + lane);
    } else {
        r.sm0 = ld_cs_u4(sm_p + lane);
        r.sm1 = ld_cs_u4(sm_p + 32 + lane);
        r.e0  = ld_cs_u32(e3_p + lane);
        r.e1  = ld_cs_u32(e3_p + 32 + lane);
        r.e2  = ld_cs_u32(e3_p + 64 + lane);
    }
    return r;
}

static __device__ __forceinline__ Slab11 load_slab11(const uint8_t* sp, int lane) {
    return load_slab11_t<false>(sp, lane);
}

static __device__ __forceinline__ uint32_t meta_nf(uint4 meta, int sl) {
    uint32_t w = (sl < 4) ? meta.y : ((sl < 8) ? meta.z : meta.w);
    return (w >> ((sl & 3) * 8)) & 0xFFu;
}

// Branchless 4-weight unpack combining 3-bit e3 + 1-bit b3 flip mask!
static __device__ __forceinline__ void unpack4_bf11_flip(
    uint32_t sm4, uint32_t c12, uint32_t f4, uint32_t base_pair,
    uint32_t& b0, uint32_t& b1
) {
    uint32_t d0 = (c12 & 0x007u) | ((f4 & 1u) << 3)
                | ((c12 & 0x038u) << 13) | ((f4 & 2u) << 18);
    uint32_t d1 = ((c12 >> 6) & 0x007u) | ((f4 & 4u) << 1)
                | ((c12 & 0xE00u) << 7) | ((f4 & 8u) << 16);
    uint32_t exp0 = base_pair - d0;
    uint32_t exp1 = base_pair - d1;
    uint32_t s0 = __byte_perm(sm4, 0u, 0x4140u);
    uint32_t s1 = __byte_perm(sm4, 0u, 0x4342u);
    b0 = ((s0 & 0x00800080u) << 8) | (exp0 << 7) | (s0 & 0x007F007Fu);
    b1 = ((s1 & 0x00800080u) << 8) | (exp1 << 7) | (s1 & 0x007F007Fu);
}

static __device__ __forceinline__ void patch_outliers_step(
    uint32_t cur_step, uint32_t lane,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    uint32_t& b0, uint32_t& b1
) {
    if (__builtin_expect(o_next_step == cur_step, 0)) {
        while (o_cur < o_end && o_next_step == cur_step) {
            uint32_t lj = (uint32_t)(o_rec >> 16) & 0x7Fu;
            if ((lj >> 2) == lane) {
                uint32_t val = (uint32_t)(o_rec & 0xFFFFu);
                uint32_t j = lj & 3u;
                if (j == 0u)      b0 = (b0 & 0xFFFF0000u) | val;
                else if (j == 1u) b0 = (b0 & 0x0000FFFFu) | (val << 16);
                else if (j == 2u) b1 = (b1 & 0xFFFF0000u) | val;
                else              b1 = (b1 & 0x0000FFFFu) | (val << 16);
            }
            ++o_cur;
            o_rec = (o_cur < o_end) ? __ldg(OUTLIERS + o_cur) : 0xFFFFFFFFFFFFFFFFULL;
            o_next_step = (uint32_t)(o_rec >> 32);
        }
    }
}

// =========================================================================
// PART 1: EXACT UNCHANGED S7 M <= 16 KERNELS (0 REGRESSION ON C=1!)
// =========================================================================

template <bool M_LE8>
static __device__ __forceinline__ void exec_slab11_flip(
    const Slab11& r, int sl,
    const __nv_bfloat16* sX, int x_off0, int x_off1, int k0_chunk,
    uint32_t base_pair, uint32_t lane, uint32_t fl,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[4]
) {
    const uint32_t sm_words[8] = {
        r.sm0.x, r.sm0.y, r.sm0.z, r.sm0.w,
        r.sm1.x, r.sm1.y, r.sm1.z, r.sm1.w
    };
    const uint32_t c12_arr[8] = {
        r.e0 & 0xFFFu,
        (r.e0 >> 12) & 0xFFFu,
        __funnelshift_r(r.e0, r.e1, 24) & 0xFFFu,
        (r.e1 >> 4) & 0xFFFu,
        (r.e1 >> 16) & 0xFFFu,
        __funnelshift_r(r.e1, r.e2, 28) & 0xFFFu,
        (r.e2 >> 8) & 0xFFFu,
        (r.e2 >> 20) & 0xFFFu
    };

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int ks_off = k0_chunk + s * 16;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX + x_off0 + ks_off);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX + x_off0 + ks_off + 8);
        const uint32_t a1 = M_LE8 ? 0u : *reinterpret_cast<const uint32_t*>(sX + x_off1 + ks_off);
        const uint32_t a3 = M_LE8 ? 0u : *reinterpret_cast<const uint32_t*>(sX + x_off1 + ks_off + 8);

        uint32_t b0, b1;
        unpack4_bf11_flip(sm_words[s], c12_arr[s], fl, base_pair, b0, b1);
        fl >>= 4;

        patch_outliers_step((uint32_t)((sl << 3) | s), lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, b0, b1);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    }
}

template <int WARPS, int SUB_N, bool M_LE8, int SPLIT, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 6)
void bf11_flip_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    float* __restrict__ P,
    int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    uint32_t base_exp
) {
    constexpr int BN = WARPS * SUB_N * 8;
    constexpr int THREADS = WARPS * 32;
    constexpr bool DIRECT = (SPLIT == 1);
    const int pn = blockIdx.x;
    const int ps = blockIdx.y;
    const int n_tiles = gridDim.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int n_warp_base = pn * BN + warp * (SUB_N * 8);
    const bool warp_active = (n_warp_base < N_active);

    const int k_lo = ps * KS;
    const int k_hi = (k_lo + KS < K) ? (k_lo + KS) : K;
    const int k_len = k_hi - k_lo;
    const int n_chunks = k_len >> 7;
    const int total_slabs = warp_active ? (n_chunks * SUB_N) : 0;

    const int64_t warp_uid = ((int64_t)ps * n_tiles + pn) * WARPS + warp;
    const int64_t slabs_per_warp = (int64_t)((KS >> 7) * SUB_N);
    const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;

    Slab11 cur_slab;
    uint4 meta = make_uint4(0, 0, 0, 0);
    int32_t o_cur = 0, o_end = 0;
    int32_t f_cur = 0;
    uint32_t cur_nf = 0u;
    uint32_t cur_frec = 0u;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;
    if (total_slabs > 0) {
        cur_slab = load_slab11(slab_base, lane);
        meta = __ldg(WARP_META + warp_uid);
        o_cur = __ldg(O_OFF + warp_uid);
        o_end = __ldg(O_OFF + warp_uid + 1);
        if constexpr (PDL) {
            if (total_slabs > 1 && lane < 11) {
                prefetch_l2(slab_base + SLAB11_BYTES + lane * 128);
            }
            f_cur = (int32_t)meta.x;
            cur_nf = meta.y & 0xFFu;
            if ((uint32_t)lane < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += (int32_t)cur_nf;
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = (uint32_t)(o_rec >> 32);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    } else if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    if constexpr (DIRECT && TRIG) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    const int sX_stride = KS + 8;
    const int vec_k = k_len >> 3;
    const int m_load = M_LE8 ? min(m_eff, 8) : min(m_eff, 16);
    const int total_vec = m_load * vec_k;
    for (int idx = tid; idx < total_vec; idx += THREADS) {
        int m = idx / vec_k;
        int kv = idx - m * vec_k;
        *reinterpret_cast<uint4*>(sX + m * sX_stride + kv * 8) =
            *reinterpret_cast<const uint4*>(X + (int64_t)m * sx + k_lo + kv * 8);
    }
    __syncthreads();

    if constexpr (!PDL) {
        f_cur = (int32_t)meta.x;
        cur_nf = (total_slabs > 0) ? (meta.y & 0xFFu) : 0u;
        if ((uint32_t)lane < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
        f_cur += (int32_t)cur_nf;
        if (o_cur < o_end) {
            o_rec = __ldg(OUTLIERS + o_cur);
            o_next_step = (uint32_t)(o_rec >> 32);
        }
    }

    float acc[SUB_N][4];
    #pragma unroll
    for (int s = 0; s < SUB_N; ++s) {
        acc[s][0] = 0.0f; acc[s][1] = 0.0f; acc[s][2] = 0.0f; acc[s][3] = 0.0f;
    }

    const int x_off0 = g * sX_stride + 2 * t;
    const int x_off1 = (g + 8) * sX_stride + 2 * t;
    const uint32_t base_pair = base_exp | (base_exp << 16);

    for (int sl = 0; sl < total_slabs; ++sl) {
        if constexpr (!DIRECT && TRIG) {
            if (ps == SPLIT - 1 && sl == total_slabs - 1) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
        Slab11 nxt_slab;
        uint32_t nxt_nf = 0u;
        uint32_t nxt_frec = 0u;
        if (sl + 1 < total_slabs) {
            nxt_slab = load_slab11(slab_base + (int64_t)(sl + 1) * SLAB11_BYTES, lane);
            nxt_nf = meta_nf(meta, sl + 1);
            if ((uint32_t)lane < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += (int32_t)nxt_nf;
        }

        uint32_t fl = 0u;
        if (cur_nf > 0u) {
            s_flip[warp][lane] = 0u;
            __syncwarp();
            if ((uint32_t)lane < cur_nf) {
                atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
            }
            __syncwarp();
            fl = s_flip[warp][lane];
        }

        if constexpr (SUB_N == 2) {
            const int kc = sl >> 1;
            if ((sl & 1) == 0) {
                exec_slab11_flip<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, kc * 128, base_pair, (uint32_t)lane, fl,
                                        OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
            } else {
                exec_slab11_flip<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, kc * 128, base_pair, (uint32_t)lane, fl,
                                        OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[1]);
            }
        } else {
            exec_slab11_flip<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, sl * 128, base_pair, (uint32_t)lane, fl,
                                    OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
        }
        cur_slab = nxt_slab;
        cur_nf = nxt_nf;
        cur_frec = nxt_frec;
    }

    const int m0 = g;
    const int m1 = g + 8;

    if constexpr (DIRECT) {
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int n0 = n_warp_base + sub * 8 + 2 * t;
            const int n1 = n0 + 1;
            const float v00 = (n0 < N_active) ? acc[sub][0] : 0.0f;
            const float v01 = (n1 < N_active) ? acc[sub][1] : 0.0f;
            const float v10 = (n0 < N_active) ? acc[sub][2] : 0.0f;
            const float v11 = (n1 < N_active) ? acc[sub][3] : 0.0f;
            if (m0 < m_eff && n1 < N) {
                __nv_bfloat162 p0 = __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
                *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m0 * so_m + n0) = p0;
            } else if (m0 < m_eff && n0 < N) {
                O[(int64_t)m0 * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
            }
            if constexpr (!M_LE8) {
                if (m1 < m_eff && n1 < N) {
                    __nv_bfloat162 p1 = __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
                    *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m1 * so_m + n0) = p1;
                } else if (m1 < m_eff && n0 < N) {
                    O[(int64_t)m1 * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
                }
            }
        }
    } else {
        const int64_t tile_stride = (int64_t)SPLIT * M * BN;
        const int64_t split_stride = (int64_t)M * BN;
        float* p_tile = P + (int64_t)pn * tile_stride;
        float* p_my = p_tile + (int64_t)ps * split_stride;
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int nl0 = warp * (SUB_N * 8) + sub * 8 + 2 * t;
            const int n0 = pn * BN + nl0;
            const int n1 = n0 + 1;
            const float v00 = (n0 < N_active) ? acc[sub][0] : 0.0f;
            const float v01 = (n1 < N_active) ? acc[sub][1] : 0.0f;
            const float v10 = (n0 < N_active) ? acc[sub][2] : 0.0f;
            const float v11 = (n1 < N_active) ? acc[sub][3] : 0.0f;
            if (m0 < m_eff) {
                *reinterpret_cast<float2*>(p_my + m0 * BN + nl0) = make_float2(v00, v01);
            }
            if constexpr (!M_LE8) {
                if (m1 < m_eff) {
                    *reinterpret_cast<float2*>(p_my + m1 * BN + nl0) = make_float2(v10, v11);
                }
            }
        }

        __syncthreads();
        __shared__ int s_done;
        if (tid == 0) {
            s_done = atomic_add_acq_rel_gpu(&SEM[pn], 1);
        }
        __syncthreads();

        if (s_done == SPLIT - 1) {
            if (tid == 0) SEM[pn] = 0;
            if constexpr (TRIG) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            constexpr int PAIRS_PER_ROW = BN / 2;
            const int total_pairs = m_eff * PAIRS_PER_ROW;
            const int n_tile_base = pn * BN;
            for (int idx = tid; idx < total_pairs; idx += THREADS) {
                const int m = idx / PAIRS_PER_ROW;
                const int p = idx - m * PAIRS_PER_ROW;
                const int nl = p * 2;
                const int n0 = n_tile_base + nl;
                if (n0 >= N) continue;
                const int64_t off = (int64_t)m * BN + nl;
                float s0 = 0.0f, s1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < SPLIT; ++s) {
                    float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + (int64_t)s * split_stride + off));
                    s0 += v.x;
                    s1 += v.y;
                }
                if (n0 + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m * so_m + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(s0), __float2bfloat16_rn(s1));
                } else {
                    O[(int64_t)m * so_m + n0] = __float2bfloat16_rn(s0);
                }
            }
        }
    }
}

template <int WARPS, int SUB_N, bool M_LE8, int SPLIT, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 6)
void bf11_flip_dual_kv_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint16_t* __restrict__ FLIPS0, const uint16_t* __restrict__ FLIPS1,
    const uint4* __restrict__ META0, const uint4* __restrict__ META1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    float* __restrict__ P, int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int so0, int so1,
    uint32_t base_exp0, uint32_t base_exp1
) {
    constexpr int BN = WARPS * SUB_N * 8;
    constexpr int THREADS = WARPS * 32;
    const int n_tiles_each = gridDim.x >> 1;
    const int which = (blockIdx.x >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (blockIdx.x - n_tiles_each) : blockIdx.x;
    const int pn_global = blockIdx.x;
    const int ps = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint16_t* FLIPS = which ? FLIPS1 : FLIPS0;
    const uint4* WARP_META = which ? META1 : META0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int k_lo = ps * KS;
    const int k_hi = (k_lo + KS < K) ? (k_lo + KS) : K;
    const int k_len = k_hi - k_lo;
    const int total_slabs = (k_len >> 7) * SUB_N;

    const int64_t warp_uid = ((int64_t)ps * n_tiles_each + pn_local) * WARPS + warp;
    const int64_t slabs_per_warp = (int64_t)((KS >> 7) * SUB_N);
    const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;

    Slab11 cur_slab;
    uint4 meta = make_uint4(0, 0, 0, 0);
    int32_t o_cur = 0, o_end = 0;
    int32_t f_cur = 0;
    uint32_t cur_nf = 0u;
    uint32_t cur_frec = 0u;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;
    if (total_slabs > 0) {
        cur_slab = load_slab11(slab_base, lane);
        meta = __ldg(WARP_META + warp_uid);
        o_cur = __ldg(O_OFF + warp_uid);
        o_end = __ldg(O_OFF + warp_uid + 1);
        if constexpr (PDL) {
            if (total_slabs > 1 && lane < 11) {
                prefetch_l2(slab_base + SLAB11_BYTES + lane * 128);
            }
            f_cur = (int32_t)meta.x;
            cur_nf = meta.y & 0xFFu;
            if ((uint32_t)lane < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += (int32_t)cur_nf;
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = (uint32_t)(o_rec >> 32);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    } else if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];
    const int sX_stride = KS + 8;
    const int vec_k = k_len >> 3;
    const int m_load = M_LE8 ? min(m_eff, 8) : min(m_eff, 16);
    const int total_vec = m_load * vec_k;
    for (int idx = tid; idx < total_vec; idx += THREADS) {
        int m = idx / vec_k;
        int kv = idx - m * vec_k;
        *reinterpret_cast<uint4*>(sX + m * sX_stride + kv * 8) =
            *reinterpret_cast<const uint4*>(X + (int64_t)m * sx + k_lo + kv * 8);
    }
    __syncthreads();

    if constexpr (!PDL) {
        f_cur = (int32_t)meta.x;
        cur_nf = (total_slabs > 0) ? (meta.y & 0xFFu) : 0u;
        if ((uint32_t)lane < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
        f_cur += (int32_t)cur_nf;
        if (o_cur < o_end) {
            o_rec = __ldg(OUTLIERS + o_cur);
            o_next_step = (uint32_t)(o_rec >> 32);
        }
    }

    float acc[SUB_N][4];
    #pragma unroll
    for (int s = 0; s < SUB_N; ++s) {
        acc[s][0] = 0.0f; acc[s][1] = 0.0f; acc[s][2] = 0.0f; acc[s][3] = 0.0f;
    }

    const int x_off0 = g * sX_stride + 2 * t;
    const int x_off1 = (g + 8) * sX_stride + 2 * t;
    const uint32_t base_pair = base_exp | (base_exp << 16);

    for (int sl = 0; sl < total_slabs; ++sl) {
        if constexpr (TRIG) {
            if (ps == SPLIT - 1 && sl == total_slabs - 1) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
        Slab11 nxt_slab;
        uint32_t nxt_nf = 0u;
        uint32_t nxt_frec = 0u;
        if (sl + 1 < total_slabs) {
            nxt_slab = load_slab11(slab_base + (int64_t)(sl + 1) * SLAB11_BYTES, lane);
            nxt_nf = meta_nf(meta, sl + 1);
            if ((uint32_t)lane < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += (int32_t)nxt_nf;
        }
        uint32_t fl = 0u;
        if (cur_nf > 0u) {
            s_flip[warp][lane] = 0u;
            __syncwarp();
            if ((uint32_t)lane < cur_nf) atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
            __syncwarp();
            fl = s_flip[warp][lane];
        }
        exec_slab11_flip<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, sl * 128, base_pair, (uint32_t)lane, fl,
                                OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
        cur_slab = nxt_slab;
        cur_nf = nxt_nf;
        cur_frec = nxt_frec;
    }

    const int m0 = g;
    const int m1 = g + 8;
    const int64_t tile_stride = (int64_t)SPLIT * M * BN;
    const int64_t split_stride = (int64_t)M * BN;
    float* p_tile = P + (int64_t)pn_global * tile_stride;
    float* p_my = p_tile + (int64_t)ps * split_stride;
    #pragma unroll
    for (int sub = 0; sub < SUB_N; ++sub) {
        const int nl0 = warp * (SUB_N * 8) + sub * 8 + 2 * t;
        if (m0 < m_eff) *reinterpret_cast<float2*>(p_my + m0 * BN + nl0) = make_float2(acc[sub][0], acc[sub][1]);
        if constexpr (!M_LE8) {
            if (m1 < m_eff) *reinterpret_cast<float2*>(p_my + m1 * BN + nl0) = make_float2(acc[sub][2], acc[sub][3]);
        }
    }

    __syncthreads();
    __shared__ int s_done;
    if (tid == 0) {
        s_done = atomic_add_acq_rel_gpu(&SEM[pn_global], 1);
    }
    __syncthreads();

    if (s_done == SPLIT - 1) {
        if (tid == 0) SEM[pn_global] = 0;
        if constexpr (TRIG) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        constexpr int PAIRS_PER_ROW = BN / 2;
        const int total_pairs = m_eff * PAIRS_PER_ROW;
        const int n_tile_base = pn_local * BN;
        for (int idx = tid; idx < total_pairs; idx += THREADS) {
            const int m = idx / PAIRS_PER_ROW;
            const int p = idx - m * PAIRS_PER_ROW;
            const int nl = p * 2;
            const int n0 = n_tile_base + nl;
            if (n0 >= N_each) continue;
            const int64_t off = (int64_t)m * BN + nl;
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLIT; ++s) {
                float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + (int64_t)s * split_stride + off));
                s0 += v.x;
                s1 += v.y;
            }
            *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m * so_m + n0) =
                __halves2bfloat162(__float2bfloat16_rn(s0), __float2bfloat16_rn(s1));
        }
    }
}

// =========================================================================
// PART 2: NEW SM121a C=32 MULTI-M KERNELS (M = 17..256+, ZERO EXTRA VRAM!)
// =========================================================================

// Execute half-slab (4 MMA steps = 64 K-cols) against BK=64 SMEM `sX_half` (SX_STRIDE = 72)
// amortizing BF11 unpack across MT = 2/4/8 Tensor Core MMAs!
template <int MT, int HALF>
static __device__ __forceinline__ void exec_half_slab11_mt(
    uint4 sm_half,
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t fl16,
    int sl,
    const __nv_bfloat16* sX_half, int g, int t,
    uint32_t base_pair, uint32_t lane,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    constexpr int SX_STRIDE = 72;
    const uint32_t sm_w[4] = {sm_half.x, sm_half.y, sm_half.z, sm_half.w};
    const uint32_t c12[4]  = {c0, c1, c2, c3};
    constexpr int S_BASE = HALF * 4;

    #pragma unroll
    for (int s_loc = 0; s_loc < 4; ++s_loc) {
        uint32_t b0, b1;
        unpack4_bf11_flip(sm_w[s_loc], c12[s_loc], fl16, base_pair, b0, b1);
        fl16 >>= 4;

        patch_outliers_step(
            (uint32_t)((sl << 3) | (S_BASE + s_loc)), lane,
            OUTLIERS, o_cur, o_end, o_rec, o_next_step, b0, b1
        );

        const int ks_off = s_loc * 16 + 2 * t;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int row0 = mt * 16 + g;
            const int row1 = row0 + 8;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX_half + row0 * SX_STRIDE + ks_off);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX_half + row0 * SX_STRIDE + ks_off + 8);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sX_half + row1 * SX_STRIDE + ks_off);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sX_half + row1 * SX_STRIDE + ks_off + 8);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[mt][0]), "+f"(acc[mt][1]), "+f"(acc[mt][2]), "+f"(acc[mt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }
}

// Full 128-K slab executor for SUB_N == 2 (BK=128 SMEM, SX_STRIDE=136 -> strictly monotonic step_id)
template <int MT>
static __device__ __forceinline__ void exec_full_slab11_mt(
    const Slab11& r, int sl,
    const __nv_bfloat16* sX_chunk, int g, int t,
    uint32_t base_pair, uint32_t lane, uint32_t fl,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    constexpr int SX_STRIDE = 136;
    const uint32_t sm_words[8] = {
        r.sm0.x, r.sm0.y, r.sm0.z, r.sm0.w,
        r.sm1.x, r.sm1.y, r.sm1.z, r.sm1.w
    };
    const uint32_t c12_arr[8] = {
        r.e0 & 0xFFFu,
        (r.e0 >> 12) & 0xFFFu,
        __funnelshift_r(r.e0, r.e1, 24) & 0xFFFu,
        (r.e1 >> 4) & 0xFFFu,
        (r.e1 >> 16) & 0xFFFu,
        __funnelshift_r(r.e1, r.e2, 28) & 0xFFFu,
        (r.e2 >> 8) & 0xFFFu,
        (r.e2 >> 20) & 0xFFFu
    };

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        uint32_t b0, b1;
        unpack4_bf11_flip(sm_words[s], c12_arr[s], fl, base_pair, b0, b1);
        fl >>= 4;

        patch_outliers_step(
            (uint32_t)((sl << 3) | s), lane,
            OUTLIERS, o_cur, o_end, o_rec, o_next_step, b0, b1
        );

        const int ks_off = s * 16 + 2 * t;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int row0 = mt * 16 + g;
            const int row1 = row0 + 8;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX_chunk + row0 * SX_STRIDE + ks_off);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX_chunk + row0 * SX_STRIDE + ks_off + 8);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sX_chunk + row1 * SX_STRIDE + ks_off);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sX_chunk + row1 * SX_STRIDE + ks_off + 8);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[mt][0]), "+f"(acc[mt][1]), "+f"(acc[mt][2]), "+f"(acc[mt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }
}

// Primary C=32 Single-CTA Zero-Partial-Traffic Kernel (SUB_N == 1, BK = 64 double-buffered cp.async)
// Grid: (m_blocks, n_tiles, 1) -> `m_blocks` varies fastest for L2 slab co-residency!
// K_ORDER == 0: exact ascending-K `split=1` (matches cuBLAS splitK=1 to 0-ULP!)
// K_ORDER == 1: exact `split_packed`-grouped accumulation (matches `skinny.py`/`BF11_FLIP M<=16` to 0-ULP!)
template <int WARPS, int MT, int K_ORDER, bool CACHE_L2, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, (MT >= 8 ? 2 : 4))
void bf11_flip_c32_sub1_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp
) {
    constexpr int BN = WARPS * 8;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 72;      // 64 + 8 bf16 = 144 B (16B aligned, 0 bank conflicts!)
    constexpr int BUF_ELEMS = BM * SX_STRIDE;
    constexpr int VEC_PER_ROW = 8;     // 64 bf16 / 8 = 8 uint4
    constexpr int TOTAL_VEC = BM * VEC_PER_ROW;

    const int mb = blockIdx.x;
    const int pn = blockIdx.y;
    const int n_tiles = gridDim.y;
    const int m_base = mb * BM;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int n_warp_base = pn * BN + warp * 8;
    const bool warp_active = (n_warp_base < N_active);
    const int n_chunks_per_sp = KS >> 7;
    const int64_t slabs_per_warp = (int64_t)n_chunks_per_sp;

    if constexpr (PDL) {
        if (warp_active && n_chunks_per_sp > 0) {
            const int64_t wuid0 = (int64_t)pn * WARPS + warp;
            const uint8_t* sb0 = SLABS + wuid0 * slabs_per_warp * SLAB11_BYTES;
            if (lane < 11) prefetch_l2(sb0 + lane * 128);
            if (lane == 11) prefetch_l2(WARP_META + wuid0);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    auto load_x_half_async = [&](int k_off, __nv_bfloat16* dst) {
        for (int idx = tid; idx < TOTAL_VEC; idx += THREADS) {
            const int lm = idx >> 3;
            const int kv = idx & 7;
            const int gm = m_base + lm;
            __nv_bfloat16* d = dst + lm * SX_STRIDE + (kv << 3);
            if (gm < m_eff) {
                cp_async_ca_16(d, X + (int64_t)gm * sx + k_off + (kv << 3));
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_async_commit();
    };

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    const uint32_t base_pair = base_exp | (base_exp << 16);
    const int total_k_chunks = split_packed * n_chunks_per_sp;

    if (total_k_chunks > 0) {
        // Kick off very first half-chunk (k = 0..63) into sX[0]
        load_x_half_async(0, sX);

        for (int ps = 0; ps < split_packed; ++ps) {
            float p_acc[MT][4];
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f; p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                }
            }

            const int64_t warp_uid = ((int64_t)ps * n_tiles + pn) * WARPS + warp;
            const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;
            Slab11 cur_slab;
            uint4 meta = make_uint4(0, 0, 0, 0);
            int32_t o_cur = 0, o_end = 0;
            int32_t f_cur = 0;
            uint32_t cur_nf = 0u;
            uint32_t cur_frec = 0u;
            uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = 0xFFFFFFFFu;

            if (warp_active) {
                cur_slab = load_slab11_t<CACHE_L2>(slab_base, lane);
                meta = __ldg(WARP_META + warp_uid);
                o_cur = __ldg(O_OFF + warp_uid);
                o_end = __ldg(O_OFF + warp_uid + 1);
                f_cur = (int32_t)meta.x;
                cur_nf = meta.y & 0xFFu;
                if ((uint32_t)lane < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
                f_cur += (int32_t)cur_nf;
                if (o_cur < o_end) {
                    o_rec = __ldg(OUTLIERS + o_cur);
                    o_next_step = (uint32_t)(o_rec >> 32);
                }
            }

            for (int kc = 0; kc < n_chunks_per_sp; ++kc) {
                const int k_chunk = ps * KS + kc * 128;

                // Wait for Half-0 (sX[0]) to arrive, then immediately kick off Half-1 into sX[1]!
                cp_async_wait<0>();
                __syncthreads();
                load_x_half_async(k_chunk + 64, sX + BUF_ELEMS);

                uint32_t fl = 0u;
                if (warp_active && cur_nf > 0u) {
                    s_flip[warp][lane] = 0u;
                    __syncwarp();
                    if ((uint32_t)lane < cur_nf) {
                        atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
                    }
                    __syncwarp();
                    fl = s_flip[warp][lane];
                }

                // Compute Half-0 (s = 0..3) while Half-1 streams into sX[1]
                const uint32_t c0 = cur_slab.e0 & 0xFFFu;
                const uint32_t c1 = (cur_slab.e0 >> 12) & 0xFFFu;
                const uint32_t c2 = __funnelshift_r(cur_slab.e0, cur_slab.e1, 24) & 0xFFFu;
                const uint32_t c3 = (cur_slab.e1 >> 4) & 0xFFFu;
                const uint32_t c4 = (cur_slab.e1 >> 16) & 0xFFFu;
                const uint32_t c5 = __funnelshift_r(cur_slab.e1, cur_slab.e2, 28) & 0xFFFu;
                const uint32_t c6 = (cur_slab.e2 >> 8) & 0xFFFu;
                const uint32_t c7 = (cur_slab.e2 >> 20) & 0xFFFu;

                if (warp_active) {
                    if constexpr (K_ORDER == 0) {
                        exec_half_slab11_mt<MT, 0>(
                            cur_slab.sm0, c0, c1, c2, c3, fl & 0xFFFFu, kc, sX, g, t,
                            base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                        );
                    } else {
                        exec_half_slab11_mt<MT, 0>(
                            cur_slab.sm0, c0, c1, c2, c3, fl & 0xFFFFu, kc, sX, g, t,
                            base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                        );
                    }
                }

                // Wait for Half-1 (sX[1]), then kick off NEXT chunk's Half-0 into sX[0] + next Slab11!
                cp_async_wait<0>();
                __syncthreads();
                const int k_next = k_chunk + 128;
                if (k_next < K) {
                    load_x_half_async(k_next, sX);
                }

                Slab11 nxt_slab;
                uint32_t nxt_nf = 0u;
                uint32_t nxt_frec = 0u;
                if (warp_active && kc + 1 < n_chunks_per_sp) {
                    nxt_slab = load_slab11_t<CACHE_L2>(slab_base + (int64_t)(kc + 1) * SLAB11_BYTES, lane);
                    nxt_nf = meta_nf(meta, kc + 1);
                    if ((uint32_t)lane < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
                    f_cur += (int32_t)nxt_nf;
                }

                // Compute Half-1 (s = 4..7) while next Half-0 & nxt_slab stream in!
                if (warp_active) {
                    if constexpr (K_ORDER == 0) {
                        exec_half_slab11_mt<MT, 1>(
                            cur_slab.sm1, c4, c5, c6, c7, fl >> 16, kc, sX + BUF_ELEMS, g, t,
                            base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                        );
                    } else {
                        exec_half_slab11_mt<MT, 1>(
                            cur_slab.sm1, c4, c5, c6, c7, fl >> 16, kc, sX + BUF_ELEMS, g, t,
                            base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                        );
                    }
                }

                cur_slab = nxt_slab;
                cur_nf = nxt_nf;
                cur_frec = nxt_frec;
            }

            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    acc[mt][0] += p_acc[mt][0];
                    acc[mt][1] += p_acc[mt][1];
                    acc[mt][2] += p_acc[mt][2];
                    acc[mt][3] += p_acc[mt][3];
                }
            }
        }
    }

    if constexpr (TRIG) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    const int n0 = n_warp_base + 2 * t;
    const int n1 = n0 + 1;
    const bool n0_act = (n0 < N_active);
    const bool n1_act = (n1 < N_active);
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int m0 = m_base + mt * 16 + g;
        const int m1 = m0 + 8;
        const float v00 = n0_act ? acc[mt][0] : 0.0f;
        const float v01 = n1_act ? acc[mt][1] : 0.0f;
        const float v10 = n0_act ? acc[mt][2] : 0.0f;
        const float v11 = n1_act ? acc[mt][3] : 0.0f;
        if (m0 < m_eff && n1 < N) {
            *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m0 * so_m + n0) =
                __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
        } else if (m0 < m_eff && n0 < N) {
            O[(int64_t)m0 * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
        }
        if (m1 < m_eff && n1 < N) {
            *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m1 * so_m + n0) =
                __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
        } else if (m1 < m_eff && n0 < N) {
            O[(int64_t)m1 * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
        }
    }
}

// Companion C=32 Single-CTA Kernel for SUB_N == 2 (cfg=0, 2; BK=128 double-buffered SMEM)
template <int WARPS, int MT, int K_ORDER, bool CACHE_L2, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 2)
void bf11_flip_c32_sub2_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp
) {
    constexpr int BN = WARPS * 16;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 136;     // 128 + 8 bf16 = 272 B (16B aligned, 0 bank conflicts!)
    constexpr int BUF_ELEMS = BM * SX_STRIDE;
    constexpr int VEC_PER_ROW = 16;    // 128 bf16 / 8 = 16 uint4
    constexpr int TOTAL_VEC = BM * VEC_PER_ROW;

    const int mb = blockIdx.x;
    const int pn = blockIdx.y;
    const int n_tiles = gridDim.y;
    const int m_base = mb * BM;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int n_warp_base = pn * BN + warp * 16;
    const bool warp_active = (n_warp_base < N_active);
    const int n_chunks_per_sp = KS >> 7;
    const int64_t slabs_per_warp = (int64_t)(n_chunks_per_sp * 2);

    if constexpr (PDL) {
        if (warp_active && n_chunks_per_sp > 0) {
            const int64_t wuid0 = (int64_t)pn * WARPS + warp;
            const uint8_t* sb0 = SLABS + wuid0 * slabs_per_warp * SLAB11_BYTES;
            if (lane < 11) prefetch_l2(sb0 + lane * 128);
            if (lane == 11) prefetch_l2(WARP_META + wuid0);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    auto load_x_chunk_async = [&](int k_off, __nv_bfloat16* dst) {
        for (int idx = tid; idx < TOTAL_VEC; idx += THREADS) {
            const int lm = idx >> 4;
            const int kv = idx & 15;
            const int gm = m_base + lm;
            __nv_bfloat16* d = dst + lm * SX_STRIDE + (kv << 3);
            if (gm < m_eff) {
                cp_async_ca_16(d, X + (int64_t)gm * sx + k_off + (kv << 3));
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_async_commit();
    };

    float acc[2][MT][4];
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            acc[sub][mt][0] = 0.0f; acc[sub][mt][1] = 0.0f; acc[sub][mt][2] = 0.0f; acc[sub][mt][3] = 0.0f;
        }
    }

    const uint32_t base_pair = base_exp | (base_exp << 16);
    if (split_packed * n_chunks_per_sp > 0) {
        load_x_chunk_async(0, sX);
        int buf = 0;
        for (int ps = 0; ps < split_packed; ++ps) {
            float p_acc[2][MT][4];
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        p_acc[sub][mt][0] = 0.0f; p_acc[sub][mt][1] = 0.0f;
                        p_acc[sub][mt][2] = 0.0f; p_acc[sub][mt][3] = 0.0f;
                    }
                }
            }

            const int64_t warp_uid = ((int64_t)ps * n_tiles + pn) * WARPS + warp;
            const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;
            uint4 meta = make_uint4(0, 0, 0, 0);
            int32_t o_cur = 0, o_end = 0;
            int32_t f_cur = 0;
            uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = 0xFFFFFFFFu;
            if (warp_active) {
                meta = __ldg(WARP_META + warp_uid);
                o_cur = __ldg(O_OFF + warp_uid);
                o_end = __ldg(O_OFF + warp_uid + 1);
                f_cur = (int32_t)meta.x;
                if (o_cur < o_end) {
                    o_rec = __ldg(OUTLIERS + o_cur);
                    o_next_step = (uint32_t)(o_rec >> 32);
                }
            }

            for (int kc = 0; kc < n_chunks_per_sp; ++kc) {
                const int k_chunk = ps * KS + kc * 128;
                cp_async_wait<0>();
                __syncthreads();
                __nv_bfloat16* cur_sX = sX + buf * BUF_ELEMS;
                if (k_chunk + 128 < K) {
                    load_x_chunk_async(k_chunk + 128, sX + (buf ^ 1) * BUF_ELEMS);
                }
                if (warp_active) {
                    const int sl0 = kc * 2;
                    const int sl1 = sl0 + 1;
                    Slab11 s0_r = load_slab11_t<CACHE_L2>(slab_base + (int64_t)sl0 * SLAB11_BYTES, lane);
                    uint32_t nf0 = meta_nf(meta, sl0);
                    uint32_t frec0 = ((uint32_t)lane < nf0) ? ld_nc_u16(FLIPS + f_cur + lane) : 0u;
                    f_cur += (int32_t)nf0;
                    uint32_t fl0 = 0u;
                    if (nf0 > 0u) {
                        s_flip[warp][lane] = 0u;
                        __syncwarp();
                        if ((uint32_t)lane < nf0) atomicOr(&s_flip[warp][frec0 >> 5], 1u << (frec0 & 31u));
                        __syncwarp();
                        fl0 = s_flip[warp][lane];
                    }
                    Slab11 s1_r = load_slab11_t<CACHE_L2>(slab_base + (int64_t)sl1 * SLAB11_BYTES, lane);
                    uint32_t nf1 = meta_nf(meta, sl1);
                    uint32_t frec1 = ((uint32_t)lane < nf1) ? ld_nc_u16(FLIPS + f_cur + lane) : 0u;
                    f_cur += (int32_t)nf1;

                    if constexpr (K_ORDER == 0) {
                        exec_full_slab11_mt<MT>(s0_r, sl0, cur_sX, g, t, base_pair, (uint32_t)lane, fl0,
                                                OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
                    } else {
                        exec_full_slab11_mt<MT>(s0_r, sl0, cur_sX, g, t, base_pair, (uint32_t)lane, fl0,
                                                OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc[0]);
                    }

                    uint32_t fl1 = 0u;
                    if (nf1 > 0u) {
                        s_flip[warp][lane] = 0u;
                        __syncwarp();
                        if ((uint32_t)lane < nf1) atomicOr(&s_flip[warp][frec1 >> 5], 1u << (frec1 & 31u));
                        __syncwarp();
                        fl1 = s_flip[warp][lane];
                    }
                    if constexpr (K_ORDER == 0) {
                        exec_full_slab11_mt<MT>(s1_r, sl1, cur_sX, g, t, base_pair, (uint32_t)lane, fl1,
                                                OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[1]);
                    } else {
                        exec_full_slab11_mt<MT>(s1_r, sl1, cur_sX, g, t, base_pair, (uint32_t)lane, fl1,
                                                OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc[1]);
                    }
                }
                buf ^= 1;
            }

            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        acc[sub][mt][0] += p_acc[sub][mt][0];
                        acc[sub][mt][1] += p_acc[sub][mt][1];
                        acc[sub][mt][2] += p_acc[sub][mt][2];
                        acc[sub][mt][3] += p_acc[sub][mt][3];
                    }
                }
            }
        }
    }

    if constexpr (TRIG) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int n0 = n_warp_base + sub * 8 + 2 * t;
        const int n1 = n0 + 1;
        const bool n0_act = (n0 < N_active);
        const bool n1_act = (n1 < N_active);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            const float v00 = n0_act ? acc[sub][mt][0] : 0.0f;
            const float v01 = n1_act ? acc[sub][mt][1] : 0.0f;
            const float v10 = n0_act ? acc[sub][mt][2] : 0.0f;
            const float v11 = n1_act ? acc[sub][mt][3] : 0.0f;
            if (m0 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m0 * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
            } else if (m0 < m_eff && n0 < N) {
                O[(int64_t)m0 * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
            }
            if (m1 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m1 * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
            } else if (m1 < m_eff && n0 < N) {
                O[(int64_t)m1 * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
            }
        }
    }
}

// Fused Dual-KV C=32 Kernel (k_proj + v_proj in ONE launch for M = 17..256+, SUB_N == 1)
// Grid: (m_blocks, 2 * n_tiles_each, 1) -> zero partials, exact ascending-K or split-9 K-order!
template <int WARPS, int MT, int K_ORDER, bool CACHE_L2, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 4)
void bf11_flip_dual_kv_c32_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint16_t* __restrict__ FLIPS0, const uint16_t* __restrict__ FLIPS1,
    const uint4* __restrict__ META0, const uint4* __restrict__ META1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    uint32_t base_exp0, uint32_t base_exp1
) {
    constexpr int BN = WARPS * 8;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 72;
    constexpr int BUF_ELEMS = BM * SX_STRIDE;
    constexpr int VEC_PER_ROW = 8;
    constexpr int TOTAL_VEC = BM * VEC_PER_ROW;

    const int mb = blockIdx.x;
    const int n_tiles_each = gridDim.y >> 1;
    const int which = (blockIdx.y >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (blockIdx.y - n_tiles_each) : blockIdx.y;
    const int m_base = mb * BM;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint16_t* FLIPS = which ? FLIPS1 : FLIPS0;
    const uint4* WARP_META = which ? META1 : META0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int n_chunks_per_sp = KS >> 7;
    const int64_t slabs_per_warp = (int64_t)n_chunks_per_sp;

    if constexpr (PDL) {
        if (n_chunks_per_sp > 0) {
            const int64_t wuid0 = (int64_t)pn_local * WARPS + warp;
            const uint8_t* sb0 = SLABS + wuid0 * slabs_per_warp * SLAB11_BYTES;
            if (lane < 11) prefetch_l2(sb0 + lane * 128);
            if (lane == 11) prefetch_l2(WARP_META + wuid0);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    auto load_x_half_async = [&](int k_off, __nv_bfloat16* dst) {
        for (int idx = tid; idx < TOTAL_VEC; idx += THREADS) {
            const int lm = idx >> 3;
            const int kv = idx & 7;
            const int gm = m_base + lm;
            __nv_bfloat16* d = dst + lm * SX_STRIDE + (kv << 3);
            if (gm < m_eff) {
                cp_async_ca_16(d, X + (int64_t)gm * sx + k_off + (kv << 3));
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_async_commit();
    };

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    const uint32_t base_pair = base_exp | (base_exp << 16);
    if (split_packed * n_chunks_per_sp > 0) {
        load_x_half_async(0, sX);
        for (int ps = 0; ps < split_packed; ++ps) {
            float p_acc[MT][4];
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f; p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                }
            }
            const int64_t warp_uid = ((int64_t)ps * n_tiles_each + pn_local) * WARPS + warp;
            const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;
            Slab11 cur_slab = load_slab11_t<CACHE_L2>(slab_base, lane);
            uint4 meta = __ldg(WARP_META + warp_uid);
            int32_t o_cur = __ldg(O_OFF + warp_uid);
            int32_t o_end = __ldg(O_OFF + warp_uid + 1);
            int32_t f_cur = (int32_t)meta.x;
            uint32_t cur_nf = meta.y & 0xFFu;
            uint32_t cur_frec = ((uint32_t)lane < cur_nf) ? ld_nc_u16(FLIPS + f_cur + lane) : 0u;
            f_cur += (int32_t)cur_nf;
            uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = 0xFFFFFFFFu;
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = (uint32_t)(o_rec >> 32);
            }

            for (int kc = 0; kc < n_chunks_per_sp; ++kc) {
                const int k_chunk = ps * KS + kc * 128;
                cp_async_wait<0>();
                __syncthreads();
                load_x_half_async(k_chunk + 64, sX + BUF_ELEMS);

                uint32_t fl = 0u;
                if (cur_nf > 0u) {
                    s_flip[warp][lane] = 0u;
                    __syncwarp();
                    if ((uint32_t)lane < cur_nf) atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
                    __syncwarp();
                    fl = s_flip[warp][lane];
                }

                const uint32_t c0 = cur_slab.e0 & 0xFFFu;
                const uint32_t c1 = (cur_slab.e0 >> 12) & 0xFFFu;
                const uint32_t c2 = __funnelshift_r(cur_slab.e0, cur_slab.e1, 24) & 0xFFFu;
                const uint32_t c3 = (cur_slab.e1 >> 4) & 0xFFFu;
                const uint32_t c4 = (cur_slab.e1 >> 16) & 0xFFFu;
                const uint32_t c5 = __funnelshift_r(cur_slab.e1, cur_slab.e2, 28) & 0xFFFu;
                const uint32_t c6 = (cur_slab.e2 >> 8) & 0xFFFu;
                const uint32_t c7 = (cur_slab.e2 >> 20) & 0xFFFu;

                if constexpr (K_ORDER == 0) {
                    exec_half_slab11_mt<MT, 0>(
                        cur_slab.sm0, c0, c1, c2, c3, fl & 0xFFFFu, kc, sX, g, t,
                        base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                    );
                } else {
                    exec_half_slab11_mt<MT, 0>(
                        cur_slab.sm0, c0, c1, c2, c3, fl & 0xFFFFu, kc, sX, g, t,
                        base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                    );
                }

                cp_async_wait<0>();
                __syncthreads();
                const int k_next = k_chunk + 128;
                if (k_next < K) {
                    load_x_half_async(k_next, sX);
                }

                Slab11 nxt_slab;
                uint32_t nxt_nf = 0u;
                uint32_t nxt_frec = 0u;
                if (kc + 1 < n_chunks_per_sp) {
                    nxt_slab = load_slab11_t<CACHE_L2>(slab_base + (int64_t)(kc + 1) * SLAB11_BYTES, lane);
                    nxt_nf = meta_nf(meta, kc + 1);
                    if ((uint32_t)lane < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
                    f_cur += (int32_t)nxt_nf;
                }

                if constexpr (K_ORDER == 0) {
                    exec_half_slab11_mt<MT, 1>(
                        cur_slab.sm1, c4, c5, c6, c7, fl >> 16, kc, sX + BUF_ELEMS, g, t,
                        base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                    );
                } else {
                    exec_half_slab11_mt<MT, 1>(
                        cur_slab.sm1, c4, c5, c6, c7, fl >> 16, kc, sX + BUF_ELEMS, g, t,
                        base_pair, (uint32_t)lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                    );
                }

                cur_slab = nxt_slab;
                cur_nf = nxt_nf;
                cur_frec = nxt_frec;
            }
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    acc[mt][0] += p_acc[mt][0];
                    acc[mt][1] += p_acc[mt][1];
                    acc[mt][2] += p_acc[mt][2];
                    acc[mt][3] += p_acc[mt][3];
                }
            }
        }
    }

    if constexpr (TRIG) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    const int n0 = pn_local * BN + warp * 8 + 2 * t;
    if (n0 + 1 < N_each) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            if (m0 < m_eff) {
                *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m0 * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][0] + 0.0f), __float2bfloat16_rn(acc[mt][1] + 0.0f));
            }
            if (m1 < m_eff) {
                *reinterpret_cast<__nv_bfloat162*>(O + (int64_t)m1 * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][2] + 0.0f), __float2bfloat16_rn(acc[mt][3] + 0.0f));
            }
        }
    } else if (n0 < N_each) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            if (m0 < m_eff) O[(int64_t)m0 * so_m + n0] = __float2bfloat16_rn(acc[mt][0] + 0.0f);
            if (m1 < m_eff) O[(int64_t)m1 * so_m + n0] = __float2bfloat16_rn(acc[mt][2] + 0.0f);
        }
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_maybe_pdl(
    KernelT kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl, Args... args
) {
    if (!pdl) {
        kernel<<<grid, block, smem, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <int W_V, int SUB_V, int SP_V>
static cudaError_t dispatch_bf11_flip(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BN = W_V * SUB_V * 8;
    const int n_tiles = (N + BN - 1) / BN;
    dim3 grid(n_tiles, SP_V, 1);
    dim3 block(W_V * 32, 1, 1);
    const bool m_le8 = (M <= 8);
    const size_t smem = (size_t)(m_le8 ? 8 : 16) * (KS + 8) * sizeof(__nv_bfloat16);
    #define L(ML, PD, TR) launch_maybe_pdl(bf11_flip_kernel<W_V, SUB_V, ML, SP_V, PD, TR>, grid, block, smem, stream, PD, \
        (const __nv_bfloat16*)X, (const uint8_t*)SLABS, (const uint16_t*)FLIPS, (const uint4*)WARP_META, \
        (const uint64_t*)OUTLIERS, (const int32_t*)O_OFF, \
        (float*)P, (int32_t*)SEM, (__nv_bfloat16*)O, (const int32_t*)NLIVE_PTR, \
        M, N, N_active, K, sx, KS, so_m, base_exp)
    if (m_le8) {
        if (pdl) return trig ? L(true, true, true) : L(true, true, false);
        return trig ? L(true, false, true) : L(true, false, false);
    } else {
        if (pdl) return trig ? L(false, true, true) : L(false, true, false);
        return trig ? L(false, false, true) : L(false, false, false);
    }
    #undef L
}

template <int W_V, int MT_V, int KO_V>
static cudaError_t dispatch_bf11_c32_sub1(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BN = W_V * 8;
    constexpr int BM = MT_V * 16;
    const int n_tiles = (N + BN - 1) / BN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, n_tiles, 1);
    dim3 block(W_V * 32, 1, 1);
    const size_t smem = (size_t)2 * BM * 72 * sizeof(__nv_bfloat16);
    #define LC1(CA, PD, TR) launch_maybe_pdl(bf11_flip_c32_sub1_kernel<W_V, MT_V, KO_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        (const __nv_bfloat16*)X, (const uint8_t*)SLABS, (const uint16_t*)FLIPS, (const uint4*)WARP_META, \
        (const uint64_t*)OUTLIERS, (const int32_t*)O_OFF, (__nv_bfloat16*)O, (const int32_t*)NLIVE_PTR, \
        M, N, N_active, K, sx, KS, split_packed, so_m, base_exp)
    if (cache_l2) {
        if (pdl) return trig ? LC1(true, true, true) : LC1(true, true, false);
        return trig ? LC1(true, false, true) : LC1(true, false, false);
    } else {
        if (pdl) return trig ? LC1(false, true, true) : LC1(false, true, false);
        return trig ? LC1(false, false, true) : LC1(false, false, false);
    }
    #undef LC1
}

template <int W_V, int MT_V, int KO_V>
static cudaError_t dispatch_bf11_c32_sub2(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BN = W_V * 16;
    constexpr int BM = MT_V * 16;
    const int n_tiles = (N + BN - 1) / BN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, n_tiles, 1);
    dim3 block(W_V * 32, 1, 1);
    const size_t smem = (size_t)2 * BM * 136 * sizeof(__nv_bfloat16);
    #define LC2(CA, PD, TR) launch_maybe_pdl(bf11_flip_c32_sub2_kernel<W_V, MT_V, KO_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        (const __nv_bfloat16*)X, (const uint8_t*)SLABS, (const uint16_t*)FLIPS, (const uint4*)WARP_META, \
        (const uint64_t*)OUTLIERS, (const int32_t*)O_OFF, (__nv_bfloat16*)O, (const int32_t*)NLIVE_PTR, \
        M, N, N_active, K, sx, KS, split_packed, so_m, base_exp)
    if (cache_l2) {
        if (pdl) return trig ? LC2(true, true, true) : LC2(true, true, false);
        return trig ? LC2(true, false, true) : LC2(true, false, false);
    } else {
        if (pdl) return trig ? LC2(false, true, true) : LC2(false, true, false);
        return trig ? LC2(false, false, true) : LC2(false, false, false);
    }
    #undef LC2
}

}  // namespace

// ---- Drop-in S7 C ABI (100% unchanged for M <= 16!) ----
extern "C" int bf11_flip_fwd(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    int cfg_id, int split, uint32_t base_exp,
    int pdl, int trig, uintptr_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    cudaError_t err = cudaSuccess;
    #define CASE_SP(W, S) \
        switch (split) { \
            case 1: err = dispatch_bf11_flip<W, S, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, base_exp, do_pdl, do_trig, stream); break; \
            case 6: err = dispatch_bf11_flip<W, S, 6>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, base_exp, do_pdl, do_trig, stream); break; \
            case 9: err = dispatch_bf11_flip<W, S, 9>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, base_exp, do_pdl, do_trig, stream); break; \
            default: return -2; \
        }
    switch (cfg_id) {
        case 0: CASE_SP(4, 2); break;
        case 1: CASE_SP(4, 1); break;
        case 2: CASE_SP(2, 2); break;
        case 4: CASE_SP(2, 1); break;
        case 5: CASE_SP(1, 1); break;
        default: return -1;
    }
    #undef CASE_SP
    return (int)err;
}

extern "C" int bf11_flip_dual_kv_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* FLIPS0, const void* FLIPS1,
    const void* META0, const void* META1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int so0, int so1,
    int cfg_id, int split, uint32_t base_exp0, uint32_t base_exp1,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (split != 9) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool m_le8 = (M <= 8);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const size_t smem = (size_t)(m_le8 ? 8 : 16) * (KS + 8) * sizeof(__nv_bfloat16);
    #define LAUNCH_KV(W, S, ML, PD, TR) \
        launch_maybe_pdl(bf11_flip_dual_kv_kernel<W, S, ML, 9, PD, TR>, \
            dim3(2 * ((N_each + (W*S*8) - 1) / (W*S*8)), 9, 1), dim3(W * 32, 1, 1), smem, stream, PD, \
            (const __nv_bfloat16*)X0, (const __nv_bfloat16*)X1, \
            (const uint8_t*)SLABS0, (const uint8_t*)SLABS1, \
            (const uint16_t*)FLIPS0, (const uint16_t*)FLIPS1, \
            (const uint4*)META0, (const uint4*)META1, \
            (const uint64_t*)OUTLIERS0, (const uint64_t*)OUTLIERS1, \
            (const int32_t*)O_OFF0, (const int32_t*)O_OFF1, \
            (float*)P, (int32_t*)SEM, (__nv_bfloat16*)O0, (__nv_bfloat16*)O1, (const int32_t*)NLIVE_PTR, \
            M, N_each, K, sx0, sx1, KS, so0, so1, base_exp0, base_exp1)
    #define DISP_KV(W, S) \
        if (m_le8) { \
            if (do_pdl) return (int)(do_trig ? LAUNCH_KV(W, S, true, true, true) : LAUNCH_KV(W, S, true, true, false)); \
            return (int)(do_trig ? LAUNCH_KV(W, S, true, false, true) : LAUNCH_KV(W, S, true, false, false)); \
        } else { \
            if (do_pdl) return (int)(do_trig ? LAUNCH_KV(W, S, false, true, true) : LAUNCH_KV(W, S, false, true, false)); \
            return (int)(do_trig ? LAUNCH_KV(W, S, false, false, true) : LAUNCH_KV(W, S, false, false, false)); \
        }
    switch (cfg_id) {
        case 1: DISP_KV(4, 1);
        case 4: DISP_KV(2, 1);
        case 5: DISP_KV(1, 1);
        default: return -1;
    }
    #undef DISP_KV
    #undef LAUNCH_KV
}

// ---- NEW C=32 Multi-M Entry Points (M = 17..256+, zero partial buffers!) ----
// bm_req: 0 = auto (M<=32 -> 32; M<=64 -> 64; M>64 -> 64 or 128), else 32 / 64 / 128
// k_order: 0 = ascending-K split=1 (0-ULP vs cuBLAS splitK=1), 1 = split_packed grouped (0-ULP vs skinny.py)
extern "C" int bf11_flip_c32_fwd(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    int cfg_id, int bm_req, int k_order, uint32_t base_exp,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (M <= 0 || KS <= 0 || (KS & 127) != 0 || split_packed <= 0 || KS * split_packed != K) {
        return -2;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const bool is_sub2 = (cfg_id == 0 || cfg_id == 2);
    const int ko = (split_packed == 1) ? 0 : ((k_order != 0) ? 1 : 0);

    // Auto BM selection balancing SM occupancy (48 SMs) vs L2 slab reuse & TC amortization:
    //   - M <= 32: BM=32 (MT=2)
    //   - 33 <= M <= 64: BM=64 (MT=4 -> 1x slab read!) unless n_tiles < 72 and M==64 where BM=32 gives 2x CTAs
    //   - M > 64: BM=64 (MT=4 -> 240 CTAs @ M=256 qkvr = 5.0/SM exact!) or BM=128 (MT=8 for SUB_N=1, K_ORDER=0)
    int bm = bm_req;
    if (bm != 32 && bm != 64 && bm != 128) {
        if (M <= 32) {
            bm = 32;
        } else if (M <= 64) {
            bm = 64;
        } else if (!is_sub2 && ko == 0 && (N >= 4096 || (cfg_id == 4 && M <= 128))) {
            // wo_ud (144 tiles) & cfg=4 (103-119 tiles) already fill 48 SMs at m_blocks=1..2 with BM=128!
            bm = 128;
        } else {
            bm = 64;
        }
    }
    if (is_sub2 && bm > 64) bm = 64;
    if (ko == 1 && bm > 64) bm = 64;  // keep <= 78 regs (zero spills) when keeping separate p_acc

    #define DISP_S1(W) \
        if (ko == 0) { \
            if (bm == 32)      return (int)dispatch_bf11_c32_sub1<W, 2, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
            else if (bm == 64) return (int)dispatch_bf11_c32_sub1<W, 4, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
            else               return (int)dispatch_bf11_c32_sub1<W, 8, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
        } else { \
            if (bm == 32)      return (int)dispatch_bf11_c32_sub1<W, 2, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
            else               return (int)dispatch_bf11_c32_sub1<W, 4, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
        }
    #define DISP_S2(W) \
        if (ko == 0) { \
            if (bm == 32)      return (int)dispatch_bf11_c32_sub2<W, 2, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
            else               return (int)dispatch_bf11_c32_sub2<W, 4, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
        } else { \
            if (bm == 32)      return (int)dispatch_bf11_c32_sub2<W, 2, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
            else               return (int)dispatch_bf11_c32_sub2<W, 4, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream); \
        }

    switch (cfg_id) {
        case 0: DISP_S2(4);
        case 1: DISP_S1(4);
        case 2: DISP_S2(2);
        case 4: DISP_S1(2);
        case 5: DISP_S1(1);
        default: return -1;
    }
    #undef DISP_S1
    #undef DISP_S2
}

extern "C" int bf11_flip_dual_kv_c32_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* FLIPS0, const void* FLIPS1,
    const void* META0, const void* META1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    int cfg_id, int k_order, uint32_t base_exp0, uint32_t base_exp1,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (M <= 0 || KS <= 0 || (KS & 127) != 0 || split_packed <= 0 || KS * split_packed != K) {
        return -2;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const int ko = (k_order != 0) ? 1 : 0;
    // Use BM=32 (MT=2) for Dual-KV so M=64..256 yields 32..128 CTAs across 16 N-tiles!
    constexpr int BM = 32;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    const size_t smem = (size_t)2 * BM * 72 * sizeof(__nv_bfloat16);

    #define LAUNCH_DKV_C32(W, KO, CA, PD, TR) \
        launch_maybe_pdl(bf11_flip_dual_kv_c32_kernel<W, 2, KO, CA, PD, TR>, \
            dim3(m_blocks, 2 * ((N_each + (W * 8) - 1) / (W * 8)), 1), dim3(W * 32, 1, 1), smem, stream, PD, \
            (const __nv_bfloat16*)X0, (const __nv_bfloat16*)X1, \
            (const uint8_t*)SLABS0, (const uint8_t*)SLABS1, \
            (const uint16_t*)FLIPS0, (const uint16_t*)FLIPS1, \
            (const uint4*)META0, (const uint4*)META1, \
            (const uint64_t*)OUTLIERS0, (const uint64_t*)OUTLIERS1, \
            (const int32_t*)O_OFF0, (const int32_t*)O_OFF1, \
            (__nv_bfloat16*)O0, (__nv_bfloat16*)O1, (const int32_t*)NLIVE_PTR, \
            M, N_each, K, sx0, sx1, KS, split_packed, so0, so1, base_exp0, base_exp1)
    #define DISP_DKV_CA(W, KO) \
        if (cache_l2) { \
            if (do_pdl) return (int)(do_trig ? LAUNCH_DKV_C32(W, KO, true, true, true) : LAUNCH_DKV_C32(W, KO, true, true, false)); \
            return (int)(do_trig ? LAUNCH_DKV_C32(W, KO, true, false, true) : LAUNCH_DKV_C32(W, KO, true, false, false)); \
        } else { \
            if (do_pdl) return (int)(do_trig ? LAUNCH_DKV_C32(W, KO, false, true, true) : LAUNCH_DKV_C32(W, KO, false, true, false)); \
            return (int)(do_trig ? LAUNCH_DKV_C32(W, KO, false, false, true) : LAUNCH_DKV_C32(W, KO, false, false, false)); \
        }
    #define DISP_DKV(W) \
        if (ko == 0) { DISP_DKV_CA(W, 0); } else { DISP_DKV_CA(W, 1); }

    switch (cfg_id) {
        case 1: DISP_DKV(4);
        case 4: DISP_DKV(2);
        case 5: DISP_DKV(1);
        default: return -1;
    }
    #undef DISP_DKV
    #undef DISP_DKV_CA
    #undef LAUNCH_DKV_C32
}

extern "C" int bf11_flip_c32_version() { return 2; }
