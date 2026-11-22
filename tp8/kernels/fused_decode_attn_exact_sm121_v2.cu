// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Unified 0-ULP Bit-Exact Fused Target Decode Attention v2:
//   Covers ALL THREE target decode paths in ONE drop-in shared library (0 Python ABI changes!):
//     1. Linear C=1 (`M <= 16`, Local `S=8` & Global `S=16`):
//        `attn_prologue_kvq_exact_fwd` + `fused_decode_attn_otf_mgs_fwd`
//     2. S8-P6v2 Fast 4-Tile BM=16 Beam-2 Tree (`M <= 8, ws <= 2`, Local `S=8` & Global `S=16`):
//        `attn_prologue_kvq_tree_exact_fwd` + `fused_decode_attn_otf_mgs_tree_fwd`
//     3. Batched C=32 (`M = 17..512`, `S in {1, 2, 4}`, `FUSE_MT=0/1`, `L2::evict_first`):
//        `attn_prologue_kvq_exact_fwd` + `fused_decode_attn_otf_mgs_c32_fwd`
//
// Key v2 0-ULP Exactness & Correctness Breakthroughs (vs c32.cu & tree.cu):
//   [FIX 1 — Root Cause of residual dy=1..2 on C=1 & C=32]:
//     Live Triton `sm12_attn._attn_kernel` (BM=16, BN=64, 4 warps) selects its softmax layout
//     based on compile-time `BIAS` column divisibility:
//       - GLOBAL (`WINDOW < 0`, `k_beg = 0` -> divisibility 64):
//           `Blocked(sizePerThread=[1,8], threadsPerWarp=[4,8], warpsPerCTA=[4,1])` (`B0`:
//           `<2xf32>` 8-elem chunk tree + `shfl 4->2->1`). PROVEN 10/10 literal `lp_u32=0,
//           op_u32=0, dy=0` on real GB10 (`spark-three`)!
//       - LOCAL (`WINDOW >= 0`, `k_beg = max(past - 1023, 0)` -> divisibility 1):
//           Live `[local]` `_attn_kernel.ptx` (`L2075..L2144`: `log2=8`, `ex2=26=8+8+8+2`,
//           `shfl 16->8->4->2->1`) proves Triton runs `tl.sum(p,1)` in `Blocked([1,1],[1,32],[2,2])`
//           (`BL1_2x2`: Warp-col 0 reduces `p[0..31]` via `shfl 16..1`, Warp-col 1 reduces `p[32..63]`
//           via `shfl 16..1`, then inter-warp `half_lo + half_hi`!). Zero extra barriers in `FUSE_MT=0`
//           (`Red_smem[64][16]` = 4 KiB -> 2 CTAs/SM = 87 KiB <= 101 KiB), and 2 CTAs/SM preserved
//           in `FUSE_MT=1` (`Red_smem[32][8]` = 1 KiB + `cudaFuncSetAttribute` for `MT_MAX=3`!).
//   [FIX 2 — S8-P6v2 Tree Stride-32 + Coherent Cooperative MGS]:
//     Upgrades `fused_decode_attn_otf_mgs_tree_fwd` with stride-32 warp K-column ownership, exact
//     window-matched `sum(p)` reduction, AND `ld.global.cg` (L1-bypass coherent L2 loads!) on
//     persistent ping-pong `OP`/`LP` scratch inside cooperative last-split MGS (fixing a latent
//     cross-SM L1 staleness hazard present in `tree.cu`!).
//   [FIX 3 — Zero-Recompile Diagnostic Mode Bitfield (tree_mode != 0)]:
//     - `tree_mode == 0` (production default): zero-branch compile-time `TREE_MODE == 0` fast path!
//         `WINDOW < 0` -> `B0` (`Blocked([1,8],[4,8],[4,1])`),
//         `WINDOW >= 0` -> `BL1_2x2` (`Blocked([1,1],[1,32],[2,2])`)
//     - `tree_mode != 0` (diagnostic `TREE_MODE == -1`):
//         bits [4:0]  sum_mode in 0..31:
//           0..4,12,13: Blocked([1,8],[4,8],[4,1]) B0..B6 | 8..10: B0/B1 cross-swap
//           14:       Blocked([1,1],[1,32],[2,2]) BL1_2x2 (LOCAL default!)
//           15:       Blocked([1,2],[2,16],[2,2]) BL2_2x2
//           5..7,11,24..29: complete MMA(v2,warps=[1,4]) matrix (M0..M2 x shfl x cross)
//           16,30,31: Blocked([1,1],[1,32],[4,1]) (BL1_w02, BL1_w01, BL1_seq)
//           17..19:   Blocked([1,2],[2,16],[4,1]) (BL2_0, BL2_1, BL2_2)
//           20..23:   Blocked([1,4],[4,8],[4,1])  (BL4_0..BL4_3)
//         bits [6:5]  fma_mode in 0..2  (0=Order A fma(scale,s,b*LOG2E), 1=Order B fma(b,LOG2E,s*scale), 2=unfused)
//         bit  [7]    pv_mode  in 0..1  (0=CombineDotAddF acc*=alpha;mma, 1=separate pv=mma(0);fma(acc,alpha,pv))
//         bit  [8]    mgs_ftz  in 0..1  (0=ex2.approx.f32 matching ns_mgs, 1=ex2.approx.ftz.f32)
//         bit  [9]    diag_dump         (0x200 forces TREE_MODE==-1 FP32 OP/LP dump even at S=1!)
//       Also dumps FP32 `OP`/`LP` even when `NUM_SPLITS == 1` if `OP != nullptr && LP != nullptr`!
//
// Build on GB10 (SM121a — MUST pass --fmad=false, OMIT --use_fast_math & --ftz=true!):
//   /usr/local/cuda/bin/nvcc -O3 -arch=sm_121a --fmad=false --shared -Xcompiler -fPIC \
//       --ptxas-options=-v fused_decode_attn_exact_sm121_v2.cu \
//       -o /rdma/libfused_decode_attn_exact_sm121_v2.so

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr float LN2   = 0.6931471805599453f;
static constexpr int   D_DIM = 128;
static constexpr int   D_REL = 16;
static constexpr int   BN    = 64;
static constexpr int   BM    = 16;
static constexpr int   B_COLS = 80;  // >= 64 + 15 max token span in BM=16 (rounded to 8)
static constexpr int   P_COLS = 72;  // 64 cols padded to 72 to avoid 2-way SMEM bank conflicts

static __device__ __forceinline__ void gdc_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static __device__ __forceinline__ void gdc_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ int atomic_add_acq_rel_gpu(int32_t* ptr, int32_t val) {
    int32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(val) : "memory");
    return old;
}

// L1-bypass coherent L2 loads (mandatory on SM121a when reading cross-SM ping-pong OP/LP!)
static __device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ float4 ld_cg_f4(const float* p) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ float ex2_approx_f32(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float rcp_approx_f32(float x) {
    float y;
    asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float rsqrt_approx_f32(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float div_full_f32(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

static __device__ __forceinline__ float triton_sigmoid(float x) {
    // Exact sm12_attn.py::_sigmoid & ns_mgs::triton_sigmoid
    float neg_x_log2e = __fmul_rn(x, -LOG2E);
    float e = ex2_approx_f32(neg_x_log2e);
    return rcp_approx_f32(__fadd_rn(e, 1.0f));
}

static __device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    float f;
    asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(b));
    return f;
}

static __device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    __nv_bfloat16 b = __float2bfloat16_rn(f);
    return *reinterpret_cast<const uint16_t*>(&b);
}

static __device__ __forceinline__ uint32_t pack_bf16x2(uint16_t a, uint16_t b) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
}

static __device__ __forceinline__ void divmod_p2(int64_t val, int n_mod, int n_shift, int64_t& q, int64_t& r) {
    if (n_shift >= 0) {
        q = val >> n_shift;
        r = val & ((1LL << n_shift) - 1);
    } else {
        q = val / n_mod;
        r = val % n_mod;
    }
}

static __device__ __forceinline__ uint64_t make_l2_evict_first_policy() {
    uint64_t policy = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float frac = 1.0f;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;"
                 : "=l"(policy) : "f"(frac));
#endif
    return policy;
}

static __device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

static __device__ __forceinline__ void cp_async_cg_16_evict_first(
    uint32_t smem_addr, const void* gmem_ptr, uint64_t policy
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr), "l"(policy) : "memory");
#else
    (void)policy;
    cp_async_cg_16(smem_addr, gmem_ptr);
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

static __device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr
) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

static __device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr
) {
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

static __device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// =========================================================================
// KERNEL 1: UNIFIED KV-CONV + Q-CLIP PROLOGUE (NO RelProj, NO rel_logits!)
//   Supports:
//     - C=1 linear (24 CTAs: 8 K + 8 V + 8 Q)
//     - C=1 S8-P6v2 Beam-2 tree (24 CTAs with TreeSconvAnc[M, 4])
//     - C=32 wave-balanced (up to 96 CTAs: 24 K [3w] + 24 V [3w] + 48 Q)
//   Bit-identical (dq=dk=dv=dc=0) to inkling_target_attn_ext_s7.cu & sconv_ops.py!
// =========================================================================
template <bool IS_V, bool DYNAMIC, bool HAS_NLIVE, bool USE_PDL, bool HAS_TREE = false>
static __device__ __forceinline__ void run_kv_tok_warp(
    int t0, int t_stride,
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ Dyn,
    uint16_t* __restrict__ ConvCache,
    uint16_t* __restrict__ KVCache,
    const int64_t* __restrict__ KVSlot,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ QStart,
    const int32_t* __restrict__ NLivePtr,
    const int32_t* __restrict__ TreeSconvAnc,
    int T, int64_t stride_x, int64_t stride_w0, int64_t stride_w1, int64_t stride_dyn,
    int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_kv_blk, int64_t stride_kv_n,
    int kv_bs, int kv_shift, int64_t stride_bt, int max_blocks, int N_cache, int N_shift,
    float scale, float clip_threshold, float clip_eps, float norm_eps
) {
    int nl = T;
    if constexpr (HAS_NLIVE) {
        int v = *NLivePtr;
        if (v < nl) nl = v;
    }
    if (t0 >= nl) {
        if constexpr (USE_PDL) { gdc_wait(); gdc_launch_dependents(); }
        return;
    }

    const int lane = threadIdx.x & 31;
    const int c0 = lane * 4, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
    const int64_t cache_off = IS_V ? 128 : 0;

    float w_t[4][4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if constexpr (DYNAMIC) {
            w_t[k][0] = bf16_to_f32(W[k * stride_w0 + c0 * stride_w1]);
            w_t[k][1] = bf16_to_f32(W[k * stride_w0 + c1 * stride_w1]);
            w_t[k][2] = bf16_to_f32(W[k * stride_w0 + c2 * stride_w1]);
            w_t[k][3] = bf16_to_f32(W[k * stride_w0 + c3 * stride_w1]);
        } else {
            w_t[k][0] = bf16_to_f32(W[c0 * stride_w0 + k * stride_w1]);
            w_t[k][1] = bf16_to_f32(W[c1 * stride_w0 + k * stride_w1]);
            w_t[k][2] = bf16_to_f32(W[c2 * stride_w0 + k * stride_w1]);
            w_t[k][3] = bf16_to_f32(W[c3 * stride_w0 + k * stride_w1]);
        }
    }

    const int64_t pos0 = Pos[t0], slot0 = Slot[t0], kvslot0 = KVSlot[t0];
    const int32_t req0 = SeqIdx[t0], qs0 = QStart[t0];
    const bool valid0 = (slot0 >= 0);

    int anc0[4] = {t0, t0 - 1, t0 - 2, t0 - 3};
    if constexpr (HAS_TREE) {
        const int32_t* ap = TreeSconvAnc + t0 * 4;
        anc0[0] = ap[0]; anc0[1] = ap[1]; anc0[2] = ap[2]; anc0[3] = ap[3];
    }

    float pre_c[4][4];
    bool has_pre[4] = {false, false, false, false};
    if (valid0) {
        const int32_t* bt_row = BlockTable + static_cast<int64_t>(req0) * stride_bt;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int lag = DYNAMIC ? k : (3 - k);
            if (lag > 0) {
                const int64_t src = pos0 - lag;
                const bool is_cached = HAS_TREE ? (anc0[lag] < 0) : (t0 - lag < qs0);
                if (src >= 0 && is_cached) {
                    has_pre[k] = true;
                    int64_t lblk, rem;
                    divmod_p2(src, N_cache, N_shift, lblk, rem);
                    if (lblk >= max_blocks) lblk = max_blocks - 1;
                    int64_t blk = static_cast<int64_t>(bt_row[lblk]);
                    const uint16_t* cp = ConvCache + blk * stride_c_blk + rem * stride_c_n + cache_off;
                    uint32_t p01 = *reinterpret_cast<const uint32_t*>(cp + c0);
                    uint32_t p23 = *reinterpret_cast<const uint32_t*>(cp + c2);
                    pre_c[k][0] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p01 & 0xFFFFu)));
                    pre_c[k][1] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p01 >> 16)));
                    pre_c[k][2] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p23 & 0xFFFFu)));
                    pre_c[k][3] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p23 >> 16)));
                }
            }
        }
    }

    if constexpr (USE_PDL) { gdc_wait(); gdc_launch_dependents(); }

    for (int t = t0; t < nl; t += t_stride) {
        const bool is_first = (t == t0);
        const int64_t pos = is_first ? pos0 : Pos[t];
        const int64_t sl = is_first ? slot0 : Slot[t];
        const int64_t kvs = is_first ? kvslot0 : KVSlot[t];
        const int32_t req = is_first ? req0 : SeqIdx[t];
        const int32_t qs = is_first ? qs0 : QStart[t];
        const bool valid = (sl >= 0);

        const uint16_t* xp = X + static_cast<int64_t>(t) * stride_x;
        uint32_t cur01 = *reinterpret_cast<const uint32_t*>(xp + c0);
        uint32_t cur23 = *reinterpret_cast<const uint32_t*>(xp + c2);
        float cur[4] = {
            bf16_to_f32(static_cast<uint16_t>(cur01 & 0xFFFFu)),
            bf16_to_f32(static_cast<uint16_t>(cur01 >> 16)),
            bf16_to_f32(static_cast<uint16_t>(cur23 & 0xFFFFu)),
            bf16_to_f32(static_cast<uint16_t>(cur23 >> 16)),
        };

        if (valid) {
            int64_t dblk, drem;
            divmod_p2(sl, N_cache, N_shift, dblk, drem);
            uint16_t* dst_c = ConvCache + dblk * stride_c_blk + drem * stride_c_n + cache_off;
            *reinterpret_cast<uint32_t*>(dst_c + c0) = cur01;
            *reinterpret_cast<uint32_t*>(dst_c + c2) = cur23;
        }

        float dyn_w[4] = {0.f, 0.f, 0.f, 0.f};
        if constexpr (DYNAMIC) {
            const uint16_t* dp = Dyn + static_cast<int64_t>(t) * stride_dyn;
            #pragma unroll
            for (int k = 0; k < 4; ++k) dyn_w[k] = bf16_to_f32(dp[k]);
        }

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int lag = DYNAMIC ? k : (3 - k);
            const int64_t src = pos - lag;
            const bool in_win = valid && (src >= 0);
            int row;
            bool intra, cached;
            if constexpr (HAS_TREE) {
                row = is_first ? anc0[lag] : TreeSconvAnc[t * 4 + lag];
                intra = in_win && (row >= 0);
                cached = in_win && (row < 0);
            } else {
                row = t - lag;
                intra = in_win && (row >= qs);
                cached = in_win && (row < qs);
            }
            float xt[4] = {0.f, 0.f, 0.f, 0.f};
            if (intra) {
                if (lag == 0) {
                    xt[0] = __fadd_rn(cur[0], 0.f); xt[1] = __fadd_rn(cur[1], 0.f);
                    xt[2] = __fadd_rn(cur[2], 0.f); xt[3] = __fadd_rn(cur[3], 0.f);
                } else {
                    const uint16_t* rx = X + static_cast<int64_t>(row) * stride_x;
                    uint32_t r01 = *reinterpret_cast<const uint32_t*>(rx + c0);
                    uint32_t r23 = *reinterpret_cast<const uint32_t*>(rx + c2);
                    xt[0] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r01 & 0xFFFFu)), 0.f);
                    xt[1] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r01 >> 16)), 0.f);
                    xt[2] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r23 & 0xFFFFu)), 0.f);
                    xt[3] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r23 >> 16)), 0.f);
                }
            } else if (cached) {
                if (is_first && has_pre[k]) {
                    xt[0] = pre_c[k][0]; xt[1] = pre_c[k][1]; xt[2] = pre_c[k][2]; xt[3] = pre_c[k][3];
                } else {
                    int64_t lblk, rem;
                    divmod_p2(src, N_cache, N_shift, lblk, rem);
                    if (lblk >= max_blocks) lblk = max_blocks - 1;
                    int64_t blk = static_cast<int64_t>(BlockTable[static_cast<int64_t>(req) * stride_bt + lblk]);
                    const uint16_t* cp = ConvCache + blk * stride_c_blk + rem * stride_c_n + cache_off;
                    uint32_t r01 = *reinterpret_cast<const uint32_t*>(cp + c0);
                    uint32_t r23 = *reinterpret_cast<const uint32_t*>(cp + c2);
                    xt[0] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r01 & 0xFFFFu)));
                    xt[1] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r01 >> 16)));
                    xt[2] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r23 & 0xFFFFu)));
                    xt[3] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r23 >> 16)));
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float wk = DYNAMIC ? __fadd_rn(w_t[k][i], dyn_w[k]) : w_t[k][i];
                if (DYNAMIC && k == 0) acc[i] = __fmul_rn(xt[i], wk);
                else acc[i] = fmaf(xt[i], wk, acc[i]);
            }
        }

        float y[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            y[i] = bf16_to_f32(f32_to_bf16(__fmul_rn(__fadd_rn(acc[i], cur[i]), scale)));
        }
        float s0 = fmaf(y[0], y[0], __fmul_rn(y[2], y[2]));
        float s1 = fmaf(y[1], y[1], __fmul_rn(y[3], y[3]));
        float sum_sq = __fadd_rn(s0, s1);
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 16));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 8));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 4));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 2));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 1));

        float mult;
        if constexpr (IS_V) {
            mult = rsqrt_approx_f32(__fadd_rn(__fdiv_rn(sum_sq, 128.0f), norm_eps));
        } else {
            float norm = (sum_sq > 0.0f) ? __fmul_rn(sum_sq, rsqrt_approx_f32(sum_sq)) : 0.0f;
            mult = fminf(__fdiv_rn(clip_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        }

        if (kvs >= 0) {
            int64_t kv_blk, kv_rem;
            divmod_p2(kvs, kv_bs, kv_shift, kv_blk, kv_rem);
            uint16_t* kv_dst = KVCache + kv_blk * stride_kv_blk + kv_rem * stride_kv_n;
            *reinterpret_cast<uint32_t*>(kv_dst + c0) = pack_bf16x2(f32_to_bf16(__fmul_rn(y[0], mult)), f32_to_bf16(__fmul_rn(y[1], mult)));
            *reinterpret_cast<uint32_t*>(kv_dst + c2) = pack_bf16x2(f32_to_bf16(__fmul_rn(y[2], mult)), f32_to_bf16(__fmul_rn(y[3], mult)));
        }
    }
}

template <bool HAS_TAU, bool HAS_NLIVE, bool USE_PDL>
static __device__ __forceinline__ void run_q_tok_cta(
    int t0, int t_stride,
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ QScale,
    const float* __restrict__ Tau,
    uint16_t* __restrict__ QOut,
    const int32_t* __restrict__ NLivePtr,
    int T, int64_t stride_q, float clip_threshold, float clip_eps
) {
    if (threadIdx.x >= 96) {
        if constexpr (USE_PDL) { gdc_wait(); gdc_launch_dependents(); }
        return;
    }
    int nl = T;
    if constexpr (HAS_NLIVE) {
        int v = *NLivePtr;
        if (v < nl) nl = v;
    }
    if (t0 >= nl) {
        if constexpr (USE_PDL) { gdc_wait(); gdc_launch_dependents(); }
        return;
    }

    const int h = threadIdx.x >> 4;
    const int l = threadIdx.x & 15;
    const int d0 = l * 8;
    float sc[8];
    {
        const uint4 sv = *reinterpret_cast<const uint4*>(QScale + h * 128 + d0);
        sc[0] = bf16_to_f32(static_cast<uint16_t>(sv.x & 0xFFFFu));
        sc[1] = bf16_to_f32(static_cast<uint16_t>(sv.x >> 16));
        sc[2] = bf16_to_f32(static_cast<uint16_t>(sv.y & 0xFFFFu));
        sc[3] = bf16_to_f32(static_cast<uint16_t>(sv.y >> 16));
        sc[4] = bf16_to_f32(static_cast<uint16_t>(sv.z & 0xFFFFu));
        sc[5] = bf16_to_f32(static_cast<uint16_t>(sv.z >> 16));
        sc[6] = bf16_to_f32(static_cast<uint16_t>(sv.w & 0xFFFFu));
        sc[7] = bf16_to_f32(static_cast<uint16_t>(sv.w >> 16));
    }

    if constexpr (USE_PDL) { gdc_wait(); gdc_launch_dependents(); }

    const bool q_aligned16 = ((stride_q & 7) == 0) && ((reinterpret_cast<uintptr_t>(Q) & 15) == 0);
    for (int t = t0; t < nl; t += t_stride) {
        const uint16_t* qp = Q + static_cast<int64_t>(t) * stride_q + h * 128 + d0;
        uint4 qv;
        if (q_aligned16) {
            qv = *reinterpret_cast<const uint4*>(qp);
        } else {
            qv.x = *reinterpret_cast<const uint32_t*>(qp);
            qv.y = *reinterpret_cast<const uint32_t*>(qp + 2);
            qv.z = *reinterpret_cast<const uint32_t*>(qp + 4);
            qv.w = *reinterpret_cast<const uint32_t*>(qp + 6);
        }
        float x[8];
        x[0] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.x & 0xFFFFu)), sc[0]);
        x[1] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.x >> 16)),      sc[1]);
        x[2] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.y & 0xFFFFu)), sc[2]);
        x[3] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.y >> 16)),      sc[3]);
        x[4] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.z & 0xFFFFu)), sc[4]);
        x[5] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.z >> 16)),      sc[5]);
        x[6] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.w & 0xFFFFu)), sc[6]);
        x[7] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.w >> 16)),      sc[7]);

        float e01 = fmaf(x[0], x[0], __fmul_rn(x[2], x[2]));
        float o01 = fmaf(x[1], x[1], __fmul_rn(x[3], x[3]));
        float e23 = fmaf(x[4], x[4], __fmul_rn(x[6], x[6]));
        float o23 = fmaf(x[5], x[5], __fmul_rn(x[7], x[7]));
        float sum_sq = __fadd_rn(__fadd_rn(e01, e23), __fadd_rn(o01, o23));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 8));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 4));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 2));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 1));

        float norm = (sum_sq > 0.0f) ? __fmul_rn(sum_sq, rsqrt_approx_f32(sum_sq)) : 0.0f;
        float mult = fminf(__fdiv_rn(clip_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        float tau = HAS_TAU ? Tau[t] : 1.0f;

        uint16_t b[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t bi = f32_to_bf16(__fmul_rn(x[i], mult));
            if constexpr (HAS_TAU) bi = f32_to_bf16(__fmul_rn(bf16_to_f32(bi), tau));
            b[i] = bi;
        }
        uint4 ov;
        ov.x = pack_bf16x2(b[0], b[1]);
        ov.y = pack_bf16x2(b[2], b[3]);
        ov.z = pack_bf16x2(b[4], b[5]);
        ov.w = pack_bf16x2(b[6], b[7]);
        *reinterpret_cast<uint4*>(QOut + (static_cast<int64_t>(t) * 6 + h) * 128 + d0) = ov;
    }
}

template <bool DYNAMIC, bool HAS_TAU, bool HAS_NLIVE, bool USE_PDL, bool C32_MULTIWARP, bool HAS_TREE = false>
__global__ void __launch_bounds__(96, 2)
attn_prologue_kvq_exact_kernel(
    const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    const uint16_t* __restrict__ WK, const uint16_t* __restrict__ WV,
    const uint16_t* __restrict__ DK, const uint16_t* __restrict__ DV,
    uint16_t* __restrict__ ConvCache, uint16_t* __restrict__ KeyCache, uint16_t* __restrict__ ValueCache,
    const int64_t* __restrict__ KVSlot, const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx, const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BlockTable, const int32_t* __restrict__ QStart,
    const uint16_t* __restrict__ Q, const uint16_t* __restrict__ QScale,
    const float* __restrict__ Tau, uint16_t* __restrict__ QOut,
    const int32_t* __restrict__ NLivePtr,
    const int32_t* __restrict__ TreeSconvAnc,
    int T, int k_ctas, int v_ctas, int q_ctas,
    int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs, int kv_shift,
    int64_t stride_bt, int max_blocks, int N_cache, int N_shift,
    int64_t stride_q, float scale, float clip_threshold, float clip_eps, float norm_eps
) {
    const int bid = blockIdx.x;
    if constexpr (!C32_MULTIWARP) {
        if (bid < 8) {
            if (threadIdx.x >= 32) {
                if constexpr (USE_PDL) { gdc_wait(); gdc_launch_dependents(); }
                return;
            }
            run_kv_tok_warp<false, DYNAMIC, HAS_NLIVE, USE_PDL, HAS_TREE>(
                bid, 8, K, WK, DK, ConvCache, KeyCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                NLivePtr, TreeSconvAnc,
                T, stride_k, stride_w0, stride_w1, stride_dk, stride_c_blk, stride_c_n,
                stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                scale, clip_threshold, clip_eps, norm_eps);
        } else if (bid < 16) {
            if (threadIdx.x >= 32) {
                if constexpr (USE_PDL) { gdc_wait(); gdc_launch_dependents(); }
                return;
            }
            run_kv_tok_warp<true, DYNAMIC, HAS_NLIVE, USE_PDL, HAS_TREE>(
                bid - 8, 8, V, WV, DV, ConvCache, ValueCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                NLivePtr, TreeSconvAnc,
                T, stride_v, stride_w0, stride_w1, stride_dv, stride_c_blk, stride_c_n,
                stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                scale, clip_threshold, clip_eps, norm_eps);
        } else {
            run_q_tok_cta<HAS_TAU, HAS_NLIVE, USE_PDL>(
                bid - 16, 8, Q, QScale, Tau, QOut, NLivePtr, T, stride_q, clip_threshold, clip_eps);
        }
    } else {
        const int wid = threadIdx.x >> 5;
        if (bid < k_ctas) {
            run_kv_tok_warp<false, DYNAMIC, HAS_NLIVE, USE_PDL, false>(
                bid * 3 + wid, k_ctas * 3,
                K, WK, DK, ConvCache, KeyCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                NLivePtr, nullptr,
                T, stride_k, stride_w0, stride_w1, stride_dk, stride_c_blk, stride_c_n,
                stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                scale, clip_threshold, clip_eps, norm_eps);
        } else if (bid < k_ctas + v_ctas) {
            const int vbid = bid - k_ctas;
            run_kv_tok_warp<true, DYNAMIC, HAS_NLIVE, USE_PDL, false>(
                vbid * 3 + wid, v_ctas * 3,
                V, WV, DV, ConvCache, ValueCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                NLivePtr, nullptr,
                T, stride_v, stride_w0, stride_w1, stride_dv, stride_c_blk, stride_c_n,
                stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                scale, clip_threshold, clip_eps, norm_eps);
        } else {
            run_q_tok_cta<HAS_TAU, HAS_NLIVE, USE_PDL>(
                bid - k_ctas - v_ctas, q_ctas,
                Q, QScale, Tau, QOut, NLivePtr, T, stride_q, clip_threshold, clip_eps);
        }
    }
}

// =========================================================================
// KERNEL 2 HELPERS: EXACT BLOCKED([1,8],[4,8],[4,1]) REDUCTION + OTF BIAS + MGS
// =========================================================================

// Exact split-KV weight sum tree matching ns_mgs::reduce_splits_sum<S> (0-ULP proven!)
template <int S>
static __device__ __forceinline__ float reduce_splits_sum(const float w[S]) {
    if constexpr (S == 2) {
        return __fadd_rn(w[0], w[1]);
    } else if constexpr (S == 4) {
        return __fadd_rn(__fadd_rn(w[0], w[2]), __fadd_rn(w[1], w[3]));
    } else if constexpr (S == 8) {
        float s02 = __fadd_rn(w[0], w[2]);
        float s13 = __fadd_rn(w[1], w[3]);
        float s46 = __fadd_rn(w[4], w[6]);
        float s57 = __fadd_rn(w[5], w[7]);
        return __fadd_rn(__fadd_rn(s02, s13), __fadd_rn(s46, s57));
    } else if constexpr (S == 16) {
        float w0 = __fadd_rn(__fadd_rn(w[0], w[2]),   __fadd_rn(w[1], w[3]));
        float w1 = __fadd_rn(__fadd_rn(w[4], w[6]),   __fadd_rn(w[5], w[7]));
        float w2 = __fadd_rn(__fadd_rn(w[8], w[10]),  __fadd_rn(w[9], w[11]));
        float w3 = __fadd_rn(__fadd_rn(w[12], w[14]), __fadd_rn(w[13], w[15]));
        return __fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3));
    } else {
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < S; ++i) s = __fadd_rn(s, w[i]);
        return s;
    }
}

template <bool EVICT_FIRST>
static __device__ __forceinline__ void issue_kv64_swiz_async(
    const uint16_t* __restrict__ KV_base,
    int64_t stride_b, int64_t stride_s,
    const int2* __restrict__ blk_slot,
    uint32_t kv_sptr,
    uint4* __restrict__ smem_kv4,
    uint64_t l2_policy,
    int tid
) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int idx = i * 128 + tid;  // 0..1023
        const int row = idx >> 4;       // 0..63
        const int c = idx & 15;         // 0..15 (16B = 8 bf16)
        const int swiz_c = c ^ (row & 7);
        const int2 bs = blk_slot[row];
        if (bs.x >= 0) {
            const int64_t roff = static_cast<int64_t>(bs.x) * stride_b + static_cast<int64_t>(bs.y) * stride_s;
            const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(row * 256 + swiz_c * 16);
            if constexpr (EVICT_FIRST) {
                cp_async_cg_16_evict_first(s_addr, KV_base + roff + c * 8, l2_policy);
            } else {
                cp_async_cg_16(s_addr, KV_base + roff + c * 8);
            }
        } else {
            smem_kv4[row * 16 + swiz_c] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

template <bool HAS_TAU>
static __device__ __forceinline__ void clip_r_row_exact_8lane(
    const uint16_t* __restrict__ r_ptr,
    const uint16_t* __restrict__ r_scale_row,
    bool valid,
    float tau, float rel_threshold, float clip_eps,
    int sub_lane,  // 0..7 -> cols (2*sub_lane, 2*sub_lane+1)
    uint16_t* dst_row16
) {
    const int d = sub_lane * 2;
    float x0 = 0.0f, x1 = 0.0f;
    if (valid) {
        uint32_t rv = *reinterpret_cast<const uint32_t*>(r_ptr + d);
        uint32_t sv = *reinterpret_cast<const uint32_t*>(r_scale_row + d);
        x0 = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(rv & 0xFFFFu)), bf16_to_f32(static_cast<uint16_t>(sv & 0xFFFFu)));
        x1 = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(rv >> 16)),      bf16_to_f32(static_cast<uint16_t>(sv >> 16)));
    }
    float s = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
    if (valid) {
        float norm = (s > 0.0f) ? __fmul_rn(s, rsqrt_approx_f32(s)) : 0.0f;
        float mult = fminf(__fdiv_rn(rel_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        uint16_t o0 = f32_to_bf16(__fmul_rn(x0, mult));
        uint16_t o1 = f32_to_bf16(__fmul_rn(x1, mult));
        if constexpr (HAS_TAU) {
            o0 = f32_to_bf16(__fmul_rn(bf16_to_f32(o0), tau));
            o1 = f32_to_bf16(__fmul_rn(bf16_to_f32(o1), tau));
        }
        *reinterpret_cast<uint32_t*>(dst_row16 + d) = pack_bf16x2(o0, o1);
    } else {
        *reinterpret_cast<uint32_t*>(dst_row16 + d) = 0u;
    }
}

// Score + bias accumulation helper:
//   TREE_MODE == 0: Order A `fma.rn.f32(scale_log2, s, __fmul_rn(b, LOG2E))` (0 runtime branch!)
//   TREE_MODE != 0: fma_mode = (tree_mode_rt >> 5) & 3:
//     0 = Order A `fmaf(scale_log2, s, __fmul_rn(b, LOG2E))`
//     1 = Order B `fmaf(b, LOG2E, __fmul_rn(s, scale_log2))`
//     2 = Unfused `__fadd_rn(__fmul_rn(s, scale_log2), __fmul_rn(b, LOG2E))`
template <int TREE_MODE>
static __device__ __forceinline__ float apply_score_bias(
    float s, float scale_log2, float b, int tree_mode_rt
) {
    if constexpr (TREE_MODE == 0) {
        (void)tree_mode_rt;
        return fmaf(scale_log2, s, __fmul_rn(b, LOG2E));
    } else {
        const int fm = (tree_mode_rt >> 5) & 3;
        if (fm == 1) {
            return fmaf(b, LOG2E, __fmul_rn(s, scale_log2));
        } else if (fm == 2) {
            return __fadd_rn(__fmul_rn(s, scale_log2), __fmul_rn(b, LOG2E));
        }
        return fmaf(scale_log2, s, __fmul_rn(b, LOG2E));
    }
}

// Exact 8-element contiguous chunk sum across the 4 consecutive lanes `tid = 0..3`
// holding `(u, v) = (x_{2*tid}, x_{2*tid+1})`:
//   B0 (Triton LLVM <2xf32> tree — proven in attn_res_v4 & _l2_clip):
//       ((x0+x2)+(x4+x6)) + ((x1+x3)+(x5+x7))
static __device__ __forceinline__ float reduce_chunk8_b0(float u, float v) {
    u = __fadd_rn(u, __shfl_xor_sync(0xFFFFFFFFu, u, 1));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    u = __fadd_rn(u, __shfl_xor_sync(0xFFFFFFFFu, u, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    return __fadd_rn(u, v);
}

//   B1 (4-pair -> 2-pair -> 1-pair -> even+odd):
//       ((x0+x4)+(x2+x6)) + ((x1+x5)+(x3+x7))
static __device__ __forceinline__ float reduce_chunk8_b1(float u, float v) {
    u = __fadd_rn(u, __shfl_xor_sync(0xFFFFFFFFu, u, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    u = __fadd_rn(u, __shfl_xor_sync(0xFFFFFFFFu, u, 1));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return __fadd_rn(u, v);
}

//   B2 (contiguous pair binary tree):
//       ((x0+x1)+(x2+x3)) + ((x4+x5)+(x6+x7))
static __device__ __forceinline__ float reduce_chunk8_b2(float u, float v) {
    float s = __fadd_rn(u, v);
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    return s;
}

//   B3 (stride-2 pair binary tree):
//       ((x0+x1)+(x4+x5)) + ((x2+x3)+(x6+x7))
static __device__ __forceinline__ float reduce_chunk8_b3(float u, float v) {
    float s = __fadd_rn(u, v);
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
    return s;
}

//   B4 (sequential left-to-right per 8-element chunk):
//       (((((((x0+x1)+x2)+x3)+x4)+x5)+x6)+x7)
static __device__ __forceinline__ float reduce_chunk8_b4(float u, float v, int grp) {
    const int base = grp * 4;
    float x0 = __shfl_sync(0xFFFFFFFFu, u, base + 0);
    float x1 = __shfl_sync(0xFFFFFFFFu, v, base + 0);
    float x2 = __shfl_sync(0xFFFFFFFFu, u, base + 1);
    float x3 = __shfl_sync(0xFFFFFFFFu, v, base + 1);
    float x4 = __shfl_sync(0xFFFFFFFFu, u, base + 2);
    float x5 = __shfl_sync(0xFFFFFFFFu, v, base + 2);
    float x6 = __shfl_sync(0xFFFFFFFFu, u, base + 3);
    float x7 = __shfl_sync(0xFFFFFFFFu, v, base + 3);
    float s = __fadd_rn(x0, x1);
    s = __fadd_rn(s, x2);
    s = __fadd_rn(s, x3);
    s = __fadd_rn(s, x4);
    s = __fadd_rn(s, x5);
    s = __fadd_rn(s, x6);
    s = __fadd_rn(s, x7);
    return s;
}

//   B5 (<2xf32> linear accumulation per 8-element chunk):
//       (((x0+x2)+x4)+x6) + (((x1+x3)+x5)+x7)
static __device__ __forceinline__ float reduce_chunk8_b5(float u, float v, int grp) {
    const int base = grp * 4;
    float x0 = __shfl_sync(0xFFFFFFFFu, u, base + 0);
    float x1 = __shfl_sync(0xFFFFFFFFu, v, base + 0);
    float x2 = __shfl_sync(0xFFFFFFFFu, u, base + 1);
    float x3 = __shfl_sync(0xFFFFFFFFu, v, base + 1);
    float x4 = __shfl_sync(0xFFFFFFFFu, u, base + 2);
    float x5 = __shfl_sync(0xFFFFFFFFu, v, base + 2);
    float x6 = __shfl_sync(0xFFFFFFFFu, u, base + 3);
    float x7 = __shfl_sync(0xFFFFFFFFu, v, base + 3);
    float e = __fadd_rn(__fadd_rn(__fadd_rn(x0, x2), x4), x6);
    float o = __fadd_rn(__fadd_rn(__fadd_rn(x1, x3), x5), x7);
    return __fadd_rn(e, o);
}

//   B6 (stride-4 -> pair -> half binary tree per 8-element chunk):
//       ((x0+x4)+(x1+x5)) + ((x2+x6)+(x3+x7))
static __device__ __forceinline__ float reduce_chunk8_b6(float u, float v) {
    float eu = __fadd_rn(u, __shfl_xor_sync(0xFFFFFFFFu, u, 2));
    float ev = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    float s = __fadd_rn(eu, ev);
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
    return s;
}

// Exact 4-element + rep intra-thread reductions for Blocked([1,4],[4,8],[4,1]):
//   `tid in {0,1}` holds half 0 (cols 8*wid+0..3 & 32+8*wid+0..3),
//   `tid in {2,3}` holds half 1 (cols 8*wid+4..7 & 32+8*wid+4..7); partner is `tid ^ 1`.
static __device__ __forceinline__ float reduce_bl4_half(
    float p00, float p10, float p01, float p11, int grp, int tid, int sm
) {
    if (sm == 20) { // BL4_0 (<2xf32> rep-first): ((x0+y0)+(x2+y2)) + ((x1+y1)+(x3+y3))
        float e = __fadd_rn(p00, p01);
        float o = __fadd_rn(p10, p11);
        e = __fadd_rn(e, __shfl_xor_sync(0xFFFFFFFFu, e, 1));
        o = __fadd_rn(o, __shfl_xor_sync(0xFFFFFFFFu, o, 1));
        return __fadd_rn(e, o);
    } else if (sm == 21) { // BL4_1 (<2xf32> chunk-first): ((x0+x2)+(y0+y2)) + ((x1+x3)+(y1+y3))
        float e0 = __fadd_rn(p00, __shfl_xor_sync(0xFFFFFFFFu, p00, 1));
        float o0 = __fadd_rn(p10, __shfl_xor_sync(0xFFFFFFFFu, p10, 1));
        float e1 = __fadd_rn(p01, __shfl_xor_sync(0xFFFFFFFFu, p01, 1));
        float o1 = __fadd_rn(p11, __shfl_xor_sync(0xFFFFFFFFu, p11, 1));
        return __fadd_rn(__fadd_rn(e0, e1), __fadd_rn(o0, o1));
    } else if (sm == 22) { // BL4_2 (pair): ((x0+x1)+(x2+x3)) + ((y0+y1)+(y2+y3))
        float s0 = __fadd_rn(p00, p10);
        float s1 = __fadd_rn(p01, p11);
        s0 = __fadd_rn(s0, __shfl_xor_sync(0xFFFFFFFFu, s0, 1));
        s1 = __fadd_rn(s1, __shfl_xor_sync(0xFFFFFFFFu, s1, 1));
        return __fadd_rn(s0, s1);
    } else { // sm == 23: BL4_3 (sequential x0..x3, y0..y3)
        const int base = grp * 4 + (tid & 2);
        float x0 = __shfl_sync(0xFFFFFFFFu, p00, base + 0);
        float x1 = __shfl_sync(0xFFFFFFFFu, p10, base + 0);
        float x2 = __shfl_sync(0xFFFFFFFFu, p00, base + 1);
        float x3 = __shfl_sync(0xFFFFFFFFu, p10, base + 1);
        float y0 = __shfl_sync(0xFFFFFFFFu, p01, base + 0);
        float y1 = __shfl_sync(0xFFFFFFFFu, p11, base + 0);
        float y2 = __shfl_sync(0xFFFFFFFFu, p01, base + 1);
        float y3 = __shfl_sync(0xFFFFFFFFu, p11, base + 1);
        float s = __fadd_rn(x0, x1);
        s = __fadd_rn(s, x2); s = __fadd_rn(s, x3);
        s = __fadd_rn(s, y0); s = __fadd_rn(s, y1);
        s = __fadd_rn(s, y2); s = __fadd_rn(s, y3);
        return s;
    }
}

// Intra-warp phase for all scalar-cross (4-warp Red_smem[0..3]) modes:
//   sm in {0..4, 8..10, 12, 13}: Blocked([1,8],[4,8],[4,1]) B0..B6
//   sm in {5..7, 11, 24..29}:    MMA(v2, warps=[1,4]) complete matrix
static __device__ __forceinline__ float reduce_p_intra_scalar(
    float p00, float p10, float p01, float p11, int grp, int sm
) {
    if (sm == 0 || sm == 8 || sm == 10) {
        return __fadd_rn(reduce_chunk8_b0(p00, p10), reduce_chunk8_b0(p01, p11));
    } else if (sm == 1 || sm == 9) {
        return __fadd_rn(reduce_chunk8_b1(p00, p10), reduce_chunk8_b1(p01, p11));
    } else if (sm == 2) {
        return __fadd_rn(reduce_chunk8_b2(p00, p10), reduce_chunk8_b2(p01, p11));
    } else if (sm == 3) {
        return __fadd_rn(reduce_chunk8_b3(p00, p10), reduce_chunk8_b3(p01, p11));
    } else if (sm == 4) {
        return __fadd_rn(reduce_chunk8_b4(p00, p10, grp), reduce_chunk8_b4(p01, p11, grp));
    } else if (sm == 12) {
        return __fadd_rn(reduce_chunk8_b5(p00, p10, grp), reduce_chunk8_b5(p01, p11, grp));
    } else if (sm == 13) {
        return __fadd_rn(reduce_chunk8_b6(p00, p10), reduce_chunk8_b6(p01, p11));
    } else {
        // Complete MMA(v2, warps=[1,4]) family: sm in {5,6,7,11,24,25,26,27,28,29}
        float lw;
        if (sm == 6 || sm == 25 || sm == 27 || sm == 29) {
            lw = __fadd_rn(__fadd_rn(p00, p10), __fadd_rn(p01, p11));       // M1
        } else if (sm == 7 || sm == 26) {
            lw = __fadd_rn(__fadd_rn(__fadd_rn(p00, p10), p01), p11);      // M2
        } else {
            lw = __fadd_rn(__fadd_rn(p00, p01), __fadd_rn(p10, p11));       // M0
        }
        if (sm == 11 || sm == 27) {
            lw = __fadd_rn(lw, __shfl_xor_sync(0xFFFFFFFFu, lw, 1));
            lw = __fadd_rn(lw, __shfl_xor_sync(0xFFFFFFFFu, lw, 2));
        } else {
            lw = __fadd_rn(lw, __shfl_xor_sync(0xFFFFFFFFu, lw, 2));
            lw = __fadd_rn(lw, __shfl_xor_sync(0xFFFFFFFFu, lw, 1));
        }
        return lw;
    }
}

static __device__ __forceinline__ float reduce_p_cross_scalar(
    float r0, float r1, float r2, float r3, int sm
) {
    if (sm == 8 || sm == 9 || sm == 11 || sm == 24 || sm == 25 || sm == 26 || sm == 27) {
        return __fadd_rn(__fadd_rn(r0, r1), __fadd_rn(r2, r3));
    } else if (sm == 10 || sm == 28 || sm == 29) {
        return __fadd_rn(__fadd_rn(__fadd_rn(r0, r1), r2), r3);
    }
    return __fadd_rn(__fadd_rn(r0, r2), __fadd_rn(r1, r3));
}

// Phase 1 (before cp_async_wait_group, 100% hidden behind V load!) -> (a0, b0) for row0, (a1, b1) for row1
template <int WINDOW, int TREE_MODE>
static __device__ __forceinline__ void reduce_p_phase1(
    float p0_0, float p1_0, float p0_1, float p1_1,
    float p2_0, float p3_0, float p2_1, float p3_1,
    int grp, int tid, int tree_mode_rt,
    float& a0, float& b0, float& a1, float& b1
) {
    if constexpr (TREE_MODE == 0) {
        (void)grp; (void)tid; (void)tree_mode_rt;
        if constexpr (WINDOW < 0) {
            // PROVEN 100% 0-ULP on GB10 GLOBAL (Suite A 11/11, B 10/10, C): Blocked([1,8],[4,8],[4,1]) B0!
            a0 = __fadd_rn(reduce_chunk8_b0(p0_0, p1_0), reduce_chunk8_b0(p0_1, p1_1));
            a1 = __fadd_rn(reduce_chunk8_b0(p2_0, p3_0), reduce_chunk8_b0(p2_1, p3_1));
            b0 = 0.0f; b1 = 0.0f;
        } else {
            // Exact live [local] PTX (L2075..L2144) layout: Blocked([1,1],[1,32],[2,2]) BL1_2x2!
            // (Reduces p[0..31] and p[32..63] independently in phase 2, then adds half_lo + half_hi!)
            (void)p0_0; (void)p1_0; (void)p0_1; (void)p1_1;
            (void)p2_0; (void)p3_0; (void)p2_1; (void)p3_1;
            a0 = 0.0f; b0 = 0.0f; a1 = 0.0f; b1 = 0.0f;
        }
    } else {
        const int sm = tree_mode_rt & 0x1F;
        if (sm == 14) {
            a0 = 0.0f; b0 = 0.0f; a1 = 0.0f; b1 = 0.0f;
        } else if (sm == 15) {
            // BL2_2x2: Blocked([1,2],[2,16],[2,2]): half_lo pairs in a, half_hi pairs in b
            a0 = __fadd_rn(p0_0, p1_0); b0 = __fadd_rn(p0_1, p1_1);
            a1 = __fadd_rn(p2_0, p3_0); b1 = __fadd_rn(p2_1, p3_1);
        } else if (sm == 16 || sm == 30 || sm == 31) {
            a0 = __fadd_rn(p0_0, p0_1); b0 = __fadd_rn(p1_0, p1_1);
            a1 = __fadd_rn(p2_0, p2_1); b1 = __fadd_rn(p3_0, p3_1);
        } else if (sm == 17) {
            a0 = __fadd_rn(__fadd_rn(p0_0, p0_1), __fadd_rn(p1_0, p1_1));
            a1 = __fadd_rn(__fadd_rn(p2_0, p2_1), __fadd_rn(p3_0, p3_1));
            b0 = 0.0f; b1 = 0.0f;
        } else if (sm == 18) {
            a0 = __fadd_rn(__fadd_rn(p0_0, p1_0), __fadd_rn(p0_1, p1_1));
            a1 = __fadd_rn(__fadd_rn(p2_0, p3_0), __fadd_rn(p2_1, p3_1));
            b0 = 0.0f; b1 = 0.0f;
        } else if (sm == 19) {
            a0 = __fadd_rn(__fadd_rn(__fadd_rn(p0_0, p1_0), p0_1), p1_1);
            a1 = __fadd_rn(__fadd_rn(__fadd_rn(p2_0, p3_0), p2_1), p3_1);
            b0 = 0.0f; b1 = 0.0f;
        } else if (sm >= 20 && sm <= 23) {
            a0 = reduce_bl4_half(p0_0, p1_0, p0_1, p1_1, grp, tid, sm);
            a1 = reduce_bl4_half(p2_0, p3_0, p2_1, p3_1, grp, tid, sm);
            b0 = 0.0f; b1 = 0.0f;
        } else {
            a0 = reduce_p_intra_scalar(p0_0, p1_0, p0_1, p1_1, grp, sm);
            a1 = reduce_p_intra_scalar(p2_0, p3_0, p2_1, p3_1, grp, sm);
            b0 = 0.0f; b1 = 0.0f;
        }
    }
}

// Helper: exact 32-lane butterfly `shfl 16 -> 8 -> 4 -> 2 -> 1` over `Red_smem[off .. off+31][row]`
static __device__ __forceinline__ float bfly32_from_64x16(
    const float Red_smem[64][16], int off, int row, int tid
) {
    float e = __fadd_rn(__fadd_rn(Red_smem[off + 0 + tid][row],  Red_smem[off + 16 + tid][row]),
                        __fadd_rn(Red_smem[off + 8 + tid][row],  Red_smem[off + 24 + tid][row]));
    float o = __fadd_rn(__fadd_rn(Red_smem[off + 4 + tid][row],  Red_smem[off + 20 + tid][row]),
                        __fadd_rn(Red_smem[off + 12 + tid][row], Red_smem[off + 28 + tid][row]));
    e = __fadd_rn(e, __shfl_xor_sync(0xFFFFFFFFu, e, 2));
    o = __fadd_rn(o, __shfl_xor_sync(0xFFFFFFFFu, o, 2));
    e = __fadd_rn(e, __shfl_xor_sync(0xFFFFFFFFu, e, 1));
    o = __fadd_rn(o, __shfl_xor_sync(0xFFFFFFFFu, o, 1));
    return __fadd_rn(e, o);
}

// Phase 2 for FUSE_MT=0 (Red_smem[64][16] = 4 KiB -> ZERO extra __syncthreads() across ALL 32 modes,
//                        total CTA SMEM = 43.5 KiB -> 2 CTAs/SM = 87 KiB <= 101 KiB!)
template <int WINDOW, int TREE_MODE>
static __device__ __forceinline__ void reduce_p_phase2_64x16(
    float Red_smem[64][16],
    float p0_0, float p1_0, float p0_1, float p1_1,
    float p2_0, float p3_0, float p2_1, float p3_1,
    float a0, float b0, float a1, float b1,
    int wid, int grp, int tid, int tree_mode_rt,
    float& sum_p0, float& sum_p1
) {
    if constexpr (TREE_MODE == 0) {
        (void)tree_mode_rt;
        if constexpr (WINDOW < 0) {
            (void)p0_0; (void)p1_0; (void)p0_1; (void)p1_1;
            (void)p2_0; (void)p3_0; (void)p2_1; (void)p3_1;
            (void)b0; (void)b1;
            if (tid == 0) {
                Red_smem[wid][grp]     = a0;
                Red_smem[wid][grp + 8] = a1;
            }
            __syncthreads();
            sum_p0 = __fadd_rn(__fadd_rn(Red_smem[0][grp],     Red_smem[2][grp]),
                               __fadd_rn(Red_smem[1][grp],     Red_smem[3][grp]));
            sum_p1 = __fadd_rn(__fadd_rn(Red_smem[0][grp + 8], Red_smem[2][grp + 8]),
                               __fadd_rn(Red_smem[1][grp + 8], Red_smem[3][grp + 8]));
        } else {
            (void)a0; (void)b0; (void)a1; (void)b1;
            // Exact live [local] Blocked([1,1],[1,32],[2,2]): half_lo=bfly32(p[0..31]), half_hi=bfly32(p[32..63])
            Red_smem[wid * 8 + tid][grp]              = p0_0;
            Red_smem[wid * 8 + 4 + tid][grp]          = p1_0;
            Red_smem[32 + wid * 8 + tid][grp]         = p0_1;
            Red_smem[32 + wid * 8 + 4 + tid][grp]     = p1_1;
            Red_smem[wid * 8 + tid][grp + 8]          = p2_0;
            Red_smem[wid * 8 + 4 + tid][grp + 8]      = p3_0;
            Red_smem[32 + wid * 8 + tid][grp + 8]     = p2_1;
            Red_smem[32 + wid * 8 + 4 + tid][grp + 8] = p3_1;
            __syncthreads();
            sum_p0 = __fadd_rn(bfly32_from_64x16(Red_smem, 0, grp, tid),
                               bfly32_from_64x16(Red_smem, 32, grp, tid));
            sum_p1 = __fadd_rn(bfly32_from_64x16(Red_smem, 0, grp + 8, tid),
                               bfly32_from_64x16(Red_smem, 32, grp + 8, tid));
        }
    } else {
        const int sm = tree_mode_rt & 0x1F;
        if (sm == 14) {
            Red_smem[wid * 8 + tid][grp]              = p0_0;
            Red_smem[wid * 8 + 4 + tid][grp]          = p1_0;
            Red_smem[32 + wid * 8 + tid][grp]         = p0_1;
            Red_smem[32 + wid * 8 + 4 + tid][grp]     = p1_1;
            Red_smem[wid * 8 + tid][grp + 8]          = p2_0;
            Red_smem[wid * 8 + 4 + tid][grp + 8]      = p3_0;
            Red_smem[32 + wid * 8 + tid][grp + 8]     = p2_1;
            Red_smem[32 + wid * 8 + 4 + tid][grp + 8] = p3_1;
            __syncthreads();
            sum_p0 = __fadd_rn(bfly32_from_64x16(Red_smem, 0, grp, tid),
                               bfly32_from_64x16(Red_smem, 32, grp, tid));
            sum_p1 = __fadd_rn(bfly32_from_64x16(Red_smem, 0, grp + 8, tid),
                               bfly32_from_64x16(Red_smem, 32, grp + 8, tid));
        } else if (sm == 15) {
            // BL2_2x2: Blocked([1,2],[2,16],[2,2])
            Red_smem[wid * 4 + tid][grp]          = a0;
            Red_smem[16 + wid * 4 + tid][grp]     = b0;
            Red_smem[wid * 4 + tid][grp + 8]      = a1;
            Red_smem[16 + wid * 4 + tid][grp + 8] = b1;
            __syncthreads();
            auto fin_bl2_half = [&](int off, int row) -> float {
                float s = __fadd_rn(__fadd_rn(Red_smem[off + 0 + tid][row],  Red_smem[off + 8 + tid][row]),
                                    __fadd_rn(Red_smem[off + 4 + tid][row],  Red_smem[off + 12 + tid][row]));
                s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
                s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
                return s;
            };
            sum_p0 = __fadd_rn(fin_bl2_half(0, grp),     fin_bl2_half(16, grp));
            sum_p1 = __fadd_rn(fin_bl2_half(0, grp + 8), fin_bl2_half(16, grp + 8));
        } else if (sm == 16 || sm == 30 || sm == 31) {
            Red_smem[wid * 8 + tid][grp]         = a0;
            Red_smem[wid * 8 + 4 + tid][grp]     = b0;
            Red_smem[wid * 8 + tid][grp + 8]     = a1;
            Red_smem[wid * 8 + 4 + tid][grp + 8] = b1;
            __syncthreads();
            auto fin_bl1 = [&](int row) -> float {
                float e0 = Red_smem[0 + tid][row],  e1 = Red_smem[8 + tid][row];
                float e2 = Red_smem[16 + tid][row], e3 = Red_smem[24 + tid][row];
                float o0 = Red_smem[4 + tid][row],  o1 = Red_smem[12 + tid][row];
                float o2 = Red_smem[20 + tid][row], o3 = Red_smem[28 + tid][row];
                float ee, oo;
                if (sm == 30) {
                    ee = __fadd_rn(__fadd_rn(e0, e1), __fadd_rn(e2, e3));
                    oo = __fadd_rn(__fadd_rn(o0, o1), __fadd_rn(o2, o3));
                } else if (sm == 31) {
                    ee = __fadd_rn(__fadd_rn(__fadd_rn(e0, e1), e2), e3);
                    oo = __fadd_rn(__fadd_rn(__fadd_rn(o0, o1), o2), o3);
                } else {
                    ee = __fadd_rn(__fadd_rn(e0, e2), __fadd_rn(e1, e3));
                    oo = __fadd_rn(__fadd_rn(o0, o2), __fadd_rn(o1, o3));
                }
                ee = __fadd_rn(ee, __shfl_xor_sync(0xFFFFFFFFu, ee, 2));
                oo = __fadd_rn(oo, __shfl_xor_sync(0xFFFFFFFFu, oo, 2));
                ee = __fadd_rn(ee, __shfl_xor_sync(0xFFFFFFFFu, ee, 1));
                oo = __fadd_rn(oo, __shfl_xor_sync(0xFFFFFFFFu, oo, 1));
                return __fadd_rn(ee, oo);
            };
            sum_p0 = fin_bl1(grp);
            sum_p1 = fin_bl1(grp + 8);
        } else if (sm >= 17 && sm <= 19) {
            Red_smem[wid * 4 + tid][grp]     = a0;
            Red_smem[wid * 4 + tid][grp + 8] = a1;
            __syncthreads();
            auto fin_bl2 = [&](int row) -> float {
                float s = __fadd_rn(__fadd_rn(Red_smem[0 + tid][row],  Red_smem[8 + tid][row]),
                                    __fadd_rn(Red_smem[4 + tid][row],  Red_smem[12 + tid][row]));
                s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
                s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
                return s;
            };
            sum_p0 = fin_bl2(grp);
            sum_p1 = fin_bl2(grp + 8);
        } else if (sm >= 20 && sm <= 23) {
            if ((tid & 1) == 0) {
                const int h = tid >> 1;
                Red_smem[wid * 2 + h][grp]     = a0;
                Red_smem[wid * 2 + h][grp + 8] = a1;
            }
            __syncthreads();
            auto fin_bl4 = [&](int row) -> float {
                float s0 = __fadd_rn(__fadd_rn(Red_smem[0][row], Red_smem[4][row]),
                                     __fadd_rn(Red_smem[2][row], Red_smem[6][row]));
                float s1 = __fadd_rn(__fadd_rn(Red_smem[1][row], Red_smem[5][row]),
                                     __fadd_rn(Red_smem[3][row], Red_smem[7][row]));
                return __fadd_rn(s0, s1);
            };
            sum_p0 = fin_bl4(grp);
            sum_p1 = fin_bl4(grp + 8);
        } else {
            if (tid == 0) {
                Red_smem[wid][grp]     = a0;
                Red_smem[wid][grp + 8] = a1;
            }
            __syncthreads();
            sum_p0 = reduce_p_cross_scalar(Red_smem[0][grp], Red_smem[1][grp], Red_smem[2][grp], Red_smem[3][grp], sm);
            sum_p1 = reduce_p_cross_scalar(Red_smem[0][grp + 8], Red_smem[1][grp + 8], Red_smem[2][grp + 8], Red_smem[3][grp + 8], sm);
        }
    }
}

// Phase 2 for FUSE_MT=1 (Red_smem[32][8] = 1,024 B -> keeps 2 CTAs/SM <= 101,376 B even at MT_MAX=3!)
template <int WINDOW, int TREE_MODE>
static __device__ __forceinline__ void reduce_p_phase2_32x8(
    float Red_smem[32][8],
    float p0_0, float p1_0, float p0_1, float p1_1,
    float p2_0, float p3_0, float p2_1, float p3_1,
    float a0, float b0, float a1, float b1,
    int wid, int grp, int tid, int tree_mode_rt,
    float& sum_p0, float& sum_p1
) {
    auto bfly32_half = [&](float u, float v) -> float {
        Red_smem[wid * 8 + tid][grp]     = u;
        Red_smem[wid * 8 + 4 + tid][grp] = v;
        __syncthreads();
        float ee = __fadd_rn(__fadd_rn(Red_smem[tid][grp],     Red_smem[16 + tid][grp]),
                             __fadd_rn(Red_smem[8 + tid][grp], Red_smem[24 + tid][grp]));
        float oo = __fadd_rn(__fadd_rn(Red_smem[4 + tid][grp],     Red_smem[20 + tid][grp]),
                             __fadd_rn(Red_smem[12 + tid][grp],    Red_smem[28 + tid][grp]));
        ee = __fadd_rn(ee, __shfl_xor_sync(0xFFFFFFFFu, ee, 2));
        oo = __fadd_rn(oo, __shfl_xor_sync(0xFFFFFFFFu, oo, 2));
        ee = __fadd_rn(ee, __shfl_xor_sync(0xFFFFFFFFu, ee, 1));
        oo = __fadd_rn(oo, __shfl_xor_sync(0xFFFFFFFFu, oo, 1));
        return __fadd_rn(ee, oo);
    };

    if constexpr (TREE_MODE == 0) {
        (void)tree_mode_rt;
        if constexpr (WINDOW < 0) {
            (void)p0_0; (void)p1_0; (void)p0_1; (void)p1_1;
            (void)p2_0; (void)p3_0; (void)p2_1; (void)p3_1;
            (void)b0; (void)b1; (void)bfly32_half;
            if (tid == 0) {
                Red_smem[wid][grp]     = a0;
                Red_smem[wid + 4][grp] = a1;
            }
            __syncthreads();
            sum_p0 = __fadd_rn(__fadd_rn(Red_smem[0][grp], Red_smem[2][grp]),
                               __fadd_rn(Red_smem[1][grp], Red_smem[3][grp]));
            sum_p1 = __fadd_rn(__fadd_rn(Red_smem[4][grp], Red_smem[6][grp]),
                               __fadd_rn(Red_smem[5][grp], Red_smem[7][grp]));
        } else {
            (void)a0; (void)b0; (void)a1; (void)b1;
            float lo0 = bfly32_half(p0_0, p1_0); __syncthreads();
            float hi0 = bfly32_half(p0_1, p1_1); __syncthreads();
            sum_p0 = __fadd_rn(lo0, hi0);
            float lo1 = bfly32_half(p2_0, p3_0); __syncthreads();
            float hi1 = bfly32_half(p2_1, p3_1);
            sum_p1 = __fadd_rn(lo1, hi1);
        }
    } else {
        const int sm = tree_mode_rt & 0x1F;
        if (sm == 14) {
            float lo0 = bfly32_half(p0_0, p1_0); __syncthreads();
            float hi0 = bfly32_half(p0_1, p1_1); __syncthreads();
            sum_p0 = __fadd_rn(lo0, hi0);
            float lo1 = bfly32_half(p2_0, p3_0); __syncthreads();
            float hi1 = bfly32_half(p2_1, p3_1);
            sum_p1 = __fadd_rn(lo1, hi1);
        } else if (sm == 15) {
            auto run_bl2_2x2 = [&](float u, float v) -> float {
                Red_smem[wid * 4 + tid][grp]      = u;
                Red_smem[16 + wid * 4 + tid][grp] = v;
                __syncthreads();
                float s0 = __fadd_rn(__fadd_rn(Red_smem[0 + tid][grp],  Red_smem[8 + tid][grp]),
                                     __fadd_rn(Red_smem[4 + tid][grp],  Red_smem[12 + tid][grp]));
                float s1 = __fadd_rn(__fadd_rn(Red_smem[16 + tid][grp], Red_smem[24 + tid][grp]),
                                     __fadd_rn(Red_smem[20 + tid][grp], Red_smem[28 + tid][grp]));
                s0 = __fadd_rn(s0, __shfl_xor_sync(0xFFFFFFFFu, s0, 2));
                s1 = __fadd_rn(s1, __shfl_xor_sync(0xFFFFFFFFu, s1, 2));
                s0 = __fadd_rn(s0, __shfl_xor_sync(0xFFFFFFFFu, s0, 1));
                s1 = __fadd_rn(s1, __shfl_xor_sync(0xFFFFFFFFu, s1, 1));
                return __fadd_rn(s0, s1);
            };
            sum_p0 = run_bl2_2x2(a0, b0); __syncthreads();
            sum_p1 = run_bl2_2x2(a1, b1);
        } else if (sm == 16 || sm == 30 || sm == 31) {
            auto run_bl1 = [&](float a, float b) -> float {
                Red_smem[wid * 8 + tid][grp]     = a;
                Red_smem[wid * 8 + 4 + tid][grp] = b;
                __syncthreads();
                float e0 = Red_smem[0 + tid][grp],  e1 = Red_smem[8 + tid][grp];
                float e2 = Red_smem[16 + tid][grp], e3 = Red_smem[24 + tid][grp];
                float o0 = Red_smem[4 + tid][grp],  o1 = Red_smem[12 + tid][grp];
                float o2 = Red_smem[20 + tid][grp], o3 = Red_smem[28 + tid][grp];
                float ee, oo;
                if (sm == 30) {
                    ee = __fadd_rn(__fadd_rn(e0, e1), __fadd_rn(e2, e3));
                    oo = __fadd_rn(__fadd_rn(o0, o1), __fadd_rn(o2, o3));
                } else if (sm == 31) {
                    ee = __fadd_rn(__fadd_rn(__fadd_rn(e0, e1), e2), e3);
                    oo = __fadd_rn(__fadd_rn(__fadd_rn(o0, o1), o2), o3);
                } else {
                    ee = __fadd_rn(__fadd_rn(e0, e2), __fadd_rn(e1, e3));
                    oo = __fadd_rn(__fadd_rn(o0, o2), __fadd_rn(o1, o3));
                }
                ee = __fadd_rn(ee, __shfl_xor_sync(0xFFFFFFFFu, ee, 2));
                oo = __fadd_rn(oo, __shfl_xor_sync(0xFFFFFFFFu, oo, 2));
                ee = __fadd_rn(ee, __shfl_xor_sync(0xFFFFFFFFu, ee, 1));
                oo = __fadd_rn(oo, __shfl_xor_sync(0xFFFFFFFFu, oo, 1));
                return __fadd_rn(ee, oo);
            };
            sum_p0 = run_bl1(a0, b0);
            __syncthreads();
            sum_p1 = run_bl1(a1, b1);
        } else if (sm >= 17 && sm <= 19) {
            Red_smem[wid * 4 + tid][grp]      = a0;
            Red_smem[16 + wid * 4 + tid][grp] = a1;
            __syncthreads();
            auto fin_bl2 = [&](int off) -> float {
                float s = __fadd_rn(__fadd_rn(Red_smem[off + tid][grp],     Red_smem[off + 8 + tid][grp]),
                                    __fadd_rn(Red_smem[off + 4 + tid][grp], Red_smem[off + 12 + tid][grp]));
                s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
                s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
                return s;
            };
            sum_p0 = fin_bl2(0);
            sum_p1 = fin_bl2(16);
        } else if (sm >= 20 && sm <= 23) {
            if ((tid & 1) == 0) {
                const int h = tid >> 1;
                Red_smem[wid * 2 + h][grp]     = a0;
                Red_smem[8 + wid * 2 + h][grp] = a1;
            }
            __syncthreads();
            auto fin_bl4 = [&](int off) -> float {
                float s0 = __fadd_rn(__fadd_rn(Red_smem[off + 0][grp], Red_smem[off + 4][grp]),
                                     __fadd_rn(Red_smem[off + 2][grp], Red_smem[off + 6][grp]));
                float s1 = __fadd_rn(__fadd_rn(Red_smem[off + 1][grp], Red_smem[off + 5][grp]),
                                     __fadd_rn(Red_smem[off + 3][grp], Red_smem[off + 7][grp]));
                return __fadd_rn(s0, s1);
            };
            sum_p0 = fin_bl4(0);
            sum_p1 = fin_bl4(8);
        } else {
            if (tid == 0) {
                Red_smem[wid][grp]     = a0;
                Red_smem[wid + 4][grp] = a1;
            }
            __syncthreads();
            sum_p0 = reduce_p_cross_scalar(Red_smem[0][grp], Red_smem[1][grp], Red_smem[2][grp], Red_smem[3][grp], sm);
            sum_p1 = reduce_p_cross_scalar(Red_smem[4][grp], Red_smem[5][grp], Red_smem[6][grp], Red_smem[7][grp], sm);
        }
    }
}

static __device__ __forceinline__ void compute_otf_bias_tile(
    const uint16_t* __restrict__ RelProjT,
    uint16_t* __restrict__ smem_b,
    uint32_t r_a0, uint32_t r_a1, uint32_t r_a2, uint32_t r_a3,
    int d_base, int d_hi, int E,
    int wid, int grp, int tid
) {
    const int n_d = min(E, d_hi + 1) - d_base;
    const int n_nt = (n_d + 7) >> 3;  // <= 10 N-tiles -> 2-3 per warp
    for (int nt = wid; nt < n_nt; nt += 4) {
        const int d_col = d_base + nt * 8 + grp;
        uint32_t b0 = 0u, b1 = 0u;
        if (d_col < E) {
            const uint16_t* rp = RelProjT + static_cast<int64_t>(d_col) * D_REL;
            b0 = *reinterpret_cast<const uint32_t*>(rp + 2 * tid);
            b1 = *reinterpret_cast<const uint32_t*>(rp + 8 + 2 * tid);
        }
        float bd0 = 0.f, bd1 = 0.f, bd2 = 0.f, bd3 = 0.f;
        mma_m16n8k16(bd0, bd1, bd2, bd3, r_a0, r_a1, r_a2, r_a3, b0, b1);
        const int c_off = nt * 8 + 2 * tid;
        *reinterpret_cast<uint32_t*>(smem_b + grp * B_COLS + c_off) =
            pack_bf16x2(f32_to_bf16(__fadd_rn(bd0, 0.f)), f32_to_bf16(__fadd_rn(bd1, 0.f)));
        *reinterpret_cast<uint32_t*>(smem_b + (grp + 8) * B_COLS + c_off) =
            pack_bf16x2(f32_to_bf16(__fadd_rn(bd2, 0.f)), f32_to_bf16(__fadd_rn(bd3, 0.f)));
    }
}

// Direct register-to-BF16 Gate+Sink epilogue for NUM_SPLITS == 1 (ZERO OP/LP/SEM in TREE_MODE==0!).
// In diagnostic mode (TREE_MODE != 0), also dumps FP32 OP/LP if non-null for stage-by-stage audit!
template <int TREE_MODE>
static __device__ __forceinline__ void store_direct_gate_sink_row(
    bool row_ok, int tok, int h, int NT,
    float m_i, float l_i,
    float a0, float a1, float a2, float a3,
    float a4, float a5, float a6, float a7,
    const uint16_t* __restrict__ GATE, const uint16_t* __restrict__ SINK,
    float* __restrict__ OP, float* __restrict__ LP,
    uint16_t* __restrict__ Y,
    int64_t so_t, int64_t so_h, int64_t sl_t, int64_t sl_h,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int wid, int tid
) {
    if (!row_ok) return;
    const bool tv = (tok < NT);
    const float l_safe = (l_i > 0.0f) ? l_i : 1.0f;
    const float lse = tv
        ? ((l_i > 0.0f) ? __fmul_rn(__fadd_rn(m_i, log2f(l_safe)), LN2) : -INFINITY)
        : 0.0f;
    const float sink_f = bf16_to_f32(SINK[static_cast<int64_t>(tok) * ssk_t + h]);
    const float sig_s = triton_sigmoid(__fsub_rn(lse, sink_f));

    if constexpr (TREE_MODE != 0) {
        if (LP != nullptr && wid == 0 && tid == 0) {
            LP[static_cast<int64_t>(tok) * sl_t + static_cast<int64_t>(h) * sl_h] = lse;
        }
    } else {
        (void)OP; (void)LP; (void)so_t; (void)so_h; (void)sl_t; (void)sl_h;
    }

    const uint16_t* gp = GATE + static_cast<int64_t>(tok) * sg_t + h * D_DIM + wid * 32;
    uint16_t* yp       = Y    + static_cast<int64_t>(tok) * sy_t + h * D_DIM + wid * 32;
    float* op_row = nullptr;
    if constexpr (TREE_MODE != 0) {
        if (OP != nullptr) {
            op_row = OP + static_cast<int64_t>(tok) * so_t + static_cast<int64_t>(h) * so_h + wid * 32;
        }
    } else {
        (void)op_row;
    }
    const float vals[8] = {a0, a1, a2, a3, a4, a5, a6, a7};

    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int c = nt * 8 + 2 * tid;
        float g0, g1;
        if ((reinterpret_cast<uintptr_t>(gp + c) & 3u) == 0u) {
            uint32_t gv = *reinterpret_cast<const uint32_t*>(gp + c);
            g0 = bf16_to_f32(static_cast<uint16_t>(gv & 0xFFFFu));
            g1 = bf16_to_f32(static_cast<uint16_t>(gv >> 16));
        } else {
            g0 = bf16_to_f32(gp[c]);
            g1 = bf16_to_f32(gp[c + 1]);
        }
        float x0 = tv ? div_full_f32(vals[2 * nt],     l_safe) : 0.0f;
        float x1 = tv ? div_full_f32(vals[2 * nt + 1], l_safe) : 0.0f;
        if constexpr (TREE_MODE != 0) {
            if (op_row != nullptr) {
                op_row[c]     = x0;
                op_row[c + 1] = x1;
            }
        }
        uint16_t y0 = f32_to_bf16(__fmul_rn(__fmul_rn(x0, triton_sigmoid(g0)), sig_s));
        uint16_t y1 = f32_to_bf16(__fmul_rn(__fmul_rn(x1, triton_sigmoid(g1)), sig_s));
        *reinterpret_cast<uint32_t*>(yp + c) = pack_bf16x2(y0, y1);
    }
}

// Cooperative last-split MergeGateSink for NUM_SPLITS > 1 across `n_rows` rows (`16` or `MT*16`).
// Matches `ns_mgs::merge_gate_sink_exact_kernel` (`inkling_target_attn_ext_s7.cu`) op-for-op
// with L1-bypassing `ld.global.cg` loads for strict SM121a cross-SM ping-pong L2 coherence!
template <int NUM_SPLITS, int TREE_MODE>
static __device__ __forceinline__ void cooperative_merge_gate_sink(
    int n_rows,
    const uint8_t* Row_ok, const int32_t* Row_tok, const int32_t* Row_h,
    const float* __restrict__ OP, const float* __restrict__ LP,
    const uint16_t* __restrict__ GATE, const uint16_t* __restrict__ SINK,
    uint16_t* __restrict__ Y,
    int64_t so_s, int64_t so_t, int64_t so_h,
    int64_t sl_s, int64_t sl_t, int64_t sl_h,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int NT, int wid, int lane, int tree_mode_rt
) {
    if constexpr (TREE_MODE == 0) {
        (void)tree_mode_rt;
    }
    const int d0 = lane * 4;
    for (int ri = wid; ri < n_rows; ri += 4) {
        if (!Row_ok[ri]) continue;
        const int tok = Row_tok[ri];
        const int h = Row_h[ri];
        const int col = h * D_DIM + d0;

        float g_f[4];
        {
            const uint16_t* gp = GATE + static_cast<int64_t>(tok) * sg_t + col;
            if ((reinterpret_cast<uintptr_t>(gp) & 7u) == 0u) {
                uint2 v = *reinterpret_cast<const uint2*>(gp);
                g_f[0] = bf16_to_f32(static_cast<uint16_t>(v.x & 0xFFFFu));
                g_f[1] = bf16_to_f32(static_cast<uint16_t>(v.x >> 16));
                g_f[2] = bf16_to_f32(static_cast<uint16_t>(v.y & 0xFFFFu));
                g_f[3] = bf16_to_f32(static_cast<uint16_t>(v.y >> 16));
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) g_f[i] = bf16_to_f32(gp[i]);
            }
        }
        const float sink_f = bf16_to_f32(SINK[static_cast<int64_t>(tok) * ssk_t + h]);
        float sig_g[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) sig_g[i] = triton_sigmoid(g_f[i]);

        const bool tv = (tok < NT);
        float x[4] = {0.f, 0.f, 0.f, 0.f};
        float lse_v = 0.0f;
        if (tv) {
            const float* lp_base = LP + static_cast<int64_t>(tok) * sl_t + static_cast<int64_t>(h) * sl_h;
            float lse[NUM_SPLITS];
            float mx = -INFINITY;
            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; ++s) {
                float v = ld_cg_f32(lp_base + static_cast<int64_t>(s) * sl_s);
                lse[s] = v;
                mx = fmaxf(mx, v);
            }
            const float mx_safe = (mx == -INFINITY) ? 0.0f : mx;
            float w[NUM_SPLITS];
            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; ++s) {
                float arg = __fmul_rn(__fsub_rn(lse[s], mx_safe), LOG2E);
                if constexpr (TREE_MODE != 0) {
                    const bool mgs_ftz = (((tree_mode_rt >> 8) & 1) != 0);
                    w[s] = mgs_ftz ? ex2_approx_ftz_f32(arg) : ex2_approx_f32(arg);
                } else {
                    w[s] = ex2_approx_f32(arg);
                }
            }
            const float tot = reduce_splits_sum<NUM_SPLITS>(w);
            const float tot_safe = (tot > 0.0f) ? tot : 1.0f;

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            const float* op_base = OP + static_cast<int64_t>(tok) * so_t + static_cast<int64_t>(h) * so_h + d0;
            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; ++s) {
                if (lse[s] > -INFINITY) {
                    float4 ov = ld_cg_f4(op_base + static_cast<int64_t>(s) * so_s);
                    float ws = w[s];
                    acc0 = fmaf(ws, ov.x, acc0);
                    acc1 = fmaf(ws, ov.y, acc1);
                    acc2 = fmaf(ws, ov.z, acc2);
                    acc3 = fmaf(ws, ov.w, acc3);
                }
            }
            x[0] = div_full_f32(acc0, tot_safe);
            x[1] = div_full_f32(acc1, tot_safe);
            x[2] = div_full_f32(acc2, tot_safe);
            x[3] = div_full_f32(acc3, tot_safe);
            lse_v = (tot > 0.0f) ? __fadd_rn(mx_safe, logf(tot_safe)) : -INFINITY;
        }
        const float sig_s = triton_sigmoid(__fsub_rn(lse_v, sink_f));
        uint16_t out_b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) out_b[i] = f32_to_bf16(__fmul_rn(__fmul_rn(x[i], sig_g[i]), sig_s));
        uint2 out_v;
        out_v.x = pack_bf16x2(out_b[0], out_b[1]);
        out_v.y = pack_bf16x2(out_b[2], out_b[3]);
        *reinterpret_cast<uint2*>(Y + static_cast<int64_t>(tok) * sy_t + col) = out_v;
    }
}

// =========================================================================
// KERNEL 2A (FUSE_MT = 0): 1 CTA per (mt, req, split), BM=16, 38.6 KB SMEM
//   Supports:
//     - Linear C=1 (`S=8/16`) & C=32 (`S=1/2/4`) when `HAS_TREE_2B == false`
//     - S8-P6v2 4-tile BM=16 Beam-2 Tree (`mt=0..2` primary, `mt=3` secondary `ws<=2`)
//       when `HAS_TREE_2B == true`!
// =========================================================================
template <int WINDOW, int NUM_SPLITS, bool HAS_TAU, bool HAS_NLIVE,
          bool USE_PDL, bool TRIG_WO, int TREE_MODE, bool HAS_TREE_2B = false>
__global__ void __launch_bounds__(128, 2)
fused_decode_attn_otf_mgs_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RScale,
    const float*    __restrict__ Tau,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const int32_t*  __restrict__ BT,
    const int32_t*  __restrict__ SEQ,
    const int32_t*  __restrict__ CUQ,
    const uint16_t* __restrict__ GATE,
    const uint16_t* __restrict__ SINK,
    float*          __restrict__ OP,
    float*          __restrict__ LP,
    int32_t*        __restrict__ SEM,
    uint16_t*       __restrict__ Y,
    const int32_t*  __restrict__ NLivePtr,
    const int32_t*  __restrict__ TreeMeta,
    int64_t sq_t, int64_t sq_h,
    int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h,
    int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int BS, int BS_shift, int E, int G, int tree_mode_rt
) {
    (void)T;
    const int mt = blockIdx.x;
    const int r = blockIdx.y;
    const int kvh = blockIdx.z / NUM_SPLITS;
    const int split = blockIdx.z % NUM_SPLITS;
    const int sem_idx = (r * gridDim.x + mt) * (gridDim.z / NUM_SPLITS) + kvh;

    const int tid_all = threadIdx.x;
    const int wid = tid_all >> 5;
    const int lane = tid_all & 31;
    const int grp = lane >> 2;
    const int tid = lane & 3;
    const int sub = lane >> 3;
    const int r_in = lane & 7;

    const int q0 = CUQ[r];
    const int qlen = CUQ[r + 1] - q0;
    const int seqlen = SEQ[r];
    const int past = seqlen - qlen;
    int r_live = qlen;
    if constexpr (HAS_NLIVE) {
        int nl = *NLivePtr;
        int d = nl - q0;
        if (d < 0) d = 0;
        if (d < r_live) r_live = d;
    }

    int tree_wp = r_live - 1;
    int tree_ws = 0;
    bool is_sec_tile = false;
    if constexpr (HAS_TREE_2B) {
        const int t_act = TreeMeta[0];
        const int t_wp  = TreeMeta[1];
        const int t_ws  = TreeMeta[2];
        const int t_nl  = TreeMeta[3];
        if (t_act != 0 && t_ws > 0) {
            tree_wp = min(max(t_wp, 0), qlen - 1);
            tree_ws = min(t_ws, min(2, qlen - 1 - tree_wp));
            r_live  = min(max(t_nl, 1), qlen);
            is_sec_tile = (mt == 3);
        } else {
            tree_wp = r_live - 1;
            tree_ws = 0;
            is_sec_tile = false;
        }
        const bool skip_tile = is_sec_tile ? (tree_ws <= 0) : (mt >= 3 || (mt * BM) / G > tree_wp);
        if (skip_tile) {
            if constexpr (USE_PDL) {
                gdc_wait();
                if constexpr (TRIG_WO) {
                    if (tid_all == 0) gdc_launch_dependents();
                }
            }
            return;
        }
    } else {
        (void)TreeMeta;
        (void)tree_wp;
        (void)tree_ws;
        (void)is_sec_tile;
        if ((mt * BM) / G >= r_live) {
            if constexpr (USE_PDL) {
                gdc_wait();
                if constexpr (TRIG_WO) {
                    if (tid_all == 0) gdc_launch_dependents();
                }
            }
            return;
        }
    }

    const int k_end = max(past + qlen, 0);
    const int k_beg = (WINDOW >= 0) ? max(past - WINDOW, 0) : 0;
    const int span = max(k_end - k_beg, 0);
    const int per = (((span + NUM_SPLITS - 1) / NUM_SPLITS) + 63) / 64 * 64;
    const int s_beg = k_beg + split * per;
    const int s_end = min(s_beg + per, k_end);
    int s_end_live = (seqlen > 0 && q0 < NT) ? s_end : s_beg;
    if constexpr (HAS_TREE_2B) {
        const int max_k = is_sec_tile ? (past + 1 + tree_ws) : (past + 1 + tree_wp);
        if (max_k < s_end_live) s_end_live = max_k;
    } else if constexpr (HAS_NLIVE) {
        if (past + r_live < s_end_live) s_end_live = past + r_live;
    }

    extern __shared__ uint4 smem_raw_u4[];
    uint8_t*  smem_raw = reinterpret_cast<uint8_t*>(smem_raw_u4);
    uint16_t* smem_kv0 = reinterpret_cast<uint16_t*>(smem_raw);             // 16,384 B
    uint16_t* smem_kv1 = reinterpret_cast<uint16_t*>(smem_raw + 16384);     // 16,384 B
    uint16_t* smem_b   = reinterpret_cast<uint16_t*>(smem_raw + 32768);     //  2,560 B
    uint16_t* smem_p   = reinterpret_cast<uint16_t*>(smem_raw + 35328);     //  2,304 B
    uint16_t* smem_r   = reinterpret_cast<uint16_t*>(smem_raw + 37632);     //    512 B

    alignas(16) __shared__ float    Red_smem[64][16];
    alignas(16) __shared__ int32_t  Row_tok[16];
    alignas(16) __shared__ int32_t  Row_h[16];
    alignas(16) __shared__ int32_t  Row_qpos[16];
    alignas(16) __shared__ uint8_t  Row_ok[16];
    alignas(16) __shared__ int2     KV_blk_slot[64];

    if (tid_all < 16) {
        if constexpr (HAS_TREE_2B) {
            if (is_sec_tile) {
                int sec_off = tid_all / G;
                int g = tid_all % G;
                bool ok = (sec_off < tree_ws);
                Row_ok[tid_all] = ok ? 1 : 0;
                Row_tok[tid_all] = ok ? (q0 + 1 + tree_wp + sec_off) : (q0 + 1 + tree_wp);
                Row_h[tid_all] = kvh * G + g;
                Row_qpos[tid_all] = past + 1 + (ok ? sec_off : (tree_ws - 1));
            } else {
                int m = mt * BM + tid_all;
                int i_loc = m / G;
                int g = m % G;
                bool ok = (i_loc <= tree_wp);
                Row_ok[tid_all] = ok ? 1 : 0;
                Row_tok[tid_all] = q0 + i_loc;
                Row_h[tid_all] = kvh * G + g;
                Row_qpos[tid_all] = past + i_loc;
            }
        } else {
            int m = mt * BM + tid_all;
            int i_loc = m / G;
            int g = m % G;
            bool ok = (i_loc < r_live);
            Row_ok[tid_all] = ok ? 1 : 0;
            Row_tok[tid_all] = q0 + i_loc;
            Row_h[tid_all] = kvh * G + g;
            Row_qpos[tid_all] = past + i_loc;
        }
    }

    const int32_t* bt_row = BT + static_cast<int64_t>(r) * sbt_r;
    const uint16_t* K_base = K + static_cast<int64_t>(kvh) * sk_h;
    const uint16_t* V_base = V + static_cast<int64_t>(kvh) * sv_h;
    const uint32_t kv0_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv0));
    const uint32_t kv1_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv1));
    uint4* smem_kv0_4 = reinterpret_cast<uint4*>(smem_kv0);
    uint4* smem_kv1_4 = reinterpret_cast<uint4*>(smem_kv1);

    // Pre-gdc_wait past-KV async prefetch (safe for BOTH linear and HAS_TREE_2B since kpos < past!)
    const bool tile0_is_past = (s_beg < s_end_live) && ((s_beg + BN) <= past);
    if (tile0_is_past) {
        if (tid_all < 64) {
            int kpos = s_beg + tid_all;
            int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
            int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
            KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
        }
        __syncthreads();
        issue_kv64_swiz_async<false>(K_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, 0ULL, tid_all);
        cp_async_commit();
    } else {
        __syncthreads();
    }

    if constexpr (USE_PDL) {
        gdc_wait();
        if constexpr (TRIG_WO) {
            if (tid_all == 0) gdc_launch_dependents();
        }
    }

    const bool ok0 = Row_ok[grp] != 0;
    const bool ok1 = Row_ok[grp + 8] != 0;
    float m_i0 = -INFINITY, m_i1 = -INFINITY;
    float l_i0 = 0.0f, l_i1 = 0.0f;
    float acc[4][4] = {{0.f}};

    if (s_beg >= s_end_live) {
        if constexpr (NUM_SPLITS > 1) {
            if (tid_all < 16 && Row_ok[tid_all]) {
                LP[split * sl_s + Row_tok[tid_all] * sl_t + Row_h[tid_all] * sl_h] = -INFINITY;
            }
        }
    } else {
        {
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                int idx = pass * 128 + tid_all;
                int row = idx >> 4;
                int c = idx & 15;
                int swiz_c = c ^ (row & 7);
                uint4 qv = make_uint4(0u, 0u, 0u, 0u);
                if (Row_ok[row]) {
                    qv = *reinterpret_cast<const uint4*>(
                        Q + static_cast<int64_t>(Row_tok[row]) * sq_t
                          + static_cast<int64_t>(Row_h[row]) * sq_h + c * 8
                    );
                }
                smem_kv1_4[row * 16 + swiz_c] = qv;
            }
        }
        {
            const int sub_grp = lane >> 3;
            const int sub_lane = lane & 7;
            const int row = wid * 4 + sub_grp;
            const bool r_ok = (Row_ok[row] != 0);
            const int tok = r_ok ? Row_tok[row] : 0;
            const int h = r_ok ? Row_h[row] : 0;
            const float tau = (HAS_TAU && r_ok) ? Tau[tok] : 1.0f;
            clip_r_row_exact_8lane<HAS_TAU>(
                R + static_cast<int64_t>(tok) * sr_t + static_cast<int64_t>(h) * sr_h,
                RScale + h * D_REL,
                r_ok, tau, rel_threshold, clip_eps, sub_lane, smem_r + row * D_REL
            );
        }
        __syncthreads();

        uint32_t q_frag[8][4];
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int m_row = lane & 15;
            const int k_col8 = ks * 2 + (lane >> 4);
            const int swiz_c = k_col8 ^ (m_row & 7);
            const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(m_row * 256 + swiz_c * 16);
            ldmatrix_x4(q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3], s_addr);
        }

        const uint32_t r_a0 = *reinterpret_cast<const uint32_t*>(smem_r + grp * D_REL + 2 * tid);
        const uint32_t r_a1 = *reinterpret_cast<const uint32_t*>(smem_r + (grp + 8) * D_REL + 2 * tid);
        const uint32_t r_a2 = *reinterpret_cast<const uint32_t*>(smem_r + grp * D_REL + 8 + 2 * tid);
        const uint32_t r_a3 = *reinterpret_cast<const uint32_t*>(smem_r + (grp + 8) * D_REL + 8 + 2 * tid);

        const int qpos0 = Row_qpos[grp];
        const int qpos1 = Row_qpos[grp + 8];
        const int qpos_min = Row_qpos[0];
        const int qpos_max = Row_qpos[15];

        if (!tile0_is_past) {
            if (tid_all < 64) {
                int kpos = s_beg + tid_all;
                if (kpos < s_end_live) {
                    int phys_kpos = kpos;
                    if constexpr (HAS_TREE_2B) {
                        if (is_sec_tile && kpos > past) phys_kpos = kpos + tree_wp;
                    }
                    int lblk = (BS_shift >= 0) ? (phys_kpos >> BS_shift) : (phys_kpos / BS);
                    int slot = (BS_shift >= 0) ? (phys_kpos & ((1 << BS_shift) - 1)) : (phys_kpos % BS);
                    KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
                } else {
                    KV_blk_slot[tid_all] = make_int2(-1, 0);
                }
            }
            __syncthreads();
            issue_kv64_swiz_async<false>(K_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, 0ULL, tid_all);
            cp_async_commit();
        }

        for (int n0 = s_beg; n0 < s_end_live; n0 += BN) {
            const int d_lo = qpos_min - (n0 + BN - 1);
            const int d_hi = qpos_max - n0;
            const bool tile_has_bias = (E > 0) && (d_hi >= 0) && (d_lo < E);
            const int d_base = max(0, d_lo);
            if (tile_has_bias) {
                compute_otf_bias_tile(RelProjT, smem_b, r_a0, r_a1, r_a2, r_a3, d_base, d_hi, E, wid, grp, tid);
            }

            cp_async_wait_group<0>();
            __syncthreads();

            issue_kv64_swiz_async<false>(V_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, 0ULL, tid_all);
            cp_async_commit();

            // Stride-32 K-column warp ownership:
            //   nt=0 (sub>>1 == 0): chunk c_{wid}   = cols [wid*8 .. wid*8+7]
            //   nt=1 (sub>>1 == 1): chunk c_{wid+4} = cols [32+wid*8 .. 32+wid*8+7]
            float s0[2] = {0.f, 0.f}, s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f}, s3[2] = {0.f, 0.f};
            const int k_row = (wid + (sub >> 1) * 4) * 8 + r_in;
            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int k_col8 = ks * 2 + (sub & 1);
                const int swiz_c = k_col8 ^ r_in;
                const uint32_t s_addr = kv0_sptr + static_cast<uint32_t>(k_row * 256 + swiz_c * 16);
                uint32_t kb0_0, kb1_0, kb0_1, kb1_1;
                ldmatrix_x4(kb0_0, kb1_0, kb0_1, kb1_1, s_addr);
                mma_m16n8k16(s0[0], s1[0], s2[0], s3[0], q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3], kb0_0, kb1_0);
                mma_m16n8k16(s0[1], s1[1], s2[1], s3[1], q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3], kb0_1, kb1_1);
            }

            float m_w0 = -INFINITY, m_w1 = -INFINITY;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int kpos_a = n0 + nt * 32 + wid * 8 + 2 * tid;
                const int kpos_b = kpos_a + 1;
                const bool ka_ok = (kpos_a < s_end_live);
                const bool kb_ok = (kpos_b < s_end_live);

                const int da0 = qpos0 - kpos_a;
                const int db0 = qpos0 - kpos_b;
                const bool va0 = ok0 && ka_ok && (da0 >= 0) && (WINDOW < 0 || da0 <= WINDOW);
                const bool vb0 = ok0 && kb_ok && (db0 >= 0) && (WINDOW < 0 || db0 <= WINDOW);
                if (tile_has_bias) {
                    float ba0 = (va0 && da0 < E) ? bf16_to_f32(smem_b[grp * B_COLS + (da0 - d_base)]) : 0.f;
                    float bb0 = (vb0 && db0 < E) ? bf16_to_f32(smem_b[grp * B_COLS + (db0 - d_base)]) : 0.f;
                    s0[nt] = apply_score_bias<TREE_MODE>(s0[nt], scale_log2, ba0, tree_mode_rt);
                    s1[nt] = apply_score_bias<TREE_MODE>(s1[nt], scale_log2, bb0, tree_mode_rt);
                } else {
                    s0[nt] = __fmul_rn(s0[nt], scale_log2);
                    s1[nt] = __fmul_rn(s1[nt], scale_log2);
                }
                s0[nt] = va0 ? s0[nt] : -INFINITY;
                s1[nt] = vb0 ? s1[nt] : -INFINITY;
                m_w0 = fmaxf(m_w0, fmaxf(s0[nt], s1[nt]));

                const int da1 = qpos1 - kpos_a;
                const int db1 = qpos1 - kpos_b;
                const bool va1 = ok1 && ka_ok && (da1 >= 0) && (WINDOW < 0 || da1 <= WINDOW);
                const bool vb1 = ok1 && kb_ok && (db1 >= 0) && (WINDOW < 0 || db1 <= WINDOW);
                if (tile_has_bias) {
                    float ba1 = (va1 && da1 < E) ? bf16_to_f32(smem_b[(grp + 8) * B_COLS + (da1 - d_base)]) : 0.f;
                    float bb1 = (vb1 && db1 < E) ? bf16_to_f32(smem_b[(grp + 8) * B_COLS + (db1 - d_base)]) : 0.f;
                    s2[nt] = apply_score_bias<TREE_MODE>(s2[nt], scale_log2, ba1, tree_mode_rt);
                    s3[nt] = apply_score_bias<TREE_MODE>(s3[nt], scale_log2, bb1, tree_mode_rt);
                } else {
                    s2[nt] = __fmul_rn(s2[nt], scale_log2);
                    s3[nt] = __fmul_rn(s3[nt], scale_log2);
                }
                s2[nt] = va1 ? s2[nt] : -INFINITY;
                s3[nt] = vb1 ? s3[nt] : -INFINITY;
                m_w1 = fmaxf(m_w1, fmaxf(s2[nt], s3[nt]));
            }

            m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 2));
            m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 1));
            m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 2));
            m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 1));
            if (tid == 0) {
                Red_smem[wid][grp] = m_w0;
                Red_smem[wid][grp + 8] = m_w1;
            }
            __syncthreads();

            const int n_next = n0 + BN;
            if (n_next < s_end_live) {
                if (tid_all < 64) {
                    int kpos = n_next + tid_all;
                    if (kpos < s_end_live) {
                        int phys_kpos = kpos;
                        if constexpr (HAS_TREE_2B) {
                            if (is_sec_tile && kpos > past) phys_kpos = kpos + tree_wp;
                        }
                        int lblk = (BS_shift >= 0) ? (phys_kpos >> BS_shift) : (phys_kpos / BS);
                        int slot = (BS_shift >= 0) ? (phys_kpos & ((1 << BS_shift) - 1)) : (phys_kpos % BS);
                        KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
                    } else {
                        KV_blk_slot[tid_all] = make_int2(-1, 0);
                    }
                }
                __syncthreads();
                issue_kv64_swiz_async<false>(K_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, 0ULL, tid_all);
                cp_async_commit();
            }

            const float m_blk0 = fmaxf(fmaxf(Red_smem[0][grp], Red_smem[1][grp]),
                                       fmaxf(Red_smem[2][grp], Red_smem[3][grp]));
            const float m_blk1 = fmaxf(fmaxf(Red_smem[0][grp + 8], Red_smem[1][grp + 8]),
                                       fmaxf(Red_smem[2][grp + 8], Red_smem[3][grp + 8]));
            const float m_new0 = fmaxf(m_i0, m_blk0);
            const float m_new1 = fmaxf(m_i1, m_blk1);
            const float m_safe0 = (m_new0 == -INFINITY) ? 0.0f : m_new0;
            const float m_safe1 = (m_new1 == -INFINITY) ? 0.0f : m_new1;
            const float alpha0 = ex2_approx_ftz_f32(__fsub_rn(m_i0, m_safe0));
            const float alpha1 = ex2_approx_ftz_f32(__fsub_rn(m_i1, m_safe1));

            const float p0_0 = ex2_approx_ftz_f32(__fsub_rn(s0[0], m_safe0));
            const float p1_0 = ex2_approx_ftz_f32(__fsub_rn(s1[0], m_safe0));
            const float p0_1 = ex2_approx_ftz_f32(__fsub_rn(s0[1], m_safe0));
            const float p1_1 = ex2_approx_ftz_f32(__fsub_rn(s1[1], m_safe0));

            const float p2_0 = ex2_approx_ftz_f32(__fsub_rn(s2[0], m_safe1));
            const float p3_0 = ex2_approx_ftz_f32(__fsub_rn(s3[0], m_safe1));
            const float p2_1 = ex2_approx_ftz_f32(__fsub_rn(s2[1], m_safe1));
            const float p3_1 = ex2_approx_ftz_f32(__fsub_rn(s3[1], m_safe1));

            *reinterpret_cast<uint32_t*>(smem_p + grp * P_COLS + wid * 8 + 2 * tid)            = pack_bf16x2(f32_to_bf16(p0_0), f32_to_bf16(p1_0));
            *reinterpret_cast<uint32_t*>(smem_p + grp * P_COLS + 32 + wid * 8 + 2 * tid)       = pack_bf16x2(f32_to_bf16(p0_1), f32_to_bf16(p1_1));
            *reinterpret_cast<uint32_t*>(smem_p + (grp + 8) * P_COLS + wid * 8 + 2 * tid)      = pack_bf16x2(f32_to_bf16(p2_0), f32_to_bf16(p3_0));
            *reinterpret_cast<uint32_t*>(smem_p + (grp + 8) * P_COLS + 32 + wid * 8 + 2 * tid) = pack_bf16x2(f32_to_bf16(p2_1), f32_to_bf16(p3_1));

            float ra0, rb0, ra1, rb1;
            reduce_p_phase1<WINDOW, TREE_MODE>(
                p0_0, p1_0, p0_1, p1_1, p2_0, p3_0, p2_1, p3_1,
                grp, tid, tree_mode_rt, ra0, rb0, ra1, rb1
            );

            if (n_next < s_end_live) cp_async_wait_group<1>();
            else                     cp_async_wait_group<0>();
            __syncthreads();

            float sum_p0, sum_p1;
            reduce_p_phase2_64x16<WINDOW, TREE_MODE>(
                Red_smem, p0_0, p1_0, p0_1, p1_1, p2_0, p3_0, p2_1, p3_1,
                ra0, rb0, ra1, rb1, wid, grp, tid, tree_mode_rt, sum_p0, sum_p1
            );
            l_i0 = fmaf(l_i0, alpha0, sum_p0);
            l_i1 = fmaf(l_i1, alpha1, sum_p1);
            m_i0 = m_new0;
            m_i1 = m_new1;

            if constexpr (TREE_MODE != 0) {
                if (((tree_mode_rt >> 7) & 1) != 0) {
                    float pv[4][4] = {{0.f}};
                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        const int k0 = ks * 16 + 2 * tid;
                        const int k1 = k0 + 8;
                        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k0);
                        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k0);
                        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k1);
                        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k1);
                        const int v_row = ks * 16 + (sub & 1) * 8 + r_in;
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const int nt0 = 2 * j;
                            const int d_col8 = wid * 4 + nt0 + (sub >> 1);
                            const int swiz_c = d_col8 ^ r_in;
                            const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(v_row * 256 + swiz_c * 16);
                            uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
                            ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, s_addr);
                            mma_m16n8k16(pv[nt0][0],     pv[nt0][1],     pv[nt0][2],     pv[nt0][3],     a0, a1, a2, a3, vb0_0, vb1_0);
                            mma_m16n8k16(pv[nt0 + 1][0], pv[nt0 + 1][1], pv[nt0 + 1][2], pv[nt0 + 1][3], a0, a1, a2, a3, vb0_1, vb1_1);
                        }
                    }
                    #pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        acc[nt][0] = fmaf(acc[nt][0], alpha0, pv[nt][0]);
                        acc[nt][1] = fmaf(acc[nt][1], alpha0, pv[nt][1]);
                        acc[nt][2] = fmaf(acc[nt][2], alpha1, pv[nt][2]);
                        acc[nt][3] = fmaf(acc[nt][3], alpha1, pv[nt][3]);
                    }
                    if (n_next < s_end_live) __syncthreads();
                    continue;
                }
            }

            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                acc[nt][0] = __fmul_rn(acc[nt][0], alpha0);
                acc[nt][1] = __fmul_rn(acc[nt][1], alpha0);
                acc[nt][2] = __fmul_rn(acc[nt][2], alpha1);
                acc[nt][3] = __fmul_rn(acc[nt][3], alpha1);
            }

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int k0 = ks * 16 + 2 * tid;
                const int k1 = k0 + 8;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k0);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k0);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k1);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k1);
                const int v_row = ks * 16 + (sub & 1) * 8 + r_in;

                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int nt0 = 2 * j;
                    const int d_col8 = wid * 4 + nt0 + (sub >> 1);
                    const int swiz_c = d_col8 ^ r_in;
                    const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(v_row * 256 + swiz_c * 16);
                    uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
                    ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, s_addr);
                    mma_m16n8k16(acc[nt0][0],     acc[nt0][1],     acc[nt0][2],     acc[nt0][3],     a0, a1, a2, a3, vb0_0, vb1_0);
                    mma_m16n8k16(acc[nt0 + 1][0], acc[nt0 + 1][1], acc[nt0 + 1][2], acc[nt0 + 1][3], a0, a1, a2, a3, vb0_1, vb1_1);
                }
            }
            if (n_next < s_end_live) __syncthreads();
        }
    }

    if constexpr (NUM_SPLITS == 1) {
        store_direct_gate_sink_row<TREE_MODE>(
            ok0, Row_tok[grp], Row_h[grp], NT, m_i0, l_i0,
            acc[0][0], acc[0][1], acc[1][0], acc[1][1], acc[2][0], acc[2][1], acc[3][0], acc[3][1],
            GATE, SINK, OP, LP, Y, so_t, so_h, sl_t, sl_h, sg_t, ssk_t, sy_t, wid, tid
        );
        store_direct_gate_sink_row<TREE_MODE>(
            ok1, Row_tok[grp + 8], Row_h[grp + 8], NT, m_i1, l_i1,
            acc[0][2], acc[0][3], acc[1][2], acc[1][3], acc[2][2], acc[2][3], acc[3][2], acc[3][3],
            GATE, SINK, OP, LP, Y, so_t, so_h, sl_t, sl_h, sg_t, ssk_t, sy_t, wid, tid
        );
        return;
    } else {
        if (s_beg < s_end_live) {
            const float l_safe0 = (l_i0 > 0.0f) ? l_i0 : 1.0f;
            const float l_safe1 = (l_i1 > 0.0f) ? l_i1 : 1.0f;
            if (ok0) {
                float* o0 = OP + split * so_s + Row_tok[grp] * so_t + Row_h[grp] * so_h + wid * 32;
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    o0[nt * 8 + 2 * tid]     = div_full_f32(acc[nt][0], l_safe0);
                    o0[nt * 8 + 2 * tid + 1] = div_full_f32(acc[nt][1], l_safe0);
                }
                if (wid == 0 && tid == 0) {
                    float lse0 = (l_i0 > 0.0f) ? __fmul_rn(__fadd_rn(m_i0, log2f(l_safe0)), LN2) : -INFINITY;
                    LP[split * sl_s + Row_tok[grp] * sl_t + Row_h[grp] * sl_h] = lse0;
                }
            }
            if (ok1) {
                float* o1 = OP + split * so_s + Row_tok[grp + 8] * so_t + Row_h[grp + 8] * so_h + wid * 32;
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    o1[nt * 8 + 2 * tid]     = div_full_f32(acc[nt][2], l_safe1);
                    o1[nt * 8 + 2 * tid + 1] = div_full_f32(acc[nt][3], l_safe1);
                }
                if (wid == 0 && tid == 0) {
                    float lse1 = (l_i1 > 0.0f) ? __fmul_rn(__fadd_rn(m_i1, log2f(l_safe1)), LN2) : -INFINITY;
                    LP[split * sl_s + Row_tok[grp + 8] * sl_t + Row_h[grp + 8] * sl_h] = lse1;
                }
            }
        }
        // Flush all threads' OP/LP stores to GPU L2 before tid 0 publishes SEM release!
        __threadfence();
        __syncthreads();
        alignas(16) __shared__ int s_done;
        if (tid_all == 0) s_done = atomic_add_acq_rel_gpu(&SEM[sem_idx], 1);
        __syncthreads();
        if (s_done == NUM_SPLITS - 1) {
            if (tid_all == 0) SEM[sem_idx] = 0;
            cooperative_merge_gate_sink<NUM_SPLITS, TREE_MODE>(
                BM, Row_ok, Row_tok, Row_h, OP, LP, GATE, SINK, Y,
                so_s, so_t, so_h, sl_s, sl_t, sl_h, sg_t, ssk_t, sy_t, NT, wid, lane, tree_mode_rt
            );
        }
    }
}

// =========================================================================
// KERNEL 2B (FUSE_MT = 1): C=32 1 CTA per (req, split) covering MT_MAX in {1, 2, 3}
//   - Guaranteed 1.000x KV DRAM reads with `L2::evict_first` + exact Blocked([1,8]) sum(p) tree!
//   - Dynamic SMEM <= 47.6 KiB (0 opt-in needed -> 2 CTAs/SM, 100% CUDA Graph safe!)
// =========================================================================
template <int WINDOW, bool KV_EVICT_FIRST, int TREE_MODE, bool IS_FIRST_MT, bool IS_LAST_MT>
static __device__ __forceinline__ void fusedmt_step_tile(
    int mt,
    int n0, int n_next, int s_end_live, int E, int BS, int BS_shift,
    float scale_log2,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ K_base,
    int64_t sk_b, int64_t sk_s,
    const int32_t* __restrict__ bt_row,
    uint32_t q_sptr_mt, uint32_t kv0_sptr, uint32_t kv1_sptr,
    uint4* __restrict__ smem_kv0_4,
    uint16_t* __restrict__ smem_b,
    uint16_t* __restrict__ smem_p,
    float Red_smem[32][8],
    int2 KV_blk_slot[64],
    const int32_t* Row_qpos,
    const uint8_t* Row_ok,
    uint32_t r_a0, uint32_t r_a1, uint32_t r_a2, uint32_t r_a3,
    float& m_i0, float& m_i1, float& l_i0, float& l_i1,
    float (&acc)[4][4],
    uint64_t l2_policy,
    int tree_mode_rt,
    int tid_all, int wid, int lane, int grp, int tid, int sub, int r_in
) {
    const int base = mt * BM;
    const bool ok0 = (Row_ok[base + grp] != 0);
    const bool ok1 = (Row_ok[base + grp + 8] != 0);
    const int qpos0 = Row_qpos[base + grp];
    const int qpos1 = Row_qpos[base + grp + 8];
    const int qpos_min = Row_qpos[base];
    const int qpos_max = Row_qpos[base + 15];

    const int d_lo = qpos_min - (n0 + BN - 1);
    const int d_hi = qpos_max - n0;
    const bool tile_has_bias = (E > 0) && (d_hi >= 0) && (d_lo < E);
    const int d_base = max(0, d_lo);

    if constexpr (!IS_FIRST_MT) {
        if (tile_has_bias) {
            compute_otf_bias_tile(RelProjT, smem_b, r_a0, r_a1, r_a2, r_a3, d_base, d_hi, E, wid, grp, tid);
            __syncthreads();
        }
    }

    float s0[2] = {0.f, 0.f}, s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f}, s3[2] = {0.f, 0.f};
    const int k_row = (wid + (sub >> 1) * 4) * 8 + r_in;
    const int q_row = lane & 15;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int q_col8 = ks * 2 + (lane >> 4);
        const int q_swiz = q_col8 ^ (q_row & 7);
        const uint32_t q_addr = q_sptr_mt + static_cast<uint32_t>(q_row * 256 + q_swiz * 16);
        uint32_t qa0, qa1, qa2, qa3;
        ldmatrix_x4(qa0, qa1, qa2, qa3, q_addr);

        const int k_col8 = ks * 2 + (sub & 1);
        const int k_swiz = k_col8 ^ r_in;
        const uint32_t k_addr = kv0_sptr + static_cast<uint32_t>(k_row * 256 + k_swiz * 16);
        uint32_t kb0_0, kb1_0, kb0_1, kb1_1;
        ldmatrix_x4(kb0_0, kb1_0, kb0_1, kb1_1, k_addr);

        mma_m16n8k16(s0[0], s1[0], s2[0], s3[0], qa0, qa1, qa2, qa3, kb0_0, kb1_0);
        mma_m16n8k16(s0[1], s1[1], s2[1], s3[1], qa0, qa1, qa2, qa3, kb0_1, kb1_1);
    }

    float m_w0 = -INFINITY, m_w1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int kpos_a = n0 + nt * 32 + wid * 8 + 2 * tid;
        const int kpos_b = kpos_a + 1;
        const bool ka_ok = (kpos_a < s_end_live);
        const bool kb_ok = (kpos_b < s_end_live);

        const int da0 = qpos0 - kpos_a;
        const int db0 = qpos0 - kpos_b;
        const bool va0 = ok0 && ka_ok && (da0 >= 0) && (WINDOW < 0 || da0 <= WINDOW);
        const bool vb0 = ok0 && kb_ok && (db0 >= 0) && (WINDOW < 0 || db0 <= WINDOW);
        if (tile_has_bias) {
            float ba0 = (va0 && da0 < E) ? bf16_to_f32(smem_b[grp * B_COLS + (da0 - d_base)]) : 0.f;
            float bb0 = (vb0 && db0 < E) ? bf16_to_f32(smem_b[grp * B_COLS + (db0 - d_base)]) : 0.f;
            s0[nt] = apply_score_bias<TREE_MODE>(s0[nt], scale_log2, ba0, tree_mode_rt);
            s1[nt] = apply_score_bias<TREE_MODE>(s1[nt], scale_log2, bb0, tree_mode_rt);
        } else {
            s0[nt] = __fmul_rn(s0[nt], scale_log2);
            s1[nt] = __fmul_rn(s1[nt], scale_log2);
        }
        s0[nt] = va0 ? s0[nt] : -INFINITY;
        s1[nt] = vb0 ? s1[nt] : -INFINITY;
        m_w0 = fmaxf(m_w0, fmaxf(s0[nt], s1[nt]));

        const int da1 = qpos1 - kpos_a;
        const int db1 = qpos1 - kpos_b;
        const bool va1 = ok1 && ka_ok && (da1 >= 0) && (WINDOW < 0 || da1 <= WINDOW);
        const bool vb1 = ok1 && kb_ok && (db1 >= 0) && (WINDOW < 0 || db1 <= WINDOW);
        if (tile_has_bias) {
            float ba1 = (va1 && da1 < E) ? bf16_to_f32(smem_b[(grp + 8) * B_COLS + (da1 - d_base)]) : 0.f;
            float bb1 = (vb1 && db1 < E) ? bf16_to_f32(smem_b[(grp + 8) * B_COLS + (db1 - d_base)]) : 0.f;
            s2[nt] = apply_score_bias<TREE_MODE>(s2[nt], scale_log2, ba1, tree_mode_rt);
            s3[nt] = apply_score_bias<TREE_MODE>(s3[nt], scale_log2, bb1, tree_mode_rt);
        } else {
            s2[nt] = __fmul_rn(s2[nt], scale_log2);
            s3[nt] = __fmul_rn(s3[nt], scale_log2);
        }
        s2[nt] = va1 ? s2[nt] : -INFINITY;
        s3[nt] = vb1 ? s3[nt] : -INFINITY;
        m_w1 = fmaxf(m_w1, fmaxf(s2[nt], s3[nt]));
    }

    m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 2));
    m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 1));
    m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 2));
    m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 1));
    if (tid == 0) {
        Red_smem[wid][grp]     = m_w0;
        Red_smem[wid + 4][grp] = m_w1;
    }
    __syncthreads();

    if constexpr (IS_LAST_MT) {
        if (n_next < s_end_live) {
            if (tid_all < 64) {
                int kpos = n_next + tid_all;
                if (kpos < s_end_live) {
                    int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
                    int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
                    KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
                } else {
                    KV_blk_slot[tid_all] = make_int2(-1, 0);
                }
            }
            __syncthreads();
            issue_kv64_swiz_async<KV_EVICT_FIRST>(K_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, l2_policy, tid_all);
            cp_async_commit();
        }
    }

    const float m_blk0 = fmaxf(fmaxf(Red_smem[0][grp], Red_smem[1][grp]),
                               fmaxf(Red_smem[2][grp], Red_smem[3][grp]));
    const float m_blk1 = fmaxf(fmaxf(Red_smem[4][grp], Red_smem[5][grp]),
                               fmaxf(Red_smem[6][grp], Red_smem[7][grp]));
    const float m_new0 = fmaxf(m_i0, m_blk0);
    const float m_new1 = fmaxf(m_i1, m_blk1);
    const float m_safe0 = (m_new0 == -INFINITY) ? 0.0f : m_new0;
    const float m_safe1 = (m_new1 == -INFINITY) ? 0.0f : m_new1;
    const float alpha0 = ex2_approx_ftz_f32(__fsub_rn(m_i0, m_safe0));
    const float alpha1 = ex2_approx_ftz_f32(__fsub_rn(m_i1, m_safe1));

    const float p0_0 = ex2_approx_ftz_f32(__fsub_rn(s0[0], m_safe0));
    const float p1_0 = ex2_approx_ftz_f32(__fsub_rn(s1[0], m_safe0));
    const float p0_1 = ex2_approx_ftz_f32(__fsub_rn(s0[1], m_safe0));
    const float p1_1 = ex2_approx_ftz_f32(__fsub_rn(s1[1], m_safe0));

    const float p2_0 = ex2_approx_ftz_f32(__fsub_rn(s2[0], m_safe1));
    const float p3_0 = ex2_approx_ftz_f32(__fsub_rn(s3[0], m_safe1));
    const float p2_1 = ex2_approx_ftz_f32(__fsub_rn(s2[1], m_safe1));
    const float p3_1 = ex2_approx_ftz_f32(__fsub_rn(s3[1], m_safe1));

    *reinterpret_cast<uint32_t*>(smem_p + grp * P_COLS + wid * 8 + 2 * tid)            = pack_bf16x2(f32_to_bf16(p0_0), f32_to_bf16(p1_0));
    *reinterpret_cast<uint32_t*>(smem_p + grp * P_COLS + 32 + wid * 8 + 2 * tid)       = pack_bf16x2(f32_to_bf16(p0_1), f32_to_bf16(p1_1));
    *reinterpret_cast<uint32_t*>(smem_p + (grp + 8) * P_COLS + wid * 8 + 2 * tid)      = pack_bf16x2(f32_to_bf16(p2_0), f32_to_bf16(p3_0));
    *reinterpret_cast<uint32_t*>(smem_p + (grp + 8) * P_COLS + 32 + wid * 8 + 2 * tid) = pack_bf16x2(f32_to_bf16(p2_1), f32_to_bf16(p3_1));

    float ra0, rb0, ra1, rb1;
    reduce_p_phase1<WINDOW, TREE_MODE>(
        p0_0, p1_0, p0_1, p1_1, p2_0, p3_0, p2_1, p3_1,
        grp, tid, tree_mode_rt, ra0, rb0, ra1, rb1
    );

    if constexpr (IS_FIRST_MT) {
        if constexpr (IS_LAST_MT) {
            if (n_next < s_end_live) cp_async_wait_group<1>();
            else                     cp_async_wait_group<0>();
        } else {
            cp_async_wait_group<0>();
        }
    }
    __syncthreads();

    float sum_p0, sum_p1;
    reduce_p_phase2_32x8<WINDOW, TREE_MODE>(
        Red_smem, p0_0, p1_0, p0_1, p1_1, p2_0, p3_0, p2_1, p3_1,
        ra0, rb0, ra1, rb1, wid, grp, tid, tree_mode_rt, sum_p0, sum_p1
    );
    l_i0 = fmaf(l_i0, alpha0, sum_p0);
    l_i1 = fmaf(l_i1, alpha1, sum_p1);
    m_i0 = m_new0;
    m_i1 = m_new1;

    if constexpr (TREE_MODE != 0) {
        if (((tree_mode_rt >> 7) & 1) != 0) {
            float pv[4][4] = {{0.f}};
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int k0 = ks * 16 + 2 * tid;
                const int k1 = k0 + 8;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k0);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k0);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k1);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k1);
                const int v_row = ks * 16 + (sub & 1) * 8 + r_in;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int nt0 = 2 * j;
                    const int d_col8 = wid * 4 + nt0 + (sub >> 1);
                    const int swiz_c = d_col8 ^ r_in;
                    const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(v_row * 256 + swiz_c * 16);
                    uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
                    ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, s_addr);
                    mma_m16n8k16(pv[nt0][0],     pv[nt0][1],     pv[nt0][2],     pv[nt0][3],     a0, a1, a2, a3, vb0_0, vb1_0);
                    mma_m16n8k16(pv[nt0 + 1][0], pv[nt0 + 1][1], pv[nt0 + 1][2], pv[nt0 + 1][3], a0, a1, a2, a3, vb0_1, vb1_1);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                acc[nt][0] = fmaf(acc[nt][0], alpha0, pv[nt][0]);
                acc[nt][1] = fmaf(acc[nt][1], alpha0, pv[nt][1]);
                acc[nt][2] = fmaf(acc[nt][2], alpha1, pv[nt][2]);
                acc[nt][3] = fmaf(acc[nt][3], alpha1, pv[nt][3]);
            }
            if constexpr (!IS_LAST_MT) {
                __syncthreads();
            } else {
                if (n_next < s_end_live) __syncthreads();
            }
            return;
        }
    }

    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        acc[nt][0] = __fmul_rn(acc[nt][0], alpha0);
        acc[nt][1] = __fmul_rn(acc[nt][1], alpha0);
        acc[nt][2] = __fmul_rn(acc[nt][2], alpha1);
        acc[nt][3] = __fmul_rn(acc[nt][3], alpha1);
    }

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = ks * 16 + 2 * tid;
        const int k1 = k0 + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k0);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k0);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k1);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k1);
        const int v_row = ks * 16 + (sub & 1) * 8 + r_in;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int nt0 = 2 * j;
            const int d_col8 = wid * 4 + nt0 + (sub >> 1);
            const int swiz_c = d_col8 ^ r_in;
            const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(v_row * 256 + swiz_c * 16);
            uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
            ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, s_addr);
            mma_m16n8k16(acc[nt0][0],     acc[nt0][1],     acc[nt0][2],     acc[nt0][3],     a0, a1, a2, a3, vb0_0, vb1_0);
            mma_m16n8k16(acc[nt0 + 1][0], acc[nt0 + 1][1], acc[nt0 + 1][2], acc[nt0 + 1][3], a0, a1, a2, a3, vb0_1, vb1_1);
        }
    }

    if constexpr (!IS_LAST_MT) {
        __syncthreads();
    } else {
        if (n_next < s_end_live) __syncthreads();
    }
}

template <int WINDOW, int NUM_SPLITS, int MT_MAX, bool HAS_TAU, bool HAS_NLIVE, bool KV_EVICT_FIRST, int TREE_MODE>
__global__ void __launch_bounds__(128, 2)
fused_decode_attn_otf_fusedmt_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RScale,
    const float*    __restrict__ Tau,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const int32_t*  __restrict__ BT,
    const int32_t*  __restrict__ SEQ,
    const int32_t*  __restrict__ CUQ,
    const uint16_t* __restrict__ GATE,
    const uint16_t* __restrict__ SINK,
    float*          __restrict__ OP,
    float*          __restrict__ LP,
    int32_t*        __restrict__ SEM,
    uint16_t*       __restrict__ Y,
    const int32_t*  __restrict__ NLivePtr,
    int64_t sq_t, int64_t sq_h,
    int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h,
    int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int BS, int BS_shift, int E, int G, int tree_mode_rt
) {
    (void)T;
    constexpr int MAX_ROWS = MT_MAX * BM;
    const int r = blockIdx.y;
    const int kvh = blockIdx.z / NUM_SPLITS;
    const int split = blockIdx.z % NUM_SPLITS;
    const int sem_idx = r * (gridDim.z / NUM_SPLITS) + kvh;

    const int tid_all = threadIdx.x;
    const int wid = tid_all >> 5;
    const int lane = tid_all & 31;
    const int grp = lane >> 2;
    const int tid = lane & 3;
    const int sub = lane >> 3;
    const int r_in = lane & 7;

    const int q0 = CUQ[r];
    const int qlen = CUQ[r + 1] - q0;
    const int seqlen = SEQ[r];
    const int past = seqlen - qlen;
    int r_live = qlen;
    if constexpr (HAS_NLIVE) {
        int nl = *NLivePtr;
        int d = nl - q0;
        if (d < 0) d = 0;
        if (d < r_live) r_live = d;
    }
    if (r_live <= 0) return;

    int actual_mt = (r_live * G + BM - 1) / BM;
    if (actual_mt > MT_MAX) actual_mt = MT_MAX;

    const int k_end = max(past + qlen, 0);
    const int k_beg = (WINDOW >= 0) ? max(past - WINDOW, 0) : 0;
    const int span = max(k_end - k_beg, 0);
    const int per = (((span + NUM_SPLITS - 1) / NUM_SPLITS) + 63) / 64 * 64;
    const int s_beg = k_beg + split * per;
    const int s_end = min(s_beg + per, k_end);
    int s_end_live = (seqlen > 0 && q0 < NT) ? s_end : s_beg;
    if constexpr (HAS_NLIVE) {
        if (past + r_live < s_end_live) s_end_live = past + r_live;
    }

    extern __shared__ uint4 smem_raw_u4[];
    uint8_t*  smem_raw = reinterpret_cast<uint8_t*>(smem_raw_u4);
    uint16_t* smem_kv0 = reinterpret_cast<uint16_t*>(smem_raw);
    uint16_t* smem_kv1 = reinterpret_cast<uint16_t*>(smem_raw + 16384);
    uint16_t* smem_q   = reinterpret_cast<uint16_t*>(smem_raw + 32768);
    uint16_t* smem_b   = reinterpret_cast<uint16_t*>(smem_raw + 32768 + MT_MAX * 4096);
    uint16_t* smem_p   = smem_b;

    alignas(16) __shared__ float    Red_smem[32][8];
    alignas(16) __shared__ int32_t  Row_tok[MAX_ROWS];
    alignas(16) __shared__ int32_t  Row_h[MAX_ROWS];
    alignas(16) __shared__ int32_t  Row_qpos[MAX_ROWS];
    alignas(16) __shared__ uint8_t  Row_ok[MAX_ROWS];
    alignas(16) __shared__ int2     KV_blk_slot[64];

    if (tid_all < MAX_ROWS) {
        int i_loc = tid_all / G;
        int g = tid_all % G;
        bool ok = (i_loc < r_live);
        Row_ok[tid_all] = ok ? 1 : 0;
        Row_tok[tid_all] = q0 + i_loc;
        Row_h[tid_all] = kvh * G + g;
        Row_qpos[tid_all] = past + i_loc;
    }
    __syncthreads();

    float m_i0[MT_MAX], m_i1[MT_MAX], l_i0[MT_MAX], l_i1[MT_MAX];
    float acc[MT_MAX][4][4];
    #pragma unroll
    for (int m = 0; m < MT_MAX; ++m) {
        m_i0[m] = -INFINITY; m_i1[m] = -INFINITY;
        l_i0[m] = 0.0f;      l_i1[m] = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            acc[m][nt][0] = 0.f; acc[m][nt][1] = 0.f;
            acc[m][nt][2] = 0.f; acc[m][nt][3] = 0.f;
        }
    }

    if (s_beg >= s_end_live) {
        if constexpr (NUM_SPLITS > 1) {
            if (tid_all < MAX_ROWS && Row_ok[tid_all]) {
                LP[split * sl_s + Row_tok[tid_all] * sl_t + Row_h[tid_all] * sl_h] = -INFINITY;
            }
        }
    } else {
        const uint64_t l2_policy = KV_EVICT_FIRST ? make_l2_evict_first_policy() : 0ULL;
        const int32_t* bt_row = BT + static_cast<int64_t>(r) * sbt_r;
        const uint16_t* K_base = K + static_cast<int64_t>(kvh) * sk_h;
        const uint16_t* V_base = V + static_cast<int64_t>(kvh) * sv_h;
        const uint32_t kv0_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv0));
        const uint32_t kv1_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv1));
        const uint32_t q_sptr   = static_cast<uint32_t>(__cvta_generic_to_shared(smem_q));
        uint4* smem_kv0_4 = reinterpret_cast<uint4*>(smem_kv0);
        uint4* smem_kv1_4 = reinterpret_cast<uint4*>(smem_kv1);
        uint4* smem_q_4   = reinterpret_cast<uint4*>(smem_q);

        if (tid_all < 64) {
            int kpos = s_beg + tid_all;
            if (kpos < s_end_live) {
                int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
                int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
                KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
            } else {
                KV_blk_slot[tid_all] = make_int2(-1, 0);
            }
        }
        __syncthreads();
        issue_kv64_swiz_async<KV_EVICT_FIRST>(K_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, l2_policy, tid_all);
        cp_async_commit();

        #pragma unroll
        for (int mt = 0; mt < MT_MAX; ++mt) {
            if (mt < actual_mt) {
                #pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    int idx = pass * 128 + tid_all;
                    int lrow = idx >> 4;
                    int c = idx & 15;
                    int swiz_c = c ^ (lrow & 7);
                    int grow = mt * BM + lrow;
                    uint4 qv = make_uint4(0u, 0u, 0u, 0u);
                    if (Row_ok[grow]) {
                        qv = *reinterpret_cast<const uint4*>(
                            Q + static_cast<int64_t>(Row_tok[grow]) * sq_t
                              + static_cast<int64_t>(Row_h[grow]) * sq_h + c * 8
                        );
                    }
                    smem_q_4[(mt * BM + lrow) * 16 + swiz_c] = qv;
                }
            }
        }

        uint32_t r_a[MT_MAX][4];
        {
            const int sub_grp = lane >> 3;
            const int sub_lane = lane & 7;
            #pragma unroll
            for (int mt = 0; mt < MT_MAX; ++mt) {
                if (mt < actual_mt) {
                    const int row = mt * BM + wid * 4 + sub_grp;
                    const bool r_ok = (Row_ok[row] != 0);
                    const int tok = r_ok ? Row_tok[row] : 0;
                    const int h = r_ok ? Row_h[row] : 0;
                    const float tau = (HAS_TAU && r_ok) ? Tau[tok] : 1.0f;
                    clip_r_row_exact_8lane<HAS_TAU>(
                        R + static_cast<int64_t>(tok) * sr_t + static_cast<int64_t>(h) * sr_h,
                        RScale + h * D_REL,
                        r_ok, tau, rel_threshold, clip_eps, sub_lane, smem_b + (wid * 4 + sub_grp) * D_REL
                    );
                    __syncthreads();
                    r_a[mt][0] = *reinterpret_cast<const uint32_t*>(smem_b + grp * D_REL + 2 * tid);
                    r_a[mt][1] = *reinterpret_cast<const uint32_t*>(smem_b + (grp + 8) * D_REL + 2 * tid);
                    r_a[mt][2] = *reinterpret_cast<const uint32_t*>(smem_b + grp * D_REL + 8 + 2 * tid);
                    r_a[mt][3] = *reinterpret_cast<const uint32_t*>(smem_b + (grp + 8) * D_REL + 8 + 2 * tid);
                    __syncthreads();
                } else {
                    r_a[mt][0] = 0u; r_a[mt][1] = 0u; r_a[mt][2] = 0u; r_a[mt][3] = 0u;
                }
            }
        }

        const int qpos0_min = Row_qpos[0];
        const int qpos0_max = Row_qpos[15];

        for (int n0 = s_beg; n0 < s_end_live; n0 += BN) {
            const int d0_lo = qpos0_min - (n0 + BN - 1);
            const int d0_hi = qpos0_max - n0;
            if ((E > 0) && (d0_hi >= 0) && (d0_lo < E)) {
                compute_otf_bias_tile(
                    RelProjT, smem_b, r_a[0][0], r_a[0][1], r_a[0][2], r_a[0][3],
                    max(0, d0_lo), d0_hi, E, wid, grp, tid
                );
            }

            cp_async_wait_group<0>();
            __syncthreads();

            issue_kv64_swiz_async<KV_EVICT_FIRST>(V_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, l2_policy, tid_all);
            cp_async_commit();

            const int n_next = n0 + BN;

            #define DO_MT(M_IDX, FIRST_B, LAST_B) \
                fusedmt_step_tile<WINDOW, KV_EVICT_FIRST, TREE_MODE, FIRST_B, LAST_B>( \
                    M_IDX, n0, n_next, s_end_live, E, BS, BS_shift, scale_log2, \
                    RelProjT, K_base, sk_b, sk_s, bt_row, \
                    q_sptr + static_cast<uint32_t>((M_IDX) * 4096), kv0_sptr, kv1_sptr, smem_kv0_4, \
                    smem_b, smem_p, Red_smem, KV_blk_slot, Row_qpos, Row_ok, \
                    r_a[M_IDX][0], r_a[M_IDX][1], r_a[M_IDX][2], r_a[M_IDX][3], \
                    m_i0[M_IDX], m_i1[M_IDX], l_i0[M_IDX], l_i1[M_IDX], acc[M_IDX], \
                    l2_policy, tree_mode_rt, tid_all, wid, lane, grp, tid, sub, r_in)

            if constexpr (MT_MAX == 1) {
                DO_MT(0, true, true);
            } else if constexpr (MT_MAX == 2) {
                if (actual_mt == 1) {
                    DO_MT(0, true, true);
                } else {
                    DO_MT(0, true,  false);
                    DO_MT(1, false, true);
                }
            } else {
                if (actual_mt == 1) {
                    DO_MT(0, true, true);
                } else if (actual_mt == 2) {
                    DO_MT(0, true,  false);
                    DO_MT(1, false, true);
                } else {
                    DO_MT(0, true,  false);
                    DO_MT(1, false, false);
                    DO_MT(2, false, true);
                }
            }
            #undef DO_MT
        }
    }

    if constexpr (NUM_SPLITS == 1) {
        #pragma unroll
        for (int mt = 0; mt < MT_MAX; ++mt) {
            if (mt < actual_mt) {
                const int r0 = mt * BM + grp;
                const int r1 = r0 + 8;
                store_direct_gate_sink_row<TREE_MODE>(
                    Row_ok[r0] != 0, Row_tok[r0], Row_h[r0], NT, m_i0[mt], l_i0[mt],
                    acc[mt][0][0], acc[mt][0][1], acc[mt][1][0], acc[mt][1][1],
                    acc[mt][2][0], acc[mt][2][1], acc[mt][3][0], acc[mt][3][1],
                    GATE, SINK, OP, LP, Y, so_t, so_h, sl_t, sl_h, sg_t, ssk_t, sy_t, wid, tid
                );
                store_direct_gate_sink_row<TREE_MODE>(
                    Row_ok[r1] != 0, Row_tok[r1], Row_h[r1], NT, m_i1[mt], l_i1[mt],
                    acc[mt][0][2], acc[mt][0][3], acc[mt][1][2], acc[mt][1][3],
                    acc[mt][2][2], acc[mt][2][3], acc[mt][3][2], acc[mt][3][3],
                    GATE, SINK, OP, LP, Y, so_t, so_h, sl_t, sl_h, sg_t, ssk_t, sy_t, wid, tid
                );
            }
        }
        return;
    } else {
        if (s_beg < s_end_live) {
            #pragma unroll
            for (int mt = 0; mt < MT_MAX; ++mt) {
                if (mt < actual_mt) {
                    const int r0 = mt * BM + grp;
                    const int r1 = r0 + 8;
                    const float l_safe0 = (l_i0[mt] > 0.0f) ? l_i0[mt] : 1.0f;
                    const float l_safe1 = (l_i1[mt] > 0.0f) ? l_i1[mt] : 1.0f;
                    if (Row_ok[r0]) {
                        float* o0 = OP + split * so_s + Row_tok[r0] * so_t + Row_h[r0] * so_h + wid * 32;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            o0[nt * 8 + 2 * tid]     = div_full_f32(acc[mt][nt][0], l_safe0);
                            o0[nt * 8 + 2 * tid + 1] = div_full_f32(acc[mt][nt][1], l_safe0);
                        }
                        if (wid == 0 && tid == 0) {
                            float lse0 = (l_i0[mt] > 0.0f) ? __fmul_rn(__fadd_rn(m_i0[mt], log2f(l_safe0)), LN2) : -INFINITY;
                            LP[split * sl_s + Row_tok[r0] * sl_t + Row_h[r0] * sl_h] = lse0;
                        }
                    }
                    if (Row_ok[r1]) {
                        float* o1 = OP + split * so_s + Row_tok[r1] * so_t + Row_h[r1] * so_h + wid * 32;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            o1[nt * 8 + 2 * tid]     = div_full_f32(acc[mt][nt][2], l_safe1);
                            o1[nt * 8 + 2 * tid + 1] = div_full_f32(acc[mt][nt][3], l_safe1);
                        }
                        if (wid == 0 && tid == 0) {
                            float lse1 = (l_i1[mt] > 0.0f) ? __fmul_rn(__fadd_rn(m_i1[mt], log2f(l_safe1)), LN2) : -INFINITY;
                            LP[split * sl_s + Row_tok[r1] * sl_t + Row_h[r1] * sl_h] = lse1;
                        }
                    }
                }
            }
        }
        __threadfence();
        __syncthreads();
        alignas(16) __shared__ int s_done;
        if (tid_all == 0) s_done = atomic_add_acq_rel_gpu(&SEM[sem_idx], 1);
        __syncthreads();
        if (s_done == NUM_SPLITS - 1) {
            if (tid_all == 0) SEM[sem_idx] = 0;
            cooperative_merge_gate_sink<NUM_SPLITS, TREE_MODE>(
                actual_mt * BM, Row_ok, Row_tok, Row_h, OP, LP, GATE, SINK, Y,
                so_s, so_t, so_h, sl_s, sl_t, sl_h, sg_t, ssk_t, sy_t, NT, wid, lane, tree_mode_rt
            );
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

static inline int dispatch_prologue_kvq_impl(
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    const void* TreeSconvAnc,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, bool has_tree, uint64_t stream_u64
) {
    if (has_tree) {
        if (T <= 0 || T > 16 || TreeSconvAnc == nullptr) return -1;
    } else {
        if (T <= 0 || T > 512) return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    int n_shift = (N_cache > 0 && (N_cache & (N_cache - 1)) == 0) ? __builtin_ctz(N_cache) : -1;
    int kv_shift = (kv_bs > 0 && (kv_bs & (kv_bs - 1)) == 0) ? __builtin_ctz(kv_bs) : -1;
    const bool c32 = (!has_tree && T > 16);
    int k_ctas = c32 ? (((T + 2) / 3 < 24) ? (T + 2) / 3 : 24) : 8;
    int v_ctas = k_ctas;
    int q_ctas = c32 ? ((T < 48) ? T : 48) : 8;
    dim3 grid(k_ctas + v_ctas + q_ctas, 1, 1);
    dim3 block(96, 1, 1);
    int mode = (dynamic ? 1 : 0) | (has_tau ? 2 : 0) | (has_nlive ? 4 : 0);
    bool pdl = (use_pdl != 0);

    #define LP(DYN, TAU, NL, PD, C32M, TR) \
        return (int)launch_maybe_pdl(attn_prologue_kvq_exact_kernel<DYN, TAU, NL, PD, C32M, TR>, grid, block, 0, stream, PD, \
            (const uint16_t*)K, (const uint16_t*)V, (const uint16_t*)WK, (const uint16_t*)WV, \
            (const uint16_t*)DK, (const uint16_t*)DV, \
            (uint16_t*)ConvCache, (uint16_t*)KeyCache, (uint16_t*)ValueCache, \
            (const int64_t*)KVSlot, (const int64_t*)Pos, (const int32_t*)SeqIdx, (const int64_t*)Slot, \
            (const int32_t*)BlockTable, (const int32_t*)QStart, \
            (const uint16_t*)Q, (const uint16_t*)QScale, (const float*)Tau, (uint16_t*)QOut, \
            (const int32_t*)NLivePtr, (const int32_t*)TreeSconvAnc, \
            T, k_ctas, v_ctas, q_ctas, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv, \
            stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, \
            N_cache, n_shift, stride_q, scale, clip_threshold, clip_eps, norm_eps)

    #define DISPATCH(DYN, TAU, NL) \
        if (has_tree) { if (pdl) { LP(DYN, TAU, NL, true, false, true); } else { LP(DYN, TAU, NL, false, false, true); } } \
        else if (c32) { if (pdl) { LP(DYN, TAU, NL, true, true,  false); } else { LP(DYN, TAU, NL, false, true,  false); } } \
        else          { if (pdl) { LP(DYN, TAU, NL, true, false, false); } else { LP(DYN, TAU, NL, false, false, false); } }

    switch (mode) {
        case 0: DISPATCH(false, false, false);
        case 1: DISPATCH(true,  false, false);
        case 2: DISPATCH(false, true,  false);
        case 3: DISPATCH(true,  true,  false);
        case 4: DISPATCH(false, false, true);
        case 5: DISPATCH(true,  false, true);
        case 6: DISPATCH(false, true,  true);
        case 7: DISPATCH(true,  true,  true);
        default: return -2;
    }
    #undef DISPATCH
    #undef LP
}

}  // namespace

// =========================================================================
// EXPORTED C ABI SYMBOLS (100% 1:1 ABI MATCH WITH patches-s6/sm12_attn_fused.py!)
//   1. attn_prologue_kvq_exact_fwd        (45 args) — Linear C=1 & C=32
//   2. fused_decode_attn_otf_mgs_fwd      (56 args) — Linear C=1 & C=32 (FUSE_MT=0)
//   3. fused_decode_attn_otf_mgs_c32_fwd  (58 args) — Batched C=32 (FUSE_MT=0/1 + evict_first)
//   4. attn_prologue_kvq_tree_exact_fwd   (46 args) — S8-P6v2 Beam-2 Tree Prologue
//   5. fused_decode_attn_otf_mgs_tree_fwd (57 args) — S8-P6v2 4-Tile BM=16 Beam-2 Tree Attn
// =========================================================================

extern "C" int attn_prologue_kvq_exact_fwd(
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, uint64_t stream_u64
) {
    return dispatch_prologue_kvq_impl(
        K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
        KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, nullptr,
        T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
        stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs,
        stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
        dynamic, has_tau, has_nlive, use_pdl, false, stream_u64);
}

extern "C" int attn_prologue_kvq_tree_exact_fwd(
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    const void* TreeSconvAnc,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, uint64_t stream_u64
) {
    return dispatch_prologue_kvq_impl(
        K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
        KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, TreeSconvAnc,
        T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
        stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs,
        stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
        dynamic, has_tau, has_nlive, use_pdl, true, stream_u64);
}

extern "C" int fused_decode_attn_otf_mgs_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    uint64_t stream_u64
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    int bs_shift = (BS > 0 && (BS & (BS - 1)) == 0) ? __builtin_ctz(BS) : -1;
    dim3 grid(mtiles, num_reqs, Hkv * num_splits);
    dim3 block(128, 1, 1);
    static constexpr size_t SMEM_BYTES = 38656;

    #define LA(WIN, S, TAU, NL, PD, TR, TM) \
        return (int)launch_maybe_pdl(fused_decode_attn_otf_mgs_kernel<WIN, S, TAU, NL, PD, TR, TM, false>, \
            grid, block, SMEM_BYTES, stream, PD, \
            (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)RScale, (const float*)Tau, (const uint16_t*)RelProjT, \
            (const uint16_t*)K, (const uint16_t*)V, (const int32_t*)BT, (const int32_t*)SEQ, (const int32_t*)CUQ, \
            (const uint16_t*)GATE, (const uint16_t*)SINK, (float*)OP, (float*)LP, (int32_t*)SEM, (uint16_t*)Y, \
            (const int32_t*)NLivePtr, (const int32_t*)nullptr, sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, \
            sbt_r, sg_t, ssk_t, sy_t, so_s, so_t, so_h, sl_s, sl_t, sl_h, \
            scale_log2, rel_threshold, clip_eps, T, NT, BS, bs_shift, E, G, tree_mode)

    #define DISP_TM(WIN, S, TAU, NL, PD, TR) \
        if (tree_mode == 0) { LA(WIN, S, TAU, NL, PD, TR, 0); } \
        else                { LA(WIN, S, TAU, NL, PD, TR, -1); }

    #define DISP_FLAGS(WIN, S, TAU) \
        if (has_nlive) { \
            if (use_pdl) { if (trig_wo) { DISP_TM(WIN, S, TAU, true, true, true); } else { DISP_TM(WIN, S, TAU, true, true, false); } } \
            else         { DISP_TM(WIN, S, TAU, true, false, false); } \
        } else { \
            if (use_pdl) { if (trig_wo) { DISP_TM(WIN, S, TAU, false, true, true); } else { DISP_TM(WIN, S, TAU, false, true, false); } } \
            else         { DISP_TM(WIN, S, TAU, false, false, false); } \
        }

    #define DISP_WIN_S(WIN, S) \
        if (has_tau) { DISP_FLAGS(WIN, S, true); } else { DISP_FLAGS(WIN, S, false); }

    if (window == 1023) {
        if (num_splits == 1) { DISP_WIN_S(1023, 1); }
        if (num_splits == 2) { DISP_WIN_S(1023, 2); }
        if (num_splits == 4) { DISP_WIN_S(1023, 4); }
        if (num_splits == 8) { DISP_WIN_S(1023, 8); }
    } else if (window < 0) {
        if (num_splits == 1)  { DISP_WIN_S(-1, 1); }
        if (num_splits == 2)  { DISP_WIN_S(-1, 2); }
        if (num_splits == 4)  { DISP_WIN_S(-1, 4); }
        if (num_splits == 16) { DISP_WIN_S(-1, 16); }
    }
    return -2;
    #undef DISP_WIN_S
    #undef DISP_FLAGS
    #undef DISP_TM
    #undef LA
}

extern "C" int fused_decode_attn_otf_mgs_c32_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    int fuse_mt, int kv_evict_first,
    uint64_t stream_u64
) {
    if (!fuse_mt || mtiles < 1 || mtiles > 3 || (num_splits != 1 && num_splits != 2 && num_splits != 4)) {
        return fused_decode_attn_otf_mgs_fwd(
            Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr,
            sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, sbt_r, sg_t, ssk_t, sy_t,
            so_s, so_t, so_h, sl_s, sl_t, sl_h, scale_log2, rel_threshold, clip_eps,
            T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits, has_tau, has_nlive,
            use_pdl, trig_wo, tree_mode, stream_u64
        );
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    int bs_shift = (BS > 0 && (BS & (BS - 1)) == 0) ? __builtin_ctz(BS) : -1;
    dim3 grid(1, num_reqs, Hkv * num_splits);
    dim3 block(128, 1, 1);
    const size_t smem_bytes = 32768 + static_cast<size_t>(mtiles) * 4096 + 2560;

    #define LF(WIN, S, MT, TAU, NL, EV, TM) \
        do { \
            if constexpr ((MT) >= 3) { \
                static bool s_smem_ok = false; \
                if (!s_smem_ok) { \
                    cudaError_t attr_err = cudaFuncSetAttribute( \
                        fused_decode_attn_otf_fusedmt_kernel<WIN, S, MT, TAU, NL, EV, TM>, \
                        cudaFuncAttributeMaxDynamicSharedMemorySize, \
                        static_cast<int>(smem_bytes)); \
                    if (attr_err != cudaSuccess) return (int)attr_err; \
                    s_smem_ok = true; \
                } \
            } \
            fused_decode_attn_otf_fusedmt_kernel<WIN, S, MT, TAU, NL, EV, TM><<<grid, block, smem_bytes, stream>>>( \
                (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)RScale, (const float*)Tau, (const uint16_t*)RelProjT, \
                (const uint16_t*)K, (const uint16_t*)V, (const int32_t*)BT, (const int32_t*)SEQ, (const int32_t*)CUQ, \
                (const uint16_t*)GATE, (const uint16_t*)SINK, (float*)OP, (float*)LP, (int32_t*)SEM, (uint16_t*)Y, \
                (const int32_t*)NLivePtr, sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, \
                sbt_r, sg_t, ssk_t, sy_t, so_s, so_t, so_h, sl_s, sl_t, sl_h, \
                scale_log2, rel_threshold, clip_eps, T, NT, BS, bs_shift, E, G, tree_mode); \
            return (int)cudaGetLastError(); \
        } while (0)

    #define DISP_EV(WIN, S, MT, TAU, NL) \
        if (kv_evict_first) { \
            if (tree_mode == 0) { LF(WIN, S, MT, TAU, NL, true,  0); } \
            else                { LF(WIN, S, MT, TAU, NL, true, -1); } \
        } else { \
            if (tree_mode == 0) { LF(WIN, S, MT, TAU, NL, false,  0); } \
            else                { LF(WIN, S, MT, TAU, NL, false, -1); } \
        }

    #define DISP_TAU_NL(WIN, S, MT) \
        if (has_nlive) { \
            if (has_tau) { DISP_EV(WIN, S, MT, true,  true); } \
            else         { DISP_EV(WIN, S, MT, false, true); } \
        } else { \
            if (has_tau) { DISP_EV(WIN, S, MT, true,  false); } \
            else         { DISP_EV(WIN, S, MT, false, false); } \
        }

    #define DISP_MT(WIN, S) \
        if (mtiles == 1)      { DISP_TAU_NL(WIN, S, 1); } \
        else if (mtiles == 2) { DISP_TAU_NL(WIN, S, 2); } \
        else if (mtiles == 3) { DISP_TAU_NL(WIN, S, 3); }

    if (window == 1023) {
        if (num_splits == 1) { DISP_MT(1023, 1); }
        if (num_splits == 2) { DISP_MT(1023, 2); }
        if (num_splits == 4) { DISP_MT(1023, 4); }
    } else if (window < 0) {
        if (num_splits == 1) { DISP_MT(-1, 1); }
        if (num_splits == 2) { DISP_MT(-1, 2); }
        if (num_splits == 4) { DISP_MT(-1, 4); }
    }
    return -2;
    #undef DISP_MT
    #undef DISP_TAU_NL
    #undef DISP_EV
    #undef LF
}

extern "C" int fused_decode_attn_otf_mgs_tree_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y,
    const void* NLivePtr, const void* TreeMeta,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    uint64_t stream_u64
) {
    if (TreeMeta == nullptr || T <= 0 || T > 8 || num_reqs != 1 || G != 6 || mtiles != 4) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    int bs_shift = (BS > 0 && (BS & (BS - 1)) == 0) ? __builtin_ctz(BS) : -1;
    dim3 grid(4, 1, Hkv * num_splits);  // 4-tile BM=16: mt=0..2 primary, mt=3 secondary (ws<=2)!
    dim3 block(128, 1, 1);
    static constexpr size_t SMEM_BYTES = 38656;

    #define LAT(WIN, S, TAU, NL, PD, TR, TM) \
        return (int)launch_maybe_pdl(fused_decode_attn_otf_mgs_kernel<WIN, S, TAU, NL, PD, TR, TM, true>, \
            grid, block, SMEM_BYTES, stream, PD, \
            (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)RScale, (const float*)Tau, (const uint16_t*)RelProjT, \
            (const uint16_t*)K, (const uint16_t*)V, (const int32_t*)BT, (const int32_t*)SEQ, (const int32_t*)CUQ, \
            (const uint16_t*)GATE, (const uint16_t*)SINK, (float*)OP, (float*)LP, (int32_t*)SEM, (uint16_t*)Y, \
            (const int32_t*)NLivePtr, (const int32_t*)TreeMeta, \
            sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, \
            sbt_r, sg_t, ssk_t, sy_t, so_s, so_t, so_h, sl_s, sl_t, sl_h, \
            scale_log2, rel_threshold, clip_eps, T, NT, BS, bs_shift, E, G, tree_mode)

    #define DISP_TMT(WIN, S, TAU, NL, PD, TR) \
        if (tree_mode == 0) { LAT(WIN, S, TAU, NL, PD, TR, 0); } \
        else                { LAT(WIN, S, TAU, NL, PD, TR, -1); }

    #define DISP_FLAGST(WIN, S, TAU) \
        if (has_nlive) { \
            if (use_pdl) { if (trig_wo) { DISP_TMT(WIN, S, TAU, true, true, true); } else { DISP_TMT(WIN, S, TAU, true, true, false); } } \
            else         { DISP_TMT(WIN, S, TAU, true, false, false); } \
        } else { \
            if (use_pdl) { if (trig_wo) { DISP_TMT(WIN, S, TAU, false, true, true); } else { DISP_TMT(WIN, S, TAU, false, true, false); } } \
            else         { DISP_TMT(WIN, S, TAU, false, false, false); } \
        }

    if (window == 1023 && num_splits == 8) {
        if (has_tau) { DISP_FLAGST(1023, 8, true); } else { DISP_FLAGST(1023, 8, false); }
    }
    if (window < 0 && num_splits == 16) {
        if (has_tau) { DISP_FLAGST(-1, 16, true); } else { DISP_FLAGST(-1, 16, false); }
    }
    return -2;
    #undef DISP_FLAGST
    #undef DISP_TMT
    #undef LAT
}
