// rdma_ar.cu: RDMA all-reduce / all-gather for small tensors, one GPU per node (DGX Spark / GB10).
//
// NCCL's net transport costs one CPU-proxy hop per ring step (14 hops for an 8-rank ring all-reduce,
// ~150-190 us inside a CUDA graph). Three modes here, all one kernel launch:
//   one-shot reduce : every rank writes its whole input into each peer's receive slot (one hop),
//                     then sums the world contributions locally in rank order
//   two-shot reduce : reduce-scatter (chunk q of my input -> rank q), rank q sums chunk q in rank
//                     order and broadcasts it (two hops, 4x fewer bytes on the wire than one-shot)
//   all-gather      : one-shot copy, out[r] = input of rank r
// Every element of a reduction is summed by one fixed rule, so all ranks see bit-identical results.
//   column all-gather (g2 colltrim, 2026-10-26): same wire protocol as all-gather; the receiver writes
//                     rank r's [outer, seg] input straight into columns [r*seg, (r+1)*seg) of the
//                     concatenated [outer, world*seg] output (no staging tensor, no movedim copy)
//
// Data path (all buffers are cudaHostAlloc'd = the same LPDDR5X on GB10, registered once per HCA):
//   GPU kernel: stage input in the send slot, last CTA publishes ctrl.send_seq
//   proxy thread: sees send_seq, posts per peer {RDMA WRITE payload, RDMA WRITE 8-byte seq flag}
//                 (same RC QP, so the flag lands after the payload)
//   GPU kernel: spins until every peer's flag for this slot reaches seq, then reduces / copies
// The sequence number lives in device memory and is advanced by the kernel itself, so the ops are
// safe to capture in CUDA graphs (kernel args never change between replays).
//
// g130 rdma (2026-11-02), on top of g2 colltrim 6173b362. Every flag defaults OFF; unset == production behaviour, and the
// production kernels (kPub == 0) are instruction-identical (tests/sass_gate.py). Wire protocol, Params, Ctrl, DevState,
// slots, sum order and rounding are untouched. Read once per instance at construction; WINDOW.md gives the commands.
//   RDMA_AR_SPLIT_POST=1  A   host only: each WR gets its own ibv_post_send (nreq == 1 qualifies for BlueFlame)
//   RDMA_AR_PUB_V1=1      B   kernel: phase-1 publish compares posted_seq with seq - 1 (no send_seq load)
//   RDMA_AR_PUB_V2=1      B   kernel: phase-2 publish has no spin at all (no send2_seq / posted2_seq load)
//   RDMA_AR_HYG_PROXY=1   C   host only: proxy spins on the one doorbell that rings next, reaps CQs after each post
//   RDMA_AR_HYG_POLL=1    C   kernel: GPU spins read the clock on the first miss and once per 1024 passes
// B and C-poll select other kernel instantiations at launch (parallel-load path only). See DESIGN notes in WINDOW.md.
//
// r3-comm (climb round 3, 2026-11-08), on top of g130 d38cfe68. Default OFF; unset == the g130 .so's behaviour: the
// existing kernels are untouched (byte-identical source, SASS-gated) and the r3 kernel ar_kernel_r3 is never launched.
//   DSV41_RDMA_TRIM=1     G06 comm:A  ar_kernel_r3: early posted_seq read, seq advance at kernel entry, single-CTA
//                                     phase-1 publish for messages <= DSV41_RDMA_TRIM_S1_BYTES (24576), single-CTA
//                                     two-shot phase-2 publish when chunk <= 512 vectors. Needs PUB_V1+V2+HYG_POLL.
//   DSV41_RDMA_SENDBUF=1  G26 comm:E  (needs TRIM) a registered send-buffer region (DSV41_RDMA_SENDBUF_MB, 64) at the
//                                     arena end; RdmaAR.send_buffer(nbytes) hands out views; an op whose input IS such
//                                     a buffer is published from it (no staging copy, no per-thread fence, no barrier).
// Same reduction (fp32 from +0.0 in r = 0..7, one bf16 RN), same Params, slots, flags, messages and wire protocol.
//
// r6-comm (climb round 6, 2026-11-09), on top of r3-comm ad0f13a8. Default OFF.
//   DSV41_RDMA_PDL=1|in   C2  (needs DSV41_RDMA_TRIM=1) ar_kernel_r3 launches with programmatic stream serialization;
//                             before griddepcontrol.wait it only reads ctrl->posted_seq (monotonic, host-written).
//   DSV41_RDMA_PDL=trig   C2  as 1, plus griddepcontrol.launch_dependents after the last flag wait.
// Same FP stream as r3 / live (gate/fpflow.py on the binary); scheduling only.
//
// r7-comm2 (climb round 7, 2026-11-09), on top of r6-comm 2bff8a89. Default OFF.
//   DSV41_RDMA_PROJ=1     HT13 (needs DSV41_RDMA_TRIM=1) RdmaAR.all_reduce_proj: the bf16 [rows <= 6, 5120] all-reduce
//                             also writes y = the dsv41_oblit _proj_rows projection of its final bf16 output, computed in
//                             the kernel tail with the served Triton program's exact op order (gate/projgate.py).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cstddef>
#include <cctype>  // r3-comm: std::tolower in env_r3_norm
#include <sys/mman.h>
#include <fstream>
#include <ctime>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include <unordered_map>

namespace {

constexpr int kSlots = 4;
constexpr int kFlagStride = 8;    // s7: 64B (8 x uint64_t) per-peer flag stride -> 0 CMN-700 cache-line false sharing
constexpr int kThreads = 512;
constexpr int kRowsThreads = 576; // s8: 18 warps = exact row_vec (4608*2/16) -> 1-pass M=8 P1/P2 & 1-CTA n_live=1!
constexpr int kSignalEvery = 32;  // request a CQE every N posts per QP so the SQ drains
constexpr uint32_t kBf12RawFlag = 0xFFFFFFFFu; // s8: BF12-MAX14+Z automatic raw fallback sentinel

enum Mode : int { kOneShot = 0, kGather = 1, kTwoShot = 2, kGatherCols = 3 };  // kGatherCols: g2 colltrim
// g130: kernel variants, chosen per launch by instantiation and never by a kernel argument, so kPub == 0 is production
enum PubBits : int { kPubV1 = 1, kPubV2 = 2, kPollThin = 4 };
// r3-comm: bits above the g130 mask select ar_kernel_r3 instantiations (never a kernel argument)
enum R3Bits : int { kR3Trim = 8, kR3S1 = 16, kR3P2 = 32, kR3Sbuf = 64 };
enum R6Bits : int { kR6Pdl = 128, kR6Trig = 256 };  // r6-comm C2: PDL in-edge / early trigger (instantiation bits)
enum R7Bits : int { kR7Proj = 512 };  // r7-comm2 HT13: projection tail (instantiation bit, bf16 one/two-shot only)
constexpr uint64_t kModeSrc = 256;    // r3 G26: descriptor mode bit: the payload is at ctrl.src_off, not in the send slot
constexpr uint64_t kModeInline = 512; // s6: 16-byte kGather payload is inline in ctrl.inline_lo/hi -> single 24B inline WR
constexpr uint64_t kModePeerLens = 1024; // c32: per-peer compressed Phase-1 byte lengths in CtrlC32Slot::p1_bytes[q]
constexpr uint64_t kModeP1Stripe = 2048; // c32: Phase-1 2-QP dual-root PCIe 50/50 striping across Root 0 + Root 2
constexpr int kC32Threads = 256;         // c32: 8 warps/CTA -> 2 CTAs/SM co-resident (__launch_bounds__(256, 2))
constexpr int kC32Blocks = 43;           // c32: 1 poller CTA + 42 workers (6/peer) = 1.0 wave on 48-SM GB10!
constexpr int kSbufRingPerDomain = 4;    // c32: 4 slots/domain x 2 domains (target, draft) = 8 slots -> 36 MiB vs 680 MiB

#define CHECK_CUDA(x)                                                                        \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess)                                                                   \
      throw std::runtime_error(std::string("rdma_ar CUDA: ") + cudaGetErrorString(e_) +      \
                               " at " #x);                                                   \
  } while (0)
#define CHECK_IB(cond, what)                                                                 \
  do {                                                                                       \
    if (!(cond))                                                                             \
      throw std::runtime_error(std::string("rdma_ar ibverbs: ") + (what) +                   \
                               " errno=" + std::to_string(errno) + " " + strerror(errno));  \
  } while (0)

struct alignas(64) Ctrl {
  uint64_t send_seq;    // GPU -> proxy: phase-1 message `seq` is staged in send slot seq % kSlots
  uint64_t send_bytes;  // whole message bytes (written before send_seq)
  uint64_t mode;        // Mode | kModeSrc | kModeInline
  uint64_t chunk_vecs;  // two-shot chunk size in 16-byte vectors
  uint64_t src_off;     // r3 G26: arena offset of the payload when mode & kModeSrc (was pad0[0]; layout unchanged)
  uint64_t inline_lo;   // s6: first 8 bytes of inline 16B kGather payload (was pad0[1]; same cache line as send_seq!)
  uint64_t inline_hi;   // s6: second 8 bytes of inline 16B kGather payload (was pad0[2])
  uint64_t pad0_tail;   // off 56
  uint64_t posted_seq;  // proxy -> GPU: phase-1 writes for this seq have been posted
  uint64_t pad1[7];
  uint64_t send2_seq;   // GPU -> proxy: two-shot reduced chunk staged in send2 slot
  uint64_t send2_bytes;
  uint64_t pad2[6];
  uint64_t posted2_seq;  // proxy -> GPU: phase-2 writes posted
  uint64_t pad3[7];
  uint64_t pub_time;    // GPU globaltimer at the phase-1 publish (for proxy hand-off timing)
  uint64_t pad4[7];
  uint64_t ping;        // GPU -> proxy hand-off probe
  uint64_t pad5[7];
  uint64_t pong;        // proxy echo
  uint64_t pad6[7];
};
static_assert(sizeof(Ctrl) == 448 && offsetof(Ctrl, posted_seq) == 64 && offsetof(Ctrl, src_off) == 32 &&
                  offsetof(Ctrl, inline_lo) == 40,
              "r3/s6: Ctrl layout must not move");

// s8: Per-slot level doorbell (RoCEnante P1, RDMA_AR_LEVEL_DB=1), placed at arena offset 512..1023 inside page 0.
// Separates Phase-1 (line 0) and Phase-2 (line 1) descriptors per slot, eliminating GPU posted_seq spins and
// primary<->helper proxy cross-L3 p2_posted1_/p2_posted2_ synchronization!
struct alignas(64) CtrlSlot {
  // Cache line 0 (off 0..63): Phase 1
  uint64_t send_seq;
  uint64_t send_bytes;
  uint64_t mode;
  uint64_t chunk_vecs;
  uint64_t src_off;
  uint64_t inline_lo;
  uint64_t inline_hi;
  uint64_t pub_time;
  // Cache line 1 (off 64..127): Phase 2
  uint64_t send2_seq;
  uint64_t send2_bytes;
  uint64_t pad2[6];
};
static_assert(sizeof(CtrlSlot) == 128 && offsetof(CtrlSlot, send2_seq) == 64, "s8: CtrlSlot must be 128B");
constexpr size_t kCtrlSlotsOff = 512;  // [512, 1024) inside arena page 0 (< 4096)

// s8: Isolated 64B cache line for helper proxy posted_seq (offset 1024 inside page 0 -> 0 cross-L3 sharing with primary!)
struct alignas(64) CtrlAux {
  uint64_t posted_seq_helper;
  uint64_t posted2_seq_helper;
  uint64_t pad[6];
};
static_assert(sizeof(CtrlAux) == 64, "s8: CtrlAux must be 64B");
constexpr size_t kCtrlAuxOff = 1024;   // [1024, 1088) inside arena page 0 (< 4096)

// c32: Per-slot extension block inside page 0 (offset 1280..1535 < 4096):
// holds per-peer BF12-G128-OVF Phase-1 byte lengths + monotonic p2_stripe_seq (0 stale state on mixed C=1/C=32!).
struct alignas(64) CtrlC32Slot {
  uint32_t p1_bytes[8];
  uint64_t p2_stripe_seq;
  uint32_t pad[6];
};
static_assert(sizeof(CtrlC32Slot) == 64, "c32: CtrlC32Slot must be 64B");
constexpr size_t kCtrlC32Off = 1280;   // [1280, 1536) inside arena page 0 (< 4096)

struct alignas(16) Bf12G128Hdr {
  uint32_t n_vec;
  uint32_t n_tiles;
  uint32_t n_ovf;
  uint32_t total_bytes;
};
static_assert(sizeof(Bf12G128Hdr) == 16, "c32: Bf12G128Hdr must be 16B");

struct alignas(64) InlineSlot {
  uint4 data;      // 16-byte payload
  uint64_t flag;   // 8-byte seq flag (lands last in same 64B cache line / PCIe TLP)
  uint64_t pad[5]; // pad to 64 bytes
};
static_assert(sizeof(InlineSlot) == 64 && offsetof(InlineSlot, flag) == 16, "s6: InlineSlot must be 64B");

struct DevState {
  unsigned long long seq;  // last completed op
  unsigned int arrive;     // CTA arrival counter, phase 1 staging
  unsigned int arrive2;    // CTA arrival counter, phase 2 staging
  unsigned int depart;     // CTA departure counter
  // timing (ns, globaltimer; recorded by thread 0 of CTA 0 / the publishing thread)
  unsigned long long t_start, t_pub;
  long long sum_stage, sum_net, sum_tail, sum_total;  // start->pub, pub->flags, flags->end, start->end
  unsigned long long n_timed;
  unsigned int entered;  // r3 G06 (iv): CTA entry counter of ar_kernel_r3 (appended: existing offsets unchanged)
  unsigned int arrive3;  // r7-comm2 HT13: CTA arrival counter of the projection tail (appended; back to 0 after each op)
  unsigned long long p1_done;     // s6: GPU-L2-cached relay flag written by CTA 0 after phase-1 peer flags arrive
  unsigned long long p2_done;     // s6: GPU-L2-cached relay flag written by CTA 0 after phase-2 peer flags arrive
  unsigned long long ar_done_cnt; // s7-v3: GPU-L2 cooperative stop-flag for l2pf_paced_stop_kernel (written at r6_gdc_launch)
  // s8 appended fields (existing offsets unchanged):
  unsigned long long p2_peer_seq[8];  // s8: per-peer L2-cached arrival seq (monotonic, never needs reset!)
  unsigned long long row_done_seq[8]; // s8: per-row 1-shot L2-cached reduction seq (monotonic, never needs reset!)
  unsigned int row_ready[8];          // s8: ConvRes preload arrival count for nq>1 ConvRes==ConvOut aliasing guard
  // c32 appended fields (existing offsets unchanged):
  unsigned int c32_arrive_a;
  unsigned int c32_arrive_b;
  unsigned int c32_arrive2_a;
  unsigned int c32_arrive2_b;
  unsigned int c32_p1_ovf[kSlots][8];
  unsigned int c32_p2_ovf[kSlots];
};

struct Params {
  const uint4* in;
  uint4* out;
  long long nvec;        // message size in 16-byte vectors
  long long chunk;       // two-shot chunk size in vectors
  char* send;            // kSlots * max_bytes
  char* send2;           // kSlots * max_chunk
  const char* recv;      // kSlots * world * recv_stride (was max_bytes)
  const char* recv2;     // kSlots * world * recv2_stride (was max_chunk)
  const uint64_t* flags;   // kSlots * world * kFlagStride (64B per peer flag)
  const uint64_t* flags2;  // kSlots * world * kFlagStride (64B per peer flag)
  Ctrl* ctrl;
  DevState* st;
  long long max_bytes, max_chunk; // stride per peer slot in recv / recv2 (= align(cap+64, 64) in s6)
  int rank, world;
  long long timeout_ns;
  int timing;  // record per-call stage timings (RDMA_AR_TIMING=1)
  long long seg;  // g2 colltrim kGatherCols only: input row width in 16-byte vectors (output row = world * seg)
  // w3 rowar (INKLING_ROW_AR): when rows_ptr != nullptr, ar_kernel_r3 reads the live row count n from *rows_ptr ON DEVICE
  // (after the grid-dependency wait), reduces only the first n rows (nvec = n * row_vec, chunk recomputed) and zero-fills
  // the remaining output vectors up to nvec_full. Appended fields: the Ctrl/wire protocol is unchanged (the proxy already
  // learns the byte count from ctrl.send_bytes / chunk_vecs / send2_bytes, all written by the kernel).
  const int* rows_ptr = nullptr;
  long long row_vec = 0;     // 16-byte vectors per row
  long long nvec_full = 0;   // vectors of the full tensor (= rows_full * row_vec)
  int oneshot_max_rows = 0;  // s5: dynamic 1-shot when n <= oneshot_max_rows (0 = always two-shot)
  int rows_p2 = 0;           // s5/s6/s7: 0=multi-CTA, 1=single-CTA, 2=split, 3=per-peer overlapped phase-2 copy
  long long s1_bytes = 24576;// s5: single-CTA phase-1 staging byte threshold in ar_kernel_rows
  const int* skip_ptr = nullptr; // s5: conditional skip for kGather (non-zero -> skip op)
  // s6 additions:
  int piggyback = 0;         // 1 = 8-byte flag piggybacked at recv_slot + expected_bytes (single 2-SGE WR)
  int l2_relay = 0;          // 1 = only CTA 0 polls sys flags, wakes CTAs 1..N via GPU-L2 relay + fence.acquire.sys
  unsigned int poll_sleep_ns = 0; // nanosleep.u32 backoff on poll miss (e.g. 32 ns to cut C2C traffic)
  int inline_ag = 0;         // 1 = 16-byte kGather uses ctrl.inline_lo/hi + single 24B inline WR to inline_recv
  const InlineSlot* inline_recv = nullptr; // kSlots * world InlineSlots (64B each)
  // s8 additions:
  int level_db = 0;          // 1 = per-slot level doorbell (ctrl_slots[slot], zero cross-proxy sync)
  int multi_proxy = 0;       // 1 = dual-proxy active (level_db checks min(posted_seq, posted_seq_helper))
  int wire_bf12 = 0;         // 1 = 0-ULP lossless BF12-MAX14+Z compression on Phase-2 2-shot payloads
  int bf12_os4 = 0;          // 1 = 0-ULP lossless BF12-MAX14+Z 1-shot unlock for n_live == 4 (~27.8 KB wire!)
  CtrlSlot* ctrl_slots = nullptr;  // kSlots * 128B descriptors at arena_dev + 512
  CtrlAux* ctrl_aux = nullptr;     // helper proxy posted_seq at arena_dev + 1024
  uint4* reduced_buf = nullptr;    // GPU-device L2-hot [8, 576] uint4 (73,728 B) for single-__ldcv L2 staging
  uint4* bf12_os_scratch = nullptr;// GPU-device L2-hot [8, 2304] uint4 (294,912 B) for overlapped BF12-OS4 unpack
  // c32 additions (all default 0/nullptr -> 0 impact on C=1):
  const uint64_t* flags_qp2 = nullptr;   // separate 64B-strided QP2 Phase-1 flags (Root 2, 0 false sharing!)
  const uint64_t* flags2_qp2 = nullptr;  // separate 64B-strided QP2 Phase-2 flags (Root 2, 0 false sharing!)
  CtrlC32Slot* ctrl_c32 = nullptr;       // kSlots * 64B per-peer lens + P2 stripe flags at arena_dev + 1280
  long long recv_stride = 0;             // decoupled recv peer stride (0 -> use p.max_bytes)
  int dual_qp = 0;                       // 1 = 2-QP dual-root striping active
  int c32_med_gather = 0;                // 1 = opt-in ar_kernel_med_gather for 9 <= nvec <= 1024
  long long stripe_min_bytes = 32768;    // per-peer byte floor for 2-QP striping (32 KiB -> M >= 32)
};

// s8: Parameters for 0-ULP fused RDMA AllReduce + SConv + AttnRes + RMSNorm (ar_sconv_attnres_kernel)
struct SconvAttnResParams {
  const uint16_t* conv_w;       // [4608, 4] bf16 contiguous
  uint16_t* cache;              // [num_blocks, 1, N_cache, D_cache] bf16
  uint16_t* conv_out;           // [T, 4608] bf16 contiguous
  uint16_t* aux_out;            // [T, 4608] bf16 contiguous (or nullptr)
  const uint16_t* conv_res;     // [T, 4608] bf16 contiguous (or nullptr; may alias conv_out!)
  const uint16_t* conv_addend;  // [T, 4608] bf16 contiguous (or nullptr)
  const int64_t* pos;           // [T] int64
  const int32_t* seq_idx;       // [T] int32
  const int64_t* slot_map;      // [T] int64
  const int32_t* block_table;   // [num_reqs, max_blocks] int32
  const int32_t* q_start;       // [T] int32
  const __nv_bfloat16* stack;   // [N_STACK, T, 4608] bf16 (stride_s along dim 0)
  const __nv_bfloat16* mix_last;// [T, 4608] bf16 (or nullptr; mutually exclusive w/ conv_addend)
  const float* queries;         // [nq, 4608] fp32 contiguous
  const __nv_bfloat16* norm_w;  // [nq, 4608] bf16 contiguous
  __nv_bfloat16* out;           // [nq, T, 4608] bf16 contiguous
  int64_t stride_c_blk;
  int64_t stride_c_n;
  int64_t stride_bt;
  int64_t stream_off;
  int64_t stride_s;
  int max_blocks;
  int n_cache;
  int n_shift;                  // >= 0 if n_cache is power-of-2, else -1
  int nq;                       // 1 or 3
  int T;                        // <= 8 (typically 8)
  float scale;                  // 5**-0.5
  float eps;                    // 1e-6
  float norm_eps;               // norm.eps
};

struct alignas(16) Bf12Hdr {
  uint32_t n_vec;       // number of uint4 (8-bf16) vectors
  uint32_t max_e;       // max non-zero biased exponent (1..255, or 0 if all exp==0)
  uint32_t n_esc;       // number of escape uint32_t entries, or kBf12RawFlag for uncompressed fallback
  uint32_t total_bytes; // 16-byte aligned total wire size
};
static_assert(sizeof(Bf12Hdr) == 16, "s8: Bf12Hdr must be 16B");

// r7-comm2 HT13: the projection operands of one kR7Proj launch (a second kernel parameter; zero for every other launch)
struct R7Proj {
  const float* d;   // unit direction, fp32 [kR7H], 16-byte aligned (dsv41_oblit._vec)
  uint4* y;         // projected output, bf16 [rows, kR7H], contiguous, 16-byte aligned
  long long rows;   // 1 .. kR7MaxRows
};
constexpr int kR7H = 5120;        // DSv4.1 hidden size: _proj_rows H (BLOCK 8192, num_warps 8, 256 threads per row)
constexpr int kR7MaxRows = 6;     // two 256-thread groups x kR7RowsPerGroup rows
constexpr int kR7RowsPerGroup = 3;

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// r3-comm helpers (used by ar_kernel_r3 only)
__device__ __forceinline__ uint64_t ld_relaxed_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// s6 GPU-L2 relay helpers (cut C2C polling traffic by 9x across 9 CTAs while keeping formal PTX sys-scope causality)
__device__ __forceinline__ unsigned long long ld_acquire_gpu(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_gpu(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_release_sys_ull(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_acquire_sys_ull(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acquire_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}
__device__ __forceinline__ void s6_nanosleep(unsigned int ns) {
#if __CUDA_ARCH__ >= 700
  if (ns > 0) asm volatile("nanosleep.u32 %0;" ::"r"(ns));
#endif
}
__device__ __forceinline__ unsigned int atom_add_acqrel_gpu(unsigned int* p, unsigned int v) {
  unsigned int old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ unsigned int ld_acquire_gpu_u32(const unsigned int* p) {
  unsigned int v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu_u32(unsigned int* p, unsigned int v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void r3_trap() {
  asm volatile("trap;");
}
// r6-comm C2 (sm_90+ PTX): wait for the prerequisite grid (completion + memory flush); allow dependents to launch
__device__ __forceinline__ void r6_gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void r6_gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// r7-comm2 HT13: bf16 halves of a 32-bit word, widened exactly to fp32 (the served kernel's cvt.f32.bf16)
__device__ __forceinline__ float r7_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float r7_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ void r7_unpack(const uint4 v, float* f) {
  f[0] = r7_lo(v.x); f[1] = r7_hi(v.x); f[2] = r7_lo(v.y); f[3] = r7_hi(v.y);
  f[4] = r7_lo(v.z); f[5] = r7_hi(v.z); f[6] = r7_lo(v.w); f[7] = r7_hi(v.w);
}
__device__ __forceinline__ unsigned r7_pack(float lo, float hi) {  // cvt.rn.bf16x2.f32 d, hi, lo (RN, one rounding)
  const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&b);
}
// The served dsv41_oblit._proj_rows program (Triton 3.7.1, sm_121a, H 5120, BLOCK 8192, 8 warps), row by row, on rows
// g, g + 2, g + 4 of out (g = this thread's 256-thread group). Every FP op and its order are the cubin's (ref/triton/):
//   acc = x[1] * d[1]; acc = fma(x[j], d[j], acc) for j = 0, 2..31 (register order: chunk c = 0..3 at c*2048 + t*8,
//   8 elements each; masked elements are +0.0); warp butterfly xor 16, 8, 4, 2, 1 (acc + shfl(acc)); lane 0 -> smem[w];
//   lanes 0..7 of warp 0: butterfly xor 4, 2, 1 over smem[lane]; lane 0 -> smem[0]; y = bf16_rn(fma(-s, d, x)).
// All threads of the CTA must call it (two __syncthreads).
__device__ __forceinline__ void r7_project(const uint4* __restrict__ xo, const R7Proj& q) {
  __shared__ float s_w[2][kR7RowsPerGroup][8];
  __shared__ float s_s[2][kR7RowsPerGroup];
  const int g = (int)(threadIdx.x >> 8);
  const int t = (int)(threadIdx.x & 255u);
  const int lane = t & 31, warp = t >> 5;
  float d[32];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int e = c * 2048 + t * 8;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
    if (e < kR7H) {
      a = __ldcg(reinterpret_cast<const float4*>(q.d + e));
      b = __ldcg(reinterpret_cast<const float4*>(q.d + e + 4));
    }
    d[c * 8 + 0] = a.x; d[c * 8 + 1] = a.y; d[c * 8 + 2] = a.z; d[c * 8 + 3] = a.w;
    d[c * 8 + 4] = b.x; d[c * 8 + 5] = b.y; d[c * 8 + 6] = b.z; d[c * 8 + 7] = b.w;
  }
  uint4 xv[kR7RowsPerGroup][4];
#pragma unroll
  for (int k = 0; k < kR7RowsPerGroup; ++k) {
    const long long row = 2 * k + g;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int e = c * 2048 + t * 8;
      xv[k][c] = make_uint4(0u, 0u, 0u, 0u);
      if (row < q.rows && e < kR7H) xv[k][c] = __ldcg(xo + row * (kR7H / 8) + e / 8);
    }
  }
  float acc[kR7RowsPerGroup];
#pragma unroll
  for (int k = 0; k < kR7RowsPerGroup; ++k) {
    float x[32];
#pragma unroll
    for (int c = 0; c < 4; ++c) r7_unpack(xv[k][c], x + c * 8);
    float a = __fmul_rn(x[1], d[1]);
    a = __fmaf_rn(x[0], d[0], a);
#pragma unroll
    for (int j = 2; j < 32; ++j) a = __fmaf_rn(x[j], d[j], a);
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, m));
    acc[k] = a;
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < kR7RowsPerGroup; ++k) s_w[g][k][warp] = acc[k];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int k = 0; k < kR7RowsPerGroup; ++k) {
      float v = s_w[g][k][lane & 7];
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m));
      if (lane == 0) s_s[g][k] = v;
    }
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < kR7RowsPerGroup; ++k) {
    const long long row = 2 * k + g;
    if (row >= q.rows) continue;
    const float ns = -s_s[g][k];
    float x[32];
#pragma unroll
    for (int c = 0; c < 4; ++c) r7_unpack(xv[k][c], x + c * 8);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int e = c * 2048 + t * 8;
      if (e < kR7H) {
        float y[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) y[j] = __fmaf_rn(ns, d[c * 8 + j], x[c * 8 + j]);
        q.y[row * (kR7H / 8) + e / 8] = make_uint4(r7_pack(y[0], y[1]), r7_pack(y[2], y[3]), r7_pack(y[4], y[5]),
                                                   r7_pack(y[6], y[7]));
      }
    }
  }
}

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void spin_until(const uint64_t* f, uint64_t seq, const Params& p, int who, int phase) {
  const unsigned long long t0 = now_ns();
  while (ld_acquire_sys(f) < seq) {
    if (now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar rank %d: timeout waiting for %d phase %d (seq %llu)\n", p.rank, who, phase,
             (unsigned long long)seq);
      asm volatile("trap;");
    }
  }
}

// g130 RDMA_AR_HYG_POLL: same wait and same timeout contract; the clock is read on the first miss and on every 1024th
// pass instead of on every pass, so a timeout can be late by about 1024 polls (about 1 ms against a 600 s limit).
__device__ __forceinline__ void spin_until_thin(const uint64_t* f, uint64_t seq, const Params& p, int who, int phase) {
  if (ld_acquire_sys(f) >= seq) return;
  const unsigned long long t0 = now_ns();
  unsigned int it = 0;
  while (ld_acquire_sys(f) < seq) {
    if (p.poll_sleep_ns) s6_nanosleep(p.poll_sleep_ns);
    if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar rank %d: timeout waiting for %d phase %d (seq %llu)\n", p.rank, who, phase,
             (unsigned long long)seq);
      asm volatile("trap;");
    }
  }
}
// s7-v3: explicit-sleep variant for 7-SM Phase-2 peer polls (cuts C2C snoop pressure 5-10x during drafter L2PF!)
__device__ __forceinline__ void spin_until_thin_sleep(const uint64_t* f, uint64_t seq, const Params& p, int who,
                                                      int phase, unsigned int sleep_ns) {
  if (ld_acquire_sys(f) >= seq) return;
  const unsigned long long t0 = now_ns();
  unsigned int it = 0;
  while (ld_acquire_sys(f) < seq) {
    if (sleep_ns) s6_nanosleep(sleep_ns);
    if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar rank %d: timeout waiting for %d phase %d (seq %llu)\n", p.rank, who, phase,
             (unsigned long long)seq);
      asm volatile("trap;");
    }
  }
}
// s6/s7-v2: wait for phase-1 peer flags (64B-strided separate flags + CTA-0 L2 relay with no GPU-L2 nanosleep)
// expected_peer_bytes = bytes each peer wrote into recv[slot][peer] (nvec*16 for 1-shot/gather, len*16 for 2-shot p1)
// Note: same-CTA ld.acquire.sys -> __syncthreads() -> __ldcv is PTX MCM cumulative (matches S5/S6-v2);
// fence_acquire_sys() is strictly kept on cross-CTA L2-relay receiver CTAs (blockIdx.x > 0).
__device__ __forceinline__ void s6_wait_p1(const Params& p, int slot, unsigned long long seq,
                                           long long expected_peer_bytes, bool all_ctas_wait) {
  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const char* rbase = p.recv + (long long)slot * p.world * rstride;
  if (p.l2_relay && all_ctas_wait && gridDim.x > 1) {
    if (blockIdx.x == 0) {
      if (threadIdx.x < p.world && threadIdx.x != p.rank) {
        const uint64_t* fp = p.piggyback
                                 ? reinterpret_cast<const uint64_t*>(rbase + (long long)threadIdx.x * rstride +
                                                                     expected_peer_bytes)
                                 : (p.flags + (slot * p.world + threadIdx.x) * kFlagStride);
        spin_until_thin(fp, seq, p, threadIdx.x, 1);
      }
      __syncthreads();
      if (threadIdx.x == 0) st_release_sys_ull(&p.st->p1_done, seq);
    } else {
      if (threadIdx.x == 0) {
        if (ld_acquire_gpu(&p.st->p1_done) < seq) {
          const unsigned long long t0 = now_ns();
          unsigned int it = 0;
          while (ld_acquire_gpu(&p.st->p1_done) < seq) {
            if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
              printf("rdma_ar rank %d: timeout waiting for L2 relay phase 1 (seq %llu)\n", p.rank,
                     (unsigned long long)seq);
              r3_trap();
            }
          }
        }
        (void)ld_acquire_sys_ull(&p.st->p1_done);
      }
      __syncthreads();
      fence_acquire_sys();
    }
  } else {
    if (threadIdx.x < p.world && threadIdx.x != p.rank) {
      const uint64_t* fp = p.piggyback
                               ? reinterpret_cast<const uint64_t*>(rbase + (long long)threadIdx.x * rstride +
                                                                   expected_peer_bytes)
                               : (p.flags + (slot * p.world + threadIdx.x) * kFlagStride);
      spin_until_thin(fp, seq, p, threadIdx.x, 1);
    }
    __syncthreads();
  }
}

// s6/s7-v2: wait for phase-2 peer flags (each peer r wrote lr(r)*16 bytes into recv2[slot][r])
__device__ __forceinline__ void s6_wait_p2(const Params& p, int slot, unsigned long long seq, bool all_ctas_wait) {
  const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
  if (p.l2_relay && all_ctas_wait && gridDim.x > 1) {
    if (blockIdx.x == 0) {
      if (threadIdx.x < p.world && threadIdx.x != p.rank) {
        const long long br = (long long)threadIdx.x * p.chunk;
        const long long lr = max(0LL, min(p.chunk, p.nvec - br));
        const uint64_t* fp = p.piggyback
                                 ? reinterpret_cast<const uint64_t*>(r2 + (long long)threadIdx.x * p.max_chunk +
                                                                     lr * 16)
                                 : (p.flags2 + (slot * p.world + threadIdx.x) * kFlagStride);
        spin_until_thin(fp, seq, p, threadIdx.x, 2);
      }
      __syncthreads();
      if (threadIdx.x == 0) st_release_sys_ull(&p.st->p2_done, seq);
    } else {
      if (threadIdx.x == 0) {
        if (ld_acquire_gpu(&p.st->p2_done) < seq) {
          const unsigned long long t0 = now_ns();
          unsigned int it = 0;
          while (ld_acquire_gpu(&p.st->p2_done) < seq) {
            if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
              printf("rdma_ar rank %d: timeout waiting for L2 relay phase 2 (seq %llu)\n", p.rank,
                     (unsigned long long)seq);
              r3_trap();
            }
          }
        }
        (void)ld_acquire_sys_ull(&p.st->p2_done);
      }
      __syncthreads();
      fence_acquire_sys();
    }
  } else {
    if (threadIdx.x < p.world && threadIdx.x != p.rank) {
      const long long br = (long long)threadIdx.x * p.chunk;
      const long long lr = max(0LL, min(p.chunk, p.nvec - br));
      const uint64_t* fp = p.piggyback
                               ? reinterpret_cast<const uint64_t*>(r2 + (long long)threadIdx.x * p.max_chunk + lr * 16)
                               : (p.flags2 + (slot * p.world + threadIdx.x) * kFlagStride);
      spin_until_thin(fp, seq, p, threadIdx.x, 2);
    }
    __syncthreads();
  }
}

// s7-v2: GPU-L2 Phase-1 completion gate so peer CTAs generate ZERO C2C traffic during Phase 1!
__device__ __forceinline__ void s7_wait_p1_gate_l2(const Params& p, unsigned long long seq) {
  if (threadIdx.x == 0) {
    if (ld_acquire_gpu(&p.st->p1_done) < seq) {
      const unsigned long long t0 = now_ns();
      unsigned int it = 0;
      while (ld_acquire_gpu(&p.st->p1_done) < seq) {
        if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
          printf("rdma_ar rank %d: timeout waiting for L2 p1_done gate (seq %llu)\n", p.rank,
                 (unsigned long long)seq);
          r3_trap();
        }
      }
    }
  }
}
// kPub without kPollThin expands to exactly the production spin_until call.
#define G130_SPIN(f, s, who, phase)                   \
  do {                                                \
    if constexpr ((kPub & kPollThin) != 0)            \
      spin_until_thin((f), (s), p, (who), (phase));   \
    else                                              \
      spin_until((f), (s), p, (who), (phase));        \
  } while (0)

// True for exactly one thread: thread 0 of the last CTA through a grid-wide arrival counter (all CTAs'
// prior writes, fenced at system scope, are visible to it). That thread then publishes to the proxy.
__device__ __forceinline__ bool last_cta(unsigned int* arrive, int* s_last) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) *s_last = (atomicAdd(arrive, 1u) == gridDim.x - 1);
  __syncthreads();
  if (*s_last && threadIdx.x == 0) {
    *arrive = 0;
    __threadfence_system();
    return true;
  }
  return false;
}

__device__ __forceinline__ void acc8(float* acc, const uint4& v) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __bfloat1622float2(h[k]);
    acc[2 * k] += f.x;
    acc[2 * k + 1] += f.y;
  }
}
__device__ __forceinline__ uint4 pack8(const float* acc) {
  uint4 o;
  __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) oh[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  return o;
}

// dtype-generic accumulate/pack over one 16-byte vector: 8 bf16 or (kF32) 4 fp32 values.
// fp32 inputs are summed in fp32 in the same fixed rank order (DSv4.1 addition, 2026-10-21).
template <bool kF32>
__device__ __forceinline__ void accv(float* acc, const uint4& v) {
  if constexpr (kF32) {
    acc[0] += __uint_as_float(v.x);
    acc[1] += __uint_as_float(v.y);
    acc[2] += __uint_as_float(v.z);
    acc[3] += __uint_as_float(v.w);
  } else {
    acc8(acc, v);
  }
}
template <bool kF32>
__device__ __forceinline__ uint4 packv(const float* acc) {
  if constexpr (kF32) {
    uint4 o;
    o.x = __float_as_uint(acc[0]);
    o.y = __float_as_uint(acc[1]);
    o.z = __float_as_uint(acc[2]);
    o.w = __float_as_uint(acc[3]);
    return o;
  } else {
    return pack8(acc);
  }
}

// GPU<->proxy round trip through host-pinned memory, timed on the GPU clock only.
__global__ void pingpong_kernel(Ctrl* ctrl, int n, unsigned long long base, unsigned long long* out_ns) {
  const unsigned long long t0 = now_ns();
  for (int i = 1; i <= n; ++i) {
    st_release_sys(&ctrl->ping, base + i);
    while (ld_acquire_sys(&ctrl->pong) != base + i) {
    }
  }
  *out_ns = now_ns() - t0;
}

template <int kMode, bool kF32>
__global__ void __launch_bounds__(kThreads) ar_kernel(Params p) {
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  if (threadIdx.x == 0) s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) p.st->t_start = now_ns();
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;

  // 1) stage the input in this rank's send slot and hand it to the proxy
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
  for (long long i = tid; i < p.nvec; i += stride) send4[i] = p.in[i];
  if (last_cta(&p.st->arrive, &s_last)) {
    // the proxy must have consumed the previous message on this channel before its descriptor is
    // overwritten (compare with what was last published, not seq-1: modes interleave)
    spin_until(&p.ctrl->posted_seq, ld_acquire_sys(&p.ctrl->send_seq), p, -1, 0);
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kMode;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = (uint64_t)p.chunk;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
    }
    st_release_sys(&p.ctrl->send_seq, seq);
  }

  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    spin_until(p.flags + (slot * p.world + threadIdx.x) * kFlagStride, seq, p, threadIdx.x, 1);
  __syncthreads();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (int r = 0; r < p.world; ++r) {
      const uint4* src = reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes);
      uint4* dst = p.out + (long long)r * p.nvec;
      for (long long i = tid; i < p.nvec; i += stride) dst[i] = (r == p.rank) ? p.in[i] : __ldcv(src + i);
    }
  } else if constexpr (kMode == kGatherCols) {
    // g2 colltrim: vector i of rank r's input (row i / seg, column i % seg) goes to output row i / seg,
    // column r * seg + i % seg. Same loads as kGather; only the destination index differs.
    const unsigned useg = (unsigned)p.seg;
    const long long orow = (long long)p.world * p.seg;
    for (int r = 0; r < p.world; ++r) {
      const uint4* src = reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes);
      uint4* dst = p.out + (long long)r * p.seg;
      for (long long i = tid; i < p.nvec; i += stride) {
        const long long row = (long long)((unsigned)i / useg);
        dst[row * orow + (i - row * p.seg)] = (r == p.rank) ? p.in[i] : __ldcv(src + i);
      }
    }
  } else if constexpr (kMode == kOneShot) {
    for (long long i = tid; i < p.nvec; i += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < p.world; ++r)
        accv<kF32>(acc, (r == p.rank) ? p.in[i]
                                      : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i));
      p.out[i] = packv<kF32>(acc);
    }
  } else {
    // two-shot: peers wrote chunk `rank` of their input at offset 0 of recv[slot][peer]
    const long long b = (long long)p.rank * p.chunk;
    const long long len = max(0LL, min(p.chunk, p.nvec - b));
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    for (long long i = tid; i < len; i += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < p.world; ++r)
        accv<kF32>(acc, (r == p.rank) ? p.in[b + i]
                                      : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i));
      const uint4 o = packv<kF32>(acc);
      p.out[b + i] = o;
      send2[i] = o;
    }
    if (last_cta(&p.st->arrive2, &s_last)) {
      spin_until(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), p, -2, 0);
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
      st_release_sys(&p.ctrl->send2_seq, seq);
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      spin_until(p.flags2 + (slot * p.world + threadIdx.x) * kFlagStride, seq, p, threadIdx.x, 2);
    __syncthreads();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    for (int r = 0; r < p.world; ++r) {
      if (r == p.rank) continue;
      const long long br = (long long)r * p.chunk;
      const long long lr = max(0LL, min(p.chunk, p.nvec - br));
      const uint4* src = reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk);
      for (long long i = tid; i < lr; i += stride) p.out[br + i] = __ldcv(src + i);
    }
  }

  // 3) last CTA out advances the sequence for the next call
  __syncthreads();
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence();
    const unsigned long long t_end = now_ns();
    const unsigned long long ts = p.st->t_start;
    const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
    atomicAdd(&p.st->n_timed, 1ull);
  }
  if (threadIdx.x == 0 && atomicAdd(&p.st->depart, 1u) == gridDim.x - 1) {
    p.st->depart = 0;
    __threadfence();
    p.st->seq = seq;
  }
}

// ---- k1/3-rdma-loads (2026-10-24): parallel-load variant of ar_kernel, world fixed at kW --------
// ar_kernel above is left byte-for-byte unchanged (it is the legacy path, RDMA_AR_PARLOAD=0).
// The legacy consumer loops issue one uncached sysmem load per peer and consume it before the
// next load issues (SASS: LDG.E.128.STRONG.SYS -> 8 FADD -> LDG ...), so each pass pays ~7
// serial GPU<->C2C round trips. Here every thread first loads all kW peer vectors into distinct
// registers, then accumulates them in the SAME r = 0..kW-1 fp32 order and rounds once, so results
// are bitwise identical to ar_kernel. The phase-2 copy and the gather issue all loads before any
// store; the send-slot staging copy is batched x4. Protocol (flags, seq, slots, publish) unchanged.
// __launch_bounds__(kThreads, 1): with the default (no min-blocks) ptxas caps the kernel at 64 regs to
// keep 2 CTAs/SM and re-uses load registers, so only ~4-5 peer loads are in flight before the first
// FADD (two round trips). With min 1 CTA/SM it keeps all 8 in flight (SASS gate: depth 8). The grid
// is at most 16 CTAs on 48 SMs, so one CTA per SM costs nothing.
template <int kMode, bool kF32, int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_pl(Params p) {
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  if (threadIdx.x == 0) s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) p.st->t_start = now_ns();
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;

  // 1) stage the input in this rank's send slot (x4 batched: 4 loads in flight, then 4 stores)
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
  {
    long long i = tid;
    for (; i + 3 * stride < p.nvec; i += 4 * stride) {
      const uint4 a0 = p.in[i];
      const uint4 a1 = p.in[i + stride];
      const uint4 a2 = p.in[i + 2 * stride];
      const uint4 a3 = p.in[i + 3 * stride];
      send4[i] = a0;
      send4[i + stride] = a1;
      send4[i + 2 * stride] = a2;
      send4[i + 3 * stride] = a3;
    }
    for (; i < p.nvec; i += stride) send4[i] = p.in[i];
  }
  if (last_cta(&p.st->arrive, &s_last)) {
    // g130 V1: every op publishes phase 1 exactly once and st->seq counts ops, so what was last published here is
    // seq - 1 and the send_seq load adds nothing. The guard itself must stay: this rank's kernel can finish while its own
    // proxy is stalled (its flags come from the peers' proxies), and the descriptor is a single mailbox.
    if constexpr ((kPub & kPubV1) != 0)
      G130_SPIN(&p.ctrl->posted_seq, seq - 1, -1, 0);
    else
      G130_SPIN(&p.ctrl->posted_seq, ld_acquire_sys(&p.ctrl->send_seq), -1, 0);
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kMode;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = (uint64_t)p.chunk;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
    }
    st_release_sys(&p.ctrl->send_seq, seq);
  }

  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    G130_SPIN(p.flags + (slot * p.world + threadIdx.x) * kFlagStride, seq, threadIdx.x, 1);
  __syncthreads();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r) p.out[(long long)r * p.nvec + i] = v[r];
    }
  } else if constexpr (kMode == kGatherCols) {
    // g2 colltrim: as kGather above (all kW loads in flight before any store), stores to the column slot
    const unsigned useg = (unsigned)p.seg;
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      const long long row = (long long)((unsigned)i / useg);
      uint4* dst = p.out + row * ((long long)kW * p.seg) + (i - row * p.seg);
#pragma unroll
      for (int r = 0; r < kW; ++r) dst[(long long)r * p.seg] = v[r];
    }
  } else if constexpr (kMode == kOneShot) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
      p.out[i] = packv<kF32>(acc);
    }
  } else {
    // two-shot: peers wrote chunk `rank` of their input at offset 0 of recv[slot][peer]
    const long long b = (long long)p.rank * p.chunk;
    const long long len = max(0LL, min(p.chunk, p.nvec - b));
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    for (long long i = tid; i < len; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[b + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
      const uint4 o = packv<kF32>(acc);
      p.out[b + i] = o;
      send2[i] = o;
    }
    if (last_cta(&p.st->arrive2, &s_last)) {
      // g130 V2: no guard is needed. This thread is past the phase-1 wait of seq, so all 7 peers published phase 1 of seq,
      // so each of them finished the previous two-shot op seq' < seq on its single stream, so each of them saw this
      // rank's phase-2 flag of seq': this rank's proxy has already read send2_bytes and posted seq' to all 7 peers. (seq - 1
      // would be WRONG here: send2_seq only moves on two-shot ops, modes interleave, and the spin would never end.)
      if constexpr ((kPub & kPubV2) == 0)
        G130_SPIN(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), -2, 0);
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
      st_release_sys(&p.ctrl->send2_seq, seq);
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      G130_SPIN(p.flags2 + (slot * p.world + threadIdx.x) * kFlagStride, seq, threadIdx.x, 2);
    __syncthreads();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    // legacy: for r != rank, for i < lr(r) = clamp(nvec - r*chunk, 0, chunk): out[r*chunk+i] = recv2[r][i].
    // Same element set (i < chunk and r*chunk + i < nvec), all kW-1 loads issued before the stores.
    for (long long i = tid; i < p.chunk; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec)
          v[r] = __ldcv(reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec) p.out[(long long)r * p.chunk + i] = v[r];
    }
  }

  // 3) last CTA out advances the sequence for the next call
  __syncthreads();
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence();
    const unsigned long long t_end = now_ns();
    const unsigned long long ts = p.st->t_start;
    const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
    atomicAdd(&p.st->n_timed, 1ull);
  }
  if (threadIdx.x == 0 && atomicAdd(&p.st->depart, 1u) == gridDim.x - 1) {
    p.st->depart = 0;
    __threadfence();
    p.st->seq = seq;
  }
}

// ---- r3-comm (climb round 3, 2026-11-08): G06 critical-path trim + G26 send-buffer input ---------------------------
// A SEPARATE kernel: ar_kernel_pl above is byte-for-byte unchanged and stays the kernel of every launch unless
// DSV41_RDMA_TRIM=1. ar_kernel_r3 has the same Params, Ctrl, slots, flags, wire protocol and consumer loops (section 2
// is copied VERBATIM from ar_kernel_pl: tests/test_source.py diffs it), so every output element sees the same loads,
// the same fp32 FADD sequence r = 0..7 from +0.0 and the same single bf16 RN rounding. Only staging, publish and seq
// bookkeeping differ. It sits on the production g130 mask 7 (V1 + V2 + thin poll; V2 dropped outside two-shot).
//   (ii)  thread 0 reads posted_seq (relaxed, sys) at entry; the publisher re-spins only when that early value is below
//         seq - 1. posted_seq is monotonic, so a passing early value is a valid proof that the proxy has consumed
//         message seq - 1; the fence.sc.sys every publish path executes between that read and the descriptor stores
//         makes read + fence an acquire pattern (descriptor stores cannot pass it).
//   (iv)  st->seq advances at ENTRY: thread 0 of every CTA reads it before the entry barrier; then the last thread of
//         each CTA does atom.acq_rel.gpu on st->entered and the last CTA through writes seq (and resets entered).
//         No tail atomic, tail fence or tail barrier (a barrier only when timing). The next kernel on the stream reads
//         seq after this one has completed, exactly as with the tail update.
//   (i)   kR3S1: a message of <= DSV41_RDMA_TRIM_S1_BYTES is staged by CTA 0 alone (4 loads in flight per thread,
//         then the stores), fenced at sys scope per thread, and published by thread 0 after one bar.sync: no grid
//         atomic, no second fence, no second barrier. Other CTAs do not stage; they wait for the flags as before
//         (their local term is read from p.in, as in ar_kernel_pl).
//   (iii) kR3P2: two-shot with chunk <= blockDim: every phase-2 element (i < len <= chunk) belongs to CTA 0, so CTA 0
//         publishes phase 2 alone (fence per thread + bar.sync, as (i)), and CTAs >= 1, which have no consumer work in
//         either phase, leave right after the phase-1 publish. The host picks P2 only when chunk <= 512; the kernel
//         traps otherwise.
//   G26   kR3Sbuf: p.in IS a registered send buffer (RdmaAR::send_buffer). No staging: CTA 0 thread 0 fences and
//         publishes with mode | kModeSrc and ctrl.src_off = p.in - arena. Reuse rule: a buffer used by op s0 may be
//         rewritten only after a LATER op s' (s0 < s' < s1) has passed its phase-1 wait on this rank (then every peer
//         finished s' - 1 >= s0, so it has our s0 payload and our NIC is done reading it). The 16 bytes before each
//         buffer hold the seq of its last use; the kernel traps when s1 - s0 < 2 (fail loud, never silent).
__device__ __forceinline__ uint64_t s8_read_early(const Params& p) {
  uint64_t e = ld_relaxed_sys(&p.ctrl->posted_seq);
  if (p.level_db && p.multi_proxy) {
    const uint64_t eh = ld_relaxed_sys(&p.ctrl_aux->posted_seq_helper);
    if (eh < e) e = eh;
  }
  return e;
}

template <int kPub>
__device__ __forceinline__ void s8_guard_slot(const Params& p, unsigned long long seq, uint64_t early) {
  if (p.level_db) {
    if (seq > (unsigned long long)kSlots && early < seq - (unsigned long long)kSlots) {
      const uint64_t need = seq - (unsigned long long)kSlots;
      G130_SPIN(&p.ctrl->posted_seq, need, -1, 0);
      if (p.multi_proxy) G130_SPIN(&p.ctrl_aux->posted_seq_helper, need, -3, 0);
    }
  } else {
    if (early < seq - 1) G130_SPIN(&p.ctrl->posted_seq, seq - 1, -1, 0);
  }
}

template <int kMode, int kPub>
__device__ __forceinline__ void r3_publish1_bytes(const Params& p, unsigned long long seq, uint64_t early,
                                                  uint64_t src_off, uint64_t send_bytes) {
  s8_guard_slot<kPub>(p, seq, early);
  const uint64_t mode_val = ((kPub & kR3Sbuf) != 0 && src_off != 0) ? ((uint64_t)kMode | kModeSrc) : (uint64_t)kMode;
  if (p.level_db) {
    CtrlSlot* cs = p.ctrl_slots + (seq % kSlots);
    *reinterpret_cast<volatile uint64_t*>(&cs->send_bytes) = send_bytes;
    *reinterpret_cast<volatile uint64_t*>(&cs->src_off) = src_off;
    *reinterpret_cast<volatile uint64_t*>(&cs->mode) = mode_val;
    *reinterpret_cast<volatile uint64_t*>(&cs->chunk_vecs) = (uint64_t)p.chunk;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&cs->pub_time) = tp;
    }
    st_release_sys(&cs->send_seq, seq);
    return;
  }
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = send_bytes;
  if constexpr ((kPub & kR3Sbuf) != 0) {
    if (src_off != 0) *reinterpret_cast<volatile uint64_t*>(&p.ctrl->src_off) = src_off;
  }
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = mode_val;
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = (uint64_t)p.chunk;
  if (p.timing) {
    const unsigned long long tp = now_ns();
    p.st->t_pub = tp;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
  }
  st_release_sys(&p.ctrl->send_seq, seq);
}

template <int kMode, int kPub>
__device__ __forceinline__ void r3_publish1(const Params& p, unsigned long long seq, uint64_t early, uint64_t src_off) {
  r3_publish1_bytes<kMode, kPub>(p, seq, early, src_off, (uint64_t)p.nvec * 16ull);
}

__device__ __forceinline__ void s8_publish2(const Params& p, int slot, unsigned long long seq, uint64_t send2_bytes) {
  if (p.level_db) {
    CtrlSlot* cs = p.ctrl_slots + slot;
    *reinterpret_cast<volatile uint64_t*>(&cs->send2_bytes) = send2_bytes;
    st_release_sys(&cs->send2_seq, seq);
  } else {
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = send2_bytes;
    st_release_sys(&p.ctrl->send2_seq, seq);
  }
}

template <int kPub>
__device__ __forceinline__ void s8_publish1_inline(const Params& p, int slot, unsigned long long seq, uint64_t early,
                                                   uint64_t lo, uint64_t hi) {
  if (p.level_db) {
    if (seq > (unsigned long long)kSlots && early < seq - (unsigned long long)kSlots) {
      const uint64_t need = seq - (unsigned long long)kSlots;
      G130_SPIN(&p.ctrl->posted_seq, need, -1, 0);
      if (p.multi_proxy) G130_SPIN(&p.ctrl_aux->posted_seq_helper, need, -3, 0);
    } else {
      __threadfence_system();
    }
    CtrlSlot* cs = p.ctrl_slots + slot;
    *reinterpret_cast<volatile uint64_t*>(&cs->send_bytes) = 16ull;
    *reinterpret_cast<volatile uint64_t*>(&cs->mode) = (uint64_t)kGather | kModeInline;
    *reinterpret_cast<volatile uint64_t*>(&cs->chunk_vecs) = 1ull;
    *reinterpret_cast<volatile uint64_t*>(&cs->inline_lo) = lo;
    *reinterpret_cast<volatile uint64_t*>(&cs->inline_hi) = hi;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_start = tp;
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&cs->pub_time) = tp;
    }
    st_release_sys(&cs->send_seq, seq);
    return;
  }
  if (early < seq - 1) G130_SPIN(&p.ctrl->posted_seq, seq - 1, -1, 0);
  else __threadfence_system();
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = 16ull;
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kGather | kModeInline;
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = 1ull;
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->inline_lo) = lo;
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->inline_hi) = hi;
  if (p.timing) {
    const unsigned long long tp = now_ns();
    p.st->t_start = tp;
    p.st->t_pub = tp;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
  }
  st_release_sys(&p.ctrl->send_seq, seq);
}

// r3-comm: the r3 bits of one launch, on top of the g130 mask (host side; the CPU emulator calls this same function)
inline int r3_bits(int mode, long long nvec, long long chunk, long long s1_bytes, bool sbuf, long long threads) {
  int b = kR3Trim;
  if (sbuf)
    b |= kR3Sbuf;
  else if (nvec * 16 <= s1_bytes)
    b |= kR3S1;
  if (mode == kTwoShot && chunk <= threads) b |= kR3P2;
  return b;
}

// r3-comm (review N5): S1 (CTA 0 stages for every CTA, so CTAs >= 1 may write p.out before p.in is staged) and Sbuf (the
// NIC reads p.in until every peer has it) are not in-place safe. True when [in, in + nvec*16) overlaps the output and the
// overlap is not the one harmless alias, a dim-0 gather whose input IS its own output segment (rewritten with identical
// bytes). The caller then drops S1/Sbuf and uses the staged r3 path, whose staging thread == consuming thread per element,
// as in ar_kernel_pl. Serving never aliases today (all_reduce: out = empty_like(x); gathers: out = torch.empty).
inline bool r3_inplace_conflict(int mode, unsigned long long in, unsigned long long out, long long nvec, int rank,
                                int world) {
  const unsigned long long nb = (unsigned long long)nvec * 16ull;
  const unsigned long long ob = (mode == kGather || mode == kGatherCols) ? nb * (unsigned long long)world : nb;
  if (nb == 0 || in >= out + ob || out >= in + nb) return false;
  return !(mode == kGather && in == out + (unsigned long long)rank * nb);
}

// s7-v3: dedicated 1-warp (32-thread, 0-smem, 0-atomic) 16-byte inline all-gather kernel (Markov inline AG!)
template <int kW, int kPub>
__global__ void __launch_bounds__(32, 1) ar_kernel_inline_ag(Params p) {
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (p.skip_ptr != nullptr) {
    int skip_v = (threadIdx.x == 0) ? *reinterpret_cast<const volatile int*>(p.skip_ptr) : 0;
    if (__shfl_sync(0xffffffffu, skip_v, 0) != 0) {
      if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      return;
    }
  }
  unsigned long long seq = 0;
  if (threadIdx.x == 0) {
    seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
    if constexpr ((kPub & kR6Pdl) == 0) early = s8_read_early(p);
    const uint4 v_self = p.in[0];
    s8_publish1_inline<kPub>(p, static_cast<int>(seq % kSlots), seq, early,
                             ((uint64_t)v_self.y << 32) | (uint64_t)v_self.x,
                             ((uint64_t)v_self.w << 32) | (uint64_t)v_self.z);
    p.out[p.rank] = v_self;
  }
  seq = __shfl_sync(0xffffffffu, seq, 0);
  const int slot = static_cast<int>(seq % kSlots);
  const InlineSlot* sl = nullptr;
  if (threadIdx.x < kW && threadIdx.x != p.rank) {
    sl = p.inline_recv + slot * kW + threadIdx.x;
    G130_SPIN(&sl->flag, seq, threadIdx.x, 1);
  }
  __syncwarp();
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
  if (sl != nullptr) p.out[threadIdx.x] = __ldcv(&sl->data);
}

// s7-v3: dedicated 1-warp (32-thread, 0-smem, 0-atomic) 32..128B small all-gather kernel
// (covers THR_U 32B/64B Global-THR broadcast + 128B sync-free argmax OUT_PAIR AG!)
// PTX MCM proof:
//   - Publish: lanes 0..nvec-1 stage p.in -> send4 -> __threadfence_system() -> __syncwarp() -> lane 0 st_release_sys(send_seq)
//     (exact same cumulativity as ar_kernel_r3 S1 path).
//   - Consume: peer lane r polls flags[slot][r] via G130_SPIN (ld_acquire_sys) and loads recv[slot][r][0..nvec-1]
//     in the SAME thread via __ldcv (same-thread po after ld.acquire.sys -> 100% PTX-sound, zero fence_acquire_sys!).
template <int kW, int kPub>
__global__ void __launch_bounds__(32, 1) ar_kernel_small_gather(Params p) {
  static_assert(kW <= 32, "ar_kernel_small_gather requires kW <= 32");
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (p.skip_ptr != nullptr) {
    int skip_v = (threadIdx.x == 0) ? *reinterpret_cast<const volatile int*>(p.skip_ptr) : 0;
    if (__shfl_sync(0xffffffffu, skip_v, 0) != 0) {
      if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      return;
    }
  }
  unsigned long long seq = 0;
  if (threadIdx.x == 0) {
    seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
    if constexpr ((kPub & kR6Pdl) == 0) early = s8_read_early(p);
    if (p.timing) p.st->t_start = now_ns();
  }
  seq = __shfl_sync(0xffffffffu, seq, 0);
  const int slot = static_cast<int>(seq % kSlots);
  const int nvec = (int)p.nvec;
  uint4* send4 = reinterpret_cast<uint4*>(p.send + (long long)slot * p.max_bytes);
  uint4 v_self = make_uint4(0u, 0u, 0u, 0u);
  if ((int)threadIdx.x < nvec) {
    v_self = p.in[threadIdx.x];
    send4[threadIdx.x] = v_self;
  }
  __threadfence_system();
  __syncwarp();
  if (threadIdx.x == 0) {
    Params pg = p;
    pg.nvec = nvec;
    pg.chunk = nvec;
    r3_publish1<kGather, kPub & ~kR3Sbuf>(pg, seq, early, 0);
  }
  // Write self slice into p.out while peers are in flight (in-place safe: p.in already read before __syncwarp)
  if ((int)threadIdx.x < nvec) {
    p.out[(long long)p.rank * nvec + threadIdx.x] = v_self;
  }
  uint4 tmp[8];
  const int r = (int)threadIdx.x;
  const bool is_peer = (r < kW && r != p.rank);
  if (is_peer) {
    G130_SPIN(p.flags + (slot * kW + r) * kFlagStride, seq, r, 1);
    const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
    const uint4* src_r = reinterpret_cast<const uint4*>(p.recv + ((long long)slot * kW + r) * rstride);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < nvec) tmp[i] = __ldcv(src_r + i);
    }
  }
  __syncwarp();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0) t_flags = now_ns();
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
  if (is_peer) {
    uint4* dst_r = p.out + (long long)r * nvec;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < nvec) dst_r[i] = tmp[i];
    }
  }
  if (p.timing) {
    __syncwarp();
    if (threadIdx.x == 0) {
      __threadfence();
      const unsigned long long t_end = now_ns();
      const unsigned long long ts = p.st->t_start;
      const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
      atomicAdd(&p.st->n_timed, 1ull);
    }
  }
}

template <int kMode, bool kF32, int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_r3(Params p, R7Proj q) {
  static_assert((kPub & kR3Trim) != 0 && (kPub & kPubV1) != 0 && (kPub & kPollThin) != 0, "r3 sits on g130 mask 7");
  static_assert(kMode != kTwoShot || (kPub & kPubV2) != 0, "r3 two-shot needs V2 (no phase-2 guard)");
  static_assert((kPub & kR3P2) == 0 || kMode == kTwoShot, "P2 is two-shot only");
  static_assert((kPub & kR3S1) == 0 || (kPub & kR3Sbuf) == 0, "S1 and Sbuf exclude each other");
  constexpr bool kOne = (kPub & (kR3S1 | kR3Sbuf)) != 0;  // CTA 0 alone publishes phase 1
  static_assert((kPub & kR6Trig) == 0 || (kPub & kR6Pdl) != 0, "r6: trig needs the PDL launch");
  static_assert((kPub & kR7Proj) == 0 || (!kF32 && (kMode == kOneShot || kMode == kTwoShot) && (kPub & kR3Sbuf) == 0),
                "r7: the projection tail is bf16 one-shot / two-shot only, never with a send buffer");
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  __shared__ int s_rows;  // w3 rowar: live row count (device-read), only meaningful when p.rows_ptr != nullptr
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    // r6 C2: the only access before the grid-dependency wait is this host-written, monotonic word (a stale value only
    // makes r3_publish1 spin, exactly as an early value below seq - 1 does in r3 (ii)). st->seq, st->entered, timing,
    // p.in, the slots, the flags and p.out are all touched after the wait, i.e. after the predecessor grid completed.
    if (threadIdx.x == 0 && (!kOne || blockIdx.x == 0)) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if constexpr (kMode == kGather) {
    if (p.skip_ptr != nullptr) {
      __shared__ int s_skip;
      if (threadIdx.x == 0) s_skip = *reinterpret_cast<const volatile int*>(p.skip_ptr);
      __syncthreads();
      if (s_skip != 0) {
        if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
        return;
      }
    }
  }
  if (threadIdx.x == 0) {
    s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    if (p.rows_ptr != nullptr) s_rows = *reinterpret_cast<const volatile int*>(p.rows_ptr);  // w3 rowar
    if constexpr ((kPub & kR6Pdl) == 0)
      if (!kOne || blockIdx.x == 0) early = s8_read_early(p);  // (ii)
    if constexpr ((kPub & kR3P2) != 0) {
      if (p.chunk > (long long)blockDim.x) {
        printf("rdma_ar r3 rank %d: P2 kernel with chunk %lld > %d threads\n", p.rank, p.chunk, (int)blockDim.x);
        r3_trap();
      }
    }
  }
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) p.st->t_start = now_ns();
  __syncthreads();
  const unsigned long long seq = s_seq;
  if (p.rows_ptr != nullptr) {
    // w3 rowar: every rank reads the same n (replicated state); out-of-range values fall back to the full tensor.
    // The host never picks S1/P2/Sbuf/R7 for a rows launch (those size decisions are static), so only p.nvec/p.chunk move.
    const long long full_rows = p.nvec_full / p.row_vec;
    const long long n = (s_rows >= 1 && s_rows <= full_rows) ? (long long)s_rows : full_rows;
    p.nvec = n * p.row_vec;
    p.chunk = (p.nvec + p.world - 1) / p.world;
  }
  // (iv) this CTA's thread 0 read st->seq before the barrier above; the acq_rel atomic orders that read before this
  // CTA's arrival, so the last CTA through writes seq only after every CTA has read the old value.
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;

  // s6 fast path: 16-byte kGather inline-in-Ctrl + single 24B inline WR (Markov inline AG + sync-free argmax AG)
  if constexpr (kMode == kGather && (kPub & kR3Sbuf) == 0) {
    if (p.inline_ag != 0 && p.nvec == 1 && p.inline_recv != nullptr && gridDim.x == 1) {
      if (threadIdx.x == 0) {
        const uint4 v_self = p.in[0];
        s8_publish1_inline<kPub>(p, slot, seq, early,
                                 ((uint64_t)v_self.y << 32) | (uint64_t)v_self.x,
                                 ((uint64_t)v_self.w << 32) | (uint64_t)v_self.z);
        p.out[p.rank] = v_self;
      }
      if (threadIdx.x < p.world && threadIdx.x != p.rank) {
        const InlineSlot* sl = p.inline_recv + slot * p.world + threadIdx.x;
        G130_SPIN(&sl->flag, seq, threadIdx.x, 1);
        p.out[threadIdx.x] = __ldcv(&sl->data);
      }
      __syncthreads();
      if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      return;
    }
  }

  // 1) phase-1 publish
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
  if constexpr ((kPub & kR3Sbuf) != 0) {
    if (blockIdx.x == 0) {
      const unsigned hthr = blockDim.x > 32 ? 32u : blockDim.x - 1;  // off the publishing thread
      if (threadIdx.x == hthr) {
        uint64_t* hdr = reinterpret_cast<uint64_t*>(const_cast<uint4*>(p.in) - 1);
        const uint64_t last = ld_relaxed_sys(hdr);
        if (last != 0 && seq - last < 2) {
          printf("rdma_ar r3 rank %d: send buffer reused by the next op (seq %llu after %llu)\n", p.rank,
                 (unsigned long long)seq, (unsigned long long)last);
          r3_trap();
        }
        st_relaxed_sys(hdr, seq);
      }
      if (threadIdx.x == 0) {
        __threadfence_system();
        r3_publish1<kMode, kPub>(p, seq, early,
                                 (uint64_t)(reinterpret_cast<const char*>(p.in) - reinterpret_cast<const char*>(p.ctrl)));
      }
    }
  } else if constexpr ((kPub & kR3S1) != 0) {
    if (blockIdx.x == 0) {
      const long long nb = blockDim.x;
      for (long long i0 = threadIdx.x; i0 < p.nvec; i0 += 4 * nb) {
        const bool b1 = i0 + nb < p.nvec, b2 = i0 + 2 * nb < p.nvec, b3 = i0 + 3 * nb < p.nvec;
        uint4 a0, a1, a2, a3;
        a0 = p.in[i0];
        if (b1) a1 = p.in[i0 + nb];
        if (b2) a2 = p.in[i0 + 2 * nb];
        if (b3) a3 = p.in[i0 + 3 * nb];
        send4[i0] = a0;
        if (b1) send4[i0 + nb] = a1;
        if (b2) send4[i0 + 2 * nb] = a2;
        if (b3) send4[i0 + 3 * nb] = a3;
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) r3_publish1<kMode, kPub>(p, seq, early, 0);
    }
  } else {
    {
      long long i = tid;
      for (; i + 3 * stride < p.nvec; i += 4 * stride) {
        const uint4 a0 = p.in[i];
        const uint4 a1 = p.in[i + stride];
        const uint4 a2 = p.in[i + 2 * stride];
        const uint4 a3 = p.in[i + 3 * stride];
        send4[i] = a0;
        send4[i + stride] = a1;
        send4[i + 2 * stride] = a2;
        send4[i + 3 * stride] = a3;
      }
      for (; i < p.nvec; i += stride) send4[i] = p.in[i];
    }
    if (last_cta(&p.st->arrive, &s_last)) r3_publish1<kMode, kPub>(p, seq, early, 0);
  }
  if constexpr ((kPub & kR3P2) != 0) {
    if (blockIdx.x != 0) return;  // (iii) no consumer work here in either phase (len <= chunk <= blockDim)
  }

  // w3 rowar: the output rows past the live rows are zeroed (finite, deterministic: they are masked rows that no live row
  // reads, but a NaN there could still poison a tile-wide p@v product). Done while waiting for the peers.
  if (p.rows_ptr != nullptr) {
    for (long long i = p.nvec + tid; i < p.nvec_full; i += stride) p.out[i] = make_uint4(0u, 0u, 0u, 0u);
  }

  // s6 split phase-2 for kTwoShot when chunk <= 1024 (e.g. drafter [8,4608] chunk=576):
  // CTA 0 alone polls p1 flags, reduces all `len` vecs, and publishes send2_seq; CTAs 1..N skip to p2 wait + all-CTA copy!
  const bool split_p2 = (kMode == kTwoShot) && ((kPub & kR3P2) == 0) && (p.rows_p2 >= 2) && (p.chunk <= 1024);
  const long long b_p2 = (long long)p.rank * p.chunk;
  const long long len_p2 = max(0LL, min(p.chunk, p.nvec - b_p2));
  const long long p1_expected_bytes = (kMode == kTwoShot) ? (len_p2 * 16) : (p.nvec * 16);

  // 2) wait for every peer's phase-1 contribution
  if (!split_p2 || blockIdx.x == 0) {
    s6_wait_p1(p, slot, seq, p1_expected_bytes, !split_p2 && ((kPub & kR3P2) == 0));
  }
  if constexpr (kMode != kTwoShot) {
    if (blockIdx.x == 0 && threadIdx.x == 0) st_relaxed_gpu(&p.st->ar_done_cnt, seq);
  }
  if constexpr ((kPub & kR6Trig) != 0 && kMode != kTwoShot) r6_gdc_launch();  // r6 C2: the last wait is behind us
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const char* rbase = p.recv + (long long)slot * p.world * rstride;
  if constexpr (kMode == kGather) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r) p.out[(long long)r * p.nvec + i] = v[r];
    }
  } else if constexpr (kMode == kGatherCols) {
    // g2 colltrim: as kGather above (all kW loads in flight before any store), stores to the column slot
    const unsigned useg = (unsigned)p.seg;
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
      const long long row = (long long)((unsigned)i / useg);
      uint4* dst = p.out + row * ((long long)kW * p.seg) + (i - row * p.seg);
#pragma unroll
      for (int r = 0; r < kW; ++r) dst[(long long)r * p.seg] = v[r];
    }
  } else if constexpr (kMode == kOneShot) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
      p.out[i] = packv<kF32>(acc);
    }
  } else {
    // two-shot: peers wrote chunk `rank` of their input at offset 0 of recv[slot][peer]
    const long long b = b_p2;
    const long long len = len_p2;
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    if (split_p2) {
      if (blockIdx.x == 0) {
        for (long long i = threadIdx.x; i < len; i += blockDim.x) {
          uint4 v[kW];
#pragma unroll
          for (int r = 0; r < kW; ++r)
            v[r] = (r == p.rank) ? p.in[b + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
          float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
          const uint4 o = packv<kF32>(acc);
          p.out[b + i] = o;
          send2[i] = o;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) s8_publish2(p, slot, seq, (uint64_t)len * 16);
      }
    } else {
      for (long long i = tid; i < len; i += stride) {
        uint4 v[kW];
#pragma unroll
        for (int r = 0; r < kW; ++r)
          v[r] = (r == p.rank) ? p.in[b + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
        const uint4 o = packv<kF32>(acc);
        p.out[b + i] = o;
        send2[i] = o;
      }
      // phase-2 publish; g130 V2 (no guard spin) holds unchanged: see ar_kernel_pl
      if constexpr ((kPub & kR3P2) != 0) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) s8_publish2(p, slot, seq, (uint64_t)len * 16);
      } else {
        if (last_cta(&p.st->arrive2, &s_last)) s8_publish2(p, slot, seq, (uint64_t)len * 16);
      }
    }
    s6_wait_p2(p, slot, seq, (kPub & kR3P2) == 0);
    if (blockIdx.x == 0 && threadIdx.x == 0) st_relaxed_gpu(&p.st->ar_done_cnt, seq);
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();  // r6 C2: the last wait is behind us
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    // legacy: for r != rank, for i < lr(r) = clamp(nvec - r*chunk, 0, chunk): out[r*chunk+i] = recv2[r][i].
    // Same element set (i < chunk and r*chunk + i < nvec), all kW-1 loads issued before the stores.
    for (long long i = tid; i < p.chunk; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec)
          v[r] = __ldcv(reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec) p.out[(long long)r * p.chunk + i] = v[r];
    }
  }

  // r7-comm2 HT13: projection tail. Under P2 only CTA 0 is here and it wrote all of p.out itself (block-scope
  // visibility after the barrier); otherwise the last CTA through arrive3 (acq_rel, after every CTA's fence) projects.
  if constexpr ((kPub & kR7Proj) != 0) {
    if constexpr (kMode == kTwoShot && (kPub & kR3P2) != 0) {
      __syncthreads();
      r7_project(p.out, q);
    } else {
      __shared__ int s_go;
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) {
        const bool last = atom_add_acqrel_gpu(&p.st->arrive3, 1u) == gridDim.x - 1;
        if (last) *reinterpret_cast<volatile unsigned int*>(&p.st->arrive3) = 0u;
        s_go = last ? 1 : 0;
      }
      __syncthreads();
      if (s_go) {
        __threadfence();
        r7_project(p.out, q);
      }
    }
  }

  // 3) seq already advanced at entry (iv); the tail barrier exists only for the timing block (unchanged)
  if (p.timing) {
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      __threadfence();
      const unsigned long long t_end = now_ns();
      const unsigned long long ts = p.st->t_start;
      const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
      atomicAdd(&p.st->n_timed, 1ull);
    }
  }
}
// =================================================================================================
// s8 Pillar 1: 0-ULP Lossless BF12-MAX14+Z Wire Compression Codec (SM121a)
// =================================================================================================
// Encodes N_vec uint4 (8*N_vec bf16) elements into:
//   [Bf12Hdr (16B) | sm8 (8*N_vec B) | code4 (4*N_vec B) | esc32 (4*n_esc B, pad16)]
// Code table per BF16 element (exp = (u >> 7) & 0xFF, sm = ((u >> 8) & 0x80) | (u & 0x7F)):
//   0..13 : exp = max_e - code  (14 contiguous top exponents -> 99.3-100.0% of Inkling-2 activations!)
//   14    : exp = 0             (exact +0.0, -0.0, and ALL subnormals with 0 escapes!)
//   15    : escape              (uint32_t (elem_idx << 8) | exp appended to esc32 tail)
// Automatic raw fallback (hdr.n_esc == kBf12RawFlag) whenever n_esc > max_esc (guaranteed <= raw + 16B).
// 100% 0-ULP lossless across all 65,536 BF16 bit patterns (including -0.0, subnormals, +-Inf, all NaNs).

__device__ __forceinline__ uint32_t bf12_u4_max_exp(uint4 v) {
  uint32_t m = 0u;
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t e0 = (w[i] >> 7) & 0xFFu;
    uint32_t e1 = (w[i] >> 23) & 0xFFu;
    m = max(m, max(e0, e1));
  }
  return m;
}

__device__ __forceinline__ uint32_t bf12_block_max_u32(uint32_t v, uint32_t* smem_w) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  if (lane == 0) smem_w[warp] = v;
  __syncthreads();
  uint32_t out = 0u;
  if (warp == 0) {
    out = (lane < nwarps) ? smem_w[lane] : 0u;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) out = max(out, __shfl_xor_sync(0xFFFFFFFFu, out, off));
    if (lane == 0) smem_w[0] = out;
  }
  __syncthreads();
  return smem_w[0];
}

__device__ __forceinline__ void bf12_encode_vec(
    uint4 v, uint32_t vec_idx, uint32_t max_e, uint32_t max_esc,
    uint2* sm_out, uint32_t* c4_out, uint32_t* esc_out, unsigned int* s_esc_cnt) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t sm_lo = 0u, sm_hi = 0u, codes = 0u;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint32_t u = (k & 1) ? (w[k >> 1] >> 16) : (w[k >> 1] & 0xFFFFu);
    const uint32_t sm = ((u >> 8) & 0x80u) | (u & 0x7Fu);
    const uint32_t e = (u >> 7) & 0xFFu;
    uint32_t c;
    if (e == 0u) {
      c = 14u;
    } else if (max_e >= e && (max_e - e) <= 13u) {
      c = max_e - e;
    } else {
      c = 15u;
      const uint32_t slot_e = atomicAdd(s_esc_cnt, 1u);
      if (slot_e < max_esc) {
        esc_out[slot_e] = (((vec_idx << 3) | (uint32_t)k) << 8) | e;
      }
    }
    if (k < 4) sm_lo |= (sm << (k * 8));
    else sm_hi |= (sm << ((k - 4) * 8));
    codes |= (c << (k * 4));
  }
  sm_out[vec_idx] = make_uint2(sm_lo, sm_hi);
  c4_out[vec_idx] = codes;
}

__device__ __forceinline__ uint32_t bf12_reconstruct2(uint32_t sm16, uint32_t c8, uint32_t max_e) {
  const uint32_t s0 = sm16 & 0xFFu, c0 = c8 & 0xFu;
  const uint32_t s1 = (sm16 >> 8) & 0xFFu, c1 = (c8 >> 4) & 0xFu;
  const uint32_t e0 = (c0 < 14u) ? (max_e - c0) : 0u;
  const uint32_t e1 = (c1 < 14u) ? (max_e - c1) : 0u;
  const uint32_t u0 = ((s0 & 0x80u) << 8) | (e0 << 7) | (s0 & 0x7Fu);
  const uint32_t u1 = ((s1 & 0x80u) << 8) | (e1 << 7) | (s1 & 0x7Fu);
  return u0 | (u1 << 16);
}

// Pack N_vec vectors held in per-thread registers rv[0..kMaxR-1] (at indices tid + k*blockDim.x) directly to wire buf.
// Returns total 16-byte aligned wire bytes (written to dst_wire[0..15] Bf12Hdr too).
template <int kMaxR>
__device__ __forceinline__ uint32_t bf12_pack_regs_cta(
    const uint4 rv[kMaxR], uint32_t n_vec, char* dst_wire, uint32_t* smem_w, unsigned int* s_esc_cnt) {
  uint32_t local_max = 0u;
#pragma unroll
  for (int k = 0; k < kMaxR; ++k) {
    const uint32_t idx = (uint32_t)threadIdx.x + (uint32_t)k * (uint32_t)blockDim.x;
    if (idx < n_vec) local_max = max(local_max, bf12_u4_max_exp(rv[k]));
  }
  if (threadIdx.x == 0) *s_esc_cnt = 0u;
  const uint32_t max_e = bf12_block_max_u32(local_max, smem_w);
  const uint32_t max_esc = (n_vec > 4u) ? (n_vec - 4u) : 0u;
  uint2* sm_out = reinterpret_cast<uint2*>(dst_wire + 16);
  uint32_t* c4_out = reinterpret_cast<uint32_t*>(dst_wire + 16 + (size_t)n_vec * 8);
  uint32_t* esc_out = reinterpret_cast<uint32_t*>(dst_wire + 16 + (size_t)n_vec * 12);
#pragma unroll
  for (int k = 0; k < kMaxR; ++k) {
    const uint32_t idx = (uint32_t)threadIdx.x + (uint32_t)k * (uint32_t)blockDim.x;
    if (idx < n_vec) bf12_encode_vec(rv[k], idx, max_e, max_esc, sm_out, c4_out, esc_out, s_esc_cnt);
  }
  __syncthreads();
  const uint32_t n_esc = *s_esc_cnt;
  if (n_esc <= max_esc) {
    const uint32_t raw_tail = 16u + n_vec * 12u + n_esc * 4u;
    const uint32_t total = (raw_tail + 15u) & ~15u;
    if (threadIdx.x == 0) {
      Bf12Hdr h{n_vec, max_e, n_esc, total};
      *reinterpret_cast<uint4*>(dst_wire) = *reinterpret_cast<const uint4*>(&h);
    }
    return total;
  }
  // Rare raw fallback (>12.4% escapes): 0-ULP guaranteed!
  uint4* raw_out = reinterpret_cast<uint4*>(dst_wire + 16);
#pragma unroll
  for (int k = 0; k < kMaxR; ++k) {
    const uint32_t idx = (uint32_t)threadIdx.x + (uint32_t)k * (uint32_t)blockDim.x;
    if (idx < n_vec) raw_out[idx] = rv[k];
  }
  const uint32_t total = 16u + n_vec * 16u;
  if (threadIdx.x == 0) {
    Bf12Hdr h{n_vec, 0u, kBf12RawFlag, total};
    *reinterpret_cast<uint4*>(dst_wire) = *reinterpret_cast<const uint4*>(&h);
  }
  return total;
}

// Pack N_vec vectors from global memory src_mem into wire buf (used for BF12-OS4 1-shot staging).
__device__ __forceinline__ uint32_t bf12_pack_mem_cta(
    const uint4* src_mem, uint32_t n_vec, char* dst_wire, uint32_t* smem_w, unsigned int* s_esc_cnt) {
  uint32_t local_max = 0u;
  for (uint32_t idx = (uint32_t)threadIdx.x; idx < n_vec; idx += (uint32_t)blockDim.x) {
    local_max = max(local_max, bf12_u4_max_exp(src_mem[idx]));
  }
  if (threadIdx.x == 0) *s_esc_cnt = 0u;
  const uint32_t max_e = bf12_block_max_u32(local_max, smem_w);
  const uint32_t max_esc = (n_vec > 4u) ? (n_vec - 4u) : 0u;
  uint2* sm_out = reinterpret_cast<uint2*>(dst_wire + 16);
  uint32_t* c4_out = reinterpret_cast<uint32_t*>(dst_wire + 16 + (size_t)n_vec * 8);
  uint32_t* esc_out = reinterpret_cast<uint32_t*>(dst_wire + 16 + (size_t)n_vec * 12);
  for (uint32_t idx = (uint32_t)threadIdx.x; idx < n_vec; idx += (uint32_t)blockDim.x) {
    bf12_encode_vec(src_mem[idx], idx, max_e, max_esc, sm_out, c4_out, esc_out, s_esc_cnt);
  }
  __syncthreads();
  const uint32_t n_esc = *s_esc_cnt;
  if (n_esc <= max_esc) {
    const uint32_t raw_tail = 16u + n_vec * 12u + n_esc * 4u;
    const uint32_t total = (raw_tail + 15u) & ~15u;
    if (threadIdx.x == 0) {
      Bf12Hdr h{n_vec, max_e, n_esc, total};
      *reinterpret_cast<uint4*>(dst_wire) = *reinterpret_cast<const uint4*>(&h);
    }
    return total;
  }
  uint4* raw_out = reinterpret_cast<uint4*>(dst_wire + 16);
  for (uint32_t idx = (uint32_t)threadIdx.x; idx < n_vec; idx += (uint32_t)blockDim.x) raw_out[idx] = src_mem[idx];
  const uint32_t total = 16u + n_vec * 16u;
  if (threadIdx.x == 0) {
    Bf12Hdr h{n_vec, 0u, kBf12RawFlag, total};
    *reinterpret_cast<uint4*>(dst_wire) = *reinterpret_cast<const uint4*>(&h);
  }
  return total;
}

// Unpack BF12-MAX14+Z wire packet from host-pinned src_wire (via __ldcv) into L2-hot dst_bf16 (0-ULP lossless!).
__device__ __forceinline__ void bf12_unpack_to_dst_cta(const char* src_wire, uint4* dst_bf16, Bf12Hdr* s_hdr) {
  if (threadIdx.x == 0) {
    uint4 hv = __ldcv(reinterpret_cast<const uint4*>(src_wire));
    *reinterpret_cast<uint4*>(s_hdr) = hv;
  }
  __syncthreads();
  const uint32_t n_vec = s_hdr->n_vec;
  const uint32_t n_esc = s_hdr->n_esc;
  if (n_esc == kBf12RawFlag) {
    const uint4* raw_src = reinterpret_cast<const uint4*>(src_wire + 16);
    for (uint32_t i = (uint32_t)threadIdx.x; i < n_vec; i += (uint32_t)blockDim.x) dst_bf16[i] = __ldcv(raw_src + i);
    return;
  }
  const uint2* sm_src = reinterpret_cast<const uint2*>(src_wire + 16);
  const uint32_t* c4_src = reinterpret_cast<const uint32_t*>(src_wire + 16 + (size_t)n_vec * 8);
  const uint32_t max_e = s_hdr->max_e;
  for (uint32_t i = (uint32_t)threadIdx.x; i < n_vec; i += (uint32_t)blockDim.x) {
    const uint2 sm = __ldcv(sm_src + i);
    const uint32_t codes = __ldcv(c4_src + i);
    uint4 o;
    o.x = bf12_reconstruct2(sm.x & 0xFFFFu, codes & 0xFFu, max_e);
    o.y = bf12_reconstruct2(sm.x >> 16, (codes >> 8) & 0xFFu, max_e);
    o.z = bf12_reconstruct2(sm.y & 0xFFFFu, (codes >> 16) & 0xFFu, max_e);
    o.w = bf12_reconstruct2(sm.y >> 16, (codes >> 24) & 0xFFu, max_e);
    dst_bf16[i] = o;
  }
  if (n_esc > 0u) {
    __syncthreads();
    const uint32_t* esc_src = reinterpret_cast<const uint32_t*>(src_wire + 16 + (size_t)n_vec * 12);
    uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst_bf16);
    for (uint32_t eidx = (uint32_t)threadIdx.x; eidx < n_esc; eidx += (uint32_t)blockDim.x) {
      const uint32_t entry = __ldcv(esc_src + eidx);
      const uint32_t pos = entry >> 8;
      const uint32_t raw_e = entry & 0xFFu;
      dst16[pos] = static_cast<uint16_t>((dst16[pos] & 0x807Fu) | (raw_e << 7));
    }
  }
}

// =================================================================================================
// s8 Pillar 3: Templated Low-Reg ar_kernel_rows<kW, kPub, kP2Mode, kWireBf12> (kRowsThreads = 576!)
// =================================================================================================
// Fixes 3 root causes found in s7c2 audit:
//   1. Templated by kP2Mode (0=multi, 1=single, 2=split, 3=peer) -> cuts regs 122 -> ~64/thr so
//      ar_kernel_rows co-resides with wo_ud (bf11_flip) during early PDL trigger!
//   2. kRowsThreads = 576 (= exact row_vec for D=4608) -> single-pass M=8 P1 reduce & P2 copy,
//      AND single-CTA 1-shot at n_live == 1 (eliminates CTA-1 L2_RELAY waste!).
//   3. kP2Mode == 3 uses spin_until_thin_sleep(..., p.poll_sleep_ns ? p.poll_sleep_ns : 32u),
//      fixing the 7-SM C2C snoop saturation bug that caused s7_a2 regression!
//   4. Optional kWireBf12 = true: 0-ULP BF12-MAX14+Z compression on Phase-2 payloads (~25% wire cut!).
template <int kW, int kPub, int kP2Mode, bool kWireBf12>
__global__ void __launch_bounds__(kRowsThreads, 1) ar_kernel_rows(Params p) {
  static_assert((kPub & kR3Trim) != 0 && (kPub & kPubV1) != 0 && (kPub & kPubV2) != 0 && (kPub & kPollThin) != 0,
                "ar_kernel_rows sits on r3 mask");
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  __shared__ int s_rows;
  __shared__ uint32_t s_bf12_w[32];
  __shared__ unsigned int s_bf12_esc;
  __shared__ Bf12Hdr s_bf12_hdr;
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if constexpr ((kPub & kR3Sbuf) != 0) {
      if (threadIdx.x == 0 && blockIdx.x == 0) early = s8_read_early(p);
    } else {
      if (threadIdx.x == 0) early = s8_read_early(p);
    }
    r6_gdc_wait();
  }
  const long long full_rows = p.nvec_full / p.row_vec;
  if (threadIdx.x == 0) {
    const unsigned long long seq0 = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    const int r0 = *reinterpret_cast<const volatile int*>(p.rows_ptr);
    s_seq = seq0;
    s_rows = r0;
    if constexpr ((kPub & kR6Pdl) == 0) {
      if constexpr ((kPub & kR3Sbuf) != 0) {
        if (blockIdx.x == 0) early = s8_read_early(p);
      } else {
        early = s8_read_early(p);
      }
    }
    if (p.timing && blockIdx.x == 0) p.st->t_start = now_ns();
    if constexpr ((kPub & kR3Sbuf) != 0) {
      if (blockIdx.x == 0) {
        const long long n0 = (r0 >= 1 && r0 <= full_rows) ? (long long)r0 : full_rows;
        Params p0 = p;
        p0.nvec = n0 * p.row_vec;
        p0.chunk = (p0.nvec + p.world - 1) / p.world;
        const bool os0 = (n0 <= (long long)p.oneshot_max_rows);
        __threadfence_system();
        const uint64_t src_off =
            (uint64_t)(reinterpret_cast<const char*>(p.in) - reinterpret_cast<const char*>(p.ctrl));
        if (os0) r3_publish1<kOneShot, kPub>(p0, seq0, early, src_off);
        else r3_publish1<kTwoShot, kPub>(p0, seq0, early, src_off);
      }
    }
  }
  __syncthreads();
  const unsigned long long seq = s_seq;
  const long long n = (s_rows >= 1 && s_rows <= full_rows) ? (long long)s_rows : full_rows;
  p.nvec = n * p.row_vec;
  p.chunk = (p.nvec + p.world - 1) / p.world;
  const bool use_oneshot = (n <= (long long)p.oneshot_max_rows);
  const bool use_s1 = (p.nvec * 16 <= p.s1_bytes);
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);

  // 1) Phase-1 stage & publish
  if constexpr ((kPub & kR3Sbuf) != 0) {
    if (blockIdx.x == 0 && threadIdx.x == 32) {
      uint64_t* hdr = reinterpret_cast<uint64_t*>(const_cast<uint4*>(p.in) - 1);
      const uint64_t last = ld_relaxed_sys(hdr);
      if (last != 0 && seq - last < 2) {
        printf("rdma_ar s8 rows rank %d: send buffer reused by next op (seq %llu after %llu)\n",
               p.rank, (unsigned long long)seq, (unsigned long long)last);
        r3_trap();
      }
      st_relaxed_sys(hdr, seq);
    }
  } else if (use_s1) {
    if (blockIdx.x == 0) {
      const long long nb = blockDim.x;
      for (long long i0 = threadIdx.x; i0 < p.nvec; i0 += 4 * nb) {
        const bool b1 = i0 + nb < p.nvec, b2 = i0 + 2 * nb < p.nvec, b3 = i0 + 3 * nb < p.nvec;
        uint4 a0 = p.in[i0], a1, a2, a3;
        if (b1) a1 = p.in[i0 + nb];
        if (b2) a2 = p.in[i0 + 2 * nb];
        if (b3) a3 = p.in[i0 + 3 * nb];
        send4[i0] = a0;
        if (b1) send4[i0 + nb] = a1;
        if (b2) send4[i0 + 2 * nb] = a2;
        if (b3) send4[i0 + 3 * nb] = a3;
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        if (use_oneshot) r3_publish1<kOneShot, kPub>(p, seq, early, 0);
        else r3_publish1<kTwoShot, kPub>(p, seq, early, 0);
      }
    }
  } else {
    long long i = tid;
    for (; i + 3 * stride < p.nvec; i += 4 * stride) {
      const uint4 a0 = p.in[i];
      const uint4 a1 = p.in[i + stride];
      const uint4 a2 = p.in[i + 2 * stride];
      const uint4 a3 = p.in[i + 3 * stride];
      send4[i] = a0;
      send4[i + stride] = a1;
      send4[i + 2 * stride] = a2;
      send4[i + 3 * stride] = a3;
    }
    for (; i < p.nvec; i += stride) send4[i] = p.in[i];
    if (last_cta(&p.st->arrive, &s_last)) {
      if (use_oneshot) r3_publish1<kOneShot, kPub>(p, seq, early, 0);
      else r3_publish1<kTwoShot, kPub>(p, seq, early, 0);
    }
  }

  // Zero output rows past n_live while peers are in flight
  for (long long i = p.nvec + tid; i < p.nvec_full; i += stride)
    p.out[i] = make_uint4(0u, 0u, 0u, 0u);

  if (use_oneshot) {
    // At kRowsThreads=576 and n_live=1 (nvec=576), ONLY CTA 0 stays active -> zero L2_RELAY!
    const bool multi_cta_os = (p.nvec > (long long)blockDim.x);
    if (blockIdx.x != 0 && (long long)blockIdx.x * blockDim.x >= p.nvec) return;
    s6_wait_p1(p, slot, seq, p.nvec * 16, multi_cta_os);
    if (blockIdx.x == 0 && threadIdx.x == 0) st_relaxed_gpu(&p.st->ar_done_cnt, seq);
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
    unsigned long long t_flags = 0;
    if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();
    const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
    const char* rbase = p.recv + (long long)slot * p.world * rstride;
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
      p.out[i] = packv<false>(acc);
    }
    if (p.timing) {
      __syncthreads();
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence();
        const unsigned long long t_end = now_ns();
        const unsigned long long ts = p.st->t_start;
        const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
        atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
        atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
        atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
        atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
        atomicAdd(&p.st->n_timed, 1ull);
      }
    }
    return;
  }

  const long long b_p2 = (long long)p.rank * p.chunk;
  const long long len_p2 = max(0LL, min(p.chunk, p.nvec - b_p2));
  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const char* rbase = p.recv + (long long)slot * p.world * rstride;
  char* send2_c = p.send2 + slot * p.max_chunk;
  const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;

  if constexpr (kP2Mode == 3) {
    // Fixed s8 per-peer overlapped P2 with 32ns sleep (0 C2C snoop saturation!) + optional BF12-MAX14+Z!
    if (blockIdx.x >= kW) return;
    if (blockIdx.x == p.rank) {
      s6_wait_p1(p, slot, seq, len_p2 * 16, false);
      uint4 rv[1];
      uint4* send2 = reinterpret_cast<uint4*>(send2_c);
      for (long long i = threadIdx.x; i < len_p2; i += blockDim.x) {
        uint4 v[kW];
#pragma unroll
        for (int r = 0; r < kW; ++r)
          v[r] = (r == p.rank) ? p.in[b_p2 + i]
                               : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
        const uint4 o = packv<false>(acc);
        p.out[b_p2 + i] = o;
        if constexpr (kWireBf12) {
          if (i < (long long)blockDim.x) rv[0] = o;
        } else {
          send2[i] = o;
        }
      }
      uint64_t p2_bytes = (uint64_t)len_p2 * 16ull;
      if constexpr (kWireBf12) {
        if (len_p2 <= (long long)blockDim.x) {
          p2_bytes = bf12_pack_regs_cta<1>(rv, (uint32_t)len_p2, send2_c, s_bf12_w, &s_bf12_esc);
        } else {
          p2_bytes = bf12_pack_mem_cta(p.out + b_p2, (uint32_t)len_p2, send2_c, s_bf12_w, &s_bf12_esc);
        }
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        s8_publish2(p, slot, seq, p2_bytes);
        st_release_gpu(&p.st->p1_done, seq);
      }
      if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      return;
    }
    const int r = blockIdx.x;
    const long long br = (long long)r * p.chunk;
    const long long lr = max(0LL, min(p.chunk, p.nvec - br));
    s7_wait_p1_gate_l2(p, seq);
    if (threadIdx.x == 0) {
      const uint64_t* fp = p.flags2 + (slot * p.world + r) * kFlagStride;
      // s8 fix: 32ns warp-sleep on 7-SM Phase-2 polls cuts C2C snoop traffic 5-10x!
      spin_until_thin_sleep(fp, seq, p, r, 2, p.poll_sleep_ns ? p.poll_sleep_ns : 32u);
      if (atom_add_acqrel_gpu(&p.st->arrive2, 1u) == (unsigned int)(kW - 2)) {
        *reinterpret_cast<volatile unsigned int*>(&p.st->arrive2) = 0u;
        st_relaxed_gpu(&p.st->ar_done_cnt, seq);
      }
    }
    __syncthreads();
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
    if constexpr (kWireBf12) {
      bf12_unpack_to_dst_cta(r2 + (long long)r * p.max_chunk, p.out + br, &s_bf12_hdr);
    } else {
      const uint4* src_r = reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk);
      for (long long i = threadIdx.x; i < lr; i += blockDim.x) p.out[br + i] = __ldcv(src_r + i);
    }
    return;
  }

  if constexpr (kP2Mode == 1) {
    if (blockIdx.x != 0) return;
  }

  // kP2Mode in {0, 1, 2}
  if (kP2Mode != 2 || blockIdx.x == 0) {
    s6_wait_p1(p, slot, seq, len_p2 * 16, (kP2Mode == 0));
  }
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  uint4* send2 = reinterpret_cast<uint4*>(send2_c);
  if constexpr (kP2Mode == 1 || kP2Mode == 2) {
    if (blockIdx.x == 0) {
      uint4 rv[1];
      for (long long i = threadIdx.x; i < len_p2; i += blockDim.x) {
        uint4 v[kW];
#pragma unroll
        for (int r = 0; r < kW; ++r)
          v[r] = (r == p.rank) ? p.in[b_p2 + i]
                               : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
        const uint4 o = packv<false>(acc);
        p.out[b_p2 + i] = o;
        if constexpr (kWireBf12) {
          if (i < (long long)blockDim.x) rv[0] = o;
        } else {
          send2[i] = o;
        }
      }
      uint64_t p2_bytes = (uint64_t)len_p2 * 16ull;
      if constexpr (kWireBf12) {
        if (len_p2 <= (long long)blockDim.x) {
          p2_bytes = bf12_pack_regs_cta<1>(rv, (uint32_t)len_p2, send2_c, s_bf12_w, &s_bf12_esc);
        } else {
          p2_bytes = bf12_pack_mem_cta(p.out + b_p2, (uint32_t)len_p2, send2_c, s_bf12_w, &s_bf12_esc);
        }
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) s8_publish2(p, slot, seq, p2_bytes);
    }
  } else {
    for (long long i = tid; i < len_p2; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[b_p2 + i]
                             : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
      const uint4 o = packv<false>(acc);
      p.out[b_p2 + i] = o;
      send2[i] = o;
    }
    if (last_cta(&p.st->arrive2, &s_last)) s8_publish2(p, slot, seq, (uint64_t)len_p2 * 16ull);
  }
  s6_wait_p2(p, slot, seq, (kP2Mode != 1));
  if (blockIdx.x == 0 && threadIdx.x == 0) st_relaxed_gpu(&p.st->ar_done_cnt, seq);
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();

  if constexpr (kWireBf12 && kP2Mode == 1) {
    for (int r = 0; r < kW; ++r) {
      if (r == p.rank) continue;
      bf12_unpack_to_dst_cta(r2 + (long long)r * p.max_chunk, p.out + (long long)r * p.chunk, &s_bf12_hdr);
      __syncthreads();
    }
  } else if constexpr (kWireBf12 && kP2Mode == 2) {
    for (int r = blockIdx.x; r < kW; r += gridDim.x) {
      if (r == p.rank) continue;
      bf12_unpack_to_dst_cta(r2 + (long long)r * p.max_chunk, p.out + (long long)r * p.chunk, &s_bf12_hdr);
      __syncthreads();
    }
  } else {
    const long long c_tid = (kP2Mode == 1) ? (long long)threadIdx.x : tid;
    const long long c_stride = (kP2Mode == 1) ? (long long)blockDim.x : stride;
    for (long long i = c_tid; i < p.chunk; i += c_stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec)
          v[r] = __ldcv(reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec) p.out[(long long)r * p.chunk + i] = v[r];
    }
  }

  if (p.timing) {
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      __threadfence();
      const unsigned long long t_end = now_ns();
      const unsigned long long ts = p.st->t_start;
      const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
      atomicAdd(&p.st->n_timed, 1ull);
    }
  }
}

// =================================================================================================
// s8 Pillar 2: 0-ULP Bit-Exact Fused RDMA AllReduce + SConv + AttnRes + RMSNorm
// =================================================================================================
// Eliminates 192 kernel launches/step (96 sconv + 96 attn_res) + 96x 73.7 KB DRAM round-trips +
// 100% of Stack/Queries/ConvW/ConvCache/ConvRes/ConvAddend/MixLast preload latency (hidden behind
// the 25-35 us RDMA wait!) while guaranteeing LITERAL 0-ULP bit-exactness against live Triton/ptxas 13.0!
//
// Grid: 1 + nq * T CTAs x 256 threads (9 CTAs for Q=1,T=8; 25 CTAs for Q=3,T=8 -> <= 48 GB10 SMs!).
//   - CTA 0 (RDMA Master): publishes Phase-1 (kR3Sbuf: 0 staging), polls C2C flags on 1 warp,
//     reduces chunk `rank` -> L2-hot p.reduced_buf + send2 (raw or BF12-MAX14+Z), then runs
//     warp-ballot pipelined Phase-2 arrival -> single-__ldcv L2 staging/unpack -> per-peer GPU-L2
//     release p2_peer_seq[r] (zero 12x redundant __ldcv on nq=3 / 4-tap conv!).
//   - Worker CTAs 1..nq*T (one (q_idx, t) per CTA): during RDMA wait, preloads Queries, NormW L2
//     prefetch, ConvW, ConvCache lags 3..1 (SMEM), ConvAddend/ConvRes (+ row_ready[t] aliasing guard)
//     / MixLast, and ALL Stack[0..N_STACK-1] + full Pass-1 sRed[0..N_STACK-1]! Wakes per-peer as
//     soon as rows [max(qs, t-3)..t] land in L2-hot p.reduced_buf, runs exact sconv_exact_v2 +
//     exact ptxas-13.0 attn_res_exact_v3 (tree16_sum_sq<MODE>, tldevice_exp_sub, rsqrt_approx)!

namespace s8_exact {

static __device__ __forceinline__ float rsqrt_approx(float x) {
  float y;
  asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

static __device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
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

static __device__ __forceinline__ uint4 ld_global_u4(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

static __device__ __forceinline__ uint32_t ld_global_u32(const void* p) {
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
  asm("ld.global.nc.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
  return v;
}

static __device__ __forceinline__ void load_f32x8_const(const float* p, float out[8]) {
  float4 v0 = ld_const_f4(p);
  float4 v1 = ld_const_f4(p + 4);
  out[0] = v0.x; out[1] = v0.y; out[2] = v0.z; out[3] = v0.w;
  out[4] = v1.x; out[5] = v1.y; out[6] = v1.z; out[7] = v1.w;
}

static __device__ __forceinline__ void unpack_u4_u16(uint4 v, uint16_t out[8]) {
  out[0] = (uint16_t)(v.x & 0xFFFFu); out[1] = (uint16_t)(v.x >> 16);
  out[2] = (uint16_t)(v.y & 0xFFFFu); out[3] = (uint16_t)(v.y >> 16);
  out[4] = (uint16_t)(v.z & 0xFFFFu); out[5] = (uint16_t)(v.z >> 16);
  out[6] = (uint16_t)(v.w & 0xFFFFu); out[7] = (uint16_t)(v.w >> 16);
}

static __device__ __forceinline__ uint4 pack_u16_u4(const uint16_t in[8]) {
  uint4 v;
  v.x = (uint32_t)in[0] | ((uint32_t)in[1] << 16);
  v.y = (uint32_t)in[2] | ((uint32_t)in[3] << 16);
  v.z = (uint32_t)in[4] | ((uint32_t)in[5] << 16);
  v.w = (uint32_t)in[6] | ((uint32_t)in[7] << 16);
  return v;
}

static __device__ __forceinline__ void unpack_bf16x8(uint4 raw, float out[8]) {
  const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(h2[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

static __device__ __forceinline__ void unpack_bf16x2(uint32_t raw, float out[2]) {
  float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
  out[0] = f.x;
  out[1] = f.y;
}

static __device__ __forceinline__ void store_f32x8_to_bf16(__nv_bfloat16* p, const float in[8]) {
  uint4 raw;
  __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    h2[i] = __halves2bfloat162(__float2bfloat16_rn(in[2 * i]), __float2bfloat16_rn(in[2 * i + 1]));
  *reinterpret_cast<uint4*>(p) = raw;
}

static __device__ __forceinline__ void store_f32x2_to_bf16(__nv_bfloat16* p, const float in[2]) {
  *reinterpret_cast<__nv_bfloat162*>(p) =
      __halves2bfloat162(__float2bfloat16_rn(in[0]), __float2bfloat16_rn(in[1]));
}

static __device__ __forceinline__ void divmod_cache(int64_t val, int n_cache, int n_shift, int64_t& q, int64_t& r) {
  if (n_shift >= 0) {
    q = val >> n_shift;
    r = val & ((1LL << n_shift) - 1);
  } else {
    q = val / n_cache;
    r = val % n_cache;
  }
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
      : "=f"(y)
      : "f"(a), "f"(b));
  return y;
}

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

static __device__ __forceinline__ void warp_sum_bfly4(float& a, float& b, float& c, float& d) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    a = __fadd_rn(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
    b = __fadd_rn(b, __shfl_xor_sync(0xFFFFFFFFu, b, off));
    c = __fadd_rn(c, __shfl_xor_sync(0xFFFFFFFFu, c, off));
    d = __fadd_rn(d, __shfl_xor_sync(0xFFFFFFFFu, d, off));
  }
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

static __device__ __forceinline__ void spin_gpu_ull(const unsigned long long* ptr, unsigned long long seq,
                                                    const Params& p, int tag) {
  if (ld_acquire_gpu(ptr) >= seq) return;
  const unsigned long long t0 = now_ns();
  unsigned int it = 0;
  while (ld_acquire_gpu(ptr) < seq) {
    if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar s8 rank %d: timeout waiting GPU-L2 tag=%d (seq %llu)\n", p.rank, tag, (unsigned long long)seq);
      r3_trap();
    }
  }
}

// Extract tap k (0..3) weights for 8 channels from 4 packed uint4 words (32 bf16 = 8 ch x 4 taps).
static __device__ __forceinline__ void unpack_w_tap8(const uint4 wp[4], int k, float w[8]) {
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const uint4 v = wp[g];
    uint16_t b0, b1;
    if (k == 0) { b0 = (uint16_t)(v.x & 0xFFFFu); b1 = (uint16_t)(v.z & 0xFFFFu); }
    else if (k == 1) { b0 = (uint16_t)(v.x >> 16); b1 = (uint16_t)(v.z >> 16); }
    else if (k == 2) { b0 = (uint16_t)(v.y & 0xFFFFu); b1 = (uint16_t)(v.w & 0xFFFFu); }
    else { b0 = (uint16_t)(v.y >> 16); b1 = (uint16_t)(v.w >> 16); }
    w[2 * g] = bf16_to_f32(b0);
    w[2 * g + 1] = bf16_to_f32(b1);
  }
}

static __device__ __forceinline__ void unpack_w_tap2(uint4 v, int k, float w[2]) {
  uint16_t b0, b1;
  if (k == 0) { b0 = (uint16_t)(v.x & 0xFFFFu); b1 = (uint16_t)(v.z & 0xFFFFu); }
  else if (k == 1) { b0 = (uint16_t)(v.x >> 16); b1 = (uint16_t)(v.z >> 16); }
  else if (k == 2) { b0 = (uint16_t)(v.y & 0xFFFFu); b1 = (uint16_t)(v.w & 0xFFFFu); }
  else { b0 = (uint16_t)(v.y >> 16); b1 = (uint16_t)(v.w >> 16); }
  w[0] = bf16_to_f32(b0);
  w[1] = bf16_to_f32(b1);
}

}  // namespace s8_exact

// kCase encoding (all 4 Inkling-2-Small call sites):
//   0: POST-WO first=True   (conv_addend=0, conv_res=0, mix_addend=0)
//   1: POST-WO first=False  (conv_addend=0, conv_res=0, mix_addend=1)
//   2: POST-MOE first=True  (conv_addend=1, conv_res=0, mix_addend=0)
//   3: POST-MOE first=False (conv_addend=1, conv_res=1, mix_addend=0; optional runtime sp.aux_out != nullptr)
template <int N_STACK, int kCase, int kPub>
__global__ void __launch_bounds__(256, 1)
ar_sconv_attnres_kernel(Params p, SconvAttnResParams sp) {
  constexpr int kW = 8;
  constexpr int D = 4608;
  constexpr int NS = N_STACK + 1;
  constexpr int REG_ST = (N_STACK < 4) ? N_STACK : 4;
  constexpr int HI_ST = (N_STACK > 4) ? (N_STACK - 4) : 0;
  constexpr bool HAS_CONV_ADDEND = (kCase >= 2);
  constexpr bool HAS_CONV_RES = (kCase == 3);
  constexpr bool HAS_MIX_ADDEND = (kCase == 1);

  extern __shared__ uint8_t dyn_smem[];
  // Dynamic SMEM layout (all 16-byte aligned, bank-conflict-free):
  //   [0 .. 12287]:       uint4 s_ctap_a0[3][256]
  //   [12288 .. 24575]:   uint4 s_ctap_a1[3][256]
  //   [24576 .. 27647]:   uint32_t s_ctap_b[3][256]
  //   [27648 .. +HI*9216]: s_stack_hi[HI_ST] (each: 256*uint4 + 256*uint4 + 256*uint32 = 9,216 B)
  uint4* s_ctap_a0 = reinterpret_cast<uint4*>(dyn_smem);
  uint4* s_ctap_a1 = reinterpret_cast<uint4*>(dyn_smem + 12288);
  uint32_t* s_ctap_b = reinterpret_cast<uint32_t*>(dyn_smem + 24576);
  uint8_t* s_stack_hi_base = dyn_smem + 27648;

  __shared__ unsigned long long s_seq;
  __shared__ int s_rows;
  __shared__ unsigned int s_ready;
  __shared__ uint32_t s_bf12_w[32];
  __shared__ unsigned int s_bf12_esc;
  __shared__ Bf12Hdr s_bf12_hdr;
  __shared__ float sRed[NS][4][8];
  __shared__ float sSq2[2][8];

  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  const long long full_rows = (long long)sp.T;
  if (threadIdx.x == 0) {
    const unsigned long long seq0 = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    const int r0 = *reinterpret_cast<const volatile int*>(p.rows_ptr);
    s_seq = seq0;
    s_rows = r0;
    if constexpr ((kPub & kR6Pdl) == 0) {
      if (blockIdx.x == 0) early = s8_read_early(p);
    }
    if (p.timing && blockIdx.x == 0) p.st->t_start = now_ns();
    if constexpr ((kPub & kR3Sbuf) != 0) {
      if (blockIdx.x == 0) {
        const long long n0 = (r0 >= 1 && r0 <= full_rows) ? (long long)r0 : full_rows;
        const bool os0 = (n0 <= (long long)p.oneshot_max_rows) || (p.bf12_os4 != 0 && n0 == 4);
        const bool use_bf12_os0 = (p.bf12_os4 != 0 && n0 == 4 && n0 > (long long)p.oneshot_max_rows);
        if (!use_bf12_os0) {
          Params p0 = p;
          p0.nvec = n0 * 576LL;
          p0.chunk = n0 * 72LL;
          __threadfence_system();
          const uint64_t src_off =
              (uint64_t)(reinterpret_cast<const char*>(p.in) - reinterpret_cast<const char*>(p.ctrl));
          if (os0) r3_publish1<kOneShot, kPub>(p0, seq0, early, src_off);
          else r3_publish1<kTwoShot, kPub>(p0, seq0, early, src_off);
        }
      }
    }
  }
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int n_live = (s_rows >= 1 && s_rows <= sp.T) ? s_rows : sp.T;
  const long long nvec = (long long)n_live * 576LL;
  const long long chunk = (long long)n_live * 72LL;  // exact! 576 / 8 == 72
  const bool use_bf12_os = (p.bf12_os4 != 0 && n_live == 4 && n_live > p.oneshot_max_rows);
  const bool use_oneshot = (n_live <= p.oneshot_max_rows) || use_bf12_os;
  p.nvec = nvec;
  p.chunk = chunk;

  // Every CTA (including dead worker CTAs!) participates in seq advance before exiting.
  if (threadIdx.x == 255 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }
  const int slot = static_cast<int>(seq % kSlots);

  // ===============================================================================================
  // ROLE 0: CTA 0 — RDMA Master (Phase-1 publish, 1-warp C2C polling, L2-hot staging/BF12 codec)
  // ===============================================================================================
  if (blockIdx.x == 0) {
    if constexpr ((kPub & kR3Sbuf) != 0) {
      if (!use_bf12_os && threadIdx.x == 32) {
        uint64_t* hdr = reinterpret_cast<uint64_t*>(const_cast<uint4*>(p.in) - 1);
        const uint64_t last = ld_relaxed_sys(hdr);
        if (last != 0 && seq - last < 2) r3_trap();
        st_relaxed_sys(hdr, seq);
      }
    }
    if ((kPub & kR3Sbuf) == 0 || use_bf12_os) {
      char* send_c = p.send + (size_t)slot * (size_t)p.max_bytes;
      uint64_t p1_bytes = (uint64_t)nvec * 16ull;
      if (use_bf12_os) {
        p1_bytes = bf12_pack_mem_cta(p.in, (uint32_t)nvec, send_c, s_bf12_w, &s_bf12_esc);
      } else {
        uint4* send4 = reinterpret_cast<uint4*>(send_c);
        for (long long i = threadIdx.x; i < nvec; i += 256) send4[i] = p.in[i];
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        if (use_oneshot) r3_publish1_bytes<kOneShot, kPub & ~kR3Sbuf>(p, seq, early, 0, p1_bytes);
        else r3_publish1_bytes<kTwoShot, kPub & ~kR3Sbuf>(p, seq, early, 0, p1_bytes);
      }
    }

    const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
    const char* rbase = p.recv + (long long)slot * kW * rstride;
    if (use_oneshot) {
      if (use_bf12_os) {
        // Overlapped warp-ballot Phase-1 arrival + BF12-OS4 unpack into L2-hot bf12_os_scratch!
        unsigned int pending = ((1u << kW) - 1u) & ~(1u << p.rank);
        const unsigned long long t0 = now_ns();
        unsigned int it = 0;
        while (pending) {
          unsigned int rmask = 0u;
          if (threadIdx.x < 32) {
            bool arrived = false;
            if (threadIdx.x < kW && (pending & (1u << threadIdx.x))) {
              const uint64_t* fp = p.flags + (slot * kW + threadIdx.x) * kFlagStride;
              arrived = (ld_acquire_sys(fp) >= seq);
            }
            rmask = __ballot_sync(0xFFFFFFFFu, arrived);
          }
          if (threadIdx.x == 0) s_ready = rmask;
          __syncthreads();
          rmask = s_ready;
          if (!rmask) {
            if (p.poll_sleep_ns) s6_nanosleep(p.poll_sleep_ns);
            if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) r3_trap();
            continue;
          }
          unsigned int m = rmask;
          while (m) {
            const int r = __ffs(m) - 1;
            m &= ~(1u << r);
            bf12_unpack_to_dst_cta(rbase + (long long)r * rstride,
                                   p.bf12_os_scratch + (long long)r * nvec, &s_bf12_hdr);
            __syncthreads();
          }
          pending &= ~rmask;
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
          st_relaxed_gpu(&p.st->ar_done_cnt, seq);
          st_release_gpu(&p.st->p1_done, seq);
        }
      } else {
        // Raw 1-shot: poll 7 flags on 1 warp -> release p1_done to q_idx==0 parallel row reducers!
        if (threadIdx.x < kW && threadIdx.x != p.rank) {
          const uint64_t* fp = p.flags + (slot * kW + threadIdx.x) * kFlagStride;
          spin_until_thin(fp, seq, p, threadIdx.x, 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          st_relaxed_gpu(&p.st->ar_done_cnt, seq);
          st_release_sys_ull(&p.st->p1_done, seq);
        }
      }
      if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      return;
    }

    // 2-shot Phase 1: poll 7 flags on 1 warp, reduce chunk `rank` -> L2-hot reduced_buf + send2
    if (threadIdx.x < kW && threadIdx.x != p.rank) {
      const uint64_t* fp = p.flags + (slot * kW + threadIdx.x) * kFlagStride;
      spin_until_thin(fp, seq, p, threadIdx.x, 1);
    }
    __syncthreads();
    const long long b_p2 = (long long)p.rank * chunk;
    char* send2_c = p.send2 + (size_t)slot * (size_t)p.max_chunk;
    uint4* send2 = reinterpret_cast<uint4*>(send2_c);
    uint4 rv[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const long long i = (long long)threadIdx.x + (long long)k * 256LL;
      if (i < chunk) {
        uint4 v[kW];
#pragma unroll
        for (int r = 0; r < kW; ++r)
          v[r] = (r == p.rank) ? p.in[b_p2 + i]
                               : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
        const uint4 o = packv<false>(acc);
        p.reduced_buf[b_p2 + i] = o;
        if (p.wire_bf12) rv[k] = o;
        else send2[i] = o;
      }
    }
    uint64_t p2_bytes = (uint64_t)chunk * 16ull;
    if (p.wire_bf12) {
      p2_bytes = bf12_pack_regs_cta<3>(rv, (uint32_t)chunk, send2_c, s_bf12_w, &s_bf12_esc);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      s8_publish2(p, slot, seq, p2_bytes);
      st_release_gpu(&p.st->p2_peer_seq[p.rank], seq);
    }

    // 2-shot Phase 2: warp-ballot pipelined arrival -> single-__ldcv L2 staging/BF12 unpack -> per-peer GPU-L2 release!
    const char* r2 = p.recv2 + (long long)slot * kW * p.max_chunk;
    unsigned int pending = ((1u << kW) - 1u) & ~(1u << p.rank);
    const unsigned long long t0 = now_ns();
    unsigned int it = 0;
    while (pending) {
      unsigned int rmask = 0u;
      if (threadIdx.x < 32) {
        bool arrived = false;
        if (threadIdx.x < kW && (pending & (1u << threadIdx.x))) {
          const uint64_t* fp = p.flags2 + (slot * kW + threadIdx.x) * kFlagStride;
          arrived = (ld_acquire_sys(fp) >= seq);
        }
        rmask = __ballot_sync(0xFFFFFFFFu, arrived);
      }
      if (threadIdx.x == 0) s_ready = rmask;
      __syncthreads();
      rmask = s_ready;
      if (!rmask) {
        if (p.poll_sleep_ns) s6_nanosleep(p.poll_sleep_ns);
        if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) r3_trap();
        continue;
      }
      unsigned int m = rmask;
      while (m) {
        const int r = __ffs(m) - 1;
        m &= ~(1u << r);
        uint4* dst_r = p.reduced_buf + (long long)r * chunk;
        if (p.wire_bf12) {
          bf12_unpack_to_dst_cta(r2 + (long long)r * p.max_chunk, dst_r, &s_bf12_hdr);
          __syncthreads();
        } else {
          const uint4* src_r = reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk);
          for (long long i = threadIdx.x; i < chunk; i += 256) dst_r[i] = __ldcv(src_r + i);
        }
      }
      __threadfence();
      __syncthreads();
      if (threadIdx.x < kW && (rmask & (1u << threadIdx.x))) {
        st_release_gpu(&p.st->p2_peer_seq[threadIdx.x], seq);
      }
      pending &= ~rmask;
    }
    if (threadIdx.x == 0) st_relaxed_gpu(&p.st->ar_done_cnt, seq);
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
    return;
  }

  // ===============================================================================================
  // ROLE 1: Worker CTAs 1..nq*T — (q_idx, t) SConv + AttnRes + RMSNorm
  // ===============================================================================================
  const int wid = blockIdx.x - 1;
  const int q_idx = wid / sp.T;
  const int t = wid - q_idx * sp.T;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int off_a0 = tid * 8;
  const int off_a1 = 2048 + tid * 8;
  const int off_b = 4096 + tid * 2;

  // Dead rows exit early (after zeroing AuxOut if present on q_idx==0)
  if (t >= n_live) {
    if (sp.aux_out != nullptr && q_idx == 0) {
      uint16_t* ap = sp.aux_out + (int64_t)t * D;
      *reinterpret_cast<uint4*>(ap + off_a0) = make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint4*>(ap + off_a1) = make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint32_t*>(ap + off_b) = 0u;
    }
    return;
  }

  const int64_t q_off = (int64_t)q_idx * D;
  const int64_t row_off = (int64_t)t * D;

  // -----------------------------------------------------------------------------------------------
  // PRELOAD PHASE (100% hidden behind the 20-35 us RDMA network wait!)
  // -----------------------------------------------------------------------------------------------
  // 1) Queries + L2 prefetch of immutable NormW (0 extra regs)
  float qa0[8], qa1[8], qb[2];
  s8_exact::load_f32x8_const(sp.queries + q_off + off_a0, qa0);
  s8_exact::load_f32x8_const(sp.queries + q_off + off_a1, qa1);
  {
    float2 qbf = s8_exact::ld_const_f2(sp.queries + q_off + off_b);
    qb[0] = qbf.x;
    qb[1] = qbf.y;
  }
  if (tid < 72) s8_exact::prefetch_l2(sp.norm_w + q_off + tid * 64);

  // 2) Conv metadata + ConvW (packed uint4: 36 regs) + ConvCache cached lags 3..1 (SMEM)
  const int64_t pos = sp.pos[t];
  const int64_t cslot = sp.slot_map[t];
  const int32_t req = sp.seq_idx[t];
  const int32_t qs = sp.q_start[t];
  const bool valid = (cslot >= 0);

  uint4 cw_a0[4], cw_a1[4], cw_b;
  {
    const uint4* wp0 = reinterpret_cast<const uint4*>(sp.conv_w + (int64_t)off_a0 * 4);
    const uint4* wp1 = reinterpret_cast<const uint4*>(sp.conv_w + (int64_t)off_a1 * 4);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      cw_a0[g] = s8_exact::ld_const_u4(wp0 + g);
      cw_a1[g] = s8_exact::ld_const_u4(wp1 + g);
    }
    cw_b = s8_exact::ld_const_u4(sp.conv_w + (int64_t)off_b * 4);
  }

  bool has_cached[3] = {false, false, false};
  if (valid) {
    const int32_t* bt_row = sp.block_table + (int64_t)req * sp.stride_bt;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const int lag = 3 - k;
      const int64_t src = pos - lag;
      if (src >= 0 && (t - lag < qs)) {
        has_cached[k] = true;
        int64_t lblk, rem;
        s8_exact::divmod_cache(src, sp.n_cache, sp.n_shift, lblk, rem);
        if (lblk >= sp.max_blocks) lblk = sp.max_blocks - 1;
        const int64_t blk = (int64_t)bt_row[lblk];
        const uint16_t* cp = sp.cache + blk * sp.stride_c_blk + rem * sp.stride_c_n + sp.stream_off;
        s_ctap_a0[k * 256 + tid] = s8_exact::ld_global_u4(cp + off_a0);
        s_ctap_a1[k * 256 + tid] = s8_exact::ld_global_u4(cp + off_a1);
        s_ctap_b[k * 256 + tid] = s8_exact::ld_global_u32(cp + off_b);
      }
    }
  }

  // 3) ConvAddend / ConvRes / MixLast (mutually exclusive -> <= 18 live regs!)
  uint4 cadd_a0, cadd_a1, cres_a0, cres_a1, mlast_a0, mlast_a1;
  uint32_t cadd_b = 0u, cres_b = 0u, mlast_b = 0u;
  if constexpr (HAS_CONV_ADDEND) {
    cadd_a0 = s8_exact::ld_global_u4(sp.conv_addend + row_off + off_a0);
    cadd_a1 = s8_exact::ld_global_u4(sp.conv_addend + row_off + off_a1);
    cadd_b = s8_exact::ld_global_u32(sp.conv_addend + row_off + off_b);
  }
  if constexpr (HAS_CONV_RES) {
    cres_a0 = s8_exact::ld_global_u4(sp.conv_res + row_off + off_a0);
    cres_a1 = s8_exact::ld_global_u4(sp.conv_res + row_off + off_a1);
    cres_b = s8_exact::ld_global_u32(sp.conv_res + row_off + off_b);
    // Aliasing guard for nq>1 when ConvRes == ConvOut (stack[num_blocks]): signal preload done during RDMA wait!
    if (sp.nq > 1) {
      __syncthreads();
      if (tid == 0) atom_add_acqrel_gpu(&p.st->row_ready[t], 1u);
    }
  }
  if constexpr (HAS_MIX_ADDEND) {
    mlast_a0 = s8_exact::ld_global_u4(sp.mix_last + row_off + off_a0);
    mlast_a1 = s8_exact::ld_global_u4(sp.mix_last + row_off + off_a1);
    mlast_b = s8_exact::ld_global_u32(sp.mix_last + row_off + off_b);
  }

  // 4) ALL Stack[0..N_STACK-1] burst loads + ENTIRE Pass-1 sum_sq & dot reductions during RDMA wait!
  //    n < 4 kept in registers; n >= 4 stashed in dynamic SMEM (zero local-memory register spills!).
  uint4 st_r0[REG_ST], st_r1[REG_ST];
  uint32_t st_rb[REG_ST];
#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    const __nv_bfloat16* sp_n = sp.stack + (int64_t)n * sp.stride_s + row_off;
    const uint4 r0 = s8_exact::ld_global_u4(sp_n + off_a0);
    const uint4 r1 = s8_exact::ld_global_u4(sp_n + off_a1);
    const uint32_t rb = s8_exact::ld_global_u32(sp_n + off_b);
    if constexpr (HI_ST > 0) {
      if (n < 4) {
        st_r0[n] = r0;
        st_r1[n] = r1;
        st_rb[n] = rb;
      } else {
        uint8_t* dst_hi = s_stack_hi_base + (size_t)(n - 4) * 9216u;
        reinterpret_cast<uint4*>(dst_hi)[tid] = r0;
        reinterpret_cast<uint4*>(dst_hi + 4096u)[tid] = r1;
        reinterpret_cast<uint32_t*>(dst_hi + 8192u)[tid] = rb;
      }
    } else {
      st_r0[n] = r0;
      st_r1[n] = r1;
      st_rb[n] = rb;
    }
    float xa0[8], xa1[8], xb[2];
    s8_exact::unpack_bf16x8(r0, xa0);
    s8_exact::unpack_bf16x8(r1, xa1);
    s8_exact::unpack_bf16x2(rb, xb);
    float dt0[8], dt1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      dt0[i] = __fmul_rn(qa0[i], xa0[i]);
      dt1[i] = __fmul_rn(qa1[i], xa1[i]);
    }
    float sqa = s8_exact::tree16_sum_sq<0>(xa0, xa1);
    float dta = s8_exact::tree16_sum(dt0, dt1);
    float sqb = __fadd_rn(__fmul_rn(xb[0], xb[0]), __fmul_rn(xb[1], xb[1]));
    float dtb = __fadd_rn(__fmul_rn(qb[0], xb[0]), __fmul_rn(qb[1], xb[1]));
    s8_exact::warp_sum_bfly4(sqa, dta, sqb, dtb);
    if (lane == 0) {
      sRed[n][0][warp] = sqa;
      sRed[n][1][warp] = sqb;
      sRed[n][2][warp] = dta;
      sRed[n][3][warp] = dtb;
    }
  }

  // -----------------------------------------------------------------------------------------------
  // WAIT FOR REQUIRED REDUCED ROWS IN L2-HOT p.reduced_buf
  // -----------------------------------------------------------------------------------------------
  int row_lo = (valid && pos > 0) ? max((int)qs, max(0, t - (int)min(pos, (int64_t)3))) : t;
  if (row_lo > t) row_lo = t;

  if (use_oneshot) {
    if (q_idx == 0) {
      // Wait for CTA 0's p1_done, then reduce THIS row `t` in parallel (only <= 3 vecs/thread!) -> p.reduced_buf!
      if (tid == 0) {
        s8_exact::spin_gpu_ull(&p.st->p1_done, seq, p, 100);
        if (!use_bf12_os) (void)ld_acquire_sys_ull(&p.st->p1_done);
      }
      __syncthreads();
      if (!use_bf12_os) fence_acquire_sys();
      const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
      const char* rbase = p.recv + (long long)slot * kW * rstride;
      const long long row_vbase = (long long)t * 576LL;
      // 576 vecs = 256 (0..255) + 256 (256..511) + 64 (512..575)
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const long long lv = (long long)tid + (long long)k * 256LL;
        if (lv < 576LL) {
          const long long gv = row_vbase + lv;
          uint4 v[kW];
#pragma unroll
          for (int r = 0; r < kW; ++r) {
            if (r == p.rank) v[r] = p.in[gv];
            else if (use_bf12_os) v[r] = p.bf12_os_scratch[(long long)r * nvec + gv];
            else v[r] = __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + gv);
          }
          float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
          p.reduced_buf[gv] = packv<false>(acc);
        }
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) st_release_gpu(&p.st->row_done_seq[t], seq);
      // Wait for any intra-batch lag rows [row_lo .. t-1]
      if (row_lo < t) {
        if (tid == 0) {
          for (int rr = row_lo; rr < t; ++rr) s8_exact::spin_gpu_ull(&p.st->row_done_seq[rr], seq, p, 200 + rr);
        }
        __syncthreads();
      }
    } else {
      if (tid == 0) {
        for (int rr = row_lo; rr <= t; ++rr) s8_exact::spin_gpu_ull(&p.st->row_done_seq[rr], seq, p, 200 + rr);
      }
      __syncthreads();
    }
  } else {
    // 2-shot: wait ONLY for the exact peers [r_lo .. r_hi] owning rows [row_lo .. t] in GPU-L2 p2_peer_seq!
    const int r_lo = (int)(((long long)row_lo * 576LL) / chunk);
    int r_hi = (int)(((long long)(t + 1) * 576LL - 1LL) / chunk);
    if (r_hi >= kW) r_hi = kW - 1;
    if (tid == 0) {
      for (int r = r_lo; r <= r_hi; ++r) s8_exact::spin_gpu_ull(&p.st->p2_peer_seq[r], seq, p, 300 + r);
    }
    __syncthreads();
  }

  // -----------------------------------------------------------------------------------------------
  // SCONV (EXACT sconv_exact_v2 FMA & BF16 ROUNDING ORDER) DIRECTLY FROM L2-HOT p.reduced_buf!
  // -----------------------------------------------------------------------------------------------
  const uint16_t* X_base = reinterpret_cast<const uint16_t*>(p.reduced_buf);
  const uint16_t* x_cur = X_base + row_off;
  const uint4 cur_u4_a0 = s8_exact::ld_global_u4(x_cur + off_a0);
  const uint4 cur_u4_a1 = s8_exact::ld_global_u4(x_cur + off_a1);
  const uint32_t cur_u32_b = s8_exact::ld_global_u32(x_cur + off_b);

  if (q_idx == 0 && valid) {
    int64_t dst_blk, dst_rem;
    s8_exact::divmod_cache(cslot, sp.n_cache, sp.n_shift, dst_blk, dst_rem);
    uint16_t* dst_c = sp.cache + dst_blk * sp.stride_c_blk + dst_rem * sp.stride_c_n + sp.stream_off;
    *reinterpret_cast<uint4*>(dst_c + off_a0) = cur_u4_a0;
    *reinterpret_cast<uint4*>(dst_c + off_a1) = cur_u4_a1;
    *reinterpret_cast<uint32_t*>(dst_c + off_b) = cur_u32_b;
  }

  uint16_t cur_a0[8], cur_a1[8], cur_b[2];
  s8_exact::unpack_u4_u16(cur_u4_a0, cur_a0);
  s8_exact::unpack_u4_u16(cur_u4_a1, cur_a1);
  cur_b[0] = (uint16_t)(cur_u32_b & 0xFFFFu);
  cur_b[1] = (uint16_t)(cur_u32_b >> 16);

  float conv_a0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float conv_a1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float conv_b[2] = {0.f, 0.f};

#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int lag = 3 - k;
    const int64_t src = pos - lag;
    const int row = t - lag;
    const bool in_win = valid && (src >= 0);
    const bool intra = in_win && (row >= qs);
    const bool cached = in_win && (row < qs);

    uint16_t t_a0[8], t_a1[8], t_b[2];
    bool nonzero = false;
    if (intra) {
      nonzero = true;
      if (lag == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { t_a0[i] = cur_a0[i]; t_a1[i] = cur_a1[i]; }
        t_b[0] = cur_b[0]; t_b[1] = cur_b[1];
      } else {
        const uint16_t* xr = X_base + (int64_t)row * D;
        s8_exact::unpack_u4_u16(s8_exact::ld_global_u4(xr + off_a0), t_a0);
        s8_exact::unpack_u4_u16(s8_exact::ld_global_u4(xr + off_a1), t_a1);
        const uint32_t xb = s8_exact::ld_global_u32(xr + off_b);
        t_b[0] = (uint16_t)(xb & 0xFFFFu);
        t_b[1] = (uint16_t)(xb >> 16);
      }
    } else if (cached && k < 3 && has_cached[k]) {
      nonzero = true;
      s8_exact::unpack_u4_u16(s_ctap_a0[k * 256 + tid], t_a0);
      s8_exact::unpack_u4_u16(s_ctap_a1[k * 256 + tid], t_a1);
      const uint32_t cb = s_ctap_b[k * 256 + tid];
      t_b[0] = (uint16_t)(cb & 0xFFFFu);
      t_b[1] = (uint16_t)(cb >> 16);
    }

    float w_a0[8], w_a1[8], w_b[2];
    s8_exact::unpack_w_tap8(cw_a0, k, w_a0);
    s8_exact::unpack_w_tap8(cw_a1, k, w_a1);
    s8_exact::unpack_w_tap2(cw_b, k, w_b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float xt0 = nonzero ? __fadd_rn(s8_exact::bf16_to_f32(t_a0[i]), 0.0f) : 0.0f;
      float xt1 = nonzero ? __fadd_rn(s8_exact::bf16_to_f32(t_a1[i]), 0.0f) : 0.0f;
      conv_a0[i] = fmaf(xt0, w_a0[i], conv_a0[i]);
      conv_a1[i] = fmaf(xt1, w_a1[i], conv_a1[i]);
    }
    {
      float xtb0 = nonzero ? __fadd_rn(s8_exact::bf16_to_f32(t_b[0]), 0.0f) : 0.0f;
      float xtb1 = nonzero ? __fadd_rn(s8_exact::bf16_to_f32(t_b[1]), 0.0f) : 0.0f;
      conv_b[0] = fmaf(xtb0, w_b[0], conv_b[0]);
      conv_b[1] = fmaf(xtb1, w_b[1], conv_b[1]);
    }
  }

  uint16_t y_a0_u16[8], y_a1_u16[8], y_b_u16[2];
  uint16_t ca_a0[8], ca_a1[8], ca_b[2], cr_a0[8], cr_a1[8], cr_b[2];
  if constexpr (HAS_CONV_ADDEND) {
    s8_exact::unpack_u4_u16(cadd_a0, ca_a0);
    s8_exact::unpack_u4_u16(cadd_a1, ca_a1);
    ca_b[0] = (uint16_t)(cadd_b & 0xFFFFu); ca_b[1] = (uint16_t)(cadd_b >> 16);
  }
  if constexpr (HAS_CONV_RES) {
    s8_exact::unpack_u4_u16(cres_a0, cr_a0);
    s8_exact::unpack_u4_u16(cres_a1, cr_a1);
    cr_b[0] = (uint16_t)(cres_b & 0xFFFFu); cr_b[1] = (uint16_t)(cres_b >> 16);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint16_t yb0 = s8_exact::f32_to_bf16(__fmul_rn(__fadd_rn(conv_a0[i], s8_exact::bf16_to_f32(cur_a0[i])), sp.scale));
    uint16_t yb1 = s8_exact::f32_to_bf16(__fmul_rn(__fadd_rn(conv_a1[i], s8_exact::bf16_to_f32(cur_a1[i])), sp.scale));
    if constexpr (HAS_CONV_ADDEND) {
      float a0 = s8_exact::bf16_to_f32(ca_a0[i]);
      float a1 = s8_exact::bf16_to_f32(ca_a1[i]);
      if constexpr (HAS_CONV_RES) {
        a0 = __fadd_rn(s8_exact::bf16_to_f32(cr_a0[i]), a0);
        a1 = __fadd_rn(s8_exact::bf16_to_f32(cr_a1[i]), a1);
      }
      yb0 = s8_exact::f32_to_bf16(__fadd_rn(a0, s8_exact::bf16_to_f32(yb0)));
      yb1 = s8_exact::f32_to_bf16(__fadd_rn(a1, s8_exact::bf16_to_f32(yb1)));
    }
    y_a0_u16[i] = yb0;
    y_a1_u16[i] = yb1;
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint16_t yb = s8_exact::f32_to_bf16(__fmul_rn(__fadd_rn(conv_b[i], s8_exact::bf16_to_f32(cur_b[i])), sp.scale));
    if constexpr (HAS_CONV_ADDEND) {
      float a = s8_exact::bf16_to_f32(ca_b[i]);
      if constexpr (HAS_CONV_RES) a = __fadd_rn(s8_exact::bf16_to_f32(cr_b[i]), a);
      yb = s8_exact::f32_to_bf16(__fadd_rn(a, s8_exact::bf16_to_f32(yb)));
    }
    y_b_u16[i] = yb;
  }

  const uint4 y_r0 = s8_exact::pack_u16_u4(y_a0_u16);
  const uint4 y_r1 = s8_exact::pack_u16_u4(y_a1_u16);
  const uint32_t y_rb = (uint32_t)y_b_u16[0] | ((uint32_t)y_b_u16[1] << 16);

  if (q_idx == 0) {
    if constexpr (HAS_CONV_RES) {
      if (sp.nq > 1) {
        if (tid == 0) {
          while (ld_acquire_gpu_u32(&p.st->row_ready[t]) < (unsigned int)sp.nq) {}
          st_relaxed_gpu_u32(&p.st->row_ready[t], 0u);
        }
        __syncthreads();
      }
    }
    uint16_t* cop = sp.conv_out + row_off;
    *reinterpret_cast<uint4*>(cop + off_a0) = y_r0;
    *reinterpret_cast<uint4*>(cop + off_a1) = y_r1;
    *reinterpret_cast<uint32_t*>(cop + off_b) = y_rb;
    if (sp.aux_out != nullptr) {
      uint16_t* aop = sp.aux_out + row_off;
      *reinterpret_cast<uint4*>(aop + off_a0) = y_r0;
      *reinterpret_cast<uint4*>(aop + off_a1) = y_r1;
      *reinterpret_cast<uint32_t*>(aop + off_b) = y_rb;
    }
  }

  // Late PDL trigger (~1.2 us before exit): dependent router/qkvr launches with ZERO Stack DRAM contention!
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();

  // -----------------------------------------------------------------------------------------------
  // ATTNRES PASS 1 (FINAL SOURCE N_STACK) + SOFTMAX + PASS 2 + RMSNORM (0-ULP ptxas-13.0 EXACT!)
  // -----------------------------------------------------------------------------------------------
  float last_f0[8], last_f1[8], last_fb[2];
  {
    float lx0[8], lx1[8], lxb[2];
    if constexpr (HAS_MIX_ADDEND) {
      s8_exact::unpack_bf16x8(mlast_a0, lx0);
      s8_exact::unpack_bf16x8(mlast_a1, lx1);
      s8_exact::unpack_bf16x2(mlast_b, lxb);
      float aa0[8], aa1[8], ab[2];
      s8_exact::unpack_bf16x8(y_r0, aa0);
      s8_exact::unpack_bf16x8(y_r1, aa1);
      s8_exact::unpack_bf16x2(y_rb, ab);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        lx0[i] = __fadd_rn(lx0[i], aa0[i]);
        lx1[i] = __fadd_rn(lx1[i], aa1[i]);
        last_f0[i] = lx0[i];
        last_f1[i] = lx1[i];
      }
      lxb[0] = __fadd_rn(lxb[0], ab[0]);
      lxb[1] = __fadd_rn(lxb[1], ab[1]);
      last_fb[0] = lxb[0];
      last_fb[1] = lxb[1];
    } else {
      s8_exact::unpack_bf16x8(y_r0, lx0);
      s8_exact::unpack_bf16x8(y_r1, lx1);
      s8_exact::unpack_bf16x2(y_rb, lxb);
    }
    float dt0[8], dt1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      dt0[i] = __fmul_rn(qa0[i], lx0[i]);
      dt1[i] = __fmul_rn(qa1[i], lx1[i]);
    }
    constexpr int P1_LAST_MODE = HAS_MIX_ADDEND ? 1 : 0;
    float sqa = s8_exact::tree16_sum_sq<P1_LAST_MODE>(lx0, lx1);
    float dta = s8_exact::tree16_sum(dt0, dt1);
    float sqb = __fadd_rn(__fmul_rn(lxb[0], lxb[0]), __fmul_rn(lxb[1], lxb[1]));
    float dtb = __fadd_rn(__fmul_rn(qb[0], lxb[0]), __fmul_rn(qb[1], lxb[1]));
    s8_exact::warp_sum_bfly4(sqa, dta, sqb, dtb);
    if (lane == 0) {
      sRed[N_STACK][0][warp] = sqa;
      sRed[N_STACK][1][warp] = sqb;
      sRed[N_STACK][2][warp] = dta;
      sRed[N_STACK][3][warp] = dtb;
    }
  }

  // BARRIER 1
  __syncthreads();

  constexpr float INV_D = 1.0f / (float)D;
  constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;
  float probs_local[NS];
  float inv_sum;
  {
    float my_lg = -INFINITY;
    float my_ir = 0.0f;
    const int grp = lane >> 3;
    const int l8 = lane & 7;
#pragma unroll
    for (int n = 0; n < NS; ++n) {
      float v = s8_exact::warp8_sum_bfly(sRed[n][grp][l8]);
      float sa = __shfl_sync(0xFFFFFFFFu, v, 0);
      float sb = __shfl_sync(0xFFFFFFFFu, v, 8);
      float da = __shfl_sync(0xFFFFFFFFu, v, 16);
      float db = __shfl_sync(0xFFFFFFFFu, v, 24);
      float rstd = s8_exact::rsqrt_approx(fmaf(__fadd_rn(sa, sb), INV_D, sp.eps));
      float dot = __fadd_rn(da, db);
      if (lane == n) {
        my_lg = __fmul_rn(dot, rstd);
        my_ir = rstd;
      }
    }
    float mx = my_lg;
    if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
    if constexpr (BLOCK_N >= 8) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
    if constexpr (BLOCK_N >= 4) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
    if constexpr (BLOCK_N >= 2) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
    float pr = (lane < NS) ? s8_exact::tldevice_exp_sub(my_lg, mx) : 0.0f;
    float s_pr = pr;
    if constexpr (BLOCK_N >= 16) s_pr = __fadd_rn(s_pr, __shfl_xor_sync(0xFFFFFFFFu, s_pr, 8));
    if constexpr (BLOCK_N >= 8) s_pr = __fadd_rn(s_pr, __shfl_xor_sync(0xFFFFFFFFu, s_pr, 4));
    if constexpr (BLOCK_N >= 4) s_pr = __fadd_rn(s_pr, __shfl_xor_sync(0xFFFFFFFFu, s_pr, 2));
    if constexpr (BLOCK_N >= 2) s_pr = __fadd_rn(s_pr, __shfl_xor_sync(0xFFFFFFFFu, s_pr, 1));
    inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, s_pr, 0));
    pr = __fmul_rn(pr, my_ir);  // normalize_values == true in Inkling-2-Small
#pragma unroll
    for (int n = 0; n < NS; ++n) probs_local[n] = __shfl_sync(0xFFFFFFFFu, pr, n);
  }

  // Load NormW (already prefetched into L2 during RDMA wait!)
  const uint4 nw_r0 = s8_exact::ld_const_u4(sp.norm_w + q_off + off_a0);
  const uint4 nw_r1 = s8_exact::ld_const_u4(sp.norm_w + q_off + off_a1);
  const uint32_t nw_rb = s8_exact::ld_const_u32(sp.norm_w + q_off + off_b);

  // Pass 2: ZERO global reads (n<4 from regs, n>=4 from SMEM)!
  float acc0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float accb[2] = {0, 0};

#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    const float pr = probs_local[n];
    uint4 r0, r1;
    uint32_t rb;
    if constexpr (HI_ST > 0) {
      if (n < 4) {
        r0 = st_r0[n]; r1 = st_r1[n]; rb = st_rb[n];
      } else {
        const uint8_t* src_hi = s_stack_hi_base + (size_t)(n - 4) * 9216u;
        r0 = reinterpret_cast<const uint4*>(src_hi)[tid];
        r1 = reinterpret_cast<const uint4*>(src_hi + 4096u)[tid];
        rb = reinterpret_cast<const uint32_t*>(src_hi + 8192u)[tid];
      }
    } else {
      r0 = st_r0[n]; r1 = st_r1[n]; rb = st_rb[n];
    }
    const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
    const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f0 = __bfloat1622float2(h0[i]);
      acc0[2 * i] = fmaf(pr, f0.x, acc0[2 * i]);
      acc0[2 * i + 1] = fmaf(pr, f0.y, acc0[2 * i + 1]);
      float2 f1 = __bfloat1622float2(h1[i]);
      acc1[2 * i] = fmaf(pr, f1.x, acc1[2 * i]);
      acc1[2 * i + 1] = fmaf(pr, f1.y, acc1[2 * i + 1]);
    }
    float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rb));
    accb[0] = fmaf(pr, fb.x, accb[0]);
    accb[1] = fmaf(pr, fb.y, accb[1]);
  }
  {
    const float pr = probs_local[N_STACK];
    if constexpr (HAS_MIX_ADDEND) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc0[i] = fmaf(pr, last_f0[i], acc0[i]);
        acc1[i] = fmaf(pr, last_f1[i], acc1[i]);
      }
      accb[0] = fmaf(pr, last_fb[0], accb[0]);
      accb[1] = fmaf(pr, last_fb[1], accb[1]);
    } else {
      const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&y_r0);
      const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&y_r1);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f0 = __bfloat1622float2(h0[i]);
        acc0[2 * i] = fmaf(pr, f0.x, acc0[2 * i]);
        acc0[2 * i + 1] = fmaf(pr, f0.y, acc0[2 * i + 1]);
        float2 f1 = __bfloat1622float2(h1[i]);
        acc1[2 * i] = fmaf(pr, f1.x, acc1[2 * i]);
        acc1[2 * i + 1] = fmaf(pr, f1.y, acc1[2 * i + 1]);
      }
      float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&y_rb));
      accb[0] = fmaf(pr, fb.x, accb[0]);
      accb[1] = fmaf(pr, fb.y, accb[1]);
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc0[i] = __fmul_rn(acc0[i], inv_sum);
    acc1[i] = __fmul_rn(acc1[i], inv_sum);
  }
  accb[0] = __fmul_rn(accb[0], inv_sum);
  accb[1] = __fmul_rn(accb[1], inv_sum);

  constexpr int P2_MODE = (N_STACK == 8) ? 3 : (HAS_MIX_ADDEND ? 1 : 0);
  float sqa = s8_exact::tree16_sum_sq<P2_MODE>(acc0, acc1);
  float sqb = __fadd_rn(__fmul_rn(accb[0], accb[0]), __fmul_rn(accb[1], accb[1]));
  s8_exact::warp_sum_bfly2(sqa, sqb);
  if (lane == 0) {
    sSq2[0][warp] = sqa;
    sSq2[1][warp] = sqb;
  }

  // BARRIER 2
  __syncthreads();

  float v2 = s8_exact::warp8_sum_bfly(sSq2[(lane >> 3) & 1][lane & 7]);
  float sa = __shfl_sync(0xFFFFFFFFu, v2, 0);
  float sb = __shfl_sync(0xFFFFFFFFu, v2, 8);
  float out_rstd = s8_exact::rsqrt_approx(__fadd_rn(__fdiv_rn(__fadd_rn(sa, sb), (float)D), sp.norm_eps));

  float wa0[8], wa1[8], wb[2];
  s8_exact::unpack_bf16x8(nw_r0, wa0);
  s8_exact::unpack_bf16x8(nw_r1, wa1);
  s8_exact::unpack_bf16x2(nw_rb, wb);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc0[i] = __fmul_rn(__fmul_rn(acc0[i], out_rstd), wa0[i]);
    acc1[i] = __fmul_rn(__fmul_rn(acc1[i], out_rstd), wa1[i]);
  }
  accb[0] = __fmul_rn(__fmul_rn(accb[0], out_rstd), wb[0]);
  accb[1] = __fmul_rn(__fmul_rn(accb[1], out_rstd), wb[1]);

  __nv_bfloat16* out_p = sp.out + (int64_t)q_idx * sp.T * D + row_off;
  s8_exact::store_f32x8_to_bf16(out_p + off_a0, acc0);
  s8_exact::store_f32x8_to_bf16(out_p + off_a1, acc1);
  s8_exact::store_f32x2_to_bf16(out_p + off_b, accb);
}

// ---- s6 Drafter Fused All-Reduce + Residual Add + RMSNorm (bf16 [8, 4608], world=8) ----
// Eliminates 10 fused_add_rmsnorm launches + 10x 73.7 KB DRAM round-trips per step (~55 us/step)!
// Exploits the exact geometry: rows=8 == world=8, row_vec = 4608*2/16 = 576 == chunk!
// So in 2-shot phase 2, recv2[slot][r] holds ALL 576 vectors of row r contiguously -> CTA r owns row r!
struct DraftNormParams {
  uint4* residual;      // in/out [8, 4608] bf16 (updated in-place: residual += reduced)
  const uint4* gamma;   // [4608] bf16 RMSNorm weight
  float eps;
  int hidden;           // 4608
  int rows;             // 8
};

__device__ __forceinline__ float s6_block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float out = 0.f;
  if (warp == 0) {
    out = (lane < (kThreads >> 5)) ? smem[lane] : 0.f;
#pragma unroll
    for (int m = 8; m >= 1; m >>= 1) out += __shfl_xor_sync(0xffffffffu, out, m);
    if (lane == 0) smem[0] = out;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float s6_block_max(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float out = 0.f;
  if (warp == 0) {
    out = (lane < (kThreads >> 5)) ? smem[lane] : 0.f;
#pragma unroll
    for (int m = 8; m >= 1; m >>= 1) out = fmaxf(out, __shfl_xor_sync(0xffffffffu, out, m));
    if (lane == 0) smem[0] = out;
  }
  __syncthreads();
  return smem[0];
}

template <int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_draft_norm(Params p, DraftNormParams dn) {
  // gridDim.x == 8 (1 CTA per row for rows=8, H=4608, row_vec=576 == chunk)
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  __shared__ float s_red[32];
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0 && ((kPub & kR3Sbuf) == 0 || blockIdx.x == 0)) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (threadIdx.x == 0) {
    const unsigned long long seq0 = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    s_seq = seq0;
    if constexpr ((kPub & kR6Pdl) == 0) {
      if ((kPub & kR3Sbuf) == 0 || blockIdx.x == 0) early = s8_read_early(p);
    }
    if constexpr ((kPub & kR3Sbuf) != 0) {
      if (blockIdx.x == 0) {
        __threadfence_system();
        const uint64_t src_off =
            (uint64_t)(reinterpret_cast<const char*>(p.in) - reinterpret_cast<const char*>(p.ctrl));
        r3_publish1<kTwoShot, kPub>(p, seq0, early, src_off);
      }
    }
  }
  __syncthreads();
  const unsigned long long seq = s_seq;
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }
  const int slot = static_cast<int>(seq % kSlots);
  const long long row_vec = p.chunk; // 576
  if constexpr ((kPub & kR3Sbuf) != 0) {
    if (blockIdx.x == 0 && threadIdx.x == 32) {
      uint64_t* hdr = reinterpret_cast<uint64_t*>(const_cast<uint4*>(p.in) - 1);
      const uint64_t last = ld_relaxed_sys(hdr);
      if (last != 0 && seq - last < 2) r3_trap();
      st_relaxed_sys(hdr, seq);
    }
  } else {
    uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = tid; i < p.nvec; i += stride) send4[i] = p.in[i];
    if (last_cta(&p.st->arrive, &s_last)) r3_publish1<kTwoShot, kPub>(p, seq, early, 0);
  }

  // s7 S1b: Overlapped per-peer phase-2 + CTA `p.rank` register bypass!
  //   - CTA `p.rank`: waits p1, reduces row `p.rank` into registers rv0/rv1 & send2, publishes send2_seq,
  //     SKIPS flags2 wait & __ldcv(send2) completely, and immediately runs residual+RMSNorm from registers!
  //   - CTAs `row != p.rank` (7 CTAs): each polls ONLY peer `row`'s dedicated 64B flags2 line (zero p2_done relay!),
  //     fuses residual+RMSNorm immediately overlapped with slower peers still in flight!
  const int row = blockIdx.x;  // 0..7
  const bool has1 = (threadIdx.x + blockDim.x) < row_vec;
  uint4 rv0, rv1 = make_uint4(0u, 0u, 0u, 0u);
  if (row == p.rank) {
    s6_wait_p1(p, slot, seq, row_vec * 16, false);  // same-CTA ld.acquire.sys + __syncthreads()
    const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
    const char* rbase = p.recv + (long long)slot * p.world * rstride;
    const long long b = (long long)p.rank * row_vec;
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    {
      const long long i = threadIdx.x;
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[b + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
      rv0 = packv<false>(acc);
      send2[i] = rv0;
    }
    if (has1) {
      const long long i1 = threadIdx.x + blockDim.x;
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[b + i1] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * rstride) + i1);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<false>(acc, v[r]);
      rv1 = packv<false>(acc);
      send2[i1] = rv1;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      s8_publish2(p, slot, seq, (uint64_t)row_vec * 16);
      st_release_gpu(&p.st->p1_done, seq);
    }
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
  } else {
    s7_wait_p1_gate_l2(p, seq);  // zero C2C traffic during Phase 1 (protects drafter L2PF!)
    if (threadIdx.x == 0) {
      const uint64_t* fp = p.flags2 + (slot * p.world + row) * kFlagStride;
      // s7-v3: 32ns warp-sleep on 7-SM Phase-2 polls cuts C2C snoop pressure 5-10x during drafter L2PF!
      spin_until_thin_sleep(fp, seq, p, row, 2, p.poll_sleep_ns ? p.poll_sleep_ns : 32u);
      if (atom_add_acqrel_gpu(&p.st->arrive2, 1u) == (unsigned int)(kW - 2)) {
        *reinterpret_cast<volatile unsigned int*>(&p.st->arrive2) = 0u;
        st_relaxed_gpu(&p.st->ar_done_cnt, seq);
      }
    }
    __syncthreads();
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    const uint4* red_row = reinterpret_cast<const uint4*>(r2 + (long long)row * p.max_chunk);
    rv0 = __ldcv(red_row + threadIdx.x);
    if (has1) rv1 = __ldcv(red_row + threadIdx.x + blockDim.x);
  }

  uint4* res_row = dn.residual + (long long)row * row_vec;
  uint4* out_row = p.out + (long long)row * row_vec;

  // Each thread handles up to 2 vectors (576 / 512 <= 2) -> keep unrounded fp32 sum in registers!
  float s0[8], s1[8];
  float ss = 0.f;
  {
    const uint4 xv = res_row[threadIdx.x];
    float a[8], b[8];
    r7_unpack(rv0, a);
    r7_unpack(xv, b);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      s0[k] = a[k] + b[k];
      ss += s0[k] * s0[k];
    }
    res_row[threadIdx.x] = pack8(s0);
  }
  if (has1) {
    const int i1 = threadIdx.x + blockDim.x;
    const uint4 xv = res_row[i1];
    float a[8], b[8];
    r7_unpack(rv1, a);
    r7_unpack(xv, b);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      s1[k] = a[k] + b[k];
      ss += s1[k] * s1[k];
    }
    res_row[i1] = pack8(s1);
  }
  const float inv = rsqrtf(s6_block_sum(ss, s_red) / (float)dn.hidden + dn.eps);
  {
    float g[8];
    r7_unpack(dn.gamma[threadIdx.x], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) s0[k] = s0[k] * inv * g[k];
    out_row[threadIdx.x] = pack8(s0);
  }
  if (has1) {
    const int i1 = threadIdx.x + blockDim.x;
    float g[8];
    r7_unpack(dn.gamma[i1], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) s1[k] = s1[k] * inv * g[k];
    out_row[i1] = pack8(s1);
  }
}

// ---- s6 Drafter Per-Row INT8 Quantized 1-Shot All-Reduce + Residual Add + RMSNorm ----
// Drafter ONLY (does NOT touch target exactness!). Shrinks [8, 4608] bf16 (73,728 B, 2-shot) to
// 8 * (4608 B int8 + 16 B fp32 scale header) = 36,992 B -> 1-shot (1 network hop instead of 2)!
template <int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_draft_int8_norm(Params p, DraftNormParams dn) {
  // gridDim.x == 8 (1 CTA per row), H == 4608, row_vec == 576 uint4 bf16 -> 288 uint4 int8 + 1 uint4 scale = 289 uint4/row
  constexpr long long kQRowVec = 289; // 4624 B/row; 8 rows = 2312 uint4 = 36,992 B total
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  __shared__ float s_red[32];
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (threadIdx.x == 0) {
    s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    if constexpr ((kPub & kR6Pdl) == 0) early = s8_read_early(p);
  }
  __syncthreads();
  const unsigned long long seq = s_seq;
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }
  const int slot = static_cast<int>(seq % kSlots);
  const int row = blockIdx.x;
  const long long row_vec = 576; // 4608 bf16 / 8
  const uint4* in_row = p.in + (long long)row * row_vec;

  // 1) Quantize row to per-row symmetric INT8 into send[slot][row * 289]
  float x0[8], x1[8];
  const bool has1 = (threadIdx.x + blockDim.x) < row_vec;
  float amax = 0.f;
  r7_unpack(in_row[threadIdx.x], x0);
#pragma unroll
  for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(x0[k]));
  if (has1) {
    r7_unpack(in_row[threadIdx.x + blockDim.x], x1);
#pragma unroll
    for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(x1[k]));
  }
  const float row_amax = s6_block_max(amax, s_red);
  const float scale = (row_amax > 0.f) ? (row_amax / 127.0f) : 1.0f;
  const float inv_scale = (row_amax > 0.f) ? (127.0f / row_amax) : 0.0f;

  // Pack 8 int8 values per thread (2x uint32 = 8 bytes) directly into send4
  int8_t* q_row = reinterpret_cast<int8_t*>(p.send + slot * p.max_bytes) + (long long)row * (kQRowVec * 16);
  auto pack_i8_4 = [&](float a, float b, float c, float d) -> uint32_t {
    uint32_t qa = (uint32_t)(uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(a * inv_scale, -127.f), 127.f));
    uint32_t qb = (uint32_t)(uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(b * inv_scale, -127.f), 127.f));
    uint32_t qc = (uint32_t)(uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(c * inv_scale, -127.f), 127.f));
    uint32_t qd = (uint32_t)(uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(d * inv_scale, -127.f), 127.f));
    return qa | (qb << 8) | (qc << 16) | (qd << 24);
  };
  const uint2 q_self0 = make_uint2(pack_i8_4(x0[0], x0[1], x0[2], x0[3]), pack_i8_4(x0[4], x0[5], x0[6], x0[7]));
  reinterpret_cast<uint2*>(q_row)[threadIdx.x] = q_self0;
  uint2 q_self1 = make_uint2(0u, 0u);
  if (has1) {
    q_self1 = make_uint2(pack_i8_4(x1[0], x1[1], x1[2], x1[3]), pack_i8_4(x1[4], x1[5], x1[6], x1[7]));
    reinterpret_cast<uint2*>(q_row)[threadIdx.x + blockDim.x] = q_self1;
  }
  if (threadIdx.x == 0) {
    *reinterpret_cast<float*>(q_row + 4608) = scale;
  }
  if (last_cta(&p.st->arrive, &s_last)) {
    Params pq = p;
    pq.nvec = kQRowVec * 8; // 2312 uint4 = 36,992 B
    pq.chunk = pq.nvec;
    r3_publish1<kOneShot, kPub & ~kR3Sbuf>(pq, seq, early, 0);
  }

  // 2) Wait for 1-shot peer flags (single hop!)
  s6_wait_p1(p, slot, seq, kQRowVec * 8 * 16, true);
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();

  // 3) Dequantize + reduce 8 ranks + residual add + RMSNorm in registers!
  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const char* rbase = p.recv + (long long)slot * p.world * rstride;
  __shared__ float s_sc[kW];
  if (threadIdx.x < kW) {
    if (threadIdx.x == p.rank) {
      s_sc[threadIdx.x] = scale;
    } else {
      const float* sp = reinterpret_cast<const float*>(rbase + (long long)threadIdx.x * rstride +
                                                       (long long)row * (kQRowVec * 16) + 4608);
      s_sc[threadIdx.x] = __ldcv(sp);
    }
  }
  __syncthreads();

  auto reduce_vec = [&](int vi, float* out_f, uint2 q_self) {
#pragma unroll
    for (int k = 0; k < 8; ++k) out_f[k] = 0.f;
    uint2 qv[kW];
#pragma unroll
    for (int r = 0; r < kW; ++r) {
      if (r == p.rank) {
        qv[r] = q_self;
      } else {
        const uint2* qp = reinterpret_cast<const uint2*>(rbase + (long long)r * rstride +
                                                         (long long)row * (kQRowVec * 16)) + vi;
        qv[r] = __ldcv(qp);
      }
    }
#pragma unroll
    for (int r = 0; r < kW; ++r) {
      const float sc = s_sc[r];
      const uint32_t w0 = qv[r].x, w1 = qv[r].y;
      out_f[0] += (float)(int8_t)(w0 & 0xffu) * sc;
      out_f[1] += (float)(int8_t)((w0 >> 8) & 0xffu) * sc;
      out_f[2] += (float)(int8_t)((w0 >> 16) & 0xffu) * sc;
      out_f[3] += (float)(int8_t)((w0 >> 24) & 0xffu) * sc;
      out_f[4] += (float)(int8_t)(w1 & 0xffu) * sc;
      out_f[5] += (float)(int8_t)((w1 >> 8) & 0xffu) * sc;
      out_f[6] += (float)(int8_t)((w1 >> 16) & 0xffu) * sc;
      out_f[7] += (float)(int8_t)((w1 >> 24) & 0xffu) * sc;
    }
  };

  uint4* res_row = dn.residual + (long long)row * row_vec;
  uint4* out_row = p.out + (long long)row * row_vec;
  float s0[8], s1[8], r0[8], r1[8];
  float ss = 0.f;
  reduce_vec(threadIdx.x, s0, q_self0);
  r7_unpack(res_row[threadIdx.x], r0);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    s0[k] += r0[k];
    ss += s0[k] * s0[k];
  }
  res_row[threadIdx.x] = pack8(s0);
  if (has1) {
    const int i1 = threadIdx.x + blockDim.x;
    reduce_vec(i1, s1, q_self1);
    r7_unpack(res_row[i1], r1);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      s1[k] += r1[k];
      ss += s1[k] * s1[k];
    }
    res_row[i1] = pack8(s1);
  }
  const float inv = rsqrtf(s6_block_sum(ss, s_red) / (float)dn.hidden + dn.eps);
  {
    float g[8];
    r7_unpack(dn.gamma[threadIdx.x], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) s0[k] = s0[k] * inv * g[k];
    out_row[threadIdx.x] = pack8(s0);
  }
  if (has1) {
    const int i1 = threadIdx.x + blockDim.x;
    float g[8];
    r7_unpack(dn.gamma[i1], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) s1[k] = s1[k] * inv * g[k];
    out_row[i1] = pack8(s1);
  }
}

template <int kPub>
void launch_draft_norm_one(const Params& p, const DraftNormParams& dn, bool use_int8, cudaStream_t stream) {
  if constexpr ((kPub & kR6Pdl) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(8u);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (use_int8)
      CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_draft_int8_norm<8, kPub>, p, dn));
    else
      CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_draft_norm<8, kPub>, p, dn));
  } else {
    if (use_int8)
      ar_kernel_draft_int8_norm<8, kPub><<<8, kThreads, 0, stream>>>(p, dn);
    else
      ar_kernel_draft_norm<8, kPub><<<8, kThreads, 0, stream>>>(p, dn);
  }
}

void launch_draft_norm_kernel(const Params& p, const DraftNormParams& dn, bool use_int8, cudaStream_t stream, int pub) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  const int m = pub & (B | kR6Pdl | kR6Trig | (use_int8 ? 0 : kR3Sbuf));
  switch (m) {
    case B: launch_draft_norm_one<B>(p, dn, use_int8, stream); break;
    case B | kR6Pdl: launch_draft_norm_one<B | kR6Pdl>(p, dn, use_int8, stream); break;
    case B | kR6Pdl | kR6Trig: launch_draft_norm_one<B | kR6Pdl | kR6Trig>(p, dn, use_int8, stream); break;
    case B | kR3Sbuf: launch_draft_norm_one<B | kR3Sbuf>(p, dn, use_int8, stream); break;
    case B | kR6Pdl | kR3Sbuf: launch_draft_norm_one<B | kR6Pdl | kR3Sbuf>(p, dn, use_int8, stream); break;
    case B | kR6Pdl | kR6Trig | kR3Sbuf: launch_draft_norm_one<B | kR6Pdl | kR6Trig | kR3Sbuf>(p, dn, use_int8, stream); break;
    default: throw std::runtime_error("rdma_ar s6: unsupported draft_norm kernel mask " + std::to_string(m));
  }
}

// ============================================================================
// s8v2-unified (dec160-drafter-speed):
//   1. ar_kernel_draft_h64_q_norm<kW, kPub, kBits, kOutH64>:
//      Register-only orthonormal H_64 + 4/6/8-bit 1-shot drafter AR+RMSNorm
//      (19,584 B / 28,800 B / 38,016 B), strict fixed-order r=0..7 reduction
//      (100% cross-rank bit-deterministic), s8_read_early/RDMA_AR_LEVEL_DB safe,
//      ar_done_cnt L2PF stop-flag fix, optional kOutH64 pre-rotated output.
//   2. markov_loop_persistent_kernel<kW, kPub, kHasIdMap>:
//      Single 48-CTA cooperative persistent kernel fusing B=1 7-step Markov RNN
//      + SM121a FP8 fold/argmax + 16B inline RDMA AG (s8_publish1_inline) +
//      coupled Dinkelbach early-stop (immediate hardware break, 0 us skipped!)
//      + in-graph _vw_conf/req_drafts epilogue.
//   3. ar_kernel_inline_ag_select_dink<kW, kPub>:
//      Standalone 1-warp non-cooperative drop-in fusing inline AG + Dink select.
// Target verifier kernels above are 100% UNTOUCHED (0-ULP bit-exact guarantee).
// ============================================================================

__device__ __forceinline__ void dec160_fwht64_8t(float v[8]) {
  // Stage 1: local 8-point FWHT in registers (distances 1, 2, 4)
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const float a = v[i], b = v[i + 1];
    v[i] = a + b;
    v[i + 1] = a - b;
  }
#pragma unroll
  for (int i = 0; i < 8; i += 4) {
    const float a0 = v[i], b0 = v[i + 2];
    const float a1 = v[i + 1], b1 = v[i + 3];
    v[i] = a0 + b0;     v[i + 2] = a0 - b0;
    v[i + 1] = a1 + b1; v[i + 3] = a1 - b1;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float a = v[i], b = v[i + 4];
    v[i] = a + b;
    v[i + 4] = a - b;
  }
  // Stage 2: cross-lane 8-point FWHT across 8 consecutive warp lanes
  const int sub = (int)(threadIdx.x & 7u);
#pragma unroll
  for (int step = 1; step <= 4; step <<= 1) {
    const float neg = (sub & step) ? -1.0f : 1.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float p = __shfl_xor_sync(0xffffffffu, v[k], step);
      v[k] = (neg < 0.0f) ? (p - v[k]) : (v[k] + p);
    }
  }
  // Orthonormal scale 1/sqrt(64) = 0.125f
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] *= 0.125f;
}

__device__ __forceinline__ float dec160_subwarp8_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  return v;
}

template <int kW, int kPub, int kBits, bool kOutH64 = false>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_draft_h64_q_norm(Params p, DraftNormParams dn) {
  static_assert(kBits == 4 || kBits == 6 || kBits == 8, "kBits must be 4, 6, or 8");
  static_assert(kThreads == 512, "expects kThreads == 512");
  constexpr long long kRowBytes = (kBits == 4) ? 2448LL : ((kBits == 6) ? 3600LL : 4752LL);
  constexpr long long kRowVecQ  = kRowBytes / 16LL;  // 153, 225, or 297 uint4/row
  constexpr long long kScaleOff = (kBits == 4) ? 2304LL : ((kBits == 6) ? 3456LL : 4608LL);
  constexpr long long kHi2Off   = 2304LL;            // used when kBits == 6

  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  __shared__ float s_red[32];

  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (threadIdx.x == 0) {
    s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1ULL;
    if constexpr ((kPub & kR6Pdl) == 0) early = s8_read_early(p);
  }
  __syncthreads();
  const unsigned long long seq = s_seq;
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }

  const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
  const int row = (int)blockIdx.x;  // 0..7
  constexpr long long kRowVecBf16 = 576LL;
  const uint4* in_row = p.in + (long long)row * kRowVecBf16;
  char* q_row = p.send + (long long)slot * p.max_bytes + (long long)row * kRowBytes;
  const bool has1 = (threadIdx.x < 64u);  // 2 full warps = 8 full 64-elem groups

  uint32_t q4_self0 = 0u, q4_self1 = 0u;
  uint16_t h2_self0 = 0u, h2_self1 = 0u;
  uint2    q8_self0 = make_uint2(0u, 0u), q8_self1 = make_uint2(0u, 0u);
  float    sc_self0 = 0.f, sc_self1 = 0.f;

  auto quantize_pass = [&](int vi, uint32_t& out_q4, uint16_t& out_h2, uint2& out_q8, float& out_sc) {
    float x[8];
    r7_unpack(in_row[vi], x);
    dec160_fwht64_8t(x);

    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(x[k]));
    const float g_amax = dec160_subwarp8_max(amax);

    constexpr float kClip = (kBits == 4) ? (0.90f / 7.5f) : ((kBits == 6) ? (0.96f / 31.5f) : (1.0f / 127.0f));
    const __half h_sc = __float2half_rn((g_amax > 0.f) ? (g_amax * kClip) : 0.f);
    const float sc = __half2float(h_sc);
    const float inv_sc = (sc > 0.f) ? (1.0f / sc) : 0.f;
    out_sc = sc;

    if ((threadIdx.x & 7u) == 0u) {
      reinterpret_cast<__half*>(q_row + kScaleOff)[vi >> 3] = h_sc;
    }
    if constexpr (kBits == 4) {
      uint32_t pack = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const uint32_t q = (uint32_t)__float2int_rn(fminf(fmaxf(x[k] * inv_sc + 7.5f, 0.f), 15.f));
        pack |= (q << (k * 4));
      }
      out_q4 = pack;
      reinterpret_cast<uint32_t*>(q_row)[vi] = pack;
    } else if constexpr (kBits == 6) {
      uint32_t lo4 = 0u;
      uint32_t hi2 = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const uint32_t q = (uint32_t)__float2int_rn(fminf(fmaxf(x[k] * inv_sc + 31.5f, 0.f), 63.f));
        lo4 |= ((q & 0xfu) << (k * 4));
        hi2 |= (((q >> 4) & 0x3u) << (k * 2));
      }
      out_q4 = lo4;
      out_h2 = (uint16_t)hi2;
      reinterpret_cast<uint32_t*>(q_row)[vi] = lo4;
      reinterpret_cast<uint16_t*>(q_row + kHi2Off)[vi] = (uint16_t)hi2;
    } else {
      uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t q0 = (uint32_t)(uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(x[k] * inv_sc, -127.f), 127.f));
        const uint32_t q1 = (uint32_t)(uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(x[k + 4] * inv_sc, -127.f), 127.f));
        w0 |= (q0 << (k * 8));
        w1 |= (q1 << (k * 8));
      }
      out_q8 = make_uint2(w0, w1);
      reinterpret_cast<uint2*>(q_row)[vi] = out_q8;
    }
  };

  quantize_pass((int)threadIdx.x, q4_self0, h2_self0, q8_self0, sc_self0);
  if (has1) {
    quantize_pass((int)threadIdx.x + 512, q4_self1, h2_self1, q8_self1, sc_self1);
  }

  if (last_cta(&p.st->arrive, &s_last)) {
    Params pq = p;
    pq.nvec = kRowVecQ * 8LL;   // 1224 (19.6 KB), 1800 (28.8 KB), or 2376 (38.0 KB)
    pq.chunk = pq.nvec;
    r3_publish1<kOneShot, kPub & ~kR3Sbuf>(pq, seq, early, 0);
  }

  s6_wait_p1(p, slot, seq, kRowBytes * 8LL, true);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    st_relaxed_gpu(&p.st->ar_done_cnt, seq);
  }
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();

  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const char* rbase = p.recv + (long long)slot * (long long)p.world * rstride;
  auto reduce_pass = [&](int vi, uint32_t q4_s, uint16_t h2_s, uint2 q8_s, float sc_s, float out_f[8]) {
    const int g = vi >> 3;
#pragma unroll
    for (int k = 0; k < 8; ++k) out_f[k] = 0.f;

    float peer_sc = sc_s;
    const int sub = (int)(threadIdx.x & 7u);
    if (sub != p.rank) {
      const uint16_t* sp = reinterpret_cast<const uint16_t*>(
          rbase + (long long)sub * rstride + (long long)row * kRowBytes + kScaleOff) + g;
      const uint16_t raw_h = __ldcv(sp);
      peer_sc = __half2float(*reinterpret_cast<const __half*>(&raw_h));
    }

#pragma unroll
    for (int r = 0; r < kW; ++r) {
      const float sc = __shfl_sync(0xffffffffu, peer_sc, (((int)threadIdx.x & 31) & ~7) | r);
      const char* rp = rbase + (long long)r * rstride + (long long)row * kRowBytes;
      if constexpr (kBits == 4) {
        const uint32_t pk = (r == p.rank) ? q4_s : __ldcv(reinterpret_cast<const uint32_t*>(rp) + vi);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float q = (float)((pk >> (k * 4)) & 0xfu) - 7.5f;
          out_f[k] = fmaf(q, sc, out_f[k]);
        }
      } else if constexpr (kBits == 6) {
        const uint32_t lo = (r == p.rank) ? q4_s : __ldcv(reinterpret_cast<const uint32_t*>(rp) + vi);
        const uint32_t hi = (r == p.rank) ? (uint32_t)h2_s
                                          : (uint32_t)__ldcv(reinterpret_cast<const uint16_t*>(rp + kHi2Off) + vi);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const uint32_t q6 = ((lo >> (k * 4)) & 0xfu) | (((hi >> (k * 2)) & 0x3u) << 4);
          out_f[k] = fmaf((float)q6 - 31.5f, sc, out_f[k]);
        }
      } else {
        const uint2 q8 = (r == p.rank) ? q8_s : __ldcv(reinterpret_cast<const uint2*>(rp) + vi);
        const uint32_t w0 = q8.x, w1 = q8.y;
        out_f[0] = fmaf((float)(int8_t)(w0 & 0xffu), sc, out_f[0]);
        out_f[1] = fmaf((float)(int8_t)((w0 >> 8) & 0xffu), sc, out_f[1]);
        out_f[2] = fmaf((float)(int8_t)((w0 >> 16) & 0xffu), sc, out_f[2]);
        out_f[3] = fmaf((float)(int8_t)((w0 >> 24) & 0xffu), sc, out_f[3]);
        out_f[4] = fmaf((float)(int8_t)(w1 & 0xffu), sc, out_f[4]);
        out_f[5] = fmaf((float)(int8_t)((w1 >> 8) & 0xffu), sc, out_f[5]);
        out_f[6] = fmaf((float)(int8_t)((w1 >> 16) & 0xffu), sc, out_f[6]);
        out_f[7] = fmaf((float)(int8_t)((w1 >> 24) & 0xffu), sc, out_f[7]);
      }
    }
    dec160_fwht64_8t(out_f);
  };

  uint4* res_row = dn.residual + (long long)row * kRowVecBf16;
  uint4* out_row = p.out + (long long)row * kRowVecBf16;
  float s0[8], s1[8], r0[8], r1[8];
  float ss = 0.f;

  reduce_pass((int)threadIdx.x, q4_self0, h2_self0, q8_self0, sc_self0, s0);
  r7_unpack(res_row[threadIdx.x], r0);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    s0[k] += r0[k];
    ss += s0[k] * s0[k];
  }
  res_row[threadIdx.x] = pack8(s0);

  if (has1) {
    const int i1 = (int)threadIdx.x + 512;
    reduce_pass(i1, q4_self1, h2_self1, q8_self1, sc_self1, s1);
    r7_unpack(res_row[i1], r1);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      s1[k] += r1[k];
      ss += s1[k] * s1[k];
    }
    res_row[i1] = pack8(s1);
  }

  const float inv = rsqrtf(s6_block_sum(ss, s_red) / (float)dn.hidden + dn.eps);
  {
    float g[8];
    r7_unpack(dn.gamma[threadIdx.x], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) s0[k] = s0[k] * inv * g[k];
    if constexpr (kOutH64) dec160_fwht64_8t(s0);
    out_row[threadIdx.x] = pack8(s0);
  }
  if (has1) {
    const int i1 = (int)threadIdx.x + 512;
    float g[8];
    r7_unpack(dn.gamma[i1], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) s1[k] = s1[k] * inv * g[k];
    if constexpr (kOutH64) dec160_fwht64_8t(s1);
    out_row[i1] = pack8(s1);
  }
}

template <int kPub, int kBits, bool kOutH64>
void launch_draft_h64_one(const Params& p, const DraftNormParams& dn, cudaStream_t stream) {
  if constexpr ((kPub & kR6Pdl) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(8u);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_draft_h64_q_norm<8, kPub, kBits, kOutH64>, p, dn));
  } else {
    ar_kernel_draft_h64_q_norm<8, kPub, kBits, kOutH64><<<8, kThreads, 0, stream>>>(p, dn);
  }
}

template <int kBits, bool kOutH64>
void launch_draft_h64_pub(const Params& p, const DraftNormParams& dn, cudaStream_t stream, int pub) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  const int m = pub & (B | kR6Pdl | kR6Trig);
  switch (m) {
    case B: launch_draft_h64_one<B, kBits, kOutH64>(p, dn, stream); break;
    case B | kR6Pdl: launch_draft_h64_one<B | kR6Pdl, kBits, kOutH64>(p, dn, stream); break;
    case B | kR6Pdl | kR6Trig: launch_draft_h64_one<B | kR6Pdl | kR6Trig, kBits, kOutH64>(p, dn, stream); break;
    default: throw std::runtime_error("rdma_ar s8v2: unsupported draft_h64 mask " + std::to_string(m));
  }
}

void launch_draft_norm_v2_kernel(const Params& p, const DraftNormParams& dn, int mode_code, bool out_h64,
                                 cudaStream_t stream, int pub) {
  // mode_code: 0=bf16, 1=int8, 2=h64_q8, 3=h64_q6, 4=h64_q4
  if (mode_code == 4) {
    if (out_h64) launch_draft_h64_pub<4, true>(p, dn, stream, pub);
    else launch_draft_h64_pub<4, false>(p, dn, stream, pub);
  } else if (mode_code == 3) {
    if (out_h64) launch_draft_h64_pub<6, true>(p, dn, stream, pub);
    else launch_draft_h64_pub<6, false>(p, dn, stream, pub);
  } else if (mode_code == 2) {
    if (out_h64) launch_draft_h64_pub<8, true>(p, dn, stream, pub);
    else launch_draft_h64_pub<8, false>(p, dn, stream, pub);
  } else {
    launch_draft_norm_kernel(p, dn, mode_code == 1, stream, pub);
  }
}

// ---- 2. Persistent 48-CTA 1-Kernel Markov Loop + Inline RDMA AG + Dink Early-Stop ----
struct MarkovPersistentParams {
  const __nv_bfloat16* base_logits;   // [7, base_stride] bf16
  const float*         rnn_ctx;       // [7, 768] fp32
  const __nv_bfloat16* w1;            // [padded_vocab, 256] bf16
  const __nv_bfloat16* rnn_w_rec;     // [768, 512] bf16
  const uint8_t*       fold_w8;       // [valid_vocab, 256] fp8 e4m3fn
  const float*         fold_s8;       // [valid_vocab] fp32
  const __nv_bfloat16* fold_bias;     // [valid_vocab] bf16
  const int32_t*       id_map;        // [valid_vocab] int32 global IDs (nullable)
  const void*          anchor_id_ptr; // &input_ids[0] (int32 or int64)
  int64_t              base_stride;
  float                softcap;       // 30.0f
  int32_t              valid_vocab;   // 25128 (full) or V_local (pruned)
  int32_t              vocab_offset;  // rank * 25128
  int32_t              k_steps;       // 7
  int32_t              anchor_is_i64; // 1 if int64_t*, else int32_t*

  // Static captured Dink buffers (_es_bufs[4..7], written pre-graph by _vw_update_dink/es_init_dink!)
  const int32_t*       greedy_dev;    // [1] int32
  const int32_t*       slot_dev;      // [1] int32
  const float*         db_dev;        // [1] fp32
  const float*         lam_dev;       // [1] fp32
  const float*         step_eff_ptr;  // [8] fp32
  const float*         cal_ptr;       // [24] fp32
  int32_t*             vw_w_ptr;      // [max_num_reqs] int32
  float*               vw_cpred_ptr;  // [max_num_reqs, 7] fp32
  int64_t*             draft_out;     // [7] int64 draft_tokens[0, 0..6]
  float*               conf_out;      // [7] fp32 conf_buf[0, 0..6] (nullable)
  int64_t*             req_drafts;    // [max_num_reqs, req_drafts_stride] int64 (nullable; INKLING_GRAPH_VW_CONF)
  int64_t              req_drafts_stride;
  const void*          sample_idx0_ptr; // &sample_idx_mapping[0] (<0 on dummy_run -> skip state writes!)
  int32_t              sample_idx_is_i64;
  int32_t              w_cap;         // 7

  // Persistent GPU-L2 scratch (~4 KB total, pre-allocated outside capture!)
  __nv_bfloat16*       scratch_state;   // [2][256] bf16
  __nv_bfloat16*       scratch_latent;  // [256] bf16
  float4*              scratch_rec;     // [64] float4
  int64_t*             scratch_ctrl;    // [4]: [0]=prev_tok, [1]=stop_flag, [2]=bar_cnt, [3]=bar_gen
};

__device__ __forceinline__ float dec160_mk_round_bf16(float x) {
  uint32_t bits = __float_as_uint(x);
  bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
  return __uint_as_float(bits);
}

__device__ __forceinline__ void dec160_mk_grid_barrier(int64_t* ctrl, int n_ctas, unsigned long long& gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned int old = atomicAdd(reinterpret_cast<unsigned int*>(&ctrl[2]), 1u);
    if (old == (unsigned int)(n_ctas - 1)) {
      *reinterpret_cast<volatile unsigned int*>(&ctrl[2]) = 0u;
      __threadfence();
      st_release_gpu(reinterpret_cast<unsigned long long*>(&ctrl[3]), gen + 1ULL);
    } else {
      while (ld_acquire_gpu(reinterpret_cast<const unsigned long long*>(&ctrl[3])) < gen + 1ULL) {
      }
    }
  }
  __syncthreads();
  gen += 1ULL;
}

__device__ __forceinline__ float2 dec160_e4m3x2_to_f32x2(uint16_t packed2) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(packed2));
  const __half2 hh = *reinterpret_cast<const __half2*>(&h2);
  return __half22float2(hh);
}

template <int kW, int kPub, bool kHasIdMap>
__global__ void __launch_bounds__(256, 1) markov_loop_persistent_kernel(Params p, MarkovPersistentParams mp) {
  constexpr int kRank = 256;
  constexpr int kWarps = 8;
  const int bid = (int)blockIdx.x;
  const int n_ctas = (int)gridDim.x;
  const int tid = (int)threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ float s_lat[kRank];
  __shared__ float s_m[kWarps];
  __shared__ int32_t s_idx[kWarps];
  __shared__ float s_se[kWarps];

  int32_t d_slot = 0, d_greedy = 0, d_best_w = 0, d_skip_epilogue = 0;
  float d_db = 0.f, d_lam = 0.14f, d_surv = 1.f, d_tok_acc = 1.f, d_best_u = 0.f;
  float d_cpred[8];
  int64_t d_draft_local[8];

  unsigned long long bar_gen =
      *reinterpret_cast<const volatile unsigned long long*>(&mp.scratch_ctrl[3]);

  // Prologue: read static captured _es_bufs[4..7]
  if (bid == 0 && tid == 0) {
    int64_t anchor = mp.anchor_is_i64
        ? reinterpret_cast<const int64_t*>(mp.anchor_id_ptr)[0]
        : (int64_t)reinterpret_cast<const int32_t*>(mp.anchor_id_ptr)[0];
    if (anchor < 0) anchor = 0;
    mp.scratch_ctrl[0] = anchor;
    mp.scratch_ctrl[1] = 0;

    d_greedy = mp.greedy_dev[0];
    d_slot = mp.slot_dev[0];
    d_db = mp.db_dev[0];
    d_lam = mp.lam_dev[0];
    if (mp.sample_idx0_ptr != nullptr) {
      const int64_t s0 = mp.sample_idx_is_i64
          ? reinterpret_cast<const int64_t*>(mp.sample_idx0_ptr)[0]
          : (int64_t)reinterpret_cast<const int32_t*>(mp.sample_idx0_ptr)[0];
      if (s0 < 0) {
        d_skip_epilogue = 1;
        d_greedy = 0;  // Dummy capture run: always execute all 7 steps cleanly!
      }
    }
    d_surv = 1.0f;
    d_tok_acc = 1.0f;
    d_best_u = 1.0f - d_lam * mp.step_eff_ptr[0];
    d_best_w = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      d_cpred[i] = 0.0f;
      d_draft_local[i] = -1;
    }
  }
  dec160_mk_grid_barrier(mp.scratch_ctrl, n_ctas, bar_gen);

  for (int k = 0; k < mp.k_steps; ++k) {
    // Stage 1: RNN step on CTAs 0..31 (32 CTAs x 8 warps = 256 output dims)
    if (bid < 32) {
      const int c = bid * 8 + warp;
      const int64_t prev_tok = mp.scratch_ctrl[0];
      const __nv_bfloat16* st_in = mp.scratch_state + ((k & 1) ? kRank : 0);
      __nv_bfloat16* st_out      = mp.scratch_state + ((k & 1) ? 0 : kRank);
      const float* ctx_k = mp.rnn_ctx + (long long)k * (3LL * kRank);

      const int col0 = lane * 8;
      float emb[8], st_v[8];
      r7_unpack(*reinterpret_cast<const uint4*>(mp.w1 + (long long)prev_tok * kRank + col0), emb);
      if (k > 0) {
        r7_unpack(*reinterpret_cast<const uint4*>(st_in + col0), st_v);
      }

      constexpr long long kStrideW = 2LL * kRank;
      auto dot_row = [&](int r_idx) -> float {
        const __nv_bfloat16* wr = mp.rnn_w_rec + (long long)r_idx * kStrideW;
        float w1v[8], sum = 0.f;
        r7_unpack(*reinterpret_cast<const uint4*>(wr + kRank + col0), w1v);
#pragma unroll
        for (int j = 0; j < 8; ++j) sum = fmaf(w1v[j], emb[j], sum);
        if (k > 0) {
          float w0v[8];
          r7_unpack(*reinterpret_cast<const uint4*>(wr + col0), w0v);
#pragma unroll
          for (int j = 0; j < 8; ++j) sum = fmaf(w0v[j], st_v[j], sum);
        }
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, m);
        return sum;
      };

      const float dg = dot_row(c);
      const float dc = dot_row(kRank + c);
      const float do_ = dot_row(2 * kRank + c);

      if (lane == 0) {
        const float raw_g = dec160_mk_round_bf16(ctx_k[c] + dg);
        const float raw_c = dec160_mk_round_bf16(ctx_k[kRank + c] + dc);
        const float raw_o = dec160_mk_round_bf16(ctx_k[2 * kRank + c] + do_);
        const float gate  = dec160_mk_round_bf16(__fdividef(1.0f, 1.0f + __expf(-raw_g)));
        const float cand  = dec160_mk_round_bf16(tanhf(raw_c));
        const float upd   = dec160_mk_round_bf16(dec160_mk_round_bf16(1.0f - gate) * cand);
        float new_st = upd;
        if (k > 0) {
          const float old_s = __bfloat162float(st_in[c]);
          new_st = dec160_mk_round_bf16(gate * old_s) + upd;
        }
        st_out[c] = __float2bfloat16_rn(new_st);
        mp.scratch_latent[c] = __float2bfloat16_rn(tanhf(raw_o));
      }
    }
    dec160_mk_grid_barrier(mp.scratch_ctrl, n_ctas, bar_gen);

    // Stage 2: SM121a FP8 fold + base_logits + softcap + local argmax across ALL n_ctas
    s_lat[tid] = __bfloat162float(mp.scratch_latent[tid]);
    __syncthreads();

    float lreg[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) lreg[j] = s_lat[lane * 8 + j];

    const __nv_bfloat16* base_k = mp.base_logits + (long long)k * (long long)mp.base_stride;
    const float inv_cap = 1.0f / mp.softcap;

    const int chunk = (mp.valid_vocab + n_ctas - 1) / n_ctas;
    const int row_lo = bid * chunk;
    const int row_hi = min(row_lo + chunk, (int)mp.valid_vocab);

    float wm = -INFINITY;
    int32_t widx = 2147483647;
    float wse = 0.0f;

    for (int r = row_lo + warp; r < row_hi; r += kWarps) {
      const uint2 w8 = *reinterpret_cast<const uint2*>(mp.fold_w8 + (long long)r * kRank + lane * 8);
      const float2 f0 = dec160_e4m3x2_to_f32x2((uint16_t)(w8.x & 0xffffu));
      const float2 f1 = dec160_e4m3x2_to_f32x2((uint16_t)(w8.x >> 16));
      const float2 f2 = dec160_e4m3x2_to_f32x2((uint16_t)(w8.y & 0xffffu));
      const float2 f3 = dec160_e4m3x2_to_f32x2((uint16_t)(w8.y >> 16));
      float dot = f0.x * lreg[0] + f0.y * lreg[1] + f1.x * lreg[2] + f1.y * lreg[3]
                + f2.x * lreg[4] + f2.y * lreg[5] + f3.x * lreg[6] + f3.y * lreg[7];
#pragma unroll
      for (int m = 16; m >= 1; m >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, m);

      if (lane == 0) {
        const float b = dec160_mk_round_bf16(dot * mp.fold_s8[r] + __bfloat162float(mp.fold_bias[r]));
        const float a = __bfloat162float(base_k[r]);
        float x = dec160_mk_round_bf16(a + b);
        x = dec160_mk_round_bf16(tanhf(x * inv_cap) * mp.softcap);
        if (isnan(x)) x = -INFINITY;
        const float m_new = fmaxf(wm, x);
        const float m_safe = isinf(m_new) ? 0.0f : m_new;
        const float ex = isinf(x) ? 0.0f : __expf(x - m_safe);
        const float sc_old = isinf(wm) ? 0.0f : __expf(wm - m_safe);
        wse = wse * sc_old + ex;
        if (x > wm || (x == wm && r < widx)) widx = r;
        wm = m_new;
      }
    }

    if (lane == 0) {
      s_m[warp] = wm;
      s_idx[warp] = widx;
      s_se[warp] = wse;
    }
    __syncthreads();

    if (warp == 0) {
      const float vm = (lane < kWarps) ? s_m[lane] : -INFINITY;
      const int32_t vi = (lane < kWarps) ? s_idx[lane] : 2147483647;
      const float vse = (lane < kWarps) ? s_se[lane] : 0.0f;
      float bm = vm;
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, m));
      int32_t bi = (vm == bm) ? vi : 2147483647;
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) bi = min(bi, __shfl_xor_sync(0xffffffffu, bi, m));
      const float bm0 = isinf(bm) ? 0.0f : bm;
      float bse = isinf(vm) ? 0.0f : (vse * __expf(vm - bm0));
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) bse += __shfl_xor_sync(0xffffffffu, bse, m);
      if (lane == 0) {
        float out_id;
        if constexpr (kHasIdMap) {
          const int32_t safe_i = (bi < mp.valid_vocab) ? bi : 0;
          out_id = (float)mp.id_map[safe_i];
        } else {
          out_id = (float)(bi + mp.vocab_offset);
        }
        mp.scratch_rec[bid] = make_float4(bm, out_id, bse, 0.0f);
      }
    }
    dec160_mk_grid_barrier(mp.scratch_ctrl, n_ctas, bar_gen);

    // Stage 3: CTA 0 warp 0 collapses n_ctas records + 16B Inline RDMA AG (s8_publish1_inline!) + Dink Early-Stop
    if (bid == 0 && warp == 0) {
      float lm = -INFINITY, lid = 0.0f, lse = 0.0f;
      for (int s = lane; s < n_ctas; s += 32) {
        const float4 rec = mp.scratch_rec[s];
        const float m_new = fmaxf(lm, rec.x);
        const float ms = isinf(m_new) ? 0.0f : m_new;
        const float e1 = isinf(rec.x) ? 0.0f : rec.z * __expf(rec.x - ms);
        const float e0 = isinf(lm) ? 0.0f : lse * __expf(lm - ms);
        if (rec.x > lm || (rec.x == lm && rec.y < lid)) lid = rec.y;
        lse = e0 + e1;
        lm = m_new;
      }
      float r_m = lm;
#pragma unroll
      for (int m = 16; m >= 1; m >>= 1) r_m = fmaxf(r_m, __shfl_xor_sync(0xffffffffu, r_m, m));
      int32_t r_id_i = (lm == r_m) ? (int32_t)lid : 2147483647;
#pragma unroll
      for (int m = 16; m >= 1; m >>= 1) r_id_i = min(r_id_i, __shfl_xor_sync(0xffffffffu, r_id_i, m));
      const float r_id = (float)r_id_i;
      const float r_m0 = isinf(r_m) ? 0.0f : r_m;
      float r_se = isinf(lm) ? 0.0f : lse * __expf(lm - r_m0);
#pragma unroll
      for (int m = 16; m >= 1; m >>= 1) r_se += __shfl_xor_sync(0xffffffffu, r_se, m);

      const uint4 v_self = make_uint4(__float_as_uint(r_m), __float_as_uint(r_id), __float_as_uint(r_se), 0u);
      unsigned long long seq = 0ULL;
      if (lane == 0) {
        seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1ULL;
        *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
        const uint64_t early = s8_read_early(p);
        s8_publish1_inline<kPub>(p, static_cast<int>(seq % (unsigned long long)kSlots), seq, early,
                                 ((uint64_t)v_self.y << 32) | (uint64_t)v_self.x,
                                 ((uint64_t)v_self.w << 32) | (uint64_t)v_self.z);
      }
      seq = __shfl_sync(0xffffffffu, seq, 0);
      const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
      uint4 v_r = make_uint4(__float_as_uint(-INFINITY), 0u, 0u, 0u);
      if (lane < kW) {
        if (lane == p.rank) {
          v_r = v_self;
        } else {
          const InlineSlot* sl = p.inline_recv + slot * kW + lane;
          G130_SPIN(&sl->flag, seq, lane, 1);
          v_r = __ldcv(&sl->data);
        }
      }
      __syncwarp();

      const float gm_l = (lane < kW) ? __uint_as_float(v_r.x) : -INFINITY;
      const float gid_l = (lane < kW) ? __uint_as_float(v_r.y) : 0.0f;
      const float gse_l = (lane < kW) ? __uint_as_float(v_r.z) : 0.0f;
      float gm = gm_l;
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, m));
      int32_t win_r = (lane < kW && gm_l == gm) ? lane : kW;
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) win_r = min(win_r, __shfl_xor_sync(0xffffffffu, win_r, m));
      const int64_t tok = (int64_t)__shfl_sync(0xffffffffu, gid_l, win_r & 7);
      const float gm0 = isinf(gm) ? 0.0f : gm;
      float tot_se = (lane < kW && !isinf(gm_l)) ? (gse_l * __expf(gm_l - gm0)) : 0.0f;
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) tot_se += __shfl_xor_sync(0xffffffffu, tot_se, m);

      if (lane == 0) {
        const float p_conf = (tot_se > 0.0f) ? (1.0f / tot_se) : 0.0f;
        d_draft_local[k] = tok;
        mp.draft_out[k] = tok;
        if (mp.conf_out != nullptr) mp.conf_out[k] = p_conf;
        mp.scratch_ctrl[0] = tok;

        if (d_greedy != 0) {
          const float pc = fminf(fmaxf(p_conf, 1e-6f), 1.0f - 1e-6f);
          const float a_k = mp.cal_ptr[k];
          const float b_k = mp.cal_ptr[8 + k];
          const float mk_ms = mp.cal_ptr[20];
          const float z = a_k * (__logf(pc) - __logf(1.0f - pc)) + b_k + d_db;
          const float c_k = 1.0f / (1.0f + __expf(-z));
          d_cpred[k] = c_k;
          d_surv *= c_k;
          d_tok_acc += d_surv;
          if (k + 1 <= mp.w_cap) {
            const float u_cur = d_tok_acc - d_lam * mp.step_eff_ptr[k + 1];
            if (u_cur > d_best_u) {
              d_best_u = u_cur;
              d_best_w = k + 1;
            }
          }
          if (k < mp.k_steps - 1) {
            const float thresh = d_best_u + d_lam * mk_ms;
            int can_beat = 0;
            for (int ww = k + 2; ww <= mp.w_cap; ++ww) {
              const float u_ub = (d_tok_acc + (float)(ww - k - 1) * d_surv) - d_lam * mp.step_eff_ptr[ww];
              if (u_ub > thresh) can_beat = 1;
            }
            if (can_beat == 0) {
              mp.scratch_ctrl[1] = 1;
              for (int rem = k + 1; rem < mp.k_steps; ++rem) {
                mp.draft_out[rem] = -1;
                if (mp.conf_out != nullptr) mp.conf_out[rem] = 0.0f;
              }
            }
          }
        }
      }
    }
    dec160_mk_grid_barrier(mp.scratch_ctrl, n_ctas, bar_gen);
    if (mp.scratch_ctrl[1] != 0) break;
  }

  // Epilogue: replaces _vw_conf_kernel + req_drafts scatter inside Draft FULL CUDA Graph!
  if (bid == 0 && tid == 0 && d_skip_epilogue == 0) {
    if (d_greedy != 0) {
      mp.vw_w_ptr[d_slot] = d_best_w;
      for (int k = 0; k < mp.k_steps; ++k) {
        mp.vw_cpred_ptr[(long long)d_slot * (long long)mp.k_steps + k] = d_cpred[k];
        const int64_t tok_masked = (k < d_best_w) ? d_draft_local[k] : -1;
        mp.draft_out[k] = tok_masked;
        if (mp.req_drafts != nullptr) {
          mp.req_drafts[(long long)d_slot * (long long)mp.req_drafts_stride + k] = tok_masked;
        }
      }
    } else if (mp.req_drafts != nullptr) {
      // Live sampling request (temp > 0): verify all k_steps unmasked drafts
      mp.vw_w_ptr[d_slot] = mp.k_steps;
      for (int k = 0; k < mp.k_steps; ++k) {
        mp.req_drafts[(long long)d_slot * (long long)mp.req_drafts_stride + k] = d_draft_local[k];
      }
    }
  }
}

// ---- 3. Standalone 1-Warp Fused Inline AG + Select + Dink Early-Stop (Zero Coop!) ----
struct InlineSelectDinkParams {
  int64_t* out_tok;
  float*   conf_out;
  int32_t* skip_ptr;
  float*   surv_ptr;
  float*   tok_ptr;
  float*   bestu_ptr;
  const int32_t* greedy_ptr;
  const float*   db_dev_ptr;
  const float*   lam_dev_ptr;
  const float*   cal_ptr;
  const float*   step_ptr;
  int32_t step_idx;
  int32_t k_steps;
};

template <int kW, int kPub>
__global__ void __launch_bounds__(32, 1) ar_kernel_inline_ag_select_dink(Params p, InlineSelectDinkParams sp) {
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (sp.step_idx > 0 && sp.skip_ptr != nullptr) {
    const int skip_v = (threadIdx.x == 0) ? *reinterpret_cast<const volatile int*>(sp.skip_ptr) : 0;
    if (__shfl_sync(0xffffffffu, skip_v, 0) != 0) {
      if (threadIdx.x == 0) {
        *sp.out_tok = -1;
        *sp.conf_out = 0.0f;
      }
      if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      return;
    }
  }
  unsigned long long seq = 0ULL;
  uint4 v_self = make_uint4(0u, 0u, 0u, 0u);
  if (threadIdx.x == 0) {
    seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1ULL;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
    if constexpr ((kPub & kR6Pdl) == 0) early = s8_read_early(p);
    v_self = p.in[0];
    s8_publish1_inline<kPub>(p, static_cast<int>(seq % (unsigned long long)kSlots), seq, early,
                             ((uint64_t)v_self.y << 32) | (uint64_t)v_self.x,
                             ((uint64_t)v_self.w << 32) | (uint64_t)v_self.z);
  }
  seq = __shfl_sync(0xffffffffu, seq, 0);
  v_self.x = __shfl_sync(0xffffffffu, v_self.x, 0);
  v_self.y = __shfl_sync(0xffffffffu, v_self.y, 0);
  v_self.z = __shfl_sync(0xffffffffu, v_self.z, 0);
  v_self.w = __shfl_sync(0xffffffffu, v_self.w, 0);
  const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
  uint4 v_r = make_uint4(__float_as_uint(-INFINITY), 0u, 0u, 0u);
  if ((int)threadIdx.x < kW) {
    if ((int)threadIdx.x == p.rank) {
      v_r = v_self;
    } else {
      const InlineSlot* sl = p.inline_recv + slot * kW + (int)threadIdx.x;
      G130_SPIN(&sl->flag, seq, (int)threadIdx.x, 1);
      v_r = __ldcv(&sl->data);
    }
  }
  __syncwarp();
  const float gm_l = ((int)threadIdx.x < kW) ? __uint_as_float(v_r.x) : -INFINITY;
  const float gid_l = ((int)threadIdx.x < kW) ? __uint_as_float(v_r.y) : 0.0f;
  const float gse_l = ((int)threadIdx.x < kW) ? __uint_as_float(v_r.z) : 0.0f;
  float gm = gm_l;
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, m));
  int32_t win_r = ((int)threadIdx.x < kW && gm_l == gm) ? (int32_t)threadIdx.x : kW;
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) win_r = min(win_r, __shfl_xor_sync(0xffffffffu, win_r, m));
  const int64_t tok = (int64_t)__shfl_sync(0xffffffffu, gid_l, win_r & 7);
  const float gm0 = isinf(gm) ? 0.0f : gm;
  float tot_se = ((int)threadIdx.x < kW && !isinf(gm_l)) ? (gse_l * __expf(gm_l - gm0)) : 0.0f;
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) tot_se += __shfl_xor_sync(0xffffffffu, tot_se, m);
  if (threadIdx.x == 0) {
    const float p_conf = (tot_se > 0.0f) ? (1.0f / tot_se) : 0.0f;
    *sp.out_tok = tok;
    *sp.conf_out = p_conf;
    if (sp.step_idx < sp.k_steps - 1 && sp.greedy_ptr != nullptr && *sp.greedy_ptr != 0) {
      const float pc = fminf(fmaxf(p_conf, 1e-6f), 1.0f - 1e-6f);
      const float z = sp.cal_ptr[sp.step_idx] * (__logf(pc) - __logf(1.0f - pc))
                    + sp.cal_ptr[8 + sp.step_idx] + *sp.db_dev_ptr;
      const float c = 1.0f / (1.0f + __expf(-z));
      const float surv = (*sp.surv_ptr) * c;
      const float tokens = (*sp.tok_ptr) + surv;
      const float lam = *sp.lam_dev_ptr;
      const float u_cur = tokens - lam * sp.step_ptr[sp.step_idx + 1];
      const float best_u = fmaxf(*sp.bestu_ptr, u_cur);
      *sp.surv_ptr = surv;
      *sp.tok_ptr = tokens;
      *sp.bestu_ptr = best_u;
      const float thresh = best_u + lam * sp.cal_ptr[20];
      int can_beat = 0;
      for (int ww = sp.step_idx + 2; ww <= sp.k_steps; ++ww) {
        const float u_ub = (tokens + (float)(ww - sp.step_idx - 1) * surv) - lam * sp.step_ptr[ww];
        if (u_ub > thresh) can_beat = 1;
      }
      if (can_beat == 0) *sp.skip_ptr = 1;
    }
  }
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
}

template <int kPub>
void launch_inline_ag_select_dink_one(const Params& p, const InlineSelectDinkParams& sp, cudaStream_t stream) {
  if constexpr ((kPub & kR6Pdl) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1u);
    cfg.blockDim = dim3(32u);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_inline_ag_select_dink<8, kPub>, p, sp));
  } else {
    ar_kernel_inline_ag_select_dink<8, kPub><<<1, 32, 0, stream>>>(p, sp);
  }
}

template <int kPub, int kP2Mode, bool kWireBf12>
void launch_rows_one(const Params& p, int blocks, cudaStream_t stream) {
  if constexpr ((kPub & kR6Pdl) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)blocks);
    cfg.blockDim = dim3(kRowsThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_rows<8, kPub, kP2Mode, kWireBf12>, p));
  } else {
    ar_kernel_rows<8, kPub, kP2Mode, kWireBf12><<<blocks, kRowsThreads, 0, stream>>>(p);
  }
}

template <int kPub, bool kWireBf12>
void launch_rows_p2(const Params& p, int blocks, cudaStream_t stream) {
  const int mode = (p.rows_p2 == 3 && blocks >= 8 && !p.timing) ? 3
                 : (p.rows_p2 == 1) ? 1
                 : (p.rows_p2 >= 2) ? 2 : 0;
  switch (mode) {
    case 1: launch_rows_one<kPub, 1, kWireBf12>(p, blocks, stream); break;
    case 2: launch_rows_one<kPub, 2, kWireBf12>(p, blocks, stream); break;
    case 3: launch_rows_one<kPub, 3, kWireBf12>(p, blocks, stream); break;
    default: launch_rows_one<kPub, 0, kWireBf12>(p, blocks, stream); break;
  }
}

void launch_rows_kernel(const Params& p, long long vec_per_cta, cudaStream_t stream, int pub) {
  const int blocks = (int)std::max(1LL, std::min(32LL, (p.nvec_full + vec_per_cta - 1) / vec_per_cta));
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  const int m = pub & (B | kR6Pdl | kR6Trig | kR3Sbuf);
  const bool bf12 = (p.wire_bf12 != 0);
  #define DISP_ROWS(M) \
    if (bf12) launch_rows_p2<M, true>(p, blocks, stream); \
    else launch_rows_p2<M, false>(p, blocks, stream)
  switch (m) {
    case B: DISP_ROWS(B); break;
    case B | kR6Pdl: DISP_ROWS(B | kR6Pdl); break;
    case B | kR6Pdl | kR6Trig: DISP_ROWS(B | kR6Pdl | kR6Trig); break;
    case B | kR3Sbuf: DISP_ROWS(B | kR3Sbuf); break;
    case B | kR6Pdl | kR3Sbuf: DISP_ROWS(B | kR6Pdl | kR3Sbuf); break;
    case B | kR6Pdl | kR6Trig | kR3Sbuf: DISP_ROWS(B | kR6Pdl | kR6Trig | kR3Sbuf); break;
    default: throw std::runtime_error("rdma_ar s8: unsupported rows kernel mask " + std::to_string(m));
  }
  #undef DISP_ROWS
}

template <int N_STACK, int kCase, int kPub>
void launch_sconv_ar_one(const Params& p, const SconvAttnResParams& sp, cudaStream_t stream) {
  constexpr size_t kDynSmem = 27648u + (N_STACK > 4 ? (size_t)(N_STACK - 4) * 9216u : 0u);
  if constexpr (kDynSmem > 40960u) {
    static bool smem_set = false;
    if (!smem_set) {
      CHECK_CUDA(cudaFuncSetAttribute(ar_sconv_attnres_kernel<N_STACK, kCase, kPub>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kDynSmem));
      smem_set = true;
    }
  }
  const unsigned blocks = (unsigned)(1 + sp.nq * sp.T);
  if constexpr ((kPub & kR6Pdl) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(256u);
    cfg.dynamicSmemBytes = kDynSmem;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_sconv_attnres_kernel<N_STACK, kCase, kPub>, p, sp));
  } else {
    ar_sconv_attnres_kernel<N_STACK, kCase, kPub><<<blocks, 256, kDynSmem, stream>>>(p, sp);
  }
}

template <int N_STACK, int kCase>
void launch_sconv_ar_pub(const Params& p, const SconvAttnResParams& sp, cudaStream_t stream, int pub) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  const int m = pub & (B | kR6Pdl | kR6Trig | kR3Sbuf);
  switch (m) {
    case B: launch_sconv_ar_one<N_STACK, kCase, B>(p, sp, stream); break;
    case B | kR6Pdl: launch_sconv_ar_one<N_STACK, kCase, B | kR6Pdl>(p, sp, stream); break;
    case B | kR6Pdl | kR6Trig: launch_sconv_ar_one<N_STACK, kCase, B | kR6Pdl | kR6Trig>(p, sp, stream); break;
    case B | kR3Sbuf: launch_sconv_ar_one<N_STACK, kCase, B | kR3Sbuf>(p, sp, stream); break;
    case B | kR6Pdl | kR3Sbuf: launch_sconv_ar_one<N_STACK, kCase, B | kR6Pdl | kR3Sbuf>(p, sp, stream); break;
    case B | kR6Pdl | kR6Trig | kR3Sbuf: launch_sconv_ar_one<N_STACK, kCase, B | kR6Pdl | kR6Trig | kR3Sbuf>(p, sp, stream); break;
    default: throw std::runtime_error("rdma_ar s8: unsupported sconv_attnres kernel mask " + std::to_string(m));
  }
}

template <int N_STACK>
void launch_sconv_ar_case(const Params& p, const SconvAttnResParams& sp, int kcase, cudaStream_t stream, int pub) {
  switch (kcase) {
    case 0: launch_sconv_ar_pub<N_STACK, 0>(p, sp, stream, pub); break;
    case 1: launch_sconv_ar_pub<N_STACK, 1>(p, sp, stream, pub); break;
    case 2: launch_sconv_ar_pub<N_STACK, 2>(p, sp, stream, pub); break;
    case 3: launch_sconv_ar_pub<N_STACK, 3>(p, sp, stream, pub); break;
    default: throw std::runtime_error("rdma_ar s8: bad sconv_attnres case " + std::to_string(kcase));
  }
}

void launch_sconv_attnres_kernel(const Params& p, const SconvAttnResParams& sp, int n_stack, int kcase,
                                 cudaStream_t stream, int pub) {
  switch (n_stack) {
    case 1: launch_sconv_ar_case<1>(p, sp, kcase, stream, pub); break;
    case 2: launch_sconv_ar_case<2>(p, sp, kcase, stream, pub); break;
    case 3: launch_sconv_ar_case<3>(p, sp, kcase, stream, pub); break;
    case 4: launch_sconv_ar_case<4>(p, sp, kcase, stream, pub); break;
    case 5: launch_sconv_ar_case<5>(p, sp, kcase, stream, pub); break;
    case 6: launch_sconv_ar_case<6>(p, sp, kcase, stream, pub); break;
    case 7: launch_sconv_ar_case<7>(p, sp, kcase, stream, pub); break;
    case 8: launch_sconv_ar_case<8>(p, sp, kcase, stream, pub); break;
    default: throw std::runtime_error("rdma_ar s8: unsupported n_stack " + std::to_string(n_stack));
  }
}
// ---- end r3-comm kernel


// =================================================================================================
// c32-rdma-comm: High-Concurrency C=32 Collectives (M = 32..512, 100% 0-ULP Bit-Exact!)
// =================================================================================================
//   1. ar_kernel_med_gather<kW, kPub>: 1-CTA 128-thread zero-atomic gather for 9 <= nvec <= 1024
//      (covers B=16..32 Markov 256..512 B AGs & M=16..256 sync_free 1..16 KiB LM-head AGs!).
//   2. BF12-G128-OVF: 100% 128-bit coalesced warp-parallel 0-ULP lossless BF12 wire codec
//      (0.768x wire bytes on real Inkling-2 activations, bijective across ALL 65,536 BF16 patterns,
//       zero global barriers, zero raw re-pass, ITS-safe half-warp overflow shuffles!).
//   3. ar_kernel_c32<kW, kPub, kBf12P1, kBf12P2, kPipe2>: 43-CTA (1.0 wave, <=128 regs, 0 spills)
//      2-shot AllReduce with BF12-G128-OVF on P2 or P1+P2, optional 2-stage software pipelining
//      (A+B halves hiding P1(B) pack, P1(A) reduce, P2(A) pack, P2(A) unpack behind full-duplex
//       2-QP Dual-Root PCIe Gen5 x4 wire!), per-peer overlapped P2 wake, and immediate ar_done_cnt.
//   4. ar_kernel_draft_c32_h64_q_norm<kW, kPub, kBits>: opt-in batched drafter 2-shot H64-Q4/Q6/Q8
//      AR + fused residual add + RMSNorm for B=2..32 (M=16..256, H=4608, 0 intermediate IFWHT!).
// =================================================================================================

template <int kW, int kPub>
__global__ void __launch_bounds__(128, 2) ar_kernel_med_gather(Params p) {
  static_assert(kW == 8, "ar_kernel_med_gather expects kW == 8");
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  __shared__ unsigned long long s_seq;
  if (p.skip_ptr != nullptr) {
    __shared__ int s_skip;
    if (threadIdx.x == 0) s_skip = *reinterpret_cast<const volatile int*>(p.skip_ptr);
    __syncthreads();
    if (s_skip != 0) {
      if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      return;
    }
  }
  if (threadIdx.x == 0) {
    const unsigned long long seq0 = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1ULL;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq0;
    s_seq = seq0;
    if constexpr ((kPub & kR6Pdl) == 0) early = s8_read_early(p);
    if (p.timing) p.st->t_start = now_ns();
  }
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
  const long long nvec = p.nvec;
  uint4* send4 = reinterpret_cast<uint4*>(p.send + (long long)slot * p.max_bytes);
  for (long long i = threadIdx.x; i < nvec; i += (long long)blockDim.x) {
    const uint4 v = p.in[i];
    send4[i] = v;
    p.out[(long long)p.rank * nvec + i] = v;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    Params pg = p;
    pg.chunk = nvec;
    r3_publish1<kGather, kPub & ~kR3Sbuf>(pg, seq, early, 0);
  }
  if ((int)threadIdx.x < kW && (int)threadIdx.x != p.rank) {
    G130_SPIN(p.flags + (slot * kW + (int)threadIdx.x) * kFlagStride, seq, (int)threadIdx.x, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) st_relaxed_gpu(&p.st->ar_done_cnt, seq);
  if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const char* rbase = p.recv + (long long)slot * kW * rstride;
  for (int r = 0; r < kW; ++r) {
    if (r == p.rank) continue;
    const uint4* src_r = reinterpret_cast<const uint4*>(rbase + (long long)r * rstride);
    uint4* dst_r = p.out + (long long)r * nvec;
    for (long long i = threadIdx.x; i < nvec; i += (long long)blockDim.x) {
      dst_r[i] = __ldcv(src_r + i);
    }
  }
}

__device__ __forceinline__ uint32_t bf12_reconstruct2_raw(uint32_t sm16, uint32_t lo8, uint32_t hi8) {
  const uint32_t s0 = sm16 & 0xFFu, e0 = ((hi8 & 0xFu) << 4) | (lo8 & 0xFu);
  const uint32_t s1 = (sm16 >> 8) & 0xFFu, e1 = ((hi8 >> 4) << 4) | ((lo8 >> 4) & 0xFu);
  const uint32_t u0 = ((s0 & 0x80u) << 8) | (e0 << 7) | (s0 & 0x7Fu);
  const uint32_t u1 = ((s1 & 0x80u) << 8) | (e1 << 7) | (s1 & 0x7Fu);
  return u0 | (u1 << 16);
}

// 100% 128-bit coalesced warp-parallel 0-ULP BF12-G128-OVF encoder (1 warp = 2 tiles = 32 uint4 = 256 BF16).
// Zero shared memory, zero __syncthreads(), ITS-safe half-warp overflow shuffles!
__device__ __forceinline__ void bf12_g128_pack_warp(
    uint4 v, uint32_t pair_idx, uint32_t n_tiles, char* dst_wire, unsigned int* ovf_cnt) {
  const int lane = (int)(threadIdx.x & 31u);
  const int hw = lane >> 4;
  const int h = lane & 15;
  const uint32_t tile_even = pair_idx << 1;

  const uint32_t u0 = v.x & 0xFFFFu, u1 = v.x >> 16;
  const uint32_t u2 = v.y & 0xFFFFu, u3 = v.y >> 16;
  const uint32_t u4 = v.z & 0xFFFFu, u5 = v.z >> 16;
  const uint32_t u6 = v.w & 0xFFFFu, u7 = v.w >> 16;

  const uint32_t e0 = (u0 >> 7) & 0xFFu, e1 = (u1 >> 7) & 0xFFu;
  const uint32_t e2 = (u2 >> 7) & 0xFFu, e3 = (u3 >> 7) & 0xFFu;
  const uint32_t e4 = (u4 >> 7) & 0xFFu, e5 = (u5 >> 7) & 0xFFu;
  const uint32_t e6 = (u6 >> 7) & 0xFFu, e7 = (u7 >> 7) & 0xFFu;

  const uint32_t sm_lo = (((u0 >> 8) & 0x80u) | (u0 & 0x7Fu))
                       | ((((u1 >> 8) & 0x80u) | (u1 & 0x7Fu)) << 8)
                       | ((((u2 >> 8) & 0x80u) | (u2 & 0x7Fu)) << 16)
                       | ((((u3 >> 8) & 0x80u) | (u3 & 0x7Fu)) << 24);
  const uint32_t sm_hi = (((u4 >> 8) & 0x80u) | (u4 & 0x7Fu))
                       | ((((u5 >> 8) & 0x80u) | (u5 & 0x7Fu)) << 8)
                       | ((((u6 >> 8) & 0x80u) | (u6 & 0x7Fu)) << 16)
                       | ((((u7 >> 8) & 0x80u) | (u7 & 0x7Fu)) << 24);

  uint32_t m = max(max(max(e0, e1), max(e2, e3)), max(max(e4, e5), max(e6, e7)));
  // Butterfly xor 8, 4, 2, 1 stays strictly inside each 16-lane half-warp!
  m = max(m, __shfl_xor_sync(0xFFFFFFFFu, m, 8));
  m = max(m, __shfl_xor_sync(0xFFFFFFFFu, m, 4));
  m = max(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
  m = max(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));

  uint32_t esc = ((e0 != 0u && (m - e0) > 13u) | (e1 != 0u && (m - e1) > 13u)
                | (e2 != 0u && (m - e2) > 13u) | (e3 != 0u && (m - e3) > 13u)
                | (e4 != 0u && (m - e4) > 13u) | (e5 != 0u && (m - e5) > 13u)
                | (e6 != 0u && (m - e6) > 13u) | (e7 != 0u && (m - e7) > 13u)) ? 1u : 0u;
  esc |= __shfl_xor_sync(0xFFFFFFFFu, esc, 8);
  esc |= __shfl_xor_sync(0xFFFFFFFFu, esc, 4);
  esc |= __shfl_xor_sync(0xFFFFFFFFu, esc, 2);
  esc |= __shfl_xor_sync(0xFFFFFFFFu, esc, 1);

  const size_t desc_bytes = ((size_t)n_tiles * 2u + 15u) & ~15ull;
  uint16_t* desc_out = reinterpret_cast<uint16_t*>(dst_wire + 16);
  uint4* sm8_out = reinterpret_cast<uint4*>(dst_wire + 16 + desc_bytes);
  uint4* c4_out = sm8_out + (size_t)n_tiles * 8u;
  uint4* ovf_out = c4_out + (size_t)n_tiles * 4u;

  uint32_t codes = 0u;
  uint16_t desc = 0u;
  if (esc == 0u) {
    codes = ((e0 == 0u) ? 14u : (m - e0))
          | (((e1 == 0u) ? 14u : (m - e1)) << 4)
          | (((e2 == 0u) ? 14u : (m - e2)) << 8)
          | (((e3 == 0u) ? 14u : (m - e3)) << 12)
          | (((e4 == 0u) ? 14u : (m - e4)) << 16)
          | (((e5 == 0u) ? 14u : (m - e5)) << 20)
          | (((e6 == 0u) ? 14u : (m - e6)) << 24)
          | (((e7 == 0u) ? 14u : (m - e7)) << 28);
    desc = static_cast<uint16_t>(m & 0xFFu);
  } else {
    codes = (e0 & 0xFu) | ((e1 & 0xFu) << 4) | ((e2 & 0xFu) << 8) | ((e3 & 0xFu) << 12)
          | ((e4 & 0xFu) << 16) | ((e5 & 0xFu) << 20) | ((e6 & 0xFu) << 24) | ((e7 & 0xFu) << 28);
    const uint32_t hi4 = ((e0 >> 4) & 0xFu) | (((e1 >> 4) & 0xFu) << 4) | (((e2 >> 4) & 0xFu) << 8)
                       | (((e3 >> 4) & 0xFu) << 12) | (((e4 >> 4) & 0xFu) << 16) | (((e5 >> 4) & 0xFu) << 20)
                       | (((e6 >> 4) & 0xFu) << 24) | (((e7 >> 4) & 0xFu) << 28);
    const uint32_t hw_mask = (hw == 0) ? 0x0000FFFFu : 0xFFFF0000u;
    uint32_t ovf_idx = 0u;
    if (h == 0) ovf_idx = atomicAdd(ovf_cnt, 1u);
    ovf_idx = __shfl_sync(hw_mask, ovf_idx, hw * 16);
    desc = static_cast<uint16_t>(0x8000u | (ovf_idx & 0x7FFFu));
    const int base_h = hw * 16 + (h & 3) * 4;
    const uint32_t w0 = __shfl_sync(hw_mask, hi4, base_h + 0);
    const uint32_t w1 = __shfl_sync(hw_mask, hi4, base_h + 1);
    const uint32_t w2 = __shfl_sync(hw_mask, hi4, base_h + 2);
    const uint32_t w3 = __shfl_sync(hw_mask, hi4, base_h + 3);
    if (h < 4) ovf_out[(size_t)ovf_idx * 4u + (size_t)h] = make_uint4(w0, w1, w2, w3);
  }

  // 16 contiguous lanes write 256 B sm8 (STG.E.128)
  const int sl = (lane & 15) * 2;
  const uint32_t a0 = __shfl_sync(0xFFFFFFFFu, sm_lo, sl);
  const uint32_t a1 = __shfl_sync(0xFFFFFFFFu, sm_hi, sl);
  const uint32_t a2 = __shfl_sync(0xFFFFFFFFu, sm_lo, sl + 1);
  const uint32_t a3 = __shfl_sync(0xFFFFFFFFu, sm_hi, sl + 1);
  if (lane < 16) sm8_out[(size_t)tile_even * 8u + (size_t)lane] = make_uint4(a0, a1, a2, a3);

  // 8 contiguous lanes write 128 B c4 (STG.E.128)
  const int cl = (lane & 7) * 4;
  const uint32_t c0 = __shfl_sync(0xFFFFFFFFu, codes, cl + 0);
  const uint32_t c1 = __shfl_sync(0xFFFFFFFFu, codes, cl + 1);
  const uint32_t c2 = __shfl_sync(0xFFFFFFFFu, codes, cl + 2);
  const uint32_t c3 = __shfl_sync(0xFFFFFFFFu, codes, cl + 3);
  if (lane < 8) c4_out[(size_t)tile_even * 4u + (size_t)lane] = make_uint4(c0, c1, c2, c3);

  const uint32_t desc_odd = __shfl_sync(0xFFFFFFFFu, (uint32_t)desc, 16);
  if (lane == 0) reinterpret_cast<uint32_t*>(desc_out)[pair_idx] = (uint32_t)desc | (desc_odd << 16);
}

// 100% 128-bit coalesced warp-parallel 0-ULP BF12-G128-OVF decoder (LDG.CV.128 -> register uint4).
__device__ __forceinline__ uint4 bf12_g128_unpack_warp(
    const char* src_wire, uint32_t pair_idx, uint32_t n_tiles) {
  const int lane = (int)(threadIdx.x & 31u);
  const int hw = lane >> 4;
  const int h = lane & 15;
  const uint32_t tile_even = pair_idx << 1;

  const size_t desc_bytes = ((size_t)n_tiles * 2u + 15u) & ~15ull;
  const uint16_t* desc_in = reinterpret_cast<const uint16_t*>(src_wire + 16);
  const uint4* sm8_in = reinterpret_cast<const uint4*>(src_wire + 16 + desc_bytes);
  const uint4* c4_in = sm8_in + (size_t)n_tiles * 8u;
  const uint4* ovf_in = c4_in + (size_t)n_tiles * 4u;

  uint32_t dpair = (lane == 0) ? __ldcv(reinterpret_cast<const uint32_t*>(desc_in) + pair_idx) : 0u;
  dpair = __shfl_sync(0xFFFFFFFFu, dpair, 0);
  const uint32_t desc = (hw == 0) ? (dpair & 0xFFFFu) : (dpair >> 16);

  uint4 sm_u4 = (lane < 16) ? __ldcv(sm8_in + (size_t)tile_even * 8u + (size_t)lane) : make_uint4(0u, 0u, 0u, 0u);
  uint4 c_u4 = (lane < 8) ? __ldcv(c4_in + (size_t)tile_even * 4u + (size_t)lane) : make_uint4(0u, 0u, 0u, 0u);

  const int src_sm = lane >> 1;
  const uint32_t sx = __shfl_sync(0xFFFFFFFFu, sm_u4.x, src_sm);
  const uint32_t sy = __shfl_sync(0xFFFFFFFFu, sm_u4.y, src_sm);
  const uint32_t sz = __shfl_sync(0xFFFFFFFFu, sm_u4.z, src_sm);
  const uint32_t sw = __shfl_sync(0xFFFFFFFFu, sm_u4.w, src_sm);
  const uint32_t sm_lo = (lane & 1) ? sz : sx;
  const uint32_t sm_hi = (lane & 1) ? sw : sy;

  const int src_c = lane >> 2;
  const uint32_t cx = __shfl_sync(0xFFFFFFFFu, c_u4.x, src_c);
  const uint32_t cy = __shfl_sync(0xFFFFFFFFu, c_u4.y, src_c);
  const uint32_t cz = __shfl_sync(0xFFFFFFFFu, c_u4.z, src_c);
  const uint32_t cw = __shfl_sync(0xFFFFFFFFu, c_u4.w, src_c);
  const uint32_t codes = ((lane & 3) == 0) ? cx : (((lane & 3) == 1) ? cy : (((lane & 3) == 2) ? cz : cw));

  uint4 o;
  if ((desc & 0x8000u) == 0u) {
    const uint32_t max_e = desc & 0xFFu;
    o.x = bf12_reconstruct2(sm_lo & 0xFFFFu, codes & 0xFFu, max_e);
    o.y = bf12_reconstruct2(sm_lo >> 16, (codes >> 8) & 0xFFu, max_e);
    o.z = bf12_reconstruct2(sm_hi & 0xFFFFu, (codes >> 16) & 0xFFu, max_e);
    o.w = bf12_reconstruct2(sm_hi >> 16, (codes >> 24) & 0xFFu, max_e);
  } else {
    const uint32_t hw_mask = (hw == 0) ? 0x0000FFFFu : 0xFFFF0000u;
    const uint32_t ovf_idx = desc & 0x7FFFu;
    uint4 hi_u4 = (h < 4) ? __ldcv(ovf_in + (size_t)ovf_idx * 4u + (size_t)h) : make_uint4(0u, 0u, 0u, 0u);
    const int src_h = hw * 16 + (h >> 2);
    const uint32_t hx = __shfl_sync(hw_mask, hi_u4.x, src_h);
    const uint32_t hy = __shfl_sync(hw_mask, hi_u4.y, src_h);
    const uint32_t hz = __shfl_sync(hw_mask, hi_u4.z, src_h);
    const uint32_t hw_w = __shfl_sync(hw_mask, hi_u4.w, src_h);
    const uint32_t hi4 = ((h & 3) == 0) ? hx : (((h & 3) == 1) ? hy : (((h & 3) == 2) ? hz : hw_w));
    o.x = bf12_reconstruct2_raw(sm_lo & 0xFFFFu, codes & 0xFFu, hi4 & 0xFFu);
    o.y = bf12_reconstruct2_raw(sm_lo >> 16, (codes >> 8) & 0xFFu, (hi4 >> 8) & 0xFFu);
    o.z = bf12_reconstruct2_raw(sm_hi & 0xFFFFu, (codes >> 16) & 0xFFu, (hi4 >> 16) & 0xFFu);
    o.w = bf12_reconstruct2_raw(sm_hi >> 16, (codes >> 24) & 0xFFu, (hi4 >> 24) & 0xFFu);
  }
  return o;
}

template <int kPub>
__device__ __forceinline__ void c32_publish1(
    const Params& p, int slot, unsigned long long seq, uint64_t early,
    uint64_t src_off, uint64_t send_bytes, uint64_t chunk_vecs, uint64_t mode_val) {
  s8_guard_slot<kPub>(p, seq, early);
  if (p.level_db) {
    CtrlSlot* cs = p.ctrl_slots + slot;
    *reinterpret_cast<volatile uint64_t*>(&cs->send_bytes) = send_bytes;
    *reinterpret_cast<volatile uint64_t*>(&cs->src_off) = src_off;
    *reinterpret_cast<volatile uint64_t*>(&cs->mode) = mode_val;
    *reinterpret_cast<volatile uint64_t*>(&cs->chunk_vecs) = chunk_vecs;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&cs->pub_time) = tp;
    }
    st_release_sys(&cs->send_seq, seq);
    return;
  }
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = send_bytes;
  if (src_off != 0) *reinterpret_cast<volatile uint64_t*>(&p.ctrl->src_off) = src_off;
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = mode_val;
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = chunk_vecs;
  if (p.timing) {
    const unsigned long long tp = now_ns();
    p.st->t_pub = tp;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
  }
  st_release_sys(&p.ctrl->send_seq, seq);
}

__device__ __forceinline__ void c32_wait_p1_flags(
    const Params& p, int slot, unsigned long long seq, bool striped) {
  if (blockIdx.x == 0) {
    if ((int)threadIdx.x < p.world && (int)threadIdx.x != p.rank) {
      const int r = (int)threadIdx.x;
      spin_until_thin(p.flags + (slot * p.world + r) * kFlagStride, seq, p, r, 1);
      if (striped && p.flags_qp2 != nullptr) {
        spin_until_thin(p.flags_qp2 + (slot * p.world + r) * kFlagStride, seq, p, r, 11);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) st_release_sys_ull(&p.st->p1_done, seq);
  } else {
    if (threadIdx.x == 0) {
      if (ld_acquire_gpu(&p.st->p1_done) < seq) {
        const unsigned long long t0 = now_ns();
        unsigned int it = 0;
        while (ld_acquire_gpu(&p.st->p1_done) < seq) {
          if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
            printf("rdma_ar c32 rank %d: timeout waiting for p1_done (seq %llu)\n", p.rank,
                   (unsigned long long)seq);
            r3_trap();
          }
        }
      }
      (void)ld_acquire_sys_ull(&p.st->p1_done);
    }
    __syncthreads();
    fence_acquire_sys();
  }
}

template <int kW, int kPub, bool kBf12P1, bool kBf12P2, bool kPipe2>
__global__ void __launch_bounds__(kC32Threads, 2) ar_kernel_c32(Params p) {
  static_assert(kW == 8, "ar_kernel_c32 requires kW == 8");
  constexpr bool kUseSbuf = ((kPub & kR3Sbuf) != 0) && !kBf12P1;

  __shared__ unsigned long long s_seq;
  __shared__ int s_last;

  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0 && (!kUseSbuf || blockIdx.x == 0)) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (threadIdx.x == 0) {
    s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1ULL;
    if constexpr ((kPub & kR6Pdl) == 0) {
      if (!kUseSbuf || blockIdx.x == 0) early = s8_read_early(p);
    }
    if (p.timing && blockIdx.x == 0) p.st->t_start = now_ns();
  }
  __syncthreads();

  const unsigned long long seq0 = s_seq;
  const unsigned long long seq_end = seq0 + (kPipe2 ? 1ULL : 0ULL);
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq_end;
  }

  long long eff_nvec = p.nvec;
  if (p.rows_ptr != nullptr && p.row_vec > 0) {
    int nl = *p.rows_ptr;
    if (nl < 1) nl = 1;
    // Round up to multiple of 8 rows so eff_nvec = m_eff * 576 is strictly divisible by
    // 2 * kW * 32 = 512 (exact for BOTH kPipe2=false & kPipe2=true, raw & BF12-G128-OVF!).
    const int m_eff = (nl + 7) & ~7;
    const long long cand = (long long)m_eff * p.row_vec;
    if (cand > 0 && cand < eff_nvec) eff_nvec = cand;
  }
  const long long st_nvec = kPipe2 ? (eff_nvec >> 1) : eff_nvec;
  const long long st_chunk = st_nvec / kW;
  const uint32_t n_tiles = (uint32_t)(st_chunk >> 4);
  const uint32_t n_pairs = (uint32_t)(st_chunk >> 5);
  const bool do_stripe = (p.dual_qp != 0) && (st_chunk * 16LL >= p.stripe_min_bytes);
  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const int lane = (int)(threadIdx.x & 31u);
  const int warp_in_cta = (int)(threadIdx.x >> 5);
  const uint32_t wid = (uint32_t)blockIdx.x * 8u + (uint32_t)warp_in_cta;
  const uint32_t n_warps = (uint32_t)gridDim.x * 8u;

  // Helper lambda: Phase 1 pack/stage + publish for stage `stg` in {0, 1}
  auto run_p1 = [&](int stg) {
    const unsigned long long seq = seq0 + (unsigned long long)stg;
    const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
    const uint4* in_base = p.in + (long long)stg * st_nvec;
    unsigned int* arrive_ptr = (stg == 0) ? &p.st->c32_arrive_a : &p.st->c32_arrive_b;
    const uint64_t stripe_bit = do_stripe ? kModeP1Stripe : 0ULL;

    if constexpr (kUseSbuf) {
      if (blockIdx.x == 0) {
        if (stg == 0 && threadIdx.x == 32) {
          uint64_t* hdr = reinterpret_cast<uint64_t*>(const_cast<uint4*>(p.in) - 1);
          const uint64_t last = ld_relaxed_sys(hdr);
          if (last != 0 && seq0 - last < 2ULL) {
            printf("rdma_ar c32 rank %d: send buffer reused too early (seq %llu after %llu)\n",
                   p.rank, (unsigned long long)seq0, (unsigned long long)last);
            r3_trap();
          }
          st_relaxed_sys(hdr, seq_end);
        }
        if (threadIdx.x == 0) {
          if (stg == 0) __threadfence_system();
          const uint64_t src_off =
              (uint64_t)(reinterpret_cast<const char*>(in_base) - reinterpret_cast<const char*>(p.ctrl));
          c32_publish1<kPub>(p, slot, seq, early, src_off, (uint64_t)st_nvec * 16ULL,
                             (uint64_t)st_chunk, (uint64_t)kTwoShot | kModeSrc | stripe_bit);
        }
      }
    } else if constexpr (kBf12P1) {
      const uint32_t total_pairs = (uint32_t)(kW - 1) * n_pairs;
      char* send_slot = p.send + (long long)slot * p.max_bytes;
      for (uint32_t idx = wid; idx < total_pairs; idx += n_warps) {
        const uint32_t peer_ord = idx / n_pairs;
        const uint32_t pair_idx = idx - peer_ord * n_pairs;
        const int dst_q = (int)((peer_ord < (uint32_t)p.rank) ? peer_ord : (peer_ord + 1u));
        const uint4 v = in_base[(long long)dst_q * st_chunk + (long long)pair_idx * 32LL + (long long)lane];
        bf12_g128_pack_warp(v, pair_idx, n_tiles, send_slot + (long long)dst_q * p.max_chunk,
                            &p.st->c32_p1_ovf[slot][dst_q]);
      }
      if (last_cta(arrive_ptr, &s_last)) {
        const size_t desc_bytes = ((size_t)n_tiles * 2u + 15u) & ~15ull;
        for (int q = 0; q < kW; ++q) {
          if (q == p.rank) {
            p.ctrl_c32[slot].p1_bytes[q] = 0u;
            continue;
          }
          const uint32_t novf = *reinterpret_cast<volatile unsigned int*>(&p.st->c32_p1_ovf[slot][q]);
          *reinterpret_cast<volatile unsigned int*>(&p.st->c32_p1_ovf[slot][q]) = 0u;
          const uint32_t tb = static_cast<uint32_t>(
              ((16u + desc_bytes + (size_t)n_tiles * 192u + (size_t)novf * 64u) + 15u) & ~15ull);
          Bf12G128Hdr hdr{static_cast<uint32_t>(st_chunk), n_tiles, novf, tb};
          *reinterpret_cast<uint4*>(send_slot + (long long)q * p.max_chunk) =
              *reinterpret_cast<const uint4*>(&hdr);
          *reinterpret_cast<volatile uint32_t*>(&p.ctrl_c32[slot].p1_bytes[q]) = tb;
        }
        __threadfence_system();
        c32_publish1<kPub & ~kR3Sbuf>(p, slot, seq, early, 0ULL, (uint64_t)st_nvec * 16ULL,
                                      (uint64_t)st_chunk,
                                      (uint64_t)kTwoShot | kModePeerLens | stripe_bit);
      }
    } else {
      // Raw 43-CTA coalesced staging (skips own rank's chunk -> -12.5% staging bandwidth!)
      uint4* send4 = reinterpret_cast<uint4*>(p.send + (long long)slot * p.max_bytes);
      const long long total_vecs = (long long)(kW - 1) * st_chunk;
      const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
      const long long stride = (long long)gridDim.x * (long long)blockDim.x;
      for (long long idx = tid; idx < total_vecs; idx += stride) {
        const long long peer_ord = idx / st_chunk;
        const long long v_off = idx - peer_ord * st_chunk;
        const long long dst_q = (peer_ord < (long long)p.rank) ? peer_ord : (peer_ord + 1LL);
        const long long g = dst_q * st_chunk + v_off;
        send4[g] = in_base[g];
      }
      if (last_cta(arrive_ptr, &s_last)) {
        c32_publish1<kPub>(p, slot, seq, early, 0ULL, (uint64_t)st_nvec * 16ULL,
                           (uint64_t)st_chunk, (uint64_t)kTwoShot | stripe_bit);
      }
    }
  };

  // Helper lambda: Wait P1 + Exact Rank 0..7 FP32 Reduce + Register-Direct P2 Pack & Publish
  auto run_reduce_p2 = [&](int stg) {
    const unsigned long long seq = seq0 + (unsigned long long)stg;
    const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
    const uint4* in_base = p.in + (long long)stg * st_nvec;
    uint4* out_base = p.out + (long long)stg * st_nvec;
    unsigned int* arrive2_ptr = (stg == 0) ? &p.st->c32_arrive2_a : &p.st->c32_arrive2_b;

    c32_wait_p1_flags(p, slot, seq, do_stripe);

    const char* rbase = p.recv + (long long)slot * kW * rstride;
    char* send2_c = p.send2 + (long long)slot * p.max_chunk;
    const long long b_own = (long long)p.rank * st_chunk;

    for (uint32_t pair_idx = wid; pair_idx < n_pairs; pair_idx += n_warps) {
      const long long i = (long long)pair_idx * 32LL + (long long)lane;
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if constexpr (kBf12P1) {
#pragma unroll 1
        for (int r = 0; r < kW; ++r) {
          const uint4 vr = (r == p.rank)
                               ? in_base[b_own + i]
                               : bf12_g128_unpack_warp(rbase + (long long)r * rstride, pair_idx, n_tiles);
          acc8(acc, vr);
        }
      } else {
        // Explicit scalar regs v0..v7 -> 0 B local stack spill + all 7 C2C loads in flight!
        const uint4 v0 = (0 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 0LL * rstride) + i);
        const uint4 v1 = (1 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 1LL * rstride) + i);
        const uint4 v2 = (2 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 2LL * rstride) + i);
        const uint4 v3 = (3 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 3LL * rstride) + i);
        const uint4 v4 = (4 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 4LL * rstride) + i);
        const uint4 v5 = (5 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 5LL * rstride) + i);
        const uint4 v6 = (6 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 6LL * rstride) + i);
        const uint4 v7 = (7 == p.rank) ? in_base[b_own + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + 7LL * rstride) + i);
        acc8(acc, v0); acc8(acc, v1); acc8(acc, v2); acc8(acc, v3);
        acc8(acc, v4); acc8(acc, v5); acc8(acc, v6); acc8(acc, v7);
      }
      const uint4 o = pack8(acc);
      out_base[b_own + i] = o;
      if constexpr (kBf12P2) {
        bf12_g128_pack_warp(o, pair_idx, n_tiles, send2_c, &p.st->c32_p2_ovf[slot]);
      } else {
        reinterpret_cast<uint4*>(send2_c)[i] = o;
      }
    }

    if (last_cta(arrive2_ptr, &s_last)) {
      uint64_t p2_bytes = (uint64_t)st_chunk * 16ULL;
      if constexpr (kBf12P2) {
        const uint32_t novf = *reinterpret_cast<volatile unsigned int*>(&p.st->c32_p2_ovf[slot]);
        *reinterpret_cast<volatile unsigned int*>(&p.st->c32_p2_ovf[slot]) = 0u;
        const size_t desc_bytes = ((size_t)n_tiles * 2u + 15u) & ~15ull;
        p2_bytes = ((16ULL + (uint64_t)desc_bytes + (uint64_t)n_tiles * 192ULL + (uint64_t)novf * 64ULL) + 15ULL) & ~15ULL;
        Bf12G128Hdr hdr{static_cast<uint32_t>(st_chunk), n_tiles, novf, static_cast<uint32_t>(p2_bytes)};
        *reinterpret_cast<uint4*>(send2_c) = *reinterpret_cast<const uint4*>(&hdr);
      }
      if (p.ctrl_c32 != nullptr) {
        *reinterpret_cast<volatile uint64_t*>(&p.ctrl_c32[slot].p2_stripe_seq) = do_stripe ? (uint64_t)seq : 0ULL;
      }
      __threadfence_system();
      s8_publish2(p, slot, seq, p2_bytes);
    }
  };

  // Helper lambda: Per-peer overlapped P2 flag poll (CTA 0) + 6-CTA/peer parallel unpack (CTAs 1..42)
  auto run_p2_unpack = [&](int stg, bool is_final) {
    const unsigned long long seq = seq0 + (unsigned long long)stg;
    const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
    uint4* out_base = p.out + (long long)stg * st_nvec;
    const char* r2 = p.recv2 + (long long)slot * kW * p.max_chunk;

    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        unsigned int pending = ((1u << kW) - 1u) & ~(1u << p.rank);
        const unsigned long long t0 = now_ns();
        unsigned int it = 0u;
        while (pending != 0u) {
#pragma unroll
          for (int r = 0; r < kW; ++r) {
            if (pending & (1u << r)) {
              bool ok = (ld_acquire_sys(p.flags2 + (slot * kW + r) * kFlagStride) >= seq);
              if (ok && do_stripe && p.flags2_qp2 != nullptr) {
                ok = (ld_acquire_sys(p.flags2_qp2 + (slot * kW + r) * kFlagStride) >= seq);
              }
              if (ok) {
                st_release_sys_ull(&p.st->p2_peer_seq[r], seq);
                pending &= ~(1u << r);
              }
            }
          }
          if (pending != 0u) {
            s6_nanosleep(p.poll_sleep_ns ? p.poll_sleep_ns : 32u);
            if ((++it & 1023u) == 0u && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
              printf("rdma_ar c32 rank %d: timeout waiting for p2 flags (seq %llu, pending=0x%x)\n",
                     p.rank, (unsigned long long)seq, pending);
              r3_trap();
            }
          }
        }
        if (is_final) {
          st_relaxed_gpu(&p.st->ar_done_cnt, seq);
        }
      }
      __syncthreads();
      if (is_final) {
        if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
      }
    } else {
      const int peer_ord = ((int)blockIdx.x - 1) / 6;
      const int sub_cta = ((int)blockIdx.x - 1) % 6;
      if (peer_ord >= kW - 1) return;
      const int peer = (peer_ord < p.rank) ? peer_ord : (peer_ord + 1);
      if (threadIdx.x == 0) {
        if (ld_acquire_gpu(&p.st->p2_peer_seq[peer]) < seq) {
          const unsigned long long t0 = now_ns();
          unsigned int it = 0u;
          while (ld_acquire_gpu(&p.st->p2_peer_seq[peer]) < seq) {
            if ((++it & 1023u) == 0u && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
              printf("rdma_ar c32 rank %d: timeout waiting for p2_peer_seq[%d] (seq %llu)\n",
                     p.rank, peer, (unsigned long long)seq);
              r3_trap();
            }
          }
        }
        (void)ld_acquire_sys_ull(&p.st->p2_peer_seq[peer]);
      }
      __syncthreads();
      fence_acquire_sys();

      uint4* dst_p = out_base + (long long)peer * st_chunk;
      const char* src_p = r2 + (long long)peer * p.max_chunk;
      const uint32_t pw_start = (uint32_t)(sub_cta * 8 + warp_in_cta);
      for (uint32_t pair_idx = pw_start; pair_idx < n_pairs; pair_idx += 48u) {
        const long long i = (long long)pair_idx * 32LL + (long long)lane;
        if constexpr (kBf12P2) {
          dst_p[i] = bf12_g128_unpack_warp(src_p, pair_idx, n_tiles);
        } else {
          dst_p[i] = __ldcv(reinterpret_cast<const uint4*>(src_p) + i);
        }
      }
    }
  };

  if constexpr (kPipe2) {
    // Zero-bubble 2-stage software pipeline:
    //   P1(A) -> P1(B) [overlaps P1(A) wire] -> Reduce+P2(A) [overlaps P1(B) wire]
    //   -> Reduce+P2(B) [overlaps P2(A) wire] -> Unpack(A) [overlaps P2(B) wire] -> Unpack(B)
    run_p1(0);
    run_p1(1);
    run_reduce_p2(0);
    run_reduce_p2(1);
    run_p2_unpack(0, false);
    run_p2_unpack(1, true);
  } else {
    run_p1(0);
    run_reduce_p2(0);
    run_p2_unpack(0, true);
  }
}

// Standalone BF12-G128-OVF single-GPU SelfTest kernels
__global__ void __launch_bounds__(256, 2) selftest_bf12_g128_pack_kernel(
    const uint4* __restrict__ in_u4, char* __restrict__ dst_wire,
    uint32_t n_vec, unsigned int* __restrict__ ovf_cnt) {
  const uint32_t n_tiles = n_vec >> 4;
  const uint32_t n_pairs = n_vec >> 5;
  const int lane = (int)(threadIdx.x & 31u);
  const uint32_t wid = (uint32_t)blockIdx.x * 8u + (uint32_t)(threadIdx.x >> 5);
  const uint32_t n_warps = (uint32_t)gridDim.x * 8u;
  for (uint32_t pair_idx = wid; pair_idx < n_pairs; pair_idx += n_warps) {
    bf12_g128_pack_warp(in_u4[(size_t)pair_idx * 32u + (size_t)lane], pair_idx, n_tiles, dst_wire, ovf_cnt);
  }
}

__global__ void __launch_bounds__(256, 2) selftest_bf12_g128_unpack_kernel(
    const char* __restrict__ src_wire, uint4* __restrict__ out_u4, uint32_t n_vec) {
  const uint32_t n_tiles = n_vec >> 4;
  const uint32_t n_pairs = n_vec >> 5;
  const int lane = (int)(threadIdx.x & 31u);
  const uint32_t wid = (uint32_t)blockIdx.x * 8u + (uint32_t)(threadIdx.x >> 5);
  const uint32_t n_warps = (uint32_t)gridDim.x * 8u;
  for (uint32_t pair_idx = wid; pair_idx < n_pairs; pair_idx += n_warps) {
    out_u4[(size_t)pair_idx * 32u + (size_t)lane] = bf12_g128_unpack_warp(src_wire, pair_idx, n_tiles);
  }
}

// =================================================================================================
// c32-rdma-comm: Batched Drafter 2-Shot H64-Q4/Q6/Q8 AR + Fused Residual + RMSNorm (B=2..32!)
// =================================================================================================
// Exploits linearity of orthonormal H_64 (sum_r H_64(x_r) == H_64(sum_r x_r)):
//   - Zero intermediate IFWHT between Phase 1 and Phase 2!
//   - 576 threads = exact H/8 = 576 uint4 -> 1 full row per CTA pass, 72 exact 8-lane sub-warps!
//   - 100% cross-rank bit-deterministic (every rank sees identical post-P2 quantized H64 values).
//   - Target verifier outputs remain 100% 0-ULP bit-exact (drafter only!).
__device__ __forceinline__ float c32_block_sum_576(float v, float* smem) {
  const int lane = (int)(threadIdx.x & 31u);
  const int warp = (int)(threadIdx.x >> 5);  // 0..17 (18 warps)
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float out = 0.f;
  if (warp == 0) {
    out = (lane < 18) ? smem[lane] : 0.f;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) out += __shfl_xor_sync(0xffffffffu, out, m);
    if (lane == 0) smem[0] = out;
  }
  __syncthreads();
  return smem[0];
}

template <int kW, int kPub, int kBits>
__global__ void __launch_bounds__(kRowsThreads, 1) ar_kernel_draft_c32_h64_q_norm(Params p, DraftNormParams dn) {
  static_assert(kW == 8, "requires world == 8");
  static_assert(kBits == 4 || kBits == 6 || kBits == 8, "kBits must be 4, 6, or 8");
  constexpr long long kRowVecBf16 = 576LL;  // 4608 / 8 == blockDim.x!
  constexpr long long kRowBytes = (kBits == 4) ? 2448LL : ((kBits == 6) ? 3600LL : 4752LL);
  constexpr long long kRowVecQ = kRowBytes / 16LL;  // 153, 225, 297
  constexpr long long kScaleOff = (kBits == 4) ? 2304LL : ((kBits == 6) ? 3456LL : 4608LL);
  constexpr long long kHi2Off = 2304LL;
  constexpr float kClip = (kBits == 4) ? (0.90f / 7.5f) : ((kBits == 6) ? (0.96f / 31.5f) : (1.0f / 127.f));

  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  __shared__ float s_red[32];

  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    if (threadIdx.x == 0) early = s8_read_early(p);
    r6_gdc_wait();
  }
  if (threadIdx.x == 0) {
    s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1ULL;
    if constexpr ((kPub & kR6Pdl) == 0) early = s8_read_early(p);
  }
  __syncthreads();
  const unsigned long long seq = s_seq;
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }

  const int slot = static_cast<int>(seq % (unsigned long long)kSlots);
  const int M = dn.rows;
  const int rows_per_rank = M / kW;
  const long long chunk_q_bytes = (long long)rows_per_rank * kRowBytes;
  const bool do_stripe = (p.dual_qp != 0) && (chunk_q_bytes >= p.stripe_min_bytes);
  const long long rstride = p.recv_stride ? p.recv_stride : p.max_bytes;
  const int vi = (int)threadIdx.x;  // 0..575
  const int g = vi >> 3;            // 0..71 (64-elem H64 group)
  const int sub = vi & 7;

  auto quantize_h64_row = [&](float x[8], char* dst_row, float* out_deq) {
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(x[k]));
    const float g_amax = dec160_subwarp8_max(amax);
    const __half h_sc = __float2half_rn((g_amax > 0.f) ? (g_amax * kClip) : 0.f);
    const float sc = __half2float(h_sc);
    const float inv_sc = (sc > 0.f) ? (1.0f / sc) : 0.f;
    if (sub == 0) reinterpret_cast<__half*>(dst_row + kScaleOff)[g] = h_sc;
    if constexpr (kBits == 4) {
      uint32_t pk = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const uint32_t q = (uint32_t)__float2int_rn(fminf(fmaxf(x[k] * inv_sc + 7.5f, 0.f), 15.f));
        pk |= (q << (k * 4));
        if (out_deq) out_deq[k] = ((float)q - 7.5f) * sc;
      }
      reinterpret_cast<uint32_t*>(dst_row)[vi] = pk;
    } else if constexpr (kBits == 6) {
      uint32_t lo4 = 0u, hi2 = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const uint32_t q = (uint32_t)__float2int_rn(fminf(fmaxf(x[k] * inv_sc + 31.5f, 0.f), 63.f));
        lo4 |= ((q & 0xfu) << (k * 4));
        hi2 |= (((q >> 4) & 0x3u) << (k * 2));
        if (out_deq) out_deq[k] = ((float)q - 31.5f) * sc;
      }
      reinterpret_cast<uint32_t*>(dst_row)[vi] = lo4;
      reinterpret_cast<uint16_t*>(dst_row + kHi2Off)[vi] = (uint16_t)hi2;
    } else {
      uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int q0 = __float2int_rn(fminf(fmaxf(x[k] * inv_sc, -127.f), 127.f));
        const int q1 = __float2int_rn(fminf(fmaxf(x[k + 4] * inv_sc, -127.f), 127.f));
        w0 |= ((uint32_t)(uint8_t)(int8_t)q0 << (k * 8));
        w1 |= ((uint32_t)(uint8_t)(int8_t)q1 << (k * 8));
        if (out_deq) {
          out_deq[k] = (float)q0 * sc;
          out_deq[k + 4] = (float)q1 * sc;
        }
      }
      reinterpret_cast<uint2*>(dst_row)[vi] = make_uint2(w0, w1);
    }
  };

  auto apply_add_rmsnorm = [&](int global_row, float red_h64[8]) {
    dec160_fwht64_8t(red_h64);  // Inverse FWHT-64 -> standard domain
    uint4* res_row = dn.residual + (long long)global_row * kRowVecBf16;
    uint4* out_row = p.out + (long long)global_row * kRowVecBf16;
    float r_f[8];
    r7_unpack(res_row[vi], r_f);
    float ss = 0.f;
    float s[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      s[k] = red_h64[k] + r_f[k];
      ss += s[k] * s[k];
    }
    res_row[vi] = pack8(s);
    const float inv = rsqrtf(c32_block_sum_576(ss, s_red) / (float)dn.hidden + dn.eps);
    float gm[8];
    r7_unpack(dn.gamma[vi], gm);
#pragma unroll
    for (int k = 0; k < 8; ++k) s[k] = s[k] * inv * gm[k];
    out_row[vi] = pack8(s);
  };

  // Phase 1: All CTAs quantize ALL M rows into send[slot] (including own rank's rows for exact cross-rank parity!)
  char* send_slot = p.send + (long long)slot * p.max_bytes;
  for (int row = (int)blockIdx.x; row < M; row += (int)gridDim.x) {
    float x[8];
    r7_unpack(p.in[(long long)row * kRowVecBf16 + vi], x);
    dec160_fwht64_8t(x);
    quantize_h64_row(x, send_slot + (long long)row * kRowBytes, nullptr);
  }
  if (last_cta(&p.st->c32_arrive_a, &s_last)) {
    const uint64_t total_q_bytes = (uint64_t)M * (uint64_t)kRowBytes;
    const uint64_t chunk_q_vecs = (uint64_t)rows_per_rank * (uint64_t)kRowVecQ;
    c32_publish1<kPub & ~kR3Sbuf>(p, slot, seq, early, 0ULL, total_q_bytes, chunk_q_vecs,
                                  (uint64_t)kTwoShot | (do_stripe ? kModeP1Stripe : 0ULL));
  }

  // Wait Phase 1 flags
  c32_wait_p1_flags(p, slot, seq, do_stripe);

  // Phase 1 Reduce in H64 domain (exact rank order 0..7!) + Phase 2 Requant + Own-Rank Fused Residual+RMSNorm
  const char* rbase = p.recv + (long long)slot * kW * rstride;
  char* send2_c = p.send2 + (long long)slot * p.max_chunk;
  for (int rl = (int)blockIdx.x; rl < rows_per_rank; rl += (int)gridDim.x) {
    const int global_row = p.rank * rows_per_rank + rl;
    float sum_h[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // Load 8 peer scales via 8-lane subwarp cooperative load + shuffle
    float peer_sc = 0.f;
    {
      const char* src_s = (sub == p.rank)
                              ? (send_slot + (long long)global_row * kRowBytes)
                              : (rbase + (long long)sub * rstride + (long long)rl * kRowBytes);
      const uint16_t raw_h = (sub == p.rank)
                                 ? reinterpret_cast<const uint16_t*>(src_s + kScaleOff)[g]
                                 : __ldcv(reinterpret_cast<const uint16_t*>(src_s + kScaleOff) + g);
      peer_sc = __half2float(*reinterpret_cast<const __half*>(&raw_h));
    }
#pragma unroll
    for (int r = 0; r < kW; ++r) {
      const float sc = __shfl_sync(0xffffffffu, peer_sc, (((int)threadIdx.x & 31) & ~7) | r);
      const char* rp = (r == p.rank)
                           ? (send_slot + (long long)global_row * kRowBytes)
                           : (rbase + (long long)r * rstride + (long long)rl * kRowBytes);
      if constexpr (kBits == 4) {
        const uint32_t pk = (r == p.rank) ? reinterpret_cast<const uint32_t*>(rp)[vi]
                                          : __ldcv(reinterpret_cast<const uint32_t*>(rp) + vi);
#pragma unroll
        for (int k = 0; k < 8; ++k) sum_h[k] = fmaf((float)((pk >> (k * 4)) & 0xfu) - 7.5f, sc, sum_h[k]);
      } else if constexpr (kBits == 6) {
        const uint32_t lo = (r == p.rank) ? reinterpret_cast<const uint32_t*>(rp)[vi]
                                          : __ldcv(reinterpret_cast<const uint32_t*>(rp) + vi);
        const uint32_t hi = (r == p.rank) ? (uint32_t)reinterpret_cast<const uint16_t*>(rp + kHi2Off)[vi]
                                          : (uint32_t)__ldcv(reinterpret_cast<const uint16_t*>(rp + kHi2Off) + vi);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const uint32_t q6 = ((lo >> (k * 4)) & 0xfu) | (((hi >> (k * 2)) & 0x3u) << 4);
          sum_h[k] = fmaf((float)q6 - 31.5f, sc, sum_h[k]);
        }
      } else {
        const uint2 q8 = (r == p.rank) ? reinterpret_cast<const uint2*>(rp)[vi]
                                       : __ldcv(reinterpret_cast<const uint2*>(rp) + vi);
        const uint32_t w0 = q8.x, w1 = q8.y;
        sum_h[0] = fmaf((float)(int8_t)(w0 & 0xffu), sc, sum_h[0]);
        sum_h[1] = fmaf((float)(int8_t)((w0 >> 8) & 0xffu), sc, sum_h[1]);
        sum_h[2] = fmaf((float)(int8_t)((w0 >> 16) & 0xffu), sc, sum_h[2]);
        sum_h[3] = fmaf((float)(int8_t)((w0 >> 24) & 0xffu), sc, sum_h[3]);
        sum_h[4] = fmaf((float)(int8_t)(w1 & 0xffu), sc, sum_h[4]);
        sum_h[5] = fmaf((float)(int8_t)((w1 >> 8) & 0xffu), sc, sum_h[5]);
        sum_h[6] = fmaf((float)(int8_t)((w1 >> 16) & 0xffu), sc, sum_h[6]);
        sum_h[7] = fmaf((float)(int8_t)((w1 >> 24) & 0xffu), sc, sum_h[7]);
      }
    }
    float deq_own[8];
    quantize_h64_row(sum_h, send2_c + (long long)rl * kRowBytes, deq_own);
    apply_add_rmsnorm(global_row, deq_own);
  }

  if (last_cta(&p.st->c32_arrive2_a, &s_last)) {
    if (p.ctrl_c32 != nullptr) {
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl_c32[slot].p2_stripe_seq) = do_stripe ? (uint64_t)seq : 0ULL;
    }
    __threadfence_system();
    s8_publish2(p, slot, seq, (uint64_t)chunk_q_bytes);
  }

  // Phase 2: CTA 0 polls per-peer flags; CTAs 1..42 (6/peer) wake per-peer -> dequant + IFWHT + Residual + RMSNorm
  const char* r2 = p.recv2 + (long long)slot * kW * p.max_chunk;
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      unsigned int pending = ((1u << kW) - 1u) & ~(1u << p.rank);
      const unsigned long long t0 = now_ns();
      unsigned int it = 0u;
      while (pending != 0u) {
#pragma unroll
        for (int r = 0; r < kW; ++r) {
          if (pending & (1u << r)) {
            bool ok = (ld_acquire_sys(p.flags2 + (slot * kW + r) * kFlagStride) >= seq);
            if (ok && do_stripe && p.flags2_qp2 != nullptr) {
              ok = (ld_acquire_sys(p.flags2_qp2 + (slot * kW + r) * kFlagStride) >= seq);
            }
            if (ok) {
              st_release_sys_ull(&p.st->p2_peer_seq[r], seq);
              pending &= ~(1u << r);
            }
          }
        }
        if (pending != 0u) {
          s6_nanosleep(p.poll_sleep_ns ? p.poll_sleep_ns : 32u);
          if ((++it & 1023u) == 0u && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
            printf("rdma_ar draft_c32 rank %d: timeout waiting for p2 (seq %llu, pending=0x%x)\n",
                   p.rank, (unsigned long long)seq, pending);
            r3_trap();
          }
        }
      }
      st_relaxed_gpu(&p.st->ar_done_cnt, seq);
    }
    __syncthreads();
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();
  } else {
    const int peer_ord = ((int)blockIdx.x - 1) / 6;
    const int sub_cta = ((int)blockIdx.x - 1) % 6;
    if (peer_ord >= kW - 1) return;
    const int peer = (peer_ord < p.rank) ? peer_ord : (peer_ord + 1);
    if (threadIdx.x == 0) {
      if (ld_acquire_gpu(&p.st->p2_peer_seq[peer]) < seq) {
        const unsigned long long t0 = now_ns();
        unsigned int it = 0u;
        while (ld_acquire_gpu(&p.st->p2_peer_seq[peer]) < seq) {
          if ((++it & 1023u) == 0u && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
            printf("rdma_ar draft_c32 rank %d: timeout waiting for p2_peer_seq[%d]\n", p.rank, peer);
            r3_trap();
          }
        }
      }
      (void)ld_acquire_sys_ull(&p.st->p2_peer_seq[peer]);
    }
    __syncthreads();
    fence_acquire_sys();

    const char* src_peer = r2 + (long long)peer * p.max_chunk;
    for (int rl = sub_cta; rl < rows_per_rank; rl += 6) {
      const char* rp = src_peer + (long long)rl * kRowBytes;
      float sc_lane = 0.f;
      if (sub == 0) {
        const uint16_t raw_h = __ldcv(reinterpret_cast<const uint16_t*>(rp + kScaleOff) + g);
        sc_lane = __half2float(*reinterpret_cast<const __half*>(&raw_h));
      }
      const float sc = __shfl_sync(0xffffffffu, sc_lane, ((int)threadIdx.x & 31) & ~7);
      float deq[8];
      if constexpr (kBits == 4) {
        const uint32_t pk = __ldcv(reinterpret_cast<const uint32_t*>(rp) + vi);
#pragma unroll
        for (int k = 0; k < 8; ++k) deq[k] = ((float)((pk >> (k * 4)) & 0xfu) - 7.5f) * sc;
      } else if constexpr (kBits == 6) {
        const uint32_t lo = __ldcv(reinterpret_cast<const uint32_t*>(rp) + vi);
        const uint32_t hi = (uint32_t)__ldcv(reinterpret_cast<const uint16_t*>(rp + kHi2Off) + vi);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const uint32_t q6 = ((lo >> (k * 4)) & 0xfu) | (((hi >> (k * 2)) & 0x3u) << 4);
          deq[k] = ((float)q6 - 31.5f) * sc;
        }
      } else {
        const uint2 q8 = __ldcv(reinterpret_cast<const uint2*>(rp) + vi);
        const uint32_t w0 = q8.x, w1 = q8.y;
        deq[0] = (float)(int8_t)(w0 & 0xffu) * sc;
        deq[1] = (float)(int8_t)((w0 >> 8) & 0xffu) * sc;
        deq[2] = (float)(int8_t)((w0 >> 16) & 0xffu) * sc;
        deq[3] = (float)(int8_t)((w0 >> 24) & 0xffu) * sc;
        deq[4] = (float)(int8_t)(w1 & 0xffu) * sc;
        deq[5] = (float)(int8_t)((w1 >> 8) & 0xffu) * sc;
        deq[6] = (float)(int8_t)((w1 >> 16) & 0xffu) * sc;
        deq[7] = (float)(int8_t)((w1 >> 24) & 0xffu) * sc;
      }
      apply_add_rmsnorm(peer * rows_per_rank + rl, deq);
    }
  }
}

template <int kPub, bool kBf12P1, bool kBf12P2, bool kPipe2>
void launch_c32_one(const Params& p, cudaStream_t stream) {
  if constexpr ((kPub & kR6Pdl) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)kC32Blocks);
    cfg.blockDim = dim3((unsigned)kC32Threads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_c32<8, kPub, kBf12P1, kBf12P2, kPipe2>, p));
  } else {
    ar_kernel_c32<8, kPub, kBf12P1, kBf12P2, kPipe2><<<kC32Blocks, kC32Threads, 0, stream>>>(p);
  }
}

template <int kPub>
void launch_c32_modes(const Params& p, bool bf12_p1, bool bf12_p2, bool pipe2, cudaStream_t stream) {
  if (bf12_p1 && bf12_p2) {
    if (pipe2) launch_c32_one<kPub, true, true, true>(p, stream);
    else launch_c32_one<kPub, true, true, false>(p, stream);
  } else if (!bf12_p1 && bf12_p2) {
    if (pipe2) launch_c32_one<kPub, false, true, true>(p, stream);
    else launch_c32_one<kPub, false, true, false>(p, stream);
  } else {
    if (pipe2) launch_c32_one<kPub, false, false, true>(p, stream);
    else launch_c32_one<kPub, false, false, false>(p, stream);
  }
}

void launch_c32_kernel(const Params& p, bool bf12_p1, bool bf12_p2, bool pipe2, cudaStream_t stream, int pub) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  const int m = pub & (B | kR6Pdl | kR6Trig | (bf12_p1 ? 0 : kR3Sbuf));
  switch (m) {
    case B: launch_c32_modes<B>(p, bf12_p1, bf12_p2, pipe2, stream); break;
    case B | kR6Pdl: launch_c32_modes<B | kR6Pdl>(p, bf12_p1, bf12_p2, pipe2, stream); break;
    case B | kR6Pdl | kR6Trig: launch_c32_modes<B | kR6Pdl | kR6Trig>(p, bf12_p1, bf12_p2, pipe2, stream); break;
    case B | kR3Sbuf: launch_c32_modes<B | kR3Sbuf>(p, bf12_p1, bf12_p2, pipe2, stream); break;
    case B | kR6Pdl | kR3Sbuf: launch_c32_modes<B | kR6Pdl | kR3Sbuf>(p, bf12_p1, bf12_p2, pipe2, stream); break;
    case B | kR6Pdl | kR6Trig | kR3Sbuf: launch_c32_modes<B | kR6Pdl | kR6Trig | kR3Sbuf>(p, bf12_p1, bf12_p2, pipe2, stream); break;
    default: throw std::runtime_error("rdma_ar c32: unsupported kernel mask " + std::to_string(m));
  }
}

template <int kPub, int kBits>
void launch_draft_c32_one(const Params& p, const DraftNormParams& dn, cudaStream_t stream) {
  if constexpr ((kPub & kR6Pdl) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)kC32Blocks);
    cfg.blockDim = dim3((unsigned)kRowsThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_draft_c32_h64_q_norm<8, kPub, kBits>, p, dn));
  } else {
    ar_kernel_draft_c32_h64_q_norm<8, kPub, kBits><<<kC32Blocks, kRowsThreads, 0, stream>>>(p, dn);
  }
}

void launch_draft_c32_h64_kernel(const Params& p, const DraftNormParams& dn, int bits, cudaStream_t stream, int pub) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  const int m = pub & (B | kR6Pdl | kR6Trig);
  auto disp = [&](auto tag_bits) {
    constexpr int kB = decltype(tag_bits)::value;
    switch (m) {
      case B: launch_draft_c32_one<B, kB>(p, dn, stream); break;
      case B | kR6Pdl: launch_draft_c32_one<B | kR6Pdl, kB>(p, dn, stream); break;
      case B | kR6Pdl | kR6Trig: launch_draft_c32_one<B | kR6Pdl | kR6Trig, kB>(p, dn, stream); break;
      default: throw std::runtime_error("rdma_ar draft_c32: unsupported mask " + std::to_string(m));
    }
  };
  if (bits == 4) disp(std::integral_constant<int, 4>{});
  else if (bits == 6) disp(std::integral_constant<int, 6>{});
  else if (bits == 8) disp(std::integral_constant<int, 8>{});
  else throw std::runtime_error("rdma_ar draft_c32: unsupported bits " + std::to_string(bits));
}

constexpr int kParW = 8;  // the only world size with a parallel-load instantiation (DSv4.1 TP8)

// g130: V2 has no meaning without a phase 2, so non-two-shot modes drop the bit instead of instantiating a duplicate kernel.
template <int kMode, bool kF32, int kPub>
void launch_pl(const Params& p, int blocks, cudaStream_t stream) {
  ar_kernel_pl<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~kPubV2))><<<blocks, kThreads, 0, stream>>>(p);
}

// r3-comm: P2 and V2 have no meaning outside two-shot, so those bits are dropped there (as launch_pl drops V2).
template <int kMode, bool kF32, int kPub>
void launch_r3_one(const Params& p, int blocks, cudaStream_t stream, const R7Proj& q) {
  // s7-v3: dedicated 1-warp (32-thread, 0-smem, 0-atomic) 16B inline all-gather (Markov inline AG!)
  if constexpr (kMode == kGather && (kPub & kR3Sbuf) == 0 && (kPub & kR7Proj) == 0) {
    if (p.inline_ag != 0 && p.nvec == 1 && p.inline_recv != nullptr && blocks == 1) {
      constexpr int kInlPub = kPub & ~(kPubV2 | kR3P2 | kR3S1);
      if constexpr ((kInlPub & kR6Pdl) != 0) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(1u);
        cfg.blockDim = dim3(32u);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_inline_ag<kParW, kInlPub>, p));
      } else {
        ar_kernel_inline_ag<kParW, kInlPub><<<1, 32, 0, stream>>>(p);
      }
      return;
    }
    if (p.piggyback == 0 && p.rows_ptr == nullptr && p.nvec >= 2 && p.nvec <= 8 && blocks == 1) {
      constexpr int kInlPub = kPub & ~(kPubV2 | kR3P2 | kR3S1);
      if constexpr ((kInlPub & kR6Pdl) != 0) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(1u);
        cfg.blockDim = dim3(32u);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_small_gather<kParW, kInlPub>, p));
      } else {
        ar_kernel_small_gather<kParW, kInlPub><<<1, 32, 0, stream>>>(p);
      }
      return;
    }
    if (p.piggyback == 0 && p.rows_ptr == nullptr && p.c32_med_gather != 0 && p.nvec >= 9 && p.nvec <= 1024) {
      constexpr int kMedPub = kPub & ~((int)kPubV2 | (int)kR3P2 | (int)kR3S1);
      if constexpr ((kMedPub & kR6Pdl) != 0) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(1u);
        cfg.blockDim = dim3(128u);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_med_gather<kParW, kMedPub>, p));
      } else {
        ar_kernel_med_gather<kParW, kMedPub><<<1, 128, 0, stream>>>(p);
      }
      return;
    }
  }
  if constexpr ((kPub & kR6Pdl) != 0) {
    // r6 C2: programmatic stream serialization; inside stream capture the kernel->kernel edge into this node becomes
    // a PROGRAMMATIC edge (device_test.py asserts it). Same grid, block, smem and Params as the <<<>>> launch.
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)blocks);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_r3<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~(kPubV2 | kR3P2)))>, p, q));
  } else {
    ar_kernel_r3<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~(kPubV2 | kR3P2)))>
        <<<blocks, kThreads, 0, stream>>>(p, q);
  }
}
// r7-comm2 HT13: the kR7Proj instantiations (bf16 one-shot / two-shot; every r6 mask without a send buffer)
template <int kMode, bool kF32>
void launch_r7(const Params& p, int blocks, cudaStream_t stream, int pub, const R7Proj& q) {
  if constexpr (!kF32 && (kMode == kOneShot || kMode == kTwoShot)) {
    constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim | kR7Proj;
    switch (pub) {
      case B: launch_r3_one<kMode, kF32, B>(p, blocks, stream, q); break;
      case B | kR3S1: launch_r3_one<kMode, kF32, B | kR3S1>(p, blocks, stream, q); break;
      case B | kR3P2: launch_r3_one<kMode, kF32, B | kR3P2>(p, blocks, stream, q); break;
      case B | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR3S1 | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl: launch_r3_one<kMode, kF32, B | kR6Pdl>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1 | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig | kR3S1 | kR3P2:
        launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1 | kR3P2>(p, blocks, stream, q);
        break;
      default: throw std::runtime_error("rdma_ar r7: unsupported projection kernel mask " + std::to_string(pub));
    }
  } else {
    throw std::runtime_error("rdma_ar r7: the projection tail is bf16 all-reduce only");
  }
}
template <int kMode, bool kF32>
void launch_r3(const Params& p, int blocks, cudaStream_t stream, int pub, const R7Proj& q) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  if ((pub & kR7Proj) != 0) {
    launch_r7<kMode, kF32>(p, blocks, stream, pub, q);
    return;
  }
  switch (pub) {
    case B: launch_r3_one<kMode, kF32, B>(p, blocks, stream, q); break;
    case B | kR3S1: launch_r3_one<kMode, kF32, B | kR3S1>(p, blocks, stream, q); break;
    case B | kR3P2: launch_r3_one<kMode, kF32, B | kR3P2>(p, blocks, stream, q); break;
    case B | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR3S1 | kR3P2>(p, blocks, stream, q); break;
    case B | kR3Sbuf: launch_r3_one<kMode, kF32, B | kR3Sbuf>(p, blocks, stream, q); break;
    case B | kR3Sbuf | kR3P2: launch_r3_one<kMode, kF32, B | kR3Sbuf | kR3P2>(p, blocks, stream, q); break;
    // r6-comm C2 (+ s5 PDL+Sbuf)
    case B | kR6Pdl: launch_r3_one<kMode, kF32, B | kR6Pdl>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3P2>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1 | kR3P2>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3Sbuf: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3Sbuf>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3Sbuf | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3Sbuf | kR3P2>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3P2>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig | kR3S1 | kR3P2:
      launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1 | kR3P2>(p, blocks, stream, q);
      break;
    case B | kR6Pdl | kR6Trig | kR3Sbuf: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3Sbuf>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig | kR3Sbuf | kR3P2:
      launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3Sbuf | kR3P2>(p, blocks, stream, q);
      break;
    default: throw std::runtime_error("rdma_ar r3: unsupported kernel mask " + std::to_string(pub));
  }
}

// Launch either kernel with the live grid rule: ceil(nvec / vec_per_cta) CTAs, clamped to [1, 16].
template <int kMode, bool kF32>
void launch_kernel(const Params& p, long long vec_per_cta, bool par, cudaStream_t stream, int pub = 0,
                   const R7Proj& q = R7Proj{nullptr, nullptr, 0}, long long max_ctas = 16LL) {
  long long blocks = (p.nvec + vec_per_cta - 1) / vec_per_cta;
  blocks = std::max<long long>(1LL, std::min<long long>(blocks, max_ctas));
  if ((pub & kR7Proj) != 0 && (pub & kR3Trim) == 0) throw std::runtime_error("rdma_ar r7: projection needs the r3 kernels");
  if ((pub & kR3Trim) != 0) {  // r3-comm: only with DSV41_RDMA_TRIM=1 (callers check par, world 8 and g130 mask 7)
    launch_r3<kMode, kF32>(p, (int)blocks, stream, pub, q);
    CHECK_CUDA(cudaGetLastError());
    return;
  }
  if (par && p.world == kParW) {
    switch (pub & 7) {
      case 0: launch_pl<kMode, kF32, 0>(p, (int)blocks, stream); break;
      case 1: launch_pl<kMode, kF32, 1>(p, (int)blocks, stream); break;
      case 2: launch_pl<kMode, kF32, 2>(p, (int)blocks, stream); break;
      case 3: launch_pl<kMode, kF32, 3>(p, (int)blocks, stream); break;
      case 4: launch_pl<kMode, kF32, 4>(p, (int)blocks, stream); break;
      case 5: launch_pl<kMode, kF32, 5>(p, (int)blocks, stream); break;
      case 6: launch_pl<kMode, kF32, 6>(p, (int)blocks, stream); break;
      default: launch_pl<kMode, kF32, 7>(p, (int)blocks, stream); break;
    }
  } else {
    ar_kernel<kMode, kF32><<<(int)blocks, kThreads, 0, stream>>>(p);
  }
  CHECK_CUDA(cudaGetLastError());
}

int env_parload_default() {
  const char* e = getenv("RDMA_AR_PARLOAD");
  return e ? (atoi(e) != 0) : 1;  // this .so exists to run the parallel-load kernels
}

// g130: unset or 0 == off
bool env_flag(const char* name) {
  const char* e = getenv(name);
  return e && atoi(e) != 0;
}
int env_pub_mask() {
  return (env_flag("RDMA_AR_PUB_V1") ? kPubV1 : 0) | (env_flag("RDMA_AR_PUB_V2") ? kPubV2 : 0) |
         (env_flag("RDMA_AR_HYG_POLL") ? kPollThin : 0);
}
// r3-comm: one parser for the r3 flags, identical to rdma_all_reduce.py's (review N4): after trimming blanks and
// lower-casing, unset / "" / 0 / off / false / no = off, 1 / on / true / yes = on, anything else = an error (never a
// silent off). Integers: decimal digits only, in range, else an error. Read only when an r3 flag is on, except the two
// on/off flags themselves.
std::string env_r3_norm(const char* name) {
  const char* e = getenv(name);
  std::string v = e ? e : "";
  const auto a = v.find_first_not_of(" \t\r\n\v\f"), b = v.find_last_not_of(" \t\r\n\v\f");
  v = a == std::string::npos ? std::string() : v.substr(a, b - a + 1);
  for (auto& c : v) c = (char)std::tolower((unsigned char)c);
  return v;
}
bool env_r3_flag(const char* name) {
  const std::string v = env_r3_norm(name);
  if (v.empty() || v == "0" || v == "off" || v == "false" || v == "no") return false;
  if (v == "1" || v == "on" || v == "true" || v == "yes") return true;
  throw std::runtime_error(std::string("rdma_ar r3: ") + name + "='" + v + "' is not 0/1/off/on/false/true/no/yes");
}
long long env_r3_int(const char* name, long long dflt, long long lo, long long hi) {
  const std::string v = env_r3_norm(name);
  if (v.empty()) return dflt;
  if (v.size() > 12 || v.find_first_not_of("0123456789") != std::string::npos || std::stoll(v) < lo || std::stoll(v) > hi)
    throw std::runtime_error(std::string("rdma_ar r3: ") + name + "='" + v + "' is not an integer in [" +
                             std::to_string(lo) + ", " + std::to_string(hi) + "]");
  return std::stoll(v);
}
// 1 = DSV41_RDMA_TRIM, 2 = DSV41_RDMA_SENDBUF
int env_r3_mask() { return (env_r3_flag("DSV41_RDMA_TRIM") ? 1 : 0) | (env_r3_flag("DSV41_RDMA_SENDBUF") ? 2 : 0); }
long long env_r3_s1_bytes() { return env_r3_int("DSV41_RDMA_TRIM_S1_BYTES", 24576, 0, 1LL << 30); }
long long env_r3_sbuf_mb() { return env_r3_int("DSV41_RDMA_SENDBUF_MB", 64, 1, 1024); }
// r6-comm C2: DSV41_RDMA_PDL. Same normalisation as the r3 flags; off values as env_r3_flag; 1 / on / true / yes / in =
// PDL launch (in-edge); trig = PDL launch + early trigger. Anything else is a startup error. rdma_all_reduce.py has the
// same parser.
int env_r6_pdl() {
  const std::string v = env_r3_norm("DSV41_RDMA_PDL");
  if (v.empty() || v == "0" || v == "off" || v == "false" || v == "no") return 0;
  if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "in") return kR6Pdl;
  if (v == "trig") return kR6Pdl | kR6Trig;
  throw std::runtime_error("rdma_ar r6: DSV41_RDMA_PDL='" + v + "' is not 0/1/off/on/false/true/no/yes/in/trig");
}

// r7-comm2 HT13: DSV41_RDMA_PROJ, the r3 on/off parser (unset / 0 / off / false / no = off; 1 / on / true / yes = on)
bool env_r7_proj() { return env_r3_flag("DSV41_RDMA_PROJ"); }

// r7-comm2 HT13: argument checks shared by RdmaAR and SelfTest; returns the rows
int64_t r7_check(const torch::Tensor& x, const torch::Tensor& out, const torch::Tensor& d, const torch::Tensor& y) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() % kR7H == 0,
              "rdma_ar r7: projection needs a contiguous bf16 CUDA [rows, ", kR7H, "] input");
  const int64_t rows = x.numel() / kR7H;
  TORCH_CHECK(rows >= 1 && rows <= kR7MaxRows, "rdma_ar r7: projection rows ", rows, " not in [1, ", kR7MaxRows, "]");
  TORCH_CHECK(d.is_cuda() && d.scalar_type() == at::kFloat && d.is_contiguous() && d.numel() == kR7H &&
                  d.get_device() == x.get_device() && ((uintptr_t)d.data_ptr() % 16) == 0,
              "rdma_ar r7: direction must be a 16-byte aligned contiguous fp32 [", kR7H, "] on the input's device");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.is_contiguous() && y.numel() == x.numel() &&
                  y.get_device() == x.get_device() && ((uintptr_t)y.data_ptr() % 16) == 0,
              "rdma_ar r7: y must be a 16-byte aligned contiguous bf16 tensor like the input");
  const auto yb = (uintptr_t)y.data_ptr(), ob = (uintptr_t)out.data_ptr(), xb = (uintptr_t)x.data_ptr();
  const auto nb = (uintptr_t)(x.numel() * 2);
  TORCH_CHECK((yb + nb <= ob || ob + nb <= yb) && (yb + nb <= xb || xb + nb <= yb),
              "rdma_ar r7: y must not overlap the input or the output");
  return rows;
}

struct Hca {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_td* td = nullptr;         // s6: lock-free per-HCA thread domain (no mlx5_spin_lock!)
  ibv_pd* parent_pd = nullptr;  // s6: parent domain binding pd + td
  ibv_cq* cq = nullptr;
  bool cq_ex = false;           // s7-v2: true when ibv_create_cq_ex + parent_domain (lock-free CQ) succeeded
  ibv_mr* mr = nullptr;         // strict-ordering MR (RO=0) for completion flags & metadata
  ibv_mr* mr_ro = nullptr;      // c32: relaxed-ordering MR (RO=1) for bulk payloads (fallback = mr)
  ibv_gid gid{};
  ibv_mtu mtu = IBV_MTU_1024;
  uint32_t max_inline = 256;
};

struct Peer {
  int hca = -1;
  ibv_qp* qp = nullptr;
  ibv_qp_ex* qpx = nullptr;     // s6: fast-path ibv_wr_* interface (no ibv_send_wr linked-list parsing)
  uint64_t remote_base = 0;
  uint32_t rkey = 0;
  uint32_t rkey_ro = 0;         // c32: payload rkey (mr_ro if pci_ro, else rkey)
  uint64_t posts = 0;
  // c32: secondary RC QP on opposite PCIe root (hca2 = hca ^ 1) for 50/50 dual-root striping
  int hca2 = -1;
  ibv_qp* qp2 = nullptr;
  ibv_qp_ex* qpx2 = nullptr;
  uint32_t rkey2 = 0;
  uint32_t rkey2_ro = 0;
  uint64_t posts2 = 0;
};

class RdmaAR {
 public:
  RdmaAR(int rank, int world, int64_t max_bytes, int64_t max_reduce_bytes, std::vector<std::string> hca_names,
         int gid_index, int ib_port, int proxy_core, int64_t timeout_ms, int device)
      : rank_(rank), world_(world), max_bytes_(max_bytes), gid_index_(gid_index), ib_port_(ib_port),
        proxy_core_(proxy_core), timeout_ns_(timeout_ms * 1000000LL), device_(device) {
    if (max_bytes % 16 || max_reduce_bytes % 16) throw std::runtime_error("rdma_ar: sizes must be multiples of 16");
    if (world < 2 || world > kThreads) throw std::runtime_error("rdma_ar: bad world size");
    max_reduce_bytes_ = std::min<int64_t>(max_reduce_bytes, max_bytes);
    max_chunk_ = align((size_t)((max_reduce_bytes_ / 16 + world - 1) / world) * 16, 64);
    // c32: room for worst-case BF12-G128-OVF metadata (16B hdr + 2B/256B desc + 128B margin)
    const size_t bf12_extra = c32_kernel_ ? (align((max_chunk_ / 256 + 1) * 2, 16) + 128) : 64;
    slot_chunk_ = align(max_chunk_ + bf12_extra, 64);
    slot_bytes_ = align(std::max<size_t>((size_t)max_bytes_ + 64,
                                         c32_kernel_ ? ((size_t)world * slot_chunk_) : 0u),
                        64);
    if (recv_decouple_) {
      const int64_t mg_env = getenv("RDMA_AR_MAX_GATHER_BYTES") ? atoll(getenv("RDMA_AR_MAX_GATHER_BYTES")) : (1024LL * 1024LL);
      const int64_t ts_env = getenv("RDMA_AR_TWOSHOT_MIN_BYTES") ? atoll(getenv("RDMA_AR_TWOSHOT_MIN_BYTES")) : 32768LL;
      const size_t need_r = std::max<size_t>(
          slot_chunk_,
          std::max<size_t>((size_t)std::max<int64_t>(0, mg_env) + 64u,
                           (size_t)std::max<int64_t>(36864, ts_env) + 64u));
      recv_stride_ = std::min<size_t>(slot_bytes_, align(need_r, 64));
    } else {
      recv_stride_ = slot_bytes_;
    }
    if (dual_qp_ && (!multi_proxy_ || hca_names.size() < 2 || (hca_names.size() % 2 != 0))) {
      throw std::runtime_error("rdma_ar c32: RDMA_AR_DUAL_QP=1 requires RDMA_AR_MULTI_PROXY=1 and even HCA count >= 2");
    }
    if (recv_decouple_ && (r3_ & 1) == 0) {
      throw std::runtime_error("rdma_ar c32: RDMA_AR_RECV_DECOUPLE=1 requires DSV41_RDMA_TRIM=1");
    }
    c10::cuda::CUDAGuard guard(device_);
    // one host-pinned arena: [ctrl | flags | flags2 | flags_qp2 | flags2_qp2 | send_flags | inline_recv | send | send2 | recv | recv2]
    const size_t fl = align(sizeof(uint64_t) * kFlagStride * kSlots * world, 4096);
    flags_off_ = 4096;
    flags2_off_ = flags_off_ + fl;
    flags_qp2_off_ = flags2_off_ + fl;
    flags2_qp2_off_ = flags_qp2_off_ + fl;
    send_flag_off_ = flags2_qp2_off_ + fl; // 2 * kSlots * 64B (registered 8B flag source for 2-SGE piggyback WRs)
    inline_recv_off_ = send_flag_off_ + 4096;
    const size_t inl_sz = align(sizeof(InlineSlot) * kSlots * world, 4096);
    send_off_ = inline_recv_off_ + inl_sz;
    send2_off_ = send_off_ + align(kSlots * slot_bytes_, 4096);
    recv_off_ = send2_off_ + align(kSlots * slot_chunk_, 4096);
    recv2_off_ = recv_off_ + align((size_t)kSlots * world * recv_stride_, 4096);
    arena_bytes_ = recv2_off_ + align((size_t)kSlots * world * slot_chunk_, 4096);
    // r3-comm: fail closed at construction (rdma_all_reduce.py turns a construction error into a refusal to start)
    if (env_flag("RDMA_AR_PIGGYBACK_FLAG"))
      throw std::runtime_error("rdma_ar s6: RDMA_AR_PIGGYBACK_FLAG=1 is disabled (unsafe under variable message sizes "
                               "and ConnectX-7 PCIe RO); use RDMA_AR_TWO_PASS_POST=1");
    if (r3_ == 2) throw std::runtime_error("rdma_ar r3: DSV41_RDMA_SENDBUF=1 needs DSV41_RDMA_TRIM=1");
    if (r6_pdl_ != 0 && (r3_ & 1) == 0)  // r6-comm C2: PDL lives in the r3 kernels only
      throw std::runtime_error("rdma_ar r6: DSV41_RDMA_PDL needs DSV41_RDMA_TRIM=1");
    if (r7_proj_ && (r3_ & 1) == 0)  // r7-comm2 HT13: the projection tail lives in the r3 kernels only
      throw std::runtime_error("rdma_ar r7: DSV41_RDMA_PROJ needs DSV41_RDMA_TRIM=1");
    if (level_db_ && (r3_ & 1) == 0)
      throw std::runtime_error("rdma_ar s8: RDMA_AR_LEVEL_DB=1 needs DSV41_RDMA_TRIM=1");
    if (r3_ != 0 && (pub_mask_ != (kPubV1 | kPubV2 | kPollThin) || !parload()))
      throw std::runtime_error("rdma_ar r3: DSV41_RDMA_TRIM needs the parallel-load kernels, world 8 and RDMA_AR_PUB_V1=1 "
                               "RDMA_AR_PUB_V2=1 RDMA_AR_HYG_POLL=1");
    if ((r3_ & 2) != 0) {  // r3 G26 (DSV41_RDMA_SENDBUF=1): registered send-buffer region at the end of the arena
      sbuf_off_ = align(arena_bytes_, 4096);
      sbuf_bytes_ = (size_t)env_r3_sbuf_mb() << 20;
      arena_bytes_ = sbuf_off_ + sbuf_bytes_;
      r3_sbuf_ = true;
    }
    // Back the arena with 2 MB transparent huge pages (THP is in madvise mode on DGX OS, so a user
    // process may ask for it) and pin it with cudaHostRegister. cudaHostAlloc takes whatever pages
    // the allocator has; on a fragmented node that is 4 KB pages, and the GPU's staging copies and
    // peer reads then miss the TLB constantly (measured 3x slower stage/reduce on such ranks, and
    // every other rank waits for them). RDMA_AR_THP=0 falls back to cudaHostAlloc.
    const size_t kHuge = 2u << 20;
    arena_bytes_ = align(arena_bytes_, kHuge);
    const char* thp = getenv("RDMA_AR_THP");
    if (!thp || atoi(thp) != 0) {
      void* raw = mmap(nullptr, arena_bytes_ + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (raw != MAP_FAILED) {
        char* a = reinterpret_cast<char*>(align((size_t)(uintptr_t)raw, kHuge));
        madvise(a, arena_bytes_, MADV_HUGEPAGE);
        // vLLM/inductor may fork helper processes after init; keep the RDMA-registered arena out of
        // children so copy-on-write can never move the parent's pinned pages (DSv4.1 addition).
        madvise(a, arena_bytes_, MADV_DONTFORK);
        memset(a, 0, arena_bytes_);  // fault in now, as huge pages where the kernel can
        if (cudaHostRegister(a, arena_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
          arena_ = a;
          arena_registered_ = true;
        } else {
          cudaGetLastError();
          munmap(raw, arena_bytes_ + kHuge);
        }
      }
    }
    if (!arena_registered_) {
      CHECK_CUDA(cudaHostAlloc(&arena_, arena_bytes_, cudaHostAllocMapped | cudaHostAllocPortable));
      memset(arena_, 0, arena_bytes_);
    }
    CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&arena_dev_), arena_, 0));
    thp_bytes_ = huge_backed_bytes(arena_, arena_bytes_);
    CHECK_CUDA(cudaMalloc(&st_, sizeof(DevState)));
    CHECK_CUDA(cudaMemset(st_, 0, sizeof(DevState)));
    CHECK_CUDA(cudaMalloc(&reduced_buf_, 8u * 576u * sizeof(uint4)));
    CHECK_CUDA(cudaMemset(reduced_buf_, 0, 8u * 576u * sizeof(uint4)));
    CHECK_CUDA(cudaMalloc(&bf12_os_scratch_, 8u * 2304u * sizeof(uint4)));
    CHECK_CUDA(cudaMemset(bf12_os_scratch_, 0, 8u * 2304u * sizeof(uint4)));
    ctrl_ = reinterpret_cast<Ctrl*>(arena_);
    ctrl_slots_ = reinterpret_cast<CtrlSlot*>(reinterpret_cast<char*>(arena_) + kCtrlSlotsOff);
    ctrl_aux_ = reinterpret_cast<CtrlAux*>(reinterpret_cast<char*>(arena_) + kCtrlAuxOff);
    ctrl_c32_ = reinterpret_cast<CtrlC32Slot*>(reinterpret_cast<char*>(arena_) + kCtrlC32Off);

    int ndev = 0;
    ibv_device** list = ibv_get_device_list(&ndev);
    CHECK_IB(list != nullptr, "ibv_get_device_list");
    for (auto& name : hca_names) {
      Hca h;
      for (int i = 0; i < ndev; ++i)
        if (name == ibv_get_device_name(list[i])) h.ctx = ibv_open_device(list[i]);
      CHECK_IB(h.ctx != nullptr, "open device " + name);
      h.pd = ibv_alloc_pd(h.ctx);
      CHECK_IB(h.pd, "alloc_pd");
      // s6: allocate lock-free Thread Domain + Parent Domain per HCA (eliminates mlx5_spin_lock on SQ/CQ!)
      if (use_qp_ex_) {
        ibv_td_init_attr tda{};
        h.td = ibv_alloc_td(h.ctx, &tda);
        if (h.td) {
          ibv_parent_domain_init_attr pda{};
          pda.pd = h.pd;
          pda.td = h.td;
          pda.comp_mask = 0;
          h.parent_pd = ibv_alloc_parent_domain(h.ctx, &pda);
        }
      }
      // s7: bind CQ to lock-free parent_domain (eliminates mlx5_spin_lock in ibv_poll_cq!), with safe fallback
      ibv_cq* cq_ptr = nullptr;
      if (use_qp_ex_ && h.parent_pd) {
        ibv_cq_init_attr_ex cqa{};
        cqa.cqe = 4096;
        cqa.cq_context = nullptr;
        cqa.channel = nullptr;
        cqa.comp_vector = 0;
        cqa.wc_flags = 0;
        cqa.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD;
        cqa.parent_domain = h.parent_pd;
        ibv_cq_ex* cqx = ibv_create_cq_ex(h.ctx, &cqa);
        if (cqx) {
          cq_ptr = ibv_cq_ex_to_cq(cqx);
          h.cq_ex = true;
        }
      }
      if (!cq_ptr) cq_ptr = ibv_create_cq(h.ctx, 4096, nullptr, nullptr, 0);
      h.cq = cq_ptr;
      CHECK_IB(h.cq, "create_cq");
      h.mr = ibv_reg_mr(h.pd, arena_, arena_bytes_, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      CHECK_IB(h.mr, "reg_mr " + name);
      h.mr_ro = h.mr;
#ifdef IBV_ACCESS_RELAXED_ORDERING
      if (pci_ro_) {
        ibv_mr* ro = ibv_reg_mr(
            h.pd, arena_, arena_bytes_,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING);
        if (ro) h.mr_ro = ro;
      }
#endif
      CHECK_IB(ibv_query_gid(h.ctx, ib_port_, gid_index_, &h.gid) == 0, "query_gid");
      ibv_port_attr pa{};
      CHECK_IB(ibv_query_port(h.ctx, ib_port_, &pa) == 0, "query_port");
      h.mtu = pa.active_mtu;
      hcas_.push_back(h);
    }
    ibv_free_device_list(list);

    peers_.resize(world);
    for (int p = 0; p < world; ++p) {
      if (p == rank) continue;
      Peer& pr = peers_[p];
      pr.hca = (rank + p) % (int)hcas_.size();  // both ends of a pair agree on the HCA
      Hca& h = hcas_[pr.hca];
      const uint32_t want_inline = 256;
      if (use_qp_ex_) {
        ibv_qp_init_attr_ex iax{};
        iax.send_cq = h.cq;
        iax.recv_cq = h.cq;
        iax.qp_type = IBV_QPT_RC;
        iax.cap.max_send_wr = 512;
        iax.cap.max_recv_wr = 1;
        iax.cap.max_send_sge = 2; // 2 SGEs for 64B BlueFlame piggyback-flag RDMA WRITE
        iax.cap.max_recv_sge = 1;
        iax.cap.max_inline_data = want_inline;
        iax.sq_sig_all = 0;
        iax.pd = h.parent_pd ? h.parent_pd : h.pd;
        iax.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        iax.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
        pr.qp = ibv_create_qp_ex(h.ctx, &iax);
        if (!pr.qp) {
          iax.cap.max_inline_data = 64;
          pr.qp = ibv_create_qp_ex(h.ctx, &iax);
        }
        if (pr.qp) {
          pr.qpx = ibv_qp_to_qp_ex(pr.qp);
          h.max_inline = std::min(h.max_inline, std::min(iax.cap.max_inline_data, 256u));
        }
      }
      if (!pr.qp) {
        ibv_qp_init_attr ia{};
        ia.send_cq = h.cq;
        ia.recv_cq = h.cq;
        ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr = 512;
        ia.cap.max_recv_wr = 1;
        ia.cap.max_send_sge = 2;
        ia.cap.max_recv_sge = 1;
        ia.cap.max_inline_data = want_inline;
        ia.sq_sig_all = 0;
        pr.qp = ibv_create_qp(h.pd, &ia);
        if (!pr.qp) {
          ia.cap.max_inline_data = 64;
          pr.qp = ibv_create_qp(h.pd, &ia);
        }
        CHECK_IB(pr.qp, "create_qp");
        h.max_inline = std::min(h.max_inline, std::min(ia.cap.max_inline_data, 256u));
      }
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT;
      a.pkey_index = 0;
      a.port_num = ib_port_;
      a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
      CHECK_IB(ibv_modify_qp(pr.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
               "qp INIT");
      if (dual_qp_) {
        pr.hca2 = pr.hca ^ 1;  // symmetric opposite PCIe root (0<->1, 2<->3)
        Hca& h2 = hcas_[pr.hca2];
        if (use_qp_ex_) {
          ibv_qp_init_attr_ex iax2{};
          iax2.send_cq = h2.cq;
          iax2.recv_cq = h2.cq;
          iax2.qp_type = IBV_QPT_RC;
          iax2.cap.max_send_wr = 512;
          iax2.cap.max_recv_wr = 1;
          iax2.cap.max_send_sge = 2;
          iax2.cap.max_recv_sge = 1;
          iax2.cap.max_inline_data = want_inline;
          iax2.sq_sig_all = 0;
          iax2.pd = h2.parent_pd ? h2.parent_pd : h2.pd;
          iax2.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
          iax2.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
          pr.qp2 = ibv_create_qp_ex(h2.ctx, &iax2);
          if (!pr.qp2) {
            iax2.cap.max_inline_data = 64;
            pr.qp2 = ibv_create_qp_ex(h2.ctx, &iax2);
          }
          if (pr.qp2) pr.qpx2 = ibv_qp_to_qp_ex(pr.qp2);
        }
        if (!pr.qp2) {
          ibv_qp_init_attr ia2{};
          ia2.send_cq = h2.cq;
          ia2.recv_cq = h2.cq;
          ia2.qp_type = IBV_QPT_RC;
          ia2.cap.max_send_wr = 512;
          ia2.cap.max_recv_wr = 1;
          ia2.cap.max_send_sge = 2;
          ia2.cap.max_recv_sge = 1;
          ia2.cap.max_inline_data = want_inline;
          ia2.sq_sig_all = 0;
          pr.qp2 = ibv_create_qp(h2.pd, &ia2);
          if (!pr.qp2) {
            ia2.cap.max_inline_data = 64;
            pr.qp2 = ibv_create_qp(h2.pd, &ia2);
          }
          CHECK_IB(pr.qp2, "create_qp2");
        }
        CHECK_IB(ibv_modify_qp(pr.qp2, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                 "qp2 INIT");
      }
    }
  }

  ~RdmaAR() {
    stop_.store(true);
    if (proxy_.joinable()) proxy_.join();
    if (proxy2_.joinable()) proxy2_.join();
    // QPs/MRs are left to process teardown; the arena may still be referenced by in-flight graphs.
  }

  // [gid_hi, gid_lo] per HCA, then rkey per HCA, then arena addr, then qpn per rank (0 for self)
  std::vector<int64_t> local_info() const {
    std::vector<int64_t> v;
    for (auto& h : hcas_) {
      uint64_t hi, lo;
      memcpy(&hi, h.gid.raw, 8);
      memcpy(&lo, h.gid.raw + 8, 8);
      v.push_back((int64_t)hi);
      v.push_back((int64_t)lo);
    }
    for (auto& h : hcas_) v.push_back((int64_t)h.mr->rkey);
    v.push_back((int64_t)(uintptr_t)arena_);
    for (int p = 0; p < world_; ++p) v.push_back(p == rank_ ? 0 : (int64_t)peers_[p].qp->qp_num);
    if (pci_ro_ || dual_qp_) {
      for (auto& h : hcas_) v.push_back((int64_t)h.mr_ro->rkey);
    }
    if (dual_qp_) {
      for (int p = 0; p < world_; ++p) v.push_back(p == rank_ ? 0 : (int64_t)peers_[p].qp2->qp_num);
    }
    return v;
  }

  void connect(const std::vector<std::vector<int64_t>>& infos) {
    const int nh = (int)hcas_.size();
    const size_t base_len = (size_t)(3 * nh + 1 + world_);
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      Peer& pr = peers_[p];
      const auto& r = infos[p];
      Hca& h = hcas_[pr.hca];
      ibv_gid rgid{};
      uint64_t hi = (uint64_t)r[2 * pr.hca], lo = (uint64_t)r[2 * pr.hca + 1];
      memcpy(rgid.raw, &hi, 8);
      memcpy(rgid.raw + 8, &lo, 8);
      pr.rkey = (uint32_t)r[2 * nh + pr.hca];
      pr.rkey_ro = ((pci_ro_ || dual_qp_) && r.size() >= base_len + (size_t)nh)
                       ? (uint32_t)r[base_len + (size_t)pr.hca]
                       : pr.rkey;
      pr.remote_base = (uint64_t)r[3 * nh];
      uint32_t rqpn = (uint32_t)r[3 * nh + 1 + rank_];

      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_RTR;
      a.path_mtu = h.mtu;
      a.dest_qp_num = rqpn;
      a.rq_psn = 0;
      a.max_dest_rd_atomic = 1;
      a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1;
      a.ah_attr.grh.dgid = rgid;
      a.ah_attr.grh.sgid_index = gid_index_;
      a.ah_attr.grh.hop_limit = 64;
      a.ah_attr.port_num = ib_port_;
      CHECK_IB(ibv_modify_qp(pr.qp, &a,
                             IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                 IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
               "qp RTR");
      ibv_qp_attr b{};
      b.qp_state = IBV_QPS_RTS;
      b.timeout = 14;
      b.retry_cnt = 7;
      b.rnr_retry = 7;
      b.sq_psn = 0;
      b.max_rd_atomic = 1;
      CHECK_IB(ibv_modify_qp(pr.qp, &b,
                             IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                 IBV_QP_MAX_QP_RD_ATOMIC) == 0,
               "qp RTS");

      if (dual_qp_) {
        Hca& h2 = hcas_[pr.hca2];
        ibv_gid rgid2{};
        uint64_t hi2 = (uint64_t)r[2 * pr.hca2], lo2 = (uint64_t)r[2 * pr.hca2 + 1];
        memcpy(rgid2.raw, &hi2, 8);
        memcpy(rgid2.raw + 8, &lo2, 8);
        pr.rkey2 = (uint32_t)r[2 * nh + pr.hca2];
        pr.rkey2_ro = (r.size() >= base_len + (size_t)nh) ? (uint32_t)r[base_len + (size_t)pr.hca2] : pr.rkey2;
        const size_t qp2_off = base_len + (size_t)nh;
        TORCH_CHECK(r.size() >= qp2_off + (size_t)world_, "rdma_ar c32: peer missing dual_qp info");
        uint32_t rqpn2 = (uint32_t)r[qp2_off + (size_t)rank_];
        ibv_qp_attr a2 = a;
        a2.path_mtu = h2.mtu;
        a2.dest_qp_num = rqpn2;
        a2.ah_attr.grh.dgid = rgid2;
        CHECK_IB(ibv_modify_qp(pr.qp2, &a2,
                               IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                   IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
                 "qp2 RTR");
        CHECK_IB(ibv_modify_qp(pr.qp2, &b,
                               IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                   IBV_QP_MAX_QP_RD_ATOMIC) == 0,
                 "qp2 RTS");
      }
    }
    const bool multi = multi_proxy_ && nh >= 2;
    if (multi) {
      const int split = nh / 2;
      proxy_ = std::thread([this, split] { proxy_loop_worker(0, split, proxy_core_, true); });
      proxy2_ = std::thread([this, split, nh] { proxy_loop_worker(split, nh, proxy_core2_, false); });
    } else {
      proxy_ = std::thread([this, nh] { proxy_loop_worker(0, nh, proxy_core_, true); });
    }
  }

  // s6: Drafter fused all-reduce + residual add + RMSNorm (bf16 [8, 4608], world=8)
  torch::Tensor all_reduce_add_rmsnorm(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, double eps,
                                       bool use_int8) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                    x.size(0) == world_ && x.size(1) == 4608,
                "rdma_ar s6: draft_norm expects bf16 [world, 4608]");
    TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() &&
                    residual.sizes() == x.sizes(),
                "rdma_ar s6: bad residual");
    TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() &&
                    gamma.numel() == 4608,
                "rdma_ar s6: bad gamma");
    TORCH_CHECK(world_ == kParW && pub_mask_ == (kPubV1 | kPubV2 | kPollThin),
                "rdma_ar s6: draft_norm requires world=8 and pub_mask=7");
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    auto out = torch::empty_like(x);
    Params p = make_base_params(x, out, x.numel() * 2);
    DraftNormParams dn{reinterpret_cast<uint4*>(residual.data_ptr()),
                       reinterpret_cast<const uint4*>(gamma.data_ptr()), (float)eps, 4608, world_};
    bool sb = false;
    if (!use_int8 && (r3_ & 2) != 0 && !sbuf_allocs_.empty()) {
      const auto it = sbuf_allocs_.find((uintptr_t)x.data_ptr());
      sb = it != sbuf_allocs_.end() && (size_t)p.nvec * 16 <= it->second;
    }
    const int pub = pub_mask_ | kR3Trim | r6_pdl_ | (sb ? kR3Sbuf : 0);
    launch_draft_norm_kernel(p, dn, use_int8, at::cuda::getCurrentCUDAStream(), pub);
    CHECK_CUDA(cudaGetLastError());
    return out;
  }

  // s8v2-unified (dec160): Hadamard-64 sub-byte drafter AR+RMSNorm
  // mode_code: 0=bf16, 1=int8, 2=h64_q8 (38.0 KB), 3=h64_q6 (28.8 KB), 4=h64_q4 (19.6 KB)
  torch::Tensor all_reduce_add_rmsnorm_v2(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, double eps,
                                          int mode_code, bool out_h64) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                    x.size(0) == world_ && x.size(1) == 4608,
                "rdma_ar s8v2: draft_norm_v2 expects bf16 [world, 4608]");
    TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() &&
                    residual.sizes() == x.sizes(),
                "rdma_ar s8v2: bad residual");
    TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() &&
                    gamma.numel() == 4608,
                "rdma_ar s8v2: bad gamma");
    TORCH_CHECK(world_ == kParW && pub_mask_ == (kPubV1 | kPubV2 | kPollThin),
                "rdma_ar s8v2: draft_norm_v2 requires world=8 and pub_mask=7");
    TORCH_CHECK(max_bytes_ >= 38016, "rdma_ar s8v2: max_bytes < 38016");
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    auto out = torch::empty_like(x);
    Params p = make_base_params(x, out, x.numel() * 2);
    DraftNormParams dn{reinterpret_cast<uint4*>(residual.data_ptr()),
                       reinterpret_cast<const uint4*>(gamma.data_ptr()), (float)eps, 4608, world_};
    bool sb = false;
    if (mode_code == 0 && (r3_ & 2) != 0 && !sbuf_allocs_.empty()) {
      const auto it = sbuf_allocs_.find((uintptr_t)x.data_ptr());
      sb = it != sbuf_allocs_.end() && (size_t)p.nvec * 16 <= it->second;
    }
    const int pub = pub_mask_ | kR3Trim | r6_pdl_ | (sb ? kR3Sbuf : 0);
    launch_draft_norm_v2_kernel(p, dn, mode_code, out_h64, at::cuda::getCurrentCUDAStream(), pub);
    CHECK_CUDA(cudaGetLastError());
    return out;
  }

  // s8v2-unified (dec160): Single 48-CTA cooperative persistent kernel for B=1 7-step Markov loop
  void markov_loop_persistent(
      torch::Tensor base_logits, torch::Tensor rnn_ctx, torch::Tensor w1, torch::Tensor rnn_w_rec,
      torch::Tensor fold_w8, torch::Tensor fold_s8, torch::Tensor fold_bias,
      std::optional<torch::Tensor> id_map, torch::Tensor anchor_id,
      double softcap, int64_t valid_vocab, int64_t vocab_offset, int64_t k_steps,
      torch::Tensor greedy_dev, torch::Tensor slot_dev, torch::Tensor db_dev, torch::Tensor lam_dev,
      torch::Tensor step_eff, torch::Tensor cal, torch::Tensor vw_w, torch::Tensor vw_cpred,
      torch::Tensor draft_out, std::optional<torch::Tensor> conf_out, int64_t w_cap,
      torch::Tensor scratch_state, torch::Tensor scratch_latent, torch::Tensor scratch_rec,
      torch::Tensor scratch_ctrl,
      std::optional<torch::Tensor> req_drafts = std::nullopt,
      std::optional<torch::Tensor> sample_idx0 = std::nullopt) {
    TORCH_CHECK(world_ == kParW && inline_ag_ && !proxy_error_.load(),
                "rdma_ar s8v2: markov_loop_persistent requires world=8, inline_ag=1, healthy proxy");
    TORCH_CHECK(base_logits.is_cuda() && base_logits.scalar_type() == at::kBFloat16 &&
                    rnn_ctx.is_cuda() && rnn_ctx.scalar_type() == at::kFloat && rnn_ctx.is_contiguous() &&
                    w1.is_cuda() && w1.scalar_type() == at::kBFloat16 && w1.is_contiguous() &&
                    rnn_w_rec.is_cuda() && rnn_w_rec.scalar_type() == at::kBFloat16 && rnn_w_rec.is_contiguous() &&
                    fold_w8.is_cuda() && fold_w8.is_contiguous() &&
                    fold_s8.is_cuda() && fold_s8.scalar_type() == at::kFloat && fold_s8.is_contiguous() &&
                    fold_bias.is_cuda() && fold_bias.scalar_type() == at::kBFloat16 && fold_bias.is_contiguous(),
                "rdma_ar s8v2: invalid markov model tensor dtype/contiguity");
    Params p = make_base_params(scratch_rec, scratch_rec, 16);
    MarkovPersistentParams mp{};
    mp.base_logits = reinterpret_cast<const __nv_bfloat16*>(base_logits.data_ptr());
    mp.rnn_ctx = rnn_ctx.data_ptr<float>();
    mp.w1 = reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr());
    mp.rnn_w_rec = reinterpret_cast<const __nv_bfloat16*>(rnn_w_rec.data_ptr());
    mp.fold_w8 = reinterpret_cast<const uint8_t*>(fold_w8.data_ptr());
    mp.fold_s8 = fold_s8.data_ptr<float>();
    mp.fold_bias = reinterpret_cast<const __nv_bfloat16*>(fold_bias.data_ptr());
    const bool has_id_map = id_map.has_value() && id_map->defined();
    mp.id_map = has_id_map ? id_map->data_ptr<int32_t>() : nullptr;
    mp.anchor_id_ptr = anchor_id.data_ptr();
    mp.anchor_is_i64 = (anchor_id.scalar_type() == at::kLong) ? 1 : 0;
    mp.base_stride = base_logits.stride(0);
    mp.softcap = (float)softcap;
    mp.valid_vocab = (int32_t)valid_vocab;
    mp.vocab_offset = (int32_t)vocab_offset;
    mp.k_steps = (int32_t)k_steps;
    mp.greedy_dev = greedy_dev.data_ptr<int32_t>();
    mp.slot_dev = slot_dev.data_ptr<int32_t>();
    mp.db_dev = db_dev.data_ptr<float>();
    mp.lam_dev = lam_dev.data_ptr<float>();
    mp.step_eff_ptr = step_eff.data_ptr<float>();
    mp.cal_ptr = cal.data_ptr<float>();
    mp.vw_w_ptr = vw_w.data_ptr<int32_t>();
    mp.vw_cpred_ptr = vw_cpred.data_ptr<float>();
    mp.draft_out = draft_out.data_ptr<int64_t>();
    mp.conf_out = (conf_out.has_value() && conf_out->defined()) ? conf_out->data_ptr<float>() : nullptr;
    const bool has_req_d = req_drafts.has_value() && req_drafts->defined();
    mp.req_drafts = has_req_d ? req_drafts->data_ptr<int64_t>() : nullptr;
    mp.req_drafts_stride = has_req_d ? req_drafts->stride(0) : 0;
    const bool has_sidx0 = sample_idx0.has_value() && sample_idx0->defined();
    mp.sample_idx0_ptr = has_sidx0 ? sample_idx0->data_ptr() : nullptr;
    mp.sample_idx_is_i64 = (has_sidx0 && sample_idx0->scalar_type() == at::kLong) ? 1 : 0;
    mp.w_cap = (int32_t)w_cap;
    mp.scratch_state = reinterpret_cast<__nv_bfloat16*>(scratch_state.data_ptr());
    mp.scratch_latent = reinterpret_cast<__nv_bfloat16*>(scratch_latent.data_ptr());
    mp.scratch_rec = reinterpret_cast<float4*>(scratch_rec.data_ptr());
    mp.scratch_ctrl = scratch_ctrl.data_ptr<int64_t>();

    int coop_ok = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&coop_ok, cudaDevAttrCooperativeLaunch, device_));
    TORCH_CHECK(coop_ok != 0, "rdma_ar s8v2: device does not support cooperative launch");
    int n_sm = 48;
    CHECK_CUDA(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device_));
    const int n_ctas = std::min(48, std::max(32, n_sm));
    constexpr int kMkPub = kPubV1 | kPubV2 | kPollThin | kR3Trim;
    dim3 grid((unsigned)n_ctas, 1u, 1u);
    dim3 block(256u, 1u, 1u);
    void* kargs[] = {&p, &mp};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (has_id_map) {
      CHECK_CUDA(cudaLaunchCooperativeKernel(
          reinterpret_cast<const void*>(&markov_loop_persistent_kernel<kParW, kMkPub, true>),
          grid, block, kargs, 0, stream));
    } else {
      CHECK_CUDA(cudaLaunchCooperativeKernel(
          reinterpret_cast<const void*>(&markov_loop_persistent_kernel<kParW, kMkPub, false>),
          grid, block, kargs, 0, stream));
    }
  }

  // s8v2-unified (dec160): Standalone 1-warp fused inline AG + select + Dink early-stop (zero coop!)
  void inline_ag_select_dink(
      torch::Tensor rec1, torch::Tensor out_tok, torch::Tensor conf_out,
      torch::Tensor skip_dev, torch::Tensor surv_dev, torch::Tensor tok_dev, torch::Tensor bestu_dev,
      torch::Tensor greedy_dev, torch::Tensor db_dev, torch::Tensor lam_dev,
      torch::Tensor cal, torch::Tensor step_eff, int64_t step_idx, int64_t k_steps) {
    TORCH_CHECK(world_ == kParW && inline_ag_ && !proxy_error_.load(),
                "rdma_ar s8v2: inline_ag_select_dink requires world=8, inline_ag=1, healthy proxy");
    Params p = make_base_params(rec1, rec1, 16);
    InlineSelectDinkParams sp{};
    sp.out_tok = out_tok.data_ptr<int64_t>();
    sp.conf_out = conf_out.data_ptr<float>();
    sp.skip_ptr = skip_dev.data_ptr<int32_t>();
    sp.surv_ptr = surv_dev.data_ptr<float>();
    sp.tok_ptr = tok_dev.data_ptr<float>();
    sp.bestu_ptr = bestu_dev.data_ptr<float>();
    sp.greedy_ptr = greedy_dev.data_ptr<int32_t>();
    sp.db_dev_ptr = db_dev.data_ptr<float>();
    sp.lam_dev_ptr = lam_dev.data_ptr<float>();
    sp.cal_ptr = cal.data_ptr<float>();
    sp.step_ptr = step_eff.data_ptr<float>();
    sp.step_idx = (int32_t)step_idx;
    sp.k_steps = (int32_t)k_steps;
    constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
    const int m = (pub_mask_ | kR3Trim | r6_pdl_) & (B | kR6Pdl | kR6Trig);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (m) {
      case B: launch_inline_ag_select_dink_one<B>(p, sp, stream); break;
      case B | kR6Pdl: launch_inline_ag_select_dink_one<B | kR6Pdl>(p, sp, stream); break;
      case B | kR6Pdl | kR6Trig: launch_inline_ag_select_dink_one<B | kR6Pdl | kR6Trig>(p, sp, stream); break;
      default: launch_inline_ag_select_dink_one<B>(p, sp, stream); break;
    }
    CHECK_CUDA(cudaGetLastError());
  }

  // s8: 0-ULP Bit-Exact Fused RDMA AllReduce + SConv + AttnRes + RMSNorm (world=8, D=4608, T<=8)
  torch::Tensor all_reduce_sconv_attnres(
      torch::Tensor x, torch::Tensor conv_w, torch::Tensor cache,
      torch::Tensor positions, torch::Tensor block_table, torch::Tensor seq_idx,
      torch::Tensor slot_map, torch::Tensor q_start,
      int64_t stream_off,
      torch::Tensor conv_out,
      std::optional<torch::Tensor> aux_out,
      std::optional<torch::Tensor> conv_res,
      std::optional<torch::Tensor> conv_addend,
      torch::Tensor stack,
      std::optional<torch::Tensor> mix_last,
      torch::Tensor queries,
      torch::Tensor norm_w,
      double eps, double norm_eps, bool two_shot) {
    TORCH_CHECK(rows_dev_ != nullptr, "rdma_ar s8: set_rows first");
    TORCH_CHECK(world_ == kParW && pub_mask_ == (kPubV1 | kPubV2 | kPollThin) && (r3_ & 1) != 0,
                "rdma_ar s8: requires world=8, pub_mask=7, DSV41_RDMA_TRIM=1");
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    const int T = (int)x.size(0);
    const int nq = (int)queries.size(0);
    const int n_stack = (int)stack.size(0);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == 4608 && T >= 1 && T <= 8 && (int64_t)T == rows_full_,
                "rdma_ar s8: x must be contiguous bf16 [rows_full<=8, 4608]");
    TORCH_CHECK((nq == 1 || nq == 3) && n_stack >= 1 && n_stack <= 8,
                "rdma_ar s8: bad nq/n_stack");
    const bool has_ca = conv_addend.has_value();
    const bool has_cr = conv_res.has_value();
    const bool has_ml = mix_last.has_value();
    TORCH_CHECK(!(has_cr && !has_ca) && !(has_ca && has_ml),
                "rdma_ar s8: invalid conv_addend/conv_res/mix_last combination");
    const int kcase = has_ca ? (has_cr ? 3 : 2) : (has_ml ? 1 : 0);
    auto out = torch::empty({nq, T, 4608}, x.options());
    const int64_t nbytes = x.numel() * 2;
    Params p = make_base_params(x, conv_out, nbytes);
    p.rows_ptr = rows_dev_;
    p.row_vec = 576;
    p.nvec_full = (long long)T * 576LL;
    p.oneshot_max_rows = two_shot ? rows_oneshot_max_ : T;
    p.s1_bytes = rows_s1_bytes_;
    bool sb = false;
    if ((r3_ & 2) != 0 && !sbuf_allocs_.empty()) {
      const auto it = sbuf_allocs_.find((uintptr_t)x.data_ptr());
      sb = it != sbuf_allocs_.end() && (size_t)nbytes <= it->second;
    }
    const int pub = pub_mask_ | kR3Trim | r6_pdl_ | (sb ? kR3Sbuf : 0);

    const int n_cache = (int)cache.size(2);
    int n_shift = -1;
    if (n_cache > 0 && (n_cache & (n_cache - 1)) == 0) {
      n_shift = 0;
      for (int v = n_cache; v > 1; v >>= 1) ++n_shift;
    }
    SconvAttnResParams sp{};
    sp.conv_w = reinterpret_cast<const uint16_t*>(conv_w.data_ptr());
    sp.cache = reinterpret_cast<uint16_t*>(cache.data_ptr());
    sp.conv_out = reinterpret_cast<uint16_t*>(conv_out.data_ptr());
    sp.aux_out = aux_out.has_value() ? reinterpret_cast<uint16_t*>(aux_out->data_ptr()) : nullptr;
    sp.conv_res = has_cr ? reinterpret_cast<const uint16_t*>(conv_res->data_ptr()) : nullptr;
    sp.conv_addend = has_ca ? reinterpret_cast<const uint16_t*>(conv_addend->data_ptr()) : nullptr;
    sp.pos = reinterpret_cast<const int64_t*>(positions.data_ptr());
    sp.seq_idx = reinterpret_cast<const int32_t*>(seq_idx.data_ptr());
    sp.slot_map = reinterpret_cast<const int64_t*>(slot_map.data_ptr());
    sp.block_table = reinterpret_cast<const int32_t*>(block_table.data_ptr());
    sp.q_start = reinterpret_cast<const int32_t*>(q_start.data_ptr());
    sp.stack = reinterpret_cast<const __nv_bfloat16*>(stack.data_ptr());
    sp.mix_last = has_ml ? reinterpret_cast<const __nv_bfloat16*>(mix_last->data_ptr()) : nullptr;
    sp.queries = reinterpret_cast<const float*>(queries.data_ptr());
    sp.norm_w = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr());
    sp.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    sp.stride_c_blk = cache.stride(0);
    sp.stride_c_n = cache.stride(2);
    sp.stride_bt = block_table.stride(0);
    sp.stream_off = stream_off;
    sp.stride_s = stack.stride(0);
    sp.max_blocks = (int)block_table.size(1);
    sp.n_cache = n_cache;
    sp.n_shift = n_shift;
    sp.nq = nq;
    sp.T = T;
    sp.scale = 0.44721359549995793928f;  // 5**-0.5 (exact float match to Python float(5**-0.5))
    sp.eps = (float)eps;
    sp.norm_eps = (float)norm_eps;
    launch_sconv_attnres_kernel(p, sp, n_stack, kcase, at::cuda::getCurrentCUDAStream(), pub);
    CHECK_CUDA(cudaGetLastError());
    return out;
  }

  std::vector<int64_t> s8_info() const {
    return {level_db_ ? 1 : 0, wire_bf12_ ? 1 : 0, bf12_os4_ ? 1 : 0, (int64_t)kRowsThreads};
  }

  std::vector<int64_t> c32_info() const {
    return {c32_kernel_ ? 1 : 0, (int64_t)c32_bf12_mode_, c32_pipe_ ? 1 : 0, dual_qp_ ? 1 : 0,
            pci_ro_ ? 1 : 0, recv_decouple_ ? 1 : 0, (int64_t)max_ctas_, (int64_t)recv_stride_, (int64_t)slot_chunk_};
  }

  // c32: Batched Drafter 2-shot H64-Q4/Q6/Q8 AR + fused residual add + RMSNorm (bf16 [M, 4608], M%8==0, 16<=M<=256)
  torch::Tensor all_reduce_add_rmsnorm_batched(
      torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, double eps, int mode_code) {
    const int64_t M = x.size(0);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                    M >= 16 && M <= 256 && (M % world_) == 0 && x.size(1) == 4608,
                "rdma_ar c32: draft_batched expects bf16 [16..256 (mult 8), 4608]");
    TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() &&
                    residual.sizes() == x.sizes(),
                "rdma_ar c32: bad residual");
    TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() &&
                    gamma.numel() == 4608,
                "rdma_ar c32: bad gamma");
    TORCH_CHECK(world_ == kParW && pub_mask_ == (kPubV1 | kPubV2 | kPollThin) && (r3_ & 1) != 0,
                "rdma_ar c32: draft_batched requires world=8, pub_mask=7, DSV41_RDMA_TRIM=1");
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    const int bits = (mode_code == 4) ? 4 : ((mode_code == 3) ? 6 : 8);
    const int64_t q_row_bytes = (bits == 4) ? 2448 : ((bits == 6) ? 3600 : 4752);
    TORCH_CHECK((size_t)(M * q_row_bytes) <= slot_bytes_ && (size_t)((M / world_) * q_row_bytes) <= slot_chunk_,
                "rdma_ar c32: draft_batched exceeds arena slot size");
    auto out = torch::empty_like(x);
    Params p = make_base_params(x, out, M * 4608 * 2);
    DraftNormParams dn{reinterpret_cast<uint4*>(residual.data_ptr()),
                       reinterpret_cast<const uint4*>(gamma.data_ptr()), (float)eps, 4608, (int)M};
    const int pub = pub_mask_ | kR3Trim | r6_pdl_;
    launch_draft_c32_h64_kernel(p, dn, bits, at::cuda::getCurrentCUDAStream(), pub);
    CHECK_CUDA(cudaGetLastError());
    return out;
  }

  uint64_t ar_done_ptr() const { return (uint64_t)(uintptr_t)&st_->ar_done_cnt; }

  std::vector<int64_t> s6_info() const {
    int64_t cq_ex_ok = 0;
    for (const auto& h : hcas_)
      if (h.cq_ex) ++cq_ex_ok;
    return {use_qp_ex_ ? 1 : 0, two_pass_post_ ? 1 : 0, piggyback_ ? 1 : 0, l2_relay_ ? 1 : 0,
            (int64_t)poll_sleep_ns_, inline_ag_ ? 1 : 0, multi_proxy_ ? 1 : 0, proxy_core_, proxy_core2_,
            rows_p2_, cq_ex_ok};
  }

  torch::Tensor all_reduce(torch::Tensor x, bool two_shot) {
    const bool f32 = x.scalar_type() == at::kFloat;
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || f32) && x.is_contiguous(),
                "rdma_ar: need contiguous bf16/fp32 CUDA");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "rdma_ar: all-reduce too large");
    auto out = torch::empty_like(x);
    if (f32) {
      if (two_shot)
        launch<kTwoShot, true>(x, out);
      else
        launch<kOneShot, true>(x, out);
    } else {
      const int64_t nbytes = x.numel() * 2;
      const int64_t nvec = nbytes / 16;
      if (two_shot && c32_kernel_ && world_ == kParW && (r3_ & 1) != 0 &&
          pub_mask_ == (kPubV1 | kPubV2 | kPollThin) &&
          nvec >= c32_min_nvec_ && (nvec % (kParW * 32)) == 0) {
        launch_c32_allreduce(x, out, nbytes);
      } else if (two_shot) {
        launch<kTwoShot, false>(x, out);
      } else {
        launch<kOneShot, false>(x, out);
      }
    }
    return out;
  }

  // w3 rowar: register the device int32 [1] holding the live row count (written once per forward by the model) and the
  // geometry of the row-aware tensors: rows_full rows of row_bytes bytes each.
  void set_rows(torch::Tensor n_live, int64_t row_bytes, int64_t rows_full) {
    TORCH_CHECK(n_live.is_cuda() && n_live.scalar_type() == at::kInt && n_live.numel() >= 1 && n_live.is_contiguous(),
                "rdma_ar rowar: n_live must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(row_bytes > 0 && row_bytes % 16 == 0 && rows_full >= 1, "rdma_ar rowar: bad row geometry");
    rows_dev_ = reinterpret_cast<const int*>(n_live.data_ptr());
    rows_row_vec_ = row_bytes / 16;
    rows_full_ = rows_full;
  }
  // w3 rowar: all_reduce of x ([rows_full, row] contiguous bf16) that reduces only the first *n_live rows (device-read at
  // kernel time); output rows >= n_live are zero. Same wire protocol/ordering contract as all_reduce.
  torch::Tensor all_reduce_rows(torch::Tensor x, bool two_shot) {
    TORCH_CHECK(rows_dev_ != nullptr, "rdma_ar rowar: set_rows first");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "rdma_ar rowar: need contiguous bf16 CUDA");
    TORCH_CHECK(x.numel() * x.element_size() == rows_full_ * rows_row_vec_ * 16, "rdma_ar rowar: x is not rows_full rows");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "rdma_ar: all-reduce too large");
    auto out = torch::empty_like(x);
    rows_active_ = true;
    try {
      if (rows_auto_ && parload() && (r3_ & 1) != 0) {
        launch_rows_auto(x, out, two_shot);
      } else {
        if (two_shot) launch<kTwoShot, false>(x, out); else launch<kOneShot, false>(x, out);
      }
    } catch (...) { rows_active_ = false; throw; }
    rows_active_ = false;
    return out;
  }
  // c32 tier-2 live-token compaction: dynamic-rows 2-shot all-reduce for M > 8 bf16 [M_pad, 4608].
  // Device-reads *n_live (rounded up to multiple of 8 rows) inside ar_kernel_c32 to shrink P1+P2
  // RDMA wire bytes with zero host sync. Falls back cleanly to all_reduce() when ineligible.
  torch::Tensor all_reduce_c32_rows(torch::Tensor x, torch::Tensor n_live, bool two_shot) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == 4608,
                "rdma_ar c32_rows: need contiguous [M, 4608] bf16 CUDA");
    TORCH_CHECK(n_live.is_cuda() && n_live.scalar_type() == at::kInt && n_live.numel() >= 1 && n_live.is_contiguous(),
                "rdma_ar c32_rows: n_live must be contiguous CUDA int32");
    const int64_t nbytes = x.numel() * 2;
    const int64_t nvec = nbytes / 16;
    TORCH_CHECK(nbytes <= max_reduce_bytes_, "rdma_ar: all-reduce too large");
    if (two_shot && c32_kernel_ && world_ == kParW && (r3_ & 1) != 0 &&
        pub_mask_ == (kPubV1 | kPubV2 | kPollThin) &&
        nvec >= c32_min_nvec_ && (nvec % (kParW * 32)) == 0 && (x.size(0) % 8) == 0) {
      auto out = torch::empty_like(x);
      launch_c32_allreduce(x, out, nbytes, reinterpret_cast<const int*>(n_live.data_ptr()), (4608 * 2) / 16);
      return out;
    }
    return all_reduce(x, two_shot);
  }
  void all_gather_into_cond(torch::Tensor out, torch::Tensor x, torch::Tensor skip_dev) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "rdma_ar: bad all_gather_cond args");
    TORCH_CHECK(skip_dev.is_cuda() && skip_dev.scalar_type() == at::kInt && skip_dev.numel() >= 1 && skip_dev.is_contiguous(),
                "rdma_ar: skip_dev must be contiguous CUDA int32");
    skip_dev_ = reinterpret_cast<const int*>(skip_dev.data_ptr());
    try {
      launch<kGather, false>(x, out);
    } catch (...) { skip_dev_ = nullptr; throw; }
    skip_dev_ = nullptr;
  }

  // r7-comm2 HT13: out = the all-reduce of x (exactly all_reduce's bytes) and y = _proj_rows(out, d), in one kernel.
  // Needs DSV41_RDMA_PROJ=1 (and TRIM) at construction; bf16 [rows <= 6, 5120]; d fp32 [5120]; y bf16 like x.
  torch::Tensor all_reduce_proj(torch::Tensor x, bool two_shot, torch::Tensor d, torch::Tensor y) {
    TORCH_CHECK(r7_proj_ && (r3_ & 1) != 0, "rdma_ar r7: all_reduce_proj needs DSV41_RDMA_PROJ=1 and DSV41_RDMA_TRIM=1");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "rdma_ar: all-reduce too large");
    auto out = torch::empty_like(x);
    const int64_t rows = r7_check(x, out, d, y);
    const R7Proj q{reinterpret_cast<const float*>(d.data_ptr()), reinterpret_cast<uint4*>(y.data_ptr()), rows};
    if (two_shot)
      launch<kTwoShot, false>(x, out, 0, &q);
    else
      launch<kOneShot, false>(x, out, 0, &q);
    return out;
  }
  bool r7() const { return r7_proj_; }

  // out = concat over ranks (dim 0) of x; any dtype, out must be contiguous with world * x.numel() elements
  void all_gather_into(torch::Tensor out, torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "rdma_ar: bad all_gather args");
    launch<kGather, false>(x, out);
  }

  // g2 colltrim: column all-gather. x = [outer, seg_bytes] bytes (any dtype, contiguous), out = [outer,
  // world * seg_bytes] bytes (contiguous): out row o, bytes [r*seg_bytes, (r+1)*seg_bytes) = row o of rank r's x.
  // seg_bytes == x bytes is the plain dim-0 gather. Same wire protocol and sequence as all_gather_into.
  void all_gather_cols(torch::Tensor out, torch::Tensor x, int64_t seg_bytes) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "rdma_ar: bad all_gather_cols args");
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(seg_bytes > 0 && seg_bytes % 16 == 0 && nbytes % seg_bytes == 0,
                "rdma_ar: all_gather_cols seg_bytes ", seg_bytes, " must be a positive multiple of 16 dividing ", nbytes);
    launch<kGatherCols, false>(x, out, seg_bytes / 16);
  }

  int64_t max_bytes() const { return max_bytes_; }
  // average GPU->proxy->GPU round trip (ns) through the control block
  double pingpong(int n) {
    unsigned long long* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(unsigned long long)));
    static unsigned long long base = 1ull << 40;
    base += 1ull << 20;
    pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<Ctrl*>(arena_dev_), n, base, d_out);
    unsigned long long h = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream()));
    CHECK_CUDA(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    CHECK_CUDA(cudaFree(d_out));
    return (double)h / n;
  }

  // [arena bytes, bytes backed by huge pages, 1 if THP+cudaHostRegister path was used]
  std::vector<int64_t> arena_info() const { return {(int64_t)arena_bytes_, thp_bytes_, arena_registered_ ? 1 : 0}; }
  int64_t max_reduce_bytes() const { return max_reduce_bytes_; }

 private:
  static size_t align(size_t x, size_t a) { return (x + a - 1) / a * a; }

  // bytes of [p, p+n) backed by anonymous huge pages, from /proc/self/smaps
  static int64_t huge_backed_bytes(void* p, size_t n) {
    std::ifstream f("/proc/self/smaps");
    std::string line;
    const uintptr_t lo = (uintptr_t)p, hi = lo + n;
    bool in = false;
    int64_t kb = 0;
    while (std::getline(f, line)) {
      unsigned long a, b;
      if (sscanf(line.c_str(), "%lx-%lx", &a, &b) == 2) {  // mapping header line
        in = a < hi && b > lo;
        continue;
      }
      long v;
      if (in && sscanf(line.c_str(), "AnonHugePages: %ld kB", &v) == 1) kb += v;
    }
    return kb * 1024;
  }

  Params make_base_params(const torch::Tensor& x, torch::Tensor& out, int64_t nbytes) const {
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.st = st_;
    p.max_bytes = (long long)slot_bytes_;
    p.max_chunk = (long long)slot_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = timeout_ns_;
    p.timing = timing_;
    p.rows_p2 = rows_p2_;
    p.piggyback = piggyback_ ? 1 : 0;
    p.l2_relay = l2_relay_ ? 1 : 0;
    p.poll_sleep_ns = poll_sleep_ns_;
    p.inline_ag = inline_ag_ ? 1 : 0;
    p.inline_recv = reinterpret_cast<const InlineSlot*>(arena_dev_ + inline_recv_off_);
    p.level_db = level_db_ ? 1 : 0;
    p.multi_proxy = (multi_proxy_ && hcas_.size() >= 2) ? 1 : 0;
    p.wire_bf12 = wire_bf12_ ? 1 : 0;
    p.bf12_os4 = bf12_os4_ ? 1 : 0;
    p.ctrl_slots = reinterpret_cast<CtrlSlot*>(arena_dev_ + kCtrlSlotsOff);
    p.ctrl_aux = reinterpret_cast<CtrlAux*>(arena_dev_ + kCtrlAuxOff);
    p.reduced_buf = reduced_buf_;
    p.bf12_os_scratch = bf12_os_scratch_;
    p.flags_qp2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_qp2_off_);
    p.flags2_qp2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_qp2_off_);
    p.ctrl_c32 = reinterpret_cast<CtrlC32Slot*>(arena_dev_ + kCtrlC32Off);
    p.recv_stride = (long long)recv_stride_;
    p.dual_qp = dual_qp_ ? 1 : 0;
    p.c32_med_gather = c32_kernel_ ? 1 : 0;
    p.stripe_min_bytes = stripe_min_bytes_;
    return p;
  }

  void launch_c32_allreduce(const torch::Tensor& x, torch::Tensor& out, int64_t nbytes,
                            const int* rows_ptr = nullptr, int64_t row_vec = 0) {
    TORCH_CHECK(nbytes % 16 == 0 && nbytes <= max_reduce_bytes_, "rdma_ar c32: bad size ", nbytes);
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    Params p = make_base_params(x, out, nbytes);
    p.rows_ptr = rows_ptr;
    p.row_vec = row_vec;
    p.nvec_full = p.nvec;
    bool sb = false;
    if ((r3_ & 2) != 0 && !sbuf_allocs_.empty()) {
      const auto it = sbuf_allocs_.find((uintptr_t)x.data_ptr());
      sb = it != sbuf_allocs_.end() && (size_t)nbytes <= it->second;
    }
    if (sb && r3_inplace_conflict(kTwoShot, (unsigned long long)(uintptr_t)p.in,
                                  (unsigned long long)(uintptr_t)p.out, p.nvec, rank_, world_))
      sb = false;
    const bool bf12_p2 = (c32_bf12_mode_ >= 1) && (nbytes >= c32_bf12_p2_min_bytes_);
    const bool bf12_p1 = (c32_bf12_mode_ >= 2) && (nbytes >= c32_bf12_p1_min_bytes_);
    const bool pipe2 = c32_pipe_ && level_db_ && (nbytes >= c32_pipe_min_bytes_) &&
                       ((p.nvec % (2LL * (long long)kParW * 32LL)) == 0);
    const int pub = pub_mask_ | kR3Trim | r6_pdl_ | ((sb && !bf12_p1) ? kR3Sbuf : 0);
    launch_c32_kernel(p, bf12_p1, bf12_p2, pipe2, at::cuda::getCurrentCUDAStream(), pub);
    CHECK_CUDA(cudaGetLastError());
  }

  void launch_rows_auto(const torch::Tensor& x, torch::Tensor& out, bool two_shot) {
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes <= max_bytes_, "rdma_ar: bad size ", nbytes);
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    if (nbytes == 0) return;
    TORCH_CHECK(pub_mask_ == (kPubV1 | kPubV2 | kPollThin), "rdma_ar s5: needs pub_mask 7");
    Params p = make_base_params(x, out, nbytes);
    p.rows_ptr = rows_dev_;
    p.row_vec = rows_row_vec_;
    p.nvec_full = p.nvec;
    p.oneshot_max_rows = two_shot ? rows_oneshot_max_ : (int)rows_full_;
    p.s1_bytes = rows_s1_bytes_;
    bool sb = false;
    if ((r3_ & 2) != 0 && !sbuf_allocs_.empty()) {
      const auto it = sbuf_allocs_.find((uintptr_t)x.data_ptr());
      sb = it != sbuf_allocs_.end() && (size_t)p.nvec_full * 16 <= it->second;
    }
    if (sb && r3_inplace_conflict(kTwoShot, (unsigned long long)(uintptr_t)p.in,
                                  (unsigned long long)(uintptr_t)p.out, p.nvec_full, rank_, world_))
      sb = false;
    const int pub = pub_mask_ | kR3Trim | r6_pdl_ | (sb ? kR3Sbuf : 0);
    launch_rows_kernel(p, vec_per_cta_, at::cuda::getCurrentCUDAStream(), pub);
  }

  template <int kMode, bool kF32>
  void launch(const torch::Tensor& x, torch::Tensor& out, int64_t seg_vec = 0, const R7Proj* q7 = nullptr) {
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes <= max_bytes_, "rdma_ar: bad size ", nbytes);
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    if (nbytes == 0) return;
    // g130: kernel variants exist on the parallel-load path only; never let a flag be silently ignored
    TORCH_CHECK(pub_mask_ == 0 || parload(), "rdma_ar: RDMA_AR_PUB_V1/V2 and RDMA_AR_HYG_POLL need the parallel-load kernels");
    Params p = make_base_params(x, out, nbytes);
    p.seg = seg_vec;  // g2 colltrim: read by kGatherCols only
    p.skip_ptr = skip_dev_;
    if (rows_active_) {  // w3 rowar: device-read live row count (set by all_reduce_rows only)
      p.rows_ptr = rows_dev_;
      p.row_vec = rows_row_vec_;
      p.nvec_full = p.nvec;
    }
    // a few CTAs are enough: the cost is latency, not bandwidth (RDMA_AR_VEC_PER_CTA tunes it)
    // k1/3-rdma-loads: parload_ picks ar_kernel_pl (world 8) over the legacy ar_kernel
    int pub = r3_ ? r3_pub(kMode, x, p) : pub_mask_;
    if (rows_active_) {
      // rows launches ride the staged r3 kernel; the static S1/P2/Sbuf choices (made from the FULL size) are dropped
      TORCH_CHECK((pub & kR3Trim) != 0, "rdma_ar rowar: needs DSV41_RDMA_TRIM=1");
      pub &= ~(kR3S1 | kR3P2 | kR3Sbuf);
    }
    if (q7 != nullptr) {  // r7-comm2 HT13 (never with a send buffer: all_reduce_proj inputs are never sbuf views)
      TORCH_CHECK((pub & kR3Trim) != 0 && (pub & kR3Sbuf) == 0, "rdma_ar r7: projection needs the staged r3 kernel");
      pub |= kR7Proj;
    }
    if constexpr (kMode != kTwoShot) {
      TORCH_CHECK((size_t)nbytes <= recv_stride_ - 64 || !recv_decouple_,
                  "rdma_ar c32: non-twoshot payload ", nbytes, " exceeds decoupled recv_stride ", recv_stride_);
    }
    const long long eff_max_ctas = (p.nvec >= c32_min_nvec_) ? max_ctas_ : 16LL;
    launch_kernel<kMode, kF32>(p, vec_per_cta_, parload_ != 0, at::cuda::getCurrentCUDAStream(), pub,
                               q7 != nullptr ? *q7 : R7Proj{nullptr, nullptr, 0}, eff_max_ctas);
  }

  // r3-comm: kernel mask of one launch with an r3 flag on (never reached with r3_ == 0)
  int r3_pub(int mode, const torch::Tensor& x, const Params& p) {
    TORCH_CHECK(parload() && pub_mask_ == (kPubV1 | kPubV2 | kPollThin),
                "rdma_ar r3: DSV41_RDMA_TRIM/SENDBUF need the parallel-load kernels and RDMA_AR_PUB_V1=1 "
                "RDMA_AR_PUB_V2=1 RDMA_AR_HYG_POLL=1 (pub mask 7), got mask ", pub_mask_);
    bool sb = false;
    if ((r3_ & 2) != 0 && !sbuf_allocs_.empty()) {
      const auto it = sbuf_allocs_.find((uintptr_t)x.data_ptr());
      sb = it != sbuf_allocs_.end() && (size_t)p.nvec * 16 <= it->second;
    }
    int b = r3_bits(mode, p.nvec, p.chunk, r3_s1_bytes_, sb, kThreads);
    if ((b & (kR3S1 | kR3Sbuf)) != 0 &&
        r3_inplace_conflict(mode, (unsigned long long)(uintptr_t)p.in, (unsigned long long)(uintptr_t)p.out, p.nvec,
                            rank_, world_))
      b &= ~(kR3S1 | kR3Sbuf);  // review N5: in-place call -> the staged r3 path
    return pub_mask_ | b | r6_pdl_;  // s5: allow PDL bits with send buffers too
  }

  // c32: lightweight QP handle supporting primary (Root 0/2) or secondary (opposite root) QP with Dual-MR RO!
  struct QpRef {
    int hca;
    ibv_qp* qp;
    ibv_qp_ex* qpx;
    uint32_t rkey_strict;
    uint32_t rkey_ro;
    uint64_t remote_base;
    uint64_t* posts;
  };

  inline bool post_payload_q(const QpRef& q, uint64_t src, uint32_t len, uint64_t dst) {
    if (len == 0) return true;
    Hca& h = hcas_[q.hca];
    const bool do_inline = len <= h.max_inline;
    if (q.qpx) {
      ibv_wr_start(q.qpx);
      q.qpx->wr_id = 0;
      q.qpx->wr_flags = do_inline ? IBV_SEND_INLINE : 0;
      ibv_wr_rdma_write(q.qpx, q.rkey_ro, dst);
      if (do_inline)
        ibv_wr_set_inline_data(q.qpx, reinterpret_cast<void*>(src), len);
      else
        ibv_wr_set_sge(q.qpx, h.mr_ro->lkey, src, len);
      if (ibv_wr_complete(q.qpx)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(payload_q) failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
        return false;
      }
      return true;
    }
    ibv_sge sge{src, len, h.mr_ro->lkey};
    ibv_send_wr wr{};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = do_inline ? IBV_SEND_INLINE : 0;
    wr.wr.rdma.remote_addr = dst;
    wr.wr.rdma.rkey = q.rkey_ro;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(q.qp, &wr, &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send(payload_q) failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
      return false;
    }
    return true;
  }

  inline void post_flag_q(const QpRef& q, uint64_t flag_dst, uint64_t flag) {
    const unsigned int sig = ((++(*q.posts) % kSignalEvery) == 0) ? IBV_SEND_SIGNALED : 0;
    if (q.qpx) {
      ibv_wr_start(q.qpx);
      q.qpx->wr_id = 0;
      q.qpx->wr_flags = IBV_SEND_INLINE | sig;
      ibv_wr_rdma_write(q.qpx, q.rkey_strict, flag_dst);
      ibv_wr_set_inline_data(q.qpx, &flag, 8);
      if (ibv_wr_complete(q.qpx)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(flag_q) failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
      }
      return;
    }
    ibv_sge fsge{(uint64_t)(uintptr_t)&flag, 8, 0};
    ibv_send_wr wr{};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &fsge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_INLINE | sig;
    wr.wr.rdma.remote_addr = flag_dst;
    wr.wr.rdma.rkey = q.rkey_strict;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(q.qp, &wr, &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send(flag_q) failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
    }
  }

  // s6 fast-path single-WR post helpers (ibv_qp_ex lock-free BlueFlame when available, else legacy ibv_post_send)
  inline bool post_payload_only(Peer& pr, uint64_t src, uint32_t len, uint64_t dst) {
    if (len == 0) return true;
    Hca& h = hcas_[pr.hca];
    const bool do_inline = len <= h.max_inline;
    if (pr.qpx) {
      ibv_wr_start(pr.qpx);
      pr.qpx->wr_id = 0;
      pr.qpx->wr_flags = do_inline ? IBV_SEND_INLINE : 0;
      ibv_wr_rdma_write(pr.qpx, pr.rkey_ro, dst);
      if (do_inline)
        ibv_wr_set_inline_data(pr.qpx, reinterpret_cast<void*>(src), len);
      else
        ibv_wr_set_sge(pr.qpx, h.mr_ro->lkey, src, len);
      if (ibv_wr_complete(pr.qpx)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(payload) failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
        return false;
      }
      return true;
    }
    ibv_sge sge{src, len, h.mr_ro->lkey};
    ibv_send_wr wr{};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = do_inline ? IBV_SEND_INLINE : 0;
    wr.wr.rdma.remote_addr = dst;
    wr.wr.rdma.rkey = pr.rkey_ro;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(pr.qp, &wr, &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send(payload) failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
      return false;
    }
    return true;
  }

  inline void post_flag_only(Peer& pr, uint64_t flag_dst, uint64_t flag) {
    const unsigned int sig = ((++pr.posts % kSignalEvery) == 0) ? IBV_SEND_SIGNALED : 0;
    if (pr.qpx) {
      ibv_wr_start(pr.qpx);
      pr.qpx->wr_id = 0;
      pr.qpx->wr_flags = IBV_SEND_INLINE | sig;
      ibv_wr_rdma_write(pr.qpx, pr.rkey, flag_dst);
      ibv_wr_set_inline_data(pr.qpx, &flag, 8);
      if (ibv_wr_complete(pr.qpx)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(flag) failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
      }
      return;
    }
    ibv_sge fsge{(uint64_t)(uintptr_t)&flag, 8, 0};
    ibv_send_wr wr{};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &fsge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_INLINE | sig;
    wr.wr.rdma.remote_addr = flag_dst;
    wr.wr.rdma.rkey = pr.rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(pr.qp, &wr, &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send(flag) failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
    }
  }

  // s7: single-doorbell batched inline payload + 8B flag (2 distinct RDMA WRITE WRs in 1 ibv_wr_complete!)
  // Safe under ConnectX-7 PCIe RO (2 ordered RC RDMA WRITEs, NOT a 2-SGE piggyback WR), saves 1 MMIO doorbell/peer!
  inline void post_inline_pair(Peer& pr, uint64_t src, uint32_t len, uint64_t dst, uint64_t flag_dst, uint64_t flag) {
    const unsigned int sig = ((++pr.posts % kSignalEvery) == 0) ? IBV_SEND_SIGNALED : 0;
    if (pr.qpx) {
      ibv_wr_start(pr.qpx);
      if (len > 0) {
        pr.qpx->wr_id = 0;
        pr.qpx->wr_flags = IBV_SEND_INLINE;
        ibv_wr_rdma_write(pr.qpx, pr.rkey, dst);
        ibv_wr_set_inline_data(pr.qpx, reinterpret_cast<void*>(src), len);
      }
      pr.qpx->wr_id = 0;
      pr.qpx->wr_flags = IBV_SEND_INLINE | sig;
      ibv_wr_rdma_write(pr.qpx, pr.rkey, flag_dst);
      ibv_wr_set_inline_data(pr.qpx, &flag, 8);
      if (ibv_wr_complete(pr.qpx)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(inl_pair) failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
      }
      return;
    }
    if (len > 0) {
      if (!post_payload_only(pr, src, len, dst)) return;
    }
    post_flag_only(pr, flag_dst, flag);
  }

  // s6: single 24-byte inline BlueFlame RDMA WRITE ([16B payload | 8B seq]) for 16B kGather
  inline void post_inline24(Peer& pr, const void* buf24, uint64_t dst) {
    const unsigned int sig = ((++pr.posts % kSignalEvery) == 0) ? IBV_SEND_SIGNALED : 0;
    if (pr.qpx) {
      ibv_wr_start(pr.qpx);
      pr.qpx->wr_id = 0;
      pr.qpx->wr_flags = IBV_SEND_INLINE | sig;
      ibv_wr_rdma_write(pr.qpx, pr.rkey, dst);
      ibv_wr_set_inline_data(pr.qpx, const_cast<void*>(buf24), 24);
      if (ibv_wr_complete(pr.qpx)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(inline24) failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
      }
      return;
    }
    ibv_sge sge{(uint64_t)(uintptr_t)buf24, 24, 0};
    ibv_send_wr wr{};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_INLINE | sig;
    wr.wr.rdma.remote_addr = dst;
    wr.wr.rdma.rkey = pr.rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(pr.qp, &wr, &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send(inline24) failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
    }
  }

  // s6: single 64-byte BlueFlame 2-SGE RDMA WRITE (payload + trailing 8-byte flag in ONE WR & ONE final packet!)
  inline void post_piggyback(Peer& pr, uint64_t src, uint32_t len, uint64_t dst, uint64_t flag_src_addr, uint64_t flag) {
    if (len == 0) {
      post_flag_only(pr, dst, flag);
      return;
    }
    Hca& h = hcas_[pr.hca];
    const unsigned int sig = ((++pr.posts % kSignalEvery) == 0) ? IBV_SEND_SIGNALED : 0;
    if (len + 8 <= h.max_inline && len + 8 <= 264u) {
      alignas(16) char tmp[264];
      memcpy(tmp, reinterpret_cast<const void*>(src), len);
      memcpy(tmp + len, &flag, 8);
      if (pr.qpx) {
        ibv_wr_start(pr.qpx);
        pr.qpx->wr_id = 0;
        pr.qpx->wr_flags = IBV_SEND_INLINE | sig;
        ibv_wr_rdma_write(pr.qpx, pr.rkey, dst);
        ibv_wr_set_inline_data(pr.qpx, tmp, len + 8);
        if (ibv_wr_complete(pr.qpx)) {
          fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(pb_inl) failed errno=%d\n", rank_, errno);
          proxy_error_.store(true);
        }
        return;
      }
    }
    ibv_sge sge[2];
    sge[0].addr = src;
    sge[0].length = len;
    sge[0].lkey = h.mr->lkey;
    sge[1].addr = flag_src_addr;
    sge[1].length = 8;
    sge[1].lkey = h.mr->lkey;
    if (pr.qpx) {
      ibv_wr_start(pr.qpx);
      pr.qpx->wr_id = 0;
      pr.qpx->wr_flags = sig;
      ibv_wr_rdma_write(pr.qpx, pr.rkey, dst);
      ibv_wr_set_sge_list(pr.qpx, 2, sge);
      if (ibv_wr_complete(pr.qpx)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_wr_complete(pb) failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
      }
      return;
    }
    ibv_send_wr wr{};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = sge;
    wr.num_sge = 2;
    wr.send_flags = sig;
    wr.wr.rdma.remote_addr = dst;
    wr.wr.rdma.rkey = pr.rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(pr.qp, &wr, &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send(pb) failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
    }
  }

  void post(Peer& pr, uint64_t src, uint32_t len, uint64_t dst, uint64_t flag_dst, uint64_t flag) {
    Hca& h = hcas_[pr.hca];
    const bool do_inline = (len > 0 && len <= h.max_inline);
    if (len && split_post_.load(std::memory_order_relaxed)) {
      if (post_payload_only(pr, src, len, dst)) post_flag_only(pr, flag_dst, flag);
      return;
    }
    ibv_sge sge{src, len, h.mr->lkey};
    ibv_sge fsge{(uint64_t)(uintptr_t)&flag, 8, 0};
    ibv_send_wr wr[2]{};
    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].sg_list = &sge;
    wr[0].num_sge = 1;
    wr[0].send_flags = do_inline ? IBV_SEND_INLINE : 0;
    wr[0].wr.rdma.remote_addr = dst;
    wr[0].wr.rdma.rkey = pr.rkey;
    wr[0].next = &wr[1];
    wr[1].opcode = IBV_WR_RDMA_WRITE;
    wr[1].sg_list = &fsge;
    wr[1].num_sge = 1;
    wr[1].send_flags = IBV_SEND_INLINE | ((++pr.posts % kSignalEvery) == 0 ? IBV_SEND_SIGNALED : 0);
    wr[1].wr.rdma.remote_addr = flag_dst;
    wr[1].wr.rdma.rkey = pr.rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(pr.qp, len ? &wr[0] : &wr[1], &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
    }
  }

  void proxy_loop_worker(int hca_lo, int hca_hi, int core, bool is_primary) {
    if (core >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(core, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    if (proxy_rt_) {
      sched_param sp{};
      sp.sched_priority = 50;
      pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);  // best-effort (succeeds in --privileged)
    }
    // Pre-build owned HCAs & peers (dual_qp_: primary owns even HCAs = Root 0, helper owns odd HCAs = Root 2!)
    std::vector<int> my_hcas;
    std::vector<int> my_peers;
    std::vector<int> all_peers_rot;
    std::vector<QpRef> my_qp_all;
    if (dual_qp_) {
      const int want_par = is_primary ? 0 : 1;
      for (int hi = 0; hi < (int)hcas_.size(); ++hi) {
        if ((hi & 1) == want_par) my_hcas.push_back(hi);
      }
      for (int i = 1; i < world_; ++i) {
        const int q = (rank_ + i) % world_;
        all_peers_rot.push_back(q);
        Peer& pr = peers_[q];
        if ((pr.hca & 1) == want_par) {
          my_peers.push_back(q);
          my_qp_all.push_back(QpRef{pr.hca, pr.qp, pr.qpx, pr.rkey, pr.rkey_ro, pr.remote_base, &pr.posts});
        } else {
          my_qp_all.push_back(QpRef{pr.hca2, pr.qp2, pr.qpx2, pr.rkey2, pr.rkey2_ro, pr.remote_base, &pr.posts2});
        }
      }
    } else {
      for (int hi = hca_lo; hi < hca_hi; ++hi) my_hcas.push_back(hi);
      for (int i = 1; i < world_; ++i) {
        const int q = (rank_ + i) % world_;
        if (peers_[q].hca >= hca_lo && peers_[q].hca < hca_hi) my_peers.push_back(q);
      }
    }
    const bool multi = multi_proxy_ && (int)hcas_.size() >= 2;
    const uint64_t base = (uint64_t)(uintptr_t)arena_;
    uint64_t last1 = 0, last2 = 0;
    uint64_t p2_q[4] = {0, 0, 0, 0};
    uint32_t p2_head = 0, p2_tail = 0;
    ibv_wc wc[32];
    const char* e1 = getenv("RDMA_AR_SPIN_US");
    const char* e2 = getenv("RDMA_AR_MAX_SLEEP_US");
    const auto spin_window = std::chrono::microseconds(e1 ? atol(e1) : 20000);
    const long max_sleep_us = e2 ? atol(e2) : 1000;
    auto last_active = std::chrono::steady_clock::now();
    long backoff_us = 8;
    unsigned idle_iters = 0;
    uint64_t last_ping = 0;
    bool last_two = false;
    const auto reap = [&]() {
      for (int hi : my_hcas) {
        int n;
        do {
          n = ibv_poll_cq(hcas_[hi].cq, 32, wc);
          for (int i = 0; i < n; ++i)
            if (wc[i].status != IBV_WC_SUCCESS) {
              fprintf(stderr, "rdma_ar rank %d: completion error %s (qp %u)\n", rank_,
                      ibv_wc_status_str(wc[i].status), wc[i].qp_num);
              proxy_error_.store(true);
            }
        } while (n == 32);
      }
    };

    const auto handle_p1 = [&](uint64_t s1) {
      const int slot = (int)(s1 % kSlots);
      const CtrlSlot* cs = level_db_ ? (ctrl_slots_ + slot) : nullptr;
      int64_t t_seen = 0, t_pub = 0;
      if (__builtin_expect(timing_ != 0 && is_primary, 0)) {
        timespec ts0;
        clock_gettime(CLOCK_REALTIME, &ts0);
        t_seen = (int64_t)ts0.tv_sec * 1000000000LL + ts0.tv_nsec;
        t_pub = (int64_t)__atomic_load_n(cs ? &cs->pub_time : &ctrl_->pub_time, __ATOMIC_RELAXED);
      }
      // s7/s8: send_seq was already loaded with __ATOMIC_ACQUIRE -> metadata loads are safely __ATOMIC_RELAXED (LDR)!
      const uint64_t nbytes = __atomic_load_n(cs ? &cs->send_bytes : &ctrl_->send_bytes, __ATOMIC_RELAXED);
      uint64_t mode = __atomic_load_n(cs ? &cs->mode : &ctrl_->mode, __ATOMIC_RELAXED);
      const uint64_t cv = __atomic_load_n(cs ? &cs->chunk_vecs : &ctrl_->chunk_vecs, __ATOMIC_RELAXED);

      if (__builtin_expect((mode & kModeInline) != 0, 0)) {
        // s6 fast path: 16B kGather payload lives in inline_lo/hi (already in L1 cache!)
        alignas(16) uint64_t buf24[3];
        buf24[0] = __atomic_load_n(cs ? &cs->inline_lo : &ctrl_->inline_lo, __ATOMIC_RELAXED);
        buf24[1] = __atomic_load_n(cs ? &cs->inline_hi : &ctrl_->inline_hi, __ATOMIC_RELAXED);
        buf24[2] = s1;
        const uint64_t rel_dst = inline_recv_off_ + ((uint64_t)slot * world_ + rank_) * sizeof(InlineSlot);
        for (int q : my_peers) {
          Peer& pr = peers_[q];
          post_inline24(pr, buf24, pr.remote_base + rel_dst);
        }
        mode &= ~kModeInline;
      } else {
        const uint64_t nvec = nbytes / 16;
        const bool peer_lens = (mode & kModePeerLens) != 0;
        const bool p1_stripe = dual_qp_ && ((mode & kModeP1Stripe) != 0);
        mode &= ~(kModePeerLens | kModeP1Stripe);
        uint64_t src0 = base + send_off_ + (uint64_t)slot * slot_bytes_;
        if (r3_sbuf_ && (mode & kModeSrc) != 0) {
          src0 = base + __atomic_load_n(cs ? &cs->src_off : &ctrl_->src_off, __ATOMIC_RELAXED);
          mode &= ~kModeSrc;
        }
        const uint64_t rel_recv = recv_off_ + ((uint64_t)slot * world_ + rank_) * recv_stride_;
        // s7 S2: 64B cache-line-strided peer flag offset (eliminates 7-NIC CMN-700 false sharing!)
        const uint64_t rel_flag = flags_off_ + ((uint64_t)slot * world_ + rank_) * (kFlagStride * 8ull);
        const uint64_t rel_flag_qp2 = flags_qp2_off_ + ((uint64_t)slot * world_ + rank_) * (kFlagStride * 8ull);
        const CtrlC32Slot* c32s = peer_lens ? (ctrl_c32_ + slot) : nullptr;
        auto peer_slice = [&](int q, uint64_t& out_src, uint64_t& out_len) {
          if (peer_lens) {
            out_src = src0 + (uint64_t)q * slot_chunk_;
            out_len = (uint64_t)__atomic_load_n(&c32s->p1_bytes[q], __ATOMIC_RELAXED);
          } else if (mode == kTwoShot) {
            const uint64_t b = (uint64_t)q * cv;
            out_len = b < nvec ? std::min<uint64_t>(cv, nvec - b) * 16ull : 0ull;
            out_src = src0 + b * 16ull;
          } else {
            out_src = src0;
            out_len = nbytes;
          }
        };
        if (p1_stripe) {
          // c32: Exact 50/50 Dual-Root PCIe striping! Primary posts [0, half0) on Root 0, helper posts [half0, len) on Root 2!
          const uint64_t my_rel_flag = is_primary ? rel_flag : rel_flag_qp2;
          const size_t np = all_peers_rot.size();
          bool ok_mask[kThreads];
          for (size_t idx = 0; idx < np; ++idx) {
            const int q = all_peers_rot[idx];
            const QpRef& qr = my_qp_all[idx];
            uint64_t p_src = 0, p_len = 0;
            peer_slice(q, p_src, p_len);
            uint64_t half0 = ((p_len >> 1) + 63ull) & ~63ull;
            if (half0 >= p_len) half0 = p_len & ~63ull;
            const uint64_t off = is_primary ? 0ull : half0;
            const uint64_t sub_len = is_primary ? half0 : (p_len - half0);
            ok_mask[idx] = post_payload_q(qr, p_src + off, (uint32_t)sub_len, qr.remote_base + rel_recv + off);
          }
          for (size_t idx = 0; idx < np; ++idx) {
            if (ok_mask[idx]) post_flag_q(my_qp_all[idx], my_qp_all[idx].remote_base + my_rel_flag, s1);
          }
        } else if (piggyback_) {
          // Registered per-slot flag source (unique per proxy worker: primary uses slot*64, helper uses (kSlots+slot)*64)
          const size_t foff = send_flag_off_ + ((size_t)(is_primary ? 0 : kSlots) + (size_t)slot) * 64;
          uint64_t* fptr = reinterpret_cast<uint64_t*>(base + foff);
          *fptr = s1;
          std::atomic_thread_fence(std::memory_order_release);
          for (int q : my_peers) {
            Peer& pr = peers_[q];
            uint64_t src = src0, len = nbytes;
            peer_slice(q, src, len);
            post_piggyback(pr, src, (uint32_t)len, pr.remote_base + rel_recv, base + foff, s1);
          }
        } else if (two_pass_post_ && split_post_.load(std::memory_order_relaxed)) {
          // s6/s7 2-pass interleaved split-post: DMA payloads launch first, then 8B flags;
          // small inline payloads (len <= max_inline, e.g. 128B sync-free argmax AG) batch into 1 MMIO doorbell!
          bool ok_mask[kThreads];
          const size_t np = my_peers.size();
          for (size_t idx = 0; idx < np; ++idx) {
            const int q = my_peers[idx];
            Peer& pr = peers_[q];
            uint64_t src = src0, len = nbytes;
            peer_slice(q, src, len);
            if (len <= hcas_[pr.hca].max_inline) {
              post_inline_pair(pr, src, (uint32_t)len, pr.remote_base + rel_recv, pr.remote_base + rel_flag, s1);
              ok_mask[idx] = false;
            } else {
              ok_mask[idx] = post_payload_only(pr, src, (uint32_t)len, pr.remote_base + rel_recv);
            }
          }
          for (size_t idx = 0; idx < np; ++idx) {
            if (ok_mask[idx]) {
              Peer& pr = peers_[my_peers[idx]];
              post_flag_only(pr, pr.remote_base + rel_flag, s1);
            }
          }
        } else {
          for (int q : my_peers) {
            Peer& pr = peers_[q];
            uint64_t src = src0, len = nbytes;
            peer_slice(q, src, len);
            post(pr, src, (uint32_t)len, pr.remote_base + rel_recv, pr.remote_base + rel_flag, s1);
          }
        }
      }

      if (level_db_) {
        // s8 level_db: ZERO cross-L3 p2_posted1_ sync! Each proxy writes its own isolated 64B cache line.
        if (is_primary) __atomic_store_n(&ctrl_->posted_seq, s1, __ATOMIC_RELEASE);
        else __atomic_store_n(&ctrl_aux_->posted_seq_helper, s1, __ATOMIC_RELEASE);
      } else if (multi) {
        if (is_primary) {
          while (p2_posted1_.load(std::memory_order_acquire) < s1 && !stop_.load(std::memory_order_relaxed)) {
          }
          __atomic_store_n(&ctrl_->posted_seq, s1, __ATOMIC_RELEASE);
        } else {
          p2_posted1_.store(s1, std::memory_order_release);
        }
      } else {
        __atomic_store_n(&ctrl_->posted_seq, s1, __ATOMIC_RELEASE);
      }
      last_two = (mode == kTwoShot);
      if (level_db_ && last_two) p2_q[p2_tail++ & 3u] = s1;
      if (__builtin_expect(timing_ != 0 && is_primary, 0)) {
        timespec ts1;
        clock_gettime(CLOCK_REALTIME, &ts1);
        const int64_t t_done = (int64_t)ts1.tv_sec * 1000000000LL + ts1.tv_nsec;
        px_n_.fetch_add(1, std::memory_order_relaxed);
        px_handoff_.fetch_add(t_seen - t_pub, std::memory_order_relaxed);
        px_post_.fetch_add(t_done - t_seen, std::memory_order_relaxed);
        if (t_done - t_seen > px_post_max_.load(std::memory_order_relaxed))
          px_post_max_.store(t_done - t_seen, std::memory_order_relaxed);
      }
      last1 = s1;
      // s7 S3: CQEs arrive every kSignalEvery=32 posts -> poll every 16 ops (16x fewer CQ polls, 0 overflow risk!)
      if ((s1 & 15u) == 0) reap();
    };

    const auto handle_p2 = [&](uint64_t s2) {
      const int slot = (int)(s2 % kSlots);
      const CtrlSlot* cs = level_db_ ? (ctrl_slots_ + slot) : nullptr;
      const uint64_t len = __atomic_load_n(cs ? &cs->send2_bytes : &ctrl_->send2_bytes, __ATOMIC_RELAXED);
      const uint64_t src = base + send2_off_ + (uint64_t)slot * slot_chunk_;
      const uint64_t rel_recv2 = recv2_off_ + ((uint64_t)slot * world_ + rank_) * slot_chunk_;
      const uint64_t rel_flag2 = flags2_off_ + ((uint64_t)slot * world_ + rank_) * (kFlagStride * 8ull);
      const uint64_t rel_flag2_qp2 = flags2_qp2_off_ + ((uint64_t)slot * world_ + rank_) * (kFlagStride * 8ull);
      const bool p2_stripe =
          dual_qp_ && (__atomic_load_n(&ctrl_c32_[slot].p2_stripe_seq, __ATOMIC_RELAXED) == s2);
      if (p2_stripe) {
        const uint64_t my_rel_flag2 = is_primary ? rel_flag2 : rel_flag2_qp2;
        uint64_t half0 = ((len >> 1) + 63ull) & ~63ull;
        if (half0 >= len) half0 = len & ~63ull;
        const uint64_t off = is_primary ? 0ull : half0;
        const uint64_t sub_len = is_primary ? half0 : (len - half0);
        const size_t np = all_peers_rot.size();
        bool ok_mask[kThreads];
        for (size_t idx = 0; idx < np; ++idx) {
          const QpRef& qr = my_qp_all[idx];
          ok_mask[idx] = post_payload_q(qr, src + off, (uint32_t)sub_len, qr.remote_base + rel_recv2 + off);
        }
        for (size_t idx = 0; idx < np; ++idx) {
          if (ok_mask[idx]) post_flag_q(my_qp_all[idx], my_qp_all[idx].remote_base + my_rel_flag2, s2);
        }
      } else if (piggyback_) {
        const size_t foff = send_flag_off_ + ((size_t)(is_primary ? 2 * kSlots : 3 * kSlots) + (size_t)slot) * 64;
        uint64_t* fptr = reinterpret_cast<uint64_t*>(base + foff);
        *fptr = s2;
        std::atomic_thread_fence(std::memory_order_release);
        for (int q : my_peers) {
          Peer& pr = peers_[q];
          post_piggyback(pr, src, (uint32_t)len, pr.remote_base + rel_recv2, base + foff, s2);
        }
      } else if (two_pass_post_ && split_post_.load(std::memory_order_relaxed)) {
        bool ok_mask[kThreads];
        const size_t np = my_peers.size();
        for (size_t idx = 0; idx < np; ++idx) {
          Peer& pr = peers_[my_peers[idx]];
          if (len <= hcas_[pr.hca].max_inline) {
            post_inline_pair(pr, src, (uint32_t)len, pr.remote_base + rel_recv2, pr.remote_base + rel_flag2, s2);
            ok_mask[idx] = false;
          } else {
            ok_mask[idx] = post_payload_only(pr, src, (uint32_t)len, pr.remote_base + rel_recv2);
          }
        }
        for (size_t idx = 0; idx < np; ++idx) {
          if (ok_mask[idx]) {
            Peer& pr = peers_[my_peers[idx]];
            post_flag_only(pr, pr.remote_base + rel_flag2, s2);
          }
        }
      } else {
        for (int q : my_peers) {
          Peer& pr = peers_[q];
          post(pr, src, (uint32_t)len, pr.remote_base + rel_recv2, pr.remote_base + rel_flag2, s2);
        }
      }
      if (level_db_) {
        if (is_primary) __atomic_store_n(&ctrl_->posted2_seq, s2, __ATOMIC_RELEASE);
        else __atomic_store_n(&ctrl_aux_->posted2_seq_helper, s2, __ATOMIC_RELEASE);
      } else if (multi) {
        if (is_primary) {
          while (p2_posted2_.load(std::memory_order_acquire) < s2 && !stop_.load(std::memory_order_relaxed)) {
          }
          __atomic_store_n(&ctrl_->posted2_seq, s2, __ATOMIC_RELEASE);
        } else {
          p2_posted2_.store(s2, std::memory_order_release);
        }
      } else {
        __atomic_store_n(&ctrl_->posted2_seq, s2, __ATOMIC_RELEASE);
      }
      last2 = s2;
      if ((s2 & 15u) == 0) reap();
    };

    while (!stop_.load(std::memory_order_relaxed)) {
      // s6/s7 fast-path hyg_proxy dispatch: when the expected doorbell rings, handle it IMMEDIATELY
      // without reading cold ctrl_->ping or the other doorbell's cache line!
      if (hyg_proxy_.load(std::memory_order_relaxed)) {
        const bool e2 = last_two && last2 != last1;
        if (level_db_) {
          const uint64_t want1 = last1 + 1;
          const uint64_t* db1 = &ctrl_slots_[want1 % kSlots].send_seq;
          const bool has_p2 = (p2_head != p2_tail);
          const uint64_t want2 = has_p2 ? p2_q[p2_head & 3u] : 0ull;
          const uint64_t* db2 = has_p2 ? &ctrl_slots_[want2 % kSlots].send2_seq : nullptr;
          int fired = 0;
          for (int k = 0; k < kHygSpin; ++k) {
            if (has_p2 && __atomic_load_n(db2, __ATOMIC_ACQUIRE) == want2) {
              fired = 2;
              break;
            }
            if (__atomic_load_n(db1, __ATOMIC_ACQUIRE) == want1) {
              fired = 1;
              break;
            }
          }
          if (fired == 2) {
            handle_p2(want2);
            ++p2_head;
            idle_iters = 0;
            backoff_us = 8;
            continue;
          }
          if (fired == 1) {
            handle_p1(want1);
            idle_iters = 0;
            backoff_us = 8;
            continue;
          }
        } else {
          const uint64_t* db = e2 ? &ctrl_->send2_seq : &ctrl_->send_seq;
          const uint64_t old = e2 ? last2 : last1;
          uint64_t v = old;
          for (int k = 0; k < kHygSpin; ++k) {
            v = __atomic_load_n(db, __ATOMIC_ACQUIRE);
            if (v != old) break;
          }
          if (v != old) {
            if (e2) handle_p2(v); else handle_p1(v);
            idle_iters = 0;
            backoff_us = 8;
            continue;
          }
        }
      }
      bool idle = true;
      if (is_primary) {
        const uint64_t pg = __atomic_load_n(&ctrl_->ping, __ATOMIC_ACQUIRE);
        if (pg != last_ping) {
          __atomic_store_n(&ctrl_->pong, pg, __ATOMIC_RELEASE);
          last_ping = pg;
          idle = false;
        }
      }
      if (level_db_) {
        const uint64_t want1 = last1 + 1;
        const uint64_t s1 = __atomic_load_n(&ctrl_slots_[want1 % kSlots].send_seq, __ATOMIC_ACQUIRE);
        if (s1 == want1) {
          idle = false;
          handle_p1(s1);
        }
        if (p2_head != p2_tail) {
          const uint64_t want2 = p2_q[p2_head & 3u];
          const uint64_t s2 = __atomic_load_n(&ctrl_slots_[want2 % kSlots].send2_seq, __ATOMIC_ACQUIRE);
          if (s2 == want2) {
            idle = false;
            handle_p2(s2);
            ++p2_head;
          }
        }
      } else {
        const uint64_t s1 = __atomic_load_n(&ctrl_->send_seq, __ATOMIC_ACQUIRE);
        if (s1 != last1) {
          idle = false;
          handle_p1(s1);
        }
        const uint64_t s2 = __atomic_load_n(&ctrl_->send2_seq, __ATOMIC_ACQUIRE);
        if (s2 != last2) {
          idle = false;
          handle_p2(s2);
        }
      }
      if (idle) {
        if (!hyg_proxy_.load(std::memory_order_relaxed) || (idle_iters & 1023) == 0) reap();
        ++idle_iters;
        if (idle_iters == 256) {
          last_active = std::chrono::steady_clock::now();
        } else if (idle_iters > 256 && (idle_iters & 255) == 0) {
          if (std::chrono::steady_clock::now() - last_active > spin_window) {
            std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
            backoff_us = std::min(backoff_us * 2, max_sleep_us);
          }
          if (idle_iters >= 0x40000000u) idle_iters = 512u;
        }
      } else {
        idle_iters = 0;
        backoff_us = 8;
      }
    }
  }

  const int* rows_dev_ = nullptr;  // w3 rowar
  const int* skip_dev_ = nullptr;  // s5 cond gather
  long long rows_row_vec_ = 0, rows_full_ = 0;
  bool rows_active_ = false;
  int rows_oneshot_max_ = getenv("RDMA_AR_ROWS_ONESHOT_MAX") ? atoi(getenv("RDMA_AR_ROWS_ONESHOT_MAX")) : 0;
  int rows_p2_ = getenv("RDMA_AR_ROWS_P2") ? atoi(getenv("RDMA_AR_ROWS_P2")) : 0;
  long long rows_s1_bytes_ = getenv("RDMA_AR_ROWS_S1_BYTES") ? atoll(getenv("RDMA_AR_ROWS_S1_BYTES")) : 28672;
  bool rows_auto_ = (rows_oneshot_max_ > 0 || rows_p2_ != 0 || (getenv("RDMA_AR_ROWS_AUTO") && atoi(getenv("RDMA_AR_ROWS_AUTO")) != 0));
  // s6 env knobs (safe defaults: qp_ex=1, two_pass_post=1; experimental wire/GPU features opt-in via env!)
  bool use_qp_ex_ = getenv("RDMA_AR_QP_EX") ? (atoi(getenv("RDMA_AR_QP_EX")) != 0) : true;
  bool two_pass_post_ = getenv("RDMA_AR_TWO_PASS_POST") ? (atoi(getenv("RDMA_AR_TWO_PASS_POST")) != 0) : true;
  bool piggyback_ = false;  // s6-v2: permanently disabled (unsafe under variable sizes + ConnectX-7 PCIe RO)
  bool l2_relay_ = env_flag("RDMA_AR_L2_RELAY");
  unsigned int poll_sleep_ns_ = getenv("RDMA_AR_POLL_SLEEP_NS") ? (unsigned int)std::max(0, atoi(getenv("RDMA_AR_POLL_SLEEP_NS"))) : 0u;
  bool inline_ag_ = env_flag("RDMA_AR_INLINE_AG");
  bool multi_proxy_ = env_flag("RDMA_AR_MULTI_PROXY");
  int proxy_core2_ = getenv("RDMA_AR_PROXY_CORE2") ? atoi(getenv("RDMA_AR_PROXY_CORE2")) : 19;
  bool proxy_rt_ = env_flag("RDMA_AR_PROXY_RT");
  // s8 opt-in flags (default OFF -> 100% s7-v3 compatible until enabled!)
  bool level_db_ = env_flag("RDMA_AR_LEVEL_DB");
  bool wire_bf12_ = env_flag("RDMA_AR_WIRE_BF12");
  bool bf12_os4_ = env_flag("RDMA_AR_ROWS_BF12_OS4");
  // c32 opt-in flags (ALL default OFF -> 100% s8/s7 C=1 compatible until enabled!)
  bool c32_kernel_ = env_flag("RDMA_AR_C32_KERNEL");
  int c32_bf12_mode_ = []() {
    const std::string v = env_r3_norm("RDMA_AR_C32_BF12");
    if (v == "both" || v == "p1p2" || v == "2") return 2;
    if (v == "p2" || v == "1" || v == "on" || v == "true") return 1;
    return 0;
  }();
  bool c32_pipe_ = env_flag("RDMA_AR_C32_PIPE");
  bool dual_qp_ = env_flag("RDMA_AR_DUAL_QP");
  bool pci_ro_ = env_flag("RDMA_AR_PCI_RO");
  bool recv_decouple_ = env_flag("RDMA_AR_RECV_DECOUPLE");
  long long max_ctas_ = getenv("RDMA_AR_MAX_CTAS")
                            ? std::max<long long>(1LL, std::min<long long>(48LL, atoll(getenv("RDMA_AR_MAX_CTAS"))))
                            : 16LL;
  long long c32_min_nvec_ = getenv("RDMA_AR_C32_MIN_NVEC") ? atoll(getenv("RDMA_AR_C32_MIN_NVEC")) : 18432LL;  // M=32
  long long c32_bf12_p2_min_bytes_ = getenv("RDMA_AR_C32_BF12_P2_MIN_BYTES") ? atoll(getenv("RDMA_AR_C32_BF12_P2_MIN_BYTES")) : 262144LL;
  long long c32_bf12_p1_min_bytes_ = getenv("RDMA_AR_C32_BF12_P1_MIN_BYTES") ? atoll(getenv("RDMA_AR_C32_BF12_P1_MIN_BYTES")) : 524288LL;
  long long c32_pipe_min_bytes_ = getenv("RDMA_AR_C32_PIPE_MIN_BYTES") ? atoll(getenv("RDMA_AR_C32_PIPE_MIN_BYTES")) : 1048576LL;
  long long stripe_min_bytes_ = getenv("RDMA_AR_STRIPE_MIN_BYTES") ? atoll(getenv("RDMA_AR_STRIPE_MIN_BYTES")) : 32768LL;
  int rank_, world_;
  int64_t max_bytes_, max_reduce_bytes_ = 0;
  int gid_index_, ib_port_, proxy_core_;
  long long timeout_ns_;
  int device_;
  size_t max_chunk_ = 0, slot_bytes_ = 0, slot_chunk_ = 0, recv_stride_ = 0;
  void* arena_ = nullptr;
  char* arena_dev_ = nullptr;
  size_t arena_bytes_ = 0, flags_off_ = 0, flags2_off_ = 0, flags_qp2_off_ = 0, flags2_qp2_off_ = 0,
         send_flag_off_ = 0, inline_recv_off_ = 0,
         send_off_ = 0, send2_off_ = 0, recv_off_ = 0, recv2_off_ = 0;
  Ctrl* ctrl_ = nullptr;
  CtrlSlot* ctrl_slots_ = nullptr;
  CtrlAux* ctrl_aux_ = nullptr;
  CtrlC32Slot* ctrl_c32_ = nullptr;
  char* sbuf_ring_ptrs_[2][kSbufRingPerDomain] = {{nullptr, nullptr, nullptr, nullptr},
                                                  {nullptr, nullptr, nullptr, nullptr}};
  size_t sbuf_ring_cap_ = 0;
  int sbuf_ring_cur_[2] = {0, 0};
  DevState* st_ = nullptr;
  uint4* reduced_buf_ = nullptr;
  uint4* bf12_os_scratch_ = nullptr;
  std::vector<Hca> hcas_;
  std::vector<Peer> peers_;
  std::thread proxy_;
  std::thread proxy2_;
  alignas(64) std::atomic<uint64_t> p2_posted1_{0};
  alignas(64) std::atomic<uint64_t> p2_posted2_{0};
  alignas(64) std::atomic<bool> stop_{false};
  std::atomic<bool> proxy_error_{false};
  std::atomic<int64_t> px_n_{0}, px_handoff_{0}, px_post_{0}, px_post_max_{0};
  bool arena_registered_ = false;
  int64_t thp_bytes_ = 0;
  int timing_ = getenv("RDMA_AR_TIMING") ? atoi(getenv("RDMA_AR_TIMING")) : 0;
  long long vec_per_cta_ = getenv("RDMA_AR_VEC_PER_CTA") ? std::max(64LL, atoll(getenv("RDMA_AR_VEC_PER_CTA"))) : 512;
  int parload_ = env_parload_default();
  // g130/s7 flags: see the file header. Off when unset. pub_mask_ acts at the next launch, the other two at the next message.
  static constexpr int kHygSpin = 4096;  // s7 S3: ~4 us inner spin on 3.9 GHz X925 (eliminates slow-path thrash!)
  int pub_mask_ = env_pub_mask();
  std::atomic<bool> split_post_{env_flag("RDMA_AR_SPLIT_POST")};
  std::atomic<bool> hyg_proxy_{env_flag("RDMA_AR_HYG_PROXY")};
  // r3-comm (off when unset; read once per instance). r3_sbuf_ is fixed before the proxy starts (connect()).
  int r3_ = env_r3_mask();
  long long r3_s1_bytes_ = r3_ ? env_r3_s1_bytes() : 24576;
  int r6_pdl_ = env_r6_pdl();  // r6-comm C2: 0, kR6Pdl or kR6Pdl | kR6Trig (DSV41_RDMA_PDL; needs TRIM)
  bool r7_proj_ = env_r7_proj();  // r7-comm2 HT13: all_reduce_proj allowed (DSV41_RDMA_PROJ; needs TRIM)
  bool r3_sbuf_ = false;
  size_t sbuf_off_ = 0, sbuf_bytes_ = 0, sbuf_used_ = 0;
  std::unordered_map<uintptr_t, size_t> sbuf_allocs_;  // device ptr of each send buffer -> its bytes

 public:
  // g130: [split_post, hyg_proxy, pub_mask]. set_variant is for benches: call it with the stream drained and the same
  // values on every rank's quiet point. A CUDA graph keeps the kernel variant it was captured with.
  std::vector<int64_t> variant() const {
    return {split_post_.load() ? 1 : 0, hyg_proxy_.load() ? 1 : 0, pub_mask_};
  }
  void set_variant(bool split_post, bool hyg_proxy, int pub_mask) {
    TORCH_CHECK(pub_mask >= 0 && pub_mask <= 7, "rdma_ar: bad pub_mask ", pub_mask);
    split_post_.store(split_post);
    hyg_proxy_.store(hyg_proxy);
    pub_mask_ = pub_mask;
  }
  // k1/3-rdma-loads: 1 = parallel-load kernels (default of this .so), 0 = legacy ar_kernel. Takes effect
  // at the next launch, so a CUDA graph keeps the variant it was captured with. All ranks must agree
  // only on the op order, not on the variant (the protocol is identical), but keep them equal anyway.
  void set_parload(bool on) { parload_ = on ? 1 : 0; }
  // r3-comm: [mask (1 trim, 2 sendbuf), S1 bytes, send-buffer region bytes, bytes handed out, buffers]
  std::vector<int64_t> r3() const {
    return {r3_, r3_s1_bytes_, (int64_t)sbuf_bytes_, (int64_t)sbuf_used_, (int64_t)sbuf_allocs_.size()};
  }
  // r3-comm, benches only: call with the stream drained, same values on every rank (a graph keeps what it captured)
  void set_r3(int mask, int64_t s1_bytes) {
    TORCH_CHECK(mask == 0 || mask == 1 || (mask == 3 && r3_sbuf_), "rdma_ar r3: bad mask ", mask,
                " (2 needs 1; 3 needs a DSV41_RDMA_SENDBUF=1 instance)");
    r3_ = mask;
    r3_s1_bytes_ = std::max<int64_t>(0, s1_bytes);
  }
  // r6-comm C2: the PDL bits of the r3 launches (0, 128 = PDL, 384 = PDL + trigger); set_r6 for benches only (stream
  // drained, same value on every rank; a graph keeps what it captured)
  int64_t r6() const { return r6_pdl_; }
  void set_r6(int bits) {
    TORCH_CHECK(bits == 0 || bits == kR6Pdl || bits == (kR6Pdl | kR6Trig), "rdma_ar r6: bad PDL bits ", bits);
    r6_pdl_ = bits;
  }
  // r3 G26: a registered send buffer of nbytes (uint8 CUDA view of the arena; never freed), or None when the region
  // is absent or full. An all_reduce / all_gather whose input starts at such a buffer is published from it.
  std::optional<torch::Tensor> send_buffer(int64_t nbytes) {
    if ((r3_ & 2) == 0 || !r3_sbuf_ || nbytes <= 0 || nbytes % 16 || nbytes > max_bytes_) return std::nullopt;
    const size_t need = align((size_t)nbytes, 256) + 256;  // 256-byte prefix: the 16-byte reuse header ends it
    if (sbuf_used_ + need > sbuf_bytes_) return std::nullopt;
    char* ptr = arena_dev_ + sbuf_off_ + sbuf_used_ + 256;
    sbuf_used_ += need;
    sbuf_allocs_[(uintptr_t)ptr] = (size_t)nbytes;
    return torch::from_blob(ptr, {nbytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_));
  }

  // c32: Domain-partitioned 8-slot registered SENDBUF ring (domain 0=target [0..3], domain 1=drafter [4..7]).
  // Shrinks M=32..512 zero-copy SENDBUF pool from ~680 MiB -> 36 MiB while guaranteeing seq - last >= 2 ALWAYS!
  std::optional<torch::Tensor> send_buffer_ring(int64_t nbytes, int domain) {
    if ((r3_ & 2) == 0 || !r3_sbuf_ || nbytes <= 0 || nbytes % 16 || nbytes > max_bytes_) return std::nullopt;
    const int dom = (domain == 1) ? 1 : 0;
    if (sbuf_ring_ptrs_[dom][0] == nullptr) {
      const size_t cap = (size_t)max_bytes_;
      const size_t per_slot = align(cap, 256) + 256;
      const size_t total_need = per_slot * (size_t)kSbufRingPerDomain;
      if (sbuf_used_ + total_need > sbuf_bytes_) return std::nullopt;
      for (int i = 0; i < kSbufRingPerDomain; ++i) {
        char* ptr = arena_dev_ + sbuf_off_ + sbuf_used_ + 256;
        sbuf_used_ += per_slot;
        sbuf_ring_ptrs_[dom][i] = ptr;
        sbuf_allocs_[(uintptr_t)ptr] = cap;
      }
      sbuf_ring_cap_ = cap;
    }
    const int idx = sbuf_ring_cur_[dom] & (kSbufRingPerDomain - 1);
    sbuf_ring_cur_[dom] = (idx + 1) & (kSbufRingPerDomain - 1);
    return torch::from_blob(sbuf_ring_ptrs_[dom][idx], {nbytes},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_));
  }
  void reset_send_buffer_ring(int domain) {
    const int dom = (domain == 1) ? 1 : 0;
    sbuf_ring_cur_[dom] = 0;
  }
  bool parload() const { return parload_ != 0 && world_ == kParW; }
  // [n_kernel, avg stage, avg pub->flags, avg flags->end, avg total, n_proxy, avg handoff, avg post, max post]
  // all in ns; GPU globaltimer vs CPU CLOCK_REALTIME for the hand-off (aligned on Linux drivers).
  std::vector<double> stats(bool reset) {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    const double n = h.n_timed ? (double)h.n_timed : 1.0;
    const double pn_raw = (double)px_n_.load();
    const double pn = pn_raw > 0 ? pn_raw : 1.0;
    std::vector<double> v{(double)h.n_timed, h.sum_stage / n, h.sum_net / n, h.sum_tail / n, h.sum_total / n,
                          pn_raw, px_handoff_.load() / pn, px_post_.load() / pn, (double)px_post_max_.load()};
    if (reset) {
      h.sum_stage = h.sum_net = h.sum_tail = h.sum_total = 0;
      h.n_timed = 0;
      CHECK_CUDA(cudaMemcpy(st_, &h, sizeof(DevState), cudaMemcpyHostToDevice));
      px_n_ = 0;
      px_handoff_ = 0;
      px_post_ = 0;
      px_post_max_ = 0;
    }
    return v;
  }
};

// ---- k1/3-rdma-loads single-GPU self-test (no HCA, no proxy, no network) -----------------------
// Same arena layout as RdmaAR (2 MiB THP-backed, cudaHostRegister'd host memory), but every peer flag
// and both posted_seq words are preset far in the future, so the kernel never waits: it stages,
// "publishes" to nobody, and goes straight to the consumer loops. The test writes what the peers
// would have RDMA-written into recv / recv2 (same image in all kSlots slots) and compares
// legacy vs parallel-load kernels bitwise. Used only by selftest_k1.py; not touched in serving.
class SelfTest {
 public:
  SelfTest(int rank, int world, int64_t max_bytes, int64_t max_reduce_bytes, int device)
      : rank_(rank), world_(world), max_bytes_(max_bytes), device_(device) {
    if (max_bytes % 16 || max_reduce_bytes % 16) throw std::runtime_error("selftest: sizes must be multiples of 16");
    if (world < 2 || world > kThreads || rank < 0 || rank >= world) throw std::runtime_error("selftest: bad rank/world");
    max_reduce_bytes_ = std::min(max_reduce_bytes, max_bytes);
    max_chunk_ = align((size_t)((max_reduce_bytes_ / 16 + world - 1) / world) * 16, 64);
    c10::cuda::CUDAGuard guard(device_);
    const size_t fl = align(sizeof(uint64_t) * kFlagStride * kSlots * world, 4096);
    flags_off_ = 4096;
    flags2_off_ = flags_off_ + fl;
    send_off_ = flags2_off_ + fl;
    send2_off_ = send_off_ + align(kSlots * max_bytes, 4096);
    recv_off_ = send2_off_ + align(kSlots * max_chunk_, 4096);
    recv2_off_ = recv_off_ + align((size_t)kSlots * world * max_bytes, 4096);
    const size_t kHuge = 2u << 20;
    arena_bytes_ = align(recv2_off_ + align((size_t)kSlots * world * max_chunk_, 4096), kHuge);
    void* raw = mmap(nullptr, arena_bytes_ + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw != MAP_FAILED) {
      char* a = reinterpret_cast<char*>(align((size_t)(uintptr_t)raw, kHuge));
      madvise(a, arena_bytes_, MADV_HUGEPAGE);
      memset(a, 0, arena_bytes_);
      if (cudaHostRegister(a, arena_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
        arena_ = a;
        raw_ = raw;
      } else {
        cudaGetLastError();
        munmap(raw, arena_bytes_ + kHuge);
      }
    }
    if (!arena_) {
      CHECK_CUDA(cudaHostAlloc(&arena_, arena_bytes_, cudaHostAllocMapped | cudaHostAllocPortable));
      memset(arena_, 0, arena_bytes_);
      host_alloc_ = true;
    }
    CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&arena_dev_), arena_, 0));
    CHECK_CUDA(cudaMalloc(&st_, sizeof(DevState)));
    CHECK_CUDA(cudaMemset(st_, 0, sizeof(DevState)));
    CHECK_CUDA(cudaMalloc(&st_reduced_buf_, 8u * 576u * sizeof(uint4)));
    CHECK_CUDA(cudaMemset(st_reduced_buf_, 0, 8u * 576u * sizeof(uint4)));
    CHECK_CUDA(cudaMalloc(&st_bf12_os_scratch_, 8u * 2304u * sizeof(uint4)));
    CHECK_CUDA(cudaMemset(st_bf12_os_scratch_, 0, 8u * 2304u * sizeof(uint4)));
    char* a = reinterpret_cast<char*>(arena_);
    const uint64_t far = 1ull << 62;
    uint64_t* f1 = reinterpret_cast<uint64_t*>(a + flags_off_);
    uint64_t* f2 = reinterpret_cast<uint64_t*>(a + flags2_off_);
    for (int i = 0; i < kSlots * world; ++i) f1[i * kFlagStride] = f2[i * kFlagStride] = far;
    Ctrl* c = reinterpret_cast<Ctrl*>(a);
    c->posted_seq = far;
    c->posted2_seq = far;
    CtrlAux* ca = reinterpret_cast<CtrlAux*>(a + kCtrlAuxOff);
    ca->posted_seq_helper = far;
    ca->posted2_seq_helper = far;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  ~SelfTest() {
    if (st_bf12_os_scratch_) cudaFree(st_bf12_os_scratch_);
    if (st_reduced_buf_) cudaFree(st_reduced_buf_);
    if (st_) cudaFree(st_);
    if (host_alloc_) {
      cudaFreeHost(arena_);
    } else if (arena_) {
      cudaHostUnregister(arena_);
      munmap(raw_, arena_bytes_ + (2u << 20));
    }
  }

  // img: CPU uint8 [world, max_bytes] -> recv[s][r] for every slot s (row `rank` is ignored by the kernels)
  void set_recv(torch::Tensor img) { copy_img(img, recv_off_, (size_t)max_bytes_); }
  // img: CPU uint8 [world, max_chunk] -> recv2[s][r] for every slot s
  void set_recv2(torch::Tensor img) { copy_img(img, recv2_off_, max_chunk_); }
  // c32: Per-slot image loaders for 2-stage pipelined SelfTest verification!
  void set_recv_slot(int slot, torch::Tensor img) {
    TORCH_CHECK(slot >= 0 && slot < kSlots && img.is_cpu() && img.scalar_type() == at::kByte && img.is_contiguous() &&
                    img.dim() == 2 && img.size(0) == world_ && img.size(1) <= max_bytes_,
                "selftest: bad set_recv_slot");
    const size_t row_b = (size_t)img.size(1);
    const char* s = reinterpret_cast<const char*>(img.data_ptr());
    char* d = reinterpret_cast<char*>(arena_) + recv_off_ + (size_t)slot * (size_t)world_ * (size_t)max_bytes_;
    for (int r = 0; r < world_; ++r) memcpy(d + (size_t)r * (size_t)max_bytes_, s + (size_t)r * row_b, row_b);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  void set_recv2_slot(int slot, torch::Tensor img) {
    TORCH_CHECK(slot >= 0 && slot < kSlots && img.is_cpu() && img.scalar_type() == at::kByte && img.is_contiguous() &&
                    img.dim() == 2 && img.size(0) == world_ && (size_t)img.size(1) <= max_chunk_,
                "selftest: bad set_recv2_slot");
    const size_t row_b = (size_t)img.size(1);
    const char* s = reinterpret_cast<const char*>(img.data_ptr());
    char* d = reinterpret_cast<char*>(arena_) + recv2_off_ + (size_t)slot * (size_t)world_ * max_chunk_;
    for (int r = 0; r < world_; ++r) memcpy(d + (size_t)r * max_chunk_, s + (size_t)r * row_b, row_b);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  int64_t bf12_pack_into(torch::Tensor dst_wire, torch::Tensor src_bf16) {
    TORCH_CHECK(src_bf16.is_cuda() && src_bf16.scalar_type() == at::kBFloat16 && src_bf16.is_contiguous() &&
                    dst_wire.is_cuda() && dst_wire.scalar_type() == at::kByte && dst_wire.is_contiguous(),
                "selftest bf12_pack_into: bad tensors");
    const int64_t nbytes = src_bf16.numel() * 2;
    TORCH_CHECK(nbytes % 512 == 0, "selftest bf12_pack_into: bytes must be multiple of 512 (32 uint4)");
    const uint32_t n_vec = (uint32_t)(nbytes / 16);
    const uint32_t n_tiles = n_vec >> 4;
    const size_t desc_bytes = ((size_t)n_tiles * 2u + 15u) & ~15ull;
    TORCH_CHECK((size_t)dst_wire.numel() >= 16u + desc_bytes + (size_t)n_tiles * 256u, "dst_wire too small");
    c10::cuda::CUDAGuard guard(device_);
    unsigned int* d_ovf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ovf, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_ovf, 0, sizeof(unsigned int)));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    selftest_bf12_g128_pack_kernel<<<kC32Blocks, kC32Threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(src_bf16.data_ptr()),
        reinterpret_cast<char*>(dst_wire.data_ptr()), n_vec, d_ovf);
    CHECK_CUDA(cudaGetLastError());
    unsigned int h_ovf = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_ovf, d_ovf, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_ovf));
    const uint32_t total = (uint32_t)(((16u + desc_bytes + (size_t)n_tiles * 192u + (size_t)h_ovf * 64u) + 15u) & ~15ull);
    Bf12G128Hdr hdr{n_vec, n_tiles, h_ovf, total};
    CHECK_CUDA(cudaMemcpy(dst_wire.data_ptr(), &hdr, sizeof(hdr), cudaMemcpyHostToDevice));
    return (int64_t)total;
  }
  void bf12_unpack_into(torch::Tensor dst_bf16, torch::Tensor src_wire) {
    TORCH_CHECK(dst_bf16.is_cuda() && dst_bf16.scalar_type() == at::kBFloat16 && dst_bf16.is_contiguous() &&
                    src_wire.is_cuda() && src_wire.scalar_type() == at::kByte && src_wire.is_contiguous(),
                "selftest bf12_unpack_into: bad tensors");
    const int64_t nbytes = dst_bf16.numel() * 2;
    TORCH_CHECK(nbytes % 512 == 0, "selftest bf12_unpack_into: bytes must be multiple of 512");
    const uint32_t n_vec = (uint32_t)(nbytes / 16);
    c10::cuda::CUDAGuard guard(device_);
    selftest_bf12_g128_unpack_kernel<<<kC32Blocks, kC32Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const char*>(src_wire.data_ptr()),
        reinterpret_cast<uint4*>(dst_bf16.data_ptr()), n_vec);
    CHECK_CUDA(cudaGetLastError());
  }
  void all_reduce_c32_into(torch::Tensor out, torch::Tensor x, bool bf12_p1, bool bf12_p2, bool pipe2, bool level_db) {
    const int64_t nbytes = x.numel() * 2;
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                    out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                    out.numel() == x.numel() && nbytes <= max_reduce_bytes_,
                "selftest c32: bad args");
    c10::cuda::CUDAGuard guard(device_);
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = p.nvec / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.ctrl_slots = reinterpret_cast<CtrlSlot*>(arena_dev_ + kCtrlSlotsOff);
    p.ctrl_aux = reinterpret_cast<CtrlAux*>(arena_dev_ + kCtrlAuxOff);
    p.ctrl_c32 = reinterpret_cast<CtrlC32Slot*>(arena_dev_ + kCtrlC32Off);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.recv_stride = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = 5000000000LL;
    p.level_db = level_db ? 1 : 0;
    const int pub = (kPubV1 | kPubV2 | kPollThin | kR3Trim) | r6_;
    launch_c32_kernel(p, bf12_p1, bf12_p2, pipe2, at::cuda::getCurrentCUDAStream(), pub);
    CHECK_CUDA(cudaGetLastError());
  }
  void draft_c32_h64_into(torch::Tensor out, torch::Tensor x, torch::Tensor residual, torch::Tensor gamma,
                          double eps, int bits, bool level_db) {
    const int64_t M = x.size(0);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                    M >= 16 && M <= 256 && (M % world_) == 0 && x.size(1) == 4608,
                "selftest draft_c32: bad x");
    c10::cuda::CUDAGuard guard(device_);
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = M * 576;
    p.chunk = p.nvec / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.ctrl_slots = reinterpret_cast<CtrlSlot*>(arena_dev_ + kCtrlSlotsOff);
    p.ctrl_aux = reinterpret_cast<CtrlAux*>(arena_dev_ + kCtrlAuxOff);
    p.ctrl_c32 = reinterpret_cast<CtrlC32Slot*>(arena_dev_ + kCtrlC32Off);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.recv_stride = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = 5000000000LL;
    p.level_db = level_db ? 1 : 0;
    DraftNormParams dn{reinterpret_cast<uint4*>(residual.data_ptr()),
                       reinterpret_cast<const uint4*>(gamma.data_ptr()), (float)eps, 4608, (int)M};
    const int pub = (kPubV1 | kPubV2 | kPollThin | kR3Trim) | r6_;
    launch_draft_c32_h64_kernel(p, dn, bits, at::cuda::getCurrentCUDAStream(), pub);
    CHECK_CUDA(cudaGetLastError());
  }
  int64_t ar_done_cnt() {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    return (int64_t)h.ar_done_cnt;
  }

  void all_reduce_into(torch::Tensor out, torch::Tensor x, bool two_shot, bool par, int64_t vec_per_cta, bool timing) {
    const bool f32 = x.scalar_type() == at::kFloat;
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || f32) && x.is_contiguous() && out.is_contiguous() &&
                    out.scalar_type() == x.scalar_type() && out.numel() == x.numel(),
                "selftest: bad all_reduce args");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "selftest: all-reduce too large");
    if (f32) {
      if (two_shot) go<kTwoShot, true>(x, out, par, vec_per_cta, timing);
      else go<kOneShot, true>(x, out, par, vec_per_cta, timing);
    } else {
      if (two_shot) go<kTwoShot, false>(x, out, par, vec_per_cta, timing);
      else go<kOneShot, false>(x, out, par, vec_per_cta, timing);
    }
  }
  void all_gather_into(torch::Tensor out, torch::Tensor x, bool par, int64_t vec_per_cta, bool timing) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "selftest: bad all_gather args");
    go<kGather, false>(x, out, par, vec_per_cta, timing);
  }
  void all_gather_cond_into(torch::Tensor out, torch::Tensor x, torch::Tensor skip_dev, bool par, int64_t vec_per_cta, bool timing) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "selftest: bad all_gather_cond args");
    TORCH_CHECK(skip_dev.is_cuda() && skip_dev.scalar_type() == at::kInt && skip_dev.numel() >= 1 && skip_dev.is_contiguous(),
                "selftest: bad skip_dev");
    st_skip_dev_ = reinterpret_cast<const int*>(skip_dev.data_ptr());
    try { go<kGather, false>(x, out, par, vec_per_cta, timing); }
    catch (...) { st_skip_dev_ = nullptr; throw; }
    st_skip_dev_ = nullptr;
  }
  void all_reduce_rows_auto_into(torch::Tensor out, torch::Tensor x, torch::Tensor n_live, int64_t row_bytes,
                                 int oneshot_max_rows, int rows_p2, int64_t s1_bytes, int64_t vec_per_cta, bool timing) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && out.is_contiguous() &&
                    out.scalar_type() == x.scalar_type() && out.numel() == x.numel(),
                "selftest: bad rows_auto args");
    TORCH_CHECK(n_live.is_cuda() && n_live.scalar_type() == at::kInt && n_live.numel() >= 1 && n_live.is_contiguous(),
                "selftest: bad n_live");
    TORCH_CHECK(r3_ == 1 && pub_ == (kPubV1 | kPubV2 | kPollThin) && world_ == kParW, "selftest: rows_auto needs r3=1 pub=7 world=8");
    const int64_t nbytes = x.numel() * x.element_size();
    c10::cuda::CUDAGuard guard(device_);
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = 5000000000LL;
    p.timing = timing ? 1 : 0;
    p.rows_ptr = reinterpret_cast<const int*>(n_live.data_ptr());
    p.row_vec = row_bytes / 16;
    p.nvec_full = p.nvec;
    p.oneshot_max_rows = oneshot_max_rows;
    p.rows_p2 = rows_p2;
    p.s1_bytes = s1_bytes;
    launch_rows_kernel(p, std::max<long long>(64, vec_per_cta), at::cuda::getCurrentCUDAStream(), pub_ | kR3Trim | r6_);
  }
  // s8: single-GPU 0-ULP verification of ar_sconv_attnres_kernel (supports raw 1-shot, 2-shot, BF12 P2, BF12-OS4, level_db)
  torch::Tensor all_reduce_sconv_attnres_into(
      torch::Tensor x, torch::Tensor n_live,
      torch::Tensor conv_w, torch::Tensor cache,
      torch::Tensor positions, torch::Tensor block_table, torch::Tensor seq_idx,
      torch::Tensor slot_map, torch::Tensor q_start, int64_t stream_off,
      torch::Tensor conv_out,
      std::optional<torch::Tensor> aux_out,
      std::optional<torch::Tensor> conv_res,
      std::optional<torch::Tensor> conv_addend,
      torch::Tensor stack,
      std::optional<torch::Tensor> mix_last,
      torch::Tensor queries, torch::Tensor norm_w,
      double eps, double norm_eps,
      int oneshot_max_rows, bool wire_bf12, bool bf12_os4, bool level_db) {
    TORCH_CHECK(r3_ == 1 && pub_ == (kPubV1 | kPubV2 | kPollThin) && world_ == kParW,
                "selftest s8: needs r3=1 pub=7 world=8");
    c10::cuda::CUDAGuard guard(device_);
    const int T = (int)x.size(0);
    const int nq = (int)queries.size(0);
    const int n_stack = (int)stack.size(0);
    const bool has_ca = conv_addend.has_value();
    const bool has_cr = conv_res.has_value();
    const bool has_ml = mix_last.has_value();
    const int kcase = has_ca ? (has_cr ? 3 : 2) : (has_ml ? 1 : 0);
    auto out = torch::empty({nq, T, 4608}, x.options());
    Params p{};
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(conv_out.data_ptr());
    p.nvec = (long long)T * 576LL;
    p.chunk = (long long)T * 72LL;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = 5000000000LL;
    p.rows_ptr = reinterpret_cast<const int*>(n_live.data_ptr());
    p.row_vec = 576;
    p.nvec_full = (long long)T * 576LL;
    p.oneshot_max_rows = oneshot_max_rows;
    p.level_db = level_db ? 1 : 0;
    p.wire_bf12 = wire_bf12 ? 1 : 0;
    p.bf12_os4 = bf12_os4 ? 1 : 0;
    p.ctrl_slots = reinterpret_cast<CtrlSlot*>(arena_dev_ + kCtrlSlotsOff);
    p.ctrl_aux = reinterpret_cast<CtrlAux*>(arena_dev_ + kCtrlAuxOff);
    p.reduced_buf = st_reduced_buf_;
    p.bf12_os_scratch = st_bf12_os_scratch_;
    const int n_cache = (int)cache.size(2);
    int n_shift = -1;
    if (n_cache > 0 && (n_cache & (n_cache - 1)) == 0) {
      n_shift = 0;
      for (int v = n_cache; v > 1; v >>= 1) ++n_shift;
    }
    SconvAttnResParams sp{};
    sp.conv_w = reinterpret_cast<const uint16_t*>(conv_w.data_ptr());
    sp.cache = reinterpret_cast<uint16_t*>(cache.data_ptr());
    sp.conv_out = reinterpret_cast<uint16_t*>(conv_out.data_ptr());
    sp.aux_out = aux_out.has_value() ? reinterpret_cast<uint16_t*>(aux_out->data_ptr()) : nullptr;
    sp.conv_res = has_cr ? reinterpret_cast<const uint16_t*>(conv_res->data_ptr()) : nullptr;
    sp.conv_addend = has_ca ? reinterpret_cast<const uint16_t*>(conv_addend->data_ptr()) : nullptr;
    sp.pos = reinterpret_cast<const int64_t*>(positions.data_ptr());
    sp.seq_idx = reinterpret_cast<const int32_t*>(seq_idx.data_ptr());
    sp.slot_map = reinterpret_cast<const int64_t*>(slot_map.data_ptr());
    sp.block_table = reinterpret_cast<const int32_t*>(block_table.data_ptr());
    sp.q_start = reinterpret_cast<const int32_t*>(q_start.data_ptr());
    sp.stack = reinterpret_cast<const __nv_bfloat16*>(stack.data_ptr());
    sp.mix_last = has_ml ? reinterpret_cast<const __nv_bfloat16*>(mix_last->data_ptr()) : nullptr;
    sp.queries = reinterpret_cast<const float*>(queries.data_ptr());
    sp.norm_w = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr());
    sp.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    sp.stride_c_blk = cache.stride(0);
    sp.stride_c_n = cache.stride(2);
    sp.stride_bt = block_table.stride(0);
    sp.stream_off = stream_off;
    sp.stride_s = stack.stride(0);
    sp.max_blocks = (int)block_table.size(1);
    sp.n_cache = n_cache;
    sp.n_shift = n_shift;
    sp.nq = nq;
    sp.T = T;
    sp.scale = 0.44721359549995793928f;
    sp.eps = (float)eps;
    sp.norm_eps = (float)norm_eps;
    launch_sconv_attnres_kernel(p, sp, n_stack, kcase, at::cuda::getCurrentCUDAStream(), pub_ | kR3Trim | r6_);
    CHECK_CUDA(cudaGetLastError());
    return out;
  }
  std::vector<int64_t> peek_ctrl() {
    CHECK_CUDA(cudaDeviceSynchronize());
    const Ctrl* c = reinterpret_cast<const Ctrl*>(arena_);
    return {(int64_t)c->send_seq, (int64_t)c->send_bytes, (int64_t)c->mode, (int64_t)c->chunk_vecs,
            (int64_t)c->send2_seq, (int64_t)c->send2_bytes};
  }
  // r7-comm2 HT13: all_reduce_into + the projection tail (needs set_r3(1)); y = _proj_rows(out, d)
  void all_reduce_proj_into(torch::Tensor out, torch::Tensor x, bool two_shot, torch::Tensor d, torch::Tensor y, bool par,
                            int64_t vec_per_cta, bool timing) {
    TORCH_CHECK(r3_ == 1, "selftest r7: the projection tail needs set_r3(1)");
    TORCH_CHECK(out.is_contiguous() && out.scalar_type() == x.scalar_type() && out.numel() == x.numel(),
                "selftest: bad all_reduce args");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "selftest: all-reduce too large");
    const int64_t rows = r7_check(x, out, d, y);
    const R7Proj q{reinterpret_cast<const float*>(d.data_ptr()), reinterpret_cast<uint4*>(y.data_ptr()), rows};
    if (two_shot) go<kTwoShot, false>(x, out, par, vec_per_cta, timing, 0, &q);
    else go<kOneShot, false>(x, out, par, vec_per_cta, timing, 0, &q);
  }
  // r7-comm2 HT13, test only: st->arrive3 (must be back at 0 after every kernel)
  int64_t arrive3() {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    return (int64_t)h.arrive3;
  }
  // g2 colltrim: column all-gather through the same kernels (see RdmaAR::all_gather_cols)
  void all_gather_cols_into(torch::Tensor out, torch::Tensor x, int64_t seg_bytes, bool par, int64_t vec_per_cta,
                            bool timing) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "selftest: bad all_gather_cols args");
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(seg_bytes > 0 && seg_bytes % 16 == 0 && nbytes % seg_bytes == 0, "selftest: bad seg_bytes ", seg_bytes);
    go<kGatherCols, false>(x, out, par, vec_per_cta, timing, seg_bytes / 16);
  }
  // [n_timed, avg stage, avg pub->flags, avg flags->end (tail), avg total] in ns
  std::vector<double> stats(bool reset) {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    const double n = h.n_timed ? (double)h.n_timed : 1.0;
    std::vector<double> v{(double)h.n_timed, h.sum_stage / n, h.sum_net / n, h.sum_tail / n, h.sum_total / n};
    if (reset) {
      h.sum_stage = h.sum_net = h.sum_tail = h.sum_total = 0;
      h.n_timed = 0;
      CHECK_CUDA(cudaMemcpy(st_, &h, sizeof(DevState), cudaMemcpyHostToDevice));
    }
    return v;
  }
  int64_t max_chunk() const { return (int64_t)max_chunk_; }
  int64_t max_bytes() const { return max_bytes_; }
  int64_t max_reduce_bytes() const { return max_reduce_bytes_; }
  bool thp() const { return !host_alloc_; }
  // g130: kernel variant for the following calls (par = true only); default 0 == production kernels
  void set_pub(int pub_mask) {
    TORCH_CHECK(pub_mask >= 0 && pub_mask <= 7, "selftest: bad pub_mask ", pub_mask);
    pub_ = pub_mask;
  }
  // r3-comm: 0 = off, 1 = trim kernels (needs set_pub(7) and par); s1_bytes as DSV41_RDMA_TRIM_S1_BYTES
  void set_r3(int mask, int64_t s1_bytes) {
    TORCH_CHECK(mask == 0 || mask == 1, "selftest: bad r3 mask ", mask);
    r3_ = mask;
    r3_s1_bytes_ = std::max<int64_t>(0, s1_bytes);
  }
  // r6-comm C2: PDL bits of the r3 launches (0, 128, 384); only with set_r3(1)
  void set_r6(int bits) {
    TORCH_CHECK(bits == 0 || bits == kR6Pdl || bits == (kR6Pdl | kR6Trig), "selftest: bad r6 bits ", bits);
    r6_ = bits;
  }
  // r3-comm, test only: bytes [0, nbytes) of send slot `slot` (which = 0) or send2 slot (which = 1), and st->seq /
  // st->entered / st->arrive / st->arrive2 / st->depart (the counters must be back at 0 after every kernel)
  torch::Tensor peek_send(int which, int slot, int64_t nbytes) {
    TORCH_CHECK(slot >= 0 && slot < kSlots && nbytes >= 0 && nbytes <= (which ? (int64_t)max_chunk_ : max_bytes_),
                "selftest: bad peek");
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t = torch::empty({nbytes}, torch::TensorOptions().dtype(torch::kUInt8));
    const char* a = reinterpret_cast<const char*>(arena_) + (which ? send2_off_ + (size_t)slot * max_chunk_
                                                                       : send_off_ + (size_t)slot * max_bytes_);
    memcpy(t.data_ptr(), a, (size_t)nbytes);
    return t;
  }
  std::vector<int64_t> counters() {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    return {(int64_t)h.seq, (int64_t)h.entered, (int64_t)h.arrive, (int64_t)h.arrive2, (int64_t)h.depart};
  }

 private:
  static size_t align(size_t x, size_t a) { return (x + a - 1) / a * a; }
  // img [world, w] with w <= row: writes the first w bytes of row r of every slot (the rest is left alone)
  void copy_img(const torch::Tensor& img, size_t off, size_t row) {
    TORCH_CHECK(img.device().is_cpu() && img.scalar_type() == at::kByte && img.is_contiguous() && img.dim() == 2 &&
                    img.size(0) == world_ && (size_t)img.size(1) <= row,
                "selftest: image must be CPU uint8 [world, <= row bytes]");
    CHECK_CUDA(cudaDeviceSynchronize());
    const size_t w = (size_t)img.size(1);
    const char* src = reinterpret_cast<const char*>(img.data_ptr());
    char* a = reinterpret_cast<char*>(arena_) + off;
    for (int s = 0; s < kSlots; ++s)
      for (int r = 0; r < world_; ++r) memcpy(a + ((size_t)s * world_ + r) * row, src + (size_t)r * w, w);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  template <int kMode, bool kF32>
  void go(const torch::Tensor& x, torch::Tensor& out, bool par, int64_t vec_per_cta, bool timing, int64_t seg_vec = 0,
          const R7Proj* q7 = nullptr) {
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes > 0 && nbytes <= max_bytes_, "selftest: bad size ", nbytes);
    c10::cuda::CUDAGuard guard(device_);
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = 5000000000LL;  // 5 s: nothing should ever wait here
    p.timing = timing ? 1 : 0;
    p.seg = seg_vec;
    p.skip_ptr = st_skip_dev_;
    TORCH_CHECK(pub_ == 0 || (par && world_ == kParW), "selftest: kernel variants need par = true and world 8");
    TORCH_CHECK(r3_ == 0 || pub_ == (kPubV1 | kPubV2 | kPollThin), "selftest: r3 needs set_pub(7)");
    const int pub = (r3_ ? (pub_ | r3_bits(kMode, p.nvec, p.chunk, r3_s1_bytes_, false, kThreads) | r6_) : pub_) |
                    (q7 != nullptr ? kR7Proj : 0);
    launch_kernel<kMode, kF32>(p, std::max<long long>(64, vec_per_cta), par, at::cuda::getCurrentCUDAStream(), pub,
                               q7 != nullptr ? *q7 : R7Proj{nullptr, nullptr, 0});
  }

  const int* st_skip_dev_ = nullptr;
  int rank_, world_;
  int pub_ = 0;  // g130
  int r3_ = 0;    // r3-comm: 1 = ar_kernel_r3 (trim), selected per launch like RdmaAR
  int r6_ = 0;    // r6-comm C2: PDL bits of the r3 launches
  long long r3_s1_bytes_ = 24576;
  int64_t max_bytes_, max_reduce_bytes_ = 0;
  int device_;
  size_t max_chunk_ = 0;
  void* arena_ = nullptr;
  void* raw_ = nullptr;
  bool host_alloc_ = false;
  char* arena_dev_ = nullptr;
  size_t arena_bytes_ = 0, flags_off_ = 0, flags2_off_ = 0, send_off_ = 0, send2_off_ = 0, recv_off_ = 0,
         recv2_off_ = 0;
  DevState* st_ = nullptr;
  uint4* st_reduced_buf_ = nullptr;
  uint4* st_bf12_os_scratch_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<RdmaAR>(m, "RdmaAR")
      .def(py::init<int, int, int64_t, int64_t, std::vector<std::string>, int, int, int, int64_t, int>(),
           py::arg("rank"), py::arg("world"), py::arg("max_bytes"), py::arg("max_reduce_bytes"), py::arg("hcas"),
           py::arg("gid_index"), py::arg("ib_port"), py::arg("proxy_core"), py::arg("timeout_ms"), py::arg("device"))
      .def("local_info", &RdmaAR::local_info)
      .def("connect", &RdmaAR::connect)
      .def("all_reduce", &RdmaAR::all_reduce, py::arg("x"), py::arg("two_shot") = false)
      .def("all_gather_into", &RdmaAR::all_gather_into)
      .def("all_gather_cols", &RdmaAR::all_gather_cols, py::arg("out"), py::arg("x"), py::arg("seg_bytes"))
      .def("stats", &RdmaAR::stats, py::arg("reset") = false)
      .def("arena_info", &RdmaAR::arena_info)
      .def("pingpong", &RdmaAR::pingpong, py::arg("n") = 1000)
      .def("max_bytes", &RdmaAR::max_bytes)
      .def("max_reduce_bytes", &RdmaAR::max_reduce_bytes)
      .def("set_parload", &RdmaAR::set_parload, py::arg("on"))
      .def("parload", &RdmaAR::parload)
      .def("variant", &RdmaAR::variant)
      .def("set_variant", &RdmaAR::set_variant, py::arg("split_post"), py::arg("hyg_proxy"), py::arg("pub_mask"))
      .def("r3", &RdmaAR::r3)
      .def("set_r3", &RdmaAR::set_r3, py::arg("mask"), py::arg("s1_bytes") = 24576)
      .def("send_buffer", &RdmaAR::send_buffer, py::arg("nbytes"))
      .def("send_buffer_ring", &RdmaAR::send_buffer_ring, py::arg("nbytes"), py::arg("domain") = 0)
      .def("reset_send_buffer_ring", &RdmaAR::reset_send_buffer_ring, py::arg("domain") = 0)
      .def("r6", &RdmaAR::r6)
      .def("set_r6", &RdmaAR::set_r6, py::arg("bits"))
      .def("all_reduce_proj", &RdmaAR::all_reduce_proj, py::arg("x"), py::arg("two_shot"), py::arg("d"), py::arg("y"))
      .def("r7", &RdmaAR::r7)
      .def("set_rows", &RdmaAR::set_rows, py::arg("n_live"), py::arg("row_bytes"), py::arg("rows_full"))
      .def("all_reduce_rows", &RdmaAR::all_reduce_rows, py::arg("x"), py::arg("two_shot") = true)
      .def("all_reduce_c32_rows", &RdmaAR::all_reduce_c32_rows, py::arg("x"), py::arg("n_live"),
           py::arg("two_shot") = true)
      .def("all_gather_into_cond", &RdmaAR::all_gather_into_cond, py::arg("out"), py::arg("x"), py::arg("skip_dev"))
      .def("all_reduce_add_rmsnorm", &RdmaAR::all_reduce_add_rmsnorm, py::arg("x"), py::arg("residual"),
           py::arg("gamma"), py::arg("eps"), py::arg("use_int8") = false)
      .def("all_reduce_add_rmsnorm_v2", &RdmaAR::all_reduce_add_rmsnorm_v2, py::arg("x"), py::arg("residual"),
           py::arg("gamma"), py::arg("eps"), py::arg("mode_code") = 4, py::arg("out_h64") = false)
      .def("markov_loop_persistent", &RdmaAR::markov_loop_persistent,
           py::arg("base_logits"), py::arg("rnn_ctx"), py::arg("w1"), py::arg("rnn_w_rec"),
           py::arg("fold_w8"), py::arg("fold_s8"), py::arg("fold_bias"),
           py::arg("id_map"), py::arg("anchor_id"),
           py::arg("softcap"), py::arg("valid_vocab"), py::arg("vocab_offset"), py::arg("k_steps"),
           py::arg("greedy_dev"), py::arg("slot_dev"), py::arg("db_dev"), py::arg("lam_dev"),
           py::arg("step_eff"), py::arg("cal"), py::arg("vw_w"), py::arg("vw_cpred"),
           py::arg("draft_out"), py::arg("conf_out"), py::arg("w_cap"),
           py::arg("scratch_state"), py::arg("scratch_latent"), py::arg("scratch_rec"),
           py::arg("scratch_ctrl"),
           py::arg("req_drafts") = py::none(), py::arg("sample_idx0") = py::none())
      .def("inline_ag_select_dink", &RdmaAR::inline_ag_select_dink)
      .def("ar_done_ptr", &RdmaAR::ar_done_ptr)
      .def("s6_info", &RdmaAR::s6_info)
      .def("s8_info", &RdmaAR::s8_info)
      .def("c32_info", &RdmaAR::c32_info)
      .def("all_reduce_add_rmsnorm_batched", &RdmaAR::all_reduce_add_rmsnorm_batched,
           py::arg("x"), py::arg("residual"), py::arg("gamma"), py::arg("eps"), py::arg("mode_code") = 4)
      .def("all_reduce_sconv_attnres", &RdmaAR::all_reduce_sconv_attnres,
           py::arg("x"), py::arg("conv_w"), py::arg("cache"),
           py::arg("positions"), py::arg("block_table"), py::arg("seq_idx"),
           py::arg("slot_map"), py::arg("q_start"), py::arg("stream_off"),
           py::arg("conv_out"), py::arg("aux_out") = py::none(),
           py::arg("conv_res") = py::none(), py::arg("conv_addend") = py::none(),
           py::arg("stack") = torch::Tensor(), py::arg("mix_last") = py::none(),
           py::arg("queries") = torch::Tensor(), py::arg("norm_w") = torch::Tensor(),
           py::arg("eps") = 1e-6, py::arg("norm_eps") = 1e-6, py::arg("two_shot") = true);
  py::class_<SelfTest>(m, "SelfTest")
      .def(py::init<int, int, int64_t, int64_t, int>(), py::arg("rank"), py::arg("world"), py::arg("max_bytes"),
           py::arg("max_reduce_bytes"), py::arg("device"))
      .def("set_recv", &SelfTest::set_recv)
      .def("set_recv2", &SelfTest::set_recv2)
      .def("set_recv_slot", &SelfTest::set_recv_slot, py::arg("slot"), py::arg("img"))
      .def("set_recv2_slot", &SelfTest::set_recv2_slot, py::arg("slot"), py::arg("img"))
      .def("bf12_pack_into", &SelfTest::bf12_pack_into, py::arg("dst_wire"), py::arg("src_bf16"))
      .def("bf12_unpack_into", &SelfTest::bf12_unpack_into, py::arg("dst_bf16"), py::arg("src_wire"))
      .def("all_reduce_c32_into", &SelfTest::all_reduce_c32_into,
           py::arg("out"), py::arg("x"), py::arg("bf12_p1") = false, py::arg("bf12_p2") = true,
           py::arg("pipe2") = false, py::arg("level_db") = true)
      .def("draft_c32_h64_into", &SelfTest::draft_c32_h64_into,
           py::arg("out"), py::arg("x"), py::arg("residual"), py::arg("gamma"),
           py::arg("eps") = 1e-6, py::arg("bits") = 4, py::arg("level_db") = true)
      .def("ar_done_cnt", &SelfTest::ar_done_cnt)
      .def("all_reduce_into", &SelfTest::all_reduce_into, py::arg("out"), py::arg("x"), py::arg("two_shot"),
           py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("all_gather_into", &SelfTest::all_gather_into, py::arg("out"), py::arg("x"), py::arg("par"),
           py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("all_gather_cols_into", &SelfTest::all_gather_cols_into, py::arg("out"), py::arg("x"),
           py::arg("seg_bytes"), py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("stats", &SelfTest::stats, py::arg("reset") = false)
      .def("max_chunk", &SelfTest::max_chunk)
      .def("max_bytes", &SelfTest::max_bytes)
      .def("max_reduce_bytes", &SelfTest::max_reduce_bytes)
      .def("thp", &SelfTest::thp)
      .def("set_pub", &SelfTest::set_pub, py::arg("pub_mask"))
      .def("set_r3", &SelfTest::set_r3, py::arg("mask"), py::arg("s1_bytes") = 24576)
      .def("set_r6", &SelfTest::set_r6, py::arg("bits"))
      .def("peek_send", &SelfTest::peek_send, py::arg("which"), py::arg("slot"), py::arg("nbytes"))
      .def("counters", &SelfTest::counters)
      .def("all_reduce_proj_into", &SelfTest::all_reduce_proj_into, py::arg("out"), py::arg("x"), py::arg("two_shot"),
           py::arg("d"), py::arg("y"), py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("arrive3", &SelfTest::arrive3)
      .def("all_gather_cond_into", &SelfTest::all_gather_cond_into, py::arg("out"), py::arg("x"), py::arg("skip_dev"),
           py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("all_reduce_rows_auto_into", &SelfTest::all_reduce_rows_auto_into, py::arg("out"), py::arg("x"),
           py::arg("n_live"), py::arg("row_bytes"), py::arg("oneshot_max_rows"), py::arg("rows_p2"),
           py::arg("s1_bytes"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("peek_ctrl", &SelfTest::peek_ctrl)
      .def("all_reduce_sconv_attnres_into", &SelfTest::all_reduce_sconv_attnres_into,
           py::arg("x"), py::arg("n_live"), py::arg("conv_w"), py::arg("cache"),
           py::arg("positions"), py::arg("block_table"), py::arg("seq_idx"),
           py::arg("slot_map"), py::arg("q_start"), py::arg("stream_off"),
           py::arg("conv_out"), py::arg("aux_out") = py::none(),
           py::arg("conv_res") = py::none(), py::arg("conv_addend") = py::none(),
           py::arg("stack") = torch::Tensor(), py::arg("mix_last") = py::none(),
           py::arg("queries") = torch::Tensor(), py::arg("norm_w") = torch::Tensor(),
           py::arg("eps") = 1e-6, py::arg("norm_eps") = 1e-6,
           py::arg("oneshot_max_rows") = 3, py::arg("wire_bf12") = false,
           py::arg("bf12_os4") = false, py::arg("level_db") = false);
  m.attr("K1_VARIANT") = "3-rdma-loads-v1";
  m.attr("G2_VARIANT") = "colltrim-cols-v1";  // g2 colltrim: all_gather_cols / all_gather_cols_into present
  m.attr("G130_VARIANT") = "rdma-flags-v1";   // g130: RdmaAR.variant / set_variant, SelfTest.set_pub
  m.attr("R3_VARIANT") = "r3comm-v1";         // r3-comm: RdmaAR.r3 / set_r3 / send_buffer, SelfTest.set_r3
  m.attr("R6_VARIANT") = "r6comm-c2-v1";      // r6-comm C2: RdmaAR.r6 / set_r6, SelfTest.set_r6 (DSV41_RDMA_PDL)
  m.attr("R7_VARIANT") = "r7comm2-ht13-v1";
  m.attr("S5_VARIANT") = "s5-rows-sbuf-v2";   // s5: dynamic oneshot/twoshot rows + zero-copy sendbuf + PDL
  m.attr("S6_VARIANT") = "s7-fast-rdma-v3";   // s7-v3: ar_done_cnt L2PF stop-flag, draft_norm 32ns P2 poll, 1-warp inline AG
  m.attr("S7_VARIANT") = "s7-fast-rdma-v3";
  m.attr("S8_VARIANT") = "s8v2-unified-ar-sconv-bf12-drafter-h64-markov";
  m.attr("S8_DRAFTER_VARIANT") = "s8v2-h64-q4q6q8-markov-persist-v1";
  m.attr("C32_VARIANT") = "c32-bf12g128-pipe2-dualqp-ro-sbufring-drafth64-v1";
}
