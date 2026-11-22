# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP Lossless BF10.7_SMEM2SYM Skinny GEMM (M=1..16).

Packer + ctypes runtime wrapper for `bf10_smem2sym_exact.cu`:
- Lossless 0-ULP compression of BF16 weights down to ~10.68-10.76 bits/weight
  (-4.6% to -5.2% DRAM bytes vs BF11_FLIP's 11.27 bits/weight, within 0.12 b/w
  of the Shannon entropy floor 10.56 b/w!).
- Works in pure NumPy (for local verification without PyTorch) AND PyTorch/CUDA
  (for live GB10 deployment).

Bit-exact contract:
- Every 16-bit BF16 bit pattern (all 65,536 values including subnormals, inf, NaN)
  round-trips to the LITERAL identical uint16_t before `mma.sync.m16n8k16`.
- Uses the exact same MMA tile, K-slab, split-K, and last-CTA finalize order as
  `skinny.py` and `skinny_bf11.py` -> literal `nd == 0` (0-ULP).
"""
from __future__ import annotations

import ctypes
import heapq
import os
import threading
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # Local Mac verification runs without PyTorch
    torch = None  # type: ignore[assignment]

ENABLED = os.environ.get("INKLING_SKINNY_BF10", "0").strip().lower() in ("1", "true", "on", "yes")
_LIB_PATH = os.environ.get("INKLING_BF10_LIB", "/rdma/libbf10_smem2sym_exact.so")
_QKVR_CFG = int(os.environ.get("INKLING_BF10_QKVR_CFG", "4"))
_QSIDE_CFG = int(os.environ.get("INKLING_BF10_QSIDE_CFG", "4"))
_KV_CFG = int(os.environ.get("INKLING_BF10_KV_CFG", "4"))

MAX_WARP_EXP_BYTES = 2688  # Must match bf10_smem2sym_exact.cu::MAX_WARP_EXP_BYTES
SM_SLAB_BYTES = 1024
ESC_SYM = 256  # Reserved escape symbol ID (length <= 8 bits, followed by raw 8-bit exp)

_CFGS = {
    1: (4, 1, 32),
    4: (2, 1, 16),
}

_SHAPES = {
    (1896, 4608): (_QKVR_CFG, 6),
    (1640, 4608): (_QSIDE_CFG, 6),
    (4608, 768): (1, 1),
    (128, 4608): (_KV_CFG, 9),
}

_LIB = None
_LIB_LOCK = threading.Lock()
_CACHE: dict[tuple[int, int, int], "PackedBF10Torch"] = {}
_PART_BUFS: dict[int, list] = {}
_SEM_BUFS: dict[int, list] = {}
_SCRATCH_IDX: dict[int, int] = {}
_MIN_PART_ELEMS = 262144
_MIN_SEM_TILES = 1024


def _bit_reverse(val: int, nbits: int) -> int:
    out = 0
    for _ in range(nbits):
        out = (out << 1) | (val & 1)
        val >>= 1
    return out


def build_bounded_huffman_lut2(
    exps_u8: np.ndarray,
    max_len: int = 8,
    max_syms: int = 15,
) -> tuple[dict[int, tuple[int, int]], np.ndarray]:
    """Build an optimal length-limited (<= max_len) LSB-first canonical prefix code
    over the most frequent BF16 exponents + 1 escape symbol (ESC_SYM=256), plus the
    256-entry uint32_t `sLUT2[256]` 2-symbol-at-once decode table matching
    `bf10_smem2sym_exact.cu` bit-for-bit.

    sLUT2[b] layout:
      bits [7:0]   = e0 (first exponent; 0 if cnt == 0)
      bits [15:8]  = e1 (second exponent if cnt == 2; 0 otherwise)
      bits [19:16] = total_bits consumed (len(e0)+len(e1) if cnt==2; len(e0) if cnt==1;
                     escape prefix_len if cnt==0)
      bits [21:20] = cnt (2 = 2 symbols, 1 = 1 symbol, 0 = escape prefix)
      bits [27:24] = len1 = len(e0) (used by decode1 even when cnt == 2!)
    """
    counts = np.bincount(exps_u8.ravel(), minlength=256).astype(np.int64)
    nonzero = np.where(counts > 0)[0]
    order = nonzero[np.argsort(-counts[nonzero])]

    # Select top-K symbols + ESC_SYM so Kraft tree naturally fits within max_len=8 bits.
    # Start with min(len(order), max_syms) and reduce K if max Huffman depth exceeds max_len.
    top_k = min(len(order), max_syms)
    lengths: dict[int, int] = {}
    while True:
        kept = [int(s) for s in order[:top_k]]
        esc_cnt = int(counts.sum() - counts[kept].sum())
        # Always include ESC_SYM with weight >= 1 so every unseen exponent is 100% representable!
        items = [(int(counts[s]), s) for s in kept] + [(max(1, esc_cnt), ESC_SYM)]
        if len(items) == 1:
            lengths = {items[0][1]: 1}
            break
        # Standard Huffman tree
        heap: list[tuple[int, int, object]] = []
        uid = 0
        for w, sym in items:
            heapq.heappush(heap, (w, uid, sym))
            uid += 1
        while len(heap) > 1:
            w1, _, n1 = heapq.heappop(heap)
            w2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (w1 + w2, uid, (n1, n2)))
            uid += 1
        lengths = {}

        def _walk(node: object, depth: int) -> None:
            if isinstance(node, int):
                lengths[node] = depth
            else:
                left, right = node  # type: ignore[misc]
                _walk(left, depth + 1)
                _walk(right, depth + 1)

        _walk(heap[0][2], 0)
        if max(lengths.values()) <= max_len:
            break
        if top_k <= 1:
            # Force balanced lengths <= max_len (guaranteed feasible for <= 2 symbols)
            lengths = {s: 1 for s in lengths}
            break
        top_k -= 1

    # Assign canonical MSB-first codes, then bit-reverse to LSB-first for `lo & 0xFF` lookup!
    sym_sorted = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    codes: dict[int, tuple[int, int]] = {}  # sym -> (lsb_code, nbits)
    code_msb = 0
    prev_len = 0
    for s in sym_sorted:
        L = lengths[s]
        code_msb <<= (L - prev_len)
        codes[s] = (_bit_reverse(code_msb, L), L)
        code_msb += 1
        prev_len = L

    # Sanity check Kraft equality (complete prefix code -> every 8-bit byte has a unique first symbol!)
    kraft_num = sum(1 << (max_len - L) for _, L in codes.values())
    assert kraft_num == (1 << max_len), f"Incomplete prefix tree: Kraft={kraft_num}/{1 << max_len}"

    # Build 256-entry 2-symbol-at-once LUT2
    lut2 = np.zeros(256, dtype=np.uint32)
    sym_list = [(s, c, L, (1 << L) - 1) for s, (c, L) in codes.items()]

    for b in range(256):
        s0 = -1
        L0 = 0
        for s, c, L, mask in sym_list:
            if (b & mask) == c:
                s0, L0 = s, L
                break
        assert s0 >= 0, f"No prefix match for byte {b}"
        if s0 == ESC_SYM:
            # cnt = 0, plen = L0 in bits [19:16]
            lut2[b] = np.uint32((L0 << 16) | (0 << 20) | (L0 << 24))
            continue

        # Try decoding a second non-escape symbol from remaining (8 - L0) bits
        rem_bits = 8 - L0
        b_rem = b >> L0
        s1 = -1
        L1 = 0
        for s, c, L, mask in sym_list:
            if s != ESC_SYM and L <= rem_bits and (b_rem & mask) == c:
                s1, L1 = s, L
                break
        if s1 >= 0:
            tlen = L0 + L1
            lut2[b] = np.uint32(s0 | (s1 << 8) | (tlen << 16) | (2 << 20) | (L0 << 24))
        else:
            lut2[b] = np.uint32(s0 | (L0 << 16) | (1 << 20) | (L0 << 24))

    return codes, lut2


@dataclass
class PackedBF10NumPy:
    sm_slabs: np.ndarray      # [wtot, slabs_per_warp, 1024] uint8
    exp_buf: np.ndarray       # [total_exp_bytes] uint8 (each warp 16B aligned)
    exp_desc: np.ndarray      # [wtot, 2] uint32 (.x = byte_off, .y = num_u4)
    stream_off: np.ndarray    # [wtot, 64] uint16 (bit offsets: 0..31 stream A, 32..63 stream B)
    lut2: np.ndarray          # [256] uint32
    codes: dict[int, tuple[int, int]]
    ks: int
    n_active: int
    cfg_id: int
    split: int
    n_tiles: int
    bn: int
    N: int
    K: int

    @property
    def total_bytes(self) -> int:
        return (
            self.sm_slabs.nbytes
            + self.exp_buf.nbytes
            + self.exp_desc.nbytes
            + self.stream_off.nbytes
            + self.lut2.nbytes
        )

    @property
    def bits_per_weight(self) -> float:
        return self.total_bytes * 8.0 / float(self.N * self.K)


def pack_bf10_smem2sym_np(
    w_u16: np.ndarray,
    cfg_id: int = 1,
    split: int = 6,
    n_active: int | None = None,
) -> PackedBF10NumPy:
    """Pack a [N, K] uint16 BF16 weight matrix into exact BF10_SMEM2SYM MMA layout.

    Matches `bf10_smem2sym_exact.cu` byte-for-byte and bit-for-bit!
    """
    assert w_u16.ndim == 2 and w_u16.dtype == np.uint16
    N, K = int(w_u16.shape[0]), int(w_u16.shape[1])
    assert K % 128 == 0, f"K={K} must be a multiple of 128"
    if n_active is None:
        n_active = N

    warps, sub_n, bn = _CFGS[cfg_id]
    assert sub_n == 1, "bf10_smem2sym_exact currently specializes SUB_N=1 (cfg_id in {1, 4})"
    ks = ((K + split - 1) // split + 127) // 128 * 128
    n_chunks_max = ks // 128
    slabs_per_warp = n_chunks_max * sub_n
    n_tiles = (N + bn - 1) // bn
    n_pad = n_tiles * bn
    k_pad = split * ks

    # Pad N and K with zeros if needed (exact match to skinny_bf11.py)
    if n_pad != N or k_pad != K:
        wp = np.zeros((n_pad, k_pad), dtype=np.uint16)
        wp[:N, :K] = w_u16
    else:
        wp = w_u16

    # Build global per-matrix 2-symbol LUT2 from real active weights
    exps_all = ((w_u16[:n_active] >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint8)
    codes, lut2 = build_bounded_huffman_lut2(exps_all)
    esc_code, esc_len = codes[ESC_SYM]

    # Precompute 256-entry fast encode lookup tables: enc_bits[e], enc_len[e]
    enc_bits = np.zeros(256, dtype=np.uint32)
    enc_len = np.zeros(256, dtype=np.uint32)
    for e in range(256):
        if e in codes:
            c, L = codes[e]
            enc_bits[e] = c
            enc_len[e] = L
        else:
            # Escape prefix (LSB-first) followed immediately by raw 8-bit exponent e!
            enc_bits[e] = esc_code | (e << esc_len)
            enc_len[e] = esc_len + 8

    # Reshape wp into MMA tile coordinates:
    #   [n_tiles, warps, 8(g), split, n_chunks_max, 8(s), 2(hi/lo 8-cols), 4(t), 2(pair)]
    # Check lane indexing in exec_slab_smem2sym:
    #   lane = g*4 + t -> j=0: col 2t, j=1: col 2t+1, j=2: col 8+2t, j=3: col 8+2t+1!
    wt = wp.reshape(n_tiles, warps, 8, split, n_chunks_max, 8, 2, 4, 2)
    # Transpose to: [split, n_tiles, warps, n_chunks_max, 8(g), 4(t), 8(s), 2(hi), 2(pair)]
    # Notice warp_uid = (ps * n_tiles + pn) * warps + warp!
    wt = wt.transpose(3, 0, 1, 4, 2, 7, 5, 6, 8)
    wtot = split * n_tiles * warps
    # Flatten (split, n_tiles, warps) -> wtot, (8g, 4t) -> 32 lanes, (2hi, 2pair) -> 4 weights j=0,1,2,3
    wt = wt.reshape(wtot, slabs_per_warp, 32, 8, 4)

    # 1. Pack SM_SLABS [wtot, slabs_per_warp, 2(sm0/sm1), 32(lane), 4(s_sub), 4(j)] -> [wtot, slabs_per_warp, 1024] uint8
    sm_byte = (((wt >> np.uint16(8)) & np.uint16(0x80)) | (wt & np.uint16(0x7F))).astype(np.uint8)
    # sm0 holds s=0..3 at p + lane (32 x 16B); sm1 holds s=4..7 at p + 32 + lane (32 x 16B)
    sm_split = sm_byte.reshape(wtot, slabs_per_warp, 32, 2, 4, 4).transpose(0, 1, 3, 2, 4, 5)
    sm_slabs = np.ascontiguousarray(sm_split.reshape(wtot, slabs_per_warp, SM_SLAB_BYTES))

    # 2. Pack unpadded per-warp exponent streams into 16B-aligned EXP_BUF
    exp_w = ((wt >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint8)  # [wtot, slabs_per_warp, 32, 8, 4]

    # Actual active slabs per split ps (last split may have fewer than n_chunks_max chunks)
    split_chunks = [max(0, (min(K, (ps + 1) * ks) - ps * ks) // 128) for ps in range(split)]

    exp_desc = np.zeros((wtot, 2), dtype=np.uint32)
    stream_off = np.zeros((wtot, 64), dtype=np.uint16)
    exp_chunks: list[bytes] = []
    cur_byte_off = 0

    for wuid in range(wtot):
        ps = wuid // (n_tiles * warps)
        rem = wuid - ps * (n_tiles * warps)
        pn = rem // warps
        warp = rem - pn * warps
        n_warp_base = pn * bn + warp * 8
        n_sl = split_chunks[ps] if n_warp_base < n_active else 0

        if n_sl == 0:
            exp_desc[wuid, 0] = cur_byte_off
            exp_desc[wuid, 1] = 0
            continue

        # Bit-append all 64 streams (lane 0..31 stream A: j=(0,2); lane 0..31 stream B: j=(1,3))
        acc_int = 0
        bit_pos = 0
        we = exp_w[wuid, :n_sl]  # [n_sl, 32, 8, 4]
        # Vectorize code/length lookup for this warp:
        bits_arr = enc_bits[we]  # [n_sl, 32, 8, 4]
        len_arr = enc_len[we]    # [n_sl, 32, 8, 4]

        # Stream A (0..31): for each lane l, emit (j=0, j=2) across sl=0..n_sl-1, s=0..7
        for l in range(32):
            stream_off[wuid, l] = bit_pos
            b_lane = bits_arr[:, l, :, :][:, :, (0, 2)].ravel().tolist()
            L_lane = len_arr[:, l, :, :][:, :, (0, 2)].ravel().tolist()
            for cb, cl in zip(b_lane, L_lane):
                acc_int |= cb << bit_pos
                bit_pos += cl

        # Stream B (32..63): for each lane l, emit (j=1, j=3) across sl=0..n_sl-1, s=0..7
        for l in range(32):
            stream_off[wuid, 32 + l] = bit_pos
            b_lane = bits_arr[:, l, :, :][:, :, (1, 3)].ravel().tolist()
            L_lane = len_arr[:, l, :, :][:, :, (1, 3)].ravel().tolist()
            for cb, cl in zip(b_lane, L_lane):
                acc_int |= cb << bit_pos
                bit_pos += cl

        nbytes = (bit_pos + 7) // 8
        nbytes_16 = (nbytes + 15) & ~15
        if nbytes_16 > MAX_WARP_EXP_BYTES:
            raise ValueError(
                f"Warp {wuid} packed exp bytes {nbytes_16} > MAX_WARP_EXP_BYTES={MAX_WARP_EXP_BYTES}; "
                f"fall back to BF11_FLIP for this layer."
            )
        exp_chunks.append(acc_int.to_bytes(nbytes_16, "little"))
        exp_desc[wuid, 0] = cur_byte_off
        exp_desc[wuid, 1] = nbytes_16 // 16
        cur_byte_off += nbytes_16

    exp_buf = np.frombuffer(b"".join(exp_chunks), dtype=np.uint8).copy()
    return PackedBF10NumPy(
        sm_slabs=sm_slabs,
        exp_buf=exp_buf,
        exp_desc=exp_desc,
        stream_off=stream_off,
        lut2=lut2,
        codes=codes,
        ks=ks,
        n_active=n_active,
        cfg_id=cfg_id,
        split=split,
        n_tiles=n_tiles,
        bn=bn,
        N=N,
        K=K,
    )


def unpack_bf10_smem2sym_ref_np(packed: PackedBF10NumPy) -> np.ndarray:
    """Bit-exact Python/NumPy mirror of `bf10_smem2sym_exact.cu::SmemBitStream` + BF16 rebuild.

    Used to verify 0-ULP round-trip lossless reconstruction of every weight before GPU launch.
    """
    warps, _, bn = _CFGS[packed.cfg_id]
    n_tiles = packed.n_tiles
    split = packed.split
    ks = packed.ks
    n_chunks_max = ks // 128
    wtot = split * n_tiles * warps
    split_chunks = [max(0, (min(packed.K, (ps + 1) * ks) - ps * ks) // 128) for ps in range(split)]
    lut2 = packed.lut2

    out_u16 = np.zeros((n_tiles * bn, split * ks), dtype=np.uint16)

    for wuid in range(wtot):
        ps = wuid // (n_tiles * warps)
        rem = wuid - ps * (n_tiles * warps)
        pn = rem // warps
        warp = rem - pn * warps
        n_warp_base = pn * bn + warp * 8
        n_sl = split_chunks[ps] if n_warp_base < packed.n_active else 0
        if n_sl == 0:
            continue

        byte_off = int(packed.exp_desc[wuid, 0])
        num_u4 = int(packed.exp_desc[wuid, 1])
        # Pad +2 uint4 (+8 uint32_t) identical to s_exp[warp][MAX_WARP_EXP_U4 + 2]
        raw = np.zeros((num_u4 + 2) * 16, dtype=np.uint8)
        raw[: num_u4 * 16] = packed.exp_buf[byte_off : byte_off + num_u4 * 16]
        words = raw.view(np.uint32)

        def make_reader(bit_off: int):
            w = bit_off >> 5
            b = bit_off & 31
            w0 = int(words[w])
            w1 = int(words[w + 1])
            if b == 0:
                lo, hi, bl = w0, w1, 64
            else:
                lo = ((w0 >> b) | ((w1 << (32 - b)) & 0xFFFFFFFF)) & 0xFFFFFFFF
                hi = (w1 >> b) & 0xFFFFFFFF
                bl = 64 - b
            return [w + 2, lo, hi, bl]

        def refill(st):
            if st[3] <= 32:
                nw = int(words[st[0]])
                st[0] += 1
                buf = (st[2] << 32) | st[1]
                buf |= nw << st[3]
                st[1] = buf & 0xFFFFFFFF
                st[2] = (buf >> 32) & 0xFFFFFFFF
                st[3] += 32

        def consume(st, nb):
            if nb > 0:
                buf = ((st[2] << 32) | st[1]) >> nb
                st[1] = buf & 0xFFFFFFFF
                st[2] = (buf >> 32) & 0xFFFFFFFF
                st[3] -= nb

        def decode1(st):
            refill(st)
            ent = int(lut2[st[1] & 0xFF])
            cnt = (ent >> 20) & 3
            if cnt != 0:
                consume(st, (ent >> 24) & 0xF)
                return ent & 0xFF
            consume(st, (ent >> 16) & 0xF)
            raw_e = st[1] & 0xFF
            consume(st, 8)
            return raw_e

        def decode2(st):
            refill(st)
            ent = int(lut2[st[1] & 0xFF])
            cnt = (ent >> 20) & 3
            if cnt == 2:
                consume(st, (ent >> 16) & 0xF)
                return ent & 0xFF, (ent >> 8) & 0xFF
            if cnt == 1:
                consume(st, (ent >> 24) & 0xF)
                return ent & 0xFF, decode1(st)
            consume(st, (ent >> 16) & 0xF)
            e0 = st[1] & 0xFF
            consume(st, 8)
            return e0, decode1(st)

        stA = [make_reader(int(packed.stream_off[wuid, l])) for l in range(32)]
        stB = [make_reader(int(packed.stream_off[wuid, 32 + l])) for l in range(32)]

        for sl in range(n_sl):
            slab_u8 = packed.sm_slabs[wuid, sl].reshape(2, 32, 4, 4)
            k_base = ps * ks + sl * 128
            for l in range(32):
                g = l >> 2
                t = l & 3
                row = n_warp_base + g
                for s in range(8):
                    e0, e2 = decode2(stA[l])
                    e1, e3 = decode2(stB[l])
                    sm4 = slab_u8[s >> 2, l, s & 3]
                    es = (e0, e1, e2, e3)
                    cols = (
                        k_base + s * 16 + 2 * t,
                        k_base + s * 16 + 2 * t + 1,
                        k_base + s * 16 + 8 + 2 * t,
                        k_base + s * 16 + 8 + 2 * t + 1,
                    )
                    for j in range(4):
                        smb = int(sm4[j])
                        sign = (smb & 0x80) << 8
                        mant = smb & 0x7F
                        out_u16[row, cols[j]] = sign | (es[j] << 7) | mant

    return out_u16[: packed.N, : packed.K]


# ================= PyTorch / GB10 Runtime Wrapper =================


@dataclass
class PackedBF10Torch:
    sm_slabs: object
    exp_buf: object
    exp_desc: object
    stream_off: object
    lut2: object
    ks: int
    n_active: int
    cfg_id: int
    split: int
    n_tiles: int
    bn: int


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is None and os.path.exists(_LIB_PATH):
            lib = ctypes.CDLL(_LIB_PATH)
            lib.bf10_smem2sym_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.bf10_smem2sym_fwd.restype = ctypes.c_int
            _LIB = lib
    return _LIB


def _ensure_scratch(device, min_part_elems: int, min_sem_tiles: int) -> None:
    dev_idx = device.index if device.index is not None else 0
    need_p = max(_MIN_PART_ELEMS, int(min_part_elems))
    need_s = max(_MIN_SEM_TILES, int(min_sem_tiles))
    parts = _PART_BUFS.get(dev_idx)
    if parts is None or parts[0].numel() < need_p:
        _PART_BUFS[dev_idx] = [
            torch.empty(need_p, dtype=torch.float32, device=device),
            torch.empty(need_p, dtype=torch.float32, device=device),
        ]
    sems = _SEM_BUFS.get(dev_idx)
    if sems is None or sems[0].numel() < need_s:
        _SEM_BUFS[dev_idx] = [
            torch.zeros(need_s, dtype=torch.int32, device=device),
            torch.zeros(need_s, dtype=torch.int32, device=device),
        ]
    _SCRATCH_IDX.setdefault(dev_idx, 0)


def _next_scratch(device, min_part_elems: int, min_sem_tiles: int):
    dev_idx = device.index if device.index is not None else 0
    _ensure_scratch(device, min_part_elems, min_sem_tiles)
    idx = _SCRATCH_IDX[dev_idx]
    _SCRATCH_IDX[dev_idx] = idx ^ 1
    return _PART_BUFS[dev_idx][idx], _SEM_BUFS[dev_idx][idx]


def prepack(w, cfg_id: int | None = None, split: int | None = None, n_active: int | None = None):
    """Prepack a CUDA bf16 [N, K] tensor into GPU BF10_SMEM2SYM buffers (offline before graph capture)."""
    assert torch is not None
    N, K = int(w.shape[0]), int(w.shape[1])
    if cfg_id is None or split is None:
        cfg_id, split = _SHAPES.get((N, K), (1, 6 if K >= 4096 else 1))
    key = (int(w.data_ptr()), N, K)
    if key in _CACHE:
        return _CACHE[key]
    w_u16 = w.detach().contiguous().view(torch.uint16).cpu().numpy()
    try:
        pk = pack_bf10_smem2sym_np(w_u16, cfg_id=cfg_id, split=split, n_active=n_active)
    except ValueError:
        return None
    dev = w.device
    entry = PackedBF10Torch(
        sm_slabs=torch.from_numpy(pk.sm_slabs).to(dev, non_blocking=False).contiguous(),
        exp_buf=torch.from_numpy(pk.exp_buf).to(dev, non_blocking=False).contiguous(),
        exp_desc=torch.from_numpy(pk.exp_desc).to(dev, non_blocking=False).contiguous(),
        stream_off=torch.from_numpy(pk.stream_off).to(dev, non_blocking=False).contiguous(),
        lut2=torch.from_numpy(pk.lut2).to(dev, non_blocking=False).contiguous(),
        ks=pk.ks,
        n_active=pk.n_active,
        cfg_id=pk.cfg_id,
        split=pk.split,
        n_tiles=pk.n_tiles,
        bn=pk.bn,
    )
    _ensure_scratch(
        dev,
        max(_MIN_PART_ELEMS, 2 * pk.n_tiles * pk.split * 16 * pk.bn),
        max(_MIN_SEM_TILES, 2 * pk.n_tiles),
    )
    _CACHE[key] = entry
    return entry


def l2pf_tensor(w):
    """Multi-segment L2PF descriptor: warm lut2 + exp_desc + stream_off + wave-0 exp_buf + sm_slabs!"""
    if torch is None or w is None or w.dim() != 2:
        return w
    hit = _CACHE.get((int(w.data_ptr()), int(w.shape[0]), int(w.shape[1])))
    if hit is None:
        return w
    return (hit.lut2, hit.exp_desc, hit.stream_off, (hit.exp_buf, 524288), hit.sm_slabs)


def bf10_linear(x, w, out=None, pdl: bool = False, trig: bool = False, n_live_buf=None):
    """Execute 0-ULP BF10_SMEM2SYM skinny GEMM (`M <= 16`). Returns None if unavailable."""
    lib = _load_lib()
    if lib is None or torch is None:
        return None
    M, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    if M <= 0 or M > 16:
        return None
    hit = _CACHE.get((int(w.data_ptr()), N, K))
    if hit is None:
        return None
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if hit.split > 1:
        p_buf, sem_buf = _next_scratch(x.device, hit.n_tiles * hit.split * M * hit.bn, hit.n_tiles)
        p_ptr, sem_ptr = int(p_buf.data_ptr()), int(sem_buf.data_ptr())
    else:
        p_ptr, sem_ptr = 0, 0
    nl_ptr = int(n_live_buf.data_ptr()) if n_live_buf is not None else 0
    stream = int(torch.cuda.current_stream(x.device).cuda_stream)
    rc = lib.bf10_smem2sym_fwd(
        int(x.data_ptr()),
        int(hit.sm_slabs.data_ptr()),
        int(hit.exp_buf.data_ptr()),
        int(hit.exp_desc.data_ptr()),
        int(hit.stream_off.data_ptr()),
        int(hit.lut2.data_ptr()),
        p_ptr,
        sem_ptr,
        int(out.data_ptr()),
        nl_ptr,
        M, N, hit.n_active, K,
        int(x.stride(0)), hit.ks, int(out.stride(0)),
        hit.cfg_id, hit.split, 1 if pdl else 0, 1 if trig else 0,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"bf10_smem2sym_fwd failed with cudaError={rc}")
    return out
