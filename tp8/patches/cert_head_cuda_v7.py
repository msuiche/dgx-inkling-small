# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP Certified LM Head v7:
2-Stage Progressive INT6_g512_ms8(2b) + 1bLSB + Full-Row L2-Prefetched Refine + Direct Logits Gather.

Key wins vs cert_head_fast_v6 (INT7_g64: 105.14 MB, 439 us screen + 20.3 us refine):
  1. Stage-1 screen (`INT6_g512_ms8(2b)`): 91.11 MB (-14.03 MB = -58.5 us vs INT7_g64!),
     100% uint4 aligned, e1 = 1.97%.
  2. Stage-2 1bLSB sparse filter: reads only 738 B/row (576 B LSB1 + 144 B MS + 18 B BIG_S)
     for Stage-1 survivors K1 (~400..800 rows = ~0.3-0.6 MB = 2-4 us), upgrading
     e1 = 1.97% -> e2 = 1.05% (TIGHTER than INT7_g64 1.20% & INT7_g32 1.09%!), pruning
     K1 -> K2 (~5..15 rows) with ZERO extra RDMA and ~identical total VRAM (105.58 MB)!
  3. Stage-3 exact 0-ULP BF16 refine: full-row parallel `prefetch.global.L2` fixes v6's
     20.3 us serial cold-DRAM latency bug -> ~4.5 us (-15.8 us/step, literal 0-ULP!).
  4. Direct `LOGITS_IDX` gather inside `cert_xnorm_pad_v7` skips PyTorch's 4.32 us
     `vectorized_gather_kernel` before CertHead!
  5. Includes `check_nw_heterogeneous(nw)` auto-probe to detect if live Inkling-2-Small
     `lm_head.weight` has heterogeneous row norms (`GM(|nw|)/RMS(nw) < 0.85`), which
     would unlock an extra -30 to -45 us via mixed INT6/INT5 water-filling.
"""
from __future__ import annotations

import ctypes
import math
import os
import threading
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # Local Mac verification runs without PyTorch
    torch = None  # type: ignore[assignment]

ENABLED = os.environ.get("INKLING_CERT_HEAD_V7", "0").strip().lower() in ("1", "true", "on", "yes")
_LIB_PATH = os.environ.get("INKLING_CERT_V7_LIB", "/rdma/libcert_head_fast_v7.so")
_CFG_ID = int(os.environ.get("INKLING_CERT_V7_CFG", "0"))  # 0: BN=64 W=4 SUB=2 (default); 1: BN=32 W=4; 2: BN=16 W=2
_REF_GRID = int(os.environ.get("INKLING_CERT_V7_REF_GRID", "96"))
_USE_PDL = os.environ.get("INKLING_CERT_V7_PDL", "1").strip().lower() not in ("0", "", "off", "false", "no")

MS_MULTS = np.array([0.375, 0.5625, 0.75, 1.0], dtype=np.float32)

# Soundness constants matching cert_head.py & cert_head_cuda.py (exact IEEE-754 bounds)
_EPS_F32 = 2.0 ** -24
_INFL_X = 1.0 + 1e-4
_X_ABS = 1e-6
_INFL_Q = 1.0 + 1e-5
_INFL_B = 1.0 + 1e-4
_SLOP_REL = 1e-5
_SLOP_ABS = 1e-4
_SLACK = 1e-4

_CFGS = {
    0: (4, 2, 64),
    1: (4, 1, 32),
    2: (2, 1, 16),
}

_LIB = None
_LIB_LOCK = threading.Lock()


def bf16_rn_np(x_f32: np.ndarray) -> np.ndarray:
    """Exact IEEE-754 round-to-nearest-even float32 -> BF16 -> float32 in pure NumPy."""
    x = np.asarray(x_f32, dtype=np.float32)
    u = x.view(np.uint32)
    # Handle NaN cleanly
    is_nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    u_rn = (u + rounding_bias) & np.uint32(0xFFFF0000)
    u_out = np.where(is_nan, (u & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), u_rn)
    return u_out.view(np.float32)


def cacc2_for_k(K: int) -> float:
    """Exact 2 * c_acc MMA accumulation error constant for K-dot products (matches cert_head.py)."""
    # K/16 MMA steps of 16-term fp32 sums; Higham gamma_n bound doubled for |W| + |W1|
    c_acc = (K * _EPS_F32) / max(1.0 - K * _EPS_F32, 0.5) * 2.0
    return float(2.0 * c_acc)


def check_nw_heterogeneous(w_f32: np.ndarray) -> dict[str, float]:
    """Probe lm_head.weight row-norm spread: if GM(||w_j||)/RMS(||w_j||) < 0.85,
    mixed INT6/INT5 row-budgeted water-filling saves an additional 8-12% Stage-1 bytes."""
    nw = np.linalg.norm(w_f32.astype(np.float64), axis=1)
    nz = nw[nw > 0]
    gm = float(np.exp(np.mean(np.log(nz)))) if len(nz) else 0.0
    rms = float(np.sqrt(np.mean(nz * nz))) if len(nz) else 1.0
    ratio = gm / rms if rms > 0 else 1.0
    return {
        "nw_min": float(nw.min()),
        "nw_p50": float(np.median(nw)),
        "nw_p95": float(np.percentile(nw, 95)),
        "nw_max": float(nw.max()),
        "gm_over_rms": ratio,
        "recommend_hetero_int6_int5": float(ratio < 0.85),
    }


@dataclass
class PackedCertV7NumPy:
    slabs_u4: np.ndarray    # [wtot, n_qchunks * sub_n, 6, 32, 4] uint32 (view as uint4)
    ms_u4: np.ndarray       # [wtot, n_qchunks * sub_n, 8, 4] uint32 (view as uint4)
    big_s: np.ndarray       # [wtot, n_qchunks * sub_n, 8] float16
    lsb1_row: np.ndarray    # [V, n_qchunks * 16] uint32 (1b/elem row-contiguous)
    ms_row: np.ndarray      # [V, n_qchunks, 4] uint32 (view as uint4)
    big_s_row: np.ndarray   # [V, n_qchunks] float16
    rt1: np.ndarray         # [V] float32 (exact Stage-1 residual L2 norm, rounded UP)
    nv1: np.ndarray         # [V] float32 (max(||W||, ||W1_bf16||), rounded UP)
    rt2: np.ndarray         # [V] float32 (exact Stage-2 residual L2 norm, rounded UP)
    nv2: np.ndarray         # [V] float32 (NV1 + ||dW_bf16||, rounded UP)
    w1_bf16: np.ndarray     # [V, K] float32 (exact reconstructed Stage-1 BF16 weights, for verification)
    dw_bf16: np.ndarray     # [V, K] float32 (exact reconstructed Stage-2 delta BF16 weights, for verification)
    V: int
    K: int
    cfg_id: int

    @property
    def stage1_bytes(self) -> int:
        return (
            self.slabs_u4.nbytes
            + self.ms_u4.nbytes
            + self.big_s.nbytes
            + self.rt1.nbytes
            + self.nv1.nbytes
        )

    @property
    def stage2_bytes(self) -> int:
        return (
            self.lsb1_row.nbytes
            + self.ms_row.nbytes
            + self.big_s_row.nbytes
            + self.rt2.nbytes
            + self.nv2.nbytes
        )


def pack_cert_v7_np(w_bf16_as_f32: np.ndarray, cfg_id: int = 0) -> PackedCertV7NumPy:
    """Exact packer for `cert_head_fast_v7.cu` (Stage-1 INT6_g512_ms8(2b) + Stage-2 1bLSB).

    Input `w_bf16_as_f32` should hold BF16-representable float32 weights `[V, K]` (`K % 512 == 0`).
    """
    w = bf16_rn_np(np.asarray(w_bf16_as_f32, dtype=np.float32))
    V, K = int(w.shape[0]), int(w.shape[1])
    assert K % 512 == 0, f"K={K} must be a multiple of 512"
    warps, sub_n, bn = _CFGS[cfg_id]
    n_qchunks = K // 512
    n_tiles = (V + bn - 1) // bn
    v_pad = n_tiles * bn
    wtot = n_tiles * warps

    # 1. Quantize all [V, n_qchunks, 64, 8] blocks
    w_blk = w.reshape(V, n_qchunks, 64, 8)
    m_abs = np.max(np.abs(w_blk), axis=3)              # [V, n_qchunks, 64]
    row_max = np.max(m_abs, axis=2)                    # [V, n_qchunks]

    # Choose fp16 big_s rounded UP so 31 * fp16(big_s) >= row_max (zero clipping!)
    bs_raw = (row_max / np.float32(31.0)).astype(np.float16)
    bs_f32 = bs_raw.astype(np.float32)
    need_up = (row_max > 0.0) & (bs_f32 * np.float32(31.0) < row_max)
    while np.any(need_up):
        bs_raw = np.where(need_up, np.nextafter(bs_raw, np.float16(np.inf)), bs_raw)
        bs_f32 = bs_raw.astype(np.float32)
        need_up = (row_max > 0.0) & (bs_f32 * np.float32(31.0) < row_max)

    big_s_row = np.ascontiguousarray(bs_raw)           # [V, n_qchunks] float16
    bs_exp = bs_f32[:, :, None]                        # [V, n_qchunks, 1]

    # Select smallest 2-bit micro-scale code in {0, 1, 2, 3} covering m_abs <= 31 * sc4[code2]
    # Thresholds: 0.375, 0.5625, 0.75, 1.0
    t0 = bs_exp * np.float32(0.375 * 31.0)
    t1 = bs_exp * np.float32(0.5625 * 31.0)
    t2 = bs_exp * np.float32(0.75 * 31.0)
    code2 = np.where(
        m_abs <= t0,
        np.uint32(0),
        np.where(m_abs <= t1, np.uint32(1), np.where(m_abs <= t2, np.uint32(2), np.uint32(3))),
    ).astype(np.uint32)                                # [V, n_qchunks, 64]

    sc_mb = bs_exp * MS_MULTS[code2]                   # [V, n_qchunks, 64] float32
    inv_sc = np.zeros_like(sc_mb, dtype=np.float32)
    np.divide(np.float32(1.0), sc_mb, out=inv_sc, where=sc_mb > 0.0)
    q1 = np.clip(np.rint(w_blk * inv_sc[:, :, :, None]), -31.0, 31.0).astype(np.int32)
    u6 = (q1 + 32).astype(np.uint32)                   # [V, n_qchunks, 64, 8] in [1..63]

    # Exact Stage-1 BF16 weight seen by `mma_step6_ms8`!
    w1_blk = bf16_rn_np(q1.astype(np.float32) * sc_mb[:, :, :, None])
    w1_bf16 = w1_blk.reshape(V, K)

    # Exact Stage-2 1bLSB quantization matching `cert_stage2_lsb1_v7` (bs * 0.25f * mult[code2])!
    r1_blk = w_blk.astype(np.float64) - w1_blk.astype(np.float64)
    lsb_bit = (r1_blk >= 0.0).astype(np.uint32)        # [V, n_qchunks, 64, 8] in {0, 1}
    sc_s2 = (bs_exp * np.float32(0.25)) * MS_MULTS[code2]
    dw_blk = bf16_rn_np(np.where(lsb_bit == 1, sc_s2[:, :, :, None], -sc_s2[:, :, :, None]))
    dw_bf16 = dw_blk.reshape(V, K)

    # Exact conservative fp64 certificate norms (rounded UP into fp32!)
    r2_f64 = w.astype(np.float64) - (w1_bf16.astype(np.float64) + dw_bf16.astype(np.float64))
    r1_f64 = w.astype(np.float64) - w1_bf16.astype(np.float64)
    nw_f64 = np.sqrt(np.sum(w.astype(np.float64) ** 2, axis=1))
    nw1_f64 = np.sqrt(np.sum(w1_bf16.astype(np.float64) ** 2, axis=1))
    ndw_f64 = np.sqrt(np.sum(dw_bf16.astype(np.float64) ** 2, axis=1))

    up = 1.0 + 1e-6
    rt1 = (np.sqrt(np.sum(r1_f64 ** 2, axis=1)) * up).astype(np.float32)
    nv1 = (np.maximum(nw_f64, nw1_f64) * up).astype(np.float32)
    rt2 = (np.sqrt(np.sum(r2_f64 ** 2, axis=1)) * up).astype(np.float32)
    nv2 = ((np.maximum(nw_f64, nw1_f64) + ndw_f64) * up).astype(np.float32)

    # 2. Pack row-contiguous Stage-2 arrays: MS_ROW [V, n_qchunks, 4] uint32 & LSB1_ROW [V, n_qchunks*16] uint32
    c2_4 = code2.reshape(V, n_qchunks, 4, 16)
    shifts_ms = (np.arange(16, dtype=np.uint32) * np.uint32(2))[None, None, None, :]
    ms_row = np.ascontiguousarray(np.bitwise_or.reduce(c2_4 << shifts_ms, axis=3).astype(np.uint32))

    lsb_u32 = lsb_bit.reshape(V, n_qchunks * 16, 32)
    shifts_b = np.arange(32, dtype=np.uint32)[None, None, :]
    lsb1_row = np.ascontiguousarray(np.bitwise_or.reduce(lsb_u32 << shifts_b, axis=2).astype(np.uint32))

    # 3. Pad V -> v_pad and pack Stage-1 warp-tiled quad-slabs:
    #    SLABS_U4 [wtot, n_qchunks * sub_n, 6, 32, 4] uint32
    #    MS_U4    [wtot, n_qchunks * sub_n, 8, 4] uint32
    #    BIG_S    [wtot, n_qchunks * sub_n, 8] float16
    if v_pad != V:
        u6_p = np.full((v_pad, n_qchunks, 512), 32, dtype=np.uint32)
        u6_p[:V] = u6.reshape(V, n_qchunks, 512)
        ms_p = np.zeros((v_pad, n_qchunks, 4), dtype=np.uint32)
        ms_p[:V] = ms_row
        bs_p = np.zeros((v_pad, n_qchunks), dtype=np.float16)
        bs_p[:V] = big_s_row
    else:
        u6_p = u6.reshape(V, n_qchunks, 512)
        ms_p = ms_row
        bs_p = big_s_row

    # Reshape rows to [n_tiles, warps, sub_n, 8(g)] -> warp_uid = pn * warps + warp, q = qc * sub_n + sub!
    ms_t = ms_p.reshape(n_tiles, warps, sub_n, 8, n_qchunks, 4).transpose(0, 1, 4, 2, 3, 5)
    ms_u4 = np.ascontiguousarray(ms_t.reshape(wtot, n_qchunks * sub_n, 8, 4))

    bs_t = bs_p.reshape(n_tiles, warps, sub_n, 8, n_qchunks).transpose(0, 1, 4, 2, 3)
    big_s = np.ascontiguousarray(bs_t.reshape(wtot, n_qchunks * sub_n, 8))

    # Reshape 512 cols into: [8(half-chunks of 64), 4(MMA steps of 16), 2(hi/lo 8), 4(t), 2(pair)]
    u6_t = u6_p.reshape(n_tiles, warps, sub_n, 8, n_qchunks, 8, 4, 2, 4, 2)
    # Transpose to: [n_tiles, warps, n_qchunks, sub_n, 8(g), 4(t), 8(hc), 4(s), 2(hi), 2(pair)]
    u6_t = u6_t.transpose(0, 1, 4, 2, 3, 8, 5, 6, 7, 9)
    # Collapse -> [wtot, total_q, 32(lane), 8(hc), 4(s), 4(j=0..3)]
    u6_t = u6_t.reshape(wtot, n_qchunks * sub_n, 32, 8, 4, 4)

    # Pack 4 weights (j=0..3) into 24-bit word u24: [wtot, total_q, 32, 8, 4]
    u24 = (
        u6_t[..., 0]
        | (u6_t[..., 1] << np.uint32(6))
        | (u6_t[..., 2] << np.uint32(12))
        | (u6_t[..., 3] << np.uint32(18))
    )
    # Pack 4 x 24-bit words (96 bits) -> 3 x uint32_t per half-chunk!
    w0 = u24[..., 0] | ((u24[..., 1] & np.uint32(0xFF)) << np.uint32(24))
    w1 = (u24[..., 1] >> np.uint32(8)) | ((u24[..., 2] & np.uint32(0xFFFF)) << np.uint32(16))
    w2 = (u24[..., 2] >> np.uint32(16)) | (u24[..., 3] << np.uint32(8))
    # Stack -> [wtot, total_q, 32(lane), 8(hc), 3] -> [wtot, total_q, 32(lane), 6(v0..v5), 4(xyzw)]
    words24 = np.stack([w0, w1, w2], axis=-1).reshape(wtot, n_qchunks * sub_n, 32, 6, 4)
    # Transpose (32, 6) -> (6, 32) so ld_cs_u4(qp + v*32 + lane) is 100% coalesced!
    slabs_u4 = np.ascontiguousarray(words24.transpose(0, 1, 3, 2, 4))

    return PackedCertV7NumPy(
        slabs_u4=slabs_u4,
        ms_u4=ms_u4,
        big_s=big_s,
        lsb1_row=lsb1_row,
        ms_row=ms_row,
        big_s_row=big_s_row,
        rt1=rt1,
        nv1=nv1,
        rt2=rt2,
        nv2=nv2,
        w1_bf16=w1_bf16,
        dw_bf16=dw_bf16,
        V=V,
        K=K,
        cfg_id=cfg_id,
    )


def unpack_cert_v7_ref_np(pk: PackedCertV7NumPy) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact NumPy mirror of `cert_screen_int6_ms8_v7` AND `cert_stage2_lsb1_v7` dequantization.

    Returns (w1_unpacked [V, K], dw_unpacked [V, K]) matching `pk.w1_bf16` and `pk.dw_bf16` to 0 diffs!
    """
    warps, sub_n, bn = _CFGS[pk.cfg_id]
    n_qchunks = pk.K // 512
    n_tiles = (pk.V + bn - 1) // bn
    v_pad = n_tiles * bn
    wtot = n_tiles * warps

    # 1. Unpack Stage-1 SLABS_U4 + MS_U4 + BIG_S exactly like `cert_screen_int6_ms8_v7`
    words = pk.slabs_u4.transpose(0, 1, 3, 2, 4).reshape(wtot, n_qchunks * sub_n, 32, 8, 3)
    w0, w1, w2 = words[..., 0], words[..., 1], words[..., 2]
    u24_0 = w0 & np.uint32(0xFFFFFF)
    u24_1 = ((w0 >> np.uint32(24)) | (w1 << np.uint32(8))) & np.uint32(0xFFFFFF)
    u24_2 = ((w1 >> np.uint32(16)) | (w2 << np.uint32(16))) & np.uint32(0xFFFFFF)
    u24_3 = (w2 >> np.uint32(8)) & np.uint32(0xFFFFFF)
    u24 = np.stack([u24_0, u24_1, u24_2, u24_3], axis=-1)  # [wtot, total_q, 32, 8(hc), 4(s)]

    q_vals = np.stack([
        (u24 & np.uint32(0x3F)).astype(np.int32) - 32,
        ((u24 >> np.uint32(6)) & np.uint32(0x3F)).astype(np.int32) - 32,
        ((u24 >> np.uint32(12)) & np.uint32(0x3F)).astype(np.int32) - 32,
        ((u24 >> np.uint32(18)) & np.uint32(0x3F)).astype(np.int32) - 32,
    ], axis=-1).astype(np.float32)  # [wtot, total_q, 32, 8, 4, 4]

    # Decode MS_U4 [wtot, total_q, 8(g), 4(ch)] -> [wtot, total_q, 8(g), 8(hc), 4(s), 2(lo/hi)]
    ms_c = np.zeros((wtot, n_qchunks * sub_n, 8, 64), dtype=np.uint32)
    for i in range(16):
        ms_c[..., 0 + i] = (pk.ms_u4[..., 0] >> np.uint32(2 * i)) & np.uint32(3)
        ms_c[..., 16 + i] = (pk.ms_u4[..., 1] >> np.uint32(2 * i)) & np.uint32(3)
        ms_c[..., 32 + i] = (pk.ms_u4[..., 2] >> np.uint32(2 * i)) & np.uint32(3)
        ms_c[..., 48 + i] = (pk.ms_u4[..., 3] >> np.uint32(2 * i)) & np.uint32(3)
    bs_f32 = pk.big_s.astype(np.float32)[:, :, :, None]
    sc_all = bs_f32 * MS_MULTS[ms_c]                       # [wtot, total_q, 8(g), 64]
    sc_all = sc_all.reshape(wtot, n_qchunks * sub_n, 8, 1, 8, 4, 2)  # broadcast across t=4
    q_gt = q_vals.reshape(wtot, n_qchunks * sub_n, 8, 4, 8, 4, 2, 2)
    w1_gt = bf16_rn_np(q_gt * sc_all[..., None])
    # Transpose back to [n_tiles, warps, sub_n, 8(g), n_qchunks, 8(hc), 4(s), 2(hi), 4(t), 2(pair)]
    w1_gt = w1_gt.reshape(n_tiles, warps, n_qchunks, sub_n, 8, 4, 8, 4, 2, 2)
    w1_unpacked = w1_gt.transpose(0, 1, 3, 4, 2, 6, 7, 8, 5, 9).reshape(v_pad, pk.K)[: pk.V]

    # 2. Unpack Stage-2 LSB1_ROW + MS_ROW + BIG_S_ROW exactly like `cert_stage2_lsb1_v7`
    ms_r = np.zeros((pk.V, n_qchunks, 64), dtype=np.uint32)
    for i in range(16):
        ms_r[..., 0 + i] = (pk.ms_row[..., 0] >> np.uint32(2 * i)) & np.uint32(3)
        ms_r[..., 16 + i] = (pk.ms_row[..., 1] >> np.uint32(2 * i)) & np.uint32(3)
        ms_r[..., 32 + i] = (pk.ms_row[..., 2] >> np.uint32(2 * i)) & np.uint32(3)
        ms_r[..., 48 + i] = (pk.ms_row[..., 3] >> np.uint32(2 * i)) & np.uint32(3)
    sc_s2 = (pk.big_s_row.astype(np.float32)[:, :, None] * np.float32(0.25)) * MS_MULTS[ms_r]

    bits_k = ((pk.lsb1_row[:, :, None] >> np.arange(32, dtype=np.uint32)[None, None, :]) & np.uint32(1))
    bits_k = bits_k.reshape(pk.V, n_qchunks, 64, 8)
    dw_unpacked = bf16_rn_np(np.where(bits_k == 1, sc_s2[:, :, :, None], -sc_s2[:, :, :, None])).reshape(pk.V, pk.K)
    return w1_unpacked, dw_unpacked


# ================= PyTorch / GB10 Runtime Wrapper =================


@dataclass
class CudaCertTableV7:
    slabs_u4: object
    ms_u4: object
    big_s: object
    lsb1_row: object
    ms_row: object
    big_s_row: object
    rt1: object
    nv1: object
    rt2: object
    nv2: object
    V: int
    K: int
    cfg_id: int
    cacc2: float

    def l2pf_segments(self):
        """Multi-segment L2PF descriptor for Layer-47 MoE AR overlap (fits inside 10 MB head budget)."""
        return (self.nv1, self.rt1, self.big_s, (self.ms_u4, 655360), self.slabs_u4)


@dataclass
class CudaCertStateV7:
    x_pad: object
    xn: object
    bad: object
    s1_out: object
    hi: object
    thr_u: object
    thr_ag: object
    t_glob: object
    idx1: object
    cnt1: object
    idx2: object
    cnt2: object
    best: object
    arrive: object
    out_pair: object
    nlive_full: object


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is None and os.path.exists(_LIB_PATH):
            lib = ctypes.CDLL(_LIB_PATH)
            lib.cert_head_v7_pre_screen.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.cert_head_v7_pre_screen.restype = ctypes.c_int

            lib.cert_head_v7_post_refine.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.cert_head_v7_post_refine.restype = ctypes.c_int

            lib.cert_head_v7_exact_all.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.cert_head_v7_exact_all.restype = ctypes.c_int
            _LIB = lib
    return _LIB


def build_cuda_table_v7(w, cfg_id: int = _CFG_ID) -> CudaCertTableV7:
    """Offline pack a CUDA bf16 LM-head shard `w [V, K]` into a GPU `CudaCertTableV7`."""
    assert torch is not None
    dev = w.device
    w_f32 = w.detach().float().cpu().numpy()
    probe = check_nw_heterogeneous(w_f32)
    pk = pack_cert_v7_np(w_f32, cfg_id=cfg_id)
    if os.environ.get("INKLING_CERT_VERBOSE", "1") != "0":
        print(
            f"[cert-v7] packed V={pk.V} K={pk.K} cfg={cfg_id}: "
            f"S1={pk.stage1_bytes/1e6:.2f}MB S2={pk.stage2_bytes/1e6:.2f}MB "
            f"(GM/RMS={probe['gm_over_rms']:.3f})",
            flush=True,
        )
    return CudaCertTableV7(
        slabs_u4=torch.from_numpy(pk.slabs_u4).to(dev).contiguous(),
        ms_u4=torch.from_numpy(pk.ms_u4).to(dev).contiguous(),
        big_s=torch.from_numpy(pk.big_s).to(dev).contiguous(),
        lsb1_row=torch.from_numpy(pk.lsb1_row).to(dev).contiguous(),
        ms_row=torch.from_numpy(pk.ms_row).to(dev).contiguous(),
        big_s_row=torch.from_numpy(pk.big_s_row).to(dev).contiguous(),
        rt1=torch.from_numpy(pk.rt1).to(dev).contiguous(),
        nv1=torch.from_numpy(pk.nv1).to(dev).contiguous(),
        rt2=torch.from_numpy(pk.rt2).to(dev).contiguous(),
        nv2=torch.from_numpy(pk.nv2).to(dev).contiguous(),
        V=pk.V,
        K=pk.K,
        cfg_id=cfg_id,
        cacc2=cacc2_for_k(pk.K),
    )


def alloc_state_v7(M: int, V: int, K: int, world_size: int, device) -> CudaCertStateV7:
    assert torch is not None
    m_rows = 8 if M <= 8 else 16
    return CudaCertStateV7(
        x_pad=torch.empty((m_rows, K), dtype=torch.bfloat16, device=device),
        xn=torch.empty((16,), dtype=torch.float32, device=device),
        bad=torch.empty((16,), dtype=torch.int32, device=device),
        s1_out=torch.empty((M, V), dtype=torch.float32, device=device),
        hi=torch.empty((M, V), dtype=torch.float16, device=device),
        thr_u=torch.empty((M,), dtype=torch.uint32, device=device),
        thr_ag=torch.empty((world_size, M), dtype=torch.uint32, device=device),
        t_glob=torch.empty((16,), dtype=torch.float32, device=device),
        idx1=torch.empty((V,), dtype=torch.int32, device=device),
        cnt1=torch.zeros((1,), dtype=torch.int32, device=device),
        idx2=torch.empty((V,), dtype=torch.int32, device=device),
        cnt2=torch.zeros((1,), dtype=torch.int32, device=device),
        best=torch.zeros((16,), dtype=torch.uint64, device=device),
        arrive=torch.zeros((1,), dtype=torch.int32, device=device),
        out_pair=torch.empty((M, 4), dtype=torch.float32, device=device),
        nlive_full=torch.tensor([M], dtype=torch.int32, device=device),
    )


def certified_local_pairs_v7(
    x,
    w_bf16,
    table: CudaCertTableV7,
    state: CudaCertStateV7,
    vreal: int,
    vstart: int,
    cap: float = 30.0,
    ag_thr_fn=None,
    world_size: int = 1,
    logits_idx=None,
    n_live_buf=None,
    pdl: bool = _USE_PDL,
):
    """Run full 0-ULP CertHead-v7 pipeline:
      1. `cert_head_v7_pre_screen` (optional direct LOGITS_IDX gather + 91.11 MB INT6_g512_ms8 screen)
      2. Optional 32B RDMA AG1 (`ag_thr_fn(state.thr_u, state.thr_ag)`)
      3. `cert_head_v7_post_refine` (`cert_cand1` -> `cert_stage2_lsb1` -> L2-prefetched `cert_refine_argmax`)
    Returns `state.out_pair` `[M, 4]` fp32 (`(max_val, global_idx, 0, 0)`).
    """
    lib = _load_lib()
    assert lib is not None and torch is not None
    M = int(logits_idx.shape[0]) if logits_idx is not None else int(x.shape[0])
    nl_buf = n_live_buf if n_live_buf is not None else state.nlive_full
    stream = int(torch.cuda.current_stream(x.device).cuda_stream)

    rc = lib.cert_head_v7_pre_screen(
        int(x.data_ptr()),
        int(logits_idx.data_ptr()) if logits_idx is not None else 0,
        int(table.slabs_u4.data_ptr()),
        int(table.ms_u4.data_ptr()),
        int(table.big_s.data_ptr()),
        int(table.rt1.data_ptr()),
        int(table.nv1.data_ptr()),
        int(state.x_pad.data_ptr()),
        int(state.xn.data_ptr()),
        int(state.bad.data_ptr()),
        int(state.s1_out.data_ptr()),
        int(state.hi.data_ptr()),
        int(state.thr_u.data_ptr()),
        int(state.cnt1.data_ptr()),
        int(state.cnt2.data_ptr()),
        int(state.best.data_ptr()),
        int(state.arrive.data_ptr()),
        int(nl_buf.data_ptr()),
        M, table.V, vreal, table.K, int(x.stride(0)),
        table.cfg_id, 1 if pdl else 0,
        table.cacc2, _INFL_X, _X_ABS, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"cert_head_v7_pre_screen failed: cudaError={rc}")

    if ag_thr_fn is not None and world_size > 1:
        ag_thr_fn(state.thr_u, state.thr_ag)
        thr_ag_ptr = int(state.thr_ag.data_ptr())
        ws = world_size
        tstride = int(state.thr_ag.stride(0))
    else:
        thr_ag_ptr = int(state.thr_u.data_ptr())
        ws = 1
        tstride = M

    inv_cap = 1.0 / cap if cap > 0.0 else 0.0
    rc = lib.cert_head_v7_post_refine(
        int(state.x_pad.data_ptr()),
        int(w_bf16.data_ptr()),
        int(table.lsb1_row.data_ptr()),
        int(table.ms_row.data_ptr()),
        int(table.big_s_row.data_ptr()),
        int(state.s1_out.data_ptr()),
        int(table.rt2.data_ptr()),
        int(table.nv2.data_ptr()),
        int(state.xn.data_ptr()),
        int(state.hi.data_ptr()),
        thr_ag_ptr,
        int(state.bad.data_ptr()),
        int(state.t_glob.data_ptr()),
        int(state.idx1.data_ptr()),
        int(state.cnt1.data_ptr()),
        int(state.idx2.data_ptr()),
        int(state.cnt2.data_ptr()),
        int(state.best.data_ptr()),
        int(state.arrive.data_ptr()),
        int(state.out_pair.data_ptr()),
        int(nl_buf.data_ptr()),
        M, table.V, vreal, vstart, table.K, int(w_bf16.stride(0)),
        ws, tstride, _REF_GRID, 1 if pdl else 0,
        table.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
        inv_cap, cap, _SLACK,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"cert_head_v7_post_refine failed: cudaError={rc}")
    return state.out_pair


def _get_tp_world_size() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


class CudaCertStateV7Adapter:
    """Drop-in adapter compatible with sync_free.py's CudaCertState interface."""

    def __init__(self, tbl: CudaCertTableV7, W, vreal: int, vstart: int, cap: float = 30.0, dip: float = 0.0):
        assert torch is not None
        self.tbl = tbl
        self.W = W
        self.vreal = int(vreal)
        self.vstart = int(vstart)
        self.cap = float(np.float32(cap)) if cap else 0.0
        self.dip = float(dip)
        self.world_size = _get_tp_world_size()
        self.state = alloc_state_v7(16, tbl.V, tbl.K, max(1, self.world_size), W.device)
        # Ensure thr_u & out_pair are 16-row graph-safe workspaces
        self.state.thr_u = torch.full((16,), 0x007FFFFF, dtype=torch.uint32, device=W.device)
        self.state.out_pair = torch.empty((16, 4), dtype=torch.float32, device=W.device)
        self._nlive_full = {m: torch.tensor([m], dtype=torch.int32, device=W.device) for m in range(1, 17)}
        self.m_ok: dict[int, bool] = {}
        self.is_v7 = True

    def qualify(self, max_m: int = 16) -> list[int]:
        lib = _load_lib()
        if lib is None:
            return list(range(2, max_m + 1))
        bad = []
        V, K = self.tbl.V, self.tbl.K
        for M in range(2, min(16, max_m) + 1):
            g = torch.Generator(device=self.W.device)
            g.manual_seed(1234 + M)
            ok = True
            m_pad = 8 if M <= 8 else 16
            for sc in (1.0, 0.3, 4.0):
                x = (torch.randn((M, K), device=self.W.device, generator=g) * sc).to(torch.bfloat16)
                ref = torch.mm(x, self.W.t(), out_dtype=torch.float32)
                self.state.x_pad[:M].copy_(x)
                if M < m_pad:
                    self.state.x_pad[M:m_pad].zero_()
                out = torch.empty_like(ref)
                rc = lib.cert_head_v7_exact_all(
                    int(self.state.x_pad.data_ptr()), int(self.W.data_ptr()), int(out.data_ptr()),
                    int(self._nlive_full[M].data_ptr()), M, V, K, int(self.W.stride(0)),
                    _REF_GRID, int(torch.cuda.current_stream(self.W.device).cuda_stream),
                )
                if rc != 0 or not bool(torch.equal(out.view(torch.int32), ref.view(torch.int32))):
                    ok = False
                    break
            self.m_ok[M] = ok
            if not ok:
                bad.append(M)
        return bad

    def local_pairs(self, x, nlive=None, logits_idx=None, pdl: bool = _USE_PDL):
        M = int(logits_idx.shape[0]) if logits_idx is not None else int(x.shape[0])
        if not ENABLED or M < 2 or M > 16 or not self.m_ok.get(M, False):
            return None
        lib = _load_lib()
        if lib is None:
            return None
        if x.dtype != torch.bfloat16 or x.stride(1) != 1:
            x = x.contiguous()
        nl_buf = nlive if nlive is not None else self._nlive_full[M]
        stream = int(torch.cuda.current_stream(x.device).cuda_stream)
        t = self.tbl
        st = self.state
        rc = lib.cert_head_v7_pre_screen(
            int(x.data_ptr()),
            int(logits_idx.data_ptr()) if logits_idx is not None else 0,
            int(t.slabs_u4.data_ptr()),
            int(t.ms_u4.data_ptr()),
            int(t.big_s.data_ptr()),
            int(t.rt1.data_ptr()),
            int(t.nv1.data_ptr()),
            int(st.x_pad.data_ptr()),
            int(st.xn.data_ptr()),
            int(st.bad.data_ptr()),
            int(st.s1_out.data_ptr()),
            int(st.hi.data_ptr()),
            int(st.thr_u.data_ptr()),
            int(st.cnt1.data_ptr()),
            int(st.cnt2.data_ptr()),
            int(st.best.data_ptr()),
            int(st.arrive.data_ptr()),
            int(nl_buf.data_ptr()),
            M, t.V, self.vreal, t.K, int(x.stride(0)),
            t.cfg_id, 1 if pdl else 0,
            t.cacc2, _INFL_X, _X_ABS, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"cert_head_v7_pre_screen failed: cudaError={rc}")

        if self.world_size > 1:
            from vllm.distributed import tensor_model_parallel_all_gather
            thr_stride = ((M + 3) // 4) * 4
            thr_ag = tensor_model_parallel_all_gather(st.thr_u[:thr_stride].view(torch.int32), dim=0)
            if not thr_ag.is_contiguous():
                thr_ag = thr_ag.contiguous()
            thr_ag_ptr = int(thr_ag.data_ptr())
            ws = self.world_size
        else:
            thr_stride = M
            thr_ag_ptr = int(st.thr_u.data_ptr())
            ws = 1

        inv_cap = 1.0 / self.cap if self.cap > 0.0 else 0.0
        rc = lib.cert_head_v7_post_refine(
            int(st.x_pad.data_ptr()),
            int(self.W.data_ptr()),
            int(t.lsb1_row.data_ptr()),
            int(t.ms_row.data_ptr()),
            int(t.big_s_row.data_ptr()),
            int(st.s1_out.data_ptr()),
            int(t.rt2.data_ptr()),
            int(t.nv2.data_ptr()),
            int(st.xn.data_ptr()),
            int(st.hi.data_ptr()),
            thr_ag_ptr,
            int(st.bad.data_ptr()),
            int(st.t_glob.data_ptr()),
            int(st.idx1.data_ptr()),
            int(st.cnt1.data_ptr()),
            int(st.idx2.data_ptr()),
            int(st.cnt2.data_ptr()),
            int(st.best.data_ptr()),
            int(st.arrive.data_ptr()),
            int(st.out_pair.data_ptr()),
            int(nl_buf.data_ptr()),
            M, t.V, self.vreal, self.vstart, t.K, int(self.W.stride(0)),
            ws, thr_stride, _REF_GRID, 1 if pdl else 0,
            t.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            inv_cap, self.cap, _SLACK,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"cert_head_v7_post_refine failed: cudaError={rc}")
        return st.out_pair[:M]
