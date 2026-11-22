#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone 8-Rank NCCL Benchmark & SNR/Correctness Verifier for Zero-Incast 3-Step RHD
ReduceScatter & AllGather on 8x DGX Spark GB10 (SM121a, 4x 200GbE ConnectX-7 RoCEv2).

Run on spark-one..eight (example via torchrun):
  torchrun --nnodes=8 --nproc_per_node=1 --node_rank=$NODE_RANK \
      --master_addr=192.168.100.1 --master_port=29610 \
      bench_rhd_8rank.py --lib /rdma/libpf_rhd_comm_sm121.so

Measures & verifies:
  1. Pairwise 1-to-1 RoCEv2 wire saturation (4.87, 9.73, 19.46, 38.93 MB) across:
       - p2p_world (8-rank PG batch_isend_irecv)
       - pynccl_p2p (vLLM PyNcclCommunicator on 8-rank comm, if vLLM installed)
       - sliced_p2p (K=4 sub-chunks in 1 group)
       - pair_subcomm (dedicated 2-rank NCCL sub-communicator {r, r^bit})
       - pair_ag2 (2-rank NCCL Ring all_gather_into_tensor on {r, r^bit})
  2. 8-Rank ReduceScatter @ T=8192 & T=16384, H=4608:
       NCCL BF16 Ring vs rhd_int8 (pipe=0/1) vs rhd_fp8 (pipe=0/1)
  3. 8-Rank AllGather @ T=8192 & T=16384, H=4608:
       NCCL BF16 Ring vs NCCL 2-call FP8 Ring vs rhd_int8 AG vs rhd_fp8 AG vs rhd_fp8 direct AG
       + verifies 100% bit-identical AG outputs across all 8 TP ranks!
  4. Multi-regime SNR/Cosine/MaxErr & edge-case verification:
       Normal, 100x Clustered Outlier, 100x Scattered Outlier, Random 4x Cancel, Tiny 1e-7,
       All-Zero, and literal 0-ULP bit-identity between pipe=0 and pipe=1!
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
from typing import Any

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Native SM121a RHD Library Loader
# ---------------------------------------------------------------------------
def load_rhd_lib(path: str) -> ctypes.CDLL | None:
    if not path or not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    _quant_args = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    _split_args = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    _daq_args = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    _dab_args = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    _agd_args = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    for n in ("rhd_quant_slice_int8_g64_fwd", "rhd_quant_slice_fp8_g64_fwd"):
        fn = getattr(lib, n)
        fn.argtypes = _quant_args
        fn.restype = ctypes.c_int
    for n in ("rhd_dequant_add_split_quant_int8_g64_fwd", "rhd_dequant_add_split_quant_fp8_g64_fwd"):
        fn = getattr(lib, n)
        fn.argtypes = _split_args
        fn.restype = ctypes.c_int
    for n in ("rhd_dequant_add_quant_int8_g64_fwd", "rhd_dequant_add_quant_fp8_g64_fwd"):
        fn = getattr(lib, n)
        fn.argtypes = _daq_args
        fn.restype = ctypes.c_int
    for n in ("rhd_dequant_add_bf16_int8_g64_fwd", "rhd_dequant_add_bf16_fp8_g64_fwd"):
        fn = getattr(lib, n)
        fn.argtypes = _dab_args
        fn.restype = ctypes.c_int
    for n in ("rhd_ag_dequant_int8_g64_fwd", "rhd_ag_dequant_fp8_g64_fwd"):
        fn = getattr(lib, n)
        fn.argtypes = _agd_args
        fn.restype = ctypes.c_int
    lib.rhd_ag_quant_fp8_rowwise_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.rhd_ag_quant_fp8_rowwise_fwd.restype = ctypes.c_int
    lib.rhd_ag_unpack_fp8_rowwise_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.rhd_ag_unpack_fp8_rowwise_fwd.restype = ctypes.c_int
    return lib


# ---------------------------------------------------------------------------
# Pure-PyTorch Fallback Codecs (when .so is absent, matches g64 + BF16 exact!)
# ---------------------------------------------------------------------------
def _g64_packed_bytes(M: int, H: int) -> tuple[int, int]:
    sc_off = M * H
    total = (sc_off + 2 * M * (H // 64) + 127) & ~127
    return sc_off, total


def _rowwise_fp8_shard_bytes(S: int, H: int) -> tuple[int, int]:
    sc_off = S * H
    total = (sc_off + 4 * S + 127) & ~127
    return sc_off, total


def _torch_quant_g64(x_bf16: torch.Tensor, dst_u8: torch.Tensor, sc_off: int, is_fp8: bool) -> None:
    M, H = x_bf16.shape
    G = H // 64
    xg = x_bf16.float().view(M, G, 64)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    inv_qmax = (1.0 / 448.0) if is_fp8 else (1.0 / 127.0)
    sc_bf = (amax * inv_qmax).clamp(min=1e-12).to(torch.bfloat16)
    scaled = xg / sc_bf.float()
    if is_fp8:
        q_u8 = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        q_u8 = scaled.round().clamp(-127.0, 127.0).to(torch.int8).view(torch.uint8)
    dst_u8[:sc_off].copy_(q_u8.reshape(-1))
    dst_u8[sc_off : sc_off + 2 * M * G].copy_(sc_bf.view(torch.uint8).reshape(-1))


def _torch_dequant_g64(src_u8: torch.Tensor, M: int, H: int, sc_off: int, is_fp8: bool) -> torch.Tensor:
    G = H // 64
    q_u8 = src_u8[:sc_off].view(M, G, 64)
    sc = src_u8[sc_off : sc_off + 2 * M * G].view(torch.bfloat16).view(M, G, 1).float()
    qf = q_u8.view(torch.float8_e4m3fn).float() if is_fp8 else q_u8.view(torch.int8).float()
    return (qf * sc).view(M, H)


# ---------------------------------------------------------------------------
# Pairwise Transport Harness (supports all 5 transports head-to-head!)
# ---------------------------------------------------------------------------
class PairTransports:
    def __init__(self, rank: int, world: int, device: torch.device, slices: int = 4) -> None:
        self.rank = rank
        self.world = world
        self.device = device
        self.slices = slices
        self.pynccl_world = None
        self.pair_pgs: dict[int, tuple[Any, int, int]] = {}  # bit -> (pg, peer_local, peer_global)
        self.pair_pynccl: dict[int, tuple[Any, int]] = {}    # bit -> (pynccl, peer_local)
        self.ag2_scratch: torch.Tensor | None = None

        # Optional vLLM PyNcclCommunicator on world & 2-rank pairs
        PyNcclCommunicator = None
        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        except Exception:
            PyNcclCommunicator = None

        if PyNcclCommunicator is not None:
            gloo_world = dist.new_group(ranks=list(range(world)), backend="gloo")
            self.pynccl_world = PyNcclCommunicator(group=gloo_world, device=device)

        bit = world >> 1
        while bit >= 1:
            for r in range(world):
                p = r ^ bit
                if r < p:
                    pg = dist.new_group(ranks=[r, p], backend="nccl")
                    if rank in (r, p):
                        local_r = 0 if rank == r else 1
                        peer_g = p if rank == r else r
                        self.pair_pgs[bit] = (pg, 1 - local_r, peer_g)
                    if PyNcclCommunicator is not None:
                        gloo_p = dist.new_group(ranks=[r, p], backend="gloo")
                        if rank in (r, p):
                            local_r = 0 if rank == r else 1
                            self.pair_pynccl[bit] = (
                                PyNcclCommunicator(group=gloo_p, device=device),
                                1 - local_r,
                            )
            bit >>= 1

    def available_names(self) -> list[str]:
        names = ["p2p_world", "sliced_p2p", "pair_subcomm", "pair_ag2"]
        if self.pynccl_world is not None:
            names.insert(0, "pynccl_p2p")
            names.append("pair_pynccl")
        return names

    def exchange(self, kind: str, s_view: torch.Tensor, r_view: torch.Tensor, partner: int, bit: int) -> None:
        nbytes = int(s_view.numel())
        if kind == "pynccl_p2p" and self.pynccl_world is not None:
            c = self.pynccl_world
            c.group_start()
            c.send(s_view, partner)
            c.recv(r_view, partner)
            c.group_end()
            return
        if kind == "pair_pynccl" and bit in self.pair_pynccl:
            c, peer_l = self.pair_pynccl[bit]
            c.group_start()
            c.send(s_view, peer_l)
            c.recv(r_view, peer_l)
            c.group_end()
            return
        if kind == "sliced_p2p":
            k = self.slices
            chunk = (nbytes // k) & ~127
            if self.pynccl_world is not None and chunk > 0:
                c = self.pynccl_world
                c.group_start()
                off = 0
                for i in range(k):
                    end = nbytes if i == k - 1 else off + chunk
                    c.send(s_view[off:end], partner)
                    c.recv(r_view[off:end], partner)
                    off = end
                c.group_end()
                return
            ops = []
            off = 0
            for i in range(k):
                end = nbytes if i == k - 1 else off + chunk
                ops.append(dist.P2POp(dist.isend, s_view[off:end], partner))
                ops.append(dist.P2POp(dist.irecv, r_view[off:end], partner))
                off = end
            for w in dist.batch_isend_irecv(ops):
                w.wait()
            return
        if kind == "pair_subcomm":
            pg, _peer_l, peer_g = self.pair_pgs[bit]
            ops = [
                dist.P2POp(dist.isend, s_view, peer_g, group=pg),
                dist.P2POp(dist.irecv, r_view, peer_g, group=pg),
            ]
            for w in dist.batch_isend_irecv(ops):
                w.wait()
            return
        if kind == "pair_ag2":
            pg, peer_l, _peer_g = self.pair_pgs[bit]
            need = 2 * nbytes
            if self.ag2_scratch is None or self.ag2_scratch.numel() < need:
                self.ag2_scratch = torch.empty((need,), dtype=torch.uint8, device=self.device)
            ag2 = self.ag2_scratch[:need].view(2, nbytes)
            dist.all_gather_into_tensor(ag2, s_view, group=pg)
            r_view.copy_(ag2[peer_l])
            return
        # Default p2p_world
        ops = [
            dist.P2POp(dist.isend, s_view, partner),
            dist.P2POp(dist.irecv, r_view, partner),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()


# ---------------------------------------------------------------------------
# RHD ReduceScatter & AllGather Runners (using libpf_rhd_comm_sm121.so!)
# ---------------------------------------------------------------------------
class RHDRunner:
    def __init__(self, lib: ctypes.CDLL | None, rank: int, world: int, device: torch.device, max_T: int, H: int) -> None:
        self.lib = lib
        self.rank = rank
        self.world = world
        self.device = device
        self.H = H
        half0 = max_T // 2
        _, b0 = _g64_packed_bytes(half0, H)
        _, sub_b = _g64_packed_bytes(half0 // 2, H)
        max_b = max(b0, 2 * sub_b)
        self.send0 = torch.empty((max_b,), dtype=torch.uint8, device=device)
        self.recv0 = torch.empty((max_b,), dtype=torch.uint8, device=device)
        self.send1 = torch.empty((max(sub_b, 256),), dtype=torch.uint8, device=device)
        self.recv1 = torch.empty((max(sub_b, 256),), dtype=torch.uint8, device=device)
        self.keep1 = torch.empty((max(half0 // 2, 1) * H,), dtype=torch.bfloat16, device=device)
        self.keep2 = torch.empty((max(max_T // world, 1) * H,), dtype=torch.bfloat16, device=device)
        _, ag_sb = _g64_packed_bytes(max_T // world, H)
        self.ag_buf = torch.empty((world, ag_sb), dtype=torch.uint8, device=device)
        self.codec_stream = torch.cuda.Stream(device=device, priority=-1)
        self.evs = {k: torch.cuda.Event() for k in ("start", "qA", "qB", "cA", "cB", "s1", "k1")}

    def reduce_scatter(
        self,
        partial: torch.Tensor,
        out_shard: torch.Tensor,
        is_fp8: bool,
        transports: PairTransports,
        t_kind: str,
        pipe: bool = False,
    ) -> None:
        T, H = partial.shape
        S = T // self.world
        K = self.world.bit_length() - 1
        cur = torch.cuda.current_stream(self.device)
        cur_u64 = int(cur.cuda_stream)
        lib = self.lib

        half0 = T // 2
        sc0_off, b0 = _g64_packed_bytes(half0, H)
        bit0 = self.world >> 1
        send_upper0 = (self.rank & bit0) == 0
        send0_off = half0 if send_upper0 else 0
        keep0_off = 0 if send_upper0 else half0

        if lib is None:
            # Pure-PyTorch fallback (always unpipelined)
            cur_send = self.send0[:b0]
            cur_recv = self.recv0[:b0]
            _torch_quant_g64(partial[send0_off : send0_off + half0], cur_send, sc0_off, is_fp8)
            cur_local = partial[keep0_off : keep0_off + half0]
            cur_sc_off = sc0_off
            cur_M = half0
            for step in range(K):
                bit = self.world >> (step + 1)
                transports.exchange(t_kind, cur_send, cur_recv, self.rank ^ bit, bit)
                s_f32 = cur_local.float() + _torch_dequant_g64(cur_recv, cur_M, H, cur_sc_off, is_fp8)
                if step == K - 1:
                    out_shard.copy_(s_f32.to(torch.bfloat16))
                else:
                    nxt_M = cur_M // 2
                    nxt_bit = bit >> 1
                    su = (self.rank & nxt_bit) == 0
                    nxt_sc_off, nxt_b = _g64_packed_bytes(nxt_M, H)
                    nxt_send = self.send1[:nxt_b] if (step % 2 == 0) else self.send0[:nxt_b]
                    nxt_recv = self.recv1[:nxt_b] if (step % 2 == 0) else self.recv0[:nxt_b]
                    s_part = s_f32[nxt_M:] if su else s_f32[:nxt_M]
                    k_part = s_f32[:nxt_M] if su else s_f32[nxt_M:]
                    _torch_quant_g64(s_part, nxt_send, nxt_sc_off, is_fp8)
                    cur_local = k_part.to(torch.bfloat16)
                    cur_send, cur_recv, cur_sc_off, cur_M = nxt_send, nxt_recv, nxt_sc_off, nxt_M
            return

        q_fn = lib.rhd_quant_slice_fp8_g64_fwd if is_fp8 else lib.rhd_quant_slice_int8_g64_fwd
        split_fn = (
            lib.rhd_dequant_add_split_quant_fp8_g64_fwd
            if is_fp8
            else lib.rhd_dequant_add_split_quant_int8_g64_fwd
        )
        daq_fn = lib.rhd_dequant_add_quant_fp8_g64_fwd if is_fp8 else lib.rhd_dequant_add_quant_int8_g64_fwd
        dab_fn = lib.rhd_dequant_add_bf16_fp8_g64_fwd if is_fp8 else lib.rhd_dequant_add_bf16_int8_g64_fwd

        if pipe and self.world >= 4:
            cs = self.codec_stream
            cs_u64 = int(cs.cuda_stream)
            sub_M = half0 // 2
            sub_sc_off, sub_b = _g64_packed_bytes(sub_M, H)
            bit1 = bit0 >> 1
            send_upper1 = (self.rank & bit1) == 0
            sub_A_off = sub_M if send_upper1 else 0
            sub_B_off = 0 if send_upper1 else sub_M

            send0_A = self.send0[:sub_b]
            send0_B = self.send0[sub_b : 2 * sub_b]
            recv0_A = self.recv0[:sub_b]
            recv0_B = self.recv0[sub_b : 2 * sub_b]
            send1 = self.send1[:sub_b]
            recv1 = self.recv1[:sub_b]
            keep1 = self.keep1[: sub_M * H].view(sub_M, H)
            wb_k1 = 1 if (sub_M * H * 2 <= (20 << 20)) else 0

            self.evs["start"].record(cur)
            cs.wait_event(self.evs["start"])

            assert q_fn(partial.data_ptr(), send0_A.data_ptr(), H, send0_off + sub_A_off, sub_sc_off, sub_M, H, cs_u64) == 0
            self.evs["qA"].record(cs)
            assert q_fn(partial.data_ptr(), send0_B.data_ptr(), H, send0_off + sub_B_off, sub_sc_off, sub_M, H, cs_u64) == 0
            self.evs["qB"].record(cs)

            cur.wait_event(self.evs["qA"])
            transports.exchange(t_kind, send0_A, recv0_A, self.rank ^ bit0, bit0)
            self.evs["cA"].record(cur)

            cs.wait_event(self.evs["cA"])
            assert daq_fn(
                recv0_A.data_ptr(), partial.data_ptr(), send1.data_ptr(),
                H, keep0_off + sub_A_off, sub_sc_off, sub_sc_off, sub_M, H, cs_u64,
            ) == 0
            self.evs["s1"].record(cs)

            cur.wait_event(self.evs["qB"])
            transports.exchange(t_kind, send0_B, recv0_B, self.rank ^ bit0, bit0)
            self.evs["cB"].record(cur)

            cs.wait_event(self.evs["cB"])
            assert dab_fn(
                recv0_B.data_ptr(), partial.data_ptr(), keep1.data_ptr(),
                H, keep0_off + sub_B_off, sub_sc_off, H, sub_M, H, wb_k1, cs_u64,
            ) == 0
            self.evs["k1"].record(cs)

            cur.wait_event(self.evs["s1"])
            transports.exchange(t_kind, send1, recv1, self.rank ^ bit1, bit1)
            cur.wait_event(self.evs["k1"])

            if K == 2:
                assert dab_fn(recv1.data_ptr(), keep1.data_ptr(), out_shard.data_ptr(), H, 0, sub_sc_off, H, S, H, 1, cur_u64) == 0
                return

            next_bit = bit1 >> 1
            su2 = 1 if ((self.rank & next_bit) == 0) else 0
            sc2_off, b2 = _g64_packed_bytes(S, H)
            send2 = self.send0[:b2]
            recv2 = self.recv0[:b2]
            keep2 = self.keep2[: S * H].view(S, H)
            assert split_fn(
                recv1.data_ptr(), keep1.data_ptr(), send2.data_ptr(), keep2.data_ptr(),
                H, 0, sub_sc_off, sc2_off, H, S, H, su2, 1, cur_u64,
            ) == 0
            transports.exchange(t_kind, send2, recv2, self.rank ^ next_bit, next_bit)
            assert dab_fn(recv2.data_ptr(), keep2.data_ptr(), out_shard.data_ptr(), H, 0, sc2_off, H, S, H, 1, cur_u64) == 0
            return

        # Unpipelined fused split-half (4 kernels for W=8)
        cur_send = self.send0[:b0]
        cur_recv = self.recv0[:b0]
        assert q_fn(partial.data_ptr(), cur_send.data_ptr(), H, send0_off, sc0_off, half0, H, cur_u64) == 0
        cur_local = partial
        cur_local_off = keep0_off
        cur_sc_off = sc0_off
        cur_M = half0
        for step in range(K):
            bit = self.world >> (step + 1)
            transports.exchange(t_kind, cur_send, cur_recv, self.rank ^ bit, bit)
            if step == K - 1:
                assert dab_fn(
                    cur_recv.data_ptr(), cur_local.data_ptr(), out_shard.data_ptr(),
                    H, cur_local_off, cur_sc_off, H, S, H, 1, cur_u64,
                ) == 0
            else:
                nxt_M = cur_M // 2
                nxt_bit = bit >> 1
                su = 1 if ((self.rank & nxt_bit) == 0) else 0
                nxt_sc_off, nxt_b = _g64_packed_bytes(nxt_M, H)
                if step % 2 == 0:
                    nxt_send = self.send1[:nxt_b]
                    nxt_recv = self.recv1[:nxt_b]
                    nxt_keep = self.keep1[: nxt_M * H].view(nxt_M, H)
                else:
                    nxt_send = self.send0[:nxt_b]
                    nxt_recv = self.recv0[:nxt_b]
                    nxt_keep = self.keep2[: nxt_M * H].view(nxt_M, H)
                wb_k = 1 if (nxt_M * H * 2 <= (20 << 20)) else 0
                assert split_fn(
                    cur_recv.data_ptr(), cur_local.data_ptr(), nxt_send.data_ptr(), nxt_keep.data_ptr(),
                    H, cur_local_off, cur_sc_off, nxt_sc_off, H, nxt_M, H, su, wb_k, cur_u64,
                ) == 0
                cur_local, cur_local_off, cur_send, cur_recv, cur_sc_off, cur_M = (
                    nxt_keep, 0, nxt_send, nxt_recv, nxt_sc_off, nxt_M
                )

    def all_gather(
        self,
        shard: torch.Tensor,
        out_bf16: torch.Tensor | None,
        out_fp8: torch.Tensor | None,
        out_scale: torch.Tensor | None,
        is_fp8: bool,
        fp8_direct: bool,
        transports: PairTransports,
        t_kind: str,
    ) -> None:
        S, H = shard.shape
        K = self.world.bit_length() - 1
        cur_u64 = int(torch.cuda.current_stream(self.device).cuda_stream)
        lib = self.lib

        if fp8_direct:
            sc_off, sb = _rowwise_fp8_shard_bytes(S, H)
        else:
            sc_off, sb = _g64_packed_bytes(S, H)
        ag_view = self.ag_buf[:, :sb]

        if fp8_direct:
            if lib is not None:
                assert lib.rhd_ag_quant_fp8_rowwise_fwd(
                    shard.data_ptr(), ag_view[self.rank].data_ptr(), H, sc_off, S, H, cur_u64
                ) == 0
            else:
                amax = shard.float().abs().amax(dim=-1, keepdim=True)
                sc = (amax * (1.0 / 448.0)).clamp(min=1e-12)
                q = (shard.float() / sc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                ag_view[self.rank, :sc_off].copy_(q.view(torch.uint8).reshape(-1))
                ag_view[self.rank, sc_off : sc_off + 4 * S].copy_(sc.view(torch.uint8).reshape(-1))
        else:
            if lib is not None:
                q_fn = lib.rhd_quant_slice_fp8_g64_fwd if is_fp8 else lib.rhd_quant_slice_int8_g64_fwd
                assert q_fn(shard.data_ptr(), ag_view[self.rank].data_ptr(), H, 0, sc_off, S, H, cur_u64) == 0
            else:
                _torch_quant_g64(shard, ag_view[self.rank], sc_off, is_fp8)

        for s in range(K):
            bit = 1 << s
            chunk = 1 << s
            mask = ~(chunk - 1)
            partner = self.rank ^ bit
            sb_idx = self.rank & mask
            rb_idx = partner & mask
            transports.exchange(
                t_kind,
                ag_view[sb_idx : sb_idx + chunk].reshape(-1),
                ag_view[rb_idx : rb_idx + chunk].reshape(-1),
                partner,
                bit,
            )

        if fp8_direct:
            assert out_fp8 is not None and out_scale is not None
            if lib is not None:
                assert lib.rhd_ag_unpack_fp8_rowwise_fwd(
                    ag_view.data_ptr(), out_fp8.data_ptr(), out_scale.data_ptr(),
                    sb, sc_off, H, S, H, self.world, cur_u64,
                ) == 0
            else:
                out_fp8.view(torch.uint8).copy_(ag_view[:, :sc_off].reshape(self.world * S, H))
                out_scale.copy_(ag_view[:, sc_off : sc_off + 4 * S].contiguous().view(torch.float32).reshape(self.world * S, 1))
        else:
            assert out_bf16 is not None
            if lib is not None:
                d_fn = lib.rhd_ag_dequant_fp8_g64_fwd if is_fp8 else lib.rhd_ag_dequant_int8_g64_fwd
                assert d_fn(ag_view.data_ptr(), out_bf16.data_ptr(), sb, sc_off, H, S, H, self.world, cur_u64) == 0
            else:
                for r in range(self.world):
                    out_bf16[r * S : (r + 1) * S].copy_(
                        _torch_dequant_g64(ag_view[r], S, H, sc_off, is_fp8).to(torch.bfloat16)
                    )


# ---------------------------------------------------------------------------
# Metrics & Timing Helpers
# ---------------------------------------------------------------------------
def compute_metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    g = got.double()
    r = ref.double()
    diff = g - r
    ref_p = float(torch.mean(r * r).item())
    mse = float(torch.mean(diff * diff).item())
    max_abs = float(diff.abs().max().item())
    max_ref = float(r.abs().max().item())
    max_rel = max_abs / max(max_ref, 1e-12)
    if ref_p == 0.0:
        return (float("inf") if mse == 0.0 else 0.0), 1.0, max_abs, max_rel
    sqnr = 10.0 * math.log10(ref_p / max(mse, 1e-30))
    cos = float(torch.sum(g * r).item() / max(math.sqrt(float(torch.sum(g * g).item()) * float(torch.sum(r * r).item())), 1e-30))
    return sqnr, cos, max_abs, max_rel


def time_collective(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times: list[float] = []
    for _ in range(iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1))
    # Max across ranks per iteration gives true collective wall time
    t_dev = torch.tensor(times, dtype=torch.float32, device="cuda")
    dist.all_reduce(t_dev, op=dist.ReduceOp.MAX)
    vals = sorted(t_dev.cpu().tolist())
    med = vals[len(vals) // 2]
    p90 = vals[min(int(0.9 * len(vals)), len(vals) - 1)]
    return med, p90


def main() -> None:
    ap = argparse.ArgumentParser(description="8-Rank Zero-Incast RHD ReduceScatter & AllGather Benchmark")
    ap.add_argument("--lib", default="/rdma/libpf_rhd_comm_sm121.so", help="Path to libpf_rhd_comm_sm121.so")
    ap.add_argument("--H", type=int, default=4608, help="Hidden size (default 4608)")
    ap.add_argument("--tokens", default="8192,16384", help="Comma-separated T values (default 8192,16384)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--slices", type=int, default=4, help="Chunk count for sliced_p2p")
    ap.add_argument("--transports", default="all", help="Comma-separated subset of transports or 'all'")
    args = ap.parse_args()

    # Initialize distributed cleanly for torchrun or OpenMPI env vars
    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world in (2, 4, 8), f"RHD requires power-of-2 world_size in (2,4,8), got {world}"

    t_list = [int(x) for x in args.tokens.split(",") if x.strip()]
    H = args.H
    lib = load_rhd_lib(args.lib)
    transports = PairTransports(rank, world, device, slices=args.slices)
    avail = transports.available_names()
    if args.transports == "all":
        t_kinds = avail
    else:
        t_kinds = [x.strip() for x in args.transports.split(",") if x.strip() in avail]
    runner = RHDRunner(lib, rank, world, device, max(t_list), H)

    if rank == 0:
        print("=" * 108)
        print(f"8-Rank Zero-Incast 3-Step RHD Benchmark | world={world} H={H} T={t_list} "
              f"lib={'SM121a CUDA (' + args.lib + ')' if lib else 'PyTorch fallback'}")
        print(f"Transports: {t_kinds} | NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS', 'unset')} "
              f"NCCL_IB_QPS={os.environ.get('NCCL_IB_QPS_PER_CONNECTION', 'unset')} "
              f"NCCL_BUFFSIZE={os.environ.get('NCCL_BUFFSIZE', 'unset')}")
        print("=" * 108)

    # -----------------------------------------------------------------------
    # Phase 1: Pairwise 1-to-1 RoCEv2 Saturation Diagnostic (detects channel caps!)
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n[Phase 1] Pairwise 1-to-1 Bidirectional Wire Saturation (r <-> r^4, target 25.8 GB/s):")
        print(f"  {'Transport':14s} | {'4.87 MB':>14s} | {'9.73 MB':>14s} | {'19.46 MB':>14s} | {'38.93 MB':>14s}")
        print("  " + "-" * 80)

    sizes_mb = [4866048, 9732096, 19464192, 38928384]
    s_raw = torch.randint(0, 255, (max(sizes_mb),), dtype=torch.uint8, device=device)
    r_raw = torch.empty_like(s_raw)
    bit_diag = world >> 1
    partner_diag = rank ^ bit_diag
    best_t = t_kinds[0]
    best_bw = -1.0

    for tk in t_kinds:
        cells = []
        bw_19 = 0.0
        for nb in sizes_mb:
            sv = s_raw[:nb]
            rv = r_raw[:nb]
            med, _ = time_collective(
                lambda sv=sv, rv=rv, tk=tk: transports.exchange(tk, sv, rv, partner_diag, bit_diag),
                warmup=args.warmup, iters=args.iters,
            )
            bw = (nb / 1e9) / (med / 1e3)  # GB/s per direction (full-duplex bidirectional)
            if nb == 19464192:
                bw_19 = bw
            cells.append(f"{med:5.2f}ms {bw:5.1f}G")
        if bw_19 > best_bw:
            best_bw = bw_19
            best_t = tk
        if rank == 0:
            print(f"  {tk:14s} | " + " | ".join(f"{c:>14s}" for c in cells))

    if rank == 0:
        print(f"  --> Fastest 19.46 MB pairwise transport: {best_t} ({best_bw:.2f} GB/s bidirectional!)")

    # -----------------------------------------------------------------------
    # Phase 2: 8-Rank ReduceScatter Head-to-Head (T = 8192 & 16384, H = 4608)
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n[Phase 2] 8-Rank ReduceScatter (BF16 Ring vs Zero-Incast 3-Step RHD INT8/FP8):")
        print(f"  {'T':>5s} | {'Mode':10s} | {'Transport':12s} | {'Pipe':4s} | {'Med(ms)':>7s} | {'P90(ms)':>7s} | "
              f"{'WireGB/s':>8s} | {'EqBusBW':>8s} | {'Speedup':>7s} | {'SQNR':>7s} | {'Cos':>8s} | {'MaxErr':>8s}")
        print("  " + "-" * 118)

    for T in t_list:
        S = T // world
        torch.manual_seed(1000 + rank * 17 + T)
        x_in = (torch.randn((T, H), dtype=torch.float32, device=device) * 0.5).to(torch.bfloat16)
        out_ref = torch.empty((S, H), dtype=torch.bfloat16, device=device)
        out_got = torch.empty((S, H), dtype=torch.bfloat16, device=device)

        # Exact FP64 reference via FP32 all_reduce
        ref_f32 = x_in.float().clone()
        dist.all_reduce(ref_f32, op=dist.ReduceOp.SUM)
        ref_shard = ref_f32[rank * S : (rank + 1) * S].to(torch.bfloat16)

        bf16_wire_gb = ((world - 1) / world) * (T * H * 2) / 1e9
        rhd_wire_gb = sum(_g64_packed_bytes(T >> (s + 1), H)[1] for s in range(world.bit_length() - 1)) / 1e9

        med_bf16, p90_bf16 = time_collective(
            lambda: dist.reduce_scatter_tensor(out_ref, x_in),
            warmup=args.warmup, iters=args.iters,
        )
        bf16_bw = bf16_wire_gb / (med_bf16 / 1e3)
        sq_b, cs_b, me_b, _ = compute_metrics(out_ref, ref_shard)
        if rank == 0:
            print(f"  {T:5d} | {'bf16_ring':10s} | {'nccl_ring':12s} | {'-':4s} | {med_bf16:7.3f} | {p90_bf16:7.3f} | "
                  f"{bf16_bw:8.2f} | {bf16_bw:8.2f} | {'1.00x':>7s} | {sq_b:6.1f}dB | {cs_b:8.6f} | {me_b:8.2e}")

        # Sweep transports (always include best_t + p2p_world/pynccl_p2p for comparison)
        sweep_ts = [best_t]
        for cand in ("pynccl_p2p", "p2p_world", "pair_subcomm", "pair_ag2"):
            if cand in t_kinds and cand not in sweep_ts:
                sweep_ts.append(cand)

        for mode_name, is_fp8 in (("rhd_int8", False), ("rhd_fp8", True)):
            for tk in sweep_ts:
                pipe_opts = (False, True) if (lib is not None and tk == best_t) else (False,)
                for use_pipe in pipe_opts:
                    med, p90 = time_collective(
                        lambda is_fp8=is_fp8, tk=tk, use_pipe=use_pipe: runner.reduce_scatter(
                            x_in, out_got, is_fp8, transports, tk, pipe=use_pipe
                        ),
                        warmup=args.warmup, iters=args.iters,
                    )
                    wire_bw = rhd_wire_gb / (med / 1e3)
                    eq_bw = bf16_wire_gb / (med / 1e3)
                    sp = med_bf16 / med
                    sq, cs, me, _ = compute_metrics(out_got, ref_shard)
                    if rank == 0:
                        print(f"  {T:5d} | {mode_name:10s} | {tk:12s} | {int(use_pipe):4d} | {med:7.3f} | {p90:7.3f} | "
                              f"{wire_bw:8.2f} | {eq_bw:8.2f} | {sp:6.2f}x | {sq:6.1f}dB | {cs:8.6f} | {me:8.2e}")

    # -----------------------------------------------------------------------
    # Phase 3: 8-Rank AllGather Head-to-Head (T = 8192 & 16384, H = 4608)
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n[Phase 3] 8-Rank AllGather (BF16 Ring vs 2-Call FP8 Ring vs 3-Step RHD INT8/FP8/Direct):")
        print(f"  {'T':>5s} | {'Mode':14s} | {'Transport':12s} | {'Med(ms)':>7s} | {'P90(ms)':>7s} | "
              f"{'WireGB/s':>8s} | {'EqBusBW':>8s} | {'Speedup':>7s} | {'SQNR':>7s} | {'Cos':>8s} | {'RankBitEq':>9s}")
        print("  " + "-" * 118)

    for T in t_list:
        S = T // world
        torch.manual_seed(2000 + rank * 31 + T)
        x_s = (torch.randn((S, H), dtype=torch.float32, device=device) * 0.5).to(torch.bfloat16)
        ag_ref = torch.empty((T, H), dtype=torch.bfloat16, device=device)
        ag_out = torch.empty((T, H), dtype=torch.bfloat16, device=device)
        ag_fp8 = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=device)
        ag_sc = torch.empty((T, 1), dtype=torch.float32, device=device)

        bf16_wire_gb = ((world - 1) / world) * (T * H * 2) / 1e9
        g64_wire_gb = (world - 1) * _g64_packed_bytes(S, H)[1] / 1e9
        dir_wire_gb = (world - 1) * _rowwise_fp8_shard_bytes(S, H)[1] / 1e9

        med_bf16, p90_bf16 = time_collective(
            lambda: dist.all_gather_into_tensor(ag_ref, x_s),
            warmup=args.warmup, iters=args.iters,
        )
        bf16_bw = bf16_wire_gb / (med_bf16 / 1e3)
        if rank == 0:
            print(f"  {T:5d} | {'bf16_ring':14s} | {'nccl_ring':12s} | {med_bf16:7.3f} | {p90_bf16:7.3f} | "
                  f"{bf16_bw:8.2f} | {bf16_bw:8.2f} | {'1.00x':>7s} | {'exact':>7s} | {'1.000000':>8s} | {'EXACT':>9s}")

        # Baseline 2-call FP8 Ring AG (matches best.env SP_FP8_AG=1 direct)
        fp8_s = torch.empty((S, H), dtype=torch.float8_e4m3fn, device=device)
        sc_s = torch.empty((S, 1), dtype=torch.float32, device=device)
        def _fp8_2call():
            amax = x_s.float().abs().amax(dim=-1, keepdim=True)
            sc_s.copy_((amax * (1.0 / 448.0)).clamp(min=1e-12))
            fp8_s.copy_((x_s.float() / sc_s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn))
            dist.all_gather_into_tensor(ag_fp8.view(torch.uint8), fp8_s.view(torch.uint8))
            dist.all_gather_into_tensor(ag_sc, sc_s)
        med_2c, p90_2c = time_collective(_fp8_2call, warmup=args.warmup, iters=args.iters)
        sq_2c, cs_2c, _, _ = compute_metrics((ag_fp8.float() * ag_sc).to(torch.bfloat16), ag_ref)
        if rank == 0:
            print(f"  {T:5d} | {'fp8_2call_ring':14s} | {'nccl_ring':12s} | {med_2c:7.3f} | {p90_2c:7.3f} | "
                  f"{dir_wire_gb / (med_2c / 1e3):8.2f} | {bf16_wire_gb / (med_2c / 1e3):8.2f} | "
                  f"{med_bf16 / med_2c:6.2f}x | {sq_2c:6.1f}dB | {cs_2c:8.6f} | {'EXACT':>9s}")

        def _check_rank_bit_identical(t: torch.Tensor) -> str:
            raw = t.view(torch.uint8)
            r0 = raw.clone()
            dist.broadcast(r0, src=0)
            nd = torch.tensor([int((raw != r0).sum().item())], dtype=torch.int64, device=device)
            dist.all_reduce(nd, op=dist.ReduceOp.SUM)
            return "EXACT" if int(nd.item()) == 0 else f"DIFF({int(nd.item())})"

        for mname, is_fp8, fp8_dir, wgb in (
            ("rhd_int8_ag", False, False, g64_wire_gb),
            ("rhd_fp8_ag", True, False, g64_wire_gb),
            ("rhd_fp8_dir_ag", True, True, dir_wire_gb),
        ):
            med, p90 = time_collective(
                lambda is_fp8=is_fp8, fp8_dir=fp8_dir: runner.all_gather(
                    x_s, ag_out, ag_fp8, ag_sc, is_fp8, fp8_dir, transports, best_t
                ),
                warmup=args.warmup, iters=args.iters,
            )
            if fp8_dir:
                eval_t = (ag_fp8.float() * ag_sc).to(torch.bfloat16)
                beq = _check_rank_bit_identical(ag_fp8)
            else:
                eval_t = ag_out
                beq = _check_rank_bit_identical(ag_out)
            sq, cs, _, _ = compute_metrics(eval_t, ag_ref)
            if rank == 0:
                print(f"  {T:5d} | {mname:14s} | {best_t:12s} | {med:7.3f} | {p90:7.3f} | "
                      f"{wgb / (med / 1e3):8.2f} | {bf16_wire_gb / (med / 1e3):8.2f} | "
                      f"{med_bf16 / med:6.2f}x | {sq:6.1f}dB | {cs:8.6f} | {beq:>9s}")

    # -----------------------------------------------------------------------
    # Phase 4: Multi-Regime Accuracy & Pipe=0 vs Pipe=1 0-ULP Bit-Identity Gate
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n[Phase 4] Multi-Regime 8-Rank RHD ReduceScatter Accuracy Gate (T=2048, H=4608):")
    T_v = 2048
    S_v = T_v // world
    out_p0 = torch.empty((S_v, H), dtype=torch.bfloat16, device=device)
    out_p1 = torch.empty((S_v, H), dtype=torch.bfloat16, device=device)

    torch.manual_seed(42)
    base_canc = torch.randn((T_v, H), dtype=torch.float32, device=device)
    torch.manual_seed(777 + rank * 13)
    x_norm = (torch.randn((T_v, H), dtype=torch.float32, device=device) * 0.5).to(torch.bfloat16)
    x_clust = x_norm.clone()
    x_clust[:, :16] = (x_clust[:, :16].float() * 100.0).to(torch.bfloat16)
    x_scat = x_norm.clone()
    x_scat[:, ::64] = (x_scat[:, ::64].float() * 100.0).to(torch.bfloat16)
    # Zero-mean random 4x cancellation across ranks
    z_rank = torch.randn((T_v, H), dtype=torch.float32, device=device) * 2.0
    z_mean = z_rank.clone()
    dist.all_reduce(z_mean, op=dist.ReduceOp.SUM)
    z_rank -= z_mean / world
    x_canc = (z_rank + torch.randn((T_v, H), dtype=torch.float32, device=device) * 0.5).to(torch.bfloat16)
    x_tiny = (torch.randn((T_v, H), dtype=torch.float32, device=device) * 1e-7).to(torch.bfloat16)
    x_zero = torch.zeros((T_v, H), dtype=torch.bfloat16, device=device)

    for rname, xin in (
        ("Normal", x_norm),
        ("100x_Clust", x_clust),
        ("100x_Scat", x_scat),
        ("4x_Cancel", x_canc),
        ("Tiny_1e-7", x_tiny),
        ("All_Zero", x_zero),
    ):
        ref_f32 = xin.float().clone()
        dist.all_reduce(ref_f32, op=dist.ReduceOp.SUM)
        ref_s = ref_f32[rank * S_v : (rank + 1) * S_v].to(torch.bfloat16)
        parts = []
        for mname, is_fp8 in (("rhd_int8", False), ("rhd_fp8", True)):
            runner.reduce_scatter(xin, out_p0, is_fp8, transports, best_t, pipe=False)
            if lib is not None and world >= 4:
                runner.reduce_scatter(xin, out_p1, is_fp8, transports, best_t, pipe=True)
                nd = int((out_p0.view(torch.uint16) != out_p1.view(torch.uint16)).sum().item())
                assert nd == 0, f"FAIL {rname}/{mname} pipe=0 vs pipe=1 diff={nd}!"
            sq, cs, me, _ = compute_metrics(out_p0, ref_s)
            if rname == "All_Zero":
                assert me == 0.0, f"All_Zero non-zero error: {me}"
                parts.append(f"{mname}=EXACT_0")
            else:
                parts.append(f"{mname}: {sq:5.2f}dB cos={cs:.6f} max={me:.2e}")
        if rank == 0:
            print(f"  [{rname:12s}] " + " | ".join(parts) + (" | pipe0==pipe1:0-ULP" if lib else ""))

    dist.barrier()
    if rank == 0:
        print("\nALL 8-RANK RHD VERIFICATION & BENCHMARK CHECKS PASSED!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
