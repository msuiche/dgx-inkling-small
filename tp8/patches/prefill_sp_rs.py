# SPDX-License-Identifier: Apache-2.0
"""Prefill-only Reduced-Traffic ReduceScatter for 8x DGX Spark GB10 (SM121a).

Strictly opt-in via `INKLING_PREFILL_SP_RS_MODE` (default `nccl`, ZERO effect on C=1/C=32
decode `T <= 512` or CUDA graph capture!). Eliminates NCCL BF16 Ring ReduceScatter's 7-hop
ring staging (`4.625 * S = 349 MB` LPDDR5X UMA traffic and `132 MB` wire per 16K call) by
quantizing/packing ONLY the 7 remote shards (`7/8 * S`) once, running 1-hop shift-rotated
AllToAll (via `pynccl` grouped P2P `send`/`recv` or optional `torch_a2a`), and accumulating
all 8 ranks in deterministic ascending FP32 rank order `0..7` (keeping the local rank's own
`1/8` shard 100% unquantized BF16!).

Supported compressed modes (`T >= INKLING_PREFILL_SP_RS_MIN_T`, default 2048, floor >= 1024):
  - `fp8_a2a` (Blockwise FP8-E4M3FN, group=128, BF16 scales — 0.5078x wire, 1.0156 B/elem):
      Quantizes `[T, H]` into `[W, S, H]` FP8-E4M3FN + `[W, S, H/128]` BF16 block scales
      (full 8-bit FP32 exponent -> zero scale underflow/overflow!).
      Wire per 16K RS: `132.1 MB -> 67.1 MB/rank` (`-49.2%`).
      Uses zero-JIT native SM121a CUDA (`libpf_fp8_rs_sm121.so`) with seamless Triton fallback.
  - `int8_a2a` (Blockwise symmetric INT8, group=128, BF16 scales — 0.5078x wire, 1.0156 B/elem):
      Exact same 1.0156 B/elem wire layout as `fp8_a2a` with +11.4 dB higher SQNR on normal
      activations (`43.8 dB`, `cos = 0.999979`).
  - `int8_col` (Column-wise symmetric INT8, group=32 tokens — 0.5313x wire, 1.0625 B/elem):
      100% immune to 100x channel-wise outliers (`SQNR = 45.1 dB`, `cos = 0.999985` across
      Normal & 100x Outlier regimes; `+13.6 dB` > FP8 under 4x rank cancellation).
      Uses native SM121a CUDA (supports arbitrary `S`) when `.so` is loaded, else Triton.
  - `bf12` (Split-plane block-exponent `BF12-E4M7` — 0.7656x wire, 1.5313 B/elem):
      Near-0-ULP fidelity (`SQNR = 95-96 dB`, `99.9%+` 0-ULP exact).
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import torch

_TRITON_OK = False
try:
    from vllm.triton_utils import tl, triton

    _TRITON_OK = True
except Exception:
    try:
        import triton
        import triton.language as tl

        _TRITON_OK = True
    except Exception:
        tl = None  # type: ignore[assignment]
        triton = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Opt-in config & SM121a CUDA sidecar loaders (strictly prefill-only!)
# ---------------------------------------------------------------------------
SP_RS_MIN_T = max(int(os.environ.get("INKLING_PREFILL_SP_RS_MIN_T", "2048") or 2048), 1024)
SP_AG_MIN_T = max(int(os.environ.get("INKLING_PREFILL_SP_AG_MIN_T", "2048") or 2048), 1024)
SP_RS_TRANSPORT = os.environ.get("INKLING_PREFILL_SP_RS_TRANSPORT", "pynccl").strip().lower()
if SP_RS_TRANSPORT not in ("pynccl", "torch_a2a"):
    SP_RS_TRANSPORT = "pynccl"
SP_RHD_TRANSPORT = os.environ.get("INKLING_PREFILL_SP_RHD_TRANSPORT", "pynccl_p2p").strip().lower()
if SP_RHD_TRANSPORT not in ("pynccl_p2p", "sliced_p2p", "pair_subcomm", "pair_ag2"):
    SP_RHD_TRANSPORT = "pynccl_p2p"
SP_RHD_SLICES = max(int(os.environ.get("INKLING_PREFILL_SP_RHD_SLICES", "4") or 4), 1)
SP_RHD_PIPE = os.environ.get("INKLING_PREFILL_SP_RHD_PIPE", "0").strip().lower() in (
    "1", "on", "true", "yes",
)
_FP8_RS_LIB_PATH = os.environ.get("INKLING_PF_FP8_RS_LIB", "/rdma/libpf_fp8_rs_sm121.so").strip()
_FP8_RS_LIB: ctypes.CDLL | None = None
_FP8_RS_TRIED = False
_RHD_COMM_LIB_PATH = os.environ.get(
    "INKLING_PF_RHD_COMM_LIB", "/rdma/libpf_rhd_comm_sm121.so"
).strip()
_RHD_COMM_LIB: ctypes.CDLL | None = None
_RHD_COMM_TRIED = False


def _get_fp8_rs_lib() -> ctypes.CDLL | None:
    """Lazily load native SM121a `libpf_fp8_rs_sm121.so` outside CUDA graph capture."""
    global _FP8_RS_LIB, _FP8_RS_TRIED
    if _FP8_RS_TRIED:
        return _FP8_RS_LIB
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    _FP8_RS_TRIED = True
    if _FP8_RS_LIB_PATH and os.path.exists(_FP8_RS_LIB_PATH):
        try:
            lib = ctypes.CDLL(_FP8_RS_LIB_PATH)
            _q_args = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            _d_args = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            for fn_name in (
                "quant_bf16_to_fp8_g128_a2a_fwd",
                "quant_bf16_to_int8_g128_a2a_fwd",
                "quant_bf16_to_int8_col32_a2a_fwd",
            ):
                fn = getattr(lib, fn_name)
                fn.argtypes = _q_args
                fn.restype = ctypes.c_int
            for fn_name in (
                "dequant_sum8_fp8_g128_to_bf16_fwd",
                "dequant_sum8_int8_g128_to_bf16_fwd",
                "dequant_sum8_int8_col32_to_bf16_fwd",
            ):
                fn = getattr(lib, fn_name)
                fn.argtypes = _d_args
                fn.restype = ctypes.c_int
            lib.detect_discard_block_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.detect_discard_block_fwd.restype = ctypes.c_int
            lib.scatter_halo_last3_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.scatter_halo_last3_fwd.restype = ctypes.c_int
            _FP8_RS_LIB = lib
            print(f"[inkling-pf-fp8-rs] loaded native SM121a CUDA kernels: {_FP8_RS_LIB_PATH}", flush=True)
        except Exception as e:
            print(f"[inkling-pf-fp8-rs] CUDA lib load failed ({e!r}); using Triton kernels", flush=True)
            _FP8_RS_LIB = None
    return _FP8_RS_LIB


# ---------------------------------------------------------------------------
# Shift-rotated 1-hop AllToAll over existing PyNccl (zero C++ rebuild!)
# ---------------------------------------------------------------------------
def _tp_all_to_all_peers(send_buf: torch.Tensor, recv_buf: torch.Tensor) -> None:
    """1-hop AllToAll of contiguous `[world_size, nbytes_per_peer]` uint8 buffers on
    current CUDA stream, skipping `peer == rank` (caller keeps local shard in BF16!).

    Uses shift-rotated peer ordering `dst = (rank + d) % W, src = (rank - d) % W`
    so every rank transmits to a distinct destination at every step `d = 1..W-1`
    (prevents 7-to-1 RoCEv2 switch incast on step 0!).
    """
    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    world_size = tp_group.world_size
    rank = tp_group.rank_in_group
    if SP_RS_TRANSPORT == "torch_a2a":
        torch.distributed.all_to_all_single(recv_buf, send_buf, group=tp_group.device_group)
        return
    comm = getattr(tp_group, "device_communicator", None)
    pynccl = getattr(comm, "pynccl_comm", None) if comm is not None else None
    if pynccl is not None and not getattr(pynccl, "disabled", False):
        pynccl.group_start()
        for d in range(1, world_size):
            dst = (rank + d) % world_size
            src = (rank - d) % world_size
            pynccl.send(send_buf[dst], dst)
            pynccl.recv(recv_buf[src], src)
        pynccl.group_end()
        return

    # Fallback via torch.distributed.all_to_all_single (copies self slice too)
    torch.distributed.all_to_all_single(recv_buf, send_buf, group=tp_group.device_group)


# ---------------------------------------------------------------------------
# Zero-Incast 3-Step RHD (XOR Butterfly) Native SM121a Loader & Pairwise Transport
# ---------------------------------------------------------------------------
def _get_rhd_comm_lib() -> ctypes.CDLL | None:
    """Lazily load native SM121a `libpf_rhd_comm_sm121.so` outside CUDA graph capture."""
    global _RHD_COMM_LIB, _RHD_COMM_TRIED
    if _RHD_COMM_TRIED:
        return _RHD_COMM_LIB
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    _RHD_COMM_TRIED = True
    if _RHD_COMM_LIB_PATH and os.path.exists(_RHD_COMM_LIB_PATH):
        try:
            lib = ctypes.CDLL(_RHD_COMM_LIB_PATH)
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
            for fn_name in ("rhd_quant_slice_int8_g64_fwd", "rhd_quant_slice_fp8_g64_fwd"):
                fn = getattr(lib, fn_name)
                fn.argtypes = _quant_args
                fn.restype = ctypes.c_int
            for fn_name in (
                "rhd_dequant_add_split_quant_int8_g64_fwd",
                "rhd_dequant_add_split_quant_fp8_g64_fwd",
            ):
                fn = getattr(lib, fn_name)
                fn.argtypes = _split_args
                fn.restype = ctypes.c_int
            for fn_name in (
                "rhd_dequant_add_quant_int8_g64_fwd",
                "rhd_dequant_add_quant_fp8_g64_fwd",
            ):
                fn = getattr(lib, fn_name)
                fn.argtypes = _daq_args
                fn.restype = ctypes.c_int
            for fn_name in (
                "rhd_dequant_add_bf16_int8_g64_fwd",
                "rhd_dequant_add_bf16_fp8_g64_fwd",
            ):
                fn = getattr(lib, fn_name)
                fn.argtypes = _dab_args
                fn.restype = ctypes.c_int
            for fn_name in ("rhd_ag_dequant_int8_g64_fwd", "rhd_ag_dequant_fp8_g64_fwd"):
                fn = getattr(lib, fn_name)
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
            _RHD_COMM_LIB = lib
            print(
                f"[inkling-pf-rhd-comm] loaded native SM121a CUDA kernels: {_RHD_COMM_LIB_PATH}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[inkling-pf-rhd-comm] CUDA lib load failed ({e!r}); using Triton fallback",
                flush=True,
            )
            _RHD_COMM_LIB = None
    return _RHD_COMM_LIB


# Lazy 2-rank sub-communicator table per (tp_group_id, device_idx):
#   bit -> (pynccl_2rank_comm | None, pg_nccl | None, peer_local_rank, peer_global_rank)
_RHD_PAIR_SUBCOMMS: dict[tuple[int, int], dict[int, tuple[Any, Any, int, int]]] = {}
_RHD_AG2_SCRATCH: dict[int, torch.Tensor] = {}


def _get_rhd_pair_subcomms(
    tp_group: Any, device: torch.device
) -> dict[int, tuple[Any, Any, int, int]]:
    """Create 2-rank XOR-partner sub-communicators (`{r, r^4}`, `{r, r^2}`, `{r, r^1}`)
    in strict deterministic global order across all ranks (outside CUDA graph capture).
    Because `nRanks == 2`, NCCL dedicates ALL 8 channels across ALL 4 ConnectX-7 HCAs
    exclusively to the single pairwise exchange (zero 7-to-1 incast!)."""
    dev_idx = device.index if device.index is not None else 0
    key = (id(tp_group), dev_idx)
    cached = _RHD_PAIR_SUBCOMMS.get(key)
    if cached is not None:
        return cached
    world_size = int(tp_group.world_size)
    rank = int(tp_group.rank_in_group)
    global_ranks = list(tp_group.ranks)
    use_pynccl = False
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        use_pynccl = True
    except Exception:
        PyNcclCommunicator = None  # type: ignore[assignment]

    out: dict[int, tuple[Any, Any, int, int]] = {}
    bit = world_size >> 1
    while bit >= 1:
        for r in range(world_size):
            p = r ^ bit
            if r < p:
                pair_ranks = [global_ranks[r], global_ranks[p]]
                if use_pynccl and PyNcclCommunicator is not None:
                    cpu_sub = torch.distributed.new_group(ranks=pair_ranks, backend="gloo")
                    if rank in (r, p):
                        pcomm = PyNcclCommunicator(group=cpu_sub, device=device)
                        local_rank = 0 if rank == r else 1
                        peer_global = global_ranks[p if rank == r else r]
                        out[bit] = (pcomm, None, 1 - local_rank, peer_global)
                else:
                    nccl_sub = torch.distributed.new_group(ranks=pair_ranks, backend="nccl")
                    if rank in (r, p):
                        local_rank = 0 if rank == r else 1
                        peer_global = global_ranks[p if rank == r else r]
                        out[bit] = (None, nccl_sub, 1 - local_rank, peer_global)
        bit >>= 1
    _RHD_PAIR_SUBCOMMS[key] = out
    return out


def _tp_pair_exchange(
    send_view: torch.Tensor,
    recv_view: torch.Tensor,
    partner: int,
    bit: int,
    transport: str | None = None,
    slices: int | None = None,
) -> None:
    """Zero-incast 1-to-1 bidirectional exchange (`send_view` <-> `recv_view`) with `partner = rank ^ bit`
    on the current CUDA stream. Supports:
      - `pynccl_p2p`: single `pynccl.send/recv` on existing 8-rank TP comm (0 extra comms)
      - `sliced_p2p`: `K` equal chunk `send/recv` calls inside 1 `group_start/end`
      - `pair_subcomm`: dedicated 2-rank NCCL sub-communicator `{r, r^bit}` (all 8 ch / 4 HCAs!)
      - `pair_ag2`: 2-rank NCCL Ring `all_gather` on `{r, r^bit}` (guaranteed Ring 4-HCA BW!)
    """
    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    comm = getattr(tp_group, "device_communicator", None)
    pynccl = getattr(comm, "pynccl_comm", None) if comm is not None else None
    mode = SP_RHD_TRANSPORT if transport is None else transport
    n_slices = SP_RHD_SLICES if slices is None else max(int(slices), 1)
    nbytes = int(send_view.numel())

    if mode in ("pair_subcomm", "pair_ag2"):
        subcomms = _get_rhd_pair_subcomms(tp_group, send_view.device)
        pcomm, pg_nccl, peer_local, peer_global = subcomms[bit]
        if mode == "pair_ag2":
            dev_idx = send_view.device.index if send_view.device.index is not None else 0
            ag2 = _RHD_AG2_SCRATCH.get(dev_idx)
            need = 2 * nbytes
            if ag2 is None or ag2.numel() < need:
                ag2 = torch.empty((need,), dtype=torch.uint8, device=send_view.device)
                _RHD_AG2_SCRATCH[dev_idx] = ag2
            ag2_view = ag2[:need].view(2, nbytes)
            ag2_view.record_stream(torch.cuda.current_stream(send_view.device))
            if pcomm is not None and not getattr(pcomm, "disabled", False):
                pcomm.all_gather(ag2_view, send_view)
            else:
                torch.distributed.all_gather_into_tensor(ag2_view, send_view, group=pg_nccl)
            recv_view.copy_(ag2_view[peer_local])
            return
        if pcomm is not None and not getattr(pcomm, "disabled", False):
            pcomm.group_start()
            pcomm.send(send_view, peer_local)
            pcomm.recv(recv_view, peer_local)
            pcomm.group_end()
            return
        ops = [
            torch.distributed.P2POp(
                torch.distributed.isend, send_view, peer_global, group=pg_nccl
            ),
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_view, peer_global, group=pg_nccl
            ),
        ]
        for w in torch.distributed.batch_isend_irecv(ops):
            w.wait()
        return

    if pynccl is not None and not getattr(pynccl, "disabled", False):
        if mode == "sliced_p2p" and n_slices > 1 and nbytes >= n_slices * 4096:
            chunk = (nbytes // n_slices) & ~127
            if chunk > 0:
                pynccl.group_start()
                off = 0
                for k in range(n_slices):
                    end = nbytes if k == (n_slices - 1) else (off + chunk)
                    pynccl.send(send_view[off:end], partner)
                    pynccl.recv(recv_view[off:end], partner)
                    off = end
                pynccl.group_end()
                return
        pynccl.group_start()
        pynccl.send(send_view, partner)
        pynccl.recv(recv_view, partner)
        pynccl.group_end()
        return

    # Fallback via torch.distributed.batch_isend_irecv on TP device_group
    peer_global = int(tp_group.ranks[partner])
    ops = [
        torch.distributed.P2POp(
            torch.distributed.isend, send_view, peer_global, group=tp_group.device_group
        ),
        torch.distributed.P2POp(
            torch.distributed.irecv, recv_view, peer_global, group=tp_group.device_group
        ),
    ]
    for w in torch.distributed.batch_isend_irecv(ops):
        w.wait()


# ---------------------------------------------------------------------------
# Triton Kernels: Blockwise FP8-E4M3 / INT8 g128 + INT8-col32 + BF12 + Zero-Sync SP
# ---------------------------------------------------------------------------
if _TRITON_OK:

    @triton.jit
    def _quant_g128_a2a_kernel(
        X_PTR,        # [T, H] bf16
        OUT_PTR,      # [W, peer_bytes] uint8
        stride_xm,    # H
        peer_bytes,   # align128(S * H + 2 * S * (H // 128))
        scale_off,    # S * H
        S,            # T // W
        H,            # hidden_size (multiple of 128)
        RANK: tl.constexpr,
        NG: tl.constexpr,      # groups of 128 per CTA (4 -> 512 cols)
        IS_FP8: tl.constexpr,  # True -> FP8-E4M3FN, False -> symmetric INT8
    ):
        """Quantize remote peers' `[S, H]` chunks into `[S, H]` FP8/INT8 + `[S, H//128]` BF16 scales.
        Skips `dst == RANK` completely (local shard stays 100% unquantized exact BF16!)."""
        pid_p = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_g = tl.program_id(2)

        dst = pid_p if pid_p < RANK else pid_p + 1
        G = H // 128
        g_offs = pid_g * NG + tl.arange(0, NG)
        g_mask = g_offs < G
        c_offs = tl.arange(0, 128)
        cols = g_offs[:, None] * 128 + c_offs[None, :]  # [NG, 128]

        src_row = (dst * S + pid_r).to(tl.int64)
        x = tl.load(
            X_PTR + src_row * stride_xm + cols,
            mask=g_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        amax = tl.max(tl.abs(x), axis=1)  # [NG]
        inv_qmax = (1.0 / 448.0) if IS_FP8 else (1.0 / 127.0)
        sc_bf = tl.maximum(amax * inv_qmax, 1e-12).to(tl.bfloat16)
        inv_sc = 1.0 / sc_bf.to(tl.float32)

        scaled = x * inv_sc[:, None]
        if IS_FP8:
            q_u8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            rnd = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
            q_u8 = tl.clamp(rnd, -127.0, 127.0).to(tl.int8).to(tl.uint8, bitcast=True)

        peer_base = dst.to(tl.int64) * peer_bytes
        tl.store(
            OUT_PTR + peer_base + pid_r.to(tl.int64) * H + cols,
            q_u8,
            mask=g_mask[:, None],
        )
        sc_ptr = (OUT_PTR + peer_base + scale_off).to(tl.pointer_type(tl.uint16))
        tl.store(
            sc_ptr + pid_r.to(tl.int64) * G + g_offs,
            sc_bf.to(tl.uint16, bitcast=True),
            mask=g_mask,
        )

    @triton.jit
    def _dequant_sum_g128_kernel(
        RECV_PTR,     # [W, peer_bytes] uint8
        SELF_X_PTR,   # [T, H] bf16 (only rows [RANK*S : (RANK+1)*S) are read!)
        OUT_PTR,      # [S, H] bf16
        stride_xm,
        stride_om,
        peer_bytes,
        scale_off,
        S,
        H,
        RANK: tl.constexpr,
        WORLD: tl.constexpr,
        NG: tl.constexpr,
        IS_FP8: tl.constexpr,
    ):
        """Dequantize 7 remote FP8/INT8-g128 peer shards + local unquantized BF16 shard,
        accumulating in deterministic ascending FP32 rank order `0..WORLD-1` -> BF16 store."""
        pid_r = tl.program_id(0)
        pid_g = tl.program_id(1)

        G = H // 128
        g_offs = pid_g * NG + tl.arange(0, NG)
        g_mask = g_offs < G
        c_offs = tl.arange(0, 128)
        cols = g_offs[:, None] * 128 + c_offs[None, :]  # [NG, 128]

        acc = tl.zeros((NG, 128), dtype=tl.float32)
        row_i64 = pid_r.to(tl.int64)

        for r in tl.static_range(WORLD):
            if r == RANK:
                s_row = (RANK * S + pid_r).to(tl.int64)
                x_self = tl.load(
                    SELF_X_PTR + s_row * stride_xm + cols,
                    mask=g_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                acc = acc + x_self
            else:
                p_base = r * peer_bytes
                sc_ptr = (RECV_PTR + p_base + scale_off).to(tl.pointer_type(tl.uint16))
                sc = (
                    tl.load(sc_ptr + row_i64 * G + g_offs, mask=g_mask, other=0)
                    .to(tl.bfloat16, bitcast=True)
                    .to(tl.float32)
                )
                q_u8 = tl.load(
                    RECV_PTR + p_base + row_i64 * H + cols,
                    mask=g_mask[:, None],
                    other=0,
                )
                if IS_FP8:
                    q_f = q_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                else:
                    q_f = q_u8.to(tl.int8, bitcast=True).to(tl.float32)
                acc = acc + q_f * sc[:, None]

        tl.store(
            OUT_PTR + row_i64 * stride_om + cols,
            acc.to(tl.bfloat16),
            mask=g_mask[:, None],
        )

    @triton.jit
    def _quant_int8_col32_kernel(
        X_PTR,        # [T, H] bf16
        OUT_PTR,      # [W, peer_bytes] uint8
        stride_xm,    # H
        peer_bytes,   # S * H + 2 * (S // 32) * H
        scale_off,    # S * H (byte offset to fp16 scales inside each peer slice)
        S,            # T // W (multiple of 32)
        H,            # hidden_size (multiple of BH)
        RANK: tl.constexpr,
        BH: tl.constexpr,
    ):
        """Quantize remote peers' `[S, H]` chunks into `[S, H]` int8 + `[S//32, H]` fp16 scales.
        Skips `dst == RANK` completely (local shard stays 100% unquantized BF16!).
        """
        pid_p = tl.program_id(0)
        pid_g = tl.program_id(1)
        pid_h = tl.program_id(2)

        dst = pid_p if pid_p < RANK else pid_p + 1
        h_offs = pid_h * BH + tl.arange(0, BH)
        r_offs = tl.arange(0, 32)
        h_mask = h_offs < H

        row_base = (dst * S + pid_g * 32).to(tl.int64)
        x_ptrs = X_PTR + (row_base + r_offs[:, None]) * stride_xm + h_offs[None, :]
        x = tl.load(x_ptrs, mask=h_mask[None, :], other=0.0).to(tl.float32)

        amax = tl.max(tl.abs(x), axis=0)  # [BH]
        sc_f16 = tl.maximum(amax * (1.0 / 127.0), 1e-7).to(tl.float16)
        inv_sc = 1.0 / sc_f16.to(tl.float32)

        scaled = x * inv_sc[None, :]
        # Portable round-to-nearest (ties away from zero, matches __float2int_rn closely)
        rnd = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
        q = tl.clamp(rnd, -127.0, 127.0).to(tl.int8)

        peer_base = dst.to(tl.int64) * peer_bytes
        q_base = OUT_PTR.to(tl.pointer_type(tl.int8)) + peer_base
        q_ptrs = q_base + (pid_g * 32 + r_offs[:, None]).to(tl.int64) * H + h_offs[None, :]
        tl.store(q_ptrs, q, mask=h_mask[None, :])

        sc_base = (OUT_PTR + peer_base + scale_off).to(tl.pointer_type(tl.float16))
        sc_ptrs = sc_base + pid_g.to(tl.int64) * H + h_offs
        tl.store(sc_ptrs, sc_f16, mask=h_mask)

    @triton.jit
    def _dequant_reduce_int8_col32_kernel(
        RECV_PTR,     # [W, peer_bytes] uint8
        SELF_X_PTR,   # [T, H] bf16 (only rows [RANK*S : (RANK+1)*S) are read!)
        OUT_PTR,      # [S, H] bf16
        stride_xm,
        stride_om,
        peer_bytes,
        scale_off,
        S,
        H,
        RANK: tl.constexpr,
        WORLD: tl.constexpr,
        BH: tl.constexpr,
    ):
        """Dequantize 7 remote peers' INT8-g32 chunks + local unquantized BF16 shard,
        accumulating in fixed FP32 rank order `0..WORLD-1` -> single BF16 RN store."""
        pid_g = tl.program_id(0)
        pid_h = tl.program_id(1)

        h_offs = pid_h * BH + tl.arange(0, BH)
        r_offs = tl.arange(0, 32)
        h_mask = h_offs < H

        acc = tl.zeros((32, BH), dtype=tl.float32)
        row_local = (pid_g * 32 + r_offs[:, None]).to(tl.int64)

        for r in tl.static_range(WORLD):
            if r == RANK:
                s_row = (RANK * S + pid_g * 32 + r_offs[:, None]).to(tl.int64)
                x_self = tl.load(
                    SELF_X_PTR + s_row * stride_xm + h_offs[None, :],
                    mask=h_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc = acc + x_self
            else:
                p_base = r * peer_bytes
                sc_ptr = (RECV_PTR + p_base + scale_off).to(tl.pointer_type(tl.float16))
                sc = tl.load(sc_ptr + pid_g.to(tl.int64) * H + h_offs, mask=h_mask, other=0.0).to(
                    tl.float32
                )
                q_ptr = (RECV_PTR + p_base).to(tl.pointer_type(tl.int8))
                q = tl.load(
                    q_ptr + row_local * H + h_offs[None, :],
                    mask=h_mask[None, :],
                    other=0,
                ).to(tl.float32)
                acc = acc + q * sc[None, :]

        out_ptrs = OUT_PTR + row_local * stride_om + h_offs[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=h_mask[None, :])

    # -----------------------------------------------------------------------
    # Mode 2: BF12-E4M7 Split-Plane Block-Exponent ReduceScatter (12.25 b/e, 1.31x)
    # -----------------------------------------------------------------------
    @triton.jit
    def _pack_bf12_e4m7_kernel(
        X_PTR,        # [T, H] bf16 (viewed as uint16)
        OUT_PTR,      # [W, peer_bytes] uint8
        stride_xm,    # H
        peer_bytes,   # S * H + S * (H // 2) + S * (H // 32)
        nib_off,      # S * H
        emax_off,     # S * H + S * (H // 2)
        S,
        H,            # multiple of 32
        RANK: tl.constexpr,
        GROUPS_PER_CTA: tl.constexpr,  # e.g. 4 groups of 32 = 128 elems
    ):
        """Pack `[S, H]` bf16 into split-plane `BF12-E4M7`:
          - Plane 0 (`[S, H]` uint8): `(sign << 7) | mant7`
          - Plane 1 (`[S, H//2]` uint8): packed 4-bit exponent deltas `(d_even | (d_odd << 4))`
            where `d = E_max - exp` (`0..14` = exact 0-ULP!, `15` = zero sentinel)
          - Plane 2 (`[S, H//32]` uint8): per-32-element `E_max`
        """
        pid_p = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_g = tl.program_id(2)

        dst = pid_p if pid_p < RANK else pid_p + 1
        g_base = pid_g * GROUPS_PER_CTA
        num_g = H // 32
        x_u32_ptr = X_PTR.to(tl.pointer_type(tl.uint32))
        src_row_u32 = (dst * S + pid_r).to(tl.int64) * (stride_xm // 2)

        p_base = dst.to(tl.int64) * peer_bytes
        lo_u16_row = (OUT_PTR + p_base + pid_r.to(tl.int64) * H).to(tl.pointer_type(tl.uint16))
        nib_row = OUT_PTR + p_base + nib_off + pid_r.to(tl.int64) * (H // 2)
        em_row = OUT_PTR + p_base + emax_off + pid_r.to(tl.int64) * num_g

        offs16 = tl.arange(0, 16)
        for gi in tl.static_range(GROUPS_PER_CTA):
            g = g_base + gi
            if g < num_g:
                pair_idx = g * 16 + offs16
                # Single coalesced 64B load (16 x uint32 = 32 x bf16)
                u32 = tl.load(x_u32_ptr + src_row_u32 + pair_idx)
                u_even = (u32 & 0xFFFF).to(tl.uint16)
                u_odd = (u32 >> 16).to(tl.uint16)

                e_even = ((u_even >> 7) & 0xFF).to(tl.int32)
                e_odd = ((u_odd >> 7) & 0xFF).to(tl.int32)
                emax = tl.maximum(tl.max(e_even, axis=0), tl.max(e_odd, axis=0))

                d_even = emax - e_even
                d_odd = emax - e_odd
                k_even = (e_even > 0) & (d_even <= 14)
                k_odd = (e_odd > 0) & (d_odd <= 14)

                code_even = tl.where(k_even, d_even, 15).to(tl.uint8)
                code_odd = tl.where(k_odd, d_odd, 15).to(tl.uint8)
                nib = code_even | (code_odd << 4)

                lo_even = tl.where(
                    k_even, ((u_even >> 8) & 0x80) | (u_even & 0x7F), 0
                ).to(tl.uint16)
                lo_odd = tl.where(
                    k_odd, ((u_odd >> 8) & 0x80) | (u_odd & 0x7F), 0
                ).to(tl.uint16)
                lo_pair = lo_even | (lo_odd << 8)

                # Single coalesced 32B store for Plane 0 + 16B store for Plane 1 + 1B store for Plane 2
                tl.store(lo_u16_row + pair_idx, lo_pair)
                tl.store(nib_row + pair_idx, nib)
                tl.store(em_row + g, emax.to(tl.uint8))

    @triton.jit
    def _unpack_reduce_bf12_e4m7_kernel(
        RECV_PTR,     # [W, peer_bytes] uint8
        SELF_X_PTR,   # [T, H] bf16
        OUT_PTR,      # [S, H] bf16
        stride_xm,
        stride_om,
        peer_bytes,
        nib_off,
        emax_off,
        S,
        H,
        RANK: tl.constexpr,
        WORLD: tl.constexpr,
        GROUPS_PER_CTA: tl.constexpr,
    ):
        """Unpack 7 remote `BF12-E4M7` chunks + local exact BF16 shard, sum in fixed FP32
        rank order `0..WORLD-1`, and store BF16 directly into `out_shard`."""
        pid_r = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_base = pid_g * GROUPS_PER_CTA
        num_g = H // 32
        offs16 = tl.arange(0, 16)
        self_u32_ptr = SELF_X_PTR.to(tl.pointer_type(tl.uint32))
        out_u32_ptr = OUT_PTR.to(tl.pointer_type(tl.uint32))
        s_row_u32 = (RANK * S + pid_r).to(tl.int64) * (stride_xm // 2)
        o_row_u32 = pid_r.to(tl.int64) * (stride_om // 2)

        for gi in tl.static_range(GROUPS_PER_CTA):
            g = g_base + gi
            if g < num_g:
                pair_idx = g * 16 + offs16
                acc_e = tl.zeros((16,), dtype=tl.float32)
                acc_o = tl.zeros((16,), dtype=tl.float32)

                for r in tl.static_range(WORLD):
                    if r == RANK:
                        u32 = tl.load(self_u32_ptr + s_row_u32 + pair_idx)
                        xe = ((u32 & 0xFFFF) << 16).to(tl.float32, bitcast=True)
                        xo = ((u32 >> 16) << 16).to(tl.float32, bitcast=True)
                        acc_e = acc_e + xe
                        acc_o = acc_o + xo
                    else:
                        p_base = r * peer_bytes
                        lo_u16_row = (RECV_PTR + p_base + pid_r.to(tl.int64) * H).to(
                            tl.pointer_type(tl.uint16)
                        )
                        nib_row = RECV_PTR + p_base + nib_off + pid_r.to(tl.int64) * (H // 2)
                        em_ptr = RECV_PTR + p_base + emax_off + pid_r.to(tl.int64) * num_g + g

                        em = tl.load(em_ptr).to(tl.uint32)
                        nib = tl.load(nib_row + pair_idx).to(tl.uint32)
                        lo_pair = tl.load(lo_u16_row + pair_idx).to(tl.uint32)
                        lo_e = lo_pair & 0xFF
                        lo_o = (lo_pair >> 8) & 0xFF

                        d_e = nib & 0xF
                        d_o = (nib >> 4) & 0xF
                        z32 = tl.zeros((16,), dtype=tl.uint32)
                        u16_e = tl.where(
                            d_e == 15,
                            z32,
                            ((lo_e & 0x80) << 8) | ((em - d_e) << 7) | (lo_e & 0x7F),
                        )
                        u16_o = tl.where(
                            d_o == 15,
                            z32,
                            ((lo_o & 0x80) << 8) | ((em - d_o) << 7) | (lo_o & 0x7F),
                        )
                        fe = (u16_e << 16).to(tl.float32, bitcast=True)
                        fo = (u16_o << 16).to(tl.float32, bitcast=True)
                        acc_e = acc_e + fe
                        acc_o = acc_o + fo

                be = acc_e.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
                bo = acc_o.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
                tl.store(out_u32_ptr + o_row_u32 + pair_idx, be | (bo << 16))

    # -----------------------------------------------------------------------
    # Zero-Sync GPU Helpers (Triton fallback when libpf_fp8_rs_sm121.so absent)
    # -----------------------------------------------------------------------
    @triton.jit
    def _detect_discard_block_triton_kernel(
        SLOTS_PTR,
        OUT_PTR,
        T,
        N_CACHE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Single-CTA GPU-only sparse _discard_block detector (ZERO DtoH sync!)."""
        cand = tl.full((), -1, dtype=tl.int64)
        offs = tl.arange(0, BLOCK)
        for base in range(0, T, BLOCK):
            idx = base + offs
            m = (idx + N_CACHE) < T
            s0 = tl.load(SLOTS_PTR + idx, mask=m, other=-1).to(tl.int64)
            s1 = tl.load(SLOTS_PTR + idx + N_CACHE, mask=m, other=-1).to(tl.int64)
            b0 = s0 // N_CACHE
            b1 = s1 // N_CACHE
            hit = m & (s0 >= 0) & (s1 >= 0) & (b0 == b1)
            hit_b = tl.where(hit, b0, -1)
            mx = tl.max(hit_b, axis=0)
            cand = tl.where((cand < 0) & (mx >= 0), mx, cand)
        if cand < 0:
            tl.store(OUT_PTR, tl.full((), -1, dtype=tl.int64))
            return
        cnt = tl.full((), 0, dtype=tl.int32)
        for base in range(0, T, BLOCK):
            idx = base + offs
            m = idx < T
            si = tl.load(SLOTS_PTR + idx, mask=m, other=-1).to(tl.int64)
            eq = m & (si >= 0) & ((si // N_CACHE) == cand)
            cnt = cnt + tl.sum(eq.to(tl.int32), axis=0)
        tl.store(OUT_PTR, tl.where(cnt > N_CACHE, cand, -1).to(tl.int64))

    @triton.jit
    def _scatter_halo_last3_triton_kernel(
        SRC_PTR,
        SLOTS_PTR,
        DISCARD_PTR,
        CACHE_PTR,
        stride_src,
        stride_c_blk,
        stride_c_n,
        stream_offset,
        D,
        N_CACHE: tl.constexpr,
        HAS_DISCARD: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Zero-sync GPU scatter for rank-0 halo tail rows (skips slot < 0 and discard_blk)."""
        r = tl.program_id(0)
        slot = tl.load(SLOTS_PTR + r).to(tl.int64)
        if slot < 0:
            return
        blk = slot // N_CACHE
        if HAS_DISCARD:
            dblk = tl.load(DISCARD_PTR).to(tl.int64)
            if dblk >= 0:
                if blk == dblk:
                    return
        off = slot % N_CACHE
        s_row = SRC_PTR + r.to(tl.int64) * stride_src
        d_row = CACHE_PTR + blk * stride_c_blk + off * stride_c_n + stream_offset
        for d0 in range(0, D, BLOCK_D):
            dd = d0 + tl.arange(0, BLOCK_D)
            m = dd < D
            v = tl.load(s_row + dd, mask=m, other=0.0)
            tl.store(d_row + dd, v, mask=m)

    # -----------------------------------------------------------------------
    # Zero-Incast 3-Step RHD Triton Fallbacks (used only if libpf_rhd_comm_sm121.so absent)
    # -----------------------------------------------------------------------
    @triton.jit
    def _rhd_quant_slice_g64_triton(
        X_PTR,
        OUT_PTR,
        stride_xm,
        row_off,
        scale_off,
        H,
        NG: tl.constexpr,      # groups of 64 per CTA (8 -> 512 cols)
        IS_FP8: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_g = tl.program_id(1)
        G = H // 64
        g_offs = pid_g * NG + tl.arange(0, NG)
        g_mask = g_offs < G
        c_offs = tl.arange(0, 64)
        cols = g_offs[:, None] * 64 + c_offs[None, :]

        src_row = (row_off + row).to(tl.int64)
        x = tl.load(X_PTR + src_row * stride_xm + cols, mask=g_mask[:, None], other=0.0).to(
            tl.float32
        )
        amax = tl.max(tl.abs(x), axis=1)
        inv_qmax = (1.0 / 448.0) if IS_FP8 else (1.0 / 127.0)
        sc_bf = tl.maximum(amax * inv_qmax, 1e-12).to(tl.bfloat16)
        inv_sc = 1.0 / sc_bf.to(tl.float32)

        scaled = x * inv_sc[:, None]
        if IS_FP8:
            q_u8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            rnd = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
            q_u8 = tl.clamp(rnd, -127.0, 127.0).to(tl.int8).to(tl.uint8, bitcast=True)

        row_i64 = row.to(tl.int64)
        tl.store(OUT_PTR + row_i64 * H + cols, q_u8, mask=g_mask[:, None])
        sc_ptr = (OUT_PTR + scale_off).to(tl.pointer_type(tl.uint16))
        tl.store(
            sc_ptr + row_i64 * G + g_offs,
            sc_bf.to(tl.uint16, bitcast=True),
            mask=g_mask,
        )

    @triton.jit
    def _rhd_dequant_add_split_quant_g64_triton(
        RECV_PTR,
        LOCAL_PTR,
        NEXT_SEND_PTR,
        NEXT_KEEP_PTR,
        stride_local,
        local_row_off,
        recv_scale_off,
        next_scale_off,
        stride_keep,
        N,
        H,
        SEND_UPPER: tl.constexpr,
        NG: tl.constexpr,
        IS_FP8: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_g = tl.program_id(1)
        G = H // 64
        g_offs = pid_g * NG + tl.arange(0, NG)
        g_mask = g_offs < G
        c_offs = tl.arange(0, 64)
        cols = g_offs[:, None] * 64 + c_offs[None, :]

        row_i64 = row.to(tl.int64)
        sc_in_ptr = (RECV_PTR + recv_scale_off).to(tl.pointer_type(tl.uint16))
        rsc = (
            tl.load(sc_in_ptr + row_i64 * G + g_offs, mask=g_mask, other=0)
            .to(tl.bfloat16, bitcast=True)
            .to(tl.float32)
        )
        q_in = tl.load(RECV_PTR + row_i64 * H + cols, mask=g_mask[:, None], other=0)
        if IS_FP8:
            rf = q_in.to(tl.float8e4nv, bitcast=True).to(tl.float32) * rsc[:, None]
        else:
            rf = q_in.to(tl.int8, bitcast=True).to(tl.float32) * rsc[:, None]
        loc = tl.load(
            LOCAL_PTR + (local_row_off + row_i64) * stride_local + cols,
            mask=g_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        s = loc + rf

        is_send = (row >= N) if SEND_UPPER else (row < N)
        if is_send:
            out_r = (row - N if SEND_UPPER else row).to(tl.int64)
            amax = tl.max(tl.abs(s), axis=1)
            inv_qmax = (1.0 / 448.0) if IS_FP8 else (1.0 / 127.0)
            sc_out = tl.maximum(amax * inv_qmax, 1e-12).to(tl.bfloat16)
            inv_sc = 1.0 / sc_out.to(tl.float32)
            scaled = s * inv_sc[:, None]
            if IS_FP8:
                q_out = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv).to(
                    tl.uint8, bitcast=True
                )
            else:
                rnd = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
                q_out = tl.clamp(rnd, -127.0, 127.0).to(tl.int8).to(tl.uint8, bitcast=True)
            tl.store(NEXT_SEND_PTR + out_r * H + cols, q_out, mask=g_mask[:, None])
            sc_out_ptr = (NEXT_SEND_PTR + next_scale_off).to(tl.pointer_type(tl.uint16))
            tl.store(
                sc_out_ptr + out_r * G + g_offs,
                sc_out.to(tl.uint16, bitcast=True),
                mask=g_mask,
            )
        else:
            out_r = (row if SEND_UPPER else row - N).to(tl.int64)
            tl.store(
                NEXT_KEEP_PTR + out_r * stride_keep + cols,
                s.to(tl.bfloat16),
                mask=g_mask[:, None],
            )

    @triton.jit
    def _rhd_dequant_add_bf16_g64_triton(
        RECV_PTR,
        LOCAL_PTR,
        OUT_PTR,
        stride_local,
        local_row_off,
        recv_scale_off,
        stride_out,
        H,
        NG: tl.constexpr,
        IS_FP8: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_g = tl.program_id(1)
        G = H // 64
        g_offs = pid_g * NG + tl.arange(0, NG)
        g_mask = g_offs < G
        c_offs = tl.arange(0, 64)
        cols = g_offs[:, None] * 64 + c_offs[None, :]
        row_i64 = row.to(tl.int64)

        sc_in_ptr = (RECV_PTR + recv_scale_off).to(tl.pointer_type(tl.uint16))
        rsc = (
            tl.load(sc_in_ptr + row_i64 * G + g_offs, mask=g_mask, other=0)
            .to(tl.bfloat16, bitcast=True)
            .to(tl.float32)
        )
        q_in = tl.load(RECV_PTR + row_i64 * H + cols, mask=g_mask[:, None], other=0)
        if IS_FP8:
            rf = q_in.to(tl.float8e4nv, bitcast=True).to(tl.float32) * rsc[:, None]
        else:
            rf = q_in.to(tl.int8, bitcast=True).to(tl.float32) * rsc[:, None]
        loc = tl.load(
            LOCAL_PTR + (local_row_off + row_i64) * stride_local + cols,
            mask=g_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            OUT_PTR + row_i64 * stride_out + cols,
            (loc + rf).to(tl.bfloat16),
            mask=g_mask[:, None],
        )

    @triton.jit
    def _rhd_ag_dequant_g64_triton(
        AG_BUF_PTR,
        OUT_PTR,
        shard_bytes,
        scale_off,
        stride_out,
        S,
        H,
        NG: tl.constexpr,
        IS_FP8: tl.constexpr,
    ):
        local_row = tl.program_id(0)
        r = tl.program_id(1)
        pid_g = tl.program_id(2)
        G = H // 64
        g_offs = pid_g * NG + tl.arange(0, NG)
        g_mask = g_offs < G
        c_offs = tl.arange(0, 64)
        cols = g_offs[:, None] * 64 + c_offs[None, :]

        slot_base = r.to(tl.int64) * shard_bytes
        lr_i64 = local_row.to(tl.int64)
        sc_ptr = (AG_BUF_PTR + slot_base + scale_off).to(tl.pointer_type(tl.uint16))
        sc = (
            tl.load(sc_ptr + lr_i64 * G + g_offs, mask=g_mask, other=0)
            .to(tl.bfloat16, bitcast=True)
            .to(tl.float32)
        )
        q_u8 = tl.load(AG_BUF_PTR + slot_base + lr_i64 * H + cols, mask=g_mask[:, None], other=0)
        if IS_FP8:
            vals = q_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * sc[:, None]
        else:
            vals = q_u8.to(tl.int8, bitcast=True).to(tl.float32) * sc[:, None]
        dst_row = (r * S + local_row).to(tl.int64)
        tl.store(OUT_PTR + dst_row * stride_out + cols, vals.to(tl.bfloat16), mask=g_mask[:, None])


# ---------------------------------------------------------------------------
# Persistent scratch buffer cache (1D flat high-water -> always contiguous!)
# ---------------------------------------------------------------------------
_RS_SCRATCH: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_RHD_RS_SCRATCH: dict[int, dict[str, Any]] = {}
_RHD_AG_SCRATCH: dict[tuple[int, str], torch.Tensor] = {}


def _rhd_g64_packed_bytes(num_rows: int, hidden: int) -> tuple[int, int]:
    """Return `(scale_off, total_packed_bytes)` (128B-aligned) for `[num_rows, hidden]` g64 + BF16."""
    scale_off = num_rows * hidden
    g = hidden // 64
    total = (scale_off + 2 * num_rows * g + 127) & ~127
    return scale_off, total


def _rhd_rowwise_fp8_shard_bytes(num_rows: int, hidden: int) -> tuple[int, int]:
    """Return `(scale_off, shard_bytes)` (128B-aligned) for `[num_rows, hidden]` FP8 + FP32 scale."""
    scale_off = num_rows * hidden
    total = (scale_off + 4 * num_rows + 127) & ~127
    return scale_off, total


def _get_rhd_rs_scratch(
    device: torch.device, max_half_bytes: int, max_keep_elems: int, need_pipe: bool
) -> dict[str, Any]:
    """High-water persistent scratch pool for 3-step RHD ReduceScatter (zero per-call malloc)."""
    dev_idx = device.index if device.index is not None else 0
    st = _RHD_RS_SCRATCH.get(dev_idx)
    if st is None:
        st = {}
        _RHD_RS_SCRATCH[dev_idx] = st
    for name in ("send0", "recv0", "send1", "recv1"):
        buf = st.get(name)
        need = max_half_bytes if name.endswith("0") else max(max_half_bytes // 2 + 128, 256)
        if buf is None or buf.numel() < need:
            buf = torch.empty((need,), dtype=torch.uint8, device=device)
            st[name] = buf
    for name, mult in (("keep1", 1), ("keep2", 1)):
        buf = st.get(name)
        need = max_keep_elems if name == "keep1" else max(max_keep_elems // 2, 1)
        if buf is None or buf.numel() < need:
            buf = torch.empty((need,), dtype=torch.bfloat16, device=device)
            st[name] = buf
    if need_pipe and "codec_stream" not in st:
        st["codec_stream"] = torch.cuda.Stream(device=device, priority=-1)
        st["ev_start"] = torch.cuda.Event()
        st["ev_qA"] = torch.cuda.Event()
        st["ev_qB"] = torch.cuda.Event()
        st["ev_cA"] = torch.cuda.Event()
        st["ev_cB"] = torch.cuda.Event()
        st["ev_s1"] = torch.cuda.Event()
        st["ev_k1"] = torch.cuda.Event()
    cur = torch.cuda.current_stream(device)
    for name in ("send0", "recv0", "send1", "recv1", "keep1", "keep2"):
        st[name].record_stream(cur)
    if need_pipe:
        cs = st["codec_stream"]
        for name in ("send0", "recv0", "send1", "keep1"):
            st[name].record_stream(cs)
    return st


def _get_rhd_ag_scratch(
    device: torch.device, world_size: int, shard_bytes: int, tag: str
) -> torch.Tensor:
    dev_idx = device.index if device.index is not None else 0
    total = world_size * shard_bytes
    key = (dev_idx, tag)
    flat = _RHD_AG_SCRATCH.get(key)
    if flat is None or flat.numel() < total:
        flat = torch.empty((total,), dtype=torch.uint8, device=device)
        _RHD_AG_SCRATCH[key] = flat
    buf = flat[:total].view(world_size, shard_bytes)
    buf.record_stream(torch.cuda.current_stream(device))
    return buf


def _get_scratch_pair(
    device: torch.device, world_size: int, peer_bytes: int, tag: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dev_idx = device.index if device.index is not None else 0
    total = world_size * peer_bytes
    key = (dev_idx, tag)
    pair = _RS_SCRATCH.get(key)
    if pair is None or pair[0].numel() < total:
        s_flat = torch.empty((total,), dtype=torch.uint8, device=device)
        r_flat = torch.empty((total,), dtype=torch.uint8, device=device)
        _RS_SCRATCH[key] = (s_flat, r_flat)
    else:
        s_flat, r_flat = pair
    sbuf = s_flat[:total].view(world_size, peer_bytes)
    rbuf = r_flat[:total].view(world_size, peer_bytes)
    cur = torch.cuda.current_stream(device)
    sbuf.record_stream(cur)
    rbuf.record_stream(cur)
    return sbuf, rbuf


# ---------------------------------------------------------------------------
# Zero-Sync GPU SP Helpers (Public)
# ---------------------------------------------------------------------------
def detect_discard_block_gpu(slot_mapping: torch.Tensor, n_cache: int) -> torch.Tensor:
    """Detect sparse write-only `_discard_block` ID strictly on GPU (ZERO DtoH sync!).
    Returns a 1-element int64 GPU tensor containing `discard_blk` (>= 0) or `-1`."""
    out = torch.full((1,), -1, dtype=torch.int64, device=slot_mapping.device)
    T = int(slot_mapping.numel())
    if T <= n_cache or not slot_mapping.is_cuda:
        return out
    sm = slot_mapping.to(torch.int64).contiguous()
    lib = _get_fp8_rs_lib()
    if lib is not None:
        stream = int(torch.cuda.current_stream(sm.device).cuda_stream)
        rc = lib.detect_discard_block_fwd(sm.data_ptr(), out.data_ptr(), T, int(n_cache), stream)
        if rc == 0:
            return out
    if _TRITON_OK:
        _detect_discard_block_triton_kernel[(1,)](
            sm, out, T, N_CACHE=int(n_cache), BLOCK=512, num_warps=4,
        )
    return out


def scatter_halo_last3_gpu(
    src_rows: torch.Tensor,
    last_slots: torch.Tensor,
    discard_blk_dev: torch.Tensor | None,
    cache: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    n_cache: int,
) -> bool:
    """Zero-sync GPU scatter of `src_rows [n_rows, D]` into `cache` (skipping `slot < 0`
    and `blk == *discard_blk_dev`)."""
    n_rows = int(last_slots.numel())
    if n_rows <= 0 or not src_rows.is_cuda or src_rows.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        return False
    if cache.shape[1] != 1 or cache.stride(3) != 1 or src_rows.stride(1) != 1:
        return False
    src_c = src_rows.contiguous()
    slots_c = last_slots.to(torch.int64).contiguous()
    lib = _get_fp8_rs_lib()
    if lib is not None:
        stream = int(torch.cuda.current_stream(src_c.device).cuda_stream)
        rc = lib.scatter_halo_last3_fwd(
            src_c.data_ptr(),
            slots_c.data_ptr(),
            0 if discard_blk_dev is None else discard_blk_dev.data_ptr(),
            cache.data_ptr(),
            int(src_c.stride(0)),
            int(cache.stride(0)),
            int(cache.stride(2)),
            int(n_cache),
            int(stream_offset),
            int(stream_width),
            n_rows,
            stream,
        )
        if rc == 0:
            return True
    if _TRITON_OK:
        _scatter_halo_last3_triton_kernel[(n_rows,)](
            src_c,
            slots_c,
            discard_blk_dev if discard_blk_dev is not None else slots_c,
            cache,
            int(src_c.stride(0)),
            int(cache.stride(0)),
            int(cache.stride(2)),
            int(stream_offset),
            int(stream_width),
            N_CACHE=int(n_cache),
            HAS_DISCARD=discard_blk_dev is not None,
            BLOCK_D=1024,
            num_warps=4,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Public Entrypoints
# ---------------------------------------------------------------------------
def compressed_reduce_scatter_into(
    partial: torch.Tensor,
    out_shard: torch.Tensor,
    mode: str,
    world_size: int,
    rank: int,
    all_to_all_fn: Any | None = None,
    min_tokens: int | None = None,
) -> bool:
    """Compressed 1-hop ReduceScatter (`partial [T, H]` bf16 -> `out_shard [S, H]` bf16).

    Strictly gated to eager prefill (`T >= SP_RS_MIN_T` = 2048, `!capturing`, ZERO touch
    to decode `T <= 512`). Returns True on success, False if shape/dtype/threshold constraints
    are not met (caller then falls back cleanly to standard NCCL BF16 ReduceScatter).
    """
    if not partial.is_cuda or partial.dtype != torch.bfloat16:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if partial.ndim != 2 or out_shard.ndim != 2 or out_shard.dtype != torch.bfloat16:
        return False
    if not partial.is_contiguous() or not out_shard.is_contiguous():
        return False
    T, H = int(partial.shape[0]), int(partial.shape[1])
    eff_min_t = SP_RS_MIN_T if min_tokens is None else max(int(min_tokens), 1024)
    if T < eff_min_t or T % world_size != 0:
        return False
    S = T // world_size
    if int(out_shard.shape[0]) != S or int(out_shard.shape[1]) != H:
        return False

    a2a = _tp_all_to_all_peers if all_to_all_fn is None else all_to_all_fn
    lib = _get_fp8_rs_lib()
    stream = int(torch.cuda.current_stream(partial.device).cuda_stream)

    # -----------------------------------------------------------------------
    # Mode 0: Blockwise FP8-E4M3FN (`fp8_a2a`) / INT8 (`int8_a2a`), group=128
    #   Layout per peer: [S, H] uint8 payload + [S, H//128] bf16 scales, 128B-aligned!
    #   Wire: 1.015625 B/elem = 0.5078x BF16 -> 67.10 MB/rank @ T=16,384 (vs 132.12 MB)!
    # -----------------------------------------------------------------------
    if mode in ("fp8_a2a", "int8_a2a"):
        if H % 128 != 0 or world_size not in (2, 4, 8):
            return False
        is_fp8 = mode == "fp8_a2a"
        G = H // 128
        scale_off = S * H
        peer_bytes = (scale_off + 2 * S * G + 127) & ~127
        send_buf, recv_buf = _get_scratch_pair(partial.device, world_size, peer_bytes, mode)

        if lib is not None:
            q_fn = (
                lib.quant_bf16_to_fp8_g128_a2a_fwd
                if is_fp8
                else lib.quant_bf16_to_int8_g128_a2a_fwd
            )
            d_fn = (
                lib.dequant_sum8_fp8_g128_to_bf16_fwd
                if is_fp8
                else lib.dequant_sum8_int8_g128_to_bf16_fwd
            )
            rc1 = q_fn(
                partial.data_ptr(),
                send_buf.data_ptr(),
                int(partial.stride(0)),
                int(peer_bytes),
                int(scale_off),
                S,
                H,
                world_size,
                rank,
                stream,
            )
            if rc1 == 0:
                a2a(send_buf, recv_buf)
                rc2 = d_fn(
                    recv_buf.data_ptr(),
                    partial.data_ptr(),
                    out_shard.data_ptr(),
                    int(partial.stride(0)),
                    int(out_shard.stride(0)),
                    int(peer_bytes),
                    int(scale_off),
                    S,
                    H,
                    world_size,
                    rank,
                    stream,
                )
                if rc2 == 0:
                    return True

        if not _TRITON_OK:
            return False
        NG = 4
        grid_g = (G + NG - 1) // NG
        _quant_g128_a2a_kernel[(world_size - 1, S, grid_g)](
            partial,
            send_buf,
            partial.stride(0),
            peer_bytes,
            scale_off,
            S,
            H,
            RANK=rank,
            NG=NG,
            IS_FP8=is_fp8,
            num_warps=4,
        )
        a2a(send_buf, recv_buf)
        _dequant_sum_g128_kernel[(S, grid_g)](
            recv_buf,
            partial,
            out_shard,
            partial.stride(0),
            out_shard.stride(0),
            peer_bytes,
            scale_off,
            S,
            H,
            RANK=rank,
            WORLD=world_size,
            NG=NG,
            IS_FP8=is_fp8,
            num_warps=4,
        )
        return True

    # -----------------------------------------------------------------------
    # Mode 1: INT8-col32 Column-Wise Compressed RS (1.0625 B/elem, outlier-immune)
    # -----------------------------------------------------------------------
    if mode == "int8_col":
        if lib is not None and world_size in (2, 4, 8):
            G = (S + 31) // 32
            scale_off = (S * H + 15) & ~15
            peer_bytes = (scale_off + 2 * G * H + 127) & ~127
            send_buf, recv_buf = _get_scratch_pair(
                partial.device, world_size, peer_bytes, "int8_col_cu"
            )
            rc1 = lib.quant_bf16_to_int8_col32_a2a_fwd(
                partial.data_ptr(),
                send_buf.data_ptr(),
                int(partial.stride(0)),
                int(peer_bytes),
                int(scale_off),
                S,
                H,
                world_size,
                rank,
                stream,
            )
            if rc1 == 0:
                a2a(send_buf, recv_buf)
                rc2 = lib.dequant_sum8_int8_col32_to_bf16_fwd(
                    recv_buf.data_ptr(),
                    partial.data_ptr(),
                    out_shard.data_ptr(),
                    int(partial.stride(0)),
                    int(out_shard.stride(0)),
                    int(peer_bytes),
                    int(scale_off),
                    S,
                    H,
                    world_size,
                    rank,
                    stream,
                )
                if rc2 == 0:
                    return True

        if not _TRITON_OK or S % 32 != 0 or H % 64 != 0:
            return False
        G = S // 32
        scale_off = S * H
        peer_bytes = scale_off + 2 * G * H
        send_buf, recv_buf = _get_scratch_pair(partial.device, world_size, peer_bytes, "int8_col")
        BH = 64
        _quant_int8_col32_kernel[(world_size - 1, G, H // BH)](
            partial,
            send_buf,
            partial.stride(0),
            peer_bytes,
            scale_off,
            S,
            H,
            RANK=rank,
            BH=BH,
            num_warps=4,
        )
        a2a(send_buf, recv_buf)
        _dequant_reduce_int8_col32_kernel[(G, H // BH)](
            recv_buf,
            partial,
            out_shard,
            partial.stride(0),
            out_shard.stride(0),
            peer_bytes,
            scale_off,
            S,
            H,
            RANK=rank,
            WORLD=world_size,
            BH=BH,
            num_warps=4,
        )
        return True

    # -----------------------------------------------------------------------
    # Mode 2: BF12-E4M7 Split-Plane Block-Exponent RS (1.5313 B/elem, 96 dB)
    # -----------------------------------------------------------------------
    if mode == "bf12":
        if not _TRITON_OK or H % 32 != 0:
            return False
        num_g = H // 32
        nib_off = S * H
        emax_off = nib_off + S * (H // 2)
        peer_bytes = emax_off + S * num_g
        send_buf, recv_buf = _get_scratch_pair(partial.device, world_size, peer_bytes, "bf12")
        groups_per_cta = 4
        grid_g = (num_g + groups_per_cta - 1) // groups_per_cta
        _pack_bf12_e4m7_kernel[(world_size - 1, S, grid_g)](
            partial,
            send_buf,
            partial.stride(0),
            peer_bytes,
            nib_off,
            emax_off,
            S,
            H,
            RANK=rank,
            GROUPS_PER_CTA=groups_per_cta,
            num_warps=2,
        )
        a2a(send_buf, recv_buf)
        _unpack_reduce_bf12_e4m7_kernel[(S, grid_g)](
            recv_buf,
            partial,
            out_shard,
            partial.stride(0),
            out_shard.stride(0),
            peer_bytes,
            nib_off,
            emax_off,
            S,
            H,
            RANK=rank,
            WORLD=world_size,
            GROUPS_PER_CTA=groups_per_cta,
            num_warps=2,
        )
        return True

    # -----------------------------------------------------------------------
    # Mode 3: Zero-Incast 3-Step Recursive Halving-Doubling (RHD / Butterfly)
    #   `rhd_int8` / `rhd_fp8` (g64 + BF16 scales = 1.03125 B/elem, 0.5156x BF16!)
    #   Only 1 bidirectional peer per step (ZERO 7-to-1 RoCEv2 incast!)
    # -----------------------------------------------------------------------
    if mode in ("rhd_int8", "rhd_fp8"):
        return _rhd_reduce_scatter_into(
            partial=partial,
            out_shard=out_shard,
            is_fp8=(mode == "rhd_fp8"),
            world_size=world_size,
            rank=rank,
            pair_exchange_fn=all_to_all_fn,
        )

    return False


def _rhd_reduce_scatter_into(
    partial: torch.Tensor,
    out_shard: torch.Tensor,
    is_fp8: bool,
    world_size: int,
    rank: int,
    pair_exchange_fn: Any | None = None,
    pipe: bool | None = None,
    transport: str | None = None,
    slices: int | None = None,
) -> bool:
    """Zero-incast MSB-first Recursive Halving ReduceScatter (`r^4 -> r^2 -> r^1` for W=8).

    Uses fused split-half SM121a CUDA kernels (`libpf_rhd_comm_sm121.so`) with seamless
    Triton fallback. Optional Step-0 2-stage pipelining (`pipe=True` / `INKLING_PREFILL_SP_RHD_PIPE=1`)
    overlaps Step 0B quant + Step 1 send-half re-quant + Step 0B keep-half add behind wire!
    """
    if world_size not in (2, 4, 8):
        return False
    T, H = int(partial.shape[0]), int(partial.shape[1])
    if H % 128 != 0 or T % world_size != 0:
        return False
    S = T // world_size
    K = world_size.bit_length() - 1  # 1, 2, or 3

    lib = _get_rhd_comm_lib()
    if lib is None and not _TRITON_OK:
        return False

    use_pipe = (SP_RHD_PIPE if pipe is None else bool(pipe)) and (lib is not None) and (world_size >= 4)
    half0 = T // 2
    sc0_off, b0 = _rhd_g64_packed_bytes(half0, H)
    sub_M = half0 // 2
    sub_sc_off, sub_b = _rhd_g64_packed_bytes(sub_M, H) if world_size >= 4 else (0, 0)
    max_half_bytes = max(b0, 2 * sub_b)
    max_keep_elems = max(sub_M * H, S * H)
    st = _get_rhd_rs_scratch(partial.device, max_half_bytes, max_keep_elems, need_pipe=use_pipe)

    if pair_exchange_fn is None:
        def _pair_ex(s_v: torch.Tensor, r_v: torch.Tensor, p: int, b: int) -> None:
            _tp_pair_exchange(s_v, r_v, p, b, transport=transport, slices=slices)
    else:
        _pair_ex = pair_exchange_fn

    cur_stream = torch.cuda.current_stream(partial.device)
    cur_stream_u64 = int(cur_stream.cuda_stream)
    G = H // 64
    NG = 8
    grid_g = (G + NG - 1) // NG

    if lib is not None:
        q_fn = lib.rhd_quant_slice_fp8_g64_fwd if is_fp8 else lib.rhd_quant_slice_int8_g64_fwd
        split_fn = (
            lib.rhd_dequant_add_split_quant_fp8_g64_fwd
            if is_fp8
            else lib.rhd_dequant_add_split_quant_int8_g64_fwd
        )
        daq_fn = (
            lib.rhd_dequant_add_quant_fp8_g64_fwd
            if is_fp8
            else lib.rhd_dequant_add_quant_int8_g64_fwd
        )
        dab_fn = (
            lib.rhd_dequant_add_bf16_fp8_g64_fwd
            if is_fp8
            else lib.rhd_dequant_add_bf16_int8_g64_fwd
        )
    else:
        q_fn = split_fn = daq_fn = dab_fn = None

    def _run_quant(src: torch.Tensor, dst_u8: torch.Tensor, row_off: int, sc_off: int, M: int, s_u64: int) -> None:
        if q_fn is not None:
            rc = q_fn(
                src.data_ptr(), dst_u8.data_ptr(), int(src.stride(0)),
                int(row_off), int(sc_off), int(M), H, s_u64,
            )
            if rc != 0:
                raise RuntimeError(f"rhd_quant_slice_g64_fwd failed: rc={rc}")
        else:
            _rhd_quant_slice_g64_triton[(M, grid_g)](
                src, dst_u8, int(src.stride(0)), row_off, sc_off, H,
                NG=NG, IS_FP8=is_fp8, num_warps=4,
            )

    def _run_split(
        recv_u8: torch.Tensor, local_bf: torch.Tensor,
        next_send_u8: torch.Tensor, next_keep_bf: torch.Tensor,
        local_off: int, r_sc_off: int, n_sc_off: int, N: int,
        send_upper: bool, wb_keep: int, s_u64: int,
    ) -> None:
        if split_fn is not None:
            rc = split_fn(
                recv_u8.data_ptr(), local_bf.data_ptr(),
                next_send_u8.data_ptr(), next_keep_bf.data_ptr(),
                int(local_bf.stride(0)), int(local_off), int(r_sc_off), int(n_sc_off),
                int(next_keep_bf.stride(0)), int(N), H, 1 if send_upper else 0, int(wb_keep), s_u64,
            )
            if rc != 0:
                raise RuntimeError(f"rhd_dequant_add_split_quant_g64_fwd failed: rc={rc}")
        else:
            _rhd_dequant_add_split_quant_g64_triton[(2 * N, grid_g)](
                recv_u8, local_bf, next_send_u8, next_keep_bf,
                int(local_bf.stride(0)), local_off, r_sc_off, n_sc_off,
                int(next_keep_bf.stride(0)), N, H,
                SEND_UPPER=send_upper, NG=NG, IS_FP8=is_fp8, num_warps=4,
            )

    def _run_dab(
        recv_u8: torch.Tensor, local_bf: torch.Tensor, out_bf: torch.Tensor,
        local_off: int, r_sc_off: int, M: int, wb_out: int, s_u64: int,
    ) -> None:
        if dab_fn is not None:
            rc = dab_fn(
                recv_u8.data_ptr(), local_bf.data_ptr(), out_bf.data_ptr(),
                int(local_bf.stride(0)), int(local_off), int(r_sc_off),
                int(out_bf.stride(0)), int(M), H, int(wb_out), s_u64,
            )
            if rc != 0:
                raise RuntimeError(f"rhd_dequant_add_bf16_g64_fwd failed: rc={rc}")
        else:
            _rhd_dequant_add_bf16_g64_triton[(M, grid_g)](
                recv_u8, local_bf, out_bf, int(local_bf.stride(0)),
                local_off, r_sc_off, int(out_bf.stride(0)), H,
                NG=NG, IS_FP8=is_fp8, num_warps=4,
            )

    bit0 = world_size >> 1
    send_upper0 = (rank & bit0) == 0
    send0_off = half0 if send_upper0 else 0
    keep0_off = 0 if send_upper0 else half0

    # -----------------------------------------------------------------------
    # Optional Step-0 2-Stage Overlapped Pipeline (`pipe=True`, W >= 4, CUDA lib)
    #   Splits Step 0 (4S) into sub-half A (2S -> becomes send1) and sub-half B
    #   (2S -> becomes keep1). Because `(rank ^ bit0) & bit1 == rank & bit1`,
    #   both partners agree on A vs B -> 0-ULP identical to unpipelined!
    # -----------------------------------------------------------------------
    if use_pipe:
        cs = st["codec_stream"]
        cs_u64 = int(cs.cuda_stream)
        partial.record_stream(cs)

        bit1 = bit0 >> 1
        send_upper1 = (rank & bit1) == 0
        sub_A_off = sub_M if send_upper1 else 0
        sub_B_off = 0 if send_upper1 else sub_M

        send0_A = st["send0"][:sub_b]
        send0_B = st["send0"][sub_b : 2 * sub_b]
        recv0_A = st["recv0"][:sub_b]
        recv0_B = st["recv0"][sub_b : 2 * sub_b]
        send1 = st["send1"][:sub_b]
        recv1 = st["recv1"][:sub_b]
        keep1 = st["keep1"][: sub_M * H].view(sub_M, H)
        wb_k1 = 1 if (sub_M * H * 2 <= (20 << 20)) else 0

        st["ev_start"].record(cur_stream)
        cs.wait_event(st["ev_start"])

        # 1. Quantize sub-half A on codec_stream -> signal ev_qA
        _run_quant(partial, send0_A, send0_off + sub_A_off, sub_sc_off, sub_M, cs_u64)
        st["ev_qA"].record(cs)

        # 2. Quantize sub-half B on codec_stream (overlaps Step 0A wire!)
        _run_quant(partial, send0_B, send0_off + sub_B_off, sub_sc_off, sub_M, cs_u64)
        st["ev_qB"].record(cs)

        # 3. Step 0A wire on cur_stream -> signal ev_cA
        partner0 = rank ^ bit0
        cur_stream.wait_event(st["ev_qA"])
        _pair_ex(send0_A, recv0_A, partner0, bit0)
        st["ev_cA"].record(cur_stream)

        # 4. Immediately compute send1 from recv0_A + local0_A on codec_stream (overlaps Step 0B wire!)
        cs.wait_event(st["ev_cA"])
        rc = daq_fn(
            recv0_A.data_ptr(), partial.data_ptr(), send1.data_ptr(),
            int(partial.stride(0)), int(keep0_off + sub_A_off), int(sub_sc_off), int(sub_sc_off),
            int(sub_M), H, cs_u64,
        )
        if rc != 0:
            raise RuntimeError(f"rhd_dequant_add_quant_g64_fwd failed: rc={rc}")
        st["ev_s1"].record(cs)

        # 5. Step 0B wire on cur_stream -> signal ev_cB
        cur_stream.wait_event(st["ev_qB"])
        _pair_ex(send0_B, recv0_B, partner0, bit0)
        st["ev_cB"].record(cur_stream)

        # 6. Compute keep1 from recv0_B + local0_B on codec_stream (overlaps Step 1 wire!)
        cs.wait_event(st["ev_cB"])
        _run_dab(recv0_B, partial, keep1, keep0_off + sub_B_off, sub_sc_off, sub_M, wb_k1, cs_u64)
        st["ev_k1"].record(cs)

        # 7. Step 1 wire on cur_stream (starts immediately once send1 ready, ZERO gap!)
        partner1 = rank ^ bit1
        cur_stream.wait_event(st["ev_s1"])
        _pair_ex(send1, recv1, partner1, bit1)
        cur_stream.wait_event(st["ev_k1"])

        if K == 2:
            # W == 4: Step 1 was final step!
            _run_dab(recv1, keep1, out_shard, 0, sub_sc_off, S, 1, cur_stream_u64)
            return True

        # W == 8: fuse Step 1 -> Step 2 (`split_quant` -> Step 2 exchange -> final add!)
        next_bit = bit1 >> 1
        send_upper2 = (rank & next_bit) == 0
        sc2_off, b2 = _rhd_g64_packed_bytes(S, H)
        send2 = st["send0"][:b2]
        recv2 = st["recv0"][:b2]
        keep2 = st["keep2"][: S * H].view(S, H)
        _run_split(
            recv1, keep1, send2, keep2,
            local_off=0, r_sc_off=sub_sc_off, n_sc_off=sc2_off,
            N=S, send_upper=send_upper2, wb_keep=1, s_u64=cur_stream_u64,
        )
        _pair_ex(send2, recv2, rank ^ next_bit, next_bit)
        _run_dab(recv2, keep2, out_shard, 0, sc2_off, S, 1, cur_stream_u64)
        return True

    # -----------------------------------------------------------------------
    # Unpipelined Fused Split-Half RHD ReduceScatter (4 kernels for W=8!)
    # -----------------------------------------------------------------------
    cur_send = st["send0"][:b0]
    cur_recv = st["recv0"][:b0]
    _run_quant(partial, cur_send, send0_off, sc0_off, half0, cur_stream_u64)

    cur_local = partial
    cur_local_off = keep0_off
    cur_sc_off = sc0_off
    cur_M = half0

    for step in range(K):
        bit = world_size >> (step + 1)
        partner = rank ^ bit
        _pair_ex(cur_send, cur_recv, partner, bit)
        if step == (K - 1):
            _run_dab(cur_recv, cur_local, out_shard, cur_local_off, cur_sc_off, S, 1, cur_stream_u64)
        else:
            next_M = cur_M // 2
            next_bit = bit >> 1
            send_upper_next = (rank & next_bit) == 0
            next_sc_off, next_b = _rhd_g64_packed_bytes(next_M, H)
            if step % 2 == 0:
                next_send = st["send1"][:next_b]
                next_recv = st["recv1"][:next_b]
                next_keep = st["keep1"][: next_M * H].view(next_M, H)
            else:
                next_send = st["send0"][:next_b]
                next_recv = st["recv0"][:next_b]
                next_keep = st["keep2"][: next_M * H].view(next_M, H)
            wb_keep = 1 if (next_M * H * 2 <= (20 << 20)) else 0
            _run_split(
                cur_recv, cur_local, next_send, next_keep,
                local_off=cur_local_off, r_sc_off=cur_sc_off, n_sc_off=next_sc_off,
                N=next_M, send_upper=send_upper_next, wb_keep=wb_keep, s_u64=cur_stream_u64,
            )
            cur_local = next_keep
            cur_local_off = 0
            cur_send = next_send
            cur_recv = next_recv
            cur_sc_off = next_sc_off
            cur_M = next_M

    return True


def compressed_all_gather_dim0(
    shard: torch.Tensor,
    mode: str,
    fp8_direct: bool = False,
    world_size: int | None = None,
    rank: int | None = None,
    pair_exchange_fn: Any | None = None,
    min_tokens: int | None = None,
    transport: str | None = None,
    slices: int | None = None,
) -> torch.Tensor | None:
    """Zero-incast 3-Step Mirror Recursive-Doubling AllGather (`r^1 -> r^2 -> r^4` for W=8).

    Strictly gated to eager prefill (`T = S * W >= SP_AG_MIN_T` = 2048, `!capturing`, ZERO
    touch to decode `T <= 512` or small `k_sync=3` halos!).
    Quantizes ONCE before Step 0 and forwards quantized bytes verbatim across all 3 doubling
    hops (ZERO intermediate re-quantization, 100% bit-identical outputs across all 8 TP ranks!).

    Modes:
      - `rhd_fp8` with `fp8_direct=True`: packs `[S, H]` FP8-E4M3FN + `[S]` FP32 row scales
        into ONE contiguous shard (`1.0009 B/elem` -> 1 collective instead of 2!), returns
        `x_fp8_full [T, H]` (`float8_e4m3fn`) carrying `._inkling_fp8_scale` `[T, 1]` fp32!
      - `rhd_int8` / `rhd_fp8` with `fp8_direct=False`: g64 + BF16 scales (`1.03125 B/elem`),
        returns `[T, H]` BF16 (`44.26 dB` SQNR for `rhd_int8`!).
    """
    if mode not in ("rhd_int8", "rhd_fp8"):
        return None
    if not shard.is_cuda or shard.dtype != torch.bfloat16 or shard.ndim != 2:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    if world_size is None or rank is None:
        from vllm.distributed import get_tp_group

        tp_group = get_tp_group()
        world_size = int(tp_group.world_size)
        rank = int(tp_group.rank_in_group)

    if world_size not in (2, 4, 8):
        return None
    S, H = int(shard.shape[0]), int(shard.shape[1])
    T = S * world_size
    eff_min_t = SP_AG_MIN_T if min_tokens is None else max(int(min_tokens), 1024)
    if T < eff_min_t:
        return None

    shard_c = shard.contiguous()
    lib = _get_rhd_comm_lib()
    cur_stream = torch.cuda.current_stream(shard_c.device)
    stream_u64 = int(cur_stream.cuda_stream)

    if pair_exchange_fn is None:
        def _pair_ex(s_v: torch.Tensor, r_v: torch.Tensor, p: int, b: int) -> None:
            _tp_pair_exchange(s_v, r_v, p, b, transport=transport, slices=slices)
    else:
        _pair_ex = pair_exchange_fn

    def _run_doubling(ag_buf: torch.Tensor) -> None:
        K = world_size.bit_length() - 1
        for s in range(K):
            bit = 1 << s
            chunk = 1 << s
            mask = ~(chunk - 1)
            partner = rank ^ bit
            send_base = rank & mask
            recv_base = partner & mask
            s_view = ag_buf[send_base : send_base + chunk].reshape(-1)
            r_view = ag_buf[recv_base : recv_base + chunk].reshape(-1)
            _pair_ex(s_view, r_view, partner, bit)

    # -----------------------------------------------------------------------
    # Direct FP8 Rowwise RHD AllGather (1 collective instead of 2!)
    # -----------------------------------------------------------------------
    if mode == "rhd_fp8" and fp8_direct:
        if H % 16 != 0:
            return None
        scale_off, shard_bytes = _rhd_rowwise_fp8_shard_bytes(S, H)
        ag_buf = _get_rhd_ag_scratch(shard_c.device, world_size, shard_bytes, "rhd_ag_fp8_dir")
        if lib is not None:
            rc1 = lib.rhd_ag_quant_fp8_rowwise_fwd(
                shard_c.data_ptr(), ag_buf[rank].data_ptr(),
                int(shard_c.stride(0)), int(scale_off), S, H, stream_u64,
            )
            if rc1 != 0:
                return None
            _run_doubling(ag_buf)
            out_fp8 = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=shard_c.device)
            out_scale = torch.empty((T, 1), dtype=torch.float32, device=shard_c.device)
            rc2 = lib.rhd_ag_unpack_fp8_rowwise_fwd(
                ag_buf.data_ptr(), out_fp8.data_ptr(), out_scale.data_ptr(),
                int(shard_bytes), int(scale_off), int(out_fp8.stride(0)),
                S, H, world_size, stream_u64,
            )
            if rc2 != 0:
                return None
            out_fp8._inkling_fp8_scale = out_scale  # type: ignore[attr-defined]
            return out_fp8
        # Fallback without CUDA .so: reuse linear.py's quantize_fp8_rowwise into packed slot
        try:
            from .ops.linear import quantize_fp8_rowwise

            x_fp8_s, x_scale_s = quantize_fp8_rowwise(shard_c, pad_m16=False)
        except Exception:
            amax = shard_c.float().abs().amax(dim=-1, keepdim=True)
            x_scale_s = (amax * (1.0 / 448.0)).clamp(min=1e-12)
            x_fp8_s = (shard_c.float() / x_scale_s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        ag_buf[rank, :scale_off].copy_(x_fp8_s.view(torch.uint8).reshape(-1))
        ag_buf[rank, scale_off : scale_off + 4 * S].copy_(x_scale_s.view(torch.uint8).reshape(-1))
        _run_doubling(ag_buf)
        out_fp8 = ag_buf[:, :scale_off].contiguous().view(T, H).view(torch.float8_e4m3fn)
        out_scale = (
            ag_buf[:, scale_off : scale_off + 4 * S]
            .contiguous()
            .view(torch.float32)
            .reshape(T, 1)
        )
        out_fp8._inkling_fp8_scale = out_scale  # type: ignore[attr-defined]
        return out_fp8

    # -----------------------------------------------------------------------
    # Blockwise g64 + BF16 Scale RHD AllGather -> BF16 `[T, H]`
    # -----------------------------------------------------------------------
    if H % 128 != 0:
        return None
    is_fp8 = mode == "rhd_fp8"
    scale_off, shard_bytes = _rhd_g64_packed_bytes(S, H)
    ag_buf = _get_rhd_ag_scratch(shard_c.device, world_size, shard_bytes, f"rhd_ag_{mode}")
    out_bf16 = torch.empty((T, H), dtype=torch.bfloat16, device=shard_c.device)

    if lib is not None:
        q_fn = lib.rhd_quant_slice_fp8_g64_fwd if is_fp8 else lib.rhd_quant_slice_int8_g64_fwd
        d_fn = lib.rhd_ag_dequant_fp8_g64_fwd if is_fp8 else lib.rhd_ag_dequant_int8_g64_fwd
        rc1 = q_fn(
            shard_c.data_ptr(), ag_buf[rank].data_ptr(),
            int(shard_c.stride(0)), 0, int(scale_off), S, H, stream_u64,
        )
        if rc1 != 0:
            return None
        _run_doubling(ag_buf)
        rc2 = d_fn(
            ag_buf.data_ptr(), out_bf16.data_ptr(),
            int(shard_bytes), int(scale_off), int(out_bf16.stride(0)),
            S, H, world_size, stream_u64,
        )
        if rc2 != 0:
            return None
        return out_bf16

    if not _TRITON_OK:
        return None
    G = H // 64
    NG = 8
    grid_g = (G + NG - 1) // NG
    _rhd_quant_slice_g64_triton[(S, grid_g)](
        shard_c, ag_buf[rank], int(shard_c.stride(0)), 0, scale_off, H,
        NG=NG, IS_FP8=is_fp8, num_warps=4,
    )
    _run_doubling(ag_buf)
    _rhd_ag_dequant_g64_triton[(S, world_size, grid_g)](
        ag_buf, out_bf16, shard_bytes, scale_off, int(out_bf16.stride(0)), S, H,
        NG=NG, IS_FP8=is_fp8, num_warps=4,
    )
    return out_bf16
