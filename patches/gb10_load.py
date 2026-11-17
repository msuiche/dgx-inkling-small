# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged checkpoint -> GPU fill for GB10 (DGX Spark, unified memory).

vLLM's lazy safetensors iterator yields CPU tensors that are views into a
MAP_PRIVATE mmap of the shard file, and the model's parameters are already
allocated on the GPU before the first weight is read. ``param.copy_(view)``
therefore hands libcuda a pageable source whose pages are not mapped in the
process yet. On a discrete GPU those faults are ordinary CPU faults and cost
nothing; on GB10 they are taken inside the driver copy, cost ~40x more, and
libcuda busy-waits on one core meanwhile (vllm-project/vllm#58726, measured
on this ring's driver and kernel). Upstream fixed the stock loaders with
``copy_weight_`` (vllm#58868); the Inkling2 expert loader calls ``copy_``
directly and still takes the slow path for 96% of each rank's bytes.

``copy_from_checkpoint_`` keeps the mmap out of the driver copy entirely:
each chunk of the rank's own slice is ``pread`` into a small page-locked
staging buffer and copied to the GPU from there, and the page cache behind
the read is dropped so it does not compete with the ~91 GiB of parameters
for the same unified memory. Extra memory is one staging chunk.

Environment (read at import, all optional):
  INKLING_LOAD_STAGED=0       stock ``dst.copy_(src)`` (A/B against the patch);
                              ``force`` stages for any device (CPU tests)
  INKLING_LOAD_CHUNK_MB=128   staging chunk size
  INKLING_LOAD_DROP_CACHE=0   keep the page cache behind the read
"""

import ctypes
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

_MODE = os.environ.get("INKLING_LOAD_STAGED", "1")
_CHUNK_BYTES = max(1, int(os.environ.get("INKLING_LOAD_CHUNK_MB", "128"))) << 20
_DROP_CACHE = os.environ.get("INKLING_LOAD_DROP_CACHE", "1") != "0"
# Below this a tensor is copied to anonymous memory in one piece instead.
_SMALL_BYTES = 1 << 20
# Bytes compared at each end of a tensor to prove the file range is the view.
_PROBE_BYTES = 4096
_LOG_EVERY_BYTES = 8 << 30
# An H2D copy from a resident staging buffer slower than this is not the
# mmap problem this module removes; say so instead of looking like a hang.
_SLOW_H2D_BYTES_PER_S = 200 << 20

_fd_key: tuple[str, int] | None = None
_fd = -1
_fd_size = 0
_warned: set[str] = set()
_stats = {"bytes": 0, "tensors": 0, "read_s": 0.0, "h2d_s": 0.0, "logged": 0}


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def _file_range(addr: int, nbytes: int) -> tuple[str, int, int] | None:
    """``(path, file offset, inode)`` of the file mapping that backs
    ``[addr, addr + nbytes)``, or None when it is not one regular file mapped
    contiguously (anonymous memory, a hole, a deleted file)."""
    end = addr + nbytes
    covered = addr
    path = None
    inode = offset = 0
    try:
        with open("/proc/self/maps", "rb") as maps:
            for line in maps:
                fields = line.split(None, 5)
                if len(fields) < 5:
                    continue
                lo_hex, _, hi_hex = fields[0].partition(b"-")
                lo, hi = int(lo_hex, 16), int(hi_hex, 16)
                if hi <= covered:
                    continue
                if lo > covered or len(fields) < 6:
                    return None
                name = fields[5].rstrip(b"\n")
                ino = int(fields[4])
                if not name.startswith(b"/") or name.endswith(b" (deleted)") or not ino:
                    return None
                file_off = int(fields[2], 16) + (covered - lo)
                if path is None:
                    path, inode, offset = name, ino, file_off
                elif name != path or ino != inode or file_off != offset + (covered - addr):
                    return None
                covered = hi
                if covered >= end:
                    return os.fsdecode(path), offset, inode
    except (OSError, ValueError):
        return None
    return None


def _open_backing(path: str, inode: int, end: int) -> int:
    """Read-only fd for ``path`` (cached: shards are loaded one at a time),
    or -1 when it is not the mapped inode or is shorter than ``end``."""
    global _fd_key, _fd, _fd_size
    if _fd_key != (path, inode):
        if _fd >= 0:
            os.close(_fd)
        _fd_key, _fd = None, -1
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        stat = os.fstat(fd)
        if stat.st_ino != inode:
            os.close(fd)
            return -1
        _fd_key, _fd, _fd_size = (path, inode), fd, stat.st_size
    return _fd if _fd_size >= end else -1


def _pread_into(fd: int, ptr: int, nbytes: int, offset: int) -> None:
    view = memoryview((ctypes.c_char * nbytes).from_address(ptr))
    done = 0
    while done < nbytes:
        got = os.preadv(fd, [view[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short read at {offset + done} ({done}/{nbytes} bytes)")
        done += got


def _backing_file(src: torch.Tensor, nbytes: int) -> tuple[int, int]:
    """``(fd, file offset)`` to pread ``src``'s bytes from, or ``(-1, 0)``.

    The mapping is found by address, then both ends of the tensor are
    compared against the file, so a wrong guess can only fall back to
    copying from the view, never load the wrong bytes."""
    ptr = src.data_ptr()
    found = _file_range(ptr, nbytes)
    if found is None:
        return -1, 0
    path, offset, inode = found
    try:
        fd = _open_backing(path, inode, offset + nbytes)
        if fd < 0:
            return -1, 0
        probe = min(nbytes, _PROBE_BYTES)
        tail = nbytes - probe
        if os.pread(fd, probe, offset) != ctypes.string_at(ptr, probe) or os.pread(
            fd, probe, offset + tail
        ) != ctypes.string_at(ptr + tail, probe):
            _warn_once("probe", "staged load: %s does not match its mapping", path)
            return -1, 0
    except OSError as e:
        _warn_once("open", "staged load: cannot read %s directly (%s)", path, e)
        return -1, 0
    return fd, offset


def _staging(numel: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
    if pin:
        try:
            return torch.empty(numel, dtype=dtype, pin_memory=True)
        except RuntimeError as e:
            _warn_once("pin", "staged load: pageable staging buffer (%s)", e)
    return torch.empty(numel, dtype=dtype)


def _staged_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Chunked ``dst.copy_(src)`` through a staging buffer; both contiguous,
    same dtype and shape."""
    itemsize = src.element_size()
    total = src.numel()
    nbytes = total * itemsize
    # Always one full chunk, so torch's pinned-memory cache reuses one block.
    step = max(1, _CHUNK_BYTES // itemsize)
    dst_flat, src_flat = dst.view(-1), src.view(-1)
    stage = _staging(step, src.dtype, pin=dst.device.type == "cuda")
    fd, offset = _backing_file(src_flat, nbytes)

    read_s = h2d_s = 0.0
    for start in range(0, total, step):
        n = min(step, total - start)
        t0 = time.perf_counter()
        if fd >= 0:
            at = offset + start * itemsize
            try:
                _pread_into(fd, stage.data_ptr(), n * itemsize, at)
                if _DROP_CACHE:
                    os.posix_fadvise(fd, at, n * itemsize, os.POSIX_FADV_DONTNEED)
            except OSError as e:
                _warn_once("read", "staged load: direct read failed (%s)", e)
                fd = -1
        if fd < 0:
            stage[:n].copy_(src_flat[start : start + n])
        t1 = time.perf_counter()
        dst_flat[start : start + n].copy_(stage[:n])
        t2 = time.perf_counter()
        read_s += t1 - t0
        h2d_s += t2 - t1

    if nbytes >= _CHUNK_BYTES and nbytes < h2d_s * _SLOW_H2D_BYTES_PER_S:
        logger.warning(
            "staged load: slow host-to-GPU copy from a resident buffer: "
            "%.0f MiB in %.1f s; this is not the mmap source path",
            nbytes / 2**20,
            h2d_s,
        )
    _stats["bytes"] += nbytes
    _stats["tensors"] += 1
    _stats["read_s"] += read_s
    _stats["h2d_s"] += h2d_s
    if _stats["bytes"] - _stats["logged"] >= _LOG_EVERY_BYTES:
        _stats["logged"] = _stats["bytes"]
        gib = _stats["bytes"] / 2**30
        logger.info(
            "staged load: %.1f GiB in %d tensors (read %.2f GiB/s, "
            "host-to-GPU %.2f GiB/s, %s)",
            gib,
            _stats["tensors"],
            gib / max(_stats["read_s"], 1e-9),
            gib / max(_stats["h2d_s"], 1e-9),
            "pread" if fd >= 0 else "mmap view",
        )


def copy_from_checkpoint_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src)`` for a checkpoint tensor that may be an mmap view,
    without letting the GPU driver take the source's page faults."""
    staged = _MODE == "force" or (
        _MODE != "0" and dst.device.type == "cuda" and src.device.type == "cpu"
    )
    if not staged or src.numel() == 0:
        dst.copy_(src)
    elif (
        src.numel() * src.element_size() <= _SMALL_BYTES
        or src.dtype != dst.dtype
        or src.shape != dst.shape
        or not src.is_contiguous()
        or not dst.is_contiguous()
    ):
        # The clone faults the pages in on the CPU; the driver then copies
        # from resident anonymous memory.
        dst.copy_(src.clone())
    else:
        _staged_copy(dst, src)
