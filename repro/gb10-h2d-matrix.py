#!/usr/bin/env python3
"""Which host-to-GPU copy pattern is slow or stalls on this GB10 node.

Reproduces the Inkling2 expert fill in isolation: ~1 GiB tensors copied
into GPU memory from a safetensors-style mmap of a file on disk, one case
per fresh process, with a stall watchdog. Takes minutes, not a 1-2 h boot.

  mmap        mmap view -> preallocated param   (stock Inkling2 loader)
  touched     same, pages touched on the CPU first (vLLM's copy_weight_)
  anon        resident anonymous copy -> preallocated param
  staged      pread + pinned staging -> preallocated param
              (patches/gb10_load.py, the fix in this repo)
  anon-fresh  resident anonymous copy -> GPU memory allocated at copy time

Reading the table:
  mmap slow or STALLED, anon/staged fast  -> the source page-fault path;
                                             the staged loader avoids it.
  anon slow, anon-fresh fast              -> the preallocated destination;
                                             not fixed by staging the source.
  everything slow                         -> memory pressure or clocks.

Run it inside the serving image on a failing node, against a real shard,
with the model's occupancy:

  docker run --rm --gpus all --privileged --ipc=host \\
    -v "$PWD":/w -v "$HF_CACHE":/root/.cache/huggingface \\
    --entrypoint python3 vllm/vllm-openai:inkling-2-small \\
    /w/repro/gb10-h2d-matrix.py --occupy-gib 85 \\
    --file /root/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Small/snapshots/<rev>/model-00002-of-00024.safetensors
"""

import argparse
import mmap
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

CASES = ("mmap", "touched", "anon", "staged", "anon-fresh")
PATCHES = Path(__file__).resolve().parent.parent / "patches"


def run_case(args) -> None:
    """One case, in its own process: prints a line per tensor."""
    import torch

    device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    nbytes = args.tensor_mib << 20
    size = os.path.getsize(args.file)

    # Held until exit: the rest of the model's parameters.
    ballast = [
        torch.empty(1 << 30, dtype=torch.uint8, device=device) for _ in range(args.occupy_gib)
    ]
    dsts = []
    if args.case != "anon-fresh":
        dsts = [torch.empty(nbytes, dtype=torch.uint8, device=device) for _ in range(args.count)]
    sync()

    # Cold page cache for the file, then the same lazy mapping safetensors
    # hands vLLM: a MAP_PRIVATE mmap of the whole shard, sliced per tensor.
    fd = os.open(args.file, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    storage = torch.UntypedStorage.from_file(args.file, shared=False, nbytes=size)
    if args.case == "staged":
        sys.path.insert(0, str(PATCHES))
        import gb10_load

    prep_s = copy_s = 0.0
    for i in range(args.count):
        lo = args.offset + i * nbytes
        src = torch.asarray(storage[lo : lo + nbytes], dtype=torch.uint8)
        t0 = time.perf_counter()
        if args.case == "touched":
            src[:: mmap.PAGESIZE].sum()
        elif args.case in ("anon", "anon-fresh"):
            src = src.clone()
        t1 = time.perf_counter()
        if args.case == "staged":
            gb10_load.copy_from_checkpoint_(dsts[i], src)
        elif args.case == "anon-fresh":
            dsts.append(src.to(device))
        else:
            dsts[i].copy_(src)
        sync()
        t2 = time.perf_counter()
        prep_s += t1 - t0
        copy_s += t2 - t1
        print(
            f"  tensor {i + 1}/{args.count}: prep {t1 - t0:6.2f} s, "
            f"copy {t2 - t1:6.2f} s, {args.tensor_mib / (t2 - t0):8.0f} MiB/s",
            flush=True,
        )
    total = args.tensor_mib * args.count
    print(f"RESULT {total / (prep_s + copy_s):.0f} {total / copy_s:.0f}", flush=True)
    del ballast, dsts


def watch(cmd: list[str], env: dict, stall_s: float) -> tuple[str, str]:
    """Run a case; ``(end-to-end MiB/s, copy-only MiB/s)`` or a STALLED note
    when it prints nothing for ``stall_s`` seconds."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # exited

    threading.Thread(target=pump, daemon=True).start()
    result = ("failed", "")
    done = 0
    while True:
        try:
            line = lines.get(timeout=stall_s)
        except queue.Empty:
            proc.kill()
            result = (f"STALLED after tensor {done} (>{stall_s:.0f} s silent)", "")
            break
        if line is None:
            break
        if line.startswith("RESULT"):
            result = tuple(line.split()[1:3])
            break
        done += line.lstrip().startswith("tensor")
        print(line, end="", flush=True)
    proc.wait()
    return result


def make_file(path: Path, nbytes: int) -> None:
    block = os.urandom(8 << 20)
    with open(path, "wb") as f:
        for _ in range(-(-nbytes // len(block))):
            f.write(block)
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--file", help="shard to map (default: a scratch file in --dir)")
    p.add_argument("--dir", default=".", help="where the scratch file goes (real disk, not tmpfs)")
    p.add_argument("--offset", type=int, default=64 << 20, help="first byte used in --file")
    p.add_argument("--tensor-mib", type=int, default=972, help="NVFP4 w13 slice of one rank")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--occupy-gib", type=int, default=0, help="GPU memory held first (85 = model)")
    p.add_argument("--stall-s", type=float, default=180.0, help="silence that counts as a stall")
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.case:
        run_case(args)
        return

    need = args.offset + (args.tensor_mib << 20) * args.count
    scratch = None
    if not args.file:
        scratch = Path(args.dir) / f"gb10-h2d-matrix-{os.getpid()}.bin"
        print(f"writing {need / 2**30:.1f} GiB scratch file {scratch} ...", flush=True)
        make_file(scratch, need)
        args.file = str(scratch)
    elif os.path.getsize(args.file) < need:
        sys.exit(f"{args.file} is smaller than offset + tensor-mib * count ({need} bytes)")

    env = dict(os.environ)
    if args.device != "cuda":
        env["INKLING_LOAD_STAGED"] = "force"
    rows = []
    try:
        for case in args.cases.split(","):
            print(f"[{case}]", flush=True)
            cmd = [sys.executable, __file__, "--case", case, "--file", args.file]
            for name in ("offset", "tensor_mib", "count", "occupy_gib", "device"):
                cmd += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
            rows.append((case, *watch(cmd, env, args.stall_s)))
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)

    print(f"\n{'case':<12}{'end-to-end MiB/s':>18}{'copy only MiB/s':>18}")
    for case, total, copy in rows:
        print(f"{case:<12}{total:>18}{copy:>18}")


if __name__ == "__main__":
    main()
