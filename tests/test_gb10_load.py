#!/usr/bin/env python3
"""Byte-exactness of the staged GB10 weight fill (patches/gb10_load.py).

Needs Linux (/proc/self/maps), torch and safetensors; no GPU. The staging
path is forced for CPU destinations, so everything except the pinned
allocation and the CUDA copy itself runs here: resolving a safetensors mmap
view to its file range, the chunked pread, the fallbacks, and the patched
``Inkling2Experts`` loaders end to end against stubbed vLLM modules.

  python3 tests/test_gb10_load.py        (or: pytest tests/test_gb10_load.py)
"""

import ctypes
import importlib
import mmap
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

os.environ["INKLING_LOAD_STAGED"] = "force"

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PATCHES = Path(__file__).resolve().parent.parent / "patches"
sys.path.insert(0, str(PATCHES))
import gb10_load  # noqa: E402

FP8 = torch.float8_e4m3fn
TP = 4


def raw(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())


def random_tensor(shape, dtype, seed) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if dtype == torch.float32:
        return torch.rand(shape, generator=gen)
    data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)
    return data if dtype == torch.uint8 else data.view(dtype)


def write_checkpoint(path: Path, tensors: dict) -> None:
    # The metadata length shifts every tensor off page alignment.
    save_file(tensors, str(path), metadata={"pad": "x" * 37})


def small_chunks(chunk=100_003, small=4096):
    """Tiny, odd chunk so a test tensor spans many unaligned chunks."""
    gb10_load._CHUNK_BYTES, gb10_load._SMALL_BYTES = chunk, small
    gb10_load._warned.clear()


def test_resolves_mmap_view_and_reads_exact_bytes(tmp_path: Path) -> None:
    small_chunks()
    tensors = {
        "a.u8": random_tensor((8, 512, 300), torch.uint8, 1),
        "b.fp8": random_tensor((8, 512, 300), FP8, 2),
        "c.f32": random_tensor((8, 128, 257), torch.float32, 3),
        "d.bf16": random_tensor((8, 128, 257), torch.float32, 4).to(torch.bfloat16),
    }
    path = tmp_path / "ckpt.safetensors"
    write_checkpoint(path, tensors)
    with safe_open(str(path), framework="pt") as f:
        for name, want in tensors.items():
            view = f.get_tensor(name)
            nbytes = view.numel() * view.element_size()
            found = gb10_load._file_range(view.data_ptr(), nbytes)
            assert found is not None, f"{name}: mmap view not resolved"
            assert os.path.samefile(found[0], path)
            with open(path, "rb") as fh:
                fh.seek(found[1])
                assert fh.read(nbytes) == raw(want), f"{name}: wrong file range"
            for rank in range(TP):
                shard = view[rank * 2 : rank * 2 + 2]
                dst = torch.empty(shard.shape, dtype=shard.dtype)
                fd, _ = gb10_load._backing_file(shard.view(-1), dst.numel() * dst.element_size())
                assert fd >= 0, f"{name} rank {rank}: not read with pread"
                gb10_load.copy_from_checkpoint_(dst, shard)
                assert raw(dst) == raw(want[rank * 2 : rank * 2 + 2]), (name, rank)
                # Each rank slice spans several staging chunks.
                assert dst.numel() * dst.element_size() > gb10_load._CHUNK_BYTES


def test_mapping_split_into_several_vmas(tmp_path: Path) -> None:
    small_chunks()
    want = random_tensor((1 << 20,), torch.uint8, 5)
    path = tmp_path / "split.safetensors"
    write_checkpoint(path, {"w": want})
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    with safe_open(str(path), framework="pt") as f:
        view = f.get_tensor("w")
        page = mmap.PAGESIZE
        middle = (view.data_ptr() + view.numel() // 2) // page * page
        assert libc.madvise(middle, page, mmap.MADV_RANDOM) == 0, ctypes.get_errno()
        found = gb10_load._file_range(view.data_ptr(), view.numel())
        assert found is not None, "range over a split mapping not resolved"
        dst = torch.empty_like(want)
        gb10_load.copy_from_checkpoint_(dst, view)
        assert raw(dst) == raw(want)


def test_fallbacks_stay_exact(tmp_path: Path) -> None:
    small_chunks()
    want = random_tensor((6, 50, 70), torch.uint8, 6)

    # Anonymous memory (eager load strategy): no file, copied from the view.
    assert gb10_load._file_range(want.data_ptr(), want.numel()) is None
    dst = torch.empty_like(want)
    gb10_load.copy_from_checkpoint_(dst, want)
    assert raw(dst) == raw(want)

    # Non-contiguous source, dtype conversion, tiny tensor: CPU clone path.
    dst = torch.empty(70, 50, 6, dtype=torch.uint8)
    gb10_load.copy_from_checkpoint_(dst, want.permute(2, 1, 0))
    assert torch.equal(dst, want.permute(2, 1, 0))
    dst = torch.empty(want.shape, dtype=torch.float32)
    gb10_load.copy_from_checkpoint_(dst, want)
    assert torch.equal(dst, want.float())
    dst = torch.empty(3, dtype=torch.float32)
    gb10_load.copy_from_checkpoint_(dst, torch.tensor([1.0, 2.0, 3.0]))
    assert dst.tolist() == [1.0, 2.0, 3.0]

    # A file that no longer matches its mapping must not be trusted.
    path = tmp_path / "stale.safetensors"
    write_checkpoint(path, {"w": want})
    with safe_open(str(path), framework="pt") as f:
        view = f.get_tensor("w")
        found = gb10_load._file_range(view.data_ptr(), view.numel())
        os.truncate(path, found[1] + 16)
        fd, _ = gb10_load._backing_file(view.view(-1), view.numel())
        assert fd < 0, "short file accepted"

    # Staging disabled: plain copy_.
    gb10_load._MODE = "0"
    try:
        dst = torch.empty_like(want)
        gb10_load.copy_from_checkpoint_(dst, want)
        assert raw(dst) == raw(want)
    finally:
        gb10_load._MODE = "force"


def _stub_vllm(root: Path, rank_holder: dict) -> None:
    """A fake ``vllm`` package holding the real patched moe.py, gb10_load.py
    and trtllm_bmm.py, with every other import stubbed."""
    pkg = root / "vllm" / "models" / "inkling2"
    (pkg / "ops").mkdir(parents=True)
    for d in (root / "vllm", root / "vllm" / "models", pkg, pkg / "ops"):
        (d / "__init__.py").write_text("")
    shutil.copy(PATCHES / "moe.py", pkg / "moe.py")
    for name in ("gb10_load.py", "trtllm_bmm.py"):
        shutil.copy(PATCHES / name, pkg / "ops" / name)

    def module(name: str, **attrs) -> None:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

    def set_weight_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)

    def unused(*args, **kwargs):
        raise AssertionError("not part of weight loading")

    module("vllm.envs", VLLM_MOE_SKIP_PADDING=False)
    module(
        "vllm.distributed",
        get_tensor_model_parallel_rank=lambda: rank_holder["rank"],
        get_tensor_model_parallel_world_size=lambda: TP,
        tensor_model_parallel_all_reduce=unused,
    )
    module("vllm.forward_context", get_forward_context=unused, is_forward_context_available=unused)
    module("vllm.model_executor")
    module("vllm.model_executor.model_loader")
    module(
        "vllm.model_executor.model_loader.weight_utils",
        default_weight_loader=lambda param, loaded: param.data.copy_(loaded),
    )
    module("vllm.model_executor.utils", set_weight_attrs=set_weight_attrs)
    ops = "vllm.models.inkling2.ops"
    module(f"{ops}.activation", swiglu_quantize=unused)
    module(f"{ops}.block_scales", shuffle_rows=unused, swizzle_block_scales=unused)
    module(f"{ops}.moe", NUM_COUNTERS=4, combine=unused, combine_push=unused, route=unused)
    module(f"{ops}.seq_parallel", SeqParallel=object)
    sys.path.insert(0, str(root))


def test_patched_expert_loaders_end_to_end(tmp_path: Path) -> None:
    """Every rank's Inkling2Experts parameters equal its slice of the
    checkpoint, for the NVFP4 and the MXFP8 layout."""
    rank_holder = {"rank": 0}
    _stub_vllm(tmp_path / "site", rank_holder)
    moe = importlib.import_module("vllm.models.inkling2.moe")
    staged = sys.modules["vllm.models.inkling2.ops.gb10_load"]
    staged._CHUNK_BYTES, staged._SMALL_BYTES = 100_003, 4096

    experts, hidden, inter = 8, 512, 256
    config = types.SimpleNamespace(
        n_routed_experts=experts,
        num_experts_per_tok=2,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        hidden_size=hidden,
        intermediate_size=inter,
    )
    prefix = "model.llm.layers.0.mlp.experts."
    for quant in ("nvfp4", "mxfp8"):
        rank_holder["rank"] = 0
        shapes = dict(moe.Inkling2Experts(config, quant).named_parameters())
        tensors = {}
        for seed, (name, param) in enumerate(shapes.items()):
            shape = tuple(param.shape)
            if name.endswith("_weight") or name.endswith("_weight_scale") or name.endswith("_amax"):
                shape = (experts, *shape[1:])
            tensors[prefix + name] = random_tensor(shape, param.dtype, 100 + seed)
        path = tmp_path / f"{quant}.safetensors"
        write_checkpoint(path, tensors)
        want = load_file(str(path))

        big = 0
        for rank in range(TP):
            rank_holder["rank"] = rank
            module = moe.Inkling2Experts(config, quant)
            lo, hi = rank * experts // TP, (rank + 1) * experts // TP
            before = staged._stats["tensors"]
            with safe_open(str(path), framework="pt") as f:
                for name, param in module.named_parameters():
                    param.weight_loader(param, f.get_tensor(prefix + name))
            big += staged._stats["tensors"] - before
            for name, param in module.named_parameters():
                full = want[prefix + name]
                expect = full if full.shape == param.shape else full[lo:hi]
                assert expect.shape == param.shape, (quant, name)
                assert raw(param.data) == raw(expect), (quant, rank, name)
        # w13/w2 weights and block scales, per rank, went through staging.
        assert big == 4 * TP, (quant, big)


def main() -> None:
    assert sys.platform == "linux", "needs /proc/self/maps"
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed (torch {torch.__version__}, page {mmap.PAGESIZE})")


if __name__ == "__main__":
    main()
