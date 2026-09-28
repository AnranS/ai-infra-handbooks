import subprocess
import sys

import torch
from minisgl.kernel import torch_ops

from conftest import ROOT


def test_torch_store_cache_semantics():
    k_cache, v_cache = torch.zeros(6, 2, 3), torch.zeros(6, 2, 3)
    k = torch.arange(12.0).view(2, 6)
    torch_ops.store_cache(k_cache, v_cache, torch.tensor([4, 1], dtype=torch.int32), k, -k)
    assert torch.equal(k_cache[4].flatten(), k[0]) and torch.equal(v_cache[1].flatten(), -k[1])


def test_fast_compare_key():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    assert torch_ops.fast_compare_key(a, torch.tensor([1, 2, 9], dtype=torch.int32)) == 2
    assert torch_ops.fast_compare_key(a, a[:2]) == 2


def test_cuda_kernels_on_cpu_emulator():
    """CUDA kernel 的自检程序在 CUDA 手册的 CPU 模拟器上运行（没有 GPU 也能验证 kernel 逻辑）。"""
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "emu_kernels.py")],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("PASS") == 4
