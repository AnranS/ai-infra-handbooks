import os
import subprocess
import sys

import torch
from minisgl.moe.torch_backend import TorchMoeBackend

from conftest import ROOT, hf_greedy
from helpers import build_llm, greedy


def test_torch_moe_matches_dense_loop():
    torch.manual_seed(0)
    T, H, I, E, k = 9, 16, 8, 4, 2
    x, g = torch.randn(T, H), torch.randn(T, E)
    w1, w2 = torch.randn(E, 2 * I, H), torch.randn(E, H, I)
    out = TorchMoeBackend().forward(x, w1, w2, g, k, renormalize=True)
    probs = torch.softmax(g, -1)
    wts, ids = probs.topk(k)
    wts = wts / wts.sum(-1, keepdim=True)
    ref = torch.zeros_like(x)
    for t in range(T):  # 最直白的写法：逐 token、逐专家
        for j in range(k):
            h = x[t] @ w1[ids[t, j]].t()
            ref[t] += wts[t, j] * (torch.nn.functional.silu(h[:I]) * h[I:]) @ w2[ids[t, j]].t()
    assert torch.allclose(out, ref, atol=1e-4)


def test_fused_triton_moe_in_interpreter():
    code = """
import torch
from minisgl.moe.fused import FusedMoeBackend, moe_align_block_size
from minisgl.moe.torch_backend import TorchMoeBackend
torch.manual_seed(0)
x, g = torch.randn(37, 64), torch.randn(37, 8)
w1, w2 = torch.randn(8, 96, 64) * 0.1, torch.randn(8, 64, 48) * 0.1
a = FusedMoeBackend().forward(x, w1, w2, g, 2, True)
b = TorchMoeBackend().forward(x, w1, w2, g, 2, True)
print((a - b).abs().max().item())
"""
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(ROOT / "python"))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert float(r.stdout.strip()) < 1e-5


def test_tiny_qwen3_moe_matches_hf(tiny_models):
    path = tiny_models["qwen3moe"]
    llm = build_llm(path)
    prompts = ["hello world", "The quick brown fox"]
    out = llm.generate(prompts, greedy(10))
    assert [o["token_ids"] for o in out] == hf_greedy(path, prompts, 10)
    llm.shutdown()


def test_tiny_llama3_rope_scaling_matches_hf(tiny_models):
    path = tiny_models["llama3"]
    llm = build_llm(path)
    out = llm.generate(["hello world"], greedy(10))
    assert out[0]["token_ids"] == hf_greedy(path, ["hello world"], 10)[0]
    llm.shutdown()
