"""测试的公共设施：模型路径、Hugging Face 参考输出（整个测试会话只算一次）、临时小模型。"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
QWEN3 = str(ROOT / "models" / "Qwen3-0.6B")
QWEN25 = str(ROOT / "tests" / "configs" / "Qwen2.5-0.5B-Instruct")   # 只用来测 Qwen2 的配置解析，仓库里只放 config.json
FAKES = str(ROOT / "tests" / "fakes")

PROMPTS = [
    "The capital of France is",
    "The capital of France is a city that",
    "List three prime numbers:",
    "Once upon a time, in a small village by the sea, there lived an old fisherman who",
    "def fibonacci(n):",
]


@functools.cache
def hf_model(path: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


@functools.cache
def _hf_greedy(path: str, prompt: str, n: int) -> Tuple[int, ...]:
    from transformers import AutoTokenizer

    ids = AutoTokenizer.from_pretrained(path)(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = hf_model(path).generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                      repetition_penalty=1.0)
    return tuple(out[0, ids.shape[1]:].tolist())


def hf_greedy(path: str, prompts: List[str], n: int) -> List[List[int]]:
    """Hugging Face 贪心解码 n 个 token（忽略 EOS），作为所有正确性测试的基准。"""
    return [list(_hf_greedy(path, p, n)) for p in prompts]


@pytest.fixture(autouse=True)
def _reset_global_state():
    """每个测试前后清理进程级全局状态，便于在同一进程里反复建引擎。"""
    from minisgl.core import reset_global_ctx
    from minisgl.distributed import reset_tp_info
    from minisgl.env import ENV

    reset_global_ctx()
    reset_tp_info()
    old = ENV.DISABLE_OVERLAP_SCHEDULING.value
    yield
    ENV.DISABLE_OVERLAP_SCHEDULING.value = old
    reset_global_ctx()
    reset_tp_info()


@pytest.fixture
def fake_kernels(monkeypatch):
    """把 tests/fakes（同接口的 FlashInfer、sgl_kernel 假实现）放到 import 路径上。"""
    monkeypatch.syspath_prepend(FAKES)
    for name in list(sys.modules):
        if name.split(".")[0] in ("flashinfer", "sgl_kernel", "_fake_ref"):
            del sys.modules[name]
    yield
    for name in list(sys.modules):
        if name.split(".")[0] in ("flashinfer", "sgl_kernel", "_fake_ref"):
            del sys.modules[name]


def make_tiny_model(path: Path, kind: str) -> str:
    """用随机权重构造一个两层的小模型并保存（词表复用 Qwen3 的 tokenizer）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, Qwen3MoeConfig

    torch.manual_seed(0)
    common = dict(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    if kind == "llama3":
        cfg = LlamaConfig(**common, max_position_embeddings=16384, rope_theta=500000.0,
                          rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                        "high_freq_factor": 4.0,
                                        "original_max_position_embeddings": 8192})
    elif kind == "qwen3moe":
        cfg = Qwen3MoeConfig(**common, head_dim=64, moe_intermediate_size=128, num_experts=8,
                             num_experts_per_tok=2, norm_topk_prob=True, max_position_embeddings=4096)
    else:
        raise ValueError(kind)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    AutoTokenizer.from_pretrained(QWEN3).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory) -> Dict[str, str]:
    base = tmp_path_factory.mktemp("tiny")
    return {k: make_tiny_model(base / k, k) for k in ("llama3", "qwen3moe")}


@pytest.fixture
def ipc_dir():
    """一个短路径的目录，用来放 ipc:// 的 socket 文件。

    Unix domain socket 的路径存在 sockaddr_un.sun_path 里，长度有硬上限：Linux 108 字节、
    macOS 只有 104。pytest 的 tmp_path 在 macOS 上形如
    /private/var/folders/9k/3m_.../T/pytest-of-you/pytest-12/test_zmq_push_pull_over_ip0/，
    光目录就一百多字节，直接拿它当 socket 路径会被 ZMQ 拒掉。所以这里单独在 /tmp 下开一个。
    """
    import shutil
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="msgl-", dir="/tmp" if Path("/tmp").is_dir() else None))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
