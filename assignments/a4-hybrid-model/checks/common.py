"""各项检查共用的设施：模型路径、Hugging Face 的参考输出、结果汇报。

被检查的是你自己的 mini-sglang：把它的 python/ 目录放进 PYTHONPATH（run_checks.py 默认用环境变量 MINISGL，
没有设置时用本仓库的 minisgl/python——那是没有接入 Qwen3.5 的版本，检查应当失败）。
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = os.environ.get("A4_MODEL", str(ROOT / "models" / "Qwen3.5-0.8B"))
QWEN3 = os.environ.get("A4_QWEN3", str(ROOT / "models" / "Qwen3-0.6B"))

PROMPTS = [
    "The capital of France is",
    "List three prime numbers:",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea, there lived an old fisherman who",
    "线性注意力用一个固定大小的状态汇总全部历史，",
]
LONG_PROMPT = ("Hybrid models keep a fixed-size state for most layers and a growing KV cache for a few. "
               "An inference engine therefore needs two memory pools, and every feature that was designed "
               "around the KV cache has to be revisited: prefix caching, speculative decoding, and "
               "prefill-decode disaggregation. Summarize the trade-offs:")


@functools.cache
def _hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return tok, model


@functools.cache
def _hf_one(prompt: str, n: int) -> tuple:
    tok, model = _hf()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    return tuple(out[0, ids.shape[1]:].tolist())


def hf_greedy(prompts: List[str], n: int) -> List[List[int]]:
    """Hugging Face transformers 在 float32 下贪心解码 n 个 token（忽略 EOS），作为标准答案"""
    return [list(_hf_one(p, n)) for p in prompts]


def minisgl_generate(prompts: List[str], n: int, **kwargs) -> List[List[int]]:
    """用你的 mini-sglang（离线接口 LLM）在 CPU、float32 下贪心解码 n 个 token"""
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    kwargs = {"dtype": torch.float32, "cache_type": "naive", "max_running_req": 8, "device": "cpu", **kwargs}
    llm = LLM(MODEL, **kwargs)
    try:
        out = llm.generate(prompts, SamplingParams(max_tokens=n, ignore_eos=True))
    finally:
        llm.shutdown()
    return [o["token_ids"] for o in out]


def first_diff(a: List[int], b: List[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    sys.exit(0 if ok else 1)
