import os
import signal

import httpx
import pytest
from transformers import AutoTokenizer

from conftest import PROMPTS, QWEN3, hf_greedy
from test_ch15_server import start_server


@pytest.mark.parametrize("tp", [2, 4])
def test_tensor_parallel_matches_hf(tp):
    """CPU 上用 gloo 做张量并行：TP=2、TP=4 的贪心输出与单卡 HF 一致。"""
    proc, port = start_server("--tp", str(tp), "--num-pages", "2048")
    try:
        tok = AutoTokenizer.from_pretrained(QWEN3)
        refs = hf_greedy(QWEN3, PROMPTS[:3], 10)
        for prompt, ref in zip(PROMPTS[:3], refs):
            r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                           json={"model": "m", "prompt": prompt, "max_tokens": 10, "temperature": 0,
                                 "ignore_eos": True}, timeout=120).json()
            assert r["choices"][0]["message"]["content"] == tok.decode(ref)
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
