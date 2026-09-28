"""导读：用离线接口跑几个请求，看看最终要做出来的东西长什么样。"""

import time

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

llm = LLM("models/Qwen3-0.6B", dtype=torch.float32, max_running_req=8,
          num_page_override=2048, max_seq_len_override=512)
prompts = ["The capital of France is", "List three prime numbers:", "def fibonacci(n):"]
t = time.perf_counter()
outputs = llm.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True))
elapsed = time.perf_counter() - t
for p, o in zip(prompts, outputs):
    print(f"{p!r} -> {o['text']!r}")
print(f"3 个请求 x 16 个 token，CPU 上用时 {elapsed:.2f} 秒")
llm.shutdown()
