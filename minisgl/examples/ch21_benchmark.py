"""第 21 章：在 CPU 上做两个消融——连续批处理的批大小，以及前缀缓存。

CPU 上的绝对数字没有参考价值，但趋势与 GPU 上一致。GPU 上的测法见正文。
"""

import time
from random import randint, seed

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

PATH = "models/Qwen3-0.6B"
seed(0)
prompts = [[randint(100, 10000) for _ in range(randint(32, 64))] for _ in range(16)]
params = SamplingParams(max_tokens=32, ignore_eos=True, temperature=0.6)

print("一、最大并发请求数（16 个请求，输入 32～64、输出 32 个 token）")
for max_running in (1, 4, 16):
    llm = LLM(PATH, dtype=torch.bfloat16, max_running_req=max_running, num_page_override=4096,
              max_seq_len_override=256)
    llm.generate([[1, 2, 3]], SamplingParams(max_tokens=2))  # 预热
    t = time.perf_counter()
    out = llm.generate(prompts, params)
    dt = time.perf_counter() - t
    n = sum(len(o["token_ids"]) for o in out)
    print(f"  max_running_req={max_running:<3} 用时 {dt:6.2f}s  输出吞吐 {n / dt:7.1f} token/s")
    llm.shutdown()

print("二、前缀缓存（16 个请求共享 400 个 token 的前缀，各自再加 16 个 token）")
shared = [randint(100, 10000) for _ in range(400)]
prompts2 = [shared + [randint(100, 10000) for _ in range(16)] for _ in range(16)]
for cache in ("naive", "radix"):
    llm = LLM(PATH, dtype=torch.bfloat16, max_running_req=16, num_page_override=16384,
              max_seq_len_override=512, cache_type=cache)
    llm.generate([[1, 2, 3]], SamplingParams(max_tokens=2))
    computed = []
    orig = llm._forward
    llm._forward = lambda fi: (computed.append(len(fi.batch.positions)) if fi.batch.is_prefill else None,
                               orig(fi))[1]
    t = time.perf_counter()
    llm.generate(prompts2[:1], SamplingParams(max_tokens=1))  # 第一个请求先把前缀算好
    llm.generate(prompts2[1:], SamplingParams(max_tokens=8, ignore_eos=True))
    dt = time.perf_counter() - t
    print(f"  {cache:<5} prefill 计算 {sum(computed):>5} 个 token，总用时 {dt:5.2f}s")
    llm.shutdown()
