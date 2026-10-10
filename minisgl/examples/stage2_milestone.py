"""阶段二的里程碑：调度器接管了阶段一里手工填 batch 的那几行。

三件事：离线接口一次吞下 8 个请求（连续批处理、准入、页分配全自动）；
同一个系统提示词的 4 个问题，Radix Cache 让 prefill 少算四分之三；
一个长提示词被分块 prefill 切成几步。
"""

import time

import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM

ENV.DISABLE_OVERLAP_SCHEDULING.value = True                   # 先关掉重叠调度，数字只反映调度器本身


def make(**kw):
    return LLM("models/Qwen3-0.6B", dtype=torch.float32, max_running_req=8, num_page_override=2048,
               max_seq_len_override=512, **kw)


def count_prefill(llm):
    """把引擎的前向包一层，数一数 prefill 实际算了多少个 token、分了几步"""
    computed = []
    orig = llm._forward
    llm._forward = lambda fi: (computed.append(sum(r.extend_len for r in fi.batch.reqs))
                               if fi.batch.is_prefill else None, orig(fi))[1]
    return computed


# ------------------------------------------------------------------ 1. 8 个请求，一次 generate
prompts = ["The capital of France is", "List three prime numbers:", "def fibonacci(n):", "Once upon a time,",
           "1 + 1 =", "The quick brown fox", "import torch\n", "Roses are red,"]
llm = make()
t = time.perf_counter()
outs = llm.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True))
dt = time.perf_counter() - t
print(f"8 个请求 x 16 个 token：{dt:.2f} s，合计 {8 * 16 / dt:.1f} tokens/s（阶段一手工组 batch 是 70，第 0 步是 25.6）")
print(f"  第一个：{outs[0]['text']!r}")
llm.shutdown()

# ------------------------------------------------------------------ 2. 同一个系统提示词：Radix Cache
system = "You are a helpful assistant. Answer concisely and accurately. " * 4
questions = ["What is 2 + 2?", "Name a primary color.", "What is the capital of Japan?", "Spell cat."]
for cache_type in ("naive", "radix"):
    llm = make(cache_type=cache_type)
    computed = count_prefill(llm)
    llm.generate([system + questions[0]], SamplingParams(max_tokens=4))          # 第一个请求先进来
    t = time.perf_counter()
    llm.generate([system + q for q in questions[1:]], SamplingParams(max_tokens=4))
    dt = time.perf_counter() - t
    total = sum(len(llm.tokenizer(system + q).input_ids) for q in questions)
    print(f"{cache_type:<5}：4 个提示词共 {total} 个 token，prefill 实际计算 {sum(computed)} 个，后 3 个请求用时 {dt * 1000:.0f} ms")
    llm.shutdown()

# ------------------------------------------------------------------ 3. 长提示词：分块 prefill
long_prompt = "Explain, step by step, why the sky is blue. " * 25
llm = make(max_extend_tokens=96)                              # 每一步 prefill 最多 96 个 token
computed = count_prefill(llm)
llm.generate([long_prompt], SamplingParams(max_tokens=2))
print(f"分块 prefill：{len(llm.tokenizer(long_prompt).input_ids)} 个 token 的提示词分了 {len(computed)} 步，"
      f"每步 {computed}；别的请求可以插在这些步之间")
llm.shutdown()
