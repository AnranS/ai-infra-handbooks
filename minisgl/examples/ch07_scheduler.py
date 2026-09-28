"""第 7 章：离线接口跑 5 个请求，打印每一轮调度出的 batch。

为了只看"骨架"，关掉重叠调度、用 naive 缓存（不复用前缀）。
"""

import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

ENV.DISABLE_OVERLAP_SCHEDULING.value = True
path = "models/Qwen3-0.6B"
llm = LLM(path, dtype=torch.float32, max_running_req=8, num_page_override=1024,
          max_seq_len_override=512, cache_type="naive")
orig = llm._forward


def traced(fi):
    b = fi.batch
    detail = ", ".join(f"uid{r.uid}:{r.extend_len}" for r in b.reqs)
    print(f"  {b.phase:<7} bs={b.size}  token 数={len(b.positions):<3} [{detail}]")
    return orig(fi)


llm._forward = traced
prompts = ["The capital of France is", "List three prime numbers:", "def fibonacci(n):",
           "1 + 1 =", "Once upon a time,"]
lengths = [6, 2, 5, 1, 4]  # 每个请求的 max_tokens 不同：短的先结束离开
print("调度轨迹：")
outputs = llm.generate(prompts, [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths])
tok = AutoTokenizer.from_pretrained(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
for p, n, o in zip(prompts, lengths, outputs):
    ids = tok(p, return_tensors="pt").input_ids
    ref = hf.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)[0, ids.shape[1]:].tolist()
    print(f"{p!r:<30} -> {o['text']!r:<32} 与 HF 一致: {o['token_ids'] == ref}")
llm.shutdown()
