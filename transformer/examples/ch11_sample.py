"""第 11 章：看一个位置的分布，比较几种采样参数，演示惩罚"""
import os

import torch
from tokenizers import Tokenizer

from qwen.cache import State
from qwen.model import Model
from qwen.sample import generate, sample
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
tk = Tokenizer.from_file(str(path / "tokenizer.json"))
STOP = (tk.token_to_id("<|im_end|>"), tk.token_to_id("<|endoftext|>"))
prompt = "<|im_start|>user\n写一句关于秋天的话。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = torch.tensor([tk.encode(prompt).ids])

logits = model(ids, State(model))[0, -1]                              # 1. 第一个 token 的分布长什么样
for T in (0.5, 1.0, 2.0):
    p = torch.softmax(logits / T, dim=-1)
    top = p.topk(5)
    print(f"温度 {T}：top-5 概率 {[f'{v:.2f}' for v in top.values.tolist()]}，"
          f"{[tk.decode([i]) for i in top.indices.tolist()]}，累计到 0.9 要 {int((p.sort(descending=True).values.cumsum(0) < 0.9).sum()) + 1} 个候选")

torch.manual_seed(0)                                                  # 2. 几种采样参数各生成一句
for kw in (dict(temperature=0), dict(temperature=1.0, top_k=20, top_p=1.0, presence_penalty=2.0),
           dict(temperature=0.7, top_p=0.8, top_k=20), dict(temperature=2.0)):
    out = generate(model, ids, max_new_tokens=50, stop=STOP, **kw)
    print(f"{str(kw):<75} {tk.decode(out)!r}")

prompt2 = "<|im_start|>user\n把“好”重复写 50 次。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # 3. 惩罚：故意诱导重复
ids2 = torch.tensor([tk.encode(prompt2).ids])
for kw in (dict(), dict(repetition_penalty=1.3), dict(presence_penalty=2.0)):
    out = generate(model, ids2, max_new_tokens=30, stop=STOP, temperature=0, **kw)
    print(f"贪心 + {str(kw):<32} {tk.decode(out)!r}")
