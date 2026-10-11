"""第 13 章：4 个长短不一的请求拼成一批；和单条逐个生成比：结果一样、吞吐量更高"""
import os
import time

import torch

from qwen.batch import generate_batch, pad_left
from qwen.chat import Chat, build_prompt
from qwen.model import Model
from qwen.sample import generate
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
max_diff = lambda a, b: (a - b).abs().max().item()
path = find_model()
model = Model.from_pretrained(path)
chat = Chat(model, path)
questions = ["用一句话介绍一下 Transformer。", "北京在哪个国家？", "写一句关于秋天的话。", "1+1 等于几？"]
seqs = [chat.tk.encode(build_prompt([{"role": "user", "content": q}])).ids for q in questions]
ids, mask = pad_left(seqs)
print(f"4 个请求的长度 {[len(s) for s in seqs]}，拼成 {tuple(ids.shape)}；mask 的第一行：{mask[0].int().tolist()}")

t = time.time()                                                      # 1. 单条逐个生成
singles = [generate(model, torch.tensor([s]), max_new_tokens=60, stop=chat.stop, temperature=0) for s in seqs]
t_single = time.time() - t
t = time.time()                                                      # 2. 拼成一批
batched = generate_batch(model, seqs, max_new_tokens=60, stop=chat.stop, temperature=0)
t_batch = time.time() - t
n = sum(map(len, batched))
print(f"单条：{sum(map(len, singles))} 个 token {t_single:.1f} 秒（{sum(map(len, singles)) / t_single:.1f} tok/s）；"
      f"批处理：{n} 个 token {t_batch:.1f} 秒（{n / t_batch:.1f} tok/s）")
for q, a, b in zip(questions, singles, batched):
    print(f"  {'一致' if a == b else '不同'}  {q} -> {chat.tk.decode(b)!r}")

logits_batch = model(ids, mask=mask)                                 # 3. 批里每一行的 logits 和单独算的一样
for b, s in enumerate(seqs):
    print(f"  第 {b} 行（左边填了 {ids.shape[1] - len(s)} 个）：logits 最大差 {max_diff(logits_batch[b, -len(s):], model(torch.tensor([s]))[0]):.1e}")
