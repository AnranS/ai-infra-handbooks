"""第 10 章：prefill 一次 + 逐 token 解码；和无状态重算逐位对；算两种状态的大小"""
import os
import time

import torch
from tokenizers import Tokenizer

from qwen.cache import State
from qwen.model import Model
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
max_diff = lambda a, b: (a - b).abs().max().item()
path = find_model()
model = Model.from_pretrained(path)
tk = Tokenizer.from_file(str(path / "tokenizer.json"))
prompt = "<|im_start|>user\n用一句话介绍一下 Transformer。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = torch.tensor([tk.encode(prompt).ids])
STOP = {tk.token_to_id("<|im_end|>"), tk.token_to_id("<|endoftext|>")}

state = State(model)                                                 # 1. prefill：整段提示词一次算完，状态留下
t = time.time()
nxt = model(ids, state)[0, -1].argmax()
t_prefill = time.time() - t
new = [nxt.item()]
t = time.time()
while len(new) < 40 and new[-1] not in STOP:                         # 2. decode：每步只喂一个 token
    nxt = model(nxt.view(1, 1), state)[0, -1].argmax()
    new.append(nxt.item())
dt = time.time() - t
print(f"prefill {ids.shape[1]} 个 token {t_prefill * 1000:.0f} ms；decode {len(new)} 个 token {dt:.2f} 秒 = {len(new) / dt:.1f} tok/s（第 9 章是 2 tok/s）")
print(tk.decode(new))
print(f"状态：已处理 {state.pos} 个 token，占 {state.megabytes():.1f} MB")

full = torch.tensor([ids[0].tolist() + new])                         # 3. 带状态算出来的 logits 和无状态重算的一样吗
ref = model(full)[0]
state = State(model)
got = [model(full[:, :ids.shape[1]], state)[0, -1]]                  # 先 prefill 提示词，再一个一个喂生成的 token
for j in range(ids.shape[1], full.shape[1] - 1):
    got.append(model(full[:, j:j + 1], state)[0, -1])
print(f"逐 token 解码 vs 整段重算：logits 最大差 {max_diff(torch.stack(got), ref[ids.shape[1] - 1:-1]):.1e}")
state = State(model)                                                 # 4. 分几段 prefill（多轮对话就是这样）
a, b = model(full[:, :10], state)[0], model(full[:, 10:30], state)[0]
c = model(full[:, 30:], state)[0]
print(f"分三段喂（10 + 20 + {full.shape[1] - 30}）vs 整段：logits 最大差 {max_diff(torch.cat([a, b, c]), ref):.1e}")

cfg = model.cfg                                                      # 5. 两种状态的大小（fp32）
kv_per_token = cfg.layer_types.count("full_attention") * 2 * cfg.num_key_value_heads * cfg.head_dim * 4
fixed = cfg.layer_types.count("linear_attention") * (cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
                                                      + (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim) * 3) * 4
print(f"每个 token 的 KV cache：6 层 × 2 × 2 组 × 256 维 × 4 B = {kv_per_token / 1024:.0f} KB；18 层线性注意力的状态固定 {fixed / 2 ** 20:.1f} MB")
for T in (1000, 10_000, 262_144):
    print(f"  上下文 {T:>7} 个 token：KV cache {kv_per_token * T / 2 ** 30:6.2f} GB，如果 24 层全是注意力则是 {kv_per_token * T * 4 / 2 ** 30:6.2f} GB")
