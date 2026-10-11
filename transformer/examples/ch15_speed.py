"""第 15 章：这台机器上能跑多快。线程数、递推 vs 分块、fp32 vs bf16、CPU / MPS / CUDA。"""
import os
import platform
import time

import torch

from qwen.cache import State
from qwen.chat import Chat, build_prompt
from qwen.chunk import chunked_delta_rule
from qwen.deltanet import delta_rule
from qwen.model import Model
from qwen.weights import find_model

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"{platform.machine()} / {device} / torch {torch.__version__} / {os.cpu_count()} 个逻辑核")


def decode_speed(model, ids, n=32):
    state = State(model)
    model(ids, state)
    tok = ids[:, -1:]
    t = time.time()
    for _ in range(n):
        tok = model(tok, state)[:, -1:].argmax(-1)
    if device == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t)


path = find_model()
model = Model.from_pretrained(path)
chat = Chat(model, path)
ids = chat.encode(build_prompt([{"role": "user", "content": "写一段 200 字的短文，介绍长江。"}]))

if device == "cpu":                                                              # 1. 线程数：不是越多越好
    for n in sorted({1, 2, 4, 8, os.cpu_count() // 2, os.cpu_count()}):
        torch.set_num_threads(n)
        print(f"{n:>3} 线程：decode {decode_speed(model, ids):.1f} tok/s")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

torch.manual_seed(0)                                                             # 2. 递推 vs 分块：prefill 一段 512 token
B, T, H, d = 1, 512, 16, 128
q, k, v = (torch.randn(B, T, H, d, device=device) for _ in range(3))
g, beta = -torch.rand(B, T, H, device=device) * 0.5, torch.rand(B, T, H, device=device)
for name, fn in (("递推", delta_rule), ("分块", chunked_delta_rule)):
    t = time.time()
    out, _ = fn(q, k, v, g, beta)
    print(f"一层线性注意力，{T} 个 token：{name} {(time.time() - t) * 1000:.0f} ms")

for dtype in (torch.float32, torch.bfloat16):                                    # 3. fp32 vs bf16
    m = Model.from_pretrained(path, dtype=dtype).to(device)
    x = ids.to(device)
    t = time.time()
    m(x, State(m))
    pre = (time.time() - t) * 1000
    print(f"{str(dtype).split('.')[-1]:>8}：prefill {ids.shape[1]} token {pre:.0f} ms，decode {decode_speed(m, x):.1f} tok/s")
