"""第 14 章：草稿头 + 投机解码，和普通贪心对答案、比速度"""
import os
import time

from qwen.chat import Chat, build_prompt
from qwen.model import Model
from qwen.mtp import MTP, generate_mtp
from qwen.sample import generate
from qwen.weights import find_model

import torch
torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
mtp = MTP.from_pretrained(model.cfg, path)
chat = Chat(model, path)
print(f"草稿头 {sum(p.numel() for p in mtp.parameters()) / 1e6:.1f}M 参数：{[k for k, _ in mtp.named_children()]}")

for q in ("用一句话介绍一下 Transformer。", "写一段 200 字的短文，介绍长江。"):
    ids = chat.encode(build_prompt([{"role": "user", "content": q}]))
    t = time.time()
    plain = generate(model, ids, max_new_tokens=120, stop=chat.stop, temperature=0)
    t_plain = time.time() - t
    t = time.time()
    spec, rounds, hits = generate_mtp(model, mtp, ids, max_new_tokens=120, stop=chat.stop)
    t_spec = time.time() - t
    print(f"\n{q}")
    print(f"  普通贪心：{len(plain)} 个 token，{t_plain:.2f} 秒（{len(plain) / t_plain:.1f} tok/s）")
    print(f"  投机解码：{len(spec)} 个 token，{t_spec:.2f} 秒（{len(spec) / t_spec:.1f} tok/s）；验证 {rounds} 轮，猜中 {hits} 次（{hits / rounds:.0%}）")
    print(f"  两者逐 token 一致：{spec[:len(plain)] == plain[:len(spec)]}")
    print(f"  {chat.tk.decode(spec)[:60]}…")
