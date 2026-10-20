"""第 9 章（二）：整模型拼好了，往里看两件事：门在真实输入下开多大、前馈网络里存着什么"""
import math
import os

import torch
from tokenizers import Tokenizer

from qwen.model import Model
from qwen.rope import apply_rope, rope_tables
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path).eval()
tk = Tokenizer.from_file(str(path / "tokenizer.json"))

text = "推理优化的核心是减少访存。大模型在解码阶段每生成一个词，都要读取全部的权重和缓存。"   # 和大模型原理手册注意力一章同一句话
ids = torch.tensor([tk.encode(text).ids])
T = ids.shape[1]
inputs = {}                                                          # 1. 抓 6 个全注意力层的真实输入
for i, layer in enumerate(model.layers):
    if hasattr(layer, "self_attn"):
        layer.self_attn.register_forward_pre_hook(lambda m, args, i=i: inputs.__setitem__(i, args[0]))
with torch.no_grad():
    model(ids)
    cos, sin = rope_tables(model.cfg, torch.arange(T))
    print(f"{T} 个 token。每个全注意力层：门的均值、门值小于 0.1 的比例、后一半 query 平均分给第一个 token 的注意力")
    for i, x in inputs.items():
        a = model.layers[i].self_attn
        q, gate = a.q_proj(x).view(1, T, a.n_head, 2 * a.d).chunk(2, dim=-1)
        k = a.k_proj(x).view(1, T, a.n_kv, a.d)
        q, k = apply_rope(a.q_norm(q), cos, sin), apply_rope(a.k_norm(k), cos, sin)
        q, k = q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(a.n_head // a.n_kv, dim=1)
        s = (q @ k.transpose(-1, -2) / math.sqrt(a.d)).masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        first = torch.softmax(s, dim=-1)[0, :, T // 2:, 0].mean()
        g = torch.sigmoid(gate)
        print(f"  第 {i:2d} 层：门均值 {g.mean():.2f}，小于 0.1 的 {(g < 0.1).float().mean():.0%}，给第一个 token {first:.0%}")

E = model.embed_tokens.weight                                       # 2. 前馈网络里存着什么：输出层和嵌入表共用这份权重
scale = 1 + model.norm.weight                                        # 最后那个零中心 RMSNorm 的逐维缩放
ids = torch.tensor([tk.encode("中国的首都是").ids])
target = tk.encode("北京").ids[0]
acts = {}
for i, layer in enumerate(model.layers):                             # 抓每层前馈网络中间那 3584 个神经元在最后一个位置的激活
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, args, i=i: acts.__setitem__(i, args[0][0, -1]))
with torch.no_grad():
    p = model(ids)[0, -1].softmax(-1)
    print("「中国的首都是」的下一个词，排前三的是", "、".join(f"{tk.decode([t])} {v:.1%}" for v, t in zip(*p.topk(3))))
    u = E[target] * scale                                            # "北京"在输出层上的方向
    best = []
    for L, act in acts.items():                                      # 每个神经元的直接贡献 = 激活 × 它写回残差流的那一列在"北京"方向上的分量
        c = act * (model.layers[L].mlp.down_proj.weight.T @ u)
        j = c.argmax().item()
        best.append((c[j].item(), L, j, act[j].item()))
    best.sort(reverse=True)
    for c, L, j, act in best[:2]:
        col = model.layers[L].mlp.down_proj.weight[:, j] * scale     # 这个神经元写回残差流的那一列，按输出层读成词
        words = list(dict.fromkeys(tk.decode([t]).strip() for t in (E @ col).topk(12).indices.tolist()))[:8]
        print(f"  贡献最大的之一：第 {L} 层第 {j} 号神经元，激活 {act:+.2f}；它写回的方向最像这些词：{' '.join(words)}")
    L, j = best[0][1], best[0][2]
    h = model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, args: (args[0].index_fill(-1, torch.tensor([j]), 0.0),))
    p2 = model(ids)[0, -1].softmax(-1)
    top = "、".join(f"{tk.decode([t])} {v:.1%}" for v, t in zip(*p2.topk(3)))
    print(f"  只把第 {L} 层第 {j} 号神经元关掉：P(北京) = {p2[target]:.1%}，排前三的变成 {top}")
    h.remove()
