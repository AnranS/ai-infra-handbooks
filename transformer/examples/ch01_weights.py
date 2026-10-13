"""第 1 章：读配置和权重，按模块算一笔参数账，看看权重长什么样"""
import os
import re
from collections import OrderedDict

import torch

from qwen.weights import Config, find_model, load_weights, sub

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
print(f"{cfg.num_hidden_layers} 层，hidden {cfg.hidden_size}，词表 {cfg.vocab_size}，RoPE theta {cfg.rope_theta:.0e}，只旋转 {cfg.partial_rotary_factor:.0%} 的维度")
print("层的排法：", "".join("A" if t == "full_attention" else "L" for t in cfg.layer_types))

W = load_weights(path)
total = sum(v.numel() for v in W.values())
print(f"语言模型 {len(W)} 个张量，{total / 1e6:.1f}M 参数；文件里是 bf16（{total * 2 / 2**30:.2f} GB），读进内存转成 fp32（{total * 4 / 2**30:.2f} GB）")
print("lm_head.weight 在文件里吗：", "lm_head.weight" in W, "；tie_word_embeddings =", cfg.tie_word_embeddings, "（输出层就是嵌入表）")

groups = OrderedDict()                                           # 1. 按模块算参数账：名字里的层号换成 N
for k, v in W.items():
    groups.setdefault(re.sub(r"\.\d+\.", ".N.", k).rsplit(".", 1)[0], []).append(v.numel())
print("模块                             张量数  参数量   占比")
for name, sizes in groups.items():
    print(f"  {name:<30} ×{len(sizes):<3} {sum(sizes) / 1e6:7.1f}M {sum(sizes) / total:5.1%}")

print("第 0 层（线性注意力）：", sorted(sub(W, "layers.0.")))                            # 2. 一层里有什么：名字的前缀就是模块
print("第 3 层（全注意力）：  ", sorted(sub(W, "layers.3.")))

for name in ("embed_tokens.weight", "layers.0.mlp.gate_proj.weight", "layers.3.self_attn.q_proj.weight", "norm.weight", "layers.0.linear_attn.A_log"):
    v = W[name]                                                   # 3. 权重的数值长什么样：矩阵都很小、归一化的权重不是
    print(f"  {name:<36} {str(tuple(v.shape)):<16} 均值 {v.mean():+.4f}  标准差 {v.std():.4f}  范围 [{v.min():+.2f}, {v.max():+.2f}]")
