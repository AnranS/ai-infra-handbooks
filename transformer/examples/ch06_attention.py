"""第 6 章：门控注意力，和 HF 的 self_attn 对答案；看一眼输出门和因果掩码"""
import os

import torch
from transformers import AutoModelForCausalLM

from qwen.attention import Attention
from qwen.rope import rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()                   # 两个张量逐元素的最大差
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()   # 参考实现：只拿它的同名模块对答案

attn = load(Attention(cfg), W, "layers.3.self_attn.")                                          # 第 3 层是第一层全注意力
torch.manual_seed(0)
T = 6
x = torch.randn(1, T, cfg.hidden_size)
cos, sin = rope_tables(cfg, torch.arange(T))
mine = attn(x, cos, sin)
causal = torch.full((T, T), float("-inf")).triu(1)[None, None]                                 # HF 的 eager 实现要加法掩码：看不到的位置加 -inf
ref = hf.model.layers[3].self_attn(hidden_states=x, position_embeddings=(cos[None], sin[None]), attention_mask=causal)[0]
print(f"门控注意力和 HF 的 layers[3].self_attn 最大差 {diff(mine, ref):.1e}")
n = sum(p.numel() for p in attn.parameters())
print(f"一层注意力 {n / 1e6:.1f}M 参数，其中 q_proj {attn.q_proj.weight.numel() / 1e6:.1f}M（一半是门）")

with torch.no_grad():                                                                          # 把门拿出来看
    q, gate = attn.q_proj(x).view(1, T, attn.n_head, 2 * attn.d).chunk(2, dim=-1)
    g = torch.sigmoid(gate)
    print(f"输出门 sigmoid(gate)：均值 {g.mean():.2f}，小于 0.1 的比例 {(g < 0.1).float().mean():.0%}，大于 0.9 的比例 {(g > 0.9).float().mean():.0%}")
print("因果掩码（1 = 看不到）：")
for row in torch.ones(T, T, dtype=torch.bool).triu(1).int()[:4]:
    print("   ", " ".join(str(v) for v in row.tolist()))
