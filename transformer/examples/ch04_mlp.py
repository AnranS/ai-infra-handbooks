"""第 4 章：前馈网络，和 HF 的 mlp 对答案；量一下它占多少参数、silu 门开了多少"""
import os

import torch
from transformers import AutoModelForCausalLM

import torch.nn.functional as F

from qwen.mlp import MLP
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()                   # 两个张量逐元素的最大差
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # 参考实现：只拿它的同名模块对答案

mlp = load(MLP(cfg), W, "layers.0.mlp.")
torch.manual_seed(0)
x = torch.randn(1, 8, cfg.hidden_size)
print(f"前馈网络和 HF 的 layers[0].mlp 最大差 {diff(mlp(x), hf.model.layers[0].mlp(x)):.1e}")
n = sum(p.numel() for p in mlp.parameters())
print(f"一层前馈 {n / 1e6:.2f}M 参数，24 层 {24 * n / 1e6:.0f}M，占全模型 {24 * n / sum(v.numel() for v in W.values()):.0%}")

with torch.no_grad():                                               # 闸门开了多少：silu(gate) 明显大于 0 的比例
    for i in (0, 11, 23):
        g = F.silu(load(MLP(cfg), W, f"layers.{i}.mlp.").gate_proj(x))
        print(f"  第 {i:2d} 层：silu(gate) > 0.1 的比例 {(g > 0.1).float().mean():.0%}，< -0.1 的比例 {(g < -0.1).float().mean():.0%}（silu 最小约 -0.28）")
