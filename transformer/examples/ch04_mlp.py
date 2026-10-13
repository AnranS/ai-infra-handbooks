"""第 4 章：前馈网络。看它的参数量、闸门开了多少、以及"逐 token"意味着什么"""
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from qwen.embedding import Embedding
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
mlp = load(MLP(cfg), W, "layers.0.mlp.")
n = sum(p.numel() for p in mlp.parameters())
print(f"一层前馈 {n / 1e6:.2f}M 参数（3 × 1024 × 3584），24 层 {24 * n / 1e6:.0f}M，占全模型 {24 * n / sum(v.numel() for v in W.values()):.0%}")

tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # 用第 2、3 章的嵌入层和归一化造出第 0 层前馈的真实输入
ids = torch.tensor([tk.encode("今天天气不错，我们去公园").ids])
h = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.post_attention_layernorm.")(load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")(ids))
y = mlp(h)
print(f"输入 {tuple(h.shape)} -> 中间 {tuple(mlp.up_proj(h).shape)} -> 输出 {tuple(y.shape)}")

with torch.no_grad():                                               # 1. 闸门开了多少：silu(gate) 明显大于 0 的比例
    for i in (0, 11, 23):
        g = F.silu(load(MLP(cfg), W, f"layers.{i}.mlp.").gate_proj(h))
        print(f"  第 {i:2d} 层：silu(gate) > 0.1 的比例 {(g > 0.1).float().mean():.0%}，< -0.1 的比例 {(g < -0.1).float().mean():.0%}（silu 最小约 -0.28）")

perm = torch.tensor([5, 3, 1, 0, 2, 4])                             # 2. 逐 token：把 token 顺序打乱，输出也只是跟着打乱——它不看邻居
print(f"打乱 token 顺序后再过前馈，和先过前馈再打乱的差：{diff(mlp(h[:, perm]), y[:, perm]):.1e}（token 之间没有交互）")
print(f"只改第 2 个 token 的输入，其他位置的输出变了多少：{diff(mlp(h + (torch.arange(6) == 2)[None, :, None] * 1.0)[:, [0, 1, 3, 4, 5]], y[:, [0, 1, 3, 4, 5]]):.1e}")
