"""第 3 章：换掉 24 层 × 2 + 1 处 RMSNorm；看看零中心权重长什么样"""
import os

import torch

from qwen.embedding import Embedding, LMHead
from qwen.harness import load_hf, logits, max_diff, status
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

w = W["layers.0.input_layernorm.weight"]
print(f"第 0 层输入归一化的权重：均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 存的是 w-1，所以在 0 附近")
w = W["norm.weight"]
print(f"最后一层的归一化权重：  均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 它把尺度调到嵌入表的尺度")

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")                  # 这一章：最后的那一处 + 每层两处
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
print(f"换掉 49 处 RMSNorm 后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

torch.manual_seed(0)
x = torch.randn(2, 3, cfg.hidden_size) * 5                        # 归一化之后每个向量的均方根都是 1（权重为 0 时）
y = RMSNorm(cfg.hidden_size)(x)
print("归一化前每个向量的均方根：", [f"{v:.2f}" for v in x.pow(2).mean(-1).sqrt().flatten()[:3]],
      "归一化后：", [f"{v:.2f}" for v in y.pow(2).mean(-1).sqrt().flatten()[:3]])
