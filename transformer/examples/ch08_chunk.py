"""第 8 章：分块版和递推版对答案、比速度；换进骨架再和 HF 对一次"""
import os
import time

import torch

from qwen.attention import Attention
from qwen.chunk import chunked_delta_rule
from qwen.deltanet import GatedDeltaNet, delta_rule
from qwen.embedding import Embedding, LMHead
from qwen.harness import AttentionAdapter, LinearAttentionAdapter, RotaryAdapter, load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.rope import rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.manual_seed(0)
B, H, d = 1, 16, 128
for T in (5, 64, 65, 512, 2048):                                     # 1. 随机输入：两个版本逐位对，顺便计时
    q, k, v = (torch.randn(B, T, H, d) for _ in range(3))
    g, beta = -torch.rand(B, T, H) * 0.5, torch.rand(B, T, H)
    t = time.time()
    o1, S1 = delta_rule(q, k, v, g, beta)
    t1 = time.time() - t
    t = time.time()
    o2, S2 = chunked_delta_rule(q, k, v, g, beta)
    t2 = time.time() - t
    print(f"T={T:4d}：输出最大差 {max_diff(o1, o2):.1e}，末状态最大差 {max_diff(S1, S2):.1e}；递推 {t1 * 1000:6.0f} ms，分块 {t2 * 1000:5.0f} ms")

path = find_model()                                                  # 2. 换进骨架：GatedDeltaNet 里把递推换成分块
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)
hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)
import qwen.deltanet
qwen.deltanet.delta_rule = chunked_delta_rule                        # 第 7 章的模块调的是模块级的 delta_rule，这里换成分块版（第 10 章正式改）
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
    if cfg.layer_types[i] == "full_attention":
        layer.self_attn = AttentionAdapter(load(Attention(cfg), W, f"layers.{i}.self_attn."))
    else:
        layer.linear_attn = LinearAttentionAdapter(load(GatedDeltaNet(cfg), W, f"layers.{i}.linear_attn."))
t = time.time()
mine = logits(hf, ids)
print(f"线性注意力换成分块版后，logits 最大差 {max_diff(ref, mine):.1e}（前向 {(time.time() - t) * 1000:.0f} ms）")
print(status(hf))

long_ids = tok("。".join(["今天天气不错，我们去公园"] * 40), return_tensors="pt")["input_ids"]   # 3. 长一点的输入：分块版的优势
for name, rule in (("递推", delta_rule), ("分块", chunked_delta_rule)):
    qwen.deltanet.delta_rule = rule
    t = time.time()
    out = logits(hf, long_ids)
    print(f"{long_ids.shape[1]} 个 token 的前向，线性注意力用{name}：{(time.time() - t) * 1000:.0f} ms")
print(f"和 HF 在长输入上的最大差：{max_diff(out, logits(load_hf(path)[0], long_ids)):.1e}")
