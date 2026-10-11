"""第 7 章：换掉 18 层线性注意力（逐 token 递推版）；看一眼衰减、β 和记忆表"""
import os
import time

import torch

from qwen.attention import Attention
from qwen.deltanet import GatedDeltaNet, delta_rule
from qwen.embedding import Embedding, LMHead
from qwen.harness import AttentionAdapter, LinearAttentionAdapter, RotaryAdapter, load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.rope import rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2–6 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
    if cfg.layer_types[i] == "full_attention":
        layer.self_attn = AttentionAdapter(load(Attention(cfg), W, f"layers.{i}.self_attn."))
    else:                                                                                      # 这一章：18 层
        layer.linear_attn = LinearAttentionAdapter(load(GatedDeltaNet(cfg), W, f"layers.{i}.linear_attn."))
t = time.time()
mine = logits(hf, ids)
print(f"换掉 18 层线性注意力后，logits 最大差 {max_diff(ref, mine):.1e}（前向 {(time.time() - t) * 1000:.0f} ms）")
print(status(hf))

dn = hf.model.layers[0].linear_attn.mine
n = sum(p.numel() for p in dn.parameters())
print(f"一层线性注意力 {n / 1e6:.1f}M 参数：in_proj_qkv {dn.in_proj_qkv.weight.numel() / 1e6:.1f}M，conv1d 只有 {dn.conv1d.weight.numel()} 个")
print(f"每个头固定的遗忘速度 exp(A_log)：最小 {dn.A_log.exp().min():.2f}，最大 {dn.A_log.exp().max():.2f}（越大忘得越快）")

with torch.no_grad():                                               # 真实输入下的衰减和 β
    x = hf.model.layers[0].input_layernorm(hf.model.embed_tokens(ids))
    beta = dn.in_proj_b(x).sigmoid()
    g = -dn.A_log.exp() * torch.nn.functional.softplus(dn.in_proj_a(x) + dn.dt_bias)
    print(f"第 0 层对这 6 个 token：衰减 exp(g) 的范围 [{g.exp().min():.2f}, {g.exp().max():.2f}]，β 的范围 [{beta.min():.2f}, {beta.max():.2f}]")

torch.manual_seed(0)                                                # delta 规则的性质：同一对 (key, value) 反复写入，读出来的值收敛到 v；普通线性注意力会越积越大
k = torch.nn.functional.normalize(torch.randn(1, 1, 1, 128), dim=-1).expand(1, 6, 1, 128)
v = torch.randn(1, 1, 1, 128).expand(1, 6, 1, 128)
out, S = delta_rule(k, k, v, torch.zeros(1, 6, 1), torch.full((1, 6, 1), 0.5))   # q = k，无衰减，β = 0.5
plain = torch.cumsum(0.5 * v, dim=1)                                           # 没有"减去预测"的写法：S += β k v^T，读出来是 t × β × v
for t in range(6):
    print(f"  写入 {t + 1} 次后读出的值和 v 的距离：delta 规则 {(out[0, t, 0] * 128 ** 0.5 - v[0, t, 0]).norm():6.3f}，普通线性注意力 {(plain[0, t, 0] - v[0, t, 0]).norm():6.3f}")
