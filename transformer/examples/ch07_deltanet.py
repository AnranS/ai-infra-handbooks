"""第 7 章：线性注意力（逐 token 递推版）。看记忆表的大小不随长度变、衰减和 β 的取值、delta 规则的性质"""
import os
import time

import torch
from tokenizers import Tokenizer

from qwen.deltanet import GatedDeltaNet, delta_rule
from qwen.embedding import Embedding
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
dn = load(GatedDeltaNet(cfg), W, "layers.0.linear_attn.")
n = sum(p.numel() for p in dn.parameters())
print(f"一层线性注意力 {n / 1e6:.1f}M 参数：in_proj_qkv {dn.in_proj_qkv.weight.numel() / 1e6:.1f}M，conv1d 只有 {dn.conv1d.weight.numel()} 个")
print(f"每个头固定的遗忘速度 exp(A_log)：最小 {dn.A_log.exp().min():.2f}，最大 {dn.A_log.exp().max():.2f}（越大忘得越快）")

tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # 用第 2、3 章的零件造出第 0 层的真实输入
emb, norm = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens."), load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")
with torch.no_grad():
    for text in ("今天天气不错，我们去公园", "。".join(["今天天气不错，我们去公园"] * 20)):   # 1. 跑一长一短两段：时间随长度涨，记忆表不涨
        h = norm(emb(torch.tensor([tk.encode(text).ids])))
        t = time.time()
        y = dn(h)
        print(f"  {h.shape[1]:3d} 个 token：输出 {tuple(y.shape)}，{(time.time() - t) * 1000:.0f} ms")
    print(f"记忆表：每个头 [{dn.dk}, {dn.dv}]，{dn.n_v} 个头共 {dn.n_v * dn.dk * dn.dv * 4 / 2**20:.1f} MB，和 token 数无关；全注意力的 KV cache 每个 token 要加 {2 * cfg.num_key_value_heads * cfg.head_dim * 4 / 1024:.0f} KB")
    h = norm(emb(torch.tensor([tk.encode("今天天气不错，我们去公园").ids])))
    beta = dn.in_proj_b(h).sigmoid()                                # 2. 真实输入下的衰减和 β
    g = -dn.A_log.exp() * torch.nn.functional.softplus(dn.in_proj_a(h) + dn.dt_bias)
    print(f"第 0 层对这 6 个 token：衰减 exp(g) 的范围 [{g.exp().min():.2f}, {g.exp().max():.2f}]，β 的范围 [{beta.min():.2f}, {beta.max():.2f}]")

torch.manual_seed(0)                                                # 3. delta 规则的性质：同一对 (key, value) 反复写入，读出来的值收敛到 v；普通线性注意力会越积越大
k = torch.nn.functional.normalize(torch.randn(1, 1, 1, 128), dim=-1).expand(1, 6, 1, 128)
v = torch.randn(1, 1, 1, 128).expand(1, 6, 1, 128)
out, S = delta_rule(k, k, v, torch.zeros(1, 6, 1), torch.full((1, 6, 1), 0.5))   # q = k，无衰减，β = 0.5
plain = torch.cumsum(0.5 * v, dim=1)                                           # 没有"减去预测"的写法：S += β k v^T，读出来是 t × β × v
for t in range(6):
    print(f"  写入 {t + 1} 次后读出的值和 v 的距离：delta 规则 {(out[0, t, 0] * 128 ** 0.5 - v[0, t, 0]).norm():6.3f}，普通线性注意力 {(plain[0, t, 0] - v[0, t, 0]).norm():6.3f}")
