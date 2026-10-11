# （七）Gated DeltaNet：逐 token 递推

<p class="lead">24 层里的 18 层是线性注意力。它没有"所有 token 两两打分"的 $T \times T$ 矩阵，而是每个头维护一张<strong>固定大小的记忆表</strong> $S$（128 × 128）：每来一个 token 就按 delta 规则改写一次——先衰减，再把"新值和表里已有预测之差"写进去；输出是 query 去表里查。token 再多，表也不会变大，这就是 Qwen3.5 能撑 262K 上下文、推理时显存不随长度涨的根本原因。这一章先写最直白的逐 token 递推版，下一章再写快的。</p>

!!! note "这一章跑什么"
    `ch07_deltanet.py`，约 15 秒。

## 先看公式，再看代码

对每个头，记忆表 $S_t \in \mathbb{R}^{128 \times 128}$，第 $t$ 个 token 带来 $k_t, v_t, q_t \in \mathbb{R}^{128}$（都已经 L2 归一化）、衰减 $\alpha_t = e^{g_t} \in (0, 1]$ 和写入强度 $\beta_t \in (0, 1)$：

$$
S_t = \alpha_t S_{t-1} + \beta_t\, k_t\,\big(v_t - S_{t-1}^\top k_t\big)^\top, \qquad o_t = S_t^\top q_t
$$

读它：$S_{t-1}^\top k_t$ 是"表里认为 $k_t$ 对应的值"；$v_t$ 减去它，是这次要修正的**差值**（delta）；乘上 $\beta_t$ 决定改多少；$k_t (\cdot)^\top$ 把这个修正写回 $k_t$ 这一"行"。$\alpha_t$ 让整张表慢慢忘。如果没有减法（$S_t = \alpha_t S_{t-1} + \beta_t k_t v_t^\top$），这就是普通的线性注意力 / RetNet；delta 规则多出来的那一减，让同一个 key 被反复写入时不会越积越大，而是收敛到新值——这是 DeltaNet 比老式线性注意力强的地方。

```python title="qwen/deltanet.py"
"""Gated DeltaNet：线性注意力层，24 层里的 18 层是它。

它没有"所有 token 两两打分"的注意力矩阵，而是每个头维护一张固定大小的记忆表 S（128×128）。
每来一个 token 就按 delta 规则改写一次：先按 g 衰减，再把"新值 v 和表里已有预测之差"按 β 写进去；
输出是 q 去表里查询。token 再多，表也不会变大——这就是它能撑 262K 上下文的原因。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen.norm import GatedRMSNorm, l2norm



def delta_rule(q, k, v, g, beta, S=None):
    """逐 token 递推。q, k: [B, T, 头, dk]，v: [B, T, 头, dv]；g（对数衰减，≤0）和 beta（写入强度，0~1）: [B, T, 头]。
    S 是开始时的记忆表（没有就从空表开始）；返回输出和算完之后的表。"""
    B, T, n_head, dk = q.shape
    dtype = v.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))   # 记忆表和中间量一律 fp32，bf16 攒不住几百步的累加
    q, k = l2norm(q), l2norm(k)                                   # q、k 先归一化到单位长度
    q = q / dk ** 0.5
    if S is None:
        S = q.new_zeros(B, n_head, dk, v.shape[-1])               # 每个头一张 128×128 的记忆表，初始为空
    out = torch.empty_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, :, None, None]                   # 1. 衰减：整张表乘一个 0~1 的数，忘掉一点旧的
        k_t, v_t = k[:, t], v[:, t]                               # [B, 头, dk]，[B, 头, dv]
        pred = torch.einsum("bhk,bhkv->bhv", k_t, S)              # 2. 用 k 查表：表里认为 v 应该是什么
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, (v_t - pred) * beta[:, t, :, None])   # 3. 只写入差值（delta）
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)     # 4. 用 q 查表得到这个 token 的输出
    return out.to(dtype), S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.hidden_size
        self.n_k, self.n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.n_k * self.dk, self.n_v * self.dv            # 2048, 2048
        conv_dim = 2 * self.key_dim + self.value_dim                                       # q、k、v 拼在一起做卷积：6144
        self.in_proj_qkv = nn.Linear(H, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(H, self.value_dim, bias=False)                          # 输出门
        self.in_proj_b = nn.Linear(H, self.n_v, bias=False)                                # β：每个头一个数
        self.in_proj_a = nn.Linear(H, self.n_v, bias=False)                                # 衰减里随输入变的那部分
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, cfg.linear_conv_kernel_dim, groups=conv_dim,
                                padding=cfg.linear_conv_kernel_dim - 1, bias=False)        # 深度可分的因果短卷积：每个通道只看自己前 3 个 token
        self.dt_bias = nn.Parameter(torch.ones(self.n_v))
        self.A_log = nn.Parameter(torch.zeros(self.n_v))
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, H, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)                                          # [B, 6144, T]
        qkv = F.silu(self.conv1d(qkv)[..., :T]).transpose(1, 2)                            # 因果卷积（截掉右边 padding 产生的"未来"）+ SiLU
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q, k, v = q.view(B, T, self.n_k, self.dk), k.view(B, T, self.n_k, self.dk), v.view(B, T, self.n_v, self.dv)
        if self.n_v > self.n_k:                                                            # 大模型里 value 头比 key 头多：q、k 复制几份
            q, k = q.repeat_interleave(self.n_v // self.n_k, 2), k.repeat_interleave(self.n_v // self.n_k, 2)
        beta = self.in_proj_b(x).sigmoid()                                                 # [B, T, 头]，0~1
        g = -self.A_log.exp() * F.softplus(self.in_proj_a(x) + self.dt_bias)               # 对数衰减，永远 ≤ 0
        out, _ = delta_rule(q, k, v, g, beta)                                              # 这一章不保存记忆表，第 10 章才用
        out = self.norm(out, self.in_proj_z(x).view(B, T, self.n_v, self.dv))             # 按头做门控归一化
        return self.out_proj(out.reshape(B, T, -1))
```

`delta_rule` 里的四步和公式一一对应。外面的 `GatedDeltaNet` 是把残差流里的 1024 维变成这些量的"外壳"，有几个 Llama 系模型里没有的零件：

| 零件 | 形状 | 做什么 |
| --- | --- | --- |
| `in_proj_qkv` + `conv1d` | 1024 → 6144，再做窗口为 4 的深度可分因果卷积 | q、k、v 不是直接投影出来的，先让每个通道看自己前 3 个 token（局部上下文），再过 SiLU |
| `in_proj_b` → sigmoid | 1024 → 16 | 每个头一个 $\beta$：这个 token 写得多重 |
| `in_proj_a`, `dt_bias`, `A_log` | 1024 → 16 | $g = -e^{A\_log}\cdot\mathrm{softplus}(a + \text{dt\_bias})$，永远 $\le 0$，所以衰减 $e^g \in (0, 1]$；`A_log` 是每个头固定的"遗忘速度" |
| `in_proj_z` + `GatedRMSNorm` | 1024 → 2048 | 输出先按头归一化，再乘 $\mathrm{silu}(z)$ 的门 |
| `out_proj` | 2048 → 1024 | 回到残差流 |

`conv1d` 的 `padding = 3` 加上 `[..., :T]` 的截断，就是因果卷积：位置 $t$ 的输出只用 $t-3 \ldots t$ 的输入。`groups = 6144` 表示每个通道各卷各的，参数只有 6144 × 4 个。

## 换掉 18 层

```python title="ch07_deltanet.py"
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
```

```text title="输出"
换掉 18 层线性注意力后，logits 最大差 1.3e-05（前向 141 ms）
已替换 9/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✓  rotary_emb✓  self_attn✓  linear_attn✓  norm✓  lm_head✓
一层线性注意力 10.5M 参数：in_proj_qkv 6.3M，conv1d 只有 24576 个
每个头固定的遗忘速度 exp(A_log)：最小 0.00，最大 1.29（越大忘得越快）
第 0 层对这 6 个 token：衰减 exp(g) 的范围 [0.01, 1.00]，β 的范围 [0.13, 1.00]
  写入 1 次后读出的值和 v 的距离：delta 规则  4.673，普通线性注意力  4.673
  写入 2 次后读出的值和 v 的距离：delta 规则  2.336，普通线性注意力  0.000
  写入 3 次后读出的值和 v 的距离：delta 规则  1.168，普通线性注意力  4.673
  写入 4 次后读出的值和 v 的距离：delta 规则  0.584，普通线性注意力  9.345
  写入 5 次后读出的值和 v 的距离：delta 规则  0.292，普通线性注意力 14.018
  写入 6 次后读出的值和 v 的距离：delta 规则  0.146，普通线性注意力 18.691
```

最后那个小实验是 delta 规则的精髓：同一对 (key, value) 连写 6 次，读出来的值和 v 的距离每次减半（$\beta = 0.5$ 只走一半），收敛到 v；普通线性注意力没有那一减，每写一次就多叠一份 $\beta v$——第 2 次恰好叠到 $v$，之后越积越偏。

!!! interview "怎么讲清楚"
    讲线性注意力，先讲"状态是固定大小"这件事：注意力的 KV cache 随长度涨，记忆表不涨，所以长上下文下显存和每 token 的计算都是常数。再讲 delta 规则和普通线性注意力的区别：写入的是"新值减去表里已有的预测"，同一个 key 反复出现不会累积，表的容量用得更省。最后说 gated：衰减 $e^g$ 让它能忘，$\beta$ 让它能选择性地写——两个门都是从输入算出来的。能提一句"Qwen3.5 的 18 : 6 混合比例，是用 6 层全注意力补线性注意力的精确回看"就完整了。

## 小结

- [x] Gated DeltaNet 每个头维护一张 128 × 128 的记忆表：衰减 → 用 k 查表 → 写入差值 × β → 用 q 查表。
- [x] 外壳里有短卷积、$\beta$ 投影、衰减投影（`A_log`、`dt_bias`）和带门的归一化。
- [x] 换掉 18 层后 logits 最大差 1e-5 量级；骨架 9 个位置全部换完，但前向还是 HF 的解码层在调我们的模块——第 9 章把骨架也拆掉。

**这一章新建的文件**：`qwen/deltanet.py`（第 10 章会加上状态和 mask）。
