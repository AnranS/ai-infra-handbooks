# （六）门控注意力

<p class="lead">24 层里只有 6 层是全注意力，但它们是模型"精确回看"的唯一手段——线性注意力的记忆表会模糊，注意力矩阵不会。Qwen3.5 的注意力和教科书版比有四处不同：GQA（8 个 query 头共用 2 组 key/value）、QK-Norm（q 和 k 各过一个 RMSNorm 再旋转）、只转前 1/4 维的 RoPE、以及<strong>输出门</strong>——<code>q_proj</code> 一次算出两份，一份当 query，一份过 sigmoid 乘在输出上。这一章把它写出来，换掉 6 层。</p>

!!! note "这一章跑什么"
    `ch06_attention.py`，约 10 秒。

## 一次前向

```python title="qwen/attention.py"
"""门控注意力：Qwen3.5 的全注意力层，24 层里的 6 层。

和教科书的多头注意力比有四处不同：GQA（8 个 query 头共用 2 组 key/value）、QK-Norm（q 和 k 各过一个
RMSNorm 再旋转）、只旋转前 1/4 维的 RoPE、以及输出门——q_proj 一次算出两份，一份当 query，
一份过 sigmoid 当门，逐维乘在注意力的输出上。
"""
import math

import torch
import torch.nn as nn

from qwen.norm import RMSNorm
from qwen.rope import apply_rope


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, self.n_head, self.n_kv, self.d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(H, self.n_head * self.d * 2, bias=False)   # 两倍宽：一半 query，一半门
        self.k_proj = nn.Linear(H, self.n_kv * self.d, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.d, H, bias=False)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps)                     # 按头归一化：权重只有 head_dim 个
        self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, gate = self.q_proj(x).view(B, T, self.n_head, 2 * self.d).chunk(2, dim=-1)   # 每个头的 512 维：前 256 是 query，后 256 是门
        k = self.k_proj(x).view(B, T, self.n_kv, self.d)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                                  # [B, 头, T, d]
        rep = self.n_head // self.n_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)           # GQA：每组 KV 复制给 4 个 query 头
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d)                               # [B, 头, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)               # 上三角：看不到后面的 token
        scores = scores.masked_fill(causal, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v                                            # [B, 头, T, d]
        out = out.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1))   # 输出门
        return self.o_proj(out)
```

逐行对着形状读：

| 步骤 | 形状 | 说明 |
| --- | --- | --- |
| `q_proj(x)` | `[B, T, 4096]` → `[B, T, 8, 512]` | 每个头 512 维：前 256 是 query，后 256 是门 |
| `k_proj`, `v_proj` | `[B, T, 2, 256]` | 只有 2 组 key/value（GQA） |
| `q_norm`, `k_norm` | 不变 | 按头做 RMSNorm，权重只有 256 个；再做 RoPE |
| `repeat_interleave` | `[B, 8, T, 256]` | 每组 KV 复制给 4 个 query 头 |
| `scores` | `[B, 8, T, T]` | $qk^\top/\sqrt{256}$，上三角填 $-\infty$ |
| `softmax @ v` | `[B, 8, T, 256]` → `[B, T, 2048]` | 按行归一化再加权求和 |
| `* sigmoid(gate)` | `[B, T, 2048]` | 输出门：逐维决定这个头的输出放行多少 |
| `o_proj` | `[B, T, 1024]` | 回到残差流的宽度 |

**GQA 省的是 KV cache。** 8 个 query 头、2 组 KV：每个 token 要存的 key/value 从 8 × 256 × 2 降到 2 × 256 × 2 个数，推理时的显存和访存都是 4 倍的差距（第 10 章会算）。质量损失很小，因为多个 query 头看同一组 key 本来就有冗余。

**QK-Norm 防的是 logits 爆炸。** 训练中 $q$ 和 $k$ 的尺度会漂，点积变得很大之后 softmax 退化成 one-hot，梯度消失。在 RoPE 之前把 $q$、$k$ 各自归一化，点积的尺度就被钉住了。Qwen3 开始用，Qwen3.5 延续。

**输出门是这一代的新东西。** 来自 Qwen3-Next 的"Gated Attention"论文：在注意力输出后乘一个由输入算出的 sigmoid 门，代价是 `q_proj` 翻倍（25M 参数里有 12.6M 是门），收益是压掉"注意力汇聚"（attention sink，第一个 token 吃掉大量注意力）并稳定训练。

## 换掉 6 层

```python title="ch06_attention.py"
"""第 6 章：换掉 6 层全注意力；看一眼注意力矩阵和输出门"""
import os

import torch

from qwen.attention import Attention
from qwen.embedding import Embedding, LMHead
from qwen.harness import AttentionAdapter, RotaryAdapter, load_hf, logits, max_diff, status
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

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2–5 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
    if cfg.layer_types[i] == "full_attention":                                                 # 这一章：6 层
        layer.self_attn = AttentionAdapter(load(Attention(cfg), W, f"layers.{i}.self_attn."))
print(f"换掉 6 层全注意力后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

attn = hf.model.layers[3].self_attn.mine                           # 第 3 层是第一层全注意力
n = sum(p.numel() for p in attn.parameters())
print(f"一层注意力 {n / 1e6:.1f}M 参数，其中 q_proj {attn.q_proj.weight.numel() / 1e6:.1f}M（一半是门）")

torch.manual_seed(0)
with torch.no_grad():                                              # 把注意力矩阵和门拿出来看
    x = torch.randn(1, ids.shape[1], cfg.hidden_size)
    cos, sin = rope_tables(cfg, torch.arange(ids.shape[1]))
    T = ids.shape[1]
    q, gate = attn.q_proj(x).view(1, T, attn.n_head, 2 * attn.d).chunk(2, dim=-1)
    g = torch.sigmoid(gate)
    print(f"输出门 sigmoid(gate)：均值 {g.mean():.2f}，小于 0.1 的比例 {(g < 0.1).float().mean():.0%}，大于 0.9 的比例 {(g > 0.9).float().mean():.0%}")
    causal = torch.ones(T, T, dtype=torch.bool).triu(1)
    print("因果掩码（1 = 看不到）：")
    for row in causal.int()[:4]:
        print("   ", " ".join(str(v) for v in row.tolist()))
```

```text title="输出"
换掉 6 层全注意力后，logits 最大差 7.9e-06
已替换 8/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✓  rotary_emb✓  self_attn✓  linear_attn✗  norm✓  lm_head✓
一层注意力 7.3M 参数，其中 q_proj 4.2M（一半是门）
输出门 sigmoid(gate)：均值 0.50，小于 0.1 的比例 0%，大于 0.9 的比例 1%
因果掩码（1 = 看不到）：
    0 1 1 1 1 1
    0 0 1 1 1 1
    0 0 0 1 1 1
    0 0 0 0 1 1
```

误差从 0 变成了 1e-5 量级：矩阵乘的分块顺序和 HF 不同，浮点加法不满足结合律，这是正常的——后面所有"自己写的算子"都会有这个量级的差，真写错了会是 1e-1 以上。

!!! interview "怎么讲清楚"
    讲注意力的实现，按数据的形状走一遍最清楚：投影 → 分头 → 归一化 → 旋转 → 打分 → 掩码 → softmax → 加权求和 → 合头 → 门 → 投影。然后点出 Qwen3.5 的四个改动各解决什么：GQA 省 KV cache、QK-Norm 钉住点积尺度、部分旋转留一些无位置的维度、输出门压 attention sink。最后把"为什么 24 层里只有 6 层注意力"留给下一章：另外 18 层用的是不长大的记忆表。

## 小结

- [x] 门控注意力 = GQA + QK-Norm + 部分 RoPE + 输出门；`q_proj` 是两倍宽的。
- [x] 换掉 6 层后 logits 最大差 1e-5 量级（矩阵乘顺序不同导致的正常浮点差）；骨架 9 个位置换了 8 个。

**这一章新建的文件**：`qwen/attention.py`（第 10 章会加上状态和 mask 两个参数）。
