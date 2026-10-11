# 4. The SwiGLU feed-forward network

<p class="lead">The second half of every layer is the feed-forward network: each token on its own passes through a two-layer network that "widens to 3584 dimensions, then narrows back to 1024", with no interaction between tokens. It has only three matrices, yet it holds nearly half of the model's parameters and compute — this chapter writes it and replaces all 24 layers.</p>

!!! note "What this chapter runs"
    `ch04_mlp.py`, about 10 seconds.

## Three matrices, one gate {#三个矩阵一个闸门}

```python title="qwen/mlp.py"
"""前馈网络（SwiGLU）：每个 token 自己过一遍"先放宽、再收窄"的两层网络，token 之间互不来往。

三个矩阵：gate 和 up 都把 1024 维放宽到 3584 维，silu(gate) 当闸门乘在 up 上，再由 down 收回 1024 维。
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

$$
\mathrm{MLP}(x) = W_{\text{down}}\,\big(\mathrm{silu}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x\big)
$$

The classic Transformer feed-forward is $W_2\,\mathrm{ReLU}(W_1 x)$, two matrices. SwiGLU adds a `gate`: `up` produces 3584 candidate values, and `silu(gate)` decides how much of each to let through (silu is almost 0 for negative inputs and roughly the identity for positive ones, so it acts as a soft gate). To keep the parameter count equal to the two-matrix version, the intermediate width drops from $4 \times 1024 = 4096$ to $\tfrac{2}{3} \times 4096 \approx 2731$ — Qwen3.5 uses 3584, a little wider than "equal".

Each of the three matrices is 1024 × 3584 = 3.67M, 11M per layer; 24 layers make 264M, the largest block of the 752M model after the embedding table (attention + linear attention together are about 233M).

## Replace 24 layers {#换掉-24-层}

```python title="ch04_mlp.py"
"""第 4 章：换掉 24 层前馈网络；量一下它占多少参数、silu 门开了多少"""
import os

import torch
import torch.nn.functional as F

from qwen.embedding import Embedding, LMHead
from qwen.harness import load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # replaced in chapters 2 and 3
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")                                          # this chapter
print(f"换掉 24 层前馈网络后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

n = sum(p.numel() for p in hf.model.layers[0].mlp.parameters())
print(f"一层前馈 {n / 1e6:.2f}M 参数，24 层 {24 * n / 1e6:.0f}M，占全模型 {24 * n / sum(p.numel() for p in hf.parameters()):.0%}")

torch.manual_seed(0)
with torch.no_grad():                                               # how open the gate is: the share of silu(gate) clearly above 0
    x = torch.randn(1, 8, cfg.hidden_size)
    for i in (0, 11, 23):
        g = F.silu(hf.model.layers[i].mlp.gate_proj(x))
        print(f"  第 {i:2d} 层：silu(gate) > 0.1 的比例 {(g > 0.1).float().mean():.0%}，< -0.1 的比例 {(g < -0.1).float().mean():.0%}（silu 最小约 -0.28）")
```

```text title="output"
换掉 24 层前馈网络后，logits 最大差 0.0e+00
已替换 6/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✓  rotary_emb✗  self_attn✗  linear_attn✗  norm✓  lm_head✓
一层前馈 11.01M 参数，24 层 264M，占全模型 35%
  第  0 层：silu(gate) > 0.1 的比例 33%，< -0.1 的比例 26%（silu 最小约 -0.28）
  第 11 层：silu(gate) > 0.1 的比例 27%，< -0.1 的比例 23%（silu 最小约 -0.28）
  第 23 层：silu(gate) > 0.1 的比例 36%，< -0.1 的比例 34%（silu 最小约 -0.28）
```

!!! interview "How to explain it"
    Three points about the feed-forward network: it is **per token** (no information exchange along the sequence, which is why it can divide the work with attention); SwiGLU adds a gate and shrinks the intermediate width by 2/3 to keep the parameter count of the two-matrix version; it is the bulk of the parameters and the compute, which is why MoE (chapter 16) replaces exactly this, not attention. Saying that "in the decode phase these three matrix multiplications are memory-bound — every token has to read 11M parameters from memory" connects it to inference optimization.

## Summary {#小结}

- [x] SwiGLU: `down(silu(gate(x)) * up(x))`, three matrices, intermediate width 3584.
- [x] 24 layers hold 264M parameters, the largest block after the embedding table; after the replacement the maximum logit difference is 0.

**File created in this chapter**: `qwen/mlp.py`.
