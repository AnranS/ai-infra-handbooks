# （四）SwiGLU 前馈网络

<p class="lead">每一层的后半段是前馈网络：每个 token 自己过一遍"先放宽到 3584 维、再收回 1024 维"的两层网络，token 之间互不来往。它只有三个矩阵，却占了全模型近一半的参数和算力——这一章把它写出来，换掉 24 层。</p>

!!! note "这一章跑什么"
    `ch04_mlp.py`，约 10 秒。

@@files@@

## 三个矩阵，一个闸门

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

经典 Transformer 的前馈是 $W_2\,\mathrm{ReLU}(W_1 x)$，两个矩阵。SwiGLU 多了一个 `gate`：`up` 算出 3584 个候选值，`silu(gate)` 决定每个值放行多少（silu 在负数区几乎为 0，正数区约等于本身，所以是个软闸门）。为了参数量和两矩阵版本持平，中间宽度从 $4 \times 1024 = 4096$ 降到 $\tfrac{2}{3} \times 4096 \approx 2731$——Qwen3.5 取了 3584，比"持平"略宽。

三个矩阵各 1024 × 3584 = 3.67M，一层 11M；24 层共 264M，是全模型 752M 里除了嵌入表之外最大的一块（注意力 + 线性注意力合计约 233M）。

## 换掉 24 层

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

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2、3 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")                                          # 这一章
print(f"换掉 24 层前馈网络后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

n = sum(p.numel() for p in hf.model.layers[0].mlp.parameters())
print(f"一层前馈 {n / 1e6:.2f}M 参数，24 层 {24 * n / 1e6:.0f}M，占全模型 {24 * n / sum(p.numel() for p in hf.parameters()):.0%}")

torch.manual_seed(0)
with torch.no_grad():                                               # 闸门开了多少：silu(gate) 明显大于 0 的比例
    x = torch.randn(1, 8, cfg.hidden_size)
    for i in (0, 11, 23):
        g = F.silu(hf.model.layers[i].mlp.gate_proj(x))
        print(f"  第 {i:2d} 层：silu(gate) > 0.1 的比例 {(g > 0.1).float().mean():.0%}，< -0.1 的比例 {(g < -0.1).float().mean():.0%}（silu 最小约 -0.28）")
```

```text title="输出"
换掉 24 层前馈网络后，logits 最大差 0.0e+00
已替换 6/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✓  rotary_emb✗  self_attn✗  linear_attn✗  norm✓  lm_head✓
一层前馈 11.01M 参数，24 层 264M，占全模型 35%
  第  0 层：silu(gate) > 0.1 的比例 33%，< -0.1 的比例 26%（silu 最小约 -0.28）
  第 11 层：silu(gate) > 0.1 的比例 27%，< -0.1 的比例 23%（silu 最小约 -0.28）
  第 23 层：silu(gate) > 0.1 的比例 36%，< -0.1 的比例 34%（silu 最小约 -0.28）
```

!!! interview "怎么讲清楚"
    讲前馈网络抓三点：它是**逐 token** 的（没有序列维的信息交换，所以能和注意力分工）；SwiGLU 多了一个门，中间宽度按 2/3 缩，参数量和两矩阵版持平；它是参数和计算的大头，所以 MoE（第 16 章）替换的正是它，而不是注意力。能说出"推理时 decode 阶段这三个矩阵乘是访存瓶颈——每个 token 都要把 11M 参数从显存读一遍"，就接上了推理优化的话题。

## 小结

- [x] SwiGLU：`down(silu(gate(x)) * up(x))`，三个矩阵，中间宽 3584。
- [x] 24 层共 264M 参数，是嵌入表之外最大的一块；换掉后 logits 最大差为 0。

**这一章新建的文件**：`qwen/mlp.py`。
