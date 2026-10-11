# （四）SwiGLU 前馈网络

<p class="lead">每一层的后半段是前馈网络：每个 token 自己过一遍"先放宽到 3584 维、再收回 1024 维"的两层网络，token 之间互不来往。它只有三个矩阵，却占了全模型近一半的参数和算力——这一章把它写出来，和 HF 的实现对答案。</p>

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

## 和 HF 的 mlp 对答案

```python title="ch04_mlp.py"
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
```

```text title="输出"
前馈网络和 HF 的 layers[0].mlp 最大差 0.0e+00
一层前馈 11.01M 参数，24 层 264M，占全模型 35%
  第  0 层：silu(gate) > 0.1 的比例 33%，< -0.1 的比例 26%（silu 最小约 -0.28）
  第 11 层：silu(gate) > 0.1 的比例 27%，< -0.1 的比例 23%（silu 最小约 -0.28）
  第 23 层：silu(gate) > 0.1 的比例 36%，< -0.1 的比例 34%（silu 最小约 -0.28）
```

!!! interview "怎么讲清楚"
    讲前馈网络抓三点：它是**逐 token** 的（没有序列维的信息交换，所以能和注意力分工）；SwiGLU 多了一个门，中间宽度按 2/3 缩，参数量和两矩阵版持平；它是参数和计算的大头，所以 MoE（第 16 章）替换的正是它，而不是注意力。能说出"推理时 decode 阶段这三个矩阵乘是访存瓶颈——每个 token 都要把 11M 参数从显存读一遍"，就接上了推理优化的话题。

## 小结

- [x] SwiGLU：`down(silu(gate(x)) * up(x))`，三个矩阵，中间宽 3584。
- [x] 24 层共 264M 参数，是嵌入表之外最大的一块；和 HF 的 `mlp` 最大差为 0。

**这一章新建的文件**：`qwen/mlp.py`。
