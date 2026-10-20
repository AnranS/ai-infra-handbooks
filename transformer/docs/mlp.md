# （四）SwiGLU 前馈网络

<p class="lead">每一层的后半段是前馈网络：每个 token 自己过一遍"先放宽到 3584 维、再收回 1024 维"的两层网络，token 之间互不来往。它只有三个矩阵，却占了全模型近一半的参数和算力——这一章把它写出来，看看闸门开了多少、"逐 token"是什么意思。</p>

!!! abstract "学完这章你能"
    1. 说清前馈网络和注意力的分工：注意力在 token 之间搬信息，前馈网络让每个 token 自己加工。
    2. 用"规则本"讲清前馈网络里存着什么：gate、up 的一行是条件，down 的一列是结论。
    3. 写出 SwiGLU，算出它的参数量，并用实验证明它是逐 token 的。

!!! note "这一章跑什么"
    `ch04_mlp.py`，约 10 秒。

@@files@@

## 它在做什么：每个 token 自己的规则本

一层分两半。前一半是混合器（第 6 章的注意力或第 7 章的线性注意力），在 token 之间搬信息：「是」去前文里把「中国」「首都」找过来。后一半是前馈网络，每个 token 关起门来，把搬来的东西自己加工一遍，token 之间不再来往。

前馈网络里有 3584 个神经元，可以看成 3584 条"如果……就……"的规则：

- **条件**：`gate_proj`、`up_proj` 的一行。当前 token 的向量和这一行对得上，这个神经元就亮。
- **结论**：`down_proj` 的一列。神经元亮了，就按亮的程度把这一列加回残差流。

研究里把这叫 key-value memory：模型学到的很多知识就存在这些规则里。第 9 章拼出整模型以后，能在真模型里找到这样的规则：

![图：前馈网络是每个 token 自己的规则本：gate、up 的一行是条件，down_proj 的一列是结论；第 23 层第 275 号神经元写回"中国城市"的方向，关掉它 P(北京) 从 18.6% 掉到 1.7%](assets/figures/mlp-neuron.svg){.aig-svg}

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

## 跑起来：闸门开了多少，逐 token 是什么意思

```python title="ch04_mlp.py"
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
d = diff(mlp(h[:, perm]), y[:, perm])
print(f"打乱 token 顺序后再过前馈，和先过前馈再打乱一样吗：{d < 1e-5}（差在浮点舍入以内，token 之间没有交互）")
print(f"只改第 2 个 token 的输入，其他位置的输出变了多少：{diff(mlp(h + (torch.arange(6) == 2)[None, :, None] * 1.0)[:, [0, 1, 3, 4, 5]], y[:, [0, 1, 3, 4, 5]]):.1e}")
```

```text title="输出"
一层前馈 11.01M 参数（3 × 1024 × 3584），24 层 264M，占全模型 35%
输入 (1, 6, 1024) -> 中间 (1, 6, 3584) -> 输出 (1, 6, 1024)
  第  0 层：silu(gate) > 0.1 的比例 32%，< -0.1 的比例 29%（silu 最小约 -0.28）
  第 11 层：silu(gate) > 0.1 的比例 32%，< -0.1 的比例 24%（silu 最小约 -0.28）
  第 23 层：silu(gate) > 0.1 的比例 43%，< -0.1 的比例 31%（silu 最小约 -0.28）
打乱 token 顺序后再过前馈，和先过前馈再打乱一样吗：True（差在浮点舍入以内，token 之间没有交互）
只改第 2 个 token 的输入，其他位置的输出变了多少：0.0e+00
```

最后两行是"逐 token"的证据：把 6 个 token 打乱顺序再过前馈，结果只是跟着打乱（差是 0 还是 1e-7 量级看机器——矩阵乘法按行分块时，同一行放在不同位置，浮点求和的顺序可能不同）；只改一个 token 的输入，其他五个位置的输出纹丝不动。前馈网络不看邻居，看邻居是注意力的事。闸门那三行：每层大约三成的中间维度被明显打开、三成被压到负数附近，剩下的接近 0——SwiGLU 的"稀疏"就是这个意思。

## 常见疑问

**SwiGLU 的 gate 和第 6 章注意力的输出门是一回事吗？**

不是，只是都叫 gate。SwiGLU 的 `gate_proj` 过的是 silu，可以是负数（最小约 −0.28），乘在 3584 维的中间值上，决定每条规则用多少；注意力的输出门过的是 sigmoid，只在 0～1 之间，乘在每个头取回来的输出上，决定这个头这次发不发言。

**为什么要先放宽到 3584 维，再收回 1024 维？**

中间的每一维是一条规则，放宽就是多放几条规则；规则之间靠非线性（silu 和相乘）分开，收回 1024 维时只把亮了的那些规则的结论加回去。中间太窄，能存的规则就少。

**MoE 为什么替换的是前馈网络，不是注意力？**

前馈网络是参数的大头，又是逐 token 的：每个 token 自己选几个专家（几本规则本）去查，互不影响，正好可以按 token 路由。第 16 章会讲。

!!! interview "怎么讲清楚"
    讲前馈网络抓三点：它是**逐 token** 的（没有序列维的信息交换，所以能和注意力分工）；SwiGLU 多了一个门，中间宽度按 2/3 缩，参数量和两矩阵版持平；它是参数和计算的大头，所以 MoE（第 16 章）替换的正是它，而不是注意力。能说出"推理时 decode 阶段这三个矩阵乘是访存瓶颈——每个 token 都要把 11M 参数从显存读一遍"，就接上了推理优化的话题。

## 小结

- [x] SwiGLU：`down(silu(gate(x)) * up(x))`，三个矩阵，中间宽 3584。
- [x] 24 层共 264M 参数，是嵌入表之外最大的一块；它是逐 token 的：打乱顺序、只改一个 token，其他位置纹丝不动。

## 思考题

**1. decode 时每生成一个 token，一层前馈网络要从显存读多少字节（bf16）？24 层呢？**

??? success "参考答案"
    一层 11.01M 个参数，bf16 每个 2 字节，约 22 MB；24 层约 528 MB。batch 只有 1 时，这些权重读一遍只换来一个 token 的计算，所以 decode 是访存瓶颈。

**2. 如果去掉 silu，直接算 `down(gate(x) * up(x))`，前馈网络还是非线性的吗？**

??? success "参考答案"
    还是。两个线性函数相乘是二次的，已经不是线性函数了，这个变体叫 bilinear。silu 让闸门在负数区几乎关上，规则更像"开关"，实验上效果也更好。

**3. 只改第 2 个 token 的输入，前馈网络在其他位置的输出纹丝不动。把一整层（注意力 + 前馈）一起看，还会这样吗？**

??? success "参考答案"
    不会。注意力会把第 2 个 token 的变化搬给后面的 token：因果掩码下，第 2 个及以后的位置都会变，只有第 0、1 个位置不变。

**这一章新建的文件**：`qwen/mlp.py`。
