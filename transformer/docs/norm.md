# （三）RMSNorm：零中心的归一化

<p class="lead">每一层的输入先经过 RMSNorm，把向量缩放到"均方根为 1"，再逐维乘一个可学的比例。Qwen3.5 的版本有个小改动——权重存的是 <code>w − 1</code>，用的时候乘 <code>1 + weight</code>——这一章讲清楚为什么，同时把第 7 章要用的"带门的 RMSNorm"和 L2 归一化也写了。全模型 54 处归一化一起换掉。</p>

!!! note "这一章跑什么"
    `ch03_norm.py`，约 10 秒。

@@files@@

## 三种归一化

```python title="qwen/norm.py"
"""RMSNorm：把每个向量缩放到"均方根为 1"，再逐维乘一个可学的比例。

Qwen3.5 存的是零中心的权重：文件里的 weight 是 w - 1，用的时候乘 (1 + weight)。
这样初始化为 0 就等于"不缩放"，bf16 下也比存 1.0x 的数精度高（1 附近的 bf16 间隔是 2^-7，0 附近细得多）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))                # 零中心：0 表示比例为 1
        self.eps = eps

    def forward(self, x):
        xf = x.float()                                              # 统计量一律用 fp32 算
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """线性注意力的输出用这个：先归一化，再乘 silu(门)。它的权重不是零中心的（初始为 1）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * y * F.silu(gate.float())).type_as(x)


def l2norm(x, eps=1e-6):
    """把最后一维缩放到单位长度（线性注意力里 q、k 进记忆表之前用）"""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)
```

三个东西的区别：

| | 除以什么 | 之后乘什么 | 用在哪 |
| --- | --- | --- | --- |
| `RMSNorm` | 均方根 $\sqrt{\mathrm{mean}(x^2)}$ | $1 + w$（逐维） | 每层前面两处、最后一层后面、注意力的 q 和 k（按头） |
| `GatedRMSNorm` | 均方根 | $w \cdot \mathrm{silu}(z)$（逐维，门 $z$ 来自输入） | 线性注意力的输出（第 7 章） |
| `l2norm` | 向量长度 $\sqrt{\sum x^2}$ | 不乘 | 线性注意力的 q、k（第 7 章） |

RMSNorm 和 LayerNorm 比少了"减均值"和偏置：只管尺度，不管中心，计算省一半，效果一样。所有统计量都在 fp32 里算：bf16 下平方再求和会丢精度，这是所有实现的共同约定。

**为什么存 w − 1。** 训练开始时比例应该是 1。如果直接存 1.0，bf16 在 1 附近的相邻可表示数间隔是 $2^{-7} \approx 0.008$，一次更新小于它就被吞掉；存 0 的话，0 附近的间隔小得多（bf16 是浮点数，越靠近 0 越密）。于是 Qwen3.5 把"偏离 1 多少"当参数存——所以 `init` 用 `zeros`，forward 里乘 `1 + weight`。Gemma 也这么做。

## 和 HF 的同名模块对答案

RMSNorm 在模型里出现在四种位置：每层的 `input_layernorm` 和 `post_attention_layernorm`（24 × 2）、最后的 `norm`（1）、全注意力的 `q_norm` / `k_norm`（6 × 2，第 6 章随注意力一起用）；带门的版本在 18 层线性注意力里。三种归一化各和 HF 的同名模块对一次：

```python title="ch03_norm.py"
"""第 3 章：三种归一化，和 HF 的同名模块对答案；看看零中心权重长什么样"""
import os

import torch
from transformers import AutoModelForCausalLM

from qwen.norm import GatedRMSNorm, RMSNorm, l2norm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()                   # 两个张量逐元素的最大差
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # 参考实现：只拿它的同名模块对答案
torch.manual_seed(0)
x = torch.randn(2, 3, cfg.hidden_size) * 5

norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")
print(f"RMSNorm 和 HF 的 layers[0].input_layernorm 最大差 {diff(norm(x), hf.model.layers[0].input_layernorm(x)):.1e}")
final = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
print(f"最后的 RMSNorm 和 HF 的 norm 最大差 {diff(final(x), hf.model.norm(x)):.1e}")
gated = load(GatedRMSNorm(cfg.linear_value_head_dim, cfg.rms_norm_eps), W, "layers.0.linear_attn.norm.")
h, z = torch.randn(96, cfg.linear_value_head_dim), torch.randn(96, cfg.linear_value_head_dim)
print(f"带门的 RMSNorm 和 HF 的 linear_attn.norm 最大差 {diff(gated(h, z), hf.model.layers[0].linear_attn.norm(h, z)):.1e}")
print("l2norm 之后向量的长度：", [f"{v:.3f}" for v in l2norm(x).norm(dim=-1).flatten()[:3]])

w = W["layers.0.input_layernorm.weight"]
print(f"第 0 层输入归一化的权重：均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 存的是 w-1，所以在 0 附近")
w = W["norm.weight"]
print(f"最后一层的归一化权重：  均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 它把尺度调到嵌入表的尺度")
y = RMSNorm(cfg.hidden_size)(x)                                   # 权重为 0 时，归一化之后每个向量的均方根都是 1
print("归一化前每个向量的均方根：", [f"{v:.2f}" for v in x.pow(2).mean(-1).sqrt().flatten()[:3]],
      "归一化后：", [f"{v:.2f}" for v in y.pow(2).mean(-1).sqrt().flatten()[:3]])
print(f"全模型 {cfg.num_hidden_layers * 2 + 1} 处 RMSNorm、{cfg.layer_types.count('full_attention') * 2} 处按头的 QK-Norm、"
      f"{cfg.layer_types.count('linear_attention')} 处带门的 RMSNorm，都是这两个类")
```

```text title="输出"
RMSNorm 和 HF 的 layers[0].input_layernorm 最大差 0.0e+00
最后的 RMSNorm 和 HF 的 norm 最大差 0.0e+00
带门的 RMSNorm 和 HF 的 linear_attn.norm 最大差 0.0e+00
l2norm 之后向量的长度： ['1.000', '1.000', '1.000']
第 0 层输入归一化的权重：均值 +0.238，范围 [-0.11, +1.00] —— 存的是 w-1，所以在 0 附近
最后一层的归一化权重：  均值 +3.309，范围 [-0.78, +5.31] —— 它把尺度调到嵌入表的尺度
归一化前每个向量的均方根： ['5.14', '5.05', '4.84'] 归一化后： ['1.00', '1.00', '1.00']
全模型 49 处 RMSNorm、12 处按头的 QK-Norm、18 处带门的 RMSNorm，都是这两个类
```

!!! interview "怎么讲清楚"
    讲 RMSNorm 先说它和 LayerNorm 的差别（不减均值、无偏置、只管尺度），再说两个工程细节：统计量用 fp32 算；Qwen3.5 / Gemma 存的是 $w-1$，原因是 bf16 在 0 附近比在 1 附近精细。能顺便说出 Pre-Norm（归一化放在每个子层**前面**）为什么比 Post-Norm 好训——残差流一路畅通，梯度不用穿过归一化——就完整了。

## 小结

- [x] RMSNorm = 除以均方根再逐维缩放；Qwen3.5 的权重是零中心的（乘 $1+w$），初始化为 0。
- [x] `GatedRMSNorm`（归一化后乘 silu 门）和 `l2norm` 是第 7 章线性注意力的零件，先放在这里。
- [x] 三种归一化和 HF 的同名模块最大差都是 0。

**这一章新建的文件**：`qwen/norm.py`。
