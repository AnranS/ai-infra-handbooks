# 前馈网络与 SwiGLU

<p class="lead">注意力负责在 token 之间搬运信息，前馈网络（FFN，也叫 MLP）负责对每个 token 独立地"加工"信息。它结构最简单，却占了一个 Transformer 层大约三分之二的参数和计算量，是推理时最大的矩阵乘法所在。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. FFN 的输入输出形状是什么？它和注意力最大的区别是什么？
    2. SwiGLU 是怎么计算的？为什么它有三个矩阵？
    3. 为什么 LLaMA-7B 的中间维度是 11008，而不是 4 × 4096？
    4. 在一个 Transformer 层里，FFN 占多少参数？
    5. 推理引擎通常怎么实现 SwiGLU？

## 经典的两层 MLP

原始 Transformer 的 FFN：先升维到 $d_{ff}$（通常是 4d），经过激活函数，再降回 d：

$$
\text{FFN}(x) = \text{Act}(x W_1)\, W_2
$$

它**对每个 token 独立计算**，token 之间没有任何交互。所以 FFN 可以看成一个作用在每个 token 上的"查表 + 加工"模块：研究发现，$W_1$ 的每一行像一个"模式探测器"，$W_2$ 的对应列是探测到这个模式时写回残差流的"内容"，许多事实性知识就存储在 FFN 的权重里。

## 激活函数

```pycon
>>> import torch
>>> import torch.nn.functional as F
>>> x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
>>> F.relu(x)
tensor([0., 0., 0., 1., 3.])
>>> F.gelu(x)
tensor([-0.0041, -0.1587,  0.0000,  0.8413,  2.9959])
>>> F.silu(x)                     # SiLU(x) = x * sigmoid(x)，也叫 Swish
tensor([-0.1423, -0.2689,  0.0000,  0.7311,  2.8577])
```

GELU 和 SiLU 是 ReLU 的"平滑版本"：在 0 附近连续可导，负数部分不完全为 0。现代大模型基本都用 SiLU 或 GELU。

## 门控：SwiGLU

**GLU（门控线性单元）** 家族用两路投影相乘：一路经过激活函数作为"门"，控制另一路的信息通过多少。LLaMA、Qwen、DeepSeek 等模型使用的 **SwiGLU**：

$$
\text{SwiGLU}(x) = \big(\text{SiLU}(x W_{gate}) \odot x W_{up}\big)\, W_{down}
$$

有三个矩阵：`gate_proj`、`up_proj`（都是 d → $d_{ff}$）和 `down_proj`（$d_{ff}$ → d）。实验表明门控结构在同样的参数量下效果更好。

为了让参数量与经典的 4d 两层 MLP 相当（$2 \times 4d^2 = 8d^2$），三矩阵结构把中间维度取为约 $\frac{8}{3}d$：$3 \times \frac{8}{3}d \times d = 8d^2$。LLaMA-7B 的 d = 4096，$\frac{8}{3} \times 4096 ≈ 10923$，再向上取整到 256 的倍数得到 11008。不过这只是一个起点，很多模型会选不同的比例，比如 Qwen2.5-0.5B 是 4864 / 896 ≈ 5.4 倍，Qwen3-0.6B 是 3072 / 1024 = 3 倍。

下面的实现和 transformers 里的 `Qwen2MLP` 完全一致：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

torch.manual_seed(0)
cfg = Qwen2Config(hidden_size=896, intermediate_size=4864, hidden_act="silu")
ref = Qwen2MLP(cfg)
ours = SwiGLU(896, 4864)
ours.load_state_dict(ref.state_dict())               # 参数名相同，直接加载
x = torch.randn(2, 5, 896)
assert torch.allclose(ours(x), ref(x), atol=1e-6)
print({k: tuple(v.shape) for k, v in ours.state_dict().items()})
```

## 参数与计算量的占比

算一算 Qwen3-0.6B 一层中各部分的参数量：

```pycon
>>> d, d_ff, n_h, n_kv, d_h = 1024, 3072, 16, 8, 128
>>> attn = d * n_h * d_h + 2 * d * n_kv * d_h + n_h * d_h * d      # q、k、v、o 四个投影（不计 QK-Norm 的 2 × 128 个参数）
>>> mlp = 3 * d * d_ff
>>> attn, mlp, round(mlp / (attn + mlp), 3)
(6291456, 9437184, 0.6)
```

在这个模型里，FFN 占了一层参数的 60%。注意力部分比"4d²"大，是因为 Qwen3 的头数 × 头维（16 × 128 = 2048）是隐藏维度的两倍，q、o 两个投影各有 2d² 个参数；K、V 用了 GQA（8 个头，见[注意力变体](attention-variants.md)），又省回来一些。在不用 GQA 的 LLaMA-7B 中，注意力是 $4d^2$、FFN 约 $8d^2$，FFN 约占三分之二。**由于每个参数对每个 token 贡献 2 次运算，参数占比基本就是计算量占比**（注意力分数本身的 $T^2$ 项另算）。

!!! inference "推理视角"
    - **FFN 是最大的 GEMM**：decode 时它决定了读多少权重，prefill 时它决定了大部分计算量；
    - **gate 和 up 融合成一个矩阵乘**：`gate_proj` 和 `up_proj` 的输入相同，推理引擎会把两个权重拼成一个 `[2·d_ff, d]` 的矩阵（vLLM 中叫 `gate_up_proj`），一次 GEMM 得到两者的结果，再用一个融合 kernel 计算 `silu(gate) * up`（vLLM 中的 `SiluAndMul`），省掉一次 kernel 启动和中间结果的读写；
    - **张量并行**：`gate_up_proj` 按输出维度切（列并行），每张卡算中间维度的一部分，激活函数可以本地计算；`down_proj` 按输入维度切（行并行），每张卡得到部分和，最后一次 all-reduce。整个 FFN 只需要一次通信；
    - **MoE**：把一个大 FFN 换成很多个小 FFN（专家），每个 token 只用其中几个，见[混合专家](moe.md)。

!!! interview "面试怎么答"
    FFN 题：对每个 token 独立计算，不用 GQA 时约占一层参数和计算的三分之二（用了 GQA 的模型里更高）；SwiGLU = `down(silu(gate(x)) * up(x))`，三个矩阵，中间维度约 8d/3 取整（LLaMA-7B 的 11008）。推理时 gate、up 合并成一个 GEMM，激活和乘法融合成一个 kernel；张量并行下 gate / up 按列切、down 按行切，整个 FFN 只需要一次 all-reduce。

## 练习

**1. 融合 gate 和 up。** 把 `SwiGLU` 改写成使用一个 `gate_up_proj`（`nn.Linear(d, 2 * d_ff)`）的版本，从上面的 `ours` 复制权重，验证输出一致。

??? success "参考答案"
    ```python
    class FusedSwiGLU(nn.Module):
        def __init__(self, d, d_ff):
            super().__init__()
            self.gate_up_proj = nn.Linear(d, 2 * d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d, bias=False)

        def forward(self, x):
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)

    fused = FusedSwiGLU(896, 4864)
    with torch.no_grad():
        fused.gate_up_proj.weight.copy_(torch.cat([ours.gate_proj.weight, ours.up_proj.weight], dim=0))
        fused.down_proj.weight.copy_(ours.down_proj.weight)
    assert torch.allclose(fused(x), ours(x), atol=1e-6)
    ```

    权重按行（输出维度）拼接，前一半输出是 gate，后一半是 up。vLLM 加载 Hugging Face 权重时，就是用这种方式把两个矩阵拼到一起的。

**2. 计算题。** LLaMA-3-8B 的 d = 4096、$d_{ff}$ = 14336、32 层。所有层的 FFN 一共有多少参数？占 80.3 亿总参数的多少？

??? success "参考答案"
    每层 3 × 4096 × 14336 ≈ 1.76 亿，32 层约 56.4 亿，占 70%。

    ```python
    ffn = 32 * 3 * 4096 * 14336
    assert ffn == 5_637_144_576
    print(f"{ffn / 8_030_261_248:.1%}")   # 70.2%
    ```

## 小结

- [x] FFN 对每个 token 独立计算，是一个 Transformer 层中参数和计算量最大的部分。
- [x] 现代大模型用 SwiGLU：`down(silu(gate(x)) * up(x))`，三个矩阵，中间维度通常不小于 8d/3。
- [x] 参数占比约等于计算量占比；GQA 让注意力部分更小，FFN 占比更高。
- [x] 推理时 gate、up 融合成一个 GEMM，激活与乘法融合成一个 kernel；张量并行下整个 FFN 只需一次 all-reduce。
