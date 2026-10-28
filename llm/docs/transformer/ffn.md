# 前馈网络与 SwiGLU

<p class="lead">注意力负责在 token 之间搬运信息，前馈网络（FFN，也叫 MLP）负责对每个 token 独立地"加工"信息。它结构最简单，却占了一个 Transformer 层大约三分之二的参数和计算量，是推理时最大的矩阵乘法所在。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. FFN 的输入输出形状是什么？它和注意力最大的区别是什么？
    2. SwiGLU 是怎么计算的？为什么它有三个矩阵？
    3. 为什么 LLaMA-7B 的中间维度是 11008，而不是 4 × 4096？
    4. 在一个 Transformer 层里，FFN 占多少参数？
    5. 推理引擎通常怎么实现 SwiGLU？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 输入输出都是 `[B, T, d]`。FFN 对每个 token 独立计算，不在 token 之间交换信息；注意力是唯一让 token 之间互相看的地方。
    2. $\mathrm{down}(\mathrm{silu}(\mathrm{gate}(x)) \odot \mathrm{up}(x))$：gate 分支过 SiLU 后决定每个中间神经元"开多大"，up 分支决定开了以后写出去的内容，两者逐元素相乘，再由 down 降回 d 维，所以有三个矩阵。
    3. 用了三个矩阵后，为了让参数量和原来两个矩阵、中间维度 4d 的 FFN 相当，中间维度取 $\frac{2}{3} \times 4d = \frac{8}{3}d \approx 10923$，再向上取整到 256 的倍数得到 11008。
    4. 不用 GQA 时注意力约 $4d^2$、FFN 约 $8d^2$，FFN 约占三分之二；用了 GQA 的模型里更高（本章的 Qwen3-0.6B 约 60%，LLaMA-3-8B 约 70%）。
    5. 把 gate 和 up 合并成一个 `gate_up_proj` 的 GEMM，再用一个融合 kernel 同时做 SiLU 和逐元素乘（`silu_and_mul`），最后是 down 的 GEMM。

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

有三个矩阵：`gate_proj`、`up_proj`（都是 d → $d_{ff}$）和 `down_proj`（$d_{ff}$ → d）。在大模型上，同样的参数量和计算量下，门控 FFN 的困惑度比 ReLU / GELU 的普通 FFN 低（Shazeer，2020）。两路相乘为什么更好？下面用几个实验拆开看。

### 从一个神经元看"门"

FFN 中间层有 $d_{ff}$ 个神经元。普通 FFN 里，一个神经元先给当前 token 打一个分 $a = w \cdot x$（可以理解成"检测到某种模式的强度"），再过 ReLU：分数为负就输出 0，为正就原样输出。**同一个分数既决定"响不响应"，又决定"输出多少"**，而且输出只能 ≥ 0。

SwiGLU 的一个神经元有两路：gate 那一路的分数 $a$ 过 SiLU，决定"这次开多大"；up 那一路的分数 $b$ 决定"开了以后写出去什么"，可正可负；输出是两者的乘积 $\text{SiLU}(a)\cdot b$。代几个打分进去看看：

```python
import torch
import torch.nn.functional as F


def show(label, t):                                           # 保留两位小数（+ 0.0 把 -0.0 变成 0.0）
    print(f"{label:<13}", [round(v, 2) + 0.0 for v in t.tolist()])

a = torch.tensor([-4.0, -1.0, 0.0, 1.0, 3.0])                 # 神经元的打分
show("a", a)
show("ReLU(a)", F.relu(a))                                    # 普通神经元：只有这一个打分
show("SiLU(a)", F.silu(a))                                    # 门控神经元的"开度"
for b in (2.0, -2.0):                                         # 门控神经元多一路"内容" b
    show(f"SiLU(a) × {b:+.0f}", F.silu(a) * b)
```

```text title="输出"
a             [-4.0, -1.0, 0.0, 1.0, 3.0]
ReLU(a)       [0.0, 0.0, 0.0, 1.0, 3.0]
SiLU(a)       [-0.07, -0.27, 0.0, 0.73, 2.86]
SiLU(a) × +2  [-0.14, -0.54, 0.0, 1.46, 5.72]
SiLU(a) × -2  [0.14, 0.54, 0.0, -1.46, -5.72]
```

- 普通神经元：a 是负的就输出 0，是正的就输出 a 本身，大小和正负都只由 a 决定；
- 门控神经元：a = −4 时开度只有 −0.07，不管 b 是 +2 还是 −2，输出都只有 ±0.14，门是关着的；a = 3 时开度 2.86，输出 ±5.72，正负完全由 b 决定；
- 所以"门控"就是把**"什么时候输出"（gate）和"输出什么"（up）交给两组不同的权重**。一个神经元就能表达"当 A 出现时，写出 B"，A 和 B 可以是两种完全不同的特征。可以把 gate 想成水龙头的开度，up 是管子里的水（冷水或热水），乘起来才是流出来的东西；
- 严格说 SiLU 不是 0～1 的开关：负的时候会稍微反向漏一点（最低约 −0.28），正的时候还会放大。

把 $\text{SiLU}(a)\cdot b$ 画成关于 $a$（门的输入）和 $b$（内容）的曲面，和不带门的 GELU、硬切的 ReGLU、不加非线性的双线性对比着看。曲面直接读不太容易，先看下面两张平面图：它们是在当前的 a、b 处各切的一刀。第一张里那条直线的斜率就是门的开度，拖动 a，看它从平（关）变到陡（开）：

<div class="aig-widget" data-widget="swiglu3d"></div>

### 实验：门控天生会做乘法

乘法相当于"并且"：只有门开了、内容也大，输出才大。这让门控神经元特别擅长表达两个特征的组合。用最直接的组合试一下：让网络学 $y = x_0 \cdot x_1$，参数量差不多时，比较门控和 ReLU：

```python
import torch.nn as nn


class Plain(nn.Module):                                       # 普通 FFN：一路投影，过 ReLU
    def __init__(self, h):
        super().__init__()
        self.up, self.down = nn.Linear(2, h), nn.Linear(h, 1)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


class Gated(nn.Module):                                       # 门控 FFN：gate、up 两路相乘
    def __init__(self, h):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(2, h), nn.Linear(2, h), nn.Linear(h, 1)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def fit(model, steps=2000):
    """学 y = x0 · x1，返回测试集上的均方误差"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.mse_loss(model(x_test), y_test).item()


torch.manual_seed(0)
x, x_test = torch.rand(4096, 2) * 4 - 2, torch.rand(1024, 2) * 4 - 2   # 输入在 [-2, 2] × [-2, 2] 里随机取
y, y_test = x[:, :1] * x[:, 1:], x_test[:, :1] * x_test[:, 1:]
print(f"y 的方差 {y_test.var():.2f}：什么都没学会、一直猜平均值，误差就是这么大")
for name, model in [("门控，2 个神经元", Gated(2)), ("ReLU，4 个神经元", Plain(4)),
                    ("ReLU，16 个神经元", Plain(16)), ("ReLU，64 个神经元", Plain(64))]:
    n = sum(p.numel() for p in model.parameters())
    print(f"{name}（{n} 个参数）：测试误差 {fit(model):.4f}")
```

```text title="输出"
y 的方差 1.77：什么都没学会、一直猜平均值，误差就是这么大
门控，2 个神经元（15 个参数）：测试误差 0.0032
ReLU，4 个神经元（17 个参数）：测试误差 0.0723
ReLU，16 个神经元（65 个参数）：测试误差 0.0018
ReLU，64 个神经元（257 个参数）：测试误差 0.0001
```

- 门控只用 2 个神经元、15 个参数，误差 0.0032，和 16 个神经元、65 个参数的 ReLU 网络差不多；
- 参数量相当的 ReLU 网络（4 个神经元、17 个参数）误差 0.072，大了 20 多倍。ReLU 网络只能用一段段平面去拼这个曲面，要拼得细就得多用神经元。

为什么 2 个门控神经元就够？不训练，手动设一组权重就明白了：

```python
exact = Gated(2)                                              # 不训练，手动设权重
with torch.no_grad():
    for p in exact.parameters():
        p.zero_()
    exact.gate.weight[:] = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])    # 两个门：x0 和 -x0
    exact.up.weight[:] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])       # 两路内容：都是 x1
    exact.down.weight[:] = torch.tensor([[1.0, -1.0]])                # 一加一减
    print(f"手动设的两个门控神经元：测试误差 {F.mse_loss(exact(x_test), y_test).item():.6f}")
```

```text title="输出"
手动设的两个门控神经元：测试误差 0.000000
```

两个神经元分别算出 $\text{SiLU}(x_0)\cdot x_1$ 和 $\text{SiLU}(-x_0)\cdot x_1$，相减得到 $\big(\text{SiLU}(x_0) - \text{SiLU}(-x_0)\big)\cdot x_1 = x_0 \cdot x_1$，因为 $\text{SiLU}(z) - \text{SiLU}(-z) = z\,\sigma(z) + z\,\sigma(-z) = z$。一点误差都没有。真实的语言当然不只是乘法，但"当 A 出现时写出 B"这样的组合到处都是，门控用更少的参数就能表达。

### 三个矩阵与中间维度

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

### 真实模型里的门

看 Qwen3-0.6B 里的门实际开成什么样。用钩子取出每层 MLP 的输入，自己算出每个神经元的开度 $\text{SiLU}(\text{gate})$ 和内容 up：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
ids = tok(open("docs/assets/sample-passage.txt", encoding="utf-8").read(), return_tensors="pt").input_ids[:, :512]

inputs = {}                                                   # 每层 MLP 的输入：[T, d]
hooks = [layer.mlp.register_forward_hook(lambda m, i, o, l=l: inputs.__setitem__(l, i[0][0]))
         for l, layer in enumerate(model.model.layers)]
with torch.no_grad():
    model(ids)
for h in hooks:
    h.remove()

for l in (2, 13, 20):
    mlp = model.model.layers[l].mlp
    with torch.no_grad():
        g, u = F.silu(mlp.gate_proj(inputs[l])), mlp.up_proj(inputs[l])   # 每个 token 在 3072 个神经元上的开度、内容
    top = [set(t.tolist()) for t in g.abs().topk(g.shape[-1] // 20, dim=-1).indices[::32]]   # 每隔 32 个 token 取一个，看它开得最大的 5%
    same = sum(len(p & q) / len(p) for i, p in enumerate(top) for q in top[i + 1:]) / (len(top) * (len(top) - 1) / 2)
    print(f"第 {l:2d} 层：几乎关上 {(g.abs() < 0.05).float().mean():.0%}，开度大于 1 的 {(g > 1).float().mean():.1%}，"
          f"内容为负的 {(u < 0).float().mean():.0%}；两个 token 开得最大的 5% 平均有 {same:.0%} 是同一批")
```

```text title="输出"
第  2 层：几乎关上 6%，开度大于 1 的 0.4%，内容为负的 50%；两个 token 开得最大的 5% 平均有 12% 是同一批
第 13 层：几乎关上 8%，开度大于 1 的 9.0%，内容为负的 50%；两个 token 开得最大的 5% 平均有 31% 是同一批
第 20 层：几乎关上 8%，开度大于 1 的 12.1%，内容为负的 50%；两个 token 开得最大的 5% 平均有 43% 是同一批
```

- **门更像调光器，不是开关**：几乎关上（开度 < 0.05）的只占 6%～8%；开度大于 1 的，在第 13 层约 9%、第 20 层约 12%，其余都是半开；
- **内容正好一半是负数**：方向由 up 决定，门只管大小；
- **不同 token 打开的是不同的一批神经元**：第 2 层两个 token 开得最大的 5% 平均只有 12% 是同一批；越深的层共用的越多，第 20 层是 43%。

门到底有多重要？把它的作用去掉试试。下面把每层 MLP 的 `forward` 换成一个能改门的版本：

```python
import types

rule = {}                                                     # 这一次怎么改门


def mlp_forward(self, h):                                     # 和 Qwen3MLP.forward 一样，只是能按 rule 改门
    g = F.silu(self.gate_proj(h))
    if self.layer_idx in rule.get("mean", ()):                # 门不再看 token 是谁：换成这段文字上的平均开度
        g = g.mean(dim=1, keepdim=True).expand_as(g)
    if "keep" in rule:                                        # 每个 token 只留开得最大的一部分，其余关掉
        idx = g.abs().topk(int(g.shape[-1] * rule["keep"]), dim=-1).indices
        g = g * torch.zeros_like(g).scatter_(-1, idx, 1.0)
    return self.down_proj(g * self.up_proj(h))


for l, layer in enumerate(model.model.layers):
    layer.mlp.layer_idx = l
    layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)


def ppl(**r):
    rule.clear()
    rule.update(r)
    with torch.no_grad():
        logits = model(ids).logits[0, :-1]
    return F.cross_entropy(logits, ids[0, 1:]).exp().item()


for name, r in [("原样", {}), ("第 13 层的门换成平均开度", {"mean": {13}}), ("所有层的门都换成平均开度", {"mean": set(range(28))}),
                ("每个 token 只留开得最大的 50%", {"keep": 0.5}), ("只留 30%", {"keep": 0.3})]:
    print(f"{name}：困惑度 {ppl(**r):.1f}")
```

```text title="输出"
原样：困惑度 28.1
第 13 层的门换成平均开度：困惑度 30.8
所有层的门都换成平均开度：困惑度 221441.7
每个 token 只留开得最大的 50%：困惑度 160.5
只留 30%：困惑度 2363.6
```

- 只把第 13 层的门换成平均开度，这一层的门不再看 token 是谁，困惑度从 28.1 涨到 30.8，一层影响不大；
- 所有层都这么换，困惑度变成 22 万，模型完全坏了。门的价值就在于**随 token 变**：每个 token 打开不同的一批神经元，相当于按神经元粒度做软选择。下一章的[混合专家](moe.md)把这个想法做成了整块的专家；
- 每个 token 只留开得最大的一半神经元，困惑度就涨到 160；只留 30%，是 2364。开度小的那些加起来也很重要，再次说明 SiLU 的门是调光器。所以 SwiGLU 模型不能像 ReLU 模型那样直接跳过"没激活"的神经元来加速；有研究把激活换回 ReLU 再继续训练，才换来真正的稀疏。

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

换几个模型的配置，看一层和整个模型的参数怎么分，以及上下文多长时注意力分数的 $T^2$ 项会反过来成为大头：

<div class="aig-widget" data-widget="param-share"></div>

!!! inference "推理视角"
    - **FFN 是最大的 GEMM**：decode 时它决定了读多少权重，prefill 时它决定了大部分计算量；
    - **gate 和 up 融合成一个矩阵乘**：`gate_proj` 和 `up_proj` 的输入相同，推理引擎会把两个权重拼成一个 `[2·d_ff, d]` 的矩阵（vLLM 中叫 `gate_up_proj`），一次 GEMM 得到两者的结果，再用一个融合 kernel 计算 `silu(gate) * up`（vLLM 中的 `SiluAndMul`），省掉一次 kernel 启动和中间结果的读写；
    - **张量并行**：`gate_up_proj` 按输出维度切（列并行），每张卡算中间维度的一部分，激活函数可以本地计算；`down_proj` 按输入维度切（行并行），每张卡得到部分和，最后一次 all-reduce。整个 FFN 只需要一次通信；
    - **MoE**：把一个大 FFN 换成很多个小 FFN（专家），每个 token 只用其中几个，见[混合专家](moe.md)。

!!! interview "怎么讲清楚"
    讲 FFN：对每个 token 独立计算，不用 GQA 时约占一层参数和计算的三分之二（用了 GQA 的模型里更高）；SwiGLU = `down(silu(gate(x)) * up(x))`，三个矩阵，中间维度约 8d/3 取整（LLaMA-7B 的 11008）。推理时 gate、up 合并成一个 GEMM，激活和乘法融合成一个 kernel；张量并行下 gate / up 按列切、down 按行切，整个 FFN 只需要一次 all-reduce。

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
- [x] 门控把"什么时候输出"（gate）和"输出什么"（up）交给两组权重，乘法让一个神经元能表达特征的组合；SiLU 的门是随 token 变化的调光器，不是开关。
- [x] 参数占比约等于计算量占比；GQA 让注意力部分更小，FFN 占比更高。
- [x] 推理时 gate、up 融合成一个 GEMM，激活与乘法融合成一个 kernel；张量并行下整个 FFN 只需一次 all-reduce。
