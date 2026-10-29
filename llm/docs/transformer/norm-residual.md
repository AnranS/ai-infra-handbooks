# 归一化与残差流

<p class="lead">残差连接让几十上百层的网络能够训练，归一化让每一层的输入保持在合适的尺度。它们的计算量微不足道，却决定了模型的数值特性：残差流中的"巨大激活"和"离群通道"，正是量化和低精度推理最头疼的问题。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 残差连接是什么？为什么说每一层是在"读写残差流"？
    2. LayerNorm 和 RMSNorm 的公式分别是什么？RMSNorm 省掉了什么？
    3. Pre-Norm 和 Post-Norm 的区别？现代大模型用哪种？
    4. 为什么归一化的统计量要用 FP32 计算？
    5. 什么是激活离群值？它对推理有什么影响？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 每个子层的输出加回到输入上：$x \leftarrow x + f(x)$。这条贯穿所有层的 $x$ 叫残差流，每一层从中读取、计算，再把结果写回（加回）去。
    2. LayerNorm：$(x - \mu)/\sqrt{\sigma^2 + \epsilon} \cdot \gamma + \beta$；RMSNorm：$x / \sqrt{\mathrm{mean}(x^2) + \epsilon} \cdot \gamma$，省掉了减均值（少一次归约）和偏置 $\beta$。
    3. Post-Norm 在残差相加之后归一化（原始 Transformer），Pre-Norm 在子层之前归一化、残差流本身不归一化，深层模型训练更稳定。现代大模型都用 Pre-Norm，并在最后加一个最终归一化。
    4. 要对整行求平方和，低精度下长累加会丢精度、FP16 下平方还可能溢出，而这个统计量会作用到整行的每个数上，所以用 FP32 计算。
    5. 少数固定的通道（或少数 token）的激活比其他大几十到上千倍。它们让按张量或按 token 的激活量化误差很大，是 W8A8 等激活量化的主要难点。

## 残差连接

每个子层（注意力、前馈网络）的输出不是直接替换输入，而是**加回**到输入上：

$$
x \leftarrow x + \text{Sublayer}(\text{Norm}(x))
$$

好处有两个：梯度可以沿着"加法"这条捷径直接传回浅层，几十上百层也能训练；每一层只需要学习"在现有表示上做什么修改"，而不是从头构造新的表示。

从这个角度看，整个模型就是一条贯穿始终的**残差流**：嵌入层把 token 写进去，每一层从中读取（经过归一化）、计算、把结果加回去，最后由输出层读出。一个 Transformer 层的完整结构：

![图：残差流贯穿所有层，每层从中读、算、再加回去](../assets/figures/residual-stream.svg){.aig-svg}

```py
h = x + attention(rms_norm_1(x))      # 注意力子层：token 之间交换信息
y = h + mlp(rms_norm_2(h))            # 前馈子层：每个 token 独立加工
```

### 实验：没有残差，梯度传不回去

先在一个 32 层的小网络上看残差连接救了什么：同样的网络，一个每层直接替换输入，一个把每层的结果加回输入，比较初始化时各层拿到的梯度，再各训练 300 步：

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Deep(nn.Module):
    """32 层，每层一个 d × d 的线性层加 tanh；residual=True 时把每层的结果加回输入"""
    def __init__(self, residual, d=64, depth=32):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(depth))
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        for layer in self.layers:
            x = x + torch.tanh(layer(x)) if self.residual else torch.tanh(layer(x))
        return self.head(x)


torch.manual_seed(0)
x = torch.randn(512, 64)
y = x[:, :8].prod(-1, keepdim=True).sign() * x[:, 8:16].norm(dim=-1, keepdim=True) / 3    # 随便造一个不太简单的目标
print(f"y 的方差 {y.var():.2f}：一直猜平均值，loss 就是这么大")
for residual in (False, True):
    torch.manual_seed(1)
    model = Deep(residual)
    F.mse_loss(model(x), y).backward()
    grads = [model.layers[i].weight.grad.norm().item() for i in (0, 15, 31)]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(300):
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"{'有残差' if residual else '无残差'}：初始时第 1、16、32 层的梯度 {[f'{g:.0e}' for g in grads]}，训练 300 步后 loss {loss.item():.3f}")
```

```text title="输出"
y 的方差 0.90：一直猜平均值，loss 就是这么大
无残差：初始时第 1、16、32 层的梯度 ['6e-09', '5e-07', '4e-03']，训练 300 步后 loss 0.894
有残差：初始时第 1、16、32 层的梯度 ['1e+01', '1e+01', '1e+01']，训练 300 步后 loss 0.000
```

- 没有残差时，梯度每往回穿过一层，就要乘一次 tanh 的导数和权重矩阵。32 层乘下来，第 1 层的梯度只剩 6×10⁻⁹，比第 32 层小了 6 个数量级，浅层几乎学不到东西。300 步后 loss 还是 0.894，和一直猜平均值（0.90）没区别；
- 有残差时，$x + f(x)$ 对 $x$ 的导数里永远有一个 1，梯度能沿着"加法"这条捷径原样传回去。每层的梯度都在 10 左右，300 步就把训练集拟合到了 0；
- 这正是 ResNet 当年让一百多层的网络能训练起来的办法。Transformer 的每个子层外面都套着这一层加法。

## LayerNorm 与 RMSNorm

**LayerNorm**（原始 Transformer、GPT-2）对每个 token 的 d 维向量做标准化，再做缩放和平移：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

**RMSNorm**（LLaMA、Qwen、DeepSeek 等几乎所有新模型）去掉了减均值和平移，只除以均方根：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

少一次归约、少一组参数，效果却基本一样。实现时有一个细节：**统计量用 FP32 计算**，因为在 BF16 下对几千个数求平方和会损失精度。下面的实现与 transformers 里 Qwen2 的 RMSNorm 逐位一致：

先看两种归一化对同一个向量各做了什么——换几种输入（整体偏移、整体放大、有离群维度）就能看出差别在哪、为什么能省掉减均值：

<div class="aig-widget" data-widget="norm"></div>

```python
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()                                                   # 统计量用 FP32
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)                                # 先转回原精度，再乘权重

torch.manual_seed(0)
ours, ref = RMSNorm(896), Qwen2RMSNorm(896, eps=1e-6)
w = torch.rand(896) + 0.5
ours.weight.data.copy_(w)
ref.weight.data.copy_(w)
for dtype in (torch.float32, torch.bfloat16):
    x = torch.randn(2, 7, 896, dtype=dtype) * 3
    assert torch.equal(ours.to(dtype)(x), ref.to(dtype)(x))            # 逐位相等

# RMSNorm 之后每个 token 向量的均方根为 1（在乘 weight 之前）
y = RMSNorm(896)(torch.randn(4, 896) * 100)
assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-3)
```

"先转回原精度，再乘权重"这种顺序上的细节，决定了 BF16 下的结果能否与官方实现逐位一致。推理引擎在实现融合 kernel 时，要非常小心地复现这些细节。

### 实验：统计量为什么要用 FP32

上面的实现先把输入转成 FP32 再求平方和。如果直接用 BF16 累加会怎样？模拟一个"一个一个往上加"的 BF16 实现：

```python
torch.manual_seed(0)
for d in (1024, 4096):
    v = torch.randn(d) * 3
    exact = v.double().pow(2).mean().sqrt().item()
    acc = torch.tensor(0.0, dtype=torch.bfloat16)
    for sq in v.bfloat16().pow(2):                            # 模拟一个只用 BF16 累加的实现：一个一个往上加
        acc = acc + sq
    naive = (acc.float() / d).sqrt().item()
    fp32 = v.float().pow(2).mean().sqrt().item()
    print(f"{d} 维：精确值 {exact:.4f}；BF16 逐个累加 {naive:.4f}（差 {abs(naive - exact) / exact:.1%}）；FP32 {fp32:.4f}")
```

```text title="输出"
1024 维：精确值 3.0839；BF16 逐个累加 2.8007（差 9.2%）；FP32 3.0839
4096 维：精确值 2.9560；BF16 逐个累加 2.0387（差 31.0%）；FP32 2.9560
```

- BF16 只有 8 位有效位（约两三位十进制数字）。累加到后面，和已经很大，再加一个小小的平方数会被舍掉，结果系统性偏小。维度越大越严重，4096 维差了 31%；
- 归一化要拿这个统计量去缩放整行，误差会原样带进下一层的输入。GPU 上的 kernel 是并行归约，误差比逐个累加小，但同样要用 FP32 累加。

## Pre-Norm 与 Post-Norm

原始 Transformer 把归一化放在残差相加**之后**（Post-Norm）：$x \leftarrow \text{Norm}(x + \text{Sublayer}(x))$。它在层数多时训练很不稳定，需要仔细的学习率预热。

现代大模型都用 **Pre-Norm**：归一化放在子层**之前**，残差流本身不被归一化，只在最后输出之前做一次**最终归一化**（`model.norm`）。残差流于是成为一条"干净"的加法通道，训练稳定得多。代价是残差流的数值可以随着层数不断变大，这一点下面会看到。

![图：Post-Norm 把归一化放在相加之后；Pre-Norm 让残差流成为一条不被归一化的加法通道](../assets/figures/pre-post-norm.svg){.aig-svg}

### 实验：Post-Norm 不加 warmup 会训崩

用 PyTorch 自带的 `nn.TransformerEncoderLayer`（`norm_first` 切换两种结构）搭一个 12 层的小语言模型，在一段中文上按字训练 300 步，学习率 3e-3，比较两种结构，再给 Post-Norm 加上 warmup：

```python
text = open("docs/assets/sample-passage.txt", encoding="utf-8").read()
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])           # 按字切分的小语料
V, T = len(chars), 64


class TinyLM(nn.Module):
    """12 层的小 Transformer；pre=True 用 Pre-Norm，False 用 Post-Norm"""
    def __init__(self, pre, depth=12, d=128):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=pre)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(d) if pre else nn.Identity()    # Pre-Norm 最后要补一个最终归一化
        self.head = nn.Linear(d, V)
        self.mask = nn.Transformer.generate_square_subsequent_mask(T)

    def forward(self, idx):
        h = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for block in self.blocks:
            h = block(h, src_mask=self.mask, is_causal=True)
        return self.head(self.norm(h))


def train(pre, lr=3e-3, warmup=0, steps=300):
    """训练 steps 步，返回最后 30 步的平均 loss"""
    torch.manual_seed(0)
    model = TinyLM(pre)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0)
    gen, losses = torch.Generator().manual_seed(0), []
    for _ in range(steps):
        i = torch.randint(0, len(data) - T - 1, (16,), generator=gen)
        xb = torch.stack([data[j:j + T] for j in i])
        yb = torch.stack([data[j + 1:j + T + 1] for j in i])
        loss = F.cross_entropy(model(xb).flatten(0, 1), yb.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return sum(losses[-30:]) / 30


freq = torch.bincount(data, minlength=V).float() / len(data)
print(f"字表 {V} 个字：什么都没学会时 loss ≈ ln {V} = {math.log(V):.2f}；只记住每个字的频率时 ≈ {-(freq * freq.log()).sum():.2f}")
print(f"Post-Norm，学习率 3e-3，不加 warmup：{train(pre=False):.2f}")
print(f"Pre-Norm，学习率 3e-3，不加 warmup：{train(pre=True):.2f}")
print(f"Post-Norm，学习率 3e-3，前 100 步 warmup：{train(pre=False, warmup=100):.2f}")
```

```text title="输出"
字表 506 个字：什么都没学会时 loss ≈ ln 506 = 6.23；只记住每个字的频率时 ≈ 5.23
Post-Norm，学习率 3e-3，不加 warmup：5.25
Pre-Norm，学习率 3e-3，不加 warmup：0.26
Post-Norm，学习率 3e-3，前 100 步 warmup：0.46
```

- Post-Norm 停在 5.25，和"只记住每个字的频率"（5.23）一样，等于训崩了；同样的学习率，Pre-Norm 训到 0.26；
- 给 Post-Norm 加上 100 步 warmup，学习率从 0 慢慢升到 3e-3，它就能训了（0.46），但还是不如 Pre-Norm；
- 原因（见文末 Xiong 等人的分析）：Post-Norm 在初始化时，靠近输出那几层的参数梯度特别大，学习率一大，第一步就会迈得太远，把模型带到坏的地方；warmup 让开头的步子小一点。Pre-Norm 的残差流是一条不被归一化的加法通道，梯度天然平稳，用不着 warmup。这就是原始 Transformer 离不开 warmup、现代大模型都改用 Pre-Norm 的原因。

## 看看真实的残差流

用 Qwen3-0.6B 看每一层残差流的数值大小：

```pycon
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> path = "models/Qwen3-0.6B"
>>> tok = AutoTokenizer.from_pretrained(path)
>>> model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
>>> text = "推理优化的核心是减少访存。大模型在解码阶段每生成一个词，都要读取全部的权重和缓存。"
>>> ids = tok(text, return_tensors="pt").input_ids
>>> with torch.no_grad():
...     hs = model(ids, output_hidden_states=True).hidden_states
>>> print("layer  rms(其他token)  rms(第0个token)  最大绝对值  位置[token,维度]")  # doctest: +NORMALIZE_WHITESPACE
layer  rms(其他token)  rms(第0个token)  最大绝对值  位置[token,维度]
>>> for layer in [0, 1, 2, 4, 8, 14, 20, 26, 27]:
...     h = hs[layer][0]
...     rms = h.pow(2).mean(-1).sqrt()
...     mx = h.abs().max()
...     pos = (h.abs() == mx).nonzero()[0].tolist()
...     print(layer, round(rms[1:].mean().item(), 2), round(rms[0].item(), 2), round(mx.item(), 1), pos)
0 0.03 0.03 0.2 [8, 126]
1 0.27 0.35 6.9 [7, 35]
2 0.35 0.44 9.4 [2, 35]
4 0.5 218.46 6938.6 [0, 35]
8 0.85 218.34 6934.4 [0, 35]
14 1.72 217.76 6915.6 [0, 35]
20 4.79 218.02 6924.2 [0, 35]
26 16.2 219.23 6955.7 [0, 35]
27 18.05 211.75 6708.4 [0, 35]
```

从这组数字能读出三件事：

1. **残差流随层数增长**：普通 token 的均方根从 0.03 增长到 18，Pre-Norm 结构下这是正常的，每一层的输入都会先经过归一化；
2. **巨大激活（massive activations）**：从第 4 层开始，**第 0 个 token** 在**第 35 维**上出现了约 6900 的值，比普通值大三到四个数量级，并且一直保持到最后一层。它和上一章看到的[注意力汇聚](attention.md#真实模型里的注意力注意力汇聚)是同一个现象的两面：模型用这个固定的巨大值，让第 0 个 token 成为注意力的"垃圾桶"；
3. **离群通道**：即使是普通 token，最大值也总出现在少数固定的维度上（这里还是第 35 维，其次是第 277 维）：

```pycon
>>> h = hs[12][0, 1:]                                  # 第 12 层，去掉第 0 个 token
>>> top = h.abs().amax(dim=0).topk(4)
>>> [(i.item(), round(v.item(), 1)) for v, i in zip(top.values, top.indices)], round(h.abs().median().item(), 3)
([(35, 34.5), (277, 15.0), (62, 10.7), (12, 10.5)], 0.545)
```

第 35 维的最大值是全体数值中位数的约 60 倍。

### 实验：均值几乎为零，每层只改一点

换一段 256 个 token 的长文本，看两件事：残差流的均值有多大（这决定了 RMSNorm 省掉减均值有没有代价），以及每一层的注意力、前馈网络往残差流里写回的向量有多大：

```python
ids_long = tok(open("docs/assets/sample-passage.txt", encoding="utf-8").read(), return_tensors="pt").input_ids[:, :256]
writes, hooks = {}, []
for i, layer in enumerate(model.model.layers):                  # 记下每层注意力、前馈网络写回残差流的向量
    hooks.append(layer.self_attn.register_forward_hook(lambda m, inp, out, i=i: writes.__setitem__(("attn", i), out[0][0])))
    hooks.append(layer.mlp.register_forward_hook(lambda m, inp, out, i=i: writes.__setitem__(("mlp", i), out[0])))
with torch.no_grad():
    hs_long = model(ids_long, output_hidden_states=True).hidden_states
for h in hooks:
    h.remove()

for i in (1, 5, 14, 27):
    stream = hs_long[i][0, 1:]                                    # 第 i 层读到的残差流（去掉有巨大激活的第 0 个 token）
    mean, rms = stream.mean(-1), stream.pow(2).mean(-1).sqrt()
    ln, rn = F.layer_norm(stream, stream.shape[-1:]), stream / rms[:, None]    # 不带缩放参数的 LayerNorm 和 RMSNorm
    a, m = writes[("attn", i)][1:].norm(dim=-1), writes[("mlp", i)][1:].norm(dim=-1)
    print(f"第 {i:2d} 层：均值只有均方根的 {(mean.abs() / rms).mean():.1%}，LayerNorm 与 RMSNorm 相差 {((ln - rn).norm(dim=-1) / ln.norm(dim=-1)).mean():.1%}；"
          f"注意力写回 {(a / stream.norm(dim=-1)).mean():.0%}，前馈网络写回 {(m / stream.norm(dim=-1)).mean():.0%}")
```

```text title="输出"
第  1 层：均值只有均方根的 2.2%，LayerNorm 与 RMSNorm 相差 2.2%；注意力写回 31%，前馈网络写回 62%
第  5 层：均值只有均方根的 2.4%，LayerNorm 与 RMSNorm 相差 2.4%；注意力写回 32%，前馈网络写回 41%
第 14 层：均值只有均方根的 2.6%，LayerNorm 与 RMSNorm 相差 2.6%；注意力写回 22%，前馈网络写回 25%
第 27 层：均值只有均方根的 5.4%，LayerNorm 与 RMSNorm 相差 5.4%；注意力写回 12%，前馈网络写回 36%
```

- 残差流向量的均值只有均方根的 2%～5%，减不减均值几乎没区别：LayerNorm 和 RMSNorm 的结果只差 2%～5%，后面还有可学习的缩放 γ 能吸收。这就是 RMSNorm 敢省掉减均值的原因；
- 每一层写回的向量，只有残差流本身的 12%～62%。每层都是在已有的表示上"改一点"，而不是推倒重来。

### 实验：跳过一层会怎样

既然每层只改一点，少改一次会怎样？依次去掉某一层（其余不变），看困惑度：

```python
def ppl():
    with torch.no_grad():
        logits = model(ids_long, use_cache=False).logits[0, :-1]
    return F.cross_entropy(logits, ids_long[0, 1:]).exp().item()


full, layers, skipped = ppl(), model.model.layers, []
for i in range(len(layers)):
    model.model.layers = nn.ModuleList(l for j, l in enumerate(layers) if j != i)    # 去掉第 i 层，其余照常
    skipped.append(ppl())
model.model.layers = layers
print(f"完整模型：困惑度 {full:.1f}")
print("跳过第 0、1、2 层：", "、".join(f"{p:,.0f}" for p in skipped[:3]))
print(f"跳过第 3～27 层中的任意一层：{min(skipped[3:]):.1f}～{max(skipped[3:]):.1f}")
```

```text title="输出"
完整模型：困惑度 31.8
跳过第 0、1、2 层： 6,064,860、150,241、5,008
跳过第 3～27 层中的任意一层：30.8～52.7
```

- 跳过中间任意一层，困惑度只从 31.8 变成 30.8～52.7，跳过某些层甚至略好一点。少改一次，后面的层照样能接着用；
- 前三层不能跳：跳过第 0 层困惑度 600 万，第 1 层 15 万，第 2 层 5000。第 2 层的输出里，第 0 个 token 第一次出现那个约 7000 的巨大激活（见注意力一章的[注意力汇聚](attention.md#真实模型里的注意力注意力汇聚)），跳过它，汇聚就没了，后面的层全乱套。

### 实验：logit lens，看答案在第几层成形

残差流是所有层共用的"草稿纸"。把中间某一层的残差流直接送进最终归一化和输出层，就能看出模型此刻倾向于哪个词，这个方法叫 logit lens。用注意力汇聚、前馈网络两章都用过的句子试一下：

```python
prompt = "迈克尔·乔丹是一名著名的"
with torch.no_grad():
    hs_q = model(tok(prompt, return_tensors="pt").input_ids, output_hidden_states=True).hidden_states
    for i in (0, 8, 16, 20, 22, 24, 26, 28):
        h = hs_q[i][0, -1]                                        # 经过 i 层之后，最后一个位置的残差流
        logits = model.lm_head(h if i == 28 else model.model.norm(h))   # hidden_states 的最后一项已经做过最终归一化
        p = logits.softmax(-1).topk(3)
        print(f"经过 {i:2d} 层：", "  ".join(f"{tok.decode([int(t)])} {v:.0%}" for v, t in zip(p.values, p.indices)))
```

```text title="输出"
经过  0 层： 著名的 100%  有名的 0%  著名 0%
经过  8 层： ly 12%  _ 8%  iss 6%
经过 16 层： . 23%  , 13%  _ 10%
经过 20 层： a 26%  在 17%  中国 6%
经过 22 层： 美国 31%  篮球 14%  中国 7%
经过 24 层： 篮球 84%  美国 10%  体育 1%
经过 26 层： 篮球 51%  美国 14%  体育 6%
经过 28 层： 篮球 75%  美国 10%  N 4%
```

- 经过 0 层（还是嵌入）时，解码出来就是当前这个词"著名的"本身：Qwen3-0.6B 的输入嵌入和输出层共用一个矩阵，见[嵌入层与输出层](embedding.md)；
- 中间十几层解码出来是"ly""iss""_"这样的碎片：残差流里写的是给后面的层看的中间结果，还不是对下一个词的预测；
- 经过 22 层，"美国""篮球"开始冒头；到 24 层，"篮球"已经是 84%；之后几层在确认和修饰，最终 75%。答案大致是在最后四分之一的层里成形的。

!!! inference "推理视角"
    - **离群值让激活量化变难**：把激活量化成 INT8 时，一个张量共用一个缩放因子，范围由最大值决定。少数维度上的离群值把范围撑得很大，普通数值只能挤在少数几个量化级别里，精度严重损失。SmoothQuant 等方法正是为此设计的：把激活的离群"转移"一部分到权重上，见[量化原理](../inference/quantization.md)；
    - **巨大激活要求足够的数值范围**：约 6900 的值在 FP16（最大约 65504）下还能表示，但只差一个数量级；这也是为什么 BF16（范围与 FP32 相同）更安全，以及 FP8 为什么需要仔细的缩放；
    - **归一化是访存瓶颈的小算子**：它和残差加法常被融合成一个 kernel（`fused_add_rms_norm`），读一次写一次，参见 CUDA 手册的 [Softmax 与归一化](cuda://kernels/softmax-norm/)。

!!! interview "怎么讲清楚"
    讲归一化：残差流是主干，每一层读取、计算、写回；RMSNorm 只除以均方根，比 LayerNorm 少一次归约和一组参数；现代大模型用 Pre-Norm（训练更稳），最后再加一个最终归一化；统计量用 FP32 计算。推理相关的两点：真实模型的残差流里有巨大的激活和固定的离群通道，是激活量化的主要难点；归一化常和残差加法融合成一个 kernel（vLLM 的 `fused_add_rms_norm`）。

!!! note "延伸阅读"
    - [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)（Anthropic）：把残差流看成各层共用的"通信总线"，每层从里面读、往里面写。本章"读写残差流"的说法就来自这里；
    - [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)（nostalgebraist）：用输出层直接解码中间层，是上面 logit lens 实验的出处；
    - [Transformer Layers as Painters](https://arxiv.org/abs/2407.09298)：系统地跳过、调换中间层，发现模型对去掉一两层相当耐受，和"跳过一层"的实验一致；
    - [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)（Xiong 等）：从梯度的角度解释 Post-Norm 为什么离不开 warmup、Pre-Norm 为什么不用；
    - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：RMSNorm 的原始论文，论证减均值那一步可以省掉；
    - [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)：本章和注意力汇聚里看到的巨大激活，在各种模型上的系统研究；
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)（ResNet）：残差连接的起源。

## 练习

**1. 手算 RMSNorm。** x = (3, 4)，γ = (1, 2)，ε = 0。RMSNorm(x) 是多少？

??? success "参考答案"
    均方根 = √((9 + 16) / 2) = √12.5 ≈ 3.536。x / RMS ≈ (0.849, 1.131)，乘以 γ 得 (0.849, 2.263)。

    ```python
    import torch
    x, g = torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0])
    y = x / x.pow(2).mean().sqrt() * g
    assert torch.allclose(y, torch.tensor([0.8485, 2.2627]), atol=1e-4)
    ```

**2. 思考题。** 如果推理时把 RMSNorm 的统计量改成用 BF16 计算，会有什么影响？

??? success "参考答案"
    BF16 只有 8 位有效精度（约 2-3 位十进制有效数字），对几千个平方数累加时，后加入的小数值会被舍入掉，均方根的误差可能达到百分之几。每一层的输入都因此被错误地缩放，误差逐层累积，模型的输出会明显偏离，严重时生成质量下降。这就是为什么所有主流实现都在 FP32 下计算统计量，只在输入输出时使用 BF16。

## 小结

- [x] 残差连接把模型变成一条残差流，每一层读取、计算、写回。
- [x] RMSNorm 只除以均方根，比 LayerNorm 少一次归约和一组参数；统计量用 FP32 计算。
- [x] 现代大模型用 Pre-Norm，最后有一个最终归一化；Post-Norm 学习率稍大又不加 warmup 就会训崩。
- [x] 每层写回的只是残差流的一小部分，跳过中间一层模型几乎不受影响；用 logit lens 能看到答案在后几层成形。
- [x] 真实模型的残差流存在巨大激活（和注意力汇聚相关）和固定的离群通道。
- [x] 离群值是激活量化的主要难点；归一化常与残差加法融合。
