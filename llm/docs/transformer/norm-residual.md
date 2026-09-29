# 归一化与残差流

<p class="lead">残差连接让几十上百层的网络能够训练，归一化让每一层的输入保持在合适的尺度。它们的计算量微不足道，却决定了模型的数值特性：残差流中的"巨大激活"和"离群通道"，正是量化和低精度推理最头疼的问题。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 残差连接是什么？为什么说每一层是在"读写残差流"？
    2. LayerNorm 和 RMSNorm 的公式分别是什么？RMSNorm 省掉了什么？
    3. Pre-Norm 和 Post-Norm 的区别？现代大模型用哪种？
    4. 为什么归一化的统计量要用 FP32 计算？
    5. 什么是激活离群值？它对推理有什么影响？

## 残差连接

每个子层（注意力、前馈网络）的输出不是直接替换输入，而是**加回**到输入上：

$$
x \leftarrow x + \text{Sublayer}(\text{Norm}(x))
$$

好处有两个：梯度可以沿着"加法"这条捷径直接传回浅层，几十上百层也能训练；每一层只需要学习"在现有表示上做什么修改"，而不是从头构造新的表示。

从这个角度看，整个模型就是一条贯穿始终的**残差流**：嵌入层把 token 写进去，每一层从中读取（经过归一化）、计算、把结果加回去，最后由输出层读出。一个 Transformer 层的完整结构：

```py
h = x + attention(rms_norm_1(x))      # 注意力子层：token 之间交换信息
y = h + mlp(rms_norm_2(h))            # 前馈子层：每个 token 独立加工
```

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

## Pre-Norm 与 Post-Norm

原始 Transformer 把归一化放在残差相加**之后**（Post-Norm）：$x \leftarrow \text{Norm}(x + \text{Sublayer}(x))$。它在层数多时训练很不稳定，需要仔细的学习率预热。

现代大模型都用 **Pre-Norm**：归一化放在子层**之前**，残差流本身不被归一化，只在最后输出之前做一次**最终归一化**（`model.norm`）。残差流于是成为一条"干净"的加法通道，训练稳定得多。代价是残差流的数值可以随着层数不断变大，这一点下面会看到。

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

!!! inference "推理视角"
    - **离群值让激活量化变难**：把激活量化成 INT8 时，一个张量共用一个缩放因子，范围由最大值决定。少数维度上的离群值把范围撑得很大，普通数值只能挤在少数几个量化级别里，精度严重损失。SmoothQuant 等方法正是为此设计的：把激活的离群"转移"一部分到权重上，见[量化原理](../inference/quantization.md)；
    - **巨大激活要求足够的数值范围**：约 6900 的值在 FP16（最大约 65504）下还能表示，但只差一个数量级；这也是为什么 BF16（范围与 FP32 相同）更安全，以及 FP8 为什么需要仔细的缩放；
    - **归一化是访存瓶颈的小算子**：它和残差加法常被融合成一个 kernel（`fused_add_rms_norm`），读一次写一次，参见 CUDA 手册的 [Softmax 与归一化](cuda://kernels/softmax-norm/)。

!!! interview "面试怎么答"
    归一化题：残差流是主干，每一层读取、计算、写回；RMSNorm 只除以均方根，比 LayerNorm 少一次归约和一组参数；现代大模型用 Pre-Norm（训练更稳），最后再加一个最终归一化；统计量用 FP32 计算。推理相关的两点：真实模型的残差流里有巨大的激活和固定的离群通道，是激活量化的主要难点；归一化常和残差加法融合成一个 kernel（vLLM 的 `fused_add_rms_norm`）。

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
- [x] 现代大模型用 Pre-Norm，最后有一个最终归一化。
- [x] 真实模型的残差流存在巨大激活（和注意力汇聚相关）和固定的离群通道。
- [x] 离群值是激活量化的主要难点；归一化常与残差加法融合。
