# 数学与 PyTorch 预备

<p class="lead">读懂大模型，需要的数学比想象中少：矩阵乘法、softmax、点积，再加上对张量形状的敏感和对数值精度的基本认识。这一章把这些工具一次讲清楚，并统一全书使用的形状记号。后面每一章都会反复用到。</p>

!!! tip "想系统补数学"
    本章只讲读懂模型代码必需的最小工具集。更系统的内容单独成了一本[数学基础手册](math://)：线性代数与低秩、概率与采样、信息论、反向传播、浮点误差、性能与排队论，每个概念都在真实模型上测量验证。用到时再去查，不需要先通读。

!!! question "自测：能答上来就可以跳过本章"
    1. `nn.Linear(4096, 11008)` 作用在形状为 `[2, 100, 4096]` 的张量上，输出形状是什么？做了多少次浮点运算？
    2. `view` 和 `reshape` 有什么区别？`transpose` 之后为什么有时要 `contiguous()`？
    3. softmax 为什么要先减去最大值？
    4. FP16 和 BF16 各有几位指数、几位尾数？为什么大模型更喜欢 BF16？
    5. 推理时为什么要用 `torch.no_grad()` 或 `torch.inference_mode()`？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 输出 `[2, 100, 11008]`；计算量 $2 \times 2 \times 100 \times 4096 \times 11008 \approx 1.8 \times 10^{10}$ 次浮点运算（每个输出元素要做 4096 次乘加，一次乘加算 2 次运算）。
    2. `view` 只改元数据，要求内存布局能用新的 stride 描述，否则报错；`reshape` 能视图就视图、不能就拷贝。`transpose` 只交换 stride，结果不再连续，很多 kernel 和 `view` 要求连续，所以要 `contiguous()`。
    3. $e^x$ 在 $x$ 大时会溢出；softmax 对所有输入加减同一个常数结果不变，减去最大值后最大的指数是 $e^0 = 1$，不会溢出。
    4. FP16：5 位指数、10 位尾数，精度较好但最大值只有 65504；BF16：8 位指数、7 位尾数，范围和 FP32 一样。大模型训练和推理里溢出比精度损失更致命，BF16 不需要损失缩放，所以更受欢迎。
    5. 关闭梯度记录：不保存反向需要的中间结果（省显存）、不建计算图（省时间）；`inference_mode` 连版本计数也省掉了。

## 形状记号

全书统一使用以下记号：

| 记号 | 含义 | Qwen3-0.6B 的取值 |
| --- | --- | --- |
| $B$ | batch 大小（同时处理的序列数） | — |
| $T$ | 序列长度（token 数） | — |
| $V$ | 词表大小 | 151936 |
| $d$ | 隐藏维度（hidden_size） | 1024 |
| $n_h$ | 注意力头数 | 16 |
| $n_{kv}$ | KV 头数（GQA） | 8 |
| $d_h$ | 每个头的维度（head_dim），通常 $d / n_h$，Qwen3 单独配置 | 128 |
| $d_{ff}$ | 前馈网络中间维度（intermediate_size） | 3072 |
| $L$ | 层数 | 28 |

Transformer 里的张量，最常见的形状是 `[B, T, d]`（每个 token 一个 d 维向量）和 `[B, n_h, T, d_h]`（按头拆开之后）。**读代码时，在每一行旁边写下张量的形状**，是理解模型最有效的习惯。

## 矩阵乘法：线性层

`nn.Linear(in_features, out_features)` 计算 $y = xW^\top + b$，权重 $W$ 的形状是 `[out, in]`。它只作用在**最后一个维度**上，前面的维度都当作批量：

```pycon
>>> import torch
>>> import torch.nn as nn
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<torch._C.Generator object at 0x...>
>>> lin = nn.Linear(4096, 11008, bias=False)
>>> lin.weight.shape
torch.Size([11008, 4096])
>>> x = torch.randn(2, 100, 4096)
>>> lin(x).shape
torch.Size([2, 100, 11008])
```

**计算量**：一个 `[M, K] × [K, N]` 的矩阵乘法需要 $M \times N \times K$ 次乘加，也就是 $2MNK$ 次浮点运算（FLOPs）。上面的例子中 M = 2 × 100 = 200，所以是 2 × 200 × 11008 × 4096 ≈ 180 亿次。

由此得到一个非常有用的结论：**一个线性层对每个 token 的计算量 = 2 × 参数量**。整个模型的绝大部分参数都在线性层里，所以

$$
\text{每个 token 的前向计算量} \approx 2 \times \text{参数量}
$$

一个 70 亿参数的模型，生成每个 token 大约需要 140 亿次浮点运算（还要加上注意力的部分，见[估算](../inference/estimation.md)）。

### einsum：用下标描述张量运算

`torch.einsum` 用下标字符串描述运算，对复杂的张量运算特别清楚：

```pycon
>>> q = torch.randn(2, 14, 5, 64)             # [B, n_h, T, d_h]
>>> k = torch.randn(2, 14, 5, 64)
>>> scores = torch.einsum("bhqd,bhkd->bhqk", q, k)   # 每个头里，每个 query 和每个 key 做点积
>>> scores.shape
torch.Size([2, 14, 5, 5])
>>> torch.allclose(scores, q @ k.transpose(-2, -1), atol=1e-5)
True
```

出现在输入但不出现在输出里的下标（这里的 `d`）被求和。

## 变形：view、reshape、transpose

多头注意力里最常见的操作是把 `[B, T, n_h × d_h]` 拆成 `[B, n_h, T, d_h]`：

```pycon
>>> B, T, nh, hd = 2, 5, 14, 64
>>> x = torch.randn(B, T, nh * hd)
>>> heads = x.view(B, T, nh, hd).transpose(1, 2)   # 先拆最后一维，再把头的维度换到前面
>>> heads.shape
torch.Size([2, 14, 5, 64])
>>> heads.is_contiguous()
False
>>> back = heads.transpose(1, 2).reshape(B, T, nh * hd)   # 合并回去
>>> torch.equal(back, x)
True
```

- `view` 只改变看待内存的方式，不复制数据，要求内存布局兼容；
- `transpose` 也不复制数据，只是交换了跨度（stride），结果在内存中不再连续；
- 对不连续的张量，`view` 会报错，`reshape` 会在必要时复制一份。`contiguous()` 显式地复制成连续布局。

"不复制、只改跨度"是什么意思，打印出来看：跨度（stride）是每一维走一步，在内存里要跳过几个数：

```pycon
>>> x.stride()                             # 连续存放：最后一维相邻的两个数隔 1 个位置，T 维隔 896 个
(4480, 896, 1)
>>> heads.stride()                         # view + transpose 只改了每一维的跨度，数据没动
(4480, 64, 896, 1)
>>> heads.data_ptr() == x.data_ptr()       # 和 x 是同一块内存
True
>>> c = heads.contiguous()
>>> c.stride(), c.data_ptr() == x.data_ptr()   # contiguous() 按新的维度顺序重排，复制了一份
((4480, 320, 64, 1), False)
```

`heads[b, h, t, i]` 在内存里的位置是 `b × 4480 + h × 64 + t × 896 + i`：同一个头里相邻的两个 token 隔着 896 个数，所以不再连续。

!!! inference "推理视角"
    这些"不复制数据"的变形在 GPU 上几乎零开销，但一旦触发复制（`contiguous()`、`reshape`），就是一次完整的读写。高性能 kernel 通常直接按需要的布局读取数据，或者把转置融合进相邻的算子，避免单独的转置 kernel，参见 CUDA 手册中的[矩阵转置](cuda://kernels/transpose/)。

## 广播

形状不同的张量做逐元素运算时，从最后一维开始对齐，大小为 1 的维度会被"复制"到匹配的大小：

```pycon
>>> x = torch.randn(2, 5, 896)       # [B, T, d]
>>> w = torch.randn(896)             # [d]：RMSNorm 的权重
>>> (x * w).shape                    # w 被广播到每个 batch、每个位置
torch.Size([2, 5, 896])
>>> mean_sq = x.pow(2).mean(dim=-1, keepdim=True)   # keepdim 保留大小为 1 的维度，方便广播
>>> mean_sq.shape
torch.Size([2, 5, 1])
```

广播出错是写模型代码时最常见的报错之一。规则只有两条，拿几种常见的形状对一对就记住了：

<div class="aig-widget" data-widget="broadcast"></div>

## softmax 与数值稳定

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

它把任意实数向量变成概率分布。直接计算会溢出：

```pycon
>>> z = torch.tensor([1000.0, 1001.0, 1002.0])
>>> torch.exp(z) / torch.exp(z).sum()      # e^1000 溢出成 inf
tensor([nan, nan, nan])
>>> zs = z - z.max()                        # 减去最大值，数学上结果不变
>>> torch.exp(zs) / torch.exp(zs).sum()
tensor([0.0900, 0.2447, 0.6652])
>>> torch.softmax(z, dim=0)                 # 库函数内部就是这样做的
tensor([0.0900, 0.2447, 0.6652])
```

两个常用的性质：softmax 只和各个 logits 之间的**差**有关（整体加一个常数不变）；logits 整体除以一个大于 1 的数，分布会变"平"，这就是采样时的**温度**（见[解码与采样](../inference/decoding.md)）。计算损失时应该用 `log_softmax` 或 `F.cross_entropy`，它们内部把 log 和 softmax 合并计算，数值更稳定。

## 点积与相似度

两个向量的点积 $a \cdot b = \|a\| \|b\| \cos\theta$ 衡量它们的"相似度"。注意力机制的核心就是：用 query 向量和每个 key 向量的点积，决定从哪些位置读取信息（见[注意力](../transformer/attention.md)）。高维随机向量的点积有一个重要的性质，它是注意力里除以 $\sqrt{d_h}$ 的原因：

```pycon
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<torch._C.Generator object at 0x...>
>>> for d in (16, 64, 256, 1024):
...     a, b = torch.randn(10000, d), torch.randn(10000, d)
...     print(d, round((a * b).sum(-1).std().item(), 1))
...
16 4.0
64 8.0
256 16.2
1024 32.1
```

两个各分量为标准正态的 d 维随机向量，点积的标准差约为 $\sqrt{d}$。维度越高点积越大，直接送进 softmax 会让分布极其尖锐，梯度几乎为零。除以 $\sqrt{d}$ 把它拉回单位尺度。

## 数值格式

| 格式 | 符号/指数/尾数 | 每个数的字节 | 最大值 | 相对精度（机器 epsilon） |
| --- | --- | --- | --- | --- |
| FP32 | 1/8/23 | 4 | 约 3.4e38 | 约 1.2e-7 |
| FP16 | 1/5/10 | 2 | 65504 | 约 9.8e-4 |
| BF16 | 1/8/7 | 2 | 约 3.4e38 | 约 7.8e-3 |

```pycon
>>> for dt in (torch.float32, torch.float16, torch.bfloat16):
...     fi = torch.finfo(dt)
...     print(str(dt), fi.bits, f"{fi.max:.3g}", f"{fi.eps:.2g}")
...
torch.float32 32 3.4e+38 1.2e-07
torch.float16 16 6.55e+04 0.00098
torch.bfloat16 16 3.39e+38 0.0078
>>> torch.tensor(70000.0).half()          # 超出 FP16 的范围
tensor(inf, dtype=torch.float16)
>>> torch.tensor(70000.0).bfloat16()      # BF16 范围够，但精度粗
tensor(70144., dtype=torch.bfloat16)
```

大模型的激活值偶尔会很大，FP16 容易溢出；BF16 的范围和 FP32 一样，只是精度低一些，而神经网络对精度的容忍度相当高。所以现在大模型的训练和推理几乎都用 BF16 存储权重和激活，关键的累加（矩阵乘法的累加器、softmax、归一化的统计量）用 FP32 计算。

### 实验：累加器为什么要用 FP32

BF16 只有 8 位有效数字，大数加小数时，小数会被"吃掉"。用 BF16 当累加器，一个一个地加 1：

```pycon
>>> s = torch.tensor(0.0, dtype=torch.bfloat16)
>>> for v in torch.ones(10000, dtype=torch.bfloat16):
...     s += v
...
>>> s
tensor(256., dtype=torch.bfloat16)
>>> torch.ones(10000, dtype=torch.bfloat16).float().sum()   # 数据还是 BF16，累加改用 FP32
tensor(10000.)
```

加到 256 就停住了：256 到 512 之间，BF16 相邻的两个数相差 2，256 + 1 舍入回 256。换成一个长 4096 的点积，也就是线性层里每个输出要做的事：

```pycon
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<torch._C.Generator object at 0x...>
>>> a, b = torch.randn(4096).bfloat16(), torch.randn(4096).bfloat16()
>>> exact = (a.double() * b.double()).sum()
>>> acc = torch.tensor(0.0, dtype=torch.bfloat16)
>>> for p in a * b:
...     acc += p
...
>>> round(((acc - exact) / exact).abs().item(), 3)                        # BF16 累加的相对误差
0.177
>>> round((((a.float() * b.float()).sum() - exact) / exact).abs().item(), 8)   # FP32 累加
2.3e-07
```

同样的 BF16 数据，BF16 累加错了 18%，FP32 累加只差千万分之二。所以 Tensor Core 的矩阵乘法输入是 BF16，累加器却是 FP32，RMSNorm 的统计量也要先转成 FP32 再算（[归一化](../transformer/norm-residual.md)一章有实验）。

!!! inference "推理视角"
    **每个参数占多少字节，直接决定了显存占用和 decode 速度**。BF16 下每个参数 2 字节，一个 70 亿参数的模型权重约 14 GB；量化到 INT4 只需约 3.5 GB（外加少量缩放因子），decode 时每步读取的数据也少了四分之三，见[量化原理](../inference/quantization.md)。

## nn.Module 与参数

PyTorch 的模型是 `nn.Module` 的嵌套结构，参数保存在 `state_dict` 里，按层级命名。数参数量、查看结构都很方便：

```pycon
>>> class TinyBlock(nn.Module):
...     def __init__(self, d, dff):
...         super().__init__()
...         self.up = nn.Linear(d, dff, bias=False)
...         self.down = nn.Linear(dff, d, bias=False)
...     def forward(self, x):
...         return self.down(torch.relu(self.up(x)))
...
>>> block = TinyBlock(896, 4864)
>>> sum(p.numel() for p in block.parameters())
8716288
>>> [(k, tuple(v.shape)) for k, v in block.state_dict().items()]
[('up.weight', (4864, 896)), ('down.weight', (896, 4864))]
```

加载预训练模型，本质上就是把文件里的张量按名字填进 `state_dict`。我们在[从零组装一个大模型](../transformer/build-llm.md)时会亲手做这件事。

## 推理时关掉梯度

训练时，PyTorch 会为每一步运算保存反向传播所需的中间结果。推理不需要反向传播，保存它们只会浪费显存和时间：

```pycon
>>> w = torch.randn(4, 4, requires_grad=True)
>>> y = (w @ torch.randn(4)).sum()
>>> y.requires_grad
True
>>> with torch.inference_mode():
...     y2 = (w @ torch.randn(4)).sum()
...
>>> y2.requires_grad
False
```

`torch.no_grad()` 和 `torch.inference_mode()` 都能关闭梯度记录，后者更彻底、开销更小。

### 实验：梯度记录多存了什么

PyTorch 提供了一个钩子，前向时每一个"要留给反向用"的张量都会经过它。拿上面的 `TinyBlock` 数一数：

```python
saved = {}
params = {p.untyped_storage().data_ptr() for p in block.parameters()}

def pack(t):
    """按存储去重，参数本来就在内存里，不算"""
    if t.untyped_storage().data_ptr() not in params:
        saved[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return t

x = torch.randn(4, 512, 896)                                   # 4 条序列、每条 512 个 token
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    block(x)
print(f"普通前向：为反向保存了 {len(saved)} 个张量，共 {sum(saved.values()) / 2**20:.0f} MB（输入只有 {x.numel() * 4 / 2**20:.0f} MB）")
saved.clear()
with torch.inference_mode(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    block(x)
print(f"inference_mode：保存了 {len(saved)} 个")
```

```text title="输出"
普通前向：为反向保存了 2 个张量，共 45 MB（输入只有 7 MB）
inference_mode：保存了 0 个
```

保存的是 `up` 的输入（7 MB，算 `up` 权重的梯度要用）和 ReLU 的输出（38 MB，算 ReLU 和 `down` 的梯度都要用）。一个两层的小块就要多存 6 倍于输入的数据，几十层叠起来，训练时激活占的显存常常比权重还多；推理用不上这些，`inference_mode` 下一个都不存。

!!! interview "怎么讲清楚"
    这一章的内容常以口算的形式出现：线性层作用在 `[B, T, d_in]` 上，输出 `[B, T, d_out]`，计算量 $2 \cdot B \cdot T \cdot d_{in} \cdot d_{out}$，所以整个模型每个 token 约 2 × 参数量次运算；BF16 有 8 位指数、7 位尾数，范围和 FP32 一样、不需要损失缩放，FP16 精度更好但最大值只有 65504；softmax 先减最大值防溢出；推理用 `torch.inference_mode()` 省掉梯度记录和版本计数。讲的时候顺手写出张量形状，会显得很扎实。

!!! note "延伸阅读"
    - [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)（Edward Yang）：张量、跨度、视图在 PyTorch 内部是怎么表示的，本章 view 和 transpose 的底层原理；
    - [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html) 和 [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)（PyTorch 文档）：哪些操作返回视图、广播的完整规则；
    - [Einsum is All you Need](https://rockt.ai/2018/04/30/einsum)（Tim Rocktäschel）：einsum 的写法和常见用法；
    - [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)（Google）：BF16 为什么这样设计，以及混合精度怎么用；
    - 数学手册的[浮点与数值计算](math://floating-point/)：舍入误差、累加顺序、混合精度的系统讲解，每个结论都有实验。

## 练习

**1. 计算量。** 一个线性层 `nn.Linear(896, 4864)`，输入 `[4, 512, 896]`。输出形状是多少？需要多少 FLOPs？权重在 BF16 下占多少字节？

??? success "参考答案"
    输出 `[4, 512, 4864]`。M = 4 × 512 = 2048，FLOPs = 2 × 2048 × 4864 × 896 ≈ 178 亿。权重 4864 × 896 ≈ 436 万个参数，BF16 下约 8.7 MB。

    ```python
    M, K, N = 4 * 512, 896, 4864
    assert 2 * M * N * K == 17_850_957_824
    assert N * K * 2 == 8_716_288   # 字节
    ```

**2. 形状变换。** 把形状为 `[B, T, n_kv, d_h]` 的 K 扩展成 `[B, n_h, T, d_h]`，使每个 KV 头被连续的 $n_h / n_{kv}$ 个 query 头共享（GQA）。写出代码。

??? success "参考答案"
    ```python
    import torch
    B, T, n_h, n_kv, d_h = 2, 5, 14, 2, 64
    k = torch.randn(B, T, n_kv, d_h)
    k_heads = k.transpose(1, 2).repeat_interleave(n_h // n_kv, dim=1)   # [B, n_h, T, d_h]
    assert k_heads.shape == (B, n_h, T, d_h)
    assert torch.equal(k_heads[:, 6], k_heads[:, 0]) and torch.equal(k_heads[:, 7], k.transpose(1, 2)[:, 1])
    ```

    query 头 0-6 共享 KV 头 0，query 头 7-13 共享 KV 头 1。实际的推理 kernel 不会真的复制 K，而是让多个 query 头读取同一份 K，见[注意力变体](../transformer/attention-variants.md)。

## 小结

- [x] 在代码旁标注张量形状；Transformer 里最常见的是 `[B, T, d]` 和 `[B, n_h, T, d_h]`。
- [x] 线性层对每个 token 的计算量是 2 × 参数量，所以整个模型每个 token 约 2 × 参数量次运算。
- [x] softmax 先减最大值；点积随维度增大，注意力要除以 $\sqrt{d_h}$。
- [x] 大模型用 BF16 存储、FP32 累加（BF16 累加 1 万个 1 只能加到 256，长 4096 的点积错 18%）；每个参数的字节数决定显存和 decode 速度。
- [x] view、transpose 只改跨度不复制，contiguous() 才复制；带梯度的前向要为反向保存中间结果，推理用 inference_mode 一个都不存。
- [x] 推理时用 `torch.inference_mode()` 关闭梯度记录。
