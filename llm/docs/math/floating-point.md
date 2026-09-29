# 浮点与数值计算

<p class="lead">推理优化天天和低精度打交道：BF16 的权重与激活、FP8 的 GEMM、FP4 的 MoE 专家、FP32 的累加器。很多"莫名其妙"的现象都是浮点算术的直接后果：同一个请求两次结果不同、加了一个数和没加一样、归一化突然输出全零、低精度累加的和完全不对。这一章讲清浮点数的表示与误差，并用一组小实验复现这些现象，最后给出量化噪声的一个实用公式。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. BF16 和 FP16 的机器精度（eps）与最大值分别是多少？各自的风险是什么？
    2. 为什么在 BF16 中 1650 + 1 仍然等于 1650 附近的同一个数？这对残差流意味着什么？
    3. 用 BF16 累加 2 万个 0～1 之间的数，结果可能错到什么程度？Tensor Core 为什么用 FP32 累加？
    4. 为什么 RMSNorm 的统计量要用 FP32 计算？
    5. 每增加 1 比特的量化位宽，信噪比大约提高多少？

??? success "自测参考答案（先自己答，再展开对照）"
    1. BF16：eps 约 0.0078（$2^{-7}$），最大值约 $3.4 \times 10^{38}$，风险是精度低；FP16：eps 约 0.00098（$2^{-10}$），最大值 65504，风险是溢出（平方、指数、大激活）和小梯度下溢。
    2. BF16 只有 7 位尾数，1024～2048 之间相邻两个数相差 8，加 1 小于间隔的一半，舍入后不变。残差流里如果有很大的激活，后面各层写进去的小更新会被吞掉——所以累加要用更高的精度。
    3. 累加值变大后，间隔超过了要加的数，后面的数全被吞掉：本章实测 BF16 逐个累加 2 万个 0～1 之间的数只得到 256。Tensor Core 用 FP32 累加器，就是为了不让矩阵乘的长累加出现这种错误。
    4. RMSNorm 要对一行做平方和：低精度下平方可能溢出（FP16），长累加会严重丢精度，而这个统计量会作用到整行的每一个数上，所以统计量用 FP32 算、再转回低精度。
    5. 约 6 dB（每多 1 比特，量化噪声的幅度减半、功率变为 1/4）。

## 浮点数的表示

一个浮点数由符号、指数、尾数三部分组成：$x = (-1)^s \times 1.m \times 2^{e - \text{bias}}$。指数位决定**范围**，尾数位决定**精度**：在 $[2^k, 2^{k+1})$ 之间，相邻两个可表示数的间隔（ulp）是 $2^{k} \times \text{eps}$，其中 $\text{eps} = 2^{-\text{尾数位数}}$。**精度是相对的**：数越大，间隔越大。

```python
import math
import torch

print(f"{'格式':16s}{'位数':>4s}{'eps（相对精度）':>16s}{'最大值':>14s}{'最小正规数':>14s}")
for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
    f = torch.finfo(dtype)
    print(f"{str(dtype).replace('torch.', ''):16s}{f.bits:4d}{f.eps:16.2e}{f.max:14.4g}{f.tiny:14.2e}")
```

```text
格式                位数       eps（相对精度）           最大值         最小正规数
float32           32        1.19e-07     3.403e+38      1.18e-38
float16           16        9.77e-04      6.55e+04      6.10e-05
bfloat16          16        7.81e-03      3.39e+38      1.18e-38
float8_e4m3fn      8        1.25e-01           448      1.56e-02
float8_e5m2        8        2.50e-01     5.734e+04      6.10e-05
```

- **FP16**：精度较好（eps ≈ $10^{-3}$），但最大值只有 65504，很容易溢出；
- **BF16**：指数位与 FP32 相同，范围足够大，但精度只有 eps ≈ $8 \times 10^{-3}$，相当于两三位有效数字；
- **FP8 E4M3**：最大值 448，相对精度 12.5%，必须配合缩放因子使用（见[量化部署](serving://perf/quantization-deploy/)）；E5M2 范围更大、精度更低。

## 舍入与"加了等于没加"

每一次浮点运算的结果都要舍入到最近的可表示数，相对误差最多为 eps/2。当一个大数加上一个小数时，小数可能完全被舍掉：

```pycon
>>> x = torch.tensor(1650.0, dtype=torch.bfloat16)
>>> x.item(), (x + 1).item(), (x + 4).item(), (x + 5).item()     # BF16 在 1024～2048 之间的间隔是 8
(1648.0, 1648.0, 1648.0, 1656.0)
>>> a = torch.tensor(1e8)                                          # FP32 在 1e8 附近的间隔是 8
>>> ((a + 1) - a).item(), ((a - a) + 1).item()                     # 加法不满足结合律
(0.0, 1.0)
```

1650 连本身都存不准（变成 1648），加 1、加 4 都没有任何效果。大模型手册在[残差流](../transformer/norm-residual.md#看看真实的残差流)中看到过这样量级的"巨大激活"：在 BF16 的残差流中，与它同一个位置的小更新会被直接吞掉。第二个例子说明浮点加法**不满足结合律**，换一个计算顺序结果就不同，这是"批大小不同、结果不同"的根源（见[哪些优化会改变输出](../synthesis/token-journey.md#哪些优化会改变输出)）。

## 累加：误差会积累

求和是矩阵乘法、归约、注意力的基本操作。用 BF16 逐个累加 2 万个 $[0, 1)$ 之间的随机数，并比较几种求和方法：

```python
torch.manual_seed(0)
values = torch.rand(20000)
exact = values.double().sum().item()
v16 = values.to(torch.bfloat16)

total = torch.tensor(0.0, dtype=torch.bfloat16)                    # 1. 朴素：BF16 累加器，逐个相加
for v in v16:
    total = total + v
naive = total.item()

pairs = v16.clone()                                                # 2. 两两求和（树形归约），仍然全程 BF16
while pairs.numel() > 1:
    if pairs.numel() % 2:
        pairs = torch.cat([pairs, torch.zeros(1, dtype=pairs.dtype)])
    pairs = pairs[0::2] + pairs[1::2]
pairwise = pairs.item()

total, comp = torch.tensor(0.0, dtype=torch.bfloat16), torch.tensor(0.0, dtype=torch.bfloat16)
for v in v16:                                                      # 3. Kahan 补偿求和：记住每次被舍掉的部分
    y = v - comp
    t = total + y
    comp = (t - total) - y
    total = t
kahan = total.item()

fp32_acc = v16.float().sum().item()                                # 4. 输入是 BF16，但用 FP32 累加
for name, value in [("BF16 逐个累加", naive), ("BF16 两两求和", pairwise), ("BF16 Kahan 求和", kahan),
                    ("FP32 累加器", fp32_acc)]:
    print(f"{name:14s} {value:10.1f}   相对误差 {abs(value - exact) / exact:.2%}")
print(f"{'精确值':14s} {exact:10.1f}")
```

```text
BF16 逐个累加           256.0   相对误差 97.45%
BF16 两两求和          9984.0   相对误差 0.50%
BF16 Kahan 求和     10048.0   相对误差 0.14%
FP32 累加器          10034.0   相对误差 0.00%
精确值               10034.2
```

朴素的 BF16 累加结果是 **256**，而正确答案约一万：当累加和达到 256 时，BF16 的间隔变成 2，再加一个小于 1 的数，舍入后原封不动，求和就此"卡死"。两两求和让相加的两个数量级接近，误差从"错得离谱"降到 0.5%；Kahan 求和用一个补偿变量记住被舍掉的部分；而最有效的办法是**用更高的精度累加**：这正是 Tensor Core 在 BF16/FP8 乘法后使用 FP32 累加器的原因，也是 GEMM 中 split-K、注意力中分块归约的数值意义。DeepSeek-V3 的 FP8 GEMM 甚至专门把部分和定期"提升"到 CUDA Core 上用 FP32 累加，就是为了控制这类误差。

## 抵消与溢出

**灾难性抵消**：两个很接近的大数相减，有效数字几乎全部抵消，剩下的全是误差。经典例子是用 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 计算方差：

```python
data = 10000 + torch.randn(100_000)                                # 均值 1 万、方差约 1 的数据
one_pass = (data.pow(2).mean() - data.mean().pow(2)).item()        # 两个约 1 亿的数相减
two_pass = ((data - data.mean()) ** 2).mean().item()               # 先减去均值再平方
print(f"单遍公式：{one_pass:.4f}    两遍公式：{two_pass:.4f}    （真实方差约为 1）")
```

```text
单遍公式：0.0000    两遍公式：1.0006    （真实方差约为 1）
```

FP32 下单遍公式给出的方差是 0，彻底错误。LayerNorm 的实现必须先减均值（或者用 Welford 这类在线算法）；RMSNorm 不减均值，也就绕开了这个问题，这也是它更受欢迎的原因之一。

**溢出**：FP16 的最大值只有 65504，$e^{11.1}$ 就超过了它。softmax 如果不先减去最大值，稍大一点的 logit 就会得到 inf，除法之后变成 NaN；RMSNorm 如果在 FP16 中计算平方，1650 的平方约为 270 万，同样溢出：

```pycon
>>> z = torch.tensor([12.0, 3.0, -1.0], dtype=torch.float16)
>>> (z.exp() / z.exp().sum()).tolist()                              # 朴素 softmax
[nan, 0.0, 0.0]
>>> [round(v, 4) for v in ((z - z.max()).exp() / (z - z.max()).exp().sum()).tolist()]   # 先减最大值
[1.0, 0.0001, 0.0]
>>> h = torch.tensor([1650.0, 3.0, -2.0, 0.5], dtype=torch.float16)
>>> (h * torch.rsqrt(h.pow(2).mean() + 1e-6)).tolist()            # FP16 中的 RMSNorm：平方溢出，输出全为 0
[0.0, 0.0, -0.0, 0.0]
>>> [round(v, 4) for v in (h.float() * torch.rsqrt(h.float().pow(2).mean() + 1e-6)).tolist()]   # 统计量用 FP32
[2.0, 0.0036, -0.0024, 0.0006]
```

FP16 的 RMSNorm 没有报错，只是**静默地**输出了全零，这类问题最难排查。所以本书的 `mini_llm` 与 vLLM 的 RMSNorm kernel 都把统计量放在 FP32 中计算（见[归一化与残差流](../transformer/norm-residual.md#layernorm-与-rmsnorm)）。

## 量化噪声：每比特 6 dB

均匀量化（步长 $\Delta$）把每个数舍入到最近的格点，误差近似均匀分布在 $[-\Delta/2, \Delta/2]$，方差为 $\Delta^2/12$。位宽每增加 1 比特，$\Delta$ 减半，噪声功率降为 1/4，**信噪比提高约 6.02 dB**。在标准正态数据上验证（量化范围固定为 ±4σ）：

```python
g = torch.randn(1_000_000)
prev = None
for bits in (4, 5, 6, 7, 8):
    qmax = 2 ** (bits - 1) - 1
    scale = 4.0 / qmax
    q = (g / scale).round().clamp(-qmax - 1, qmax) * scale
    snr = 10 * math.log10(g.var().item() / (g - q).var().item())
    gain = "" if prev is None else f"（比上一行多 {snr - prev:.2f} dB）"
    print(f"INT{bits}：信噪比 {snr:5.2f} dB{gain}")
    prev = snr

y = torch.linspace(1, 2, 10001)
print(f"FP8 E4M3 在 [1, 2) 内的最大相对舍入误差：{((y.to(torch.float8_e4m3fn).float() - y).abs() / y).max():.3f}")
```

```text
INT4：信噪比 15.66 dB
INT5：信噪比 22.26 dB（比上一行多 6.60 dB）
INT6：信噪比 28.56 dB（比上一行多 6.30 dB）
INT7：信噪比 34.67 dB（比上一行多 6.11 dB）
INT8：信噪比 40.58 dB（比上一行多 5.90 dB）
FP8 E4M3 在 [1, 2) 内的最大相对舍入误差：0.059
```

实测每比特约 6 dB，与理论一致。这给了一个很好用的估算尺度：INT8 比 INT4 的信噪比高约 24 dB，也就是噪声幅度小约 16 倍。FP8 与整数不同，它的误差是**相对的**：在正规数范围内最大相对误差约 $2^{-4} \approx 6\%$，与数值的大小无关，这正是浮点格式能容忍离群值、而整数格式需要精心选择缩放的原因。

!!! interview "面试怎么答"
    数值题先报几个数：BF16 的 eps 约 0.0078，FP16 的最大值 65504；大数加小数会被吞掉，浮点加法不满足结合律，所以 batch 组成会改变结果；低精度累加会错得离谱（BF16 逐个累加 2 万个 0～1 之间的数只得到 256），所以 Tensor Core 用 FP32 累加、归一化和 softmax 的统计量用 FP32；量化每多 1 比特信噪比约高 6 dB，FP8 的相对误差约 6%。能把"为什么推理结果不逐位一致"和这些事实联系起来，是这类题的高分答法。

## 练习

**1. 累加器的位宽。** 一个 FP8 GEMM 的 K 维长度是 8192，每个乘积的相对误差约为 FP8 的舍入误差。如果用 FP16 累加，累加和的误差大约是什么量级？为什么 FP32 累加器足够？

??? success "参考思路"
    顺序累加时，舍入误差大致随加法次数增长，最坏情况约为 $K \times \text{eps}_\text{acc}$ 量级的相对误差（随机情况下约 $\sqrt{K}\,\text{eps}$）。FP16 的 eps ≈ $10^{-3}$，$\sqrt{8192} \approx 90$，相对误差可达约 10%，与本章 BF16 累加的例子类似，不可接受；FP32 的 eps ≈ $10^{-7}$，即使最坏情况也只有约 $10^{-3}$，足够。实际 kernel 还会分块累加（相当于两两求和的变体），进一步降低误差。

**2. 为什么 BF16 训练不需要 loss scaling，FP16 需要？**

??? success "参考答案"
    梯度常常很小（例如 $10^{-6}$ 以下），FP16 的最小正规数约 $6 \times 10^{-5}$，更小的值会变成非正规数甚至下溢为 0，梯度信息丢失；loss scaling 先把损失乘以一个大数，让梯度落在 FP16 能表示的范围内，更新前再除回去。BF16 的指数位与 FP32 相同，最小正规数约 $10^{-38}$，不会下溢，所以不需要。代价是 BF16 的精度更低，这对推理中的累加与归一化更关键（本章的几个例子）。

## 小结

- [x] 浮点数的精度是相对的：eps 决定有效数字，数越大间隔越大；BF16 范围大、精度低，FP16 精度较好但最大值只有 65504。
- [x] 大数加小数会丢失小数；浮点加法不满足结合律，计算顺序改变结果。
- [x] 低精度累加会严重出错（BF16 逐个累加 2 万个数得到 256），要用 FP32 累加器、两两求和或补偿求和。
- [x] 警惕灾难性抵消（单遍方差公式）和溢出（softmax 不减最大值、FP16 中的平方）；统计量用 FP32 计算。
- [x] 量化每增加 1 比特，信噪比提高约 6 dB；FP8 的误差是相对的，约 6%。
