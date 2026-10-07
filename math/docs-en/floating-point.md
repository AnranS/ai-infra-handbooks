# Floating point and numerical computation

<p class="lead">Inference optimization deals with low precision every day: BF16 weights and activations, FP8 GEMMs, FP4 MoE experts, FP32 accumulators. Many "inexplicable" phenomena are direct consequences of floating-point arithmetic: the same request gives two different results, adding a number changes nothing, a normalization suddenly outputs all zeros, a low-precision sum is completely wrong. This chapter explains how floating-point numbers are represented and where their errors come from, reproduces these phenomena with a set of small experiments, and ends with a practical formula for quantization noise.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What are the machine precision (eps) and the maximum value of BF16 and FP16? What is the risk of each?
    2. Why does 1650 + 1 in BF16 still equal the same number near 1650? What does this mean for the residual stream?
    3. Accumulating 20,000 numbers between 0 and 1 in BF16, how wrong can the result be? Why do Tensor Cores accumulate in FP32?
    4. Why should RMSNorm's statistics be computed in FP32?
    5. How much does each extra bit of quantization width improve the signal-to-noise ratio?

??? success "Answers (try first, then expand to compare)"
    1. BF16: eps about 0.0078 ($2^{-7}$), maximum about $3.4 \times 10^{38}$, and the risk is low precision. FP16: eps about 0.00098 ($2^{-10}$), maximum 65504, and the risks are overflow (squares, exponentials, large activations) and underflow of small gradients.
    2. BF16 has only 7 mantissa bits, so between 1024 and 2048 adjacent numbers are 8 apart; adding 1 is less than half the spacing and rounds back to the same value. If the residual stream holds a very large activation, the small updates later layers write into it are swallowed, which is why accumulation needs higher precision.
    3. Once the running sum grows, the spacing exceeds the numbers being added, and every later number is swallowed: in this chapter's experiment, accumulating 20,000 numbers between 0 and 1 one at a time in BF16 gives only 256. Tensor Cores use FP32 accumulators precisely so that the long accumulations in matrix multiplication do not go wrong like this.
    4. RMSNorm takes the sum of squares over a row: in low precision the squares can overflow (FP16), a long accumulation loses a lot of precision, and this statistic is applied to every number in the row. So the statistic is computed in FP32 and then converted back to low precision.
    5. About 6 dB (each extra bit halves the amplitude of the quantization noise and quarters its power).

## How floating-point numbers are represented {#浮点数的表示}

A floating-point number has three parts, a sign, an exponent and a mantissa: $x = (-1)^s \times 1.m \times 2^{e - \text{bias}}$. The exponent bits determine the **range** and the mantissa bits the **precision**: within $[2^k, 2^{k+1})$, the spacing between adjacent representable numbers (the ulp) is $2^{k} \times \text{eps}$, where $\text{eps} = 2^{-\text{mantissa bits}}$. **Precision is relative**: the larger the number, the larger the spacing.

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

- **FP16**: fairly good precision (eps ≈ $10^{-3}$), but the maximum is only 65504, so it overflows easily;
- **BF16**: the same exponent bits as FP32, so plenty of range, but a precision of only eps ≈ $8 \times 10^{-3}$, about two or three significant digits;
- **FP8 E4M3**: maximum 448 and a relative precision of 12.5%, so it must be used with scales (see [quantized deployment](serving://perf/quantization-deploy/)); E5M2 has a wider range and lower precision.

Draw the numbers the three 8-bit formats can represent on one number line and "precision is relative" becomes obvious: floating-point numbers are dense near 0 and sparse far away, with the same number of points in every binary interval $[2^k, 2^{k+1})$; an integer format with a scale has the same spacing everywhere.

![Figure: the numbers three 8-bit formats can represent between 0 and 16](assets/figures/float-numberline.svg){.aig-svg}

Type in any number to see which bits it splits into in each format, how far the stored value is from it, and how large the spacing (ulp) is at that magnitude:

<div class="aig-widget" data-widget="float-bits"></div>

## Rounding, and "adding changes nothing" {#舍入与加了等于没加}

The result of every floating-point operation is rounded to the nearest representable number, with a relative error of at most eps/2. When a large number is added to a small one, the small one can be rounded away entirely:

```pycon
>>> x = torch.tensor(1650.0, dtype=torch.bfloat16)
>>> x.item(), (x + 1).item(), (x + 4).item(), (x + 5).item()     # in BF16 the spacing between 1024 and 2048 is 8
(1648.0, 1648.0, 1648.0, 1656.0)
>>> a = torch.tensor(1e8)                                          # in FP32 the spacing near 1e8 is 8
>>> ((a + 1) - a).item(), ((a - a) + 1).item()                     # addition is not associative
(0.0, 1.0)
```

1650 cannot even be stored exactly (it becomes 1648), and adding 1 or 4 has no effect at all. The LLM book saw "massive activations" of this magnitude in [the residual stream](llm://transformer/norm-residual/#看看真实的残差流): in a BF16 residual stream, small updates at the same position are simply swallowed. The second example shows that floating-point addition is **not associative**: change the order of computation and the result changes, which is the root of "a different batch size gives a different result" (see [which optimizations change the output](llm://synthesis/token-journey/#哪些优化会改变输出)).

## Accumulation: errors add up {#累加误差会积累}

Summation is the basic operation of matrix multiplication, reductions and attention. Accumulate 20,000 random numbers in $[0, 1)$ one at a time in BF16, and compare several summation methods:

```python
torch.manual_seed(0)
values = torch.rand(20000)
exact = values.double().sum().item()
v16 = values.to(torch.bfloat16)

total = torch.tensor(0.0, dtype=torch.bfloat16)                    # 1. naive: a BF16 accumulator, adding one at a time
for v in v16:
    total = total + v
naive = total.item()

pairs = v16.clone()                                                # 2. pairwise summation (tree reduction), still BF16 throughout
while pairs.numel() > 1:
    if pairs.numel() % 2:
        pairs = torch.cat([pairs, torch.zeros(1, dtype=pairs.dtype)])
    pairs = pairs[0::2] + pairs[1::2]
pairwise = pairs.item()

total, comp = torch.tensor(0.0, dtype=torch.bfloat16), torch.tensor(0.0, dtype=torch.bfloat16)
for v in v16:                                                      # 3. Kahan compensated summation: remember what each step rounds away
    y = v - comp
    t = total + y
    comp = (t - total) - y
    total = t
kahan = total.item()

fp32_acc = v16.float().sum().item()                                # 4. BF16 inputs, but an FP32 accumulator
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

Naive BF16 accumulation gives **256**, while the correct answer is about ten thousand: once the running sum reaches 256, the BF16 spacing becomes 2, adding a number below 1 rounds back to the same value, and the sum "gets stuck". Pairwise summation keeps the two numbers being added at similar magnitudes and brings the error down from "absurdly wrong" to 0.5%; Kahan summation uses a compensation variable to remember what was rounded away; and the most effective fix is to **accumulate in higher precision**. That is exactly why Tensor Cores use FP32 accumulators after BF16/FP8 multiplications, and it is the numerical meaning of split-K in GEMM and blocked reductions in attention. DeepSeek-V3's FP8 GEMM even periodically "promotes" partial sums to CUDA cores for FP32 accumulation, precisely to control this kind of error.

## Cancellation and overflow {#抵消与溢出}

**Catastrophic cancellation**: subtracting two large, very close numbers cancels almost all significant digits, and what remains is all error. The classic example is computing the variance as $\mathbb{E}[x^2] - \mathbb{E}[x]^2$:

```python
data = 10000 + torch.randn(100_000)                                # data with mean 10,000 and variance about 1
one_pass = (data.pow(2).mean() - data.mean().pow(2)).item()        # subtracting two numbers of about 100 million
two_pass = ((data - data.mean()) ** 2).mean().item()               # subtract the mean first, then square
print(f"单遍公式：{one_pass:.4f}    两遍公式：{two_pass:.4f}    （真实方差约为 1）")
```

```text
单遍公式：0.0000    两遍公式：1.0006    （真实方差约为 1）
```

In FP32 the one-pass formula gives a variance of 0, completely wrong. A LayerNorm implementation must subtract the mean first (or use an online algorithm such as Welford's); RMSNorm does not subtract the mean and so sidesteps the problem, which is one reason it is more popular.

**Overflow**: the maximum of FP16 is only 65504, and $e^{11.1}$ already exceeds it. Without subtracting the maximum first, softmax turns a slightly large logit into inf, and the division then gives NaN; if RMSNorm computes the squares in FP16, the square of 1650 is about 2.7 million and overflows too:

```pycon
>>> z = torch.tensor([12.0, 3.0, -1.0], dtype=torch.float16)
>>> (z.exp() / z.exp().sum()).tolist()                              # naive softmax
[nan, 0.0, 0.0]
>>> [round(v, 4) for v in ((z - z.max()).exp() / (z - z.max()).exp().sum()).tolist()]   # subtract the max first
[1.0, 0.0001, 0.0]
>>> h = torch.tensor([1650.0, 3.0, -2.0, 0.5], dtype=torch.float16)
>>> (h * torch.rsqrt(h.pow(2).mean() + 1e-6)).tolist()            # RMSNorm in FP16: the square overflows, the output is all 0
[0.0, 0.0, -0.0, 0.0]
>>> [round(v, 4) for v in (h.float() * torch.rsqrt(h.float().pow(2).mean() + 1e-6)).tolist()]   # statistics in FP32
[2.0, 0.0036, -0.0024, 0.0006]
```

The FP16 RMSNorm raised no error; it just **silently** output all zeros, and problems like this are the hardest to track down. That is why this book's `mini_llm` and vLLM's RMSNorm kernel both compute the statistics in FP32 (see [normalization and the residual stream](llm://transformer/norm-residual/#layernorm-与-rmsnorm)).

## Quantization noise: 6 dB per bit {#量化噪声每比特-6-db}

Uniform quantization (step $\Delta$) rounds each number to the nearest grid point, so the error is roughly uniform on $[-\Delta/2, \Delta/2]$ with variance $\Delta^2/12$. Each extra bit halves $\Delta$ and quarters the noise power, **improving the signal-to-noise ratio by about 6.02 dB**. Verify it on standard normal data (with the quantization range fixed at ±4σ):

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

The measurement gives about 6 dB per bit, matching the theory. This is a very handy yardstick: INT8 has about 24 dB more signal-to-noise ratio than INT4, that is, about 16 times smaller noise amplitude. FP8 differs from integers in that its error is **relative**: within the normal range the maximum relative error is about $2^{-4} \approx 6\%$, regardless of the magnitude of the value. That is why floating-point formats tolerate outliers, while integer formats need carefully chosen scales.

!!! interview "In an interview"
    For numerical questions, start with a few numbers: BF16's eps is about 0.0078 and FP16's maximum is 65504; a small number added to a large one gets swallowed, and floating-point addition is not associative, so the batch composition changes the result; low-precision accumulation goes absurdly wrong (accumulating 20,000 numbers between 0 and 1 one at a time in BF16 gives only 256), so Tensor Cores accumulate in FP32 and normalization and softmax compute their statistics in FP32; each extra bit of quantization adds about 6 dB of signal-to-noise ratio, and FP8's relative error is about 6%. Connecting "why inference results are not bitwise identical" to these facts is what earns top marks on this kind of question.

## Exercises {#练习}

**1. Accumulator width.** An FP8 GEMM has a K dimension of 8192, and each product's relative error is about FP8's rounding error. With an FP16 accumulator, roughly how large is the error of the sum? Why is an FP32 accumulator enough?

??? success "Approach"
    In sequential accumulation, rounding error grows roughly with the number of additions: in the worst case the relative error is of order $K \times \text{eps}_\text{acc}$ (about $\sqrt{K}\,\text{eps}$ in the random case). FP16's eps ≈ $10^{-3}$ and $\sqrt{8192} \approx 90$, so the relative error can reach about 10%, similar to this chapter's BF16 accumulation example and unacceptable; FP32's eps ≈ $10^{-7}$, so even the worst case is only about $10^{-3}$, which is enough. Real kernels also accumulate in blocks (a variant of pairwise summation), which reduces the error further.

**2. Why does BF16 training not need loss scaling while FP16 does?**

??? success "Answer"
    Gradients are often very small (for example below $10^{-6}$). FP16's smallest normal number is about $6 \times 10^{-5}$; smaller values become subnormal or even underflow to 0, and the gradient information is lost. Loss scaling multiplies the loss by a large number so the gradients land in the range FP16 can represent, then divides it back out before the update. BF16 has the same exponent bits as FP32, with a smallest normal number of about $10^{-38}$, so it does not underflow and needs no scaling. The price is BF16's lower precision, which matters more for accumulation and normalization in inference (the examples in this chapter).

## Summary {#小结}

- [x] Floating-point precision is relative: eps sets the significant digits, and the larger the number, the larger the spacing; BF16 has a wide range and low precision, while FP16 has better precision but a maximum of only 65504.
- [x] Adding a small number to a large one loses the small number; floating-point addition is not associative, so the order of computation changes the result.
- [x] Low-precision accumulation goes badly wrong (20,000 numbers accumulated one at a time in BF16 give 256); use FP32 accumulators, pairwise summation or compensated summation.
- [x] Watch out for catastrophic cancellation (the one-pass variance formula) and overflow (softmax without subtracting the max, squares in FP16); compute statistics in FP32.
- [x] Each extra bit of quantization improves the signal-to-noise ratio by about 6 dB; FP8's error is relative, about 6%.
