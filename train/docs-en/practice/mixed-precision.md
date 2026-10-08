# Mixed precision and FP8 training

<p class="lead">Almost every large model is trained in low precision: bf16 doubles the matrix multiplies' throughput and halves the memory, and FP8 doubles it again. But low-precision numbers have two problems: a limited range (a gradient that is too small becomes 0) and limited precision (a small update is rounded away). Mixed-precision training is a set of rules about where high precision is still required. This chapter runs the reason for each rule as a small experiment, and ends on the most important question in FP8 training, the granularity of the scaling factor.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How many exponent and mantissa bits do fp16 and bf16 have? Why does fp16 training need loss scaling while bf16 usually does not?
    2. Why does the optimizer keep a copy of the weights in fp32?
    3. Which operations in mixed-precision training still have to be in fp32?
    4. Which two FP8 formats are there? Where is each used?
    5. Why does FP8 training use fine-grained (block) scaling factors?

??? success "Answers for the self-test (answer first, then open this)"
    1. fp16: 5 exponent bits and 10 mantissa bits; bf16: 8 and 7. The smallest positive number fp16 can represent is about $6 \times 10^{-8}$, and a smaller gradient simply becomes 0, so the loss has to be scaled up before the backward pass; bf16's range matches fp32's, so gradients do not underflow and loss scaling is usually unnecessary.
    2. bf16's machine epsilon is about 0.0078, so at a weight of 1.0 an update of $10^{-3}$ is smaller than the smallest representable change and is rounded away: however many updates you apply, nothing moves. The update is done on fp32 master weights, which are converted to bf16 each step for the forward pass.
    3. The matrix multiplies' accumulation (Tensor Cores use an fp32 accumulator), softmax, the statistics of normalisation (LayerNorm, RMSNorm), computing the loss, and the optimizer update (the master weights and the two moments); large-scale training also often reduces the gradients in fp32.
    4. E4M3 (4 exponent bits, 3 mantissa bits, more precision and less range, for the forward pass's weights and activations) and E5M2 (5 and 2, more range, for the backward pass's gradients).
    5. FP8's range is very narrow, so one outlier pushes the whole tensor's scaling factor up and squeezes the rest into subnormals or to 0. Scaling in blocks of 1x128 (activations) and 128x128 (weights) confines an outlier's effect to its own block.

## Range and precision {#范围与精度}

Type any number and see which bits it is split into in FP32, FP16, BF16 and FP8, and how much is lost in storing it (the same tool as in the large-model handbook):

<div class="aig-widget" data-widget="float-bits"></div>

```python title="precision.py"
import torch

print("各格式能表示的最小正规数与机器精度：")
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"  {str(dt):<15} 最大 {fi.max:.3g}，最小正规数 {fi.tiny:.3g}，eps {fi.eps:.3g}")

g = torch.tensor([1e-8, 1e-6, 2e-5])                     # very small gradients (common in a deep network)
print("fp16 直接存：", g.half().tolist())
scale = 2.0 ** 16
print("先乘 2^16 再存 fp16、用时再除回：", (g * scale).half().float().div(scale).tolist())
print("bf16 直接存：", g.bfloat16().float().tolist())

w16, w32 = torch.tensor(1.0, dtype=torch.bfloat16), torch.tensor(1.0)
for _ in range(1000):                                    # a small update of learning rate x gradient = 1e-3, applied 1000 times
    w16 += 1e-3
    w32 += 1e-3
print(f"bf16 权重直接更新 1000 次：{w16.item():.4f}；fp32 主权重：{w32.item():.4f}")
```

```text title="output"
各格式能表示的最小正规数与机器精度：
  torch.float32   最大 3.4e+38，最小正规数 1.18e-38，eps 1.19e-07
  torch.float16   最大 6.55e+04，最小正规数 6.1e-05，eps 0.000977
  torch.bfloat16  最大 3.39e+38，最小正规数 1.18e-38，eps 0.00781
fp16 直接存： [0.0, 1.0132789611816406e-06, 2.002716064453125e-05]
先乘 2^16 再存 fp16、用时再除回： [9.997165761888027e-09, 1.000240445137024e-06, 1.9997358322143555e-05]
bf16 直接存： [1.0011717677116394e-08, 9.98377799987793e-07, 2.002716064453125e-05]
bf16 权重直接更新 1000 次：1.0000；fp32 主权重：2.0000
```

Point by point:

- **fp16** has only 5 exponent bits, so the smallest positive number it can represent is about $6 \times 10^{-8}$ (a subnormal), and a smaller gradient simply becomes 0 (the first value), with poor precision near the limit too. **Loss scaling**: multiply the loss by a large number ($2^{16}$, say) before the backward pass so that the gradients are scaled into fp16's range, and divide back before updating. Too large a factor overflows to `inf`, so in practice it is adjusted dynamically: on an `inf`, skip the step and reduce the factor (`torch.amp.GradScaler`).
- **bf16** has 8 exponent bits and the same range as fp32, so gradients do not underflow and loss scaling is usually **unnecessary**, which is why it is the default format for large-model training; the price is only 7 mantissa bits, which is less precision than fp16.
- **Master weights**: bf16's machine epsilon is about $0.0078$, so at a weight of 1.0 an update of $10^{-3}$ is smaller than the smallest representable change and is rounded away. After 1000 updates the weight has not moved. So the optimizer updates **fp32 master weights** and converts them to bf16 each step for the forward pass; Adam's two moments are in fp32 as well. This is where the memory budget's 16 bytes per parameter comes from.

## Which operation uses which precision {#哪些运算用什么精度}

| Operation | Precision | Why |
| --- | --- | --- |
| Matrix multiplies (forward and backward) | bf16 / FP8 inputs, **fp32 accumulation** | the main source of throughput; the accumulator is high-precision, as Tensor Cores do by default |
| softmax, LayerNorm / RMSNorm, the loss | fp32 | sums, exponentials and divisions are sensitive to precision |
| The optimizer update | fp32 master weights and moments | see the experiment above |
| The gradients' all-reduce | bf16 or fp32 | bf16 halves the communication, but the error grows when accumulating over many cards, so large-scale training often reduces in fp32 |

In PyTorch, `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` picks the precision by this table automatically (bf16 for matrix multiplies, fp32 for normalisation and softmax); a kernel you write yourself has to follow the rules itself.

## FP8 training {#fp8-训练}

FP8 has two formats: **E4M3** (4 exponent bits, 3 mantissa bits, a maximum of 448) with more precision and less range, generally for the forward pass's weights and activations; and **E5M2** (5 and 2) with more range and less precision, traditionally for the backward pass's gradients (DeepSeek-V3 uses E4M3 throughout).

With a range of only a few hundred, the values have to be multiplied by a **scaling factor** to map them in. The granularity of that scaling is the crux:

- **Per tensor**: one factor for the whole tensor. One outlier is enough to decide the factor, squeezing the ordinary values into a very small range, into the subnormals or to underflow, with a large loss of precision.
- **Fine-grained** (DeepSeek-V3): one factor per 1x128 elements of the activations and one per 128x128 block of the weights. An outlier affects only its own small block.

![Figure: the data flow of FP8 training, where the matrix multiplies' inputs are quantised to FP8 while the accumulation and the master weights stay in high precision](../assets/figures/fp8-training.svg){.aig-svg}

```python title="fp8_scaling.py"
import torch

torch.manual_seed(0)
FP8 = torch.float8_e4m3fn
MAX = torch.finfo(FP8).max                                # 448


def quant(x, scale):
    return (x / scale).clamp(-MAX, MAX).to(FP8).float() * scale   # scale, quantise to FP8, dequantise


def per_tensor(x):
    return quant(x, x.abs().max() / MAX)


def per_block(x, rows, cols):
    """按 rows × cols 的块各自缩放（DeepSeek-V3：激活 1×128，权重 128×128）"""
    R, C = x.shape
    b = x.reshape(R // rows, rows, C // cols, cols)
    s = b.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / MAX
    return quant(b, s).reshape(R, C)


def rel_err(q, x):
    return ((q - x).norm() / x.norm()).item()


print(f"FP8 E4M3：最大值 {MAX:.0f}，最小正规数 {torch.finfo(FP8).tiny}")
for outlier in (1e2, 1e4, 3e4):
    a = torch.randn(256, 1024)
    a[:, :4] *= outlier                                  # a few channels have outliers (common in an LLM's activations)
    normal = a[:, 128:]                                  # the ordinary channels that are not in the same block of 128 as the outlier channels
    t = rel_err(per_tensor(a)[:, 128:], normal)
    b = rel_err(per_block(a, 1, 128)[:, 128:], normal)
    print(f"离群值放大 {outlier:>7.0f} 倍：普通通道的相对误差，逐张量缩放 {t:5.1%}，每 1×128 块缩放 {b:5.1%}")
```

```text title="output"
FP8 E4M3：最大值 448，最小正规数 0.015625
离群值放大     100 倍：普通通道的相对误差，逐张量缩放  2.6%，每 1×128 块缩放  2.6%
离群值放大   10000 倍：普通通道的相对误差，逐张量缩放  4.9%，每 1×128 块缩放  2.6%
离群值放大   30000 倍：普通通道的相对误差，逐张量缩放 14.1%，每 1×128 块缩放  2.6%
```

The larger the outlier, the larger the error on the ordinary channels under per-tensor scaling; the error under 1x128 block scaling stays at FP8's own precision (3 mantissa bits, about 2 to 3%). An LLM's activations contain exactly this sort of handful of channels with very large magnitudes, which is why fine-grained scaling is standard in FP8 training.

Two more measures go with it in the hardware:

- **High-precision accumulation**: the H800's FP8 Tensor Cores have a limited accumulator precision, so DeepSeek-V3 promotes the partial sum to the CUDA cores and accumulates in fp32 every 128 elements (DeepGEMM's implementation).
- **Blackwell's MX formats** (MXFP8, MXFP4) support one scaling factor per 32 elements natively in hardware, which makes fine-grained scaling the standard.

## The other ways to save activations {#省激活的其他手段}

Mixed precision saves the model states and the throughput; the activations still need:

- **Full recomputation**: store only each layer's input and recompute the whole layer in the backward pass, which takes the activations down to $2\,sbh$ per layer at about 33% more compute.
- **Selective recomputation** (Megatron): recompute only the parts of attention that take a lot of memory and little compute (the intermediates of softmax and dropout), saving most of the activations for very little extra compute; with FlashAttention those are not stored in the first place.
- **Activation offload**: copy the activations to CPU memory and back for the backward pass, trading PCIe bandwidth for device memory, which is only worth it when overlapped with the computation.

!!! interview "How to explain it"
    On mixed precision: fp16 has only 5 exponent bits so small gradients underflow and it needs dynamic loss scaling (skip the step and halve the factor on an overflow); bf16 has as many exponent bits as fp32 and usually does not, which makes it the default for large-model training, at the price of only 7 mantissa bits, so the optimizer keeps fp32 master weights (in bf16, 1.0 + 0.001 is rounded away). Matrix multiplies take low-precision inputs and accumulate in fp32, while softmax, normalisation, the loss and the optimizer update are fp32. FP8 splits into E4M3 (forward) and E5M2 (gradients), per-tensor scaling is wrecked by outliers, and fine-grained (1x128, 128x128) scaling with high-precision accumulation is standard.

## Exercises {#练习}

1. Use `precision.py`'s approach to check whether fp16 master weights have the same problem. Change `w16` to fp16, try it, and explain the result.

??? success "Answer"
    fp16's spacing in $[1, 2)$ is $2^{-10} \approx 0.000977$, so $1.0 + 10^{-3}$ rounds to the nearest $1.000977$ and each step really adds only $0.000977$: after 1000 steps it is $1.9766$ rather than $2.0$ (a systematic rounding error on every step).
    In $[4, 8)$ the spacing becomes $0.0039$ and $10^{-3}$ is less than half of it, so the update is rounded away entirely: starting from 1.0 and adding 5000 times, the weight stops at $4.0$. The conclusion is the same: the master weights have to be fp32.

2. A tensor's values are mostly in $[-1, 1]$ but one of them is 5000. Under per-tensor E4M3 quantisation, what is the scaling factor? How much precision is left for a value like $0.01$?

??? success "Answer"
    The factor is $5000 / 448 \approx 11.2$, so $0.01$ divided by it is about $0.0009$, less than half of E4M3's smallest subnormal ($2^{-9} \approx 0.00195$), and it quantises straight to 0. Most of the values in $[-1, 1]$ are left with one or two significant figures or become 0, which is the problem fine-grained scaling exists to solve.

## Summary {#小结}

- [x] fp16 has a small range and needs loss scaling; bf16's range matches fp32's and usually does not, which makes it the default for large-model training.
- [x] Matrix multiplies take low-precision inputs and accumulate in fp32; softmax, normalisation, the loss and the optimizer update are fp32, and the optimizer keeps fp32 master weights.
- [x] FP8 splits into E4M3 and E5M2; per-tensor scaling is wrecked by outliers, so fine-grained (1x128, 128x128) scaling is standard, paired with high-precision accumulation.
- [x] The activations are saved by recomputation (full or selective) and by offloading.
