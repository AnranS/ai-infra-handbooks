# Low-bit inference for very large MoE: INT4, FP4 and quantization-aware training

<p class="lead">The <a href="../../perf/quantization-deploy/">quantization in deployment</a> chapter compared the accuracy of various low-precision formats on small models. At a trillion parameters, low bit widths are no longer just "saving a bit of memory": a trillion-parameter model in FP8 needs two 8-GPU machines, in 4 bits only one, and the minimum deployment unit, the communication pattern and decode's latency floor all change with it. On the other hand, reasoning models routinely output tens of thousands of tokens, and a little quantization error at each step gets amplified along the autoregressive chain. This chapter first works out what 4 bits mean for a very large MoE, then uses a small experiment to show why publishers increasingly produce INT4 / FP4 weights directly with <b>quantization-aware training</b> (QAT), and finally looks at the corresponding kernels on different hardware.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How many H200s does a trillion-parameter MoE need in BF16, FP8 and 4 bits?
    2. Why can a single-step accuracy difference of only 1% make long outputs much worse?
    3. How do post-training quantization (PTQ) and quantization-aware training (QAT) differ? How does QAT's gradient get through rounding?
    4. How do the kernels differ for INT4 weights on Hopper versus FP4 weights on Blackwell?
    5. Which scenarios do W4A16, W4A8 and W4A4 each suit?

??? success "Answers (try first, then expand to compare)"
    1. With this chapter's example of "a trillion total parameters, 32 billion activated" (leaving 30% for KV and activations): BF16 is about 2 TB, needing 21 GPUs (three 8-GPU machines); FP8 about 1 TB, needing 11 (two machines); 4 bits about 0.53–0.56 TB, where 6 suffice (one machine).
    2. Reasoning models' outputs are long, and when one token is wrong the reasoning chain after it can go astray; with 99% single-step accuracy, the probability of getting all 256 steps right is only about 6% ($0.99^{256}$): small errors are amplified by long chains.
    3. PTQ quantizes a trained model directly without further training; QAT uses a fake-quantized forward pass during training (post-training), letting the model adapt to quantization error. Rounding's gradient is zero almost everywhere, so the straight-through estimator (STE) is used: in the backward pass, rounding is treated as the identity, and the gradient passes straight through.
    4. Hopper has no FP4 Tensor Cores: INT4 weights are dequantized to BF16 / FP8 in registers, then computed on Tensor Cores of that precision (W4A16 / W4A8); Blackwell has native block-scaled Tensor Cores for MXFP4 / NVFP4, where weights (and activations) enter the matrix multiply directly as FP4.
    5. W4A16: only cuts the bytes of weights read, suited to decode, small batches and latency-sensitive scenarios; W4A8: compute in 8 bits too, covering both decode and prefill on Hopper; W4A4 (FP4): compute in 4 bits too on Blackwell, so prefill and large batches benefit as well, but with stricter accuracy demands (usually paired with QAT).

## What 4 bits mean for a trillion-parameter MoE {#4-比特对万亿-moe-意味着什么}

Take an MoE with "a trillion total parameters, 32 billion activated per token" (illustrative sizes), and compute each format's total weight size, the minimum number of H200s (141 GB each, leaving 30% for KV and activations), and the lower bound on time to read the weights per token for 8-GPU decode:

```python
import math

TOTAL, ACTIVE = 1.0e12, 32e9                          # illustrative: an MoE with a trillion total parameters and 32 billion activated per token
FORMATS = {                                           # average bytes per parameter (scales included)
    "BF16": 2.0,
    "FP8（128×128 分块）": 1 + 4 / 128**2,
    "INT4（每 32 个一个 bf16 缩放）": 0.5 + 2 / 32,
    "MXFP4（每 32 个一个 E8M0）": 0.5 + 1 / 32,
    "NVFP4（每 16 个一个 E4M3）": 0.5 + 1 / 16,
}
GPU_MEM, GPU_BW = 141e9, 4.8e12                       # memory and bandwidth of one H200
for name, b in FORMATS.items():
    w = TOTAL * b
    gpus = math.ceil(w / (GPU_MEM * 0.7))
    print(f"{name}：权重 {w / 1e12:.2f} TB，至少 {gpus} 张 H200（留 30% 给 KV，{math.ceil(gpus / 8)} 台 8 卡机），"
          f"8 卡 decode 每 token 读权重至少 {ACTIVE * b / (8 * GPU_BW) * 1e3:.2f} ms")
```

```text title="output"
BF16：权重 2.00 TB，至少 21 张 H200（留 30% 给 KV，3 台 8 卡机），8 卡 decode 每 token 读权重至少 1.67 ms
FP8（128×128 分块）：权重 1.00 TB，至少 11 张 H200（留 30% 给 KV，2 台 8 卡机），8 卡 decode 每 token 读权重至少 0.83 ms
INT4（每 32 个一个 bf16 缩放）：权重 0.56 TB，至少 6 张 H200（留 30% 给 KV，1 台 8 卡机），8 卡 decode 每 token 读权重至少 0.47 ms
MXFP4（每 32 个一个 E8M0）：权重 0.53 TB，至少 6 张 H200（留 30% 给 KV，1 台 8 卡机），8 卡 decode 每 token 读权重至少 0.44 ms
NVFP4（每 16 个一个 E4M3）：权重 0.56 TB，至少 6 张 H200（留 30% 给 KV，1 台 8 卡机），8 卡 decode 每 token 读权重至少 0.47 ms
```

Going from FP8 to 4 bits, the deployment unit shrinks from "two machines" to "one machine", with effects far beyond memory:

- **No cross-machine expert parallelism needed**: the whole model fits in one NVLink domain, EP's all-to-all goes entirely over NVLink ([previous part](../comm/interconnect.md): 9× faster than a NIC), the cross-machine optimizations of DeepEP are no longer necessary, and small deployments work too;
- **Decode's latency floor halves**: with small batches decode is limited by weight reads, so reading half the bytes doubles the single-request speed limit, which matters especially for reasoning models with very long outputs;
- **The overhead of scales cannot be ignored**: one bf16 scale per 32 4-bit numbers adds 12.5%; NVFP4's one FP8 scale per 16 likewise adds 12.5%; MXFP4's scale is only 1 byte per 32, about 6% extra.

## Why quantization-aware training {#为什么要量化感知训练}

First recall where quantization error comes from (the widget from the LLM handbook):

<div class="aig-widget" data-widget="quant"></div>

**Post-training quantization** (PTQ): after training, round the weights to low precision directly (methods such as GPTQ and AWQ can reduce the error). **Quantization-aware training** (QAT): use the quantized weights in the forward pass of training (or fine-tuning), letting the model "adapt" to rounding error; rounding has no gradient, so backpropagation uses the **straight-through estimator** (STE), treating rounding as the identity and passing the gradient as is to the high-precision weights behind it.

For reasoning models, the problem with PTQ is that errors are amplified along the output chain: once one step goes wrong, every later step builds on a wrong context. Simulate this with a small experiment: a two-layer network learns a table where "the previous two tokens determine the next" (a "language model" that has fully memorized its rule), then apply INT4 PTQ and QAT separately, and look at single-step accuracy and the correctness of 256-step greedy generation:

```python ci="loose"
import torch

torch.manual_seed(0)
torch.set_num_threads(4)
V, H, G = 64, 224, 32                                # vocabulary, hidden size, INT4 quantization group size
table = torch.randint(0, V, (V, V))                  # the "teacher": the next token is looked up from the previous two
ctx = torch.cartesian_prod(torch.arange(V), torch.arange(V))
target = table[ctx[:, 0], ctx[:, 1]]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, 32)
        self.fc1 = torch.nn.Linear(64, H)
        self.fc2 = torch.nn.Linear(H, V)

    def forward(self, x, quant=False):
        h = self.emb(x).flatten(1)
        w1, w2 = (fake_int4(self.fc1.weight), fake_int4(self.fc2.weight)) if quant else (self.fc1.weight, self.fc2.weight)
        h = torch.relu(torch.nn.functional.linear(h, w1, self.fc1.bias))
        return torch.nn.functional.linear(h, w2, self.fc2.bias)


def fake_int4(w):
    """对称 INT4、每 G 个输入通道一组；前向用量化值，反向把梯度原样传给 fp32 权重（直通估计 STE）"""
    g = w.view(w.shape[0], -1, G)
    s = g.abs().amax(-1, keepdim=True) / 7
    q = (g / s).round().clamp(-8, 7) * s
    return w + (q.view_as(w) - w).detach()


def train(model, steps, quant, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(model(ctx, quant), target)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(model, quant):
    acc = (model(ctx, quant).argmax(-1) == target).float().mean().item()
    starts = torch.randint(0, V, (200, 2), generator=torch.Generator().manual_seed(1))
    seq, ok = starts.clone(), torch.ones(200, dtype=torch.bool)
    lengths = torch.zeros(200)
    for _ in range(256):                             # greedy autoregression for 256 steps: one wrong step derails the whole chain after it
        nxt = model(seq[:, -2:], quant).argmax(-1)
        ok &= nxt == table[seq[:, -2], seq[:, -1]]
        lengths += ok.float()
        seq = torch.cat([seq, nxt[:, None]], 1)
    return acc, lengths.mean().item(), ok.float().mean().item()


base = Tiny()
train(base, 3000, False, 3e-3)
ptq = base                                           # quantize directly after training (PTQ)
qat = Tiny()
qat.load_state_dict(base.state_dict())
train(qat, 2000, True, 1e-3)                         # quantization-aware training (QAT): keep training on a fake-quantized forward pass
for name, model, quant in (("原模型（fp32）", base, False), ("INT4 训练后量化", ptq, True), ("INT4 量化感知训练", qat, True)):
    acc, run, full = evaluate(model, quant)
    print(f"{name}：单步准确率 {acc:.1%}，连续正确的平均步数 {run:.0f} / 256，整条链全对 {full:.0%}")
```

```text title="output"
原模型（fp32）：单步准确率 100.0%，连续正确的平均步数 256 / 256，整条链全对 100%
INT4 训练后量化：单步准确率 98.9%，连续正确的平均步数 96 / 256，整条链全对 29%
INT4 量化感知训练：单步准确率 100.0%，连续正确的平均步数 256 / 256，整条链全对 100%
```

PTQ's single-step accuracy drops by only 1.1%, yet 71% of the 256-step chains go wrong partway, getting only 96 steps on average: with a 1% error rate per step, the probability of all 256 steps being right is $0.989^{256} \approx 6\%$, and wrong inputs cause yet more errors afterwards (a wrong context is exactly what the model has never seen). QAT keeps training for 2000 steps on the fake-quantized forward pass, single-step accuracy returns to 100%, and every chain is fully correct.

A real model is of course not a lookup table, but the reasoning is the same: reasoning models often output tens of thousands of tokens while keeping their logic consistent along long chains, so PTQ that is "nearly lossless" on short benchmarks can lose noticeably on long reasoning tasks. That is why more and more models publish the low-bit weights from QAT directly: gpt-oss publishes its MoE weights in MXFP4, and trillion-parameter reasoning models have done INT4 QAT in post-training and published INT4 weights directly. The cost of QAT is the work on the training side: fake quantization must be inserted into the forward pass, and used throughout post-training stages such as RL as well (otherwise post-training "trains" the weights back to a high-precision distribution).

## Kernels on different hardware {#不同硬件上的-kernel}

| Scheme | Weights | Activations and compute | Hardware | Suits |
| --- | --- | --- | --- | --- |
| W4A16 | INT4 / FP4 | dequantize weights to BF16 in registers, compute on BF16 Tensor Cores | every GPU generation (Hopper has no FP4 Tensor Cores) | decode: limited by weight reads, so fewer bytes means faster |
| W4A8 | INT4 | dequantize weights to FP8, quantize activations to FP8, compute on FP8 Tensor Cores | Hopper | covers both decode's reads and prefill's compute |
| W4A4 (FP4) | MXFP4 / NVFP4 | quantize activations to FP4 too, compute directly on FP4 Tensor Cores | Blackwell | prefill and large batches: doubles compute again |

A few details:

- **Where dequantization happens**: W4A16 kernels (Marlin, for example) dequantize on the way from shared memory to registers, and pre-shuffle the weights in GPU memory into a layout convenient for vectorized unpacking; this is exactly what the [quantization and GEMV](cuda://advanced/quantization/) chapter of the CUDA handbook does;
- **MoE's grouped GEMM must support low bits too**: one low-bit weight matrix per expert, grouped by routing; vLLM provides expert implementations for different formats under `model_executor/layers/fused_moe/` (`experts/marlin_moe.py` and others), with `int_wna16.py`, `mxfp4.py` and `fp8.py` under `oracle/` choosing kernels by format and hardware;
- **Decode and prefill trade off differently**: in decode, 4-bit weights halve the reads, and W4A16 is enough; prefill is compute-bound, and W4A16 still computes in BF16, so it is no faster than FP8; only Blackwell's FP4 Tensor Cores let prefill benefit too;
- **The KV Cache counts separately**: once weights are 4 bits, KV often becomes the bulk of memory with long contexts, usually paired with FP8 KV.

!!! interview "In an interview"
    When asked "how do you deploy a trillion-parameter model" or "why INT4 QAT", start with the accounting: FP8 needs two 8-GPU machines, 4 bits one, cross-machine EP becomes in-machine EP, and decode's latency floor halves; then accuracy: reasoning models' long outputs amplify PTQ's small errors along the chain (with 99% single-step accuracy, only about 6% of 256-step chains are all correct), so QAT is done in post-training, with STE letting gradients through rounding; finally hardware: W4A16 / W4A8 on Hopper (dequantizing in registers), native MXFP4 / NVFP4 Tensor Cores on Blackwell, with different gains for decode and prefill.

## Exercises {#练习}

**1. Why does STE work?** Rounding's derivative is zero almost everywhere, yet STE passes the gradient straight through. Why are the weights trained this way more quantization-"friendly"?

??? success "Answer"
    STE lets the gradient of the loss with respect to "the quantized weights" act on the high-precision weights behind them: the forward pass sees the real loss caused by quantization error, so the backward pass pushes the high-precision weights in the direction of "lower loss after quantization". The result is high-precision weights that sit at "safe" spots on the quantization grid (away from rounding boundaries, or with rounding errors canceling each other), with decision boundaries adjusted accordingly, making the model insensitive to the remaining quantization error. STE's gradient is a biased approximation, but it works well enough when the quantization noise is not too large, which is also why QAT is usually a short fine-tune of an already trained model rather than training from scratch.

**2. The precision of scales.** MXFP4's scales are E8M0 (powers of 2 only), while NVFP4's are E4M3 (with 3 mantissa bits) and come one per 16 numbers. What are the pros and cons of each?

??? success "Answer"
    E8M0 scales can only be powers of 2, so multiplication becomes exponent addition and the hardware is simple, but the scaling is imprecise: when a group's maximum is scaled into FP4's range, the worst case wastes nearly 2× of the dynamic range (equivalent to losing one level). NVFP4's E4M3 scales can align each group's maximum precisely with FP4's maximum of 6, and groups of 16 are finer too, so accuracy is better (measured in the [quantization in deployment](../perf/quantization-deploy.md) chapter), at the cost of twice the space for scales (1 byte per 16 numbers), plus an extra tensor-level FP32 scale to extend E4M3's range.

## Summary {#小结}

- [x] A trillion-parameter MoE going from FP8 to 4 bits: the deployment unit shrinks from two machines to one, cross-machine EP becomes in-machine EP, and decode's latency floor halves; scales take an extra 6%–12.5%.
- [x] Long outputs amplify quantization error: with a 1% error per step, most 256-step chains go wrong partway; QAT fine-tunes on a fake-quantized forward pass with STE and can train the error back out, and more and more models publish QAT low-bit weights directly.
- [x] W4A16 / W4A8 on Hopper (dequantizing in registers), native MXFP4 / NVFP4 Tensor Cores on Blackwell; decode benefits from reading fewer bytes, while prefill benefits only from low-precision compute.
