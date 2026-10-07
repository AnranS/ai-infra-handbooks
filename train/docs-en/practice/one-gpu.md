# What one consumer card can train: a small text model and a small image generator

<p class="lead">The three from-scratch chapters got the whole pipeline running on a CPU; this chapter moves to one real card, a 16 GB RTX 5070 Ti. It answers three questions: how large a model fits, how long training takes, and how to configure it so no throughput is wasted. Then it trains two things with it: a small text model that continues Chinese prose, and a small image generator. The image side uses <strong>flow matching</strong>, whose training objective is one line and which has half the hyperparameters of DDPM; this chapter's minimal implementation shows it learning a distribution within seconds on a CPU.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Within 16 GB, how much does each part take when training a 350M model? Which part most easily gets out of hand?
    2. How much memory does full-layer activation recomputation save, and how much extra time does it cost? When should it be turned on?
    3. What is flow matching's training objective? How does it relate to DDPM?
    4. On one consumer card, which of `torch.compile`, bf16, FlashAttention and gradient accumulation gains the most?
    5. Training blew up (the loss became NaN or suddenly shot up). In what order do you investigate?

??? success "Answers for the self-test (answer first, then open this)"
    1. Under mixed-precision AdamW the model states are $16\Psi$ bytes (2Ψ bf16 parameters, 2Ψ bf16 gradients, 4Ψ fp32 parameter copy, 4Ψ each for Adam's m and v), so 350M is 5.2 GB; the activations grow linearly with layers x sequence length x batch and are the part that most easily gets out of hand, at 6.4 GB for a sequence of 1024 and a micro-batch of 8 without recomputation.
    2. Full-layer recomputation keeps only each layer's input, taking the activations from about $34d$ bytes per token to about $2d$, roughly 17 times less; the price is one extra forward pass in the backward, about 30% more time. Turn it on when memory runs out, or to grow the batch enough to fill the throughput.
    3. The objective is to have the network predict the straight-line velocity from the noise end to the data end: regress $x_1 - x_0$ on $x_t = (1-t)x_0 + t x_1$, one line of MSE. DDPM predicts the noise that was added and needs a noise schedule and many hyperparameters; flow matching is equivalent to a particular probability path, is simpler in form, and samples by integrating an ordinary differential equation, so few steps suffice.
    4. Usually **bf16 plus FlashAttention** (saving memory and gaining speed), then `torch.compile` (eliminating kernel launches and memory round trips, especially visible on a small model); gradient accumulation only solves "not enough memory but I want a large batch" and gains no speed itself.
    5. First check whether it is numerical (does it still blow up in fp32 for a few steps), then the learning rate and warm-up (does it still blow up at a tenth the rate), then the data (any unusually long samples, or a stretch of one repeated token), and finally the gradient clipping and loss scaling. Watch the `grad_norm` curve throughout: a spike there usually appears dozens of steps before the loss.

## Measure this card first {#先量这张卡}

Before any estimate, get two numbers: **the measured memory bandwidth** and **the measured bf16 throughput**. The specification (the 5070 Ti: 16 GB of GDDR7, 256 bit, 28 Gbps, a theoretical bandwidth of about 896 GB/s) is only a ceiling, and what you can actually use has to be measured.

```python title="bench_gpu.py" run="no"
# measure two numbers on your own card: the memory bandwidth and the bf16 matrix-multiply throughput. Every estimate below uses them as the denominator.
import time

import torch

assert torch.cuda.is_available()
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), f"| 显存 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB",
      f"| 计算能力 sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


n = 1 << 26                                     # a 256 MB bf16 tensor, memory access only
a = torch.randn(n, dtype=torch.bfloat16, device=dev)
b = torch.empty_like(a)
dt = timed(lambda: b.copy_(a))
print(f"显存带宽（读+写）{2 * a.numel() * 2 / dt / 1e9:.0f} GB/s")

for m in (2048, 4096, 8192):                    # a square matrix multiply, compute only
    x = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    y = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    dt = timed(lambda: torch.mm(x, y))
    print(f"bf16 矩阵乘 {m}×{m}：{2 * m ** 3 / dt / 1e12:.0f} TFLOPS")
```

Consumer Blackwell's dense bf16 tensor-core throughput is **on the order of 150 to 180 TFLOPS** (take your own measurement as authoritative; the time estimates later in this chapter use the effective throughput, the measured peak times the utilization, typically 40% to 50%). The measured bandwidth usually reaches 80% to 90% of the specification.

!!! warning "The environment traps on Blackwell (sm_120)"
    The RTX 50 series is `sm_120`, a generation newer than Ada (`sm_89`). When setting up:

    - **PyTorch has to be 2.7 or newer and a cu128 or later wheel**. An older wheel has no `sm_120` cubin, so it either reports `no kernel image is available for execution on the device` or quietly falls back to PTX JIT (compiling every kernel on its first run, absurdly slow).
    - Confirm `sm_120` appears in `torch.cuda.get_arch_list()`.
    - **Prebuilt FlashAttention wheels often have not caught up**. Use PyTorch's own `F.scaled_dot_product_attention` first (it picks the FlashAttention or memory-efficient backend automatically); do not install a third-party wheel straight away.
    - `torch.compile`'s first compilation takes a few minutes, which is normal; point `TORCHINDUCTOR_CACHE_DIR` at a fixed directory and later runs reuse it.

## The memory budget: what fits in 16 GB {#显存账本16-gb-放得下什么}

```python title="budget.py"
# one 16 GB card: how large a model fits, and how long training takes
GB = 1024 ** 3
VRAM = 16 * GB
RESERVED = 1.2 * GB                      # the driver, the CUDA context, the cuBLAS workspace and fragmentation, deducted first


def states(params):
    """混合精度 AdamW 的常驻显存 = 16Ψ 字节：
    bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 参数副本 4Ψ + Adam 的 m、v 各 4Ψ"""
    return 16 * params


def activations(layers, d, seq, batch, recompute):
    """激活。不重计算时每层每 token 大约 34d 字节（bf16，含注意力里的中间量）；
    整层重计算只留每层的输入，约 2d 字节，代价是多做一次前向（约 +30% 时间）。"""
    return layers * seq * batch * (2 if recompute else 34) * d


print("模型能不能放进 16 GB（seq 1024，micro-batch 8，整层重计算）")
for name, layers, d, params in [
    ("30M", 6, 384, 30e6), ("124M / GPT-2", 12, 768, 124e6), ("350M", 24, 1024, 350e6),
    ("770M", 24, 1536, 770e6), ("1.4B", 24, 2048, 1.4e9),
]:
    st, act = states(params), activations(layers, d, 1024, 8, True)
    total = st + act + RESERVED
    print(f"  {name:<14} {layers:>2} 层 × {d:<5} 权重+优化器 {st / GB:>5.1f} GB  "
          f"激活 {act / GB:>4.1f} GB  合计 {total / GB:>5.1f} GB  "
          f"{'放得下' if total < VRAM else '放不下，要更小的 batch 或 LoRA'}")

print()
print("不重计算的话激活是多少（同样 seq 1024、micro-batch 8）")
for name, layers, d in [("124M / GPT-2", 12, 768), ("350M", 24, 1024)]:
    a1 = activations(layers, d, 1024, 8, False)
    a2 = activations(layers, d, 1024, 8, True)
    print(f"  {name:<14} 不重计算 {a1 / GB:>5.1f} GB，整层重计算 {a2 / GB:>4.1f} GB，省了 {a1 / a2:>4.0f} 倍")

print()
print("训完要多久（Chinchilla 的 20 token / 参数，6ΨN FLOP）")
for name, params, tflops in [("30M", 30e6, 60), ("124M / GPT-2", 124e6, 90), ("350M", 350e6, 110)]:
    tokens = 20 * params
    hours = 6 * params * tokens / (tflops * 1e12) / 3600
    print(f"  {name:<14} {tokens / 1e9:>5.1f}B token，按实测有效算力 {tflops} TFLOPS 约 {hours:>5.1f} 小时")
```

```text title="output"
模型能不能放进 16 GB（seq 1024，micro-batch 8，整层重计算）
  30M             6 层 × 384   权重+优化器   0.4 GB  激活  0.0 GB  合计   1.7 GB  放得下
  124M / GPT-2   12 层 × 768   权重+优化器   1.8 GB  激活  0.1 GB  合计   3.2 GB  放得下
  350M           24 层 × 1024  权重+优化器   5.2 GB  激活  0.4 GB  合计   6.8 GB  放得下
  770M           24 层 × 1536  权重+优化器  11.5 GB  激活  0.6 GB  合计  13.2 GB  放得下
  1.4B           24 层 × 2048  权重+优化器  20.9 GB  激活  0.8 GB  合计  22.8 GB  放不下，要更小的 batch 或 LoRA

不重计算的话激活是多少（同样 seq 1024、micro-batch 8）
  124M / GPT-2   不重计算   2.4 GB，整层重计算  0.1 GB，省了   17 倍
  350M           不重计算   6.4 GB，整层重计算  0.4 GB，省了   17 倍

训完要多久（Chinchilla 的 20 token / 参数，6ΨN FLOP）
  30M              0.6B token，按实测有效算力 60 TFLOPS 约   0.5 小时
  124M / GPT-2     2.5B token，按实测有效算力 90 TFLOPS 约   5.7 小时
  350M             7.0B token，按实测有效算力 110 TFLOPS 约  37.1 小时
```

Three conclusions:

- **The model states are not the bottleneck; the activations are.** 770M's weights and optimizer take 11.5 GB, which just fits; what actually causes an out-of-memory error is the activations, and recomputation squeezes those to almost nothing.
- **The sweet spot on one card is 100M to 350M.** 124M (GPT-2's scale) trains a full Chinchilla-proportioned run overnight, and 350M takes a day and a half, which is enough to put on a CV and enough for ablation studies.
- **Anything larger calls for a different approach**: do not try to train 1.4B or more from scratch; **fine-tune an existing open model with LoRA** instead (training 0.1% of the parameters takes the optimizer states from $12\Psi$ to almost nothing).

## The small text model: one card, one night {#文本小模型一张卡一晚上}

The architecture is the one from [From scratch (2)](../scratch/model.md), scaled to GPT-2 (12 layers, 768 dimensions, 12 heads, a sequence of 1024). What matters is the training configuration:

```python title="train_text.py" run="no"
# a small GPT-2-scale model, configured for training on one 16 GB card. The architecture is in From scratch (2).
import math
import time

import torch

from model import GPT, GPTConfig          # the model from From scratch (2)

dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")          # put the fp32 matrix multiplies on TF32 too

cfg = GPTConfig(n_layer=12, n_head=12, d_model=768, seq_len=1024, vocab_size=32000)
model = GPT(cfg).to(dev)
model = torch.compile(model)                        # the first compile takes a few minutes, and later runs reuse the cache

MICRO_BS, ACCUM = 8, 16                             # the effective batch is 8 x 16 x 1024, about 131k tokens
STEPS, WARMUP, LR = 20000, 400, 6e-4

# AdamW: weight decay only on parameters of two dimensions or more (not the LayerNorms or the biases)
decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": nodecay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=True)


def lr_at(step):                                     # a linear warm-up plus a cosine decay to 10%
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.1 * LR + 0.45 * LR * (1 + math.cos(math.pi * t))


def get_batch():                                     # replace with your own data loading
    raise NotImplementedError


for step in range(STEPS):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    for micro in range(ACCUM):                       # gradient accumulation: a large batch even with little memory
        x, y = get_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)                        # GPT.forward computes the loss when given targets (using fp32 for the softmax internally)
        (loss / ACCUM).backward()                        # the loss has to be divided by the accumulation steps
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 20 == 0:
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok = MICRO_BS * ACCUM * cfg.seq_len
        print(f"step {step:>6} loss {loss.item():.4f} grad_norm {gnorm:.2f} "
              f"{tok / dt:>8.0f} tok/s 显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
```

The reasons behind each setting:

| Setting | Why |
| --- | --- |
| `bf16` autocast rather than fp16 | bf16's exponent is as wide as fp32's, so it needs no loss scaling and there is no "the scaler keeps adjusting and many steps are wasted" problem (see [Mixed precision and FP8 training](mixed-precision.md)) |
| `torch.compile` | fuses the memory-bound operations of LayerNorm, the residuals and the activation functions. They are a large share on a small model, so the gain is especially visible |
| AdamW with `fused=True` | the parameter update is otherwise hundreds of small kernels, fused into one |
| 16 steps of gradient accumulation | an effective batch of about 131,000 tokens. With too small a batch the loss is erratic and the learning rate cannot be raised |
| `set_float32_matmul_precision("high")` | the remaining fp32 matrix multiplies use TF32, at almost no cost in precision |
| Full-layer recomputation (off in this example) | leave it off while the memory is sufficient, since it costs about 30% more time. To turn it on, wrap just the Transformer layers with `checkpoint_sequential` |

**How to tell whether it is training correctly**:

- **`grad_norm` raises the alarm earlier than the loss.** Normally it settles into a narrow band after the warm-up (0.2 to 1.0, say). A sudden jump into the tens means the loss will take off a few steps later.
- **Reference values for the loss**: at GPT-2's scale with a 32k vocabulary, trained to Chinchilla's proportion, the validation loss is about 3.0 to 3.5 (a perplexity of 20 to 33). The first step's loss should be about $\ln(\text{vocab\_size}) \approx 10.4$; far below that on the first step usually means label leakage (`y` was not shifted right).
- **Reference values for throughput**: compute the model FLOPs utilization as $6\Psi \times \text{tokens/s} / \text{the measured throughput}$. On a consumer card, 35% to 45% is normal; below 20% calls for a profile, usually because data loading cannot keep up or the batch is too small.

!!! tip "The order to follow when memory runs short"
    1. Reduce `MICRO_BS` and raise `ACCUM` to match (the effective batch is unchanged and the throughput barely moves).
    2. Then turn on full-layer recomputation (17 times less activation, 30% more time).
    3. Then halve the sequence length (the activations fall linearly, but long-text ability suffers).
    4. Only then consider an 8-bit optimizer (`bitsandbytes`'s `AdamW8bit`, compressing $8\Psi$ of m and v into $2\Psi$).
    5. If that is still not enough, the model is too large: switch to fine-tuning an existing one with LoRA.

## The small image generator: flow matching {#图像生成小模型流匹配}

Diffusion's formulas look intimidating, but **flow matching** reduces them to one line. The idea is to draw a straight line between the noise and the data:

$$x_t = (1 - t)\,x_0 + t\,x_1, \qquad x_0 \sim \mathcal{N}(0, I),\ x_1 \sim \text{the data}$$

and have the network $v_\theta(x_t, t)$ predict that line's velocity, which is exactly the difference between the ends, $x_1 - x_0$. The training objective is one MSE:

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\ \big\|\, v_\theta(x_t, t) - (x_1 - x_0) \,\big\|^2$$

To sample, start from noise and integrate along this velocity field to $t = 1$ with Euler's method. No noise schedule, no $\bar\alpha_t$, no choice of variance parameterisation: just interpolation and regression.

Here is what it is learning, shown on a two-dimensional distribution on a CPU:

```python title="flow2d.py"
# a minimal flow-matching implementation: the training objective is one line, and it learns a distribution within seconds on a CPU
import math

import torch
import torch.nn as nn

torch.manual_seed(0)
K, R = 8, 2.0                                    # the target distribution: 8 Gaussians on a circle of radius 2


def sample_data(n):
    idx = torch.randint(0, K, (n,))
    ang = idx.float() * (2 * math.pi / K)
    center = torch.stack([R * ang.cos(), R * ang.sin()], dim=1)
    return center + 0.08 * torch.randn(n, 2), idx


class Velocity(nn.Module):
    """v(x, t)：给定位置和时间，预测该往哪个方向走"""

    def __init__(self, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


model = Velocity()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
recent = []
for step in range(1, 4001):
    x1, _ = sample_data(256)                     # the data end
    x0 = torch.randn(256, 2)                     # the noise end
    t = torch.rand(256, 1)
    xt = (1 - t) * x0 + t * x1                   # the straight-line interpolation: t=0 is noise and t=1 is data
    loss = ((model(xt, t) - (x1 - x0)) ** 2).mean()   # the target velocity is the difference between the ends, and this one line is the whole method
    opt.zero_grad()
    loss.backward()
    opt.step()
    recent.append(loss.item())
    if step in (1, 200, 1000, 4000):
        print(f"step {step:>4}  最近 {len(recent):>4} 步的平均 loss {sum(recent) / len(recent):.2f}")
        recent = []


@torch.no_grad()
def sample(n, steps=100):                        # sampling: start from noise and integrate along the velocity field to t=1 with Euler's method
    x = torch.randn(n, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt)
        x = x + model(x, t) * dt
    return x


gen = sample(2000)
ang = torch.arange(K).float() * (2 * math.pi / K)
centers = torch.stack([R * ang.cos(), R * ang.sin()], dim=1)
dist = torch.cdist(gen, centers)                 # each sample's distance to the nearest mode
near = dist.min(dim=1)
hit = (near.values < 0.3).float().mean().item()
counts = torch.bincount(near.indices, minlength=K)
print(f"落在某个模态 0.3 半径内的样本：{hit:.0%}")
print(f"八个模态各分到：{counts.tolist()}（理想是每个 250）")
```

```text title="output"
step    1  最近    1 步的平均 loss 3.20
step  200  最近  199 步的平均 loss 2.26
step 1000  最近  800 步的平均 loss 1.93
step 4000  最近 3000 步的平均 loss 1.87
落在某个模态 0.3 半径内的样本：81%
八个模态各分到：[223, 260, 306, 192, 279, 242, 277, 221]（理想是每个 250）
```

Two things to understand:

- **The loss will not fall to 0, and that is correct.** Given $x_t$ there are infinitely many possible $(x_0, x_1)$ pairs, so the network can only predict the conditional expectation and the remaining variance is irreducible. So judge the training by **the sampling quality** rather than by the loss alone. The real metric above is that 81% of the samples land near a mode and the eight modes are roughly evenly covered.
- **The modes being covered evenly** says there is no mode collapse, which is a GAN's worst problem and one that diffusion and flow matching are naturally unlikely to have.

### Moving to images: a small UNet {#换成图像一个小-unet}

Two dimensions become a $32\times32\times3$ image and the network becomes a UNet; everything else is identical:

```python title="unet.py" run="no"
# a small UNet for 32x32 images: the time is injected into each residual block as a sinusoidal encoding
import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(torch.nn.functional.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(torch.nn.functional.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=128, tdim=256):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        c1, c2, c3 = base, base * 2, base * 2
        self.inp = nn.Conv2d(3, c1, 3, padding=1)
        self.d1, self.d2 = Block(c1, c1, tdim), Block(c1, c2, tdim)        # 32 → 16
        self.down1, self.down2 = nn.Conv2d(c1, c1, 3, 2, 1), nn.Conv2d(c2, c2, 3, 2, 1)
        self.mid = Block(c2, c3, tdim)                                     # an 8x8 bottleneck
        self.u2, self.u1 = Block(c3 + c2, c2, tdim), Block(c2 + c1, c1, tdim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 3, 3, padding=1))

    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t * 1000, self.tdim))
        h0 = self.inp(x)
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.down1(h1), temb)
        m = self.mid(self.down2(h2), temb)
        u2 = self.u2(torch.cat([self.up(m), h2], dim=1), temb)
        u1 = self.u1(torch.cat([self.up(u2), h1], dim=1), temb)
        return self.out(u1)
```

```python title="train_image.py" run="no"
# training a flow-matching image generator on CIFAR-10. The objective is word for word flow2d.py's, with only the network and the data changed.
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from unet import UNet

dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")

tf = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),          # normalised to [-1, 1]
])
ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8, pin_memory=True,
                drop_last=True, persistent_workers=True)

model = UNet().to(dev).to(memory_format=torch.channels_last)
model = torch.compile(model)
ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda a, b, _: 0.999 * a + 0.001 * b)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0, fused=True)

EPOCHS = 200
for epoch in range(EPOCHS):
    t0, total, n = time.perf_counter(), 0.0, 0
    for x1, _ in dl:
        x1 = x1.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=dev)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((model(xt, t) - (x1 - x0)) ** 2).mean()          # exactly the same line as the two-dimensional version
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update_parameters(model)
        total, n = total + loss.item(), n + 1
    print(f"epoch {epoch:>3} loss {total / n:.4f} {time.perf_counter() - t0:.0f}s "
          f"显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
    if epoch % 20 == 19:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, f"ckpt_{epoch}.pt")
```

```python title="sample_image.py" run="no"
# sampling: start from noise and integrate along the velocity field to t=1. As few as 20 steps still looks reasonable
import torch
import torchvision

from unet import UNet

dev = torch.device("cuda")
model = UNet().to(dev)
model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load("ckpt_199.pt")["ema"].items()
                       if not k.startswith("n_averaged")})
model.eval()


@torch.no_grad()
def sample(n=64, steps=50):
    x = torch.randn(n, 3, 32, 32, device=dev)
    for i in range(steps):                                   # Euler's method; Heun's gives better quality at the same step count
        t = torch.full((n,), i / steps, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, t).float()
        x = x + v / steps
    return (x.clamp(-1, 1) + 1) / 2


torchvision.utils.save_image(sample(), "samples.png", nrow=8)
print("已写出 samples.png")
```

This UNet is **6.1M parameters** at `base=128` (the shapes and the backward pass above were verified on a CPU). For better quality raise `base`: 160 is 9.3M and 192 is 13.3M, with the memory and the time growing roughly linearly with the parameters, so 16 GB leaves plenty of room even at 192.

**The expected time and memory** (an estimate; take your own measurement as authoritative): at a batch of 256, 32x32, in bf16, the memory is on the order of 4 to 6 GB; one epoch (50,000 images) takes a minute or two on a 5070 Ti, so 200 epochs finish in a few hours. After a few dozen epochs the outlines and the colours are recognisable; it takes two or three hundred epochs with `base` at 160 or more before the samples obviously look like CIFAR. Do not expect a low FID from a model this size; its value is in having built the whole chain yourself.

**How to tell whether it is training correctly**:

- The loss should fall clearly over the first 5 epochs, then enter a long slow tail.
- Look first at the **colour distribution** of the sampled images (all grey means the velocity field was not learned, all saturated means a missing clamp or a wrong normalisation).
- Images sampled from the EMA weights are noticeably cleaner than from the raw weights, which is common knowledge for diffusion and flow-matching models; do not forget to save the EMA.

### Going one step further {#往上走一步}

- **Higher resolution**: do not train on 256x256 pixels directly, which blows up both the memory and the time. Use **latent diffusion**: take an existing VAE to compress $256\times256\times3$ into $32\times32\times4$, run exactly the same flow matching in the latent space, and decode with the VAE at the end. Training a latent model on one 16 GB card is feasible.
- **Conditional generation**: adding a class embedding to the time embedding gives class conditioning; adding **classifier-free guidance** (dropping the condition with 10% probability during training, and at sampling time extrapolating the conditional and unconditional velocities as $v = v_\varnothing + w\,(v_c - v_\varnothing)$) improves the quality noticeably.
- **Text conditioning**: inject a frozen text encoder's output through cross-attention, which is Stable Diffusion's architecture. Training text-to-image from scratch on one card is unrealistic, but **LoRA fine-tuning an existing model** is entirely feasible and is the commonest practical case.
- **Few-step sampling**: flow matching's paths are close to straight, so few-step sampling works naturally well. Try Heun's method first (two network calls per step but second-order accuracy), then look at reflow and distillation.

## The memory and speed checklist for one card {#一张卡上的省显存与提速清单}

| Measure | Memory saved | Speed | Cost |
| --- | --- | --- | --- |
| bf16 autocast | activations halved | 2 to 4 times | almost none (bf16 needs no loss scaling) |
| `torch.compile` | a little (fusion removes intermediates) | 1.2 to 1.8 times on a small model | a few minutes on the first compile; dynamic shapes recompile repeatedly |
| FlashAttention (`sdpa`) | the attention matrix is never materialised, saving $O(S^2)$ | noticeable on long sequences | none |
| Full-layer activation recomputation | about 17 times | −30% | one extra forward pass |
| Gradient accumulation | linear in the activations | none | none (the effective batch is unchanged) |
| `channels_last` (convolutional networks) | none | 1.1 to 1.3 times | none |
| An 8-bit optimizer | optimizer states $8\Psi \to 2\Psi$ | slightly faster | affects convergence in rare cases |
| LoRA | optimizer states almost nothing | faster | limited expressiveness, for fine-tuning rather than pretraining |
| A smaller micro-batch | linear | slower (the throughput is not fed) | use it last |

The suggested order: **bf16, compile, sdpa, gradient accumulation, recomputation, an 8-bit optimizer, LoRA**. The first three are all but free and everything after has a cost.

## Troubleshooting {#排障}

- **`no kernel image is available`**: the PyTorch wheel has no `sm_120`. Confirm with `torch.cuda.get_arch_list()` and switch to a cu128 or later wheel.
- **Out of memory on the very first step**: usually `torch.compile` trying a very large shape while compiling, or a data loader with `pin_memory` and too many `num_workers`. Turn compile off first to locate it.
- **Out of memory only after a few hundred steps**: fragmentation. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, or fix the sequence length (varying lengths make the allocator carve blocks repeatedly).
- **The loss becomes NaN**: first confirm it is not fp16 (use bf16); then check the learning rate and the warm-up; then look for a division by zero (a row of the attention mask that is entirely $-\infty$).
- **The throughput will not rise and the GPU utilization oscillates**: data loading cannot keep up. Raise `num_workers`, turn on `persistent_workers`, and pack the data into one `.bin` read through memory mapping (see [From scratch (1)](../scratch/data.md)).
- **Recompiling on every step**: the shapes are changing. Drop the last incomplete batch (`drop_last=True`) and fix the sequence length.

!!! interview "How to answer in an interview"
    Asked whether you have trained a model yourself, a complete practical run on one consumer card is a good answer, and the key is to be clear about **the arithmetic**: within 16 GB the model states are $16\Psi$ and the activations are layers x sequence x batch x 34d, so the bottleneck is the activations and recomputation saves 17 times; compute the model FLOPs utilization from the measured throughput and explain why it is 40% rather than 80%; and say what each setting (bf16 rather than fp16, compile, fused AdamW, gradient accumulation) solves. On the image side, explain why flow matching is simpler than DDPM: the objective regresses the difference between the ends on a straight-line interpolation between noise and data, one line of MSE, and sampling solves an ordinary differential equation, which makes few-step sampling naturally good.

## Exercises {#练习}

1. Change `budget.py` to support LoRA: train only a low-rank increment of rank $r$, with the optimizer states acting on those parameters alone. Work out whether 16 GB is enough for a 1.4B model at $r = 16$.

??? success "Answer"
    Under LoRA the base weights need only one bf16 copy ($2\Psi$, with no fp32 copy, no gradients and no m or v), so 1.4B is 2.6 GB. The trainable parameters are about $2 r d L \times 2$ (a pair of $d\times r$ matrices added to each of two projections per layer), which at $r=16$, $d=2048$ and $L=24$ is about 3M parameters, only 50 MB even at $16\Psi$. With the activations and some headroom, about 4 GB in all: plenty, with room to raise both the sequence length and the batch. This is why fine-tuning on one card is entirely realistic while pretraining on one card tops out in the hundreds of millions.

2. In `flow2d.py`, change the sampling steps from 100 to 10, 5 and 2 and watch how the fraction landing near a mode changes. Then replace Euler's method with Heun's (a second-order method with two network calls); how much better is the quality at the same step count?

??? success "Answer"
    At 10 steps most samples usually still land on the modes (flow matching's paths are close to straight, so it is insensitive to the step count), while 2 to 5 steps deviate clearly. Heun's method predicts a step with Euler and then corrects with the average of the velocities at both ends, taking the error from first order to second, which gives better quality for the same budget of network calls. This is also why real systems rarely use plain Euler.

3. Use this chapter's memory formula to explain why, within the same 16 GB, a 350M model with a sequence of 1024 and a 124M model with a sequence of 4096 use about the same activation memory. What does that suggest about choosing a model size?

??? success "Answer"
    The activations are proportional to layers x dimension x sequence length x batch. 350M (24 layers x 1024) has about 2.7 times 124M's (12 x 768) layers times dimension, while a sequence of 4096 is 4 times 1024: the two offset and the magnitudes are close. The lesson is that **the sequence length and the model size are interchangeable in memory**, so a long-context experiment should deliberately shrink the model, or you will conclude the card is too small when the budget was really spent on the wrong dimension.

## Summary {#小结}

- [x] Within 16 GB the model states ($16\Psi$) are not the bottleneck; **the activations** are. Full-layer recomputation squeezes them 17 times at about 30% more time.
- [x] One consumer card's sweet spot is **training 100M to 350M from scratch**; anything larger calls for LoRA fine-tuning.
- [x] Every estimate rests on **the measured bandwidth and throughput**, not the specification; a model FLOPs utilization around 40% is normal.
- [x] **Flow matching** is the easiest way into image generation: regress the difference between the ends on a straight-line interpolation between noise and data, one line of MSE; sampling solves an ordinary differential equation, so few steps produce an image.
- [x] Blackwell (`sm_120`) needs a cu128 wheel of PyTorch 2.7 or newer, and the built-in `sdpa` rather than a third-party FlashAttention wheel.
