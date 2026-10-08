# Samplers and schedulers

<p class="lead">With the same denoising network, changing the scheduler can take the step count from 50 to 20 with better quality; and if the timesteps are distributed wrongly, a high-resolution image comes out a blur. The scheduler decides at which noise levels the network is called and how the latents are updated each time, which makes it the cheapest and most overlooked dial on the inference side. This chapter sets out the differences between the mainstream schedulers, the timestep shift, how to tune the guidance strength, and reproducibility, with every number read straight out of diffusers' scheduler objects.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How do DDIM, DPM-Solver++, UniPC and Euler relate to each other? Why does the same 20 steps give different quality?
    2. What is SD3's and FLUX's "shift"? Why does a higher resolution move the timesteps toward the noise end?
    3. How is the number of function evaluations computed for first- and second-order solvers? What should a benchmark compare?
    4. With the seed fixed, why can two generations still differ? What has to hold for pixel-exact reproducibility?
    5. What happens when the guidance scale is too large? What remedies are there that do not change the model?

??? success "Answers for the self-test (answer first, then open this)"
    1. They are all numerical methods for the denoising equation. Euler and DDIM are first order (using only the current prediction); DPM-Solver++ and UniPC exploit the diffusion equation's semi-linear structure for second- and third-order approximations (reusing earlier steps' predictions with no extra network calls), giving an order of magnitude less trajectory error at the same step count, so 20 steps match a first-order method's 50.
    2. Flow matching's timesteps go through a transform $t' = \frac{s\,t}{1 + (s-1)\,t}$, where $s > 1$ allocates more steps to the high-noise end. The higher the resolution, the higher the image's signal-to-noise ratio at the same noise level (neighbouring pixels are correlated), so the range that genuinely looks like noise shrinks toward the high-noise end and the steps have to move there; FLUX computes $s$ dynamically from the token count.
    3. The number of function evaluations is the number of calls to the denoising network. A first-order method's count equals the step count; a method like Heun with two predictions per step is twice the step count; DPM-Solver++'s multistep version reuses history, so its count equals the step count. Compare quality at the same number of evaluations.
    4. The seed fixes only the initial noise; the order of the reductions in the denoising network's matrix multiplies and attention varies with the kernel, the batch composition and the hardware, and dozens of steps amplify it into a visible difference. Pixel-exact reproducibility requires fixing the hardware, the framework and kernel versions, the batch composition (a request run alone differs from the same one batched with others) and the deterministic-algorithm switches.
    5. Oversaturation, exploded contrast, broken detail. The remedies: a guidance rescale, a dynamic or piecewise guidance strength (large early and small late, or guidance only on the middle steps), and a lower default paired with a better sampler.

## How the timesteps are laid out {#时间步怎么排调度器给出的序列}

The scheduler's first job is to compress training's 1000 timesteps into the N used at inference. Different schedulers lay them out differently; read them out directly:

```python
import torch
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler,
                       EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler)

common = dict(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
N = 8
for name, sch in [("DDIM", DDIMScheduler(**common)), ("DPM-Solver++ (2M)", DPMSolverMultistepScheduler(**common)),
                  ("UniPC", UniPCMultistepScheduler(**common)), ("Euler", EulerDiscreteScheduler(**common))]:
    sch.set_timesteps(N)
    print(f"{name:<18} {[int(t) for t in sch.timesteps]}")
flow = FlowMatchEulerDiscreteScheduler(shift=1.0)
flow.set_timesteps(N)
print(f"{'流匹配 (shift=1)':<18} σ = {[round(s, 3) for s in flow.sigmas.tolist()]}")
```

```text title="output"
DDIM               [875, 750, 625, 500, 375, 250, 125, 0]
DPM-Solver++ (2M)  [999, 874, 749, 624, 500, 375, 250, 125]
UniPC              [999, 874, 749, 624, 500, 375, 250, 125]
Euler              [999, 856, 713, 570, 428, 285, 142, 0]
流匹配 (shift=1)      σ = [1.0, 0.857, 0.715, 0.572, 0.429, 0.286, 0.144, 0.001, 0.0]
```

The DDPM family (the first four) distributes the timesteps roughly evenly over 0 to 999; flow matching gives the noise fraction $\sigma$ directly, falling evenly from 1 to 0. The real difference is not where the network is called but **how each step uses the prediction to update the latents**.

## The solver's order: who is more accurate at the same evaluation count {#求解器的阶数同样的-nfe-谁更准}

Taking [the first chapter](diffusion-inference.md)'s toy equation back and adding a DPM-Solver-style multistep method, which makes no extra network calls and instead extrapolates linearly from the previous step's prediction (like Adams-Bashforth). The three methods' errors at the same number of evaluations:

```python
import math

def v(x, sigma):                                   # the velocity field v = -2σx, whose exact solution from σ=1 to 0 is x(0) = x(1) · e
    return -2.0 * sigma * x

def euler(nfe):
    x, s, d = 1.0, 1.0, 1.0 / nfe
    for _ in range(nfe):
        x, s = x - d * v(x, s), s - d
    return x

def heun(nfe):                                     # two calls per step, so only nfe/2 steps are possible
    steps = nfe // 2
    x, s, d = 1.0, 1.0, 1.0 / steps
    for _ in range(steps):
        v1 = v(x, s); v2 = v(x - d * v1, s - d)
        x, s = x - d * (v1 + v2) / 2, s - d
    return x

def multistep2(nfe):                               # second-order multistep: reuses the previous step's prediction and still calls once per step
    x, s, d = 1.0, 1.0, 1.0 / nfe
    prev = None
    for _ in range(nfe):
        cur = v(x, s)
        slope = cur if prev is None else 1.5 * cur - 0.5 * prev   # extrapolated linearly to the interval's midpoint
        x, s, prev = x - d * slope, s - d, cur
    return x

print(f"{'NFE':>4} {'欧拉 (一阶)':>12} {'Heun (二阶,每步2次)':>18} {'多步二阶 (每步1次)':>18}")
for nfe in (8, 16, 32, 64):
    print(f"{nfe:>4} {abs(euler(nfe) - math.e):>12.5f} {abs(heun(nfe) - math.e):>18.5f} {abs(multistep2(nfe) - math.e):>18.5f}")
```

```text title="output"
 NFE      欧拉 (一阶)     Heun (二阶,每步2次)        多步二阶 (每步1次)
   8      0.10188            0.05740            0.02761
  16      0.05374            0.01443            0.00698
  32      0.02759            0.00359            0.00176
  64      0.01397            0.00089            0.00044
```

**A multistep second-order method gets second-order accuracy at a first-order price**, which is why DPM-Solver++ 2M and UniPC became the defaults: every step still makes one network call while using the previous one or two steps' predictions. The practical rules of thumb:

| Scheduler | Order | Evaluations / step | Recommended steps | Suits |
| --- | --- | --- | --- | --- |
| DDIM | 1 | 1 | 50 | compatibility with older models, and deterministic inversion (image editing) |
| Euler / Euler a | 1 | 1 | 25 to 30 | one of SDXL's defaults; the ancestral variant adds noise each step, which is non-deterministic but more varied |
| DPM-Solver++ 2M | 2 | 1 | 20 to 25 | the best value for SD 1.5 and SDXL |
| DPM-Solver++ SDE | 2 | 1 | 25 to 30 | adds randomness for richer detail, but worse than 2M at few steps |
| UniPC | 2 to 3 | 1 | 15 to 20 | the general choice when steps are scarcest |
| Flow-matching Euler | 1 | 1 | 20 to 50 | SD3's, FLUX's and Wan's default; the trajectory is near straight so first order suffices |
| Distilled-model schedulers (LCM, Lightning) | — | 1 | 1 to 8 | only usable with their matching models, see [Few-step generation](../perf/distillation.md) |

## The timestep shift: why high resolution moves toward the noise end {#时间步偏移高分辨率为什么要往噪声端挪}

A flow-matching model's scheduler has an extra parameter, `shift`, applying a monotone transform to the uniform timesteps:

$$
t' = \frac{s \cdot t}{1 + (s - 1)\, t}
$$

With $s > 1$, more steps are allocated to the high-noise end (where $\sigma$ is near 1). Its effect on an allocation of 8 steps:

```python
for shift in (1.0, 3.0, 6.0):
    sch = FlowMatchEulerDiscreteScheduler(shift=shift)
    sch.set_timesteps(8)
    sig = sch.sigmas.tolist()[:-1]
    high = sum(1 for s in sig if s > 0.5)
    print(f"shift={shift:<4} σ = {[round(s, 2) for s in sig]}   σ>0.5 的步数 {high}/8")
```

```text title="output"
shift=1.0  σ = [1.0, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14, 0.0]   σ>0.5 的步数 4/8
shift=3.0  σ = [1.0, 0.95, 0.88, 0.8, 0.69, 0.55, 0.34, 0.01]   σ>0.5 的步数 6/8
shift=6.0  σ = [1.0, 0.97, 0.94, 0.89, 0.82, 0.71, 0.51, 0.03]   σ>0.5 的步数 7/8
```

Pull the step count and the shift and see where each step lands on the σ axis:

<div class="aig-widget" data-widget="sigma-schedule"></div>

Why a higher resolution needs it more: at the same noise fraction $\sigma$, a high-resolution image's neighbouring pixels are strongly correlated and the signal's energy is more concentrated in the latent space, so the effective signal-to-noise ratio is higher than at low resolution. At high resolution $\sigma = 0.5$ is already fairly clean and what needs walking carefully is the stretch where $\sigma$ is near 1. SD3's paper gives the rule of thumb $s = \sqrt{m / n}$, where $m$ is the target token count and $n$ the baseline (256²'s token count). FLUX goes further and interpolates $\mu = \text{shift}$ linearly from the token count, then $s = e^{\mu}$:

```python
def flux_shift(image_seq_len, base_len=256, max_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_len - base_len)
    mu = m * image_seq_len + (base_shift - m * base_len)
    return math.exp(mu)

for res in (512, 1024, 2048):
    tokens = (res // 8 // 2) ** 2
    print(f"FLUX {res}²：{tokens:>5} 个 token → shift = {flux_shift(tokens):.2f}")
```

```text title="output"
FLUX 512²： 1024 个 token → shift = 1.88
FLUX 1024²： 4096 个 token → shift = 3.16
FLUX 2048²：16384 个 token → shift = 25.28
```

One common cause of "FLUX produces a blurry 2K image" is exactly this: the scheduler's shift was not updated for the resolution, so the token count quadrupled while the timestep allocation stayed at 1K's.

## The guidance strength: the numbers and the tuning {#引导强度数值与调法}

The guidance scale is the parameter users touch most and the easiest to overdo. It affects an inference system in three ways:

1. **Cost**: any $w \ne 1$ means two forward passes (see [the first chapter](diffusion-inference.md)).
2. **Quality**: too large oversaturates and breaks the detail, too small disobeys the prompt; each model has its own sweet spot (about 7 for SD 1.5, 5 for SDXL, 4 for SD3, and about 3.5 for FLUX dev's `guidance`).
3. **It can vary by step**: the early steps decide the composition and want strong guidance, the later ones only fill in detail and can have it weakened or turned off. The steps with it off compute only one forward pass, which saves compute directly.

```python
def guidance_schedule(steps, w_max=7.5, w_min=1.0, mode="constant"):
    if mode == "constant":
        return [w_max] * steps
    if mode == "linear_decay":                      # strong early and weak late
        return [w_max - (w_max - w_min) * i / (steps - 1) for i in range(steps)]
    if mode == "interval":                          # guidance only on the middle 60% of the steps (w=1 at both ends, computing one path)
        lo, hi = int(steps * 0.1), int(steps * 0.7)
        return [w_max if lo <= i < hi else 1.0 for i in range(steps)]

steps = 20
for mode in ("constant", "linear_decay", "interval"):
    ws = guidance_schedule(steps, mode=mode)
    nfe = sum(2 if w != 1.0 else 1 for w in ws)
    print(f"{mode:<13} 平均 w {sum(ws) / steps:>4.1f}   去噪网络前向次数 {nfe:>3}（最多 {2 * steps}）")
```

```text title="output"
constant      平均 w  7.5   去噪网络前向次数  40（最多 40）
linear_decay  平均 w  4.2   去噪网络前向次数  39（最多 40）
interval      平均 w  4.9   去噪网络前向次数  32（最多 40）
```

Using guidance only over an interval of timesteps saves 40% of the forward passes without touching the model. The reasoning behind it is that guidance is most useful at the middle noise levels (too early there is no structure to guide and too late only texture is left). A guidance rescale is a different direction: after the guidance, pull the prediction's standard deviation back to the conditional prediction's, which treats oversaturation specifically.

## Reproducibility: what besides the seed has to be fixed {#可复现种子之外还要固定什么}

"Fix the seed and it reproduces" holds only for the initial noise in a diffusion model. What else affects the final pixels:

| Factor | Why it varies | How to fix it |
| --- | --- | --- |
| The initial noise | set by the `Generator` and the seed; the same seed's random sequence also differs across devices | generate the noise on the CPU and move it to the GPU (diffusers' common practice) |
| The batch composition | run alone or batched, a matrix multiply is partitioned differently and the reductions happen in a different order | run requests that need reproducibility on their own, or fix the batch shape |
| The attention kernel | FlashAttention and SDPA's different backends and block sizes reduce in different orders | fix the backend; use a deterministic implementation when needed |
| The framework and hardware | cuBLAS's algorithm selection, the GPU architecture | record the versions; do not promise pixel-exact results across hardware |
| The scheduler's state | a multistep scheduler (DPM-Solver++ 2M) reuses earlier predictions, and a different history gives a different result | give each request its own scheduler instance and never share one between requests |

That last one is the serving layer's easiest trap: **a multistep scheduler has state**. Using one global scheduler object for concurrent requests mixes their prediction histories, and the result is neither reproducible nor correct. Each request in an inference engine needs its own scheduler instance (or the state has to live explicitly in the request object, see [Scheduling a generation service](../serving/scheduling.md)).

```python
sch = DPMSolverMultistepScheduler(**common)
sch.set_timesteps(4)
print(f"DPM-Solver++ 2M 的内部状态：model_outputs 缓存 {sch.config.solver_order} 项，lower_order_nums 计数器——这些都随请求走")
print(f"调度器实例是有状态的：{hasattr(sch, 'model_outputs')}；把它在并发请求间共享，预测历史就会串")
```

```text title="output"
DPM-Solver++ 2M 的内部状态：model_outputs 缓存 2 项，lower_order_nums 计数器——这些都随请求走
调度器实例是有状态的：True；把它在并发请求间共享，预测历史就会串
```

!!! interview "How to explain it"
    To explain how to choose a sampler and the step count, give the framework first: they all solve the same equation and differ in the solver's order and the network calls per step, so comparisons go by the number of function evaluations; a multistep higher-order method (DPM-Solver++ 2M, UniPC) gets second-order accuracy at a first-order price and is the default for the SD family; a flow-matching model's trajectory is near straight so first-order Euler suffices, but the shift has to follow the resolution. Then the systems view: guidance decides whether a step is one forward pass or two, and interval guidance saves 40% without touching the model; a multistep scheduler has state and cannot be shared between requests; and reproducibility needs more than the seed, namely the batch composition and the kernels too.

## Exercises {#练习}

1. Change `multistep2` to third order (extrapolating quadratically from the previous two steps' predictions). How much further does the error fall at the same evaluation count? Does it still win at very few steps (4 evaluations)?

??? success "Answer"
    Third-order extrapolation's coefficients are $\frac{23}{12}c_0 - \frac{16}{12}c_1 + \frac{5}{12}c_2$ (Adams-Bashforth 3), and at 32 evaluations the error falls another order of magnitude. But at very few evaluations the history points are too far apart for the extrapolation to be accurate and it can be worse than second order, which is exactly why DPM-Solver++ falls back to a lower order at very few steps and UniPC lowers the order on its final steps.

2. SD3's rule of thumb is $s = \sqrt{m/n}$ against a 256² baseline: compute the shift for 512², 1024² and 2048², and compare with FLUX's formula.

??? success "Answer"
    256²'s token count is n = 256 (f8 with patch 2) and 1024²'s is 4096, so $s = \sqrt{16} = 4$; 512² gives $s = 2$ and 2048² gives $s = 8$. FLUX's formula gives $e^{1.15} \approx 3.16$ at 1024², the same order as SD3's; but it extrapolates $\mu$ **linearly**, so 2048²'s 16384 tokens push it to $\mu \approx 3.2$ and $s \approx 25$, which is clearly too far (that is the number the code above prints). So when generating above 2K with FLUX, practice often caps $\mu$ near `max_shift`, or switches to SD3's square-root rule. The two agree on only one thing: quadrupling the resolution roughly doubles the shift.

3. A request in the service asks to be pixel-identical to yesterday's result. List everything you have to fix, and say which is most easily overlooked.

??? success "Answer"
    The seed and the device the noise is generated on, the model weights' version, the scheduler type and step count, the guidance scale and its schedule, the resolution, the framework and kernel versions, the hardware model, the batch composition (run alone or batched with others), the attention backend, and the deterministic-algorithm switches. The one most easily missed is the batch composition: under different concurrency the same request lands in a batch of a different shape and the result changes. A request that needs reproducibility has to run on its own.

## Summary {#小结}

- [x] The scheduler decides at which noise levels the network is called and how each step updates; the DDPM family distributes the timesteps evenly while flow matching gives the noise fraction $\sigma$ directly.
- [x] Compare samplers by the number of function evaluations: a multistep higher-order method (DPM-Solver++ 2M, UniPC) makes one call per step with second-order accuracy and 20 steps suffice for the SD family; a flow-matching model's trajectory is near straight, so 20 to 50 Euler steps.
- [x] High resolution moves the timesteps toward the noise end (the shift): SD3 by $\sqrt{m/n}$ and FLUX by $\mu$ from the token count; forgetting to update the shift is a common cause of blurry high-resolution images.
- [x] Guidance decides whether a step is one forward pass or two, and interval guidance saves 40% of the compute; a multistep scheduler has state and cannot be shared between requests; reproducibility needs the batch composition, the kernels and the hardware fixed as well as the seed.
