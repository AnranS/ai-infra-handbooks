# Diffusion and flow matching from an inference point of view

<p class="lead">A language model's inference is one forward pass per token; an image or video generator's inference is dozens of forward passes turning a cloud of noise into a picture. This chapter covers only the mathematics you actually meet at inference time: how the noise is added and then removed step by step, what the network is really predicting, why flow matching makes this simpler, and why classifier-free guidance doubles the computation of every step. With that in hand, every later technique for taking fewer steps or caching the previous one has somewhere to stand.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What are the network's input and output during diffusion inference? What is the difference between predicting the noise and predicting the velocity?
    2. Why does DDPM need 1000 steps while DDIM and DPM-Solver need only 20 to 50?
    3. What equation does flow matching's sampling solve? Why is it friendlier to few-step sampling?
    4. What is classifier-free guidance's formula? Why does it double a step's forward computation?
    5. With the same seed and the same prompt, why might another card give a different result?

??? success "Answers for the self-test (answer first, then open this)"
    1. The input is the noisy sample $x_t$ (a tensor in the latent space) and the timestep $t$, plus the text condition; the output is a prediction of the noise or the velocity. Predicting the noise $\epsilon$ is DDPM's parameterisation; predicting the velocity $v = x_1 - x_0$ (flow matching) or $v$-prediction is numerically steadier, especially on the steps close to pure noise.
    2. DDPM's sampling is a Markov chain that adds fresh noise at every step, so the steps cannot be large. DDIM rewrites it as a deterministic ordinary differential equation, and higher-order solvers like DPM-Solver exploit the equation's semi-linear structure to take more accurate steps, so 20 to 50 reach the same quality.
    3. It solves an ordinary differential equation $\mathrm{d}x/\mathrm{d}t = v_\theta(x_t, t)$, integrating from noise at $t=0$ to data at $t=1$. Flow matching trains on a straight-line interpolation, so the learned velocity field's trajectories are close to straight and Euler's method is accurate within a few steps.
    4. $\tilde\epsilon = \epsilon_\varnothing + w\,(\epsilon_c - \epsilon_\varnothing)$: compute conditionally and unconditionally once each and extrapolate by the guidance strength $w$. The two forward passes are usually batched together at a batch of 2, which doubles both the compute and the activation memory.
    5. The initial noise is set by the seed, but the order of the floating-point reductions in the denoising network's matrix multiplies and attention varies with the kernel, the batch size and the hardware, and dozens of steps later a tiny difference is amplified into a visible one. Reproducibility requires fixing the hardware, the kernel implementations and the batch composition (see [Evaluation and load testing](../serving/benchmark.md)).

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/diffusion.webp is in Chinese; put it back once the English version exists -->

## Adding noise: a path from data to noise {#加噪从数据到噪声的一条路}

A diffusion model's forward process is not learned; it is just a formula mixing the clean sample $x_0$ with Gaussian noise $\epsilon$ by the timestep $t$. The DDPM family uses

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

where $\bar\alpha_t$ falls monotonically from 1 ($t=0$, no noise) to near 0 ($t = T$, nearly pure noise). Flow matching uses a simpler straight line:

$$
x_t = (1 - t)\, x_0 + t\, \epsilon,\qquad t \in [0, 1]
$$

Both are really defining how the signal-to-noise ratio varies with $t$. At inference the model's job is to walk **back along** this path.

Here is the path drawn out: a ring of data points on a plane, each with a fixed noise sample, travelling along t from 0 to 1. Drag to rotate, pull t for a cross-section, and change the noising scheme to see the path's shape:

<div class="aig-widget" data-widget="diffusion3d"></div>

Drawing both paths with two of diffusers' schedulers, which are the very objects used at inference:

```python
import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler

ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")   # the β schedule SD 1.5 and SDXL use
flow = FlowMatchEulerDiscreteScheduler()                                        # the flow matching SD3, FLUX and Wan use

print("DDPM 族：ᾱ_t 决定信号占比，sqrt(1-ᾱ_t) 决定噪声占比")
for t in (0, 250, 500, 750, 999):
    a = ddim.alphas_cumprod[t].item()
    print(f"  t={t:>3}  信号 {a ** 0.5:.3f}  噪声 {(1 - a) ** 0.5:.3f}  信噪比 {a / (1 - a + 1e-12):9.4f}")

flow.set_timesteps(4)
print("流匹配：t 直接就是噪声占比（这里把 1000 步归一化到 0～1）")
for s in flow.sigmas.tolist():
    print(f"  σ={s:.3f}  信号 {1 - s:.3f}  噪声 {s:.3f}")
```

```text title="output"
DDPM 族：ᾱ_t 决定信号占比，sqrt(1-ᾱ_t) 决定噪声占比
  t=  0  信号 1.000  噪声 0.010  信噪比 9997.3408
  t=250  信号 0.906  噪声 0.424  信噪比    4.5559
  t=500  信号 0.576  噪声 0.818  信噪比    0.4954
  t=750  信号 0.195  噪声 0.981  信噪比    0.0397
  t=999  信号 0.027  噪声 1.000  信噪比    0.0007
流匹配：t 直接就是噪声占比（这里把 1000 步归一化到 0～1）
  σ=1.000  信号 0.000  噪声 1.000
  σ=0.667  信号 0.333  噪声 0.667
  σ=0.334  信号 0.666  噪声 0.334
  σ=0.001  信号 0.999  噪声 0.001
  σ=0.000  信号 1.000  噪声 0.000
```

Note that SD 1.5's schedule still has 6.8% signal at $t=999$: a little mean information survives in the pure noise, which is why the old SD's images are grey overall and struggle to produce pure black or white (later work corrected it with zero terminal SNR). Flow matching does not have the problem: $\sigma = 1$ is pure noise.

## What the model predicts {#模型在预测什么}

In training the model sees $x_t$ and $t$, and there are several equivalent choices for what it predicts:

| Parameterisation | Prediction target | Who uses it | How $x_0$ is recovered at inference |
| --- | --- | --- | --- |
| $\epsilon$-prediction | the noise $\epsilon$ that was added | SD 1.5, SDXL | $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ |
| $v$-prediction | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | SD 2.x, some video models | a linear combination, numerically steadier at the high-noise end |
| flow matching's velocity | $v = \epsilon - x_0$ | SD3, FLUX, Wan, HunyuanVideo | $\hat x_0 = x_t - \sigma\,\hat v$ |

For an inference system they mean one thing: **every step's input and output have exactly the same shape** (a latent tensor), so one denoising step is one forward pass of fixed shape. That is the fundamental reason CUDA graphs, static batching and feature caching all apply here.

## Denoising: solving an ordinary differential equation {#去噪解一个常微分方程}

DDPM's original sampling adds fresh random noise at every step, so it has to take 1000 small ones. DDIM's insight is that the same training objective can be rewritten as a **deterministic ordinary differential equation**, integrated from pure noise to data, so the step count becomes a question of the solver's accuracy rather than a requirement of the model. Flow matching is more direct: what it learns is the equation's velocity field.

$$
\frac{\mathrm{d}x}{\mathrm{d}\sigma} = v_\theta(x_\sigma, \sigma), \qquad x_{\sigma - \Delta} = x_\sigma - \Delta \cdot v_\theta(x_\sigma, \sigma)
$$

That last expression is Euler's method. A toy velocity field with a closed-form solution shows what the step count and the solver each contribute, with no neural network needed to make the conclusion precise:

```python
import math

# a toy: the velocity field v(x, σ) = -2σx, whose exact solution integrating from σ=1 to 0 is x(0) = x(1) · e
def v(x, sigma):
    return -2.0 * sigma * x

def euler(steps):
    x, sigma, d = 1.0, 1.0, 1.0 / steps
    for _ in range(steps):
        x = x - d * v(x, sigma)                 # one forward pass
        sigma -= d
    return x

def heun(steps):                                # second order: two forward passes per step
    x, sigma, d = 1.0, 1.0, 1.0 / steps
    for _ in range(steps):
        v1 = v(x, sigma)
        x_pred = x - d * v1
        v2 = v(x_pred, sigma - d)
        x = x - d * (v1 + v2) / 2
        sigma -= d
    return x

print(f"{'步数':>4} {'欧拉法误差':>12} {'Heun 误差':>12}  （同样的网络调用次数：Heun 用一半的步数）")
for steps in (4, 8, 16, 32, 64):
    e1 = abs(euler(steps) - math.e)
    e2 = abs(heun(steps // 2) - math.e)
    print(f"{steps:>4} {e1:>12.5f} {e2:>12.5f}")
```

```text title="output"
  步数        欧拉法误差      Heun 误差  （同样的网络调用次数：Heun 用一半的步数）
   4      0.18211      0.21828
   8      0.10188      0.05740
  16      0.05374      0.01443
  32      0.02759      0.00359
  64      0.01397      0.00089
```

Dial the number of network calls and watch both solvers' trajectories and errors:

<div class="aig-widget" data-widget="ode-solver"></div>

Two conclusions for inference:

- **The error falls linearly with the step count (Euler) or quadratically (Heun)**: double the steps and Euler's error halves while Heun's falls to a quarter. Past a modest step count, a higher-order solver is an order of magnitude more accurate for the same budget of network calls (at very few steps it can be worse, as the table's first row shows). DPM-Solver++ and UniPC are higher-order methods designed for the diffusion equation's semi-linear structure, and they are why SD 1.5 went from 50 DDIM steps to 20.
- **The number of network calls is the cost**, not the step count. One Heun step is two forward passes. Read a benchmark by its number of function evaluations.

A real model's velocity field is not straight, but flow matching trains on a straight-line interpolation so the learned trajectories are **close** to straight, which is why FLUX, Wan and similar models need only 20 to 50 steps of plain Euler; pushing below that takes distillation (see [Few-step generation](../perf/distillation.md)).

## Classifier-free guidance: why one step computes twice {#无分类器引导为什么一步要算两次}

The model learns $p(x \mid c)$, but sampling from it directly does not follow the prompt closely enough. Classifier-free guidance drops the condition $c$ at random during training so that the same model can also make an unconditional prediction, and extrapolates between the two at inference:

$$
\tilde v = v_\varnothing + w\,(v_c - v_\varnothing)
$$

In distributional terms that extrapolation is equivalent to sampling from $p_c(x)^w\, p_\varnothing(x)^{1-w}$. Here is what happens as $w$ goes from 0 to 10, on two one-dimensional distributions:

<div class="aig-widget" data-widget="cfg-guide"></div>

$w$ is what is usually called the guidance scale (7.5 for SD 1.5, 5 to 7 for SDXL, 3.5 for FLUX dev). The cost is plain: **every step computes two forward passes**. In practice the conditional and unconditional inputs are concatenated into one batch:

```python
import torch

torch.manual_seed(0)
B, C, H, W = 1, 4, 8, 8
x_t = torch.randn(B, C, H, W)
cond = torch.randn(B, 8, 32)                    # the text encoding
uncond = torch.zeros(B, 8, 32)                  # the empty prompt's encoding (in a real model, the vector from encoding "")

calls = 0
def fake_model(x, c):                           # standing in for the denoising network: counting the calls and the batch size
    global calls
    calls += 1
    return x * 0.1 + c.mean(dim=(1, 2))[:, None, None, None]

# the naive form: two forward passes
v_c, v_u = fake_model(x_t, cond), fake_model(x_t, uncond)
w = 7.5
v_guided = v_u + w * (v_c - v_u)
print(f"朴素写法：调用 {calls} 次，每次 batch={B}")

# batched: one forward pass at twice the batch
calls = 0
v_both = fake_model(torch.cat([x_t, x_t]), torch.cat([uncond, cond]))
v_u2, v_c2 = v_both.chunk(2)
v_guided2 = v_u2 + w * (v_c2 - v_u2)
print(f"拼 batch：调用 {calls} 次，batch={2 * B}，结果一致：{torch.allclose(v_guided, v_guided2)}")
print(f"引导后的预测范数 {v_guided.norm():.3f}，无条件 {v_u.norm():.3f}，有条件 {v_c.norm():.3f}：w=7.5 把差异放大了")
```

```text title="output"
朴素写法：调用 2 次，每次 batch=1
拼 batch：调用 1 次，batch=2，结果一致：True
引导后的预测范数 12.037，无条件 1.503，有条件 2.215：w=7.5 把差异放大了
```

What guidance means for an inference system:

- **Both the compute and the activation memory are twice the unguided ones.** Batching them together only saves kernel launches; it saves no FLOPs.
- The two passes differ only in the condition, so the conditional path and the unconditional one can be computed in parallel on two cards (guidance parallelism, see [Multi-GPU parallelism](../perf/parallel.md)).
- Distillation can **bake** the guidance into the model (FLUX.1-dev's `guidance` is an input scalar rather than two forward passes), so one step computes once. That is guidance distillation (see [Few-step generation](../perf/distillation.md)).
- Too large a $w$ oversaturates and breaks the details; the usual remedies are a guidance rescale, or guidance only on some timesteps, which also saves compute.

## The inference summary: how many forward passes one generation takes {#推理视角的总结一次生成要算多少次前向}

Putting this chapter's quantities together, the cost of one image is:

$$
\text{forward passes} = \text{steps} \times (2 \text{ with guidance, else } 1) \times (\text{calls per step of a higher-order solver})
$$

```python
cases = [("SD 1.5 · DDIM 50 步 · CFG",            50, 2, 1), ("SD 1.5 · DPM-Solver++ 20 步 · CFG",  20, 2, 1),
         ("SDXL · 30 步 · CFG",                      30, 2, 1), ("FLUX.1-dev · 28 步 · 引导已蒸馏",      28, 1, 1),
         ("FLUX.1-schnell · 4 步",                    4, 1, 1), ("Wan 2.1 · 50 步 · CFG",                 50, 2, 1),
         ("SDXL-Lightning · 4 步 · 无 CFG",           4, 1, 1)]
print(f"{'配置':<36} 去噪网络前向次数")
for name, steps, cfg, per in cases:
    print(f"{name:<36} {steps * cfg * per:>6}")
```

```text title="output"
配置                                   去噪网络前向次数
SD 1.5 · DDIM 50 步 · CFG                100
SD 1.5 · DPM-Solver++ 20 步 · CFG         40
SDXL · 30 步 · CFG                        60
FLUX.1-dev · 28 步 · 引导已蒸馏                28
FLUX.1-schnell · 4 步                      4
Wan 2.1 · 50 步 · CFG                    100
SDXL-Lightning · 4 步 · 无 CFG              4
```

From 100 passes to 4 is a factor of 25. Every chapter of the performance part works on one term of this formula: a better solver reduces the steps, distillation removes the guidance and the steps, caching makes some passes cheaper, and parallelism makes one pass faster.

!!! interview "How to answer in an interview"
    Asked what most distinguishes diffusion inference from LLM inference, there are three points: (1) an LLM is autoregressive with a shape that changes every step (the KV grows), while diffusion passes a fixed-shape tensor through the same network dozens of times, which makes CUDA graphs, static batching and feature caching all more useful; (2) diffusion's cost formula is steps x the guidance factor x the forward passes per step, and every optimisation turns on those three; (3) diffusion has nothing like a KV cache to reuse across steps, so every step computes attention in full and the token count (resolution, frame count) decides everything. A video model's attention token count easily reaches a hundred thousand, which is why it is even more compute-intensive than an LLM.

## Exercises {#练习}

1. Replace the toy equation with $v(x, \sigma) = -x/(1+\sigma)$ (whose exact solution is $x(0) = 2\,x(1)$) and repeat the Euler and Heun experiment. What does Heun's error become? Why?

??? success "Answer"
    Heun's error is 0 (to floating-point precision). This field's solution is $x(\sigma) = C/(1+\sigma)$, and Heun's method (the trapezoidal rule) is exact whenever the velocity varies linearly within a step, which this field satisfies. Euler's method still has a first-order error. It shows that a solver's error depends on the velocity field's shape: a real diffusion model's field curves most at the high-noise end ($\sigma$ near 1), which is why many schedules place the steps more densely there (see [Samplers and schedulers](schedulers.md)).

2. Take $w$ from 1 to 15 and use this chapter's `fake_model` to see how the guided prediction's norm changes. What goes wrong in a real model when $w$ is too large? What are the engineering remedies?

??? success "Answer"
    The norm grows linearly with $w$, because the extrapolation term $w(v_c - v_\varnothing)$ is proportional to it. In a real model, too large a $w$ pushes the latents out of the training distribution: oversaturated colours, exploded contrast, broken structure. The remedies: a guidance rescale (pulling the guided standard deviation back to the conditional prediction's), dynamic guidance (large early, small late), guidance only on the middle timesteps, or simply a guidance-distilled model.

3. Count them: how many denoising forward passes does FLUX.1-dev take at 28 steps without guidance, and SDXL at 30 steps with it? If one FLUX pass takes 7 times what an SDXL pass does, which is slower?

??? success "Answer"
    FLUX: 28. SDXL: 60. At 7 times each, FLUX's total is equivalent to 196 SDXL passes, over three times SDXL's 60. The per-step cost of a larger parameter count is not offset by dropping guidance. That is why acceleration for a model like FLUX concentrates on the step itself (quantization, caching, parallelism) rather than on the step count.

## Summary {#小结}

- [x] Adding noise follows a fixed path (DDPM's $\bar\alpha_t$ or flow matching's straight line) and inference walks back along it; the model predicts the noise or the velocity at each step with fixed input and output shapes.
- [x] Denoising solves an ordinary differential equation: the step count is a question of the solver's accuracy, a higher-order solver is an order of magnitude more accurate for the same budget of network calls, and the cost is counted in function evaluations.
- [x] Classifier-free guidance makes every step compute two forward passes, doubling both compute and activation memory; guidance distillation can bake it into the model.
- [x] One image's cost = steps x the guidance factor x the forward passes per step, and from 100 to 4 is a factor of 25. Every later acceleration works on this formula.
