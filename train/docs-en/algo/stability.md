# Training stability: loss spikes, QK-Norm, QK-Clip and low precision

<p class="lead">Small models rarely give trouble, while large ones often throw a loss spike or even diverge after weeks of training, leaving no choice but to roll back to an earlier checkpoint, skip a batch of data and lower the learning rate. The root cause can usually be traced to a few places where the numbers are growing: attention's logits, the output layer's logits, the residual stream, and the small quantities rounded away in low precision. This chapter runs each of them as a small experiment, then looks at what Gemma, PaLM, Kimi K2, DeepSeek and others each did about it.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why do attention's logits grow over training? What happens when they do?
    2. How do QK-Norm, logit soft-capping and QK-Clip each control attention's logits? Why does Kimi K2 not use QK-Norm?
    3. What does z-loss penalise? Why does cross-entropy not control it by itself?
    4. Hyper-Connections turn the residual stream into several streams; why does mHC constrain the mixing matrix to be doubly stochastic?
    5. In FP4 training, why do the gradients use stochastic rounding and the weights two-dimensional block scaling?

??? success "Answers for the self-test (answer first, then open this)"
    1. Attention's logits are roughly the product of the norms of $W_q$ and $W_k$, and those weights' norms grow slowly over training, so the logits get larger and larger; the softmax approaches one-hot (attention entropy collapse), the gradients become unstable, and the result is a loss spike or divergence.
    2. QK-Norm: normalise each head's q and k first, so the logits are independent of the weights' scale. Soft-capping: clip the logits smoothly with $\mathrm{cap} \cdot \tanh(x/\mathrm{cap})$. QK-Clip: after each optimizer update, check each head's maximum logit and multiply the offending heads' $W_q$ and $W_k$ each by $\sqrt{\tau / S_{max}}$. Kimi K2 uses MLA, where the keys are never computed explicitly per head at inference time (they are absorbed into the query), so there is nowhere to insert a QK-Norm.
    3. It penalises the softmax's normalising constant $\log Z$ for drifting from 0 ($10^{-4} \log^2 Z$). Cross-entropy only sees the differences between logits, so adding a constant to all of them leaves the loss unchanged and $\log Z$ can drift far, where bf16 does not have the precision and the numbers become unstable.
    4. The several residual streams exchange information through a mixing matrix, and multiplying dozens of layers' matrices together amplifies or shrinks the signal (destroying the identity path). A doubly stochastic matrix (rows and columns summing to 1, entries non-negative) always has a spectral norm of 1 and stays doubly stochastic under multiplication, which preserves the residual's identity path; the projection uses a Sinkhorn iteration.
    5. If the gradients' quantisation error always rounds the same way, it accumulates into a bias; stochastic rounding makes the expectation unbiased. The weights are used as $W$ in the forward pass and $W^\top$ in the backward, and one-dimensional row blocks no longer line up after the transpose, while 16x16 two-dimensional blocks give the forward and backward passes the same quantisation.

## Attention logits growing, and entropy collapse {#注意力-logits-的增长与熵坍缩}

Attention's logits are $q \cdot k / \sqrt{d}$, where $q = W_q x$ and $k = W_k x$. Over training, the norms of $W_q$ and $W_k$ tend to grow together and the logits grow with their **product**. Compressing that process into one experiment, here is what happens to the attention distribution:

Get a feel for the relation between scale and entropy first:

<div class="aig-widget" data-widget="softmax-entropy"></div>

```python title="logit_growth.py"
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, d = 512, 64
x = torch.randn(T, 256)                                       # one attention layer's input (already normalised)
Wq, Wk = torch.randn(256, d) / 16, torch.randn(256, d) / 16   # at initialisation each component of q and k is about N(0, 1)


def attention_stats(scale, qk_norm):
    q, k = x @ (Wq * scale), x @ (Wk * scale)                  # over training, the norms of W_q and W_k grow together
    if qk_norm:                                               # QK-Norm: an RMSNorm on each of q and k before the dot product
        q, k = F.rms_norm(q, (d,)), F.rms_norm(k, (d,))
    logits = (q @ k.T / math.sqrt(d)).tril() + torch.full((T, T), float("-inf")).triu(1)
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)[T // 2:].mean()
    return logits[torch.isfinite(logits)].abs().max().item(), entropy.item(), p.max(-1).values[T // 2:].mean().item()


print(f"均匀分布的熵：{math.log(T // 2):.2f}～{math.log(T):.2f}（后一半 query 各自能看到 256～512 个 key）")
for qk_norm in (False, True):
    for scale in (1, 2, 4, 8):
        m, h, top = attention_stats(scale, qk_norm)
        print(f"{'QK-Norm ' if qk_norm else '普通     '} 权重放大 {scale} 倍：最大 logit {m:6.1f}，平均熵 {h:4.2f}，最大注意力权重 {top:.2f}")
```

```text title="output"
均匀分布的熵：5.55～6.24（后一半 query 各自能看到 256～512 个 key）
普通      权重放大 1 倍：最大 logit    5.1，平均熵 5.43，最大注意力权重 0.04
普通      权重放大 2 倍：最大 logit   20.6，平均熵 1.88，最大注意力权重 0.51
普通      权重放大 4 倍：最大 logit   82.4，平均熵 0.32，最大注意力权重 0.88
普通      权重放大 8 倍：最大 logit  329.4，平均熵 0.08，最大注意力权重 0.97
QK-Norm  权重放大 1 倍：最大 logit    4.9，平均熵 5.45，最大注意力权重 0.03
QK-Norm  权重放大 2 倍：最大 logit    4.9，平均熵 5.45，最大注意力权重 0.03
QK-Norm  权重放大 4 倍：最大 logit    4.9，平均熵 5.45，最大注意力权重 0.03
QK-Norm  权重放大 8 倍：最大 logit    4.9，平均熵 5.45，最大注意力权重 0.03
```

Scaling the weights by only 4 makes the logits 16 times larger, and attention's entropy falls from 5.4 (close to uniform) to 0.3: each query looks at essentially one key (an attention weight of 0.88). This is **attention entropy collapse**: the softmax enters saturation, the gradients for most positions approach 0 while those for a few are very large, and training becomes fragile here, often showing up as a loss spike. ViT-22B and others name it the leading cause of large-model instability.

**QK-Norm** (an RMSNorm on each of q and k before the dot product) makes the logits independent of $W_q$ and $W_k$'s scale; scaling by 8 above does not move them at all. Qwen3, Gemma 3 and OLMo 2 all adopted it. Its own scale becomes the normalisation's gain: Qwen3-0.6B's layer 0 `k_norm` weights reach 96.5 at most (measured in [Quantized deployment](serving://perf/quantization-deploy/#kv-cache-量化误差要和信号比) in the inference-systems handbook), so the gain can grow too, just far more controllably than the product of two matrices.

Two other approaches:

- **Logit soft-capping** (Gemma 2): $\text{logits} \leftarrow c \cdot \tanh(\text{logits} / c)$, with $c = 50$ in attention and 30 in the final output layer. Simple and effective, but it changes attention's computation and kernels like FlashAttention have to support it specifically; Gemma 3 went back to QK-Norm.
- **QK-Clip** (Kimi K2's MuonClip): leave the architecture alone, and after each optimizer update check each head's maximum logit on that batch, multiplying the heads above a threshold $\tau$ by $\sqrt{\tau / S_{\max}}$ on each of $W_q$ and $W_k$:

```python title="qk_clip.py"
import math

import torch

torch.manual_seed(0)
H, d, D, T, tau = 4, 32, 128, 256, 100.0                      # 4 heads; the threshold τ = 100 (Kimi K2's value)
x = torch.randn(T, D)
Wq = torch.randn(H, D, d) / math.sqrt(D)
Wk = torch.randn(H, D, d) / math.sqrt(D)
Wq[1] *= 6                                                    # pretend that heads 1 and 3 have grown their q and k weights over training
Wk[1] *= 5
Wq[3] *= 3
Wk[3] *= 2.5


def max_logits():
    q, k = torch.einsum("td,hde->hte", x, Wq), torch.einsum("td,hde->hte", x, Wk)
    return (q @ k.transpose(1, 2) / math.sqrt(d)).amax(dim=(1, 2))   # each head's maximum logit on this batch


def qk_clip(alpha=0.5):
    """优化器更新之后执行：最大 logit 超过 τ 的头，把 W_q、W_k 分别乘 γ^α、γ^(1-α)，γ = τ / S_max"""
    s = max_logits()
    gamma = (tau / s).clamp(max=1.0)                          # a head below the threshold has γ = 1 and is untouched
    Wq.mul_(gamma[:, None, None] ** alpha)
    Wk.mul_(gamma[:, None, None] ** (1 - alpha))
    return gamma


before = max_logits()
gamma = qk_clip()
after = max_logits()
for h in range(H):
    print(f"头 {h}：最大 logit {before[h]:6.1f} → {after[h]:6.1f}（γ = {gamma[h]:.3f}）")
```

```text title="output"
头 0：最大 logit    5.6 →    5.6（γ = 1.000）
头 1：最大 logit  167.0 →  100.0（γ = 0.599）
头 2：最大 logit    4.2 →    4.2（γ = 1.000）
头 3：最大 logit   36.3 →   36.3（γ = 1.000）
```

Only the heads above the threshold are scaled down and the rest are untouched. Kimi K2's reason for it is practical: K2 uses MLA, where the keys are never computed explicitly per head at inference time (they are absorbed into the query, see [MLA](serving://moe/mla/) in the inference-systems handbook), so a QK-Norm cannot be inserted in the middle; and Muon makes attention's logits grow faster than AdamW does, so something has to hold them down. They trained a 1T-parameter model on 15.5T tokens at $\tau = 100$ with no loss spike at any point. For MLA, the implementation scales only each head's own part of q and k, leaving the shared RoPE key alone.

## The output layer: z-loss {#输出层z-loss}

The output layer's softmax is insensitive to a uniform shift of the logits: add the same number to all of them and the probabilities and the cross-entropy are unchanged. So the logits' overall level ($\log Z$, where $Z = \sum_i e^{\ell_i}$) can drift freely. Drift far enough and bf16 no longer has the precision:

```python title="z_loss.py"
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 32000
base = torch.randn(V) * 3                                     # one position's output logits

print("logits 整体平移后用 bf16 存：")
p = base.softmax(-1)
for shift in (0, 50, 500):
    q = (base + shift).bfloat16().float().softmax(-1)        # a shift does not change the softmax, but bf16's spacing grows with the magnitude
    gap = 2.0 ** (math.floor(math.log2(max(shift, 1))) - 7)  # bf16 has only 7 mantissa bits: the spacing in [2^e, 2^(e+1)) is 2^(e-7)
    print(f"  平移 {shift:>3}：bf16 在这附近的间隔 {gap:5.3f}，概率的总变差距离 {0.5 * (q - p).abs().sum():.3f}")

logits = (base + 20).requires_grad_()                          # suppose log Z has already drifted to around 20
target = torch.tensor([7])
ce = F.cross_entropy(logits[None], target)
log_z = torch.logsumexp(logits, -1)
g_ce, = torch.autograd.grad(ce, logits, retain_graph=True)
g_z, = torch.autograd.grad(1e-4 * log_z ** 2, logits)
print(f"log Z = {log_z:.1f}")
print(f"交叉熵对 logits 的梯度之和 {abs(g_ce.sum()):.4f}：整体平移不改变交叉熵，没有力量把 log Z 拉回来")
print(f"z-loss（1e-4·log²Z）的梯度之和 {g_z.sum():+.1e} = 2·1e-4·log Z：把所有 logits 一起往下推")
```

```text title="output"
logits 整体平移后用 bf16 存：
  平移   0：bf16 在这附近的间隔 0.008，概率的总变差距离 0.006
  平移  50：bf16 在这附近的间隔 0.250，概率的总变差距离 0.030
  平移 500：bf16 在这附近的间隔 2.000，概率的总变差距离 0.232
log Z = 34.8
交叉熵对 logits 的梯度之和 0.0000：整体平移不改变交叉熵，没有力量把 log Z 拉回来
z-loss（1e-4·log²Z）的梯度之和 +7.0e-03 = 2·1e-4·log Z：把所有 logits 一起往下推
```

Shifted to around 500, bf16's spacing is 2 and the differences between the logits are quantised away, leaving the distribution unrecognisable. The cross-entropy's gradients sum to 0, so there is no force pulling $\log Z$ back. PaLM added a **z-loss** of $10^{-4} \log^2 Z$: its gradient is proportional to $\log Z$ and pushes all of the logits toward 0 together, keeping the softmax's denominator near 1. Later models such as OLMo 2 use it too.

## The residual stream: from Hyper-Connections to mHC {#残差流从-hyper-connections-到-mhc}

Hyper-Connections (HC) expand one residual stream into $n$, mixed at each layer by a learnable $n \times n$ matrix, which is more expressive. The problem is that this matrix multiplies in at every layer, and over dozens of layers the signal may be amplified or attenuated without end, which is the same thing as an RNN's exploding gradients. DeepSeek's **mHC** (Manifold-Constrained HC) projects the mixing matrix to be **doubly stochastic** (every row and column summing to 1, entries non-negative) with a Sinkhorn-Knopp iteration:

```python title="hyper_connections.py"
import torch

torch.manual_seed(0)
n, layers = 4, 60                                             # 4 residual streams (DeepSeek-V4's hc_mult), 60 layers


def sinkhorn(M, iters=20):
    """Sinkhorn-Knopp：交替把行和、列和归一化成 1，得到双随机矩阵（mHC 用 20 次迭代）"""
    for _ in range(iters):
        M = M / M.sum(dim=1, keepdim=True)
        M = M / M.sum(dim=0, keepdim=True)
    return M


def product_norm(make):
    P = torch.eye(n)
    norms = []
    for layer in range(1, layers + 1):
        P = make() @ P                                         # the residual streams passing through each layer's mixing matrix
        if layer in (10, 30, 60):
            norms.append(torch.linalg.matrix_norm(P, ord=2).item())
    return norms


# HC: the mixing matrix unconstrained (here the identity plus a small perturbation; a learned matrix will not happen to preserve the norm)
hc = product_norm(lambda: torch.eye(n) + 0.15 * torch.randn(n, n))
# mHC: make it positive first, then project to doubly stochastic with Sinkhorn (rows and columns summing to 1)
mhc = product_norm(lambda: sinkhorn((torch.eye(n) + 0.15 * torch.randn(n, n)).exp()))
print("混合矩阵连乘之后的谱范数     10 层      30 层      60 层")
print("HC（不加约束）          " + "".join(f"{v:10.3g}" for v in hc))
print("mHC（双随机矩阵）        " + "".join(f"{v:10.3g}" for v in mhc))
M = sinkhorn((torch.eye(n) + 0.15 * torch.randn(n, n)).exp())
print(f"一个双随机矩阵：行和 {[round(v, 3) for v in M.sum(1).tolist()]}，谱范数 {torch.linalg.matrix_norm(M, ord=2):.3f}")
```

```text title="output"
混合矩阵连乘之后的谱范数     10 层      30 层      60 层
HC（不加约束）                1.42      6.31      23.2
mHC（双随机矩阵）                 1         1         1
一个双随机矩阵：行和 [1.0, 1.0, 1.0, 1.0]，谱范数 1.000
```

Unconstrained, each layer departs from the identity only slightly and the amplification after 60 layers is already over twenty times; a doubly stochastic matrix has a spectral norm of exactly 1, and the product stays doubly stochastic, so there is no amplification at any depth. DeepSeek-V4 uses 4 residual streams and 20 Sinkhorn iterations (the cost on the inference side is in [The latest open models](serving://frontier/new-models/) in the inference-systems handbook).

## The learning rate, warm-up and gradient clipping {#学习率warmup-与梯度裁剪}

A few standard measures at the optimizer level:

| Measure | Why |
| --- | --- |
| **Warm-up** (the learning rate rising linearly from 0 over the first few thousand steps) | Adam's early second-moment estimate is very inaccurate, so the first update amounts to $\eta \cdot \mathrm{sign}(g)$ with every parameter taking a full step, exactly when a randomly initialised network is most fragile |
| **The warmup-stable-decay schedule** (warm up, hold constant, then decay quickly over the last 10% to 20%) | more flexible than a cosine schedule: the constant phase can be continued at any point or branched into a decayed version, which suits continued pretraining and data-proportion experiments |
| **Gradient clipping** (the global norm clipped to 1.0) | blocks the large gradients from the occasional bad batch; how often the clipping fires is itself a health indicator |
| **Weight decay** of 0.1, but not on the normalisation layers or the biases | controls the weight norms, which are one of the common roots of the problems above, see [Optimizers](optimizer.md) |

The shapes of the various schedules:

<div class="aig-widget" data-widget="lr-schedule"></div>

When a loss spike appears, the common emergency measure (described in the PaLM report) is to roll back to a checkpoint about 100 steps before it and skip the next few hundred batches. That requires checkpoints frequent enough and a data loader that can skip exactly the specified batches, which is why a training framework has to support a deterministic data order. The metrics to watch routinely: each layer's maximum attention logit, the output layer's $\log Z$, the gradient norm, the ratio of the update to the weights, and each layer's activation root mean square.

## FP4 training {#fp4-训练}

FP4 (E2M1) has only 8 non-negative values: 0, 0.5, 1, 1.5, 2, 3, 4 and 6. Training with it is far harder than inferring with it: inference only needs the forward error to be small, while training also needs **the gradients' expectation to be right**, or the error accumulates step by step. NVIDIA's NVFP4 training recipe (reaching the FP8 baseline's accuracy on a 12B model over 10T tokens) has several key measures, two of which an experiment shows directly:

```python title="fp4_training.py"
import torch

torch.manual_seed(0)
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # the non-negative values FP4 E2M1 can represent


def round_e2m1(y, stochastic=False, gen=None):
    """y 已经除过缩放因子（|y| ≤ 6）：取到 E2M1 网格上，就近舍入或随机舍入"""
    a = y.abs().clamp(max=6).contiguous()
    hi = torch.bucketize(a, GRID).clamp(1, 7)                 # a falls between GRID[hi-1] and GRID[hi]
    lo_v, hi_v = GRID[hi - 1], GRID[hi]
    frac = (a - lo_v) / (hi_v - lo_v)
    up = torch.rand(a.shape, generator=gen) < frac if stochastic else frac >= 0.5
    return torch.where(up, hi_v, lo_v) * y.sign()


def e4m3(s):
    return s.to(torch.float8_e4m3fn).float().clamp_min(1e-12)  # the scaling factor itself is stored in FP8 E4M3


def nvfp4(x, stochastic=False, gen=None):
    """NVFP4 伪量化：沿最后一维每 16 个数一个缩放因子"""
    b = x.reshape(-1, 16)
    scale = e4m3(b.abs().amax(1, keepdim=True) / 6)
    return (round_e2m1(b / scale, stochastic, gen) * scale).reshape(x.shape)


def nvfp4_2d(w):
    """16×16 的二维块共用一个缩放因子"""
    R, C = w.shape
    b = w.reshape(R // 16, 16, C // 16, 16)
    scale = e4m3(b.abs().amax(dim=(1, 3), keepdim=True) / 6)
    return (round_e2m1(b / scale) * scale).reshape(R, C)


# 1. the small components of a gradient: one 6.0 in every 16 values (which sets the scaling factor) and 0.1 for the rest
g = torch.full((1024,), 0.1)
g[::16] = 6.0
gen = torch.Generator().manual_seed(1)
rtn = torch.stack([nvfp4(g) for _ in range(1000)]).mean(0)
sr = torch.stack([nvfp4(g, True, gen) for _ in range(1000)]).mean(0)
small = torch.ones(1024, dtype=torch.bool)
small[::16] = False
print(f"小分量的真实值 0.100；量化 1000 次再平均：就近舍入 {rtn[small].mean():.3f}，随机舍入 {sr[small].mean():.3f}")

# 2. the weights' blocking direction: the forward Y = X Wᵀ blocks along W's rows (the input dimension), while the backward dX = dY W has to block along W's columns
w = torch.randn(256, 512)
fwd, bwd = nvfp4(w), nvfp4(w.T).T                             # the W used in the forward pass against the W used in the backward (blocked by column and transposed back for the comparison)
print(f"一维分块（1×16）：前向和反向用的权重有 {(fwd != bwd).float().mean():.0%} 的元素不同")
print(f"二维分块（16×16）：有 {(nvfp4_2d(w) != nvfp4_2d(w.T).T).float().mean():.0%} 的元素不同")
```

```text title="output"
小分量的真实值 0.100；量化 1000 次再平均：就近舍入 0.000，随机舍入 0.100
一维分块（1×16）：前向和反向用的权重有 76% 的元素不同
二维分块（16×16）：有 0% 的元素不同
```

- **Stochastic rounding for the gradients**: the large value in a block sets the scaling factor, so a small component (0.1) falls between 0 and the first grid point, and round-to-nearest **always rounds it to 0**, which stays 0 however many steps accumulate: that part of the gradient's signal is systematically thrown away. Stochastic rounding picks between the two neighbouring grid points at random in proportion to the distances, which has a larger single-shot error but **the right expectation**, recovering 0.1 on average. The forward pass's weights and activations still use round-to-nearest (what they need is a small single-shot error).
- **16x16 two-dimensional blocks for the weights**: the forward pass blocks along the weights' rows (the input dimension), while computing the input's gradient in the backward pass blocks along the columns. With one-dimensional blocks the two quantisations produce two different matrices, so the backward pass is not computing the gradient of the function the forward pass computed; a two-dimensional block shares one scaling factor and the quantisation is identical after a transpose.
- The rest of the recipe: a 16x16 **random Hadamard transform** on the inputs of the weight-gradient matrix multiply alone, spreading the outliers within a block (we tried it on the forward matrix multiply in a small experiment and it gave no benefit, which is why it is used only for the weight gradients); bf16 kept for the last few layers and the other precision-sensitive ones; and an fp32 global scaling factor per tensor, so that the block scaling factors land in FP8 E4M3's range.

!!! interview "How to answer in an interview"
    Asked how you would investigate a loss spike in large-model training: start with **the symptoms** (does it recover by itself, is it tied to a particular batch, at what stage of training did it appear), then **the metrics** (each layer's maximum attention logit, the output layer's $\log Z$, the gradient norm and how often clipping fires, the update-to-weight ratio, the activation root mean square, and which of them rose first), then the matching **causes and remedies** (growing attention logits: QK-Norm, soft-capping, QK-Clip; a drifting output layer: z-loss; an amplifying residual stream: a constraint like mHC, a sensible initialisation; low precision: fine-grained scaling, stochastic rounding, high precision on the sensitive layers; the optimizer: warm-up, gradient clipping, a lower learning rate, adjusting Adam's $\beta_2$ and $\epsilon$), and finally **the emergency measure** (roll back to a checkpoint before the spike and skip the data). Citing Kimi K2's QK-Clip (MLA cannot use QK-Norm) and PaLM's rolling back and skipping data shows you know the real engineering practice.

## Exercises {#练习}

**1. Why does QK-Clip scale per head rather than the whole layer at once? And why is $\gamma$ split evenly between $W_q$ and $W_k$?**

??? success "Answer"
    The heads' logit scales differ greatly: only one head exceeded the threshold in the experiment above, and scaling the whole layer would shrink the healthy heads too and change their attention distributions. Splitting $\gamma$ into $\gamma^{\alpha}$ and $\gamma^{1-\alpha}$ ($\alpha = 0.5$) on the two sides shrinks the logits by exactly $\gamma$ while keeping the two matrices' scales balanced, so neither keeps shrinking. For MLA, half of the RoPE part of q and k is a key shared by every head and cannot be scaled per head, so only each head's own part is.

**2. A model uses soft-capping ($c = 30$) on its final layer. What does an inference engine have to do about it?**

??? success "Answer"
    The output layer's soft-capping has to apply $30 \tanh(\ell / 30)$ to the logits before sampling, with the temperature and top-p after it; it changes the logits' relative sizes, so missing it makes the distribution differ from training's (which also affects speculative decoding's verification and the returned log probabilities). Soft-capping inside attention needs kernel support (FlashAttention and FlashInfer both have a parameter for it), without which you fall back to the slow implementation. This is one of the differences to look for in the config when bringing up a new model (Gemma 2's `attn_logit_softcapping` and `final_logit_softcapping`).

**3. Why is z-loss's coefficient only $10^{-4}$? What happens if it is larger?**

??? success "Answer"
    z-loss only has to provide a weak pull bringing $\log Z$ back toward 0, and cross-entropy's gradient in that direction is 0, so a very small coefficient is enough to decide which way it goes while barely affecting the main objective. Too large a coefficient makes it compete with the cross-entropy for the logits' scale: to bring $\log Z$ near 0 the model lowers the correct class's logit, reducing the prediction's confidence and worsening the main task's loss.

## Summary {#小结}

- [x] Attention's logits grow with the product of $W_q$'s and $W_k$'s norms, causing entropy collapse and loss spikes; QK-Norm makes the logits independent of the weights' scale, soft-capping clips them directly, and QK-Clip scales the offending heads' weights after each optimizer update (MLA cannot use QK-Norm).
- [x] The output layer's $\log Z$ is unconstrained by the cross-entropy and drifts beyond bf16's precision; z-loss ($10^{-4} \log^2 Z$) pulls it back.
- [x] Multiplying several residual streams' mixing matrices together amplifies the signal; mHC projects them to doubly stochastic with Sinkhorn, giving a spectral norm of exactly 1.
- [x] Warm-up, the warmup-stable-decay schedule, gradient clipping and weight decay are standard; after a spike, roll back the checkpoint and skip the data.
- [x] FP4 training: stochastic rounding for the gradients (an unbiased expectation), 16x16 two-dimensional blocks for the weights (consistent forward and backward), a random Hadamard transform on the weight gradients' inputs, and bf16 for the sensitive layers.
