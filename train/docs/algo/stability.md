# 训练稳定性：loss 突刺、QK-Norm、QK-Clip 与低精度

<p class="lead">小模型训练很少出问题，大模型却经常在跑了几周之后突然出现 loss 突刺（spike），甚至发散，只能回滚到之前的 checkpoint、跳过一批数据、降低学习率再试。根因大多能追到几个"数值在长大"的地方：注意力的 logits、输出层的 logits、残差流，以及低精度下被舍掉的小量。这一章逐个用小实验把它们跑出来，再看 Gemma、PaLM、Kimi K2、DeepSeek 等模型各自用了什么办法。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 注意力的 logits 为什么会越训越大？变大之后会发生什么？
    2. QK-Norm、logit soft-capping、QK-Clip 分别怎样控制注意力 logits？为什么 Kimi K2 不用 QK-Norm？
    3. z-loss 惩罚的是什么？交叉熵本身为什么管不住它？
    4. Hyper-Connections 把残差流变成了多条，mHC 为什么要把混合矩阵约束成双随机矩阵？
    5. FP4 训练里，为什么梯度要用随机舍入、权重要用二维分块缩放？

## 注意力 logits 的增长与熵坍缩

注意力的 logits 是 $q \cdot k / \sqrt{d}$，$q = W_q x$、$k = W_k x$。训练中 $W_q$、$W_k$ 的范数往往一起慢慢变大，logits 按两者的**乘积**增长。把这个过程压缩成一个实验，看注意力分布怎样变化：

```python title="logit_growth.py"
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, d = 512, 64
x = torch.randn(T, 256)                                       # 某一层注意力的输入（已经过归一化）
Wq, Wk = torch.randn(256, d) / 16, torch.randn(256, d) / 16   # 初始化时 q、k 的每个分量约为 N(0, 1)


def attention_stats(scale, qk_norm):
    q, k = x @ (Wq * scale), x @ (Wk * scale)                  # 训练中 W_q、W_k 的范数一起变大
    if qk_norm:                                               # QK-Norm：点积之前，q、k 各做一次 RMSNorm
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

```text title="输出"
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

权重只放大到 4 倍，logits 就大了 16 倍，注意力的熵从接近均匀分布的 5.4 掉到 0.3：每个 query 几乎只看一个 key（注意力权重 0.88）。这就是**注意力熵坍缩**：softmax 进入饱和区，对大多数位置的梯度接近 0，对少数位置的梯度又很大，训练在这里变得很脆弱，常常表现为 loss 突刺。ViT-22B 等工作把它列为大模型不稳定的首要原因。

**QK-Norm**（在点积之前对 q、k 各做一次 RMSNorm）让 logits 与 $W_q$、$W_k$ 的尺度无关——上面放大 8 倍也纹丝不动。Qwen3、Gemma 3、OLMo 2 等都采用了它。它的"刻度"变成了归一化层的增益：Qwen3-0.6B 第 0 层 `k_norm` 的权重最大到 96.5（推理系统手册的[量化部署](serving://perf/quantization-deploy/#kv-cache-量化误差要和信号比)一章测过），增益本身也可以长大，只是比两个矩阵的乘积可控得多。

另外两种办法：

- **logit soft-capping**（Gemma 2）：$\text{logits} \leftarrow c \cdot \tanh(\text{logits} / c)$，注意力取 $c = 50$，最后的输出层取 30。简单有效，但它改变了注意力的计算，FlashAttention 这类 kernel 要专门支持；Gemma 3 又换回了 QK-Norm；
- **QK-Clip**（Kimi K2 的 MuonClip）：不改模型结构，在每次优化器更新之后检查每个头在这批数据上的最大 logit，超过阈值 $\tau$ 的头，把它的 $W_q$、$W_k$ 各乘上 $\sqrt{\tau / S_{\max}}$：

```python title="qk_clip.py"
import math

import torch

torch.manual_seed(0)
H, d, D, T, tau = 4, 32, 128, 256, 100.0                      # 4 个头；阈值 τ = 100（Kimi K2 的取值）
x = torch.randn(T, D)
Wq = torch.randn(H, D, d) / math.sqrt(D)
Wk = torch.randn(H, D, d) / math.sqrt(D)
Wq[1] *= 6                                                    # 假装训练中第 1、3 个头的 q、k 权重长大了
Wk[1] *= 5
Wq[3] *= 3
Wk[3] *= 2.5


def max_logits():
    q, k = torch.einsum("td,hde->hte", x, Wq), torch.einsum("td,hde->hte", x, Wk)
    return (q @ k.transpose(1, 2) / math.sqrt(d)).amax(dim=(1, 2))   # 每个头在这批数据上的最大 logit


def qk_clip(alpha=0.5):
    """优化器更新之后执行：最大 logit 超过 τ 的头，把 W_q、W_k 分别乘 γ^α、γ^(1-α)，γ = τ / S_max"""
    s = max_logits()
    gamma = (tau / s).clamp(max=1.0)                          # 没超过阈值的头 γ = 1，不受影响
    Wq.mul_(gamma[:, None, None] ** alpha)
    Wk.mul_(gamma[:, None, None] ** (1 - alpha))
    return gamma


before = max_logits()
gamma = qk_clip()
after = max_logits()
for h in range(H):
    print(f"头 {h}：最大 logit {before[h]:6.1f} → {after[h]:6.1f}（γ = {gamma[h]:.3f}）")
```

```text title="输出"
头 0：最大 logit    5.6 →    5.6（γ = 1.000）
头 1：最大 logit  167.0 →  100.0（γ = 0.599）
头 2：最大 logit    4.2 →    4.2（γ = 1.000）
头 3：最大 logit   36.3 →   36.3（γ = 1.000）
```

只有超过阈值的头被缩小，其余的头完全不受影响。Kimi K2 用它的原因很实际：K2 用的是 MLA，推理时 key 不会按头显式算出来（吸收进了 query，见推理系统手册的 [MLA](serving://moe/mla/)），没法在中间插一个 QK-Norm；而 Muon 让注意力 logits 增长得比 AdamW 更快，必须有一个办法按住它。他们用 $\tau = 100$ 在 15.5T token 上训练 1T 参数的模型，全程没有出现 loss 突刺。对 MLA，实际实现只缩放每个头自己的那部分 q、k，共享的 RoPE key 不动。

## 输出层：z-loss

输出层的 softmax 对 logits 的整体平移不敏感：所有 logits 加同一个数，概率不变，交叉熵也不变。于是 logits 的整体水平（$\log Z$，$Z = \sum_i e^{\ell_i}$）可以随意漂移。漂得太远时，bf16 的精度就不够用了：

```python title="z_loss.py"
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 32000
base = torch.randn(V) * 3                                     # 某个位置的输出 logits

print("logits 整体平移后用 bf16 存：")
p = base.softmax(-1)
for shift in (0, 50, 500):
    q = (base + shift).bfloat16().float().softmax(-1)        # 平移不改变 softmax，但 bf16 的间隔随数值变大
    gap = 2.0 ** (math.floor(math.log2(max(shift, 1))) - 7)  # bf16 只有 7 位尾数：[2^e, 2^(e+1)) 里的间隔是 2^(e-7)
    print(f"  平移 {shift:>3}：bf16 在这附近的间隔 {gap:5.3f}，概率的总变差距离 {0.5 * (q - p).abs().sum():.3f}")

logits = (base + 20).requires_grad_()                          # 假设 log Z 已经漂到了 20 附近
target = torch.tensor([7])
ce = F.cross_entropy(logits[None], target)
log_z = torch.logsumexp(logits, -1)
g_ce, = torch.autograd.grad(ce, logits, retain_graph=True)
g_z, = torch.autograd.grad(1e-4 * log_z ** 2, logits)
print(f"log Z = {log_z:.1f}")
print(f"交叉熵对 logits 的梯度之和 {abs(g_ce.sum()):.4f}：整体平移不改变交叉熵，没有力量把 log Z 拉回来")
print(f"z-loss（1e-4·log²Z）的梯度之和 {g_z.sum():+.1e} = 2·1e-4·log Z：把所有 logits 一起往下推")
```

```text title="输出"
logits 整体平移后用 bf16 存：
  平移   0：bf16 在这附近的间隔 0.008，概率的总变差距离 0.006
  平移  50：bf16 在这附近的间隔 0.250，概率的总变差距离 0.030
  平移 500：bf16 在这附近的间隔 2.000，概率的总变差距离 0.232
log Z = 34.8
交叉熵对 logits 的梯度之和 0.0000：整体平移不改变交叉熵，没有力量把 log Z 拉回来
z-loss（1e-4·log²Z）的梯度之和 +7.0e-03 = 2·1e-4·log Z：把所有 logits 一起往下推
```

logits 平移到 500 附近时，bf16 的间隔是 2，logits 之间的差别被量化掉，概率分布面目全非。交叉熵的梯度加起来是 0，没有任何力量把 $\log Z$ 拉回来。PaLM 加了一项 **z-loss** $10^{-4} \log^2 Z$：它的梯度正比于 $\log Z$，把所有 logits 一起往 0 附近推，让 softmax 的分母保持在 1 左右。OLMo 2 等后来的模型也用它。

## 残差流：从 Hyper-Connections 到 mHC

Hyper-Connections（HC）把一条残差流扩成 $n$ 条，每层用一个可学习的 $n \times n$ 矩阵混合它们，表达能力更强。问题在于这个矩阵每层都乘一次，几十层乘下来，信号可能被不断放大或衰减——和 RNN 的梯度爆炸是一回事。DeepSeek 的 **mHC**（Manifold-Constrained HC）把混合矩阵投影成**双随机矩阵**（每行、每列的和都是 1，元素非负），用 Sinkhorn-Knopp 迭代实现：

```python title="hyper_connections.py"
import torch

torch.manual_seed(0)
n, layers = 4, 60                                             # 4 条残差流（DeepSeek-V4 的 hc_mult），60 层


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
        P = make() @ P                                         # 残差流经过每一层的混合矩阵
        if layer in (10, 30, 60):
            norms.append(torch.linalg.matrix_norm(P, ord=2).item())
    return norms


# HC：混合矩阵不加约束（这里是单位阵加一点扰动，训练中学出来的矩阵不会恰好保持范数）
hc = product_norm(lambda: torch.eye(n) + 0.15 * torch.randn(n, n))
# mHC：先变成正矩阵，再用 Sinkhorn 投影成双随机矩阵（行和、列和都是 1）
mhc = product_norm(lambda: sinkhorn((torch.eye(n) + 0.15 * torch.randn(n, n)).exp()))
print("混合矩阵连乘之后的谱范数     10 层      30 层      60 层")
print("HC（不加约束）          " + "".join(f"{v:10.3g}" for v in hc))
print("mHC（双随机矩阵）        " + "".join(f"{v:10.3g}" for v in mhc))
M = sinkhorn((torch.eye(n) + 0.15 * torch.randn(n, n)).exp())
print(f"一个双随机矩阵：行和 {[round(v, 3) for v in M.sum(1).tolist()]}，谱范数 {torch.linalg.matrix_norm(M, ord=2):.3f}")
```

```text title="输出"
混合矩阵连乘之后的谱范数     10 层      30 层      60 层
HC（不加约束）                1.42      6.31      23.2
mHC（双随机矩阵）                 1         1         1
一个双随机矩阵：行和 [1.0, 1.0, 1.0, 1.0]，谱范数 1.000
```

不加约束时，每层只偏离单位阵一点点，60 层连乘之后的放大倍数也到了二十多倍；双随机矩阵的谱范数恰好是 1，而且乘积仍然是双随机矩阵，无论多少层都不会放大。DeepSeek-V4 用 4 条残差流、20 次 Sinkhorn 迭代（推理侧的代价见推理系统手册的[新一代开源模型](serving://frontier/new-models/)）。

## 学习率、warmup 与梯度裁剪

优化器层面的几个标配：

| 做法 | 为什么 |
| --- | --- |
| **warmup**（前几千步学习率从 0 线性增加） | Adam 早期的二阶矩估计很不准，第一步的更新相当于 $\eta \cdot \mathrm{sign}(g)$，每个参数都走满一步；随机初始化的网络这时最脆弱 |
| **WSD 调度**（warmup → 恒定 → 最后 10%～20% 快速衰减） | 比余弦调度更灵活：恒定阶段可以随时接着训练或分叉出衰减版本，便于做持续预训练和数据配比实验 |
| **梯度裁剪**（全局范数裁到 1.0） | 挡住个别坏 batch 造成的大梯度；裁剪发生的频率本身就是一个健康指标 |
| **权重衰减** 0.1，但不衰减归一化层和偏置 | 控制权重范数（上面几个问题的共同根源之一），见[优化器](optimizer.md) |

出现 loss 突刺时，常见的应急做法（PaLM 报告里描述过）是回滚到突刺之前约 100 步的 checkpoint，跳过接下来的几百个 batch 再继续。能这样做的前提是 checkpoint 足够频繁，并且数据加载器能精确地跳过指定的 batch——这也是训练框架要支持"确定性的数据顺序"的原因。平时要监控的指标：每层注意力的最大 logit、输出层的 $\log Z$、梯度范数、更新量与权重的比值（update/weight ratio）、各层激活的均方根。

## FP4 训练

FP4（E2M1）只有 8 个非负取值：0、0.5、1、1.5、2、3、4、6。用它训练比用它推理难得多：推理只需要前向的误差小，训练还要求**梯度的期望是对的**，否则误差会一步步累积。NVIDIA 的 NVFP4 训练配方（在 12B 模型、10T token 上达到了 FP8 基线的精度）有几个关键做法，其中两个可以直接用实验看出来：

```python title="fp4_training.py"
import torch

torch.manual_seed(0)
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # FP4 E2M1 能表示的非负值


def round_e2m1(y, stochastic=False, gen=None):
    """y 已经除过缩放因子（|y| ≤ 6）：取到 E2M1 网格上，就近舍入或随机舍入"""
    a = y.abs().clamp(max=6).contiguous()
    hi = torch.bucketize(a, GRID).clamp(1, 7)                 # a 落在 GRID[hi-1] 和 GRID[hi] 之间
    lo_v, hi_v = GRID[hi - 1], GRID[hi]
    frac = (a - lo_v) / (hi_v - lo_v)
    up = torch.rand(a.shape, generator=gen) < frac if stochastic else frac >= 0.5
    return torch.where(up, hi_v, lo_v) * y.sign()


def e4m3(s):
    return s.to(torch.float8_e4m3fn).float().clamp_min(1e-12)  # 缩放因子本身用 FP8 E4M3 存


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


# ① 梯度里的小分量：每 16 个数里有一个 6.0（决定了缩放因子），其余是 0.1
g = torch.full((1024,), 0.1)
g[::16] = 6.0
gen = torch.Generator().manual_seed(1)
rtn = torch.stack([nvfp4(g) for _ in range(1000)]).mean(0)
sr = torch.stack([nvfp4(g, True, gen) for _ in range(1000)]).mean(0)
small = torch.ones(1024, dtype=torch.bool)
small[::16] = False
print(f"小分量的真实值 0.100；量化 1000 次再平均：就近舍入 {rtn[small].mean():.3f}，随机舍入 {sr[small].mean():.3f}")

# ② 权重的分块方向：前向 Y = X Wᵀ 沿 W 的行（输入维）分块；反向 dX = dY W 要沿 W 的列分块
w = torch.randn(256, 512)
fwd, bwd = nvfp4(w), nvfp4(w.T).T                             # 前向用的 W，反向用的 W（按列分块后转置回来比较）
print(f"一维分块（1×16）：前向和反向用的权重有 {(fwd != bwd).float().mean():.0%} 的元素不同")
print(f"二维分块（16×16）：有 {(nvfp4_2d(w) != nvfp4_2d(w.T).T).float().mean():.0%} 的元素不同")
```

```text title="输出"
小分量的真实值 0.100；量化 1000 次再平均：就近舍入 0.000，随机舍入 0.100
一维分块（1×16）：前向和反向用的权重有 76% 的元素不同
二维分块（16×16）：有 0% 的元素不同
```

- **梯度用随机舍入**：一个块里的大值决定了缩放因子，小分量（0.1）落在 0 和第一个格点之间，就近舍入**每次都舍成 0**，累积多少步都是 0——梯度里的这部分信号被系统性地丢掉了。随机舍入按距离的比例随机取上下两个格点，单次误差更大，但**期望是对的**，平均下来恢复了 0.1。前向的权重和激活仍然用就近舍入（它们要的是单次误差小）；
- **权重用 16×16 的二维块**：前向沿权重的行（输入维）分块，反向算输入梯度时要沿列分块。一维分块时两次量化出来的是两个不同的矩阵，反向传播算的就不是前向那个函数的梯度；二维块共用一个缩放因子，转置之后量化结果完全一致；
- 其余做法：只对权重梯度那次矩阵乘的输入做 16×16 的**随机 Hadamard 变换**，把块内的离群值摊开（我们在小实验里试过，对前向的矩阵乘并没有好处，这也是它只用在权重梯度上的原因）；最后几层等少数对精度敏感的层保留 bf16；每个张量再乘一个 fp32 的全局缩放因子，让块的缩放因子落在 FP8 E4M3 的范围里。

!!! interview "面试怎么答"
    被问"大模型训练出现 loss 突刺，你怎么排查"：先分**现象**（突刺后能否自己恢复、是否和特定数据 batch 相关、出现在训练的哪个阶段）→ 再看**指标**（各层注意力的最大 logit、输出层 $\log Z$、梯度范数和裁剪频率、更新量与权重之比、激活均方根，哪个在突刺前先涨）→ 对应**根因与办法**（注意力 logits 增长：QK-Norm / soft-capping / QK-Clip；输出层漂移：z-loss；残差流放大：mHC 这类约束、合理的初始化；低精度：细粒度缩放、随机舍入、敏感层保留高精度；优化器：warmup、梯度裁剪、降低学习率、调整 Adam 的 $\beta_2$ 和 $\epsilon$）→ **应急**（回滚到突刺前的 checkpoint、跳过数据）。能举出 Kimi K2 的 QK-Clip（MLA 不能用 QK-Norm）和 PaLM 的回滚跳数据，说明你了解真实的工程做法。

## 练习

**1. QK-Clip 为什么要按头缩放，而不是整层一起缩放？为什么把 $\gamma$ 平均分给 $W_q$ 和 $W_k$？**

??? success "参考答案"
    每个头的 logits 规模差别很大：上面的实验里只有一个头超过了阈值，整层一起缩放会把正常的头也压小，改变它们的注意力分布。把 $\gamma$ 拆成 $\gamma^{\alpha}$ 和 $\gamma^{1-\alpha}$（$\alpha = 0.5$）乘到两边，logits 恰好缩小 $\gamma$ 倍，同时两个矩阵的尺度保持平衡，不会让其中一个越来越小。对 MLA，q、k 里带 RoPE 的部分有一半是所有头共享的 key，不能按头缩放，所以只缩放每个头自己的那部分。

**2. 一个模型的最后一层用了 soft-capping（$c = 30$）。推理引擎需要为它做什么？**

??? success "参考答案"
    输出层的 soft-capping 要在采样之前对 logits 做 $30 \tanh(\ell / 30)$，温度、top-p 都在它之后；它会改变 logits 的相对大小，漏掉就和训练时的分布不一致（投机解码的验证、logprobs 的返回也都受影响）。注意力里的 soft-capping 要注意力 kernel 支持（FlashAttention、FlashInfer 都有相应的参数），否则只能退回慢的实现。这也是新模型接入时要在 config 里找的差异之一（Gemma 2 的 `attn_logit_softcapping`、`final_logit_softcapping`）。

**3. 为什么 z-loss 的系数只有 $10^{-4}$？太大会怎样？**

??? success "参考答案"
    z-loss 只需要提供一个把 $\log Z$ 拉回 0 附近的弱约束，交叉熵本身在这个方向上的梯度为 0，所以很小的系数就足够决定这个方向的走向，又几乎不影响主目标。系数太大时，它会和交叉熵争夺 logits 的尺度：为了让 $\log Z$ 接近 0，模型会压低正确类别的 logit，降低预测的置信度，主任务的 loss 变差。

## 小结

- [x] 注意力 logits 随 $W_q$、$W_k$ 的范数乘积增长，导致熵坍缩与 loss 突刺；QK-Norm 让 logits 与权重尺度无关，soft-capping 直接截断，QK-Clip 在优化器更新之后按头缩放超标的权重（MLA 不能用 QK-Norm）。
- [x] 输出层的 $\log Z$ 不受交叉熵约束，会漂移到 bf16 精度不够的范围；z-loss（$10^{-4} \log^2 Z$）把它拉回来。
- [x] 多条残差流的混合矩阵连乘会放大信号；mHC 用 Sinkhorn 投影成双随机矩阵，谱范数恒为 1。
- [x] warmup、WSD 调度、梯度裁剪、权重衰减是标配；突刺后回滚 checkpoint、跳过数据。
- [x] FP4 训练：梯度随机舍入（期望无偏）、权重 16×16 二维分块（前向反向一致）、权重梯度的输入做随机 Hadamard 变换、敏感层保留 bf16。
