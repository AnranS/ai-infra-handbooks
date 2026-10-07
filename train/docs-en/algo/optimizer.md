# Optimizers: AdamW, Muon and the distributed optimizer

<p class="lead">The optimizer decides two things: how fast training converges, and the largest item in the memory budget, since 12 of the 16 bytes per parameter belong to the optimizer (the fp32 master weights and the two moments). This chapter first sets out every detail of AdamW and its budget, then turns to Muon, the newer optimizer that Kimi, GLM and others have used at trillion-parameter scale since 2025: its heart is one matrix orthogonalisation, which happens to conflict with the way ZeRO and FSDP partition the optimizer states elementwise. It ends with a multi-process experiment showing how distributed Muon resolves that conflict.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does AdamW differ from Adam with L2 regularisation? Why do large models all use AdamW?
    2. In mixed-precision training, what are each parameter's 16 bytes? How much does Muon save?
    3. What does Muon do to the gradient? Why a Newton-Schulz iteration rather than an SVD?
    4. Why is Muon used only on the hidden layers' two-dimensional matrices?
    5. ZeRO partitions the optimizer states elementwise, so why does that conflict with Muon? What are the ways around it, and what does each cost?

??? success "Answers for the self-test (answer first, then open this)"
    1. Adam with L2 adds $\lambda w$ to the gradient and then divides it all by $\sqrt{v}$: the decay is weakened for parameters with large gradients and amplified for those with small ones, so the regularisation's strength is distorted by the gradients' magnitude. AdamW takes the decay out of the gradient and applies $w \leftarrow w - \eta \lambda w$ to the parameters directly, treating every parameter alike, which works better and is easier to tune.
    2. 2 bytes of bf16 parameters, 2 of bf16 gradients, 4 of fp32 master weights, 4 for Adam's first moment and 4 for the second. Muon stores only one momentum, 12 bytes per parameter, saving the 4 of the second moment.
    3. It orthogonalises the momentum matrix: approximating it as $UV^\top$ (every singular value set to 1), so that the update's magnitude is the same in every direction, then scaling by $0.2\sqrt{\max(A,B)}$. The Newton-Schulz iteration uses only matrix multiplies, converges in a few steps and runs efficiently in bf16 on a GPU; an SVD is both slow and hard to parallelise.
    4. Orthogonalisation only means anything for a two-dimensional matrix. The embedding and the output layer are a lookup and a scoring whose row and column semantics differ from a hidden layer's linear transform, and one-dimensional parameters (normalisation weights, biases) cannot be orthogonalised at all; those still use AdamW.
    5. Orthogonalisation needs the complete matrix, while ZeRO and FSDP cut each matrix elementwise across cards. The options: all-gather the complete matrix before orthogonalising (one extra collective and duplicated computation); give different matrices to different ranks to compute in full and distribute afterwards (the load has to be balanced); or do an approximate orthogonalisation within the tensor-parallel shard (at a cost in accuracy).

## AdamW: decoupled weight decay {#adamw解耦的权重衰减}

Adam keeps the gradient's first moment $m$ (the momentum) and second moment $v$ per parameter, and the update is $m / (\sqrt{v} + \epsilon)$: each coordinate is normalised by its own gradient's magnitude, which makes it insensitive to the gradient's scale. Weight decay can be added in two ways:

- **Adam with L2 regularisation**: add $\lambda w$ to the gradient, so it gets normalised by $\sqrt{v}$ along with it.
- **AdamW** (Loshchilov and Hutter): the decay bypasses the normalisation and applies $w \leftarrow w - \eta \lambda w$ directly.

The difference is in one place and the consequences are large. Give two parameters gradients a thousand times apart and look at the decay alone:

Dial it and see the result first (the script below is this experiment):

<div class="aig-widget" data-widget="adamw-step"></div>

```python title="adamw_decay.py"
import torch

lr, wd, steps = 1e-2, 0.1, 1000
scale = torch.tensor([0.01, 10.0])                     # two parameters: one with a very small gradient and one with a very large one


def train(decoupled):
    w = torch.ones(2, requires_grad=True)
    if decoupled:
        opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)            # the decay acts on the weights directly
    else:
        opt = torch.optim.Adam([w], lr=lr, weight_decay=wd)             # the decay is added to the gradient (L2 regularisation) and then normalised by Adam
    for t in range(steps):
        opt.zero_grad()
        w.grad = scale * (-1) ** t                                      # a gradient alternating in sign and averaging 0: it only gives Adam's v a magnitude
        opt.step()
    return w.detach()


for name, decoupled in [("Adam + L2 正则", False), ("AdamW（解耦的权重衰减）", True)]:
    w = train(decoupled)
    print(f"{name:16s} 小梯度的参数 {w[0]:.3f}，大梯度的参数 {w[1]:.3f}")
print(f"只有衰减时的理论值：(1 - lr·wd)^{steps} = {(1 - lr * wd) ** steps:.3f}")
```

```text title="output"
Adam + L2 正则     小梯度的参数 0.000，大梯度的参数 0.889
AdamW（解耦的权重衰减）   小梯度的参数 0.361，大梯度的参数 0.361
只有衰减时的理论值：(1 - lr·wd)^1000 = 0.368
```

Under Adam with L2, the regularisation term is divided by $\sqrt{v}$: a parameter with a small gradient is over-decayed (going to zero quickly) and one with a large gradient is barely decayed, so the regularisation's strength depends on the gradient's magnitude, which is not what we want. AdamW treats every parameter alike and the decay matches the theoretical value. So large-model training all uses AdamW, commonly with `weight_decay` at 0.1, and usually **without** decay on the normalisation layers' gains and the biases.

## The optimizer's memory budget {#优化器的显存账本}

What each parameter costs in mixed-precision training (see [Mixed precision](../practice/mixed-precision.md) and [The memory budget](../basics/overview.md)):

| Item | AdamW | Muon (a hidden-layer matrix) |
| --- | --- | --- |
| bf16 weights | 2 | 2 |
| bf16 gradients | 2 | 2 |
| fp32 master weights | 4 | 4 |
| Optimizer states | 4 bytes each for $m$ and $v$ = 8 | 4 for the momentum |
| **Total** | **16** | **12** |

Measuring the two optimizers' stored state in PyTorch (Muon's implementation is in the next section):

```python title="optimizer_state.py"
import torch
from muon import Muon


def state_bytes(opt):
    return sum(t.numel() * t.element_size() for s in opt.state.values() for t in s.values()
               if torch.is_tensor(t) and t.dim() > 0)          # scalars such as the step count are not counted


for name in ("AdamW", "Muon"):
    torch.manual_seed(0)
    w1, w2 = torch.randn(4096, 1024, requires_grad=True), torch.randn(1024, 4096, requires_grad=True)
    opt = torch.optim.AdamW([w1, w2], lr=1e-3) if name == "AdamW" else Muon([w1, w2], lr=1e-3)
    (torch.randn(8, 1024) @ w1.T @ w2.T).sum().backward()
    opt.step()
    n = w1.numel() + w2.numel()
    print(f"{name:5s}：{n / 1e6:.1f}M 个参数，优化器状态 {state_bytes(opt) / n:.0f} 字节/参数")
```

```text title="output"
AdamW：8.4M 个参数，优化器状态 8 字节/参数
Muon ：8.4M 个参数，优化器状态 4 字节/参数
```

For a 70B model those 16 bytes alone are 1.1 TB, far beyond one card's memory, which is why [ZeRO](../data/zero-fsdp.md) partitions the optimizer states across the data-parallel ranks. Muon stores one moment fewer and saves a quarter of the model states' memory.

## Muon: orthogonalising the update {#muon把更新正交化}

Muon (MomentUm Orthogonalized by Newton-Schulz, Keller Jordan 2024) updates each hidden-layer weight matrix like this:

1. Keep a momentum as SGD does, $M \leftarrow \mu M + G$.
2. **Orthogonalise** the (Nesterov-style) momentum matrix $G + \mu M = U S V^\top$ into $U V^\top$: keep the directions and set every singular value to 1.
3. $W \leftarrow W - \eta \cdot 0.2\sqrt{\max(A, B)} \cdot U V^\top$, where $A \times B$ is the matrix's shape.

Why orthogonalise? A Transformer weight's gradient and momentum tend to be dominated by a few directions (singular values orders of magnitude apart), and updating by it directly leaves the rare but useful directions barely updated. Orthogonalisation makes every singular direction take the same size of step, which amounts to steepest descent under the spectral norm. An SVD is slow and hard to parallelise on a GPU, so Muon approximates it with a **Newton-Schulz iteration** using only matrix multiplies, stable even in bf16:

Dial the iterations and see what orthogonalisation does to the singular values:

<div class="aig-widget" data-widget="newton-schulz"></div>

```python title="muon.py"
"""Muon：动量 + Newton-Schulz 正交化，只用于隐藏层的二维权重矩阵"""
import math

import torch


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """近似 G 的极分解因子 U Vᵀ（G = U S Vᵀ）：把所有奇异值推到 1 附近，只用矩阵乘"""
    a, b, c = 3.4445, -4.7750, 2.0315                       # the quintic polynomial's coefficients: fast convergence, with the singular values landing in a band around 1
    X = G / (G.norm() + 1e-7)                               # scale to a spectral norm of at most 1 first
    transposed = G.shape[0] > G.shape[1]
    if transposed:                                          # make X Xᵀ the smaller of the two square matrices
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X                 # X ← a·X + b·(X Xᵀ)X + c·(X Xᵀ)²X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """每个矩阵：m ← μ·m + g；更新 = NS(g + μ·m)（Nesterov），再乘 0.2·√max(行, 列)，让更新的均方根和 AdamW 相当"""

    def __init__(self, params, lr, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                m = self.state[p].setdefault("momentum", torch.zeros_like(p))
                m.mul_(group["momentum"]).add_(p.grad)
                update = newton_schulz(p.grad + group["momentum"] * m)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * 0.2 * math.sqrt(max(p.shape)))
```

```python title="orthogonalize.py"
import torch
from muon import newton_schulz

torch.manual_seed(0)
# build a gradient whose singular values decay exponentially from 1 to 0.01 (a real weight gradient is usually dominated by a few directions)
U, _ = torch.linalg.qr(torch.randn(256, 256))
V, _ = torch.linalg.qr(torch.randn(1024, 256))
S = torch.logspace(0, -2, 256)
G = U @ torch.diag(S) @ V.T                                  # 256 × 1024

for steps in (1, 3, 5):
    s = torch.linalg.svdvals(newton_schulz(G, steps))
    print(f"Newton-Schulz {steps} 步：奇异值在 [{s.min():.2f}, {s.max():.2f}]，落在 [0.7, 1.2] 的占 {((s > 0.7) & (s < 1.2)).float().mean():.0%}")
X = newton_schulz(G)
top = (X @ V[:, :8]).norm() ** 2 / X.norm() ** 2
print(f"梯度最大的 8 个方向（共 256 个）占的能量：原始梯度 {(S[:8] ** 2).sum() / (S ** 2).sum():.0%}，正交化之后 {top:.1%}")
print(f"正交化之后的均方根 {X.pow(2).mean().sqrt():.4f} ≈ 1/√1024 = {1024 ** -0.5:.4f}；乘上 0.2·√1024 之后是 {(X * 0.2 * 1024 ** 0.5).pow(2).mean().sqrt():.2f}")
```

```text title="output"
Newton-Schulz 1 步：奇异值在 [0.01, 0.62]，落在 [0.7, 1.2] 的占 0%
Newton-Schulz 3 步：奇异值在 [0.08, 1.20]，落在 [0.7, 1.2] 的占 45%
Newton-Schulz 5 步：奇异值在 [0.68, 1.20]，落在 [0.7, 1.2] 的占 89%
梯度最大的 8 个方向（共 256 个）占的能量：原始梯度 25%，正交化之后 3.5%
正交化之后的均方根 0.0294 ≈ 1/√1024 = 0.0312；乘上 0.2·√1024 之后是 0.19
```

- After 5 steps the great majority of the singular values are near 1 (not exactly 1: the coefficients were chosen for fast convergence rather than accurate convergence, and in practice that is enough). The quarter of the energy that a few directions held is spread evenly over all of them after orthogonalisation.
- After orthogonalisation the matrix's root mean square is about $1/\sqrt{\max(A, B)}$, which depends on its shape. Moonshot multiplies by $0.2\sqrt{\max(A, B)}$ so that the update's root mean square is about 0.2, comparable to AdamW's typical value, which lets them **reuse AdamW's learning rate and weight decay directly** (Keller Jordan's original implementation uses a different scaling of $\sqrt{\max(1, A/B)}$, which needs its own learning rate).
- The embedding, the output layer, the normalisation gains and the biases **do not use Muon** and stay on AdamW: the embedding has gradients only on the rows whose tokens appeared, the output layer sets the logits' scale, and vector parameters have no matrix structure.

Comparing the two optimizers on a small two-layer GPT (three learning rates each; an experiment of this size takes about a minute and a half on a CPU). The training data is a language generated by a second-order Markov chain, where the next token is determined by the previous two, so the model has to learn to use attention to look back:

```python title="muon_vs_adamw.py"
import torch
import torch.nn as nn
import torch.nn.functional as F
from muon import Muon

V, D, T = 64, 96, 64
g = torch.Generator().manual_seed(0)
# a language: a second-order Markov chain where the next token is determined by the previous two (each context has only a few common successors)
P = (torch.randn(V, V, V, generator=g) * 3).softmax(-1)
seqs = torch.randint(0, V, (8192, 2), generator=g)
for _ in range(T - 1):
    nxt = torch.multinomial(P[seqs[:, -2], seqs[:, -1]], 1, generator=g)
    seqs = torch.cat([seqs, nxt], dim=1)                    # 8192 sequences of length 65
train, val = seqs[:7168], seqs[7168:]
print(f"这个语言的熵：每个 token {-(P * P.log()).sum(-1).mean():.2f} nat（loss 的下限）")


class Block(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.up, self.down = nn.Linear(D, 4 * D, bias=False), nn.Linear(4 * D, D, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(Block(), Block())
        self.norm, self.head = nn.RMSNorm(D), nn.Linear(D, V, bias=False)

    def forward(self, idx):
        return self.head(self.norm(self.blocks(self.emb(idx) + self.pos(torch.arange(idx.shape[1])))))


def train_run(name, lr, steps=400, batch=32):
    torch.manual_seed(0)
    model = TinyGPT()
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks") and p.ndim == 2]
    others = [p for n, p in model.named_parameters() if not (n.startswith("blocks") and p.ndim == 2)]
    if name == "AdamW":
        opts = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)]
    else:                                                    # the embedding, the output layer and the normalisation stay on AdamW
        opts = [Muon(hidden, lr=lr), torch.optim.AdamW(others, lr=lr, weight_decay=0.0)]
    gen, curve = torch.Generator().manual_seed(1), []
    for step in range(1, steps + 1):
        x = train[torch.randint(0, len(train), (batch,), generator=gen)]
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()
        if step % 100 == 0:
            with torch.no_grad():
                curve.append(F.cross_entropy(model(val[:, :-1]).flatten(0, 1), val[:, 1:].flatten()).item())
    return curve


print("验证集 loss        第 100 步  第 200 步  第 300 步  第 400 步")
for name, lrs in [("AdamW", (3e-3, 1e-2, 2e-2)), ("Muon", (1e-2, 2e-2, 4e-2))]:
    for lr in lrs:
        print(f"{name:5s} lr={lr:<6}  " + "".join(f"{v:9.3f}" for v in train_run(name, lr)))
```

```text title="output"
这个语言的熵：每个 token 1.81 nat（loss 的下限）
验证集 loss        第 100 步  第 200 步  第 300 步  第 400 步
AdamW lr=0.003       4.026    3.974    3.708    3.121
AdamW lr=0.01        4.017    3.841    3.069    2.741
AdamW lr=0.02        4.027    4.003    3.913    3.295
Muon  lr=0.01        4.006    3.480    2.826    2.634
Muon  lr=0.02        3.989    3.170    2.754    2.612
Muon  lr=0.04        4.015    3.938    3.159    2.796
```

For the first hundred-odd steps both optimizers sit near 4.0 (having learned only each token's frequency); getting below that requires learning to look at the previous two tokens. Muon clearly comes off the plateau by step 200, and its best run is 0.13 below AdamW's best at step 400. On large models, Moonshot's scaling experiments have Muon reach the same loss at about half of AdamW's compute; Kimi K2 (1T parameters) and GLM-4.5 both pretrained with it.

The cost is the Newton-Schulz matrix multiplies each step. For an $m \times n$ matrix ($m \le n$), each iteration is about $4m^2n + 2m^3$ operations, and after 5 steps, compared with that matrix's training compute of $6mn \times$ the token count, it is usually under 1% at a large batch, provided each matrix is computed once. The next section shows that this does not hold automatically in distributed training.

## Distributed Muon: the conflict with ZeRO {#分布式-muon和-zero-的冲突}

[ZeRO and FSDP](../data/zero-fsdp.md) partition the optimizer states across the data-parallel ranks, with each rank updating only its own share: ZeRO by intervals of the flattened elements, FSDP2 by each parameter's dimension 0. AdamW is elementwise and the partition makes no difference to the result; Muon's orthogonalisation needs **the complete matrix**, and a rank with only a few rows cannot compute the whole matrix's $U V^\top$. Simulating FSDP2's row partition over 4 processes and comparing three approaches:

```python title="distributed_muon.py" torchrun="4"
import torch
import torch.distributed as dist
from muon import newton_schulz

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                        # every rank builds the same already-all-reduced gradient
shapes = [(256, 256), (256, 1024), (1024, 256), (256, 768)]  # the various matrices of attention and the MLP
grads = [torch.randn(s) @ torch.diag(torch.logspace(0, -2, s[1])) for s in shapes]


def rows(t):                                                # FSDP2's partition: each parameter cut along dimension 0, with each rank taking consecutive rows
    n = t.shape[0] // world
    return t[rank * n:(rank + 1) * n]


def gather_rows(local):
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local)
    return torch.cat(parts)


ref = [newton_schulz(g) for g in grads]                     # single-card Muon's update (without momentum, looking only at the orthogonalisation)

# 1. each rank orthogonalises only its own rows: the matrix has been broken apart and the result is wrong
local = [newton_schulz(rows(g)) for g in grads]
err1 = max(((gather_rows(u) - r).norm() / r.norm()).item() for u, r in zip(local, ref))

# 2. all-gather the complete matrix, orthogonalise it on every rank and take your own rows (Moonshot's Distributed Muon approach)
calls2 = 0
out2 = []
for g in grads:
    full = gather_rows(rows(g))
    out2.append(rows(newton_schulz(full)))
    calls2 += 1
err2 = max((gather_rows(u) - r).abs().max().item() for u, r in zip(out2, ref))

# 3. one matrix per rank: gather to that rank, orthogonalise there alone, and scatter back to the ranks
calls3 = 0
out3 = []
for i, g in enumerate(grads):
    owner = i % world
    parts = [torch.empty_like(rows(g)) for _ in range(world)] if rank == owner else None
    dist.gather(rows(g), parts, dst=owner)
    pieces = None
    if rank == owner:
        pieces = list(newton_schulz(torch.cat(parts)).chunk(world))
        calls3 += 1
    mine = torch.empty_like(rows(g))
    dist.scatter(mine, pieces, src=owner)
    out3.append(mine)
err3 = max((gather_rows(u) - r).abs().max().item() for u, r in zip(out3, ref))
calls = torch.tensor([calls3])
dist.all_reduce(calls, op=dist.ReduceOp.MAX)

if rank == 0:
    numel = sum(a * b for a, b in shapes)
    print(f"{world} 个 rank，{len(shapes)} 个矩阵，共 {numel / 1e6:.2f}M 个参数")
    print(f"① 只对本地的行正交化：与单卡的相对误差 {err1:.0%}")
    print(f"② all-gather 后各自正交化：与单卡的最大差 {err2:.1e}，每个 rank 做 {calls2} 次 Newton-Schulz")
    print(f"③ 每个矩阵交给一个 rank：与单卡的最大差 {err3:.1e}，每个 rank 最多做 {calls.item()} 次 Newton-Schulz")
dist.destroy_process_group()
```

```text title="output"
4 个 rank，4 个矩阵，共 0.79M 个参数
① 只对本地的行正交化：与单卡的相对误差 120%
② all-gather 后各自正交化：与单卡的最大差 0.0e+00，每个 rank 做 4 次 Newton-Schulz
③ 每个矩阵交给一个 rank：与单卡的最大差 0.0e+00，每个 rank 最多做 1 次 Newton-Schulz
```

- **1. Orthogonalising only the local rows**: completely wrong. Orthogonalisation couples all of the rows, and doing it piecewise gives something else entirely.
- **2. All-gather then orthogonalise on each rank** (Moonshot's Distributed Muon, built on ZeRO-1): correct, at the cost of one all-gather of the momentum matrix and **every rank orthogonalising every matrix**. The larger the matrices and the higher the data-parallel degree, the more this duplicated computation shows (see exercise 2).
- **3. One matrix per rank**: gather to that rank, orthogonalise there alone, and scatter back. No computation is duplicated and the total communication is lower, but the load has to be balanced: the matrices differ in size and have to be assigned by size (modded-nanogpt and others deal the parameters round-robin to the ranks and all-gather the updates afterwards).

Tensor parallelism complicates it further: a matrix is already partitioned across several cards, so the complete matrix has to be assembled within the tensor-parallel group before orthogonalising, and that communication goes over the fast interconnect, which is usually acceptable.

!!! interview "How to answer in an interview"
    Asked what Muon is: **what it does** (the momentum matrix, approximately orthogonalised into $U V^\top$ by Newton-Schulz, scaled by $0.2\sqrt{\max(A,B)}$ before the update; only on hidden-layer matrices, with the embedding, the output layer and the vector parameters still on AdamW), **why it works** (the gradient matrix is dominated by a few directions, and orthogonalisation gives every direction the same step, which amounts to steepest descent under the spectral norm), and **what it costs in engineering** (one moment less memory than AdamW, 12 against 16 bytes per parameter; a few extra matrix multiplies per step; and a conflict with ZeRO and FSDP's elementwise partition that requires gathering the complete matrix while avoiding every rank duplicating the work). Being able to add that Moonshot's 0.2 scaling and weight decay exist to reuse AdamW's hyperparameters, and that Kimi K2 used MuonClip to solve exploding attention logits (see [Training stability](stability.md)), counts in your favour.

## Exercises {#练习}

**1. Why is Muon called steepest descent under the spectral norm?** Hint: under the constraint $\|\Delta W\|_2 \le \eta$, which $\Delta W$ minimises $\langle G, \Delta W \rangle$?

??? success "Answer"
    Let $G = U S V^\top$. Then $\langle G, \Delta W \rangle = \mathrm{tr}(G^\top \Delta W)$, and under a spectral norm of at most $\eta$ the minimum is at $\Delta W = -\eta U V^\top$, with a value of $-\eta \sum_i s_i$ (the nuclear norm). That is exactly Muon's update direction. By contrast, under a Frobenius-norm constraint the optimum is $-\eta G / \|G\|_F$, which is (normalised) gradient descent; under an elementwise infinity-norm constraint it is $-\eta\,\mathrm{sign}(G)$, which corresponds to Adam without its smoothing. The difference between the three optimizers can be seen as which norm measures how far one step goes.

**2. How expensive is the duplication?** A 4096x4096 matrix, a global batch of 4 million tokens, a data-parallel degree of 64. If every rank orthogonalises it (approach 2), what share of that matrix's per-rank training compute is the Newton-Schulz? And under approach 3?

??? success "Answer"
    Each Newton-Schulz iteration is about $4m^2n + 2m^3 = 6 \times 4096^3 \approx 4.1 \times 10^{11}$ operations, so 5 steps is about $2.1 \times 10^{12}$. Each rank handles $4 \times 10^6 / 64 \approx 6.3 \times 10^4$ tokens, so this matrix's training compute (forward plus backward) is about $6 \times 4096^2 \times 6.3 \times 10^4 \approx 6.3 \times 10^{12}$. Approach 2's overhead is about **33%**; under approach 3 each matrix is computed on one rank, which averages to about 0.5% per rank. The higher the data-parallel degree, the fewer tokens per rank and the larger the duplication's share, which is why large-scale training has to spread the orthogonalisation out.

**3. Why did Moonshot add weight decay to Muon?**

??? success "Answer"
    They found that without weight decay, the root mean square of the weights and the layer outputs kept growing as training went on, beyond bf16's high-precision range, and the results degraded later in training; with the same decoupled weight decay as AdamW, the weights' scale stayed steady and long training came out better. This follows the same reasoning as AdamW replacing Adam with L2: the decay belongs on the weights directly rather than mixed into a normalised update.

## Summary {#小结}

- [x] AdamW's decay bypasses the $\sqrt{v}$ normalisation and treats every parameter alike; Adam with L2 has its regularisation distorted by the gradients' magnitude.
- [x] Mixed precision with AdamW is 16 bytes per parameter (8 of them optimizer states); Muon stores only the momentum, 12 bytes.
- [x] Muon: the momentum matrix is approximately orthogonalised into $U V^\top$ by Newton-Schulz and scaled by $0.2\sqrt{\max(A,B)}$, used only on hidden-layer matrices; on a small GPT it reaches a lower loss in the same number of steps.
- [x] Orthogonalisation needs the complete matrix, which conflicts with ZeRO and FSDP's elementwise partition: gather before orthogonalising, and spread the matrices across ranks to avoid duplicating the work.
