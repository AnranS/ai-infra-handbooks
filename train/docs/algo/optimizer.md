# 优化器：AdamW、Muon 与分布式优化器

<p class="lead">优化器决定了两件事：训练收敛得多快，以及显存账本里最大的那一项——每个参数 16 字节里有 12 字节属于优化器（fp32 主权重和两个矩）。这一章先把 AdamW 的每个细节和它的账本讲清楚，再看 2025 年以来被 Kimi、GLM 等模型用到万亿参数规模的新优化器 Muon：它的核心是一次矩阵"正交化"，这恰好和 ZeRO / FSDP 按元素切分优化器状态的做法冲突。最后用多进程实验看分布式 Muon 怎样解决这个冲突。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. AdamW 和"Adam + L2 正则"有什么区别？为什么大模型都用 AdamW？
    2. 混合精度训练里，每个参数的 16 字节分别是什么？换成 Muon 能省多少？
    3. Muon 对梯度做了什么？为什么用 Newton-Schulz 迭代而不是 SVD？
    4. Muon 为什么只用在隐藏层的二维矩阵上？
    5. ZeRO 按元素切分优化器状态，为什么和 Muon 冲突？有哪几种解决办法，各自的代价是什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. Adam + L2 把 $\lambda w$ 加到梯度上，再一起除以 $\sqrt{v}$：梯度大的参数衰减被削弱、梯度小的被放大，正则强度被梯度的量级扭曲。AdamW 把衰减从梯度里拿出来，直接对参数做 $w \leftarrow w - \eta \lambda w$，对所有参数一视同仁，效果更好、超参数也更好调。
    2. bf16 参数 2、bf16 梯度 2、fp32 主权重 4、Adam 一阶矩 4、二阶矩 4；Muon 只存一份动量，每参数 12 字节，省下 4 字节（二阶矩）。
    3. 把动量矩阵正交化：近似成 $UV^\top$（奇异值全变成 1），让更新在各个方向上的幅度一致，再按 $0.2\sqrt{\max(A,B)}$ 缩放。Newton-Schulz 迭代只用矩阵乘，几步就够、能在 GPU 上高效地用 bf16 运行；SVD 又慢又难并行。
    4. 正交化只对二维矩阵有意义；嵌入层、输出层是"查表"和"打分"，按行 / 列的语义与隐藏层的线性变换不同，一维的参数（归一化的权重、偏置）也无法正交化，这些仍然用 AdamW。
    5. 正交化需要完整的矩阵，而 ZeRO / FSDP 把每个矩阵按元素切到不同的卡上。办法：先 all-gather 出完整的矩阵再正交化（多一次通信、重复计算），或者把不同的矩阵分给不同的 rank 各自完整地算、算完再分发（负载要均衡），或者只在 TP 的分片内做近似正交化（有精度损失）。

## AdamW：解耦的权重衰减

Adam 为每个参数维护梯度的一阶矩 $m$（动量）和二阶矩 $v$，更新量是 $m / (\sqrt{v} + \epsilon)$：每个坐标按自己梯度的量级归一化，所以对梯度的尺度不敏感。权重衰减有两种加法：

- **Adam + L2 正则**：把 $\lambda w$ 加进梯度，再和梯度一起被 $\sqrt{v}$ 归一化；
- **AdamW**（Loshchilov & Hutter）：衰减不经过归一化，直接 $w \leftarrow w - \eta \lambda w$。

区别只在一处，后果却很大。让两个参数的梯度量级相差 1000 倍，只看衰减的效果：

先拨一拨看结果（下面的脚本就是这个实验）：

<div class="aig-widget" data-widget="adamw-step"></div>

```python title="adamw_decay.py"
import torch

lr, wd, steps = 1e-2, 0.1, 1000
scale = torch.tensor([0.01, 10.0])                     # 两个参数：梯度的量级一个很小、一个很大


def train(decoupled):
    w = torch.ones(2, requires_grad=True)
    if decoupled:
        opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)            # 衰减直接作用在权重上
    else:
        opt = torch.optim.Adam([w], lr=lr, weight_decay=wd)             # 衰减加进梯度（L2 正则），再被 Adam 归一化
    for t in range(steps):
        opt.zero_grad()
        w.grad = scale * (-1) ** t                                      # 正负交替、平均为 0 的梯度：只让 Adam 的 v 有量级
        opt.step()
    return w.detach()


for name, decoupled in [("Adam + L2 正则", False), ("AdamW（解耦的权重衰减）", True)]:
    w = train(decoupled)
    print(f"{name:16s} 小梯度的参数 {w[0]:.3f}，大梯度的参数 {w[1]:.3f}")
print(f"只有衰减时的理论值：(1 - lr·wd)^{steps} = {(1 - lr * wd) ** steps:.3f}")
```

```text title="输出"
Adam + L2 正则     小梯度的参数 0.000，大梯度的参数 0.889
AdamW（解耦的权重衰减）   小梯度的参数 0.361，大梯度的参数 0.361
只有衰减时的理论值：(1 - lr·wd)^1000 = 0.368
```

Adam + L2 里，正则项被 $\sqrt{v}$ 除掉：梯度小的参数被过度衰减（很快归零），梯度大的参数几乎不衰减——正则的强度取决于梯度的量级，这不是我们想要的。AdamW 对所有参数一视同仁，衰减的效果和理论值一致。所以大模型训练用的都是 AdamW，`weight_decay` 常取 0.1，而且通常**不对**归一化层的增益和偏置做衰减。

## 优化器的显存账本

混合精度训练中每个参数的开销（见[混合精度](../practice/mixed-precision.md)和[显存账本](../basics/overview.md)）：

| 项目 | AdamW | Muon（隐藏层矩阵） |
| --- | --- | --- |
| bf16 权重 | 2 | 2 |
| bf16 梯度 | 2 | 2 |
| fp32 主权重 | 4 | 4 |
| 优化器状态 | $m$、$v$ 各 4 字节 = 8 | 动量 4 |
| **合计** | **16** | **12** |

用 PyTorch 实际量一下两种优化器保存的状态（Muon 的实现见下一节）：

```python title="optimizer_state.py"
import torch
from muon import Muon


def state_bytes(opt):
    return sum(t.numel() * t.element_size() for s in opt.state.values() for t in s.values()
               if torch.is_tensor(t) and t.dim() > 0)          # 不算 step 这类标量


for name in ("AdamW", "Muon"):
    torch.manual_seed(0)
    w1, w2 = torch.randn(4096, 1024, requires_grad=True), torch.randn(1024, 4096, requires_grad=True)
    opt = torch.optim.AdamW([w1, w2], lr=1e-3) if name == "AdamW" else Muon([w1, w2], lr=1e-3)
    (torch.randn(8, 1024) @ w1.T @ w2.T).sum().backward()
    opt.step()
    n = w1.numel() + w2.numel()
    print(f"{name:5s}：{n / 1e6:.1f}M 个参数，优化器状态 {state_bytes(opt) / n:.0f} 字节/参数")
```

```text title="输出"
AdamW：8.4M 个参数，优化器状态 8 字节/参数
Muon ：8.4M 个参数，优化器状态 4 字节/参数
```

一个 70B 的模型，只算这 16 字节就是 1.1 TB，远超一张卡的显存，这就是 [ZeRO](../data/zero-fsdp.md) 要把优化器状态切到各个数据并行 rank 上的原因。Muon 少存一个矩，能省下四分之一的模型状态显存。

## Muon：把更新"正交化"

Muon（MomentUm Orthogonalized by Newton-Schulz，Keller Jordan 2024）对隐藏层的每个权重矩阵这样更新：

1. 和 SGD 一样维护动量 $M \leftarrow \mu M + G$；
2. 把（Nesterov 式的）动量矩阵 $G + \mu M = U S V^\top$ **正交化**成 $U V^\top$：保留方向，把所有奇异值都换成 1；
3. $W \leftarrow W - \eta \cdot 0.2\sqrt{\max(A, B)} \cdot U V^\top$，$A \times B$ 是矩阵的形状。

为什么要正交化？Transformer 权重的梯度和动量往往被少数几个方向主导（奇异值相差几个数量级），直接按它更新，稀有但有用的方向几乎得不到更新。正交化让每个奇异方向走同样大的一步，等价于在"谱范数"意义下做最速下降。SVD 在 GPU 上又慢又难并行，Muon 用 **Newton-Schulz 迭代**近似它——只用矩阵乘，bf16 下也稳定：

正交化在奇异值上做了什么，拨几次迭代看看：

<div class="aig-widget" data-widget="newton-schulz"></div>

```python title="muon.py"
"""Muon：动量 + Newton-Schulz 正交化，只用于隐藏层的二维权重矩阵"""
import math

import torch


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """近似 G 的极分解因子 U Vᵀ（G = U S Vᵀ）：把所有奇异值推到 1 附近，只用矩阵乘"""
    a, b, c = 3.4445, -4.7750, 2.0315                       # 五次多项式的系数：收敛快，奇异值落在 1 附近的一个区间
    X = G / (G.norm() + 1e-7)                               # 先缩放到谱范数 ≤ 1
    transposed = G.shape[0] > G.shape[1]
    if transposed:                                          # 让 X Xᵀ 是较小的那个方阵
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
# 造一个"梯度"：奇异值从 1 到 0.01 按指数衰减（真实的权重梯度往往被少数几个方向主导）
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

```text title="输出"
Newton-Schulz 1 步：奇异值在 [0.01, 0.62]，落在 [0.7, 1.2] 的占 0%
Newton-Schulz 3 步：奇异值在 [0.08, 1.20]，落在 [0.7, 1.2] 的占 45%
Newton-Schulz 5 步：奇异值在 [0.68, 1.20]，落在 [0.7, 1.2] 的占 89%
梯度最大的 8 个方向（共 256 个）占的能量：原始梯度 25%，正交化之后 3.5%
正交化之后的均方根 0.0294 ≈ 1/√1024 = 0.0312；乘上 0.2·√1024 之后是 0.19
```

- 5 步之后，绝大部分奇异值落在 1 附近（不是精确的 1，系数是为"收敛快"而不是"收敛准"挑的，实践中这就够了）。原来被少数方向占掉四分之一的能量，正交化之后均匀地分给了所有方向；
- 正交化之后矩阵的均方根约为 $1/\sqrt{\max(A, B)}$，和矩阵的形状有关。Moonshot 的做法是乘上 $0.2\sqrt{\max(A, B)}$，让更新的均方根约为 0.2，和 AdamW 的典型值相当，这样可以**直接沿用 AdamW 的学习率和权重衰减**（Keller Jordan 的原始实现用的是另一种缩放 $\sqrt{\max(1, A/B)}$，学习率要单独调）；
- 嵌入层、输出层、归一化的增益和偏置**不用 Muon**，仍然用 AdamW：嵌入每次只有出现过的 token 那几行有梯度，输出层决定 logits 的尺度，向量参数没有矩阵结构。

在一个两层的小 GPT 上比较两种优化器（各自扫三个学习率，这个规模的实验在 CPU 上约一分半）。训练数据是一个二阶马尔可夫链生成的"语言"，下一个 token 由前两个决定，所以模型必须学会用注意力看前面的 token：

```python title="muon_vs_adamw.py"
import torch
import torch.nn as nn
import torch.nn.functional as F
from muon import Muon

V, D, T = 64, 96, 64
g = torch.Generator().manual_seed(0)
# 一个"语言"：二阶马尔可夫链，下一个 token 由前两个决定（每个上下文只有少数几个常见的后继）
P = (torch.randn(V, V, V, generator=g) * 3).softmax(-1)
seqs = torch.randint(0, V, (8192, 2), generator=g)
for _ in range(T - 1):
    nxt = torch.multinomial(P[seqs[:, -2], seqs[:, -1]], 1, generator=g)
    seqs = torch.cat([seqs, nxt], dim=1)                    # 8192 条长 65 的序列
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
    else:                                                    # 嵌入、输出层、归一化仍然用 AdamW
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

```text title="输出"
这个语言的熵：每个 token 1.81 nat（loss 的下限）
验证集 loss        第 100 步  第 200 步  第 300 步  第 400 步
AdamW lr=0.003       4.026    3.974    3.708    3.121
AdamW lr=0.01        4.017    3.841    3.069    2.741
AdamW lr=0.02        4.027    4.003    3.913    3.295
Muon  lr=0.01        4.006    3.480    2.826    2.634
Muon  lr=0.02        3.989    3.170    2.754    2.612
Muon  lr=0.04        4.015    3.938    3.159    2.796
```

前 100 多步，两个优化器都停在 4.0 附近（只学会了每个 token 出现的频率），要"学会看前两个 token"才能继续下降。Muon 在第 200 步就明显走出了平台，最好的一组在第 400 步比 AdamW 最好的一组低 0.13。在大模型上，Moonshot 的扩展实验里 Muon 达到同样的 loss 大约只需要 AdamW 一半的计算量；Kimi K2（1T 参数）和 GLM-4.5 都用它做了预训练。

代价是每步多了 Newton-Schulz 的矩阵乘。对 $m \times n$（$m \le n$）的矩阵，每步迭代约 $4m^2n + 2m^3$ 次浮点运算，5 步之后和"这个矩阵在这一步训练里的计算量" $6mn \times$ token 数相比，大 batch 下通常不到 1%——前提是每个矩阵只算一次，下一节会看到这一点在分布式训练里并不自动成立。

## 分布式 Muon：和 ZeRO 的冲突

[ZeRO / FSDP](../data/zero-fsdp.md) 把优化器状态切到各个数据并行 rank 上，每个 rank 只更新自己那一份参数：ZeRO 按展平后的元素区间切，FSDP2 按每个参数的第 0 维切。AdamW 是逐元素的，怎么切都不影响结果；Muon 的正交化却需要**完整的矩阵**——每个 rank 只有几行，算不出整个矩阵的 $U V^\top$。用 4 个进程模拟 FSDP2 式的按行切分，比较三种做法：

```python title="distributed_muon.py" torchrun="4"
import torch
import torch.distributed as dist
from muon import newton_schulz

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                        # 每个 rank 造出同样的"已经 all-reduce 过的"梯度
shapes = [(256, 256), (256, 1024), (1024, 256), (256, 768)]  # 注意力、MLP 的几种矩阵
grads = [torch.randn(s) @ torch.diag(torch.logspace(0, -2, s[1])) for s in shapes]


def rows(t):                                                # FSDP2 式的切分：每个参数按第 0 维切，rank 拿连续的若干行
    n = t.shape[0] // world
    return t[rank * n:(rank + 1) * n]


def gather_rows(local):
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local)
    return torch.cat(parts)


ref = [newton_schulz(g) for g in grads]                     # 单卡 Muon 的更新（不带动量，只看正交化这一步）

# ① 每个 rank 只对自己那几行做正交化：矩阵被拆开了，结果是错的
local = [newton_schulz(rows(g)) for g in grads]
err1 = max(((gather_rows(u) - r).norm() / r.norm()).item() for u, r in zip(local, ref))

# ② 先 all-gather 出完整矩阵，每个 rank 都做一遍正交化，再取自己的行（Moonshot 的 Distributed Muon 思路）
calls2 = 0
out2 = []
for g in grads:
    full = gather_rows(rows(g))
    out2.append(rows(newton_schulz(full)))
    calls2 += 1
err2 = max((gather_rows(u) - r).abs().max().item() for u, r in zip(out2, ref))

# ③ 每个矩阵交给一个 rank：gather 到那个 rank，只在那里正交化，再 scatter 回各个 rank
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

```text title="输出"
4 个 rank，4 个矩阵，共 0.79M 个参数
① 只对本地的行正交化：与单卡的相对误差 120%
② all-gather 后各自正交化：与单卡的最大差 0.0e+00，每个 rank 做 4 次 Newton-Schulz
③ 每个矩阵交给一个 rank：与单卡的最大差 0.0e+00，每个 rank 最多做 1 次 Newton-Schulz
```

- **① 只对本地的行正交化**：结果完全不对。正交化把所有行耦合在一起，拆开做得到的是另一个东西；
- **② 先 all-gather 再各自正交化**（Moonshot 的 Distributed Muon，建在 ZeRO-1 上）：结果正确，多了一次动量矩阵的 all-gather，而且**每个 rank 都把所有矩阵正交化一遍**。矩阵越大、数据并行度越高，这份重复计算越显眼（见练习 2）；
- **③ 每个矩阵交给一个 rank**：gather 到这个 rank，只在它上面正交化，再 scatter 回去。计算不重复，通信总量也更少，但负载要均衡：矩阵大小不一，需要按大小分配（modded-nanogpt 等实现按参数轮流分给各个 rank，算完再 all-gather 更新）。

张量并行让问题更复杂：一个矩阵本来就被切在几张卡上，正交化之前要先在 TP 组内拼出完整矩阵，这部分通信在 TP 的高速互连上，代价通常可以接受。

!!! interview "怎么讲清楚"
    讲"Muon 是什么"：**做法**（动量矩阵 → Newton-Schulz 近似正交化成 $U V^\top$ → 按 $0.2\sqrt{\max(A,B)}$ 缩放后更新；只用于隐藏层矩阵，嵌入、输出层、向量参数仍用 AdamW）→ **为什么有效**（梯度矩阵被少数方向主导，正交化让每个方向步长相同，相当于谱范数下的最速下降）→ **工程代价**（显存比 AdamW 少一个矩，12 对 16 字节/参数；每步多几次矩阵乘；和 ZeRO/FSDP 的逐元素切分冲突，需要 gather 完整矩阵，要避免每个 rank 重复计算）。能说出"Moonshot 为了沿用 AdamW 的超参数加了 0.2 倍的缩放和权重衰减"、"Kimi K2 用 MuonClip 解决了注意力 logits 爆炸"（见[训练稳定性](stability.md)）会是加分项。

## 练习

**1. 为什么说 Muon 是"谱范数下的最速下降"？** 提示：在 $\|\Delta W\|_2 \le \eta$ 的约束下，让 $\langle G, \Delta W \rangle$ 最小的 $\Delta W$ 是什么？

??? success "参考答案"
    设 $G = U S V^\top$。$\langle G, \Delta W \rangle = \mathrm{tr}(G^\top \Delta W)$，在谱范数 $\le \eta$ 的约束下，最小值在 $\Delta W = -\eta U V^\top$ 取到，最小值是 $-\eta \sum_i s_i$（核范数）。这正是 Muon 的更新方向。对比：在 Frobenius 范数约束下，最优解是 $-\eta G / \|G\|_F$，也就是（归一化的）梯度下降；在逐元素的无穷范数约束下是 $-\eta\,\mathrm{sign}(G)$，对应 Adam 去掉平滑之后的形式。三种优化器的差别，可以看成是"用什么范数度量一步走了多远"。

**2. 重复计算有多贵？** 一个 4096×4096 的矩阵，全局 batch 是 400 万个 token，数据并行度 64。如果每个 rank 都把它正交化一遍（做法②），Newton-Schulz 的计算量占这个矩阵在每个 rank 上训练计算量的多少？换成做法③呢？

??? success "参考答案"
    Newton-Schulz 每步迭代约 $4m^2n + 2m^3 = 6 \times 4096^3 \approx 4.1 \times 10^{11}$ 次浮点运算，5 步约 $2.1 \times 10^{12}$。每个 rank 处理 $4 \times 10^6 / 64 \approx 6.3 \times 10^4$ 个 token，这个矩阵的训练计算量（前向 + 反向）约 $6 \times 4096^2 \times 6.3 \times 10^4 \approx 6.3 \times 10^{12}$。做法②的额外开销约 **33%**；做法③里每个矩阵只在一个 rank 上算，平均到每个 rank 约 0.5%。数据并行度越高，每个 rank 的 token 越少，重复计算的比例越高——这就是为什么大规模训练必须把正交化分摊出去。

**3. 为什么 Moonshot 要给 Muon 加上权重衰减？**

??? success "参考答案"
    他们发现不加权重衰减时，随着训练进行，权重和层输出的均方根持续增大，超出了 bf16 的高精度范围，后期效果变差；加上和 AdamW 一样的解耦权重衰减之后，权重的尺度保持稳定，长时间训练的效果反而更好。这和 AdamW 取代 Adam + L2 的道理一脉相承：衰减要直接作用在权重上，而不是混进被归一化的更新里。

## 小结

- [x] AdamW 的衰减不经过 $\sqrt{v}$ 归一化，对所有参数一视同仁；Adam + L2 的正则强度会被梯度量级扭曲。
- [x] 混合精度 + AdamW 每参数 16 字节（其中优化器状态 8 字节）；Muon 只存动量，12 字节。
- [x] Muon：动量矩阵用 Newton-Schulz 近似正交化成 $U V^\top$，按 $0.2\sqrt{\max(A,B)}$ 缩放，只用于隐藏层矩阵；在小 GPT 上同样的步数 loss 更低。
- [x] 正交化需要完整矩阵，和 ZeRO / FSDP 的逐元素切分冲突：先 gather 再正交化，并把各个矩阵的计算分摊到不同的 rank，避免重复计算。
