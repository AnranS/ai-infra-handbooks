# （八）Gated DeltaNet：分块并行

<p class="lead">上一章的递推一步只做几个 128 × 128 的小运算，T 个 token 就是 T 次 Python 循环，GPU 完全吃不饱——这正是线性注意力早年"理论上快、实际上慢"的原因。把序列切成 64 个 token 一块：块内 64 次"衰减 + 写入"合起来是一个下三角线性方程组，一次三角求解就能得到每个 token 真正写入的量；块与块之间仍像递推一样只传一张表。结果和逐 token 版一样，速度差一个数量级。</p>

!!! abstract "学完这章你能"
    1. 说清逐 token 递推为什么在 GPU 上慢：每一步都要等上一步，每一步又只有很小的运算。
    2. 用 3 个 token 手算：块内的依赖怎么写成一个下三角方程组 $(I + A)U = \beta \odot V$，一次解出每个 token 真正写入的量。
    3. 读懂 `chunked_delta_rule`：块内是三角求解加矩阵乘，块间只传一张表；知道块长 64 是在权衡什么。

!!! note "这一章跑什么"
    3 个 token 的手算例子 `ch08_toy.py`（1 秒），和对答案、比速度的 `ch08_chunk.py`（约 15 秒）。

@@files@@

## 为什么要分块

上一章的递推，第 t 步要用第 t − 1 步改完的表，一步等一步；每一步只是几个 128 × 128 的小运算。GPU 擅长的是一次做很大的矩阵乘，这样一步一步喂，它大部分时间都在等。训练和 prefill 一次要处理成千上万个 token，这就成了瓶颈。

能不能全部并行？可以把整段的依赖一次写出来，但中间量是 $T \times T$ 的，又回到了注意力的平方复杂度。折中的办法是分块：64 个 token 一块，**块内**把 64 步的依赖写成一个方程组一次解完，**块间**像递推一样只传一张表。

## 3 个 token 手算一遍

先不管衰减，表从空的开始。第 t 个 token 真正写进表里的量记作 $u_t = \beta_t\,(v_t - S_{t-1}^\top k_t)$。表是前面写入的累加 $S_{t-1} = \sum_{j<t} k_j u_j^\top$，代进去：

$$
u_t = \beta_t \Big(v_t - \sum_{j<t} (k_t \cdot k_j)\, u_j\Big)
$$

3 个 token 时，$k = (1, 0), (0.6, 0.8), (0, 1)$，$v = (1, 0), (0, 1), (2, 2)$，$\beta = 1, 1, 0.5$：

- $u_1 = \beta_1 v_1 = (1, 0)$
- $u_2 = \beta_2\,(v_2 - 0.6\,u_1) = (-0.6, 1)$
- $u_3 = \beta_3\,(v_3 - 0 \cdot u_1 - 0.8\,u_2) = 0.5 \times (2.48, 1.2) = (1.24, 0.6)$

算 $u_3$ 得先有 $u_1$、$u_2$，这就是递推慢的原因。把带 $u$ 的项都移到左边：

$$
u_t + \beta_t \sum_{j<t} (k_t \cdot k_j)\, u_j = \beta_t v_t \qquad\Longrightarrow\qquad (I + A)\,U = \beta \odot V,\quad A_{tj} = \beta_t\,(k_t \cdot k_j)\ (j < t)
$$

$I + A$ 是下三角的，一次 `solve_triangular` 就同时解出 $u_1, u_2, u_3$。有了 $U$，剩下全是矩阵乘：输出 $O = (QK^\top \text{ 只留下三角})\,U$，块末的表 $S = K^\top U$。

![图：分块并行的 3 个 token 例子：左边逐 token 递推一个等一个；右边把依赖写成下三角方程组 (I + A)U = β⊙V，一次解出 u₁、u₂、u₃；块间只传一张表](assets/figures/chunk-example.svg){.aig-svg}

```python title="ch08_toy.py"
"""第 8 章（一）：3 个 token 手算分块：先逐 token 递推，再把同一件事写成一个下三角方程组，一次解出来，对答案"""
import torch

r = lambda t: [round(x, 3) for x in t.tolist()]
K = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])           # 3 个 token 的 key（已经是单位长度）
V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])           # value
beta = torch.tensor([1.0, 1.0, 0.5])                              # 写入强度
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])           # query
print("1. 逐 token 递推（先不管衰减，表从空的开始）")
S = torch.zeros(2, 2)
for t in range(3):
    pred = S.T @ K[t]                                             # 表里现在认为 k_t 对应什么
    u = beta[t] * (V[t] - pred)                                   # 这个 token 真正写进去的量
    S = S + torch.outer(K[t], u)
    print(f"   token {t + 1}：表里预测 {r(pred)}，真正写入 u{t + 1} = {r(u)}，输出 {r(S.T @ Q[t])}")
print(f"   最后的表：{[r(row) for row in S]}")
print("2. 分块：三个 u 一次解出来")
A = torch.tril(beta[:, None] * (K @ K.T), -1)                     # A[t, j] = β_t · (k_t · k_j)，只留 j < t
print(f"   A = {[r(row) for row in A]}")
U = torch.linalg.solve_triangular(torch.eye(3) + A, beta[:, None] * V, upper=False)   # 解 (I + A) U = β ⊙ V
print(f"   解 (I + A) U = β ⊙ V 得 U = {[r(row) for row in U]}")
print(f"   输出 = (Q Kᵀ 只留下三角，含对角线) · U = {[r(row) for row in torch.tril(Q @ K.T) @ U]}")
print(f"   最后的表 = Kᵀ · U = {[r(row) for row in K.T @ U]}")
```

```text title="输出"
1. 逐 token 递推（先不管衰减，表从空的开始）
   token 1：表里预测 [0.0, 0.0]，真正写入 u1 = [1.0, 0.0]，输出 [1.0, 0.0]
   token 2：表里预测 [0.6, 0.0]，真正写入 u2 = [-0.6, 1.0]，输出 [-0.48, 0.8]
   token 3：表里预测 [-0.48, 0.8]，真正写入 u3 = [1.24, 0.6]，输出 [0.992, 1.48]
   最后的表：[[0.64, 0.6], [0.76, 1.4]]
2. 分块：三个 u 一次解出来
   A = [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.4, 0.0]]
   解 (I + A) U = β ⊙ V 得 U = [[1.0, 0.0], [-0.6, 1.0], [1.24, 0.6]]
   输出 = (Q Kᵀ 只留下三角，含对角线) · U = [[1.0, 0.0], [-0.48, 0.8], [0.992, 1.48]]
   最后的表 = Kᵀ · U = [[0.64, 0.6], [0.76, 1.4]]
```

两种算法每一个数都一样：递推一个一个算出的 $u_t$，就是方程组一次解出的 $U$ 的每一行。

## 公式：一块 64 个 token、加上衰减

先不管衰减。一块里有 64 个 token，从块首状态 $S_0$ 出发连写 64 次：

$$
S_t = S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top = S_{t-1}\,(I - \beta_t k_t k_t^\top) + \beta_t k_t v_t^\top
$$

每一步都是"乘一个 $(I - \beta k k^\top)$ 再加一个秩一矩阵"。把 64 步展开，块末状态可以写成 $S_{64} = S_0 \cdot \prod(\cdots) + K^\top U$，其中 $U$ 的每一行 $u_t$ 是第 $t$ 个 token**真正写入**的向量——它等于 $\beta_t v_t$ 减去前面 $t-1$ 个 token 写入对它的"抵消"。这个抵消关系是一个下三角方程组：

$$
(I + A)\,U = \beta \odot V, \qquad A_{ij} = \beta_i\, k_i^\top k_j \ (j < i)
$$

解它用一次 `solve_triangular`（64 × 64 的下三角矩阵，每个头一个），而不是 64 步循环。有了 $U$，块内每个位置的输出 = 读块首状态（$Q S_0$）+ 读块内新写入（$(QK^\top)_{\text{因果}} U$），两个矩阵乘。加上衰减之后每一项都要乘对应的累计衰减因子，就是代码里那几个 `G.exp()`。这套变换在文献里叫 WY 表示 / UT 变换，FlashLinearAttention 库里的 `chunk_gated_delta_rule` 就是它的 CUDA 版。

![图：分块并行：块内一次三角求解加两个矩阵乘，块间只传一张表](assets/figures/chunked.svg){.aig-svg}

```python title="qwen/chunk.py"
"""Gated DeltaNet 的分块并行版：块内用几次矩阵乘一次算完，块与块之间只传一张状态表 S。

逐 token 递推一步只做几个小矩阵运算，T 个 token 就是 T 次 Python 循环，GPU 完全吃不饱。
把序列切成 64 个 token 一块：块内 64 次"衰减 + 写入"合起来是一个下三角线性方程组（WY 表示），
用一次三角求解得到每个 token 真正写入的差值 u；然后块与块之间像递推一样传 S。结果和逐 token 版一样，误差在 1e-5 量级。
"""
import torch
import torch.nn.functional as F

from qwen.norm import l2norm


def chunked_delta_rule(q, k, v, g, beta, S=None, chunk=64):
    """和 delta_rule 同样的输入输出：q, k: [B, T, 头, dk]，v: [B, T, 头, dv]，g、beta: [B, T, 头]；S 是初始的记忆表"""
    B, T, n_head, dk = q.shape
    dv, dtype = v.shape[-1], v.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))                   # 记忆表和中间量一律 fp32（bf16 攒不住 64 步的累加）
    q, k = l2norm(q), l2norm(k)
    q = q / dk ** 0.5
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))                               # [B, 头, T, d]
    g, beta = g.transpose(1, 2), beta.transpose(1, 2)                               # [B, 头, T]
    pad = (chunk - T % chunk) % chunk                                               # 补到 chunk 的整数倍
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (T + pad) // chunk
    q, k, v = (t.reshape(B, n_head, n, chunk, -1) for t in (q, k, v))               # [B, 头, 块数, 块长, d]
    g, beta = g.reshape(B, n_head, n, chunk), beta.reshape(B, n_head, n, chunk)

    G = g.cumsum(-1)                                                                # 块内累计衰减（对数）：从块首到第 t 个 token 一共衰减了多少
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    decay = (G[..., :, None] - G[..., None, :]).masked_fill(upper, float("-inf")).exp()   # decay[i, j] = 从 j 到 i 的衰减（j > i 时为 0）
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    A = (k_beta @ k.transpose(-1, -2)) * decay                                      # 块内每对 token 之间"写入互相抵消"的系数
    u = torch.linalg.solve_triangular(A, v_beta, upper=False, unitriangular=True)   # 解下三角方程组：每个 token 真正写入的差值（还没减去块首状态的预测）
    w = torch.linalg.solve_triangular(A, k_beta * G.exp()[..., None], upper=False, unitriangular=True)   # 用来从块首状态里读预测的系数
    attn = (q @ k.transpose(-1, -2)) * decay                                        # 块内 q 对块内 k 的"注意力"
    q_decay = q * G.exp()[..., None]                                                # 读块首状态时，q 要带上从块首到自己的衰减
    k_decay = k * (G[..., -1:] - G).exp()[..., None]                                # 写入状态时，k 要带上从自己到块尾的衰减
    chunk_decay = G[..., -1].exp()                                                  # 整块的总衰减

    if S is None:
        S = q.new_zeros(B, n_head, dk, dv)
    out = torch.zeros_like(v)
    for i in range(n):                                                              # 块间：和递推一样只传 S
        u_i = u[:, :, i] - w[:, :, i] @ S                                           # 减掉块首状态已经能预测的部分，得到真正的 delta
        out[:, :, i] = q_decay[:, :, i] @ S + attn[:, :, i] @ u_i                   # 输出 = 读块首状态 + 块内读新写入
        S = S * chunk_decay[:, :, i, None, None] + k_decay[:, :, i].transpose(-1, -2) @ u_i   # 整块衰减一次，再把块内的写入一次加上
    out = out.reshape(B, n_head, n * chunk, dv)[:, :, :T].transpose(1, 2).contiguous()   # 去掉补的部分，变回 [B, T, 头, dv]
    return out.to(dtype), S
```

## 跑起来：对答案，比速度

```python title="ch08_chunk.py" ci="loose"
"""第 8 章：分块版和递推版对答案、比速度；一层真实的线性注意力过一段长输入"""
import os
import time

import torch
from tokenizers import Tokenizer

import qwen.deltanet
from qwen.chunk import chunked_delta_rule
from qwen.deltanet import GatedDeltaNet, delta_rule
from qwen.embedding import Embedding
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()
torch.manual_seed(0)
B, H, d = 1, 16, 128
for T in (5, 64, 65, 512, 2048):                                     # 1. 随机输入：两个版本逐位对，顺便计时
    q, k, v = (torch.randn(B, T, H, d) for _ in range(3))
    g, beta = -torch.rand(B, T, H) * 0.5, torch.rand(B, T, H)
    t = time.time()
    o1, S1 = delta_rule(q, k, v, g, beta)
    t1 = time.time() - t
    t = time.time()
    o2, S2 = chunked_delta_rule(q, k, v, g, beta)
    t2 = time.time() - t
    print(f"T={T:4d}：输出最大差 {diff(o1, o2):.1e}，末状态最大差 {diff(S1, S2):.1e}；递推 {t1 * 1000:6.0f} ms，分块 {t2 * 1000:5.0f} ms")

path = find_model()                                                  # 2. 一层真正的线性注意力过一段长输入：递推 vs 分块
cfg = Config.load(path)
W = load_weights(path)
tk = Tokenizer.from_file(str(path / "tokenizer.json"))
ids = torch.tensor([tk.encode("。".join(["今天天气不错，我们去公园"] * 40)).ids])
h = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")(load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")(ids))
dn = load(GatedDeltaNet(cfg), W, "layers.0.linear_attn.")
outs = {}
for name, rule in (("递推", delta_rule), ("分块", chunked_delta_rule)):
    qwen.deltanet.delta_rule = rule                                  # 第 7 章的模块调的是模块级的 delta_rule，这里换成分块版（第 10 章正式改）
    t = time.time()
    outs[name] = dn(h)
    print(f"{ids.shape[1]} 个 token 过一层线性注意力，{name}：{(time.time() - t) * 1000:.0f} ms")
print(f"两种算法在这一层上的输出最大差：{diff(outs['递推'], outs['分块']):.1e}")
```

```text title="输出"
T=   5：输出最大差 2.2e-08，末状态最大差 1.2e-07；递推      4 ms，分块     5 ms
T=  64：输出最大差 5.2e-08，末状态最大差 3.6e-07；递推     20 ms，分块     3 ms
T=  65：输出最大差 3.4e-08，末状态最大差 3.0e-07；递推     15 ms，分块     5 ms
T= 512：输出最大差 4.5e-08，末状态最大差 7.2e-07；递推    111 ms，分块    13 ms
T=2048：输出最大差 7.1e-08，末状态最大差 6.6e-07；递推    428 ms，分块    49 ms
279 个 token 过一层线性注意力，递推：283 ms
279 个 token 过一层线性注意力，分块：30 ms
两种算法在这一层上的输出最大差：3.6e-07
```

两个版本在任何长度下都逐位一致（差 1e-7 量级，纯粹是求和顺序）；一层真实的线性注意力过 279 个 token，分块版快将近十倍。Hugging Face 的实现在 prefill 时用的也是这个算法（没装 `flash-linear-attention` 时退回到和我们一样的 PyTorch 实现）。

## 第 7、8 章一图看懂

![图：Gated DeltaNet 一图看懂：为什么要它、记忆表是一本字典、三步改进、每个 token 的四步、Qwen3.5 的外壳、表会糊所以 3:1 配全注意力、分块并行，以及来历](assets/figures/gdn-poster.svg){.aig-svg}

## 常见疑问

**块长为什么是 64？**

块太小，块内的矩阵乘太小，喂不饱 GPU，块间的串行步数也多；块太大，块内的 $A$ 是块长 × 块长的，三角求解和中间量都按平方涨。64 是常见的折中，Hugging Face 里 Qwen3.5 的 PyTorch 实现默认也是 64。

**decode 时用分块还是递推？**

decode 一次只来一个 token，没有可以并行的块，用递推的一步就够了；prefill 和训练一次来一大段，用分块。第 10 章加上状态以后，两条路共用同一张表。

**加上衰减以后，公式哪里变了？**

每一项再乘上两个 token 之间的累计衰减。$G$ 是块内对数衰减的前缀和，从第 j 个 token 到第 i 个 token 一共衰减了 $e^{G_i - G_j}$，代码里的 `decay`、`q_decay`、`k_decay`、`chunk_decay` 都是它。

**这和 FlashAttention 的分块是一回事吗？**

不是。FlashAttention 分块是为了不把 $T \times T$ 的分数矩阵写回显存，算的还是完整的注意力，计算量仍是平方的；这里分块之后，块与块之间只传一张固定大小的表，计算量是线性的。

!!! interview "怎么讲清楚"
    讲"线性注意力为什么能并行训练"：递推形式一步依赖上一步，但块内的 64 步可以写成一个下三角方程组 + 两个矩阵乘，块间只传状态；这和 Mamba-2 的 SSD、FlashLinearAttention 的 chunk 算法是同一个思路。要说清"为什么要分块而不是全部并行"：全并行的中间量是 $T \times T$，又回到了注意力的平方复杂度；块长 64 是在"并行度"和"中间量大小"之间折中。

## 小结

- [x] 块内：$(I + A)U = \beta \odot V$ 一次三角求解得到真正写入量，输出 = 读块首状态 + 读块内写入；块间：传一张表。
- [x] 和递推版逐位一致，长输入快一个数量级；工业实现（FlashLinearAttention、HF 的 prefill）用的就是这个算法。

## 思考题

**1. 序列长 4096、块长 64、16 个头，一层要解多少个三角方程组？每个多大？**

??? success "参考答案"
    每个头 4096 ÷ 64 = 64 块，16 个头共 1024 个，每个都是 64 × 64 的下三角方程组。它们互相独立，可以一次批量解完；真正串行的只有块间传表的 64 步。

**2. 块长设成整段 T 会怎样？设成 1 呢？**

??? success "参考答案"
    设成 T：只有一块，$A$ 是 $T \times T$ 的，又回到了平方的计算和显存。设成 1：每块一个 token，没有可以并行的东西，退化成逐 token 递推。

**3. 3 个 token 的例子里，如果 $k_2$ 改成 $(0, 1)$，和 $k_1$ 垂直，$A$ 和 $u_2$ 会变成什么？**

??? success "参考答案"
    $A_{21} = \beta_2\,(k_2 \cdot k_1) = 0$，$u_2 = \beta_2 v_2 = (0, 1)$：互相垂直的 key 写入时互不抵消，各存各的。这也是正交的 key 能把表存满而不出错的原因。

**这一章新建的文件**：`qwen/chunk.py`。
