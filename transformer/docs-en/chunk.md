# 8. Gated DeltaNet: chunked parallelism

<p class="lead">One step of the previous chapter's recurrence does a few small 128 × 128 operations, and T tokens mean T iterations of a Python loop; a GPU is nowhere near busy — this is exactly why linear attention was "fast in theory, slow in practice" for years. Cut the sequence into chunks of 64 tokens: the 64 "decay + write" steps within a chunk combine into one lower-triangular linear system, and a single triangular solve gives what each token really writes; between chunks, a single table is passed along just like the recurrence. The result is the same as the token-by-token version and the speed differs by an order of magnitude.</p>

!!! abstract "After this chapter you can"
    1. Say why the token-by-token recurrence is slow on a GPU: every step waits for the previous one, and every step is only a tiny computation.
    2. Work 3 tokens by hand: how the dependencies within a chunk become one lower-triangular system $(I + A)U = \beta \odot V$ that gives every token's real write in one go.
    3. Read `chunked_delta_rule`: within a chunk a triangular solve plus matrix products, between chunks only one table; and say what a chunk length of 64 trades off.

!!! note "What this chapter runs"
    The 3-token hand-worked example `ch08_toy.py` (1 second), and `ch08_chunk.py`, which checks the answers and compares speed (about 15 seconds).

@@files@@

## Why chunk at all {#为什么要分块}

In the previous chapter's recurrence, step t needs the table as step t − 1 left it, so every step waits for the one before; and every step is only a few small 128 × 128 operations. What a GPU is good at is one very large matrix multiplication, so feeding it step by step keeps it waiting most of the time. Training and prefill process thousands of tokens at once, and this becomes the bottleneck.

Could everything run in parallel? The dependencies of the whole sequence can be written out at once, but the intermediates are $T \times T$, back to the quadratic cost of attention. The compromise is chunking: 64 tokens per chunk; **within a chunk**, the 64 steps of dependencies are written as one system and solved at once; **between chunks**, only one table is passed along, just like the recurrence.

## 3 tokens by hand {#3-个-token-手算一遍}

Ignore the decay for now, and start from an empty table. Call what token t really writes into the table $u_t = \beta_t\,(v_t - S_{t-1}^\top k_t)$. The table is the sum of the earlier writes, $S_{t-1} = \sum_{j<t} k_j u_j^\top$; substituting it in:

$$
u_t = \beta_t \Big(v_t - \sum_{j<t} (k_t \cdot k_j)\, u_j\Big)
$$

With 3 tokens, $k = (1, 0), (0.6, 0.8), (0, 1)$, $v = (1, 0), (0, 1), (2, 2)$, $\beta = 1, 1, 0.5$:

- $u_1 = \beta_1 v_1 = (1, 0)$
- $u_2 = \beta_2\,(v_2 - 0.6\,u_1) = (-0.6, 1)$
- $u_3 = \beta_3\,(v_3 - 0 \cdot u_1 - 0.8\,u_2) = 0.5 \times (2.48, 1.2) = (1.24, 0.6)$

$u_3$ needs $u_1$ and $u_2$ first, which is why the recurrence is slow. Move every term with a $u$ to the left:

$$
u_t + \beta_t \sum_{j<t} (k_t \cdot k_j)\, u_j = \beta_t v_t \qquad\Longrightarrow\qquad (I + A)\,U = \beta \odot V,\quad A_{tj} = \beta_t\,(k_t \cdot k_j)\ (j < t)
$$

$I + A$ is lower triangular, so one `solve_triangular` gives $u_1, u_2, u_3$ at the same time. With $U$ in hand, everything else is matrix multiplication: the output $O = (QK^\top \text{ keeping only the lower triangle})\,U$, and the table at the end of the chunk $S = K^\top U$.

![Figure: a 3-token example of chunked parallelism: on the left the recurrence, one token waiting for the next; on the right the dependencies written as a lower-triangular system (I + A)U = β⊙V that gives u₁, u₂, u₃ at once; between chunks only one table is passed](assets/figures/chunk-example.svg){.aig-svg}

```python title="ch08_toy.py"
"""第 8 章（一）：3 个 token 手算分块：先逐 token 递推，再把同一件事写成一个下三角方程组，一次解出来，对答案"""
import torch

r = lambda t: [round(x, 3) for x in t.tolist()]
K = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])           # keys of the 3 tokens (already unit length)
V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])           # value
beta = torch.tensor([1.0, 1.0, 0.5])                              # write strength
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])           # query
print("1. 逐 token 递推（先不管衰减，表从空的开始）")
S = torch.zeros(2, 2)
for t in range(3):
    pred = S.T @ K[t]                                             # what the table currently thinks k_t maps to
    u = beta[t] * (V[t] - pred)                                   # what this token actually writes
    S = S + torch.outer(K[t], u)
    print(f"   token {t + 1}：表里预测 {r(pred)}，真正写入 u{t + 1} = {r(u)}，输出 {r(S.T @ Q[t])}")
print(f"   最后的表：{[r(row) for row in S]}")
print("2. 分块：三个 u 一次解出来")
A = torch.tril(beta[:, None] * (K @ K.T), -1)                     # A[t, j] = β_t · (k_t · k_j), keeping only j < t
print(f"   A = {[r(row) for row in A]}")
U = torch.linalg.solve_triangular(torch.eye(3) + A, beta[:, None] * V, upper=False)   # solve (I + A) U = β ⊙ V
print(f"   解 (I + A) U = β ⊙ V 得 U = {[r(row) for row in U]}")
print(f"   输出 = (Q Kᵀ 只留下三角，含对角线) · U = {[r(row) for row in torch.tril(Q @ K.T) @ U]}")
print(f"   最后的表 = Kᵀ · U = {[r(row) for row in K.T @ U]}")
```

```text title="output"
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

The two algorithms agree on every number: the $u_t$ the recurrence computes one at a time are exactly the rows of the $U$ the system gives in one go.

## The formula: a chunk of 64 tokens, with decay {#公式一块-64-个-token加上衰减}

Ignore the decay for a moment. A chunk has 64 tokens; starting from the chunk-start state $S_0$, write 64 times:

$$
S_t = S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top = S_{t-1}\,(I - \beta_t k_t k_t^\top) + \beta_t k_t v_t^\top
$$

Every step is "multiply by an $(I - \beta k k^\top)$ and add a rank-one matrix". Unrolling 64 steps, the chunk-end state can be written as $S_{64} = S_0 \cdot \prod(\cdots) + K^\top U$, where each row $u_t$ of $U$ is the vector the $t$-th token **really writes** — it equals $\beta_t v_t$ minus the "cancellation" by the writes of the $t-1$ tokens before it. That cancellation is a lower-triangular system:

$$
(I + A)\,U = \beta \odot V, \qquad A_{ij} = \beta_i\, k_i^\top k_j \ (j < i)
$$

It is solved with one `solve_triangular` (a 64 × 64 lower-triangular matrix per head) instead of a 64-step loop. With $U$ in hand, every position's output within the chunk = reading the chunk-start state ($Q S_0$) + reading the new writes within the chunk ($(QK^\top)_{\text{causal}} U$), two matrix products. With the decay, every term is multiplied by its cumulative decay factor, which is what the `G.exp()` expressions in the code are. In the literature this transformation is called the WY representation / UT transform, and `chunk_gated_delta_rule` in the FlashLinearAttention library is its CUDA version.

![Figure: chunked parallelism: one triangular solve plus two matrix products within a chunk, one table passed between chunks](assets/figures/chunked.svg){.aig-svg}

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
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))                   # the memory table and intermediates are always fp32 (bf16 cannot hold 64 accumulated steps)
    q, k = l2norm(q), l2norm(k)
    q = q / dk ** 0.5
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))                               # [B, heads, T, d]
    g, beta = g.transpose(1, 2), beta.transpose(1, 2)                               # [B, heads, T]
    pad = (chunk - T % chunk) % chunk                                               # pad to a multiple of chunk
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (T + pad) // chunk
    q, k, v = (t.reshape(B, n_head, n, chunk, -1) for t in (q, k, v))               # [B, heads, chunks, chunk length, d]
    g, beta = g.reshape(B, n_head, n, chunk), beta.reshape(B, n_head, n, chunk)

    G = g.cumsum(-1)                                                                # cumulative decay within the chunk (log): total decay from the chunk start to token t
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    decay = (G[..., :, None] - G[..., None, :]).masked_fill(upper, float("-inf")).exp()   # decay[i, j] = decay from j to i (0 when j > i)
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    A = (k_beta @ k.transpose(-1, -2)) * decay                                      # how much each pair of tokens in the chunk cancels each other's writes
    u = torch.linalg.solve_triangular(A, v_beta, upper=False, unitriangular=True)   # solve the lower-triangular system: what each token really writes (before subtracting the chunk-start state's prediction)
    w = torch.linalg.solve_triangular(A, k_beta * G.exp()[..., None], upper=False, unitriangular=True)   # coefficients for reading predictions from the chunk-start state
    attn = (q @ k.transpose(-1, -2)) * decay                                        # the "attention" of q over k within the chunk
    q_decay = q * G.exp()[..., None]                                                # when reading the chunk-start state, q carries the decay from the chunk start to itself
    k_decay = k * (G[..., -1:] - G).exp()[..., None]                                # when writing the state, k carries the decay from itself to the chunk end
    chunk_decay = G[..., -1].exp()                                                  # total decay of the whole chunk

    if S is None:
        S = q.new_zeros(B, n_head, dk, dv)
    out = torch.zeros_like(v)
    for i in range(n):                                                              # between chunks: only S is passed, like the recurrence
        u_i = u[:, :, i] - w[:, :, i] @ S                                           # subtract what the chunk-start state already predicts: the real delta
        out[:, :, i] = q_decay[:, :, i] @ S + attn[:, :, i] @ u_i                   # output = read the chunk-start state + read the new writes within the chunk
        S = S * chunk_decay[:, :, i, None, None] + k_decay[:, :, i].transpose(-1, -2) @ u_i   # decay the whole chunk once, then add the chunk's writes in one go
    out = out.reshape(B, n_head, n * chunk, dv)[:, :, :T].transpose(1, 2).contiguous()   # drop the padding, back to [B, T, heads, dv]
    return out.to(dtype), S
```

## Run it: check the answers, compare the speed {#跑起来对答案比速度}

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
for T in (5, 64, 65, 512, 2048):                                     # 1. random input: the two versions match element by element, timed on the side
    q, k, v = (torch.randn(B, T, H, d) for _ in range(3))
    g, beta = -torch.rand(B, T, H) * 0.5, torch.rand(B, T, H)
    t = time.time()
    o1, S1 = delta_rule(q, k, v, g, beta)
    t1 = time.time() - t
    t = time.time()
    o2, S2 = chunked_delta_rule(q, k, v, g, beta)
    t2 = time.time() - t
    print(f"T={T:4d}：输出最大差 {diff(o1, o2):.1e}，末状态最大差 {diff(S1, S2):.1e}；递推 {t1 * 1000:6.0f} ms，分块 {t2 * 1000:5.0f} ms")

path = find_model()                                                  # 2. one real linear-attention layer over a long input: recurrent vs. chunked
cfg = Config.load(path)
W = load_weights(path)
tk = Tokenizer.from_file(str(path / "tokenizer.json"))
ids = torch.tensor([tk.encode("。".join(["今天天气不错，我们去公园"] * 40)).ids])
h = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")(load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")(ids))
dn = load(GatedDeltaNet(cfg), W, "layers.0.linear_attn.")
outs = {}
for name, rule in (("递推", delta_rule), ("分块", chunked_delta_rule)):
    qwen.deltanet.delta_rule = rule                                  # chapter 7's module calls the module-level delta_rule; swap in the chunked version here (chapter 10 changes it properly)
    t = time.time()
    outs[name] = dn(h)
    print(f"{ids.shape[1]} 个 token 过一层线性注意力，{name}：{(time.time() - t) * 1000:.0f} ms")
print(f"两种算法在这一层上的输出最大差：{diff(outs['递推'], outs['分块']):.1e}")
```

```text title="output"
T=   5：输出最大差 2.2e-08，末状态最大差 1.2e-07；递推      4 ms，分块     5 ms
T=  64：输出最大差 5.2e-08，末状态最大差 3.6e-07；递推     20 ms，分块     3 ms
T=  65：输出最大差 3.4e-08，末状态最大差 3.0e-07；递推     15 ms，分块     5 ms
T= 512：输出最大差 4.5e-08，末状态最大差 7.2e-07；递推    111 ms，分块    13 ms
T=2048：输出最大差 7.1e-08，末状态最大差 6.6e-07；递推    428 ms，分块    49 ms
279 个 token 过一层线性注意力，递推：283 ms
279 个 token 过一层线性注意力，分块：30 ms
两种算法在这一层上的输出最大差：3.6e-07
```

The two versions agree element by element at every length (differences around 1e-7, purely the order of summation); one real linear-attention layer over 279 tokens is nearly ten times faster chunked. The Hugging Face implementation uses this very algorithm for prefill (without `flash-linear-attention` installed it falls back to a PyTorch implementation like ours).

## Chapters 7 and 8 in one picture {#第-78-章一图看懂}

![Figure: Gated DeltaNet in one picture: why it exists, the memory table as a dictionary, three steps of improvement, the four steps per token, Qwen3.5's shell, the table blurs so it is paired 3:1 with full attention, chunked parallelism, and where it came from](assets/figures/gdn-poster.svg){.aig-svg}

## Common questions {#常见疑问}

**Why is the chunk length 64?**

Too small a chunk makes the matrix multiplications within it too small to keep the GPU busy, and leaves many serial steps between chunks; too large a chunk makes $A$ chunk length × chunk length, so the triangular solve and the intermediates grow quadratically. 64 is a common compromise, and Hugging Face's PyTorch implementation of Qwen3.5 also defaults to 64.

**Does decode use chunks or the recurrence?**

Decode receives one token at a time, so there is no chunk to parallelize, and one step of the recurrence is enough; prefill and training receive a long stretch at once and use chunks. Once chapter 10 adds the state, both paths share the same table.

**With decay, what changes in the formula?**

Every term is also multiplied by the cumulative decay between the two tokens. $G$ is the prefix sum of the log-decay within the chunk, and from token j to token i the total decay is $e^{G_i - G_j}$; `decay`, `q_decay`, `k_decay` and `chunk_decay` in the code are all this.

**Is this the same chunking as FlashAttention's?**

No. FlashAttention chunks so that the $T \times T$ score matrix is never written back to memory; it still computes full attention, and the work is still quadratic. Here, after chunking, only one fixed-size table passes between chunks, and the work is linear.

!!! interview "How to explain it"
    To explain "why linear attention can be trained in parallel": the recurrent form makes every step depend on the previous one, but the 64 steps within a chunk can be written as one lower-triangular system plus two matrix products, and only the state is passed between chunks; this is the same idea as Mamba-2's SSD and FlashLinearAttention's chunk algorithm. Be clear about "why chunk rather than parallelize everything": fully parallel intermediates are $T \times T$, back to attention's quadratic cost; a chunk length of 64 is the compromise between parallelism and the size of the intermediates.

## Summary {#小结}

- [x] Within a chunk: one triangular solve of $(I + A)U = \beta \odot V$ gives the real writes; the output = read the chunk-start state + read the writes within the chunk; between chunks: pass one table.
- [x] Element by element the same as the recurrence, an order of magnitude faster on long inputs; the production implementations (FlashLinearAttention, HF's prefill) use this very algorithm.

## Questions {#思考题}

**1. With a sequence of 4096, a chunk length of 64 and 16 heads, how many triangular systems does one layer solve? How large is each?**

??? success "Answer"
    4096 ÷ 64 = 64 chunks per head, 1024 for 16 heads, each a 64 × 64 lower-triangular system. They are independent and can be solved as one batch; the only serial part is the 64 steps of passing the table between chunks.

**2. What if the chunk length were the whole sequence T? And 1?**

??? success "Answer"
    T: there is only one chunk and $A$ is $T \times T$, back to quadratic compute and memory. 1: one token per chunk, nothing left to parallelize, and it degenerates into the token-by-token recurrence.

**3. In the 3-token example, if $k_2$ became $(0, 1)$, perpendicular to $k_1$, what would $A$ and $u_2$ be?**

??? success "Answer"
    $A_{21} = \beta_2\,(k_2 \cdot k_1) = 0$ and $u_2 = \beta_2 v_2 = (0, 1)$: perpendicular keys do not cancel each other when they write; each stores its own value. This is also why perpendicular keys can fill the table without errors.

**File created in this chapter**: `qwen/chunk.py`.
