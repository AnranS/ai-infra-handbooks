# 7. Gated DeltaNet: the token-by-token recurrence

<p class="lead">18 of the 24 layers are linear attention. It has no $T \times T$ matrix of "every token scoring every other"; instead each head keeps a <strong>fixed-size memory table</strong> $S$ (128 × 128) that is rewritten once per token by the delta rule — decay first, then write in "the difference between the new value and what the table already predicts". The output is the query looking the table up. However many tokens arrive, the table never grows, which is the root reason Qwen3.5 can hold a 262K context without its inference memory growing with length. This chapter writes the most direct token-by-token recurrence; the next one writes the fast version.</p>

!!! abstract "After this chapter you can"
    1. Explain the memory table as "a key → value dictionary": writing adds an outer product, reading is $S^\top q$.
    2. Say what each step from linear attention to DeltaNet to Gated DeltaNet changes and which problem it solves.
    3. Say why the table "blurs", and why Qwen3.5 pairs every 3 linear-attention layers with 1 full-attention layer.
    4. Write the token-by-token `delta_rule` and the `GatedDeltaNet` shell around it.

!!! note "What this chapter runs"
    A hand-worked example that needs no model, `ch07_toy.py` (about 2 seconds), and `ch07_deltanet.py` with real weights (about 15 seconds).

@@files@@

## The memory table: a key → value dictionary {#记忆表一本-key--value-的字典}

Chapter 6's attention keeps every token's key and value as they are, and reading compares the query with every key. Without the softmax, what is read is $\sum_j (q \cdot k_j)\, v_j$, which can be rewritten as

$$
\sum_j (q \cdot k_j)\, v_j \;=\; \Big(\sum_j k_j v_j^\top\Big)^{\top} q
$$

The bracket is a $d_k \times d_v$ table, called $S$. The context can be added into this table as it arrives, and reading it is one multiplication by $q$, with no need to go back over every token:

- **Write**: $S \leftarrow S + k\,v^\top$. The outer product $k\,v^\top$ means "store $v$ in the slot of $k$".
- **Read**: $o = S^\top q$. Whichever key $q$ resembles, mostly that key's value comes out.

By hand in two dimensions: after writing $k_1 = (1, 0) \to v_1 = (3, 1)$ and $k_2 = (0, 1) \to v_2 = (-1, 2)$, the table is $\begin{bmatrix} 3 & 1 \\ -1 & 2 \end{bmatrix}$; the first row is what $k_1$ stored and the second what $k_2$ stored. Reading with $q = (1, 0)$ gives exactly $(3, 1)$; reading with $q = (0.6, 0.8)$ gives $0.6 \times (3, 1) + 0.8 \times (-1, 2) = (1.0, 2.2)$, the two values mixed by similarity.

The size of this table does not depend on how long the context is: Qwen3.5 has one 128 × 128 table per head. That is "linear" attention: every new token costs a fixed amount of work, and the state does not grow.

## Step by step: linear attention → DeltaNet → Gated DeltaNet {#一步步改进线性注意力--deltanet--gated-deltanet}

A table that can only be added to has a flaw: write the same key again and the old and new values pile up. In the figure below the table already has $k_1 \to (3, 1)$, and now $k_1$ should become $(5, 0)$:

![Figure: from linear attention to Gated DeltaNet: linear attention adds the new value on top and gets (8, 1); DeltaNet looks the key up first and writes only the difference, overwriting it with (5, 0); Gated DeltaNet multiplies the whole table by α before writing the difference, so old memories that are not written again fade](assets/figures/delta-evolution.svg){.aig-svg}

1. **Linear attention**: $S \leftarrow S + k\,v^\top$ reads back $(3, 1) + (5, 0) = (8, 1)$, which is wrong.
2. **DeltaNet**: look $k$ up first to get what the table currently thinks, $S^\top k = (3, 1)$, and write in only the difference $(5, 0) - (3, 1) = (2, -1)$: $S \leftarrow S + \beta\, k\,(v - S^\top k)^\top$. With $\beta = 1$ this overwrites the slot with exactly $(5, 0)$; with $\beta = 0.5$ it goes only half way. This is one step of gradient descent on "$S^\top k$ should be close to $v$", with $\beta$ as the learning rate.
3. **Gated DeltaNet**: before writing, multiply the whole table by $\alpha \in (0, 1]$, then write the difference as DeltaNet does. $\alpha$ is computed from the current token, so the model decides how much to forget this time; old memories that are not written again (the $k_2$ row in the figure) slowly fade.

```python title="ch07_toy.py" ci="loose"
"""第 7 章（一）：在两维的记忆表上手算：写 = 外积相加，读 = Sᵀq；线性注意力 → DeltaNet → Gated DeltaNet 一步步改；最后看表能记多少"""
import torch

r = lambda t, n=3: [round(x, n) for x in t.tolist()]
k1, k2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
S = torch.outer(k1, torch.tensor([3.0, 1.0])) + torch.outer(k2, torch.tensor([-1.0, 2.0]))   # 1. write: add k vᵀ into the table
print(f"1. 写入 k1 → [3, 1]、k2 → [-1, 2] 之后，表 S = {[r(row) for row in S]}")
for q in (k1, k2, torch.tensor([0.6, 0.8])):                         #    read: o = Sᵀq
    print(f"   用 q = {r(q)} 去读：Sᵀq = {r(S.T @ q)}")

v_new = torch.tensor([5.0, 0.0])                                      # 2. now k1 should become [5, 0]
linear = S + torch.outer(k1, v_new)                                   #    linear attention: just add it on top
delta = S + torch.outer(k1, v_new - S.T @ k1)                         #    DeltaNet: look up k1 first, write only the difference (β = 1)
alpha = 0.5
gated = alpha * S + torch.outer(k1, v_new - (alpha * S).T @ k1)       #    Gated DeltaNet: multiply the whole table by α first, then write the difference
for name, M in (("线性注意力", linear), ("DeltaNet", delta), ("Gated DeltaNet（α = 0.5）", gated)):
    print(f"2. {name}：读 k1 得 {r(M.T @ k1)}，读 k2 得 {r(M.T @ k2)}")

for a in (1.0, 0.9, 0.5):                                             # 3. decay: old content that is not written again is multiplied by α at every step
    print(f"3. α = {a}：10 步后旧内容剩 {a ** 10:.3f}，50 步后剩 {a ** 50:.3g}")

torch.manual_seed(0)                                                  # 4. capacity: put n (k, v) pairs into one 128 × 128 table, then read each one back with its k
dk = 128


def fill(K, V):
    plain, dlt = torch.zeros(dk, dk), torch.zeros(dk, dk)
    for k, v in zip(K, V):
        plain += torch.outer(k, v)
        dlt += torch.outer(k, v - dlt.T @ k)
    err = lambda M, s: ((K[s] @ M - V[s]).norm(dim=-1) / V[s].norm(dim=-1)).mean().item()   # how far the value read back is from v (relative error)
    return err(plain, slice(None)), err(dlt, slice(None)), err(dlt, slice(-16, None)), err(dlt, slice(0, 16))


Q, _ = torch.linalg.qr(torch.randn(dk, dk))
p, d, _, _ = fill(Q, torch.randn(dk, dk))
print(f"4. 128 个互相垂直的 key：线性注意力误差 {p:.0%}，delta 规则 {d:.0%}")
for n in (256, 1024):
    p, d, last, first = fill(torch.nn.functional.normalize(torch.randn(n, dk), dim=-1), torch.randn(n, dk))
    print(f"   {n} 个随机 key：线性注意力误差 {p:.0%}，delta 规则 {d:.0%}（其中最后写的 16 个 {last:.0%}，最早写的 16 个 {first:.0%}）")
```

```text title="output"
1. 写入 k1 → [3, 1]、k2 → [-1, 2] 之后，表 S = [[3.0, 1.0], [-1.0, 2.0]]
   用 q = [1.0, 0.0] 去读：Sᵀq = [3.0, 1.0]
   用 q = [0.0, 1.0] 去读：Sᵀq = [-1.0, 2.0]
   用 q = [0.6, 0.8] 去读：Sᵀq = [1.0, 2.2]
2. 线性注意力：读 k1 得 [8.0, 1.0]，读 k2 得 [-1.0, 2.0]
2. DeltaNet：读 k1 得 [5.0, 0.0]，读 k2 得 [-1.0, 2.0]
2. Gated DeltaNet（α = 0.5）：读 k1 得 [5.0, 0.0]，读 k2 得 [-0.5, 1.0]
3. α = 1.0：10 步后旧内容剩 1.000，50 步后剩 1
3. α = 0.9：10 步后旧内容剩 0.349，50 步后剩 0.00515
3. α = 0.5：10 步后旧内容剩 0.001，50 步后剩 8.88e-16
4. 128 个互相垂直的 key：线性注意力误差 0%，delta 规则 0%
   256 个随机 key：线性注意力误差 143%，delta 规则 100%（其中最后写的 16 个 31%，最早写的 16 个 131%）
   1024 个随机 key：线性注意力误差 283%，delta 规则 131%（其中最后写的 16 个 33%，最早写的 16 个 145%）
```

Parts 1 and 2 are the hand calculation and the figure above. Part 3 is how fast things are forgotten: with $\alpha = 0.9$, 35% of the old content is left after 10 steps and only 0.5% after 50.

Part 4 is the table's capacity. A 128-dimensional key has at most 128 mutually perpendicular directions, and 128 pairs stored with such keys read back perfectly. But the context is usually far longer than 128 tokens, and keys are not perpendicular to each other: with 256 random (k, v) pairs, ordinary linear attention reads back with an error of 143% and the delta rule with 100%, but the 16 pairs written last are off by only 31% while the 16 written first are off by 131%. The delta rule leans towards remembering what was written recently, but in the end it only "remembers roughly". That is why Qwen3.5 pairs every 3 linear-attention layers with 1 full-attention layer: the cheap layers keep the gist, and full attention does the precise looking back, such as quoting an earlier sentence word for word.

## The formula {#公式}

For each head, the memory table is $S_t \in \mathbb{R}^{128 \times 128}$; the $t$-th token brings $k_t, v_t, q_t \in \mathbb{R}^{128}$ (already L2-normalized), a decay $\alpha_t = e^{g_t} \in (0, 1]$ and a write strength $\beta_t \in (0, 1)$:

$$
S_t = \alpha_t S_{t-1} + \beta_t\, k_t\,\big(v_t - \alpha_t S_{t-1}^\top k_t\big)^\top, \qquad o_t = S_t^\top q_t
$$

Read it: $\alpha_t S_{t-1}^\top k_t$ is "the value the decayed table thinks $k_t$ maps to" (the code decays first, then looks up); $v_t$ minus that is the **difference** (delta) to correct this time; $\beta_t$ decides how much of it to apply; $k_t (\cdot)^\top$ writes the correction back into the "row" of $k_t$. $\alpha_t$ makes the whole table slowly forget. Without the subtraction ($S_t = \alpha_t S_{t-1} + \beta_t k_t v_t^\top$) this would be ordinary linear attention / RetNet; the one subtraction the delta rule adds keeps a key that is written repeatedly from piling up and instead makes it converge to the new value — this is where DeltaNet beats the old linear attention.

![Figure: one step of the delta rule: decay, look up, write the difference, read out](assets/figures/delta-rule.svg){.aig-svg}

```python title="qwen/deltanet.py"
"""Gated DeltaNet：线性注意力层，24 层里的 18 层是它。

它没有"所有 token 两两打分"的注意力矩阵，而是每个头维护一张固定大小的记忆表 S（128×128）。
每来一个 token 就按 delta 规则改写一次：先按 g 衰减，再把"新值 v 和表里已有预测之差"按 β 写进去；
输出是 q 去表里查询。token 再多，表也不会变大——这就是它能撑 262K 上下文的原因。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen.norm import GatedRMSNorm, l2norm



def delta_rule(q, k, v, g, beta, S=None):
    """逐 token 递推。q, k: [B, T, 头, dk]，v: [B, T, 头, dv]；g（对数衰减，≤0）和 beta（写入强度，0~1）: [B, T, 头]。
    S 是开始时的记忆表（没有就从空表开始）；返回输出和算完之后的表。"""
    B, T, n_head, dk = q.shape
    dtype = v.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))   # the memory table and intermediates are always fp32; bf16 cannot hold hundreds of accumulated steps
    q, k = l2norm(q), l2norm(k)                                   # q and k are normalized to unit length first
    q = q / dk ** 0.5
    if S is None:
        S = q.new_zeros(B, n_head, dk, v.shape[-1])               # one 128×128 memory table per head, empty at the start
    out = torch.empty_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, :, None, None]                   # 1. decay: multiply the whole table by a number in (0, 1], forgetting a little
        k_t, v_t = k[:, t], v[:, t]                               # [B, heads, dk], [B, heads, dv]
        pred = torch.einsum("bhk,bhkv->bhv", k_t, S)              # 2. look up with k: what the table thinks v should be
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, (v_t - pred) * beta[:, t, :, None])   # 3. write only the difference (delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)     # 4. look up with q to get this token's output
    return out.to(dtype), S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.hidden_size
        self.n_k, self.n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.n_k * self.dk, self.n_v * self.dv            # 2048, 2048
        conv_dim = 2 * self.key_dim + self.value_dim                                       # q, k and v concatenated for the convolution: 6144
        self.in_proj_qkv = nn.Linear(H, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(H, self.value_dim, bias=False)                          # output gate
        self.in_proj_b = nn.Linear(H, self.n_v, bias=False)                                # β: one number per head
        self.in_proj_a = nn.Linear(H, self.n_v, bias=False)                                # the input-dependent part of the decay
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, cfg.linear_conv_kernel_dim, groups=conv_dim,
                                padding=cfg.linear_conv_kernel_dim - 1, bias=False)        # depthwise causal short convolution: each channel only sees its own previous 3 tokens
        self.dt_bias = nn.Parameter(torch.ones(self.n_v))
        self.A_log = nn.Parameter(torch.zeros(self.n_v))
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, H, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)                                          # [B, 6144, T]
        qkv = F.silu(self.conv1d(qkv)[..., :T]).transpose(1, 2)                            # causal convolution (cut off the "future" produced by the right padding) + SiLU
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q, k, v = q.view(B, T, self.n_k, self.dk), k.view(B, T, self.n_k, self.dk), v.view(B, T, self.n_v, self.dv)
        if self.n_v > self.n_k:                                                            # bigger models have more value heads than key heads: repeat q and k
            q, k = q.repeat_interleave(self.n_v // self.n_k, 2), k.repeat_interleave(self.n_v // self.n_k, 2)
        beta = self.in_proj_b(x).sigmoid()                                                 # [B, T, heads], in (0, 1)
        g = -self.A_log.exp() * F.softplus(self.in_proj_a(x) + self.dt_bias)               # log decay, always ≤ 0
        out, _ = delta_rule(q, k, v, g, beta)                                              # the memory table is not kept in this chapter; chapter 10 uses it
        out = self.norm(out, self.in_proj_z(x).view(B, T, self.n_v, self.dv))             # gated normalization per head
        return self.out_proj(out.reshape(B, T, -1))
```

The four steps in `delta_rule` match the formula one to one. `GatedDeltaNet` around it is the "shell" that turns the 1024 dimensions of the residual stream into those quantities, with several parts that Llama-style models do not have:

| Part | Shape | What it does |
| --- | --- | --- |
| `in_proj_qkv` + `conv1d` | 1024 → 6144, then a depthwise causal convolution with a window of 4 | q, k and v are not projected directly: each channel first sees its own previous 3 tokens (local context), then passes through SiLU |
| `in_proj_b` → sigmoid | 1024 → 16 | one $\beta$ per head: how strongly this token writes |
| `in_proj_a`, `dt_bias`, `A_log` | 1024 → 16 | $g = -e^{A\_log}\cdot\mathrm{softplus}(a + \text{dt\_bias})$, always $\le 0$, so the decay $e^g \in (0, 1]$; `A_log` is each head's fixed "forgetting speed" |
| `in_proj_z` + `GatedRMSNorm` | 1024 → 2048 | the output is normalized per head and then multiplied by the gate $\mathrm{silu}(z)$ |
| `out_proj` | 2048 → 1024 | back to the residual stream |

`conv1d` with `padding = 3` plus the `[..., :T]` cut is the causal convolution: the output at position $t$ uses only the inputs $t-3 \ldots t$. `groups = 6144` means every channel convolves on its own, with only 6144 × 4 parameters.

## Run it: the table does not grow with length {#跑起来表不随长度变大}

```python title="ch07_deltanet.py" ci="loose"
"""第 7 章：线性注意力（逐 token 递推版）。看记忆表的大小不随长度变、衰减和 β 的取值、delta 规则的性质"""
import os
import time

import torch
from tokenizers import Tokenizer

from qwen.deltanet import GatedDeltaNet, delta_rule
from qwen.embedding import Embedding
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
dn = load(GatedDeltaNet(cfg), W, "layers.0.linear_attn.")
n = sum(p.numel() for p in dn.parameters())
print(f"一层线性注意力 {n / 1e6:.1f}M 参数：in_proj_qkv {dn.in_proj_qkv.weight.numel() / 1e6:.1f}M，conv1d 只有 {dn.conv1d.weight.numel()} 个")
print(f"每个头固定的遗忘速度 exp(A_log)：最小 {dn.A_log.exp().min():.2f}，最大 {dn.A_log.exp().max():.2f}（越大忘得越快）")

tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # build layer 0's real input with the parts from chapters 2 and 3
emb, norm = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens."), load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")
with torch.no_grad():
    for text in ("今天天气不错，我们去公园", "。".join(["今天天气不错，我们去公园"] * 20)):   # 1. a short and a long passage: time grows with length, the memory table does not
        h = norm(emb(torch.tensor([tk.encode(text).ids])))
        t = time.time()
        y = dn(h)
        print(f"  {h.shape[1]:3d} 个 token：输出 {tuple(y.shape)}，{(time.time() - t) * 1000:.0f} ms")
    print(f"记忆表：每个头 [{dn.dk}, {dn.dv}]，{dn.n_v} 个头共 {dn.n_v * dn.dk * dn.dv * 4 / 2**20:.1f} MB，和 token 数无关；全注意力的 KV cache 每个 token 要加 {2 * cfg.num_key_value_heads * cfg.head_dim * 4 / 1024:.0f} KB")
    h = norm(emb(torch.tensor([tk.encode("今天天气不错，我们去公园").ids])))
    beta = dn.in_proj_b(h).sigmoid()                                # 2. decay and β on real input
    g = -dn.A_log.exp() * torch.nn.functional.softplus(dn.in_proj_a(h) + dn.dt_bias)
    print(f"第 0 层对这 6 个 token：衰减 exp(g) 的范围 [{g.exp().min():.2f}, {g.exp().max():.2f}]，β 的范围 [{beta.min():.2f}, {beta.max():.2f}]")

torch.manual_seed(0)                                                # 3. a property of the delta rule: writing the same (key, value) pair repeatedly converges to v; plain linear attention keeps piling up
k = torch.nn.functional.normalize(torch.randn(1, 1, 1, 128), dim=-1).expand(1, 6, 1, 128)
v = torch.randn(1, 1, 1, 128).expand(1, 6, 1, 128)
out, S = delta_rule(k, k, v, torch.zeros(1, 6, 1), torch.full((1, 6, 1), 0.5))   # q = k, no decay, β = 0.5
plain = torch.cumsum(0.5 * v, dim=1)                                           # the version without "subtract the prediction": S += β k v^T, reading gives t × β × v
for t in range(6):
    print(f"  写入 {t + 1} 次后读出的值和 v 的距离：delta 规则 {(out[0, t, 0] * 128 ** 0.5 - v[0, t, 0]).norm():6.3f}，普通线性注意力 {(plain[0, t, 0] - v[0, t, 0]).norm():6.3f}")
```

```text title="output"
一层线性注意力 10.5M 参数：in_proj_qkv 6.3M，conv1d 只有 24576 个
每个头固定的遗忘速度 exp(A_log)：最小 0.00，最大 1.29（越大忘得越快）
    6 个 token：输出 (1, 6, 1024)，9 ms
  139 个 token：输出 (1, 139, 1024)，104 ms
记忆表：每个头 [128, 128]，16 个头共 1.0 MB，和 token 数无关；全注意力的 KV cache 每个 token 要加 4 KB
第 0 层对这 6 个 token：衰减 exp(g) 的范围 [0.01, 1.00]，β 的范围 [0.13, 1.00]
  写入 1 次后读出的值和 v 的距离：delta 规则  4.673，普通线性注意力  4.673
  写入 2 次后读出的值和 v 的距离：delta 规则  2.336，普通线性注意力  0.000
  写入 3 次后读出的值和 v 的距离：delta 规则  1.168，普通线性注意力  4.673
  写入 4 次后读出的值和 v 的距离：delta 规则  0.584，普通线性注意力  9.345
  写入 5 次后读出的值和 v 的距离：delta 规则  0.292，普通线性注意力 14.018
  写入 6 次后读出的值和 v 的距离：delta 规则  0.146，普通线性注意力 18.691
```

The first two lines: 6 tokens take 9 ms and 139 tokens 104 ms — time grows with length, but the memory table is always 16 tables of 128 × 128, 1 MB; with full attention every token appends 4 KB to the KV cache, so a million tokens would be 4 GB — that is what the word "linear" means. The middle line shows the two gates on real input: the decay runs from 0.01 (this head looks almost only at the current token) to 1.00 (it almost never forgets), and $\beta$ from 0.13 to 1.00. The small experiment at the end is the essence of the delta rule: writing the same (key, value) pair six times, the distance between the value read back and $v$ halves each time ($\beta = 0.5$ only goes half way) and converges to $v$; ordinary linear attention lacks the subtraction, so every write piles on another $\beta v$ — the second write happens to land exactly on $v$, and it drifts further away after that.

## Where it came from {#来历}

| When | What |
| --- | --- |
| 2021 | Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers*: linear attention writes its table with the delta rule, which is DeltaNet |
| 2024 | Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (NeurIPS 2024): DeltaNet can be trained in parallel over chunks of the sequence, the algorithm of chapter 8 |
| late 2024 | *Gated Delta Networks* (ICLR 2025): adds an input-dependent decay $\alpha$ |
| 2025 on | Qwen3-Next and Qwen3.5 mix it with full attention at 3:1; the KDA of Kimi Linear is a variant of it |

## Common questions {#常见疑问}

**What is the difference between the memory table and the KV cache?**

The KV cache keeps every token's key and value as they are, as much as the context is long, and reading it is precise; the memory table stacks every write into one fixed-size table that does not grow, but what it reads back is "mixed". Chapter 10 puts the two kinds of state side by side.

**Why are q and k normalized to unit length first?**

The delta rule reads $S^\top k$ and writes the difference multiplied by $k$. When $k$ has unit length, $\beta = 1$ overwrites the slot with exactly the new value; a $k$ that is too long overshoots and one that is too short falls short, and over hundreds of steps the scale of the table drifts. Normalizing $q$ and then dividing it by $\sqrt{d_k}$ keeps the scale of what is read stable too.

**Are $\alpha$ and $\beta$ the same thing as chapter 6's gate?**

Same idea: switches between 0 and 1 computed from the current token. They control different things: $\alpha$ how much to forget this time, $\beta$ how strongly to write this time, and the `silu(z)` gate before the output how much of what was read gets through, the last one being the closest to chapter 6's output gate.

**If the table blurs, why use it at all?**

It is cheap: the compute per token and the size of the state are constant, however long the context. The 6 full-attention layers pay for what it loses.

!!! interview "How to explain it"
    When explaining linear attention, first explain that "the state has a fixed size": attention's KV cache grows with length, the memory table does not, so memory and the compute per token are constant over long contexts. Then the difference between the delta rule and ordinary linear attention: what is written is "the new value minus what the table already predicts", so a key that recurs does not accumulate, and the table's capacity is used more economically. Finally the gating: the decay $e^g$ lets it forget, $\beta$ lets it write selectively — both gates are computed from the input. Mentioning that "Qwen3.5's 18 : 6 mix uses 6 full-attention layers to make up for the linear layers' lack of precise recall" completes the picture.

## Summary {#小结}

- [x] Gated DeltaNet keeps one 128 × 128 memory table per head: decay → look up with k → write the difference × β → look up with q.
- [x] The shell has a short convolution, a $\beta$ projection, a decay projection (`A_log`, `dt_bias`) and a gated normalization.
- [x] The memory table is 16 × 128 × 128 = 1 MB, independent of the token count; on real input the decay runs 0.01–1.00 and $\beta$ 0.13–1.00; all the parts exist, and chapter 9 assembles them.

## Questions {#思考题}

**1. One layer's memory tables for 16 heads take 1 MB (fp32), and full attention's KV cache takes 4 KB per token (fp32). Beyond what context length does one full-attention layer's KV cache become larger than one linear-attention layer's state?**

??? success "Answer"
    1 MB ÷ 4 KB = 256 tokens. Beyond 256 tokens the full-attention layer stores more, and it keeps growing; the linear-attention layer is always 1 MB.

**2. With $\beta = 1$ and a unit-length $k$, if you write $(k, v)$ and immediately read with the same $k$, what comes back? And with $\beta = 0.5$?**

??? success "Answer"
    $\beta = 1$: exactly $v$; the old value is overwritten. $\beta = 0.5$: the point halfway between the old value and $v$, half the distance to $v$. The last experiment of `ch07_deltanet.py` writes the same (k, v) pair six times, and the distance halves every time.

**3. What happens if a head's $\alpha$ is always 1? And if it is always close to 0?**

??? success "Answer"
    Always 1: it never forgets; old memories keep piling up and the table gets more and more crowded and blurred. Always close to 0: the table is emptied before every step, what is read back is almost only what the current token itself wrote, and the layer degenerates into looking only at the current token. In the real model the heads' $\alpha$ differ a lot (measured from 0.01 to 1.00 in layer 0 in the main text): some heads keep long-term memory, others only look at what is in front of them.

**File created in this chapter**: `qwen/deltanet.py` (chapter 10 adds the state and mask).
