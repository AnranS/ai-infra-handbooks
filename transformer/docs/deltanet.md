# （七）Gated DeltaNet：逐 token 递推

<p class="lead">24 层里的 18 层是线性注意力。它没有"所有 token 两两打分"的 $T \times T$ 矩阵，而是每个头维护一张<strong>固定大小的记忆表</strong> $S$（128 × 128）：每来一个 token 就按 delta 规则改写一次——先衰减，再把"新值和表里已有预测之差"写进去；输出是 query 去表里查。token 再多，表也不会变大，这就是 Qwen3.5 能撑 262K 上下文、推理时显存不随长度涨的根本原因。这一章先写最直白的逐 token 递推版，下一章再写快的。</p>

!!! abstract "学完这章你能"
    1. 用"一本 key → value 的字典"讲清记忆表：写是外积相加，读是 $S^\top q$。
    2. 说清线性注意力、DeltaNet、Gated DeltaNet 每一步改了什么、各解决什么问题。
    3. 说出表为什么会"糊"，以及 Qwen3.5 为什么每 3 层线性注意力配 1 层全注意力。
    4. 写出逐 token 递推的 `delta_rule` 和外面那层 `GatedDeltaNet`。

!!! note "这一章跑什么"
    不用模型的手算例子 `ch07_toy.py`（约 2 秒），和用真实权重的 `ch07_deltanet.py`（约 15 秒）。

@@files@@

## 记忆表：一本 key → value 的字典

第 6 章的注意力把每个 token 的 key、value 原样存着，读的时候拿 query 和每个 key 比一遍。去掉 softmax，读出来的是 $\sum_j (q \cdot k_j)\, v_j$，可以改写成

$$
\sum_j (q \cdot k_j)\, v_j \;=\; \Big(\sum_j k_j v_j^\top\Big)^{\top} q
$$

括号里是一张 $d_k \times d_v$ 的表，记作 $S$。前文可以边来边加进这张表，用的时候拿 $q$ 一乘就读出来，不用再回头看每个 token：

- **写**：$S \leftarrow S + k\,v^\top$。外积 $k\,v^\top$ 就是"把 $v$ 存到 $k$ 这一格"。
- **读**：$o = S^\top q$。$q$ 和哪个 key 像，就主要读出那个 key 存的 value。

两维手算一遍：写入 $k_1 = (1, 0) \to v_1 = (3, 1)$、$k_2 = (0, 1) \to v_2 = (-1, 2)$ 之后，表是 $\begin{bmatrix} 3 & 1 \\ -1 & 2 \end{bmatrix}$，第一行是 $k_1$ 存的、第二行是 $k_2$ 存的。用 $q = (1, 0)$ 去读，正好读出 $(3, 1)$；用 $q = (0.6, 0.8)$ 去读，读出 $0.6 \times (3, 1) + 0.8 \times (-1, 2) = (1.0, 2.2)$，按相似度把两个 value 混在一起。

这张表多大，和前文多长无关：Qwen3.5 每个头一张 128 × 128 的表。这就是"线性"注意力：每来一个 token 做固定量的计算，状态不长大。

## 一步步改进：线性注意力 → DeltaNet → Gated DeltaNet

只会往表里加，有个毛病：同一个 key 再写一次，新旧会叠在一起。下图里表已有 $k_1 \to (3, 1)$，现在 $k_1$ 要改成 $(5, 0)$：

![图：从线性注意力到 Gated DeltaNet：线性注意力把新值直接加上去得到 (8, 1)；DeltaNet 先查表、只写差值，覆盖成 (5, 0)；Gated DeltaNet 先把整张表乘 α 再写差值，不再写的旧记忆慢慢淡掉](assets/figures/delta-evolution.svg){.aig-svg}

1. **线性注意力**：$S \leftarrow S + k\,v^\top$，读出 $(3, 1) + (5, 0) = (8, 1)$，错了。
2. **DeltaNet**：先用 $k$ 查表，得到表里现在认为的值 $S^\top k = (3, 1)$；只把差值 $(5, 0) - (3, 1) = (2, -1)$ 写进去：$S \leftarrow S + \beta\, k\,(v - S^\top k)^\top$。$\beta = 1$ 时正好覆盖成 $(5, 0)$；$\beta = 0.5$ 时只走一半。这等于对"$S^\top k$ 要接近 $v$"做一步梯度下降，$\beta$ 是学习率。
3. **Gated DeltaNet**：写之前先把整张表乘一个 $\alpha \in (0, 1]$，再按 DeltaNet 写差值。$\alpha$ 由当前 token 算出来，模型自己决定这次忘多少；没再被写的旧记忆（图里 $k_2$ 那一行）就慢慢淡掉。

```python title="ch07_toy.py" ci="loose"
"""第 7 章（一）：在两维的记忆表上手算：写 = 外积相加，读 = Sᵀq；线性注意力 → DeltaNet → Gated DeltaNet 一步步改；最后看表能记多少"""
import torch

r = lambda t, n=3: [round(x, n) for x in t.tolist()]
k1, k2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
S = torch.outer(k1, torch.tensor([3.0, 1.0])) + torch.outer(k2, torch.tensor([-1.0, 2.0]))   # 1. 写：k vᵀ 加进表里
print(f"1. 写入 k1 → [3, 1]、k2 → [-1, 2] 之后，表 S = {[r(row) for row in S]}")
for q in (k1, k2, torch.tensor([0.6, 0.8])):                         #    读：o = Sᵀq
    print(f"   用 q = {r(q)} 去读：Sᵀq = {r(S.T @ q)}")

v_new = torch.tensor([5.0, 0.0])                                      # 2. 现在 k1 要改成 [5, 0]
linear = S + torch.outer(k1, v_new)                                   #    线性注意力：直接加上去
delta = S + torch.outer(k1, v_new - S.T @ k1)                         #    DeltaNet：先用 k1 查表，只写差值（β = 1）
alpha = 0.5
gated = alpha * S + torch.outer(k1, v_new - (alpha * S).T @ k1)       #    Gated DeltaNet：整张表先 × α，再写差值
for name, M in (("线性注意力", linear), ("DeltaNet", delta), ("Gated DeltaNet（α = 0.5）", gated)):
    print(f"2. {name}：读 k1 得 {r(M.T @ k1)}，读 k2 得 {r(M.T @ k2)}")

for a in (1.0, 0.9, 0.5):                                             # 3. 衰减：不再写的旧内容每一步乘一次 α
    print(f"3. α = {a}：10 步后旧内容剩 {a ** 10:.3f}，50 步后剩 {a ** 50:.3g}")

torch.manual_seed(0)                                                  # 4. 容量：一张 128 × 128 的表塞进 n 对 (k, v)，再用每个 k 读回来
dk = 128


def fill(K, V):
    plain, dlt = torch.zeros(dk, dk), torch.zeros(dk, dk)
    for k, v in zip(K, V):
        plain += torch.outer(k, v)
        dlt += torch.outer(k, v - dlt.T @ k)
    err = lambda M, s: ((K[s] @ M - V[s]).norm(dim=-1) / V[s].norm(dim=-1)).mean().item()   # 读回来的值和 v 差多少（相对误差）
    return err(plain, slice(None)), err(dlt, slice(None)), err(dlt, slice(-16, None)), err(dlt, slice(0, 16))


Q, _ = torch.linalg.qr(torch.randn(dk, dk))
p, d, _, _ = fill(Q, torch.randn(dk, dk))
print(f"4. 128 个互相垂直的 key：线性注意力误差 {p:.0%}，delta 规则 {d:.0%}")
for n in (256, 1024):
    p, d, last, first = fill(torch.nn.functional.normalize(torch.randn(n, dk), dim=-1), torch.randn(n, dk))
    print(f"   {n} 个随机 key：线性注意力误差 {p:.0%}，delta 规则 {d:.0%}（其中最后写的 16 个 {last:.0%}，最早写的 16 个 {first:.0%}）")
```

```text title="输出"
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

第 1、2 段就是上面的手算和那张图。第 3 段是"忘"的速度：$\alpha = 0.9$ 时，10 步后旧内容还剩 35%，50 步后只剩 0.5%。

第 4 段是表的容量。128 维的 key 最多有 128 个互相垂直的方向，用这样的 key 存 128 对，读回来一点不差。可前文往往远多于 128 个 token，key 也不会互相垂直：塞进 256 对随机的 (k, v)，普通线性注意力读回来的误差 143%，delta 规则 100%，但其中最后写的 16 对只差 31%、最早写的 16 对差 131%。delta 规则偏向记住最近写的，可终究只能"记个大概"。所以 Qwen3.5 每 3 层线性注意力配 1 层全注意力：便宜的层管大概，全注意力管精确回看，比如原样引用前文的一句话。

## 公式

对每个头，记忆表 $S_t \in \mathbb{R}^{128 \times 128}$，第 $t$ 个 token 带来 $k_t, v_t, q_t \in \mathbb{R}^{128}$（都已经 L2 归一化）、衰减 $\alpha_t = e^{g_t} \in (0, 1]$ 和写入强度 $\beta_t \in (0, 1)$：

$$
S_t = \alpha_t S_{t-1} + \beta_t\, k_t\,\big(v_t - \alpha_t S_{t-1}^\top k_t\big)^\top, \qquad o_t = S_t^\top q_t
$$

读它：$\alpha_t S_{t-1}^\top k_t$ 是"衰减之后的表里认为 $k_t$ 对应的值"（代码里先衰减、再查表）；$v_t$ 减去它，是这次要修正的**差值**（delta）；乘上 $\beta_t$ 决定改多少；$k_t (\cdot)^\top$ 把这个修正写回 $k_t$ 这一"行"。$\alpha_t$ 让整张表慢慢忘。如果没有减法（$S_t = \alpha_t S_{t-1} + \beta_t k_t v_t^\top$），这就是普通的线性注意力 / RetNet；delta 规则多出来的那一减，让同一个 key 被反复写入时不会越积越大，而是收敛到新值——这是 DeltaNet 比老式线性注意力强的地方。

![图：delta 规则的一步：衰减、查表、写入差值、读出](assets/figures/delta-rule.svg){.aig-svg}

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
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))   # 记忆表和中间量一律 fp32，bf16 攒不住几百步的累加
    q, k = l2norm(q), l2norm(k)                                   # q、k 先归一化到单位长度
    q = q / dk ** 0.5
    if S is None:
        S = q.new_zeros(B, n_head, dk, v.shape[-1])               # 每个头一张 128×128 的记忆表，初始为空
    out = torch.empty_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, :, None, None]                   # 1. 衰减：整张表乘一个 0~1 的数，忘掉一点旧的
        k_t, v_t = k[:, t], v[:, t]                               # [B, 头, dk]，[B, 头, dv]
        pred = torch.einsum("bhk,bhkv->bhv", k_t, S)              # 2. 用 k 查表：表里认为 v 应该是什么
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, (v_t - pred) * beta[:, t, :, None])   # 3. 只写入差值（delta）
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)     # 4. 用 q 查表得到这个 token 的输出
    return out.to(dtype), S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.hidden_size
        self.n_k, self.n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.n_k * self.dk, self.n_v * self.dv            # 2048, 2048
        conv_dim = 2 * self.key_dim + self.value_dim                                       # q、k、v 拼在一起做卷积：6144
        self.in_proj_qkv = nn.Linear(H, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(H, self.value_dim, bias=False)                          # 输出门
        self.in_proj_b = nn.Linear(H, self.n_v, bias=False)                                # β：每个头一个数
        self.in_proj_a = nn.Linear(H, self.n_v, bias=False)                                # 衰减里随输入变的那部分
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, cfg.linear_conv_kernel_dim, groups=conv_dim,
                                padding=cfg.linear_conv_kernel_dim - 1, bias=False)        # 深度可分的因果短卷积：每个通道只看自己前 3 个 token
        self.dt_bias = nn.Parameter(torch.ones(self.n_v))
        self.A_log = nn.Parameter(torch.zeros(self.n_v))
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, H, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)                                          # [B, 6144, T]
        qkv = F.silu(self.conv1d(qkv)[..., :T]).transpose(1, 2)                            # 因果卷积（截掉右边 padding 产生的"未来"）+ SiLU
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q, k, v = q.view(B, T, self.n_k, self.dk), k.view(B, T, self.n_k, self.dk), v.view(B, T, self.n_v, self.dv)
        if self.n_v > self.n_k:                                                            # 大模型里 value 头比 key 头多：q、k 复制几份
            q, k = q.repeat_interleave(self.n_v // self.n_k, 2), k.repeat_interleave(self.n_v // self.n_k, 2)
        beta = self.in_proj_b(x).sigmoid()                                                 # [B, T, 头]，0~1
        g = -self.A_log.exp() * F.softplus(self.in_proj_a(x) + self.dt_bias)               # 对数衰减，永远 ≤ 0
        out, _ = delta_rule(q, k, v, g, beta)                                              # 这一章不保存记忆表，第 10 章才用
        out = self.norm(out, self.in_proj_z(x).view(B, T, self.n_v, self.dv))             # 按头做门控归一化
        return self.out_proj(out.reshape(B, T, -1))
```

`delta_rule` 里的四步和公式一一对应。外面的 `GatedDeltaNet` 是把残差流里的 1024 维变成这些量的"外壳"，有几个 Llama 系模型里没有的零件：

| 零件 | 形状 | 做什么 |
| --- | --- | --- |
| `in_proj_qkv` + `conv1d` | 1024 → 6144，再做窗口为 4 的深度可分因果卷积 | q、k、v 不是直接投影出来的，先让每个通道看自己前 3 个 token（局部上下文），再过 SiLU |
| `in_proj_b` → sigmoid | 1024 → 16 | 每个头一个 $\beta$：这个 token 写得多重 |
| `in_proj_a`, `dt_bias`, `A_log` | 1024 → 16 | $g = -e^{A\_log}\cdot\mathrm{softplus}(a + \text{dt\_bias})$，永远 $\le 0$，所以衰减 $e^g \in (0, 1]$；`A_log` 是每个头固定的"遗忘速度" |
| `in_proj_z` + `GatedRMSNorm` | 1024 → 2048 | 输出先按头归一化，再乘 $\mathrm{silu}(z)$ 的门 |
| `out_proj` | 2048 → 1024 | 回到残差流 |

`conv1d` 的 `padding = 3` 加上 `[..., :T]` 的截断，就是因果卷积：位置 $t$ 的输出只用 $t-3 \ldots t$ 的输入。`groups = 6144` 表示每个通道各卷各的，参数只有 6144 × 4 个。

## 跑起来：表不随长度变大

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

tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # 用第 2、3 章的零件造出第 0 层的真实输入
emb, norm = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens."), load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")
with torch.no_grad():
    for text in ("今天天气不错，我们去公园", "。".join(["今天天气不错，我们去公园"] * 20)):   # 1. 跑一长一短两段：时间随长度涨，记忆表不涨
        h = norm(emb(torch.tensor([tk.encode(text).ids])))
        t = time.time()
        y = dn(h)
        print(f"  {h.shape[1]:3d} 个 token：输出 {tuple(y.shape)}，{(time.time() - t) * 1000:.0f} ms")
    print(f"记忆表：每个头 [{dn.dk}, {dn.dv}]，{dn.n_v} 个头共 {dn.n_v * dn.dk * dn.dv * 4 / 2**20:.1f} MB，和 token 数无关；全注意力的 KV cache 每个 token 要加 {2 * cfg.num_key_value_heads * cfg.head_dim * 4 / 1024:.0f} KB")
    h = norm(emb(torch.tensor([tk.encode("今天天气不错，我们去公园").ids])))
    beta = dn.in_proj_b(h).sigmoid()                                # 2. 真实输入下的衰减和 β
    g = -dn.A_log.exp() * torch.nn.functional.softplus(dn.in_proj_a(h) + dn.dt_bias)
    print(f"第 0 层对这 6 个 token：衰减 exp(g) 的范围 [{g.exp().min():.2f}, {g.exp().max():.2f}]，β 的范围 [{beta.min():.2f}, {beta.max():.2f}]")

torch.manual_seed(0)                                                # 3. delta 规则的性质：同一对 (key, value) 反复写入，读出来的值收敛到 v；普通线性注意力会越积越大
k = torch.nn.functional.normalize(torch.randn(1, 1, 1, 128), dim=-1).expand(1, 6, 1, 128)
v = torch.randn(1, 1, 1, 128).expand(1, 6, 1, 128)
out, S = delta_rule(k, k, v, torch.zeros(1, 6, 1), torch.full((1, 6, 1), 0.5))   # q = k，无衰减，β = 0.5
plain = torch.cumsum(0.5 * v, dim=1)                                           # 没有"减去预测"的写法：S += β k v^T，读出来是 t × β × v
for t in range(6):
    print(f"  写入 {t + 1} 次后读出的值和 v 的距离：delta 规则 {(out[0, t, 0] * 128 ** 0.5 - v[0, t, 0]).norm():6.3f}，普通线性注意力 {(plain[0, t, 0] - v[0, t, 0]).norm():6.3f}")
```

```text title="输出"
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

前两行：6 个 token 9 ms、139 个 token 104 ms，时间随长度涨，但记忆表永远是 16 张 128 × 128 的表、1 MB；换成全注意力，每个 token 要往 KV cache 里追加 4 KB，一百万个 token 就是 4 GB——这就是"线性"二字的含义。中间一行是真实输入下的两个门：衰减从 0.01（这个头几乎只看当前 token）到 1.00（几乎不忘），$\beta$ 从 0.13 到 1.00。最后那个小实验是 delta 规则的精髓：同一对 (key, value) 连写 6 次，读出来的值和 v 的距离每次减半（$\beta = 0.5$ 只走一半），收敛到 v；普通线性注意力没有那一减，每写一次就多叠一份 $\beta v$——第 2 次恰好叠到 $v$，之后越积越偏。

## 来历

| 时间 | 事件 |
| --- | --- |
| 2021 | Schlag 等人的 *Linear Transformers Are Secretly Fast Weight Programmers*：线性注意力用 delta 规则写表，这就是 DeltaNet |
| 2024 | Yang 等人的 *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*（NeurIPS 2024）：DeltaNet 能按序列分块并行训练了，就是第 8 章的算法 |
| 2024 末 | *Gated Delta Networks*（ICLR 2025）：再加上随输入变的衰减 $\alpha$ |
| 2025 起 | Qwen3-Next、Qwen3.5 用 3:1 的比例把它和全注意力混在一起；Kimi Linear 的 KDA 是它的一个变体 |

## 常见疑问

**记忆表和 KV cache 有什么区别？**

KV cache 把每个 token 的 key、value 原样存着，前文多长就存多少，读的时候精确；记忆表把所有写入叠在一张固定大小的表里，不长大，但读出来是"混过的"。第 10 章会把两种状态放在一起比。

**q、k 为什么要先归一化成单位长度？**

delta 规则读出的是 $S^\top k$，写入的是差值乘 $k$。$k$ 是单位长度时，$\beta = 1$ 正好把这一格覆盖成新值；$k$ 太长会写过头、太短会写不够，几百步下来表的尺度会漂。$q$ 归一化后再除以 $\sqrt{d_k}$，读出来的尺度也稳定。

**$\alpha$、$\beta$ 和第 6 章的门是一回事吗？**

思路一样，都是由当前 token 算出来的 0～1 的开关，管的东西不同：$\alpha$ 管这次忘多少，$\beta$ 管这次写多重，输出前那个 `silu(z)` 的门管这次读出来的东西放行多少，最后这个和第 6 章的输出门最像。

**既然表会糊，为什么还要用它？**

便宜：每个 token 的计算和状态大小都是常数，上下文再长也不涨。代价由那 6 层全注意力来补。

!!! interview "怎么讲清楚"
    讲线性注意力，先讲"状态是固定大小"这件事：注意力的 KV cache 随长度涨，记忆表不涨，所以长上下文下显存和每 token 的计算都是常数。再讲 delta 规则和普通线性注意力的区别：写入的是"新值减去表里已有的预测"，同一个 key 反复出现不会累积，表的容量用得更省。最后说 gated：衰减 $e^g$ 让它能忘，$\beta$ 让它能选择性地写——两个门都是从输入算出来的。能提一句"Qwen3.5 的 18 : 6 混合比例，是用 6 层全注意力补线性注意力的精确回看"就完整了。

## 小结

- [x] Gated DeltaNet 每个头维护一张 128 × 128 的记忆表：衰减 → 用 k 查表 → 写入差值 × β → 用 q 查表。
- [x] 外壳里有短卷积、$\beta$ 投影、衰减投影（`A_log`、`dt_bias`）和带门的归一化。
- [x] 记忆表 16 × 128 × 128 = 1 MB，和 token 数无关；真实输入下衰减 0.01～1.00、$\beta$ 0.13～1.00；零件齐了，第 9 章拼起来。

## 思考题

**1. 一层 16 个头的记忆表是 1 MB（fp32），全注意力的 KV cache 每个 token 要 4 KB（fp32）。上下文多长时，一层全注意力的 KV cache 就比一层线性注意力的状态大了？**

??? success "参考答案"
    1 MB ÷ 4 KB = 256 个 token。超过 256 个 token，全注意力那一层存的就更多，而且还在涨；线性注意力那一层永远是 1 MB。

**2. $\beta = 1$、$k$ 是单位向量时，写入 $(k, v)$ 之后马上用同一个 $k$ 去读，读出来是什么？$\beta = 0.5$ 呢？**

??? success "参考答案"
    $\beta = 1$：正好读出 $v$，旧值被覆盖。$\beta = 0.5$：读出旧值和 $v$ 的正中间，离 $v$ 的距离减半。正文 `ch07_deltanet.py` 最后那个实验，就是同一对 (k, v) 连写 6 次、距离每次减半。

**3. 如果某个头的 $\alpha$ 一直是 1，会怎样？一直接近 0 呢？**

??? success "参考答案"
    一直是 1：从不忘，旧记忆越积越多，表越来越挤、越来越糊。一直接近 0：每一步先把表清空，读出来的几乎只有当前 token 自己写进去的东西，这一层就退化成只看当前 token。真实模型里不同的头 $\alpha$ 差别很大（正文实测第 0 层从 0.01 到 1.00），有的头管长期记忆，有的只看眼前。

**这一章新建的文件**：`qwen/deltanet.py`（第 10 章会加上状态和 mask）。
