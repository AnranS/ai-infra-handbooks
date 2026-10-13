# （八）Gated DeltaNet：分块并行

<p class="lead">上一章的递推一步只做几个 128 × 128 的小运算，T 个 token 就是 T 次 Python 循环，GPU 完全吃不饱——这正是线性注意力早年"理论上快、实际上慢"的原因。把序列切成 64 个 token 一块：块内 64 次"衰减 + 写入"合起来是一个下三角线性方程组，一次三角求解就能得到每个 token 真正写入的量；块与块之间仍像递推一样只传一张表。结果和逐 token 版一样，速度差一个数量级。</p>

!!! note "这一章跑什么"
    `ch08_chunk.py`，约 15 秒。

@@files@@

## 块内：把 64 步递推变成矩阵乘

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

## 对数，计时

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

!!! interview "怎么讲清楚"
    讲"线性注意力为什么能并行训练"：递推形式一步依赖上一步，但块内的 64 步可以写成一个下三角方程组 + 两个矩阵乘，块间只传状态；这和 Mamba-2 的 SSD、FlashLinearAttention 的 chunk 算法是同一个思路。要说清"为什么要分块而不是全部并行"：全并行的中间量是 $T \times T$，又回到了注意力的平方复杂度；块长 64 是在"并行度"和"中间量大小"之间折中。

## 小结

- [x] 块内：$(I + A)U = \beta \odot V$ 一次三角求解得到真正写入量，输出 = 读块首状态 + 读块内写入；块间：传一张表。
- [x] 和递推版逐位一致，长输入快一个数量级；工业实现（FlashLinearAttention、HF 的 prefill）用的就是这个算法。

**这一章新建的文件**：`qwen/chunk.py`。
