# 新一代开源模型：压缩注意力、超连接与新的 MoE

<p class="lead">vLLM 0.30 和 SGLang 0.5.20 已经支持了新一代的开源模型：DeepSeek-V4（Flash 与 Pro）、Kimi-K3、GLM-5、MiniMax-M3、腾讯的 HY4、Gemma 4，以及本书用作例子的 Qwen3.5。和上一代相比，它们在三个地方同时动了刀：注意力（压缩 + 稀疏 + 滑窗，或者线性注意力混合）、残差连接（多条残差流的"超连接"）、MoE（按 token 编号路由、在低维空间里算专家）。每一处都给推理引擎带来新的工作：一个模型里有好几种缓存、每个请求多了要保存和回滚的状态、需要一批新的 kernel。这一章从各模型的 config.json 和两个框架的实现出发，把这些变化和它们对推理的影响讲清楚。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. DeepSeek-V4 的注意力由哪几部分组成？一个 1M 上下文的请求，decode 一步每层要看多少条 KV？
    2. 为什么 V4 每个 token 的 KV 只有 V3.2 的十分之一左右？每个请求又多了哪些固定开销？
    3. mHC 把残差变成了什么样子？为什么要把流与流之间的混合矩阵约束成双随机矩阵？推理时要付出什么代价？
    4. "哈希路由"的 MoE 层和普通 MoE 层有什么不同？推理引擎要为它做什么？
    5. Kimi-K3 的 latent MoE 省的是什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 128 个 token 的滑窗 + 压缩 KV：C4 层每 4 个 token 压成一条，由索引器挑出 top-k（512 条）来看；C128 层每 128 个 token 压成一条，全看。1M 上下文时，C4 层每层看 128 + 512 = 640 条，C128 层看 128 + 8192 = 8320 条。
    2. 大部分历史被压缩成条目（C4 层 1/4、C128 层 1/128），每 token 的 KV 只有约 3.8 KB，V3.2 约 47 KB（MLA 加索引器的键）。代价是每个请求多了固定开销：每层都要存一个滑窗的原始 KV，以及压缩器还没凑满一条的状态。
    3. 残差从一条变成 4 条并行的流，层与层之间用一个混合矩阵在流之间交换信息。混合矩阵连乘几十层会放大或缩小信号，约束成双随机矩阵（谱范数为 1）就保住了恒等路径。推理时激活和 PP 的通信都变成 4 倍，要把相关的小算子融合起来。
    4. 前几层按 token 编号查表决定路由（哈希路由），不看隐藏状态：路由可以提前算、负载可以预测，但会受高频 token 的影响。引擎要支持这种按 token id 的路由表，并利用"提前可知"来提前准备 dispatch。
    5. 在降到一半维度的潜空间里计算专家（先投影到低维、在低维里做专家的 FFN），dispatch / combine 传输的向量也是低维的，通信减半；于是能用更多、更细的专家。

## 一张表看新一代

| 模型 | 注意力 | 残差 | MoE 与数值格式 |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash / Pro（43 / 61 层） | 每层都有 128 个 token 的滑窗；C4 层每 4 个 token 压成一条 KV、再用索引器挑 512 / 1024 条；C128 层每 128 个 token 压成一条、全看；Flash 另有 2 层只有滑窗 | mHC：4 条残差流 | 256 / 384 个专家选 6 个；前 3 层按 token 编号查表路由；专家权重 FP4，其余 FP8（缩放因子是 2 的幂，UE8M0）；最长 1M |
| Kimi-K3（93 层） | 每 4 层 1 层 MLA，其余 69 层是线性注意力 KDA | 普通残差 | 896 个专家选 16 个 + 2 个共享专家；路由专家在 3584 维的"潜空间"里算（latent MoE），权重是 MXFP4；最长 1M |
| GLM-5 | 在 MoE 模型上用 DSA（索引器选 top-k 的稀疏注意力） | 普通残差 | vLLM 里走 DeepSeek-V2 的实现，长上下文 decode 默认用 all-to-all 的上下文并行 |
| MiniMax-M3 | MSA：带索引器的块稀疏注意力，kernel 针对 Blackwell | 普通残差 | 提供 MXFP8 版本 |
| HY4（腾讯） | MLA + 索引器稀疏注意力 + attention sink | iHC：多条独立的残差通道 | MoE |
| Gemma 4 | 滑窗与全注意力 5:1；全注意力层只有 1 个 512 维的 KV 头，而且 K 和 V 共用 | 普通残差 | 12B 是稠密模型（配置里有 MoE 的开关） |
| Qwen3.5（最小 0.8B，本书的例子） | Gated DeltaNet 与门控全注意力 3:1；全注意力层 2 个 KV 头、头维 256，只对前 1/4 的维度做 RoPE | 普通残差 | 0.8B 是稠密模型；原生多模态（同一个模型带视觉编码器）；自带 1 层 MTP，可以直接做投机解码 |

几个共同的方向：**稀疏注意力成了标配**（DeepSeek-V3.2 的 DSA 之后，GLM-5、MiniMax-M3、HY4 都用索引器挑 token）；**KV 越压越小**（压缩、滑窗、线性注意力、K=V 共用）；**残差连接开始变宽**（V4、HY4、GLM-5 的下一代都用了多条残差流）；**MoE 更细、更省通信**；**4 比特成为发布格式**。下面按注意力、残差、MoE 三部分看，最后汇总对推理引擎的要求。

## 注意力：滑窗 + 压缩 + 稀疏

DeepSeek-V4 的注意力仍然是 MLA 家族（Q 走低秩投影；所有头共用一个 512 维的 KV 向量，其中 64 维带 RoPE），但每层能看到的历史被分成了三路：

- **滑窗**：最近 128 个 token 的原始 KV，每层都有；
- **压缩 KV**：一个**压缩器**把每 $r$ 个 token 压成一条 KV——每个 token 算出一个 KV 向量和一组门控分数，块内按分数做 softmax 加权求和，再做 RMSNorm 和 RoPE。$r = 4$ 的层（C4）每条实际汇总 8 个 token（本块 4 个加上一块 4 个，块与块有重叠）；$r = 128$ 的层（C128）不重叠；
- **稀疏选择**：C4 层的压缩条目仍然很多（1M 上下文有 26 万条），于是像 V3.2 的 DSA 一样，用一个轻量的**索引器**（64 个头、头维 128，它自己也有一个压缩器）给所有压缩条目打分，只对前 512 条（Pro 是 1024 条）做注意力；C128 层的条目少（1M 上下文 8192 条），直接全看。

config.json 里的 `compress_ratios` 逐层给出压缩比：Flash 是 2 层只有滑窗、21 层 C4、20 层 C128，Pro 是 30 层 C4、31 层 C128，C4 和 C128 交替排列。下面的示例用一个小的压缩器说明两件事——逐 token 压缩和一次压完结果相同；还没凑满一块的 token 留在"部分状态"里（示例省略了 C4 的重叠）：

```python
import torch

torch.manual_seed(0)
D, R = 8, 4                          # 每个 token 的 KV 维度、压缩比（V4：512，4 或 128）


class Compressor:
    """每 R 个 token 压成一条 KV：门控分数在块内做 softmax，对 KV 加权求和（权重随机，只演示结构）。
    还没凑满 R 个的 token 留在"部分状态"里——它属于每个请求，和 KV 一样要保存、传输、回滚。"""

    def __init__(self):
        self.w_kv, self.w_gate = torch.randn(D, D) / D**0.5, torch.randn(D, D) / D**0.5
        self.ape = torch.randn(R, D) / 4                      # 块内位置的偏置（V4 里叫 ape）
        self.partial_kv, self.partial_score = [], []          # 部分状态：本块已经到了的 token
        self.entries = []                                     # 压缩好的 KV 条目

    def push(self, h):
        """decode 一步来一个 token：凑满 R 个就产出一条压缩 KV"""
        i = len(self.partial_kv)
        self.partial_kv.append(h @ self.w_kv)
        self.partial_score.append(h @ self.w_gate + self.ape[i])
        if len(self.partial_kv) == R:
            kv, score = torch.stack(self.partial_kv), torch.stack(self.partial_score)
            self.entries.append((score.softmax(0) * kv).sum(0))     # 每个维度各自在 R 个 token 上做 softmax 加权
            self.partial_kv, self.partial_score = [], []


def compress_all(h, c):
    """prefill：一次压完整段"""
    n = len(h) // R * R
    kv = (h[:n] @ c.w_kv).view(-1, R, D)
    score = (h[:n] @ c.w_gate).view(-1, R, D) + c.ape
    return (score.softmax(1) * kv).sum(1)


hidden = torch.randn(22, D)
c = Compressor()
for t in range(len(hidden)):
    c.push(hidden[t])
print(f"逐 token 压缩 {len(c.entries)} 条，与一次压完一致：{torch.allclose(torch.stack(c.entries), compress_all(hidden, c), atol=1e-6)}；"
      f"部分状态里还有 {len(c.partial_kv)} 个 token")


def attended(pos, ratio, topk=None, window=128):
    """位置 pos 的 query 要看多少条 KV：滑窗里的原始 token + 已经压好的条目（C4 层再用索引器挑 top-k）"""
    swa = min(window, pos + 1)
    if ratio <= 1:
        return swa                                            # 只有滑窗的层
    done = (pos + 1) // ratio
    return swa + (min(done, topk) if topk else done)


for L in (4_096, 131_072, 1_048_576):
    print(f"上下文 {L:>9,}：全注意力看 {L:>9,} 条；V4-Flash 的 C4 层看 {attended(L - 1, 4, 512):>3} 条（索引器给 {L // 4:>7,} 条打分），"
          f"C128 层看 {attended(L - 1, 128):>5,} 条，只有滑窗的层看 {attended(L - 1, 1)} 条")
```

```text title="输出"
逐 token 压缩 5 条，与一次压完一致：True；部分状态里还有 2 个 token
上下文     4,096：全注意力看     4,096 条；V4-Flash 的 C4 层看 640 条（索引器给   1,024 条打分），C128 层看   160 条，只有滑窗的层看 128 条
上下文   131,072：全注意力看   131,072 条；V4-Flash 的 C4 层看 640 条（索引器给  32,768 条打分），C128 层看 1,152 条，只有滑窗的层看 128 条
上下文 1,048,576：全注意力看 1,048,576 条；V4-Flash 的 C4 层看 640 条（索引器给 262,144 条打分），C128 层看 8,320 条，只有滑窗的层看 128 条
```

- **decode 的注意力几乎不随上下文变长**：C4 层固定看 512 条压缩 KV + 128 个近邻，C128 层的条目是上下文的 1/128；随上下文线性增长的只剩索引器的打分（头维 128、FP8，Blackwell 上用 MXFP4），这部分和 [DSA](../moe/mtp-sparse.md#稀疏注意力nsa-与-dsa) 一样便宜；
- **三路各司其职**：滑窗负责精确的近邻，C4 + 索引器负责"在很长的历史里精确地找到相关的几段"，C128 负责粗粒度的全局概览；Flash 的前两层只有滑窗；
- **压缩器有状态**：部分状态和 [线性注意力的状态](linear-attn.md) 是同一类东西——属于每个请求、随 decode 更新。vLLM 把它实现成一种滑窗缓存（C4 留最近 8 个 token 的中间量，C128 留 128 个，float32），和 KV 放在同一个内存池里。

## 新的 KV 账本

换层数和全注意力的比例，看混合模型的显存账本怎么变（线性注意力一章的同一个工具）：

<div class="aig-widget" data-widget="linear-memory"></div>

把各模型 config.json 里的结构参数代进去，算一个请求的 KV（V3.2 与 V4 按 FP8 格式，Kimi-K3 的 MLA 按 bf16，KDA 状态按 float32）：

```python
KB, MB, GB = 1024, 1024**2, 1024**3
# 每个模型：(每 token 增加的字节数, 每个请求固定的字节数, 最大上下文)


def v4(c4, c128, swa_only, idx_bytes=132):
    """DeepSeek-V4：每层都存最近 128 个 token 的原始 KV（FP8 格式 584 B/token）；C4 层每 4 个 token 一条压缩 KV + 一条
    索引器 key（FP8 时 132 B）；C128 层每 128 个 token 一条压缩 KV；压缩器的部分状态是 float32，C4 层留 8 个 token、C128 层留 128 个"""
    per_token = c4 * (584 + idx_bytes) / 4 + c128 * 584 / 128
    swa = (c4 + c128 + swa_only) * 128 * 584
    state = c4 * 8 * (2 * 2 * 512 + 2 * 2 * 128) * 4 + c128 * 128 * (2 * 512) * 4
    return per_token, swa + state


MODELS = {
    "DeepSeek-V3.2": (61 * (656 + 132), 0, 163_840),              # 61 层 MLA（FP8 656 B/token）+ 索引器 key
    "DeepSeek-V4-Flash": (*v4(c4=21, c128=20, swa_only=2), 1_048_576),
    "DeepSeek-V4-Pro": (*v4(c4=30, c128=31, swa_only=0), 1_048_576),
    "Kimi-K3": (24 * 576 * 2, 69 * 96 * 128 * 128 * 4, 1_048_576),   # 24 层 MLA（bf16）+ 69 层 KDA（96 头 × 128×128 float32 状态）
    "Gemma 4 12B": (8 * 512 * 2, 40 * 1024 * 8 * 256 * 2 * 2, 262_144),  # 8 层全注意力（K=V 共用 1 个 512 维头）+ 40 层 1024 滑窗
    # 6 层全注意力（2 个 KV 头 × 256，bf16）+ 18 层 Gated DeltaNet（16 头 × 128×128 float32 状态 + 3 列卷积缓存）
    "Qwen3.5-0.8B": (6 * 2 * 2 * 256 * 2, 18 * 16 * 128 * 128 * 4 + 18 * 6144 * 3 * 2, 262_144),
}
for name, (grow, fixed, max_len) in MODELS.items():
    total = [f"{(grow * L + fixed) / GB:.2f} GB" if L <= max_len else "不支持" for L in (131_072, 1_048_576)]
    print(f"{name}：每 token {grow / KB:.1f} KB，每请求固定 {fixed / MB:.0f} MB；一个 128K 的请求 {total[0]}，1M 的请求 {total[1]}")
```

```text title="输出"
DeepSeek-V3.2：每 token 46.9 KB，每请求固定 0 MB；一个 128K 的请求 5.87 GB，1M 的请求 不支持
DeepSeek-V4-Flash：每 token 3.8 KB，每请求固定 15 MB；一个 128K 的请求 0.48 GB，1M 的请求 3.77 GB
DeepSeek-V4-Pro：每 token 5.4 KB，每请求固定 22 MB；一个 128K 的请求 0.69 GB，1M 的请求 5.40 GB
Kimi-K3：每 token 27.0 KB，每请求固定 414 MB；一个 128K 的请求 3.78 GB，1M 的请求 27.40 GB
Gemma 4 12B：每 token 8.0 KB，每请求固定 320 MB；一个 128K 的请求 1.31 GB，1M 的请求 不支持
Qwen3.5-0.8B：每 token 12.0 KB，每请求固定 19 MB；一个 128K 的请求 1.52 GB，1M 的请求 不支持
```

- **两条通往 1M 的路**：V4 靠压缩（每 4 个或 128 个 token 一条），每 token 只有 V3.2 的十分之一左右，1M 的请求 4～5 GB；Kimi-K3 靠线性注意力把 3/4 的层变成固定大小的状态，剩下 24 层 MLA 仍按 token 增长，1M 时是 V4 的 5～7 倍；
- **固定开销变大了**：V4 每个请求有十几到二十几 MB 的滑窗和压缩器状态，Kimi-K3 有 400 多 MB 的 KDA 状态，Gemma 4 的滑窗层有 320 MB。**短请求、高并发**时，能同时服务多少个请求由这部分决定，而不是 KV（[线性注意力](linear-attn.md#新的显存账本)一章算过分界点）；
- **KV 小了，瓶颈就移走了**：V4 的 128K 请求只有 0.5～0.7 GB KV，一张卡能放下的并发数比 V3.2 高一个数量级，decode 的瓶颈更多地落在权重读取和专家通信上——这也是它把专家权重做成 FP4 的原因之一。

## 残差：从一条流到四条流

标准 Transformer 的残差只有一条流：$x \leftarrow x + f(\text{norm}(x))$。**超连接**（Hyper-Connections）把它加宽成 $n$ 条并行的流，每个子层（注意力或 FFN）之前和之后各做一次混合：

- 子层之前：用当前的 $n$ 条流算出三组系数（一个小的线性层 + sigmoid / softmax），按 `pre` 权重把 $n$ 条流合成子层的输入；
- 子层之后：新的第 $j$ 条流 $= \sum_i \text{comb}_{ij} \cdot$ 第 $i$ 条流 $+ \text{post}_j \cdot$ 子层输出。

DeepSeek-V4 用 $n = 4$，并且把流与流之间的混合矩阵 `comb` 用 Sinkhorn 迭代（20 次）约束成**双随机矩阵**（每行、每列之和都是 1）——这是 mHC（manifold-constrained，约束在双随机矩阵构成的集合上）名字的来历。下面按 vLLM 的参考实现（`model_executor/kernels/mhc/torch.py`）写一遍，并看看约束的作用：

```python
import torch

torch.manual_seed(0)
N, H = 4, 64                                   # 4 条残差流（hc_mult），隐藏维度


def sinkhorn(logits, iters=20, eps=1e-6):
    """先按行 softmax，再交替按列、按行归一化：得到每行、每列之和都是 1 的双随机矩阵"""
    m = logits.softmax(-1) + eps
    m = m / (m.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(-1, keepdim=True) + eps)
        m = m / (m.sum(-2, keepdim=True) + eps)
    return m


def mhc_pre(streams, fn, base, scale):
    """子层之前：由 4 条流算出三组系数，并把 4 条流加权合成子层的输入"""
    x = streams.flatten()                                  # [N*H]
    mix = (fn @ x) * torch.rsqrt(x.square().mean() + 1e-6)
    pre = torch.sigmoid(mix[:N] * scale[0] + base[:N])                 # 合成输入的权重
    post = torch.sigmoid(mix[N:2 * N] * scale[1] + base[N:2 * N]) * 2  # 子层输出写回每条流的权重
    comb = sinkhorn((mix[2 * N:] * scale[2] + base[2 * N:]).view(N, N))  # 流与流之间的混合矩阵
    return (pre[:, None] * streams).sum(0), post, comb


def mhc_post(out, streams, post, comb):
    """子层之后：新的第 j 条流 = Σ_i comb[i, j] · 第 i 条流 + post[j] · 子层输出"""
    return comb.T @ streams + post[:, None] * out[None, :]


streams = torch.randn(H).repeat(N, 1)                      # 嵌入复制成 4 条流
fn, base, scale = torch.randn(2 * N + N * N, N * H) * 0.02, torch.randn(2 * N + N * N), torch.ones(3)
inp, post, comb = mhc_pre(streams, fn, base, scale)
print("comb 每行之和：", [round(v, 3) for v in comb.sum(1).tolist()], "每列之和：", [round(v, 3) for v in comb.sum(0).tolist()])
streams = mhc_post(torch.tanh(inp), streams, post, comb)
print("子层输入", tuple(inp.shape), "；残差流", tuple(streams.shape))

# 为什么要约束成双随机矩阵：只看残差路径，把 60 层的混合矩阵连乘起来
depth, prod_free, prod_ds = 60, torch.eye(N), torch.eye(N)
for _ in range(depth):
    free = torch.eye(N) + torch.randn(N, N) * 0.15           # 原始的超连接：混合矩阵不加约束，接近单位阵
    prod_free = free @ prod_free
    prod_ds = sinkhorn(torch.eye(N) * 3 + torch.randn(N, N) * 0.5).T @ prod_ds
print(f"{depth} 层之后残差路径的放大倍数（矩阵的最大奇异值）：不约束 {torch.linalg.matrix_norm(prod_free, 2):.3g}，"
      f"双随机约束 {torch.linalg.matrix_norm(prod_ds, 2):.3g}")
```

```text title="输出"
comb 每行之和： [1.0, 1.0, 1.0, 1.0] 每列之和： [1.0, 1.0, 1.0, 1.0]
子层输入 (64,) ；残差流 (4, 64)
60 层之后残差路径的放大倍数（矩阵的最大奇异值）：不约束 11.5，双随机约束 1
```

- **为什么要约束**：普通残差的恒等路径不改变信号的大小，这是深层网络能训练的关键；不加约束的混合矩阵每层都接近单位阵，但 60 层连乘下来偏差会累积（上面放大了十几倍，也可能缩小）。双随机矩阵的乘积还是双随机矩阵，最大奇异值恒为 1——恒等路径的性质被保住了；
- **推理的代价**：残差从 $[T, H]$ 变成 $[T, 4, H]$，每个子层前后都要读写 4 倍的残差；流水线并行时相邻 stage 传的激活也是 4 倍（vLLM 的实现里，PP 的中间张量就保持 `(num_tokens, hc_mult, hidden_size)` 的形状）；
- **一堆小算子要融合**：每个子层前后是一个 $24 \times 4H$ 的小 GEMV、sigmoid、20 次 Sinkhorn、加权求和——单独发射全是访存受限的小 kernel。vLLM 用 TileLang 写了 `mhc_pre`、`mhc_post` 以及"上一个子层的 post + 下一个子层的 pre"融合在一起的 kernel；SGLang 的实现在 `srt/layers/communicator_mhc.py`；
- **草稿模型拿什么当输入**：MTP、EAGLE3、DSpark 需要目标模型某几层的隐藏状态，V4 的实现把 4 条流取平均再交给草稿模型。

腾讯的 HY4 用的是另一种变体 iHC（独立的超连接，`vllm/models/hy_v4/nvidia/hc.py`），结构相同：每个子层先把多条通道合成一个输入，算完再分散回各条通道，最后由一个 head 层合并。

## MoE：按 token 编号路由、在潜空间里算专家

**哈希路由。** V4 的前 3 层 MoE 不用路由器选专家：权重文件里有一张表 `tid2eid`（词表大小 × 6），第 $t$ 个 token 固定去表里写的 6 个专家；路由器的输出只用来算这 6 个专家的权重（V4 的打分函数是 $\sqrt{\text{softplus}(\text{logit})}$，归一化后乘 `routed_scaling_factor`）。其余层照常用路由器选专家（带偏置的 noaux 方式，和 V3 一样）。

这对推理有两个直接的影响：一是**路由在前向之前就知道了**——只要有 token 编号，前 3 层每个 token 去哪些专家、每个专家有多少 token 都能提前算出来，专家并行的分发可以提前规划；二是**负载只取决于这批 token 的分布**：

```python
import torch

torch.manual_seed(0)
VOCAB, E, K = 129_280, 256, 6                       # DeepSeek-V4-Flash：词表、路由专家数、每个 token 选几个专家
tid2eid = torch.stack([torch.randperm(E)[:K] for _ in range(VOCAB)])   # 哈希路由层：token 编号 → 6 个专家（这里随机造一张表）

# 一批文本的 token 近似服从 Zipf 分布：少数高频 token（标点、虚词）占了很大比例
rank = torch.arange(1, VOCAB + 1, dtype=torch.float64)
freq = rank ** -1.1 / (rank ** -1.1).sum()
for batch in (256, 4096, 65536):
    ids = torch.multinomial(freq, batch, replacement=True)
    hashed = torch.bincount(tid2eid[ids].flatten(), minlength=E).float()
    learned = torch.bincount(torch.stack([torch.randperm(E)[:K] for _ in range(batch)]).flatten(), minlength=E).float()
    print(f"batch {batch:>6} 个 token：最忙的专家是平均的 哈希路由 {hashed.max() / hashed.mean():.1f} 倍，"
          f"理想的均衡路由 {learned.max() / learned.mean():.1f} 倍")

# 专家并行时，每个 token 在每个 MoE 层要发出去的字节数（分发用 FP8）
for name, hidden, topk in [("DeepSeek-V3（7168 维，选 8 个）", 7168, 8), ("DeepSeek-V4-Pro（7168 维，选 6 个）", 7168, 6),
                           ("Kimi-K3（先降到 3584 维，选 16 个）", 3584, 16), ("假如 Kimi-K3 不降维", 7168, 16)]:
    print(f"{name}：分发 {hidden * topk / 1024:.0f} KB/token")
```

```text title="输出"
batch    256 个 token：最忙的专家是平均的 哈希路由 8.0 倍，理想的均衡路由 2.2 倍
batch   4096 个 token：最忙的专家是平均的 哈希路由 7.6 倍，理想的均衡路由 1.3 倍
batch  65536 个 token：最忙的专家是平均的 哈希路由 7.5 倍，理想的均衡路由 1.1 倍
DeepSeek-V3（7168 维，选 8 个）：分发 56 KB/token
DeepSeek-V4-Pro（7168 维，选 6 个）：分发 42 KB/token
Kimi-K3（先降到 3584 维，选 16 个）：分发 56 KB/token
假如 Kimi-K3 不降维：分发 112 KB/token
```

- **高频 token 决定负载**：随机的表在 Zipf 分布下，最忙的专家是平均的 7～8 倍，而且 batch 再大也不会变均衡（高频 token 永远去同样的专家）。真实的表来自训练，能不能兼顾均衡要用真实流量统计；对引擎来说，好消息是这几层的负载可以**提前预测**，EPLB 可以直接用 token 频率来复制热门专家；
- **引擎必须带着 token 编号**：V4 的实现在没有 `input_ids` 时直接报错。只传 embedding 的调用方式（有些多模态流水线就是这样）对这几层不可用；V4 的视觉版本给图片 token 预留了几个编号，这些位置改用单独的偏置来路由；
- **latent MoE**：Kimi-K3 先把 7168 维的隐藏状态投影到 3584 维，路由专家的输入输出都在这个"潜空间"里，算完再投影回去。专家并行的分发与合并流量减半，每个专家的权重也减半——于是它能用 896 个专家、每个 token 选 16 个，而通信量和 V3 选 8 个时一样。代价是多了两个投影；vLLM 里的实现是 `models/kimi_k3/nvidia/latent_moe_runner.py`，把潜空间结果和共享专家的结果拼在一起做**一次** all-reduce；
- **MegaMoE**：V4 和 Kimi-K3 在 vLLM、SGLang 里都能用 FlashInfer 的 MegaMoE：把专家并行的 all-to-all 分发、专家计算和合并做进同一个大 kernel（通信走对称内存），省掉 MoE 层里多次 kernel 发射和同步（SGLang 的封装在 `srt/layers/moe/flashinfer_megamoe.py`）。

**数值格式。** V4 的专家权重是 FP4（`expert_dtype: fp4`），其余是块大小 128×128 的 FP8，缩放因子用 UE8M0（只能是 2 的幂，乘法变成指数加法，见 [FP8 GEMM](../moe/fp8-gemm.md) 与 [低比特推理](low-bit.md)）；Kimi-K3 的路由专家权重是 MXFP4（每 32 个数一个 E8M0 缩放），注意力、共享专家和稠密 MLP 保持高精度；V4 的 SwiGLU 对激活做了截断（`swiglu_limit: 10`），让低精度的激活不溢出。

## 对推理引擎的要求

| 变化 | 引擎要做的事 | 在哪里看实现 |
| --- | --- | --- |
| 一层有好几种缓存：滑窗 KV、压缩 KV、索引器的 key、压缩器状态；混合模型还有线性注意力状态 | KV 管理器要支持多种缓存规格共用一个内存池（vLLM 让压缩器状态和 KV 块共用同一块物理张量，所以页大小要对齐） | vLLM `v1/kv_cache_interface.py`（`SlidingWindowMLASpec` 等）、`models/deepseek_v4/compressor.py`；SGLang `srt/layers/attention/dsv4/` |
| 每个请求都有状态（压缩器的部分状态、KDA 状态） | 前缀缓存只能在状态可复用的位置命中（对齐到压缩块、或者存检查点）；投机解码被拒绝的草稿 token 要回滚状态；PD 分离要连状态一起传 | [线性注意力](linear-attn.md#前缀缓存要重做)一章的同类问题 |
| 稀疏注意力的索引、元数据每步都在变 | 稀疏索引器和注意力在 CUDA Graph 捕获时单独留成 eager 段（V4 的 `_prepare_and_attn` 用 `eager_break_during_capture`），其余部分照常捕获 | `models/deepseek_v4/attention.py` |
| 注意力前有好几个互不依赖的小计算 | Q 投影、压缩器、索引器分别放在不同的 CUDA stream 上并行 | 同上，`execute_in_parallel` |
| 残差变成 4 条流 | 激活和 PP 通信 × 4；mHC 的小算子要融合 | vLLM `model_executor/kernels/mhc/` |
| 新的 kernel | V4 的稀疏 MLA（FlashMLA 与 FlashInfer 各有一个 DSV4 后端）、CuTe DSL 写的压缩器、TileLang 写的 mHC、MegaMoE、Blackwell 上 MXFP4 的索引器缓存 | `v1/attention/backends/mla/` |
| 各家硬件 | V4 在 vLLM 里有 nvidia、amd、xpu、cpu 四套实现；SGLang 另有昇腾 NPU 的实现 | vLLM `models/deepseek_v4/{nvidia,amd,xpu,cpu}/`；SGLang `srt/hardware_backend/npu/dsv4/` |

投机解码也跟着变了：V4 和 Kimi-K3 在 vLLM 里都有 DSpark 草稿（一次前向给出一整块草稿）的实现，vLLM 还为它加了按负载决定验证多少 token 的自适应验证——见[投机解码的新做法](spec-next.md)。

!!! interview "怎么讲清楚"
    讲"新一代模型对推理引擎有什么影响"，按四块讲，每块都带数字：**注意力**——V4 用滑窗 + 压缩（每 4 / 128 个 token 一条）+ 索引器选 top-k，decode 每层只看几百到几千条 KV，每 token 的 KV 只有 V3.2 的十分之一，1M 上下文一个请求 4～5 GB；Kimi-K3 走线性注意力混合的路，每请求有 400 多 MB 固定状态。**残差**——mHC 把残差变成 4 条流，混合矩阵用 Sinkhorn 约束成双随机矩阵保住恒等路径，推理时激活和 PP 通信 × 4，小算子必须融合。**MoE**——前 3 层按 token 编号查表路由，路由可以提前算、负载可预测但受高频 token 影响；latent MoE 在半维空间里算专家，通信减半。**引擎**——多种缓存共用内存池、状态要进前缀缓存 / 投机解码 / PD 分离、CUDA Graph 分段捕获、多流并行、每种硬件一套实现。

## 练习

**1. 估算 DeepSeek-V4-Pro 在 256K 上下文时一个请求的 KV。** 用上面的账本。

??? success "参考答案"
    每 token 约 5.4 KB（精确值 5511 B），256K 个 token 约 1.34 GB，再加每请求固定的约 22 MB（滑窗 4.6 MB + 压缩器状态约 18 MB），共约 1.37 GB。作为对比，V3.2 最长只到 160K，128K 时已经 5.9 GB。

**2. 为什么 C4 层要用索引器挑 top-k，C128 层却直接全看？**

??? success "参考答案"
    看条目数：1M 上下文时 C4 层有 26 万条压缩 KV，全看的计算量和 26 万 token 的稠密注意力差不多，必须稀疏；C128 层只有 8192 条，全看的代价和 8K 上下文的注意力相当，可以接受，还能给每个 query 一个不漏掉任何位置的粗粒度全局视图。C4 + 索引器负责"精确地找到相关的几段"，C128 负责"不遗漏"，滑窗负责近邻——三路互补。

**3. 前缀缓存命中了一个 1000 个 token 的前缀，新请求从第 1001 个 token 开始 prefill，C128 层要准备什么？**

??? success "参考答案"
    前 896 个 token（7 个完整的块）的 7 条压缩 KV 可以直接复用；第 897～1000 这 104 个 token 还没凑满一块，它们的中间量在压缩器的部分状态里，要么随前缀一起缓存了这份状态，要么从第 897 个 token 开始重新计算。滑窗也需要最近 128 个 token 的原始 KV。所以引擎通常把前缀缓存的命中位置对齐到压缩块的边界（C128 是 128 的倍数），多出的尾巴重算——和线性注意力的"只在存过状态的位置命中"是同一类取舍。

**4. 哈希路由层对专家并行的负载均衡意味着什么？**

??? success "参考答案"
    这几层每个专家的负载 = 这批 token 里"路由到它的 token 编号"出现的次数之和，只取决于文本本身，可以在前向之前、甚至在分词之后就算出来。坏处是高频 token 固定落在同几个专家上，batch 再大也不会自然变均衡；好处是可预测——EPLB 可以按 token 频率的统计复制热门专家，调度器也可以提前知道这几层的分发量，把通信和前面的计算重叠起来。

## 小结

- [x] 新一代模型同时改了注意力、残差和 MoE：稀疏注意力成为标配，KV 被压缩或换成状态，残差变宽，MoE 更细，4 比特成为发布格式。
- [x] DeepSeek-V4 的注意力 = 128 个 token 的滑窗 + 压缩 KV（C4 用索引器选 top-k，C128 全看）；每 token 的 KV 约为 V3.2 的十分之一，但每个请求多了滑窗和压缩器状态这类固定开销。
- [x] mHC 把残差变成 4 条流，混合矩阵约束成双随机矩阵保住恒等路径；推理时激活与 PP 通信 × 4，小算子要融合。
- [x] 哈希路由让前几层的路由提前可知、负载可预测；latent MoE 在半维空间里算专家，通信减半，于是能用更多、更细的专家。
- [x] 引擎要支持多种缓存共用内存池、让前缀缓存、投机解码和 PD 分离都带上状态、分段捕获 CUDA Graph，并为每种硬件准备实现。
