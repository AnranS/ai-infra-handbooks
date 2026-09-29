# 线性注意力与混合架构的推理

<p class="lead">稀疏注意力让每个 query 少看一些 token，但 KV Cache 照样随上下文线性增长。另一条路是<b>线性注意力</b>：用一个固定大小的状态矩阵汇总全部历史，decode 每一步的计算和显存都与上下文长度无关。纯线性注意力的效果不够好，于是新一代模型采用<b>混合架构</b>——大部分层用线性注意力、少数层保留全注意力（例如 Qwen3-Next 的 Gated DeltaNet 与门控注意力 3:1 混合，MiniMax 的 lightning attention，Nemotron-H、Jamba 的 Mamba 与注意力混合）。这一章从推理引擎的角度看它们：同一个计算的两种写法（decode 递推、prefill 分块），状态带来的新账本，以及前缀缓存、投机解码这些"为 KV 设计的机制"为什么都要重做。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 线性注意力的状态是什么形状？decode 一步的计算量和上下文长度有什么关系？
    2. prefill 时为什么不逐 token 递推，而用分块的写法？
    3. 混合模型一定比全注意力模型省显存吗？
    4. 线性注意力层的前缀缓存为什么不能按块共享？vLLM、SGLang 怎么做？
    5. 投机解码在线性注意力模型上多了什么麻烦？

## 同一个计算的两种写法

softmax 注意力要把每个历史 token 的 K、V 都留着。去掉 softmax 之后，$o_t = \sum_{s \le t} (q_t \cdot k_s)\, v_s = q_t \sum_{s \le t} k_s^\top v_s$，求和项可以累积成一个 $d_k \times d_v$ 的**状态** $S_t$。现代的线性注意力都会加上"遗忘"：

- 带门控的线性注意力（GLA、RetNet、Mamba2 的 SSD 形式）：$S_t = \alpha_t S_{t-1} + k_t^\top v_t$，$\alpha_t \in (0, 1)$ 由输入决定；
- DeltaNet 用"纠错"式的更新 $S_t = (I - \beta_t k_t^\top k_t) S_{t-1} + \beta_t k_t^\top v_t$（先擦掉 $k_t$ 方向上旧的记忆，再写入新的），Gated DeltaNet 再乘上衰减门 $\alpha_t$。

推理时有两种等价的算法，下面以带门控的线性注意力为例验证：

```python
import torch

torch.manual_seed(0)
T, DK, DV, C = 64, 16, 16, 16                      # 序列长度、key / value 维度、分块大小
q, k, v = torch.randn(T, DK) / 4, torch.randn(T, DK) / 4, torch.randn(T, DV)
alpha = torch.sigmoid(torch.randn(T) + 3)          # 每步的衰减门（接近 1）：S_t = α_t·S_{t-1} + k_tᵀ v_t


def recurrent(q, k, v, alpha):
    """decode 的形式：逐 token 更新固定大小的状态 S（DK×DV），输出 o_t = q_t S_t"""
    S, out = torch.zeros(DK, DV), []
    for t in range(len(q)):
        S = alpha[t] * S + k[t, :, None] * v[t, None, :]
        out.append(q[t] @ S)
    return torch.stack(out), S


def chunked(q, k, v, alpha):
    """prefill 的形式：块内像注意力一样用矩阵乘（带衰减的因果掩码），块间只传递状态"""
    S, out = torch.zeros(DK, DV), []
    for s in range(0, len(q), C):
        qc, kc, vc = q[s:s + C], k[s:s + C], v[s:s + C]
        g = torch.log(alpha[s:s + C]).cumsum(0)                     # 块内累计的对数衰减
        decay = torch.exp(g[:, None] - g[None, :]).tril()           # 位置 i 看位置 j（j ≤ i）时的衰减
        intra = ((qc @ kc.T) * decay) @ vc                          # 块内：C×C 的"注意力"
        inter = torch.exp(g)[:, None] * (qc @ S)                    # 块间：之前所有 token 汇总在 S 里
        out.append(intra + inter)
        S = torch.exp(g[-1]) * S + (torch.exp(g[-1] - g)[:, None] * kc).T @ vc
    return torch.cat(out), S


o1, s1 = recurrent(q, k, v, alpha)
o2, s2 = chunked(q, k, v, alpha)
print("递推与分块的输出一致：", torch.allclose(o1, o2, atol=1e-5), "；最终状态一致：", torch.allclose(s1, s2, atol=1e-5))
print(f"每个头的状态：{DK}×{DV} 个数，与序列长度无关；同样长度的 KV Cache 要 {T}×({DK}+{DV}) 个数")
```

```text title="输出"
递推与分块的输出一致： True ；最终状态一致： True
每个头的状态：16×16 个数，与序列长度无关；同样长度的 KV Cache 要 64×(16+16) 个数
```

- **decode 用递推**：每一步读一次状态、做一次秩一更新，计算和访存都是常数，和上下文多长无关；
- **prefill 用分块**：逐 token 递推是串行的，而且每步都是很小的向量运算，用不上 Tensor Core。分块的写法在块内做 $C \times C$ 的矩阵乘（像一小段带衰减掩码的注意力），块与块之间只传递状态，整体是 $O(T \cdot C)$ 的计算，而且大部分是矩阵乘。flash-linear-attention（fla）库的 Triton kernel、Mamba2 的 SSD kernel 都是这种结构；DeltaNet 的块内部分更复杂一些（要先把块内的纠错更新写成矩阵形式），但思路相同；
- **分块 prefill 天然支持**：推理引擎把一个长提示词切成几段时，只要把上一段结束时的状态交给下一段即可——这和块间传递状态是同一件事。

线性注意力层前面通常还有一个短的**因果卷积**（窗口 4 左右），它也需要为每个请求保存最后几个 token 的输入，和状态一起构成这一层的"缓存"。

## 新的显存账本

混合模型里，全注意力层照样有随上下文增长的 KV，线性注意力层则是每个请求一份固定大小的状态。状态通常用 fp32 保存（递推的数值误差会累积），一个头 128×128 就是 64 KB。以一个示意的混合模型（48 层，每 4 层 1 层全注意力）为例：

```python
MiB = 2**20
LAYERS, FULL = 48, 12                              # 一个示意的混合模型：48 层，每 4 层 1 层全注意力、3 层线性注意力
KV = 2 * 2 * 256 * 2                               # 全注意力层每 token：K、V × 2 个 KV 头 × 头维 256 × bf16 = 2 KiB
STATE = 32 * 128 * 128 * 4                         # 线性注意力层每请求：32 个头 × 128×128 的状态 × fp32 = 2 MiB
LINEAR = LAYERS - FULL

print("上下文    全部用全注意力    混合（KV + 固定状态）   混合 / 全注意力")
for L in (512, 1024, 4096, 32768, 262144):
    dense, hybrid = LAYERS * KV * L, FULL * KV * L + LINEAR * STATE
    print(f"{L:>7}   {dense / MiB:>10.0f} MiB   {hybrid / MiB:>12.0f} MiB   {hybrid / dense:>12.0%}")
print(f"分界点：上下文短于 {STATE // KV} 个 token 时，混合模型每个请求反而占得更多")

print("前缀缓存：每 B 个 token 存一个状态检查点，相对这 B 个 token 的 KV 要多占多少")
for B in (256, 1024, 4096):
    print(f"  B = {B:>4}：一个检查点 {LINEAR * STATE / MiB:.0f} MiB，这 {B} 个 token 的 KV {FULL * KV * B / MiB:.0f} MiB，"
          f"比例 {LINEAR * STATE / (FULL * KV * B):.1f} 倍")
```

```text title="输出"
上下文    全部用全注意力    混合（KV + 固定状态）   混合 / 全注意力
    512           48 MiB             84 MiB           175%
   1024           96 MiB             96 MiB           100%
   4096          384 MiB            168 MiB            44%
  32768         3072 MiB            840 MiB            27%
 262144        24576 MiB           6216 MiB            25%
分界点：上下文短于 1024 个 token 时，混合模型每个请求反而占得更多
前缀缓存：每 B 个 token 存一个状态检查点，相对这 B 个 token 的 KV 要多占多少
  B =  256：一个检查点 72 MiB，这 256 个 token 的 KV 6 MiB，比例 12.0 倍
  B = 1024：一个检查点 72 MiB，这 1024 个 token 的 KV 24 MiB，比例 3.0 倍
  B = 4096：一个检查点 72 MiB，这 4096 个 token 的 KV 96 MiB，比例 0.8 倍
```

- **长上下文时省得多**：上下文越长，越接近"只有 1/4 的层有 KV"，显存降到四分之一；
- **短请求反而更贵**：每个请求一上来就要 72 MiB 的状态，上下文短于约 1000 个 token 时比全注意力还多。大量短请求的高并发场景，能同时服务的请求数受状态池的大小限制，而不是 KV；
- 所以推理引擎要为混合模型准备**两种内存池**：分页的 KV 池（全注意力层）和按请求分配的状态池（线性注意力层），两者的容量要按负载的长度分布一起规划。vLLM 用不同的 KV 缓存组管理，线性注意力层由 `MambaManager`（`v1/core/single_type_kv_cache_manager.py`）负责；SGLang 用 `HybridReqToTokenPool`、`MambaPool` 和 `HybridLinearKVPool`（`srt/mem_cache/memory_pool.py`）。

## 前缀缓存要重做

KV 的前缀缓存能按块共享，是因为第 $i$ 块的 KV 只依赖前 $i$ 块的 token，而且每块的 KV 是独立存放的。线性注意力的状态却是**整个前缀的一个汇总**：位置 1000 的状态无法从位置 1024 的状态里"切"出来，也不能把两个块的状态拼起来。要复用一段前缀，就必须在那个位置**恰好存了一份状态**（检查点）。

上面的账本说明检查点很贵：每 256 个 token 存一个，状态的显存是这段 KV 的 12 倍。所以实际的做法是有选择地存：

- **vLLM** 的 `--mamba-cache-mode`：`all` 在每个块边界都存一份状态；`align` 是开启前缀缓存时的默认值，只在每个调度步结束、且恰好落在块边界的位置存——相当于只在"一段 prefill 刚做完"的地方留检查点；
- **SGLang** 的 `mamba_radix_cache.py` 把全注意力的 KV 和线性层的状态放进同一棵基数树，状态只挂在特定的节点上（比如一个请求的提示词结尾），`mamba_checkpoint_pool.py` 管理这些检查点。

代价是命中的粒度变粗：前缀只能在"存过状态的位置"复用，之后的部分要重新 prefill。多轮对话恰好合适——上一轮结束的位置就是下一轮的前缀，在那里存一份状态，下一轮就能完整命中。

## 投机解码与 PD 分离

- **投机解码要能回滚**：KV 的回滚只需要丢掉被拒绝的草稿 token 对应的 KV；状态却已经被草稿 token 更新过了，无法"减回去"。要么为每个草稿位置保存一份状态（显存 × 草稿数），要么记录每一步的更新量、需要时从某个位置重放（vLLM 的配置里就有为此准备的重放缓冲区）；
- **PD 分离要传状态**：prefill 结束时，除了全注意力层的 KV，还要把每个线性层的状态和卷积缓存发给 decode 实例。状态大小固定、与提示词长度无关，长提示词时传输量比纯注意力模型小得多；
- **CUDA Graph 和批处理**：decode 时每个请求的状态在状态池里有固定的槽位，批处理 kernel 按槽位索引读写，和分页 KV 的块表是同一个思路。

!!! interview "面试怎么答"
    被问到"推理引擎如何支持 Qwen3-Next 这类混合架构"，按三件事回答：**计算**（decode 递推、prefill 分块，块内矩阵乘、块间传状态，分块 prefill 天然支持）；**内存**（分页 KV 池 + 按请求的状态池，状态常用 fp32，短请求反而更占显存，容量要按长度分布规划）；**机制重做**（前缀缓存只能在存过状态的位置命中，vLLM 的 align 模式、SGLang 的混合基数树；投机解码的状态回滚；PD 分离要传状态）。给出"每请求 72 MiB 状态、1000 token 以内不省显存"这类数字，能说明你真的算过。

## 练习

**1. 分块大小怎么选？** 分块写法里，块大小 $C$ 变大或变小，计算量和效率会怎样变化？

??? success "参考答案"
    块内部分是 $C \times C$ 的矩阵乘，总计算量约 $T \cdot C \cdot d$，随 $C$ 线性增长；块间部分每块一次 $d_k \times d_v$ 的状态更新，总量约 $(T/C) \cdot d_k d_v \cdot C = T d_k d_v$，与 $C$ 无关，但串行的步数是 $T/C$。$C$ 太小，矩阵乘太小、用不满 Tensor Core，串行步数多；$C$ 太大，块内的"注意力"计算浪费（本来是线性的，块内却变成了二次的）。实际的 kernel 通常取 64 左右，和 Tensor Core 的分块大小匹配。

**2. 状态用什么精度？** 为什么线性注意力的状态常用 fp32 保存，而 KV Cache 可以用 bf16 甚至 FP8？

??? success "参考答案"
    KV 的每个值只被写一次，量化误差不会累积；状态则在每一步都被"乘上衰减、加上新的外积"，bf16 的舍入误差会随着 token 数一步步累积，长序列时可能明显偏离。另外，状态是汇总量，数值范围随上下文变化，低精度更容易溢出或丢失小的更新。代价是状态池的显存翻倍，这也是本章账本里"短请求更贵"的原因之一。

## 小结

- [x] 线性注意力用固定大小的状态汇总全部历史：decode 递推（常数计算与显存），prefill 分块（块内矩阵乘、块间传状态），两者等价。
- [x] 混合模型的内存 = 全注意力层的分页 KV + 线性层每请求固定的状态（常用 fp32）；长上下文省得多，短请求反而更贵，需要两种内存池一起规划。
- [x] 前缀缓存只能在存过状态检查点的位置命中；检查点很贵，vLLM 的 align 模式、SGLang 的混合基数树都只在特定位置保存。
- [x] 投机解码需要状态回滚（多份状态或重放），PD 分离要额外传输状态和卷积缓存。
