# 投机解码的新做法：整块草稿、按负载验证与 PD 分离

<p class="lead"><a href="../../topics/speculative/">投机解码</a>一章讲了经典的做法：草稿模型逐个给出 K 个 token（EAGLE、MTP），目标模型一次验证，用树形草稿提高命中。它在低负载时效果很好，但负载一高就会变慢——验证的 token 和真实的 token 抢算力。2026 年的推理框架在三个方向上改进了它：<b>一次前向给出一整块草稿</b>（PARD、DFlash、DSpark），<b>按负载决定验证多少</b>（动态 K、自适应验证），以及和 <b>PD 分离、大规模部署的配合</b>。这一章用一个简单的代价模型把这些改进串起来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么 batch 越大，投机解码的最佳草稿长度越短？什么时候干脆不投机？
    2. DFlash 怎样一次前向给出一整块草稿？DSpark 在它的基础上加了什么，为什么要加？
    3. vLLM 的自适应验证怎样决定验证哪些草稿 token？它需要草稿模型额外提供什么？
    4. PD 分离时，decode 实例要开始投机解码，需要 prefill 实例额外传什么？
    5. 为什么"按负载调整 K"和数据并行放在一起时要特别小心？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 投机解码用算力换步数：batch 小时 GPU 算力大量闲置，验证多几个 token 几乎免费；batch 大时计算已经吃紧，验证的草稿 token 和真实 token 抢算力，多验证的代价超过了接受带来的收益。负载高到一定程度（本章的 8B 模型 batch 512 时），最佳的 K 是 0，就不该投机。
    2. DFlash 把目标模型的隐藏状态投影成草稿模型的上下文 K、V，query 只有"下一个 token + K 个掩码位置"，块内用非因果注意力，一次前向给出整块。DSpark 在这个骨干上加了马尔可夫头（按上一个选出的 token 给下一个位置的 logits 加一个低秩的转移偏置，补上块内位置之间的依赖）和置信度头（估计每个位置被接受的概率，供自适应验证使用）。
    3. 每个（请求，位置）的得分是它的存活概率——置信度沿位置的累乘；把所有请求的所有位置按得分排序，从高到低放行，放行多少个由启动时测出的代价模型决定，让"期望产出的 token 数 / 这一步的耗时"最大。它需要草稿模型提供每个位置的置信度（目前只有 DSpark 带置信度头）。
    4. 目标模型最后一层的隐藏状态（草稿模型的输入）和第一轮的草稿（例如 top-k），让 decode 实例一接手就能开始投机，而不用先空跑一步。
    5. 数据并行（尤其是 DP Attention + EP）时各 rank 在每一层的 all-to-all 里同步，每个 rank 的 token 数要一致或可预期；各 rank 自己按负载调整 K，会让验证的 token 数不同、互相等待。要让所有 rank 的 K 保持一致。

## 负载越高，K 越要小

投机解码的收益来自一个前提：decode 是访存受限的，一次前向多算几个 token 几乎不花时间。batch 大了，一次前向里已经有很多真实 token，前向变成算力受限，每多验证一个草稿 token 就要多花一份算力，而被拒绝的那些全是浪费。用一个 8B 模型在 H100 上的简单代价模型看最佳的草稿长度 $K$：

```python
# 一个 8B 模型在 H100 上：每步读一遍权重（访存下限），token 多了以后变成算力受限
W_BYTES, BW = 16e9, 3.35e12 * 0.8                  # bf16 权重、有效带宽
FLOP_PER_TOKEN, FLOPS = 2 * 8e9, 989e12 * 0.5      # 每 token 的 FLOPs、有效算力
T_DRAFT = 0.25e-3                                  # 草稿模型跑一次前向（一层的 EAGLE 头或小模型，访存受限）
ALPHA = 0.8                                        # 前面的草稿都被接受时，下一个位置被接受的概率


def step_time(tokens):
    return max(W_BYTES / BW, tokens * FLOP_PER_TOKEN / FLOPS)


def throughput(batch, k, parallel):
    """每秒生成的 token 数：每步 = 出草稿（逐个出 k 次 / 一次出整块）+ 验证（batch×(1+k) 个 token 一次前向）"""
    accepted = sum(ALPHA ** i for i in range(1, k + 1))          # 期望接受的草稿数；另外验证时总会多得 1 个
    draft = 0 if k == 0 else (T_DRAFT if parallel else k * T_DRAFT)
    return batch * (1 + accepted) / (draft + step_time(batch * (1 + k)))


for batch in (1, 32, 128, 512):
    base = throughput(batch, 0, False)
    best = {p: max(range(0, 9), key=lambda k: throughput(batch, k, p)) for p in (False, True)}
    print(f"batch {batch:>3}：不投机 {base:>7.0f} tok/s；逐个出草稿 最佳 K={best[False]}（{throughput(batch, best[False], False) / base:.2f} 倍）；"
          f"一次出整块 最佳 K={best[True]}（{throughput(batch, best[True], True) / base:.2f} 倍）")
```

```text title="输出"
batch   1：不投机     168 tok/s；逐个出草稿 最佳 K=8（3.24 倍）；一次出整块 最佳 K=8（4.15 倍）
batch  32：不投机    5360 tok/s；逐个出草稿 最佳 K=5（2.95 倍）；一次出整块 最佳 K=5（3.41 倍）
batch 128：不投机   21440 tok/s；逐个出草稿 最佳 K=1（1.26 倍）；一次出整块 最佳 K=1（1.26 倍）
batch 512：不投机   30906 tok/s；逐个出草稿 最佳 K=0（1.00 倍）；一次出整块 最佳 K=0（1.00 倍）
```

- **最佳 K 随负载下降**：batch 1 时 K 越大越好（这里搜到 8 为止），batch 128 只剩 1，batch 512 时任何草稿都是亏的。一个部署如果白天高峰、夜里低谷，固定的 K 总有一段时间是错的；
- **逐个出草稿的代价随 K 线性增长**：K 个草稿要跑 K 次草稿模型，K 大时这部分占了一步耗时的相当比例；一次出整块的草稿模型（下一节）把这部分压成一次前向，低负载时多赚了三成；
- 高负载时两者都不如不投机——这是下面"按负载调整"的出发点。

## 一次前向给出一整块草稿

逐个出草稿慢，是因为第 $i$ 个草稿要等第 $i-1$ 个出来。**并行草稿**让草稿模型一次前向同时预测后面 K 个位置：

- **PARD**（Parallel Draft Model）：在草稿模型的输入末尾放 K 个"占位"token，训练它一次预测出这 K 个位置。vLLM 里用 `"parallel_drafting": true` 打开（适用于独立草稿模型和 EAGLE）；
- **DFlash**：草稿模型不自己算上下文，而是把目标模型的隐藏状态投影成上下文的 K、V（"上下文 KV 预计算"）；query 只有"下一个 token + K 个掩码位置"这一小块，块内用**非因果**注意力（每个位置都能看到块里的其他位置），一次前向给出整块；
- **DSpark**：在 DFlash 的并行骨干上加两个小头：一个**马尔可夫头**（一对低秩矩阵 $V \times r$、$r \times V$，按上一个选出的 token 给下一个位置的 logits 加一个"转移偏置"），从左到右顺序采样，补上块内位置之间的依赖；一个**置信度头**，为每个位置估计被接受的概率，供下一节的自适应验证使用。DeepSeek-V4 发布了 DSpark 草稿（V4-Flash-DSpark），vLLM 的实现（`models/deepseek_v4/nvidia/dspark.py`）借用稀疏注意力的 top-k 索引，把"未来的"query 位置也加进每个 query 的可见集合，从而实现块内的非因果注意力。

为什么要加马尔可夫头：并行预测的各个位置互不知道对方选了什么。如果下一个 token 有两种同样可能的接法，第 2 个位置在两条路上各有不同的最佳答案，独立地选就可能拼出一条"前后不搭"的草稿。用一条马尔可夫链当目标模型看看差别：

```python
import torch

torch.manual_seed(0)
V, K = 64, 6
# 目标模型：一阶马尔可夫链；每个 token 后面大多是两个"常见接法"之一（各 45%），其余 10% 分给别的 token
P = torch.full((V, V), 0.1 / (V - 2))
for a in range(V):
    b, c = torch.randperm(V)[:2]
    P[a, b], P[a, c] = 0.45, 0.45


def expected_accept(draft, last):
    """目标模型按分布采样时，草稿被逐个接受的期望个数 = Σ_i（草稿前 i 个 token 恰好是目标生成的那条路的概率）"""
    p, total, prev = 1.0, 0.0, last
    for x in draft:
        p *= P[prev, x].item()
        total += p
        prev = x
    return total


def parallel_draft(last):
    """一次出整块、位置之间互不相干：第 i 个位置取 i 步之后的边缘分布里最可能的 token"""
    dist, out = torch.zeros(V), []
    dist[last] = 1
    for _ in range(K):
        dist = dist @ P
        out.append(int(dist.argmax()))
    return out


def markov_draft(last):
    """再用一个很小的"转移头"从左到右补上依赖：第 i 个位置看第 i-1 个位置选了什么"""
    out, prev = [], last
    for _ in range(K):
        prev = int(P[prev].argmax())
        out.append(prev)
    return out


par = sum(expected_accept(parallel_draft(t), t) for t in range(V)) / V
mar = sum(expected_accept(markov_draft(t), t) for t in range(V)) / V
print(f"每步期望接受的草稿数：各位置互不相干 {par:.2f}，加上位置间的转移 {mar:.2f}")
```

```text title="输出"
每步期望接受的草稿数：各位置互不相干 0.62，加上位置间的转移 0.81
```

马尔可夫头只是一对低秩矩阵，按位置顺序跑也几乎不花时间（vLLM 把它的权重在各个 rank 上复制，避免每个位置都做一次 all-reduce）；草稿的主体仍然一次前向完成。这就是"半自回归"：重的部分并行，轻的部分串行。

## 按负载决定验证多少

三个框架层面的做法，从粗到细：

- **动态 K**（vLLM 的 `num_speculative_tokens_per_batch_size`）：按并发数分段配置 K，例如 1～64 个并发用 3、65～128 用 1、更高就不投机。RL rollout 很适合：一开始 batch 很大，最后只剩几条长尾请求时 K 自动变大；
- **按接受长度调整**（SGLang 的 adaptive spec，`speculative/adaptive_spec_params.py`）：按 batch 大小分档，每档有几个候选的草稿步数，运行时根据观测到的平均接受长度在候选之间切换（带迟滞，避免来回跳），目前支持 topk=1 的 EAGLE / EAGLE3；
- **自适应验证**（vLLM，目前只支持带置信度头的 DSpark）：草稿照常出一整块，但**每个 (请求, 位置) 单独决定验不验**。一个位置的得分是它的**存活概率**——前面所有位置都被接受的概率，即置信度的累乘；所有请求的所有位置放在一起按得分排序，从高到低放行，放行多少个由启动时测出的代价模型决定，让"期望产出的 token 数 / 这一步的耗时"最大。

用上一节的代价模型实现自适应验证，一半请求的草稿很准、一半不太准：

```python
import torch

torch.manual_seed(0)
K = 7                                               # 草稿模型一次给出 7 个位置


def plan(conf):
    """每个 (请求, 位置) 的得分 = 存活概率（置信度沿位置累乘）；按得分从高到低放行 b 个，b 取让吞吐最大的那个"""
    batch = conf.shape[0]
    survival = conf.cumprod(1)                                  # 每行单调不增，所以放行的总是每个请求的一段前缀
    order = survival.flatten().sort(descending=True).values
    gain = torch.cat([torch.zeros(1), order.cumsum(0)])         # 放行前 b 个位置时，期望多接受多少个 token
    rates = [(batch + gain[b]) / (T_DRAFT + step_time(batch + b)) for b in range(len(gain))]
    b = max(range(len(gain)), key=lambda i: rates[i])
    return b, rates[b], survival


for batch in (16, 128, 512):
    # 一半请求草稿很准（代码、复述），一半不太准（开放式创作）：每个位置的置信度
    conf = torch.cat([torch.full((batch // 2, K), 0.9), torch.full((batch - batch // 2, K), 0.55)]) * torch.rand(batch, K).mul(0.2).add(0.9)
    conf = conf.clamp(max=0.99)
    b, rate, survival = plan(conf)
    fixed = (batch + survival.sum()) / (T_DRAFT + step_time(batch * (1 + K)))
    none = batch / step_time(batch)
    kept = (survival >= survival.flatten().sort(descending=True).values[b - 1]).sum(1).float() if b else torch.zeros(batch)
    print(f"batch {batch:>3}：放行 {b:>4} 个草稿位置（准的请求平均 {kept[:batch // 2].mean():.1f} 个，不准的 {kept[batch // 2:].mean():.1f} 个）；"
          f"吞吐 不投机 {none:>6.0f}、全部验证 {fixed:>6.0f}、自适应 {rate:>6.0f} tok/s")
```

```text title="输出"
batch  16：放行  112 个草稿位置（准的请求平均 7.0 个，不准的 7.0 个）；吞吐 不投机   2680、全部验证  10104、自适应  10104 tok/s
batch 128：放行   57 个草稿位置（准的请求平均 0.9 个，不准的 0.0 个）；吞吐 不投机  21440、全部验证  15182、自适应  29053 tok/s
batch 512：放行    7 个草稿位置（准的请求平均 0.0 个，不准的 0.0 个）；吞吐 不投机  30906、全部验证  15146、自适应  30448 tok/s
```

- **低负载全部验证**：还在访存受限区，多验证几乎免费；
- **中等负载按请求挑**：batch 128 时只放行 57 个位置，几乎都给了草稿准的请求的第一个位置——同一步里，一个请求验证好几个 token、另一个一个都不验，这是固定 K 做不到的。吞吐比全部验证高近一倍，也比不投机高 35%；
- **高负载几乎不验**：只剩草稿本身的开销，吞吐略低于不投机。所以实际部署会把它和"高负载时干脆不出草稿"结合起来。

自适应验证对引擎的要求也很具体（vLLM 的文档列出了限制）：每个请求这一步验证几个 token 是在 GPU 上决定的，CPU 侧只知道上限，所以**注意力后端必须接受由设备决定的 query 长度**；步耗时的代价曲线是在启动时对捕获好的 CUDA Graph 测出来的，所以**必须用完整的 CUDA Graph**，不能 `--enforce-eager`；暂不支持 LoRA（LoRA 的逐 token 映射在 CPU 上构建）和流水线并行（代价曲线和置信度只在最后一个 stage 上）。

## 和其他机制的配合

- **PD 分离**：decode 实例拿到 KV 和第一个输出 token 之后，要立刻开始投机解码，还需要草稿模型第一轮的输入。SGLang 的做法是 prefill 实例在传 KV 时一并传过去最后一个位置的隐藏状态和草稿的 top-k（`disaggregation/utils.py` 里的元数据缓冲区），decode 端用它们拼出草稿输入；EAGLE、DSpark、DFlash 各有一个实现（`speculative/*_disaggregation.py`）；
- **不带 KV 的草稿层**：SGLang 的 Frozen-KV MTP 让 MTP 草稿层只读目标模型的 KV Cache，自己不维护 KV 池，省下草稿的那份显存；
- **数据并行**：K 变了，一步前向的 token 数就变了。数据并行的各个 rank 各自调度，如果各自选了不同的 K，跨 rank 的集合通信（例如 DP attention 之后的专家并行）就会对不上甚至死锁。vLLM 在开启数据并行时自动关闭动态 K；SGLang 的 adaptive spec 不支持 DP attention，也不支持双 batch 重叠；
- **有状态的模型**：压缩器状态（DeepSeek-V4）、线性注意力状态（Kimi-K3、Qwen3-Next）都会被草稿 token 更新，被拒绝时要回滚，见[新一代开源模型](new-models.md)和[线性注意力](linear-attn.md#投机解码与-pd-分离)两章。

!!! interview "面试怎么答"
    先讲原理上的矛盾：投机解码用算力换步数，低负载时算力是闲的，高负载时草稿 token 和真实 token 抢算力，所以最佳 K 随 batch 下降，高到一定程度就不该投机——用"8B 模型 batch 1 时 K=8 快 3～4 倍、batch 512 时 K=0"这类数字说明。再讲三个改进：**整块草稿**（PARD、DFlash 一次前向出 K 个；DSpark 再加一个低秩的马尔可夫头补上位置间的依赖，加一个置信度头）；**按负载验证**（动态 K、SGLang 按接受长度调步数、vLLM 按存活概率在所有请求的所有位置里挑、预算由启动时测的代价模型决定）；**工程配合**（PD 分离要传隐藏状态和草稿 top-k，数据并行时 K 要一致，有状态的模型要能回滚）。

## 练习

**1. 为什么存活概率排序得到的放行集合，总是每个请求的一段前缀？**

??? success "参考答案"
    同一个请求第 $i$ 个位置的存活概率是 $c_1 c_2 \cdots c_i$，每个因子都不超过 1，所以沿位置单调不增：第 $i+1$ 个位置的得分不会高于第 $i$ 个。按得分从高到低放行时，一个请求的第 $i+1$ 个位置被放行，第 $i$ 个一定已经放行。这正好和验证的语义一致——前面的草稿被拒绝后，后面的即使验证了也没用。

**2. 自适应验证的代价曲线为什么要在启动时测，而不是用公式算？**

??? success "参考答案"
    真实的步耗时不只是"访存与算力取大"：注意力的耗时和上下文长度、稀疏注意力的索引器有关，CUDA Graph 按几个固定的大小捕获（token 数补齐到最近的档位，耗时是台阶状的），MoE 的耗时和路由分布有关，不同 GPU、不同并行配置也不一样。启动时直接对捕获好的图测每个大小的耗时最准。vLLM 默认按 8192 个 token 的上下文测，长上下文的部署可以用 `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` 调大（对 DeepSeek-V4 这类稀疏注意力模型影响较小，随上下文增长的主要是便宜的索引器）。

**3. 在上面的代价模型里，把草稿模型的一次前向从 0.25 ms 调到 1 ms，最佳 K 会怎样变化？**

??? success "参考答案"
    逐个出草稿的代价变成每个位置 1 ms，K 的收益被草稿本身吃掉，低负载时的最佳 K 明显变小；一次出整块的草稿只多付一次 1 ms，受影响小得多。这说明草稿模型越大，并行草稿的优势越明显——并行草稿可以用更大、更准的草稿模型而仍然划算。可以直接改 `T_DRAFT` 跑一遍验证。

## 小结

- [x] 投机解码用算力换步数：最佳 K 随负载下降，高负载时不该投机；固定的 K 总有一段时间是错的。
- [x] 整块草稿：PARD 和 DFlash 一次前向出 K 个草稿，DSpark 再加低秩的马尔可夫头补上块内依赖、加置信度头估计接受概率。
- [x] 按负载验证：动态 K、SGLang 按接受长度调步数、vLLM 按存活概率在所有请求的所有位置里挑，预算来自启动时测的代价曲线。
- [x] 工程配合：PD 分离要传隐藏状态和草稿 top-k，数据并行时各 rank 的 K 要一致，有状态的模型要能回滚草稿的更新。
