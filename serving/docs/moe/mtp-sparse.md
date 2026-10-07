# MTP 与稀疏注意力：NSA 与 DSA

<p class="lead">这一章讲新一代 MoE 模型里"为推理而设计"的两个结构。<b>MTP</b>（多 token 预测）训练时让模型多预测几个 token，推理时把它当成投机解码的草稿；<b>稀疏注意力</b>（NSA、DSA）让每个 query 只和一小部分历史 token 做注意力。通用的原理在<a href="../../topics/speculative/">投机解码进阶</a>和<a href="../../topics/long-context/">长上下文</a>两章讲过，这里关注它们放进大规模 MoE 推理系统后的账：MTP 什么时候能提速、什么时候反而拖慢；稀疏注意力省下了什么、没省下什么，以及对推理引擎提出了哪些新要求。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. DeepSeek-V3 的 MTP 模块长什么样？推理时怎样当作草稿？
    2. 在大规模 EP 的 decode 里，验证多个草稿 token 的代价落在哪些部分？
    3. 为什么同样的 MTP，长上下文时收益更大？
    4. DSA 的"闪电索引器"是什么？稀疏注意力省下了计算和带宽，省下显存了吗？
    5. 为什么短序列的 prefill 反而不适合用稀疏注意力？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 主模型之后串联一层：把主模型最后一层的隐藏状态和"下一个 token"的嵌入分别归一化、拼接、投影回隐藏维度，再过一个完整的 Transformer 层，用和主模型共享的输出层预测再下一个 token。推理时主模型每步出一个 token，MTP 接着猜下一个，下一步主模型一起验证。
    2. 按请求计费的部分（注意力权重、潜向量 KV 的读取）被 k+1 个 token 分摊，不变；按 token 计费的部分（注意力和专家的计算、EP 的 all-to-all）乘以 k+1。
    3. 长上下文时，按请求计费的 KV 读取占每一步的大头，而这部分被验证的多个 token 分摊了；多出来的按 token 计费的部分相对变小，所以收益更大。
    4. DSA 的闪电索引器是一个很轻的打分器（64 个头、FP8、ReLU 加权），给所有历史 token 打分，每个 query 只对得分最高的 2048 个 token 做 MLA 注意力。它省下了计算和带宽，但没有省显存：所有 token 的 KV 都还要存着，索引器自己的键还多占约 11%。
    5. 短序列时本来就没多少 token，稀疏选择省不了多少计算，索引器的打分和 top-k 反而是额外的开销，所以要回退到稠密的实现。

## MTP：长在模型身上的草稿

DeepSeek-V3 在主模型之后串联了一个 **MTP 模块**：它把主模型最后一层的隐藏状态和"下一个 token"的嵌入分别做归一化、拼起来、线性投影回隐藏维度，再过一个完整的 Transformer 层（含 MoE），最后用和主模型**共享**的输出层预测再下一个 token。训练时它提供额外的训练信号；推理时它就是一个现成的草稿模型：主模型每步产出一个 token，MTP 模块接着猜下一个，下一步主模型一次验证两个。论文报告第二个 token 的接受率在 85%～90% 之间，生成速度（TPS）提升到 1.8 倍。

推理引擎里的用法：SGLang 用 `--speculative-algorithm NEXTN`（沿用 EAGLE 的实现），vLLM 用 `--speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 1}'`。MTP 模块只有一层，可以自回归地多跑几次、产生多个草稿，但它只训练过预测"再下一个"，后面的草稿接受率会下降。

## 在大规模 EP 里，验证 token 贵在哪

[投机解码进阶](../topics/speculative.md#什么时候有效)一章用一个稠密的 7B 模型说明了"batch 越大，投机解码的收益越小"。MoE 大规模 EP 的 decode 有自己的账：验证 $k+1$ 个 token 时，

- **按请求计费**的部分不变：注意力权重、潜向量 KV 都只读一次，$k+1$ 个 query 共享；
- **按 token 计费**的部分乘以 $k+1$：注意力的计算、专家的计算，以及——最关键的——**EP 的 all-to-all**，每个 token 都要发给 8 个专家再收回来。

沿用[上一章](ep-deploy.md#双-batch-重叠)的单层模型（假设双 batch 重叠把通信完全藏在计算后面，每层时间取计算和通信中较长的那个），比较不同上下文长度和 batch 下 MTP 的收益：

```python
HBM, BF16, FP8, NIC = 3.35e12, 989e12, 1979e12, 50e9
ATTN_W, EXPERT_W, EXPERT_FLOP, LOCAL_EXPERTS, LAYERS = 187e6, 44e6, 88e6, 3, 61
TOK_BYTES = (7168 + 224) + 7168 * 2                      # 每个 token 每个专家：dispatch（FP8）+ combine（BF16）
ACCEPT = [0.85, 0.75, 0.65]                              # 第 1、2、3 个草稿 token 在前面都被接受时的接受率（假设）


def layer(seqs, tokens, ctx):
    """一张卡、一层：seqs 个请求、共 tokens 个 token（验证时每个请求 k+1 个），平均上下文 ctx"""
    attn = max((ATTN_W + seqs * ctx * 1152) / HBM, tokens * ctx * 278528 / BF16)   # 潜向量 KV 每个请求只读一次
    moe = max(LOCAL_EXPERTS * EXPERT_W / HBM, tokens * 9 * EXPERT_FLOP / FP8)
    return max(attn + moe, tokens * 8 * TOK_BYTES / NIC)  # 假设双 batch 重叠把 all-to-all 和计算完全重叠


print("上下文   每卡请求数   不用 MTP（token/s/卡）   1 个草稿   2 个草稿   3 个草稿")
for ctx in (4096, 16384):
    for b in (16, 64):
        row = []
        for k in range(4):
            step = LAYERS * layer(b, b * (k + 1), ctx) + k * layer(b, b, ctx)   # 验证 k+1 个 token；MTP 模块自回归跑 k 次
            adv, p = 1.0, 1.0
            for a in ACCEPT[:k]:
                p *= a
                adv += p                                 # 期望前进的 token 数：1 + p1 + p1·p2 + ...
            row.append(b * adv / step)
        print(f"{ctx:>6}   {b:>10}   {row[0]:>20.0f}   " + "   ".join(f"{r / row[0]:>7.2f}x" for r in row[1:]))
```

```text title="输出"
上下文   每卡请求数   不用 MTP（token/s/卡）   1 个草稿   2 个草稿   3 个草稿
  4096           16                   2227      1.82x      1.72x      1.50x
  4096           64                   4716      0.92x      0.82x      0.72x
 16384           16                   1415      1.80x      1.73x      1.56x
 16384           64                   2302      1.30x      1.16x      1.01x
```

- **每卡请求少时**（16），一步的时间主要花在读权重和 KV 上，多验证一个 token 几乎免费，1 个草稿就接近 1.8 倍——和论文报告的数字一致；
- **每卡请求多、上下文短时**（64、4K），all-to-all 已经是瓶颈，验证 token 让通信量翻倍，MTP 反而**变慢**；
- **上下文变长**（16K）后，读 KV 这个"按请求计费"的部分变大，MTP 又有了 1.3 倍的收益。

所以 MTP 不是"打开就快"：它在延迟敏感、每卡 batch 小、上下文长的场景最划算；在追求吞吐、通信已经吃紧的配置下要谨慎，或者只用 1 个草稿。实际系统还会按负载动态调整草稿数，甚至在高负载时关掉投机解码。模型里的接受率、上下文长度、网络带宽都是假设值，结论的方向比具体数字更重要。

## 稀疏注意力：NSA 与 DSA

![图：两种原生稀疏注意力——NSA 的三路加门控，DSA 的轻量索引器加 MLA](../assets/figures/nsa-branches.svg){.aig-svg}

[长上下文](../topics/long-context.md#稀疏注意力每一步动态选择)一章在真实模型上测过：如果每个 query 能精确选出分数最高的 256 个 token，注意力的误差只有 9% 左右——注意力本身是高度稀疏的，难点在于"又快又准地选"。两种训练时就内建稀疏性的方案：

- **NSA**（Native Sparse Attention）：三路注意力加门控合并——**压缩**分支把相邻的一段 token 压成一个粗粒度的键值，给出全局的概览；**选择**分支根据压缩分支的注意力分数挑出最重要的若干个**块**，在块内做细粒度注意力；**滑动窗口**分支保证最近的上下文。选择以块为单位、同一个 GQA 组内的 query 头共享选中的块，这样读 KV 时是连续的大块，对硬件友好；
- **DSA**（DeepSeek Sparse Attention，DeepSeek-V3.2）：在 MLA 之上加一个**闪电索引器**（lightning indexer）。每个 query token 有 64 个索引头、每个 128 维，每个历史 token 有一个 128 维的索引键；索引分数是 $I_{t,s} = \sum_j w_{t,j}\, \text{ReLU}(q^I_{t,j} \cdot k^I_s)$，用 FP8 计算。每个 query 按索引分数选出前 2048 个 token，只对它们做（吸收形式的）MLA 注意力。

索引器仍然要和**所有**历史 token 打分，只是每对的代价小得多。算一下 DSA 相对稠密 MLA 省了多少：

```python
H, DC, ROPE = 128, 512, 64
PAIR_MLA = 2 * H * (DC + ROPE) + 2 * H * DC               # 吸收后的 MLA：每对 (query, key) 的 FLOPs
PAIR_EXPAND = 2 * H * (128 + ROPE) + 2 * H * 128          # 展开后的多头注意力（稠密 prefill 用）
IDX_H, IDX_D, TOPK = 64, 128, 2048                        # DeepSeek-V3.2 的闪电索引器：64 个头、头维 128；每个 query 选 2048 个 token
PAIR_IDX = 2 * IDX_H * IDX_D                              # 索引器：每对 (query, key) 的 FLOPs（FP8）
KV_BYTES, IDX_BYTES = (DC + ROPE) * 2, IDX_D + 4          # 每个 token 缓存：潜向量（bf16）；索引器的键（FP8 + 缩放）
print(f"每对 (query, key)：MLA {PAIR_MLA / 1e3:.0f}K FLOPs，索引器 {PAIR_IDX / 1e3:.0f}K FLOPs；每个 token 的缓存多 {IDX_BYTES / KV_BYTES:.0%}")
print("上下文    decode 每层计算（稠密 → DSA）      decode 每层读取（稠密 → DSA）        prefill 每层计算（稠密 → DSA）")
for L in (4096, 32768, 131072):
    k = min(L, TOPK)
    dec = (L * PAIR_MLA, L * PAIR_IDX + k * PAIR_MLA)
    rd = (L * KV_BYTES, L * IDX_BYTES + k * KV_BYTES)
    pre = (L * L / 2 * PAIR_EXPAND, L * L / 2 * PAIR_IDX + L * k * PAIR_MLA)   # DSA 的 prefill 也按吸收的方式算选中的 token
    print(f"{L // 1024:>4}K   {dec[0] / 1e9:6.2f} → {dec[1] / 1e9:5.2f} GFLOPs（{dec[0] / dec[1]:4.1f} 倍）"
          f"   {rd[0] / 2**20:6.1f} → {rd[1] / 2**20:5.1f} MiB（{rd[0] / rd[1]:4.1f} 倍）"
          f"   {pre[0] / 1e12:7.1f} → {pre[1] / 1e12:6.1f} TFLOPs（{pre[0] / pre[1]:4.1f} 倍）")
```

```text title="输出"
每对 (query, key)：MLA 279K FLOPs，索引器 16K FLOPs；每个 token 的缓存多 11%
上下文    decode 每层计算（稠密 → DSA）      decode 每层读取（稠密 → DSA）        prefill 每层计算（稠密 → DSA）
   4K     1.14 →  0.64 GFLOPs（ 1.8 倍）      4.5 →   2.8 MiB（ 1.6 倍）       0.7 →    2.5 TFLOPs（ 0.3 倍）
  32K     9.13 →  1.11 GFLOPs（ 8.2 倍）     36.0 →   6.4 MiB（ 5.6 倍）      44.0 →   27.5 TFLOPs（ 1.6 倍）
 128K    36.51 →  2.72 GFLOPs（13.4 倍）    144.0 →  18.8 MiB（ 7.7 倍）     703.7 →  215.5 TFLOPs（ 3.3 倍）
```

- **省的是计算和带宽，不是显存**：所有历史 token 的潜向量都要留着（下一个 query 可能选中任何一个），还要多存一份索引器的键（多 11%）。长上下文的**容量**问题仍然要靠 KV 卸载、分层缓存解决，稀疏注意力解决的是**速度**；
- **上下文越长越划算**：decode 在 128K 时计算省 13 倍、读取省近 8 倍；4K 时只省一半左右；
- **短序列的 prefill 反而更贵**：稠密 prefill 用展开的多头注意力（每对 82K FLOPs），而稀疏注意力每个 query 选的 token 不同，只能用吸收的 MQA 形式（每对 279K FLOPs）；序列只有 4K、却要选 2048 个时，省下的对数抵不上每对变贵的 3.4 倍。DeepSeek-V3.2 的报告里专门为短序列的 prefill 实现了一个"带掩码的 MHA 模式"来模拟 DSA，就是这个原因。

## 对推理引擎的新要求

- **两套 KV 缓存**：潜向量 KV 之外，索引器的键也要分页存放、参与前缀缓存和 PD 分离的传输，淘汰和卸载时两者要一起处理；
- **新的 kernel**：索引器打分（FP8 的小矩阵乘 + ReLU + 加权求和）、每个 query 的 top-k 选择（2048 / 数万，要在 GPU 上高效完成）、只读选中 token 的稀疏 MLA 注意力（FlashMLA 等提供了稀疏版本）；
- **选择结果的复用**：相邻层选中的 token 往往相近，有的实现让某些层直接复用上一层的 top-k 结果，省掉索引器的计算（SGLang 的 DSA 配置里就有这样的开关）；
- **调度与容量规划**：稀疏注意力让长上下文的 decode 延迟几乎不随长度增长，但显存占用照样增长——调度器仍然按 KV 容量接纳请求，容量规划要分开看"算得动"和"放得下"。

在源码里：vLLM 的 `v1/attention/backends/mla/` 下有 `flashmla_sparse.py`、`flashinfer_mla_sparse.py` 等稀疏 MLA 后端和 `indexer.py`；SGLang 的 `srt/layers/attention/` 下有 `dsa_backend.py` 和 `dsa/` 目录。

!!! interview "面试怎么答"
    问 MTP，不要只说"投机解码能加速"：讲清它是训练时带的一层草稿、接受率 85%～90%，然后给出条件——验证 token 的代价里，按请求计费的（权重、KV 读取）被分摊，按 token 计费的（计算、EP 的 all-to-all）乘以 k+1，所以小 batch、长上下文收益大，通信吃紧时可能变慢。问 DSA，先讲索引器（64 个头、FP8、ReLU 加权）+ top-2048 选择，再讲账：长上下文 decode 计算省一个数量级，但显存一点没省（还多 11%），短序列 prefill 反而更贵、要回退到稠密实现。

## 练习

**1. MTP 的接受率要多高才划算？** 在本章的模型里，每卡 64 个请求、上下文 4K 时，1 个草稿让一步的时间变成原来的约 2 倍。接受率要多高，MTP 才不亏？

??? success "参考答案"
    1 个草稿时每步期望前进 $1 + p$ 个 token，时间约为原来的 $r$ 倍（这里 $r \approx 2$），不亏的条件是 $(1 + p) / r \ge 1$，即 $p \ge r - 1 \approx 1$——接受率要达到 100% 才能打平，实际上做不到。原因是这个配置下 all-to-all 已经是瓶颈，验证 token 让通信量翻倍。结论：在通信受限的 decode 配置下，MTP 基本不可能带来吞吐收益，只能看是否需要降低单请求的延迟。

**2. 稀疏注意力与前缀缓存。** 两个请求共享一段 100K 的前缀，用 DSA 的模型服务。前缀缓存还能用吗？需要缓存什么？

??? success "参考答案"
    能用。前缀的潜向量 KV 和索引器的键都只依赖前缀本身，和普通前缀缓存一样可以共享——两者都要缓存，而且要一起命中、一起淘汰。不能缓存的是 top-k 的选择结果：它依赖每个 query，新请求的 query 不同，选中的 token 也不同。另外，前缀命中之后，新 token 的 prefill 仍然要让索引器和整个 100K 前缀打分（每对 16K FLOPs），这一部分的代价随前缀长度线性增长。

## 小结

- [x] MTP 是训练时附带的一层草稿模块，第二个 token 的接受率约 85%～90%；SGLang 用 NEXTN、vLLM 用 `deepseek_mtp` 启用。
- [x] 大规模 EP 的 decode 里，验证 token 分摊了按请求计费的部分（权重、KV 读取），但放大了按 token 计费的部分（计算、all-to-all）；小 batch、长上下文时 MTP 收益大，通信受限时可能变慢。
- [x] NSA 用压缩、选择、滑动窗口三路结合；DSA 用 FP8 的闪电索引器给所有历史 token 打分，每个 query 只对前 2048 个做 MLA 注意力。
- [x] 稀疏注意力省计算和带宽、不省显存（索引器的键还多 11%）；长上下文 decode 收益最大，短序列 prefill 反而要回退到稠密实现。
- [x] 推理引擎需要：索引器的分页缓存（参与前缀缓存与传输）、打分和 top-k 的 kernel、稀疏 MLA kernel，以及把"放得下"和"算得动"分开的容量规划。
