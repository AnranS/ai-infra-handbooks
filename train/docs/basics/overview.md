# 分布式训练总论：显存账本与时间模型

<p class="lead">为什么要并行？因为一张卡既放不下，也算不完。这一章先把"放不下"和"算不完"都变成数字：每张卡的显存账本由哪几项组成、各随什么变化；训练要多少算力、多少时间、通信占多少。有了这两本账，后面每一种并行都只是在回答同一个问题：把哪一项切开、代价是多少通信。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 用 bf16 混合精度 + Adam 训练一个 70B 模型，每个参数要占多少字节？
    2. 训练时的激活和哪些量成正比？怎样减少？
    3. 训练 1T token 的 70B 模型大约需要多少 FLOPs？
    4. 数据并行、张量并行、流水线并行、上下文并行、专家并行分别切的是什么？
    5. 为什么张量并行通常只在一台机器内部做？

## 显存账本

训练时每张卡上的显存由四部分组成（$\Psi$ 为这张卡负责的参数量）：

| 项目 | 大小 | 说明 |
| --- | --- | --- |
| 参数 | $2\Psi$ | bf16 |
| 梯度 | $2\Psi$ | bf16 |
| 优化器状态 | $12\Psi$ | fp32 的主参数、Adam 的一阶矩和二阶矩 |
| 激活 | 与 批大小 × 序列长度 × 隐藏维度 × 层数 成正比 | 前向保存下来、给反向用的中间结果 |

前三项合称**模型状态**，每个参数 16 字节。激活的精确估算（Megatron 论文的公式，bf16、用 FlashAttention）是每层 $34\,sbh$ 字节，$s$ 序列长度、$b$ micro-batch 大小、$h$ 隐藏维度；张量并行（开序列并行）能把它除以 TP 的度数，全量重计算能把它降到每层 $2\,sbh$（只存每层的输入）。

用这几条规则算一个 70B 模型在 64 张 80 GB 的卡上、8K 序列的训练：

```python title="ledger.py"
GiB = 2**30

# Llama-3-70B 量级的稠密模型
PSI, LAYERS, HIDDEN = 70.6e9, 80, 8192
SEQ, MBS = 8192, 1            # 序列长度、micro-batch 大小


def model_states(psi, zero, dp):
    """混合精度 + Adam：bf16 参数 2 + bf16 梯度 2 + fp32 主参数/一阶矩/二阶矩 12 = 16 字节/参数"""
    return {0: 16 * psi, 1: 4 * psi + 12 * psi / dp, 2: 2 * psi + 14 * psi / dp, 3: 16 * psi / dp}[zero]


def activations(layers, tp=1, sp=False, recompute=False, inflight=1):
    """每层激活（bf16，用 FlashAttention）：不切分 34·sbh；TP 不开 SP 时 sbh·(10 + 24/t)；开 SP 时 34·sbh/t；全量重计算 2·sbh/t"""
    sbh = SEQ * MBS * HIDDEN
    if recompute:
        per = 2 * sbh / tp
    elif sp:
        per = 34 * sbh / tp
    else:
        per = sbh * (10 + 24 / tp)
    return per * layers * inflight


configs = [
    ("DP=64，不切分", dict(zero=0, dp=64, tp=1, pp=1)),
    ("DP=64，ZeRO-3", dict(zero=3, dp=64, tp=1, pp=1)),
    ("DP=64，ZeRO-3 + 全量重计算", dict(zero=3, dp=64, tp=1, pp=1, recompute=True)),
    ("TP=8（SP）× PP=4 × DP=2，ZeRO-1", dict(zero=1, dp=2, tp=8, pp=4, sp=True)),
]
for name, c in configs:
    psi_local = PSI / (c["tp"] * c["pp"])                  # 每张卡负责的参数
    states = model_states(psi_local, c["zero"], c["dp"])
    layers_local = LAYERS // c["pp"]
    inflight = c["pp"]                                     # 1F1B：第一个 stage 最多同时保存 pp 个 micro-batch 的激活
    act = activations(layers_local, c["tp"], c.get("sp", False), c.get("recompute", False), inflight)
    total = (states + act) / GiB
    print(f"{name}：模型状态 {states / GiB:.1f} GiB，激活 {act / GiB:.1f} GiB，合计 {total:.1f} GiB，"
          f"{'放得下' if total < 80 * 0.9 else '放不下'}")

TOKENS = 1e9
flops = 6 * PSI * TOKENS
seconds = flops / (64 * 989e12 * 0.40)
print(f"训练 10 亿 token：{flops:.2e} FLOPs，64 张 H100、MFU 40% 需要 {seconds / 3600:.1f} 小时")
```

```text title="输出"
DP=64，不切分：模型状态 1052.0 GiB，激活 170.0 GiB，合计 1222.0 GiB，放不下
DP=64，ZeRO-3：模型状态 16.4 GiB，激活 170.0 GiB，合计 186.4 GiB，放不下
DP=64，ZeRO-3 + 全量重计算：模型状态 16.4 GiB，激活 10.0 GiB，合计 26.4 GiB，放得下
TP=8（SP）× PP=4 × DP=2，ZeRO-1：模型状态 20.5 GiB，激活 21.2 GiB，合计 41.8 GiB，放得下
训练 10 亿 token：4.24e+20 FLOPs，64 张 H100、MFU 40% 需要 4.6 小时
```

四行结论几乎就是整本书的提纲：

1. **什么都不切**：光模型状态就要 1 TB，一张卡放不下——数据并行本身不省显存；
2. **ZeRO-3** 把模型状态切到 64 张卡上，只剩 16 GiB，但 8K 序列的激活有 170 GiB——激活成了新的瓶颈；
3. **全量重计算**把激活压到 10 GiB，放得下了，代价是多做约 33% 的计算（每层前向重算一遍）；
4. **张量并行 + 序列并行 + 流水线并行**同时切模型状态和激活，不需要重计算也放得下，代价是 TP 的高频通信和 PP 的气泡。

后面的章节会逐个实现这些手段，并量出各自的代价。

## 时间模型

训练一个 token 的计算量约是 $6N$（前向 $2N$、反向 $4N$，$N$ 为参数量），整个训练约 $6ND$ 次浮点运算。实际时间取决于 **MFU**（模型 FLOPs 利用率）：

$$T = \frac{6ND}{\text{卡数} \times \text{单卡峰值} \times \text{MFU}}$$

上面的例子里，64 张 H100 训练 10 亿 token 要 4.6 小时；训练 15T token 就是约 8 年——所以前沿模型用上万张卡。MFU 通常在 30%～50% 之间，损失的部分来自：

- **通信没有被计算藏住**：梯度同步、TP 的 all-reduce、PP 的点对点传输、EP 的 all-to-all；
- **流水线气泡**：部分 stage 在等待；
- **重计算**：多做的前向不算进 MFU；
- **小 kernel 和 CPU 开销**：逐元素运算、优化器步骤、数据加载。

每种并行都在"显存"和"时间"之间换东西，下一节是它们的一张总表。

## 五种并行一览

| 并行 | 切的是什么 | 每步通信什么 | 通信量级 | 通常在哪一层网络上 |
| --- | --- | --- | --- | --- |
| 数据并行（DP / ZeRO） | 数据（batch）；ZeRO 再切模型状态 | 梯度的 all-reduce（ZeRO：reduce-scatter + all-gather） | 每步约 $2\Psi$～$3\Psi$ 字节，可以和反向重叠 | 跨节点（最外层） |
| 张量并行（TP）+ 序列并行（SP） | 每一层的权重矩阵；SP 再切 LayerNorm 等处的激活 | 每层 2 次 all-reduce（或 reduce-scatter + all-gather），在关键路径上 | 每层约 $4\,sbh$ 字节 | 节点内（NVLink） |
| 流水线并行（PP） | 层（按深度切成若干 stage） | 相邻 stage 之间的激活和梯度（点对点） | 每个 micro-batch 约 $sbh$ 字节 | 可以跨节点 |
| 上下文并行（CP） | 序列 | 注意力所需的 KV（ring）或按头的 all-to-all（Ulysses） | 与序列长度成正比 | 节点内或跨节点 |
| 专家并行（EP） | MoE 的专家 | token 的 all-to-all（dispatch 与 combine） | 每层约 $2 \times$ top-k $\times sbh$ 字节 | 节点内或跨节点 |

"张量并行只在节点内"的原因就在这张表里：它的通信频繁（每层两次）、在关键路径上（不能和计算完全重叠）、量大，只有 NVLink（单向几百 GB/s）扛得住；跨节点的 InfiniBand / RoCE 每张卡只有几十 GB/s，适合通信量小、或者能和计算重叠的 DP 和 PP。
把几种并行组合起来就是 3D（DP × TP × PP）乃至 5D（再加 CP、EP）并行，组合的方法见[3D / 5D 并行的组合与选择](../practice/strategy.md)。

!!! interview "面试怎么答"
    被问"训练一个 70B 模型要多少资源"，先算显存账本：bf16 混合精度加 Adam 每个参数 16 字节（bf16 参数和梯度 + fp32 主权重与两个矩），70B 就是 1.1 TB，单卡放不下；激活每层约 $34\,sbh$ 字节，和序列长度、batch 成正比。再算时间：训练约 $6ND$ 次运算，70B、1T token 约 $4.2 \times 10^{23}$，按实测 MFU 换算成卡时。最后讲各种并行切的是什么（DP 切数据、TP 切层内矩阵、PP 切层、CP 切序列、EP 切专家），以及摆放原则：通信频繁、在关键路径上的 TP 放在节点内，能重叠、量小的 DP、PP 放在节点之间。

## 练习

1. 一个 8B 模型（32 层，隐藏维度 4096），在 8 张 80 GB 的卡上训练，序列长度 4096、micro-batch 2。只用数据并行 + ZeRO-2，不做重计算，每张卡需要多少显存？放得下吗？如果放不下，你会先加什么？

??? success "参考答案"
    模型状态：ZeRO-2 下 $2\Psi + 14\Psi/8 = 16 + 14 = 30$ GB（约 27.9 GiB）。激活：$34 \times 4096 \times 2 \times 4096 \times 32 = 36.5$ GB（约 34 GiB）。合计约 62 GiB，在 80 GB 的卡上（留 10% 余量）勉强放得下。
    如果还要加长序列或加大 micro-batch，先加**选择性重计算**（只重算注意力部分）或全量重计算，而不是上张量并行：8B 模型在 8 卡上用 TP 的通信代价不划算。

2. 为什么"数据并行不省显存"，却几乎所有大规模训练都用它？

??? success "参考答案"
    数据并行的目的是**加速**：每张卡处理不同的数据，吞吐随卡数线性增长；而它的通信（梯度同步）每步只有一次、量是固定的 $2\Psi$ 量级，并且可以和反向计算重叠，扩展效率很高。
    显存问题交给 ZeRO（切模型状态）和其他并行；最终的配置里，数据并行几乎总是最外层、跨节点的那一维。

## 小结

- [x] 显存账本：模型状态每参数 16 字节（bf16 参数和梯度 + fp32 优化器状态），激活每层约 $34\,sbh$ 字节。
- [x] ZeRO 切模型状态，TP + SP 切模型状态和激活，重计算用计算换激活，PP 按层切。
- [x] 训练计算量约 $6ND$；实际时间看 MFU，损失来自未被藏住的通信、气泡、重计算和小 kernel。
- [x] 通信频繁、在关键路径上的并行（TP）放在节点内；能重叠、量小的（DP、PP）放在节点之间。
