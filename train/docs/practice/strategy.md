# 3D / 5D 并行的组合与选择

<p class="lead">前面每一章讲一种并行，真实的训练总是几种一起用：TP 在节点内切层内的矩阵，CP 切序列，PP 按深度切，DP 在最外层扩展，MoE 模型再加上 EP。这一章回答两个问题：几种并行怎样在卡上排布（哪一维放在节点内），以及给定模型、序列长度、全局 batch 和卡数，怎样选出一组合理的度数。最后用一个几十行的搜索脚本把前面各章的显存和时间公式合在一起，看看它会选出什么、为什么。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 5 种并行在卡上的排布顺序是怎样的？为什么 TP 总在最内层？
    2. 为一个模型选并行配置，一般按什么步骤？
    3. 为什么 TP 的度数越大，通信占的比例越高？
    4. 卡数不断增加、全局 batch 不能再大时，会发生什么？
    5. 流水线的首尾 stage 为什么常常少放几层？

## 维度的排布：谁放在节点内

把卡看成一个多维数组（device mesh），每一维对应一种并行；沿某一维的一条"线"上的卡组成这种并行的通信组。rank 号相邻的卡在同一个节点里，所以**最内层的维度落在节点内**：

![图：3D 并行在集群上的排布](../assets/figures/parallelism-3d.svg){.aig-svg}

```python title="mesh.py"
import numpy as np

GPUS_PER_NODE = 8
TP, CP, DP, PP = 2, 2, 2, 2                    # 16 张卡：2 个节点

# 由外到内 PP → DP → CP → TP：最内层的维度 rank 号相邻，落在同一个节点里
mesh = np.arange(PP * DP * CP * TP).reshape(PP, DP, CP, TP)
axes = {"TP": 3, "CP": 2, "DP": 1, "PP": 0}

for name, axis in axes.items():
    groups = np.moveaxis(mesh, axis, -1).reshape(-1, mesh.shape[axis])   # 沿这一维的每一条"线"是一个通信组
    local = all(len({r // GPUS_PER_NODE for r in g}) == 1 for g in groups)
    print(f"{name} 组：{' '.join(str(g.tolist()) for g in groups[:4])} …共 {len(groups)} 组，"
          f"{'都在节点内' if local else '跨节点'}")

rank = 13
pp, dp, cp, tp = (int(i[0]) for i in np.nonzero(mesh == rank))
print(f"rank {rank} 的坐标：PP={pp} DP={dp} CP={cp} TP={tp}")
```

```text title="输出"
TP 组：[0, 1] [2, 3] [4, 5] [6, 7] …共 8 组，都在节点内
CP 组：[0, 2] [1, 3] [4, 6] [5, 7] …共 8 组，都在节点内
DP 组：[0, 4] [1, 5] [2, 6] [3, 7] …共 8 组，都在节点内
PP 组：[0, 8] [1, 9] [2, 10] [3, 11] …共 8 组，跨节点
rank 13 的坐标：PP=1 DP=1 CP=0 TP=1
```

Megatron-LM 和 torchtitan 建通信组的方式就是这样：Megatron 用 `--tensor-model-parallel-size` 等参数加一个顺序字符串（默认 `tp-cp-ep-dp-pp`，由内到外），torchtitan 用 PyTorch 的 `init_device_mesh("cuda", (pp, dp, cp, tp), mesh_dim_names=(...))`，再从 mesh 里按名字取出每一维的进程组。

排布的原则来自[总论](../basics/overview.md)里那张表：

- **TP 永远在最内层**：每层前向、反向各两次通信，在关键路径上，只有 NVLink 扛得住；所以 TP 的度数一般不超过一个节点的卡数（8）；
- **CP 次之**：ring 的 KV 传输可以和注意力计算重叠，但序列不太长时藏不完，能放在节点内最好；
- **DP 和 PP 在外层**：DP 的梯度同步每步一次、能和反向重叠；PP 只在相邻 stage 之间点对点传激活，量小。两者谁在最外层都有人用——Megatron 默认 PP 在最外层；Llama 3 的顺序是 `[TP, CP, PP, DP]`，把 DP（FSDP）放在最外层，因为 FSDP 的通信能提前发起、最能容忍跨节点的延迟；
- **EP 借用 DP 的维度**：MoE 层的专家并行不额外占卡，而是把注意力部分的 DP（×CP）那批卡重新分组（见[MoE 与专家并行](../model/moe-ep.md)）。

## 选择的步骤

大多数团队的做法可以归纳成下面几步（HuggingFace 的 *Ultra-Scale Playbook* 用大量实测给出了同样的流程）：

1. **先让模型状态放下**：从 DP + ZeRO-1（分布式优化器）开始；如果连 bf16 的参数和梯度（每参数 4 字节）都放不下，在节点内加 TP；一个节点的 TP 还不够，再加 PP 跨节点切层，或者改用 ZeRO-3 / FSDP；
2. **再让激活放下**：TP 一定配 SP；序列很长时加 CP；剩下的缺口用**部分层**的全量重计算补上——只重算放不下的那几层；
3. **用 DP 扩展到目标卡数**：剩下的卡都给 DP；但全局 batch 有上限（太大会损害收敛），DP × micro-batch 大小 × 每条流水线的 micro-batch 数 = 全局 batch，DP 不能无限增长；
4. **调流水线**：micro-batch 数要远大于 stage 数才能摊薄气泡；首尾 stage 少放层（嵌入层、LM head 和损失在那里）；气泡还大就用交错调度或零气泡调度（见[流水线并行](../model/pipeline.md)）；
5. **实测**：用 profiler 看每一步里计算、各类通信和空闲各占多少，和估算对照，再调。

前四步都可以先在纸面上估算——这就是下面这个脚本做的事。

## 一个配置搜索脚本

脚本把前面各章的公式合在一起：显存用[总论](../basics/overview.md)的账本（ZeRO-1、每层激活 $34\,sbh/t$、部分层重计算）；时间按"一个 micro-batch 在最慢的 stage 上要多久"累加，其中 TP 的 all-reduce 不重叠、CP 的 ring 与注意力计算重叠、DP 的梯度同步大部分被反向藏住、PP 多出 $p-1$ 个 micro-batch 的气泡，最后一个 stage 多算一个 LM head。枚举所有 TP × CP × PP 的组合，按估算的 MFU 排序：

```python title="parallel_plan.py"
from dataclasses import dataclass
from itertools import product

GiB = 2**30
PEAK, EFF = 989e12, 0.6                 # H100 BF16 稠密峰值；大矩阵乘实际能达到的比例
NVLINK, NET = 450e9, 50e9               # 每张卡单方向带宽：节点内 NVLink、节点间网卡（400 Gb/s）
NODE, MEM = 8, 80e9 * 0.9               # 每节点 8 卡；每卡 80 GB，留 10% 给碎片和临时缓冲
MIN_CP_CHUNK = 4096                     # 每个 CP rank 至少分到这么多 token，否则 ring 的每一步太小


@dataclass
class Model:
    N: float     # 参数量（含词表）
    L: int       # 层数
    H: int       # 隐藏维度
    KV: int      # K 或 V 一行的宽度（GQA：KV 头数 × 头维度）
    V: int       # 词表大小


def bandwidth(inner, size):
    """维度由内到外依次是 TP → CP → DP → PP：一个并行组的跨度不超过一个节点就走 NVLink"""
    return NVLINK if inner * size <= NODE else NET


def plan(m, seq, gbs, gpus, tp, cp, pp, mbs=1):
    if tp > NODE or gpus % (tp * cp * pp) or m.L % pp or (cp > 1 and seq // cp < MIN_CP_CHUNK):
        return None
    dp = gpus // (tp * cp * pp)
    if gbs % (dp * mbs):
        return None
    micro = gbs // (dp * mbs)                          # 每条流水线每步要跑的 micro-batch 数
    s, layers = seq // cp, m.L // pp                   # 每张卡上的序列长度和层数

    # ---- 显存：第一个 stage 最紧（参数最多、同时保存 pp 个 micro-batch 的激活）
    psi = m.N / (tp * pp)
    states = 4 * psi + 12 * psi / (dp * cp)            # ZeRO-1：优化器状态在 DP×CP 上切分
    sbh = s * mbs * m.H / tp
    for rc in range(layers + 1):                       # 从 0 开始加全量重计算的层数，直到放得下
        act = (34 * (layers - rc) + 2 * rc) * sbh * min(pp, micro)
        if states + act <= MEM:
            break
    else:
        return None

    # ---- 时间：一个 micro-batch 在最慢的 stage（最后一个，多了 LM head）上要多久
    f_layer = 6 * (m.N - 2 * m.V * m.H) / m.L + 6 * seq * m.H     # 每个 token 每层：线性层 + 因果注意力
    f_head = 6 * m.V * m.H
    sec = s * mbs / tp / (PEAK * EFF)                  # 每 FLOP/token 在这张卡上要多少秒
    avg = (m.L * f_layer + f_head) / pp * sec          # 各 stage 平均分到的计算
    recompute = rc * f_layer / 3 * sec                 # 重算的层多做一次前向（前向是前向 + 反向的 1/3）
    slow = layers * f_layer * sec + recompute + f_head * sec
    act_bytes = s * mbs * m.H * 2
    tp_c = layers * 4 * 2 * (tp - 1) / tp * act_bytes / bandwidth(1, tp)            # 每层前向 2 次、反向 2 次，不重叠
    ring = layers * 3 * (cp - 1) * (2 * s * mbs * m.KV * 2) / bandwidth(tp, cp)      # ring 传 KV（反向再传 dKV）
    cp_c = max(0.0, ring - layers * 6 * seq * m.H * sec)                           # 与注意力计算重叠，只算露出来的部分
    pp_c = 2 * act_bytes / tp / bandwidth(tp * cp * dp, pp) * 0.5 if pp > 1 else 0  # 激活和梯度的点对点，一半藏住
    t_mb = slow + tp_c + cp_c + pp_c
    dpc = dp * cp
    dp_c = 2 * (dpc - 1) / dpc * 2 * psi / bandwidth(tp, dpc) * 0.2                  # 梯度 RS + 参数 AG，80% 与反向重叠
    parts = {"计算": micro * avg, "重算": micro * recompute, "不均衡": micro * (slow - recompute - avg),
             "TP": micro * tp_c, "CP": micro * cp_c, "PP": micro * pp_c, "气泡": (pp - 1) * t_mb, "DP": dp_c}
    step = sum(parts.values())
    model_flops = (m.L * f_layer + f_head) * gbs * seq          # 不含重计算：MFU 只认模型本身要做的运算
    return dict(tp=tp, cp=cp, pp=pp, dp=dp, mfu=model_flops / (step * gpus * PEAK),
                mem=(states + act) / GiB, rc=rc, layers=layers, step=step, parts=parts)


def show(p):
    share = " ".join(f"{k} {v / p['step']:.0%}" for k, v in p["parts"].items() if v / p["step"] >= 0.005)
    rc = f"，重算 {p['rc']}/{p['layers']} 层" if p["rc"] else ""
    print(f"TP={p['tp']} CP={p['cp']} PP={p['pp']} DP={p['dp']}：MFU {p['mfu']:.1%}，"
          f"{p['mem']:.0f} GiB/卡{rc}｜{share}")


def search(title, m, seq, gbs, gpus, top=3, extra=()):
    plans = [p for tp, cp, pp in product((1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16))
             if (p := plan(m, seq, gbs, gpus, tp, cp, pp))]
    plans.sort(key=lambda p: -p["mfu"])
    print(f"== {title}：{len(plans)} 种配置放得下")
    for p in plans[:top]:
        show(p)
    for tp, cp, pp in extra:                           # 再看几种"直觉配置"排在哪里
        rank, p = next((i, q) for i, q in enumerate(plans, 1) if (q["tp"], q["cp"], q["pp"]) == (tp, cp, pp))
        print(f"  第 {rank} 名：", end="")
        show(p)


llama70b = Model(N=70.6e9, L=80, H=8192, KV=1024, V=128256)
llama8b = Model(N=8.03e9, L=32, H=4096, KV=1024, V=128256)

if __name__ == "__main__":
    search("70B，64 卡，8K 序列，每步 4M token", llama70b, 8192, 512, 64, extra=[(8, 1, 1), (8, 1, 4)])
    search("70B，64 卡，128K 序列，每步 4M token", llama70b, 131072, 32, 64, extra=[(8, 1, 1)])
    search("8B，8 卡，8K 序列，每步 0.5M token", llama8b, 8192, 64, 8, extra=[(1, 1, 1)])
```

```text title="输出"
== 70B，64 卡，8K 序列，每步 4M token：24 种配置放得下
TP=4 CP=2 PP=2 DP=4：MFU 53.7%，66 GiB/卡｜计算 89% 不均衡 1% TP 8% 气泡 1%
TP=4 CP=2 PP=4 DP=2：MFU 52.2%，50 GiB/卡｜计算 87% 不均衡 4% TP 8% 气泡 1%
TP=4 CP=2 PP=8 DP=1：MFU 49.6%，42 GiB/卡｜计算 83% 不均衡 8% TP 8% 气泡 1%
  第 5 名：TP=8 CP=1 PP=1 DP=8：MFU 49.3%，66 GiB/卡｜计算 82% TP 18%
  第 9 名：TP=8 CP=1 PP=4 DP=2：MFU 47.1%，42 GiB/卡｜计算 78% 不均衡 3% TP 17% 气泡 1%
== 70B，64 卡，128K 序列，每步 4M token：23 种配置放得下
TP=8 CP=4 PP=2 DP=1：MFU 44.4%，66 GiB/卡，重算 24/40 层｜计算 74% 重算 15% 不均衡 1% TP 8% 气泡 3%
TP=4 CP=8 PP=2 DP=1：MFU 44.2%，66 GiB/卡，重算 32/40 层｜计算 74% 重算 19% TP 3% 气泡 3%
TP=8 CP=4 PP=1 DP=2：MFU 43.8%，66 GiB/卡，重算 64/80 层｜计算 73% 重算 19% TP 8%
  第 6 名：TP=8 CP=1 PP=1 DP=8：MFU 41.8%，65 GiB/卡，重算 80/80 层｜计算 70% 重算 23% TP 7%
== 8B，8 卡，8K 序列，每步 0.5M token：16 种配置放得下
TP=1 CP=2 PP=1 DP=4：MFU 59.9%，58 GiB/卡｜计算 100%
TP=2 CP=1 PP=1 DP=4：MFU 56.9%，43 GiB/卡｜计算 95% TP 5%
TP=2 CP=2 PP=1 DP=2：MFU 56.9%，35 GiB/卡｜计算 95% TP 5%
  第 4 名：TP=1 CP=1 PP=1 DP=8：MFU 55.0%，66 GiB/卡，重算 9/32 层｜计算 92% 重算 8%
```

MFU 的上限是脚本里假设的矩阵乘效率 60%，每一行后面是一步时间的构成。这几行里能读出不少东西。

**TP 越大，通信占比越高。** 70B、8K 的场景里，"TP=8 节点内、DP 节点间"这个教科书配置排在第 5，TP 通信占了 18%；TP=4 时只有 8%。原因是每张卡每层的计算量与 $1/t$ 成正比，而 TP 的通信量是 $2(t-1)/t \cdot sbh$ 的若干倍、几乎不随 $t$ 减少，两者之比随 $t - 1$ 增长：$t$ 从 4 到 8，比例约变成 $7/3$ 倍。这个比例还和 $h$ 成反比（每层计算约 $\propto h^2$，通信 $\propto h$），所以小模型用 TP 更不划算——8B 的场景里 TP=2 就有 5% 的通信。

**CP 在这里替 TP 分担了激活。** 排第一的配置把 TP 降到 4，用 CP=2 把序列切成两半，激活同样减半。GQA 让 K、V 只有隐藏维度的 1/8，ring 要传的数据远少于 TP 的 all-reduce，而且能被注意力计算藏住。脚本限制每个 CP rank 至少 4K token，所以 8K 序列最多只能 CP=2。

**PP 在 batch 足够时几乎没有气泡，但有不均衡。** DP=4 时每条流水线每步有 128 个 micro-batch，1F1B 的气泡 $(p-1)/(m+p-1)$ 只有 1%。真正的代价是最后一个 stage 多了 LM head：128K 词表的输出层约等于 1.1 个 Transformer 层的计算，PP=8 时每个 stage 只有 10 层，最后一个 stage 比平均慢约 10%，其他 stage 都在等它（上表中"不均衡"从 1% 涨到 8%）。所以实际训练会让首尾 stage 少放一层（Llama 3 就是这样做的），Megatron 也有专门的参数（`--decoder-first-pipeline-num-layers` / `--decoder-last-pipeline-num-layers`）。

**长序列靠 CP 省重计算。** 128K 序列下，64 张卡上所有配置都要重计算；不用 CP 时必须全部 80 层都重算（第 6 名）。CP=4 把每张卡的激活降到 1/4，只需重算 24/40 层，MFU 高出 2.6 个百分点。卡更多时，CP 的度数会更大——Llama 3 405B 在 128K 阶段用了 CP=16。

**小模型别急着上模型并行。** 8B 在 8 张卡上，最好的配置和"纯 DP + 重算 9 层"只差 5 个百分点。工程上很多人会选后者（或 FSDP），因为它最简单，换模型、换序列长度都不用重新切分。

### 模型没有算进去的东西

这个脚本只用来**缩小候选范围**，它的假设都很粗：

- 矩阵乘效率是常数。实际上 TP 或 CP 切得越细，每张卡上的矩阵越小，效率越低；
- TP 的通信被当作完全不重叠。Megatron 的 `--tp-comm-overlap` 会把 all-gather / reduce-scatter 与相邻的矩阵乘拆块重叠，能藏住一大部分；
- 忽略了 LM head 的 logits（每个 micro-batch $s \times V$ 个 fp32，8K × 128K 就是 4 GB）、临时缓冲、通信缓冲区和显存碎片；
- 忽略了优化器步骤、数据加载、kernel 启动、通信延迟（小消息时占主导）和掉队的节点。

所以估算完要**上机实测**：用前几名的配置各跑几十步，用 profiler 看时间构成，再在它们之间选。脚本的价值在于排除掉明显不行的组合，并告诉你每一种配置的瓶颈大概在哪。

## 两个公开的真实配置

| | Llama 3 405B（稠密） | DeepSeek-V3 671B（MoE，激活 37B） |
| --- | --- | --- |
| 集群 | 最多 16K 张 H100 | 2048 张 H800 |
| 配置 | 8K 序列：TP8 × PP16 × DP128；128K 序列：TP8 × CP16 × PP16 × DP8 | PP16 × EP64（跨 8 个节点）× ZeRO-1 的 DP，**不用 TP** |
| 维度顺序 | 由内到外 `[TP, CP, PP, DP]`，DP 用 FSDP | EP 复用 DP 的卡 |
| 流水线 | 交错调度，首尾 stage 各少放一层 | DualPipe：双向流水线，把 EP 的 all-to-all 和计算重叠 |
| 效率 | BF16 MFU 38%～43% | 公布的是总训练成本（约 280 万 H800 GPU 小时） |

两者的思路都能用本章的框架解释：

- **Llama 3** 是典型的"TP 占满节点、PP 跨节点切深度、DP 吸收剩下的卡"。卡数到了一万多，全局 batch（16M token）已经不允许 DP 再大，PP 必须到 16；上下文扩到 128K 时，用 CP=16 替换掉 DP 的 16 倍，全局 batch 仍是 16M token；
- **DeepSeek-V3** 每个 token 只激活 37B 参数，注意力部分的计算和激活都不大。它通过重计算 RMSNorm 和 MLA 的上投影、把参数的 EMA 放到 CPU 等手段省下显存，干脆不用 TP，免掉了每层的 all-reduce；代价转移到了 EP 的跨节点 all-to-all 上，于是用 DualPipe 把它和计算重叠起来，并限制每个 token 最多发往 4 个节点。

MoE 模型的"5D"通常是这样折叠的：注意力部分按 TP × CP × DP × PP 排布；进入 MoE 层时，同一批卡重新分组成 专家 TP × EP × 专家 DP × PP，EP 一般取自 DP × CP 那几维（Megatron 称为 MoE parallel folding）。这样 EP 不需要额外的卡，专家的参数也按 EP 切开。

!!! interview "面试怎么答"
    "给你 N 张卡训练某个模型，并行怎么配"是综合题，按步骤答：先放下模型状态（ZeRO → TP → PP / FSDP），再放下激活（SP、CP、部分层重计算），剩下的卡给 DP，最后调流水线的 micro-batch 数和层的划分。摆放上 TP 最内层（节点内 NVLink）、CP 次之、DP 和 PP 在外层，EP 复用 DP 的卡。再讲取舍：TP 的通信占比随度数增大，PP 有气泡和首尾 stage 不均衡（首尾常少放几层），全局 batch 限制了 DP × micro-batch 的总数。最后强调估算只用来缩小范围，要用 profiler 实测排名靠前的几种配置。

## 练习

1. 用本章的公式说明：为什么 TP 通信占一步时间的比例大约与 $(t-1)$ 成正比、与隐藏维度 $h$ 成反比？

??? success "参考答案"
    每层每个 micro-batch，一张卡的计算时间约为 $s \cdot f_{\text{layer}} / (t \cdot P \cdot e)$，其中 $f_{\text{layer}} \approx 12h^2 \cdot c$（线性层，$c$ 是与 MLP 宽度有关的常数）加上注意力项；TP 的通信时间是 $4 \times 2\frac{t-1}{t} \cdot 2sh / B$。两者相除：

    $$\frac{T_{\text{TP}}}{T_{\text{计算}}} = \frac{16 (t-1)\, h\, P e}{B\, f_{\text{layer}}} \propto \frac{t-1}{h}$$

    代入 70B（$h = 8192$、8K 序列时 $f_{\text{layer}} \approx 5.5 \times 10^9$）、$B = 450$ GB/s、$Pe = 593$ TFLOPS，系数约 0.031：$t=4$ 时通信约是计算的 9%，$t=8$ 时约 22%，和脚本里 8% 与 18% 的占比一致（占比的分母是整步时间，所以略小）。

2. 把 70B、8K 的场景扩到 1024 张卡，全局 batch 仍然是每步 4M token（512 条），最佳配置会怎样变？如果全局 batch 放大到 16M token 呢？先预测，再用脚本验证。

??? success "参考答案"
    ```python title="scale_out.py"
    from parallel_plan import llama70b, search

    search("70B，1024 卡，8K 序列，每步 4M token", llama70b, 8192, 512, 1024)
    search("70B，1024 卡，8K 序列，每步 16M token", llama70b, 8192, 2048, 1024)
    ```

    ```text title="输出"
    == 70B，1024 卡，8K 序列，每步 4M token：28 种配置放得下
    TP=8 CP=1 PP=1 DP=128：MFU 47.6%，55 GiB/卡｜计算 79% TP 17% DP 4%
    TP=4 CP=2 PP=2 DP=64：MFU 46.5%，55 GiB/卡｜计算 77% 不均衡 1% TP 7% 气泡 11% DP 3%
    TP=4 CP=2 PP=4 DP=32：MFU 43.7%，38 GiB/卡｜计算 73% 不均衡 3% TP 7% 气泡 16% DP 2%
    == 70B，1024 卡，8K 序列，每步 16M token：28 种配置放得下
    TP=4 CP=2 PP=2 DP=64：MFU 52.0%，55 GiB/卡｜计算 87% 不均衡 1% TP 8% 气泡 3% DP 1%
    TP=4 CP=2 PP=4 DP=32：MFU 50.2%，38 GiB/卡｜计算 84% 不均衡 4% TP 8% 气泡 4%
    TP=8 CP=1 PP=1 DP=128：MFU 48.9%，55 GiB/卡｜计算 82% TP 18% DP 1%
    ```

    卡数多了 16 倍、batch 不变，每条流水线分到的 micro-batch 就少了：64 卡时 DP=4、每条流水线 128 个，1024 卡时 DP=64、只剩 8 个，PP=2 的气泡从 1% 涨到 11%，于是不用 PP 的 TP=8 × DP=128 排到了第一。这就是"全局 batch 限制扩展"的含义：卡越多，每张卡分到的工作越少，气泡和不能重叠的通信占比越高。
    把全局 batch 放大到 16M token，每条流水线又有了 32 个 micro-batch，气泡回到 3%，TP=4 × CP=2 × PP=2 × DP=64 重新排在第一。实际训练中全局 batch 由收敛性决定（Llama 3 在训练过程中把 batch 从 4M 逐步加到 16M token），并行配置要跟着它调整。

3. 如果要让脚本支持 MoE 模型（比如 256 个专家、每个 token 选 8 个），需要改哪些地方？

??? success "参考答案"
    - **参数与显存**：参数分成两部分——注意力和共享部分按 TP × PP 切，专家部分按 EP × 专家 TP × PP 切；优化器状态分别在各自的 DP 组（注意力：DP × CP；专家：专家 DP）上切分；
    - **计算量**：每个 token 只经过 $k$ 个专家，计算按**激活参数量**算，而不是总参数量；
    - **新增 EP 通信**：每个 MoE 层前向两次 all-to-all（dispatch、combine），反向再两次，每次约 $k \cdot sbh$ 字节，跨节点时走网卡；可以按 DualPipe 的思路假设一部分与计算重叠；
    - **负载不均**：最忙的专家所在的卡决定整层时间，可以乘一个不均衡系数（比如 1.1～1.3）；
    - **约束**：EP 的度数要整除专家数，并且取自 DP × CP 的卡；专家的 TP 通常为 1。

## 小结

- [x] 并行维度在卡上由内到外排布：TP 最内（节点内 NVLink），CP 次之，DP 和 PP 在外层；EP 复用 DP 的卡。
- [x] 选配置的顺序：先放下模型状态（ZeRO → TP → PP / FSDP），再放下激活（SP、CP、部分层重计算），剩下的卡给 DP，最后调流水线。
- [x] TP 通信占比随 $(t-1)/h$ 增长；PP 的代价是气泡和首尾 stage 的不均衡；CP 在长序列下用很少的通信换掉大量重计算。
- [x] 全局 batch 限制了 DP × micro-batch 的总数；卡数越多，越需要更大的 batch 或者用更多的 TP / PP 吸收卡数。
- [x] 估算只用来缩小范围，最终要用 profiler 实测前几名的配置。
