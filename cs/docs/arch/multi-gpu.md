# 多卡系统：NVLink、拓扑、切分与功耗

<p class="lead">一张卡放不下的模型要拆到多张卡上，拆开之后每一层都要通信。机内 NVLink 和机间网络的带宽差近一个数量级，这条鸿沟决定了"机内张量并行、机间流水或专家并行"的分工；一个 NVLink 域能连多少张卡，决定了大规模 MoE 能不能不走网络。这一章讲一台 8 卡服务器和一个 NVL72 机柜的内部结构：NVLink 与 NVSwitch、PCIe 与网卡的拓扑、集合通信的时间怎么估、一张卡怎么切给多个任务用（MIG 与 MPS），以及功耗和散热为什么会写进性能账里。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一台 8 卡服务器里，GPU 之间怎么连？为什么跨机的张量并行几乎不可行？
    2. 环形 all-reduce 要几步？decode 阶段的小消息，时间主要花在哪？
    3. `nvidia-smi topo -m` 里的 `NV18`、`PIX`、`NODE`、`SYS` 各是什么意思？为什么要关心网卡和 GPU 的位置关系？
    4. MIG 和 MPS 有什么区别？把一张 H100 切成三份，每份的 decode 速度会变成多少？
    5. 为什么一台机器上 8 张卡同时满载时，单卡的性能比单独跑一张卡时低？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 8 张 SXM 卡通过 NVSwitch 芯片全互联，任意两张卡之间都是完整的 NVLink 带宽（H100 单向 450 GB/s）。跨机只能走网卡：每张卡配一张 400 Gb/s 的网卡也只有约 50 GB/s，比 NVLink 低约 9 倍，而张量并行每层要做两次 all-reduce，通信在关键路径上，所以张量并行基本限制在机内。
    2. 2(p-1) 步：reduce-scatter p-1 步、all-gather p-1 步。decode 时每次只传几百 KB，传输时间很短，8 卡 28 个步骤、每步几微秒的固定延迟加起来就成了主要开销，这就是各引擎自己实现 one-shot / two-shot all-reduce 的原因。
    3. `NV18` 表示两张卡之间有 18 条 NVLink；`PIX` 表示同一个 PCIe 交换芯片下（最近）；`NODE` 表示要经过同一个 NUMA 节点内的 PCIe 根桥；`SYS` 表示要跨 CPU 插槽（最远）。GPUDirect RDMA 要让网卡直接读写显存，网卡和 GPU 挂在同一个 PCIe 交换芯片下（`PIX`）时最快；隔了插槽的话，数据要穿过插槽间的互联，带宽和延迟都变差。
    4. MIG 在硬件上把 SM、L2、显存通道、显存控制器切成互不干扰的几份，每份是一张独立的"小 GPU"，互相隔离、性能可预期；MPS 让多个进程的 kernel 同时跑在整张卡上，共享全部资源，隔离性弱、互相有干扰，但切换灵活、不浪费。H100 切成 3g.40gb（一张卡两份）时，每份约 45% 的 SM 和 3/7 的显存带宽，decode 一步从 4.2 ms 变成 9.8 ms。
    5. 功耗和散热：一张 H100 SXM 的上限是 700 W，8 张卡加 CPU 超过 10 kW，整机的供电和散热按"典型负载"设计，全部满载时 GPU 会降频（`nvidia-smi` 的 `SW Power Cap` 或 `HW Thermal Slowdown`）。此外还有共享资源的争抢：PCIe 通道、主机内存带宽、NVSwitch 的转发能力。

## 一台 8 卡服务器的内部

![图：一台 8 卡 H100 服务器里的数据通路](../assets/figures/server-topology.svg){.aig-svg}

一台典型的 HGX 服务器：

```text
CPU 插槽 0 ──── 内存                      CPU 插槽 1 ──── 内存
   │                                          │
 PCIe 交换芯片 ×2                          PCIe 交换芯片 ×2
   │       │                                  │       │
 GPU0..3  网卡0..3                          GPU4..7  网卡4..7
   └──────────── NVSwitch ×4（GPU 之间全互联）────────────┘
```

两条互不相同的通路：

- **GPU 之间走 NVLink**。H100 每张卡有 18 条 NVLink 4，单向合计 450 GB/s（双向 900 GB/s）。8 张卡不是两两直连（那要 28 条链路），而是每张卡把 18 条链路接到 4 颗 NVSwitch 上，交换芯片负责转发，于是任意两张卡之间都有完整带宽，任意通信模式都不会撞车。这是 all-to-all 这种"人人都给人人发"的通信（MoE 的专家并行）能跑得快的前提。
- **GPU 和外界走 PCIe**。GPU 通过 PCIe 交换芯片连到 CPU，网卡也挂在同一个交换芯片下。这样做是为了 **GPUDirect RDMA**：网卡直接读写显存，数据不进主机内存，只在交换芯片内部走一跳。GPU 和它的"搭档网卡"必须挂在同一个交换芯片下，否则数据要绕到 CPU 甚至跨插槽（见[锁页内存、DMA 与 NUMA](../os/pinned-numa.md)）。

查看拓扑用 `nvidia-smi topo -m`，输出里的记号：

| 记号 | 含义 | 快慢 |
| --- | --- | --- |
| `NV#` | 两者之间有 # 条 NVLink | 最快 |
| `PIX` | 同一个 PCIe 交换芯片下 | 快，GPUDirect RDMA 的理想位置 |
| `PXB` | 经过多个 PCIe 交换芯片 | 中 |
| `PHB` / `NODE` | 经过 PCIe 根桥 / 同一个 NUMA 节点内 | 慢 |
| `SYS` | 跨 CPU 插槽（经过插槽间互联） | 最慢 |

NCCL 启动时会自己探测这张表，决定用哪些链路、组成什么样的环或树；`NCCL_TOPO_DUMP_FILE` 可以把它探测到的拓扑导出来。排查通信性能问题，第一步就是看这张表和 `NCCL_DEBUG=INFO` 的日志。

## 从 8 卡到 72 卡

NVLink 的范围一代比一代大：

| 代 | 每卡单向带宽 | 一个 NVLink 域的规模 | 典型形态 |
| --- | --- | --- | --- |
| NVLink 3（A100） | 300 GB/s | 8 张卡 | 机内 NVSwitch |
| NVLink 4（H100） | 450 GB/s | 8 张卡（少数 256 卡方案） | 机内 NVSwitch |
| NVLink 5（B200） | 900 GB/s | 72 张卡（GB200/GB300 NVL72） | 整个机柜一个域 |

NVL72 把 18 个计算托盘（每个 2 颗 Grace CPU + 4 张 Blackwell GPU）和 9 个交换托盘放进一个机柜，用铜背板连起来，72 张 GPU 在一个 NVLink 域里，任意两张之间都是 NVLink 带宽。对推理来说，这直接改变了部署方式：几百个专家的大 MoE 用专家并行铺开时，token 的 all-to-all 分发原来要走机间网络，现在全部走 NVLink，快一个数量级；超长上下文的 KV 也可以在域内的卡之间快速搬运。代价是功耗密度（一个机柜 120 kW 以上）必须液冷。

## 集合通信的时间怎么估

用 α-β 模型比较几种 all-reduce 的走法（推理系统手册里的同一个工具）：

<div class="aig-widget" data-widget="collective-cost"></div>

估一次集合通信，用"延迟 + 传输"两项：

```python title="collective.py"
# 集合通信的时间 = 每步的延迟 × 步数 + 传输量 ÷ 带宽。对比机内 NVLink 和机间 400 Gb/s 网络
LINKS = {  # 名称: (单向带宽 GB/s, 单步延迟 us)
    "NVLink 4": (450, 2),
    "NVLink 5": (900, 2),
    "IB NDR 400G": (50, 8),
}


def ring_allreduce(nbytes, n, bw, lat):
    """环形 all-reduce：2(n-1) 步，每步每张卡发出 nbytes/n 字节"""
    steps = 2 * (n - 1)
    return steps * lat * 1e-6 + steps * (nbytes / n) / (bw * 1e9)


print("8 卡 all-reduce 的耗时（ms）")
print(f"{'消息大小':28s}" + "".join(f"{k:>14s}" for k in LINKS))
for name, size in [("decode 一层的激活 256 KB", 256 << 10), ("prefill 一层的激活 64 MB", 64 << 20),
                   ("梯度同步 1 GB", 1 << 30)]:
    print(f"{name:26s}" + "".join(f"{ring_allreduce(size, 8, bw, lat) * 1e3:14.3f}"
                                  for bw, lat in LINKS.values()))
print()
print("同一条 256 KB 的消息，卡数从 8 张涨到 72 张（NVLink 5）：")
for n in (8, 16, 32, 72):
    t = ring_allreduce(256 << 10, n, 900, 2) * 1e3
    print(f"  {n:2d} 卡：{t:6.3f} ms，其中固定延迟 {2 * (n - 1) * 2 / 1e3:5.3f} ms（占 {2 * (n - 1) * 2 / 1e3 / t:.0%}）")
```

```text title="输出"
8 卡 all-reduce 的耗时（ms）
消息大小                              NVLink 4      NVLink 5   IB NDR 400G
decode 一层的激活 256 KB                0.029         0.029         0.121
prefill 一层的激活 64 MB                0.289         0.158         2.461
梯度同步 1 GB                          4.204         2.116        37.693

同一条 256 KB 的消息，卡数从 8 张涨到 72 张（NVLink 5）：
   8 卡： 0.029 ms，其中固定延迟 0.028 ms（占 98%）
  16 卡： 0.061 ms，其中固定延迟 0.060 ms（占 99%）
  32 卡： 0.125 ms，其中固定延迟 0.124 ms（占 100%）
  72 卡： 0.285 ms，其中固定延迟 0.284 ms（占 100%）
```

三个结论：

- **decode 的小消息由延迟主导**。张量并行下每层要做两次 all-reduce，每次几百 KB，机内只要 0.03 ms，但一个 80 层的模型一步就有 160 次，加起来 4～5 ms，和计算时间同一个量级。这就是各引擎都自己实现小消息 all-reduce 的原因：用 one-shot（每张卡直接读别人的缓冲区求和）或 two-shot（先 reduce-scatter 再 all-gather）算法，靠 NVLink 的点对点直接读写把 28 步压成一两步，延迟降到几微秒，并且和后面的 RMSNorm 融合（见[张量并行](serving://distributed/tensor-parallel/#通信的代价)）。
- **prefill 的大消息由带宽主导**，机内 0.16～0.29 ms，机间 2.5 ms，差了近 10 倍。这就是张量并行不跨机的根本原因。
- **卡数变多时，环形算法的固定延迟线性增长**。72 卡的环走 142 步，小消息几乎全是延迟。所以大规模通信不用环：NVSwitch 支持在交换芯片里直接做归约（NVLS），一步完成；NCCL 也会在小消息时改用树形算法。

顺便记住两个概念：`nccl-tests` 报告的 **algbw**（数据量 ÷ 耗时）和 **busbw**（换算成每条链路实际承载的带宽，all-reduce 时 busbw = algbw × 2(p-1)/p）。和硬件峰值比较要用 busbw。

## 一张卡切给多个任务

有时模型很小，一张卡放得下好几个实例。两种做法：

**MIG（多实例 GPU）**，Ampere 起支持。在硬件层面把 GPU 切成最多 7 份：SM、L2 的通路、显存控制器和显存都按份分开，每一份是一张独立的"小 GPU"，有自己的设备号，互相之间完全隔离，一份出故障不影响别的。代价是切分粒度固定（只能按预设的档位切）、切换要重新配置，而且**每一份的带宽也只有相应的份额**：

```python title="sharing.py"
# 一张卡上跑多个模型实例：MIG（硬件切分）和 MPS（共享）各自的账
# MIG 把 SM、L2、显存控制器都按份切开；MPS 让多个进程的 kernel 同时上一张完整的卡
SM, L2, MEM, BW = 132, 50, 80, 3350        # H100：SM 数、L2（MB）、显存（GB）、带宽（GB/s）
print("MIG 切分（H100 的 7 种档位里挑 3 种）：")
for name, slices, sms, mem in [("1g.10gb", 1, 16, 10), ("2g.20gb", 2, 32, 20), ("3g.40gb", 3, 60, 40)]:
    print(f"  {name:8s} {sms:3d} 个 SM（{sms / SM:4.0%}） 显存 {mem} GB  带宽约 {BW * slices / 7 / 1000:.2f} TB/s  "
          f"一张卡能切 {7 // slices} 个")
print()
print("同一个 7B 模型（BF16 权重 14 GB，decode 读一遍权重）：")
for name, sms, mem, share in [("整卡", SM, MEM, 1.0), ("3g.40gb", 60, 40, 3 / 7), ("2g.20gb", 32, 20, 2 / 7)]:
    bw = BW * share
    rest = f"还剩 {mem - 14:.0f} GB 放 KV" if mem > 14 else "放不下权重"
    print(f"  {name:8s} 带宽 {bw / 1000:.2f} TB/s  decode 一步下限 {14 / bw * 1000:5.1f} ms  {rest}")
```

```text title="输出"
MIG 切分（H100 的 7 种档位里挑 3 种）：
  1g.10gb   16 个 SM（ 12%） 显存 10 GB  带宽约 0.48 TB/s  一张卡能切 7 个
  2g.20gb   32 个 SM（ 24%） 显存 20 GB  带宽约 0.96 TB/s  一张卡能切 3 个
  3g.40gb   60 个 SM（ 45%） 显存 40 GB  带宽约 1.44 TB/s  一张卡能切 2 个

同一个 7B 模型（BF16 权重 14 GB，decode 读一遍权重）：
  整卡       带宽 3.35 TB/s  decode 一步下限   4.2 ms  还剩 66 GB 放 KV
  3g.40gb  带宽 1.44 TB/s  decode 一步下限   9.8 ms  还剩 26 GB 放 KV
  2g.20gb  带宽 0.96 TB/s  decode 一步下限  14.6 ms  还剩 6 GB 放 KV
```

decode 是带宽受限的，切成 3g.40gb 之后带宽只剩 43%，一步从 4.2 ms 变成 9.8 ms。所以 MIG 适合"很多个互不相干的小任务、要求互相隔离"（比如多租户的小模型服务、CI），不适合追求单请求延迟的大模型推理。

**MPS（多进程服务）**是另一条路：多个进程的 kernel 提交到同一个硬件调度器，可以真正同时占用 SM（没有 MPS 时，多个进程的 kernel 是时间片轮转的）。它不切分资源，也就没有隔离：一个进程可以占满显存或把 SM 全用掉，一个进程崩溃可能影响其他进程。好处是灵活、不浪费，可以用 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 粗略限制每个进程能用的 SM 比例。

推理引擎内部还有第三条路：**在一个进程里用不同的 stream 和 [green context](serving://frontier/pd-multiplex/) 划分 SM**，让 prefill 和 decode 在同一张卡上并行，互相填补空隙。这比 MIG 灵活，比 MPS 可控。

## 功耗、散热与频率

性能的账最后要落到电和热上：

- **单卡功耗上限**：A100 SXM 400 W、H100 SXM 700 W、B200 约 1000 W、GB200 一个超级芯片 2700 W。`nvidia-smi -q -d POWER` 能看到当前功耗和上限，`nvidia-smi --query-gpu=clocks_throttle_reasons.active` 能看到降频原因。
- **满载时会降频**。Tensor Core 满负荷的矩阵乘功耗最大，频率会从加速频率掉下来，这也是实测算力通常只有规格峰值七八成的原因之一（上一章用规格反推频率时已经见过：H100 的 FP32 峰值按 1.98 GHz 算，Tensor Core 峰值只对应 1.83 GHz）。
- **8 张卡不是 1 张卡的 8 倍**。整机供电、散热、NVSwitch 的转发能力、PCIe 通道、主机内存带宽都是共享的。压测时要压整机，不能拿单卡数据乘以 8。
- **液冷不只是省电**。风冷的机柜放不下 NVL72 这样的功耗密度；液冷还能让 GPU 稳定在更高的频率上，实际吞吐反而更高。

做容量规划时，这些会直接变成成本：机房的每机柜供电上限往往比"塞满卡"要低，能装多少卡是被电和散热决定的，而不是被机架空间决定的。

!!! interview "面试怎么答"
    被问多卡系统：一台 8 卡机里 GPU 之间通过 NVSwitch 全互联（H100 单向 450 GB/s），GPU 和网卡挂在同一个 PCIe 交换芯片下以便 GPUDirect RDMA；跨机只有约 50 GB/s，差约 9 倍，所以张量并行限于机内、跨机用流水或专家并行。估通信用"延迟 × 步数 + 数据量 ÷ 带宽"：环形 all-reduce 是 2(p-1) 步，decode 的小消息由延迟主导（一步 160 次 all-reduce 能到 4～5 ms），所以有 one-shot/two-shot 的自定义 all-reduce 和 NVSwitch 内归约；prefill 的大消息由带宽主导。NVLink 域从 8 卡扩到 NVL72 的 72 卡，大规模专家并行的 all-to-all 因此不用走网络。切卡：MIG 是硬件隔离、连带宽一起切（H100 切三份，decode 从 4.2 ms 变 9.8 ms），MPS 是共享不隔离，引擎内部还可以用 green context 让 prefill 和 decode 并行。最后提功耗：单卡 700 W 到 1000 W，满载降频，整机不是单卡的 8 倍，容量规划受限于供电和散热。

!!! info "相关章节"
    - [多 GPU 与 NCCL](cuda://tools/multi-gpu/)（CUDA：用 NCCL 写多卡程序）
    - [GPU 互联与网络](serving://comm/interconnect/)、[集合通信：NCCL 的算法与协议](serving://comm/nccl/)（推理系统）
    - [集合通信原语](train://basics/collectives/)（分布式训练）

## 练习

**1. 张量并行的通信账。** 一个 70B 模型（80 层、hidden 8192）用 TP=8 部署，BF16。decode 时 batch 为 64，每层两次 all-reduce。机内 NVLink 4（单向 450 GB/s、每步 2 us）用环形算法，一步 decode 的通信要多久？如果换成延迟 5 us 的 one-shot 算法呢？

??? success "参考答案"
    每次 all-reduce 的数据量 = 64 × 8192 × 2 字节 = 1 MB。环形：28 步 × 2 us = 56 us 的延迟，加上 28 × (1 MB / 8) / 450 GB/s ≈ 8 us 的传输，约 64 us；每层两次、80 层共 160 次，约 10 ms。one-shot 每次约 5 us + 传输，160 次约 1 ms 多。

    decode 一步的计算本身只有几毫秒，10 ms 的通信完全不可接受，1 ms 才能接受——这就是自定义 all-reduce 必不可少的原因。

**2. 拓扑排错。** 一个跨机的 EP 部署，实测 all-to-all 带宽只有预期的一半。`nvidia-smi topo -m` 显示 GPU4 和它使用的网卡之间是 `SYS`。可能是什么问题？怎么改？

??? success "参考答案"
    `SYS` 表示 GPU 和网卡分属两个 CPU 插槽，数据要经过插槽之间的互联，带宽受限、延迟更高，GPUDirect RDMA 的收益也大打折扣。改法：让每张 GPU 使用与它挂在同一个 PCIe 交换芯片下的网卡（NCCL 的 `NCCL_IB_HCA` 指定网卡、或按 rank 设置亲和性），同时用 `numactl` 把进程绑到对应的 NUMA 节点上。如果机器本身的网卡数量不够，就只能接受，或者调整 rank 到 GPU 的映射，让通信量大的 rank 用位置好的网卡。

**3. 该不该切卡。** 一个 1.5B 的小模型服务，延迟要求是每个 token 30 ms 以内，QPS 不高但要求多租户互相隔离。手上是 H100。用 MIG 切成 7 份（每份 16 个 SM、10 GB、带宽约 0.48 TB/s）可行吗？算一算。

??? success "参考答案"
    1.5B 的 BF16 权重 3 GB，10 GB 显存放得下，还剩 7 GB 放 KV。decode 一步至少要读一遍权重：3 GB / 0.48 TB/s ≈ 6.3 ms，远低于 30 ms 的要求，可行（再加上注意力和内核启动开销，实测大约十几毫秒）。这正是 MIG 适合的场景：小模型、隔离要求高、单请求延迟有余量。反过来，如果模型是 70B，MIG 的任何一份都放不下权重，只能用整卡甚至多卡。

**4. 为什么不是 8 倍。** 单卡跑一个模型实例测得 1000 token/s。在同一台 8 卡机上起 8 个独立实例（每卡一个，不做并行），总吞吐大概率达不到 8000。列出至少三个原因。

??? success "参考答案"
    （1）功耗与散热：8 张卡同时满载会触到整机的功耗上限和散热上限，GPU 降频；（2）主机侧共享资源：8 个进程的调度、分词、HTTP 处理共用 CPU 和内存带宽，CPU 可能先成为瓶颈（小模型尤其明显，因为 GPU 一步只要几毫秒）；（3）PCIe 和主机内存：权重加载、KV 卸载、日志与监控都走 PCIe，多个实例互相争抢；（4）NUMA：8 个进程未必都绑在自己 GPU 所在的插槽上，跨插槽访问内存更慢；（5）如果实例之间还共享一个前端或路由，那里也可能先饱和。定位方法是先看 `nvidia-smi` 的利用率和降频原因，再看 CPU 利用率和 `numastat`。

## 小结

- [x] 8 卡机内 GPU 通过 NVSwitch 全互联（H100 单向 450 GB/s），GPU 与网卡同挂一个 PCIe 交换芯片以便 GPUDirect RDMA；跨机约 50 GB/s，差约 9 倍。
- [x] `nvidia-smi topo -m`：`NV#` > `PIX` > `PXB` > `NODE` > `SYS`；NCCL 按这张表组环，排错先看它。
- [x] 通信时间 = 延迟 × 步数 + 数据量 ÷ 带宽；环形 all-reduce 2(p-1) 步，decode 的小消息由延迟主导，要用 one-shot/two-shot 或 NVSwitch 内归约。
- [x] NVLink 域从 8 卡扩到 NVL72 的 72 卡，大规模 EP 的 all-to-all 不再走网络，代价是功耗密度必须液冷。
- [x] MIG 是硬件切分（连带宽一起切，H100 切三份 decode 慢一倍多），MPS 是共享不隔离，引擎内部还能用 green context 让 prefill 与 decode 并行。
- [x] 单卡 700～1000 W，满载降频，整机吞吐不是单卡的 8 倍；容量规划往往受限于供电和散热。
