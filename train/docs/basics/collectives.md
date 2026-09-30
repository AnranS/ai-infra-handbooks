# 集合通信原语

<p class="lead">所有并行最终都落到少数几个通信原语上：数据并行是 all-reduce，ZeRO 是 reduce-scatter 和 all-gather，专家并行是 all-to-all，流水线是点对点的 send / recv。这一章在 4 个 CPU 进程上把每个原语跑一遍，看清它们的语义；再用点对点通信手写一个环形 all-reduce，算清它的通信量——这是估算所有并行代价的基础。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. all-reduce、reduce-scatter、all-gather、all-to-all 分别做什么？
    2. 为什么说 all-reduce = reduce-scatter + all-gather？
    3. 环形 all-reduce 中每张卡发送多少数据？和卡数有什么关系？
    4. 什么是"总线带宽"（bus bandwidth）？为什么 nccl-tests 报告它而不是"算法带宽"？
    5. 小消息的 all-reduce 为什么慢？怎么优化？

??? success "自测参考答案（先自己答，再展开对照）"
    1. all-reduce：所有人的数据求和，每个人都拿到总和；reduce-scatter：求和之后每个人只拿到其中一段；all-gather：每个人贡献一段，最后每个人都拿到拼好的完整数据；all-to-all：每个人给其他每个人各发一份不同的数据（像矩阵转置）。
    2. 先 reduce-scatter，每个人得到总和的一段；再 all-gather，把各段拼起来发给所有人，结果正好是总和的完整拷贝。环形算法就是这样实现 all-reduce 的，ZeRO 也把这两步拆开来用。
    3. 每张卡发送 $2(n-1)/n \times S$ 字节，卡数多时趋近 $2S$，几乎与卡数无关（"带宽最优"），但步数 $2(n-1)$ 随卡数线性增长。
    4. 总线带宽是按算法在链路上实际传输的数据量换算出来的带宽（all-reduce 在算法带宽上乘以 $2(n-1)/n$），它和卡数无关，可以直接与链路的峰值带宽比较，看是否跑满了硬件。
    5. 小消息时每一步传输的时间很短，固定的延迟（启动、同步、跨卡往返）占了大头，环形算法的 $2(n-1)$ 步都要付这个延迟。优化：用步数少的算法（tree、单步直接读对端的 one-shot）、把小消息合并成大消息、用能被 CUDA Graph 录制的定制实现。

## 五个原语

用 `torchrun` 启动 4 个进程，每个进程（rank）持有一个 4 元素的张量，看每个原语之后每个 rank 手里有什么：

```python title="collectives.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def show(title, t):
    """把每个 rank 上的结果收集到 rank 0 打印"""
    parts = [torch.empty_like(t) for _ in range(world)] if rank == 0 else None
    dist.gather(t, parts, dst=0)
    if rank == 0:
        print(title)
        for r, p in enumerate(parts):
            print(f"  rank {r}: {p.tolist()}")


x = torch.arange(4, dtype=torch.float32) + 10 * rank     # rank r 持有 [10r, 10r+1, 10r+2, 10r+3]
show("输入", x)

y = x.clone()
dist.all_reduce(y)                                        # 所有 rank 得到逐元素之和
show("all_reduce（求和）", y)

chunk = torch.empty(1)
dist.reduce_scatter_tensor(chunk, x)                      # 求和之后切成 world 份，rank r 拿第 r 份
show("reduce_scatter", chunk)

gathered = torch.empty(world)
dist.all_gather_into_tensor(gathered, chunk)              # 把每个 rank 的一份拼起来，所有 rank 都拿到全部
show("all_gather（接在 reduce_scatter 后面）", gathered)

out = torch.empty(4)
dist.all_to_all_single(out, x)                            # rank r 的第 p 个元素发给 rank p
show("all_to_all", out)

ok = torch.equal(gathered, y)
flag = torch.tensor([int(ok)])
dist.all_reduce(flag, op=dist.ReduceOp.MIN)
if rank == 0:
    print("reduce_scatter + all_gather == all_reduce：", bool(flag.item()))
dist.destroy_process_group()
```

```text title="输出"
输入
  rank 0: [0.0, 1.0, 2.0, 3.0]
  rank 1: [10.0, 11.0, 12.0, 13.0]
  rank 2: [20.0, 21.0, 22.0, 23.0]
  rank 3: [30.0, 31.0, 32.0, 33.0]
all_reduce（求和）
  rank 0: [60.0, 64.0, 68.0, 72.0]
  rank 1: [60.0, 64.0, 68.0, 72.0]
  rank 2: [60.0, 64.0, 68.0, 72.0]
  rank 3: [60.0, 64.0, 68.0, 72.0]
reduce_scatter
  rank 0: [60.0]
  rank 1: [64.0]
  rank 2: [68.0]
  rank 3: [72.0]
all_gather（接在 reduce_scatter 后面）
  rank 0: [60.0, 64.0, 68.0, 72.0]
  rank 1: [60.0, 64.0, 68.0, 72.0]
  rank 2: [60.0, 64.0, 68.0, 72.0]
  rank 3: [60.0, 64.0, 68.0, 72.0]
all_to_all
  rank 0: [0.0, 10.0, 20.0, 30.0]
  rank 1: [1.0, 11.0, 21.0, 31.0]
  rank 2: [2.0, 12.0, 22.0, 32.0]
  rank 3: [3.0, 13.0, 23.0, 33.0]
reduce_scatter + all_gather == all_reduce： True
```

| 原语 | 语义 | 在哪里用 |
| --- | --- | --- |
| all-reduce | 所有 rank 的张量逐元素求和（或最大值等），每个 rank 都得到结果 | DDP 的梯度同步、TP 的行切分之后 |
| reduce-scatter | 求和后切成 $n$ 份，每个 rank 只拿自己那一份 | ZeRO 的梯度同步、SP 里代替 all-reduce 的前一半 |
| all-gather | 每个 rank 贡献一份，拼起来之后每个 rank 都得到全部 | ZeRO-3 取回参数、SP 里代替 all-reduce 的后一半 |
| all-to-all | rank $r$ 的第 $p$ 块发给 rank $p$，相当于一次"转置" | MoE 的 token 分发、Ulysses 在序列和头之间切换 |
| send / recv | 点对点 | 流水线并行相邻 stage 之间 |

all-to-all 的输出正好是输入按 rank 的"转置"：rank $p$ 收到每个 rank 的第 $p$ 个元素。

## 环形 all-reduce 与通信量

NCCL 在大消息上默认用**环形算法**：所有 rank 连成一个环，数据切成 $n$ 块。

![图：环形 all-reduce](../assets/figures/ring-allreduce.svg){.aig-svg}

1. **reduce-scatter 阶段**（$n-1$ 步）：每一步每个 rank 把一块发给右边、从左边收一块并累加。$n-1$ 步之后，每个 rank 手里有一块已经累加完所有 rank 的结果；
2. **all-gather 阶段**（$n-1$ 步）：把累加好的块沿着环再传一圈，每个 rank 都拿到全部。

按"播放"看这 2(n−1) 步里每张卡手上的块是怎么一点点变完整的：

<div class="aig-widget" data-widget="ringreduce"></div>

用点对点的 `isend` / `irecv` 实现一遍，并统计每个 rank 发送的字节数：

```python title="ring_allreduce.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, n = dist.get_rank(), dist.get_world_size()
right, left = (rank + 1) % n, (rank - 1) % n


def ring_all_reduce(t):
    """环形 all-reduce：先 n-1 步 reduce-scatter，再 n-1 步 all-gather。返回本 rank 发送的字节数。"""
    chunks = list(t.chunk(n))                      # 视图：直接在 t 上原地累加
    sent = 0
    for step in range(n - 1):                      # reduce-scatter：第 step 步把块 (rank - step) 发给右边
        send_idx, recv_idx = (rank - step) % n, (rank - step - 1) % n
        buf = torch.empty_like(chunks[recv_idx])
        reqs = [dist.isend(chunks[send_idx].contiguous(), right), dist.irecv(buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx] += buf
        sent += chunks[send_idx].numel() * t.element_size()
    for step in range(n - 1):                      # all-gather：把已经求好和的块沿环传一圈
        send_idx, recv_idx = (rank + 1 - step) % n, (rank - step) % n
        buf = torch.empty_like(chunks[recv_idx])
        reqs = [dist.isend(chunks[send_idx].contiguous(), right), dist.irecv(buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx].copy_(buf)
        sent += chunks[send_idx].numel() * t.element_size()
    return sent


torch.manual_seed(rank)
x = torch.randn(1024)
ref = x.clone()
dist.all_reduce(ref)
sent = ring_all_reduce(x)
ok = torch.tensor([int(torch.allclose(x, ref, atol=1e-5))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
sizes = [torch.zeros(1, dtype=torch.long) for _ in range(n)]
dist.all_gather(sizes, torch.tensor([sent]))
if rank == 0:
    total = x.numel() * x.element_size()
    print(f"{n} 个 rank，每个 rank 的数据 {total} 字节")
    print("与 dist.all_reduce 结果一致：", bool(ok.item()))
    print("每个 rank 发送的字节：", [int(s.item()) for s in sizes], f"= 2(n-1)/n × {total} = {2 * (n - 1) * total // n}")
dist.destroy_process_group()
```

```text title="输出"
4 个 rank，每个 rank 的数据 4096 字节
与 dist.all_reduce 结果一致： True
每个 rank 发送的字节： [6144, 6144, 6144, 6144] = 2(n-1)/n × 4096 = 6144
```

每个 rank 发送（也接收）$2(n-1)/n \cdot S$ 字节，$S$ 是张量的大小。卡数变大时它趋近于 $2S$，**几乎与卡数无关**——这是环形算法的优点，也是数据并行能扩展到上千张卡的原因。
reduce-scatter 和 all-gather 各占一半：$(n-1)/n \cdot S$。all-to-all 每个 rank 发出去 $(n-1)/n \cdot S$（自己那一块不用发）。

### α-β 模型

一次通信的时间可以粗略地写成

$$T = \alpha \cdot \text{步数} + \frac{\text{发送的字节数}}{\beta}$$

$\alpha$ 是每一步的固定延迟（几微秒），$\beta$ 是链路带宽。环形 all-reduce 有 $2(n-1)$ 步：大消息时带宽项占主导，接近最优；**小消息时延迟项占主导**，而且随卡数线性增长。所以：

- NCCL 对小消息改用**树形算法**（步数是 $\log n$ 量级），或者在 NVSwitch 上用 NVLS（交换机内的归约）；
- 推理的 decode 阶段每次 all-reduce 只有几十 KB，vLLM、SGLang 都实现了自己的 one-shot / two-shot all-reduce：每个 rank 直接读所有其他 rank 的缓冲区（NVLink 上互相可见的显存）一步完成（见 C++ 手册里读 vLLM custom all-reduce 的那一节，[读懂推理基础库的 C++](cpp://engineering/reading-code/)）；
- 训练里 DDP 把很多小梯度合并成大的"桶"再做 all-reduce（下一章）。

### 算法带宽与总线带宽

nccl-tests 报告两个带宽：**算法带宽** = $S / T$，**总线带宽** = 算法带宽 × $2(n-1)/n$（对 all-reduce）。总线带宽把算法本身的数据放大系数除掉了，可以直接和硬件的链路带宽比较：8 张 H100 用 NVLink 做 all-reduce，总线带宽能到 450 GB/s 左右（单方向的 NVLink 带宽），说明链路基本打满。
练习题网站的 `python practice/judge.py bench` 在多卡机器上会测出这个数。

## 硬件拓扑

- **节点内**：NVLink + NVSwitch，每张 H100 单方向 450 GB/s，任意两张卡之间都是全带宽；
- **节点间**：每张卡配一张 400 Gb/s（50 GB/s）的 InfiniBand 或 RoCE 网卡，GPUDirect RDMA 让网卡直接读写显存；
- NCCL 会自动探测拓扑，节点内走 NVLink、节点间走网卡，并用"同一台机器里号相同的卡走同一条网络轨道"（rail-optimized）的方式组网。

节点内和节点间带宽差了约 10 倍，这决定了几乎所有并行组合的布局：通信密集的放节点内，其余的放节点间。

!!! interview "面试怎么答"
    集合通信是所有并行题的基础：all-reduce、reduce-scatter、all-gather、all-to-all 各做什么，all-reduce = reduce-scatter + all-gather（ZeRO 正是把它拆开用）。环形 all-reduce 每张卡发送 $2(n-1)/n \cdot S$ 字节，几乎与卡数无关，所以说它"带宽最优"；但步数是 $2(n-1)$，小消息被固定延迟主导，NCCL 会改用 tree 或分层算法。看测试报告用总线带宽（busbw），因为它能直接和链路带宽比；节点内 NVLink 比节点间网络快约 10 倍，这决定了各种并行放在哪一层。

## 练习

1. 8 张卡做 all-reduce，张量大小 1 GB，NVLink 单方向 450 GB/s，每步延迟 5 微秒。估算环形算法的时间。换成 1 MB 的张量呢？

??? success "参考答案"
    1 GB：带宽项 $2 \times 7/8 \times 1\text{ GB} / 450\text{ GB/s} = 3.9$ ms，延迟项 $14 \times 5\,\mu s = 0.07$ ms，合计约 3.96 ms，带宽主导。
    1 MB：带宽项 3.9 微秒，延迟项 70 微秒——延迟是带宽的近 20 倍。小消息要用步数少的算法（树、one-shot），或者合并成大消息。

2. 用 `all_to_all_single` 实现一次"矩阵转置"：4 个 rank 各持有一个 $4 \times 2$ 矩阵的一行块（第 $r$ 个 rank 持有第 $r$ 行），操作之后第 $r$ 个 rank 持有整个矩阵……想一想，all-to-all 在 Ulysses 上下文并行里起的作用是不是也是一种"转置"？

??? success "参考答案"
    是的。Ulysses 里每个 rank 起初持有**一段序列的全部注意力头**，all-to-all 之后变成持有**全部序列的一部分头**——在"序列维"和"头维"之间做了一次分布式转置，这样每个 rank 就能独立地对自己那几个头做完整序列的注意力；算完再用一次 all-to-all 转回来（见[上下文并行](../model/context.md)）。

## 小结

- [x] all-reduce、reduce-scatter、all-gather、all-to-all、send / recv 是所有并行的基础；all-reduce = reduce-scatter + all-gather。
- [x] 环形 all-reduce 每个 rank 发送 $2(n-1)/n \cdot S$ 字节，几乎与卡数无关；但步数随卡数增长，小消息被延迟主导。
- [x] 总线带宽可以直接和链路带宽比较；节点内 NVLink 比节点间网络快约 10 倍，决定了各种并行的摆放位置。
