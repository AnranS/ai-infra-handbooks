# 集合通信：NCCL 的算法、协议与推理框架的定制 all-reduce

<p class="lead">CUDA 手册的<a href="cuda://tools/multi-gpu/">多 GPU 与 NCCL</a>一章讲了 NCCL 的用法和 ring all-reduce。这一章往下走一层：NCCL 内部有哪几种算法和协议、它怎么选、各自在什么消息大小上占优；为什么 vLLM 和 SGLang 还要自己写一套 all-reduce；以及怎样读 nccl-tests 的数字、排查通信卡死。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. NCCL 的 ring 和 tree 算法各适合什么场景？NVLS 又是什么？
    2. LL、LL128、Simple 三种协议的区别是什么？
    3. nccl-tests 输出的 algbw 和 busbw 是什么关系？
    4. 推理框架的定制 all-reduce 为什么在小消息上比 NCCL 快？one-shot 和 two-shot 有什么区别？
    5. 多卡推理服务卡住不动，你怎么判断是不是通信的问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. ring：带宽最优，每卡收发约 2S 字节，但步数 $2(n-1)$，适合大消息、机内；tree：步数约 $\log n$，延迟低，适合多机的小消息；NVLS：利用 NVSwitch 在交换机里做归约，通信量减半，还不占 SM。
    2. LL（low latency）：每 8 字节里带 4 字节的标志，接收方轮询标志就知道数据到了，延迟最低，带宽效率只有一半；LL128：每 128 字节带 8 字节标志，延迟和带宽折中；Simple：大块传输 + 显式同步，带宽最高、延迟最大。
    3. busbw = algbw × 修正系数，all-reduce 的系数是 $2(n-1)/n$。busbw 反映链路上实际的传输速率，可以直接和链路的峰值带宽比较。
    4. 小消息时 NCCL 的 ring 要走 $2(n-1)$ 步，每步都有同步和启动开销；定制实现用 CUDA IPC 直接读写对端的显存，步数和同步次数少得多。one-shot：每张卡直接读所有对端的数据，一步归约完（适合小消息）；two-shot：先 reduce-scatter 各自归约一段、再 all-gather，两步完成，传输量更小（适合中等消息）。
    5. 先看各 rank 卡在哪个集合通信上、调用顺序是否一致（有没有某个 rank 少调了一次或参数不同）、有没有 rank 已经崩溃；再查网卡和网络配置。工具：`NCCL_DEBUG=INFO` 看初始化和拓扑、PyTorch 的 flight recorder 记录每个 rank 最近的集合通信、py-spy 看卡住的调用栈。

## 算法：一次 all-reduce 有几种走法

NCCL 为每次集合通信选择一种**算法**（数据在卡之间怎么流动）和一种**协议**（每一步怎么传、怎么通知对方数据到了）。all-reduce 的主要算法：

- **ring**：reduce-scatter + all-gather 各 $n-1$ 步，每卡收发 $2\frac{n-1}{n}S$，带宽最优，但步数随卡数线性增长；
- **tree**（双二叉树）：先沿树归约、再沿树广播，步数是 $O(\log n)$，大消息切块流水后带宽也接近最优。它主要用在**多机**：机内仍然走链，机间走树；
- **NVLS**（NVLink SHARP）：H100 的 NVSwitch 能在交换机里做加法。每张卡把数据发给交换机一次，再从交换机取回求和结果一次，通信量从 $2\frac{n-1}{n}S$ 降到约 $S$，还几乎不占 SM；
- **CollNet**：类似地利用 InfiniBand 交换机的 SHARP 做机间归约。

推理框架还会绕开 NCCL，直接用 CUDA IPC 映射其他卡的显存，在一个 kernel 里完成通信：

- **one-shot**：每张卡直接读其他 $n-1$ 张卡的**完整**数据、在本地求和。只需一次同步，但每卡要读 $(n-1)S$；
- **two-shot**：第一步每张卡只读其他卡的 $1/n$ 并求和（相当于 reduce-scatter），第二步再把求和结果读回来（all-gather）。两次同步，每卡读 $2\frac{n-1}{n}S$。

用 α-β 模型（[上一章](interconnect.md)）比较它们，8 卡 NVLink：

```python
import math

N, BETA, ALPHA = 8, 450e9, 1.5e-6        # 8 卡 NVLink：每卡单向带宽、每一步的固定开销（同步 + 发起）

ALGOS = {                                 # 一次 all-reduce（每卡 S 字节）的 α-β 时间
    "ring":     lambda S: 2 * (N - 1) * ALPHA + 2 * (N - 1) / N * S / BETA,
    "tree":     lambda S: 2 * math.log2(N) * ALPHA + 2 * S / BETA,
    "one-shot": lambda S: 2 * ALPHA + (N - 1) * S / BETA,               # 每卡直接读其他 7 卡的完整数据，本地求和
    "two-shot": lambda S: 4 * ALPHA + 2 * (N - 1) / N * S / BETA,       # 直接读写完成 reduce-scatter + all-gather
    "NVLS":     lambda S: 2 * ALPHA + S / BETA,                         # NVSwitch 在交换机里做加法：每卡只发一份、收一份
}

print(f"{'每卡数据':>8}" + "".join(f"{k:>10}" for k in ALGOS) + "   不用 NVLS 时最快（单位 µs）")
for S in (16 << 10, 256 << 10, 1 << 20, 8 << 20, 64 << 20):
    t = {k: f(S) * 1e6 for k, f in ALGOS.items()}
    size = f"{S >> 20} MiB" if S >= 1 << 20 else f"{S >> 10} KiB"
    best = min((k for k in t if k != "NVLS"), key=t.get)
    print(f"{size:>8}" + "".join(f"{v:>10.1f}" for v in t.values()) + f"   {best}")

S = 1 << 30                               # nccl-tests 的两种带宽：algbw = S / t，busbw = algbw × 2(n-1)/n
t = ALGOS["ring"](S)
print(f"1 GiB 的 ring all-reduce：{t * 1e3:.2f} ms，algbw {S / t / 1e9:.0f} GB/s，busbw {S / t * 2 * (N - 1) / N / 1e9:.0f} GB/s")
```

```text title="输出"
    每卡数据      ring      tree  one-shot  two-shot      NVLS   不用 NVLS 时最快（单位 µs）
  16 KiB      21.1       9.1       3.3       6.1       3.0   one-shot
 256 KiB      22.0      10.2       7.1       7.0       3.6   two-shot
   1 MiB      25.1      13.7      19.3      10.1       5.3   two-shot
   8 MiB      53.6      46.3     133.5      38.6      21.6   two-shot
  64 MiB     282.0     307.3    1046.9     267.0     152.1   two-shot
1 GiB 的 ring all-reduce：4.20 ms，algbw 256 GB/s，busbw 448 GB/s
```

换卡数、换链路、拉消息大小，看哪种走法最快：

<div class="aig-widget" data-widget="collective-cost"></div>

读这张表：

- **小消息（decode）**：ring 要走 14 步，光固定开销就 21 µs；one-shot 一步到位，3 µs。decode 一层两次 all-reduce、几十层下来，这就是毫秒级的差距。这正是 vLLM、SGLang 定制 all-reduce 的由来：小消息用 one-shot，中等消息用 two-shot；
- **大消息（prefill）**：one-shot 每卡要读 7 份数据，反而最慢；two-shot 和 ring 通信量相同。模型里 two-shot 一直比 ring 略快，但实际上定制实现要把输入先拷进一块预先注册好的共享缓冲区、占用较多 SM，缓冲区大小也有上限，所以推理框架在消息超过阈值（vLLM 默认约 8 MB）时交还给 NCCL；
- **NVLS** 在所有大小上都占优（模型偏乐观，但方向是对的），在支持的硬件上 NCCL 会自动启用；SGLang、vLLM 也可以直接调用 NCCL 的 NVLS 或 PyTorch 的对称内存（SymmetricMemory）实现的 all-reduce；
- 模型里 α 取 1.5 µs 只是示意，真实的分界点要在自己的机器上测。

## 协议：怎么知道数据到了

接收方要知道"数据已经完整到达"才能开始用。NCCL 有三种协议：

| 协议 | 做法 | 带宽效率 | 延迟 |
| --- | --- | --- | --- |
| **LL**（low latency） | 每 8 字节里 4 字节数据、4 字节标志，接收方轮询标志；靠 8 字节写入的原子性，不需要内存屏障 | 约 50% | 最低 |
| **LL128** | 每 128 字节里 120 字节数据、8 字节标志；依赖 NVLink 对 128 字节写入的保证 | 约 94% | 低 |
| **Simple** | 大块数据写完后加内存屏障，再写一个标志 | 接近 100% | 高（每块一次屏障） |

NCCL 按消息大小和拓扑自动选择：小消息用 LL，中等用 LL128，大消息用 Simple。自己写的 one-shot kernel 也面临同样的问题——每张卡写完自己的数据后，要用一个"信号量"通知其他卡，这个同步就是 α 的主要来源。

另一个维度是**通道**（channel）：NCCL 把一次通信拆到多个通道上并行，每个通道由一个 CUDA block 执行。也就是说，**NCCL 的通信 kernel 是占 SM 的**：它和计算 kernel 同时运行时会抢 SM。这就是 DeepEP 要让用户指定通信用多少个 SM、DeepSeek-V3 在训练时专门为通信划出 20 个 SM 的原因，也是 NVLS、拷贝引擎（copy engine）这类不占 SM 的传输方式受欢迎的原因。

## 读懂 nccl-tests

测多卡通信性能的标准工具是 `nccl-tests`（`all_reduce_perf -b 8 -e 1G -f 2 -g 8`）。它的输出有两种带宽：

- **algbw**（算法带宽）$= S / t$：用户视角，"每卡 $S$ 字节的 all-reduce 花了 $t$ 秒"；
- **busbw**（总线带宽）$= \text{algbw} \times$ 修正系数：换算成"每张卡的链路实际跑了多快"，可以直接和硬件的链路带宽比较。all-reduce 的系数是 $2\frac{n-1}{n}$，all-gather 和 reduce-scatter 是 $\frac{n-1}{n}$，broadcast 是 1。

上面的输出里，1 GiB 的 ring all-reduce 的 busbw 是 448 GB/s，接近 NVLink 的 450 GB/s，说明链路跑满了。实测时 8 卡 H100 的 all-reduce busbw 通常在 350～480 GB/s（后者依赖 NVLS）；跨机时看网卡带宽，每卡 400 Gb/s 的网络 busbw 在 40～48 GB/s 之间。**如果 busbw 远低于这些值，先查拓扑和配置，而不是怀疑算法。**

练习题库的 `judge.py bench` 会在你的机器上测多卡 all-reduce 的 busbw（见[练习题](root://practice/)）。

## 常用的环境变量与排查

| 变量 | 作用 |
| --- | --- |
| `NCCL_DEBUG=INFO`（配合 `NCCL_DEBUG_SUBSYS=INIT,GRAPH`） | 打印 NCCL 检测到的拓扑、选择的网卡、建立的 ring / tree，是排查的第一步 |
| `NCCL_ALGO` / `NCCL_PROTO` | 强制使用某种算法（`Ring`、`Tree`、`NVLS`）或协议（`LL`、`LL128`、`Simple`），用于对比实验 |
| `NCCL_NVLS_ENABLE` | 开关 NVLS |
| `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` | 指定用哪些 RDMA 网卡、哪个网口做初始化握手 |
| `NCCL_NET_GDR_LEVEL` / `NCCL_P2P_LEVEL` | 控制在什么拓扑距离内启用 GPUDirect RDMA 和 P2P |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | 通道数：影响带宽和占用的 SM 数 |

多卡推理服务"卡住"的常见原因：

- **各 rank 调用集合通信的顺序或参数不一致**：比如某个 rank 因为一个 if 分支多做了一次 all-reduce，或者张量形状不同。集合通信要求所有 rank 以相同顺序调用，否则就会永远等待。推理引擎里要特别注意"只有 rank 0 做的事"（采样、调度）不能夹在通信中间；
- **某个 rank 已经崩溃**：其他 rank 在通信里等它。看每个 rank 的日志，找最先出错的那个；
- **网络配置问题**：网卡选错、防火墙、RoCE 的 PFC 配置不当导致丢包重传。`NCCL_DEBUG=INFO` 会显示选中的网卡和传输方式（`NET/IB`、`NET/Socket`）；
- PyTorch 的 **flight recorder**（`TORCH_NCCL_TRACE_BUFFER_SIZE`）会记录每个 rank 最近的集合通信，卡住时导出来比对，就能看出哪个 rank 少调或多调了一次。

!!! source "源码对照"
    - **vLLM**：`vllm/distributed/device_communicators/` 下有 `custom_all_reduce.py`（CUDA IPC 的 one-shot / two-shot，kernel 在 `csrc/custom_all_reduce.cuh`）、`pynccl.py`（直接通过 ctypes 调 NCCL，便于在 CUDA Graph 里使用）、`symm_mem.py`（PyTorch 对称内存）、`flashinfer_all_reduce.py` 等，`cuda_communicator.py` 按消息大小和环境选择走哪一条路径。
    - **SGLang**：`srt/distributed/device_communicators/` 下有对应的 `custom_all_reduce.py`、`pynccl.py`、`torch_symm_mem.py`，`srt/layers/flashinfer_comm_fusion.py` 集成了 FlashInfer 的 all-reduce + RMSNorm 融合 kernel。

!!! interview "怎么讲清楚"
    "为什么推理框架要自己写 all-reduce"是常被追问的一点。讲法的骨架：decode 的消息只有几百 KB，NCCL 的 ring 要 $2(n-1)$ 步，时间几乎全是固定开销；定制实现用 CUDA IPC 直接读对端显存，one-shot 一步完成（小消息），two-shot 两步（中等消息），大消息交还 NCCL；再补充它必须能被 CUDA Graph 录制（缓冲区预先注册、地址固定），以及 NVLS 这类硬件归约让这个问题在新硬件上有了新答案。

## 练习

**1. 由 busbw 反推时间。** 8 卡机器上测得 all-reduce 的 busbw 是 360 GB/s。一次 256 MiB 的 all-reduce 要多久？如果换成 all-gather（每卡最终得到 256 MiB），又要多久？

??? success "参考答案"
    all-reduce：algbw $= 360 / (2 \times 7/8) = 205.7$ GB/s，$t = 268\,\text{MB} / 205.7\,\text{GB/s} \approx 1.3$ ms。
    all-gather 的系数是 $(n-1)/n$，假设它跑出同样的 busbw：algbw $= 360 / (7/8) = 411$ GB/s，$t \approx 0.65$ ms——all-gather 的通信量只有 all-reduce 的一半。这也是序列并行把 all-reduce 拆成 reduce-scatter + all-gather 后总通信量不变的原因。

**2. one-shot 的分界点。** 用本章的模型推导 one-shot 与 two-shot 的分界点公式。卡数从 8 变成 2 时，分界点怎么变？

??? success "参考答案"
    令 $2\alpha + (n-1)S/\beta = 4\alpha + 2\frac{n-1}{n}S/\beta$，得 $S^* = \frac{2\alpha\beta}{(n-1)(1 - 2/n)}$。$n = 8$ 时 $S^* = 2\alpha\beta / 5.25 \approx 257$ KB（表中 256 KiB 时两者几乎打平，正好在分界点附近）。$n = 2$ 时分母为 0：两张卡时 one-shot 和 two-shot 的通信量相同（都是 $S$），one-shot 只少一次同步，所以在任何大小上都不差于 two-shot。卡越少，one-shot 越划算；vLLM 的定制 all-reduce 也按卡数设置不同的阈值。

**3. 为什么定制 all-reduce 要预先注册缓冲区？**

??? success "参考答案"
    CUDA IPC 需要先把一块显存的句柄交换给其他进程，对方打开后得到映射地址——这是一次昂贵的、需要 CPU 参与的操作，不能每次通信都做。而且 decode 要用 CUDA Graph 录制，图里的 kernel 参数（包括对端缓冲区的地址）是固定的。所以定制 all-reduce 在启动时分配并交换好一块固定的缓冲区，每次通信先把输入拷进去（或者在录制 CUDA Graph 时直接注册图里用到的张量地址），这也限制了它能处理的最大消息大小。

## 小结

- [x] all-reduce 的算法：ring（带宽最优、步数多）、tree（步数 $\log n$，用于多机）、NVLS（交换机内归约，通信量减半、不占 SM）；推理框架另有 CUDA IPC 的 one-shot / two-shot。
- [x] 小消息看步数和同步次数，大消息看每卡收发的字节数；decode 用 one-shot / two-shot，prefill 交给 NCCL。
- [x] 协议 LL / LL128 / Simple 在延迟和带宽效率之间取舍；NCCL 的通道由 CUDA block 执行，会和计算抢 SM。
- [x] busbw = algbw × 修正系数（all-reduce 为 $2\frac{n-1}{n}$），可以直接和链路带宽比较。
- [x] 通信卡住先查：各 rank 调用顺序是否一致、有没有 rank 已崩溃、网卡和网络配置；工具是 `NCCL_DEBUG=INFO` 和 PyTorch 的 flight recorder。
