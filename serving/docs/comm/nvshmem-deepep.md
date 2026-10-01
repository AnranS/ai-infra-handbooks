# NVSHMEM 与 DeepEP：GPU 发起的通信

<p class="lead">MoE 的专家并行每一层都要做两次 all-to-all（dispatch 与 combine），数据量不大、形状不规则、对延迟极其敏感。用 NCCL 做，要先把每个目标收多少个 token 同步回 CPU、再发起通信；跨机时还有三分之二的流量要穿过脊交换机。DeepEP 用两种模式解决这个问题：高吞吐模式按节点去重、先 RDMA 再 NVLink 转发；低延迟模式由 GPU 直接发起 RDMA，用固定槽位换掉所有的 CPU 同步。它们都建立在 NVSHMEM 之上。这一章先讲 NVSHMEM 的编程模型，再用模拟和估算把 DeepEP 两种模式的设计讲清楚。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. NVSHMEM 的"对称堆"是什么？和 NCCL 的编程方式有什么不同？
    2. 用 NCCL 做 MoE 的 all-to-all，为什么需要一次 CPU 同步？
    3. DeepEP 的高吞吐模式怎样减少跨节点流量？"限制每个 token 最多去 4 个节点"起什么作用？
    4. 低延迟模式为什么要为每个来源预留固定的槽位？代价是什么？
    5. 低延迟模式的 hook 是怎样做到"通信不占 SM"的？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 所有 PE（每张 GPU）在启动时对称地分配同样大小的一块内存，同一个对象在每个 PE 上的偏移相同，所以用"对象地址 + 目标 PE 编号"就能访问远端的数据。和 NCCL 由 CPU 发起集合操作不同，NVSHMEM 让 GPU 线程在 kernel 里直接 put / get / 发信号，粒度可以到单个线程。
    2. 接收方要先知道自己会收到多少个 token，才能分配接收缓冲区、确定 `all_to_all_single` 的切分；这个数量在 GPU 上算出来，要拷回 CPU 再发起 all-to-all——一次 GPU 到 CPU 的同步，也让 CUDA Graph 无法录制。
    3. 按节点去重：一个 token 发往一个远端节点只发一份，先用 RDMA 发给那个节点上同轨的卡，再用 NVLink 转发给真正持有专家的卡。节点受限路由让每个 token 最多去 4 个节点，两者叠加，跨节点流量从每个 token 6.77 份降到 3.36 份。
    4. 固定槽位让发送方自己就知道往哪里写，接收方不需要事先知道会收到多少——省掉了数量交换和 CPU 同步，所有形状固定，可以录进 CUDA Graph。代价是按最坏情况（所有来源的所有 token 都选中同一个专家）预留显存：本章的例子里 232 MiB 只用到约 3%。
    5. dispatch 只发出 RDMA 请求就返回，把"等待并接收数据"留给一个 hook 函数，调用方在合适的时候再调用；数据在网络上传输时没有任何 kernel 在等，SM 全部可以去算别的（比如另一个 micro-batch）。

## NVSHMEM：在 kernel 里通信

NCCL 的模型是"主机端发起的集合通信"：CPU 调用 `ncclAllReduce`，NCCL 在流上启动一个通信 kernel，所有 rank 必须以相同的顺序调用。**NVSHMEM** 走的是另一条路——PGAS（分区全局地址空间）模型：

- 每个 GPU 是一个 **PE**（processing element）；
- **对称堆**：`nvshmem_malloc(size)` 是一个集体调用，在所有 PE 上分配同样大小的一块显存，而且在各自堆里的**偏移相同**。于是"PE 3 上的这块缓冲区的第 100 个元素"只需要一个本地指针加一个 PE 编号就能表示；
- **在 kernel 里通信**：GPU 线程可以直接调用 `nvshmem_put` / `nvshmem_get`（以及非阻塞的 `_nbi` 版本、按 warp / block 协作的版本）读写其他 PE 的对称内存，还有原子操作和"写数据 + 设置信号"的组合操作；
- **传输方式**：同一个 NVLink 域内，put / get 直接编译成对远端显存的加载和存储；跨机时走 RDMA——早期要由 CPU 上的代理线程替 GPU 提交请求，有了 IBGDA 之后 GPU 线程直接操作网卡（见上一章）。

一个"把本卡的一段数据发给另一张卡、并通知它"的 kernel 大致是这样（示意）：

```cuda
// buf、flags 都在对称堆上：在所有 PE 上偏移相同
__global__ void send_block(float* buf, uint64_t* flags, const float* x, size_t n, int dst_pe) {
    nvshmemx_float_put_nbi_block(buf, x, n, dst_pe);        // 整个 block 协作，把 x 写到 dst_pe 的 buf
    nvshmem_fence();                                        // 保证数据先于下面的信号到达
    if (threadIdx.x == 0)
        nvshmemx_signal_op(flags + nvshmem_my_pe(), 1, NVSHMEM_SIGNAL_SET, dst_pe);   // 通知对方：我的数据到了
}
// 接收方：nvshmem_signal_wait_until(flags + src_pe, NVSHMEM_CMP_EQ, 1);
```

和 NCCL 相比，NVSHMEM 让通信可以**和计算写在同一个 kernel 里**、按数据依赖细粒度地同步，而不必等一个完整的集合通信结束。代价是编程更底层：内存序、信号、缓冲区复用都要自己管。PyTorch 的对称内存（SymmetricMemory）提供了类似的能力。

## 为什么 MoE 的 all-to-all 难做

回顾[专家并行](../distributed/expert-parallel.md)一章的流程：路由 → 按目标排序 → **交换每个目标收多少个 token** → dispatch → 专家计算 → combine。用 NCCL 实现时，问题出在第三步：

- 接收方要知道自己会收到多少个 token，才能分配接收缓冲区、确定 `all_to_all_single` 的切分。这个数字在 GPU 上算出来，要**拷回 CPU**，CPU 再发起 all-to-all——一次 GPU 到 CPU 的同步，打断了流水，也让 CUDA Graph 无法录制（形状每次都不同）；
- 跨机时，一个 token 选中的 8 个专家分布在不同节点、不同卡上，直接发送会产生大量重复和跨轨的流量；
- NCCL 的 all-to-all 是一组点对点收发，占用 SM，和专家的矩阵乘抢资源。

DeepEP 针对训练和 prefill（大 batch、追求吞吐）与 decode（小 batch、追求延迟）分别设计了两种模式。

## 高吞吐模式：按节点去重、两跳转发

![图：DeepEP 高吞吐模式的两跳 dispatch——RDMA 到同位置的卡，再 NVLink 转发](../assets/figures/deepep-two-hop.svg){.aig-svg}

高吞吐模式（normal mode）的 dispatch 分两跳：

1. **RDMA**：一个 token 要去的每个**远端节点**只发一份，发给目标节点上与自己**位置相同**的那张卡（同轨，见[轨道拓扑](interconnect.md#节点之间infinibandroce-与轨道拓扑)）；
2. **NVLink**：目标节点上的这张卡再把 token 转发给本节点内真正持有目标专家的卡。

配合 DeepSeek-V3 的**节点受限路由**（每个 token 最多发往 4 个节点：先按每个节点上最高的 2 个专家分数之和选出 4 个节点，再在这 4 个节点里选 8 个专家），跨节点的流量进一步减少。模拟 64 张卡（8 个节点）、256 个专家、每个 token 选 8 个，统计节点 0 上一张卡的 4096 个 token 的跨节点流量：

```python
import numpy as np

NODES, GPN, EXPERTS, TOPK, MAX_NODES = 8, 8, 256, 8, 4
PER_GPU = EXPERTS // (NODES * GPN)                      # 每张卡 4 个专家
TOKENS, TOKEN_BYTES = 4096, 7168 + 7168 // 128 * 4      # 一张卡上的 token 数；FP8 隐藏向量 + 每 128 个元素一个 fp32 缩放
rng = np.random.default_rng(0)
scores = rng.random((TOKENS, EXPERTS)) + rng.random(EXPERTS) * 0.5    # 路由分数，专家之间有一点冷热差异


def route(limit_nodes):
    s = scores.copy()
    if limit_nodes:                                     # DeepSeek-V3：每个节点取最高的 2 个分数之和，只在得分最高的 4 个节点里选专家
        per_node = s.reshape(TOKENS, NODES, -1)
        node_score = np.sort(per_node, axis=-1)[:, :, -TOPK // MAX_NODES:].sum(-1)
        banned = np.argsort(node_score, axis=1)[:, :NODES - MAX_NODES]
        per_node[np.arange(TOKENS)[:, None], banned] = -np.inf
    return np.argsort(-s, axis=1)[:, :TOPK]


for limit in (False, True):
    experts = route(limit)                              # 源卡在节点 0
    gpu = experts // PER_GPU
    node = gpu // GPN
    per_expert = (node != 0).sum(1).mean()
    per_gpu = np.mean([len({g for g in row if g // GPN != 0}) for row in gpu])
    per_node = np.mean([len(set(row) - {0}) for row in node])
    print(f"{'限制每个 token 最多去 4 个节点' if limit else '不限制节点数'}：每个 token 平均涉及 {np.mean([len(set(r)) for r in node]):.2f} 个节点（含本节点）")
    for name, copies in (("按专家发", per_expert), ("按目标卡去重", per_gpu), ("按目标节点去重（DeepEP）", per_node)):
        t = TOKENS * copies * TOKEN_BYTES / 50e9 * 1e3
        print(f"  {name}：每个 token 跨节点 {copies:.2f} 份，{TOKENS} 个 token 走 50 GB/s 网卡 {t:.2f} ms")
```

```text title="输出"
不限制节点数：每个 token 平均涉及 5.28 个节点（含本节点）
  按专家发：每个 token 跨节点 6.77 份，4096 个 token 走 50 GB/s 网卡 4.10 ms
  按目标卡去重：每个 token 跨节点 6.50 份，4096 个 token 走 50 GB/s 网卡 3.93 ms
  按目标节点去重（DeepEP）：每个 token 跨节点 4.51 份，4096 个 token 走 50 GB/s 网卡 2.73 ms
限制每个 token 最多去 4 个节点：每个 token 平均涉及 3.97 个节点（含本节点）
  按专家发：每个 token 跨节点 6.70 份，4096 个 token 走 50 GB/s 网卡 4.05 ms
  按目标卡去重：每个 token 跨节点 6.26 份，4096 个 token 走 50 GB/s 网卡 3.79 ms
  按目标节点去重（DeepEP）：每个 token 跨节点 3.36 份，4096 个 token 走 50 GB/s 网卡 2.03 ms
```

两项措施叠加，跨节点流量从每个 token 6.77 份降到 3.36 份，正好减半；节点内的转发走 NVLink，带宽是网卡的 9 倍，基本不成为瓶颈。还有几个细节：

- **dispatch 用 FP8、combine 用 BF16**：dispatch 发出去的是专家的输入，可以量化（细粒度缩放，每 128 个元素一个缩放因子）；combine 送回来的是专家输出的加权和，保留 BF16 精度；
- **占用的 SM 可以配置**：通信 kernel 占多少个 SM 由用户指定（DeepSeek-V3 训练时只用 20 个 SM 就跑满了 IB 和 NVLink 的带宽），剩下的 SM 留给计算；
- **仍然需要一次 CPU 同步**：高吞吐模式先算出发往每个 rank、每个专家的 token 数（layout），交换之后 CPU 要知道收多少个 token 才能分配接收张量。所以它适合 prefill 和训练（batch 大，一次同步的开销可以摊薄），不能被 CUDA Graph 录制。

## 低延迟模式：固定槽位，没有 CPU 同步

decode 时每张卡每次只有几十到几百个 token，一次 CPU 同步（几十微秒）就可能超过通信本身。低延迟模式（low-latency mode）的做法：

- **纯 RDMA、GPU 直接发起**（IBGDA）：每个 token 直接发给目标专家所在的卡，不做节点内转发（少一跳，延迟更低）；
- **固定槽位**：每张卡为"每个本地专家 × 每个来源 rank"预留最多 `num_max_dispatch_tokens_per_rank` 个槽位。发送方知道自己往哪个槽位写，接收方不需要事先知道会收到多少个 token——省掉了 layout 的交换和 CPU 同步，所有形状都是固定的，可以录进 CUDA Graph。

代价是显存。按 DeepSeek-V3 的规模估算接收缓冲区：

```python
ranks, experts, max_tokens, hidden = 64, 256, 128, 7168     # 64 卡 EP、每卡每次最多发 128 个 token
local = experts // ranks
msg = hidden + hidden // 128 * 4 + 16                      # FP8 数据 + 缩放因子 + 少量头部
recv = local * ranks * max_tokens * msg                    # 每个本地专家为每个来源 rank 预留 max_tokens 个槽位
print(f"每张卡上 {local} 个专家 × {ranks} 个来源 × {max_tokens} 个槽位 × {msg} 字节 = {recv / 2**20:.0f} MiB")
print(f"真正用到的（每个 token 选 8 个专家、平均分摊）：{max_tokens * ranks * 8 / experts * local * msg / 2**20:.1f} MiB")
```

```text title="输出"
每张卡上 4 个专家 × 64 个来源 × 128 个槽位 × 7408 字节 = 232 MiB
真正用到的（每个 token 选 8 个专家、平均分摊）：7.2 MiB
```

只有约 3% 的槽位会被用到，但必须按最坏情况（所有来源的所有 token 都选中同一个专家）预留——这是用显存换延迟的典型设计。实际的缓冲区还要加上发送缓冲区、BF16 的 combine 缓冲区和双缓冲，所以低延迟模式的显存开销在 GB 量级，EP 规模越大越多；`num_max_dispatch_tokens_per_rank` 也因此成了一个要按 decode 的 batch 大小仔细设置的参数。

**hook：通信不占 SM。** 低延迟模式的 dispatch 可以只发出 RDMA 请求就返回，把"等待并接收数据"留给一个 hook 函数，调用方在合适的时候再调用它。数据在网络上飞的这段时间里，没有任何 kernel 在等待，SM 全部可以用来算别的东西。SGLang 的**双 batch 重叠**（two-batch overlap）就利用了这一点：把一个 batch 拆成两个 micro-batch，一个在做注意力和专家计算时，另一个的 dispatch / combine 在网络上传输，两者交替。

| | 高吞吐模式 | 低延迟模式 |
| --- | --- | --- |
| 场景 | 训练、prefill（大 batch） | decode（小 batch） |
| 路径 | RDMA 到同轨的卡，再 NVLink 转发；按节点去重 | 纯 RDMA 直达目标卡 |
| 接收缓冲区 | 先交换数量、按需分配（需要 CPU 同步） | 固定槽位，按最坏情况预留 |
| CUDA Graph | 不支持 | 支持 |
| 与计算重叠 | 指定通信占用的 SM 数，与计算并行 | hook：发出后返回，不占 SM |

## 在推理框架里使用

- **SGLang**：`--moe-a2a-backend deepep` 启用 DeepEP，`--deepep-mode` 可选 `normal`、`low_latency` 或 `auto`（prefill 用高吞吐、decode 用低延迟）；配合 DP Attention 和双 batch 重叠使用。在 PD 分离部署里，prefill 实例和 decode 实例正好分别对应两种模式；
- **vLLM**：`--all2all-backend` 选 `deepep_high_throughput` 或 `deepep_low_latency`（默认是基于 all-gather / reduce-scatter 的实现，另有 MoRI、NIXL-EP、FlashInfer 等后端）；
- 这些后端都要求 NVSHMEM 和 IBGDA 可用：驱动、网卡固件、内核参数都要配置好，这是部署大规模 EP 时最常见的"环境问题"来源。

!!! interview "面试怎么答"
    DeepEP 几乎是大规模 MoE 推理面试的必考点。回答时先讲问题：MoE all-to-all 的数据依赖路由结果、量小、延迟敏感，NCCL 需要 CPU 同步交换数量且占 SM；再讲两种模式：高吞吐模式按节点去重、同轨 RDMA + NVLink 转发、FP8 dispatch / BF16 combine、可配置 SM 数，适合 prefill；低延迟模式 IBGDA 直发、固定槽位免同步、支持 CUDA Graph、hook 不占 SM，适合 decode。最后用数字收尾：节点受限路由加按节点去重让跨节点流量减半；低延迟模式的槽位只有几个百分点被用到，是用显存换延迟。

## 练习

**1. 为什么是"同位置"的卡？** 高吞吐模式跨节点时，为什么要发给目标节点上与自己位置相同的卡，而不是直接发给持有目标专家的那张卡？

??? success "参考答案"
    轨道优化的拓扑里，不同节点上位置相同的卡接在同一个叶交换机上，只隔一跳；发给其他位置的卡要经过脊交换机，和其他流量争抢上行链路。另外，按节点去重后一个 token 发往每个节点只有一份，必须由目标节点上的某张卡负责再分发——选同位置的卡，既走了最短的网络路径，又把分发工作均匀地摊到节点内的每张卡上。

**2. 槽位设多大？** decode 实例每张卡的 batch 最大是 64 个 token，EP=32，每卡 8 个专家，隐藏维度 7168（FP8 加缩放约 7.4 KB）。`num_max_dispatch_tokens_per_rank` 至少要设多少？dispatch 的接收缓冲区要多大？如果 batch 增加到 256 呢？

??? success "参考答案"
    至少等于每张卡每次最多发出的 token 数，即 64。接收缓冲区 $= 8 \times 32 \times 64 \times 7.4\,\text{KB} \approx 121$ MB。batch 增加到 256 时槽位要跟着设成 256，缓冲区变成约 485 MB——四倍。槽位设小了，超出的 token 发不出去（实现会报错或要求拆批）；设大了浪费显存、挤占 KV Cache。这就是 decode 的最大 batch、EP 规模和 KV Cache 容量之间要一起规划的原因。

**3. 什么时候不需要 DeepEP？**

??? success "参考思路"
    EP 规模在一个 NVLink 域以内（比如单机 8 卡，或 NVL72 超节点）时，没有跨节点流量，按节点去重和两跳转发没有意义；这时 NCCL 的 all-to-all、基于 CUDA IPC 的机内实现，甚至不用 EP、改用 TP 都可能更简单。模型的专家数少、每个 token 选的专家少时，all-to-all 的量本来就小，收益也有限。DeepEP 的价值主要在"大规模、跨节点的细粒度 MoE"。

## 小结

- [x] NVSHMEM：PGAS 模型，对称堆上偏移相同，GPU 线程在 kernel 里直接 put / get / 发信号；机内走 NVLink 访存，跨机走 RDMA（IBGDA 让 GPU 直接操作网卡）。
- [x] 用 NCCL 做 MoE all-to-all 的问题：需要把每个目标的 token 数同步回 CPU、不能录 CUDA Graph、跨机流量大、占 SM。
- [x] 高吞吐模式：按节点去重，同轨 RDMA 再 NVLink 转发，配合节点受限路由把跨节点流量减半；FP8 dispatch、BF16 combine；需要 CPU 同步，用于训练和 prefill。
- [x] 低延迟模式：IBGDA 直发目标卡，固定槽位免去数量交换，可录 CUDA Graph；hook 让通信不占 SM，配合双 batch 重叠；代价是按最坏情况预留的显存。
