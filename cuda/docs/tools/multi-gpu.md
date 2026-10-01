# 多 GPU 与 NCCL

<p class="lead">大模型放不进一张卡，也不可能只用一张卡训练和服务。多卡之间怎么切分模型、用什么互联、集合通信怎么实现、通信和计算怎么重叠，是 AI Infra 岗位绕不开的话题。这一章讲清楚这些概念，并用 NCCL 写一个多卡 all-reduce。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 张量并行（TP）的每一层需要做什么集合通信？专家并行（EP）呢？
    2. ring all-reduce 中每张卡发送的数据量是多少？为什么和卡数几乎无关？
    3. NCCL 测试报告里的 algbw 和 busbw 有什么区别？
    4. decode 阶段 TP 的 all-reduce 通常是延迟瓶颈还是带宽瓶颈？
    5. 怎么让通信和计算重叠？

??? success "自测参考答案（先自己答，再展开对照）"
    1. TP：每层的注意力和 MLP 各一次 all-reduce（把按行切分的矩阵乘的部分和加起来）；EP：每个 MoE 层两次 all-to-all（把 token 发给专家、把结果送回来）。
    2. 每张卡发送 $2(n-1)/n \times S$ 字节：reduce-scatter 和 all-gather 各发 $n-1$ 块、每块 $S/n$。卡数增加时这个值趋近 $2S$，所以几乎与卡数无关。
    3. algbw = 数据量 ÷ 时间，是用户视角的带宽；busbw 按算法实际在链路上传输的数据量做了换算（all-reduce 乘以 $2(n-1)/n$），可以直接和链路的峰值带宽比较、判断跑满了没有。
    4. 延迟瓶颈：decode 时每次 all-reduce 只有几百 KB，环形算法的 $2(n-1)$ 步里每一步都有固定开销，时间几乎全花在延迟上。
    5. 把通信放在单独的流上，和不依赖它的计算并行：训练里梯度分桶、边反向边 all-reduce；把计算切成块，算一块传一块；用融合通信的 kernel（计算与通信在一个 kernel 里交替）；MoE 用专门的 all-to-all 库和双 batch 重叠。

## 为什么要多卡，以及怎么切分

| 并行方式 | 切分什么 | 通信 | 典型场景 |
| --- | --- | --- | --- |
| 数据并行（DP） | 数据；每张卡一份完整模型 | 训练时梯度 all-reduce | 模型放得下单卡时的训练；推理时的多副本 |
| 张量并行（TP） | 每个矩阵按行或列切到多张卡 | 每层 1-2 次 all-reduce（或 all-gather + reduce-scatter） | 单机多卡（NVLink），推理的主要手段 |
| 流水线并行（PP） | 按层切分 | 相邻阶段之间点对点传激活 | 跨机、超大模型 |
| 专家并行（EP） | MoE 的不同专家放在不同卡上 | 每个 MoE 层两次 all-to-all（分发 token、收回结果） | DeepSeek、Qwen-MoE 等 MoE 模型 |
| 序列/上下文并行（SP/CP） | 沿序列长度切分 | 注意力需要交换 K、V（如 Ring Attention） | 超长上下文 |

以 Transformer 的 MLP 为例，Megatron 风格的 TP 把第一个线性层按**列**切（每张卡得到输出的一部分，激活函数可以本地计算），第二个线性层按**行**切（每张卡得到完整形状的部分和），最后做一次 **all-reduce** 求和。注意力层同理：按头切分，输出投影之后一次 all-reduce。所以 TP 每一层需要两次 all-reduce。

## 互联

| 互联 | 带宽量级 | 说明 |
| --- | --- | --- |
| PCIe 4.0 / 5.0 x16 | 每方向约 32 / 64 GB/s（理论） | 消费级卡和部分推理卡只有它；卡间通信经常要绕过 CPU |
| NVLink（H100，第四代） | 每张卡合计 900 GB/s（双向） | 单机 8 卡通过 NVSwitch 全互联 |
| NVLink（B200，第五代） | 每张卡合计 1.8 TB/s（双向） | GB200 NVL72 把 72 张卡连成一个 NVLink 域 |
| InfiniBand / RoCE | 每张网卡 400 Gb/s（约 50 GB/s） | 跨机通信；配合 GPUDirect RDMA，网卡直接读写显存 |

单机内的 TP 靠 NVLink；跨机的 TP 通信代价太高，所以通常"机内 TP、机间 PP/EP/DP"。

## CUDA 点对点访问

同一台机器上的两张卡开启对等访问（peer access）后，可以直接拷贝，甚至在 kernel 里直接读写对方的显存（走 NVLink 或 PCIe）：

```cuda
int can = 0;
cudaDeviceCanAccessPeer(&can, 0, 1);
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);                    // 设备 0 可以访问设备 1 的内存
cudaMemcpyPeerAsync(dst_on_1, 1, src_on_0, 0, bytes, stream);
```

vLLM、TensorRT-LLM 里的**自定义 all-reduce** 就是基于这种能力：对于 decode 阶段的小消息（几十 KB），每张卡直接从其他卡的缓冲区读取数据并求和（one-shot），或者先 reduce-scatter 再 all-gather（two-shot），比通用的 NCCL 延迟更低。

## 集合通信

| 操作 | 含义 |
| --- | --- |
| broadcast | 一张卡的数据发给所有卡 |
| reduce | 所有卡的数据求和（或 max 等）到一张卡 |
| all-reduce | 所有卡的数据求和，结果所有卡都有 |
| all-gather | 每张卡的一块拼起来，所有卡得到完整数据 |
| reduce-scatter | 求和后切块，每张卡得到结果的一块 |
| all-to-all | 每张卡给每张卡发一块不同的数据（MoE 的 token 分发） |

一个重要的等式：**all-reduce = reduce-scatter + all-gather**。

### ring all-reduce

每一步谁把什么发给谁，用分布式训练手册里的同一个工具一步步看：

<div class="aig-widget" data-widget="ringreduce"></div>

p 张卡排成一个环，数据切成 p 块：

1. **reduce-scatter 阶段**（p-1 步）：每一步，每张卡把一块发给下一张卡，下一张卡把收到的块和自己对应的块相加。p-1 步之后，每张卡恰好拥有某一块的完整和；
2. **all-gather 阶段**（p-1 步）：每张卡把已完成的块沿环传下去，p-1 步之后所有卡都拿到全部结果。

每张卡总共发送 $2(p-1) \cdot \frac{S}{p}$ 字节（S 是数据总大小），当 p 较大时约为 **2S，几乎与卡数无关**，并且所有链路同时工作，充分利用了带宽。代价是需要 $2(p-1)$ 步，每一步都有固定的延迟，所以**小消息时延迟占主导**。NCCL 会根据消息大小和拓扑，在 ring、tree 以及 NVSwitch 上的 NVLS（交换机内归约）等算法之间选择。

### 带宽的两种统计：algbw 与 busbw

`nccl-tests`（NVIDIA 的官方测试工具）报告两个带宽：

- **algbw** = S / 耗时，直观但不同操作、不同卡数之间不可比；
- **busbw**：换算成"每条链路实际承载的带宽"。对 all-reduce，busbw = algbw × $\frac{2(p-1)}{p}$。它可以直接与硬件链路的峰值带宽比较。

```bash
# 单机 8 卡测试 all-reduce，消息从 8 B 到 1 GB，每次翻倍
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

## NCCL 编程

NCCL 的基本流程：创建通信器（communicator），在指定的流上发起集合通信。下面的例子在**单个进程里驱动多张卡**（`ncclCommInitAll`），每张卡填入不同的数据，做一次 all-reduce 并验证结果。多进程（每张卡一个进程，训练和推理框架的常见做法）时，改用 `ncclGetUniqueId` + 通过 MPI 或 TCP 分发 id + `ncclCommInitRank`。

```cuda title="nccl_allreduce.cu"
// nccl_allreduce.cu —— 单进程多 GPU 的 NCCL all-reduce
// 编译：nvcc -O3 -arch=sm_75 nccl_allreduce.cu -o nccl_allreduce -lnccl
#include "common.cuh"
#include <nccl.h>

#define NCCL_CHECK(call)                                                                 \
  do {                                                                                   \
    ncclResult_t r_ = (call);                                                            \
    if (r_ != ncclSuccess) {                                                             \
      std::fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                           \
    }                                                                                    \
  } while (0)

__global__ void fill(float* x, int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = value + i % 7;
}

int main() {
  int ngpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ngpus));
  if (ngpus < 2) {
    std::printf("SKIP: needs at least 2 GPUs, found %d\n", ngpus);
    return 0;
  }
  const int n = 32 * 1024 * 1024;   // 每张卡 128 MB
  std::vector<int> devs(ngpus);
  for (int i = 0; i < ngpus; ++i) devs[i] = i;
  std::vector<ncclComm_t> comms(ngpus);
  NCCL_CHECK(ncclCommInitAll(comms.data(), ngpus, devs.data()));

  std::vector<float*> buf(ngpus);
  std::vector<cudaStream_t> streams(ngpus);
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&buf[g], n * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&streams[g]));
    fill<<<(n + 255) / 256, 256, 0, streams[g]>>>(buf[g], n, static_cast<float>(g));   // 第 g 张卡：g + i % 7
    CUDA_CHECK_LAST();
  }

  auto allreduce = [&] {
    // 一个线程为多个通信器发起集合通信时，必须用 group 包起来，否则会死锁
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < ngpus; ++g)
      NCCL_CHECK(ncclAllReduce(buf[g], buf[g], n, ncclFloat, ncclSum, comms[g], streams[g]));   // 原地
    NCCL_CHECK(ncclGroupEnd());
  };
  allreduce();
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }

  // 期望：sum_g (g + i % 7) = ngpus * (ngpus - 1) / 2 + ngpus * (i % 7)
  std::vector<float> h(n);
  size_t bad = 0;
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMemcpy(h.data(), buf[g], n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) bad += h[i] != ngpus * (ngpus - 1) / 2.f + ngpus * static_cast<float>(i % 7);
  }
  std::printf("all-reduce across %d GPUs: %s (%zu mismatches)\n", ngpus, bad ? "FAIL" : "PASS", bad);

  // 计时：以卡 0 的流为准（其他卡同步完成）
  CUDA_CHECK(cudaSetDevice(0));
  GpuTimer t;
  const int iters = 20;
  t.start(streams[0]);
  for (int it = 0; it < iters; ++it) allreduce();
  float ms = t.stop(streams[0]) / iters;
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }
  const double algbw = n * sizeof(float) / (ms * 1e-3) / 1e9;
  std::printf("%.3f ms, algbw %.1f GB/s, busbw %.1f GB/s\n", ms, algbw, algbw * 2.0 * (ngpus - 1) / ngpus);

  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaFree(buf[g]));
    CUDA_CHECK(cudaStreamDestroy(streams[g]));
    NCCL_CHECK(ncclCommDestroy(comms[g]));
  }
  return bad ? 1 : 0;
}
```

几个常见的坑：

- **group 调用**：一个线程为多个通信器发起操作时必须用 `ncclGroupStart/End` 包起来；
- **所有 rank 必须以相同的顺序发起相同的集合通信**，否则会挂起。分布式程序"卡住不动"最常见的原因就是某个 rank 少调了或多调了一次集合通信；
- 调试时设置 `NCCL_DEBUG=INFO` 可以看到 NCCL 选择的算法、使用的互联和拓扑。

## 通信与计算重叠

集合通信在独立的流上执行（NCCL 的 kernel 也会占用一部分 SM）。常见的重叠手段：

- **训练**：反向传播时，一个梯度桶（bucket）算完就立刻开始它的 all-reduce，同时继续计算下一层的梯度，PyTorch DDP 就是这样做的；
- **分块重叠**：把一个大 GEMM 和它后面的 all-reduce 都切成若干块，第 i 块的通信与第 i+1 块的计算并行；
- **融合通信与计算的 kernel**：在同一个 kernel 里一边算 GEMM 一边通过 NVLink 读写其他卡的数据，比如 all-gather + GEMM、GEMM + reduce-scatter 的融合，FLUX、NVIDIA 的 Transformer Engine userbuffers 都在做这方面的工作；
- **MoE 的 all-to-all**：DeepSeek 开源的 DeepEP 为 MoE 的分发和合并提供了高吞吐（训练、prefill）和低延迟（decode）两套 kernel，低延迟版本使用纯 RDMA 并支持与计算重叠。

!!! interview "面试怎么答"
    多卡通信题：TP 每层做 all-reduce，EP 每个 MoE 层做两次 all-to-all，PP 在阶段之间点对点传输；机内用 NVLink / NVSwitch，机间用 InfiniBand / RoCE 加 GPUDirect RDMA，所以通常机内 TP、机间 PP / EP / DP。ring all-reduce 每张卡发送约 2S 字节，与卡数几乎无关，但步数是 2(n−1)，decode 的小消息受延迟而不是带宽限制。测试报告看 busbw 并与链路峰值比较；通信与计算的重叠靠梯度分桶、分块流水、融合 kernel 和专用的 all-to-all 库。

!!! info "相关章节"
    - [集合通信原语](train://basics/collectives/)（分布式训练：五个原语与环形 all-reduce 的通信量）
    - [GPU 互联与网络](serving://comm/interconnect/)、[集合通信：NCCL 的算法与协议](serving://comm/nccl/)（推理系统）
    - [多卡系统：NVLink、拓扑、切分与功耗](cs://arch/multi-gpu/)（计算机基础）

## 练习

**1. 估算 TP 通信开销。** 一个 hidden = 8192 的模型在 8 张 H100 上做 TP，BF16。估算以下两种情况下一次 all-reduce 的耗时，并判断是延迟瓶颈还是带宽瓶颈：（a）decode，batch = 1；（b）prefill，一次 8192 个 token。假设 NVLink 每方向可用带宽约 400 GB/s，每次 all-reduce 的固定延迟约 10-20 µs。

??? success "参考答案"
    - **（a）**：数据量 = 1 × 8192 × 2 B = 16 KB。按带宽算，1.75 × 16 KB / 400 GB/s ≈ 0.07 µs，远小于固定延迟，**延迟瓶颈**，耗时约为固定延迟本身（十几微秒）。每层两次，几十层下来就是毫秒级，这正是 decode 阶段要用自定义 all-reduce、CUDA Graphs 的原因。
    - **（b）**：数据量 = 8192 × 8192 × 2 B = 128 MB。ring all-reduce 每张卡发送约 2 × 7/8 × 128 MB ≈ 224 MB，耗时约 224 MB / 400 GB/s ≈ 0.56 ms，**带宽瓶颈**。此时通信和计算的重叠就变得非常重要。

**2. 为什么 all-to-all 比 all-reduce 更难优化？**

??? success "参考答案"
    all-reduce 的数据量固定、模式规则，ring/tree 算法可以把所有链路均匀地用满。MoE 的 all-to-all 则取决于路由结果：每张卡发给每张卡的数据量不同且每一步都在变化，负载可能严重不均衡（热门专家所在的卡要接收更多 token）；跨机时还要同时用好 NVLink 和 RDMA 两种链路。所以需要专门的实现（如 DeepEP），以及模型层面的负载均衡（辅助损失、无辅助损失的偏置调整、冗余专家等）。

## 小结

- [x] TP 每层做 all-reduce，EP 每个 MoE 层做两次 all-to-all，PP 在阶段间点对点传输。
- [x] 机内用 NVLink/NVSwitch，机间用 InfiniBand/RoCE + GPUDirect RDMA；通常机内 TP、机间 PP/EP/DP。
- [x] ring all-reduce 每卡发送约 2S 字节，与卡数几乎无关，但步数多，小消息受延迟限制。
- [x] 用 busbw 与链路峰值带宽比较；NCCL 多通信器操作要用 group 包裹，所有 rank 调用顺序必须一致。
- [x] 通信计算重叠：梯度分桶、分块流水、融合 kernel、专用 all-to-all 库。
