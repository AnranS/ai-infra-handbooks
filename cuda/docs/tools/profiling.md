# 性能分析：Nsight Systems 与 Nsight Compute

<p class="lead">优化之前先测量，这一点在 GPU 上尤其重要：程序慢，可能是 kernel 本身慢，也可能是 kernel 之间有空隙、CPU 跟不上、拷贝没有重叠。NVIDIA 提供了两个互补的工具：Nsight Systems 看整个程序的时间线，Nsight Compute 深入分析单个 kernel。会用它们、能读懂指标，是算子开发岗位的硬性要求。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Nsight Systems 和 Nsight Compute 分别回答什么问题？
    2. 时间线上 kernel 之间有很多小空隙，说明什么？怎么解决？
    3. Nsight Compute 的 "Speed Of Light" 部分给出的两个百分比是什么意思？
    4. 怎么用指标确认一个 kernel 有 bank 冲突、访存没有合并？
    5. warp 停顿原因里的 "Long Scoreboard" 和 "MIO Throttle" 分别意味着什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. Nsight Systems 看整体的时间线：CPU 和 GPU 在做什么、kernel 之间有没有空隙、拷贝和计算是否重叠——回答"时间花在哪里"；Nsight Compute 深入分析单个 kernel：访存、计算、停顿原因——回答"这个 kernel 为什么慢"。
    2. GPU 在等 CPU：kernel 太小太多、CPU 提交跟不上，或者中间有同步。用 CUDA Graph 一次提交整串 kernel、融合小算子、去掉不必要的同步。
    3. 分别是显存（及各级内存）吞吐和计算（SM）吞吐占峰值的百分比。哪个接近峰值，就是被它限制；两个都低，说明是延迟瓶颈（并行度不够、等待太多）。
    4. 看全局访存的扇区数 / 请求数：合并访问时每个请求约 4 个扇区（每线程读 4 字节），远高于它说明没合并；共享内存看 bank conflict 相关的指标（每条指令的访问轮次、冲突次数）。
    5. Long Scoreboard：在等全局内存（或 local memory）的数据回来，说明访存延迟没被掩盖；MIO Throttle：共享内存等访存指令的队列满了，通常是共享内存指令太多或者有 bank 冲突。

## 先看全局，再看单个 kernel

| 工具 | 命令 | 回答的问题 |
| --- | --- | --- |
| **Nsight Systems**（nsys） | `nsys profile ./app` | 时间都花在哪里？GPU 有没有空闲？CPU 和 GPU、拷贝和计算有没有重叠？哪个 kernel 最耗时？ |
| **Nsight Compute**（ncu） | `ncu ./app` | 这个 kernel 为什么慢？瓶颈是访存、计算还是延迟？具体是哪行代码？ |

标准的分析流程：

1. 用 nsys 看时间线，找出最耗时的部分，确认是 kernel 本身慢，还是 kernel 之外（启动开销、拷贝、同步、CPU 计算）的问题；
2. 对最耗时的几个 kernel，用 ncu 分析瓶颈；
3. 修改代码，**重新测量**，确认优化真的有效。

先用一个简单的时间模型感受"全局"是什么意思——decode 一步上千个小 kernel 时，启动开销和空隙占了多少（推理系统手册里的同一个工具）：

<div class="aig-widget" data-widget="launch-overhead"></div>

## Nsight Systems

```bash
nsys profile -o report --trace=cuda,nvtx,osrt ./app       # 生成 report.nsys-rep
nsys stats report.nsys-rep                                 # 在命令行输出各类统计
nsys profile --trace=cuda,nvtx python train.py            # Python 程序同样适用
```

用图形界面（Nsight Systems GUI，可以装在自己的电脑上，打开服务器上生成的报告）查看时间线时，重点关注：

| 现象 | 可能的原因 | 应对 |
| --- | --- | --- |
| kernel 之间有很多几微秒的空隙 | kernel 太小太多，启动开销占主导；或者 CPU 发射速度跟不上 | 融合 kernel；用 CUDA Graphs 一次提交整个计算图，见[流与 CUDA Graphs](streams.md) |
| GPU 大段空闲，CPU 很忙 | CPU 端的预处理、Python 开销、同步等待 | 把工作搬到 GPU；异步执行；减少 `cudaDeviceSynchronize` 和 `.item()` 之类的隐式同步 |
| 拷贝和 kernel 串行执行 | 没有使用锁页内存和多个流 | 锁页内存 + 异步拷贝 + 多流流水 |
| 某个 kernel 占了大部分时间 | 它就是优化目标 | 用 ncu 深入分析 |

### 用 NVTX 标注代码

在时间线上，一堆 kernel 很难分清属于程序的哪个阶段。NVTX 让你给代码段打上带名字的标记，它们会作为彩色区间显示在时间线上：

```cuda title="nvtx_demo.cu"
// nvtx_demo.cu —— 用 NVTX 标注程序阶段，在 Nsight Systems 时间线上查看
// 编译：nvcc -O3 -arch=sm_75 nvtx_demo.cu -o nvtx_demo
// 分析：nsys profile --trace=cuda,nvtx -o nvtx_demo ./nvtx_demo
#include "common.cuh"
#include <nvtx3/nvToolsExt.h>

__global__ void scale(float* x, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}

int main() {
  const int n = 1 << 24;
  std::vector<float> h(n, 1.f);
  float* d;

  nvtxRangePushA("setup");
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  nvtxRangePop();

  nvtxRangePushA("compute");
  for (int step = 0; step < 10; ++step) {
    nvtxRangePushA("step");
    scale<<<(n + 255) / 256, 256>>>(d, n, 1.01f);
    CUDA_CHECK_LAST();
    nvtxRangePop();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  const float expect = std::pow(1.01f, 10.f);
  const bool ok = std::fabs(h[0] - expect) < 1e-4f && std::fabs(h[n - 1] - expect) < 1e-4f;
  std::printf("%s (x = %.6f, expected %.6f)\n", ok ? "PASS" : "FAIL", h[0], expect);
  CUDA_CHECK(cudaFree(d));
  return ok ? 0 : 1;
}
```

NVTX v3 是只有头文件的库，不需要额外链接。注意 CPU 上的 NVTX 区间只表示"发出这些 CUDA 调用的时间"，kernel 实际在 GPU 上的执行时间要看 GPU 那一行。PyTorch 里可以用 `torch.cuda.nvtx.range_push/range_pop` 或者 `torch.profiler` 获得类似的效果。

## Nsight Compute

```bash
ncu ./app                                          # 分析所有 kernel（默认的指标集）
ncu --set full -o gemm_report ./gemm               # 收集全部指标，保存为 gemm_report.ncu-rep
ncu -k regex:sgemm_v4 --launch-skip 2 --launch-count 1 --set full -o v4 ./gemm
                                                    # 只分析名字匹配的 kernel，跳过前 2 次，只抓 1 次
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./app   # 只收集指定的指标
ncu --query-metrics                                 # 列出当前 GPU 支持的所有指标
```

ncu 会**多次重放**每个被分析的 kernel 以收集不同的指标，所以程序会变慢很多，只分析你关心的几个 kernel（`-k`、`--launch-count`）。默认情况下它还会把 GPU 时钟锁在基础频率（`--clock-control base`），并在每次重放前清空缓存（`--cache-control all`），这让结果更稳定、更可比，但绝对耗时会和正常运行时不同。

在服务器上没有权限访问性能计数器时，会报 `ERR_NVGPUCTRPERM`，需要管理员开启（或者在容器里加上相应的权限）。

### 读懂报告：从 Speed Of Light 开始

**GPU Speed Of Light Throughput** 是最先看的部分，给出两个百分比：

- **Compute (SM) Throughput**：计算相关单元（各个执行管线、指令发射）的利用率，相对于峰值；
- **Memory Throughput**：各级存储（DRAM、L2、L1、共享内存）中最繁忙的那一个的利用率。

用它们判断 kernel 属于哪一类：

| 情况 | 判断 | 下一步看什么 |
| --- | --- | --- |
| Memory 高（比如 > 70%），Compute 低 | 访存瓶颈 | Memory Workload Analysis：是哪一级存储满了？DRAM 满了说明已经接近上限，只能减少访存量；L1/共享内存满了则可能有 bank 冲突或未合并访问 |
| Compute 高，Memory 低 | 计算瓶颈 | Compute Workload Analysis：哪个管线最忙（FMA、Tensor、ALU、SFU）？能不能换成更快的指令（Tensor Core、快速数学函数） |
| 两者都低（比如都 < 40%） | **延迟瓶颈**：硬件既没算满也没传满，warp 大部分时间在等待 | Occupancy 和 Warp State Statistics：驻留的 warp 够不够？在等什么？ |
| 两者都高 | 已经接近硬件极限 | 考虑算法层面的改变（融合、换数据类型、减少计算量） |

报告里的 **Roofline 图** 把 kernel 画在算术强度-性能坐标系里，直观显示它离哪条"屋顶"更近。

### 常用指标

| 想知道 | 指标 / 报告位置 |
| --- | --- |
| kernel 耗时 | `gpu__time_duration.sum` |
| 从显存读写了多少字节 | `dram__bytes_read.sum`、`dram__bytes_write.sum` |
| 显存带宽利用率 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| 全局内存访问是否合并 | 扇区数 / 请求数：`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 除以 `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`。每个线程读 4 字节、完全合并时是 4；越大说明浪费越多。报告的 Source Counters 部分还会直接指出未合并访问的代码行 |
| 共享内存 bank 冲突 | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`（读）、`..._op_st.sum`（写） |
| 实际占用率 | `sm__warps_active.avg.pct_of_peak_sustained_active`；Occupancy 部分还会给出理论占用率和限制因素（寄存器、共享内存、block 大小） |
| 寄存器溢出 | Source 视图里的 local memory 访问；编译时 `-Xptxas -v` |

指标名字很长，但有规律：`单元__计数对象_限定.汇总方式`。比如 `dram__bytes_read.sum` 是"DRAM 单元读取的字节数，求和"。不确定时用 `ncu --query-metrics` 搜索。

### warp 停顿原因

**Warp State Statistics** 统计 warp 在不能发射指令时都在等什么。最常见的几种：

| 停顿原因 | 含义 | 常见的对策 |
| --- | --- | --- |
| Long Scoreboard | 等待全局内存/本地内存（L1TEX）的数据返回 | 增加在途请求（向量化、每线程多个元素、预取）；提高占用率；检查寄存器溢出 |
| Short Scoreboard | 等待共享内存或特殊函数单元的结果 | 减少共享内存访问次数（寄存器分块）；消除 bank 冲突 |
| MIO Throttle | 共享内存等"存储输入输出"指令队列满了 | 同上，典型出现在共享内存读取过于密集的 GEMM v2 这类 kernel |
| LG Throttle | 全局/本地内存指令队列满了 | 用向量化访存减少指令数 |
| Barrier | 在 `__syncthreads()` 等待其他 warp | 减少同步次数；平衡 warp 间的工作量 |
| Math Pipe Throttle | 某个计算管线忙不过来 | 通常说明已经是计算瓶颈，属于好现象；可以考虑换用其他管线 |
| Not Selected | 已就绪但本周期调度器选了别的 warp | 说明就绪的 warp 足够多，不是问题 |

**Source 视图**（需要编译时加 `-lineinfo`）能把这些指标对应到每一行源码和每一条 SASS 指令上，精确找出热点。

## 一个完整的分析示例

以[矩阵转置](../kernels/transpose.md)为例，验证 bank 冲突的推断：

```bash
nvcc -O3 -lineinfo -arch=sm_80 transpose.cu -o transpose
ncu -k regex:transpose_smem --metrics \
    gpu__time_duration.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./transpose
```

预期看到：`transpose_smem<0>`（`[32][32]`）的共享内存读冲突数量很大，`transpose_smem<1>`（`[32][33]`）几乎为 0；后者的 DRAM 吞吐率更高、耗时更短。

再以 [GEMM](../kernels/gemm.md) 为例，对比 v2 和 v4：v2 的主要停顿原因会是 MIO Throttle / Short Scoreboard（共享内存访问过于密集），Compute 利用率不高；v4 的 FMA 管线利用率明显上升。这就是"瓶颈从共享内存转移到了计算"这句话的数据依据。

面试时能讲出这样的"推断 → 用指标验证 → 优化 → 再验证"的过程，比罗列优化技巧有说服力得多。

!!! interview "面试怎么答"
    被问"一个 kernel 慢，你怎么分析"：先用 Nsight Systems 看全局时间线（kernel 之间的空隙说明 CPU 提交跟不上，用 CUDA Graph 和融合解决；NVTX 给代码段打标记），再用 Nsight Compute 看最耗时的 kernel：从 Speed Of Light 的访存、计算两个百分比判断瓶颈类型；用扇区数 / 请求数判断是否合并访问，用 bank conflict 指标验证共享内存冲突，用 warp 停顿原因定位在等什么（Long Scoreboard 等全局内存、MIO Throttle 是共享内存指令排队）。编译加 `-lineinfo`，在 Source 视图里把指标对到代码行。

!!! info "相关章节"
    - [Profiling 推理引擎](serving://perf/profiling/)（推理系统：整条服务链路怎么 profile）
    - [Linux 性能分析工具](cs://os/perf-tools/)（计算机基础：CPU 侧的 perf、火焰图）

## 练习

**1. 分析 reduction。** 用 ncu 分析[归约](../kernels/reduction.md)的 v2 和 v5，对比它们的 DRAM 吞吐率、实际占用率和主要停顿原因，解释 v5 更快的原因。

??? success "参考思路"
    预期 v2 的 DRAM 吞吐率明显低于 v5；v2 的主要停顿在 Barrier（每轮都要 `__syncthreads()`）和 Long Scoreboard（每个线程只有一个在途请求）；v5 的在途请求多（float4、每线程多个元素），DRAM 吞吐率接近峰值，停顿主要是 Long Scoreboard，但此时显存已经接近满载，这是访存瓶颈 kernel 的理想状态。

**2. 读 nsys 时间线。** 一个推理服务的 decode 步骤里，nsys 显示每步有 300 多个 kernel，每个只运行 5-20 µs，kernel 之间平均间隔 4 µs，GPU 利用率只有 60%。可能的原因和优化方向是什么？

??? success "参考答案"
    间隔主要来自 kernel 启动开销和 CPU 端（Python、框架调度）的发射速度：每步 300 个 kernel × 4 µs ≈ 1.2 ms 的空闲。优化方向：（1）**CUDA Graphs**：把整个 decode 步骤捕获成图，一次启动，vLLM、SGLang 都默认对 decode 使用 CUDA Graphs；（2）**算子融合**：把逐元素操作、归一化、残差加法等融合，减少 kernel 数量，`torch.compile` 可以自动完成一部分；（3）让 CPU 端的调度与 GPU 执行重叠（比如 SGLang 的 overlap scheduler 在 GPU 计算当前批次时，CPU 已经准备下一批）。

## 小结

- [x] 先用 nsys 看全局时间线，再用 ncu 分析最耗时的 kernel。
- [x] NVTX 给代码段打标记；kernel 之间的空隙用 CUDA Graphs 和融合解决。
- [x] ncu 从 Speed Of Light 开始：访存瓶颈、计算瓶颈、延迟瓶颈对应不同的优化方向。
- [x] 用扇区数/请求数判断合并访问，用 bank conflict 指标验证共享内存冲突，用停顿原因定位等待的对象。
- [x] 编译加 `-lineinfo`，在 Source 视图里把指标对应到代码行。
