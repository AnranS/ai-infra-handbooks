# 执行模型与性能基础

<p class="lead">知道了硬件长什么样，接下来要搞清楚它怎么执行你的代码：warp 如何被调度、分支为什么会变慢、占用率意味着什么、为什么"更多线程"不一定更快。最后用 Roofline 模型把这些串起来，得到判断一个 kernel 性能上限的方法。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `if (threadIdx.x % 2 == 0)` 和 `if ((threadIdx.x / 32) % 2 == 0)` 哪个会导致分支发散？
    2. 占用率越高性能越好吗？
    3. 为什么一个线程只发一个访存请求时，很难把显存带宽跑满？
    4. 算术强度是什么？怎么用它判断一个 kernel 是访存瓶颈还是计算瓶颈？
    5. 108 个 SM 的 GPU 上启动 109 个 block（每个 SM 只能放一个），会发生什么？

## 分支发散

warp 的 32 个线程共享一条指令流。遇到条件分支时，如果**同一个 warp 内**的线程走向不同的分支，硬件会依次执行每条路径，执行某条路径时屏蔽不走这条路的线程，最后再汇合：

```cuda
if (threadIdx.x % 2 == 0) {
  a();          // 偶数线程执行，奇数线程空等
} else {
  b();          // 奇数线程执行，偶数线程空等
}
```

这个 warp 的耗时是 `a()` 与 `b()` 之和，而不是较大者。关键在于**发散只发生在 warp 内部**。如果分支条件对同一个 warp 的所有线程取值相同，就没有代价：

```cuda
if ((threadIdx.x / 32) % 2 == 0) { a(); } else { b(); }   // 按 warp 分组，没有发散
```

实用的判断方法：

- 条件只依赖 `blockIdx`、kernel 参数、`threadIdx.x / 32` 这类 warp 内一致的量：不发散；
- 边界检查 `if (i < n)`：只有最后一个 warp 可能发散，可以忽略；
- 依赖数据内容的分支（比如 `if (x[i] > 0)`）：可能严重发散，考虑改写成无分支的形式，如 `y = max(x, 0.f)`，或者用 `fmaxf`、条件赋值让编译器生成选择指令。

下面的程序对比了两种分支方式：

```cuda title="divergence.cu"
// divergence.cu —— warp 内分支发散的代价
// 编译：nvcc -O3 -arch=sm_75 divergence.cu -o divergence
#include "common.cuh"

__device__ __forceinline__ float heavy(float x, int iters, float c) {
  for (int k = 0; k < iters; ++k) x = x * c + 0.5f;
  return x;
}

// 同一个 warp 内奇偶线程走不同分支
__global__ void divergent(float* out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = static_cast<float>(threadIdx.x);
  if (threadIdx.x % 2 == 0) x = heavy(x, iters, 0.999f);
  else                      x = heavy(x, iters, 0.998f);
  out[i] = x;
}

// 分支条件以 warp 为单位，同一个 warp 走同一条路
__global__ void uniform(float* out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = static_cast<float>(threadIdx.x);
  if ((threadIdx.x / 32) % 2 == 0) x = heavy(x, iters, 0.999f);
  else                             x = heavy(x, iters, 0.998f);
  out[i] = x;
}

int main() {
  const int threads = 256, blocks = sm_count() * 8, iters = 2000;
  const int n = threads * blocks;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  // 正确性：两个 kernel 每个线程做的计算完全相同，只是分组方式不同
  std::vector<float> h(n), ref(n);
  uniform<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < n; ++t) {
    int tid = t % threads;
    float x = static_cast<float>(tid), c = ((tid / 32) % 2 == 0) ? 0.999f : 0.998f;
    for (int k = 0; k < iters; ++k) x = x * c + 0.5f;
    ref[t] = x;
  }
  bool ok = check_close(h.data(), ref.data(), n, 1e-3f, 1e-3f);

  float t_div = time_ms([&] { divergent<<<blocks, threads>>>(d_out, iters); });
  float t_uni = time_ms([&] { uniform<<<blocks, threads>>>(d_out, iters); });
  std::printf("divergent: %.3f ms\nuniform  : %.3f ms\nratio    : %.2fx\n", t_div, t_uni, t_div / t_uni);
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

预期 `divergent` 大约是 `uniform` 的两倍耗时。

## 延迟掩盖

一条指令发出后，结果要过一段时间才能用：算术指令大约 4-6 个周期，访问共享内存几十个周期，访问显存几百个周期。GPU 不会像 CPU 那样乱序执行同一个线程的后续指令，而是**切换到另一个 warp**。要让计算单元一直忙碌，需要足够多"就绪"的工作。这些工作有两个来源：

- **线程级并行（TLP）**：SM 上驻留更多的 warp；
- **指令级并行（ILP）**：同一个线程里有多条互不依赖的指令，可以连续发射，不必等前一条的结果。

一个有用的估算是 **Little 定律**：为了维持带宽 B，在途（已发出、未返回）的数据量要达到 `B × 延迟`。以 A100 为例，约 2 TB/s × 约 600 ns ≈ 1.2 MB，平均到 108 个 SM，每个 SM 需要约 11 KB 的数据同时在途。如果每个线程一次只有一个 4 字节的读请求在途，每个 SM 就需要近 3000 个线程，超过了 2048 的上限。

所以**只靠提高占用率，常常跑不满带宽**。解决办法是让每个线程同时发出更多的访存请求：用 `float4` 一次读 16 字节，或者每个线程处理多个元素并把读操作提前集中发出。这也是为什么后面的高性能 kernel 几乎都是"每个线程处理多个元素"。

## 占用率（occupancy）

**占用率 = SM 上实际驻留的 warp 数 / 硬件上限**（A100、H100 每 SM 64 个 warp）。一个 block 能否驻留，取决于它对四种资源的需求：

| 资源 | A100 每 SM 上限 | 限制方式 |
| --- | --- | --- |
| 线程数 | 2048 | block 大小越大，能放的 block 越少 |
| block 数 | 32 | block 太小（比如 32 线程）时会先碰到这个上限 |
| 寄存器 | 65536 个 | 每线程寄存器 × 线程数 |
| 共享内存 | 164 KB | 每个 block 的共享内存用量 |

四者取最严格的一个。CUDA 提供 API 直接计算：

```cuda title="occupancy.cu"
// occupancy.cu —— 用 Occupancy API 计算不同配置下的占用率
// 编译：nvcc -O3 -arch=sm_75 occupancy.cu -o occupancy
#include "common.cuh"

__global__ void light_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 2.f + 1.f;
}

// 每个 block 使用 dynamic shared memory，模拟共享内存限制
__global__ void smem_kernel(float* x, int n) {
  extern __shared__ float buf[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[threadIdx.x] = (i < n) ? x[i] : 0.f;
  __syncthreads();
  if (i < n) x[i] = buf[blockDim.x - 1 - threadIdx.x];
}

template <typename K>
void report(const char* name, K kernel, int block, size_t smem) {
  int dev = 0, blocks_per_sm = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block, smem));
  float occ = static_cast<float>(blocks_per_sm * block) / p.maxThreadsPerMultiProcessor;
  std::printf("%-12s block=%4d smem=%6zu B -> %2d blocks/SM, occupancy %5.1f%%\n",
              name, block, smem, blocks_per_sm, occ * 100);
}

int main() {
  for (int block : {32, 64, 128, 256, 512, 1024}) report("light", light_kernel, block, 0);
  for (size_t smem : {0, 16 * 1024, 32 * 1024, 48 * 1024}) report("smem", smem_kernel, 256, smem);

  // 让运行时推荐一个 block 大小（以占用率最大化为目标）
  int min_grid = 0, best_block = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, light_kernel, 0, 0));
  std::printf("suggested block size for light_kernel: %d (min grid %d)\n", best_block, min_grid);
  return 0;
}
```

!!! warning "占用率不是越高越好"
    占用率只是掩盖延迟的手段之一。很多高性能 kernel（GEMM、FlashAttention）故意使用大量寄存器和共享内存，占用率只有 10%-25%，但每个线程有大量 ILP、数据复用率高，反而更快。经验是：**访存瓶颈、每线程工作量小的 kernel 需要较高占用率；计算密集、ILP 充足的 kernel 可以接受低占用率**。判断标准永远是实测性能，而不是占用率本身。

### `__launch_bounds__`

编译器不知道你会用多大的 block 启动 kernel，可能分配过多寄存器，导致想要的配置无法驻留。可以告诉它：

```cuda
__global__ void __launch_bounds__(256, 2) my_kernel(...) { ... }
// 最多用 256 线程的 block 启动，希望每个 SM 至少驻留 2 个 block
```

编译器会据此限制每个线程的寄存器用量（必要时溢出）。

## 启动配置怎么选

没有万能的答案，但有一些靠谱的起点：

- **block 大小**：取 32 的倍数，从 128 或 256 开始试。太小（32、64）容易先碰到每 SM 的 block 数上限；太大（1024）则调度粒度粗，而且 `__syncthreads()` 等待的线程更多。
- **grid 大小**：至少让每个 SM 有几个 block。数据量小、block 数不到 SM 数时，GPU 大部分是空闲的，此时应该让每个 block 处理更少的数据，或者用 split-K 等方式增加并行度（大模型 decode 阶段的很多算子正是这种情况）。
- **尾波效应（wave quantization）**：所有 SM 同时处理的一批 block 叫一个"波"。假设每个 SM 只能驻留 1 个 block，108 个 SM 上启动 109 个 block，需要两个波，第二个波只有 1 个 block 在跑，时间几乎翻倍。block 数略多于"SM 数 × 每 SM 驻留数"的整数倍时要格外注意。GEMM 库会为此调整分块大小，或者采用 stream-K 这类负载均衡策略。

## Roofline 模型

Roofline 模型用两个硬件参数给一个 kernel 的性能画出上限：

- 峰值算力 **P**（FLOP/s）
- 峰值带宽 **B**（Byte/s）

以及 kernel 的一个属性：

- **算术强度** I = 总计算量（FLOP）/ 总访存量（Byte，指与显存之间传输的数据量）

可达性能的上限为：

$$
\text{Performance} \le \min(P,\ I \times B)
$$

两条线的交点 $I^* = P / B$ 叫**脊点**：

- $I < I^*$：**访存瓶颈（memory-bound）**。优化方向是减少访存量（融合、复用、量化）或者提高带宽利用率（合并访问、向量化、增加在途请求）。
- $I > I^*$：**计算瓶颈（compute-bound）**。优化方向是用更快的计算单元（Tensor Core）、减少多余计算、提高指令效率。

几个典型算子（FP32，A100 的脊点约 9.6 FLOP/Byte；若使用 BF16 Tensor Core，脊点约 153）：

| 算子 | 算术强度 | 结论 |
| --- | --- | --- |
| 向量加法 `c = a + b` | 1 FLOP / 12 B ≈ 0.08 | 严重访存瓶颈 |
| Softmax、LayerNorm | 每元素几次到十几次运算 / 8 B 左右 | 访存瓶颈 |
| 矩阵向量乘 GEMV（M×K 矩阵） | 2MK / 4MK ≈ 0.5 | 访存瓶颈（大模型 decode 的主要开销） |
| 方阵乘法 GEMM（N×N） | 2N³ / 12N² = N/6 | N 大时计算瓶颈 |

这张表解释了大模型推理的很多现象：**prefill 阶段以 GEMM 为主，是计算瓶颈；decode 阶段每次只生成一个 token，线性层退化成 GEMV，是访存瓶颈**。所以 decode 的优化重点是减少读权重和 KV Cache 的字节数：批处理（把多个请求的 GEMV 合成 GEMM）、量化（权重从 16 位降到 8 位或 4 位）、KV Cache 压缩等。

**用 Roofline 评估一个 kernel 的步骤**：先算出（或用 Nsight Compute 测出）它的算术强度，确定它是访存瓶颈还是计算瓶颈，然后把实测性能和对应的上限（`I × B` 或 `P`）比较。达到上限的 70%-80% 以上就已经是很好的实现了。Nsight Compute 能直接画出 Roofline 图，见[性能分析](../tools/profiling.md)。

## 指令层面的几个要点

- **FMA**：`a * b + c` 会被编译成一条融合乘加指令（FFMA），算作 2 次浮点运算。峰值算力就是按 FMA 统计的。
- **快速数学函数**：`__expf`、`__logf`、`__sinf`、`__fdividef` 由特殊函数单元（SFU）计算，比标准的 `expf` 等快得多，但精度略低。编译选项 `--use_fast_math` 会全局替换，同时开启非规格化数清零等行为，用在训练和推理 kernel 里通常没问题，但要验证精度。
- **整数除法和取模**很慢；除数是 2 的幂时用移位和按位与，或者让编译器看到编译期常量。
- **双精度**：消费级 GPU 的 FP64 算力只有 FP32 的 1/64，数据中心卡（A100、H100）是 1/2。AI 场景基本不用 FP64。

## 练习

**1. 占用率计算。** H100 上（每 SM 2048 线程、32 个 block、65536 个寄存器、228 KB 共享内存），一个 kernel 的 block 有 128 个线程，每线程 168 个寄存器，每个 block 用 64 KB 共享内存。每个 SM 能驻留几个 block？瓶颈资源是什么？

??? success "参考答案"
    - 线程：2048 / 128 = 16 个 block
    - block 数：32 个
    - 寄存器：每个 block 128 × 168 = 21504 个，65536 / 21504 ≈ 3.05，即 3 个 block
    - 共享内存：228 / 64 ≈ 3.56，即 3 个 block（实际还要扣除每个 block 的少量保留空间）

    最多 3 个 block，12 个 warp，占用率 18.75%。寄存器和共享内存都是瓶颈。这是高性能 GEMM/attention kernel 常见的配置，靠 ILP 和数据复用取得性能。

**2. 判断瓶颈。** RMSNorm：输入 `[tokens, hidden]` 的 BF16 张量，每个元素要读一次、写一次，另外读一次长度为 hidden 的权重（可以忽略），每个元素约 4 次浮点运算。在 H100（3.35 TB/s，FP32 非 Tensor Core 算力 67 TFLOPS）上它是什么瓶颈？`tokens = 4096, hidden = 8192` 时理论最短耗时是多少？

??? success "参考答案"
    每个元素读写共 4 字节，约 4 次运算，算术强度约 1 FLOP/Byte，远低于脊点 67 / 3.35 = 20，是**访存瓶颈**。

    数据量 4096 × 8192 × 4 B ≈ 134 MB，理论最短时间 134 MB / 3.35 TB/s ≈ 40 µs。实现得好的 kernel 能达到峰值带宽的 80%-90%，约 45-50 µs。如果它和前一个算子（比如残差加法）融合，就能省掉一次完整的读写，这正是算子融合的价值。

**3. 尾波效应。** 一个 kernel 在 A100 上每个 SM 只能驻留 2 个 block，你启动了 220 个 block，每个 block 运行时间相同。和启动 216 个 block 相比，耗时大约是多少倍？怎么改进？

??? success "参考答案"
    一波能容纳 108 × 2 = 216 个 block。216 个 block 正好一波；220 个需要两波，第二波只有 4 个 block，耗时约为 **2 倍**。

    改进方法：调整每个 block 处理的数据量，使 block 数正好是 216 的整数倍，或者远大于它（波数越多，最后一个不满的波占比越小）；也可以用持久化 kernel（persistent kernel）：只启动 216 个 block，每个 block 循环领取任务。

## 小结

- [x] 分支发散只在 warp 内部发生；让分支条件在 warp 内一致。
- [x] 延迟靠 TLP（更多 warp）和 ILP（每线程更多独立操作）掩盖；带宽瓶颈的 kernel 常常需要每线程多个在途请求。
- [x] 占用率受线程数、block 数、寄存器、共享内存共同限制；它是手段不是目标。
- [x] 注意尾波效应和"block 数不够填满 GPU"的问题。
- [x] Roofline：算术强度决定上限，先判断瓶颈类型，再选择优化方向。
