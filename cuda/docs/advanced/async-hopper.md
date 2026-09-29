# 异步拷贝与 Hopper/Blackwell 新特性

<p class="lead">从 Ampere 开始，NVIDIA 每一代 GPU 都在做同一件事：让数据搬运和计算彻底并行起来。Ampere 有 cp.async，Hopper 有 TMA、线程块集群和 wgmma，Blackwell 又加入了 Tensor Memory。这一章讲清楚每个特性解决什么问题、怎么用，并给出可以编译运行的例子。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `cp.async` 和普通的"全局内存 → 寄存器 → 共享内存"拷贝相比，好在哪里？
    2. 多级流水里 `wait_prior(STAGES - 2)` 的含义是什么？为什么每轮都要 commit，哪怕没发出拷贝？
    3. TMA 需要在主机端准备什么？kernel 里由几个线程发起拷贝？
    4. 线程块集群（cluster）给了 block 之间什么新能力？
    5. 什么是 warp 专门化（warp specialization）？为什么 Hopper 上的 GEMM 和 FlashAttention-3 都用它？

## 为什么需要异步拷贝

回顾 [GEMM](../kernels/gemm.md) 的主循环：每一轮先把全局内存的数据读进寄存器，再写进共享内存，同步，计算，再同步。存在两个问题：

1. **搬运和计算串行**：加载时计算单元闲着，计算时访存单元闲着；
2. **数据要经过寄存器中转**：占用寄存器，而且需要额外的指令。

双缓冲可以缓解第一个问题，但数据仍然经过寄存器。Ampere 引入的 `cp.async` 同时解决了这两点：它**直接从全局内存拷贝到共享内存**，并且是**异步**的，发出之后线程可以立刻去做别的事，之后再等待完成。

## cp.async 与多级流水 <span class="arch">sm_80+</span>

PTX 指令是 `cp.async.ca.shared.global` / `cp.async.cg.shared.global`（每次 4、8 或 16 字节，`.cg` 绕过 L1，只能用于 16 字节），配合：

- `cp.async.commit_group`：把之前发出的所有 cp.async 打包成一"组"；
- `cp.async.wait_group N`：等待，直到**最多还剩 N 组**未完成。

CUDA 在 `<cuda_pipeline.h>` 里提供了对应的 C++ 函数：`__pipeline_memcpy_async`、`__pipeline_commit`、`__pipeline_wait_prior`。更高层的封装还有 `cuda::memcpy_async` 与 `cuda::pipeline`。

**多级流水（multi-stage pipeline）**：分配 S 份共享内存缓冲区（S 通常是 3 或 4），让加载始终领先计算 S-1 步：

```cuda
// 预取：先发出前 S-1 块的加载，每块一组
for (int s = 0; s < S - 1; ++s) { load_tile_async(buf = s, tile = s); __pipeline_commit(); }

for (int kt = 0; kt < num_tiles; ++kt) {
  __pipeline_wait_prior(S - 2);   // 最多还剩 S-2 组在路上 => 第 kt 块已经到了
  __syncthreads();                // 所有线程的第 kt 块都到了；并且大家都算完了第 kt-1 块
  int next = kt + S - 1;
  if (next < num_tiles) load_tile_async(buf = next % S, tile = next);   // 覆盖的是第 kt-1 块的缓冲区
  __pipeline_commit();            // 即使没有发出加载也要提交一个空组，保持"组数"的计数规律
  compute(buf = kt % S);
}
```

两个细节容易写错：

- **`wait_prior(S-2)` 的含义**：已提交的组里，最多允许最近的 S-2 组还没完成。在第 kt 轮，已经提交了 kt + S - 1 组，最多 S-2 组未完成，意味着前 kt + 1 组（第 0 到第 kt 块）都已完成；
- **空组也要提交**：最后几轮没有新的加载，如果不提交空组，`wait_prior(S-2)` 等待的就不是你以为的那一组了。

下面把 GEMM 的 v4（二维寄存器分块）改成 3 级 cp.async 流水：

```cuda title="gemm_cp_async.cu"
// gemm_cp_async.cu —— 在二维寄存器分块 SGEMM 上加入 3 级 cp.async 流水
// 编译：nvcc -O3 -arch=sm_80 gemm_cp_async.cu -o gemm_cp_async
// 要求：M、N 是 128 的倍数，K 是 8 的倍数
#include "common.cuh"
#include <cuda_pipeline.h>

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, STAGES = 3;
constexpr int kThreads = (BM * BN) / (TM * TN);   // 256

__global__ void __launch_bounds__(kThreads)
sgemm_cp_async(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
               float beta, float* __restrict__ C) {
  __shared__ __align__(16) float As[STAGES][BM * BK];
  __shared__ __align__(16) float Bs[STAGES][BK * BN];
  const int tid = threadIdx.x;
  const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
  // 每个线程每块负责搬 A 的一个 16 字节和 B 的一个 16 字节
  const int a_row = tid / (BK / 4), a_col = (tid % (BK / 4)) * 4;   // A 块 128x8：每行 2 个 16 字节
  const int b_row = tid / (BN / 4), b_col = (tid % (BN / 4)) * 4;   // B 块 8x128：每行 32 个 16 字节
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  auto load_tile = [&](int buf, int kt) {
    const int k0 = kt * BK;
    __pipeline_memcpy_async(&As[buf][a_row * BK + a_col], &A[a_row * K + k0 + a_col], 16);
    __pipeline_memcpy_async(&Bs[buf][b_row * BN + b_col], &B[(k0 + b_row) * N + b_col], 16);
  };

  const int num_tiles = K / BK;
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_tiles) load_tile(s, s);
    __pipeline_commit();
  }

  float acc[TM][TN] = {{0.f}};
  float regM[TM], regN[TN];
  for (int kt = 0; kt < num_tiles; ++kt) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < num_tiles) load_tile(next % STAGES, next);
    __pipeline_commit();

    const float* as = As[kt % STAGES];
    const float* bs = Bs[kt % STAGES];
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = as[(threadRow * TM + i) * BK + dot];
#pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = bs[dot * BN + threadCol * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] += regM[m] * regN[n];
    }
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      float& c = C[(threadRow * TM + m) * N + threadCol * TN + n];
      c = alpha * acc[m][n] + beta * c;
    }
}

int main(int argc, char** argv) {
  require_sm(8, 0);
  const int check_n = 256;
  const int n_bench = argc > 1 ? std::atoi(argv[1]) : 4096;
  const float alpha = 1.f, beta = 0.f;
  {
    const int n = check_n;
    std::vector<float> hA(n * n), hB(n * n), ref(n * n, 0.f), got(n * n);
    fill_random(hA, 1);
    fill_random(hB, 2);
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j) ref[i * n + j] += hA[i * n + k] * hB[k * n + j];
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C, 0, n * n * sizeof(float)));
    sgemm_cp_async<<<dim3(n / BN, n / BM), kThreads>>>(n, n, n, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), C, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = check_close(got.data(), ref.data(), n * n, 1e-4f, 1e-4f);
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    if (!ok) return 1;
  }
  const int n = n_bench;
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, bytes));
  CUDA_CHECK(cudaMalloc(&B, bytes));
  CUDA_CHECK(cudaMalloc(&C, bytes));
  CUDA_CHECK(cudaMemset(A, 0, bytes));
  CUDA_CHECK(cudaMemset(B, 0, bytes));
  float ms = time_ms([&] { sgemm_cp_async<<<dim3(n / BN, n / BM), kThreads>>>(n, n, n, alpha, A, B, beta, C); });
  std::printf("cp.async 3-stage SGEMM, n=%d: %.3f ms, %.2f TFLOPS (compare with gemm v4)\n", n, ms,
              tflops(2.0 * n * n * n, ms));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
```

注意预取阶段的 `if (s < num_tiles)`：K 很小、块数少于流水级数时也要正确。和 [GEMM](../kernels/gemm.md) 里的 v4 对比性能，就能看到流水带来的收益。它在 FP32 CUDA Core 版本上的收益有限（计算部分已经很"重"），但在 Tensor Core 版本上至关重要：Tensor Core 太快，不把搬运藏起来，它大部分时间都在等数据。

## TMA：张量内存加速器 <span class="arch">sm_90+</span>

cp.async 仍然需要每个线程计算自己搬哪 16 个字节。Hopper 的 **TMA（Tensor Memory Accelerator）** 是一个专门的搬运单元：**一个线程**发出一条指令，就能把全局内存中一个多维张量的整块（box）搬进共享内存，地址计算、边界处理（越界填 0）、swizzle 全部由硬件完成。

使用 TMA 需要：

1. **主机端**创建一个**张量描述符**（`CUtensorMap`，128 字节），描述全局张量的维度、跨度、每次搬运的块大小、swizzle 模式等，用驱动 API `cuTensorMapEncodeTiled` 生成；
2. 把描述符以 `const __grid_constant__ CUtensorMap` 的形式传给 kernel；
3. kernel 里由一个线程发出 `cp.async.bulk.tensor` 指令，完成情况通过共享内存里的 **mbarrier** 通知：mbarrier 不仅计数到达的线程，还计数**到达的字节数**（transaction count），数据全部到齐时屏障才会翻转。

下面的例子用 TMA 把一个整数矩阵按 16×32 的块读入共享内存，每个元素加 1，再用 TMA 写回：

```cuda title="tma_copy.cu"
// tma_copy.cu —— 用 TMA 读写二维块：一个线程发起拷贝，mbarrier 按字节数等待完成
// 编译：nvcc -O3 -arch=sm_90 tma_copy.cu -o tma_copy -lcuda
#include "common.cuh"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int BOX_H = 16, BOX_W = 32;   // 每块 16 行 x 32 列的 int，一行 128 字节

__global__ void add_one_tma(const __grid_constant__ CUtensorMap tensor_map) {
  __shared__ alignas(128) int tile[BOX_H][BOX_W];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  const int32_t coords[2] = {static_cast<int32_t>(blockIdx.x * BOX_W),    // 最内层维度（列）在前
                             static_cast<int32_t>(blockIdx.y * BOX_H)};

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    ptx::fence_proxy_async(ptx::space_shared);   // 让 TMA（异步代理）看到初始化后的 barrier
  }
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, &tile, &tensor_map, coords,
                              cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile));   // 声明还要等 sizeof(tile) 个字节
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token));

  for (int i = threadIdx.x; i < BOX_H * BOX_W; i += blockDim.x) tile[i / BOX_W][i % BOX_W] += 1;

  ptx::fence_proxy_async(ptx::space_shared);     // 让 TMA 看到普通线程对共享内存的写入
  __syncthreads();
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, &tensor_map, coords, &tile);
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());   // 等 TMA 读完共享内存再退出
    (&bar)->~barrier();
  }
}

int main() {
  require_sm(9, 0);
  const int rows = 1024, cols = 2048;   // 行跨度 cols * 4 字节必须是 16 的倍数
  std::vector<int> h(static_cast<size_t>(rows) * cols), got(h.size());
  for (size_t i = 0; i < h.size(); ++i) h[i] = static_cast<int>(i % 1000);
  int* d;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));

  CUtensorMap map{};
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};   // 最内层维度在前
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * sizeof(int)};                  // 第 1 维的跨度（字节）
  cuuint32_t box[2] = {BOX_W, BOX_H};
  cuuint32_t elem_strides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, d, dims, strides, box, elem_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::printf("cuTensorMapEncodeTiled failed: %d\n", static_cast<int>(r));
    return 1;
  }
  add_one_tma<<<dim3(cols / BOX_W, rows / BOX_H), 128>>>(map);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d, h.size() * sizeof(int), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < h.size(); ++i) bad += got[i] != h[i] + 1;
  std::printf("TMA add-one: %s (%zu mismatches)\n", bad ? "FAIL" : "PASS", bad);
  CUDA_CHECK(cudaFree(d));
  return bad ? 1 : 0;
}
```

这里的两个 `fence_proxy_async` 很关键：TMA 属于"异步代理"，普通线程对共享内存的写入默认对它不可见，反之亦然，需要显式的代理栅栏。这类细节正是直接使用 TMA 容易出错的地方，所以工程中通常通过 CUTLASS/CuTe 的封装来使用。

在真实的 GEMM 里，TMA 的描述符会配置 128 字节 swizzle，搬进共享内存的数据正好是 wgmma 需要的布局，并配合多个 mbarrier 组成多级流水。

## 线程块集群与分布式共享内存 <span class="arch">sm_90+</span>

Hopper 在 grid 和 block 之间加了一层：**线程块集群（thread block cluster）**。同一个集群的 block（最多 8 个，部分 GPU 支持 16 个）保证被**同时调度**到同一个 GPC（GPU 处理集群）的多个 SM 上，因此它们可以：

- 用 `cluster.sync()` 在集群范围内同步；
- **直接读写彼此的共享内存**，这叫**分布式共享内存（DSMEM）**。

这打破了"block 之间不能直接通信"的限制。典型用途：几个 block 协作处理一个比单个 SM 共享内存还大的数据块；TMA 的**多播（multicast）**把同一块数据一次送进集群内多个 SM 的共享内存，减少重复的 L2 读取（GEMM 中相邻的 block 需要同一块 A 或 B）。

```cuda title="cluster_sum.cu"
// cluster_sum.cu —— 两个 block 组成一个集群，block 0 通过 DSMEM 读取 block 1 的共享内存
// 编译：nvcc -O3 -arch=sm_90 cluster_sum.cu -o cluster_sum
#include "common.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

constexpr int kThreads = 256;

__global__ void __cluster_dims__(2, 1, 1) pair_sum(const float* __restrict__ in, float* __restrict__ out, int per_block) {
  __shared__ float partial[kThreads];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();   // 本 block 在集群中的编号：0 或 1

  // 每个 block 先求自己那一段的和
  const float* src = in + static_cast<size_t>(blockIdx.x) * per_block;
  float v = 0.f;
  for (int i = threadIdx.x; i < per_block; i += blockDim.x) v += src[i];
  partial[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }

  cluster.sync();   // 两个 block 的 partial[0] 都已就绪，并且对集群内可见
  if (rank == 0 && threadIdx.x == 0) {
    float* remote = cluster.map_shared_rank(&partial[0], 1);   // 映射到 block 1 的共享内存
    out[blockIdx.x / 2] = partial[0] + *remote;
  }
  cluster.sync();   // block 1 必须等 block 0 读完才能退出，否则它的共享内存会被释放
}

int main() {
  require_sm(9, 0);
  const int clusters = 64, per_block = 10000;
  const int blocks = clusters * 2;
  std::vector<float> h(static_cast<size_t>(blocks) * per_block), ref(clusters), got(clusters);
  fill_random(h, 5);
  for (int c = 0; c < clusters; ++c) {
    double s = 0;
    for (int i = 0; i < 2 * per_block; ++i) s += h[static_cast<size_t>(c) * 2 * per_block + i];
    ref[c] = static_cast<float>(s);
  }
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, clusters * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  pair_sum<<<blocks, kThreads>>>(d_in, d_out, per_block);   // grid 大小必须是集群大小的整数倍
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, clusters * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), clusters, 1e-4f, 1e-3f);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

集群大小也可以在启动时通过 `cudaLaunchKernelEx` 的属性 `cudaLaunchAttributeClusterDimension` 动态指定。

## wgmma 与 warp 专门化 <span class="arch">sm_90a</span>

Hopper 的 Tensor Core 指令 `wgmma.mma_async` 由一个 **warpgroup**（4 个 warp）共同发出，计算 m64nNk16 的矩阵乘，B 操作数（以及可选的 A 操作数）直接从共享内存读取，并且是异步的。它和 TMA 结合，形成了 Hopper 上高性能 kernel 的标准结构：**warp 专门化（warp specialization）**。

```text
一个 block（例如 3 个 warpgroup）
├── 生产者 warpgroup（只用很少的寄存器）
│     循环：等待"缓冲区 i 空闲"的 mbarrier → 发出 TMA 加载到缓冲区 i
└── 消费者 warpgroup × 2（分到大部分寄存器）
      循环：等待"缓冲区 i 已填满"的 mbarrier → 发出 wgmma → 计算完成后通知"缓冲区 i 空闲"
```

生产者和消费者通过共享内存里的一组 mbarrier 构成**环形缓冲区**，TMA 和 Tensor Core 持续并行工作。Hopper 还提供了 `setmaxnreg` 指令，让生产者把寄存器"让给"消费者。FlashAttention-3 在此基础上还让两个消费者 warpgroup 交替执行 GEMM 和 softmax（ping-pong 调度），把 softmax 的开销也藏在了 GEMM 后面，在 H100 上 FP16 前向达到约 740 TFLOPS（约 75% 利用率，数据来自 FlashAttention-3 论文）。

直接手写 wgmma 需要自己构造共享内存矩阵描述符、处理 swizzle 和寄存器布局，代码量很大。学习路径建议：

1. 读 CUTLASS 的 Hopper GEMM 示例（`examples/48_hopper_warp_specialized_gemm` 等）和 CuTe 教程；
2. 读 DeepSeek 开源的 **DeepGEMM**，它用相对精简的代码实现了 Hopper 上的 FP8 GEMM；
3. 读 ThunderKittens 等以教学友好为目标的库；
4. Colfax Research 发表过一系列讲解 wgmma、TMA、warp 专门化的教程，可读性很好。

## Blackwell 概览 <span class="arch">sm_100</span>

Blackwell（B200、GB200、B300）的 Tensor Core 编程模型又有一次较大的变化：

- **`tcgen05.mma`**：由**单个线程**发起，不再需要整个 warp 或 warpgroup 参与；
- **Tensor Memory（TMEM）**：每个 SM 新增 256 KB 的专用存储，矩阵乘的累加结果放在这里，而不是寄存器里，大幅缓解了寄存器压力；需要显式地分配、释放以及在 TMEM 与寄存器之间搬运；
- **2-SM MMA**：一对 SM（CTA pair）可以协作完成一个更大的矩阵乘；
- **更低精度与块缩放**：原生支持 FP6、FP4，以及带块缩放因子的格式，比如 MXFP8/MXFP4（每 32 个元素共享一个 8 位指数缩放因子）和 NVFP4（每 16 个元素共享一个 FP8 缩放因子，再加一个张量级 FP32 缩放）。

RTX 50 系列（sm_120）虽然也叫 Blackwell，但其 Tensor Core 编程模型与数据中心版不同，不支持 tcgen05。学习 Blackwell 最好的材料是 CUTLASS 4.x 的示例和 CuTe DSL（Python 接口）。

!!! interview "面试怎么答"
    被问 Hopper 上的高性能 kernel 怎么写：`cp.async` 让全局内存直接异步拷进共享内存，多级流水让加载领先计算几步；TMA 由一个线程发起整块搬运，需要主机端的张量描述符和按字节计数的 mbarrier；线程块集群让 block 之间能同步、访问彼此的共享内存（DSMEM）并支持 TMA 多播；warp 专门化让一部分 warp 专门搬数据、一部分专门做 wgmma，靠 mbarrier 环形缓冲区交接——FlashAttention-3 和 Hopper 上的 GEMM 都是这个结构。再提 Blackwell 的 tcgen05、Tensor Memory 和块缩放的低精度格式。

## 练习

**1. 流水级数。** 对于 `gemm_cp_async.cu`，把 `STAGES` 改成 2、3、4，共享内存用量分别是多少？流水级数越多越好吗？

??? success "参考答案"
    每级需要 `(128 × 8 + 8 × 128) × 4 B = 8 KB`，2、3、4 级分别是 16、24、32 KB。级数越多，能藏住的访存延迟越长，但共享内存占用越多，可能导致每个 SM 能驻留的 block 数减少；而且一旦级数足以覆盖延迟，再增加就没有收益了。常见的选择是 Ampere 上 3-4 级、Hopper 上 4-8 级（Hopper 的共享内存更大，TMA 的延迟也更适合深流水），最终要靠实测和自动调优决定。

**2. 为什么 TMA 只需要一个线程发起？** 这对 kernel 的设计有什么影响？

??? success "参考答案"
    TMA 是独立的硬件单元，地址计算和数据搬运都由它完成，线程只需要发出一条指令并告诉它结果写到哪里、用哪个 mbarrier 通知。这带来两个影响：（1）负责计算的线程完全不需要参与搬运，也不再为搬运占用寄存器，于是可以把 warp 分成专门搬运的生产者和专门计算的消费者；（2）同步方式从 `__syncthreads()` 变成基于 mbarrier 的生产者-消费者协议，需要仔细管理每个缓冲区的"已满"和"空闲"两个状态，以及 mbarrier 的相位（phase）。

## 小结

- [x] cp.async 直接从全局内存异步拷贝到共享内存；多级流水让加载领先计算 S-1 步，`wait_prior(S-2)` + 每轮提交。
- [x] TMA 由一个线程发起整块搬运，需要主机端的张量描述符和按字节计数的 mbarrier，注意代理栅栏。
- [x] 线程块集群让 block 之间可以同步和访问彼此的共享内存（DSMEM），并支持 TMA 多播。
- [x] Hopper 高性能 kernel 的标准结构：TMA + wgmma + warp 专门化 + mbarrier 环形缓冲区。
- [x] Blackwell 引入单线程发起的 tcgen05、Tensor Memory 和块缩放低精度格式。
