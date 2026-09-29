# 矩阵转置

<p class="lead">矩阵转置没有任何计算，却是理解访存优化最好的例子：读和写不可能同时按行连续，朴素实现必然有一侧不合并。解决办法是借共享内存"中转"，这又会引出 bank 冲突。转置的这套思路在 GEMM、attention 里反复出现：在共享内存里改变数据布局，让全局内存的读写都合并。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 朴素转置 `out[x][y] = in[y][x]` 中，读和写哪一侧是合并的？
    2. 用共享内存中转后，为什么读写都能合并？
    3. 共享内存 tile 为什么要声明成 `[32][33]`？
    4. 转置 kernel 的性能上限是多少？用什么作为参照？
    5. 为什么常用 32×8 的 block 处理 32×32 的 tile？

## 朴素实现的问题

输入 `in` 是 `rows × cols` 的行主序矩阵，输出 `out` 是 `cols × rows`。朴素写法让线程 `(x, y)` 负责一个元素：

```cuda
int x = blockIdx.x * 32 + threadIdx.x;   // 列
int y = blockIdx.y * 32 + threadIdx.y;   // 行
out[x * rows + y] = in[y * cols + x];
```

同一个 warp 的线程 `threadIdx.x` 连续：读 `in[y][x]` 时地址连续，是**合并**的；写 `out[x][y]` 时，相邻线程写的地址相差 `rows` 个元素，**完全不合并**，每个线程独占一个 32 字节扇区，写带宽只剩八分之一。把读写的角色对调，只是把问题换到读这一侧。

## 共享内存中转

思路：一个 block 负责一个 32×32 的块（tile）。

1. 按行**合并地读**输入块，写进共享内存 `tile[threadIdx.y][threadIdx.x]`；
2. `__syncthreads()`；
3. 从共享内存按列读出 `tile[threadIdx.x][threadIdx.y]`，按行**合并地写**到输出块（输出块的位置是输入块坐标的对调）。

全局内存的读和写都合并了，"跨步访问"被转移到了共享内存上，而共享内存的跨步访问只会带来 bank 冲突，代价比全局内存小得多。再把 tile 声明为 `[32][33]`，按列读取的冲突也消失了（原理见[内存层次](../basics/memory.md#bank-冲突)）。

## 每个线程处理多个元素

用 32×32 = 1024 个线程处理一个 tile，每个线程只搬一个元素，线程的启动和索引计算开销占比很高，而且一个 block 1024 个线程的调度粒度很粗。常用的做法是 block 大小取 32×8，每个线程在 y 方向上循环处理 4 个元素。这就是 NVIDIA 官方博客 *An Efficient Matrix Transpose in CUDA C/C++* 中的写法。

## 完整代码

程序实现了四个版本，并用一个同样形状的**拷贝 kernel** 作为性能上限：转置的数据量和拷贝相同，理想情况下应该和拷贝一样快。

```cuda title="transpose.cu"
// transpose.cu —— 矩阵转置：朴素 → 共享内存 → 消除 bank 冲突，并与拷贝对比
// 编译：nvcc -O3 -arch=sm_75 transpose.cu -o transpose
#include "common.cuh"

constexpr int TILE = 32;
constexpr int ROWS_PER_ITER = 8;   // block 为 32 x 8，每个线程处理 TILE / 8 = 4 个元素

// 上限参照：同样的访问形状，只是不转置
__global__ void copy_tile(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) out[(y + j) * cols + x] = in[(y + j) * cols + x];
}

// 朴素：读合并，写不合并
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) out[x * rows + (y + j)] = in[(y + j) * cols + x];
}

// 共享内存中转，tile 宽 32：按列读共享内存时有 32 路 bank 冲突
// 共享内存中转，tile 宽 33：填充一列消除冲突
template <int PAD>
__global__ void transpose_smem(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + PAD];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
  __syncthreads();

  // 输出块的位置：块坐标对调
  x = blockIdx.y * TILE + threadIdx.x;   // 输出的列 = 输入的行
  y = blockIdx.x * TILE + threadIdx.y;   // 输出的行 = 输入的列
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < rows && y + j < cols) out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
  const int rows = 4096 + 17, cols = 8192 + 5;   // 故意取非 32 的倍数，检验边界处理
  const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
  std::vector<float> h(n), ref(n), got(n);
  fill_random(h, 9);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) ref[static_cast<size_t>(c) * rows + r] = h[static_cast<size_t>(r) * cols + c];

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, ROWS_PER_ITER);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  bool ok = true;

  auto bench = [&](const char* name, auto kernel, bool check) {
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    std::printf("%-22s ", name);
    if (check) {
      CUDA_CHECK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
      ok &= check_close(got.data(), ref.data(), n, 0.f, 0.f);
      std::printf("%-22s ", "");
    }
    float ms = time_ms([&] { kernel<<<grid, block>>>(d_in, d_out, rows, cols); });
    std::printf("%.3f ms, %.1f GB/s\n", ms, gbps(2.0 * bytes, ms));
  };

  bench("copy (upper bound)", copy_tile, false);
  bench("naive", transpose_naive, true);
  bench("smem [32][32]", transpose_smem<0>, true);
  bench("smem [32][33]", transpose_smem<1>, true);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

预期的规律：朴素版本明显最慢；共享内存版本大幅提升；消除 bank 冲突后再提升一截，接近拷贝的速度。如果你的 GPU 上 `[32][32]` 与 `[32][33]` 差距不大，可以用 Nsight Compute 查看共享内存的冲突次数，确认冲突确实被消除了。

## 讨论

- **边界处理**：读和写两个阶段都要检查边界，而且两个阶段检查的维度是对调的，这是最容易写错的地方。上面的测试故意用了不能被 32 整除的尺寸。
- **向量化**：可以让每个线程读写 `float2`/`float4` 进一步减少指令数，但共享内存的布局要相应调整，否则会引入新的 bank 冲突。
- **批量转置与维度置换**：深度学习里更常见的是高维张量的维度置换（permute），比如把 `[B, S, H, D]` 变成 `[B, H, S, D]`。核心思路相同：让最内层维度的读写都连续，必要时经过共享内存改变布局。很多时候更好的做法是**根本不做转置**，而是让下一个 kernel 直接按新布局读取（比如 GEMM 支持 `A^T` 输入），或者把转置融合进前后的算子。
- **Hopper 上的 TMA** 可以在全局内存和共享内存之间搬运整块数据，并在搬运时完成一些布局变换，见 [Hopper](../advanced/async-hopper.md)。

!!! interview "面试怎么答"
    转置题考的是合并访问：朴素写法读和写总有一侧不连续；用共享内存中转后两侧都按行连续，按列读共享内存的 bank 冲突用 `[32][33]` 填充消除；常用 32×8 的 block、每个线程处理 4 个元素；性能上限用同样大小的拷贝 kernel 做参照。最后补一句工程上的判断：能不转置就不转置——把转置融合进前后的算子，或者让消费方直接按新布局读取。

## 练习

**1. 原地转置方阵。** 对 N×N 方阵原地转置，不使用额外的全局内存。提示：只有对角线上方的 tile 需要处理，每个 block 同时加载 `(i, j)` 和 `(j, i)` 两个 tile，交换后写回；对角线上的 tile 单独处理。

??? success "参考思路"
    - grid 只覆盖上三角（含对角线）的 tile，可以用一维 grid 并把线性编号映射到 `(i, j)`，其中 `i ≤ j`；
    - 非对角 tile：把 `(i, j)` 读进 `tileA`，`(j, i)` 读进 `tileB`，同步后把 `tileB` 转置写到 `(i, j)`，`tileA` 转置写到 `(j, i)`；
    - 对角 tile：读进一个 tile，转置后写回原位；
    - 两个 tile 都要做 `[32][33]` 填充。因为每个 block 在写回之前已经把两个 tile 都读进了共享内存，不同 block 处理的 tile 互不重叠，所以没有竞争。

**2. 分析题。** 如果 tile 的元素是 `double`（8 字节），`[32][33]` 的填充还有效吗？

??? success "参考答案"
    不完全有效。每行 33 个 double 是 66 个字，按列读取时第 k 行的元素位于字 `66k`，bank 为 `66k % 32 = 2k % 32`，而 64 位访问按半个 warp 分阶段处理：每个阶段 16 个线程访问的 bank 是 0、2、4……30（每个 double 占相邻两个 bank），正好错开，没有冲突。所以对 double 来说 `[32][33]` 仍然可以消除冲突，但原因和 float 不同。更稳妥的办法是用 Nsight Compute 验证，或者用 swizzle 方式计算下标。

## 小结

- [x] 转置的读和写不可能同时按行连续；用共享内存中转，让全局内存两侧都合并。
- [x] 按列读共享内存会有 bank 冲突，用 `[32][33]` 填充消除。
- [x] 用同形状的拷贝 kernel 作为性能上限。
- [x] 32×8 的 block、每线程 4 个元素，是转置的常用配置。
- [x] 能不转置就不转置：融合进前后算子，或者让消费方直接按新布局读取。
