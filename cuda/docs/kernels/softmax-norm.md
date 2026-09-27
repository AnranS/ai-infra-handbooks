# Softmax 与归一化

<p class="lead">Softmax、LayerNorm、RMSNorm 是 Transformer 里调用最频繁的"行归约"类算子：对矩阵的每一行做一次归约，再用归约结果逐元素变换。它们都是访存瓶颈，优化的核心是"每个元素只读一次、写一次"，以及和前后算子融合。online softmax 更是 FlashAttention 的基础。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么 softmax 要先减去最大值？
    2. online softmax 怎么在一次遍历中同时求出最大值和指数和？
    3. 一行只有 128 个元素时，用一个 block 处理一行合适吗？
    4. LayerNorm 和 RMSNorm 的区别是什么？大模型为什么多用 RMSNorm？
    5. vLLM 里的 `fused_add_rms_norm` 融合了哪些操作？省下了多少访存？

## Softmax 与数值稳定性

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

直接计算 $e^{x_i}$ 很容易溢出：FP32 下 $e^{89}$ 就超出了表示范围，而注意力分数、logits 经常有这么大。所以实际计算时先减去这一行的最大值 $m = \max_j x_j$，结果在数学上不变，但所有指数都小于等于 1：

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

这就是 **safe softmax**。朴素的实现需要遍历三次：求最大值、求指数和、计算输出。

## Online softmax：一次遍历求出最大值和指数和

能不能一边找最大值，一边累加指数和？可以。维护当前的最大值 $m$ 和"以 $m$ 为基准"的指数和 $d$，每读到一个新元素 $x$：

$$
m' = \max(m, x),\qquad d' = d \cdot e^{m - m'} + e^{x - m'}
$$

最大值变大时，把之前累加的和按比例缩小（乘以 $e^{m - m'}$）即可。这个算法来自 Milakov 和 Gimelshein 2018 年的论文 *Online normalizer calculation for softmax*。

更重要的是，**两组部分结果也能合并**：

$$
m = \max(m_1, m_2),\qquad d = d_1 e^{m_1 - m} + d_2 e^{m_2 - m}
$$

这意味着 $(m, d)$ 可以像求和一样做**并行归约**：每个线程先处理自己的元素，再用 warp shuffle、共享内存逐级合并。softmax 于是从三次遍历变成两次：一次求出 $(m, d)$，一次写输出。FlashAttention 正是把这个合并规则用在了分块计算的注意力上，见 [FlashAttention](../advanced/attention.md)。

## 按行长度选择并行方式

| 行长度 | 方式 | 说明 |
| --- | --- | --- |
| 很短（≤ 32） | 一个 warp 处理多行 | 用 `tiled_partition<8>` 之类的小分组 |
| 中等（≤ 1024 左右） | **一个 warp 处理一行** | 整行放进寄存器，只读一次显存，归约全在 warp 内完成，不需要共享内存和 `__syncthreads()` |
| 较长（几千到几万） | 一个 block 处理一行 | 大模型的词表 logits（3 万到 15 万）属于这种 |
| 极长、行数很少 | 多个 block 处理一行 | 先各自求部分 $(m, d)$，再合并 |

一行一个 block 在行很短时非常浪费：128 个元素分给 256 个线程，一半线程没事做，还要承担 block 级同步的开销。反过来，一行一个 warp 在行很长时，每个线程要在寄存器里存几百个元素，寄存器会溢出。OneFlow、PyTorch 等框架的 softmax 实现都会根据列数在几种 kernel 之间切换。

## 实现

```cuda title="softmax.cu"
// softmax.cu —— 行 softmax 的三种实现：block 三遍、warp 一行（寄存器缓存）、block online 两遍
// 编译：nvcc -O3 -arch=sm_75 softmax.cu -o softmax
#include "common.cuh"

__device__ __forceinline__ float warp_max(float v) {
  for (int m = 16; m > 0; m /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}

// block 内归约，所有线程都拿到结果。op: 0 = sum, 1 = max
template <int OP>
__device__ float block_allreduce(float v) {
  __shared__ float buf[32];
  __shared__ float result;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  v = OP ? warp_max(v) : warp_sum(v);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < nwarps ? buf[lane] : (OP ? -INFINITY : 0.f);
    v = OP ? warp_max(v) : warp_sum(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  v = result;
  __syncthreads();   // 防止下一次调用覆盖 buf/result 时还有线程没读完
  return v;
}

// v1：一个 block 一行，三次遍历全局内存（max、sum、写出）
__global__ void softmax_block_3pass(const float* __restrict__ x, float* __restrict__ y, int cols) {
  const float* row = x + static_cast<size_t>(blockIdx.x) * cols;
  float* out = y + static_cast<size_t>(blockIdx.x) * cols;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, row[i]);
  m = block_allreduce<1>(m);
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) s += __expf(row[i] - m);
  s = block_allreduce<0>(s);
  const float inv = 1.f / s;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = __expf(row[i] - m) * inv;
}

// v2：一个 warp 一行，整行缓存在寄存器里，显存只读一次、写一次。要求 cols <= COLS
template <int COLS>
__global__ void softmax_warp(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
  constexpr int kPerLane = (COLS + 31) / 32;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= rows) return;   // 整个 warp 一起退出，不影响其他 warp
  const float* in = x + static_cast<size_t>(row) * cols;
  float v[kPerLane];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    int c = k * 32 + lane;                 // 相邻 lane 读相邻元素：合并访问
    v[k] = c < cols ? in[c] : -INFINITY;
    m = fmaxf(m, v[k]);
  }
  m = warp_max(m);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    v[k] = __expf(v[k] - m);               // exp(-inf) = 0，越界位置不影响求和
    s += v[k];
  }
  s = warp_sum(s);
  const float inv = 1.f / s;
  float* out = y + static_cast<size_t>(row) * cols;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    int c = k * 32 + lane;
    if (c < cols) out[c] = v[k] * inv;
  }
}

// online softmax 的 (m, d) 合并
struct MD {
  float m, d;
};
__device__ __forceinline__ MD md_combine(MD a, MD b) {
  float m = fmaxf(a.m, b.m);
  // 两边都是 -inf（还没有元素）时避免出现 inf - inf = nan
  float da = a.m == -INFINITY ? 0.f : a.d * __expf(a.m - m);
  float db = b.m == -INFINITY ? 0.f : b.d * __expf(b.m - m);
  return {m, da + db};
}
__device__ MD warp_md(MD v) {
  for (int k = 16; k > 0; k /= 2) {
    MD o{__shfl_xor_sync(0xffffffff, v.m, k), __shfl_xor_sync(0xffffffff, v.d, k)};
    v = md_combine(v, o);
  }
  return v;
}

// v3：一个 block 一行，online softmax：一次遍历求 (m, d)，一次遍历写出
__global__ void softmax_block_online(const float* __restrict__ x, float* __restrict__ y, int cols) {
  const float* row = x + static_cast<size_t>(blockIdx.x) * cols;
  float* out = y + static_cast<size_t>(blockIdx.x) * cols;
  MD md{-INFINITY, 0.f};
  for (int i = threadIdx.x; i < cols; i += blockDim.x) md = md_combine(md, MD{row[i], 1.f});
  __shared__ MD buf[32];
  __shared__ MD total;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  md = warp_md(md);
  if (lane == 0) buf[warp] = md;
  __syncthreads();
  if (warp == 0) {
    md = lane < blockDim.x / 32 ? buf[lane] : MD{-INFINITY, 0.f};
    md = warp_md(md);
    if (lane == 0) total = md;
  }
  __syncthreads();
  const float m = total.m, inv = 1.f / total.d;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = __expf(row[i] - m) * inv;
}

void softmax_cpu(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* in = x + static_cast<size_t>(r) * cols;
    float* out = y + static_cast<size_t>(r) * cols;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = std::max(m, in[c]);
    double s = 0;
    for (int c = 0; c < cols; ++c) s += std::exp(double(in[c]) - m);
    for (int c = 0; c < cols; ++c) out[c] = static_cast<float>(std::exp(double(in[c]) - m) / s);
  }
}

int main() {
  bool ok = true;
  // 两种形状：注意力分数（行短、行数多）和词表 logits（行长、行数少）
  struct Shape { int rows, cols; } shapes[] = {{4096, 1000}, {64, 50000}};
  for (const Shape& sh : shapes) {
    const int rows = sh.rows, cols = sh.cols;
    const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
    std::vector<float> hx(n), ref(n), got(n);
    fill_random(hx, rows, -20.f, 20.f);
    hx[5] = 90.f;   // 一个很大的值：没有减最大值的实现会溢出
    softmax_cpu(hx.data(), ref.data(), rows, cols);
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    std::printf("rows=%d cols=%d\n", rows, cols);

    auto run = [&](const char* name, auto launch) {
      CUDA_CHECK(cudaMemset(dy, 0, bytes));
      launch();
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost));
      std::printf("  %-22s ", name);
      ok &= check_close(got.data(), ref.data(), n, 1e-3f, 1e-6f);
      float ms = time_ms(launch);
      std::printf("  %-22s %.3f ms, %.1f GB/s (counting one read + one write)\n", "", ms, gbps(2.0 * bytes, ms));
    };
    run("block, 3 passes", [&] { softmax_block_3pass<<<rows, 256>>>(dx, dy, cols); });
    run("block, online", [&] { softmax_block_online<<<rows, 256>>>(dx, dy, cols); });
    if (cols <= 1024) {
      const int warps_per_block = 4;
      run("warp per row", [&] {
        softmax_warp<1024><<<(rows + warps_per_block - 1) / warps_per_block, 32 * warps_per_block>>>(dx, dy, rows, cols);
      });
    }
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }
  return ok ? 0 : 1;
}
```

`__expf` 是快速指数函数，精度略低于 `expf`，softmax 的检查用了 `1e-3` 的相对误差。正式的库一般在 FP32 下计算、用 BF16/FP16 读写，读写都用 128 位向量化指令。

## LayerNorm 与 RMSNorm

LayerNorm 对每一行（一个 token 的隐藏向量）做标准化：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,\qquad
\mu = \frac{1}{H}\sum_i x_i,\quad \sigma^2 = \frac{1}{H}\sum_i (x_i - \mu)^2
$$

RMSNorm 去掉了减均值和偏置 $\beta$：

$$
y = \frac{x}{\sqrt{\frac{1}{H}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

它少一次归约、少一个参数，效果相当，所以 LLaMA、Qwen、DeepSeek 等大模型都用 RMSNorm。

实现结构和 softmax 一样：一个 block（或 warp）处理一行，先归约、再逐元素变换。计算方差有两种方式：

- **两遍法**：先求均值，再求 $\sum (x - \mu)^2$。数值稳定，但要么读两遍显存，要么把整行缓存在寄存器/共享内存里；
- **一遍法**：同时累加 $\sum x$ 和 $\sum x^2$，方差 $= E[x^2] - E[x]^2$。只要一次归约，但当均值远大于标准差时会有严重的抵消误差；
- **Welford 算法**：像 online softmax 一样，维护 (count, mean, M2) 并逐个更新，部分结果也能合并。数值稳定且只需一遍，PyTorch 和 Apex 的 LayerNorm 就用它。

隐藏维度 H 一般是几千（4096、8192），一个 block 处理一行、整行缓存在寄存器里是常见的做法：H = 4096、256 个线程时每个线程 16 个元素。

## 算子融合：fused_add_rms_norm

Transformer 每一层里有这样的模式：

```text
residual = residual + hidden        # 残差连接
hidden   = rms_norm(residual) * w   # 下一个子层的输入
```

分开做需要：残差加法读 2 次写 1 次，RMSNorm 读 1 次写 1 次，共 5 次整行访存。融合成一个 kernel：读 `residual` 和 `hidden` 各一次，写回新的 `residual` 和归一化结果各一次，共 4 次，并且省掉一次 kernel 启动。对于访存瓶颈的算子，**访存次数就是耗时**。vLLM、SGLang 里的 `fused_add_rms_norm` 就是这个 kernel，也是本章的练习。

```cuda title="rmsnorm.cu"
// rmsnorm.cu —— RMSNorm 与融合了残差加法的 fused_add_rms_norm
// 编译：nvcc -O3 -arch=sm_75 rmsnorm.cu -o rmsnorm
#include "common.cuh"

__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}
__device__ float block_sum(float v) {
  __shared__ float buf[32];
  __shared__ float total;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = warp_sum(v);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = warp_sum(lane < blockDim.x / 32 ? buf[lane] : 0.f);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// 一个 block 一行；每个线程把自己负责的元素缓存在寄存器里（最多 kMaxPerThread 个）
constexpr int kThreads = 256;
constexpr int kMaxPerThread = 32;   // 支持 hidden <= 8192

__global__ void rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                         int hidden, float eps) {
  const size_t off = static_cast<size_t>(blockIdx.x) * hidden;
  float v[kMaxPerThread];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    v[k] = c < hidden ? x[off + c] : 0.f;
    ss += v[k] * v[k];
  }
  const float scale = rsqrtf(block_sum(ss) / hidden + eps);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    if (c < hidden) y[off + c] = v[k] * scale * w[c];
  }
}

// residual += hidden；hidden = rms_norm(residual) * w。两个张量都原地更新
__global__ void fused_add_rms_norm(float* __restrict__ hidden, float* __restrict__ residual,
                                   const float* __restrict__ w, int n_hidden, float eps) {
  const size_t off = static_cast<size_t>(blockIdx.x) * n_hidden;
  float v[kMaxPerThread];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    if (c < n_hidden) {
      v[k] = residual[off + c] + hidden[off + c];
      residual[off + c] = v[k];            // 新的残差写回，供下一层使用
    } else {
      v[k] = 0.f;
    }
    ss += v[k] * v[k];
  }
  const float scale = rsqrtf(block_sum(ss) / n_hidden + eps);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    if (c < n_hidden) hidden[off + c] = v[k] * scale * w[c];
  }
}

int main() {
  const int tokens = 512, hidden = 4096;
  const float eps = 1e-6f;
  const size_t n = static_cast<size_t>(tokens) * hidden, bytes = n * sizeof(float);
  std::vector<float> hx(n), hr(n), hw(hidden), ref_y(n), ref_r(n), got(n);
  fill_random(hx, 1);
  fill_random(hr, 2);
  fill_random(hw, 3, 0.5f, 1.5f);

  // CPU 参考：先算 RMSNorm(x)，再算融合版本
  for (int t = 0; t < tokens; ++t) {
    double ss = 0;
    for (int c = 0; c < hidden; ++c) ss += double(hx[t * hidden + c]) * hx[t * hidden + c];
    float scale = 1.f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int c = 0; c < hidden; ++c) ref_y[t * hidden + c] = hx[t * hidden + c] * scale * hw[c];
  }

  float *dx, *dr, *dw, *dy;
  CUDA_CHECK(cudaMalloc(&dx, bytes));
  CUDA_CHECK(cudaMalloc(&dr, bytes));
  CUDA_CHECK(cudaMalloc(&dy, bytes));
  CUDA_CHECK(cudaMalloc(&dw, hidden * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dr, hr.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw.data(), hidden * sizeof(float), cudaMemcpyHostToDevice));

  rms_norm<<<tokens, kThreads>>>(dx, dw, dy, hidden, eps);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost));
  std::printf("rms_norm            ");
  bool ok = check_close(got.data(), ref_y.data(), n, 1e-4f, 1e-5f);

  // 融合版本的参考结果
  for (int t = 0; t < tokens; ++t) {
    double ss = 0;
    for (int c = 0; c < hidden; ++c) {
      float r = hr[t * hidden + c] + hx[t * hidden + c];
      ref_r[t * hidden + c] = r;
      ss += double(r) * r;
    }
    float scale = 1.f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int c = 0; c < hidden; ++c) ref_y[t * hidden + c] = ref_r[t * hidden + c] * scale * hw[c];
  }
  fused_add_rms_norm<<<tokens, kThreads>>>(dx, dr, dw, hidden, eps);   // 原地修改 dx 和 dr
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dx, bytes, cudaMemcpyDeviceToHost));
  std::printf("fused (hidden out)  ");
  ok &= check_close(got.data(), ref_y.data(), n, 1e-4f, 1e-5f);
  CUDA_CHECK(cudaMemcpy(got.data(), dr, bytes, cudaMemcpyDeviceToHost));
  std::printf("fused (residual)    ");
  ok &= check_close(got.data(), ref_r.data(), n, 1e-6f, 1e-6f);

  float t1 = time_ms([&] { rms_norm<<<tokens, kThreads>>>(dx, dw, dy, hidden, eps); });
  float t2 = time_ms([&] { fused_add_rms_norm<<<tokens, kThreads>>>(dx, dr, dw, hidden, eps); });
  std::printf("rms_norm: %.3f ms (%.1f GB/s)\nfused_add_rms_norm: %.3f ms (%.1f GB/s)\n",
              t1, gbps(2.0 * bytes, t1), t2, gbps(4.0 * bytes, t2));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dr));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dw));
  return ok ? 0 : 1;
}
```

## 练习

**1. LayerNorm。** 仿照 `rms_norm`，实现带 γ、β 的 LayerNorm，用两遍法（先求均值，再用寄存器里缓存的值求方差），并与 CPU 结果对比。

??? success "参考答案"
    ```cuda title="layernorm.cu"
    // layernorm.cu —— 一个 block 一行，寄存器缓存整行，两遍法求均值和方差
    // 编译：nvcc -O3 -arch=sm_75 layernorm.cu -o layernorm
    #include "common.cuh"

    constexpr int kThreads = 256, kMaxPerThread = 32;

    __device__ __forceinline__ float warp_sum(float v) {
      for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
      return v;
    }
    __device__ float block_sum(float v) {
      __shared__ float buf[32];
      __shared__ float total;
      const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
      v = warp_sum(v);
      if (lane == 0) buf[warp] = v;
      __syncthreads();
      if (warp == 0) {
        v = warp_sum(lane < blockDim.x / 32 ? buf[lane] : 0.f);
        if (lane == 0) total = v;
      }
      __syncthreads();
      float r = total;
      __syncthreads();   // 同一个 kernel 里会调用两次，读完再允许下一次覆盖
      return r;
    }

    __global__ void layer_norm(const float* __restrict__ x, const float* __restrict__ gamma,
                               const float* __restrict__ beta, float* __restrict__ y, int H, float eps) {
      const size_t off = static_cast<size_t>(blockIdx.x) * H;
      float v[kMaxPerThread];
      float s = 0.f;
    #pragma unroll
      for (int k = 0; k < kMaxPerThread; ++k) {
        int c = k * kThreads + threadIdx.x;
        v[k] = c < H ? x[off + c] : 0.f;
        s += v[k];
      }
      const float mean = block_sum(s) / H;
      float sq = 0.f;
    #pragma unroll
      for (int k = 0; k < kMaxPerThread; ++k) {
        int c = k * kThreads + threadIdx.x;
        float d = c < H ? v[k] - mean : 0.f;
        sq += d * d;
      }
      const float rstd = rsqrtf(block_sum(sq) / H + eps);
    #pragma unroll
      for (int k = 0; k < kMaxPerThread; ++k) {
        int c = k * kThreads + threadIdx.x;
        if (c < H) y[off + c] = (v[k] - mean) * rstd * gamma[c] + beta[c];
      }
    }

    int main() {
      const int rows = 256, H = 5120;
      const float eps = 1e-5f;
      const size_t n = static_cast<size_t>(rows) * H;
      std::vector<float> hx(n), hg(H), hb(H), ref(n), got(n);
      fill_random(hx, 1, 100.f, 102.f);   // 均值远大于标准差，检验数值稳定性
      fill_random(hg, 2, 0.5f, 1.5f);
      fill_random(hb, 3);
      for (int r = 0; r < rows; ++r) {
        double mean = 0, var = 0;
        for (int c = 0; c < H; ++c) mean += hx[r * H + c];
        mean /= H;
        for (int c = 0; c < H; ++c) var += (hx[r * H + c] - mean) * (hx[r * H + c] - mean);
        var /= H;
        for (int c = 0; c < H; ++c)
          ref[r * H + c] = static_cast<float>((hx[r * H + c] - mean) / std::sqrt(var + eps) * hg[c] + hb[c]);
      }
      float *dx, *dg, *db, *dy;
      CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dg, H * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&db, H * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dg, hg.data(), H * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(db, hb.data(), H * sizeof(float), cudaMemcpyHostToDevice));
      layer_norm<<<rows, kThreads>>>(dx, dg, db, dy, H, eps);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(got.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
      bool ok = check_close(got.data(), ref.data(), n, 1e-3f, 1e-3f);
      CUDA_CHECK(cudaFree(dx));
      CUDA_CHECK(cudaFree(dy));
      CUDA_CHECK(cudaFree(dg));
      CUDA_CHECK(cudaFree(db));
      return ok ? 0 : 1;
    }
    ```

    注意 `block_sum` 最后多了一次 `__syncthreads()`：同一个 kernel 里连续调用两次时，如果没有它，第二次调用可能在某些线程读取 `total` 之前就把它覆盖了。输入的均值约为 101、标准差约 0.6，如果用 $E[x^2] - E[x]^2$ 的一遍法，两个约 10000 的数相减，FP32 下会损失大部分有效数字。

**2. 思考题：大词表采样。** 大模型解码时要对词表大小（比如 152064）的 logits 做 softmax，再做 top-p 采样。batch 很小时（比如只有 1 行），一行一个 block 只能用上一个 SM。怎么改进？

??? success "参考思路"
    把一行切给多个 block：每个 block 处理一段，算出自己这段的 $(m, d)$ 写到临时缓冲区；第二个 kernel（或者最后完成的 block）合并这些部分结果，得到全局的 $(m, d)$；最后再按需要计算概率。这和 [FlashDecoding](../advanced/attention.md#decode-阶段的并行flash-decoding-与-split-k) 的 split-K 思路完全相同。另外，采样通常不需要真的把整行 softmax 写出来：比如贪心采样只需要 argmax，top-k 采样可以先在 logits 上选出 top-k 再做 softmax，FlashInfer 等库提供了专门融合的采样 kernel。

## 小结

- [x] softmax 先减最大值防止溢出；online softmax 把最大值和指数和合并成一次可并行归约的遍历。
- [x] 按行长度选择 warp 一行或 block 一行；短行用 warp 并把整行缓存在寄存器里。
- [x] 大模型多用 RMSNorm；方差计算注意数值稳定性（两遍法或 Welford）。
- [x] 访存瓶颈的算子，融合就是最有效的优化：fused_add_rms_norm 是标准例子。
