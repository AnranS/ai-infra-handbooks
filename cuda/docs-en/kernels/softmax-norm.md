# Softmax and normalization

<p class="lead">Softmax, LayerNorm and RMSNorm are the most frequently called "row reduction" kernels in a Transformer: reduce each row of a matrix, then transform it elementwise with the result. All of them are memory-bound, so the heart of optimizing them is "read and write each element once" plus fusion with the surrounding kernels. Online softmax, moreover, is the foundation of FlashAttention.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why does softmax subtract the maximum first?
    2. How does online softmax find the maximum and the sum of exponentials in one pass?
    3. With rows of only 128 elements, is one block per row a good fit?
    4. How do LayerNorm and RMSNorm differ? Why do LLMs mostly use RMSNorm?
    5. What does vLLM's `fused_add_rms_norm` fuse? How much traffic does it save?

??? success "Answers (try it yourself first, then expand)"
    1. $e^x$ overflows for large $x$; softmax is unchanged by subtracting a constant from every element, and after subtracting the maximum the largest exponential is 1, which cannot overflow.
    2. Keep the running maximum $m$ and the sum of exponentials $d$: on a new element $x$, if $x > m$, first scale the existing sum by $e^{m - x}$ and update $m$, then add $e^{x - m}$. The merge is associative, so it also reduces in parallel across threads.
    3. Not a good fit: most of a block's few hundred threads have nothing to do and a block-level synchronization is needed anyway. For short rows use one warp per row, keep the whole row in registers, and only warp shuffles are needed.
    4. LayerNorm subtracts the mean, divides by the standard deviation, then scales and shifts; RMSNorm only divides by the root mean square and scales, with one less reduction and one less parameter set, equally effective and cheaper.
    5. The residual add and RMSNorm: read the hidden state and the residual once, write back the new residual and the normalized output. Done separately that is 5 full-row accesses (the add reads 2 and writes 1, RMSNorm reads 1 and writes 1); fused it is 4, plus one fewer kernel launch. For a memory-bound kernel the number of accesses is the time.

## Softmax and numerical stability {#softmax-与数值稳定性}

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

Computing $e^{x_i}$ directly overflows easily: in FP32 even $e^{89}$ is out of range, and attention scores and logits are often that large. So in practice the row's maximum $m = \max_j x_j$ is subtracted first, which changes nothing mathematically but keeps every exponential at or below 1:

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

That is **safe softmax**. A naive implementation needs three passes: the maximum, the sum of exponentials, and the output.

## Online softmax: the maximum and the sum in one pass {#online-softmax一次遍历求出最大值和指数和}

Can the maximum be found while the exponentials are accumulated? Yes. Keep the running maximum $m$ and the sum $d$ measured against it, and for each new element $x$:

$$
m' = \max(m, x),\qquad d' = d \cdot e^{m - m'} + e^{x - m'}
$$

When the maximum grows, scale the accumulated sum down by $e^{m - m'}$. The algorithm comes from Milakov and Gimelshein's 2018 paper *Online normalizer calculation for softmax*.

More importantly, **two partial results also merge**:

$$
m = \max(m_1, m_2),\qquad d = d_1 e^{m_1 - m} + d_2 e^{m_2 - m}
$$

which means $(m, d)$ **reduces in parallel** just like a sum: each thread handles its own elements and warp shuffles and shared memory combine them level by level. Softmax therefore goes from three passes to two: one for $(m, d)$ and one to write the output. FlashAttention applies exactly this merge rule to attention computed in tiles; see [FlashAttention](../advanced/attention.md).

Step through it and watch (m, d) update and the old sum get rescaled:

<div class="aig-widget" data-widget="online-softmax"></div>

## Choosing the parallelization by row length {#按行长度选择并行方式}

| Row length | Approach | Notes |
| --- | --- | --- |
| very short (≤ 32) | one warp per several rows | with a small group like `tiled_partition<8>` |
| medium (up to about 1024) | **one warp per row** | the whole row fits in registers, memory is read once, the reduction stays within the warp, and neither shared memory nor `__syncthreads()` is needed |
| long (thousands to tens of thousands) | one block per row | an LLM's vocabulary logits (30,000 to 150,000) are this case |
| very long with few rows | several blocks per row | each computes a partial $(m, d)$ and they are merged |

One block per row is very wasteful on short rows: 128 elements split across 256 threads leaves half of them idle and still pays for the block-level synchronization. Conversely, one warp per row on a long row makes each thread hold hundreds of elements and the registers spill. The softmax implementations in OneFlow, PyTorch and others switch between several kernels based on the column count.

## Implementation {#实现}

```cuda title="softmax.cu"
// softmax.cu - three row-softmax implementations: a three-pass block, one warp per row (cached in registers), and a two-pass online block
// build: nvcc -O3 -arch=sm_75 softmax.cu -o softmax
#include "common.cuh"

__device__ __forceinline__ float warp_max(float v) {
  for (int m = 16; m > 0; m /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}

// a block reduction with the result in every thread. op: 0 = sum, 1 = max
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
  __syncthreads();   // stops the next call overwriting buf/result while threads are still reading
  return v;
}

// v1: one block per row, three passes over global memory (max, sum, write)
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

// v2: one warp per row with the whole row in registers, so memory is read once and written once. Requires cols <= COLS
template <int COLS>
__global__ void softmax_warp(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
  constexpr int kPerLane = (COLS + 31) / 32;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= rows) return;   // the whole warp exits together, leaving other warps alone
  const float* in = x + static_cast<size_t>(row) * cols;
  float v[kPerLane];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    int c = k * 32 + lane;                 // neighbouring lanes read neighbouring elements: coalesced
    v[k] = c < cols ? in[c] : -INFINITY;
    m = fmaxf(m, v[k]);
  }
  m = warp_max(m);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    v[k] = __expf(v[k] - m);               // exp(-inf) = 0, so out-of-range positions do not affect the sum
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

// merging two (m, d) pairs of the online softmax
struct MD {
  float m, d;
};
__device__ __forceinline__ MD md_combine(MD a, MD b) {
  float m = fmaxf(a.m, b.m);
  // avoids inf - inf = nan when both sides are -inf (no elements yet)
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

// v3: one block per row, online softmax: one pass for (m, d) and one to write the output
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
  // two shapes: attention scores (short rows, many of them) and vocabulary logits (long rows, few of them)
  struct Shape { int rows, cols; } shapes[] = {{4096, 1000}, {64, 50000}};
  for (const Shape& sh : shapes) {
    const int rows = sh.rows, cols = sh.cols;
    const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
    std::vector<float> hx(n), ref(n), got(n);
    fill_random(hx, rows, -20.f, 20.f);
    hx[5] = 90.f;   // one very large value: an implementation that does not subtract the maximum overflows
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

`__expf` is the fast exponential, slightly less accurate than `expf`, which is why the softmax check uses a relative tolerance of `1e-3`. Production libraries generally compute in FP32 while reading and writing BF16/FP16, with 128-bit vectorized instructions on both.

## LayerNorm and RMSNorm {#layernorm-与-rmsnorm}

LayerNorm standardizes each row (one token's hidden vector):

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,\qquad
\mu = \frac{1}{H}\sum_i x_i,\quad \sigma^2 = \frac{1}{H}\sum_i (x_i - \mu)^2
$$

RMSNorm drops the mean subtraction and the bias $\beta$:

$$
y = \frac{x}{\sqrt{\frac{1}{H}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

It needs one less reduction and one less parameter at comparable quality, which is why LLaMA, Qwen, DeepSeek and the rest all use RMSNorm.

The implementation has the same shape as softmax: one block (or warp) per row, reduce first and transform elementwise second. There are a few ways to compute the variance:

- **two passes**: find the mean, then $\sum (x - \mu)^2$. Numerically stable, but it either reads memory twice or caches the whole row in registers or shared memory;
- **one pass**: accumulate $\sum x$ and $\sum x^2$ together, with the variance as $E[x^2] - E[x]^2$. One reduction, but severe cancellation when the mean is much larger than the standard deviation;
- **Welford's algorithm**: like online softmax, keep (count, mean, M2) and update incrementally, with partial results that merge. Stable and single-pass, which is what PyTorch's and Apex's LayerNorm use.

The hidden dimension H is usually a few thousand (4096, 8192), and one block per row with the whole row in registers is the common choice: at H = 4096 with 256 threads that is 16 elements per thread.

## Fusion: fused_add_rms_norm {#算子融合fused_add_rms_norm}

Every Transformer layer has this pattern:

```text
residual = residual + hidden        # 残差连接
hidden   = rms_norm(residual) * w   # 下一个子层的输入
```

Done separately it needs 5 full-row accesses: the residual add reads 2 and writes 1, and RMSNorm reads 1 and writes 1. Fused into one kernel it reads `residual` and `hidden` once each and writes the new `residual` and the normalized result once each, 4 in all, with one fewer kernel launch. For a memory-bound kernel, **the number of accesses is the time**. `fused_add_rms_norm` in vLLM and SGLang is exactly this kernel, and it is this chapter's exercise.

```cuda title="rmsnorm.cu"
// rmsnorm.cu - RMSNorm, and fused_add_rms_norm with the residual add merged in
// build: nvcc -O3 -arch=sm_75 rmsnorm.cu -o rmsnorm
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

// one block per row; each thread caches its own elements in registers (at most kMaxPerThread of them)
constexpr int kThreads = 256;
constexpr int kMaxPerThread = 32;   // supports hidden <= 8192

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

// residual += hidden; hidden = rms_norm(residual) * w. Both tensors are updated in place
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
      residual[off + c] = v[k];            // the new residual is written back for the next layer
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

  // the CPU reference: RMSNorm(x) first, then the fused version
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

  // the reference result for the fused version
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
  fused_add_rms_norm<<<tokens, kThreads>>>(dx, dr, dw, hidden, eps);   // modifies dx and dr in place
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

!!! interview "Answering in an interview"
    On softmax and normalization: subtract the maximum first to avoid overflow; online softmax keeps the maximum and the sum of exponentials in one pass (scaling the existing sum by $e^{m_{old}-m_{new}}$ on a larger value), which is what FlashAttention rests on; choose the parallelization by row length, one warp per row with the row in registers for short rows and one block per row for long ones. These kernels are bandwidth-bound, so fusion is the most effective optimization: vLLM's `fused_add_rms_norm` merges the residual add with RMSNorm and saves a full read and write of the hidden state.

## Exercises {#练习}

**1. LayerNorm.** Following `rms_norm`, implement LayerNorm with γ and β using two passes (the mean first, then the variance from the values cached in registers), and compare against the CPU.

??? success "Answer"
    ```cuda title="layernorm.cu"
    // layernorm.cu - one block per row, the whole row in registers, two passes for the mean and the variance
    // build: nvcc -O3 -arch=sm_75 layernorm.cu -o layernorm
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
      __syncthreads();   // called twice in one kernel, so the next call may overwrite only after the read
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
      fill_random(hx, 1, 100.f, 102.f);   // a mean far above the standard deviation, to exercise numerical stability
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

    Note the extra `__syncthreads()` at the end of `block_sum`: when it is called twice in one kernel, without it the second call could overwrite `total` before some threads have read it. The input has a mean of about 101 and a standard deviation of about 0.6, so the one-pass $E[x^2] - E[x]^2$ would subtract two numbers near 10000 and lose most of the significant digits in FP32.

**2. A thinking exercise: sampling over a large vocabulary.** During decoding an LLM softmaxes logits over the whole vocabulary (152064, say) and then does top-p sampling. With a small batch (one row, say), one block per row uses only one SM. How would you improve it?

??? success "Approach"
    Split the row across several blocks: each handles a stretch and writes its own $(m, d)$ to a scratch buffer; a second kernel (or the last block to finish) merges those partial results into the global $(m, d)$; and the probabilities are computed afterwards as needed. This is exactly the split-K idea of [FlashDecoding](../advanced/attention.md#decode-阶段的并行flash-decoding-与-split-k). Note also that sampling rarely needs the full softmax written out: greedy sampling only needs the argmax, and top-k sampling can select the top k on the logits before the softmax, with libraries like FlashInfer providing fused sampling kernels.

## Summary {#小结}

- [x] Softmax subtracts the maximum to avoid overflow; online softmax merges the maximum and the sum of exponentials into one pass that reduces in parallel.
- [x] Choose one warp per row or one block per row by row length; short rows use a warp and cache the whole row in registers.
- [x] LLMs mostly use RMSNorm; mind the numerical stability of the variance (two passes or Welford).
- [x] For memory-bound kernels, fusion is the most effective optimization: fused_add_rms_norm is the standard example.
