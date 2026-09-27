# 量化与 GEMV

<p class="lead">大模型 decode 阶段每生成一个 token，都要把全部权重从显存读一遍，而 batch 较小时计算量很少。这时"读多少字节"几乎就是耗时本身。量化把权重从 16 位压到 8 位、4 位，直接减少读取量。这一章讲量化的基本格式、GEMV kernel 的写法，以及在寄存器里反量化的 W4A16 kernel。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么 batch = 1 的 decode 阶段，线性层是访存瓶颈？
    2. W4A16、W8A8、FP8 分别是什么意思？各适合什么场景？
    3. 按组量化（group size = 128）的缩放因子和零点是怎么存的？
    4. INT4 权重在 kernel 里是怎么解包和反量化的？
    5. 为什么 W4A16 的加速比会随 batch 增大而下降？

## 为什么 decode 需要量化

一个线性层 $y = Wx$，W 是 N×K。batch = 1 时这是矩阵向量乘（GEMV）：每个权重读一次、只做一次乘加，算术强度约为每 2 字节（BF16）1 次乘加，远低于任何 GPU 的脊点。**耗时 ≈ 权重字节数 / 显存带宽**。

以 70 亿参数的模型为例，BF16 权重约 14 GB，在 H100（3.35 TB/s）上每个 token 至少要约 4.2 ms，最多 240 token/s 左右。把权重压缩到 4 位（约 3.5 GB 加上少量缩放因子），理论上可以快接近 4 倍。

batch 增大时，同一份权重被多个请求复用，算术强度随 batch 线性增长。batch 足够大时（通常几十到上百），线性层重新变成计算瓶颈，这时只压缩权重就不够了，需要让计算也使用低精度（INT8/FP8 Tensor Core）。

## 常见的量化方案

| 方案 | 权重 | 激活 | 计算 | 适合 |
| --- | --- | --- | --- | --- |
| W8A16 / W4A16（weight-only） | INT8 / INT4 | FP16/BF16 | 在寄存器里把权重反量化成 FP16，用 FP16 Tensor Core 计算 | 小 batch 的 decode，访存瓶颈 |
| W8A8 INT8 | INT8 | INT8 | INT8 Tensor Core，INT32 累加 | 大 batch、prefill，计算瓶颈 |
| FP8（W8A8） | FP8 | FP8 | FP8 Tensor Core（sm_89+） | Hopper/Ada 上的通用选择，精度损失小 |
| FP4（NVFP4 / MXFP4） | FP4 | FP4 | FP4 Tensor Core（Blackwell） | 最新硬件上的极致吞吐 |

AWQ、GPTQ、SmoothQuant 等是**决定量化后权重取什么值**的算法（离线完成）；在 kernel 层面，它们最终都表现为"低位整数 + 缩放因子（+ 零点）"的格式。

### 量化公式

**非对称（带零点）量化**，把一组浮点数映射到 $[0, 2^b - 1]$：

$$
s = \frac{\max - \min}{2^b - 1},\qquad z = \text{round}\left(-\frac{\min}{s}\right),\qquad q = \text{clamp}\left(\text{round}\left(\frac{w}{s}\right) + z,\ 0,\ 2^b - 1\right)
$$

反量化：$\hat{w} = (q - z) \cdot s$。**对称量化**没有零点，把数值映射到 $[-2^{b-1}, 2^{b-1} - 1]$（或 $\pm (2^{b-1} - 1)$）。

**缩放因子的粒度**决定精度和开销的平衡：

- **按张量（per-tensor）**：整个矩阵一个缩放因子，最简单，精度最差；
- **按通道（per-channel）**：每个输出通道（W 的每一行）一个；
- **按组（per-group）**：每行内每 G 个连续元素一组，G 常取 128。INT4 权重几乎总是按组量化；
- **按块（block-wise）**：DeepSeek-V3 的 FP8 权重按 128×128 的块缩放，激活按 1×128 的组缩放。

## GEMV kernel 的写法

GEMV $y = Wx$，W 为 N×K 行主序。每个输出元素是 W 的一行与 x 的点积，最常见的并行方式是**一个 warp 负责一行**：32 个线程沿 K 方向分工读取这一行（合并访问，最好用 128 位的向量化读取），各自累加后做 warp 归约。x 被所有行共享，可以先加载到共享内存（或者依赖 L1/L2 缓存）。

这个结构简单有效：读 W 是完全连续的，每个 warp 的访存量足够大，warp 之间没有同步。优化的重点在于：

- **向量化**：每条指令读 16 字节；
- **每个线程有足够多的在途请求**：K 不大时，让一个 warp 同时处理 2-4 行；
- **填满 GPU**：N 较小时（比如 N = 4096），一行一个 warp 只有 4096 个 warp，要注意 block 大小和每 SM 的 warp 数。

## W4A16：在寄存器里反量化

INT4 权重的常见存储方式：**每个 32 位字存 8 个连续的 4 位权重**，低 4 位是第一个元素。每组（G 个元素）有一个缩放因子和一个零点。kernel 读一个 32 位字，用移位和掩码解出 8 个整数，减去零点、乘以缩放因子，再和 x 的对应 8 个元素做乘加：

```cuda
uint32_t word = w_packed[...];
#pragma unroll
for (int j = 0; j < 8; ++j) {
  int q = (word >> (4 * j)) & 0xF;
  acc += (q - zero) * scale * x[k + j];
}
```

一个 32 位字对应 8 个权重，读取量只有 FP32 权重的 1/8、BF16 权重的 1/4（外加每组一个缩放因子和零点，按 G = 128 计算，开销约为 3%）。

```cuda title="gemv_w4.cu"
// gemv_w4.cu —— FP32 GEMV 与 W4A16 风格的 INT4 按组量化 GEMV（一个 warp 负责一行）
// 编译：nvcc -O3 -arch=sm_75 gemv_w4.cu -o gemv_w4
// 为了让示例聚焦在解包与反量化上，激活和缩放因子都用 FP32；真实 kernel 用 FP16/BF16
#include "common.cuh"
#include <cstdint>

constexpr int GROUP = 128;       // 每 128 个 K 方向的元素共享一个缩放因子和零点
constexpr int kWarps = 4;        // 每个 block 4 个 warp，每个 warp 一行

__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}

// 基线：FP32 权重，每个 lane 每次读一个 float4
__global__ void gemv_fp32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K) {
  const int row = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= N) return;
  const float4* w4 = reinterpret_cast<const float4*>(W + static_cast<size_t>(row) * K);
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc = 0.f;
  for (int i = lane; i < K / 4; i += 32) {
    const float4 w = w4[i], v = x4[i];
    acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// INT4：W 打包成 [N, K/8] 个 uint32；scales、zeros 为 [N, K/GROUP]
__global__ void gemv_w4(const uint32_t* __restrict__ Wq, const float* __restrict__ scales,
                        const float* __restrict__ zeros, const float* __restrict__ x, float* __restrict__ y,
                        int N, int K) {
  extern __shared__ float xs[];          // x 被整个 block 共用，先放进共享内存
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();

  const int row = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= N) return;                  // 同步之后再退出，不影响 __syncthreads
  const uint32_t* wrow = Wq + static_cast<size_t>(row) * (K / 8);
  const float* srow = scales + static_cast<size_t>(row) * (K / GROUP);
  const float* zrow = zeros + static_cast<size_t>(row) * (K / GROUP);
  float acc = 0.f;
  for (int i = lane; i < K / 8; i += 32) {            // 相邻 lane 读相邻的 32 位字：合并访问
    const uint32_t word = wrow[i];
    const int k = i * 8, g = k / GROUP;
    const float s = srow[g], z = zrow[g];
    const float* xv = xs + k;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float q = static_cast<float>((word >> (4 * j)) & 0xFu);
      acc += (q - z) * s * xv[j];
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

int main() {
  const int N = 4096, K = 4096;   // K 需要是 GROUP 的整数倍
  std::vector<float> W(static_cast<size_t>(N) * K), x(K);
  fill_random(W, 1);
  fill_random(x, 2);

  // 离线量化：每组 128 个元素，非对称 4 位
  const int groups = K / GROUP;
  std::vector<uint32_t> Wq(static_cast<size_t>(N) * K / 8, 0u);
  std::vector<float> scales(static_cast<size_t>(N) * groups), zeros(scales.size()), W_deq(W.size());
  for (int r = 0; r < N; ++r)
    for (int g = 0; g < groups; ++g) {
      const float* w = &W[static_cast<size_t>(r) * K + g * GROUP];
      float lo = *std::min_element(w, w + GROUP), hi = *std::max_element(w, w + GROUP);
      float s = (hi - lo) / 15.f;
      if (s == 0.f) s = 1.f;
      float z = std::round(-lo / s);
      scales[static_cast<size_t>(r) * groups + g] = s;
      zeros[static_cast<size_t>(r) * groups + g] = z;
      for (int j = 0; j < GROUP; ++j) {
        const int k = g * GROUP + j;
        int q = static_cast<int>(std::round(w[j] / s + z));
        q = std::min(15, std::max(0, q));
        Wq[(static_cast<size_t>(r) * K + k) / 8] |= static_cast<uint32_t>(q) << (4 * (k % 8));
        W_deq[static_cast<size_t>(r) * K + k] = (q - z) * s;
      }
    }

  // 参考结果：FP32 GEMV，以及"用反量化后的权重"做的 GEMV（INT4 kernel 应与后者一致）
  std::vector<float> ref_fp32(N), ref_q(N), got(N);
  double qerr = 0;
  for (int r = 0; r < N; ++r) {
    double a = 0, b = 0;
    for (int k = 0; k < K; ++k) {
      a += double(W[static_cast<size_t>(r) * K + k]) * x[k];
      b += double(W_deq[static_cast<size_t>(r) * K + k]) * x[k];
    }
    ref_fp32[r] = static_cast<float>(a);
    ref_q[r] = static_cast<float>(b);
    qerr = std::max(qerr, std::fabs(a - b));
  }
  std::printf("max |y_fp32 - y_int4| on CPU (quantization error): %.4f\n", qerr);

  float *dW, *dx, *dy, *ds, *dz;
  uint32_t* dWq;
  CUDA_CHECK(cudaMalloc(&dW, W.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWq, Wq.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&ds, scales.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dz, zeros.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx, K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWq, Wq.data(), Wq.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dz, zeros.data(), zeros.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));

  const int blocks = (N + kWarps - 1) / kWarps;
  bool ok = true;
  gemv_fp32<<<blocks, kWarps * 32>>>(dW, dx, dy, N, K);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("gemv_fp32 ");
  ok &= check_close(got.data(), ref_fp32.data(), N, 1e-3f, 1e-3f);
  gemv_w4<<<blocks, kWarps * 32, K * sizeof(float)>>>(dWq, ds, dz, dx, dy, N, K);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("gemv_w4   ");
  ok &= check_close(got.data(), ref_q.data(), N, 1e-3f, 1e-3f);

  const double fp32_bytes = double(W.size()) * 4;
  const double w4_bytes = double(Wq.size()) * 4 + double(scales.size()) * 8;
  float t1 = time_ms([&] { gemv_fp32<<<blocks, kWarps * 32>>>(dW, dx, dy, N, K); });
  float t2 = time_ms([&] { gemv_w4<<<blocks, kWarps * 32, K * sizeof(float)>>>(dWq, ds, dz, dx, dy, N, K); });
  std::printf("fp32 weights: %.3f ms (%.1f GB/s)\nint4 weights: %.3f ms (%.1f GB/s), speedup %.2fx\n",
              t1, gbps(fp32_bytes, t1), t2, gbps(w4_bytes, t2), t1 / t2);
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(dWq));
  CUDA_CHECK(cudaFree(ds));
  CUDA_CHECK(cudaFree(dz));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return ok ? 0 : 1;
}
```

程序同时输出了**量化误差**（FP32 权重与反量化权重得到的结果之差）和 kernel 的**实现误差**（INT4 kernel 与"反量化权重的 CPU 结果"之差）。这两者要分开看：前者是量化方案本身带来的，后者必须接近 0，否则就是 kernel 的 bug。

在 GPU 上运行时留意两件事：INT4 版本的**有效带宽**（按实际读取的字节数计算）比 FP32 版本低，因为解包和反量化带来了额外的指令；但因为读取的字节数少了约 7 倍，总耗时仍然明显更短。

## 生产级 kernel 用了哪些技巧

- **快速的 INT4 → FP16 转换**：不用逐个移位再转浮点，而是利用 FP16 的位模式。把 4 位整数放到 FP16 尾数的低位，拼上一个固定的指数（`0x6400` 对应 1024），得到的就是 1024 + q，再减去 1024 即可。配合 `lop3` 指令一次处理多个元素。FasterTransformer、Marlin 都使用了这类技巧；
- **权重离线重排**：量化时就把权重按 kernel 读取的顺序排好，让一次 128 位读取正好得到 Tensor Core fragment 需要的数据，避免在 kernel 里做数据重排；
- **Marlin**（W4A16 GEMM）：在 batch 为 1 到几十的范围内都接近理论最优的加速比，vLLM 默认使用；Hopper 上还有 Machete 等后续实现；
- **FP8 GEMM**：Tensor Core 直接计算 FP8，缩放因子在累加后应用。按块缩放时，每个 K 方向的块结束后要把 Tensor Core 的部分结果乘上对应的缩放因子再累加到 FP32 累加器上，DeepGEMM 对此做了专门的优化；
- **MoE**：多个专家的 GEMM 用分组 GEMM 一次完成，vLLM 的 fused_moe 把 token 分发、分组 GEMM、激活函数融合在一起。

## 练习

**1. W8A16 GEMV。** 实现对称的按通道 INT8 量化（每行一个缩放因子，$q = \text{round}(w / s)$，$s = \max|w| / 127$），权重用 `int8_t` 存储，每个 lane 每次读一个 `int4`（16 个 INT8）。

??? success "参考思路"
    离线量化：每行求 $s = \max_k |w_k| / 127$，$q_k = \text{clamp}(\text{round}(w_k / s), -127, 127)$。kernel 中每个 lane 读取 16 字节 `int4`，通过 `reinterpret_cast<const int8_t*>` 取出 16 个值，与 x 的 16 个元素做乘加，最后乘上本行的 $s$。由于缩放因子按行共享，可以在累加完成后再乘一次，而不是每个元素都乘，这和 W4 按组量化需要在组内累加后乘 scale 的思路一样。

**2. 分析题。** 在 batch = 1、4、32、128 时，W4A16 相对 BF16 的加速比分别大致会怎样变化？为什么？

??? success "参考答案"
    - **batch = 1、4**：严重访存瓶颈，读取量约为 BF16 的 1/4（加上少量缩放因子），加速比接近 3-4 倍；
    - **batch = 32**：权重被 32 个请求复用，算术强度上升，BF16 GEMM 逐渐接近计算瓶颈；W4A16 仍然要用 FP16 Tensor Core 计算，还多了反量化的开销，加速比明显下降；
    - **batch = 128**：两者都是计算瓶颈，W4A16 因反量化开销可能反而更慢。这时应该用 W8A8（INT8 或 FP8）让计算本身也变快。

    这就是推理引擎通常同时支持多种量化 kernel，并按 batch 大小选择的原因。

## 小结

- [x] 小 batch 的 decode 是访存瓶颈，耗时 ≈ 权重字节数 / 带宽，weight-only 量化直接减少读取量。
- [x] 大 batch 时重新变成计算瓶颈，需要 W8A8（INT8/FP8）或 FP4 让计算也变快。
- [x] INT4 通常按组量化（G = 128），8 个权重打包成一个 32 位字，kernel 在寄存器里解包反量化。
- [x] GEMV 常用一个 warp 负责一行，向量化读取，warp 归约。
- [x] 分清量化误差和 kernel 实现误差；生产级 kernel 还会用位技巧加速转换并离线重排权重。
