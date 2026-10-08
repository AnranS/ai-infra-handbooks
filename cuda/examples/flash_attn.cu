// flash_attn.cu —— 教学版 FlashAttention 前向：分块 + online softmax，支持因果掩码
// 编译：nvcc -O3 -arch=sm_75 flash_attn.cu -o flash_attn
// 布局：Q、K、V、O 都是 [batch * heads, seq_len, D] 连续存储
#include "common.cuh"

constexpr int Br = 64;   // 每个 block 处理的 query 行数（= 线程数）
constexpr int Bc = 32;   // 每次加载的 key/value 行数

template <int D>
__global__ void __launch_bounds__(Br)
flash_attn_fwd(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
               float* __restrict__ O, int N, float scale, bool causal) {
  __shared__ float Ks[Bc][D];
  __shared__ float Vs[Bc][D];
  const size_t head_off = static_cast<size_t>(blockIdx.y) * N * D;
  const int q_row = blockIdx.x * Br + threadIdx.x;
  const bool active = q_row < N;

  float q[D], o[D];
#pragma unroll
  for (int k = 0; k < D; ++k) {
    q[k] = active ? Q[head_off + static_cast<size_t>(q_row) * D + k] * scale : 0.f;   // 预先乘上 1/sqrt(d)
    o[k] = 0.f;
  }
  float m = -INFINITY, l = 0.f;

  // 因果掩码：这个 block 最后一个 query 的位置之后的 key 都用不到
  const int kv_end = causal ? min(N, static_cast<int>(blockIdx.x + 1) * Br) : N;
  for (int j0 = 0; j0 < kv_end; j0 += Bc) {
    // 协作加载 K、V 的一块（越界填 0，后面用掩码排除）
    for (int idx = threadIdx.x; idx < Bc * D; idx += Br) {
      const int r = idx / D, c = idx % D, g = j0 + r;
      Ks[r][c] = g < N ? K[head_off + static_cast<size_t>(g) * D + c] : 0.f;
      Vs[r][c] = g < N ? V[head_off + static_cast<size_t>(g) * D + c] : 0.f;
    }
    __syncthreads();

    // 1) 这一块的分数与块内最大值
    float s[Bc];
    float block_max = -INFINITY;
#pragma unroll
    for (int c = 0; c < Bc; ++c) {
      const int key = j0 + c;
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < D; ++k) acc += q[k] * Ks[c][k];   // 所有线程读同一行：广播
      const bool masked = key >= N || (causal && key > q_row);
      s[c] = masked ? -INFINITY : acc;
      block_max = fmaxf(block_max, s[c]);
    }

    // 2) online softmax 更新；整块都被掩码时 m 保持不变
    const float m_new = fmaxf(m, block_max);
    if (m_new != -INFINITY) {
      const float correction = __expf(m - m_new);   // m 为 -inf 时结果是 0
      float row_sum = 0.f;
#pragma unroll
      for (int c = 0; c < Bc; ++c) {
        s[c] = __expf(s[c] - m_new);                // 被掩码的位置得到 0
        row_sum += s[c];
      }
      l = l * correction + row_sum;
      // 3) 缩放旧的累加器，加上 P V
#pragma unroll
      for (int k = 0; k < D; ++k) {
        float acc = o[k] * correction;
#pragma unroll
        for (int c = 0; c < Bc; ++c) acc += s[c] * Vs[c][k];
        o[k] = acc;
      }
      m = m_new;
    }
    __syncthreads();   // 下一轮覆盖 Ks/Vs 之前，确保大家都用完了
  }

  if (active) {
    const float inv_l = 1.f / l;
#pragma unroll
    for (int k = 0; k < D; ++k) O[head_off + static_cast<size_t>(q_row) * D + k] = o[k] * inv_l;
  }
}

void attention_cpu(const float* Q, const float* K, const float* V, float* O, int BH, int N, int D, bool causal) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  std::vector<double> s(N);
  for (int h = 0; h < BH; ++h) {
    const size_t off = static_cast<size_t>(h) * N * D;
    for (int i = 0; i < N; ++i) {
      double mx = -1e300;
      const int end = causal ? i + 1 : N;
      for (int j = 0; j < end; ++j) {
        double acc = 0;
        for (int k = 0; k < D; ++k) acc += double(Q[off + i * D + k]) * K[off + j * D + k];
        s[j] = acc * scale;
        mx = std::max(mx, s[j]);
      }
      double sum = 0;
      for (int j = 0; j < end; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
      for (int k = 0; k < D; ++k) {
        double acc = 0;
        for (int j = 0; j < end; ++j) acc += s[j] * V[off + j * D + k];
        O[off + i * D + k] = static_cast<float>(acc / sum);
      }
    }
  }
}

int main() {
  constexpr int D = 64;
  const int batch_heads = 8, N = 1000;   // 故意不是块大小的整数倍
  const size_t n = static_cast<size_t>(batch_heads) * N * D, bytes = n * sizeof(float);
  std::vector<float> hq(n), hk(n), hv(n), ref(n), got(n);
  fill_random(hq, 1);
  fill_random(hk, 2);
  fill_random(hv, 3);
  float *dq, *dk, *dv, *dout;
  CUDA_CHECK(cudaMalloc(&dq, bytes));
  CUDA_CHECK(cudaMalloc(&dk, bytes));
  CUDA_CHECK(cudaMalloc(&dv, bytes));
  CUDA_CHECK(cudaMalloc(&dout, bytes));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), bytes, cudaMemcpyHostToDevice));

  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  dim3 grid((N + Br - 1) / Br, batch_heads);
  bool ok = true;
  for (bool causal : {false, true}) {
    attention_cpu(hq.data(), hk.data(), hv.data(), ref.data(), batch_heads, N, D, causal);
    flash_attn_fwd<D><<<grid, Br>>>(dq, dk, dv, dout, N, scale, causal);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), dout, bytes, cudaMemcpyDeviceToHost));
    std::printf("causal=%d  ", causal);
    ok &= check_close(got.data(), ref.data(), n, 1e-3f, 1e-4f);
    float ms = time_ms([&] { flash_attn_fwd<D><<<grid, Br>>>(dq, dk, dv, dout, N, scale, causal); });
    const double flops = 4.0 * batch_heads * N * N * D * (causal ? 0.5 : 1.0);
    std::printf("           %.3f ms, %.2f TFLOPS\n", ms, tflops(flops, ms));
  }
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dout));
  return ok ? 0 : 1;
}
