// paged_decode.cu —— 分页 KV Cache 上的 decode 注意力，支持 GQA
// 编译：nvcc -O3 -arch=sm_75 paged_decode.cu -o paged_decode
// KV Cache 布局：[num_physical_blocks, PAGE, num_kv_heads, D]
#include "common.cuh"

constexpr int PAGE = 16;          // 每个物理块存 16 个 token
constexpr int kThreads = 128;     // 4 个 warp
constexpr int kMaxCtx = 4096;     // 本示例支持的最大上下文长度（分数放在共享内存里）

__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int m = 16; m > 0; m /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
  return v;
}

template <int D>
__global__ void __launch_bounds__(kThreads)
paged_decode_attention(const float* __restrict__ q,          // [num_seqs, num_heads, D]
                       const float* __restrict__ k_cache,    // [num_blocks, PAGE, num_kv_heads, D]
                       const float* __restrict__ v_cache,
                       const int* __restrict__ block_table,  // [num_seqs, max_blocks_per_seq]
                       const int* __restrict__ context_lens, // [num_seqs]
                       float* __restrict__ out,              // [num_seqs, num_heads, D]
                       int num_heads, int num_kv_heads, int max_blocks_per_seq, float scale) {
  static_assert(D % 32 == 0 && D <= kThreads, "one output dim per thread");
  constexpr int kDimsPerLane = D / 32;
  __shared__ float scores[kMaxCtx];
  __shared__ float red[kThreads / 32];

  const int seq = blockIdx.y, head = blockIdx.x;
  const int kv_head = head / (num_heads / num_kv_heads);   // GQA：若干个 query 头共享一个 KV 头
  const int ctx = context_lens[seq];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = kThreads / 32;
  const int* table = block_table + static_cast<size_t>(seq) * max_blocks_per_seq;

  // 每个 lane 持有 q 的 kDimsPerLane 个维度
  const float* qp = q + (static_cast<size_t>(seq) * num_heads + head) * D;
  float qr[kDimsPerLane];
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) qr[i] = qp[lane + i * 32] * scale;

  // 第 1 步：分数。warp w 处理 token w, w + nwarps, ...
  float local_max = -INFINITY;
  for (int t = warp; t < ctx; t += nwarps) {
    const int phys = table[t / PAGE], slot = t % PAGE;
    const float* kp = k_cache + ((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kv_head) * D;
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) dot += qr[i] * kp[lane + i * 32];
    dot = warp_sum(dot);
    if (lane == 0) scores[t] = dot;
    local_max = fmaxf(local_max, dot);
  }

  // 第 2 步：block 内最大值 → 指数与求和
  if (lane == 0) red[warp] = local_max;
  __syncthreads();
  float mx = lane < nwarps ? red[lane] : -INFINITY;
  mx = warp_max(mx);            // 每个 warp 都算一遍，结果相同，省去一次广播
  __syncthreads();
  float local_sum = 0.f;
  for (int t = threadIdx.x; t < ctx; t += kThreads) {
    float p = __expf(scores[t] - mx);
    scores[t] = p;
    local_sum += p;
  }
  local_sum = warp_sum(local_sum);
  if (lane == 0) red[warp] = local_sum;
  __syncthreads();
  float sum = lane < nwarps ? red[lane] : 0.f;
  sum = warp_sum(sum);
  const float inv_sum = 1.f / sum;

  // 第 3 步：输出。线程 d 负责第 d 个维度
  if (threadIdx.x < D) {
    const int d = threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < ctx; ++t) {
      const int phys = table[t / PAGE], slot = t % PAGE;
      acc += scores[t] * v_cache[((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kv_head) * D + d];
    }
    out[(static_cast<size_t>(seq) * num_heads + head) * D + d] = acc * inv_sum;
  }
}

int main() {
  constexpr int D = 128;
  const int num_seqs = 4, num_heads = 32, num_kv_heads = 8;
  const std::vector<int> ctx_lens = {1, 17, 1000, 4096};   // 各种长度，包括不满一页的情况
  int max_blocks_per_seq = 0, total_blocks = 0;
  for (int c : ctx_lens) {
    max_blocks_per_seq = std::max(max_blocks_per_seq, (c + PAGE - 1) / PAGE);
    total_blocks += (c + PAGE - 1) / PAGE;
  }
  // 物理块随机打乱分配，模拟真实的分页情况
  std::vector<int> perm(total_blocks);
  for (int i = 0; i < total_blocks; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), std::mt19937(7));
  std::vector<int> table(static_cast<size_t>(num_seqs) * max_blocks_per_seq, -1);
  for (int s = 0, next = 0; s < num_seqs; ++s)
    for (int b = 0; b < (ctx_lens[s] + PAGE - 1) / PAGE; ++b) table[s * max_blocks_per_seq + b] = perm[next++];

  const size_t cache_elems = static_cast<size_t>(total_blocks) * PAGE * num_kv_heads * D;
  const size_t q_elems = static_cast<size_t>(num_seqs) * num_heads * D;
  std::vector<float> hq(q_elems), hk(cache_elems), hv(cache_elems), ref(q_elems), got(q_elems);
  fill_random(hq, 1);
  fill_random(hk, 2);
  fill_random(hv, 3);
  const float scale = 1.f / std::sqrt(static_cast<float>(D));

  // CPU 参考：按逻辑顺序展开 KV
  for (int s = 0; s < num_seqs; ++s)
    for (int h = 0; h < num_heads; ++h) {
      const int kvh = h / (num_heads / num_kv_heads), ctx = ctx_lens[s];
      std::vector<double> sc(ctx);
      double mx = -1e300;
      for (int t = 0; t < ctx; ++t) {
        const int phys = table[s * max_blocks_per_seq + t / PAGE], slot = t % PAGE;
        const float* kp = &hk[((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kvh) * D];
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += double(hq[(s * num_heads + h) * D + d]) * kp[d];
        sc[t] = dot * scale;
        mx = std::max(mx, sc[t]);
      }
      double sum = 0;
      for (int t = 0; t < ctx; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
      for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int t = 0; t < ctx; ++t) {
          const int phys = table[s * max_blocks_per_seq + t / PAGE], slot = t % PAGE;
          acc += sc[t] * hv[((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kvh) * D + d];
        }
        ref[(s * num_heads + h) * D + d] = static_cast<float>(acc / sum);
      }
    }

  float *dq, *dk, *dv, *dout;
  int *dtable, *dctx;
  CUDA_CHECK(cudaMalloc(&dq, q_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, q_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dk, cache_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, cache_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dtable, table.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dctx, num_seqs * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), q_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), cache_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), cache_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dtable, table.data(), table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dctx, ctx_lens.data(), num_seqs * sizeof(int), cudaMemcpyHostToDevice));

  dim3 grid(num_heads, num_seqs);
  auto launch = [&] {
    paged_decode_attention<D><<<grid, kThreads>>>(dq, dk, dv, dtable, dctx, dout, num_heads, num_kv_heads,
                                                  max_blocks_per_seq, scale);
  };
  launch();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dout, q_elems * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), q_elems, 1e-3f, 1e-5f);
  float ms = time_ms(launch);
  size_t kv_bytes = 0;
  for (int c : ctx_lens) kv_bytes += static_cast<size_t>(c) * num_heads * D * 2 * sizeof(float);
  std::printf("paged decode: %.3f ms, %.1f GB/s of K/V reads (GQA heads re-read shared KV)\n", ms, gbps(kv_bytes, ms));
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dout));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dtable));
  CUDA_CHECK(cudaFree(dctx));
  return ok ? 0 : 1;
}
