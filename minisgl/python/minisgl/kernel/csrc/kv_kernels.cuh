// mini-sglang 的两个访存 kernel：把新算出的 K/V 按位置写进 KV 池，以及（词表并行的）嵌入查表。
// 两者都是纯粹的"按下标搬运整行数据"，瓶颈只有显存带宽：
//   * 一个 warp 负责一行（一个 token），32 个线程协作拷贝；
//   * 按向量类型 V 拷贝（行字节数是 16 的倍数时用 uint4，一次 16 字节），访存是合并的。
#pragma once
#include <cstdint>

namespace minisgl {

// k_cache/v_cache：[池中 token 数, row]，行步长 cache_stride；k/v：[n, row]，行步长 input_stride。
// 所有长度和步长都以 V 为单位。
template <typename V>
__global__ void store_kv_kernel(V* __restrict__ k_cache, V* __restrict__ v_cache,
                                const int32_t* __restrict__ indices, const V* __restrict__ k,
                                const V* __restrict__ v, int64_t n, int64_t row,
                                int64_t cache_stride, int64_t input_stride) {
  const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= n) return;
  const int64_t dst = static_cast<int64_t>(indices[warp]) * cache_stride;
  const int64_t src = warp * input_stride;
  for (int64_t i = lane; i < row; i += 32) {
    k_cache[dst + i] = k[src + i];
    v_cache[dst + i] = v[src + i];
  }
}

// out[t] = weight[ids[t] - start]，若 ids[t] 不在 [start, start + length) 内则填 0（词表并行）。
template <typename V>
__global__ void embedding_kernel(V* __restrict__ out, const V* __restrict__ weight,
                                 const int32_t* __restrict__ ids, int64_t n, int64_t row,
                                 int64_t start, int64_t length) {
  const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= n) return;
  const int64_t local = static_cast<int64_t>(ids[warp]) - start;
  const bool valid = local >= 0 && local < length;
  for (int64_t i = lane; i < row; i += 32) {
    out[warp * row + i] = valid ? weight[local * row + i] : V{};
  }
}

}  // namespace minisgl
