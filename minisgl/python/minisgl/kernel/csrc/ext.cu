// PyTorch 扩展：把 kv_kernels.cuh 里的 kernel 包装成 Python 可调用的函数。
// 由 minisgl/kernel/cuda_ext.py 在第一次使用时即时编译（torch.utils.cpp_extension.load）。
// CUDA stream 由 Python 端传入（torch.cuda.current_stream().cuda_stream），这样只依赖 torch 的
// CPU 头文件和 CUDA runtime，不需要 c10/cuda 的头文件。
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kv_kernels.cuh"

namespace {

constexpr int kThreads = 128;  // 每个 block 4 个 warp

bool aligned16(const at::Tensor& t, int64_t row_bytes, int64_t stride_bytes) {
  return row_bytes % 16 == 0 && stride_bytes % 16 == 0 &&
         reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

template <typename V>
void launch_store(at::Tensor& kc, at::Tensor& vc, const at::Tensor& idx, const at::Tensor& k,
                  const at::Tensor& v, int64_t n, int64_t row_bytes, cudaStream_t stream) {
  const int64_t es = kc.element_size();
  const int64_t blocks = (n * 32 + kThreads - 1) / kThreads;
  minisgl::store_kv_kernel<V><<<blocks, kThreads, 0, stream>>>(
      static_cast<V*>(kc.data_ptr()), static_cast<V*>(vc.data_ptr()), idx.data_ptr<int32_t>(),
      static_cast<const V*>(k.data_ptr()), static_cast<const V*>(v.data_ptr()), n,
      row_bytes / sizeof(V), kc.stride(0) * es / sizeof(V), k.stride(0) * es / sizeof(V));
}

}  // namespace

// k_cache/v_cache：[池中 token 数, H * D]（视图）；k/v：[n, H * D]；indices：int32 [n]
void store_kv(at::Tensor k_cache, at::Tensor v_cache, at::Tensor indices, at::Tensor k,
              at::Tensor v, int64_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(k_cache.stride(1) == 1 && k.stride(1) == 1, "rows must be contiguous");
  const int64_t n = indices.size(0);
  const int64_t row_bytes = k.size(1) * k.element_size();
  if (n == 0) return;
  const bool vec = aligned16(k_cache, row_bytes, k_cache.stride(0) * k_cache.element_size()) &&
                   aligned16(k, row_bytes, k.stride(0) * k.element_size());
  if (vec) {  // 快路径：每个线程一次搬 16 字节
    launch_store<uint4>(k_cache, v_cache, indices, k, v, n, row_bytes, stream);
  } else if (k.element_size() == 4) {
    launch_store<uint32_t>(k_cache, v_cache, indices, k, v, n, row_bytes, stream);
  } else {
    TORCH_CHECK(k.element_size() == 2, "unsupported element size");
    launch_store<uint16_t>(k_cache, v_cache, indices, k, v, n, row_bytes, stream);
  }
}

at::Tensor indexing(at::Tensor weight, at::Tensor ids, int64_t start, int64_t length,
                    int64_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  TORCH_CHECK(ids.scalar_type() == at::kInt && weight.is_contiguous());
  const int64_t n = ids.size(0);
  auto out = at::empty({n, weight.size(1)}, weight.options());
  const int64_t row_bytes = weight.size(1) * weight.element_size();
  TORCH_CHECK(row_bytes % 16 == 0, "embedding row must be a multiple of 16 bytes");
  const int64_t blocks = (n * 32 + kThreads - 1) / kThreads;
  if (n > 0) {
    minisgl::embedding_kernel<uint4><<<blocks, kThreads, 0, stream>>>(
        static_cast<uint4*>(out.data_ptr()), static_cast<const uint4*>(weight.data_ptr()),
        ids.data_ptr<int32_t>(), n, row_bytes / 16, start, length);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("store_kv", &store_kv);
  m.def("indexing", &indexing);
}
