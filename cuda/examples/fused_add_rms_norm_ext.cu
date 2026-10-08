// fused_add_rms_norm_ext.cu —— 注册为 PyTorch 自定义算子 torch.ops.handbook.fused_add_rms_norm
// 构建：见下方的 setup.py（CUDAExtension）；也可以用 torch.utils.cpp_extension.load 即时编译
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

template <int kThreads>
__device__ float block_sum(float v) {
  __shared__ float buf[kThreads / 32];
  __shared__ float total;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < kThreads / 32 ? buf[lane] : 0.f;
    for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// residual += hidden；hidden = rms_norm(residual) * weight。统计量用 FP32 计算
template <typename scalar_t, int kThreads>
__global__ void fused_add_rms_norm_kernel(scalar_t* __restrict__ hidden, scalar_t* __restrict__ residual,
                                          const scalar_t* __restrict__ weight, int n, float eps) {
  const size_t off = static_cast<size_t>(blockIdx.x) * n;
  float ss = 0.f;
  for (int c = threadIdx.x; c < n; c += kThreads) {
    const float v = static_cast<float>(residual[off + c]) + static_cast<float>(hidden[off + c]);
    residual[off + c] = static_cast<scalar_t>(v);
    ss += v * v;
  }
  const float scale = rsqrtf(block_sum<kThreads>(ss) / n + eps);
  for (int c = threadIdx.x; c < n; c += kThreads) {
    // 读回刚写入的残差（已经按 scalar_t 舍入），保证与"先加再归一化"的非融合实现一致
    const float v = static_cast<float>(residual[off + c]);
    hidden[off + c] = static_cast<scalar_t>(v * scale * static_cast<float>(weight[c]));
  }
}

void fused_add_rms_norm(at::Tensor& hidden, at::Tensor& residual, const at::Tensor& weight, double eps) {
  TORCH_CHECK(hidden.is_cuda() && residual.is_cuda() && weight.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(hidden.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(hidden.sizes() == residual.sizes(), "hidden and residual must have the same shape");
  TORCH_CHECK(hidden.scalar_type() == residual.scalar_type() && hidden.scalar_type() == weight.scalar_type(),
              "all tensors must have the same dtype");
  const int n = static_cast<int>(hidden.size(-1));
  TORCH_CHECK(weight.numel() == n, "weight must have hidden_size elements");
  const int rows = static_cast<int>(hidden.numel() / n);
  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(hidden.device());          // 在张量所在的设备上启动
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); // 使用 PyTorch 的当前流
  constexpr int kThreads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, hidden.scalar_type(),
                                  "fused_add_rms_norm", [&] {
    fused_add_rms_norm_kernel<scalar_t, kThreads><<<rows, kThreads, 0, stream>>>(
        hidden.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), n,
        static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// 声明算子的 schema：Tensor(a!) 表示这个参数会被原地修改
TORCH_LIBRARY(handbook, m) {
  m.def("fused_add_rms_norm(Tensor(a!) hidden, Tensor(b!) residual, Tensor weight, float eps) -> ()");
}
// 注册 CUDA 实现
TORCH_LIBRARY_IMPL(handbook, CUDA, m) { m.impl("fused_add_rms_norm", &fused_add_rms_norm); }

// 让它同时是一个可 import 的 Python 模块（import 时完成上面的注册）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
