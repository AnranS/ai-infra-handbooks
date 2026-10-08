#include <torch/extension.h>

#include <cmath>

// y = x / sqrt(mean(x^2) + eps) * weight，沿最后一维归一化
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.device().is_cpu() && weight.device().is_cpu(), "这个示例只实现了 CPU 版本");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32, "只支持 float32");
  TORCH_CHECK(x.size(-1) == weight.size(0), "weight 的长度必须等于 x 的最后一维");

  auto xc = x.contiguous();      // 按指针遍历之前，先保证内存是连续的
  auto wc = weight.contiguous(); // 必须存进变量：临时张量在语句结束时就释放了
  auto y = torch::empty_like(xc);
  const int64_t d = xc.size(-1), rows = xc.numel() / d;
  const float* px = xc.data_ptr<float>();
  const float* pw = wc.data_ptr<float>();
  float* py = y.data_ptr<float>();

  at::parallel_for(0, rows, 16, [&](int64_t begin, int64_t end) {   // 用 PyTorch 自己的 CPU 线程池
    for (int64_t r = begin; r < end; ++r) {
      const float* row = px + r * d;
      double ss = 0;
      for (int64_t i = 0; i < d; ++i) ss += double(row[i]) * row[i];
      const float inv = static_cast<float>(1.0 / std::sqrt(ss / d + eps));
      for (int64_t i = 0; i < d; ++i) py[r * d + i] = row[i] * inv * pw[i];
    }
  });
  return y;
}

// 注册成 torch.ops.demo.rmsnorm：有明确的 schema，torch.compile、CUDA Graph 都能识别
TORCH_LIBRARY(demo, m) { m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor"); }
TORCH_LIBRARY_IMPL(demo, CPU, m) { m.impl("rmsnorm", &rmsnorm); }

// 同时也用 pybind11 暴露一个普通函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rmsnorm", &rmsnorm, "RMSNorm（CPU）"); }
