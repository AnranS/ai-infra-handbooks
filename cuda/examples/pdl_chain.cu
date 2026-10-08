// pdl_chain.cu —— Programmatic Dependent Launch：让下一个 kernel 的前奏和上一个 kernel 的收尾重叠
// 编译：nvcc -O3 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 pdl_chain.cu -o pdl_chain
// sm_90 的机器码和 compute_90 的 PTX 带 PDL（Hopper 直接运行，更新的 GPU 即时编译）；更老的 GPU 用 compute_75 的 PTX，
// 不带 PDL，主机端也不打开属性，四种方式照样核对结果
#include "common.cuh"

// 一"层"：out[i] = relu(in[i]) * w[i] + b[i]。w、b 是这一层的权重，不依赖上一层的输出。
// in 故意不加 __restrict__：打开 PDL 时本 kernel 启动后上一层可能还在写它，不能让编译器走只读缓存（ld.global.nc）
__global__ void layer(const float* in, float* out, const float* __restrict__ w, const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float wi = 0.f, bi = 0.f;
  if (i < n) {  // 前奏：只读权重。打开 PDL 时，这一段可以在上一层还没结束时就开始
    wi = w[i];
    bi = b[i];
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();  // 等上一层的所有 block 结束、写入对本 kernel 可见，之后才能读 in
#endif
  if (i < n) out[i] = fmaxf(in[i], 0.f) * wi + bi;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();  // 本 block 的输出写完了：下一层可以提前启动
#endif
}

// 用 cudaLaunchKernelEx 启动；pdl 为真时带上"允许和前一个 kernel 重叠"的属性
void launch_layer(bool pdl, cudaStream_t s, int n, const float* in, float* out, const float* w, const float* b) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((n + 255) / 256);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, layer, in, out, w, b, n));
}

int main() {
  const int n = 1 << 14, layers = 200, iters = 20;  // 每层只有 16K 个元素：kernel 很短，边界开销占大头
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  const bool pdl_ok = p.major >= 9;
  if (!pdl_ok) std::printf("note: PDL needs sm_90+, this GPU is sm_%d%d, so every mode runs without it\n", p.major, p.minor);

  std::vector<float> h_x(n), h_w(static_cast<size_t>(layers) * n), h_b(static_cast<size_t>(layers) * n);
  fill_random(h_x, 1);
  fill_random(h_w, 2, 0.5f, 1.5f);
  fill_random(h_b, 3, -0.1f, 0.1f);
  std::vector<float> ref = h_x, got(n);
  for (int l = 0; l < layers; ++l)
    for (int i = 0; i < n; ++i)
      ref[i] = std::fmax(ref[i], 0.f) * h_w[static_cast<size_t>(l) * n + i] + h_b[static_cast<size_t>(l) * n + i];

  const size_t bytes = n * sizeof(float);
  float *d_x, *d_a, *d_c, *d_w, *d_b;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // 一次"前向"：layers 个 kernel 首尾相连，在 d_a、d_c 之间来回写
  auto run_chain = [&](bool pdl) {
    const float* in = d_x;
    for (int l = 0; l < layers; ++l) {
      float* out = l % 2 == 0 ? d_a : d_c;
      launch_layer(pdl && pdl_ok, s, n, in, out, d_w + static_cast<size_t>(l) * n, d_b + static_cast<size_t>(l) * n);
      in = out;
    }
  };
  float* result = (layers - 1) % 2 == 0 ? d_a : d_c;

  const char* names[4] = {"stream", "stream + PDL", "graph", "graph + PDL"};
  bool all_ok = true;
  for (int mode = 0; mode < 4; ++mode) {
    const bool graph = mode >= 2, pdl = mode % 2 == 1;
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    cudaGraphExec_t exec = nullptr;
    if (graph) {  // 流捕获：带 PDL 属性的启动在图里变成 programmatic 类型的边
      cudaGraph_t g;
      CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      run_chain(pdl);
      CUDA_CHECK(cudaStreamEndCapture(s, &g));
      CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
    }
    auto forward = [&] {
      if (graph) CUDA_CHECK(cudaGraphLaunch(exec, s));
      else run_chain(pdl);
    };
    forward();  // 预热
    GpuTimer t;
    t.start(s);
    for (int it = 0; it < iters; ++it) forward();
    const float us = t.stop(s) * 1e3f / iters;
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), result, bytes, cudaMemcpyDeviceToHost));
    std::printf("%-13s %8.1f us per forward, %6.2f us per layer   ", names[mode], us, us / layers);
    all_ok &= check_close(got.data(), ref.data(), n, 1e-4f, 1e-5f);
    if (exec) CUDA_CHECK(cudaGraphExecDestroy(exec));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  for (float* ptr : {d_x, d_a, d_c, d_w, d_b}) CUDA_CHECK(cudaFree(ptr));
  return all_ok ? 0 : 1;
}
