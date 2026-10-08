// cuda_graph.cu —— 用流捕获把大量小 kernel 录成 CUDA Graph，对比逐个启动的开销
// 编译：nvcc -O3 -arch=sm_75 cuda_graph.cu -o cuda_graph
#include "common.cuh"

__global__ void add_one(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.f;
}

int main() {
  const int n = 4096, kernels_per_step = 200, steps = 20;
  float* d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // 1) 逐个启动
  GpuTimer t;
  t.start(s);
  for (int step = 0; step < steps; ++step)
    for (int k = 0; k < kernels_per_step; ++k) add_one<<<(n + 255) / 256, 256, 0, s>>>(d, n);
  float eager_ms = t.stop(s);
  CUDA_CHECK_LAST();

  // 2) 捕获一个 step 的 200 个 kernel，之后每个 step 只启动一次图
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int k = 0; k < kernels_per_step; ++k) add_one<<<(n + 255) / 256, 256, 0, s>>>(d, n);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  t.start(s);
  for (int step = 0; step < steps; ++step) CUDA_CHECK(cudaGraphLaunch(exec, s));
  float graph_ms = t.stop(s);

  std::vector<float> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  const float expect = 2.f * steps * kernels_per_step;   // 两种方式各执行了 steps * kernels_per_step 次加一
  size_t bad = 0;
  for (float v : h) bad += v != expect;
  std::printf("%s (value %.0f, expected %.0f)\neager: %.3f ms\ngraph: %.3f ms\n", bad ? "FAIL" : "PASS", h[0],
              expect, eager_ms, graph_ms);
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(d));
  return bad ? 1 : 0;
}
