# 生态：cuBLAS、CUTLASS 与 PyTorch 扩展

<p class="lead">实际工作中，大部分时候不是从零手写 kernel，而是选对库、用好库，在库不够用的地方写定制 kernel，再把它接入 PyTorch 或推理引擎。这一章介绍 CUDA 生态里最重要的几个库，重点讲 CUTLASS/CuTe 的核心概念，以及如何把自己的 kernel 注册成 PyTorch 算子。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. cuBLAS 是列主序的，怎么用它计算行主序的矩阵乘法？
    2. cuBLASLt 比 cuBLAS 多了什么？
    3. CuTe 的 Layout 由哪两部分组成？`(4,8):(8,1)` 是什么意思？
    4. 把 CUDA kernel 接入 PyTorch 有哪几种方式？哪种能和 torch.compile 配合？
    5. 什么时候该用 CUTLASS，什么时候该用 Triton？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 行主序的 $C = AB$ 在内存里等于列主序的 $C^\top = B^\top A^\top$：把 A、B 对调传给 cuBLAS（把行主序的 B 当作列主序的 $B^\top$），得到的列主序结果正好就是行主序的 C。
    2. 更灵活的矩阵乘接口：支持 FP8 等更多数据类型和缩放因子、融合的 epilogue（偏置、激活、量化）、更多的布局选项，还能按问题规模搜索算法（heuristics）。
    3. Shape 和 Stride：`(4,8):(8,1)` 表示形状 4×8，第 0 维每走一步跨 8 个元素、第 1 维跨 1 个，即 4×8 的行主序布局。
    4. `torch.utils.cpp_extension`（`load_inline` / setup.py 编译）、pybind11 直接暴露、`TORCH_LIBRARY` 注册成算子（或 Python 里的 `torch.library.custom_op`）。只有注册成算子（带 schema 和 fake 实现）才能被 torch.compile 捕获而不断图。
    5. 要极致的 GEMM 性能、要用最新的硬件特性（TMA、wgmma、warp 专门化）或者要和 cuBLAS 比肩时用 CUTLASS；要快速写出融合算子、性能差一点也能接受时用 Triton。先用库，再用 Triton，最后才攻坚。

## 库的全景

![图：CUDA 库的全景——调库、模板库、DSL、手写](../assets/figures/library-map.svg){.aig-svg}

| 库 | 用途 | 说明 |
| --- | --- | --- |
| **cuBLAS / cuBLASLt** | 稠密线性代数，GEMM | 闭源，性能标杆；Lt 版本支持更多数据类型和融合 |
| **cuDNN** | 卷积、归一化、注意力 | cuDNN 的 frontend API 提供融合的 SDPA（注意力） |
| **CUTLASS / CuTe** | 可定制的 GEMM、卷积、注意力模板库 | 开源、header-only；FlashAttention-3、众多推理引擎的 GEMM 基于它 |
| **CUB / Thrust / libcu++** | 并行原语、STL 风格算法、C++ 标准库的设备端实现 | 随 CUDA Toolkit 发布（统称 CCCL） |
| **NCCL** | 多 GPU 集合通信 | 见[多 GPU](multi-gpu.md) |
| **FlashAttention / FlashInfer** | 注意力 kernel | 推理引擎的注意力后端 |
| **Triton** | 块级 GPU 编程语言 | 见 [Triton](triton.md) |

## cuBLAS

cuBLAS 沿用 Fortran BLAS 的**列主序**约定。C/C++ 里的行主序矩阵，在 cuBLAS 看来正好是它的转置。计算行主序的 $C = AB$，等价于计算列主序的 $C^\top = B^\top A^\top$，所以把 A、B 的顺序对调即可，不需要真的转置数据：

```cuda title="cublas_gemm.cu"
// cublas_gemm.cu —— 用 cublasGemmEx 计算行主序的 FP16 GEMM（FP32 累加、FP32 输出）
// 编译：nvcc -O3 -arch=sm_75 cublas_gemm.cu -o cublas_gemm -lcublas
#include "common.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUBLAS_CHECK(call)                                                           \
  do {                                                                               \
    cublasStatus_t s_ = (call);                                                      \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                               \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                                \
  } while (0)

int main() {
  const int M = 512, N = 384, K = 256;
  std::vector<float> fa(M * K), fb(K * N), ref(M * N, 0.f), got(M * N);
  fill_random(fa, 1);
  fill_random(fb, 2);
  std::vector<half> ha(fa.size()), hb(fb.size());
  for (size_t i = 0; i < fa.size(); ++i) { ha[i] = __float2half(fa[i]); fa[i] = __half2float(ha[i]); }
  for (size_t i = 0; i < fb.size(); ++i) { hb[i] = __float2half(fb[i]); fb[i] = __half2float(hb[i]); }
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j) ref[i * N + j] += fa[i * K + k] * fb[k * N + j];

  half *dA, *dB;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dA, ha.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, hb.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC, got.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, ha.data(), ha.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.f, beta = 0.f;
  // 行主序 C(MxN) = A(MxK) B(KxN)  <=>  列主序 C^T(NxM) = B^T(NxK) A^T(KxM)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            dB, CUDA_R_16F, N,     // 第一个矩阵传 B，leading dimension 是 B 的行长 N
                            dA, CUDA_R_16F, K,     // 第二个矩阵传 A，leading dimension 是 K
                            &beta, dC, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), got.size(), 1e-3f, 1e-3f);
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return ok ? 0 : 1;
}
```

使用要点：

- **handle 要复用**：`cublasCreate` 开销很大，一个线程/流一个 handle，设置好流（`cublasSetStream`）后反复使用；
- `CUBLAS_COMPUTE_32F_FAST_TF32` 等计算类型允许用 TF32 Tensor Core 计算 FP32 GEMM；
- **cuBLASLt**（`cublasLtMatmul`）是更灵活的接口：支持 FP8（带缩放因子）、**epilogue 融合**（在 GEMM 写回之前顺便加偏置、做 GELU/ReLU、输出辅助结果），以及启发式地列出多种算法供你挑选。PyTorch 的 `torch._scaled_mm`（FP8 GEMM）就是基于它。

## CUTLASS 与 CuTe

CUTLASS 是 NVIDIA 开源的 C++ 模板库，把高性能 GEMM 拆成可以组合的层次：

```text
Device 层    ：启动配置、分块调度（包括 stream-K、持久化 kernel）
Kernel 层    ：一个 block 的主循环
Collective 层：主循环（mainloop：TMA/cp.async 流水 + MMA）与尾声（epilogue：缩放、激活、写回）
Atom 层      ：一条 MMA 指令（mma.sync / wgmma / tcgen05）或一次拷贝（ldmatrix / TMA）
```

CUTLASS 3.x 以后，这一切建立在 **CuTe** 之上。CuTe 的核心是一个概念：**Layout（布局）= Shape（形状）: Stride（跨度）**，它把逻辑坐标映射到内存下标。线程到数据的分配、分块、swizzle、Tensor Core 的寄存器布局，全都用同一套布局代数表达。

下面这个程序只在主机端运行（不需要 GPU），用来体会布局代数。输出是在编写本手册时实际运行得到的：

```cuda title="cute_layout.cu"
// cute_layout.cu —— CuTe 的布局代数：只在主机端运行，不需要 GPU
// 编译：nvcc -std=c++17 -I${CUTLASS_DIR}/include cute_layout.cu -o cute_layout
#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
  // 1. Layout = Shape : Stride，把逻辑坐标映射成内存下标
  auto col_major = make_layout(make_shape(4, 8));                  // 默认列主序
  auto row_major = make_layout(make_shape(4, 8), LayoutRight{});   // 行主序
  print(col_major); printf("\n");
  print(row_major); printf("\n");
  printf("(1,2) -> col_major %d, row_major %d\n\n", int(col_major(1, 2)), int(row_major(1, 2)));

  // 2. 编译期常量用 Int<N>{}，打印时带下划线
  auto tile = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{});
  print_layout(tile);

  // 3. 分块：把 8x8 的行主序矩阵切成 4x4 的块，得到 ((块内), (块号)) 的层次化布局
  auto mat = make_layout(make_shape(Int<8>{}, Int<8>{}), LayoutRight{});
  auto tiled = zipped_divide(mat, make_shape(Int<4>{}, Int<4>{}));
  print(tiled); printf("\n");
  printf("block (1,0), element (2,3) -> %d\n\n", int(tiled(make_coord(make_coord(2, 3), make_coord(1, 0)))));

  // 4. swizzle：第 r 行的列号与 r 做异或，打散 bank
  auto swizzled = composition(Swizzle<2, 0, 3>{}, tile);
  print_layout(swizzled);
  return 0;
}
```

```text
(4,8):(_1,4)
(4,8):(8,_1)
(1,2) -> col_major 9, row_major 10

(_4,_8):(_8,_1)
       0    1    2    3    4    5    6    7
    +----+----+----+----+----+----+----+----+
 0  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
    +----+----+----+----+----+----+----+----+
 1  |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
    +----+----+----+----+----+----+----+----+
 2  | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
    +----+----+----+----+----+----+----+----+
 3  | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
    +----+----+----+----+----+----+----+----+
((_4,_4),(_2,_2)):((_8,_1),(_32,_4))
block (1,0), element (2,3) -> 51

Sw<2,0,3> o _0 o (_4,_8):(_8,_1)
       0    1    2    3    4    5    6    7
    +----+----+----+----+----+----+----+----+
 0  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
    +----+----+----+----+----+----+----+----+
 1  |  9 |  8 | 11 | 10 | 13 | 12 | 15 | 14 |
    +----+----+----+----+----+----+----+----+
 2  | 18 | 19 | 16 | 17 | 22 | 23 | 20 | 21 |
    +----+----+----+----+----+----+----+----+
 3  | 27 | 26 | 25 | 24 | 31 | 30 | 29 | 28 |
    +----+----+----+----+----+----+----+----+
```

怎么读这些输出：

- `(4,8):(_1,4)`：形状 4×8，第 0 维跨度 1、第 1 维跨度 4，即列主序。坐标 (1, 2) 映射到 1×1 + 2×4 = 9；行主序 `(4,8):(8,_1)` 下映射到 1×8 + 2 = 10。
- `zipped_divide` 的结果 `((_4,_4),(_2,_2)):((_8,_1),(_32,_4))` 是一个**层次化布局**：第一个模式是 4×4 块内的坐标（跨度 8 和 1，就是原矩阵的行和列），第二个模式是 2×2 的块号（跨度 32 = 4 行 × 8，以及 4 列）。块 (1, 0) 中的元素 (2, 3) 对应原矩阵的第 6 行第 3 列，下标 51。GEMM 里"每个 block 取 C 的一块"就是这样表达的。
- swizzle 之后，第 r 行的列号与 r 异或：第 1 行的 0、1 对调，第 2 行的 0-1 与 2-3 对调……同一列的不同行被分散到不同的位置，这就是[共享内存](../basics/memory.md#bank-冲突)一章提到的消除 bank 冲突的方式。

布局代数本身（合并、复合、补集、划分，线程划分与 MMA 的 TV 布局）在 [CuTe 的布局代数](../advanced/cute-layout.md) 一章里有完整的推导和可运行的实现。学习 CuTe 的建议路径：先读 CUTLASS 仓库里 `media/docs/cpp/cute/` 下的教程（从 layout、layout algebra、tensor 到 MMA atom 和 TiledCopy），再读 `examples/cute/tutorial/` 下的 sgemm 示例，最后读 Hopper/Blackwell 的 GEMM 示例。CUTLASS 4.x 还提供了 **CuTe DSL**，可以用 Python 写 CuTe kernel，编译速度快得多。

## CUB、Thrust 与 libcu++

- **CUB**：块级和设备级的并行原语（归约、扫描、排序、直方图、选择），比如 `cub::BlockReduce`、`cub::DeviceRadixSort`。在自己的 kernel 里需要一个 block 内的归约或扫描时，`cub::BlockReduce` 比手写更可靠；
- **Thrust**：STL 风格的高层算法，`thrust::sort(thrust::device, v.begin(), v.end())`，适合快速实现和原型；
- **libcu++**：C++ 标准库的设备端实现，`cuda::std::atomic`、`cuda::barrier`、`cuda::memcpy_async`、`cuda::ptx` 等。[TMA 示例](../advanced/async-hopper.md#tma张量内存加速器-sm_90)就用到了它。

## 把 kernel 接入 PyTorch

三种方式，从快到正式：

| 方式 | 适合 |
| --- | --- |
| `torch.utils.cpp_extension.load_inline` | 在 Python 里直接写 C++/CUDA 源码字符串，第一次调用时即时编译，适合实验 |
| `setup.py` + `CUDAExtension` | 正式的扩展包，预先编译 |
| `TORCH_LIBRARY` 注册算子（配合上面任意一种构建方式） | 算子进入 PyTorch 的调度系统，用 `torch.ops.<命名空间>.<名字>` 调用，**能被 torch.compile 和 CUDA Graphs 正确处理** |

下面把[Softmax 与归一化](../kernels/softmax-norm.md#算子融合fused_add_rms_norm)一章的 `fused_add_rms_norm` 注册成 PyTorch 算子，支持 FP32、FP16、BF16：

```cuda title="fused_add_rms_norm_ext.cu"
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
```

构建和使用：

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="handbook_ops",
    ext_modules=[CUDAExtension("handbook_ops", ["fused_add_rms_norm_ext.cu"],
                               extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]})],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# pip install --no-build-isolation -e .   然后：
import torch
import handbook_ops  # noqa: F401  导入即注册

@torch.library.register_fake("handbook::fused_add_rms_norm")
def _(hidden, residual, weight, eps):   # 告诉 torch.compile 这个算子不返回新张量
    return None

hidden = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(hidden)
weight = torch.rand(4096, device="cuda", dtype=torch.bfloat16)

ref_res = residual + hidden
ref_out = (ref_res.float() * torch.rsqrt(ref_res.float().pow(2).mean(-1, keepdim=True) + 1e-6)
           * weight.float()).to(torch.bfloat16)
torch.ops.handbook.fused_add_rms_norm(hidden, residual, weight, 1e-6)
torch.testing.assert_close(residual, ref_res)
torch.testing.assert_close(hidden, ref_out, rtol=2e-2, atol=2e-2)
```

要点：

- 用 `c10::cuda::getCurrentCUDAStream()` 获取当前流，而不是默认流，否则会和 PyTorch 的流语义冲突，也无法被 CUDA Graphs 捕获；
- 用 `CUDAGuard` 切换到张量所在的设备，多卡时不会出错；
- `AT_DISPATCH_*` 宏根据张量的 dtype 实例化对应的模板；`c10::Half`、`c10::BFloat16` 可以和 float 互相 `static_cast`；
- schema 里正确标注原地修改（`Tensor(a!)`），torch.compile 才能正确处理别名和依赖；返回新张量的算子还需要注册 fake（meta）实现，让编译器在不执行 kernel 的情况下推断输出形状；
- 较新版本的 PyTorch 要求用 C++20 编译扩展（本手册验证时使用的 2.14 就是如此），`BuildExtension` 会自动加上对应的编译选项，自己写 CMake 时要注意；
- vLLM、SGLang 的 `csrc/` 目录里就是大量这样注册的算子，读它们的源码是很好的练习。

## 怎么选

| 需求 | 选择 |
| --- | --- |
| 标准 GEMM、批量 GEMM | cuBLAS / cuBLASLt |
| GEMM + 特殊的 epilogue、特殊的数据类型组合、分组 GEMM | CUTLASS（或 Triton） |
| 逐元素 / 归约类融合算子，需要快速迭代 | Triton，或 torch.compile 自动生成 |
| 注意力 | FlashAttention / FlashInfer / cuDNN SDPA |
| 需要 warp 级精细控制、最新硬件特性、极致性能 | CUDA + CUTLASS/CuTe |
| 排序、扫描、选择等通用并行原语 | CUB / Thrust |

!!! interview "面试怎么答"
    生态题：cuBLAS 是列主序，行主序的 $C = AB$ 通过对调 A、B 算 $C^\top = B^\top A^\top$ 实现，handle 要复用，cuBLASLt 支持 FP8 和 epilogue 融合；CUTLASS 把 GEMM 拆成 device、kernel、collective、atom 几层，CuTe 用 Shape:Stride 的布局代数统一描述数据和线程的映射；把 kernel 接进 PyTorch 用 `TORCH_LIBRARY` 注册，使用当前流和 `CUDAGuard`，如实标注原地修改并提供 fake 实现，才能和 torch.compile 配合。选型：库优先，Triton 提效，CUDA / CUTLASS 攻坚。

## 练习

**1. 列主序技巧。** 如果要用 cuBLAS 计算行主序的 $C = A^\top B$（A 为 K×M，B 为 K×N），`cublasGemmEx` 的前几个参数应该怎么写？

??? success "参考答案"
    行主序的 $C = A^\top B$ 等价于列主序的 $C^\top = B^\top A$。在 cuBLAS 看来，行主序的 B（K×N）就是列主序的 $B^\top$（N×K，leading dimension N），不需要转置，用 `CUBLAS_OP_N`；行主序的 A（K×M）在 cuBLAS 看来是列主序的 $A^\top$（M×K，leading dimension M），而我们需要的是 A（K×M），所以对它用 `CUBLAS_OP_T`。调用为 `cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, ..., N, A, ..., M, &beta, C, ..., N, ...)`。

**2. 读 CuTe。** 布局 `((_2,_4),_8):((_1,_16),_2)` 的形状是多少？坐标 ((1, 2), 3) 映射到哪个下标？

??? success "参考答案"
    这是一个两个模式的布局：第一个模式是 (2, 4)，第二个模式是 8，总共 2 × 4 × 8 = 64 个元素。坐标 ((1, 2), 3) 的下标 = 1 × 1 + 2 × 16 + 3 × 2 = 39。CuTe 里层次化的形状和跨度就是这样按对应位置相乘再求和的。

## 小结

- [x] cuBLAS 是列主序，行主序 GEMM 通过对调 A、B 实现；handle 复用；cuBLASLt 支持 FP8 和 epilogue 融合。
- [x] CUTLASS 把 GEMM 拆成 device、kernel、collective、atom 层；CuTe 用 Shape:Stride 布局代数统一表达数据与线程的映射。
- [x] CUB/Thrust/libcu++ 提供并行原语和设备端标准库。
- [x] 接入 PyTorch 用 TORCH_LIBRARY 注册，使用当前流和 CUDAGuard，正确标注原地修改，必要时注册 fake 实现。
- [x] 按需求选工具：库优先，Triton 提效，CUDA/CUTLASS 攻坚。
