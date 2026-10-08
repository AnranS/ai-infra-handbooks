# The ecosystem: cuBLAS, CUTLASS and PyTorch extensions

<p class="lead">In real work, most of the time is not spent writing kernels from scratch: it is choosing the right library and using it well, writing a custom kernel where the library falls short, and wiring that into PyTorch or an inference engine. This chapter introduces the most important libraries in the CUDA ecosystem, with the focus on CUTLASS/CuTe's core concepts and on how to register your own kernel as a PyTorch operator.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. cuBLAS is column-major. How do you use it to compute a row-major matrix multiply?
    2. What does cuBLASLt add over cuBLAS?
    3. What two parts make up a CuTe Layout? What does `(4,8):(8,1)` mean?
    4. What are the ways to wire a CUDA kernel into PyTorch? Which one works with torch.compile?
    5. When should you use CUTLASS and when Triton?

??? success "Answers (try it yourself first, then expand)"
    1. A row-major $C = AB$ is, in memory, exactly a column-major $C^\top = B^\top A^\top$: pass A and B to cuBLAS the other way round (treating the row-major B as a column-major $B^\top$), and the column-major result is precisely the row-major C.
    2. A more flexible matrix multiply interface: more data types including FP8 and their scaling factors, a fused epilogue (bias, activation, quantization), more layout options, and the ability to search for an algorithm by problem size (heuristics).
    3. Shape and Stride: `(4,8):(8,1)` means shape 4×8, with a step along mode 0 skipping 8 elements and a step along mode 1 skipping 1, that is a 4×8 row-major layout.
    4. `torch.utils.cpp_extension` (`load_inline` / compiling through setup.py), exposing it directly with pybind11, and registering it as an operator with `TORCH_LIBRARY` (or `torch.library.custom_op` in Python). Only a registered operator (with a schema and a fake implementation) is captured by torch.compile without a graph break.
    5. Use CUTLASS when you need the last word in GEMM performance, the newest hardware features (TMA, wgmma, warp specialization), or to match cuBLAS; use Triton to write a fused operator quickly where slightly less performance is acceptable. Library first, then Triton, and hand-tuning last.

## The library landscape {#库的全景}

![Figure: the CUDA library landscape - call a library, a template library, a DSL, by hand](../assets/figures/library-map.svg){.aig-svg}

| Library | Purpose | Notes |
| --- | --- | --- |
| **cuBLAS / cuBLASLt** | dense linear algebra, GEMM | closed source, the performance benchmark; the Lt version supports more data types and fusion |
| **cuDNN** | convolution, normalization, attention | cuDNN's frontend API offers a fused SDPA (attention) |
| **CUTLASS / CuTe** | a customizable template library for GEMM, convolution and attention | open source, header-only; FlashAttention-3 and many inference engines' GEMMs are built on it |
| **CUB / Thrust / libcu++** | parallel primitives, STL-style algorithms, a device-side implementation of the C++ standard library | shipped with the CUDA Toolkit (collectively CCCL) |
| **NCCL** | multi-GPU collective communication | see [multi-GPU](multi-gpu.md) |
| **FlashAttention / FlashInfer** | attention kernels | the attention back ends of inference engines |
| **Triton** | a block-level GPU programming language | see [Triton](triton.md) |

## cuBLAS {#cublas}

cuBLAS keeps Fortran BLAS's **column-major** convention. A row-major matrix in C/C++ is, as far as cuBLAS is concerned, exactly its transpose. Computing a row-major $C = AB$ is equivalent to computing a column-major $C^\top = B^\top A^\top$, so swapping the order of A and B is enough and the data never has to be transposed:

```cuda title="cublas_gemm.cu"
// cublas_gemm.cu - a row-major FP16 GEMM with cublasGemmEx (FP32 accumulation, FP32 output)
// build: nvcc -O3 -arch=sm_75 cublas_gemm.cu -o cublas_gemm -lcublas
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
  // row-major C(MxN) = A(MxK) B(KxN)  <=>  column-major C^T(NxM) = B^T(NxK) A^T(KxM)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            dB, CUDA_R_16F, N,     // the first matrix is B, whose leading dimension is B's row length N
                            dA, CUDA_R_16F, K,     // the second matrix is A, whose leading dimension is K
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

Points to keep in mind:

- **reuse the handle**: `cublasCreate` is expensive, so keep one handle per thread/stream, set its stream (`cublasSetStream`) and use it over and over;
- compute types like `CUBLAS_COMPUTE_32F_FAST_TF32` allow an FP32 GEMM to be computed on the TF32 Tensor Cores;
- **cuBLASLt** (`cublasLtMatmul`) is the more flexible interface: it supports FP8 (with scaling factors), **epilogue fusion** (adding a bias, doing a GELU/ReLU or producing an auxiliary output on the way to writing the GEMM back), and listing several algorithms heuristically for you to choose from. PyTorch's `torch._scaled_mm` (FP8 GEMM) is built on it.

## CUTLASS and CuTe {#cutlass-与-cute}

CUTLASS is NVIDIA's open-source C++ template library, which breaks a high-performance GEMM into composable levels:

<!-- i18n:diagram 5ac678e8df -->
```text
Device level    : the launch configuration and tile scheduling (including stream-K and persistent kernels)
Kernel level    : one block's main loop
Collective level: the main loop (mainloop: a TMA/cp.async pipeline + MMA) and the epilogue (scaling, activation, write-back)
Atom level      : one MMA instruction (mma.sync / wgmma / tcgen05) or one copy (ldmatrix / TMA)
```

From CUTLASS 3.x onward, all of this is built on **CuTe**. CuTe's core is one concept: **a Layout = a Shape : a Stride**, mapping a logical coordinate to a memory index. The assignment of threads to data, tiling, swizzling and the Tensor Cores' register layouts are all expressed in the same layout algebra.

The program below runs on the host only (no GPU needed) and is there to get a feel for the layout algebra. Its output was produced by actually running it while this handbook was written:

```cuda title="cute_layout.cu"
// cute_layout.cu - CuTe's layout algebra: host only, no GPU needed
// build: nvcc -std=c++17 -I${CUTLASS_DIR}/include cute_layout.cu -o cute_layout
#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
  // 1. a Layout = a Shape : a Stride, mapping a logical coordinate to a memory index
  auto col_major = make_layout(make_shape(4, 8));                  // column-major by default
  auto row_major = make_layout(make_shape(4, 8), LayoutRight{});   // row-major
  print(col_major); printf("\n");
  print(row_major); printf("\n");
  printf("(1,2) -> col_major %d, row_major %d\n\n", int(col_major(1, 2)), int(row_major(1, 2)));

  // 2. compile-time constants are Int<N>{}, printed with a leading underscore
  auto tile = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{});
  print_layout(tile);

  // 3. tiling: cutting the 8x8 row-major matrix into 4x4 tiles gives a hierarchical ((within a tile), (tile number)) layout
  auto mat = make_layout(make_shape(Int<8>{}, Int<8>{}), LayoutRight{});
  auto tiled = zipped_divide(mat, make_shape(Int<4>{}, Int<4>{}));
  print(tiled); printf("\n");
  printf("block (1,0), element (2,3) -> %d\n\n", int(tiled(make_coord(make_coord(2, 3), make_coord(1, 0)))));

  // 4. the swizzle: xor row r's column number with r, scattering the banks
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

How to read that output:

- `(4,8):(_1,4)`: shape 4×8, stride 1 along mode 0 and 4 along mode 1, that is column-major. The coordinate (1, 2) maps to 1×1 + 2×4 = 9; under the row-major `(4,8):(8,_1)` it maps to 1×8 + 2 = 10.
- `zipped_divide`'s result `((_4,_4),(_2,_2)):((_8,_1),(_32,_4))` is a **hierarchical layout**: the first mode is the coordinate within a 4×4 tile (strides 8 and 1, the original matrix's row and column), and the second is the tile number (stride 32 = 4 rows × 8, and 4 columns). Element (2, 3) of tile (1, 0) is the original matrix's row 6, column 3, index 51. "Each block takes one tile of C" in a GEMM is expressed exactly this way.
- After the swizzle, row r's column number is xored with r: row 1 swaps 0 and 1, row 2 swaps 0-1 with 2-3, and so on. Different rows of the same column are scattered to different places, which is how the [shared memory](../basics/memory.md#bank-冲突) chapter removed bank conflicts.

The layout algebra itself (coalescing, composition, complement, division, the thread partition and an MMA's TV layout) is derived in full, with a runnable implementation, in [CuTe's layout algebra](../advanced/cute-layout.md). A suggested path for learning CuTe: read the tutorials under `media/docs/cpp/cute/` in the CUTLASS repository first (from layout, layout algebra and tensor to the MMA atom and TiledCopy), then the sgemm examples under `examples/cute/tutorial/`, and finally the Hopper/Blackwell GEMM examples. CUTLASS 4.x also provides a **CuTe DSL** for writing CuTe kernels in Python, which compiles far faster.

## CUB, Thrust and libcu++ {#cubthrust-与-libcu}

- **CUB**: block-level and device-level parallel primitives (reduction, scan, sort, histogram, selection), such as `cub::BlockReduce` and `cub::DeviceRadixSort`. When your own kernel needs a reduction or a scan within a block, `cub::BlockReduce` is more reliable than writing one;
- **Thrust**: STL-style high-level algorithms, `thrust::sort(thrust::device, v.begin(), v.end())`, good for getting something working and for prototypes;
- **libcu++**: a device-side implementation of the C++ standard library: `cuda::std::atomic`, `cuda::barrier`, `cuda::memcpy_async`, `cuda::ptx` and so on. The [TMA example](../advanced/async-hopper.md#tma张量内存加速器-sm_90) used it.

## Wiring a kernel into PyTorch {#把-kernel-接入-pytorch}

Three ways, from quickest to most formal:

| Way | Suits |
| --- | --- |
| `torch.utils.cpp_extension.load_inline` | write the C++/CUDA source as a string in Python, compiled just in time on the first call; good for experiments |
| `setup.py` + `CUDAExtension` | a proper extension package, compiled ahead of time |
| registering an operator with `TORCH_LIBRARY` (with either build method above) | the operator enters PyTorch's dispatch system, called as `torch.ops.<namespace>.<name>`, and **is handled correctly by torch.compile and CUDA Graphs** |

Below, the `fused_add_rms_norm` from [softmax and normalization](../kernels/softmax-norm.md#算子融合fused_add_rms_norm) is registered as a PyTorch operator supporting FP32, FP16 and BF16:

```cuda title="fused_add_rms_norm_ext.cu"
// fused_add_rms_norm_ext.cu - registered as the PyTorch custom operator torch.ops.handbook.fused_add_rms_norm
// build: see the setup.py below (CUDAExtension); torch.utils.cpp_extension.load compiles it just in time instead
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

// residual += hidden; hidden = rms_norm(residual) * weight. The statistics are computed in FP32
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
    // read back the residual just written (already rounded to scalar_t), to match the unfused "add then normalize" exactly
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

  const c10::cuda::CUDAGuard guard(hidden.device());          // launch on the device the tensors are on
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); // use PyTorch's current stream
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

// declare the operator's schema: Tensor(a!) says this argument is modified in place
TORCH_LIBRARY(handbook, m) {
  m.def("fused_add_rms_norm(Tensor(a!) hidden, Tensor(b!) residual, Tensor weight, float eps) -> ()");
}
// register the CUDA implementation
TORCH_LIBRARY_IMPL(handbook, CUDA, m) { m.impl("fused_add_rms_norm", &fused_add_rms_norm); }

// also make it an importable Python module (the import performs the registration above)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
```

Building and using it:

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
# pip install --no-build-isolation -e .   then:
import torch
import handbook_ops  # noqa: F401  importing registers it

@torch.library.register_fake("handbook::fused_add_rms_norm")
def _(hidden, residual, weight, eps):   # tell torch.compile this operator returns no new tensor
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

The key points:

- take the current stream with `c10::cuda::getCurrentCUDAStream()` rather than the default stream, or it conflicts with PyTorch's stream semantics and cannot be captured by CUDA Graphs;
- switch to the tensor's device with a `CUDAGuard`, so nothing goes wrong on multiple GPUs;
- the `AT_DISPATCH_*` macros instantiate the matching template for the tensor's dtype; `c10::Half` and `c10::BFloat16` `static_cast` to and from float;
- declare in-place modification correctly in the schema (`Tensor(a!)`), or torch.compile cannot handle the aliasing and the dependencies; an operator returning a new tensor also needs a fake (meta) implementation registered, so the compiler can infer the output's shape without running the kernel;
- newer versions of PyTorch require C++20 to compile an extension (the 2.14 used to verify this handbook does), and `BuildExtension` adds the right compile options automatically, which is worth remembering when writing your own CMake;
- vLLM's and SGLang's `csrc/` directories are full of operators registered just like this, and reading their source is good practice.

## How to choose {#怎么选}

| Need | Choice |
| --- | --- |
| a standard GEMM, a batched GEMM | cuBLAS / cuBLASLt |
| GEMM with a special epilogue, an unusual data type combination, a grouped GEMM | CUTLASS (or Triton) |
| a fused elementwise / reduction operator, with fast iteration | Triton, or generated automatically by torch.compile |
| attention | FlashAttention / FlashInfer / cuDNN SDPA |
| fine warp-level control, the newest hardware features, the last word in performance | CUDA + CUTLASS/CuTe |
| general parallel primitives like sort, scan and selection | CUB / Thrust |

!!! interview "How to explain it"
    On the ecosystem: cuBLAS is column-major, so a row-major $C = AB$ is done by swapping A and B to compute $C^\top = B^\top A^\top$, the handle is reused, and cuBLASLt supports FP8 and epilogue fusion; CUTLASS breaks a GEMM into the device, kernel, collective and atom levels, and CuTe describes the mapping of both data and threads in one Shape:Stride layout algebra; wiring a kernel into PyTorch means registering it with `TORCH_LIBRARY`, using the current stream and a `CUDAGuard`, declaring in-place modification honestly and providing a fake implementation, which is what makes it work with torch.compile. Choosing: library first, Triton for productivity, CUDA / CUTLASS for the hard cases.

## Exercises {#练习}

**1. The column-major trick.** To compute the row-major $C = A^\top B$ with cuBLAS (A being K×M and B being K×N), how should `cublasGemmEx`'s first few arguments read?

??? success "Answer"
    The row-major $C = A^\top B$ is equivalent to the column-major $C^\top = B^\top A$. As far as cuBLAS is concerned, the row-major B (K×N) is already the column-major $B^\top$ (N×K, leading dimension N) and needs no transpose, so `CUBLAS_OP_N`; the row-major A (K×M) looks to cuBLAS like the column-major $A^\top$ (M×K, leading dimension M) while what we need is A (K×M), so it takes `CUBLAS_OP_T`. The call is `cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, ..., N, A, ..., M, &beta, C, ..., N, ...)`.

**2. Reading CuTe.** What is the shape of the layout `((_2,_4),_8):((_1,_16),_2)`? Which index does the coordinate ((1, 2), 3) map to?

??? success "Answer"
    This is a layout of two modes: the first is (2, 4) and the second is 8, for 2 × 4 × 8 = 64 elements in all. The coordinate ((1, 2), 3) has index 1 × 1 + 2 × 16 + 3 × 2 = 39. A hierarchical shape and stride in CuTe multiply position by position and sum, exactly like this.

## Summary {#小结}

- [x] cuBLAS is column-major and a row-major GEMM comes from swapping A and B; reuse the handle; cuBLASLt supports FP8 and epilogue fusion.
- [x] CUTLASS breaks a GEMM into the device, kernel, collective and atom levels; CuTe expresses the mapping of both data and threads in one Shape:Stride layout algebra.
- [x] CUB/Thrust/libcu++ provide parallel primitives and a device-side standard library.
- [x] Wiring into PyTorch means registering with TORCH_LIBRARY, using the current stream and a CUDAGuard, declaring in-place modification correctly, and registering a fake implementation where needed.
- [x] Pick the tool by the need: library first, Triton for productivity, CUDA/CUTLASS for the hard cases.
