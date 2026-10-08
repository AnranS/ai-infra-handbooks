# Quantization and GEMV

<p class="lead">Every token an LLM generates during decode reads all the weights from memory, while a small batch does very little arithmetic. At that point "how many bytes you read" essentially is the time. Quantization squeezes the weights from 16 bits to 8 or 4 and cuts that reading directly. This chapter covers the basic quantization formats, how to write a GEMV kernel, and a W4A16 kernel that dequantizes in registers.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why is a linear layer memory-bound during decode at batch 1?
    2. What do W4A16, W8A8 and FP8 mean? What does each suit?
    3. How are the scales and zero points of per-group quantization (group size 128) stored?
    4. How does a kernel unpack and dequantize INT4 weights?
    5. Why does W4A16's speedup fall as the batch grows?

??? success "Answers (try it yourself first, then expand)"
    1. At batch 1 each weight takes part in one multiply-add, an arithmetic intensity of about 1 FLOP/byte in BF16, far below the ridge point; the time is about the weight bytes divided by the bandwidth.
    2. W4A16: 4-bit weights and 16-bit activations, cutting only the weight bytes, which suits small-batch decode; W8A8: 8-bit integer weights and activations with the arithmetic on INT8 Tensor Cores, which suits large batches and prefill; FP8: 8-bit floating point (E4M3) with a better dynamic range than INT8, supported natively from Hopper on.
    3. Every 128 consecutive weights (along the input dimension) form a group with one scale (fp16) and one zero point (packed into 4 bits or kept as fp16), stored separately from the packed weights; dequantization is $w = (q - z) \cdot s$.
    4. Eight 4-bit weights are packed into one 32-bit integer, and the kernel extracts them with shifts and masks, `(x >> 4i) & 0xF`, subtracts the zero point and multiplies by the scale to get an fp16 value in a register. Production kernels also use bit tricks to assemble floats from the integers directly and reorder the weights offline.
    5. As the batch grows the arithmetic intensity rises and the bottleneck moves from reading weights to computing, while W4A16 still computes in 16 bits (with dequantization on top), so the time saved on reading becomes a smaller share and the speedup falls.

## Why decode needs quantization {#为什么-decode-需要量化}

For a linear layer $y = Wx$ with W of N×K, batch 1 makes it a matrix-vector product (GEMV): each weight is read once for one multiply-add, an arithmetic intensity of about one multiply-add per 2 bytes (BF16), far below any GPU's ridge point. **The time is about the weight bytes divided by the memory bandwidth.**

For a 7-billion-parameter model, BF16 weights are about 14 GB, so on an H100 (3.35 TB/s) each token takes at least about 4.2 ms, around 240 tokens/s at most. Squeeze the weights to 4 bits (about 3.5 GB plus a few scales) and in theory it is nearly 4 times faster.

As the batch grows, the same weights serve several requests and the arithmetic intensity rises linearly. At a large enough batch (usually tens to hundreds), the linear layer becomes compute-bound again and compressing the weights alone is no longer enough; the arithmetic has to go low-precision too (INT8/FP8 Tensor Cores).

## Common quantization schemes {#常见的量化方案}

| Scheme | Weights | Activations | Arithmetic | Suits |
| --- | --- | --- | --- | --- |
| W8A16 / W4A16 (weight-only) | INT8 / INT4 | FP16/BF16 | dequantize to FP16 in registers and compute on FP16 Tensor Cores | small-batch decode, memory-bound |
| W8A8 INT8 | INT8 | INT8 | INT8 Tensor Cores with INT32 accumulation | large batches and prefill, compute-bound |
| FP8 (W8A8) | FP8 | FP8 | FP8 Tensor Cores (sm_89+) | the general choice on Hopper/Ada, with little accuracy loss |
| FP4 (NVFP4 / MXFP4) | FP4 | FP4 | FP4 Tensor Cores (Blackwell) | maximum throughput on the newest hardware |

AWQ, GPTQ, SmoothQuant and the rest are algorithms that **decide what the quantized weights should be** (offline); at the kernel level they all end up as "low-bit integers plus scales (and zero points)".

### The quantization formulas {#量化公式}

**Asymmetric (with a zero point)** quantization maps a group of floats into $[0, 2^b - 1]$:

$$
s = \frac{\max - \min}{2^b - 1},\qquad z = \text{round}\left(-\frac{\min}{s}\right),\qquad q = \text{clamp}\left(\text{round}\left(\frac{w}{s}\right) + z,\ 0,\ 2^b - 1\right)
$$

Dequantization: $\hat{w} = (q - z) \cdot s$. **Symmetric** quantization has no zero point and maps into $[-2^{b-1}, 2^{b-1} - 1]$ (or $\pm (2^{b-1} - 1)$).

**The granularity of the scale** balances accuracy against overhead:

- **per tensor**: one scale for the whole matrix, the simplest and the least accurate;
- **per channel**: one per output channel (one per row of W);
- **per group**: one per G consecutive elements within a row, with G commonly 128. INT4 weights are nearly always quantized per group;
- **block-wise**: DeepSeek-V3's FP8 weights are scaled in 128×128 blocks with activations scaled in 1×128 groups.

Where the quantization error comes from, and why per-group quantization isolates outliers, shown with the tool from the LLM handbook:

<div class="aig-widget" data-widget="quant"></div>

## Writing a GEMV kernel {#gemv-kernel-的写法}

For GEMV $y = Wx$ with W of N×K row-major, each output element is the dot product of a row of W with x, and the commonest parallelization is **one warp per row**: 32 threads split the row along K (coalesced, ideally with 128-bit vectorized reads), each accumulating before a warp reduction. x is shared by every row and can go into shared memory first (or be left to L1/L2).

This structure is simple and effective: reading W is fully contiguous, each warp moves plenty of data, and warps never synchronize. The optimizations that matter:

- **vectorization**: 16 bytes per instruction;
- **enough requests in flight per thread**: when K is small, have one warp handle 2-4 rows at once;
- **filling the GPU**: with a smaller N (4096, say), one warp per row is only 4096 warps, so mind the block size and the warps per SM.

## W4A16: dequantizing in registers {#w4a16在寄存器里反量化}

The usual storage for INT4 weights: **8 consecutive 4-bit weights per 32-bit word**, with the first element in the low 4 bits. Each group of G elements has one scale and one zero point. The kernel reads a word, extracts 8 integers with shifts and masks, subtracts the zero point, multiplies by the scale, and multiply-accumulates against the matching 8 elements of x:

```cuda
uint32_t word = w_packed[...];
#pragma unroll
for (int j = 0; j < 8; ++j) {
  int q = (word >> (4 * j)) & 0xF;
  acc += (q - zero) * scale * x[k + j];
}
```

One 32-bit word holds 8 weights, so the reads are an eighth of FP32 weights and a quarter of BF16 ones (plus one scale and zero point per group, which at G = 128 costs about 3%).

```cuda title="gemv_w4.cu"
// gemv_w4.cu - an FP32 GEMV and a W4A16-style per-group INT4 GEMV (one warp per row)
// build: nvcc -O3 -arch=sm_75 gemv_w4.cu -o gemv_w4
// to keep the example about unpacking and dequantizing, the activations and scales are FP32; a real kernel uses FP16/BF16
#include "common.cuh"
#include <cstdint>

constexpr int GROUP = 128;       // every 128 elements along K share one scale and zero point
constexpr int kWarps = 4;        // 4 warps per block, one row each

__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}

// the baseline: FP32 weights, one float4 per lane per step
__global__ void gemv_fp32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K) {
  const int row = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= N) return;
  const float4* w4 = reinterpret_cast<const float4*>(W + static_cast<size_t>(row) * K);
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc = 0.f;
  for (int i = lane; i < K / 4; i += 32) {
    const float4 w = w4[i], v = x4[i];
    acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// INT4: W packed as [N, K/8] uint32; scales and zeros are [N, K/GROUP]
__global__ void gemv_w4(const uint32_t* __restrict__ Wq, const float* __restrict__ scales,
                        const float* __restrict__ zeros, const float* __restrict__ x, float* __restrict__ y,
                        int N, int K) {
  extern __shared__ float xs[];          // x is shared by the whole block, so it goes into shared memory first
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();

  const int row = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= N) return;                  // exit after synchronizing, so __syncthreads is unaffected
  const uint32_t* wrow = Wq + static_cast<size_t>(row) * (K / 8);
  const float* srow = scales + static_cast<size_t>(row) * (K / GROUP);
  const float* zrow = zeros + static_cast<size_t>(row) * (K / GROUP);
  float acc = 0.f;
  for (int i = lane; i < K / 8; i += 32) {            // neighbouring lanes read neighbouring 32-bit words: coalesced
    const uint32_t word = wrow[i];
    const int k = i * 8, g = k / GROUP;
    const float s = srow[g], z = zrow[g];
    const float* xv = xs + k;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float q = static_cast<float>((word >> (4 * j)) & 0xFu);
      acc += (q - z) * s * xv[j];
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

int main() {
  const int N = 4096, K = 4096;   // K has to be a multiple of GROUP
  std::vector<float> W(static_cast<size_t>(N) * K), x(K);
  fill_random(W, 1);
  fill_random(x, 2);

  // quantized offline: 128 elements per group, asymmetric 4-bit
  const int groups = K / GROUP;
  std::vector<uint32_t> Wq(static_cast<size_t>(N) * K / 8, 0u);
  std::vector<float> scales(static_cast<size_t>(N) * groups), zeros(scales.size()), W_deq(W.size());
  for (int r = 0; r < N; ++r)
    for (int g = 0; g < groups; ++g) {
      const float* w = &W[static_cast<size_t>(r) * K + g * GROUP];
      float lo = *std::min_element(w, w + GROUP), hi = *std::max_element(w, w + GROUP);
      float s = (hi - lo) / 15.f;
      if (s == 0.f) s = 1.f;
      float z = std::round(-lo / s);
      scales[static_cast<size_t>(r) * groups + g] = s;
      zeros[static_cast<size_t>(r) * groups + g] = z;
      for (int j = 0; j < GROUP; ++j) {
        const int k = g * GROUP + j;
        int q = static_cast<int>(std::round(w[j] / s + z));
        q = std::min(15, std::max(0, q));
        Wq[(static_cast<size_t>(r) * K + k) / 8] |= static_cast<uint32_t>(q) << (4 * (k % 8));
        W_deq[static_cast<size_t>(r) * K + k] = (q - z) * s;
      }
    }

  // the references: an FP32 GEMV, and a GEMV over the dequantized weights (which the INT4 kernel should match)
  std::vector<float> ref_fp32(N), ref_q(N), got(N);
  double qerr = 0;
  for (int r = 0; r < N; ++r) {
    double a = 0, b = 0;
    for (int k = 0; k < K; ++k) {
      a += double(W[static_cast<size_t>(r) * K + k]) * x[k];
      b += double(W_deq[static_cast<size_t>(r) * K + k]) * x[k];
    }
    ref_fp32[r] = static_cast<float>(a);
    ref_q[r] = static_cast<float>(b);
    qerr = std::max(qerr, std::fabs(a - b));
  }
  std::printf("max |y_fp32 - y_int4| on CPU (quantization error): %.4f\n", qerr);

  float *dW, *dx, *dy, *ds, *dz;
  uint32_t* dWq;
  CUDA_CHECK(cudaMalloc(&dW, W.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWq, Wq.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&ds, scales.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dz, zeros.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx, K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWq, Wq.data(), Wq.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dz, zeros.data(), zeros.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));

  const int blocks = (N + kWarps - 1) / kWarps;
  bool ok = true;
  gemv_fp32<<<blocks, kWarps * 32>>>(dW, dx, dy, N, K);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("gemv_fp32 ");
  ok &= check_close(got.data(), ref_fp32.data(), N, 1e-3f, 1e-3f);
  gemv_w4<<<blocks, kWarps * 32, K * sizeof(float)>>>(dWq, ds, dz, dx, dy, N, K);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("gemv_w4   ");
  ok &= check_close(got.data(), ref_q.data(), N, 1e-3f, 1e-3f);

  const double fp32_bytes = double(W.size()) * 4;
  const double w4_bytes = double(Wq.size()) * 4 + double(scales.size()) * 8;
  float t1 = time_ms([&] { gemv_fp32<<<blocks, kWarps * 32>>>(dW, dx, dy, N, K); });
  float t2 = time_ms([&] { gemv_w4<<<blocks, kWarps * 32, K * sizeof(float)>>>(dWq, ds, dz, dx, dy, N, K); });
  std::printf("fp32 weights: %.3f ms (%.1f GB/s)\nint4 weights: %.3f ms (%.1f GB/s), speedup %.2fx\n",
              t1, gbps(fp32_bytes, t1), t2, gbps(w4_bytes, t2), t1 / t2);
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(dWq));
  CUDA_CHECK(cudaFree(ds));
  CUDA_CHECK(cudaFree(dz));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return ok ? 0 : 1;
}
```

The program reports both the **quantization error** (between the FP32 weights' result and the dequantized weights' result) and the kernel's **implementation error** (between the INT4 kernel and the CPU's result from the dequantized weights). Keep them apart: the first comes from the quantization scheme itself, while the second must be near 0 or the kernel has a bug.

Two things to watch on a GPU: the INT4 version's **effective bandwidth** (by the bytes actually read) is lower than the FP32 version's because unpacking and dequantizing add instructions; but since it reads about 7 times fewer bytes, the total time is still clearly shorter.

## What production kernels do {#生产级-kernel-用了哪些技巧}

- **Fast INT4 to FP16 conversion**: rather than shifting each value and converting, use FP16's bit pattern. Put the 4-bit integer in the low bits of an FP16 mantissa with a fixed exponent (`0x6400` is 1024) and you get 1024 + q, from which subtracting 1024 gives q. Combined with `lop3` this handles several elements at once. FasterTransformer and Marlin both use tricks like this;
- **Reordering the weights offline**: arrange the weights at quantization time in the order the kernel reads them, so one 128-bit read lands exactly the data a Tensor Core fragment needs and no shuffling happens in the kernel;
- **Marlin** (W4A16 GEMM): close to the theoretical best speedup from batch 1 to a few dozen, and vLLM's default; on Hopper there are successors such as Machete;
- **FP8 GEMM**: the Tensor Cores compute FP8 directly with the scales applied after accumulation. With block scaling, the Tensor Core's partial result has to be multiplied by the matching scale at the end of each K block before accumulating into the FP32 accumulator, which DeepGEMM optimizes specifically;
- **MoE**: several experts' GEMMs are done at once as a grouped GEMM, and vLLM's fused_moe fuses token dispatch, the grouped GEMM and the activation together.

!!! interview "How to explain it"
    On quantized GEMV: a linear layer during batch-1 decode is memory-bound, with the time about the weight bytes divided by the bandwidth, so weight-only quantization (W4A16) cuts the reading directly; as the batch grows it becomes compute-bound again and W4A16's speedup falls, so W8A8, FP8 or FP4 are needed to make the arithmetic faster too. The implementation: INT4 quantized per group (128 elements with one scale and zero point each), 8 weights packed into a 32-bit word, and the kernel unpacking and dequantizing in registers; GEMV usually gives one warp a row with vectorized reads and a warp reduction. When evaluating, keep the quantization error and the kernel's implementation error apart.

!!! info "Related chapters"
    - [Quantization](llm://inference/quantization/) (LLM Internals: why it works and where the error comes from)
    - [Deploying quantized models](serving://perf/quantization-deploy/) and [fine-grained FP8 and grouped GEMM](serving://moe/fp8-gemm/) (Inference Systems: deployment and DeepGEMM)

## Exercises {#练习}

**1. A W8A16 GEMV.** Implement symmetric per-channel INT8 quantization (one scale per row, $q = \text{round}(w / s)$ with $s = \max|w| / 127$), storing the weights as `int8_t` and having each lane read one `int4` (16 INT8s) at a time.

??? success "Approach"
    Quantize offline: per row, $s = \max_k |w_k| / 127$ and $q_k = \text{clamp}(\text{round}(w_k / s), -127, 127)$. In the kernel each lane reads 16 bytes as an `int4`, takes the 16 values out through `reinterpret_cast<const int8_t*>`, multiply-accumulates them against 16 elements of x, and multiplies by the row's $s$ at the end. Since the scale is shared by the row, multiplying once after the accumulation beats multiplying every element, the same thinking as W4's per-group scale applied after accumulating within the group.

**2. An analysis question.** How does W4A16's speedup over BF16 change at batch 1, 4, 32 and 128? Why?

??? success "Answer"
    - **batch 1 and 4**: heavily memory-bound, reading about a quarter of BF16's bytes (plus a few scales), so the speedup is close to 3-4×;
    - **batch 32**: the weights serve 32 requests, the arithmetic intensity rises, and the BF16 GEMM approaches compute-bound; W4A16 still computes on FP16 Tensor Cores with dequantization on top, so the speedup falls clearly;
    - **batch 128**: both are compute-bound and W4A16 may even be slower for its dequantization. This is where W8A8 (INT8 or FP8) is needed to make the arithmetic itself faster.

    That is why inference engines support several quantized kernels and choose by batch size.

## Summary {#小结}

- [x] Small-batch decode is memory-bound with a time of about the weight bytes divided by the bandwidth, so weight-only quantization cuts the reading directly.
- [x] Large batches are compute-bound again and need W8A8 (INT8/FP8) or FP4 to make the arithmetic faster too.
- [x] INT4 is usually quantized per group (G = 128) with 8 weights packed into a 32-bit word, and the kernel unpacks and dequantizes in registers.
- [x] GEMV usually gives one warp a row, with vectorized reads and a warp reduction.
- [x] Keep the quantization error and the kernel's implementation error apart; production kernels also use bit tricks for the conversion and reorder the weights offline.
