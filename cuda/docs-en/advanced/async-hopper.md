# Asynchronous copies and the Hopper/Blackwell features

<p class="lead">From Ampere on, every NVIDIA generation has pursued the same thing: making data movement and computation fully parallel. Ampere brought cp.async, Hopper brought TMA, thread-block clusters and wgmma, and Blackwell added tensor memory. This chapter explains what each feature solves and how to use it, with examples that compile and run.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does `cp.async` do better than an ordinary "global memory to registers to shared memory" copy?
    2. What does `wait_prior(STAGES - 2)` mean in a multi-stage pipeline? Why commit every round even when nothing was issued?
    3. What does TMA need prepared on the host? How many threads issue the copy inside the kernel?
    4. What new ability does a thread-block cluster give blocks?
    5. What is warp specialization? Why do Hopper's GEMM and FlashAttention-3 both use it?

??? success "Answers (try it yourself first, then expand)"
    1. It copies from global memory straight into shared memory without registers, saving both the register pressure and the staging instructions, and it is asynchronous, so a thread can carry on after issuing it and the copy overlaps the compute. It is the basis of a multi-stage pipeline.
    2. `wait_prior(STAGES - 2)` waits until at most STAGES − 2 groups of copies are still outstanding, that is, until the group about to be used has arrived. Committing every round even when nothing was issued keeps the group count aligned with the iteration, or the wait would be for the wrong group.
    3. The host builds a tensor descriptor with something like `cuTensorMapEncodeTiled` (the global tensor's shape, strides, the tile to move, the swizzle) and passes it to the kernel; inside, one thread issues the whole-tile copy and an mbarrier counting bytes waits for it.
    4. Blocks of one cluster can synchronize with each other and read and write each other's shared memory directly (distributed shared memory, DSMEM), and TMA can multicast a tile to several blocks of the cluster.
    5. Splitting a block's warps into "producers" (moving data with TMA) and "consumers" (computing with wgmma), handing off through an mbarrier ring buffer. Movement and compute then proceed in parallel in different warps and the Tensor Cores never wait for data, which is why Hopper's GEMM and FlashAttention-3 both adopt it.

## Why asynchronous copies {#为什么需要异步拷贝}

Recall [GEMM](../kernels/gemm.md)'s main loop: each round reads global memory into registers, writes shared memory, synchronizes, computes, and synchronizes again. There are two problems:

1. **movement and compute are serial**: the math units idle during the load and the memory units during the compute;
2. **the data is staged through registers**: occupying registers and needing extra instructions.

Double buffering eases the first, but the data still goes through registers. Ampere's `cp.async` solves both: it **copies from global memory straight into shared memory**, and it is **asynchronous**, so a thread issues it and immediately does something else, waiting later.

## cp.async and multi-stage pipelines <span class="arch">sm_80+</span> {#cpasync-与多级流水-sm_80}

The PTX instructions are `cp.async.ca.shared.global` / `cp.async.cg.shared.global` (4, 8 or 16 bytes at a time, with `.cg` bypassing L1 and only for 16 bytes), together with:

- `cp.async.commit_group`: package everything issued so far into one "group";
- `cp.async.wait_group N`: wait until **at most N groups** remain outstanding.

CUDA provides matching C++ functions in `<cuda_pipeline.h>`: `__pipeline_memcpy_async`, `__pipeline_commit` and `__pipeline_wait_prior`. Higher-level wrappers are `cuda::memcpy_async` and `cuda::pipeline`.

**A multi-stage pipeline**: allocate S shared-memory buffers (S is usually 3 or 4) and keep the loads S-1 steps ahead of the compute:

```cuda
// prefetch: issue the loads for the first S-1 tiles, one group each
for (int s = 0; s < S - 1; ++s) { load_tile_async(buf = s, tile = s); __pipeline_commit(); }

for (int kt = 0; kt < num_tiles; ++kt) {
  __pipeline_wait_prior(S - 2);   // at most S-2 groups still in flight, so tile kt has arrived
  __syncthreads();                // every thread's tile kt has arrived, and everyone has finished tile kt-1
  int next = kt + S - 1;
  if (next < num_tiles) load_tile_async(buf = next % S, tile = next);   // this overwrites tile kt-1's buffer
  __pipeline_commit();            // commit an empty group even when nothing was issued, to keep the group count regular
  compute(buf = kt % S);
}
```

Two details are easy to get wrong:

- **what `wait_prior(S-2)` means**: of the groups committed, at most the most recent S-2 may still be outstanding. At round kt, kt + S - 1 groups have been committed and at most S-2 are outstanding, which means the first kt + 1 groups (tiles 0 through kt) have arrived;
- **commit empty groups too**: the last few rounds issue no new loads, and without an empty commit, `wait_prior(S-2)` would not be waiting for the group you think it is.

Below, GEMM's v4 (two-dimensional register tiling) becomes a 3-stage cp.async pipeline:

```cuda title="gemm_cp_async.cu"
// gemm_cp_async.cu - a 3-stage cp.async pipeline on top of the two-dimensional register-tiled SGEMM
// build: nvcc -O3 -arch=sm_80 gemm_cp_async.cu -o gemm_cp_async
// requires M and N to be multiples of 128 and K a multiple of 8
#include "common.cuh"
#include <cuda_pipeline.h>

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, STAGES = 3;
constexpr int kThreads = (BM * BN) / (TM * TN);   // 256

__global__ void __launch_bounds__(kThreads)
sgemm_cp_async(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
               float beta, float* __restrict__ C) {
  __shared__ __align__(16) float As[STAGES][BM * BK];
  __shared__ __align__(16) float Bs[STAGES][BK * BN];
  const int tid = threadIdx.x;
  const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
  // per tile, each thread moves 16 bytes of A and 16 of B
  const int a_row = tid / (BK / 4), a_col = (tid % (BK / 4)) * 4;   // a 128x8 tile of A: 2 lots of 16 bytes per row
  const int b_row = tid / (BN / 4), b_col = (tid % (BN / 4)) * 4;   // an 8x128 tile of B: 32 lots of 16 bytes per row
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  auto load_tile = [&](int buf, int kt) {
    const int k0 = kt * BK;
    __pipeline_memcpy_async(&As[buf][a_row * BK + a_col], &A[a_row * K + k0 + a_col], 16);
    __pipeline_memcpy_async(&Bs[buf][b_row * BN + b_col], &B[(k0 + b_row) * N + b_col], 16);
  };

  const int num_tiles = K / BK;
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_tiles) load_tile(s, s);
    __pipeline_commit();
  }

  float acc[TM][TN] = {{0.f}};
  float regM[TM], regN[TN];
  for (int kt = 0; kt < num_tiles; ++kt) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < num_tiles) load_tile(next % STAGES, next);
    __pipeline_commit();

    const float* as = As[kt % STAGES];
    const float* bs = Bs[kt % STAGES];
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = as[(threadRow * TM + i) * BK + dot];
#pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = bs[dot * BN + threadCol * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] += regM[m] * regN[n];
    }
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      float& c = C[(threadRow * TM + m) * N + threadCol * TN + n];
      c = alpha * acc[m][n] + beta * c;
    }
}

int main(int argc, char** argv) {
  require_sm(8, 0);
  const int check_n = 256;
  const int n_bench = argc > 1 ? std::atoi(argv[1]) : 4096;
  const float alpha = 1.f, beta = 0.f;
  {
    const int n = check_n;
    std::vector<float> hA(n * n), hB(n * n), ref(n * n, 0.f), got(n * n);
    fill_random(hA, 1);
    fill_random(hB, 2);
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j) ref[i * n + j] += hA[i * n + k] * hB[k * n + j];
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C, 0, n * n * sizeof(float)));
    sgemm_cp_async<<<dim3(n / BN, n / BM), kThreads>>>(n, n, n, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), C, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = check_close(got.data(), ref.data(), n * n, 1e-4f, 1e-4f);
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    if (!ok) return 1;
  }
  const int n = n_bench;
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, bytes));
  CUDA_CHECK(cudaMalloc(&B, bytes));
  CUDA_CHECK(cudaMalloc(&C, bytes));
  CUDA_CHECK(cudaMemset(A, 0, bytes));
  CUDA_CHECK(cudaMemset(B, 0, bytes));
  float ms = time_ms([&] { sgemm_cp_async<<<dim3(n / BN, n / BM), kThreads>>>(n, n, n, alpha, A, B, beta, C); });
  std::printf("cp.async 3-stage SGEMM, n=%d: %.3f ms, %.2f TFLOPS (compare with gemm v4)\n", n, ms,
              tflops(2.0 * n * n * n, ms));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
```

Note the `if (s < num_tiles)` in the prefetch stage: it has to be correct when K is small and there are fewer tiles than pipeline stages. Compare its performance with v4 in [GEMM](../kernels/gemm.md) to see what the pipeline buys. The gain is modest on the FP32 CUDA Core version (the compute is already heavy) but crucial on a Tensor Core version: a Tensor Core is so fast that without hiding the movement it spends most of its time waiting for data.

## TMA: the Tensor Memory Accelerator <span class="arch">sm_90+</span> {#tma张量内存加速器-sm_90}

![Figure: Hopper's asynchronous pipeline, with a producer warp issuing TMA copies, an mbarrier counting, and a consumer warp group running wgmma](../assets/figures/tma-pipeline.svg){.aig-svg}

cp.async still has each thread work out which 16 bytes it moves. Hopper's **TMA (Tensor Memory Accelerator)** is a dedicated movement unit: **one thread** issues one instruction and a whole box of a multidimensional tensor moves from global memory into shared memory, with the address arithmetic, the bounds handling (zero-filling out of range) and the swizzle all done in hardware.

Using TMA takes:

1. **on the host**, a **tensor descriptor** (`CUtensorMap`, 128 bytes) describing the global tensor's dimensions, strides, the tile moved each time and the swizzle, built with the driver API `cuTensorMapEncodeTiled`;
2. passing the descriptor to the kernel as a `const __grid_constant__ CUtensorMap`;
3. inside the kernel, one thread issuing a `cp.async.bulk.tensor`, with completion signalled through an **mbarrier** in shared memory: an mbarrier counts not only arriving threads but also **arriving bytes** (a transaction count), and flips only once all the data is in.

The example below uses TMA to read an integer matrix into shared memory in 16×32 tiles, add 1 to each element, and write it back with TMA:

```cuda title="tma_copy.cu"
// tma_copy.cu - reading and writing two-dimensional tiles with TMA: one thread issues the copy and an mbarrier waits by byte count
// build: nvcc -O3 -arch=sm_90 tma_copy.cu -o tma_copy -lcuda
#include "common.cuh"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int BOX_H = 16, BOX_W = 32;   // a tile of 16 rows by 32 int columns, 128 bytes per row

__global__ void add_one_tma(const __grid_constant__ CUtensorMap tensor_map) {
  __shared__ alignas(128) int tile[BOX_H][BOX_W];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  const int32_t coords[2] = {static_cast<int32_t>(blockIdx.x * BOX_W),    // the innermost dimension (the column) comes first
                             static_cast<int32_t>(blockIdx.y * BOX_H)};

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    ptx::fence_proxy_async(ptx::space_shared);   // make the initialized barrier visible to TMA (the async proxy)
  }
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, &tile, &tensor_map, coords,
                              cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile));   // declare that sizeof(tile) more bytes are expected
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token));

  for (int i = threadIdx.x; i < BOX_H * BOX_W; i += blockDim.x) tile[i / BOX_W][i % BOX_W] += 1;

  ptx::fence_proxy_async(ptx::space_shared);     // make the ordinary threads' shared-memory writes visible to TMA
  __syncthreads();
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, &tensor_map, coords, &tile);
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());   // wait for TMA to finish reading shared memory before exiting
    (&bar)->~barrier();
  }
}

int main() {
  require_sm(9, 0);
  const int rows = 1024, cols = 2048;   // the row stride, cols * 4 bytes, must be a multiple of 16
  std::vector<int> h(static_cast<size_t>(rows) * cols), got(h.size());
  for (size_t i = 0; i < h.size(); ++i) h[i] = static_cast<int>(i % 1000);
  int* d;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));

  CUtensorMap map{};
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};   // the innermost dimension comes first
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * sizeof(int)};                  // dimension 1's stride, in bytes
  cuuint32_t box[2] = {BOX_W, BOX_H};
  cuuint32_t elem_strides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, d, dims, strides, box, elem_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::printf("cuTensorMapEncodeTiled failed: %d\n", static_cast<int>(r));
    return 1;
  }
  add_one_tma<<<dim3(cols / BOX_W, rows / BOX_H), 128>>>(map);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d, h.size() * sizeof(int), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < h.size(); ++i) bad += got[i] != h[i] + 1;
  std::printf("TMA add-one: %s (%zu mismatches)\n", bad ? "FAIL" : "PASS", bad);
  CUDA_CHECK(cudaFree(d));
  return bad ? 1 : 0;
}
```

The two `fence_proxy_async` calls matter: TMA belongs to the "async proxy", where ordinary threads' writes to shared memory are not visible to it by default, nor the reverse, so an explicit proxy fence is needed. Details like this are exactly why using TMA directly is error-prone, and why production code goes through CUTLASS/CuTe's wrappers.

In a real GEMM, the TMA descriptor is configured with a 128-byte swizzle so that what arrives in shared memory is exactly the layout wgmma wants, with several mbarriers forming a multi-stage pipeline.

## Thread-block clusters and distributed shared memory <span class="arch">sm_90+</span> {#线程块集群与分布式共享内存-sm_90}

Hopper added a level between the grid and the block: the **thread-block cluster**. Blocks of one cluster (at most 8, 16 on some GPUs) are guaranteed to be **scheduled together** onto SMs of one GPC (GPU processing cluster), so they can:

- synchronize cluster-wide with `cluster.sync()`;
- **read and write each other's shared memory**, which is **distributed shared memory (DSMEM)**.

That breaks the rule that blocks cannot communicate directly. The typical uses: several blocks cooperating on a tile larger than one SM's shared memory; and TMA's **multicast**, which delivers one tile into the shared memory of several SMs of the cluster at once, cutting repeated L2 reads (neighbouring blocks in a GEMM need the same tile of A or B).

```cuda title="cluster_sum.cu"
// cluster_sum.cu - two blocks form a cluster, and block 0 reads block 1's shared memory through DSMEM
// build: nvcc -O3 -arch=sm_90 cluster_sum.cu -o cluster_sum
#include "common.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

constexpr int kThreads = 256;

__global__ void __cluster_dims__(2, 1, 1) pair_sum(const float* __restrict__ in, float* __restrict__ out, int per_block) {
  __shared__ float partial[kThreads];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();   // this block's index within the cluster: 0 or 1

  // each block sums its own stretch first
  const float* src = in + static_cast<size_t>(blockIdx.x) * per_block;
  float v = 0.f;
  for (int i = threadIdx.x; i < per_block; i += blockDim.x) v += src[i];
  partial[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }

  cluster.sync();   // both blocks' partial[0] are ready and visible within the cluster
  if (rank == 0 && threadIdx.x == 0) {
    float* remote = cluster.map_shared_rank(&partial[0], 1);   // mapped onto block 1's shared memory
    out[blockIdx.x / 2] = partial[0] + *remote;
  }
  cluster.sync();   // block 1 must wait for block 0's read before exiting, or its shared memory is released
}

int main() {
  require_sm(9, 0);
  const int clusters = 64, per_block = 10000;
  const int blocks = clusters * 2;
  std::vector<float> h(static_cast<size_t>(blocks) * per_block), ref(clusters), got(clusters);
  fill_random(h, 5);
  for (int c = 0; c < clusters; ++c) {
    double s = 0;
    for (int i = 0; i < 2 * per_block; ++i) s += h[static_cast<size_t>(c) * 2 * per_block + i];
    ref[c] = static_cast<float>(s);
  }
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, clusters * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  pair_sum<<<blocks, kThreads>>>(d_in, d_out, per_block);   // the grid size must be a multiple of the cluster size
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, clusters * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), clusters, 1e-4f, 1e-3f);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

The cluster size can also be set at launch through `cudaLaunchKernelEx`'s `cudaLaunchAttributeClusterDimension`.

## wgmma and warp specialization <span class="arch">sm_90a</span> {#wgmma-与-warp-专门化-sm_90a}

Hopper's Tensor Core instruction `wgmma.mma_async` is issued by a **warpgroup** (4 warps) computing an m64nNk16 matrix multiply, with the B operand (and optionally A) read straight from shared memory, asynchronously. Combined with TMA it forms the standard shape of a high-performance Hopper kernel: **warp specialization**.

<!-- i18n:diagram cc19d2e6a6 -->
```text
one block (3 warpgroups, say)
├── the producer warpgroup (using very few registers)
│     loop: wait on the "buffer i is free" mbarrier, then issue a TMA load into buffer i
└── 2 consumer warpgroups (holding most of the registers)
      loop: wait on the "buffer i is full" mbarrier, issue wgmma, and signal "buffer i is free" when done
```

Producers and consumers form a **ring buffer** through a set of mbarriers in shared memory, keeping TMA and the Tensor Cores working in parallel. Hopper also offers the `setmaxnreg` instruction so producers can hand registers over to consumers. On top of this, FlashAttention-3 has two consumer warpgroups alternate between GEMM and softmax (ping-pong scheduling), hiding the softmax behind the GEMM too, and reaches about 740 TFLOPS of FP16 forward on an H100 (about 75% utilization, from the FlashAttention-3 paper).

Writing wgmma by hand means building shared-memory matrix descriptors and handling the swizzle and the register layouts yourself, which is a lot of code. A suggested path:

1. read CUTLASS's Hopper GEMM examples (`examples/48_hopper_warp_specialized_gemm` and others) and the CuTe tutorials;
2. read DeepSeek's open-source **DeepGEMM**, which implements FP8 GEMM on Hopper in relatively compact code;
3. read teaching-oriented libraries such as ThunderKittens;
4. Colfax Research has published a very readable series of tutorials on wgmma, TMA and warp specialization.

## Blackwell at a glance <span class="arch">sm_100</span> {#blackwell-概览-sm_100}

Blackwell (B200, GB200, B300) changes the Tensor Core programming model considerably again:

- **`tcgen05.mma`**: issued by **a single thread**, with no whole warp or warpgroup needed;
- **Tensor Memory (TMEM)**: 256 KB of dedicated storage per SM holding the matrix multiply's accumulators instead of registers, which relieves the register pressure greatly; it has to be allocated, freed and moved to and from registers explicitly;
- **2-SM MMA**: a pair of SMs (a CTA pair) can cooperate on a larger matrix multiply;
- **lower precision and block scaling**: native FP6 and FP4 plus block-scaled formats such as MXFP8/MXFP4 (an 8-bit exponent scale per 32 elements) and NVFP4 (an FP8 scale per 16 elements plus a tensor-level FP32 scale).

The RTX 50 series (sm_120) is also called Blackwell but has a different Tensor Core programming model from the data-centre parts and does not support tcgen05. The best material for learning Blackwell is CUTLASS 4.x's examples and the CuTe DSL (its Python interface).

!!! interview "How to explain it"
    To explain how a high-performance Hopper kernel is written: `cp.async` copies from global memory into shared memory asynchronously, and a multi-stage pipeline keeps the loads several steps ahead of the compute; TMA has one thread issue a whole-tile move, needing a host-side tensor descriptor and an mbarrier counting bytes; thread-block clusters let blocks synchronize and reach each other's shared memory (DSMEM) and support TMA multicast; warp specialization has some warps move data and others run wgmma, handing off through an mbarrier ring buffer, which is the structure of both FlashAttention-3 and Hopper's GEMM. Then mention Blackwell's tcgen05, tensor memory and block-scaled low-precision formats.

## Exercises {#练习}

**1. Pipeline depth.** In `gemm_cp_async.cu`, how much shared memory does `STAGES` of 2, 3 and 4 use? Are more stages always better?

??? success "Answer"
    Each stage needs `(128 × 8 + 8 × 128) × 4 B = 8 KB`, so 2, 3 and 4 stages are 16, 24 and 32 KB. More stages hide more memory latency but use more shared memory, which may cut how many blocks are resident per SM; and once the depth covers the latency, more buys nothing. The usual choice is 3-4 stages on Ampere and 4-8 on Hopper (whose shared memory is larger and whose TMA latency suits a deeper pipeline), decided in the end by measurement and autotuning.

**2. Why does TMA need only one thread to issue it?** What does that mean for the kernel's design?

??? success "Answer"
    TMA is a separate hardware unit that does both the address arithmetic and the movement, so a thread only issues one instruction saying where the result goes and which mbarrier signals it. Two consequences: (1) the threads doing the computing take no part in the movement and spend no registers on it, so the warps can be split into producers that only move and consumers that only compute; (2) synchronization moves from `__syncthreads()` to an mbarrier-based producer-consumer protocol, which means carefully managing each buffer's "full" and "empty" states and the mbarriers' phases.

## Summary {#小结}

- [x] cp.async copies from global memory into shared memory asynchronously; a multi-stage pipeline keeps the loads S-1 steps ahead, with `wait_prior(S-2)` and a commit every round.
- [x] TMA has one thread issue a whole-tile move, needing a host-side tensor descriptor and a byte-counting mbarrier, and minding the proxy fences.
- [x] Thread-block clusters let blocks synchronize and reach each other's shared memory (DSMEM), and support TMA multicast.
- [x] The standard shape of a high-performance Hopper kernel: TMA plus wgmma plus warp specialization plus an mbarrier ring buffer.
- [x] Blackwell introduces single-thread-issued tcgen05, tensor memory and block-scaled low-precision formats.
