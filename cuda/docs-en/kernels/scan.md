# Prefix sum and other parallel patterns

<p class="lead">A prefix sum (scan) looks inherently serial: result i depends on every element before it. Solving it in parallel is the classic case study in GPU algorithm design, and the building block of stream compaction, radix sort, MoE token dispatch, top-p sampling and a great deal besides.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How do an inclusive and an exclusive prefix sum differ?
    2. How much work do the Hillis-Steele and Blelloch scans each do?
    3. How do you scan 32 values with warp shuffles? And a whole block?
    4. When the array needs several blocks, how are their results stitched together?
    5. How does stream compaction (selecting the elements that satisfy a condition) relate to a prefix sum?

??? success "Answers (try it yourself first, then expand)"
    1. Inclusive: output i includes input i ($y_i = \sum_{j \le i} x_j$); exclusive does not ($y_i = \sum_{j < i} x_j$), which is "how many are before me" and exactly where to write.
    2. Hillis-Steele: $\log n$ steps with every element adding each step, so $O(n \log n)$ work, few steps and well suited within a warp; Blelloch: an up-sweep and a down-sweep for $O(n)$ work in about $2\log n$ steps.
    3. Within a warp: 5 `__shfl_up_sync` steps at offsets 1, 2, 4, 8 and 16, with each lane at or above the offset adding what it receives. Within a block: each warp scans, each warp's total goes into shared memory, one warp scans those totals, and each warp adds the sum of every warp before it.
    4. Three passes: each block scans its own stretch and emits its total, then the block totals are scanned, then each block adds the total of everything before it. Faster is decoupled look-back: a single pass in which each block looks back at what earlier blocks have published.
    5. Compaction needs to know where each kept element goes: an exclusive prefix sum over the "keep" flags (0 / 1) gives its position in the output; within a warp a ballot plus popc computes it directly.

## Definitions {#定义}

For the input `[3, 1, 7, 0, 4, 1, 6, 3]`:

- **inclusive prefix sum**: `[3, 4, 11, 11, 15, 16, 22, 25]`, where item i includes `x[i]` itself;
- **exclusive prefix sum**: `[0, 3, 4, 11, 11, 15, 16, 22]`, where item i is the sum of `x[0..i-1]`.

The most useful reading of the exclusive prefix sum is **"how many are before me"**, that is, "where should my data go". That is why it turns up everywhere in parallel algorithms.

## Two classic parallel algorithms {#两种经典的并行算法}

**Hillis-Steele**: log n rounds, with each element in round d adding the element 2^d before it. Few steps (log n) but $O(n \log n)$ total work, more than the serial $O(n)$.

**Blelloch**: two phases, building a tree of partial sums bottom up like a reduction (the up-sweep) and distributing them top down (the down-sweep). $O(n)$ total work in 2 log n steps.

Drag the step slider or press play to watch each algorithm scan 16 values:

<div class="aig-widget" data-widget="scanviz"></div>

In practice a GPU layers them: **Hillis-Steele with shuffles within a warp** (32 elements in 5 steps with no synchronization, where the extra work does not matter), and **another layer across warps and blocks**.

## Warp scan and block scan {#warp-扫描与-block-扫描}

The warp scan already appeared in [the previous chapter's exercises](../basics/sync-warp.md#练习):

```cuda
__device__ int warp_inclusive_scan(int v) {
  const int lane = threadIdx.x % 32;
  for (int offset = 1; offset < 32; offset *= 2) {
    int n = __shfl_up_sync(0xffffffff, v, offset);
    if (lane >= offset) v += n;
  }
  return v;
}
```

A block scan wraps a layer around it:

1. each warp scans itself;
2. each warp's last lane writes the warp's total into shared memory;
3. warp 0 scans those totals (at most 32 of them);
4. each thread adds "the total of every warp before mine".

## A device-level scan: scan then propagate {#设备级扫描scan-then-propagate}

When the array is larger than one block can handle, the classic approach is three passes:

1. **scan the tiles**: each block scans its own stretch and writes its total to `block_sums[blockIdx.x]`;
2. **scan the block totals**: an exclusive scan over `block_sums` gives each block's starting offset;
3. **propagate**: each block adds its starting offset to every element of its stretch.

That reads the data twice and writes it twice. CUB's `DeviceScan` uses the more advanced single-pass **decoupled look-back** (Merrill & Garland, 2016): once a block has its local sum it looks back at the states earlier blocks have published ("local sum only" or "inclusive prefix available"), and the moment it gets the previous block's inclusive prefix it knows its own offset. That reads once and writes once, with performance close to a copy.

```cuda title="scan.cu"
// scan.cu - a warp scan, then a block scan, then a three-pass device scan (exclusive prefix sum)
// build: nvcc -O3 -arch=sm_75 scan.cu -o scan
#include "common.cuh"

constexpr int kThreads = 256;
constexpr int kItems = 4;                       // 4 elements per thread
constexpr int kTile = kThreads * kItems;        // 1024 elements per block

__device__ __forceinline__ int warp_inclusive_scan(int v) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    int n = __shfl_up_sync(0xffffffff, v, offset);
    if (lane >= offset) v += n;
  }
  return v;
}

// a block exclusive scan: returns the sum of v over every thread before this one (excluding it), with total being the block's sum
__device__ int block_exclusive_scan(int v, int& total) {
  __shared__ int warp_totals[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  int inclusive = warp_inclusive_scan(v);
  if (lane == 31) warp_totals[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int t = lane < nwarps ? warp_totals[lane] : 0;
    int s = warp_inclusive_scan(t);
    if (lane < nwarps) warp_totals[lane] = s - t;   // each warp's starting offset (exclusive)
    if (lane == nwarps - 1) warp_totals[31] = s;     // holds the block's total
  }
  __syncthreads();
  total = warp_totals[31];
  int result = warp_totals[warp] + inclusive - v;
  __syncthreads();   // so it can be called again in the same kernel
  return result;
}

// step 1: each block scans its own kTile elements and writes the within-tile exclusive prefix sum and the tile total
__global__ void scan_tiles(const int* __restrict__ in, int* __restrict__ out, int* __restrict__ block_sums, int n) {
  const int base = blockIdx.x * kTile + threadIdx.x * kItems;
  int items[kItems];
  int thread_sum = 0;
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    items[k] = base + k < n ? in[base + k] : 0;
    thread_sum += items[k];
  }
  int total;
  int running = block_exclusive_scan(thread_sum, total);   // the sum of everything before this thread
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    if (base + k < n) out[base + k] = running;
    running += items[k];
  }
  if (threadIdx.x == 0) block_sums[blockIdx.x] = total;
}

// step 2: one block scans block_sums exclusively (handling any number of tiles kThreads at a time)
__global__ void scan_block_sums(int* block_sums, int num_blocks) {
  __shared__ int carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int start = 0; start < num_blocks; start += blockDim.x) {
    int i = start + threadIdx.x;
    int v = i < num_blocks ? block_sums[i] : 0;
    int total;
    int ex = block_exclusive_scan(v, total);
    if (i < num_blocks) block_sums[i] = ex + carry;
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
}

// step 3: each element adds its tile's starting offset
__global__ void add_offsets(int* __restrict__ out, const int* __restrict__ block_sums, int n) {
  const int base = blockIdx.x * kTile + threadIdx.x * kItems;
  const int offset = block_sums[blockIdx.x];
#pragma unroll
  for (int k = 0; k < kItems; ++k)
    if (base + k < n) out[base + k] += offset;
}

int main() {
  const int n = 10'000'000 + 7;
  std::vector<int> h(n), ref(n), got(n);
  std::mt19937 gen(1);
  std::uniform_int_distribution<int> dist(0, 9);
  for (auto& v : h) v = dist(gen);
  int acc = 0;
  for (int i = 0; i < n; ++i) { ref[i] = acc; acc += h[i]; }

  const int num_blocks = (n + kTile - 1) / kTile;
  int *d_in, *d_out, *d_sums;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sums, num_blocks * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  auto run = [&] {
    scan_tiles<<<num_blocks, kThreads>>>(d_in, d_out, d_sums, n);
    scan_block_sums<<<1, kThreads>>>(d_sums, num_blocks);
    add_offsets<<<num_blocks, kThreads>>>(d_out, d_sums, n);
  };
  run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (int i = 0; i < n; ++i) bad += got[i] != ref[i];
  std::printf("exclusive scan of %d ints: %s (%zu mismatches)\n", n, bad ? "FAIL" : "PASS", bad);

  float ms = time_ms(run);
  // read in, write out, then read and write out again (step 3)
  std::printf("time %.3f ms, %.1f GB/s effective (4 passes over the data)\n", ms, gbps(4.0 * n * sizeof(int), ms));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_sums));
  return bad ? 1 : 0;
}
```

In production simply use `cub::DeviceScan::ExclusiveSum`, whose interface is the two-call shape of `DeviceReduce` (query the temporary space, then run).

## Stream compaction {#流压缩}

**Stream compaction** picks out the elements satisfying a condition and writes them compactly: dropping zeros, selecting the chosen tokens. The key question is where each selected element goes, and the answer is "how many selected elements are before me", that is, an exclusive prefix sum over the 0/1 flags.

Within a warp, that prefix sum takes one vote instruction:

```cuda
unsigned mask = __ballot_sync(0xffffffff, keep);            // which lanes are kept
int rank = __popc(mask & ((1u << lane) - 1));               // how many kept lanes are before me
int count = __popc(mask);                                   // how many this warp keeps in total
```

When **the original order does not matter** it gets simpler still: lane 0 of each warp does one `atomicAdd` to claim `count` slots and shuffles the starting position to the other lanes. This is a **warp-aggregated atomic**, cutting the global atomics to 1/32:

```cuda title="compact.cu"
// compact.cu - stream compaction with ballot, popc and a warp-aggregated atomic (order not preserved)
// build: nvcc -O3 -arch=sm_75 compact.cu -o compact
#include "common.cuh"

__global__ void compact_positive(const float* __restrict__ in, float* __restrict__ out,
                                 int* __restrict__ count, int n) {
  const int lane = threadIdx.x % 32;
  // the whole warp loops together so every thread is present at the __ballot_sync
  for (int base = blockIdx.x * blockDim.x; base < n; base += blockDim.x * gridDim.x) {
    const int i = base + threadIdx.x;
    const float v = i < n ? in[i] : 0.f;
    const bool keep = i < n && v > 0.f;
    const unsigned mask = __ballot_sync(0xffffffff, keep);
    const int rank = __popc(mask & ((1u << lane) - 1));
    int start = 0;
    if (lane == 0 && mask) start = atomicAdd(count, __popc(mask));
    start = __shfl_sync(0xffffffff, start, 0);
    if (keep) out[start + rank] = v;
  }
}

int main() {
  const int n = 1 << 22;
  std::vector<float> h(n);
  fill_random(h, 3);
  std::vector<float> ref;
  for (float v : h) if (v > 0.f) ref.push_back(v);

  float *d_in, *d_out;
  int* d_count;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  compact_positive<<<sm_count() * 8, 256>>>(d_in, d_out, d_count, n);
  CUDA_CHECK_LAST();
  int count = 0;
  CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> got(count);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
  // the order is not guaranteed, so compare after sorting
  std::sort(got.begin(), got.end());
  std::sort(ref.begin(), ref.end());
  bool ok = got == ref;
  std::printf("kept %d of %d: %s\n", count, n, ok ? "PASS" : "FAIL");
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_count));
  return ok ? 0 : 1;
}
```

Note how the loop is written: rather than each thread testing `i < n` and breaking out, the whole warp loops over `base` together and the out-of-range threads still take part in `__ballot_sync`, merely with `keep` false. That guarantees every thread in the mask is present when the warp-level primitive executes.

## In LLMs: MoE token dispatch {#在大模型里的应用moe-的-token-分发}

In every layer of a mixture-of-experts model, the router assigns each token to its top-k experts. For each expert to handle all its tokens in one GEMM, the tokens have to be reordered by expert:

1. **count**: how many tokens each expert got (a histogram);
2. **exclusive prefix sum**: each expert's starting position in the reordered array;
3. **scatter**: each token is written to "the expert's start plus its index within that expert" (the index coming from an atomic);
4. after the grouped GEMM, gather the results back in the original order and sum them weighted by the routing weights.

This "histogram, prefix sum, scatter" structure appears in vLLM's `moe_align_block_size`, in SGLang, and in DeepSeek's open-source DeepEP. The cumulative sum over the sorted probabilities in top-p sampling is a prefix sum too.

!!! interview "Answering in an interview"
    The one line that matters: an exclusive prefix sum is "how many are before me" is "where I should write". Within a warp use shuffles for Hillis-Steele (5 steps), within a block scan the warps' results again, and at device level use three passes or decoupled look-back; stream compaction computes the write offset within a warp with ballot plus popc, and a warp-aggregated atomic cuts the contention. The typical application in inference is MoE token dispatch: a histogram by expert, a prefix sum for each expert's start, and a scatter of the tokens into place.

## Exercises {#练习}

**1. Order-preserving stream compaction.** Use the three-pass method in `scan.cu` for an order-preserving compaction: turn "keep" into 0/1 flags, scan the flags exclusively for each element's output position, and write.

??? success "Approach"
    The most direct implementation is three kernels: `flags[i] = in[i] > 0`; an exclusive scan of `flags` giving `pos`; and `if (flags[i]) out[pos[i]] = in[i]`, with the total kept being `pos[n-1] + flags[n-1]`. A more efficient version fuses the flag computation into the scan's first pass and the write into its third, so `flags` and `pos` never reach device memory. CUB's `DeviceSelect::If` does it in a single pass.

**2. MoE counts and offsets.** Given a `num_tokens × top_k` array `topk_ids` of expert indices with E experts, produce each expert's token count `counts[E]` and starting offset `offsets[E]`. Sketch the kernel's structure.

??? success "Approach"
    - count with a shared-memory histogram (E is usually tens to hundreds, which fits), with each block handling a stretch and atomically adding into the global `counts` at the end; see [histogram](../basics/sync-warp.md#直方图共享内存私有化);
    - when E is small, one block can compute the offsets directly with `block_exclusive_scan`;
    - real implementations usually round the offsets up to a multiple of the GEMM tile size (64, say) so the grouped GEMM can work tile by tile, which is where the name `moe_align_block_size` in vLLM comes from.

## Summary {#小结}

- [x] An exclusive prefix sum is "how many are before me" is "where I should write".
- [x] Use shuffles for Hillis-Steele within a warp, another layer over the warp scans within a block, and three passes or decoupled look-back at device level.
- [x] Stream compaction computes the within-warp offset with ballot plus popc, and a warp-aggregated atomic cuts the contention.
- [x] MoE token dispatch is the classic application of "histogram, prefix sum, scatter".
