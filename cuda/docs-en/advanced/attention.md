# FlashAttention and inference kernels

<p class="lead">Attention is the most important kernel in an LLM and the most frequently asked about. This chapter starts from the memory problem in standard attention, derives FlashAttention's tiling plus online softmax, and writes a working forward kernel; then it turns to inference: the KV cache, PagedAttention, split-K during decode (Flash-Decoding), and a paged decode attention kernel supporting GQA.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Where is the bottleneck in a standard attention implementation? Why is it memory-bound?
    2. How does FlashAttention avoid reading and writing the N×N matrix without changing the result?
    3. What did FlashAttention-2 improve over the first version? And FlashAttention-3?
    4. During decode there is only one query per step. How should the attention kernel parallelize?
    5. What does PagedAttention solve? What extra work does the kernel do?

??? success "Answers (try it yourself first, then expand)"
    1. The N×N score matrix and the attention weights go out to memory and come back, while the arithmetic itself is modest, so the time goes into those intermediates and it is memory-bound (and at large N the memory cannot even hold them).
    2. Compute in tiles: take a tile of Q and a tile of K and V on chip, compute that tile's scores, keep each row's maximum and sum of exponentials with online softmax, and merge into the output as you go. The intermediates never reach memory, and mathematically it is the same softmax.
    3. FA2: parallelize along the sequence dimension too, split by Q across warps (which cuts inter-warp communication), and do less non-matmul arithmetic. FA3: Hopper's TMA for asynchronous movement, wgmma, and warp specialization that overlaps softmax with the matmuls, plus FP8 support.
    4. One query cannot fill the GPU: beyond parallelizing over the batch and the heads, split a long KV into segments computed by different blocks, merged at the end with log-sum-exp (Flash-Decoding / split-KV).
    5. KV cache fragmentation and waste: store in blocks, look up through a block table, allocate on demand, and share a prefix between requests. The kernel gains one indirection: look the logical block up in the block table to find the physical block, then read the KV.

A six-panel strip before the text:

<!-- comic ../assets/comics/flash-attention.webp is in Chinese; put it back once the English version exists -->

## What is wrong with standard attention {#标准注意力的问题}

$$
S = \frac{QK^\top}{\sqrt{d}},\qquad P = \text{softmax}(S),\qquad O = PV
$$

where Q, K and V are N×d (N the sequence length, d the head dimension, commonly 64 or 128). The standard implementation is three steps, one kernel each:

1. compute S and write it out: N×N elements;
2. read S, softmax it, write P: another N×N;
3. read P and V and compute O.

At N = 8192, each head's S has 67 million elements, 128 MB in FP16. The whole thing reads and writes an N×N matrix several times over, while the actual arithmetic (two GEMMs, about 4N²d) is modest. **Attention's bottleneck is the memory traffic of the N×N intermediates**, and its memory footprint grows with the square of the sequence length.

## FlashAttention: tiling plus online softmax {#flashattention分块--online-softmax}

The core idea: **never write S and P to memory**. Split Q by row into tiles $Q_i$ ($B_r$ rows each) and K and V by row into $K_j, V_j$ ($B_c$ rows each). For each $Q_i$, work through every $K_j, V_j$:

![Figure: FlashAttention computed in tiles](../assets/figures/flash-attention.svg){.aig-svg}

1. compute $S_{ij} = Q_i K_j^\top / \sqrt{d}$ on chip ($B_r \times B_c$, in registers or shared memory);
2. update each row's maximum $m$ and sum of exponentials $\ell$ the online-softmax way;
3. rescale the output accumulated so far to the new maximum and add this tile's contribution $\tilde{P}_{ij} V_j$.

For row i, after tile j:

$$
\begin{aligned}
m^{(j)} &= \max\left(m^{(j-1)},\ \text{rowmax}(S_{ij})\right) \\
\tilde{P}_{ij} &= \exp\left(S_{ij} - m^{(j)}\right) \\
\ell^{(j)} &= e^{m^{(j-1)} - m^{(j)}}\,\ell^{(j-1)} + \text{rowsum}(\tilde{P}_{ij}) \\
O^{(j)} &= e^{m^{(j-1)} - m^{(j)}}\,O^{(j-1)} + \tilde{P}_{ij} V_j
\end{aligned}
$$

After every tile, $O = O^{(\text{last})} / \ell^{(\text{last})}$. This is [the previous chapter's online softmax](../kernels/softmax-norm.md#online-softmax一次遍历求出最大值和指数和) extended to "and then multiply by V": because the output is linear in $\tilde{P}$, the scaling factor can be applied directly to the accumulator O.

The result is **mathematically identical** to standard attention, not an approximation. The memory traffic drops from $O(N^2)$ to the order of $O(N d)$ (each tile of Q reads all of K and V once) with every intermediate on chip. The memory footprint drops from $O(N^2)$ to $O(N)$ too.

### How FlashAttention evolved {#几代-flashattention-的演进}

| Version | Main improvements |
| --- | --- |
| FlashAttention (2022) | tiling plus online softmax, IO-aware; the backward pass recomputes S rather than storing it |
| FlashAttention-2 (2023) | parallelize along the sequence length too (one block per Q tile), filling the GPU better at long sequences and small batches; less non-matmul arithmetic (dividing by ℓ only at the end); split across warps by Q rather than by K, so warps need not exchange intermediates through shared memory |
| FlashAttention-3 (2024) | built for Hopper: TMA plus wgmma plus warp specialization; two warpgroups ping-pong so softmax overlaps the GEMMs; FP8 support |

**The causal mask** (self-attention in a decoder): query i can only see the first i keys. In the tiled algorithm, tiles entirely above the diagonal are skipped, saving nearly half the compute, and the tiles on the diagonal are masked elementwise.

## A working FlashAttention forward kernel {#一个能跑通的-flashattention-前向-kernel}

Below is a teaching version: FP32, no Tensor Cores, one thread per query row, one block per $B_r = 64$ queries, with $B_c = 32$ rows of K and V brought into shared memory at a time. Its purpose is to put the algorithm into code; it is far slower than a real implementation (which computes $QK^\top$ and $PV$ on Tensor Cores with 16 rows per warp and does the online softmax in register fragments).

```cuda title="flash_attn.cu"
// flash_attn.cu - a teaching FlashAttention forward: tiling plus online softmax, with a causal mask
// build: nvcc -O3 -arch=sm_75 flash_attn.cu -o flash_attn
// layout: Q, K, V and O are all stored contiguously as [batch * heads, seq_len, D]
#include "common.cuh"

constexpr int Br = 64;   // query rows per block (equal to the thread count)
constexpr int Bc = 32;   // key/value rows loaded at a time

template <int D>
__global__ void __launch_bounds__(Br)
flash_attn_fwd(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
               float* __restrict__ O, int N, float scale, bool causal) {
  __shared__ float Ks[Bc][D];
  __shared__ float Vs[Bc][D];
  const size_t head_off = static_cast<size_t>(blockIdx.y) * N * D;
  const int q_row = blockIdx.x * Br + threadIdx.x;
  const bool active = q_row < N;

  float q[D], o[D];
#pragma unroll
  for (int k = 0; k < D; ++k) {
    q[k] = active ? Q[head_off + static_cast<size_t>(q_row) * D + k] * scale : 0.f;   // scaled by 1/sqrt(d) beforehand
    o[k] = 0.f;
  }
  float m = -INFINITY, l = 0.f;

  // the causal mask: keys past this block's last query are never needed
  const int kv_end = causal ? min(N, static_cast<int>(blockIdx.x + 1) * Br) : N;
  for (int j0 = 0; j0 < kv_end; j0 += Bc) {
    // cooperatively load a tile of K and V (zero-filled out of range, masked out later)
    for (int idx = threadIdx.x; idx < Bc * D; idx += Br) {
      const int r = idx / D, c = idx % D, g = j0 + r;
      Ks[r][c] = g < N ? K[head_off + static_cast<size_t>(g) * D + c] : 0.f;
      Vs[r][c] = g < N ? V[head_off + static_cast<size_t>(g) * D + c] : 0.f;
    }
    __syncthreads();

    // 1) this tile's scores and its maximum
    float s[Bc];
    float block_max = -INFINITY;
#pragma unroll
    for (int c = 0; c < Bc; ++c) {
      const int key = j0 + c;
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < D; ++k) acc += q[k] * Ks[c][k];   // every thread reads the same row: a broadcast
      const bool masked = key >= N || (causal && key > q_row);
      s[c] = masked ? -INFINITY : acc;
      block_max = fmaxf(block_max, s[c]);
    }

    // 2) the online softmax update; m stays put when the whole tile is masked
    const float m_new = fmaxf(m, block_max);
    if (m_new != -INFINITY) {
      const float correction = __expf(m - m_new);   // with m at -inf the result is 0
      float row_sum = 0.f;
#pragma unroll
      for (int c = 0; c < Bc; ++c) {
        s[c] = __expf(s[c] - m_new);                // a masked position gives 0
        row_sum += s[c];
      }
      l = l * correction + row_sum;
      // 3) rescale the old accumulators and add P V
#pragma unroll
      for (int k = 0; k < D; ++k) {
        float acc = o[k] * correction;
#pragma unroll
        for (int c = 0; c < Bc; ++c) acc += s[c] * Vs[c][k];
        o[k] = acc;
      }
      m = m_new;
    }
    __syncthreads();   // before the next round overwrites Ks/Vs, make sure everyone is done with them
  }

  if (active) {
    const float inv_l = 1.f / l;
#pragma unroll
    for (int k = 0; k < D; ++k) O[head_off + static_cast<size_t>(q_row) * D + k] = o[k] * inv_l;
  }
}

void attention_cpu(const float* Q, const float* K, const float* V, float* O, int BH, int N, int D, bool causal) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  std::vector<double> s(N);
  for (int h = 0; h < BH; ++h) {
    const size_t off = static_cast<size_t>(h) * N * D;
    for (int i = 0; i < N; ++i) {
      double mx = -1e300;
      const int end = causal ? i + 1 : N;
      for (int j = 0; j < end; ++j) {
        double acc = 0;
        for (int k = 0; k < D; ++k) acc += double(Q[off + i * D + k]) * K[off + j * D + k];
        s[j] = acc * scale;
        mx = std::max(mx, s[j]);
      }
      double sum = 0;
      for (int j = 0; j < end; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
      for (int k = 0; k < D; ++k) {
        double acc = 0;
        for (int j = 0; j < end; ++j) acc += s[j] * V[off + j * D + k];
        O[off + i * D + k] = static_cast<float>(acc / sum);
      }
    }
  }
}

int main() {
  constexpr int D = 64;
  const int batch_heads = 8, N = 1000;   // deliberately not a multiple of the tile size
  const size_t n = static_cast<size_t>(batch_heads) * N * D, bytes = n * sizeof(float);
  std::vector<float> hq(n), hk(n), hv(n), ref(n), got(n);
  fill_random(hq, 1);
  fill_random(hk, 2);
  fill_random(hv, 3);
  float *dq, *dk, *dv, *dout;
  CUDA_CHECK(cudaMalloc(&dq, bytes));
  CUDA_CHECK(cudaMalloc(&dk, bytes));
  CUDA_CHECK(cudaMalloc(&dv, bytes));
  CUDA_CHECK(cudaMalloc(&dout, bytes));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), bytes, cudaMemcpyHostToDevice));

  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  dim3 grid((N + Br - 1) / Br, batch_heads);
  bool ok = true;
  for (bool causal : {false, true}) {
    attention_cpu(hq.data(), hk.data(), hv.data(), ref.data(), batch_heads, N, D, causal);
    flash_attn_fwd<D><<<grid, Br>>>(dq, dk, dv, dout, N, scale, causal);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), dout, bytes, cudaMemcpyDeviceToHost));
    std::printf("causal=%d  ", causal);
    ok &= check_close(got.data(), ref.data(), n, 1e-3f, 1e-4f);
    float ms = time_ms([&] { flash_attn_fwd<D><<<grid, Br>>>(dq, dk, dv, dout, N, scale, causal); });
    const double flops = 4.0 * batch_heads * N * N * D * (causal ? 0.5 : 1.0);
    std::printf("           %.3f ms, %.2f TFLOPS\n", ms, tflops(flops, ms));
  }
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dout));
  return ok ? 0 : 1;
}
```

Three places matter when reading it: how `correction` rescales the old accumulators `o` and `l`; how the mask is handled (a masked score becomes -inf and its exponential is naturally 0); and how NaN is avoided when a whole tile is masked and m is still -inf. These three are also what interviewers most often probe when they ask you to derive FlashAttention.

## Inference: prefill and decode {#推理prefill-与-decode}

LLM inference has two phases:

- **prefill**: the user's whole prompt at once (hundreds to tens of thousands of tokens), with long Q, K and V, much like a training forward pass. It is **compute-bound** and uses a FlashAttention-style kernel;
- **decode**: one new token per step, so only 1 query, but attending to **all** the previous tokens' K and V. Those are saved at every step, which is the **KV cache**. Decode's attention is essentially a matrix-vector product and is **memory-bound**: every step reads the whole KV cache from memory.

The KV cache's size: `2 × layers × KV heads × head dimension × sequence length × bytes per element`. For a model with 32 layers, 8 KV heads and a head dimension of 128, a token takes 2 × 32 × 8 × 128 × 2 B = 128 KB in BF16, so a 32K-long request needs 4 GB. That is why **GQA** (several query heads sharing one set of KV heads), **MLA** (DeepSeek compressing the KV into a low-rank latent) and KV cache quantization exist.

### PagedAttention {#pagedattention}

Allocating a contiguous stretch of KV cache per request at the maximum length wastes a great deal of memory (the real length is not known in advance) and fragments it. vLLM's **PagedAttention** borrows paging from operating systems:

- the KV cache is cut into fixed-size **pages**, 16 tokens each, say;
- each request has a **block table** recording which physical block holds its i-th logical block;
- physical blocks are allocated on demand and recycled when the request ends; several requests sharing a prefix can share physical blocks (prefix caching).

For the kernel, the only change is **one indirection when reading K and V**: token t's K lives at position `t % page_size` of physical block `block_table[t / page_size]`. SGLang's RadixAttention builds on this with a radix tree managing prefixes, maximizing KV cache reuse.

### Parallelism during decode: Flash-Decoding and split-K {#decode-阶段的并行flash-decoding-与-split-k}

During decode each (request, head) has only one query. With one block per (request, head), a small batch gives far fewer blocks than SMs and the GPU idles; and with a long sequence one block has to scan tens of thousands of tokens serially.

**Flash-Decoding** cuts the KV sequence into segments (split-K), one block per segment, each computing its own partial output and $(m, \ell)$, and merges them at the end with the online-softmax rule:

$$
m = \max_s m_s,\qquad \ell = \sum_s e^{m_s - m}\ell_s,\qquad O = \frac{\sum_s e^{m_s - m}\,\ell_s\,O_s}{\ell}
$$

where $O_s$ is segment s's already normalized output. That uses every SM even at batch 1.

### A paged decode attention kernel (with GQA) {#一个分页-decode-attention-kernel支持-gqa}

The kernel below implements decode attention: one block per (sequence, query head), with K and V read from the paged KV cache through the block table, and several query heads sharing one KV head (GQA). For clarity it does no split-K, and works in three steps:

1. each warp owns some tokens, and its 32 threads compute the dot product of q and k in parallel (each thread taking D/32 dimensions, then a warp reduction), writing the scores to shared memory;
2. the block finds the maximum and the sum of exponentials;
3. each thread owns one output dimension and walks every token accumulating $p_t V_t$. At any moment neighbouring threads read neighbouring dimensions of V, so the accesses coalesce.

```cuda title="paged_decode.cu"
// paged_decode.cu - decode attention over a paged KV cache, with GQA
// build: nvcc -O3 -arch=sm_75 paged_decode.cu -o paged_decode
// the KV cache layout: [num_physical_blocks, PAGE, num_kv_heads, D]
#include "common.cuh"

constexpr int PAGE = 16;          // 16 tokens per physical block
constexpr int kThreads = 128;     // 4 warps
constexpr int kMaxCtx = 4096;     // the longest context this example supports (the scores live in shared memory)

__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int m = 16; m > 0; m /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
  return v;
}

template <int D>
__global__ void __launch_bounds__(kThreads)
paged_decode_attention(const float* __restrict__ q,          // [num_seqs, num_heads, D]
                       const float* __restrict__ k_cache,    // [num_blocks, PAGE, num_kv_heads, D]
                       const float* __restrict__ v_cache,
                       const int* __restrict__ block_table,  // [num_seqs, max_blocks_per_seq]
                       const int* __restrict__ context_lens, // [num_seqs]
                       float* __restrict__ out,              // [num_seqs, num_heads, D]
                       int num_heads, int num_kv_heads, int max_blocks_per_seq, float scale) {
  static_assert(D % 32 == 0 && D <= kThreads, "one output dim per thread");
  constexpr int kDimsPerLane = D / 32;
  __shared__ float scores[kMaxCtx];
  __shared__ float red[kThreads / 32];

  const int seq = blockIdx.y, head = blockIdx.x;
  const int kv_head = head / (num_heads / num_kv_heads);   // GQA: several query heads share one KV head
  const int ctx = context_lens[seq];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = kThreads / 32;
  const int* table = block_table + static_cast<size_t>(seq) * max_blocks_per_seq;

  // each lane holds kDimsPerLane dimensions of q
  const float* qp = q + (static_cast<size_t>(seq) * num_heads + head) * D;
  float qr[kDimsPerLane];
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) qr[i] = qp[lane + i * 32] * scale;

  // step 1: the scores. Warp w handles tokens w, w + nwarps, ...
  float local_max = -INFINITY;
  for (int t = warp; t < ctx; t += nwarps) {
    const int phys = table[t / PAGE], slot = t % PAGE;
    const float* kp = k_cache + ((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kv_head) * D;
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) dot += qr[i] * kp[lane + i * 32];
    dot = warp_sum(dot);
    if (lane == 0) scores[t] = dot;
    local_max = fmaxf(local_max, dot);
  }

  // step 2: the block's maximum, then the exponentials and their sum
  if (lane == 0) red[warp] = local_max;
  __syncthreads();
  float mx = lane < nwarps ? red[lane] : -INFINITY;
  mx = warp_max(mx);            // every warp computes it, with the same result, which saves a broadcast
  __syncthreads();
  float local_sum = 0.f;
  for (int t = threadIdx.x; t < ctx; t += kThreads) {
    float p = __expf(scores[t] - mx);
    scores[t] = p;
    local_sum += p;
  }
  local_sum = warp_sum(local_sum);
  if (lane == 0) red[warp] = local_sum;
  __syncthreads();
  float sum = lane < nwarps ? red[lane] : 0.f;
  sum = warp_sum(sum);
  const float inv_sum = 1.f / sum;

  // step 3: the output. Thread d owns dimension d
  if (threadIdx.x < D) {
    const int d = threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < ctx; ++t) {
      const int phys = table[t / PAGE], slot = t % PAGE;
      acc += scores[t] * v_cache[((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kv_head) * D + d];
    }
    out[(static_cast<size_t>(seq) * num_heads + head) * D + d] = acc * inv_sum;
  }
}

int main() {
  constexpr int D = 128;
  const int num_seqs = 4, num_heads = 32, num_kv_heads = 8;
  const std::vector<int> ctx_lens = {1, 17, 1000, 4096};   // a range of lengths, including one that does not fill a page
  int max_blocks_per_seq = 0, total_blocks = 0;
  for (int c : ctx_lens) {
    max_blocks_per_seq = std::max(max_blocks_per_seq, (c + PAGE - 1) / PAGE);
    total_blocks += (c + PAGE - 1) / PAGE;
  }
  // physical blocks assigned in a shuffled order, imitating real paging
  std::vector<int> perm(total_blocks);
  for (int i = 0; i < total_blocks; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), std::mt19937(7));
  std::vector<int> table(static_cast<size_t>(num_seqs) * max_blocks_per_seq, -1);
  for (int s = 0, next = 0; s < num_seqs; ++s)
    for (int b = 0; b < (ctx_lens[s] + PAGE - 1) / PAGE; ++b) table[s * max_blocks_per_seq + b] = perm[next++];

  const size_t cache_elems = static_cast<size_t>(total_blocks) * PAGE * num_kv_heads * D;
  const size_t q_elems = static_cast<size_t>(num_seqs) * num_heads * D;
  std::vector<float> hq(q_elems), hk(cache_elems), hv(cache_elems), ref(q_elems), got(q_elems);
  fill_random(hq, 1);
  fill_random(hk, 2);
  fill_random(hv, 3);
  const float scale = 1.f / std::sqrt(static_cast<float>(D));

  // the CPU reference: the KV laid out in logical order
  for (int s = 0; s < num_seqs; ++s)
    for (int h = 0; h < num_heads; ++h) {
      const int kvh = h / (num_heads / num_kv_heads), ctx = ctx_lens[s];
      std::vector<double> sc(ctx);
      double mx = -1e300;
      for (int t = 0; t < ctx; ++t) {
        const int phys = table[s * max_blocks_per_seq + t / PAGE], slot = t % PAGE;
        const float* kp = &hk[((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kvh) * D];
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += double(hq[(s * num_heads + h) * D + d]) * kp[d];
        sc[t] = dot * scale;
        mx = std::max(mx, sc[t]);
      }
      double sum = 0;
      for (int t = 0; t < ctx; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
      for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int t = 0; t < ctx; ++t) {
          const int phys = table[s * max_blocks_per_seq + t / PAGE], slot = t % PAGE;
          acc += sc[t] * hv[((static_cast<size_t>(phys) * PAGE + slot) * num_kv_heads + kvh) * D + d];
        }
        ref[(s * num_heads + h) * D + d] = static_cast<float>(acc / sum);
      }
    }

  float *dq, *dk, *dv, *dout;
  int *dtable, *dctx;
  CUDA_CHECK(cudaMalloc(&dq, q_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, q_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dk, cache_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, cache_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dtable, table.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dctx, num_seqs * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), q_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), cache_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), cache_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dtable, table.data(), table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dctx, ctx_lens.data(), num_seqs * sizeof(int), cudaMemcpyHostToDevice));

  dim3 grid(num_heads, num_seqs);
  auto launch = [&] {
    paged_decode_attention<D><<<grid, kThreads>>>(dq, dk, dv, dtable, dctx, dout, num_heads, num_kv_heads,
                                                  max_blocks_per_seq, scale);
  };
  launch();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dout, q_elems * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), q_elems, 1e-3f, 1e-5f);
  float ms = time_ms(launch);
  size_t kv_bytes = 0;
  for (int c : ctx_lens) kv_bytes += static_cast<size_t>(c) * num_heads * D * 2 * sizeof(float);
  std::printf("paged decode: %.3f ms, %.1f GB/s of K/V reads (GQA heads re-read shared KV)\n", ms, gbps(kv_bytes, ms));
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dout));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dtable));
  CUDA_CHECK(cudaFree(dctx));
  return ok ? 0 : 1;
}
```

There is plenty left to improve in this kernel, which makes good exercise and interview material:

- **GQA's repeated reads**: one KV head is read once by each of the 4 blocks of the 4 query heads sharing it. Better to have one block handle every query head sharing a KV head, reading the KV once;
- **split-K**: split as in the Flash-Decoding section above for long sequences and small batches;
- **vectorization and low precision**: the KV cache is really BF16 or FP8, read 128 bits at a time and converted to FP32 in registers;
- **keep the scores out of shared memory**: with a long context the scores do not fit, so switch to online softmax accumulating the output as it scans, which is what the decode kernels in FlashInfer and vLLM do.

In production, FlashInfer, FlashAttention's `flash_attn_with_kvcache` and the attention backends built into vLLM and SGLang all provide heavily optimized paged prefill and decode kernels. Reading their source is excellent further practice.

!!! interview "Answering in an interview"
    FlashAttention is a certainty: standard attention's bottleneck is the memory traffic of the N×N intermediates; FlashAttention computes in tiles and merges with online softmax on chip, with an exact result and the footprint down from $O(N^2)$ to $O(N)$; FA2 parallelizes along the sequence and splits across warps by Q, and FA3 uses Hopper's TMA, wgmma and warp specialization. On the inference side: prefill is compute-bound and decode memory-bound, with only one query per step, so split-KV (Flash-Decoding) parallelizes long sequences; PagedAttention adds one block-table indirection in the kernel; and under GQA one block should handle every query head sharing a KV head, to avoid reading the KV repeatedly.

## Exercises {#练习}

**1. Derive it by hand.** With online softmax, for one row of scores `[1, 3]` (the first tile) and `[2, 5]` (the second), with the matching rows of V being the scalars `[10, 20]` and `[30, 40]`, write out m, ℓ and O step by step and check that the final result equals the softmax-weighted sum computed directly.

??? success "Answer"
    - First tile: m = 3, $\tilde{p} = [e^{-2}, 1] ≈ [0.1353, 1]$, ℓ = 1.1353, O = 0.1353 × 10 + 1 × 20 = 21.353.
    - Second tile: the tile maximum is 5, so m' = 5 and correction = $e^{3-5} = e^{-2} ≈ 0.1353$; $\tilde{p} = [e^{-3}, 1] ≈ [0.0498, 1]$; ℓ = 1.1353 × 0.1353 + 1.0498 ≈ 1.2034; O = 21.353 × 0.1353 + 0.0498 × 30 + 1 × 40 ≈ 2.890 + 1.494 + 40 = 44.384.
    - The final output is O / ℓ ≈ 44.384 / 1.2034 ≈ 36.88.
    - Directly: the weights are $\propto [e^{1}, e^{3}, e^{2}, e^{5}] = [2.718, 20.09, 7.389, 148.4]$ summing to 178.6; the weighted sum is (27.18 + 401.7 + 221.7 + 5936) / 178.6 ≈ 36.88. The two agree.

**2. Implement the split-K merge.** Given S segments' partial results, each with a normalized output $O_s$ (of length D) and $(m_s, \ell_s)$, write a kernel producing the final output.

??? success "Approach"
    One block per (sequence, head) with thread d owning dimension d: have any thread (or every thread redundantly) compute $m = \max_s m_s$ and the weights $w_s = e^{m_s - m} \ell_s$ with $\ell = \sum_s w_s$; then $O[d] = \sum_s w_s O_s[d] / \ell$. S is usually small (a few to a few dozen), so the weights can go into shared memory first. Note that a segment with no valid tokens (ℓ = 0, m = -inf) has to be skipped to avoid NaN.

**3. A thinking exercise: GQA's read amplification.** In `paged_decode.cu`, num_heads = 32 and num_kv_heads = 8. How many times is the KV cache actually read from memory per decode step? How would you make it once?

??? success "Answer"
    Each KV head is shared by 32 / 8 = 4 query heads, and each query head is its own block, so the KV cache is read 4 times (L2 catches some of it, but not reliably). The fix is one block per (sequence, KV head) computing all 4 query heads of that KV head at once: read a row of K once and dot it with all 4 q, read a row of V once and accumulate into all 4 outputs. Decode is memory-bound, so this change is worth close to a multiple in speed. It is exactly the GQA specialization libraries like FlashInfer make.

## Summary {#小结}

- [x] Standard attention's bottleneck is the memory traffic of the N×N intermediates; FlashAttention keeps them on chip with tiling plus online softmax, with an exact result.
- [x] FA2 parallelizes along the sequence and splits across warps by Q; FA3 uses Hopper's TMA, wgmma and warp specialization.
- [x] Prefill is compute-bound and decode memory-bound; the KV cache's size decides decode's cost and how much concurrency fits.
- [x] PagedAttention manages the KV cache with a block table, adding one indirection in the kernel; Flash-Decoding parallelizes long sequences with split-K.
- [x] Under GQA, one block should handle every query head sharing a KV head, to avoid reading it repeatedly.
