# Multi-GPU and NCCL

<p class="lead">A large model does not fit on one card, and training or serving it on one card is out of the question anyway. How a model is split across cards, what interconnect they use, how collective communication is implemented, and how communication overlaps with computation are topics an AI infrastructure role cannot avoid. This chapter makes these concepts clear and writes a multi-GPU all-reduce with NCCL.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What collective communication does each layer of tensor parallelism (TP) need? And expert parallelism (EP)?
    2. How much data does each card send in a ring all-reduce? Why is it nearly independent of the number of cards?
    3. What is the difference between algbw and busbw in a NCCL test report?
    4. During decode, is TP's all-reduce usually latency-bound or bandwidth-bound?
    5. How do you overlap communication with computation?

??? success "Answers (try it yourself first, then expand)"
    1. TP: one all-reduce for each layer's attention and one for its MLP (summing the partial results of the row-split matrix multiply); EP: two all-to-alls per MoE layer (sending the tokens to the experts and bringing the results back).
    2. Each card sends $2(n-1)/n \times S$ bytes: reduce-scatter and all-gather each send $n-1$ chunks of $S/n$. As the card count grows this approaches $2S$, hence nearly independent of it.
    3. algbw = the data size ÷ the time, the bandwidth from the user's point of view; busbw converts it to the data the algorithm actually moves over the links (multiplied by $2(n-1)/n$ for all-reduce), which compares directly against a link's peak bandwidth and says whether it is saturated.
    4. Latency-bound: a decode step's all-reduce is only a few hundred KB, and each of the ring algorithm's $2(n-1)$ steps has a fixed cost, so almost all the time goes to latency.
    5. Put the communication on its own stream, parallel with the computation that does not depend on it: in training, bucket the gradients and all-reduce while the backward pass continues; split the computation into chunks and send one while computing the next; use a kernel with fused communication (computation and communication alternating within one kernel); and for MoE, use a dedicated all-to-all library and two-batch overlap.

## Why several cards, and how to split {#为什么要多卡以及怎么切分}

| Parallelism | What is split | Communication | Typical use |
| --- | --- | --- | --- |
| data parallelism (DP) | the data; a full model per card | a gradient all-reduce in training | training when the model fits one card; several replicas when serving |
| tensor parallelism (TP) | every matrix split across cards by rows or columns | 1-2 all-reduces per layer (or an all-gather + a reduce-scatter) | several cards in one machine (NVLink), the main device for inference |
| pipeline parallelism (PP) | split by layer | point-to-point activations between adjacent stages | across machines, very large models |
| expert parallelism (EP) | an MoE's experts on different cards | two all-to-alls per MoE layer (dispatching the tokens, collecting the results) | MoE models like DeepSeek and Qwen-MoE |
| sequence/context parallelism (SP/CP) | split along the sequence length | attention has to exchange K and V (Ring Attention, say) | very long context |

Take a Transformer's MLP: Megatron-style TP splits the first linear layer by **column** (each card gets part of the output and the activation function is computed locally) and the second by **row** (each card gets a partial result of the full shape), then does one **all-reduce** to sum them. The attention layer is the same: split by head, with one all-reduce after the output projection. So TP needs two all-reduces per layer.

## Interconnects {#互联}

| Interconnect | Bandwidth | Notes |
| --- | --- | --- |
| PCIe 4.0 / 5.0 x16 | about 32 / 64 GB/s per direction (theoretical) | all that consumer cards and some inference cards have; card-to-card traffic often has to go through the CPU |
| NVLink (H100, 4th generation) | 900 GB/s per card in total (bidirectional) | 8 cards in one machine fully connected through NVSwitch |
| NVLink (B200, 5th generation) | 1.8 TB/s per card in total (bidirectional) | GB200 NVL72 joins 72 cards into one NVLink domain |
| InfiniBand / RoCE | 400 Gb/s per network card (about 50 GB/s) | communication across machines; with GPUDirect RDMA the network card reads and writes device memory directly |

TP within a machine relies on NVLink; TP across machines costs too much in communication, so the usual arrangement is "TP within a machine, PP/EP/DP between machines".

## CUDA peer-to-peer access {#cuda-点对点访问}

Once two cards in the same machine enable peer access, they can copy directly, and even read and write each other's device memory inside a kernel (over NVLink or PCIe):

```cuda
int can = 0;
cudaDeviceCanAccessPeer(&can, 0, 1);
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);                    // device 0 can access device 1's memory
cudaMemcpyPeerAsync(dst_on_1, 1, src_on_0, 0, bytes, stream);
```

The **custom all-reduce** in vLLM and TensorRT-LLM rests on exactly this ability: for decode's small messages (tens of KB), each card reads the others' buffers directly and sums (one-shot), or reduce-scatters first and then all-gathers (two-shot), with lower latency than general-purpose NCCL.

## Collective communication {#集合通信}

| Operation | Meaning |
| --- | --- |
| broadcast | one card's data goes to all cards |
| reduce | all cards' data is summed (or maxed, and so on) onto one card |
| all-reduce | all cards' data is summed and every card has the result |
| all-gather | each card's chunk is concatenated and every card gets the whole |
| reduce-scatter | sum, then split, so each card gets one chunk of the result |
| all-to-all | every card sends a different chunk to every card (an MoE's token dispatch) |

One important identity: **all-reduce = reduce-scatter + all-gather**.

### ring all-reduce {#ring-all-reduce}

Who sends what to whom at each step, stepped through with the same tool as in the distributed training handbook:

<div class="aig-widget" data-widget="ringreduce"></div>

p cards form a ring and the data is cut into p chunks:

1. **the reduce-scatter phase** (p-1 steps): at each step every card sends one chunk to the next card, which adds what it received into its own corresponding chunk. After p-1 steps each card holds the complete sum of exactly one chunk;
2. **the all-gather phase** (p-1 steps): each card passes its finished chunk around the ring, and after p-1 steps every card has the whole result.

Each card sends $2(p-1) \cdot \frac{S}{p}$ bytes in all (S being the total size), which for larger p is about **2S, nearly independent of the card count**, and all the links work at once, using the bandwidth fully. The cost is $2(p-1)$ steps, each with a fixed latency, so **latency dominates for small messages**. NCCL chooses among ring, tree and, on NVSwitch, NVLS (reduction inside the switch) by the message size and the topology.

### Two ways to count bandwidth: algbw and busbw {#带宽的两种统计algbw-与-busbw}

`nccl-tests` (NVIDIA's official test tool) reports two bandwidths:

- **algbw** = S / the time, intuitive but not comparable between operations or card counts;
- **busbw**: converted to "the bandwidth each link actually carries". For all-reduce, busbw = algbw × $\frac{2(p-1)}{p}$. It compares directly against the hardware link's peak bandwidth.

```bash
# test all-reduce on 8 cards in one machine, messages from 8 B to 1 GB, doubling each time
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

## Programming with NCCL {#nccl-编程}

NCCL's basic flow: create a communicator, then start a collective on a given stream. The example below **drives several cards from one process** (`ncclCommInitAll`), filling each card with different data, doing one all-reduce and checking the result. With several processes (one per card, as training and inference frameworks usually do), use `ncclGetUniqueId` plus distributing the id over MPI or TCP plus `ncclCommInitRank`.

```cuda title="nccl_allreduce.cu"
// nccl_allreduce.cu - a NCCL all-reduce over several GPUs from one process
// build: nvcc -O3 -arch=sm_75 nccl_allreduce.cu -o nccl_allreduce -lnccl
#include "common.cuh"
#include <nccl.h>

#define NCCL_CHECK(call)                                                                 \
  do {                                                                                   \
    ncclResult_t r_ = (call);                                                            \
    if (r_ != ncclSuccess) {                                                             \
      std::fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                           \
    }                                                                                    \
  } while (0)

__global__ void fill(float* x, int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = value + i % 7;
}

int main() {
  int ngpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ngpus));
  if (ngpus < 2) {
    std::printf("SKIP: needs at least 2 GPUs, found %d\n", ngpus);
    return 0;
  }
  const int n = 32 * 1024 * 1024;   // 128 MB per card
  std::vector<int> devs(ngpus);
  for (int i = 0; i < ngpus; ++i) devs[i] = i;
  std::vector<ncclComm_t> comms(ngpus);
  NCCL_CHECK(ncclCommInitAll(comms.data(), ngpus, devs.data()));

  std::vector<float*> buf(ngpus);
  std::vector<cudaStream_t> streams(ngpus);
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&buf[g], n * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&streams[g]));
    fill<<<(n + 255) / 256, 256, 0, streams[g]>>>(buf[g], n, static_cast<float>(g));   // card g: g + i % 7
    CUDA_CHECK_LAST();
  }

  auto allreduce = [&] {
    // one thread starting collectives for several communicators has to wrap them in a group, or it deadlocks
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < ngpus; ++g)
      NCCL_CHECK(ncclAllReduce(buf[g], buf[g], n, ncclFloat, ncclSum, comms[g], streams[g]));   // in place
    NCCL_CHECK(ncclGroupEnd());
  };
  allreduce();
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }

  // expected: sum_g (g + i % 7) = ngpus * (ngpus - 1) / 2 + ngpus * (i % 7)
  std::vector<float> h(n);
  size_t bad = 0;
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMemcpy(h.data(), buf[g], n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) bad += h[i] != ngpus * (ngpus - 1) / 2.f + ngpus * static_cast<float>(i % 7);
  }
  std::printf("all-reduce across %d GPUs: %s (%zu mismatches)\n", ngpus, bad ? "FAIL" : "PASS", bad);

  // timing: on card 0's stream (the other cards synchronize to finish)
  CUDA_CHECK(cudaSetDevice(0));
  GpuTimer t;
  const int iters = 20;
  t.start(streams[0]);
  for (int it = 0; it < iters; ++it) allreduce();
  float ms = t.stop(streams[0]) / iters;
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }
  const double algbw = n * sizeof(float) / (ms * 1e-3) / 1e9;
  std::printf("%.3f ms, algbw %.1f GB/s, busbw %.1f GB/s\n", ms, algbw, algbw * 2.0 * (ngpus - 1) / ngpus);

  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaFree(buf[g]));
    CUDA_CHECK(cudaStreamDestroy(streams[g]));
    NCCL_CHECK(ncclCommDestroy(comms[g]));
  }
  return bad ? 1 : 0;
}
```

A few common traps:

- **group calls**: one thread starting operations for several communicators has to wrap them in `ncclGroupStart/End`;
- **every rank must start the same collectives in the same order**, or it hangs. The most common reason a distributed program "freezes" is that some rank called a collective one time too few or too many;
- setting `NCCL_DEBUG=INFO` while debugging shows the algorithm NCCL chose, the interconnect it used and the topology.

## Overlapping communication with computation {#通信与计算重叠}

Collective communication runs on its own stream (and NCCL's kernels take a share of the SMs). The common ways to overlap:

- **training**: during the backward pass, a gradient bucket starts its all-reduce as soon as it is complete while the next layer's gradients keep computing, which is what PyTorch DDP does;
- **chunked overlap**: cut both a large GEMM and the all-reduce after it into chunks, so chunk i's communication runs parallel with chunk i+1's computation;
- **kernels fusing communication with computation**: compute a GEMM while reading and writing other cards' data over NVLink in the same kernel, fusing all-gather + GEMM or GEMM + reduce-scatter; FLUX and NVIDIA's Transformer Engine userbuffers both work on this;
- **MoE's all-to-all**: DeepSeek's open-source DeepEP provides two sets of kernels for an MoE's dispatch and combine, high-throughput (training, prefill) and low-latency (decode), the low-latency one using pure RDMA and supporting overlap with computation.

!!! interview "Answering in an interview"
    On multi-GPU communication: TP all-reduces per layer, EP does two all-to-alls per MoE layer, and PP transfers point to point between stages; within a machine it is NVLink / NVSwitch and between machines InfiniBand / RoCE plus GPUDirect RDMA, hence TP within a machine and PP / EP / DP between them. A ring all-reduce has each card send about 2S bytes, nearly independent of the card count, but takes 2(n−1) steps, so decode's small messages are latency-bound rather than bandwidth-bound. In a test report, read busbw and compare it against the link's peak; overlapping communication with computation rests on gradient bucketing, chunked pipelining, fused kernels and a dedicated all-to-all library.

!!! info "Related chapters"
    - [collective communication primitives](train://basics/collectives/) (distributed training: the five primitives and a ring all-reduce's traffic)
    - [GPU interconnects and networking](serving://comm/interconnect/), [collective communication: NCCL's algorithms and protocols](serving://comm/nccl/) (inference systems)
    - [multi-GPU systems: NVLink, topology, partitioning and power](cs://arch/multi-gpu/) (computer fundamentals)

## Exercises {#练习}

**1. Estimate TP's communication cost.** A model with hidden = 8192 runs TP on 8 H100s in BF16. Estimate one all-reduce's time in each of these cases and say whether it is latency- or bandwidth-bound: (a) decode with batch = 1; (b) prefill with 8192 tokens at once. Assume about 400 GB/s of usable NVLink bandwidth per direction and a fixed cost of about 10-20 µs per all-reduce.

??? success "Answer"
    - **(a)**: the data is 1 × 8192 × 2 B = 16 KB. By bandwidth, 1.75 × 16 KB / 400 GB/s ≈ 0.07 µs, far below the fixed cost, so it is **latency-bound** and takes about that fixed cost itself (a dozen or so microseconds). Two per layer over several dozen layers is a millisecond, which is exactly why decode wants a custom all-reduce and CUDA Graphs.
    - **(b)**: the data is 8192 × 8192 × 2 B = 128 MB. A ring all-reduce has each card send about 2 × 7/8 × 128 MB ≈ 224 MB, taking about 224 MB / 400 GB/s ≈ 0.56 ms, so it is **bandwidth-bound**. Here overlapping communication with computation becomes very important.

**2. Why is all-to-all harder to optimize than all-reduce?**

??? success "Answer"
    An all-reduce has a fixed size and a regular pattern, and the ring/tree algorithms can fill every link evenly. An MoE's all-to-all depends on the routing result: how much each card sends to each other card differs and changes every step, and the load can be badly unbalanced (the card holding a popular expert receives more tokens); across machines it also has to use both NVLink and RDMA well. Hence the need for a dedicated implementation (DeepEP, say) and for load balancing at the model level (an auxiliary loss, auxiliary-loss-free bias adjustment, redundant experts and so on).

## Summary {#小结}

- [x] TP all-reduces per layer, EP does two all-to-alls per MoE layer, and PP transfers point to point between stages.
- [x] Within a machine it is NVLink/NVSwitch and between machines InfiniBand/RoCE + GPUDirect RDMA; usually TP within a machine and PP/EP/DP between them.
- [x] A ring all-reduce has each card send about 2S bytes, nearly independent of the card count, but takes many steps, so small messages are latency-bound.
- [x] Compare busbw against the link's peak bandwidth; wrap operations on several communicators in a group, and every rank must call them in the same order.
- [x] Overlapping communication with computation: gradient bucketing, chunked pipelining, fused kernels, a dedicated all-to-all library.
