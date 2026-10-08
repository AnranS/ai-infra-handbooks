# A bank of interview questions

<p class="lead">This page collects the questions that come up most in CUDA / AI infrastructure interviews, grouped by topic, each with a reference answer in points and a link to where the body of the handbook explains it. The suggested use: cover the answer and say it out loud yourself first, and go back to the chapter for whatever you cannot explain. An interviewer cares more about whether you can say "why", and whether you have numbers you measured yourself.</p>

## What interviews usually cover {#面试通常考什么}

| Stage | Content | How to prepare |
| --- | --- | --- |
| basic questions | GPU architecture, the memory hierarchy, the execution model, synchronization | this page's first three sections |
| writing a kernel by hand | reduction, transpose, softmax, GEMM, LayerNorm, prefix sum | write each from scratch once, and be able to explain the optimizations as you write |
| performance analysis | given a kernel or a scenario, find the bottleneck and propose optimizations | the roofline + Nsight's metrics, see [profiling](../tools/profiling.md) |
| domain knowledge | FlashAttention, the KV cache, quantization, parallelism strategies, inference engines | the second half of this page |
| digging into your projects | an optimization you did: where the bottleneck was, how you verified it, how much it gained | [portfolio projects](projects.md) |

## GPU architecture and the programming model {#gpu-架构与编程模型}

??? note "1. How do threads, warps, blocks, grids and SMs relate?"
    A grid is made of blocks and a block of threads. Each block as a whole is scheduled onto one SM and does not migrate while it runs; one SM can hold several blocks at once. Every 32 consecutive threads in a block form a warp, which is the unit of scheduling and execution, and a warp's threads execute the same instruction (SIMT). See [GPU architecture](../basics/gpu-architecture.md#软件层次如何映射到硬件).

??? note "2. How does a GPU hide memory latency? How does that differ from a CPU?"
    A CPU lowers a single thread's latency with large caches, out-of-order execution and branch prediction. A GPU keeps a great many warps resident on each SM, and when one warp waits on memory the scheduler switches to another ready warp at no cost, hiding latency with throughput. So a GPU needs enough parallelism: enough resident warps (TLP) and enough independent instructions within each thread (ILP).

??? note "3. What is warp divergence? How do you avoid it?"
    When threads in one warp take different branches, the branches run one after the other with the threads not on that path masked off, so the time is the sum of the branches. Divergence only happens within a warp, so make the branch condition uniform within a warp (grouping by `threadIdx.x / 32`), or write it without a branch using `fmaxf` or a select. See [the execution model](../basics/execution.md#分支发散).

??? note "4. What is occupancy? Is higher always better?"
    The ratio of the warps actually resident on an SM to the hardware limit, constrained by four resources: threads per SM, blocks, registers and shared memory. Higher is not always better: kernels like GEMM and FlashAttention trade a lot of registers and shared memory for data reuse and ILP, running at very low occupancy and very high performance. It is bandwidth-bound kernels with little work per thread that depend on high occupancy.

??? note "5. How do you choose the block size and the grid size?"
    Make the block a multiple of 32, with 128 or 256 a common starting point; make the grid large enough that every SM gets several blocks, watching for the tail effect; for very large data use a grid-stride loop and size the grid by the SM count. Measurement decides in the end, and `cudaOccupancyMaxPotentialBlockSize` is a reasonable reference.

??? note "6. Does a program compiled with `-arch=sm_80` run on an H100? On a T4?"
    On an H100, yes: the fatbinary contains compute_80 PTX and the driver JIT-compiles it into sm_90 SASS. On a T4 (sm_75), no: there is no matching SASS and the PTX is newer than it. Code with the `a` suffix, sm_90a, runs only on Hopper. See [the compilation flow](../basics/gpu-architecture.md#编译流程ptx-与-sass).

??? note "7. What happens if `__syncthreads()` is inside an if?"
    Every thread in the block has to reach the same `__syncthreads()`, and the behaviour is undefined when only some of them can, usually a deadlock or a wrong result. Out-of-range threads should not return early: skip the computation but still take part in the synchronization.

??? note "8. What changed with independent thread scheduling from Volta onward?"
    Each thread in a warp has its own program counter, diverged threads can interleave, and implicit synchronization within a warp is no longer guaranteed. Old code that relied on "a warp executes in lockstep" (unrolling the last warp through volatile shared memory, say) is no longer safe and has to use explicitly synchronizing primitives like `__shfl_*_sync` and `__syncwarp()`.

## The memory hierarchy and access optimization {#内存层次与访存优化}

??? note "9. Describe a GPU's memory hierarchy."
    Registers (private to a thread, the fastest) → shared memory / L1 (per SM, shared by a block, tens of cycles) → L2 (shared by the whole GPU, tens of MB) → device memory HBM (tens of GB, hundreds of cycles). There is also constant memory (with a broadcasting cache) and local memory (register spills, actually in device memory). Optimization is all about keeping data at a higher level and reusing it many times. See [the memory hierarchy](../basics/gpu-architecture.md#内存层次速览).

??? note "10. What is coalesced access?"
    Device memory transfers in units of 32-byte sectors. A warp's accesses are coalesced if they land in as few sectors as possible (adjacent threads at contiguous addresses); a strided access drops the effective bandwidth several times over. For two-dimensional data, let `threadIdx.x` vary along the contiguous dimension, and prefer SoA layouts. See [coalesced access](../basics/memory.md#全局内存合并访问).

??? note "11. What is a bank conflict? How do you fix it?"
    Shared memory has 32 banks, each 4 bytes wide. When several threads in a warp access different addresses in the same bank the accesses serialize; the same address is a broadcast and does not conflict. The typical case is accessing a `[32][32]` array by column. The fixes: padding (`[32][33]`), a swizzle (xoring the index), or a different access pattern. Nsight Compute's `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` confirms it.

??? note "12. When do you use shared memory?"
    Two cases: the data is read several times by several threads in the block (the tiled reuse of GEMM or convolution); or threads in the block have to exchange data (reduction, transpose, scan). If the data is read once, putting it in shared memory is pointless.

??? note "13. What does a vectorized access (float4) gain, and what does it require?"
    One instruction reads 16 bytes, cutting instruction count and address arithmetic and increasing the data in flight per thread, which helps saturate bandwidth. It requires the address to be 16-byte aligned, and the tail has to be handled when the element count is not a multiple of 4.

??? note "14. What is a register spill? How do you find and handle it?"
    When registers run short the compiler puts variables in local memory (in device memory, through the caches) and performance drops sharply. `-Xptxas -v` shows the spilled bytes. The handling: hold less data per thread, unroll loops so local array indices become constants, adjust `__launch_bounds__`.

??? note "15. What is pinned memory? Why does it matter?"
    Page-locked memory is never swapped out by the operating system and DMA can reach it directly. `cudaMemcpyAsync` is only truly asynchronous when the host memory is pinned, which is also the prerequisite for overlapping copies with computation. It is a limited system resource and cannot be allocated without restraint.

## Performance analysis {#性能分析}

??? note "16. What is the roofline model? How do you use it?"
    The performance ceiling = min(peak compute, arithmetic intensity × peak bandwidth). Below the ridge point (peak compute / bandwidth) a kernel is bandwidth-bound and the direction is less traffic and better bandwidth utilization; above it the kernel is compute-bound and the direction is faster units and less wasted computation. To judge a kernel, compare the measurement against its own ceiling. See [the roofline](../basics/execution.md#roofline-模型).

??? note "17. A kernel is slow. How do you analyze it?"
    First confirm with nsys that the time really goes to this kernel rather than to launch overhead, copies or the CPU; then look at Speed Of Light in ncu to decide whether it is bandwidth-, compute- or latency-bound; then read the matching section: Memory Workload (coalescing, bank conflicts, hit rates at each level), Warp State (the stall reasons), Occupancy; locate the line with the Source view; and measure again after the change to confirm.

??? note "18. How do you optimize a bandwidth-bound kernel?"
    First reduce the traffic: operator fusion, data reuse, lower precision / quantization; then improve bandwidth utilization: coalesced access, vectorization, more elements per thread to raise the requests in flight, higher occupancy, no bank conflicts. The target is 80%-90% of peak effective bandwidth.

??? note "19. How do you optimize a compute-bound kernel?"
    Use the Tensor Cores (low-precision matrix operations); remove redundant computation; use the fast math functions; raise the ILP to keep the pipelines full; check for divergence; confirm it is not limited by instruction issue or register dependencies.

??? note "20. What do you do about large gaps between kernels?"
    That is launch overhead or the CPU's issue rate. Submit the whole graph at once with CUDA Graphs; fuse the small kernels; remove implicit synchronization; overlap the CPU's scheduling with the GPU's execution. Inference engines use CUDA Graphs for decode across the board.

## The classic kernels {#经典-kernel}

??? note "21. Write a reduction by hand, optimize it, and explain every step."
    A sequential-addressing shared-memory tree reduction (no divergence, no bank conflicts) → shuffles for the last warp → each thread first accumulating many elements in a register with a grid-stride loop and float4 reads → a two-level warp shuffle + shared memory reduction → combining the blocks' results with a second kernel or an atomic (noting that a floating-point atomic add is not deterministic). The measure is effective bandwidth. See [reduction](../kernels/reduction.md#怎么讲清楚).

??? note "22. What is the difference between `__shfl_down_sync` and `__shfl_xor_sync` for a reduction?"
    The down version reads lane + offset's value at each step, leaving only lane 0 with the complete result; the xor version is a butterfly exchange that leaves every lane with the result, which suits softmax, LayerNorm and anything else where every thread needs the reduction. The first argument is the mask of participating threads.

??? note "23. How do you optimize a matrix transpose?"
    The naive version necessarily leaves one side uncoalesced. Go through shared memory: read a tile in coalesced, write it out coalesced at the transposed coordinates, which moves the strided access into shared memory; then pad to `[32][33]` to remove the column read's bank conflicts; and use a 32×8 block with 4 elements per thread. The ceiling is a copy of the same shape. See [matrix transpose](../kernels/transpose.md).

??? note "24. What is the optimization path for GEMM?"
    Naive (noting that x corresponds to the column) → shared-memory tiling (traffic down to 1/TILE) → one- and two-dimensional register tiling (8×8 per thread, which cuts the ratio of shared-memory reads to computation sharply and moves the bottleneck to computation) → vectorization and storing A transposed → double buffering / a multi-stage cp.async pipeline → warp tiling, removing bank conflicts, swizzling → the Tensor Cores (mma/wgmma) → the TMA + warp specialization. The core idea is raising the data reuse level by level: "traffic grows with the perimeter, computation with the area". See [GEMM](../kernels/gemm.md#怎么讲清楚).

??? note "25. Why does softmax subtract the maximum? What is online softmax?"
    To stop the exponential from overflowing. Online softmax keeps the running maximum m and the exponential sum d relative to m, updating by $d' = d e^{m - m'} + e^{x - m'}$ as each new element arrives; two (m, d) pairs can also be merged, so the reduction parallelizes and three passes become two. This is FlashAttention's basis. See [online softmax](../kernels/softmax-norm.md#online-softmax一次遍历求出最大值和指数和).

??? note "26. How do you implement LayerNorm/RMSNorm? How do you compute the variance stably?"
    One block (or warp) handles one row, with the whole row cached in registers, a block reduction for the statistics, and then the elementwise transform. Compute the variance in two passes (the mean first, then the squared differences) or with Welford's online algorithm, avoiding the cancellation error of $E[x^2] - E[x]^2$. Compute the statistics in FP32. Fusing the residual add (fused_add_rms_norm) saves one full read and write.

??? note "27. How do you do a parallel prefix sum?"
    Within a warp, Hillis-Steele with `__shfl_up_sync` (5 steps); within a block, each warp scans and then the warps' totals are scanned once more and added back; at device level, the three-step "scan the tiles → scan the tile totals → add the offsets back", or CUB's single-pass decoupled look-back. Applications: stream compaction, radix sort, an MoE's token dispatch. See [prefix sum](../kernels/scan.md).

??? note "28. How do you build a histogram efficiently with atomics?"
    Keep a per-block private histogram in shared memory first, accumulating with shared-memory atomics, and have each block atomically add its result into the global one at the end. Under heavy contention, keep several copies within one block.

## Tensor Cores and new hardware {#tensor-core-与新硬件}

??? note "29. What is a Tensor Core? How is it used?"
    A unit dedicated to small matrix multiply-accumulates (D = AB + C), issued per warp (per warpgroup on Hopper), an order of magnitude more capable than the CUDA cores. The interfaces: WMMA (simple, with an opaque layout), mma.sync + ldmatrix (Ampere, with an explicit layout), wgmma (Hopper, asynchronous, operands from shared memory) and tcgen05 (Blackwell). In practice they are reached through CUTLASS/CuTe and Triton. See [Tensor Cores](../advanced/tensor-core.md).

??? note "30. What is the difference between FP16, BF16 and FP8?"
    FP16: 5 exponent bits and 10 mantissa bits, reasonably precise but narrow in range; BF16: 8 exponent bits and 7 mantissa bits, the same range as FP32, the mainstream for large models; FP8 comes in E4M3 (precision first) and E5M2 (range first) and needs scaling factors. A matrix multiply usually accumulates in FP32.

??? note "31. What problem does cp.async solve?"
    It copies asynchronously from global memory straight into shared memory without going through registers, and the issuing thread can keep computing; with commit/wait_group it builds a multi-stage pipeline that hides the memory latency behind the computation. See [cp.async](../advanced/async-hopper.md#cpasync-与多级流水-sm_80).

??? note "32. What are Hopper's important new features?"
    The TMA (one thread starts a whole multidimensional tile's transfer and the hardware handles the addressing, the bounds and the swizzle); thread block clusters and distributed shared memory; wgmma (warpgroup-level asynchronous matrix instructions); FP8 Tensor Cores; and producer-consumer warp specialization built on mbarriers. See [Hopper](../advanced/async-hopper.md).

??? note "33. What is warp specialization?"
    Dividing the labour among a block's warps: the producer warps only move data with the TMA and the consumer warps only compute with wgmma, the two synchronizing through an mbarrier ring buffer in shared memory, so copying and computing stay parallel. setmaxnreg can also move registers from the producers to the consumers. Hopper's high-performance GEMM and FlashAttention-3 both have this structure.

## Attention and inference {#注意力与推理}

??? note "34. How does FlashAttention work? Why is it fast?"
    Standard attention writes the N×N S and P back to device memory, which is bandwidth-bound. FlashAttention tiles Q, K and V, computes one tile of $QK^\top$ on chip, updates each row's maximum and exponential sum with online softmax, and rescales the accumulated output accordingly, so the N×N matrix never goes back to device memory. The result is exact and both the memory traffic and the memory use drop sharply. See [FlashAttention](../advanced/attention.md#flashattention分块--online-softmax).

??? note "35. What did FlashAttention-2 and -3 improve over the first version?"
    FA2: parallelism along the sequence dimension (a fuller GPU at small batch and long sequence), fewer non-matrix-multiply operations (dividing by ℓ only at the end), and splitting by Q between warps to avoid shared-memory communication. FA3: aimed at Hopper, the TMA + wgmma + warp specialization, two warpgroups ping-ponging so softmax overlaps the GEMM, and FP8 support.

??? note "36. How do prefill's and decode's performance characteristics differ?"
    Prefill processes the whole input with long Q/K/V, so both the GEMMs and attention are compute-bound and TTFT (time to first token) is what matters. Decode generates one token per step, the linear layers become GEMVs and attention has to read the whole KV cache, so it is bandwidth-bound and TPOT (time per output token) and throughput are what matter. Hence decode is optimized by batching, quantization, KV cache compression and CUDA Graphs.

??? note "37. How large is the KV cache? How do you estimate it?"
    Per token: 2 (K and V) × layers × KV heads × head dimension × bytes per element. A model with 32 layers, 8 KV heads, head dimension 128 and BF16 is about 128 KB per token, so a 32K context is about 4 GB. GQA, MLA, KV cache quantization and prefix sharing all exist to bring that down.

??? note "38. What problem does PagedAttention solve? What changes in the kernel?"
    Preallocating a contiguous KV cache wastes a lot of memory and fragments it, because the length is unknown. PagedAttention cuts the KV cache into fixed-size blocks, with each request's block table recording the logical-to-physical block mapping, allocated on demand and able to share a prefix. The kernel gains one indirection through the block table when reading K/V. See [PagedAttention](../advanced/attention.md#pagedattention).

??? note "39. How does decode attention parallelize at very small batch and very long sequence?"
    Flash-Decoding: cut the KV sequence into segments (split-K), one block per segment computing a partial output and its (m, ℓ), merged at the end by online softmax's rule. Under GQA, let one block handle several query heads that share a KV head so the KV is read once.

??? note "40. What are GQA and MLA? How do they affect the kernel?"
    GQA: several query heads share one set of KV heads, shrinking the KV cache by the head ratio; the kernel should handle the query heads that share a KV head in the same block. MLA (DeepSeek): the KV is compressed into a low-rank latent cache and the projections are absorbed into Q and the output by matrix absorption at compute time, so decode's attention becomes "many query heads sharing one very wide KV", closer to compute-bound and needing a dedicated kernel (FlashMLA, say).

??? note "41. What quantization schemes are common? When do you use which?"
    weight-only (W8A16, W4A16): cuts the bytes read directly when small-batch decode is bandwidth-bound, dequantizing in registers; W8A8 (INT8/FP8): makes the computation faster too when large batches and prefill are compute-bound; FP4 (Blackwell). INT4 is usually quantized by group (G=128) with 8 weights packed into one 32-bit word. See [quantization](../advanced/quantization.md).

??? note "42. Why does W4A16's speedup get worse at large batch?"
    As the batch grows the weights are reused and the GEMM gradually becomes compute-bound; W4A16 still computes on the FP16 Tensor Cores and adds dequantization work on top, so the gain falls and it can even become slower, which is when W8A8 is the right choice.

??? note "43. Which kernels and optimizations matter in an inference engine (vLLM/SGLang)?"
    Attention (paged prefill/decode, the FlashAttention/FlashInfer back ends), a fused RMSNorm/RoPE/activation, a quantized GEMM (Marlin, FP8), a fused MoE, sampling, a custom all-reduce; and at the system level continuous batching, chunked prefill, prefix caching (RadixAttention), CUDA Graphs, speculative decoding and prefill-decode disaggregation.

## Concurrency and multiple GPUs {#并发与多卡}

??? note "44. What are CUDA streams for? What is the trap with the default stream?"
    A stream is a queue of operations executed in order, and different streams can run concurrently, which is used to overlap copies with computation and to run small tasks in parallel. The legacy default stream implicitly synchronizes with other blocking streams and destroys the concurrency; `cudaStreamNonBlocking` or `--default-stream per-thread` avoids it. See [streams](../tools/streams.md).

??? note "45. How do CUDA Graphs work, and what are their limits?"
    Record a series of operations into a graph and submit the whole graph with one call later, cutting the launch overhead sharply. The limits: the arguments, shapes and memory addresses are fixed; nothing may synchronize during capture; a changing shape needs several graphs (inference engines capture one per batch size).

??? note "46. How does a ring all-reduce work and how much does it communicate?"
    reduce-scatter + all-gather, p-1 steps each, with each card sending about 2(p-1)/p × S bytes, nearly independent of the card count, and all the links working at once; but there are many steps, so small messages are latency-bound. NCCL chooses among ring, tree, NVLS and others. busbw = algbw × 2(p-1)/p. See [NCCL](../tools/multi-gpu.md#ring-all-reduce).

??? note "47. What communication do TP, PP and EP each need?"
    TP: 1-2 all-reduces per layer (or an all-gather + a reduce-scatter); PP: point-to-point activations between adjacent stages; EP: two all-to-alls per MoE layer. NVLink within a machine suits TP, and PP/EP/DP are usual across machines.

??? note "48. How do you overlap communication with computation?"
    In training, bucket the gradients and start a bucket's all-reduce as soon as it is complete; pipeline a GEMM and the communication after it in chunks; fuse communication with computation in one kernel; and for MoE use a dedicated all-to-all library (DeepEP) overlapped with the computation.

## Open-ended questions {#开放题}

??? note "49. An inference service for a new model has too high a TPOT. How do you diagnose and optimize it?"
    Measure first: nsys shows what one decode step's time consists of (which kernels, whether there are gaps, the CPU cost). The common problems and what to do: many gaps between kernels → CUDA Graphs, fusion; the GEMM/GEMV dominating → check the bandwidth utilization, consider quantization, grow the batch; attention dominating → use an efficient paged decode kernel, split-K, check whether GQA is reading the KV repeatedly; communication dominating (TP) → a custom all-reduce, a lower TP degree; scheduling overhead → overlap the CPU with the GPU. Verify every step with numbers.

??? note "50. If you had to write a fused MoE kernel, how would you design it?"
    After routing, count each expert's tokens and prefix-sum them to get the group offsets (aligned to the GEMM's tile size); arrange the tokens by expert (or only produce the sorted indices, avoiding an actual data movement); do all the experts' first linear layer in one grouped GEMM with the activation function (SwiGLU) fused; the same for the second linear layer; and finally weight by the routing weights and restore the order (which can fuse into the epilogue or be its own kernel). Watch out for: load imbalance, the GEMM efficiency of a small expert (a very small M), quantized weights. vLLM's fused_moe (Triton) and SGLang's and DeepEP's implementations are worth reading.

??? note "51. Your kernel is 2x faster than PyTorch's. How do you prove the result is right and the speed is real?"
    Correctness: compare against a reference implementation, covering the boundary shapes (unaligned, tiny, huge), every dtype and the special values (inf, nan, all zeros), with a sensible tolerance for floating-point error; and check for out-of-bounds accesses and races with compute-sanitizer. Performance: warm up, measure several times and take the median, with CUDA events or Nsight; compare against the theoretical ceiling (the roofline); test on several shapes and GPUs; and say what the baseline is (which PyTorch implementation, fused or not).

## Advice on answering {#答题建议}

- **The conclusion first, then the reason, then the numbers**: "this is bandwidth-bound, because the arithmetic intensity is only 0.25; I measured 85% bandwidth utilization, which is close to the ceiling".
- When writing code, **talk as you write**: why the threads are assigned this way, why a synchronization is needed here, how the boundary is handled. When you finish, volunteer what could be optimized next.
- For a question you cannot answer, say **how you would find out**: which tool and which metric, which beats "I don't know" by a long way.
- Prepare 2-3 optimization cases of your own that you can take through the whole "analysis → hypothesis → verification → optimization → result".
