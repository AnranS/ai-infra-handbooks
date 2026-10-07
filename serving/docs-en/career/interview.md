# Inference role interview question bank

<p class="lead">This collects the most common questions in interviews for inference optimization, inference framework engineering and AI Infra roles, grouped by topic, each with key points and the chapters to read for more. Questions on model principles are in the LLM handbook's <a href="llm://synthesis/quiz/">self-test bank</a>, and CUDA and operator questions in the CUDA handbook's <a href="cuda://career/interview/">interview question bank</a>; here the focus is the inference system itself.</p>

!!! tip "How to use this"
    - First read only the question and answer out loud in 2–3 minutes, then expand the key points and compare;
    - The structure of an answer is usually: **what it is → why (which bottleneck it solves) → how it works → the costs and trade-offs → the numbers**. The last one separates candidates the most: give the magnitude wherever you can estimate;
    - Questions marked ★ are frequent, so know them well.

## I. Estimation and fundamentals {#一估算与基础}

**1. ★ Why is decode memory-bound? And prefill?**

??? success "Key points"
    In decode each request processes 1 token per step, doing about 2 floating-point operations per weight read (2 bytes per parameter), so the arithmetic intensity is about the batch size, far below H100's ridge point of about 295 FLOP/byte; prefill processes hundreds or thousands of tokens at once, with high intensity, and is compute-bound. So decode optimizes bytes read (batching, quantization, KV compression) and prefill optimizes compute (FP8, chunking, efficient attention). See [the whole journey of a token](llm://synthesis/token-journey/#decode-的时间花在哪里).

**2. ★ Estimate the TPOT floor of LLaMA-3-8B on an H100 at batch=1.**

??? success "Key points"
    The weights are 8B × 2 bytes = 16 GB, H100's bandwidth is 3.35 TB/s, so about 4.8 ms per token; adding KV (negligible at short contexts) and kernel overhead, about 6–7 ms in practice. INT4 weights take about 1/3.5 of the time. See [estimating parameters, compute and memory](llm://inference/estimation/).

**3. ★ How big is one token's KV Cache? And for LLaMA-3-70B?**

??? success "Key points"
    2 (K and V) × layers × KV heads × head_dim × bytes. LLaMA-3-70B: 2 × 80 × 8 × 128 × 2 = 320 KB/token, about 43 GB at 128K of context. DeepSeek-V3 with MLA is about 69 KB/token. See [a tour of mainstream model architectures](llm://synthesis/models/).

**4. Deploying Qwen2.5-7B on one 80 GB GPU with an average context of 3000, how many concurrent requests fit?**

??? success "Key points"
    The weights are about 15 GB, about 10 GB is reserved for activations and the runtime, leaving about 55 GB; at 56 KB of KV per token, each request takes about 168 MB, so about 340 concurrent (paged, block size 16). Reserving by maximum length gives an order of magnitude fewer. See [the exercises of the paged KV Cache](../engine/paged-kv.md#练习).

**5. What are TTFT, TPOT, ITL and goodput?**

??? success "Key points"
    TTFT: time to first token (queueing + prefill); TPOT: the average time per token after the first; ITL: the gap between adjacent tokens (for spotting stalls); goodput: requests per second that meet the SLO. Read latency at P99. See [load testing](../perf/benchmark.md#指标).

**6. Why can the same request give different results twice at temperature 0?**

??? success "Key points"
    Floating-point addition is not associative, so batch composition, tiling and kernel choice change the accumulation order, logits differ slightly, and the argmax flips for nearly tied tokens. When strict consistency is required, use batch-invariant kernels. See [which optimizations change the output](llm://synthesis/token-journey/#哪些优化会改变输出).

## II. The inference engine {#二推理引擎}

**7. ★ From the user sending a request to receiving the first token, what steps happen inside the inference engine?**

??? success "Key points"
    HTTP parsing → the chat template and tokenization → sending to the engine core (across processes) → a prefix cache lookup and entering the waiting queue → the scheduler allocating KV blocks and a token budget → building the batch metadata → the forward pass (possibly prefilling over several steps) → sampling → detokenization and stop-condition checks → streaming back. TTFT = queueing + prefill + CPU overhead. See [the life of a request](../engine/overview.md).

**8. ★ How is continuous batching implemented?**

??? success "Key points"
    Iteration-level scheduling: rebatch at every step, with finished requests leaving immediately and new ones joining immediately; a unified token budget lets chunked prefill and decode mix in one step; paged KV is allocated on demand, with preemption when it runs short. See [the scheduler](../engine/scheduler.md).

**9. ★ What problem does PagedAttention solve? What does it cost?**

??? success "Key points"
    The KV Cache is allocated in fixed-size blocks, eliminating reservation waste and external fragmentation (waste drops from 60%–80% to under 4%) and enabling block-level sharing (prefix caching, parallel sampling). The costs: attention kernels must address indirectly through the block table, there is a little internal fragmentation within blocks, and metadata like block tables and slot mappings must be maintained. See [the paged KV Cache](../engine/paged-kv.md).

**10. What does it mean that vLLM's V1 scheduler "has no prefill and decode phases"?**

??? success "Key points"
    Each request has only `num_computed_tokens` and `num_tokens`, and the scheduler lets the former catch up with the latter within the token budget. Prefill, chunked prefill, decode, recomputation after preemption, prefix cache hits and speculative drafts are all special cases of this rule. See [the scheduler](../engine/scheduler.md#统一的-token-预算).

**11. What are the effects of raising or lowering `max_num_batched_tokens`?**

??? success "Key points"
    Large: a long prompt is computed in one step, so TTFT is short and GPU utilization high, but decode in the same batch stalls (ITL spikes); small: decode is smooth but TTFT grows. Measured in this book: a budget of 8192 gives a maximum ITL of 252 ms, and 512 gives 16 ms. See [load testing](../perf/benchmark.md#容量规划).

**12. ★ How do vLLM and SGLang each handle running out of memory?**

??? success "Key points"
    vLLM: admit if allocation succeeds, and when short, preempt the requests at the end of the running queue (or the lowest priority), freeing their blocks and putting them at the front of the waiting queue to be recomputed later (V1 dropped swapping); SGLang: admit conservatively by estimating future demand with `new_token_ratio`, and when short during decode, `retract_decode` pulls some requests back and raises the ratio. See [the scheduler](../engine/scheduler.md#抢占vllm-与-sglang-的两种思路).

**13. Why does vLLM put EngineCore in its own process?**

??? success "Key points"
    Tokenization, detokenization, HTTP and JSON are CPU-intensive Python work, and competing for the GIL with the engine's main loop leaves the GPU idle; split into processes, they truly run in parallel. The cost is inter-process communication, hence ZMQ + msgpack, with only incremental scheduling results sent. See [the sampler and the API](../engine/sampler-api.md#为什么要多进程).

**14. ★ Why do CUDA Graphs speed up decode? What are the limits?**

??? success "Key points"
    One decode step launches thousands of kernels of a few microseconds each, and launch overhead often exceeds the GPU compute; a CUDA Graph captures the whole step and submits it once. The limits: shapes and addresses are fixed (capture in buckets by batch size, pad, use static buffers), no CPU synchronization or data-dependent branches inside the graph, and attention metadata must live in fixed buffers (hence the piecewise and full modes). See [CUDA Graphs and torch.compile](../engine/graphs-compile.md).

**15. What problem does asynchronous scheduling (or SGLang's overlap scheduler) solve?**

??? success "Key points"
    Scheduling and input preparation are CPU work, and serialized with GPU compute they leave the GPU idle; asynchronous scheduling prepares step N+1 while the GPU executes step N, bridging the sampled results with placeholder tokens or "future tokens" on the GPU. The cost is that result processing lags one step, and overlap is turned off when continuous prefill or structured output needs synchronization. See [the SGLang walkthrough](../source/sglang.md#主线二scheduler-的事件循环).

**16. Why do inference engines compute logits only for the last position?**

??? success "Key points"
    Generation needs only the last position's distribution, so the output layer (with its huge vocabulary) computes only the positions to be sampled, saving considerable compute and `[tokens, vocabulary]` memory during prefill. vLLM uses `logits_indices = query_start_loc[1:] - 1`. See [variable-length batching](../engine/batch-layout.md).

**17. Why isn't `torch.multinomial` used for sampling?**

??? success "Key points"
    It introduces CPU-GPU synchronization. vLLM uses an exponential race: `argmax(p / E)` with E exponentially distributed, equivalent to sampling by p, and each request can use its own generator for reproducible seeds. See [batched sampling](../engine/sampler-api.md#批量采样).

**18. How are stop strings handled with streaming output?**

??? success "Key points"
    After incremental detokenization produces new text, check whether it contains a stop string; to avoid having already emitted the first half of a stop string, hold back `max(len(stop)) - 1` characters at the end and emit them once confirmed. Split multi-byte characters must also be handled. See [stop conditions and incremental detokenization](../engine/sampler-api.md#停止条件与增量反分词).

## III. The KV Cache and prefix caching {#三kv-cache-与前缀缓存}

**19. ★ How do vLLM's and SGLang's prefix caches differ?**

??? success "Key points"
    vLLM: chained hashed blocks (each hash includes the parent block's hash), with the free queue doubling as an LRU, and blocks invalidated only when reused; SGLang: a radix tree at token granularity, with leaf LRU and reference locks protecting nodes in use, paired with cache-aware scheduling (LPM). Hit rates are similar; the difference is in engineering trade-offs. See [prefix caching](../engine/prefix-cache.md).

**20. Why is the last token recomputed even when the prompt hits the cache completely?**

??? success "Key points"
    A forward pass is needed to get the next token's logits; the cache holds only KV, not logits. vLLM hits at most `num_tokens - 1` tokens.

**21. What are the risks of sharing a prefix cache across tenants?**

??? success "Key points"
    A timing side channel: judging whether some content is in the cache from TTFT. Include a tenant-specific `cache_salt` in the first block's hash so tenants never hit each other's. Multimodal requests must also include image hashes in block hashes, or different images get wrongly shared.

**22. What do you do when the KV Cache doesn't fit?**

??? success "Key points"
    Reduce KV (GQA/MLA, KV quantization, sliding windows, eviction) → manage it better (paging, prefix caching, preemption) → extend capacity (CPU/SSD/distributed tiered caching, where reading back is tens of times faster than recomputing) → spread it (TP, context parallelism). See [tiered KV caching](../distributed/kv-offload.md).

**23. What should you watch out for with FP8 KV Cache quantization?**

??? success "Key points"
    FP8's error is proportional to the values themselves, so compare it with **the useful signal**: if K is "a large constant bias + small variation" (models like Qwen2.5 whose K projection has a bias and no QK-Norm), the error is produced at the scale of the bias and drowns the part that varies with tokens, so quantizing only K raises perplexity by 12% while quantizing only V is nearly lossless; with QK-Norm (Qwen3), K varies around 0 and per-tensor FP8 is nearly lossless. Remedies: calibrated scales, per-head scaling, or per-channel quantization of K with zero points (KIVI's approach); evaluate long contexts before going live. See [quantization in deployment](../perf/quantization-deploy.md#kv-cache-量化误差要和信号比).

**24. Why does StreamingLLM keep the first few tokens?**

??? success "Key points"
    Attention sinks: the model puts a lot of attention on the very first tokens, and evicting them makes the softmax denominator collapse and distorts the distribution. Measured in this book: keeping only the last 256 tokens raises perplexity from 24 to 190, while keeping 4 extra tokens from the start brings it back to 29.6. See [long context](../topics/long-context.md).

## IV. Parallelism and distribution {#四并行与分布式}

**25. ★ Derive tensor parallelism by hand: how are the MLP and attention split? How many communications?**

??? success "Key points"
    MLP: gate/up split by columns, SiLU×up done locally, down split by rows, one all-reduce; attention: QKV split by heads with the KV Cache split along, o_proj split by rows, one all-reduce. Two per layer, with the volume = tokens × hidden. See [tensor parallelism](../distributed/tensor-parallel.md).

**26. What do you do when TP exceeds the KV head count?**

??? success "Key points"
    KV heads are replicated, with several GPUs storing the same KV head and KV memory wasted several-fold; also, the query head count must be divisible by TP (Qwen2.5-7B's 28 heads cannot do TP=8).

**27. ★ Where is TP's communication bottleneck during decode? How do you optimize it?**

??? success "Key points"
    The messages are small (batch × hidden) and dominated by launch latency (two per layer, so 160 for 80 layers). Optimizations: custom all-reduce (NVLink P2P, one-shot/two-shot), symmetric memory, fusing all-reduce with RMSNorm, lowering TP. In prefill it is bandwidth-dominated, and compute can overlap communication.

**28. ★ Why do DeepSeek-style models use DP Attention + EP rather than TP?**

??? success "Key points"
    MLA's latent KV cannot be split by heads, so under TP every GPU stores the full KV, a severe waste; DP Attention has each GPU handle different requests and store its own KV; the MoE part uses EP to pool the global batch onto the experts and raise their compute efficiency. The two are joined by all-to-all. See [expert parallelism](../distributed/expert-parallel.md).

**29. What do EP's two all-to-alls carry? How large is the traffic?**

??? success "Key points"
    dispatch sends tokens' hidden states to the GPUs holding their experts (usually in FP8), and combine sends the results back (BF16); the magnitude = tokens × top-k × hidden. Across machines it can reach hundreds of microseconds per layer, calling for limited cross-node routing, DeepEP and two-batch overlap.

**30. What does EPLB do?**

??? success "Key points"
    It measures expert load, replicates hot experts (redundant experts), then re-places them to balance load across GPUs. A layer's time is set by the slowest GPU, and in this book's simulation the busiest GPU drops from 2.26× the average load to 1.00×.

**31. What are DeepEP's normal and low-latency modes each for?**

??? success "Key points"
    Normal mode: prefill, large batches and high throughput, using both NVLink and RDMA; low-latency mode: decode, pure RDMA with low latency for small messages, capturable in CUDA Graphs.

**32. Can PP lower latency? How is PP used in inference?**

??? success "Key points"
    No: a token still passes through every stage in turn, plus the transfers; PP raises throughput and scales model size, with point-to-point communication only at stage boundaries, which suits going across machines. It needs several batches in the pipeline at once (vLLM's batch queue). See [pipeline parallelism](../distributed/pp-cp.md).

**33. How does ring attention merge blocked results? Why the zigzag split?**

??? success "Key points"
    Record each row's log-sum-exp and merge as $O = \sum_i O_i e^{\text{LSE}_i - \text{LSE}}$; under causal attention, contiguous splits give later GPUs far more compute, and a zigzag split equalizes the load (in this book: 136–904 becomes 520 everywhere).

**34. ★ The motivation, flow and costs of PD disaggregation?**

??? success "Key points"
    Motivation: prefill and decode interfere, their best configurations differ, and TTFT/TPOT are decoupled; the flow: decode pre-allocates blocks → prefill computes and pushes KV layer by layer (RDMA) → decode takes over; the costs: KV transfer (KV per token × length, overlappable with compute), layout conversion, the xPyD ratio shifting with load, and system complexity. See [PD disaggregation](../distributed/pd-disagg.md).

## V. Quantization {#五量化}

**35. ★ What do W4A16 and W8A8 each speed up? How do you choose?**

??? success "Key points"
    W4A16 cuts the bytes of weights read and speeds up memory-bound decode without speeding up compute, leaving saturated capacity nearly unchanged; W8A8 (FP8/INT8) also speeds up compute and doubles capacity. Choose W4A16 for low concurrency and latency sensitivity, FP8 for high throughput. This book's simulation: W4A16 cuts low-load TPOT from 5.2 to 1.9 ms with capacity going 22.5 → 24.6 req/s; FP8 reaches 46.5 req/s. See [quantization in deployment](../perf/quantization-deploy.md#对容量的影响).

**36. Why does FP8 use E4M3? How do you choose the scaling granularity?**

??? success "Key points"
    E4M3 has more precision, and its range is sufficient together with scales; E5M2 is used mostly for gradients in training. Granularity: per tensor < per channel/per token < block (128×128 for weights, 1×128 for activations), and this book measured perplexity losses of 4.5% / 1.2% / 0.3%.

**37. What is the difference between MXFP4 and NVFP4?**

??? success "Key points"
    Both use E2M1 values; MXFP4 has one power-of-2 scale (E8M0) per 32 numbers, while NVFP4 has one FP8 scale per 16 numbers plus a tensor-level scale, which is more accurate (this book's naive quantization: MXFP4 +27%, NVFP4 +7%). Blackwell supports them natively.

**38. Why are activations harder to quantize than weights? What does SmoothQuant do?**

??? success "Key points"
    Activations have huge outliers on fixed channels, and since they differ every time, scales can only be computed online; SmoothQuant uses a mathematically equivalent transformation to move the difficulty from the activations to the weights. See the LLM handbook's [quantization principles](llm://inference/quantization/#激活量化与离群值).

**39. Which layers are usually left unquantized?**

??? success "Key points"
    The embedding layer (a lookup, so quantization doesn't speed it up), lm_head (it directly affects the ranking of logits), the MoE router (a small error can change expert selection), and some sensitive first and last layers.

**40. What is the evaluation process before shipping quantization?**

??? success "Key points"
    A quick perplexity check → general benchmarks (lm-evaluation-harness: MMLU, GSM8K and others) → the business's own evaluation set; evaluate long contexts, code and math separately; compare against the baseline and prepare a rollback.

## VI. Speculative decoding {#六投机解码}

**41. ★ Why does speculative decoding speed things up? Why doesn't it change the output?**

??? success "Key points"
    Decode is memory-bound, so verifying k tokens is nearly as fast as generating 1; under greedy decoding the argmax is compared one by one and the output is unchanged; under sampling, rejection sampling (accept with min(1, p/q), and on rejection sample from norm(max(0, p−q))) leaves the distribution strictly unchanged. See the LLM handbook's [speculative decoding](llm://inference/serving/#投机解码).

**42. How are tree drafts verified?**

??? success "Key points"
    All nodes are lined up as one sequence, a tree attention mask lets each node see only the prefix, its ancestors and itself, positions are "prefix length + depth", and one forward pass suffices; from the root, follow the target model's predictions to the longest matching path, and the KV along it is reused directly. See [advanced speculative decoding](../topics/speculative.md).

**43. What are EAGLE and MTP?**

??? success "Key points"
    EAGLE: a small draft network of about one Transformer layer taking the target model's hidden state and the next token's embedding and predicting autoregressively; EAGLE-3 fuses features from several layers. MTP: multi-token prediction modules a model is trained with (DeepSeek-V3 and others), used as the draft at inference.

**44. ★ When does speculative decoding slow things down?**

??? success "Key points"
    With large batches, decode approaches compute-bound and verifying k+1 tokens costs several times more; with a low acceptance rate the work is wasted. This book's estimate: 2.8× at batch 1, 1.3× at batch 128, 0.9× at batch 256.

**45. What changes does speculative decoding require in the engine?**

??? success "Key points"
    The scheduler allocates budget and slots for draft tokens, and rejected ones don't advance num_computed; verification batches have several queries per request; rejection sampling runs in batches on the GPU; the draft network has its own KV and CUDA Graphs; and it interacts with structured output, prefix caching and PD disaggregation.

## VII. Performance analysis and tuning {#七性能分析与调优}

**46. ★ TPOT is twice as slow as expected; how do you investigate?**

??? success "Key points"
    Compare with the roofline floor → look at server-side metrics (batch, KV usage, preemption, queueing) → look at the timeline in nsys: gaps between kernels mean CPU overhead (CUDA Graphs, asynchronous scheduling, removing syncs), while a constantly busy GPU points at the kernels themselves (ncu, bandwidth utilization, kernel choice) → check communication (TP's all-reduce). See [profiling](../perf/profiling.md).

**47. Why does `.item()` lower throughput?**

??? success "Key points"
    It triggers CPU-GPU synchronization: the CPU waits for the GPU to finish, then the GPU waits for the CPU to prepare the next step, and the two idle in turn.

**48. ★ How do you assess how many QPS a service can take?**

??? success "Key points"
    Set the SLO → build a realistic workload → load test open-loop (at a fixed arrival rate) and plot the latency-throughput curve → find the maximum rate meeting the SLO (the point of maximum goodput) → compute the GPU count from target traffic plus redundancy. Note that goodput has often already collapsed when throughput peaks, and closed-loop tests mask overload. See [load testing](../perf/benchmark.md).

**49. How do you lower TTFT?**

??? success "Key points"
    Queueing: scale up, scheduling policy, rate limiting; prefill: prefix caching, FP8, chunking, PD disaggregation (dedicated prefill instances), context parallelism; CPU overhead: tokenization in its own process, several API servers. First confirm which part is long.

**50. How do you lower TPOT?**

??? success "Key points"
    Read fewer bytes per step (weight quantization, KV quantization, MLA/GQA), reduce interference (the chunked prefill budget, PD disaggregation), reduce CPU overhead (CUDA Graphs, asynchronous scheduling), speculative decoding (at low concurrency), TP (single-request latency).

## VIII. Special topics {#八专题}

**51. How is structured output implemented? Where is the performance bottleneck?**

??? success "Key points"
    Grammar → automaton, computing a bitmask of allowed tokens each step and sampling after masking. The bottleneck is computing masks over a large vocabulary: xgrammar precomputes and caches context-independent tokens, and the engine overlaps mask computation with the GPU forward pass; jump-forward skips determined text (minding tokenization consistency). See [structured output](../topics/structured-output.md).

**52. How is tool calling implemented in the engine?**

??? success "Key points"
    The chat template renders tools → the model outputs a call in its trained format (such as `<tool_call>`) → a tool parser extracts it into OpenAI's `tool_calls` (parsed incrementally when streaming) → constrained decoding with the argument Schema when needed. Reasoning models also need a reasoning parser to separate the thinking content.

**53. What is special about multimodal inference?**

??? success "Key points"
    The image token count grows linearly with resolution (Qwen3.5, Qwen3-VL: pixels / 1024; Qwen2.5-VL: pixels / 784); vision encoding happens only in prefill and takes a third to a half of it, reused via an encoder cache; M-RoPE gives three-dimensional positions; the prefix cache must include image hashes in its keys; EPD disaggregation. See [multimodal inference](../topics/multimodal.md).

**54. ★ What special requirements does RL training place on an inference engine?**

??? success "Key points"
    Long-tailed rollout (partial rollout, asynchronous RL, over-sampling); inconsistent probabilities between training and inference (differences of tens of percent in BF16, calling for returning sampling logprobs, importance sampling correction and batch-invariant kernels); memory switching (sleep/wake); efficient weight sync (IPC, NCCL, re-sharding). See [inference in RL training](../topics/rl-rollout.md).

**55. How do you support a 1M context?**

??? success "Key points"
    Model architecture (GQA/MLA, mixed local-global, linear attention, native sparse attention such as DSA) + the system (chunked prefill, context parallelism, KV offloading, sparse attention kernels) + lossy approximation (KV eviction).

## IX. Projects and open-ended questions {#九项目与开放题}

**56. Describe a performance optimization you have done.**

??? success "Key points"
    Use the structure "symptom → hypothesis → tool → evidence → change → result", giving before-and-after numbers and a comparison with the theoretical floor. Without a real project, use this book's mini engine: for example, "profiling found per-request attention taking 82% at batch 64, and switching to a batched kernel...". See [the portfolio](projects.md).

**57. If you designed an inference service platform from scratch, how would you do it?**

??? success "Key points"
    See [system design problems](system-design.md): requirements and SLOs → capacity estimates → single-instance configuration (engine, parallelism, quantization) → multiple instances (routing, cache-aware, PD disaggregation) → elasticity and reliability → monitoring and cost.

**58. Which are you more familiar with, vLLM or SGLang? What important changes have they made recently?**

??? success "Key points"
    Be able to name the specific modules you have read (the scheduler, KV management, some attention backend) and recent directions: both are working on CPU/GPU overlap, large-scale EP and PD disaggregation, cache management for hybrid architectures (linear attention, sparse attention), and RL support; vLLM has Model Runner V2 and the Rust frontend, while SGLang has a unified radix tree cache and staged CUDA Graph configuration. Go by the source code, showing you follow the latest.

**59. Have you contributed code to an open-source inference framework?**

??? success "Key points"
    If so, explain the problem, the approach, the discussion in review and the final result; if not, you can talk about problems found while reading source code or issues you reproduced. Note that projects have explicit rules on AI-assisted contributions (vLLM, for example, requires PRs to state how AI was used and rejects PRs with no substance), so don't submit trivial changes to "farm" contributions.

**60. What are the most important directions in inference systems over the next year or two?**

??? success "Key points"
    An open question; substance is what counts: EP and PD disaggregation for large-scale MoE, KV-centric architectures (distributed KV storage, reuse across instances), long context (sparse attention, hybrid models with linear attention), low precision (FP4 compute), agent workloads (multi-turn, tool calling, prefix reuse), integrating RL and inference, and inference cost optimization. Going deep on one or two beats listing them all.

## X. MoE and large-scale deployment {#十moe-与大规模部署}

**61. ★ What is MLA's "matrix absorption"? Why do prefill and decode take different paths?**

??? success "Key points"
    MLA caches only a 512-dimensional latent $c$ per token plus a 64-dimensional RoPE key shared by all heads. **Expand**: use $W_{UK}$ and $W_{UV}$ to expand the cache into each head's K and V and do ordinary multi-head attention, with each (query, key) pair dotted over 192 dimensions; **absorb**: by associativity, fold $W_{UK}$ into the query and $W_{UV}$ into the output projection and compute directly in latent space, equivalent to an MQA where 128 heads share the same 576-dimensional "key", making each pair about 3.4× more expensive but avoiding expanding the cache. Decode has few new tokens and many cached ones, so expanding (one up-projection multiply per cached token) costs far more than the pricier dot products, hence absorb; prefill has many new tokens, so expansion is amortized over many queries, hence expand. RoPE sits between the two matrices and cannot be absorbed, which is why 64 shared dimensions are split out. See [MLA inference](../moe/mla.md).

**62. In a fine-grained FP8 quantized GEMM, where are the scales multiplied in? Why promote partial sums to fp32 every 128 elements?**

??? success "Key points"
    Activations have one scale per 128 channels per token, and weights one per 128×128. The GEMM does one FP8 matrix multiply per 128 elements along K (Tensor Core), and the partial sum is multiplied by both scales in fp32 registers on CUDA Cores before accumulating. Two benefits: an outlier affects only its own group; and the H800's FP8 Tensor Cores have limited internal accumulation precision (about 14 bits), so accumulating all of K = 7168 at once loses precision, which promoting every 128 fixes. The scale multiplication is nearly free, since the partial sum has to be moved out at that step anyway. See [fine-grained FP8 quantization and DeepGEMM](../moe/fp8-gemm.md).

**63. ★ What is two-batch overlap (TBO)? When does it gain the most?**

??? success "Key points"
    In decode, split each rank's batch into two micro-batches: while one computes attention and experts, the other's dispatch / combine is on the network. The cost is that after splitting, each micro-batch re-reads the weights (decode is limited by weight reads). Estimated for DeepSeek-V3 on H800: at a per-GPU batch of 32 it is only 7% faster (compute is almost all weight reads), at 64–128 compute and communication take about as long and it is 1.45–1.65× faster; with larger batches communication dominates and the gain falls back, so traffic must be reduced. DeepEP's low-latency hook keeps communication off the SMs, a prerequisite for overlap. See [large-scale EP deployment](../moe/ep-deploy.md#双-batch-重叠).

**64. What do node-limited routing and DeepEP's "per-node deduplication" each save?**

??? success "Key points"
    Node-limited routing sends each token to at most 4 nodes (first pick nodes by the sum of the top 2 expert scores on each, then pick 8 experts within them); DeepEP's high-throughput mode sends only one copy to each remote node (over RDMA to the same-rail GPU on the target node, which forwards over NVLink). Together, cross-node traffic drops from 6.77 copies per token to 3.36, exactly half; forwarding within the node has 9× a NIC's NVLink bandwidth and is not a bottleneck. See [NVSHMEM and DeepEP](../comm/nvshmem-deepep.md#高吞吐模式按节点去重两跳转发).

**65. Does MTP speculative decoding always gain in decode with large-scale EP?**

??? success "Key points"
    Not necessarily. MTP is a one-layer draft trained with the model, with an acceptance rate of 85%–90%. When verifying $k+1$ tokens, the **per-request** costs (attention weights, reading the latent KV) are amortized while the **per-token** costs (compute, EP's all-to-all) multiply by $k+1$. So gains are large with small batches and long contexts; with large batches and tight communication it may even slow down. See [MTP and sparse attention](../moe/mtp-sparse.md#在大规模-ep-里验证-token-贵在哪).

## XI. The new generation of models and architectures {#十一新一代模型与架构}

**66. ★ What changes must an inference engine make for hybrid architectures like Qwen3.5 and Qwen3-Next?**

??? success "Key points"
    **Compute**: decode recurs token by token, prefill uses the chunked algorithm (matrix multiplies within a chunk, state passed between chunks), and chunked prefill comes for free. **Memory**: besides the paged KV pool, add a per-request state pool, with states usually in fp32; Qwen3.5-0.8B has 18.8 MiB of state per request and 12 KiB of KV per token, so under 540 tokens it takes more memory than full attention, and capacity must be planned by the length distribution. **Mechanisms redone**: prefix caching can hit only where state was saved, speculative decoding must be able to roll back state, and PD disaggregation must transfer state. See [linear attention and hybrid architectures](../frontier/linear-attn.md).

**67. Why is prefix caching hard for hybrid architectures? How is it done now?**

??? success "Key points"
    KV can be shared by block because block $i$ depends only on the first $i$ blocks' tokens and each block is stored separately; a linear-attention state is **a summary of the entire prefix** that can neither be split nor concatenated, so reusing a prefix requires a state saved at exactly that position (a checkpoint, over ten MiB each). vLLM's `--mamba-cache-mode align` saves only where "a stretch of prefill just finished and lands on a block boundary"; SGLang hangs states on specific nodes of a hybrid radix tree (such as the end of a prompt). The cost is coarser hit granularity; multi-turn conversations fit nicely, since where the previous turn ended is the next turn's prefix. See [linear attention and hybrid architectures](../frontier/linear-attn.md#前缀缓存要重做).

**68. How does "compression + sparsity + sliding window" attention, as in DeepSeek-V4, affect the KV ledger and decode?**

??? success "Key points"
    Each layer keeps a recent stretch of raw KV (the sliding window), then compresses every 4 or 128 tokens into one entry, with C4 layers having an indexer pick the top-k entries. Decode looks at only hundreds to thousands of KV entries per layer, KV per token is about a tenth of V3.2's, and a 1M-context request is 4–5 GB; the ledger splits into "growing with context" and "fixed per request" (the sliding window, the compressor's state), and the latter makes short requests relatively more expensive. On the engine side, several kinds of cache must share a memory pool and CUDA Graphs must be captured in segments. See [the new generation of open models](../frontier/new-models.md).

**69. ★ On a new model's release day, how do you get it "running correctly" in the engine?**

??? success "Key points"
    First compare the config, weight names and reference implementation to list the differences (the $1 + w$ normalization, the embedding scale, per-layer RoPE bases, sliding windows, the activation function); reuse existing layers and fill in the differences; hook layer by layer in FP32 against the reference and infer the cause from "which layer and which position starts going wrong" (position 0 fine means RoPE, starting at 512 means the sliding window); end to end, look at KL rather than only top-1, with normal noise at the same precision as the baseline; cover different lengths, both prefill and decode paths, batch composition and tensor parallelism in tests; finally run evaluations, checking the chat template and stop tokens first. See [onboarding new models and aligning accuracy](../ops/new-model.md).

**70. Why does a trillion-parameter MoE use INT4 QAT rather than plain PTQ?**

??? success "Key points"
    Start with the accounting: FP8 needs two 8-GPU machines while 4 bits needs one, cross-machine EP becomes in-machine EP, and decode's latency floor halves. Then accuracy: reasoning models' outputs are long, so PTQ's small errors are amplified along the reasoning chain (with 99% single-step accuracy, only about 6% of 256-step chains are all correct), hence quantization-aware training in post-training, with the straight-through estimator (STE) letting gradients through rounding. Finally hardware: W4A16 / W4A8 dequantize in registers on Hopper, while Blackwell has native Tensor Cores for MXFP4 / NVFP4. See [low-bit inference for very large MoE](../frontier/low-bit.md).

## XII. Scheduling, caching and communication {#十二调度缓存与通信}

**71. ★ After PD disaggregation, how are requests routed? How is the prefill-to-decode ratio set?**

??? success "Key points"
    Looking only at cache hits creates hotspots, and looking only at load repeats prefill; use a unified cost such as "predicted TTFT = queueing time + compute time for the part that missed", and reject early by prediction under overload (in this book's simulation, SLO attainment rises from 86% to 98%). The cache index is maintained from KV events workers publish and is approximate. The ratio shifts with the load's shape, the bottleneck jumps between the two sides, and a planner must adjust it by the minute, counting the cost of switching roles (Mooncake's Conductor, Dynamo's Router and Planner). See [global scheduling in disaggregated architectures](../frontier/disagg-sched.md).

**72. What problems do KV transfer engines and distributed KV storage each solve?**

??? success "Key points"
    The transfer engine handles "how to move it": registering memory, exchanging metadata out of band, batched reads and writes, multiple NICs and topology awareness (Mooncake Transfer Engine, NIXL). Distributed KV storage handles "where it goes and how to find it": a prefix-chained hash as the global key, decoupling "hitting the cache" from "which machine the request is routed to"; the SSD tier (3FS and the like) provides capacity. With random routing half the prefill is recomputation, and a shared pool cuts it to just the new content; the premise is that reading back beats recomputing. See [KV transfer engines and distributed KV storage](../comm/kv-storage.md).

**73. ★ Why do inference frameworks write their own all-reduce?**

??? success "Key points"
    In decode, tensor parallelism's messages are only a few hundred KB, NCCL's ring takes $2(n-1)$ steps, and the time is almost all fixed overhead. The custom implementation reads peers' memory directly through CUDA IPC: one-shot for small messages (done in one step), two-shot for medium ones (reduce-scatter + all-gather), and large ones handed back to NCCL. It must also be capturable by CUDA Graphs (buffers pre-registered, addresses fixed). Hardware reduction in switches like NVLS gives the problem a new answer on new hardware. See [collective communication](../comm/nccl.md).

**74. Why does KV transfer use RDMA one-sided writes? What must you watch out for?**

??? success "Key points"
    RDMA's three keywords: kernel bypass, zero copy, one-sided operations (the peer's CPU doesn't participate). The flow: register memory to get an `rkey`, build a QP, exchange addresses and keys out of band, post a WRITE, poll the completion queue. In inference: the KV pool is registered once at startup; decode pre-allocates blocks, prefill pushes layer by layer, and the last write carries an immediate to notify the peer; small blocks must be merged, since with a page size of 1 token-by-token transfer gets throttled by the NIC's message rate; several NICs run in parallel by topology. See [the RDMA programming model](../comm/rdma.md).

**75. Why doesn't tensor parallelism cross machines? Where is cross-machine EP's bottleneck?**

??? success "Key points"
    Start with numbers: NVLink is about 450 GB/s per GPU one way and a NIC about 50 GB/s, a 9× gap. Then latency: decode's communication messages are only a few hundred KB, close to or below the "half-bandwidth point", with fixed costs taking half the time, hence custom all-reduce, CUDA Graphs and fusing communication with compute. Cross-machine EP must follow the rail topology: "send on the same rail first, then forward within the machine" (NCCL's PXN, DeepEP's two hops). See [GPU interconnects and networks](../comm/interconnect.md).

## XIII. Speculative decoding and determinism {#十三投机解码与确定性}

**76. ★ The higher the load, the worse the deal speculative decoding is; how do the new approaches handle this?**

??? success "Key points"
    Speculative decoding trades compute for steps: at low load compute sits idle, and at high load draft tokens compete with real tokens for compute, so the best draft length K falls with batch size (for an 8B model, K = 8 is 3–4× faster at batch 1, while K = 0 is best at batch 512). Three improvements: **block drafts** (one forward pass gives K drafts, then dependencies between positions and confidences are added); **load-aware verification** (dynamic K, adjusting steps by acceptance length, picking the tokens to verify by "survival probability" among all positions of all requests, with the budget from a cost model); **engineering cooperation** (PD disaggregation must transfer hidden states and drafts, K must be consistent under data parallelism, and stateful models must be able to roll back). See [new approaches to speculative decoding](../frontier/spec-next.md).

**77. ★ How do you make one request's result independent of the batch? What does it cost?**

??? success "Key points"
    Make each request's reduction order depend only on itself: no split-K or a fixed split in matrix multiplication; split attention's KV by a fixed length (SGLang's deterministic mode has FlashInfer split decode every 2048 tokens, with FA3 not splitting); one thread block per row for normalization and softmax; identical results across chunked prefill and prefix caching; NCCL fixed to the tree algorithm with a single channel; sampling randomness determined by the request's seed and position. The cost is less parallelism for small batches and long-context decode, and slower communication. Used for on-policy RL rollout (SGLang's `--rl-on-policy-target`), reproducible evaluations and regression tests. See [deterministic inference](../topics/deterministic.md).

## XIV. Production and the ecosystem {#十四生产与生态}

**78. ★ How is multi-LoRA serving implemented? Where is the bottleneck?**

??? success "Key points"
    The base part is computed for all requests together and the LoRA part per request: decode uses BGMV (each token gathers its own adapter's weights) and prefill uses SGMV (segmented matrix multiplies by adapter), with real kernels fusing gathering the weights with shrink / expand. On scheduling: adapters are tens of MB each, with popular ones resident in GPU memory and all of them cached on the host; adapters per batch are capped (`--max-loras`); routing is affine by adapter; and prefix cache keys must include the adapter id. A rank-16 LoRA's compute is only about 0.5% of the base, but decode is memory-bound, so the more adapters in a batch, the more LoRA weights must be read. See [serving multiple LoRAs](../ops/multi-lora.md).

**79. How do you estimate the speed limit of on-device inference? Why is GGUF's Q4_K more accurate than Q4_0?**

??? success "Key points"
    Single-user decode reads all the weights every step: the speed limit = effective bandwidth ÷ the weights' bytes. An 8B model at 4.5 bits is about 4.5 GB, and a phone's 60 GB/s at 70% effective gives about 9 tokens per second; longer contexts add KV on top. Q4_0 has one scale per 32 weights with symmetric quantization; Q4_K uses superblocks of 256 containing 8 sub-blocks that each have their own scale and minimum (asymmetric), with those two numbers quantized to 6 bits and restored by the superblock's fp16 scales, so two levels of scaling reach nearly a 5-bit format's accuracy at 4.5 bits. On-device also means the NPU doing prefill and the CPU / GPU doing decode, plus thermal throttling. See [on-device inference](../ops/edge.md).

**80. How do you choose between vLLM and SGLang? How do you run a fair comparison?**

??? success "Key points"
    Ask in order: model and hardware support → the key features the scenario needs → measured goodput on your own workload → operability → customizability. A fair comparison: the same model and precision, the same request distribution; tune each framework's parallelism, batch limits, chunk size and CUDA Graph buckets per its official advice (comparing defaults is unfair); compare goodput curves under the SLO rather than peak throughput; sample the outputs to confirm nothing is "faster but worse". Giving a concrete trade-off is more convincing than quoting a benchmark number. See [choosing an inference framework](../ops/frameworks.md).

**81. Launching a 70B model, how do you release without interrupting service?**

??? success "Key points"
    Deployment: multi-machine instances with LeaderWorkerSet plus gang scheduling and topology awareness; loading: a local NVMe cache, streaming parallel loading, pre-sharding and compilation caches, cutting cold starts from minutes to tens of seconds; health checks: enough time in the startup probe, real requests in the readiness probe, draining in-flight requests on exit; release: add before removing, never taking down more at once than the headroom, with model versions rolled out gradually and caches isolated; scaling: on queue length and KV usage rather than GPU utilization, fast up and slow down, with a minimum instance count above 0. See [production deployment and operations](../ops/deploy.md).

**82. Why did vLLM build a Rust frontend?**

??? success "Key points"
    Numbers first: the Python frontend spends a dozen to several dozen microseconds per output token (detokenization, building JSON, coroutine scheduling), several times more with logprobs, so one core handles ten to several tens of thousands of tokens per second, the same order of magnitude as an 8-GPU machine's throughput; the GIL makes long-prompt tokenization, template rendering and streaming output compete for one core, leaving only more processes as an option, and tail latency suffers too. vLLM's approach leaves the engine core alone and swaps only the frontend, relying on a language-independent boundary: ZMQ plus msgpack arrays encoded in field order (so new fields can only be appended and both sides must change together); the Rust frontend is multithreaded in one process, with a gRPC control plane for gateways and RL frameworks. See [vLLM's Rust frontend](../source/rust-frontend.md).

## XV. RL and inference {#十五rl-与推理}

**83. ★ How is "staleness" distributed in fully asynchronous RL? What bias does it bring?**

??? success "Key points"
    Full asynchrony keeps the inference pool fully loaded, with 3× the training steps of synchronous RL in the same time (this book's simulation: 34 steps in 2 hours versus 11), and average staleness under 1 version. But staleness is uneven: short answers lag only 0.4 versions on average, while answers above 8K lag over 2; with a staleness cap, the ones dropped are long answers too, so long reasoning chains are both staler and more likely to be dropped, and the model may lean toward short answers. Countermeasures: importance-weighted correction (a PPO objective decoupling the behavior policy from the proximal policy), length-balanced batching, and interruptible generation (recording the sampling logprob per token). See [RL inference systems](../frontier/rl-async.md).

**84. How do you sync a trillion-parameter model's weights to dozens of inference instances within seconds?**

??? success "Key points"
    Co-located deployments go over NVLink within the machine with CUDA IPC, in under a second. Disaggregated: first compute the mapping between the training side's and the inference side's partitions so each inference rank receives exactly its shard without gathering on one GPU; the sender uses several GPUs in parallel for about 2.5 seconds per instance; instances then relay in a chunked pipeline (forwarding each chunk to the next as it arrives), making the total time nearly independent of the instance count, about 2.6 seconds even for 64 instances. The inference side must support "pause → update → resume" and clear the prefix cache computed with old weights. See [RL inference systems](../frontier/rl-async.md#权重同步的时间账).

**85. Why does RL on an MoE model need "routing replay"?**

??? success "Key points"
    The inference and training sides' numbers for the same sequence differ slightly; for MoE, that difference can make top-k routing pick different experts, which is no longer numerical error but a different computation path, distorting both importance ratios and gradients. Routing replay: record the experts each token picked at each layer during rollout, and use them directly when the training side recomputes logprobs and backpropagates. The inference engine must be able to return the routing results (`--enable-return-routed-experts` in vLLM and SGLang), about 500 bytes per token for 61 layers with 8 ids each. See [RL inference systems](../frontier/rl-async.md#moe-的路由重放).

## Summary {#小结}

- [x] Estimation questions must be answerable mentally: weight/KV bytes against bandwidth, KV per token, concurrency.
- [x] Engine questions revolve around the request lifecycle, scheduling, KV management, CUDA Graphs, multiple processes and overlap.
- [x] Distribution questions revolve around communication patterns: TP's all-reduce, EP's all-to-all, PP's point-to-point, CP's LSE merging, PD's KV transfer.
- [x] For quantization and speculative decoding, always explain "what it speeds up and when it stops working", with numbers.
- [x] Groups X–XV are the newer frontier and production questions: large-scale MoE (MLA absorption, FP8 blocks, TBO, MTP), hybrid architectures and new models, KV storage and communication, determinism, multi-LoRA and on-device, asynchronous RL; answer them the same way, mechanism first, then costs and numbers.
