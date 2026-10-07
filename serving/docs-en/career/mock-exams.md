# Mock interview sets: four sets arranged to the real structure

<p class="lead">The <a href="../mock-interview/">mock interview</a> chapter gave the structure, scoring rubric and retrospective template; this chapter arranges four sets of questions to that same structure: the first three lean toward the inference framework, inference optimization and inference platform roles respectively, and the fourth toward large-scale MoE and RL inference. Each set is 60 minutes: 20 minutes of coding, 15 of fundamentals follow-ups, 15 of system design and 10 of project deep dive. Every follow-up comes with key points and the corresponding chapters, so say your answer out loud and record it first, then expand and compare.</p>

!!! tip "How to use this"
    Have a colleague or study partner ask the questions below, strictly timed; follow up level by level until you can't answer, because that spot is what you need to go back and review. Do each set once, a week apart; after each, score with the rubric and write a retrospective. You can also hand this page's questions and rubric to a large model and have it play the interviewer (the prompt is in the mock interview chapter).

## Set one: the inference framework direction {#第一套推理框架方向}

**Coding (20 minutes)**: hand-write the core loop of a continuous batching scheduler: maintain a waiting queue and a running queue, in each step schedule decode first within the token budget and then chunked prefill with what remains, preempting the most recently arrived requests when KV runs short. Practice problem: [the continuous batching scheduler](root://practice/#/p/sv-scheduler). Alternative: [an LRU cache](root://practice/#/p/py-lru-cache).

**Fundamentals follow-ups (15 minutes)**

**1. Why is a paged KV Cache needed?** → How do you choose the block size? → When two requests share a prefix, how do the block table and reference counts change?

??? success "Key points"
    Contiguous allocation must reserve for the maximum length, and internal plus external fragmentation makes actual utilization very low; with paging, fixed-size blocks are allocated on demand and a block table maps logical positions to physical blocks, leaving only part of the last block as fragmentation. Larger blocks mean less metadata and more contiguous kernel access, but more internal fragmentation and coarser prefix sharing; in transfers, blocks that are too small get throttled by the NIC's message rate ([RDMA](../comm/rdma.md#请求数与消息速率)). Shared prefixes: both requests' block tables point at the same physical blocks and reference counts increase by one, with copy-on-write when a request needs to write into a shared block. See [the paged KV Cache](../engine/paged-kv.md) and [prefix caching](../engine/prefix-cache.md).

**2. What problem does chunked prefill solve?** → How large should the token budget be? → How does it relate to PD disaggregation?

??? success "Key points"
    A long prompt's prefill occupies a whole step and makes decoding requests wait, spiking TPOT; chunked prefill cuts it into chunks mixed with decode in each step, capping each step's total tokens. The larger the budget, the faster prefill (lower TTFT) but the slower each step (higher ITL), so choose by load testing against the SLO, commonly 2K–8K. PD disaggregation is the more thorough fix: prefill and decode on different instances with no interference, at the cost of KV transfer and the ratio problem; when interference is mild, chunked prefill is enough. See [the scheduler](../engine/scheduler.md) and [global scheduling](../frontier/disagg-sched.md#xpyd-配比跟着负载变).

**3. Why does decode use CUDA Graphs?** → The batch size changes every step; how is that handled? → What cannot be captured in a graph?

??? success "Key points"
    Each decode step has hundreds of small kernels, and the CPU's cost of submitting them one by one can match the GPU's compute time; a CUDA Graph captures the whole step and submits it once. Batch sizes are captured in buckets (1, 2, 4, 8...) and padded up to the nearest bucket at runtime; addresses in a graph are fixed, so inputs must be copied into preallocated buffers. What can't be captured: operations depending on CPU synchronization (such as an all-to-all that must copy counts back to the CPU), dynamic shapes, and operations that change memory allocation, which is also why DeepEP's low-latency mode uses fixed slots. See [CUDA Graphs and torch.compile](../engine/graphs-compile.md).

**System design (15 minutes)**: [global scheduling and a cache-aware inference gateway](design-answers-1.md#3-全局调度与缓存感知的推理网关).

**Project deep dive (10 minutes)**: present your inference engine (hand-written mini-sglang, or project A): which modules does a request pass through from HTTP to the last token? What was the most effective performance optimization you made, and what were the numbers? Any failed attempts?

## Set two: the inference optimization direction {#第二套推理优化方向}

**Coding (20 minutes)**: write an online softmax kernel in CUDA or Triton (one thread block per row, maintaining the maximum and the sum in one pass), and explain why it is the foundation of FlashAttention. Practice problem: [online softmax](root://practice/#/p/cu-online-softmax). Alternatives: [tiled GEMM](root://practice/#/p/cu-gemm-tiled), [the FlashAttention forward pass](root://practice/#/p/cu-flash-attn).

**Fundamentals follow-ups (15 minutes)**

**1. Why is decode memory-bound?** → At what batch size does it become compute-bound? → Why is MLA's decode attention different?

??? success "Key points"
    In decode each request has one token per step, so linear layers are matrix × vector, doing 2 operations per weight read; at batch $b$ the arithmetic intensity is about $b$ (bf16 weights), and H100's ridge point is about 295, so the batch must reach several hundred to approach compute-bound, while KV reads grow linearly with batch so attention stays memory-bound throughout. After MLA absorption, 128 heads share one 576-dimensional latent, about 240 operations per byte, close to compute-bound, which calls for a dedicated kernel (FlashMLA). See [MLA inference](../moe/mla.md#decode-注意力变成了计算问题).

**2. Why does DeepSeek-V3's FP8 use block scaling?** → At which step of the GEMM are the scales multiplied in? → What is special about MoE's grouped GEMM?

??? success "Key points"
    E4M3 has only 3 mantissa bits, and extreme outliers make per-tensor scaling squeeze normal values into subnormals; with 1×128 blocks for activations and 128×128 for weights, an outlier affects only its own group. The GEMM does one FP8 matrix multiply per 128 elements along K, and promoting the partial sum to fp32 multiplies in both scales, which also fixes Tensor Cores' accumulation precision of only about 14 bits. MoE: each expert gets a different number of tokens, so either a contiguous layout (padded to the block size, for prefill) or a masked fixed layout (pairing with DeepEP's low-latency mode, capturable in CUDA Graphs, for decode); in decode each expert needs about 300 tokens to escape the limit of weight reads. See [FP8 and grouped GEMM](../moe/fp8-gemm.md).

**3. Why does vLLM implement its own all-reduce?** → What is the difference between one-shot and two-shot? → How does this change on new hardware?

??? success "Key points"
    Decode's TP all-reduce is only a few hundred KB, NCCL's ring takes $2(n-1)$ steps, and the time is almost all fixed overhead; the custom implementation reads peers' memory directly through CUDA IPC. one-shot has each GPU read all peers' full data with one synchronization, suited to small messages; two-shot does a reduce-scatter then an all-gather, with two synchronizations and less traffic, suited to medium messages; large messages go to NCCL. H100's NVSwitch supports NVLS (reduction inside the switch), halving traffic and using no SMs. See [collective communication](../comm/nccl.md).

**System design (15 minutes)**: [a million-token long-context service](design-answers-2.md#6-百万-token-长上下文服务).

**Project deep dive (10 minutes)**: present the fastest kernel you have written (project C): what share of peak did it reach? Where did Nsight Compute show the remaining gap? Given another week, what would you optimize next?

## Set three: the inference platform direction {#第三套推理平台方向}

**Coding (20 minutes)**: write a dynamic batcher with asyncio: after a request arrives, wait at most $t$ milliseconds or until $b$ requests accumulate before processing them together, return each result to its own caller, and handle timeouts and exceptions correctly. Practice problem: [the dynamic batcher](root://practice/#/p/py-dynamic-batcher). Alternative: [routing by predicted TTFT with early rejection](root://practice/#/p/sv-ttft-router).

**Fundamentals follow-ups (15 minutes)**

**1. When should you do PD disaggregation?** → How long does the KV transfer take? → How do you set the ratio of prefill to decode instances?

??? success "Key points"
    It's worth it only when long prompts' prefill badly disturbs decode's TPOT, or when the two phases want different parallelism and hardware; at small scale with short prompts, chunked prefill is enough. The transfer volume = prompt length × KV per token (a 70B GQA model in FP8 is about 160 KB/token, so a 4K prompt is about 0.65 GB, a dozen-odd milliseconds on a 400G NIC), and it can be pushed layer by layer overlapping with compute. The ratio follows "the token compute prefill needs / the output throughput decode needs", and when the load's shape changes the bottleneck jumps between the two sides, so a planner must adjust it by the minute. See [PD disaggregation](../distributed/pd-disagg.md) and [global scheduling](../frontier/disagg-sched.md).

**2. What problems does cache-aware routing bring?** → How do you weigh cache and load together? → How does the router know what an instance has cached?

??? success "Key points"
    Looking only at the cache piles requests of the same kind onto one instance and creates hotspots; unify the two with "predicted TTFT = queueing time + prefill time for the part that missed", and reject early by prediction under overload (in simulation, attainment rises from 86% to 98% under high load). The index is maintained as a global prefix tree from KV events instances publish (stores, evictions), and it is approximate, so a routing mistake only costs extra compute, never wrong results. See [global scheduling](../frontier/disagg-sched.md#路由缓存与排队的权衡).

**3. What metrics does autoscaling use?** → How long does a cold start take, and how do you speed it up? → What must you watch when scaling down?

??? success "Key points"
    Look at queue length, KV usage and the trend of TTFT, not GPU utilization (naturally modest in decode); scale ahead by the daily curve and absorb bursts with hot standby (traffic keeps rising during a cold start, so size the standby from Little's law and the growth rate). A cold start = pulling weights (hundreds of seconds from object storage, twenty seconds from local NVMe, under a second pulling from a running instance over RDMA) + initialization + CUDA Graph capture and compilation (cacheable). Scaling down must wait for requests to finish and avoid flapping, and should go slower when long requests are common. See [system design reference answers (part two)](design-answers-2.md#9-推理可观测性与自动扩缩容).

**System design (15 minutes)**: [a KV-Cache-centric multi-tier cache pool](design-answers-1.md#2-以-kv-cache-为中心的多级缓存池).

**Project deep dive (10 minutes)**: present your inference gateway (project B): how much did cache-aware routing improve over round-robin in load tests? What happened when an instance failed, and how long did recovery take? What was the hardest trade-off in the design document?

## Set four: the large-scale MoE and RL inference direction {#第四套大规模-moe-与-rl-推理方向}

**Coding (20 minutes)**: implement expert-parallel dispatch and combine: each rank sorts tokens by target expert, computes how many go to each rank, the experts compute after the all-to-all, and the results are merged back into the original order by gate weights. Practice problem: [expert-parallel all-to-all dispatch and combine](root://practice/#/p/sv-ep-dispatch). Alternatives: [the ledger of EP communication](root://practice/#/p/sv-deepep-layout), [MLA's absorb path](root://practice/#/p/sv-mla-absorb).

**Fundamentals follow-ups (15 minutes)**

**1. Why does MLA's decode take the "absorb" path and prefill the "expand" path?** → With MLA's KV stored in FP8, what becomes decode attention's bottleneck? → What's the problem with deploying an MLA model with tensor parallelism?

??? success "Key points"
    Absorption folds $W_{UK}$ into the query and $W_{UV}$ into the output projection, so all heads share a 576-dimensional latent "key", making each (query, key) pair about 3.4× more expensive but avoiding expanding the cache; decode has few new tokens and a large cache, so expanding costs far more than the pricier dot products, and prefill is the reverse. With FP8 KV, each cached token reads only 576 bytes, for an arithmetic intensity of about 484 FLOP/byte, beyond H800's BF16 ridge point, so decode attention turns from memory-bound to compute-bound and the kernel's focus shifts to Tensor Core utilization (FlashMLA). Tensor parallelism cannot split the latent, since all heads share it, so every GPU stores a full copy and KV capacity does not grow with GPU count; hence MLA models usually use DP attention + expert parallelism. See [MLA inference](../moe/mla.md) and [expert parallelism and DP Attention](../distributed/expert-parallel.md).

**2. In decode with large-scale EP, where does a step's time go?** → When does two-batch overlap help? → How do you handle expert load imbalance?

??? success "Key points"
    Three parts per layer: attention (reading weights and KV), the expert GEMM (reading expert weights), and two all-to-alls (FP8 for dispatch, BF16 for combine), with communication in decode often as long as compute or longer. Two-batch overlap splits the batch in two to alternate compute and communication, 1.45–1.65× faster when the per-GPU batch is 64–128 and compute matches communication; with a very small batch, splitting means re-reading the weights and it barely helps. Load: EPLB replicates hot experts and re-places them from measured expert loads (hierarchical for prefill, global for decode); DP attention assigns requests by total KV, since the slowest rank sets the whole step's speed. See [large-scale EP deployment](../moe/ep-deploy.md).

**3. What special requirements does RL training place on an inference engine?** → What bias does staleness bring in fully asynchronous RL? → Why do the training and inference sides of an MoE model "take different experts"?

??? success "Key points"
    Four things: the long tail (partial rollout, asynchronous RL, over-sampling), inconsistent probabilities (importance correction, batch-invariant kernels), memory switching (sleep / wake), and weight sync (CUDA IPC when co-located; point-to-point by the partition mapping with pipelined relays between instances when disaggregated, finishing within seconds independent of the instance count). Full asynchrony's staleness concentrates on long answers, and with a cap the dropped ones are long answers too, so the model may lean toward short answers, calling for importance weighting, length-balanced batching and per-token logprob records. MoE: tiny numerical differences between the two sides can make top-k routing pick different experts, turning into an entirely different computation path; routing replay records the experts chosen per layer at inference, and the training side uses them directly. See [inference in RL training](../topics/rl-rollout.md) and [RL inference systems](../frontier/rl-async.md).

**System design (15 minutes)**: [an online inference service for a 600B-class MoE model](design-answers-1.md#1-600b-级-moe-模型的在线推理服务). Alternative: [an RL rollout system](design-answers-1.md#5-rl-rollout-系统).

**Project deep dive (10 minutes)**: present your assignment onboarding a hybrid-architecture model ([assignment four](minisgl://wrap/assignment-hybrid/)) or the MoE part of your hand-written engine: how did you align accuracy, and where did the first mismatch appear, in which layer and at which position? How large is each request's state, and why does it take more memory for short requests? How must prefix caching and speculative decoding change?

## Scoring and retrospectives {#评分与复盘}

Use the rubric in the [mock interview](mock-interview.md#评分表) chapter, 1–5 points on each of four items (correctness, depth and numbers, structure of expression, trade-offs). After doing several sets, compare: whichever kind of question always stalls at the same level of follow-up is the chapter to reread, and write the question that stalled you onto your own flashcard.

## Summary {#小结}

- [x] The four sets lean toward the inference framework (scheduling, paged KV, CUDA Graphs), inference optimization (roofline, FP8, communication kernels), the inference platform (PD disaggregation, routing, scaling), and large-scale MoE and RL inference (MLA, EP, asynchronous RL).
- [x] Time each set to the real structure: 20 minutes of coding, 15 of follow-ups, 15 of system design, 10 on projects; follow up level by level to find the level you can't answer.
- [x] When reviewing, score with the rubric, write the questions that stalled you onto flashcards, and go back to the corresponding chapters.
