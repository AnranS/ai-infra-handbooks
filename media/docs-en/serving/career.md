# Interviews and a portfolio: making this book your own

<p class="lead">In inference interviews, image and video generation is going from a bonus to a second battlefield: the interviewer wants to know whether you can carry LLM inference's skills over to a model with no KV cache, that produces its result through dozens of forward passes, and whose attention sequence is a hundred thousand tokens. This chapter harvests the previous 19 chapters into three things: a table matching capabilities to chapters to how they are examined, the skeleton answers and key numbers for ten frequent questions, and several portfolio pieces you can build on one consumer card (or even a CPU) and put on a CV.</p>

## What interviews examine {#面试考什么}

Nearly every question about generative-model inference falls into four kinds, each with a requirement to bring numbers:

| What is examined | Typical phrasing | What the interviewer wants to hear | Where in this book |
| --- | --- | --- | --- |
| The arithmetic is clear | "How long does FLUX take for one 1024² image, and how much memory?" "Why is video slow?" | deriving FLOPs and time from the parameter count, the token count, the steps and the guidance, and matching the measurements | [the accounting](../perf/accounting.md), [memory](../perf/memory.md), [the video bottleneck](../video/bottleneck.md) |
| Knowing why a technique works | "Why can a cache skip?" "Why is quantization different from an LLM's?" "Why does video need sequence parallelism?" | each technique's preconditions, how much it saves, what it costs, and whether it stacks | [few steps](../perf/distillation.md), [caching](../perf/caching.md), [quantization](../perf/quantization.md), [parallelism](../perf/parallel.md) |
| Service design | "Design a text-to-image API" "How do you do several LoRAs?" | the latency-against-throughput trade-off, pooling, pipelining, previews and cancellation, switching weights | [scheduling](scheduling.md), [several LoRAs / ControlNet](lora-controlnet.md), [deployment](deploy.md) |
| Verification and evaluation | "How do you prove your optimisation is lossless?" "How do you load test?" | warm-up, synchronisation, percentiles; open and closed loop; the limits of PSNR and FID | [benchmarking and load testing](benchmark.md) |

The difference from an LLM inference interview: there is no industry-wide metric here like TTFT or TPOT, so **defining your metrics clearly before answering** is itself a point in your favour; on the other hand nearly every question can be followed with a sentence on "what corresponds to this in an LLM" (guidance is like a prefill at a batch of 2, feature caching is like speculative decoding, sequence parallelism is like tensor parallelism but splitting the tokens), a line of correspondence interviewers like to hear.

## Skeleton answers for ten frequent questions {#十道高频问题的答题骨架}

Each one gets a skeleton and the key numbers, with the details in the corresponding chapter. The numbers are the orders of magnitude computed or cited in this book's chapters, and when answering you say "the order of magnitude" rather than a precise value.

**1. What is the most fundamental difference between diffusion inference and LLM inference?**
No autoregression and no KV cache: one generation = the same network run $N$ steps x the guidance factor, each step a "prefill" at full compute; an LLM's decode is bandwidth-bound and diffusion's denoising is compute-bound. The consequences: a batch saves no time (the compute is already saturated), scheduling does not chase a large batch, and quantization saves memory but not time (unless there is low-precision Tensor Core throughput). See [Diffusion and flow matching from an inference point of view](../basics/diffusion-inference.md) and [the accounting](../perf/accounting.md).

**2. How much compute and time does one 1024² image take on SDXL and on FLUX?**
Per forward pass: about 1.2 TFLOP for SD 1.5, about 12 for SDXL, about 20 for SD3 and about 90 TFLOP for FLUX; multiply by the steps and the guidance (FLUX at 28 steps without guidance ≈ 2.5 PFLOP, SDXL at 30 steps x 2 ≈ 0.7 PFLOP); divide by the effective throughput (an H100 in bf16 at 40% to 50% utilization is 400 to 500 TFLOP/s) for about 5 to 6 seconds for FLUX and about 1.5 to 2 for SDXL; then add a few tenths of a second for the VAE decode. See [the accounting](../perf/accounting.md).

**3. How does FLUX fit into 16 GB or 24 GB?**
The arithmetic: the denoising network is 24 GB in bf16 plus T5's 9.5 GB; FP8 halves each (17 GB, which fits a 24 GB card); NF4 halves again (9 GB, which runs on a 16 GB card). The strategies' costs: model-level offload adds a few seconds of transfer per generation; per-layer offload over 28 steps walks the weights across PCIe 28 times, several times slower than computing; the text encoder can be offloaded as soon as it has run. See [Memory and offload](../perf/memory.md).

**4. How does few-step generation (distillation) turn 50 steps into 4? What does it cost?**
Consistency, adversarial and distribution-matching distillation let the student network jump in one step across several of the teacher's; 4 to 8 steps produce an image and often need no guidance (saving half again); the cost is reduced diversity and controllability, worse compatibility with LoRA and ControlNet, and no freely swapping the scheduler. See [Few-step generation](../perf/distillation.md).

**5. Why does feature caching (TeaCache / FBCache) speed things up, and when does it blur?**
Neighbouring steps' outputs differ relatively by only a few percent through the middle; a probe (the timestep-modulated input, or the first block's residual) accumulates the change and only a crossing of the threshold triggers a real computation, otherwise the previous step's residual is reused; 1.5 to 2.5 times, without changing the weights; a threshold of 0.1 to 0.2 is hard to see and above 0.3 starts to blur; the first and last few steps cannot be skipped, and requests in a batch that are out of step cannot skip at all. See [Feature caching](../perf/caching.md).

**6. How does quantizing a diffusion model differ from quantizing an LLM?**
An LLM's decode is bandwidth-bound, so quantizing the weights makes it faster directly; diffusion is compute-bound, so quantizing the weights only saves memory, and going faster requires quantizing the activations too and using FP8 or INT4 Tensor Cores (W8A8, SVDQuant's W4A4); the activations' outliers change with the timestep, so calibration is per step (timestep-aware smoothing); the sensitive layers (the timestep embedding, AdaLN, either side of attention's softmax) stay at a higher precision. See [quantization](../perf/quantization.md).

**7. Why does multi-card parallelism not use an LLM's tensor parallelism?**
One step is only tens of milliseconds to a second, the weights are small (a few GB to 30 GB) and the activations are a token sequence, so splitting the tokens pays better than splitting the weights: guidance parallelism (a card for each side), sequence parallelism (Ulysses splitting the heads, Ring splitting the sequence), PipeFusion (a pipeline over patches reusing the previous step's features); the communication volume is in the same range as an LLM's tensor parallelism but there is no per-token synchronisation, so it scales better; one image should not use several cards and video must. See [Multi-GPU parallelism](../perf/parallel.md).

**8. Why is video generation so slow, and how is it optimised?**
One 5-second 720p video is still on the order of a hundred thousand tokens after the 3D VAE's compression, full 3D attention is seventy or eighty percent of a step, a 14B model's forward pass is about 6900 TFLOP, and a single H100 takes over twenty minutes. The techniques come in three layers: the kernel (SageAttention's 2.5 times, nearly free), sparsity (spatiotemporal locality lets ten or twenty percent of the blocks cover eighty or ninety percent of the mass, which a block-level kernel turns into 3 to 4 times), and caching and fewer steps; stacked, one card reaches a few minutes, 8 cards about a minute on the ideal accounting, against published figures of 1.5 to 3. See [the video bottleneck](../video/bottleneck.md).

**9. Design a text-to-image service. How does it differ from an LLM service?**
The request shapes are heterogeneous (resolution, steps, plugins) but the duration is predictable (steps x the per-step cost); pool by expected duration and serve first come first served rather than accumulating a batch; pipeline the three stages of text encoding, denoising and the VAE, each saturated; previews plus cancellation save compute; LoRAs at tens of MB all stay resident while ControlNets at a few GB move on demand or get their own instances; under overload, lower the resolution or the steps, or reject. See [Scheduling a generation service](scheduling.md) and [several LoRAs / ControlNet](lora-controlnet.md).

**10. How do you prove your optimisation is lossless?**
At a fixed prompt and seed: find the worst few by per-image PSNR and LPIPS (40 dB is "the same image") and look at them; compare the overall FID and CLIP score as relative values at a fixed sample count; time with warm-up, synchronisation and percentiles; load test a service open loop to find the knee. See [benchmarking and load testing](benchmark.md).

## Two system-design questions {#两道系统设计题}

### Bring one 14B video model's generation from 25 minutes to 2 {#把一条-14b-视频模型的生成从-25-分钟压到-2-分钟}

A skeleton to work from (the arithmetic first, then stacking by layer, then the costs):

1. **The arithmetic**: 5 seconds of 720p, about 100 thousand tokens, 50 steps x guidance 2 = 100 forward passes at about 6900 TFLOP each; attention seventy percent, the linear layers twenty, the VAE tens of seconds. Over twenty minutes on a single H100.
2. **The first layer, without changing the output**: FA3 / SageAttention (2 to 2.5x); FP8 plus compilation (1.4x on the linear layers); CUDA graphs to remove the small operations' overhead.
3. **The second layer, slightly lossy**: block-sparse attention (3 to 4x); TeaCache skipping (1.5 to 2x); guidance distillation to remove half the forward passes.
4. **The third layer, more cards**: 8 cards with Ulysses plus guidance parallelism, the communication a few percent of overhead over NVLink.
5. **The stacking and the costs**: about a minute on the ideal accounting, 1.5 to 3 minutes for published systems; each item's preconditions — sparsity needs calibration, caching needs a threshold, FP8 needs Hopper, parallelism needs NVLink — and the quality verification (VBench sub-scores at a fixed seed plus a human look).

### Design an online text-to-image service supporting several LoRAs, with a P99 under 8 seconds {#设计一个支持多-lora-的在线文生图服务p99--8-秒}

The skeleton:

1. **Metrics and shapes**: define the supported resolutions and the maximum step count, and compute the per-step cost and the expected duration for each; a P99 under 8 seconds means each request's service time has to be 4 to 5 seconds at most (leaving headroom for queueing) → a few-step version, or caching plus FP8.
2. **Scheduling**: pool by expected duration (a short-task pool and a long-task pool), first come first served within a pool, admission by the expected completion time, degrading or rejecting past a threshold.
3. **The pipeline**: the text encoding can take a large batch and be cached; the denoising is one request per card; the VAE decode overlaps the next request's denoising; a preview every N steps produces a low-resolution image and allows cancellation.
4. **LoRA**: tens of MB each, all resident in memory, computed alongside when a request arrives (not merged, which avoids the switching cost and invalidating the compiled graph); a popular combination can be pre-merged into its own weights on the compilation route.
5. **Elasticity and verification**: scale on the queue's waiting time; find the knee with an open-loop load test; run [the deployment chapter](deploy.md)'s checklist before going live.

## A portfolio: four directions you can do on one card {#作品一张卡就能做出来的四个方向}

An inference role looks for "really did it, has numbers, can explain the trade-offs". All four directions below can be done on one 16 GB consumer card, and for the first two this book's code has already set up most of the experiments' skeleton on a CPU.

### 1. A single-card text-to-image optimisation report (★ recommended, two to three weeks) {#1-单卡文生图推理优化报告-推荐两到三周}

Build one optimisation chain on SDXL or FLUX: a bf16 baseline → FP8 / NF4 → `torch.compile` plus CUDA graphs → swapping the attention kernel → TeaCache / FBCache → a few-step version; record the latency (warmed up, percentiles, per stage), the memory peak and the quality (PSNR / LPIPS / FID over a fixed 100 prompts) for every step with [benchmarking and load testing](benchmark.md)'s method.

**Deliverables**: a table of technique x speed x memory x quality, a profiler screenshot and an explanation for each step, and a technical blog post. Being able to explain "FP8 only saves memory and does not speed things up on my card, because…" shows more understanding than "3 times faster".

### 2. A minimal generation service plus a load-test report (two weeks) {#2-最小生成服务--压测报告两周}

Turn the simulators in the [scheduling](scheduling.md) and [several LoRAs](lora-controlnet.md) chapters into the real thing: a three-stage pipeline (the text encoding batched, the denoising, the VAE decode), a queue pooled by expected duration, previews and cancellation, hot-switching among several LoRAs, degradation under overload; load test with an open-loop load and plot the arrival-rate-against-P99 curve and its knee.

**Deliverables**: the code repository, the load-test curves, a performance report written to this book's template, and a post-mortem of deliberately blowing the queue up.

### 3. A video model's attention ledger and a sparsity check (two weeks, no GPU needed) {#3-视频模型的注意力账本与稀疏验证两周可以没有-gpu}

Write a tool: given the model configuration, the resolution, the frame count and the steps, output the token count, the FLOP breakdown (attention / linear / VAE), the memory estimate and the expected time on one card and several ([the video bottleneck](../video/bottleneck.md)'s ledger as a command line); then measure the attention map's spatiotemporal decay with a small 3D attention configuration and check whether "keeping ten or twenty percent of the blocks covers eighty or ninety percent" holds on your model; with a card, connect SageAttention or a block-sparse kernel and measure it.

**Deliverables**: the tool plus an analysis piece on where video generation's time actually goes, with a comparison of the ledger against the measurements.

### 4. Get a PR merged into an inference framework (ongoing) {#4-给推理框架提一个被合并的-pr持续}

The issue lists of diffusers, xDiT, SGLang Diffusion and ComfyUI always have something suitable for a start: a model missing a caching method, offload incompatible with some quantization format, performance numbers out of date in the documentation. Pick one you can verify with this book's method — submitting the PR with a benchmark complete with warm-up, percentiles and quality metrics is itself a portfolio piece.

**Deliverables**: the PR link and your benchmark report; once merged, a CV line saying "contributed Y to X, bringing a Z-times speedup / fixing the W case".

## Write-it-out questions {#手撕题}

A few pieces interviews often ask you to write by hand, all of which have a reference implementation in this book you can adapt directly:

| Question | Key points | Reference |
| --- | --- | --- |
| Write one DDIM or Euler update | predict $x_0$ from $\epsilon$ or $v$ and recombine at the next noise level; flow matching is one Euler step | [Samplers and schedulers](../basics/schedulers.md) |
| Guidance's batch implementation | concatenate the conditional and unconditional into a batch of 2 for one forward pass and extrapolate by the guidance scale; say why they can be concatenated (the same shape) and when they should not be (memory) | [Diffusion and flow matching](../basics/diffusion-inference.md) |
| TeaCache's skip decision | accumulate the probe's relative change, computing for real and resetting only past the threshold, otherwise reusing the previous step's residual; force the computation for the first and last few steps | [Feature caching](../perf/caching.md) |
| A tiled VAE decode | tiles with an overlap, a linear blend in the overlap, and the per-tile memory peak estimated | [The VAE and the latent space](../basics/vae-latent.md) |
| A scheduler pooled by expected duration | request → expected service time → pool; FIFO within a pool, admission by the expected completion time; a cancellation has to release | [Scheduling a generation service](scheduling.md) |
| One of Ulysses's all-to-alls | the conversion between splitting the sequence and splitting the heads, with both all-to-alls' shapes written out | [Multi-GPU parallelism](../perf/parallel.md) |

## How to write it on a CV {#简历上怎么写}

One sentence, one number, one trade-off:

- "Took FLUX.1 1024² from OOM to 9 seconds on a single 16 GB card (NF4 plus compilation plus FBCache), at a median PSNR of 36 dB and an FID of +0.8";
- "Implemented a text-to-image service with a three-stage pipeline and pooled scheduling, at a P99 of 6.5 seconds at 80% capacity utilization under an open-loop load test, 40% lower than a single-queue design";
- "Contributed the Z caching support for the Y model to the X framework (merged), a 1.7-times speedup for HunyuanVideo on one card".

The numbers have to be reproducible and the trade-offs explainable: when the interviewer follows up with "how was that 36 dB of PSNR measured", the [benchmarking and load testing](benchmark.md) chapter is the answer.

## Summary {#小结}

- [x] Four kinds of question: the arithmetic, a technique's preconditions and costs, service design, and verification and evaluation; each one wants numbers, with "what corresponds to this in an LLM" added in passing.
- [x] Ten frequent questions' skeletons and orders of magnitude: the forward FLOPs (1.2 / 12 / 20 / 90 TFLOP), FLUX's memory ladder (24 → 12 → 6 GB), caching's 1.5 to 2.5x, and video's twenty-odd minutes on one card down to a minute or two on 8.
- [x] Answer the two system-design questions as "the arithmetic → stacking by layer → the costs and the verification".
- [x] Four portfolio pieces for one card: an optimisation report, a minimal service plus a load test, a video ledger and sparsity check, and a framework PR; one CV line = one number plus one trade-off.
