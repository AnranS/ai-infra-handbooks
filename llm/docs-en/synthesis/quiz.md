# Self-test questions

<p class="lead">After a first pass through the book, use these questions to check whether it has really come together for you. They are organized by chapter, and the last part is cross-chapter synthesis questions, which are also the most common kind in inference interviews. Answer first (ideally out loud or in writing), then expand the reference answer. For any question you cannot answer, go back and reread the corresponding chapter.</p>

!!! tip "How to use these questions"
    - First pass: 2 minutes per question, just mark "can answer / fuzzy / cannot";
    - Second pass: only the "fuzzy" and "cannot" ones; reread the linked chapter, then answer again **without looking at the answer**;
    - Before an interview: pick the synthesis questions and practice explaining each one fully in 3 to 5 minutes, including the numerical estimates.
    - Math questions (low rank, rejection sampling, KL divergence, GPTQ, floating-point accumulation, queueing theory) are in the [Math Fundamentals self-test](math://quiz/); do them when you need them.

## Basics {#基础}

**1. What are a language model's input and output at inference time? Why can generation only proceed one token at a time?**

??? success "Answer"
    The input is a token sequence, and the output is the probability distribution of the next token at **every position** (logits through softmax). In generation, token t+1 is conditioned on token t, and token t has to wait for the previous step's sampling result, so it can only be serial. In training, the targets of all positions are known, so they can be computed in parallel (teacher forcing). See [language models](../basics/language-model.md#训练可以并行推理只能串行).

**2. What perplexity does a cross-entropy loss of 2.166 correspond to? What does perplexity mean intuitively?**

??? success "Answer"
    $e^{2.166} \approx 8.72$. Intuitively, the model on average is like guessing uniformly among about 8.7 candidates. A uniform distribution over the whole vocabulary has a perplexity equal to the vocabulary size (151936 for Qwen). See [the training objective: cross-entropy](../basics/language-model.md#训练目标交叉熵).

**3. Why does softmax subtract the maximum? How does this trick evolve into online softmax in attention?**

??? success "Answer"
    $e^x$ overflows for large x; after subtracting the maximum every exponential is ≤ 1, and the result is unchanged. Online softmax, processing block by block, keeps a "current maximum" and a "current denominator", and whenever a larger value appears it rescales the accumulated results by $e^{m_{old} - m_{new}}$. This is the key to FlashAttention not materializing the whole score matrix. See [softmax and numerical stability](../basics/math-torch.md#softmax-与数值稳定).

**4. What is the difference between BF16 and FP16? Why do large-model training and inference mostly use BF16?**

??? success "Answer"
    Both are 16 bits. BF16 has 8 exponent bits (the same as FP32) and 7 mantissa bits; FP16 has 5 exponent bits and 10 mantissa bits. BF16 covers the same range as FP32, rarely overflows and needs no loss scaling; the price is lower precision. Large models' activations contain very large outliers, so range matters more than precision. See [number formats](../basics/math-torch.md#数值格式).

**5. How is BPE tokenization trained? What are the advantages of byte-level BPE?**

??? success "Answer"
    Start from single bytes (or characters), repeatedly count the frequency of adjacent token pairs, and merge the most frequent pair into a new token until the target vocabulary size is reached. Encoding applies the merge rules in the order they were learned. Byte-level BPE's base vocabulary is the 256 bytes, so any text (any language, emoji, garbage) can be encoded, and unknown tokens never occur. See [BPE](../basics/tokenization.md#bpe从字节开始不断合并).

**6. In streaming output, why can't each token be decoded on its own as it is generated?**

??? success "Answer"
    A Chinese character usually takes 3 bytes in UTF-8, and byte-level BPE may split it across two tokens, so decoding one token alone gives incomplete bytes (shown as �). The right way is incremental detokenization: remember the text already output, decode a window each time, and output only the new, complete part. See [streaming output and incremental detokenization](../basics/tokenization.md#流式输出与增量反分词).

**7. What happens if you use the wrong chat template?**

??? success "Answer"
    In post-training the model only saw conversations in a specific format (special tokens, role markers); with the wrong format it treats the input as ordinary text to continue, and its behavior degrades noticeably: continuing the user's words, not stopping, answering off topic. Inference engines must use the model's own chat template and set the stop tokens correctly. See [special tokens and chat templates](../basics/tokenization.md#特殊-token-与对话模板).

## Transformer {#transformer}

**8. Write the formula for scaled dot-product attention. Why divide by $\sqrt{d_k}$?**

??? success "Answer"
    $\text{softmax}(QK^\top / \sqrt{d_k}) V$. If the components of q and k are independent with variance 1, the dot product has variance $d_k$ and standard deviation $\sqrt{d_k}$. Without scaling, the scores grow with the dimension, softmax tends to one-hot, and the gradients become tiny. See [scaled dot-product attention](../transformer/attention.md#缩放点积注意力).

**9. With a KV cache, how is the causal mask for new tokens written?**

??? success "Answer"
    There are T queries (the new tokens) and S keys (history + new tokens). New token i sits at absolute position S − T + i and can see keys at positions ≤ S − T + i, that is, `ones(T, S).tril(diagonal=S - T)`. In decode T = 1 and no mask is needed. See the implementation in [assembling a large model from scratch](../transformer/build-llm.md#完整代码).

**10. What is the core idea of RoPE? Why do we say it encodes relative position?**

??? success "Answer"
    Treat every two dimensions of q and k as a 2D vector and rotate it by an angle of "position × frequency". When q at position m is dotted with k at position n, the rotation matrices combine into $R_{n-m}$, so the result depends only on the relative distance n − m. It acts only on q and k, not v; at inference time the cached K is already rotated. See [RoPE](../transformer/position.md#rope用旋转编码位置).

**11. What is an attention sink? How does it affect inference?**

??? success "Answer"
    Many heads put a large share of their attention on the first token (this handbook measured 40% to 77% from layer 6 on in Qwen3-0.6B), making the first token a kind of trash can for "looking at nothing". The effects: KV eviction methods such as StreamingLLM must keep the first few tokens; these tokens carry massive activations, which makes activation quantization hard; gpt-oss simply gives every head a learnable sink term. See [attention sinks](../transformer/attention.md#真实模型里的注意力注意力汇聚).

**12. What is the difference between Pre-Norm and Post-Norm? Why is Pre-Norm used everywhere now?**

??? success "Answer"
    Post-Norm normalizes after the residual addition; Pre-Norm normalizes before the sublayer and keeps the residual path an identity. With Pre-Norm, gradients flow straight back along the residual path, so deep models train more stably without delicate warmup. The price is that the residual stream's values grow with depth, so a final norm is needed at the end. See [Pre-Norm and Post-Norm](../transformer/norm-residual.md#pre-norm-与-post-norm).

**13. What does SwiGLU add over an ordinary MLP? How is the parameter count kept comparable?**

??? success "Answer"
    A gating projection: $\text{down}(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x))$, three matrices instead of two. To keep the parameter count comparable, the intermediate dimension shrinks from 4d to about $\frac{8}{3}d$ (LLaMA rounds it up to a multiple of 256). At inference time gate and up are merged into one GEMM, and SiLU and the multiplication are fused into one kernel. See [gating: SwiGLU](../transformer/ffn.md#门控swiglu).

**14. Where are the parameters of a dense model mostly?**

??? success "Answer"
    About $4d^2$ per layer for attention (less with GQA) + $3 d \cdot d_{ff}$ for the FFN (usually 2 to 3 times attention), plus vocabulary × d for the embedding and output layers. In large models the FFN holds about two thirds; in small models the vocabulary takes a large share (Qwen3-0.6B's embedding is 26.1%). See [parameters of each part](../transformer/build-llm.md#各部分的参数量).

**15. How do MQA, GQA and MLA each reduce the KV cache? What does MLA cost?**

??? success "Answer"
    MQA: all query heads share one set of K and V; GQA: each group of query heads shares one set of K and V (the compromise); MLA: compress K and V into a low-dimensional latent vector for caching and "decompress" it with a matrix when used; at inference time the decompression matrices can be absorbed into the query and output projections. MLA's costs: RoPE is incompatible with the low-rank compression, so it needs separate decoupled RoPE dimensions; the attention kernel is more complex; and the latent KV cannot be split by head, which does not suit tensor parallelism. See [attention variants](../transformer/attention-variants.md).

**16. What is MoE routing? What does the difference between total and active parameters mean for inference?**

??? success "Answer"
    The router computes each expert's score for every token, picks the top k, and weights their outputs by the (normalized) scores. Memory must hold all the experts (total parameters), but each token computes only k experts (active parameters). So MoE needs a lot of memory but little computation per token; in decode the batch's tokens are scattered across experts, so a larger batch is needed to amortize the weight reads. See [MoE](../transformer/moe.md).

## Training and alignment {#训练与对齐}

**17. Roughly how much compute does training a 7B model on 2T tokens take? How long on 1000 H100s at 40% MFU?**

??? success "Answer"
    $6ND = 6 \times 7 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{22}$ FLOPs. 1000 × 989 TFLOPS × 0.4 ≈ $4 \times 10^{17}$ FLOP/s, so about $2.1 \times 10^5$ seconds, about 59 hours. See [how much compute training needs](../training/pretraining.md#训练需要多少算力).

**18. What does the Chinchilla law say? Why are today's small models all "overtrained"?**

??? success "Answer"
    For a given training compute, the optimal number of tokens is about 20 times the parameter count. But Chinchilla only optimizes training cost; a deployed model's inference cost is proportional to its parameter count and independent of the training tokens, so it pays to train a smaller model on far more than 20 times the data (LLaMA 3 8B used 15T tokens, about 1900 times). See [scaling laws](../training/pretraining.md#scaling-law).

**19. How does the SFT loss differ from pretraining?**

??? success "Answer"
    The loss function is the same (next-token cross-entropy), but it is computed only on the answer; the positions of the system prompt and user input are masked out (labels set to −100). The data is in conversation format, using the chat template. See [SFT](../training/post-training.md#sft监督微调).

**20. What does DPO save compared with RLHF (PPO)? And what does GRPO save?**

??? success "Answer"
    DPO saves the reward model and the reinforcement-learning sampling loop, building a classification-style loss directly from preference pairs (a good and a bad answer) and a frozen reference model. GRPO is still online reinforcement learning, but it saves the value network (critic): it samples a group of answers to the same question and normalizes by the mean and standard deviation of the group's rewards to get advantages. Reasoning models (the R1 kind) are mainly trained with GRPO + verifiable rewards. See [post-training](../training/post-training.md).

**21. How does LoRA work? How is multi-LoRA serving done?**

??? success "Answer"
    Freeze the original weights W and train only a low-rank update $BA$ (r much smaller than d), so the output is $Wx + BAx$. At inference time BA can be merged into W or kept separate: multi-LoRA serving lets all requests share one GEMM of the base model, then computes each request's low-rank part with its own adapter (the segmented GEMM kernels of Punica and S-LoRA), so one service can serve hundreds or thousands of fine-tuned versions at once. See [LoRA](../training/post-training.md#lora低秩微调).

## How inference works {#推理原理}

**22. How do temperature, top-k, top-p and min-p each act on the logits? Does the order matter?**

??? success "Answer"
    Temperature divides the logits by T; top-k keeps only the k largest; top-p keeps the smallest set whose cumulative probability reaches p; min-p keeps tokens with probability at least "maximum probability × p". The order affects the result (temperature before top-p, for example, changes the cumulative probabilities), and frameworks do not all use the same order, which is one reason "the same parameters give different results in different frameworks". See [decoding and sampling](../inference/decoding.md).

**23. Why can HF generate's "greedy" result differ from an argmax you write yourself?**

??? success "Answer"
    The model's own `generation_config.json` may set default repetition penalty, temperature, top-p and top-k (Qwen2.5 sets repetition_penalty 1.1, among others), and these defaults are applied automatically. For a strict comparison, turn them off explicitly. See [a model's own default parameters](../inference/decoding.md#模型自带的默认参数).

**24. Why can K and V be cached, while Q need not be?**

??? success "Answer"
    The causal mask guarantees that past tokens' K and V depend only on themselves and earlier tokens, so new tokens do not change them; Q is used only once, when the current token computes its attention, and is never needed again. See [why K and V can be cached](../inference/kv-cache.md#为什么-kv-可以缓存).

**25. Are prefill and decode compute bound or memory bound? Why?**

??? success "Answer"
    Prefill processes hundreds or thousands of tokens at once and does many operations per weight read: high arithmetic intensity, compute bound. Decode processes only 1 token per request per step and does about 2 operations per 2 bytes of weights read: an arithmetic intensity of about the batch size, far below the GPU's ridge point (about 295 for an H100), memory bound. See [prefill and decode](../inference/kv-cache.md#prefill-与-decode).

**26. LLaMA-3-70B (BF16) with tensor parallelism on 4 H100s: what is the lower bound on decode time per token at batch = 1? How large is the KV cache per token?**

??? success "Answer"
    The weights are 70.55B × 2 bytes ≈ 141 GB, and the 4 GPUs' total bandwidth is 4 × 3.35 TB/s = 13.4 TB/s, so the lower bound is about 10.5 ms per token (plus communication and other overhead). KV cache: 2 × 80 layers × 8 KV heads × 128 × 2 bytes = 320 KB per token. See [estimating parameters, compute and memory](../inference/estimation.md).

**27. Why can INT4 weight-only quantization make decode about 3 to 4 times faster, yet hardly speed up prefill at all?**

??? success "Answer"
    Decode is memory bound and its time is about the time to read the weights; with weights at about 1/4 the bytes, the time approaches 1/4 too. Prefill is compute bound; weight-only quantization dequantizes the weights back to BF16 before computing, so the computation is unchanged and dequantization adds overhead. Speeding up prefill needs schemes that quantize activations too, such as W8A8 (FP8/INT8), computing on low-precision Tensor Cores. See [how quantization works](../inference/quantization.md).

**28. Why is per-group quantization (group size 128) so much more accurate than per-channel? Why are activations harder to quantize than weights?**

??? success "Answer"
    The fewer numbers share a scale, the less an outlier affects them (this handbook measured on Qwen3-0.6B a W4 per-channel perplexity of 55.02 and per-group 41.49, against 25.94 for the original model). Activations are hard to quantize because a few channels have huge outliers (this handbook measured massive activations of about 6900 in the residual stream, and in the FFN input the largest channel is over 50 times the median), and activations differ every time, so scales can only be computed online. SmoothQuant "moves" activation outliers into the weights, and AWQ protects the important weight channels according to activation magnitude. See [how quantization works](../inference/quantization.md#激活量化与离群值).

## Inference serving {#推理服务}

**29. What decides TTFT, TPOT and throughput? How do they constrain each other?**

??? success "Answer"
    TTFT = queueing time + prefill time; TPOT is set by the time of each decode step and grows with batch size and context length; throughput rises with the batch. A larger batch raises throughput but slows TPOT; inserting new requests' prefills makes the requests in decode stall. The goal is to maximize goodput while meeting the SLO. See [metrics](../inference/serving.md#指标).

**30. What problem does continuous batching solve? What does it require of the attention kernel?**

??? success "Answer"
    In static batching, short requests that finish must wait for the long requests in the same batch, leaving the GPU idle. Continuous batching schedules step by step: finished requests leave and new ones join immediately. Requirements: requests in the same batch have different context lengths, keep their KV in different places and may even be in different phases, so the attention kernel must support variable-length sequences and paged KV (block tables), and the input layout concatenates all tokens into one dimension instead of padding into a rectangle. See [continuous batching](../inference/serving.md#连续批处理).

**31. Which operating-system idea does PagedAttention borrow? What does it bring?**

??? success "Answer"
    Virtual memory and paging: the KV cache is cut into fixed-size blocks, and each request uses a block table to map logical blocks to physical blocks. Benefits: almost no fragmentation or reservation waste, so concurrency rises significantly; identical prefixes can share physical blocks (prefix caching, parallel sampling, beam search); and preemption can swap out block by block. See [managing KV cache memory](../inference/kv-cache.md#kv-cache-的显存管理).

**32. Why does speculative decoding speed things up? Why does it not change the output distribution? When does it gain little?**

??? success "Answer"
    Decode is memory bound, and verifying k tokens reads as many weights as generating 1, so correct guesses mean one forward pass advances several tokens. Under greedy decoding the argmaxes are compared one by one, and the output is identical to ordinary greedy; under sampling, rejection sampling (accept with $\min(1, p/q)$; on rejection, sample from the normalized $\max(0, p-q)$) makes the output distribution exactly the large model's. It gains little when the draft is poor (low acceptance rate) or the batch is large (already near compute bound, so verification is no longer "free"). See [speculative decoding](../inference/serving.md#投机解码).

**33. Why do PD disaggregation? What does it cost?**

??? success "Answer"
    Prefill is compute intensive and decode memory intensive; mixed together they interfere (TTFT and TPOT drag each other down), and neither can choose its own optimal parallelism and batch size. Separated, each can be optimized and scaled independently. The costs: the KV cache must be transferred across machines (needing fast networks such as RDMA), the system gets more complex, and the resource ratio must be adjusted with the load. See [PD disaggregation](../inference/serving.md#pd-分离).

## Synthesis {#综合题}

**34. What happens between the user pressing Enter and seeing the first character? Walk through it as completely as you can.**

??? success "Answer"
    The HTTP request reaches the API server → the chat template is applied and the text tokenized → it enters the scheduler's waiting queue → the scheduler looks up the prefix cache, allocates KV blocks for the part that missed, and adds it (possibly in chunks) to some step's batch → model execution: embedding → each layer (RMSNorm → QKV projections → RoPE → write paged KV → FlashAttention → O projection → residual → RMSNorm → SwiGLU MLP → residual), with all-reduces per layer under tensor parallelism → final norm → the output layer for the last position only → sampling (temperature, top-p and so on) → incremental detokenization → the first token streams back. TTFT = queueing + prefill (+ network and tokenization). See [the complete journey of a token](token-journey.md).

**35. As a service's batch grows from 1 to 64, how does the time of each decode step change? Where does the bottleneck move from and to?**

??? success "Answer"
    The weight operators' intensity is about the batch size, so from 1 to 64 their time barely changes (still a bit below the ridge point); attention's intensity equals the GQA group size, independent of the batch, and the KV read grows linearly with the batch. For LLaMA-3-8B at a 4K context on an H100, the lower bound grows from 4.6 ms to 14.8 ms, and attention's share from 3% to 70%. The bottleneck moves from "reading weights" to "reading the KV cache". That is why KV quantization, MLA and paged decode kernels matter more at large batches. See [where decode time goes](token-journey.md#decode-的时间花在哪里).

**36. Why can the same request at temperature 0 give two different results in a live service?**

??? success "Answer"
    Floating-point addition is not associative. The size of the batch a request lands in, which requests it is batched with, how chunked prefill is split and which kernel implementation is chosen can all change the order of accumulation, giving slightly different logits. When two candidate tokens have very close probabilities, the argmax flips, and the text after it is completely different. This handbook measured, on a CPU alone, a logits difference of about 3e-5 between computing alone and within a batch. Strict reproducibility needs batch-invariant kernels. See [which optimizations change the output](token-journey.md#哪些优化会改变输出).

**37. Given a new model's config.json, how would you estimate the resources needed to deploy it?**

??? success "Answer"
    1. Count parameters: total parameters (deciding weight memory) and active parameters (deciding compute);
    2. Compute the KV cache per token (watch for GQA, MLA, sliding windows, linear-attention layers);
    3. Memory: weights + KV cache (target concurrency × average context × KV per token) + activations and runtime overhead (reserve 10% to 20%);
    4. Latency: the decode lower bound ≈ (weight bytes + the batch's KV bytes) / bandwidth; prefill ≈ 2 × active parameters × tokens / effective compute;
    5. Parallelism: whether the heads and KV heads divide evenly by TP; whether MoE needs EP;
    6. Special structures: whether specific kernels are needed (MLA, soft-capping, attention sinks, linear attention) and whether the inference engine supports them.

    See [a tour of mainstream model architectures](models.md) and [estimating parameters, compute and memory](../inference/estimation.md).

**38. List the inference optimizations you know that "trade compute for memory traffic" or "memory for compute".**

??? success "Answer"
    - Compute for memory traffic: KV cache quantization (read fewer bytes, do extra dequantization), weight-only quantization (likewise), FlashAttention (do not write out the scores; recompute in the backward pass), MLA (cache a latent vector and do a matrix multiplication when using it), speculative decoding (extra verification compute for fewer weight reads);
    - Memory for compute: the KV cache itself (store K and V to avoid recomputing the whole prefix), prefix caching (likewise, reused across requests).

    In essence they all trade between the two dimensions of the roofline: see which side the bottleneck is on, and move the cost to the other side.

**39. If you were optimizing inference for a RAG application (very long inputs, very short outputs, many requests sharing the same set of documents), where would you start?**

??? success "Answer"
    This workload is dominated by prefill, and TTFT is the key metric:

    - Prefix caching: put the shared system prompt and documents at the very front of the prompt so prefixes match as much as possible, maximizing the hit rate (RadixAttention / APC);
    - Chunked prefill, so long prefills do not block other requests' decode;
    - Prefill is compute bound, so FP8 (W8A8) is more effective than weight-only INT4;
    - With high concurrency, consider PD disaggregation and give prefill more resources;
    - Attention costs a lot at long contexts, so make sure efficient kernels such as FlashAttention are used;
    - Outputs are short, so speculative decoding gains little.

**40. Compress the whole handbook into three sentences.**

??? success "Answer"
    (A sample answer; you may have your own version.)

    1. A large model is a Transformer trained on "the probability of the next token": an embedding, a number of layers (attention + FFN, with residuals and normalization) and an output layer; at inference time it can only generate one token at a time.
    2. Inference has two phases: prefill processes the prompt in parallel and is compute bound; decode produces one token per step, must read all the weights and the KV cache, and is memory bound.
    3. Nearly every inference optimization does one of three things: read fewer bytes (quantization, GQA/MLA, sparsity), use each read more (batching, speculative decoding, prefix sharing), or keep the hardware from idling (continuous batching, chunked prefill, PD disaggregation, operator fusion, CUDA Graphs).
