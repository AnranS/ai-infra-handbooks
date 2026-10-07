# Learning resources

<p class="lead">This handbook is a "map". To go deeper in any direction, you need to read the original papers and source code. Below, the most worthwhile material is listed by chapter, each item with why it is worth reading and what to watch for; at the end come the next steps toward inference optimization after finishing this handbook.</p>

## How to read papers {#读论文的方法}

- **Figures and tables first, then formulas, then the text**. The core of an architecture paper is often one structure diagram and one ablation table.
- **Read with questions**: what bottleneck does this design address? What price does it pay? (Compare with the cause-and-effect picture in [the complete journey of a token](token-journey.md#把整本手册串起来).)
- **Reproduce what you can**: every concept in this handbook has a runnable piece of code, and when reading a paper you can also write a minimal implementation and check it against the paper's numbers.

## By chapter {#按章节}

### Basics and the Transformer {#基础与-transformer}

| Material | Why read it |
| --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) | The original Transformer paper. Note the encoder-decoder structure, Post-Norm and sinusoidal positional encoding, all since changed by later models, and think about why |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Illustrated, good for building intuition on a first pass |
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE) | Where BPE started in NLP |
| [karpathy/minbpe](https://github.com/karpathy/minbpe), [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), [karpathy/nanochat](https://github.com/karpathy/nanochat) | The most readable minimal tokenizer, GPT training code, and a complete small pipeline from pretraining to chat; the accompanying videos are well worth watching too |
| [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE) | The derivation of RoPE; section 3 is enough |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | RMSNorm |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | Where SwiGLU comes from, only 3 pages |
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | Understanding the Transformer from the "residual stream" point of view, echoing this handbook's [residual stream](../transformer/embedding.md#残差流) section |
| [transformers' modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | Compare it line by line with this handbook's `mini_llm.py`, a shortcut to understanding every LLaMA-style model implementation |

### How architectures evolved {#架构演进}

| Material | Why read it |
| --- | --- |
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (MQA) | One of the first papers to design model structure starting from "decode is memory bound" |
| [GQA](https://arxiv.org/abs/2305.13245) | GQA, and how to convert MHA checkpoints to GQA |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | Where MLA comes from; section 2.1 details the low-rank compression, weight absorption and decoupled RoPE |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | Fine-grained MoE, auxiliary-loss-free load balancing, MTP, FP8 training, and the inference deployment plan (section 3.4) |
| [Switch Transformers](https://arxiv.org/abs/2101.03961), [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | MoE routing, capacity and load balancing |
| [LLaMA](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288), [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | Where the standard template came from; the Llama 3 report describes data, training infrastructure and inference (FP8, pipeline parallelism) in detail |
| [Qwen2.5](https://arxiv.org/abs/2412.15115), [Qwen3](https://arxiv.org/abs/2505.09388) technical reports | The "family tree" of the model this handbook uses |
| [Gemma 2](https://arxiv.org/abs/2408.00118), [Gemma 3](https://arxiv.org/abs/2503.19786) | Local/global attention mixes, soft-capping, QK-Norm |
| [gpt-oss model card](https://arxiv.org/abs/2508.10925) | Attention sinks, alternating sliding windows, MXFP4 |
| [YaRN](https://arxiv.org/abs/2309.00071) | The most common RoPE scaling method for long-context extension |

### Training and alignment {#训练与对齐}

| Material | Why read it |
| --- | --- |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), [Chinchilla](https://arxiv.org/abs/2203.15556) | $6ND$, the compute-optimal ratio of parameters to data |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | The classic SFT + RLHF pipeline |
| [DPO](https://arxiv.org/abs/2305.18290) | Follow the derivation of the loss and you understand preference optimization |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) (GRPO), [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | How reasoning models are trained with reinforcement learning; the massive sampling in RL training is a new battlefield for inference engines |
| [LoRA](https://arxiv.org/abs/2106.09685) | Low-rank fine-tuning, and the basis of multi-LoRA serving |

### How inference works, and inference serving {#推理原理与推理服务}

| Material | Why read it |
| --- | --- |
| [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) | The classic blog post on inference estimation, complementing this handbook's [estimation](../inference/estimation.md) chapter |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Google DeepMind's "roofline + parallelism" tutorial; the inference chapter is especially worth reading, from a TPU angle but with general principles |
| [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691), [FlashAttention-3](https://arxiv.org/abs/2407.08608) | IO-aware attention; read with [FlashAttention and inference operators](cuda://advanced/attention/) in the CUDA book |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM) | The design of PagedAttention and vLLM |
| [SGLang](https://arxiv.org/abs/2312.07104) | RadixAttention prefix caching, compressed state machines for structured output |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefill and "stall-free" scheduling |
| [DistServe](https://arxiv.org/abs/2401.09670), [Mooncake](https://arxiv.org/abs/2407.00079) | PD disaggregation; Mooncake is a production architecture centered on the KV cache |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | Speculative decoding and the proof that speculative sampling is correct |
| [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE](https://arxiv.org/abs/2401.15077) | Speculative decoding without a separate draft model |
| [LLM.int8()](https://arxiv.org/abs/2208.07339), [SmoothQuant](https://arxiv.org/abs/2211.10438), [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978) | The outlier problem and the mainstream quantization methods |
| [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453), [Massive Activations in LLMs](https://arxiv.org/abs/2402.17762) | The attention sinks and massive activations this handbook observed in Qwen |
| [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Why results change even at temperature 0, and how to build batch-invariant kernels |

## After this handbook: toward inference optimization {#学完之后走向推理优化}

This handbook covers "what the model computes". The next two parts:

**1. How the hardware executes it: CUDA.** See [Advanced CUDA](cuda://). The chapters most closely tied to this handbook:

- [The road to GEMM optimization](cuda://kernels/gemm/): every linear layer;
- [Softmax and normalization](cuda://kernels/softmax-norm/): RMSNorm, fused kernels;
- [FlashAttention and inference operators](cuda://advanced/attention/): prefill and paged decode attention;
- [Quantization and GEMV](cuda://advanced/quantization/): decode with INT4 weights;
- [Streams, concurrency and CUDA Graphs](cuda://tools/streams/), [multi-GPU and NCCL](cuda://tools/multi-gpu/).

**2. How the system schedules it:** the fourth book of this series, [Inference Systems](serving://), is devoted to this part: write an inference engine from scratch, then read the vLLM and SGLang source against it, followed by distributed inference, performance engineering and interview preparation. To go straight to the source code, the suggested order:

1. [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): implements the core of vLLM (paged KV, continuous batching, prefix caching, CUDA Graphs, tensor parallelism) in a little over a thousand lines; readable in a day or two, the best starting point for understanding a complete engine;
2. [vLLM](https://github.com/vllm-project/vllm): start from a request's lifecycle (API server → engine → scheduler → model executor → sampler), then look at the model implementations you know under `model_executor/models/`, compared against `mini_llm.py`;
3. [SGLang](https://github.com/sgl-project/sglang): focus on the scheduler (prefix caching, overlap scheduling) and the DeepSeek-related optimizations (MLA, DP attention, EP).

When reading source code, use this handbook's "stop by stop" table (see [the complete journey of a token](token-journey.md#逐站解读)) as an index: for every module you meet, ask which stop of the table it corresponds to and which bottleneck it addresses.

**3. Build projects.** A few directions, from easy to hard:

- Add a paged KV cache and continuous batching to `mini_llm.py`, writing a "mini inference engine" of a few hundred lines, and measure how throughput changes with batch size;
- Write fused kernels for RMSNorm, SwiGLU and RoPE in Triton on a GPU, replace the corresponding parts of `mini_llm.py`, and compare speed and accuracy;
- Implement (or reproduce) a small feature in vLLM or SGLang: a new sampling parameter, support for a new model structure, or a kernel optimization, and submit a PR.

## Summary {#小结}

- [x] Read papers figures and tables first, with the questions "which bottleneck does it address, and at what price".
- [x] Every chapter has corresponding original papers; the inference papers are the focus for going deeper.
- [x] Next: the CUDA book covers "how the hardware executes it", and the inference systems book (plus the source code of nano-vllm → vLLM → SGLang) covers "how the system schedules it".
