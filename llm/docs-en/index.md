# LLM Internals

<p class="lead">Before optimizing inference, you need to really understand what the model computes: how a piece of text becomes numbers, which matrix operations each layer performs, how the next token is chosen, why the KV cache works, and how much compute and memory a request needs. This handbook strings that knowledge together from the start, so that when you read the vLLM or SGLang source or a CUDA kernel, you know exactly which step of the model each line of code corresponds to.</p>

## Why learn this {#为什么要学这些}

Inference optimization sits where three things meet:

- **What the model computes**: which matrix multiplications each Transformer layer has, how attention is computed, and the shapes and parameter counts of each part. That is this handbook;
- **How the hardware executes it**: the GPU memory hierarchy and how to write fast kernels. See [Advanced CUDA](cuda://);
- **How the system schedules it**: batching, KV cache management, parallelism. The last part of this handbook leads into these, after which you can read the vLLM and SGLang code directly.

If you know kernels but not models, you will not know what to optimize; if you know models but not systems, you will not know where the bottleneck is. This handbook covers the first part and connects it to the other two.

## What you will be able to do {#学完能做到}

- Write a LLaMA/Qwen-architecture model from scratch, load real weights, and match the output of the official Hugging Face implementation;
- Explain the principles, shapes and parameter counts of attention, RoPE, RMSNorm, SwiGLU, GQA, MLA and MoE;
- Work out by hand any model's parameter count, compute per token, and the memory taken by weights and KV cache, and estimate a lower bound on decode latency;
- Understand sampling strategies, the KV cache, prefill and decode, quantization, continuous batching, prefix caching, speculative decoding and PD disaggregation, and know which part of the model each optimization acts on;
- Look at a new model's `config.json` and say what is special about its architecture and what to watch for at inference time.

## Learning path {#学习路线}

For the chapter-by-chapter route across all the handbooks (17 weeks, matching the sprint plan week by week, with core and optional chapters, key chapters for different directions, and dependencies across books), see the [roadmap](root://roadmap/). Below is the order within this book.

<div class="roadmap" markdown>

| Stage | Chapters | Goal | Suggested time |
| --- | --- | --- | --- |
| 1. Basics | [Language models](basics/language-model.md) · [Math and PyTorch](basics/math-torch.md) · [Tokenization](basics/tokenization.md) | See the big picture: a large model is a conditional probability function | 3–4 days |
| 2. Anatomy of the Transformer | [Embedding](transformer/embedding.md) · [Attention](transformer/attention.md) · [RoPE](transformer/position.md) · [Normalization](transformer/norm-residual.md) · [FFN](transformer/ffn.md) · [Assembling the model](transformer/build-llm.md) | Write every component yourself and put them together into a model that loads real weights | 1.5 weeks |
| 3. How architectures evolved | [MQA/GQA/MLA](transformer/attention-variants.md) · [MoE](transformer/moe.md) | Understand why today's large models look the way they do | 4–5 days |
| 4. Training and alignment | [Pretraining](training/pretraining.md) · [Post-training](training/post-training.md) | Know where models come from; just enough | 3 days |
| 5. How inference works | [Decoding](inference/decoding.md) · [KV cache](inference/kv-cache.md) · [Estimation](inference/estimation.md) · [Quantization](inference/quantization.md) · [Serving](inference/serving.md) | Master every core concept of inference optimization | 1.5 weeks |
| 6. Putting it together | [The journey of a token](synthesis/token-journey.md) · [Model tour](synthesis/models.md) · [Self-test](synthesis/quiz.md) | Tie all of it into one thread | Come back any time |

</div>

!!! tip "Look up the math when you need it"
    Linear algebra and low rank, probability and sampling, information theory, backpropagation, floating-point error, rooflines and queueing theory have their own book, [Math Fundamentals](math://). This book links there whenever it uses that math, so you do not need to read it first; to check where you stand, try its [self-test](math://quiz/).

## Threads that run through the book {#贯穿全书的主线}

- **One real model**: the whole book uses [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) as its example. It has only 600 million parameters and runs on a CPU, yet its architecture is exactly that of the 8B and 32B models of the same family (GQA, QK-Norm, RoPE, RMSNorm, SwiGLU). The newer Qwen3.5 family switched to a hybrid linear-attention architecture, covered in the [linear attention](serving://frontier/linear-attn/) chapter of the inference systems book.
- **An implementation of your own**: the chapter [assembling a large model from scratch](transformer/build-llm.md) writes `mini_llm.py` (about 200 lines), loads real weights and compares it item by item against the output of Hugging Face transformers. Later chapters run their experiments on it.
- **"Inference view" boxes**: every chapter has boxes like this, explaining what this piece of knowledge corresponds to in inference optimization:

!!! inference "Inference view"
    For example, the attention chapter tells you why attention is memory bound during decode; the FFN chapter tells you why SwiGLU's three matrices are often fused into two GEMMs.

## Setting up {#准备环境}

All the code runs on a CPU; no GPU needed:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers tokenizers safetensors numpy
```

Download the example model (about 1.5 GB):

```py
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", local_dir="models/Qwen3-0.6B")
```

If Hugging Face is hard to reach, set the environment variable `HF_ENDPOINT=https://hf-mirror.com` to use a mirror, or download the model of the same name from ModelScope. The book's code assumes the model is in `models/Qwen3-0.6B` under the current directory. The module files defined in the book (`mini_llm.py` and others) are packaged in [llm-code.tar.gz](assets/llm-code.tar.gz).

!!! note "How the code is verified"
    Every `python` code block and `>>>` interactive example has actually been run under PyTorch 2.14 (CPU) and transformers 5.17, and the interactive examples' output is checked verbatim with doctest. The hand-written model's output has been compared item by item with the official transformers implementation (logits differ on the order of 1e-4, and greedy generation produces exactly the same tokens); the parameter counts listed for well-known models were computed by actually building each model from its official config.
