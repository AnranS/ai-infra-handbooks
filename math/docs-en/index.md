# Math Fundamentals

<p class="lead">The math you need when reading about LLMs and inference systems, in a book of its own to look up when you need it: linear algebra and low rank, probability and sampling, information theory, backpropagation and second-order information, floating-point error, rooflines and queueing. Every result is measured on a real Qwen3-0.6B. You do not need to read it first: the other handbooks link here wherever they use this math, so when something stops you, come and read that section.</p>

## How to use this book {#怎么用这本书}

Two ways:

- **Look things up**: when you hit something from the left column of the table below in another handbook and are not sure about a derivation, open the section on the right, read it, and go back.
- **Check where you stand first**: take the [self-test](quiz.md); its 8 questions cover the whole book, and you can read the chapters behind the ones you cannot answer.

| When you read | The math you need | Read this section |
| --- | --- | --- |
| [Attention](llm://transformer/attention/) | A dot product is a projection; a matrix is a transformation of space | [Linear algebra: geometry first](linear-algebra.md#先看几何矩阵是对空间做的一件事) |
| [RoPE](llm://transformer/position/), rotation-based quantization such as QuaRot | Orthogonal matrices preserve lengths and dot products | [Linear algebra: orthogonal matrices and rotations](linear-algebra.md#正交矩阵与旋转) |
| [Tensor parallelism](serving://distributed/tensor-parallel/), split-K GEMM | Splitting a matrix multiplication by rows, by columns, or along k | [Linear algebra: three views of matrix multiplication](linear-algebra.md#矩阵乘法的三种视角) |
| [LoRA](llm://training/post-training/), [MLA](llm://transformer/attention-variants/) | Rank, singular values and low-rank approximation | [Linear algebra: rank and low-rank approximation](linear-algebra.md#秩与低秩近似) |
| [Decoding and sampling](llm://inference/decoding/) | What temperature, top-k and top-p do to a distribution | [Probability and sampling](probability.md#先看形状温度top-ktop-p-各自在做什么) |
| [Speculative decoding](serving://topics/speculative/) | Why rejection sampling leaves the output distribution unchanged, and how to compute the acceptance rate | [Probability and sampling: rejection sampling and speculative decoding](probability.md#拒绝采样与投机解码) |
| [The training objective of a language model](llm://basics/language-model/) | Entropy, cross-entropy, perplexity | [Information theory: entropy, cross-entropy and KL divergence](information-theory.md#熵交叉熵与-kl-散度) |
| [How quantization works](llm://inference/quantization/) | How far the distribution moves after quantization, and where the error comes from | [Information theory: KL divergence](information-theory.md#kl-散度量化改变了多少), [floating point: quantization noise](floating-point.md#量化噪声每比特-6-db), [calculus: second-order information and GPTQ](calculus.md#二阶信息gptq) |
| [Pretraining](llm://training/pretraining/), [training from scratch](train://scratch/model/) | The chain rule and backpropagation | [Calculus and backpropagation](calculus.md) |
| [Normalization and the residual stream](llm://transformer/norm-residual/), BF16 and FP8 | Rounding, accumulation error, overflow | [Floating point and numerical computing](floating-point.md) |
| [Estimating parameters, compute and memory](llm://inference/estimation/), [profiling](serving://perf/profiling/) | Arithmetic intensity and rooflines | [Performance math: FLOPs, bytes and arithmetic intensity](performance-math.md#flops字节与算术强度) |
| [Load testing, SLOs and capacity planning](serving://perf/benchmark/) | Little's law, queueing theory, tail latency, measurement statistics | [Performance math: queueing theory](performance-math.md#排队论为什么接近满载时延迟爆炸), [the statistics of measurement](performance-math.md#性能测量的统计学) |

## Chapters {#章节}

<div class="roadmap" markdown>

| Part | Chapter | What it covers |
| --- | --- | --- |
| Math in the model | [Linear algebra](linear-algebra.md) | What matrices mean geometrically, three ways to split a matrix multiplication, SVD and low rank (why the weights are not low rank but K and V are), orthogonal matrices and rotations |
| | [Probability and sampling](probability.md) | Temperature and top-k / top-p, sampling algorithms, Monte Carlo error, the probability of a sequence, rejection sampling and speculative decoding, importance sampling |
| | [Information theory](information-theory.md) | Entropy, cross-entropy and KL divergence, where the model is confident, measuring the effect of quantization with KL, forward and reverse KL |
| | [Calculus and backpropagation](calculus.md) | The chain rule, reverse-mode automatic differentiation, backpropagation through a linear layer, the second-order information GPTQ uses |
| Numerics and performance | [Floating point and numerical computing](floating-point.md) | Floating-point formats, rounding and accumulation error, cancellation and overflow, 6 dB of quantization noise per bit |
| | [The math of performance and serving](performance-math.md) | Arithmetic intensity and rooflines, Amdahl's law, Little's law, queueing theory, tail latency amplification, the statistics of performance measurement |
| Self-test | [Self-test questions](quiz.md) | 8 questions covering the whole book, each pointing to its section |

</div>

## Code and running it {#代码与运行}

The examples share an environment with [LLM Internals](llm://): the `mini_llm` implemented there plus real Qwen3-0.6B weights, all running on CPU. The code is written to run from the repository's `llm/` directory (`models/Qwen3-0.6B` and the frozen sample text `docs/assets/sample-passage.txt` are paths relative to `llm/`); see "Setting up" on the LLM Internals home page for the environment.

To check every example in the book, with output compared line by line against the page:

```bash
llm/.venv-llm/bin/python math/tools/check_code.py                         # all 6 chapters
llm/.venv-llm/bin/python math/tools/check_code.py math/docs/probability.md   # check just one chapter
```
