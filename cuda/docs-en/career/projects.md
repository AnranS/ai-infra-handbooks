# Portfolio projects

<p class="lead">"Familiar with CUDA" on a resume convinces nobody; "my SGEMM reaches 85% of cuBLAS on an A100, and here is the optimization history and the Nsight analysis" does. The 6 projects below are ordered by difficulty, and each one can go straight on a resume and be discussed in depth in an interview. The first three are enough to face an interview for a foundational role with confidence; all six put the core inference-optimization and kernel-development roles within reach.</p>

!!! tip "What makes a project worth doing"
    - **a baseline and numbers**: compare every project against an accepted baseline (cuBLAS, PyTorch, FlashAttention, vLLM's existing implementation), give results for several shapes on several GPUs, and say what percentage of the theoretical ceiling that is.
    - **a visible analysis**: attach the change in Nsight Compute's key metrics to every optimization step and say why you made it.
    - **correctness assured**: tests, covering the boundary cases.
    - **written up**: a GitHub README or a blog post, structured as "the problem → the analysis → optimizing step by step → the results → the limitations". That write-up is the script for what you say in the interview.

## Project 1: the full record of optimizing SGEMM {#项目一sgemm-优化全记录}

**Difficulty**: ★★☆ · **Related chapters**: [GEMM](../kernels/gemm.md), [profiling](../tools/profiling.md), [cp.async](../advanced/async-hopper.md)

Start from the naive implementation and work up through shared-memory tiling, two-dimensional register tiling, vectorization, double buffering / a cp.async pipeline, warp tiling and auto-tuning, until you reach over 80% of cuBLAS's FP32 performance on one GPU.

**Deliverables**
- one kernel per version, with a single test and timing framework;
- a performance progression chart (versions on the x axis, TFLOPS and the percentage of cuBLAS on the y);
- Nsight Compute's analysis of each step: the main stall reason, the shared-memory and device-memory throughput, the change in occupancy.

**Going further**: switch to FP16/BF16 inputs and write a Tensor Core version with WMMA or mma.sync, compared against cuBLAS's HGEMM.

## Project 2: a library of the fused operators a Transformer uses {#项目二transformer-常用融合算子库}

**Difficulty**: ★★☆ · **Related chapters**: [softmax and normalization](../kernels/softmax-norm.md), [Triton](../tools/triton.md), [PyTorch extensions](../tools/ecosystem.md#把-kernel-接入-pytorch)

Implement a set of fused operators in CUDA and register them as PyTorch custom operators: fused_add_rms_norm, RoPE (rotary position embedding, applied in place to Q and K), the SwiGLU activation (`silu(x1) * x2`), and a masked softmax. Write a Triton version of each as well.

**Deliverables**
- operators in `torch.ops` form supporting FP16/BF16, with a full pytest suite;
- a comparison against PyTorch's native implementation, the version `torch.compile` generates and the Triton version, reporting the effective bandwidth as a fraction of the peak;
- substituting these operators into one small model's inference (a small Qwen or LLaMA, say) and measuring the end-to-end speedup.

## Project 3: FlashAttention's forward pass (CUDA + Triton) {#项目三flashattention-前向cuda--triton}

**Difficulty**: ★★★ · **Related chapters**: [FlashAttention](../advanced/attention.md), [Tensor Cores](../advanced/tensor-core.md)

Starting from this handbook's teaching version, implement a FlashAttention-2 style forward kernel that uses the Tensor Cores: each warp handles 16 rows of Q, computing $QK^\top$ and $PV$ with mma.sync and doing the online softmax on the register fragments; support the causal mask and head_dim 64/128. Write a Triton version too.

**Deliverables**
- a comparison against the official FlashAttention-2 and PyTorch SDPA at various sequence lengths;
- a clear account of the fragment layout, how online softmax works on registers, and the shared-memory swizzle;
- correctness tests covering unaligned sequence lengths, causal and non-causal.

**Going further**: implement the backward pass; or implement Flash-Decoding (split-K decode attention).

## Project 4: a paged decode attention kernel {#项目四分页-decode-注意力-kernel}

**Difficulty**: ★★★ · **Related chapters**: [PagedAttention](../advanced/attention.md#pagedattention)

Building on this handbook's `paged_decode.cu`, implement a decode attention close to production quality: a BF16/FP8 KV cache, 128-bit vectorized reads, one block handling all the query heads that share a KV head under GQA, online softmax accumulating as it scans, and split-K for long sequences. Wire it into vLLM's or SGLang's attention back-end interface (or at least use the same KV cache layout they do).

**Deliverables**
- a comparison against FlashInfer's and vLLM's existing kernels at various batch sizes and context lengths (bandwidth utilization);
- an analysis of how the GQA ratio, the number of splits and the page size affect performance.

## Project 5: a quantized GEMV / GEMM {#项目五量化-gemv--gemm}

**Difficulty**: ★★★☆ · **Related chapters**: [quantization and GEMV](../advanced/quantization.md)

Implement a W4A16 (group-wise quantization, group size 128) GEMV and small-batch GEMM: offline weight reordering, fast dequantization in registers (bit tricks), computation on the Tensor Cores. Compare against BF16 cuBLAS and Marlin for batch = 1 to 64.

**Deliverables**
- a speedup curve across batch sizes, with the knee explained;
- the quantization error and the kernel implementation's error reported separately;
- a clear account of the weight packing format and dequantization's bit operations.

## Project 6: contribute to an open-source inference engine {#项目六为开源推理引擎做贡献}

**Difficulty**: ★★★★ · **Related chapters**: all of them

Find a concrete performance problem or a missing kernel in vLLM, SGLang, FlashInfer, TensorRT-LLM, DeepGEMM or a similar project (a new model's unusual operator, a performance regression at some shape, support for new hardware), submit a PR and get it merged.

**How to start**
- read the project's issues and filter for the performance, kernel and good first issue labels;
- get the project's benchmark scripts running first, and use nsys to find the most time-consuming kernel in one concrete workload;
- start with a small improvement (one fusion, one parameter adjustment, one boundary-case fix) to get familiar with the code structure and the review process.

A merged PR is the most convincing project on a resume: it proves your code meets industrial standards and shows you can work in a large code base.

## How to write it on a resume {#简历怎么写}

A counterexample:
> Familiar with CUDA programming, understand GPU architecture, have done GEMM optimization.

A good example:
> Implemented an FP32 SGEMM (shared-memory/register tiling, a three-stage cp.async pipeline, warp tiling) reaching 86% of cuBLAS on an A100 at size 4096; located and removed shared-memory bank conflicts with Nsight Compute (conflicts down 97%). Code and analysis: github.com/xxx

The elements: **what you did (the techniques) + the result (a number + a baseline) + how you verified it (the tools) + a link**.
