# Learning resources

<p class="lead">This handbook covers the backbone of what interviews and the work itself need. To go deeper, here is a filtered list: the official documentation is the authoritative reference, a few books and courses build a systematic understanding, and open-source code is the best material for the next level.</p>

## Official documentation {#官方文档}

| Document | When to read it |
| --- | --- |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | the authoritative account of the programming model and every feature (including the TMA, clusters and asynchronous copies), with the specification tables for each architecture in the appendix |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | a systematic guide to performance optimization |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | the exact semantics and register layouts of mma, ldmatrix, wgmma, cp.async, mbarrier and the rest |
| [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/) | what the metrics mean, and the hardware model in the Profiling Guide |
| [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) | each API's arguments and behaviour |
| the architecture whitepapers (Ampere, Hopper, Blackwell) | hardware details and specifications, downloadable from NVIDIA's site |

## Books {#书}

| Book | Notes |
| --- | --- |
| *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj, 4th edition) | the classic GPU programming textbook, very systematic on parallel patterns like reduction, scan, convolution and sparse matrices |
| *CUDA C Programming Guide / Professional CUDA C Programming* | practical books from beginner to intermediate, partly dated, good for a quick pass |

## Courses and videos {#课程与视频}

- **GPU MODE** (the YouTube channel and Discord community): a great many high-quality lectures covering CUDA, Triton, CUTLASS, FlashAttention, quantization and inference engines, mostly given by working engineers. PMPP's author has given a companion series too. Strongly recommended.
- **NVIDIA GTC's technical sessions**: every year brings talks on Hopper/Blackwell programming, CUTLASS and Nsight, watchable free on NVIDIA On-Demand.
- University courses on parallel computing (CMU 15-418/618, say) explain parallel computing's basic principles clearly.

## Articles and papers worth reading {#必读的文章和论文}

- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*: the classic blog post on optimizing SGEMM step by step, whose approach this handbook's GEMM chapter follows;
- Mark Harris, *Optimizing Parallel Reduction in CUDA*: the classic slides on reduction optimization (note that its warp synchronization style is now out of date);
- the NVIDIA developer blog: *An Efficient Matrix Transpose in CUDA C/C++*, *How to Access Global Memory Efficiently*, *Using Shared Memory in CUDA C/C++*, the *CUDA Refresher* series and others;
- the FlashAttention papers (Dao et al., 2022; Dao, 2023; Shah et al., 2024);
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, SOSP 2023);
- *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention);
- *Online normalizer calculation for softmax* (Milakov & Gimelshein, 2018);
- *Single-pass Parallel Prefix Scan with Decoupled Look-back* (Merrill & Garland, 2016);
- Colfax Research's CUTLASS / Hopper tutorial series (wgmma, the TMA, warp specialization, FlashAttention-3's implementation details);
- the parts of the DeepSeek-V3 technical report on FP8 training and inference deployment.

## Open-source code worth reading {#值得读的开源代码}

| Project | What to learn |
| --- | --- |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | the CuTe tutorials and examples, Hopper/Blackwell GEMM, how industrial-grade kernels are designed |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | the FA2 (Ampere) and FA3 (Hopper) implementations |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | attention, sampling and MoE kernels for inference, with a clear code organization |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | the CUDA operators under `csrc/` and `sgl-kernel/`, and the Triton kernels |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) / [DeepEP](https://github.com/deepseek-ai/DeepEP) / [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | a compact, high-performance Hopper FP8 GEMM, MoE communication, and an MLA decode kernel |
| [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | a tile-level kernel library aimed at being friendly to learn from |
| [Triton](https://github.com/triton-lang/triton) | the official tutorials (the tutorials directory), from vector addition to FlashAttention |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | fused training operators written in Triton, short and easy to read |

## Places to practise {#练习平台}

- [LeetGPU](https://leetgpu.com/): write CUDA/Triton in the browser and run it on a real GPU, with a problem set and a leaderboard;
- GPU programming contest platforms like Tensara, and the kernel contests the GPU MODE community organizes;
- this handbook's [example code bundle](../assets/cuda-examples.tar.gz): every program checks its own correctness, so they can be rewritten and optimized on top.

## Keeping up {#保持更新}

The GPU hardware and software stack moves very fast. Following the NVIDIA developer blog, CUTLASS's and PyTorch's release notes, vLLM's and SGLang's blogs and roadmaps, and the GPU MODE community's discussions is the most effective way to keep up.
