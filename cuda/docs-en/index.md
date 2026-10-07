# Advanced CUDA

<p class="lead">A CUDA path for AI infrastructure, LLM inference optimization and GPU kernel development roles. It starts from the GPU architecture and works all the way to hand-written GEMM, softmax, FlashAttention and quantized GEMV, then on to Tensor Cores, the newer Hopper features, Nsight performance analysis and multi-GPU communication. It ends with an interview question bank and portfolio projects.</p>

## Who this handbook is for {#这份手册适合谁}

- You know C++: pointers, arrays, templates and the basics of classes give you no trouble.
- You know what a matrix multiply and a softmax are, and ideally you have used PyTorch.
- You want to move into GPU programming or go deeper, with the goal of passing interviews for AI infrastructure, inference optimization or kernel development, and of writing and tuning kernels on your own at work.

No parallel-programming or GPU experience is required.

## What these roles ask for {#目标岗位需要什么}

| Role | Day to day | What interviews focus on |
| --- | --- | --- |
| Inference optimization / AI infrastructure | optimizing inference engines like vLLM, SGLang and TensorRT-LLM; writing fused kernels, quantized kernels and attention kernels; analyzing end-to-end latency and throughput | CUDA fundamentals, hand-written kernels, how FlashAttention / PagedAttention work, quantization, Nsight analysis, inference-engine architecture |
| Kernel development / GPU performance | writing high-performance kernels for new models and new hardware (GEMM, attention, MoE, fused communication and compute), measured against cuBLAS and the best community implementations | the full toolkit of memory and compute optimization, Tensor Cores, CUTLASS/CuTe or Triton, roofline analysis |
| Training frameworks / distributed | training speedups, communication optimization, memory optimization | CUDA fundamentals, streams and concurrency, NCCL and collectives, mixed precision |
| HPC / scientific computing | numerical simulation, solver acceleration | CUDA fundamentals, memory optimization, parallel patterns such as reduction and scan, multi-GPU |

Almost every one of these interviews tests three things: **explaining the GPU's execution and memory model**, **writing a kernel on the spot and optimizing it step by step** (reduction, transpose, softmax and GEMM are the most common), and **explaining with numbers why a kernel is slow** (roofline, bandwidth utilization, Nsight metrics). This handbook is built around those three.

## What you will be able to do {#学完能做到}

- Explain how SMs, warps, thread blocks, registers, shared memory, L2 and HBM relate, and explain any kernel's performance from that.
- Write and optimize a reduction, a transpose, a softmax, a LayerNorm and a GEMM on your own, with the GEMM reaching 70-80% of cuBLAS in FP32.
- Call Tensor Cores through WMMA / mma.sync, and know what cp.async, TMA and wgmma are for.
- Explain why FlashAttention is fast, and write a working FlashAttention forward kernel.
- Locate bottlenecks with Nsight Systems and Nsight Compute, and propose optimizations backed by data.
- Write high-performance kernels quickly in Triton, and register custom kernels with PyTorch.

## Learning path {#学习路线}

The chapter-by-chapter path across all eight handbooks (17 weeks, matching the sprint plan week by week, marking core and optional chapters, the focus for different roles, and the dependencies between books) is in the [roadmap](root://roadmap/). Below is the order inside this book.

<div class="roadmap" markdown>

| Stage | Chapters | Goal | Suggested time |
| --- | --- | --- | --- |
| 1. Fundamentals | [GPU architecture](basics/gpu-architecture.md) · [Your first program](basics/first-kernel.md) · [The memory hierarchy](basics/memory.md) · [The execution model](basics/execution.md) · [Synchronization and warp programming](basics/sync-warp.md) | write correct kernels, and explain where their performance comes from | 3 weeks |
| 2. Classic kernels | [Reduction](kernels/reduction.md) · [Transpose](kernels/transpose.md) · [GEMM](kernels/gemm.md) · [Softmax and normalization](kernels/softmax-norm.md) · [Prefix sum](kernels/scan.md) | write every interview kernel from scratch and optimize it step by step | 4 weeks |
| 3. Tools | [Nsight](tools/profiling.md) · [Streams and CUDA Graphs](tools/streams.md) · [PDL and megakernels](tools/pdl-megakernel.md) · [Triton](tools/triton.md) | analyze, optimize at the system level, and move faster with Triton | 2 weeks |
| 4. Modern GPUs and AI kernels | [Tensor Cores](advanced/tensor-core.md) · [Hopper/Blackwell](advanced/async-hopper.md) · [CuTe layout algebra](advanced/cute-layout.md) · [FlashAttention](advanced/attention.md) · [Quantization and GEMV](advanced/quantization.md) | read and rewrite the core kernels inside an inference engine | 4 weeks |
| 5. Frameworks and compilers | [A tensor's memory model](framework/tensor.md) · [autograd](framework/autograd.md) · [The dispatcher and custom ops](framework/dispatcher.md) · [The CUDA runtime](framework/cuda-runtime.md) · [torch.compile](framework/compile.md) · [The AI compiler landscape](framework/compilers.md) | see what PyTorch does above the kernels, register custom ops correctly, and get the most from torch.compile | 1-2 weeks |
| 6. Engineering and job hunting | [Multi-GPU](tools/multi-gpu.md) · [The ecosystem](tools/ecosystem.md) · [Interview questions](career/interview.md) · [Portfolio](career/projects.md) | have projects worth showing, and answer interview questions fluently | 3 weeks or more |

</div>

Start weaving the tools stage (Nsight) in right after GEMM; there is no need to wait until the end.

## How to study {#怎么学}

1. **Write every kernel yourself.** Get a working version down before looking at the answer, then optimize it against the text. Reading without writing guarantees you cannot write it in an interview.
2. **Measure every optimization.** Record each version's time, bandwidth or throughput, and work out the percentage of hardware peak. An optimization you cannot put a number on did not happen.
3. **Check your guesses with Nsight Compute.** "I think it is a bank conflict" is not a conclusion; seeing the metric is.
4. **Write notes or a blog.** Turning each kernel's optimization into an article gives you something for your résumé and the best material for an interview.

## Setting up a GPU {#准备-gpu-环境}

The code in this handbook needs an NVIDIA GPU to run. The usual ways to get one:

| Option | Notes |
| --- | --- |
| A company or university development machine | the most convenient, usually an A100/H100/H20/L20 |
| Cloud GPUs by the hour | every major cloud has them; an RTX 4090 or A100 costs a few dollars an hour, which suits a concentrated practice session |
| Google Colab | the free tier gives a T4 (sm_75), enough for the fundamentals and the classic kernels |
| [LeetGPU](https://leetgpu.com/) | write CUDA in the browser and run it on a real GPU, with a question bank for practice |

**On a Mac**: there is no NVIDIA GPU, but the repository's CUDA-to-CPU simulator can check a kernel's correctness first (not its performance): `python tools/emu_run.py reduction.cu` runs the book's examples and `python tools/emu_run.py yourfile.cu` runs your own, and the exercises' local grader switches to the simulator on a Mac automatically. Save the performance experiments for when you have a GPU and do them together; the full setup is in [the learning environment](root://setup/).

What different chapters need from the hardware:

| GPU | Architecture | What you can study |
| --- | --- | --- |
| T4, RTX 20 series | Turing, sm_75 | fundamentals, classic kernels, WMMA (FP16 Tensor Cores) |
| A100, RTX 30/40 series, L4, L20 | Ampere / Ada, sm_80, sm_86, sm_89 | all of the above, plus cp.async, BF16, mma.sync m16n8k16, FlashAttention and nearly everything else |
| H100, H800, H20 | Hopper, sm_90 | all of the above, plus TMA, thread-block clusters, wgmma and FP8 |
| B200, RTX 50 series | Blackwell, sm_100, sm_120 | the newest features (tcgen05, FP4), which the handbook only introduces conceptually |

### Installing and checking {#安装与检查}

```bash
nvidia-smi          # seeing the GPU and the driver version means the driver is fine
nvcc --version      # the CUDA compiler; install the CUDA Toolkit if it is missing
```

Download and install the CUDA Toolkit from NVIDIA. **CUDA 13 dropped support for architectures below sm_75** (Maxwell, Pascal, Volta), so use CUDA 12.x on an older card like a V100.

Compiling and running a program:

```bash
nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add
./vector_add
```

Write your own GPU's architecture after `-arch`, or simply `-arch=native` to let nvcc detect it.

### The example code {#示例代码包}

Every complete program in the text is packaged in [cuda-examples.tar.gz](assets/cuda-examples.tar.gz) with a Makefile:

```bash
tar xzf cuda-examples.tar.gz && cd cuda-examples
make -j8      # build everything
make run      # run them one by one
```

Each program compares the GPU result against a CPU reference and prints `PASS` or `FAIL`, or `SKIP` when the architecture does not meet its requirements.

!!! note "How the code was verified"
    The machine this handbook was written on has no GPU, so the code was verified like this:

    - **Compilation**: all 37 CUDA programs really compile with **both nvcc 12.9 and 13.4** (for whichever architecture each needs, including sm_80, sm_90 and sm_90a), and the PyTorch extensions compile against PyTorch 2.14's headers.
    - **Execution**: 30 of those programs really run on a home-made CUDA-to-CPU simulator and match the CPU reference. The simulator turns each CUDA thread into a coroutine, and `__syncthreads()`, warp shuffles and ballots are real barriers, so it catches index errors, missing synchronization and deadlocks from a barrier inside a branch; it does not simulate performance and cannot find data races. The 7 programs that use PTX, TMA, thread-block clusters, NCCL or cuBLAS directly are only compile-checked (the CuTe layout example among them really runs on the host, and the output on the page is what it printed).
    - **Triton**: the 4 Triton examples really run in interpreter mode.

    So the first time you run these on a real GPU, watch the `PASS` / `FAIL` each program prints. The performance numbers in the text only cite NVIDIA's official specifications and public material, with no invented measurements: the numbers you measure on your own GPU are the valuable ones.
