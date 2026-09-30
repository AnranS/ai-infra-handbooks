# CUDA 进阶手册

<p class="lead">面向 AI Infra、大模型推理优化、GPU 算子开发岗位的 CUDA 学习路线。从 GPU 架构讲起，一路写到手写 GEMM、Softmax、FlashAttention、量化 GEMV，再到 Tensor Core、Hopper 新特性、Nsight 性能分析和多卡通信。最后是面试题库和作品集项目。</p>

## 这份手册适合谁

- 会 C++：指针、数组、模板和类的基本用法没问题。
- 知道矩阵乘法、softmax 是什么，最好用过 PyTorch。
- 想转向或者深入 GPU 编程，目标是能通过 AI Infra / 推理优化 / 算子开发岗位的面试，并在工作中独立写和调优 kernel。

不需要有并行编程或 GPU 的经验。

## 目标岗位需要什么

| 岗位方向 | 日常工作 | 面试重点 |
| --- | --- | --- |
| 推理优化 / AI Infra | 优化 vLLM、SGLang、TensorRT-LLM 这类推理引擎；写融合算子、量化算子、attention kernel；分析端到端延迟和吞吐 | CUDA 基础、手写 kernel、FlashAttention / PagedAttention 原理、量化、Nsight 分析、推理引擎架构 |
| 算子开发 / GPU 性能优化 | 为新模型、新硬件写高性能算子（GEMM、attention、MoE、通信计算融合），对标 cuBLAS 和社区最优实现 | 访存与计算优化的全套手段、Tensor Core、CUTLASS/CuTe 或 Triton、Roofline 分析 |
| 训练框架 / 分布式 | 训练加速、通信优化、显存优化 | CUDA 基础、流与并发、NCCL 与集合通信、混合精度 |
| HPC / 科学计算 | 数值模拟、求解器加速 | CUDA 基础、访存优化、归约与扫描等并行模式、多 GPU |

几乎所有方向的面试都会考三件事：**讲清楚 GPU 的执行和内存模型**、**现场手写一个 kernel 并逐步优化**（归约、转置、softmax、GEMM 最常见）、**用数据说明一个 kernel 为什么慢**（Roofline、带宽利用率、Nsight 指标）。这份手册的结构就是围绕这三件事设计的。

## 学完能做到

- 说清楚 SM、warp、线程块、寄存器、共享内存、L2、HBM 之间的关系，并据此解释任何一个 kernel 的性能。
- 独立写出并优化归约、转置、softmax、LayerNorm、GEMM，GEMM 在 FP32 上达到 cuBLAS 性能的七到八成以上。
- 用 WMMA / mma.sync 调用 Tensor Core，理解 cp.async、TMA、wgmma 等新硬件特性的用途。
- 讲清楚 FlashAttention 为什么快，写出一个能跑通的 FlashAttention 前向 kernel。
- 用 Nsight Systems 和 Nsight Compute 定位瓶颈，给出有数据支撑的优化方案。
- 用 Triton 快速写出高性能算子，并把自定义算子接入 PyTorch。

## 学习路线

八本手册合在一起的逐章路线（17 周，与冲刺计划逐周对应、每章是必学还是选学、不同岗位方向的重点、跨书的知识依赖）见[学习路线图](root://roadmap/)。下面是本书内部的顺序。

<div class="roadmap" markdown>

| 阶段 | 章节 | 目标 | 建议用时 |
| --- | --- | --- | --- |
| 一、基础 | [GPU 架构](basics/gpu-architecture.md) · [第一个程序](basics/first-kernel.md) · [内存层次](basics/memory.md) · [执行模型](basics/execution.md) · [同步与 warp 编程](basics/sync-warp.md) | 能写正确的 kernel，能解释性能的来龙去脉 | 3 周 |
| 二、经典算子 | [归约](kernels/reduction.md) · [转置](kernels/transpose.md) · [GEMM](kernels/gemm.md) · [Softmax 与归一化](kernels/softmax-norm.md) · [前缀和](kernels/scan.md) | 面试手写题全部能写出来并逐步优化 | 4 周 |
| 三、工具 | [Nsight](tools/profiling.md) · [流与 CUDA Graphs](tools/streams.md) · [PDL 与 megakernel](tools/pdl-megakernel.md) · [Triton](tools/triton.md) | 会分析、会系统级优化、会用 Triton 提效 | 2 周 |
| 四、现代 GPU 与 AI 算子 | [Tensor Core](advanced/tensor-core.md) · [Hopper/Blackwell](advanced/async-hopper.md) · [CuTe 布局代数](advanced/cute-layout.md) · [FlashAttention](advanced/attention.md) · [量化与 GEMV](advanced/quantization.md) | 能读懂并改写推理引擎里的核心 kernel | 4 周 |
| 五、框架与编译器 | [张量的内存模型](framework/tensor.md) · [autograd](framework/autograd.md) · [dispatcher 与自定义算子](framework/dispatcher.md) · [CUDA 运行时](framework/cuda-runtime.md) · [torch.compile](framework/compile.md) · [AI 编译器全景](framework/compilers.md) | 看懂 PyTorch 在 kernel 之上做了什么，正确地注册自定义算子，用好 torch.compile | 1～2 周 |
| 六、工程与求职 | [多 GPU](tools/multi-gpu.md) · [生态](tools/ecosystem.md) · [面试题库](career/interview.md) · [作品集](career/projects.md) | 有拿得出手的项目，面试对答如流 | 3 周以上 |

</div>

工具阶段（Nsight）建议在学完 GEMM 之后就开始穿插使用，不必等到最后。

## 怎么学

1. **每个 kernel 都自己写一遍。** 先不看答案写出能跑的版本，再对照正文逐步优化。只看不写，面试时一定写不出来。
2. **每一步优化都要测。** 记下每个版本的耗时、带宽或算力，算出占硬件峰值的百分比。说不出数字的优化等于没做。
3. **用 Nsight Compute 验证你的猜测。** "我觉得是 bank conflict"不算结论，看到指标才算。
4. **写学习笔记或博客。** 把每个算子的优化过程写成文章，它就是你简历上的作品，也是面试时最好的谈资。

## 准备 GPU 环境

这份手册的代码需要 NVIDIA GPU 才能运行。常见的获取方式：

| 方式 | 说明 |
| --- | --- |
| 公司或学校的 GPU 开发机 | 最方便，通常是 A100/H100/H20/L20 等 |
| 按小时租的云 GPU | AutoDL、各大云厂商都有，RTX 4090 或 A100 每小时几元到十几元，适合集中练习 |
| Google Colab | 免费版提供 T4（sm_75），足够学完基础和经典算子部分 |
| [LeetGPU](https://leetgpu.com/) | 在浏览器里写 CUDA 并在真实 GPU 上运行，有题库，适合刷题 |

**在 Mac 上**：没有 NVIDIA GPU，但可以先用仓库里的 CUDA→CPU 模拟器检查 kernel 的正确性（不测性能）：`python tools/emu_run.py reduction.cu` 运行书里的例子，`python tools/emu_run.py 你的文件.cu` 运行自己写的 kernel；练习题的本地判题在 Mac 上也会自动改用模拟器。测性能的实验攒到有 GPU 的时候集中做，完整说明见[学习环境](root://setup/)。

不同章节对硬件的要求：

| GPU | 架构 | 可以学的内容 |
| --- | --- | --- |
| T4、RTX 20 系列 | Turing，sm_75 | 基础、经典算子、WMMA（FP16 Tensor Core） |
| A100、RTX 30/40 系列、L4、L20 | Ampere / Ada，sm_80、sm_86、sm_89 | 以上全部，加上 cp.async、BF16、mma.sync m16n8k16、FlashAttention 等绝大部分内容 |
| H100、H800、H20 | Hopper，sm_90 | 以上全部，加上 TMA、线程块集群、wgmma、FP8 |
| B200、RTX 50 系列 | Blackwell，sm_100、sm_120 | 最新特性（tcgen05、FP4），手册只做概念介绍 |

### 安装与检查

```bash
nvidia-smi          # 能看到 GPU 和驱动版本，说明驱动正常
nvcc --version      # CUDA 编译器，没有的话安装 CUDA Toolkit
```

CUDA Toolkit 从 NVIDIA 官网下载安装。**CUDA 13 起不再支持 sm_75 以下的架构**（Maxwell、Pascal、Volta），在 V100 这类老卡上请用 CUDA 12.x。

编译并运行一个程序：

```bash
nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add
./vector_add
```

`-arch` 写你的 GPU 架构，或者直接写 `-arch=native` 让 nvcc 自动识别本机的 GPU。

### 示例代码包

正文里所有完整程序都打包在 [cuda-examples.tar.gz](assets/cuda-examples.tar.gz) 里，附带 Makefile：

```bash
tar xzf cuda-examples.tar.gz && cd cuda-examples
make -j8      # 编译全部
make run      # 逐个运行
```

每个程序都会把 GPU 结果和 CPU 参考结果比较，打印 `PASS` 或 `FAIL`；硬件架构不满足要求时打印 `SKIP`。

!!! note "关于代码的验证"
    写这份手册的机器没有 GPU，代码按以下方式验证：

    - **编译**：全部 37 个 CUDA 程序都用 **nvcc 12.9 和 13.4 两个版本**实际编译通过（按各自要求的架构，包括 sm_80、sm_90、sm_90a），PyTorch 扩展按 PyTorch 2.14 的头文件编译通过。
    - **执行**：其中 30 个程序在一个自制的 CUDA→CPU 模拟器上实际运行，并与 CPU 参考结果比对通过。模拟器把每个 CUDA 线程变成一个协程，`__syncthreads()`、warp shuffle/ballot 都是真实的屏障，能发现下标错误、同步遗漏和分支内屏障导致的死锁；它不模拟性能，也发现不了数据竞争。直接使用 PTX、TMA、线程块集群、NCCL、cuBLAS 的 7 个程序只做了编译检查（其中 CuTe 的布局示例在主机上实际运行过，页面上的输出就是运行结果）。
    - **Triton**：4 个 Triton 示例在解释器模式下实际运行通过。

    所以第一次在真实 GPU 上运行时，请留意每个程序打印的 `PASS` / `FAIL`。正文中的性能数字只引用 NVIDIA 官方规格和公开资料，没有编造的测试结果：你在自己 GPU 上测出来的数字才是最有价值的。
