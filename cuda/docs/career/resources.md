# 学习资源

<p class="lead">这份手册覆盖了面试和工作的主干知识。要继续深入，下面是筛选过的资料：官方文档是权威参考，几本书和课程帮你建立系统的理解，开源代码是最好的进阶教材。</p>

## 官方文档

| 文档 | 什么时候看 |
| --- | --- |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 编程模型、各项特性（包括 TMA、集群、异步拷贝）的权威说明，附录里有各架构的规格表 |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 性能优化的系统性指南 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | mma、ldmatrix、wgmma、cp.async、mbarrier 等指令的精确语义和寄存器布局 |
| [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/) | 指标的含义、Profiling Guide 里的硬件模型 |
| [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) | 每个 API 的参数和行为 |
| 各代架构白皮书（Ampere、Hopper、Blackwell） | 硬件细节和规格，NVIDIA 官网可以下载 |

## 书

| 书 | 说明 |
| --- | --- |
| *Programming Massively Parallel Processors*（Hwu、Kirk、El Hajj，第 4 版） | GPU 编程的经典教材，归约、扫描、卷积、稀疏矩阵等并行模式讲得非常系统。有中文版《大规模并行处理器编程实战》 |
| *CUDA C Programming Guide / Professional CUDA C Programming* | 入门到中级的实践书，部分内容偏老，适合快速过一遍 |

## 课程与视频

- **GPU MODE**（YouTube 频道和 Discord 社区）：大量高质量的讲座，覆盖 CUDA、Triton、CUTLASS、FlashAttention、量化、推理引擎，讲者多是一线工程师。PMPP 的作者也做过配套讲解。强烈推荐。
- **NVIDIA GTC 的技术讲座**：每年都有关于 Hopper/Blackwell 编程、CUTLASS、Nsight 的讲座，可以在 NVIDIA On-Demand 免费观看。
- 各大学的并行计算课程（如 CMU 15-418/618）讲清楚了并行计算的基本原理。

## 必读的文章和论文

- Simon Boehm，*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*：SGEMM 逐步优化的经典博客，本手册 GEMM 一章的思路与它一致；
- Mark Harris，*Optimizing Parallel Reduction in CUDA*：归约优化的经典幻灯片（注意其中 warp 同步的写法已过时）；
- NVIDIA 开发者博客：*An Efficient Matrix Transpose in CUDA C/C++*、*How to Access Global Memory Efficiently*、*Using Shared Memory in CUDA C/C++*、*CUDA Refresher* 系列等；
- FlashAttention 系列论文（Dao et al., 2022；Dao, 2023；Shah et al., 2024）；
- *Efficient Memory Management for Large Language Model Serving with PagedAttention*（vLLM，SOSP 2023）；
- *SGLang: Efficient Execution of Structured Language Model Programs*（RadixAttention）；
- *Online normalizer calculation for softmax*（Milakov & Gimelshein, 2018）；
- *Single-pass Parallel Prefix Scan with Decoupled Look-back*（Merrill & Garland, 2016）；
- Colfax Research 的 CUTLASS / Hopper 系列教程（wgmma、TMA、warp 专门化、FlashAttention-3 的实现细节）；
- DeepSeek-V3 技术报告中关于 FP8 训练和推理部署的部分。

## 值得读的开源代码

| 项目 | 学什么 |
| --- | --- |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | CuTe 教程与示例、Hopper/Blackwell GEMM，工业级 kernel 的设计方式 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FA2（Ampere）和 FA3（Hopper）的实现 |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | 面向推理的注意力、采样、MoE kernel，代码组织清晰 |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | `csrc/` 与 `sgl-kernel/` 下的 CUDA 算子，以及 Triton kernel |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) / [DeepEP](https://github.com/deepseek-ai/DeepEP) / [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 精简而高性能的 Hopper FP8 GEMM、MoE 通信、MLA decode kernel |
| [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | 以教学友好为目标的 tile 级 kernel 库 |
| [Triton](https://github.com/triton-lang/triton) | 官方教程（tutorials 目录）从向量加法到 FlashAttention |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | 用 Triton 写的训练融合算子，代码短小易读 |

## 练习平台

- [LeetGPU](https://leetgpu.com/)：在浏览器里写 CUDA/Triton，在真实 GPU 上运行，有题库和排行榜；
- Tensara 等 GPU 编程竞赛平台，以及 GPU MODE 社区组织的 kernel 竞赛；
- 本手册的[示例代码包](../assets/cuda-examples.tar.gz)：每个程序都自带正确性检查，可以在它们的基础上改写和优化。

## 保持更新

GPU 硬件和软件栈的更新速度非常快。关注 NVIDIA 开发者博客、CUTLASS 和 PyTorch 的发布说明、vLLM/SGLang 的博客与路线图，以及 GPU MODE 社区的讨论，是跟上变化最有效的方式。
