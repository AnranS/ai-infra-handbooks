# 作品集项目

<p class="lead">简历上的"熟悉 CUDA"没有说服力，"我写的 SGEMM 在 A100 上达到 cuBLAS 的 85%，这是优化过程和 Nsight 分析"才有。下面 6 个项目按难度排列，每个都能直接写进简历、在面试中深入讨论。做完前三个，基础岗位的面试就有底气了；做完后三个，可以冲击推理优化和算子开发的核心岗位。</p>

!!! tip "怎么做项目才有价值"
    - **有基线、有数据**：每个项目都和一个公认的基线比较（cuBLAS、PyTorch、FlashAttention、vLLM 现有实现），给出多个形状、多种 GPU 上的结果，说明占理论上限的百分比。
    - **有分析过程**：每一步优化都附上 Nsight Compute 的关键指标变化，说明为什么这样改。
    - **正确性有保障**：有测试，覆盖边界情况。
    - **写成文章**：GitHub README 或博客，结构是"问题 → 分析 → 逐步优化 → 结果 → 局限"。这篇文章就是你在面试中讲述的底稿。

## 项目一：SGEMM 优化全记录

**难度**：★★☆　**相关章节**：[GEMM](../kernels/gemm.md)、[性能分析](../tools/profiling.md)、[cp.async](../advanced/async-hopper.md)

从朴素实现开始，逐步实现共享内存分块、二维寄存器分块、向量化、双缓冲/cp.async 流水、warp 分块、自动调参，最终在一种 GPU 上达到 cuBLAS FP32 性能的 80% 以上。

**交付物**
- 每个版本一个 kernel，统一的测试和计时框架；
- 一张性能演进图（横轴版本，纵轴 TFLOPS 和占 cuBLAS 的百分比）；
- 每一步的 Nsight Compute 分析：主要停顿原因、共享内存/显存吞吐、占用率的变化。

**进阶**：换成 FP16/BF16 输入，用 WMMA 或 mma.sync 实现 Tensor Core 版本，与 cuBLAS 的 HGEMM 比较。

## 项目二：Transformer 常用融合算子库

**难度**：★★☆　**相关章节**：[Softmax 与归一化](../kernels/softmax-norm.md)、[Triton](../tools/triton.md)、[PyTorch 扩展](../tools/ecosystem.md#把-kernel-接入-pytorch)

用 CUDA 实现一组融合算子，注册为 PyTorch 自定义算子：fused_add_rms_norm、RoPE（旋转位置编码，原地作用于 Q、K）、SwiGLU 激活（`silu(x1) * x2`）、带 mask 的 softmax。每个算子同时写一个 Triton 版本。

**交付物**
- `torch.ops` 形式的算子，支持 FP16/BF16，有完整的 pytest 测试；
- 与 PyTorch 原生实现、`torch.compile` 生成的版本、Triton 版本的对比，报告有效带宽占峰值的比例；
- 在一个小模型（如 Qwen 或 LLaMA 的小尺寸版本）的推理中替换这些算子，测端到端的加速。

## 项目三：FlashAttention 前向（CUDA + Triton）

**难度**：★★★　**相关章节**：[FlashAttention](../advanced/attention.md)、[Tensor Core](../advanced/tensor-core.md)

从本手册的教学版出发，实现一个使用 Tensor Core 的 FlashAttention-2 风格前向 kernel：每个 warp 负责 16 行 Q，用 mma.sync 计算 $QK^\top$ 和 $PV$，在寄存器 fragment 上完成 online softmax；支持因果掩码和 head_dim 64/128。同时用 Triton 实现一版。

**交付物**
- 与官方 FlashAttention-2 和 PyTorch SDPA 在不同序列长度下的对比；
- 讲清楚 fragment 布局、online softmax 在寄存器上的实现、共享内存 swizzle；
- 正确性测试覆盖非对齐的序列长度、因果/非因果。

**进阶**：实现反向传播；或者实现 Flash-Decoding（split-K 的 decode 注意力）。

## 项目四：分页 decode 注意力 kernel

**难度**：★★★　**相关章节**：[PagedAttention](../advanced/attention.md#pagedattention)

在本手册 `paged_decode.cu` 的基础上，实现一个接近生产水平的 decode 注意力：BF16/FP8 KV Cache、128 位向量化读取、GQA 下一个 block 处理共享 KV 头的全部 query 头、online softmax 边扫描边累加、split-K 支持长序列。接入 vLLM 或 SGLang 的 attention 后端接口（或者至少用与它们相同的 KV Cache 布局）。

**交付物**
- 与 FlashInfer、vLLM 现有 kernel 在不同 batch 与上下文长度下的对比（带宽利用率）；
- 分析 GQA 比例、split 数、页大小对性能的影响。

## 项目五：量化 GEMV / GEMM

**难度**：★★★☆　**相关章节**：[量化与 GEMV](../advanced/quantization.md)

实现 W4A16（按组量化，group size 128）的 GEMV 与小 batch GEMM：离线权重重排、寄存器中的快速反量化（位技巧）、Tensor Core 计算。在 batch = 1 到 64 的范围内与 BF16 cuBLAS、Marlin 比较。

**交付物**
- 不同 batch 下的加速比曲线，并解释拐点；
- 分开报告量化误差与 kernel 实现误差；
- 讲清楚权重打包格式与反量化的位操作。

## 项目六：为开源推理引擎做贡献

**难度**：★★★★　**相关章节**：全部

在 vLLM、SGLang、FlashInfer、TensorRT-LLM、DeepGEMM 等项目里找一个具体的性能问题或缺失的 kernel（新模型的特殊算子、某种形状下的性能退化、新硬件的支持），提交 PR 并被合入。

**怎么开始**
- 读项目的 issue，筛选带 performance、kernel、good first issue 标签的问题；
- 先跑通项目的 benchmark 脚本，用 nsys 找出一个具体 workload 中最耗时的 kernel；
- 从小的改进开始（一个融合、一个参数调整、一个边界情况的修复），熟悉代码结构和评审流程。

被合入的 PR 是最有说服力的简历项目：它证明你的代码达到了工业级标准，也说明你能在大型代码库中工作。

## 简历怎么写

反例：
> 熟悉 CUDA 编程，了解 GPU 架构，做过 GEMM 优化。

正例：
> 实现 FP32 SGEMM（共享内存/寄存器分块、cp.async 三级流水、warp 分块），4096 规模下在 A100 上达到 cuBLAS 的 86%；用 Nsight Compute 定位并消除共享内存 bank 冲突（冲突数下降 97%）。代码与分析：github.com/xxx

要素：**做了什么（技术点）+ 结果（数字 + 基线）+ 怎么验证的（工具）+ 链接**。
