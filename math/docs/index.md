# 数学基础手册

<p class="lead">读大模型和推理系统时用到的数学，单独成一本，用到再查：线性代数与低秩、概率与采样、信息论、反向传播与二阶信息、浮点误差、屋顶线与排队论。每个结论都在真实的 Qwen3-0.6B 上测量验证。不需要先通读：其他手册里用到这些数学的地方都链接到了这里，读到哪里卡住，就来查对应的那一节。</p>

## 怎么用这本书

两种用法：

- **按需查**：在别的手册里读到下表左边的内容、对某个推导没把握时，点进右边那一节，读完就回去。
- **先摸底**：做一遍[自测题库](quiz.md)，8 道题覆盖全书，答不上来的再读对应章节。

| 读到 | 需要的数学 | 查这一节 |
| --- | --- | --- |
| [注意力机制](llm://transformer/attention/) | 点积就是投影，矩阵就是对空间做的变换 | [线性代数：先看几何](linear-algebra.md#先看几何矩阵是对空间做的一件事) |
| [RoPE](llm://transformer/position/)、QuaRot 这类旋转量化 | 正交矩阵保持长度和点积 | [线性代数：正交矩阵与旋转](linear-algebra.md#正交矩阵与旋转) |
| [张量并行](serving://distributed/tensor-parallel/)、GEMM 的 split-K | 矩阵乘法按行、按列、按 k 切分 | [线性代数：矩阵乘法的三种视角](linear-algebra.md#矩阵乘法的三种视角) |
| [LoRA](llm://training/post-training/)、[MLA](llm://transformer/attention-variants/) | 秩、奇异值与低秩近似 | [线性代数：秩与低秩近似](linear-algebra.md#秩与低秩近似) |
| [解码与采样](llm://inference/decoding/) | 温度、top-k、top-p 对分布做了什么 | [概率与采样](probability.md#先看形状温度top-ktop-p-各自在做什么) |
| [投机解码](serving://topics/speculative/) | 拒绝采样为什么不改变输出分布，接受率怎么算 | [概率与采样：拒绝采样与投机解码](probability.md#拒绝采样与投机解码) |
| [语言模型的训练目标](llm://basics/language-model/) | 熵、交叉熵、困惑度 | [信息论：熵、交叉熵与 KL 散度](information-theory.md#熵交叉熵与-kl-散度) |
| [量化原理](llm://inference/quantization/) | 量化前后分布差多少、误差从哪来 | [信息论：KL 散度](information-theory.md#kl-散度量化改变了多少)、[浮点：量化噪声](floating-point.md#量化噪声每比特-6-db)、[微积分：二阶信息与 GPTQ](calculus.md#二阶信息gptq) |
| [预训练](llm://training/pretraining/)、[从零训练](train://scratch/model/) | 链式法则与反向传播 | [微积分与反向传播](calculus.md) |
| [归一化与残差流](llm://transformer/norm-residual/)、BF16 与 FP8 | 舍入、累加误差、溢出 | [浮点与数值计算](floating-point.md) |
| [参数量、算力与显存估算](llm://inference/estimation/)、[Profiling](serving://perf/profiling/) | 算术强度与屋顶线 | [性能数学：FLOPs、字节与算术强度](performance-math.md#flops字节与算术强度) |
| [压测、SLO 与容量规划](serving://perf/benchmark/) | Little 定律、排队论、尾延迟、测量统计 | [性能数学：排队论](performance-math.md#排队论为什么接近满载时延迟爆炸)、[测量的统计学](performance-math.md#性能测量的统计学) |

## 章节

<div class="roadmap" markdown>

| 部分 | 章节 | 讲什么 |
| --- | --- | --- |
| 模型里的数学 | [线性代数](linear-algebra.md) | 矩阵的几何意义、矩阵乘法的三种切分、SVD 与低秩（为什么权重不是低秩、K 和 V 却是）、正交矩阵与旋转 |
| | [概率与采样](probability.md) | 温度与 top-k / top-p、采样算法、蒙特卡洛的误差、序列的概率、拒绝采样与投机解码、重要性采样 |
| | [信息论](information-theory.md) | 熵、交叉熵与 KL 散度，模型在哪些位置有把握，用 KL 衡量量化的影响，正向与反向 KL |
| | [微积分与反向传播](calculus.md) | 链式法则、反向模式自动微分、线性层的反向传播、GPTQ 用到的二阶信息 |
| 数值与性能 | [浮点与数值计算](floating-point.md) | 浮点格式、舍入与累加误差、抵消与溢出、每比特 6 dB 的量化噪声 |
| | [性能与服务中的数学](performance-math.md) | 算术强度与屋顶线、Amdahl 定律、Little 定律、排队论、尾延迟放大、性能测量的统计学 |
| 自测 | [自测题库](quiz.md) | 8 道题覆盖全书，每题指向对应的一节 |

</div>

## 代码与运行

例子和[大模型原理手册](llm://)共用一套环境：自己实现的 `mini_llm` 加上真实的 Qwen3-0.6B 权重，全部在 CPU 上运行。代码按在仓库的 `llm/` 目录下运行来写（`models/Qwen3-0.6B`、冻结的样本文本 `docs/assets/sample-passage.txt` 都是相对 `llm/` 的路径），环境准备见大模型原理手册首页的"准备环境"。

核对全书的例子，输出与页面逐行比对：

```bash
llm/.venv-llm/bin/python math/tools/check_code.py                         # 全部 6 章
llm/.venv-llm/bin/python math/tools/check_code.py math/docs/probability.md   # 只核对一章
```
