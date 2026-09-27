# 学习资源

<p class="lead">这份手册是一张"地图"。想在某个方向上走得更深，就需要读原始论文和源码。下面按章节列出最值得读的材料，每一项都说明为什么值得读、读的时候关注什么；最后给出学完本手册之后，走向推理优化的下一步。</p>

## 读论文的方法

- **先看图和表，再看公式，最后看正文**。架构论文的核心往往就是一张结构图和一张消融表。
- **带着问题读**：这个设计解决了什么瓶颈？付出了什么代价？（对照[一个 token 的完整旅程](token-journey.md#把整本手册串起来)里的因果图。）
- **能复现的就复现**：本手册的每个概念都有一段可运行的代码，读论文时也可以写一个最小实现，和论文中的数字对一对。

## 按章节

### 基础与 Transformer

| 材料 | 为什么读 |
| --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762)（2017） | Transformer 的原始论文。注意编码器-解码器结构、Post-Norm、正弦位置编码，这些都已被后来的模型改掉了，想想为什么 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图解，适合第一遍建立直觉 |
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)（BPE） | BPE 用于 NLP 的起点 |
| [karpathy/minbpe](https://github.com/karpathy/minbpe)、[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)、[karpathy/nanochat](https://github.com/karpathy/nanochat) | 最小可读的分词器、GPT 训练代码，以及从预训练到对话的完整小型流水线；配套视频也很值得看 |
| [RoFormer](https://arxiv.org/abs/2104.09864)（RoPE） | RoPE 的推导，读第 3 节即可 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | RMSNorm |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | SwiGLU 的来源，只有 3 页 |
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | 从"残差流"的视角理解 Transformer，和本手册的[残差流](../transformer/embedding.md#残差流)一节相呼应 |
| [transformers 的 modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | 把它和本手册的 `mini_llm.py` 逐行对照，是读懂所有 LLaMA 类模型实现的捷径 |

### 架构演进

| 材料 | 为什么读 |
| --- | --- |
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)（MQA） | 最早从"decode 访存受限"出发设计模型结构的论文之一 |
| [GQA](https://arxiv.org/abs/2305.13245) | GQA，以及如何把 MHA 的检查点转换成 GQA |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | MLA 的出处，第 2.1 节详细讲了低秩压缩、权重吸收和解耦 RoPE |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 细粒度 MoE、无辅助损失的负载均衡、MTP、FP8 训练，以及推理部署方案（第 3.4 节） |
| [Switch Transformers](https://arxiv.org/abs/2101.03961)、[Mixtral of Experts](https://arxiv.org/abs/2401.04088) | MoE 的路由、容量和负载均衡 |
| [LLaMA](https://arxiv.org/abs/2302.13971)、[Llama 2](https://arxiv.org/abs/2307.09288)、[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 标准模板的来源；Llama 3 的报告对数据、训练基础设施和推理（FP8、流水线并行）都有详细描述 |
| [Qwen2.5](https://arxiv.org/abs/2412.15115)、[Qwen3](https://arxiv.org/abs/2505.09388) 技术报告 | 本手册所用模型的"家谱" |
| [Gemma 2](https://arxiv.org/abs/2408.00118)、[Gemma 3](https://arxiv.org/abs/2503.19786) | 局部/全局注意力混合、软截断、QK-Norm |
| [gpt-oss 模型卡](https://arxiv.org/abs/2508.10925) | 注意力汇聚、交替的滑动窗口、MXFP4 |
| [YaRN](https://arxiv.org/abs/2309.00071) | 长上下文扩展中最常用的 RoPE 缩放方法 |

### 训练与对齐

| 材料 | 为什么读 |
| --- | --- |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)、[Chinchilla](https://arxiv.org/abs/2203.15556) | $6ND$、算力最优的参数与数据配比 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | SFT + RLHF 的经典流程 |
| [DPO](https://arxiv.org/abs/2305.18290) | 看懂损失函数的推导，就理解了偏好优化 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300)（GRPO）、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 推理模型是怎样用强化学习训练出来的；RL 训练中大量的采样正是推理引擎的新战场 |
| [LoRA](https://arxiv.org/abs/2106.09685) | 低秩微调，以及多 LoRA 服务的基础 |

### 推理原理与推理服务

| 材料 | 为什么读 |
| --- | --- |
| [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) | 推理估算的经典博客，和本手册的[估算](../inference/estimation.md)一章互为补充 |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Google DeepMind 的"屋顶线 + 并行"教程，推理一章尤其值得读，TPU 视角但原理通用 |
| [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)、[FlashAttention-3](https://arxiv.org/abs/2407.08608) | IO 感知的注意力；配合 CUDA 手册的 [FlashAttention 与推理算子](cuda://advanced/attention/) |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（vLLM） | PagedAttention 与 vLLM 的设计 |
| [SGLang](https://arxiv.org/abs/2312.07104) | RadixAttention 前缀缓存、结构化输出的压缩状态机 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | 分块 prefill 与"无停顿"的调度 |
| [DistServe](https://arxiv.org/abs/2401.09670)、[Mooncake](https://arxiv.org/abs/2407.00079) | PD 分离；Mooncake 是以 KV Cache 为中心的生产架构 |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)、[Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | 投机解码与投机采样的正确性证明 |
| [Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077) | 不需要独立草稿模型的投机解码 |
| [LLM.int8()](https://arxiv.org/abs/2208.07339)、[SmoothQuant](https://arxiv.org/abs/2211.10438)、[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978) | 离群值问题与主流量化方法 |
| [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)、[Massive Activations in LLMs](https://arxiv.org/abs/2402.17762) | 本手册在 Qwen 上观察到的注意力汇聚与巨大激活值 |
| [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | 为什么温度为 0 结果也会变、如何实现 batch 不变的 kernel |

## 学完之后：走向推理优化

本手册解决的是"模型在算什么"。接下来的两块：

**1. 硬件怎么执行：CUDA。** 见 [CUDA 进阶手册](cuda://)。其中与本手册衔接最紧密的章节：

- [GEMM 优化之路](cuda://kernels/gemm/)：所有线性层；
- [Softmax 与归一化](cuda://kernels/softmax-norm/)：RMSNorm、融合 kernel；
- [FlashAttention 与推理算子](cuda://advanced/attention/)：prefill 与分页 decode 注意力；
- [量化与 GEMV](cuda://advanced/quantization/)：INT4 权重的 decode；
- [流、并发与 CUDA Graphs](cuda://tools/streams/)、[多 GPU 与 NCCL](cuda://tools/multi-gpu/)。

**2. 系统怎么调度：读推理引擎的源码。** 建议的顺序：

1. [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)：用一千多行代码实现了 vLLM 的核心（分页 KV、连续批处理、前缀缓存、CUDA Graphs、张量并行），一两天就能读完，是理解完整引擎的最佳起点；
2. [vLLM](https://github.com/vllm-project/vllm)：从一个请求的生命周期读起（API 服务 → 引擎 → 调度器 → 模型执行器 → 采样器），再看 `model_executor/models/` 下你熟悉的模型实现，对照 `mini_llm.py`；
3. [SGLang](https://github.com/sgl-project/sglang)：重点看调度器（前缀缓存、overlap 调度）和 DeepSeek 相关的优化（MLA、DP Attention、EP）。

读源码时，把本手册的"逐站解读"表（见[一个 token 的完整旅程](token-journey.md#逐站解读)）当作索引：每遇到一个模块，就问它对应表中的哪一站、解决了哪个瓶颈。

**3. 动手做项目。** 几个由浅入深的方向：

- 给 `mini_llm.py` 加上分页 KV Cache 与连续批处理，写一个几百行的"迷你推理引擎"，测量吞吐随 batch 的变化；
- 在 GPU 上用 Triton 写 RMSNorm、SwiGLU、RoPE 的融合 kernel，替换 `mini_llm.py` 中对应的部分，对比速度与精度；
- 在 vLLM 或 SGLang 中实现（或复现）一个小特性：新的采样参数、一个新模型结构的支持、一个 kernel 的优化，并提交 PR。

## 小结

- [x] 读论文先看图表，带着"解决了什么瓶颈、代价是什么"的问题读。
- [x] 每一章都有对应的原始论文，推理部分的论文是后续深入的重点。
- [x] 下一步：CUDA 手册解决"硬件怎么执行"，nano-vllm → vLLM → SGLang 解决"系统怎么调度"。
