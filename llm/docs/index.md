# 大模型原理手册

<p class="lead">做推理优化之前，先要真正懂模型在算什么：一段文字怎么变成数字，每一层做了哪些矩阵运算，下一个 token 是怎么选出来的，KV Cache 为什么成立，一个请求要多少算力和显存。这份手册把这些知识从头串起来，目标是让你在看 vLLM、SGLang 源码或者 CUDA kernel 时，清楚地知道每一行代码对应模型里的哪一步。</p>

## 为什么要学这些

推理优化是三件事的交叉：

- **模型在算什么**：Transformer 的每一层有哪些矩阵乘、注意力怎么计算、各部分的形状和参数量。这是本手册的内容；
- **硬件怎么执行**：GPU 的内存层次、kernel 怎么写得快。见 [CUDA 进阶手册](cuda://)；
- **系统怎么调度**：批处理、KV Cache 管理、并行策略。本手册最后一部分会衔接到这里，之后就可以直接读 vLLM、SGLang 的代码。

只懂 kernel 不懂模型，会不知道该优化什么；只懂模型不懂系统，会不知道瓶颈在哪里。这份手册负责第一块，并把它和另外两块接上。

## 学完能做到

- 从零写出一个 LLaMA/Qwen 结构的模型，加载真实权重，输出和 Hugging Face 官方实现一致；
- 讲清楚注意力、RoPE、RMSNorm、SwiGLU、GQA、MLA、MoE 的原理、形状和参数量；
- 手算任意模型的参数量、每个 token 的计算量、权重和 KV Cache 占多少显存，估算 decode 的延迟下限；
- 理解采样策略、KV Cache、prefill 与 decode、量化、连续批处理、前缀缓存、投机解码、PD 分离，知道每一项优化作用在模型的哪个环节；
- 对着一个新模型的 `config.json`，说出它的架构特点，以及推理时需要注意什么。

## 学习路线

各本手册合在一起的逐章路线（17 周，与冲刺计划逐周对应、每章是必学还是选学、不同岗位方向的重点、跨书的知识依赖）见[学习路线图](root://roadmap/)。下面是本书内部的顺序。

<div class="roadmap" markdown>

| 阶段 | 章节 | 目标 | 建议用时 |
| --- | --- | --- | --- |
| 一、基础 | [语言模型](basics/language-model.md) · [数学与 PyTorch](basics/math-torch.md) · [分词](basics/tokenization.md) | 建立"大模型就是一个条件概率函数"的整体认识 | 3-4 天 |
| 二、Transformer 解剖 | [嵌入](transformer/embedding.md) · [注意力](transformer/attention.md) · [RoPE](transformer/position.md) · [归一化](transformer/norm-residual.md) · [FFN](transformer/ffn.md) · [组装模型](transformer/build-llm.md) | 每个组件都能写出来，拼成一个能加载真实权重的模型 | 1.5 周 |
| 三、架构演进 | [MQA/GQA/MLA](transformer/attention-variants.md) · [MoE](transformer/moe.md) | 理解主流大模型为什么长这样 | 4-5 天 |
| 四、训练与对齐 | [预训练](training/pretraining.md) · [后训练](training/post-training.md) | 知道模型从哪来，够用即可 | 3 天 |
| 五、推理原理 | [解码](inference/decoding.md) · [KV Cache](inference/kv-cache.md) · [估算](inference/estimation.md) · [量化](inference/quantization.md) · [推理服务](inference/serving.md) | 掌握推理优化的全部核心概念 | 1.5 周 |
| 六、融会贯通 | [一个 token 的旅程](synthesis/token-journey.md) · [模型巡礼](synthesis/models.md) · [自测](synthesis/quiz.md) | 把所有知识串成一条线 | 随时回顾 |

</div>

!!! tip "数学用到再查"
    线性代数与低秩、概率与采样、信息论、反向传播、浮点误差、屋顶线与排队论单独成了一本[数学基础手册](math://)。本书正文用到这些数学时会链接过去，不需要先通读；想先摸底，可以做它的[自测题库](math://quiz/)。

## 贯穿全书的主线

- **一个真实的模型**：全书以 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 为例。它只有 6 亿参数，CPU 上就能跑，但结构和同系列 8B、32B 的大模型完全一样（GQA、QK-Norm、RoPE、RMSNorm、SwiGLU）。更新的 Qwen3.5 全系列改用了线性注意力混合架构，在[推理系统手册的线性注意力](serving://frontier/linear-attn/)一章里讲。
- **一份自己写的实现**：在[从零组装一个大模型](transformer/build-llm.md)一章里写出 `mini_llm.py`（约 200 行），加载真实权重，逐项和 Hugging Face transformers 的输出对比。之后的章节都在它上面做实验。
- **"推理视角"提示框**：每一章都有这样的提示框，说明这部分知识在推理优化中对应什么：

!!! inference "推理视角"
    比如：注意力一章会告诉你，为什么 decode 阶段注意力是访存瓶颈；FFN 一章会告诉你，为什么 SwiGLU 的三个矩阵常被融合成两个 GEMM。

## 准备环境

全部代码在 CPU 上就能运行，不需要 GPU：

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers tokenizers safetensors numpy
```

下载示例模型（约 1.5 GB）：

```py
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", local_dir="models/Qwen3-0.6B")
```

访问 Hugging Face 不方便时，可以设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像，或者从 ModelScope 下载同名模型。书中的代码默认模型放在当前目录的 `models/Qwen3-0.6B` 下。书中定义的模块文件（`mini_llm.py` 等）打包在 [llm-code.tar.gz](assets/llm-code.tar.gz) 里。

!!! note "关于代码的验证"
    所有 `python` 代码块和 `>>>` 交互示例都在 PyTorch 2.14（CPU）和 transformers 5.17 下实际运行过，交互示例的输出用 doctest 逐字核对。自己实现的模型与 transformers 官方实现的输出做了逐项对比（logits 差异在 1e-4 量级，贪心生成的 token 完全一致）；书中列出的知名模型参数量，是用官方配置实际构建模型计算出来的。
