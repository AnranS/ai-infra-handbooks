# 第 0 步：一个能跑的最小推理引擎

<p class="lead">在写任何正式的模块之前，先用不到两百行、不 import <code>minisgl</code> 里任何东西的代码，把一个真正的推理引擎跑起来：读 safetensors、手写 Qwen3 的前向、用最朴素的方式存 KV cache、贪心解码一个提示词、和 Hugging Face 逐 token 对答案、量出每秒多少个 token。这就是推理引擎的全部骨架。后面二十几章没有一章在"从零开始"——它们都是在这一份代码上做加法：换掉某个朴素的部件，再量一次，看快了多少、多撑了多少请求。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一次"生成"由哪两种前向组成？它们的输入长度分别是多少？
    2. KV cache 存的是什么？不存它会怎样？
    3. Qwen3 的注意力和标准的多头注意力有哪两处不同？
    4. 同一份代码，decode 从 0.4 tokens/s 变成 25 tokens/s，改了哪一行？为什么？
    5. 这个最小引擎缺的东西里，哪一项最先在真实服务里出问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. prefill 一次：整段提示词一起算，输入长度是提示词的 token 数；之后每生成一个 token 做一次 decode，输入长度是 1。
    2. 每一层算过的 K 和 V（本书的模型是每个 token、每层 8 个头 × 128 维 × 2，fp32 下约 224 KB）。不存的话每一步都要把整段序列重新算一遍，第 n 步的代价和 n 成正比，总代价是平方级的。
    3. GQA（16 个 query 头共用 8 个 KV 头，`repeat_interleave` 把 K、V 复制给对应的 query 头）和 q_norm / k_norm（q、k 在旋转之前各过一次 RMSNorm）。
    4. `torch.set_num_threads(物理核数)`。decode 时矩阵乘的"批"只有 1 行，32 个线程之间同步的开销比计算本身大得多；线程数减半，decode 快了 68 倍。
    5. 一次只能服务一个请求。真实服务同时有几十上百个请求在飞，逐个处理意味着 GPU 的算力几乎全部闲置——这是阶段一和阶段二要解决的。

**本章要写的文件**：`examples/ch00_tiny_engine.py`，一个文件、不到两百行。它不进 `minisgl` 包，是整本书的起点和参照物。

## 先跑起来

```bash
cd minisgl && python examples/ch00_tiny_engine.py
```

第一次要等上一分钟——这本身就是本章要解释的第一个现象。输出：

@@output ch00_tiny_engine@@

三件事在这几行里都发生了：模型真的加载了（596M 参数）、生成的内容是对的（`' Paris. The capital of Italy is Rome. ...'`，与 Hugging Face `generate` 的 16 个 token **逐个相同**）、而且有了第一组数字——默认设置下 0.4 tokens/s，改一行之后 25.6 tokens/s。

## 这份代码在做什么

@@code examples/ch00_tiny_engine.py@@

整份代码分四段，对应推理引擎永远绕不开的四件事。

**1. 权重。** `config.json` 给出形状（28 层、hidden 1024、16 个 query 头、8 个 KV 头、head_dim 128），`model.safetensors` 给出数值。我们把所有张量读成一个 `dict`，bf16 转成 fp32——不是因为 fp32 更好，而是为了和 Hugging Face 的 fp32 结果**逐位**对齐，这是本书每一章验证正确性的基准。`tie_word_embeddings` 为真时输出层直接复用词嵌入。

**2. 四个算子。** `rmsnorm`、`rope`、`attention`、`mlp`，加起来三十行：

- RMSNorm 没有均值和偏置，只有一个缩放；
- RoPE 按"前后两半配对"旋转（Hugging Face 和 Qwen 的约定，Meta 原版 LLaMA 是相邻两维配对，见[大模型原理手册](llm://transformer/position/)）；
- 注意力里有 Qwen3 的两处特别：**GQA**（`repeat_interleave` 把 8 个 KV 头复制成 16 个，和 query 头一一对应）和 **q_norm / k_norm**（q、k 在旋转前各过一次 RMSNorm，防止注意力 logit 过大）。因果掩码用绝对位置算：本轮第 $t$ 个 query 的位置是 $S - T + t$，它只能看到位置不超过自己的 key；
- MLP 是 SwiGLU：`down(silu(gate(x)) * up(x))`。

**3. 一层和整个模型。** `layer` 是两次"归一化 → 子层 → 残差相加"。KV cache 是一个长度为 28 的 list，每一项是这一层到目前为止所有 token 的 `(k, v)`，每轮算完就 `torch.cat` 追加上去。`forward` 只返回**最后一个位置**的 logits：生成时只需要它。

**4. 生成。** `generate` 做一次 prefill（整段提示词一起算，5 个 token），然后 16 次 decode（每次只算 1 个新 token，位置接着往后数）。采样是最简单的贪心 `argmax`。两次计时分别量 prefill 和 decode——这两个数在整本书里会反复出现，因为它们的瓶颈完全不同：prefill 是矩阵乘密集的，decode 是访存密集的。

## 第一个优化：先量，再改

默认设置下 decode 是 0.4 tokens/s，每步 2.7 秒。模型只有 0.6B，CPU 做一步 decode 的计算量大约是 1.2 GFLOP，远不该这么慢。问题出在线程：PyTorch 默认用"逻辑核数"个线程（这台机器 32 个），而 decode 时矩阵乘的"批"只有 1 行，每个算子的工作量小到不够 32 个线程分，线程间同步的开销反而成了大头。`torch.set_num_threads(16)` 之后 decode 0.63 秒，**快了 68 倍**，输出一个 token 都没变。

这件事值得放在第 0 步，是因为它示范了本书对"优化"的全部态度：**先有一个能量的基线，改一处，再量一次，输出必须不变。** 后面每一章末尾的测试和基准都是这一句话的展开。顺便一提，正式的 `Engine`（第 6 章）在 CPU 上会自动把线程数设成"物理核数 / TP 数"，你就不用每次手改了。

!!! tip "在自己的机器上"
    线程数那一行用的是 `os.cpu_count() // 2`，它假设每个物理核有两个逻辑核（超线程）。Apple Silicon 没有超线程，`cpu_count()` 是性能核加能效核的总数，减半正好接近性能核数，一般也合适。真正的做法是两种都试一下，取快的那个——这就是"先量再改"。

## 它缺什么：整本书的路线图

这个引擎能生成正确的 token，但离"推理系统"还差得远。下面每一行都是它的一个朴素之处，以及本书在哪一章把它换掉。读完这张表，整本书的结构就清楚了。

| 这份代码的做法 | 问题 | 换成什么 | 在哪一章 |
| --- | --- | --- | --- |
| KV 用 `torch.cat` 每步重新拼一份 | 第 $n$ 步复制 $n$ 个 token 的 KV，总流量平方级；fp32 下每个 token 224 KB，1024 个 token 就是 224 MB，每步都搬一遍 | 预分配的分页 KV 池 + page table，新 token 只写自己那一页 | 阶段一 [KV 池](../compute/kvcache.md) |
| 一次只服务一个请求 | 计算单元几乎全部闲置：decode 一步的矩阵乘只有 1 行 | `Req` / `Batch`：多个请求的 token 拼成一个 batch 一起算 | 阶段一 [核心数据结构](../compute/core.md)，阶段二 [调度器](../schedule/scheduler.md) |
| 权重一次全读进 `dict`，按名字查表 | 大模型读不进内存；每一层的权重名散落在各处 | 流式加载 + 算子层：每个 `Layer` 自己管权重和加载 | 阶段一 [算子层](../compute/layers.md)、[模型与权重](../compute/models.md) |
| 注意力算完整的 $T \times S$ 分数矩阵 | prefill 4096 个 token 时，16 个头的分数矩阵占 1 GiB；掩码也是显式物化的 | 可插拔的注意力后端：CPU 参考实现、FlashInfer、FlashAttention | 阶段一 [注意力后端](../compute/attention.md)，阶段四 [GPU 注意力](../perf/gpu-attention.md) |
| 采样只有 `argmax` | 没有温度、top-k、top-p，没法按请求区别对待 | 批量采样器 | 阶段一 [Engine 与采样器](../compute/engine.md) |
| 两个请求共享的前缀算两遍 | 系统提示词、多轮对话的历史每次都重算 | Radix Cache：按前缀复用 KV | 阶段二 [Radix Cache](../schedule/radix-cache.md) |
| 长提示词一口气算完 | 一个 8K 的 prefill 会把所有正在 decode 的请求卡住几秒 | 分块 prefill | 阶段二 [分块 prefill](../schedule/chunked-prefill.md) |
| Python 调度和计算串行 | 每一步都在等 CPU 准备下一步 | 重叠调度（CPU 侧）、CUDA Graph（启动侧） | 阶段二 [重叠调度](../schedule/overlap.md)，阶段四 [CUDA Graph](../perf/cuda-graph.md) |
| 一个脚本、一个进程 | 没有 HTTP、没有流式、没有分词进程 | ZMQ 消息、tokenizer / detokenizer 进程、OpenAI 兼容 API | 阶段三 [服务化](../serve/message.md) |
| 只能用一张卡 | 模型放不进单卡显存 | 张量并行：四种线性层切法 | 阶段四 [张量并行](../perf/tensor-parallel.md) |

每个阶段结束时，你手里都有一个比上一阶段更强、但同样能跑、同样和 Hugging Face 逐 token 对齐的引擎。

!!! interview "怎么讲清楚"
    讲"推理引擎最核心的部分是什么"，用这份代码就够了：权重加载、前向（归一化、注意力、MLP、残差）、KV cache、采样循环四件事，prefill 和 decode 两种前向。然后讲它为什么不能直接上线——一次一个请求、KV 每步拷贝、注意力分数矩阵物化、前缀不复用——每一条对应一个成熟引擎里的一个模块（批处理、分页 KV、FlashAttention、前缀缓存）。能把"朴素做法 → 问题 → 对应模块"这条线讲顺，就说明真的理解了推理系统为什么长成这个样子。

## 练习

1. 把 `generate` 里的 KV cache 去掉（每步把全部 token 重新 `forward` 一遍），量一下 16 步 decode 的时间，和有 cache 的版本比。提示词换成 200 个 token 时差距有多大？
2. 在 `attention` 里打印 `scores` 的形状和占用的字节数，分别在 prefill 和 decode 时各看一次。提示词 4096 个 token 时 prefill 的分数矩阵有多大？
3. 把 `torch.set_num_threads` 的参数从 1 试到逻辑核数，画出 decode 的 tokens/s 曲线。拐点在哪里？和物理核数是什么关系？

??? success "参考答案"
    1. 没有 cache 时第 $n$ 步要算 $5 + n$ 个 token，16 步合计算了约 200 个 token 的前向，而有 cache 时只算 16 个；200 个 token 的提示词下差距超过十倍。这就是 KV cache 存在的全部理由——用内存换掉平方级的计算。
    2. prefill 时 `scores` 是 `[16, T, T]`；T = 4096 时是 $16 \times 4096^2 \times 4$ 字节 = 1 GiB，而且只是其中一层、一个 batch。decode 时是 `[16, 1, S]`，很小。FlashAttention 的意义就是永远不物化前者。
    3. 曲线先升后降，拐点通常在物理核数附近；超过之后同步开销超过并行收益。多卡、多实例部署时，每个实例分到的核数也按这个逻辑定。

## 小结

- [x] 推理引擎的骨架是四件事：权重、前向、KV cache、采样循环；两种前向：prefill 和 decode。
- [x] Qwen3 的注意力有 GQA 和 q_norm / k_norm 两处特别；RoPE 按前后两半配对。
- [x] 优化的方法论：先有可量的基线，改一处，再量，输出不变。线程数一行就让 decode 快了 68 倍。
- [x] 这份代码的每一个朴素之处，都对应本书后面的一章；每个阶段结束时手里都有一个能跑的引擎。
