# 推理引擎（vLLM / SGLang）学习路线

学完三本手册之后的下一站。总体思路：先吃透概念，再追一条请求在代码里怎么走，然后按专题深挖，最后自己动手改。

## 1. 吃透核心概念（1～2 周）

必读的论文和文章（大多已经在[大模型原理手册的学习资源](llm/docs/synthesis/resources.md)中列出）：

- Orca：连续批处理（continuous batching）的来源
- vLLM / PagedAttention（SOSP'23）
- SGLang / RadixAttention（NeurIPS'24）
- FlashAttention 1/2/3
- Sarathi-Serve：分块 prefill
- DistServe / Splitwise：PD 分离
- EAGLE / Medusa：投机解码

读完要能自己讲清楚：prefill 和 decode 的算力特征有什么不同、KV Cache 的显存怎么算、为什么要分页、前缀缓存的命中率受什么影响。这些问题在大模型原理手册的[推理原理](llm/docs/inference/)部分和[自测题库](llm/docs/synthesis/quiz.md)中都有对应的内容。

## 2. 先读精简版，再读正式版

直接读 vLLM 主仓库很容易迷路。建议先读 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)：一千多行 Python，把分页 KV、调度、CUDA Graphs、张量并行都实现了一遍。读懂之后，再回头看正式版多出来的代码在解决什么问题。

## 3. 追一条请求的完整生命周期（最关键的一步）

起一个小模型，发一条请求，一路打断点或加日志，自己画出进程和数据流图：

- **vLLM（V1）**：API server → AsyncLLM → EngineCore（独立进程，通过 ZMQ 通信）→ Scheduler（统一的 token 预算，prefill 和 decode 不再分开调度）→ KVCacheManager（block pool，前缀缓存按 block 哈希查找）→ GPUModelRunner（准备输入、CUDA Graphs）→ 注意力后端 → Sampler → 输出处理与反分词
- **SGLang**：TokenizerManager → Scheduler 进程（调度策略与 RadixCache）→ TpModelWorker / ModelRunner（ForwardBatch）→ 注意力后端（FlashInfer / Triton / FA3）→ DetokenizerManager；另外重点看 overlap scheduler，也就是 CPU 调度和 GPU 计算怎么重叠

能凭记忆画出这张图，就算入门了。大模型原理手册中[一个 token 的完整旅程](llm/docs/synthesis/token-journey.md)的“逐站解读”表可以当作索引：每遇到一个模块，就问它对应表中的哪一站、解决了哪个瓶颈。

## 4. 按专题深挖，两家对比着看

- 前缀缓存：vLLM 的哈希块与 SGLang 的基数树，淘汰策略有什么区别
- 调度：抢占、分块 prefill、优先级
- 并行：TP / PP / EP / DP Attention（DeepSeek 这类 MLA + MoE 模型最值得研究）
- PD 分离与 KV 传输
- 投机解码
- 结构化输出（xgrammar）
- 量化（FP8 / AWQ / GPTQ），以及 kernel 是怎么接进来的（对照 CUDA 手册的量化与 GEMV 一章）

同一个问题看两家的不同解法，理解会比只看一家深得多。

## 5. 动手

- 用 `vllm bench serve` / `sglang.bench_serving` 在同一个负载下对比两家，改参数，看 TTFT、TPOT、吞吐怎么变
- 用 Nsight Systems 或 torch profiler 抓一次 decode step，看时间花在哪里（kernel、调度开销、同步）
- 自己接入一个新模型，或者给某个注意力后端加一个小功能
- 挑 good first issue 提 PR；平时多看 GitHub 上的 RFC issue 和大 PR 的讨论，设计上的取舍基本都写在那里

## 学习资料

- vLLM 官方博客（V1 架构的几篇必看）、LMSYS 博客（SGLang 每个大版本都有技术文章）
- GitHub 上的 Awesome-ML-SYS-Tutorial（SGLang 社区维护，有很多中文源码解读）
- 知乎上两家的源码解析很多，但版本变化快，注意看发布时间
