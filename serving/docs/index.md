# 推理系统手册

<p class="lead">这是四本手册的最后一本，面向大模型推理岗位：推理框架开发、推理优化、推理平台。前三本解决了"用什么语言写""模型在算什么""GPU 怎么算得快"，这一本回答"推理系统怎样把它们组织起来"。我们先从零写一个能用的推理引擎（分页 KV、连续批处理、前缀缓存、流式 API），再对照它读懂 vLLM 与 SGLang 的源码，然后深入分布式推理、性能工程和各个进阶专题，最后是面试、系统设计与作品集。</p>

## 学完能做到

- 讲清一个请求从 HTTP 到最后一个 token 在推理引擎中经过的每一步，以及 vLLM、SGLang 中对应的类和函数；
- 独立实现分页 KV Cache、变长批处理、连续批处理调度器（分块 prefill、抢占）、前缀缓存（哈希块与基数树）、批量采样与流式 OpenAI 接口；
- 手推张量并行、专家并行、流水线并行与上下文并行的切分与通信，估算它们的代价；解释 PD 分离与 KV 分层缓存的设计；
- 用开环压测与模拟器做容量规划，用 profiler 定位瓶颈，为给定负载选择量化方案；
- 讲清投机解码（含树形草稿）、长上下文与稀疏注意力、结构化输出、多模态、RL rollout 中的推理问题；
- 从容应对推理岗的面试题、手撕代码与系统设计。

## 学习路线

<div class="roadmap" markdown>

| 部分 | 章节 | 目标 | 建议用时 |
| --- | --- | --- | --- |
| 全景 | [一个请求的一生](engine/overview.md) | 建立整本书的地图 | 半天 |
| 从零写一个推理引擎 | [分页 KV](engine/paged-kv.md) · [变长批处理](engine/batch-layout.md) · [调度器](engine/scheduler.md) · [前缀缓存](engine/prefix-cache.md) · [采样与 API](engine/sampler-api.md) · [CUDA Graphs](engine/graphs-compile.md) | 亲手实现引擎的每个核心组件，并逐 token 验证 | 1.5 周 |
| 源码导读 | [vLLM V1](source/vllm.md) · [SGLang](source/sglang.md) | 把自己写过的组件对应到真实代码 | 1 周 |
| 分布式推理 | [张量并行](distributed/tensor-parallel.md) · [专家并行](distributed/expert-parallel.md) · [流水线与上下文并行](distributed/pp-cp.md) · [PD 分离](distributed/pd-disagg.md) · [KV 分层缓存](distributed/kv-offload.md) | 理解大模型的部署方式与通信代价 | 1 周 |
| 性能工程 | [压测与容量规划](perf/benchmark.md) · [Profiling](perf/profiling.md) · [量化部署](perf/quantization-deploy.md) | 会测、会找瓶颈、会选方案 | 4～5 天 |
| 进阶专题 | [投机解码](topics/speculative.md) · [长上下文](topics/long-context.md) · [结构化输出](topics/structured-output.md) · [多模态](topics/multimodal.md) · [RL 中的推理](topics/rl-rollout.md) | 覆盖当前推理系统的前沿问题 | 1 周 |
| 求职 | [面试题库](career/interview.md) · [手撕代码](career/coding.md) · [系统设计](career/system-design.md) · [作品集与学习计划](career/projects.md) · [硬件速查](career/hardware.md) | 把知识转化为面试表现 | 按需 |

</div>

建议先读完[大模型原理手册](llm://)（至少 Transformer 解剖与推理原理两部分）和 [CUDA 进阶手册](cuda://)的前半部分，再开始本书。本书大量引用它们的章节。

## 贯穿全书的主线

- **一个自己写的推理引擎**：以大模型手册的 `mini_llm.py` 为模型，逐章加上分页 KV（`paged.py`）、批处理前向（`runner.py`）、调度与引擎主循环（`nano_engine.py`）、前缀缓存（`prefix_cache.py`、`radix.py`）、采样与服务（`sampler.py`、`detokenizer.py`、`api_server.py`），之后的分布式与专题章节也都在它上面做实验。每一步都与"逐个请求单独生成"的结果逐 token 比较。
- **真实的源码**：源码导读与各章的"源码对照"基于 **vLLM 0.30.0** 与 **SGLang 0.5.20**（2026 年 9 月）。所有文件路径、类名、函数名、命令行参数都在源码中核对过。
- **三种提示框**：

!!! source "源码对照"
    这个概念在 vLLM 和 SGLang 中的具体实现位置。

!!! interview "面试怎么答"
    这个知识点在面试中常见的问法，以及回答的结构。

!!! inference "推理视角"
    与大模型手册相同：这部分知识对推理性能意味着什么。

## 准备环境

与大模型手册相同的环境（CPU 版 PyTorch、transformers、Qwen2.5-0.5B-Instruct），另外：

```bash
uv pip install pillow                                                      # 多模态一章需要
uv pip install torchvision --index-url https://download.pytorch.org/whl/cpu
```

多模态一章使用 Qwen2.5-VL-3B-Instruct（约 7.5 GB，放在 `models/Qwen2.5-VL-3B-Instruct`）。张量并行、专家并行、流水线并行、PD 分离的实验通过 `torch.distributed` 的 gloo 后端在 CPU 上多进程运行，不需要 GPU。书中定义的模块文件（连同依赖的 `mini_llm.py` 等）打包在 [serving-code.tar.gz](assets/serving-code.tar.gz) 里。

!!! note "关于代码的验证"
    所有 `python` 代码块和 `>>>` 示例都在 PyTorch 2.14（CPU）和 transformers 5.17 下实际运行过。引擎的输出与逐个请求单独生成的结果逐 token 比较；张量并行、流水线并行、PD 分离与单进程结果比较；分页注意力、ring attention、树注意力与标准注意力比较；量化、KV 淘汰等在真实模型上测量困惑度。书中的 GPU 性能数字来自屋顶线估算和模拟器，文中会明确说明，上线前请以实测为准。
