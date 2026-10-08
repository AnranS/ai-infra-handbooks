# 导读：架构与复刻路线

<p class="lead">动手之前，先把整个系统看一遍：一个请求从 HTTP 进来，经过哪些进程、变成哪些数据结构、在哪一步被计算，最后怎样一个 token 一个 token 地流回客户端。这一章给出 mini-sglang 的全貌、模块之间的依赖、我们的复刻顺序，以及怎样在一台没有 GPU 的机器上验证每一步。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. mini-sglang 由哪几类进程组成？一个 TP=4 的服务一共有几个进程？
    2. 调度器进程里，"调度器"和"引擎"分别负责什么？
    3. 为什么模型的 `forward()` 不需要任何参数？
    4. 官方实现只支持 CUDA，本书靠哪三个手段在 CPU 上验证 GPU 相关的代码？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 三类：API Server（HTTP 入口）、tokenizer / detokenizer 进程、每个 TP rank 一个调度器进程（里面有调度器和引擎）。TP=4 时是 1 + 1 + 4 = 6 个进程。
    2. 调度器决定"这一轮算什么、KV 放在哪里"：接收请求、选出 batch、分配 KV、处理结果；引擎负责"算"：执行模型的前向和采样。
    3. 当前 batch 的所有信息（请求、位置、注意力元数据）放在一个全局的 `Context` 里，模型的每一层需要时直接从那里读取，所以 `forward()` 不需要传参数。
    4. 设备抽象（同一份代码在 CPU 上跑）、与 GPU 实现同接口的参考实现和替身（比如注意力后端、FlashInfer / FlashAttention 的假实现），以及模拟器（CUDA Graph 仿真、CUDA kernel 的 CPU 模拟器、Triton 解释器）。

## 先看结果

这是最终要做出来的东西：离线接口 `LLM` 在 CPU 上跑 Qwen3-0.6B，三个请求一起生成。

@@code examples/ch00_quickstart.py@@

@@output ch00_quickstart@@

在线服务则是一条命令：`python -m minisgl --model Qwen/Qwen3-0.6B`，之后用任何 OpenAI 客户端访问 `http://127.0.0.1:1919/v1/chat/completions`。

## 进程结构

mini-sglang 是一个多进程系统。以 TP=2 为例：

@@diagram processes mini-sglang 的进程结构（TP=2）@@

- **API Server**（主进程）：接收 HTTP 请求，给每个请求分配 `uid`，把文本发给 tokenizer；再把收到的增量文本以 SSE 流式写回客户端。
- **tokenizer / detokenizer**：分词和增量反分词。默认两者共用一个进程（`--num-tokenizer 0`）。
- **调度器进程**：每个 TP rank 一个。每个进程里有一个 `Scheduler`（决定这一轮算哪些请求、分配 KV 缓存）和一个 `Engine`（持有模型和 KV 池，执行前向和采样）。只有 rank 0 与 tokenizer 通信；它把收到的消息原样广播给其他 rank，保证所有 rank 做出完全相同的调度决策。

所以 TP=4 的服务一共有 1 + 1 + 4 = 6 个进程（API Server、tokenizer、4 个调度器）。进程之间的控制消息走 ZMQ，张量并行的数据走 NCCL。

!!! upstream "官方实现"
    启动逻辑在 @@upstream server/launch.py:launch_server@@，调度器主循环在 @@upstream scheduler/scheduler.py:Scheduler.overlap_loop@@。官方文档 [docs/structures.md](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/docs/structures.md) 有一张进程图，与上图一致。

## 一个请求的旅程

@@diagram request-lifecycle 一个请求经过的进程与消息@@

1. 客户端 `POST /v1/chat/completions`。API Server 分配 `uid = 7`，发送 `TokenizeMsg(uid=7, text=[消息列表], sampling_params)`。
2. tokenizer 套用对话模板、分词，发送 `UserMsg(uid=7, input_ids=张量)` 给调度器 rank 0。
3. 调度器把它放进 prefill 等待队列。某一轮调度时，`PrefillAdder` 在 Radix Cache 里查找最长的已缓存前缀，检查剩余显存够不够、有没有空闲的请求槽，够就接纳：分配一行 page table、锁住命中的前缀。
4. `_prepare_batch` 为这一轮要算的 token 分配 KV 页，算出每个 token 的位置、要写入的 KV 位置，注意力后端准备好元数据。
5. `Engine.forward_batch`：模型前向 → 取每个请求最后一个位置的 logits → 采样。采样结果一方面直接写回 GPU 上的 `token_pool`（作为下一轮的输入），另一方面异步拷回 CPU。
6. 调度器处理结果：把新 token 追加到请求上，判断是否结束，发送 `DetokenizeMsg(uid=7, next_token, finished)`。prefill 刚结束的请求，提示词的 KV 被插入 Radix Cache。
7. 之后每一轮 decode 重复第 4～6 步，直到遇到 EOS 或达到 `max_tokens`。结束时释放请求槽，把 KV 交还给缓存。
8. detokenizer 把 token 增量地变成文本，发送 `UserReply(uid=7, incremental_output, finished)`；API Server 把它写进 SSE 流。

这条路径上的每一步，都对应本书的一章。

@@video lifecycle 动画：一个请求的一生（约 2 分钟，带配音和字幕）@@

## 模块与依赖

`python/minisgl/` 下的模块，按依赖从底到顶：

@@diagram modules 模块分层@@

| 模块 | 内容 | 章节 |
| --- | --- | --- |
| `utils`、`env` | 日志、注册表、ZMQ 队列、HF 工具；（我们加的）设备抽象 | 各章用到时介绍 |
| `core` | `SamplingParams`、`Req`、`Batch`、`Context` | [核心数据结构](../compute/core.md) |
| `distributed` | TP 信息、all-reduce / all-gather | [张量并行](../perf/tensor-parallel.md) |
| `kernel` | 算子：RMSNorm、RoPE、写 KV、查表……（CPU 参考实现 + GPU 版本） | [算子层](../compute/layers.md)、[CUDA kernel](../perf/kernels.md) |
| `layers` | `BaseOP` 体系与各种层 | [算子层](../compute/layers.md) |
| `models` | 模型配置、模型结构、权重加载 | [模型与权重加载](../compute/models.md) |
| `kvcache` | KV 池、前缀缓存（naive 与 radix） | [KV 池](../compute/kvcache.md)、[Radix Cache](../schedule/radix-cache.md) |
| `attention` | 注意力后端：torch（我们加的）、FlashInfer、FlashAttention | [注意力后端](../compute/attention.md)、[GPU 注意力](../perf/gpu-attention.md) |
| `moe` | MoE 后端：torch 参考实现、Triton fused MoE | [MoE](../perf/moe.md) |
| `engine` | `Engine`、`Sampler`、`GraphRunner` | [Engine 与采样器](../compute/engine.md)、[CUDA Graph](../perf/cuda-graph.md) |
| `scheduler` | 调度器与各个 Manager | 第三部分全部 |
| `message` | 进程间消息与序列化 | [消息与 ZMQ](../serve/message.md) |
| `tokenizer` | tokenizer / detokenizer 进程 | [Tokenizer](../serve/tokenizer.md) |
| `server`、`llm` | API Server、启动器；离线接口 `LLM` | [API Server](../serve/api-server.md)、[调度器骨架](../schedule/scheduler.md) |

有两个贯穿全局的设计值得先记住：

- **全局上下文 `Context`。** 模型的 `forward()` 没有参数：输入 token、位置、注意力元数据都挂在"当前 batch"上，而当前 batch 放在进程级的全局 `Context` 里。任何一层需要这些信息时，调用 `get_global_ctx().batch` 读取。这让模型代码非常干净，代价是模型只能在 `ctx.forward_batch(batch)` 这个上下文里运行。
- **注册表 `Registry`。** 命令行里的 `--attn fa,fi`、`--cache-type radix`、`--moe-backend fused` 都是名字，由注册表映射到具体实现。新增一种后端只需要写一个类、注册一个名字。

## 复刻路线

本书按"先算得对、再排得好、然后服务化、最后变快变大"的顺序：

1. **算得对**（第 1～6 章）：没有调度器，手工组 batch，让模型在分页 KV 缓存上正确地 prefill 和 decode。终点是手工循环的贪心输出与 HF 逐 token 相同。
2. **排得好**（第 7～11 章）：加上调度器。先是最朴素的连续批处理，再依次加入准入控制、Radix Cache、分块 prefill、重叠调度。每一步的输出都不能变。
3. **服务化**（第 12～15 章）：消息、tokenizer 进程、调度器的收发、API Server，得到完整的在线服务。
4. **更快、更大**（第 16～21 章）：张量并行、GPU 注意力后端、CUDA Graph、自定义 kernel、MoE，以及基准测试。

每章都遵循同样的结构：章首自测 → 原理 → 我们的实现（"本章要写的文件"）→ 官方实现对照 → 测试与运行结果 → 练习 → 小结。

## 在 CPU 上验证 GPU 代码

开发机没有 GPU 是常态。官方代码里 CUDA 专有的东西有四类，本书分别处理：

| 官方用到的 | 本书的做法 | 在哪一章 |
| --- | --- | --- |
| `torch.cuda.Stream`、`Event`、锁页内存 | `utils/device.py`：CUDA 上原样使用，CPU 上换成什么也不做的 `NullStream`、`NullEvent` | [Engine](../compute/engine.md)、[重叠调度](../schedule/overlap.md) |
| FlashInfer / FlashAttention 的注意力 kernel | 多一个 PyTorch 参考后端 `torch`；测试时再用"同接口的假 FlashInfer / 假 sgl_kernel"验证官方后端代码的参数语义 | [注意力后端](../compute/attention.md)、[GPU 注意力](../perf/gpu-attention.md) |
| CUDA Graph | CPU 上的仿真 `EmulatedGraph`：replay 时只能读固定缓冲区，漏拷任何输入都会让结果出错 | [CUDA Graph](../perf/cuda-graph.md) |
| 自定义 CUDA / Triton kernel | CUDA kernel 用 nvcc 12.9、13.4 编译，并在 [CUDA 手册](cuda://)的 CPU 模拟器上运行自检；Triton kernel 用解释器模式运行 | [CUDA kernel](../perf/kernels.md)、[MoE](../perf/moe.md) |

在 GPU 上，同一份代码自动选择 FlashInfer / FlashAttention（`--attn auto`）、开启 CUDA Graph、使用自定义 kernel 和 Triton fused MoE。

!!! diff "与官方的差异：设备抽象"
    官方的 `Engine.__init__` 第一件事就是 `torch.cuda.set_device`，调度器里直接创建 `torch.cuda.Stream()`。我们把这几处换成 `create_stream(device)`、`create_event(device)`、`pin(device)` 等函数，行为在 CUDA 上完全相同。另外 `EngineConfig` 多了 `device` 和 `cpu_kv_cache_bytes` 两个字段。

## 环境

```bash
git clone https://github.com/AnranS/ai-infra-handbooks && cd ai-infra-handbooks/minisgl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # CPU 版 torch 即可；GPU 上再装 ".[gpu]"
# 模型：放到 models/ 下（本书用 float32 跑，与 HF 对齐时最稳定）
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
```

官方仓库建议也克隆一份放在旁边，读每章时对照：

```bash
git clone https://github.com/sgl-project/mini-sglang && git -C mini-sglang checkout 9a91cfa
```

!!! warning "CPU 上的线程数"
    PyTorch 在 CPU 上默认使用"逻辑核数"个线程。decode 时的矩阵乘很小（批大小只有几），32 个线程之间的同步开销会让它慢几十倍：在本书的开发机上，默认设置下 decode 一步要 2.4 秒，把线程数降到物理核数后只要 40 毫秒。所以我们的 `Engine` 在 CPU 上会把线程数设为 `物理核数 / TP 数`（除非你设置了 `OMP_NUM_THREADS`）。

!!! interview "怎么讲清楚"
    讲"讲讲一个推理引擎的架构"的时候，可以用 mini-sglang 当骨架：API Server（HTTP、OpenAI 接口）+ tokenizer / detokenizer 进程 + 每个 TP rank 一个调度器进程，控制消息走 ZMQ，张量走 NCCL；TP=4 时一共 1 + 1 + 4 个进程。调度器进程里，调度器决定"这一轮算哪些请求、KV 放在哪"，引擎负责"算"（模型、注意力后端、采样、CUDA Graph）；模型从全局的 `Context` 读取当前 batch，所以 `forward()` 不需要参数。再讲一个请求的旅程（分词 → 排队 → prefill → 逐步 decode → 反分词流式返回 → 释放 KV），并说出多进程的理由：绕开 GIL，让 CPU 上的工作和 GPU 计算并行。

## 练习

**1. 数进程。** 用 mini-sglang 部署 TP=8 的服务，一共有几个进程？哪些进程之间传的是控制消息、哪些之间传的是张量？分别用什么通信？

??? success "参考答案"
    1 个 API Server + 1 个 tokenizer / detokenizer 进程 + 8 个调度器进程（每个 TP rank 一个，里面是调度器和引擎），共 10 个。请求、token 化后的请求和生成的结果这类控制消息走 ZMQ：API Server → tokenizer → rank 0 的调度器，rank 0 再把原始消息转发给其他 rank，结果从 rank 0 经 detokenizer 回到 API Server；8 个调度器进程之间的张量（张量并行的 all-reduce 等）走 NCCL。

**2. 复刻的顺序。** 复刻路线是"算得对 → 排得好 → 服务化 → 变快变大"。为什么不先写 CUDA Graph、自定义 kernel 这些性能优化？

??? success "参考答案"
    性能优化不改变结果，只改变执行方式，所以它们的正确性只能拿一个已经正确的版本来比对：先让参考实现和 HF 逐 token 对齐，之后每加一个优化（重叠调度、CUDA Graph、FlashInfer、自定义 kernel），都用同一组贪心生成的结果检查有没有变。反过来先做性能，出了错分不清是模型、调度还是优化本身的问题。另外，调度和服务化决定了性能优化的边界：比如 CUDA Graph 要求输入放在固定的缓冲区里，重叠调度要求输入 token 留在 GPU 上，这些都要在数据结构定下来之后才好做。

## 小结

- [x] mini-sglang = API Server + tokenizer/detokenizer + 每个 TP rank 一个调度器进程；控制消息走 ZMQ，张量走 NCCL。
- [x] 调度器决定"这一轮算什么、KV 放哪"，引擎负责"算"；模型从全局 `Context` 读取当前 batch。
- [x] 复刻顺序：算得对 → 排得好 → 服务化 → 变快变大；每一步都与 HF 逐 token 对齐。
- [x] GPU 专有的部分用设备抽象、参考实现、同接口替身和模拟器在 CPU 上验证。
