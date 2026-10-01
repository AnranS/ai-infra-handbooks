# SGLang 源码导读

<p class="lead">SGLang 与 vLLM 解决的是同一个问题，结构也很相似：前端进程负责 HTTP 和分词，调度器进程负责调度与执行，另有一个反分词进程。它的特色在于基数树前缀缓存、CPU 与 GPU 重叠的调度循环，以及面向 DeepSeek 这类大规模 MoE 模型的一整套并行与 PD 分离方案。这一章基于 SGLang 0.5.20，沿请求的路径读一遍主线，并与 vLLM 对照。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `TokenizerManager`、`Scheduler`、`DetokenizerManager` 各在哪个进程？请求和结果在它们之间怎么流动？
    2. `ScheduleBatch` 和 `ForwardBatch` 有什么区别？
    3. `event_loop_overlap` 是怎样让 CPU 调度与 GPU 计算重叠的？什么情况下会关掉重叠？
    4. SGLang 的 KV Cache 为什么要用 `ReqToTokenPool` 和 `TokenToKVPoolAllocator` 两级结构？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `TokenizerManager` 在主进程里（和 HTTP 服务在一起），负责分词、把请求发给调度器；`Scheduler` 在每个 TP rank 各一个的调度器进程里（同时执行模型）；`DetokenizerManager` 单独一个进程。请求：主进程 → 调度器；结果：调度器 → 反分词进程 → 主进程，都用 ZMQ。
    2. `ScheduleBatch` 是调度层在 CPU 上的数据结构（请求列表、调度用的元信息）；`ForwardBatch` 是为一次前向准备的 GPU 张量（input_ids、positions、KV 位置、注意力元数据），由前者转换而来。
    3. 先把本批提交到 GPU，再在 CPU 上处理上一批的结果（反分词、判断结束、准备下一批），CPU 的调度和 GPU 的计算重叠。连续两个批次都是 prefill 时默认关掉重叠：否则第一个批次的首 token 要等第二个批次提交之后才被处理和发出，TTFT 被拉长一整个批次（由 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 控制）；调试时也可以用 `--disable-overlap-schedule` 整体关掉。
    4. `ReqToTokenPool` 记录每个请求的每个位置对应哪个 KV 槽位（请求 → token 位置 → 槽位），`TokenToKVPoolAllocator` 管理槽位的分配和释放（槽位 → 实际的 KV 数据）。两级结构把"请求用了哪些槽"和"槽的分配"分开：前缀缓存共享槽位、请求结束释放、基数树决定保留什么，都只需要改映射。

!!! note "版本"
    本章基于 SGLang 0.5.20（2026 年 9 月）的 `sglang/srt/` 目录。SGLang 同样迭代很快，近期把不少逻辑拆进了 `managers/scheduler_components/`、`mem_cache/` 的子目录和 `arg_groups/`，类名与主线结构则相对稳定。

## 准备工作

```bash
pip download sglang==0.5.20 --no-deps -d . && python -m zipfile -e sglang-0.5.20-*.whl sglang-src
# 或者：git clone https://github.com/sgl-project/sglang && git checkout v0.5.20
```

启动服务：`python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000`。调试时常用 `--disable-cuda-graph`、`--disable-overlap-schedule`、`--log-level debug`；`python -m sglang.benchmark.one_batch`（旧路径 `sglang.bench_one_batch` 已弃用）可以不启动服务、直接对单个批次做前向，适合单步跟踪模型执行。

## 目录地图

| 目录 | 内容 |
| --- | --- |
| `srt/entrypoints/` | HTTP 服务（`http_server.py`）、离线 `Engine`（`engine.py`，负责启动各子进程）、OpenAI 兼容接口（`openai/`） |
| `srt/managers/` | `TokenizerManager`、`Scheduler`、`DetokenizerManager`、`TpModelWorker`、批次数据结构（`schedule_batch.py`）、调度策略（`schedule_policy.py`）、`scheduler_components/`（结果处理、输出流等从调度器拆出来的组件） |
| `srt/mem_cache/` | KV 内存池与分配器、基数树前缀缓存、分层缓存（HiCache） |
| `srt/model_executor/` | `ModelRunner`、`ForwardBatch`、CUDA Graph 执行器 |
| `srt/layers/` | 并行线性层、`RadixAttention`、注意力后端（`attention/`）、MoE（`moe/`）、量化、采样器 |
| `srt/models/` | 各模型的实现 |
| `srt/speculative/` | EAGLE、MTP、n-gram 等投机解码 |
| `srt/disaggregation/` | PD 分离：prefill/decode 两侧的逻辑与 Mooncake、NIXL 等传输后端 |
| `srt/constrained/` | 结构化输出（xgrammar 等语法后端） |
| `srt/eplb/`、`srt/elastic_ep/` | 专家负载均衡、弹性专家并行 |

高性能算子在独立的 `sgl-kernel` 包中；跨实例的路由由 Rust 编写的 SGLang Model Gateway（原 sgl-router）负责。

## 进程结构

![图：SGLang 的进程结构——TokenizerManager、Scheduler、DetokenizerManager](../assets/figures/sglang-processes.svg){.aig-svg}

```text
主进程                                  Scheduler 进程（每个 TP rank 一个）         DetokenizerManager 进程
┌─────────────────────────┐   ZMQ     ┌───────────────────────────────┐   ZMQ    ┌───────────────────────┐
│ HTTP 服务（FastAPI）       │ ───────▶ │ event_loop_overlap / normal    │ ───────▶ │ 增量反分词              │
│ TokenizerManager         │           │  接收请求 → 组批 → run_batch     │           │ 停止字符串              │
│  分词、对话模板、多模态预处理 │ ◀─────────────────────────────────────────────── │ 结果发回 TokenizerManager│
└─────────────────────────┘           │  TpModelWorker → ModelRunner    │           └───────────────────────┘
                                      └───────────────────────────────┘
```

与 vLLM 最大的结构差别是：SGLang 的调度器**和模型执行在同一个进程**（每个 TP rank 各有一个 Scheduler 进程，各自调度出相同的批次），没有单独的"EngineCore → worker"一层；反分词则有专门的进程。数据并行（`--dp-size`）时，`DataParallelController` 在前面把请求分发给多个调度器。

## 主线一：TokenizerManager

`TokenizerManager.generate_request()`（`srt/managers/tokenizer_manager.py`）做的事与 vLLM 的前端类似：`_tokenize_one_request` 分词（以及多模态输入的预处理），构造请求对象，通过 ZMQ 发给调度器；然后等待这个请求的结果。结果由 `handle_loop` 从 DetokenizerManager 接收，`_handle_batch_output` 分发给各个请求的等待者，HTTP 层据此生成流式响应。

## 主线二：Scheduler 的事件循环

调度器进程由 `run_scheduler_process` 启动，进入两种事件循环之一（`srt/managers/scheduler.py`）：

```py
def event_loop_normal(self):                     # 普通循环：接收 → 组批 → 执行 → 处理结果
    while True:
        self.ingest_requests()
        plan = self.get_next_batch_to_run(running_batch=self.running_batch, last_batch=self.last_batch)
        batch = plan.batch_to_run
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        self.last_batch = batch
```

```py
def event_loop_overlap(self):                    # 重叠循环（默认）：先提交本批，再处理上一批的结果
    while True:
        self.ingest_requests()
        plan = self.get_next_batch_to_run(...)
        batch = plan.batch_to_run
        if batch:
            batch_result = self.run_batch(batch)                 # 异步提交到 GPU，立即返回
            self.result_queue.append((batch.copy(), batch_result))
        if self.last_batch:
            pop_and_process()                                    # 在 GPU 计算本批的同时，处理上一批的结果
        self.launch_batch_sample_if_needed(batch_result, batch)
        self.last_batch = batch
```

重叠循环的关键在于：**GPU 执行第 N 批的同时，CPU 在处理第 N−1 批的结果、准备第 N+1 批**。第 N 批的输入依赖第 N−1 批刚采样出的 token，这些 token 在 CPU 上还不知道，SGLang 用"未来 token"的占位（结果直接在 GPU 上写入下一批的输入缓冲区）解决。代价是结果的处理晚一步，所以在两个连续的 prefill 批次之间、以及结构化输出需要同步语法状态时，会关闭这一步的重叠（`is_disable_overlap_for_batch`）。vLLM 的异步调度是同一个思想。

`get_next_batch_to_run()` 的决策顺序是：

1. 把上一步刚做完 prefill 的请求并入运行批次（`running_batch`）；
2. 尝试组一个新的 prefill 批次（`get_new_batch_prefill`）：按调度策略（`SchedulePolicy`，默认 LPM）对等待队列排序，用 `PrefillAdder` 在剩余 token 预算与显存余量内逐个接收请求，超长的请求被分块（`chunked_prefill_size`）；
3. 组不出 prefill 批次时，让运行批次做一步 decode（`update_running_batch`）；显存不够就 `retract_decode` 撤回部分请求。

也就是说，SGLang 默认 **prefill 优先**：有新请求能 prefill 就先做 prefill，这一步里 decode 请求暂停；开启 `--enable-mixed-chunk` 后，prefill 分块与 decode 才会合并到同一个批次（`ForwardMode.MIXED`）。这与 vLLM"所有请求都在同一个 token 预算里"的做法不同，是两者调度行为的一个重要差别。

`process_batch_result()` 处理前向结果：`scheduler_components/batch_result_processor.py` 中的 `process_batch_result_prefill` 和 `process_batch_result_decode` 把 token 追加到请求、检查停止条件、把结束请求的 KV 插入基数树（`cache_finished_req`），`output_streamer.py` 的 `stream_output` 把增量发给 DetokenizerManager。

## 主线三：批次数据结构

`schedule_batch.py` 开头的注释说明了一个批次的数据流：

> `ScheduleBatch` → `ForwardBatch`。`ScheduleBatch` 由调度器管理，包含高层的调度数据，大部分在 CPU 上；`ForwardBatch` 由 `ModelRunner` 管理，包含底层的张量数据，大部分是 GPU 张量，由 `ForwardBatch.init_new` 从 `ScheduleBatch` 直接构造。

- **`Req`**：一个请求的全部状态（输入 token、输出 token、前缀匹配结果 `prefix_indices`、完成原因……）；
- **`ScheduleBatch`**：一批 `Req` 及其调度信息，`forward_mode` 表示这是 `EXTEND`（prefill）、`DECODE` 还是 `MIXED` 等；
- **`ForwardBatch`**（`srt/model_executor/forward_batch_info.py`）：`input_ids`、`positions`、`seq_lens`、`extend_prefix_lens`、`extend_seq_lens`、`req_pool_indices`、`out_cache_loc`（即 slot mapping）……对应迷你引擎的 `BatchInput`。

## 主线四：KV 内存与前缀缓存

SGLang 的 KV 管理分两级：

- **`ReqToTokenPool`**（`mem_cache/memory_pool.py`）：一张 `[最大请求数, 最大上下文长度]` 的表，第 r 行第 i 列是请求 r 的第 i 个 token 的 KV 槽位。它相当于 vLLM 的块表，只是粒度是 token（或页）而不是块；
- **`TokenToKVPoolAllocator`**（`mem_cache/allocator/`）：管理空闲槽位，"给我 n 个槽位"；
- **`MHATokenToKVPool`** 等 `KVCache` 子类：真正的 K/V 张量，按 `(layer_id, 槽位)` 读写。

基数树 `RadixCache`（`mem_cache/radix_cache.py`）记录"哪些 token 序列的 KV 在哪些槽位"，见[前缀缓存一章](../engine/prefix-cache.md)。请求被接收时 `match_prefix` 得到 `prefix_indices`，只为剩下的部分分配新槽位；请求结束时 `cache_finished_req` 把它的 KV 插入树中，而不是直接释放。显存不够时，分配器先让基数树淘汰（`evict`）不再被使用的叶子。`mem_cache/README.md` 给出了这一层的完整分层（分配策略 → 多池路由 → 分配器 → 设备池 → 主机池 → 存储后端）。

## 主线五：模型执行

`run_batch` 调用 `TpModelWorker.forward_batch_generation`（`srt/managers/tp_worker.py`），它构造 `ForwardBatch` 并交给 `ModelRunner.forward`（`srt/model_executor/model_runner.py`）。`_forward_raw` 先判断能否使用 decode 的 CUDA Graph（`decode_cuda_graph_runner.can_run_graph`），能则重放，否则按 `forward_mode` 走 extend 或 decode 的普通前向。

模型代码（如 `srt/models/qwen2.py`）与 vLLM 高度相似：同样使用合并的 QKV、合并的 gate/up、`RMSNorm(x, residual)` 的融合形式，以及按列、按行切分的并行线性层（SGLang 早期直接复用了 vLLM 的这些层）。注意力层是 `RadixAttention`（`srt/layers/radix_attention.py`），它不直接计算，而是把 `q, k, v` 和 `forward_batch` 交给当前的注意力后端（`srt/layers/attention/` 下的 `flashinfer_backend.py`、`flashattention_backend.py`、`triton_backend.py`、`flashmla_backend.py` 等）。默认后端按硬件和模型自动选择，启动日志会打印选择结果，也可以用 `--attention-backend` 指定，或用 `--prefill-attention-backend`、`--decode-attention-backend` 分别指定。

## SGLang 与 vLLM 对照

| | vLLM V1 | SGLang |
| --- | --- | --- |
| 进程 | 前端 / EngineCore / 每卡一个 worker | 主进程（前端）/ 每个 TP rank 一个 Scheduler（含模型执行）/ Detokenizer |
| 调度 | 统一 token 预算，prefill 与 decode 总在同一步 | 默认 prefill 优先，可开启混合批次 |
| 显存准入 | 能分配就接收，不够再抢占 | 预估未来需求（`new_token_ratio`），保守接收，必要时撤回 |
| 前缀缓存 | 链式哈希块 + LRU 空闲队列 | 基数树 + 叶子 LRU + 引用锁 |
| 调度策略 | FCFS、优先级 | LPM、DFS-weight（缓存感知）、FCFS、LOF 等 |
| CPU/GPU 重叠 | 异步调度（`AsyncScheduler`）、批次队列 | `event_loop_overlap` |
| 批次数据 | `SchedulerOutput`（增量）→ 持久批次 `InputBatch` | `ScheduleBatch` → `ForwardBatch` |
| 大规模 MoE | DP + EP、DeepEP、EPLB | DP Attention + EP、DeepEP、EPLB、两批重叠（TBO） |

两者在不断互相借鉴，这张表描述的是 2026 年 9 月的默认行为，细节以源码为准。

## 建议的阅读顺序

1. `srt/entrypoints/engine.py` 的 `_launch_subprocesses`：看清有哪些进程、如何启动；
2. `Scheduler.event_loop_normal` → `get_next_batch_to_run` → `get_new_batch_prefill`（配合 `schedule_policy.py` 的 `PrefillAdder`）→ `update_running_batch`；
3. `schedule_batch.py`：`Req`、`ScheduleBatch` 以及准备 extend/decode 的方法；
4. `mem_cache/radix_cache.py` 与 `memory_pool.py`；
5. `ModelRunner.forward` → `RadixAttention` → 一个注意力后端；
6. 再看 `event_loop_overlap`，理解重叠调度的时序。

!!! interview "面试怎么答"
    "SGLang 和 vLLM 有什么区别？"不要停留在"一个用基数树、一个用哈希"。可以按上面的对照表挑三点讲：**进程结构**（SGLang 的调度器与模型执行在同一进程）、**调度行为**（prefill 优先 vs 统一预算，保守准入 vs 激进准入加抢占）、**前缀缓存**（基数树与缓存感知调度 vs 哈希块）。最后补一句"两者在持续互相借鉴，比如都做了 CPU/GPU 重叠、都支持 PD 分离与大规模 EP"，显示你看的是最新代码。

## 练习

**1. 重叠调度的代价。** 为什么 SGLang 在"两个连续的 prefill 批次"之间默认关闭重叠？

??? success "参考答案"
    重叠调度让第 N 批的结果晚一步处理。对 prefill 来说，结果处理包括发出第一个 token；如果连续两个批次都是 prefill，第一个批次的首 token 要等第二个批次提交之后才被处理和发出，TTFT 被拉长了整整一个批次的时间。为了 TTFT，SGLang 在这种情况下先处理完上一批再继续（由 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 控制）。

**2. token 粒度的代价。** `ReqToTokenPool` 为每个请求的每个 token 记录一个槽位号。与 vLLM 为每 16 个 token 记录一个块号相比，有什么代价？SGLang 如何缓解？

??? success "参考思路"
    元数据多了 16 倍：块表更大，构造和拷贝更慢；更重要的是注意力 kernel 按 token 间接寻址，K/V 在显存中可能完全不连续，访存效率低。缓解办法是支持 `--page-size`（例如 16、32、64），按页分配和匹配，基数树的匹配也按页对齐。一些注意力后端（FlashMLA、TensorRT-LLM MLA）本身就要求特定的页大小。

## 小结

- [x] SGLang 的主进程负责 HTTP 与分词，每个 TP rank 一个调度器进程（同时执行模型），反分词单独一个进程。
- [x] 调度循环默认重叠 CPU 与 GPU 工作；组批时 prefill 优先，显存不足时撤回 decode 请求。
- [x] `ScheduleBatch`（CPU、调度层）→ `ForwardBatch`（GPU 张量）；KV 用 `ReqToTokenPool` + 槽位分配器两级管理，基数树决定保留什么。
- [x] 与 vLLM 相比：进程划分、调度行为、前缀缓存结构不同，核心思想（连续批处理、分页/槽位 KV、CPU/GPU 重叠）相同。
