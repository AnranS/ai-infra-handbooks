# vLLM V1 源码导读

<p class="lead">vLLM 有几十万行代码，但一个请求真正经过的主线只有五段：前端（HTTP → AsyncLLM）、EngineCore 主循环、调度器与 KV Cache 管理、执行器与模型执行器、模型代码。这一章基于 vLLM 0.30.0 的源码，逐段给出关键文件、类和函数，并与前面写的迷你引擎一一对应。读完之后，你应该能在源码里迅速找到任何一个功能的落脚点。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `AsyncLLM`、`EngineCoreClient`、`EngineCoreProc`、`Executor`、`Worker`、`GPUModelRunner` 各在哪个进程里？它们之间怎么通信？
    2. `SchedulerOutput` 为什么把新请求和已有请求分成 `scheduled_new_reqs` 和 `scheduled_cached_reqs` 两部分？
    3. 在 vLLM 的 Qwen2 模型代码里，`qkv_proj`、`gate_up_proj`、`input_layernorm(hidden_states, residual)` 分别对应什么优化？
    4. 怎样让 vLLM 在单个进程里运行，方便打断点调试？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `AsyncLLM` 和 `EngineCoreClient` 在前端（API Server）进程里；`EngineCoreProc` 是单独的 EngineCore 进程（调度器、KV 管理、执行器）；多卡时 `Executor` 为每张卡拉起一个 worker 进程，里面是 `Worker` 和 `GPUModelRunner`。前端和 EngineCore 之间用 ZMQ，EngineCore 和 worker 之间用共享内存的消息队列。
    2. 新请求要把完整的信息（提示词、采样参数、块表）发给 worker；已经在运行的请求 worker 那边已经缓存着状态（持久批次），只需要发增量（新分配的块、新的 token 数）。分成两部分可以只传增量，减少每一步的序列化和通信。
    3. `qkv_proj`：把 q、k、v 三个投影合并成一次矩阵乘；`gate_up_proj`：把 gate 和 up 合并成一次矩阵乘；`input_layernorm(hidden_states, residual)`：融合的残差加法 + RMSNorm（一个 kernel 同时完成加残差和归一化）。
    4. 设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，让 EngineCore 和前端在同一个进程里运行（`InprocClient`）；用离线的 `LLM` 类、TP=1、`--enforce-eager` 跑一个小模型，就可以在调度器和 `GPUModelRunner` 里直接打断点。

!!! note "版本"
    本章的文件路径和函数名基于 vLLM 0.30.0（2026 年 9 月）。vLLM 迭代很快，名字可能变化，但主线结构自 V1 引擎以来一直稳定。读新版本时，用 `grep -rn "def schedule" vllm/v1` 这类搜索就能重新找到位置。

## 准备工作

获取源码最快的方式是下载源码包（不需要编译）：

```bash
pip download vllm==0.30.0 --no-deps --no-binary :all: -d . && tar xzf vllm-0.30.0.tar.gz
# 或者：git clone https://github.com/vllm-project/vllm && git checkout v0.30.0
```

调试时有几个开关非常有用：

| 设置 | 作用 |
| --- | --- |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | EngineCore 与前端运行在同一个进程（`InprocClient`），可以直接在调度器里打断点 |
| `--enforce-eager` | 关闭 CUDA Graph 与 torch.compile，调试时堆栈更清晰 |
| `VLLM_LOGGING_LEVEL=DEBUG` | 输出调度、KV Cache 等的详细日志 |
| 离线的 `vllm.LLM` 类 | 不启动 HTTP 服务，`llm.generate(...)` 直接走引擎，最适合单步跟踪 |
| `py-spy dump --pid <pid>` | 查看正在运行的进程卡在哪里 |

## 目录地图

| 目录 | 内容 |
| --- | --- |
| `vllm/entrypoints/` | `vllm serve` 命令行、OpenAI 兼容服务（`openai/api_server.py`、`openai/chat_completion/serving.py`）、离线 `LLM` 类（`llm.py`） |
| `vllm/v1/engine/` | 前端引擎 `AsyncLLM`、输入输出处理、`EngineCore` 及其多进程客户端 |
| `vllm/v1/core/` | 调度器（`sched/`）、KV Cache 管理（`kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py`） |
| `vllm/v1/executor/` | 执行器：单进程、多进程、Ray |
| `vllm/v1/worker/` | `Worker`（每张 GPU 一个）与 `GPUModelRunner`；`gpu/` 下是实验中的 Model Runner V2 |
| `vllm/v1/attention/` | 注意力后端（FlashAttention、FlashInfer、MLA、Triton……）与后端选择 |
| `vllm/v1/sample/` | 采样器、拒绝采样（投机解码）、logits 处理器 |
| `vllm/v1/spec_decode/` | 投机解码的各种草稿方法：EAGLE、n-gram、Medusa、草稿模型…… |
| `vllm/v1/structured_output/` | 结构化输出（xgrammar、guidance、outlines 后端） |
| `vllm/model_executor/` | 模型代码（`models/`）、并行化的层（`layers/`：线性层、MoE、量化、RoPE……）、权重加载 |
| `vllm/distributed/` | 并行组（`parallel_state.py`）、通信（`device_communicators/`）、PD 分离的 KV 传输（`kv_transfer/`）、EPLB |
| `vllm/compilation/` | torch.compile 集成、自定义融合 pass、CUDA Graph 包装 |
| `csrc/` | C++/CUDA kernel：分页注意力、量化 GEMM、MoE、自定义 all-reduce 等 |

## 进程结构

```text
API server 进程                         EngineCore 进程                    Worker 进程（每张 GPU 一个）
┌──────────────────────────┐   ZMQ    ┌──────────────────────────┐  共享内存   ┌─────────────────────┐
│ FastAPI 路由               │ ───────▶ │ 输入线程 → input_queue    │  消息队列    │ Worker              │
│ OpenAIServingChat         │ msgpack  │ 主线程：run_busy_loop     │ ─────────▶ │  GPUModelRunner     │
│ AsyncLLM                  │          │   Scheduler              │ (广播)      │   模型 + 注意力后端   │
│  ├ InputProcessor（分词）   │ ◀─────── │   KVCacheManager         │ ◀───────── │   Sampler           │
│  └ OutputProcessor（反分词）│          │   Executor               │            └─────────────────────┘
└──────────────────────────┘          │ 输出线程 ← output_queue   │
                                      └──────────────────────────┘
```

- 前端与 EngineCore 之间用 ZMQ 通信，消息用 msgpack 编码（`vllm/v1/serial_utils.py`）。EngineCore 里有专门的输入、输出线程负责收发和编解码，主线程只管调度与执行。
- 单卡时 `UniProcExecutor` 直接在 EngineCore 进程里调用 worker；多卡时 `MultiprocExecutor`（`v1/executor/multiproc_executor.py`）为每张卡起一个 worker 进程，用共享内存消息队列（`distributed/device_communicators/shm_broadcast.py` 中的 `MessageQueue`）把每一步的 `SchedulerOutput` 广播给所有 worker。
- 数据并行（DP）时，每个 DP rank 有自己的 EngineCore（`DPEngineCoreProc`），前端用 `DPLBAsyncMPClient` 做负载均衡。

## 主线一：前端

`AsyncLLM.generate()`（`v1/engine/async_llm.py`）的文档字符串概括了前端做的四件事：

1. 为请求创建一个输出流（`RequestOutputCollector`）；
2. 处理输入：`InputProcessor` 把提示词渲染、分词，构造成 `EngineCoreRequest`；
3. 把请求登记到 `OutputProcessor`（负责反分词、停止字符串、组装 `RequestOutput`）；
4. 通过 `EngineCoreClient` 把请求发给 EngineCore 进程。

另有一个后台的 `output_handler` 协程（`_run_output_handler`）不断从 EngineCore 拉取 `EngineCoreOutputs`，交给 `OutputProcessor.process_outputs`，再放进各请求的输出流。HTTP 层（`entrypoints/openai/chat_completion/serving.py` 的 `OpenAIServingChat`）迭代这个流，生成 SSE 响应。

对应迷你引擎：[`api_server.py`](../engine/sampler-api.md#openai-兼容的流式服务) 的 `AsyncEngine` + `IncrementalDetokenizer`。区别是 vLLM 的引擎核心在另一个进程里。

## 主线二：EngineCore 主循环

`EngineCoreProc.run_busy_loop()`（`v1/engine/core.py`）极其简洁：

```py
while self._handle_shutdown():
    self._process_input_queue()      # 1) 取出新请求、中止请求等（没有工作时阻塞等待）
    self._process_engine_step()      # 2) 执行一步，把输出放进 output_queue
```

一步的内容在 `EngineCore.step()` 中：

```py
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)   # 结构化输出的掩码，与前向并行计算
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

注意前向（`execute_model`）和采样（`sample_tokens`）是分开的两次调用：这样调度器可以在 GPU 做前向的同时，在 CPU 上计算结构化输出的语法掩码。

流水线并行和异步调度时，使用 `step_with_batch_queue()`：维护一个批次队列，**先尽可能调度并提交新批次，再等待最早的批次返回**，让 GPU 不必等 CPU 调度。异步调度（`--async-scheduling`，条件满足时默认开启）由 `AsyncScheduler`（`v1/core/sched/async_scheduler.py`）实现：下一步的调度不等上一步的采样结果回来，而是先为每个请求放一个"占位" token（`num_output_placeholders`），结果回来后再补上。

对应迷你引擎：`LLMEngine.step()`。

## 主线三：调度器与 KV Cache 管理

调度器的算法已在[调度器一章](../engine/scheduler.md)详细对照过，这里补充数据结构：

- **`Request`**（`v1/request.py`）：请求在 EngineCore 中的状态，包括 `num_computed_tokens`、`spec_token_ids`、`block_hashes`、状态机 `RequestStatus`。
- **`SchedulerOutput`**（`v1/core/sched/output.py`）：调度结果。新请求以 `NewRequestData` 完整发送（提示词 token、采样参数、块号）；已经在运行的请求以 `CachedRequestData` 只发送**增量**（新分配的块、新 token、`num_computed_tokens`）。worker 端维护一份"持久批次"（`InputBatch`），只需按增量更新。这样每一步跨进程传输的数据量很小。
- **`KVCacheManager`**（`v1/core/kv_cache_manager.py`）：`get_computed_blocks` 查前缀缓存，`allocate_slots` 为本步分配块，`free` 释放。它下面是 `KVCacheCoordinator` 和按注意力类型区分的 `SingleTypeKVCacheManager` 子类（`FullAttentionManager`、`SlidingWindowManager`、`MambaManager`……）：混合架构的模型（例如部分层是滑动窗口、部分层是线性注意力）会被分成多个 **KV Cache 组**，每组有自己的管理规则。
- **`BlockPool`**（`v1/core/block_pool.py`）：物理块、空闲队列、哈希到块的映射，见[前缀缓存一章](../engine/prefix-cache.md)。

`update_from_output()` 在前向之后运行：把采样出的 token 追加到请求、处理投机解码被拒绝的 token、检查停止条件（EOS、最大长度、停止 token）、释放结束请求的块，并把结果打包成按前端分组的 `EngineCoreOutputs`。

## 主线四：Worker 与 GPUModelRunner

`Worker`（`v1/worker/gpu_worker.py`）的生命周期：

| 方法 | 做什么 |
| --- | --- |
| `init_device` | 设置 GPU、初始化分布式环境（NCCL 通信组） |
| `load_model` | 构建模型并加载权重（`model_executor/model_loader/`） |
| `determine_available_memory` | 用最大批次做一次"性能剖析"前向（`profile_run`），测出峰值显存，剩下的（按 `gpu_memory_utilization` 折算）留给 KV Cache |
| `initialize_from_config` | 按 KV Cache 配置分配 KV 张量（`initialize_kv_cache`） |
| `compile_or_warm_up_model` | torch.compile、录制 CUDA Graph（`capture_model`） |

每一步，`GPUModelRunner.execute_model()`（`v1/worker/gpu_model_runner.py`）做的事情是：

1. `_update_states`：按 `SchedulerOutput` 的增量更新持久批次（加入新请求、移除结束的请求、更新块表）；
2. `_prepare_inputs`：构造 `input_ids`、`positions`、`query_start_loc`、`slot_mapping`、`logits_indices`；
3. 构造注意力元数据（每个后端有自己的 metadata builder），选择 CUDA Graph 模式与填充大小；
4. 在 `set_forward_context(...)` 中执行模型前向。注意力层不通过参数接收元数据，而是从这个"前向上下文"中读取，这样模型代码就不必关心批次结构；
5. 只对 `logits_indices` 计算 logits。

随后 `sample_tokens()` 执行采样（以及投机解码的验证和下一轮草稿的生成）。

对应迷你引擎：`build_batch`（第 2 步）和 `ModelRunner.forward`（第 4、5 步）。

## 主线五：模型代码

vLLM 中 Qwen2 模型（`model_executor/models/qwen2.py`）的注意力和解码层是这样的（节选，Apache-2.0 许可）：

```py
class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, ...)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, ...)
        self.act_fn = SiluAndMul()

class Qwen2Attention(nn.Module):
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

class Qwen2DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

和大模型手册里的 `mini_llm` 对比，每一处不同都是一项推理优化：

| mini_llm | vLLM | 优化 |
| --- | --- | --- |
| `q_proj`、`k_proj`、`v_proj` 三个线性层 | 一个 `QKVParallelLinear`，输出再 `split` | 三次 GEMM 合并为一次；张量并行时按头切分 |
| `gate_proj`、`up_proj` + `silu(g) * u` | `MergedColumnParallelLinear` + `SiluAndMul` | 两次 GEMM 合并；激活与乘法融合成一个 kernel |
| `x = x + attn(norm(x))` | `input_layernorm(hidden_states, residual)` 同时返回归一化结果和新的残差 | 残差相加与 RMSNorm 融合（fused add RMSNorm） |
| `o_proj`、`down_proj` 普通线性层 | `RowParallelLinear` | 张量并行时按行切分，之后 all-reduce |
| 输入 `[B, T]`，拼接连续 KV Cache | 输入是一维 token + `positions`，注意力从前向上下文读取元数据、写分页 KV | 变长批处理、分页 KV |
| `rope_cos_sin` 每次现算 | `get_rope` 返回预先算好 cos/sin 缓存的 RoPE 模块 | 缓存 cos/sin，融合 kernel |

模型类上的 `@support_torch_compile(dynamic_arg_dims={"input_ids": {0: "b"}, ...})` 装饰器声明了哪一维是动态的 token 数，这就是[上一章](../engine/graphs-compile.md)说的"以符号化的 token 数编译一次"。这些并行层的实现见[张量并行一章](../distributed/tensor-parallel.md)。

## 其他模块速查

| 想了解 | 去看 |
| --- | --- |
| 注意力后端的选择 | `v1/attention/selector.py`，各后端在 `v1/attention/backends/` |
| 投机解码 | `v1/spec_decode/eagle.py`（EAGLE/MTP 草稿）、`ngram_proposer.py`，验证在 `v1/sample/rejection_sampler.py` |
| 结构化输出 | `v1/structured_output/`，语法掩码在 EngineCore 中计算（`get_grammar_bitmask`） |
| PD 分离、KV 卸载 | `distributed/kv_transfer/`（KV connector 接口与 NIXL、LMCache 等实现）、`v1/kv_offload/` |
| MoE 与专家并行 | `model_executor/layers/fused_moe/`、`distributed/device_communicators/all2all.py`、`distributed/eplb/` |
| 量化 | `model_executor/layers/quantization/` |
| LoRA | `vllm/lora/`，worker 侧的 `lora_model_runner_mixin.py` |
| 多模态 | `vllm/multimodal/`、调度器中的 encoder cache（`v1/core/encoder_cache_manager.py`） |
| 并行组 | `distributed/parallel_state.py`（TP、PP、DP、EP 组的建立） |

## 建议的阅读顺序

1. 用离线 `LLM` 类、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、`--enforce-eager` 跑一个小模型，在 `Scheduler.schedule` 和 `GPUModelRunner.execute_model` 打断点，单步走一遍；
2. 读 `EngineCore.step` → `Scheduler.schedule` → `update_from_output`，对照本手册的调度器一章；
3. 读 `KVCacheManager` 与 `BlockPool`，对照分页 KV 和前缀缓存两章；
4. 读 `GPUModelRunner._prepare_inputs` 和一个注意力后端（从 `flash_attn.py` 开始），对照变长批处理一章；
5. 读一个模型文件（`qwen2.py` 或 `llama.py`）和 `layers/linear.py`，对照张量并行一章；
6. 按兴趣深入专题：投机解码、PD 分离、MoE。

!!! interview "面试怎么答"
    被问到"讲讲 vLLM 的架构"时，按进程讲最清楚：**前端进程**（HTTP、分词、反分词，`AsyncLLM`）、**EngineCore 进程**（忙循环：调度 → 执行 → 更新，`Scheduler` + `KVCacheManager`）、**worker 进程**（每卡一个，`GPUModelRunner` 执行模型）。进程之间分别用 ZMQ 和共享内存队列通信，调度结果只传增量。再讲一两个你深入读过的模块（例如调度器的统一 token 预算，或 KV 块池的 LRU 设计），比泛泛而谈更有说服力。

## 练习

**1. 找参数。** `max_num_seqs` 在哪里生效？在源码里找到它，并说明它和 `max_num_batched_tokens` 分别限制了什么。

??? success "参考答案"
    在 `Scheduler.__init__` 中读取为 `self.max_num_running_reqs`，在 `schedule()` 调度等待队列时检查 `len(self.running) + ... >= self.max_num_running_reqs` 就停止接收新请求；它限制的是**同时运行的请求数**（也决定了 CUDA Graph 最大录制大小等）。`max_num_batched_tokens` 在 `schedule()` 开头作为 `token_budget`，限制的是**一步计算的 token 总数**。

**2. 加一行日志。** 想在每一步打印"本步调度了多少个请求、多少个 token、KV Cache 使用率"，应该加在哪里？

??? success "参考答案"
    加在 `EngineCore.step()` 中 `schedule()` 之后：`len(scheduler_output.num_scheduled_tokens)` 是请求数，`scheduler_output.total_num_scheduled_tokens` 是 token 数，`self.scheduler.get_kv_cache_usage()` 是使用率。实际上 vLLM 已经通过 `make_stats()` 收集了这些指标，并以 Prometheus 格式在 `/metrics` 接口暴露（`vllm:num_requests_running`、`vllm:kv_cache_usage_perc` 等），生产环境直接看监控即可。

## 小结

- [x] vLLM V1 分三类进程：前端（`AsyncLLM`）、EngineCore（调度 + KV 管理 + 执行器）、worker（`GPUModelRunner`），分别用 ZMQ 和共享内存队列通信。
- [x] 主循环是 `schedule → execute_model → sample_tokens → update_from_output`；异步调度和批次队列让 CPU 调度与 GPU 执行重叠。
- [x] `SchedulerOutput` 只传增量，worker 维护持久批次；注意力层从前向上下文读取元数据。
- [x] 模型代码中的合并投影、融合算子、并行线性层，每一处都对应一项推理优化。
