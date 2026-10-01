# Profiling 推理引擎

<p class="lead">压测告诉你"慢"，profiling 告诉你"慢在哪里"。推理引擎的性能问题分布在很多层：调度与输入准备的 CPU 开销、GPU 上 kernel 之间的空隙、某个 kernel 本身的低效、通信、不必要的同步。这一章给出一套自顶向下的定位方法，在迷你引擎上用 PyTorch profiler 实际定位一个瓶颈，再介绍在 GPU 上对 vLLM、SGLang 做 profiling 的工具与常见症状。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 拿到一个"TPOT 比预期慢一倍"的问题，你按什么顺序排查？
    2. 在 Nsight Systems 的时间线上，GPU 上一段段的空白通常意味着什么？
    3. 怎样让 vLLM 或 SGLang 输出一份 torch profiler 的 trace？
    4. 为什么一个 `.item()` 或 `.tolist()` 就可能让吞吐明显下降？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 自顶向下：先看端到端的指标和理论下限（差多少）→ 服务端的指标（排队、batch 大小、KV 使用率）→ 一步的时间线（CPU 空隙、通信、哪类算子占大头）→ 最后才看单个 kernel。
    2. GPU 在等 CPU：kernel 太小、发射跟不上，或者中间有同步（`.item()`、拷贝），或者 CPU 在做调度、准备元数据。解决办法是 CUDA Graph、融合、去掉同步、让 CPU 的工作和 GPU 的计算重叠。
    3. vLLM：启动时加 `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/abs/path"}'`，然后 `POST /start_profile`、`POST /stop_profile`；SGLang：设置 `SGLANG_TORCH_PROFILER_DIR`，同样调用 `/start_profile` 与 `/stop_profile`，或者在压测命令里加 `--profile`。得到的 trace 用 Perfetto 打开。
    4. 它们会把 GPU 上的值拷回 CPU，必须等 GPU 执行完之前提交的所有工作：CPU 停下来等，GPU 做完后又等 CPU 提交下一步，流水被打断，两边互相空等。

## 自顶向下的方法

1. **端到端指标**：先用压测确认问题（TTFT 还是 TPOT？所有负载都慢，还是高负载才慢？），并与理论下限对比（大模型手册的[延迟下限](llm://inference/estimation/#延迟的下限)）：实测 TPOT 是下限的 1.2 倍还是 3 倍，决定了后续该往哪里找。
2. **服务端指标**：排队请求数、KV Cache 使用率、抢占次数、前缀缓存命中率、每步的 batch 大小。很多"慢"其实是调度与容量问题，与 kernel 无关。
3. **一步的时间线**：用 profiler 看一步 decode 由哪些阶段组成：CPU 上的调度与输入准备、GPU 上的前向、采样、结果处理。GPU 是否一直在忙？
4. **算子与 kernel**：时间主要花在哪些 kernel 上？它们离各自的屋顶线有多远？

![图：自顶向下地找瓶颈——端到端指标、服务端指标、一步的时间线、kernel](../assets/figures/top-down.svg){.aig-svg}

## 在迷你引擎上定位瓶颈

用 `record_function` 给引擎一步中的几个阶段打上标签（不修改引擎代码，只是包一层），然后用 PyTorch profiler 记录 10 步 decode：

```python
import functools
import os
import tempfile
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer
from mini_llm import Transformer
import nano_engine
import runner
from nano_engine import LLMEngine, SamplingParams

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

def traced(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with record_function(name):
            return fn(*args, **kwargs)
    return wrapper

engine = LLMEngine(model, eos_token_id=tok.eos_token_id)
engine.scheduler.schedule = traced("1 调度", engine.scheduler.schedule)
nano_engine.build_batch = traced("2 构造批次", nano_engine.build_batch)
engine.runner.forward = traced("3 前向", engine.runner.forward)
engine.sample_fn = traced("4 采样", engine.sample_fn)
for i in range(16):
    engine.add_request(tok(f"第{i}个问题：介绍一种水果。").input_ids, SamplingParams(max_tokens=40, ignore_eos=True))
for _ in range(3):                                          # 先跑几步：prefill 和预热不计入
    engine.step()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(10):
        engine.step()

stats = {e.key: e for e in prof.key_averages()}
for name in ("1 调度", "2 构造批次", "3 前向", "4 采样"):
    print(f"{name}：每步 {stats[name].cpu_time_total / 10 / 1000:6.2f} ms")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=6))
prof.export_chrome_trace(os.path.join(tempfile.gettempdir(), "nano_engine_trace.json"))   # 可在 ui.perfetto.dev 打开
```

```text
1 调度：每步   0.03 ms
2 构造批次：每步   0.09 ms
3 前向：每步 241.65 ms
4 采样：每步   0.40 ms
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   aten::mm        24.56%     594.859ms        24.59%     595.573ms     302.321us          1970  
                       3 前向        19.75%     478.372ms        99.78%        2.416s     241.649ms            10  
                aten::copy_         8.28%     200.510ms         8.28%     200.510ms       8.118us         24700  
                aten::index         5.86%     141.849ms         6.46%     156.465ms      17.443us          8970  
                  aten::bmm         5.30%     128.391ms         5.39%     130.446ms      14.559us          8960  
               aten::einsum         4.32%     104.613ms        16.53%     400.300ms      44.676us          8960  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.422s
```

几乎所有时间都在前向里，调度和输入准备可以忽略（在 GPU 上，这个比例会完全不同，见下文）。但前向内部的算子表透露了一个问题：`aten::einsum`、`aten::bmm`、`aten::index` 被调用了几千次，远多于矩阵乘法 `aten::mm`。这是[分页 KV 一章](../engine/paged-kv.md#分页注意力)写的参考实现：注意力对批次中的**每个请求**单独循环一次。把注意力单独标出来，看它随 batch 大小怎样变化：

```python
runner.paged_attention = traced("paged_attention", runner.paged_attention)
for batch in (4, 16, 64):
    engine = LLMEngine(model, eos_token_id=tok.eos_token_id, max_num_seqs=64, num_blocks=1024)
    engine.runner.forward = traced("forward", engine.runner.forward)
    for i in range(batch):
        engine.add_request(tok(f"第{i}个问题：介绍一种水果。").input_ids, SamplingParams(max_tokens=40, ignore_eos=True))
    for _ in range(3):
        engine.step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(5):
            engine.step()
    stats = {e.key: e for e in prof.key_averages()}
    forward, attention = stats["forward"].cpu_time_total, stats["paged_attention"].cpu_time_total
    print(f"batch {batch:2d}：每步 {forward / 5 / 1000:5.0f} ms，其中注意力占 {attention / forward:.0%}，"
          f"每步 einsum 调用 {stats['aten::einsum'].count // 5} 次")
```

```text
batch  4：每步    94 ms，其中注意力占 41%，每步 einsum 调用 224 次
batch 16：每步   237 ms，其中注意力占 62%，每步 einsum 调用 896 次
batch 64：每步   715 ms，其中注意力占 81%，每步 einsum 调用 3584 次
```

注意力的调用次数与 batch 大小成正比，它在一步中的占比从 40% 涨到 80% 以上。这就是真实推理引擎必须使用**一个 kernel 处理整个批次**的分页注意力（FlashAttention 的 varlen 接口、FlashInfer、vLLM 的 paged attention kernel）的原因：一次启动，内部按块表为所有请求并行计算。也正是 CUDA Graph 一章说的发射开销问题的一个具体例子。

profiler 导出的 `trace.json` 可以在 Perfetto 中打开，看到每一步里各阶段、各算子的时间线，这比表格更直观。

## 在 GPU 上：工具

| 工具 | 看什么 | 用法 |
| --- | --- | --- |
| PyTorch profiler | Python 调用栈 + CUDA kernel，时间线 | vLLM：`--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/abs/path"}'`，然后 `POST /start_profile`、`POST /stop_profile`；SGLang：设置 `SGLANG_TORCH_PROFILER_DIR`，调用 `/start_profile` 与 `/stop_profile`，或在压测命令中加 `--profile` |
| Nsight Systems（nsys） | 整个系统的时间线：CPU 线程、CUDA API、kernel、内存拷贝、NCCL、NVTX 标记 | `nsys profile -t cuda,nvtx,osrt --trace-fork-before-exec=true --cuda-graph-trace=node -o report vllm serve ...`（多进程需要跟踪 fork，CUDA Graph 内部的 kernel 需要按节点展开） |
| Nsight Compute（ncu） | 单个 kernel 的细节：带宽、算力利用率、占用率、各级缓存命中率、屋顶线 | 对选定的 kernel 采样分析，参见 CUDA 手册的[性能分析：Nsight](cuda://tools/profiling/) |

## 常见症状与原因

| 症状 | 常见原因 | 方向 |
| --- | --- | --- |
| GPU 时间线上 kernel 之间有大量空隙，小 batch 时尤其明显 | CPU 发射开销、调度与输入准备太慢 | CUDA Graph、异步调度、减少 Python 开销（[CUDA Graphs](../engine/graphs-compile.md)） |
| 每一步的开头 GPU 都空闲一段 | 输入准备与 GPU 计算没有重叠；H2D 拷贝没用 pinned memory 或 non_blocking | 异步调度、持久批次、pinned buffer |
| 采样或结果处理后出现一段空隙 | `.item()`、`.tolist()`、`torch.multinomial` 等导致 CPU-GPU 同步 | 把依赖结果的逻辑移到下一步，或者放到 GPU 上 |
| decode 步中注意力占比随上下文增长越来越高 | 读 KV 是瓶颈（这是正常现象），或者 decode kernel 在长上下文下没有做 split-KV | 检查注意力后端；KV 量化；FlashDecoding 式的切分 |
| 每层都有明显的 NCCL all-reduce 时间 | TP 通信延迟（小消息） | custom all-reduce、all-reduce 与 RMSNorm 融合、调整 TP（[张量并行](../distributed/tensor-parallel.md#通信的代价)） |
| MoE 层中 all-to-all 与专家计算串行 | EP 通信没有被重叠 | 两批重叠、DeepEP 低延迟模式（[专家并行](../distributed/expert-parallel.md)） |
| 某些步特别慢，与 prefill 的出现相关 | 长 prefill 与 decode 混批 | 调小分块 prefill 的预算、PD 分离 |
| GEMM 的耗时远高于屋顶线估计 | 形状不佳（batch 太小导致 GEMV）、没用上 Tensor Core、量化 kernel 不合适 | 检查 kernel 名称与形状；换更合适的量化方案 |

!!! source "源码对照"
    - **vLLM**：profiler 配置在 `vllm/config/profiler.py`（`ProfilerConfig`，可选 `torch`、`cuda`、`proton` 三种），HTTP 接口在 `vllm/entrypoints/serve/profile/`；`vllm/profiler/layerwise_profile.py` 可以按层统计耗时。代码中大量使用 `record_function_or_nullcontext("schedule: allocate_slots")` 这样的标签，在 trace 中直接看到调度器各阶段的耗时。
    - **SGLang**：`srt/managers/scheduler_components/profiler_manager.py` 负责启停 profiler，`SGLANG_TORCH_PROFILER_DIR` 指定输出目录；`sglang.benchmark.one_batch` 可以在不启动服务的情况下 profile 单个批次。

!!! interview "面试怎么答"
    被问到"你是怎么做性能优化的"时，最好的回答是一个具体的故事，按"现象 → 假设 → 工具 → 证据 → 改动 → 效果"讲：例如"TPOT 是理论下限的 2 倍 → 怀疑 CPU 开销 → nsys 看到 kernel 之间有空隙、每步开头 GPU 空闲 3 ms → 定位到输入准备中的 Python 循环和一次同步 → 改成向量化、去掉同步 → TPOT 下降 40%"。能说出屋顶线下限作为参照、能说出具体的 trace 特征，会显得非常扎实。本章的迷你引擎例子也可以作为一个小故事来讲。

## 练习

**1. 读 trace。** 在 nsys 的时间线上，你看到 decode 的每一步里，GPU 上的 kernel 紧密排列、几乎没有空隙，但总时间仍是理论下限的 1.8 倍。接下来查什么？

??? success "参考思路"
    GPU 一直在忙，说明不是 CPU 开销的问题，而是 kernel 本身慢了。按耗时排序 kernel：如果 GEMM/GEMV 占大头，用 ncu 看它们的实际带宽利用率（decode 的 GEMV 应该接近峰值带宽的 70%～90%），检查是否用了合适的 kernel（量化格式、split-K）；如果注意力占大头，检查上下文长度与 batch 大小，算一下读 KV 的理论时间，看注意力 kernel 离它有多远；再看是否有意料之外的 kernel（例如多余的拷贝、类型转换、未融合的逐元素操作）。

**2. 同步在哪里？** 为什么采样之后调用 `next_tokens.tolist()` 会影响吞吐？vLLM 是怎样避免这个问题的？

??? success "参考答案"
    `.tolist()` 需要把 GPU 上的数据拷回 CPU，必须等 GPU 完成之前提交的所有工作（同步）。在这段时间里 CPU 什么也做不了，不能提前准备下一步的输入；拷回之后 GPU 又要等 CPU 准备好下一步才能开始，两者交替空闲。vLLM 的异步调度让调度器先用占位 token 调度下一步、提前把下一步的输入准备好，采样结果到达后再补上；worker 端用 pinned memory 与 non_blocking 拷贝，把结果的回传与下一步的计算重叠。SGLang 的重叠调度做的是同一件事。

## 小结

- [x] 自顶向下：端到端指标与理论下限 → 服务端指标 → 一步的时间线 → 单个 kernel。
- [x] 用 `record_function` 给引擎阶段打标签，profiler 的算子表与 trace 能快速暴露问题（本章：逐请求循环的注意力随 batch 线性变慢）。
- [x] GPU 上用 torch profiler（vLLM、SGLang 都内置了开关）、Nsight Systems 看时间线、Nsight Compute 看单个 kernel。
- [x] 常见问题：CPU 开销与空隙、同步、KV 读取、TP/EP 通信、prefill 干扰、kernel 选择不当，各有对应的解决方向。
