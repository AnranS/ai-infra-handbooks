# 一个请求的一生

<p class="lead">这一章是整本手册的地图。从用户按下回车，到最后一个字出现在屏幕上，一个请求要经过 HTTP 服务、对话模板与分词、调度器、KV Cache 管理、模型执行、采样、反分词，最后流式返回。先把这条路完整走一遍，看清每一站由谁负责、在 vLLM 和 SGLang 中叫什么、在本手册的哪一章展开；再在迷你引擎里跟踪几个真实请求，把 TTFT、TPOT 拆成具体的时间片段。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个聊天请求从到达服务器到返回第一个 token，依次经过哪些组件？哪些在 CPU 上，哪些在 GPU 上？
    2. TTFT 由哪几段时间组成？高并发时通常是哪一段变长？
    3. 在 vLLM 和 SGLang 中，调度器和 HTTP 服务是否在同一个进程里？为什么？

## 从回车到最后一个字

| # | 发生了什么 | vLLM V1 | SGLang | 本手册 |
| --- | --- | --- | --- | --- |
| 1 | HTTP 请求到达，解析 OpenAI 格式的参数 | `entrypoints/openai/api_server.py`（FastAPI） | `srt/entrypoints/http_server.py` | [采样器与 API](sampler-api.md) |
| 2 | 应用对话模板、分词 | `renderers/`、`InputProcessor` | `TokenizerManager` | 同上 |
| 3 | 请求交给引擎核心（跨进程） | `AsyncLLM` → `EngineCoreClient`（ZMQ） | ZMQ 发给 `Scheduler` 进程 | 同上 |
| 4 | 查前缀缓存，进入等待队列 | `Scheduler` + `KVCacheManager` | `Scheduler` + `RadixCache` | [前缀缓存](prefix-cache.md) |
| 5 | 被调度：分配 KV 块，决定本步算多少 token | `Scheduler.schedule()` | `get_next_batch_to_run()` | [调度器](scheduler.md)、[分页 KV](paged-kv.md) |
| 6 | 构造批次元数据，发给 GPU worker | `SchedulerOutput` → `GPUModelRunner._prepare_inputs` | `ScheduleBatch` → `ForwardBatch` | [变长批处理](batch-layout.md) |
| 7 | 前向：prefill（可能分多步）| 模型代码 + 注意力后端 + CUDA Graph | 同左 | [CUDA Graphs](graphs-compile.md)、[张量并行](../distributed/tensor-parallel.md) |
| 8 | 采样得到第一个 token | `Sampler` | `Sampler` | [采样器](sampler-api.md) |
| 9 | 反分词、检查停止条件，流式发回第一个字 | 前端进程的 `OutputProcessor` | `DetokenizerManager` | 同上 |
| 10 | 每一步 decode 一个 token（与其他请求组批） | 重复 5～9 | 重复 5～9 | [调度器](scheduler.md) |
| 11 | 结束：释放 KV 块（内容留在前缀缓存里） | `Scheduler.finish_requests` | `cache_finished_req` | [前缀缓存](prefix-cache.md) |

![图：一个请求在推理引擎里的旅程](../assets/figures/request-life.svg){.aig-svg}

几个值得记住的事实：

- **只有第 7、8 步在 GPU 上**，其余全是 CPU 工作。所以推理引擎的很多优化都是在减少或隐藏 CPU 开销：多进程、异步调度、CUDA Graph。
- **第 5～9 步每一步都要重复**，一个请求生成 500 个 token，就要经过 500 次调度。调度器每次处理的是整个批次，而不是单个请求。
- **TTFT = 第 1～9 步的时间**：排队等待（第 4 步到第 5 步之间）、prefill 计算（第 7 步，可能分成多步）、以及各种 CPU 处理。**TPOT = 每次重复 5～9 步的时间**，主要由 decode 一步的 GPU 时间决定，但会被同一批次中插入的 prefill 拉长。

## 在迷你引擎里跟踪请求

用前面几章写成的迷你引擎跑一个小场景：三个短请求先到，开始 decode 之后，来了一个一百多 token 的长请求。token 预算设为 64，长请求的 prefill 会被切成几块。给调度器套一层记录，看长请求每一步经历了什么：

```python
import time
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer
from nano_engine import LLMEngine, SamplingParams

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

def chat(q):
    return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)).input_ids

engine = LLMEngine(model, eos_token_id=tok.eos_token_id, max_num_batched_tokens=64)
events = {}
schedule = engine.scheduler.schedule
def traced_schedule():                                    # 记录每个请求每一步被分到了多少 token
    out = schedule()
    for req, n in out.scheduled:
        events.setdefault(req.request_id, []).append((len(engine.step_log), time.perf_counter(), req.num_computed, n))
    return out
engine.scheduler.schedule = traced_schedule

shorts = [engine.add_request(chat(q), SamplingParams(max_tokens=12, ignore_eos=True))
          for q in ["你好", "今天星期几？", "讲个笑话"]]
for _ in range(3):
    engine.step()
long_req = engine.add_request(chat("请总结：" + "调度器决定每一步计算哪些请求。" * 14), SamplingParams(max_tokens=6, ignore_eos=True))
while engine.scheduler.has_unfinished():
    engine.step()

t0, n_prompt = long_req.arrival_time, len(long_req.prompt_ids)
print(f"长请求：提示词 {n_prompt} 个 token")
for step, t, computed, n in events[long_req.request_id]:
    kind = f"prefill 第 {computed}～{computed + n - 1} 个 token" if computed < n_prompt else "decode 1 个 token"
    print(f"  第 {step:2d} 步  +{(t - t0) * 1000:6.0f} ms  {kind}")
print(f"  第一个 token 出现在 +{(long_req.token_times[0] - t0) * 1000:.0f} ms")

print("请求    TTFT      平均 TPOT")
for r in shorts + [long_req]:
    tpot = (r.token_times[-1] - r.token_times[0]) / (len(r.token_times) - 1)
    print(f"{'长' if r is long_req else '短'} {r.request_id}  {(r.token_times[0] - r.arrival_time) * 1000:6.0f} ms  {tpot * 1000:6.0f} ms")
```

```text
长请求：提示词 141 个 token
  第  3 步  +     0 ms  prefill 第 0～60 个 token
  第  4 步  +   147 ms  prefill 第 61～121 个 token
  第  5 步  +   295 ms  prefill 第 122～140 个 token
  第  6 步  +   454 ms  decode 1 个 token
  第  7 步  +   540 ms  decode 1 个 token
  第  8 步  +   620 ms  decode 1 个 token
  第  9 步  +   699 ms  decode 1 个 token
  第 10 步  +   779 ms  decode 1 个 token
  第一个 token 出现在 +454 ms
请求    TTFT      平均 TPOT
短 0     128 ms      95 ms
短 1     127 ms      95 ms
短 2     127 ms      95 ms
长 3     454 ms      81 ms
```

可以清楚地看到：

- 长请求到达后，下一步就被调度了（没有排队，因为并发和显存都还宽裕）；
- 它的 prefill 被切成几块，每步只能用上"预算减去 decode 占用"的那部分 token；
- prefill 最后一块算完的那一步，就采样出了第一个 token，TTFT 等于这几步的总时间；
- 短请求的平均 TPOT 被拉长了：长请求 prefill 的那几步，它们只能跟着慢下来；
- 三个短请求的提示词都只有十几个 token，第一步 64 个 token 的预算能同时装下，所以它们的 TTFT 相同；如果提示词再长一些，第三个请求就会被推迟到下一步，TTFT 多出整整一步的时间。

在真实的服务里，TTFT 往往还包含一段**排队时间**：并发太高、显存不够时，请求要在等待队列里等其他请求结束。这一段在压测中最容易随负载陡增，是 SLO 超标的主要来源（见[压测一章](../perf/benchmark.md)）。

## 这本手册的结构

```text
一个请求的一生（本章）
│
├── 从零写一个推理引擎：分页 KV → 变长批处理 → 调度器 → 前缀缓存 → 采样与 API → CUDA Graphs
│     每章在迷你引擎上实现一个组件，并与参考结果逐 token 核对
├── 源码导读：vLLM V1、SGLang
│     把迷你引擎的每个组件对应到真实代码的文件与函数
├── 分布式推理：张量并行 → 专家并行与 DP Attention → 流水线与上下文并行 → PD 分离 → KV 分层缓存
├── 性能工程：压测与容量规划 → Profiling → 量化部署
├── 进阶专题：投机解码、长上下文与稀疏注意力、结构化输出、多模态、RL 中的推理
└── 求职：面试题库、手撕代码、系统设计、作品集、硬件速查
```

建议先读完"从零写一个推理引擎"的六章并亲手运行代码，再读源码导读：有了自己写过的对照，几十万行的真实代码就不再是迷宫。

!!! interview "面试怎么答"
    "从用户发出请求到收到第一个 token，中间发生了什么？"是推理岗最常见的开场题之一。按本章的表格从头讲到尾，每一步点出负责的组件（最好能说出 vLLM 或 SGLang 中的名字），再把 TTFT 拆成"排队 + prefill + CPU 开销"，并说出每一段对应的优化：排队看调度和容量，prefill 看前缀缓存、分块 prefill 和 PD 分离，CPU 开销看多进程、异步调度和 CUDA Graph。

## 小结

- [x] 一个请求依次经过 HTTP、分词、调度、KV 管理、前向、采样、反分词，只有前向和采样在 GPU 上。
- [x] 调度到反分词的循环每生成一个 token 重复一次；TTFT 包含排队、prefill 与 CPU 开销，TPOT 由每步时间决定。
- [x] vLLM 与 SGLang 都把 HTTP/分词和引擎核心拆到不同进程；组件名称不同，职责一一对应。
