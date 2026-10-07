# The life of a request

<p class="lead">This chapter is the map of the whole handbook. From the moment the user presses Enter to the last character appearing on screen, a request passes through the HTTP server, the chat template and tokenization, the scheduler, KV cache management, model execution, sampling and detokenization, and is finally streamed back. First walk the whole path once to see who is responsible for each stop, what it is called in vLLM and SGLang, and which chapter of this handbook covers it; then trace a few real requests in the mini engine and break TTFT and TPOT into concrete slices of time.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Which components does a chat request pass through, in order, from reaching the server to returning its first token? Which run on the CPU and which on the GPU?
    2. What segments of time make up TTFT? Which one usually grows under high concurrency?
    3. In vLLM and SGLang, are the scheduler and the HTTP server in the same process? Why?

??? success "Answers (try first, then expand to compare)"
    1. HTTP server (parse the request) → chat template and tokenization → engine core (scheduler queueing, KV allocation, prefix cache lookup) → build the batch → model forward (GPU) → sampling (GPU) → detokenization and stop checks → stream back. Only the forward pass and sampling are on the GPU; everything else is on the CPU.
    2. Queueing time + prefill compute time + CPU overhead (tokenization, scheduling, preparing metadata). Under high concurrency it is usually the queueing time that grows.
    3. No: putting CPU work such as HTTP and tokenization together with scheduling and the forward pass makes them compete for the CPU through the GIL, leaving the GPU waiting; split apart, the frontend can scale independently and the engine's main loop is never interrupted.

## From Enter to the last character {#从回车到最后一个字}

| # | What happens | vLLM V1 | SGLang | This handbook |
| --- | --- | --- | --- | --- |
| 1 | The HTTP request arrives, OpenAI-format parameters are parsed | `entrypoints/openai/api_server.py` (FastAPI) | `srt/entrypoints/http_server.py` | [Sampler and API](sampler-api.md) |
| 2 | Apply the chat template, tokenize | `renderers/`, `InputProcessor` | `TokenizerManager` | as above |
| 3 | Hand the request to the engine core (across processes) | `AsyncLLM` → `EngineCoreClient` (ZMQ) | ZMQ to the `Scheduler` process | as above |
| 4 | Look up the prefix cache, enter the waiting queue | `Scheduler` + `KVCacheManager` | `Scheduler` + `RadixCache` | [Prefix caching](prefix-cache.md) |
| 5 | Get scheduled: allocate KV blocks, decide how many tokens to compute this step | `Scheduler.schedule()` | `get_next_batch_to_run()` | [Scheduler](scheduler.md), [paged KV](paged-kv.md) |
| 6 | Build batch metadata, send to the GPU worker | `SchedulerOutput` → `GPUModelRunner._prepare_inputs` | `ScheduleBatch` → `ForwardBatch` | [Variable-length batching](batch-layout.md) |
| 7 | Forward: prefill (possibly over several steps) | model code + attention backend + CUDA Graph | same | [CUDA Graphs](graphs-compile.md), [tensor parallelism](../distributed/tensor-parallel.md) |
| 8 | Sample the first token | `Sampler` | `Sampler` | [Sampler](sampler-api.md) |
| 9 | Detokenize, check stop conditions, stream back the first character | `OutputProcessor` in the frontend process | `DetokenizerManager` | as above |
| 10 | Decode one token per step (batched with other requests) | repeat 5–9 | repeat 5–9 | [Scheduler](scheduler.md) |
| 11 | Finish: free the KV blocks (their content stays in the prefix cache) | `Scheduler.finish_requests` | `cache_finished_req` | [Prefix caching](prefix-cache.md) |

![Figure: a request's journey through an inference engine](../assets/figures/request-life.svg){.aig-svg}

A few facts worth remembering:

- **Only steps 7 and 8 are on the GPU**; everything else is CPU work. So many inference engine optimizations reduce or hide CPU overhead: multiple processes, asynchronous scheduling, CUDA Graphs.
- **Steps 5–9 repeat for every token**: a request that generates 500 tokens goes through scheduling 500 times. The scheduler handles a whole batch each time, not a single request.
- **TTFT = the time of steps 1–9**: queueing (between step 4 and step 5), prefill computation (step 7, possibly over several steps), and various CPU processing. **TPOT = the time of each repetition of steps 5–9**, mainly set by the GPU time of one decode step, but stretched by prefills inserted into the same batch.

## Tracing requests in the mini engine {#在迷你引擎里跟踪请求}

Run a small scenario on the mini engine built in the next few chapters: three short requests arrive first, and after they start decoding, a long request of over a hundred tokens arrives. With a token budget of 64, the long request's prefill is cut into several chunks. Wrap the scheduler to record what the long request goes through at each step:

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
def traced_schedule():                                    # record how many tokens each request gets at each step
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

You can clearly see that:

- The long request is scheduled the step after it arrives (no queueing, since there is plenty of room in concurrency and memory);
- Its prefill is cut into several chunks, and each step can use only the tokens left from "the budget minus what decode uses";
- The step that computes the last chunk of the prefill samples the first token, so TTFT equals the total time of these steps;
- The short requests' average TPOT is stretched: during the steps of the long request's prefill, they can only slow down along with it;
- The three short requests' prompts are each only a dozen or so tokens, so the first step's 64-token budget fits them all, and their TTFTs are the same; with slightly longer prompts, the third request would be pushed to the next step and its TTFT would grow by a whole step.

In a real service, TTFT often also includes **queueing time**: when concurrency is too high or memory runs short, requests wait in the queue for others to finish. Under load testing this is the segment most likely to rise steeply with load, and it is the main source of SLO violations (see [the load testing chapter](../perf/benchmark.md)).

## How this handbook is organized {#这本手册的结构}

<!-- i18n:diagram fa255b191f -->
```text
The life of a request (this chapter)
│
├── Building an inference engine from scratch: paged KV → variable-length batching → scheduler → prefix caching → sampling and API → CUDA Graphs
│     each chapter implements one component on the mini engine and checks it token by token against a reference
├── Source walkthroughs: vLLM V1, SGLang
│     map every component of the mini engine to the files and functions of the real code
├── Distributed inference: tensor parallelism → expert parallelism and DP attention → pipeline and context parallelism → PD disaggregation → hierarchical KV caching
├── Performance engineering: load testing and capacity planning → profiling → quantized deployment
├── Advanced topics: speculative decoding, long context and sparse attention, structured output, multimodality, inference in RL
└── Job hunting: interview bank, coding questions, system design, portfolio, hardware cheat sheet
```

We suggest reading the six chapters of "building an inference engine from scratch" and running the code yourself before the source walkthroughs: with something of your own to compare against, hundreds of thousands of lines of real code are no longer a maze.

!!! interview "In an interview"
    "What happens between a user sending a request and receiving the first token?" is one of the most common opening questions for inference roles. Walk through this chapter's table from top to bottom, naming the component responsible for each step (ideally with its name in vLLM or SGLang), then break TTFT into "queueing + prefill + CPU overhead" and name the optimizations for each segment: for queueing, scheduling and capacity; for prefill, prefix caching, chunked prefill and PD disaggregation; for CPU overhead, multiple processes, asynchronous scheduling and CUDA Graphs.

## Exercises {#练习}

**1. Timing a request.** A request is scheduled after queueing for 300 ms, its 1024-token prefill takes 120 ms, and it then decodes together with other requests at 25 ms per step, outputting 200 tokens in total. What are its TTFT, TPOT and end-to-end latency? What if, during decode, 10 steps are each stretched by 60 ms by someone else's prefill?

??? success "Answer"
    TTFT ≈ 300 + 120 = 420 ms (ignoring CPU processing such as tokenization and detokenization); after the first token there are 199 more, so TPOT = 25 ms; end to end ≈ 420 + 199 × 25 = 5395 ms. When stretched, end to end grows by 600 ms to 5995 ms, and TPOT = (199 × 25 + 600) / 199 ≈ 28 ms: only 3 ms more on average, but the ITL of those 10 steps is 85 ms, and the user clearly feels the stutter. So load tests must look at both TPOT and the tail of ITL (see [load testing, SLOs and capacity planning](../perf/benchmark.md)).

**2. Finding it in the source.** Using this chapter's table: where do you start reading the code that schedules a request and allocates its KV blocks, in vLLM and in SGLang? And in which process are the streamed tokens turned into strings?

??? success "Answer"
    Scheduling and KV allocation: in vLLM V1, start from `Scheduler.schedule()`, with KV blocks allocated by `KVCacheManager`; in SGLang, start from the scheduler's `get_next_batch_to_run()`, with KV and prefix reuse managed by `RadixCache` and friends. Detokenization: vLLM does it in the frontend process's `OutputProcessor`, and SGLang in a separate `DetokenizerManager` process. Both keep detokenization out of the process that drives the GPU, so CPU work runs in parallel with GPU computation.

## Summary {#小结}

- [x] A request passes through HTTP, tokenization, scheduling, KV management, the forward pass, sampling and detokenization; only the forward pass and sampling are on the GPU.
- [x] The loop from scheduling to detokenization repeats once per generated token; TTFT includes queueing, prefill and CPU overhead, and TPOT is set by the time per step.
- [x] vLLM and SGLang both split HTTP/tokenization and the engine core into different processes; the components have different names but correspond one to one in responsibility.
