# The scheduler: continuous batching, chunked prefill and preemption

<p class="lead">The scheduler is the brain of an inference engine. At every step it must decide which requests to compute, how many tokens to compute for each, and whom to sacrifice when memory runs short. This chapter follows vLLM V1's approach to implement a complete scheduler and engine main loop: continuous batching, chunked prefill and recompute-based preemption, and verifies that under all kinds of scheduling parameters the output is token-for-token identical to generating each request on its own.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why is vLLM V1's scheduler said to have "no prefill phase and no decode phase"? What does it do at each step?
    2. What does raising or lowering `max_num_batched_tokens` affect?
    3. When memory runs short, which request does vLLM preempt? How does the preempted request recover later? How does SGLang's approach differ?
    4. Why does a step in which preemption happens stop accepting new requests?

??? success "Answers (try first, then expand to compare)"
    1. It does not distinguish phases; each step does one thing: within the token budget, let each request's computed tokens (`num_computed_tokens`) catch up with its total tokens; prefill, chunked prefill and decode differ only in "how far behind", and can appear in the same step.
    2. Raising it: each step can process more prefill tokens, lowering TTFT and raising throughput, but each step gets slower, so requests in decode see larger, jumpier ITL / TPOT; lowering it does the opposite: decode is smooth, but long prompts' TTFT grows.
    3. vLLM preempts from the tail of the running queue (the most recently scheduled, lowest priority), frees its KV, and later re-prefills it as a waiting request (recomputation); SGLang estimates future demand when forming batches and admits new requests conservatively, and when memory runs short it retracts some decode requests, putting them back in the waiting queue.
    4. Preemption means memory is already insufficient; admitting new requests would only get them preempted right away, wasting computation back and forth; this step first lets the running requests advance and free up space.

![Figure: static batching vs. continuous batching](../assets/figures/continuous-batching.svg){.aig-svg}

<!-- comic ../assets/comics/scheduler.webp is in Chinese; put it back once the English version exists -->

## A unified token budget {#统一的-token-预算}

The core idea of vLLM V1's scheduler is written in the comment at the top of `Scheduler.schedule()`:

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`. At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`. This is general enough to cover chunked prefills, prefix caching, speculative decoding, and the "jump decoding" optimization in the future.

The previous chapter showed why this works: in one forward pass, prefill, chunked prefill and decode are just different combinations of "cached length" and "this step's length". So scheduling becomes one problem: **within the token budget (`max_num_batched_tokens`), assign each request the number of tokens to compute this step.**

- A new request: `num_computed = 0`, needing `len(prompt)` tokens; if the budget is short it gets only part of them, which is **chunked prefill**;
- Chunked prefill in progress: keep giving it the rest;
- Decode: `num_tokens - num_computed = 1`, give it 1;
- Resuming after preemption: `num_computed` is reset to zero, and the prompt and all generated tokens are recomputed;
- A prefix cache hit: before scheduling, `num_computed` is set directly to the hit length ([next chapter](prefix-cache.md));
- Speculative decoding: `num_tokens` includes the draft tokens, so several are given at once.

Switch between the timelines of the three scheduling methods, and it is obvious why static batching does not work and what chunked prefill rescues:

<div class="aig-widget" data-widget="contbatch"></div>

## Implementation {#实现}

```python title="nano_engine.py"
"""nano_engine.py —— 一个最小但完整的推理引擎：调度器 + 模型执行器 + 采样。

调度思路与 vLLM V1 相同：没有"prefill 阶段"和"decode 阶段"之分，每个请求只记录
num_computed（KV 已在缓存中的 token 数）和 num_tokens（提示词 + 已生成）。每一步，调度器在
token 预算内给请求分配 token，让 num_computed 追上 num_tokens。分块 prefill、decode、抢占后的重算
都是这个规则的特例。
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import torch

from paged import BlockPool, PagedKVCache
from runner import ModelRunner, build_batch


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0          # 0 means greedy
    top_p: float = 1.0
    top_k: int = 0                    # 0 means no limit
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Request:
    def __init__(self, request_id: str, prompt_ids: list[int], params: SamplingParams):
        self.request_id, self.prompt_ids, self.params = request_id, list(prompt_ids), params
        self.output_ids: list[int] = []
        self.num_computed = 0            # tokens already written to the KV cache
        self.block_ids: list[int] = []   # block table
        self.status = Status.WAITING
        self.finish_reason: str | None = None
        self.num_preemptions = 0
        self.num_cached_tokens = 0       # tokens hit in the prefix cache
        self.generator = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
        self.arrival_time = time.perf_counter()
        self.token_times: list[float] = []

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass
class SchedulerOutput:
    scheduled: list[tuple[Request, int]]   # (request, tokens computed for it this step)
    num_preempted: int


class Scheduler:
    def __init__(self, pool: BlockPool, block_size: int, max_num_batched_tokens: int = 256,
                 max_num_seqs: int = 16, enable_chunked_prefill: bool = True):
        self.pool, self.block_size = pool, block_size
        self.max_num_batched_tokens, self.max_num_seqs = max_num_batched_tokens, max_num_seqs
        self.enable_chunked_prefill = enable_chunked_prefill
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        scheduled: list[tuple[Request, int]] = []
        num_preempted = 0

        # 1. schedule running requests first (decode or unfinished chunked prefill), in arrival order
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed, budget)
            while not self._allocate(req, req.num_computed + n):
                victim = self.running.pop()            # out of memory: preempt the lowest-priority (latest-arrived) request
                self._preempt(victim)
                num_preempted += 1
                if victim is req:
                    break
            else:
                scheduled.append((req, n))
                budget -= n
                i += 1
                continue
            break                                      # the current request itself was preempted; stop scheduling the running queue this round

        # 2. then take new requests from the waiting queue (none if anything was preempted this step, to avoid thrashing)
        while self.waiting and budget > 0 and not num_preempted and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            self._lookup_prefix_cache(req)
            n = req.num_tokens - req.num_computed
            if not self.enable_chunked_prefill and n > budget:
                self._release_prefix_hit(req)
                break
            n = min(n, budget)
            if not self._allocate(req, req.num_computed + n):
                self._release_prefix_hit(req)
                break
            self.waiting.popleft()
            req.status = Status.RUNNING
            self.running.append(req)
            scheduled.append((req, n))
            budget -= n
        return SchedulerOutput(scheduled, num_preempted)

    def _allocate(self, req: Request, num_tokens: int) -> bool:
        """保证请求的块表能容纳 num_tokens 个 token。"""
        need = math.ceil(num_tokens / self.block_size) - len(req.block_ids)
        if need <= 0:
            return True
        blocks = self.pool.allocate(need)
        if blocks is None:
            return False
        req.block_ids += blocks
        return True

    def _preempt(self, req: Request) -> None:
        """重算式抢占：释放全部块，回到等待队列最前面；下次调度时从头计算（提示词 + 已生成的 token）。"""
        self.pool.free(req.block_ids)
        req.block_ids, req.num_computed = [], 0
        req.status = Status.WAITING
        req.num_preemptions += 1
        self.waiting.appendleft(req)

    def _lookup_prefix_cache(self, req: Request) -> None:
        if getattr(self.pool, "enable_caching", False) and req.num_computed == 0:
            # leave at least the last token as a miss: a real forward pass is needed to get the next token's logits
            hit = self.pool.lookup(req.token_ids[: req.num_tokens - 1])
            self.pool.touch(hit)
            req.block_ids = hit
            req.num_computed = req.num_cached_tokens = len(hit) * self.block_size

    def _release_prefix_hit(self, req: Request) -> None:
        if req.block_ids:
            self.pool.free(req.block_ids)
            req.block_ids, req.num_computed, req.num_cached_tokens = [], 0, 0

    def finish(self, req: Request, reason: str) -> None:
        req.status, req.finish_reason = Status.FINISHED, reason
        self.running.remove(req)
        self.pool.free(req.block_ids)
        req.block_ids = []


def greedy_sample(logits: torch.Tensor, reqs: list[Request]) -> list[int]:
    return logits.argmax(-1).tolist()


class LLMEngine:
    def __init__(self, model, eos_token_id: int | None = None, num_blocks: int = 256, block_size: int = 16,
                 max_num_batched_tokens: int = 256, max_num_seqs: int = 16, enable_chunked_prefill: bool = True,
                 pool: BlockPool | None = None, sample_fn=greedy_sample):
        cfg = model.cfg
        self.block_size, self.eos_token_id, self.sample_fn = block_size, eos_token_id, sample_fn
        self.pool = pool if pool is not None else BlockPool(num_blocks)
        self.kv = PagedKVCache(cfg.num_hidden_layers, self.pool.num_blocks, block_size, cfg.num_key_value_heads, cfg.hd)
        self.runner = ModelRunner(model, self.kv)
        self.scheduler = Scheduler(self.pool, block_size, max_num_batched_tokens, max_num_seqs, enable_chunked_prefill)
        self.step_log: list[dict] = []
        self._next_id = 0

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> Request:
        req = Request(str(self._next_id), prompt_ids, params or SamplingParams())
        self._next_id += 1
        self.scheduler.add_request(req)
        return req

    def step(self) -> list[Request]:
        """调度 → 构造批次 → 前向 → 采样 → 更新状态。返回本步产生了新 token 的请求。"""
        out = self.scheduler.schedule()
        if not out.scheduled:
            return []
        items, sampled = [], []
        for req, n in out.scheduled:
            need_logits = req.num_computed + n == req.num_tokens   # sample only once the last token has been computed
            items.append((req.token_ids[req.num_computed:req.num_computed + n], req.num_computed, req.block_ids,
                          need_logits))
            if need_logits:
                sampled.append(req)
        logits = self.runner.forward(build_batch(items, self.block_size))
        next_tokens = self.sample_fn(logits, sampled) if sampled else []

        for req, n in out.scheduled:
            req.num_computed += n
            if hasattr(self.pool, "cache_blocks"):
                self.pool.cache_blocks(req.token_ids, req.block_ids, req.num_computed)
        now = time.perf_counter()
        for req, tok in zip(sampled, next_tokens):
            req.output_ids.append(tok)
            req.token_times.append(now)
            p = req.params
            if (tok == self.eos_token_id and not p.ignore_eos) or tok in p.stop_token_ids:
                self.scheduler.finish(req, "stop")
            elif len(req.output_ids) >= p.max_tokens:
                self.scheduler.finish(req, "length")
        self.step_log.append({"num_tokens": sum(n for _, n in out.scheduled), "num_reqs": len(out.scheduled),
                              "num_prefill_tokens": sum(n for r, n in out.scheduled if n > 1),
                              "num_preempted": out.num_preempted, "free_blocks": self.pool.num_free(),
                              "waiting": len(self.scheduler.waiting)})
        return sampled

    def generate(self, prompts: list[list[int]], params=None) -> list[list[int]]:
        """params 可以是一个 SamplingParams（所有请求共用），也可以是每个请求一个的列表。"""
        if not isinstance(params, list):
            params = [params] * len(prompts)
        reqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        while self.scheduler.has_unfinished():
            self.step()
        return [r.output_ids for r in reqs]
```

`schedule()` has two passes:

1. **Schedule the running requests first** (in arrival order). They already hold KV cache, so letting them continue first avoids wasting computation already invested. Before giving a request tokens, make sure its block table can hold the new tokens; if no blocks can be allocated, preempt a request from the **end** of the running queue (the latest arrival, lowest priority) and free its blocks, until allocation succeeds or the preempted request is the current one itself.
2. **Then admit new requests from the waiting queue**, until the budget runs out, the concurrency limit `max_num_seqs` is reached, or no blocks can be allocated. If preemption happened in this step, no new requests are admitted: memory is already tight, and admitting more would only cause more preemption next step, i.e. thrashing.

`step()` is one iteration of the engine's main loop: schedule → build the batch → forward → sample → update state and check stop conditions. Only requests that "have computed up to their last token" need sampling; the middle segments of chunked prefill only write the KV cache.

## Check: however it is scheduled, the result is the same {#验证怎样调度结果都一样}

Run 6 chat requests under three configurations: the default parameters, a very small token budget (forcing chunked prefill), and very few KV blocks (forcing preemption). Each configuration's output is compared with `mini_llm.generate` generating each request on its own:

```python
import time
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer, generate
from nano_engine import LLMEngine, SamplingParams

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

def chat(q):
    return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)).input_ids

prompts = [chat(q) for q in ["什么是 KV Cache？", "用一句话解释连续批处理。", "Python 的 GIL 是什么？",
                             "写一个关于月亮的比喻。", "Explain tensor parallelism in one sentence.", "1+1 等于几？请直接回答。"]]
t0 = time.perf_counter()
reference = [generate(model, torch.tensor([p]), 24, eos_token_id=tok.eos_token_id)[0].tolist() for p in prompts]
print(f"逐个单独生成：{time.perf_counter() - t0:.1f} s")

configs = {"默认": {}, "预算 32 个 token": {"max_num_batched_tokens": 32},
           "只有 10 个 KV 块": {"max_num_batched_tokens": 64, "num_blocks": 10}}
for name, kw in configs.items():
    engine = LLMEngine(model, eos_token_id=tok.eos_token_id, **kw)
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, SamplingParams(max_tokens=24))
    preempted = sum(s["num_preempted"] for s in engine.step_log)
    print(f"{name:12s} {len(engine.step_log):3d} 步，抢占 {preempted} 次，{time.perf_counter() - t0:.1f} s，"
          f"与单独生成一致：{outputs == reference}")
    assert outputs == reference
```

```text
逐个单独生成：6.0 s
默认            24 步，抢占 0 次，2.4 s，与单独生成一致：True
预算 32 个 token  27 步，抢占 0 次，2.6 s，与单独生成一致：True
只有 10 个 KV 块  36 步，抢占 2 次，2.9 s，与单独生成一致：True
```

The output is identical under all three configurations, while batching more than halves the total time. This directly demonstrates that "scheduling affects only performance, not results" (provided floating-point error does not change the argmax; see the LLM book's [which optimizations change the output](llm://synthesis/token-journey/#哪些优化会改变输出)).

## A scheduling trace {#调度轨迹}

See what happens in the first few steps with a token budget of 32:

```python
engine = LLMEngine(model, eos_token_id=tok.eos_token_id, max_num_batched_tokens=32)
engine.generate(prompts, SamplingParams(max_tokens=24))
print("步  token 数  请求数  其中 prefill token  等待队列")
for i, s in enumerate(engine.step_log[:6]):
    print(f"{i:2d}  {s['num_tokens']:6d}  {s['num_reqs']:6d}  {s['num_prefill_tokens']:12d}  {s['waiting']:8d}")
```

```text title="输出"
步  token 数  请求数  其中 prefill token  等待队列
 0      32       2            32         4
 1      32       4            31         2
 2      32       5            29         1
 3      31       6            27         0
 4       6       6             0         0
 5       6       6             0         0
```

Each prompt is 16–25 tokens and the budget is only 32: step 0 finishes the first request's prompt and uses the remaining budget on part of the second request; after that, each request already decoding takes 1 token per step, and the remaining budget advances the prefill chunks of the later requests. By step 4, all 6 requests are decoding, using just 6 tokens per step. Decode and prefill chunks "ride along" in the same step, which is exactly what Sarathi-Serve proposed.

## Trading off the token budget {#token-预算的取舍}

`max_num_batched_tokens` sets the most tokens a step can compute. It directly affects two metrics:

- A large budget: a long prompt finishes in one step, so TTFT is short; but that step is slow, and the decode requests in the same batch must wait for it, producing one very long gap between tokens (a TPOT spike);
- A small budget: a long prompt is cut into many chunks and no step is slow, so decode is smooth; but the prompt takes many steps to finish, so TTFT grows, and each step's matrix multiplications get smaller, lowering GPU utilization.

An experiment: 4 short requests are decoding when a long request of 786 tokens arrives. Compare the short requests' largest gap between tokens and the long request's TTFT under different budgets:

```python
short = [chat(q) for q in ["介绍一下杭州。", "介绍一下成都。", "介绍一下西安。", "介绍一下广州。"]]
long_prompt = tok("请总结下面这段文字：" + "推理引擎需要在吞吐和延迟之间做权衡。" * 60).input_ids
for budget in (2048, 256, 64):
    engine = LLMEngine(model, eos_token_id=tok.eos_token_id, max_num_batched_tokens=budget, num_blocks=512)
    shorts = [engine.add_request(p, SamplingParams(max_tokens=40, ignore_eos=True)) for p in short]
    for _ in range(4):
        engine.step()
    long_req = engine.add_request(long_prompt, SamplingParams(max_tokens=8, ignore_eos=True))
    while engine.scheduler.has_unfinished():
        engine.step()
    max_gap = max(max(b - a for a, b in zip(r.token_times, r.token_times[1:])) for r in shorts)
    ttft = long_req.token_times[0] - long_req.arrival_time
    print(f"预算 {budget:5d}：短请求最大 token 间隔 {max_gap * 1000:5.0f} ms，长请求 TTFT {ttft * 1000:5.0f} ms")
```

```text
预算  2048：短请求最大 token 间隔   747 ms，长请求 TTFT   747 ms
预算   256：短请求最大 token 间隔   413 ms，长请求 TTFT  1200 ms
预算    64：短请求最大 token 间隔   203 ms，长请求 TTFT  2469 ms
```

The smaller the budget, the smoother decode is, but the longer the long request's TTFT (these are CPU numbers; the trend on a GPU is the same). In real deployments this value is usually several thousand to over ten thousand. vLLM 0.30's default depends on the GPU: 16384 for GPUs with at least 160 GB of memory (B200 and the like); 8192 for online serving and 16384 for offline inference on H100/H200; 2048 for online serving and 8192 for offline inference on other GPUs. The exact value should be set by load testing against the SLO (see [the load testing chapter](../perf/benchmark.md)).

## Preemption: two approaches in vLLM and SGLang {#抢占vllm-与-sglang-的两种思路}

| | vLLM V1 | SGLang |
| --- | --- | --- |
| Admitting new requests | admit as long as blocks can be allocated for the current step | estimate future memory demand (remaining output length × `new_token_ratio`) and admit only with headroom |
| When memory runs short | preempt the request at the end of the running queue (or the lowest priority), free all its blocks, put it at the front of the waiting queue, and recompute later | when decode finds memory short, `retract_decode` withdraws some requests and raises `new_token_ratio` so later admissions are more conservative |
| Preempted requests | `num_computed_tokens` reset to zero, the prompt + generated content all recomputed; prefix caching often makes the recomputation much cheaper | also recomputed; the retracted KV can go into the radix tree cache |

Each approach has pros and cons: vLLM is more aggressive, with higher concurrency normally and an occasional preemption cost; SGLang is more conservative and rarely retracts requests, but admits somewhat fewer when its estimate runs high. Early vLLM also supported swapping a preempted request's KV out to CPU memory (swap); V1 removed it: recomputation together with prefix caching is usually faster, and much simpler to implement.

!!! source "Source code"
    - **vLLM**: the structure of `Scheduler.schedule()` (`vllm/v1/core/sched/scheduler.py`) matches this chapter's implementation: it first iterates over `self.running`, allocating blocks with `kv_cache_manager.allocate_slots`, and on failure preempts `self.running[-1]` (the lowest priority under priority scheduling); `_preempt_request` sets `num_computed_tokens` to 0 and puts the request at the front of `self.waiting`; only when `not preempted_reqs` does it schedule the waiting queue. Related parameters: `max_num_batched_tokens` (the token budget), `max_num_seqs` (the concurrency limit) and `long_prefill_token_threshold` (the most tokens one request may compute in a step).
    - After each forward pass, `update_from_output` writes the sampled results back to the requests, checks stop conditions and frees finished requests' blocks. `EngineCore.step()` (`vllm/v1/engine/core.py`) chains it all together: `schedule → execute_model → sample_tokens → update_from_output`.
    - **SGLang**: `Scheduler.get_next_batch_to_run` (`srt/managers/scheduler.py`) first tries to form a new prefill batch (`get_new_batch_prefill`, where `PrefillAdder` admits requests by the remaining token budget and `chunked_prefill_size` controls chunking), and only if none can be formed lets the running batch continue decoding (`update_running_batch`, calling `retract_decode` when memory runs short). With `--enable-mixed-chunk`, prefill chunks and decode are merged into one `MIXED` batch.

!!! interview "In an interview"
    When asked "how is continuous batching implemented", don't just say "finished requests leave and new ones join at any time"; make three points:

    1. **The scheduling granularity is one iteration**: the batch is re-decided every step, rather than running a request from start to finish;
    2. **A unified token budget**: prefill and decode are scheduled with the same budget, long prompts are chunked and mixed with decode in the same step;
    3. **Memory management**: the paged KV cache is allocated on demand, and when memory runs short requests are preempted and recomputed.

    Add one trade-off (how the token budget relates to TTFT/TPOT) and one comparison (vLLM's vs. SGLang's preemption strategies), and the answer is complete.

## Exercises {#练习}

**1. Priority scheduling.** Add a `priority` field to `Request` (smaller numbers first) and modify the scheduler: the waiting queue dequeues by priority, and when preemption is needed, the lowest-priority request in the running queue is preempted. What new problem does this bring?

??? success "Approach"
    Turn the waiting queue into a heap ordered by `(priority, arrival_time)`; when preempting, pick `max(running, key=lambda r: (r.priority, r.arrival_time))`, which is exactly what vLLM's `SchedulingPolicy.PRIORITY` does. The new problem is **starvation**: a steady stream of high-priority requests can keep low-priority ones from ever being scheduled, or get them preempted and recomputed again and again. The common mitigation is "aging": the longer a request waits, the higher its effective priority.

**2. Budget and concurrency limit.** What happens if `max_num_seqs = 256` but `max_num_batched_tokens = 128`?

??? success "Answer"
    In decode each request needs 1 token per step, so the budget lets only 128 requests decode at once; the other 128, even though they are in the running queue, get no tokens this step and can only advance in turns, doubling TPOT. On top of that, new requests' prefills can hardly squeeze in. So `max_num_batched_tokens` must be at least larger than `max_num_seqs`, and usually much larger, to leave room for prefill.

## Summary {#小结}

- [x] vLLM V1's scheduler has no phases: within the token budget, it lets each request's `num_computed` catch up with `num_tokens`.
- [x] Running requests are scheduled first, then new ones admitted; when memory runs short, requests are preempted from the tail and recomputed, and no new requests are admitted in a step with preemption.
- [x] Chunked prefill lets decode and prefill chunks proceed in the same steps: a small budget gives smooth decode and longer TTFT, a large one the opposite.
- [x] vLLM admits aggressively and preempts on demand; SGLang estimates future demand, admits conservatively and retracts when necessary.
