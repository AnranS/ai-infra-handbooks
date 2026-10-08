# System design problems

<p class="lead">"Design a large-model inference service" is a common big question for inference roles, especially platform-leaning ones. It has no standard answer; it tests whether you can translate requirements into numbers, numbers into configuration, and configuration into an operable system, explaining every trade-off. This chapter gives an answering framework, demonstrates it on one complete example (with code for the capacity estimate), and finally lists the key points of other common problems.</p>

## The answering framework {#答题框架}

1. **Clarify the requirements** (ask before answering): the model and its size, traffic (peak QPS, the distributions of prompt/output length, whether prefixes are shared), the SLO (P99 TTFT, P99 TPOT/ITL), availability, cost constraints, features (streaming, tool calling, multimodal, LoRA);
2. **Capacity estimate**: how many prefill and decode tokens/s are needed → how much one instance can take under the SLO → how many instances and how many GPUs;
3. **Single-instance design**: the inference engine, parallelism, quantization, KV management, scheduling parameters;
4. **Multi-instance architecture**: the gateway, routing (load-aware, cache-aware), PD disaggregation, the KV cache layer, elastic scaling;
5. **Reliability and operations**: health checks, failover, rate limiting and degradation, model releases, cold starts;
6. **Monitoring and cost**: key metrics, cost per million tokens, directions for optimization.

Interview time is limited, so giving **a decision + a reason + a number** at each step matters more than covering everything.

## Example: a chat service for a 70B model {#例题一个-70b-模型的对话服务}

> Design an inference service for a chat product: the model is a 70B dense model (LLaMA-3-70B scale), peak 200 QPS, average prompt 2000 tokens (including the system prompt and multi-turn history), average output 500 tokens, requiring P99 TTFT < 2 s, P99 TPOT < 50 ms and 99.9% availability.

### 1. Clarifications and assumptions {#1-澄清与假设}

First confirm a few things that would change the design: whether multi-turn history is kept on the server (which decides the prefix cache hit rate), whether there are very long context requests (which decides whether a separate resource pool is needed), the daily traffic swing (which decides the elasticity policy), and whether quantization is allowed (which decides the hardware needed). Here we assume: FP8 quantization is allowed, prefix sharing is moderate, and requests above 32K occur occasionally.

### 2. Capacity estimate {#2-容量估算}

Demand: prefill 200 × 2000 = 400 thousand tokens per second, decode 200 × 500 = 100 thousand tokens per second.

One instance: a 70B model's FP8 weights are about 70 GB, so take 8 H100s with TP=8 (NVLink within a machine), and an FP8 KV Cache of 160 KB per token. Use the simulator from [the load-testing chapter](../perf/benchmark.md) to estimate the request rate such an instance sustains under the SLO (8 GPUs' bandwidth and compute combined, MFU at 40%, with a fixed cost of 1.5 ms per step to approximate TP communication and scheduling overhead):

```python
import math
from sim import Setup, simulate, summarize

kv_per_token = 2 * 80 * 8 * 128 * 1                                  # FP8 KV
instance = Setup(params=70.6e9, weight_bytes=70.6e9, kv_bytes_per_token=kv_per_token,
                 kv_capacity_tokens=int((8 * 80e9 * 0.9 - 70.6e9) / kv_per_token),
                 peak_flops=8 * 1979e12, bandwidth=8 * 3.35e12, mfu=0.4, step_overhead=1.5e-3,
                 max_num_batched_tokens=8192, max_num_seqs=512)
low = summarize(simulate(instance, 1, 300, 2000, 500), 2.0, 0.05)
print(f"低负载：TTFT {low['ttft_p50'] * 1e3:.0f} ms，TPOT {low['tpot_p50'] * 1e3:.1f} ms")

lo, hi = 1.0, 60.0
for _ in range(12):
    mid = (lo + hi) / 2
    ok = summarize(simulate(instance, mid, 1500, 2000, 500), 2.0, 0.05)["slo_ok"] >= 0.99
    lo, hi = (mid, hi) if ok else (lo, mid)
instances = math.ceil(200 / lo * 1.3)                                  # leave 30% headroom for bursts and failures
print(f"单实例满足 SLO 的最大速率约 {lo:.1f} req/s；200 QPS 需要 {math.ceil(200 / lo)} 个实例，"
      f"加 30% 余量为 {instances} 个（{instances * 8} 张 H100）")

gpu_hour_cost = 2.5                                                    # assume $2.5 per H100 per hour
output_tokens_per_hour = 200 * 500 * 3600
print(f"按每卡每小时 ${gpu_hour_cost} 计：每小时 ${instances * 8 * gpu_hour_cost:.0f}，"
      f"约合每百万输出 token ${instances * 8 * gpu_hour_cost / output_tokens_per_hour * 1e6:.2f}")
```

```text
低负载：TTFT 48 ms，TPOT 4.3 ms
单实例满足 SLO 的最大速率约 16.8 req/s；200 QPS 需要 12 个实例，加 30% 余量为 16 个（128 张 H100）
按每卡每小时 $2.5 计：每小时 $320，约合每百万输出 token $0.89
```

You don't need to run a simulator in an interview, but you should be able to state the same reasoning: at low load TPOT is about 4 ms (reading 70 GB of weights / 26.8 TB/s), and the bottleneck at high load is prefill and decode competing for compute; one instance handles about 15–20 req/s; 200 QPS needs a dozen-odd instances and over a hundred GPUs; the cost is on the order of a dollar per million tokens. **Always add afterwards: this is an estimate, to be calibrated with real-workload load tests before going live.**

### 3. Single-instance design {#3-单实例设计}

| Decision | Choice | Reason |
| --- | --- | --- |
| Engine | vLLM or SGLang | mature, supporting every feature below; choose by the team's familiarity |
| Parallelism | TP=8 (within a machine) | 70B in FP8 doesn't fit on one GPU; TP lowers single-request latency and meets the 50 ms TPOT |
| Quantization | FP8 weights + FP8 KV | roughly doubles capacity ([quantization in deployment](../perf/quantization-deploy.md#对容量的影响)); run accuracy evaluations before going live |
| Scheduling | chunked prefill, budget about 4K–8K | balances TTFT against ITL; tune by load testing |
| Caching | prefix caching + a CPU tier | the system prompt and multi-turn history are reused, lowering TTFT and compute |
| CUDA Graphs | on | decode's CPU overhead |
| Speculative decoding | on depending on load (EAGLE) | lowers latency off-peak; at peak with large batches the gain vanishes or turns negative, so it can be turned off dynamically |

### 4. Multi-instance architecture {#4-多实例架构}

<!-- i18n:diagram 666f0fa3b3 -->
```text
client → API gateway (auth, rate limiting, quotas, billing)
           │
           ▼
        router ─────── instance state (queue length, KV usage, cached prefixes)
   (load-aware + cache-aware + session affinity)
      ┌────┼─────────────┬──────────────┐
      ▼    ▼             ▼              ▼
  instance 1 … N    long-context pool   (optional) PD disaggregation: prefill pool + decode pool
      │                                  │
      └──────── distributed KV cache (Mooncake / LMCache) ───┘
      monitoring: TTFT/TPOT/ITL percentiles, goodput, queues, KV usage, preemptions, hit rate
```

Key points:

- **Routing**: send a session's requests to the same instance where possible (session affinity), send requests sharing a prefix to the instance that already cached it (cache-aware), while avoiding hotspots (load-aware); the three must be weighted and balanced;
- **Isolating long contexts**: put requests above 32K in a separate resource pool (larger TP or context parallelism) so they don't drag down ordinary requests' TTFT and ITL;
- **PD disaggregation**: worth considering at this scale (a dozen-odd instances), with prefill and decode scaled separately to remove interference; but it adds complexity and KV transfer, so launch with chunked prefill first and introduce it only after confirming TPOT jitter is the main problem;
- **Elastic scaling**: scale on queue length, KV usage and the trend of TTFT rather than GPU utilization (naturally modest during decode); a 70B model's cold start (loading weights, compiling, capturing CUDA Graphs) takes minutes, so keep hot standby instances and optimize weight loading (a local NVMe cache, parallel loading).

### 5. Reliability {#5-可靠性}

- Health checks must not just look at the process being alive; they must send real probe requests;
- When an instance fails, streaming requests in progress break off: the client retries, or the gateway continues from the content already generated as a prefix;
- Rate limiting and degradation: when over capacity, prioritize paying users and interactive requests (priority scheduling), and defer offline tasks to off-peak;
- Model releases: gradual rollout shifting traffic by ratio, comparing quality and latency metrics between versions, with a fast rollback kept available.

### 6. Monitoring and continuous optimization {#6-监控与持续优化}

Watch these metrics: P50 and P99 of TTFT/TPOT/ITL, goodput, queue length, KV usage, preemption count, prefix cache hit rate, and cost per million tokens. Directions for optimization, ordered by payoff: raise the prefix cache hit rate → quantization → scheduling parameters → PD disaggregation → speculative decoding (off-peak) → upgrade the hardware.

## Other common problems {#其他常见题目}

All 10 system design problems in the sprint plan have complete reference answers (including runnable capacity estimates): see [reference answers (part one)](design-answers-1.md) and [reference answers (part two)](design-answers-2.md).

| Problem | Key points |
| --- | --- |
| Offline batch inference (labeling a billion tokens a day) | latency doesn't matter, only throughput and cost: the largest batches, speculative decoding off, bucketing by length, the cheapest hardware and the most aggressive quantization; exploit prefix sharing (identical instructions); checkpointing and retries |
| An RL rollout service | the long tail (partial rollout, asynchrony), memory switching and weight sync when co-located with training, returning sampling logprobs, batch invariance (see [inference in RL training](../topics/rl-rollout.md)) |
| A multi-tenant LoRA platform (thousands of fine-tuned versions) | a shared base, batched multi-LoRA computation (segmented GEMM), tiered caching and on-demand loading of LoRA weights, routing by tenant to raise LoRA hits, quota isolation |
| An agent platform | multi-turn, ever-growing context, tool calling: prefix caching and session affinity are crucial, tiered KV caching (KV can be offloaded while the user thinks and tools run), structured output, long-context isolation |
| A RAG service (long input, short output) | prefill-dominated: FP8, prefix caching (put shared documents at the front of the prompt), chunked prefill, more prefill instances than decode; speculative decoding gains little |
| Deploying a very large MoE (DeepSeek scale) | DP Attention + EP, DeepEP, EPLB, two-batch overlap, PD disaggregation with different EP sizes on each side (see [expert parallelism](../distributed/expert-parallel.md)) |
| A multimodal service | resolution caps, an encoder cache, EPD disaggregation, the CPU bottleneck of preprocessing (see [multimodal inference](../topics/multimodal.md)) |

!!! interview "How to explain it"
    The most common ways to lose points on system design are **no numbers** and **no trade-offs**. Anyone can say "deploy with vLLM, add PD disaggregation, add caching"; what interviewers want to hear is "one instance handles about 17 req/s, so 16 instances are needed; we won't add PD disaggregation yet, because chunked prefill already meets the ITL in load tests, and introducing it would add KV transfer and operational complexity". Another bonus is proactively discussing **how to verify**: every estimate must end in a load test.

## Summary {#小结}

- [x] The framework: clarify requirements → capacity estimate → single instance → multiple instances → reliability → monitoring and cost.
- [x] Capacity estimate: the prefill/decode token rates needed, divided by one instance's capacity under the SLO, plus redundancy; estimate one instance's capacity with the roofline or a simulator, then calibrate with load tests.
- [x] The keys for multiple instances are routing (session affinity, cache-aware, load-aware), isolating long requests, scaling on queueing rather than GPU utilization, and hot standby for cold starts.
- [x] Every decision needs a reason and a number, along with the trade-offs and how to verify it.
