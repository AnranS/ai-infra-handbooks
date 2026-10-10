# Benchmarks and ablations

<p class="lead">With every feature written, the last question is what each optimization is actually worth. This chapter gives two load-testing tools, an offline throughput benchmark and an online streaming client, along with a way to run ablations: turn one optimization off and see what happens to throughput and latency. On a CPU we run the two ablations whose trends are clear; for the ones that need a GPU we give the commands and what to watch.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What question does an offline throughput benchmark answer, and what question does an online load test answer?
    2. What affects TTFT and what affects TPOT? Which optimizations mainly improve each?
    3. Why does going from batch size 1 to 16 raise throughput far less on a CPU than on a GPU?
    4. To quantify what overlap scheduling buys, should you use a large model or a small one?

??? success "Answers (try it yourself first, then expand)"
    1. The offline throughput benchmark hands the engine every request at once and measures how many tokens per second it generates at full load (how fast the engine itself is); the online load test sends requests at a given rate and measures TTFT, TPOT and tail latency under that load (whether it meets the SLO).
    2. TTFT: queueing time, the prefill work (prompt length, prefix hit rate), the scheduling policy. TPOT: the time per decode step (batch size, KV length, kernel efficiency, CPU overhead). Prefix caching and chunked prefill mainly improve TTFT; CUDA Graph, overlap scheduling and better attention kernels mainly improve TPOT.
    3. A CPU's ratio of compute to bandwidth differs from a GPU's: on a CPU even a batch of 1 already spends a substantial share on compute, rather than almost entirely on reading weights as a GPU does, so growing the batch brings far less "free" throughput.
    4. A small model: overlap scheduling hides a fixed CPU cost, so the smaller the model and the shorter each step on the GPU, the larger that cost's share and the clearer the gain; on a large model the GPU time is long and the CPU cost was already drowned out.

**Files you will write**: `benchmark/offline.py`, `benchmark/client.py`.

@@tree@@

**This step's main**: `examples/ch21_benchmark.py` — it uses only the files above; `python tools/steps.py check` rebuilds this tree chapter by chapter and runs it.

## Offline throughput {#离线吞吐}

@@code python/minisgl/benchmark/offline.py:run@@

The same approach as the official `benchmark/offline/bench.py` (which came from nano-vllm): generate a number of random requests with input and output lengths uniform within a range, use `ignore_eos=True` so every request generates its full `max_tokens`, and measure the output-token throughput. It measures **how fast the engine can go at full load** and says nothing about any single request's latency.

```bash
python -m minisgl.benchmark.offline --model Qwen/Qwen3-0.6B --num-seqs 256 \
    --max-input-len 1024 --max-output-len 1024 --page-size 256 --cuda-graph-max-bs 256
```

## Online load testing {#在线压测}

@@code python/minisgl/benchmark/client.py:one_request@@

The online load test imitates real users: requests arrive as a Poisson process (exponentially distributed gaps), responses stream, and it records each request's TTFT (from sending to the first chunk), TPOT (the average gap between the chunks after that) and end-to-end latency, reporting the mean, the median and P99. It answers **what the experience is like at a given request rate**.

```bash
python -m minisgl --model Qwen/Qwen3-0.6B &
python -m minisgl.benchmark.client --num-requests 256 --rate 8 --max-tokens 256
```

Raise `--rate` step by step, plot request rate against P99 TTFT / P99 TPOT, and the highest rate that still meets the latency target (the SLO) is this machine's capacity (the method is in the [load testing and capacity planning chapter of the Inference Systems handbook](serving://perf/benchmark/)).

## Two ablations on a CPU {#cpu-上的两个消融}

@@code examples/ch21_benchmark.py@@

@@output ch21_benchmark@@

**Continuous batching**: with the same 16 requests, allowing only 1 to run at a time gives about 21 tokens/s, and allowing 16 gives roughly twice that. On a GPU the gap would be far larger: decode is memory-bound, so going from a batch of 1 to 16 leaves the time per step almost unchanged (reading the weights dominates) and throughput grows nearly linearly, whereas a CPU's compute is much weaker relative to its bandwidth, so compute becomes the bottleneck as soon as the batch grows and the gain is squeezed.

**Prefix caching**: with 16 requests sharing a 400-token prefix, the naive cache computes that prefix 16 times (6656 tokens); the radix cache computes it once, after which each request computes only its own 16 tokens plus the last prefix token (656 in total), and the time drops from about 18 seconds to about 4. What prefix caching buys is proportional to "shared prefix length × request count", which is substantial in multi-turn chat, agents and few-shot prompting.

## The ablation list for a GPU {#gpu-上的消融清单}

| Ablation | Command | What to watch |
| --- | --- | --- |
| overlap scheduling | `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` | offline throughput; the gap is clearest with a small model and a large batch |
| CUDA Graph | `--cuda-graph-max-bs 0` | TPOT; the smaller the batch the larger the gap |
| prefix caching | `--cache-type naive` | TTFT and throughput on a shared-prefix workload |
| attention backend | `--attn fi` / `--attn fa` / `--attn fa,fi` | prefill throughput (FA3 is strong on Hopper), decode TPOT (FlashInfer is strong) |
| chunk size | `--max-prefill-length 2048` / `16384` | TTFT on long prompts and the TPOT spike during decode |
| page size | `--page-size 1` / `16` / `256` | prefix hit rate (coarser with bigger pages) against management overhead |

Change one thing at a time in an ablation and leave everything else alone; run each configuration three times and take the median. To see where the time actually goes, capture a timeline with Nsight Systems: the gaps between CPU scheduling and GPU kernels are exactly what overlap scheduling and CUDA Graph exist to remove (the method is in the [profiling chapter of the Inference Systems handbook](serving://perf/profiling/)).

!!! upstream "The official implementation"
    - offline: [`benchmark/offline/bench.py`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/benchmark/offline/bench.py) (256 requests, 100-1024 input and output, `page_size=256`, `cuda_graph_max_bs=256`)
    - online: [`benchmark/online/bench_qwen.py`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/benchmark/online/bench_qwen.py), which replays the public Qwen request trace from Alibaba Cloud Bailian; the client at @@upstream benchmark/client.py@@ is about 500 lines and reports more statistics
    - the official README gives offline throughput and online latency results on an H200 (the online test compared against SGLang)

## Tests {#测试}

@@code tests/test_ch21_benchmark.py:test_offline_benchmark_runs@@

!!! interview "How to explain it"
    On benchmarking: an offline throughput test measures how many tokens per second the engine handles at full load, while an online load test measures TTFT, TPOT and tail latency (P99) at a given request rate, and the two measure different things. TTFT depends on queueing and the prefill work (prompt length, prefix hits), so chunked prefill, prefix caching and prefill/decode disaggregation mainly improve it; TPOT depends on the time per decode step, so CUDA Graph, overlap scheduling, quantization and speculative decoding mainly improve it. Ablations change one thing at a time; overlap scheduling and CUDA Graph pay off most with small models and small batches, where the CPU's share is high, so a small model is the right one to quantify them. On a CPU, going from batch 1 to 16 raises throughput only a little because the CPU's compute is already saturated rather than memory-bound as on a GPU.

## Exercises {#练习}

1. Use `benchmark/client.py` against a `--max-running-requests 4` service on a CPU and raise `--rate` from 0.5 to 4, watching TTFT. Explain where the knee appears.
2. Design a "multi-turn chat" workload: each user sends 5 turns in a row, with each prompt containing all the earlier turns. Compare radix against naive on TTFT under that load.
3. Reproduce the official offline benchmark on a GPU with overlap scheduling and CUDA Graph each turned off, recording throughput. Which matters more for Qwen3-0.6B? What about Qwen3-14B? Why?

??? success "Answers"
    1. Once the arrival rate exceeds the service's capacity (the throughput `max_running_req` concurrent requests can provide), requests pile up in the waiting queue and TTFT jumps from "prefill time" to "queueing plus prefill time" and keeps growing with time. That is queueing theory's latency explosion as utilization approaches 1.
    2. In multi-turn chat each turn's prompt prefix is the previous turn's complete conversation, so the radix cache hits almost the whole prefix and TTFT depends only on the new part, while naive prefills from scratch every turn and TTFT grows linearly with the turn number.
    3. A small model's GPU time per step is very short, so the CPU cost and kernel-launch overhead are a large share and both optimizations show clear gains; a large model's step is long, the overheads are a small share, and the gains shrink. Which one matters more depends on the batch size: the smaller the batch, the larger the share of launch overhead that CUDA Graph removes.

## Summary {#小结}

- [x] Offline throughput measures engine speed at full load; online load testing measures TTFT, TPOT and tail latency at a given request rate.
- [x] A CPU is enough to see what continuous batching and prefix caching buy; the former is far clearer on a GPU.
- [x] Ablations change one thing at a time, and overlap scheduling and CUDA Graph pay off most with small models and small batches.
