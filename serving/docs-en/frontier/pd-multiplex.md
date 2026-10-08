# PD multiplexing: splitting prefill and decode by SM on one GPU

<p class="lead">Prefill and decode on the same GPU interfere with each other: when a long prompt's prefill slips in, requests in decode wait for it that step. Chunked prefill cuts the interference into small pieces, and PD disaggregation puts the two on different GPUs altogether. There is a third path: split by SM on the same GPU, so prefill and decode each use part of the SMs and run at the same time. CUDA's green contexts make a stream execute only on specified SMs, and SGLang's PD multiplexing (PD-Multiplexing) is built on them. This chapter explains what green contexts can and cannot isolate, reads SGLang's implementation (how it splits SMs, how it chooses a split, and why it cuts prefill by layer), then uses an estimation model to compare its ITL and prefill time against chunked prefill.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. On the same GPUs, what ways are there for prefill and decode to coexist? What does each cost?
    2. What is a green context? Which resources can it isolate, and which can't it?
    3. How does SGLang's PD multiplexing decide how many SMs decode gets?
    4. Why does PD multiplexing cut prefill by layer, computing only a few layers per round?
    5. When the same long prefill slips in, how do ITL and that request's prefill time differ between PD multiplexing and chunked prefill?

??? success "Answers (try first, then expand to compare)"
    1. Alternating in time: prefill first (decode stalls for the whole stretch) or chunked prefill (one chunk of prefill mixed into each step, so every decode step slows down); splitting in space: on the same GPU, divide the SMs between prefill and decode, which run at the same time (PD multiplexing), at the cost of each side getting only part of the compute while memory bandwidth and L2 are still shared; on different GPUs (PD disaggregation): no interference, but KV must be transferred, more GPUs are needed, and the xPyD ratio must be tuned.
    2. A lightweight CUDA context bound to a subset of SMs; kernels from streams created on it execute only on those SMs. It isolates SMs (compute units, registers, shared memory); it cannot isolate memory bandwidth, the L2 cache and other shared resources, and the documentation states plainly that kernels on different green contexts are not guaranteed to actually run concurrently.
    3. At startup it cuts the SMs into several levels (on H100, prefill : decode from 112 : 20 to 72 : 60, in steps of 8 SMs, with decode getting at least 16), plus two groups of ordinary streams, "all to prefill" and "all to decode"; at runtime it picks a level by the number of requests in decode: `level = decode requests × number of split levels // decode_bs_divisor` (36 by default), clamped to the split range, so the more decode requests, the more SMs decode gets; with only decode or only prefill, it uses the whole GPU.
    4. Every round of the scheduling loop must sync the decode stream and process decode results; if prefill computed all layers at once, the round would be stretched long by it, and the split could not follow decode's load. Cut by layer, each round submits only `split_forward_token_budget // prefill tokens` layers (budget 65536 by default), prefill and decode advance alternately, and the split changes level once the prefill ends.
    5. Under PD multiplexing decode always has its own SMs, and ITL is only slightly slower than with the whole GPU (9 ms becomes 9–15 ms in this chapter's estimate), while chunked prefill's ITL becomes 19–74 ms during the prefill; the cost is that prefill uses only part of the compute and takes longer (291 ms becomes 480 ms in the estimate).

## Three ways to coexist {#三种共存方式}

| | Alternating in time (prefill first / chunked prefill) | Splitting in space (PD multiplexing) | On different GPUs (PD disaggregation) |
| --- | --- | --- | --- |
| Interference with decode | prefill first: stalls for the whole stretch; chunked: every step slows down | small: it has its own SMs, competing with prefill only for bandwidth and L2 | none |
| Prefill time of new requests | fastest (whole GPU) | longer (only part of the SMs) | depends on the prefill instances' load |
| KV transfer | not needed | not needed (same memory) | needed (RDMA / NVLink) |
| Deployment scale | from one GPU | from one GPU | at least one prefill instance + one decode instance |
| Main tuning knobs | token budget, chunk size | split level, layers of prefill per round | xPyD ratio, routing |

The [scheduler](../engine/scheduler.md) and [PD disaggregation](../distributed/pd-disagg.md) chapters covered the first and last. PD multiplexing sits between them: no extra GPUs to buy, no KV to transfer, and decode is still not stalled by prefill. The hardware capability it relies on is the green context.

![Figure: three ways for prefill and decode to coexist on one GPU](../assets/figures/pd-multiplex-sm.svg){.aig-svg}

## Green contexts: carving out SMs for a stream {#green-context给流划一块-sm}

A **green context** is a lightweight context offered by the driver API since CUDA 12.4: it owns only part of the device's resources (currently mainly SMs). Kernels submitted to streams created on it execute only on those SMs. Using it takes five steps (driver API; CUDA 13's runtime API now has corresponding functions such as `cudaGreenCtxCreate`):

```cpp
CUdevResource all, parts[2], rest;
cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM);             // 1. get the whole GPU's SM resource
unsigned n = 1;
cuDevSmResourceSplitByCount(parts, &n, &all, &rest, 0, 104);            // 2. split off 104 SMs; the remainder goes to rest
CUdevResourceDesc desc;
cuDevResourceGenerateDesc(&desc, &parts[0], 1);                         // 3. generate a resource descriptor
CUgreenCtx g;
cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);           // 4. create the green context
CUstream prefill_stream;
cuGreenCtxStreamCreate(&prefill_stream, g, CU_STREAM_NON_BLOCKING, 0);  // 5. create a stream on it: this stream's kernels use only these 104 SMs
// build the second green context and decode_stream from rest the same way
```

A few limits to be clear about:

- **Splits have a granularity**: on compute capability 7.x and 8.x the SM count must be a multiple of 2, and on 9.0 and above a multiple of 8 (according to the notes in the CUDA headers);
- **Only SMs are isolated**: memory bandwidth, the L2 cache, copy engines and work queues are still shared. CUDA's documentation is explicit: even when two green contexts have disjoint SMs, kernels on them are not guaranteed to actually execute concurrently; CUDA 13 additionally provides a work queue configuration (`cudaDevWorkqueueConfigScopeGreenCtxBalanced`) to keep submissions from different green contexts from blocking each other as far as possible;
- **Not a strict cap**: the documentation lists two cases where a kernel uses more SMs than allocated (when MPS limits the thread percentage; and on Hopper, when a module using dynamic parallelism is loaded, 2 extra SMs are used), but never fewer.

Decode is memory-bound and gets close to the bandwidth limit as long as enough SMs issue memory requests at once; prefill is compute-bound, and more SMs mean more speed. So "a small share of SMs to decode, the rest to prefill" costs nothing in bandwidth, which is the starting point of PD multiplexing.

## SGLang's PD multiplexing {#sglang-的-pd-复用}

SGLang turns PD multiplexing on with `--enable-pdmux`, with a configuration file given by `--pdmux-config-path` (YAML, with the fields `sm_group_num`, `manual_divisions`, `split_forward_token_budget` and `decode_bs_divisor`). The code is in `srt/multiplex/`: `pdmux_context.py` splits the SMs and creates the streams, and `multiplexing_mixin.py` is the scheduling loop `event_loop_pdmux`. Below, its splitting and selection rules are written out in Python as is (checked against `divide_sm` in SGLang 0.5.20 for every combination of 40–200 SMs and compute capability 7–9, with identical results):

```python title="pdmux_policy.py"
"""pdmux_policy.py —— SGLang PD 复用（srt/multiplex/pdmux_context.py、multiplexing_mixin.py）的切分与选择规则。"""

ARCH = {6: (1, 1), 7: (2, 2), 8: (4, 2), 9: (8, 8)}     # compute capability major version → (minimum SMs per part, SM granularity)


def divide_sm(total_sms, major, groups):
    """候选的 (prefill SM, decode SM) 切分：prefill 不少于一半，decode 至少 16 个，按粒度取值，均匀挑 groups 个"""
    min_per_part, multiple = ARCH[major]
    cand = [x for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
            if x >= total_sms - x and total_sms - x >= 16]
    if len(cand) >= groups:
        cand = cand[::max(1, len(cand) // groups)][:groups]
    return [(x, total_sms - x) for x in reversed(cand)]            # splits giving prefill more come first


def stream_groups(total_sms, major, sm_group_num=8):
    """第 0 组：全部 SM 给 prefill（普通流）；中间 sm_group_num-2 组：green context 切分；最后一组：全部给 decode"""
    return [(total_sms, 0)] + divide_sm(total_sms, major, sm_group_num - 2) + [(0, total_sms)]


def choose(groups, decode_bs, has_prefill, decode_bs_divisor=36):
    """调度器按正在 decode 的请求数选一组：decode 越多，分给 decode 的 SM 越多"""
    n = len(groups)
    if decode_bs and has_prefill:
        return max(1, min(n - 2, decode_bs * (n - 2) // decode_bs_divisor))
    return n - 1 if decode_bs else 0


def prefill_layers_per_step(extend_tokens, num_layers, token_budget=65536):
    """prefill 按层切开：每轮只算 token_budget // extend_tokens 层（至少 1 层），和 decode 交替推进"""
    return min(num_layers, max(1, token_budget // extend_tokens))
```

```python
from pdmux_policy import choose, prefill_layers_per_step, stream_groups

for name, sms, major in (("H100 SXM", 132, 9), ("A100", 108, 8), ("H20", 78, 9)):
    print(f"{name}（{sms} 个 SM）：", stream_groups(sms, major))
groups = stream_groups(132, 9)
print("H100 上 decode 请求数 → 选中的组 (prefill SM, decode SM)：")
for bs in (1, 6, 12, 18, 24, 30, 36, 64):
    k = choose(groups, bs, has_prefill=True)
    print(f"  {bs:3d} → 第 {k} 组 {groups[k]}")
print("只有 decode：", groups[choose(groups, 32, False)], " 只有 prefill：", groups[choose(groups, 0, True)])
for tokens in (512, 8192, 32768, 131072):
    print(f"prefill {tokens:6d} 个 token：每轮 {prefill_layers_per_step(tokens, 36):2d} 层（共 36 层）")
```

```text title="output"
H100 SXM（132 个 SM）： [(132, 0), (112, 20), (104, 28), (96, 36), (88, 44), (80, 52), (72, 60), (0, 132)]
A100（108 个 SM）： [(108, 0), (84, 24), (78, 30), (72, 36), (66, 42), (60, 48), (54, 54), (0, 108)]
H20（78 个 SM）： [(78, 0), (56, 22), (48, 30), (40, 38), (0, 78)]
H100 上 decode 请求数 → 选中的组 (prefill SM, decode SM)：
    1 → 第 1 组 (112, 20)
    6 → 第 1 组 (112, 20)
   12 → 第 2 组 (104, 28)
   18 → 第 3 组 (96, 36)
   24 → 第 4 组 (88, 44)
   30 → 第 5 组 (80, 52)
   36 → 第 6 组 (72, 60)
   64 → 第 6 组 (72, 60)
只有 decode： (0, 132)  只有 prefill： (132, 0)
prefill    512 个 token：每轮 36 层（共 36 层）
prefill   8192 个 token：每轮  8 层（共 36 层）
prefill  32768 个 token：每轮  2 层（共 36 层）
prefill 131072 个 token：每轮  1 层（共 36 层）
```

- **Splitting**: `divide_sm` picks, among values meeting the granularity, splits where prefill gets at least half and decode at least 16 SMs, taking `sm_group_num - 2` levels evenly (6 by default). For each level, `sgl_kernel.spatial.create_greenctx_stream_by_value` creates a pair of green-context streams; two groups of ordinary streams are added at the ends: group 0 gives the whole GPU to prefill, and the last group gives the whole GPU to decode;
- **Choosing a level** (`adjust_stream_groups`): with both decode and prefill present, the level is chosen linearly by the number of requests in decode, with 32 requests choosing 80 : 52; `manual_divisions` in the configuration can also spell out "at how many decode requests to use which level" directly. With only decode or only prefill, the whole GPU is used. Each level change must first sync both streams, so it is adjusted only at moments like the start and end of a prefill batch;
- **Which attention backend decode uses**: each level has its own decode attention backend and CUDA Graph state (`decode_attn_backend_group`), and `update_decode_attn_backend` switches over on a level change;
- **Prefill cut by layer**: a new prefill batch is marked `ForwardMode.SPLIT_PREFILL`, and the model must implement `forward_split_prefill(..., split_interval)`, computing only layers `[start, end)` (common models such as Qwen, Llama and Gemma implement it). Each round computes `split_forward_token_budget // prefill tokens` layers: 8 layers per round for an 8K-token prefill, 1 layer per round for 128K tokens;
- **The main loop**: each round runs one decode step on the decode stream, submits the next few layers of prefill on the prefill stream, then syncs only the decode stream and processes decode's results; after the last layers of prefill are submitted, an event is recorded, and only once all TP ranks confirm through a CPU-side all-reduce that the event has completed are these requests merged into the decode batch, followed by a level change.

Current limitations (as written in the startup argument checks): it cannot be combined with pipeline parallelism (`pp_size` must be 1), chunked prefill (`chunked_prefill_size` must be -1), PD disaggregation, or overlap scheduling; on torch 2.7 and later, using green contexts together with CUDA Graphs may degrade performance, and a warning is printed at startup.

## Is it worth it: an estimate {#值不值一个估算}

Compare the approaches with a simple model: Qwen3-8B on an H100 decoding 32 requests (each with a 2048 context) when an 8192-token prefill slips in. A decode step reads the weights and KV once; prefill is compute-bound, with time inversely proportional to its SMs. How much bandwidth decode loses when using only some SMs is the most uncertain term in this model, so `S_SAT` (the SM count needed to saturate bandwidth) takes three values:

```python
from pdmux_policy import choose, stream_groups

# assumptions: H100 SXM, Qwen3-8B, BF16; 85% of bandwidth used, prefill MFU 50%; when decode uses only some SMs,
# bandwidth is scaled by min(1, SMs / S_SAT), where S_SAT is the SM count needed to saturate bandwidth (uncertain, so three values)
SMS, BW, FLOPS = 132, 3.35e12 * 0.85, 989e12 * 0.5
LINEAR = 36 * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 12288) + 4096 * 151936   # parameters of every layer's linear layers + lm_head
KV_PER_TOKEN = 36 * 2 * 1024 * 2                                                  # bytes: 36 layers × K, V × 8 heads × 128 dims × BF16
DECODE_BS, CONTEXT, PROMPT = 32, 2048, 8192


def decode_step(sms, s_sat):
    """decode 一步：读一遍权重和 32 个请求的 KV"""
    nbytes = 2 * LINEAR + DECODE_BS * CONTEXT * KV_PER_TOKEN
    return nbytes / (BW * min(1.0, sms / s_sat))


def prefill_flops(tokens, start=0):
    """prefill 从第 start 个 token 算到 start + tokens：线性层 2·参数·token，外加因果注意力"""
    attn = 2 * 2 * 36 * 4096 * (tokens * start + tokens * tokens / 2)
    return 2 * LINEAR * tokens + attn


def fmt(sec):
    return f"{sec * 1e3:.0f} ms"


def cell(text, width):
    """按显示宽度右对齐（汉字占两格）"""
    return " " * (width - sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)) + text


groups = stream_groups(SMS, 9)
p_sm, d_sm = groups[choose(groups, DECODE_BS, has_prefill=True)]    # the split SGLang picks for 32 decode requests
base = decode_step(SMS, 66)                                         # S_SAT has no effect when all SMs are used
rows = [("只有 decode（参考）", fmt(base), "-", "-")]
t = prefill_flops(PROMPT) / FLOPS
rows.append(("prefill 优先", f"有一步 {fmt(t + base)}", fmt(t), "0"))
for chunk in (2048, 512):
    steps = [max((prefill_flops(chunk, k) + 2 * LINEAR * DECODE_BS) / FLOPS, base) for k in range(0, PROMPT, chunk)]
    rows.append((f"分块 prefill {chunk}", f"{fmt(sum(steps) / len(steps))} × {len(steps)} 步", fmt(sum(steps)), str(len(steps))))
t = prefill_flops(PROMPT) / (FLOPS * p_sm / SMS)
for s_sat in (44, 66, 88):
    itl = decode_step(d_sm, s_sat)
    rows.append((f"PD 复用 {p_sm}/{d_sm}，S_SAT={s_sat}", fmt(itl), fmt(t), f"{t / itl:.0f}"))

print(f"decode {DECODE_BS} 个请求（上下文 {CONTEXT}）时，插进一个 {PROMPT} token 的 prefill：")
print(cell("做法", 26) + cell("decode 的 ITL", 16) + cell("prefill 用时", 14) + cell("期间每个 decode 请求出的 token", 34))
for name, itl, pf, n in rows:
    print(cell(name, 26) + cell(itl, 16) + cell(pf, 14) + cell(n, 34))
```

```text title="output"
decode 32 个请求（上下文 2048）时，插进一个 8192 token 的 prefill：
                      做法   decode 的 ITL  prefill 用时    期间每个 decode 请求出的 token
       只有 decode（参考）            9 ms             -                                 -
              prefill 优先   有一步 299 ms        291 ms                                 0
         分块 prefill 2048    74 ms × 4 步        295 ms                                 4
          分块 prefill 512   19 ms × 16 步        306 ms                                16
   PD 复用 80/52，S_SAT=44            9 ms        480 ms                                55
   PD 复用 80/52，S_SAT=66           11 ms        480 ms                                43
   PD 复用 80/52，S_SAT=88           15 ms        480 ms                                33
```

- **Prefill first**: this prefill finishes fastest, but decode has one step stalled for 300 ms, a disaster for an ITL SLO;
- **Chunked prefill**: the smaller the chunk, the faster each decode step, but every step during the prefill slows down (74 ms per step with 2048-token chunks, 19 ms with 512), and when chunks get too small, prefill itself loses efficiency;
- **PD multiplexing**: decode always runs on its own 52 SMs, and ITL only goes from 9 ms to 9–15 ms, with each decode request producing dozens more tokens during the prefill; the cost is that prefill uses only 80 SMs, and its time goes from 291 ms to 480 ms.

So PD multiplexing trades new requests' prefill time (TTFT) for steady decode ITL. Workloads with tight ITL SLOs and long prompts suit it best; if TTFT matters more, chunked prefill or a level giving prefill more SMs fits better. The basis for choosing a level is exactly decode's load: the more decode requests, the more bandwidth and compute they need, and the more SMs go to decode.

## When to use it {#什么时候用}

- **At small scale, where PD disaggregation isn't worth it**: one or two machines with a handful of GPUs each; PD multiplexing needs no KV transfer and no separate instance groups for prefill and decode;
- **When prompt lengths vary widely**: an occasional long prompt no longer stalls every decode;
- **It does not conflict with PD disaggregation**: after disaggregating, decode instances may still run a little prefill (such as speculative drafts or recomputation), and prefill instances may do some decode; how SMs are allocated within one GPU is a question at another level;
- **Don't expect full isolation**: bandwidth and L2 are shared. Attention reading KV inside prefill and MoE reading expert weights both take bandwidth, so decode's ITL still fluctuates; load test with real workloads (see [load testing, SLOs and capacity planning](../perf/benchmark.md)).

!!! interview "How to explain it"
    To explain "without PD disaggregation, how do you reduce prefill's interference with decode": the way to alternate in time is chunked prefill (mix one chunk of prefill into each step, with a token budget controlling each step's length); the way to split in space is PD multiplexing: use CUDA green contexts to split the SMs in two, with prefill and decode each running at the same time on their own streams. Decode is memory-bound and gets close to the bandwidth limit with a few SMs, so giving it a small share of SMs costs nothing in bandwidth. SGLang's implementation: at startup, cut several levels at a granularity of 8 SMs; at runtime, choose a level by the number of decode requests; cut prefill by layer, computing only a few layers per round, alternating with decode, and change levels when it ends. The costs are slower prefill (TTFT traded for ITL), and green contexts isolate only SMs, with memory bandwidth and L2 still shared.

## Exercises {#练习}

**1. Splitting on H20.** H20 has 78 SMs and compute capability 9.0. How many levels does SGLang's rule produce? Why fewer than H100?

??? success "Answer"
    The granularity is 8, so candidate prefill SM counts are 8, 16, ..., 64; they must also give prefill at least half (≥ 39, so from 40) and decode at least 16 (prefill ≤ 62), leaving only 40, 48 and 56, hence just three levels, (56, 22), (48, 30) and (40, 38), plus the two groups of ordinary streams at the ends, 5 groups in all. With fewer SMs, fewer values satisfy "decode at least 16, prefill at least half", so there are naturally fewer levels; the "number of split levels" in the selection formula becomes 3 accordingly.

**2. The selection formula.** Under the default configuration (H100, 6 levels), at how many decode requests is the 72 : 60 level first chosen? If you want this level at 24 decode requests, how could you change the configuration?

??? success "Answer"
    Level = decode requests × 6 // 36, which must equal 6 (the cap), so at least 36 requests. To use the last level at 24 requests, set `decode_bs_divisor` to 24 (level = requests × 6 // 24, exactly 6 at 24 requests); or use `manual_divisions` to write a threshold for each level directly, with 24 as the last level's threshold.

**3. The overhead of cutting prefill by layer.** A 131072-token prefill computes only 1 layer per round. For a 36-layer model, what does this mean? What extra overhead does it bring?

??? success "Answer"
    This prefill takes 36 rounds of the scheduling loop to finish, submitting one layer per round; the hidden states and residuals between layers (each 131072 × 4096 × 2 bytes ≈ 1 GB) must stay in GPU memory waiting for the next round. The overheads: one CPU scheduling pass and one decode-stream sync per round; possible gaps between layers for the kernels on the prefill stream; and the memory taken by intermediate activations. The benefit: each round's time is bounded, decode is never held up by one huge prefill, and the split can stay stable while the prefill is in progress.

## Summary {#小结}

- [x] Prefill and decode on the same GPUs can coexist in three ways: alternating in time (chunked prefill), splitting in space (PD multiplexing), and on different GPUs (PD disaggregation).
- [x] A green context makes a stream execute only on specified SMs (split in units of 8 SMs from 9.0 on); it isolates only SMs, with bandwidth and L2 still shared, and does not guarantee actual concurrency.
- [x] SGLang cuts several levels of SM splits at startup and picks a level by the number of decode requests at runtime; prefill is cut by layer, computing `split_forward_token_budget // tokens` layers per round, alternating with decode.
- [x] PD multiplexing trades prefill time (TTFT) for steady decode ITL; it suits small-scale deployments, strict ITL requirements, and workloads where prompt lengths vary widely.
