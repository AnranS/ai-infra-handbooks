# RL inference systems: asynchronous rollout, policy staleness and weight sync

<p class="lead">The <a href="../../topics/rl-rollout/">inference in RL training</a> chapter covered rollout's long tail, inconsistent probabilities between training and inference, and memory switching; the <a href="train://practice/frameworks-rl/">training frameworks and RL training systems</a> chapter of the distributed training handbook ran a GRPO loop end to end from the training side and implemented weight re-sharding. This chapter does the accounting at large scale: how fast each of three pipelines (synchronous, one-step asynchronous, fully asynchronous) runs, and which samples the "staleness" of asynchrony falls on; how a trillion parameters' worth of weights reaches every inference instance within seconds at every step; and why MoE models also need to carry inference-time routing results back to the training side.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why is rollout utilization so low in synchronous RL? How much does "one-step asynchrony" fix?
    2. In fully asynchronous RL, which samples are the most "stale"? What bias does this introduce?
    3. When the inference engine updates its weights, what happens to requests mid-generation?
    4. When a trillion parameters' worth of weights must be synced to dozens of inference instances, how can the time be made independent of the instance count?
    5. What problem does "routing replay" solve for MoE models? What must the inference engine provide?

??? success "Answers (try first, then expand to compare)"
    1. Rollout answer lengths are long-tailed: each batch waits for its few longest answers, and most of the time the inference pool has only a few requests still running (in this chapter's simulation, about 70% of the time is idle). One-step asynchrony overlaps generating the next batch with training this one, hiding only the training time; the long tail itself is unchanged, so it is only slightly faster.
    2. Long answers: they take long to write, and the weights they started with are long out of date (over 2 versions behind on average), while short answers lag only about 0.4 versions; with a staleness cap, the ones dropped are long answers too: long reasoning chains are both staler and more likely to be dropped, which can push the model toward short answers.
    3. Pause → update → resume: vLLM can abort in-progress requests, wait for them to finish, or freeze them in the queue and continue after the update (then the earlier and later parts of one answer come from different weights, and the sampling-time logprob must be recorded per token); the prefix cache is usually cleared after an update.
    4. Each inference rank receives exactly the shard it needs (with the mapping between the training-side and inference-side partitions computed in advance), and senders use several GPUs in parallel; instances relay in a chunked pipeline, forwarding each chunk to the next instance as soon as it arrives, so total time is nearly independent of the instance count (this chapter estimates about 2.6 seconds for 64 instances).
    5. Tiny numerical differences between the inference and training sides can make MoE's top-k routing pick different experts, turning into an entirely different computation path; routing replay records the experts each token picked at each layer during inference, and the training side uses them directly. The inference engine must be able to return each token's routing results (`--enable-return-routed-experts` in vLLM and SGLang).

## Three pipelines {#三种流水}

![Figure: three RL pipelines: synchronous, one-step asynchronous, fully asynchronous](../assets/figures/rl-pipelines.svg){.aig-svg}

Rollout's long tail makes synchronous RL spend most of its time "waiting for the last few answers to finish". Simulate an inference pool with 256-way concurrency, 512 samples per step (say 64 questions × 8 answers), 60 seconds of training per step, and long-tailed answer lengths (median about 2000 tokens, truncated at 16K), comparing three pipelines:

- **Synchronous**: generate a whole batch → train → update weights → generate the next batch;
- **One-step asynchronous**: train batch $i$ while generating batch $i+1$ with the previous weights, so samples always lag exactly one version;
- **Fully asynchronous**: as soon as a slot in the inference pool frees up, start a new sample with the latest weights; the training side trains once it has a full batch and bumps the version number.

```python
import heapq
import random

SLOTS, SPEED, BATCH, TRAIN, HOURS = 256, 30.0, 512, 60.0, 2.0   # rollout concurrency slots, tokens generated per second per sequence, samples per step, training seconds per step, simulated duration


def length(rng):
    return min(rng.lognormvariate(7.6, 1.0), 16384)            # answer lengths: median about 2000, long tail truncated at 16K


def sync_like(overlap):
    """同步（overlap=False）与一步异步（overlap=True）：每步生成一整批，批内用连续批处理"""
    rng, t, steps, busy = random.Random(0), 0.0, 0, 0.0
    while t < HOURS * 3600:
        slots = [0.0] * SLOTS                                   # when each slot becomes free (relative to the start of this batch)
        for _ in range(BATCH):
            d = length(rng) / SPEED
            s = heapq.heappop(slots) if len(slots) == SLOTS else 0.0
            heapq.heappush(slots, s + d)
            busy += d
        rollout = max(slots)                                    # this batch waits for its longest answer to finish
        t += max(rollout, TRAIN) if overlap else rollout + TRAIN
        steps += 1
    return steps, busy / (t * SLOTS), (1.0, 1) if overlap else (0.0, 0)


def fully_async(max_stale=None):
    """全异步：槽位一空就用最新的权重开始新样本；训练端凑够一批就更新，版本号加一"""
    rng, version, steps, t, busy, done, stale, wasted = random.Random(0), 0, 0, 0.0, 0.0, [], [], 0.0
    running = [(n / SPEED, 0, n) for n in (length(rng) for _ in range(SLOTS))]   # (finish time, version at start, length)
    heapq.heapify(running)
    next_train = None
    while t < HOURS * 3600:
        finish, ver, n = heapq.heappop(running)
        if next_train is not None and next_train <= finish:     # training finishes first: bump the version
            t, version, steps, next_train = next_train, version + 1, steps + 1, None
            heapq.heappush(running, (finish, ver, n))
        else:
            t = finish
            m = length(rng)
            busy += m / SPEED
            heapq.heappush(running, (t + m / SPEED, version, m))
            done.append((ver, n))
        if max_stale is not None:                               # drop samples that are too stale (judged against the current version)
            keep = [x for x in done if version - x[0] <= max_stale]
            wasted, done = wasted + len(done) - len(keep), keep
        if next_train is None and len(done) >= BATCH:            # a full batch is ready: start training
            batch, done = done[:BATCH], done[BATCH:]
            stale += [(version - v, n) for v, n in batch]
            next_train = t + TRAIN
    total = len(stale) + wasted
    lag = [x for x, _ in stale]
    by_len = [sum(x for x, n in stale if cond(n)) / max(1, sum(cond(n) for _, n in stale))
              for cond in (lambda n: n < 1000, lambda n: n > 8000)]
    return steps, busy / (t * SLOTS), (sum(lag) / len(lag), max(lag)), wasted / total, by_len


for name, res in [("同步", sync_like(False)), ("一步异步（生成下一批与训练重叠）", sync_like(True))]:
    steps, util, (avg, mx) = res
    print(f"{name}：{HOURS:.0f} 小时 {steps} 步，rollout 槽位利用率 {util:.0%}，样本陈旧度平均 {avg:.1f}、最大 {mx}")
for name, bound in [("全异步（不限陈旧度）", None), ("全异步（最多落后 2 个版本）", 2)]:
    steps, util, (avg, mx), wasted, (short, long_) = fully_async(bound)
    print(f"{name}：{HOURS:.0f} 小时 {steps} 步，rollout 槽位利用率 {util:.0%}，样本陈旧度平均 {avg:.1f}、最大 {mx}，丢弃 {wasted:.0%}")
    print(f"  其中短回答（<1K token）平均陈旧 {short:.1f} 个版本，长回答（>8K token）平均陈旧 {long_:.1f} 个版本")
```

```text title="output"
同步：2 小时 11 步，rollout 槽位利用率 30%，样本陈旧度平均 0.0、最大 0
一步异步（生成下一批与训练重叠）：2 小时 12 步，rollout 槽位利用率 33%，样本陈旧度平均 1.0、最大 1
全异步（不限陈旧度）：2 小时 34 步，rollout 槽位利用率 100%，样本陈旧度平均 0.8、最大 4，丢弃 0%
  其中短回答（<1K token）平均陈旧 0.4 个版本，长回答（>8K token）平均陈旧 2.2 个版本
全异步（最多落后 2 个版本）：2 小时 33 步，rollout 槽位利用率 100%，样本陈旧度平均 0.7、最大 2，丢弃 2%
  其中短回答（<1K token）平均陈旧 0.4 个版本，长回答（>8K token）平均陈旧 1.8 个版本
```

- **Synchronous**: every batch waits for its longest answer (about 9 minutes), and the inference pool is idle 70% of the time;
- **One-step asynchronous** only hides the 60 seconds of training behind generation; the long tail itself is no shorter, and it gains just one step, which is what happens when training is far shorter than rollout;
- **Fully asynchronous** keeps the inference pool fully loaded and runs twice as many additional training steps in the same time; average staleness is under one version, which looks cheap.

But staleness is **not uniformly distributed**: short answers finish quickly and are consumed by training quickly, lagging only 0.4 versions on average; long answers take a long time to write, the weights they started with are long out of date, and they lag over 2 versions on average. With a staleness cap, the ones dropped are precisely the long answers too. For RL on reasoning models this is a systematic bias: **long reasoning chains are both staler and more likely to be dropped**, which can push the model toward short answers. Common countermeasures:

- **Correct rather than drop**: weight by the probability ratio between "the policy at sampling time" and "the current policy" (with truncation), correcting stale samples' contributions; the "decoupled" PPO objective used by fully asynchronous systems such as AReaL handles the behavior policy and the proximal policy separately for exactly this;
- **Batch evenly by length**: rather than "first finished, first trained", wait until each length range has a certain number of samples;
- **Interruptible generation**: when the weights update, interrupt long answers mid-generation and continue writing them with the new weights (a form of partial rollout). This means different parts of one answer come from different policy versions, so the sampling-time logprob must be recorded per token for the training side to correct properly; after the interruption, the KV of the already generated prefix was computed with the old weights, so either drop it and recompute with the new weights (one more prefill), or keep using it (introducing additional inconsistency).

## The time budget of weight sync {#权重同步的时间账}

After every training step, the new weights must reach every inference instance. For a trillion-parameter model in FP8 on the inference side, that is 1 TB per instance:

```python
W = 1e12                                      # a trillion-parameter model in FP8 on the inference side: each inference instance receives 1 TB
NIC, NVLINK, PER_INST = 50e9, 450e9, 8         # NIC bandwidth per GPU, NVLink bandwidth, 8 GPUs per inference instance

print(f"共置（同一批卡）：机内 NVLink 聚合后交给推理进程，每卡约 {W / PER_INST / NVLINK:.2f} s")
print("分离部署，推理实例数：      1        8       64")
naive = [n * W / NIC for n in (1, 8, 64)]                       # one training GPU sends to each instance in turn
star = [n * W / (PER_INST * NIC) for n in (1, 8, 64)]           # 8 training GPUs send in parallel, but separately to each instance
tree = [W / (PER_INST * NIC) * (1 + 0.05 * (n > 1)) for n in (1, 8, 64)]   # chunked pipelined broadcast: instances relay to one another
for name, row in (("单卡依次发送", naive), ("8 卡并行、逐实例发送", star), ("8 卡并行 + 实例间流水接力", tree)):
    print(f"  {name}：" + "  ".join(f"{x:>7.1f} s" for x in row))
```

```text title="output"
共置（同一批卡）：机内 NVLink 聚合后交给推理进程，每卡约 0.28 s
分离部署，推理实例数：      1        8       64
  单卡依次发送：   20.0 s    160.0 s   1280.0 s
  8 卡并行、逐实例发送：    2.5 s     20.0 s    160.0 s
  8 卡并行 + 实例间流水接力：    2.5 s      2.6 s      2.6 s
```

- **Co-location** is simplest: data moves only within the machine and is handed to the inference process via CUDA IPC, in under a second;
- **Disaggregated deployments**: the naive approach (gather onto one GPU, then send to each instance in turn) takes tens of minutes, completely unusable; letting each of an inference instance's 8 GPUs receive 1/8, with the sender also using several GPUs in parallel, takes only 2.5 seconds per instance; then having instances **relay in a chunked pipeline** (forwarding each chunk to the next instance as soon as it arrives, the same idea as [ring / pipeline broadcast](../comm/nccl.md)) makes the total time nearly independent of the instance count (the model adds 5% overhead for relaying);
- All of this assumes each inference rank can get exactly the shard it needs directly; since training and inference partition differently, the mapping of "who sends to whom" must be computed first (see [weight re-sharding](train://practice/frameworks-rl/#权重的重新切分)), rather than first gathering the full weights;
- If the inference side uses FP8 or lower precision, a quantization step is also needed before sending (training side) or after receiving (inference side).

**On the inference engine's side**, the cooperation needed is "pause → update → resume": vLLM's `pause_generation(mode=...)` can abort in-progress requests (`abort`), wait for them to finish (`wait`), or freeze them in the queue and continue after resuming (`keep`), and can optionally clear the KV and prefix caches; SGLang provides `/pause_generation` and `/continue_generation` endpoints, plus weight-update interfaces such as `update_weights_from_distributed` and `update_weights_from_tensor`. After an update, KV computed with the old weights in the prefix cache should generally be cleared, or new requests will hit the "old policy's" cache. The format conversion, in-place updates and CUDA Graph issues that hot updates must handle inside the engine are covered in [hot weight updates](../ops/weight-update.md).

## Routing replay for MoE {#moe-的路由重放}

The [inference in RL training](../topics/rl-rollout.md#问题二训练与推理的概率不一致) chapter measured that the inference and training sides compute different probabilities for the same sequence in bf16. MoE models amplify this: the slightest numerical difference between the two sides can make the router pick different top-k experts; a token that went through expert 3 at inference time goes through expert 7 when the training side recomputes, and the probability difference is no longer "a bit of numerical error" but a different computation path.

**Routing replay** works like this: during rollout, record the experts each token picked at each layer, and when the training side recomputes logprobs and backpropagates, use those experts directly instead of routing again. What the inference engine must provide is "return each token's routing results": both vLLM and SGLang have an `--enable-return-routed-experts` option (implemented in vLLM's `model_executor/layers/fused_moe/routed_experts_capturer.py` and SGLang's `srt/state_capturer/routed_experts.py`). The data is small: 61 layers with 8 expert IDs each, about 500 bytes per token, about 8 MB for a 16K answer.

!!! interview "In an interview"
    When asked "how do you make RL training more efficient", first show the bottleneck with numbers: in synchronous RL, rollout is held back by the long tail and the inference pool is idle 70% of the time, and one-step asynchrony can only hide the training time; full asynchrony keeps the inference pool fully loaded and multiplies the steps, at the cost of staleness, which concentrates on long answers and calls for importance correction, length-balanced batching and per-token logprob records. Then weight sync: co-located over NVLink, within a second; disaggregated, point-to-point by mapping with instances relaying in a pipeline, so even a trillion parameters take a few seconds independent of the instance count; the inference side must support pause, update, resume and cache clearing. Finally, mention routing replay for MoE to show you know the instability specific to MoE RL.

## Exercises {#练习}

**1. When is one-step asynchrony enough?** In this chapter's model, change the training time from 60 seconds to 600 (say, a bigger model and more samples per step). How does the gap between synchronous and one-step asynchronous change?

??? success "Answer"
    A synchronous step is about "rollout time + training time", and a one-step asynchronous step about "the larger of the two". With 60 seconds of training and about 9 minutes of rollout, one-step asynchrony saves only 60 seconds, almost no gain; when training and rollout take about as long (both around 10 minutes), one-step asynchrony nearly halves each step and pays off well. So one-step asynchrony suits scenarios where "training and generation weigh about the same"; when generation far outweighs training (reasoning models' long answers), the long tail itself must be removed with full asynchrony or partial rollout.

**2. Why clear the prefix cache after updating weights?** What happens if you don't? Are there exceptions?

??? success "Answer"
    The KV in the prefix cache was computed with the old weights. When a new request hits it, the prefix's KV comes from the old policy and later tokens from the new one, so the resulting probabilities belong to neither policy, importance correction becomes impossible, and multiple answers to the same question end up with different distributions depending on whether they hit. Exceptions: content that is "policy-independent", such as shared system prompts, still changes with the weights (KV depends on the weights), so strictly everything must be cleared; some systems accept this error for efficiency (for example, reusing prefixes only within one version), but must be clear that it is an approximation.

## Summary {#小结}

- [x] Synchronous RL is held back by the long tail, leaving the inference pool idle most of the time; one-step asynchrony only hides the training time; full asynchrony keeps the inference pool fully loaded and multiplies the training steps.
- [x] Full asynchrony's staleness concentrates on long answers, and with a cap the dropped ones are long answers too, pushing the model toward short answers; countermeasures are importance correction, length-balanced batching, and interruptible generation that records per-token logprobs.
- [x] Weight sync: co-located over NVLink, within a second; disaggregated, point-to-point by the partition mapping with instances relaying in a pipeline, in time independent of the instance count; the inference side needs pause / update / resume and cache clearing.
- [x] MoE routing can differ between the two sides; routing replay records the experts chosen at inference and uses them directly in training, and both vLLM and SGLang can return each token's routing results.
