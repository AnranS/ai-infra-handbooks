# Fault tolerance, elastic scaling and troubleshooting for large-scale EP

<p class="lead">Expert parallelism spreads an MoE model across tens or hundreds of GPUs, with an all-to-all among all of them at every layer. This makes the whole EP group a single unit: if any GPU breaks or slows down, every GPU stops with it. The more GPUs, the more often this happens, and the more each occurrence costs. This chapter first works out how often failures happen and how far they reach, then reads SGLang's elastic EP: communication that is not hung by a broken GPU, re-placing experts on the healthy GPUs, rejoining once the failed GPU recovers, and scaling up online; next it uses an estimate to see whether redundant experts can cover a broken GPU, compares vLLM's approach, and finally assembles a troubleshooting checklist.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why does a single GPU failure cost more the larger the EP group?
    2. When a GPU breaks, how does SGLang's elastic EP keep the other GPUs serving?
    3. Why does breaking one GPU almost certainly "lose experts" when only the hot experts have redundant replicas? Where are the lost experts restored from?
    4. How do SGLang's and vLLM's elastic scaling differ?
    5. A whole EP group hangs and no GPU produces results. How do you find which GPU is at fault?

??? success "Answers (try first, then expand to compare)"
    1. The number of single-GPU failures depends only on the total number of GPUs, not on how they are grouped; but each layer's all-to-all binds the whole EP group together, so one broken GPU stops the whole group, and each failure stops as many GPUs as the group holds. The larger the group, the more GPU-hours each failure loses, and the more in-flight requests stop.
    2. With Mooncake (or NIXL) as elastic EP's communication backend, communication with a broken GPU does not hang forever, and the broken rank is marked 0 in `active_ranks`; when the GPUs notice the active set has changed, they trigger EPLB to re-place experts on the healthy GPUs and recompute the batch; once the failed GPU is repaired, it rejoins the communication groups in recover mode and syncs the expert placement from a healthy GPU.
    3. Redundant replicas exist only for a small fraction of the hottest experts (say 32 of 256), and the rest have a single copy; one GPU holds several single-copy experts, which disappear when it breaks. During re-placement, experts still on other GPUs are moved by GPU-to-GPU transfer, while experts whose every replica is gone must be loaded from outside: SGLang can enable expert backup (`--enable-elastic-expert-backup`), where each machine backs up part of the experts' weights in memory and reads them back over RDMA with Mooncake Transfer Engine; otherwise they are re-read from disk.
    4. SGLang reserves a maximum size (`--max-ep-size`); after `/scale_elastic_ep` starts a scale-up, new ranks join the expandable global communication group in scale mode, the expert placement is synced to them, then committed, and only growth is supported; vLLM's `--enable-elastic-ep` scales in units of data-parallel engines (the parameter of `/scale_elastic_ep` is the new DP size, larger or smaller), rebuilds communication with stateless NCCL groups, returns 503 to new requests during scaling, and can optionally drain in-flight requests first.
    5. Look at what each rank was doing last: where each rank's log stops, each process's call stack printed with py-spy, NCCL debug logs or PyTorch's flight recorder, to find the rank that did not enter this collective (or has already exited); then look at that GPU and its NIC: GPU Xid errors, ECC errors, NIC link errors and downgraded speeds.

## One GPU breaks, the whole group stops {#一张卡坏了整组都停}

The MoE layer's dispatch and combine are all-to-alls among all EP ranks (see [expert parallelism](../distributed/expert-parallel.md) and [NVSHMEM and DeepEP](../comm/nvshmem-deepep.md)): every rank waits for all ranks' data to arrive before moving on. So as soon as one rank has a problem (a process crash, a GPU error, a NIC dropping off), the others hang in that communication until a timeout. Without special handling, the only option is restarting the whole group: reloading hundreds of GB of weights, capturing CUDA Graphs, warming up, and every in-flight request fails.

How frequent are failures? Training clusters have published fairly complete statistics, which can be borrowed for an order of magnitude:

```python
# figures published in the Llama 3 paper: 16384 H100s pretraining for 54 days, 419 unexpected interruptions (about 78% hardware issues)
per_gpu_hour = 419 / (54 * 24 * 16384)
TOTAL, MONTH_H = 2048, 30 * 24                  # an inference service on 2048 GPUs, counted per month
RESTART_MIN, PAUSE_MIN = 20, 1                  # assumptions: a full group restart (reload weights, capture graphs, warm up) takes 20 minutes; an elastic recovery pauses the group for 1 minute
failures = TOTAL * per_gpu_hour * MONTH_H
print(f"每张卡每小时的意外故障率约 {per_gpu_hour:.2e}；{TOTAL} 张卡每月约 {failures:.0f} 次单卡故障，和怎么分组无关")
print("EP 组大小   组数   每组平均多久坏一次   每月损失的卡·小时：整组重启 / 弹性恢复")
for gpus in (16, 64, 128, 256):
    mtbf_days = 1 / (gpus * per_gpu_hour) / 24
    lost_restart = failures * gpus * RESTART_MIN / 60        # one GPU fails, the whole group stops
    lost_elastic = failures * gpus * PAUSE_MIN / 60
    print(f"{gpus:8d} {TOTAL // gpus:6d} {mtbf_days:14.1f} 天 {lost_restart:18.0f} / {lost_elastic:.0f}")
```

```text title="输出"
每张卡每小时的意外故障率约 1.97e-05；2048 张卡每月约 29 次单卡故障，和怎么分组无关
EP 组大小   组数   每组平均多久坏一次   每月损失的卡·小时：整组重启 / 弹性恢复
      16    128          132.0 天                155 / 8
      64     32           33.0 天                621 / 31
     128     16           16.5 天               1241 / 62
     256      8            8.2 天               2483 / 124
```

The number of single-GPU failures depends only on the total GPU count; but the larger the EP group, the more GPUs each failure takes down with it. With groups of 256 GPUs, the same service loses 16 times as many GPU-hours per month to restarts as with groups of 16. Making a broken GPU affect only itself (plus one very short pause) is the problem elastic EP sets out to solve. The restart and pause durations in the table are assumptions; real deployments should measure them for their own model size, storage and network.

## SGLang's elastic EP {#sglang-的弹性-ep}

SGLang's elastic EP serves deployments of DP attention + large-scale EP (each rank is both a data-parallel attention rank and an expert-parallel rank), with the code in `srt/elastic_ep/`.

### Keeping communication from hanging on a broken GPU {#不让通信被坏卡挂住}

`--elastic-ep-backend` specifies a fault-tolerant communication backend (`mooncake` or `nixl`). Taking Mooncake as the example: every communication group and EP's dispatch / combine carry the same `active_ranks` (a tensor in `ElasticEPState`, one 0/1 per rank) and a timeout. When a rank times out without responding, it is set to 0 in `active_ranks`, and later communication skips it instead of hanging forever.

### Continuing to serve: re-placing experts on healthy GPUs {#继续服务在健康的卡上重新放置专家}

After every forward pass, the model checks whether the active set matches the previous snapshot (`maybe_rebalance_after_rank_fault`). If not, a GPU has broken:

1. EPLB immediately recomputes the expert placement over the remaining healthy GPUs (EPLB's algorithm is in [large-scale EP deployment](ep-deploy.md));
2. `expert_location_updater` moves experts that are still on some GPU to their new locations by GPU-to-GPU transfer;
3. Experts whose every replica was on broken GPUs (`p2p_missing_logical_experts`) can only be loaded from outside: with `--enable-elastic-expert-backup`, from the in-memory expert backup (each machine backs up 1/number-of-nodes of the expert weights in host memory, read over RDMA through Mooncake Transfer Engine); otherwise these experts' weights are re-read from disk;
4. The batch is recomputed with the new placement, and serving continues.

### Recovery: the failed GPU rejoins {#恢复故障卡重新加入}

A repaired (or replaced) GPU starts with `--elastic-ep-join-mode recover` (the old flag `--elastic-ep-rejoin` is an alias for it) and rejoins the global communication group and the parallel groups set up at startup. Healthy ranks notice in `maybe_join_ep_ranks` that a rank is waiting to recover, wait for the peer to be ready, run recovery on each communication group, then broadcast the expert placement from a healthy rank and reset the active set.

### Scaling up: adding GPUs online {#扩容在线加卡}

At startup, `--max-ep-size` reserves the maximum size it can grow to (the active set and communication buffers are preallocated for it). The scale-up process:

1. Call `POST /scale_elastic_ep` with `{"new_ep_size": N}`: N must be larger than the current size and at most `--max-ep-size`, and the previous scale-up must have finished (otherwise it returns 409);
2. The new machines start with `--elastic-ep-join-mode scale` (requiring `--node-rank 1`), with `--elastic-ep-join-rank-offset` telling them the current EP size and `--elastic-ep-initial-size` the EP size the original deployment started with (which determines the storage layout of experts on each rank); they register the target size they are joining in the global TCPStore;
3. The existing ranks check at every step: when the registered target matches the requested N, they bring the new ranks into the expandable global communication group, extend EPLB's metadata, broadcast the expert placement, update the DP attention size, and commit after a global barrier;
4. The states go `waiting_for_cohort` → `pending` → `joining` → `configuring_data_plane` → `syncing_new_world` → `serving_expanded`; `GET /is_scaling_elastic_ep` queries them; if not finished within `--elastic-ep-scale-timeout` (600 seconds by default), it is marked failed.

### Limitations {#限制}

- It can only grow, not shrink; after a scale-up, fault recovery is no longer supported (the code simply reports "Restart the expanded deployment");
- Periodic EPLB rebalancing is paused while a scale-up is in progress;
- The active-set check sits on the forward path and introduces a host-device synchronization (the code's comments mention this too).

## Is the redundancy enough? An estimate {#冗余够不够一个估算}

Whether elastic EP can ride out a broken GPU "seamlessly" depends on whether the broken GPU's experts have replicas elsewhere. Below, place expert replicas randomly (never two replicas of the same expert on one GPU), break k random GPUs, and count how many experts are left with no replica on any GPU. The first two are the prefill and decode configurations from the [large-scale EP deployment](ep-deploy.md) chapter (replicas only for the hottest experts); the third places two copies of every expert:

```python
import random

E = 256                                              # DeepSeek-V3's number of routed experts


def place(gpus, slots_per_gpu, replicas, rng):
    """把各专家的副本放到卡上：同一个专家的副本不放在同一张卡上。返回每张卡上的专家列表"""
    copies = [e for e in range(E) for _ in range(replicas[e])]
    assert len(copies) == gpus * slots_per_gpu
    while True:
        rng.shuffle(copies)
        cards = [copies[g * slots_per_gpu:(g + 1) * slots_per_gpu] for g in range(gpus)]
        if all(len(set(c)) == len(c) for c in cards):
            return cards


def lost_after(cards, k, rng):
    """随机坏 k 张卡，数一数有多少个专家的副本全在坏卡上"""
    dead = set(rng.sample(range(len(cards)), k))
    alive = {e for g, c in enumerate(cards) if g not in dead for e in c}
    return E - len(alive)


rng = random.Random(0)
configs = [  # (label, GPUs, slots per GPU, extra replicas: one more copy each for the hottest experts)
    ("EP32，每卡 9 个，32 个冗余", 32, 9, 32),
    ("EP320，每卡 1 个，64 个冗余", 320, 1, 64),
    ("EP64，每卡 8 个，每个专家 2 份", 64, 8, 256),
]
print("坏 k 张卡之后，至少有一个专家在所有卡上都没了副本的概率（平均丢几个）：")
for name, gpus, slots, extra in configs:
    replicas = [2 if e < extra else 1 for e in range(E)]
    cards = place(gpus, slots, replicas, rng)
    cells = []
    for k in (1, 2, 4):
        lost = [lost_after(cards, k, rng) for _ in range(4000)]
        cells.append(f"k={k}: {sum(x > 0 for x in lost) / len(lost):4.0%}（{sum(lost) / len(lost):.2f}）")
    print(f"  {name}  " + "  ".join(cells))
```

```text title="输出"
坏 k 张卡之后，至少有一个专家在所有卡上都没了副本的概率（平均丢几个）：
  EP32，每卡 9 个，32 个冗余  k=1: 100%（6.98）  k=2: 100%（14.00）  k=4: 100%（28.40）
  EP320，每卡 1 个，64 个冗余  k=1:  59%（0.59）  k=2:  84%（1.20）  k=4:  98%（2.40）
  EP64，每卡 8 个，每个专家 2 份  k=1:   0%（0.00）  k=2:  12%（0.13）  k=4:  56%（0.78）
```

- **With replicas only for hot experts, breaking one GPU almost certainly loses experts**: under EP32 each GPU has about 7 single-copy experts, so breaking one GPU loses 7; under EP320 with one expert per GPU, the broken GPU holds a single-copy expert about 60% of the time;
- **Two copies of every expert** only guarantees no loss with one broken GPU; with two broken GPUs there is a better-than-one-in-ten chance of loss, and with four, over half;
- So "restoring lost experts from elsewhere" is an essential step: an in-memory expert backup turns it from a disk read (picking these experts out of a checkpoint of hundreds of GB) into one RDMA read, an order of magnitude or more faster to recover. The cost is that each machine devotes host memory to the backup.

## vLLM's elastic EP and fault handling {#vllm-的弹性-ep-与故障处理}

vLLM takes a different path:

- **`--enable-elastic-ep`**: scales in units of data-parallel engines. It requires EPLB (`--enable-eplb`), supports neither pipeline parallelism nor external or hybrid DP load balancing (scaling must be coordinated by the single API server and engine client); `--elastic-ep-max-dp-size` sets the maximum DP size it can grow to;
- **`POST /scale_elastic_ep`**: takes the new DP size `new_data_parallel_size` (larger or smaller) and `drain_timeout`; during scaling, a middleware returns 503 to new requests outright, and with `VLLM_ELASTIC_EP_DRAIN_REQUESTS=1` it first waits for in-flight requests to drain (returning 408 on timeout), then rebuilds DP/EP communication with stateless NCCL groups and re-places the experts;
- **`--enable-fault-tolerance`**: when a DP engine errors, first abort its requests and report its state as unhealthy, then wait for a recovery instruction from above (such as a retry, which rebuilds its DP communication group), and only error out for real after `engine_recovery_timeout_sec` (120 seconds by default).

The trade-off: SGLang focuses on "other GPUs keep going when one breaks" and "adding GPUs online", relying on fault-tolerant communication backends and expert backups; vLLM scales at the granularity of DP engines with a more general mechanism (ordinary NCCL), at the cost of briefly not accepting requests while scaling.

## Troubleshooting checklist {#排障清单}

Problems in large-scale EP mostly show up as "the whole group is slow" or "the whole group is stuck", and the first task is to find which rank, which GPU, which link:

| Symptom | Common causes | What to look at |
| --- | --- | --- |
| Every rank is stuck, with no results | a rank's process crashed, a GPU errored, or ranks took different execution paths (say one rank did an extra collective) | where each rank's log stops; `py-spy dump` for each process's call stack; `NCCL_DEBUG=INFO`; PyTorch's NCCL flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE` and friends) records each rank's recent collectives on timeout, so compare who did not enter this call |
| The whole group slows down, though every rank is running | a GPU or NIC is degraded: GPU clock throttling, PCIe / NVLink downgrade, IB link errors; or expert load imbalance | each rank's dispatch and combine times, to find the one that always arrives last; `nvidia-smi` clocks and Xid errors; NIC link state and error counters; expert load statistics (SGLang's `/start_expert_distribution_record` and `/dump_expert_distribution_record`) |
| One DP rank's queue keeps growing | uneven DP attention load: long-context requests concentrated on a few ranks | each DP rank's token count and KV usage; the routing policy (see [large-scale EP deployment](ep-deploy.md#dp-attention-的负载最慢的-rank-决定速度)) |
| A GPU OOMs after experts are re-placed | redundant experts or newly moved experts took memory | experts and free memory per GPU; the number of redundant experts; whether the KV pool size left headroom |
| Garbled output or NaN after fault recovery | the expert placement does not match the weights actually loaded; the recovery path did not reload lost experts | the placement broadcast and weight loading in the recovery logs; token-by-token comparison on a fixed batch of inputs (see [onboarding new models and aligning accuracy](../ops/new-model.md)) |

Two more lessons: first, **find the slowest one first**: a collective's time is set by the slowest rank, and averages hide the problem; second, **tell "broken" from "slow"**: process crashes and GPUs falling off are easy to spot, but clock throttling, degraded links and ECC errors on a single GPU do not make processes exit, they just quietly slow the whole group, and only continuous metric monitoring (each rank's communication time, GPU clocks and error counters) catches them.

!!! interview "In an interview"
    Asked "in an EP deployment of hundreds of GPUs, what do you do when one GPU breaks": first explain why it is hard: each layer's all-to-all binds the whole EP group together, so one broken GPU leaves the whole group stuck in communication, and the traditional approach can only restart the whole group; the number of failures is set by the total GPU count, but each failure's blast radius equals the group size. Then the three steps of elastic EP: a fault-tolerant communication backend (communication with a broken GPU does not hang, and the lost rank is flagged), EPLB re-placing experts on healthy GPUs (surviving experts moved GPU to GPU, lost ones read back over RDMA from in-memory backups or loaded from disk), and the repaired GPU rejoining the communication groups and syncing the expert placement. Add the limits of redundancy (with replicas only for hot experts, one broken GPU always loses experts), online scale-up (reserve a maximum size, new ranks join, commit), and how to troubleshoot: first find the slowest rank or the one that did not enter the collective, and tell "broken" from "slow".

## Exercises {#练习}

**1. Blast radius.** Using this chapter's estimate, with the same 2048 GPUs, if the EP group grows from 64 GPUs to 256, by what factor do the GPU-hours lost per month to whole-group restarts change? And if restart time drops from 20 minutes to 5?

??? success "Answer"
    GPU-hours lost = failures × group size × restart duration. Failures stay the same and the group grows from 64 to 256, so losses quadruple (621 → 2483 GPU-hours). Cutting restart time to 1/4 cuts losses to 1/4, exactly offsetting it: groups of 256 with 5-minute restarts lose the same as groups of 64 with 20-minute restarts. This shows both paths help: shrinking the blast radius (elastic EP) and shortening recovery (faster weight loading, say from memory or nearby nodes).

**2. Redundancy versus backup.** Why not simply place two copies of every expert and skip expert backups?

??? success "Answer"
    First, memory: DeepSeek-V3 has a copy of each expert in each of its 58 MoE layers, about 44 MB of FP8 weights per copy, so one extra copy of all 256 experts takes about 650 GB more GPU memory (256 × 58 × 44 MB), squeezing out a lot of KV Cache; second, it guarantees nothing: in this chapter's estimate, with two copies per expert, breaking two GPUs still loses experts more than one time in ten, and breaking four, over half the time. The main use of redundant replicas is load balancing (spreading traffic for hot experts); the safety net still has to be backups, kept in host memory without using GPU memory, with lost experts read back over RDMA.

**3. Stuck.** All requests in a 72-GPU EP group produce no results, and GPU utilization shows 100%. How would you investigate step by step?

??? success "Answer"
    100% GPU utilization with no results typically means NCCL / DeepEP kernels spinning while waiting for some peer. Steps: ① look at the last line of each rank's log and find the rank that differs from the others (say one rank reported an error, or has had no new logs for a long time); ② run `py-spy dump` on every rank to see which call each process is stuck in, finding the rank that did not enter this collective or took a different path (one communication too many or too few); ③ if the flight recorder is on, compare the ranks' recent collective sequence numbers; ④ once a suspect GPU is found, check the hardware: `nvidia-smi` Xid and ECC errors, the NIC's link state; ⑤ without elastic EP, isolate the GPU and restart the whole group; with elastic EP, confirm it was flagged as lost and the experts re-placed, then bring the repaired GPU back in recover mode.

## Summary {#小结}

- [x] Each layer's all-to-all binds an EP group into one unit: the number of failures is set by the total GPU count, and each failure's blast radius equals the group size.
- [x] SGLang's elastic EP: a fault-tolerant communication backend (Mooncake / NIXL) + `active_ranks` flagging lost ranks → EPLB re-places experts on healthy GPUs and recomputes the batch → the failed GPU rejoins in recover mode; `--max-ep-size` + `/scale_elastic_ep` scale up online.
- [x] With replicas only for hot experts, one broken GPU always loses experts, which must be restored from in-memory expert backups (read back over RDMA) or from disk.
- [x] vLLM: `--enable-elastic-ep` scales in units of DP engines (503 during scaling, with optional draining), and `--enable-fault-tolerance` has an erroring engine wait for recovery instructions.
- [x] Troubleshooting: first find the slowest rank or the one that did not enter the collective, tell "broken" from "slow", and continuously monitor each rank's communication time and hardware errors.
