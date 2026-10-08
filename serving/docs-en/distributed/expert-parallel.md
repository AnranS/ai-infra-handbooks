# Expert parallelism and DP Attention

<p class="lead">DeepSeek-V3 has 256 routed experts and 671 billion parameters: it does not fit on one GPU, and tensor parallelism is dragged down by MLA and communication. The mainstream deployment is data parallelism for attention (each GPU handles different requests, DP Attention) and expert parallelism for the MoE part (each GPU holds some of the experts, EP), with all-to-all exchanging tokens between the two. This chapter implements one MoE layer with EP + DP Attention using 4 processes on CPU and checks it against a single process; then it discusses load balancing (EPLB), DeepEP and overlapping compute with communication.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What are the two communications in an MoE layer under EP? What does each carry?
    2. Why does the attention part of models like DeepSeek use DP rather than TP?
    3. What happens when expert load is imbalanced? How does EPLB mitigate it?
    4. Which phases are DeepEP's "normal mode" and "low-latency mode" for?
    5. What is two-batch overlap? What does it hide?

??? success "Answers (try first, then expand to compare)"
    1. dispatch: send each token's hidden vector to the GPUs holding the experts it picked; combine: send the experts' outputs back to the token's original GPU and sum them with the gate weights. Both are all-to-alls.
    2. DeepSeek uses MLA, where all heads share one latent KV; TP splits by heads and cannot split it, so every GPU must store a full copy and KV capacity does not grow with the number of GPUs. DP Attention has each GPU handle different requests and store only its own requests' KV, combined with EP for the MoE layers.
    3. The most heavily loaded GPU sets each layer's time while the others wait idle; the GPUs holding hot experts may not even fit that many tokens. Based on measured expert loads, EPLB replicates hot experts (redundant experts) and rearranges where experts live, balancing the load across GPUs.
    4. Normal mode (high throughput): deduplicates per node and forwards in two hops, needs one CPU sync, used for training and prefill; low-latency mode: sends straight to the destination GPU with fixed slots and can be captured in a CUDA Graph, used for decode.
    5. Split a batch into two micro-batches: while one computes attention and experts, the other's dispatch / combine is on the network, alternating. It hides EP's all-to-all communication time.

<!-- comic ../assets/comics/expert-parallel.webp is in Chinese; put it back once the English version exists -->

## EP and DP Attention {#ep-与-dp-attention}

The [MoE chapter](llm://transformer/moe/) of the LLM handbook covered the computation of an MoE layer: the router picks top-k experts for each token, tokens are grouped and handed to the experts, and the results are added back with the routing weights. Expert parallelism spreads E experts evenly over n GPUs, E/n each. An MoE layer becomes:

![Figure: the two all-to-alls of expert parallelism](../assets/figures/ep-a2a.svg){.aig-svg}

1. **Routing**: each GPU computes which experts its tokens go to (the router is tiny, one copy per GPU);
2. **dispatch** (all-to-all): send each token to the GPUs holding its experts;
3. **Expert compute**: each GPU computes the tokens it received with its own experts, grouped by expert;
4. **combine** (all-to-all): send the results back to each token's original GPU and sum them with the routing weights.

What about attention? With TP, every GPU has to process the tokens of **all** requests, and the data layout must be converted before and after the MoE; worse, MLA's latent KV cannot be split by heads, so under TP every GPU stores the full KV Cache (see [tensor parallelism](tensor-parallel.md#kv-头数与-tp-的上限)). DP Attention instead has **each GPU handle different requests in attention**, each keeping the KV Cache of its own requests with no redundancy at all; at the MoE layers, all GPUs form one large EP group through all-to-all.

<!-- i18n:diagram 2f7178985b -->
```text
    GPU 0             GPU 1             GPU 2             GPU 3
  requests A, B     request C         requests D, E     request F           ← DP: each GPU serves different requests
  attention         attention         attention         attention
  (local KV)        (local KV)        (local KV)        (local KV)
        └─────────────── dispatch: all-to-all ────────────────┘
  experts 0–3       experts 4–7       experts 8–11      experts 12–15       ← EP: each GPU holds different experts
        └──────────────── combine: all-to-all ────────────────┘
  next layer's attention ...
```

## Implementation {#实现}

We reuse the LLM handbook's `moe.py` (`MoE.route` does the routing, and `forward_grouped` is the single-process reference):

```python title="ep.py"
"""ep.py —— 专家并行（EP）+ DP Attention 的最小实现：两次 all-to-all（dispatch 与 combine）。

用法：torchrun --standalone --nproc-per-node 4 ep.py
每个 rank 处理自己的一批 token（DP Attention：注意力部分各 rank 处理不同的请求），
MoE 层的 E 个专家平均分给各 rank；token 按路由结果发给专家所在的 rank，算完再发回来。
"""

import os

import torch
import torch.distributed as dist

from moe import MoE

D, D_FF, N_EXPERTS, TOP_K = 64, 128, 16, 2


def all_to_all(x: torch.Tensor, send_counts: list[int]) -> tuple[torch.Tensor, list[int]]:
    """先交换"各发多少"，再按这个数量交换数据。"""
    world = dist.get_world_size()
    counts = torch.tensor(send_counts)
    recv = torch.empty(world, dtype=torch.long)
    dist.all_to_all_single(recv, counts, [1] * world, [1] * world)
    recv_counts = recv.tolist()
    out = torch.empty((sum(recv_counts),) + tuple(x.shape[1:]), dtype=x.dtype)
    dist.all_to_all_single(out, x.contiguous(), recv_counts, send_counts)
    return out, recv_counts


def ep_forward(moe: MoE, x: torch.Tensor, rank: int, world: int) -> tuple[torch.Tensor, int]:
    per_rank = N_EXPERTS // world
    weights, idx, _ = moe.route(x)                                     # every rank has a copy of the router
    flat_expert = idx.flatten()                                        # [N*k] the expert each (token, slot) goes to
    flat_token = torch.arange(x.shape[0]).repeat_interleave(TOP_K)
    dest = flat_expert // per_rank                                     # the rank holding the expert
    order = dest.argsort(stable=True)                                  # sort by destination rank so we can send in segments
    send_counts = torch.bincount(dest, minlength=world).tolist()

    # dispatch: send each token's hidden state and "which local expert it goes to" to the rank holding the expert
    recv_x, recv_counts = all_to_all(x[flat_token[order]], send_counts)
    recv_e, _ = all_to_all(flat_expert[order] - dest[order] * per_rank, send_counts)

    # local compute: group by expert; each local expert does one matrix multiply over all tokens assigned to it
    y = torch.empty_like(recv_x)
    for e in range(per_rank):
        sel = (recv_e == e).nonzero().flatten()
        if len(sel):
            y[sel] = moe.experts[rank * per_rank + e](recv_x[sel])

    # combine: send results back to each token's original rank, then add them into place with the routing weights
    back, _ = all_to_all(y, recv_counts)
    out = torch.zeros_like(x)
    out.index_add_(0, flat_token[order], back * weights.flatten()[order, None])
    return out + sum(s(x) for s in moe.shared), recv_x.shape[0]


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    moe = MoE(D, D_FF, N_EXPERTS, TOP_K)                              # every rank uses the same seed and gets the same full weights
    torch.manual_seed(100 + rank)
    n_tokens = [96, 32, 64, 128][rank % 4]                             # batch sizes differ across ranks (DP)
    x = torch.randn(n_tokens, D)
    with torch.no_grad():
        out, n_received = ep_forward(moe, x, rank, world)

    # gather all tokens on rank 0 and use a single-process full MoE as the reference (sizes differ across ranks, so gather as objects)
    gathered = [None] * world
    dist.all_gather_object(gathered, (x, out, n_received))
    if rank == 0:
        xs, outs, loads = zip(*gathered)
        with torch.no_grad():
            ref = moe.forward_grouped(torch.cat(xs))
        err = (torch.cat(outs) - ref).abs().max().item()
        print(f"EP={world}：每个 rank {N_EXPERTS // world} 个专家；各 rank 的 token 数 {[len(t) for t in xs]}")
        print(f"各 rank 收到的 (token, 专家) 对数：{list(loads)}（总数 = token 数 × top-{TOP_K} = {sum(len(t) for t in xs) * TOP_K}）")
        print(f"与单进程完整 MoE 的最大误差：{err:.1e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

`all_to_all` takes two steps: first exchange "how much I will send you" so each GPU can allocate receive buffers, then exchange the data itself. Each (token, expert) pair sends the hidden state once, so dispatch moves `tokens × top-k × hidden`.

4 processes, 16 experts, top-2, and a different token count on each GPU (under DP each GPU's load differs anyway):

```python
import os
import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4",
                         "build/code/ep.py"], capture_output=True, text=True,
                        env={**os.environ, "OMP_NUM_THREADS": "2"}, timeout=1200)
print(result.stdout.strip())
assert float(result.stdout.split("最大误差：")[1]) < 1e-5
```

```text
EP=4：每个 rank 4 个专家；各 rank 的 token 数 [96, 32, 64, 128]
各 rank 收到的 (token, 专家) 对数：[160, 165, 179, 136]（总数 = token 数 × top-2 = 640）
与单进程完整 MoE 的最大误差：0.0e+00
```

The result matches the single process exactly. Note that the compute each GPU **receives** (second line) has nothing to do with the GPU's **own** token count (first line): GPU 1 has only 32 tokens of its own, yet computes 165 (token, expert) pairs for everyone. Under EP, each GPU's compute is set by the routing.

## Load balancing: EPLB {#负载均衡eplb}

The router's choices cannot be controlled at inference time. If some experts are especially "popular", the GPUs holding them must process far more tokens, and the other GPUs finish and can only wait: **the slowest GPU sets each layer's time**.

The load-balancing loss in training (the LLM handbook's [MoE chapter](llm://transformer/moe/#负载均衡)) only keeps the average roughly balanced, and the actual load still shifts with the input distribution. Inference engines use **EPLB (Expert Parallelism Load Balancer)**: measure each expert's load over a period of time, then

1. **Redundant experts**: replicate the few hottest experts several times on different GPUs, with their tokens split evenly among the replicas;
2. **Re-placement**: reassign experts (replicas included) to GPUs so that each GPU's total load is as close as possible.

Simulate it with a skewed load distribution (64 experts, 8 GPUs, 8 extra redundant slots):

```python
import heapq
import random

random.seed(0)
num_experts, num_gpus, num_redundant = 64, 8, 8
load = [1 / (i + 1) ** 0.8 for i in range(num_experts)]      # Zipf-like distribution: a few experts are very hot
random.shuffle(load)

def imbalance(gpu_loads):
    return max(gpu_loads) / (sum(gpu_loads) / len(gpu_loads))

per_gpu = num_experts // num_gpus
naive = [sum(load[g * per_gpu:(g + 1) * per_gpu]) for g in range(num_gpus)]

# 1. redundancy: repeatedly add one more replica of the expert with the highest load per replica
replicas = [1] * num_experts
for _ in range(num_redundant):
    hottest = max(range(num_experts), key=lambda e: load[e] / replicas[e])
    replicas[hottest] += 1
items = sorted((load[e] / replicas[e] for e in range(num_experts) for _ in range(replicas[e])), reverse=True)

# 2. placement: largest first, put each replica on the least-loaded GPU that still has a free slot
slots = (num_experts + num_redundant) // num_gpus
heap = [(0.0, g, 0) for g in range(num_gpus)]                 # (load, GPU id, number placed)
for w in items:
    skipped = []
    while True:
        l, g, n = heapq.heappop(heap)
        if n < slots:
            break
        skipped.append((l, g, n))
    heapq.heappush(heap, (l + w, g, n + 1))
    for s in skipped:
        heapq.heappush(heap, s)
balanced = [l for l, _, _ in heap]
print(f"按顺序放置：最忙的卡是平均负载的 {imbalance(naive):.2f} 倍")
print(f"冗余 {num_redundant} 个专家 + 重新放置：{imbalance(balanced):.2f} 倍")
```

```text
按顺序放置：最忙的卡是平均负载的 2.26 倍
冗余 8 个专家 + 重新放置：1.00 倍
```

The busiest GPU drops from 2.26× the average load to 1.00×, cutting the MoE part's time by more than half. The costs are the extra memory taken by redundant experts and moving expert weights during re-placement (usually done asynchronously, or periodically when the load changes slowly).

## Communication: DeepEP and overlapping compute with communication {#通信deepep-与计算通信重叠}

Every EP layer has two all-to-alls; how big are they? Take DeepSeek-V3 (hidden 7168, top-8) with 128 tokens per GPU per decode step:

```python
hidden, top_k, tokens = 7168, 8, 128
dispatch = tokens * top_k * hidden * 1          # dispatch is sent in FP8, 1 byte per number
combine = tokens * top_k * hidden * 2           # combine uses BF16
for name, bw in [("NVLink（约 450 GB/s）", 450e9), ("InfiniBand 400 Gb/s（约 50 GB/s）", 50e9)]:
    print(f"{name}：每层 dispatch + combine 共 {(dispatch + combine) / 1e6:.0f} MB，约 {(dispatch + combine) / bw * 1e6:.0f} μs")
```

```text
NVLink（约 450 GB/s）：每层 dispatch + combine 共 22 MB，约 49 μs
InfiniBand 400 Gb/s（约 50 GB/s）：每层 dispatch + combine 共 22 MB，约 440 μs
```

This is an upper bound where "every (token, expert) pair crosses GPUs". Across machines, a few hundred microseconds per layer times 58 MoE layers is tens of milliseconds, longer than one decode step's compute. Several key techniques exist to deal with this:

- **Limiting routing across machines**: DeepSeek-V3 is constrained in training to send each token to at most 4 nodes, and only once per node, after which NVLink forwards it to the experts within the node, sharply cutting cross-machine traffic;
- **DeepEP**: DeepSeek's open-source EP communication library. **Normal mode** targets prefill: large batches and high throughput, using both NVLink (within a machine) and RDMA (between machines) bandwidth; **low-latency mode** targets decode: pure RDMA, very low latency for small messages, and capturable by CUDA Graphs;
- **Two-batch overlap (TBO)**: split a batch into two micro-batches; while one computes attention or experts, the other does all-to-all communication, alternating, so communication hides behind compute;
- **PD disaggregation**: prefill and decode have entirely different best EP sizes and communication patterns, and only deploying them separately lets each be optimal (see [PD disaggregation](pd-disagg.md)).

!!! source "Source code"
    - **vLLM**: the MoE layer is in `vllm/model_executor/layers/fused_moe/` (`FusedMoE` and several expert kernels); all-to-all lives in `vllm/distributed/device_communicators/all2all.py`, selected with `--all2all-backend`: `deepep_high_throughput` (DeepEP normal mode, for prefill), `deepep_low_latency` (low-latency mode, for decode), `pplx`, `flashinfer_nvlink_*` and others, with `allgather_reducescatter` as the default; EPLB is in `vllm/distributed/eplb/` (`--enable-eplb`). Data parallelism + expert parallelism are turned on with `--data-parallel-size` and `--enable-expert-parallel`.
    - **SGLang**: `--enable-dp-attention` turns on DP Attention, `--ep-size` sets the EP size, and `--moe-a2a-backend deepep` selects DeepEP; the code is in `srt/layers/dp_attention.py`, `srt/layers/moe/` and `srt/eplb/`, and two-batch overlap in `srt/batch_overlap/` (`--enable-two-batch-overlap`). The SGLang team's blog post documents in detail how they deployed DeepSeek on 96 H100s with PD disaggregation + large-scale EP, and is worth reading.

!!! interview "How to explain it"
    A framework for "how would you deploy a large MoE model like DeepSeek-V3?": **memory** (671B parameters, about 700 GB in FP8, at least one 8-GPU H200 machine or several machines) → **parallelism** (DP for attention to avoid MLA's KV redundancy, EP for the MoE, all-to-all connecting the two) → **communication** (DeepEP: normal mode for prefill, low-latency mode for decode; two-batch overlap to hide communication; limiting cross-node routing) → **load** (EPLB: redundant experts + re-placement) → **disaggregation** (PD disaggregation, with different EP sizes for prefill and decode). If you can say "why" for each point, that is a strong explanation.

## Exercises {#练习}

**1. Why does decode need large EP?** 256 experts, top-8, a decode batch of 64 per GPU. With EP on only 8 GPUs (32 experts per GPU), how many tokens does each expert get on average? With 256 GPUs (1 expert per GPU)? What does this mean for the efficiency of expert compute?

??? success "Answer"
    All 8 GPUs together have 512 tokens, giving 512 × 8 = 4096 (token, expert) pairs, 16 tokens per expert on average. Whatever the EP size, **the number of tokens per expert depends on the global batch**; but the larger the EP, the larger the global batch (the sum over DP): with 256 GPUs at 64 tokens each, the global batch is 16384 tokens, and each expert averages 512. Expert compute is a GEMM of `[expert's tokens, hidden] × [hidden, d_ff]`; with too few tokens it is a memory-bound GEMV that reads the expert's weights once to compute a dozen or so tokens; only with many tokens does it approach being compute-bound. So large-scale EP is not just about fitting the model, but about letting each expert collect enough tokens for efficient compute.

**2. The load problem of DP Attention.** Under DP Attention, at some step GPU 0 has 100 decode requests and GPU 1 only 10. What happens?

??? success "Answer"
    GPU 1 finishes attention quickly, but the MoE layer's all-to-all is a collective that all GPUs must join at the same time, so GPU 1 can only wait for GPU 0. Worse, CUDA Graphs and all-to-all require the same shapes on every GPU, so each GPU's batch is usually padded to the same size (SGLang calls this MLP sync, and a GPU with no requests still runs an `IDLE` forward). So DP Attention needs **balancing requests across DP ranks** (routing by request count and KV usage), which is what vLLM's `DPLBAsyncMPClient` and SGLang's `DataParallelController` both do.

!!! tip "Expert parallelism in training"
    In training, the all-to-all must also backpropagate, and load balancing relies on an auxiliary loss or auxiliary-loss-free bias adjustment. The [MoE and expert parallelism](train://model/moe-ep/) chapter of the distributed training handbook implements dispatch / combine with autograd and simulates DeepSeek-V3's bias balancing.

## Summary {#小结}

- [x] EP spreads experts across GPUs, with two all-to-alls per MoE layer: dispatch sends tokens, combine brings results back.
- [x] DP Attention has each GPU handle different requests, avoiding MLA's KV redundancy; combined with EP it is the mainstream deployment for large MoE models.
- [x] Load imbalance lets the slowest GPU set each layer's time; EPLB balances it with redundant experts and re-placement.
- [x] Cross-machine all-to-all is expensive: limiting the nodes per token, DeepEP's two modes, two-batch overlap and PD disaggregation tackle it together.
