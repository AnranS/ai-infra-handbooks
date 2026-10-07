# NVSHMEM and DeepEP: GPU-initiated communication

<p class="lead">Expert parallelism in MoE does two all-to-alls per layer (dispatch and combine), with small, irregularly shaped data that is extremely latency-sensitive. With NCCL, the number of tokens each destination receives must first be synced back to the CPU before communication can start; across machines, two thirds of the traffic also crosses spine switches. DeepEP solves this with two modes: the high-throughput mode deduplicates per node, sending over RDMA first and then forwarding over NVLink; the low-latency mode has the GPU issue RDMA directly and replaces every CPU sync with fixed slots. Both are built on NVSHMEM. This chapter first covers NVSHMEM's programming model, then uses simulation and estimates to explain the design of DeepEP's two modes.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What is NVSHMEM's "symmetric heap"? How does its programming style differ from NCCL's?
    2. Why does doing MoE's all-to-all with NCCL need a CPU sync?
    3. How does DeepEP's high-throughput mode cut cross-node traffic? What does "each token goes to at most 4 nodes" do?
    4. Why does the low-latency mode reserve fixed slots for each source? What does it cost?
    5. How does the low-latency mode's hook achieve "communication without using SMs"?

??? success "Answers (try first, then expand to compare)"
    1. At startup every PE (each GPU) symmetrically allocates a block of the same size, and the same object has the same offset on every PE, so "object address + target PE number" suffices to access remote data. Unlike NCCL, where the CPU launches collectives, NVSHMEM lets GPU threads put / get / signal directly inside kernels, at a granularity down to a single thread.
    2. The receiver must first know how many tokens it will receive to allocate receive buffers and set the splits of `all_to_all_single`; this count is computed on the GPU and must be copied back to the CPU before the all-to-all is launched: one GPU-to-CPU sync, which also prevents CUDA Graph capture.
    3. Per-node deduplication: a token bound for a remote node is sent only once, over RDMA to the same-rail GPU on that node, then forwarded over NVLink to the GPU that actually holds the expert. Node-limited routing sends each token to at most 4 nodes; together, cross-node traffic drops from 6.77 copies per token to 3.36.
    4. Fixed slots let the sender know where to write on its own, and the receiver need not know in advance how many it will receive: the count exchange and CPU sync disappear, all shapes are fixed, and it can be captured in a CUDA Graph. The cost is reserving memory for the worst case (every token from every source picking the same expert): in this chapter's example only about 3% of 232 MiB is used.
    5. dispatch returns as soon as it has issued the RDMA requests, leaving "wait for and receive the data" to a hook function that the caller invokes when convenient; while data is on the network no kernel is waiting, and all SMs are free to compute something else (such as another micro-batch).

## NVSHMEM: communicating inside kernels {#nvshmem在-kernel-里通信}

NCCL's model is "host-initiated collectives": the CPU calls `ncclAllReduce`, NCCL launches a communication kernel on the stream, and all ranks must call in the same order. **NVSHMEM** takes a different path, the PGAS (partitioned global address space) model:

- each GPU is a **PE** (processing element);
- **Symmetric heap**: `nvshmem_malloc(size)` is a collective call that allocates a block of the same size on every PE, at the **same offset** in each heap. So "element 100 of this buffer on PE 3" can be expressed with just a local pointer plus a PE number;
- **Communication inside kernels**: GPU threads can call `nvshmem_put` / `nvshmem_get` directly (plus non-blocking `_nbi` versions and warp / block cooperative versions) to read and write other PEs' symmetric memory, along with atomics and combined "write data + set a signal" operations;
- **Transport**: within one NVLink domain, put / get compile straight into loads and stores to remote GPU memory; across machines they go over RDMA. Early on, a proxy thread on the CPU had to post requests on the GPU's behalf; with IBGDA, GPU threads operate the NIC directly (see the previous chapter).

A kernel that "sends a chunk of this GPU's data to another GPU and notifies it" looks roughly like this (illustrative):

```cuda
// buf and flags both live on the symmetric heap: same offset on every PE
__global__ void send_block(float* buf, uint64_t* flags, const float* x, size_t n, int dst_pe) {
    nvshmemx_float_put_nbi_block(buf, x, n, dst_pe);        // the whole block cooperates to write x into dst_pe's buf
    nvshmem_fence();                                        // make sure the data arrives before the signal below
    if (threadIdx.x == 0)
        nvshmemx_signal_op(flags + nvshmem_my_pe(), 1, NVSHMEM_SIGNAL_SET, dst_pe);   // tell the peer: my data has arrived
}
// receiver: nvshmem_signal_wait_until(flags + src_pe, NVSHMEM_CMP_EQ, 1);
```

Compared with NCCL, NVSHMEM lets communication be **written in the same kernel as compute**, synchronizing finely by data dependency instead of waiting for a complete collective to finish. The price is lower-level programming: memory ordering, signals and buffer reuse are all yours to manage. PyTorch's symmetric memory (SymmetricMemory) offers similar capabilities.

## Why MoE's all-to-all is hard {#为什么-moe-的-all-to-all-难做}

Recall the flow from the [expert parallelism](../distributed/expert-parallel.md) chapter: route → sort by destination → **exchange how many tokens each destination receives** → dispatch → expert compute → combine. Implemented with NCCL, the trouble is the third step:

- The receiver must know how many tokens it will receive to allocate receive buffers and set the splits of `all_to_all_single`. This number is computed on the GPU and must be **copied back to the CPU**, which then launches the all-to-all: one GPU-to-CPU sync that breaks the pipeline and prevents CUDA Graph capture (the shapes differ every time);
- Across machines, a token's 8 chosen experts are spread over different nodes and GPUs, and sending directly produces a lot of duplicated and cross-rail traffic;
- NCCL's all-to-all is a set of point-to-point sends and receives that occupies SMs, competing with the experts' matrix multiplies.

DeepEP designs two separate modes, for training and prefill (large batches, aiming for throughput) and for decode (small batches, aiming for latency).

## High-throughput mode: per-node deduplication, two-hop forwarding {#高吞吐模式按节点去重两跳转发}

![Figure: two-hop dispatch in DeepEP's high-throughput mode: RDMA to the GPU in the same position, then NVLink forwarding](../assets/figures/deepep-two-hop.svg){.aig-svg}

Dispatch in the high-throughput mode (normal mode) takes two hops:

1. **RDMA**: for each **remote node** a token goes to, send only one copy, to the GPU on the target node **in the same position** as yourself (same rail; see [rail topology](interconnect.md#节点之间infinibandroce-与轨道拓扑));
2. **NVLink**: that GPU on the target node then forwards the token to the GPUs within the node that actually hold the target experts.

Combined with DeepSeek-V3's **node-limited routing** (each token goes to at most 4 nodes: first pick 4 nodes by the sum of the top 2 expert scores on each node, then pick 8 experts within those 4 nodes), cross-node traffic shrinks further. Simulate 64 GPUs (8 nodes), 256 experts and 8 picks per token, and count the cross-node traffic of 4096 tokens from one GPU on node 0:

```python
import numpy as np

NODES, GPN, EXPERTS, TOPK, MAX_NODES = 8, 8, 256, 8, 4
PER_GPU = EXPERTS // (NODES * GPN)                      # 4 experts per GPU
TOKENS, TOKEN_BYTES = 4096, 7168 + 7168 // 128 * 4      # tokens on one GPU; FP8 hidden vector + one fp32 scale per 128 elements
rng = np.random.default_rng(0)
scores = rng.random((TOKENS, EXPERTS)) + rng.random(EXPERTS) * 0.5    # routing scores, with some hot/cold difference between experts


def route(limit_nodes):
    s = scores.copy()
    if limit_nodes:                                     # DeepSeek-V3: score each node by the sum of its top 2 scores, and pick experts only from the 4 best nodes
        per_node = s.reshape(TOKENS, NODES, -1)
        node_score = np.sort(per_node, axis=-1)[:, :, -TOPK // MAX_NODES:].sum(-1)
        banned = np.argsort(node_score, axis=1)[:, :NODES - MAX_NODES]
        per_node[np.arange(TOKENS)[:, None], banned] = -np.inf
    return np.argsort(-s, axis=1)[:, :TOPK]


for limit in (False, True):
    experts = route(limit)                              # the source GPU is on node 0
    gpu = experts // PER_GPU
    node = gpu // GPN
    per_expert = (node != 0).sum(1).mean()
    per_gpu = np.mean([len({g for g in row if g // GPN != 0}) for row in gpu])
    per_node = np.mean([len(set(row) - {0}) for row in node])
    print(f"{'限制每个 token 最多去 4 个节点' if limit else '不限制节点数'}：每个 token 平均涉及 {np.mean([len(set(r)) for r in node]):.2f} 个节点（含本节点）")
    for name, copies in (("按专家发", per_expert), ("按目标卡去重", per_gpu), ("按目标节点去重（DeepEP）", per_node)):
        t = TOKENS * copies * TOKEN_BYTES / 50e9 * 1e3
        print(f"  {name}：每个 token 跨节点 {copies:.2f} 份，{TOKENS} 个 token 走 50 GB/s 网卡 {t:.2f} ms")
```

```text title="output"
不限制节点数：每个 token 平均涉及 5.28 个节点（含本节点）
  按专家发：每个 token 跨节点 6.77 份，4096 个 token 走 50 GB/s 网卡 4.10 ms
  按目标卡去重：每个 token 跨节点 6.50 份，4096 个 token 走 50 GB/s 网卡 3.93 ms
  按目标节点去重（DeepEP）：每个 token 跨节点 4.51 份，4096 个 token 走 50 GB/s 网卡 2.73 ms
限制每个 token 最多去 4 个节点：每个 token 平均涉及 3.97 个节点（含本节点）
  按专家发：每个 token 跨节点 6.70 份，4096 个 token 走 50 GB/s 网卡 4.05 ms
  按目标卡去重：每个 token 跨节点 6.26 份，4096 个 token 走 50 GB/s 网卡 3.79 ms
  按目标节点去重（DeepEP）：每个 token 跨节点 3.36 份，4096 个 token 走 50 GB/s 网卡 2.03 ms
```

The two measures together cut cross-node traffic from 6.77 copies per token to 3.36, exactly half; forwarding within the node goes over NVLink, with 9 times a NIC's bandwidth, and is basically never the bottleneck. A few more details:

- **FP8 for dispatch, BF16 for combine**: dispatch sends the experts' inputs, which can be quantized (fine-grained scaling, one scale per 128 elements); combine brings back weighted sums of expert outputs, kept in BF16 precision;
- **The SMs used are configurable**: the user specifies how many SMs the communication kernel occupies (in training DeepSeek-V3 saturated IB and NVLink bandwidth with only 20 SMs), leaving the rest for compute;
- **One CPU sync is still needed**: the high-throughput mode first computes the token counts bound for each rank and each expert (the layout), and after the exchange the CPU must know how many tokens to receive to allocate the receive tensors. So it suits prefill and training (large batches amortize the cost of one sync) and cannot be captured in a CUDA Graph.

## Low-latency mode: fixed slots, no CPU sync {#低延迟模式固定槽位没有-cpu-同步}

In decode each GPU has only tens to hundreds of tokens at a time, and one CPU sync (tens of microseconds) can exceed the communication itself. The low-latency mode's approach:

- **Pure RDMA, issued by the GPU** (IBGDA): each token goes straight to the GPU holding the target expert, without forwarding within the node (one hop fewer, lower latency);
- **Fixed slots**: each GPU reserves up to `num_max_dispatch_tokens_per_rank` slots for "each local expert × each source rank". The sender knows which slot to write, and the receiver need not know in advance how many tokens it will receive: the layout exchange and CPU sync disappear, all shapes are fixed, and it can be captured in a CUDA Graph.

The cost is memory. Estimate the receive buffer at DeepSeek-V3's scale:

```python
ranks, experts, max_tokens, hidden = 64, 256, 128, 7168     # EP over 64 GPUs, each sending at most 128 tokens at a time
local = experts // ranks
msg = hidden + hidden // 128 * 4 + 16                      # FP8 data + scales + a small header
recv = local * ranks * max_tokens * msg                    # each local expert reserves max_tokens slots for each source rank
print(f"每张卡上 {local} 个专家 × {ranks} 个来源 × {max_tokens} 个槽位 × {msg} 字节 = {recv / 2**20:.0f} MiB")
print(f"真正用到的（每个 token 选 8 个专家、平均分摊）：{max_tokens * ranks * 8 / experts * local * msg / 2**20:.1f} MiB")
```

```text title="output"
每张卡上 4 个专家 × 64 个来源 × 128 个槽位 × 7408 字节 = 232 MiB
真正用到的（每个 token 选 8 个专家、平均分摊）：7.2 MiB
```

Only about 3% of the slots are ever used, but they must be reserved for the worst case (every token from every source picking the same expert): a classic design trading memory for latency. The real buffers add send buffers, BF16 combine buffers and double buffering, so the low-latency mode's memory overhead is on the order of GB and grows with the EP size; `num_max_dispatch_tokens_per_rank` thus becomes a parameter to set carefully according to the decode batch size.

**The hook: communication without SMs.** The low-latency mode's dispatch can return as soon as it has issued the RDMA requests, leaving "wait for and receive the data" to a hook function the caller invokes when convenient. While the data is in flight on the network, no kernel is waiting, and all SMs can compute something else. SGLang's **two-batch overlap** exploits this: split a batch into two micro-batches; while one computes attention and experts, the other's dispatch / combine is on the network, alternating.

| | High-throughput mode | Low-latency mode |
| --- | --- | --- |
| Scenario | training, prefill (large batches) | decode (small batches) |
| Path | RDMA to the same-rail GPU, then NVLink forwarding; per-node deduplication | pure RDMA straight to the target GPU |
| Receive buffer | exchange counts first, allocate as needed (needs a CPU sync) | fixed slots reserved for the worst case |
| CUDA Graph | not supported | supported |
| Overlap with compute | set the number of SMs communication uses, running alongside compute | hook: return after issuing, no SMs used |

## Using it in inference frameworks {#在推理框架里使用}

- **SGLang**: `--moe-a2a-backend deepep` enables DeepEP, and `--deepep-mode` can be `normal`, `low_latency` or `auto` (high-throughput for prefill, low-latency for decode); used together with DP Attention and two-batch overlap. In a PD-disaggregated deployment, prefill and decode instances map neatly onto the two modes;
- **vLLM**: `--all2all-backend` selects `deepep_high_throughput` or `deepep_low_latency` (the default is an implementation based on all-gather / reduce-scatter, and there are also MoRI, NIXL-EP, FlashInfer and other backends);
- All these backends require NVSHMEM and IBGDA to be available: drivers, NIC firmware and kernel parameters must all be configured, the most common source of "environment problems" when deploying large-scale EP.

!!! interview "In an interview"
    DeepEP is nearly a must-ask in interviews on large-scale MoE inference. Start with the problem: MoE's all-to-all depends on routing results, is small and latency-sensitive, and NCCL needs a CPU sync to exchange counts and occupies SMs; then the two modes: the high-throughput mode deduplicates per node, uses same-rail RDMA + NVLink forwarding, FP8 dispatch / BF16 combine and a configurable SM count, suited to prefill; the low-latency mode sends directly via IBGDA, avoids syncs with fixed slots, supports CUDA Graphs and uses no SMs thanks to the hook, suited to decode. Finish with numbers: node-limited routing plus per-node deduplication halves cross-node traffic; only a few percent of the low-latency mode's slots get used, trading memory for latency.

## Exercises {#练习}

**1. Why the GPU "in the same position"?** When the high-throughput mode crosses nodes, why send to the GPU on the target node in the same position as yourself, rather than straight to the GPU holding the target expert?

??? success "Answer"
    In a rail-optimized topology, GPUs in the same position on different nodes connect to the same leaf switch, only one hop apart; sending to a GPU in another position goes through the spine switches and competes with other traffic for uplinks. Also, after per-node deduplication a token has only one copy per node, so some GPU on the target node must redistribute it; choosing the GPU in the same position takes the shortest network path and spreads the redistribution work evenly over every GPU in the node.

**2. How big should the slots be?** A decode instance's maximum batch per GPU is 64 tokens, EP=32, 8 experts per GPU, hidden size 7168 (about 7.4 KB with FP8 plus scales). What is the minimum `num_max_dispatch_tokens_per_rank`? How big is dispatch's receive buffer? And if the batch grows to 256?

??? success "Answer"
    At least the most tokens each GPU sends at a time, which is 64. The receive buffer is $8 \times 32 \times 64 \times 7.4\,\text{KB} \approx 121$ MB. When the batch grows to 256 the slots must follow to 256, and the buffer becomes about 485 MB, four times as much. Set the slots too small and the excess tokens cannot be sent (the implementation errors out or requires splitting the batch); too large and memory is wasted, squeezing the KV Cache. This is why decode's maximum batch, the EP size and the KV Cache capacity must be planned together.

**3. When don't you need DeepEP?**

??? success "Approach"
    When the EP size stays within one NVLink domain (say, a single 8-GPU machine, or an NVL72 supernode), there is no cross-node traffic, and per-node deduplication and two-hop forwarding are pointless; then NCCL's all-to-all, an in-machine implementation over CUDA IPC, or even dropping EP for TP may all be simpler. When a model has few experts and each token picks few of them, the all-to-all volume is small to begin with, and the gains are limited. DeepEP's value lies mainly in "large-scale, cross-node, fine-grained MoE".

## Summary {#小结}

- [x] NVSHMEM: the PGAS model with identical offsets on the symmetric heap, and GPU threads putting / getting / signaling directly inside kernels; NVLink memory access within a machine, RDMA across machines (IBGDA lets the GPU operate the NIC directly).
- [x] The problems with MoE all-to-all over NCCL: per-destination token counts must be synced back to the CPU, no CUDA Graph capture, heavy cross-machine traffic, and SM usage.
- [x] High-throughput mode: per-node deduplication with same-rail RDMA then NVLink forwarding, halving cross-node traffic together with node-limited routing; FP8 dispatch, BF16 combine; needs a CPU sync; for training and prefill.
- [x] Low-latency mode: IBGDA sends straight to the target GPU, fixed slots remove the count exchange, and it can be captured in a CUDA Graph; the hook keeps communication off the SMs, pairing with two-batch overlap; the cost is memory reserved for the worst case.
