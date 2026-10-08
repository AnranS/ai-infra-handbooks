# Collective primitives

<p class="lead">Every kind of parallelism comes down to a handful of communication primitives in the end: data parallelism is an all-reduce, ZeRO is a reduce-scatter and an all-gather, expert parallelism is an all-to-all, and the pipeline is point-to-point send and recv. This chapter runs each primitive across 4 CPU processes to make its semantics clear, then writes a ring all-reduce by hand out of point-to-point communication and works out its volume, which is the basis for estimating the cost of every kind of parallelism.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What do all-reduce, reduce-scatter, all-gather and all-to-all each do?
    2. Why is it said that all-reduce = reduce-scatter + all-gather?
    3. How much data does each card send in a ring all-reduce? How does that relate to the card count?
    4. What is bus bandwidth? Why does nccl-tests report it rather than the algorithm bandwidth?
    5. Why is an all-reduce of a small message slow? How is it optimised?

??? success "Answers for the self-test (answer first, then open this)"
    1. all-reduce: everyone's data is summed and everyone gets the total. reduce-scatter: after the sum, everyone gets only one segment of it. all-gather: everyone contributes a segment and everyone ends up with the whole thing assembled. all-to-all: everyone sends a different piece to every other one, like a transpose.
    2. First a reduce-scatter, after which everyone holds one segment of the total; then an all-gather, which assembles the segments and sends them to everyone, giving exactly a complete copy of the total. That is how the ring algorithm implements all-reduce, and ZeRO uses the two halves separately as well.
    3. Each card sends $2(n-1)/n \times S$ bytes, which approaches $2S$ as the card count grows and is thus all but independent of it (bandwidth-optimal), although the step count $2(n-1)$ grows linearly with the cards.
    4. The bus bandwidth is the bandwidth worked out from the data the algorithm actually puts on the links (for an all-reduce, the algorithm bandwidth times $2(n-1)/n$). It is independent of the card count and can be compared directly against the link's peak, which shows whether the hardware is saturated.
    5. For a small message each step's transfer is very short, so the fixed latency (startup, synchronisation, the round trip between cards) dominates, and the ring algorithm pays it on all $2(n-1)$ steps. The optimisations: an algorithm with fewer steps (a tree, or a one-shot that reads the peers' buffers directly), merging small messages into large ones, and a custom implementation that a CUDA graph can capture.

## The five primitives {#五个原语}

Launching 4 processes with `torchrun`, each process (rank) holding a 4-element tensor, here is what each rank holds after each primitive:

```python title="collectives.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def show(title, t):
    """把每个 rank 上的结果收集到 rank 0 打印"""
    parts = [torch.empty_like(t) for _ in range(world)] if rank == 0 else None
    dist.gather(t, parts, dst=0)
    if rank == 0:
        print(title)
        for r, p in enumerate(parts):
            print(f"  rank {r}: {p.tolist()}")


x = torch.arange(4, dtype=torch.float32) + 10 * rank     # rank r holds [10r, 10r+1, 10r+2, 10r+3]
show("输入", x)

y = x.clone()
dist.all_reduce(y)                                        # every rank gets the elementwise sum
show("all_reduce（求和）", y)

chunk = torch.empty(1)
dist.reduce_scatter_tensor(chunk, x)                      # summed, then cut into world pieces, with rank r taking the r-th
show("reduce_scatter", chunk)

gathered = torch.empty(world)
dist.all_gather_into_tensor(gathered, chunk)              # assemble every rank's piece, and every rank gets all of it
show("all_gather（接在 reduce_scatter 后面）", gathered)

out = torch.empty(4)
dist.all_to_all_single(out, x)                            # rank r's p-th element goes to rank p
show("all_to_all", out)

ok = torch.equal(gathered, y)
flag = torch.tensor([int(ok)])
dist.all_reduce(flag, op=dist.ReduceOp.MIN)
if rank == 0:
    print("reduce_scatter + all_gather == all_reduce：", bool(flag.item()))
dist.destroy_process_group()
```

```text title="output"
输入
  rank 0: [0.0, 1.0, 2.0, 3.0]
  rank 1: [10.0, 11.0, 12.0, 13.0]
  rank 2: [20.0, 21.0, 22.0, 23.0]
  rank 3: [30.0, 31.0, 32.0, 33.0]
all_reduce（求和）
  rank 0: [60.0, 64.0, 68.0, 72.0]
  rank 1: [60.0, 64.0, 68.0, 72.0]
  rank 2: [60.0, 64.0, 68.0, 72.0]
  rank 3: [60.0, 64.0, 68.0, 72.0]
reduce_scatter
  rank 0: [60.0]
  rank 1: [64.0]
  rank 2: [68.0]
  rank 3: [72.0]
all_gather（接在 reduce_scatter 后面）
  rank 0: [60.0, 64.0, 68.0, 72.0]
  rank 1: [60.0, 64.0, 68.0, 72.0]
  rank 2: [60.0, 64.0, 68.0, 72.0]
  rank 3: [60.0, 64.0, 68.0, 72.0]
all_to_all
  rank 0: [0.0, 10.0, 20.0, 30.0]
  rank 1: [1.0, 11.0, 21.0, 31.0]
  rank 2: [2.0, 12.0, 22.0, 32.0]
  rank 3: [3.0, 13.0, 23.0, 33.0]
reduce_scatter + all_gather == all_reduce： True
```

| Primitive | Semantics | Where it is used |
| --- | --- | --- |
| all-reduce | every rank's tensor summed elementwise (or maximum, and so on), with every rank getting the result | DDP's gradient synchronisation, after tensor parallelism's row partition |
| reduce-scatter | summed, then cut into $n$ pieces, each rank taking its own | ZeRO's gradient synchronisation, replacing the first half of the all-reduce under sequence parallelism |
| all-gather | each rank contributes a piece, and after assembling them every rank has all of it | ZeRO-3 fetching parameters, replacing the second half of the all-reduce under sequence parallelism |
| all-to-all | rank $r$'s $p$-th block goes to rank $p$, which amounts to a transpose | the mixture of experts' token dispatch, Ulysses switching between the sequence and the heads |
| send / recv | point-to-point | between neighbouring stages in pipeline parallelism |

The all-to-all's output is exactly the input transposed by rank: rank $p$ receives every rank's $p$-th element.

## The ring all-reduce and its volume {#环形-all-reduce-与通信量}

NCCL defaults to the **ring algorithm** for large messages: every rank is connected in a ring and the data is cut into $n$ blocks.

![Figure: the ring all-reduce](../assets/figures/ring-allreduce.svg){.aig-svg}

1. **The reduce-scatter phase** ($n-1$ steps): at each step every rank sends one block to the right and receives and accumulates one from the left. After $n-1$ steps every rank holds one block that has accumulated every rank's contribution.
2. **The all-gather phase** ($n-1$ steps): pass the accumulated blocks around the ring once more, so every rank ends up with all of them.

Press play to watch the blocks on each card fill in over these 2(n−1) steps:

<div class="aig-widget" data-widget="ringreduce"></div>

Implementing it with point-to-point `isend` and `irecv`, and counting the bytes each rank sends:

```python title="ring_allreduce.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, n = dist.get_rank(), dist.get_world_size()
right, left = (rank + 1) % n, (rank - 1) % n


def ring_all_reduce(t):
    """环形 all-reduce：先 n-1 步 reduce-scatter，再 n-1 步 all-gather。返回本 rank 发送的字节数。"""
    chunks = list(t.chunk(n))                      # views: accumulate in place on t directly
    sent = 0
    for step in range(n - 1):                      # reduce-scatter: at step `step`, send block (rank - step) to the right
        send_idx, recv_idx = (rank - step) % n, (rank - step - 1) % n
        buf = torch.empty_like(chunks[recv_idx])
        reqs = [dist.isend(chunks[send_idx].contiguous(), right), dist.irecv(buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx] += buf
        sent += chunks[send_idx].numel() * t.element_size()
    for step in range(n - 1):                      # all-gather: pass the already summed blocks once around the ring
        send_idx, recv_idx = (rank + 1 - step) % n, (rank - step) % n
        buf = torch.empty_like(chunks[recv_idx])
        reqs = [dist.isend(chunks[send_idx].contiguous(), right), dist.irecv(buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx].copy_(buf)
        sent += chunks[send_idx].numel() * t.element_size()
    return sent


torch.manual_seed(rank)
x = torch.randn(1024)
ref = x.clone()
dist.all_reduce(ref)
sent = ring_all_reduce(x)
ok = torch.tensor([int(torch.allclose(x, ref, atol=1e-5))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
sizes = [torch.zeros(1, dtype=torch.long) for _ in range(n)]
dist.all_gather(sizes, torch.tensor([sent]))
if rank == 0:
    total = x.numel() * x.element_size()
    print(f"{n} 个 rank，每个 rank 的数据 {total} 字节")
    print("与 dist.all_reduce 结果一致：", bool(ok.item()))
    print("每个 rank 发送的字节：", [int(s.item()) for s in sizes], f"= 2(n-1)/n × {total} = {2 * (n - 1) * total // n}")
dist.destroy_process_group()
```

```text title="output"
4 个 rank，每个 rank 的数据 4096 字节
与 dist.all_reduce 结果一致： True
每个 rank 发送的字节： [6144, 6144, 6144, 6144] = 2(n-1)/n × 4096 = 6144
```

Each rank sends (and receives) $2(n-1)/n \cdot S$ bytes, where $S$ is the tensor's size. As the card count grows this approaches $2S$ and is **all but independent of the card count**, which is the ring algorithm's virtue and the reason data parallelism scales to thousands of cards.
The reduce-scatter and the all-gather take half each: $(n-1)/n \cdot S$. An all-to-all sends $(n-1)/n \cdot S$ per rank (its own block does not have to go anywhere).

### The alpha-beta model {#α-β-模型}

The time of one collective can roughly be written as

$$T = \alpha \cdot \text{steps} + \frac{\text{bytes sent}}{\beta}$$

where $\alpha$ is the fixed latency per step (a few microseconds) and $\beta$ is the link bandwidth. A ring all-reduce has $2(n-1)$ steps: for a large message the bandwidth term dominates and it is close to optimal; **for a small message the latency term dominates**, and it grows linearly with the card count. So:

- NCCL switches to a **tree algorithm** for small messages (on the order of $\log n$ steps), or uses NVLS on an NVSwitch (reduction inside the switch).
- Inference's decode phase does all-reduces of only tens of kilobytes, and vLLM and SGLang both implement their own one-shot and two-shot all-reduce: every rank reads all of the other ranks' buffers directly (device memory visible to each other over NVLink) and finishes in one step (see the section on reading vLLM's custom all-reduce in the C++ handbook, [Reading the C++ of the inference libraries](cpp://engineering/reading-code/)).
- In training, DDP merges many small gradients into large buckets before the all-reduce (the next chapter).

### Algorithm bandwidth and bus bandwidth {#算法带宽与总线带宽}

nccl-tests reports two bandwidths: the **algorithm bandwidth** = $S / T$, and the **bus bandwidth** = the algorithm bandwidth times $2(n-1)/n$ (for an all-reduce). The bus bandwidth divides out the algorithm's own amplification factor and can be compared directly against the hardware's link bandwidth: 8 H100s doing an all-reduce over NVLink reach a bus bandwidth of about 450 GB/s (NVLink's one-way bandwidth), which says the link is essentially saturated.
`python practice/judge.py bench` on the exercise site measures this number on a multi-GPU machine.

## The hardware topology {#硬件拓扑}

- **Within a node**: NVLink and NVSwitch, 450 GB/s one way per H100, with full bandwidth between any two cards.
- **Between nodes**: one 400 Gb/s (50 GB/s) InfiniBand or RoCE card per GPU, with GPUDirect RDMA letting the card read and write device memory directly.
- NCCL probes the topology automatically, using NVLink within a node and the network cards between nodes, and builds the network so that cards with the same index in different machines share a network rail (rail-optimized).

The bandwidth within a node and between nodes differs by about a factor of 10, which decides the layout of almost every combination of parallelism: what communicates intensively goes inside a node and the rest goes between nodes.

!!! interview "How to explain it"
    Collective communication underlies every form of parallelism: what all-reduce, reduce-scatter, all-gather and all-to-all each do, and that all-reduce = reduce-scatter + all-gather (which is exactly how ZeRO uses the two halves). A ring all-reduce sends $2(n-1)/n \cdot S$ bytes per card, all but independent of the card count, which is why it is called bandwidth-optimal; but the step count is $2(n-1)$, so a small message is dominated by the fixed latency and NCCL switches to a tree or a hierarchical algorithm. Read a test report by its bus bandwidth, because that compares directly against the link bandwidth. NVLink within a node is about 10 times faster than the network between nodes, which decides which layer each kind of parallelism goes on.

!!! info "Related chapters"
    - [Collective communication: NCCL's algorithms and protocols](serving://comm/nccl/), [GPU interconnect and networking](serving://comm/interconnect/) (inference systems: bandwidth and topology on real hardware)
    - [Multi-GPU and NCCL](cuda://tools/multi-gpu/) (CUDA: how NCCL's API is used)
    - [Practising without multiple GPUs](no-multi-gpu.md) (this book: verifying these primitives on the CPU)

## Exercises {#练习}

1. 8 cards doing an all-reduce on a 1 GB tensor over NVLink at 450 GB/s one way, with 5 microseconds of latency per step. Estimate the ring algorithm's time. And for a 1 MB tensor?

??? success "Answer"
    1 GB: the bandwidth term is $2 \times 7/8 \times 1\text{ GB} / 450\text{ GB/s} = 3.9$ ms and the latency term $14 \times 5\,\mu s = 0.07$ ms, about 3.96 ms in all, dominated by bandwidth.
    1 MB: the bandwidth term is 3.9 microseconds and the latency term 70, so the latency is nearly 20 times the bandwidth. A small message needs an algorithm with fewer steps (a tree, a one-shot), or merging into a large message.

2. Use `all_to_all_single` to implement a matrix transpose: 4 ranks each hold one row block of a $4 \times 2$ matrix (rank $r$ holds row $r$), and after the operation rank $r$ holds the whole matrix... and think about it: is the role all-to-all plays in Ulysses context parallelism also a kind of transpose?

??? success "Answer"
    It is. In Ulysses each rank starts out holding **all of the attention heads for one segment of the sequence**, and after the all-to-all it holds **some of the heads for the whole sequence**: a distributed transpose between the sequence dimension and the head dimension, which lets each rank do complete-sequence attention for its own few heads independently. Another all-to-all transposes back afterwards (see [Context parallelism](../model/context.md)).

## Summary {#小结}

- [x] all-reduce, reduce-scatter, all-gather, all-to-all and send/recv underlie every kind of parallelism; all-reduce = reduce-scatter + all-gather.
- [x] A ring all-reduce sends $2(n-1)/n \cdot S$ bytes per rank, all but independent of the card count; but the step count grows with the cards, so a small message is dominated by latency.
- [x] The bus bandwidth compares directly against the link bandwidth; NVLink within a node is about 10 times faster than the network between nodes, which decides where each kind of parallelism goes.
