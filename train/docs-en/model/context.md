# Context parallelism: Ring Attention and Ulysses

<p class="lead">Once the sequence is long, both the activations and attention's computation grow with it: at 128K, one layer's activations alone are tens of gigabytes. Tensor parallelism partitions the hidden dimension and the pipeline partitions the layers, and neither addresses one sequence being too long. Context parallelism partitions along the **sequence**, so each card holds one segment of it. The difficulty is attention, where every token has to see every token before it. This chapter implements the two mainstream approaches, Ulysses (using an all-to-all to turn partitioning the sequence into partitioning the heads) and Ring Attention (passing the KV once around a ring), both lined up elementwise against single-process causal attention.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Besides attention, which operations in a Transformer can be partitioned along the sequence directly with no communication?
    2. What do Ulysses's two all-to-alls each do? What limits its degree of parallelism?
    3. What does Ring Attention pass at each step? How are the attention results of the blocks merged into the final one?
    4. Under causal attention, why does partitioning the sequence in order give an uneven load? How is it fixed?
    5. What does each of Ulysses and Ring Attention suit?

??? success "Answers for the self-test (answer first, then open this)"
    1. Apart from attention, every operation (the embedding, normalisation, the projections, the MLP, the loss) is computed independently per token, so after partitioning along the sequence each card does its own with no communication.
    2. The first turns "partitioned by sequence, all heads on each card" into "the complete sequence, some heads on each card", after which each card computes its own heads with an ordinary attention kernel; the second turns it back. The degree is limited by the head count (the KV head count under grouped-query attention) and cannot exceed it.
    3. At each step it passes the KV block it holds to the next card while computing the partial attention (the output and the log-sum-exp) of the current KV block against the local queries; the blocks are merged with $\mathrm{lse} = \log(e^{\mathrm{lse}_1} + e^{\mathrm{lse}_2})$ and $o = o_1 e^{\mathrm{lse}_1 - \mathrm{lse}} + o_2 e^{\mathrm{lse}_2 - \mathrm{lse}}$.
    4. Under a causal mask the later tokens have to see all of the earlier ones, so the card with the last segment has the most to compute and the one with the first has almost nothing, and the slowest card sets the pace. The zigzag partition: cut into $2P$ blocks and give each card one early and one late block, so the work is exactly complementary and identical.
    5. Ulysses: within a node, with enough heads, using an off-the-shelf attention kernel, communicating with 4 all-to-alls. Ring Attention: across nodes, for very long sequences, essentially unlimited by the head count, with the communication overlappable with the computation. The two can also be combined (Ulysses within a node, Ring between nodes).

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/context-parallel.webp is in Chinese; put it back once the English version exists -->

## Which operations partition directly {#哪些运算可以直接切}

LayerNorm, the MLP and the various projections are all **per token**: after partitioning the sequence each card does its own with no communication. Only attention needs information across tokens. So context parallelism has only one thing to solve: letting each query see all of the keys and values it needs. The part the two approaches share (causal attention that returns a log-sum-exp, plus the test data):

```python title="cp_common.py"
import torch


def attention(q, k, v, causal=True, q_offset=0, k_offset=0):
    """q: [Sq, H, D], k/v: [Sk, H, D]；返回输出和每行的 log-sum-exp（用于合并分块结果）。offset 是这一块在完整序列中的起点。"""
    scores = torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1] ** 0.5
    if causal:
        qi = torch.arange(q.shape[0])[:, None] + q_offset
        ki = torch.arange(k.shape[0])[None, :] + k_offset
        scores = scores.masked_fill(ki > qi, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)                     # [H, Sq]
    out = torch.einsum("hqk,khd->qhd", torch.exp(scores - lse[..., None]), v)
    return out, lse


def make_qkv(S=32, H=8, D=16):
    torch.manual_seed(0)
    return torch.randn(S, H, D), torch.randn(S, H, D), torch.randn(S, H, D)
```

## Ulysses: transposing between the sequence and the heads {#ulysses在序列和头之间转置}

The heads in attention are independent of each other. Ulysses's idea: one all-to-all turns each card's "all of the heads for one segment of the sequence" into "some of the heads for the whole sequence", so each card can do complete-sequence attention for its own heads independently; another all-to-all turns it back afterwards.

```python title="ulysses.py" torchrun="4"
import torch
import torch.distributed as dist

from cp_common import attention, make_qkv

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
q, k, v = make_qkv()                        # the complete [S, H, D], used for the reference result and for slicing out this rank's input
S, H, D = q.shape
ref, _ = attention(q, k, v)
rows = slice(rank * S // P, (rank + 1) * S // P)


def seq_to_head(x):
    """[S/P, H, D]（本 rank 持有一段序列的全部头）→ [S, H/P, D]（全部序列的一部分头）"""
    x = x.reshape(S // P, P, H // P, D).transpose(0, 1).contiguous()   # grouped by destination rank: group p is the p-th share of the heads
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)                                    # group p goes to rank p; the group r received comes from rank r's segment of the sequence
    return out.reshape(S, H // P, D)


def head_to_seq(x):
    """[S, H/P, D] → [S/P, H, D]：上面的逆操作"""
    x = x.reshape(P, S // P, H // P, D).contiguous()                   # group p is the p-th segment of the sequence
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    return out.transpose(0, 1).reshape(S // P, H, D)


ql, kl, vl = seq_to_head(q[rows]), seq_to_head(k[rows]), seq_to_head(v[rows])
out_local, _ = attention(ql, kl, vl)       # this rank does complete-sequence causal attention for its own H/P heads
out = head_to_seq(out_local)
ok = torch.tensor([int(torch.allclose(out, ref[rows], atol=1e-5))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"Ulysses，{P} 个 rank：每个 rank 输入 {tuple(q[rows].shape)}，注意力时 {tuple(ql.shape)}")
    print("输出与单进程的因果注意力一致：", bool(ok.item()))
dist.destroy_process_group()
```

```text title="output"
Ulysses，4 个 rank：每个 rank 输入 (8, 8, 16)，注意力时 (32, 2, 16)
输出与单进程的因果注意力一致： True
```

- Each all-to-all sends $(P-1)/P$ of the data per card, once each for $q$, $k$, $v$ and the output; the volume is proportional to the sequence length and essentially independent of the context-parallel degree.
- The attention part calls an off-the-shelf FlashAttention with no kernel changes, which is Ulysses's greatest virtue.
- The limit: the degree cannot exceed the attention head count (and under grouped-query attention it is stricter, not exceeding the KV head count unless the KV is replicated), and the all-to-all is expensive across nodes.

## Ring Attention: passing the KV once around a ring {#ring-attentionkv-沿环传一圈}

Ring Attention leaves the heads alone and has each card keep its own segment of queries while the KV blocks are passed round the ring. After $P$ steps every query segment has seen every KV block. Each step computes the partial attention of that query segment against one KV block, merged with the **log-sum-exp** (the same formula as FlashAttention's online softmax):

$$\text{lse} = \log(e^{\text{lse}_1} + e^{\text{lse}_2}), \qquad o = o_1 e^{\text{lse}_1 - \text{lse}} + o_2 e^{\text{lse}_2 - \text{lse}}$$

![Figure: Ring Attention passing the KV around the ring](../assets/figures/ring-attention.svg){.aig-svg}

```python title="ring_attention.py" torchrun="4"
import torch
import torch.distributed as dist

from cp_common import attention, make_qkv

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
q, k, v = make_qkv()
S = q.shape[0]
ref, _ = attention(q, k, v)
C = S // P
mine = slice(rank * C, (rank + 1) * C)
q_l, kv = q[mine], torch.stack([k[mine], v[mine]])          # this rank's query block and KV block

out, lse = None, None
src = rank                                                    # which rank the KV block in hand came from (that is, which block of the sequence it is)
computed = 0
for step in range(P):
    if src <= rank:                                           # causal: only a KV block no later than your own contributes
        o, l = attention(q_l, kv[0], kv[1], causal=True, q_offset=rank * C, k_offset=src * C)
        if out is None:
            out, lse = o, l
        else:                                                 # merge the two partial attentions with a log-sum-exp (online softmax)
            new = torch.logaddexp(lse, l)
            out = out * torch.exp(lse - new).T[..., None] + o * torch.exp(l - new).T[..., None]
            lse = new
        computed += 1
    if step < P - 1:                                          # pass the KV block to the right and receive the next one from the left
        recv = torch.empty_like(kv)
        reqs = [dist.isend(kv, (rank + 1) % P), dist.irecv(recv, (rank - 1) % P)]
        for r in reqs:
            r.wait()
        kv, src = recv, (src - 1) % P

ok = torch.tensor([int(torch.allclose(out, ref[mine], atol=1e-5))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
counts = [None] * P
dist.all_gather_object(counts, computed)
if rank == 0:
    print(f"Ring Attention，{P} 个 rank：输出与单进程的因果注意力一致：", bool(ok.item()))
    print("每个 rank 实际计算的块数：", counts)
dist.destroy_process_group()
```

```text title="output"
Ring Attention，4 个 rank：输出与单进程的因果注意力一致： True
每个 rank 实际计算的块数： [1, 2, 3, 4]
```

- Each step passes one KV block while computing the previous one, so the communication can be overlapped with the computation, as long as one block's computation takes longer than the transfer (which holds once the sequence is long enough).
- There is no limit on the head count, so it extends to many cards and across nodes.
- The problem is in the last line: **the causal mask makes the load uneven**. Rank 0's queries are at the front of the sequence and only need the first KV block, while rank 3 needs all 4. The slowest rank sets the pace.

### The zigzag partition {#之字形切分}

Cut the sequence into $2P$ blocks and give rank $r$ block $r$ and block $2P-1-r$: one early (little to compute) and one late (much to compute), exactly complementary:

```python title="cp_balance.py"
def work(query_chunks, n_chunks):
    """因果注意力里，一个 query 块需要和多少个 KV 块做计算（对角线上的块算半个）"""
    return sum(q + 0.5 for q in query_chunks)


P = 4
contiguous = {r: [2 * r, 2 * r + 1] for r in range(P)}          # partitioned in order: rank r takes blocks 2r and 2r+1 (of 2P blocks)
zigzag = {r: [r, 2 * P - 1 - r] for r in range(P)}               # zigzag: rank r takes block r and the (r+1)-th block from the end
for name, split in (("顺序切分", contiguous), ("之字形切分", zigzag)):
    print(name, [work(split[r], 2 * P) for r in range(P)])
```

```text title="output"
顺序切分 [2.0, 6.0, 10.0, 14.0]
之字形切分 [8.0, 8.0, 8.0, 8.0]
```

Megatron's context parallelism and Llama 3's long-context training both use this partition (or a similar striped one).

## How to choose {#怎么选}

| | Ulysses | Ring Attention |
| --- | --- | --- |
| Communication | 4 all-to-alls (q, k, v, the output) | $P-1$ point-to-point KV passes, overlappable with the computation |
| Degree ceiling | the head count (the KV head count under grouped-query attention) | essentially unlimited |
| Attention kernel | an off-the-shelf FlashAttention | one that supports blocks and log-sum-exp merging (or merging outside) |
| Suits | within a node, with enough heads | across nodes, very long sequences |

They can also be combined: Ulysses within a node and Ring between nodes (as in USP and similar schemes). Inference's prefill phase uses the same methods on a very long prompt (see [Pipeline and context parallelism](serving://distributed/pp-cp/) in the inference-systems handbook).

!!! interview "How to explain it"
    On context parallelism: the per-token operations partition along the sequence directly with no communication, and only attention needs other segments' KV. Ulysses uses two all-to-alls to transpose between partitioning the sequence and partitioning the heads, which allows an off-the-shelf attention kernel but limits the degree to the head count (the KV head count under grouped-query attention); Ring Attention passes the KV blocks round a ring and merges the blocks' results with a log-sum-exp, which overlaps with the computation and scales to many cards. The causal mask makes an in-order partition uneven (the last segment computes everything), and the zigzag partition, where each rank takes one early and one late block, gives every rank the same work at every step. Ulysses within a node and Ring between nodes can also be combined.

## Exercises {#练习}

1. On a grouped-query model (32 query heads, 8 KV heads), what is the highest degree of context parallelism Ulysses can use without replicating the KV? What would you do to use 16?

??? success "Answer"
    Each rank needs at least one complete KV head, so at most 8. To use 16, either replicate each KV head onto two ranks (replicating the KV by group before the all-to-all), or use 8-way Ulysses within a node and wrap 2-way Ring Attention around it.

2. In Ring Attention, does each step compute first and then pass, or pass first and then compute? How would you write it so that the communication overlaps the computation?

??? success "Answer"
    This chapter's implementation computes a block and then passes the next one synchronously, for clarity. To overlap, fire the next block's asynchronous `isend` and `irecv` **before** starting the current block's computation and `wait` afterwards: the communication then proceeds in the background alongside the computation. On a GPU the communication also has to go on its own stream. As long as each block's computation takes longer than the transfer (which gets easier the longer the sequence), the communication is hidden entirely.

## Summary {#小结}

- [x] Context parallelism partitions along the sequence; the per-token operations need no communication, and only attention needs other segments' KV.
- [x] Ulysses uses an all-to-all to transpose between partitioning the sequence and partitioning the heads, which allows an off-the-shelf attention kernel and limits the degree to the head count.
- [x] Ring Attention passes the KV blocks round a ring and merges the blocks' results with a log-sum-exp, which overlaps with the computation and scales well.
- [x] The causal mask makes an in-order partition uneven, and the zigzag partition gives every rank the same work.
