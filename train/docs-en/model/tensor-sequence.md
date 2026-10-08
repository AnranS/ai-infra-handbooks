# Tensor and sequence parallelism

<p class="lead">Tensor parallelism partitions every layer's weight matrices across several cards that compute at the same time, and it is shared between training and inference. [Tensor parallelism](serving://distributed/tensor-parallel/) in the inference-systems handbook covers the forward pass at inference time; training also has to handle the backward pass, namely how the gradients are synchronised and which parameters' gradients need an extra sum. This chapter implements Megatron-LM's tensor and sequence parallelism with four custom autograd functions, and verifies that the forward output and every shard's gradients match the single-process result.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why does Megatron's MLP partition the first matrix by column and the second by row? Is any communication needed in between?
    2. What do the two operators `f` and `g` in the Megatron paper do in the forward and backward passes?
    3. What does sequence parallelism partition? What does it replace the all-reduce with? Does the volume change?
    4. With sequence parallelism on, why do the LayerNorm weights' gradients need an extra all-reduce?
    5. How many times does tensor parallelism communicate per layer, and how much each time? Why is it usually kept to at most 8 cards?

??? success "Answers for the self-test (answer first, then open this)"
    1. The first matrix is partitioned along the output dimension (by column), so each card computes some of the intermediate result's columns; the activation function is elementwise, so each card does its own without communicating. The second matrix is partitioned along the input dimension (by row), which lines up with that card's columns and gives a full-shaped partial sum, finished with one all-reduce. Nothing is communicated in between.
    2. `f`: the identity forward (every card takes the same input), and an all-reduce of each card's input gradients backward. `g`: an all-reduce forward (summing the partial sums), and the identity backward. They sit at the entry and exit of a tensor-parallel region, and each one's forward is the other's backward.
    3. Sequence parallelism partitions the regions outside the tensor-parallel one, the LayerNorms, dropouts and residuals, along the sequence, so each card handles only $s/t$ tokens. The all-reduce becomes a reduce-scatter (leaving the tensor-parallel region) plus an all-gather (entering it); the volume is unchanged, but those regions' activations shrink by another factor of $t$.
    4. The LayerNorm weights are shared by every sequence shard, so each card has only computed the gradient contributed by its own segment of the sequence; they have to be summed within the tensor-parallel group (an all-reduce) to be the complete gradient.
    5. Twice per layer forward (once for attention and once for the MLP) and twice backward, each about $s \cdot b \cdot h$ elements, all on the critical path. As cards are added, each one's matrices shrink and the compute efficiency drops while the communication stays the same, so tensor parallelism stays inside the NVLink domain and is usually kept to at most 8.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/tensor-sequence.webp is in Chinese; put it back once the English version exists -->

## How Megatron partitions {#megatron-的切法}

![Figure: a tensor-parallel MLP, with the first matrix partitioned by column and the second by row, no communication in between and one all-reduce of the partial sums at the end](../assets/figures/tp-mlp.svg){.aig-svg}

An MLP block is $Y = \text{GELU}(X W_1^\top) W_2^\top$. Partition $W_1$ along the **output dimension** (the FFN's intermediate dimension) into $t$ pieces and $W_2$ along the **input dimension** into the matching $t$ pieces:

- Each card multiplies the complete $X$ by its own piece of $W_1$ and gets some of the intermediate result's columns; GELU is elementwise, so each card does its own and **nothing is communicated**.
- Each card multiplies its own columns by its own piece of $W_2$ and gets a full-shaped **partial sum**; adding the $t$ partial sums (an all-reduce) gives the final result.

Attention is the same: partition the $Q$, $K$ and $V$ projections by head (each card takes some heads) and the output projection along the input dimension, with an all-reduce at the end. So each Transformer layer's forward pass has two all-reduces, one for attention and one for the MLP.

The communication runs the other way in the backward pass. Megatron wraps them into two conjugate operators:

| Operator | Where it goes | Forward | Backward |
| --- | --- | --- | --- |
| `f` | entering the tensor-parallel region (before the column-partitioned matrix multiply) | the identity | all-reduce (each card's input gradient is only a part and has to be summed) |
| `g` | leaving the tensor-parallel region (after the row-partitioned matrix multiply) | all-reduce (summing the partial sums) | the identity |

## Sequence parallelism {#序列并行}

Outside the tensor-parallel region are the LayerNorms, dropouts and residual additions, which are computed and stored **in full on every card**: their activations are not partitioned (in the overview, with tensor parallelism but no sequence parallelism, the activations are $sbh(10 + 24/t)$, and that 10 is these). Sequence parallelism partitions these regions along the **sequence**, so each card handles only $s/t$ tokens. At the boundary between the two kinds of region:

- Entering the tensor-parallel region: the complete sequence is needed, so **all-gather** (along the sequence).
- Leaving it: sum the partial sums and partition back into sequence shards, so **reduce-scatter**.

An all-reduce is exactly a reduce-scatter plus an all-gather, so **the volume is unchanged**, while the activations at the LayerNorms and so on shrink to $1/t$. The price is one detail: the LayerNorm weights are shared by all of the sequence shards, so each card has computed only its own segment's contribution to the gradient and another all-reduce within the tensor-parallel group is needed.

## The implementation {#实现}

All four communication operators are custom `autograd.Function`s:

```python title="tp_ops.py"
import torch
import torch.distributed as dist


class CopyToTP(torch.autograd.Function):
    """Megatron 的 f：前向是恒等（每个 rank 拿同一份输入），反向把各 rank 对输入的梯度 all-reduce 求和。"""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad)
        return grad


class ReduceFromTP(torch.autograd.Function):
    """Megatron 的 g：前向 all-reduce 求和（合并行切分矩阵乘的部分和），反向是恒等。"""

    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class GatherSeq(torch.autograd.Function):
    """序列并行的 g-bar：前向沿序列维 all-gather（拼回完整序列），反向 reduce-scatter。"""

    @staticmethod
    def forward(ctx, x):
        out = x.new_empty(x.shape[0] * dist.get_world_size(), *x.shape[1:])
        dist.all_gather_into_tensor(out, x.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty(grad.shape[0] // dist.get_world_size(), *grad.shape[1:])
        dist.reduce_scatter_tensor(out, grad.contiguous())
        return out


class ScatterSeq(torch.autograd.Function):
    """序列并行的 f-bar：前向 reduce-scatter（求和并沿序列维切开），反向 all-gather。"""

    @staticmethod
    def forward(ctx, x):
        out = x.new_empty(x.shape[0] // dist.get_world_size(), *x.shape[1:])
        dist.reduce_scatter_tensor(out, x.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty(grad.shape[0] * dist.get_world_size(), *grad.shape[1:])
        dist.all_gather_into_tensor(out, grad.contiguous())
        return out
```

Verified with a LayerNorm and an MLP. Every rank first generates the same complete weights and input and computes the single-process reference result and gradients; then tensor parallelism and tensor plus sequence parallelism each compute it, and the output and **every shard's gradients** are compared:

```python title="tp_mlp_check.py" torchrun="4"
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tp_ops import CopyToTP, GatherSeq, ReduceFromTP, ScatterSeq

dist.init_process_group("gloo")
rank, tp = dist.get_rank(), dist.get_world_size()
S, H, FFN = 8, 16, 64                                   # the sequence length, the hidden dimension, the FFN's intermediate dimension

torch.manual_seed(0)                                    # every rank generates the same complete weights and input
x_full = torch.randn(S, H)
ln_w = torch.randn(H)
w1 = torch.randn(FFN, H) / H ** 0.5                     # up-projection: [FFN, H]
w2 = torch.randn(H, FFN) / FFN ** 0.5                   # down-projection: [H, FFN]
cols = slice(rank * FFN // tp, (rank + 1) * FFN // tp)  # the FFN intermediate dimensions this rank is responsible for
rows = slice(rank * S // tp, (rank + 1) * S // tp)      # the tokens this rank is responsible for under sequence parallelism


def block(x, ln_w, w1, w2):                             # the single-process reference: LayerNorm, up-projection, GELU, down-projection
    h = F.layer_norm(x, (H,), weight=ln_w)
    return F.linear(F.gelu(F.linear(h, w1)), w2)


ref_in = [t.clone().requires_grad_() for t in (x_full, ln_w, w1, w2)]
ref_out = block(*ref_in)
ref_out.square().sum().backward()
r_x, r_ln, r_w1, r_w2 = (t.grad for t in ref_in)

# ---- tensor parallelism: w1 partitioned by row (the output's columns), w2 by column (the input's rows), with each card doing its own GELU in between
x = x_full.clone().requires_grad_()
lw = ln_w.clone().requires_grad_()
w1_s = w1[cols].clone().requires_grad_()
w2_s = w2[:, cols].clone().requires_grad_()
h = CopyToTP.apply(F.layer_norm(x, (H,), weight=lw))    # f
out = ReduceFromTP.apply(F.linear(F.gelu(F.linear(h, w1_s)), w2_s))   # g: summing the partial sums
out.square().sum().backward()
tp_ok = [torch.allclose(out, ref_out, atol=1e-5), torch.allclose(w1_s.grad, r_w1[cols], atol=1e-5),
         torch.allclose(w2_s.grad, r_w2[:, cols], atol=1e-5), torch.allclose(x.grad, r_x, atol=1e-5)]

# ---- sequence parallelism: the LayerNorm handles only this rank's segment of the sequence; all-gather before entering the tensor-parallel region and reduce-scatter on leaving it
x = x_full[rows].clone().requires_grad_()
lw = ln_w.clone().requires_grad_()
w1_s = w1[cols].clone().requires_grad_()
w2_s = w2[:, cols].clone().requires_grad_()
h = GatherSeq.apply(F.layer_norm(x, (H,), weight=lw))   # the activations at the LayerNorm have only S/tp rows
out = ScatterSeq.apply(F.linear(F.gelu(F.linear(h, w1_s)), w2_s))
out.square().sum().backward()
dist.all_reduce(lw.grad)                                 # the LayerNorm weights are shared by all of the sequence segments: the gradients have to be summed within the tensor-parallel group
sp_ok = [torch.allclose(out, ref_out[rows], atol=1e-5), torch.allclose(x.grad, r_x[rows], atol=1e-5),
         torch.allclose(w1_s.grad, r_w1[cols], atol=1e-5), torch.allclose(lw.grad, r_ln, atol=1e-5)]

flags = torch.tensor([int(all(tp_ok)), int(all(sp_ok))])
dist.all_reduce(flags, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"TP={tp}：输出、w1 分片梯度、w2 分片梯度、输入梯度都与单进程一致：", bool(flags[0]))
    print(f"TP={tp} + 序列并行：输出分片、输入分片梯度、w1 分片梯度、LayerNorm 权重梯度都一致：", bool(flags[1]))
dist.destroy_process_group()
```

```text title="output"
TP=4：输出、w1 分片梯度、w2 分片梯度、输入梯度都与单进程一致： True
TP=4 + 序列并行：输出分片、输入分片梯度、w1 分片梯度、LayerNorm 权重梯度都一致： True
```

A few things worth noting:

- Without sequence parallelism, `x.grad` is the complete input gradient on every rank, because `f`'s backward summed each rank's partial gradients. Without that step, the gradient passed to the previous layer is wrong.
- With sequence parallelism, the LayerNorm's input and output have only $S/t$ rows, which is exactly the activation saved; the all-reduce of `lw.grad` is the extra step described above, and Megatron does the same for every parameter that lives in a sequence-parallel region and is shared across ranks.
- The shard gradients of `w1` and `w2` need no communication at all: each card's shard is used only by itself.

## Volume and scale {#通信量与规模}

Two all-reduces per layer forward (or two reduce-scatter plus all-gather pairs under sequence parallelism), each of $s \cdot b \cdot h$ elements, and two more backward. All of them are on the **critical path**: the next computation has to wait for the communication. So:

Work out the ratio of communication to computation:

<div class="aig-widget" data-widget="tp-comm"></div>

- Tensor parallelism is essentially done within one machine connected by NVLink, usually at a degree of at most 8.
- At too high a degree, each card's matrices shrink and the compute efficiency drops while the communication is unchanged, so beyond TP=8 it is rarely worth it.
- In practice the communication is overlapped with the computation: the matrix multiply is cut into blocks and each block is sent as the next is computed (Megatron's tp-comm-overlap, and the various GEMM kernels that fuse communication with computation).

Inference partitions the same way (see [Tensor parallelism](serving://distributed/tensor-parallel/) in the inference-systems handbook); the only difference is that there is no backward pass, and during decode each all-reduce's data is tiny and dominated by latency.

!!! interview "How to explain it"
    Tensor parallelism always comes with "derive Megatron's partitioning": the MLP's first matrix by column and the second by row, with the activation function in between elementwise and needing no communication, and one all-reduce of the partial sums at the end; attention by head. `f` (the identity forward, all-reduce backward) and `g` (all-reduce forward, the identity backward) appear as a pair at the two ends of the region. Sequence parallelism partitions the LayerNorm and dropout regions along the sequence, replacing the all-reduce with an all-gather plus a reduce-scatter: the volume is unchanged, the activations shrink by another factor of $t$, but the LayerNorm weights shared across the segments need their gradients all-reduced within the tensor-parallel group. Four communications per layer on the critical path is what keeps tensor parallelism inside the NVLink domain at a degree of at most 8.

## Exercises {#练习}

1. When an attention layer is partitioned by head, how are the $Q$, $K$ and $V$ projections and the output projection each partitioned? What do you do when a grouped-query model's KV head count (8, say) is smaller than the tensor-parallel degree (16, say)?

??? success "Answer"
    The $Q$, $K$ and $V$ projections are partitioned along the output dimension (by head), with each card taking $n_h/t$ query heads and the matching KV heads and computing its own heads' attention in full; the output projection is partitioned along the input dimension with an all-reduce at the end (the same as the MLP's second matrix).
    When there are fewer KV heads than the tensor-parallel degree, the KV heads cannot be partitioned further and have to be **replicated**: each KV head sits on $t / n_{kv}$ cards, while each card still handles only its own query heads. The price is that the KV projection weights and the KV cache exist in several copies.

2. Why are `f` and `g` called conjugate? If you wrongly used `g` at the start of the MLP and `f` at the end, what would go wrong in the forward and backward passes?

??? success "Answer"
    `f` is the identity forward and an all-reduce backward; `g` is an all-reduce forward and the identity backward, so each operator's forward is the other's backward.
    Swapped: the forward pass does an all-reduce at the start (summing $t$ identical inputs, inflating the input by a factor of $t$) and fails to sum the partial sums at the end (so the output is only a partial sum); and in the backward pass the input gradients are never summed, so each card gets only a part. Both the output and the gradients are wrong, which is exactly why each kind of parallelism is verified item by item against a single process.

## Summary {#小结}

- [x] Megatron's tensor parallelism: the first matrix by column and the second by row, with the elementwise operations in between needing no communication and one all-reduce of the partial sums at the end.
- [x] `f` (the identity forward, all-reduce backward) and `g` (all-reduce forward, the identity backward) appear as a pair at the two ends of a tensor-parallel region.
- [x] Sequence parallelism partitions the LayerNorm regions along the sequence and replaces the all-reduce with an all-gather plus a reduce-scatter: the volume is unchanged and the activations shrink by another factor of $t$, but shared parameters' gradients have to be summed within the tensor-parallel group.
- [x] Four communications per layer on the critical path is what makes tensor parallelism suit the NVLink domain alone, at a degree usually no higher than 8.
