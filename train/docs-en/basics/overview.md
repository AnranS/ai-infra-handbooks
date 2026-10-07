# An overview of distributed training: the memory budget and the time model

<p class="lead">Why parallelise? Because one card can neither hold the model nor finish the computation. This chapter turns both of those into numbers: which items make up each card's memory budget and what each one scales with, and how much compute, how much time and how much communication the training takes. With those two accounts in hand, every kind of parallelism that follows is answering the same question: which item does it partition, and how much communication does that cost.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Training a 70B model with bf16 mixed precision and Adam, how many bytes does each parameter take?
    2. What are the activations during training proportional to? How can they be reduced?
    3. Roughly how many FLOPs does training a 70B model on 1T tokens take?
    4. What do data, tensor, pipeline, context and expert parallelism each partition?
    5. Why is tensor parallelism usually done only inside one machine?

??? success "Answers for the self-test (answer first, then open this)"
    1. About 16 bytes: 2 for the bf16 parameter and 2 for the gradient, plus 4 for the fp32 master weight and 4 each for Adam's first and second moments. For a 70B model that is about 1.1 TB, counting the model states alone.
    2. Proportional to the sequence length s, the micro-batch size b, the hidden dimension h and the layer count (about $34\,sbh$ bytes per layer, plus a term proportional to $s^2$ if the attention scores are materialised). The ways to reduce them: activation recomputation, sequence or context parallelism to partition the activations, a smaller micro-batch, and FlashAttention.
    3. About $6ND = 6 \times 70 \times 10^9 \times 10^{12} \approx 4.2 \times 10^{23}$ floating-point operations.
    4. Data parallelism partitions the data (each card holds a complete model, and ZeRO or FSDP then partitions the model states); tensor parallelism partitions the matrices within a layer; pipeline parallelism partitions by layer; context parallelism partitions along the sequence; expert parallelism spreads a mixture of experts' experts across cards.
    5. Every layer does two all-reduces on the critical path (and two more in the backward pass), the next computation has to wait for them, and the volume is as large as the activations. Only NVLink inside a machine has the bandwidth (about 10 times the inter-machine network); across machines the communication drags it down.

## The memory budget {#显存账本}

The memory on each card during training has four parts ($\Psi$ is the parameter count that card is responsible for):

| Item | Size | Notes |
| --- | --- | --- |
| Parameters | $2\Psi$ | bf16 |
| Gradients | $2\Psi$ | bf16 |
| Optimizer states | $12\Psi$ | the fp32 master parameters and Adam's first and second moments |
| Activations | proportional to batch size x sequence length x hidden dimension x layers | the intermediates saved in the forward pass for the backward one |

The first three together are the **model states**, 16 bytes per parameter. A precise estimate of the activations (the formula from the Megatron paper, in bf16 with FlashAttention) is $34\,sbh$ bytes per layer, where $s$ is the sequence length, $b$ the micro-batch size and $h$ the hidden dimension; tensor parallelism with sequence parallelism on divides that by the tensor-parallel degree, and full recomputation brings it down to $2\,sbh$ per layer (storing only each layer's input).

Applying those rules to training a 70B model on 64 cards of 80 GB with an 8K sequence:

```python title="ledger.py"
GiB = 2**30

# a dense model on the scale of Llama-3-70B
PSI, LAYERS, HIDDEN = 70.6e9, 80, 8192
SEQ, MBS = 8192, 1            # the sequence length and the micro-batch size


def model_states(psi, zero, dp):
    """混合精度 + Adam：bf16 参数 2 + bf16 梯度 2 + fp32 主参数/一阶矩/二阶矩 12 = 16 字节/参数"""
    return {0: 16 * psi, 1: 4 * psi + 12 * psi / dp, 2: 2 * psi + 14 * psi / dp, 3: 16 * psi / dp}[zero]


def activations(layers, tp=1, sp=False, recompute=False, inflight=1):
    """每层激活（bf16，用 FlashAttention）：不切分 34·sbh；TP 不开 SP 时 sbh·(10 + 24/t)；开 SP 时 34·sbh/t；全量重计算 2·sbh/t"""
    sbh = SEQ * MBS * HIDDEN
    if recompute:
        per = 2 * sbh / tp
    elif sp:
        per = 34 * sbh / tp
    else:
        per = sbh * (10 + 24 / tp)
    return per * layers * inflight


configs = [
    ("DP=64，不切分", dict(zero=0, dp=64, tp=1, pp=1)),
    ("DP=64，ZeRO-3", dict(zero=3, dp=64, tp=1, pp=1)),
    ("DP=64，ZeRO-3 + 全量重计算", dict(zero=3, dp=64, tp=1, pp=1, recompute=True)),
    ("TP=8（SP）× PP=4 × DP=2，ZeRO-1", dict(zero=1, dp=2, tp=8, pp=4, sp=True)),
]
for name, c in configs:
    psi_local = PSI / (c["tp"] * c["pp"])                  # the parameters this card is responsible for
    states = model_states(psi_local, c["zero"], c["dp"])
    layers_local = LAYERS // c["pp"]
    inflight = c["pp"]                                     # 1F1B: the first stage holds the activations of at most pp micro-batches at once
    act = activations(layers_local, c["tp"], c.get("sp", False), c.get("recompute", False), inflight)
    total = (states + act) / GiB
    print(f"{name}：模型状态 {states / GiB:.1f} GiB，激活 {act / GiB:.1f} GiB，合计 {total:.1f} GiB，"
          f"{'放得下' if total < 80 * 0.9 else '放不下'}")

TOKENS = 1e9
flops = 6 * PSI * TOKENS
seconds = flops / (64 * 989e12 * 0.40)
print(f"训练 10 亿 token：{flops:.2e} FLOPs，64 张 H100、MFU 40% 需要 {seconds / 3600:.1f} 小时")
```

```text title="output"
DP=64，不切分：模型状态 1052.0 GiB，激活 170.0 GiB，合计 1222.0 GiB，放不下
DP=64，ZeRO-3：模型状态 16.4 GiB，激活 170.0 GiB，合计 186.4 GiB，放不下
DP=64，ZeRO-3 + 全量重计算：模型状态 16.4 GiB，激活 10.0 GiB，合计 26.4 GiB，放得下
TP=8（SP）× PP=4 × DP=2，ZeRO-1：模型状态 20.5 GiB，激活 21.2 GiB，合计 41.8 GiB，放得下
训练 10 亿 token：4.24e+20 FLOPs，64 张 H100、MFU 40% 需要 4.6 小时
```

Those four lines are almost the outline of the whole book:

1. **Partitioning nothing**: the model states alone are 1 TB, which will not fit on one card. Data parallelism by itself saves no memory.
2. **ZeRO-3** spreads the model states over 64 cards, leaving only 16 GiB, but the activations for an 8K sequence are 170 GiB, so the activations become the new bottleneck.
3. **Full recomputation** squeezes the activations to 10 GiB and it fits, at the price of about 33% more compute (one extra forward pass per layer).
4. **Tensor parallelism with sequence parallelism plus pipeline parallelism** partitions the model states and the activations at once and fits without recomputation, at the price of tensor parallelism's frequent communication and the pipeline's bubble.

The chapters that follow implement each of these and measure what it costs.

## The time model {#时间模型}

Training on one token takes about $6N$ operations ($2N$ forward and $4N$ backward, where $N$ is the parameter count), so the whole training is about $6ND$ floating-point operations. The actual time depends on the **model FLOPs utilization**:

$$T = \frac{6ND}{\text{cards} \times \text{peak per card} \times \text{MFU}}$$

<div class="aig-widget" data-widget="train-time"></div>

In the example above, 64 H100s take 4.6 hours to train on a billion tokens; on 15T tokens that would be about 8 years, which is why frontier models use tens of thousands of cards. The model FLOPs utilization is usually between 30% and 50%, and what is lost goes to:

- **Communication that is not hidden behind computation**: gradient synchronisation, tensor parallelism's all-reduces, the pipeline's point-to-point transfers, expert parallelism's all-to-all.
- **The pipeline bubble**: some stages are waiting.
- **Recomputation**: the extra forward passes do not count toward the utilization.
- **Small kernels and CPU overhead**: elementwise operations, the optimizer step, data loading.

Every kind of parallelism trades memory against time, and the next section is one table of all of them.

## The five kinds of parallelism at a glance {#五种并行一览}

| Parallelism | What it partitions | What it communicates per step | Volume | Which layer of network |
| --- | --- | --- | --- | --- |
| Data parallelism (DP / ZeRO) | the data (the batch); ZeRO also partitions the model states | an all-reduce of the gradients (ZeRO: reduce-scatter plus all-gather) | about $2\Psi$ to $3\Psi$ bytes per step, overlappable with the backward pass | across nodes (the outermost) |
| Tensor parallelism (TP) plus sequence parallelism (SP) | each layer's weight matrices; SP also partitions the activations at the LayerNorms and similar | 2 all-reduces per layer (or reduce-scatter plus all-gather), on the critical path | about $4\,sbh$ bytes per layer | within a node (NVLink) |
| Pipeline parallelism (PP) | the layers (cut by depth into stages) | the activations and gradients between neighbouring stages (point-to-point) | about $sbh$ bytes per micro-batch | may cross nodes |
| Context parallelism (CP) | the sequence | the KV that attention needs (ring), or an all-to-all over the heads (Ulysses) | proportional to the sequence length | within or across nodes |
| Expert parallelism (EP) | the mixture of experts' experts | an all-to-all of the tokens (dispatch and combine) | about $2 \times$ top-k $\times sbh$ bytes per layer | within or across nodes |

The reason tensor parallelism stays inside a node is in this table: its communication is frequent (twice per layer), on the critical path (it cannot be fully overlapped with computation) and large, so only NVLink (several hundred GB/s one way) can carry it. InfiniBand or RoCE across nodes gives each card only tens of GB/s, which suits data and pipeline parallelism, whose volume is small or which can be overlapped with computation.
Combining several kinds gives 3D parallelism (DP x TP x PP) or even 5D (adding CP and EP); how to combine them is in [Combining and choosing 3D and 5D parallelism](../practice/strategy.md).

!!! interview "How to answer in an interview"
    Asked what it takes to train a 70B model, start with the memory budget: bf16 mixed precision with Adam is 16 bytes per parameter (the bf16 parameter and gradient plus the fp32 master weight and two moments), so 70B is 1.1 TB and will not fit on one card; the activations are about $34\,sbh$ bytes per layer, proportional to the sequence length and the batch. Then the time: training takes about $6ND$ operations, so 70B on 1T tokens is about $4.2 \times 10^{23}$, converted into card-hours at a measured model FLOPs utilization. Finish with what each kind of parallelism partitions (data parallelism the data, tensor parallelism the matrices within a layer, pipeline parallelism the layers, context parallelism the sequence, expert parallelism the experts) and the placement rule: tensor parallelism, whose communication is frequent and on the critical path, goes inside a node, while data and pipeline parallelism, which overlap or are small, go between nodes.

## Exercises {#练习}

1. An 8B model (32 layers, hidden dimension 4096) trains on 8 cards of 80 GB with a sequence length of 4096 and a micro-batch of 2. With data parallelism and ZeRO-2 only and no recomputation, how much memory does each card need? Does it fit? If it does not, what would you add first?

??? success "Answer"
    The model states: under ZeRO-2 that is $2\Psi + 14\Psi/8 = 16 + 14 = 30$ GB (about 27.9 GiB). The activations: $34 \times 4096 \times 2 \times 4096 \times 32 = 36.5$ GB (about 34 GiB). About 62 GiB in all, which just fits on an 80 GB card (leaving 10% headroom).
    If the sequence or the micro-batch has to grow, add **selective recomputation** (recomputing only the attention part) or full recomputation rather than tensor parallelism: tensor parallelism's communication is not worth it for an 8B model on 8 cards.

2. Why is it said that data parallelism saves no memory, and yet almost all large-scale training uses it?

??? success "Answer"
    The purpose of data parallelism is **speed**: each card handles different data and the throughput grows linearly with the card count, while its communication (synchronising the gradients) happens once per step, is a fixed $2\Psi$ or so, and overlaps with the backward computation, which scales very well.
    The memory problem goes to ZeRO (partitioning the model states) and the other kinds of parallelism; in the final configuration, data parallelism is almost always the outermost dimension, the one that crosses nodes.

## Summary {#小结}

- [x] The memory budget: the model states are 16 bytes per parameter (the bf16 parameter and gradient plus the fp32 optimizer states), and the activations are about $34\,sbh$ bytes per layer.
- [x] ZeRO partitions the model states, tensor plus sequence parallelism partitions both the model states and the activations, recomputation trades compute for activations, and pipeline parallelism cuts by layer.
- [x] Training takes about $6ND$ operations; the actual time depends on the model FLOPs utilization, and what is lost goes to unhidden communication, the bubble, recomputation and small kernels.
- [x] The parallelism whose communication is frequent and on the critical path (tensor parallelism) goes inside a node; the ones that overlap or are small (data and pipeline parallelism) go between nodes.
