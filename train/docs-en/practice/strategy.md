# Combining and choosing 3D and 5D parallelism

<p class="lead">Each chapter so far has covered one kind of parallelism; real training always uses several at once: tensor parallelism partitions the matrices within a layer inside a node, context parallelism partitions the sequence, pipeline parallelism partitions by depth, data parallelism scales on the outside, and a mixture-of-experts model adds expert parallelism. This chapter answers two questions: how the several kinds are laid out across the cards (which dimension goes inside a node), and how, given a model, a sequence length, a global batch and a card count, to pick a sensible set of degrees. It ends with a search script of a few dozen lines that puts the earlier chapters' memory and time formulas together, to see what it picks and why.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. In what order are the 5 kinds of parallelism laid out across the cards? Why is tensor parallelism always innermost?
    2. What are the usual steps for choosing a parallel configuration for a model?
    3. Why does a higher tensor-parallel degree mean a larger share of communication?
    4. What happens when the cards keep being added and the global batch cannot grow any further?
    5. Why do the first and last pipeline stages often hold fewer layers?

??? success "Answers for the self-test (answer first, then open this)"
    1. From the inside out: tensor parallelism (NVLink within a node), then context parallelism, then data and pipeline parallelism on the outside, with expert parallelism reusing the data-parallel cards. Tensor parallelism communicates most often, on the critical path and in volume, so it has to go on the innermost layer with the highest bandwidth.
    2. First make the model states fit (ZeRO, then tensor parallelism, then pipeline parallelism or FSDP), then make the activations fit (sequence parallelism, context parallelism, recomputing some layers), give the remaining cards to data parallelism, and finally tune the pipeline (the micro-batch count, how the layers are divided). Estimate to narrow the field, then measure the top few configurations with a profiler.
    3. As the tensor-parallel degree grows, each card's computation falls as $1/t$ while each all-reduce's volume is unchanged (and even grows slightly per card), so the communication share rises roughly as $(t-1)/h$, and the compute efficiency also falls because the matrices are smaller.
    4. The data-parallel count times the micro-batch per share cannot exceed the global batch; with more cards, data parallelism cannot scale further and more tensor, pipeline or context parallelism has to absorb them, but those are less efficient (communication, bubbles), so the overall scaling efficiency falls.
    5. The first stage also has the embedding and the last has the output layer and the loss (heavy with a large vocabulary), so they compute more and use more memory than the middle stages; giving them fewer Transformer layers evens the stages out.

## Laying out the dimensions: what goes inside a node {#维度的排布谁放在节点内}

Think of the cards as a multidimensional array (a device mesh), with each dimension one kind of parallelism; the cards along one line of a dimension form that kind's communication group. Cards with adjacent ranks are in the same node, so **the innermost dimension falls within a node**:

![Figure: 3D parallelism laid out over a cluster](../assets/figures/parallelism-3d.svg){.aig-svg}

```python title="mesh.py"
import numpy as np

GPUS_PER_NODE = 8
TP, CP, DP, PP = 2, 2, 2, 2                    # 16 cards: 2 nodes

# from the outside in, PP, DP, CP, TP: the innermost dimension's ranks are adjacent and fall within one node
mesh = np.arange(PP * DP * CP * TP).reshape(PP, DP, CP, TP)
axes = {"TP": 3, "CP": 2, "DP": 1, "PP": 0}

for name, axis in axes.items():
    groups = np.moveaxis(mesh, axis, -1).reshape(-1, mesh.shape[axis])   # each line along this dimension is one communication group
    local = all(len({r // GPUS_PER_NODE for r in g}) == 1 for g in groups)
    print(f"{name} 组：{' '.join(str(g.tolist()) for g in groups[:4])} …共 {len(groups)} 组，"
          f"{'都在节点内' if local else '跨节点'}")

rank = 13
pp, dp, cp, tp = (int(i[0]) for i in np.nonzero(mesh == rank))
print(f"rank {rank} 的坐标：PP={pp} DP={dp} CP={cp} TP={tp}")
```

```text title="output"
TP 组：[0, 1] [2, 3] [4, 5] [6, 7] …共 8 组，都在节点内
CP 组：[0, 2] [1, 3] [4, 6] [5, 7] …共 8 组，都在节点内
DP 组：[0, 4] [1, 5] [2, 6] [3, 7] …共 8 组，都在节点内
PP 组：[0, 8] [1, 9] [2, 10] [3, 11] …共 8 组，跨节点
rank 13 的坐标：PP=1 DP=1 CP=0 TP=1
```

This is how Megatron-LM and torchtitan build their communication groups: Megatron takes `--tensor-model-parallel-size` and similar parameters plus an order string (`tp-cp-ep-dp-pp` by default, from the inside out), while torchtitan uses PyTorch's `init_device_mesh("cuda", (pp, dp, cp, tp), mesh_dim_names=(...))` and takes each dimension's process group out of the mesh by name.

The layout principle comes from the table in [the overview](../basics/overview.md):

- **Tensor parallelism is always innermost**: two communications per layer in each of the forward and backward passes, on the critical path, which only NVLink can carry; so its degree generally does not exceed a node's card count (8).
- **Context parallelism comes next**: the ring's KV transfers can overlap with the attention computation, but they are not fully hidden when the sequence is not very long, so inside a node is best.
- **Data and pipeline parallelism go outside**: data parallelism synchronises gradients once per step and overlaps with the backward pass; the pipeline only passes activations point-to-point between neighbouring stages, in small volumes. Either can be outermost and both are used: Megatron defaults to the pipeline outermost, while Llama 3's order is `[TP, CP, PP, DP]` with data parallelism (FSDP) outermost, because FSDP's communication can be fired early and tolerates inter-node latency best.
- **Expert parallelism borrows the data-parallel dimension**: a mixture-of-experts layer's expert parallelism takes no extra cards and instead regroups the cards that were doing data (and context) parallelism for attention (see [Mixture of experts and expert parallelism](../model/moe-ep.md)).

## The steps for choosing {#选择的步骤}

What most teams do comes down to these steps (HuggingFace's *Ultra-Scale Playbook* gives the same procedure with a great deal of measurement):

1. **Make the model states fit first**: start from data parallelism with ZeRO-1 (the distributed optimizer); if even the bf16 parameters and gradients (4 bytes per parameter) do not fit, add tensor parallelism within a node; if one node's worth of it is not enough, add pipeline parallelism to cut layers across nodes, or switch to ZeRO-3 / FSDP.
2. **Then make the activations fit**: tensor parallelism always comes with sequence parallelism; add context parallelism for a very long sequence; close the remaining gap with full recomputation on **some layers**, recomputing only the ones that do not fit.
3. **Scale to the target card count with data parallelism**: give the remaining cards to data parallelism; but the global batch has a ceiling (too large damages convergence), and data parallelism times the micro-batch size times the micro-batches per pipeline equals the global batch, so it cannot grow without bound.
4. **Tune the pipeline**: the micro-batch count has to be far larger than the stage count to amortise the bubble; the first and last stages take fewer layers (the embedding, the LM head and the loss are there); if the bubble is still large, use an interleaved or zero-bubble schedule (see [Pipeline parallelism](../model/pipeline.md)).
5. **Measure**: use a profiler to see how much of each step is computation, each kind of communication and idling, compare it against the estimate, and tune again.

The first four steps can all be estimated on paper, which is what the script below does.

## A configuration search script {#一个配置搜索脚本}

The script puts the earlier chapters' formulas together: memory from [the overview](../basics/overview.md)'s budget (ZeRO-1, $34\,sbh/t$ of activations per layer, recomputation on some layers); time accumulated as how long one micro-batch takes on the slowest stage, where tensor parallelism's all-reduce does not overlap, context parallelism's ring overlaps with the attention computation, most of data parallelism's gradient synchronisation hides behind the backward pass, the pipeline adds $p-1$ micro-batches of bubble, and the last stage computes an extra LM head. It enumerates every tensor x context x pipeline combination and sorts by the estimated model FLOPs utilization:

```python title="parallel_plan.py"
from dataclasses import dataclass
from itertools import product

GiB = 2**30
PEAK, EFF = 989e12, 0.6                 # the H100's dense BF16 peak; the fraction a large matrix multiply actually reaches
NVLINK, NET = 450e9, 50e9               # one-way bandwidth per card: NVLink within a node, the network card between nodes (400 Gb/s)
NODE, MEM = 8, 80e9 * 0.9               # 8 cards per node; 80 GB per card, leaving 10% for fragmentation and temporary buffers
MIN_CP_CHUNK = 4096                     # the minimum tokens per context-parallel rank, below which the ring's steps are too small


@dataclass
class Model:
    N: float     # the parameter count (including the vocabulary)
    L: int       # the layer count
    H: int       # the hidden dimension
    KV: int      # the width of one row of K or V (grouped-query attention: KV heads x head dimension)
    V: int       # the vocabulary size


def bandwidth(inner, size):
    """维度由内到外依次是 TP → CP → DP → PP：一个并行组的跨度不超过一个节点就走 NVLink"""
    return NVLINK if inner * size <= NODE else NET


def plan(m, seq, gbs, gpus, tp, cp, pp, mbs=1):
    if tp > NODE or gpus % (tp * cp * pp) or m.L % pp or (cp > 1 and seq // cp < MIN_CP_CHUNK):
        return None
    dp = gpus // (tp * cp * pp)
    if gbs % (dp * mbs):
        return None
    micro = gbs // (dp * mbs)                          # the micro-batches each pipeline runs per step
    s, layers = seq // cp, m.L // pp                   # the sequence length and layer count on each card

    # ---- memory: the first stage is tightest (the most parameters, and pp micro-batches' activations at once)
    psi = m.N / (tp * pp)
    states = 4 * psi + 12 * psi / (dp * cp)            # ZeRO-1: the optimizer states are partitioned over DP x CP
    sbh = s * mbs * m.H / tp
    for rc in range(layers + 1):                       # add fully recomputed layers from 0 upward until it fits
        act = (34 * (layers - rc) + 2 * rc) * sbh * min(pp, micro)
        if states + act <= MEM:
            break
    else:
        return None

    # ---- time: how long one micro-batch takes on the slowest stage (the last, with the extra LM head)
    f_layer = 6 * (m.N - 2 * m.V * m.H) / m.L + 6 * seq * m.H     # per token per layer: the linear layers plus causal attention
    f_head = 6 * m.V * m.H
    sec = s * mbs / tp / (PEAK * EFF)                  # the seconds per FLOP per token on this card
    avg = (m.L * f_layer + f_head) / pp * sec          # the computation an average stage gets
    recompute = rc * f_layer / 3 * sec                 # a recomputed layer does one extra forward pass (the forward pass is 1/3 of forward plus backward)
    slow = layers * f_layer * sec + recompute + f_head * sec
    act_bytes = s * mbs * m.H * 2
    tp_c = layers * 4 * 2 * (tp - 1) / tp * act_bytes / bandwidth(1, tp)            # twice per layer forward and twice backward, not overlapped
    ring = layers * 3 * (cp - 1) * (2 * s * mbs * m.KV * 2) / bandwidth(tp, cp)      # the ring passes the KV (and dKV in the backward pass)
    cp_c = max(0.0, ring - layers * 6 * seq * m.H * sec)                           # overlapped with the attention computation, so only the exposed part counts
    pp_c = 2 * act_bytes / tp / bandwidth(tp * cp * dp, pp) * 0.5 if pp > 1 else 0  # the point-to-point activations and gradients, half of it hidden
    t_mb = slow + tp_c + cp_c + pp_c
    dpc = dp * cp
    dp_c = 2 * (dpc - 1) / dpc * 2 * psi / bandwidth(tp, dpc) * 0.2                  # the gradients' reduce-scatter plus the parameters' all-gather, 80% overlapped with the backward pass
    parts = {"计算": micro * avg, "重算": micro * recompute, "不均衡": micro * (slow - recompute - avg),
             "TP": micro * tp_c, "CP": micro * cp_c, "PP": micro * pp_c, "气泡": (pp - 1) * t_mb, "DP": dp_c}
    step = sum(parts.values())
    model_flops = (m.L * f_layer + f_head) * gbs * seq          # excluding recomputation: the utilization counts only the operations the model itself requires
    return dict(tp=tp, cp=cp, pp=pp, dp=dp, mfu=model_flops / (step * gpus * PEAK),
                mem=(states + act) / GiB, rc=rc, layers=layers, step=step, parts=parts)


def show(p):
    share = " ".join(f"{k} {v / p['step']:.0%}" for k, v in p["parts"].items() if v / p["step"] >= 0.005)
    rc = f"，重算 {p['rc']}/{p['layers']} 层" if p["rc"] else ""
    print(f"TP={p['tp']} CP={p['cp']} PP={p['pp']} DP={p['dp']}：MFU {p['mfu']:.1%}，"
          f"{p['mem']:.0f} GiB/卡{rc}｜{share}")


def search(title, m, seq, gbs, gpus, top=3, extra=()):
    plans = [p for tp, cp, pp in product((1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16))
             if (p := plan(m, seq, gbs, gpus, tp, cp, pp))]
    plans.sort(key=lambda p: -p["mfu"])
    print(f"== {title}：{len(plans)} 种配置放得下")
    for p in plans[:top]:
        show(p)
    for tp, cp, pp in extra:                           # then see where a few intuitive configurations rank
        rank, p = next((i, q) for i, q in enumerate(plans, 1) if (q["tp"], q["cp"], q["pp"]) == (tp, cp, pp))
        print(f"  第 {rank} 名：", end="")
        show(p)


llama70b = Model(N=70.6e9, L=80, H=8192, KV=1024, V=128256)
llama8b = Model(N=8.03e9, L=32, H=4096, KV=1024, V=128256)

if __name__ == "__main__":
    search("70B，64 卡，8K 序列，每步 4M token", llama70b, 8192, 512, 64, extra=[(8, 1, 1), (8, 1, 4)])
    search("70B，64 卡，128K 序列，每步 4M token", llama70b, 131072, 32, 64, extra=[(8, 1, 1)])
    search("8B，8 卡，8K 序列，每步 0.5M token", llama8b, 8192, 64, 8, extra=[(1, 1, 1)])
```

```text title="output"
== 70B，64 卡，8K 序列，每步 4M token：24 种配置放得下
TP=4 CP=2 PP=2 DP=4：MFU 53.7%，66 GiB/卡｜计算 89% 不均衡 1% TP 8% 气泡 1%
TP=4 CP=2 PP=4 DP=2：MFU 52.2%，50 GiB/卡｜计算 87% 不均衡 4% TP 8% 气泡 1%
TP=4 CP=2 PP=8 DP=1：MFU 49.6%，42 GiB/卡｜计算 83% 不均衡 8% TP 8% 气泡 1%
  第 5 名：TP=8 CP=1 PP=1 DP=8：MFU 49.3%，66 GiB/卡｜计算 82% TP 18%
  第 9 名：TP=8 CP=1 PP=4 DP=2：MFU 47.1%，42 GiB/卡｜计算 78% 不均衡 3% TP 17% 气泡 1%
== 70B，64 卡，128K 序列，每步 4M token：23 种配置放得下
TP=8 CP=4 PP=2 DP=1：MFU 44.4%，66 GiB/卡，重算 24/40 层｜计算 74% 重算 15% 不均衡 1% TP 8% 气泡 3%
TP=4 CP=8 PP=2 DP=1：MFU 44.2%，66 GiB/卡，重算 32/40 层｜计算 74% 重算 19% TP 3% 气泡 3%
TP=8 CP=4 PP=1 DP=2：MFU 43.8%，66 GiB/卡，重算 64/80 层｜计算 73% 重算 19% TP 8%
  第 6 名：TP=8 CP=1 PP=1 DP=8：MFU 41.8%，65 GiB/卡，重算 80/80 层｜计算 70% 重算 23% TP 7%
== 8B，8 卡，8K 序列，每步 0.5M token：16 种配置放得下
TP=1 CP=2 PP=1 DP=4：MFU 59.9%，58 GiB/卡｜计算 100%
TP=2 CP=1 PP=1 DP=4：MFU 56.9%，43 GiB/卡｜计算 95% TP 5%
TP=2 CP=2 PP=1 DP=2：MFU 56.9%，35 GiB/卡｜计算 95% TP 5%
  第 4 名：TP=1 CP=1 PP=1 DP=8：MFU 55.0%，66 GiB/卡，重算 9/32 层｜计算 92% 重算 8%
```

The ceiling on the utilization is the 60% matrix-multiply efficiency the script assumes, and each line is followed by the breakdown of one step's time. There is a lot to read in these few lines.

**The larger the tensor-parallel degree, the larger the communication share.** In the 70B, 8K case, the textbook configuration of tensor parallelism filling a node with data parallelism between nodes comes 5th, with 18% of the time in tensor-parallel communication; at TP=4 it is only 8%. The reason: each card's computation per layer is proportional to $1/t$ while tensor parallelism's volume is some multiple of $2(t-1)/t \cdot sbh$ and barely falls with $t$, so the ratio grows with $t - 1$: going from $t=4$ to 8 multiplies it by about $7/3$. The ratio is also inversely proportional to $h$ (each layer's computation is about $\propto h^2$ and its communication $\propto h$), so tensor parallelism is even less worthwhile for a small model: in the 8B case, TP=2 already costs 5% in communication.

**Context parallelism takes over the activations from tensor parallelism here.** The top configuration drops tensor parallelism to 4 and uses CP=2 to halve the sequence, which halves the activations just the same. Grouped-query attention leaves K and V at only 1/8 of the hidden dimension, so the ring moves far less data than tensor parallelism's all-reduce, and it hides behind the attention computation. The script requires at least 4K tokens per context-parallel rank, so an 8K sequence allows at most CP=2.

**The pipeline has almost no bubble when the batch is large enough, but it is unbalanced.** At DP=4 each pipeline runs 128 micro-batches per step, and 1F1B's bubble of $(p-1)/(m+p-1)$ is only 1%. The real cost is the extra LM head on the last stage: an output layer over a 128K vocabulary is about 1.1 Transformer layers of computation, and at PP=8 each stage has only 10 layers, so the last stage is about 10% slower than the average and the others wait for it (the imbalance in the table above grows from 1% to 8%). So real training gives the first and last stages one layer fewer (which is what Llama 3 does), and Megatron has parameters for it (`--decoder-first-pipeline-num-layers` and `--decoder-last-pipeline-num-layers`).

**A long sequence uses context parallelism to avoid recomputation.** At 128K, every configuration on 64 cards needs recomputation; without context parallelism all 80 layers have to be recomputed (the 6th place). CP=4 takes each card's activations to a quarter and only 24 of 40 layers have to be recomputed, 2.6 percentage points better on utilization. With more cards the context-parallel degree goes higher still: Llama 3 405B used CP=16 in its 128K phase.

**Do not reach for model parallelism on a small model.** For 8B on 8 cards, the best configuration is only 5 percentage points ahead of plain data parallelism with 9 layers recomputed. Many engineers would take the latter (or FSDP) because it is the simplest and needs no repartitioning when the model or the sequence length changes.

### What the model leaves out {#模型没有算进去的东西}

This script exists only to **narrow the field**, and its assumptions are all rough:

- The matrix-multiply efficiency is a constant. In reality, the finer the tensor or context partition, the smaller each card's matrices and the lower the efficiency.
- Tensor parallelism's communication is treated as not overlapping at all. Megatron's `--tp-comm-overlap` splits the all-gather and reduce-scatter into chunks that overlap the neighbouring matrix multiplies, which hides a large part of it.
- It ignores the LM head's logits ($s \times V$ fp32 values per micro-batch, which at 8K x 128K is 4 GB), temporary buffers, communication buffers and memory fragmentation.
- It ignores the optimizer step, data loading, kernel launches, communication latency (which dominates for small messages) and stragglers.

So after estimating, **measure on the machines**: run a few dozen steps of each of the top configurations, look at the time breakdown with a profiler, and choose among them. The script's value is in ruling out the combinations that clearly will not do, and telling you roughly where each configuration's bottleneck lies.

## Two published real configurations {#两个公开的真实配置}

| | Llama 3 405B (dense) | DeepSeek-V3 671B (mixture of experts, 37B active) |
| --- | --- | --- |
| Cluster | up to 16K H100s | 2048 H800s |
| Configuration | 8K sequence: TP8 x PP16 x DP128; 128K sequence: TP8 x CP16 x PP16 x DP8 | PP16 x EP64 (across 8 nodes) x data parallelism with ZeRO-1, **no tensor parallelism** |
| Dimension order | `[TP, CP, PP, DP]` from the inside out, with FSDP for data parallelism | expert parallelism reuses the data-parallel cards |
| Pipeline | interleaved, with one layer fewer on each of the first and last stages | DualPipe: a bidirectional pipeline overlapping expert parallelism's all-to-all with the computation |
| Efficiency | 38% to 43% BF16 model FLOPs utilization | the published figure is the total training cost (about 2.8 million H800 GPU-hours) |

Both approaches can be explained with this chapter's framework:

- **Llama 3** is the classic "tensor parallelism fills the node, the pipeline cuts depth across nodes, data parallelism absorbs the rest". At over ten thousand cards, the global batch (16M tokens) no longer allows data parallelism to grow, so the pipeline has to reach 16; when the context extends to 128K, CP=16 replaces a factor of 16 of data parallelism and the global batch is still 16M tokens.
- **DeepSeek-V3** activates only 37B parameters per token, so the attention part's computation and activations are both modest. It saves memory by recomputing the RMSNorms and MLA's up-projection and by keeping the parameters' exponential moving average on the CPU, and simply does without tensor parallelism, avoiding the per-layer all-reduce entirely. The cost moves to expert parallelism's cross-node all-to-all, which DualPipe overlaps with the computation, with each token capped at 4 nodes.

A mixture-of-experts model's 5D is usually folded like this: the attention part is laid out as TP x CP x DP x PP; on entering a mixture-of-experts layer, the same cards are regrouped into expert TP x EP x expert DP x PP, with expert parallelism generally taken from the DP x CP dimensions (what Megatron calls MoE parallel folding). Expert parallelism then needs no extra cards and the experts' parameters are partitioned along it.

!!! interview "How to explain it"
    "Here are N cards and this model, how would you configure the parallelism" pulls everything together, so go in steps: make the model states fit (ZeRO, tensor parallelism, pipeline parallelism or FSDP), make the activations fit (sequence parallelism, context parallelism, recomputing some layers), give the rest to data parallelism, then tune the pipeline's micro-batch count and layer division. On the layout, tensor parallelism innermost (NVLink within a node), context parallelism next, data and pipeline parallelism outside, with expert parallelism reusing the data-parallel cards. Then the trade-offs: tensor parallelism's communication share grows with its degree, the pipeline costs a bubble and an imbalance at the first and last stages (which often take fewer layers), and the global batch limits data parallelism times the micro-batch. Finish by stressing that the estimate only narrows the field and the top few configurations have to be measured with a profiler.

## Exercises {#练习}

1. Use this chapter's formulas to explain why tensor parallelism's share of a step's time is roughly proportional to $(t-1)$ and inversely proportional to the hidden dimension $h$.

??? success "Answer"
    Per layer per micro-batch, one card's computation takes about $s \cdot f_{\text{layer}} / (t \cdot P \cdot e)$, where $f_{\text{layer}} \approx 12h^2 \cdot c$ (the linear layers, with $c$ a constant depending on the MLP's width) plus the attention term; tensor parallelism's communication takes $4 \times 2\frac{t-1}{t} \cdot 2sh / B$. Dividing one by the other:

    $$\frac{T_{\text{TP}}}{T_{\text{compute}}} = \frac{16 (t-1)\, h\, P e}{B\, f_{\text{layer}}} \propto \frac{t-1}{h}$$

    Substituting 70B ($h = 8192$, $f_{\text{layer}} \approx 5.5 \times 10^9$ at an 8K sequence), $B = 450$ GB/s and $Pe = 593$ TFLOPS gives a coefficient of about 0.031: at $t=4$ the communication is about 9% of the computation and at $t=8$ about 22%, matching the script's 8% and 18% (whose denominator is the whole step's time, so they are slightly smaller).

2. Scale the 70B, 8K case to 1024 cards with the global batch still 4M tokens per step (512 sequences). How does the best configuration change? And with the global batch raised to 16M tokens? Predict first, then check with the script.

??? success "Answer"
    ```python title="scale_out.py"
    from parallel_plan import llama70b, search

    search("70B，1024 卡，8K 序列，每步 4M token", llama70b, 8192, 512, 1024)
    search("70B，1024 卡，8K 序列，每步 16M token", llama70b, 8192, 2048, 1024)
    ```

    ```text title="output"
    == 70B，1024 卡，8K 序列，每步 4M token：28 种配置放得下
    TP=8 CP=1 PP=1 DP=128：MFU 47.6%，55 GiB/卡｜计算 79% TP 17% DP 4%
    TP=4 CP=2 PP=2 DP=64：MFU 46.5%，55 GiB/卡｜计算 77% 不均衡 1% TP 7% 气泡 11% DP 3%
    TP=4 CP=2 PP=4 DP=32：MFU 43.7%，38 GiB/卡｜计算 73% 不均衡 3% TP 7% 气泡 16% DP 2%
    == 70B，1024 卡，8K 序列，每步 16M token：28 种配置放得下
    TP=4 CP=2 PP=2 DP=64：MFU 52.0%，55 GiB/卡｜计算 87% 不均衡 1% TP 8% 气泡 3% DP 1%
    TP=4 CP=2 PP=4 DP=32：MFU 50.2%，38 GiB/卡｜计算 84% 不均衡 4% TP 8% 气泡 4%
    TP=8 CP=1 PP=1 DP=128：MFU 48.9%，55 GiB/卡｜计算 82% TP 18% DP 1%
    ```

    With 16 times the cards and the same batch, each pipeline gets fewer micro-batches: at 64 cards DP=4 gives 128 per pipeline, and at 1024 cards DP=64 leaves only 8, so PP=2's bubble grows from 1% to 11% and TP=8 x DP=128 without a pipeline comes first. That is what "the global batch limits the scaling" means: the more cards there are, the less work each one gets and the larger the share of bubble and unoverlapped communication.
    Raising the global batch to 16M tokens gives each pipeline 32 micro-batches again, the bubble falls back to 3%, and TP=4 x CP=2 x PP=2 x DP=64 is first once more. In real training the global batch is set by convergence (Llama 3 raised it from 4M to 16M tokens over the course of training) and the parallel configuration follows it.

3. What would have to change for the script to support a mixture-of-experts model (256 experts with 8 chosen per token, say)?

??? success "Answer"
    - **Parameters and memory**: the parameters split in two, with the attention and shared parts partitioned by TP x PP and the expert part by EP x expert TP x PP; the optimizer states are partitioned over their respective data-parallel groups (DP x CP for attention, the expert DP for the experts).
    - **Computation**: each token passes through only $k$ experts, so the compute is counted from the **active** parameter count, not the total.
    - **New expert-parallel communication**: two all-to-alls per mixture-of-experts layer forward (dispatch and combine) and two more backward, each about $k \cdot sbh$ bytes, going over the network cards between nodes; some of it can be assumed to overlap, along DualPipe's lines.
    - **Load imbalance**: the card holding the busiest expert sets the whole layer's time, so multiply by an imbalance factor (1.1 to 1.3, say).
    - **Constraints**: the expert-parallel degree has to divide the expert count and be taken from the DP x CP cards; the experts' tensor-parallel degree is usually 1.

## Summary {#小结}

- [x] The parallel dimensions are laid out from the inside out: tensor parallelism innermost (NVLink within a node), context parallelism next, data and pipeline parallelism outside; expert parallelism reuses the data-parallel cards.
- [x] The order for choosing: make the model states fit (ZeRO, tensor parallelism, pipeline parallelism or FSDP), make the activations fit (sequence parallelism, context parallelism, recomputing some layers), give the rest to data parallelism, then tune the pipeline.
- [x] Tensor parallelism's communication share grows as $(t-1)/h$; the pipeline costs a bubble and an imbalance at the first and last stages; context parallelism trades a little communication for a great deal of recomputation on a long sequence.
- [x] The global batch limits data parallelism times the micro-batch; the more cards, the larger the batch needed, or the more tensor and pipeline parallelism needed to absorb them.
- [x] The estimate only narrows the field, and the top few configurations have to be measured with a profiler.
