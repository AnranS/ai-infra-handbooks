# Pipeline parallelism and context parallelism

<p class="lead">Tensor parallelism splits each layer's matrices, and expert parallelism splits the experts. There are two more dimensions to split along: by **layer** (pipeline parallelism, PP) and by **sequence** (context parallelism, CP). PP needs little communication and suits scaling model size across machines; CP targets the prefill and KV Cache of very long contexts. This chapter implements a two-stage pipeline with two processes, and uses ring attention to show how context parallelism computes causal attention across several GPUs.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Can PP lower a single request's latency? What does it improve?
    2. Why does PP in inference need "several batches in the pipeline at once"?
    3. In ring attention, how are each GPU's partial attention results merged? Why is log-sum-exp needed?
    4. Under causal attention, what goes wrong when the sequence is split into contiguous ranges? How do you fix it?

??? success "Answers (try first, then expand to compare)"
    1. No: a request still passes through every stage in turn, plus the transfers between stages. What it improves is throughput (several GPUs process different batches at once), and it makes models that do not fit on one machine deployable.
    2. While each stage processes its part, the other stages sit idle unless they have other batches; only with several batches in the pipeline at once (while one batch is at stage 2, the next is already at stage 1) can all stages be busy at the same time.
    3. Each GPU computes its query block's partial output $o_i$ against some KV block, along with the log-sum-exp $\mathrm{lse}_i$, and they merge as $\mathrm{lse} = \log \sum e^{\mathrm{lse}_i}$, $o = \sum o_i e^{\mathrm{lse}_i - \mathrm{lse}}$. Since each part's softmax denominator differs, the parts must be reweighted with lse to be equivalent to one softmax over the full sequence.
    4. Severe load imbalance: later segments must look at all earlier KV and have the most compute, earlier ones have almost none, and the slowest GPU sets the speed. Use a zigzag split: cut into 2P chunks and give each GPU one early and one late chunk, whose workloads exactly complement each other.

## Pipeline parallelism {#流水线并行}

PP divides the model's layers into several segments (stages), with each GPU (or group of GPUs) responsible for one. When one stage finishes, it sends the hidden state to the next. Each stage stores only the weights and KV Cache of its own layers.

```python title="pp.py"
"""pp.py —— 流水线并行的最小实现：两个进程各持有一半的层，点对点传递隐藏状态。

用法：torchrun --standalone --nproc-per-node 2 pp.py
stage 0：嵌入 + 前一半的层；stage 1：后一半的层 + 最终归一化 + 输出层。
每个 stage 只存自己那些层的权重和 KV Cache；每生成一个 token，隐藏状态从 stage 0 发到 stage 1，
采样出的 token 再发回 stage 0。
"""

import os

import torch
import torch.distributed as dist

from mini_llm import KVCache, Transformer, rope_cos_sin


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    full = Transformer.from_pretrained(os.environ.get("MODEL", "models/Qwen3-0.6B"))
    cfg = full.cfg
    L = cfg.num_hidden_layers
    my_layers = range(0, L // 2) if rank == 0 else range(L // 2, L)
    cache = KVCache(L)                                   # only the slots of this stage's own layers get used
    prompt = [151644, 872, 198, 105043, 100165, 11319, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
    ids, generated, sent_bytes = torch.tensor([prompt]), [], 0

    with torch.no_grad():
        for step in range(20):
            T = ids.shape[1]
            first = cache.k[my_layers[0]]                # the KV Cache length of this stage's first layer is the number of tokens processed
            start = 0 if first is None else first.shape[2]
            cos, sin = rope_cos_sin(torch.arange(start, start + T), cfg.hd, cfg.rope_theta)
            if rank == 0:
                x = full.embed_tokens(ids)
            else:
                x = torch.empty(1, T, cfg.hidden_size)
                dist.recv(x, src=0)                      # receive the hidden state from the previous stage
            for i in my_layers:
                x = full.layers[i](x, cos, sin, cache)
            if rank == 0:
                dist.send(x, dst=1)                      # send to the next stage: [T, hidden]
                sent_bytes += x.numel() * 2              # communication volume of a real deployment, counted in BF16
                nxt = torch.empty(1, dtype=torch.long)
                dist.recv(nxt, src=1)                    # wait for the token sampled by the last stage
            else:
                nxt = full.lm_head(full.norm(x[:, -1])).argmax(-1)
                dist.send(nxt, dst=0)
            generated.append(nxt.item())
            ids = nxt.view(1, 1)

    if rank == 0:
        ref = []
        with torch.no_grad():
            ref_cache, x = KVCache(L), torch.tensor([prompt])
            for _ in range(20):
                nxt = full(x, ref_cache)[0, -1].argmax().item()
                ref.append(nxt)
                x = torch.tensor([[nxt]])
        stage_layers = [f"{r.start}～{r.stop - 1}" for r in (range(0, L // 2), range(L // 2, L))]
        print(f"PP=2：stage 0 负责第 {stage_layers[0]} 层，stage 1 负责第 {stage_layers[1]} 层")
        print(f"生成 20 个 token，stage 之间共传输 {sent_bytes / 1024:.0f} KB 隐藏状态")
        print(f"与单进程一致：{generated == ref}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

```python
import os
import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2",
                         "build/code/pp.py"], capture_output=True, text=True,
                        env={**os.environ, "OMP_NUM_THREADS": "8"}, timeout=1200)
print(result.stdout.strip())
assert "一致：True" in result.stdout
```

```text title="output"
PP=2：stage 0 负责第 0～13 层，stage 1 负责第 14～27 层
生成 20 个 token，stage 之间共传输 68 KB 隐藏状态
与单进程一致：True
```

Compare with tensor parallelism: TP does two all-reduces per layer, each over the whole `[tokens, hidden]`; PP has only one point-to-point send at each stage boundary, of the same size. Splitting a 28-layer model into 2 stages cuts communication from 56 all-reduces to 1 send. So **PP needs far less interconnect bandwidth and suits going across machines**.

### Pipeline bubbles {#流水线的气泡}

PP's problem: a batch must pass through the stages one by one, so only one stage works at any moment.

<!-- i18n:diagram b4bd9c9870 -->
```text
One batch in the pipeline (PP=4):         Four batches in the pipeline at once:
stage 0  ██░░░░░░██░░░░░░██               stage 0  ██████████████████
stage 1  ░░██░░░░░░██░░░░░░               stage 1  ░░████████████████
stage 2  ░░░░██░░░░░░██░░░░               stage 2  ░░░░██████████████
stage 3  ░░░░░░██░░░░░░██░░               stage 3  ░░░░░░████████████
  each GPU works only 1/4 of the time       once full, every GPU is busy
```

The bubble size depends on the schedule. Use the same widget as in the [distributed training handbook](train://model/pipeline/) to look at GPipe and 1F1B (the round-robin of several batches in inference decode works the same way):

<div class="aig-widget" data-widget="pipeline"></div>

Therefore:

- **PP does not lower a single request's latency**: a token still passes through every layer in turn, plus the transfers between stages;
- **PP improves throughput**, provided there are enough batches in the pipeline at once. Inference engines let the scheduler run several batches ahead: vLLM's `step_with_batch_queue` keeps a batch queue as long as the PP size (see [the vLLM walkthrough](../source/vllm.md#主线二enginecore-主循环)), and SGLang has a dedicated PP event loop (`scheduler_pp_mixin.py`);
- In decode, step N's input depends on step N−1's output, so the same request cannot be in two in-flight batches at once. Requests are therefore split into several groups that take turns entering the pipeline.

Typical use: when a model is too big for one machine, TP within a machine and PP across machines (for example, a 405B model on two 8-GPU machines with TP=8, PP=2).

## Context parallelism {#上下文并行}

When contexts reach hundreds of thousands or millions of tokens, two new problems appear: prefill attention compute grows with the square of the length, so one GPU takes a long time; and one request's KV Cache alone may exceed one GPU's memory. Context parallelism splits the **sequence**: each GPU handles the queries of one segment of tokens, and the K/V of that segment.

### Ring attention {#ring-attention}

Each GPU has only its own segment of K/V, but its queries need to see all earlier K/V. Ring attention arranges the GPUs in a ring: at each step a GPU passes the K/V block it holds to the next GPU while computing part of the attention with the block it just received; after one full round, every GPU's queries have seen all the K/V.

The key is **how to merge attention computed in blocks**. Each block yields "softmax over just these keys", and merging into "softmax over all keys" requires each row's log-sum-exp (LSE), the same trick as FlashAttention's online softmax:

$$
\text{LSE} = \log(e^{\text{LSE}_1} + e^{\text{LSE}_2}),\quad
O = O_1 e^{\text{LSE}_1 - \text{LSE}} + O_2 e^{\text{LSE}_2 - \text{LSE}}
$$

![Figure: two ways to do sequence parallelism: Ulysses swaps heads with all-to-all, Ring passes K and V blocks around a ring](../assets/figures/ring-attention.svg){.aig-svg}

Simulate ring attention on 4 GPUs in a single process and check that it equals full causal attention:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
heads, seq, dim, n_dev = 4, 64, 32, 4
q, k, v = (torch.randn(heads, seq, dim) for _ in range(3))
reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)

def partial_attention(q_idx, k_idx):
    """一张卡用自己的 query 和收到的一块 K/V 计算：返回部分输出和每行的 LSE。"""
    s = q[:, q_idx] @ k[:, k_idx].transpose(-1, -2) * dim ** -0.5
    s = s.masked_fill(q_idx[:, None] < k_idx[None, :], float("-inf"))   # the causal mask uses global positions
    return torch.softmax(s, -1).nan_to_num() @ v[:, k_idx], torch.logsumexp(s, -1)

def merge(o1, lse1, o2, lse2):
    lse = torch.logaddexp(lse1, lse2)
    return o1 * torch.exp(lse1 - lse).nan_to_num()[..., None] + o2 * torch.exp(lse2 - lse).nan_to_num()[..., None], lse

def ring_attention(partition):
    out, work = torch.empty_like(q), []
    for dev, q_idx in enumerate(partition):
        o = lse = None
        for step in range(n_dev):                                  # at step `step`, the K/V received comes from the GPU `step` places back
            k_idx = partition[(dev - step) % n_dev]
            if q_idx.max() < k_idx.min():                          # the whole block is in the future: skip
                continue
            po, pl = partial_attention(q_idx, k_idx)
            o, lse = (po, pl) if o is None else merge(o, lse, po, pl)
        out[:, q_idx] = o
        work.append(int((q_idx[:, None] >= torch.cat(partition)[None, :]).sum()))   # number of (q, k) pairs to compute
    return out, work

contiguous = [torch.arange(i * 16, (i + 1) * 16) for i in range(n_dev)]
zigzag = [torch.cat([torch.arange(i * 8, (i + 1) * 8), torch.arange((7 - i) * 8, (8 - i) * 8)]) for i in range(n_dev)]
for name, partition in [("连续切分", contiguous), ("之字形切分", zigzag)]:
    out, work = ring_attention(partition)
    print(f"{name}：与完整因果注意力的最大误差 {(out - reference).abs().max().item():.1e}，各卡的计算量 {work}")
```

```text
连续切分：与完整因果注意力的最大误差 6.0e-07，各卡的计算量 [136, 392, 648, 904]
之字形切分：与完整因果注意力的最大误差 4.2e-07，各卡的计算量 [520, 520, 520, 520]
```

Both splits give correct results, but **the compute differs a lot**. Under causal attention, later tokens look at more keys: with contiguous ranges, the last GPU does over 6 times the compute of the first, and the whole thing is held back by it. The fix is a **zigzag split**: cut the sequence into 2n segments and give GPU i segment i and the i-th segment from the end, one early and one late, so every GPU does exactly the same compute.

### Context parallelism in inference {#推理中的上下文并行}

- **CP for prefill**: as above, for very long prompts. Each GPU computes only 1/n of the queries while KV passes around the ring, and communication can overlap with compute (send the next block while computing the current one).
- **CP for decode**: in decode the query is a single token, and the bottleneck is reading the KV Cache. One request's KV Cache can be split by sequence across several GPUs; each GPU computes partial attention over its segment of KV, the results merge with LSE, and the time to read KV also drops to 1/n. vLLM calls this DCP (decode context parallel), and it particularly suits models like MLA whose KV cannot be split by heads: under TP every GPU would store a full copy of the KV anyway, and splitting by sequence instead removes the redundancy.

!!! source "Source code"
    - **vLLM**: PP is turned on with `--pipeline-parallel-size`, with `IntermediateTensors` passed between stages; on the scheduling side it is `EngineCore.step_with_batch_queue`. DCP is turned on with `--decode-context-parallel-size`, and KV blocks are interleaved across GPUs by `dcp_world_size` (`resolve_dcp_kv_block_size` and others in `kv_cache_utils.py`); the attention backend computes partial results on each GPU and merges them with LSE. Context parallelism for prefill (`--prefill-context-parallel-size`) is also maturing.
    - **SGLang**: `--pp-size` turns on PP, with the scheduling logic in `managers/scheduler_pp_mixin.py`; context-parallel code is in `srt/layers/cp/` (`zigzag.py` there is this chapter's zigzag split), `srt/layers/dcp/`, and the CP implementation for DeepSeek sparse attention (`communicator_dsa_cp.py`).

!!! interview "How to explain it"
    To explain "how do you choose among TP, PP, DP, EP and CP", go by **communication pattern**: TP does an all-reduce every layer, needs high intra-machine bandwidth, and lowers single-request latency; PP transfers point-to-point only at stage boundaries, suits going across machines and scaling model size, but does not lower latency and needs several batches to fill the pipeline; DP has no communication and scales throughput; EP does two all-to-alls per MoE layer, for large-scale MoE; CP splits by sequence, for very long contexts, and needs LSE merging. Real deployments are often combinations: TP or EP within a machine, PP or DP across machines, plus CP for long contexts.

## Exercises {#练习}

**1. PP latency.** A model is deployed with PP=4; each stage takes 5 ms per decode step, and a transfer between stages takes 0.2 ms. What is a single request's TPOT? With the pipeline full, how many steps complete per second (across all batches)?

??? success "Answer"
    Each step of a single request passes through 4 stages and 3 transfers, plus sending the sampled result back to the first stage: about 4 × 5 + 4 × 0.2 = 20.8 ms, a TPOT of about 21 ms, slightly slower than one GPU's 20 ms (assuming it fit). With the pipeline full, each stage completes a step every 5 ms, about 200 steps per second overall, 4 times a single batch. PP trades latency for fitting the model and for throughput.

**2. LSE merging.** Why can't you just average the two partial outputs? What happens to the merge formula if one block is fully masked for some row (all −∞)?

??? success "Answer"
    The two blocks have different softmax denominators; averaging assumes the two blocks' keys carry equal total weight in attention, which is wrong. The correct weight is each block's softmax denominator (that is, $e^{\text{LSE}}$) as a share of the total denominator. A fully masked block has LSE −∞, $e^{\text{LSE}_i - \text{LSE}} = 0$, and its contribution is naturally 0; in implementations, beware that −∞ − (−∞) gives NaN and needs separate handling (the code above uses `nan_to_num`; you can also skip such blocks outright).

!!! tip "Pipeline and context parallelism in training"
    Training pipelines must also schedule the backward pass (1F1B, interleaved, zero bubble), and context parallelism must handle the backward KV gradients and load balancing under the causal mask. See the [pipeline parallelism](train://model/pipeline/) and [context parallelism](train://model/context/) chapters of the distributed training handbook.

## Summary {#小结}

- [x] PP splits by layer, with only point-to-point communication between stages, which suits going across machines; it does not lower single-request latency and needs several batches in the pipeline at once to raise throughput.
- [x] CP splits by sequence, for very long contexts; ring attention passes K/V blocks around a ring and merges partial results with LSE.
- [x] Under causal attention, contiguous splits make compute badly imbalanced, and a zigzag split gives every GPU the same load.
- [x] Decode CP (DCP) spreads one request's KV across several GPUs by sequence, which particularly suits MLA models.
