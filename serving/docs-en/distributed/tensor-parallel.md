# Tensor parallelism from scratch

<p class="lead">When a model does not fit on one GPU, or one GPU is too slow, the first choice is usually tensor parallelism (TP): split each layer's matrices across several GPUs that compute at the same time. This chapter implements a tensor-parallel Qwen3-0.6B from scratch, following Megatron-LM: column splits, row splits, attention split by heads, and an embedding and output layer split by vocabulary. We launch two processes on CPU with torchrun, check that they match a single process's output, then estimate the cost of communication in real deployments.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How do you split the MLP's three matrices so the whole MLP needs only one all-reduce?
    2. Why is attention split by heads? How is the KV Cache split? What if TP exceeds the number of KV heads?
    3. How many communications does one Transformer layer need under TP? What is the volume proportional to?
    4. Why is TP usually used only within one machine?

??? success "Answers (try first, then expand to compare)"
    1. Split gate and up along the output dimension (columns) and down along the input dimension (rows): the column splits each compute part of the intermediate features with no communication, the activation is elementwise, and down's row split takes exactly that GPU's part of the features, giving a partial sum, followed by a single all-reduce.
    2. Each head's attention is computed independently, so after splitting by heads each GPU computes its own heads with no communication in between; the KV Cache is split along with the KV heads, and each GPU stores only its own. When TP exceeds the number of KV heads, KV heads are replicated across several GPUs (each GPU gets at least one).
    3. Two all-reduces (after attention's output projection and after the MLP). Each one's volume is proportional to this step's token count × hidden; in decode the messages are small and latency dominates, in prefill they are large and bandwidth dominates.
    4. The two all-reduces per layer are on the critical path, and only NVLink within a machine (about 450 GB/s per GPU in each direction) is fast enough; NICs across machines give only about 50 GB/s with higher latency, so across machines one usually switches to pipeline, expert or data parallelism.

<!-- comic ../assets/comics/tensor-parallel.webp is in Chinese; put it back once the English version exists -->

## Two ways to split {#两种切法}

For a linear layer $Y = XW$ where $W$ has shape `[input dim, output dim]`, there are two ways to split:

- **Column split** (column parallel): split $W$ along the output dimension into $[W_1, W_2]$; GPU i computes $Y_i = XW_i$. Each GPU gets a slice of the output with **no communication**, but every GPU needs the full input $X$.
- **Row split** (row parallel): split $W$ along the input dimension into $\begin{bmatrix}W_1 \\ W_2\end{bmatrix}$; GPU i computes $X_iW_i$ from slice i of the input. Each GPU gets a **partial sum**, and an all-reduce (adding up every GPU's result) is needed to get the full $Y$.

Megatron's key observation: **a column split followed by a row split needs no communication in between**. The column split's output is exactly the "slice of the input" the row split needs. Hence:

- **MLP**: `gate_proj` and `up_proj` are column-split, `silu(gate) * up` is elementwise and done locally on each GPU, `down_proj` is row-split, and a single all-reduce comes at the end;
- **Attention**: `q/k/v_proj` are column-split, and split **by heads** (each GPU gets several complete heads); each GPU computes attention for its heads independently, `o_proj` is row-split, and a single all-reduce comes at the end.

![Figure: tensor-parallel MLP (2 GPUs)](../assets/figures/tp-mlp.svg){.aig-svg}

Two all-reduces per layer, everything else computed locally. The KV Cache is split along with it: each GPU stores only the KV heads it is responsible for.

<!-- i18n:diagram f3b884abca -->
```text
                              x (every GPU holds a full copy)
          ┌───────────────────┴───────────────────┐
  GPU 0: q/k/v heads 0–6                  GPU 1: q/k/v heads 7–13              ← column split, no communication
          │ attention (local KV Cache: KV head 0) │ attention (local KV Cache: KV head 1)
  o_proj input slice 0 → partial sum      o_proj input slice 1 → partial sum   ← row split
          └────────── all-reduce (sum) ───────────┘
                              x + attn (every GPU holds a full copy again)
```

The embedding and output layers are split by **vocabulary**: each GPU stores a slice of the vocabulary; in the embedding, tokens outside this GPU's slice look up 0, and after an all-reduce every GPU has the full embedding; in the output layer each GPU computes a slice of the logits, and an all-gather joins them.

## Implementation {#实现}

```python title="tp.py"
"""tp.py —— 从零实现张量并行（Megatron 式），在 CPU 上用 gloo 后端跑多进程。

用法：torchrun --standalone --nproc-per-node 2 tp.py
每个进程只持有 1/tp 的权重和 KV Cache，每层两次 all-reduce；rank 0 与单进程的完整模型比较输出。
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mini_llm import KVCache, RMSNorm, Transformer, apply_rope, rope_cos_sin

NUM_ALL_REDUCE = 0


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    global NUM_ALL_REDUCE
    NUM_ALL_REDUCE += 1
    dist.all_reduce(x)                     # sums by default: adds up every rank's partial result
    return x


class ColumnParallelLinear(nn.Module):
    """按输出维切分：每个 rank 算输出的一段，不需要通信。"""

    def __init__(self, full: nn.Linear, rank: int, world: int, rows: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(full.weight[rows].clone())
        self.bias = nn.Parameter(full.bias[rows].clone()) if full.bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """按输入维切分：每个 rank 用自己那段输入算出一个"部分和"，all-reduce 后才是完整结果。"""

    def __init__(self, full: nn.Linear, rank: int, world: int, cols: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(full.weight[:, cols].clone())
        assert full.bias is None

    def forward(self, x):
        return all_reduce(F.linear(x, self.weight))


class VocabParallelEmbedding(nn.Module):
    """词表切分：每个 rank 只存一段词表；不在本段的 token 查出 0，all-reduce 后得到完整的嵌入。"""

    def __init__(self, full: nn.Embedding, rank: int, world: int):
        super().__init__()
        per = (full.num_embeddings + world - 1) // world
        self.start, self.end = rank * per, min((rank + 1) * per, full.num_embeddings)
        self.weight = nn.Parameter(full.weight[self.start:self.end].clone())

    def forward(self, ids):
        mask = (ids < self.start) | (ids >= self.end)
        out = F.embedding((ids - self.start).masked_fill(mask, 0), self.weight)
        return all_reduce(out.masked_fill(mask[..., None], 0.0))


class TPAttention(nn.Module):
    """按头切分：每个 rank 负责 nh/tp 个 query 头和 nkv/tp 个 KV 头，KV Cache 也只存自己的头。"""

    def __init__(self, attn, cfg, rank: int, world: int):
        super().__init__()
        hd = cfg.hd
        self.nh, self.nkv, self.hd = cfg.num_attention_heads // world, cfg.num_key_value_heads // world, hd
        q_rows = torch.arange(rank * self.nh * hd, (rank + 1) * self.nh * hd)
        kv_rows = torch.arange(rank * self.nkv * hd, (rank + 1) * self.nkv * hd)
        self.q_proj = ColumnParallelLinear(attn.q_proj, rank, world, q_rows)
        self.k_proj = ColumnParallelLinear(attn.k_proj, rank, world, kv_rows)
        self.v_proj = ColumnParallelLinear(attn.v_proj, rank, world, kv_rows)
        self.o_proj = RowParallelLinear(attn.o_proj, rank, world, q_rows)
        self.q_norm, self.k_norm = attn.q_norm, attn.k_norm     # QK-Norm is per head with a weight of only head_dim: every rank keeps a full copy
        self.layer = attn.layer

    def forward(self, x, cos, sin, cache):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = cache.update(self.layer, k, v)
        S = k.shape[2]
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.nh * self.hd))


class TPMLP(nn.Module):
    """gate/up 按列切分，down 按行切分：中间的 SiLU × up 在各 rank 本地完成，整个 MLP 只需一次 all-reduce。"""

    def __init__(self, mlp, rank: int, world: int):
        super().__init__()
        n = mlp.gate_proj.out_features // world
        rows = torch.arange(rank * n, (rank + 1) * n)
        self.gate_proj = ColumnParallelLinear(mlp.gate_proj, rank, world, rows)
        self.up_proj = ColumnParallelLinear(mlp.up_proj, rank, world, rows)
        self.down_proj = RowParallelLinear(mlp.down_proj, rank, world, rows)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPTransformer(nn.Module):
    def __init__(self, full: Transformer, rank: int, world: int):
        super().__init__()
        cfg = self.cfg = full.cfg
        assert cfg.num_attention_heads % world == 0 and cfg.num_key_value_heads % world == 0
        self.world = world
        self.embed_tokens = VocabParallelEmbedding(full.embed_tokens, rank, world)
        self.layers = nn.ModuleList()
        for layer in full.layers:
            new = nn.Module()
            new.input_layernorm, new.post_attention_layernorm = layer.input_layernorm, layer.post_attention_layernorm
            new.self_attn = TPAttention(layer.self_attn, cfg, rank, world)
            new.mlp = TPMLP(layer.mlp, rank, world)
            self.layers.append(new)
        self.norm = full.norm
        # the output layer is split by vocabulary (the embedding's slice when weights are tied); each rank computes a slice of logits, then all-gather
        self.lm_head_weight = self.embed_tokens.weight if cfg.tie_word_embeddings else \
            nn.Parameter(full.lm_head.weight[self.embed_tokens.start:self.embed_tokens.end].clone())

    def forward(self, ids, cache):
        start = cache.length
        cos, sin = rope_cos_sin(torch.arange(start, start + ids.shape[1]), self.cfg.hd, self.cfg.rope_theta)
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = x + layer.self_attn(layer.input_layernorm(x), cos, sin, cache)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        local = F.linear(self.norm(x[:, -1:]), self.lm_head_weight)       # only the last position
        parts = [torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(parts, local)
        return torch.cat(parts, dim=-1)[..., : self.cfg.vocab_size]


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(max(1, int(os.environ.get("THREADS", "8"))))
    full = Transformer.from_pretrained(os.environ.get("MODEL", "models/Qwen3-0.6B"))
    ids = torch.tensor([[151644, 872, 198, 105043, 100165, 11319, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]])  # chat template for "你是谁？" ("Who are you?"), thinking off
    model = TPTransformer(full, rank, world)
    n_local = sum(p.numel() for p in model.parameters())

    global NUM_ALL_REDUCE
    with torch.no_grad():
        cache, tokens = KVCache(full.cfg.num_hidden_layers), []
        logits = model(ids, cache)
        prefill_all_reduce = NUM_ALL_REDUCE
        for _ in range(20):
            nxt = logits[0, -1].argmax().item()
            tokens.append(nxt)
            logits = model(torch.tensor([[nxt]]), cache)
    kv_heads_local = cache.k[0].shape[1]

    if rank == 0:
        with torch.no_grad():
            ref_cache, ref_tokens = KVCache(full.cfg.num_hidden_layers), []
            ref_logits = full(ids, ref_cache)[:, -1:]
            for _ in range(20):
                nxt = ref_logits[0, -1].argmax().item()
                ref_tokens.append(nxt)
                ref_logits = full(torch.tensor([[nxt]]), ref_cache)[:, -1:]
        n_full = sum(p.numel() for p in full.parameters())
        print(f"TP={world}：每个 rank 持有 {n_local / 1e6:.0f}M 参数（完整模型 {n_full / 1e6:.0f}M），"
              f"每层 KV Cache 存 {kv_heads_local} 个 KV 头（共 {full.cfg.num_key_value_heads} 个）")
        print(f"prefill 一次前向的 all-reduce 次数：{prefill_all_reduce}（{full.cfg.num_hidden_layers} 层 × 2 + 嵌入 1）")
        print(f"最后一步 logits 与单进程的最大误差：{(logits[0, -1] - ref_logits[0, -1]).abs().max().item():.1e}")
        print(f"贪心生成 20 个 token 与单进程一致：{tokens == ref_tokens}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

The weights are sliced directly from the full model: `ColumnParallelLinear` takes some rows of the weight matrix (PyTorch's `nn.Linear` weight has shape `[output dim, input dim]`, so splitting along the output dimension means taking rows), and `RowParallelLinear` takes some columns. RMSNorm's weights are tiny, so every GPU keeps a full copy.

Launch two processes with torchrun (gloo as the communication backend on CPU; on GPU, just switch to NCCL):

```python
import os
import subprocess
import sys

env = {**os.environ, "OMP_NUM_THREADS": "8", "THREADS": "8"}
result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2",
                         "build/code/tp.py"], capture_output=True, text=True, env=env, timeout=1200)
print(result.stdout.strip())
assert "一致：True" in result.stdout
```

```text
TP=2：每个 rank 持有 298M 参数（完整模型 596M），每层 KV Cache 存 4 个 KV 头（共 8 个）
prefill 一次前向的 all-reduce 次数：57（28 层 × 2 + 嵌入 1）
最后一步 logits 与单进程的最大误差：1.4e-05
贪心生成 20 个 token 与单进程一致：True
```

Each process holds only half of the parameters and half of the KV Cache, with two all-reduces per layer, and the result matches the single process (the error comes from the different order of summation).

## KV heads and the limit on TP {#kv-头数与-tp-的上限}

Splitting by heads requires the head count to be divisible by TP. Qwen3-0.6B has 16 query heads and 8 KV heads, so TP can be 2, 4 or 8; beyond that there are not enough KV heads to go around. More generally:

- **The query head count must be divisible by TP**: Qwen2.5-7B has 28 heads, so it cannot do TP=8 (inference engines simply raise an error);
- **When TP exceeds the KV head count, KV heads are replicated**: for example, with a model with 8 KV heads at TP=16, every two GPUs share one KV head, each storing its own copy. The KV Cache's total memory doubles, and that extra memory is wasted;
- **MLA's latent vector cannot be split by heads**: every GPU must store the full latent KV, and the larger TP, the more waste, which is why DeepSeek-style models lean towards [DP Attention](expert-parallel.md).

## The cost of communication {#通信的代价}

TP does two all-reduces per layer, each moving `tokens × hidden × 2 bytes` (BF16). A ring all-reduce takes about $\alpha + 2\frac{n-1}{n} \cdot \frac{\text{size}}{\text{bus bandwidth}}$, where $\alpha$ is the launch latency. Estimate the communication time for LLaMA-3-70B (80 layers, hidden 8192) with TP on 8 H100s (assuming an all-reduce bus bandwidth of about 360 GB/s over NVLink and a fixed latency of about 10 μs per call):

```python
hidden, layers, tp = 8192, 80, 8
busbw, alpha = 360e9, 10e-6

def comm_ms(tokens):
    size = tokens * hidden * 2
    return 2 * layers * (alpha + 2 * (tp - 1) / tp * size / busbw) * 1e3

decode_compute = 70.55e9 * 2 / tp / 3.35e12 * 1e3                    # each GPU reads 1/8 of the weights
prefill_compute = 2 * 70.55e9 * 4096 / (tp * 989e12 * 0.5) * 1e3     # MFU 50%
print(f"decode（batch 64）：通信约 {comm_ms(64):.1f} ms，读权重的下限 {decode_compute:.1f} ms")
print(f"prefill（4096 token）：通信约 {comm_ms(4096):.0f} ms，计算约 {prefill_compute:.0f} ms")
```

```text title="output"
decode（batch 64）：通信约 2.4 ms，读权重的下限 5.3 ms
prefill（4096 token）：通信约 54 ms，计算约 146 ms
```

A few conclusions:

- **In decode, communication is latency-bound**: each call moves only 1 MB, and the time goes mostly to the fixed overhead of 160 calls, close to half the compute time. That is why inference engines implement all-reduces optimized for small messages (vLLM's custom all-reduce, SGLang's custom all-reduce and FlashInfer's fused all-reduce), using one-shot/two-shot algorithms that read and write directly over NVLink P2P to push latency down to a few microseconds, and fusing the all-reduce with the following RMSNorm.
- **In prefill, communication is bandwidth-bound**: the volume is large, and communication is over a third of the compute. It can be hidden by overlapping compute and communication (split the tokens in two and alternate computing and communicating), or by lowering TP and using other kinds of parallelism.
- **TP needs high-bandwidth, low-latency interconnect**: across machines (InfiniBand at 400 Gb/s per GPU, about 50 GB/s) bandwidth is nearly an order of magnitude below NVLink, which makes TP almost unusable. So TP is usually confined to one machine, with pipeline, data or expert parallelism across machines.

!!! source "Source code"
    - **vLLM**: the parallel linear layers are in `vllm/model_executor/layers/linear.py`: `ColumnParallelLinear`, `RowParallelLinear`, and the merged versions `MergedColumnParallelLinear` (gate/up) and `QKVParallelLinear` (which handles KV head replication: `num_kv_head_replicas`); the vocabulary split is in `vocab_parallel_embedding.py` (`VocabParallelEmbedding`, `ParallelLMHead`). Communication groups are in `vllm/distributed/parallel_state.py` (`get_tp_group()`), and `tensor_model_parallel_all_reduce` picks custom all-reduce, symmetric memory (`symm_mem.py`), FlashInfer or NCCL according to message size and hardware (`device_communicators/cuda_communicator.py`).
    - **SGLang**: the parallel layers of the same names in `srt/layers/linear.py`, communication groups and custom all-reduce in `srt/distributed/`, and `srt/layers/communicator.py`, which orchestrates communication between layers in the various parallel modes (for example fusing all-reduce with RMSNorm, and gather/scatter under DP Attention).

!!! interview "In an interview"
    Deriving TP by hand is a frequent question. Key points: **for the MLP, columns then rows, no communication in between, one all-reduce**; **for attention, split by heads with the KV Cache split along, one all-reduce**; **two all-reduces per layer, volume = tokens × hidden**. Then add engineering details: the divisibility constraint on heads, KV head replication, the latency problem of small messages in decode and custom all-reduce, and TP being confined to one machine. If you can write `RowParallelLinear`'s forward (local GEMM + all-reduce), you have basically passed.

## Exercises {#练习}

**1. Sequence parallelism.** Under TP, RMSNorm and the residual add are recomputed on every GPU over the **full** `[tokens, hidden]`. How do you remove this redundancy?

??? success "Approach"
    Split the all-reduce into reduce-scatter + all-gather: after o_proj/down_proj do a reduce-scatter so each GPU gets the full result for 1/tp of the tokens, do the residual add and RMSNorm on those tokens, then all-gather back to all tokens before the next column-split layer. The total communication is unchanged (reduce-scatter + all-gather = all-reduce), but the compute and activation memory of per-token operations such as RMSNorm drop to 1/tp. This is Megatron's sequence parallelism, useful in inference for prefilling long sequences.

**2. TP or DP?** An 8B model on 8 H100s. What are the pros and cons of deploying one instance with TP=8 versus 8 independent instances with DP=8?

??? success "Answer"
    - TP=8: low single-request latency (each step reads only 1/8 of the weights), and one instance can hold a larger context and batch; but two all-reduces per layer add significant communication overhead, and total throughput is usually below DP.
    - DP=8: no communication, the highest throughput, simple to implement, and a broken GPU affects only 1/8; but single-request latency is one GPU's latency, and each instance's KV Cache gets only one GPU's memory.

    An 8B model fits on one GPU, so DP is the better deal when aiming for throughput; with strict TPOT requirements or very long contexts, consider a compromise such as TP=2 or TP=4 (for example DP=4 × TP=2).

!!! tip "Tensor parallelism in training"
    Training also has to handle the backward pass: the input gradient of a column split needs an all-reduce, and the row split is the reverse; on top of that, sequence parallelism splits the all-reduce into reduce-scatter + all-gather to partition the activations at LayerNorm. The [tensor and sequence parallelism](train://model/tensor-sequence/) chapter of the distributed training handbook implements a version with the backward pass from scratch and matches a single process's gradients term by term.

## Summary {#小结}

- [x] A column split needs no communication, a row split needs an all-reduce; with columns then rows, an MLP or an attention block needs only one all-reduce.
- [x] Attention is split by heads with the KV Cache split along; the head count must be divisible by TP, and when TP exceeds the KV head count, KV heads are replicated.
- [x] Two all-reduces per layer, with volume proportional to tokens × hidden; decode is latency-bound, prefill bandwidth-bound.
- [x] TP depends on intra-machine interconnect such as NVLink; across machines, other kinds of parallelism are usually used.
