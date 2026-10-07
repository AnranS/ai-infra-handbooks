# Variable-length batching and one mixed forward pass

<p class="lead">With a paged KV cache, the next step is to make one forward pass handle many requests at once. These requests have different lengths: some are doing prefill, some decode, some are computing only a segment of their prompt. Inference engines do not pad them into a rectangular tensor; instead they lay all the tokens to compute end to end in one dimension, and use a few "metadata" arrays to describe which request each token belongs to, its position, and where its K/V go. This chapter implements this layout and verifies that the result of one mixed forward pass is exactly the same as computing each request on its own.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why don't inference engines use a padded `[batch, max_len]` layout?
    2. What are `query_start_loc` (`cu_seqlens_q`), `seq_lens`, `positions`, `slot_mapping` and `logits_indices`?
    3. What is the only difference, within one forward pass, between prefill, a middle segment of chunked prefill, and decode?
    4. Why do all the layers except attention not need to know the structure of the batch at all?

??? success "Answers (try first, then expand to compare)"
    1. Request lengths vary enormously, and padding to the longest wastes a lot of compute and memory (in decode each request has only 1 token, while prefills range from dozens to tens of thousands); besides, the layers other than attention do not need to know the batch structure at all.
    2. `query_start_loc`: where each request's tokens for this step start and end in the flat array (a prefix sum); `seq_lens`: each request's total length including this step (the KV length); `positions`: each token's position; `slot_mapping`: which slot each new token's KV is written to; `logits_indices`: which positions need logits (the last position of each request that needs sampling).
    3. Only "the length already cached" and "the length of this step" differ: prefill is 0 and the whole prompt, a middle segment of chunked prefill is the previous chunks and this chunk, decode is the whole history and 1. They can mix in the same forward pass.
    4. The layers other than attention (projections, normalization, MLP, RoPE) compute each token independently, so they can process the tokens laid out in one dimension uniformly; only attention needs to know which tokens belong to the same request and which KV they can see, and all of that is in the attention metadata.

## Why not pad {#为什么不填充}

Padding a batch to `[B, max_len]` wastes compute in proportion to the length differences. A decode request has only 1 new token; put it alongside a 2000-token prefill and it gets padded to 2000 tokens:

```python
query_lens = [2000, 512, 1, 1, 1, 1, 1, 1]             # two prefills and six decodes
padded = len(query_lens) * max(query_lens)
print(f"填充布局要计算 {padded} 个 token，实际只需要 {sum(query_lens)} 个，浪费 {1 - sum(query_lens) / padded:.0%}")
```

```text title="输出"
填充布局要计算 16000 个 token，实际只需要 2518 个，浪费 84%
```

Change the token counts of the requests to see the difference between the two layouts:

<div class="aig-widget" data-widget="ragged-waste"></div>

And in a Transformer, apart from attention, **every layer computes each token independently** (see the LLM book's [stop by stop](llm://synthesis/token-journey/#逐站解读)): the embedding, QKV projections, RoPE, MLP and normalization need only an `[N, hidden]` matrix and do not care which request the tokens come from. Only attention needs to know the boundaries: each token may see only **its own request's** context. So once attention has its metadata, everything else is one big matrix multiplication.

## Layout and metadata {#布局与元数据}

Take three requests: request A does a full prefill (7 tokens), request B decodes (5 tokens already in its context, 1 new one to compute), and request C is the second segment of a chunked prefill (the first 10 tokens are done; this step computes tokens 10–21):

<!-- i18n:diagram e1f5d68291 -->
```text
input_ids        [A0 A1 A2 A3 A4 A5 A6 | B5 | C10 C11 ... C21]   N = 7 + 1 + 12 = 20
positions        [ 0  1  2  3  4  5  6 |  5 |  10  11 ...  21]   each token's position in its own sequence
query_start_loc  [0, 7, 8, 20]                                    request i's tokens are [qsl[i], qsl[i+1])
seq_lens         [7, 6, 22]                                       each request's context length after this step
slot_mapping     which slot each token's K/V is written to (computed from the block tables)
block_tables     each request's block table
logits_indices   [6, 7, 19]                                       only these positions need the output layer and sampling
```

Prefill, decode and chunked prefill are **no different at all** in this layout: each is just "a request has n new tokens this step, with k tokens before them already in the cache". Decode is n = 1, prefill is k = 0, and chunked prefill has both nonzero. vLLM's scheduler is designed around exactly this (next chapter).

The output layer needs computing only for each request's **last** token: a middle segment of chunked prefill needs no sampling, so it can skip the output layer, and prefill needs only the last position's logits (the LLM book [worked it out](llm://synthesis/token-journey/#prefill-时只需要最后一个位置的-logits): this saves considerable computation).

## Implementation {#实现}

`build_batch` assembles each request's new tokens for this step, the length already computed and the block table into the metadata above; `ModelRunner` reuses `mini_llm`'s weights directly and computes layer by layer:

```python title="runner.py"
"""runner.py —— 把多个请求的新 token 拼成一维，一次前向完成：prefill、分块 prefill 与 decode 可以混在同一个批次里。

权重直接复用 mini_llm.Transformer 的模块，只是把"每个请求一个 [T, d] 张量"换成"所有 token 拼成 [N, d]"。
"""

import math
from dataclasses import dataclass

import torch

from mini_llm import Transformer, apply_rope, rope_cos_sin
from paged import PagedKVCache, paged_attention, slot_mapping_for


@dataclass
class BatchInput:
    input_ids: torch.Tensor        # [N] the tokens all requests compute this step, end to end
    positions: torch.Tensor        # [N] each token's position in its own sequence
    slot_mapping: torch.Tensor     # [N] which slot each token's K/V goes to
    query_start_loc: list[int]     # [B+1] request i's tokens are input_ids[qsl[i]:qsl[i+1]] (cu_seqlens_q)
    seq_lens: list[int]            # [B] each request's context length after this step
    block_tables: list[list[int]]  # [B] each request's block table
    logits_indices: torch.Tensor   # [M] indices in N of the tokens to sample (usually each request's last token)


def build_batch(items, block_size: int) -> BatchInput:
    """items: [(token_ids, num_computed, block_table, need_logits)]，token_ids 是本步要算的新 token。"""
    input_ids, positions, slots, qsl, seq_lens, tables, logits_idx = [], [], [], [0], [], [], []
    for token_ids, num_computed, table, need_logits in items:
        pos = list(range(num_computed, num_computed + len(token_ids)))
        input_ids += token_ids
        positions += pos
        slots += slot_mapping_for(table, pos, block_size)
        qsl.append(qsl[-1] + len(token_ids))
        seq_lens.append(num_computed + len(token_ids))
        tables.append(table)
        if need_logits:
            logits_idx.append(qsl[-1] - 1)
    return BatchInput(torch.tensor(input_ids), torch.tensor(positions), torch.tensor(slots), qsl, seq_lens,
                      tables, torch.tensor(logits_idx, dtype=torch.long))


class ModelRunner:
    def __init__(self, model: Transformer, kv: PagedKVCache):
        self.model, self.kv = model, kv
        cfg = model.cfg
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hd
        self.scale = 1 / math.sqrt(cfg.hd)

    @torch.no_grad()
    def forward(self, b: BatchInput) -> torch.Tensor:
        """返回 [M, vocab]：只为 logits_indices 指向的 token 计算输出层。"""
        m, N = self.model, b.input_ids.shape[0]
        x = m.embed_tokens(b.input_ids)                                        # [N, d]
        cos, sin = rope_cos_sin(b.positions, self.hd, m.cfg.rope_theta)       # [N, hd]
        cos, sin = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)   # broadcast over all heads
        for i, layer in enumerate(m.layers):
            attn, h = layer.self_attn, layer.input_layernorm(x)
            q = apply_rope(attn.q_norm(attn.q_proj(h).view(N, self.nh, self.hd)), cos, sin)     # QK-Norm before RoPE
            k = apply_rope(attn.k_norm(attn.k_proj(h).view(N, self.nkv, self.hd)), cos, sin)
            v = attn.v_proj(h).view(N, self.nkv, self.hd)
            self.kv.write(i, b.slot_mapping, k, v)                            # write first, then read back the whole context
            o = paged_attention(q, self.kv, i, b.block_tables, b.seq_lens, b.query_start_loc, self.scale)
            x = x + attn.o_proj(o.reshape(N, self.nh * self.hd))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.lm_head(m.norm(x[b.logits_indices]))
```

Compared with `mini_llm.Transformer.forward`, there are only two differences: the input changes from `[B, T]` to `[N]`, and attention changes from "concatenate a contiguous KV cache" to "write into the paged cache, then read back by block table".

## Check: one mixed forward pass {#验证一次混合前向}

Build the three requests above: in the first step A and B do a full prefill and C computes only its first 10 tokens; in the second step, A's decode, B's decode and the rest of C go into **the same forward pass**. Every result is compared with the logits `mini_llm` computes on the full sequence alone:

```python
import math
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer
from paged import BlockPool, PagedKVCache
from runner import ModelRunner, build_batch

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
cfg, block_size = model.cfg, 16
pool = BlockPool(64)
kv = PagedKVCache(cfg.num_hidden_layers, 64, block_size, cfg.num_key_value_heads, cfg.hd)
runner = ModelRunner(model, kv)

a, b, c = (tok(t).input_ids for t in ["请用一句话介绍 KV Cache。", "The capital of France is",
                                      "写一首关于大海的五言绝句，并解释其中的意象。然后再用英文翻译一遍。"])
tables = [pool.allocate(math.ceil(len(ids) / block_size) + 1) for ids in (a, b, c)]

def reference(ids):                                     # mini_llm computes the full sequence; take the last position
    with torch.no_grad():
        return model(torch.tensor([ids]))[0, -1]

# step 1: full prefill for A and B, only the first 10 tokens of C (no logits needed)
step1 = build_batch([(a, 0, tables[0], True), (b, 0, tables[1], True), (c[:10], 0, tables[2], False)], block_size)
logits = runner.forward(step1)
next_a, next_b = logits.argmax(-1).tolist()

# step 2: A decodes, B decodes, and C's tokens 10–21, all in the same forward pass
step2 = build_batch([([next_a], len(a), tables[0], True), ([next_b], len(b), tables[1], True),
                     (c[10:], 10, tables[2], True)], block_size)
logits = runner.forward(step2)
print("query_start_loc", step2.query_start_loc, " seq_lens", step2.seq_lens,
      " logits_indices", step2.logits_indices.tolist())
print("positions", step2.positions.tolist())
for name, got, ids in [("A decode", logits[0], a + [next_a]), ("B decode", logits[1], b + [next_b]),
                       ("C 分块 prefill", logits[2], c)]:
    err = (got - reference(ids)).abs().max().item()
    print(f"{name:14s} 与单独计算的最大误差 {err:.1e}")
    assert err < 1e-3
```

```text
query_start_loc [0, 1, 2, 14]  seq_lens [8, 6, 22]  logits_indices [0, 1, 13]
positions [7, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
A decode       与单独计算的最大误差 1.8e-05
B decode       与单独计算的最大误差 1.5e-05
C 分块 prefill   与单独计算的最大误差 2.5e-05
```

Requests at three different stages complete in the same forward pass, and every result matches computing alone (the errors come from a different order of floating-point accumulation).

!!! source "Source code"
    - **vLLM**: `GPUModelRunner._prepare_inputs` (`vllm/v1/worker/gpu_model_runner.py`) builds `positions`, `query_start_loc` and `slot_mapping` vectorized with numpy, where `logits_indices = query_start_loc[1:] - 1`. To avoid rebuilding all state every step, vLLM keeps an `InputBatch` that persists across steps (`gpu_input_batch.py`, the "persistent batch") and updates it incrementally each step (`_update_states`).
    - The attention metadata is built by each backend's metadata builder. The comments of `FlashAttentionMetadata` (`vllm/v1/attention/backends/flash_attn.py`) define three lengths: `context_len` (already cached), `query_len` (newly computed this step) and `seq_len` (their sum).
    - **SGLang**: the input of one forward pass is a `ForwardBatch` (`srt/model_executor/forward_batch_info.py`). `extend_prefix_lens` corresponds to "the length already cached", `extend_seq_lens` to "the length computed this step", and `out_cache_loc` to the slot mapping. `ForwardMode` distinguishes `EXTEND` (prefill), `DECODE` and `MIXED` (chunked prefill mixed with decode), because SGLang takes faster paths for pure decode batches (CUDA Graphs, for example).

!!! inference "Inference view"
    Building this metadata is **CPU work**. When the model is fast (small models, decode batches), the time the CPU takes to prepare inputs can be comparable to the GPU's compute time, leaving the GPU waiting. So inference engines care a lot about this part's efficiency: vLLM uses numpy and preallocated pinned buffers to avoid Python loops and repeated allocations, and overlaps "scheduling the next step" with "running the current step" (async scheduling); SGLang's overlap scheduler does the same thing. This book's `build_batch` is written with Python loops for readability and is far less efficient.

!!! interview "In an interview"
    On variable-length batching: inference engines do not use a padded `[batch, max_len]` layout (with large length differences, most compute is wasted on padding); they lay the new tokens of all requests for this step end to end in one dimension. The layers other than attention (linear layers, normalization, MLP) compute token by token and do not care about the batch structure at all; only attention tells requests apart using metadata: `query_start_loc` (`cu_seqlens_q`) partitions each request's queries, `seq_lens` gives the total KV length, block tables and `slot_mapping` give where the KV lives, `positions` feeds RoPE, and `logits_indices` picks the positions to sample. Prefill, a middle segment of chunked prefill and decode are just different combinations of "cached length" and "this step's length", so they can mix in the same forward pass; this is the basis of the unified token budget in vLLM V1's scheduler.

## Exercises {#练习}

**1. Deriving the metadata.** A batch has three requests: request 1 has 30 tokens cached and decodes this step; request 2 is new with a 6-token prompt; request 3 has 16 tokens cached and computes the next 4 tokens this step, but its prompt is not finished (total length 40). Write `query_start_loc`, `seq_lens`, `positions` and `logits_indices`.

??? success "Answer"
    - `query_start_loc = [0, 1, 7, 11]`;
    - `seq_lens = [31, 6, 20]`;
    - `positions = [30, 0, 1, 2, 3, 4, 5, 16, 17, 18, 19]`;
    - `logits_indices = [0, 6]`. Request 3's prompt is not finished, so it needs no sampling.

**2. Why are decode batches special?** If all stages can use the same layout, why do engines still distinguish "pure decode batches"?

??? success "Answer"
    In a pure decode batch each request has exactly 1 query token and the shape is regular (N = B), so it can use dedicated decode kernels (split-KV, Flash-Decoding) and can be recorded as a CUDA Graph and replayed (the [CUDA Graphs chapter](graphs-compile.md)). The shape of a mixed batch differs every step, which is hard to cover with CUDA Graphs. vLLM's `FULL_AND_PIECEWISE` mode works this way: pure decode batches use a full CUDA Graph, and mixed batches use "piecewise" CUDA Graphs, with the attention part executed separately outside the graph.

## Summary {#小结}

- [x] Inference engines lay the new tokens of all requests end to end in one dimension, without padding.
- [x] Layers other than attention compute token by token and do not care about the batch structure; attention tells requests apart with `query_start_loc`, `seq_lens`, block tables and `slot_mapping`.
- [x] Prefill, chunked prefill and decode are just different combinations of "cached length" and "this step's length", and can mix in the same forward pass.
- [x] The output layer is computed only for positions that need sampling; building the metadata is CPU overhead, which inference engines try to overlap with GPU computation.
