# Advanced speculative decoding: tree drafts and EAGLE

<p class="lead">The <a href="llm://inference/serving/#投机解码">speculative decoding</a> section of the LLM handbook covered the basics: a draft guesses a few tokens, the target model verifies them in one forward pass, the output is unchanged under greedy decoding, and the distribution is unchanged under sampling. This chapter covers the advanced techniques inference engines actually use: organizing drafts into a tree and verifying several candidate paths at once with tree attention; drafts "grown on the target model" such as EAGLE and MTP; and under what loads speculative decoding helps, and how engines implement it.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why use tree drafts? What does the tree attention mask look like? What position encodings do tree nodes get?
    2. After verification, does the KV of accepted tokens need recomputing?
    3. What is the input to EAGLE's draft model? And what is MTP?
    4. Why does speculative decoding slow down when the batch is large?

??? success "Answers (try first, then expand to compare)"
    1. A chain draft has only one candidate, so if an early token is wrong everything after it is wasted; a tree draft branches where the draft model is uncertain, preparing several candidates at once and verifying the whole tree in one pass, so more tokens are accepted on average. The tree attention mask lets each node see only the prefix, its ancestors and itself; its position is "prefix length + depth".
    2. No: verification already computed KV for every node in the tree; the nodes on the accepted path are reused directly (compacted into contiguous positions), and the rejected ones are discarded.
    3. EAGLE's draft network reads the target model's hidden states (features) and the embedding of the next token, predicts the next step's features, then passes them through the target model's output layer to get tokens; MTP is the multi-token prediction layers a model is trained with, which can serve directly as the draft at inference time.
    4. With a large batch the GPU's compute is already fully used, so computing several more draft tokens for verification is no longer "free"; the cost of each slower step outweighs the gain from the extra accepted tokens, and the speedup vanishes or even turns negative.

![Figure: one step of speculative decoding: draft, verify, correct](../assets/figures/spec-decode.svg){.aig-svg}

<!-- comic ../assets/comics/speculative.webp is in Chinese; put it back once the English version exists -->

## Why trees {#为什么要用树}

If each draft step guesses only one token (a chain draft), the first wrong guess wastes everything after it. But the draft model often "knows it is unsure": at the first position it thinks both A and B are plausible. If continuations are prepared for both A and B, the target model only needs to approve one of them to accept a longer sequence. Organize these candidates into a tree:

<!-- i18n:diagram 4d91423884 -->
```text
            root (the token the target model gave in the previous step)
           /                    \
         A1                      B1             ← level 1: the draft's top-2
       /     \                /     \
     A2      A2'            B2      B2'         ← level 2: top-2 again under each node
     |        |              |       |
     A3      A3'            B3      B3'         ← level 3: top-1
```

Drag the width, depth and draft hit rate to see how many tokens to expect per step and when the speedup turns negative:

<div class="aig-widget" data-widget="spectree"></div>

The key is that **the whole tree can be verified in one forward pass**. Line up all the nodes as one sequence for the target model, and use a special attention mask so each node sees only "the committed prefix + its ancestors + itself", as if each path were computed on its own; each node's RoPE position is "prefix length + depth", so siblings on the same level share a position. After verification, start from the root and walk down in the direction where "the target model's prediction is exactly one of the children", obtaining the longest accepted path.

## Implementation {#实现}

```python title="tree_spec.py"
"""tree_spec.py —— 树形投机解码：草稿组织成一棵 token 树，用"树注意力掩码"一次前向验证整棵树。

树中每个节点只能看到：已提交的前缀、自己的所有祖先、自己。节点的位置 = 前缀长度 + 深度。
验证后沿树走出与目标模型贪心结果一致的最长路径，并直接复用这条路径上的 KV，不需要重算。
"""

import math

import torch
import torch.nn.functional as F

from mini_llm import KVCache, apply_rope, rope_cos_sin


@torch.no_grad()
def forward_with_mask(model, ids, positions, cache: KVCache, mask):
    """ids: [T] 新 token；mask: [T, T] 新 token 之间的可见性（True 表示可见）；它们都能看到缓存中的全部前缀。
    返回 logits [T, V] 和各层新 token 的 (K, V)，不修改 cache。"""
    cfg = model.cfg
    T, P = len(ids), cache.length
    full_mask = torch.cat([torch.ones(T, P, dtype=torch.bool), mask], dim=1)
    cos, sin = rope_cos_sin(positions, cfg.hd, cfg.rope_theta)
    x = model.embed_tokens(ids[None])
    new_kv = []
    for i, layer in enumerate(model.layers):
        attn, h = layer.self_attn, layer.input_layernorm(x)
        q = apply_rope(attn.q_norm(attn.q_proj(h).view(1, T, attn.nh, attn.hd).transpose(1, 2)), cos, sin)
        k = apply_rope(attn.k_norm(attn.k_proj(h).view(1, T, attn.nkv, attn.hd).transpose(1, 2)), cos, sin)
        v = attn.v_proj(h).view(1, T, attn.nkv, attn.hd).transpose(1, 2)
        new_kv.append((k, v))
        K = torch.cat([cache.k[i], k], dim=2) if P else k
        V = torch.cat([cache.v[i], v], dim=2) if P else v
        rep = attn.nh // attn.nkv
        out = F.scaled_dot_product_attention(q, K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1),
                                             attn_mask=full_mask)
        x = x + attn.o_proj(out.transpose(1, 2).reshape(1, T, attn.nh * attn.hd))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return model.lm_head(model.norm(x))[0], new_kv


def tree_mask(parents: list[int]) -> torch.Tensor:
    """parents[i] 是节点 i 的父节点下标（根为 -1）。节点 i 能看到自己和所有祖先。"""
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = i
        while j != -1:
            mask[i, j] = True
            j = parents[j]
    return mask


def depths(parents):
    d = []
    for p in parents:
        d.append(0 if p == -1 else d[p] + 1)
    return d


@torch.no_grad()
def build_draft_tree(draft_model, context: list[int], root: int, branching: list[int]):
    """用草稿模型从 root 往下展开：第 l 层每个节点取草稿模型的 top-branching[l] 个 token。
    返回 (tokens, parents)，第 0 个节点是 root 本身。"""
    tokens, parents, frontier = [root], [-1], [0]
    for width in branching:
        next_frontier = []
        for node in frontier:
            path, j = [], node
            while j != -1:
                path.append(tokens[j])
                j = parents[j]
            logits = draft_model(torch.tensor([context + path[::-1]]))[0, -1]
            for t in logits.topk(width).indices.tolist():
                tokens.append(t)
                parents.append(node)
                next_frontier.append(len(tokens) - 1)
        frontier = next_frontier
    return tokens, parents


@torch.no_grad()
def tree_speculative_generate(target, draft, prompt: list[int], max_new: int, branching: list[int], eos=None):
    cache = KVCache(target.cfg.num_hidden_layers)
    root = target(torch.tensor([prompt]), cache)[0, -1].argmax().item()   # prefill, getting the first token
    committed, out, target_calls, accepted_per_call = list(prompt), [], 1, []
    while len(out) < max_new:
        tokens, parents = build_draft_tree(draft, committed, root, branching)
        pos = torch.tensor([len(committed) + d for d in depths(parents)])
        logits, new_kv = forward_with_mask(target, torch.tensor(tokens), pos, cache, tree_mask(parents))
        target_calls += 1
        path, node = [0], 0                                  # from the root, follow the longest path the target model agrees with
        while True:
            pred = logits[node].argmax().item()
            child = next((c for c, p in enumerate(parents) if p == node and tokens[c] == pred), None)
            if child is None:
                break
            path.append(child)
            node = child
        accepted_per_call.append(len(path))
        for layer, (k, v) in enumerate(new_kv):              # append only the KV of nodes on the path to the cache
            cache.update(layer, k[:, :, path], v[:, :, path])
        for n in path:
            committed.append(tokens[n])
            out.append(tokens[n])
            if tokens[n] == eos or len(out) >= max_new:
                return out[:max_new], target_calls, accepted_per_call
        root = logits[node].argmax().item()                  # the next token the target model gives at the end of the path
    return out, target_calls, accepted_per_call
```

Note the last step: the K/V of the nodes on the path were already computed during verification with the right positions and context, so just pick them out and append them to the cache, with no recomputation. In real engines this step "compacts" the KV slots of accepted nodes into the right positions in the sequence (or only updates the block table), and the rejected nodes' slots are simply discarded.

First check that tree attention itself is correct: each node's logits in the tree should equal the result of computing "prefix + its ancestors + itself" as an ordinary sequence:

```python
import copy
import time
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer, generate
from quant import fake_quant_int
from tree_spec import depths, forward_with_mask, tree_mask, tree_speculative_generate

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
target = Transformer.from_pretrained(path)

prefix = tok("树形投机解码的核心是").input_ids
tokens, parents = [100, 200, 300, 400, 500, 600], [-1, 0, 0, 1, 1, 2]   # a small tree: the root has two children
cache = KVCache(target.cfg.num_hidden_layers)
with torch.no_grad():
    target(torch.tensor([prefix]), cache)
pos = torch.tensor([len(prefix) + d for d in depths(parents)])
logits, _ = forward_with_mask(target, torch.tensor(tokens), pos, cache, tree_mask(parents))
worst = 0.0
for node in range(len(tokens)):
    chain, j = [], node
    while j != -1:
        chain.append(tokens[j])
        j = parents[j]
    with torch.no_grad():
        ref = target(torch.tensor([prefix + chain[::-1]]))[0, -1]
    worst = max(worst, (logits[node] - ref).abs().max().item())
print("掩码（行：节点，列：可见的节点）：")
print(tree_mask(parents).int())
print(f"6 个节点与逐条路径单独计算的最大误差：{worst:.1e}")
assert worst < 1e-3
```

```text
掩码（行：节点，列：可见的节点）：
tensor([[1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1]], dtype=torch.int32)
6 个节点与逐条路径单独计算的最大误差：1.5e-05
```

Then run a complete tree-based speculative decode. The draft model is an INT4 fake-quantized version of the target model (a quantized self makes a decent draft: cheap, and very similar to the target), comparing a chain draft (top-1 per level, 4 levels deep) with a tree draft (top-2 for the first two levels, top-1 for the last two):

```python
draft = copy.deepcopy(target)
for layer in draft.layers:
    a, f = layer.self_attn, layer.mlp
    for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, f.gate_proj, f.up_proj, f.down_proj):
        lin.weight.data = fake_quant_int(lin.weight.data, 4, "group")

prompt = tok(tok.apply_chat_template([{"role": "user", "content": "解释一下什么是投机解码，以及它为什么能加速。"}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)).input_ids
reference = generate(target, torch.tensor([prompt]), 48, eos_token_id=tok.eos_token_id)[0].tolist()
for name, branching in [("链式草稿 [1, 1, 1, 1]", [1, 1, 1, 1]), ("树形草稿 [2, 2, 1, 1]", [2, 2, 1, 1])]:
    out, calls, accepted = tree_speculative_generate(target, draft, prompt, 48, branching, eos=tok.eos_token_id)
    print(f"{name}：目标模型前向 {calls} 次（普通解码需要 48 次），每次平均前进 {sum(accepted) / len(accepted):.2f} 个 token，"
          f"输出与贪心一致：{out == reference}")
    assert out == reference
```

```text title="输出"
链式草稿 [1, 1, 1, 1]：目标模型前向 23 次（普通解码需要 48 次），每次平均前进 2.23 个 token，输出与贪心一致：True
树形草稿 [2, 2, 1, 1]：目标模型前向 21 次（普通解码需要 48 次），每次平均前进 2.45 个 token，输出与贪心一致：True
```

Both drafts produce output identical to the target model's greedy result token for token. The tree draft spends more draft tokens (14 nodes per step versus 4) to get a longer average acceptance length and fewer target-model forward passes. In real systems the tree's shape is a parameter to tune: a wider tree accepts longer runs but also verifies more tokens (as we will see, this gets expensive at large batch sizes). EAGLE-2 goes further and decides the tree's shape **dynamically** from the draft model's confidence.

## Better drafts: EAGLE and MTP {#更好的草稿eagle-与-mtp}

Draft quality decides the acceptance rate. Common sources of drafts:

| Draft | How | Characteristics |
| --- | --- | --- |
| n-gram / prompt lookup | find the same suffix in the context and use the tokens after it as the draft | zero cost; works well for restating and code editing tasks (the LLM handbook's example) |
| A separate small model | a small model of the same family as the draft | needs a matching vocabulary; the draft itself has noticeable overhead |
| Medusa | add a few prediction heads on the target model's last layer, predicting tokens at +2, +3, ... | simple to train; the heads predict independently without conditioning on each other, so accuracy is limited |
| EAGLE | a very small draft network (about one Transformer layer) whose input is **the target model's hidden state** plus the next token's embedding, autoregressively predicting the following hidden states and tokens | uses the target model's internal features for a high acceptance rate; EAGLE-3 fuses features from several layers and simulates inference-time error accumulation during training |
| MTP | models such as DeepSeek-V3 are trained with multi-token prediction modules, used directly as the draft at inference | trained together with the target model, high quality; Qwen3-Next, Qwen3.5, GLM and others also have MTP layers |

EAGLE and MTP are currently the most effective and most widely used options in inference engines. What they share is being "grown on the target model": they read the hidden states the target model just computed, and the draft network has only a layer or two, adding almost no memory.

## When it helps {#什么时候有效}

The gain from speculative decoding depends on two things: **how many tokens each verification advances on average**, and **how much more verifying k draft tokens costs than ordinary decode**. The latter is set by the batch size: with a small batch decode is memory-bound and verifying a few more tokens is nearly free; with a large batch each step is already close to compute-bound, and multiplying the tokens to verify multiplies the time too. Estimate the speedup of Qwen2.5-7B on an H100 with the roofline model (acceptance probability 0.8 per draft token, 4 drafts per step, draft cost estimated at 5% of a decode step per token):

```python
P, weights, kv_per_token, bw, peak, ctx = 7.6e9, 15.2e9, 57344, 3.35e12, 989e12, 1024

def step_time(tokens, batch):                          # one forward step: the slower of computing and reading, plus a fixed cost
    return max(2 * P * tokens / (peak * 0.6), (weights + kv_per_token * ctx * batch) / bw) + 0.3e-3

alpha, k = 0.8, 4
expected = (1 - alpha ** (k + 1)) / (1 - alpha)        # average tokens advanced per verification
print(f"每次验证平均前进 {expected:.2f} 个 token")
print("batch   普通 decode   验证 k+1 个 token   加速比")
for batch in (1, 4, 16, 64, 128, 256):
    decode, verify = step_time(batch, batch), step_time(batch * (k + 1), batch)
    speedup = expected * decode / (verify + k * 0.05 * decode)
    print(f"{batch:5d}   {decode * 1e3:8.2f} ms   {verify * 1e3:10.2f} ms       {speedup:5.2f}x")
```

```text title="输出"
每次验证平均前进 3.36 个 token
batch   普通 decode   验证 k+1 个 token   加速比
    1       4.85 ms         4.85 ms        2.80x
    4       4.91 ms         4.91 ms        2.80x
   16       5.12 ms         5.12 ms        2.80x
   64       5.96 ms         8.50 ms        2.07x
  128       7.08 ms        16.69 ms        1.31x
  256       9.32 ms        33.09 ms        0.90x
```

With smaller batches the speedup approaches 3×; at a batch of 128 only 1.3× remains, and at 256 it is actually slower. So speculative decoding fits **low-concurrency scenarios sensitive to single-request latency** best (interactive chat, an agent's single long chain); high-throughput offline batch processing should usually turn it off, or adjust the draft length dynamically with load (both vLLM and SGLang are working on load-based adaptation).

## Implementing it in an engine {#在引擎中实现}

Speculative decoding places demands on almost every part of the engine:

- **Scheduler**: a request's `num_tokens` includes the draft tokens (vLLM's `num_tokens_with_spec`), and the scheduler allocates budget and KV slots by "the number of tokens to verify"; rejected drafts do not advance `num_computed_tokens`, and the KV they wrote is overwritten in the next step;
- **Model execution**: verification is a forward pass where "each request has k+1 queries", which the attention backend must support (tree drafts also need custom masks); to use CUDA Graphs, verification batches must have regular shapes;
- **Sampling**: compare one by one for greedy; for random sampling use rejection sampling (vLLM's `rejection_sampler.py`), done in batches on the GPU;
- **Drafts**: EAGLE/MTP need the target model's hidden states, and the draft network has its own KV Cache and CUDA Graphs;
- **Interaction with other features**: structured output (draft tokens must satisfy the grammar too), prefix caching, PD disaggregation (running drafts on decode instances), pipeline parallelism...

!!! source "Source code"
    - **vLLM**: `--speculative-config '{"method": "eagle3", "model": "<draft model>", "num_speculative_tokens": 3}'`, where `method` can also be `ngram`, `mtp`, `medusa`, `draft_model`, `suffix` and others. Drafting is in `vllm/v1/spec_decode/` (`eagle.py`, `ngram_proposer.py`, `suffix_decoding.py`, ...), verification in `vllm/v1/sample/rejection_sampler.py`, and `scheduled_spec_decode_tokens` and `update_draft_token_ids` in the scheduler move the draft tokens along.
    - **SGLang**: `--speculative-algorithm EAGLE3` (or `EAGLE`, `NEXTN` (MTP), `STANDALONE` (a separate draft model), `NGRAM` and others), `--speculative-num-steps` (draft depth), `--speculative-eagle-topk` (branches per level; greater than 1 means a tree draft), `--speculative-num-draft-tokens` (nodes taking part in verification); the implementation is in `srt/speculative/` (tree building and verification in `eagle_worker_v2.py`, `eagle_utils.py` and others).

!!! interview "In an interview"
    The common chain of follow-ups on speculative decoding: **why it speeds things up** (decode is memory-bound, so verifying several tokens is nearly free) → **why it doesn't change the output** (greedy compares one by one; sampling uses rejection sampling, which provably preserves the distribution) → **where drafts come from** (n-gram, small models, Medusa, EAGLE, MTP, each with pros and cons) → **tree drafts** (verifying several paths at once, the tree attention mask, depth as position, reusing the accepted path's KV directly) → **when not to use it** (verification gets expensive with large batches; this chapter's estimate table). Explaining the last point with numbers shows you truly understand the roofline.

## Exercises {#练习}

**1. Tree node count.** How many draft nodes (excluding the root) does a tree with branching `[2, 2, 1, 1]` have? And `[3, 3, 2]`? At batch = 64, which would you choose?

??? success "Answer"
    `[2, 2, 1, 1]`: 2 + 4 + 4 + 4 = 14 nodes; `[3, 3, 2]`: 3 + 9 + 18 = 30 nodes. At batch = 64, verifying 64 × 15 ≈ 960 tokens is already close to compute-bound (in the table above, verification gets noticeably more expensive at k+1 = 5), and a 30-node tree would double the verification cost again, not worth it; use a narrower tree, or even a chain draft, or turn speculative decoding off.

**2. Trees under sampling.** With random sampling, how does a tree draft keep the output distribution unchanged?

??? success "Approach"
    Do rejection sampling level by level: at a node, consider its children (the candidates the draft proposed) in turn, accepting each with probability $\min(1, p(x)/q(x))$; on rejection, update the target distribution to $\text{norm}(\max(0, p - q))$ before considering the next candidate (with $q$ adjusted according to how the candidates were sampled); when all candidates are rejected, sample a token from the final residual distribution and stop. The SpecInfer and EAGLE papers give proofs for the multi-candidate case. This is more complex than the chain case, so many systems use simpler trees or chains with random sampling.

## Summary {#小结}

- [x] Tree drafts prepare several candidates at once; the tree attention mask lets each node see only the prefix, its ancestors and itself, positions are "prefix length + depth", and one forward pass verifies the whole tree.
- [x] The KV on the accepted path is already computed during verification and reused directly; the rejected nodes are discarded.
- [x] EAGLE (a small draft network reading the target model's hidden states) and MTP (the model's own multi-token prediction layers) are the most widely used drafts today.
- [x] With small batches the speedup approaches the average acceptance length; with large batches verification gets expensive and the gain vanishes or turns negative.
