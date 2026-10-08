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
    root = target(torch.tensor([prompt]), cache)[0, -1].argmax().item()   # prefill，得到第一个 token
    committed, out, target_calls, accepted_per_call = list(prompt), [], 1, []
    while len(out) < max_new:
        tokens, parents = build_draft_tree(draft, committed, root, branching)
        pos = torch.tensor([len(committed) + d for d in depths(parents)])
        logits, new_kv = forward_with_mask(target, torch.tensor(tokens), pos, cache, tree_mask(parents))
        target_calls += 1
        path, node = [0], 0                                  # 从根出发，走目标模型认可的最长路径
        while True:
            pred = logits[node].argmax().item()
            child = next((c for c, p in enumerate(parents) if p == node and tokens[c] == pred), None)
            if child is None:
                break
            path.append(child)
            node = child
        accepted_per_call.append(len(path))
        for layer, (k, v) in enumerate(new_kv):              # 只把路径上节点的 KV 追加进缓存
            cache.update(layer, k[:, :, path], v[:, :, path])
        for n in path:
            committed.append(tokens[n])
            out.append(tokens[n])
            if tokens[n] == eos or len(out) >= max_new:
                return out[:max_new], target_calls, accepted_per_call
        root = logits[node].argmax().item()                  # 目标模型在路径终点给出的下一个 token
    return out, target_calls, accepted_per_call
