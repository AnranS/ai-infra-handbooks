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
    input_ids: torch.Tensor        # [N] 所有请求本步要计算的 token，首尾相接
    positions: torch.Tensor        # [N] 每个 token 在自己序列中的位置
    slot_mapping: torch.Tensor     # [N] 每个 token 的 K/V 写到哪个 slot
    query_start_loc: list[int]     # [B+1] 第 i 个请求的 token 是 input_ids[qsl[i]:qsl[i+1]]（cu_seqlens_q）
    seq_lens: list[int]            # [B] 计算完本步后，每个请求的上下文长度
    block_tables: list[list[int]]  # [B] 每个请求的块表
    logits_indices: torch.Tensor   # [M] 需要采样的 token 在 N 中的下标（通常是各请求的最后一个 token）


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
        cos, sin = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)   # 对所有头广播
        for i, layer in enumerate(m.layers):
            attn, h = layer.self_attn, layer.input_layernorm(x)
            q = apply_rope(attn.q_norm(attn.q_proj(h).view(N, self.nh, self.hd)), cos, sin)     # QK-Norm 在 RoPE 之前
            k = apply_rope(attn.k_norm(attn.k_proj(h).view(N, self.nkv, self.hd)), cos, sin)
            v = attn.v_proj(h).view(N, self.nkv, self.hd)
            self.kv.write(i, b.slot_mapping, k, v)                            # 先写入，再读出整段上下文
            o = paged_attention(q, self.kv, i, b.block_tables, b.seq_lens, b.query_start_loc, self.scale)
            x = x + attn.o_proj(o.reshape(N, self.nh * self.hd))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.lm_head(m.norm(x[b.logits_indices]))
