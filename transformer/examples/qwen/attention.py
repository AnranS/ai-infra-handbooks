"""门控注意力：Qwen3.5 的全注意力层，24 层里的 6 层。

和教科书的多头注意力比有四处不同：GQA（8 个 query 头共用 2 组 key/value）、QK-Norm（q 和 k 各过一个
RMSNorm 再旋转）、只旋转前 1/4 维的 RoPE、以及输出门——q_proj 一次算出两份，一份当 query，
一份过 sigmoid 当门，逐维乘在注意力的输出上。
"""
import math

import torch
import torch.nn as nn

from qwen.norm import RMSNorm
from qwen.rope import apply_rope


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, self.n_head, self.n_kv, self.d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(H, self.n_head * self.d * 2, bias=False)   # 两倍宽：一半 query，一半门
        self.k_proj = nn.Linear(H, self.n_kv * self.d, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.d, H, bias=False)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps)                     # 按头归一化：权重只有 head_dim 个
        self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, state=None, mask=None):
        B, T, _ = x.shape
        q, gate = self.q_proj(x).view(B, T, self.n_head, 2 * self.d).chunk(2, dim=-1)   # 每个头的 512 维：前 256 是 query，后 256 是门
        k = self.k_proj(x).view(B, T, self.n_kv, self.d)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                                  # [B, 头, T, d]
        if state is not None:                                                             # KV cache：接上以前的 key/value，再把新的存回去
            if state.k is not None:
                k, v = torch.cat([state.k, k], dim=2), torch.cat([state.v, v], dim=2)
            state.k, state.v = k, v
        S = k.shape[2]                                                                    # key 的总数（以前的 + 这次的）
        rep = self.n_head // self.n_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)           # GQA：每组 KV 复制给 4 个 query 头
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d)                               # [B, 头, T, S]
        causal = torch.ones(T, S, dtype=torch.bool, device=x.device).triu(S - T + 1)       # 第 i 个新 token 只能看到第 S-T+i 个及之前的 key
        scores = scores.masked_fill(causal, float("-inf"))
        if mask is not None:                                                              # 批处理时的填充位：[B, S]，False 的 key 谁都不许看
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)   # 用最小的有限数而不是 -inf：填充位自己那一行全被挡住时，softmax 不会算出 NaN
        out = torch.softmax(scores, dim=-1) @ v                                            # [B, 头, T, d]
        out = out.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1))   # 输出门
        return self.o_proj(out)
