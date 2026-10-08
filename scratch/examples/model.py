"""一个小 GPT：RMSNorm、RoPE、SwiGLU、共享词嵌入——和 LLaMA / Qwen 同一类结构，只是小得多"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    seq_len: int = 128
    rope_theta: float = 10000.0


def rope_cos_sin(T, head_dim, theta, device=None):
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    freqs = torch.outer(torch.arange(T, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()                          # [T, head_dim / 2]


def apply_rope(x, cos, sin):                                  # x: [B, H, T, D]，前后两半配对旋转
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.d_model * 2 // 3 // 32 * 32 or 32     # SwiGLU：三个矩阵，中间维取约 8/3·d，参数量和 4d 的两层 MLP 相当
        self.gate_up = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)            # 前置归一化 + 残差
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight                  # 输入输出共享词嵌入：小模型里词表占了大部分参数
        self.apply(self._init)
        for name, p in self.named_parameters():               # 写回残差流的投影按层数缩小，深层网络一开始也不会发散
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        cos, sin = rope_cos_sin(idx.shape[1], self.cfg.d_model // self.cfg.n_head, self.cfg.rope_theta, idx.device)
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())

    def num_params(self):
        return sum(p.numel() for p in self.parameters())      # 共享的词嵌入只算一次

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.seq_len:])[:, -1] / temperature
            if top_k is not None:
                kth = logits.topk(top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
