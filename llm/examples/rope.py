"""rope.py —— 旋转位置编码的两种等价实现。"""

import torch


def inv_freq(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))   # [d/2]


def rope_half(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """"前后两半一组"（transformers 的写法）。x: [..., T, d]，pos: [T]"""
    freqs = pos.float()[:, None] * inv_freq(x.shape[-1], base)[None, :]   # [T, d/2]
    cos, sin = torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)                                 # rotate_half
    return x * cos + rotated * sin


def rope_complex(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """"相邻两维一组"（原始 LLaMA 的写法）：把 (x0, x1) 看成复数 x0 + i·x1，乘以 e^{i·m·θ}。"""
    freqs = pos.float()[:, None] * inv_freq(x.shape[-1], base)[None, :]
    rot = torch.polar(torch.ones_like(freqs), freqs)                       # e^{i m θ}
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * rot).flatten(-2)
