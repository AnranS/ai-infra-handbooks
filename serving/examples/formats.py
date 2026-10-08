"""formats.py —— 部署中常见的低精度格式（伪量化实现）：FP8 的几种缩放粒度、MXFP4、NVFP4，以及 FP8 KV Cache。

所有函数都是"量化后立刻反量化"，返回与输入同形状的 FP32 张量，用来评估精度损失。
"""

import torch

from mini_llm import KVCache

FP8_MAX = 448.0                                                     # E4M3 能表示的最大值
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # E2M1 能表示的全部非负值


def fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale


def fp8_per_tensor(x):
    return fp8(x, x.abs().max().clamp(min=1e-12) / FP8_MAX)


def fp8_per_row(x):
    """权重按输出通道、激活按 token：每行一个缩放因子。"""
    return fp8(x, x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX)


def fp8_blockwise(x, block=(128, 128)):
    """DeepSeek-V3 的做法：权重每 128×128 一个缩放因子；激活用 (1, 128)，即每个 token 每 128 个通道一个。"""
    rows, cols = x.shape[-2], x.shape[-1]
    br, bc = min(block[0], rows), block[1]
    g = x.reshape(*x.shape[:-2], rows // br, br, cols // bc, bc)
    scale = g.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-12) / FP8_MAX
    return fp8(g, scale).reshape(x.shape)


def fp4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """把 x / scale 舍入到最近的 E2M1 值。"""
    y = (x / scale).clamp(-6.0, 6.0)
    idx = (y.abs().unsqueeze(-1) - FP4_VALUES).abs().argmin(-1)
    return FP4_VALUES[idx] * y.sign() * scale


def mxfp4(w, block=32):
    """OCP MX 格式：每 32 个数共享一个 2 的幂次缩放（E8M0），使块内最大值落在 E2M1 的范围内。"""
    g = w.reshape(*w.shape[:-1], -1, block)
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = 2.0 ** (torch.floor(torch.log2(amax)) - 2)                  # E2M1 的最大指数是 2
    return fp4(g, scale).reshape(w.shape)


def nvfp4(w, block=16):
    """NVFP4：每 16 个数一个 FP8（E4M3）缩放因子，外加一个整个张量的 FP32 缩放因子。"""
    g = w.reshape(*w.shape[:-1], -1, block)
    global_scale = w.abs().max().clamp(min=1e-12) / (FP8_MAX * 6.0)
    block_scale = (g.abs().amax(-1, keepdim=True) / 6.0 / global_scale).to(torch.float8_e4m3fn).float()
    return fp4(g, block_scale.clamp(min=1e-12) * global_scale).reshape(w.shape)


class FP8KVCache(KVCache):
    """写入 KV Cache 时量化成 FP8（每层一个缩放因子，由写入的数据决定）。"""

    def update(self, layer, k, v):
        return super().update(layer, fp8_per_tensor(k), fp8_per_tensor(v))
