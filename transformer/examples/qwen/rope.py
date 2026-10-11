"""旋转位置编码（RoPE）：把"第几个位置"转进 query 和 key 里，让注意力分数只跟相对距离有关。

Qwen3.5 的三个细节：只旋转每个头前 1/4 的维度（256 维里的 64 维，partial_rotary_factor=0.25）；
基频 theta = 10,000,000；为了图像和视频，位置有三个轴（时间、高、宽，M-RoPE），32 个频率按 11/11/10 交错分给三个轴。
纯文本时三个轴的位置相同，M-RoPE 就退化成普通的 RoPE——这个文件写的就是这个退化版。
"""
import torch


def rope_tables(cfg, positions):
    """positions: [T]（或每行不同的 [B, T]）的整数位置 -> cos, sin 各 [T, rot_dim]（rot_dim = head_dim * partial_rotary_factor = 64）"""
    rot = int(cfg.head_dim * cfg.partial_rotary_factor)
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 32 个频率：从 1 一路降到 1/theta
    angles = positions.float()[..., None] * inv_freq                                  # [T, 32]：位置 × 频率 = 转角
    angles = torch.cat([angles, angles], dim=-1)                                      # [T, 64]：前后两半配成一对，用同一个角
    return angles.cos(), angles.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, T, 头数, head_dim]。只转前 rot_dim 维，剩下的维度原样拼回去。"""
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    cos, sin = cos[..., None, :].to(x.dtype), sin[..., None, :].to(x.dtype)         # 对批和头广播；表是 fp32 算的，用的时候转成 x 的精度
    x_rot = x_rot * cos + rotate_half(x_rot) * sin                                    # 二维旋转：(a, b) -> (a cos - b sin, b cos + a sin)
    return torch.cat([x_rot, x_pass], dim=-1)
