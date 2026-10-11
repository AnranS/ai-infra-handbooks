import torch


def rope_tables(positions, rot_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    half = rot // 2
    rotated = torch.cat([-x_rot[..., half:], x_rot[..., :half]], dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)
