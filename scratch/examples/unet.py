# 给 32×32 图像用的小 UNet：时间用正弦编码注入每个残差块
import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(torch.nn.functional.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(torch.nn.functional.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=128, tdim=256):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        c1, c2, c3 = base, base * 2, base * 2
        self.inp = nn.Conv2d(3, c1, 3, padding=1)
        self.d1, self.d2 = Block(c1, c1, tdim), Block(c1, c2, tdim)        # 32 → 16
        self.down1, self.down2 = nn.Conv2d(c1, c1, 3, 2, 1), nn.Conv2d(c2, c2, 3, 2, 1)
        self.mid = Block(c2, c3, tdim)                                     # 8×8 的瓶颈
        self.u2, self.u1 = Block(c3 + c2, c2, tdim), Block(c2 + c1, c1, tdim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 3, 3, padding=1))

    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t * 1000, self.tdim))
        h0 = self.inp(x)
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.down1(h1), temb)
        m = self.mid(self.down2(h2), temb)
        u2 = self.u2(torch.cat([self.up(m), h2], dim=1), temb)
        u1 = self.u1(torch.cat([self.up(u2), h1], dim=1), temb)
        return self.out(u1)
