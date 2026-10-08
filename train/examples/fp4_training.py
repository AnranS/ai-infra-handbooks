import torch

torch.manual_seed(0)
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # FP4 E2M1 能表示的非负值


def round_e2m1(y, stochastic=False, gen=None):
    """y 已经除过缩放因子（|y| ≤ 6）：取到 E2M1 网格上，就近舍入或随机舍入"""
    a = y.abs().clamp(max=6).contiguous()
    hi = torch.bucketize(a, GRID).clamp(1, 7)                 # a 落在 GRID[hi-1] 和 GRID[hi] 之间
    lo_v, hi_v = GRID[hi - 1], GRID[hi]
    frac = (a - lo_v) / (hi_v - lo_v)
    up = torch.rand(a.shape, generator=gen) < frac if stochastic else frac >= 0.5
    return torch.where(up, hi_v, lo_v) * y.sign()


def e4m3(s):
    return s.to(torch.float8_e4m3fn).float().clamp_min(1e-12)  # 缩放因子本身用 FP8 E4M3 存


def nvfp4(x, stochastic=False, gen=None):
    """NVFP4 伪量化：沿最后一维每 16 个数一个缩放因子"""
    b = x.reshape(-1, 16)
    scale = e4m3(b.abs().amax(1, keepdim=True) / 6)
    return (round_e2m1(b / scale, stochastic, gen) * scale).reshape(x.shape)


def nvfp4_2d(w):
    """16×16 的二维块共用一个缩放因子"""
    R, C = w.shape
    b = w.reshape(R // 16, 16, C // 16, 16)
    scale = e4m3(b.abs().amax(dim=(1, 3), keepdim=True) / 6)
    return (round_e2m1(b / scale) * scale).reshape(R, C)


# ① 梯度里的小分量：每 16 个数里有一个 6.0（决定了缩放因子），其余是 0.1
g = torch.full((1024,), 0.1)
g[::16] = 6.0
gen = torch.Generator().manual_seed(1)
rtn = torch.stack([nvfp4(g) for _ in range(1000)]).mean(0)
sr = torch.stack([nvfp4(g, True, gen) for _ in range(1000)]).mean(0)
small = torch.ones(1024, dtype=torch.bool)
small[::16] = False
print(f"小分量的真实值 0.100；量化 1000 次再平均：就近舍入 {rtn[small].mean():.3f}，随机舍入 {sr[small].mean():.3f}")

# ② 权重的分块方向：前向 Y = X Wᵀ 沿 W 的行（输入维）分块；反向 dX = dY W 要沿 W 的列分块
w = torch.randn(256, 512)
fwd, bwd = nvfp4(w), nvfp4(w.T).T                             # 前向用的 W，反向用的 W（按列分块后转置回来比较）
print(f"一维分块（1×16）：前向和反向用的权重有 {(fwd != bwd).float().mean():.0%} 的元素不同")
print(f"二维分块（16×16）：有 {(nvfp4_2d(w) != nvfp4_2d(w.T).T).float().mean():.0%} 的元素不同")
