import torch

torch.manual_seed(0)
FP8 = torch.float8_e4m3fn
MAX = torch.finfo(FP8).max                                # 448


def quant(x, scale):
    return (x / scale).clamp(-MAX, MAX).to(FP8).float() * scale   # 缩放 → 量化成 FP8 → 反量化


def per_tensor(x):
    return quant(x, x.abs().max() / MAX)


def per_block(x, rows, cols):
    """按 rows × cols 的块各自缩放（DeepSeek-V3：激活 1×128，权重 128×128）"""
    R, C = x.shape
    b = x.reshape(R // rows, rows, C // cols, cols)
    s = b.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / MAX
    return quant(b, s).reshape(R, C)


def rel_err(q, x):
    return ((q - x).norm() / x.norm()).item()


print(f"FP8 E4M3：最大值 {MAX:.0f}，最小正规数 {torch.finfo(FP8).tiny}")
for outlier in (1e2, 1e4, 3e4):
    a = torch.randn(256, 1024)
    a[:, :4] *= outlier                                  # 少数几个通道有离群值（LLM 的激活里很常见）
    normal = a[:, 128:]                                  # 与离群通道不在同一个 128 块里的普通通道
    t = rel_err(per_tensor(a)[:, 128:], normal)
    b = rel_err(per_block(a, 1, 128)[:, 128:], normal)
    print(f"离群值放大 {outlier:>7.0f} 倍：普通通道的相对误差，逐张量缩放 {t:5.1%}，每 1×128 块缩放 {b:5.1%}")
