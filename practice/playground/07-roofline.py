"""matplotlib：画 H100 的屋顶线图"""
import matplotlib.pyplot as plt
import numpy as np

PEAK, BW = 989e12, 3.35e12                           # H100 SXM：BF16 稠密算力、HBM 带宽
ai = np.logspace(-1, 4, 200)                         # 算术强度：FLOPs / 字节
plt.figure(figsize=(6, 3.6))
plt.loglog(ai, np.minimum(PEAK, ai * BW) / 1e12)
for name, x in [("decode, batch 1", 1), ("decode, batch 64", 64), ("prefill", 2000)]:          # 图里用英文：浏览器里没有中文字体
    y = min(PEAK, x * BW) / 1e12
    plt.scatter([x], [y])
    plt.annotate(name, (x, y), textcoords="offset points", xytext=(5, -12), fontsize=8)
plt.axvline(PEAK / BW, ls="--", c="gray")
plt.xlabel("arithmetic intensity (FLOP/byte)")
plt.ylabel("TFLOP/s")
plt.title(f"H100 roofline, ridge point = {PEAK / BW:.0f}")
print(f"屋脊点：算术强度 {PEAK / BW:.0f} FLOP/字节；低于它是访存受限，高于它是算力受限")
