# 集合通信的时间 = 每步的延迟 × 步数 + 传输量 ÷ 带宽。对比机内 NVLink 和机间 400 Gb/s 网络
LINKS = {  # 名称: (单向带宽 GB/s, 单步延迟 us)
    "NVLink 4": (450, 2),
    "NVLink 5": (900, 2),
    "IB NDR 400G": (50, 8),
}


def ring_allreduce(nbytes, n, bw, lat):
    """环形 all-reduce：2(n-1) 步，每步每张卡发出 nbytes/n 字节"""
    steps = 2 * (n - 1)
    return steps * lat * 1e-6 + steps * (nbytes / n) / (bw * 1e9)


print("8 卡 all-reduce 的耗时（ms）")
print(f"{'消息大小':28s}" + "".join(f"{k:>14s}" for k in LINKS))
for name, size in [("decode 一层的激活 256 KB", 256 << 10), ("prefill 一层的激活 64 MB", 64 << 20),
                   ("梯度同步 1 GB", 1 << 30)]:
    print(f"{name:26s}" + "".join(f"{ring_allreduce(size, 8, bw, lat) * 1e3:14.3f}"
                                  for bw, lat in LINKS.values()))
print()
print("同一条 256 KB 的消息，卡数从 8 张涨到 72 张（NVLink 5）：")
for n in (8, 16, 32, 72):
    t = ring_allreduce(256 << 10, n, 900, 2) * 1e3
    print(f"  {n:2d} 卡：{t:6.3f} ms，其中固定延迟 {2 * (n - 1) * 2 / 1e3:5.3f} ms（占 {2 * (n - 1) * 2 / 1e3 / t:.0%}）")
