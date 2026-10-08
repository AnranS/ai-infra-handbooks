# TLB 覆盖范围 = 表项数 × 页大小。2048 项是服务器 CPU 二级 TLB（STLB）的典型规模
entries = 2048
for name, page in [("4 KiB 页", 4 << 10), ("2 MiB 大页", 2 << 20)]:
    print(f"{name}：{entries} 项能覆盖 {entries * page / (1 << 20):,.0f} MiB")

# 一个 70B 模型的 BF16 权重有 140 GB：在 CPU 上做 offload 或者 CPU 推理时，扫一遍要碰多少个页
weights = 140e9
print(f"140 GB 的权重：4 KiB 页约 {weights / 4096 / 1e6:.0f} 百万个，2 MiB 页约 {weights / (2 << 20) / 1e3:.0f} 千个")
