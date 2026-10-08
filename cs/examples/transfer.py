# 把 16 GB 的权重从 CPU 内存搬到 GPU，各种路径大约要多久（带宽取实际能跑到的量级）
GB = 1e9
size = 16 * GB
paths = [
    ("PCIe 4.0 x16，锁页内存", 25 * GB),
    ("PCIe 4.0 x16，可换页内存（先拷进驱动的锁页缓冲区）", 12 * GB),
    ("PCIe 5.0 x16，锁页内存", 50 * GB),
    ("NVLink-C2C（Grace Hopper 的 CPU 到 GPU）", 450 * GB),
    ("400 Gb/s 网卡从远端读（GPUDirect RDMA）", 45 * GB),
]
for name, bw in paths:
    print(f"{name}：{size / bw * 1e3:,.0f} ms")
