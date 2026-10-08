import os
import time

MiB = 1 << 20


def dirty_mib():                                   # 整个系统里还没写回磁盘的脏页
    for line in open("/proc/meminfo"):
        if line.startswith("Dirty:"):
            return int(line.split()[1]) / 1024


data = os.urandom(MiB)
with open("big.bin", "wb") as f:
    t0 = time.perf_counter()
    for _ in range(256):
        f.write(data)
    f.flush()                                      # 从 Python 的缓冲区交给内核：write 系统调用返回
    t1 = time.perf_counter()
    print(f"write 256 MiB：{(t1 - t0) * 1e3:.0f} ms（数据在页缓存里，脏页约 {dirty_mib():.0f} MiB）")
    os.fsync(f.fileno())                           # 等数据和元数据真正写到盘上
    t2 = time.perf_counter()
    print(f"fsync：{(t2 - t1) * 1e3:.0f} ms（之后脏页约 {dirty_mib():.0f} MiB）")

# 像数据库的预写日志那样：每次写 4 KiB 就 fsync 一次
block = os.urandom(4096)
with open("wal.log", "wb") as f:
    t0 = time.perf_counter()
    for _ in range(200):
        f.write(block)
        f.flush()
        os.fsync(f.fileno())
    per = (time.perf_counter() - t0) / 200
print(f"4 KiB 写入 + fsync：每次 {per * 1e3:.2f} ms，每秒最多约 {1 / per:,.0f} 次")
