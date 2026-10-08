import os
import time

fd = os.open("big.bin", os.O_RDONLY)               # 上一个脚本写好并 fsync 过的 256 MiB 文件
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # 把这个文件的页从页缓存里丢掉（只能丢干净页，所以前面要 fsync）


def read_all():
    os.lseek(fd, 0, os.SEEK_SET)
    t = time.perf_counter()
    while os.read(fd, 8 << 20):
        pass
    return time.perf_counter() - t


cold, warm = read_all(), read_all()
print(f"第一次读（从盘上读）：{256 / cold:,.0f} MiB/s")
print(f"第二次读（页缓存命中）：{256 / warm:,.0f} MiB/s，快 {cold / warm:.0f} 倍")
