import mmap
import resource

MiB = 1 << 20
SIZE = 256 * MiB


def minor_faults():
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def touch(buf, step):
    for off in range(0, SIZE, step):     # 每页写一个字节：第一次写一页时才真正分配物理内存
        buf[off] = 1


PRIVATE = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS   # 私有的匿名映射，和 malloc 大块内存时拿到的一样
buf = mmap.mmap(-1, SIZE, flags=PRIVATE)  # 只是保留了 256 MiB 的虚拟地址，还没有物理页
before = minor_faults()
touch(buf, 4096)
print(f"普通页：写满 256 MiB 触发了约 {round((minor_faults() - before) / 1000)} 千次缺页")

buf2 = mmap.mmap(-1, SIZE, flags=PRIVATE)
buf2.madvise(mmap.MADV_HUGEPAGE)          # 建议内核用透明大页（2 MiB）
before = minor_faults()
touch(buf2, 4096)
print(f"透明大页：同样写满 256 MiB 触发了 {minor_faults() - before} 次缺页")
