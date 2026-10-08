import mmap

GiB = 1 << 30
MAP_NORESERVE = 0x4000          # Linux 上的取值；Python 的 mmap 模块没有导出这个常量


def rss_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE   # 不为这段地址预留交换空间，允许超额申请
before = rss_mib()
m = mmap.mmap(-1, 100 * GiB, flags=flags)      # 比这台机器的物理内存还大
print("保留 100 GiB 的虚拟地址：成功")
print("常驻内存（RSS）几乎没变：", rss_mib() - before < 1)
m[: 64 << 20] = b"\1" * (64 << 20)                # 真正写 64 MiB
print("写了 64 MiB 之后 RSS 约增加 64 MiB：", 63 < rss_mib() - before < 66)
