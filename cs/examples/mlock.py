import ctypes
import mmap

libc = ctypes.CDLL(None, use_errno=True)
MiB = 1 << 20


def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])            # 单位是 kB


buf = mmap.mmap(-1, 64 * MiB, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
before_lck, before_rss = status("VmLck"), status("VmRSS")
if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(64 * MiB)) != 0:   # 锁住：全部分配好物理页，并且不许换出
    raise OSError(ctypes.get_errno(), "mlock 失败（检查 ulimit -l）")
print("VmLck 增加了", (status("VmLck") - before_lck) // 1024, "MiB")
print("mlock 顺带把这 64 MiB 全部分配好了，RSS 增加约 64 MiB：", 63 < (status("VmRSS") - before_rss) / 1024 < 66)
libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(64 * MiB))
print("解锁之后 VmLck 回到", status("VmLck") - before_lck, "kB")
