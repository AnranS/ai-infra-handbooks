import mmap
import os

fd = os.open("direct.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
try:
    os.write(fd, b"x" * 1000)                       # 长度不是块大小的整数倍，缓冲区也没有对齐
except OSError as e:
    print("不对齐的 O_DIRECT 写入失败：", e.strerror)
buf = mmap.mmap(-1, 4096)                          # mmap 分配的内存按页对齐
buf.write(b"y" * 4096)
print("对齐的写入成功：", os.write(fd, buf), "字节")
os.close(fd)
