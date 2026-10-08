import os
import socket
import sys
import time
from multiprocessing import shared_memory

MiB = 1 << 20
TOTAL, CHUNK = 512 * MiB, MiB
payload = b"\1" * CHUNK


def through(send, recv_all):
    sys.stdout.flush()
    pid = os.fork()
    if pid == 0:
        recv_all()
        os._exit(0)
    t = time.perf_counter()
    for _ in range(TOTAL // CHUNK):
        send(payload)
    os.waitpid(pid, 0)
    return TOTAL / (time.perf_counter() - t) / 1e9


# 1. 管道
r, w = os.pipe()
def pipe_recv():
    got = 0
    while got < TOTAL:
        got += len(os.read(r, CHUNK))
print(f"管道：{through(lambda b: os.write(w, b), pipe_recv):.1f} GB/s")

# 2. Unix 域套接字
a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
def sock_recv():
    got = 0
    while got < TOTAL:
        got += len(b.recv(CHUNK))
print(f"Unix 域套接字：{through(a.sendall, sock_recv):.1f} GB/s")

# 3. 共享内存：生产者把数据写进共享缓冲区，消费者直接在原地读，数据不经过内核
BUF = 64 * MiB
shm = shared_memory.SharedMemory(create=True, size=BUF)
shm.buf[:BUF] = bytes(BUF)                          # 先把页都分配好，只测稳定状态
block = b"\1" * BUF
go_r, go_w = os.pipe()                              # 每一轮用管道传 1 个字节做同步，不传数据
done_r, done_w = os.pipe()
sys.stdout.flush()
pid = os.fork()
if pid == 0:
    for _ in range(TOTAL // BUF):
        os.read(go_r, 1)
        assert shm.buf[0] == 1 and shm.buf[BUF - 1] == 1   # 消费者直接读同一块物理内存
        os.write(done_w, b"k")
    os._exit(0)
t = time.perf_counter()
for _ in range(TOTAL // BUF):
    shm.buf[:BUF] = block                           # 生产者：一次内存拷贝写进共享缓冲区
    os.write(go_w, b"g")
    os.read(done_r, 1)
os.waitpid(pid, 0)
print(f"共享内存：{TOTAL / (time.perf_counter() - t) / 1e9:.1f} GB/s")
shm.close()
shm.unlink()
