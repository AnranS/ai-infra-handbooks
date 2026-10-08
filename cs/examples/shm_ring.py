import os
import sys
import threading
import time
from multiprocessing import shared_memory

# 按 vLLM 的 ShmRingBuffer 的布局写一个简化版：一个写者、多个读者的广播环形缓冲区
N_READER, CHUNKS, CHUNK = 2, 4, 64
META = CHUNKS * CHUNK                              # 数据区之后是元数据区：每块 1 个写标志 + 每个读者 1 个读标志
_lock = threading.Lock()


def fence():                                       # 和 vLLM 的 memory_fence 一样：拿一次锁再放掉，当作一道内存屏障
    with _lock:
        pass


class Ring:
    def __init__(self, shm):
        self.buf = shm.buf

    def _meta(self, i):
        return META + i * (1 + N_READER)

    def write(self, seq, payload):
        i, m = seq % CHUNKS, self._meta(seq % CHUNKS)
        while True:                                # 这一块要么没写过，要么所有读者都读完了，才能覆盖
            fence()
            if self.buf[m] == 0 or all(self.buf[m + 1:m + 1 + N_READER]):
                break
            time.sleep(0)
        self.buf[m] = 0                            # 先清写标志
        self.buf[i * CHUNK] = len(payload)
        self.buf[i * CHUNK + 1:i * CHUNK + 1 + len(payload)] = payload
        self.buf[m + 1:m + 1 + N_READER] = bytes(N_READER)   # 再清读者标志……
        fence()
        self.buf[m] = 1                            # ……最后才标记"写好了"。顺序反过来，读者可能看到中间状态

    def read(self, seq, reader):
        i, m = seq % CHUNKS, self._meta(seq % CHUNKS)
        while True:                                # 写好了、而且自己还没读过
            fence()
            if self.buf[m] == 1 and self.buf[m + 1 + reader] == 0:
                break
            time.sleep(0)
        n = self.buf[i * CHUNK]
        data = bytes(self.buf[i * CHUNK + 1:i * CHUNK + 1 + n])
        fence()
        self.buf[m + 1 + reader] = 1               # 读完才标记，写者这时才能覆盖这一块
        return data


shm = shared_memory.SharedMemory(create=True, size=META + CHUNKS * (1 + N_READER))
shm.buf[:shm.size] = bytes(shm.size)
N_MSG = 20
results = []
for reader in range(N_READER):
    r, w = os.pipe()
    sys.stdout.flush()
    if os.fork() == 0:                             # 读者进程：按顺序读 20 条，把结果通过管道交回父进程
        ring = Ring(shm)
        got = [ring.read(s, reader).decode() for s in range(N_MSG)]
        os.write(w, ("|".join(got)).encode())
        os._exit(0)
    results.append(r)

ring = Ring(shm)                                   # 父进程是写者：只有 4 个块，写得比读得快时会等读者
for s in range(N_MSG):
    ring.write(s, f"step {s}".encode())
for reader, r in enumerate(results):
    got = os.read(r, 4096).decode().split("|")
    os.wait()
    print(f"读者 {reader}：按顺序收到 {len(got)} 条广播，第一条 {got[0]!r}，最后一条 {got[-1]!r}，全部正确：",
          got == [f"step {s}" for s in range(N_MSG)])
shm.close()
shm.unlink()
