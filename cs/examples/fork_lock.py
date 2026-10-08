import os
import sys
import threading
import time
import warnings

warnings.simplefilter("ignore", DeprecationWarning)    # Python 3.12 起，多线程进程里调用 fork 会给出警告
lock = threading.Lock()


def worker():
    with lock:                  # 后台线程拿着一把锁（日志锁、内存分配器的锁、某个库的内部锁……）
        time.sleep(1.0)


threading.Thread(target=worker, daemon=True).start()
time.sleep(0.1)                 # 确保后台线程已经拿到锁
sys.stdout.flush()              # fork 会把还没写出去的输出缓冲区也复制一份，先刷新，否则这几行会打印两遍
pid = os.fork()
if pid == 0:                    # 子进程：只复制了调用 fork 的这一个线程，锁的"已占用"状态却原样复制了过来
    print("子进程等 2 秒，拿到锁了吗：", lock.acquire(timeout=2), flush=True)   # os._exit 不会刷新缓冲区
    os._exit(0)
os.waitpid(pid, 0)
print("父进程里，后台线程 1 秒后照常释放了锁：", lock.acquire(timeout=2))
