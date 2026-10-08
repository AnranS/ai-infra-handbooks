import os
import sys
import threading

os.sched_setaffinity(0, {0, 1})                  # 只允许在 CPU 0、1 上运行（等价于 taskset -c 0,1）
print("绑核之后：", sorted(os.sched_getaffinity(0)))

seen = []
t = threading.Thread(target=lambda: seen.append(sorted(os.sched_getaffinity(0))))
t.start()
t.join()
print("之后创建的线程：", seen[0])

sys.stdout.flush()              # fork 会把还没写出去的输出缓冲区也复制一份，先刷新，否则这几行会打印两遍
pid = os.fork()
if pid == 0:
    print("fork 出来的子进程：", sorted(os.sched_getaffinity(0)), flush=True)
    os._exit(0)
os.waitpid(pid, 0)
