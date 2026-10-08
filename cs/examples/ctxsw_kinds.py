import os
import sys
import time


def switches(pid="self"):
    d = {}
    for line in open(f"/proc/{pid}/status"):
        if "ctxt_switches" in line:
            k, v = line.split(":")
            d[k] = int(v)
    return d["voluntary_ctxt_switches"], d["nonvoluntary_ctxt_switches"]


v0, _ = switches()
for _ in range(200):
    time.sleep(0.001)                              # 主动睡眠：每次都是一次"自愿"切换
v1, _ = switches()
print("睡眠 200 次，自愿切换增加了至少 150 次：", v1 - v0 >= 150)

os.sched_setaffinity(0, {0})                       # 两个一直在算的进程挤在同一个 CPU 上
sys.stdout.flush()
pid = os.fork()
if pid == 0:
    end = time.time() + 1.0
    while time.time() < end:
        pass
    os._exit(0)
_, n0 = switches(pid)
end = time.time() + 1.0
while time.time() < end:
    pass
_, n1 = switches(pid)
os.waitpid(pid, 0)
print("两个进程抢同一个 CPU 1 秒，子进程被抢占（非自愿切换）至少 20 次：", n1 - n0 >= 20)
