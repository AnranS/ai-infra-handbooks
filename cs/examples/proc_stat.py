import os
import time


def cpu_times():
    fields = open("/proc/stat").readline().split()[1:]   # 第一行是所有 CPU 的合计，单位是时钟滴答
    user, nice, system, idle, iowait, irq, softirq, steal = map(int, fields[:8])
    return {"用户态": user + nice, "内核态": system + irq + softirq, "等 I/O": iowait, "被宿主机拿走": steal, "空闲": idle}


a = cpu_times()
time.sleep(0.5)
b = cpu_times()
total = sum(b.values()) - sum(a.values())
print("过去 0.5 秒的 CPU 时间分布：" + "，".join(f"{k} {100 * (b[k] - a[k]) / total:.1f}%" for k in a))
print("1、5、15 分钟平均负载（可运行 + 不可中断等待的任务数）：", os.getloadavg())
mem = {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo")}
print(f"可用内存 {mem['MemAvailable'] / 2**20:.1f} GiB（总共 {mem['MemTotal'] / 2**20:.1f} GiB，页缓存 {mem['Cached'] / 2**20:.1f} GiB）")
