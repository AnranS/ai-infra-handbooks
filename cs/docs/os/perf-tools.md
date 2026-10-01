# Linux 性能分析工具

<p class="lead">线上的推理服务变慢了、GPU 利用率上不去、偶尔卡住几秒，第一步不是改代码，而是搞清楚时间花在哪里。这一章讲一套系统的方法（USE）和一组最常用的工具：所有工具的数据来源 `/proc`，看整体的 top、vmstat、mpstat、pidstat、iostat，采样分析的 perf 和火焰图，看系统调用的 strace，看 Python 调用栈的 py-spy，生产环境里的 eBPF，以及 GPU 侧的 nvidia-smi 和 DCGM。最后用一个"GPU 利用率上不去"的排查把它们串起来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. USE 方法是什么？拿它检查一台推理服务器，要看哪些资源？
    2. 平均负载（load average）很高，CPU 却很空闲，可能是什么情况？
    3. `/proc/<pid>/status` 里的自愿切换和非自愿切换分别说明什么？
    4. `perf stat` 和 `perf record` 分别做什么？火焰图怎么读？
    5. 一个线上的 Python 推理进程卡住了，不重启、不改代码，怎么看它的每个线程在干什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 对每一种资源检查三件事：使用率（Utilization，忙的时间比例）、饱和度（Saturation，排队等待的程度）、错误（Errors）。推理服务器要看的资源：CPU（使用率、运行队列、节流）、内存（可用内存、换页、OOM）、磁盘（使用率、等待时间）、网络（带宽、重传、丢包）、GPU（SM 活跃度、显存带宽、显存用量、PCIe / NVLink 带宽、降频原因、Xid 错误）。
    2. 平均负载统计的是"可运行的任务 + 不可中断等待（D 状态）的任务"。CPU 空闲而负载高，说明有很多任务卡在 D 状态，通常是在等 I/O：磁盘很慢、网络文件系统（NFS）挂死、或者内存回收时等待写回。用 `ps -eo state,pid,cmd | grep '^D'` 找出这些任务，`cat /proc/<pid>/stack`（需要 root）看它卡在内核的哪里，`iostat -x` 看磁盘。
    3. 自愿切换是任务主动让出 CPU：等 I/O、等锁、`sleep`、等条件变量，多说明它在等资源；非自愿切换是任务还想跑，却被调度器抢占了：时间片用完或者有更高优先级的任务，多说明 CPU 不够用、线程太多在争抢（或者被 cgroup 节流）。
    4. `perf stat` 计数：程序运行期间发生了多少次某种事件（上下文切换、缺页，硬件支持时还有周期数、指令数、缓存缺失），用来判断"是什么类型的问题"；`perf record` 采样：每秒上千次记录当前的调用栈，统计时间主要花在哪些函数上，用来定位"具体在哪里"。火焰图把采样的调用栈画成一层层的方块：横向宽度是这个函数（包括它调用的函数）占的时间比例，纵向是调用深度；找又宽又平的"平顶"，那就是最耗时的地方。横轴的先后顺序没有时间含义。
    5. `py-spy dump --pid <pid>` 打印所有 Python 线程当前的调用栈（加 `--native` 连 C 扩展和 CUDA 库的栈一起），卡住时一眼就能看出每个线程停在哪一行；`py-spy top` 像 top 一样实时显示最耗时的函数，`py-spy record` 生成火焰图。它从外部读进程的内存，不需要重启、不需要改代码，开销也很小。

## 先有方法：USE

![图：USE 方法——对每种资源问利用率、饱和度、错误](../assets/figures/use-method.svg){.aig-svg}

性能问题最怕"看到什么查什么"。**USE 方法**对每一种资源检查三件事：

- **使用率**（Utilization）：这个资源忙的时间比例；
- **饱和度**（Saturation）：有多少工作在排队等它；
- **错误**（Errors）：出错的次数。

| 资源 | 使用率 | 饱和度 | 错误 | 常用工具 |
| --- | --- | --- | --- | --- |
| CPU | 各核的使用率（`mpstat -P ALL`） | 运行队列长度、非自愿切换、cgroup 节流 | | `mpstat`、`pidstat`、`vmstat` |
| 内存 | 已用 / 可用内存 | 换页（`vmstat` 的 si/so）、内存回收 | OOM kill（`dmesg`） | `free`、`vmstat`、`/proc/meminfo` |
| 磁盘 | `iostat -x` 的 %util | 请求的平均等待时间（await）、队列长度 | I/O 错误（`dmesg`） | `iostat` |
| 网络 | 带宽 | 丢包、重传、套接字队列 | 错误计数 | `ss -tin`、`sar -n DEV` |
| GPU | SM 活跃度、显存带宽 | 降频、等待 | Xid 错误、ECC 错误 | `nvidia-smi`、DCGM |

## /proc：所有工具的数据来源

`top`、`vmstat`、`ps` 这些工具都是在读 `/proc`：`/proc/stat` 是 CPU 时间的累计值，`/proc/meminfo` 是内存，`/proc/<pid>/status`、`/proc/<pid>/stat` 是某个进程。自己读一下：

```python title="proc_stat.py"
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
```

```text title="输出（本机示例）"
过去 0.5 秒的 CPU 时间分布：用户态 5.8%，内核态 0.2%，等 I/O 0.0%，被宿主机拿走 0.0%，空闲 93.9%
1、5、15 分钟平均负载（可运行 + 不可中断等待的任务数）： (1.7841796875, 1.9814453125, 2.08935546875)
可用内存 55.2 GiB（总共 62.6 GiB，页缓存 30.4 GiB）
```

几个字段值得注意：**等 I/O**（iowait）是 CPU 空闲、但有任务在等 I/O 的时间，它高说明磁盘或者网络存储是瓶颈；**被宿主机拿走**（steal）是虚拟机里本该运行、却被宿主机调度给别的虚拟机的时间，在云主机上它高说明邻居太吵。**平均负载**统计的是"可运行 + 不可中断等待"的任务数，所以负载高不一定是 CPU 忙，也可能是很多任务卡在磁盘或 NFS 上（D 状态）。

`/proc/<pid>/status` 里还有两个很有用的计数：自愿切换和非自愿切换（`pidstat -w` 按秒显示它们）：

```python title="ctxsw_kinds.py"
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
```

```text title="输出"
睡眠 200 次，自愿切换增加了至少 150 次： True
两个进程抢同一个 CPU 1 秒，子进程被抢占（非自愿切换）至少 20 次： True
```

一个推理引擎的调度线程非自愿切换很多，说明它经常被抢占：同一个核上有别的线程在争抢，或者容器被节流了（见[容器](containers.md)）。

## 常用命令

| 命令 | 看什么 |
| --- | --- |
| `top` / `htop`（按 `H` 显示线程） | 哪个进程、哪个线程在占 CPU 和内存 |
| `vmstat 1` | 每秒一行：运行队列（r）、D 状态任务（b）、换页（si/so）、上下文切换（cs）、CPU 时间分布（us/sy/wa/st） |
| `mpstat -P ALL 1` | 每个核的使用率：只有一个核是 100%，说明是单线程瓶颈（比如 Python 的调度循环） |
| `pidstat -u -w -t 1` | 每个线程的 CPU 使用率和自愿 / 非自愿切换 |
| `iostat -x 1` | 每块盘的吞吐、IOPS、等待时间、使用率 |
| `ss -tinp` | 每个 TCP 连接的 RTT、拥塞窗口、重传次数 |
| `dmesg -T` | 内核日志：OOM kill、硬件错误、GPU 的 Xid 错误 |
| `nvidia-smi dmon` / `pmon` | 每张 GPU / 每个进程的利用率、显存、功耗、频率 |
| `nvidia-smi -q -d PERFORMANCE,CLOCK` | 当前频率和降频原因（温度、功耗上限） |
| `dcgmi dmon` | DCGM 的细粒度指标：SM 活跃度、Tensor Core 活跃度、显存带宽、PCIe / NVLink 流量 |

GPU 要特别注意：`nvidia-smi` 的 GPU-Util 只表示"采样周期内有没有 kernel 在跑"，一个只用了一个 SM 的小 kernel 也会让它显示 100%。判断 GPU 真正忙不忙，要看 DCGM 的 SM 活跃度（SM Active）、SM 占用率和 Tensor Core 活跃度。

## perf：计数与采样

**perf** 是 Linux 自带的性能分析工具，有两种主要用法。

**计数**（`perf stat`）：统计程序运行期间发生了多少次某种事件。软件事件（上下文切换、CPU 迁移、缺页）在哪里都能用；硬件事件（周期数、指令数、缓存缺失、分支预测失败）要 CPU 的性能计数器，很多虚拟机和容器里不可用（本机就显示 `<not supported>`）。有硬件计数器时，最有用的是 **IPC**（每周期执行的指令数）：计算密集的循环通常能到 2～4，而随机访问内存的程序（比如[虚拟内存](virtual-memory.md)一章里的指针追逐）可能只有零点几——CPU 使用率都是 100%，但后者大部分时间在等内存。

**采样**（`perf record`）：每秒上千次打断程序、记录当前的调用栈，最后统计时间花在哪些函数上。下面的程序有两个函数，一个做的工作是另一个的 4 倍：

```c title="hot.c" flags="-g -fno-omit-frame-pointer"
#include <stdio.h>

/* 两个函数：一个占大部分时间，一个占小部分。用 perf record 采样，看它能不能找出热点 */
__attribute__((noinline)) static double hot_loop(long n) {
  double x = 0;
  for (long i = 1; i < n; i++) x += 1.0 / i;
  return x;
}

__attribute__((noinline)) static double cold_loop(long n) {
  double x = 0;
  for (long i = 1; i < n; i++) x += 1.0 / (i + 1);
  return x;
}

int main(void) {
  double s = 0;
  for (int r = 0; r < 20; r++) s += hot_loop(40000000 + r) + cold_loop(10000000 + r);   /* 参数每轮不同，编译器没法把调用提到循环外 */
  printf("%.3f\n", s);
  return 0;
}
```

```text title="输出"
675.538
```

用 `perf record` 采样（虚拟机里用软件事件 `cpu-clock`），再用 `perf report` 统计：

```python title="perf_profile.py" ci="no"
import re
import shutil
import subprocess

if not shutil.which("perf"):
    print("没有 perf，跳过（Ubuntu 上装 linux-tools-$(uname -r)）")
    raise SystemExit
# 按时间采样（cpu-clock 是软件事件，虚拟机里没有硬件计数器也能用），每秒 999 次
subprocess.run(["perf", "record", "-q", "-e", "cpu-clock", "-F", "999", "-g", "-o", "perf.data", "./hot"],
               check=True, capture_output=True)
report = subprocess.run(["perf", "report", "-i", "perf.data", "--stdio", "--no-children", "--sort", "symbol"],
                        capture_output=True, text=True).stdout
for name in ("hot_loop", "cold_loop"):
    m = re.search(rf"([\d.]+)%.*\b{name}\b", report)
    print(f"{name} 占了 {m.group(1)}% 的采样" if m else f"没有找到 {name}")
```

```text title="输出（本机示例）"
hot_loop 占了 79.78% 的采样
cold_loop 占了 20.22% 的采样
```

编译时加 `-fno-omit-frame-pointer` 是为了让 perf 能沿着帧指针回溯调用栈（或者用 `--call-graph dwarf`）。把采样画成**火焰图**：`perf script | stackcollapse-perf.pl | flamegraph.pl > out.svg`（FlameGraph 是 Brendan Gregg 的一组脚本）。每个方块是一个函数，宽度是它占的时间比例，上面一层是它调用的函数；找最宽的"平顶"，那就是时间真正花掉的地方。

对 Python 程序，perf 默认只看得到解释器的 C 函数（`_PyEval_EvalFrameDefault` 占满全图），看不到是哪个 Python 函数。Python 3.12 起可以用 `python -X perf` 让 perf 看到 Python 函数名；更常用的是 **py-spy**：

- `py-spy dump --pid <pid>`：打印每个线程当前的 Python 调用栈，进程卡住时首选（加 `--native` 还能看到 C 扩展、CUDA 库里的栈）；
- `py-spy top --pid <pid>`：实时显示最耗时的函数；
- `py-spy record -o flame.svg --pid <pid>`：采样生成火焰图。

它从外部读取目标进程的内存，不需要重启、不需要改代码，开销很小，适合线上。Python 代码的剖析方法见 Python 手册的[性能分析与优化](python://concurrency/performance/)。

## strace：看系统调用

`strace` 记录一个进程的每一次系统调用：参数、返回值、耗时（`-T`）、时间戳（`-tt`），`-c` 汇总次数和总耗时，`-f` 跟踪子进程和线程，`-P` 只看涉及某个文件的调用。比如同样读一个 64 MiB 的文件，每次读的块大小不同，系统调用次数差了两个多数量级：

```python title="strace_reads.py"
import os
import shutil
import subprocess
import sys

with open("data.bin", "wb") as f:                   # 准备一个 64 MiB 的文件
    f.write(os.urandom(64 << 20))

reader = """
import os, sys
fd = os.open("data.bin", os.O_RDONLY)
while os.read(fd, int(sys.argv[1])):
    pass
"""
if not shutil.which("strace"):
    print("没有 strace，跳过")
    raise SystemExit
for chunk in (4096, 1 << 20):
    # -c 汇总系统调用次数，-P 只统计涉及这个文件的调用
    out = subprocess.run(["strace", "-c", "-P", "data.bin", "-e", "trace=read", sys.executable, "-c", reader, str(chunk)],
                         capture_output=True, text=True).stderr
    calls = next(line.split()[3] for line in out.splitlines() if line.rstrip().endswith("read"))
    print(f"每次读 {chunk // 1024} KiB：{calls} 次 read 系统调用")
```

```text title="输出"
每次读 4 KiB：16385 次 read 系统调用
每次读 1024 KiB：65 次 read 系统调用
```

（最后一次 `read` 返回 0，表示读到了文件末尾，所以多一次。）strace 适合回答"它为什么卡住"（停在哪个系统调用上）、"它在打开哪些文件"、"为什么这么多系统调用"。它基于 ptrace，每次系统调用都要停下来通知 strace，会让程序慢很多倍，线上只能短时间用；需要低开销时用 `perf trace` 或 eBPF。

## eBPF：生产环境里的探针

eBPF 让你把一小段经过内核校验的程序挂到内核的几乎任何位置（系统调用、调度事件、网络包、磁盘 I/O、函数入口），在内核里汇总统计，只把结果交给用户态，开销很低，适合在生产环境上长时间跑。常用的现成工具（bcc 工具集和 bpftrace）：

| 工具 | 回答的问题 |
| --- | --- |
| `runqlat` | 任务在运行队列里等了多久才拿到 CPU（CPU 争抢） |
| `offcputime` | 任务不在 CPU 上时，都在等什么（锁、I/O、睡眠），配合火焰图 |
| `biolatency` | 磁盘 I/O 的延迟分布 |
| `tcpretrans` | 哪些连接在重传 |
| `execsnoop`、`opensnoop` | 谁在启动进程、谁在打开文件 |

它们需要 root 或 `CAP_BPF` 权限，容器里通常要在宿主机上运行。

## 一次排查：GPU 利用率上不去

现象：一个推理服务 decode 的吞吐上不去，`nvidia-smi` 显示 GPU-Util 在 60% 左右，DCGM 的 SM 活跃度更低。按 USE 的思路：

1. **GPU 不忙，谁在拖它？** 看 CPU：`mpstat -P ALL 1` 发现有一个核一直是 100%，`pidstat -u -t 1` 找到是调度进程的主线程——典型的单线程 CPU 瓶颈；
2. **主线程在忙什么？** `py-spy top --pid <调度进程>`：时间主要花在准备下一批输入、处理输出和反分词上；
3. **有没有被打扰？** `pidstat -w -t 1` 看到主线程的非自愿切换很多，同一个核上还有分词的线程；容器里再看 cgroup 的 `nr_throttled` 有没有增长；
4. **GPU 时间线上的空隙**：用 Nsight Systems 抓一段时间线，kernel 之间有明显的空白，每一步开头 GPU 都在等 CPU 发射 kernel。

对应的改进：用 CUDA Graphs 减少 kernel 发射的 CPU 开销，重叠调度让 CPU 准备下一步和 GPU 计算这一步同时进行，把反分词挪到单独的进程，调度主线程绑核、别的线程别挤在它的核上。这些在[Profiling 推理引擎](serving://perf/profiling/)、[性能分析：Nsight](cuda://tools/profiling/)和[重叠调度](minisgl://schedule/overlap/)里有详细的讲解。

!!! interview "面试怎么答"
    被问"线上推理服务延迟变高了，你怎么排查"：先确认现象和范围（哪个指标、从什么时候、所有请求还是部分请求），再按 USE 方法逐个资源看：CPU（各核使用率，有没有单核 100% 的单线程瓶颈，非自愿切换和 cgroup 节流）、内存（可用内存、换页、OOM）、磁盘和网络（等待时间、重传）、GPU（DCGM 的 SM 活跃度而不是 `nvidia-smi` 的 GPU-Util，降频原因，Xid 错误）。定位到进程以后，用 `py-spy dump` / `top` 看 Python 线程在做什么，`perf` 采样看 C/C++ 部分，`strace` 看卡在哪个系统调用，Nsight Systems 看 GPU 时间线上的空隙。最后给出改进并用同样的指标验证。

## 练习

**1. 负载很高，CPU 却很闲。** 一台推理服务器的平均负载从平时的 5 涨到了 60，但 `top` 显示 CPU 空闲超过 80%，服务的部分请求卡住不返回。可能是什么？怎么确认？

??? success "参考答案"
    负载统计的是可运行加上不可中断等待（D 状态）的任务，CPU 空闲说明新增的负载几乎都是 D 状态的任务，它们在内核里等 I/O。常见原因是网络文件系统（比如模型或日志所在的 NFS）挂死或者极慢，也可能是本地盘出了问题、或者内存紧张时等待回收写回。确认：`ps -eo state,pid,wchan:32,cmd | awk '$1=="D"'` 列出 D 状态的任务和它们等待的内核函数；`cat /proc/<pid>/stack` 看完整的内核栈（需要 root）；`iostat -x 1` 看各个盘的 await 和使用率；`dmesg -T` 看有没有 NFS 超时（"server not responding"）或磁盘错误。卡住的请求多半是在读写这些文件（比如同步写日志）。

**2. GPU-Util 100%，吞吐却很低。** `nvidia-smi` 显示 GPU-Util 一直是 100%，但吞吐远低于预期。这说明什么？应该看哪些指标？

??? success "参考答案"
    GPU-Util 只表示采样周期内有没有 kernel 在运行，不表示 SM 有多忙：一个只用了很少几个线程块的小 kernel 在跑，它也是 100%。应该看 DCGM 的细粒度指标：SM 活跃度（有 warp 驻留的 SM 占比）、SM 占用率（驻留的 warp 数占上限的比例）、Tensor Core 活跃度、显存带宽利用率（DRAM Active）。如果 SM 活跃度低，说明 kernel 太小、并行度不够（batch 太小、kernel 的网格太小）；如果显存带宽利用率高而 Tensor Core 活跃度低，就是 decode 这类访存受限的负载，符合预期，要从 batch、量化、KV 读取上想办法。再用 Nsight Compute 看具体 kernel 的瓶颈。

**3. perf 只看到解释器。** 对一个 Python 进程做 `perf record`，结果 90% 的时间都在 `_PyEval_EvalFrameDefault` 里，看不出是哪个 Python 函数。怎么办？

??? success "参考答案"
    `_PyEval_EvalFrameDefault` 是 CPython 执行字节码的主循环，所有 Python 函数都在它里面执行，perf 只看到 C 层面的栈。办法：Python 3.12 以上用 `python -X perf`（或设置环境变量 `PYTHONPERFSUPPORT=1`）启动，解释器会为每个 Python 函数生成一小段跳板，perf 就能显示 Python 函数名；不能重启的线上进程用 `py-spy record --pid <pid>`（需要看 C 扩展里的时间时加 `--native`），它直接从进程内存里读出 Python 的调用栈。

## 小结

- [x] USE 方法：对 CPU、内存、磁盘、网络、GPU 逐个检查使用率、饱和度和错误。
- [x] 所有工具都在读 `/proc`：iowait 高是 I/O 瓶颈，steal 高是邻居太吵；负载包含 D 状态任务；自愿切换多是在等资源，非自愿切换多是在抢 CPU。
- [x] `perf stat` 计数（有硬件计数器时看 IPC），`perf record` 采样定位热点，火焰图找最宽的平顶；Python 用 `-X perf` 或 py-spy。
- [x] strace 看系统调用和卡住的位置，但开销大；生产环境用 eBPF（runqlat、offcputime、biolatency）。
- [x] GPU 的 GPU-Util 不代表 SM 忙，要看 DCGM 的 SM 活跃度；GPU 利用率上不去时先查 CPU 侧的单线程瓶颈和 kernel 发射空隙。
