# Linux profiling tools

<p class="lead">When a production inference service slows down, the GPU utilization will not rise, or it hangs for seconds now and then, the first step is not changing code but working out where the time goes. This chapter covers one systematic method (USE) and the tools used most: `/proc`, where every tool's data comes from; top, vmstat, mpstat, pidstat and iostat for the overall picture; perf and flame graphs for sampling; strace for the system calls; py-spy for Python call stacks; eBPF for production; and nvidia-smi and DCGM on the GPU side. It ends by tying them together through one "the GPU utilization will not rise" investigation.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is the USE method? Which resources does it check on an inference server?
    2. The load average is high while the CPU is idle. What could that be?
    3. What do voluntary and involuntary context switches in `/proc/<pid>/status` each say?
    4. What do `perf stat` and `perf record` each do? How do you read a flame graph?
    5. A production Python inference process has hung. Without restarting it or changing code, how do you see what each thread is doing?

??? success "Answers (try it yourself first, then expand)"
    1. For every resource, check three things: utilization (the fraction of time it is busy), saturation (how much work is queued for it) and errors. The resources on an inference server: the CPU (utilization, the run queue, throttling), memory (available memory, swapping, OOM), the disk (utilization, wait time), the network (bandwidth, retransmissions, drops) and the GPU (SM activity, memory bandwidth, memory use, PCIe / NVLink bandwidth, the throttle reasons, Xid errors).
    2. The load average counts "runnable tasks plus tasks in uninterruptible sleep (state D)". An idle CPU with a high load says many tasks are stuck in D, usually waiting on I/O: a very slow disk, a hung network filesystem (NFS), or waiting on write-back during reclamation. Find them with `ps -eo state,pid,cmd | grep '^D'`, see where they are stuck in the kernel with `cat /proc/<pid>/stack` (as root), and look at the disks with `iostat -x`.
    3. A voluntary switch is a task giving the CPU up itself: waiting on I/O, on a lock, in a `sleep`, on a condition variable, so many of them say it is waiting for a resource; an involuntary switch is a task that wanted to keep running being preempted: its slice ran out or something higher-priority arrived, so many of them say the CPU is short, too many threads are contending, or a cgroup is throttling it.
    4. `perf stat` counts: how many of some event happened while the program ran (context switches, page faults, and with hardware support cycles, instructions and cache misses), which says "which class of problem"; `perf record` samples: recording the current call stack a thousand times a second and summing which functions the time goes to, which says "exactly where". A flame graph draws the sampled stacks as layered boxes: the width is that function's share of the time (including what it called) and the height is the call depth; look for a wide flat "plateau", which is where the time goes. The horizontal order carries no meaning in time.
    5. `py-spy dump --pid <pid>` prints every Python thread's current call stack (with `--native` for the C extensions' and CUDA libraries' stacks too), and when it is hung, which line each thread is stopped on is immediately clear; `py-spy top` shows the most expensive functions live like top, and `py-spy record` produces a flame graph. It reads the process's memory from outside with no restart, no code change and very little overhead.

## A method first: USE {#先有方法use}

![Figure: the USE method - ask utilization, saturation and errors of every resource](../assets/figures/use-method.svg){.aig-svg}

The worst way to handle a performance problem is to chase whatever you happen to see. **The USE method** checks three things for every resource:

- **utilization**: the fraction of time this resource is busy;
- **saturation**: how much work is queued for it;
- **errors**: how many failures.

| Resource | Utilization | Saturation | Errors | The usual tools |
| --- | --- | --- | --- | --- |
| CPU | each core's utilization (`mpstat -P ALL`) | the run queue's length, involuntary switches, cgroup throttling | | `mpstat`, `pidstat`, `vmstat` |
| memory | used / available | swapping (`vmstat`'s si/so), reclamation | OOM kills (`dmesg`) | `free`, `vmstat`, `/proc/meminfo` |
| disk | %util in `iostat -x` | the average wait (await), the queue length | I/O errors (`dmesg`) | `iostat` |
| network | bandwidth | drops, retransmissions, socket queues | error counters | `ss -tin`, `sar -n DEV` |
| GPU | SM activity, memory bandwidth | throttling, waiting | Xid errors, ECC errors | `nvidia-smi`, DCGM |

## /proc: where every tool's data comes from {#proc所有工具的数据来源}

`top`, `vmstat` and `ps` all read `/proc`: `/proc/stat` holds the cumulative CPU times, `/proc/meminfo` the memory, and `/proc/<pid>/status` and `/proc/<pid>/stat` one process. Reading it yourself:

```python title="proc_stat.py"
import os
import time


def cpu_times():
    fields = open("/proc/stat").readline().split()[1:]   # the first line is every CPU's total, in clock ticks
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

```text title="output (on this machine)"
过去 0.5 秒的 CPU 时间分布：用户态 5.8%，内核态 0.2%，等 I/O 0.0%，被宿主机拿走 0.0%，空闲 93.9%
1、5、15 分钟平均负载（可运行 + 不可中断等待的任务数）： (1.7841796875, 1.9814453125, 2.08935546875)
可用内存 55.2 GiB（总共 62.6 GiB，页缓存 30.4 GiB）
```

A few fields deserve attention: **iowait** is time the CPU was idle while a task waited on I/O, and a high value says the disk or the network storage is the bottleneck; **steal** is time a virtual machine should have run and the host gave to another, and a high value on a cloud instance says the neighbours are noisy. **The load average** counts runnable plus uninterruptible tasks, so a high load is not necessarily a busy CPU and may be many tasks stuck on a disk or on NFS (state D).

`/proc/<pid>/status` also has two very useful counters, the voluntary and involuntary switches (`pidstat -w` shows them per second):

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
    time.sleep(0.001)                              # sleeping deliberately: every one is a "voluntary" switch
v1, _ = switches()
print("睡眠 200 次，自愿切换增加了至少 150 次：", v1 - v0 >= 150)

os.sched_setaffinity(0, {0})                       # two processes computing continuously crowded onto one CPU
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

```text title="output"
睡眠 200 次，自愿切换增加了至少 150 次： True
两个进程抢同一个 CPU 1 秒，子进程被抢占（非自愿切换）至少 20 次： True
```

Many involuntary switches on an inference engine's scheduling thread say it is preempted often: another thread is contending for the same core, or the container is being throttled (see [containers](containers.md)).

## The commands in common use {#常用命令}

| Command | What it shows |
| --- | --- |
| `top` / `htop` (`H` shows threads) | which process and thread are using the CPU and memory |
| `vmstat 1` | a line per second: the run queue (r), tasks in D (b), swapping (si/so), context switches (cs), the CPU time split (us/sy/wa/st) |
| `mpstat -P ALL 1` | each core's utilization: one core at 100% says a single-threaded bottleneck (Python's scheduling loop, say) |
| `pidstat -u -w -t 1` | each thread's CPU use and its voluntary / involuntary switches |
| `iostat -x 1` | each disk's throughput, IOPS, wait time and utilization |
| `ss -tinp` | each TCP connection's RTT, congestion window and retransmissions |
| `dmesg -T` | the kernel log: OOM kills, hardware errors, the GPU's Xid errors |
| `nvidia-smi dmon` / `pmon` | each GPU's / each process's utilization, memory, power and clocks |
| `nvidia-smi -q -d PERFORMANCE,CLOCK` | the current clocks and the throttle reasons (temperature, the power limit) |
| `dcgmi dmon` | DCGM's fine-grained metrics: SM activity, Tensor Core activity, memory bandwidth, PCIe / NVLink traffic |

The GPU deserves particular care: `nvidia-smi`'s GPU-Util only says whether a kernel was running during the sampling period, and a tiny kernel using one SM shows 100% too. To judge whether the GPU is really busy, read DCGM's SM activity (SM Active), SM occupancy and Tensor Core activity.

## perf: counting and sampling {#perf计数与采样}

**perf** is Linux's own profiler, used two main ways.

**Counting** (`perf stat`): how many of some event happened while the program ran. The software events (context switches, CPU migrations, page faults) work anywhere; the hardware events (cycles, instructions, cache misses, branch mispredictions) need the CPU's performance counters, which are unavailable in many virtual machines and containers (this machine shows `<not supported>`). With hardware counters, the most useful is **IPC** (instructions per cycle): a compute-heavy loop usually reaches 2 to 4, while a program chasing pointers through memory (the pointer chase in [virtual memory](virtual-memory.md)) may be a fraction of one, both at 100% CPU, with the latter waiting on memory most of the time.

**Sampling** (`perf record`): interrupt the program a thousand times a second, record the current call stack, and sum which functions the time goes to. The program below has two functions, one doing four times the work of the other:

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

```text title="output"
675.538
```

Sampling it with `perf record` (the software event `cpu-clock` inside a virtual machine) and summing with `perf report`:

```python title="perf_profile.py" ci="no"
import re
import shutil
import subprocess

if not shutil.which("perf"):
    print("没有 perf，跳过（Ubuntu 上装 linux-tools-$(uname -r)）")
    raise SystemExit
# sample by time (cpu-clock is a software event, usable in a virtual machine without hardware counters), 999 times a second
subprocess.run(["perf", "record", "-q", "-e", "cpu-clock", "-F", "999", "-g", "-o", "perf.data", "./hot"],
               check=True, capture_output=True)
report = subprocess.run(["perf", "report", "-i", "perf.data", "--stdio", "--no-children", "--sort", "symbol"],
                        capture_output=True, text=True).stdout
for name in ("hot_loop", "cold_loop"):
    m = re.search(rf"([\d.]+)%.*\b{name}\b", report)
    print(f"{name} 占了 {m.group(1)}% 的采样" if m else f"没有找到 {name}")
```

```text title="output (on this machine)"
hot_loop 占了 79.78% 的采样
cold_loop 占了 20.22% 的采样
```

Compiling with `-fno-omit-frame-pointer` is what lets perf walk the stack through the frame pointers (`--call-graph dwarf` is the alternative). To draw the samples as a **flame graph**: `perf script | stackcollapse-perf.pl | flamegraph.pl > out.svg` (FlameGraph is Brendan Gregg's set of scripts). Each box is a function, its width is its share of the time, and the layer above is what it called; look for the widest flat plateau, which is where the time really goes.

For a Python program, perf sees only the interpreter's C functions by default (`_PyEval_EvalFrameDefault` filling the graph) and not which Python function. From Python 3.12, `python -X perf` lets perf see the Python functions; more commonly used is **py-spy**:

- `py-spy dump --pid <pid>`: prints every thread's current Python call stack, the first choice when a process hangs (and `--native` shows the stacks inside C extensions and CUDA libraries too);
- `py-spy top --pid <pid>`: the most expensive functions live;
- `py-spy record -o flame.svg --pid <pid>`: samples into a flame graph.

It reads the target process's memory from outside with no restart, no code change and very little overhead, which suits production. Profiling Python code is covered in the Python handbook's [profiling and optimization](python://concurrency/performance/).

## strace: watching the system calls {#strace看系统调用}

`strace` records a process's every system call: the arguments, the return value, the duration (`-T`), a timestamp (`-tt`), with `-c` summing the counts and total time, `-f` following children and threads, and `-P` showing only the calls touching one file. Reading the same 64 MiB file in different block sizes, for instance, differs in system calls by over two orders of magnitude:

```python title="strace_reads.py"
import os
import shutil
import subprocess
import sys

with open("data.bin", "wb") as f:                   # prepare a 64 MiB file
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
    # -c sums the system call counts and -P counts only the calls touching this file
    out = subprocess.run(["strace", "-c", "-P", "data.bin", "-e", "trace=read", sys.executable, "-c", reader, str(chunk)],
                         capture_output=True, text=True).stderr
    calls = next(line.split()[3] for line in out.splitlines() if line.rstrip().endswith("read"))
    print(f"每次读 {chunk // 1024} KiB：{calls} 次 read 系统调用")
```

```text title="output"
每次读 4 KiB：16385 次 read 系统调用
每次读 1024 KiB：65 次 read 系统调用
```

(The last `read` returns 0 at the end of the file, hence the extra one.) strace suits answering "why is it stuck" (which system call it is stopped on), "which files is it opening" and "why so many system calls". It is built on ptrace and stops the process on every system call to notify strace, which makes the program many times slower, so production use has to be brief; where low overhead is needed, use `perf trace` or eBPF.

## eBPF: probes for production {#ebpf生产环境里的探针}

eBPF lets you attach a small kernel-verified program almost anywhere in the kernel (a system call, a scheduling event, a network packet, a disk I/O, a function's entry), aggregate in the kernel and hand only the results to user space, with very little overhead, which suits running in production for a long time. The ready-made tools (the bcc tools and bpftrace):

| Tool | The question it answers |
| --- | --- |
| `runqlat` | how long tasks waited in the run queue for a CPU (CPU contention) |
| `offcputime` | what tasks wait for while off the CPU (locks, I/O, sleeping), with a flame graph |
| `biolatency` | the distribution of disk I/O latency |
| `tcpretrans` | which connections are retransmitting |
| `execsnoop`, `opensnoop` | who is starting processes and opening files |

They need root or `CAP_BPF`, and inside a container they usually have to run on the host.

## One investigation: the GPU utilization will not rise {#一次排查gpu-利用率上不去}

The symptom: an inference service's decode throughput will not rise, `nvidia-smi` shows GPU-Util around 60%, and DCGM's SM activity is lower still. Following USE:

1. **the GPU is not busy, so who is holding it back?** Look at the CPU: `mpstat -P ALL 1` finds one core pinned at 100% and `pidstat -u -t 1` identifies the scheduling process's main thread, a textbook single-threaded bottleneck;
2. **what is the main thread doing?** `py-spy top --pid <the scheduling process>`: the time goes mostly to preparing the next batch's input, handling the output and detokenizing;
3. **is it being disturbed?** `pidstat -w -t 1` shows many involuntary switches on the main thread, with tokenizing threads on the same core; inside a container, also check whether the cgroup's `nr_throttled` is growing;
4. **the gaps on the GPU's timeline**: capture a stretch with Nsight Systems and the kernels have visible blanks between them, with the GPU waiting on the CPU to launch them at every step's start.

What to change: CUDA Graphs to cut the CPU's launch cost, overlap scheduling so the CPU prepares the next step while the GPU computes this one, detokenizing moved to its own process, and the scheduling thread pinned with nothing else crowding its core. These are covered in detail in [profiling an inference engine](serving://perf/profiling/), [profiling: Nsight](cuda://tools/profiling/) and [overlap scheduling](minisgl://schedule/overlap/).

!!! interview "Answering in an interview"
    Asked "latency rose in the production inference service, how do you investigate": confirm the symptom and its scope first (which metric, since when, every request or some), then go resource by resource through USE: the CPU (each core's utilization, whether one core at 100% is a single-threaded bottleneck, involuntary switches and cgroup throttling), memory (available, swapping, OOM), the disk and network (wait times, retransmissions), the GPU (DCGM's SM activity rather than `nvidia-smi`'s GPU-Util, the throttle reasons, Xid errors). Once a process is identified, use `py-spy dump` / `top` for what the Python threads are doing, perf's sampling for the C/C++ side, strace for which system call it is stuck on, and Nsight Systems for the gaps on the GPU's timeline. Finish by proposing a change and verifying it against the same metrics.

## Exercises {#练习}

**1. A high load with an idle CPU.** An inference server's load average rose from its usual 5 to 60 while `top` shows the CPU over 80% idle, and some of the service's requests hang without returning. What could it be? How do you confirm it?

??? success "Answer"
    The load counts runnable plus uninterruptible (D) tasks, and an idle CPU says nearly all the added load is tasks in D waiting on I/O in the kernel. The common causes are a network filesystem (holding the model or the logs) hung or extremely slow, a failing local disk, or waiting on write-back during memory reclamation. To confirm: `ps -eo state,pid,wchan:32,cmd | awk '$1=="D"'` lists the D tasks and the kernel function each waits in; `cat /proc/<pid>/stack` gives the full kernel stack (as root); `iostat -x 1` shows each disk's await and utilization; and `dmesg -T` shows any NFS timeouts ("server not responding") or disk errors. The hung requests are most likely reading or writing those files (writing a log synchronously, say).

**2. GPU-Util at 100% with low throughput.** `nvidia-smi` shows GPU-Util pinned at 100% while the throughput is far below expectations. What does that say? Which metrics should you read?

??? success "Answer"
    GPU-Util only says whether a kernel was running during the sampling period and not how busy the SMs are: a small kernel using very few blocks shows 100% too. Read DCGM's fine-grained metrics: SM activity (the fraction of SMs with a resident warp), SM occupancy (the resident warps as a fraction of the ceiling), Tensor Core activity and memory bandwidth utilization (DRAM Active). Low SM activity says the kernels are too small and the parallelism too low (a small batch, a small grid); high memory bandwidth with low Tensor Core activity is a bandwidth-bound workload like decode, which is as expected and calls for work on the batch, quantization and how the KV is read. Then use Nsight Compute for a particular kernel's bottleneck.

**3. perf sees only the interpreter.** A `perf record` on a Python process puts 90% of the time in `_PyEval_EvalFrameDefault`, with no sign of which Python function. What now?

??? success "Answer"
    `_PyEval_EvalFrameDefault` is CPython's bytecode loop, where every Python function runs, so perf sees only the C-level stack. The ways out: on Python 3.12 or later, start it with `python -X perf` (or the environment variable `PYTHONPERFSUPPORT=1`) and the interpreter emits a small trampoline per Python function so perf shows the names; for a running process that cannot be restarted, `py-spy record --pid <pid>` (with `--native` where time inside C extensions matters) reads Python's call stacks straight from the process's memory.

## Summary {#小结}

- [x] The USE method: check utilization, saturation and errors for the CPU, memory, the disk, the network and the GPU in turn.
- [x] Every tool reads `/proc`: high iowait is an I/O bottleneck and high steal is noisy neighbours; the load includes tasks in D; many voluntary switches mean waiting for a resource and many involuntary ones mean contending for the CPU.
- [x] `perf stat` counts (read IPC where there are hardware counters) and `perf record` samples to find the hot spots, with the flame graph's widest plateau the answer; Python needs `-X perf` or py-spy.
- [x] strace shows the system calls and where it is stuck but costs a great deal; production uses eBPF (runqlat, offcputime, biolatency).
- [x] A GPU's GPU-Util does not mean the SMs are busy, for which DCGM's SM activity is the metric; when the GPU's utilization will not rise, look first for a single-threaded bottleneck on the CPU and for gaps between kernel launches.
