# Containers: namespaces and cgroups

<p class="lead">Inference services nearly all run in containers scheduled by Kubernetes. A container is not a virtual machine: it is a group of ordinary processes on the host, limited by namespaces in what they can see and by cgroups in what they can use. A great many production-only oddities come from these two things: latency spikes after a CPU limit is set, <code>os.cpu_count()</code> returning the host's 128 cores and opening a pile of threads, a 64 MiB <code>/dev/shm</code> breaking NCCL and PyTorch, the page cache counting towards the memory use. This chapter makes namespaces and cgroups clear and then covers how a GPU and an RDMA network card get into a container.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What fundamentally separates a container from a virtual machine? What does that mean for an inference service?
    2. Which namespaces are in common use and what does each isolate?
    3. cgroups limit the CPU by "shares" and by "quota". How do they differ? Why does a quota cause latency spikes?
    4. Why is `os.cpu_count()` misleading in a container? What should the thread count be based on?
    5. How large is `/dev/shm` in a container by default? What breaks when it is too small?

??? success "Answers (try it yourself first, then expand)"
    1. A virtual machine has a kernel of its own, while a container shares the host's and merely isolates its view with namespaces, limits its resources with cgroups and gets its filesystem from an image. The consequences: the kernel's features (whether io_uring is available, the scheduler's version) come from the host; the GPU driver is the host's and the container carries only the userspace CUDA runtime, so the versions have to be compatible; and the isolation is weaker than a virtual machine's, so a neighbour's load reaches you through the shared kernel, caches and memory bandwidth.
    2. mnt (mount points and the filesystem view), pid (process numbers, with the container's first process as PID 1), net (network cards, routes, ports), ipc (System V IPC and POSIX message queues), uts (the hostname), user (the mapping of users and privileges, so an ordinary user can be "root" inside), cgroup (the cgroup hierarchy seen) and time (a clock offset).
    3. Shares (v1's `cpu.shares`, v2's `cpu.weight`, Kubernetes's requests) only apportion the CPU when it is contended, and an idle CPU may be used freely; a quota (v1's `cfs_quota_us` / `cfs_period_us`, v2's `cpu.max`, Kubernetes's limits) is a hard ceiling: the container as a whole may use so much CPU time per period (100 ms by default), and once it is used up every thread is suspended until the next period. A multithreaded burst easily exhausts the quota in the first half of a period and then waits out the rest, so 40 ms of work stretches to 115 ms or even 300.
    4. `os.cpu_count()` reads the whole machine's CPU count regardless of the cgroup's quota and of which cores the cpuset allows. Many libraries open their thread pools from it (PyTorch's operator threads, OpenMP, the Rust tokenizer), so a container given 4 CPUs opens dozens or hundreds of threads that preempt each other and exhaust the quota faster. Base it on the CPUs really available: `len(os.sched_getaffinity(0))` reflects the cpuset, take the smaller of that and the quota (quota / period from `cpu.max`), and set `OMP_NUM_THREADS`, `torch.set_num_threads`, `RAYON_NUM_THREADS` and the rest explicitly.
    5. 64 MiB by default in Docker. PyTorch's multi-process DataLoader, NCCL's within-machine transfers and vLLM's and SGLang's shared-memory message queues all create shared memory files in `/dev/shm`, and too little space either raises an error or kills the process with a SIGBUS (Bus error) on access. The fix: `docker run --shm-size=16g` or `--ipc=host`, and in Kubernetes an emptyDir with `medium: Memory` mounted at `/dev/shm`; note that what is in `/dev/shm` counts towards the container's memory.

## What a container is {#容器是什么}

![Figure: a container = namespaces + cgroups + the devices mounted in](../assets/figures/namespaces-cgroups.svg){.aig-svg}

A container = a group of processes on the host's kernel + several namespaces (isolating the view) + cgroups (limiting the resources) + a root filesystem stacked from the image's layers (overlayfs) + some security restrictions (dropped capabilities, system calls filtered by seccomp). It has no kernel of its own, which decides a great deal:

- **the kernel's features come from the host**: io_uring from the last chapter, for instance, is unavailable when the host's kernel is too old or seccomp filters it;
- **the GPU driver comes from the host**: the image carries only the userspace CUDA runtime (`libcudart`) and the libraries, while the kernel driver and `libcuda.so` come from the host. The image's CUDA must not be newer than the driver supports (short of CUDA's forward compatibility package);
- **the isolation is weaker than a virtual machine's**: containers on one machine share the kernel, the CPU caches and the memory bandwidth, so a neighbour's burst jitters your latency.

## Namespaces: what can be seen {#namespace看得到什么}

| Namespace | What it isolates | What it means for an inference service |
| --- | --- | --- |
| mnt | mount points, the filesystem view | the image, the model volume, `/dev/shm`'s mount |
| pid | process numbers | the container's main process is PID 1: it has to reap zombie children and by default ignores SIGTERM, so inference services commonly run a small program like tini as PID 1 to forward the signals to the engine for a graceful exit |
| net | network cards, routes, ports | one network namespace per Pod by default; an RDMA card has to be exposed separately, and host networking is used where performance demands it |
| ipc | System V shared memory, message queues | two containers sharing memory have to share the ipc namespace or mount the same `/dev/shm` |
| uts | the hostname | a distributed launch identifies nodes by hostname |
| user | the mapping of users and privileges | an ordinary user can be "root" in their own namespace, which is the basis of unprivileged containers |
| cgroup, time | the cgroup hierarchy seen, a clock offset | |

Namespaces are created by the `clone` or `unshare` system calls. The following needs no root: a child process creates a new user namespace and a new network namespace, where only a loopback interface remains:

```python title="ns_demo.py"
import os
import socket
import sys


def ns(kind):
    return os.readlink(f"/proc/self/ns/{kind}")    # of the form net:[4026531840], with the namespace's number in the brackets


parent_net = ns("net")
r, w = os.pipe()
sys.stdout.flush()
if os.fork() == 0:
    same = ns("net") == parent_net
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)  # create a new user namespace (no root needed) and a new network namespace
    nics = [name for _, name in socket.if_nameindex()]
    os.write(w, f"{same}|{ns('net') != parent_net}|{nics}".encode())
    os._exit(0)
os.wait()
same, differ, nics = os.read(r, 4096).decode().split("|")
print("fork 出的子进程默认和父进程在同一个网络命名空间：", same)
print("unshare 之后换了一个新的网络命名空间：", differ)
print("新的网络命名空间里只有这些网卡：", nics)
```

```text title="output"
fork 出的子进程默认和父进程在同一个网络命名空间： True
unshare 之后换了一个新的网络命名空间： True
新的网络命名空间里只有这些网卡： ['lo']
```

(Some systems forbid unprivileged user namespaces for security, where `unshare` raises a permission error.)

## cgroups: how much can be used {#cgroup能用多少}

A cgroup groups processes and limits and accounts for their resources by group. There are two generations of interface (v1 with a tree per resource, v2 with one unified tree), and newer distributions and Kubernetes are moving to v2. The controllers in common use:

- **cpu**: shares (v1's `cpu.shares`, v2's `cpu.weight`) and a quota (v1's `cpu.cfs_quota_us` / `cpu.cfs_period_us`, v2's `cpu.max`);
- **cpuset**: which CPUs and NUMA nodes it may run on and allocate memory from. Kubernetes's CPU Manager under the static policy gives a Guaranteed Pod with an integral CPU count exclusive cores through this;
- **memory**: the memory ceiling (v2's `memory.max`), past which the OOM killer fires inside the container and the Pod shows `OOMKilled`;
- **pids**, **io**, **hugetlb**: limits on processes, disk I/O and huge pages.

Looking at the cgroup this process is in:

```python title="cgroup_limits.py"
import os


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


v2 = os.path.exists("/sys/fs/cgroup/cgroup.controllers")
print("cgroup 版本：", "v2" if v2 else "v1")
if v2:
    rel = open("/proc/self/cgroup").read().strip().split("::")[-1]
    base = "/sys/fs/cgroup" + rel
    quota, period = (read(f"{base}/cpu.max") or "max 100000").split()
    mem = read(f"{base}/memory.max")
else:
    quota, period = read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us"), read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    quota = "max" if quota in (None, "-1") else quota
    mem = read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
limit = None if quota == "max" else int(quota) / int(period)
print("CPU 配额：", "不限" if limit is None else f"{limit:g} 个 CPU")
print("内存上限：", "不限" if mem in (None, "max") or int(mem) > 1 << 60 else f"{int(mem) / 2**30:.1f} GiB")

affinity = len(os.sched_getaffinity(0))
usable = affinity if limit is None else min(affinity, limit)
print(f"os.cpu_count() = {os.cpu_count()}，可调度的 CPU = {affinity}，按配额真正能用的约 {usable:g} 个")

st = os.statvfs("/dev/shm")
print(f"/dev/shm 大小：{st.f_blocks * st.f_frsize / 2**30:.1f} GiB")
```

```text title="output (on this machine)"
cgroup 版本： v1
CPU 配额： 不限
内存上限： 不限
os.cpu_count() = 32，可调度的 CPU = 32，按配额真正能用的约 32 个
/dev/shm 大小：31.3 GiB
```

### The CPU quota and throttling {#cpu-配额与节流}

In Kubernetes, `resources.requests.cpu` becomes shares and `resources.limits.cpu` becomes a quota. Shares matter only under contention; a quota is a hard ceiling settled per period (100 ms by default): once the container's threads have used more CPU time than the quota within a period, they are all suspended until the next period starts. A multithreaded burst easily exhausts the quota in the period's first few tens of milliseconds:

```python title="throttle_sim.py"
# CFS bandwidth control: per period (100 ms) the whole container may use quota milliseconds of CPU time, and is suspended until the next period once it is gone
PERIOD, QUOTA = 100, 200             # the equivalent of limits.cpu = 2


def finish_time(threads, work_ms):
    """threads 个线程同时开工，每个要 work_ms 毫秒 CPU（机器上核足够多），返回全部完成的时刻"""
    left = [work_ms] * threads
    t = 0.0
    while True:
        budget = QUOTA                       # a new period starts and the quota resets
        start = t
        while budget > 1e-9 and any(left):
            running = [i for i, x in enumerate(left) if x > 0]
            step = min(min(left[i] for i in running), budget / len(running))   # they run in parallel until one finishes or the quota runs out
            for i in running:
                left[i] -= step
            budget -= step * len(running)
            t += step
        if not any(x > 1e-9 for x in left):
            return t
        t = start + PERIOD                   # the quota is gone: the whole container is suspended until the next period


for threads in (1, 4, 8, 16):
    print(f"{threads} 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 {finish_time(threads, 40):.0f} ms 完成")
```

```text title="output"
1 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 40 ms 完成
4 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 40 ms 完成
8 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 115 ms 完成
16 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 302 ms 完成
```

Eight threads computing 40 ms each would finish in 40 ms, but the 200 ms quota is gone in the first 25 ms and the remaining 75 ms is spent waiting for the next period. For an inference service, that is the P99 spike: the average CPU use is nowhere near the limit and the latency jumps now and then. To confirm it, see whether `nr_throttled` and `throttled_usec` in the cgroup's `cpu.stat` (`throttled_time` on v1) are growing. What to do:

- leave CPU limits off a latency-sensitive service and set only requests; or use an integral CPU count with the static CPU Manager for exclusive cores;
- limit the thread count by the quota (the next section), so a burst cannot eat it all at once;
- where necessary, lengthen the period or enable burst (v2's `cpu.max.burst`) to allow brief borrowing.

### The CPU count trap {#cpu-数量的坑}

`os.cpu_count()` returns the whole machine's logical CPU count, knowing nothing of the cgroup's quota or of the cpuset. `len(os.sched_getaffinity(0))` reflects the cpuset (as does Python 3.13's `os.process_cpu_count()`), and still knows nothing of the quota. Many libraries open threads by the CPU count:

- PyTorch's intra-operator parallelism defaults to the physical core count;
- OpenMP and MKL default to every CPU;
- the Rust tokenizer (HF tokenizers) uses a rayon thread pool, likewise every CPU by default.

A container with a quota of 4 CPUs on a 128-core machine has these libraries open hundreds of threads that preempt each other and exhaust the quota faster. The practice is to work out the CPUs really available (the smaller of the affinity and the quota) and set `OMP_NUM_THREADS`, `torch.set_num_threads(...)`, `RAYON_NUM_THREADS` and `TOKENIZERS_PARALLELISM` explicitly.

### The memory ceiling {#内存上限}

A container's memory use includes, besides the processes' anonymous memory, the **page cache** it generates and the files in **tmpfs** (including `/dev/shm`). The page cache is reclaimable and the kernel reclaims it as the use approaches the ceiling, so "memory nearly full" is not necessarily a leak; but the shared memory files in `/dev/shm` cannot be reclaimed and filling it triggers an OOM inside the container. v2 also has `memory.high`: exceeding it kills nothing but forces reclamation and slows the process down, which works as an early warning.

## /dev/shm and shared memory {#devshm-与共享内存}

Docker gives a container only 64 MiB of `/dev/shm` by default, while inference and training frameworks use it heavily:

- PyTorch's DataLoader passes tensors between processes through shared memory;
- NCCL stages through buffers in `/dev/shm` when GPUs in one machine cannot reach each other by P2P;
- vLLM and SGLang build their shared-memory message queues with `multiprocessing.shared_memory` (see [interprocess communication](ipc.md)), and that shared memory is files in `/dev/shm`.

Without the space, it either fails at creation (NCCL prints "Error while creating shared memory segment") or receives a SIGBUS on a write and dies with "Bus error", the latter being particularly hard to diagnose. The fix: `docker run --shm-size=16g` or `--ipc=host`; and in Kubernetes, an `emptyDir` with `medium: Memory` at `/dev/shm` and a suitable `sizeLimit`.

## How a GPU and an RDMA card get into a container {#gpu-和-rdma-网卡怎么进容器}

A container sees no GPU by default. The NVIDIA Container Toolkit does three things when the container is created: puts the device nodes (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) into the container, mounts the host's driver libraries (`libcuda.so`, `libnvidia-ml.so`) in, and sets the cgroup's device permissions. `docker run --gpus` or the environment variable `NVIDIA_VISIBLE_DEVICES` decides which cards, and newer versions describe these devices through CDI (the Container Device Interface). In Kubernetes, NVIDIA's device plugin registers each node's GPUs as the resource `nvidia.com/gpu` for Pods to request; and from A100 onwards a GPU can also be cut into several independent small GPUs with MIG for different Pods.

An RDMA card is the same: a device plugin exposes `/dev/infiniband/*`, and the container also needs the `IPC_LOCK` capability and a raised `memlock` limit (RDMA's memory registration pins pages, see [pinned memory](pinned-numa.md)). Multi-machine inference and prefill-decode split Pods usually also have to be scheduled by network topology (under one switch), which is in [production deployment and operations](serving://ops/deploy/).

!!! interview "Answering in an interview"
    Asked "P99 latency got worse after moving the inference service to Kubernetes, what would you look at": CPU throttling first, reading `nr_throttled` and `throttled_usec` from the cgroup's `cpu.stat`, since limits are a hard quota settled per 100 ms period and one multithreaded burst exhausts it, leaving the whole container waiting; a latency-sensitive service drops its CPU limits or takes exclusive cores through the static CPU Manager. Then the thread count: `os.cpu_count()` returns the host's cores and PyTorch, OpenMP and the tokenizer open dozens or hundreds of threads from it, which have to be set explicitly from the CPUs really available. Then NUMA and the cpuset: whether the process and its GPU are on the same node. Then whether `/dev/shm` is too small (NCCL, the shared-memory queues), whether the memory ceiling is so tight that the page cache is reclaimed constantly, and whether a neighbour on the machine is interfering.

## Exercises {#练习}

**1. The throttling arithmetic.** A container has `limits.cpu = 4` (a 100 ms period), and an inference service doing 20 ms × 16 threads of CPU work on a request (parallel tokenizing, say). With nothing else running, how long does that CPU work take? And with the threads limited to 4?

??? success "Answer"
    The total work is 320 ms of CPU and the quota is 400 ms per period. With 16 threads, as long as the machine has cores enough, all 16 run in parallel and finish at 20 ms having used 320 ms of the quota, which is under 400, so nothing is throttled and it takes 20 ms. With 4 threads, each does 80 ms and the four finish in 80 ms, likewise using 320 ms of quota, in 80 ms. More threads is faster here: throttling only happens when a period's total use exceeds the quota. With other requests using the CPU in the same period, a 16-thread burst exhausts the quota far more easily and then waits for the next period; so limiting the threads is mainly about steadier use and more predictable latency, not about one request being faster.

**2. How many threads did the tokenizer open.** An inference gateway's container has `limits.cpu = 4`, runs on a 96-core machine, and tokenizes in batches with the Rust tokenizer. `top` shows the process with over 100 threads, the CPU use is low, and the latency spikes often. Why, and what do you change?

??? success "Answer"
    The tokenizer's rayon pool opens threads by the machine's CPU count, 96 of them, and PyTorch's and OpenMP's pools add more, far beyond the 4 CPUs the container may use. One batch of tokenizing has dozens of threads contending at once, exhausting a period's quota instantly, and every thread is suspended until the next period, which gives tens of milliseconds of spike; the average CPU use stays low because most of the time is spent waiting. The fix: set `RAYON_NUM_THREADS=4` by the CPUs really available (or `TOKENIZERS_PARALLELISM=false` to turn the internal parallelism off and control the concurrency above), plus `OMP_NUM_THREADS` and `torch.set_num_threads`; and consider dropping the CPU limits and keeping only requests. To verify: see whether `nr_throttled` in `cpu.stat` grows with the spikes.

**3. Bus error.** A multi-GPU inference service tests fine in local Docker and occasionally dies at startup with "Bus error" after deployment to Kubernetes, with no Python traceback. What could it be? How do you confirm it?

??? success "Answer"
    Most likely `/dev/shm` is too small: Kubernetes does not give a Pod a large `/dev/shm` by default (the container runtime's default is 64 MiB), while the local Docker test may have used `--ipc=host` or `--shm-size`. A shared memory file's pages are allocated on demand, so creation may succeed, and writing past the capacity delivers a SIGBUS that kills the process outright before Python can print a traceback. To confirm: `df -h /dev/shm` inside the Pod for the capacity and the use; `NCCL_DEBUG=INFO` to see whether NCCL is creating shared memory segments; and `strace -f -e trace=none` or `dmesg` for the SIGBUS's origin while reproducing. The fix: mount an emptyDir with `medium: Memory` at `/dev/shm`, and count that memory in the Pod's limit.

## Summary {#小结}

- [x] A container = processes on the host's kernel + namespaces + cgroups + the image's filesystem; the kernel's features and the GPU driver both come from the host.
- [x] Namespaces govern "what can be seen": mnt, pid, net, ipc, uts, user, cgroup, time; the container's main process is PID 1 and has to reap children and forward signals.
- [x] cgroups govern "how much can be used": CPU shares and a quota, the cpuset, the memory ceiling; the quota is settled per 100 ms period, so a burst brings throttling and latency spikes, confirmed through `cpu.stat`.
- [x] `os.cpu_count()` knows nothing of the quota or the cpuset, so thread pools have to be set explicitly from the CPUs really available; the page cache and `/dev/shm` count towards the container's memory.
- [x] `/dev/shm` is 64 MiB by default, and too little breaks NCCL, the DataLoader and the shared-memory queues or kills them with a SIGBUS; GPUs and RDMA cards enter a container through the Container Toolkit and device plugins.
