# 容器：namespace 与 cgroup

<p class="lead">推理服务几乎都跑在容器里，由 Kubernetes 调度。容器不是虚拟机：它就是宿主机上的一组普通进程，只是被 namespace 限制了"看得到什么"，被 cgroup 限制了"能用多少"。很多只在线上出现的怪象都来自这两样东西：设了 CPU 上限之后延迟反而出现毛刺，<code>os.cpu_count()</code> 返回宿主机的 128 个核导致开了一大堆线程，<code>/dev/shm</code> 只有 64 MiB 让 NCCL 和 PyTorch 报错，页缓存算进了内存用量。这一章讲清 namespace 和 cgroup，再讲 GPU 和 RDMA 网卡是怎么进容器的。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 容器和虚拟机的根本区别是什么？这对推理服务有什么影响？
    2. 常用的 namespace 有哪几种，各隔离什么？
    3. cgroup 限制 CPU 有"份额"和"配额"两种方式，区别是什么？为什么配额会让延迟出现毛刺？
    4. 为什么容器里 `os.cpu_count()` 会误导人？应该按什么来设置线程数？
    5. 容器里的 `/dev/shm` 默认多大？太小会出什么问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 虚拟机有自己的内核，容器和宿主机共用同一个内核，只是用 namespace 隔离视图、用 cgroup 限制资源、用镜像提供文件系统。影响：内核特性（比如 io_uring 是否可用、调度器版本）由宿主机决定；GPU 驱动是宿主机的，容器里只带用户态的 CUDA 运行时，两者版本要兼容；隔离比虚拟机弱，邻居的负载会通过共享的内核、缓存和内存带宽影响你。
    2. mnt（挂载点和文件系统视图）、pid（进程编号，容器里的第一个进程是 1 号）、net（网卡、路由、端口）、ipc（System V IPC 和 POSIX 消息队列）、uts（主机名）、user（用户和权限的映射，普通用户也能在里面当"root"）、cgroup（看到的 cgroup 层级）、time（时钟偏移）。
    3. 份额（v1 的 `cpu.shares`，v2 的 `cpu.weight`，Kubernetes 的 requests）只在 CPU 紧张时按比例分配，空闲的 CPU 可以随便用；配额（v1 的 `cfs_quota_us` / `cfs_period_us`，v2 的 `cpu.max`，Kubernetes 的 limits）是硬上限：每个周期（默认 100 ms）里整个容器最多用多少 CPU 时间，用完就被挂起到下一个周期开始。多线程程序一次突发很容易在周期前半段就用光配额，剩下的时间全部干等，本来 40 ms 的工作被拖到 115 ms 甚至 300 ms。
    4. `os.cpu_count()` 读的是整台机器的 CPU 数，不管 cgroup 的配额，也不管 cpuset 绑了哪些核。很多库按它来开线程池（PyTorch 的算子线程、OpenMP、Rust 实现的分词器），在一个只给了 4 个 CPU 的容器里开出几十上百个线程，互相抢占，还会更快地耗尽配额。应该按实际可用的 CPU 设置：`len(os.sched_getaffinity(0))` 反映 cpuset，再和配额（`cpu.max` 里的 quota / period）取小，显式设置 `OMP_NUM_THREADS`、`torch.set_num_threads`、`RAYON_NUM_THREADS` 等。
    5. Docker 默认 64 MiB。PyTorch 的 DataLoader 多进程、NCCL 的机内传输、vLLM 和 SGLang 的共享内存消息队列都在 `/dev/shm` 里建共享内存文件，空间不够时要么报错，要么在访问时收到 SIGBUS（Bus error）崩溃。解决：`docker run --shm-size=16g` 或 `--ipc=host`，Kubernetes 里挂一个 `medium: Memory` 的 emptyDir 到 `/dev/shm`；注意 `/dev/shm` 里的数据计入容器的内存用量。

## 容器是什么

![图：容器 = namespace + cgroup + 挂进来的设备](../assets/figures/namespaces-cgroups.svg){.aig-svg}

一个容器 = 宿主机内核上的一组进程 + 几种 namespace（隔离视图）+ cgroup（限制资源）+ 一个由镜像层叠加出来的根文件系统（overlayfs）+ 一些安全限制（去掉的特权、seccomp 过滤的系统调用）。它没有自己的内核，这一点决定了很多事：

- **内核特性看宿主机**：比如上一章的 io_uring，宿主机内核太旧或者被 seccomp 过滤了，容器里就用不了；
- **GPU 驱动看宿主机**：容器镜像里只带用户态的 CUDA 运行时（`libcudart`）和各种库，内核驱动和 `libcuda.so` 来自宿主机。镜像里 CUDA 的版本不能比驱动支持的高（或者要借助 CUDA 的前向兼容包）；
- **隔离比虚拟机弱**：同一台机器上的容器共享内核、CPU 缓存和内存带宽，邻居的突发负载会让你的延迟抖动。

## namespace：看得到什么

| namespace | 隔离的东西 | 和推理服务有关的点 |
| --- | --- | --- |
| mnt | 挂载点、文件系统视图 | 镜像、模型卷、`/dev/shm` 的挂载 |
| pid | 进程编号 | 容器里的主进程是 1 号进程：它要负责回收僵尸子进程，默认也不响应 SIGTERM，推理服务常用 tini 之类的小程序当 1 号进程，把信号转给引擎、优雅退出 |
| net | 网卡、路由、端口 | 默认每个 Pod 一个网络命名空间；RDMA 网卡要额外暴露，追求性能时也有用宿主机网络的 |
| ipc | System V 共享内存、消息队列 | 两个容器之间要共享内存，就要共享 ipc 命名空间或挂同一个 `/dev/shm` |
| uts | 主机名 | 分布式启动时用主机名识别节点 |
| user | 用户和权限的映射 | 普通用户可以在自己的命名空间里当"root"，这是无特权容器的基础 |
| cgroup、time | 看到的 cgroup 层级、时钟偏移 | |

namespace 是用 `clone` 或 `unshare` 系统调用创建的。下面不需要 root：在子进程里新建一个用户命名空间和一个网络命名空间，里面就只剩一块回环网卡：

```python title="ns_demo.py"
import os
import socket
import sys


def ns(kind):
    return os.readlink(f"/proc/self/ns/{kind}")    # 形如 net:[4026531840]，方括号里是命名空间的编号


parent_net = ns("net")
r, w = os.pipe()
sys.stdout.flush()
if os.fork() == 0:
    same = ns("net") == parent_net
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)  # 新建一个用户命名空间（不需要 root）和一个网络命名空间
    nics = [name for _, name in socket.if_nameindex()]
    os.write(w, f"{same}|{ns('net') != parent_net}|{nics}".encode())
    os._exit(0)
os.close(w)                                        # 父进程关掉写端：子进程万一挂了，下面的 read 才会读到 EOF 而不是一直卡着
os.wait()
same, differ, nics = os.read(r, 4096).decode().split("|")
print("fork 出的子进程默认和父进程在同一个网络命名空间：", same)
print("unshare 之后换了一个新的网络命名空间：", differ)
print("新的网络命名空间里只有这些网卡：", nics)
```

```text title="输出"
fork 出的子进程默认和父进程在同一个网络命名空间： True
unshare 之后换了一个新的网络命名空间： True
新的网络命名空间里只有这些网卡： ['lo']
```

（`os.unshare` 是 Python 3.12 才有的；更老的版本用 `ctypes` 调 libc 的 `unshare`。有的系统出于安全考虑禁止普通用户创建用户命名空间，这时 `unshare` 会报权限错误。）

## cgroup：能用多少

cgroup 把进程分组，按组限制和统计资源。有 v1 和 v2 两代接口（v1 每种资源一棵树，v2 统一成一棵树），新的发行版和 Kubernetes 都在转向 v2。常用的控制器：

- **cpu**：份额（v1 `cpu.shares`，v2 `cpu.weight`）和配额（v1 `cpu.cfs_quota_us` / `cpu.cfs_period_us`，v2 `cpu.max`）；
- **cpuset**：只能在哪些 CPU、哪些 NUMA 节点上运行和分配内存。Kubernetes 的 CPU Manager 在 static 策略下，会给整数个 CPU 的 Guaranteed Pod 分配独占的核，就是靠它；
- **memory**：内存上限（v2 `memory.max`），超过就在容器内部触发 OOM killer，Pod 显示 `OOMKilled`；
- **pids**、**io**、**hugetlb**：进程数、磁盘 I/O、大页的限制。

看看本机当前进程所在的 cgroup：

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

```text title="输出（本机示例）"
cgroup 版本： v1
CPU 配额： 不限
内存上限： 不限
os.cpu_count() = 32，可调度的 CPU = 32，按配额真正能用的约 32 个
/dev/shm 大小：31.3 GiB
```

### CPU 配额与节流

Kubernetes 里 `resources.requests.cpu` 变成份额，`resources.limits.cpu` 变成配额。份额只在争抢时起作用，配额是硬上限，按周期（默认 100 ms）结算：整个容器在一个周期里用的 CPU 时间加起来超过配额，所有线程被挂起到下一个周期开始。多线程程序的一次突发，很容易在周期的前几十毫秒就用光配额：

```python title="throttle_sim.py"
# CFS 带宽控制：每个周期（100 ms）里，整个容器最多用 quota 毫秒的 CPU 时间，用完就被挂起到下一个周期
PERIOD, QUOTA = 100, 200             # 相当于 limits.cpu = 2


def finish_time(threads, work_ms):
    """threads 个线程同时开工，每个要 work_ms 毫秒 CPU（机器上核足够多），返回全部完成的时刻"""
    left = [work_ms] * threads
    t = 0.0
    while True:
        budget = QUOTA                       # 新周期开始，配额重置
        start = t
        while budget > 1e-9 and any(left):
            running = [i for i, x in enumerate(left) if x > 0]
            step = min(min(left[i] for i in running), budget / len(running))   # 大家并行跑，直到有人跑完或配额用光
            for i in running:
                left[i] -= step
            budget -= step * len(running)
            t += step
        if not any(x > 1e-9 for x in left):
            return t
        t = start + PERIOD                   # 配额用光：整个容器被挂起，直到下一个周期开始


for threads in (1, 4, 8, 16):
    print(f"{threads} 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 {finish_time(threads, 40):.0f} ms 完成")
```

```text title="输出"
1 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 40 ms 完成
4 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 40 ms 完成
8 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 115 ms 完成
16 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 302 ms 完成
```

8 个线程各算 40 ms，本来 40 ms 就能完成，但前 25 ms 就用光了 200 ms 的配额，剩下的 75 ms 全在等下一个周期。对推理服务来说，这就是 P99 延迟的毛刺：平均 CPU 使用率远没到上限，延迟却时不时跳上去。怎么确认：看 cgroup 的 `cpu.stat` 里 `nr_throttled`、`throttled_usec`（v1 是 `throttled_time`）在不在增长。怎么处理：

- 延迟敏感的服务不设 CPU limits，只设 requests；或者用整数个 CPU 加 static CPU Manager，拿到独占的核；
- 按配额限制线程数（下一节），别让突发一下子把配额吃光；
- 必要时调大周期或者开启 burst（v2 的 `cpu.max.burst`），允许短时间借用。

### CPU 数量的坑

`os.cpu_count()` 返回的是整台机器的逻辑 CPU 数，它不知道 cgroup 配额，也不知道 cpuset。`len(os.sched_getaffinity(0))` 反映 cpuset 的限制（Python 3.13 起的 `os.process_cpu_count()` 同样如此），但仍然不知道配额。很多库按 CPU 数开线程：

- PyTorch 的算子内并行默认按物理核数开线程；
- OpenMP、MKL 默认用所有 CPU；
- Rust 实现的分词器（HF tokenizers）用 rayon 线程池，默认也是全部 CPU。

一个配额只有 4 个 CPU 的容器跑在 128 核的机器上，这些库会开出上百个线程，互相抢占，还更快地耗尽配额。做法是算出真正可用的 CPU 数（亲和性和配额取小），再显式设置 `OMP_NUM_THREADS`、`torch.set_num_threads(...)`、`RAYON_NUM_THREADS`、`TOKENIZERS_PARALLELISM` 等。

### 内存上限

容器的内存用量里除了进程的匿名内存，还包括它产生的**页缓存**和 **tmpfs**（包括 `/dev/shm`）里的文件。页缓存是可回收的，用量接近上限时内核会先回收它，所以"内存用量快满了"不一定是泄漏；但 `/dev/shm` 里的共享内存文件回收不了，写满了就会触发容器内的 OOM。v2 还有一个 `memory.high`：超过它不会被杀，但会被强制回收、降速，可以当作预警线。

## /dev/shm 与共享内存

Docker 默认只给容器 64 MiB 的 `/dev/shm`，而推理和训练框架大量使用它：

- PyTorch 的 DataLoader 多进程之间通过共享内存传递张量；
- NCCL 在同一台机器的 GPU 之间不能直接 P2P 时，经过 `/dev/shm` 里的缓冲区中转；
- vLLM、SGLang 用 `multiprocessing.shared_memory` 建共享内存消息队列（见[进程间通信](ipc.md)），这些共享内存就是 `/dev/shm` 里的文件。

空间不够时，要么创建时报错（NCCL 会打印 "Error while creating shared memory segment"），要么在写入时收到 SIGBUS，进程以 "Bus error" 崩溃——后者尤其难查。解决办法：`docker run --shm-size=16g` 或者 `--ipc=host`；Kubernetes 里给 `/dev/shm` 挂一个 `emptyDir`，`medium: Memory`，设置合适的 `sizeLimit`。

## GPU 和 RDMA 网卡怎么进容器

容器默认看不到 GPU。NVIDIA Container Toolkit 在创建容器时做三件事：把 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 等设备节点放进容器、把宿主机的驱动库（`libcuda.so`、`libnvidia-ml.so` 等）挂载进来、设置 cgroup 的设备访问权限。`docker run --gpus` 或环境变量 `NVIDIA_VISIBLE_DEVICES` 决定给哪几张卡，新版本用 CDI（Container Device Interface）描述这些设备。在 Kubernetes 里，NVIDIA 的 device plugin 把每个节点上的 GPU 注册成 `nvidia.com/gpu` 资源，Pod 按这个申请；A100 以后的 GPU 还可以用 MIG 切成几个独立的小 GPU 分给不同的 Pod。

RDMA 网卡同理，需要对应的 device plugin 把 `/dev/infiniband/*` 暴露进来，还要给容器 `IPC_LOCK` 能力、放开 `memlock` 限制（RDMA 注册内存要锁页，见[锁页内存](pinned-numa.md)）。多机推理和 PD 分离的 Pod 通常还要按网络拓扑调度（同一个交换机下），这部分见[生产部署与运维](serving://ops/deploy/)。

!!! interview "怎么讲清楚"
    讲"推理服务迁到 Kubernetes 之后 P99 延迟变差了，你会查什么"：先查 CPU 节流：看 cgroup `cpu.stat` 的 `nr_throttled` 和 `throttled_usec`，limits 是按 100 ms 周期结算的硬配额，多线程一次突发就能把配额用光，剩下的时间整个容器干等；延迟敏感的服务去掉 CPU limits 或者用 static CPU Manager 拿独占核。再查线程数：`os.cpu_count()` 返回宿主机的核数，PyTorch、OpenMP、分词器按它开了几十上百个线程，要按真正可用的 CPU 显式设置。然后查 NUMA 和 cpuset：进程和它的 GPU 是否在同一个节点。还有 `/dev/shm` 是否太小（NCCL、共享内存队列）、内存上限是否太紧导致频繁回收页缓存，以及同机邻居的干扰。

## 练习

**1. 节流计算。** 一个容器 `limits.cpu = 4`（周期 100 ms），推理服务收到一个请求时要做 20 ms × 16 个线程的 CPU 工作（比如并行分词）。在没有其他负载的情况下，这个请求的 CPU 部分多久完成？如果把线程数限制成 4 呢？

??? success "参考答案"
    总工作量 320 ms CPU，配额每周期 400 ms。16 个线程时，只要机器的核足够多，16 个线程并行跑，第 20 ms 就全部完成，用掉 320 ms 配额，没有超过 400 ms，不会被节流，20 ms 完成。4 个线程时，每个线程要做 80 ms 的工作，4 个线程并行 80 ms 完成，同样用 320 ms 配额，80 ms 完成。这个例子里线程多反而快——节流只在"一个周期里的总用量超过配额"时才发生。如果同一周期里还有别的请求也在用 CPU，16 个线程的突发更容易把配额用光，一旦被节流就要等到下一个周期；所以限制线程数的主要目的是让用量更平稳、延迟更可预测，而不是单个请求更快。

**2. 分词器开了多少线程。** 一个推理网关的容器 `limits.cpu = 4`，跑在 96 核的机器上，用 Rust 实现的分词器做批量分词。`top` 显示这个进程有 100 多个线程，CPU 使用率不高，但延迟毛刺很多。原因和改法？

??? success "参考答案"
    分词器的 rayon 线程池默认按机器的 CPU 数开线程，96 核就开 96 个，加上 PyTorch、OpenMP 等其他线程池，远多于容器能用的 4 个 CPU。一次批量分词就有几十个线程同时抢 CPU，瞬间把一个周期的配额用光，所有线程被挂起到下一个周期，延迟出现几十毫秒的毛刺；平均 CPU 使用率却不高，因为大部分时间在等。改法：按真正可用的 CPU 数设置 `RAYON_NUM_THREADS=4`（或者 `TOKENIZERS_PARALLELISM=false` 关掉内部并行，由上层控制并发）、`OMP_NUM_THREADS`、`torch.set_num_threads`；同时考虑去掉 CPU limits 只保留 requests。验证：看 `cpu.stat` 的 `nr_throttled` 是否随毛刺增长。

**3. Bus error。** 一个多卡推理服务在本机 Docker 里测试正常，部署到 Kubernetes 后启动时偶尔以 "Bus error" 崩溃，没有 Python 调用栈。可能的原因？怎么确认？

??? success "参考答案"
    很可能是 `/dev/shm` 太小：Kubernetes 默认没有给 Pod 挂大的 `/dev/shm`（容器运行时默认 64 MiB），而本机测试时 Docker 可能用了 `--ipc=host` 或 `--shm-size`。共享内存文件是按需分配页的，创建时不一定报错，写到超出容量的页时进程收到 SIGBUS，直接崩溃，Python 来不及打印调用栈。确认：在 Pod 里 `df -h /dev/shm` 看容量和用量；用 `NCCL_DEBUG=INFO` 看 NCCL 是否在建共享内存段；复现时用 `strace -f -e trace=none` 或 `dmesg` 看 SIGBUS 的来源。修复：挂载 `medium: Memory` 的 emptyDir 到 `/dev/shm`，并把这部分内存算进 Pod 的内存上限。

## 小结

- [x] 容器 = 宿主机内核上的进程 + namespace + cgroup + 镜像文件系统；内核特性和 GPU 驱动都来自宿主机。
- [x] namespace 管"看得到什么"：mnt、pid、net、ipc、uts、user、cgroup、time；容器的主进程是 1 号进程，要负责回收子进程、转发信号。
- [x] cgroup 管"能用多少"：CPU 份额与配额、cpuset、内存上限；配额按 100 ms 周期结算，突发会导致节流和延迟毛刺，看 `cpu.stat` 确认。
- [x] `os.cpu_count()` 不知道配额和 cpuset，线程池要按真正可用的 CPU 显式设置；页缓存和 `/dev/shm` 计入容器内存。
- [x] `/dev/shm` 默认 64 MiB，太小会让 NCCL、DataLoader、共享内存队列出错或 SIGBUS；GPU 和 RDMA 网卡通过 Container Toolkit、device plugin 进容器。
