# 进程、线程与调度

<p class="lead">推理引擎是一个多进程、多线程的程序：API 服务器、分词、调度循环、每张 GPU 一个 worker，各自跑在不同的进程里，进程里又有一堆线程。很多线上现象都要从操作系统的角度才解释得清：为什么 fork 之后用 CUDA 会报错，为什么调度循环被别的线程挤一下 decode 就变慢，为什么容器里开太多线程反而更慢，为什么有的系统宁可空转 CPU 也不睡眠。这一章讲进程和线程各自拥有什么、fork 与 spawn 的区别、Linux 调度器怎么挑下一个任务、一次上下文切换有多贵，以及什么时候该绑核。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 同一个进程里的线程共享哪些东西，各自独有哪些东西？
    2. 为什么在已经初始化了 CUDA 的进程里 fork 出的子进程不能再用 CUDA？vLLM 和 SGLang 各自怎么处理？
    3. Linux 的 CFS 调度器怎么决定下一个运行谁？nice 值是怎么影响 CPU 份额的？
    4. 一次上下文切换大概多少开销？除了直接开销，还有什么间接开销？
    5. 推理服务里什么情况下应该绑核？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 共享：地址空间（代码、堆、全局变量、mmap 的区域）、文件描述符表、信号处理函数、当前目录。独有：寄存器和程序计数器、栈、线程局部存储、调度状态（优先级、所在 CPU）、信号掩码。在 Linux 里线程和进程都是"任务"（`task_struct`），区别只在创建时共享了哪些资源。
    2. fork 只复制调用它的那一个线程，其他线程（包括 CUDA 驱动的内部线程）在子进程里都不存在，而它们持有的锁、正在进行的操作的状态却原样复制了过来；CUDA 上下文本身也不能跨 fork 继承。所以 PyTorch 会报 "Cannot re-initialize CUDA in forked subprocess"。vLLM 默认用 fork 启动 worker，但在 CUDA 已经初始化、在 Ray actor 里、开了 NUMA 绑定或者在 WSL 上时强制改用 spawn；SGLang 直接把启动方式设成 spawn。
    3. 每个任务有一个虚拟运行时间，实际运行时间按权重折算后累加（权重越大，虚拟时间走得越慢）；调度器总是挑虚拟运行时间最小的任务。nice 值对应权重，nice 0 是 1024，每差一级约差 1.25 倍，所以 CPU 份额按权重成比例分配。Linux 6.6 起换成了 EEVDF，仍然基于权重和虚拟时间，额外考虑了延迟（虚拟截止时间）。
    4. 直接开销是微秒量级（保存和恢复寄存器、进入内核、跑调度器、切换页表），本机测到同一个 CPU 上约 2 µs；间接开销是切换后缓存和 TLB 里都是别人的数据，要重新预热，往往比直接开销更大。跨 CPU 唤醒一个睡眠的任务还要加上核间中断和从低功耗状态醒来的时间，本机约 10 µs。
    5. 延迟敏感、CPU 占用高的线程：调度主循环（被别的线程抢占会直接让 GPU 空等）、忙轮询的通信线程；多 GPU 的机器上把每个 worker 绑到离它的 GPU 最近的 NUMA 节点的 CPU 和内存上（vLLM 的 `--numa-bind`）；容器里限制各种线程池的规模，避免线程数远多于能用的 CPU。

## 进程和线程各自拥有什么

![图：进程和线程各自拥有什么](../assets/figures/process-vs-thread.svg){.aig-svg}

在 Linux 内核里，进程和线程都是一个"任务"（`task_struct`），用同一个系统调用 `clone` 创建，区别只在于创建时和父任务共享了哪些资源：

| 资源 | 同一进程的线程之间 | 父子进程之间（fork） |
| --- | --- | --- |
| 地址空间（代码、堆、全局变量、mmap 的区域） | 共享 | 各自一份（写时复制） |
| 文件描述符表（打开的文件、套接字、管道） | 共享 | 各自一份（复制时指向同样的打开文件） |
| 寄存器、栈、线程局部存储 | 各自独有 | 各自独有 |
| 调度状态（优先级、所在 CPU、绑核设置） | 各自独有（新线程继承创建者的设置） | 各自独有（继承父进程的设置） |

共享地址空间意味着一个线程写的变量另一个线程马上能看到（也因此需要锁），而 fork 出的子进程改的是自己那一份：

```python title="share.py"
import multiprocessing as mp
import threading

counter = {"n": 0}


def bump():
    counter["n"] += 1


t = threading.Thread(target=bump)
t.start()
t.join()
print("线程改完之后，主线程看到：", counter["n"])

p = mp.get_context("fork").Process(target=bump)
p.start()
p.join()
print("子进程改完之后，父进程看到：", counter["n"])
```

```text title="输出"
线程改完之后，主线程看到： 1
子进程改完之后，父进程看到： 1
```

子进程确实把 `counter["n"]` 改成了 2，但改的是自己的副本。fork 之后父子进程的内存并没有立刻复制：两边共享同样的物理页，页被标记为只读，谁先写谁触发一次缺页、内核这时才复制那一页——这就是**写时复制**（copy-on-write）。所以 fork 一个占几十 GB 内存的进程也很快，代价被推迟到了真正写的时候。

## fork、spawn 与 CUDA

fork 有一个容易忽视的规则：**子进程里只有调用 fork 的那一个线程**。其他线程没有被复制，但它们留下的状态（持有的锁、写了一半的数据结构）都原样复制了过来。如果 fork 的那一刻某个后台线程正拿着一把锁，子进程里这把锁永远不会被释放：

```python title="fork_lock.py"
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
```

```text title="输出"
子进程等 2 秒，拿到锁了吗： False
父进程里，后台线程 1 秒后照常释放了锁： True
```

注意脚本在 fork 之前先刷新了标准输出。标准输出的缓冲区也是进程内存的一部分，fork 会把还没写出去的内容复制一份，不刷新的话前面的输出会打印两遍——和锁是同一个问题。

CUDA 正是这种情况：初始化后，CUDA 驱动会启动自己的线程、持有各种内部状态，GPU 上的上下文也不能跨 fork 继承。所以在已经初始化了 CUDA 的进程里 fork，子进程一用 CUDA 就会出错，PyTorch 会直接报 "Cannot re-initialize CUDA in forked subprocess"。另一种启动方式 **spawn** 不复制父进程，而是启动一个全新的 Python 解释器，重新导入模块，再把参数用 pickle 传过去：没有继承问题，代价是启动慢（重新导入 torch 就要几秒）、参数必须能被 pickle。

推理框架的做法：

- **vLLM** 的 worker 默认用 fork 启动（`VLLM_WORKER_MULTIPROC_METHOD`，默认 `fork`，启动快），但 `vllm/utils/system_utils.py` 里的 `_maybe_force_spawn` 会在几种情况下强制改用 spawn：CUDA 已经初始化、运行在 Ray actor 里、开了 NUMA 绑定（`--numa-bind` 要通过 `numactl` 启动子进程）、运行在 WSL 上（那里的 NVML 与 fork 不兼容）；
- **SGLang** 在 `sglang/srt/entrypoints/engine.py` 里直接 `mp.set_start_method("spawn", force=True)`，所有子进程都用 spawn。

经验规则：父进程在创建完所有 GPU 子进程之前不要碰 CUDA；如果必须先用，就用 spawn。

## 调度器怎么挑下一个任务

一台机器上可运行的任务通常比 CPU 多，调度器决定每个 CPU 下一刻跑谁。Linux 的普通任务（`SCHED_OTHER`）从 2.6.23 到 6.5 由 **CFS**（完全公平调度器）管理，核心思想很简单：

- 每个任务记一个**虚拟运行时间**（vruntime）：实际运行了多久，按权重折算后累加，权重越大，虚拟时间走得越慢；
- 调度器总是挑虚拟运行时间最小的任务来运行（内核用红黑树按虚拟运行时间排序，取最左边的节点）；
- 权重由 nice 值决定：nice 0 是 1024，每差一级约差 1.25 倍。

结果是每个任务拿到的 CPU 时间和它的权重成正比。下面模拟一个 CPU 上的三个任务：

```python title="cfs.py"
# 内核里 nice 值到权重的对照表（kernel/sched/core.c 的 sched_prio_to_weight）：nice 0 是 1024，每差一级约差 1.25 倍
WEIGHT = {-5: 3121, 0: 1024, 5: 335, 10: 110}
tasks = {"调度主循环": -5, "tokenizer": 0, "指标上报": 10}

vruntime = {name: 0.0 for name in tasks}
ran = {name: 0 for name in tasks}
for _ in range(1000):                                    # 模拟 1 秒，每次运行 1 ms
    name = min(vruntime, key=lambda n: (vruntime[n], n))  # 总是挑虚拟运行时间最小的任务
    ran[name] += 1
    vruntime[name] += 1024 / WEIGHT[tasks[name]]         # 权重越大，虚拟时间走得越慢，被挑中的次数越多

total = sum(WEIGHT[n] for n in tasks.values())
for name, nice in tasks.items():
    print(f"{name}（nice {nice:+d}）：运行了 {ran[name]} ms，按权重应得 {1000 * WEIGHT[nice] / total:.1f} ms")
```

```text title="输出"
调度主循环（nice -5）：运行了 733 ms，按权重应得 733.5 ms
tokenizer（nice +0）：运行了 241 ms，按权重应得 240.7 ms
指标上报（nice +10）：运行了 26 ms，按权重应得 25.9 ms
```

几个补充：

- 新唤醒或新创建的任务，虚拟运行时间会被设成接近当前的最小值，否则一个睡了很久的任务醒来后会独占 CPU 很久；
- Linux 6.6 起普通任务改由 **EEVDF** 调度：仍然按权重分配 CPU 份额，但每个任务还有一个"虚拟截止时间"，请求的时间片越短、截止时间越早，对延迟敏感的任务更友好；6.12 起还能用 **sched_ext** 通过 eBPF 程序实现自己的调度策略；
- 实时调度类（`SCHED_FIFO`、`SCHED_RR`，用 `chrt` 设置）总是优先于普通任务，一个死循环的实时任务会饿死同一个 CPU 上的所有普通任务（内核默认只把每秒 95% 的时间留给实时任务，作为最后的保护），推理服务里很少用；
- 每个 CPU 有自己的运行队列，负载均衡会在 CPU 之间迁移任务。迁移意味着缓存要重新预热，这也是绑核的理由之一。

## 上下文切换的代价

一个任务阻塞（等 I/O、等锁、`sleep`）或者时间片用完被抢占，CPU 就要切换到另一个任务：保存当前任务的寄存器、进入内核跑调度器、切换页表（不同进程之间）、恢复新任务的寄存器。下面让两个进程通过管道来回传一个字节，一次在同一个 CPU 上（每一轮是两次切换），一次在两个不同的 CPU 上：

```c title="ctxsw.c"
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void pin(int cpu) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  sched_setaffinity(0, sizeof s, &s);
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* 两个进程通过两根管道来回传 1 个字节：每一轮 A 写、B 读后回写、A 读，各自阻塞等待对方 */
static double pingpong(int cpu_a, int cpu_b, int n) {
  int ab[2], ba[2];
  char c = 0;
  if (pipe(ab) || pipe(ba)) return -1;
  pid_t pid = fork();
  if (pid == 0) {
    pin(cpu_b);
    for (int i = 0; i < n; i++)
      if (read(ab[0], &c, 1) != 1 || write(ba[1], &c, 1) != 1) _exit(1);
    _exit(0);
  }
  pin(cpu_a);
  double t0 = now();
  for (int i = 0; i < n; i++)
    if (write(ab[1], &c, 1) != 1 || read(ba[0], &c, 1) != 1) return -1;
  double t = now() - t0;
  waitpid(pid, NULL, 0);
  return t / n;
}

int main(void) {
  const int n = 200000;
  double same = pingpong(0, 0, n), cross = pingpong(0, 1, n);
  /* 同一个 CPU 上：每一轮是两次上下文切换（A→B、B→A） */
  printf("两个进程在同一个 CPU 上来回：每次切换约 %.2f µs\n", same / 2 * 1e6);
  /* 不同 CPU 上：没有切换，但每次都要唤醒另一个核上睡着的进程 */
  printf("两个进程在不同 CPU 上来回：每一轮约 %.2f µs\n", cross * 1e6);
  return 0;
}
```

```text title="输出（本机示例）"
两个进程在同一个 CPU 上来回：每次切换约 1.76 µs
两个进程在不同 CPU 上来回：每一轮约 10.11 µs
```

直接开销是一两微秒；间接开销更大也更难测：切换之后 L1/L2 缓存和 TLB 里装的都是上一个任务的数据，新任务要重新把自己的数据读进来。跨 CPU 的那次没有切换，却更慢——对方的 CPU 在睡眠，唤醒它要发一个核间中断，CPU 还可能要从低功耗状态（C-state）醒来。

这解释了推理系统里的一些设计：

- **忙轮询**：NCCL 的代理线程、RDMA 的完成队列、vLLM 在共享内存里广播调度结果的 `MessageQueue`，都倾向于让等待的一方先空转检查一段时间，而不是马上睡眠（`MessageQueue` 在上一次读到数据后的 1 秒内一直轮询共享内存，期间调用 `sched_yield` 让出 CPU，超过 1 秒没有新数据才改为睡眠、等 ZMQ 的通知）——每一步 decode 只有几十毫秒，每次多花 10 µs 的唤醒延迟积累起来就很可观。代价是占满一个 CPU 核；
- **调度循环怕被打断**：vLLM、SGLang 的调度主循环是 Python 写的，每一步要花几毫秒准备下一批输入。它一旦被别的线程抢占，GPU 就在空等（这正是[重叠调度](minisgl://schedule/overlap/)要解决的问题）；
- **线程不是越多越好**：PyTorch 的 CPU 算子默认开和 CPU 核数一样多的线程，一台 8 卡机器上 8 个 worker 各开几十个线程，远多于 CPU 数，大家互相抢占，反而更慢。常见做法是设置 `OMP_NUM_THREADS` 或调用 `torch.set_num_threads`，容器里的情况见[容器](containers.md)一章。

## 绑核

**CPU 亲和性**（affinity）规定一个任务只能在哪些 CPU 上运行，命令行用 `taskset`，代码里用 `sched_setaffinity`。设置会被之后创建的线程和 fork 出的子进程继承：

```python title="affinity.py"
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
```

```text title="输出"
绑核之后： [0, 1]
之后创建的线程： [0, 1]
fork 出来的子进程： [0, 1]
```

推理服务里值得绑核的场景：

- **调度主循环、忙轮询的线程**：独占一个核，不被别的任务抢占，也不在 CPU 之间迁移。极端情况下可以用内核参数 `isolcpus` 把几个核从普通调度里隔离出来；
- **多路服务器上的 GPU worker**：每张 GPU 挂在某一个 CPU 插槽的 PCIe 下，worker 进程和它的内存应该在同一个 NUMA 节点上，否则每次拷贝都要跨插槽（详见[锁页内存、DMA 与 NUMA](pinned-numa.md)）。vLLM 的 `--numa-bind` 就是用 `numactl --cpunodebind` 和 `--membind` 把每个 worker 绑到它的 GPU 所在的节点；
- **同一台机器上跑多个服务**：各自绑一组核，互不干扰，延迟更稳定。

## 推理引擎里的进程结构

把前面的内容放到推理引擎里看。Python 有全局解释器锁（GIL），同一个进程里同一时刻只有一个线程在执行 Python 代码，所以 CPU 密集的工作要放进不同的进程才能真正并行：

- **vLLM V1**：API 服务器进程（HTTP、分词、反分词）→ EngineCore 进程（调度器，驱动执行器）→ 每张 GPU 一个 worker 进程。API 服务器和 EngineCore 之间用 ZMQ 传递 msgpack 编码的消息，EngineCore 把每一步的调度结果通过共享内存里的 `MessageQueue`（`vllm/distributed/device_communicators/shm_broadcast.py`）广播给所有 worker；
- **SGLang**：主进程里的 TokenizerManager 负责 HTTP 和分词，每个张量并行的 rank 一个 Scheduler 进程（调度和模型执行在同一个进程里），DetokenizerManager 单独一个进程做反分词，彼此之间用 ZMQ 通信。

这样拆分的好处：分词和反分词这些 CPU 工作不和调度主循环抢 GIL；一个进程崩溃不至于把状态弄乱；每个 GPU 一个进程，互不干扰。代价是进程间要序列化和传递数据，这是[进程间通信](ipc.md)一章的内容。源码的细节见 [vLLM V1](serving://source/vllm/) 和 [SGLang](serving://source/sglang/) 两章，从零实现一遍见[消息与 ZMQ](minisgl://serve/message/)。

!!! interview "怎么讲清楚"
    讲"推理引擎为什么要拆成多个进程"：先讲 GIL，Python 同一进程里同一时刻只有一个线程在跑 Python 代码，分词、反分词、HTTP 处理和调度主循环放在一起会互相拖慢，而调度主循环每一步多花一毫秒，GPU 就空等一毫秒。再讲结构：vLLM 是 API 服务器、EngineCore、每卡一个 worker，SGLang 是 TokenizerManager、每个 rank 一个 Scheduler、DetokenizerManager，之间用 ZMQ 和共享内存通信。然后讲启动方式：CUDA 初始化后不能 fork，所以用 spawn（SGLang）或者在 fork 之前不碰 CUDA、必要时强制 spawn（vLLM）。最后补充部署时的细节：worker 绑到 GPU 所在的 NUMA 节点，限制每个进程的线程数，避免线程数远超 CPU。

## 练习

**1. 为什么子进程卡住了。** 某人在加载完模型（已经把权重放到 GPU 上）之后，用 `multiprocessing.Pool(8)` 并行做数据预处理，预处理函数里只用 numpy，没有用 CUDA，却偶尔发现有的子进程永远不返回。可能的原因是什么？怎么改？

??? success "参考答案"
    `multiprocessing.Pool` 在 Linux 上默认用 fork（Python 3.14 起默认改为 forkserver）。父进程此时有很多后台线程：CUDA 驱动的线程、PyTorch 的线程池、日志线程等。fork 的一瞬间如果其中某个线程拿着一把锁（比如内存分配器的锁、OpenMP 线程池的锁），子进程里这把锁永远不会被释放，子进程一碰到它就永远阻塞——即使子进程不用 CUDA。改法：用 `multiprocessing.get_context("spawn").Pool(8)`（或 forkserver），或者在加载模型、创建任何线程之前先把进程池建好。

**2. nice 值的份额。** 一个 CPU 上有两个一直在跑的任务，nice 值分别是 0 和 5，各分到多少 CPU？如果再加一个 nice 0 的任务呢？

??? success "参考答案"
    权重分别是 1024 和 335，份额 1024 / 1359 ≈ 75.3%、335 / 1359 ≈ 24.7%。再加一个 nice 0 的任务，总权重是 1024 + 1024 + 335 = 2383，两个 nice 0 的任务各 43%，nice 5 的任务约 14%。份额只看权重之比，所以任务越多，每个任务分得越少，低优先级任务被压得更厉害。

**3. 要不要忙轮询。** 一个推理引擎的调度进程每一步把调度结果发给 8 个 worker，每一步 decode 约 20 ms。如果 worker 用阻塞读（睡眠等待）接收，每次唤醒约 10 µs；如果忙轮询，每个 worker 占满一个 CPU 核。你会怎么选？

??? success "参考答案"
    唤醒延迟占一步的 10 µs / 20 ms = 0.05%，看起来不大，但它串在关键路径上：调度结果发出后，最慢的那个 worker 醒来才能开始算，而且唤醒延迟有长尾（CPU 在深度睡眠状态时可能到几十微秒甚至更长），每一步都会被放大到"8 个里最慢的"。常见的折中是先忙轮询一小段时间（比如几十微秒到一毫秒），等不到再退回睡眠：大多数时候结果很快就到，不用付唤醒的代价；空闲时也不会一直占着 CPU。vLLM 的 `MessageQueue` 就是这样：上一次读到数据后的 1 秒内一直轮询，之后才睡眠等通知。如果机器的 CPU 核很充裕（每卡能分到好几个核），直接忙轮询也可以接受。

## 小结

- [x] Linux 里进程和线程都是任务，区别在共享了什么：线程共享地址空间和文件描述符，各有栈和寄存器；fork 出的子进程写时复制父进程的内存。
- [x] fork 只复制调用它的线程，别的线程持有的锁会永远锁住；CUDA 初始化之后不能 fork，vLLM 必要时强制 spawn，SGLang 一律 spawn。
- [x] CFS 按权重分配 CPU：总是运行虚拟运行时间最小的任务，nice 每差一级权重约差 1.25 倍；6.6 起换成 EEVDF。
- [x] 上下文切换直接开销是微秒量级，间接开销（缓存、TLB）更大；跨 CPU 唤醒更慢，所以延迟敏感的路径常用忙轮询。
- [x] 绑核让关键线程不被抢占、不迁移；多路服务器上把 GPU worker 绑到它所在的 NUMA 节点；线程数不要远超可用的 CPU。
