# Processes, threads and scheduling

<p class="lead">An inference engine is a multi-process, multi-threaded program: the API server, tokenization, the scheduling loop and one worker per GPU all run in different processes, each with a pile of threads inside. A great many production symptoms only make sense from the operating system's point of view: why CUDA fails after a fork, why decode slows down when another thread elbows the scheduling loop aside, why too many threads in a container make things slower, why some systems would rather spin a CPU than sleep. This chapter covers what a process and a thread each own, the difference between fork and spawn, how the Linux scheduler picks the next task, what a context switch costs, and when to pin to a core.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What do threads of one process share, and what does each have of its own?
    2. Why can a child forked from a process that has already initialized CUDA not use CUDA? How do vLLM and SGLang each handle it?
    3. How does Linux's CFS scheduler decide who runs next? How does the nice value affect the CPU share?
    4. Roughly what does a context switch cost? Besides the direct cost, what indirect cost is there?
    5. When should an inference service pin threads to cores?

??? success "Answers (try it yourself first, then expand)"
    1. Shared: the address space (the code, the heap, the globals, the mmaped regions), the file descriptor table, the signal handlers, the current directory. Private: the registers and the program counter, the stack, thread-local storage, the scheduling state (the priority, which CPU) and the signal mask. In Linux a thread and a process are both "tasks" (`task_struct`) and differ only in which resources were shared at creation.
    2. fork copies only the thread that called it, so the other threads (including the CUDA driver's internal ones) do not exist in the child, while the locks they held and the state of the operations they were in the middle of are copied verbatim; a CUDA context itself cannot be inherited across a fork either. Hence PyTorch's "Cannot re-initialize CUDA in forked subprocess". vLLM starts its workers with fork by default but forces spawn when CUDA is already initialized, inside a Ray actor, with NUMA binding on, or on WSL; SGLang simply sets the start method to spawn.
    3. Every task has a virtual runtime that accumulates the real time it ran scaled by its weight (a larger weight makes the virtual time run slower); the scheduler always picks the task with the smallest virtual runtime. The nice value maps to a weight, nice 0 being 1024 with about 1.25x per step, so the CPU share is proportional to the weight. From Linux 6.6 this became EEVDF, still based on weights and virtual time, additionally accounting for latency (a virtual deadline).
    4. The direct cost is microseconds (saving and restoring registers, entering the kernel, running the scheduler, switching the page tables), measured at about 2 µs on one CPU on this machine; the indirect cost is that the caches and the TLB now hold somebody else's data and have to be warmed again, which is often larger than the direct cost. Waking a sleeping task on another CPU adds an inter-processor interrupt and the time to come out of a low-power state, about 10 µs here.
    5. Latency-sensitive threads with high CPU use: the scheduling loop (preempting it leaves the GPU idle) and busy-polling communication threads; on a multi-socket machine, pin each worker to the CPUs and memory of the NUMA node nearest its GPU (vLLM's `--numa-bind`); and inside a container, limit the thread pools so the thread count does not far exceed the usable CPUs.

## What a process and a thread each own {#进程和线程各自拥有什么}

![Figure: what a process and a thread each own](../assets/figures/process-vs-thread.svg){.aig-svg}

In the Linux kernel a process and a thread are both a "task" (`task_struct`) created by the same system call, `clone`, and differ only in which resources were shared with the parent at creation:

| Resource | Between threads of one process | Between parent and child (fork) |
| --- | --- | --- |
| the address space (code, heap, globals, mmaped regions) | shared | one each (copy-on-write) |
| the file descriptor table (open files, sockets, pipes) | shared | one each (the copies point at the same open files) |
| the registers, the stack, thread-local storage | private | private |
| the scheduling state (priority, which CPU, the affinity) | private (a new thread inherits its creator's) | private (inherited from the parent) |

A shared address space means a variable one thread writes is visible to another at once (which is why locks are needed), while a forked child changes its own copy:

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

```text title="output"
线程改完之后，主线程看到： 1
子进程改完之后，父进程看到： 1
```

The child really did change `counter["n"]` to 2, in its own copy. The parent's and the child's memory is not duplicated at the fork: both share the same physical pages, marked read-only, and whoever writes first takes a page fault and the kernel copies that page only then. That is **copy-on-write**. So forking a process holding tens of GB is fast, with the cost deferred to the actual writes.

## fork, spawn and CUDA {#forkspawn-与-cuda}

fork has a rule that is easy to overlook: **the child has only the thread that called fork**. The other threads are not copied, while the state they left behind (the locks they held, the half-written data structures) is copied verbatim. If a background thread happened to hold a lock at the moment of the fork, that lock is never released in the child:

```python title="fork_lock.py"
import os
import sys
import threading
import time
import warnings

warnings.simplefilter("ignore", DeprecationWarning)    # from Python 3.12, forking from a multithreaded process warns
lock = threading.Lock()


def worker():
    with lock:                  # a background thread holds a lock (a logging lock, the allocator's lock, some library's internal lock)
        time.sleep(1.0)


threading.Thread(target=worker, daemon=True).start()
time.sleep(0.1)                 # make sure the background thread has taken the lock
sys.stdout.flush()              # fork copies the unflushed output buffer too, so flush first or these lines print twice
pid = os.fork()
if pid == 0:                    # the child: only the thread that called fork was copied, while the lock's "held" state came across verbatim
    print("子进程等 2 秒，拿到锁了吗：", lock.acquire(timeout=2), flush=True)   # os._exit does not flush the buffers
    os._exit(0)
os.waitpid(pid, 0)
print("父进程里，后台线程 1 秒后照常释放了锁：", lock.acquire(timeout=2))
```

```text title="output"
子进程等 2 秒，拿到锁了吗： False
父进程里，后台线程 1 秒后照常释放了锁： True
```

Note that the script flushes standard output before the fork. The output buffer is part of the process's memory too, and fork copies whatever has not been written out, so without the flush the earlier lines print twice, which is the same problem as the lock.

CUDA is exactly this case: once initialized, the CUDA driver starts threads of its own and holds internal state, and the context on the GPU cannot be inherited across a fork. So forking from a process that has initialized CUDA makes the child fail as soon as it touches CUDA, and PyTorch reports "Cannot re-initialize CUDA in forked subprocess" outright. The other start method, **spawn**, does not copy the parent but starts a fresh Python interpreter, re-imports the modules and passes the arguments through pickle: no inheritance problems, at the cost of a slow start (re-importing torch alone takes seconds) and arguments that have to be picklable.

What the inference frameworks do:

- **vLLM** starts its workers with fork by default (`VLLM_WORKER_MULTIPROC_METHOD`, defaulting to `fork`, which starts fast), but `_maybe_force_spawn` in `vllm/utils/system_utils.py` forces spawn in several cases: CUDA is already initialized, it is running inside a Ray actor, NUMA binding is on (`--numa-bind` starts the children through `numactl`), or it is running on WSL (where NVML is incompatible with fork);
- **SGLang** simply calls `mp.set_start_method("spawn", force=True)` in `sglang/srt/entrypoints/engine.py`, so every subprocess is spawned.

The rule of thumb: the parent should not touch CUDA until every GPU subprocess has been created; where it must, use spawn.

## How the scheduler picks the next task {#调度器怎么挑下一个任务}

A machine usually has more runnable tasks than CPUs, and the scheduler decides what each CPU runs next. Linux's ordinary tasks (`SCHED_OTHER`) were managed from 2.6.23 to 6.5 by **CFS** (the completely fair scheduler), whose idea is simple:

- every task keeps a **virtual runtime** (vruntime): the time it really ran, accumulated scaled by its weight, with a larger weight making the virtual time run slower;
- the scheduler always picks the task with the smallest virtual runtime (the kernel keeps them in a red-black tree ordered by vruntime and takes the leftmost node);
- the weight comes from the nice value: nice 0 is 1024, with about 1.25x per step.

The result is that each task's CPU time is proportional to its weight. Below, three tasks on one CPU are simulated:

```python title="cfs.py"
# the kernel's nice-to-weight table (sched_prio_to_weight in kernel/sched/core.c): nice 0 is 1024, about 1.25x per step
WEIGHT = {-5: 3121, 0: 1024, 5: 335, 10: 110}
tasks = {"调度主循环": -5, "tokenizer": 0, "指标上报": 10}

vruntime = {name: 0.0 for name in tasks}
ran = {name: 0 for name in tasks}
for _ in range(1000):                                    # simulate 1 second, 1 ms per run
    name = min(vruntime, key=lambda n: (vruntime[n], n))  # always pick the task with the smallest virtual runtime
    ran[name] += 1
    vruntime[name] += 1024 / WEIGHT[tasks[name]]         # a larger weight makes the virtual time run slower, so it is picked more often

total = sum(WEIGHT[n] for n in tasks.values())
for name, nice in tasks.items():
    print(f"{name}（nice {nice:+d}）：运行了 {ran[name]} ms，按权重应得 {1000 * WEIGHT[nice] / total:.1f} ms")
```

```text title="output"
调度主循环（nice -5）：运行了 733 ms，按权重应得 733.5 ms
tokenizer（nice +0）：运行了 241 ms，按权重应得 240.7 ms
指标上报（nice +10）：运行了 26 ms，按权重应得 25.9 ms
```

A few additions:

- a newly woken or newly created task has its virtual runtime set near the current minimum, or a task that slept a long time would monopolize the CPU on waking;
- from Linux 6.6 ordinary tasks are scheduled by **EEVDF**: still shares by weight, but each task also has a "virtual deadline", and a shorter requested slice gives an earlier deadline, which is friendlier to latency-sensitive tasks; from 6.12, **sched_ext** also allows a scheduling policy of your own through an eBPF program;
- the real-time classes (`SCHED_FIFO`, `SCHED_RR`, set with `chrt`) always outrank ordinary tasks, and one looping real-time task starves every ordinary task on that CPU (the kernel reserves 95% of each second for real-time tasks by default as a last protection); inference services rarely use them;
- every CPU has its own run queue and load balancing migrates tasks between them. A migration means the caches have to be warmed again, which is one of the reasons for pinning.

## What a context switch costs {#上下文切换的代价}

A task that blocks (on I/O, on a lock, in a `sleep`) or is preempted when its slice runs out makes the CPU switch to another task: save the current registers, enter the kernel and run the scheduler, switch the page tables (between processes), restore the new task's registers. Below, two processes pass one byte back and forth through a pipe, once on the same CPU (two switches per round) and once on two different CPUs:

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

```text title="output (on this machine)"
两个进程在同一个 CPU 上来回：每次切换约 1.76 µs
两个进程在不同 CPU 上来回：每一轮约 10.11 µs
```

The direct cost is one or two microseconds; the indirect cost is larger and harder to measure: after the switch, the L1/L2 caches and the TLB hold the previous task's data and the new task has to read its own back in. The cross-CPU case has no switch at all and is still slower: the other CPU is asleep, waking it takes an inter-processor interrupt, and the CPU may have to come out of a low-power state (a C-state).

That explains some of the design in inference systems:

- **busy polling**: NCCL's proxy threads, RDMA's completion queues and vLLM's `MessageQueue`, which broadcasts the scheduling results through shared memory, all prefer the waiting side to spin for a while before sleeping (the `MessageQueue` polls the shared memory for a second after the last read, calling `sched_yield` to give the CPU up meanwhile, and only sleeps and waits for a ZMQ notification after a second with no new data). Each decode step is only tens of milliseconds and 10 µs of extra wakeup latency per step adds up. The cost is one CPU core;
- **the scheduling loop must not be interrupted**: vLLM's and SGLang's scheduling loops are written in Python and take a few milliseconds per step to prepare the next batch. Preempt it and the GPU waits (which is exactly what [overlap scheduling](minisgl://schedule/overlap/) sets out to solve);
- **more threads are not better**: PyTorch's CPU operators open as many threads as there are cores by default, and 8 workers on an 8-GPU machine each opening dozens leaves far more threads than CPUs, all preempting each other and running slower. The usual practice is to set `OMP_NUM_THREADS` or call `torch.set_num_threads`; inside a container, see [containers](containers.md).

## Pinning to cores {#绑核}

**CPU affinity** says which CPUs a task may run on, set with `taskset` from the command line or `sched_setaffinity` from code. The setting is inherited by threads created afterwards and by forked children:

```python title="affinity.py"
import os
import sys
import threading

os.sched_setaffinity(0, {0, 1})                  # allowed on CPUs 0 and 1 only (the equivalent of taskset -c 0,1)
print("绑核之后：", sorted(os.sched_getaffinity(0)))

seen = []
t = threading.Thread(target=lambda: seen.append(sorted(os.sched_getaffinity(0))))
t.start()
t.join()
print("之后创建的线程：", seen[0])

sys.stdout.flush()              # fork copies the unflushed output buffer too, so flush first or these lines print twice
pid = os.fork()
if pid == 0:
    print("fork 出来的子进程：", sorted(os.sched_getaffinity(0)), flush=True)
    os._exit(0)
os.waitpid(pid, 0)
```

```text title="output"
绑核之后： [0, 1]
之后创建的线程： [0, 1]
fork 出来的子进程： [0, 1]
```

Where pinning is worth it in an inference service:

- **the scheduling loop and busy-polling threads**: a core to themselves, never preempted and never migrated. In the extreme, the kernel parameter `isolcpus` takes a few cores out of ordinary scheduling altogether;
- **GPU workers on a multi-socket server**: each GPU hangs off the PCIe of one socket, so a worker process and its memory should be on the same NUMA node, or every copy crosses the sockets (see [pinned memory, DMA and NUMA](pinned-numa.md)). vLLM's `--numa-bind` does exactly this with `numactl --cpunodebind` and `--membind`;
- **several services on one machine**: a set of cores each, so they do not interfere and the latency is steadier.

## An inference engine's process structure {#推理引擎里的进程结构}

Putting all of this into an inference engine. Python has a global interpreter lock (GIL) and only one thread executes Python code in a process at a time, so CPU-bound work has to go in different processes to run in parallel:

- **vLLM V1**: the API server process (HTTP, tokenizing, detokenizing) → the EngineCore process (the scheduler, driving the executor) → one worker process per GPU. The API server and EngineCore pass msgpack-encoded messages over ZMQ, and EngineCore broadcasts each step's scheduling results to every worker through a `MessageQueue` in shared memory (`vllm/distributed/device_communicators/shm_broadcast.py`);
- **SGLang**: the TokenizerManager in the main process handles HTTP and tokenizing, one Scheduler process per tensor-parallel rank (scheduling and model execution in the same process), and a separate DetokenizerManager process for detokenizing, all communicating over ZMQ.

What this split buys: CPU work like tokenizing and detokenizing does not fight the scheduling loop for the GIL; one process crashing does not corrupt the state; and one process per GPU keeps them apart. The cost is serializing and passing data between processes, which is [interprocess communication](ipc.md)'s subject. The source details are in [vLLM V1](serving://source/vllm/) and [SGLang](serving://source/sglang/), and building it from scratch is in [messages and ZMQ](minisgl://serve/message/).

!!! interview "How to explain it"
    To explain "why is an inference engine split into several processes": start with the GIL, since only one thread runs Python code in a process at a time, so tokenizing, detokenizing, HTTP handling and the scheduling loop together slow each other down, and every extra millisecond in the scheduling loop is a millisecond the GPU waits. Then the structure: vLLM is the API server, EngineCore and one worker per card, SGLang is the TokenizerManager, one Scheduler per rank and the DetokenizerManager, communicating over ZMQ and shared memory. Then the start method: a process cannot fork after CUDA is initialized, so either spawn (SGLang) or do not touch CUDA before the fork and force spawn where necessary (vLLM). Finish with the deployment details: pin each worker to its GPU's NUMA node, and limit each process's threads so they do not far exceed the CPUs.

## Exercises {#练习}

**1. Why did the child hang.** Somebody loaded a model (with the weights already on the GPU) and then used `multiprocessing.Pool(8)` to preprocess data in parallel, where the preprocessing uses numpy only and no CUDA, and occasionally found that a child never returned. What could it be? How do you fix it?

??? success "Answer"
    `multiprocessing.Pool` uses fork by default on Linux (changing to forkserver by default in Python 3.14). The parent has many background threads at that point: the CUDA driver's, PyTorch's thread pools, a logging thread. If one of them holds a lock at the moment of the fork (the memory allocator's lock, the OpenMP pool's lock), that lock is never released in the child and the child blocks forever the moment it touches it, even without using CUDA. The fix: `multiprocessing.get_context("spawn").Pool(8)` (or forkserver), or create the pool before loading the model and before any thread exists.

**2. Shares from the nice value.** Two tasks run continuously on one CPU with nice values 0 and 5. How much CPU does each get? And with another nice 0 task added?

??? success "Answer"
    The weights are 1024 and 335, so the shares are 1024 / 1359 ≈ 75.3% and 335 / 1359 ≈ 24.7%. Adding another nice 0 task makes the total weight 1024 + 1024 + 335 = 2383, so the two nice 0 tasks get 43% each and the nice 5 task about 14%. The shares depend only on the ratio of the weights, so more tasks mean less each, and the low-priority one is squeezed harder.

**3. To busy-poll or not.** An inference engine's scheduling process sends each step's results to 8 workers, and a decode step takes about 20 ms. With a blocking read (sleeping), each wakeup costs about 10 µs; with busy polling, each worker occupies a core. Which would you choose?

??? success "Answer"
    The wakeup is 10 µs of a 20 ms step, 0.05%, which sounds small, but it sits on the critical path: the slowest worker to wake is the one that starts the computation, and the wakeup latency has a long tail (tens of microseconds or more when the CPU is in a deep sleep state), amplified every step to "the slowest of 8". The usual compromise is to spin briefly (tens of microseconds to a millisecond) and fall back to sleeping: most of the time the result arrives quickly with no wakeup cost, and an idle system does not hold a CPU. vLLM's `MessageQueue` does exactly this: it polls for a second after the last read and only then sleeps for a notification. With cores to spare (several per card), busy polling outright is acceptable too.

## Summary {#小结}

- [x] A process and a thread are both tasks in Linux and differ in what is shared: threads share the address space and the file descriptors and have their own stacks and registers; a forked child copies the parent's memory on write.
- [x] fork copies only the calling thread, so a lock another thread held stays locked forever; a process cannot fork after CUDA is initialized, so vLLM forces spawn where necessary and SGLang always spawns.
- [x] CFS shares the CPU by weight: it always runs the task with the smallest virtual runtime, with about 1.25x per nice step; from 6.6 it is EEVDF.
- [x] A context switch's direct cost is microseconds and its indirect cost (the caches, the TLB) is larger; waking across CPUs is slower still, which is why latency-sensitive paths busy-poll.
- [x] Pinning keeps a critical thread from being preempted or migrated; on a multi-socket server, pin each GPU worker to its own NUMA node; and do not let the thread count far exceed the usable CPUs.
