# I/O multiplexing: epoll and io_uring

<p class="lead">A large model's API service holds thousands of streaming connections at once, each lasting tens of seconds and pushing out a word at a time; and inside the inference engine, the scheduling process and the workers, and the nodes themselves, exchange messages continuously. How does one thread look after that many connections? This chapter starts from what is wrong with "one thread per connection", covers the differences between select, poll and epoll, Python asyncio's event loop and why a blocking call must not happen in a coroutine, and ends with io_uring: how it fundamentally differs from epoll, and why a KV cache store uses it.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What do blocking I/O, non-blocking I/O, I/O multiplexing and asynchronous I/O each mean?
    2. How do select, poll and epoll differ? Why is epoll far faster with many connections?
    3. What is the difference between level-triggered and edge-triggered? What does edge-triggered require?
    4. What is the fundamental difference between io_uring and epoll? Which problem that epoll cannot solve does it solve?
    5. What does Python's asyncio use underneath? Why does calling a slow synchronous function in a coroutine stall the whole service?

??? success "Answers (try it yourself first, then expand)"
    1. Blocking I/O: wait until the data is ready, with the thread able to do nothing. Non-blocking I/O: return `EAGAIN` at once when it is not ready, leaving you to try again later. I/O multiplexing: one call waits on many descriptors and tells you which are ready, after which you read or write them (select / poll / epoll), with the read or write still your own system call. Asynchronous I/O: hand "read this data" to the kernel whole and be notified on completion, with the data already in your buffer (io_uring).
    2. select and poll pass the whole descriptor set to the kernel on every call and the kernel checks each one, so the cost is proportional to the total number (and select caps at 1024). epoll separates "register the descriptors to watch" (`epoll_ctl`, once) from "wait for readiness" (`epoll_wait`): the kernel keeps a ready list and a device becoming ready puts the descriptor on it directly, so `epoll_wait` returns only the ready ones and costs in proportion to how many are ready. With 2000 connections and 1 ready on this machine, poll takes 24 µs per call and epoll 0.5 µs.
    3. Level-triggered (the default): as long as data remains in the buffer, every `epoll_wait` reports it; edge-triggered (`EPOLLET`) reports once when the state changes (new data arrives). Edge-triggered requires reading until `EAGAIN`, or the remaining data never triggers another notification and the connection "hangs"; and the descriptor has to be non-blocking.
    4. epoll only tells you "it can be read", and the reads and writes are still yours to make one system call at a time, and it does not work for ordinary files (a file is always "ready" and the read blocks on the disk anyway). io_uring is a genuinely asynchronous interface: the process and the kernel share two ring queues, a batch of requests (reads, writes, sends, receives) goes into the submission queue and one system call submits them (or none, when a kernel thread polls), and the results appear in the completion queue. It works for files and for the network alike, and batched submission removes a great many system calls.
    5. epoll on Linux (`EpollSelector`; uvloop, which vLLM's and SGLang's HTTP services use, is libuv underneath and likewise epoll-based). The event loop is single-threaded and every coroutine takes turns on that thread, yielding only at an `await`; a synchronous function running for 300 ms means no other request is handled for those 300 ms and every stream stalls. CPU-bound or blocking work belongs in a thread pool (`run_in_executor`) or another process.

## What is wrong with one thread per connection {#一个连接一个线程的问题}

The most direct way to write a server opens a thread per connection and uses blocking `recv` / `send` in it. With few connections this is fine, and with many it does not hold up: every thread needs a stack (8 MiB of virtual address space, tens of KiB really used), and the scheduling and context switching of thousands or tens of thousands of threads cost a great deal (see [processes, threads and scheduling](process-thread.md)). A large model's streaming API is exactly this load: many connections, each held a long time (hundreds of tokens take tens of seconds), mostly waiting for the next token.

The other idea is to make the connections **non-blocking** and have one thread do the waiting for all of them: watch every connection at once, handle whichever has data, and go back to waiting. That is **I/O multiplexing**.

## select, poll and epoll {#selectpoll-与-epoll}

`select` and `poll` hand the kernel every descriptor being watched on every call, and the kernel checks each one and copies the results back, so the cost is proportional to the total number of descriptors rather than to how many are ready. `epoll` separates the two:

- `epoll_ctl`: register, modify or remove a descriptor, once per descriptor;
- `epoll_wait`: wait, with the kernel returning only the ready ones. When a device (a network card, a pipe) has data, the kernel's callback puts the descriptor straight on epoll's ready list, with no scan.

With 2000 connections and only 1 ready:

```python title="poll_vs_epoll.py"
import os
import select
import time

pipes = [os.pipe() for _ in range(2000)]          # 2000 connections with no data (pipes standing in)
p = select.poll()
ep = select.epoll()
for r, _ in pipes:
    p.register(r, select.POLLIN)
    ep.register(r, select.EPOLLIN)
os.write(pipes[-1][1], b"x")                      # only the last "connection" has data


def per_call(fn, n=2000):
    t = time.perf_counter()
    for _ in range(n):
        ready = fn()
    return (time.perf_counter() - t) / n * 1e6, len(ready)


for name, fn in [("poll", lambda: p.poll(0)), ("epoll", lambda: ep.poll(0))]:
    us, ready = per_call(fn)
    print(f"{name}：2000 个连接里 {ready} 个就绪，每次调用 {us:.1f} µs")
```

```text title="output (on this machine)"
poll：2000 个连接里 1 个就绪，每次调用 23.1 µs
epoll：2000 个连接里 1 个就绪，每次调用 0.5 µs
```

Below, one thread and one epoll serve 200 concurrent connections (the clients running in forked children):

```python title="epoll_echo.py"
import os
import select
import socket
import sys

N = 200
srv = socket.socket()
srv.bind(("127.0.0.1", 0))
srv.listen(1024)
srv.setblocking(False)
port = srv.getsockname()[1]

sys.stdout.flush()
pid = os.fork()
if pid == 0:                                        # the children are the clients: 200 connections at once
    socks = [socket.create_connection(("127.0.0.1", port)) for _ in range(N)]
    for i, s in enumerate(socks):
        s.sendall(f"hello {i}\n".encode())
    ok = sum(s.recv(64) == f"hello {i}\n".encode() for i, s in enumerate(socks))
    print(f"客户端：{N} 个连接，收到 {ok} 条正确的回声", flush=True)
    os._exit(0)

ep = select.epoll()                                 # the server: one thread, one epoll
ep.register(srv.fileno(), select.EPOLLIN)
conns, echoed = {}, 0
while echoed < N:
    for fd, _ in ep.poll():                         # returns only the ready connections
        if fd == srv.fileno():
            while True:                             # several new connections may have arrived, so accept until there are none
                try:
                    c, _ = srv.accept()
                except BlockingIOError:
                    break
                c.setblocking(False)
                ep.register(c.fileno(), select.EPOLLIN)
                conns[c.fileno()] = c
        else:
            data = conns[fd].recv(64)
            if data:
                conns[fd].sendall(data)
                echoed += 1
os.waitpid(pid, 0)
print(f"服务端：一个线程处理了 {len(conns)} 个连接，回声 {echoed} 条")
```

```text title="output"
客户端：200 个连接，收到 200 条正确的回声
服务端：一个线程处理了 200 个连接，回声 200 条
```

Two details:

- **level-triggered and edge-triggered**. The default is level-triggered: as long as data remains in the receive buffer, every `epoll_wait` reports that descriptor. Edge-triggered (`EPOLLET`) reports once when "new data arrived" happens, so it has to read until `EAGAIN`, or the remaining data never triggers another notification. Accepting new connections "until there are none left" above follows the same reasoning: one notification may correspond to several new connections;
- **several threads sharing one epoll**: one event may wake several threads to contend (a thundering herd), which is what `EPOLLEXCLUSIVE` and `EPOLLONESHOT` are for. The usual approach is one epoll per thread, each handling part of the connections.

## asyncio: the event loop {#asyncio事件循环}

Python's asyncio wraps epoll into an **event loop**: a coroutine awaiting I/O that is not ready suspends, the event loop `epoll_wait`s, and whichever descriptor becomes ready resumes its coroutine.

vLLM's `vllm serve` (`vllm/entrypoints/cli/serve.py`) starts the HTTP service with `uvloop.run(...)` and SGLang's HTTP service (`sglang/srt/entrypoints/http_server.py`) sets the event loop policy to uvloop: uvloop is a faster event loop implementation built on libuv, which is epoll underneath all the same.

The event loop is single-threaded and a coroutine yields only at an `await`. Hence asyncio's most important rule: **do not make a blocking or slow synchronous call inside a coroutine**.

```python title="asyncio_demo.py"
import asyncio
import time


async def handle(i):
    await asyncio.sleep(0.1)                        # yields the event loop while waiting, so other coroutines run
    return i


async def main():
    loop = asyncio.get_running_loop()
    print("事件循环底下的选择器：", type(loop._selector).__name__)
    t = time.perf_counter()
    done = await asyncio.gather(*(handle(i) for i in range(1000)))
    print("1000 个各等 0.1 秒的请求，总共不到 0.5 秒：", len(done) == 1000 and time.perf_counter() - t < 0.5)

    t = time.perf_counter()
    blocker = loop.run_in_executor(None, time.sleep, 0.3)   # blocking work goes in a thread pool
    await asyncio.gather(blocker, *(handle(i) for i in range(10)))
    print("阻塞调用放进线程池后，其他请求照常进行：", time.perf_counter() - t < 0.45)

    t = time.perf_counter()

    async def bad():
        time.sleep(0.3)                             # a blocking call straight inside a coroutine: the whole event loop stops

    await asyncio.gather(bad(), *(handle(i) for i in range(10)))
    print("协程里直接阻塞 0.3 秒，其他请求被拖到 0.4 秒以后：", time.perf_counter() - t >= 0.4)


asyncio.run(main())
```

```text title="output"
事件循环底下的选择器： EpollSelector
1000 个各等 0.1 秒的请求，总共不到 0.5 秒： True
阻塞调用放进线程池后，其他请求照常进行： True
协程里直接阻塞 0.3 秒，其他请求被拖到 0.4 秒以后： True
```

For a large model service, the synchronous work that creeps into the event loop includes tokenizing and detokenizing (a long input can take tens of milliseconds), serializing a large JSON, and synchronous logging and metrics. One stall pauses every connection that is streaming at once, which the users see as "everybody's output hiccupped together". The answer is a thread pool (`run_in_executor`, `asyncio.to_thread`), or another process altogether, which is one of the reasons an inference engine splits tokenizing, detokenizing and scheduling into separate processes.

## io_uring: genuinely asynchronous I/O {#io_uring真正的异步-io}

epoll has two limits:

- it only reports readiness, and the reads and writes are still one system call each. With many connections and little data each, the system calls themselves become the main cost;
- it does not work for ordinary files: a file always looks "readable" to epoll, and the read blocks on the disk anyway when the data is not in the page cache. Linux's early asynchronous file I/O (`io_submit`, libaio) supported only `O_DIRECT` and was hedged about with limits.

**io_uring** (from Linux 5.1) takes another approach: the process and the kernel share two ring queues. The process writes requests (reads, writes, sends, receives, opening a file, fsync) into the **submission queue** (SQ) and one `io_uring_enter` system call hands a batch to the kernel; the kernel writes the results into the **completion queue** (CQ) and the process reads them straight from the shared memory. With `SQPOLL` on, a kernel thread polls the submission queue and submitting takes no system call at all. Below, without liburing, calling the system calls directly and submitting 4 reads at once:

```c title="uring_read.c"
#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

/* 不用 liburing，直接调系统调用：看清楚 io_uring 就是两个和内核共享的环形队列 */
#define BLOCK 4096
#define NREQ 4

int main(void) {
  /* 准备一个 16 KiB 的文件：第 i 个 4 KiB 块全是字母 'A' + i */
  int wfd = open("data.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  char blk[BLOCK];
  for (int i = 0; i < NREQ; i++) {
    memset(blk, 'A' + i, BLOCK);
    if (write(wfd, blk, BLOCK) != BLOCK) return 1;
  }
  close(wfd);
  int fd = open("data.bin", O_RDONLY);

  struct io_uring_params p;
  memset(&p, 0, sizeof p);
  int ring = syscall(__NR_io_uring_setup, 8, &p);
  if (ring < 0) { perror("io_uring_setup（容器可能禁用了 io_uring）"); return 1; }

  /* 把提交队列（SQ）、完成队列（CQ）和提交项数组映射进来 */
  size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
  size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
  if (cq_sz > sq_sz) sq_sz = cq_sz;                 /* 新内核的两个队列在同一块映射里（IORING_FEAT_SINGLE_MMAP） */
  char *sq = mmap(NULL, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQ_RING);
  char *cq = (p.features & IORING_FEAT_SINGLE_MMAP) ? sq
             : mmap(NULL, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_CQ_RING);
  struct io_uring_sqe *sqes = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQES);
  if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) return 1;
  unsigned *sq_tail = (unsigned *)(sq + p.sq_off.tail), *sq_mask = (unsigned *)(sq + p.sq_off.ring_mask);
  unsigned *sq_array = (unsigned *)(sq + p.sq_off.array);
  unsigned *cq_head = (unsigned *)(cq + p.cq_off.head), *cq_tail = (unsigned *)(cq + p.cq_off.tail);
  unsigned *cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
  struct io_uring_cqe *cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);

  /* 一次填好 4 个读请求：倒着读，看完成的顺序和数据是否对得上 */
  static char bufs[NREQ][BLOCK];
  unsigned tail = *sq_tail;
  for (int i = 0; i < NREQ; i++) {
    unsigned idx = tail & *sq_mask;
    struct io_uring_sqe *e = &sqes[idx];
    memset(e, 0, sizeof *e);
    e->opcode = IORING_OP_READ;
    e->fd = fd;
    e->addr = (uint64_t)(uintptr_t)bufs[i];
    e->len = BLOCK;
    e->off = (uint64_t)(NREQ - 1 - i) * BLOCK;
    e->user_data = i;                                /* 完成事件会原样带回这个值 */
    sq_array[idx] = idx;
    tail++;
  }
  __atomic_store_n(sq_tail, tail, __ATOMIC_RELEASE); /* 先写好提交项，再发布新的队尾 */

  /* 一次系统调用：提交 4 个请求，并等待 4 个完成 */
  int n = syscall(__NR_io_uring_enter, ring, NREQ, NREQ, IORING_ENTER_GETEVENTS, NULL, 0);
  printf("一次 io_uring_enter 提交了 %d 个读请求\n", n);

  int ok = 0;
  unsigned head = *cq_head, ctail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
  for (; head != ctail; head++) {
    struct io_uring_cqe *c = &cqes[head & *cq_mask];
    int i = (int)c->user_data;
    ok += c->res == BLOCK && bufs[i][0] == 'A' + (NREQ - 1 - i) && bufs[i][BLOCK - 1] == 'A' + (NREQ - 1 - i);
  }
  __atomic_store_n(cq_head, head, __ATOMIC_RELEASE); /* 告诉内核这些完成事件已经取走 */
  printf("收到 %d 个完成事件，数据全部正确：%s\n", ok, ok == NREQ ? "是" : "否");
  return 0;
}
```

```text title="output"
一次 io_uring_enter 提交了 4 个读请求
收到 4 个完成事件，数据全部正确：是
```

The key point: after writing the submission entries, update the tail with an atomic store carrying release semantics, so that the kernel seeing the new tail certainly sees the entries' contents too; and read the completion queue's tail with an acquire load. That is the same thing as [memory order](cpp://concurrency/atomics/) in C++, only with the kernel on the other side.

io_uring's costs are complexity and security: its attack surface is large, it has had its share of vulnerabilities, and Docker's default seccomp profile and some distributions disable it (from Linux 6.6 there is a `kernel.io_uring_disabled` switch), so a program using it has to be ready to fall back inside a container. One step further is **SPDK**: driving the NVMe drive from user space, bypassing the kernel entirely and replacing interrupts with busy polling, which has the lowest latency at the price of owning the whole drive. Systems chasing the limits of IOPS, a KV cache store or a distributed filesystem, use io_uring or SPDK.

## The three models side by side {#三种模型放在一起看}

![Figure: the system call counts of the three I/O models](../assets/figures/io-models.svg){.aig-svg}

## The I/O models in an inference system {#推理系统里的-io-模型}

| Component | The I/O | Why |
| --- | --- | --- |
| the HTTP / OpenAI-compatible service | asyncio + uvloop (epoll) | many connections held a long time, mostly waiting for the next token |
| messages between processes (ZMQ) | ZMQ's own I/O threads, epoll underneath | the application only sends and receives, with connection management left to the library (see [interprocess communication](ipc.md)) |
| NCCL, RDMA | busy-polled completion queues | every message's latency is on the critical path and waiting for an interrupt is too slow (see [what a context switch costs](process-thread.md#上下文切换的代价)) |
| the KV cache's SSD tier, distributed storage | io_uring / SPDK, usually with `O_DIRECT` | chasing high IOPS and steady latency, with no page cache wanted (see [how a write reaches the disk](io-stack.md)) |
| loading weights | sequential reads on several threads, prefetching, or GPUDirect Storage | large sequential reads, with the storage's bandwidth as the bottleneck |

!!! interview "Answering in an interview"
    Asked "the difference between epoll and io_uring": epoll is readiness notification, telling you which descriptors can be read or written, with the reads and writes still one system call each; `epoll_ctl` registers once, the kernel keeps a ready list, and `epoll_wait`'s cost depends only on how many are ready, which is what makes it suit many connections better than select / poll; but it does not work for ordinary files, where reads and writes block anyway. io_uring is completion notification: the process and the kernel share a submission queue and a completion queue, a batch of requests is submitted in one system call (none in SQPOLL mode), and the kernel puts the results in the completion queue, for files and the network alike. Then tie it to inference systems: the HTTP service uses an epoll event loop through asyncio / uvloop, and slow synchronous work like tokenizing must not run in it; the KV cache's SSD tier and distributed storage, chasing IOPS, use io_uring or even SPDK. Mentioning that io_uring's tail updates need release / acquire semantics earns credit.

## Exercises {#练习}

**1. The hung connection.** Somebody wrote a server with edge-triggered epoll whose read path is `data = conn.recv(4096); handle(data)`. Small requests work in testing, but after a client sends a 10 KB request the connection freezes. Why? How do you fix it?

??? success "Answer"
    Edge-triggered notifies once when new data arrives. The 10 KB request triggers one event, the code reads 4096 bytes, and the remaining 6 KB sits in the receive buffer; with no new data arriving there is no further notification and those 6 KB are never read. The fix: loop on `recv` after an event until it raises `BlockingIOError` (`EAGAIN`), and hand what was read to the protocol parser (one request may span several reads and one read may contain several requests); or switch to level-triggered.

**2. Everybody hiccupped together.** An asyncio-based inference gateway streams smoothly, but whenever someone submits a request with a 100,000-token document, every user's output pauses for tens of milliseconds at the same moment. What is the most likely cause? How do you fix it?

??? success "Answer"
    The gateway tokenized that long document synchronously inside the event loop (or parsed and validated a large JSON synchronously), and tokenizing 100,000 tokens takes tens of milliseconds, during which the event loop is stopped and every connection's streaming stops with it. The fix: move the tokenizing into a thread pool (`await asyncio.to_thread(tokenizer.encode, text)`; a fast tokenizer library releases the GIL while it runs, so tokenizing in a thread does not hold the loop up), or into a separate process pool. Architecturally, vLLM and SGLang putting tokenizing and scheduling in different processes is for exactly this kind of interference.

**3. Why not epoll the GPU.** Why do NCCL's and RDMA's completion notifications mostly busy-poll rather than sleep on epoll as a network service does? When should it be the other way round?

??? success "Answer"
    These operations sit on the critical path of every inference or training step, with latencies measured in microseconds. Interrupts and sleeping pay a wakeup every time (about 10 µs across CPUs on this machine, longer from a deep sleep state) and have a long tail; busy polling sees the data the moment it arrives, at the price of one CPU core. A GPU server has cores to spare, and one core for steady low latency on the critical path is a good trade. The other way round: when the wait is long and unpredictable (an idle connection, an infrequent control message), or the CPU is scarce, sleeping is right; many systems compromise by polling briefly and sleeping after a timeout (vLLM's shared-memory message queue does exactly this).

## Summary {#小结}

- [x] One thread per connection does not hold up with many long-lived connections; non-blocking plus multiplexing lets one thread look after thousands.
- [x] select / poll scan every descriptor on every call; epoll registers once and the kernel keeps a ready list, so the cost depends only on how many are ready (24 µs against 0.5 µs here). Edge-triggered has to read until `EAGAIN`.
- [x] asyncio is a single-threaded epoll event loop, with vLLM's and SGLang's HTTP services on uvloop; a blocking call in a coroutine stalls every connection and belongs in a thread pool or another process.
- [x] io_uring implements genuinely asynchronous I/O through a shared submission queue and completion queue, submitting a batch in one system call, and it works for files; it may be disabled inside a container; SPDK goes further and bypasses the kernel.
- [x] In an inference system: HTTP on an event loop, NCCL / RDMA busy-polling, and the KV store on io_uring / SPDK.
