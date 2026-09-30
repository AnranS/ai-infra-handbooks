# I/O 多路复用：epoll 与 io_uring

<p class="lead">一个大模型 API 服务同时挂着成千上万个流式连接，每个连接要持续几十秒，一个字一个字地往外推；推理引擎内部，调度进程和 worker 之间、节点和节点之间也在不停地收发消息。一个线程怎么同时照看这么多连接？这一章从"一个连接一个线程"的问题讲起，讲 select、poll、epoll 的区别，Python asyncio 的事件循环以及为什么不能在协程里做阻塞调用，最后讲 io_uring：它和 epoll 的根本区别，以及为什么 KV Cache 存储这类系统要用它。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 阻塞 I/O、非阻塞 I/O、I/O 多路复用、异步 I/O 各是什么意思？
    2. select、poll 和 epoll 有什么区别？为什么连接很多时 epoll 快得多？
    3. 水平触发和边缘触发有什么区别？用边缘触发要注意什么？
    4. io_uring 和 epoll 的根本区别是什么？它解决了 epoll 解决不了的什么问题？
    5. Python 的 asyncio 底下用的是什么？为什么在协程里调用一个耗时的同步函数会让整个服务卡住？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 阻塞 I/O：数据没准备好就一直等，线程什么也做不了。非阻塞 I/O：没准备好就立刻返回 `EAGAIN`，要自己过一会儿再试。I/O 多路复用：用一个调用同时等很多个描述符，哪个就绪了告诉你，你再去读写（select / poll / epoll），读写本身仍然是你发起的系统调用。异步 I/O：把"读这些数据"整个交给内核，完成后通知你，数据已经在你的缓冲区里了（io_uring）。
    2. select 和 poll 每次调用都要把整个描述符集合传给内核，内核逐个检查，代价和描述符总数成正比（select 还有 1024 个的上限）。epoll 把"注册要关注的描述符"（`epoll_ctl`，一次）和"等待就绪"（`epoll_wait`）分开，内核维护一个就绪列表，设备就绪时直接把描述符挂上去，`epoll_wait` 只返回就绪的，代价和就绪的数量成正比。本机 2000 个连接里只有 1 个就绪时，poll 每次 24 µs，epoll 0.5 µs。
    3. 水平触发（默认）：只要缓冲区里还有数据，每次 `epoll_wait` 都会报告；边缘触发（`EPOLLET`）：只在状态变化时（新数据到达）报告一次。用边缘触发必须把数据一直读到 `EAGAIN` 为止，否则剩下的数据不会再触发通知，连接就"卡住"了；描述符也必须是非阻塞的。
    4. epoll 只告诉你"可以读了"，真正的读写还要你一个个系统调用去做，而且对普通文件无效（文件总是"就绪"的，读的时候照样阻塞在磁盘上）。io_uring 是真正的异步接口：进程和内核共享两个环形队列，把一批请求（读、写、发送、接收……）写进提交队列，一次系统调用（甚至零次，内核线程轮询时）提交，完成的结果出现在完成队列里。它对文件和网络都有效，批量提交省掉了大量系统调用。
    5. 在 Linux 上是 epoll（`EpollSelector`，vLLM 和 SGLang 的 HTTP 服务用的 uvloop 底下是 libuv，同样基于 epoll）。事件循环是单线程的，所有协程轮流在这个线程上运行，只有在 `await` 的地方才让出；一个同步函数跑 300 ms，这 300 ms 里所有其他请求都不会被处理，流式输出全部停顿。CPU 密集或阻塞的工作要放进线程池（`run_in_executor`）或者别的进程。

## 一个连接一个线程的问题

最直接的服务器写法是每来一个连接就开一个线程，线程里用阻塞的 `recv` / `send`。连接少时没问题，连接多了就撑不住：每个线程要一块栈（虚拟地址 8 MiB，实际占用几十 KiB 起），几千上万个线程的调度和上下文切换开销很大（见[进程、线程与调度](process-thread.md)）。大模型的流式 API 恰恰是这种负载：连接数多，每个连接都挂很久（生成几百个 token 要几十秒），大部分时间在等下一个 token。

另一种思路是：把连接设成**非阻塞**，由一个线程统一"等"——同时关注所有连接，哪个有数据了就处理哪个，处理完接着等。这就是 **I/O 多路复用**。

## select、poll 与 epoll

`select` 和 `poll` 每次调用都要把关注的全部描述符交给内核，内核逐个检查是否就绪，再把结果拷回来——代价和描述符的总数成正比，而不是和就绪的数量成正比。`epoll` 把两件事拆开：

- `epoll_ctl`：注册、修改、删除要关注的描述符，每个描述符只做一次；
- `epoll_wait`：等待，内核只返回就绪的描述符。设备（网卡、管道）有数据时，内核的回调直接把描述符挂到 epoll 的就绪列表上，不需要扫描。

2000 个连接、只有 1 个就绪时的对比：

```python title="poll_vs_epoll.py"
import os
import select
import time

pipes = [os.pipe() for _ in range(2000)]          # 2000 个都没有数据的连接（用管道代替）
p = select.poll()
ep = select.epoll()
for r, _ in pipes:
    p.register(r, select.POLLIN)
    ep.register(r, select.EPOLLIN)
os.write(pipes[-1][1], b"x")                      # 只有最后一个"连接"来了数据


def per_call(fn, n=2000):
    t = time.perf_counter()
    for _ in range(n):
        ready = fn()
    return (time.perf_counter() - t) / n * 1e6, len(ready)


for name, fn in [("poll", lambda: p.poll(0)), ("epoll", lambda: ep.poll(0))]:
    us, ready = per_call(fn)
    print(f"{name}：2000 个连接里 {ready} 个就绪，每次调用 {us:.1f} µs")
```

```text title="输出（本机示例）"
poll：2000 个连接里 1 个就绪，每次调用 23.1 µs
epoll：2000 个连接里 1 个就绪，每次调用 0.5 µs
```

下面用一个线程、一个 epoll 服务 200 个并发连接（客户端在 fork 出的子进程里）：

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
if pid == 0:                                        # 子进程当客户端：同时开 200 个连接
    socks = [socket.create_connection(("127.0.0.1", port)) for _ in range(N)]
    for i, s in enumerate(socks):
        s.sendall(f"hello {i}\n".encode())
    ok = sum(s.recv(64) == f"hello {i}\n".encode() for i, s in enumerate(socks))
    print(f"客户端：{N} 个连接，收到 {ok} 条正确的回声", flush=True)
    os._exit(0)

ep = select.epoll()                                 # 服务端：一个线程、一个 epoll
ep.register(srv.fileno(), select.EPOLLIN)
conns, echoed = {}, 0
while echoed < N:
    for fd, _ in ep.poll():                         # 只返回就绪的连接
        if fd == srv.fileno():
            while True:                             # 可能一次来了好几个新连接，接到没有为止
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

```text title="输出"
客户端：200 个连接，收到 200 条正确的回声
服务端：一个线程处理了 200 个连接，回声 200 条
```

两个细节：

- **水平触发与边缘触发**。默认是水平触发：只要接收缓冲区里还有数据，每次 `epoll_wait` 都会报告这个描述符。边缘触发（`EPOLLET`）只在"新数据到达"这个变化发生时报告一次，所以必须一直读到 `EAGAIN` 为止，否则剩下的数据再也不会触发通知。上面接受新连接时"一直 `accept` 到没有为止"也是同样的道理：一次通知可能对应好几个新连接；
- **多线程共用一个 epoll**：一个事件可能唤醒多个线程去抢（惊群），`EPOLLEXCLUSIVE` 和 `EPOLLONESHOT` 就是为此设计的。常见的做法是每个线程一个 epoll，各管一部分连接。

## asyncio：事件循环

Python 的 asyncio 把 epoll 包装成**事件循环**：协程在 `await` 一个还没就绪的 I/O 时挂起，事件循环去 `epoll_wait`，哪个描述符就绪了就恢复对应的协程。

vLLM 的 `vllm serve`（`vllm/entrypoints/cli/serve.py`）用 `uvloop.run(...)` 启动 HTTP 服务，SGLang 的 HTTP 服务（`sglang/srt/entrypoints/http_server.py`）把事件循环策略设成 uvloop：uvloop 是基于 libuv 的更快的事件循环实现，底下同样是 epoll。

事件循环是单线程的，协程只在 `await` 的地方让出。这带来了 asyncio 最重要的规则：**不要在协程里做阻塞或者耗时的同步操作**。

```python title="asyncio_demo.py"
import asyncio
import time


async def handle(i):
    await asyncio.sleep(0.1)                        # 等待期间让出事件循环，别的协程接着跑
    return i


async def main():
    loop = asyncio.get_running_loop()
    print("事件循环底下的选择器：", type(loop._selector).__name__)
    t = time.perf_counter()
    done = await asyncio.gather(*(handle(i) for i in range(1000)))
    print("1000 个各等 0.1 秒的请求，总共不到 0.5 秒：", len(done) == 1000 and time.perf_counter() - t < 0.5)

    t = time.perf_counter()
    blocker = loop.run_in_executor(None, time.sleep, 0.3)   # 阻塞的工作放进线程池
    await asyncio.gather(blocker, *(handle(i) for i in range(10)))
    print("阻塞调用放进线程池后，其他请求照常进行：", time.perf_counter() - t < 0.45)

    t = time.perf_counter()

    async def bad():
        time.sleep(0.3)                             # 在协程里直接做阻塞调用：整个事件循环停住

    await asyncio.gather(bad(), *(handle(i) for i in range(10)))
    print("协程里直接阻塞 0.3 秒，其他请求被拖到 0.4 秒以后：", time.perf_counter() - t >= 0.4)


asyncio.run(main())
```

```text title="输出"
事件循环底下的选择器： EpollSelector
1000 个各等 0.1 秒的请求，总共不到 0.5 秒： True
阻塞调用放进线程池后，其他请求照常进行： True
协程里直接阻塞 0.3 秒，其他请求被拖到 0.4 秒以后： True
```

对大模型服务来说，事件循环里的同步工作包括：分词和反分词（长输入的分词可能要几十毫秒）、大 JSON 的序列化、同步的日志和指标上报。一次卡住，所有正在流式输出的连接同时停顿，用户看到的就是"大家的输出一起卡了一下"。解决办法是放进线程池（`run_in_executor`、`asyncio.to_thread`），或者干脆放进别的进程——这也是推理引擎把分词、反分词、调度拆成多个进程的原因之一。

## io_uring：真正的异步 I/O

epoll 有两个局限：

- 它只报告"就绪"，真正的读写还要一个个系统调用去做。连接很多、每次数据都很少时，系统调用本身成了主要开销；
- 它对普通文件无效：文件在 epoll 看来总是"可读"的，真正读的时候如果数据不在页缓存里，照样阻塞在磁盘上。Linux 早期的异步文件 I/O（`io_submit`，libaio）只支持 `O_DIRECT`，限制很多。

**io_uring**（Linux 5.1 起）换了一种思路：进程和内核共享两个环形队列。进程把请求（读、写、发送、接收、打开文件、fsync……）写进**提交队列**（SQ），一次 `io_uring_enter` 系统调用把一批请求交给内核；内核完成后把结果写进**完成队列**（CQ），进程直接从共享内存里读。开启 `SQPOLL` 时，内核有一个线程专门轮询提交队列，进程提交请求连系统调用都不用。下面不用 liburing，直接调系统调用，一次提交 4 个读请求：

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

```text title="输出"
一次 io_uring_enter 提交了 4 个读请求
收到 4 个完成事件，数据全部正确：是
```

要点：提交项写好之后，用带"释放"语义的原子写更新队尾，保证内核看到新队尾时一定也能看到提交项的内容；读完成队列时用带"获取"语义的原子读取队尾——这和 C++ 里讲的[内存序](cpp://concurrency/atomics/)是同一件事，只不过另一方是内核。

io_uring 的代价是复杂度和安全性：它的攻击面很大，历史上漏洞不少，Docker 的默认 seccomp 配置和一些发行版会禁用它（Linux 6.6 起有 `kernel.io_uring_disabled` 开关），所以用它的程序要准备好在容器里退回到别的方式。再往下走一步是 **SPDK**：在用户态直接驱动 NVMe 盘，完全绕过内核，用忙轮询代替中断，延迟最低，但要独占整块盘。KV Cache 存储、分布式文件系统这类追求极限 IOPS 的系统会用 io_uring 或 SPDK。

## 推理系统里的 I/O 模型

| 组件 | I/O 方式 | 为什么 |
| --- | --- | --- |
| HTTP / OpenAI 接口服务 | asyncio + uvloop（epoll） | 连接多、每个连接挂很久，大部分时间在等下一个 token |
| 进程间消息（ZMQ） | ZMQ 自己的 I/O 线程，底下是 epoll | 应用只管收发消息，连接管理交给库（见[进程间通信](ipc.md)） |
| NCCL、RDMA | 忙轮询完成队列 | 每条消息的延迟都在关键路径上，等中断太慢（见[上下文切换的代价](process-thread.md#上下文切换的代价)） |
| KV Cache 的 SSD 层、分布式存储 | io_uring / SPDK，常配合 `O_DIRECT` | 追求高 IOPS 和稳定延迟，不要页缓存（见[一次写入如何落盘](io-stack.md)） |
| 权重加载 | 多线程顺序读、预读，或 GPUDirect Storage | 大块顺序读，瓶颈在存储带宽 |

!!! interview "面试怎么答"
    被问"epoll 和 io_uring 的区别"：epoll 是就绪通知，告诉你哪些描述符可以读写了，读写本身仍然是一个个系统调用；它用 `epoll_ctl` 注册一次、内核维护就绪列表，`epoll_wait` 的代价只和就绪数量有关，所以比 select / poll 适合大量连接；但它对普通文件无效，文件读写照样阻塞。io_uring 是完成通知：进程和内核共享提交队列和完成队列，一批请求一次系统调用提交（SQPOLL 模式下零次），内核完成后把结果放进完成队列，对文件和网络都有效。再结合推理系统：HTTP 服务用 asyncio / uvloop 这种 epoll 事件循环，事件循环里不能做分词这类耗时的同步工作；KV Cache 的 SSD 层、分布式存储这类追求 IOPS 的系统用 io_uring 甚至 SPDK。能说出 io_uring 的队尾更新要用释放 / 获取语义，是加分项。

## 练习

**1. 卡住的连接。** 某人用边缘触发的 epoll 写了一个服务，读数据的代码是 `data = conn.recv(4096); handle(data)`。测试时小请求都正常，但客户端一次发了 10 KB 的请求后，连接就卡住不动了。为什么？怎么改？

??? success "参考答案"
    边缘触发只在新数据到达时通知一次。10 KB 的请求到达后只触发一次事件，代码只读了 4096 字节，剩下的 6 KB 留在接收缓冲区里；之后没有新数据到达，就不会再有通知，这 6 KB 永远不会被读到。改法：收到事件后循环 `recv`，直到抛出 `BlockingIOError`（`EAGAIN`）为止，把读到的数据拼起来交给协议解析（一个请求可能跨多次读取，也可能一次读到多个请求）；或者改用水平触发。

**2. 大家一起卡了一下。** 一个基于 asyncio 的推理网关，平时流式输出很顺畅，但每当有人提交一个 10 万 token 的长文档请求，所有用户的输出都会同时停顿几十毫秒。最可能的原因是什么？怎么改？

??? success "参考答案"
    网关在事件循环里同步地对这个长文档做了分词（或者同步的 JSON 解析、校验），10 万 token 的分词要几十毫秒，这段时间事件循环停住，所有连接的流式输出都停了。改法：把分词放进线程池（`await asyncio.to_thread(tokenizer.encode, text)`；快的分词器库在执行时会释放 GIL，线程池里的分词不会拖住事件循环），或者放进单独的进程池；从架构上看，vLLM、SGLang 把分词、调度放在不同进程里，也是为了避免这种干扰。

**3. 为什么不用 epoll 等 GPU。** NCCL 和 RDMA 的完成通知为什么多用忙轮询，而不是像网络服务那样用 epoll 睡眠等待？什么时候应该反过来？

??? success "参考答案"
    这些操作在每一步推理或训练的关键路径上，单次延迟在微秒级。用中断加睡眠等待，每次都要付唤醒的代价（本机跨 CPU 唤醒约 10 µs，深度睡眠时更久），而且有长尾；忙轮询在数据到达的瞬间就能看到，代价是占满一个 CPU 核。GPU 服务器上 CPU 核相对充裕，用一个核换关键路径上稳定的低延迟是划算的。反过来，当等待时间长而且不确定（比如空闲的连接、低频的控制消息），或者 CPU 紧张时，应该睡眠等待；很多系统的折中是先轮询一小段时间，超时再睡眠（vLLM 的共享内存消息队列就是这样）。

## 小结

- [x] 一个连接一个线程撑不住大量长连接；非阻塞 + 多路复用让一个线程同时照看成千上万个连接。
- [x] select / poll 每次扫描全部描述符；epoll 注册一次、内核维护就绪列表，代价只和就绪数量有关（本机 24 µs 对 0.5 µs）。边缘触发要读到 `EAGAIN`。
- [x] asyncio 是单线程的 epoll 事件循环，vLLM 和 SGLang 的 HTTP 服务用 uvloop；协程里的阻塞调用会卡住所有连接，要放进线程池或别的进程。
- [x] io_uring 用共享的提交队列和完成队列实现真正的异步 I/O，一次系统调用提交一批请求，对文件也有效；容器里可能被禁用；SPDK 更进一步绕过内核。
- [x] 推理系统里：HTTP 用事件循环，NCCL / RDMA 忙轮询，KV 存储用 io_uring / SPDK。
