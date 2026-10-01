# 进程间通信：共享内存、ZMQ 与 CUDA IPC

<p class="lead">推理引擎拆成多个进程之后（为什么要拆，见[进程、线程与调度](process-thread.md)），进程之间就要不停地传东西：API 服务器把请求交给调度进程，调度进程每一步把调度结果广播给所有 GPU worker，worker 把生成的 token 送回去；RL 训练里，训练进程还要把新权重交给推理进程。每一步只有几十毫秒，传递本身的开销直接叠加在延迟上。这一章比较管道、套接字和共享内存，讲 vLLM、SGLang 都在用的 ZMQ，拆开 vLLM 用来广播调度结果的共享内存环形缓冲区，最后讲 GPU 之间不拷贝地共享显存的 CUDA IPC。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 管道、Unix 域套接字、共享内存传数据，各要经过几次拷贝？哪个最快？
    2. 推理引擎进程之间为什么常用 ZMQ，而不是直接用套接字？
    3. 用 pickle 传每一步的调度结果有什么问题？
    4. vLLM 用共享内存广播调度结果，写者怎么知道一块缓冲区可以覆盖了？更新标志的顺序为什么重要？
    5. CUDA IPC 是什么？用它共享显存要注意什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 管道和 Unix 域套接字：发送方从用户缓冲区拷进内核，接收方再从内核拷出来，两次拷贝，外加系统调用和唤醒；共享内存：两个进程映射同一块物理内存，生产者写进去，消费者原地读，一次拷贝（生产者直接在共享内存里生成数据时是零次），只需要少量同步。本机管道 1.6 GB/s（受 64 KiB 的管道缓冲区限制）、Unix 域套接字约 8 GB/s、共享内存约 11 GB/s。
    2. ZMQ 在套接字之上提供了消息边界（不用自己拆包）、常用的通信模式（PUSH/PULL、PUB/SUB、ROUTER/DEALER）、自动的连接建立和重连（先 connect 后 bind 也行）、后台 I/O 线程和发送队列，同一套代码可以走进程间的 `ipc://` 或跨机的 `tcp://`。推理引擎需要的正是"可靠地收发一条条消息"，ZMQ 省掉了大量样板代码。
    3. pickle 通用但慢：要遍历整个对象图、为每个对象写类型信息，反序列化时再逐个重建 Python 对象，一步几百个请求的调度结果本机要几百微秒，而且每一步都要做；它还能执行任意代码，不能接收不可信的数据。高频路径上的做法是用 msgpack 这类更快的格式（vLLM 用 msgspec），或者把数据压成扁平的整数数组（本机快了上百倍），大张量走共享内存。
    4. 每一块有一个写标志和每个读者一个读标志：写标志为 1、所有读标志也为 1 时（所有读者都读完了），写者才能覆盖。写完之后要先把读标志全部清零，再把写标志置 1；反过来的话，读者可能看到"写标志为 1、部分读标志为 1"的中间状态，以为自己已经读过而跳过这条消息。每次读写标志前后还要加内存屏障，保证数据和标志的可见顺序。
    5. CUDA IPC 让一个进程把自己的一块显存导出成句柄（`cudaIpcGetMemHandle`），另一个进程打开句柄（`cudaIpcOpenMemHandle`）后得到指向同一块显存的指针，不需要拷贝。注意：导出方必须保证在对方使用期间这块显存不被释放或复用；打开方看到的是同一块内存，读写要自己同步；一个进程里只能打开一次同一个句柄；PyTorch 的 `torch.multiprocessing` 传 CUDA 张量就是用它，并且要用 spawn 启动。RL 里训练进程和推理进程共卡时，靠它不拷贝地交换权重。

## 三种基本方式

操作系统提供的进程间通信方式很多，推理系统里常用的是这几种：

| 方式 | 数据怎么走 | 特点 |
| --- | --- | --- |
| 管道（pipe） | 用户缓冲区 → 内核的管道缓冲区 → 对方的用户缓冲区 | 单向、字节流；缓冲区默认只有 64 KiB，写满就阻塞 |
| Unix 域套接字 | 同上，但缓冲区更大，可以双向，可以传文件描述符 | 本机通信的首选；ZMQ 的 `ipc://` 就是它 |
| TCP 回环 | 同上，还要走一遍 TCP/IP 协议栈 | 比 Unix 域套接字慢，但代码可以直接换成跨机 |
| 共享内存 | 两个进程映射同一块物理内存 | 数据不经过内核，最快；同步要自己做 |

下面在两个进程之间传 512 MiB：

```python title="ipc_bench.py"
import os
import socket
import sys
import time
from multiprocessing import shared_memory

MiB = 1 << 20
TOTAL, CHUNK = 512 * MiB, MiB
payload = b"\1" * CHUNK


def through(send, recv_all):
    sys.stdout.flush()
    pid = os.fork()
    if pid == 0:
        recv_all()
        os._exit(0)
    t = time.perf_counter()
    for _ in range(TOTAL // CHUNK):
        send(payload)
    os.waitpid(pid, 0)
    return TOTAL / (time.perf_counter() - t) / 1e9


# 1. 管道
r, w = os.pipe()
def pipe_recv():
    got = 0
    while got < TOTAL:
        got += len(os.read(r, CHUNK))
print(f"管道：{through(lambda b: os.write(w, b), pipe_recv):.1f} GB/s")

# 2. Unix 域套接字
a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
def sock_recv():
    got = 0
    while got < TOTAL:
        got += len(b.recv(CHUNK))
print(f"Unix 域套接字：{through(a.sendall, sock_recv):.1f} GB/s")

# 3. 共享内存：生产者把数据写进共享缓冲区，消费者直接在原地读，数据不经过内核
BUF = 64 * MiB
shm = shared_memory.SharedMemory(create=True, size=BUF)
shm.buf[:BUF] = bytes(BUF)                          # 先把页都分配好，只测稳定状态
block = b"\1" * BUF
go_r, go_w = os.pipe()                              # 每一轮用管道传 1 个字节做同步，不传数据
done_r, done_w = os.pipe()
sys.stdout.flush()
pid = os.fork()
if pid == 0:
    for _ in range(TOTAL // BUF):
        os.read(go_r, 1)
        assert shm.buf[0] == 1 and shm.buf[BUF - 1] == 1   # 消费者直接读同一块物理内存
        os.write(done_w, b"k")
    os._exit(0)
t = time.perf_counter()
for _ in range(TOTAL // BUF):
    shm.buf[:BUF] = block                           # 生产者：一次内存拷贝写进共享缓冲区
    os.write(go_w, b"g")
    os.read(done_r, 1)
os.waitpid(pid, 0)
print(f"共享内存：{TOTAL / (time.perf_counter() - t) / 1e9:.1f} GB/s")
shm.close()
shm.unlink()
```

```text title="输出（本机示例）"
管道：1.6 GB/s
Unix 域套接字：7.9 GB/s
共享内存：11.0 GB/s
```

管道慢，主要是因为它的缓冲区只有 64 KiB，每写满一次就要切换到对方去读（可以用 `fcntl(F_SETPIPE_SZ)` 调大，上限由 `/proc/sys/fs/pipe-max-size` 决定，默认 1 MiB）；Unix 域套接字的缓冲区大得多，但数据仍然要在内核里走两次拷贝；共享内存只有生产者写入的那一次拷贝，消费者原地读，同步只用了一个字节的管道信号。如果生产者本来就是在共享内存里生成数据（比如直接把张量建在共享内存上），连这一次拷贝都省了。

## ZMQ：推理引擎的消息总线

直接用套接字写进程间通信，要自己处理很多事：TCP 是字节流，要自己划分消息边界；连接要先 bind 再 connect，断了要重连；发送太快时要处理缓冲区满。**ZMQ**（ZeroMQ）把这些都封装好了：

- **消息**而不是字节流：发一条就是一条，对方收到的是完整的一条；
- **通信模式**：PUSH/PULL（流水线，负载均衡地分给多个接收者）、PUB/SUB（广播）、REQ/REP（请求-应答）、ROUTER/DEALER（异步的多对多）；
- **连接管理**：连接在后台 I/O 线程里异步建立，断了自动重连，先 connect 后 bind 也能工作；
- **传输方式可换**：`ipc://`（Unix 域套接字）、`tcp://`、`inproc://`（同一进程的线程之间），代码不用改。

```python title="zmq_pushpull.py"
import os
import sys

import zmq

ADDR = f"ipc:///tmp/cs-handbook-{os.getpid()}.ipc"       # 带上 pid：两个人同时跑、或者 CI 和本地同时跑也不会撞到同一个套接字文件
N = 10000

sys.stdout.flush()
pid = os.fork()
if pid == 0:                                        # 子进程：PUSH 端，先 connect，这时对面还没有 bind
    sock = zmq.Context().socket(zmq.PUSH)
    sock.connect(ADDR)
    for i in range(N):
        sock.send(i.to_bytes(4, "little"))          # 发出去的消息先排在本地队列里，连上以后自动送达
    sock.close(linger=-1)                           # 关闭前等队列里的消息都发完
    os._exit(0)

pull = zmq.Context().socket(zmq.PULL)               # 父进程：PULL 端，晚一点才 bind
pull.bind(ADDR)
got = [int.from_bytes(pull.recv(), "little") for _ in range(N)]
os.waitpid(pid, 0)
print(f"收到 {len(got)} 条消息，顺序和内容都对：", got == list(range(N)))
print("先 connect、后 bind 也能工作：ZMQ 在后台自动建立（和重建）连接")
```

```text title="输出"
收到 10000 条消息，顺序和内容都对： True
先 connect、后 bind 也能工作：ZMQ 在后台自动建立（和重建）连接
```

vLLM 的 API 服务器和 EngineCore 进程之间、SGLang 的 TokenizerManager、Scheduler 和 DetokenizerManager 之间，都是用 ZMQ 的 `ipc://` 套接字传消息（从零实现一遍见[消息与 ZMQ](minisgl://serve/message/)）。

## 序列化：消息里装什么

ZMQ 只负责传字节，Python 对象要先序列化。最省事的是 pickle，但它在高频路径上很贵。假设一步调度的结果是 256 个请求，每个请求带着这一步的 token 和它的块号：

```python title="pickle_cost.py"
import array
import pickle
import time

# 一步调度的结果：256 个请求，每个请求这一步要算的 token（大部分在 decode，只有 1 个 token）
batch = [{"req_id": f"req-{i}", "token_ids": [1000 + i] if i % 16 else list(range(512)), "block_ids": list(range(i, i + 8))}
         for i in range(256)]

t = time.perf_counter()
for _ in range(100):
    blob = pickle.dumps(batch)
    pickle.loads(blob)
pk = (time.perf_counter() - t) / 100

# 同样的信息压成扁平的整数数组：每个请求的 token 数 + 所有 token + 块号
flat = array.array("i")
for r in batch:
    flat.append(len(r["token_ids"]))
    flat.extend(r["token_ids"])
    flat.extend(r["block_ids"])
t = time.perf_counter()
for _ in range(100):
    raw = flat.tobytes()
    array.array("i").frombytes(raw)
fl = (time.perf_counter() - t) / 100
print(f"pickle：{len(blob) / 1024:.0f} KiB，序列化 + 反序列化 {pk * 1e6:.0f} µs")
print(f"扁平数组：{len(raw) / 1024:.0f} KiB，{fl * 1e6:.0f} µs")
```

```text title="输出（本机示例）"
pickle：31 KiB，序列化 + 反序列化 376 µs
扁平数组：42 KiB，3 µs
```

pickle 要遍历整个对象图，给每个对象写类型信息，反序列化时再一个个重建 Python 对象；每一步都做一次，几百微秒就吃掉了一步 decode 的百分之几。常见的优化：

- 换更快的格式：vLLM V1 用 msgspec 的 msgpack 编码（`vllm/v1/serial_utils.py` 里的 `MsgpackEncoder`），对张量只传元数据，数据本身另外走共享内存或原始缓冲区；
- 只传增量：调度器只把这一步"变化"的部分（新加入的请求、新分配的块）发给 worker，worker 自己维护完整的状态；
- 压成扁平数组：请求用整数编号代替字符串，token 和块号拼成一个整数数组，接收方用 `numpy.frombuffer` 零拷贝地"看"出来。

另外，pickle 反序列化可以执行任意代码，只能用在完全信任的进程之间，绝不能接收外部数据。

## 共享内存环形缓冲区

vLLM 的每一步，EngineCore 都要把调度结果发给所有 worker（张量并行时每张卡一个）。它用的是共享内存里的**广播环形缓冲区**（`vllm/distributed/device_communicators/shm_broadcast.py` 里的 `ShmRingBuffer` 和在它之上的 `MessageQueue`）：一个写者、多个读者，缓冲区分成若干块，每块配一组标志：

- 1 个**写标志**：这一块是否写好了；
- 每个读者 1 个**读标志**：这个读者是否已经读过这一块。

规则：写标志为 1 且自己的读标志为 0 时，读者可以读，读完把自己的读标志置 1；写标志为 0，或者所有读标志都为 1 时，写者可以覆盖这一块。写者写完后要**先清读标志、再置写标志**：反过来的话，读者可能看到"写标志为 1、但自己的读标志还是上一轮留下的 1"，以为已经读过而跳过这条新消息。下面按同样的布局写一个简化版，一个写者广播 20 条消息给 2 个读者，缓冲区只有 4 块，写得太快时写者会等读者：

```python title="shm_ring.py"
import os
import sys
import threading
import time
from multiprocessing import shared_memory

# 按 vLLM 的 ShmRingBuffer 的布局写一个简化版：一个写者、多个读者的广播环形缓冲区
N_READER, CHUNKS, CHUNK = 2, 4, 64
META = CHUNKS * CHUNK                              # 数据区之后是元数据区：每块 1 个写标志 + 每个读者 1 个读标志
_lock = threading.Lock()


def fence():                                       # 和 vLLM 的 memory_fence 一样：拿一次锁再放掉，当作一道内存屏障
    with _lock:
        pass


class Ring:
    def __init__(self, shm):
        self.buf = shm.buf

    def _meta(self, i):
        return META + i * (1 + N_READER)

    def write(self, seq, payload):
        i, m = seq % CHUNKS, self._meta(seq % CHUNKS)
        while True:                                # 这一块要么没写过，要么所有读者都读完了，才能覆盖
            fence()
            if self.buf[m] == 0 or all(self.buf[m + 1:m + 1 + N_READER]):
                break
            time.sleep(0)
        self.buf[m] = 0                            # 先清写标志
        self.buf[i * CHUNK] = len(payload)
        self.buf[i * CHUNK + 1:i * CHUNK + 1 + len(payload)] = payload
        self.buf[m + 1:m + 1 + N_READER] = bytes(N_READER)   # 再清读者标志……
        fence()
        self.buf[m] = 1                            # ……最后才标记"写好了"。顺序反过来，读者可能看到中间状态

    def read(self, seq, reader):
        i, m = seq % CHUNKS, self._meta(seq % CHUNKS)
        while True:                                # 写好了、而且自己还没读过
            fence()
            if self.buf[m] == 1 and self.buf[m + 1 + reader] == 0:
                break
            time.sleep(0)
        n = self.buf[i * CHUNK]
        data = bytes(self.buf[i * CHUNK + 1:i * CHUNK + 1 + n])
        fence()
        self.buf[m + 1 + reader] = 1               # 读完才标记，写者这时才能覆盖这一块
        return data


shm = shared_memory.SharedMemory(create=True, size=META + CHUNKS * (1 + N_READER))
shm.buf[:shm.size] = bytes(shm.size)
N_MSG = 20
results = []
for reader in range(N_READER):
    r, w = os.pipe()
    sys.stdout.flush()
    if os.fork() == 0:                             # 读者进程：按顺序读 20 条，把结果通过管道交回父进程
        ring = Ring(shm)
        got = [ring.read(s, reader).decode() for s in range(N_MSG)]
        os.write(w, ("|".join(got)).encode())
        os._exit(0)
    results.append(r)

ring = Ring(shm)                                   # 父进程是写者：只有 4 个块，写得比读得快时会等读者
for s in range(N_MSG):
    ring.write(s, f"step {s}".encode())
for reader, r in enumerate(results):
    got = os.read(r, 4096).decode().split("|")
    os.wait()
    print(f"读者 {reader}：按顺序收到 {len(got)} 条广播，第一条 {got[0]!r}，最后一条 {got[-1]!r}，全部正确：",
          got == [f"step {s}" for s in range(N_MSG)])
shm.close()
shm.unlink()
```

```text title="输出"
读者 0：按顺序收到 20 条广播，第一条 'step 0'，最后一条 'step 19'，全部正确： True
读者 1：按顺序收到 20 条广播，第一条 'step 0'，最后一条 'step 19'，全部正确： True
```

几个细节和 vLLM 的实现一致：

- **内存屏障**：写数据和写标志之间、读标志和读数据之间要有屏障，否则 CPU 或编译器可能重排，读者看到了标志却读到旧数据。vLLM 的 `memory_fence` 用"拿一次 `threading.Lock` 再放掉"来得到一道全屏障，上面的 `fence` 照搬了这个做法；
- **等待策略**：没有数据时先忙轮询（期间 `sched_yield`），超过一段时间没有新数据才睡眠等 ZMQ 的通知（见[上下文切换的代价](process-thread.md#上下文切换的代价)）；
- **大消息**：放不进一块的消息，退回到走 ZMQ 发送；
- **跨机**：读者在别的机器上时，同样的 `MessageQueue` 接口改用 ZMQ 的 PUB/SUB。

## CUDA IPC：进程之间共享显存

共享内存解决了 CPU 内存的共享，GPU 显存也有对应的机制。**CUDA IPC**：一个进程用 `cudaIpcGetMemHandle` 把自己的一块显存导出成一个几十字节的句柄，通过任何方式（ZMQ、管道）传给另一个进程，对方用 `cudaIpcOpenMemHandle` 打开，得到指向同一块显存的指针——没有拷贝。PyTorch 的 `torch.multiprocessing` 在进程之间传 CUDA 张量时，自动做了这件事：

```python title="cuda_ipc.py" run="no"
# 需要 GPU。两个进程共享同一块显存：生产者导出 IPC 句柄，消费者打开它，拿到的是同一块显存，不拷贝
import torch
import torch.multiprocessing as mp


def consumer(q):
    t = q.get()                    # 收到的张量底下是 cudaIpcOpenMemHandle 打开的同一块显存
    t.add_(1)                      # 改的是生产者那块显存
    q.put("done")


if __name__ == "__main__":
    mp.set_start_method("spawn")   # CUDA 进程只能用 spawn（见"进程、线程与调度"一章）
    q = mp.Queue()
    x = torch.zeros(4, device="cuda")
    p = mp.Process(target=consumer, args=(q,))
    p.start()
    q.put(x)                       # 传过去的只是 IPC 句柄和元数据
    q.get()                        # 等对方用完：生产者必须保证 x 在对方使用期间一直有效
    print(x)                       # tensor([1., 1., 1., 1.], device='cuda:0')
    p.join()
```

要注意：

- **生命周期**：导出方必须保证在对方使用期间这块显存一直有效，不能释放，也不能被缓存分配器拿去给别的张量（PyTorch 对共享出去的张量做了引用计数，但跨进程的逻辑仍要自己理清）；
- **同步**：两个进程看到的是同一块内存，谁先写、谁后读要自己同步（CUDA event 也可以跨进程共享）；
- **范围**：只能在同一台机器的 GPU 之间用，而且两张卡之间要能 P2P 访问。更新的 CUDA 虚拟内存接口（`cuMemExportToShareableHandle`）可以导出成文件描述符，多节点 NVLink 系统上还有跨机的 fabric 句柄。

推理系统里的用途：RL 训练时训练进程和推理进程在同一张卡上，训练完的权重通过 IPC 句柄交给推理引擎，不经过 CPU（SGLang 的 `/update_weights_from_ipc`、vLLM 权重传输的 `ipc` 方式，见[权重热更新](serving://ops/weight-update/)）；vLLM 的 `ipc_cache` 加载方式让一个常驻的守护进程持有已经量化好的权重，引擎重启时直接映射过来。

!!! interview "面试怎么答"
    被问"vLLM 的调度进程怎么把每一步的调度结果发给多个 worker"：用共享内存里的广播环形缓冲区（`MessageQueue`），一个写者、多个读者，每块有一个写标志和每个读者一个读标志；读者看到写标志为 1、自己的读标志为 0 就读，读完置 1；所有读者都读完了写者才能覆盖；写者写完先清读标志再置写标志，顺序反了读者会漏消息；标志和数据之间要加内存屏障。等待时先忙轮询再睡眠，大消息退回到 ZMQ，跨机时换成 ZMQ 的 PUB/SUB。再对比其他方式：管道和套接字都要在内核里拷两次，共享内存只有一次；API 服务器和调度进程之间用 ZMQ 加 msgpack，因为 pickle 太慢、也不安全。

## 练习

**1. 标志的顺序。** 在上面的环形缓冲区里，如果写者写完数据后先置写标志、再清读标志，举一个读者出错的具体时序。

??? success "参考答案"
    设读者 A 在上一轮已经读过第 2 块，它的读标志是 1。写者这一轮要覆盖第 2 块：清写标志、写数据，然后先把写标志置 1——此刻读标志还没清，A 的读标志仍然是上一轮的 1。A 恰好在这时检查第 2 块，看到"写标志为 1、自己的读标志为 1"，判断"这一块我已经读过了"，于是继续等下去（或者，如果它按序号推进，会跳过这条消息），而写者随后才把读标志清零。结果 A 漏掉了这一轮的消息。先清读标志、再置写标志，就保证了只要读者看到写标志为 1，读标志一定已经是这一轮的初始值 0。

**2. 选通信方式。** 下面三个场景分别该用什么方式传数据？(a) 调度进程每一步把 200 个请求的元数据发给 8 个 worker；(b) API 服务器把生成的 token 流式地送回给 HTTP 连接所在的进程；(c) RL 训练结束一步后，把 70B 模型的新权重交给同一台机器上、用同一组 GPU 的推理引擎。

??? success "参考答案"
    (a) 一对多、每步都发、延迟敏感：共享内存广播环形缓冲区，元数据压成紧凑的格式（msgpack 或扁平数组），读者忙轮询。(b) 一条条的小消息、多对一、要可靠：ZMQ（PUSH/PULL 或 ROUTER/DEALER）走 `ipc://`，消息用 msgpack 编码；频率高时可以把多个请求的 token 攒成一批再发。(c) 数据量大（上百 GB）、源和目的都在 GPU 上：CUDA IPC 传显存句柄，推理引擎直接从训练进程的显存里读、原地拷进自己的参数（按层分批，注意生命周期和同步），完全不经过 CPU 和套接字。

**3. 管道为什么慢。** 上面的测试里管道只有 1.6 GB/s，Unix 域套接字接近 8 GB/s。怎样验证"管道缓冲区太小"这个解释？

??? success "参考答案"
    用 `fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, 1 << 20)` 把管道缓冲区调到 1 MiB 再测：如果带宽明显上升，说明瓶颈确实在缓冲区大小（每 64 KiB 就要在两个进程之间切换一次）。也可以用 `perf stat -e context-switches` 或者看 `/proc/<pid>/status` 里的 `voluntary_ctxt_switches`，比较两种方式传同样多的数据时各发生了多少次上下文切换（工具见 [Linux 性能分析工具](perf-tools.md)）。

## 小结

- [x] 管道和 Unix 域套接字要在内核里拷两次；共享内存只有生产者写入的一次，同步要自己做。本机：管道 1.6、Unix 域套接字 8、共享内存 11 GB/s。
- [x] ZMQ 提供消息边界、通信模式、自动连接管理和可替换的传输方式，是 vLLM、SGLang 进程之间的消息总线。
- [x] pickle 在高频路径上太慢且不安全：用 msgpack（vLLM 的 msgspec）、只传增量、压成扁平数组，大张量走共享内存。
- [x] vLLM 的 `ShmRingBuffer`：每块一个写标志加每个读者一个读标志，先清读标志再置写标志，读写之间加内存屏障，先轮询后睡眠。
- [x] CUDA IPC 不拷贝地共享显存：注意生命周期和同步；RL 共卡时用它交换权重。
