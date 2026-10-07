# Interprocess communication: shared memory, ZMQ and CUDA IPC

<p class="lead">Once an inference engine is split into several processes (why, in [processes, threads and scheduling](process-thread.md)), those processes pass things back and forth continuously: the API server hands requests to the scheduling process, the scheduler broadcasts each step's results to every GPU worker, the workers send the generated tokens back; and in RL training, the training process also hands new weights to the inference process. A step is only tens of milliseconds, so the cost of the passing adds straight to the latency. This chapter compares pipes, sockets and shared memory, covers the ZMQ that vLLM and SGLang both use, takes apart the shared-memory ring buffer vLLM broadcasts its scheduling results through, and ends with CUDA IPC, which shares device memory between processes without a copy.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How many copies does data take through a pipe, a Unix domain socket and shared memory? Which is fastest?
    2. Why do an inference engine's processes usually use ZMQ rather than sockets directly?
    3. What is wrong with passing each step's scheduling results through pickle?
    4. In vLLM's shared-memory broadcast, how does the writer know a buffer may be overwritten? Why does the order of the flag updates matter?
    5. What is CUDA IPC? What does sharing device memory with it require?

??? success "Answers (try it yourself first, then expand)"
    1. A pipe and a Unix domain socket: the sender copies from its buffer into the kernel and the receiver copies out again, two copies plus the system calls and the wakeup; shared memory: both processes map the same physical memory, the producer writes and the consumer reads in place, one copy (none when the producer generates the data in the shared memory directly) with only a little synchronization. On this machine: a pipe 1.6 GB/s (limited by the 64 KiB pipe buffer), a Unix domain socket about 8 GB/s, shared memory about 11 GB/s.
    2. Over sockets, ZMQ adds message boundaries (no framing of your own), the common patterns (PUSH/PULL, PUB/SUB, ROUTER/DEALER), automatic connection setup and reconnection (connecting before the bind works), background I/O threads and send queues, and one body of code runs over `ipc://` within a machine or `tcp://` across machines. What an inference engine needs is exactly "send and receive messages reliably", and ZMQ removes a great deal of boilerplate.
    3. pickle is general and slow: it walks the whole object graph writing type information for every object and rebuilds every Python object on the way back, so a step's results for a few hundred requests take hundreds of microseconds on this machine, every step; and it can execute arbitrary code, so it must never receive untrusted data. On a hot path the practice is a faster format like msgpack (vLLM uses msgspec), or flattening the data into integer arrays (over a hundred times faster here), with large tensors going through shared memory.
    4. Each block has one write flag and one read flag per reader: the writer may overwrite when the write flag is 1 and every read flag is 1 (every reader has read it). After writing, it has to clear the read flags first and set the write flag second; the other order lets a reader see the intermediate state of "the write flag is 1 and my read flag is 1" and skip the message thinking it has read it. Memory barriers go around the flag accesses too, so the data and the flags become visible in the right order.
    5. CUDA IPC lets a process export a block of its device memory as a handle (`cudaIpcGetMemHandle`), and another process that opens it (`cudaIpcOpenMemHandle`) gets a pointer to the same memory with no copy. Watch out: the exporter has to keep that memory alive and unreused while the other side is using it; both see the same memory, so reads and writes need synchronizing; one process can open a given handle only once; PyTorch's `torch.multiprocessing` uses it to pass CUDA tensors and requires the spawn start method. When RL training and inference share a card, this is how the weights are exchanged without a copy.

## The three basic ways {#三种基本方式}

The operating system offers many ways for processes to communicate, and these are the ones inference systems use:

| Way | How the data moves | Notes |
| --- | --- | --- |
| a pipe | the user buffer -> the kernel's pipe buffer -> the other side's user buffer | one direction, a byte stream; the buffer is 64 KiB by default and a full one blocks |
| a Unix domain socket | the same, with a larger buffer, bidirectional, and able to pass file descriptors | the first choice within a machine; ZMQ's `ipc://` is this |
| TCP over loopback | the same, plus a trip through the TCP/IP stack | slower than a Unix domain socket, but the code becomes cross-machine unchanged |
| shared memory | both processes map the same physical memory | the data never enters the kernel, the fastest; the synchronization is yours |

Below, 512 MiB passed between two processes:

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


# 1. a pipe
r, w = os.pipe()
def pipe_recv():
    got = 0
    while got < TOTAL:
        got += len(os.read(r, CHUNK))
print(f"管道：{through(lambda b: os.write(w, b), pipe_recv):.1f} GB/s")

# 2. a Unix domain socket
a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
def sock_recv():
    got = 0
    while got < TOTAL:
        got += len(b.recv(CHUNK))
print(f"Unix 域套接字：{through(a.sendall, sock_recv):.1f} GB/s")

# 3. shared memory: the producer writes into the shared buffer and the consumer reads in place, with the data never entering the kernel
BUF = 64 * MiB
shm = shared_memory.SharedMemory(create=True, size=BUF)
shm.buf[:BUF] = bytes(BUF)                          # allocate every page first, so only the steady state is measured
block = b"\1" * BUF
go_r, go_w = os.pipe()                              # one byte through a pipe per round for synchronization, carrying no data
done_r, done_w = os.pipe()
sys.stdout.flush()
pid = os.fork()
if pid == 0:
    for _ in range(TOTAL // BUF):
        os.read(go_r, 1)
        assert shm.buf[0] == 1 and shm.buf[BUF - 1] == 1   # the consumer reads the same physical memory directly
        os.write(done_w, b"k")
    os._exit(0)
t = time.perf_counter()
for _ in range(TOTAL // BUF):
    shm.buf[:BUF] = block                           # the producer: one memory copy into the shared buffer
    os.write(go_w, b"g")
    os.read(done_r, 1)
os.waitpid(pid, 0)
print(f"共享内存：{TOTAL / (time.perf_counter() - t) / 1e9:.1f} GB/s")
shm.close()
shm.unlink()
```

```text title="output (on this machine)"
管道：1.6 GB/s
Unix 域套接字：7.9 GB/s
共享内存：11.0 GB/s
```

The pipe is slow mainly because its buffer is only 64 KiB, so every 64 KiB written has to switch to the other side to be read (`fcntl(F_SETPIPE_SZ)` enlarges it, up to `/proc/sys/fs/pipe-max-size`, 1 MiB by default); the Unix domain socket's buffer is far larger, though the data still takes two copies through the kernel; shared memory takes only the producer's one write, with the consumer reading in place and the synchronization costing one byte through a pipe. When the producer generates the data in the shared memory to begin with (building the tensors there), even that copy disappears.

## ZMQ: the inference engine's message bus {#zmq推理引擎的消息总线}

Writing interprocess communication over sockets directly leaves a great deal to handle: TCP is a byte stream and the message boundaries are yours; connections have to bind before they connect and reconnect when they drop; and a full buffer when sending too fast has to be handled. **ZMQ** (ZeroMQ) wraps all of it:

- **messages** rather than a byte stream: one send is one message and the other side receives it whole;
- **patterns**: PUSH/PULL (a pipeline, load-balanced across receivers), PUB/SUB (broadcast), REQ/REP (request-reply), ROUTER/DEALER (asynchronous many to many);
- **connection management**: connections are established asynchronously in background I/O threads, reconnected automatically when dropped, and connecting before the bind works;
- **interchangeable transports**: `ipc://` (a Unix domain socket), `tcp://`, `inproc://` (between threads of one process), with no change to the code.

```python title="zmq_pushpull.py"
import os
import sys

import zmq

ADDR = f"ipc:///tmp/cs-handbook-{os.getpid()}.ipc"       # with the pid: two people running at once, or CI and a local run, never collide on the socket file
N = 10000

sys.stdout.flush()
pid = os.fork()
if pid == 0:                                        # the child: the PUSH end, connecting first, before the other side has bound
    sock = zmq.Context().socket(zmq.PUSH)
    sock.connect(ADDR)
    for i in range(N):
        sock.send(i.to_bytes(4, "little"))          # the messages queue locally and are delivered once connected
    sock.close(linger=-1)                           # wait for the queued messages to go out before closing
    os._exit(0)

pull = zmq.Context().socket(zmq.PULL)               # the parent: the PULL end, binding later
pull.bind(ADDR)
got = [int.from_bytes(pull.recv(), "little") for _ in range(N)]
os.waitpid(pid, 0)
print(f"收到 {len(got)} 条消息，顺序和内容都对：", got == list(range(N)))
print("先 connect、后 bind 也能工作：ZMQ 在后台自动建立（和重建）连接")
```

```text title="output"
收到 10000 条消息，顺序和内容都对： True
先 connect、后 bind 也能工作：ZMQ 在后台自动建立（和重建）连接
```

vLLM's API server and EngineCore process, and SGLang's TokenizerManager, Scheduler and DetokenizerManager, all pass messages over ZMQ's `ipc://` sockets (building it from scratch is in [messages and ZMQ](minisgl://serve/message/)).

## Serialization: what goes in a message {#序列化消息里装什么}

ZMQ only moves bytes, so Python objects have to be serialized first. pickle is the easiest and very expensive on a hot path. Suppose a step's scheduling results are 256 requests, each carrying this step's tokens and its block numbers:

```python title="pickle_cost.py"
import array
import pickle
import time

# one step's results: 256 requests, each with the tokens to compute this step (mostly decode, one token each)
batch = [{"req_id": f"req-{i}", "token_ids": [1000 + i] if i % 16 else list(range(512)), "block_ids": list(range(i, i + 8))}
         for i in range(256)]

t = time.perf_counter()
for _ in range(100):
    blob = pickle.dumps(batch)
    pickle.loads(blob)
pk = (time.perf_counter() - t) / 100

# the same information flattened into integer arrays: each request's token count + every token + the block numbers
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

```text title="output (on this machine)"
pickle：31 KiB，序列化 + 反序列化 376 µs
扁平数组：42 KiB，3 µs
```

pickle walks the whole object graph writing type information for every object and rebuilds every Python object on the way back; done every step, a few hundred microseconds eats a few per cent of a decode step. The usual optimizations:

- a faster format: vLLM V1 uses msgspec's msgpack encoding (`MsgpackEncoder` in `vllm/v1/serial_utils.py`), sending only metadata for a tensor with the data itself going through shared memory or a raw buffer;
- send only the delta: the scheduler sends the workers only what **changed** this step (new requests, newly allocated blocks) and each worker keeps the full state itself;
- flatten into arrays: requests numbered by integers rather than strings, with the tokens and block numbers packed into one integer array that the receiver "sees" through `numpy.frombuffer` with no copy.

Besides, pickle's deserialization can execute arbitrary code, so it belongs only between fully trusted processes and must never take data from outside.

## A shared-memory ring buffer {#共享内存环形缓冲区}

![Figure: a shared-memory ring buffer - the producer writes head and the consumer reads tail](../assets/figures/ring-buffer.svg){.aig-svg}

Every step, vLLM's EngineCore has to send the scheduling results to every worker (one per card under tensor parallelism). It uses a **broadcast ring buffer** in shared memory (`ShmRingBuffer` and the `MessageQueue` on top of it, in `vllm/distributed/device_communicators/shm_broadcast.py`): one writer, several readers, the buffer divided into blocks, each with a set of flags:

- 1 **write flag**: whether this block has been written;
- 1 **read flag per reader**: whether that reader has read this block.

The rules: a reader may read when the write flag is 1 and its own read flag is 0, and sets its read flag to 1 afterwards; the writer may overwrite when the write flag is 0 or every read flag is 1. After writing, the writer has to **clear the read flags first and set the write flag second**: the other order lets a reader see "the write flag is 1 but my read flag is still the 1 left from the previous round" and skip the new message thinking it has read it. Below is a simplified version with the same layout, one writer broadcasting 20 messages to 2 readers through only 4 blocks, so writing too fast makes the writer wait:

```python title="shm_ring.py"
import os
import sys
import threading
import time
from multiprocessing import shared_memory

# a simplified version with vLLM's ShmRingBuffer layout: a broadcast ring buffer with one writer and several readers
N_READER, CHUNKS, CHUNK = 2, 4, 64
META = CHUNKS * CHUNK                              # the metadata area follows the data: 1 write flag per block + 1 read flag per reader
_lock = threading.Lock()


def fence():                                       # as vLLM's memory_fence does: take and release a lock as a memory barrier
    with _lock:
        pass


class Ring:
    def __init__(self, shm):
        self.buf = shm.buf

    def _meta(self, i):
        return META + i * (1 + N_READER)

    def write(self, seq, payload):
        i, m = seq % CHUNKS, self._meta(seq % CHUNKS)
        while True:                                # this block may be overwritten only when it was never written or every reader has read it
            fence()
            if self.buf[m] == 0 or all(self.buf[m + 1:m + 1 + N_READER]):
                break
            time.sleep(0)
        self.buf[m] = 0                            # clear the write flag first
        self.buf[i * CHUNK] = len(payload)
        self.buf[i * CHUNK + 1:i * CHUNK + 1 + len(payload)] = payload
        self.buf[m + 1:m + 1 + N_READER] = bytes(N_READER)   # then the readers' flags ...
        fence()
        self.buf[m] = 1                            # ... and only last mark it "written". The other order lets a reader see the intermediate state

    def read(self, seq, reader):
        i, m = seq % CHUNKS, self._meta(seq % CHUNKS)
        while True:                                # written, and not yet read by us
            fence()
            if self.buf[m] == 1 and self.buf[m + 1 + reader] == 0:
                break
            time.sleep(0)
        n = self.buf[i * CHUNK]
        data = bytes(self.buf[i * CHUNK + 1:i * CHUNK + 1 + n])
        fence()
        self.buf[m + 1 + reader] = 1               # marked only after reading, which is when the writer may overwrite this block
        return data


shm = shared_memory.SharedMemory(create=True, size=META + CHUNKS * (1 + N_READER))
shm.buf[:shm.size] = bytes(shm.size)
N_MSG = 20
results = []
for reader in range(N_READER):
    r, w = os.pipe()
    sys.stdout.flush()
    if os.fork() == 0:                             # the reader process: read 20 messages in order and hand the result back through a pipe
        ring = Ring(shm)
        got = [ring.read(s, reader).decode() for s in range(N_MSG)]
        os.write(w, ("|".join(got)).encode())
        os._exit(0)
    results.append(r)

ring = Ring(shm)                                   # the parent is the writer: with only 4 blocks, writing faster than reading waits for the readers
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

```text title="output"
读者 0：按顺序收到 20 条广播，第一条 'step 0'，最后一条 'step 19'，全部正确： True
读者 1：按顺序收到 20 条广播，第一条 'step 0'，最后一条 'step 19'，全部正确： True
```

Several details match vLLM's implementation:

- **memory barriers**: there has to be a barrier between writing the data and writing the flag, and between reading the flag and reading the data, or the CPU or the compiler may reorder them and a reader that saw the flag reads stale data. vLLM's `memory_fence` takes and releases a `threading.Lock` to get a full barrier, which the `fence` above copies;
- **the waiting policy**: with no data, busy-poll first (with `sched_yield` meanwhile) and only sleep for a ZMQ notification after a while with nothing new (see [what a context switch costs](process-thread.md#上下文切换的代价));
- **large messages**: a message too large for one block falls back to ZMQ;
- **across machines**: when a reader is on another machine, the same `MessageQueue` interface switches to ZMQ's PUB/SUB.

## CUDA IPC: sharing device memory between processes {#cuda-ipc进程之间共享显存}

Shared memory solves sharing CPU memory, and device memory has its counterpart. **CUDA IPC**: a process exports a block of its device memory as a handle of a few dozen bytes with `cudaIpcGetMemHandle`, passes it by any means (ZMQ, a pipe) to another process, which opens it with `cudaIpcOpenMemHandle` and gets a pointer to the same memory, with no copy. PyTorch's `torch.multiprocessing` does this automatically when passing CUDA tensors between processes:

```python title="cuda_ipc.py" run="no"
# needs a GPU. Two processes share one block of device memory: the producer exports an IPC handle and the consumer opens it for the same memory, with no copy
import torch
import torch.multiprocessing as mp


def consumer(q):
    t = q.get()                    # the tensor received is the same memory, opened by cudaIpcOpenMemHandle
    t.add_(1)                      # this modifies the producer's memory
    q.put("done")


if __name__ == "__main__":
    mp.set_start_method("spawn")   # a CUDA process has to use spawn (see the processes, threads and scheduling chapter)
    q = mp.Queue()
    x = torch.zeros(4, device="cuda")
    p = mp.Process(target=consumer, args=(q,))
    p.start()
    q.put(x)                       # only the IPC handle and the metadata are passed
    q.get()                        # wait for the other side: the producer has to keep x valid while it is in use
    print(x)                       # tensor([1., 1., 1., 1.], device='cuda:0')
    p.join()
```

Things to watch:

- **lifetime**: the exporter has to keep that memory valid while the other side uses it, neither freeing it nor letting the caching allocator hand it to another tensor (PyTorch reference-counts shared tensors, but the cross-process logic is still yours to get right);
- **synchronization**: both processes see the same memory, so who writes and who reads first is yours to order (CUDA events can be shared across processes too);
- **scope**: only between GPUs of one machine, and the two cards have to be able to reach each other through P2P. The newer CUDA virtual memory interface (`cuMemExportToShareableHandle`) can export a file descriptor, and multi-node NVLink systems have cross-machine fabric handles as well.

What it is used for in an inference system: in RL training, where the training and inference processes share a card, the trained weights go to the inference engine through an IPC handle without passing through the CPU (SGLang's `/update_weights_from_ipc` and vLLM's `ipc` weight transfer, see [updating weights live](serving://ops/weight-update/)); and vLLM's `ipc_cache` load format has a resident daemon hold the already-quantized weights so a restarting engine maps them straight in.

!!! interview "Answering in an interview"
    Asked "how does vLLM's scheduling process send each step's results to several workers": through a broadcast ring buffer in shared memory (the `MessageQueue`), one writer and several readers, each block with a write flag and a read flag per reader; a reader that sees the write flag at 1 and its own read flag at 0 reads and sets its flag to 1; the writer may overwrite only once every reader has read it; after writing, the read flags are cleared before the write flag is set, and the other order makes readers miss messages; and memory barriers go between the flags and the data. The waiting busy-polls before sleeping, large messages fall back to ZMQ, and across machines it becomes ZMQ's PUB/SUB. Then compare the alternatives: pipes and sockets both copy twice through the kernel while shared memory copies once; and the API server and the scheduling process use ZMQ with msgpack, because pickle is too slow and unsafe besides.

## Exercises {#练习}

**1. The order of the flags.** In the ring buffer above, if the writer set the write flag before clearing the read flags, give a concrete sequence in which a reader goes wrong.

??? success "Answer"
    Suppose reader A read block 2 in the previous round, so its read flag is 1. The writer is about to overwrite block 2 this round: it clears the write flag, writes the data, and then sets the write flag to 1 first, at which moment the read flags are not cleared and A's is still the 1 from the previous round. A happens to check block 2 right then, sees "the write flag is 1 and my read flag is 1", concludes "I have read this block" and goes on waiting (or, if it advances by sequence number, skips the message), and only afterwards does the writer clear the read flags. A has missed this round's message. Clearing the read flags before setting the write flag guarantees that whenever a reader sees the write flag at 1, the read flags are already this round's initial 0.

**2. Choosing a mechanism.** Which way would you pass the data in each of these? (a) the scheduling process sending 200 requests' metadata to 8 workers every step; (b) the API server streaming generated tokens back to the process holding the HTTP connection; (c) after one RL training step, handing a 70B model's new weights to an inference engine on the same machine using the same GPUs.

??? success "Answer"
    (a) One to many, every step, latency-sensitive: a shared-memory broadcast ring buffer with the metadata in a compact format (msgpack or flat arrays) and the readers busy-polling. (b) Small messages one at a time, many to one, reliability wanted: ZMQ (PUSH/PULL or ROUTER/DEALER) over `ipc://` with msgpack encoding; at high rates, batch several requests' tokens before sending. (c) A great deal of data (hundreds of GB) with both source and destination on the GPU: CUDA IPC handles, with the inference engine reading straight from the training process's device memory and copying into its own parameters (layer by layer, minding the lifetime and the synchronization), never touching the CPU or a socket.

**3. Why the pipe is slow.** The measurement above gives the pipe 1.6 GB/s and the Unix domain socket nearly 8. How would you verify the explanation that "the pipe buffer is too small"?

??? success "Answer"
    Raise the pipe buffer to 1 MiB with `fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, 1 << 20)` and measure again: a marked rise in bandwidth says the buffer size really was the bottleneck (every 64 KiB forces a switch between the processes). You can also use `perf stat -e context-switches`, or read `voluntary_ctxt_switches` in `/proc/<pid>/status`, and compare how many context switches each mechanism takes for the same amount of data (the tools are in [Linux profiling tools](perf-tools.md)).

## Summary {#小结}

- [x] A pipe and a Unix domain socket copy twice through the kernel; shared memory takes only the producer's one write, with the synchronization yours. On this machine: a pipe 1.6, a Unix domain socket 8 and shared memory 11 GB/s.
- [x] ZMQ provides message boundaries, the patterns, automatic connection management and interchangeable transports, and is vLLM's and SGLang's message bus between processes.
- [x] pickle is too slow and unsafe on a hot path: use msgpack (vLLM's msgspec), send only the delta, flatten into arrays, and move large tensors through shared memory.
- [x] vLLM's `ShmRingBuffer`: a write flag plus a read flag per reader on each block, the read flags cleared before the write flag is set, memory barriers between the flags and the data, and polling before sleeping.
- [x] CUDA IPC shares device memory with no copy: mind the lifetime and the synchronization; it is how weights are exchanged when RL shares a card.
