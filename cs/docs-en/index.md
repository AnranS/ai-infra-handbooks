# CS Fundamentals

<p class="lead">An inference engineer's job description nearly always contains a line about "familiarity with operating systems, networking, data structures and algorithms". This handbook covers only the part that bears on inference systems, and every topic starts from something you really see in one: why CUDA fails after a fork, which ideas PagedAttention borrowed from virtual memory, how vLLM broadcasts the scheduler's results to every GPU through shared memory, how CPU throttling inside a container drags the latency up. Every conclusion comes with a Python or C program that has actually been run on Linux.</p>

## Who this handbook is for {#这份手册适合谁}

- people working on inference frameworks, inference optimization or an inference platform, who have forgotten some of the fundamentals or never studied them systematically;
- people reading vLLM's or SGLang's source who have only a vague grasp of shared memory, ZMQ, pinned memory, NUMA and io_uring;
- people preparing for interviews who need to explain "how a write reaches the disk", "the difference between epoll and io_uring" and "how PagedAttention relates to virtual memory".

Having read this handbook and done the exercises, you should be able to:

- explain why an inference engine is split into several processes, how to choose between fork and spawn, and what scheduling and context switching cost;
- explain virtual memory, page faults, the TLB and huge pages, and how PagedAttention and CUDA's virtual memory interface relate to them;
- work out how long loading weights or offloading KV takes along different paths, and how pinned memory and NUMA binding should be used;
- describe a write's whole journey from the system call to the disk, and use the page cache, mmap, O_DIRECT, epoll and io_uring well;
- diagnose CPU throttling in a container, too many threads and a `/dev/shm` that is too small, and locate a performance problem with the USE method, perf, py-spy and strace.
- take a GPU's peak compute and memory bandwidth apart, explain why the Tensor Cores grow every generation and why the exp inside attention becomes the bottleneck;
- use the ridge point to decide whether a workload is compute- or bandwidth-bound, work out how large a batch decode needs to become compute-bound, and read a new card's specification sheet;
- explain the internals of an 8-GPU server and of an NVLink rack, estimate what a collective costs, and decide whether a card should be shared between tasks.
- describe a streaming request's whole path from DNS to the first token, and locate the latency Nagle, buffering and backpressure cause;
- size a cluster with queueing theory, configure rate limiting, retries and circuit breaking, and know why retries amplify a failure;
- route cache-aware with consistent hashing, explain quorums, Raft elections and split brain, and know what belongs in etcd and what does not.
- read a medium-difficulty algorithm problem in 40 minutes, explain the approach, get the boundaries right and analyze the complexity (six chapters with 63 exercises graded in the browser).

## The learning path {#学习路线}

<div class="roadmap" markdown>

| Part | Chapters | What you can do afterwards | Suggested time |
| --- | --- | --- | --- |
| 1. Operating systems | [processes, threads and scheduling](os/process-thread.md) · [virtual memory, page tables and huge pages](os/virtual-memory.md) · [pinned memory, DMA and NUMA](os/pinned-numa.md) · [how a write reaches the disk](os/io-stack.md) · [epoll and io_uring](os/io-models.md) · [interprocess communication](os/ipc.md) · [containers](os/containers.md) · [Linux profiling tools](os/perf-tools.md) | explain an inference engine's process structure, memory and I/O paths, and diagnose a performance problem inside a container | 1 to 1.5 weeks |
| 2. Architecture: from CPU to GPU | [a CPU architecture crash course](arch/cpu.md) · [a GPU's SMs and Tensor Cores](arch/gpu-sm.md) · [a GPU's memory system](arch/gpu-memory.md) · [the architectures: Volta to Blackwell](arch/evolution.md) · [multi-GPU systems](arch/multi-gpu.md) | explain inference's performance numbers from the hardware: where the peak comes from, where the bottleneck is, why quantization and batching exist | 1 week |
| 3. Networking | [TCP: a request's journey over the network](net/tcp.md) · [HTTP and streaming](net/http-stream.md) · [load balancing and queueing](net/load-balance.md) | explain a streaming request's whole path over the network, and compute RTT, queueing and capacity | 3 to 4 days |
| 4. Distributed systems | [consistent hashing and sharding](dist/hash-shard.md) · [replication and consensus](dist/replication.md) | the distributed basics that routing and KV storage design need | 2 to 3 days |
| 5. Data structures and algorithms | [how to prepare for an algorithm interview](algo/overview.md) · [arrays and strings](algo/array-string.md) · [linked lists, stacks and hashing](algo/linked-stack-hash.md) · [trees and graphs](algo/tree-graph.md) · [sorting, heaps and greedy algorithms](algo/sort-heap-greedy.md) · [dynamic programming and backtracking](algo/dp-backtrack.md) | handle an algorithm interview, and connect the data structures to inference systems | alongside the main line |

</div>

The chapter-by-chapter route across all the handbooks is in the [roadmap](root://roadmap/), and the week-by-week arrangement for a job-hunting sprint is in the [sprint plan](root://plan/). This book's chapters are spread across the plan's weeks and studied alongside the related inference material: interprocess communication falls in the same week as [mini-sglang's messages and ZMQ](minisgl://serve/message/), and pinned memory and NUMA in the same week as [distributed inference](serving://distributed/tensor-parallel/).

## How to use it {#怎么用}

1. **Take the self-test first**: every chapter opens with one, and answering it means going straight to the exercises while failing it means reading carefully.
2. **Run the programs**: every one runs directly (`python3 x.py`, `gcc -O2 x.c -o x && ./x`), and changing a parameter to see what happens sticks better than ten readings.
3. **Read "Answering in an interview" beforehand**: every chapter has one box organized the way an interview answer should go.

## How it was verified {#怎么验证的}

- Every program has been run on Linux (x86-64, kernel 5.15), with the networking programs sending real TCP over the local loopback. A block marked "output" was checked line by line against the run; one marked "output (an example from one machine)" is a machine-dependent measurement (a time, a bandwidth, a fault count), so your numbers will differ while the pattern should hold.
- The examples target Linux: macOS's process, memory and I/O interfaces differ (no epoll, no io_uring, no cgroups and no `/proc`). To study on a Mac, start a Linux container with Docker (`docker run -it --rm -v "$PWD":/w -w /w python:3.12 bash`), or work directly on a rented GPU host.

```bash
# needs Python 3.10+ (pyzmq for the interprocess communication chapter, numpy for the files and I/O chapter) and gcc; the profiling chapter uses perf and strace
pip install pyzmq numpy
python3 tools/check_code.py              # verify every example from the cs/ directory
python3 tools/check_code.py docs/os/*.md # verify the operating systems part only
```
