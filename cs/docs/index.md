# 计算机基础手册

<p class="lead">推理工程师的招聘要求里几乎总有一句"熟悉操作系统、计算机网络、数据结构和算法"。这本手册只讲和推理系统有关的那部分，并且都从推理系统里的真实现象讲起：fork 之后用 CUDA 为什么报错，PagedAttention 借用了虚拟内存的哪些想法，vLLM 怎么用共享内存把调度结果广播给所有 GPU，容器里的 CPU 节流怎么拖高延迟……每个结论都配有一段在 Linux 上实际运行过的 Python 或 C 程序。</p>

## 这份手册适合谁

- 做推理框架、推理优化或推理平台，科班的基础课有些忘了，或者没有系统学过；
- 读 vLLM、SGLang 的源码时，对共享内存、ZMQ、锁页内存、NUMA、io_uring 这些名词一知半解；
- 准备面试，需要讲清"一次写入如何落盘""epoll 和 io_uring 的区别""PagedAttention 和虚拟内存是什么关系"这类经典问题。

学完并练完这份手册，你应该能做到：

- 解释推理引擎为什么拆成多个进程，fork 和 spawn 怎么选，调度和上下文切换的代价有多大；
- 讲清虚拟内存、缺页、TLB 和大页，以及 PagedAttention、CUDA 虚拟内存接口和它们的关系；
- 算出权重加载、KV 卸载走不同路径要多久，知道锁页内存和 NUMA 绑定该怎么用；
- 描述一次写入从系统调用到落盘的全过程，用好页缓存、mmap、O_DIRECT、epoll 和 io_uring；
- 排查容器里的 CPU 节流、线程过多、`/dev/shm` 不足，用 USE 方法和 perf、py-spy、strace 定位性能问题。
- 拆开一张 GPU 的峰值算力和显存带宽，解释 Tensor Core 为什么一代比一代大、注意力里的 exp 为什么会成为瓶颈；
- 用屋脊点判断一个负载受算力还是受带宽限制，算出 decode 要多大的 batch 才算力受限，读懂新卡的规格表；
- 讲清一台 8 卡服务器和一个 NVLink 机柜的内部结构，估算集合通信的耗时，判断一张卡该不该切给多个任务。
- 说清一个流式请求从 DNS 到第一个 token 的全过程，定位 Nagle、缓冲、背压造成的延迟；
- 用排队论给集群定容量，配好限流、重试与熔断，知道重试为什么会把故障放大；
- 用一致性哈希做缓存感知路由，讲清法定人数、Raft 选举与脑裂，知道什么该放进 etcd、什么不该。

## 学习路线

<div class="roadmap" markdown>

| 部分 | 章节 | 学完能做什么 | 建议用时 |
| --- | --- | --- | --- |
| 一、操作系统 | [进程、线程与调度](os/process-thread.md) · [虚拟内存、页表与大页](os/virtual-memory.md) · [锁页内存、DMA 与 NUMA](os/pinned-numa.md) · [一次写入如何落盘](os/io-stack.md) · [epoll 与 io_uring](os/io-models.md) · [进程间通信](os/ipc.md) · [容器](os/containers.md) · [Linux 性能分析工具](os/perf-tools.md) | 讲清推理引擎的进程结构、内存和 I/O 路径，能在容器里排查性能问题 | 1～1.5 周 |
| 二、体系结构：从 CPU 到 GPU | [CPU 体系结构速成](arch/cpu.md) · [GPU 的 SM 与 Tensor Core](arch/gpu-sm.md) · [GPU 内存系统](arch/gpu-memory.md) · [架构演进：Volta 到 Blackwell](arch/evolution.md) · [多卡系统](arch/multi-gpu.md) | 从硬件层面解释推理的性能数字：峰值怎么来的、瓶颈在哪、为什么要量化和攒批 | 1 周 |
| 三、计算机网络 | [TCP：一个请求的网络之旅](net/tcp.md) · [HTTP 与流式输出](net/http-stream.md) · [负载均衡与排队](net/load-balance.md) | 讲清一个流式请求在网络上的完整路径，会算 RTT、排队与容量 | 3～4 天 |
| 四、分布式系统 | [一致性哈希与分片](dist/hash-shard.md) · [复制与共识](dist/replication.md) | 设计路由和 KV 存储时用得上的分布式基础 | 2～3 天 |
| 五、数据结构与算法 | 即将上线：高频题型与推理系统里的数据结构 | 应对算法面试，并把数据结构和推理系统联系起来 | 与主线并行 |

</div>

八本手册的逐章路线见[学习路线图](root://roadmap/)，求职冲刺的逐周安排见[冲刺计划](root://plan/)。这本书的各章分散排在计划的各周里，和相关的推理系统内容放在一起学：比如进程间通信和[手写 mini-sglang 的消息与 ZMQ](minisgl://serve/message/)同一周，锁页内存和 NUMA 与[分布式推理](serving://distributed/tensor-parallel/)同一周。

## 怎么用

1. **先做自测**：每章开头有自测题，能答上来就直接做练习，答不上来再细读。
2. **把程序跑一遍**：每段程序都能直接运行（`python3 x.py`、`gcc -O2 x.c -o x && ./x`），改改参数看结果怎么变，比读十遍都记得牢。
3. **面试前看「面试怎么答」**：每章有一个按面试回答组织好的提示框。

## 怎么验证的

- 所有程序都在 Linux（x86-64，内核 5.15）上实际运行过（网络部分的程序用本机 loopback 收发真实的 TCP 报文）。标着"输出"的块与运行结果逐行核对；标着"输出（本机示例）"的是和机器相关的测量结果（耗时、带宽、缺页次数），你机器上的数字会不同，但规律应该一致。
- 例子针对 Linux：macOS 的进程、内存和 I/O 接口不一样（没有 epoll、io_uring、cgroup 和 `/proc`）。在 Mac 上学的话，用 Docker 起一个 Linux 容器（`docker run -it --rm -v "$PWD":/w -w /w python:3.12 bash`），或者直接在租的 GPU 云主机上跑。

```bash
# 需要 Python 3.10+（进程间通信一章要 pyzmq，文件与 I/O 一章要 numpy）和 gcc；性能工具一章用到 perf 和 strace
pip install pyzmq numpy
python3 tools/check_code.py              # 在 cs/ 目录下校验全部示例
python3 tools/check_code.py docs/os/*.md # 只校验操作系统部分
```
