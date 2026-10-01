# 锁页内存、DMA 与 NUMA

<p class="lead">GPU、网卡、NVMe 盘读写内存都不经过 CPU，而是自己用 DMA 搬数据。DMA 只认物理地址，所以被搬的内存必须"锁住"，不能被换出，也不能被挪走——这就是锁页内存。一台多路服务器上，内存和 PCIe 设备还分属不同的 CPU 插槽，访问远端插槽的内存要多走一段路——这就是 NUMA。推理系统的很多细节都和这两件事有关：权重加载和 KV 卸载的速度、`non_blocking=True` 什么时候真的异步、RDMA 在容器里为什么注册内存失败、GPU worker 为什么要绑到特定的 CPU 上。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. DMA 是什么？为什么 DMA 要求内存锁页？
    2. 为什么从普通（可换页）内存执行 `x.to("cuda", non_blocking=True)` 其实没有和计算重叠？
    3. 锁页内存有什么代价？为什么不能把能锁的都锁上？
    4. GPUDirect RDMA 和 GPUDirect Storage 各自省掉了哪一步？
    5. 什么是 NUMA？一个 GPU worker 跑在离它的 GPU 远的那个 CPU 插槽上，会慢在哪里？

??? success "自测参考答案（先自己答，再展开对照）"
    1. DMA（直接内存访问）是设备自己按物理地址读写内存，CPU 只负责下发请求、接收完成通知。普通内存的物理页随时可能被换出到磁盘，或者被内核迁移、合并，设备拿着旧的物理地址就会读写到错误的地方，所以 DMA 期间内存必须锁住：物理页固定、不换出、不迁移。
    2. 可换页内存不能直接 DMA，驱动要先用 CPU 把数据拷进自己的一块锁页缓冲区，再从那里 DMA 到 GPU。这个拷贝由调用线程同步完成，函数返回时数据基本已经搬完了，所以既没有和 GPU 计算重叠，还多拷了一次。只有源数据本身在锁页内存里，`non_blocking=True` 才是真的异步。
    3. 锁住的内存不能被换出、也不能被回收成页缓存，锁多了整台机器可用于页缓存和其他进程的内存就少了，严重时导致内存紧张；分配锁页内存本身也慢（要逐页锁定、在驱动或网卡里登记），所以通常一次分配好反复用。普通进程能锁的量还受 `ulimit -l`（RLIMIT_MEMLOCK）限制，RDMA 注册内存就受它约束。
    4. GPUDirect RDMA 让网卡直接读写 GPU 显存，省掉"显存 → CPU 内存 → 网卡"的中转拷贝（NCCL 跨机通信、KV 传输都用它）；GPUDirect Storage 让 NVMe 盘直接 DMA 到显存，省掉"盘 → 页缓存 → 用户缓冲区 → 锁页缓冲区 → 显存"这一串 CPU 内存里的拷贝。
    5. NUMA（非一致内存访问）：多路服务器上每个 CPU 插槽有自己的内存控制器，访问本插槽的内存快，访问另一个插槽的内存要经过插槽间的互联，延迟更高、带宽更低；PCIe 设备也挂在某一个插槽下。worker 放错插槽，它准备输入时访问的内存是远端的，往 GPU 拷数据还要跨插槽再走 PCIe，带宽和延迟都变差，而且互联被多个 worker 争抢，抖动更大。

## DMA：设备自己搬数据

![图：可分页内存、锁页内存与 GPUDirect 的三条搬运路径，以及 NUMA](../assets/figures/pinned-dma.svg){.aig-svg}

把一块数据从 CPU 内存拷到 GPU，真正搬数据的是 GPU 上的拷贝引擎：CPU 告诉它"从这个物理地址开始，搬这么多字节"，它自己通过 PCIe 读内存，搬完了再通知 CPU。这叫 **DMA**（直接内存访问）。网卡收发数据、NVMe 盘读写数据也都是这样。

DMA 只认物理地址，而进程的内存是虚拟内存：一个虚拟页对应的物理页可能被换出到磁盘，可能被内核迁移（内存规整、NUMA 平衡），甚至还没分配。所以被 DMA 的内存必须**锁页**（pinned / page-locked）：物理页全部分配好，在 DMA 完成之前不换出、不迁移。操作系统提供的原语是 `mlock`：

```python title="mlock.py" ci="no"
import ctypes
import mmap

libc = ctypes.CDLL(None, use_errno=True)
MiB = 1 << 20


def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])            # 单位是 kB


buf = mmap.mmap(-1, 64 * MiB, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
before_lck, before_rss = status("VmLck"), status("VmRSS")
if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(64 * MiB)) != 0:   # 锁住：全部分配好物理页，并且不许换出
    raise OSError(ctypes.get_errno(), "mlock 失败（检查 ulimit -l）")
print("VmLck 增加了", (status("VmLck") - before_lck) // 1024, "MiB")
print("mlock 顺带把这 64 MiB 全部分配好了，RSS 增加约 64 MiB：", 63 < (status("VmRSS") - before_rss) / 1024 < 66)
libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(64 * MiB))
print("解锁之后 VmLck 回到", status("VmLck") - before_lck, "kB")
```

```text title="输出"
VmLck 增加了 64 MiB
mlock 顺带把这 64 MiB 全部分配好了，RSS 增加约 64 MiB： True
解锁之后 VmLck 回到 0 kB
```

CUDA 用驱动自己的方式锁页（`cudaHostAlloc`、`cudaHostRegister`，PyTorch 里是 `tensor.pin_memory()` 或 `torch.empty(..., pin_memory=True)`），道理一样。从锁页内存拷贝，拷贝引擎直接 DMA，CPU 马上返回，可以和 GPU 上的计算重叠；从普通内存拷贝，驱动要先用 CPU 把数据拷进自己的一块锁页中转缓冲区，再 DMA，这一步是同步的：

| 源内存 | 拷贝过程 | `non_blocking=True` 的效果 |
| --- | --- | --- |
| 锁页内存 | 拷贝引擎直接 DMA | 真正异步，可以和计算重叠 |
| 可换页内存 | CPU 先拷进驱动的锁页缓冲区，再 DMA | 基本是同步的，还多一次 CPU 拷贝 |

所以推理引擎准备每一步的输入（token id、位置、块表）时，会先写进预先分配好的锁页 CPU 缓冲区，再异步拷到 GPU；KV Cache 卸载到 CPU 内存时，CPU 侧的缓冲区也是锁页的。

锁页内存的代价：

- **挤占内存**：锁住的页不能换出、不能被回收成页缓存，锁得越多，留给页缓存和其他进程的越少；
- **分配慢**：要逐页分配、锁定并在驱动里登记，大块分配往往要几十毫秒到几秒，所以都是启动时一次分配、之后反复使用（PyTorch 有专门缓存锁页内存的分配器）；
- **有上限**：普通进程能 `mlock` 的量受 `ulimit -l` 限制。RDMA 注册内存（`ibv_reg_mr`）受这个限制约束，容器里默认值往往很小，于是 RDMA 在容器里报 "Cannot allocate memory"，要用 `--ulimit memlock=-1` 放开（CUDA 的锁页分配不受这个限制）。

## 带宽：搬一次权重要多久

各种路径搬 16 GB 数据（一个 8B 模型的 BF16 权重）的时间，带宽取实际能跑到的量级：

```python title="transfer.py"
# 把 16 GB 的权重从 CPU 内存搬到 GPU，各种路径大约要多久（带宽取实际能跑到的量级）
GB = 1e9
size = 16 * GB
paths = [
    ("PCIe 4.0 x16，锁页内存", 25 * GB),
    ("PCIe 4.0 x16，可换页内存（先拷进驱动的锁页缓冲区）", 12 * GB),
    ("PCIe 5.0 x16，锁页内存", 50 * GB),
    ("NVLink-C2C（Grace Hopper 的 CPU 到 GPU）", 450 * GB),
    ("400 Gb/s 网卡从远端读（GPUDirect RDMA）", 45 * GB),
]
for name, bw in paths:
    print(f"{name}：{size / bw * 1e3:,.0f} ms")
```

```text title="输出"
PCIe 4.0 x16，锁页内存：640 ms
PCIe 4.0 x16，可换页内存（先拷进驱动的锁页缓冲区）：1,333 ms
PCIe 5.0 x16，锁页内存：320 ms
NVLink-C2C（Grace Hopper 的 CPU 到 GPU）：36 ms
400 Gb/s 网卡从远端读（GPUDirect RDMA）：356 ms
```

几个推论：

- 从 CPU 内存（比如已经在页缓存里的权重）加载到 GPU，PCIe 本身只要零点几秒，慢的往往是前面的环节：从磁盘或网络存储读到内存、反序列化、可换页内存的中转拷贝；
- decode 的每一步只有几十毫秒，这期间 CPU 和 GPU 之间只能搬很小的数据。KV Cache 卸载到 CPU 可以做，但换回来的时间要和重算比较（见下面的练习）；
- Grace Hopper、GB200 这类用 NVLink-C2C 连接 CPU 和 GPU 的机器，CPU 内存到 GPU 快了一个数量级，卸载到 CPU 内存变得便宜得多。

## GPUDirect：绕开 CPU 内存

默认情况下，GPU 和其他设备之间的数据要经过 CPU 内存中转。GPUDirect 是一组让设备之间直接 DMA 的技术：

- **P2P**：同一台机器上的 GPU 之间通过 NVLink 或 PCIe 直接读写对方的显存，张量并行的 all-reduce、PD 分离的机内 KV 传输都走这条路；
- **GPUDirect RDMA**：网卡直接读写 GPU 显存。跨机的 NCCL 通信、PD 分离的跨机 KV 传输（Mooncake、NIXL 这类传输引擎）都依赖它，否则每次都要"显存 → CPU 内存 → 网卡"中转（见 [RDMA 编程模型](serving://comm/rdma/)、[KV 传输引擎与分布式存储](serving://comm/kv-storage/)）；
- **GPUDirect Storage**：NVMe 盘直接 DMA 到显存（通过 cuFile 接口）。普通的读文件路径是"盘 → 页缓存 → 用户缓冲区 → 锁页缓冲区 → 显存"，中间两三次拷贝都在 CPU 内存里；GDS 把它变成一次 DMA，适合从本地 NVMe 快速加载权重或 KV。

这些路径都要求设备在 PCIe 拓扑上"离得近"：网卡和 GPU 挂在同一个 PCIe 交换芯片下时，GPUDirect RDMA 的带宽最好；如果要跨 CPU 插槽，性能会明显下降，NCCL 选网卡时就会考虑这一点。这就引出了 NUMA。

## NUMA：内存也分远近

多路服务器上，每个 CPU 插槽有自己的内存控制器和内存条，插槽之间用互联（Intel 的 UPI、AMD 的 Infinity Fabric）连接。访问本插槽的内存快，访问另一个插槽的内存要经过互联，这叫**非一致内存访问**（NUMA），每个插槽连同它的内存是一个 NUMA 节点。本机正好是两路服务器，先测一下默认情况：

```c title="numabw.c"
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
  /* 带宽：顺序读 1 GiB，读 4 遍 */
  size_t n = (1UL << 30) / sizeof(uint64_t);
  uint64_t *a = malloc(n * sizeof *a);
  if (!a) return 1;
  memset(a, 1, n * sizeof *a);                   /* 先写一遍，让物理页按当前的内存策略分配好 */
  uint64_t sum = 0;
  double t0 = now();
  for (int r = 0; r < 4; r++)
    for (size_t i = 0; i < n; i++) sum += a[i];
  double bw = 4.0 * n * sizeof *a / (now() - t0) / 1e9;

  /* 延迟：在 256 MiB 里随机跳着读（指针追逐） */
  size_t m = (256UL << 20) / 64;
  uint32_t *next = (uint32_t *)a;                /* 复用同一块内存，每 64 字节一个节点 */
  uint32_t *perm = malloc(m * sizeof *perm);
  if (!perm) return 1;
  for (size_t i = 0; i < m; i++) perm[i] = i;
  uint64_t x = 88172645463325252ULL;
  for (size_t i = m - 1; i > 0; i--) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    size_t j = x % (i + 1);
    uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (size_t i = 0; i < m; i++) next[(size_t)perm[i] * 16] = perm[(i + 1) % m];
  uint32_t cur = perm[0];
  const int steps = 10000000;
  t0 = now();
  for (int s = 0; s < steps; s++) cur = next[(size_t)cur * 16];
  double lat = (now() - t0) / steps * 1e9;
  printf("单线程顺序读带宽 %.1f GB/s，随机访问延迟 %.0f ns%s\n", bw, lat, (sum == 0 && cur == 0) ? " " : "");
  return 0;
}
```

```text title="输出（本机示例）"
单线程顺序读带宽 17.8 GB/s，随机访问延迟 121 ns
```

再用 `numactl` 把进程固定在节点 0 的 CPU 上，内存分别放在节点 0 和节点 1：

```python title="numa.py" ci="no"
import glob
import subprocess

for node in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
    cpus = open(f"{node}/cpulist").read().strip()
    print(f"{node.rsplit('/', 1)[1]}：CPU {cpus}")

# 进程固定跑在节点 0 的 CPU 上，内存分别放在本地（节点 0）和远端（节点 1）
for mem in (0, 1):
    out = subprocess.run(["numactl", "--cpunodebind=0", f"--membind={mem}", "./numabw"],
                         capture_output=True, text=True, check=True).stdout.strip()
    print(f"CPU 在节点 0、内存在节点 {mem}（{'本地' if mem == 0 else '远端'}）：{out}")
```

```text title="输出（本机示例）"
node0：CPU 0-15
node1：CPU 16-31
CPU 在节点 0、内存在节点 0（本地）：单线程顺序读带宽 17.7 GB/s，随机访问延迟 121 ns
CPU 在节点 0、内存在节点 1（远端）：单线程顺序读带宽 14.3 GB/s，随机访问延迟 186 ns
```

远端内存的延迟高了一半左右，带宽也低了一截；多个进程同时跨节点访问时，互联带宽还会被争抢。Linux 默认的内存策略是**首次访问**（first touch）：一页内存分配在第一次写它的那个 CPU 所在的节点上。所以一个进程在节点 0 上初始化好缓冲区、之后被调度到节点 1 上运行，它的内存就全是远端的（内核的自动 NUMA 平衡会慢慢迁移，但不及时）。

PCIe 设备也属于某个 NUMA 节点。查看方式：`/sys/bus/pci/devices/<PCI 地址>/numa_node`，或者 `nvidia-smi topo -m`：它列出每张 GPU 的 CPU 亲和性和 NUMA 亲和性，以及 GPU 与 GPU、GPU 与网卡之间的连接方式（`NV#` 表示 NVLink，`PIX` / `PXB` 表示同一个 PCIe 交换芯片下，`NODE` 表示同一个 NUMA 节点内要经过 CPU，`SYS` 表示要跨插槽）。一台 8 卡服务器通常是 0～3 号 GPU 挂在节点 0 下、4～7 号挂在节点 1 下。

推理服务的做法是把每个 GPU worker 绑到它的 GPU 所在的节点：CPU 用那个节点的核，内存从那个节点分配。vLLM 的 `--numa-bind`（配置在 `vllm/config/parallel.py`）就是用 `numactl --cpunodebind` 和 `--membind` 启动每个 worker，节点可以自动探测，也可以用 `--numa-bind-nodes` 手动指定（比如 `[0, 0, 1, 1]`），或者用 `--numa-bind-cpus` 指定具体的核。

!!! interview "面试怎么答"
    被问"为什么要用锁页内存"：DMA 只认物理地址，普通内存的物理页可能被换出或迁移，所以驱动要先用 CPU 拷进自己的锁页缓冲区再 DMA，这一步是同步的；源数据本身在锁页内存里，拷贝引擎才能直接 DMA、真正异步地和计算重叠。推理引擎每一步的输入缓冲区、KV 卸载的 CPU 缓冲区都用锁页内存，一次分配、反复使用。再补充代价：锁页内存不能换出、不能回收，分配慢，RDMA 注册还受 `ulimit -l` 限制。被追问 NUMA：多路服务器上内存和 PCIe 设备分属不同插槽，本机测到远端内存延迟高一半、带宽低两成，所以 GPU worker 要用 `numactl` 绑到 GPU 所在的节点（vLLM 的 `--numa-bind`），网卡也选和 GPU 在同一个 PCIe 交换芯片下的。

## 练习

**1. non_blocking 的坑。** 下面的代码有什么问题？如果把第一行改成 `buf = torch.empty(n, pin_memory=True)` 呢？

```python
buf = torch.empty(n)                        # CPU 上的输入缓冲区
for step in range(steps):
    buf.copy_(next_inputs(step))            # 准备这一步的输入
    x = buf.to("cuda", non_blocking=True)
    y = model(x)
```

??? success "参考答案"
    `buf` 是可换页内存，`to("cuda", non_blocking=True)` 实际上由 CPU 同步地拷进驱动的中转缓冲区，所以代码是对的，但没有任何重叠，每一步都白等一次拷贝。改成锁页内存后拷贝真正异步了，问题也来了：`to()` 返回时 DMA 可能还没完成，下一轮循环的 `buf.copy_(...)` 就开始改写 `buf`，GPU 读到的可能是下一步的一部分数据。正确做法是轮换两块（或多块）锁页缓冲区，并在改写一块缓冲区之前确认用它的那次拷贝已经完成（记录一个 CUDA event，改写前 `event.synchronize()`，或者让拷贝所在的流和计算流正确同步）。推理引擎的输入准备就是这样做的。

**2. 换回来还是重算。** 一个 70B 模型（80 层，GQA 有 8 个 KV 头，头维度 128，BF16），某个请求有 32K token 的上下文被卸载到了 CPU 内存。用 PCIe 5.0（约 50 GB/s）换回 GPU 要多久？如果不换、直接重算这 32K token 的 prefill 呢（按单卡约 600 TFLOPS 的有效算力估算）？

??? success "参考答案"
    每个 token 的 KV：80 层 × 8 头 × 128 维 × 2（K 和 V）× 2 字节 = 327,680 字节 ≈ 0.31 MiB，32K token 共 10 GiB，换回来约 10.7 GB / 50 GB/s ≈ 0.21 秒。重算 prefill 约需 2 × 70e9 × 32768 ≈ 4.6e15 次浮点运算，单卡 600 TFLOPS 要约 7.6 秒，8 卡张量并行也要 1 秒左右。长上下文时换回来比重算便宜得多；上下文很短时（几百个 token），重算的计算量很小、还能和别的请求一起批处理，而换回来有固定的开销，重算反而更划算。这就是 KV 分层缓存的基本账（见 [KV 分层缓存与卸载](serving://distributed/kv-offload/)）。

**3. 放错节点。** 一台两路 8 卡服务器，GPU 4～7 挂在 NUMA 节点 1 下。负责 GPU 5 的 worker 进程被操作系统调度在节点 0 的 CPU 上，它的输入缓冲区也在节点 0 的内存里。哪些操作会变慢？怎么修？

??? success "参考答案"
    CPU 侧准备输入时读写的是本地内存，这部分不受影响，但 worker 访问的其他数据（比如 GPU 5 附近的共享内存消息队列，如果它们在节点 1 上）是远端访问；更主要的是每次把输入从节点 0 的内存拷到 GPU 5，DMA 都要先跨插槽互联、再走 GPU 5 所在的 PCIe，带宽更低、延迟更高，多个 worker 都这样时互联还会成为瓶颈，延迟抖动变大。修法：启动时用 `numactl --cpunodebind=1 --membind=1` 把 worker 绑到节点 1（vLLM 用 `--numa-bind` 自动完成），并确保锁页缓冲区是在绑定之后分配的（首次访问策略决定页的位置）。

## 小结

- [x] DMA 是设备自己按物理地址搬数据，被搬的内存必须锁页；`mlock`、`cudaHostAlloc`、`pin_memory()` 做的都是这件事。
- [x] 只有从锁页内存拷贝，`non_blocking=True` 才真正异步；可换页内存要经过驱动的中转缓冲区，基本同步。锁页缓冲区要在拷贝完成前保持不变。
- [x] 锁页内存挤占页缓存、分配慢、受 `ulimit -l` 限制（RDMA 注册），所以一次分配、反复使用。
- [x] 16 GB 权重过 PCIe 5.0 约 0.3 秒；长上下文的 KV 换回来比重算便宜得多。GPUDirect 让 GPU 与网卡、NVMe、其他 GPU 之间直接 DMA。
- [x] NUMA：本机远端内存延迟高一半、带宽低约两成；GPU worker 要绑到 GPU 所在的节点（`numactl`、vLLM 的 `--numa-bind`），内存按首次访问分配，绑定要在分配之前。
