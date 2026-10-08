# Pinned memory, DMA and NUMA

<p class="lead">A GPU, a network card and an NVMe drive all read and write memory without going through the CPU, moving the data themselves by DMA. DMA only knows physical addresses, so the memory being moved has to be "pinned": never swapped out and never relocated. That is pinned memory. On a multi-socket server, the memory and the PCIe devices also belong to different CPU sockets, and reaching the far socket's memory takes a longer road. That is NUMA. A great many details of an inference system turn on these two things: how fast the weights load and the KV offloads, when <code>non_blocking=True</code> is really asynchronous, why RDMA fails to register memory inside a container, why a GPU worker should be pinned to particular CPUs.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is DMA? Why does it require the memory to be pinned?
    2. Why does `x.to("cuda", non_blocking=True)` from ordinary (pageable) memory not actually overlap with computation?
    3. What does pinned memory cost? Why not pin everything that can be pinned?
    4. Which step does each of GPUDirect RDMA and GPUDirect Storage remove?
    5. What is NUMA? Where does a GPU worker running on the socket far from its GPU lose time?

??? success "Answers (try it yourself first, then expand)"
    1. DMA (direct memory access) is a device reading and writing memory by physical address itself, with the CPU only issuing requests and taking the completions. Ordinary memory's physical pages may be swapped to disk at any moment, or migrated or merged by the kernel, and a device holding the old physical address would read or write the wrong place, so during a DMA the memory has to be pinned: the physical pages fixed, never swapped, never migrated.
    2. Pageable memory cannot be DMAed directly, so the driver first copies the data with the CPU into a pinned buffer of its own and DMAs from there. That copy is done synchronously by the calling thread, and by the time the function returns the data is essentially moved, so there is no overlap with GPU computation and there is one extra copy. Only when the source is itself pinned is `non_blocking=True` genuinely asynchronous.
    3. Pinned memory cannot be swapped out or reclaimed as page cache, so pinning a lot leaves the machine less for the page cache and other processes and can make memory tight; allocating it is slow too (every page has to be locked and registered with the driver or the network card), so it is usually allocated once and reused. An ordinary process is also limited in how much it can lock by `ulimit -l` (RLIMIT_MEMLOCK), which constrains RDMA's memory registration.
    4. GPUDirect RDMA lets the network card read and write GPU memory directly, removing the "device memory to CPU memory to network card" staging copy (which NCCL's cross-machine communication and KV transfer both use); GPUDirect Storage lets an NVMe drive DMA straight into device memory, removing the chain of copies in CPU memory, "drive to page cache to user buffer to pinned buffer to device memory".
    5. NUMA (non-uniform memory access): on a multi-socket server each CPU socket has its own memory controller, so reaching that socket's memory is fast while reaching the other socket's goes over the interconnect with higher latency and lower bandwidth; PCIe devices hang off one socket too. A worker on the wrong socket reads remote memory while preparing its input, and copying to the GPU crosses the sockets before the PCIe, so both the bandwidth and the latency suffer, and with several workers contending for the interconnect the jitter grows.

## DMA: the device moves the data itself {#dma设备自己搬数据}

![Figure: the three paths of pageable memory, pinned memory and GPUDirect, plus NUMA](../assets/figures/pinned-dma.svg){.aig-svg}

Copying a block of data from CPU memory to a GPU, what really moves it is the copy engine on the GPU: the CPU tells it "start at this physical address and move this many bytes", it reads the memory itself over PCIe, and it signals the CPU when it is done. That is **DMA** (direct memory access). A network card sending and receiving and an NVMe drive reading and writing all work the same way.

DMA only knows physical addresses, while a process's memory is virtual: the physical page behind a virtual page may be swapped to disk, may be migrated by the kernel (compaction, NUMA balancing), or may not be allocated at all. So memory being DMAed has to be **pinned** (page-locked): every physical page allocated, never swapped and never migrated before the DMA finishes. The operating system's primitive is `mlock`:

```python title="mlock.py" ci="no"
import ctypes
import mmap

libc = ctypes.CDLL(None, use_errno=True)
MiB = 1 << 20


def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])            # in kB


buf = mmap.mmap(-1, 64 * MiB, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
before_lck, before_rss = status("VmLck"), status("VmRSS")
if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(64 * MiB)) != 0:   # pinned: every physical page allocated, and never swapped
    raise OSError(ctypes.get_errno(), "mlock 失败（检查 ulimit -l）")
print("VmLck 增加了", (status("VmLck") - before_lck) // 1024, "MiB")
print("mlock 顺带把这 64 MiB 全部分配好了，RSS 增加约 64 MiB：", 63 < (status("VmRSS") - before_rss) / 1024 < 66)
libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(64 * MiB))
print("解锁之后 VmLck 回到", status("VmLck") - before_lck, "kB")
```

```text title="output"
VmLck 增加了 64 MiB
mlock 顺带把这 64 MiB 全部分配好了，RSS 增加约 64 MiB： True
解锁之后 VmLck 回到 0 kB
```

CUDA pins memory its own way (`cudaHostAlloc`, `cudaHostRegister`, and `tensor.pin_memory()` or `torch.empty(..., pin_memory=True)` in PyTorch), on the same principle. Copying from pinned memory, the copy engine DMAs directly and the CPU returns at once, which can overlap with computation on the GPU; copying from ordinary memory, the driver first copies the data with the CPU into a pinned staging buffer of its own and then DMAs, and that step is synchronous:

| The source memory | How the copy goes | What `non_blocking=True` achieves |
| --- | --- | --- |
| pinned | the copy engine DMAs directly | genuinely asynchronous, able to overlap with computation |
| pageable | the CPU copies into the driver's pinned buffer, then DMAs | essentially synchronous, with one extra CPU copy |

So when an inference engine prepares each step's input (token ids, positions, the block table), it writes into preallocated pinned CPU buffers and copies them to the GPU asynchronously; and when the KV cache is offloaded to CPU memory, the buffer on the CPU side is pinned too.

What pinned memory costs:

- **it squeezes memory**: a pinned page cannot be swapped or reclaimed as page cache, so the more is pinned the less is left for the page cache and other processes;
- **it allocates slowly**: every page has to be allocated, locked and registered with the driver, and a large allocation routinely takes tens of milliseconds to seconds, so it is allocated once at startup and reused (PyTorch has an allocator that caches pinned memory);
- **there is a limit**: how much an ordinary process can `mlock` is bounded by `ulimit -l`. RDMA's memory registration (`ibv_reg_mr`) is subject to it, and the default inside a container is often tiny, so RDMA reports "Cannot allocate memory" there and needs `--ulimit memlock=-1` (CUDA's pinned allocations are not subject to this limit).

## Bandwidth: how long it takes to move the weights once {#带宽搬一次权重要多久}

The time to move 16 GB (an 8B model's BF16 weights) along various paths, with bandwidths taken at what is really achievable:

```python title="transfer.py"
# how long moving 16 GB of weights from CPU memory to the GPU takes along various paths (bandwidths at what is really achievable)
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

```text title="output"
PCIe 4.0 x16，锁页内存：640 ms
PCIe 4.0 x16，可换页内存（先拷进驱动的锁页缓冲区）：1,333 ms
PCIe 5.0 x16，锁页内存：320 ms
NVLink-C2C（Grace Hopper 的 CPU 到 GPU）：36 ms
400 Gb/s 网卡从远端读（GPUDirect RDMA）：356 ms
```

A few conclusions:

- loading from CPU memory (weights already in the page cache) to the GPU takes a fraction of a second on PCIe itself, and what is slow is usually the earlier steps: reading from the disk or network storage into memory, deserializing, and pageable memory's staging copy;
- a decode step is only tens of milliseconds, within which very little data can move between the CPU and the GPU. Offloading the KV cache to the CPU is possible, but bringing it back has to be weighed against recomputing it (see the exercises);
- machines connecting the CPU and the GPU with NVLink-C2C (Grace Hopper, GB200) make CPU memory to GPU an order of magnitude faster, which makes offloading to CPU memory far cheaper.

## GPUDirect: going around CPU memory {#gpudirect绕开-cpu-内存}

By default, data between a GPU and another device stages through CPU memory. GPUDirect is a set of technologies letting devices DMA to each other directly:

- **P2P**: GPUs in one machine read and write each other's memory directly over NVLink or PCIe, which is the path for tensor parallelism's all-reduce and for KV transfer within a machine in a prefill-decode split;
- **GPUDirect RDMA**: the network card reads and writes GPU memory directly. Cross-machine NCCL communication and cross-machine KV transfer (transfer engines like Mooncake and NIXL) depend on it, since otherwise every transfer stages "device memory to CPU memory to network card" (see [the RDMA programming model](serving://comm/rdma/) and [KV transfer engines and distributed storage](serving://comm/kv-storage/));
- **GPUDirect Storage**: an NVMe drive DMAs straight into device memory (through the cuFile interface). The ordinary read path is "drive to page cache to user buffer to pinned buffer to device memory", with two or three of those copies in CPU memory; GDS turns it into one DMA, which suits loading weights or KV quickly from local NVMe.

All of these paths require the devices to be "close" in the PCIe topology: GPUDirect RDMA's bandwidth is best when the network card and the GPU hang off the same PCIe switch, and crossing CPU sockets drops the performance markedly, which is something NCCL takes into account when choosing a network card. Which brings us to NUMA.

## NUMA: memory has near and far too {#numa内存也分远近}

On a multi-socket server, each CPU socket has its own memory controller and memory modules, with the sockets joined by an interconnect (Intel's UPI, AMD's Infinity Fabric). Reaching the local socket's memory is fast and reaching the other socket's goes over the interconnect, which is **non-uniform memory access** (NUMA), and each socket with its memory is a NUMA node. This machine happens to be a two-socket server, so first the default:

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

```text title="output (on this machine)"
单线程顺序读带宽 17.8 GB/s，随机访问延迟 121 ns
```

Then with `numactl` pinning the process to node 0's CPUs and its memory on node 0 and node 1 in turn:

```python title="numa.py" ci="no"
import glob
import subprocess

for node in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
    cpus = open(f"{node}/cpulist").read().strip()
    print(f"{node.rsplit('/', 1)[1]}：CPU {cpus}")

# the process pinned to node 0's CPUs, with its memory local (node 0) and remote (node 1) in turn
for mem in (0, 1):
    out = subprocess.run(["numactl", "--cpunodebind=0", f"--membind={mem}", "./numabw"],
                         capture_output=True, text=True, check=True).stdout.strip()
    print(f"CPU 在节点 0、内存在节点 {mem}（{'本地' if mem == 0 else '远端'}）：{out}")
```

```text title="output (on this machine)"
node0：CPU 0-15
node1：CPU 16-31
CPU 在节点 0、内存在节点 0（本地）：单线程顺序读带宽 17.7 GB/s，随机访问延迟 121 ns
CPU 在节点 0、内存在节点 1（远端）：单线程顺序读带宽 14.3 GB/s，随机访问延迟 186 ns
```

Remote memory's latency is about half again as high and its bandwidth lower; and with several processes reaching across at once, the interconnect's bandwidth is contended as well. Linux's default memory policy is **first touch**: a page is allocated on the node of the CPU that first writes it. So a process that initializes its buffers on node 0 and is later scheduled onto node 1 has entirely remote memory (the kernel's automatic NUMA balancing migrates it slowly, but not promptly).

A PCIe device belongs to a NUMA node too. To see it: `/sys/bus/pci/devices/<PCI address>/numa_node`, or `nvidia-smi topo -m`, which lists each GPU's CPU affinity and NUMA affinity and how GPUs connect to each other and to the network cards (`NV#` is NVLink, `PIX` / `PXB` is under the same PCIe switch, `NODE` is within one NUMA node through the CPU, and `SYS` is across sockets). An 8-GPU server usually has GPUs 0-3 under node 0 and 4-7 under node 1.

What an inference service does is pin each GPU worker to its GPU's node: the CPUs from that node, the memory allocated from that node. vLLM's `--numa-bind` (configured in `vllm/config/parallel.py`) starts each worker through `numactl --cpunodebind` and `--membind`, with the node detected automatically, specified by hand with `--numa-bind-nodes` (`[0, 0, 1, 1]`, say), or given as particular cores with `--numa-bind-cpus`.

!!! interview "How to explain it"
    To explain "why use pinned memory": DMA only knows physical addresses and ordinary memory's pages may be swapped or migrated, so the driver first copies into a pinned buffer of its own and DMAs from there, synchronously; only when the source is itself pinned can the copy engine DMA directly and genuinely overlap with computation. An inference engine's per-step input buffers and the CPU buffers for offloaded KV are all pinned, allocated once and reused. Then the costs: pinned memory cannot be swapped or reclaimed, allocates slowly, and RDMA's registration is bounded by `ulimit -l`. Pressed on NUMA: on a multi-socket server the memory and the PCIe devices belong to different sockets, with remote memory measured here at half again the latency and a fifth less bandwidth, so GPU workers are pinned to their GPU's node with `numactl` (vLLM's `--numa-bind`) and the network card chosen under the same PCIe switch as the GPU.

## Exercises {#练习}

**1. The `non_blocking` trap.** What is wrong with the code below? And what if the first line becomes `buf = torch.empty(n, pin_memory=True)`?

```python
buf = torch.empty(n)                        # the input buffer on the CPU
for step in range(steps):
    buf.copy_(next_inputs(step))            # prepare this step's input
    x = buf.to("cuda", non_blocking=True)
    y = model(x)
```

??? success "Answer"
    `buf` is pageable, so `to("cuda", non_blocking=True)` really has the CPU copy it synchronously into the driver's staging buffer; the code is correct but overlaps nothing, and every step waits out a copy for no reason. Made pinned, the copy becomes genuinely asynchronous and a problem appears: the DMA may not have finished when `to()` returns, and the next iteration's `buf.copy_(...)` starts overwriting `buf`, so the GPU may read part of the next step's data. The correct approach rotates two (or more) pinned buffers and confirms that the copy using one has finished before overwriting it (record a CUDA event and `event.synchronize()` before overwriting, or synchronize the copy's stream against the compute stream properly). That is exactly how an inference engine prepares its inputs.

**2. Bring it back or recompute it.** A 70B model (80 layers, GQA with 8 KV heads, head dimension 128, BF16) has one request whose 32K tokens of context were offloaded to CPU memory. How long does bringing it back take over PCIe 5.0 (about 50 GB/s)? And recomputing the prefill of those 32K tokens instead (estimating about 600 TFLOPS of effective compute per card)?

??? success "Answer"
    Per token the KV is 80 layers × 8 heads × 128 dimensions × 2 (K and V) × 2 bytes = 327,680 bytes ≈ 0.31 MiB, so 32K tokens is 10 GiB and bringing it back takes about 10.7 GB / 50 GB/s ≈ 0.21 seconds. Recomputing the prefill takes about 2 × 70e9 × 32768 ≈ 4.6e15 floating-point operations, which at 600 TFLOPS per card is about 7.6 seconds, or around 1 second with 8-way tensor parallelism. For a long context, bringing it back is far cheaper than recomputing; for a short one (a few hundred tokens) the recomputation is tiny and batches with other requests while the transfer has a fixed overhead, so recomputing wins. That is the basic arithmetic of tiered KV caching (see [tiered KV caching and offloading](serving://distributed/kv-offload/)).

**3. The wrong node.** On a two-socket, 8-GPU server, GPUs 4-7 hang off NUMA node 1. The worker for GPU 5 is scheduled by the operating system onto node 0's CPUs and its input buffers are in node 0's memory. What gets slower? How do you fix it?

??? success "Answer"
    Preparing the input on the CPU reads and writes local memory, which is unaffected, but anything else the worker touches (the shared-memory message queue near GPU 5, if it sits on node 1) is a remote access; more importantly, every copy of the input from node 0's memory to GPU 5 has the DMA cross the socket interconnect before reaching GPU 5's PCIe, with lower bandwidth and higher latency, and with every worker doing this the interconnect becomes a bottleneck and the latency jitter grows. The fix: start the worker under `numactl --cpunodebind=1 --membind=1` to pin it to node 1 (which vLLM's `--numa-bind` does automatically), and make sure the pinned buffers are allocated after the binding (first touch decides where the pages land).

## Summary {#小结}

- [x] DMA is a device moving data itself by physical address, and the memory being moved has to be pinned; `mlock`, `cudaHostAlloc` and `pin_memory()` all do this.
- [x] Only a copy from pinned memory makes `non_blocking=True` genuinely asynchronous; pageable memory goes through the driver's staging buffer and is essentially synchronous. A pinned buffer has to stay unchanged until its copy finishes.
- [x] Pinned memory squeezes the page cache, allocates slowly and is bounded by `ulimit -l` (for RDMA registration), so it is allocated once and reused.
- [x] 16 GB of weights over PCIe 5.0 is about 0.3 seconds; bringing a long context's KV back is far cheaper than recomputing it. GPUDirect lets a GPU DMA directly to and from network cards, NVMe and other GPUs.
- [x] NUMA: remote memory measured here at half again the latency and about a fifth less bandwidth; GPU workers are pinned to their GPU's node (`numactl`, vLLM's `--numa-bind`), and since memory is allocated on first touch, the binding has to come before the allocation.
