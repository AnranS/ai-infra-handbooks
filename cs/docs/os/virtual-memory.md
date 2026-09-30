# 虚拟内存、页表与大页

<p class="lead">进程看到的每一个地址都是虚拟的，CPU 每访问一次内存，都要先把虚拟地址翻译成物理地址。这套机制决定了很多和推理有关的事：malloc 一大块内存为什么几乎不花时间，mmap 一个几十 GB 的权重文件为什么瞬间完成，进程为什么会被 OOM killer 杀掉，CPU 上随机访问大数组为什么慢。PagedAttention 的"分页 KV Cache"也是直接从这里借来的想法。这一章讲地址翻译和四级页表、按需分配与缺页、TLB 与大页，最后对照 PagedAttention，并看看 GPU 上的虚拟内存。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. x86-64 上一个虚拟地址是怎么翻译成物理地址的？页表有几级？
    2. malloc 或 mmap 一大块内存之后，物理内存什么时候才真正分配？缺页分哪两种？
    3. TLB 是什么？为什么大页能让随机访问变快？
    4. 什么是 overcommit？进程为什么会被 OOM killer 杀掉？GPU 显存有没有类似的机制？
    5. PagedAttention 借用了虚拟内存的哪些想法？和操作系统的分页有什么不同？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 48 位虚拟地址拆成 4 个 9 位的下标和 12 位的页内偏移。CPU 从 CR3 寄存器找到第 4 级页表，依次用 4 个下标查 4 级页表，最后一级的表项给出物理页的起始地址，加上偏移就是物理地址。每级页表 512 项、每项 8 字节，正好占一页。
    2. 只保留了虚拟地址，物理内存在第一次写某一页时才分配：CPU 查页表发现没有映射，触发缺页异常，内核分配一个物理页、填好页表，再让指令重新执行。不需要读磁盘的是次缺页（minor fault），比如第一次写匿名内存；需要从磁盘读数据的是主缺页（major fault），比如 mmap 的文件不在页缓存里。
    3. TLB 是 CPU 里缓存"虚拟页 → 物理页"翻译结果的小表，命中时不用查四级页表。它的表项数有限（二级 TLB 约两千项），4 KiB 页只能覆盖约 8 MiB，随机访问大数组时几乎每次都缺失、都要做一次页表遍历；2 MiB 的大页让同样的表项覆盖 4 GiB，还少查一级页表。
    4. overcommit 是允许进程申请的虚拟内存总量超过物理内存加交换空间，赌大家不会同时用满。真用满了、内核又回收不出内存时，OOM killer 按评分挑一个进程杀掉；容器里超过 cgroup 的内存上限也会触发。GPU 显存没有按需分配：`cudaMalloc` 当场就要拿到物理显存，不够就直接返回 OOM（统一内存 UVM 例外，但推理里很少用）。
    5. 借用的想法：固定大小的块（页）、块表（页表）把逻辑位置映射到物理位置、按需分配、引用计数加写时复制（前缀共享、并行采样），显存不够时换出或重算。不同之处：没有硬件 TLB，注意力 kernel 每次都要显式读块表做一次间接寻址；块里存的是若干个 token 的 K/V（通常 16 个），只有每个序列最后一个块有内部碎片，没有外部碎片。

## 地址翻译：四级页表

x86-64 用 48 位虚拟地址（最新的 CPU 支持 57 位、五级页表），页大小 4 KiB。一个地址拆成 4 个 9 位的页表下标和 12 位的页内偏移：

```python title="translate.py"
# x86-64 四级页表：48 位虚拟地址 = 4 个 9 位的页表下标 + 12 位的页内偏移（页大小 4 KiB）
addr = 0x7F3A_1C2D_5E6F

offset = addr & 0xFFF
idx = [(addr >> shift) & 0x1FF for shift in (39, 30, 21, 12)]
for name, i in zip(["PML4（第 4 级）", "PDPT（第 3 级）", "PD（第 2 级）", "PT（第 1 级）"], idx):
    print(f"{name}下标：{i}")
print(f"页内偏移：{offset:#x}")
print(f"一张页表 512 项 × 8 字节 = {512 * 8} 字节，正好一页")
print(f"一个 2 MiB 大页直接由第 2 级页表项指向，偏移占低 {21} 位，少查一级")
```

```text title="输出"
PML4（第 4 级）下标：254
PDPT（第 3 级）下标：232
PD（第 2 级）下标：225
PT（第 1 级）下标：213
页内偏移：0xe6f
一张页表 512 项 × 8 字节 = 4096 字节，正好一页
一个 2 MiB 大页直接由第 2 级页表项指向，偏移占低 21 位，少查一级
```

换个地址、换个页大小，看这几段下标是怎么切出来的：

<div class="aig-widget" data-widget="pagewalk"></div>

翻译的过程叫**页表遍历**（page walk）：CPU 从 CR3 寄存器拿到第 4 级页表的物理地址，用第一个下标找到表项，表项里是下一级页表的地址……四次之后拿到物理页的地址，加上偏移。每级表项除了地址，还有一些标志位：是否存在、是否可写、是否允许用户态访问、是否被访问过、是否被写过（脏页）。写时复制、按需分配、换出，都是靠这些标志位加上缺页异常实现的。

每访问一次内存都要先查四次内存，显然太慢。CPU 用 **TLB**（Translation Lookaside Buffer）缓存最近的翻译结果，命中时一个周期就完成翻译。切换进程要换页表（写 CR3），旧的 TLB 表项就不能用了；现代 CPU 用 PCID 给表项打上进程标签，切换时不必整个清空，这也是上一章说的上下文切换间接开销的一部分。

## 按需分配与缺页

malloc 一大块内存（glibc 对大于 128 KiB 的请求直接用 mmap）只是在进程的虚拟地址空间里划出一段，页表里还没有任何映射。第一次写某一页时，CPU 找不到映射，触发**缺页异常**，内核这才分配一个物理页、清零、填好页表，然后让那条指令重新执行：

```python title="faults.py"
import mmap
import resource

MiB = 1 << 20
SIZE = 256 * MiB


def minor_faults():
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def touch(buf, step):
    for off in range(0, SIZE, step):     # 每页写一个字节：第一次写一页时才真正分配物理内存
        buf[off] = 1


PRIVATE = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS   # 私有的匿名映射，和 malloc 大块内存时拿到的一样
buf = mmap.mmap(-1, SIZE, flags=PRIVATE)  # 只是保留了 256 MiB 的虚拟地址，还没有物理页
before = minor_faults()
touch(buf, 4096)
print(f"普通页：写满 256 MiB 触发了约 {round((minor_faults() - before) / 1000)} 千次缺页")

buf2 = mmap.mmap(-1, SIZE, flags=PRIVATE)
buf2.madvise(mmap.MADV_HUGEPAGE)          # 建议内核用透明大页（2 MiB）
before = minor_faults()
touch(buf2, 4096)
print(f"透明大页：同样写满 256 MiB 触发了 {minor_faults() - before} 次缺页")
```

```text title="输出（本机示例）"
普通页：写满 256 MiB 触发了约 66 千次缺页
透明大页：同样写满 256 MiB 触发了 639 次缺页
```

256 MiB 正好是 65,536 个 4 KiB 页，每一页第一次写都缺页一次。透明大页的情况下，内核一次分配一个 2 MiB 的页，缺页次数降到几百次（映射的首尾不是按 2 MiB 对齐的部分仍然用小页，Python 解释器自己也会缺页）。

缺页分两种：

- **次缺页**（minor fault）：不需要读磁盘，比如第一次写匿名内存、或者要的文件页已经在页缓存里、只是还没映射进来；
- **主缺页**（major fault）：要从磁盘读数据，比如 mmap 的权重文件第一次被访问、又不在页缓存里。一次主缺页就是一次磁盘 I/O，慢几个数量级。[下一章](pinned-numa.md)和[一次写入如何落盘](io-stack.md)会用到这个区别。

**虚拟地址不等于内存占用。**`ps` 里的 VSZ 是虚拟地址空间的大小，RSS（常驻内存）才是实际占用的物理内存。Linux 默认允许超额申请（overcommit）：

```python title="vmsize.py"
import mmap

GiB = 1 << 30
MAP_NORESERVE = 0x4000          # Linux 上的取值；Python 的 mmap 模块没有导出这个常量


def rss_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE   # 不为这段地址预留交换空间，允许超额申请
before = rss_mib()
m = mmap.mmap(-1, 100 * GiB, flags=flags)      # 比这台机器的物理内存还大
print("保留 100 GiB 的虚拟地址：成功")
print("常驻内存（RSS）几乎没变：", rss_mib() - before < 1)
m[: 64 << 20] = b"\1" * (64 << 20)                # 真正写 64 MiB
print("写了 64 MiB 之后 RSS 约增加 64 MiB：", 63 < rss_mib() - before < 66)
```

```text title="输出"
保留 100 GiB 的虚拟地址：成功
常驻内存（RSS）几乎没变： True
写了 64 MiB 之后 RSS 约增加 64 MiB： True
```

`vm.overcommit_memory` 控制这个行为：0（默认）用启发式规则拒绝明显过分的申请，1 总是允许，2 严格按"物理内存 × 比例 + 交换空间"记账。超额申请的代价是，大家真的都去写的时候内存可能不够：内核先回收页缓存、换出，实在不行就启动 **OOM killer**，按每个进程的评分（主要看内存占用，可以用 `/proc/<pid>/oom_score_adj` 调整）挑一个杀掉。推理服务被 OOM killer 杀掉，通常是 CPU 内存没算够：多个 worker 各自把权重读进内存、分词器和请求队列的开销、KV Cache 卸载到 CPU 的缓冲区。容器里还有 cgroup 的内存上限，见[容器](containers.md)一章。

## TLB 与大页

TLB 的表项数有限。按一个服务器 CPU 二级 TLB 的典型规模估算：

```python title="reach.py"
# TLB 覆盖范围 = 表项数 × 页大小。2048 项是服务器 CPU 二级 TLB（STLB）的典型规模
entries = 2048
for name, page in [("4 KiB 页", 4 << 10), ("2 MiB 大页", 2 << 20)]:
    print(f"{name}：{entries} 项能覆盖 {entries * page / (1 << 20):,.0f} MiB")

# 一个 70B 模型的 BF16 权重有 140 GB：在 CPU 上做 offload 或者 CPU 推理时，扫一遍要碰多少个页
weights = 140e9
print(f"140 GB 的权重：4 KiB 页约 {weights / 4096 / 1e6:.0f} 百万个，2 MiB 页约 {weights / (2 << 20) / 1e3:.0f} 千个")
```

```text title="输出"
4 KiB 页：2048 项能覆盖 8 MiB
2 MiB 大页：2048 项能覆盖 4,096 MiB
140 GB 的权重：4 KiB 页约 34 百万个，2 MiB 页约 67 千个
```

4 KiB 页时，TLB 只能覆盖 8 MiB；随机访问一个 1 GiB 的数组，几乎每次都 TLB 缺失，要额外做一次页表遍历（页表本身也可能不在缓存里）。换成 2 MiB 的大页，同样的表项能覆盖 4 GiB，页表也少一级。下面在 1 GiB 的数组里随机跳着读（每次读到的值决定下一次读哪里，CPU 没法预取）：

```c title="tlb.c"
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

#define SIZE (1UL << 30)            /* 1 GiB，远大于缓存，也远大于 4 KiB 页时 TLB 能覆盖的范围 */
#define LINE 64                     /* 每个缓存行放一个"指向下一个缓存行"的下标 */
#define STEPS 20000000

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t rng = 88172645463325252ULL;
static uint64_t next_rand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

/* 随机的指针追逐：每次读到的值决定下一次读哪里，CPU 没法提前预取，每一步都是一次缓存缺失（和可能的 TLB 缺失） */
static double chase(uint32_t *perm, size_t n, int huge) {
  char *buf = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED) return -1;
  madvise(buf, SIZE, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  for (size_t i = 0; i < n; i++) *(uint32_t *)(buf + (size_t)perm[i] * LINE) = perm[(i + 1) % n];
  uint32_t cur = perm[0];
  double t0 = now();
  for (int s = 0; s < STEPS; s++) cur = *(uint32_t *)(buf + (size_t)cur * LINE);
  double t = now() - t0;
  if (cur == 0xFFFFFFFF) printf("%u\n", cur);   /* 让编译器不能把循环优化掉 */
  munmap(buf, SIZE);
  return t / STEPS * 1e9;
}

int main(void) {
  size_t n = SIZE / LINE;
  uint32_t *perm = malloc(n * sizeof *perm);
  if (!perm) return 1;
  for (size_t i = 0; i < n; i++) perm[i] = i;
  for (size_t i = n - 1; i > 0; i--) {         /* 随机打乱缓存行的访问顺序 */
    size_t j = next_rand() % (i + 1);
    uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  double small = chase(perm, n, 0), huge = chase(perm, n, 1);
  printf("4 KiB 页：每次随机访问 %.0f ns\n", small);
  printf("2 MiB 透明大页：每次随机访问 %.0f ns（快了 %.0f%%）\n", huge, (small - huge) / small * 100);
  free(perm);
  return 0;
}
```

```text title="输出（本机示例）"
4 KiB 页：每次随机访问 145 ns
2 MiB 透明大页：每次随机访问 93 ns（快了 36%）
```

两种用法：

- **透明大页**（THP）：内核自动用 2 MiB 的页。`/sys/kernel/mm/transparent_hugepage/enabled` 是 `always`（所有匿名内存都尝试）、`madvise`（只对调用了 `madvise(MADV_HUGEPAGE)` 的区域，本机就是这个设置）或 `never`。代价是分配大页时可能要整理内存碎片（直接内存规整），造成偶发的延迟尖刺，所以一些对尾延迟敏感的服务（很多数据库、KV 存储）建议关掉；
- **显式大页**（hugetlbfs）：启动时预留（`vm.nr_hugepages`），2 MiB 或 1 GiB，不会被换出、不需要规整，但要提前规划容量。DPDK、SPDK 这类用户态网络和存储栈、以及 RDMA 注册的大缓冲区常用它。

## 页表与 PagedAttention

PagedAttention 把 KV Cache 按固定大小的块管理，几乎一一对应地借用了虚拟内存的设计（实现见[分页 KV Cache](serving://engine/paged-kv/)）：

| 虚拟内存 | PagedAttention |
| --- | --- |
| 页（4 KiB） | KV 块（通常 16 个 token 的 K 和 V） |
| 页表：虚拟页号 → 物理页 | 块表：序列里的第几个块 → 显存里的物理块 |
| 按需分配：写到哪一页才分配哪一页 | 序列每长出一个块才分配一个块 |
| 写时复制（fork 后共享页，写时再复制） | 前缀共享、并行采样共享块，引用计数，写时复制 |
| 内存不够时换出到磁盘 | 显存不够时抢占：换出到 CPU 内存，或者丢掉以后重算 |

不同之处在于：GPU 上没有替 KV 块做翻译的 TLB，注意力 kernel 每次都要显式读块表、再按块号去取 K/V，这次间接寻址的开销要靠 kernel 设计（一次读一整块、块表放进共享内存）摊掉；块大小是在"最后一个块的内部碎片"和"块表太长、访存不连续"之间权衡的结果。

## GPU 上的虚拟内存

GPU 也有自己的页表和 MMU，每个 CUDA 上下文有独立的虚拟地址空间。和 CPU 不同的是，`cudaMalloc` 当场分配物理显存，没有按需分配，不够就直接失败。几种和虚拟内存有关的机制：

- **统一内存**（UVM，`cudaMallocManaged`）：CPU 和 GPU 共用一段虚拟地址，GPU 访问不在显存里的页时缺页，驱动把页迁移过来。方便，但缺页迁移很慢，推理的热路径里很少用；
- **虚拟内存管理 API**（`cuMemAddressReserve`、`cuMemCreate`、`cuMemMap`）：先保留一段虚拟地址，再把物理显存块映射上去或者解除映射，地址不变。PyTorch 的 `expandable_segments:True`（设置在 `PYTORCH_CUDA_ALLOC_CONF`）用它让一段显存原地增长，减少碎片；vLLM 的睡眠模式（`vllm/device_allocator/cumem.py` 里的 `CuMemAllocator`）用它在 `sleep()` 时释放权重和 KV Cache 的物理显存（或者先卸载到 CPU），`wake_up()` 时在**同样的虚拟地址**上重新映射——这样 CUDA Graph 里记住的指针仍然有效（为什么这很重要，见[权重热更新](serving://ops/weight-update/)）。RL 训练里推理引擎和训练共用一张卡时，就靠这个在两者之间腾挪显存。

!!! interview "面试怎么答"
    被问"PagedAttention 和操作系统的虚拟内存是什么关系"：先说问题，每个请求最终生成多长事先不知道，按最大长度预留显存会有大量内部碎片，各请求长度不一又带来外部碎片。然后一一对应：KV 块对应页，块表对应页表，按需分配、引用计数和写时复制（前缀共享、并行采样）都照搬了虚拟内存，显存不够时的换出或重算对应换页。最后讲不同：没有硬件 TLB，kernel 要显式读块表做间接寻址；块的大小（常见 16 个 token）是内部碎片和寻址开销之间的权衡；碎片只剩每个序列最后一个块里的空位。能再补一句 GPU 自己也有虚拟内存：vLLM 的睡眠模式用 CUDA 的虚拟内存 API 释放和重新映射物理显存，虚拟地址不变，CUDA Graph 不用重新捕获。

## 练习

**1. mmap 权重文件。** 一个 16 GB 的 safetensors 文件，4 个张量并行的 worker 都 mmap 它，每个 worker 只读自己那 4 GB 的切片并拷到 GPU 上。拷贝过程中每个 worker 的 RSS 大约涨多少？整台机器的内存占用大约涨多少？读完之后这些内存会怎样？

??? success "参考答案"
    每个 worker 读到的 4 GB 通过缺页映射进它的地址空间，这部分文件页计入它的 RSS（作为共享的文件页），所以每个 worker 的 RSS 大约涨 4 GB。但这些页是**页缓存**里的同一份：文件页只在内存里存一份，谁 mmap 都映射到同一份物理页，4 个 worker 读的切片不重叠，整台机器大约增加 16 GB 的页缓存，而不是 4 × 16 GB。读完、拷到 GPU 之后，页缓存仍然保留（下次加载更快），它们是干净的文件页，内存紧张时内核可以直接丢掉，不会导致 OOM；如果每个 worker 用 `read` 把整个文件读进自己的缓冲区，那才是真正的 4 份匿名内存。

**2. 延迟尖刺。** 一个 KV Cache 存储服务在把透明大页设成 `always` 之后，吞吐略有提升，但 P99 延迟偶尔从 1 ms 跳到几十毫秒。可能的原因是什么？怎么验证和处理？

??? success "参考答案"
    透明大页在分配时需要连续的 2 MiB 物理内存，内存碎片化以后，缺页处理里可能同步做内存规整（direct compaction），一次要搬很多页，这就是尖刺的来源；后台的 khugepaged 合并小页时也会带来抖动，大页还会让内存占用变大（写一个字节也分配 2 MiB）。验证：看 `/proc/vmstat` 里的 `compact_stall`、`thp_fault_fallback` 在尖刺时是否增长，或者用 `perf` 看缺页路径里 `compact_zone` 的耗时。处理：把设置改回 `madvise`，只对确实需要的大缓冲区（比如预先分配好的 KV 块池）调用 `madvise(MADV_HUGEPAGE)`；或者改用启动时预留的显式大页，彻底避开运行时规整。

**3. 块大小。** 为什么 vLLM 的 KV 块通常是 16 个 token，而不是 1 个或 1024 个？

??? success "参考答案"
    块越大，每个序列最后一个块平均浪费半个块：1024 个 token 的块每个请求平均浪费约 512 个 token 的 KV，并发一多就是大量显存，前缀共享的粒度也变粗（只有整块相同才能共享）。块越小，块表越长，注意力 kernel 每读一小段 K/V 都要做一次间接寻址，访存不连续，带宽利用率下降，块的分配和回收也更频繁。16 个 token 是两者的折中：浪费的显存很少，一个块里连续存放的 K/V 又足够让 kernel 以较大的粒度读取。不同的注意力后端对块大小有自己的要求（有的要求 64 或更大），所以这个值是可配置的。

## 小结

- [x] 地址翻译：48 位虚拟地址 = 4 个 9 位下标 + 12 位偏移，逐级查四级页表；TLB 缓存翻译结果，PCID 让切换进程时不必清空。
- [x] 按需分配：malloc / mmap 只保留虚拟地址，第一次写时缺页才分配物理页；次缺页不读磁盘，主缺页要读。
- [x] RSS 才是真实占用；overcommit 允许超额申请，真用满时 OOM killer 按评分杀进程。
- [x] 大页：2 MiB 页让 TLB 覆盖 4 GiB，本机随机访问快了三成多；透明大页可能带来规整造成的延迟尖刺，对尾延迟敏感的服务慎用 `always`。
- [x] PagedAttention 照搬了分页的思想（块、块表、按需分配、写时复制、换出），但没有硬件 TLB；GPU 的虚拟内存 API 让 vLLM 睡眠模式释放显存时保持虚拟地址不变。
