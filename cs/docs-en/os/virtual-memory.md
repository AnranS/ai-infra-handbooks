# Virtual memory, page tables and huge pages

<p class="lead">Every address a process sees is virtual, and every memory access has the CPU translate a virtual address into a physical one first. That machinery decides a great deal about inference: why mallocing a large block costs almost nothing, why mmaping a weight file of tens of GB finishes instantly, why a process gets killed by the OOM killer, why random access into a large array on a CPU is slow. PagedAttention's "paged KV cache" borrowed its idea from here directly. This chapter covers address translation and the four-level page table, on-demand allocation and page faults, the TLB and huge pages, and ends by setting PagedAttention beside them and looking at virtual memory on a GPU.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How is a virtual address translated into a physical one on x86-64? How many levels does the page table have?
    2. After mallocing or mmaping a large block, when is physical memory really allocated? What are the two kinds of page fault?
    3. What is the TLB? Why do huge pages make random access faster?
    4. What is overcommit? Why does the OOM killer kill a process? Does GPU memory have anything similar?
    5. Which of virtual memory's ideas did PagedAttention borrow? How does it differ from the operating system's paging?

??? success "Answers (try it yourself first, then expand)"
    1. A 48-bit virtual address splits into four 9-bit indices and a 12-bit offset within the page. The CPU finds the fourth-level table from the CR3 register and walks four levels with the four indices, and the last level's entry gives the physical page's start, which plus the offset is the physical address. Each level holds 512 entries of 8 bytes, exactly one page.
    2. Only the virtual addresses are reserved, and physical memory is allocated the first time a page is written: the CPU finds no mapping, raises a page fault, and the kernel allocates a physical page, fills the table in and re-executes the instruction. A fault needing no disk read is a minor fault (writing anonymous memory the first time); one that has to read from disk is a major fault (an mmaped file not in the page cache).
    3. The TLB is a small table in the CPU caching "virtual page to physical page" translations, and a hit skips the four-level walk. Its entries are limited (about two thousand in the second-level TLB), so 4 KiB pages cover only about 8 MiB and random access into a large array misses nearly every time and walks the page table; a 2 MiB huge page makes the same entries cover 4 GiB and skips a level besides.
    4. Overcommit allows processes to reserve more virtual memory in total than the physical memory plus swap, betting that they will not all use it at once. When they do and the kernel cannot reclaim anything, the OOM killer picks a process by score and kills it; exceeding a cgroup's memory limit in a container triggers it too. GPU memory has no on-demand allocation: `cudaMalloc` takes physical memory there and then and returns OOM when there is not enough (unified memory, UVM, excepted, though inference rarely uses it).
    5. What was borrowed: fixed-size blocks (pages), a block table (the page table) mapping a logical position to a physical one, on-demand allocation, reference counts plus copy-on-write (prefix sharing, parallel sampling), and swapping out or recomputing when memory runs short. What differs: there is no hardware TLB, so the attention kernel reads the block table explicitly for an indirection every time; a block holds several tokens' K/V (16 usually), so only each sequence's last block has internal fragmentation and there is no external fragmentation.

## Address translation: the four-level page table {#地址翻译四级页表}

x86-64 uses 48-bit virtual addresses (the newest CPUs support 57 bits and five levels) with 4 KiB pages. An address splits into four 9-bit page table indices and a 12-bit offset within the page:

```python title="translate.py"
# x86-64's four-level page table: a 48-bit virtual address = four 9-bit page table indices + a 12-bit offset within the page (4 KiB pages)
addr = 0x7F3A_1C2D_5E6F

offset = addr & 0xFFF
idx = [(addr >> shift) & 0x1FF for shift in (39, 30, 21, 12)]
for name, i in zip(["PML4（第 4 级）", "PDPT（第 3 级）", "PD（第 2 级）", "PT（第 1 级）"], idx):
    print(f"{name}下标：{i}")
print(f"页内偏移：{offset:#x}")
print(f"一张页表 512 项 × 8 字节 = {512 * 8} 字节，正好一页")
print(f"一个 2 MiB 大页直接由第 2 级页表项指向，偏移占低 {21} 位，少查一级")
```

```text title="output"
PML4（第 4 级）下标：254
PDPT（第 3 级）下标：232
PD（第 2 级）下标：225
PT（第 1 级）下标：213
页内偏移：0xe6f
一张页表 512 项 × 8 字节 = 4096 字节，正好一页
一个 2 MiB 大页直接由第 2 级页表项指向，偏移占低 21 位，少查一级
```

Change the address and the page size and watch how those indices are cut out:

<div class="aig-widget" data-widget="pagewalk"></div>

The translation is called a **page walk**: the CPU takes the fourth-level table's physical address from CR3, finds an entry with the first index, which holds the next level's address, and four lookups later has the physical page's address, plus the offset. Besides an address, every entry carries flags: present, writable, user-accessible, accessed, and dirty (written). Copy-on-write, on-demand allocation and swapping are all implemented from these flags plus the page fault.

Four memory lookups per memory access would clearly be too slow. The CPU caches recent translations in the **TLB** (Translation Lookaside Buffer), where a hit translates in one cycle. Switching processes switches the page table (writing CR3) and invalidates the old entries; modern CPUs tag entries with a PCID so the whole TLB need not be flushed, which is part of the context switch's indirect cost from the previous chapter.

## On-demand allocation and page faults {#按需分配与缺页}

Mallocing a large block (glibc mmaps anything over 128 KiB directly) only marks out a range of the process's virtual address space, with no mapping in the page table. The first write to a page finds no mapping, raises a **page fault**, and only then does the kernel allocate a physical page, zero it, fill the table in and re-execute the instruction:

```python title="faults.py"
import mmap
import resource

MiB = 1 << 20
SIZE = 256 * MiB


def minor_faults():
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def touch(buf, step):
    for off in range(0, SIZE, step):     # one byte written per page: physical memory is allocated only on the first write to a page
        buf[off] = 1


PRIVATE = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS   # a private anonymous mapping, the same as mallocing a large block gives
buf = mmap.mmap(-1, SIZE, flags=PRIVATE)  # only 256 MiB of virtual addresses reserved, with no physical pages yet
before = minor_faults()
touch(buf, 4096)
print(f"普通页：写满 256 MiB 触发了约 {round((minor_faults() - before) / 1000)} 千次缺页")

buf2 = mmap.mmap(-1, SIZE, flags=PRIVATE)
buf2.madvise(mmap.MADV_HUGEPAGE)          # advise the kernel to use transparent huge pages (2 MiB)
before = minor_faults()
touch(buf2, 4096)
print(f"透明大页：同样写满 256 MiB 触发了 {minor_faults() - before} 次缺页")
```

```text title="output (on this machine)"
普通页：写满 256 MiB 触发了约 66 千次缺页
透明大页：同样写满 256 MiB 触发了 639 次缺页
```

256 MiB is exactly 65,536 pages of 4 KiB, each faulting once on its first write. With transparent huge pages the kernel allocates 2 MiB at a time and the fault count drops to a few hundred (the unaligned ends of the mapping still use small pages, and the Python interpreter faults on its own account too).

There are two kinds of fault:

- **a minor fault**: no disk read, as when anonymous memory is written the first time, or the file's page is already in the page cache and merely not mapped in yet;
- **a major fault**: data has to be read from disk, as when an mmaped weight file is touched for the first time and is not in the page cache. One major fault is one disk I/O, orders of magnitude slower. [The next chapter](pinned-numa.md) and [how a write reaches the disk](io-stack.md) use this distinction.

**A virtual address is not memory use.** VSZ in `ps` is the size of the virtual address space and RSS (the resident set) is the physical memory really held. Linux allows overcommit by default:

```python title="vmsize.py"
import mmap

GiB = 1 << 30
MAP_NORESERVE = 0x4000          # the value on Linux; Python's mmap module does not export this constant


def rss_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE   # reserve no swap for this range, allowing overcommit
before = rss_mib()
m = mmap.mmap(-1, 100 * GiB, flags=flags)      # larger than this machine's physical memory
print("保留 100 GiB 的虚拟地址：成功")
print("常驻内存（RSS）几乎没变：", rss_mib() - before < 1)
m[: 64 << 20] = b"\1" * (64 << 20)                # really write 64 MiB
print("写了 64 MiB 之后 RSS 约增加 64 MiB：", 63 < rss_mib() - before < 66)
```

```text title="output"
保留 100 GiB 的虚拟地址：成功
常驻内存（RSS）几乎没变： True
写了 64 MiB 之后 RSS 约增加 64 MiB： True
```

`vm.overcommit_memory` controls this: 0 (the default) refuses obviously excessive requests heuristically, 1 always allows, and 2 accounts strictly by "physical memory × a ratio + swap". Overcommit's cost is that memory may run out when everyone really does write: the kernel reclaims the page cache and swaps first, and failing that starts the **OOM killer**, which picks a process by score (mostly its memory use, adjustable through `/proc/<pid>/oom_score_adj`). An inference service killed by the OOM killer usually underestimated its CPU memory: several workers each reading the weights in, the tokenizer's and the request queue's overhead, the buffer for KV offloaded to the CPU. A container also has the cgroup's memory limit, see [containers](containers.md).

## The TLB and huge pages {#tlb-与大页}

The TLB's entries are limited. Estimating from a server CPU's typical second-level TLB:

```python title="reach.py"
# the TLB's reach = entries x page size. 2048 entries is typical of a server CPU's second-level TLB (STLB)
entries = 2048
for name, page in [("4 KiB 页", 4 << 10), ("2 MiB 大页", 2 << 20)]:
    print(f"{name}：{entries} 项能覆盖 {entries * page / (1 << 20):,.0f} MiB")

# a 70B model's BF16 weights are 140 GB: how many pages one pass touches when offloading to the CPU or running inference there
weights = 140e9
print(f"140 GB 的权重：4 KiB 页约 {weights / 4096 / 1e6:.0f} 百万个，2 MiB 页约 {weights / (2 << 20) / 1e3:.0f} 千个")
```

```text title="output"
4 KiB 页：2048 项能覆盖 8 MiB
2 MiB 大页：2048 项能覆盖 4,096 MiB
140 GB 的权重：4 KiB 页约 34 百万个，2 MiB 页约 67 千个
```

With 4 KiB pages the TLB covers 8 MiB; random access into a 1 GiB array misses nearly every time and needs an extra page walk (whose own tables may not be in cache either). With 2 MiB huge pages the same entries cover 4 GiB and there is one level less. Below, a 1 GiB array is read in random jumps (each value read decides where to read next, so the CPU cannot prefetch):

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

```text title="output (on this machine)"
4 KiB 页：每次随机访问 145 ns
2 MiB 透明大页：每次随机访问 93 ns（快了 36%）
```

Two ways to use them:

- **transparent huge pages** (THP): the kernel uses 2 MiB pages automatically. `/sys/kernel/mm/transparent_hugepage/enabled` is `always` (every anonymous mapping tries), `madvise` (only regions that called `madvise(MADV_HUGEPAGE)`, which is this machine's setting) or `never`. The cost is that allocating a huge page may have to compact memory, which causes occasional latency spikes, so services sensitive to tail latency (many databases and KV stores) are advised to turn it off;
- **explicit huge pages** (hugetlbfs): reserved at boot (`vm.nr_hugepages`), 2 MiB or 1 GiB, never swapped and never needing compaction, but requiring the capacity to be planned in advance. The userspace networking and storage stacks (DPDK, SPDK) and large RDMA-registered buffers commonly use them.

## Page tables and PagedAttention {#页表与-pagedattention}

PagedAttention manages the KV cache in fixed-size blocks, borrowing virtual memory's design almost one for one (the implementation is in [the paged KV cache](serving://engine/paged-kv/)):

| Virtual memory | PagedAttention |
| --- | --- |
| a page (4 KiB) | a KV block (16 tokens' K and V, usually) |
| the page table: a virtual page number to a physical page | the block table: the nth block of a sequence to a physical block in device memory |
| on-demand allocation: a page is allocated when it is written | a block is allocated when the sequence grows into one |
| copy-on-write (pages shared after a fork, copied on a write) | a shared prefix and parallel sampling share blocks, with reference counts and copy-on-write |
| swapping to disk when memory runs short | preemption when device memory runs short: swapped to CPU memory, or dropped and recomputed |

What differs: a GPU has no TLB translating KV blocks, so the attention kernel reads the block table explicitly every time and fetches the K/V by block number, and that indirection's cost has to be amortized by the kernel's design (reading a whole block at once, keeping the block table in shared memory); and the block size is a trade-off between "the last block's internal fragmentation" and "a long block table with scattered accesses".

## Virtual memory on a GPU {#gpu-上的虚拟内存}

A GPU has page tables and an MMU of its own, and every CUDA context has its own virtual address space. Unlike a CPU, `cudaMalloc` allocates physical memory there and then, with no on-demand allocation, and fails outright when there is not enough. A few mechanisms related to virtual memory:

- **unified memory** (UVM, `cudaMallocManaged`): the CPU and the GPU share one range of virtual addresses, and a GPU access to a page not in device memory faults and the driver migrates it. Convenient, but fault-driven migration is very slow and inference's hot path rarely uses it;
- **the virtual memory management API** (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`): reserve a range of virtual addresses first, then map physical memory into it or unmap it, with the address unchanged. PyTorch's `expandable_segments:True` (set in `PYTORCH_CUDA_ALLOC_CONF`) uses it to let a segment grow in place and cut fragmentation; vLLM's sleep mode (`CuMemAllocator` in `vllm/device_allocator/cumem.py`) uses it to release the weights' and the KV cache's physical memory on `sleep()` (or offload it to the CPU first) and map it back **at the same virtual addresses** on `wake_up()`, so the pointers a CUDA Graph remembers stay valid (why that matters is in [updating weights live](serving://ops/weight-update/)). In RL training, where the inference engine and the training share one card, this is what shuffles memory between them.

!!! interview "Answering in an interview"
    Asked "how does PagedAttention relate to the operating system's virtual memory": start with the problem, that how long each request will be is not known in advance, so reserving the maximum wastes a great deal of memory internally while requests of differing lengths fragment it externally. Then map them one for one: a KV block is a page, the block table is the page table, and on-demand allocation, reference counts and copy-on-write (prefix sharing, parallel sampling) are all taken from virtual memory, with swapping or recomputing when memory runs short standing in for paging. Finish with the differences: there is no hardware TLB, so the kernel reads the block table for an explicit indirection; the block size (16 tokens commonly) trades internal fragmentation against the addressing cost; and the only fragmentation left is the spare slots in each sequence's last block. Adding that a GPU has virtual memory of its own earns credit: vLLM's sleep mode releases and remaps physical memory through CUDA's virtual memory API with the virtual addresses unchanged, so no CUDA Graph has to be recaptured.

## Exercises {#练习}

**1. mmaping a weight file.** Four tensor-parallel workers all mmap a 16 GB safetensors file, and each reads only its own 4 GB slice and copies it to the GPU. Roughly how much does each worker's RSS grow during the copy? Roughly how much does the whole machine's memory use grow? What happens to that memory afterwards?

??? success "Answer"
    The 4 GB each worker reads is mapped into its address space by faults, and those file pages count towards its RSS (as shared file pages), so each worker's RSS grows by about 4 GB. But those pages are the same copy in the **page cache**: a file's pages are held once in memory and every mmap maps the same physical pages, and since the four workers' slices do not overlap, the whole machine gains about 16 GB of page cache rather than 4 × 16 GB. After the copy to the GPU the page cache remains (making the next load faster), and being clean file pages the kernel can simply drop them under memory pressure without causing an OOM; had each worker `read` the whole file into a buffer of its own, that would really be four copies of anonymous memory.

**2. Latency spikes.** A KV cache storage service gained a little throughput after transparent huge pages were set to `always`, but its P99 latency occasionally jumps from 1 ms to tens of milliseconds. What could it be? How do you verify and handle it?

??? success "Answer"
    A transparent huge page needs 2 MiB of contiguous physical memory, and once memory is fragmented the fault handler may compact memory synchronously (direct compaction), moving a great many pages, which is where the spikes come from; the background khugepaged merging small pages adds jitter too, and huge pages also increase the memory used (writing one byte allocates 2 MiB). To verify: see whether `compact_stall` and `thp_fault_fallback` in `/proc/vmstat` grow during the spikes, or use `perf` to see how long `compact_zone` takes in the fault path. To handle it: set it back to `madvise` and call `madvise(MADV_HUGEPAGE)` only on the buffers that need it (a preallocated KV block pool); or switch to explicit huge pages reserved at boot and avoid run-time compaction altogether.

**3. The block size.** Why is vLLM's KV block usually 16 tokens rather than 1 or 1024?

??? success "Answer"
    The larger the block, the more each sequence's last block wastes on average, half a block: 1024-token blocks waste about 512 tokens' KV per request on average, which with any concurrency is a great deal of memory, and prefix sharing gets coarser (only whole identical blocks can be shared). The smaller the block, the longer the block table, so the attention kernel does an indirection for every short run of K/V, the accesses scatter, the bandwidth utilization falls, and blocks are allocated and freed more often. 16 tokens is the compromise: very little memory wasted, and enough K/V stored contiguously in one block for the kernel to read at a decent granularity. Different attention back ends have their own requirements (some want 64 or more), so the value is configurable.

## Summary {#小结}

- [x] Address translation: a 48-bit virtual address is four 9-bit indices plus a 12-bit offset, walked through four levels; the TLB caches the translations and a PCID avoids flushing it on a process switch.
- [x] On-demand allocation: malloc / mmap only reserve virtual addresses, and the first write faults and allocates the physical page; a minor fault reads no disk and a major fault does.
- [x] RSS is the real use; overcommit allows reserving more than exists, and when it really runs out the OOM killer picks a process by score.
- [x] Huge pages: 2 MiB pages let the TLB cover 4 GiB, which made random access on this machine over 30% faster; transparent huge pages can bring compaction's latency spikes, so `always` deserves care in a service sensitive to tail latency.
- [x] PagedAttention took paging's ideas wholesale (blocks, a block table, on-demand allocation, copy-on-write, swapping) without a hardware TLB; and a GPU's virtual memory API is what lets vLLM's sleep mode release memory while keeping the virtual addresses.
