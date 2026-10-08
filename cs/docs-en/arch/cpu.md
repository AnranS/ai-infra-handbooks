# A CPU architecture crash course: pipelines, caches and coherence

<p class="lead">A great deal of a GPU's design is the opposite of a CPU's: a CPU uses out-of-order execution, branch prediction and large caches to make one thread as fast as possible, while a GPU uses thousands of threads to make the whole throughput as high as possible. To explain why a GPU is designed that way, you first have to know how a CPU does it. The CPU in an inference system is not idle either: scheduling, tokenizing, batching and preparing metadata all happen there, and for a small model's decode the CPU's cost can match the GPU's computation. This chapter uses six programs actually run on this machine to see instruction-level parallelism, branch prediction, the cache hierarchy, cache coherence, memory order and SIMD, and ends by mapping each onto a GPU.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Summing an array, why are 8 accumulators 8 times faster than 1? How does that relate to "each thread handling several elements" on a GPU?
    2. The same loop ran 7 times faster once the data was sorted. Why? A compiler can sometimes make that difference disappear; what did it do?
    3. Roughly what are the latencies of L1, L2, L3 and memory? Why does a 2 MB huge page make random access into a large array faster?
    4. Two threads each write their own variable. Why might they slow each other down? How do you avoid it?
    5. On x86, thread A writes x then reads y and thread B writes y then reads x. Can both read 0?

??? success "Answers (try it yourself first, then expand)"
    1. A floating-point addition's latency is about 4 cycles while 2 can be issued per cycle. With one accumulator every addition waits for the previous result, 4 cycles each; 8 independent accumulators keep 8 additions in the pipeline at once (latency 4 × throughput 2 = 8) and each addition costs half a cycle. Having each thread handle several elements on a GPU, issuing several independent loads and computations at once, is the same thing: filling the pipeline with instruction-level parallelism (ILP).
    2. With random data, `if (d[i] >= 128)` is taken half the time, the branch predictor is wrong half the time, and every miss discards the work already done on the wrong path, about twenty cycles; sorted, the first half is never taken and the second half always is, so nearly every prediction is right. A compiler can rewrite that branch as a conditional move (`cmov`) or as branchless vectorized code, and with no branch there is nothing to mispredict and both data sets run equally fast.
    3. Measured here: L1 about 1.7 ns, L2 about 5 ns, L3 about 25-40 ns, memory about 90 ns (in cycles, about 5, a dozen or so, nearly a hundred, and two or three hundred). Random access into a large array also misses the TLB and walks the page table; a 2 MB huge page makes one TLB entry cover 512 times as much memory and removes the walk, bringing a 512 MB working set from 130 ns to 90 ns here.
    4. Caches stay coherent in units of 64-byte cache lines: a core about to write a line has to invalidate every other core's copy (the MESI protocol). Two variables in one line, written in turn by two cores, send that line back and forth between them, which is false sharing. The way to avoid it is to give each frequently written variable a cache line of its own (aligned and padded to 64 bytes).
    5. Yes. Each core has a store buffer: a write goes into the buffer first and becomes visible to the other cores later, and the reads that follow need not wait for it. So both threads read the other's stale 0, which has not "landed" yet. That is the one reordering x86 allows (a store followed by a load), and forbidding it takes an `mfence` between the write and the read, or `seq_cst` atomics for both.

## The CPU in an inference system {#推理系统里的-cpu}

In an inference service the GPU does the matrix multiplies and the attention, and nearly everything else is on the CPU:

- **every round of the scheduler**: taking requests, tokenizing, prefix matching, forming the batch, preparing attention's metadata, handling the sampling results, detokenizing, sending the results back. For a small model, one decode step is a few milliseconds on the GPU and this round is a few milliseconds on the CPU too. Hence [overlap scheduling](minisgl://schedule/overlap/) (the CPU handling round N while the GPU already computes round N+1), [CUDA Graphs](minisgl://perf/cuda-graph/) (hundreds of kernel launches compressed into one) and [a front end rewritten in Rust](serving://source/rust-frontend/).
- **the CPU does inference itself**: on-device [llama.cpp](serving://ops/edge/) runs the whole model on the CPU; heterogeneous inference computes some of an MoE's experts there.
- **data passes through the CPU's side**: weights read from disk into memory and the KV cache offloaded to memory both cross the CPU's memory system and PCIe (see [pinned memory and NUMA](../os/pinned-numa.md)).

More importantly, every phenomenon in this chapter has a counterpart on a GPU. Seeing how a CPU does it is what shows why a GPU does not.

## Pipelines and instruction-level parallelism {#流水线与指令级并行}

An instruction's execution splits into fetch, decode, execute, memory access and write-back, and like a factory's assembly line the next instruction starts before the previous one finishes. A modern CPU is also **superscalar**: it fetches and issues 4 to 6 instructions per cycle; and **out-of-order**: instructions enter a reorder buffer (ROB) of a few hundred entries, whoever's operands are ready executes first, and the results are committed in program order at the end.

So what really limits the speed is often not "how many instructions in total" but the **dependency chain**: an instruction has to wait for what it depends on. Two numbers matter here:

- **latency**: how many cycles from issuing an instruction to its result being available. A floating-point addition is about 4 cycles;
- **throughput**: how many such instructions can be issued per cycle. Two for a floating-point addition.

Summing the same 4096 floats, differing only in the number of accumulators:

```c title="ilp.c" flags="-fno-tree-vectorize"
// summing the same 4096 floats, differing only in how many independent accumulators
#include <stdio.h>
#include <time.h>

#define N 4096                           // 16 KB, fits in L1
#define REPS 100000
float a[N];

static float sum1(void) {                // every addition waits for the previous result: one dependency chain
    float s = 0;
    for (int i = 0; i < N; i++) s += a[i];
    return s;
}
static float sum2(void) {
    float s0 = 0, s1 = 0;
    for (int i = 0; i < N; i += 2) s0 += a[i], s1 += a[i + 1];
    return s0 + s1;
}
static float sum4(void) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < N; i += 4) s0 += a[i], s1 += a[i + 1], s2 += a[i + 2], s3 += a[i + 3];
    return (s0 + s1) + (s2 + s3);
}
static float sum8(void) {                // 8 independent chains, all in the pipeline at once
    float s[8] = {0};
    for (int i = 0; i < N; i += 8) {
        s[0] += a[i], s[1] += a[i + 1], s[2] += a[i + 2], s[3] += a[i + 3];
        s[4] += a[i + 4], s[5] += a[i + 5], s[6] += a[i + 6], s[7] += a[i + 7];
    }
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

static void bench(const char *name, float (*f)(void)) {
    struct timespec t0, t1;
    volatile float sink = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) sink += f();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ((double)REPS * N);
    printf("%s：每次加法 %.2f ns\n", name, ns);
}

int main(void) {
    for (int i = 0; i < N; i++) a[i] = 1.0f / (i + 1);
    bench("1 个累加器", sum1);
    bench("2 个累加器", sum2);
    bench("4 个累加器", sum4);
    bench("8 个累加器", sum8);
    return 0;
}
```

```text title="output (on this machine)"
1 个累加器：每次加法 1.33 ns
2 个累加器：每次加法 0.67 ns
4 个累加器：每次加法 0.34 ns
8 个累加器：每次加法 0.17 ns
```

With one accumulator every addition waits for the previous result, and 1.33 ns is exactly 4 cycles (at about 3 GHz); two accumulators double it; and with 8 accumulators each addition takes 0.17 ns, half a cycle, reaching the ceiling of 2 per cycle. To fill the pipeline, the independent instructions in flight have to be latency × throughput = 4 × 2 = 8, which is Little's law, the same one used to [estimate how many warps saturate a GPU's bandwidth](cuda://basics/execution/#延迟掩盖).

Two details:

- Floating-point addition is not associative, and `(a+b)+c` and `a+(b+c)` round differently, so a compiler will not turn `sum1` into several accumulators on its own (short of `-ffast-math`). That rewrite is the programmer's to make, and the reductions on a GPU and FlashAttention's tiled accumulation follow the same idea; and because the order of summation changes, the results differ slightly, which is what [deterministic inference](serving://topics/deterministic/) has to handle.
- The option `-fno-tree-vectorize` keeps the compiler from using SIMD instructions, so the scalar pipeline can be seen on its own. SIMD comes later.

## Branch prediction {#分支预测}

The pipeline is deep, and at a conditional jump the CPU does not stop to wait for the condition but **guesses a direction** and carries on (speculative execution). A correct guess costs nothing; a wrong one discards all the work done on the wrong path and refetches from the right place, a loss of ten to twenty-odd cycles.

The same loop over the same random bytes, before and after sorting:

```c title="branch.c" flags="-fno-if-conversion -fno-if-conversion2 -fno-tree-vectorize"
// the same loop over the same data, sorted or not: the difference is all branch prediction
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 20)

static int cmp(const void *a, const void *b) { return *(const unsigned char *)a - *(const unsigned char *)b; }

static double sum_big(const unsigned char *d, long *out) {
    struct timespec t0, t1;
    long s = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 50; r++)
        for (int i = 0; i < N; i++)
            if (d[i] >= 128) s += d[i];               // with random data this branch is taken half the time and cannot be predicted
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *out = s;
    return ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / (50.0 * N) * 1e9;
}

int main(void) {
    unsigned char *d = malloc(N);
    srand(1);
    for (int i = 0; i < N; i++) d[i] = rand() & 255;
    long s1, s2;
    double random_ns = sum_big(d, &s1);
    qsort(d, N, 1, cmp);                               // sorted: the first half never taken, the second half always
    double sorted_ns = sum_big(d, &s2);
    printf("随机顺序：每个元素 %.2f ns\n排好序后：每个元素 %.2f ns\n结果相同：%s\n",
           random_ns, sorted_ns, s1 == s2 ? "是" : "否");
    return 0;
}
```

```text title="output (on this machine)"
随机顺序：每个元素 4.70 ns
排好序后：每个元素 0.64 ns
结果相同：是
```

With random data the branch is taken half the time, the predictor is right about half the time, and each element takes about 4.7 ns; sorted, nearly every guess is right at about 0.65 ns, 7 times faster. Estimating half the elements as mispredictions, each miss costs about (4.7 - 0.65) / 0.5 ≈ 8 ns, twenty-odd cycles.

The compile options are there to turn the compiler's branch elimination off. Without them, `gcc -O2` rewrites `if (d[i] >= 128) s += d[i]` into a conditional move, `cmov`: both results are computed and one is selected, with no branch in the program, and on this machine both data sets take 0.69 ns. When writing performance-sensitive code, a data-dependent branch that is hard to predict is best rewritten into exactly this "compute both, then select" form.

A GPU takes the other road: **no branch prediction and no speculative execution**. When a warp's 32 threads take different branches, the two paths run one after the other with the other half of the threads masked off, which is [branch divergence](cuda://basics/execution/#分支发散). So on a GPU too, a data-dependent branch is best made branchless, or the neighbouring threads made to take the same path.

## The cache hierarchy {#缓存层次}

Memory's latency is hundreds of cycles, so a CPU keeps what it uses often close by in several levels of cache. A cache reads and writes in units of a **cache line** (64 bytes): read one byte and the whole line comes in. This machine's caches:

```bash
lscpu -C
```

```text
NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL  SETS PHY-LINE COHERENCY-SIZE
L1d       48K     768K   12 Data            1    64        1             64
L1i       32K     512K    8 Instruction     1    64        1             64
L2       1.3M      20M   20 Unified         2  1024        1             64
L3        54M     108M   12 Unified         3 73728        1             64
```

Every core has its own 48 KB L1 data cache and 1.25 MB L2, and each socket's 16 cores share one 54 MB L3. To measure each level's latency, the CPU has to be unable to issue the next access early: chain the cache lines of a block of memory into a random cycle, so each value read is the next address to read (a pointer chase). Whichever level the working set fits in is the latency measured:

```c title="cache_latency.c"
// a pointer chase: each value read is the next address, so the CPU cannot issue the next access early and the pure latency is measured
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

typedef struct node { struct node *next; char pad[56]; } node;   // one node occupies exactly one 64-byte cache line

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double chase(size_t bytes, int huge) {
    size_t n = bytes / sizeof(node), len = (bytes + (2u << 20) - 1) & ~((2ul << 20) - 1);
    node *a = aligned_alloc(2u << 20, len);
    if (huge) madvise(a, len, MADV_HUGEPAGE);            // 2 MB huge pages: TLB misses excluded, the caches alone
    size_t *perm = malloc(n * sizeof(size_t));
    uint64_t seed = 42;
    for (size_t i = 0; i < n; i++) perm[i] = i;
    for (size_t i = n - 1; i > 0; i--) {                 // Sattolo's algorithm: a single random cycle, returning to the start only after every node
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = (seed >> 33) % i, t = perm[i];
        perm[i] = perm[j], perm[j] = t;
    }
    for (size_t i = 0; i < n; i++) a[perm[i]].next = &a[perm[(i + 1) % n]];
    node *p = &a[perm[0]];
    for (size_t i = 0; i < n; i++) p = p->next;           // warm-up: whatever fits in the cache goes in first
    size_t steps = 10 * 1000 * 1000;
    double t0 = now();
    for (size_t i = 0; i < steps; i++) p = p->next;
    double ns = (now() - t0) / steps * 1e9;
    if (p == NULL) puts("");                               // use p, so the loop is not optimized away
    free(perm);
    free(a);
    return ns;
}

int main(void) {
    size_t kb[] = {16, 32, 256, 1024, 8192, 32768, 131072, 524288};
    printf("工作集        4 KB 页    2 MB 大页\n");
    for (size_t i = 0; i < sizeof kb / sizeof *kb; i++)
        printf("%7zu KB  %7.1f ns  %7.1f ns\n", kb[i], chase(kb[i] << 10, 0), chase(kb[i] << 10, 1));
    return 0;
}
```

```text title="output (on this machine)"
工作集        4 KB 页    2 MB 大页
     16 KB      1.7 ns      1.7 ns
     32 KB      1.7 ns      1.7 ns
    256 KB      4.8 ns      4.7 ns
   1024 KB      7.0 ns      4.7 ns
   8192 KB     29.3 ns     25.1 ns
  32768 KB    102.4 ns     87.7 ns
 131072 KB    109.6 ns     89.4 ns
 524288 KB    132.0 ns     89.9 ns
```

The result is a staircase:

- 16 KB and 32 KB are in L1, 1.7 ns, about 5 cycles;
- 256 KB and 1 MB are in L2, about 5 ns;
- 8 MB is in L3, 25 to 40 ns;
- 32 MB and above approach memory's latency, about 90 ns (with huge pages). The L3 is nominally 54 MB, but it is shared by the whole socket's cores, and this machine is a cloud virtual machine sharing with other tenants, so one core gets far less than that.

The difference between the two columns is the **TLB**: every access translates a virtual address into a physical one and the TLB caches the recent translations. With 4 KB pages, 512 MB needs 130,000 page table entries, which the TLB cannot hold, so every access also walks the page table; with 2 MB huge pages one entry covers 512 times as much, the walks nearly vanish, and 512 MB comes down from 130 ns to 90. In a virtual machine this costs more still, since the guest's and the host's page tables are both walked. The TLB and huge pages are covered in [virtual memory, page tables and huge pages](../os/virtual-memory.md).

This table explains a great deal in inference systems. A Python list holds pointers to objects, so walking a request list, looking through a prefix tree or searching a dict is essentially a pointer chase where every step may be a cache miss; stored contiguously instead (NumPy, tensors), the 64 bytes brought in at once are all useful and the hardware prefetcher can read the following lines in advance. Whether the scheduler's metadata is organized as an array of structs or a struct of arrays is exactly this question (see the C++ handbook's [object layout, alignment and the cache](cpp://memory/layout/)).

### Set associativity and conflict misses {#组相联与冲突缺失}

A cache does not put a line wherever there is room. A 48 KB, 12-way set-associative L1 has 64 sets of 12 lines: an address picks its set by `line number mod 64` and may only take one of that set's 12 places, with the least recently used evicted when the set is full. So addresses a multiple of 64 × 64 B = 4 KB apart all land in one set.

Walking a 512 × 512 float matrix by column: each row is 2 KB, so two adjacent elements of a column are 32 cache lines apart, 512 rows land in only 2 sets, 24 places cannot hold 512 rows, and a line is evicted before it is used a second time. Padding each row with 16 floats (one cache line) makes the row stride 33 cache lines and spreads the rows over every set. Move the capacity, the associativity and the stride and watch how an address is cut into tag / set / offset and when they start evicting each other:

<div class="aig-widget" data-widget="cachemap"></div>

A set-associative cache simulator showing the hit rate:

```python title="cache_sim.py"
from collections import OrderedDict


class Cache:
    """组相联缓存，组内按 LRU 替换。size、line 以字节计。"""

    def __init__(self, size, ways, line=64):
        self.line, self.ways = line, ways
        self.nsets = size // (ways * line)
        self.sets = [OrderedDict() for _ in range(self.nsets)]
        self.hits = self.misses = 0

    def access(self, addr):
        tag = addr // self.line                     # which cache line
        s = self.sets[tag % self.nsets]             # the line number modulo decides the set
        if tag in s:
            s.move_to_end(tag)
            self.hits += 1
        else:
            self.misses += 1
            if len(s) == self.ways:
                s.popitem(last=False)               # the set is full: evict the least recently used
            s[tag] = True


def walk(rows, cols, row_bytes, by_column):
    c = Cache(48 * 1024, ways=12)                   # as this machine's L1d: 48 KB, 12-way, 64 sets
    order = ((r, k) for k in range(cols) for r in range(rows)) if by_column else \
            ((r, k) for r in range(rows) for k in range(cols))
    for r, k in order:
        c.access(r * row_bytes + k * 4)             # a float matrix
    return c.hits / (c.hits + c.misses)


print(f"512x512 按行遍历：        命中率 {walk(512, 512, 512 * 4, False):.1%}")
print(f"512x512 按列遍历：        命中率 {walk(512, 512, 512 * 4, True):.1%}")
print(f"每行补 16 个 float 再按列：命中率 {walk(512, 512, 528 * 4, True):.1%}")
```

```text title="output"
512x512 按行遍历：        命中率 93.8%
512x512 按列遍历：        命中率 0.0%
每行补 16 个 float 再按列：命中率 93.8%
```

Walking by row, 16 floats share one cache line, so one miss is followed by 15 hits, a hit rate of 15/16; walking by column, the next column could have reused the lines this column brought in (the capacity is ample), yet every access misses because they are crowded into two sets, which is a **conflict miss**. With one row of padding, the same column walk returns to 15/16.

Powers of two deserve particular care as strides: a matrix's row width, a hash table's size or one dimension of a multidimensional tensor landing exactly on 4 KB or a larger power of two can hit this. A GPU's shared memory [bank conflicts](cuda://basics/memory/#bank-冲突) are the same thing: addresses modulo 32 banks, and a stride that is a multiple of 32 puts everything in one bank, with padding a column as the same answer.

## Cache coherence: MESI {#缓存一致性mesi}

Every core has its own L1 and L2, and one cache line may sit in several cores' caches at once. The hardware keeps what every core sees consistent with a **cache coherence protocol**, most often explained as MESI, where every line in every core is in one of four states:

| State | Meaning | This core reads | This core writes |
| --- | --- | --- | --- |
| M (Modified) | only this core has it and it has changed, so memory is stale | reads directly | writes directly |
| E (Exclusive) | only this core has it and it matches memory | reads directly | writes directly, becoming M |
| S (Shared) | several cores have it, all matching memory | reads directly | invalidates the others' copies first, becoming M |
| I (Invalid) | absent, or invalidated | fetched from elsewhere, becoming S or E | fetched and the others invalidated, becoming M |

The "writes" column is the key: **writing a line requires owning it exclusively**, and every other core's copy is discarded. Two threads writing different variables that happen to share a cache line send that line back and forth between the cores, invalidated each way, which is **false sharing**. Two threads each incrementing their own counter (atomically), with the counters next to each other or on lines of their own:

```c title="false_sharing.c"
// two threads each increment their own counter, unrelated; only where the counters sit differs
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <time.h>

#define N 20000000L

struct { volatile long a, b; } same;                                // a and b are adjacent, in one cache line
struct { _Alignas(64) volatile long a; _Alignas(64) volatile long b; } padded;   // a cache line each

typedef struct { volatile long *p; int cpu; } arg_t;

static void *worker(void *v) {
    arg_t *a = v;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(a->cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);      // pinned, so the two threads are on different cores
    for (long i = 0; i < N; i++) __atomic_fetch_add(a->p, 1, __ATOMIC_RELAXED);   // an atomic increment: it has to own the line every time
    return NULL;
}

static double run(volatile long *x, volatile long *y, int cpu_x, int cpu_y) {
    struct timespec t0, t1;
    pthread_t t[2];
    arg_t args[2] = {{x, cpu_x}, {y, cpu_y}};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < 2; i++) pthread_create(&t[i], NULL, worker, &args[i]);
    for (int i = 0; i < 2; i++) pthread_join(t[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

int main(void) {
    printf("两个计数器在同一个缓存行：%.2f s\n", run(&same.a, &same.b, 0, 2));
    printf("各占一个缓存行：          %.2f s\n", run(&padded.a, &padded.b, 0, 2));
    return 0;
}
```

```text title="output (on this machine)"
两个计数器在同一个缓存行：0.64 s
各占一个缓存行：          0.14 s
```

Nearly 5 times slower in one cache line. The atomic increment is used because it has to own the line every time and is affected most; replacing it with an ordinary `volatile` increment makes the two cases nearly equal on this machine, because an ordinary write goes into the store buffer first and a run of them is written back at once after the line arrives. How to separate data by cache lines in C++ (`alignas(64)`) is in the C++ handbook's [false sharing](cpp://memory/layout/#伪共享).

A GPU does it differently again: the SMs' L1 caches have **no** coherence protocol between them, so what one SM writes to global memory may still be stale in another SM's L1, and coherence is guaranteed only at L2. So communicating between blocks on a GPU takes atomics, accesses that bypass L1, or scoped fences.

## Memory order: the reordering the store buffer causes {#内存序存储缓冲区带来的重排}

A write cannot really enter the cache until the line's ownership arrives, which may take hundreds of cycles. To keep the following instructions from waiting, every core has a **store buffer**: a write goes in and the core carries on, with the buffered writes landing in the cache later. This core's later reads can take its own just-written value from the buffer, but **other cores do not see it until it lands**.

That produces a counter-intuitive result. Two threads each write a variable and then read the other's:

```c title="reorder.c" ci="no"
// the store buffering experiment: two threads each write a variable and then read the other's
// in "program order", r1 and r2 could not both be 0
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>

#define ROUNDS 1000000

static atomic_int x, y, r1, r2, go, done;
static int use_fence;

static void pin(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static void *t1(void *arg) {
    (void)arg;
    pin(2);
    for (int i = 1; i <= ROUNDS; i++) {
        while (atomic_load_explicit(&go, memory_order_acquire) != i) {}    // wait for the main thread's signal
        atomic_store_explicit(&x, 1, memory_order_relaxed);
        if (use_fence) atomic_thread_fence(memory_order_seq_cst);          // compiles to mfence on x86
        atomic_store_explicit(&r1, atomic_load_explicit(&y, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_add_explicit(&done, 1, memory_order_release);
    }
    return NULL;
}

static void *t2(void *arg) {
    (void)arg;
    pin(4);
    for (int i = 1; i <= ROUNDS; i++) {
        while (atomic_load_explicit(&go, memory_order_acquire) != i) {}
        atomic_store_explicit(&y, 1, memory_order_relaxed);
        if (use_fence) atomic_thread_fence(memory_order_seq_cst);
        atomic_store_explicit(&r2, atomic_load_explicit(&x, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_add_explicit(&done, 1, memory_order_release);
    }
    return NULL;
}

static int trial(int fence) {
    pthread_t a, b;
    int both_zero = 0;
    use_fence = fence;
    atomic_store(&go, 0);
    pthread_create(&a, NULL, t1, NULL);
    pthread_create(&b, NULL, t2, NULL);
    pin(0);
    for (int i = 1; i <= ROUNDS; i++) {
        atomic_store(&x, 0), atomic_store(&y, 0), atomic_store(&done, 0);
        atomic_store_explicit(&go, i, memory_order_release);               // the signal: both threads start at once
        while (atomic_load_explicit(&done, memory_order_acquire) != 2) {}
        both_zero += atomic_load(&r1) == 0 && atomic_load(&r2) == 0;
    }
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    return both_zero;
}

int main(void) {
    printf("不加屏障：%d 轮里 r1 == r2 == 0 出现了 %d 次\n", ROUNDS, trial(0));
    printf("加 mfence：%d 轮里 r1 == r2 == 0 出现了 %d 次\n", ROUNDS, trial(1));
    return 0;
}
```

```text title="output (on this machine)"
不加屏障：1000000 轮里 r1 == r2 == 0 出现了 195 次
加 mfence：1000000 轮里 r1 == r2 == 0 出现了 0 次
```

If each thread's write and read happened strictly in program order and became visible to the other immediately, at least one of `r1` and `r2` would be 1. But on x86, both writes can still be in their store buffers when both reads execute, so both read 0. Adding an `mfence` (what a `seq_cst` barrier compiles to on x86), which waits for the store buffer to drain before letting the read proceed, makes this result disappear.

x86's memory model is TSO (total store order) and allows only this one reordering, a store followed by a load, keeping store-store and load-load order; ARM servers (Grace, Kunpeng, Graviton) have a far weaker model where nearly anything can be reordered, so code that uses atomic variables without specifying a memory order runs fine on x86 and can break on ARM. A GPU's memory model is weak too, and scoped besides (the block, the whole GPU, the whole system). The rules for writing such code are in the C++ handbook's [atomics and memory order](cpp://concurrency/atomics/). In an inference system, passing messages between processes through shared memory ([vLLM's shared-memory broadcast queue](../os/ipc.md)) and sharing [io_uring's rings](../os/io-models.md) with the kernel both rest on these guarantees.

## SIMD and peak compute {#simd-与峰值算力}

One SIMD instruction handles several data at once: AVX2's registers are 256 bits and take 8 floats, and AVX-512's take 16. Every core here has two 512-bit FMA units, so at most 2 × 16 = 32 multiply-adds per cycle, 64 FLOP. The same loop (8 multiply-adds per element) compiled into three versions:

```c title="simd.c" ci="no"
// the same loop (an 8-term polynomial, 8 multiply-adds, per element) compiled into scalar, AVX2 (8-wide) and AVX-512 (16-wide) versions
#include <stdio.h>
#include <time.h>

#define N 1024                       // the array fits in L1 and each element takes 8 multiply-adds: this measures computation, not memory
#define REPS 200000

float x[N] __attribute__((aligned(64))), y[N] __attribute__((aligned(64)));

#define BODY                                                                    \
    for (int i = 0; i < N; i++) {                                               \
        float v = x[i], r = v;                                                  \
        _Pragma("GCC unroll 8")                                                 \
        for (int k = 0; k < 8; k++) r = r * v + 0.5f;  /* 编译器会生成 FMA */   \
        y[i] = r;                                                               \
    }

__attribute__((target("fma"), optimize("no-tree-vectorize"))) void poly_scalar(void) { BODY }
__attribute__((target("avx2,fma"), optimize("tree-vectorize"))) void poly_avx2(void) { BODY }
__attribute__((target("avx512f,prefer-vector-width=512"), optimize("tree-vectorize"))) void poly_avx512(void) { BODY }

static double gflops(void (*f)(void)) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) f();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return 2.0 * 8 * N * REPS / s / 1e9;                        // each multiply-add = 2 FLOP
}

int main(void) {
    for (int i = 0; i < N; i++) x[i] = (i % 7) * 0.1f;
    printf("标量：   %6.1f GFLOPS\n", gflops(poly_scalar));
    printf("AVX2：   %6.1f GFLOPS\n", gflops(poly_avx2));
    printf("AVX-512：%6.1f GFLOPS\n", gflops(poly_avx512));
    return 0;
}
```

```text title="output (on this machine)"
标量：     11.4 GFLOPS
AVX2：     87.8 GFLOPS
AVX-512： 165.4 GFLOPS
```

The scalar version is 2 multiply-adds, 4 FLOP, per cycle, which at about 3 GHz is 12 GFLOPS and measures around 11; AVX2 is 8 times that; AVX-512 about 15 times (the clock drops somewhat while running AVX-512). Roughly for the whole machine: two CPUs, 32 cores × about 165 GFLOPS ≈ 5 TFLOPS. Against an H100: about 67 TFLOPS of FP32 and about 989 TFLOPS of BF16 Tensor Core, two orders of magnitude apart, which is why the matrix multiplies all go to the GPU.

CPUs are adding matrix units too: Intel has AMX from Sapphire Rapids onwards, doing BF16 and INT8 matrix multiplies in dedicated matrix registers at roughly an order of magnitude more throughput than AVX-512, which heterogeneous inference uses when MoE experts run on the CPU. But decode on a CPU is still bandwidth-bound: a server CPU has 8 to 12 memory channels, and 12 channels of DDR5-4800 is about 460 GB/s against an H100's 3.35 TB/s.

## From the CPU to the GPU {#从-cpu-到-gpu}

The same problems, with entirely different answers from the two kinds of chip:

| The problem | The CPU's answer | The GPU's answer |
| --- | --- | --- |
| memory takes hundreds of cycles | large caches, out-of-order execution and hardware prefetching, so one thread waits less | dozens of warps issuing in turn, switching to another whenever one waits |
| instructions depend on each other | find independent instructions to run first in a ROB of a few hundred entries | issue in order, filling the pipeline with other warps and with ILP inside a thread |
| branches | branch prediction + speculative execution | no prediction; a diverged warp runs both paths in turn |
| data parallelism | SIMD: explicit vector registers and vector instructions | SIMT: write a single thread's code and the hardware packs 32 threads into a warp |
| on-chip storage | tens of MB of hardware-managed cache | an enormous register file (256 KB per SM, five times a CPU core's L1) + programmer-managed shared memory |
| coherence | hardware keeps every core's cache coherent | the SMs' L1 caches are not coherent, relying on L2 and scoped fences |
| where the die area goes | control logic and caches | compute units and registers |

A CPU is after finishing one thread as soon as possible (latency) and a GPU after finishing as much work per unit time as possible (throughput). The next chapter, [a GPU's SMs and Tensor Cores](gpu-sm.md), starts from a GPU chip's internals.

!!! interview "How to explain it"
    To explain CPU architecture or "how do a CPU and a GPU differ": a CPU lowers one thread's latency with superscalar out-of-order execution, branch prediction and several levels of cache; what really limits it is the dependency chain, needing independent instructions in flight equal to latency × throughput (4 × 2 = 8 accumulators for floating-point addition). Caches are organized in 64-byte lines and by associativity, with a staircase of latencies at roughly a few cycles for L1, a dozen or so for L2, nearly a hundred for L3 and two or three hundred for memory; random access adds TLB misses, which huge pages ease; and power-of-two strides cause conflict misses. Several cores stay coherent through MESI, where writing requires exclusive ownership, hence false sharing; x86 is TSO and allows only the store buffer's store-load reordering, while ARM and GPUs are weaker. A GPU is the reverse: in-order issue, no branch prediction, latency hidden by switching among a great many warps, the die area spent on compute units and registers, and small caches whose L1 is not coherent.

## Exercises {#练习}

**1. How many accumulators.** A core here has two 512-bit FMA units and an FMA latency of 4 cycles. Computing the dot product of two long float vectors with AVX-512, how many vector accumulators does saturating it take at minimum? And on a GPU, where an H100's SM has 4 warp schedulers each issuing one instruction per cycle, if an FMA's result is usable 4 cycles later, how many independent FMAs does each scheduler need in flight?

??? success "Answer"
    The CPU: latency 4 × throughput 2 = 8, so at least 8 zmm accumulators, each holding 16 partial sums to be added together at the end (assuming the data is in L1 and bandwidth is not the limit).

    The GPU: one instruction issued per cycle per scheduler with a 4-cycle latency needs 4 independent FMAs in flight. They come from two sources: 4 warps with one each (thread-level parallelism), or 1 warp with 4 mutually independent FMAs (instruction-level parallelism), usually both. A real GPU kernel also has far longer memory latencies to hide, so the work in flight has to be far greater.

**2. Conflict misses.** This machine's L1d is 48 KB, 12-way, 64 sets, 64-byte lines. Walking a `float a[64][1024]` by column, how many sets do a column's 64 elements land in? What happens? How do you fix it?

??? success "Answer"
    Each row is 1024 × 4 = 4096 bytes, exactly 64 sets × 64 bytes. Two adjacent elements of a column are 4096 bytes apart, 64 lines apart, which is the same modulo 64, so a column's 64 elements all land in **one set**. That set has 12 places and the 64 rows cycle through it, so a line is evicted before the next column can reuse it and nearly every access misses (where 16 floats per row could have shared one line).

    The fix: pad each row by one cache line, declaring `float a[64][1024 + 16]`, which makes the row stride 4160 bytes = 65 cache lines, so adjacent rows land in adjacent sets and the 64 rows spread over 64 sets. Or change the traversal order and walk by row.

**3. Decode on a CPU.** A server's CPU has 12 DDR5-4800 memory channels (4800 M transfers per second per channel, 8 bytes each) and about 5 TFLOPS with AVX-512 across its cores. Running a 7B BF16 model on it: at most how many tokens per second does batch-1 decode give? At minimum how long does prefilling 1000 tokens take? What limits each? And with 4-bit quantization?

??? success "Answer"
    The bandwidth is 12 × 4800e6 × 8 ≈ 461 GB/s. The BF16 weights are 14 GB and batch-1 decode reads them once per token: 14 GB / 461 GB/s ≈ 30 ms, at most about 33 tokens per second; the computation is only 2 × 7e9 = 14 GFLOP, under 3 ms at 5 TFLOPS, so decode is bandwidth-bound.

    Prefilling 1000 tokens is 14 TFLOP, at least 2.8 s at 5 TFLOPS (and the real matrix multiply's efficiency takes a further discount), which is compute-bound; the same work on an H100's BF16 Tensor Cores at 50% utilization takes about 28 ms.

    Quantized to 4 bits the weights are about 3.5 GB and decode's ceiling rises to about 130 tokens/s; prefill's computation is unchanged (still done at higher precision after dequantizing) and gets no faster. That is how on-device and CPU inference estimate "decode by bandwidth, prefill by compute", exactly as on a GPU.

**4. Memory order.** Two processes pass messages through shared memory: the producer writes the message's contents and then increments an "written" counter; the consumer sees the counter change and then reads the contents. On x86, can the hardware let the consumer see the counter before the contents? What else does the code need? And porting it to an ARM server?

??? success "Answer"
    x86's TSO keeps store-store and load-load order, so the producer's two writes are not swapped by the hardware and neither are the consumer's two reads, and at the hardware level nothing goes wrong. But the **compiler** may still reorder them or cache the read in a register, so the counter has to be atomic: a release store by the producer and an acquire load by the consumer. On x86 those produce no extra barrier instruction and only constrain the compiler; on ARM the compiler emits real barriers (or `stlr` / `ldar`) for them, with no change to the code.

    Written instead as "write my own flag, then read the other's" in a mutual-wait pattern (this chapter's store buffer experiment, which is also the shape of Dekker's and Peterson's lock algorithms), x86 does go wrong and needs `seq_cst` or an explicit `mfence`.

## Summary {#小结}

- [x] What limits the speed is the dependency chain: the independent instructions in flight have to equal latency × throughput, several accumulators fill the pipeline, and a GPU likewise needs ILP and warps enough.
- [x] A mispredicted branch costs about twenty cycles; an unpredictable branch is best rewritten as "compute both, then select". A GPU does not predict and runs a diverged warp's two paths in turn.
- [x] Caches are organized in 64-byte lines and by associativity: L1 about 5 cycles, L2 about 14, L3 nearly a hundred, memory two or three hundred; random access adds TLB misses, which huge pages ease; power-of-two strides cause conflict misses, which padding a row fixes.
- [x] MESI: writing requires exclusive ownership, so different variables in one cache line slow each other down (false sharing) and have to be separated by a line. A GPU's SMs' L1 caches are not coherent.
- [x] x86's store buffer causes only store-load reordering, which `mfence` or `seq_cst` forbids; ARM's and a GPU's memory models are weaker.
- [x] SIMD raises a core's peak by over a dozen times, and a whole CPU is still two orders of magnitude below one GPU; decode on a CPU is bandwidth-bound.
