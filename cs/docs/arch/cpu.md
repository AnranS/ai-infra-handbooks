# CPU 体系结构速成：流水线、缓存与一致性

<p class="lead">GPU 的很多设计都是和 CPU 反着来的：CPU 用乱序执行、分支预测和大缓存让一个线程跑得尽量快，GPU 用成千上万个线程让整体吞吐尽量高。想讲清 GPU 为什么这样设计，先要知道 CPU 是怎么做的。推理系统里的 CPU 也不清闲：调度、分词、组 batch、准备元数据都在 CPU 上，小模型 decode 时 CPU 的开销能和 GPU 计算一样长。这一章用六个在本机实跑的程序看清指令级并行、分支预测、缓存层次、缓存一致性、内存序和 SIMD，最后把它们和 GPU 一一对应起来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 把一个数组加起来，为什么用 8 个累加器比 1 个快 8 倍？这和 GPU 上"每个线程多处理几个元素"是什么关系？
    2. 同样的循环，数据排好序后快了 7 倍，为什么？编译器有时能让这个差别消失，它做了什么？
    3. L1、L2、L3 和内存的延迟大概各是多少？随机访问一个大数组，用 2 MB 大页为什么会变快？
    4. 两个线程各写各的变量，为什么可能互相拖慢？怎么避免？
    5. x86 上，线程 A 先写 x 再读 y，线程 B 先写 y 再读 x，两个线程都读到 0 可能吗？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 浮点加法的延迟约 4 个周期，但每个周期能发射 2 条。只有一个累加器时，每次加法都要等上一次的结果，一次 4 个周期；8 个互相独立的累加器能让 8 条加法同时在流水线里（延迟 4 × 吞吐 2 = 8），每次加法只摊到半个周期。GPU 上让每个线程处理多个元素、同时发出多个独立的访存和计算，是同一个道理：用指令级并行（ILP）填满流水线。
    2. 数据随机时，`if (d[i] >= 128)` 一半跳一半不跳，分支预测器有一半猜错，每猜错一次要作废流水线里错误路径上的指令，损失二十个周期左右；排好序后前一半全不跳、后一半全跳，几乎每次都猜对。编译器可以把这个分支改写成条件传送（`cmov`）或者向量化的无分支代码，没有分支也就没有猜错，两种数据一样快。
    3. 本机测得 L1 约 1.7 ns、L2 约 5 ns、L3 约 25～40 ns、内存约 90 ns（换算成周期是约 5 个、十几个、近百个、两三百个）。随机访问大数组时，除了缓存缺失，还有 TLB 缺失，要走页表；2 MB 大页让一个 TLB 项覆盖的内存大 512 倍，省掉了页表遍历，本机 512 MB 的工作集从 130 ns 降到 90 ns。
    4. 缓存以 64 字节的缓存行为单位保持一致：一个核要写某一行，必须先让其他核里这一行的副本失效（MESI 协议）。两个变量在同一行里，两个核轮流写，这一行就在两个核之间来回搬，这叫伪共享。避免的办法是让不同线程频繁写的变量各占一个缓存行（按 64 字节对齐、补齐）。
    5. 可能。每个核有一个存储缓冲区（store buffer），写操作先进缓冲区，稍后才对其他核可见，后面的读不用等它。于是两个线程都先读到了对方还没"落地"的旧值 0。这是 x86 唯一允许的一种重排（先写后读），要禁止它，得在写和读之间加 `mfence`，或者都用 `seq_cst` 的原子操作。

## 推理系统里的 CPU

推理服务里，GPU 负责矩阵乘和注意力，其余的事几乎都在 CPU 上：

- **调度器的每一轮**：收请求、分词、前缀匹配、组 batch、准备注意力的元数据、处理采样结果、反分词、发回结果。一个小模型 decode 一步，GPU 只要几毫秒，CPU 这一圈也要几毫秒。所以才有[重叠调度](minisgl://schedule/overlap/)（CPU 处理第 N 轮时 GPU 已经在算第 N+1 轮）、[CUDA Graph](minisgl://perf/cuda-graph/)（把几百次 kernel 启动压成一次）和用 [Rust 重写的前端](serving://source/rust-frontend/)。
- **CPU 自己也做推理**：端侧的 [llama.cpp](serving://ops/edge/) 在 CPU 上跑整个模型；异构推理把 MoE 的部分专家放在 CPU 上算。
- **数据要经过 CPU 这一侧**：权重从盘读进内存、KV Cache 卸载到内存，都要穿过 CPU 的内存系统和 PCIe（见[锁页内存与 NUMA](../os/pinned-numa.md)）。

更重要的是，这一章的每个现象在 GPU 上都有对应。先看 CPU 怎么做，才知道 GPU 为什么不那样做。

## 流水线与指令级并行

一条指令的执行分成取指、译码、执行、访存、写回等多个阶段，像工厂的流水线一样，前一条还没做完，后一条已经开始了。现代 CPU 还是**超标量**的：每个周期能取、发射 4～6 条指令；并且**乱序执行**：指令进入一个几百项的重排序缓冲区（ROB），谁的操作数先准备好谁先执行，最后再按程序顺序提交结果。

于是真正限制速度的，往往不是"一共有多少条指令"，而是**依赖链**：一条指令必须等它依赖的结果算出来。这里要分清两个数：

- **延迟**：一条指令从发射到结果可用要几个周期。浮点加法约 4 个周期；
- **吞吐**：每个周期能发射几条这种指令。浮点加法每周期 2 条。

同样是把 4096 个 float 加起来，只是累加器的个数不同：

```c title="ilp.c" flags="-fno-tree-vectorize"
// 同样是把 4096 个 float 加起来，只是用了几个互相独立的累加器
#include <stdio.h>
#include <time.h>

#define N 4096                           // 16 KB，放得进 L1
#define REPS 100000
float a[N];

static float sum1(void) {                // 每次加法都要等上一次的结果：一条依赖链
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
static float sum8(void) {                // 8 条互不依赖的链，可以同时在流水线里
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

```text title="输出（本机示例）"
1 个累加器：每次加法 1.33 ns
2 个累加器：每次加法 0.67 ns
4 个累加器：每次加法 0.34 ns
8 个累加器：每次加法 0.17 ns
```

一个累加器时，每次加法都要等上一次的结果，1.33 ns 正好是 4 个周期（频率约 3 GHz）；两个累加器快一倍；8 个累加器时每次加法只要 0.17 ns，即半个周期，达到了每周期 2 条的吞吐上限。要填满流水线，需要同时在途的独立指令数 = 延迟 × 吞吐 = 4 × 2 = 8，这就是 Little 定律，[GPU 上估算要多少个 warp 才能跑满带宽](cuda://basics/execution/#延迟掩盖)用的也是它。

两个细节：

- 浮点加法不满足结合律，`(a+b)+c` 和 `a+(b+c)` 的舍入不同，所以编译器不会自作主张把 `sum1` 改成多个累加器（除非加 `-ffast-math`）。这种改写要程序员自己做，GPU 上的归约、FlashAttention 里的分块累加都是这个思路；也正因为改变了求和顺序，结果会有微小差别，这是[确定性推理](serving://topics/deterministic/)要处理的问题。
- 编译选项 `-fno-tree-vectorize` 是为了不让编译器改用 SIMD 指令，单独看标量流水线。SIMD 在后面讲。

## 分支预测

流水线很深，遇到条件跳转时，CPU 不会停下来等条件算出来，而是先**猜一个方向**继续往下执行（推测执行）。猜对了没有任何损失；猜错了，要把错误路径上已经做的工作全部作废，从正确的位置重新取指，损失十几到二十几个周期。

同一批随机字节，排序前后各跑一遍同一个循环：

```c title="branch.c" flags="-fno-if-conversion -fno-if-conversion2 -fno-tree-vectorize"
// 同一个循环、同一批数据，只是先排不排序：差别全在分支预测
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
            if (d[i] >= 128) s += d[i];               // 数据随机时，这个分支一半跳一半不跳，猜不准
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
    qsort(d, N, 1, cmp);                               // 排序后：前一半全不跳，后一半全跳
    double sorted_ns = sum_big(d, &s2);
    printf("随机顺序：每个元素 %.2f ns\n排好序后：每个元素 %.2f ns\n结果相同：%s\n",
           random_ns, sorted_ns, s1 == s2 ? "是" : "否");
    return 0;
}
```

```text title="输出（本机示例）"
随机顺序：每个元素 4.70 ns
排好序后：每个元素 0.64 ns
结果相同：是
```

数据随机时，分支一半跳一半不跳，预测器只能猜对一半左右，每个元素约 4.7 ns；排好序后几乎每次都猜对，约 0.65 ns，快了 7 倍。按一半的元素猜错估算，每次猜错损失约 (4.7 - 0.65) / 0.5 ≈ 8 ns，二十多个周期。

编译选项是为了关掉编译器的"消除分支"优化。不加这几个选项，`gcc -O2` 会把 `if (d[i] >= 128) s += d[i]` 改写成条件传送指令 `cmov`：两个结果都算出来，再按条件选一个，程序里没有了分支，本机上两种数据都是 0.69 ns。写性能敏感的代码时，数据相关、难以预测的分支，最好改写成这种"都算、再选"的形式。

GPU 走的是另一条路：**不做分支预测，也不推测执行**。一个 warp 的 32 个线程走不同分支时，两条路径依次执行，各自屏蔽掉另一半线程，这叫[分支发散](cuda://basics/execution/#分支发散)。所以在 GPU 上，同样要把数据相关的分支改写成无分支的形式，或者让相邻线程走同样的路径。

## 缓存层次

内存的延迟是几百个周期，CPU 用几级缓存把常用的数据留在身边。缓存以**缓存行**（64 字节）为单位读写：读一个字节，整行都被搬进来。本机的缓存结构：

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

每个核有自己的 48 KB L1 数据缓存和 1.25 MB L2，每个插槽的 16 个核共享一块 54 MB 的 L3。测每一级的延迟，要让 CPU 没法提前发出下一次访存：把一块内存里的缓存行随机串成一个环，每次读出的值就是下一次要读的地址（指针追逐）。工作集放得进哪一级缓存，测出来的就是哪一级的延迟：

```c title="cache_latency.c"
// 指针追逐：每次读出的值就是下一次要读的地址，CPU 没法提前发出下一次访存，测到的就是纯延迟
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

typedef struct node { struct node *next; char pad[56]; } node;   // 一个节点正好占一条 64 字节的缓存行

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double chase(size_t bytes, int huge) {
    size_t n = bytes / sizeof(node), len = (bytes + (2u << 20) - 1) & ~((2ul << 20) - 1);
    node *a = aligned_alloc(2u << 20, len);
    if (huge) madvise(a, len, MADV_HUGEPAGE);            // 用 2 MB 大页：排除 TLB 缺失，只看缓存
    size_t *perm = malloc(n * sizeof(size_t));
    uint64_t seed = 42;
    for (size_t i = 0; i < n; i++) perm[i] = i;
    for (size_t i = n - 1; i > 0; i--) {                 // Sattolo 算法：随机排成一个大环，走完所有节点才回到起点
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = (seed >> 33) % i, t = perm[i];
        perm[i] = perm[j], perm[j] = t;
    }
    for (size_t i = 0; i < n; i++) a[perm[i]].next = &a[perm[(i + 1) % n]];
    node *p = &a[perm[0]];
    for (size_t i = 0; i < n; i++) p = p->next;           // 预热：能装进缓存的部分先装进去
    size_t steps = 10 * 1000 * 1000;
    double t0 = now();
    for (size_t i = 0; i < steps; i++) p = p->next;
    double ns = (now() - t0) / steps * 1e9;
    if (p == NULL) puts("");                               // 用一下 p，防止循环被优化掉
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

```text title="输出（本机示例）"
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

结果是一级一级的台阶：

- 16 KB、32 KB 在 L1 里，1.7 ns，约 5 个周期；
- 256 KB、1 MB 在 L2 里，约 5 ns；
- 8 MB 在 L3 里，25～40 ns；
- 32 MB 以上就接近内存的延迟了，约 90 ns（用大页时）。L3 标称 54 MB，但它由整个插槽的核共享，这台机器又是云上的虚拟机，还要和其他租户分，单个核能用上的远没有这么多。

两列的差别来自 **TLB**：每次访存都要把虚拟地址翻译成物理地址，TLB 缓存了最近用过的翻译。4 KB 的页，512 MB 要 13 万个页表项，TLB 装不下，每次访存还要多走一遍页表；2 MB 的大页，一个表项覆盖的内存大 512 倍，页表遍历几乎消失，512 MB 时从 130 ns 降到 90 ns。虚拟机里这笔账更贵，客户机和宿主机的页表要各走一遍。TLB 和大页的来龙去脉见[虚拟内存、页表与大页](../os/virtual-memory.md)。

这张表解释了很多推理系统里的现象。Python 的列表里存的是对象指针，遍历一个请求列表、查一棵前缀树、在字典里找东西，本质上都是指针追逐，每一步都可能是一次缓存缺失；换成连续存储的数组（NumPy、张量）后，一次搬进来的 64 字节全是有用的数据，硬件预取器还能提前把后面的行读进来。调度器里的元数据该按"结构体的数组"还是"数组的结构体"组织，就是这个道理（见 C++ 手册的[对象布局、对齐与缓存](cpp://memory/layout/)）。

### 组相联与冲突缺失

缓存不是"哪里有空放哪里"。一块 48 KB、12 路组相联的 L1，分成 64 组，每组 12 行：一个地址先按 `行号 mod 64` 决定放进哪一组，只能在这一组的 12 个位置里挑一个，组满了就踢掉最久没用的。于是相距 64 × 64 B = 4 KB 整数倍的地址全都落进同一组。

按列遍历一个 512 × 512 的 float 矩阵：每行 2 KB，同一列上相邻两个元素相距 32 个缓存行，512 行只落进 2 个组，24 个位置装不下 512 行，读进来的行还没等到被用第二次就被踢掉了。每行末尾补 16 个 float（一个缓存行），行距变成 33 个缓存行，各行就分散到所有组里。拖一拖容量、路数和访问跨步，看地址怎么被切成 tag / 组号 / 偏移，以及什么时候开始互相踢：

<div class="aig-widget" data-widget="cachemap"></div>

用一个组相联缓存的模拟器看命中率：

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
        tag = addr // self.line                     # 第几条缓存行
        s = self.sets[tag % self.nsets]             # 行号取模决定放进哪一组
        if tag in s:
            s.move_to_end(tag)
            self.hits += 1
        else:
            self.misses += 1
            if len(s) == self.ways:
                s.popitem(last=False)               # 组满了：踢掉最久没用的
            s[tag] = True


def walk(rows, cols, row_bytes, by_column):
    c = Cache(48 * 1024, ways=12)                   # 和本机 L1d 一样：48 KB、12 路、64 组
    order = ((r, k) for k in range(cols) for r in range(rows)) if by_column else \
            ((r, k) for r in range(rows) for k in range(cols))
    for r, k in order:
        c.access(r * row_bytes + k * 4)             # float 矩阵
    return c.hits / (c.hits + c.misses)


print(f"512x512 按行遍历：        命中率 {walk(512, 512, 512 * 4, False):.1%}")
print(f"512x512 按列遍历：        命中率 {walk(512, 512, 512 * 4, True):.1%}")
print(f"每行补 16 个 float 再按列：命中率 {walk(512, 512, 528 * 4, True):.1%}")
```

```text title="输出"
512x512 按行遍历：        命中率 93.8%
512x512 按列遍历：        命中率 0.0%
每行补 16 个 float 再按列：命中率 93.8%
```

按行遍历时，每 16 个 float 共用一个缓存行，第一次缺失、后 15 次命中，命中率 15/16；按列遍历时，本来下一列能复用这一列读进来的行（容量上完全放得下），却因为都挤在两个组里而全部缺失，这叫**冲突缺失**。补一行之后，同样按列遍历，命中率回到 15/16。

2 的幂的步长要格外小心：矩阵的行宽、哈希表的大小、多维张量的某一维，恰好是 4 KB 或更大的 2 的幂时，就可能撞上这个问题。GPU 共享内存的 [bank 冲突](cuda://basics/memory/#bank-冲突)是同一回事：地址对 32 个 bank 取模，步长是 32 的倍数就全撞在同一个 bank 上，解法也是补一列。

## 缓存一致性：MESI

每个核都有私有的 L1、L2，同一个缓存行可能同时在好几个核的缓存里。硬件用**缓存一致性协议**保证所有核看到的是同一份数据，最常讲的是 MESI，每个核里的每一行处于四种状态之一：

| 状态 | 含义 | 本核读 | 本核写 |
| --- | --- | --- | --- |
| M（Modified） | 只有本核有，并且改过，内存里是旧的 | 直接读 | 直接写 |
| E（Exclusive） | 只有本核有，和内存一致 | 直接读 | 直接写，变成 M |
| S（Shared） | 多个核都有，都和内存一致 | 直接读 | 先让其他核的副本失效，变成 M |
| I（Invalid） | 没有，或者已经失效 | 从别处取来，变成 S 或 E | 取来并让其他副本失效，变成 M |

关键在"写"这一列：**要写一行，必须先独占它**，其他核里这一行的副本全部作废。两个线程写的是不同的变量，但这两个变量恰好在同一个缓存行里，这一行就会在两个核之间来回失效、来回搬运，这就是**伪共享**。两个线程各自累加自己的计数器（原子加），计数器挨在一起还是各占一行：

```c title="false_sharing.c"
// 两个线程各自累加自己的计数器，互不相干；只是计数器放的位置不同
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <time.h>

#define N 20000000L

struct { volatile long a, b; } same;                                // a、b 挨着，在同一条缓存行里
struct { _Alignas(64) volatile long a; _Alignas(64) volatile long b; } padded;   // 各占一条缓存行

typedef struct { volatile long *p; int cpu; } arg_t;

static void *worker(void *v) {
    arg_t *a = v;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(a->cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);      // 绑核，保证两个线程在不同的核上
    for (long i = 0; i < N; i++) __atomic_fetch_add(a->p, 1, __ATOMIC_RELAXED);   // 原子加：每次都要独占这条缓存行
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

```text title="输出（本机示例）"
两个计数器在同一个缓存行：0.64 s
各占一个缓存行：          0.14 s
```

同一个缓存行时慢了近 5 倍。这里用原子加，是因为它每次都必须独占缓存行，受影响最大；本机上换成普通的 `volatile` 自增，两种情况几乎一样快，因为普通的写先进存储缓冲区，拿到缓存行后能一口气写回一串。C++ 里怎么按缓存行隔开数据（`alignas(64)`），见 C++ 手册的[伪共享](cpp://memory/layout/#伪共享)一节。

GPU 的做法又不一样：各个 SM 的 L1 之间**没有**一致性协议，一个 SM 写到全局内存的数据，另一个 SM 的 L1 里可能还是旧值，一致性只在 L2 上保证。所以 GPU 上跨 block 通信，要用原子操作、绕过 L1 的读写，或者带范围的 fence。

## 内存序：存储缓冲区带来的重排

写操作要等拿到缓存行的所有权才能真正写进缓存，这可能要上百个周期。为了不让后面的指令干等，每个核有一个**存储缓冲区**：写操作先放进去，核接着往下执行，缓冲区里的写稍后再落到缓存里。本核后面的读能从缓冲区里读到自己刚写的值，但**其他核要等它落地才看得到**。

这会造成一种反直觉的结果。两个线程各写一个变量，再读对方的变量：

```c title="reorder.c" ci="no"
// 存储缓冲区实验（store buffering）：两个线程各写一个变量、再读对方的变量
// 按"程序顺序"执行的话，r1 和 r2 不可能同时为 0
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
        while (atomic_load_explicit(&go, memory_order_acquire) != i) {}    // 等主线程发令
        atomic_store_explicit(&x, 1, memory_order_relaxed);
        if (use_fence) atomic_thread_fence(memory_order_seq_cst);          // x86 上编译成 mfence
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
        atomic_store_explicit(&go, i, memory_order_release);               // 发令：两个线程同时开跑
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

```text title="输出（本机示例）"
不加屏障：1000000 轮里 r1 == r2 == 0 出现了 195 次
加 mfence：1000000 轮里 r1 == r2 == 0 出现了 0 次
```

如果每个线程的写和读严格按程序顺序、立刻对另一个线程可见，`r1` 和 `r2` 至少有一个是 1。但在 x86 上，两个线程的写都还在各自的存储缓冲区里时，两个读就已经执行了，于是都读到 0。加上 `mfence`（`seq_cst` 的屏障在 x86 上编译成这条指令），它要等存储缓冲区排空才让后面的读执行，这种结果就消失了。

x86 的内存模型叫 TSO（全存储定序），只允许"先写后读"这一种重排，写和写、读和读之间的顺序都保持；ARM 服务器（Grace、鲲鹏、Graviton）的内存模型更弱，几乎什么都能重排，只写原子变量、不写内存序的代码在 x86 上跑得好好的，到 ARM 上就可能出错。GPU 的内存模型同样是弱模型，而且带范围（block、整张 GPU、整个系统）。写这类代码的规则，见 C++ 手册的 [atomic 与内存序](cpp://concurrency/atomics/)。推理系统里，进程之间用共享内存传消息（[vLLM 的共享内存广播队列](../os/ipc.md)）、和内核共享 [io_uring 的环形队列](../os/io-models.md)，靠的都是这些顺序保证。

## SIMD 与峰值算力

一条 SIMD 指令同时处理多个数据：AVX2 的寄存器 256 位，一次处理 8 个 float；AVX-512 一次 16 个。本机每个核有两个 512 位的 FMA 单元，每个周期最多 2 × 16 = 32 次乘加，即 64 FLOP。同一个循环（对每个元素算 8 次乘加），让编译器分别生成三个版本：

```c title="simd.c" ci="no"
// 同一个循环（对每个元素算一个 8 次多项式，8 次乘加），让编译器分别生成标量、AVX2（8 路）、AVX-512（16 路）版本
#include <stdio.h>
#include <time.h>

#define N 1024                       // 数组放得进 L1，每个元素又要算 8 次乘加：测的是计算，不是内存
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
    return 2.0 * 8 * N * REPS / s / 1e9;                        // 每次乘加 = 2 FLOP
}

int main(void) {
    for (int i = 0; i < N; i++) x[i] = (i % 7) * 0.1f;
    printf("标量：   %6.1f GFLOPS\n", gflops(poly_scalar));
    printf("AVX2：   %6.1f GFLOPS\n", gflops(poly_avx2));
    printf("AVX-512：%6.1f GFLOPS\n", gflops(poly_avx512));
    return 0;
}
```

```text title="输出（本机示例）"
标量：     11.4 GFLOPS
AVX2：     87.8 GFLOPS
AVX-512： 165.4 GFLOPS
```

标量版本每个周期 2 次乘加、4 FLOP，按约 3 GHz 算是 12 GFLOPS，实测 11 GFLOPS 左右；AVX2 是它的 8 倍；AVX-512 约 15 倍（跑 AVX-512 时频率会降一些）。粗算整台机器：两颗 CPU 共 32 个核 × 约 165 GFLOPS ≈ 5 TFLOPS。对照一张 H100：FP32 约 67 TFLOPS，BF16 Tensor Core 约 989 TFLOPS，差了两个数量级，这就是矩阵乘都交给 GPU 的原因。

CPU 也在加矩阵单元：Intel 从 Sapphire Rapids 开始有 AMX，用专门的矩阵寄存器做 BF16 和 INT8 的矩阵乘，吞吐比 AVX-512 高一个数量级左右，异构推理把 MoE 专家放在 CPU 上算时常用它。但 CPU 跑 decode 仍然受带宽限制：一路服务器 CPU 有 8～12 个内存通道，DDR5-4800 的 12 通道约 460 GB/s，而一张 H100 的显存带宽是 3.35 TB/s。

## 从 CPU 到 GPU

同样的问题，两种芯片给出了完全不同的答案：

| 问题 | CPU 的做法 | GPU 的做法 |
| --- | --- | --- |
| 访存要几百个周期 | 大缓存、乱序执行、硬件预取，让一个线程少等 | 几十个 warp 轮流发射，一个在等数据，就换另一个 |
| 指令之间有依赖 | 几百项的 ROB 里找独立的指令先执行 | 按顺序发射，靠别的 warp 和线程内的 ILP 填满流水线 |
| 分支 | 分支预测 + 推测执行 | 不预测；warp 内发散时两条路径依次执行 |
| 数据并行 | SIMD：显式的向量寄存器和向量指令 | SIMT：写单个线程的代码，硬件把 32 个线程打包成一个 warp |
| 片上存储 | 几十 MB、由硬件管理的缓存 | 巨大的寄存器文件（每个 SM 256 KB，是 CPU 一个核 L1 的 5 倍）+ 程序员管理的共享内存 |
| 一致性 | 硬件保证所有核的缓存一致 | 各 SM 的 L1 不保证一致，靠 L2 和带范围的 fence |
| 芯片面积花在哪 | 控制逻辑和缓存 | 计算单元和寄存器 |

CPU 追求让一个线程尽快完成（延迟），GPU 追求单位时间完成尽量多的工作（吞吐）。下一章[GPU 的 SM 与 Tensor Core](gpu-sm.md)从一块 GPU 芯片的内部结构讲起。

!!! interview "怎么讲清楚"
    讲 CPU 体系结构或"CPU 和 GPU 有什么区别"：CPU 用超标量、乱序执行、分支预测和多级缓存降低单个线程的延迟；真正的限制是依赖链，需要同时在途的独立指令数 = 延迟 × 吞吐（浮点加法 4 × 2 = 8 个累加器）。缓存按 64 字节的行、按组相联组织，延迟台阶大约 L1 几个周期、L2 十几个、L3 近百个、内存两三百个；随机访问还要加上 TLB 缺失，大页能缓解；2 的幂的步长会造成冲突缺失。多核靠 MESI 保持一致，要写先独占，所以有伪共享；x86 是 TSO，只允许存储缓冲区造成的"先写后读"重排，ARM 和 GPU 更弱。GPU 反过来：按顺序发射、不做分支预测，用海量 warp 切换掩盖延迟，面积花在计算单元和寄存器上，缓存小且 L1 不一致。

## 练习

**1. 要几个累加器。** 本机一个核有两个 512 位的 FMA 单元，FMA 的延迟是 4 个周期。用 AVX-512 算两个很长的 float 向量的点积，至少要几个向量累加器才能跑满？换到 GPU 上，H100 的一个 SM 有 4 个 warp 调度器，每个调度器每周期发射一条指令，如果一条 FMA 指令的结果 4 个周期后才能用，每个调度器至少要有几条独立的 FMA 在途？

??? success "参考答案"
    CPU：延迟 4 × 吞吐 2 = 8，至少 8 个 zmm 累加器，每个累加器里是 16 个部分和，最后再把它们加起来（假设数据在 L1 里，不受带宽限制）。

    GPU：每个调度器每周期发射一条，延迟 4 个周期，需要 4 条独立的 FMA 同时在途。来源有两种：4 个 warp 各有一条（线程级并行），或者 1 个 warp 自己有 4 条互不依赖的 FMA（指令级并行），通常两者都有。实际的 GPU kernel 里还有更长的访存延迟要掩盖，所需的在途工作量要大得多。

**2. 冲突缺失。** 本机 L1d 是 48 KB、12 路、64 组、64 字节的行。一个 `float a[64][1024]` 的矩阵按列遍历，同一列的 64 个元素落进几个组？会发生什么？怎么修？

??? success "参考答案"
    每行 1024 × 4 = 4096 字节，正好是 64 组 × 64 字节。同一列上相邻两个元素相距 4096 字节，行号相差 64，对 64 取模相同，所以同一列的 64 个元素全部落进**同一个组**。这一组只有 12 个位置，64 行轮流进出，读进来的行还没等到下一列复用就被踢掉，几乎每次都缺失（每行 16 个 float 本来可以共用一次缓存行）。

    修法：每行补一个缓存行，声明成 `float a[64][1024 + 16]`，行距变成 4160 字节 = 65 个缓存行，相邻两行落进相邻的组，64 行分散到 64 个组里。或者改变遍历顺序，按行遍历。

**3. CPU 上跑 decode。** 一台服务器的 CPU 有 12 个 DDR5-4800 内存通道（每通道每秒 4800 M 次传输、每次 8 字节），全核 AVX-512 约 5 TFLOPS。在它上面跑一个 7B 的 BF16 模型：batch 为 1 的 decode 每秒最多多少个 token？prefill 1000 个 token 至少要多久？各受什么限制？换成 4 比特量化呢？

??? success "参考答案"
    带宽 = 12 × 4800e6 × 8 ≈ 461 GB/s。BF16 权重 14 GB，batch 为 1 的 decode 每个 token 要把权重读一遍：14 GB / 461 GB/s ≈ 30 ms，每秒最多约 33 个 token；计算量只有 2 × 7e9 = 14 GFLOP，按 5 TFLOPS 算不到 3 ms，所以 decode 受带宽限制。

    prefill 1000 个 token 的计算量是 14 TFLOP，按 5 TFLOPS 至少 2.8 s（实际的矩阵乘效率还要打折），受算力限制；同样的事 H100 的 BF16 Tensor Core 按 50% 利用率只要约 28 ms。

    4 比特量化后权重约 3.5 GB，decode 上限提高到约 130 token/s；prefill 的计算量不变（反量化后仍然按高精度算），不会变快。这就是端侧和 CPU 推理"decode 看带宽、prefill 看算力"的估算方法，和 GPU 上一样。

**4. 内存序。** 两个进程通过共享内存传消息：生产者先写消息内容，再把"已写入"的计数器加一；消费者读到计数器变化后，再读消息内容。在 x86 上，硬件会不会让消费者先看到计数器、后看到内容？代码里还需要做什么？如果要移植到 ARM 服务器呢？

??? success "参考答案"
    x86 的 TSO 保持写和写、读和读之间的顺序，生产者的两次写不会被硬件调换，消费者的两次读也不会，所以在硬件层面不会出错。但**编译器**仍然可能重排或者把读缓存在寄存器里，所以计数器必须用原子操作：生产者用 release 写，消费者用 acquire 读。在 x86 上这两者不产生额外的屏障指令，只约束编译器；到了 ARM 上，编译器会为它们生成真正的屏障指令（或者 `stlr` / `ldar`），代码不用改。

    如果写成"写自己的标志、再读对方的标志"这种互相等待的模式（本章的存储缓冲区实验，也是 Dekker、Peterson 这类锁算法的结构），x86 也会出错，必须用 `seq_cst` 或显式的 `mfence`。

## 小结

- [x] 限制速度的是依赖链：同时在途的独立指令数 = 延迟 × 吞吐，多个累加器让流水线跑满，GPU 上同样要靠 ILP 和足够多的 warp。
- [x] 分支猜错一次损失约二十个周期；难以预测的分支改写成"都算、再选"。GPU 不预测，warp 内发散的两条路径依次执行。
- [x] 缓存按 64 字节的行、按组相联组织：L1 约 5 个周期、L2 约 14 个、L3 近百个、内存两三百个；随机访问还有 TLB 缺失，大页能缓解；2 的幂的步长会造成冲突缺失，补一行解决。
- [x] MESI：要写先独占，同一缓存行里的不同变量会互相拖慢（伪共享），要按缓存行隔开。GPU 各 SM 的 L1 不保证一致。
- [x] x86 的存储缓冲区只造成"先写后读"的重排，需要 `mfence` 或 `seq_cst`；ARM 和 GPU 的内存模型更弱。
- [x] SIMD 让一个核的峰值提高十几倍，但整颗 CPU 仍然比一张 GPU 低两个数量级；CPU 上的 decode 受内存带宽限制。
