# GPU 的 SM 与 Tensor Core：一块芯片里有什么

<p class="lead">H100 的 989 TFLOPS 是怎么来的？Tensor Core 的指令为什么一代比一代大，从一个 warp 发起变成四个 warp 一起发起，到 Blackwell 又变成一个线程发起？累加结果为什么要从寄存器搬进专门的 Tensor Memory？注意力里一个不起眼的 exp，为什么会拖住整个 kernel？这一章从芯片的层次结构讲起，拆开 SM 看 warp 调度器、寄存器文件和 Tensor Core，用几个小模型把峰值、延迟掩盖和操作数带宽算清楚。CUDA 手册讲怎么在这些硬件上写程序，这一章讲硬件为什么长成这样。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. H100 SXM 的 BF16 稠密算力 989 TFLOPS 可以拆成哪三个数相乘？
    2. 一个 warp 调度器每周期发射一条指令。一个 warp 每发一条访存就要等 500 个周期，之后有 16 条计算指令，要几个 warp 才能让调度器不闲着？一个调度器最多只能放 16 个 warp，怎么办？
    3. GPU 的寄存器文件为什么那么大（每个 SM 256 KB）？
    4. 从 Ampere 的 `mma.sync` 到 Hopper 的 `wgmma` 再到 Blackwell 的 `tcgen05.mma`，一条矩阵乘指令算的块越来越大，为什么？
    5. FlashAttention 在 Hopper、Blackwell 上为什么要专门处理 softmax 里的 exp？

??? success "自测参考答案（先自己答，再展开对照）"
    1. SM 数 × 每个 SM 每周期的 Tensor Core 运算量 × 频率：132 个 SM × 4096 FLOP（2048 次乘加）× 约 1.83 GHz ≈ 989 TFLOPS。
    2. 每个 warp 一轮要 1 + 500 + 16 个周期，其中只有 17 个周期在发射指令，需要约 517 / 17 ≈ 30 个 warp 轮流发射才能填满，超过了一个调度器 16 个 warp 的上限。办法是提高指令级并行：每个线程一次发出多条互不依赖的访存（比如处理 4 个元素、用向量化的读），每轮 4 条访存、64 条计算时，8 个 warp 就够了。
    3. GPU 靠在 warp 之间切换来掩盖延迟，切换要做到零开销，所有驻留 warp 的寄存器必须同时留在寄存器文件里，不能像 CPU 那样换出去。一个 SM 要驻留几十个 warp、上千个线程，每个线程还要几十上百个寄存器，寄存器文件自然很大；反过来，每个线程用的寄存器越多，能驻留的 warp 越少。
    4. 每代 Tensor Core 的吞吐翻一倍，而共享内存的带宽（每 SM 每周期 128 字节）没变。一条指令算的块越大，每次乘加分摊到的操作数字节越少，才能在不超过共享内存带宽的前提下喂饱 Tensor Core；块大了，发射指令、计算地址的开销也被摊薄。块大到一定程度，累加器也大得寄存器放不下，于是 Blackwell 把累加器挪进 Tensor Memory，由一个线程发起整块运算。
    5. 注意力的每个分数都要做一次 exp，而 exp 由特殊函数单元（SFU）计算，每个 SM 每周期只能做 16 次，远低于 Tensor Core 的吞吐。头维度 128 时，H100 上 exp 需要占到 SFU 能力的一半，FP8 和 Blackwell 上要占满甚至超出，于是 exp 成了瓶颈。FlashAttention-3 让 softmax 和矩阵乘在不同的 warpgroup 之间交错进行，Blackwell 上的实现还用 FMA 单元上的多项式近似分担一部分 exp，Blackwell Ultra 则直接把 SFU 的指数吞吐翻了一倍。

## 从整块芯片到 SM

一块 GPU 芯片是一层套一层的结构。以 Hopper 为例：

```text
GH100 芯片（800 多平方毫米，800 亿个晶体管）
├── 8 个 GPC（图形处理簇）
│   └── 每个 GPC 9 个 TPC（纹理处理簇）
│       └── 每个 TPC 2 个 SM
├── L2 缓存（50 MB，分成两半）
├── 显存控制器（接 HBM）
└── NVLink、PCIe、拷贝引擎等
```

完整的 GH100 有 8 × 9 × 2 = 144 个 SM，但 H100 SXM 只打开了 132 个，H100 PCIe 只有 114 个。原因是**良率**：这么大的芯片上几乎总有几处制造缺陷，厂商设计时就留出冗余，出厂时把有缺陷的单元关掉，按能用的 SM 数分档卖。A100 也一样：完整的 GA100 有 128 个 SM，A100 打开 108 个。

对写 kernel 的人来说，GPC 这一层有一个实际意义：Hopper 的[线程块集群](cuda://advanced/async-hopper/#线程块集群与分布式共享内存-sm_90)只能调度在同一个 GPC 里，集群里的 block 才能互相访问共享内存。

Blackwell 的 B200 走得更远：单个芯片已经做到光刻机能曝光的最大面积，于是把**两个芯片**封装在一起，用每秒 10 TB 的片间互连连成一块 GPU，共 2080 亿个晶体管、148 个 SM。对软件来说它仍然是一个设备，但跨芯片访问 L2 和显存要多走一段路。

## 拆开一个 SM

SM 分成 4 个**处理分区**（SMSP），每个分区像一个小核心，有自己的：

| 部件 | H100 每个分区 | 作用 |
| --- | --- | --- |
| warp 调度器 + 发射单元 | 1 个 | 每个周期从本分区的 warp 里挑一个就绪的，发射一条指令 |
| 寄存器文件 | 16384 个 32 位寄存器（64 KB） | 本分区所有驻留线程的寄存器，一直留在这里 |
| FP32 单元 | 32 个 | 一个周期执行完一条 warp 的 FP32 指令（A100 每分区只有 16 个，要两个周期） |
| INT32 / FP64 单元 | 各 16 个 | 整数运算（地址计算）和双精度 |
| Tensor Core | 1 个 | 矩阵乘加 |
| 特殊函数单元（SFU） | 4 个 | exp、log、sin、rsqrt 等超越函数 |
| 访存单元（LD/ST） | 若干 | 计算地址、发出访存请求 |

四个分区共享 SM 级的资源：256 KB 的 L1 / 共享内存（共享内存最多划 228 KB）、TMA 单元、纹理单元。一个 SM 最多驻留 64 个 warp（每个分区 16 个）、2048 个线程、32 个 block。

几个数值得记住：

- **寄存器文件比共享内存还大**：每个 SM 256 KB，整张 H100 是 132 × 256 KB ≈ 33 MB，和全部 L1 / 共享内存加起来一样多，也接近 50 MB 的 L2。它大，是因为 GPU 在 warp 之间切换不保存、不恢复任何状态：所有驻留 warp 的寄存器同时留在寄存器文件里，切换才能零开销。代价是每个线程用的寄存器越多，能驻留的 warp 越少，这就是[占用率](cuda://basics/execution/#占用率occupancy)。
- **每个分区每周期只发射一条指令**。一个 SM 每周期最多 4 条 warp 指令，每条管 32 个线程。算力要靠每条指令做的事多（Tensor Core 的一条指令做几千到上百万次乘加）来堆，而不是靠多发指令。
- **没有乱序执行，没有分支预测**。发射是按顺序的：一个 warp 的下一条指令依赖的结果没算出来，这个 warp 就等着，调度器去发别的 warp。

## 峰值算力是怎么算出来的

峰值 = SM 数 × 每个 SM 每周期的运算量 × 频率。每周期的运算量来自硬件结构：FP32 是"FP32 单元数 × 2"（一次乘加算 2 次运算）；Tensor Core 是"每个 SM 每周期的乘加数 × 2"。用规格表里的峰值反推频率，就能检验这个拆法：

```python title="peaks.py"
# 峰值 = SM 数 × 每个 SM 每周期的运算量 × 频率。用规格表里的峰值反推频率，检验这个拆法
gpus = [
    # 名称, SM 数, 每 SM 的 FP32 单元, 规格 FP32 TFLOPS, 每 SM 每周期 Tensor Core 稠密 FP16 乘加, 规格 FP16/BF16 TFLOPS
    ("V100", 80, 64, 15.7, 512, 125),
    ("A100", 108, 64, 19.5, 1024, 312),
    ("H100 SXM", 132, 128, 67, 2048, 989.4),
    ("B200", 148, None, None, 4096, 2250),
]
print("GPU       SM数 FP32单元/SM 反推频率  Tensor FLOP/周期/SM 反推频率")
for name, sm, lanes, fp32, fma, tc in gpus:
    f_fp32 = f"{fp32 * 1e12 / (sm * lanes * 2) / 1e9:.2f} GHz" if fp32 else "-"
    f_tc = f"{tc * 1e12 / (sm * fma * 2) / 1e9:.2f} GHz"
    print(f"{name:9s} {sm:4d} {lanes or '-':>11} {f_fp32:>9} {fma * 2:>20} {f_tc:>9}")
```

```text title="输出"
GPU       SM数 FP32单元/SM 反推频率  Tensor FLOP/周期/SM 反推频率
V100        80          64  1.53 GHz                 1024  1.53 GHz
A100       108          64  1.41 GHz                 2048  1.41 GHz
H100 SXM   132         128  1.98 GHz                 4096  1.83 GHz
B200       148           -         -                 8192  1.86 GHz
```

V100 和 A100 两种算法反推出同一个频率，正好是它们的最高加速频率（1.53 GHz、1.41 GHz），说明拆法是对的。H100 有意思：FP32 峰值按 1.98 GHz 算，Tensor Core 峰值却对应 1.83 GHz。规格表里不同单元的峰值并不是按同一个频率算的。Tensor Core 满负荷时功耗最大，功耗墙会把频率压下来，实际跑大矩阵乘时频率往往还要更低，这是实测矩阵乘通常只能达到峰值七八成的原因之一。

表里还能看出每一代的规律：每个 SM 每周期的 Tensor Core 运算量，从 Volta 到 Blackwell 每一代翻一倍（B200 这一格是按 148 个 SM 和规格反推的），而 SM 数和频率只涨了一点。算力的增长几乎全部来自 Tensor Core 本身。

看规格表还要当心几件事：

- **稀疏与稠密**：很多规格表把 2:4 结构化稀疏的数字写在最显眼的位置，是稠密的两倍。估算推理性能用稠密的数字；
- **形态**：同一个型号的 SXM 版和 PCIe 版，SM 数、频率、功耗上限、显存带宽都不同（H100 PCIe 的 BF16 稠密约 756 TFLOPS，SXM 是 989）；
- **精度**：FP8 是 BF16 的两倍，FP4 再翻倍；TF32 是 BF16 的一半，FP32（不走 Tensor Core）又低好几倍。

各型号的具体参数见推理手册的[硬件与生态速查](serving://career/hardware/)。

## 延迟掩盖：要多少个 warp

访存要几百个周期，GPU 不做乱序执行，靠的是**轮流发射**：一个 warp 在等数据，调度器就发别的 warp 的指令。要多少个 warp 才够？用一个小模型模拟一个调度器：每个 warp 反复"发访存、等 500 个周期、发一批计算指令"，看调度器有多少比例的周期在发射指令：

```python title="latency_hiding.py"
# 一个 warp 调度器每周期最多发射一条指令。每个 warp 反复做：连续发出 ilp 条互不依赖的访存，
# 等数据回来（latency 个周期），再发出 compute 条计算指令。模拟调度器有多少比例的周期在发射指令。
# 调度策略是"贪心再取最老"（GTO）：一直发射同一个 warp，直到它要等数据，再换编号最小的就绪 warp。
# 只统计稳定阶段：有任何一个 warp 做完 iters 轮就停止
def simulate(warps, ilp, compute, latency=500, iters=100):
    phase = ["load"] * warps                 # load：发访存；wait：等数据；compute：发计算
    left = [ilp] * warps                     # 本阶段还剩几条指令
    ready_at = [0] * warps                   # 数据到达的周期
    done = [0] * warps                       # 已完成几轮
    cycle = busy = 0
    cur = 0
    while max(done) < iters:
        def ready(w):
            return phase[w] != "wait" or cycle >= ready_at[w]
        if not ready(cur):
            cur = next((w for w in range(warps) if ready(w)), None)
        if cur is None:                      # 所有 warp 都在等数据：这个周期空转
            cycle, cur = cycle + 1, 0
            continue
        w = cur
        if phase[w] == "wait":
            phase[w], left[w] = "compute", compute
        left[w] -= 1                         # 发射一条指令
        busy += 1
        if phase[w] == "load" and left[w] == 0:
            phase[w], ready_at[w] = "wait", cycle + latency
        elif phase[w] == "compute" and left[w] == 0:
            phase[w], left[w], done[w] = "load", ilp, done[w] + 1
        cycle += 1
    return busy / cycle


print("每个调度器上的 warp 数       1     2     4     8    16")
for ilp, compute in [(1, 16), (4, 64)]:
    row = [f"{simulate(w, ilp, compute):5.0%}" for w in (1, 2, 4, 8, 16)]
    print(f"每轮 {ilp} 条访存、{compute:2d} 条计算：" + " ".join(row))
```

```text title="输出"
每个调度器上的 warp 数       1     2     4     8    16
每轮 1 条访存、16 条计算：   3%    7%   13%   26%   52%
每轮 4 条访存、64 条计算：  12%   24%   48%   95%   99%
```

第一行，每轮 1 条访存、16 条计算：每个 warp 一轮要 517 个周期，只有 17 个周期在发射，需要约 517 / 17 ≈ 30 个 warp 才能填满调度器。但一个调度器最多只能放 16 个 warp（一个 SM 64 个），**占用率拉满也只有 52%**。

第二行，每轮先连发 4 条互不依赖的访存，再做 4 倍的计算，访存与计算的比例不变：8 个 warp 就接近满载了。这就是**指令级并行**的作用，和上一章 CPU 上用多个累加器是同一个道理。它说明了为什么高性能 kernel 几乎都让每个线程处理多个元素、用 `float4` 一次读 16 字节：光靠堆 warp 数，掩盖不住访存延迟。估算的通用公式是 Little 定律：需要同时在途的工作量 = 延迟 × 吞吐（见 CUDA 手册的[延迟掩盖](cuda://basics/execution/#延迟掩盖)）。

模拟里的调度策略是"贪心再取最老"（GTO）：一直发射同一个 warp，直到它要等数据才换人。它比"每周期轮换一个 warp"更好，因为轮换会让所有 warp 步调一致，同时发访存、同时等，谁也藏不住谁。

## SIMT 与独立线程调度

程序员写的是**一个线程**的代码，硬件把 32 个线程打包成一个 warp，一条指令同时驱动 32 个线程，这叫 SIMT（单指令多线程）。它和 CPU 的 SIMD 本质相同，区别在于编程模型：SIMD 要程序员（或编译器）显式地用向量寄存器，SIMT 让每个线程看起来都是独立的，分支、循环照写，发散时由硬件屏蔽掉不走这条路径的线程。

从 Volta 开始，warp 里每个线程有自己的程序计数器，发散之后两条路径可以交错执行，不再保证"同一个 warp 的线程天然同步"。所以 warp 内交换数据要用带 `_sync` 的函数，需要同步时显式调用 `__syncwarp()`（见 CUDA 手册的 [warp 编程](cuda://basics/sync-warp/)）。

## Tensor Core：每一代为什么越做越大

Tensor Core 专做矩阵乘加 D = A × B + C。每一代的变化：

| 代 | 架构 | 每 SM 每周期 FP16 乘加 | 指令由谁发起 | A、B 从哪来 | 累加器在哪 | 新增精度 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Volta（V100） | 512 | 一个 warp | 寄存器 | 寄存器 | FP16 |
| 2 | Turing（T4） | 512 | 一个 warp | 寄存器 | 寄存器 | INT8、INT4 |
| 3 | Ampere（A100） | 1024 | 一个 warp（`mma.sync`） | 寄存器（用 `ldmatrix` 从共享内存装入） | 寄存器 | BF16、TF32、2:4 稀疏 |
| 4 | Hopper（H100） | 2048 | 一个 warpgroup，即 4 个 warp（`wgmma`，异步） | B 直接从共享内存读，A 从共享内存或寄存器 | 寄存器 | FP8 |
| 5 | Blackwell（B200） | 4096 | 一个线程（`tcgen05.mma`），可以两个 SM 合作 | 共享内存或 Tensor Memory | Tensor Memory | FP6、FP4、块缩放格式 |

每一代吞吐翻倍，而喂数据的通道没有跟着翻倍：共享内存仍是 32 个 bank、每个 bank 每周期 4 字节，每 SM 每周期 128 字节。一条指令算一个 M × N 的块、K 方向走一步，要读 (M + N) × K 个 16 位的操作数，做 M × N × K 次乘加，每次乘加分摊到的字节数是 2(M + N) / (MN)：**块越大，每次乘加要的数据越少**。估算各代 Tensor Core 满速时，每个 SM 每周期要从共享内存读多少字节：

```python title="operand_bw.py"
# 估算：Tensor Core 满速时，每个 SM 每周期要从共享内存读多少字节的 A、B 操作数（16 位），
# 以及一块累加器（FP32）有多大。共享内存按每 SM 每周期 128 字节（32 个 bank × 4 字节）算
SMEM = 128
cases = [
    # 方案, 每 SM 每周期乘加数, 一个 SM 算的 M, N（每个 K 步）, 这个 SM 从自己的共享内存读的 A 行数、B 列数
    ("Ampere mma.sync（每个 warp 算 64x64）", 1024, 64, 64, 64, 64),
    ("Hopper wgmma m64n64k16", 2048, 64, 64, 64, 64),
    ("Hopper wgmma m64n256k16", 2048, 64, 256, 64, 256),
    ("Blackwell tcgen05 m128n256k16", 4096, 128, 256, 128, 256),
    ("Blackwell 双 SM m256n256k16", 4096, 128, 256, 128, 128),   # 每个 SM 只放一半的 B
]
print("字节/乘加  共享内存 B/周期  占带宽  累加器  方案")
for name, fma, m, n, a_rows, b_cols in cases:
    per_fma = 2 * (a_rows + b_cols) / (m * n)       # 每个 K 步读 (a_rows + b_cols) × 2 字节，做 m × n 次乘加
    need = fma * per_fma
    print(f"{per_fma:8.4f} {need:12.0f} {need / SMEM:11.0%} {m * n * 4 // 1024:5d} KB  {name}")
```

```text title="输出"
字节/乘加  共享内存 B/周期  占带宽  累加器  方案
  0.0625           64         50%    16 KB  Ampere mma.sync（每个 warp 算 64x64）
  0.0625          128        100%    16 KB  Hopper wgmma m64n64k16
  0.0391           80         62%    64 KB  Hopper wgmma m64n256k16
  0.0234           96         75%   128 KB  Blackwell tcgen05 m128n256k16
  0.0156           64         50%   128 KB  Blackwell 双 SM m256n256k16
```

- Ampere 上，一个 warp 用多条 `mma.sync` 算一个 64 × 64 的块，操作数先用 `ldmatrix` 装进寄存器、在寄存器里反复使用，只占共享内存带宽的一半；
- Hopper 的吞吐翻倍，如果还是 64 × 64 的块，就要占满全部共享内存带宽，别的读写全没了位置。所以 `wgmma` 把指令扩大到 4 个 warp 一起算 64 × N（N 最大 256），N = 256 时降到 62%；
- Blackwell 再翻倍，单个 SM 用 128 × 256 的块也要 75%；两个 SM 合作算 256 × 256 时，每个 SM 只需要放一半的 B，回到 50%。这就是双 SM 矩阵乘（CTA pair）存在的理由。

块大了，累加器也跟着变大。128 × 256 的 FP32 累加器是 128 KB：放在寄存器里的话，一个 128 线程的 warpgroup 每个线程要 256 个寄存器，超过了每个线程最多 255 个的上限，更别说还要留寄存器做别的事。每条矩阵乘指令还要把整块累加器读出来、加上、写回去，吞吐翻倍，这部分寄存器带宽也翻倍。所以 Blackwell 给每个 SM 加了 256 KB 的 **Tensor Memory**（128 行 × 512 列 × 32 位），累加器常驻在这里，正好放得下两块 128 × 256 的累加器：一块在做矩阵乘，另一块同时被读出来做收尾（加偏置、转精度、写回显存），两者重叠。

发起方式也随之变化：块越大，一条指令里的工作越多，用不着一整个 warp 去发。Blackwell 上一个线程发起 `tcgen05.mma`，数据由 TMA 搬进共享内存，结果在 Tensor Memory 里，负责计算的线程只剩"发令"和"收尾"。高性能 GEMM 和注意力的结构，也就从 Ampere 的"所有 warp 都搬数据、都做计算"，变成了 Hopper、Blackwell 上的 [warp 专门化](cuda://advanced/async-hopper/#wgmma-与-warp-专门化-sm_90a)：有的 warp 只搬数据，有的只发矩阵乘，有的只做收尾。

## 特殊函数单元与注意力里的 exp

exp、log、rsqrt 这些超越函数由特殊函数单元（SFU）计算，每个 SM 每周期只能做 16 次。H100 上这相当于 132 × 16 × 1.83 GHz ≈ 3.9 T 次每秒，而 Tensor Core 是 989 TFLOPS，差了 250 倍。

平时这不是问题，但注意力里每个分数都要做一次 exp。一个分数前后各有一次矩阵乘（QK<sup>T</sup> 和 PV），头维度为 d 时共 4d 次 FLOP。Tensor Core 满速时，每个 SM 每周期要做多少次 exp？占 SFU 能力的多少？

```python title="exp_bound.py"
# 注意力里每个分数 s = q·k 要做一次 exp，前后是两次矩阵乘：QK^T 和 PV，每个分数各 2d 次 FLOP。
# Tensor Core 满速时，每个 SM 每周期要做多少次 exp？特殊函数单元（SFU）每 SM 每周期只能做 16 次
cases = [
    # GPU/精度, Tensor Core 每 SM 每周期 FLOP, SFU 每 SM 每周期 exp 次数
    ("H100 BF16", 4096, 16),
    ("H100 FP8", 8192, 16),
    ("B200 BF16", 8192, 16),
    ("B200 BF16，SFU 翻倍", 8192, 32),
]
dims = [(64, 64), (128, 128), (576, 512)]            # (q·k 的维度, v 的维度)；最后一个是 MLA 的 decode
print("SFU 要达到的利用率（超过 100% 就是 exp 拖了 Tensor Core 的后腿）")
print("  d=64  d=128    MLA  GPU 与精度")
for name, tc, sfu in cases:
    need = [tc / (2 * dqk + 2 * dv) / sfu for dqk, dv in dims]   # 每周期需要的 exp 次数 ÷ SFU 的能力
    print(f"{need[0]:>6.0%}{need[1]:>7.0%}{need[2]:>7.0%}  {name}")
```

```text title="输出"
SFU 要达到的利用率（超过 100% 就是 exp 拖了 Tensor Core 的后腿）
  d=64  d=128    MLA  GPU 与精度
  100%    50%    12%  H100 BF16
  200%   100%    24%  H100 FP8
  200%   100%    24%  B200 BF16
  100%    50%    12%  B200 BF16，SFU 翻倍
```

头维度 128 的 BF16 注意力在 H100 上，exp 要用掉 SFU 一半的能力，还能藏在矩阵乘后面；换成 FP8，或者换到 Tensor Core 吞吐翻倍的 B200 上，就要占满 SFU，exp 和矩阵乘平起平坐，谁也藏不住谁；头维度 64 更糟。于是：

- **FlashAttention-3**（Hopper）让两个 warpgroup 交错执行：一个在做 softmax 的时候，另一个在做矩阵乘，让 SFU 和 Tensor Core 同时忙；
- Blackwell 上的实现把一部分 exp 用 FMA 单元上的多项式近似来算，分担 SFU 的压力；
- **Blackwell Ultra**（B300）直接把 SFU 的指数吞吐翻倍，NVIDIA 把它称为"注意力层加速"，效果相当于表里最后一行。

MLA 的 decode 就没有这个问题：它的"头维度"是 576 和 512，每个分数对应的矩阵乘多得多，exp 只占 SFU 的一小部分。硬件的每一个单元都有自己的吞吐，一个 kernel 的瓶颈是用得最满的那个单元，不一定是 Tensor Core。

!!! interview "面试怎么答"
    被问 GPU 的硬件结构：芯片 → GPC → TPC → SM；SM 分 4 个分区，每个分区一个 warp 调度器（每周期发射一条指令）、64 KB 寄存器、FP32 单元、一个 Tensor Core 和 SFU；SM 共享 L1 / 共享内存和 TMA。峰值 = SM 数 × 每 SM 每周期运算量 × 频率（H100：132 × 4096 × 约 1.83 GHz ≈ 989 TFLOPS），Tensor Core 每代翻倍是算力增长的主要来源。GPU 不乱序、不预测分支，靠 warp 切换掩盖延迟，寄存器文件大到能同时装下所有驻留 warp 的状态；只靠占用率掩盖不了几百个周期的访存，要加 ILP（Little 定律）。Tensor Core 指令从 warp 级 `mma.sync` 到 warpgroup 级 `wgmma` 再到单线程发起的 `tcgen05`，是因为吞吐翻倍而共享内存带宽不变，只能加大块来降低每次乘加的操作数字节；块大了累加器寄存器放不下，就有了 Tensor Memory 和双 SM 矩阵乘。最后提一句 SFU：注意力里的 exp 在 FP8 和 Blackwell 上会成为瓶颈，所以有 FlashAttention-3 的交错执行和 Blackwell Ultra 的 SFU 翻倍。

## 练习

**1. 拆峰值。** A100 的 TF32 Tensor Core 峰值是 156 TFLOPS，INT8 是 624 TOPS。按本章的拆法，每个 SM 每周期做多少次 TF32 乘加、多少次 INT8 乘加？H100 SXM 的 FP8 稠密峰值是 1979 TFLOPS，它对应的频率是多少？

??? success "参考答案"
    A100 按 108 个 SM、1.41 GHz：TF32 是 156e12 / (108 × 1.41e9) ≈ 1024 FLOP/周期/SM，即 512 次乘加，是 FP16 的一半；INT8 是 624e12 / (108 × 1.41e9) ≈ 4096 OP/周期/SM，即 2048 次乘加，是 FP16 的两倍。数据越窄，同样的硬件每周期能做的乘法越多。

    H100 的 FP8 每 SM 每周期 4096 次乘加（8192 FLOP），1979e12 / (132 × 8192) ≈ 1.83 GHz，和 BF16 一样，FP8 就是 BF16 的两倍。

**2. 延迟掩盖。** 一个逐元素的 kernel，每个线程读一个 float、做 8 条计算指令、写一个 float，访存延迟按 600 个周期算。每个调度器至少要几个 warp？如果改成每个线程处理 4 个元素、先把 4 次读全部发出去呢？这两种情况下，最终的瓶颈是什么？

??? success "参考答案"
    按本章的模型，每个 warp 一轮是 1 条读 + 600 个周期的等待 + 8 条计算 + 1 条写，发射 10 条指令，一轮约 610 个周期，需要约 61 个 warp，远超 16 的上限，调度器大部分时间在空等。每个线程处理 4 个元素、4 次读一起发出后，一轮发射约 40 条指令、仍然只等一次 600 个周期，需要约 16 个 warp，刚好能填满。

    但对这种 kernel，真正的上限是显存带宽，不是发射：每个元素只做 8 次运算、搬 8 个字节，算术强度 1 FLOP/字节，远低于屋脊点。提高 ILP 的目的是让在途的访存足够多，把带宽跑满（Little 定律：在途字节数 = 带宽 × 延迟）。

**3. 块大小与共享内存带宽。** 在 Hopper 上，一个 kernel 用 `wgmma` 算 64 × 128 的块（m64n128k16），按本章的模型，Tensor Core 满速时要占共享内存带宽的多少？如果 A 操作数改成从寄存器读（`wgmma` 允许这样做），又是多少？

??? success "参考答案"
    每次乘加的字节数 = 2 × (64 + 128) / (64 × 128) = 0.047，每 SM 每周期 2048 次乘加，需要 96 字节 / 周期，占 128 字节的 75%。

    A 从寄存器读，共享内存只读 B：2 × 128 / (64 × 128) = 0.031，需要 64 字节 / 周期，占 50%。FlashAttention-3 在算 PV 时就是这样做的：P 刚在寄存器里算完 softmax，直接作为 A 操作数，不用再写回共享内存。

**4. Tensor Memory 的容量。** Blackwell 每个 SM 的 Tensor Memory 是 256 KB。一块 FP32 累加器是 M × N × 4 字节，用 `tcgen05.mma` 算 128 × 256 的块时，Tensor Memory 能同时放几块累加器？为什么需要不止一块？

??? success "参考答案"
    128 × 256 × 4 = 128 KB，256 KB 放得下两块。一块累加器算完之后，要把它读回寄存器做收尾（缩放、加偏置、转成 BF16、写回显存），这需要不少时间；只有一块的话，Tensor Core 要等收尾结束才能开始下一个块。两块轮流用，一块在收尾，另一块已经在做下一个块的矩阵乘，收尾就藏在矩阵乘后面了。这和共享内存里多级流水的道理一样，只是换成了累加器。

## 小结

- [x] 芯片 → GPC → TPC → SM → 4 个分区；为了良率，出厂的 SM 数比完整芯片少（H100 SXM 132 / 144）；B200 是两个芯片拼成的一块 GPU。
- [x] 每个分区每周期发射一条指令，不乱序、不预测分支；寄存器文件大到装得下所有驻留 warp，切换零开销。
- [x] 峰值 = SM 数 × 每 SM 每周期运算量 × 频率；Tensor Core 每代翻倍，是算力增长的主要来源；规格表要分清稠密与稀疏、SXM 与 PCIe。
- [x] 只靠占用率掩盖不了访存延迟，要加指令级并行，按 Little 定律估算在途工作量。
- [x] 吞吐翻倍而共享内存带宽不变，所以 Tensor Core 指令越做越大；累加器随之变大，于是有了 Tensor Memory、单线程发起和双 SM 矩阵乘。
- [x] SFU 每 SM 每周期只能做 16 次 exp，注意力在 FP8 和 Blackwell 上会被 exp 拖住。
