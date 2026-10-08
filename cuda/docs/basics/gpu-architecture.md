# GPU 架构与编程模型

<p class="lead">写出快的 CUDA 代码，前提是脑子里有一张 GPU 的"地图"：计算单元怎么组织、线程怎么被调度、数据放在哪里、每一层有多快。这一章先把这张地图画出来，后面每一个优化技巧都能在这张图上找到原因。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. GPU 为什么能用"大量线程"来掩盖访存延迟？CPU 靠什么？
    2. 线程块（block）和 SM 是什么关系？一个 block 可以跨 SM 执行吗？
    3. warp 是什么？为什么是 32？
    4. `-arch=sm_80` 编译出的程序能在 H100 上运行吗？能在 T4 上运行吗？
    5. A100 的 FP32 算力约 19.5 TFLOPS、显存带宽约 2 TB/s。一个 kernel 每读 1 个 float 只做 2 次浮点运算，它的性能上限由哪个决定？

??? success "自测参考答案（先自己答，再展开对照）"
    1. GPU 在每个 SM 上同时驻留大量 warp，一个 warp 等待访存时，调度器立刻切到另一个就绪的 warp，切换几乎没有开销，只要并行的工作足够多，延迟就被别的 warp 的计算"藏"住了。CPU 靠大缓存、乱序执行和分支预测降低单个线程的延迟。
    2. 一个 block 被整个调度到一个 SM 上，从开始到结束都在那里，不能跨 SM；一个 SM 可以同时驻留多个 block（受寄存器、共享内存、线程数限制）。
    3. warp 是 32 个线程组成的执行单位，同一条指令同时发给这 32 个线程（SIMT）。32 是硬件设计的选择：足够宽以摊薄取指和调度的开销，又不至于让分支发散的浪费太大。
    4. 能在 H100（sm_90）上运行：程序里带有 compute_80 的 PTX，驱动会即时编译成 sm_90 的 SASS；不能在 T4（sm_75）上运行：SASS 和 PTX 都比它新，会报 no kernel image。
    5. 每读 4 字节做 2 次运算，算术强度 0.5 FLOP/字节，远低于 A100 FP32 的屋脊点（约 19.5 T / 2 T ≈ 10 FLOP/字节），性能上限由带宽决定：约 2 TB/s × 0.5 = 1 TFLOPS。

## CPU 与 GPU：两种设计哲学

CPU 为**低延迟**设计：少量强大的核心，大量芯片面积用在缓存、分支预测、乱序执行上，目的是让**单个线程**尽快跑完。

GPU 为**高吞吐**设计：成千上万个简单的计算单元，控制逻辑和缓存相对很少。单个线程很慢，访存一次要等几百个时钟周期，但 GPU 同时驻留着海量线程：一组线程在等内存时，调度器立刻切换去执行另一组已经准备好的线程。只要"准备好"的线程足够多，计算单元就一直有活干，访存延迟被**掩盖**了。

这就是 GPU 编程的第一原则：**给硬件足够多的并行工作**。一个只启动几十个线程的 kernel，在 GPU 上一定跑不快。

## 硬件层次

从大到小：

```text
GPU
├── 显存（HBM / GDDR）：几十到上百 GB，所有 SM 共享
├── L2 缓存：几十 MB，所有 SM 共享
└── 多个 SM（Streaming Multiprocessor，流式多处理器）
    ├── 寄存器文件：64K 个 32 位寄存器（256 KB）
    ├── L1 缓存 / 共享内存：同一块 SRAM，按配置划分
    └── 4 个处理分区（SMSP），每个分区有：
        ├── warp 调度器与指令发射单元
        ├── FP32 / INT32 计算单元（常说的 "CUDA core"）
        ├── Tensor Core（矩阵乘专用单元）
        ├── 特殊函数单元（SFU：exp、sin、rsqrt 等）
        └── 访存单元（LD/ST）
```

**SM 是 GPU 的基本执行单元**，可以粗略地把它看成一个"CPU 核心"，只不过它能同时驻留上千个线程。GPU 的规模主要体现在 SM 的数量上。

几款常见 GPU 的关键规格（来自 NVIDIA 官方规格，Tensor Core 算力为不含稀疏的稠密算力）：

| GPU | 架构 | SM 数 | FP32 | BF16/FP16 Tensor | 显存带宽 | L2 | 每 SM 最大共享内存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T4 | Turing sm_75 | 40 | 8.1 TFLOPS | 65 TFLOPS（FP16） | 320 GB/s | 4 MB | 64 KB |
| A100 SXM 80GB | Ampere sm_80 | 108 | 19.5 TFLOPS | 312 TFLOPS | 2039 GB/s | 40 MB | 164 KB |
| RTX 4090 | Ada sm_89 | 128 | 82.6 TFLOPS | — | 1008 GB/s | 72 MB | 100 KB |
| H100 SXM | Hopper sm_90 | 132 | 67 TFLOPS | 989 TFLOPS | 3.35 TB/s | 50 MB | 228 KB |

记住两个比例，会反复用到：

- **算力 / 带宽**：A100 上 FP32 为 19.5e12 / 2.039e12 ≈ 9.6 FLOP/Byte，BF16 Tensor Core 为 312e12 / 2.039e12 ≈ 153 FLOP/Byte。一个 kernel 每从显存读 1 字节数据，要做这么多次运算才能把算力喂饱。做不到的话，瓶颈就在带宽上。这就是 [Roofline 模型](execution.md#roofline-模型)的核心。
- **片上存储很小**：一个 SM 的寄存器加共享内存只有几百 KB，整个 GPU 的 L2 也只有几十 MB，而显存有几十 GB。高性能 kernel 的本质，是让数据在片上尽可能多地被重复使用。

## 编程模型：线程、线程块、网格

CUDA 用三层结构组织线程：

![图：线程层次](../assets/figures/thread-hierarchy.svg){.aig-svg}

- **线程（thread）**：执行 kernel 函数的最小单位，有自己的寄存器和编号 `threadIdx`。
- **线程块（block）**：一组线程，最多 1024 个，编号 `blockIdx`，大小 `blockDim`。同一个 block 的线程可以通过**共享内存**交换数据，可以用 `__syncthreads()` 同步。
- **网格（grid）**：一次 kernel 启动的所有 block，大小 `gridDim`。不同 block 之间**没有**保证的执行顺序，也不能直接同步。

```cuda
// 启动 grid：numBlocks 个 block，每个 block 有 threadsPerBlock 个线程
kernel<<<numBlocks, threadsPerBlock>>>(args...);

// kernel 内部计算全局线程编号
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`threadIdx`、`blockIdx` 等都是三维的（`.x .y .z`），处理二维图像、矩阵时很方便。

### 软件层次如何映射到硬件

| 软件 | 硬件 | 关键事实 |
| --- | --- | --- |
| grid | 整个 GPU | block 被分配到各个 SM 上执行，数量远超 SM 数时分批执行 |
| block | 一个 SM | 一个 block 从开始到结束**只在一个 SM 上运行**，不会迁移；一个 SM 可以同时驻留多个 block |
| warp | SM 的一个处理分区 | block 内每 32 个连续线程组成一个 warp，是调度和执行的真正单位 |
| thread | 一条执行通道 | 有私有寄存器 |

**一个 SM 能同时驻留多少线程**受几个上限约束（以 A100 为例）：每 SM 最多 2048 个线程（64 个 warp）、最多 32 个 block、65536 个寄存器、164 KB 共享内存。一个 block 要驻留下来，必须同时满足所有这些资源限制。实际驻留的 warp 数占上限的比例叫**占用率（occupancy）**，见[执行模型](execution.md#占用率occupancy)。

## warp 与 SIMT

GPU 以 **warp（32 个线程）** 为单位发射指令：同一时刻，一个 warp 里的线程执行**同一条指令**，各自处理不同的数据。NVIDIA 把这叫做 **SIMT**（Single Instruction, Multiple Threads）。

几个直接后果：

- **block 大小应该是 32 的倍数。** 100 个线程的 block 实际占用 4 个 warp，最后一个 warp 有 28 个线程空转。
- **同一 warp 内的分支会串行化。** 如果 warp 里一半线程走 `if`、一半走 `else`，两条路径会先后执行，各自屏蔽掉另一半线程。这叫**分支发散（warp divergence）**，见[执行模型](execution.md#分支发散)。
- **访存按 warp 合并。** 一个 warp 的 32 个线程同时发出访存请求，如果它们访问的是连续的地址，硬件能合并成很少的几次内存事务。这是访存优化的第一要点，见[内存层次](memory.md#全局内存合并访问)。

每个时钟周期，每个 warp 调度器从它管理的 warp 中挑一个"就绪"（操作数已经准备好）的 warp 发射指令。一个 warp 在等待访存结果时不会占用计算单元，调度器会去执行别的 warp。**切换 warp 没有开销**，因为所有驻留 warp 的寄存器都一直保存在寄存器文件里，这和 CPU 的线程切换完全不同。

!!! warning "独立线程调度：不要假设 warp 内线程天然同步"
    从 Volta（sm_70）开始，warp 内每个线程有自己的程序计数器，发散后的线程可以交错执行。老代码里"同一个 warp 的线程一定同步执行，所以不用同步"的写法不再安全。warp 内交换数据要用带 `_sync` 后缀的内置函数（如 `__shfl_sync`），需要同步时用 `__syncwarp()`。详见 [warp 级编程](sync-warp.md)。

## 内存层次速览

| 存储 | 位置 | 作用域 | 容量量级 | 延迟量级 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 寄存器 | SM 内 | 单个线程 | 每线程最多 255 个 | ~1 周期 | 最快；用太多会降低占用率或溢出 |
| 共享内存 | SM 内 | 同一 block | 每 SM 几十到两百多 KB | ~几十周期 | 程序员手动管理的高速缓存 |
| L1 缓存 | SM 内 | 硬件管理 | 与共享内存共用 | ~几十周期 | 缓存全局内存访问 |
| L2 缓存 | 芯片上 | 所有 SM | 几 MB 到几十 MB | ~200 周期 | 所有全局内存访问都经过 L2 |
| 全局内存 | 显存 | 所有线程，跨 kernel 持久 | 几十 GB | ~400-800 周期 | `cudaMalloc` 分配的内存 |
| 常量内存 | 显存 + 专用缓存 | 所有线程，只读 | 64 KB | 缓存命中时很快 | 适合 warp 内所有线程读同一个值 |
| 本地内存 | 显存 | 单个线程 | — | 同全局内存 | 寄存器溢出、动态索引的局部数组会放到这里，很慢 |

![图：H100 的存储层次](../assets/figures/memory-hierarchy.svg){.aig-svg}

延迟数字只是数量级，不同架构差别不小。但它们之间的**相对关系**稳定：寄存器 ≪ 共享内存 ≈ L1 ≪ L2 ≪ 显存。优化访存，就是想办法让数据留在更靠上的层次。下一章 [内存层次与访存优化](memory.md) 会详细展开。

## 架构代号与计算能力

每一代 GPU 有一个**计算能力**（Compute Capability，CC）版本号，编译时用 `sm_XY` 表示：

| 架构 | CC | 代表产品 | 与本手册相关的新特性 |
| --- | --- | --- | --- |
| Volta | 7.0 | V100 | 第一代 Tensor Core、独立线程调度 |
| Turing | 7.5 | T4、RTX 20 | INT8/INT4 Tensor Core；**CUDA 13 支持的最低架构** |
| Ampere | 8.0 / 8.6 | A100 / RTX 30、A10 | BF16、TF32、`cp.async` 异步拷贝、更大的共享内存 |
| Ada Lovelace | 8.9 | RTX 40、L4、L40、L20 | FP8 Tensor Core |
| Hopper | 9.0 | H100、H800、H20 | TMA、线程块集群、wgmma、FP8 Transformer Engine |
| Blackwell | 10.0 / 10.3 / 12.0 | B200 / B300 / RTX 50 | 第五代 Tensor Core（tcgen05）、Tensor Memory、FP4 |

写代码时要清楚每个特性需要的最低架构，本手册会用 <span class="arch">sm_80+</span> 这样的标记注明。

## 编译流程：PTX 与 SASS

`nvcc` 把一个 `.cu` 文件拆成两部分：主机代码交给 g++ 编译；设备代码先编译成 **PTX**（一种与具体硬件无关的虚拟汇编），再由 `ptxas` 编译成 **SASS**（某一代 GPU 的真实机器码）。最终的可执行文件里嵌着一个 "fatbinary"，可以同时包含多份 SASS 和 PTX。

```bash
nvcc -arch=sm_80 app.cu          # 等价于同时生成 sm_80 的 SASS 和 compute_80 的 PTX
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 app.cu   # 多架构 + PTX
nvcc -arch=native app.cu         # 为本机的 GPU 编译
```

运行时，驱动先找与当前 GPU 匹配的 SASS；找不到的话，用 PTX **即时编译（JIT）**成当前 GPU 的 SASS。所以：

- `-arch=sm_80` 编译的程序**可以**在 H100（sm_90）上运行：靠 PTX JIT。
- 但**不能**在 T4（sm_75）上运行：SASS 和 PTX 都比它新，会报 `no kernel image is available for execution on the device`。
- 带 `a` 后缀的目标（如 `sm_90a`、`sm_100a`）启用只有这一代才有的特性（如 Hopper 的 wgmma），生成的代码**不向前兼容**，只能在这一代 GPU 上运行。CUDA 12.9 起还有 `f` 后缀（如 `sm_100f`），表示同一"家族"内可移植的特性。

查看一个 kernel 用了多少寄存器和共享内存，在编译时加 `--ptxas-options=-v`（或 `-Xptxas -v`）：

```text
ptxas info    : Used 14 registers, used 0 barriers, 360 bytes cmem[0]
```

这个信息在分析占用率时非常有用。

!!! interview "怎么讲清楚"
    讲"GPU 和 CPU 有什么不同"：CPU 用大缓存、乱序执行和分支预测降低单线程延迟，GPU 用海量线程在 warp 之间切换来掩盖延迟，所以第一原则是给它足够多的并行工作。层次：grid → block → warp → thread；一个 block 固定在一个 SM 上，warp（32 个线程）是调度和执行的单位。再用算力与带宽之比判断瓶颈：A100 的 FP32 约 19.5 TFLOPS、带宽约 2 TB/s，每读 4 字节只算 2 次的 kernel 必然受带宽限制。编译：`-arch` 决定生成哪些 SASS / PTX，PTX 可以 JIT 到更新的 GPU，反过来不行。

## 练习

**1. 一张 A100 最多能同时驻留多少个线程？** 如果一个 kernel 启动了 10 万个线程，会发生什么？

??? success "参考答案"
    108 个 SM × 每 SM 2048 个线程 = 221,184 个线程（6912 个 warp）。

    启动 10 万个线程没问题：只要资源允许，所有 block 能同时驻留。如果启动 1000 万个线程，超出驻留上限的 block 会排队，等前面的 block 执行完释放资源后再被调度上去。block 之间没有执行顺序的保证，这也是 block 之间不能互相等待的原因：等待的那个 block 可能永远等不到被调度的机会。

**2. 占用率计算。** A100 上，一个 kernel 每个 block 256 个线程，每个线程用 64 个寄存器，不使用共享内存。一个 SM 最多能驻留几个 block？占用率是多少？如果每线程寄存器降到 32 个呢？

??? success "参考答案"
    - 64 个寄存器：每个 block 需要 256 × 64 = 16384 个寄存器，一个 SM 有 65536 个，最多 4 个 block，即 1024 个线程、32 个 warp，占用率 32/64 = 50%。
    - 32 个寄存器：每个 block 需要 8192 个寄存器，寄存器允许 8 个 block；线程数上限 2048 / 256 = 8 个 block；block 数上限 32。取最小值 8 个 block，2048 个线程，占用率 100%。

    （实际分配时寄存器按一定粒度取整，精确数值可以用 Nsight Compute 或 CUDA 的 Occupancy Calculator 查看。）

**3. 判断瓶颈。** 一个 kernel 读两个 float 数组、写一个 float 数组（`c[i] = a[i] * b[i] + 1`），每个元素做 2 次浮点运算。在 A100 上，它的性能瓶颈是算力还是带宽？理论上处理 10 亿个元素最少需要多久？

??? success "参考答案"
    每个元素读写 12 字节，做 2 次运算，**算术强度** = 2 / 12 ≈ 0.17 FLOP/Byte，远低于 A100 FP32 的 9.6 FLOP/Byte，所以是**带宽瓶颈**。

    10 亿个元素共需搬运 12 GB 数据，按 2039 GB/s 计算，理论最短时间约 5.9 ms。实际能达到的带宽通常是峰值的 80%-90%，所以大约 6.5-7.5 ms。对这类 kernel，优化目标就是让实测带宽接近峰值，算力根本不重要。

**4. 编译选项。** 团队的推理服务要部署在 A100、A10（sm_86）和 H100 上，你会怎么设置 `-gencode`？

??? success "参考答案"
    为每个目标架构生成 SASS，保证启动时不需要 JIT；再附带一份最高版本的 PTX，兼容未来的 GPU：

    ```bash
    nvcc -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_90,code=sm_90 \
         -gencode arch=compute_90,code=compute_90 ...
    ```

    如果 H100 上要用 wgmma 等 Hopper 专属指令，需要用 `sm_90a`，并且这部分代码要和其他架构的实现分开编译和分派。

## 小结

- [x] GPU 靠海量线程掩盖延迟，第一原则是给它足够多的并行工作。
- [x] grid → block → warp → thread；block 固定在一个 SM 上，warp（32 线程）是执行单位。
- [x] 片上存储（寄存器、共享内存、L1/L2）小而快，显存大而慢；性能的关键是数据复用。
- [x] 算力与带宽之比决定了 kernel 是计算瓶颈还是访存瓶颈。
- [x] `-arch` 决定生成哪些 SASS/PTX；PTX 可以 JIT 到更新的 GPU，反之不行。
