# 面试题库

<p class="lead">这一页按主题整理了 CUDA / AI Infra 面试中的高频问题，每题给出要点式的参考答案，并链接到正文中详细讲解的位置。建议的用法：先遮住答案自己说一遍，说不清楚的回到对应章节补课。面试官更看重你能不能讲出"为什么"，以及有没有实测过的数据。</p>

## 面试通常考什么

| 环节 | 内容 | 准备方式 |
| --- | --- | --- |
| 基础问答 | GPU 架构、内存层次、执行模型、同步 | 本页前三部分 |
| 手写 kernel | 归约、转置、softmax、GEMM、LayerNorm、前缀和 | 自己从零写一遍，并能边写边讲优化思路 |
| 性能分析 | 给一个 kernel 或场景，分析瓶颈、提出优化 | Roofline + Nsight 指标，见[性能分析](../tools/profiling.md) |
| 领域知识 | FlashAttention、KV Cache、量化、并行策略、推理引擎 | 本页后半部分 |
| 项目深挖 | 你做过的优化：瓶颈在哪、怎么验证、提升多少 | [作品集项目](projects.md) |

## GPU 架构与编程模型

??? note "1. 线程、warp、block、grid 与 SM 的关系？"
    grid 由 block 组成，block 由线程组成。每个 block 整体调度到一个 SM 上执行，执行期间不会迁移；一个 SM 可以同时驻留多个 block。block 内每 32 个连续线程组成一个 warp，warp 是调度和执行的基本单位，同一 warp 的线程执行同一条指令（SIMT）。详见 [GPU 架构](../basics/gpu-architecture.md#软件层次如何映射到硬件)。

??? note "2. GPU 如何掩盖访存延迟？和 CPU 有什么不同？"
    CPU 用大缓存、乱序执行、分支预测降低单线程的延迟。GPU 在每个 SM 上驻留大量 warp，一个 warp 等待访存时，调度器零开销地切换到另一个就绪的 warp，用吞吐量掩盖延迟。所以 GPU 需要足够的并行度：足够多的驻留 warp（TLP）和每个线程内足够多的独立指令（ILP）。

??? note "3. 什么是 warp divergence？怎么避免？"
    同一 warp 内的线程走向不同分支时，各分支串行执行，不走该分支的线程被屏蔽，耗时是各分支之和。发散只在 warp 内部发生，所以让分支条件在 warp 内一致（按 `threadIdx.x / 32` 分组），或者用 `fmaxf`、选择指令等无分支写法。详见[执行模型](../basics/execution.md#分支发散)。

??? note "4. 什么是占用率？是不是越高越好？"
    SM 上实际驻留的 warp 数与硬件上限之比，受每 SM 的线程数、block 数、寄存器、共享内存四种资源限制。不是越高越好：GEMM、FlashAttention 等 kernel 用大量寄存器和共享内存换取数据复用和 ILP，占用率很低但性能很高。访存瓶颈、每线程工作量小的 kernel 才更依赖高占用率。

??? note "5. 怎么选择 block 大小和 grid 大小？"
    block 取 32 的倍数，128 或 256 是常见起点；grid 至少让每个 SM 有若干个 block，注意尾波效应；数据很大时可以用 grid-stride loop，按 SM 数量决定 grid。最终以实测为准，也可以用 `cudaOccupancyMaxPotentialBlockSize` 作为参考。

??? note "6. `-arch=sm_80` 编译的程序能在 H100 上跑吗？能在 T4 上跑吗？"
    能在 H100 上跑：fatbinary 中包含 compute_80 的 PTX，驱动会 JIT 编译成 sm_90 的 SASS。不能在 T4（sm_75）上跑：没有匹配的 SASS，PTX 也比它新。带 `a` 后缀的 sm_90a 代码只能在 Hopper 上运行。详见[编译流程](../basics/gpu-architecture.md#编译流程ptx-与-sass)。

??? note "7. `__syncthreads()` 放在 if 里面会怎样？"
    block 内所有线程都必须执行到同一个 `__syncthreads()`，只有部分线程能到达时行为未定义，通常是死锁或结果错误。越界的线程不要提前 return，而是跳过计算但仍然参与同步。

??? note "8. Volta 之后的独立线程调度带来了什么变化？"
    warp 内每个线程有独立的程序计数器，发散的线程可以交错执行，不再保证 warp 内隐式同步。老代码里依赖"warp 同步执行"的写法（如 volatile 共享内存展开最后一个 warp）不再安全，要用 `__shfl_*_sync`、`__syncwarp()` 等显式同步的原语。

## 内存层次与访存优化

??? note "9. 说一下 GPU 的内存层次。"
    寄存器（线程私有，最快）→ 共享内存 / L1（每个 SM，block 共享，几十周期）→ L2（全 GPU 共享，几十 MB）→ 显存 HBM（几十 GB，几百周期）。另有常量内存（带广播缓存）和本地内存（寄存器溢出，实际在显存）。优化的核心是让数据留在更高层并被多次复用。详见[内存层次](../basics/gpu-architecture.md#内存层次速览)。

??? note "10. 什么是合并访问？"
    显存以 32 字节扇区为单位传输。一个 warp 的访存请求如果落在尽量少的扇区里（相邻线程访问连续地址），就是合并的；跨步访问会让有效带宽成倍下降。二维数据要让 `threadIdx.x` 沿内存连续的维度变化，数据布局优先 SoA。详见[合并访问](../basics/memory.md#全局内存合并访问)。

??? note "11. 什么是 bank conflict？怎么解决？"
    共享内存分 32 个 bank，每个 4 字节宽。同一 warp 中多个线程访问同一 bank 的不同地址时会串行化；访问同一地址是广播，不冲突。典型场景是按列访问 `[32][32]` 数组。解决办法：填充（`[32][33]`）、swizzle（下标异或）、改变访问模式。可以用 Nsight Compute 的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 验证。

??? note "12. 什么时候用共享内存？"
    两种情况：数据会被 block 内多个线程多次读取（GEMM、卷积的分块复用）；block 内线程之间需要交换数据（归约、转置、扫描）。如果数据只被读一次，放进共享内存没有意义。

??? note "13. 向量化访存（float4）有什么好处和前提？"
    一条指令读 16 字节，减少指令数和地址计算，增加每个线程的在途数据量，有助于跑满带宽。前提：地址 16 字节对齐，元素个数要处理不是 4 的倍数的尾部。

??? note "14. 什么是寄存器溢出？怎么发现和处理？"
    寄存器不够时编译器把变量放到本地内存（在显存上，经缓存），性能大幅下降。用 `-Xptxas -v` 查看 spill 字节数。处理：减少每线程持有的数据、展开循环使局部数组下标为常量、调整 `__launch_bounds__`。

??? note "15. Pinned memory 是什么？为什么重要？"
    锁页内存不会被操作系统换出，DMA 可以直接访问。`cudaMemcpyAsync` 只有在主机内存是锁页内存时才真正异步，也是拷贝与计算重叠的前提。它是有限的系统资源，不能无节制地分配。

## 性能分析

??? note "16. 什么是 Roofline 模型？怎么用？"
    性能上限 = min(峰值算力, 算术强度 × 峰值带宽)。算术强度低于脊点（峰值算力/带宽）的是访存瓶颈，优化方向是减少访存和提高带宽利用率；高于脊点的是计算瓶颈，优化方向是用更快的计算单元、减少无效计算。评价 kernel 时，把实测性能和对应的上限比较。详见 [Roofline](../basics/execution.md#roofline-模型)。

??? note "17. 一个 kernel 很慢，你怎么分析？"
    先用 nsys 确认时间确实花在这个 kernel 上，而不是启动开销、拷贝或 CPU；再用 ncu 看 Speed Of Light，判断是访存瓶颈、计算瓶颈还是延迟瓶颈；然后看对应的部分：Memory Workload（合并、bank 冲突、各级缓存命中）、Warp State（停顿原因）、Occupancy；用 Source 视图定位代码行；修改后重新测量验证。

??? note "18. 访存瓶颈的 kernel 怎么优化？"
    首先减少访存量：算子融合、数据复用、低精度/量化；其次提高带宽利用率：合并访问、向量化、每线程处理多个元素增加在途请求、提高占用率、避免 bank 冲突。目标是有效带宽达到峰值的 80%-90%。

??? note "19. 计算瓶颈的 kernel 怎么优化？"
    用 Tensor Core（低精度矩阵运算）；减少冗余计算；用快速数学函数；提高 ILP 让计算管线满载；检查是否有发散；确认不是被指令发射或寄存器依赖限制。

??? note "20. kernel 之间有很多空隙怎么办？"
    这是启动开销或 CPU 发射速度的问题。用 CUDA Graphs 一次提交整个计算图；融合小 kernel；减少隐式同步；让 CPU 调度与 GPU 执行重叠。推理引擎的 decode 普遍使用 CUDA Graphs。

## 经典 kernel

??? note "21. 手写归约并优化，讲清楚每一步。"
    顺序寻址的共享内存树形归约（无发散、无 bank 冲突）→ 最后一个 warp 用 shuffle → 每个线程先用 grid-stride 在寄存器中累加大量元素并用 float4 读取 → warp shuffle + 共享内存的两级归约 → 多 block 结果用两次 kernel 或原子操作合并（注意浮点原子加不确定）。衡量标准是有效带宽。详见[归约](../kernels/reduction.md#面试怎么答)。

??? note "22. `__shfl_down_sync` 和 `__shfl_xor_sync` 做归约有什么区别？"
    down 版本每步读取 lane + offset 的值，最后只有 lane 0 得到完整结果；xor 版本是蝶形交换，所有 lane 都得到结果，适合 softmax、LayerNorm 这类每个线程都需要归约结果的场景。第一个参数是参与线程的掩码。

??? note "23. 矩阵转置怎么优化？"
    朴素写法读写必有一侧不合并。用共享内存中转：合并地读入 tile，按转置的坐标合并地写出，把跨步访问转移到共享内存上；再用 `[32][33]` 填充消除按列读取的 bank 冲突；用 32×8 的 block 每线程处理 4 个元素。性能上限是同形状的拷贝。详见[矩阵转置](../kernels/transpose.md)。

??? note "24. GEMM 的优化路线？"
    朴素（注意 x 对应列）→ 共享内存分块（访存降到 1/TILE）→ 一维、二维寄存器分块（每线程算 8×8，共享内存读取与计算之比大幅下降，瓶颈转向计算）→ 向量化与 A 转置存储 → 双缓冲 / cp.async 多级流水 → warp 分块、消除 bank 冲突、swizzle → Tensor Core（mma/wgmma）→ TMA + warp 专门化。核心思想是逐级提高数据复用，"访存随周长增长、计算随面积增长"。详见 [GEMM](../kernels/gemm.md#面试怎么答)。

??? note "25. 为什么 softmax 要减最大值？online softmax 是什么？"
    防止指数溢出。online softmax 维护当前最大值 m 和以 m 为基准的指数和 d，读到新元素时按 $d' = d e^{m - m'} + e^{x - m'}$ 更新；两组 (m, d) 也能合并，所以可以并行归约，把三次遍历变成两次。这是 FlashAttention 的基础。详见 [Online softmax](../kernels/softmax-norm.md#online-softmax一次遍历求出最大值和指数和)。

??? note "26. LayerNorm/RMSNorm 怎么实现？方差怎么算才稳定？"
    一个 block（或 warp）处理一行，整行缓存在寄存器里，block 归约求统计量，再逐元素变换。方差用两遍法（先均值再平方差）或 Welford 在线算法，避免 $E[x^2] - E[x]^2$ 的抵消误差。统计量用 FP32 计算。融合残差加法（fused_add_rms_norm）可以减少一次完整的读写。

??? note "27. 并行前缀和怎么做？"
    warp 内用 `__shfl_up_sync` 做 Hillis-Steele（5 步）；block 内每个 warp 扫描后，对各 warp 总和再扫描一次并回加；设备级用"分块扫描 → 扫描块总和 → 回加偏移"的三步法，或 CUB 的单遍 decoupled look-back。应用：流压缩、基数排序、MoE 的 token 分发。详见[前缀和](../kernels/scan.md)。

??? note "28. 怎么用原子操作高效地做直方图？"
    先在共享内存中维护每个 block 私有的直方图，用共享内存原子操作累加，最后每个 block 把结果原子加到全局。争用严重时可以在一个 block 内维护多份副本。

## Tensor Core 与新硬件

??? note "29. Tensor Core 是什么？怎么使用？"
    专门做小矩阵乘加（D = AB + C）的单元，以 warp（Hopper 为 warpgroup）为单位发出指令，算力比 CUDA Core 高一个数量级。接口：WMMA（简单，布局不透明）、mma.sync + ldmatrix（Ampere，布局明确）、wgmma（Hopper，异步，操作数来自共享内存）、tcgen05（Blackwell）。实践中通过 CUTLASS/CuTe、Triton 使用。详见 [Tensor Core](../advanced/tensor-core.md)。

??? note "30. FP16、BF16、FP8 的区别？"
    FP16：5 位指数、10 位尾数，精度较好但范围小；BF16：8 位指数、7 位尾数，范围与 FP32 相同，大模型主流；FP8 有 E4M3（精度优先）和 E5M2（范围优先）两种，需要缩放因子。矩阵乘一般用 FP32 累加。

??? note "31. cp.async 解决了什么问题？"
    直接从全局内存异步拷贝到共享内存，不经过寄存器，并且线程发出后可以继续计算，配合 commit/wait_group 实现多级流水，把访存延迟藏在计算后面。详见 [cp.async](../advanced/async-hopper.md#cpasync-与多级流水-sm_80)。

??? note "32. Hopper 有哪些重要的新特性？"
    TMA（一个线程发起整块多维数据搬运，硬件处理地址、边界和 swizzle）；线程块集群与分布式共享内存；wgmma（warpgroup 级异步矩阵指令）；FP8 Tensor Core；配合 mbarrier 实现生产者-消费者的 warp 专门化。详见 [Hopper](../advanced/async-hopper.md)。

??? note "33. 什么是 warp specialization？"
    把 block 内的 warp 分工：生产者 warp 只负责用 TMA 搬数据，消费者 warp 只负责用 wgmma 计算，两者通过共享内存中的 mbarrier 环形缓冲区同步，搬运和计算持续并行。还可以用 setmaxnreg 把寄存器从生产者转移给消费者。Hopper 上的高性能 GEMM 和 FlashAttention-3 都采用这种结构。

## 注意力与推理

??? note "34. FlashAttention 的原理？为什么快？"
    标准注意力要把 N×N 的 S 和 P 写回显存，是访存瓶颈。FlashAttention 把 Q、K、V 分块，在片上计算 $QK^\top$ 的一块，用 online softmax 更新每行的最大值和指数和，并按比例缩放已累积的输出，从不把 N×N 矩阵写回显存。结果精确，显存读写和占用都大幅下降。详见 [FlashAttention](../advanced/attention.md#flashattention分块--online-softmax)。

??? note "35. FlashAttention-2 和 -3 相对第一版的改进？"
    FA2：在序列维度上并行（小 batch 长序列时 GPU 更满），减少非矩阵乘运算（最后才除以 ℓ），warp 间按 Q 划分避免共享内存通信。FA3：面向 Hopper，TMA + wgmma + warp 专门化，两个 warpgroup 乒乓让 softmax 与 GEMM 重叠，支持 FP8。

??? note "36. prefill 和 decode 的性能特征有什么不同？"
    prefill 处理整段输入，Q/K/V 都很长，GEMM 和注意力都是计算瓶颈，关注 TTFT（首 token 延迟）。decode 每步只生成一个 token，线性层变成 GEMV、注意力要读完整的 KV Cache，是访存瓶颈，关注 TPOT（每 token 延迟）和吞吐。所以 decode 靠批处理、量化、KV Cache 压缩、CUDA Graphs 优化。

??? note "37. KV Cache 有多大？怎么估算？"
    每个 token：2（K 和 V）× 层数 × KV 头数 × 头维度 × 每元素字节数。32 层、8 个 KV 头、头维度 128、BF16 的模型每 token 约 128 KB，32K 上下文约 4 GB。GQA、MLA、KV Cache 量化、前缀共享都是为了降低它。

??? note "38. PagedAttention 解决什么问题？kernel 有什么变化？"
    连续预分配 KV Cache 会因长度未知而浪费大量显存并产生碎片。PagedAttention 把 KV Cache 分成固定大小的块，每个请求用块表记录逻辑块到物理块的映射，按需分配、可共享前缀。kernel 读取 K/V 时多一次通过块表的间接寻址。详见 [PagedAttention](../advanced/attention.md#pagedattention)。

??? note "39. decode 注意力在 batch 很小、序列很长时怎么并行？"
    Flash-Decoding：沿 KV 序列切成多段（split-K），每段一个 block 算出部分输出和 (m, ℓ)，最后按 online softmax 的规则合并。GQA 下让一个 block 同时处理共享同一 KV 头的多个 query 头，KV 只读一次。

??? note "40. GQA 和 MLA 是什么？对 kernel 有什么影响？"
    GQA：多个 query 头共享一组 KV 头，KV Cache 缩小为头数之比；kernel 应该让共享 KV 的 query 头在同一个 block 里处理。MLA（DeepSeek）：把 KV 压缩成低秩的潜向量缓存，计算时通过矩阵吸收把投影合并进 Q 和输出，decode 时注意力的形状变成"很多 query 头共享一个很宽的 KV"，更接近计算密集，需要专门的 kernel（如 FlashMLA）。

??? note "41. 常见的量化方案？什么时候用哪种？"
    weight-only（W8A16、W4A16）：小 batch decode 访存瓶颈时直接减少读取量，在寄存器中反量化；W8A8（INT8/FP8）：大 batch、prefill 计算瓶颈时让计算也变快；FP4（Blackwell）。INT4 通常按组量化（G=128），8 个权重打包进一个 32 位字。详见[量化](../advanced/quantization.md)。

??? note "42. 为什么 W4A16 在大 batch 时加速变差？"
    batch 增大后权重被复用，GEMM 逐渐变成计算瓶颈；W4A16 仍用 FP16 Tensor Core 计算，还多了反量化开销，所以收益下降甚至变慢，这时应该用 W8A8。

??? note "43. 推理引擎（vLLM/SGLang）里有哪些关键的 kernel 和优化？"
    注意力（paged prefill/decode，FlashAttention/FlashInfer 后端）、融合的 RMSNorm/RoPE/激活、量化 GEMM（Marlin、FP8）、fused MoE、采样、自定义 all-reduce；系统层面有连续批处理（continuous batching）、分块 prefill（chunked prefill）、前缀缓存（RadixAttention）、CUDA Graphs、投机解码、PD 分离等。

## 并发与多卡

??? note "44. CUDA 流有什么用？默认流有什么坑？"
    流是按顺序执行的操作队列，不同流之间可以并发，用于拷贝计算重叠、并行执行小任务。传统默认流会和其他阻塞型流隐式同步，破坏并发；用 `cudaStreamNonBlocking` 或 `--default-stream per-thread` 避免。详见[流](../tools/streams.md)。

??? note "45. CUDA Graphs 的原理和限制？"
    把一系列操作录制成图，之后一次调用提交整张图，大幅降低启动开销。限制：参数、形状和内存地址固定；捕获期间不能同步；形状变化需要多张图（推理引擎按 batch 大小分别捕获）。

??? note "46. ring all-reduce 的原理和通信量？"
    reduce-scatter + all-gather，各 p-1 步，每张卡发送约 2(p-1)/p × S 字节，与卡数几乎无关，所有链路同时工作；但步数多，小消息受延迟限制。NCCL 会在 ring、tree、NVLS 等算法间选择。busbw = algbw × 2(p-1)/p。详见 [NCCL](../tools/multi-gpu.md#ring-all-reduce)。

??? note "47. TP、PP、EP 分别需要什么通信？"
    TP 每层 1-2 次 all-reduce（或 all-gather + reduce-scatter）；PP 相邻阶段点对点传激活；EP 每个 MoE 层两次 all-to-all。机内 NVLink 适合 TP，跨机常用 PP/EP/DP。

??? note "48. 怎么让通信和计算重叠？"
    训练中梯度分桶，一桶算完就开始 all-reduce；把 GEMM 与其后的通信分块流水；在同一个 kernel 里融合通信与计算；MoE 用专门的 all-to-all 库（DeepEP）并与计算重叠。

## 开放题

??? note "49. 给你一个新模型的推理服务，TPOT 太高，怎么排查和优化？"
    先测量：nsys 看一个 decode 步骤的时间构成（哪些 kernel、有没有空隙、CPU 开销）。常见问题和对策：kernel 间空隙多 → CUDA Graphs、融合；GEMM/GEMV 占大头 → 检查带宽利用率、考虑量化、增大 batch；注意力占大头 → 用高效的 paged decode kernel、split-K、检查 GQA 是否重复读取；通信占大头（TP）→ 自定义 all-reduce、减少 TP 度数；调度开销 → CPU 与 GPU 重叠。每一步都用数据验证。

??? note "50. 如果让你写一个 fused MoE kernel，你会怎么设计？"
    路由后先统计每个专家的 token 数并做前缀和，得到分组偏移（对齐到 GEMM 分块大小）；把 token 按专家排列（或者只生成排序后的下标，避免真的搬数据）；用分组 GEMM 一次完成所有专家的第一个线性层，融合激活函数（SwiGLU）；第二个线性层同理；最后按路由权重加权并还原顺序（可以融合进 epilogue 或单独的 kernel）。关注：负载不均衡、小专家的 GEMM 效率（M 很小）、量化权重。vLLM 的 fused_moe（Triton）和 SGLang、DeepEP 的实现可以参考。

??? note "51. 你写的 kernel 比 PyTorch 快了 2 倍，怎么证明结果是对的、快是真的？"
    正确性：与参考实现对比，覆盖边界形状（非对齐、极小、极大）、各种 dtype、特殊值（inf、nan、全零），浮点误差要有合理的容差；用 compute-sanitizer 检查越界和竞争。性能：预热后多次测量取中位数，用 CUDA event 或 Nsight；与理论上限（Roofline）比较；在多个形状和 GPU 上测试；说明对比基线是什么（PyTorch 的哪个实现、是否融合）。

## 答题建议

- **先给结论，再讲原因，最后给数据**：比如"这是访存瓶颈，因为算术强度只有 0.25；我测到带宽利用率 85%，已经接近上限"。
- 手写代码时**边写边说**：为什么这样分配线程、为什么这里需要同步、边界怎么处理。写完主动说明可以继续怎么优化。
- 不会的问题，说出你会**怎么去验证**：用什么工具看什么指标，比直接说"不知道"好得多。
- 准备 2-3 个自己的优化案例，能讲清楚完整的"分析 → 假设 → 验证 → 优化 → 结果"过程。
