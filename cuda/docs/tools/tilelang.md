# TileLang：比 Triton 低一层的 tile 语言

<p class="lead">Hopper 和 Blackwell 上真正快的 kernel 长这样：一部分 warp 专门用 TMA 搬数据、一部分专门发 wgmma、中间用 mbarrier 做生产者-消费者交接。这些东西 Triton 藏在编译器里，手写 CUDA 又太费劲。TileLang 正好卡在中间——用 Python 写块级的搬运和矩阵乘，但保留对流水线级数、共享内存布局、warp 分工的控制。这一章写两个能编译的 kernel（GEMM 和 FlashAttention），看它生成的 CUDA 源码里出现了什么，再和 Triton、CuTe DSL 对照着说清楚各自适合什么。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. TileLang 和 Triton 的分工有什么不同？
    2. `T.Pipelined(..., num_stages=3)` 做了什么？为什么需要它？
    3. `T.alloc_shared` 和 `T.alloc_fragment` 分别对应硬件上的什么？
    4. 一个二十行的 TileLang GEMM，在 Hopper 上会生成什么样的 CUDA 代码？
    5. 什么时候该用 TileLang，什么时候用 Triton 或 CuTe DSL？

??? success "自测参考答案（先自己答，再展开对照）"
    1. Triton 让你决定"每个程序实例处理哪一块数据"，块**内部**的线程映射、共享内存、流水线、Tensor Core 指令全交给编译器。TileLang 让你继续往下控制：显式分配共享内存和寄存器片段、指定流水线级数、必要时指定每块数据的布局与 warp 分工，而循环内的线程映射仍由编译器补全。
    2. 把"搬下一块数据"和"算当前这块"重叠起来：编译器把循环体拆成生产者（TMA / cp.async）和消费者（wgmma），用 `num_stages` 个共享内存缓冲区和 mbarrier 轮转。没有它，每一轮都要等数据到齐才能算，Tensor Core 大部分时间在等。
    3. `alloc_shared` 是共享内存里的一块（对应 `__shared__`），`alloc_fragment` 是分布在 warp 寄存器里的一块（对应 mma 的累加器 fragment）。数据流一般是：全局 → 共享（TMA）→ 寄存器片段（mma 的操作数与累加器）→ 全局。
    4. 本章实测：生成 99 行 CUDA，里面有 `CUtensorMap` 描述符、`tl::tma_load`、9 个 `mbarrier`、`wgmma_ss`、以及 `warpgroup_reg_dealloc/alloc`——也就是一套完整的 warp 专门化流水线，手写要几百行。
    5. 常规的逐元素、归约、简单 GEMM 交给 `torch.compile` 或 Triton；需要精确控制 Hopper/Blackwell 特性（TMA、wgmma、tcgen05、warp 专门化）又不想写几百行 CUTLASS 模板时用 TileLang；需要极致控制布局代数、或者要和 CUTLASS 生态对接时用 CuTe DSL。

## 为什么会有 TileLang

[AI 编译器全景](../framework/compilers.md#新一代的-tile-语言)里提到过这条分界线：Triton 的抽象在 Ampere 上很合适，但到了 Hopper，决定性能的几件事恰好都在"块内部"——

- **TMA**：一条指令搬一整块多维数据，需要主机端的张量描述符和 mbarrier 计数；
- **wgmma**：四个 warp 组成一个 warpgroup 一起发的异步矩阵乘，操作数可以直接来自共享内存；
- **warp 专门化**：一部分 warp 只搬数据、一部分只计算，两边用 mbarrier 做环形缓冲区交接；
- **寄存器再分配**：生产者 warp 不需要那么多寄存器，可以让给消费者（`setmaxnreg`）。

这些都能写进 Triton 编译器，但你无法直接指挥它；而手写 CUDA 或 CUTLASS 模板的代价又很高。TileLang（构建在 TVM 之上）的取舍是：**保留块级的 Python 写法，同时把流水线、布局、warp 分工暴露成可调的参数**。

## 第一个 kernel：GEMM

![图：以块为单位的编程模型——一个 program 处理一个 BLOCK](../assets/figures/triton-program-grid.svg){.aig-svg}

```bash
pip install tilelang        # 本章用 0.1.15；运行需要 NVIDIA GPU，只生成代码则需要 CUDA 工具链
```

```python title="tl_matmul.py" run="no"
# TileLang 的 GEMM：在 Python 里直接写"块级"的搬运与矩阵乘，编译器负责线程映射与流水线
# 装：pip install tilelang；跑：需要 NVIDIA GPU（编译成 CUDA 源码则只需要 CUDA 工具链）
from pathlib import Path

import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M=128, block_N=128, block_K=64, num_stages=3, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        # 每个 block 负责输出里的一个 block_M x block_N 的 tile
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")   # 共享内存里的两块输入
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")    # 累加器放在寄存器里
            T.clear(C_local)
            # Pipelined：编译器自动把"搬下一块"和"算这一块"重叠起来，num_stages 是流水级数
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)        # 全局 -> 共享（自动用 TMA / cp.async）
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)                    # 自动展开成 mma / wgmma
            T.copy(C_local, C[by * block_M, bx * block_N])             # 累加器 -> 全局
    return kernel


target = {"kind": "cuda", "arch": "sm_90a"}       # 没有 GPU 时显式指定架构，只做代码生成
jit_kernel = tilelang.compile(matmul(1024, 1024, 1024), out_idx=[2], target=target)
src = jit_kernel.get_kernel_source()
Path("gen.cu").write_text(src)
print(len(src.splitlines()), "行 CUDA 源码")
```

读这段代码的顺序和写 CUDA 一样，只是每一步都在"块"这个粒度上：

| 这一行 | 对应 CUDA 里的什么 |
| --- | --- |
| `T.Kernel(gx, gy, threads=128)` | `kernel<<<dim3(gx, gy), 128>>>`，`bx`、`by` 就是 `blockIdx` |
| `T.alloc_shared((block_M, block_K), "float16")` | `__shared__ half A_shared[block_M][block_K]` |
| `T.alloc_fragment((block_M, block_N), "float")` | mma 累加器所在的寄存器片段 |
| `T.copy(A[by * block_M, ko * block_K], A_shared)` | 全局 → 共享的整块搬运（编译器选 TMA 或 `cp.async`） |
| `T.gemm(A_shared, B_shared, C_local)` | 展开成一串 `mma` / `wgmma` |
| `T.Pipelined(n, num_stages=3)` | 多级流水：多开几块共享内存缓冲区，搬运与计算重叠 |

注意**没有出现 `threadIdx`**：块内每个线程负责哪些元素，是编译器根据 layout 推导出来的。这就是"tile 语言"的含义——你管块，编译器管块内。

## 它生成了什么

上面那段 Python 编译到 `sm_90a`（Hopper）后是 99 行 CUDA。挑几段关键的看：

```cuda title="生成的 CUDA（节选）" run="no"
extern "C" __global__ void __launch_bounds__(256, 1)
kernel_kernel(__grid_constant__ const CUtensorMap A_desc,      // TMA 的张量描述符
              __grid_constant__ const CUtensorMap B_desc,
              half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];     // 3 级流水的共享内存缓冲区
  __shared__ __align__(16) uint64_t mbarrier_mem[6];           // 3 个"数据已到"+ 3 个"缓冲区已空"

  if (((int)threadIdx.x) >= 128) {                             // 生产者 warpgroup：只搬数据
    tl::warpgroup_reg_dealloc<24>();                           //   把寄存器让给消费者
    for (int ko = 0; ko < 16; ++ko) {
      mbarrier[((ko % 3) + 3)].wait(...);                      //   等这个缓冲区被消费完
      mbarrier[(ko % 3)].expect_transaction(16384);
      tl::tma_load(A_desc, mbarrier[(ko % 3)], ..., (ko * 64), (((int)blockIdx.y) * 128));
    }
  } else {                                                     // 消费者 warpgroup：只算
    tl::warpgroup_reg_alloc<240>();
    for (int ko_1 = 0; ko_1 < 16; ++ko_1) {
      mbarrier[(ko_1 % 3)].wait(...);                          //   等数据到位
      tl::wgmma_ss<tl::DataType::kFloat16, ..., 64, 128, 16, ...>(desc_a, desc_b, C_local, ...);
      mbarrier[((ko_1 % 3) + 3)].arrive();                     //   通知生产者这块可以覆盖了
    }
  }
}
```

二十行 Python 变成了一套完整的 Hopper 流水线：TMA 描述符、6 个 mbarrier 组成的环形缓冲区、生产者-消费者的 warp 专门化、寄存器再分配、wgmma。手写这套东西要几百行，而且每处同步都可能写错。

调试时的常用手段就是把这段源码打出来看：**有没有用上 `tma_load` 和 `wgmma`**（没有就说明布局或形状不满足条件，退化成了 `cp.async` + `mma`）、**流水级数对不对**、**共享内存有没有超**。TileLang 也提供 `tilelang.disassemble` 看 PTX/SASS。

## FlashAttention：在线 softmax 写成 tile 代码

```python title="tl_flash.py" run="no"
# TileLang 写的 FlashAttention 前向（因果掩码），展示在线 softmax 怎么写成 tile 级代码
from pathlib import Path

import tilelang
import tilelang.language as T


def flash_attn(batch, heads, seq, dim, block_M=64, block_N=64, num_stages=2, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504          # 乘 log2(e)，之后用 exp2 更快
    shape = (batch, seq, heads, dim)

    @T.prim_func
    def kernel(Q: T.Tensor(shape, "float16"), K: T.Tensor(shape, "float16"),
               V: T.Tensor(shape, "float16"), Out: T.Tensor(shape, "float16")):
        with T.Kernel(T.ceildiv(seq, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), "float16")
            K_shared = T.alloc_shared((block_N, dim), "float16")
            V_shared = T.alloc_shared((block_N, dim), "float16")
            scores = T.alloc_fragment((block_M, block_N), "float")     # QK^T 的一块
            probs = T.alloc_fragment((block_M, block_N), "float16")
            acc = T.alloc_fragment((block_M, dim), "float")            # 输出累加器
            row_max = T.alloc_fragment((block_M,), "float")            # 在线 softmax 的两个统计量
            row_sum = T.alloc_fragment((block_M,), "float")
            prev_max = T.alloc_fragment((block_M,), "float")
            rescale = T.alloc_fragment((block_M,), "float")

            T.copy(Q[bz, bx * block_M, by, 0], Q_shared)
            T.fill(acc, 0)
            T.fill(row_sum, 0)
            T.fill(row_max, -T.infinity(scores.dtype))
            loop_end = T.ceildiv((bx + 1) * block_M, block_N)          # 因果掩码：只看自己之前的块
            for ko in T.Pipelined(loop_end, num_stages=num_stages):
                T.copy(K[bz, ko * block_N, by, 0], K_shared)
                T.clear(scores)
                T.gemm(Q_shared, K_shared, scores, transpose_B=True)   # QK^T
                for i, j in T.Parallel(block_M, block_N):              # 块内的因果掩码
                    scores[i, j] = T.if_then_else(bx * block_M + i >= ko * block_N + j,
                                                  scores[i, j] * scale, -T.infinity(scores.dtype))
                T.copy(row_max, prev_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)      # 更新行最大值
                for i in T.Parallel(block_M):
                    rescale[i] = T.exp2(prev_max[i] - row_max[i])      # 旧的累加值要按新最大值缩放
                for i, j in T.Parallel(block_M, block_N):
                    probs[i, j] = T.exp2(scores[i, j] - row_max[i])
                for i, j in T.Parallel(block_M, dim):
                    acc[i, j] *= rescale[i]
                for i in T.Parallel(block_M):
                    row_sum[i] *= rescale[i]
                T.reduce_sum(probs, row_sum, dim=1, clear=False)
                T.copy(V[bz, ko * block_N, by, 0], V_shared)
                T.gemm(probs, V_shared, acc)                           # PV
            for i, j in T.Parallel(block_M, dim):
                acc[i, j] /= row_sum[i]                                # 最后统一除以归一化因子
            T.copy(acc, Out[bz, bx * block_M, by, 0])
    return kernel


target = {"kind": "cuda", "arch": "sm_90a"}
jit_kernel = tilelang.compile(flash_attn(1, 8, 1024, 64), out_idx=[3], target=target)
src = jit_kernel.get_kernel_source()
Path("flash.cu").write_text(src)
print(len(src.splitlines()), "行 CUDA 源码")
```

这段代码的结构就是 FlashAttention 的伪代码本身（见 CUDA 手册的 [FlashAttention 与推理算子](../advanced/attention.md)）：

- 每个 block 负责 `block_M` 个 query，沿 key 的方向分块遍历；
- `row_max` 和 `row_sum` 是在线 softmax 的两个统计量，每处理完一块就按新的最大值**缩放**已有的累加结果（`rescale`）；
- 用 `exp2` 代替 `exp`（提前把 scale 乘上 `log2(e)`），因为 GPU 上 `exp2` 是单条硬件指令；
- 因果掩码分两层：`loop_end` 跳过整块在对角线右上的、`T.if_then_else` 处理对角线上那一块。

编译到 Hopper 后是 224 行 CUDA，同样带着 TMA、mbarrier 和 wgmma。作为对照，FlashAttention-3 的手写 CUTLASS 实现是几千行。

TileLang 不会自动帮你做算法层面的事：在线 softmax 的缩放顺序、掩码的处理、什么时候除以归一化因子，都要自己写对。它省掉的是**把算法映射到硬件**的那一层苦工。

## 和 Triton、CuTe DSL 的对照

| | Triton | TileLang | CuTe DSL / CUTLASS |
| --- | --- | --- | --- |
| 抽象层次 | 块级，块内全自动 | 块级 + 可控的流水线、布局、warp 分工 | 布局代数，精确到每个线程持有哪些元素 |
| 代码量（GEMM） | 最短 | 略长 | 长，但最灵活 |
| Hopper/Blackwell 特性 | 编译器决定，可控性弱（Gluon 在补） | 显式可控 | 完全可控 |
| 自动调优 | `triton.autotune` | `tilelang.autotune`（搜块大小、流水级数、线程数） | 模板参数 + profiler 选 tactic |
| 生态 | 最成熟，PyTorch Inductor 的后端 | 新，算子库在成长 | NVIDIA 官方，FlashAttention-3、DeepGEMM 的基础 |
| 适合 | 绝大多数自定义算子 | 需要 Hopper/Blackwell 特性的关键算子 | 极致性能、要和 CUTLASS 对接 |

一个现实的选择顺序：

1. 先看能不能不写 kernel——`torch.compile` 的融合、已有的库（FlashInfer、FlashMLA、DeepGEMM）；
2. 要写就先写 Triton，够快就停；
3. Triton 明显打不满（Nsight 显示 Tensor Core 利用率低、或者没用上 TMA/wgmma），再用 TileLang 精调这一个算子；
4. 还要更极致、或者要贡献回 CUTLASS 生态，才动 CuTe DSL。

国内的模型团队在开源算子时经常同时给 Triton 和 TileLang 两个版本（例如 DeepSeek 在 V3.2 里放出的稀疏注意力算子），因为前者好读、后者在 Hopper 上更快。

## 工程上的几个注意点

- **版本与硬件绑定**：TileLang 还在快速迭代，API 有变化；生成的代码对 `sm_90a` / `sm_100a` 这些带 `a` 后缀的目标才会用上 wgmma / tcgen05，编译时要指定对架构。
- **正确性要自己验**：和 PyTorch 的参考实现逐元素对拍（注意累加顺序不同会有浮点误差，用 `rtol`），再测边界形状（非整除的 M/N/K、序列长度不是块大小的整数倍）。
- **先测带宽和算力的上限**：写之前用屋顶线算出这个算子的理论下限，写完对比实测，别在一个本来就受带宽限制的算子上花力气调流水线。
- **自动调优要固定环境**：搜出来的最优配置和 GPU 型号、CUDA 版本、甚至功耗上限有关，换机器要重搜；生产上把结果固化成配置表。

!!! interview "怎么讲清楚"
    讲"Triton 之外还了解什么 kernel 语言"：先说分界线——Ampere 上 Triton 够用，到 Hopper/Blackwell，决定性能的 TMA、wgmma、warp 专门化、寄存器再分配都在块内部，Triton 不给控制权。TileLang 保留块级 Python 写法，但把流水线级数、共享内存布局、warp 分工暴露出来；一个二十行的 GEMM 会生成带 TMA 描述符、mbarrier 环形缓冲、生产者-消费者 warpgroup 和 wgmma 的 CUDA，手写要几百行。再说选择顺序：能不写 kernel 就不写 → Triton → TileLang 精调关键算子 → CuTe DSL/CUTLASS 极致优化。最后补一句工程实践：和 PyTorch 对拍验证正确性、先用屋顶线判断值不值得优化、自动调优的结果要固化并按机器重搜。

## 练习

**1. 算共享内存。** 上面的 GEMM 用 `block_M = block_N = 128`、`block_K = 64`、`num_stages = 3`、FP16。共享内存要多少字节？H100 每个 SM 最多 228 KB，这个配置每个 SM 能驻留几个 block？

??? success "参考答案"
    每级流水要存 A 的 `128 × 64` 和 B 的 `64 × 128`，各 8192 个 FP16 = 16 KB，两块共 32 KB；三级就是 96 KB（生成的代码里 `B_shared` 的偏移正好是 49152 字节 = 48 KB，即 A 的三级缓冲区大小）。

    228 KB 只能放下 2 个这样的 block（192 KB）。实际上 Hopper 上这类 GEMM 通常每个 SM 就跑 1 个 block、靠 warp 专门化和深流水填满 Tensor Core，而不是靠多个 block 并发——这和"占用率越高越好"的直觉相反。

**2. 改流水级数。** 把 `num_stages` 从 3 改成 2 或 4，共享内存分别是多少？级数是不是越多越好？

??? success "参考答案"
    2 级 64 KB、4 级 128 KB。级数越多，能藏住的搬运延迟越长，但共享内存占用线性增加，可能挤掉并发的 block；而且一旦级数足以覆盖延迟，再加就没有收益。Hopper 上常见 3～4 级（TMA 的延迟长但带宽大），Ampere 上 3～5 级。这正是自动调优要搜的参数之一。

**3. 因果掩码的两层。** FlashAttention 那段代码里，`loop_end = T.ceildiv((bx + 1) * block_M, block_N)` 和块内的 `T.if_then_else` 各处理了什么？只写其中一个会怎样？

??? success "参考答案"
    `loop_end` 在**块的粒度**上跳过完全在对角线右上方的 key 块（那些块里所有元素都被掩码掉，算了纯属浪费）；`if_then_else` 处理**对角线上那一块**内部的三角形部分。

    只写块级跳过：对角线块里本该被掩码的元素没被屏蔽，结果错误。只写块内掩码：结果正确，但要多算大约一半的块——因果注意力的计算量本来只有全注意力的一半，这个优化直接决定性能。

**4. 判断值不值得写。** 一个 kernel 要对 `[B, S, H, D] = [32, 4096, 32, 128]` 的张量做 RMSNorm + 量化到 FP8。用屋顶线判断它是什么瓶颈，该用哪种工具？

??? success "参考答案"
    读入 BF16 张量 32 × 4096 × 32 × 128 × 2 ≈ 1.07 GB，写出 FP8 约 0.54 GB（外加 scale），总访存约 1.6 GB；计算量只有每个元素几次乘加，算术强度远低于屋脊点，**纯带宽瓶颈**。

    这种算子交给 `torch.compile` 或 Triton 就能接近带宽上限，没必要用 TileLang——TileLang 的价值在于能精确控制 Tensor Core 的喂数据方式，而这里根本没用 Tensor Core。真要优化，方向是和上游算子融合（省一次读写），而不是换 kernel 语言。

## 小结

- [x] TileLang 卡在 Triton 和 CUTLASS 之间：块级 Python 写法，但流水线级数、共享内存布局、warp 分工可控。
- [x] `T.Kernel` / `alloc_shared` / `alloc_fragment` / `T.copy` / `T.gemm` / `T.Pipelined` 对应 CUDA 里的 block、共享内存、寄存器片段、TMA 搬运、mma、多级流水。
- [x] 二十行 GEMM 生成带 TMA 描述符、mbarrier 环形缓冲、生产者-消费者 warpgroup 和 wgmma 的 99 行 CUDA；FlashAttention 224 行。
- [x] 调试就看生成的源码：有没有 `tma_load` 和 `wgmma`、流水级数、共享内存用量。
- [x] 选择顺序：不写 kernel → Triton → TileLang 精调关键算子 → CuTe DSL；先用屋顶线判断值不值得。
