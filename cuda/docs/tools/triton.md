# Triton

<p class="lead">Triton 是 OpenAI 开源的 GPU 编程语言：用 Python 写，以"块"为单位思考，编译器负责线程分配、合并访问、共享内存和 Tensor Core。同样性能的算子，Triton 的代码量往往只有 CUDA 的几分之一。PyTorch 的 torch.compile 默认生成 Triton 代码，vLLM、SGLang 里也有大量 Triton kernel，它已经是算子开发岗位的必备技能。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Triton 里的一个 "program" 对应 CUDA 里的什么？
    2. `tl.load` 的 `mask` 参数是做什么的？
    3. 在 Triton 里写矩阵乘法，谁负责使用 Tensor Core？
    4. `num_warps` 和 `num_stages` 分别影响什么？
    5. 什么情况下还需要写 CUDA 而不是 Triton？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 一个 program 相当于 CUDA 的一个 block：`tl.program_id` 是 block 的下标；block 内部的线程怎么分工由编译器决定。
    2. 边界处理：只加载 mask 为真的位置，mask 为假的位置用 `other` 指定的值填充（不会越界访问）；`tl.store` 的 mask 同理。
    3. 编译器：写 `tl.dot` 就行，Triton 会把它编译成 Tensor Core 指令（mma / wgmma），并安排共享内存的布局。
    4. `num_warps`：一个 program 用多少个 warp 执行（每个 warp 分到多少工作、寄存器压力）；`num_stages`：软件流水的级数，决定提前加载多少块数据（越多越能藏延迟，但占用更多共享内存）。
    5. 需要精细控制的时候：warp 专门化、TMA、寄存器分配、非规则的数据结构，或者要榨干最后一点性能（如 Hopper 上的 FlashAttention-3、极致的 GEMM）；或者要用到 Triton 不支持的硬件特性。

## 编程模型：以块为单位

![图：Triton 的编程模型——一个 program 处理一个 BLOCK，program_id 决定负责哪一块](../assets/figures/triton-program-grid.svg){.aig-svg}

CUDA 要求你思考"每个线程做什么"，Triton 让你思考"每个**程序实例（program）**处理哪一块数据"。一个 program 大致相当于 CUDA 的一个 block，但你操作的是整块的张量（向量、矩阵），而不是单个线程的标量：

| | CUDA | Triton |
| --- | --- | --- |
| 基本单位 | 线程 | 程序实例（处理一个数据块） |
| 数据 | 标量、寄存器 | 块状张量（`tl.arange`、二维块） |
| 共享内存、同步 | 手动管理 | 编译器自动处理 |
| 合并访问、向量化 | 手动 | 编译器根据指针的连续性自动处理 |
| Tensor Core | WMMA/mma/wgmma | `tl.dot` 自动选择 |
| warp 级控制 | 完全可控 | 基本不可见 |

代价是控制粒度变粗：warp 专门化、精细的寄存器分配、特殊的数据布局等，Triton 很难或者无法表达。但对于绝大多数逐元素、归约、GEMM 类的融合算子，Triton 可以达到接近手写 CUDA 的性能。

## 向量加法

```python title="triton_add.py"
# triton_add.py —— Triton 的第一个 kernel：向量加法
# 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_add.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # 第几个程序实例，相当于 blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 这个实例负责的 BLOCK 个下标（一个向量）
    mask = offsets < n                           # 越界的位置不读不写
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)               # 启动多少个程序实例
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(98_765, device=DEVICE)
    y = torch.randn(98_765, device=DEVICE)
    out = add(x, y)
    assert torch.allclose(out, x + y), "mismatch"
    print("PASS triton add")
```

几个要点：

- `BLOCK` 被标记为 `tl.constexpr`：编译期常量，每个不同的取值会编译出一个不同的 kernel；
- `tl.arange(0, BLOCK)` 的长度必须是 2 的幂；
- 传给 kernel 的 `torch.Tensor` 会自动变成指向其数据的指针；
- 调试时设置环境变量 `TRITON_INTERPRET=1`，kernel 会在 CPU 上用 NumPy 解释执行，可以直接用 `print` 和断点。本页所有示例都用这种方式验证过。

## 融合 softmax

每个程序实例处理一行，整行一次读进来，在"寄存器"里完成求最大值、求指数和、归一化，最后写回。这就是[Softmax 一章](../kernels/softmax-norm.md)里"一个 warp 一行、整行缓存在寄存器里"的 Triton 版本：

```python title="triton_softmax.py"
# triton_softmax.py —— 每个程序实例处理一行的融合 softmax
# 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_softmax.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                    # 减去最大值，数值稳定
    num = tl.exp(x)                              # 被掩码的位置 exp(-inf) = 0
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)         # 一行必须能放进一个块
    num_warps = 4 if BLOCK <= 2048 else 8 if BLOCK <= 8192 else 16
    softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols, BLOCK=BLOCK, num_warps=num_warps)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(123, 781, device=DEVICE) * 10
    out = softmax(x)
    torch.testing.assert_close(out, torch.softmax(x, dim=1), rtol=1e-5, atol=1e-6)
    print("PASS triton softmax")
    if DEVICE == "cuda":
        big = torch.randn(4096, 4096, device=DEVICE)
        ms = triton.testing.do_bench(lambda: softmax(big))
        print(f"softmax 4096x4096: {ms:.3f} ms, {2 * big.numel() * 4 / ms / 1e6:.1f} GB/s")
```

`num_warps` 决定每个程序实例用多少个 warp 执行，行越长需要的并行度越高。行长到放不进一个块时，就要像 CUDA 版本那样改成循环分块 + online softmax。

## 矩阵乘法

Triton 最有代表性的例子。每个程序实例计算 C 的一个 `BLOCK_M × BLOCK_N` 块，沿 K 方向循环，每次加载 A、B 的一块并用 `tl.dot` 累加。`tl.dot` 会被编译成 Tensor Core 指令（mma 或 wgmma），共享内存的分配、流水（`num_stages`）、布局转换都由编译器完成：

```python title="triton_matmul.py"
# triton_matmul.py —— 分块矩阵乘法，tl.dot 自动使用 Tensor Core
# 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_matmul.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # 分组排列程序实例：相邻的实例共享 A 的行块，提高 L2 命中率
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # BLOCK_M x BLOCK_K 的指针块
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # BLOCK_K x BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)                   # FP16/BF16 输入时走 Tensor Core，FP32 累加
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32   # CPU 解释器下用 FP32 验证逻辑
    a = torch.randn(200, 150, device=DEVICE, dtype=dtype)           # 故意取非块大小整数倍的形状
    b = torch.randn(150, 300, device=DEVICE, dtype=dtype)
    c = matmul(a, b)
    tol = 1e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(c, a @ b, rtol=tol, atol=tol)
    print("PASS triton matmul")
    if DEVICE == "cuda":
        a = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: matmul(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32))
        ref = triton.testing.do_bench(lambda: a @ b)
        flops = 2 * 4096 ** 3
        print(f"triton {flops / ms / 1e9:.1f} TFLOPS vs torch (cuBLAS) {flops / ref / 1e9:.1f} TFLOPS")
```

把它和 [GEMM](../kernels/gemm.md) 一章的 CUDA 代码对比：分块、沿 K 循环、累加器都是同样的思路，但共享内存、`__syncthreads()`、寄存器分块、Tensor Core 指令全部消失了。

## 自动调优

最佳的块大小、warp 数、流水级数因形状和硬件而异。`@triton.autotune` 会在第一次调用时把给定的几组配置都跑一遍，缓存最快的那一组：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],        # 这些参数变化时重新调优
)
@triton.jit
def matmul_kernel(...): ...
```

- `num_warps`：每个程序实例的 warp 数；
- `num_stages`：K 循环的软件流水级数，编译器会用 cp.async 或 TMA 实现多级流水，见[异步拷贝](../advanced/async-hopper.md)。

## 调试与性能要点

- **正确性**：`TRITON_INTERPRET=1` 在 CPU 上运行；`tl.device_print` 在 GPU 上打印；`tl.static_assert` 做编译期检查。
- **查看生成的代码**：`kernel[grid](...)` 返回的对象里有 `asm` 字典，可以查看 TTIR、TTGIR、LLVM IR、PTX、cubin；也可以用 Nsight Compute 分析，和 CUDA kernel 没有区别。
- **掩码**：所有可能越界的 `tl.load`/`tl.store` 都要带 `mask`；`other` 指定越界位置的填充值（归约取最大值时填 `-inf`，求和时填 0）。
- **连续性提示**：编译器能推断出 `tl.arange` 生成的下标是连续的，从而使用向量化访存。经过复杂计算的下标，可以用 `tl.multiple_of`、`tl.max_contiguous` 提示。
- **Hopper 上的 TMA**：较新的 Triton 支持张量描述符（`tl.make_tensor_descriptor` 等），让编译器生成 TMA 指令。

## Triton 在工业界的使用

- **PyTorch**：`torch.compile` 的默认后端 Inductor 会把融合后的逐元素、归约、部分 GEMM 生成 Triton 代码。阅读它生成的代码（`TORCH_LOGS=output_code`）是学习 Triton 的好材料；
- **推理引擎**：vLLM、SGLang 的 fused MoE、部分注意力、采样、量化 kernel 用 Triton 实现；
- **训练加速库**：Liger Kernel、Unsloth 用 Triton 写了大量融合算子（RMSNorm、RoPE、交叉熵等）；国内的 FlagGems 用 Triton 实现了一整套 PyTorch 算子。
- 同类的块级编程工具还有 TileLang、CuTe DSL，以及 NVIDIA 从 CUDA 13.1 开始推出的 CUDA Tile 编程模型（Python 接口为 cuTile）等，思路相近。

**什么时候仍然需要 CUDA**：需要 warp 级别的精细控制（warp 专门化、特殊的 shuffle 模式）；需要 Triton 还不支持的硬件特性；追求 cuBLAS 级别的极致 GEMM；或者需要在同一个 kernel 里做复杂的控制流和通信。实际工作中两者经常并存：先用 Triton 快速实现并验证收益，再对最关键的少数 kernel 用 CUDA/CUTLASS 深度优化。

!!! interview "怎么讲清楚"
    讲"Triton 和 CUDA 怎么选"：Triton 以块为单位编程，一个 program 相当于一个 block，编译器负责线程映射、共享内存、合并访问和 Tensor Core（`tl.dot`）；`mask` 处理边界，`num_warps`、`num_stages` 和块大小交给 autotune。它适合快速写出接近手写性能的融合算子，torch.compile 生成的也是 Triton；需要精细控制 warp 专门化、TMA、寄存器分配（比如 Hopper 上的 FlashAttention-3、极致的 GEMM）时再用 CUDA / CUTLASS——两者在工作中经常并存。

## 练习

**1. 融合的 RMSNorm。** 用 Triton 实现 `y = x / sqrt(mean(x^2) + eps) * w`，每个程序实例处理一行，与 PyTorch 的实现对比结果。

??? success "参考答案"
    ```python title="triton_rmsnorm.py"
    # triton_rmsnorm.py —— 每个程序实例处理一行的 RMSNorm
    # 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_rmsnorm.py
    import torch
    import triton
    import triton.language as tl

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    @triton.jit
    def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
        y = x * rstd * w
        tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


    def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        rows, cols = x.shape
        y = torch.empty_like(x)
        rmsnorm_kernel[(rows,)](x, w, y, x.stride(0), cols, eps, BLOCK=triton.next_power_of_2(cols))
        return y


    if __name__ == "__main__":
        torch.manual_seed(0)
        x = torch.randn(64, 1000, device=DEVICE)
        w = torch.rand(1000, device=DEVICE) + 0.5
        ref = x / torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-6) * w
        torch.testing.assert_close(rmsnorm(x, w), ref, rtol=1e-5, atol=1e-5)
        print("PASS triton rmsnorm")
    ```

    统计量用 FP32 计算（`.to(tl.float32)`），输入输出可以是 BF16，这是归一化类算子的标准做法。

**2. 思考题。** Triton 的矩阵乘法里为什么要对程序实例做"分组排列"（`GROUP_M`）？

??? success "参考答案"
    按行优先的顺序启动程序实例时，同时在运行的实例会覆盖 C 的一整行块，它们需要 A 的同一个行块，但需要 B 的**所有**列块，B 的数据很难在 L2 中被复用。分组后，同时运行的实例覆盖 C 的一个近似正方形的区域（比如 8 行块 × 若干列块），读取的 A、B 数据量都更小，L2 命中率更高。这和[GEMM](../kernels/gemm.md) 中"分块越接近正方形，数据复用越好"是同一个道理，只是作用在 L2 这一级。CUTLASS 里对应的概念叫 threadblock swizzle / rasterization。

## 小结

- [x] Triton 以块为单位编程，编译器处理线程映射、共享内存、合并访问和 Tensor Core。
- [x] `tl.program_id` + `tl.arange` 生成下标块，`mask` 处理边界；`tl.constexpr` 是编译期常量。
- [x] `tl.dot` 自动使用 Tensor Core；`num_warps`、`num_stages` 和块大小交给 autotune。
- [x] `TRITON_INTERPRET=1` 在 CPU 上调试；torch.compile 生成的 Triton 代码是很好的学习材料。
- [x] Triton 快速实现、CUDA/CUTLASS 深度优化，两者在工作中经常并存。
