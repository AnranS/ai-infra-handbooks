# Triton

<p class="lead">Triton is OpenAI's open-source GPU programming language: written in Python, thought about in blocks, with the compiler handling thread assignment, coalescing, shared memory and Tensor Cores. For the same performance, Triton code is often a fraction of the length of CUDA. PyTorch's torch.compile generates Triton by default, vLLM and SGLang contain plenty of Triton kernels, and it is already a requirement for a kernel development role.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does a Triton "program" correspond to in CUDA?
    2. What is `tl.load`'s `mask` argument for?
    3. Writing a matrix multiply in Triton, who uses the Tensor Cores?
    4. What do `num_warps` and `num_stages` each affect?
    5. When do you still need CUDA rather than Triton?

??? success "Answers (try it yourself first, then expand)"
    1. A program is CUDA's block: `tl.program_id` is the block index, and how the threads inside divide the work is the compiler's business.
    2. Bounds handling: only the positions where the mask is true are loaded, and the rest take the value given by `other` (nothing out of bounds is touched); `tl.store`'s mask works the same way.
    3. The compiler: write `tl.dot` and Triton compiles it into Tensor Core instructions (mma / wgmma) and arranges the shared-memory layout.
    4. `num_warps`: how many warps execute one program (how much work each warp gets, and the register pressure); `num_stages`: the depth of the software pipeline, which decides how far ahead data is loaded (deeper hides more latency but uses more shared memory).
    5. When you need fine control: warp specialization, TMA, register allocation, irregular data structures, or squeezing out the last of the performance (FlashAttention-3 on Hopper, an all-out GEMM); or when you need a hardware feature Triton does not support.

## The programming model: thinking in blocks {#编程模型以块为单位}

![Figure: Triton's programming model, where one program handles one BLOCK and program_id says which](../assets/figures/triton-program-grid.svg){.aig-svg}

CUDA asks you to think about "what each thread does"; Triton asks you to think about "which block of data each **program** handles". A program is roughly a CUDA block, but you operate on whole tensors (vectors, matrices) rather than one thread's scalars:

| | CUDA | Triton |
| --- | --- | --- |
| the basic unit | a thread | a program (handling one block of data) |
| data | scalars, registers | block tensors (`tl.arange`, two-dimensional blocks) |
| shared memory, synchronization | managed by hand | handled by the compiler |
| coalescing, vectorization | by hand | the compiler infers it from pointer contiguity |
| Tensor Cores | WMMA/mma/wgmma | `tl.dot` chooses automatically |
| warp-level control | fully yours | essentially invisible |

The price is coarser control: warp specialization, careful register allocation and unusual data layouts are hard or impossible to express in Triton. But for the great majority of elementwise, reduction and GEMM-shaped fused kernels, Triton comes close to hand-written CUDA.

## Vector addition {#向量加法}

```python title="triton_add.py"
# triton_add.py - the first Triton kernel: vector addition
# run on the CPU in the interpreter: TRITON_INTERPRET=1 python triton_add.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # which program this is, the equivalent of blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # the BLOCK indices this program owns (a vector)
    mask = offsets < n                           # out-of-range positions are neither read nor written
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)               # how many programs to launch
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

A few points:

- `BLOCK` is marked `tl.constexpr`: a compile-time constant, with a separate kernel compiled for each value;
- `tl.arange(0, BLOCK)`'s length must be a power of two;
- a `torch.Tensor` passed to a kernel becomes a pointer to its data automatically;
- for debugging, set `TRITON_INTERPRET=1` and the kernel runs on the CPU interpreted with NumPy, where `print` and breakpoints work. Every example on this page was verified that way.

## A fused softmax {#融合-softmax}

Each program handles one row, reading the whole row in at once, finding the maximum, the sum of exponentials and the normalization "in registers", and writing it back. This is the Triton version of "one warp per row with the row cached in registers" from [the softmax chapter](../kernels/softmax-norm.md):

```python title="triton_softmax.py"
# triton_softmax.py - a fused softmax with one program per row
# run on the CPU in the interpreter: TRITON_INTERPRET=1 python triton_softmax.py
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
    x = x - tl.max(x, axis=0)                    # subtract the maximum for numerical stability
    num = tl.exp(x)                              # a masked position gives exp(-inf) = 0
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)         # a row has to fit in one block
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

`num_warps` sets how many warps execute one program, and longer rows need more parallelism. Once a row is too long for one block, it has to become a loop over tiles with online softmax, as the CUDA version does.

## Matrix multiplication {#矩阵乘法}

Triton's most representative example. Each program computes one `BLOCK_M × BLOCK_N` tile of C, looping along K, loading a tile of A and of B each time and accumulating with `tl.dot`. `tl.dot` compiles to Tensor Core instructions (mma or wgmma), and the shared-memory allocation, the pipeline (`num_stages`) and the layout conversions are all the compiler's doing:

```python title="triton_matmul.py"
# triton_matmul.py - a tiled matrix multiply, with tl.dot using Tensor Cores automatically
# run on the CPU in the interpreter: TRITON_INTERPRET=1 python triton_matmul.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # group the programs: neighbouring ones share A's row tile, which raises the L2 hit rate
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
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # a BLOCK_M x BLOCK_K block of pointers
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # BLOCK_K x BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)                   # FP16/BF16 inputs go through Tensor Cores with FP32 accumulation
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
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32   # the CPU interpreter verifies the logic in FP32
    a = torch.randn(200, 150, device=DEVICE, dtype=dtype)           # a shape deliberately not a multiple of the tile size
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

Compare it with the CUDA code in [the GEMM chapter](../kernels/gemm.md): the tiling, the loop along K and the accumulator are the same idea, but shared memory, `__syncthreads()`, register tiling and the Tensor Core instructions have all disappeared.

## Autotuning {#自动调优}

The best tile size, warp count and pipeline depth depend on the shape and the hardware. `@triton.autotune` runs each given configuration once on the first call and caches the fastest:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],        # retune when these parameters change
)
@triton.jit
def matmul_kernel(...): ...
```

- `num_warps`: warps per program;
- `num_stages`: the depth of the software pipeline over the K loop, which the compiler implements with cp.async or TMA; see [asynchronous copies](../advanced/async-hopper.md).

## Debugging and performance notes {#调试与性能要点}

- **Correctness**: `TRITON_INTERPRET=1` runs on the CPU; `tl.device_print` prints from the GPU; `tl.static_assert` checks at compile time.
- **Looking at the generated code**: the object `kernel[grid](...)` returns has an `asm` dictionary with the TTIR, TTGIR, LLVM IR, PTX and cubin; Nsight Compute analyses it exactly like a CUDA kernel.
- **Masks**: every `tl.load`/`tl.store` that might go out of bounds needs a `mask`, with `other` giving the fill value (`-inf` for a maximum reduction, 0 for a sum).
- **Contiguity hints**: the compiler can infer that indices from `tl.arange` are contiguous and use vectorized accesses. For indices that went through complicated arithmetic, hint with `tl.multiple_of` and `tl.max_contiguous`.
- **TMA on Hopper**: newer Triton supports tensor descriptors (`tl.make_tensor_descriptor` and friends) so the compiler can emit TMA instructions.

## Triton in industry {#triton-在工业界的使用}

- **PyTorch**: Inductor, `torch.compile`'s default backend, generates Triton for the fused elementwise, reduction and some GEMM kernels. Reading what it generates (`TORCH_LOGS=output_code`) is good material for learning Triton;
- **Inference engines**: vLLM's and SGLang's fused MoE, some attention, sampling and quantization kernels are written in Triton;
- **Training-acceleration libraries**: Liger Kernel and Unsloth wrote many fused kernels in Triton (RMSNorm, RoPE, cross entropy); FlagGems implements a whole set of PyTorch operators in it.
- Similar block-level tools include TileLang, the CuTe DSL, and NVIDIA's CUDA Tile programming model introduced in CUDA 13.1 (with cuTile as its Python interface), all along the same lines.

**When CUDA is still needed**: fine warp-level control (warp specialization, unusual shuffle patterns); a hardware feature Triton does not support yet; a cuBLAS-level all-out GEMM; or complex control flow and communication within one kernel. In real work the two coexist: use Triton to implement and validate the gain quickly, then optimize the few most critical kernels deeply in CUDA/CUTLASS.

!!! interview "How to explain it"
    To explain "Triton or CUDA": Triton programs in blocks, where a program is a block and the compiler handles thread mapping, shared memory, coalescing and Tensor Cores (`tl.dot`); `mask` handles the boundaries, and `num_warps`, `num_stages` and the tile size go to autotune. It suits writing fused kernels quickly at close to hand-written performance, and torch.compile generates Triton too; reach for CUDA / CUTLASS when you need fine control over warp specialization, TMA or register allocation (FlashAttention-3 on Hopper, an all-out GEMM). The two coexist in practice.

## Exercises {#练习}

**1. A fused RMSNorm.** Implement `y = x / sqrt(mean(x^2) + eps) * w` in Triton with one program per row, and compare against PyTorch.

??? success "Answer"
    ```python title="triton_rmsnorm.py"
    # triton_rmsnorm.py - RMSNorm with one program per row
    # run on the CPU in the interpreter: TRITON_INTERPRET=1 python triton_rmsnorm.py
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

    The statistics are computed in FP32 (`.to(tl.float32)`) while the input and output can be BF16, which is standard for normalization kernels.

**2. A thinking exercise.** Why does Triton's matrix multiply rearrange the programs into groups (`GROUP_M`)?

??? success "Answer"
    Launched in row-major order, the programs running at the same time cover a whole row of tiles of C: they need the same row tile of A but **every** column tile of B, so B's data is hard to reuse in L2. Grouped, the programs running at once cover a roughly square region of C (8 row tiles by a few column tiles, say), reading less of both A and B and hitting L2 more often. This is the same "the squarer the tile, the better the reuse" from [GEMM](../kernels/gemm.md), applied one level up at L2. CUTLASS calls the equivalent threadblock swizzle or rasterization.

## Summary {#小结}

- [x] Triton programs in blocks, with the compiler handling thread mapping, shared memory, coalescing and Tensor Cores.
- [x] `tl.program_id` plus `tl.arange` builds index blocks and `mask` handles the boundaries; `tl.constexpr` is a compile-time constant.
- [x] `tl.dot` uses Tensor Cores automatically; `num_warps`, `num_stages` and the tile size go to autotune.
- [x] `TRITON_INTERPRET=1` debugs on the CPU; the Triton that torch.compile generates is good study material.
- [x] Triton for implementing quickly, CUDA/CUTLASS for deep optimization; the two coexist in practice.
