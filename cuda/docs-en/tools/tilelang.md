# TileLang: a tile language one level below Triton

<p class="lead">A genuinely fast kernel on Hopper and Blackwell looks like this: some warps move data with the TMA, some issue wgmma, and an mbarrier hands off between them as producer and consumer. Triton hides all of that inside the compiler, and hand-written CUDA is too much work. TileLang sits exactly in between: the block-level copies and matrix multiplies are written in Python, while the pipeline depth, the shared-memory layout and the warp division of labour stay under your control. This chapter writes two kernels that compile (a GEMM and FlashAttention), looks at what appears in the CUDA source it generates, and sets it against Triton and the CuTe DSL to say what each suits.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does the division of labour differ between TileLang and Triton?
    2. What does `T.Pipelined(..., num_stages=3)` do? Why is it needed?
    3. What do `T.alloc_shared` and `T.alloc_fragment` correspond to in hardware?
    4. What sort of CUDA code does a twenty-line TileLang GEMM generate on Hopper?
    5. When should you use TileLang, and when Triton or the CuTe DSL?

??? success "Answers (try it yourself first, then expand)"
    1. Triton lets you decide "which tile of data each program instance handles" and leaves the thread mapping **within** a tile, shared memory, the pipeline and the Tensor Core instructions entirely to the compiler. TileLang lets you keep control further down: allocating shared memory and register fragments explicitly, specifying the pipeline depth, and where needed the layout of each tile and the warp division of labour, while the thread mapping inside a loop is still filled in by the compiler.
    2. It overlaps "move the next tile" with "compute this one": the compiler splits the loop body into a producer (TMA / cp.async) and a consumer (wgmma), rotating through `num_stages` shared-memory buffers and mbarriers. Without it, every round waits for the data to arrive before computing and the Tensor Cores spend most of their time waiting.
    3. `alloc_shared` is a tile in shared memory (corresponding to `__shared__`) and `alloc_fragment` is a tile spread across a warp's registers (corresponding to an mma's accumulator fragment). The dataflow is usually global → shared (TMA) → register fragments (an mma's operands and accumulator) → global.
    4. Measured in this chapter: 99 lines of CUDA, containing `CUtensorMap` descriptors, `tl::tma_load`, 9 `mbarrier`s, `wgmma_ss` and `warpgroup_reg_dealloc/alloc`, that is a complete warp-specialized pipeline, which by hand takes several hundred lines.
    5. Ordinary elementwise operations, reductions and simple GEMMs go to `torch.compile` or Triton; use TileLang when you need precise control of Hopper/Blackwell features (TMA, wgmma, tcgen05, warp specialization) without writing hundreds of lines of CUTLASS templates; use the CuTe DSL when you need the last word in layout algebra or have to fit into the CUTLASS ecosystem.

## Why TileLang exists {#为什么会有-tilelang}

[The AI compiler landscape](../framework/compilers.md#新一代的-tile-语言) mentioned this dividing line: Triton's abstraction fits Ampere well, but on Hopper the few things that decide performance all happen to sit "inside a tile":

- **the TMA**: one instruction moves a whole multidimensional tile, needing a host-side tensor descriptor and an mbarrier count;
- **wgmma**: an asynchronous matrix multiply issued by four warps together as a warpgroup, whose operands can come straight from shared memory;
- **warp specialization**: some warps only move data and some only compute, with an mbarrier handing a ring buffer between them;
- **register reallocation**: the producer warps do not need that many registers and can give them to the consumers (`setmaxnreg`).

All of this can be written into the Triton compiler, but you cannot direct it; and hand-written CUDA or CUTLASS templates cost a great deal. TileLang's trade-off (built on top of TVM) is: **keep the block-level Python style while exposing the pipeline, the layouts and the warp division of labour as parameters you can tune**.

## The first kernel: GEMM {#第一个-kernelgemm}

![Figure: the tile-level programming model - one program handles one BLOCK](../assets/figures/triton-program-grid.svg){.aig-svg}

```bash
pip install tilelang        # this chapter uses 0.1.15; running it needs an NVIDIA GPU, generating the code alone needs the CUDA toolchain
```

```python title="tl_matmul.py" run="no"
# a TileLang GEMM: the tile-level copies and matrix multiply written straight in Python, with the compiler handling the thread mapping and the pipeline
# install: pip install tilelang; run: needs an NVIDIA GPU (compiling to CUDA source needs only the CUDA toolchain)
from pathlib import Path

import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M=128, block_N=128, block_K=64, num_stages=3, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        # each block handles one block_M x block_N tile of the output
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")   # the two input tiles in shared memory
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")    # the accumulator lives in registers
            T.clear(C_local)
            # Pipelined: the compiler overlaps "move the next tile" with "compute this one", num_stages being the pipeline depth
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)        # global -> shared (TMA / cp.async chosen automatically)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)                    # expanded into mma / wgmma automatically
            T.copy(C_local, C[by * block_M, bx * block_N])             # the accumulator -> global
    return kernel


target = {"kind": "cuda", "arch": "sm_90a"}       # with no GPU, name the architecture explicitly and only generate code
jit_kernel = tilelang.compile(matmul(1024, 1024, 1024), out_idx=[2], target=target)
src = jit_kernel.get_kernel_source()
Path("gen.cu").write_text(src)
print(len(src.splitlines()), "行 CUDA 源码")
```

The order in which to read that code is the same as for CUDA, only every step is at the granularity of a tile:

| This line | What it is in CUDA |
| --- | --- |
| `T.Kernel(gx, gy, threads=128)` | `kernel<<<dim3(gx, gy), 128>>>`, with `bx` and `by` as `blockIdx` |
| `T.alloc_shared((block_M, block_K), "float16")` | `__shared__ half A_shared[block_M][block_K]` |
| `T.alloc_fragment((block_M, block_N), "float")` | the register fragment holding an mma accumulator |
| `T.copy(A[by * block_M, ko * block_K], A_shared)` | a whole-tile global → shared copy (the compiler picks TMA or `cp.async`) |
| `T.gemm(A_shared, B_shared, C_local)` | expanded into a sequence of `mma` / `wgmma` |
| `T.Pipelined(n, num_stages=3)` | a multi-stage pipeline: several shared-memory buffers, with copies overlapping computation |

Note that **`threadIdx` never appears**: which elements each thread in the block handles is derived by the compiler from the layout. That is what "tile language" means: you manage the tile and the compiler manages what is inside it.

## What it generated {#它生成了什么}

The Python above, compiled for `sm_90a` (Hopper), is 99 lines of CUDA. A few key passages:

```cuda title="the generated CUDA (excerpt)" run="no"
extern "C" __global__ void __launch_bounds__(256, 1)
kernel_kernel(__grid_constant__ const CUtensorMap A_desc,      // the TMA's tensor descriptors
              __grid_constant__ const CUtensorMap B_desc,
              half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];     // the shared-memory buffers of a 3-stage pipeline
  __shared__ __align__(16) uint64_t mbarrier_mem[6];           // 3 "data has arrived" + 3 "the buffer is empty"

  if (((int)threadIdx.x) >= 128) {                             // the producer warpgroup: only moves data
    tl::warpgroup_reg_dealloc<24>();                           //   give the registers to the consumers
    for (int ko = 0; ko < 16; ++ko) {
      mbarrier[((ko % 3) + 3)].wait(...);                      //   wait for this buffer to be consumed
      mbarrier[(ko % 3)].expect_transaction(16384);
      tl::tma_load(A_desc, mbarrier[(ko % 3)], ..., (ko * 64), (((int)blockIdx.y) * 128));
    }
  } else {                                                     // the consumer warpgroup: only computes
    tl::warpgroup_reg_alloc<240>();
    for (int ko_1 = 0; ko_1 < 16; ++ko_1) {
      mbarrier[(ko_1 % 3)].wait(...);                          //   wait for the data to arrive
      tl::wgmma_ss<tl::DataType::kFloat16, ..., 64, 128, 16, ...>(desc_a, desc_b, C_local, ...);
      mbarrier[((ko_1 % 3) + 3)].arrive();                     //   tell the producer this tile may be overwritten
    }
  }
}
```

Twenty lines of Python became a complete Hopper pipeline: TMA descriptors, a ring buffer of 6 mbarriers, producer-consumer warp specialization, register reallocation and wgmma. By hand this takes several hundred lines, and every synchronization point is a chance to get it wrong.

The usual debugging move is to print that source and read it: **is `tma_load` and `wgmma` being used** (if not, the layout or the shape failed a condition and it fell back to `cp.async` + `mma`), **is the pipeline depth right**, **is shared memory over the limit**. TileLang also offers `tilelang.disassemble` for the PTX/SASS.

## FlashAttention: online softmax as tile code {#flashattention在线-softmax-写成-tile-代码}

```python title="tl_flash.py" run="no"
# FlashAttention's forward pass in TileLang (causal mask), showing how online softmax is written as tile-level code
from pathlib import Path

import tilelang
import tilelang.language as T


def flash_attn(batch, heads, seq, dim, block_M=64, block_N=64, num_stages=2, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504          # multiply by log2(e), so exp2 can be used later, which is faster
    shape = (batch, seq, heads, dim)

    @T.prim_func
    def kernel(Q: T.Tensor(shape, "float16"), K: T.Tensor(shape, "float16"),
               V: T.Tensor(shape, "float16"), Out: T.Tensor(shape, "float16")):
        with T.Kernel(T.ceildiv(seq, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), "float16")
            K_shared = T.alloc_shared((block_N, dim), "float16")
            V_shared = T.alloc_shared((block_N, dim), "float16")
            scores = T.alloc_fragment((block_M, block_N), "float")     # one tile of QK^T
            probs = T.alloc_fragment((block_M, block_N), "float16")
            acc = T.alloc_fragment((block_M, dim), "float")            # the output accumulator
            row_max = T.alloc_fragment((block_M,), "float")            # online softmax's two statistics
            row_sum = T.alloc_fragment((block_M,), "float")
            prev_max = T.alloc_fragment((block_M,), "float")
            rescale = T.alloc_fragment((block_M,), "float")

            T.copy(Q[bz, bx * block_M, by, 0], Q_shared)
            T.fill(acc, 0)
            T.fill(row_sum, 0)
            T.fill(row_max, -T.infinity(scores.dtype))
            loop_end = T.ceildiv((bx + 1) * block_M, block_N)          # the causal mask: only the tiles before this one
            for ko in T.Pipelined(loop_end, num_stages=num_stages):
                T.copy(K[bz, ko * block_N, by, 0], K_shared)
                T.clear(scores)
                T.gemm(Q_shared, K_shared, scores, transpose_B=True)   # QK^T
                for i, j in T.Parallel(block_M, block_N):              # the causal mask within the tile
                    scores[i, j] = T.if_then_else(bx * block_M + i >= ko * block_N + j,
                                                  scores[i, j] * scale, -T.infinity(scores.dtype))
                T.copy(row_max, prev_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)      # update the row maximum
                for i in T.Parallel(block_M):
                    rescale[i] = T.exp2(prev_max[i] - row_max[i])      # the old accumulated value is rescaled to the new maximum
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
                acc[i, j] /= row_sum[i]                                # divide by the normalizer once at the end
            T.copy(acc, Out[bz, bx * block_M, by, 0])
    return kernel


target = {"kind": "cuda", "arch": "sm_90a"}
jit_kernel = tilelang.compile(flash_attn(1, 8, 1024, 64), out_idx=[3], target=target)
src = jit_kernel.get_kernel_source()
Path("flash.cu").write_text(src)
print(len(src.splitlines()), "行 CUDA 源码")
```

This code's structure is FlashAttention's pseudocode itself (see the CUDA handbook's [FlashAttention and inference operators](../advanced/attention.md)):

- each block handles `block_M` queries and walks tile by tile along the keys;
- `row_max` and `row_sum` are online softmax's two statistics, and after each tile the accumulated result is **rescaled** to the new maximum (`rescale`);
- `exp2` replaces `exp` (with `log2(e)` folded into the scale beforehand), because `exp2` is a single hardware instruction on a GPU;
- the causal mask has two levels: `loop_end` skips the tiles wholly above the diagonal and `T.if_then_else` handles the tile on the diagonal.

Compiled for Hopper it is 224 lines of CUDA, again with TMA, mbarriers and wgmma. For comparison, FlashAttention-3's hand-written CUTLASS implementation is several thousand lines.

TileLang does nothing for you at the algorithm level: online softmax's rescaling order, the mask's handling and when to divide by the normalizer all have to be written correctly yourself. What it saves is the drudgery of **mapping the algorithm onto the hardware**.

## Against Triton and the CuTe DSL {#和-tritoncute-dsl-的对照}

| | Triton | TileLang | CuTe DSL / CUTLASS |
| --- | --- | --- | --- |
| Level of abstraction | tile level, fully automatic within a tile | tile level + a controllable pipeline, layouts and warp division of labour | layout algebra, down to which elements each thread holds |
| Lines of code (GEMM) | shortest | slightly longer | long, but the most flexible |
| Hopper/Blackwell features | decided by the compiler, little control (Gluon is filling this in) | explicitly controllable | fully controllable |
| Auto-tuning | `triton.autotune` | `tilelang.autotune` (searching the tile size, the pipeline depth, the thread count) | template parameters + a profiler to pick the tactic |
| Ecosystem | the most mature, the back end of PyTorch Inductor | new, with a growing operator library | NVIDIA's own, the basis of FlashAttention-3 and DeepGEMM |
| Suits | the great majority of custom operators | the critical operators that need Hopper/Blackwell features | the last word in performance, fitting into CUTLASS |

A realistic order to choose in:

1. first see whether a kernel can be avoided: `torch.compile`'s fusion, an existing library (FlashInfer, FlashMLA, DeepGEMM);
2. if one is needed, write Triton first and stop if it is fast enough;
3. where Triton clearly falls short (Nsight shows low Tensor Core utilization, or no TMA/wgmma), tune that one operator in TileLang;
4. only for something still more extreme, or to contribute back to the CUTLASS ecosystem, reach for the CuTe DSL.

Model teams open-sourcing operators often ship both a Triton and a TileLang version (the sparse attention operators DeepSeek released in V3.2, say), because the former reads more easily and the latter is faster on Hopper.

## A few engineering points {#工程上的几个注意点}

- **versions and hardware are coupled**: TileLang is still iterating quickly and its API changes; the generated code only uses wgmma / tcgen05 for targets with the `a` suffix, `sm_90a` / `sm_100a`, so the architecture has to be specified at compile time.
- **correctness is yours to verify**: compare elementwise against a PyTorch reference implementation (noting that a different accumulation order gives floating-point differences, so use `rtol`), then test the boundary shapes (M/N/K that do not divide evenly, a sequence length that is not a multiple of the tile size).
- **measure the bandwidth and compute ceilings first**: work out this operator's theoretical floor with the roofline before writing it and compare the measurement afterwards; do not spend effort tuning a pipeline for an operator that was bandwidth-bound all along.
- **auto-tuning needs a fixed environment**: the best configuration found depends on the GPU model, the CUDA version and even the power limit, so a different machine needs a new search; in production, freeze the results into a configuration table.

!!! interview "How to explain it"
    To explain "which kernel languages do you know besides Triton": start with the dividing line. Triton is enough on Ampere, but on Hopper/Blackwell the things that decide performance, the TMA, wgmma, warp specialization and register reallocation, all sit inside a tile, and Triton gives you no control over them. TileLang keeps the block-level Python style while exposing the pipeline depth, the shared-memory layout and the warp division of labour; a twenty-line GEMM generates CUDA with TMA descriptors, an mbarrier ring buffer, producer-consumer warpgroups and wgmma, which by hand takes several hundred lines. Then give the order to choose in: avoid a kernel if you can → Triton → TileLang for the critical operators → the CuTe DSL/CUTLASS for the last word. Finish with the engineering practice: verify correctness against PyTorch, use the roofline first to judge whether optimizing is worth it, and freeze the auto-tuning results and re-search them per machine.

## Exercises {#练习}

**1. Compute the shared memory.** The GEMM above uses `block_M = block_N = 128`, `block_K = 64`, `num_stages = 3` and FP16. How many bytes of shared memory does it need? With at most 228 KB per SM on an H100, how many blocks of this configuration can an SM hold?

??? success "Answer"
    Each pipeline stage stores A's `128 × 64` and B's `64 × 128`, 8192 FP16 = 16 KB each, for 32 KB in both; three stages make 96 KB (`B_shared`'s offset in the generated code is exactly 49152 bytes = 48 KB, which is the size of A's three buffers).

    228 KB fits only 2 such blocks (192 KB). In practice this kind of GEMM on Hopper usually runs one block per SM and keeps the Tensor Cores fed by warp specialization and a deep pipeline rather than by several concurrent blocks, which runs against the intuition that "higher occupancy is better".

**2. Change the pipeline depth.** With `num_stages` at 2 or 4, how much shared memory is used? Are more stages always better?

??? success "Answer"
    2 stages is 64 KB and 4 is 128 KB. More stages hide a longer copy latency, but the shared-memory use grows linearly and may squeeze out concurrent blocks; and once the depth covers the latency, more brings nothing. 3 to 4 stages is common on Hopper (the TMA's latency is long but its bandwidth is high) and 3 to 5 on Ampere. This is exactly one of the parameters auto-tuning searches.

**3. The causal mask's two levels.** In the FlashAttention code, what does `loop_end = T.ceildiv((bx + 1) * block_M, block_N)` handle and what does the `T.if_then_else` inside the tile handle? What happens with only one of them?

??? success "Answer"
    `loop_end` skips, **at tile granularity**, the key tiles lying wholly above the diagonal (every element in them is masked out, so computing them is pure waste); the `if_then_else` handles the triangular part inside **the tile on the diagonal**.

    With only the tile-level skip: the elements in the diagonal tile that should be masked are not, and the result is wrong. With only the in-tile mask: the result is right, but about twice as many tiles are computed; causal attention's work is only half of full attention's to begin with, so this optimization decides performance outright.

**4. Judge whether it is worth writing.** A kernel has to do RMSNorm plus quantization to FP8 on a tensor of `[B, S, H, D] = [32, 4096, 32, 128]`. Use the roofline to say what bounds it and which tool to use.

??? success "Answer"
    Reading the BF16 tensor is 32 × 4096 × 32 × 128 × 2 ≈ 1.07 GB and writing FP8 about 0.54 GB (plus the scales), for about 1.6 GB of traffic; the computation is a few multiply-adds per element, an arithmetic intensity far below the ridge point, so it is **purely bandwidth-bound**.

    This kind of operator reaches close to the bandwidth ceiling with `torch.compile` or Triton, and TileLang is unnecessary: TileLang's value is precise control of how the Tensor Cores are fed, and there are no Tensor Cores here at all. If it really needs optimizing, the direction is fusing with the upstream operator (saving one read and write), not changing kernel language.

## Summary {#小结}

- [x] TileLang sits between Triton and CUTLASS: the block-level Python style, with the pipeline depth, the shared-memory layout and the warp division of labour under control.
- [x] `T.Kernel` / `alloc_shared` / `alloc_fragment` / `T.copy` / `T.gemm` / `T.Pipelined` correspond to CUDA's block, shared memory, register fragments, a TMA copy, mma and a multi-stage pipeline.
- [x] A twenty-line GEMM generates 99 lines of CUDA with TMA descriptors, an mbarrier ring buffer, producer-consumer warpgroups and wgmma; FlashAttention is 224 lines.
- [x] To debug, read the generated source: whether `tma_load` and `wgmma` appear, the pipeline depth, the shared memory used.
- [x] The order to choose in: no kernel → Triton → TileLang for the critical operators → the CuTe DSL; use the roofline first to judge whether it is worth it.
