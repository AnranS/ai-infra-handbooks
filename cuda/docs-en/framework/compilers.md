# The AI compiler landscape

<p class="lead">Model architectures change every month and hardware changes generation every year or two, so writing a kernel by hand for every "operator × shape × hardware" cannot keep up. What an AI compiler sets out to do is turn a computation graph automatically into code that runs fast on the target hardware. This chapter gives a map: how many levels a compiler has, what each optimizes, where the mainstream compilers sit, and when an inference engineer leans on the compiler versus writes a kernel by hand.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What do graph-level and operator-level (loop-level) optimization each do? Give two examples of each.
    2. Why does "operator fusion" matter so much for inference? Which operators fuse easily and which do not?
    3. What does Halide's "separation of algorithm and schedule" mean? Which scheduling decisions does Triton hand to the compiler?
    4. Which levels of representation does a Triton kernel pass through from Python to GPU machine code?
    5. What problem with Triton do the newer kernel languages like TileLang and the CuTe DSL set out to solve?

??? success "Answers (try it yourself first, then expand)"
    1. Graph level: transformations on the computation graph, such as operator fusion, constant folding, layout transformation and common subexpression elimination; operator level: optimizing a single operator's loop implementation, such as tiling, loop reordering, vectorization and software pipelining.
    2. Inference (decode especially) is mostly bandwidth-bound, and fusion keeps the intermediate results on chip, reads and writes device memory less, and saves kernel launches besides. Elementwise operations fuse easily with each other, as does the pre- and post-processing around a row-wise reduction; it gets hard when the reduction directions disagree (row-wise then column-wise), when the register pressure after fusing is too high, or when an intermediate result has several consumers.
    3. The algorithm (what to compute) and the schedule (how: how the loops are split, ordered and which level of storage they use) are described separately, and changing the schedule does not affect correctness. Triton lets the programmer decide how the tiles are cut (the upper part of the schedule) and leaves the thread mapping within a tile, shared memory, coalescing and the Tensor Cores to the compiler.
    4. Python (`@triton.jit`) → Triton IR (ttir) → Triton GPU IR (ttgir, with layout information added) → LLVM IR → PTX → cubin (machine code).
    5. Triton hides the details inside a tile, which on Hopper / Blackwell makes it very hard to control warp specialization, TMA and register allocation, the things that decide performance; TileLang and the CuTe DSL expose a lower, tile-level abstraction, letting one write a kernel close to hand-written CUDA while staying more concise than CUDA.

## A compiler's three levels {#编译器的三层}

![Figure: a compiler's successive lowerings - graph-level IR, loop-level IR, target IR, machine code](../assets/figures/ir-lowering.svg){.aig-svg}

| Level | Input | Typical optimizations | Representatives |
| --- | --- | --- | --- |
| graph | a computation graph (FX, XLA HLO, ONNX, MLIR's high-level dialects) | constant folding, common subexpression elimination, dead code elimination, **operator fusion**, layout transformation, quantization and precision conversion | Dynamo + Inductor's graph passes, XLA, TVM Relax, TensorRT's graph optimization |
| operator (loop) | one (fused) operator's loop nest | tiling, vectorization, loop interchange and unrolling, using shared memory and the Tensor Cores, pipelining | Inductor's loop IR, TVM TensorIR, Halide, MLIR linalg / affine, Triton |
| code generation | a low-level IR | register allocation, instruction selection and scheduling | LLVM → PTX → SASS (`ptxas`) |

An inference engineer meets the first two levels most: the graph level decides "which operations become one kernel" and the operator level decides "how one kernel tiles and moves data inside".

## The graph level: what fusion saves is memory traffic {#图级融合省下的是访存}

Inference (decode especially) is mostly bandwidth-bound, so the graph level's most important optimization is **fusion**: keep the intermediate results in registers or shared memory rather than writing them back to device memory and reading them out again. Below is the most simplified "fusion analysis" written on an FX graph, measuring the memory traffic of RMSNorm + SiLU + multiply (half of SwiGLU) before and after fusing:

```python title="fx_fusion.py"
import operator

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

# elementwise operations and reductions: every operator in this example is one of the two
POINTWISE_FN = {operator.add, operator.mul, operator.truediv, torch.rsqrt, torch.nn.functional.silu}
POINTWISE_METHOD = {"pow"}
REDUCE_METHOD = {"mean", "sum"}


def rmsnorm_silu_mul(x, w, up):
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    return torch.nn.functional.silu(h) * up


def nbytes(node):
    meta = node.meta.get("tensor_meta")
    return meta.shape.numel() * meta.dtype.itemsize if meta is not None else 0


def fusible(node):
    if node.op == "call_function":
        return node.target in POINTWISE_FN
    if node.op == "call_method":
        return node.target in POINTWISE_METHOD | REDUCE_METHOD
    return False


gm = fx.symbolic_trace(rmsnorm_silu_mul)
x, w, up = torch.randn(8, 4096), torch.randn(4096), torch.randn(8, 4096)
ShapeProp(gm).propagate(x, w, up)   # infer every node's shape and type on the graph

eager = 0
ops = [n for n in gm.graph.nodes if fusible(n)]
for n in ops:   # eager: every operator reads all of its inputs and writes its output
    read = sum(nbytes(a) for a in n.all_input_nodes)
    eager += read + nbytes(n)
    name = n.target if isinstance(n.target, str) else n.target.__name__
    print(f"{name:<8} 读 {read:>7} 字节，写 {nbytes(n):>7} 字节")

# fused: the whole chain reads only the graph's inputs and writes only the final output, the intermediates staying in registers
group = set(ops)
inputs = {a for n in ops for a in n.all_input_nodes if a not in group}
outputs = [n for n in ops if any(u not in group for u in n.users)]
fused = sum(nbytes(a) for a in inputs) + sum(nbytes(o) for o in outputs)
print(f"{len(ops)} 个算子逐个执行：共 {eager} 字节；融合成 1 个 kernel：{fused} 字节，是原来的 {fused / eager:.1%}")
```

```text title="output"
pow      读  131072 字节，写  131072 字节
mean     读  131072 字节，写      32 字节
add      读      32 字节，写      32 字节
rsqrt    读      32 字节，写      32 字节
mul      读  131104 字节，写  131072 字节
mul      读  147456 字节，写  131072 字节
silu     读  131072 字节，写  131072 字节
mul      读  262144 字节，写  131072 字节
8 个算子逐个执行：共 1589440 字节；融合成 1 个 kernel：409600 字节，是原来的 25.8%
```

Fusion cuts the memory traffic to a quarter; under a bandwidth bottleneck the time falls to about a quarter too, and 7 kernel launches are saved. A real compiler's fusion is far more involved and has to answer:

- **what can fuse**: elementwise operations with each other most easily; an elementwise operation can fuse into a reduction's prologue or epilogue; a matrix multiply's output can take elementwise operations (bias, activation, quantization) as an epilogue.
- **what should not fuse**: two reductions that both need a "global view" (row-wise softmax followed by a column-wise sum, say); a fusion whose register pressure cuts the parallelism; and when an intermediate result has several consumers, recomputing it may cost more than writing it back to device memory.
- So fusion is a decision driven by a **cost model**: Inductor scores it by the traffic saved and the register pressure, and TensorRT has a set of preset fusion patterns.

## The operator level: separating algorithm from schedule {#算子级算法与调度分离}

The **algorithm** of one operator (a matrix multiply, say) is fixed, but the **schedule** — how the loops are tiled, in what order, which level of storage, how it is parallelized — has countless forms that differ by tens of times in performance. Halide first proposed writing the two separately: the algorithm is written once and the schedule is described apart, to be tried again and again. TVM inherited that idea and replaced the hand-written schedule with **auto-tuning** (searching the schedule space and picking the fastest by a cost model or by measurement).

Triton took another road: the programmer writes the kernel in units of **tiles**, deciding which tile of data each program instance handles, and what happens **within** a tile goes to the compiler: which elements each thread takes, how accesses coalesce, how shared memory is used, how the Tensor Cores are called, how the multi-stage pipeline is built. Parameters like the tile size and the warp count are chosen from a handful of candidates by measurement with `triton.autotune`.
That lowers the bar to "able to write NumPy-style block operations", at the cost of handing some low-level control to the compiler (see the CUDA handbook's [Triton](../tools/triton.md)).

### A Triton kernel's compilation flow {#一个-triton-kernel-的编译流程}

<!-- i18n:diagram cb074e0b41 -->
```text
A Python function (@triton.jit)
  → Triton IR (ttir): block-level tensor operations, hardware independent
  → TritonGPU IR (ttgir): assigns every tensor a distribution across threads (a layout), inserting shared memory, asynchronous copies and pipelining
  → LLVM IR (llir)
  → PTX
  → cubin (the GPU machine code ptxas generates)
```

Every level can be pulled out of the compiled kernel object (`compiled.asm["ttgir"]`, `["ptx"]` and so on), and when performance falls short of expectations, looking at the layouts and the pipelining in the ttgir and at whether the PTX uses `mma` / `wgmma` instructions is the usual way to locate the problem.

### The new generation of tile languages {#新一代的-tile-语言}

Hopper and Blackwell brought new hardware features: the TMA (Tensor Memory Accelerator), `wgmma` (warpgroup-level matrix multiply), thread block clusters, and Blackwell's Tensor Memory. Using them to the full requires controlling "which warp does what" precisely (warp specialization: some warps only move data and some only compute), which is exactly the part Triton hands to the compiler and the compiler does not necessarily do well. Hence a batch of languages "one level above CUDA and one below Triton":

| Language | The idea |
| --- | --- |
| CUTLASS / CuTe (C++) and the CuTe DSL (Python) | describe exactly, with a "layout algebra", how a tensor is distributed across threads, registers and shared memory; FlashAttention-3 is built on it and DeepGEMM borrowed its concepts |
| TileLang | a tile-level DSL on top of TVM, allowing explicit control of the pipeline and the warp division of labour; some of the operators DeepSeek-V3.2 open-sourced come in a TileLang version |
| Triton Gluon | Triton's low-level extension, allowing layouts and warp specialization to be written directly |
| ThunderKittens | a C++ library whose basic unit is a 16×16 register tile |

The trend is: ordinary operators go to the compiler (generated automatically by Inductor), and the few performance-critical ones (attention, GEMM, MoE) are tuned by hand in these tile languages.

## Other compilers and inference engines {#其他编译器与推理引擎}

- **XLA**: TensorFlow / JAX's compiler, with HLO as its intermediate representation, and the main programming path for TPUs; JAX's `jit` is exactly "compile this function into XLA";
- **MLIR**: not a compiler but infrastructure for building compilers, an extensible multi-level intermediate representation (dialects). Triton, IREE, torch-mlir and many domestic accelerators' compilers are built on it;
- **TensorRT / TensorRT-LLM**: NVIDIA's inference optimizer, which does graph fusion, precision calibration (INT8 / FP8) and picking the fastest of several implementations (tactics) per operator by measurement; TensorRT-LLM adds LLM-specific plugins on top (paged attention, in-flight batching);
- **TVM**: an end-to-end open-source compilation stack from the graph level to code generation, with auto-tuning at its core, and the choice for many on-device and non-NVIDIA deployments.

## Lean on the compiler or write the kernel {#靠编译器还是手写-kernel}

| Case | Choice |
| --- | --- |
| elementwise operations, reductions and combinations of them (normalization, activation, residual, quantization's pre- and post-processing) | leave it to `torch.compile` / Inductor, usually already close to the bandwidth ceiling |
| a matrix multiply of standard shape | cuBLAS / cuBLASLt, or Inductor's `max-autotune` |
| attention (paged, variable-length, GQA / MLA, sparse), MoE's grouped GEMM, GEMM with non-standard quantization | use a mature library (FlashAttention, FlashInfer, FlashMLA, DeepGEMM), or write it in Triton / CuTe / TileLang |
| fusing communication with computation, operators spanning devices | by hand (NVSHMEM, a custom all-reduce); compilers can hardly generate these automatically today |

Asked in an interview "how would you optimize operator X", a good order to answer in is: work out with the roofline whether it is bandwidth- or compute-bound and how far from the ceiling; then see whether fusion (the compiler) solves it; and only then a hand-written kernel, in which language and why.

!!! interview "How to explain it"
    On AI compilers: three levels, graph, operator and code generation; inference (decode especially) is bandwidth-bound, so what benefits most is graph-level fusion, and what it saves is memory traffic and kernel launches (this chapter's RMSNorm + SiLU + multiply drops to about a quarter of the traffic once fused). Fusion is decided by a cost model: elementwise operations and the pre- and post-processing around a row-wise reduction fuse easily, while mismatched reduction directions or excessive register pressure do not. The operator level's core is "separating algorithm from schedule": Halide / TVM search the schedule, and Triton has the programmer manage the tile and the compiler manage what is inside it; Hopper / Blackwell brought about lower-level tile languages like the CuTe DSL and TileLang.

## Exercises {#练习}

1. Add a `torch.softmax` call (over the last dimension) to `fx_fusion.py`. Is putting it in the fusion group reasonable? What if the softmax is over dimension 0?

??? success "Answer"
    A softmax over the last dimension is a "row-wise" reduction plus elementwise operations, just like RMSNorm: one row's data can be finished within one thread block, which suits fusion (this is exactly the basis of online softmax and FlashAttention).
    Over dimension 0, the reduction direction differs from RMSNorm's (the last dimension): a thread block handling one row cannot get a whole column, so fusing would need the whole tensor on chip or synchronization across thread blocks, which usually does not pay, and the compiler cuts into two kernels here.

2. Why does Triton's `autotune` usually tune separately keyed by "input shape"? What problem does that cause in an inference service?

??? success "Answer"
    The best tile size, warp count and number of pipeline stages depend on the problem's shape: a small M (decode's small batch) wants finer tiling along N and K or even split-K, while a large M (prefill) wants large tiles. So different shapes are tuned separately and the results cached.
    In an inference service the batch and the sequence length change constantly: tuning once per new shape introduces long stalls in production. The common practice is to bucket by range (rounding the batch up to a power of two, say), to pre-tune the common buckets at startup, or to tune offline and write the configuration table into the code (many inference libraries' MoE and quantized GEMM ship configuration files generated per GPU model).

## Summary {#小结}

- [x] An AI compiler has three levels, graph, operator and code generation; inference benefits most from graph-level fusion, which saves memory traffic and kernel launches.
- [x] Fusion is cost-model driven: elementwise operations and the pre- and post-processing around a row-wise reduction fuse easily; mismatched reduction directions and excessive register pressure do not.
- [x] Operator-level optimization's core is "separating algorithm from schedule": Halide / TVM search the schedule, and Triton has the programmer manage the tile and the compiler manage what is inside it.
- [x] Triton's compilation chain: ttir → ttgir → LLVM IR → PTX → cubin; for a performance problem, look at the ttgir's layouts and the PTX's instructions.
- [x] Hopper / Blackwell brought about lower-level tile languages like the CuTe DSL, TileLang and Gluon; ordinary operators go to the compiler, while attention, GEMM, MoE and fused communication go to a library or a hand-written kernel.
