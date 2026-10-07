# GPU architecture and the programming model

<p class="lead">Writing fast CUDA starts with having a map of the GPU in your head: how the compute units are organized, how threads are scheduled, where the data sits, and how fast each level is. This chapter draws that map, and every optimization later in the book has its reason somewhere on it.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why can a GPU hide memory latency with "lots of threads"? What does a CPU rely on instead?
    2. How do a thread block and an SM relate? Can one block run across several SMs?
    3. What is a warp? Why 32?
    4. Will a program compiled with `-arch=sm_80` run on an H100? On a T4?
    5. An A100 does about 19.5 TFLOPS of FP32 with about 2 TB/s of memory bandwidth. A kernel does 2 floating-point operations per float it reads. Which of the two bounds its performance?

??? success "Answers (try it yourself first, then expand)"
    1. A GPU keeps many warps resident on each SM, and when one warp waits on memory the scheduler switches to another ready warp at essentially no cost; as long as there is enough parallel work, the latency hides behind other warps' compute. A CPU lowers a single thread's latency with large caches, out-of-order execution and branch prediction.
    2. A block is scheduled onto one SM and stays there from start to finish, never crossing SMs; one SM can hold several blocks at once (bounded by registers, shared memory and thread count).
    3. A warp is an execution unit of 32 threads, with one instruction issued to all 32 at once (SIMT). The 32 is a hardware choice: wide enough to amortize fetch and scheduling, narrow enough that divergence does not waste too much.
    4. It will run on an H100 (sm_90): the binary carries compute_80 PTX, which the driver compiles just in time into sm_90 SASS. It will not run on a T4 (sm_75): both the SASS and the PTX are newer than the device, which raises "no kernel image".
    5. 2 operations per 4 bytes read is an arithmetic intensity of 0.5 FLOP/byte, far below the A100's FP32 ridge point (about 19.5 T / 2 T ≈ 10 FLOP/byte), so bandwidth bounds it: about 2 TB/s × 0.5 = 1 TFLOPS.

## CPU and GPU: two design philosophies {#cpu-与-gpu两种设计哲学}

A CPU is designed for **low latency**: a few powerful cores, with much of the die spent on caches, branch prediction and out-of-order execution, all so that **one thread** finishes as soon as possible.

A GPU is designed for **high throughput**: thousands of simple compute units with comparatively little control logic and cache. One thread is slow and waits hundreds of cycles for a memory access, but a GPU keeps an enormous number of threads resident: while one group waits on memory, the scheduler immediately runs another group that is ready. As long as enough threads are ready, the compute units always have work and the memory latency is **hidden**.

That is the first rule of GPU programming: **give the hardware enough parallel work**. A kernel that launches a few dozen threads will never be fast on a GPU.

## The hardware hierarchy {#硬件层次}

From the top down:

<!-- i18n:diagram 50878c4600 -->
```text
GPU
├── device memory (HBM / GDDR): tens to hundreds of GB, shared by every SM
├── L2 cache: tens of MB, shared by every SM
└── many SMs (Streaming Multiprocessors)
    ├── register file: 64K 32-bit registers (256 KB)
    ├── L1 cache / shared memory: one SRAM, split by configuration
    └── 4 processing partitions (SMSP), each with:
        ├── a warp scheduler and an instruction issue unit
        ├── FP32 / INT32 units (the so-called "CUDA cores")
        ├── Tensor Cores (units dedicated to matrix multiply)
        ├── special function units (SFU: exp, sin, rsqrt and so on)
        └── load/store units (LD/ST)
```

**The SM is the GPU's basic execution unit**, roughly a "CPU core" except that it holds thousands of threads at once. A GPU's size is mostly a matter of how many SMs it has.

Key specifications of a few common GPUs (from NVIDIA's official figures; the Tensor Core numbers are dense, without sparsity):

| GPU | Architecture | SMs | FP32 | BF16/FP16 Tensor | Memory bandwidth | L2 | Max shared memory per SM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T4 | Turing sm_75 | 40 | 8.1 TFLOPS | 65 TFLOPS (FP16) | 320 GB/s | 4 MB | 64 KB |
| A100 SXM 80GB | Ampere sm_80 | 108 | 19.5 TFLOPS | 312 TFLOPS | 2039 GB/s | 40 MB | 164 KB |
| RTX 4090 | Ada sm_89 | 128 | 82.6 TFLOPS | — | 1008 GB/s | 72 MB | 100 KB |
| H100 SXM | Hopper sm_90 | 132 | 67 TFLOPS | 989 TFLOPS | 3.35 TB/s | 50 MB | 228 KB |

Two ratios come up over and over:

- **Compute over bandwidth**: on an A100, FP32 is 19.5e12 / 2.039e12 ≈ 9.6 FLOP/byte and BF16 Tensor Core is 312e12 / 2.039e12 ≈ 153 FLOP/byte. That is how many operations a kernel must do per byte read from memory to keep the math units fed. Fall short and bandwidth is the bottleneck. This is the heart of the [roofline model](execution.md#roofline-模型).
- **On-chip storage is tiny**: an SM's registers plus shared memory come to a few hundred KB, the whole GPU's L2 to tens of MB, while device memory is tens of GB. A high-performance kernel is essentially an exercise in reusing data on chip as much as possible.

## The programming model: threads, blocks, grids {#编程模型线程线程块网格}

CUDA organizes threads in three levels:

![Figure: the thread hierarchy](../assets/figures/thread-hierarchy.svg){.aig-svg}

- **Thread**: the smallest unit running the kernel function, with its own registers and index `threadIdx`.
- **Block**: a group of up to 1024 threads, indexed by `blockIdx` and sized by `blockDim`. Threads in one block exchange data through **shared memory** and synchronize with `__syncthreads()`.
- **Grid**: all the blocks of one kernel launch, sized by `gridDim`. There is **no** guaranteed order between blocks and no way to synchronize them directly.

```cuda
// launch the grid: numBlocks blocks of threadsPerBlock threads each
kernel<<<numBlocks, threadsPerBlock>>>(args...);

// inside the kernel, compute the global thread index
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`threadIdx`, `blockIdx` and the rest are three-dimensional (`.x .y .z`), which is convenient for two-dimensional images and matrices.

### How software maps onto hardware {#软件层次如何映射到硬件}

| Software | Hardware | The key fact |
| --- | --- | --- |
| grid | the whole GPU | blocks are handed out to the SMs, and run in waves when there are far more of them than SMs |
| block | one SM | a block **runs on exactly one SM** from start to finish and never migrates; one SM can hold several blocks |
| warp | one processing partition of an SM | every 32 consecutive threads of a block form a warp, which is the real unit of scheduling and execution |
| thread | one execution lane | with private registers |

**How many threads an SM can hold** is bounded by several limits (on an A100): at most 2048 threads (64 warps), at most 32 blocks, 65536 registers and 164 KB of shared memory. A block becomes resident only if every one of these fits. The fraction of the warp ceiling actually resident is called **occupancy**; see [the execution model](execution.md#占用率occupancy).

## Warps and SIMT {#warp-与-simt}

A GPU issues instructions in units of a **warp (32 threads)**: at any instant the threads of a warp run **the same instruction** on different data. NVIDIA calls this **SIMT** (Single Instruction, Multiple Threads).

A few direct consequences:

- **Block sizes should be multiples of 32.** A block of 100 threads really occupies 4 warps, with 28 lanes of the last one idle.
- **A branch within one warp serializes.** If half the warp takes the `if` and half the `else`, the two paths run one after the other, each masking off the other half. This is **warp divergence**; see [the execution model](execution.md#分支发散).
- **Memory accesses coalesce per warp.** A warp's 32 threads issue their accesses together, and if the addresses are contiguous the hardware merges them into very few memory transactions. This is the first rule of memory optimization; see [the memory hierarchy](memory.md#全局内存合并访问).

Every cycle, each warp scheduler picks one "ready" warp (whose operands have arrived) from those it manages and issues an instruction. A warp waiting on memory holds no compute unit, and the scheduler runs another warp instead. **Switching warps is free**, because every resident warp's registers stay in the register file the whole time, which is nothing like a CPU's thread switch.

!!! warning "Independent thread scheduling: never assume threads in a warp are synchronized"
    From Volta (sm_70) on, each thread in a warp has its own program counter and diverged threads may interleave. Old code that assumed "threads in a warp always execute in lockstep, so no synchronization is needed" is no longer safe. Exchange data within a warp through the `_sync` intrinsics (such as `__shfl_sync`), and synchronize with `__syncwarp()` where needed. See [warp-level programming](sync-warp.md).

## The memory hierarchy at a glance {#内存层次速览}

| Storage | Where | Scope | Capacity | Latency | Notes |
| --- | --- | --- | --- | --- | --- |
| registers | in the SM | one thread | at most 255 per thread | ~1 cycle | the fastest; too many lower occupancy or spill |
| shared memory | in the SM | one block | tens to a couple of hundred KB per SM | ~tens of cycles | a fast cache you manage by hand |
| L1 cache | in the SM | hardware managed | shares the shared-memory SRAM | ~tens of cycles | caches global-memory accesses |
| L2 cache | on chip | every SM | a few MB to tens of MB | ~200 cycles | every global-memory access goes through L2 |
| global memory | device memory | every thread, persists across kernels | tens of GB | ~400-800 cycles | what `cudaMalloc` returns |
| constant memory | device memory + its own cache | every thread, read only | 64 KB | fast on a cache hit | good when every thread of a warp reads the same value |
| local memory | device memory | one thread | — | like global memory | where register spills and dynamically indexed local arrays go; very slow |

![Figure: the H100 memory hierarchy](../assets/figures/memory-hierarchy.svg){.aig-svg}

The latency numbers are orders of magnitude and vary quite a bit across architectures. Their **relative order** is stable, though: registers ≪ shared memory ≈ L1 ≪ L2 ≪ device memory. Optimizing memory access is the art of keeping data at a higher level. The next chapter, [the memory hierarchy and access optimization](memory.md), goes into it.

## Architecture names and compute capability {#架构代号与计算能力}

Every GPU generation has a **compute capability** (CC) version, written `sm_XY` at compile time:

| Architecture | CC | Representative products | New features this handbook uses |
| --- | --- | --- | --- |
| Volta | 7.0 | V100 | first-generation Tensor Cores, independent thread scheduling |
| Turing | 7.5 | T4, RTX 20 | INT8/INT4 Tensor Cores; **the lowest architecture CUDA 13 supports** |
| Ampere | 8.0 / 8.6 | A100 / RTX 30, A10 | BF16, TF32, `cp.async`, more shared memory |
| Ada Lovelace | 8.9 | RTX 40, L4, L40, L20 | FP8 Tensor Cores |
| Hopper | 9.0 | H100, H800, H20 | TMA, thread-block clusters, wgmma, the FP8 Transformer Engine |
| Blackwell | 10.0 / 10.3 / 12.0 | B200 / B300 / RTX 50 | fifth-generation Tensor Cores (tcgen05), tensor memory, FP4 |

Know the minimum architecture each feature needs; the handbook marks them like <span class="arch">sm_80+</span>.

## The compilation flow: PTX and SASS {#编译流程ptx-与-sass}

`nvcc` splits a `.cu` file in two: the host code goes to g++, and the device code is compiled first into **PTX** (a virtual assembly independent of any particular hardware) and then by `ptxas` into **SASS** (the real machine code of one GPU generation). The final executable embeds a "fatbinary" that can hold several versions of SASS along with PTX.

```bash
nvcc -arch=sm_80 app.cu          # equivalent to generating both sm_80 SASS and compute_80 PTX
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 app.cu   # several architectures plus PTX
nvcc -arch=native app.cu         # compile for this machine's GPU
```

At run time the driver looks for SASS matching the current GPU; failing that, it compiles the PTX **just in time (JIT)** into SASS for it. So:

- A program compiled with `-arch=sm_80` **can** run on an H100 (sm_90), through PTX JIT.
- But it **cannot** run on a T4 (sm_75): both the SASS and the PTX are newer, which raises `no kernel image is available for execution on the device`.
- Targets with an `a` suffix (`sm_90a`, `sm_100a`) enable features exclusive to that generation (Hopper's wgmma, say), and the generated code is **not forward compatible**: it runs on that generation only. Since CUDA 12.9 there is also an `f` suffix (`sm_100f`) for features portable within one "family".

To see how many registers and how much shared memory a kernel uses, compile with `--ptxas-options=-v` (or `-Xptxas -v`):

```text
ptxas info    : Used 14 registers, used 0 barriers, 360 bytes cmem[0]
```

This is very useful when analyzing occupancy.

!!! interview "Answering in an interview"
    Asked "how does a GPU differ from a CPU": a CPU lowers single-thread latency with large caches, out-of-order execution and branch prediction, while a GPU hides latency by switching between a huge number of warps, which is why the first rule is to give it enough parallel work. The hierarchy is grid, block, warp, thread; a block is pinned to one SM, and a warp (32 threads) is the unit of scheduling and execution. Then judge the bottleneck from the ratio of compute to bandwidth: an A100 does about 19.5 TFLOPS of FP32 at about 2 TB/s, so a kernel doing 2 operations per 4 bytes read is necessarily bandwidth-bound. On compilation: `-arch` decides which SASS and PTX are generated, and PTX can JIT onto a newer GPU but never the other way round.

## Exercises {#练习}

**1. How many threads can one A100 hold at once?** What happens if a kernel launches 100,000 threads?

??? success "Answer"
    108 SMs × 2048 threads per SM = 221,184 threads (6912 warps).

    Launching 100,000 threads is no problem: if the resources allow, every block can be resident at once. Launch 10 million and the blocks beyond the residency limit queue up, scheduled once earlier blocks finish and free their resources. There is no guaranteed order between blocks, which is exactly why blocks cannot wait on each other: the one being waited for may never get scheduled.

**2. Computing occupancy.** On an A100, a kernel uses 256 threads per block and 64 registers per thread, with no shared memory. How many blocks can one SM hold? What is the occupancy? And with 32 registers per thread?

??? success "Answer"
    - 64 registers: each block needs 256 × 64 = 16384 registers, an SM has 65536, so at most 4 blocks, that is 1024 threads or 32 warps, an occupancy of 32/64 = 50%.
    - 32 registers: each block needs 8192 registers, so registers allow 8 blocks; the thread ceiling allows 2048 / 256 = 8 blocks; the block ceiling is 32. The minimum is 8 blocks, 2048 threads, 100% occupancy.

    (Registers are allocated in granules in practice; Nsight Compute or CUDA's occupancy calculator gives the exact numbers.)

**3. Finding the bottleneck.** A kernel reads two float arrays and writes one (`c[i] = a[i] * b[i] + 1`), doing 2 floating-point operations per element. On an A100, is it compute-bound or bandwidth-bound? What is the theoretical minimum time for a billion elements?

??? success "Answer"
    Each element moves 12 bytes and does 2 operations, an **arithmetic intensity** of 2 / 12 ≈ 0.17 FLOP/byte, far below the A100's FP32 ridge of 9.6, so it is **bandwidth-bound**.

    A billion elements move 12 GB, which at 2039 GB/s is about 5.9 ms in theory. Achievable bandwidth is usually 80-90% of peak, so about 6.5-7.5 ms in practice. For a kernel like this the goal is to get measured bandwidth close to peak; compute does not matter at all.

**4. Compiler options.** Your team's inference service deploys on A100, A10 (sm_86) and H100. How would you set `-gencode`?

??? success "Answer"
    Generate SASS for each target architecture so that no JIT is needed at startup, and add PTX for the highest version for future GPUs:

    ```bash
    nvcc -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_90,code=sm_90 \
         -gencode arch=compute_90,code=compute_90 ...
    ```

    If the H100 build needs Hopper-only instructions such as wgmma, it has to use `sm_90a`, and that code has to be compiled and dispatched separately from the other architectures.

## Summary {#小结}

- [x] A GPU hides latency with a huge number of threads, so the first rule is to give it enough parallel work.
- [x] grid, block, warp, thread; a block is pinned to one SM, and a warp (32 threads) is the unit of execution.
- [x] On-chip storage (registers, shared memory, L1/L2) is small and fast while device memory is large and slow, so performance hinges on reusing data.
- [x] The ratio of compute to bandwidth decides whether a kernel is compute-bound or memory-bound.
- [x] `-arch` decides which SASS and PTX are generated; PTX can JIT onto a newer GPU, never the reverse.
