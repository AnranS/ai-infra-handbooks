# Profiling: Nsight Systems and Nsight Compute

<p class="lead">Measure before optimizing, which matters especially on a GPU: a slow program may have a slow kernel, or gaps between kernels, or a CPU that cannot keep up, or copies that do not overlap. NVIDIA provides two complementary tools: Nsight Systems for the whole program's timeline and Nsight Compute for one kernel in depth. Using them and reading their metrics is a hard requirement for a kernel development role.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What question does each of Nsight Systems and Nsight Compute answer?
    2. The timeline shows many small gaps between kernels. What does that mean? How do you fix it?
    3. What are the two percentages in Nsight Compute's "Speed Of Light" section?
    4. Which metrics confirm that a kernel has bank conflicts or uncoalesced accesses?
    5. What do the warp stall reasons "Long Scoreboard" and "MIO Throttle" mean?

??? success "Answers (try it yourself first, then expand)"
    1. Nsight Systems shows the overall timeline: what the CPU and GPU are doing, whether there are gaps between kernels, whether copies and compute overlap. It answers "where does the time go". Nsight Compute analyses one kernel in depth: memory, compute, stall reasons. It answers "why is this kernel slow".
    2. The GPU is waiting on the CPU: too many kernels that are too small, a CPU that cannot submit fast enough, or a synchronization in between. Submit the whole chain at once with a CUDA Graph, fuse small kernels, and remove unnecessary synchronizations.
    3. The percentages of peak for memory throughput (across the levels) and compute (SM) throughput. Whichever is near peak is the limit; if both are low the kernel is latency-bound (not enough parallelism, too much waiting).
    4. Look at global sectors per request: a coalesced access is about 4 sectors per request (4 bytes per thread), and much more than that means it is not coalescing; for shared memory, look at the bank-conflict metrics (access rounds per instruction, conflict counts).
    5. Long Scoreboard: waiting for data from global (or local) memory, meaning the memory latency is not hidden. MIO Throttle: the queue for shared-memory and similar instructions is full, usually too many shared-memory instructions or bank conflicts.

## Look at the whole first, then one kernel {#先看全局再看单个-kernel}

| Tool | Command | The question it answers |
| --- | --- | --- |
| **Nsight Systems** (nsys) | `nsys profile ./app` | Where does the time go? Is the GPU ever idle? Do CPU and GPU, copies and compute, overlap? Which kernel takes longest? |
| **Nsight Compute** (ncu) | `ncu ./app` | Why is this kernel slow? Is the bottleneck memory, compute or latency? Which line of code? |

The standard flow:

1. look at the timeline with nsys, find the most expensive part, and decide whether the kernel itself is slow or the cost is outside it (launch overhead, copies, synchronization, CPU work);
2. analyse the slowest few kernels with ncu;
3. change the code and **measure again** to confirm the optimization really worked.

A simple time model first, for a feel of what "the whole" means: with a thousand small kernels in one decode step, how much is launch overhead and gaps (the same tool as in the Inference Systems handbook):

<div class="aig-widget" data-widget="launch-overhead"></div>

## Nsight Systems {#nsight-systems}

```bash
nsys profile -o report --trace=cuda,nvtx,osrt ./app       # produces report.nsys-rep
nsys stats report.nsys-rep                                 # prints the various statistics on the command line
nsys profile --trace=cuda,nvtx python train.py            # works just as well on a Python program
```

Viewing the timeline in the GUI (Nsight Systems can be installed on your own machine and open a report generated on a server), watch for:

| What you see | Possible cause | What to do |
| --- | --- | --- |
| many gaps of a few microseconds between kernels | too many kernels that are too small, so launch overhead dominates; or the CPU cannot issue fast enough | fuse kernels; submit the whole graph at once with CUDA Graphs, see [streams and CUDA Graphs](streams.md) |
| long GPU idle stretches with a busy CPU | CPU-side preprocessing, Python overhead, waiting on a synchronization | move work to the GPU; run asynchronously; cut `cudaDeviceSynchronize` and implicit synchronizations like `.item()` |
| copies and kernels running in series | no pinned memory and no multiple streams | pinned memory, asynchronous copies and a multi-stream pipeline |
| one kernel taking most of the time | that is your target | analyse it with ncu |

### Annotating code with NVTX {#用-nvtx-标注代码}

On a timeline, a pile of kernels is hard to attribute to phases of the program. NVTX lets you mark named ranges that show up as coloured intervals:

```cuda title="nvtx_demo.cu"
// nvtx_demo.cu - marking the program's phases with NVTX and viewing them on the Nsight Systems timeline
// build: nvcc -O3 -arch=sm_75 nvtx_demo.cu -o nvtx_demo
// profile: nsys profile --trace=cuda,nvtx -o nvtx_demo ./nvtx_demo
#include "common.cuh"
#include <nvtx3/nvToolsExt.h>

__global__ void scale(float* x, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}

int main() {
  const int n = 1 << 24;
  std::vector<float> h(n, 1.f);
  float* d;

  nvtxRangePushA("setup");
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  nvtxRangePop();

  nvtxRangePushA("compute");
  for (int step = 0; step < 10; ++step) {
    nvtxRangePushA("step");
    scale<<<(n + 255) / 256, 256>>>(d, n, 1.01f);
    CUDA_CHECK_LAST();
    nvtxRangePop();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  const float expect = std::pow(1.01f, 10.f);
  const bool ok = std::fabs(h[0] - expect) < 1e-4f && std::fabs(h[n - 1] - expect) < 1e-4f;
  std::printf("%s (x = %.6f, expected %.6f)\n", ok ? "PASS" : "FAIL", h[0], expect);
  CUDA_CHECK(cudaFree(d));
  return ok ? 0 : 1;
}
```

NVTX v3 is header-only and needs no extra linking. Note that a CPU-side NVTX range only marks "when these CUDA calls were issued"; when the kernels actually ran on the GPU is in the GPU rows. In PyTorch, `torch.cuda.nvtx.range_push/range_pop` or `torch.profiler` gives the same effect.

## Nsight Compute {#nsight-compute}

```bash
ncu ./app                                          # profile every kernel (the default metric set)
ncu --set full -o gemm_report ./gemm               # collect every metric and save as gemm_report.ncu-rep
ncu -k regex:sgemm_v4 --launch-skip 2 --launch-count 1 --set full -o v4 ./gemm
                                                    # profile only the kernels whose name matches, skipping the first 2 launches and capturing 1
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./app   # collect only the named metrics
ncu --query-metrics                                 # list every metric this GPU supports
```

ncu **replays** each profiled kernel several times to collect different metrics, so the program runs much slower; profile only the few kernels you care about (`-k`, `--launch-count`). By default it also locks the GPU clocks to base (`--clock-control base`) and clears the caches before each replay (`--cache-control all`), which makes results stable and comparable but means the absolute times differ from a normal run.

Without permission to read the performance counters on a server you get `ERR_NVGPUCTRPERM`, which an administrator has to enable (or the container needs the right capability).

### Reading the report: start from Speed Of Light {#读懂报告从-speed-of-light-开始}

**GPU Speed Of Light Throughput** is the first section to read, giving two percentages:

- **Compute (SM) Throughput**: utilization of the compute-side units (the execution pipelines, instruction issue) against peak;
- **Memory Throughput**: utilization of whichever level of storage (DRAM, L2, L1, shared memory) is busiest.

Use them to classify the kernel:

| Case | Verdict | What to look at next |
| --- | --- | --- |
| memory high (above 70%, say), compute low | memory-bound | Memory Workload Analysis: which level is saturated? DRAM saturated means you are near the ceiling and can only move less data; L1 or shared memory saturated may mean bank conflicts or uncoalesced accesses |
| compute high, memory low | compute-bound | Compute Workload Analysis: which pipeline is busiest (FMA, Tensor, ALU, SFU)? Can a faster instruction be used (Tensor Cores, fast math)? |
| both low (under 40%, say) | **latency-bound**: neither computing nor moving at capacity, with warps waiting most of the time | Occupancy and Warp State Statistics: are there enough resident warps? What are they waiting for? |
| both high | already near the hardware limit | consider an algorithmic change (fusion, a different data type, less computation) |

The report's **roofline chart** plots the kernel in arithmetic-intensity-against-performance space, showing at a glance which "roof" it is closer to.

### Common metrics {#常用指标}

| What you want to know | Metric / where in the report |
| --- | --- |
| the kernel's time | `gpu__time_duration.sum` |
| bytes read and written to device memory | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| memory bandwidth utilization | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| whether global accesses coalesce | sectors per request: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` divided by `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`. With 4 bytes per thread and full coalescing it is 4; the larger it is, the more is wasted. The report's Source Counters section even names the lines with uncoalesced accesses |
| shared-memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (loads), `..._op_st.sum` (stores) |
| achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active`; the Occupancy section also gives the theoretical occupancy and the limiting factor (registers, shared memory, block size) |
| register spills | local-memory accesses in the Source view; `-Xptxas -v` at compile time |

The metric names are long but regular: `unit__what_is_counted_qualifier.rollup`. `dram__bytes_read.sum` is "bytes read by the DRAM unit, summed". When unsure, search with `ncu --query-metrics`.

### Warp stall reasons {#warp-停顿原因}

**Warp State Statistics** counts what warps are waiting for when they cannot issue. The common ones:

| Stall reason | Meaning | Usual remedy |
| --- | --- | --- |
| Long Scoreboard | waiting for data from global/local memory (L1TEX) | more requests in flight (vectorization, several elements per thread, prefetching); higher occupancy; check for register spills |
| Short Scoreboard | waiting for shared memory or a special function unit | fewer shared-memory accesses (register tiling); remove bank conflicts |
| MIO Throttle | the queue for shared-memory and similar "memory input/output" instructions is full | the same, and typical of a kernel with very dense shared-memory reads such as GEMM v2 |
| LG Throttle | the global/local memory instruction queue is full | vectorize to cut the instruction count |
| Barrier | waiting for other warps at `__syncthreads()` | synchronize less; balance the work between warps |
| Math Pipe Throttle | one compute pipeline cannot keep up | usually means you are compute-bound, which is a good sign; consider another pipeline |
| Not Selected | ready, but the scheduler picked another warp this cycle | plenty of ready warps, so not a problem |

The **Source view** (which needs `-lineinfo` at compile time) maps these metrics onto each source line and each SASS instruction, pinpointing the hot spots.

## A complete analysis example {#一个完整的分析示例}

Take [matrix transpose](../kernels/transpose.md) and verify the bank-conflict reasoning:

```bash
nvcc -O3 -lineinfo -arch=sm_80 transpose.cu -o transpose
ncu -k regex:transpose_smem --metrics \
    gpu__time_duration.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./transpose
```

Expect to see `transpose_smem<0>` (`[32][32]`) with a large shared-memory read conflict count and `transpose_smem<1>` (`[32][33]`) with almost none; the latter has higher DRAM throughput and a shorter time.

Then take [GEMM](../kernels/gemm.md) and compare v2 with v4: v2's main stall reasons will be MIO Throttle / Short Scoreboard (shared-memory accesses too dense) with low compute utilization, while v4's FMA pipeline utilization rises clearly. That is the data behind "the bottleneck moved from shared memory to compute".

Being able to narrate that "hypothesis, verify with a metric, optimize, verify again" process in an interview is far more convincing than listing techniques.

!!! interview "Answering in an interview"
    Asked "a kernel is slow, how do you analyse it": start with Nsight Systems and the overall timeline (gaps between kernels mean the CPU cannot keep up, which CUDA Graphs and fusion fix; NVTX marks the phases), then take the slowest kernels to Nsight Compute: classify the bottleneck from Speed Of Light's memory and compute percentages; use sectors per request to judge coalescing, the bank-conflict metrics to confirm shared-memory conflicts, and the warp stall reasons to see what is being waited for (Long Scoreboard is global memory, MIO Throttle is shared-memory instructions queueing). Compile with `-lineinfo` and map the metrics onto source lines in the Source view.

!!! info "Related chapters"
    - [Profiling an inference engine](serving://perf/profiling/) (Inference Systems: profiling the whole serving path)
    - [Linux performance tools](cs://os/perf-tools/) (Computer fundamentals: perf and flame graphs on the CPU side)

## Exercises {#练习}

**1. Analyse the reduction.** Profile v2 and v5 of [reduction](../kernels/reduction.md) with ncu, comparing their DRAM throughput, achieved occupancy and main stall reasons, and explain why v5 is faster.

??? success "Approach"
    Expect v2's DRAM throughput to be clearly lower than v5's; v2 stalls mainly on Barrier (a `__syncthreads()` every round) and Long Scoreboard (one request in flight per thread); v5 has many requests in flight (float4, several elements per thread), its DRAM throughput approaches peak, and it stalls mainly on Long Scoreboard, but with memory nearly saturated, which is the ideal state for a memory-bound kernel.

**2. Reading an nsys timeline.** In an inference service's decode step, nsys shows over 300 kernels per step, each running 5-20 µs, with an average gap of 4 µs and only 60% GPU utilization. What are the likely causes and the directions to optimize?

??? success "Answer"
    The gaps come mostly from kernel launch overhead and the CPU side's issue rate (Python, framework scheduling): 300 kernels × 4 µs ≈ 1.2 ms of idle per step. Directions: (1) **CUDA Graphs**: capture the whole decode step as a graph and launch it once, which both vLLM and SGLang do for decode by default; (2) **fusion**: merge elementwise operations, normalization and residual adds to cut the kernel count, which `torch.compile` does partly on its own; (3) overlap the CPU's scheduling with the GPU's execution (SGLang's overlap scheduler prepares the next batch on the CPU while the GPU computes the current one).

## Summary {#小结}

- [x] Look at the whole timeline with nsys first, then analyse the slowest kernels with ncu.
- [x] NVTX marks the phases; gaps between kernels are fixed with CUDA Graphs and fusion.
- [x] Start ncu at Speed Of Light: memory-bound, compute-bound and latency-bound each point to different optimizations.
- [x] Judge coalescing from sectors per request, confirm shared-memory conflicts with the bank-conflict metrics, and locate what is being waited for from the stall reasons.
- [x] Compile with `-lineinfo` and map the metrics onto source lines in the Source view.
