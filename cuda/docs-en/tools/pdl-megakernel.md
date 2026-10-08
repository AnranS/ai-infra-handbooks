# The gaps between kernels: PDL and megakernels

<p class="lead">CUDA Graphs removed the CPU's cost of launching kernels one by one, but between the hundreds of kernels in a decode step the GPU still has gaps: the next kernel cannot start until the previous one has ended completely, and once it starts it first does a stretch of preparation that has nothing to do with the data. The smaller the model and the batch, the shorter each kernel and the larger those gaps' share. This chapter first estimates how large the gaps are, then covers three answers: PDL (Programmatic Dependent Launch, from Hopper on) overlaps neighbouring kernels, fusion cuts the number of kernels, and a megakernel writes the whole forward pass into one resident kernel. It ends with what vLLM and SGLang actually do.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. With CUDA Graphs in use, what overhead remains between two neighbouring kernels?
    2. What do `cudaTriggerProgrammaticLaunchCompletion()` and `cudaGridDependencySynchronize()` do in PDL? What can and cannot the next kernel do before it waits?
    3. Why do inference engines usually turn PDL on only when there are few tokens?
    4. What lets a megakernel remove the kernel boundaries? What does it cost?
    5. Why does a 0.6B model at batch 1 need these optimizations more than an 8B model?

??? success "Answers (try it yourself first, then expand)"
    1. Within one stream, the next kernel waits for every block of the previous one to end and its writes to become visible before the GPU's front end even schedules it and distributes its blocks to the SMs; each block then has a stretch of data-independent preparation (index arithmetic, reading weights, initializing shared memory); and the previous kernel's last wave of blocks usually leaves most SMs idle (the tail effect). All of that happens on the GPU, where CUDA Graphs cannot help.
    2. Once every block of the previous kernel has called `cudaTriggerProgrammaticLaunchCompletion()` (or exited), a next kernel launched with the PDL attribute may start. In that next kernel, `cudaGridDependencySynchronize()` returns only once the previous kernel has fully finished and its writes are visible. Before that wait it may only do things unrelated to the previous kernel: read weights, compute indices, initialize shared memory. It must not read the previous kernel's output, nor write data the previous kernel may still be reading.
    3. An early-launched block holds the SM's registers, shared memory and a block slot while it waits. With many tokens each kernel is long and runs several waves of blocks, so the gaps are a small share and the early blocks merely compete with the previous kernel; with few tokens the kernels are short and dominated by launch overhead, which is when overlapping pays. Several of vLLM's fused kernels turn PDL on only at 16 tokens or fewer.
    4. It launches a single resident (persistent) kernel, one block per SM, that claims tasks in turn from an instruction table (a GEMV tile, one attention head and so on), with dependencies expressed through counters in global memory; a task starts the moment its dependencies are met, and it can even prefetch the next task's weights while waiting. The costs: every task shares one kernel's resource configuration (registers and shared memory sized for the largest), all the scheduling and synchronization is hand-written, memory ordering and deadlocks are your problem, and it generalizes badly across models, batch sizes and parallelism schemes.
    5. A 0.6B model at batch 1 moves very little data per kernel, with half of them under 1 µs, the same order as the one or two microseconds at each kernel boundary; an 8B model moves a dozen or more microseconds' worth per kernel, so the gaps are a small fraction. The same holds as the batch and the context grow: more data, a smaller share of gaps.

## Where the gaps come from {#空隙从哪里来}

The Inference Systems handbook's [CUDA Graphs and torch.compile](serving://engine/graphs-compile/) worked it out: without CUDA Graphs, a decode step's thousand kernels each cost a few microseconds of CPU launch time and the GPU spends most of its time waiting on the CPU. [CUDA Graphs](streams.md#cuda-graphs) record the whole step and submit it once, which solves the CPU side.

But on the GPU side, there is still a blank between two kernels of one stream:

<!-- i18n:diagram f1c5d52b7a -->
```text
without PDL:
  kernel A  [==== body ====][tail: the last wave of blocks, SMs not full]
  kernel B                                               |sched| [prologue: read weights, compute indices] [==== body ====]
                                                         ↑ starts only after every block of A ends and its writes are visible

with PDL:
  kernel A  [==== body ====][tail]  ← each block triggers after writing its output
  kernel B            |sched| [prologue] [wait for A]  [==== body ====]
                                    ↑ cudaGridDependencySynchronize()
```

- **a kernel boundary is a global synchronization**: the next kernel's blocks start only after every block of the previous one has ended and its writes are visible, and then the GPU's front end schedules the new kernel and distributes its blocks;
- **the tail effect**: the previous kernel's last wave of blocks usually cannot fill every SM, leaving most of them idle for that stretch;
- **the prologue**: every block opens with preparation that does not depend on the previous step: index arithmetic, reading this layer's weights from global memory, initializing shared memory.

All of this happens on the GPU, beyond CUDA Graphs' reach. Removing it means overlapping neighbouring kernels (PDL), having fewer kernels (fusion), or doing away with kernel boundaries altogether (a megakernel).

## How large are the gaps: an estimate {#空隙占多少一个估算}

Every decode kernel is memory-bound, so its time is roughly the bytes it moves divided by the bandwidth. Below, one Qwen3 decode step is broken into kernels (fused the way vLLM does: residual add plus RMSNorm is one kernel, QK-Norm plus RoPE another), the data time computed, and g microseconds of gap added at each boundary. g differs by GPU and by kernel, so several values show the sensitivity:

```python title="decode_gaps.py"
"""decode_gaps.py —— 把 decode 一步的时间拆成"读写数据"和"kernel 边界的空隙"，看空隙占多少。

假设：H100 的带宽 3.35 TB/s，kernel 能跑到其中的 85%；decode 的每个 kernel 都受访存限制，时间 = 读写字节数 / 有效带宽。
kernel 之间的空隙 g 取 0.5～3 µs 几档看敏感度；逐个启动（不用 CUDA Graph）时，CPU 发射每个 kernel 算 5 µs。
"""

BW = 3.35e12 * 0.85                  # effective bandwidth, bytes per second
CPU_LAUNCH_US = 5.0
GAPS_US = (0.5, 1.0, 2.0, 3.0)
MODELS = {  # the Qwen3 configurations (hidden, intermediate, layers, Q heads, KV heads, head_dim, vocabulary)
    "Qwen3-0.6B": dict(hidden=1024, inter=3072, layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936),
    "Qwen3-8B": dict(hidden=4096, inter=12288, layers=36, heads=32, kv_heads=8, head_dim=128, vocab=151936),
}


def kernels(cfg, batch, context, nbytes=2):
    """decode 一步的 kernel 列表：(名字, 读写的字节数)。融合程度参照 vLLM：残差加 + RMSNorm、QK-Norm + RoPE 各是一个 kernel"""
    h, inter = cfg["hidden"], cfg["inter"]
    q, kv = cfg["heads"] * cfg["head_dim"], cfg["kv_heads"] * cfg["head_dim"]
    act = batch * h * nbytes
    layer = [
        ("add_rmsnorm", 4 * act),
        ("qkv_proj", h * (q + 2 * kv) * nbytes),
        ("qk_norm_rope", 2 * batch * (q + kv) * nbytes),
        ("kv_cache_write", 2 * batch * kv * nbytes),
        ("attention", batch * context * 2 * kv * nbytes),
        ("o_proj", q * h * nbytes),
        ("add_rmsnorm", 4 * act),
        ("gate_up_proj", h * 2 * inter * nbytes),
        ("silu_mul", 3 * batch * inter * nbytes),
        ("down_proj", inter * h * nbytes),
    ]
    tail = [("final_norm", 2 * act), ("lm_head", h * cfg["vocab"] * nbytes), ("sample", batch * cfg["vocab"] * 4)]
    return layer * cfg["layers"] + tail


def cell(text, width):
    """按显示宽度右对齐（汉字占两格）"""
    w = sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)
    return " " * (width - w) + text


print("CUDA Graph 下 decode 一步的时间（括号里是 kernel 边界的空隙所占的比例）")
print(cell("场景", 22) + cell("kernel 数", 10) + cell("读写数据", 10) + cell("逐个启动", 10)
      + "".join(cell(f"g = {g:g} µs", 16) for g in GAPS_US))
for name, batch, context in (("Qwen3-0.6B", 1, 1024), ("Qwen3-8B", 1, 1024), ("Qwen3-0.6B", 64, 4096)):
    ks = kernels(MODELS[name], batch, context)
    n = len(ks)
    t_data = sum(b for _, b in ks) / BW * 1e6                 # µs
    t_eager = max(n * CPU_LAUNCH_US, t_data + n * GAPS_US[0])  # when the CPU cannot issue fast enough, the GPU waits for it
    print(cell(f"{name[6:]} b={batch} ctx={context}", 22) + cell(str(n), 10) + cell(f"{t_data:.0f} µs", 10)
          + cell(f"{t_eager:.0f} µs", 10) + "".join(cell(f"{t_data + n * g:.0f} µs ({n * g / (t_data + n * g):.0%})", 16)
                                                    for g in GAPS_US))

ks = kernels(MODELS["Qwen3-0.6B"], 1, 1024)
small = sum(1 for _, b in ks if b / BW * 1e6 < 1.0)
print(f"Qwen3-0.6B b=1：{small} / {len(ks)} 个 kernel 读写数据的时间不到 1 µs，平均每个 {sum(b for _, b in ks) / BW * 1e6 / len(ks):.2f} µs")
```

```text title="output"
CUDA Graph 下 decode 一步的时间（括号里是 kernel 边界的空隙所占的比例）
                  场景 kernel 数  读写数据  逐个启动      g = 0.5 µs        g = 1 µs        g = 2 µs        g = 3 µs
     0.6B b=1 ctx=1024       283    461 µs   1415 µs    602 µs (24%)    744 µs (38%)   1027 µs (55%)   1310 µs (65%)
       8B b=1 ctx=1024       363   5371 µs   5552 µs    5552 µs (3%)    5734 µs (6%)   6097 µs (12%)   6460 µs (17%)
    0.6B b=64 ctx=4096       283  11023 µs  11164 µs   11164 µs (1%)   11306 µs (3%)   11589 µs (5%)   11872 µs (7%)
Qwen3-0.6B b=1：142 / 283 个 kernel 读写数据的时间不到 1 µs，平均每个 1.63 µs
```

- **0.6B at batch 1**: 283 kernels, half of them moving data for under 1 µs and averaging 1.63 µs, the same order as the gap at each boundary. At g = 2 µs the gaps are more than half the step: 461 µs of data movement in a step that takes 1 ms;
- **8B at batch 1**: a dozen or more microseconds per kernel, so the same gaps are only 3-17%;
- **batch 64 with a 4096 context**: reading the KV cache dominates and the gaps fall to single digits.

So these optimizations are worth most with **small models, small batches and low latency**: on-device inference, latency-sensitive online serving, and the draft model in speculative decoding (small, and run several steps in a row). The g in the table is an assumption; measure your own system with [Nsight Systems](profiling.md#nsight-systems): during a CUDA Graph replay, the blanks between kernels on the timeline are g.

## PDL: letting the next kernel start early <span class="arch">sm_90+</span> {#pdl让下一个-kernel-提前上场-sm_90}

![Figure: PDL, with the next kernel's prologue starting early and overlapping the previous kernel's tail](../assets/figures/pdl-overlap.svg){.aig-svg}

Hopper's **Programmatic Dependent Launch (PDL)** lets two neighbouring kernels in one stream overlap. It has three parts:

1. **a launch attribute**: the next kernel is launched with `cudaLaunchKernelEx` carrying `cudaLaunchAttributeProgrammaticStreamSerialization`, which says "I handle the dependency on the previous kernel myself, so I may start early";
2. **the previous kernel signals**: each block calls `cudaTriggerProgrammaticLaunchCompletion()` (the PTX instruction `griddepcontrol.launch_dependents`). Once every block has called it (or exited), the next kernel may begin. A block that never calls it counts as having triggered when it exits;
3. **the next kernel waits**: before its first read of the previous kernel's output it calls `cudaGridDependencySynchronize()` (PTX `griddepcontrol.wait`), which returns only once the previous kernel has fully finished and its writes are visible. The code before that wait is the "prologue" that can overlap the previous kernel.

The program below chains 200 "layers", each reading the previous layer's output and multiplying by its own weights. The prologue reads only the weights, then waits, then reads the previous layer's output; the trigger is signalled after the output is written. All four variants (one-by-one launches, with PDL, a CUDA Graph, a CUDA Graph with PDL) are checked against the CPU:

```cuda title="pdl_chain.cu"
// pdl_chain.cu - Programmatic Dependent Launch: overlapping the next kernel's prologue with the previous kernel's tail
// build: nvcc -O3 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 pdl_chain.cu -o pdl_chain
// the sm_90 machine code and the compute_90 PTX carry PDL (Hopper runs it directly, newer GPUs JIT it); older GPUs use the compute_75 PTX,
// without PDL and with the host attribute off, all four variants still check their results
#include "common.cuh"

// one "layer": out[i] = relu(in[i]) * w[i] + b[i]. w and b are this layer's weights and do not depend on the previous layer's output.
// in deliberately lacks __restrict__: with PDL on, the previous layer may still be writing it after this kernel starts, so the compiler must not use the read-only cache (ld.global.nc)
__global__ void layer(const float* in, float* out, const float* __restrict__ w, const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float wi = 0.f, bi = 0.f;
  if (i < n) {  // the prologue: weights only. With PDL on, this part can start before the previous layer ends
    wi = w[i];
    bi = b[i];
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();  // wait for every block of the previous layer to end and its writes to become visible before reading in
#endif
  if (i < n) out[i] = fmaxf(in[i], 0.f) * wi + bi;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();  // this block's output is written: the next layer may start early
#endif
}

// launched with cudaLaunchKernelEx; when pdl is true it carries the "may overlap the previous kernel" attribute
void launch_layer(bool pdl, cudaStream_t s, int n, const float* in, float* out, const float* w, const float* b) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((n + 255) / 256);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, layer, in, out, w, b, n));
}

int main() {
  const int n = 1 << 14, layers = 200, iters = 20;  // only 16K elements per layer: the kernels are short and the boundary cost dominates
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  const bool pdl_ok = p.major >= 9;
  if (!pdl_ok) std::printf("note: PDL needs sm_90+, this GPU is sm_%d%d, so every mode runs without it\n", p.major, p.minor);

  std::vector<float> h_x(n), h_w(static_cast<size_t>(layers) * n), h_b(static_cast<size_t>(layers) * n);
  fill_random(h_x, 1);
  fill_random(h_w, 2, 0.5f, 1.5f);
  fill_random(h_b, 3, -0.1f, 0.1f);
  std::vector<float> ref = h_x, got(n);
  for (int l = 0; l < layers; ++l)
    for (int i = 0; i < n; ++i)
      ref[i] = std::fmax(ref[i], 0.f) * h_w[static_cast<size_t>(l) * n + i] + h_b[static_cast<size_t>(l) * n + i];

  const size_t bytes = n * sizeof(float);
  float *d_x, *d_a, *d_c, *d_w, *d_b;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // one "forward pass": `layers` kernels chained end to end, alternating between d_a and d_c
  auto run_chain = [&](bool pdl) {
    const float* in = d_x;
    for (int l = 0; l < layers; ++l) {
      float* out = l % 2 == 0 ? d_a : d_c;
      launch_layer(pdl && pdl_ok, s, n, in, out, d_w + static_cast<size_t>(l) * n, d_b + static_cast<size_t>(l) * n);
      in = out;
    }
  };
  float* result = (layers - 1) % 2 == 0 ? d_a : d_c;

  const char* names[4] = {"stream", "stream + PDL", "graph", "graph + PDL"};
  bool all_ok = true;
  for (int mode = 0; mode < 4; ++mode) {
    const bool graph = mode >= 2, pdl = mode % 2 == 1;
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    cudaGraphExec_t exec = nullptr;
    if (graph) {  // stream capture: a launch with the PDL attribute becomes a programmatic edge in the graph
      cudaGraph_t g;
      CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      run_chain(pdl);
      CUDA_CHECK(cudaStreamEndCapture(s, &g));
      CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
    }
    auto forward = [&] {
      if (graph) CUDA_CHECK(cudaGraphLaunch(exec, s));
      else run_chain(pdl);
    };
    forward();  // warm-up
    GpuTimer t;
    t.start(s);
    for (int it = 0; it < iters; ++it) forward();
    const float us = t.stop(s) * 1e3f / iters;
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), result, bytes, cudaMemcpyDeviceToHost));
    std::printf("%-13s %8.1f us per forward, %6.2f us per layer   ", names[mode], us, us / layers);
    all_ok &= check_close(got.data(), ref.data(), n, 1e-4f, 1e-5f);
    if (exec) CUDA_CHECK(cudaGraphExecDestroy(exec));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  for (float* ptr : {d_x, d_a, d_c, d_w, d_b}) CUDA_CHECK(cudaFree(ptr));
  return all_ok ? 0 : 1;
}
```

Run it on Hopper or newer and compare the four lines' per-layer times; an older GPU prints a note, runs all four without PDL, and only checks the results. This book's CPU simulator runs the kernels one after another, so it can verify this program's logic (all four PASS) but cannot show the time the overlap saves.

A few things to watch when writing PDL code:

- **Do not touch the previous kernel's data before the wait**: do not read its output, and do not write a buffer it may still be reading. The program above alternates between two buffers, and the buffer layer k writes is exactly the input layer k−1 reads, so even the write has to come after the wait. Put every access to data shared with another kernel after `cudaGridDependencySynchronize()`, and let the prologue touch only read-only data such as weights;
- **Do not route the previous layer's output through the read-only cache**: `const float* __restrict__` makes the compiler use `ld.global.nc`, which requires the data to be unmodified for the kernel's whole lifetime, while under PDL the previous layer may still be writing when this kernel starts. The program above adds `__restrict__` only to the weights, and in the generated PTX the weights use `ld.global.nc` while the previous layer's output uses a plain `ld.global`;
- **Where to trigger**: the earlier the trigger, the sooner the next kernel starts, but an early block holds the SM's registers, shared memory and a block slot while it waits, which can slow the previous kernel's remaining waves. The common choice is to trigger after writing the output (as above), overlapping the tail with the next kernel's scheduling and prologue;
- **Compatibility with older GPUs**: wrap the two device-side calls in `#if __CUDA_ARCH__ >= 900`, and turn the attribute on only on sm_90 and above on the host side. The converse matters too: a GPU with the attribute on must really be running code that waits (compiled with sm_90 machine code or compute_90 PTX, which the build command at the top of the program does), or the next kernel starts early without waiting, which is a data race;
- **It works inside a CUDA Graph**: during stream capture, a launch with the PDL attribute is recorded as a programmatic edge; building a graph by hand, an edge can be marked `cudaGraphDependencyTypeProgrammatic` (with the matching `cudaGraphKernelNodePortProgrammatic` port). Inference engines replay their decode graphs with PDL exactly this way;
- **In Triton**: the equivalents are `tl.extra.cuda.gdc_wait()` and `tl.extra.cuda.gdc_launch_dependents()`, with PDL turned on at launch likewise.

## PDL in inference engines {#推理引擎里的-pdl}

vLLM and SGLang both use PDL extensively, and both only when the kernels are short:

- **vLLM**: `is_arch_support_pdl()` in `vllm/platforms/cuda.py` returns true at compute capability 9.0 and above, and the `pdl` / `launch_pdl` arguments of the various kernels follow it. `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` launches the fused QK-Norm plus RoPE with `cudaLaunchKernelEx` and `cudaLaunchAttributeProgrammaticStreamSerialization`, but only when the token count is at most `kPDLEnableTokens` (16); the comment says it plainly: overlapping with the preceding QKV projection only pays when this kernel is dominated by launch overhead. LoRA's Triton kernels use `gdc_wait()` / `gdc_launch_dependents()`, and `VLLM_LORA_DISABLE_PDL` turns it off when something goes wrong;
- **SGLang**: it sets `TRTLLM_ENABLE_PDL=1` at startup by default, turning PDL on for the TensorRT-LLM-derived MoE and quantization kernels inside FlashInfer (`srt/entrypoints/engine.py`); `SGLANG_DEEPGEMM_PDL` (on by default) turns it on for DeepGEMM (`deep_gemm.set_pdl(True)` in `srt/layers/deep_gemm_wrapper/entrypoint.py`); and `SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS` (8192 by default) sets the token count above which the MoE's grouped GEMM stops using PDL.

When PDL goes wrong it usually does not crash; it produces occasional wrong answers. vLLM's fused all-reduce plus RMSNorm (`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`) has a real example: if a one-shot Lamport all-reduce signals PDL completion before the output buffer is really written, the next PDL-launched kernel can read an uninitialized buffer and produce NaN, and only at shapes with 16 tokens or fewer (batch 1, speculative decoding). The fix was to signal at the end of the kernel on that path (`trigger_completion_at_end`). The lesson: **"the next kernel may start" and "my output is ready" are two different things**, so be careful where you trigger whenever data is synchronized by something other than the kernel's end, such as cross-GPU writes or flags.

## Fewer kernels: fusion {#减少-kernel-的个数融合}

The total gap is "gap per boundary × number of boundaries". PDL shrinks the former and fusion the latter:

- **fuse elementwise kernels into their neighbours**: [fused_add_rms_norm](../kernels/softmax-norm.md#算子融合fused_add_rms_norm) merges the residual add with RMSNorm, vLLM merges QK-Norm with RoPE, and [torch.compile's Inductor](../framework/compile.md#inductor-做了什么融合) fuses neighbouring elementwise operations automatically;
- **fuse communication with compute**: vLLM's compilation pass replaces the tensor-parallel all-reduce and the RMSNorm after it with one FlashInfer kernel; FlashInfer's MegaMoE puts expert-parallel dispatch, the expert GEMM and the combine into one kernel (communicating through symmetric memory; see the Inference Systems handbook on [the newest open models](serving://frontier/new-models/)), with SGLang's wrapper in `srt/layers/moe/flashinfer_megamoe.py`.

Fusion has a bonus: the intermediate results never go back to global memory.

## Megakernels: the whole forward pass in one kernel {#megakernel把整个前向写进一个-kernel}

Push fusion to its limit and you get a **megakernel**: the whole forward pass launches one resident (persistent) kernel, one block per SM, which claims tasks in turn from a precomputed "instruction table" (a GEMV tile, one attention head, a stretch of RMSNorm) with the dependencies between them expressed as counters in global memory. Roughly (in sketch form):

```cuda
__global__ void megakernel(const Instr* prog, int num_instr, int* done) {
  for (int k = blockIdx.x; k < num_instr; k += gridDim.x) {   // each block claims instructions by index
    const Instr& ins = prog[k];
    prefetch_weights(ins);                                     // the weights do not depend on earlier instructions: start moving them into shared memory first
    for (int d = 0; d < ins.num_deps; ++d)                     // wait for the instructions depended on (reading the counter with acquire semantics)
      while (load_acquire(&done[ins.dep[d]]) < ins.dep_count[d]) {}
    execute(ins);                                              // compute this tile
    __syncthreads();
    if (threadIdx.x == 0) release_add(&done[ins.id], 1);       // publish completion (incrementing the counter with release semantics)
  }
}
```

What it gets:

- **no kernel boundaries**: an SM that finishes its task starts the next as soon as its dependencies are met, with no global synchronization;
- **finer dependencies**: one attention head depends only on the few tiles of the QKV projection it needs, rather than on the whole projection;
- **prefetching across kernels**: while waiting on a dependency it can already pull the next task's weights into shared memory, which is PDL's "prologue" idea taken to the extreme.

The costs are equally clear:

- **resources sized for the largest task**: every task shares one kernel's register and shared-memory configuration, so a small task still occupies a large task's resources;
- **synchronization and memory ordering are yours**: counter reads and writes need acquire / release semantics (`cuda::atomic_ref`, or PTX's `ld.acquire.gpu` and `red.release.gpu`), or stale data gets read. Deadlocks are yours too: every block must be resident at once (the grid no larger than the number of blocks that fit), and a dependency may only point to a lower-numbered instruction, so the lowest-numbered unfinished instruction can always run;
- **it generalizes badly**: the instruction table is generated for one model, batch size and parallelism scheme, and a change means regenerating it; tensor parallelism also needs communication inside the kernel (one-sided communication such as NVSHMEM).

Published work includes Hazy Research's hand-written low-latency megakernel for Llama-1B, and Mirage Persistent Kernel (MPK), which compiles a model into a megakernel automatically; both target small-batch decode. What mainstream inference engines use today is a "partial megakernel": a large kernel like MegaMoE fusing a whole layer or a stretch of communication and compute, on top of PDL and CUDA Graphs.

!!! interview "How to explain it"
    To explain "what is still slow about decode after CUDA Graphs": CUDA Graphs only remove the CPU's launch cost; on the GPU each kernel boundary still has a global synchronization, scheduling, the tail effect and a prologue, and with a small model and a small batch each kernel is only a microsecond or two, so the gaps can be half the step. Three ways out: PDL (sm_90+, where the next kernel launched with `ProgrammaticStreamSerialization` may start once the previous kernel's blocks called `cudaTriggerProgrammaticLaunchCompletion`, doing its prologue before waiting with `cudaGridDependencySynchronize`, and recordable in a CUDA Graph); fusion (fewer kernels, including communication-compute fusions like all-reduce plus RMSNorm and MegaMoE); and megakernels (a resident kernel with an instruction table and counter dependencies, no boundaries and prefetching across kernels, at the price of resources, synchronization and generality). Mentioning that PDL is only turned on at low token counts, and the traps around where to trigger and the read-only cache, counts for extra.

## Exercises {#练习}

**1. What fusion saves.** With this chapter's model, for Qwen3-0.6B at batch 1 and g = 2 µs, how much faster is a step if each layer's 10 kernels become 6?

??? success "Answer"
    The kernel count goes from 28 × 10 + 3 = 283 to 28 × 6 + 3 = 171, so the gaps fall from 283 × 2 = 566 µs to 171 × 2 = 342 µs. The data time barely changes (what was fused away were tiny intermediates), about 461 µs, so a step goes from 1027 µs to about 803 µs, roughly 22% faster. Squeeze each boundary to 1 µs with PDL as well and a step is about 461 + 171 = 632 µs.

**2. An in-place update.** A layer's kernel modifies a buffer in place: it reads `x` and writes the result back to `x`, and the kernel before it also reads `x`. Launching this layer with PDL, may the read and the write go before `cudaGridDependencySynchronize()`?

??? success "Answer"
    The write certainly may not: the previous kernel may still be reading `x`, and writing early destroys its input (a write-after-read hazard). The read should not either: `x` was written by some earlier kernel, and if the previous kernel triggered before its own wait, that earlier kernel may not have finished and the read would see a stale value; the read is safe only once you know every kernel in the chain triggers after its wait. The safest rule is to put every access to data another kernel may read or write after the wait, and let the prologue touch only read-only data such as this layer's weights.

**3. A megakernel deadlock.** In the sketch above, what happens if the grid has more blocks than can be resident on the GPU at once? What if an instruction depends on a higher-numbered one?

??? success "Answer"
    With more blocks than fit, some can only start once others exit, while the resident blocks may be spinning on instructions owned by those blocks that have not started, so they never exit: a deadlock. With a dependency on a higher-numbered instruction, a block may wait on an instruction queued behind itself that it will only execute later, which deadlocks too. With both conditions satisfied, the lowest-numbered unfinished instruction always has its dependencies met and progress is guaranteed.

## Summary {#小结}

- [x] CUDA Graphs only remove the CPU's launch cost; on the GPU each kernel boundary still has a global synchronization, scheduling, the tail effect and a prologue.
- [x] With a small model and a small batch each kernel is a microsecond or two and the gaps can be half a decode step; with a large model and a large batch they are a small fraction.
- [x] PDL (sm_90+): once the previous kernel triggers, the next starts early and does its prologue, touching the previous kernel's data only after `cudaGridDependencySynchronize()`; keep the previous layer's output out of the read-only cache; it records into a CUDA Graph.
- [x] Inference engines turn PDL on only at low token counts; "the next kernel may start" is not "the output is ready".
- [x] Fusion cuts the number of boundaries, including fusions of communication and compute (all-reduce plus RMSNorm, MegaMoE).
- [x] Megakernels: a resident kernel with an instruction table and counter dependencies, no boundaries and prefetching across kernels, at the price of resources, memory ordering, deadlocks and generality.
