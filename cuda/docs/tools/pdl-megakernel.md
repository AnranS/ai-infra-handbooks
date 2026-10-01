# kernel 之间的空隙：PDL 与 megakernel

<p class="lead">CUDA Graph 消掉了 CPU 逐个发射 kernel 的开销，但 decode 一步的几百个 kernel 之间，GPU 上仍然有空隙：后一个 kernel 要等前一个彻底结束才能开始，开始后还要先做一段与数据无关的准备。模型越小、batch 越小，每个 kernel 越短，这些空隙占的比例就越大。这一章先用一个估算模型算清楚空隙占多少，再讲三种办法：Hopper 起的 PDL（Programmatic Dependent Launch）让相邻 kernel 重叠，算子融合减少 kernel 的个数，megakernel 把整个前向写进一个常驻的 kernel。最后看 vLLM 和 SGLang 实际怎么用。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 用了 CUDA Graph 之后，相邻两个 kernel 之间还剩下哪些开销？
    2. PDL 里 `cudaTriggerProgrammaticLaunchCompletion()` 和 `cudaGridDependencySynchronize()` 各做什么？后一个 kernel 在等待之前能做什么、不能做什么？
    3. 为什么推理引擎通常只在 token 数很少时才打开 PDL？
    4. megakernel 靠什么消掉 kernel 边界？它的代价是什么？
    5. 为什么 0.6B 模型 batch 1 的 decode 比 8B 模型更需要这些优化？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 同一个流里，后一个 kernel 要等前一个 kernel 的所有 block 都结束、写入对后续可见，GPU 前端才开始调度它、把 block 分发到 SM；每个 block 开头还有一段与数据无关的准备（算下标、读权重、初始化共享内存）；前一个 kernel 最后一波 block 往往填不满所有 SM（尾部效应）。这些都发生在 GPU 上，CUDA Graph 消除不了。
    2. 前一个 kernel 里，所有 block 都调用了 `cudaTriggerProgrammaticLaunchCompletion()`（或已经退出）之后，带 PDL 属性启动的后一个 kernel 就可以开始执行；后一个 kernel 里，`cudaGridDependencySynchronize()` 一直等到前一个 kernel 全部完成、写入可见才返回。等待之前只能做与前一个 kernel 无关的事：读权重、算下标、初始化共享内存；不能读前一个 kernel 的输出，也不能写前一个 kernel 可能还在读的数据。
    3. 提前启动的 block 在等待期间也占着 SM 的寄存器、共享内存和 block 槽位。token 多时每个 kernel 本身就很长、有好几波 block，空隙占比很小，提前上场的 block 反而和前一个 kernel 抢资源；token 少时 kernel 短、受启动开销主导，重叠才划算。vLLM 的几个融合 kernel 只在 token 数不超过 16 时打开 PDL。
    4. 只启动一个常驻（persistent）kernel，每个 SM 上一个 block，按一张指令表依次领取任务（一块 GEMV、一个头的注意力……），任务之间用全局内存里的计数器表达依赖，一个任务的依赖满足就立刻开始，还可以在等待时预取下一个任务的权重。代价：所有任务共用一个 kernel 的资源配置（寄存器、共享内存按最大的任务分配），调度和同步全靠手写，要自己处理内存序和死锁，很难推广到任意模型、batch 大小和并行方式。
    5. 0.6B 模型 batch 1 时每个 kernel 读写的数据很少，一半的 kernel 连 1 µs 都不到，和每个 kernel 边界上一两微秒的空隙是同一个量级；8B 模型每个 kernel 平均读写十几微秒的数据，空隙只占一小部分。batch 和上下文变大时也一样：数据量变大，空隙的占比下降。

## 空隙从哪里来

推理系统手册的 [CUDA Graphs 与 torch.compile](serving://engine/graphs-compile/) 一章算过：不用 CUDA Graph 时，decode 一步上千个 kernel、每个几微秒的 CPU 发射开销，让 GPU 大部分时间在等 CPU。[CUDA Graph](streams.md#cuda-graphs) 把整步录成一张图、一次提交，CPU 这一侧的问题就解决了。

但 GPU 这一侧，同一个流里的两个 kernel 之间还有一段空白：

```text
没有 PDL：
  kernel A  [==== 主体 ====][尾部：最后一波 block，SM 没占满]
  kernel B                                               |调度| [前奏：读权重、算下标] [==== 主体 ====]
                                                         ↑ A 的所有 block 结束、写入可见之后才开始

打开 PDL：
  kernel A  [==== 主体 ====][尾部]  ← 每个 block 写完输出后调用 trigger
  kernel B            |调度| [前奏] [等待 A 完成]  [==== 主体 ====]
                                    ↑ cudaGridDependencySynchronize()
```

- **kernel 边界是一次全局同步**：后一个 kernel 的 block 要等前一个 kernel 的所有 block 都结束、写入对后续可见之后才开始，然后 GPU 前端调度新的 kernel、把 block 分发到 SM；
- **尾部效应**：前一个 kernel 的最后一波 block 往往填不满所有 SM，这段时间大部分 SM 空着；
- **前奏**：每个 block 开头都有一段不依赖上一步结果的准备工作：算下标、从全局内存读这一层的权重、初始化共享内存。

这些都发生在 GPU 上，CUDA Graph 无能为力。要去掉它们，只能让相邻的 kernel 重叠（PDL）、减少 kernel 的个数（融合），或者干脆不要 kernel 边界（megakernel）。

## 空隙占多少：一个估算

decode 的每个 kernel 都受访存限制，时间约等于读写的字节数除以带宽。下面把 Qwen3 的 decode 一步拆成 kernel（融合程度参照 vLLM：残差加 + RMSNorm、QK-Norm + RoPE 各算一个 kernel），算出读写数据的时间，再给每个 kernel 边界加上 g 微秒的空隙。g 在不同 GPU、不同 kernel 上并不相同，所以取几档看敏感度：

```python title="decode_gaps.py"
"""decode_gaps.py —— 把 decode 一步的时间拆成"读写数据"和"kernel 边界的空隙"，看空隙占多少。

假设：H100 的带宽 3.35 TB/s，kernel 能跑到其中的 85%；decode 的每个 kernel 都受访存限制，时间 = 读写字节数 / 有效带宽。
kernel 之间的空隙 g 取 0.5～3 µs 几档看敏感度；逐个启动（不用 CUDA Graph）时，CPU 发射每个 kernel 算 5 µs。
"""

BW = 3.35e12 * 0.85                  # 有效带宽，字节/秒
CPU_LAUNCH_US = 5.0
GAPS_US = (0.5, 1.0, 2.0, 3.0)
MODELS = {  # Qwen3 的配置（hidden、intermediate、层数、Q 头数、KV 头数、head_dim、词表）
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
    t_eager = max(n * CPU_LAUNCH_US, t_data + n * GAPS_US[0])  # CPU 发射跟不上时，GPU 等 CPU
    print(cell(f"{name[6:]} b={batch} ctx={context}", 22) + cell(str(n), 10) + cell(f"{t_data:.0f} µs", 10)
          + cell(f"{t_eager:.0f} µs", 10) + "".join(cell(f"{t_data + n * g:.0f} µs ({n * g / (t_data + n * g):.0%})", 16)
                                                    for g in GAPS_US))

ks = kernels(MODELS["Qwen3-0.6B"], 1, 1024)
small = sum(1 for _, b in ks if b / BW * 1e6 < 1.0)
print(f"Qwen3-0.6B b=1：{small} / {len(ks)} 个 kernel 读写数据的时间不到 1 µs，平均每个 {sum(b for _, b in ks) / BW * 1e6 / len(ks):.2f} µs")
```

```text title="输出"
CUDA Graph 下 decode 一步的时间（括号里是 kernel 边界的空隙所占的比例）
                  场景 kernel 数  读写数据  逐个启动      g = 0.5 µs        g = 1 µs        g = 2 µs        g = 3 µs
     0.6B b=1 ctx=1024       283    461 µs   1415 µs    602 µs (24%)    744 µs (38%)   1027 µs (55%)   1310 µs (65%)
       8B b=1 ctx=1024       363   5371 µs   5552 µs    5552 µs (3%)    5734 µs (6%)   6097 µs (12%)   6460 µs (17%)
    0.6B b=64 ctx=4096       283  11023 µs  11164 µs   11164 µs (1%)   11306 µs (3%)   11589 µs (5%)   11872 µs (7%)
Qwen3-0.6B b=1：142 / 283 个 kernel 读写数据的时间不到 1 µs，平均每个 1.63 µs
```

- **0.6B、batch 1**：283 个 kernel，一半读写数据的时间不到 1 µs，平均 1.63 µs，和每个边界的空隙是同一个量级。g = 2 µs 时空隙占了一步的一半以上：读写数据只要 461 µs，一步却要 1 ms；
- **8B、batch 1**：每个 kernel 平均十几微秒，同样的空隙只占 3%～17%；
- **batch 64、上下文 4096**：读 KV Cache 的时间占了大头，空隙的占比降到个位数。

所以这些优化最有价值的场景是**小模型、小 batch、低延迟**：端侧推理、对延迟敏感的在线服务、投机解码里的草稿模型（它小，而且每轮要串行跑好几步）。表里的 g 是假设，自己的系统要用 [Nsight Systems](profiling.md#nsight-systems) 看：CUDA Graph 重放时，时间线上 kernel 与 kernel 之间的空白就是 g。

## PDL：让下一个 kernel 提前上场 <span class="arch">sm_90+</span>

![图：PDL——下一个 kernel 的序言提前上场，和上一个 kernel 的尾巴重叠](../assets/figures/pdl-overlap.svg){.aig-svg}

Hopper 引入的 **Programmatic Dependent Launch（PDL，程序化依赖启动）**允许同一个流里相邻的两个 kernel 重叠。它由三部分组成：

1. **启动属性**：后一个 kernel 用 `cudaLaunchKernelEx` 启动，带上 `cudaLaunchAttributeProgrammaticStreamSerialization`，表示"我自己处理对前一个 kernel 的依赖，可以提前启动"；
2. **前一个 kernel 发信号**：每个 block 调用 `cudaTriggerProgrammaticLaunchCompletion()`（PTX 指令 `griddepcontrol.launch_dependents`）。所有 block 都调用了（或者已经退出）之后，后一个 kernel 就可以开始执行。没调用的 block 在退出时自动算作已触发；
3. **后一个 kernel 等依赖**：在第一次读前一个 kernel 的输出之前调用 `cudaGridDependencySynchronize()`（PTX `griddepcontrol.wait`），它一直等到前一个 kernel 全部完成、写入对本 kernel 可见才返回。等待之前的代码就是可以和前一个 kernel 重叠的"前奏"。

下面的程序把 200 个"层"首尾相连，每层读上一层的输出、乘上这一层的权重。前奏只读权重，然后等待，再读上一层的输出；写完输出后发出触发信号。四种方式（逐个启动、加 PDL、CUDA Graph、CUDA Graph 加 PDL）都和 CPU 结果比对：

```cuda title="pdl_chain.cu"
// pdl_chain.cu —— Programmatic Dependent Launch：让下一个 kernel 的前奏和上一个 kernel 的收尾重叠
// 编译：nvcc -O3 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 pdl_chain.cu -o pdl_chain
// sm_90 的机器码和 compute_90 的 PTX 带 PDL（Hopper 直接运行，更新的 GPU 即时编译）；更老的 GPU 用 compute_75 的 PTX，
// 不带 PDL，主机端也不打开属性，四种方式照样核对结果
#include "common.cuh"

// 一"层"：out[i] = relu(in[i]) * w[i] + b[i]。w、b 是这一层的权重，不依赖上一层的输出。
// in 故意不加 __restrict__：打开 PDL 时本 kernel 启动后上一层可能还在写它，不能让编译器走只读缓存（ld.global.nc）
__global__ void layer(const float* in, float* out, const float* __restrict__ w, const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float wi = 0.f, bi = 0.f;
  if (i < n) {  // 前奏：只读权重。打开 PDL 时，这一段可以在上一层还没结束时就开始
    wi = w[i];
    bi = b[i];
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();  // 等上一层的所有 block 结束、写入对本 kernel 可见，之后才能读 in
#endif
  if (i < n) out[i] = fmaxf(in[i], 0.f) * wi + bi;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();  // 本 block 的输出写完了：下一层可以提前启动
#endif
}

// 用 cudaLaunchKernelEx 启动；pdl 为真时带上"允许和前一个 kernel 重叠"的属性
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
  const int n = 1 << 14, layers = 200, iters = 20;  // 每层只有 16K 个元素：kernel 很短，边界开销占大头
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

  // 一次"前向"：layers 个 kernel 首尾相连，在 d_a、d_c 之间来回写
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
    if (graph) {  // 流捕获：带 PDL 属性的启动在图里变成 programmatic 类型的边
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
    forward();  // 预热
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

在 Hopper 或更新的 GPU 上运行，比较四行的每层耗时；更老的 GPU 会打印一行说明，四种方式都不开 PDL，只核对结果。本书的 CPU 模拟器把 kernel 一个接一个地执行，能核对这个程序的逻辑（四种方式都 PASS），但体现不出重叠带来的时间差。

写 PDL 代码要注意这几点：

- **等待之前不能碰上一个 kernel 的数据**：不能读它的输出，也不能写它可能还在读的缓冲区。上面的程序在两个缓冲区之间来回写，第 k 层要写的缓冲区正是第 k−1 层在读的输入，所以写也必须放在等待之后。凡是和别的 kernel 共享的数据，访问都放在 `cudaGridDependencySynchronize()` 之后，前奏只碰权重这类只读数据；
- **上一层的输出不要走只读缓存**：`const float* __restrict__` 会让编译器用 `ld.global.nc` 读取，它要求数据在 kernel 的整个生命周期内不被修改，而 PDL 下本 kernel 启动时上一层可能还在写。上面的程序只给权重加了 `__restrict__`，编译出的 PTX 里权重用 `ld.global.nc`、上一层的输出用普通的 `ld.global`；
- **触发放在哪**：触发得越早，后一个 kernel 越早上场，但提前上场的 block 在等待期间也占着 SM 的寄存器、共享内存和 block 槽位，可能拖慢前一个 kernel 还没跑完的几波 block。常见做法是在写完输出之后触发（和上面的程序一样），重叠掉尾部和后一个 kernel 的调度、前奏；
- **老 GPU 上的兼容**：设备代码用 `#if __CUDA_ARCH__ >= 900` 包住这两个调用，主机端只在 sm_90 及以上的 GPU 上打开属性。反过来也要保证：打开属性的 GPU 上实际运行的必须是带等待的代码（编译时带上 sm_90 的机器码或 compute_90 的 PTX，程序开头的编译命令就是这样做的），否则后一个 kernel 提前上场却不等待，就是数据竞争；
- **CUDA Graph 里也能用**：流捕获时，带 PDL 属性的启动会被记录成 programmatic 类型的边；手工建图时，也可以给边标上 `cudaGraphDependencyTypeProgrammatic`（配合 `cudaGraphKernelNodePortProgrammatic` 等出口）。推理引擎的 decode 图就是这样带着 PDL 重放的；
- **Triton 里**：对应的函数是 `tl.extra.cuda.gdc_wait()` 和 `tl.extra.cuda.gdc_launch_dependents()`，启动时同样要打开 PDL。

## 推理引擎里的 PDL

vLLM 和 SGLang 都已经大量使用 PDL，而且都只在 kernel 短的时候打开：

- **vLLM**：`vllm/platforms/cuda.py` 的 `is_arch_support_pdl()` 在计算能力 9.0 及以上时返回真，各个 kernel 的 `pdl` / `launch_pdl` 参数都由它决定。`csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` 用 `cudaLaunchKernelEx` 加 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动融合的 QK-Norm + RoPE，但只在 token 数不超过 `kPDLEnableTokens`（16）时打开——注释写得很直白：和前面的 QKV 投影重叠，只在这个 kernel 受启动开销主导时才划算。LoRA 的 Triton kernel 用 `gdc_wait()` / `gdc_launch_dependents()`，出问题时可以用 `VLLM_LORA_DISABLE_PDL` 关掉；
- **SGLang**：启动时默认设置 `TRTLLM_ENABLE_PDL=1`，让 FlashInfer 里来自 TensorRT-LLM 的 MoE、量化等 kernel 打开 PDL（`srt/entrypoints/engine.py`）；`SGLANG_DEEPGEMM_PDL`（默认开）让 DeepGEMM 打开 PDL（`srt/layers/deep_gemm_wrapper/entrypoint.py` 里调用 `deep_gemm.set_pdl(True)`）；`SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS`（默认 8192）规定 token 数超过多少时 MoE 的分组 GEMM 不再用 PDL。

PDL 出错时往往不是崩溃，而是偶发的错误结果。vLLM 的 all-reduce + RMSNorm 融合（`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`）里有一个真实的例子：一次性（one-shot）的 Lamport all-reduce 如果在输出缓冲区真正写好之前就发出 PDL 的完成信号，下一个用 PDL 启动的 kernel 就可能读到没初始化的缓冲区、算出 NaN，而且只在 token 数不超过 16（batch 1、投机解码）的形状上出现。修复办法是这条路径改在 kernel 末尾才发信号（`trigger_completion_at_end`）。教训是：**"可以启动下一个 kernel"和"我的输出已经可用"是两回事**，涉及跨 GPU 写入、标志位这类不经过 kernel 结束来同步的数据时，要格外小心触发的位置。

## 减少 kernel 的个数：融合

空隙的总量是"每个边界的空隙 × 边界的个数"。PDL 缩小前者，融合减少后者：

- **逐元素算子融进相邻的 kernel**：[fused_add_rms_norm](../kernels/softmax-norm.md#算子融合fused_add_rms_norm) 把残差加和 RMSNorm 合成一个 kernel，vLLM 把 QK-Norm 和 RoPE 合成一个 kernel，[torch.compile 的 Inductor](../framework/compile.md#inductor-做了什么融合) 自动融合相邻的逐元素算子；
- **把通信和计算融在一起**：vLLM 的编译 pass 把张量并行的 all-reduce 和其后的 RMSNorm 换成 FlashInfer 的一个融合 kernel；FlashInfer 的 MegaMoE 把专家并行的分发、专家 GEMM 和合并做进同一个 kernel（通信走对称内存，见推理系统手册的[新一代开源模型](serving://frontier/new-models/)），SGLang 的封装在 `srt/layers/moe/flashinfer_megamoe.py`。

融合还有一个额外的好处：中间结果不用写回全局内存再读出来。

## megakernel：把整个前向写进一个 kernel

把融合推到极致，就是 **megakernel**：整个前向只启动一个常驻（persistent）kernel，每个 SM 上一个 block，按一张事先排好的"指令表"依次领取任务（一块 GEMV、一个注意力头、一段 RMSNorm……），任务之间的依赖用全局内存里的计数器表达。结构大致如下（示意代码）：

```cuda
__global__ void megakernel(const Instr* prog, int num_instr, int* done) {
  for (int k = blockIdx.x; k < num_instr; k += gridDim.x) {   // 每个 block 按编号领取指令
    const Instr& ins = prog[k];
    prefetch_weights(ins);                                     // 权重不依赖前面的指令：先开始搬进共享内存
    for (int d = 0; d < ins.num_deps; ++d)                     // 等依赖的指令完成（acquire 语义地读计数器）
      while (load_acquire(&done[ins.dep[d]]) < ins.dep_count[d]) {}
    execute(ins);                                              // 算这一块
    __syncthreads();
    if (threadIdx.x == 0) release_add(&done[ins.id], 1);       // 发布完成（release 语义地加计数器）
  }
}
```

它得到的是：

- **没有 kernel 边界**：一个 SM 做完手上的任务，依赖一满足就开始下一个，不用等全局同步；
- **更细的依赖**：比如某个注意力头只依赖 QKV 投影里对应的那几块，不必等整个投影完成；
- **跨算子的预取**：等依赖时就开始把下一个任务的权重搬进共享内存，这是 PDL"前奏"思路的极致。

代价同样明显：

- **资源按最大的任务分配**：所有任务共用一个 kernel 的寄存器和共享内存配置，小任务也要占着大任务的资源；
- **同步和内存序全靠自己**：计数器的读写要有 acquire / release 语义（`cuda::atomic_ref`，或 PTX 的 `ld.acquire.gpu`、`red.release.gpu`），否则可能读到旧数据。死锁也要自己避免：所有 block 必须同时驻留（网格大小不超过能同时驻留的 block 数），依赖只能指向编号更小的指令——这样编号最小的未完成指令总能执行；
- **很难通用**：指令表针对具体的模型、batch 大小和并行方式生成，换一个就要重新生成；张量并行还需要在 kernel 内部通信（NVSHMEM 这类单边通信）。

公开的工作有 Hazy Research 为 Llama-1B 手写的低延迟 megakernel，以及把模型自动编译成 megakernel 的 Mirage Persistent Kernel（MPK），它们都针对小 batch 的 decode。主流推理引擎目前采用的是"局部的 megakernel"：MegaMoE 这类把一整层或一段通信计算融在一起的大 kernel，再加上 PDL 和 CUDA Graph。

!!! interview "面试怎么答"
    被问"CUDA Graph 之后 decode 还慢在哪"：CUDA Graph 只消掉 CPU 的发射开销；GPU 上每个 kernel 边界还有全局同步、调度、尾部效应和前奏，小模型、小 batch 时每个 kernel 只有一两微秒，这些空隙能占一步的一半。三种办法：PDL（sm_90+，前一个 kernel 的 block 调用 `cudaTriggerProgrammaticLaunchCompletion` 后，带 `ProgrammaticStreamSerialization` 属性启动的后一个 kernel 就能上场，做完前奏再用 `cudaGridDependencySynchronize` 等依赖，可以录进 CUDA Graph）；融合（减少 kernel 个数，包括 all-reduce + RMSNorm、MegaMoE 这类通信计算融合）；megakernel（常驻 kernel + 指令表 + 计数器依赖，没有边界、能跨算子预取，代价是资源、同步、通用性）。能说出 PDL 只在 token 少时打开、触发位置和只读缓存的坑，会加分。

## 练习

**1. 融合省多少。** 用本章的估算模型，Qwen3-0.6B batch 1、g = 2 µs 时，如果把每层的 10 个 kernel 融合成 6 个，一步能快多少？

??? success "参考答案"
    kernel 数从 28 × 10 + 3 = 283 变成 28 × 6 + 3 = 171，空隙从 283 × 2 = 566 µs 降到 171 × 2 = 342 µs。读写数据的时间基本不变（融掉的只是很小的中间结果），约 461 µs，所以一步从 1027 µs 降到约 803 µs，快了约 22%。如果再用 PDL 把每个边界的空隙压到 1 µs，一步约 461 + 171 = 632 µs。

**2. 原地更新。** 某一层的 kernel 原地修改一个缓冲区：先读 `x`，再把结果写回 `x`，它的上一个 kernel 也读 `x`。用 PDL 启动这一层时，对 `x` 的读和写分别能不能放在 `cudaGridDependencySynchronize()` 之前？

??? success "参考答案"
    写肯定不能：上一个 kernel 可能还在读 `x`，提前写会破坏它的输入（读后写冲突）。读也不应该：`x` 是更早的某个 kernel 写的，如果上一个 kernel 在它自己等待依赖之前就发出了触发信号，那个更早的 kernel 可能还没结束，读到的就是旧值；只有确认链上每个 kernel 都是在等待之后才触发，这个读才安全。最稳妥的做法是：凡是可能被别的 kernel 读写的数据，访问都放在等待之后，前奏只碰这一层的权重这类只读数据。

**3. megakernel 的死锁。** 上面的示意代码里，如果网格里的 block 数超过了能同时驻留在 GPU 上的 block 数，会发生什么？如果某条指令依赖一条编号更大的指令呢？

??? success "参考答案"
    block 数超过驻留上限时，一部分 block 要等别的 block 退出才能开始，而驻留的 block 可能正在自旋等待这些还没开始的 block 负责的指令，它们永远不会退出——死锁。依赖指向编号更大的指令时，同一个 block 可能在等一条排在自己后面、将来才会由自己执行的指令，同样会死锁。两个条件都满足时，编号最小的未完成指令的依赖都已完成，总能向前推进，所以不会死锁。

## 小结

- [x] CUDA Graph 只消掉 CPU 的发射开销；GPU 上每个 kernel 边界还有全局同步、调度、尾部效应和前奏。
- [x] 小模型、小 batch 时每个 kernel 只有一两微秒，空隙能占 decode 一步的一半；大模型、大 batch 时占比很小。
- [x] PDL（sm_90+）：前一个 kernel 触发后，后一个 kernel 提前上场做前奏，`cudaGridDependencySynchronize()` 之后才能碰上一个 kernel 的数据；上一层的输出别走只读缓存；可以录进 CUDA Graph。
- [x] 推理引擎只在 token 少时打开 PDL；"可以启动下一个"不等于"输出已可用"。
- [x] 融合减少边界的个数，包括通信和计算的融合（all-reduce + RMSNorm、MegaMoE）。
- [x] megakernel：常驻 kernel + 指令表 + 计数器依赖，没有边界、能跨算子预取；代价是资源、内存序、死锁和通用性。
