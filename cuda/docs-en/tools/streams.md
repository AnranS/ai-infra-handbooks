# Streams, concurrency and CUDA Graphs

<p class="lead">Once a single kernel is optimized as far as it goes, the next level is system-wide concurrency: overlapping copies with compute, filling the GPU with several small tasks at once, and removing the cost of launching kernels. This chapter covers CUDA streams, events, pinned memory, CUDA Graphs and unified memory. They are infrastructure you find everywhere in inference engines and training frameworks.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is a CUDA stream? What ordering is guaranteed within one stream and between streams?
    2. When is `cudaMemcpyAsync` not really asynchronous?
    3. How do you make stream B wait for a kernel in stream A without blocking the CPU?
    4. What problem do CUDA Graphs solve? What are their constraints?
    5. Why is unified memory (`cudaMallocManaged`) sometimes very slow? How do you mitigate it?

??? success "Answers (try it yourself first, then expand)"
    1. A stream is a queue of operations executed in order. Operations within one stream run strictly in submission order; there is no ordering between streams, which may run concurrently, so any dependency has to be established explicitly.
    2. When the host memory is not pinned (pageable), the driver first copies the data into a pinned staging buffer, which makes the call effectively synchronous.
    3. `cudaEventRecord(e, A)` in stream A, then `cudaStreamWaitEvent(B, e)`: later operations in stream B wait for that event while the CPU carries on.
    4. They remove the launch overhead of many small kernels: record a chain of kernels as a graph and submit it in one call. The constraints: the addresses and shapes are fixed at capture, so inputs must be copied into fixed buffers; nothing requiring CPU synchronization can be captured; and changed parameters mean recapturing or updating the graph.
    5. Touching a page that is not in device memory faults, and the page migrates from the host, which is expensive per fault. Migrate ahead of time with `cudaMemPrefetchAsync` and give hints with `cudaMemAdvise`.

A six-panel strip before the text:

<!-- comic ../assets/comics/streams.webp is in Chinese; put it back once the English version exists -->

## Streams {#流stream}

**A stream is a queue of operations executed in order.** Operations within one stream (kernels, copies, event records) run in submission order; **operations in different streams have no ordering guarantee** and may run concurrently when the hardware allows.

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
kernelA<<<grid, block, 0, s1>>>(...);        // the fourth execution-configuration argument names the stream
kernelB<<<grid, block, 0, s2>>>(...);        // may run concurrently with kernelA
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s1);   // queued after kernelA in s1
cudaStreamSynchronize(s1);                   // the CPU waits for everything in s1
```

Without a stream argument, the **default stream** is used. The legacy default stream has a particular property: it synchronizes with every other "blocking" stream, so an operation on it waits for whatever other streams submitted earlier, and vice versa. That quietly destroys concurrency more often than you would think. Two ways to avoid it:

- create streams with `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)`;
- compile with `--default-stream per-thread` so each host thread's default stream is an ordinary stream.

PyTorch has a current stream per device (`torch.cuda.current_stream()`, `torch.cuda.Stream()`) with exactly these semantics.

## Events: timing and cross-stream dependencies {#事件计时与跨流依赖}

An event is a marker in a stream. Besides [timing](../basics/first-kernel.md#kernel-启动是异步的), it establishes dependencies between streams:

```cuda
cudaEvent_t done;
cudaEventCreateWithFlags(&done, cudaEventDisableTiming);   // turn timing off when it is only used for synchronization, which costs less
producer<<<g, b, 0, s1>>>(buf);
cudaEventRecord(done, s1);
cudaStreamWaitEvent(s2, done);                             // later operations in s2 wait for done
consumer<<<g, b, 0, s2>>>(buf);                            // the CPU does not block
```

This is the basic way of expressing a directed acyclic graph of dependencies on a GPU.

## Pinned memory, and overlapping copies with compute {#锁页内存与拷贝计算重叠}

For `cudaMemcpyAsync` to be genuinely asynchronous, **the host memory must be pinned**. Ordinary `malloc`/`new` gives pageable memory that the operating system may swap out at any moment, which the DMA engine cannot touch, so the driver has to copy it synchronously into an internal pinned buffer first. Allocate pinned memory with `cudaMallocHost` (or `cudaHostAlloc`), or pin existing memory with `cudaHostRegister`. Pinned memory is a limited system resource, so do not allocate it without restraint.

With pinned memory and several streams, a large task can be cut into chunks and pipelined: while chunk i computes, chunk i+1 copies host to device and chunk i-1 copies device to host. GPUs usually have separate copy engines, so both directions and the compute can run at once.

```cuda title="overlap.cu"
// overlap.cu - pinned memory and several streams, overlapping copies with compute
// build: nvcc -O3 -arch=sm_75 overlap.cu -o overlap
#include "common.cuh"

__global__ void heavy(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    for (int k = 0; k < 64; ++k) v = v * 0.999f + 0.001f;   // extra arithmetic on purpose, so compute and copy take comparable time
    x[i] = v;
  }
}

float ref_value(float v) {
  for (int k = 0; k < 64; ++k) v = v * 0.999f + 0.001f;
  return v;
}

int main() {
  const int n = 1 << 26, chunks = 8, chunk = n / chunks;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float *h, *d;
  CUDA_CHECK(cudaMallocHost(&h, bytes));   // pinned memory: the prerequisite for an asynchronous copy
  CUDA_CHECK(cudaMalloc(&d, bytes));
  auto init = [&] { for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i % 100) / 100.f; };

  // the serial version: copy it all in, compute, copy it all out
  init();
  GpuTimer t;
  t.start();
  CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  heavy<<<(n + 255) / 256, 256>>>(d, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
  float serial_ms = t.stop();

  // the pipelined version: in chunks, each copying in, computing and copying out in its own stream
  init();
  std::vector<cudaStream_t> streams(4);
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  t.start();   // recorded on the default stream; cudaDeviceSynchronize inside the timed region covers every stream
  for (int c = 0; c < chunks; ++c) {
    cudaStream_t s = streams[c % streams.size()];
    const size_t off = static_cast<size_t>(c) * chunk;
    CUDA_CHECK(cudaMemcpyAsync(d + off, h + off, chunk * sizeof(float), cudaMemcpyHostToDevice, s));
    heavy<<<(chunk + 255) / 256, 256, 0, s>>>(d + off, chunk);
    CUDA_CHECK(cudaMemcpyAsync(h + off, d + off, chunk * sizeof(float), cudaMemcpyDeviceToHost, s));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  float pipelined_ms = t.stop();

  size_t bad = 0;
  for (int i = 0; i < n; ++i) bad += std::fabs(h[i] - ref_value(static_cast<float>(i % 100) / 100.f)) > 1e-5f;
  std::printf("%s (%zu mismatches)\nserial   : %.3f ms\npipelined: %.3f ms\n", bad ? "FAIL" : "PASS", bad,
              serial_ms, pipelined_ms);
  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFreeHost(h));
  CUDA_CHECK(cudaFree(d));
  return bad ? 1 : 0;
}
```

Viewing this program in Nsight Systems shows the copies and the compute clearly interleaved on the timeline. Ideally the pipelined version takes about as long as the slowest of the three, not their sum.

Move the chunk count, the stream count and the three durations and watch the timeline overlap:

<div class="aig-widget" data-widget="stream-overlap"></div>

## CUDA Graphs {#cuda-graphs}

Every kernel launch costs a few microseconds of CPU time. For a workload made of many small kernels (an LLM's decode step, with hundreds of kernels each running a dozen microseconds), the launch overhead and the CPU's scheduling leave the GPU idle a great deal.

**CUDA Graphs** record a series of operations as a graph that a single call then submits. The driver can do most of the preparation in advance, and the launch overhead drops sharply. The most convenient way to build one is **stream capture**:

```cuda
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... submit kernels and copies on the stream as usual; they are recorded rather than executed ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graph_exec, graph, 0);   // instantiation is expensive, so do it once
for (int step = 0; step < steps; ++step)
  cudaGraphLaunch(graph_exec, stream);         // each replay: one call submits the whole graph
```

```cuda title="cuda_graph.cu"
// cuda_graph.cu - capturing many small kernels into a CUDA Graph, against launching them one by one
// build: nvcc -O3 -arch=sm_75 cuda_graph.cu -o cuda_graph
#include "common.cuh"

__global__ void add_one(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.f;
}

int main() {
  const int n = 4096, kernels_per_step = 200, steps = 20;
  float* d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // 1) launched one by one
  GpuTimer t;
  t.start(s);
  for (int step = 0; step < steps; ++step)
    for (int k = 0; k < kernels_per_step; ++k) add_one<<<(n + 255) / 256, 256, 0, s>>>(d, n);
  float eager_ms = t.stop(s);
  CUDA_CHECK_LAST();

  // 2) capture one step's 200 kernels, after which each step launches the graph once
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int k = 0; k < kernels_per_step; ++k) add_one<<<(n + 255) / 256, 256, 0, s>>>(d, n);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  t.start(s);
  for (int step = 0; step < steps; ++step) CUDA_CHECK(cudaGraphLaunch(exec, s));
  float graph_ms = t.stop(s);

  std::vector<float> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  const float expect = 2.f * steps * kernels_per_step;   // both ways did steps * kernels_per_step increments
  size_t bad = 0;
  for (float v : h) bad += v != expect;
  std::printf("%s (value %.0f, expected %.0f)\neager: %.3f ms\ngraph: %.3f ms\n", bad ? "FAIL" : "PASS", h[0],
              expect, eager_ms, graph_ms);
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(d));
  return bad ? 1 : 0;
}
```

**Constraints**: a graph records a fixed sequence of operations with fixed kernel arguments and **fixed memory addresses**. So:

- inputs and outputs have to live in fixed buffers, with new data copied in before each replay;
- a change of shape needs a different graph. vLLM and SGLang capture decode graphs for a set of common batch sizes (1, 2, 4, 8 … 256) and round the real batch up to the nearest one at run time;
- nothing synchronizing may be captured (a synchronous `cudaMemcpy`, `cudaDeviceSynchronize`), and the CPU cannot branch on a GPU result;
- for small parameter changes, `cudaGraphExecUpdate` or `cudaGraphExecKernelNodeSetParams` updates an instantiated graph instead of reinstantiating it.

PyTorch offers the same through `torch.cuda.CUDAGraph` and `torch.cuda.graph()`, and `torch.compile(mode="reduce-overhead")` uses CUDA Graphs automatically.

!!! tip "PDL on Hopper"
    Hopper supports **Programmatic Dependent Launch**: the next kernel can begin its prologue (loading weights that do not depend on the previous kernel's result, say) before the previous one ends, calling `cudaGridDependencySynchronize()` where the dependent data is actually needed. It squeezes the gap between neighbouring kernels further and is already widely used in inference engines. The mechanism, how to write it, its traps and its use in inference engines are in [the gaps between kernels: PDL and megakernels](pdl-megakernel.md).

## Unified memory {#统一内存}

Memory from `cudaMallocManaged` uses **one pointer** on both CPU and GPU, with the data migrating on demand: touching a page that is not local faults and the driver migrates it. That is convenient for prototyping but has two performance problems:

- **page faults are expensive**: on the GPU's first access, the faults thousands of threads trigger are handled page by page, far slower than one large copy;
- **ping-ponging**: when CPU and GPU take turns on the same data, pages shuttle back and forth.

The mitigation is **prefetching**: call `cudaMemPrefetchAsync` before the launch to migrate the data in one go, and use `cudaMemAdvise` to hint the access pattern. Note that CUDA 13 changed both signatures (the destination is now a `cudaMemLocation` struct), so code supporting both 12.x and 13.x has to branch on the version:

```cuda title="managed_prefetch.cu"
// managed_prefetch.cu - unified memory with prefetching, and how to support both the CUDA 12 and CUDA 13 APIs
// build: nvcc -O3 -arch=sm_75 managed_prefetch.cu -o managed_prefetch
#include "common.cuh"

__global__ void square(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * x[i];
}

void prefetch(const void* p, size_t bytes, int device, cudaStream_t s) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  loc.id = device == cudaCpuDeviceId ? 0 : device;
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, loc, 0, s));
#else
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, device, s));
#endif
}

int main() {
  const int n = 1 << 24;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float* x;
  CUDA_CHECK(cudaMallocManaged(&x, bytes));
  for (int i = 0; i < n; ++i) x[i] = static_cast<float>(i % 1000) * 0.01f;   // written straight from the CPU

  int dev = 0, concurrent = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev));
  if (concurrent) prefetch(x, bytes, dev, 0);   // migrate to the GPU in one go, avoiding a storm of faults inside the kernel

  square<<<(n + 255) / 256, 256>>>(x, n);
  CUDA_CHECK_LAST();
  if (concurrent) prefetch(x, bytes, cudaCpuDeviceId, 0);   // migrate back to the CPU
  CUDA_CHECK(cudaDeviceSynchronize());                       // synchronize before touching it

  size_t bad = 0;
  for (int i = 0; i < n; ++i) {
    float v = static_cast<float>(i % 1000) * 0.01f;
    bad += std::fabs(x[i] - v * v) > 1e-4f;
  }
  std::printf("%s (%zu mismatches)\n", bad ? "FAIL" : "PASS", bad);
  CUDA_CHECK(cudaFree(x));
  return bad ? 1 : 0;
}
```

Unified memory does much better on systems like Grace Hopper (GH200) where CPU and GPU share an address space over a fast NVLink-C2C link. On an ordinary PCIe server, performance-critical code should still manage memory explicitly.

!!! interview "How to explain it"
    On streams and CUDA Graphs: operations run in order within a stream and may run concurrently across streams, with the legacy default stream's implicit synchronization to watch out for; cross-stream dependencies use events (`cudaStreamWaitEvent`) and do not block the CPU; `cudaMemcpyAsync` is only really asynchronous on pinned memory, and chunking plus several streams overlaps copies with compute. A CUDA Graph records a chain of kernels and submits it once, removing the launch overhead of the hundreds of small kernels in a decode step, at the price of fixed addresses and shapes, which is why inference engines capture one per batch-size bucket. Unified memory is convenient but its page faults are expensive, so prefetch.

## Exercises {#练习}

**1. How many chunks in the pipeline.** In `overlap.cu`, how does the time change as `chunks` goes from 2 to 64? Why are more chunks not always better?

??? success "Answer"
    With too few chunks the pipeline's fill and drain phases (the first chunk only copying in, the last only copying out) are a large share and the overlap is poor; more chunks overlap better and the time falls. But with too many, every copy and kernel is tiny: each operation has fixed costs (launch, DMA setup), small copies get lower bandwidth, and a kernel too small cannot fill the GPU, so the time rises again. A few to a dozen chunks is usually enough, but measure.

**2. Expressing a dependency with events.** Three kernels: A and B are independent and C needs both. Implement it with two streams and events so that A and B run concurrently and the CPU never blocks.

??? success "Answer"
    ```cuda
    cudaEvent_t evB;
    cudaEventCreateWithFlags(&evB, cudaEventDisableTiming);
    A<<<g, b, 0, s1>>>(...);
    B<<<g, b, 0, s2>>>(...);
    cudaEventRecord(evB, s2);
    cudaStreamWaitEvent(s1, evB);   // s1 waits for B after A
    C<<<g, b, 0, s1>>>(...);        // queued after A in the same stream, and waiting on B
    ```

    Put this inside a stream capture and the resulting graph is the diamond: A and B in parallel, joining at C.

## Summary {#小结}

- [x] Operations run in order within a stream and may run concurrently across streams; watch the legacy default stream's implicit synchronization.
- [x] Events serve for timing and cross-stream dependencies (`cudaStreamWaitEvent`) without blocking the CPU.
- [x] Asynchronous copies need pinned memory; chunking plus several streams overlaps copies with compute.
- [x] CUDA Graphs remove the launch overhead of many small kernels; addresses and shapes are fixed, so inference engines capture one per batch size.
- [x] Unified memory is convenient but its page faults are expensive, so use prefetching and access hints; CUDA 13 changed the prefetch API's signature.
