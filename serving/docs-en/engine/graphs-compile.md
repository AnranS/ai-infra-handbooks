# CUDA Graphs and torch.compile

<p class="lead">At every decode step, the real computation on the GPU may take only a fraction of a millisecond, yet the CPU must launch over a thousand kernels for it. The smaller the model and the batch, the more this overhead shows, and the GPU spends most of its time waiting for the CPU. Inference engines fight it with two weapons: CUDA Graphs record a whole step's forward pass and submit it at once; torch.compile fuses scattered small operators into a few kernels. This chapter first quantifies how serious the problem is, then explains the principles and constraints of both, and how vLLM and SGLang use them.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How many operators does one decode step of a 0.6B model launch? On an H100, which is larger, the launch overhead or the compute time?
    2. Why does a CUDA Graph require fixed shapes and fixed addresses? How do engines make it work for varying batch sizes?
    3. What is the difference between vLLM's `PIECEWISE` and `FULL` CUDA Graph modes? Why is the default `FULL_AND_PIECEWISE`?
    4. What does torch.compile mainly do in an inference engine? How does it relate to CUDA Graphs?

??? success "Answers (try first, then expand to compare)"
    1. This chapter's mini_llm (0.6B) has about 2269 operator calls and about 1600 kernel launches per decode step. At 5 μs each, launching alone takes about 8 ms, while an H100 reads the weights once in about 0.36 ms: the launch overhead far exceeds the computation, and the GPU spends over 95% of its time waiting for the CPU.
    2. A graph records concrete kernels, arguments and memory addresses and replays them as is, so shapes and addresses must be fixed. Engines capture one graph per batch size in a set (bucketing), pad the actual batch to the nearest bucket, and copy inputs and metadata into the static buffers used at capture time.
    3. `PIECEWISE`: split the graph at attention into several pieces, capture each piece separately, and run attention normally outside the graphs; it works with any attention backend and with mixed batches. `FULL`: capture attention into one graph too, for the smallest launch overhead, but it requires support from the attention backend. `FULL_AND_PIECEWISE` uses the full graph for pure decode batches and piecewise graphs for prefill and mixed batches, getting the best of both.
    4. It fuses small memory-bound operators (normalization, activations, RoPE and so on), compiles once with symbolic shapes and then specializes for the captured sizes, and runs custom fusion passes. It reduces the number of kernels and memory traffic, while CUDA Graphs reduce launch overhead; the two are used together (the compiled code is then captured into graphs); the price is startup time and memory.

<!-- comic ../assets/comics/graphs-compile.webp is in Chinese; put it back once the English version exists -->

## How serious the problem is {#问题有多严重}

Use PyTorch's profiler to count how many operators one decode step of `mini_llm` calls:

```python
import torch
from collections import Counter
from torch.profiler import ProfilerActivity, profile
from mini_llm import KVCache, Transformer

torch.set_num_threads(16)
model = Transformer.from_pretrained("models/Qwen3-0.6B")
cache = KVCache(model.cfg.num_hidden_layers)
with torch.no_grad():
    model(torch.randint(0, 1000, (1, 64)), cache)                    # prefill 64 tokens first
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        model(torch.tensor([[5]]), cache)                             # one decode step

ops = [e.name for e in prof.events() if e.name.startswith("aten::") and e.cpu_parent is None]
views = {"aten::view", "aten::transpose", "aten::chunk", "aten::reshape", "aten::expand", "aten::slice",
         "aten::unsqueeze", "aten::t", "aten::split", "aten::detach", "aten::alias", "aten::select",
         "aten::to"}                                                  # to does nothing when the dtype is unchanged
compute = [o for o in ops if o not in views]                          # view operations launch no kernel
print(f"decode 一步：{len(ops)} 个算子调用，其中约 {len(compute)} 个需要启动 kernel")
print(Counter(compute).most_common(6))
```

```text
decode 一步：2269 个算子调用，其中约 1617 个需要启动 kernel
[('aten::mul', 368), ('aten::add', 225), ('aten::linear', 197), ('aten::pow', 114), ('aten::cat', 113), ('aten::mean', 113)]
```

On a GPU, launching each kernel takes a few microseconds of CPU time (Python dispatch + PyTorch dispatch + the CUDA driver). At about 5 μs each, 1600 kernels take around 8 ms, while an H100 reads the 0.6B model's BF16 weights (about 1.2 GB) once in only 0.36 ms. **The GPU spends over 95% of its time waiting for the CPU to launch the next kernel.** Even for an 8B model (about 4.8 ms to read the weights), the launch overhead is comparable to the compute time. This is why the decode phase can hardly do without CUDA Graphs.

Put these numbers into a simple time model to see how large a share launch overhead takes and how much each of the two techniques saves:

<div class="aig-widget" data-widget="launch-overhead"></div>

## CUDA Graphs: capture once, replay many times {#cuda-graphs录一次放很多次}

A CUDA Graph records a sequence of GPU operations (kernel launches, memory copies) as a graph, which can afterwards be submitted with a single call and executed in order on the GPU side, saving each kernel's CPU launch overhead:

```py
# capture: put the inputs in fixed "static buffers" and run the forward pass once inside the graph context
static_input_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
static_positions = torch.zeros(batch_size, dtype=torch.long, device="cuda")
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_logits = model(static_input_ids, static_positions, attn_metadata)

# replay: each step just copies new data into the same buffers, then replays
static_input_ids.copy_(new_input_ids)
static_positions.copy_(new_positions)
graph.replay()
next_logits = static_logits.clone()
```

What gets recorded are **concrete kernels and concrete memory addresses**, so there are three hard constraints:

1. **Fixed shapes**: a graph captured with batch 8 can only be used for inputs with batch 8;
2. **Fixed addresses**: inputs, outputs and intermediate results all live in the memory allocated at capture time, so every step must copy its data into the same input buffers;
3. **No CPU logic inside the graph**: no Python branches that depend on GPU data, and no CPU-GPU synchronization such as `.item()`.

### Fixed graphs for varying batches {#让变化的-batch-用上固定的图}

Every step has a different batch size, so engines capture one graph for each of a set of batch sizes ahead of time, and at run time **pad** the actual batch up to the nearest captured size no smaller than it. vLLM's default capture sizes are:

```pycon
>>> import bisect
>>> max_size = 512
>>> sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))
>>> len(sizes), sizes[:8], sizes[-3:]
(51, [1, 2, 4, 8, 16, 24, 32, 40], [480, 496, 512])
>>> def padded(n):                       # pad to the nearest captured size; beyond the maximum, no CUDA Graph
...     i = bisect.bisect_left(sizes, n)
...     return sizes[i] if i < len(sizes) else None
>>> [padded(n) for n in (1, 3, 9, 100, 300, 600)]
[1, 4, 16, 104, 304, None]
```

Captures are dense for small batches (little padding waste) and sparse for large ones (too many graphs would take memory and slow down startup). The padded tokens are "fake": their KV is written to a reserved empty block (block 0, called `null_block` in vLLM; SGLang's `ReqToTokenPool` likewise reserves row 0), and their results are discarded.

### Attention is the hard part: piecewise and full graphs {#注意力是难点分段图与完整图}

A decode batch's shape is determined only by the batch size, but attention also depends on metadata that changes every step: each request's context length and block table. For the attention kernel to be captured into a graph, this metadata must also live in fixed buffers and be read by the kernel from memory, rather than passed into the kernel as Python arguments. Not every attention backend supports this. So vLLM has two modes:

| Mode | How | Suited for |
| --- | --- | --- |
| `PIECEWISE` | split the computation graph at attention into several pieces, capture each piece separately, and run attention normally outside the graphs | any attention backend; prefill and mixed batches |
| `FULL` | capture everything, attention included, into one graph | attention backends that support metadata in fixed buffers (FlashAttention 3, FlashInfer, FlashMLA decode and others) |
| `FULL_AND_PIECEWISE` (the V1 default) | full graphs for pure decode batches, piecewise graphs for prefill and mixed batches | the best performance for most models |

Piecewise graphs cut thousands of kernel launches down to about "layers × 2" graph replays plus the attention calls, already saving most of the overhead; full graphs go further, needing just one replay per step.

## torch.compile: fusing small operators {#torchcompile把小算子融合起来}

CUDA Graphs save **launch** overhead, but each kernel still reads and writes memory on its own. Small elementwise or reduction operators such as RMSNorm, residual addition, SiLU × up and RoPE are memory bound: fused into one kernel, the data is read and written only once. torch.compile (with the TorchInductor backend) does this automatically:

```python
import re
from torch._inductor.utils import run_and_get_code

def fused_add_rms_norm(x, residual, weight, eps: float = 1e-6):
    residual = x + residual
    var = residual.pow(2).mean(-1, keepdim=True)
    return weight * (residual * torch.rsqrt(var + eps)), residual

x, r, w = torch.randn(16, 1024), torch.randn(16, 1024), torch.randn(1024)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    eager = fused_add_rms_norm(x, r, w)
eager_ops = [e.name for e in prof.events() if e.name.startswith("aten::") and e.cpu_parent is None]
compiled, code = run_and_get_code(torch.compile(fused_add_rms_norm), x, r, w)
kernels = sorted(set(re.findall(r"cpp_fused_\w+", code[0])))
print(f"eager：{len(eager_ops)} 个算子 {eager_ops}")
print(f"compile 之后：{len(kernels)} 个融合 kernel {kernels}")
print("结果一致：", all(torch.allclose(a, b, atol=1e-5) for a, b in zip(eager, compiled)))
```

```text
eager：7 个算子 ['aten::add', 'aten::pow', 'aten::mean', 'aten::add', 'aten::rsqrt', 'aten::mul', 'aten::mul']
compile 之后：1 个融合 kernel ['cpp_fused_add_mean_mul_pow_rsqrt_0']
结果一致： True
```

7 operators are fused into 1 kernel (C++ is generated on CPU, Triton on GPU). On a GPU, this means the residual is read once and written once, instead of being read and written repeatedly by 7 kernels.

torch.compile has a shape problem too. By default it compiles for concrete shapes first; when it meets a new shape it recompiles and tries to treat the varying dimension as symbolic:

```python
from torch._dynamo.utils import counters

for dynamic in (False, True):
    torch._dynamo.reset()
    counters.clear()
    f = torch.compile(fused_add_rms_norm, dynamic=dynamic)
    for n in (1, 2, 3, 5, 8, 13):                                 # 6 different token counts
        f(torch.randn(n, 1024), torch.randn(n, 1024), w)
    print(f"dynamic={dynamic}：6 种形状共编译了 {counters['stats']['unique_graphs']} 次")
```

```text
dynamic=False：6 种形状共编译了 6 次
dynamic=True：6 种形状共编译了 2 次
```

With `dynamic=False`, every shape needs its own compilation; with `dynamic=True`, the token count becomes symbolic and only two compilations are needed (dimensions of size 1 are specialized separately, a torch.compile convention). The token count of an inference engine changes every step, so vLLM's approach is to **compile once with a symbolic token count**, producing a general version that works for any token count; if needed, `compile_sizes` can add specialized versions for given sizes (all the CUDA Graph capture sizes, for example), or `compile_ranges_endpoints` can compile by range. Compilation results are cached on disk (`~/.cache/vllm/torch_compile_cache` by default) and loaded directly on restart.

## How the two work together, and what they cost {#两者的配合与代价}

In vLLM, torch.compile and CUDA Graphs are two orthogonal layers:

1. torch.compile captures the model's computation graph, fuses with custom Inductor passes (RMSNorm + quantization, SiLU × up + quantization, all-reduce + RMSNorm and so on), and cuts the graph at attention (piecewise compilation);
2. CUDA Graphs capture each compiled piece (or the whole step) and replay it.

The cost is mainly **startup time and memory**: compiling and capturing graphs for dozens of batch sizes can take tens of seconds to several minutes, and the captured graphs take some memory. For debugging, turn them off with `--enforce-eager` (vLLM) or `--disable-cuda-graph` (SGLang) to investigate problems such as "results are wrong once CUDA Graphs are on".

!!! source "Source code"
    - **vLLM**: the CUDA Graph mode is controlled by `CompilationConfig.cudagraph_mode` (`vllm/config/compilation.py`), whose docstring explains `NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE` in detail; the default rule for generating `cudagraph_capture_sizes` is exactly the `[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)` above, with `max_cudagraph_capture_size` defaulting to at most 512 (1024 on data-center Blackwell). At run time `vllm/v1/cudagraph_dispatcher.py` chooses the mode and padded size by batch type, and capture happens in `GPUModelRunner.capture_model` / `_capture_cudagraphs`. Compilation code is in `vllm/compilation/`.
    - **SGLang**: `srt/model_executor/runner/` has `decode_cuda_graph_runner.py` and `prefill_cuda_graph_runner.py`, and `runner_backend/` has backends for full graphs, piecewise graphs based on torch.compile (`tc_piecewise_cuda_graph_backend.py`) and more. In 0.5.20 the decode and prefill phases are configured separately; common flags are `--cuda-graph-max-bs-decode`, `--cuda-graph-bs-decode`, `--disable-cuda-graph` and `--disable-prefill-cuda-graph`.

!!! interview "In an interview"
    "Why do CUDA Graphs speed up decode? What are their limits?" First state the problem with numbers: a decode step launches over a thousand kernels at a few microseconds each, while the GPU computation takes only a fraction of a millisecond to a few milliseconds, so launch overhead dominates; a CUDA Graph records the whole step and submits it at once, eliminating that overhead. Then the limits: fixed shapes and addresses (hence bucketed capture by batch size, padding and static buffers), no CPU synchronization or data-dependent branches inside the graph, and attention metadata in fixed buffers (hence the piecewise and full modes). Finish with the cost: startup time and memory.

## Exercises {#练习}

**1. The cost of padding.** With vLLM's default capture sizes, what does a batch of 257 pad to? What share of the computation is wasted? Why is it acceptable to capture more sparsely at large batch sizes?

??? success "Answer"
    257 is padded to 272, wasting 15 / 272 ≈ 5.5%. At large batch sizes decode is closer to compute bound and each step takes longer, so CPU launch overhead is a small share to begin with, and even without CUDA Graphs little would be lost; at very small batch sizes launch overhead is the largest share, and precise capture sizes matter most. Also, every graph takes memory and adds startup time, so the number of captured sizes cannot grow without limit.

**2. Why do results change?** Someone finds that with CUDA Graphs on, the greedy output of the same request differs from when they are off. What are the possible causes?

??? success "Approach"
    - Padding changes the batch size, so matrix multiplications may pick different kernels or splits, the order of floating-point accumulation differs, the logits differ slightly, and near-tied tokens flip the argmax (see the LLM book's [which optimizations change the output](llm://synthesis/token-journey/#哪些优化会改变输出));
    - torch.compile's fusion changes the order of computation;
    - A real bug: the graph reads a stale buffer (some input was not copied into the static buffers), padded requests' KV is written into a real request's blocks, and so on.

    The first two are normal numerical differences, judged with a logits error threshold; the third requires comparing each layer's output with CUDA Graphs on and off to pin down.

## Summary {#小结}

- [x] A decode step launches over a thousand kernels at a few microseconds each, and launch overhead is often longer than the GPU computation.
- [x] A CUDA Graph records one step's forward pass and submits it at once; it requires fixed shapes and addresses, hence bucketed capture by batch size, padding and static buffers.
- [x] Attention metadata changes every step, so vLLM has piecewise graphs (attention outside) and full graphs (attention inside), defaulting to full graphs for decode and piecewise graphs for mixed batches.
- [x] torch.compile fuses small memory-bound operators, compiling once with symbolic shapes and specializing for the captured sizes; it works together with CUDA Graphs, at the cost of startup time and memory.
