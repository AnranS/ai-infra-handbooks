# torch.compile: Dynamo, AOTAutograd and Inductor

<p class="lead">In eager mode an RMSNorm is 6 kernels, each reading and writing the tensor once; a decode step has a few hundred small kernels and the CPU's submission cost takes a large share. <code>torch.compile</code> captures Python code into a graph, fuses operators, generates kernels, and can wrap a CUDA Graph around the result. Both vLLM and SGLang use it. This chapter covers its three layers and the problems that come up most when using it inside an inference framework: graph breaks, recompilation and dynamic shapes.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does each of `torch.compile`'s three components, Dynamo, AOTAutograd and Inductor, do?
    2. What is a graph break? How do you find them?
    3. Calling the same compiled function with a different batch size, does it recompile?
    4. How much memory traffic does Inductor's "operator fusion" save on an RMSNorm?
    5. Why do inference frameworks often compile a model in **segments** and leave attention outside the graph?

??? success "Answers (try it yourself first, then expand)"
    1. Dynamo: captures an FX graph at the level of Python bytecode and generates guards (the assumptions about shapes, types and so on); AOTAutograd: generates the backward graph ahead of time and functionalizes (removing in-place modifications and views); Inductor: fuses and schedules, generating Triton (GPU) or C++ (CPU) code.
    2. When Dynamo meets code it cannot capture (data-dependent control flow, `.item()`, printing, an unregistered C extension), it breaks the graph into several pieces and returns to Python in between. Find them with `torch._dynamo.explain` or `TORCH_LOGS="graph_breaks"`, and make it raise instead with `fullgraph=True`.
    3. The first time the batch size changes, yes: the guards hold a concrete shape, the failure triggers a recompilation, and this time the batch dimension is marked dynamic; after that any batch size compiles no further (sizes 0 and 1 excepted).
    4. In eager mode an RMSNorm is about 6 separate kernels each reading and writing the tensor once; fused into one kernel it reads the input once and writes the output once, cutting the traffic to a fraction (to about a quarter in the next chapter's RMSNorm + SiLU + multiply example).
    5. Attention needs its own kernel (FlashAttention, paged KV) and metadata that changes every step, which compilers handle poorly and which does not record well into a CUDA Graph; leaving it outside, compiling the rest in segments and recording CUDA Graphs for them, gets the fusion without giving up attention's flexibility.

## The three layers {#三层结构}

![Figure: torch.compile's three layers - Dynamo captures the graph, AOT Autograd expands it into operators, Inductor fuses and generates kernels](../assets/figures/compile-pipeline.svg){.aig-svg}

1. **Dynamo** (graph capture): "symbolically executes" your function at the level of Python bytecode, recording the tensor operations it meets into an FX graph, and generating a set of **guards** along with it (say "the input's shape is `(4, 16)` and its dtype is float32"). On the next call, all guards satisfied means the compiled result is reused, and otherwise it recompiles. Where it meets something it cannot capture (data-dependent control flow, an unregistered C extension, a `print`), it **breaks the graph** there, compiling a piece on each side and returning to Python in between;
2. **AOTAutograd**: generates the forward graph together with its backward graph ahead of time (needed for training), and "functionalizes" by rewriting in-place modifications into pure-function form for later optimization; at inference time it is mainly the latter that matters;
3. **Inductor** (code generation): lowers the graph to a loop-level intermediate representation, fuses what can be fused, and generates Triton kernels on a GPU and C++ / OpenMP code on a CPU, while matrix multiplies call cuBLAS / CUTLASS or an auto-tuned Triton template.

Each layer can be swapped out on its own: `backend="eager"` does Dynamo's capture and no code generation, and `backend="aot_eager"` adds AOTAutograd, both commonly used to bisect "which layer went wrong".

## Seeing what Dynamo captured {#看-dynamo-捕获了什么}

`backend` can be any function that takes an FX graph. Write a backend that only prints the graph and you see what Dynamo captured:

```python title="show_graph.py"
import torch


def show_graph(gm, example_inputs):
    print(gm.code.strip())
    return gm.forward          # return a callable: here the unoptimized graph itself


def mlp(x, w):
    return torch.nn.functional.silu(x @ w)


torch.compile(mlp, backend=show_graph)(torch.randn(4, 16), torch.randn(16, 16))
```

```text title="output"
def forward(self, L_x_ : torch.Tensor, L_w_ : torch.Tensor):
    l_x_ = L_x_
    l_w_ = L_w_
    matmul = l_x_ @ l_w_;  l_x_ = l_w_ = None
    silu = torch.nn.functional.silu(matmul);  matmul = None
    return (silu,)
```

## graph break {#graph-break}

Python control flow that depends on a tensor's **value** gives Dynamo no way to decide at compile time which branch to take, so it breaks the graph:

```python title="graph_break.py"
import torch


def step(x, w):
    h = x @ w
    if h.sum().item() > 0:      # data-dependent Python control flow
        h = h * 2
    return torch.relu(h)


exp = torch._dynamo.explain(step)(torch.randn(4, 16), torch.randn(16, 16))
print("捕获了", exp.graph_count, "张图，断开", exp.graph_break_count, "次")
for r in exp.break_reasons:
    print("原因：", str(r.reason).splitlines()[0])
```

```text title="output"
捕获了 2 张图，断开 1 次
原因： Unsupported Tensor.item() call with capture_scalar_outputs=False
```

Every break costs one more "return to Python, synchronize, re-enter compiled code", and loses the fusion across the break; on a GPU, `.item()` is itself a CPU-GPU synchronization (previous chapter). Ways to find graph breaks:

- `torch._dynamo.explain(fn)(*args)`, or the environment variable `TORCH_LOGS="graph_breaks"`;
- `torch.compile(fn, fullgraph=True)`: raise on any break, which suits an inference framework's "this stretch has to be one whole graph";
- the fix is usually to rewrite the data-dependent decision as a tensor operation (`torch.where`), or to register the unsupported call as a custom operator (previous chapter).

## Guards and recompilation {#guard-与重新编译}

The guards include the inputs' shapes. A changed shape fails a guard and forces a recompilation. PyTorch's default policy is: compile for the concrete shape the first time, and when a dimension takes a second different value, mark that dimension **dynamic** and compile a version valid for any size:

```python title="recompile.py"
import torch

compiles = []


def counting_backend(gm, example_inputs):
    compiles.append(len(example_inputs))
    return gm.forward


def mlp(x, w):
    return torch.nn.functional.silu(x @ w)


f = torch.compile(mlp, backend=counting_backend)
w = torch.randn(16, 16)
for batch in (4, 4, 8, 16, 32):
    f(torch.randn(batch, 16), w)
    print(f"batch={batch:<3} 累计编译 {len(compiles)} 次")
```

```text title="output"
batch=4   累计编译 1 次
batch=4   累计编译 1 次
batch=8   累计编译 2 次
batch=16  累计编译 2 次
batch=32  累计编译 2 次
```

The second compilation produces a dynamic-shape graph (the batch dimension became a symbol), and no batch size after that compiles again. In an inference service the batch size and the sequence length change every step, so the common practice is:

- mark the dimensions that will vary explicitly with `torch._dynamo.mark_dynamic(x, 0)` before the first call, avoiding that first compilation for a concrete shape;
- watch the reason for each recompilation with `TORCH_LOGS="recompiles"`, to keep production requests from triggering compilation (seconds to tens of seconds each, which is a serious latency spike);
- warm every path up with typical shapes when the service starts.

## What Inductor does: fusion {#inductor-做了什么融合}

RMSNorm + SiLU again. Eager mode is 7 separate aten operators (the previous chapter showed the first 6) and Inductor fuses them into **one** kernel:

```python title="inductor_fusion.py"
import re

import torch
from torch._inductor.utils import run_and_get_code


def rmsnorm_silu(x, w):
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    return torch.nn.functional.silu(h)


torch.manual_seed(0)
x, w = torch.randn(8, 4096), torch.randn(4096)
out, codes = run_and_get_code(torch.compile(rmsnorm_silu), x, w)   # get the result and the generated code at once
print("与 eager 结果一致：", torch.allclose(out, rmsnorm_silu(x, w), atol=1e-5))
print("生成的 kernel：", sorted(set(re.findall(r"cpp_fused_\w+", "\n".join(codes)))))
```

```text title="output"
与 eager 结果一致： True
生成的 kernel： ['cpp_fused_add_mean_mul_pow_rsqrt_silu_0']
```

The kernel's name lists the operations fused into it. On a CPU the generated code is C++, and on a GPU it is a Triton kernel (`triton_per_fused_...`).
The change in memory traffic: in eager mode `pow`, `mul`, `mul` and `silu` each read and write the whole tensor, `mean` reads it once, and with the intermediates written back that is about 10 times the input's size in reads and writes; fused, each row reads the input and the weight once and writes the output once. For a bandwidth-bound setting like decode, that is close to a 5 to 10 times difference.

Setting `TORCH_LOGS="output_code"` prints the generated code directly; `TORCH_COMPILE_DEBUG=1` writes every step's intermediate representation to disk.

## Compilation modes and inference frameworks {#编译模式与推理框架}

| Usage | What it does |
| --- | --- |
| `torch.compile(model)` | the default: capture, fuse, generate kernels |
| `mode="reduce-overhead"` | also wraps a CUDA Graph around it, removing the CPU cost of submitting kernels, which suits small-batch decode |
| `mode="max-autotune"` | auto-tunes operators like matrix multiply (timing several Triton templates against cuBLAS and picking the fastest), compiling more slowly |
| `dynamic=True / False` | compile for dynamic / static shapes from the start |

An inference framework does not simply call `torch.compile(model)` and stop there:

- **compiling in segments**: attention reads and writes paged KV and depends on the block table and on each request's own length, so it usually stays outside the graph as an opaque custom operator; vLLM cuts the model into segments at the attention calls, compiles each separately, and records a CUDA Graph per segment per batch size;
- **custom passes**: specific pattern replacements on the FX graph, say fusing "all-reduce + RMSNorm" into one kernel that fuses communication with computation, or fusing quantization into the preceding operator;
- **a compilation cache**: caching the compiled result on disk, so startup does not wait minutes every time.

The Inference Systems handbook's [CUDA Graphs and torch.compile](serving://engine/graphs-compile/) discusses all of this more concretely.

!!! interview "Answering in an interview"
    On torch.compile: Dynamo captures an FX graph at the bytecode level and generates guards, AOTAutograd generates the backward graph and functionalizes, and Inductor fuses and generates Triton / C++ code; data-dependent control flow, `.item()` and unregistered extensions break the graph, found with `fullgraph=True` and `TORCH_LOGS`. A failed guard recompiles: the first compilation is for the concrete shape and a dimension's second change marks it dynamic, so in production you mark the varying dimensions in advance and warm up. Inference frameworks usually compile in segments, leave attention outside the graph, record a CUDA Graph per segment, and run custom fusion passes on the FX graph.

## Exercises {#练习}

1. Rewrite `step` in this chapter's `graph_break.py` without a graph break (same results as before) and verify with `fullgraph=True`.

??? success "Answer"
    Rewrite Python's `if` as tensor operations: the condition and both branches are computed in the graph and `torch.where` picks between them:

    ```python title="no_break.py"
    import torch


    def step(x, w):
        h = x @ w
        if h.sum().item() > 0:
            h = h * 2
        return torch.relu(h)


    def step_nobreak(x, w):
        h = x @ w
        h = torch.where(h.sum() > 0, h * 2, h)   # the condition is a 0-d tensor and stays in the graph
        return torch.relu(h)


    torch.manual_seed(0)
    compiled = torch.compile(step_nobreak, fullgraph=True, backend="eager")
    ok = all(torch.allclose(compiled(x, w), step(x, w)) for x, w in [(torch.randn(4, 16), torch.randn(16, 16)) for _ in range(5)])
    print("fullgraph 编译成功，结果一致：", ok)
    ```

    ```text title="output"
    fullgraph 编译成功，结果一致： True
    ```

    The cost is computing both branches. For inference that usually pays: one extra elementwise multiply is far cheaper than a synchronization and a graph break.

2. A production service occasionally shows latency spikes of a few seconds, and the logs contain Dynamo recompile messages. What could cause this? How do you diagnose and avoid it?

??? success "Answer"
    Causes: some input's shape (the batch, the sequence length, the number of LoRAs) took a new value while that dimension is not yet marked dynamic; a guard checks the value of a Python object (a configuration integer or a global that changes); an input's dtype or device changed; the recompilation limit was exceeded and it fell back to eager.
    Diagnosis: `TORCH_LOGS="recompiles"` shows which guard failed each time. Avoidance: `mark_dynamic` for the dimensions that vary; turn varying scalars into tensor inputs; warm up with every typical shape at startup; and inference frameworks usually also pad the batch up to a few fixed sizes (the same set as the CUDA Graphs').

## Summary {#小结}

- [x] Dynamo captures an FX graph at the bytecode level and generates guards; AOTAutograd generates the backward graph and functionalizes; Inductor fuses and generates Triton / C++ code.
- [x] Data-dependent control flow, `.item()` and unregistered extensions break the graph; find them with `explain`, `TORCH_LOGS` and `fullgraph=True`, and remove them with tensor operations or a custom operator.
- [x] A failed guard recompiles; mark the dimensions that vary as dynamic and warm up at startup, to avoid compiling in production.
- [x] Inductor fuses a chain of elementwise operations and a reduction into one kernel, which pays most for bandwidth-bound inference.
- [x] Inference frameworks compile in segments, leave attention outside the graph, record a CUDA Graph per segment, and run custom fusion passes on the FX graph.
