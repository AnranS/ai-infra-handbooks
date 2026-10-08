# The dispatcher and custom operators

<p class="lead">When you call <code>torch.add(a, b)</code>, PyTorch has to decide: record autograd or not, cast for automatic mixed precision or not, and finally call the CPU or the CUDA kernel. The dispatcher does that. Understanding it is what lets you register a custom operator correctly so it supports autograd, <code>torch.compile</code> and shape inference on the meta device, which every custom kernel in an inference framework has to pass through.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which dispatcher layers does one `torch.mm` call pass through?
    2. What is an aten operator? How do you see which aten operators a piece of model code actually calls?
    3. When registering a custom operator, why must you declare which arguments are modified in place?
    4. What are the meta device and a fake tensor? What do inference frameworks use them for?
    5. What is the difference under `torch.compile` between exposing a function with pybind11 and registering an operator with `torch.library`?

??? success "Answers (try it yourself first, then expand)"
    1. From the outside in: automatic mixed precision (Autocast, casting the inputs when needed), Autograd (recording a backward node when the inputs need gradients), ADInplaceOrView (maintaining the version counters for in-place modifications and views), and the device kernel (CPU / CUDA).
    2. aten is PyTorch's low-level operator library (`aten::mm`, `aten::silu` and so on), where all model code ends up. A `TorchDispatchMode` intercepts every call that reaches the bottom layer, and printing `func` shows them.
    3. The compiler and functionalization decide from the schema whether an operator has side effects: declare an in-place modification and the compiler will not delete, reorder or reuse its result; declare it wrongly and eager mode is fine while `torch.compile` silently computes the wrong thing.
    4. A tensor on the meta device has only a shape and a type with no data; a fake tensor goes further and pretends to be on a real device. Inference frameworks use them to build a model, count parameters, plan memory and shard it without allocating anything, and `torch.compile` uses fake tensors to infer every intermediate's shape.
    5. A function exposed with pybind11 is a black box to torch.compile and Dynamo can only graph break there; an operator registered with `torch.library` has a schema, a fake implementation (and optionally a backward), so it is captured into the graph in full and its shapes are inferable on the meta device.

## From a Python call to a kernel {#从-python-调用到-kernel}

![Figure: how one torch.add reaches a kernel, dispatched layer by layer on the dispatch keys](../assets/figures/dispatcher-keys.svg){.aig-svg}

Every PyTorch operator (`aten::mm`, `aten::silu` and so on) has a table in the dispatcher registering implementations under **dispatch keys**. One call passes through each key the tensor carries, in priority order:

```python title="dispatch_keys.py"
import torch

x = torch.randn(3)
print(torch._C._dispatch_keys(x))
```

```text title="output"
DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
```

From the outside in: `AutocastCPU` (automatic mixed precision, casting the inputs down when needed), `AutogradCPU` (recording a backward node when the inputs need gradients), `ADInplaceOrView` (maintaining version counters for in-place modifications and views), and `CPU` (the real kernel). Each layer does its job and redispatches to the next. With the tensors on a GPU the last layer is `CUDA`; on the meta device it is `Meta` (shapes only, no data).

All model code ends up in this set of **aten operators**. A `TorchDispatchMode` intercepts every call that reaches the bottom layer and shows exactly what a piece of code executes:

```python title="log_aten_ops.py"
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class LogOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


torch.manual_seed(0)
x = torch.randn(2, 16)
w = torch.ones(16)
lin = torch.nn.Linear(16, 32, bias=False)
with LogOps() as log:
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w   # RMSNorm
    y = torch.nn.functional.silu(lin(h))                               # Linear + SiLU
for op in log.ops:
    print(op)
```

```text title="output"
aten.pow.Tensor_Scalar
aten.mean.dim
aten.add.Tensor
aten.rsqrt.default
aten.mul.Tensor
aten.mul.Tensor
aten.t.default
aten.mm.default
aten.silu.default
```

An RMSNorm in eager mode is 6 separate kernels, each reading and writing the whole tensor once, which is why inference frameworks write a fused RMSNorm kernel and why `torch.compile` exists (the next chapter).
`TorchDispatchMode` is also the basis of many tools: counting FLOPs, checking numerics and recording an operator sequence for replay all intercept this way.

## Registering a custom operator {#注册自定义算子}

An inference framework's custom kernels (a fused RMSNorm, writing the KV cache, MoE routing and sorting) all have to be registered as operators in the dispatcher. From PyTorch 2.4, `torch.library.custom_op` is the recommended way:

```python title="custom_ops.py"
import torch


@torch.library.custom_op("demo::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # a real system would call its own CUDA / Triton kernel here; PyTorch stands in
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@rms_norm.register_fake
def _(x, w, eps):
    return torch.empty_like(x)   # describes the output's shape and type only, computing nothing


@torch.library.custom_op("demo::store_kv", mutates_args=("cache",))
def store_kv(cache: torch.Tensor, slots: torch.Tensor, value: torch.Tensor) -> None:
    cache.index_copy_(0, slots, value)   # write the new token's KV into the given slot of the cache


print(torch.ops.demo.rms_norm.default._schema)
print(torch.ops.demo.store_kv.default._schema)

torch.manual_seed(0)
x, w = torch.randn(2, 16), torch.ones(16)
print("CPU 上计算：", tuple(torch.ops.demo.rms_norm(x, w, 1e-6).shape))

m = torch.ops.demo.rms_norm(torch.empty(4096, 8192, device="meta"), torch.empty(8192, device="meta"), 1e-6)
print("meta 设备上只推导形状：", tuple(m.shape), m.device)

cache = torch.zeros(8, 4)
torch.ops.demo.store_kv(cache, torch.tensor([1, 5]), torch.ones(2, 4))
print("写入的槽位：", cache.sum(dim=1).nonzero().flatten().tolist())


def block(x, w):
    return torch.ops.demo.rms_norm(x, w, 1e-6).relu()


compiled = torch.compile(block, fullgraph=True, backend="eager")   # fullgraph: the whole thing has to be captured as one graph
print("torch.compile 能完整捕获：", torch.allclose(compiled(x, w), block(x, w)))
```

```text title="output"
demo::rms_norm(Tensor x, Tensor w, float eps) -> Tensor
demo::store_kv(Tensor(a0!) cache, Tensor slots, Tensor value) -> ()
CPU 上计算： (2, 16)
meta 设备上只推导形状： (4096, 8192) meta
写入的槽位： [1, 5]
torch.compile 能完整捕获： True
```

A few points:

- **the schema** comes from the type annotations: `Tensor(a0!)` says `cache` is modified in place. That is how the compiler and functionalization know the operator has side effects and must not be deleted, reordered or treated as a pure function whose result can be cached; get it wrong and you get a silently wrong answer;
- **the fake implementation** (`register_fake`) computes only the output's shape and type from the inputs'. `torch.compile` uses it for shape inference at compile time, as does the meta device; without it the compiler can only graph break there;
- when a backward is needed, register it with `torch.library.register_autograd`, just like the previous chapter's `autograd.Function`;
- a C++ / CUDA implementation registers with `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` to exactly the same effect (see the C++ handbook's [pybind11 and PyTorch C++ extensions](cpp://engineering/python-binding/)). vLLM's `torch.ops._C.*` and SGLang's `sgl_kernel` are registered this way.

## The meta device and fake tensors {#meta-设备与-fake-tensor}

A tensor on the **meta device** has metadata and no data, and any operation only infers the output's shape and type. Inference frameworks use it to:

- **build a model without allocating memory**: instantiate on the meta device first, count the parameters, plan the memory and the tensor-parallel sharding, then load the real weights onto the target device piece by piece (avoiding the doubled memory of "build a whole model on the CPU and move it to the GPU");
- **estimate peak memory**: run a forward pass on fake inputs and record every intermediate's size.

A **fake tensor** goes further: it pretends to be on a real device (`cuda:0`, say) without allocating. `torch.compile` infers all its shapes at compile time by running the model on fake tensors:

```python title="meta_model.py"
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

with torch.device("meta"):   # build an MLP the size of a 70B model's without allocating anything
    mlp = torch.nn.Sequential(torch.nn.Linear(8192, 28672, bias=False), torch.nn.Linear(28672, 8192, bias=False))
n = sum(p.numel() for p in mlp.parameters())
print(f"参数量 {n}，bf16 下 {n * 2 / 2**30:.2f} GiB，权重在 {mlp[0].weight.device} 上")

with FakeTensorMode():
    x = torch.empty(4, 128, 8192)        # a batch of hidden states, taking no memory
    h = torch.nn.functional.silu(x @ torch.empty(8192, 28672))
    print(type(h).__name__, tuple(h.shape), f"这个中间激活在 bf16 下需要 {h.numel() * 2 / 2**20:.0f} MiB")
```

```text title="output"
参数量 469762048，bf16 下 0.88 GiB，权重在 meta 上
FakeTensor (4, 128, 28672) 这个中间激活在 bf16 下需要 28 MiB
```

!!! interview "How to explain it"
    On the dispatcher: a call is handled layer by layer on the dispatch keys, automatic mixed precision, then autograd, then version counting, then the device kernel; all models end up in aten operators, which a `TorchDispatchMode` intercepts and records (counting FLOPs and replaying an operator sequence both rest on it). An inference framework's custom kernels register with `torch.library.custom_op` or C++'s `TORCH_LIBRARY`: the schema must declare in-place modification honestly (or torch.compile computes the wrong thing silently), there has to be a fake implementation, and a backward when needed. The meta device and fake tensors have shapes without data, used to build a model and plan memory without allocating, and as how the compiler infers shapes.

## Exercises {#练习}

1. Write an "operator counter" with `TorchDispatchMode` and count which aten operators `torch.nn.functional.scaled_dot_product_attention` reaches on the CPU (input `[1, 8, 128, 64]`, causal mask).

??? success "Answer"
    ```python title="sdpa_ops.py"
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode


    class CountOps(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            self.counts[name] = self.counts.get(name, 0) + 1
            return func(*args, **(kwargs or {}))


    q = k = v = torch.randn(1, 8, 128, 64)
    with CountOps() as c:
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    for name, n in sorted(c.counts.items()):
        print(name, n)
    ```

    ```text title="output"
    aten._scaled_dot_product_flash_attention_for_cpu.default 1
    ```

    On the CPU, SDPA goes straight to one fused FlashAttention implementation, a single aten operator; write `softmax(q @ k^T / sqrt(d)) @ v` by hand and you see a chain of `mm`, `div`, `masked_fill`, `softmax`, `mm`. That is exactly what fusing attention is worth.

2. A custom operator `demo::append_kv(Tensor cache, Tensor new) -> Tensor` writes `cache` in place in its implementation, but was registered with `mutates_args=()`. Everything works in eager mode. What can go wrong under `torch.compile`?

??? success "Answer"
    The compiler treats it as a pure function: the same inputs give the same output and there are no side effects. So it may merge two calls into one, move a call elsewhere, or delete the whole call when "nobody uses the result", and then the KV never reaches the cache and later decode steps read stale data, silently wrong.
    PyTorch's debug mode checks whether an input declared unmodified was modified and raises. Keeping the declaration consistent with the implementation is the single most important rule when registering a custom operator.

## Summary {#小结}

- [x] The dispatcher handles a call layer by layer on the dispatch keys: automatic mixed precision, autograd, version counting, the device kernel.
- [x] Models end up in aten operators; a `TorchDispatchMode` intercepts and records them and is the basis of many analysis tools.
- [x] Custom operators register with `torch.library.custom_op` (or C++'s `TORCH_LIBRARY`): the schema must declare in-place modification honestly, there has to be a fake implementation, and a backward when needed.
- [x] The meta device and fake tensors have shapes without data: used to build a model and plan memory without allocating, and as how `torch.compile` infers shapes.
