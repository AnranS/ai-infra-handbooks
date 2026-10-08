# autograd: the computation graph and how backpropagation works

<p class="lead">Inference needs no backpropagation, but an inference engineer cannot avoid autograd: RL training shares one code base between inference and training, a custom kernel needs a backward, activation recomputation and memory estimates both rest on "which tensors the forward pass saved", and the difference between <code>inference_mode</code> and <code>no_grad</code> comes up often. This chapter looks at what autograd actually records at run time.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. After `y = (x * w).sum()` runs, what has autograd left in memory?
    2. Which tensors exactly are the "activations" a training forward pass saves? How do you count them?
    3. What does activation recomputation (gradient checkpointing) trade for what?
    4. When do you need to write a `torch.autograd.Function`? How do you check the backward is right?
    5. How do `torch.inference_mode()` and `torch.no_grad()` differ?

??? success "Answers (try it yourself first, then expand)"
    1. The result `y` carries `grad_fn = SumBackward`, whose `next_functions` points at `MulBackward`, which saved both inputs `x` and `w` (a multiplication's backward needs the other operand), and below that are `x`'s and `w`'s `AccumulateGrad` nodes.
    2. The tensors the backward nodes saved: a linear layer saves its input, an activation saves its input or its output. `torch.autograd.graph.saved_tensors_hooks` records each tensor's shape and byte count as it is packed, which counts them exactly (weights are only referenced and cost nothing extra).
    3. It saves only each segment's input, discards the intermediate activations, and redoes that segment's forward pass during the backward: roughly one extra forward pass in exchange for a large cut in activation memory.
    4. When PyTorch does not know how to differentiate your computation (your own CUDA / Triton kernel, or a numerically stable backward you want to specify). Check it against finite differences in float64 with `torch.autograd.gradcheck`.
    5. `no_grad` merely stops recording the graph; `inference_mode` also turns off version counting and view tracking, costing less, but the tensors it produces cannot take part in a later differentiable computation.

## The graph recorded during the forward pass {#前向时记录的计算图}

![Figure: the forward pass records a graph and the backward propagates through it by the chain rule](../assets/figures/autograd-graph.svg){.aig-svg}

Operating on a tensor with `requires_grad=True` creates a **backward node** (`grad_fn`) for each operation, recording how to compute the gradient, which tensors the backward needs (the saved tensors), and the backward nodes of its inputs (`next_functions`). The whole graph is built **dynamically** during the forward pass, and the backward walks from the output along `next_functions`:

```python title="graph_walk.py"
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
y = (x * w).sum()


def walk(fn, depth=0):
    if fn is None:
        return
    print("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1)


walk(y.grad_fn)
mul = y.grad_fn.next_functions[0][0]
print("MulBackward0 保存了：", sorted(a for a in dir(mul) if a.startswith("_saved")))
y.backward()
print("x.grad =", x.grad.tolist(), "w.grad =", w.grad.tolist())
```

```text title="output"
SumBackward0
  MulBackward0
    AccumulateGrad
    AccumulateGrad
MulBackward0 保存了： ['_saved_other', '_saved_self']
x.grad = [0.5, -1.0, 2.0] w.grad = [1.0, 2.0, 3.0]
```

- a multiplication's backward needs the other operand ($\partial (xw)/\partial x = w$), so `MulBackward0` saved both inputs;
- a leaf tensor (one the user created with `requires_grad=True`) has an `AccumulateGrad` node that **adds** the gradient into `.grad`, which is why a training loop zeroes the gradients each step;
- after the backward the graph is released by default (and the saved tensors with it); backpropagating again needs `retain_graph=True`.

## Activation memory: what the forward pass saved {#激活显存前向保存了什么}

"How much memory the activations take" during training has an exact answer: "which tensors the backward nodes saved". `torch.autograd.graph.saved_tensors_hooks` intercepts every save, which counts the bytes, and can also offload activations to the CPU (`save_on_cpu` is implemented this way).
**Activation recomputation** (`torch.utils.checkpoint`) instead saves only a segment's input and redoes that forward pass during the backward:

```python title="saved_bytes.py"
import torch
from torch.utils.checkpoint import checkpoint


def mlp(x, w1, w2):
    return torch.nn.functional.silu(x @ w1) @ w2


saved = []


def pack(t):
    saved.append(t.numel() * t.element_size())
    return t


def unpack(t):
    return t


torch.manual_seed(0)
X = torch.randn(64, 256, requires_grad=True)
W1 = torch.randn(256, 1024, requires_grad=True)
W2 = torch.randn(1024, 256, requires_grad=True)

with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    out = mlp(X, W1, W2)
print(f"普通前向：保存了 {len(saved)} 个张量，共 {sum(saved)} 字节")

saved.clear()
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    out2 = checkpoint(mlp, X, W1, W2, use_reentrant=False)
print(f"激活重计算：保存了 {len(saved)} 个张量，共 {sum(saved)} 字节")
out2.sum().backward()
print("反向照常完成，X.grad 形状", tuple(X.grad.shape))
```

```text title="output"
普通前向：保存了 5 个张量，共 2686976 字节
激活重计算：保存了 3 个张量，共 2162688 字节
反向照常完成，X.grad 形状 (64, 256)
```

Item by item: the ordinary forward pass saved `X` (64 KB), `W1` (1 MB), the result of `x @ w1` (for SiLU's backward, 256 KB), SiLU's output (for the second matmul's backward, 256 KB) and `W2` (1 MB).
The weights are in memory anyway and what is saved is only a reference, costing nothing extra; what really grows is the two `64 × 1024` intermediate activations. With recomputation, only the segment's three inputs are saved and the two intermediates are gone, at the cost of one extra forward pass during the backward.
In an LLM the activations are proportional to `batch × seq × hidden`, far larger than this example, which is why long-sequence training cannot do without recomputation (see the LLM handbook's exercise on [the training memory ledger](root://practice/#/p/llm-est-train-memory)).

## A custom `autograd.Function` {#自定义-autogradfunction}

Two situations call for writing your own backward:

- the forward uses an implementation autograd cannot see into (your own CUDA / Triton kernel, a C++ extension);
- you want a backward that is cheaper in memory or faster than the derived one (a fused kernel saving only its inputs and recomputing the intermediates in the backward).

Below, SwiGLU's "`silu(gate) * up`" becomes one fused kernel that saves only the two inputs, with the backward derived by hand:

$$\frac{\partial}{\partial g}\big(\mathrm{silu}(g) \cdot u\big) = u \cdot \sigma(g)\,\big(1 + g\,(1 - \sigma(g))\big), \qquad \frac{\partial}{\partial u} = \mathrm{silu}(g)$$

```python title="swiglu_function.py"
import torch


class SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)          # only the inputs are saved; the intermediates are recomputed during the backward
        return torch.nn.functional.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        sig = torch.sigmoid(gate)
        silu = gate * sig
        d_gate = grad_out * up * sig * (1 + gate * (1 - sig))
        d_up = grad_out * silu
        return d_gate, d_up


torch.manual_seed(0)
g = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
u = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
print("gradcheck：", torch.autograd.gradcheck(SiluMul.apply, (g, u)))

y = SiluMul.apply(g, u)
ref = torch.nn.functional.silu(g) * u
print("前向与参考实现一致：", torch.allclose(y, ref))
print("反向节点：", type(y.grad_fn).__name__)
```

```text title="output"
gradcheck： True
前向与参考实现一致： True
反向节点： SiluMulBackward
```

`gradcheck` verifies the analytic gradient against finite differences in `float64`, and **every custom backward should be run through it**. Using `ctx.save_for_backward` rather than `ctx.gate = gate` is what lets autograd check "was this tensor modified in place after it was saved".
Training-acceleration libraries like Liger Kernel and Unsloth are exactly this pattern of a fused forward with a hand-written backward, implemented in Triton.

## `no_grad` and `inference_mode` {#no_grad-与-inference_mode}

Inference does not need the graph. Both context managers turn autograd off; they differ in **the tensors they produce**:

```python title="inference_mode.py"
import torch

x = torch.randn(4, requires_grad=True)
with torch.no_grad():
    a = x * 2
with torch.inference_mode():
    b = x * 2
print("no_grad：requires_grad", a.requires_grad, "是 inference 张量", a.is_inference())
print("inference_mode：requires_grad", b.requires_grad, "是 inference 张量", b.is_inference())
try:
    (b * x).sum().backward()
except RuntimeError as e:
    print("把 inference 张量用在需要求导的计算里：", str(e).split(".")[0])
```

```text title="output"
no_grad：requires_grad False 是 inference 张量 False
inference_mode：requires_grad False 是 inference 张量 True
把 inference 张量用在需要求导的计算里： Inference tensors cannot be saved for backward
```

`inference_mode` goes further: the tensors it produces carry no version counter and cannot take part in later autograd, which saves that bookkeeping, and inference frameworks wrap their forward passes in it.
RL training needs particular care: a tensor generated during the rollout under `inference_mode` cannot go straight into the training phase's forward pass and needs a `clone()` into an ordinary tensor.

!!! interview "How to explain it"
    On autograd: the forward pass creates a backward node per operation recording the saved tensors and the edges to the inputs, the backward walks that graph from the output, and leaf tensors accumulate into `.grad` (hence zeroing each step); "activation memory" is exactly the tensors the backward nodes saved, countable with `saved_tensors_hooks`; and activation recomputation trades one extra forward pass for memory. Your own kernel needs an `autograd.Function` for its backward, verified in float64 with `gradcheck`. Inference uses `inference_mode`, which saves even the version counting that `no_grad` keeps.

## Exercises {#练习}

1. Count with `saved_tensors_hooks`: for a forward pass through `nn.Linear(4096, 4096)` and `GELU` (input `[8, 4096]`, float32), which tensors are saved and how many bytes each? Which are the "extra" activations?

??? success "Answer"
    ```python title="linear_gelu_saved.py"
    import torch

    records = []


    def pack(t):
        records.append((tuple(t.shape), t.numel() * t.element_size()))
        return t


    torch.manual_seed(0)
    lin = torch.nn.Linear(4096, 4096)
    x = torch.randn(8, 4096, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = torch.nn.functional.gelu(lin(x))
    for shape, n in records:
        print(shape, n)
    ```

    ```text title="output"
    (8, 4096) 131072
    (4096, 4096) 67108864
    (8, 4096) 131072
    ```

    The linear layer saved its input `x` (for the weight's gradient) and the weight (for the input's gradient, only a reference), and GELU saved the linear layer's output (for its own gradient). The extra activations are two `8 × 4096` tensors of 128 KB each.

2. In a custom `Function`'s `backward`, what happens if you modify the `gate` from `ctx.saved_tensors` in place (with `gate.mul_(2)`, say)?

??? success "Answer"
    `gate` is the forward's input, which the caller may still hold and another backward node may have saved. Modifying it in place quietly changes someone else's data and gives wrong gradients. autograd uses a "version counter" to check whether a saved tensor was modified in place, and if it was after the forward pass, the backward raises "one of the variables needed for gradient computation has been modified by an inplace operation".
    Inside `backward`, use only out-of-place operations, or `clone()` first.

## Summary {#小结}

- [x] The forward pass creates a backward node per operation recording the saved tensors and the edges to the inputs; the backward walks that graph and the leaves accumulate into `.grad`.
- [x] "Activation memory" is exactly the tensors the backward nodes saved, countable with `saved_tensors_hooks`; weights are only referenced.
- [x] Activation recomputation saves only the inputs and recomputes during the backward, trading one extra forward pass for memory.
- [x] Your own kernel needs an `autograd.Function` for its backward, verified in float64 with `gradcheck`.
- [x] Inference uses `inference_mode`: cheaper than `no_grad` by the version counting, but its tensors cannot be differentiated later.
