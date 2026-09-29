# autograd：计算图与反向传播的实现

<p class="lead">推理不需要反向传播，但推理工程师躲不开 autograd：RL 训练里推理和训练共用一套代码，自定义算子要写反向，激活重计算、显存估算都建立在"前向保存了哪些张量"之上，<code>inference_mode</code> 和 <code>no_grad</code> 的区别也常被问到。这一章看 autograd 在运行时到底记录了什么。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `y = (x * w).sum()` 执行完之后，autograd 在内存里留下了什么？
    2. 训练时前向保存的"激活"具体是哪些张量？怎么数出来？
    3. 激活重计算（gradient checkpointing）用什么换什么？
    4. 什么时候需要写 `torch.autograd.Function`？怎么验证反向写对了？
    5. `torch.inference_mode()` 和 `torch.no_grad()` 有什么区别？

## 前向时记录的计算图

对 `requires_grad=True` 的张量做运算时，每个运算会创建一个**反向节点**（`grad_fn`），记下：怎么计算梯度、反向时需要的张量（saved tensors）、以及指向输入的反向节点（`next_functions`）。整张图在前向时**动态**建立，反向时从输出沿着 `next_functions` 走回去：

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

```text title="输出"
SumBackward0
  MulBackward0
    AccumulateGrad
    AccumulateGrad
MulBackward0 保存了： ['_saved_other', '_saved_self']
x.grad = [0.5, -1.0, 2.0] w.grad = [1.0, 2.0, 3.0]
```

- 乘法的反向需要另一个操作数（$\partial (xw)/\partial x = w$），所以 `MulBackward0` 保存了两个输入；
- 叶子张量（用户创建的、`requires_grad=True` 的张量）对应 `AccumulateGrad` 节点，反向时把梯度**累加**到 `.grad` 上——所以训练循环里每一步要清零梯度；
- 反向结束后，图默认被释放（保存的张量也随之释放）；需要再反向一次时要 `retain_graph=True`。

## 激活显存：前向保存了什么

训练时"激活占多少显存"，精确的答案就是"反向节点保存了哪些张量"。`torch.autograd.graph.saved_tensors_hooks` 可以拦截每一次保存，用来数字节——也可以用来把激活卸载到 CPU（`save_on_cpu` 就是这么实现的）。
**激活重计算**（`torch.utils.checkpoint`）则只保存一段计算的输入，反向时把这段前向重新算一遍：

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

```text title="输出"
普通前向：保存了 5 个张量，共 2686976 字节
激活重计算：保存了 3 个张量，共 2162688 字节
反向照常完成，X.grad 形状 (64, 256)
```

逐个对一下：普通前向保存了 `X`（64 KB）、`W1`（1 MB）、`x @ w1` 的结果（给 SiLU 的反向用，256 KB）、SiLU 的输出（给第二个矩阵乘的反向用，256 KB）、`W2`（1 MB）。
权重本来就在显存里，保存的只是引用，不额外占内存；真正多出来的是两个 `64 × 1024` 的中间激活。重计算之后只保存这段计算的三个输入，两个中间激活没有了——代价是反向时多做一次前向。
在大模型里激活和 `batch × seq × hidden` 成正比，远大于这个例子，这就是长序列训练离不开重计算的原因（见大模型手册的练习[训练显存账本](root://practice/#/p/llm-est-train-memory)）。

## 自定义 `autograd.Function`

两种情况需要自己写反向：

- 前向用了 autograd 看不懂的实现（自己写的 CUDA / Triton kernel、C++ 扩展）；
- 想要比自动推导更省显存或更快的反向（比如融合算子只保存输入、反向时重算中间结果）。

下面把 SwiGLU 的"`silu(gate) * up`"写成一个融合算子，前向只保存两个输入，反向手工推导：

$$\frac{\partial}{\partial g}\big(\mathrm{silu}(g) \cdot u\big) = u \cdot \sigma(g)\,\big(1 + g\,(1 - \sigma(g))\big), \qquad \frac{\partial}{\partial u} = \mathrm{silu}(g)$$

```python title="swiglu_function.py"
import torch


class SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)          # 只保存输入，中间结果反向时重算
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

```text title="输出"
gradcheck： True
前向与参考实现一致： True
反向节点： SiluMulBackward
```

`gradcheck` 用有限差分在 `float64` 下数值验证解析梯度，**写任何自定义反向都应该跑一遍**。`ctx.save_for_backward` 而不是直接 `ctx.gate = gate`，是为了让 autograd 能检查"保存之后张量是否被原地修改了"。
Liger Kernel、Unsloth 这类训练加速库就是把这种融合前向 + 手写反向的算子用 Triton 实现了一遍。

## `no_grad` 与 `inference_mode`

推理时不需要建图。两个上下文管理器都能关掉 autograd，区别在于**产出的张量**：

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

```text title="输出"
no_grad：requires_grad False 是 inference 张量 False
inference_mode：requires_grad False 是 inference 张量 True
把 inference 张量用在需要求导的计算里： Inference tensors cannot be saved for backward
```

`inference_mode` 更进一步：它产出的张量不记录版本计数、不能参与之后的 autograd，于是省掉了这部分簿记开销，推理框架的前向都包在它里面。
RL 训练里要特别注意：推理阶段（rollout）在 `inference_mode` 下生成的张量，不能直接拿去做训练阶段的前向，需要 `clone()` 出一份普通张量。

!!! interview "面试怎么答"
    autograd 题：前向时每个运算创建一个反向节点，记下保存的张量和指向输入的边，反向从输出沿这张图走回去，叶子张量把梯度累加到 `.grad`（所以每步要清零）；"激活显存"就是反向节点保存的张量，可以用 `saved_tensors_hooks` 精确计数；激活重计算用一次额外的前向换显存。自己写的 kernel 要用 `autograd.Function` 提供反向，并在 float64 下用 `gradcheck` 验证。推理用 `inference_mode`，比 `no_grad` 还省掉了版本计数。

## 练习

1. 用 `saved_tensors_hooks` 数一下：一个 `nn.Linear(4096, 4096)` 加 `GELU` 的前向（输入 `[8, 4096]`，float32），保存了哪些张量、各多少字节？哪些是"额外"的激活？

??? success "参考答案"
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

    ```text title="输出"
    (8, 4096) 131072
    (4096, 4096) 67108864
    (8, 4096) 131072
    ```

    线性层保存了输入 `x`（给权重的梯度用）和权重（给输入的梯度用，只是引用），GELU 保存了线性层的输出（给自己的梯度用）。额外的激活是两个 `8 × 4096` 的张量，各 128 KB。

2. 自定义 `Function` 的 `backward` 里，如果把 `ctx.saved_tensors` 取出来的 `gate` 原地修改（比如 `gate.mul_(2)`），会发生什么？

??? success "参考答案"
    `gate` 是前向的输入，调用者可能还持有它，也可能被别的反向节点保存着。原地修改会悄悄改掉别人的数据，得到错误的梯度。autograd 靠"版本计数器"检查保存后的张量有没有被原地修改，前向之后如果被改过，反向时会报 "one of the variables needed for gradient computation has been modified by an inplace operation"。
    在 `backward` 里应该只做非原地运算，或者先 `clone()`。

## 小结

- [x] 前向时每个运算创建一个反向节点，记录保存的张量和指向输入的边；反向沿这张图走回去，叶子节点把梯度累加到 `.grad`。
- [x] "激活显存"就是反向节点保存的张量，可以用 `saved_tensors_hooks` 精确计数；权重只是引用。
- [x] 激活重计算只保存输入，反向时重算，用一次额外的前向换显存。
- [x] 自己写的 kernel 需要 `autograd.Function` 提供反向，用 `gradcheck` 在 float64 下验证。
- [x] 推理用 `inference_mode`：比 `no_grad` 省掉版本计数，但产出的张量不能再参与求导。
