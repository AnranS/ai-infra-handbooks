# 4. Using autograd

<p class="lead">autograd's surface is surprisingly small: `requires_grad`, `backward()`, `.grad`, plus `no_grad` and `detach`. What is hard are the "why"s — why gradients accumulate, why my `.grad` is None, why adding `+= 1` raises. This chapter turns each of them into an experiment you can run.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why does a training loop call `zero_grad()` every step? When should it deliberately not?
    2. How do `no_grad` and `detach` differ?
    3. In which three situations is `.grad` None?
    4. Why do some in-place operations make backward raise and others do not?
    5. What happens if you call `backward()` twice on the same loss?

??? success "Answers (try first, then expand to compare)"
    1. Gradients **accumulate** into `.grad`, so without clearing them the previous step's are added in. Gradient accumulation (several small batches making one large one) is exactly the case where you deliberately do not clear.
    2. `no_grad` is a context in which nothing builds a graph; `detach` acts on one tensor and takes it off the graph (while still sharing storage). Use the former for a whole region, the latter to cut one path.
    3. `requires_grad` was never set; the tensor is not a leaf (an intermediate result, unless `retain_grad()`); it was computed inside `no_grad`, so no graph exists.
    4. It depends on whether that operator's backward needs the value that was modified. Multiplication's backward only needs its inputs, so changing its output is fine; `sigmoid`'s backward needs its own output, so changing it breaks.
    5. It raises. The graph is freed after the first backward, unless you pass `backward(retain_graph=True)`.

## Five experiments {#五个实验}

```python title="grad.py"
"""autograd 怎么用：backward、梯度累加、no_grad 与 detach，以及"梯度为什么是 None" """
import torch

print("—— 最小的例子 ——")
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()                                            # y = x1² + x2²
y.backward()
print(f"y = {y.item():.1f}，dy/dx = 2x = {x.grad.tolist()}")
print("只有 requires_grad=True 的**叶子**张量才有 .grad")

print("\n—— 梯度是累加的，所以每一步都要清零 ——")
x.grad.zero_()
for _ in range(3):
    (x ** 2).sum().backward()
print("连着 backward 三次，梯度变成三倍：", x.grad.tolist())
print("训练循环里 optimizer.zero_grad() 就是干这个的；要做梯度累积，恰恰是**故意**不清零")

print("\n—— 计算图用完就释放 ——")
z = (x ** 2).sum()
z.backward()
try:
    z.backward()
except RuntimeError as e:
    print("再来一次会报错：", str(e).split(".")[0])
print("要多次反传同一张图，用 backward(retain_graph=True)；但多数时候这是写错了的信号")

print("\n—— no_grad 与 detach ——")
w = torch.ones(3, requires_grad=True)
with torch.no_grad():                                         # nothing inside builds a graph, which saves memory and time
    out = w * 2
print("no_grad 里算出来的结果 requires_grad =", out.requires_grad)
tmp = w * 2
d = tmp.detach()                                              # taken off the graph, but still the same storage
print("detach 之后 requires_grad =", d.requires_grad, "，和原张量共享存储 =", d.data_ptr() == tmp.data_ptr())
print("推理用 no_grad（或者更快的 inference_mode）；只想切断一条路径的梯度用 detach")

print("\n—— 三种「梯度是 None」 ——")
a = torch.ones(2)                                             # (1) requires_grad was never set
b = (a * 2).sum()
print("① 没开 requires_grad：", a.grad)
c = torch.ones(2, requires_grad=True)
mid = c * 2                                                   # (2) an intermediate result is not a leaf
mid.sum().backward()
print("② 非叶子张量：mid.is_leaf =", mid.is_leaf, "，.grad 不会被填（想看就先 mid.retain_grad()）")
e = torch.ones(2, requires_grad=True)
with torch.no_grad():                                         # (3) computed inside no_grad, so no graph was built
    f = (e * 2).sum()
print("③ 在 no_grad 里算的：f.requires_grad =", f.requires_grad, "，backward 会直接报错")

print("\n—— 原地操作会破坏反向需要的值 ——")
u = torch.tensor([1.0, 2.0], requires_grad=True)
v = (u * 3).sigmoid()                                         # sigmoid's backward needs its own output
try:
    v += 1                                                    # this modifies that very output in place
    v.sum().backward()
except RuntimeError as e:
    print("报错：", str(e).split(",")[0])
print("改成 v = v + 1 就好了。不是所有原地操作都会出事——乘法的反向只要输入，改输出没关系；")
print("但 sigmoid、exp 这类反向要用自己输出的就会炸，所以带下划线的方法（add_、relu_、scatter_）都要留心")

print("\n—— 手动验证一次梯度：和数值差分对比 ——")
def fn(t):
    return (t.sin() * t).sum()

t = torch.tensor([0.3, 1.2], requires_grad=True)
fn(t).backward()
eps = 1e-4
num = [((fn(t.detach() + eps * torch.eye(2)[i]) - fn(t.detach() - eps * torch.eye(2)[i])) / (2 * eps)).item()
       for i in range(2)]
print("autograd：", [f"{v:.5f}" for v in t.grad.tolist()], " 数值差分：", [f"{v:.5f}" for v in num])
print("写了自定义算子就该这么验一遍；torch.autograd.gradcheck 是它的正式版本")
```

```text title="output"
—— 最小的例子 ——
y = 13.0，dy/dx = 2x = [4.0, 6.0]
只有 requires_grad=True 的**叶子**张量才有 .grad

—— 梯度是累加的，所以每一步都要清零 ——
连着 backward 三次，梯度变成三倍： [12.0, 18.0]
训练循环里 optimizer.zero_grad() 就是干这个的；要做梯度累积，恰恰是**故意**不清零

—— 计算图用完就释放 ——
再来一次会报错： Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)
要多次反传同一张图，用 backward(retain_graph=True)；但多数时候这是写错了的信号

—— no_grad 与 detach ——
no_grad 里算出来的结果 requires_grad = False
detach 之后 requires_grad = False ，和原张量共享存储 = True
推理用 no_grad（或者更快的 inference_mode）；只想切断一条路径的梯度用 detach

—— 三种「梯度是 None」 ——
① 没开 requires_grad： None
② 非叶子张量：mid.is_leaf = False ，.grad 不会被填（想看就先 mid.retain_grad()）
③ 在 no_grad 里算的：f.requires_grad = False ，backward 会直接报错

—— 原地操作会破坏反向需要的值 ——
报错： one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [2]]
改成 v = v + 1 就好了。不是所有原地操作都会出事——乘法的反向只要输入，改输出没关系；
但 sigmoid、exp 这类反向要用自己输出的就会炸，所以带下划线的方法（add_、relu_、scatter_）都要留心

—— 手动验证一次梯度：和数值差分对比 ——
autograd： ['0.58212', '1.36687']  数值差分： ['0.58174', '1.36733']
写了自定义算子就该这么验一遍；torch.autograd.gradcheck 是它的正式版本
```

Point by point:

- **Accumulation is not a bug, it is the design**. When a large batch does not fit, split it, run backward on each piece, and the gradients add up naturally — that is gradient accumulation. So PyTorch will not clear them for you; clearing is `optimizer.zero_grad()`'s job;
- **The graph is freed after use**: to save memory, the intermediates are dropped after one backward. If you genuinely need two backward passes (second-order gradients, some GAN formulations), pass `retain_graph=True`; but most of the time this error means the code is wrong (a loss was stored and reused);
- **`no_grad` saves memory and time**: no graph means no saved intermediates. Always use it at inference, or memory grows linearly with the sequence length;
- **Three kinds of None**: `requires_grad` not set, not a leaf, computed inside `no_grad`. Work through those three in order and you will find it;
- **In-place depends on what backward needs**. Many tutorials put this too absolutely ("never use in-place"). The real rule is that it breaks only when the value backward needs was the one modified. `x.add_(1)` on the result of a multiplication is fine; on the result of a `sigmoid` it raises. When in doubt, do not; come back and work it out when you need the memory.

!!! tip "Wrote your own operator? Compare against finite differences once"
    Move the input by $\varepsilon$ either way and approximate the derivative with $(f(x+\varepsilon) - f(x-\varepsilon)) / 2\varepsilon$, then compare against autograd. In the example above the two agree to three decimals; the rest is the truncation error of the difference itself. The formal version is `torch.autograd.gradcheck`, which does this in float64.

!!! interview "How to explain it"
    On using autograd: **gradients accumulate**, so each step calls `zero_grad()`, and gradient accumulation is deliberately not clearing; **the graph is freed after one backward**, so repeating it needs `retain_graph=True`; **inference uses `no_grad`** (no graph, less memory and time) and **cutting one path uses `detach`**. Then the three reasons a gradient is None (requires_grad unset / not a leaf / computed inside no_grad). Finish with the point that shows depth: in-place operations are not banned outright — what matters is whether that operator's backward needs the modified value. `sigmoid` and `exp`, which save their own output, break; multiplication does not.

## Exercises {#练习}

**1. Gradient accumulation.** Split a batch into 4 parts, run backward on each and update once; check the gradient matches computing the whole batch at once. How should the loss be scaled?

??? success "Answer"
    Each part's loss must be divided by the number of parts (the default loss averages over the batch), so that the sum equals the whole batch's average gradient. `loss = loss_fn(...) / accum_steps`, then call `optimizer.step()` and `zero_grad()` every `accum_steps`.

**2. Freezing part of a model.** Setting `requires_grad = False` on the parameters, or calling `detach()` in the forward pass — what does each do? What else does the optimizer need?

??? success "An approach"
    `requires_grad=False` stops those parameters receiving gradients, but they are still in the optimizer's parameter groups (with weight decay they would still be decayed, so exclude them when building the optimizer). `detach()` cuts the **data flow**, so everything upstream receives nothing either. Freezing a backbone for fine-tuning normally uses the former.

**3. Track down a None.** Write code whose `.grad` is None, then use the three cases above to work out which one it is.

??? success "An approach"
    The most common in practice is the second: you want the gradient of an intermediate activation and find None. `h.retain_grad()` fixes it. In production code the first is more common — a parameter was rebuilt inside a `no_grad` block and lost its `requires_grad`.

## Summary {#小结}

- [x] Gradients accumulate into `.grad`, so every step calls `zero_grad()`; gradient accumulation deliberately does not.
- [x] The graph is freed after one backward; repeating it needs `retain_graph=True` (usually a sign of a bug).
- [x] `no_grad` turns off graph building for a region, `detach` for one tensor; know the three reasons `.grad` is None.
- [x] Whether an in-place operation is safe depends on whether that operator's backward needs the modified value.
