# 1. Tensors: shape, dtype and device

<p class="lead">A tensor is a block of numbers plus some metadata describing how to read it. This chapter settles the three things that bite first: shape, dtype and device. The point is not an API list but a handful of rules that keep coming back — integer division turns into floating point, converting to and from NumPy shares memory, and an operation across devices always fails.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What are the dtypes of `torch.tensor([1, 2, 3])` and `torch.tensor([1.0, 2.0])`?
    2. Is `torch.tensor([7, 2]) / 2` an integer or a float?
    3. After `t.numpy()`, if you modify `t`, does the NumPy array change?
    4. Does `x.to("cpu")` copy when `x` is already on the CPU?
    5. What is the difference between `torch.empty(3)` and `torch.zeros(3)`?

??? success "Answers (try first, then expand to compare)"
    1. `int64` and `float32`. Python integers become 64-bit integers, floats become 32-bit (not float64, unlike NumPy).
    2. A float. `/` is always true division; use `//` for integer division.
    3. Yes. Both `numpy()` and `from_numpy()` share one block of memory; `.clone()` to break the link.
    4. No. `.to()` returns the original object when neither the device nor the dtype changes — it does not even create a new Python object.
    5. `empty` allocates without initialising, so it holds whatever was there before; `zeros` writes zeros. Use `empty` only when you want an uninitialised buffer.

## The three: shape, dtype, device {#三件套形状dtype设备}

```python title="basics.py"
"""张量的三件套：形状、dtype、设备。顺便看看哪些操作是共享内存的"""
import numpy as np
import torch

g = torch.Generator().manual_seed(0)
x = torch.randn(2, 3, 4, generator=g)
print(f"shape={tuple(x.shape)}  dtype={x.dtype}  device={x.device}  "
      f"numel={x.numel()}  每个元素 {x.element_size()} 字节  共 {x.numel() * x.element_size()} 字节")
print(f"stride={x.stride()}  连续={x.is_contiguous()}")

print("\n—— 几种构造方式 ——")
print("torch.tensor([1, 2, 3])      ", torch.tensor([1, 2, 3]).dtype, "  从 Python 整数推出来的是 int64")
print("torch.tensor([1.0, 2.0])     ", torch.tensor([1.0, 2.0]).dtype, "  浮点默认 float32，不是 float64")
print("torch.zeros(2, 3).dtype      ", torch.zeros(2, 3).dtype)
print("torch.arange(5)              ", torch.arange(5).tolist(), torch.arange(5).dtype)
print("torch.full((2,), 3.0).dtype  ", torch.full((2,), 3.0).dtype)
print("torch.empty(3) 只分配不初始化，里面是上一次留下的垃圾值，别当零用")

print("\n—— dtype 的坑 ——")
a = torch.tensor([7, 2])
print("整数相除得到的是 float：", (a / 2).dtype, (a / 2).tolist())
print("要整除用 //（floor_divide）：", (a // 2).tolist())
print("int 和 float 相加，按类型提升规则走：", (a + 0.5).dtype)
half = x.to(torch.float16)
print(f"float32 -> float16 后每个元素 {half.element_size()} 字节，最大误差 {(half.float() - x).abs().max():.3e}")

print("\n—— 和 NumPy 互转：共享同一块内存 ——")
t = torch.ones(3)
n = t.numpy()
t[0] = 99
print("改 tensor，numpy 跟着变：", n)
n2 = np.arange(3, dtype=np.float32)
t2 = torch.from_numpy(n2)
n2[0] = -1
print("改 numpy，tensor 也跟着变：", t2.tolist())
print("要断开就复制一份：", torch.from_numpy(n2).clone().tolist())

print("\n—— 设备 ——")
print("有没有 CUDA：", torch.cuda.is_available(), " 当前默认设备：", torch.empty(0).device)
dev = "cuda" if torch.cuda.is_available() else "cpu"
y = x.to(dev)                                                 # .to returns the tensor itself when the device and dtype already match; no copy
print(f"x.to({dev!r}) 之后 device={y.device}，是不是同一个对象：{y is x}")
print("跨设备的运算会直接报错，不会偷偷帮你搬——这是故意的，搬运很贵")
```

```text title="output"
shape=(2, 3, 4)  dtype=torch.float32  device=cpu  numel=24  每个元素 4 字节  共 96 字节
stride=(12, 4, 1)  连续=True

—— 几种构造方式 ——
torch.tensor([1, 2, 3])       torch.int64   从 Python 整数推出来的是 int64
torch.tensor([1.0, 2.0])      torch.float32   浮点默认 float32，不是 float64
torch.zeros(2, 3).dtype       torch.float32
torch.arange(5)               [0, 1, 2, 3, 4] torch.int64
torch.full((2,), 3.0).dtype   torch.float32
torch.empty(3) 只分配不初始化，里面是上一次留下的垃圾值，别当零用

—— dtype 的坑 ——
整数相除得到的是 float： torch.float32 [3.5, 1.0]
要整除用 //（floor_divide）： [3, 1]
int 和 float 相加，按类型提升规则走： torch.float32
float32 -> float16 后每个元素 2 字节，最大误差 8.788e-04

—— 和 NumPy 互转：共享同一块内存 ——
改 tensor，numpy 跟着变： [99.  1.  1.]
改 numpy，tensor 也跟着变： [-1.0, 1.0, 2.0]
要断开就复制一份： [-1.0, 1.0, 2.0]

—— 设备 ——
有没有 CUDA： False  当前默认设备： cpu
x.to('cpu') 之后 device=cpu，是不是同一个对象：True
跨设备的运算会直接报错，不会偷偷帮你搬——这是故意的，搬运很贵
```

A few places where people trip:

- **A stride is "how many elements to jump to move one step along this dimension"**: a `(2,3,4)` tensor has strides `(12,4,1)`, so the last dimension is contiguous. `permute` in the next chapter scrambles this, and most "why can't I view this" questions start here;
- **Floats default to float32**: people used to NumPy expect float64. Weights and activations are float32 or narrower, so this default is the right one;
- **Integer division**: `/` gives a float, `//` is integer division. Using the wrong one for something like `n_heads` gives `1.0`, and indexing with it fails straight away;
- **Shared memory with NumPy**: `t.numpy()` does not copy. That is a good thing (one copy saved), but modifying one changes the other, which is enough to make you doubt your sanity while debugging. `.clone()` when you want an independent copy;
- **`.to()` does not guarantee a copy**: with the same device and dtype it returns the original object. Use `.clone()` for a copy, not `.to()`.

!!! tip "A CPU is enough to learn from"
    All the code in this book runs on a CPU. There is only one GPU-related rule to remember: **an operation happens on the device its tensors live on**, and PyTorch will not quietly move them for you — a transfer crosses PCIe and costs more than most operators, so it would rather raise than decide for you.

!!! interview "How to explain it"
    To explain "how a tensor differs from a NumPy array": it adds three things — a **device** (it can live on a GPU), **autograd** (it can record a graph and produce gradients), and a **more restrained dtype policy** (float32 by default, because deep learning does not need float64). Add that conversion is **zero-copy and shares memory**, so watch for aliasing; and that `.to()` does not copy when nothing needs converting, so `.clone()` is what gives you a real copy.

## Exercises {#练习}

**1. Guess the dtype.** Without running it, say the dtype of `torch.tensor([1, 2]) + torch.tensor([1.5])`, `torch.ones(2, dtype=torch.int32) * 2.0` and `torch.arange(3) / 1`, then check.

??? success "Answer"
    `float32` in all three cases. PyTorch's promotion rules: an integer meeting a float goes to float; two of the same kind go to the wider one; Python scalars do not force a promotion, they follow the tensor's category. `/` always produces a float, even between two integers.

**2. Aliasing.** Write a function that zeroes the negative entries of a tensor, once with `x[x < 0] = 0` and once with `x = torch.clamp(x, min=0)`, and check afterwards whether the caller's tensor changed.

??? success "An approach"
    The first modifies in place and the caller's tensor changes; the second returns a new tensor and leaves the caller alone. A library function should be explicit about which it is: PyTorch's convention is that **only the trailing-underscore versions are in place** (`clamp_`); everything else returns a new tensor.

**3. Saving memory.** How many MiB does a `(1024, 1024)` float32 tensor take? And in bfloat16? Verify with `element_size()` and `numel()`.

??? success "Answer"
    4 MiB and 2 MiB. `numel() * element_size()` is the byte count. Casting weights to bf16/fp16 halves memory at inference and is faster too — but note that gradients and optimizer states usually stay in fp32.

## Summary {#小结}

- [x] A tensor is a block of data plus shape, strides, dtype and device; `numel() * element_size()` is what it occupies.
- [x] Python integers become `int64`, floats become `float32`; `/` is always true division.
- [x] Conversion to and from NumPy shares memory, and `.to()` does not copy when nothing changes — `.clone()` for a real copy.
- [x] An operation happens on the device its tensors live on; PyTorch will not move them for you.
