# （一）张量：形状、dtype 与设备

<p class="lead">张量就是一块连续的数字，外加一组描述"怎么读它"的元数据。这一章把最先会咬人的三件事讲清楚：形状、dtype 和设备。重点不是 API 列表，而是几条会反复出现的规则——整数相除会变成浮点、和 NumPy 互转是共享内存的、跨设备的运算一定报错。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `torch.tensor([1, 2, 3])` 和 `torch.tensor([1.0, 2.0])` 的 dtype 分别是什么？
    2. `torch.tensor([7, 2]) / 2` 的结果是整数还是浮点？
    3. `t.numpy()` 之后改 `t`，numpy 数组会变吗？
    4. `x.to("cpu")`，当 `x` 本来就在 CPU 上时，会复制一份吗？
    5. `torch.empty(3)` 和 `torch.zeros(3)` 有什么区别？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `int64` 和 `float32`。Python 的整数推成 64 位整数，浮点推成 32 位（不是 float64，这一点和 NumPy 不同）。
    2. 浮点。`/` 永远是真除法，要整除用 `//`。
    3. 会。`numpy()` 和 `from_numpy()` 都共享同一块内存，要断开就 `.clone()`。
    4. 不会。`.to()` 在设备和 dtype 都没变时直接返回原对象，连新 Python 对象都不建。
    5. `empty` 只分配内存、不初始化，里面是上一次留下的值；`zeros` 会写一遍 0。想要未初始化的缓冲区才用 `empty`。

## 三件套：形状、dtype、设备

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
y = x.to(dev)                                                 # .to 在同设备同 dtype 时直接返回自己，不复制
print(f"x.to({dev!r}) 之后 device={y.device}，是不是同一个对象：{y is x}")
print("跨设备的运算会直接报错，不会偷偷帮你搬——这是故意的，搬运很贵")
```

```text title="输出"
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

逐条看几个容易踩的地方：

- **stride 是"沿着这一维走一步要跳几个元素"**：`(2,3,4)` 的张量 stride 是 `(12,4,1)`，所以最后一维是连着的。下一章的 `permute` 会把它打乱，很多"为什么不能 view"的问题都在这里；
- **浮点默认 float32**：写惯 NumPy 的人常以为是 float64。模型权重、激活几乎都用 float32 或更低，这个默认值是对的；
- **整数除法**：`/` 得到浮点，`//` 才是整除。算 `n_heads` 这类下标时用错会得到 `1.0` 这样的浮点数，后面索引直接报错；
- **和 NumPy 共享内存**：`t.numpy()` 不复制。这是好事（省一次拷贝），但改一个另一个会跟着变，调试时足以让人怀疑人生。要独立的一份就 `.clone()`；
- **`.to()` 不保证复制**：同设备同 dtype 时返回的就是原对象。想要一份独立的拷贝，用 `.clone()`，别指望 `.to()`。

!!! tip "只有 CPU 也不影响学"
    这本书的代码全部在 CPU 上跑。GPU 相关的只有一条规则要记住：**张量在哪个设备上，运算就必须在哪个设备上**，PyTorch 不会偷偷帮你搬——搬运一次要走 PCIe，比大多数算子本身还贵，所以它宁可报错也不替你决定。

!!! interview "怎么讲清楚"
    讲"张量和 NumPy 数组有什么区别"：张量多了三样东西——**设备**（能放在 GPU 上）、**autograd**（能记录计算图求梯度）、**dtype 更克制**（默认 float32，因为深度学习不需要 float64）。再补一句互转是**共享内存**的，所以零拷贝但要小心别名；以及 `.to()` 在无需转换时不复制，真要副本得 `.clone()`。

## 练习

**1. 猜 dtype。** 不运行，先说出 `torch.tensor([1, 2]) + torch.tensor([1.5])`、`torch.ones(2, dtype=torch.int32) * 2.0`、`torch.arange(3) / 1` 各是什么 dtype，再跑一遍验证。

??? success "参考答案"
    依次是 `float32`、`float32`、`float32`。PyTorch 的类型提升规则是：整数遇到浮点升到浮点；不同宽度的同类相遇升到宽的那个；Python 标量不参与"升级"，只按张量的类别走。`/` 的结果永远是浮点，即使两边都是整数。

**2. 别名。** 写一个函数，接收一个张量并把它所有负数置零。分别用 `x[x < 0] = 0` 和 `x = torch.clamp(x, min=0)` 实现，调用之后检查调用方的张量有没有被改掉。

??? success "参考思路"
    第一种是原地修改，调用方的张量被改了；第二种返回新张量，调用方不受影响。库函数应该明确自己属于哪一种：PyTorch 的约定是**带下划线的才原地**（`clamp_`），不带的一律返回新张量。

**3. 省内存。** 一个 `(1024, 1024)` 的 float32 张量占多少 MiB？换成 bfloat16 呢？用 `element_size()` 和 `numel()` 验证一遍。

??? success "参考答案"
    4 MiB 和 2 MiB。`numel() * element_size()` 就是字节数。推理时把权重换成 bf16/fp16 能省一半显存，也能快——但要注意梯度和优化器状态通常仍然留在 fp32。

## 小结

- [x] 张量 = 一块数据 + 形状、stride、dtype、设备这组元数据；`numel() * element_size()` 就是它占的字节数。
- [x] Python 整数推成 `int64`，浮点推成 `float32`；`/` 永远是真除法。
- [x] 和 NumPy 互转共享内存，`.to()` 在不需要转换时不复制——要独立的副本就 `.clone()`。
- [x] 张量在哪个设备上，运算就得在哪个设备上；PyTorch 不会偷偷帮你搬。
