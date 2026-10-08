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
