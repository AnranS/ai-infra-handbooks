---
title: Triton 融合 softmax
chapter: tools/triton.md
difficulty: 中等
tags: [Triton, 融合, mask]
---
用 Triton 写一个逐行 softmax kernel：每个 program 处理一行，一次 `tl.load` 把整行读进来，在"块"上算完 max、exp、sum，再一次 `tl.store` 写出——输入只读一遍、输出只写一遍（PyTorch 的非融合实现要读写好几遍）。

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, x_stride, y_stride, BLOCK: tl.constexpr): ...

def softmax(x):          # x: 二维、行主序、连续的数组；返回同形状的结果
    ...
```

- `BLOCK` 取 `triton.next_power_of_2(n_cols)`（`tl.arange` 的长度必须是 2 的幂）；列数不是 2 的幂时，多出来的位置用 `mask` 屏蔽，并且 `other=-inf`，保证它们不影响 max 和 sum；
- `softmax(x)` 分配输出、以 `grid = (rows,)` 启动 kernel、返回结果。

在浏览器和 macOS 上，判题器用模拟器 minitl 运行（写法和真 Triton 一样，还会检查越界访问）；在 WSL2 + NVIDIA GPU 上用真 Triton。
测试里用 `tritonkit.to_dev` / `to_host` 在 numpy 数组和 CUDA 张量之间转换，你的 `softmax` 函数里请用 `tritonkit.empty_like(x)` 分配输出。

<!-- 题解 -->
```python
row = tl.program_id(0)
offs = tl.arange(0, BLOCK)
mask = offs < n_cols
x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=-float("inf"))
x = x - tl.max(x, axis=0)
num = tl.exp(x)
y = num / tl.sum(num, axis=0)
tl.store(y_ptr + row * y_stride + offs, y, mask=mask)
```

被屏蔽的位置 `exp(-inf) = 0`，不影响求和。这个写法要求一整行能放进一个块（几千列没问题）；更长的行要像 online softmax 那样分块循环。
