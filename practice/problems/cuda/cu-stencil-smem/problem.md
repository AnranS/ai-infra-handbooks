---
title: 一维卷积：用共享内存复用 halo
chapter: basics/memory.md
difficulty: 中等
tags: [gpusim, 共享内存, 数据复用]
---
一维模板计算（卷积）：$\text{out}[i] = \sum_{j=-R}^{R} w[j + R] \cdot \text{in}[i + j]$，越界的 `in` 视为 0。

朴素写法里每个线程从全局内存读 $2R + 1$ 个元素，相邻线程读的数据几乎完全重叠。实现 kernel `conv1d(t, inp, w, out, n, R)`：

- 启动配置：`block = 128`，`grid = cdiv(n, 128)`，每个线程计算一个输出；
- 每个 block 先把自己需要的输入（128 个元素加两侧各 $R$ 个 halo，共 $128 + 2R$ 个）**合并地**读进共享内存，`__syncthreads()` 之后再从共享内存计算；
- 权重 `w` 也先读进共享内存（它只有 $2R + 1$ 个）。

测试会检查：结果正确；**全局内存读取的扇区数**不超过朴素写法的 1/3（朴素写法对 `n = 2048`、`R = 8` 要读约 4000 个扇区，复用之后只需要约 300 个）。$R \le 32$。

<!-- 题解 -->
```python
tile = t.shared("tile", 128 + 2 * 32)          # 按最大的 R 分配
ws = t.shared("w", 2 * 32 + 1)
base = t.blockIdx.x * 128 - R                   # tile[0] 对应的全局下标
for k in range(t.threadIdx.x, 128 + 2 * R, 128):    # 每个线程搬 1～2 个，合并访问
    g = base + k
    tile[k] = inp[g] if 0 <= g < n else 0.0
if t.threadIdx.x < 2 * R + 1:
    ws[t.threadIdx.x] = w[t.threadIdx.x]
yield t.syncthreads()
```

之后 `out[i] = sum(ws[j] * tile[threadIdx.x + j] for j in range(2R + 1))`。
注意 `yield t.syncthreads()` 必须所有线程都执行，所以"`i >= n` 的线程直接返回"要放在屏障之后。
