---
title: INT4 权重的 GEMV：一个 warp 算一行
chapter: advanced/quantization.md
difficulty: 中等
tags: [gpusim, GEMV, 量化, warp shuffle]
---
decode 阶段的线性层是 GEMV：$y = W x$，$W$ 形状 `(N, K)`。权重用 4 位对称量化存储：

- `packed`：`(N, K // 2)` 的 `uint8`，第 `j` 个字节的低 4 位是第 `2j` 列、高 4 位是第 `2j + 1` 列；
- `scales`：`(N,)` 的 float32，每行一个缩放；反量化 $w_{n,k} = (q_{n,k} - 8) \cdot \text{scale}_n$。

用 gpusim 实现 kernel `gemv_w4(t, packed, scales, x, y, N, K)`：`block = (32, 4)`（4 个 warp），`grid = cdiv(N, 4)`，**每个 warp 负责一行**：

- lane `l` 处理第 `l, l + 32, l + 64, ...` 个字节（相邻 lane 读相邻字节，合并访问），每个字节解出两个权重，乘上对应的 `x` 累加；
- warp 内用 `shfl_down` 求和，lane 0 写 `y[n] = sum * scale[n]`。

测试检查：结果正确（`K` 是 2 的倍数，可以不是 64 的倍数）；读权重 `packed` 的全局内存效率 $\ge 90\%$（`stats.array("packed")`）。模板里是"一个线程算一行"的写法：结果对，但相邻线程读的是相隔 `K/2` 字节的地址，读效率只有约 3%。

<!-- 题解 -->
```python
lane, row = t.threadIdx.x, t.blockIdx.x * 4 + t.threadIdx.y
acc = 0.0
if row < N:
    for j in range(lane, K // 2, 32):
        b = int(packed[row, j])
        acc += ((b & 0xF) - 8) * x[2 * j] + ((b >> 4) - 8) * x[2 * j + 1]
s = yield from warp_sum(t, acc)          # 超出 N 的那几个 warp 也要参与 shuffle（acc = 0）
if lane == 0 and row < N:
    y[row] = s * scales[row]
```

GEMV 的算术强度只有约 1 FLOP/Byte，完全是带宽瓶颈：权重从 16 位变成 4 位，读的字节数少了 4 倍，速度也就接近快 4 倍——这是 decode 阶段做权重量化的主要收益。
