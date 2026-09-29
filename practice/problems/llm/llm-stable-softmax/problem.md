---
title: 数值稳定的 softmax 与 logsumexp
chapter: basics/math-torch.md
difficulty: 简单
tags: [numpy, 数值稳定, 广播]
---
用 numpy 实现三个函数，都沿着 `axis`（默认最后一维）计算，输入是任意形状的浮点数组：

- `logsumexp(x, axis=-1)`：$\log \sum_j e^{x_j}$；
- `softmax(x, axis=-1)`；
- `log_softmax(x, axis=-1)`。

要求：

1. **数值稳定**：`x` 里有 `1000` 这样的大数时不能溢出成 `inf` 或 `nan`；
2. 支持 `-inf`（被掩码的位置）：`softmax([0, -inf])` 是 `[1, 0]`；如果一整行都是 `-inf`，`logsumexp` 返回 `-inf`，`softmax` 返回全 0、不出现 `nan`；
3. 不能用 `for` 循环遍历元素（用广播），不能调用 scipy。

<!-- 题解 -->
减去最大值：$\log\sum e^{x_j} = m + \log\sum e^{x_j - m}$，$m = \max_j x_j$，这样指数的最大值是 $e^0 = 1$，不会溢出。
`keepdims=True` 让最大值能和原数组广播。

整行都是 `-inf` 时 $m = -\infty$，`x - m` 是 `nan`：把这种行的 `m` 替换成 0（`np.where(np.isfinite(m), m, 0)`），算出的和是 0，`log(0) = -inf`，正好是正确答案。
`log_softmax = x - logsumexp(x)`，比 `log(softmax(x))` 更精确（小概率不会下溢成 0）。
