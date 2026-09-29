---
title: 动态损失缩放：fp16 训练里的 GradScaler
chapter: practice/mixed-precision.md
difficulty: 中等
tags: [混合精度, fp16, 损失缩放, GradScaler]
requires: [numpy]
---
fp16 的最小正数约 $6 \times 10^{-8}$，更小的梯度直接变成 0；最大值 65504，超过就是 `inf`。损失缩放把损失乘以 `scale` 再反向，梯度随之放大，更新前再除回来。模板里的 `fp16_backward(true_grads, scale)` 模拟这一过程：返回"损失乘以 `scale` 之后、以 fp16 存下来的梯度"。

实现 `GradScaler(init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)`（和 `torch.amp.GradScaler` 的规则相同）：

1. `unscale(grads16)`：转成 fp32 再除以 `scale`，返回 `(grads32, found_inf)`，`found_inf` 表示有没有 `inf` / `nan`；
2. `update(found_inf)`：发现 `inf` 时 `scale *= backoff_factor`，并把"连续正常的步数"清零；否则计数加一，攒满 `growth_interval` 步就 `scale *= growth_factor`、计数清零；
3. `step(params, grads16, lr)`：反缩放；有 `inf` 就**跳过**这一步（参数不变），否则对 fp32 参数（原地）做 `p -= lr * g`；然后调用 `update`。返回这一步是否真的更新了；
4. 属性 `scale` 是当前的缩放因子。

另外实现 `underflow_fraction(true_grads, scale)`：非零的真实梯度里，有多大比例在 `fp16_backward` 之后变成了 0。

```python
underflow_fraction([np.array([1e-8, 1e-6, 1e-3])], 1.0)       # 0.333…：1e-8 在 fp16 里是 0
underflow_fraction([np.array([1e-8, 1e-6, 1e-3])], 2.0**16)   # 0.0
```

<!-- 题解 -->
动态缩放的效果是自动找到"刚好不溢出"的最大缩放：溢出一次就减半并跳过这一步，连续 `growth_interval` 步正常就翻倍试探一下。测试里梯度最大是 10：`scale` 从 $2^{20}$ 开始连续溢出 8 次降到 4096，之后每 10 步翻倍到 8192、溢出、再退回 4096——被跳过的步数是可以算出来的。

- 跳过的一步不更新参数，但数据已经用掉了，相当于损失了一个 batch；`growth_interval` 太小会频繁溢出，太大又恢复得慢，默认 2000；
- 反缩放要在梯度裁剪**之前**做（裁剪的阈值是针对真实梯度的），`torch.amp.GradScaler` 的 `unscale_` 就是为此单独暴露的；
- bf16 的指数位和 fp32 一样多，梯度不会下溢，所以 bf16 训练不需要这一套——这是它取代 fp16 成为默认格式的主要原因。
