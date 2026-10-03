---
title: 手写 float32 → bfloat16 舍入
chapter: floating-point.md
difficulty: 中等
tags: [浮点数, 位运算, bfloat16]
---
bfloat16 就是 float32 的**高 16 位**：1 位符号、8 位指数、7 位尾数。它和 float32 的表示范围相同，只是精度低。
numpy 没有 bfloat16 类型，我们用 `uint16` 保存它的位模式，实现两个函数：

- `to_bf16_bits(x)`：`x` 是 `float32` 数组，返回同形状的 `uint16` 数组。使用**就近舍入、平局取偶**（round-to-nearest-even）：
  把 float32 的 32 位看成整数，舍掉低 16 位时，如果低 16 位 > `0x8000` 就进位，< `0x8000` 就舍去，**正好等于** `0x8000` 时让结果的最低位为偶数；
  NaN 要保持是 NaN（结果的尾数不能变成 0，统一返回 `0x7FC0`）；进位可能让很大的有限数变成 `inf`，这是正确的。
- `from_bf16_bits(bits)`：`uint16` → `float32`（低 16 位补 0）。

然后回答一个问题：实现 `bf16_ulp_at(x)`，返回 bfloat16 在 `x`（正的规格化数）附近的**相邻可表示数间隔**，例如 `bf16_ulp_at(1.0) == 2**-7`。
想一想 bfloat16 里 `256 + 1` 等于多少。

<!-- 题解 -->
```python
u = x.astype(np.float32).view(np.uint32)
lsb = (u >> 16) & 1                       # 保留部分的最低位
rounded = (u + 0x7FFF + lsb) >> 16        # 加 0x7FFF + lsb：大于一半进位，正好一半时只有 lsb=1 才进位
```

这是 PyTorch 和各类 kernel 里 float → bf16 的标准写法。NaN 要单独处理，因为加法可能把 NaN 的尾数进位成 0、变成 inf。

间隔：bfloat16 有 7 位尾数，指数为 $e$ 时间隔是 $2^{e-7}$，$e = \lfloor \log_2 x \rfloor$。所以 256 附近间隔是 2，`256 + 1` 舍入回 256（平局取偶）。
训练时如果用 bfloat16 累加很多小梯度，它们会被"吃掉"，这就是为什么累加器、优化器状态要用 float32。
