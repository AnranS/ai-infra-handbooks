---
title: 零中心的 RMSNorm 与带门的 RMSNorm
chapter: norm.md
difficulty: 简单
tags: [RMSNorm, 零中心, 门控, 精度]
requires: [torch]
---
Qwen3.5 的 RMSNorm 存的是 $w - 1$，用的时候乘 $(1 + w)$；线性注意力的输出用的是"先归一化、再乘 silu(门)"的版本。把两个函数写出来：

1. `rmsnorm(x, weight, eps=1e-6)`：对最后一维做 RMSNorm，返回 `x * rsqrt(mean(x²) + eps) * (1 + weight)`。**统计量必须在 fp32 里算**：输入是 bf16 时也要先转 fp32 再算均方，最后转回输入的 dtype；
2. `gated_rmsnorm(x, gate, weight, eps=1e-6)`：先对 `x` 做同样的归一化（这个版本的 `weight` 不是零中心的，直接乘 `weight`），再乘 `silu(gate)`（门也在 fp32 里算），返回输入的 dtype。

```python
x = torch.randn(2, 3, 8) * 5
y = rmsnorm(x, torch.zeros(8))
y.pow(2).mean(-1).sqrt()          # 全是 1.0：权重为 0 时就是"不缩放"
rmsnorm(x.bfloat16(), torch.zeros(8)).dtype   # torch.bfloat16
```

<!-- 题解 -->
两个函数的骨架一样：`xf = x.float()`，`y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)`，最后 `.type_as(x)`。区别只在乘什么：零中心版乘 `1 + weight.float()`，带门版乘 `weight * F.silu(gate.float())`。

为什么要在 fp32 里算：bf16 只有 8 位尾数，几百个数平方再求和会把小的全吞掉，算出来的均方根偏小，归一化之后的向量就偏大。所有实现（HF、vLLM、SGLang 的 kernel）都是先升到 fp32 算统计量，再按输入精度输出。

为什么存 $w - 1$：训练开始时比例应该是 1；bf16 在 1 附近的相邻可表示数间隔是 $2^{-7}$，比 1 小的更新会被吞掉；存 0 就没有这个问题（0 附近密得多）。测试里会用一个很小的权重更新检查这一点。
