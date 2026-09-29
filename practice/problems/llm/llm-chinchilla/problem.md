---
title: Scaling Law：拟合幂律与算力最优配比
chapter: training/pretraining.md
difficulty: 中等
tags: [Scaling Law, 最小二乘, 估算]
---
两个和预训练规划有关的计算：

**1. 拟合幂律** `fit_power_law(x, y)`：给定若干组观测，拟合 $y = a \cdot x^{b}$，返回 `(a, b)`。方法：两边取对数得到 $\log y = \log a + b \log x$，做普通最小二乘直线拟合（可以用 `np.polyfit`，也可以自己写闭式解）。

**2. 算力最优配比** `chinchilla_optimal(compute_flops, tokens_per_param=20)`：训练计算量近似 $C = 6ND$（$N$ 参数量，$D$ 训练 token 数）。
Chinchilla 的结论是最优时 $D \approx 20N$。给定 $C$，返回 `(N, D)`（浮点数）。

**3. 训练时间** `train_days(n_params, n_tokens, n_gpus, peak_tflops, mfu)`：用 $6ND$ 估算总计算量，按 `n_gpus` 张卡、每张峰值 `peak_tflops`（$10^{12}$ FLOPs/s）、模型算力利用率 `mfu`（0～1）计算需要多少天。

```python
chinchilla_optimal(5.76e23)                 # 约 (6.93e10, 1.39e12)：70B 参数、1.4T token
train_days(7e9, 2e12, 1024, 989, 0.4)       # 约 2.4 天
```

<!-- 题解 -->
$C = 6ND = 6N \cdot 20N = 120 N^2$，所以 $N = \sqrt{C / 120}$，$D = 20N$。

幂律拟合：`b, log_a = np.polyfit(np.log(x), np.log(y), 1)`，`a = exp(log_a)`。注意 `polyfit` 返回的系数是**从高次到低次**。
真实的 scaling law 论文还会拟合一个不可约损失项 $L = E + A/N^\alpha + B/D^\beta$，那需要非线性拟合，这里做最简单的版本。
