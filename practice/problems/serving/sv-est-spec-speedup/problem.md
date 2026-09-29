---
title: 估算：投机解码的期望加速比与最佳草稿长度
chapter: topics/speculative.md
difficulty: 简单
tags: [估算, 投机解码, 接受率]
---
投机解码每步先用草稿模型猜 $\gamma$ 个 token，再让目标模型一次验证。设每个草稿 token 被接受的概率都是 $\alpha$（相互独立），第一个被拒绝的位置之后全部作废，
但验证这一步总能额外得到 1 个目标模型自己的 token。于是：

- 每步期望产出的 token 数：$\frac{1-\alpha^{\gamma+1}}{1-\alpha}$（$\alpha = 1$ 时为 $\gamma + 1$）；
- 草稿模型跑一个 token 的时间是目标模型的 $c$ 倍，一步的时间约为 $\gamma c + 1$（以目标模型一次前向为单位，验证 $\gamma+1$ 个 token 和生成 1 个 token 耗时相同）；
- 加速比 = 期望 token 数 ÷ 一步的时间。

实现：

1. `expected_tokens(alpha, gamma)`；
2. `speedup(alpha, gamma, c)`；
3. `best_gamma(alpha, c, max_gamma=16)`：在 $0 \ldots$ `max_gamma` 里选加速比最高的 $\gamma$（$\gamma = 0$ 表示不投机，加速比为 1；相同时选较小的）；
4. `estimate_alpha(accepted, gamma)`：从线上日志估计接受率。`accepted` 是每一步接受的草稿 token 数（0 到 $\gamma$）。
   每个被接受的 token 是一次"成功"，每个接受数小于 $\gamma$ 的步骤有一次"失败"，估计值 = 成功数 ÷（成功数 + 失败数）；没有任何记录时返回 0.0。

```python
expected_tokens(0.8, 4)        # 约 3.36
speedup(0.8, 4, 0.05)          # 约 2.80
best_gamma(0.8, 0.05)          # 8
best_gamma(0.3, 0.5)           # 0：接受率低、草稿又贵，不如不投机
```

<!-- 题解 -->
- 接受率决定上限：$\gamma \to \infty$ 时期望 token 数趋近 $1/(1-\alpha)$，$\alpha = 0.8$ 时最多 5 个；
- 草稿越贵（$c$ 越大），最佳 $\gamma$ 越小；接受率和 $c$ 都随请求和上下文变化，所以线上系统会动态调整 $\gamma$，甚至按请求关掉投机；
- 这个公式假设验证一步和普通 decode 一步耗时相同。batch 很大时 decode 已经接近算力瓶颈，验证 $\gamma+1$ 个 token 的代价不再能忽略，投机解码的收益会明显下降——高并发时收益变小就是这个原因。
