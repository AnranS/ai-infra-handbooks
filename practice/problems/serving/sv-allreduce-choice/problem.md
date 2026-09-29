---
title: 估算：选择 all-reduce 的算法
chapter: comm/nccl.md
difficulty: 简单
tags: [估算, all-reduce, α-β 模型, 定制 all-reduce]
---
用 α-β 模型估算一次 all-reduce（每卡 `S` 字节、`n` 张卡、每一步固定开销 `alpha` 秒、每卡单向带宽 `beta` 字节/秒）的时间：

| 算法 | 时间 |
| --- | --- |
| `"ring"` | $2(n-1)\alpha + 2\frac{n-1}{n}\cdot S/\beta$ |
| `"one-shot"` | $2\alpha + (n-1)\cdot S/\beta$ |
| `"two-shot"` | $4\alpha + 2\frac{n-1}{n}\cdot S/\beta$ |

实现：

1. `allreduce_time(algo, S, n, alpha, beta)`：按上表返回秒数；未知算法抛出 `ValueError`；
2. `best_algo(S, n, alpha, beta)`：返回三者中最快的算法名（时间相同时按 `ring`、`one-shot`、`two-shot` 的顺序取第一个）；
3. `crossover(n, alpha, beta)`：one-shot 与 two-shot 用时相等的消息大小 $S^*$（字节）；`n <= 2` 时两者的带宽项相同，返回 `math.inf`；
4. `busbw(S, t, n)`：nccl-tests 的总线带宽 $\frac{S}{t}\cdot\frac{2(n-1)}{n}$。

```python
best_algo(16 * 1024, 8, 1.5e-6, 450e9)      # 'one-shot'
best_algo(8 * 2**20, 8, 1.5e-6, 450e9)      # 'two-shot'
crossover(8, 1.5e-6, 450e9)                 # 约 257 KB
```

<!-- 题解 -->
令 $2\alpha + (n-1)S/\beta = 4\alpha + 2\frac{n-1}{n}S/\beta$，得 $S^* = \frac{2\alpha\beta}{(n-1)(1-2/n)}$。decode 的 TP all-reduce 通常是几十到几百 KB，正好落在 one-shot 和 two-shot 的范围里；这就是推理框架自己实现 all-reduce、只把大消息交给 NCCL 的原因。
模型里 two-shot 总是不慢于 ring，但真实的定制实现要把输入拷进预先注册的共享缓冲区、占较多 SM，缓冲区大小也有上限（vLLM 默认 8 MB），所以大消息仍然交给 NCCL（在支持的硬件上还有 NVLS）。
