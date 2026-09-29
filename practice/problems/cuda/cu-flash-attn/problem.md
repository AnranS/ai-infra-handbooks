---
title: FlashAttention 前向：分块 + online softmax
chapter: advanced/attention.md
difficulty: 困难
tags: [FlashAttention, online softmax, 分块]
---
用 numpy 实现单头注意力的 FlashAttention 前向 `flash_attn_fwd(q, k, v, causal=False, Br=64, Bc=64)`：

- `q`：`(T, d)`；`k`、`v`：`(S, d)`，$S \ge T$；缩放系数 $1/\sqrt{d}$；
- `causal=True` 时按"右下角对齐"：第 $i$ 个 query 的绝对位置是 $S - T + i$，只能看到位置 $\le S - T + i$ 的 key；
- 返回 `(o, lse)`：输出 `(T, d)`，以及每个 query 的 $\log\sum_j e^{s_{ij}}$（`(T,)`，反向传播和 split-KV 合并都要用它）。

要求**不能构造完整的 $T \times S$ 分数矩阵**：外层按 `Br` 行分块遍历 query，内层按 `Bc` 列分块遍历 key/value，
每个 query 块维护行最大值 `m`、分母 `l` 和未归一化的输出 `acc`，遇到新的 key 块时用 online softmax 的公式更新：

$$m' = \max(m, \max_j s_j),\quad l' = l\, e^{m - m'} + \sum_j e^{s_j - m'},\quad \text{acc}' = \text{acc}\, e^{m - m'} + \sum_j e^{s_j - m'} v_j$$

最后 `o = acc / l`，`lse = m + log(l)`。`causal=True` 时，完全在对角线右上方的 key 块直接跳过（不计算）。

测试会用 `tracemalloc` 检查峰值内存：$T = S = 1024$、$d = 64$ 时，完整分数矩阵要 8 MB，分块实现的峰值应该远小于它。

<!-- 题解 -->
```python
for i0 in range(0, T, Br):
    qi = q[i0:i0+Br]
    m = full(len(qi), -inf); l = zeros(len(qi)); acc = zeros((len(qi), d))
    for j0 in range(0, S, Bc):
        if causal and j0 > S - T + i0 + len(qi) - 1: break       # 整块都在未来
        s = qi @ k[j0:j0+Bc].T * scale
        if causal: s = where(列位置 <= 行位置, s, -inf)
        m_new = maximum(m, s.max(1)); p = exp(s - m_new[:, None])
        alpha = exp(m - m_new)
        l = l * alpha + p.sum(1); acc = acc * alpha[:, None] + p @ v[j0:j0+Bc]; m = m_new
    o[i0:i0+Br] = acc / l[:, None]; lse[i0:i0+Br] = m + log(l)
```

GPU 上每个 query 块由一个 thread block 处理，`qi`、`acc` 放在寄存器/共享内存里，K、V 块从 HBM 流过——HBM 读写量从 $O(TS)$ 降到 $O(Td + Sd)$ 的若干倍，这就是 FlashAttention 快的原因。
