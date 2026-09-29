---
title: DeltaNet 的分块算法：把块内的纠错更新写成矩阵
chapter: frontier/linear-attn.md
difficulty: 困难
tags: [线性注意力, DeltaNet, 分块, prefill]
requires: [numpy]
---
DeltaNet 的状态更新是"先擦掉 $k_t$ 方向上的旧记忆，再写入新的"：

$$S_t = S_{t-1} + \beta_t\, k_t^\top \big(v_t - k_t S_{t-1}\big), \qquad o_t = q_t S_t$$

（$q_t, k_t$ 是 `dk` 维行向量，$v_t$ 是 `dv` 维行向量，状态 $S$ 是 `dk × dv`。）模板里给出了逐 token 递推的 `delta_recurrent`。它和门控线性注意力不同：$u_t = \beta_t (v_t - k_t S_{t-1})$ 依赖上一步的状态，块内的 token 互相牵连，不能直接套用本章的分块写法。

推导：在一个块内（状态从 $S_0$ 开始），$S_i = S_0 + \sum_{j \le i} k_j^\top u_j$，代入 $u_i$ 的定义得到

$$u_i = \beta_i \Big(v_i - k_i S_0 - \sum_{j < i} (k_i \cdot k_j)\, u_j\Big)$$

写成矩阵就是一个下三角方程组：$(I + \mathrm{diag}(\beta)\, L)\, U = \mathrm{diag}(\beta)(V - K S_0)$，其中 $L$ 是 $K K^\top$ 的严格下三角部分。实现：

1. `ut_inverse(k, beta)`：一个块的 $T = (I + \mathrm{diag}(\beta) L)^{-1}$（`C × C`，下三角，对角线为 1），用前代法算；
2. `delta_chunked(q, k, v, beta, C, S0=None)`：按块大小 `C` 分块（最后一块可以不满），块内用矩阵乘算出 $U$、输出和块末的状态，块间只传递状态。返回 `(O, S)`，与 `delta_recurrent` 一致。

<!-- 题解 -->
有了 $T$，块内的一切都是矩阵乘：

- $W = T\,\mathrm{diag}(\beta) K$、$\tilde U = T\,\mathrm{diag}(\beta) V$，则 $U = \tilde U - W S_0$（这就是 DeltaNet 论文里的 **WY 表示**：$W$、$\tilde U$ 只依赖本块，可以所有块并行算；只有 $- W S_0$ 这一项要等上一块的状态）；
- 输出 $O = Q S_0 + (Q K^\top \odot M)\, U$，$M$ 是包括对角线的下三角掩码；
- 块末状态 $S_C = S_0 + K^\top U$。

前代法求 $T$ 是块内唯一串行的部分，但它只有 `C × C`，实际的 kernel（flash-linear-attention 的 `chunk_delta_rule`）用 64 左右的块。Gated DeltaNet（Qwen3.5、Qwen3-Next 用的）在此基础上每一步再乘衰减门 $\alpha_t$，块内的各项要带上累计衰减的比值，思路完全相同——这是大作业四的延伸内容。
