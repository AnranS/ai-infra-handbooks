---
title: 上下文并行：用 LSE 合并分段注意力
chapter: distributed/pp-cp.md
difficulty: 中等
tags: [上下文并行, Ring Attention, logsumexp]
---
上下文并行（CP / Ring Attention）把很长的 KV 序列切到多张卡上：每张卡只对自己那段 KV 算注意力，得到**局部输出**和**局部 logsumexp**，再合并成完整结果。
split-KV 的 decode kernel（Flash-Decoding）用的也是同一个合并公式。

1. `partial_attention(q, k, v, q_pos, k_pos, causal)`：`q` 形状 `(T, d)`，`k`、`v` 形状 `(S, d)` 是 KV 的一段；`q_pos`、`k_pos` 是它们在完整序列里的绝对位置。
   缩放 $1/\sqrt{d}$；`causal=True` 时只能看到 `k_pos <= q_pos` 的 key。返回 `(o, lse)`：`o` 形状 `(T, d)`，`lse` 形状 `(T,)`。
   某个 query 在这一段里一个 key 都看不到时，它的 `o` 为 0、`lse` 为 `-inf`；
2. `merge(parts)`：`parts` 是 `[(o_1, lse_1), (o_2, lse_2), ...]`，返回合并后的 `(o, lse)`：

$$\text{lse} = \log \sum_i e^{\text{lse}_i},\qquad o = \sum_i e^{\text{lse}_i - \text{lse}}\, o_i$$

   要数值稳定，并且正确处理某些分段 `lse = -inf` 的情况（整行都是 `-inf` 时输出 0、`lse = -inf`）；
3. `context_parallel_attention(q, k, v, n, causal)`：把 KV 按顺序均匀切成 `n` 段（`np.array_split`），每段算 `partial_attention`，再 `merge`。结果应该等于完整注意力（query 的位置是序列的最后 T 个）。

<!-- 题解 -->
`partial_attention` 就是普通注意力，额外返回 `lse = m + log(sum(exp(s - m)))`。

`merge`：`L = stack(lses)`，`M = max(L)`（全是 `-inf` 时当 0），`w_i = exp(L_i - M)`，`lse = M + log(sum(w))`，`o = sum(w_i * o_i) / sum(w)`。
因为 $o_i$ 已经是各段内部归一化过的结果，乘上 $e^{\text{lse}_i}$ 就还原成未归一化的加权和，这就是合并的原理。

Ring Attention 里每张卡还会把自己的 KV 沿环传给下一张卡，这样每张卡的 query 能依次见到所有段，计算和通信可以重叠。
