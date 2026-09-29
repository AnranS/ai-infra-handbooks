---
title: 序列并行的 MLP 块：前向、反向与通信
chapter: model/tensor-sequence.md
difficulty: 困难
tags: [张量并行, 序列并行, Megatron, 反向传播]
requires: [numpy]
---
用 `t` 个模拟的 rank 实现"张量并行 + 序列并行"的一个 MLP 块：`out = x + act(rmsnorm(x) · g · W1ᵀ) · W2ᵀ`（`x` 是 `(s, h)`，`W1` 是 `(f, h)`，`W2` 是 `(h, f)`，`rmsnorm(x) = x / sqrt(mean(x², 最后一维) + eps)`）。

- 序列并行区域（RMSNorm、残差）：rank `r` 只拿第 `r` 段序列 `x_shards[r]`（`s / t` 行）；
- 张量并行区域：`W1` 按输出维切（rank `r` 拿 `W1_shards[r]`，`f / t` 行），`W2` 按输入维切（`W2_shards[r]`，`f / t` 列）；
- 边界上的通信只能用模板里的 `Comm`：进入 TP 区域时 `all_gather`（沿序列拼起来），离开时 `reduce_scatter`（部分和相加、再切回序列分片）。它会记录每次通信的种类和元素个数。

实现：

1. `sp_forward(x_shards, g, W1_shards, W2_shards, comm, eps=1e-6)` → `(out_shards, cache)`，`cache` 里放反向要用的东西；
2. `sp_backward(dout_shards, cache, comm)` → `(dx_shards, dg, dW1_shards, dW2_shards)`：`dg` 是 RMSNorm 权重的完整梯度——每个 rank 只算出了自己那段序列的贡献，要在 TP 组里 `all_reduce`。

前向恰好一次 `all_gather` 和一次 `reduce_scatter`；反向是它们的共轭：一次 `all_gather`（`reduce_scatter` 的反向）、一次 `reduce_scatter`（`all_gather` 的反向），再加上 `dg` 的一次 `all_reduce`。结果要和单卡计算一致。模板提供了激活函数 `act` 和它的导数 `act_grad`。

<!-- 题解 -->
反向逐项对着前向写：`dZ = all_gather(dout)`；`dW2_r = dZᵀ A_r`、`dA_r = dZ W2_r`、`dH_r = dA_r ⊙ act'(H_r)`、`dW1_r = dH_rᵀ Y`；对 `Y` 的梯度在每个 rank 上只是部分和（每个 rank 只负责 `f / t` 个中间特征），`reduce_scatter` 之后正好是自己那段序列的完整梯度。

RMSNorm 的反向：设 $\hat x = x / \mathrm{rms}$，$y = \hat x \odot g$，则 $d g = \sum_{\text{行}} dy \odot \hat x$，$d\hat x = dy \odot g$，$dx = \frac{1}{\mathrm{rms}}\big(d\hat x - \hat x \cdot \mathrm{mean}(d\hat x \odot \hat x)\big)$，每一行独立，所以在序列分片上就能算完——这正是序列并行能切开这部分激活的原因。

通信量：`all_gather` + `reduce_scatter` 与 TP 原本的一次 `all_reduce` 相同，而 RMSNorm、残差处的激活从每卡 `s × h` 降到 `s / t × h`。`dg` 的 `all_reduce` 很小（`h` 个数），但忘了它，各个 rank 上的 `g` 就会慢慢不一致——Megatron 里这类参数带有 `sequence_parallel` 标记，专门在梯度同步时额外 all-reduce。
