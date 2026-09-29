---
title: 张量并行的 MLP 与词表并行嵌入
chapter: distributed/tensor-parallel.md
difficulty: 中等
tags: [张量并行, all-reduce, 切分]
---
在 numpy 里模拟 $n$ 张卡的张量并行：每张卡是一个 rank，"通信"用函数调用模拟。实现：

1. `shard_column(W, rank, n)`：列并行，按**输出维**（`W` 的第 1 维，`W` 是 `(in, out)` 的右乘布局）切成 $n$ 份，返回第 `rank` 份；
2. `shard_row(W, rank, n)`：行并行，按**输入维**（第 0 维）切；
3. `all_reduce(parts)`：输入每个 rank 的部分结果，返回每个 rank 拿到的和（列表，每个元素是独立的数组）；并把调用次数记在全局计数器 `COMM["all_reduce"]` 里；
4. `tp_swiglu(x, w_gate, w_up, w_down, n)`：SwiGLU MLP 的 TP 版本：gate、up 列并行，down 行并行，**只做一次 all-reduce**。返回 rank 0 拿到的输出（应该等于单卡结果）；
5. `vocab_parallel_embedding(ids, E, n)`：词表并行的嵌入：词表（`E` 的第 0 维）切成 $n$ 段，每个 rank 只查自己那段的 token，其余位置填 0，再 all-reduce。返回 rank 0 的结果。

切分要求：维度能被 $n$ 整除（不能整除时抛出 `ValueError`）。

<!-- 题解 -->
MLP 先列后行：每个 rank 算 `h_r = silu(x @ Wg_r) * (x @ Wu_r)`（中间维的一段，逐元素运算不需要通信），再 `y_r = h_r @ Wd_r` 得到**完整形状的部分和**，最后 all-reduce 相加。

词表并行嵌入：rank $r$ 负责 token id 在 `[r·V/n, (r+1)·V/n)` 的部分，`local = ids - start`，不在范围内的位置输出 0；
all-reduce 之后每个位置恰好有一个 rank 贡献了真实向量。输出层（LM head）则用 all-gather 拼出完整的 logits。
