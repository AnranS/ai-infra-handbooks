---
title: 批量采样：每个请求一组参数
chapter: compute/engine.md
difficulty: 中等
tags: [采样, 向量化, 批处理]
---
一个 decode batch 里的每个请求都有自己的采样参数（`temperature`、`top_k`、`top_p`）。逐个请求调用采样函数太慢，引擎要**一次性**对整个 batch 采样。
用 numpy 实现 `sample_batch(logits, temperature, top_k, top_p, u)`：

- `logits`：`(B, V)`；`temperature`、`top_p`：长度 `B` 的浮点数组；`top_k`：长度 `B` 的整数数组（`-1` 表示不限制）；`u`：长度 `B` 的 $[0, 1)$ 随机数；
- **贪心**的行：`temperature <= 0` 或 `top_k == 1`（并且 `top_p == 1`），直接取 argmax（并列取下标最小的）；
- 其余的行：`logits / temperature` → softmax → top-k（保留概率最大的 k 个，并列按下标）→ top-p（按概率降序、并列按下标，保留累计概率第一次 $\ge$ top_p 的最短前缀）→ 归一化 → 按下标从小到大累加，取第一个累计和 $> u$ 的下标；
- 返回长度 `B` 的 int64 数组。

结果要和逐行的参考实现完全一致。建议整批向量化（不对 batch 做 Python 循环），测试会用 `B = 256`、`V = 5000` 检查耗时。

<!-- 题解 -->
关键是对整个矩阵一起排序：`order = np.lexsort((col_index, -probs), axis=-1)` 按行得到"概率降序、并列按下标"的顺序；
top-k 用排名掩码：`rank_of_each_token < k[:, None]`（`k = -1` 时当作 V）；top-p 在排好序的概率上做 `cumsum`，
保留 `cumsum - p < top_p`（即"加上自己之前还没达到 top_p"）的位置，再用 `np.put_along_axis` 映射回原来的列。
最后 `np.argmax(np.cumsum(p, axis=1) > u[:, None], axis=1)` 就是逆变换采样。

书中的 `Sampler` 在 GPU 上做同样的事：全部是贪心时直接 argmax；否则用 FlashInfer 的 `top_k_top_p_sampling_from_probs` 一次性完成。
