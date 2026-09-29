---
title: 蒸馏的损失：温度、正反 KL 与只存 top-k 的老师
chapter: inference/sparsity-distill.md
difficulty: 中等
tags: [蒸馏, KL 散度, softmax, 数值稳定]
requires: [numpy]
---
学生和老师的 logits 都是 `(N, V)`（`N` 个位置、词表 `V`），损失对位置取平均。实现（都要数值稳定：logits 到 $10^4$ 也不能出现 `inf` / `nan`）：

1. `log_softmax(z, T=1.0)`：沿最后一维，对 `z / T` 做 log-softmax；
2. `kd_loss(student, teacher, T)`：Hinton 的蒸馏损失 $T^2 \cdot \mathrm{KL}(p_T \,\|\, q_T)$，$p_T = \mathrm{softmax}(\text{teacher}/T)$，$q_T = \mathrm{softmax}(\text{student}/T)$；
3. `kd_grad(student, teacher, T)`：`kd_loss` 对 `student` 的梯度（`(N, V)`）；
4. `reverse_kl(student, teacher)`：反向 KL $\mathrm{KL}(q \,\|\, p)$（温度为 1）；
5. `topk_kd_loss(student, topk_ids, topk_logprobs)`：离线蒸馏时老师的输出太大（每个位置 `V` 个数），通常只存 top-k：`topk_ids`、`topk_logprobs` 都是 `(N, k)`，后者是老师在全词表上的 log 概率。把这 k 个概率重新归一化成 $\tilde p$，损失是 $\sum_i \tilde p_i\,(\log \tilde p_i - \log q(\text{id}_i))$，其中 $q$ 是学生在**全词表**上的 softmax。

```python
s = np.array([[1.0, 2.0, 3.0]])
t = np.array([[3.0, 2.0, 1.0]])
kd_loss(s, s, 2.0)          # 0.0
kd_loss(s, t, 1.0)          # 1.2242…
```

<!-- 题解 -->
稳定的 log-softmax：先减去每行的最大值，$\log\mathrm{softmax}(z) = z - m - \log\sum e^{z - m}$。KL 用 log 概率相减来算，不要先算概率再取 log（小概率会变成 0，取 log 得到 `-inf`）。

梯度：$\frac{\partial}{\partial s}\, T^2\,\mathrm{KL}(p_T \| q_T) = T\,(q_T - p_T)$，再除以位置数 `N`。温度很高时 $q_T - p_T \approx \frac{(s - t) - \overline{(s - t)}}{V T}$，所以梯度趋向 $\frac{(s - t) - \overline{(s - t)}}{V}$——与 $T$ 无关，这就是乘 $T^2$ 的原因，也说明高温蒸馏近似于"让学生的 logits 去拟合老师的 logits"。

正向 KL 要求学生覆盖老师所有的高概率区域（mode covering），反向 KL 允许学生只抓住一个峰（mode seeking）：老师是两个峰时，"一峰独大"的学生在反向 KL 下更好、在正向 KL 下很差。在线蒸馏（学生自己生成、老师打分）常用反向 KL，就是希望学生专注于老师最确定的回答。

只存 top-k 时，没存的那部分概率被当作 0，重新归一化让 $\tilde p$ 仍是一个分布；k 取 32～64 时通常已经覆盖了老师 95% 以上的概率质量，存储从 `V` 个数降到 `2k` 个。
