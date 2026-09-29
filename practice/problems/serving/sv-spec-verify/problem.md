---
title: 投机解码的验证：拒绝采样
chapter: topics/speculative.md
difficulty: 困难
tags: [投机解码, 拒绝采样, 概率]
---
草稿模型一次提议 $k$ 个 token，目标模型一次前向算出这 $k+1$ 个位置的分布，然后逐个验证。拒绝采样保证：**输出的分布和只用目标模型采样完全相同**。实现：

**`verify(draft, q, p, u, u_resample, u_bonus)`**：

- `draft`：草稿 token 列表（长度 $k$）；`q`：`(k, V)`，草稿模型采样第 $i$ 个 token 时的分布；`p`：`(k+1, V)`，目标模型在这 $k+1$ 个位置的分布；
- `u`：长度 $k$ 的 $[0,1)$ 随机数；`u_resample`、`u_bonus`：各一个随机数（由调用方给出，结果可复现）；
- 从 $i = 0$ 开始：令 $t = \text{draft}[i]$，如果 `u[i] < min(1, p[i][t] / q[i][t])` 就接受；否则拒绝，从**残差分布** $r \propto \max(0, p_i - q_i)$ 里用 `u_resample` 采样一个 token 替换它，结束；
- 全部接受时，再从 $p_k$ 里用 `u_bonus` 采样一个"奖励" token；
- 用随机数 $u$ 从分布 $d$ 采样：归一化后按下标从小到大累加，返回第一个累计和 $> u$ 的下标（最后一个概率非零的下标兜底）；

返回输出的 token 列表（长度 1 到 $k+1$）。

**`verify_greedy(draft, target_argmax)`**：贪心验证，`target_argmax` 是目标模型在 $k+1$ 个位置的 argmax。逐个比较，第一个不相等的位置换成目标模型的 token 并结束；全部相等时追加最后一个位置的 token。

<!-- 题解 -->
```python
for i, t in enumerate(draft):
    if u[i] < min(1.0, p[i][t] / q[i][t]):
        out.append(t); continue
    r = np.maximum(p[i] - q[i], 0)
    out.append(sample(r / r.sum(), u_resample)); return out
out.append(sample(p[k], u_bonus)); return out
```

为什么分布不变：接受 $t$ 的概率是 $q(t)\min(1, p(t)/q(t)) = \min(q(t), p(t))$；拒绝后从 $\max(0, p - q)$ 采样，两部分加起来恰好是 $p(t)$。
测试里有一个统计检验：固定 $p$、$q$，用大量随机数运行 `verify`，第一个输出 token 的频率应该接近 $p_0$。
