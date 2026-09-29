---
title: 专家并行的 all-to-all 分发与合并
chapter: distributed/expert-parallel.md
difficulty: 困难
tags: [专家并行, all-to-all, MoE]
---
$E$ 个专家平均分到 $n$ 个 rank 上（rank $r$ 持有专家 $[rE/n, (r+1)E/n)$）。每个 rank 上有自己的一批 token，路由结果已经算好：

- `x[r]`：rank $r$ 的 token，形状 `(N_r, d)`；
- `topk_ids[r]`、`topk_w[r]`：形状 `(N_r, k)`，每个 token 选中的专家和权重；
- `experts[e]`：一个函数，输入 `(m, d)` 返回 `(m, d)`（专家 $e$ 的 FFN）。

实现 `ep_moe(x, topk_ids, topk_w, experts, n)`，模拟一次完整的 EP 前向，返回 `(outputs, send_counts)`：

1. **dispatch**：每个 rank 把每个 (token, 选中的专家) 对发给持有该专家的 rank。发送顺序：先按目标 rank，同一目标内按 **(token 下标, k 槽位)** 的顺序。`send_counts[r][s]` 是 rank $r$ 发给 rank $s$ 的 (token, 专家) 对的数量（$n \times n$ 的列表）；
2. **all-to-all** 之后，每个 rank 把收到的 token **按专家分组**，每个本地专家只调用一次（对分到它的所有 token 一次性计算）；
3. **combine**：结果按原路发回源 rank，源 rank 对每个 token 做加权求和 $\sum_k w_k \cdot y_k$。

返回的 `outputs[r]` 是 rank $r$ 的输出 `(N_r, d)`，应该等于单机 MoE 的结果。

<!-- 题解 -->
dispatch 时对每个 (token i, 槽位 j) 算出目标 rank `dst = e // (E // n)`，按 `(dst, i, j)` 排序后打包，记下每个元素的来源 `(src_rank, i, j)`，combine 时靠它把结果放回去。

接收端：把收到的所有行拼起来，按专家分组（`np.nonzero(expert == e)`），每个专家一次矩阵乘。
真实实现（DeepEP 等）用两次 all-to-all（dispatch、combine），通信量 ∝ token 数 × k × d，所以 EP 的瓶颈常常在网络上；
各 rank 收到的 token 数不均衡（热门专家）时，慢的 rank 会拖住所有人，这就是负载均衡损失和 EPLB（冗余专家）要解决的问题。
