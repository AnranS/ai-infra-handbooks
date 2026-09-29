---
title: MoE 路由、分发与负载均衡损失
chapter: transformer/moe.md
difficulty: 困难
tags: [MoE, top-k 路由, 负载均衡]
---
混合专家层：每个 token 只送给打分最高的 $k$ 个专家，输出是这些专家输出的加权和。实现 `moe_forward(x, router_w, experts, top_k, norm_topk_prob=True)`：

- `x`：`(N, d)`；`router_w`：`(d, E)`；`experts`：长度为 $E$ 的列表，每个元素是 `(w_gate, w_up, w_down)`，一个 SwiGLU 专家（形状同 SwiGLU 那道题）；
- 路由概率 $p = \mathrm{softmax}(x \cdot \text{router\_w})$，每个 token 选概率最大的 `top_k` 个专家（并列时选下标小的）；
- `norm_topk_prob=True`（Qwen3-MoE、Mixtral）时，选中的 $k$ 个概率重新归一化为和 1；为 `False`（Qwen2-MoE）时直接用原始概率；
- 输出 $y_i = \sum_{e \in \mathrm{topk}(i)} w_{i,e} \cdot \mathrm{expert}_e(x_i)$。

返回一个字典：

| 键 | 内容 |
| --- | --- |
| `"out"` | `(N, d)` 输出 |
| `"topk_ids"` | `(N, k)`，每行按概率从大到小（并列按下标从小到大）|
| `"topk_weights"` | `(N, k)`，与 `topk_ids` 对应 |
| `"tokens_per_expert"` | 长度为 $E$ 的列表，第 $e$ 个元素是被路由到专家 $e$ 的 token 下标（升序的 Python 列表）|
| `"aux_loss"` | Switch Transformer 的负载均衡损失 $E \sum_e f_e P_e$：$f_e$ 是分到专家 $e$ 的**路由次数**占总路由次数 $N k$ 的比例，$P_e$ 是所有 token 对专家 $e$ 的平均路由概率（`float`）|

要求**按专家分组计算**：每个专家只对分给它的那些 token 调用一次模板里的 `swiglu`（不能对每个 token 单独调用，也不能让每个专家处理全部 token；测试会统计 `swiglu` 的调用）。

<!-- 题解 -->
1. `order = np.lexsort((np.arange(E)[None].repeat(N, 0), -probs), axis=-1)` 或者用稳定排序 `np.argsort(-probs, kind="stable")`，取前 $k$ 列；
2. 分发：对每个专家 $e$，`rows, slots = np.nonzero(topk_ids == e)`，`y[rows] += w[rows, slots, None] * expert_e(x[rows])`（`np.add.at` 或逐专家累加）；
3. 负载均衡损失鼓励 $f_e$ 和 $P_e$ 都接近均匀分布 $1/E$，此时损失等于 1。

推理引擎里的 fused MoE kernel 做的正是第 2 步：先按专家把 token 排好序（`moe_align_block_size`），再对每个专家做分组 GEMM。
