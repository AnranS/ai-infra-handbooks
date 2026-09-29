---
title: moe_align_block_size 与分组 GEMM
chapter: perf/moe.md
difficulty: 困难
tags: [MoE, 排序, 分块 GEMM]
---
fused MoE kernel 的第一步：把所有 (token, 专家) 对**按专家排好序**，每个专家的那一段补齐到 `block_m` 的倍数，这样每个大小为 `block_m` 的块只属于一个专家，kernel 可以对每个块做一次普通的 GEMM。

**`moe_align_block_size(topk_ids, block_m, num_experts)`**：`topk_ids` 形状 `(T, k)`。把它展平成 `num_pairs = T*k` 个 pair（pair `i` 对应 token `i // k`）。返回 `(sorted_ids, expert_ids)`：

- `sorted_ids`：长度为 $\sum_e \lceil c_e / \text{block\_m} \rceil \cdot \text{block\_m}$ 的 int32 数组（$c_e$ 是专家 $e$ 的 pair 数）。按专家编号依次排列每个专家的段；段内是属于这个专家的 pair 下标（**按原来的顺序**），不足的补齐位填 `num_pairs`；没有 pair 的专家不占位置；
- `expert_ids`：每个块（每 `block_m` 个位置）属于哪个专家（int32）。

**`fused_moe_grouped(x, w, topk_ids, topk_weights, block_m)`**：用上面的结果做一次分组 GEMM：`w` 形状 `(E, d_out, d_in)`，对每个块：取出块内有效的 pair 对应的 token 行 `x[pair // k]`，乘以这个块所属专家的 `w[e].T`，
再乘以 pair 的路由权重，**累加**到输出的对应 token 上。返回 `(T, d_out)`。一次只处理一个块（每个块调用一次矩阵乘）。

<!-- 题解 -->
与书中的实现一致：

```python
flat = topk_ids.ravel(); counts = bincount(flat, minlength=E)
padded = (counts + bm - 1) // bm * bm; seg_start = cumsum(padded) - padded
order = argsort(flat, kind="stable")                  # 稳定排序：段内保持原来的顺序
rank = arange(num_pairs) - (cumsum(counts) - counts)[flat[order]]
sorted_ids = full(padded.sum(), num_pairs); sorted_ids[seg_start[flat[order]] + rank] = order
expert_ids = repeat(arange(E), padded // bm)
```

GPU 上这一步本身也是一个 kernel（SGLang 的 `moe_align_block_size` 用共享内存计数 + 前缀和）。补齐带来的浪费最多每个专家 `block_m - 1` 个位置，专家多、token 少时比例可能很高，所以 `block_m` 常按 token 数动态选择。
