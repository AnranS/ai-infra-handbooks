---
title: 估算：MLA 该走展开还是吸收
chapter: moe/mla.md
difficulty: 简单
tags: [估算, MLA, 注意力, prefill, decode]
---
MLA 有两条等价的计算路径（形状：`H` 个头，每头非位置部分 `nope` 维、RoPE 部分 `rope` 维、v 为 `dv` 维，潜向量 `dc` 维）：

- **展开**：每个缓存 token 先展开出 K、V，代价 `2·dc·H·(nope + dv)`；每对 (query, key) 的代价 `2·H·(nope + rope) + 2·H·dv`；
- **吸收**：每个新 token 额外付出 `2·H·nope·dc + 2·H·dc·dv`；每对 (query, key) 的代价 `2·H·(dc + rope) + 2·H·dc`。

`q` 个新 token 接在前缀后面、总长 `k`（含新 token）时，因果注意力的对数是 `q·k − q(q−1)/2`。

实现（`shape` 是字典 `{"H", "nope", "rope", "dv", "dc"}`）：

1. `path_flops(q, k, shape)`：返回 `(展开的 FLOPs, 吸收的 FLOPs)`；
2. `choose(q, k, shape)`：返回 `"expand"` 或 `"absorb"`（相等时选 `"absorb"`）；
3. `crossover_q(shape)`：`k` 远大于 `q` 时的近似分界 `展开每 token 代价 / (吸收每对 − 展开每对)`。

```python
V3 = {"H": 128, "nope": 128, "rope": 64, "dv": 128, "dc": 512}
choose(1, 8192, V3)        # 'absorb'（decode）
choose(8192, 8192, V3)     # 'expand'（prefill）
round(crossover_q(V3))     # 171
```

<!-- 题解 -->
两条路径的固定开销落在不同的地方：展开付在每个**缓存** token 上，吸收付在每个**新** token 上；每对的代价则是吸收更贵（3.4 倍）。所以 decode 必须吸收，prefill 用展开，带前缀缓存的 extend 看新 token 的数量。真实引擎还要考虑显存（展开整个长前缀会占很多显存，要分块展开）和 kernel 的可用性。
