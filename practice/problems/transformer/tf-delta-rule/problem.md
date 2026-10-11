---
title: Gated DeltaNet 的逐 token 递推
chapter: deltanet.md
difficulty: 中等
tags: [线性注意力, delta 规则, 递推状态, einsum]
requires: [torch]
---
写出 Gated DeltaNet 的递推 `delta_rule(q, k, v, g, beta, S=None)`：

- `q, k`：`[B, T, H, dk]`；`v`：`[B, T, H, dv]`；`g`（对数衰减，≤ 0）和 `beta`（0～1）：`[B, T, H]`；`S`：初始记忆表 `[B, H, dk, dv]`，没有就从全零开始；
- 先把 `q`、`k` 在最后一维做 L2 归一化（`x * rsqrt(sum(x²) + 1e-6)`），再把 `q` 除以 $\sqrt{d_k}$；
- 对每个 token $t$：$S \leftarrow S \cdot e^{g_t}$；$\text{pred} = S^\top k_t$；$S \leftarrow S + \beta_t\, k_t\,(v_t - \text{pred})^\top$；$o_t = S^\top q_t$；
- 返回 `(out, S)`：`out` 形状 `[B, T, H, dv]`，`S` 是算完之后的记忆表。

```python
out, S = delta_rule(q, k, v, g, beta)
out.shape, S.shape          # [B, T, H, dv], [B, H, dk, dv]
```

<!-- 题解 -->
```python
def delta_rule(q, k, v, g, beta, S=None):
    B, T, H, dk = q.shape
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / dk ** 0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if S is None:
        S = q.new_zeros(B, H, dk, v.shape[-1])
    out = torch.empty_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, :, None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - pred) * beta[:, t, :, None])
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return out, S
```

三个 einsum 分别是"用 k 查表"（$S^\top k$，得到 dv 维）、"写回一个秩一的修正"（$k \otimes \delta$）和"用 q 查表"。测试里有两个性质：同一对 (k, v) 反复写入时读出来的值收敛到 v（$\beta=0.5$ 每次走一半）；分两段递推（把末状态 S 传给下一段）和一次算完结果相同——这正是第 10 章增量解码靠的性质。

常见错误：忘了 L2 归一化或 $\sqrt{d_k}$（数值对不上）；衰减放在写入之后（顺序错了，状态会多衰减一次）；`beta` 没有在头那一维广播。
