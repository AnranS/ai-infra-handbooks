---
title: 因果多头注意力
chapter: transformer/attention.md
difficulty: 中等
tags: [注意力, 多头, 因果掩码]
---
用 numpy 实现一个多头自注意力层的前向：

```python
def mha(x, wq, wk, wv, wo, n_heads, causal=True, mask=None) -> np.ndarray
```

- `x`：`(B, T, d)`；`wq`、`wk`、`wv`、`wo`：`(d, d)`（右乘布局，`q = x @ wq`）；`d` 能被 `n_heads` 整除，`head_dim = d // n_heads`；
- 每个头：$\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V$；`causal=True` 时位置 $i$ 只能看到 $j \le i$；
- `mask`：可选的 `(B, T)` bool 数组（padding 掩码），`False` 的位置**作为 key** 不能被任何 query 看到；
- 多头结果拼回 `(B, T, d)` 后乘 `wo`；
- 如果某个 query 一个 key 都看不到（例如它自己是 padding，且前面也全是 padding），它的输出定义为 0 向量（不能出现 nan）。

不能对 batch 或头做 Python 循环以外的逐元素循环；对头做循环是允许的，但更推荐 reshape 成 `(B, H, T, d_h)` 一次算完。

<!-- 题解 -->
```python
q = (x @ wq).reshape(B, T, H, dh).transpose(0, 2, 1, 3)       # (B, H, T, dh)
scores = q @ k.transpose(0, 1, 3, 2) / sqrt(dh)                # (B, H, T, T)
allowed = tril(ones(T, T))[None, None] & mask[:, None, None, :]
scores = where(allowed, scores, -inf)
```

softmax 时一整行都是 `-inf` 会得到 nan：先把这种行的最大值当 0 处理，求和为 0 时输出 0。
注意掩码加在 **key** 维（最后一维）上：padding 的 key 不能被看到，但 padding 的 query 照样会算出东西（通常之后被丢掉）。
