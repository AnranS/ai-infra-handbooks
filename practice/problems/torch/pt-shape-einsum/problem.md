---
title: 形状的功夫：einsum、合并头与广播规则
chapter: shape.md
difficulty: 简单
tags: [einsum, reshape, 广播, 多头注意力]
requires: [torch]
---
多头注意力里最容易写错的不是公式，是形状。把四件事写出来：

1. `attn_scores(q, k)`：`q` 形状 `(B, H, T, D)`，`k` 形状 `(B, H, S, D)`，返回 `(B, H, T, S)` 的打分，已经除以 $\sqrt{D}$。用 `torch.einsum` 写，不要手工 `transpose` + `matmul`；
2. `merge_heads(x)`：`(B, H, T, D)` → `(B, T, H * D)`，也就是把多头的输出拼回去喂给 `o_proj`；
3. `split_heads(x, h)`：`(B, T, H * D)` → `(B, H, T, D)`，必须是 `merge_heads` 的逆运算；
4. `broadcast_shape(sa, sb)`：两个形状元组（如 `(4, 1, 3)` 和 `(2, 3)`）广播后的形状；不能广播时抛 `ValueError`。自己按规则算，不要调用 `torch.broadcast_shapes`。

```python
q = torch.randn(2, 4, 6, 8); k = torch.randn(2, 4, 10, 8)
attn_scores(q, k).shape            # torch.Size([2, 4, 6, 10])
broadcast_shape((4, 1, 3), (2, 3)) # (4, 2, 3)
broadcast_shape((3,), (4,))        # ValueError
```

<!-- 题解 -->
`attn_scores` 就是 `torch.einsum("bhtd,bhsd->bhts", q, k) * D ** -0.5`：下标里重复出现又不在输出里的 `d` 被求和，这正是点积；`b`、`h` 两个下标两边都有、输出里也有，于是按批处理。einsum 的好处是**形状写在字符串里**，读代码的人不用在脑子里跟踪 transpose。

`merge_heads` 的坑在于顺序：`(B, H, T, D)` 要先 `transpose(1, 2)` 变成 `(B, T, H, D)` 再 `reshape(B, T, H*D)`。直接 `reshape` 得到的是把 `H` 和 `T` 混在一起的垃圾，而且**不报错**。transpose 之后张量不连续，`reshape` 会自己复制一份（`view` 则会报错并提示你加 `contiguous()`）—— 这也是为什么推理框架里到处是 `.contiguous()`。

广播规则只有两条：从右往左对齐，缺的维度补 1；每一维要么相等、要么其中一个是 1。写成循环就是：

```python
out = []
for i in range(max(len(sa), len(sb))):
    a = sa[-1 - i] if i < len(sa) else 1
    b = sb[-1 - i] if i < len(sb) else 1
    if a != b and a != 1 and b != 1:
        raise ValueError(...)
    out.append(max(a, b))
return tuple(reversed(out))
```

`(3,)` 和 `(4,)` 不能广播，是因为两个 3 和 4 都不是 1。实际项目里广播出错往往不报错，只是悄悄把 `(T, 1)` 和 `(T,)` 广播成 `(T, T)` —— 内存和时间都翻 T 倍，loss 也不降。
