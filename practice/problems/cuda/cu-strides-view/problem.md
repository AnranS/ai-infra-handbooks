---
title: stride 与视图：什么时候 view 不用拷贝
chapter: framework/tensor.md
difficulty: 困难
tags: [stride, view, expand, 张量布局]
---
张量的元素 `x[i0, i1, ...]` 在 storage 里的位置是 `Σ i_d * stride[d]`（不考虑 offset）。实现三个只看元数据的函数（形状、stride 都用元组，元素个数大于 0）：

1. `is_contiguous(shape, stride)`：按行主序排列、没有空隙。**长度为 1 的维度不看它的 stride**；
2. `view_strides(shape, stride, new_shape)`：`x.view(new_shape)` 能不能只改元数据完成？能就返回新的 stride（元组），不能返回 `None`。`new_shape` 里最多有一个 `-1`（由元素个数推出）；元素个数对不上、或者有多个 `-1` 时抛出 `ValueError`。新形状里长度为 1 的维度，stride 可以是任意值（测试不检查）；
3. `expand_strides(shape, stride, new_shape)`：`x.expand(new_shape)` 的 stride。维数可以增加（加在前面，stride 为 0）；长度为 1 的维度可以扩展成任意长度（stride 变成 0）；`-1` 表示这一维不变；其他情况抛出 `ValueError`。

```python
view_strides((3, 4), (4, 1), (2, 6))      # (6, 1)
view_strides((4, 3), (1, 4), (12,))       # None：转置之后不能直接展平
view_strides((3, 4), (0, 1), (3, 2, 2))   # (0, 2, 1)：expand 出来的张量也能拆维
expand_strides((3, 1), (1, 1), (2, 3, 4)) # (0, 1, 0)
```

提示：从最内层往外看，把在内存上"接得上"的相邻维度（`stride[d-1] == shape[d] * stride[d]`）归成一块。一块内部就像一段连续内存（步长是块最内层的 stride），可以任意拆分、合并；新形状的每一维都必须完整地落在某一块里。

<!-- 题解 -->
这就是 PyTorch 的 `computeStride`：从最后一维往前走，累计当前块的元素个数 `tensor_numel`；走到块的开头（第 0 维，或者前一维接不上）时，从新形状的最后一维往前"认领"维度，直到认领的元素个数 `view_numel` 不小于这一块的元素个数——两者必须正好相等，否则某一维跨过了块的边界，做不成视图。认领到的维度的 stride 依次是 `块最内层的 stride × 已认领的元素个数`。

几个直观的结论：转置之后不能展平（两维接不上）；切片之后，被切的那一维与内层接不上，只能在块内拆分；`expand` 出来的维度 stride 为 0，也可以拆分（0 乘任何数还是 0），但不能和相邻的维度合并。`reshape` 在 `view` 返回 `None` 时拷贝一份——所以写 kernel 前常常要 `.contiguous()`，而一个"看起来没拷贝"的 `reshape` 可能悄悄拷贝了整个张量。
