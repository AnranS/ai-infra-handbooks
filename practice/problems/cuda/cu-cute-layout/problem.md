---
title: CuTe 布局代数：坐标映射、合并、补集与复合
chapter: advanced/cute-layout.md
difficulty: 困难
tags: [CuTe, 布局代数, layout, coalesce, composition]
---
CuTe 的布局是一对结构相同、可以嵌套的元组 `(shape, stride)`，把坐标映射成下标：下标 = 坐标与步长的内积。整数坐标落在一个嵌套的维上时，按"第 0 维变化最快"拆开：前面各维依次取 `crd % size`、再 `crd //= size`，最后一维拿剩下的全部（不取余，所以超出 size 的坐标会沿最后一维延伸，和 CuTe 一致）。实现四个函数（形状里的数都是正整数，步长是非负整数）：

1. `crd2idx(crd, shape, stride)`：坐标 → 下标。`crd` 可以是整数（一维坐标），也可以是和形状对应的（嵌套）元组，元组里的某一项又可以是整数；
2. `coalesce(shape, stride)`：展平，去掉长度为 1 的维，把首尾相接的相邻维（`shape[k] * stride[k] == stride[k+1]`）并成一维，返回新的 `(shape, stride)`。只剩一维时返回两个整数；全部是长度 1 时返回 `(1, 0)`；
3. `complement(shape, stride, max_idx)`：返回布局 `A*`，使 `(A, A*)` 合起来把 `[0, max_idx)` 每个下标恰好覆盖一次（`max_idx` 不是 A 覆盖范围的整数倍时，向上取整到下一个整倍数）。结果要先 `coalesce`。可以假设 A 的各维按步长排序后，每一维的步长都能被"前面各维覆盖的范围"整除；
4. `composition(a, b)`：`a` 是任意布局 `(shape, stride)`，`b` 是单维布局 `(n, d)`（两个整数）。返回布局 R，使对所有 `0 <= i < n`，`R(i) == A(B(i))`，且 R 的 size 等于 n。结果可以是嵌套的，测试只检查函数和 size；测试的输入都满足整除条件（见提示），不必处理复合没有定义的情形。

```python
crd2idx((1, 5), (4, (2, 4)), (2, (1, 8)))          # 19
crd2idx(5, (4, (2, 4)), (2, (1, 8)))               # 3
coalesce((2, (1, 6)), (1, (6, 2)))                 # (12, 1)
complement(4, 2, 24)                               # ((2, 3), (1, 8))
composition(((6, 2), (8, 2)), (4, 3))              # ((2, 2), (24, 2)) 之类，R(i) == A(3i)
```

提示：`composition` 的思路是"先跳过 d，再取出 n"：把 A 先合并，从第 0 维开始，形状能被 d 整除就把这一维的形状除以 d、步长乘以 d；不够除就把 d 除以这一维的形状（向上取整）带到下一维。跳完再从剩下的形状里取 n 个，A 的最后一维可以无限延伸。复合有定义要求两个整除条件：每一维的形状和当时的 d 能互相整除，每一维取出的个数能整除还要取的个数。

<!-- 题解 -->
四个函数就是正文 `layout_core.py`、`layout_algebra.py` 里的 `crd2idx`、`coalesce`、`complement`、`composition`（单维 B 的情形）。

- `crd2idx`：元组坐标逐维递归相加；整数坐标遇到元组形状时，按每个子维的 size 取余、整除，逐个递归。
- `coalesce`：展平后顺序扫描，维护"当前维"。长度为 1 的维直接跳过；当前维的 `shape * stride` 等于下一维的步长就把形状乘上去，否则开一个新维。
- `complement`：按步长从小到大排序。`cur` 记录已经覆盖的连续范围（初始为 1），每遇到一维 `(n, d)`，先补一个 `(d // cur) : cur` 的维填上空隙，再把 `cur` 更新为 `n * d`；最后补一个 `ceil(max_idx / cur) : cur` 的维把范围延伸到 `max_idx`，合并后返回。
- `composition`：对合并后的 A 除了最后一维以外的每一维 `(s, t)`，取 `take = min(max(1, s // d), n)`：`take > 1` 时输出一维 `take : d * t`，然后 `n //= take`，`d = ceil(d / s)`；最后在 A 的最后一维上输出 `n : d * t_last`。整除条件（`s % d == 0` 或 `d % s == 0`）不满足时复合没有定义，CuTe 会在编译期报错。

这四个运算加上 `logical_divide(A, B) = A ∘ (B, complement(B, size(A)))`，就是 CuTe 分块和按线程划分的全部基础：`local_tile` 和 `local_partition` 都是先 `zipped_divide` 再固定其中一维。
