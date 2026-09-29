---
title: mma.sync 的寄存器布局
chapter: advanced/tensor-core.md
difficulty: 困难
tags: [Tensor Core, mma.sync, 寄存器布局]
---
PTX 指令 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 由一个 warp 的 32 个线程合作完成 $D = AB + C$（A 16×16、B 16×8、C/D 16×8）。
每个线程持有每个矩阵的几个元素，位置由正文的表格规定（记 `g = lane // 4`，`t = lane % 4`）：

| 操作数 | 每线程元素 | 位置（行, 列） |
| --- | --- | --- |
| A | a0～a7 | a0,a1: (g, 2t), (g, 2t+1)；a2,a3: (g+8, 2t), (g+8, 2t+1)；a4,a5: (g, 2t+8), (g, 2t+9)；a6,a7: (g+8, 2t+8), (g+8, 2t+9) |
| B | b0～b3 | b0,b1: (2t, g), (2t+1, g)；b2,b3: (2t+8, g), (2t+9, g) |
| C/D | c0～c3 | c0,c1: (g, 2t), (g, 2t+1)；c2,c3: (g+8, 2t), (g+8, 2t+1) |

实现：

1. `a_coord(lane, i)`、`b_coord(lane, i)`、`c_coord(lane, i)`：返回第 `lane` 个线程的第 `i` 个元素的 `(行, 列)`；
2. `load_fragments(A, B, C)`：按布局把矩阵拆成每个线程的片段，返回 `(fa, fb, fc)`，形状分别是 `(32, 8)`、`(32, 4)`、`(32, 4)`；
3. `mma_sync(fa, fb, fc)`：**只用片段**模拟这条指令，返回 D 的片段 `(32, 4)`。真实硬件里线程之间交换数据由 Tensor Core 完成，这里你可以先把片段按布局拼回完整矩阵、相乘，再拆成片段；
4. `store_fragment(fd)`：把 D 的片段拼回 16×8 的矩阵。

另外回答一个"面试题"：实现 `lanes_holding_row_of_c(r)`，返回持有 C 第 `r` 行元素的所有 lane（升序列表）。

<!-- 题解 -->
坐标函数直接照表格写，例如 A：

```python
def a_coord(lane, i):
    g, t = lane // 4, lane % 4
    row = g + 8 * ((i // 2) % 2)
    col = 2 * t + (i % 2) + 8 * (i // 4)
    return row, col
```

检验布局是否正确的办法：每个矩阵的每个元素恰好被一个 (lane, i) 覆盖。C 的第 $r$ 行由 `g = r % 8` 的 4 个线程（lane $4g$～$4g+3$）持有。
FlashAttention 2 在寄存器里直接对 $S = QK^\top$ 的结果做 softmax，就需要知道每个线程拿到的是哪几行哪几列——这就是 `mma.sync` 布局公开的意义。
