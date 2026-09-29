---
title: 用列主序的 cuBLAS 算行主序的矩阵乘
chapter: tools/ecosystem.md
difficulty: 中等
tags: [cuBLAS, 列主序, 转置技巧]
---
cuBLAS 沿用 Fortran 的约定：矩阵按**列主序**存放。模板里有一个模拟 `cublasSgemm` 的函数（在一维缓冲区上操作）：

```python
sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
# C(m×n) = alpha · op(A)(m×k) · op(B)(k×n) + beta · C
# 缓冲区都是列主序：矩阵 X 的元素 (i, j) 在 X[i + j * ld]；op 是 'N'（不转置）或 'T'（转置）
```

numpy 数组默认是**行主序**。实现两个函数，**只调用一次 `sgemm`**，并且不能转置或复制输入（直接把 `A.reshape(-1)` 这样的视图传进去，测试会检查传入的缓冲区与输入共享内存）：

1. `matmul(A, B)`：`A` 是 `(M, K)`、`B` 是 `(K, N)` 的行主序 float32 数组，返回行主序的 `A @ B`（`(M, N)`）；
2. `matmul_at_b(A, B)`：`A` 是 `(K, M)`、`B` 是 `(K, N)`，返回 `A.T @ B`（`(M, N)`）。

输出缓冲区自己分配（一维的 `np.empty(M * N, dtype=np.float32)`），最后 reshape 成 `(M, N)` 返回。

<!-- 题解 -->
关键观察：一块行主序的 `M × N` 内存，按列主序解读就是它的转置 `N × M`（ld = N）。所以：

- $C = AB$ 行主序 ⇔ $C^\top = B^\top A^\top$ 列主序。行主序的 `B` 按列主序看正好是 $B^\top$（`N × K`，ld = N），`A` 看作 $A^\top$（`K × M`，ld = K），都不用转置：
  `sgemm('N', 'N', N, M, K, 1, B, N, A, K, 0, C, N)`；
- $C = A^\top B$ ⇔ $C^\top = B^\top A$。`B` 仍是 'N'；行主序的 `A`（`K × M`）按列主序看是 $A^\top$（`M × K`，ld = M），而我们需要 $A$，所以用 'T'：
  `sgemm('N', 'T', N, M, K, 1, B, N, A, M, 0, C, N)`。

PyTorch 调用 cuBLAS 时就是这样"免费"地处理行主序的，这也是面试里常问的问题。
