---
title: 分组 INT4 量化、打包与 W4A16 GEMV
chapter: inference/quantization.md
difficulty: 中等
tags: [量化, 位运算, GEMV]
---
权重量化成 4 位（AWQ / GPTQ 的存储格式），推理时一边解包一边算。权重 `W` 形状 `(K, N)`（右乘布局 `y = x @ W`），沿输入维 `K` 每 `group_size` 行分一组，每组、每列有自己的缩放和零点。实现：

1. `quantize(W, group_size)`：非对称量化到 0～15。对每组每列，先把范围扩展到包含 0：`lo = min(min(W), 0)`，`hi = max(max(W), 0)`，
   `scale = (hi - lo) / 15`（整组都是 0 时 `scale = 1`），`zero = round(-lo / scale)`，`q = clip(round(W / scale) + zero, 0, 15)`。
   返回 `(q, scales, zeros)`：`q` 是 `(K, N)` 的 `uint8`，`scales` 是 `(K/g, N)` 的 `float32`，`zeros` 是 `(K/g, N)` 的 `uint8`；
2. `dequantize(q, scales, zeros, group_size)`：$\hat W = (q - \text{zero}) \cdot \text{scale}$，返回 `float32`；
3. `pack(q)`：两个 4 位数放进一个字节，第 $2i$ 行放低 4 位、第 $2i+1$ 行放高 4 位，返回 `(K/2, N)` 的 `uint8`；`unpack(packed)` 是逆操作；
4. `gemv_w4(x, packed, scales, zeros, group_size)`：`x` 形状 `(K,)`，返回 `x @ \hat W`（形状 `(N,)`，`float32`）。**不能**先解出完整的 `(K, N)` 矩阵，要按组解包、累加（每次只处理 `group_size` 行）。

`K` 保证是 `group_size` 的倍数，`group_size` 是偶数；`round` 用 numpy 的 `np.round`（四舍六入五成双）。

<!-- 题解 -->
分组：`Wg = W.reshape(K // g, g, N)`，沿 `axis=1` 求 min / max，得到 `(K/g, N)` 的 lo、hi。
量化误差：每个元素的误差不超过 `scale / 2`，测试会检查这一点。把 0 包含进范围保证了 0 能被精确表示（padding、ReLU 之后的 0 很常见）。

打包：`packed = q[0::2] | (q[1::2] << 4)`；解包：`lo = packed & 0xF`、`hi = packed >> 4`，再交错放回。

GEMV 按组累加：`y += x[rows] @ ((q_rows - zero) * scale)`，内存里始终只有一组的反量化结果，这正是 W4A16 kernel 的思路：
权重以 4 位从显存读进来（带宽省 4 倍），在寄存器里反量化成 16 位再乘。
