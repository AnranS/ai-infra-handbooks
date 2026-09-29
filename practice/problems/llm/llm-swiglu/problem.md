---
title: SwiGLU 前馈网络与参数量
chapter: transformer/ffn.md
difficulty: 简单
tags: [SwiGLU, FFN, 合并权重]
---
LLaMA / Qwen 的 FFN 是 SwiGLU：$\mathrm{FFN}(x) = \big(\mathrm{SiLU}(x W_g) \odot x W_u\big) W_d$，其中 $\mathrm{SiLU}(z) = z \cdot \sigma(z)$。

约定：`x` 形状 `(N, d)`；`w_gate`、`w_up` 形状 `(d, d_ff)`；`w_down` 形状 `(d_ff, d)`（都是"右乘"的布局）。实现：

1. `silu(z)`：数值稳定（`z = -1000` 时不能溢出报警告或得到 nan，结果应为 0）；
2. `swiglu_ffn(x, w_gate, w_up, w_down)`；
3. `swiglu_ffn_merged(x, w_gate_up, w_down)`：推理引擎把 `w_gate` 和 `w_up` 拼成一个 `(d, 2·d_ff)` 的矩阵 `w_gate_up = concat([w_gate, w_up], axis=1)`，只做一次矩阵乘，再把结果切成两半。结果要与 `swiglu_ffn` 相同；
4. `ffn_dim_for_params(d, multiple_of=256)`：SwiGLU 有 3 个矩阵，为了和"4d 的普通 FFN（2 个矩阵，$8d^2$ 参数）"参数量相当，$d_{ff}$ 取 $\frac{2}{3} \cdot 4d$，再**向上**取整到 `multiple_of` 的倍数。返回这个 $d_{ff}$。

```python
ffn_dim_for_params(4096)   # 11008，LLaMA-7B 的 intermediate_size
```

<!-- 题解 -->
稳定的 sigmoid：$\sigma(z) = \frac{1}{1 + e^{-z}}$ 在 $z$ 很负时 $e^{-z}$ 溢出。分情况：$z \ge 0$ 用 $\frac{1}{1+e^{-z}}$，$z < 0$ 用 $\frac{e^{z}}{1 + e^{z}}$；或者直接 `0.5 * (1 + np.tanh(z / 2))`。

合并权重：`h = x @ w_gate_up`，`g, u = h[:, :d_ff], h[:, d_ff:]`。一次大矩阵乘比两次小的效率高（更好地利用 GPU、少一次 kernel 启动），vLLM 里的 `MergedColumnParallelLinear` 就是做这个的。

$\frac{2}{3} \cdot 4 \cdot 4096 = 10922.67$，向上取整到 256 的倍数是 11008。
