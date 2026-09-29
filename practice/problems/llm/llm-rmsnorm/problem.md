---
title: RMSNorm 与残差流
chapter: transformer/norm-residual.md
difficulty: 简单
tags: [RMSNorm, LayerNorm, pre-norm]
---
用 numpy 实现（输入 `x` 形状 `(..., d)`，都沿最后一维计算）：

1. `rms_norm(x, weight, eps=1e-6)`：$y = \dfrac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \odot w$；
2. `layer_norm(x, weight, bias, eps=1e-5)`：$y = \dfrac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot w + b$（$\sigma^2$ 用**有偏**方差，即除以 $d$）；
3. `pre_norm_block(x, sublayer, weight, eps=1e-6)`：pre-norm 残差块 $x + \mathrm{sublayer}(\mathrm{RMSNorm}(x))$；
4. `fused_add_rms_norm(x, residual, weight, eps=1e-6)`：推理引擎里常见的融合算子：先 `residual = x + residual`，再对新的 `residual` 做 RMSNorm，返回 `(normed, residual)`。

计算要在 **float32** 里进行：如果输入是 `float16`，先转成 float32 算，最后把结果转回输入的 dtype（`fused_add_rms_norm` 返回的两个数组都是输入的 dtype）。

<!-- 题解 -->
`np.mean(x * x, axis=-1, keepdims=True)` 求均方，`keepdims` 让结果能和 `x` 广播。

float16 的最大值只有 65504，平方很容易溢出，所以归一化一律在 float32 里算，这也是 Hugging Face 实现里 `hidden_states.to(torch.float32)` 那一行的原因。
`fused_add_rms_norm` 把"加残差"和"归一化"合成一次读写，vLLM、SGLang 都有这个 kernel，因为这两步都是访存密集的。
