---
title: 从零组装一个 Qwen 风格的小模型
chapter: transformer/build-llm.md
difficulty: 困难
tags: [Transformer, 前向传播, 综合]
---
把前面的零件拼成一个完整的仅解码器模型，用 numpy 实现前向。配置 `cfg` 是一个字典：

```python
cfg = {"vocab": 50, "d": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "d_ff": 64,
       "rope_base": 10000.0, "eps": 1e-6}
```

权重 `w` 也是字典（所有矩阵都是右乘布局，`y = x @ W`）：

| 键 | 形状 | 说明 |
| --- | --- | --- |
| `embed` | `(vocab, d)` | 嵌入，同时也是输出层（`tie_word_embeddings`）|
| `layers[i]["attn_norm"]`、`["mlp_norm"]` | `(d,)` | RMSNorm 的权重 |
| `layers[i]["wq"]` | `(d, n_heads · d_h)` | $d_h = d / n_{heads}$ |
| `layers[i]["wk"]`、`["wv"]` | `(d, n_kv_heads · d_h)` | GQA |
| `layers[i]["wo"]` | `(n_heads · d_h, d)` | |
| `layers[i]["w_gate"]`、`["w_up"]` | `(d, d_ff)` | SwiGLU |
| `layers[i]["w_down"]` | `(d_ff, d)` | |
| `final_norm` | `(d,)` | |

每一层（pre-norm）：

```text
h = x + Attention(RMSNorm(x))         # 注意力：q、k 按"前后两半"的写法加 RoPE（位置从 0 开始），因果掩码，GQA
x = h + SwiGLU(RMSNorm(h))
```

最后 `logits = RMSNorm(x) @ embed.T`。实现 `forward(tokens, w, cfg)`：`tokens` 是一维整数数组（一个序列），返回 `(T, vocab)` 的 logits。
再实现 `greedy_generate(prompt, w, cfg, n)`：贪心地生成 `n` 个新 token（每步都对整个序列重新 `forward`），返回新生成的 token 列表。

<!-- 题解 -->
按顺序写几个小函数：`rms_norm`、`rope`（前后两半）、`attention`（GQA + 因果）、`swiglu`，再在 `forward` 里循环各层。常见错误：

- RoPE 只加在 q、k 上，不加在 v 上；
- GQA：第 $h$ 个 query 头用第 $h // (n_{heads}/n_{kv})$ 个 KV 头；
- 残差加的是**归一化之前**的 `x`；
- 输出层前还有一个 `final_norm`。

这道题和"KV Cache 增量解码"那道题（推理原理部分）是一对：那里要求用 KV Cache 把每步的计算量从 $O(T)$ 个 token 降到 1 个，结果必须和这里逐字相同。
