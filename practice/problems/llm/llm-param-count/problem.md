---
title: 从 config.json 精确计算参数量（含 MoE）
chapter: inference/estimation.md
difficulty: 中等
tags: [参数量, MoE, 估算]
---
正文给出了稠密模型参数量的公式。把它扩展到 Qwen3 系列（含 MoE），实现 `count_params(c)`，返回 `(total, active)`：总参数量，以及每个 token 实际用到的参数量。`c` 是 `config.json` 解析出的字典：

| 字段 | 含义 |
| --- | --- |
| `vocab_size`、`hidden_size`（$d$）、`num_hidden_layers`（$L$） | |
| `num_attention_heads`、`num_key_value_heads`、`head_dim` | 没有 `head_dim` 时取 $d / n_{heads}$；没有 `num_key_value_heads` 时等于 `num_attention_heads` |
| `tie_word_embeddings` | 为真时输出层与嵌入共享；默认 `False` |
| `qk_norm` | 为真时每层有 q_norm、k_norm 两个 RMSNorm，各 `head_dim` 个参数；默认 `False` |
| `intermediate_size` | 稠密 FFN（SwiGLU，3 个矩阵）的中间维 |
| `num_experts` | 为 0 或不存在时是稠密模型；否则每层的 FFN 是 MoE |
| `num_experts_per_tok`、`moe_intermediate_size` | 每个 token 用几个专家、每个专家的中间维 |

逐项计算（都没有偏置）：

- 嵌入 $V d$；输出层 $V d$（共享时为 0）；最后一个 RMSNorm $d$；
- 每层：注意力 $d \cdot n_h d_h + 2 \cdot d \cdot n_{kv} d_h + n_h d_h \cdot d$，两个 RMSNorm $2d$，`qk_norm` 为真时再加 $2 d_h$；
- 稠密 FFN：$3 d \cdot d_{ff}$；MoE FFN：路由器 $d \cdot E$，加上 $E$ 个专家，每个 $3 d \cdot d_{moe}$。

**激活参数量**：总参数量减去每层"没被选中的专家"：$L \cdot (E - k) \cdot 3 d \, d_{moe}$（稠密模型的激活参数量就是总参数量）。

```python
count_params(qwen3_0_6b)     # (596049920, 596049920)
count_params(qwen3_30b_a3b)  # 约 (30.5e9, 3.35e9)
```

<!-- 题解 -->
照表格逐项相加即可，关键是别漏项：`qk_norm`、共享嵌入、MoE 的路由器。

Qwen3-30B-A3B：每层注意力 18,874,368 个参数，128 个专家各 4,718,592 个，路由器 262,144 个；
总计约 305 亿，每个 token 只用 8 个专家，激活约 33 亿——这就是名字里 "A3B" 的含义。
MoE 省的是**计算**，不是**显存**：全部专家的权重都要放在显存里。
