---
title: 流式加载权重：合并投影、打包专家
chapter: compute/models.md
difficulty: 中等
tags: [权重加载, 生成器, MoE]
---
Hugging Face 的 checkpoint 里 q、k、v 三个投影、gate、up 两个投影是分开存的，而 mini-sglang 的模型用合并后的 `qkv_proj`、`gate_up_proj`（一次大矩阵乘）；
MoE 的每个专家也是分开存的，模型里却是一个 `[E, ...]` 的大张量。实现一个**流式**的转换器 `load_weights(items, num_experts=0)`：

- `items` 是 `(name, array)` 的迭代器，顺序**任意**（同一组的投影可能分散在不同的文件里）；
- 名字里含 `.q_proj`、`.k_proj`、`.v_proj` 的，合并成把这一段换成 `.qkv_proj` 的名字，按 q、k、v 的顺序沿第 0 维拼接；`.gate_proj`、`.up_proj` 合并成 `.gate_up_proj`（gate 在前）；
- `num_experts > 0` 时，名字形如 `{prefix}.experts.{i}.{rest}` 的张量（合并之后的名字）打包成 `{prefix}.experts.{rest 去掉末尾的 .weight}`，沿新的第 0 维按专家编号堆叠；
- 其余的原样输出；
- 这是一个**生成器**：一个组一凑齐就立即 `yield`，不能先把全部权重读进来再处理（测试会检查缓冲区里同时存在的张量数量）；
- 结束时还有没凑齐的组，抛出 `ValueError`。

```text
model.layers.0.self_attn.q_proj.weight  (32, 16)  ┐
model.layers.0.self_attn.k_proj.weight  ( 8, 16)  ├─> model.layers.0.self_attn.qkv_proj.weight  (48, 16)
model.layers.0.self_attn.v_proj.weight  ( 8, 16)  ┘
model.layers.0.mlp.experts.3.gate_up_proj.weight  ─> model.layers.0.mlp.experts.gate_up_proj  [E, ...]
```

<!-- 题解 -->
两个缓冲区：`merge_buf[合并后的名字][槽位]` 和 `expert_buf[打包后的名字][专家编号]`。
每读到一个张量，先看它是否属于合并组：凑齐了就得到合并后的 `(name, tensor)`，否则 `continue`；再看合并后的名字是否是专家张量：凑齐 E 个就 `np.stack`，否则 `continue`；都不是就直接 `yield`。

正则 `^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$` 可以解析专家名字。书中的实现还在这一步做张量并行切分（下一道题），峰值内存只比一个完整张量多一点。
