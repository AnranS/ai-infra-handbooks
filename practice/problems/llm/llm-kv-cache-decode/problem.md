---
title: 用 KV Cache 做增量解码
chapter: inference/kv-cache.md
difficulty: 中等
tags: [KV Cache, prefill, decode]
---
模板里给出了一个 Qwen 风格小模型的完整前向 `forward(tokens, w, cfg)`（和"从零组装"那道题的结构相同），每生成一个 token 都要把整个序列重新算一遍。
实现 `IncrementalDecoder(w, cfg)`，用 KV Cache 让每一步只算新 token：

- `prefill(tokens)`：处理提示词（一次算完所有位置），把每一层的 K、V（**已经加过 RoPE 的 K**）存进缓存，返回 `(T, vocab)` 的 logits；
- `step(token)`：输入一个新 token（位置是当前缓存长度），只对它做一次前向：算出它的 q、k、v，把 k、v 追加到缓存，用 q 和缓存里的全部 K、V 做注意力。返回 `(vocab,)` 的 logits；
- `length` 属性：当前缓存了多少个位置。

要求：`prefill` + 若干次 `step` 得到的 logits 与对完整序列调用 `forward` 的结果一致（误差 1e-8 以内）；`step` 里**不能调用 `forward`**（测试会检查）。

<!-- 题解 -->
把 `forward` 拆开：每层的注意力部分改成"先算本次输入的 q、k、v → 对 k 加 RoPE（位置是 `self.length + arange(T)`）→ 追加到缓存 → q 对缓存里全部 K、V 做注意力"。
因果掩码也要按绝对位置算：本次输入的第 $i$ 个 token 位置是 `start + i`，只能看到缓存里位置 $\le$ `start + i` 的 key。
`step` 就是 `T = 1` 的情况，这时不需要掩码。

缓存里存的是加过 RoPE 的 K：位置编码只和 token 自己的位置有关，加一次就够了，以后不再变。
这就是推理引擎 prefill / decode 两阶段的由来：prefill 是计算密集的大矩阵乘，decode 每步只有一个 token，瓶颈变成读 KV Cache 和权重。
