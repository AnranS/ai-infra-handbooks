---
title: MLA 的吸收路径：只用潜向量缓存做 decode
chapter: moe/mla.md
difficulty: 困难
tags: [MLA, 权重吸收, decode, KV Cache]
requires: [numpy]
---
MLA 每个 token 只缓存潜向量 $c$（`DC` 维）和所有头共享的 RoPE 键 $k^R$（`ROPE` 维）。decode 时不把缓存展开成每个头的 K、V，而是把两个上投影**吸收**进别的矩阵：

- $W_{UK}$ 吸收进 query：第 $h$ 个头的非位置 query $q^N = x W_{q}^{N,h}$，分数里的 $q^{N\top} W_{UK}^{h\top} c$ 可以写成 $(x\, W_{q}^{N,h} W_{UK}^{h\top})\, c$，于是 hidden 直接投影到潜空间；
- $W_{UV}$ 吸收进输出投影：头 $h$ 的输出 $(p\,c)\, W_{UV}^h$ 再乘 $W_o$ 里对应的那 `DV` 行，两个矩阵可以预先乘成一个 `DC × D` 的矩阵。

模板里给出了 `rope(x, pos)` 和展开路径的参考实现 `mla_reference(h, W)`（一次算完整个序列的因果注意力），权重的形状见模板的注释。实现：

1. `absorb(W)`：返回一个字典，至少包含 `"W_q_lat"`（`(H, D, DC)`）、`"W_q_rope"`（`(H, D, ROPE)`）、`"W_o_lat"`（`(H, DC, D)`），以及 decode 还需要的其他东西（生成缓存用的 `W_dkv`、`W_kr` 等）；
2. `decode_step(x, pos, cache, A)`：`x` 是位置 `pos` 上新 token 的 hidden（`(D,)`），`cache` 是 `{"c": [...], "kr": [...]}`（每个元素是一个 token 的潜向量、旋转过的 RoPE 键），`A` 是 `absorb` 的结果。先把新 token 的 `c`、`kr` 追加进缓存，再用吸收后的矩阵算出这个位置的输出（`(D,)`），不要展开 K、V。

逐个位置调用 `decode_step` 得到的输出应该和 `mla_reference` 一次算出的逐行相同。分数的缩放是 $1/\sqrt{\text{NOPE} + \text{ROPE}}$。

<!-- 题解 -->
吸收只是矩阵乘法的结合律：$q^{N\top} k^N = (x W_q^{N})(W_{UK} c)^{\top}$ 里的 $W_{UK}$ 挪到 query 一侧，加权求和 $\sum_s p_s (c_s W_{UV})$ 里的 $W_{UV}$ 提到求和外面。RoPE 部分挡在中间（旋转矩阵依赖位置），所以 $q^R$、$k^R$ 保持原样，单独算一项分数。

`W_q_lat`、`W_o_lat` 在加载权重时算一次就行。吸收之后，所有头面对的是同一份 `DC + ROPE` 维的"键"和 `DC` 维的"值"——相当于一个头维很大的 MQA，decode kernel（FlashMLA、FlashInfer 的 MLA kernel）正是按这个形状写的。代价是每对 (query, key) 的点积从 `NOPE + ROPE` 维变成了 `DC + ROPE` 维，所以 prefill（新 token 多）时展开路径反而更快，见本章"谁快"一节。
