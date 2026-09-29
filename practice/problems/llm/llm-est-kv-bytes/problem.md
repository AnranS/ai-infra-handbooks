---
title: 估算：KV Cache 账本（MHA、GQA、MLA 与滑动窗口）
chapter: transformer/attention-variants.md
difficulty: 简单
tags: [估算, KV Cache, MLA, 滑动窗口]
---
KV Cache 的大小决定了能服务多长的上下文、多少并发。不同的注意力结构差别很大。`cfg` 是一个字典：

- `{"attn": "mha", "n_layers", "n_heads", "head_dim"}`：每层每 token 存 K、V 各 `n_heads × head_dim` 个元素；
- `{"attn": "gqa", "n_layers", "n_kv_heads", "head_dim"}`：只存 `n_kv_heads` 个头；
- `{"attn": "mla", "n_layers", "kv_lora_rank", "qk_rope_head_dim"}`：每层每 token 只存压缩后的潜向量和 RoPE 部分，共 `kv_lora_rank + qk_rope_head_dim` 个元素；
- 可选的 `"window": W` 和 `"n_global_layers": g`（只用于 mha / gqa）：其中 `g` 层是全局注意力，其余 `n_layers - g` 层是滑动窗口，只保留最近 `W` 个 token。

实现：

1. `kv_bytes_per_token(cfg, dtype_bytes=2)`：不考虑滑动窗口时，每个 token 在所有层上的 KV 字节数；
2. `kv_bytes(cfg, seq_len, dtype_bytes=2)`：一条长度为 `seq_len` 的序列实际要存的 KV 字节数（滑动窗口层最多存 `W` 个 token）；
3. `max_context(cfg, budget_gib, dtype_bytes=2)`：`budget_gib`（单位 $2^{30}$ 字节）能放下的最长单条序列（整数，向下取整）；
   如果所有层都是滑动窗口、预算又放得下一个窗口，返回 `math.inf`。

```python
llama3_70b = {"attn": "gqa", "n_layers": 80, "n_kv_heads": 8, "head_dim": 128}
dsv3 = {"attn": "mla", "n_layers": 61, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
kv_bytes_per_token(llama3_70b)   # 327680：每 token 320 KiB
kv_bytes_per_token(dsv3)         # 70272：约 68.6 KiB，只有前者的 1/4.7
max_context(llama3_70b, 40)      # 131072：40 GiB 刚好放下一条 128K 序列
```

<!-- 题解 -->
MLA 把每层的 K、V 压成一个 512 维的潜向量（加 64 维 RoPE），decode 时再用"矩阵吸收"直接在潜空间里算注意力，
所以 671B 的模型每 token 的 KV 反而比 70B 的 GQA 模型小很多。

滑动窗口层的 KV 不随序列长度增长：48 层里只有 8 层全局、窗口 4096 时，128K 序列的 KV 只有全部全局注意力时的约 1/5。
这就是"局部 + 全局交替"的模型能便宜地支持长上下文的原因。
