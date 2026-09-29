---
title: 估算：decode 注意力的计算访存比（MHA、GQA、MLA）
chapter: inference/kv-cache.md
difficulty: 中等
tags: [估算, 算术强度, MLA, GQA, 屋顶线]
---
面试常问："MLA 的 decode 是算力瓶颈还是带宽瓶颈？计算访存比和序列长度、batch size 有没有关系？"
考虑 decode 时**一条序列**、每层的注意力：已有 $s$ 个 token 的 KV，本步有 `q_len` 个 query token（普通 decode 为 1，带 MTP 或投机解码验证时大于 1）。
只算读 KV 的字节数和注意力的矩阵乘 FLOPs（$s$ 很大时 Q 与输出的读写可以忽略）：

- **MHA / GQA**：KV 字节 $= 2 \cdot n_{kv} \cdot d_h \cdot s \cdot b$；FLOPs $= 4 \cdot q \cdot n_h \cdot d_h \cdot s$（$QK^\top$ 与 $PV$ 各 $2 \cdot q \cdot n_h \cdot d_h \cdot s$）；
- **MLA（矩阵吸收后）**：每 token 只存 $r + \rho$ 个元素（潜向量维度 $r$ = `kv_lora_rank`，RoPE 维度 $\rho$ = `rope_dim`），KV 字节 $= (r+\rho) \cdot s \cdot b$；
  打分在 $r+\rho$ 维上做，输出在 $r$ 维潜空间里做：FLOPs $= 2 \cdot q \cdot n_h \cdot s \cdot (r+\rho) + 2 \cdot q \cdot n_h \cdot s \cdot r$。

实现：

1. `decode_attn_intensity(kind, n_heads, n_kv_heads=None, head_dim=128, kv_lora_rank=512, rope_dim=64, q_len=1, dtype_bytes=2)`：
   `kind` 为 `"mha"`、`"gqa"`、`"mla"`，返回 FLOPs / 字节（MHA 时 `n_kv_heads` 等于 `n_heads`）；
2. `tp_intensity(kind, n_heads, tp, n_kv_heads=None, **kw)`：张量并行度为 `tp` 时每张卡上的计算访存比。
   注意力头按 `tp` 切分；MLA 的潜向量 KV 每张卡都要完整保存一份；GQA 的 KV 头也按 `tp` 切分，但 `tp` 超过 KV 头数时每张卡至少保存 1 个 KV 头；
3. `bound(intensity, peak_tflops, bw_gbs)`：返回 `"compute"` 或 `"memory"`（等于屋脊点时算 `"compute"`）。

```python
decode_attn_intensity("mha", 32)                     # 1.0
decode_attn_intensity("gqa", 64, n_kv_heads=8)       # 8.0
decode_attn_intensity("mla", 128)                    # 约 241.8，H100 的屋脊点约 295：仍是带宽瓶颈
```

<!-- 题解 -->
几个结论（面试时直接说）：

- 计算访存比**与序列长度无关**（FLOPs 和字节都与 $s$ 成正比），**与 batch 无关**（每条序列读自己的 KV，batch 变大只是多做几份同样强度的工作）；
- 它只取决于"一次读进来的 KV 被多少个 query 头复用"：MHA 为 1，GQA 为组大小，MLA 所有头共享同一个潜向量，所以约为 $1.9 \times$ 头数；
- 128 个头的 MLA 约 242 FLOPs/Byte，已经接近 H100 的屋脊点（约 295）；再乘上 `q_len`（MTP 一次验证 2 个 token）就越过屋脊点变成算力瓶颈，这正是 FlashMLA 这类 kernel 要按算力瓶颈来优化的原因；
- 张量并行把 MLA 的头切开但 KV 不能切，每张卡的计算访存比按 `tp` 下降，所以 MLA 模型的注意力部分更适合数据并行（DP Attention）而不是张量并行。
