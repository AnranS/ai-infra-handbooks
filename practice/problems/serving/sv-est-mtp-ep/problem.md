---
title: 估算：大规模 EP 的 decode 里，MTP 什么时候变快、什么时候变慢
chapter: moe/mtp-sparse.md
difficulty: 中等
tags: [估算, MTP, 投机解码, 专家并行]
---
DeepSeek-V3 规模的 MoE 在大规模 EP 下 decode，一张卡、一层的时间按正文的单层模型估算（常量已经在模板里给出）：

- **注意力**：`max((ATTN_W + seqs × ctx × 1152) / HBM, tokens × ctx × 278528 / BF16)`——权重和每个请求的潜向量 KV 只读一次（按请求计费），计算量按 token 计费；
- **MoE**：`max(LOCAL_EXPERTS × EXPERT_W / HBM, tokens × 9 × EXPERT_FLOP / FP8)`；
- **all-to-all**：`tokens × 8 × TOK_BYTES / NIC`，每个 token 发给 8 个专家再收回来；
- 一层的时间 = `max(注意力 + MoE, all-to-all)`（假设双 batch 重叠把通信完全藏在计算后面）。

用 MTP 做 k 个草稿时，每个请求验证 k+1 个 token，MTP 模块自回归跑 k 次（每次一层、每个请求一个 token）。实现：

1. `expected_advance(accept)`：`accept[i]` 是前面的草稿都被接受时第 i+1 个草稿被接受的概率，每步期望前进 `1 + p1 + p1·p2 + …` 个 token；
2. `layer_time(seqs, tokens, ctx)`：上面的单层模型（秒）；
3. `mtp_throughput(seqs, ctx, k, accept)`：每张卡每秒生成的 token 数 = `seqs × expected_advance(accept[:k]) / 一步的时间`，一步的时间 = `LAYERS × layer_time(seqs, seqs × (k+1), ctx) + k × layer_time(seqs, seqs, ctx)`；
4. `best_k(seqs, ctx, accept, k_max=3)`：0～k_max 里吞吐最高的草稿数（相同时取小的）。

```python
A = [0.85, 0.75, 0.65]
mtp_throughput(16, 4096, 1, A) / mtp_throughput(16, 4096, 0, A)    # 1.82：每卡 16 个请求时 1 个草稿接近 1.8 倍
best_k(64, 4096, A)                                                # 0：每卡 64 个请求、上下文 4K 时 MTP 反而变慢
```

<!-- 题解 -->
验证 k+1 个 token 时，按请求计费的部分（读注意力权重、读潜向量 KV、读专家权重）不变，按 token 计费的部分（计算和 all-to-all）乘以 k+1。每卡请求少时，一步的时间由读权重和 KV 决定，多验证几个 token 几乎免费，1 个草稿就有 1.8 倍；每卡请求多、上下文短时，all-to-all 已经是瓶颈，验证 token 让通信量翻倍，MTP 反而变慢；上下文变长后，读 KV 这个按请求计费的部分变大，MTP 又有收益。

所以 MTP 不是"打开就快"，实际系统会按负载调整草稿数，高负载时甚至关掉。模型里的接受率、网络带宽都是假设值，结论的方向比具体数字重要。
