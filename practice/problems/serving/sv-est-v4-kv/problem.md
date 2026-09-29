---
title: 估算：压缩 + 稀疏 + 滑窗注意力的 KV 账本
chapter: frontier/new-models.md
difficulty: 简单
tags: [估算, KV Cache, 稀疏注意力, DeepSeek-V4]
---
DeepSeek-V4 这类模型的注意力分三路：每层都存最近 `window` 个 token 的原始 KV（滑窗）；C4 层每 4 个 token 压成一条 KV，再为每条存一个索引器的 key；C128 层每 128 个 token 压成一条 KV；压缩比为 0 的层只有滑窗。
`config.json` 里的 `compress_ratios` 逐层给出压缩比（0、4 或 128）。

实现（字节数都用 `entry_bytes` 表示一条 KV、`index_bytes` 表示一个索引器 key）：

1. `per_token_bytes(ratios, entry_bytes=584, index_bytes=132)`：上下文每多一个 token，KV 平均多占多少字节（C4 层：`(entry_bytes + index_bytes) / 4`；C128 层：`entry_bytes / 128`；只有滑窗的层：0）；
2. `fixed_bytes(ratios, window=128, entry_bytes=584)`：每个请求的滑窗部分（上下文不短于窗口时），**每一层**都有；
3. `request_bytes(ratios, context_len, window=128, entry_bytes=584, index_bytes=132)`：一个请求的 KV 总字节数 = 每 token 增量 × 上下文长度 + 滑窗部分（上下文短于窗口时，滑窗只存实际的 token 数；这里不考虑压缩块没凑满的零头，按 `context_len / ratio` 的实数算）；
4. `attended(ratio, pos, topk=512, window=128)`：位置 `pos`（从 0 开始）的 query 在这一层要看多少条 KV = 滑窗里的原始 token（`min(window, pos + 1)`）+ 已经压好的条目 `(pos + 1) // ratio`（C4 层最多看 `topk` 条，C128 层全看；压缩比为 0 的层只有滑窗）。

```python
flash = [0, 0] + [4, 128] * 20 + [4]           # DeepSeek-V4-Flash：43 层
per_token_bytes(flash)                         # 3850.25
fixed_bytes(flash)                             # 43 × 128 × 584 = 3214336
attended(4, 1_048_575)                         # 128 + 512 = 640
attended(128, 1_048_575)                       # 128 + 8192 = 8320
```

<!-- 题解 -->
- 每层的账本分成"随上下文增长"和"每个请求固定"两部分，分开算最清楚。V4-Flash 每 token 约 3.8 KB，只有 V3.2（每层 MLA 656 B + 索引器 132 B，61 层，约 47 KB）的十二分之一；
- decode 的注意力工作量几乎和上下文长度无关：C4 层固定看 top-k 条压缩 KV 加滑窗，C128 层的条目只有上下文的 1/128；随上下文线性增长的只剩索引器打分；
- 固定部分（滑窗、压缩器状态）让短请求相对更贵，高并发的短请求场景要按它来规划并发数。
