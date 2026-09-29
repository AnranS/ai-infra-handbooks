---
title: 束搜索
chapter: inference/decoding.md
difficulty: 困难
tags: [束搜索, 解码, 长度惩罚]
---
实现束搜索 `beam_search(step_fn, bos, eos, beam_size, max_len, length_penalty=1.0)`：

- `step_fn(prefix)`：`prefix` 是 token 的元组（以 `bos` 开头），返回下一个 token 的**对数概率**（一维 numpy 数组，长度为词表大小）；
- 每个假设（hypothesis）有 token 序列和累计对数概率 `score`。一开始只有一个假设 `(bos,)`，`score = 0`；
- 每一步：对每个**活着的**假设，用 `step_fn` 扩展出所有"加一个 token"的候选，候选的 `score` 是原来的加上这个 token 的对数概率。
  把所有候选按 `score` **从大到小**排序（并列时按 token 序列的字典序从小到大），依次处理：
  - 以 `eos` 结尾的候选放进"已完成"列表（已完成的数量达到 `beam_size` 后不再加入）；
  - 其他候选成为下一步活着的假设，最多 `beam_size` 个，够了就停止处理剩下的候选；
- 结束条件：已完成的假设达到 `beam_size` 个，或者没有活着的假设，或者已经生成了 `max_len` 步。结束时把还活着的假设也并入已完成列表；
- 排序打分：`score / L ** length_penalty`，`L` 是生成的 token 数（不含 `bos`，含 `eos`）。

返回所有已完成的假设，按打分从大到小（并列按序列字典序），每个元素是 `(tokens, normalized_score)`，`tokens` 是**不含** `bos` 的元组。

<!-- 题解 -->
每一步的候选数是 `活着的假设数 × 词表大小`，排序用 `key=lambda c: (-c.score, c.tokens)`。

为什么要单独维护"已完成"列表：以 `eos` 结尾的假设不能再扩展，但它可能比后面更长的假设更好；如果把它和活着的假设混在一起，就会占掉一个束的位置。
长度惩罚：累计对数概率是负数，越长越小，不做归一化时束搜索偏爱短句；`length_penalty > 0` 时除以 $L^\alpha$ 抵消这种偏好。

Hugging Face 的实现还有 `early_stopping` 等细节，这里是一个定义清楚的简化版本。
