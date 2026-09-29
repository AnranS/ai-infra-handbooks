---
title: Req 的几个长度
chapter: compute/core.md
difficulty: 简单
tags: [数据结构, prefill, decode]
---
mini-sglang 的 `Req` 用几个长度描述一个请求在任意时刻的状态。用纯 Python 实现它（`input_ids` 用列表代替张量）：

```python
Req(input_ids, cached_len, output_len)
```

| 属性 | 含义 |
| --- | --- |
| `input_ids` | 提示词 + 已生成的 token |
| `cached_len` | 前 `cached_len` 个 token 的 KV 已经在缓存里 |
| `device_len` | 本轮前向结束后缓存里会有多少个 token 的 KV；创建时等于 `len(input_ids)` |
| `max_device_len` | `len(input_ids) + output_len`（创建时的值，之后不变） |
| `remain_len` | `max_device_len - device_len`：还能再生成几个 |
| `extend_len` | `device_len - cached_len`：本轮要送进模型的 token 数 |
| `can_decode` | `remain_len > 0` |

方法：

- `complete_one()`：一轮前向完成：本轮的 token 都进了缓存（`cached_len = device_len`），下一轮再多算一个（`device_len += 1`）；
- `append_host(token)`：把采样出的 token 追加到 `input_ids`。

创建时要检查 `0 <= cached_len < device_len <= max_device_len`（`output_len >= 0`），不满足抛出 `ValueError`。

<!-- 题解 -->
和书中 `minisgl/core.py` 一致：`device_len`、`max_device_len` 在 `__post_init__` 里算好，其余是 property。

一个请求的生命周期（提示词 5 个 token、命中缓存 2 个、最多生成 3 个）：

| 时刻 | cached_len | device_len | extend_len | 说明 |
| --- | --- | --- | --- | --- |
| 创建 | 2 | 5 | 3 | prefill 要算 3 个 token |
| prefill 后 `complete_one` | 5 | 6 | 1 | 之后每轮 decode 算 1 个 |
| 第 1 轮 decode 后 | 6 | 7 | 1 | |
| 第 2 轮 decode 后 | 7 | 8 | 1 | `remain_len = 0`，不能再 decode |

注意 `complete_one` 在**发射**前向时就调用（重叠调度需要提前推进 CPU 上的状态），而 `append_host` 在拿到采样结果之后才调用，所以两者之间 `len(input_ids)` 会比 `device_len` 少 1。
