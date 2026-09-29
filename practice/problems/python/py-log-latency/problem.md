---
title: 从访问日志统计延迟分位数
chapter: engineering/stdlib.md
difficulty: 中等
tags: [re, datetime, collections, statistics]
---
推理服务的访问日志每行形如：

```text
2026-05-17T10:00:01.250Z INFO req=a1 POST /v1/chat/completions status=200 latency_ms=812.5 tokens=256
2026-05-17T10:00:01.900Z WARN req=a2 POST /v1/completions status=429 latency_ms=3.1 tokens=0
```

实现 `summarize(lines, start=None, end=None)`，返回一个字典：键是接口路径，值是这个接口的统计：

```python
{"/v1/chat/completions": {"count": 2, "errors": 0, "p50": 812.5, "p99": 950.0, "tokens_per_s": 315.5}, ...}   # 示意
```

- 只统计时间戳在 `[start, end)` 内的行（`start`、`end` 是带时区的 `datetime`，为 `None` 表示不限制）；
- 格式不对的行（缺字段、数字解析失败）直接跳过；
- `count`：请求数；`errors`：状态码 `>= 500` 的请求数；
- `p50`、`p99`：`latency_ms` 的分位数，用**最近秩法**：把延迟排序，第 $\lceil q \cdot n \rceil$ 个（从 1 开始数）；
- `tokens_per_s`：这个接口所有请求的 `tokens` 之和，除以 `latency_ms` 之和（换算成秒），保留 1 位小数；延迟之和为 0 时为 `0.0`。

字段在一行里的顺序可能不同，但时间戳总在行首，接口路径总是以 `/` 开头的那一段。

<!-- 题解 -->
- 时间戳：`datetime.fromisoformat(ts.replace("Z", "+00:00"))`（Python 3.11 起 `fromisoformat` 也直接支持 `Z`）；
- `key=value` 字段用 `re.findall(r"(\w+)=(\S+)", line)` 一次取出，路径用 `re.search(r"\s(/\S*)", line)`；
- 按路径分组用 `defaultdict(list)`；最近秩法的下标是 `math.ceil(q * n) - 1`。
