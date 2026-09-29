---
title: 从 Nsight 时间线里找瓶颈
chapter: tools/profiling.md
difficulty: 中等
tags: [性能分析, 区间合并, 时间线]
---
Nsight Systems 导出的时间线可以简化成一串事件：

```python
{"name": "gemm_kernel", "cat": "kernel", "ts": 100.0, "dur": 50.0, "stream": 7}   # 时间单位：微秒
```

`cat` 是 `"kernel"`、`"memcpy"`（都在 GPU 上执行）或 `"cpu"`（CPU 上的函数，比如 `cudaLaunchKernel`）。多个 stream 上的 GPU 事件可能重叠。实现 `analyze(events)`，返回字典：

| 键 | 含义 |
| --- | --- |
| `span` | 从最早的 GPU 事件开始到最晚的 GPU 事件结束的时长 |
| `gpu_busy` | GPU 事件（kernel + memcpy）的时间区间**合并**之后的总长度（重叠部分只算一次） |
| `utilization` | `gpu_busy / span` |
| `largest_gap` | `span` 内 GPU 完全空闲的最长一段 `(开始, 结束)`；没有空闲时为 `None` |
| `top_kernels` | 按总耗时从大到小的前 3 个 kernel：`[(name, 总耗时, 次数), ...]`，总耗时相同按名字排序 |
| `memcpy_ratio` | memcpy 的总耗时（直接求和，不合并）占 kernel 与 memcpy 总耗时之和的比例 |

没有 GPU 事件时返回 `{"span": 0.0, "gpu_busy": 0.0, "utilization": 0.0, "largest_gap": None, "top_kernels": [], "memcpy_ratio": 0.0}`。

<!-- 题解 -->
区间合并：按开始时间排序，维护当前合并区间的结束位置；下一个区间的开始大于当前结束时，就出现了一段空闲（记录最长的一段），否则延长当前区间。

利用率低、空闲段多，通常说明 CPU 发射 kernel 跟不上（小 kernel 太多），这时该考虑 CUDA Graph 或 kernel 融合；
`memcpy_ratio` 高说明数据搬运占比大，考虑锁页内存 + 异步拷贝与计算重叠。
