---
title: 引擎单步耗时分解与重叠调度的收益
chapter: perf/profiling.md
difficulty: 中等
tags: [性能分析, CPU 开销, 重叠调度]
---
用 profiler 采到了推理引擎连续若干步的时间线（毫秒），每一步是一个字典：

```python
{"schedule": 0.8, "prepare": 1.2, "forward": 6.0, "sample": 0.3, "process": 1.5}
```

其中 `forward` 在 GPU 上，其余都在 CPU 上；普通循环里它们**依次执行**（CPU 等 GPU 算完才处理结果）。实现 `analyze(steps)`，返回：

| 键 | 含义 |
| --- | --- |
| `step_ms` | 普通循环的平均每步时间（各阶段之和的平均） |
| `gpu_idle` | GPU 空闲的比例：`1 - Σforward / Σ每步总时间` |
| `breakdown` | 每个阶段占总时间的比例（字典，键与输入相同） |
| `bottleneck` | CPU 阶段里总耗时最大的那个阶段名 |
| `overlap_step_ms` | 如果改成**重叠调度**：GPU 算第 N+1 步时 CPU 处理第 N 步。稳态下每步时间约为 `max(CPU 部分, GPU 部分)`（CPU 部分 = 除 `forward` 外各阶段之和），取所有步的平均 |
| `overlap_speedup` | `step_ms / overlap_step_ms` |

所有步的阶段名相同（至少有 `forward` 和一个 CPU 阶段）。

<!-- 题解 -->
对每一步算 `cpu = sum(v for k, v in s.items() if k != "forward")`、`gpu = s["forward"]`，普通循环 `cpu + gpu`，重叠后 `max(cpu, gpu)`。
小模型、小 batch 时 `forward` 只有几毫秒，CPU 部分占比很高，重叠调度能带来接近 2 倍的提升；
如果 CPU 部分本身比 GPU 还慢（例如 `process` 里的反分词太慢），重叠之后瓶颈就是 CPU，需要继续优化 CPU 端或者把反分词挪到别的进程（这就是 SGLang、vLLM 都把 detokenizer 放在独立进程的原因）。
