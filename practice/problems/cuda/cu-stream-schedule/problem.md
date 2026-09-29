---
title: 多个 stream 的执行时间线
chapter: tools/streams.md
difficulty: 中等
tags: [stream, event, 拷贝计算重叠]
---
模拟 CUDA stream 的调度，算出每个操作的开始、结束时间。按**发出顺序**给出一串操作：

```python
{"stream": 1, "op": "h2d", "dur": 4}       # 主机到设备的拷贝
{"stream": 1, "op": "kernel", "dur": 6}
{"stream": 1, "op": "d2h", "dur": 4}       # 设备到主机的拷贝
{"stream": 1, "op": "record", "event": "e1"}
{"stream": 2, "op": "wait", "event": "e1"}
```

硬件模型：

- 三个引擎：一个 H2D 拷贝引擎、一个 D2H 拷贝引擎、一个计算引擎（同一时刻只运行一个 kernel）；
- 每个引擎按**发出顺序**依次处理分给它的操作（先发出的先执行，即使后发出的早就可以开始）；
- 同一个 stream 里的操作按顺序执行：一个操作必须等它在同一 stream 里的前一个操作结束；
- `record`：记录一个事件，事件的时间是这个 stream 里之前所有操作都结束的时刻（stream 为空时是 0）；不占用引擎和时间；
- `wait`：这个 stream 之后的操作要等到该事件的时间；等一个还没被记录的事件时视为时间 0（和 CUDA 一样，等待未记录的事件立即返回）。

实现 `simulate(ops)`，返回列表：每个 `h2d`/`kernel`/`d2h` 操作对应一个 `(start, end)`（按发出顺序，`record`、`wait` 不出现在结果里），以及总完成时间：`(timeline, makespan)`。

```python
# 3 个 stream 各自 h2d(4) → kernel(6) → d2h(4)，按 stream 依次发出：
simulate(ops)[1]    # 26：拷贝和计算重叠；串行执行要 42
```

<!-- 题解 -->
按发出顺序依次处理，维护三个量：每个 stream 的"就绪时间"、每个引擎的"空闲时间"、每个事件的时间：

```text
h2d / kernel / d2h:  start = max(stream_ready[s], engine_free[e]);  end = start + dur
                     stream_ready[s] = engine_free[e] = end
record:              event[name] = stream_ready[s]
wait:                stream_ready[s] = max(stream_ready[s], event.get(name, 0))
```

因为引擎按发出顺序服务，发出顺序会影响重叠效果：在较老的 GPU（没有 Hyper-Q）上，"按 stream 深度优先"和"按阶段广度优先"发出，时间线可能差很多。
