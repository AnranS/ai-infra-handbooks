---
title: warp 调度器的利用率
chapter: arch/gpu-sm.md
difficulty: 中等
tags: [GPU, warp 调度, 延迟掩盖, 模拟]
---
一个 warp 调度器每周期最多发射一条指令。每个 warp 反复做同一件事：连发 `ilp` 条互不依赖的访存指令，等 `latency` 个周期数据回来，再连发 `compute` 条计算指令，如此循环。调度策略是**贪心再取最老**（GTO）：一直发射当前 warp，直到它要等数据，才换成编号最小的就绪 warp。

实现 `utilization(warps, ilp, compute, latency, rounds)`：模拟到**任意一个 warp 完成 `rounds` 轮**为止，返回"发射了指令的周期数 ÷ 总周期数"。所有 warp 从第 0 周期开始，都处于"要发访存"的状态。

```python
utilization(1, 1, 16, 500, 100)     # 约 0.033：一个 warp 藏不住 500 周期的延迟
utilization(16, 1, 16, 500, 100)    # 约 0.52：16 个 warp 也只能填一半
utilization(8, 4, 64, 500, 100)     # 约 0.95：提高 ILP 后 8 个 warp 就够了
```

<!-- 题解 -->
逐周期模拟：每个 warp 有阶段（发访存 / 等数据 / 发计算）、本阶段还剩几条指令、数据到达的周期。每个周期挑一个就绪的 warp 发一条指令；当前 warp 一旦进入"等数据"，就换编号最小的就绪 warp；一个都没有就空转一个周期。

一个 warp 一轮要 `ilp + latency + compute` 个周期，其中只有 `ilp + compute` 个周期在发射，所以填满调度器需要约 `(ilp + latency + compute) / (ilp + compute)` 个 warp。`ilp=1、compute=16、latency=500` 时是 30 个，超过了一个调度器 16 个 warp 的上限——**光靠堆 warp 数掩盖不了访存延迟**。把 `ilp` 提到 4、`compute` 提到 64，同样的访存/计算比例，只要 8 个 warp 就能填满。

GTO 也比轮转（每个周期换下一个 warp）好：轮转让所有 warp 步调一致，同时发访存、同时等，同样的参数下利用率从 0.95 掉到 0.53。这就是高性能 kernel 让每个线程处理多个元素、用 `float4` 一次读 16 字节的定量理由，也是 GPU 版的 Little 定律：需要的在途工作量 = 延迟 × 吞吐。
