---
title: 估算：RL 每一步把新权重送到推理实例要多久
chapter: frontier/rl-async.md
difficulty: 简单
tags: [估算, RL, 权重同步, 网络]
---
每个训练步之后，新权重（`weight_bytes` 字节）要送到所有推理实例，每个实例有 `gpus_per_inst` 张卡，每张卡只需要收到自己那 1/`gpus_per_inst`。实现（带宽单位都是字节/秒，时间单位秒）：

1. `colocated_s(weight_bytes, gpus_per_inst, nvlink_bw)`：训练和推理共置在同一批卡上，每张卡通过机内 NVLink 拿到自己那一片的时间；
2. `disaggregated_s(weight_bytes, n_inst, nic_bw, gpus_per_inst, mode)`：训练和推理分离部署，每张卡的网卡带宽 `nic_bw`：
    - `"naive"`：训练端汇总到一张卡，依次把整份权重发给每个实例；
    - `"parallel"`：训练端用 `gpus_per_inst` 张卡并行发送，每张卡发一片，但每个实例都要单独发一遍；
    - `"relay"`：在 `"parallel"` 的基础上，实例之间分块流水接力（收到一块就转发给下一个实例），总时间和一个实例几乎一样，实例多于一个时加 5% 的开销；

    其他 `mode` 抛出 `ValueError`；
3. `max_instances(budget_s, weight_bytes, nic_bw, gpus_per_inst, mode)`：同步时间不超过 `budget_s` 时，最多能带多少个推理实例；没有上限时返回 `None`，一个都带不了时返回 0。

```python
W = 1e12                                    # 万亿参数、FP8
colocated_s(W, 8, 450e9)                    # 0.278
disaggregated_s(W, 64, 50e9, 8, "relay")    # 2.625
```

<!-- 题解 -->
所有情况都是"要搬的字节 / 能并行用的带宽"。共置时数据不出机器，每张卡只拿 1/8，走 NVLink，不到一秒。分离部署时，朴素做法把整份权重一个实例一个实例地发，时间和实例数成正比，64 个实例要二十多分钟；发送端也用 8 张卡并行，快 8 倍，但仍和实例数成正比；实例之间流水接力之后，发送端只需要把权重发出去一次，后面的实例边收边转发，总时间几乎不随实例数增长。

`max_instances` 把这个结论反过来问：在一步训练允许的同步时间里，前两种做法能带的实例数有上限（预算 ÷ 每个实例的时间，向下取整），接力的做法只要一个实例的时间（含 5% 开销）在预算内，实例数就没有上限。
