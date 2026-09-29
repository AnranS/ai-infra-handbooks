---
title: 估算：滚动发布要多久、会不会过载、要多少额外的卡
chapter: ops/deploy.md
difficulty: 简单
tags: [估算, 发布, Kubernetes, 容量]
---
N 个推理实例，流量是满容量的 `load`（比如 0.8）。换版本时，每一轮下线 `max_unavailable` 个旧实例、额外拉起 `max_surge` 个新实例；新实例冷启动要 `cold_s` 秒，旧实例排空在途请求要 `drain_s` 秒，一轮按 `cold_s + drain_s` 算。实现：

1. `rollout(n, load, cold_s, drain_s, max_unavailable, max_surge)`：返回字典 `{"rounds", "minutes", "min_capacity", "extra_gpu", "overloaded"}`：每轮替换 `max_unavailable + max_surge` 个实例（两者都为 0 时抛出 `ValueError`），轮数向上取整；最低容量 `(n − max_unavailable) / n`；额外 GPU 的比例 `max_surge / n`；最低容量小于 `load` 时算过载；
2. `max_safe_unavailable(n, load)`：不过载的前提下，一次最多能下线几个实例；
3. `fastest_plan(n, load, cold_s, drain_s, extra_budget)`：额外的卡最多是 `extra_budget` 个实例的量（`max_surge <= extra_budget`），在不过载的前提下选总时间最短的 `(max_unavailable, max_surge)`；时间相同选额外卡少的，再相同选下线少的。

```python
rollout(16, 0.8, 180, 120, 1, 0)["minutes"]    # 80.0：逐个替换
rollout(16, 0.8, 180, 120, 4, 0)["overloaded"]  # True：一次下线 4 个，只剩 75% 的容量
```

<!-- 题解 -->
这组公式就是正文里比较四种发布策略的那段代码：每轮能替换的实例数 = 下线的 + 额外拉起的，轮数决定总时间；下线的实例越多越快，但剩下的容量要扛住流量；额外拉起的实例不降容量，代价是临时多用卡。

流量 80% 时，16 个实例一次最多下线 3 个（13/16 = 81%）；没有额外的卡时最快的方案就是每轮下线 3 个，6 轮、30 分钟。有 2 个实例的额外卡时，每轮下线 3 个 + 额外拉起 1 个就够了：每轮替换 4 个，4 轮、20 分钟；再多拉起 1 个（每轮 5 个）还是要 4 轮，白白多用卡——轮数是向上取整的，只有跨过整除的边界才会变快。蓝绿发布就是 `max_surge = n` 的极端：1 轮，但要双倍的卡。
