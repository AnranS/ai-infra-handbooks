"""滚动更新与自动扩缩容的时间线模拟。

滚动更新：Deployment 按 maxSurge / maxUnavailable 调整新旧两个 ReplicaSet 的副本数，
推理服务的特殊之处是"就绪"很慢（要加载几十 GB 权重）、"退出"也很慢（要把手上的请求做完）。
"""


def rollout(replicas, ready_s, drain_s, max_surge, max_unavailable, step_s=5, horizon_s=600):
    """返回 [(时刻, 可用副本数, 总副本数), ...]，以及整个发布耗时"""
    old = [{"ready": True} for _ in range(replicas)]      # 旧版本：都已就绪
    new = []                                              # 新版本：正在启动
    timeline, t = [], 0
    while t <= horizon_s:
        for pod in new:                                   # 新 Pod 到点就绪
            if not pod["ready"] and t >= pod["at"] + ready_s:
                pod["ready"] = True
        old = [p for p in old if not (p.get("draining") and t >= p["drain_at"] + drain_s)]
        available = sum(1 for p in old if p["ready"] and not p.get("draining")) + \
                    sum(1 for p in new if p["ready"])
        total = len(old) + len(new)
        timeline.append((t, available, total))
        if not old and all(p["ready"] for p in new) and len(new) == replicas:
            break
        if total < replicas + max_surge and len(new) < replicas:          # 还能再起新的
            new.append({"ready": False, "at": t})
        elif available - 1 >= replicas - max_unavailable:                  # 可以开始撤一个旧的
            for pod in old:
                if not pod.get("draining"):
                    pod["draining"], pod["drain_at"] = True, t
                    break
        t += step_s
    return timeline, t


def summarize(timeline, replicas):
    """发布过程中最少可用了几个副本、低于目标容量的时间有多久"""
    worst = min(a for _, a, _ in timeline)
    degraded = sum(1 for _, a, _ in timeline if a < replicas)
    return worst, degraded


def hpa_step(current, desired_metric, target_metric, replicas, min_r, max_r, tolerance=0.1):
    """HPA 的核心公式：期望副本数 = 当前副本数 × (当前指标 / 目标指标)，偏差在容忍带内就不动"""
    ratio = desired_metric / target_metric
    if abs(ratio - 1) <= tolerance:
        return replicas
    return max(min_r, min(max_r, -(-int(replicas * ratio * 100) // 100)))


def simulate_hpa(load, replicas, target_qps_per_pod, ready_s, step_s=15, min_r=1, max_r=20):
    """load 是每一步的总 QPS；返回 [(步, 负载, 副本数, 每副本 QPS, 是否过载)]"""
    pending = []                                       # 正在启动的副本：(就绪时刻, 数量)
    out = []
    for i, qps in enumerate(load):
        t = i * step_s
        ready_now = sum(n for at, n in pending if t >= at)
        pending = [(at, n) for at, n in pending if t < at]
        replicas += ready_now
        per_pod = qps / replicas if replicas else float("inf")
        out.append((t, qps, replicas, round(per_pod, 1), per_pod > target_qps_per_pod * 1.2))
        want = hpa_step(qps, per_pod, target_qps_per_pod, replicas, min_r, max_r)
        if want > replicas:
            pending.append((t + ready_s, want - replicas))   # 扩容要等启动
        elif want < replicas:
            replicas = want                                   # 缩容立刻生效（真实 HPA 还有冷却窗口）
    return out
