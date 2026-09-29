import math


def _limit(points, slo):
    if points[0][1] > slo:
        return 0.0
    for (q0, v0), (q1, v1) in zip(points, points[1:]):
        if v1 > slo:
            return q0 + (slo - v0) * (q1 - q0) / (v1 - v0)
    return float(points[-1][0])


def max_qps_under_slo(curve, ttft_slo, tpot_slo):
    a = _limit([(q, t) for q, t, _ in curve], ttft_slo)
    b = _limit([(q, p) for q, _, p in curve], tpot_slo)
    return float(min(a, b))


def replicas_needed(target_qps, per_replica_qps, headroom=0.2):
    if per_replica_qps <= 0:
        raise ValueError("per_replica_qps 必须为正")
    return math.ceil(target_qps / (per_replica_qps * (1 - headroom)) - 1e-12)


def littles_law_concurrency(qps, mean_latency_s):
    return qps * mean_latency_s
