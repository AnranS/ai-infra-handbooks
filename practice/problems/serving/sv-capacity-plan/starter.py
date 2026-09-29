import math


def max_qps_under_slo(curve, ttft_slo, tpot_slo):
    ok = [q for q, t, p in curve if t <= ttft_slo and p <= tpot_slo]   # 只看测量点，没有插值
    return float(max(ok)) if ok else 0.0


def replicas_needed(target_qps, per_replica_qps, headroom=0.2):
    pass


def littles_law_concurrency(qps, mean_latency_s):
    pass
