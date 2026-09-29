from checker import check, check_close, raises
from solution import littles_law_concurrency, max_qps_under_slo, replicas_needed

CURVE = [(1, 0.20, 0.020), (2, 0.25, 0.022), (4, 0.40, 0.028), (6, 0.90, 0.035), (8, 2.50, 0.050)]


def test_example():
    check_close(max_qps_under_slo(CURVE, ttft_slo=0.5, tpot_slo=0.05), 4 + 0.1 * 2 / 0.5, what="TTFT 限制在 0.5s")
    check_close(max_qps_under_slo(CURVE, ttft_slo=10, tpot_slo=0.03), 4 + 0.002 * 2 / 0.007, what="TPOT 限制在 30ms")


def test_edges():
    check(max_qps_under_slo(CURVE, 0.1, 1.0), 0.0, "最小 QPS 就违反 SLO")
    check(max_qps_under_slo(CURVE, 100, 1.0), 8.0, "全部满足时不外推")
    check_close(max_qps_under_slo(CURVE, 0.4, 1.0), 4.0, what="正好等于某个测量点")


def test_replicas():
    check(replicas_needed(100, 6), 21, "100 QPS，每副本 6 QPS，留 20% 余量")
    check(replicas_needed(48, 6, headroom=0.0), 8, "不留余量、正好整除")
    check(replicas_needed(0, 6), 0, "没有流量")
    with raises(ValueError, "per_replica_qps=0"):
        replicas_needed(10, 0)


def test_littles_law():
    check_close(littles_law_concurrency(50, 4.0), 200.0, what="50 QPS × 4 秒")
