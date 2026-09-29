from checker import check, check_close
from solution import percentile, request_metrics, summarize


def close_all(got, want, what, **kw):
    """逐个比较（不依赖 numpy）"""
    got, want = list(got), list(want)
    check(len(got), len(want), f"{what}：个数")
    for i, (g, w) in enumerate(zip(got, want)):
        check_close(g, w, what=f"{what}（第 {i + 1} 个）", **kw)


def test_example():
    m = request_metrics(0.0, [0.5, 0.52, 0.54, 0.60])
    close_all((m["ttft"], m["tpot"], m["e2e"]), (0.5, 0.1 / 3, 0.6), what="TTFT、TPOT、E2E")
    close_all(m["itl"], [0.02, 0.02, 0.06], what="ITL")
    check(percentile([5, 1, 4, 2, 3], 50), 3, "中位数")


def test_request_metrics():
    m = request_metrics(10.0, [10.8])
    close_all((m["ttft"], m["tpot"], m["e2e"]), (0.8, 0.0, 0.8), what="只有一个 token")
    check(m["itl"], [], "没有间隔")
    m = request_metrics(1.0, [1.2, 1.25, 1.55, 1.6])
    check_close(m["tpot"], 0.4 / 3, what="TPOT 从第一个 token 算起")


def test_percentile():
    vals = list(range(1, 101))
    check(percentile(vals, 99), 99, "1..100 的 P99")
    check(percentile(vals, 100), 100, "P100 是最大值")
    check(percentile(vals, 1), 1, "P1")
    check(percentile([7], 99), 7, "只有一个样本")
    check(percentile([3, 1, 2], 99), 3, "样本少时 P99 就是最大值")
    check(percentile([3, 1, 2], 0), 1, "P0 取最小值")


def test_summarize():
    smooth = [(0.1 * i, [0.1 * i + 0.3 + 0.02 * k for k in range(50)]) for i in range(20)]
    s = summarize(smooth, ttft_slo=0.5, tpot_slo=0.03)
    close_all((s["ttft_p50"], s["tpot_p99"], s["itl_p99"]), (0.3, 0.02, 0.02), what="平稳的负载")
    close_all((s["slo_ok"], s["goodput"]), (1.0, 20 / (1.9 + 0.3 + 0.98)), what="全部满足 SLO")
    spiky = [(0.0, [0.2] + [0.2 + 0.02 * k for k in range(1, 20)] + [0.58 + 0.3] + [0.88 + 0.02 * k for k in range(1, 30)])]
    s = summarize(spiky * 1, ttft_slo=0.5, tpot_slo=0.05)
    check(s["slo_ok"], 1.0, "一个 300 ms 的卡顿平均下来，TPOT 仍满足 SLO")
    check_close(s["itl_p99"], 0.3, what="但 ITL 的 P99 暴露了卡顿")
    mixed = smooth[:10] + [(2.0 + 0.1 * i, [2.0 + 0.1 * i + 0.9 + 0.02 * k for k in range(50)]) for i in range(10)]
    s = summarize(mixed, ttft_slo=0.5, tpot_slo=0.03)
    check_close(s["slo_ok"], 0.5, what="后一半请求排队太久，TTFT 超标")
    check_close(s["ttft_p99"], 0.9, what="TTFT 的 P99")
    check_close(s["goodput"], 10 / (2.9 + 0.9 + 0.98), what="goodput 只数满足 SLO 的请求")
