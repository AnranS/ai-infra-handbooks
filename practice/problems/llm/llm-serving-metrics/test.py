import numpy as np

from checker import check, check_close
from solution import summarize


def rec(a, f, e, n):
    return {"arrival": a, "first_token": f, "finish": e, "output_tokens": n}


def test_example():
    rs = [rec(0.0, 0.35, 2.35, 101), rec(0.5, 0.7, 1.7, 11), rec(1.0, 3.0, 3.0, 1)]
    got = summarize(rs, ttft_slo=1.0, tpot_slo=0.05)
    want = {"ttft_p50": 0.35, "ttft_p99": float(np.percentile([0.35, 0.2, 2.0], 99)),
            "tpot_p50": float(np.percentile([0.02, 0.1], 50)), "tpot_p99": float(np.percentile([0.02, 0.1], 99)),
            "e2e_p50": 2.0, "throughput": 113 / 3.0, "goodput": 1 / 3.0, "slo_attainment": 1 / 3}
    check(sorted(got), sorted(want), "返回的键")
    for k in want:
        check_close(got[k], want[k], rtol=1e-9, atol=1e-12, what=k)


def test_single_token_requests_only():
    rs = [rec(0, 0.1, 0.1, 1), rec(1, 1.5, 1.5, 1)]
    got = summarize(rs, 0.2, 0.01)
    check((got["tpot_p50"], got["tpot_p99"]), (0.0, 0.0), "没有多 token 请求时 TPOT 为 0")
    check_close(got["slo_attainment"], 0.5, what="slo_attainment")
    check_close(got["goodput"], 1 / 1.5, what="goodput")


def test_random_against_manual():
    rng = np.random.default_rng(0)
    rs = []
    for i in range(200):
        a = i * 0.05
        f = a + rng.exponential(0.3)
        n = int(rng.integers(1, 300))
        e = f + (n - 1) * rng.uniform(0.01, 0.05)
        rs.append(rec(a, f, e, n))
    got = summarize(rs, 0.5, 0.03)
    ttft = np.array([r["first_token"] - r["arrival"] for r in rs])
    m = [r for r in rs if r["output_tokens"] >= 2]
    tpot = np.array([(r["finish"] - r["first_token"]) / (r["output_tokens"] - 1) for r in m])
    ok = [r for r in rs if r["first_token"] - r["arrival"] <= 0.5 and
          (r["output_tokens"] < 2 or (r["finish"] - r["first_token"]) / (r["output_tokens"] - 1) <= 0.03)]
    span = max(r["finish"] for r in rs) - min(r["arrival"] for r in rs)
    check_close(got["ttft_p99"], np.percentile(ttft, 99), rtol=1e-9, what="ttft_p99")
    check_close(got["tpot_p50"], np.percentile(tpot, 50), rtol=1e-9, what="tpot_p50")
    check_close(got["goodput"], len(ok) / span, rtol=1e-9, what="goodput")
    check_close(got["throughput"], sum(r["output_tokens"] for r in rs) / span, rtol=1e-9, what="throughput")
