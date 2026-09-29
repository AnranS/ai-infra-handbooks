import numpy as np


def summarize(records, ttft_slo, tpot_slo):
    arr = np.array([r["arrival"] for r in records], dtype=np.float64)
    first = np.array([r["first_token"] for r in records], dtype=np.float64)
    fin = np.array([r["finish"] for r in records], dtype=np.float64)
    out = np.array([r["output_tokens"] for r in records], dtype=np.int64)
    ttft = first - arr
    multi = out >= 2
    tpot = np.zeros(len(records))
    tpot[multi] = (fin[multi] - first[multi]) / (out[multi] - 1)
    span = fin.max() - arr.min()
    ok = (ttft <= ttft_slo) & (~multi | (tpot <= tpot_slo))
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else 0.0  # noqa: E731
    return {
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "tpot_p50": pct(tpot[multi], 50), "tpot_p99": pct(tpot[multi], 99),
        "e2e_p50": pct(fin - arr, 50),
        "throughput": float(out.sum() / span),
        "goodput": float(ok.sum() / span),
        "slo_attainment": float(ok.mean()),
    }
