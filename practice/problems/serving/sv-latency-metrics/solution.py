import math


def request_metrics(arrival, token_times):
    first, last, n = token_times[0], token_times[-1], len(token_times)
    return {"ttft": first - arrival, "tpot": (last - first) / (n - 1) if n > 1 else 0.0,
            "itl": [b - a for a, b in zip(token_times, token_times[1:])], "e2e": last - arrival}


def percentile(values, p):
    values = sorted(values)
    k = max(1, math.ceil(p / 100 * len(values)))
    return values[min(len(values), k) - 1]


def summarize(requests, ttft_slo, tpot_slo):
    ms = [request_metrics(a, t) for a, t in requests]
    ttft = [m["ttft"] for m in ms]
    tpot = [m["tpot"] for m in ms]
    itl = [x for m in ms for x in m["itl"]]
    good = sum(m["ttft"] <= ttft_slo and m["tpot"] <= tpot_slo for m in ms)
    duration = max(t[-1] for _, t in requests) - min(a for a, _ in requests)
    return {"ttft_p50": percentile(ttft, 50), "ttft_p99": percentile(ttft, 99),
            "tpot_p50": percentile(tpot, 50), "tpot_p99": percentile(tpot, 99),
            "itl_p99": percentile(itl, 99) if itl else 0.0,
            "slo_ok": good / len(ms), "goodput": good / duration}
