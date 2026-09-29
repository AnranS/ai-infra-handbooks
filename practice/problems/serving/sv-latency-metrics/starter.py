import math


def request_metrics(arrival, token_times):
    n = len(token_times)
    return {"ttft": token_times[0] - arrival,
            "tpot": (token_times[-1] - arrival) / n,        # 把 TTFT 也平均进去了，分母也不对
            "itl": [], "e2e": token_times[-1] - arrival}


def percentile(values, p):
    pass


def summarize(requests, ttft_slo, tpot_slo):
    pass
