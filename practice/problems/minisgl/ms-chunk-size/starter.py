import math


def simulate(L, C, n_decode, a, b):
    ttft = a + b * L                        # 当成一次性 prefill 了
    return ttft, ttft + a + b * n_decode


def best_chunk(L, n_decode, a, b, tpot_slo, candidates):
    pass
