import math


def simulate(L, C, n_decode, a, b):
    blocks = [C] * (L // C) + ([L % C] if L % C else [])
    decode_round = a + b * n_decode
    ttft = sum(a + b * c for c in blocks) + (len(blocks) - 1) * decode_round
    max_gap = a + b * max(blocks) + decode_round
    return ttft, max_gap


def best_chunk(L, n_decode, a, b, tpot_slo, candidates):
    best = None
    for C in candidates:
        ttft, gap = simulate(L, C, n_decode, a, b)
        if gap <= tpot_slo and (best is None or ttft < best[0] or (ttft == best[0] and C > best[1])):
            best = (ttft, C)
    return None if best is None else best[1]
