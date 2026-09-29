HBM, BF16, FP8, NIC = 3.35e12, 989e12, 1979e12, 50e9
ATTN_W, EXPERT_W, EXPERT_FLOP, LOCAL_EXPERTS, LAYERS = 187e6, 44e6, 88e6, 3, 61
TOK_BYTES = (7168 + 224) + 7168 * 2


def expected_advance(accept):
    adv, p = 1.0, 1.0
    for a in accept:
        p *= a
        adv += p
    return adv


def layer_time(seqs, tokens, ctx):
    attn = max((ATTN_W + seqs * ctx * 1152) / HBM, tokens * ctx * 278528 / BF16)
    moe = max(LOCAL_EXPERTS * EXPERT_W / HBM, tokens * 9 * EXPERT_FLOP / FP8)
    return max(attn + moe, tokens * 8 * TOK_BYTES / NIC)


def mtp_throughput(seqs, ctx, k, accept):
    step = LAYERS * layer_time(seqs, seqs * (k + 1), ctx) + k * layer_time(seqs, seqs, ctx)
    return seqs * expected_advance(accept[:k]) / step


def best_k(seqs, ctx, accept, k_max=3):
    return max(range(k_max + 1), key=lambda k: (mtp_throughput(seqs, ctx, k, accept), -k))
