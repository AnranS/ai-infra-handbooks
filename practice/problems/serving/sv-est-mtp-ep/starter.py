HBM, BF16, FP8, NIC = 3.35e12, 989e12, 1979e12, 50e9                  # 显存带宽、BF16 / FP8 算力、每卡网卡带宽
ATTN_W, EXPERT_W, EXPERT_FLOP, LOCAL_EXPERTS, LAYERS = 187e6, 44e6, 88e6, 3, 61
TOK_BYTES = (7168 + 224) + 7168 * 2                                     # 每个 token 每个专家：dispatch（FP8）+ combine（BF16）


def expected_advance(accept):
    return 1 + sum(accept)                              # 第 i 个草稿要前面的都被接受才有用：应该是连乘


def layer_time(seqs, tokens, ctx):
    pass


def mtp_throughput(seqs, ctx, k, accept):
    pass


def best_k(seqs, ctx, accept, k_max=3):
    pass
