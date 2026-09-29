import bisect


def pad_to(bs, sizes):
    pass


def expected_waste(dist, sizes):
    pass


def best_sizes(dist, max_bs, k):
    # 贪心：取出现次数最多的 k-1 个批大小再加上 max_bs（不一定最优）
    top = sorted(dist, key=lambda b: -dist[b])[: k - 1]
    return sorted(set(top) | {max_bs})
