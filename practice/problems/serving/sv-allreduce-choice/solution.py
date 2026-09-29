import math


def allreduce_time(algo, S, n, alpha, beta):
    if algo == "ring":
        return 2 * (n - 1) * alpha + 2 * (n - 1) / n * S / beta
    if algo == "one-shot":
        return 2 * alpha + (n - 1) * S / beta
    if algo == "two-shot":
        return 4 * alpha + 2 * (n - 1) / n * S / beta
    raise ValueError(f"未知的算法：{algo}")


def best_algo(S, n, alpha, beta):
    return min(("ring", "one-shot", "two-shot"), key=lambda a: allreduce_time(a, S, n, alpha, beta))


def crossover(n, alpha, beta):
    if n <= 2:
        return math.inf
    return 2 * alpha * beta / ((n - 1) * (1 - 2 / n))


def busbw(S, t, n):
    return S / t * 2 * (n - 1) / n
