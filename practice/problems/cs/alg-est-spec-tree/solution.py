def accept_expect(p, k):
    return sum(p ** i for i in range(k + 1))


def speedup(p, k, draft_cost, verify_cost):
    produced = accept_expect(p, k)
    cost = k * draft_cost + verify_cost
    return produced * verify_cost / cost


def best_k(p, draft_cost, verify_cost, max_k=16):
    best, best_val = 0, speedup(p, 0, draft_cost, verify_cost)
    for k in range(1, max_k + 1):
        val = speedup(p, k, draft_cost, verify_cost)
        if val > best_val + 1e-12:             # 并列时取小的 k
            best, best_val = k, val
    return best
