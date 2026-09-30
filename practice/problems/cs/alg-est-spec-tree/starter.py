def accept_expect(p, k):
    return sum(p ** i for i in range(1, k + 1))    # 漏了目标模型自己补的那一个


def speedup(p, k, draft_cost, verify_cost):
    return accept_expect(p, k) / (k * draft_cost + verify_cost)   # 没有乘上不用投机时的成本


def best_k(p, draft_cost, verify_cost, max_k=16):
    best, best_val = 0, 0
    for k in range(1, max_k + 1):
        val = speedup(p, k, draft_cost, verify_cost)
        if val >= best_val:                        # 用 >=：并列时取到了更大的 k
            best, best_val = k, val
    return best
