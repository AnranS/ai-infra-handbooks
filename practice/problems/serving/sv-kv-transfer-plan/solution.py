def transfer_plan(num_tokens, src_table, src_bs, dst_table, dst_bs):
    plan = []
    for p in range(num_tokens):
        s = src_table[p // src_bs] * src_bs + p % src_bs
        d = dst_table[p // dst_bs] * dst_bs + p % dst_bs
        if plan and plan[-1][0] + plan[-1][2] == s and plan[-1][1] + plan[-1][2] == d:
            plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + 1)
        else:
            plan.append((s, d, 1))
    return plan


def apply_plan(src_pool, dst_pool, plan):
    for s, d, n in plan:
        dst_pool[d:d + n] = src_pool[s:s + n]
