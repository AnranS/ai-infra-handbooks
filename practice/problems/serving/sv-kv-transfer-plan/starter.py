def transfer_plan(num_tokens, src_table, src_bs, dst_table, dst_bs):
    plan = []
    for p in range(num_tokens):                 # 每个 token 一段：正确但段数太多
        s = src_table[p // src_bs] * src_bs + p % src_bs
        d = dst_table[p // dst_bs] * dst_bs + p % dst_bs
        plan.append((s, d, 1))
    return plan


def apply_plan(src_pool, dst_pool, plan):
    pass
