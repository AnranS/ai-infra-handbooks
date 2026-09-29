import math


def prefill_tokens(prompt_lens, n, share):
    return sum(prompt_lens) * (1 if share else n)


def kv_tokens_per_group(prompt_len, n, max_new, share, block_size):
    bs = block_size
    if not share:
        return n * math.ceil((prompt_len + max_new) / bs) * bs
    return (prompt_len // bs) * bs + n * math.ceil((prompt_len % bs + max_new) / bs) * bs


def max_concurrent_groups(prompt_lens, n, max_new, kv_budget, share, block_size):
    used, count = 0, 0
    for p in prompt_lens:
        need = kv_tokens_per_group(p, n, max_new, share, block_size)
        if used + need > kv_budget:
            break
        used += need
        count += 1
    return count
