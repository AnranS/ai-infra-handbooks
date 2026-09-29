import math


def prefill_tokens(prompt_lens, n, share):
    return sum(prompt_lens) * n            # 没有考虑共享


def kv_tokens_per_group(prompt_len, n, max_new, share, block_size):
    pass


def max_concurrent_groups(prompt_lens, n, max_new, kv_budget, share, block_size):
    pass
