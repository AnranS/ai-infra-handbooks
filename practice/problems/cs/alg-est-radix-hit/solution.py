def matched_len(cached, request):
    n = 0
    for a, b in zip(cached, request):
        if a != b:
            break
        n += 1
    return n


def prefill_ms(tokens, per_token_ms=0.15):
    return tokens * per_token_ms


def saved_ms(cached, request, per_token_ms=0.15):
    return prefill_ms(matched_len(cached, request), per_token_ms)


def hit_rate_needed(ttft_target_ms, prompt_len, per_token_ms=0.15, overhead_ms=20):
    budget = ttft_target_ms - overhead_ms
    if budget <= 0:
        return 1.0
    max_tokens = budget / per_token_ms         # 预算内最多能算多少 token
    if max_tokens >= prompt_len:
        return 0.0
    return 1 - max_tokens / prompt_len
