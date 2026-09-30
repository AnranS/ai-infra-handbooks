def matched_len(cached, request):
    return len(set(cached) & set(request))     # 用集合交集：顺序完全被忽略了


def prefill_ms(tokens, per_token_ms=0.15):
    return tokens * per_token_ms


def saved_ms(cached, request, per_token_ms=0.15):
    return prefill_ms(matched_len(cached, request), per_token_ms)


def hit_rate_needed(ttft_target_ms, prompt_len, per_token_ms=0.15, overhead_ms=20):
    max_tokens = ttft_target_ms / per_token_ms  # 忘了减去固定开销
    return max(0.0, 1 - max_tokens / prompt_len)
