def per_token_bytes(ratios, entry_bytes=584, index_bytes=132):
    total = 0.0
    for r in ratios:
        if r == 4:
            total += (entry_bytes + index_bytes) / 4
        elif r > 1:
            total += entry_bytes / r
    return total


def fixed_bytes(ratios, window=128, entry_bytes=584):
    return len(ratios) * window * entry_bytes


def request_bytes(ratios, context_len, window=128, entry_bytes=584, index_bytes=132):
    swa = len(ratios) * min(window, context_len) * entry_bytes
    return per_token_bytes(ratios, entry_bytes, index_bytes) * context_len + swa


def attended(ratio, pos, topk=512, window=128):
    swa = min(window, pos + 1)
    if ratio <= 1:
        return swa
    done = (pos + 1) // ratio
    return swa + (min(done, topk) if ratio == 4 else done)
