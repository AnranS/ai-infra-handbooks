ARCH = {6: (1, 1), 7: (2, 2), 8: (4, 2), 9: (8, 8)}


def divide_sm(total_sms, major, groups):
    if major not in ARCH:
        raise ValueError(f"unsupported compute capability {major}")
    min_per_part, multiple = ARCH[major]
    cand = [x for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
            if x >= total_sms - x and total_sms - x >= 16]
    if not cand:
        raise ValueError(f"no valid partition for {total_sms} SMs")
    if len(cand) >= groups:
        cand = cand[::max(1, len(cand) // groups)][:groups]
    return [(x, total_sms - x) for x in reversed(cand)]


def stream_groups(total_sms, major, sm_group_num=8):
    return [(total_sms, 0)] + divide_sm(total_sms, major, sm_group_num - 2) + [(0, total_sms)]


def choose(groups, decode_bs, has_prefill, decode_bs_divisor=36, thresholds=None):
    n = len(groups)
    if decode_bs and has_prefill:
        if thresholds:
            idx = 1
            for i, t in enumerate(thresholds):
                if decode_bs >= t:
                    idx = i + 1
            return idx
        return max(1, min(n - 2, decode_bs * (n - 2) // decode_bs_divisor))
    return n - 1 if decode_bs else 0


def prefill_layers_per_step(extend_tokens, num_layers, done_layers=0, token_budget=65536):
    left = num_layers - done_layers
    if extend_tokens <= 0:
        return left
    return min(left, max(1, token_budget // extend_tokens))
