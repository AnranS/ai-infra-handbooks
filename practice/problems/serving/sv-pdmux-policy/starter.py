def divide_sm(total_sms, major, groups):
    step = 8 if major >= 9 else 2
    cand = [x for x in range(step, total_sms, step) if total_sms - x >= 16]    # 忘了"prefill 不少于一半"，也没按 groups 挑
    return [(x, total_sms - x) for x in cand]


def stream_groups(total_sms, major, sm_group_num=8):
    pass


def choose(groups, decode_bs, has_prefill, decode_bs_divisor=36, thresholds=None):
    pass


def prefill_layers_per_step(extend_tokens, num_layers, done_layers=0, token_budget=65536):
    pass
