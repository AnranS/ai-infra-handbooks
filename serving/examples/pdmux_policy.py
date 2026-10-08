"""pdmux_policy.py —— SGLang PD 复用（srt/multiplex/pdmux_context.py、multiplexing_mixin.py）的切分与选择规则。"""

ARCH = {6: (1, 1), 7: (2, 2), 8: (4, 2), 9: (8, 8)}     # 计算能力大版本 → (每份最少 SM 数, SM 数的粒度)


def divide_sm(total_sms, major, groups):
    """候选的 (prefill SM, decode SM) 切分：prefill 不少于一半，decode 至少 16 个，按粒度取值，均匀挑 groups 个"""
    min_per_part, multiple = ARCH[major]
    cand = [x for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
            if x >= total_sms - x and total_sms - x >= 16]
    if len(cand) >= groups:
        cand = cand[::max(1, len(cand) // groups)][:groups]
    return [(x, total_sms - x) for x in reversed(cand)]            # prefill 分得多的排在前面


def stream_groups(total_sms, major, sm_group_num=8):
    """第 0 组：全部 SM 给 prefill（普通流）；中间 sm_group_num-2 组：green context 切分；最后一组：全部给 decode"""
    return [(total_sms, 0)] + divide_sm(total_sms, major, sm_group_num - 2) + [(0, total_sms)]


def choose(groups, decode_bs, has_prefill, decode_bs_divisor=36):
    """调度器按正在 decode 的请求数选一组：decode 越多，分给 decode 的 SM 越多"""
    n = len(groups)
    if decode_bs and has_prefill:
        return max(1, min(n - 2, decode_bs * (n - 2) // decode_bs_divisor))
    return n - 1 if decode_bs else 0


def prefill_layers_per_step(extend_tokens, num_layers, token_budget=65536):
    """prefill 按层切开：每轮只算 token_budget // extend_tokens 层（至少 1 层），和 decode 交替推进"""
    return min(num_layers, max(1, token_budget // extend_tokens))
