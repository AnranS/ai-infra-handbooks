from math import ceil


def first_byte_ms(rtt_ms, server_ms, reuse=False, tls=True):
    return rtt_ms + server_ms                          # 忘了握手和 TLS


def stream_total_ms(rtt_ms, server_ms, tokens, per_token_ms, reuse=False, tls=True):
    return first_byte_ms(rtt_ms, server_ms, reuse, tls) + tokens * per_token_ms   # 第一个 token 算重了


def throughput_mbps(window_kb, rtt_ms):
    return window_kb / rtt_ms                          # 单位换算没做


def window_for_mbps(target_mbps, rtt_ms):
    return ceil(target_mbps * rtt_ms)


def slow_start_rtts(response_kb, init_window_kb=14):
    return ceil(response_kb / init_window_kb)          # 窗口不会翻倍？
