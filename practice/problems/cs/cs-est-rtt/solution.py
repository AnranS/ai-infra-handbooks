from math import ceil


def first_byte_ms(rtt_ms, server_ms, reuse=False, tls=True):
    handshakes = 0 if reuse else 1 + (1 if tls else 0)
    return handshakes * rtt_ms + rtt_ms + server_ms


def stream_total_ms(rtt_ms, server_ms, tokens, per_token_ms, reuse=False, tls=True):
    return first_byte_ms(rtt_ms, server_ms, reuse, tls) + max(0, tokens - 1) * per_token_ms


def throughput_mbps(window_kb, rtt_ms):
    return window_kb * 1024 / (rtt_ms * 1e-3) / 1e6


def window_for_mbps(target_mbps, rtt_ms):
    return ceil(target_mbps * 1e6 * (rtt_ms * 1e-3) / 1024)


def slow_start_rtts(response_kb, init_window_kb=14):
    sent, window, rtts = 0, init_window_kb, 0
    while sent < response_kb:
        sent += window
        window *= 2
        rtts += 1
    return rtts
