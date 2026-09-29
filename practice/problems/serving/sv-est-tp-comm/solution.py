def allreduce_ms(nbytes, n, link_gbs, latency_us=0.0):
    if n == 1:
        return 0.0
    return 2 * (n - 1) / n * nbytes / (link_gbs * 1e9) * 1e3 + 2 * (n - 1) * latency_us / 1e3


def tp_comm_ms(tokens, hidden, n_layers, tp, link_gbs, dtype_bytes=2, latency_us=0.0):
    per = allreduce_ms(tokens * hidden * dtype_bytes, tp, link_gbs, latency_us)
    return 2 * n_layers * per


def comm_share(comm_ms, compute_ms):
    return comm_ms / (comm_ms + compute_ms)
