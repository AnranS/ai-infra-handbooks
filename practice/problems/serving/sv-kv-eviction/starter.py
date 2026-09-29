def streaming_keep(n, sink, window):
    return list(range(max(0, n - window), n))       # 忘了保留 sink


def h2o_keep(acc, budget, recent):
    pass


def simulate_h2o(attn_steps, budget, recent):
    pass
