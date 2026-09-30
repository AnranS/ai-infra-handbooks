def run_cycles(n_ops, accumulators, latency, throughput, width):
    per_cycle = min(throughput, width)
    ready = [0] * accumulators
    cycle = issued = 0
    in_cycle = 0
    while issued < n_ops:
        acc = issued % accumulators
        if in_cycle < per_cycle and ready[acc] <= cycle:
            ready[acc] = cycle + latency
            issued += 1
            in_cycle += 1
        else:
            cycle += 1
            in_cycle = 0
    return cycle                                # 返回的是最后一条发射的周期，漏掉了它的延迟


def min_accumulators(latency, throughput):
    return latency                              # 只看延迟，忘了每周期能发射多条
