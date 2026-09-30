def run_cycles(n_ops, accumulators, latency, throughput, width):
    per_cycle = min(throughput, width)
    ready = [0] * accumulators                 # 每个累加器下一次可用的周期
    cycle = issued = 0
    done = 0
    in_cycle = 0
    while issued < n_ops:
        acc = issued % accumulators
        if in_cycle < per_cycle and ready[acc] <= cycle:
            ready[acc] = cycle + latency
            done = cycle + latency
            issued += 1
            in_cycle += 1
        else:
            cycle += 1
            in_cycle = 0
    return done


def min_accumulators(latency, throughput):
    return latency * throughput
