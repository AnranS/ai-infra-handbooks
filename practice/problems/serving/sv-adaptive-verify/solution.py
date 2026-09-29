def survival(conf):
    out, p = [], 1.0
    for c in conf:
        p *= c
        out.append(p)
    return out


def plan(confs, step_time, draft_time):
    batch = len(confs)
    slots = []
    for r, conf in enumerate(confs):
        for i, s in enumerate(survival(conf)):
            slots.append((-s, r, i))
    slots.sort()
    best_b, best_rate, gain = 0, batch / (draft_time + step_time(batch)), 0.0
    for b, (neg_s, _, _) in enumerate(slots, start=1):
        gain += -neg_s
        rate = (batch + gain) / (draft_time + step_time(batch + b))
        if rate > best_rate:
            best_b, best_rate = b, rate
    counts = [0] * batch
    for _, r, _ in slots[:best_b]:
        counts[r] += 1
    return counts
