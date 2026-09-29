def analyze(steps):
    names = list(steps[0])
    totals = {k: sum(s[k] for s in steps) for k in names}
    grand = sum(totals.values())
    n = len(steps)
    serial = [sum(s.values()) for s in steps]
    overlap = [max(sum(v for k, v in s.items() if k != "forward"), s["forward"]) for s in steps]
    step_ms = sum(serial) / n
    overlap_ms = sum(overlap) / n
    cpu_names = [k for k in names if k != "forward"]
    return {
        "step_ms": step_ms,
        "gpu_idle": 1 - totals["forward"] / grand,
        "breakdown": {k: totals[k] / grand for k in names},
        "bottleneck": max(cpu_names, key=lambda k: totals[k]),
        "overlap_step_ms": overlap_ms,
        "overlap_speedup": step_ms / overlap_ms,
    }
