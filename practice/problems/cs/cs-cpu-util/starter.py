def _parse(text):
    cpus = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("cpu") and parts[0] != "cpu":
            cpus[int(parts[0][3:])] = [int(x) for x in parts[1:9]]
    return cpus


def cpu_usage(before, after):
    a, b = _parse(before), _parse(after)
    per = {}
    for cpu in sorted(b):
        d = [y - x for x, y in zip(a[cpu], b[cpu])]
        total = sum(d)
        per[cpu] = round(100 * (total - d[3]) / total, 1) if total else 0.0    # 只减了 idle
    busiest = max(sorted(per), key=lambda c: per[c])
    others = [v for c, v in per.items() if c != busiest]
    return {"per_cpu": per, "busiest": busiest,
            "single_thread_bottleneck": per[busiest] >= 90 and all(v <= 30 for v in others)}
