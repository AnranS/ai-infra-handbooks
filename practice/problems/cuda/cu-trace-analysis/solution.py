from collections import defaultdict


def analyze(events):
    gpu = [e for e in events if e["cat"] in ("kernel", "memcpy")]
    if not gpu:
        return {"span": 0.0, "gpu_busy": 0.0, "utilization": 0.0, "largest_gap": None, "top_kernels": [],
                "memcpy_ratio": 0.0}
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    busy, gap = 0.0, None
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            if gap is None or s - cur_e > gap[1] - gap[0]:
                gap = (cur_e, s)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = max(e for _, e in iv) - iv[0][0]
    per = defaultdict(lambda: [0.0, 0])
    for e in gpu:
        if e["cat"] == "kernel":
            per[e["name"]][0] += e["dur"]
            per[e["name"]][1] += 1
    top = sorted(((n, t, c) for n, (t, c) in per.items()), key=lambda x: (-x[1], x[0]))[:3]
    mem = sum(e["dur"] for e in gpu if e["cat"] == "memcpy")
    total = sum(e["dur"] for e in gpu)
    return {"span": span, "gpu_busy": busy, "utilization": busy / span if span else 0.0, "largest_gap": gap,
            "top_kernels": top, "memcpy_ratio": mem / total if total else 0.0}
