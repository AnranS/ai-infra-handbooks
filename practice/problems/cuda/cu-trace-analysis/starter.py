from collections import defaultdict


def analyze(events):
    gpu = [e for e in events if e["cat"] in ("kernel", "memcpy")]
    busy = sum(e["dur"] for e in gpu)          # 没有合并重叠的区间
    start = min(e["ts"] for e in gpu)
    end = max(e["ts"] + e["dur"] for e in gpu)
    return {"span": end - start, "gpu_busy": busy, "utilization": busy / (end - start), "largest_gap": None,
            "top_kernels": [], "memcpy_ratio": 0.0}
