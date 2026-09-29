import random

from checker import check, check_close
from solution import analyze


def ev(name, cat, ts, dur, stream=0):
    return {"name": name, "cat": cat, "ts": ts, "dur": dur, "stream": stream}


def test_example():
    events = [ev("cudaLaunchKernel", "cpu", 0, 5), ev("gemm", "kernel", 10, 50, 7), ev("softmax", "kernel", 40, 30, 8),
              ev("H2D", "memcpy", 100, 20, 9), ev("gemm", "kernel", 130, 10, 7)]
    got = analyze(events)
    check_close(got["span"], 130.0, what="span")
    check_close(got["gpu_busy"], 60 + 20 + 10, what="gpu_busy（重叠只算一次）")
    check_close(got["utilization"], 90 / 130, what="utilization")
    check(got["largest_gap"], (70, 100), "largest_gap")
    check(got["top_kernels"], [("gemm", 60, 2), ("softmax", 30, 1)], "top_kernels")
    check_close(got["memcpy_ratio"], 20 / 110, what="memcpy_ratio")


def test_empty_and_cpu_only():
    empty = {"span": 0.0, "gpu_busy": 0.0, "utilization": 0.0, "largest_gap": None, "top_kernels": [],
             "memcpy_ratio": 0.0}
    check(analyze([]), empty, "没有事件")
    check(analyze([ev("x", "cpu", 0, 10)]), empty, "只有 CPU 事件")


def test_fully_busy_and_ties():
    events = [ev("b", "kernel", 0, 10), ev("a", "kernel", 10, 10), ev("c", "kernel", 5, 3), ev("d", "kernel", 20, 1)]
    got = analyze(events)
    check(got["largest_gap"], None, "首尾相接，没有空闲")
    check_close(got["utilization"], 1.0, what="利用率")
    check(got["top_kernels"], [("a", 10, 1), ("b", 10, 1), ("c", 3, 1)], "并列时按名字")


def test_random_against_bitmap():
    rng = random.Random(0)
    for trial in range(20):
        events = [ev(f"k{rng.randint(0, 5)}", rng.choice(["kernel", "kernel", "memcpy", "cpu"]), rng.randint(0, 500),
                     rng.randint(1, 60), rng.randint(0, 3)) for _ in range(rng.randint(1, 30))]
        gpu = [e for e in events if e["cat"] != "cpu"]
        if not gpu:
            continue
        t0 = min(e["ts"] for e in gpu)
        t1 = max(e["ts"] + e["dur"] for e in gpu)
        busy = [False] * (t1 - t0)
        for e in gpu:
            for t in range(e["ts"], e["ts"] + e["dur"]):
                busy[t - t0] = True
        got = analyze(events)
        check_close(got["gpu_busy"], sum(busy), what=f"随机用例 {trial} 的 gpu_busy")
        best, run, start = 0, 0, None
        for i, b in enumerate(busy + [True]):
            if not b:
                run += 1
            else:
                if run > best:
                    best, start = run, i - run
                run = 0
        want_gap = None if best == 0 else (t0 + start, t0 + start + best)
        if want_gap is None:
            check(got["largest_gap"], None, f"随机用例 {trial} 的 largest_gap")
        else:
            check(got["largest_gap"][1] - got["largest_gap"][0], best, f"随机用例 {trial} 最长空闲段的长度")
