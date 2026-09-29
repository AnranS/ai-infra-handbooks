import random

from checker import check, check_close
from solution import analyze


def test_example():
    steps = [{"schedule": 0.8, "prepare": 1.2, "forward": 6.0, "sample": 0.3, "process": 1.5},
             {"schedule": 1.0, "prepare": 1.0, "forward": 4.0, "sample": 0.5, "process": 2.5}]
    got = analyze(steps)
    check_close(got["step_ms"], (9.8 + 9.0) / 2, what="step_ms")
    check_close(got["gpu_idle"], 1 - 10 / 18.8, what="gpu_idle")
    check(got["bottleneck"], "process", "bottleneck")
    check_close(got["overlap_step_ms"], (6.0 + 5.0) / 2, what="overlap_step_ms")
    check_close(got["overlap_speedup"], 9.4 / 5.5, what="overlap_speedup")
    check_close(sum(got["breakdown"].values()), 1.0, what="breakdown 之和")


def test_cpu_bound():
    steps = [{"schedule": 3.0, "forward": 1.0, "process": 4.0}] * 5
    got = analyze(steps)
    check_close(got["overlap_step_ms"], 7.0, what="CPU 比 GPU 慢时，重叠后受限于 CPU")
    check_close(got["overlap_speedup"], 8 / 7, what="加速比")
    check_close(got["breakdown"]["forward"], 1 / 8, what="forward 的占比")


def test_random():
    rng = random.Random(0)
    steps = [{"a": rng.uniform(0, 2), "forward": rng.uniform(1, 5), "b": rng.uniform(0, 2)} for _ in range(50)]
    got = analyze(steps)
    ser = sum(s["a"] + s["forward"] + s["b"] for s in steps) / 50
    ov = sum(max(s["a"] + s["b"], s["forward"]) for s in steps) / 50
    check_close((got["step_ms"], got["overlap_step_ms"]), (ser, ov), rtol=1e-12, what="随机时间线")
    check(got["bottleneck"], max(["a", "b"], key=lambda k: sum(s[k] for s in steps)), "bottleneck")
