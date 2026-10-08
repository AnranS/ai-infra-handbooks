"""最小的生成基准：预热、重复、分位数、分阶段计时。clock / sync 可注入，便于用模拟时钟复现各种测错的方式。"""
import statistics
import time
from contextlib import contextmanager


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class StageTimer:
    """with timer.stage("denoise"): ...  按阶段累计时间；每次计时前后都同步"""

    def __init__(self, clock=time.perf_counter, sync=lambda: None):
        self.clock, self.sync, self.stages = clock, sync, {}

    @contextmanager
    def stage(self, name):
        self.sync()
        t0 = self.clock()
        yield
        self.sync()
        self.stages[name] = self.stages.get(name, 0.0) + self.clock() - t0


def run_benchmark(generate, warmup=2, runs=10, clock=time.perf_counter, sync=lambda: None):
    """generate() 做一次完整生成。返回预热轮的时间和正式轮的统计"""
    def timed():
        sync()
        t0 = clock()
        generate()
        sync()
        return clock() - t0

    warm = [timed() for _ in range(warmup)]
    times = [timed() for _ in range(runs)]
    return {"warmup": warm, "times": times, "mean": statistics.fmean(times), "min": min(times), "max": max(times),
            "p50": percentile(times, 0.5), "p90": percentile(times, 0.9), "p99": percentile(times, 0.99)}
