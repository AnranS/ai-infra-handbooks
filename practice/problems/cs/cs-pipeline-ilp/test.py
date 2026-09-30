from checker import check
from solution import min_accumulators, run_cycles


def test_example():
    check(run_cycles(8, 1, 4, 2, 4), 32, "一个累加器：每次都要等上一次")
    check(run_cycles(8, 8, 4, 2, 4), 7, "8 个累加器：发完只用 4 个周期，最后一条 4 周期后产出")
    check(min_accumulators(4, 2), 8, "延迟 × 吞吐")


def test_scaling():
    cycles = [run_cycles(64, k, 4, 2, 4) for k in (1, 2, 4, 8, 16)]
    check(cycles, [256, 128, 65, 35, 35], "累加器翻倍，周期数减半，到 8 个之后不再变快")


def test_width_limits():
    check(run_cycles(16, 16, 4, 8, 2), 11, "每周期只能发 2 条时，发射带宽成了瓶颈")


def test_two_accumulators():
    check(run_cycles(8, 2, 4, 2, 4), 16, "两个累加器：两条链交替，周期数减半")


def test_latency_one():
    check(run_cycles(10, 1, 1, 1, 4), 10, "延迟 1：一条接一条，没有空隙")


def test_min_acc():
    check(min_accumulators(1, 1), 1, "延迟 1、吞吐 1")
    check(min_accumulators(5, 2), 10, "延迟 5、吞吐 2")
