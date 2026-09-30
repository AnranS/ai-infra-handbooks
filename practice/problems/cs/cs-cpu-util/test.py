from checker import check
from solution import cpu_usage

B = "cpu  0 0 0 0 0 0 0 0\ncpu0 100 0 50 850 0 0 0 0\ncpu1 10 0 10 980 0 0 0 0\n"
A = "cpu  0 0 0 0 0 0 0 0\ncpu0 190 0 58 852 0 0 0 0\ncpu1 12 0 12 1076 0 0 0 0\n"


def test_example():
    check(cpu_usage(B, A), {"per_cpu": {0: 98.0, 1: 4.0}, "busiest": 0, "single_thread_bottleneck": True}, "单线程热点")


def test_iowait_is_idle():
    b = "cpu0 0 0 0 0 0 0 0 0\n"
    a = "cpu0 10 0 10 30 50 0 0 0\n"
    check(cpu_usage(b, a)["per_cpu"], {0: 20.0}, "iowait 算空闲：100 个滴答里只有 20 个在干活")


def test_steal_counts_in_total():
    b = "cpu0 0 0 0 0 0 0 0 0\n"
    a = "cpu0 40 0 0 40 0 0 0 20\n"
    check(cpu_usage(b, a)["per_cpu"], {0: 60.0}, "steal 计入总时间，不算空闲")


def test_balanced_not_bottleneck():
    b = "cpu0 0 0 0 0 0 0 0 0\ncpu1 0 0 0 0 0 0 0 0\ncpu2 0 0 0 0 0 0 0 0\n"
    a = "cpu0 95 0 0 5 0 0 0 0\ncpu1 93 0 0 7 0 0 0 0\ncpu2 20 0 0 80 0 0 0 0\n"
    got = cpu_usage(b, a)
    check(got["busiest"], 0, "最忙的是 cpu0")
    check(got["single_thread_bottleneck"], False, "有两个核都很忙，不是单线程瓶颈")


def test_tie_and_zero_delta():
    b = "cpu0 0 0 0 0 0 0 0 0\ncpu1 0 0 0 0 0 0 0 0\ncpu2 5 0 0 5 0 0 0 0\n"
    a = "cpu0 50 0 0 50 0 0 0 0\ncpu1 50 0 0 50 0 0 0 0\ncpu2 5 0 0 5 0 0 0 0\n"
    got = cpu_usage(b, a)
    check(got["per_cpu"], {0: 50.0, 1: 50.0, 2: 0.0}, "没有增量的 CPU 记 0")
    check(got["busiest"], 0, "一样高时取编号小的")
