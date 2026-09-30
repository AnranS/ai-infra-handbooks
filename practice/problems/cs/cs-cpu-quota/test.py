from checker import check, check_close
from solution import finish_ms, usable_cpus


def test_example():
    check_close(finish_ms(8, 40, 200, 100, cores=64), 115.0, what="8 个线程被节流")
    check_close(finish_ms(8, 40, None, 100, cores=2), 160.0, what="只有 2 个核")
    check_close(usable_cpus(32, 200000, 100000), 2.0, what="配额 2 个 CPU")


def test_no_throttle_when_under_quota():
    check_close(finish_ms(1, 40, 200, 100, cores=64), 40.0, what="单线程用不完配额")
    check_close(finish_ms(4, 40, 200, 100, cores=64), 40.0, what="4 × 40 = 160 < 200")


def test_long_burst():
    check_close(finish_ms(16, 40, 200, 100, cores=64), 302.5, what="要 3 个多周期")


def test_cores_and_quota():
    # 4 个核、配额 2 个 CPU：8 个线程各 50 ms（共 400 ms）
    check_close(finish_ms(8, 50, 200, 100, cores=4), 150.0, what="每周期跑 50 ms 就用光配额")


def test_usable_cpus():
    check_close(usable_cpus(32, -1, 100000), 32.0, what="不限配额")
    check_close(usable_cpus(4, 800000, 100000), 4.0, what="配额比亲和性多：受亲和性限制")
    check_close(usable_cpus(16, 150000, 100000), 1.5, what="小数个 CPU")
