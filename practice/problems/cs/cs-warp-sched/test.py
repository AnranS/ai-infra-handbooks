from checker import check, check_close
from solution import utilization


def test_example():
    check_close(utilization(1, 1, 16, 500, 100), 0.0329, rtol=0.02, what="一个 warp 藏不住 500 周期的延迟")
    check_close(utilization(16, 1, 16, 500, 100), 0.5225, rtol=0.02, what="16 个 warp 也只能填一半")
    check_close(utilization(8, 4, 64, 500, 100), 0.9515, rtol=0.02, what="提高 ILP 后 8 个 warp 就够了")


def test_more_warps_never_worse():
    us = [utilization(w, 1, 16, 500, 100) for w in (1, 2, 4, 8, 16)]
    check(all(b > a for a, b in zip(us, us[1:])), True, "warp 越多利用率越高")
    check(all(u <= 1.0 for u in us), True, "利用率不超过 100%")


def test_no_latency():
    check_close(utilization(1, 1, 3, 0, 50), 1.0, rtol=1e-9, what="延迟为 0 时一个 warp 就能填满")


def test_single_warp_formula():
    # 一个 warp 一轮 ilp + latency + compute 个周期，其中 ilp + compute 个在发射
    check_close(utilization(1, 2, 8, 90, 200), 10 / 100, rtol=0.03, what="单个 warp 的利用率有闭式解")


def test_enough_warps_saturate():
    check_close(utilization(32, 1, 16, 100, 100), 0.994, rtol=0.01, what="延迟小、warp 多：几乎完全填满")
