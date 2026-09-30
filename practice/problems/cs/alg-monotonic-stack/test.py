from checker import check
from solution import daily_temperatures, trap


def test_example():
    check(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0], "每日温度")
    check(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6, "接雨水")


def test_temp_edges():
    check(daily_temperatures([]), [], "空数组")
    check(daily_temperatures([50]), [0], "单个")
    check(daily_temperatures([50, 50, 50]), [0, 0, 0], "全相同：没有更高的")
    check(daily_temperatures([50, 49, 48]), [0, 0, 0], "递减")
    check(daily_temperatures([48, 49, 50]), [1, 1, 0], "递增")


def test_temp_far_answer():
    check(daily_temperatures([70, 60, 60, 60, 71]), [4, 3, 2, 1, 0], "答案在很远的地方")


def test_trap_edges():
    check(trap([]), 0, "空")
    check(trap([1]), 0, "单根柱子")
    check(trap([3, 2, 1]), 0, "递减接不住")
    check(trap([1, 2, 3]), 0, "递增接不住")
    check(trap([3, 0, 3]), 3, "最简单的坑")


def test_trap_flat_bottom():
    check(trap([3, 0, 0, 3]), 6, "平底的坑")
    check(trap([4, 2, 0, 3, 2, 5]), 9, "多级台阶")


def test_large():
    n = 30000
    t = list(range(n, 0, -1)) + [n + 1]        # 单调递减后接一个最大值
    got = daily_temperatures(t)
    check(got[0], n, "第一个要等到最后")
    check(got[-1], 0, "最后一个没有更高的")
    check(trap([5, 0] * 15000 + [5]), 5 * 15000, "锯齿形")
