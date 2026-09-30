from checker import check
from solution import finish_time, lpt_finish, min_machines


def test_example():
    tasks = [5, 3, 8, 2, 7, 1, 6]
    check(finish_time(tasks, 3), 12, "按到达顺序")
    check(lpt_finish(tasks, 3), 11, "最长优先更好")
    check(min_machines([5, 3, 8], 8), 2, "两台机器")


def test_edges():
    check(finish_time([], 3), 0, "没有任务")
    check(finish_time([5], 1), 5, "一台机器一个任务")
    check(finish_time([1, 2, 3], 10), 3, "机器比任务多")
    check(min_machines([], 5), 0, "没有任务不需要机器")


def test_impossible():
    check(min_machines([10, 1], 5), -1, "有任务比 deadline 还长")
    check(min_machines([5], 5), 1, "正好等于 deadline")


def test_single_machine():
    check(finish_time([1, 2, 3], 1), 6, "一台机器串行")
    check(lpt_finish([1, 2, 3], 1), 6, "顺序不影响总时间")


def test_lpt_not_worse():
    import random
    rng = random.Random(5)
    for _ in range(100):
        tasks = [rng.randrange(1, 20) for _ in range(rng.randrange(2, 12))]
        m = rng.randrange(1, 5)
        lower = max(max(tasks), (sum(tasks) + m - 1) // m)
        got = lpt_finish(tasks, m)
        check(got >= lower, True, "不可能快过理论下界")
        check(got <= lower * 4 / 3 + 1, True, f"LPT 的近似比不超过 4/3：{tasks}, {m}")


def test_min_machines_search():
    tasks = [7, 6, 5, 4, 3, 2, 1]
    check(min_machines(tasks, 10), 3, "28 个单位、每台 10：需要三台")
    check(min_machines(tasks, 28), 1, "一台就够")
    check(min_machines(tasks, 7), 4, "deadline 很紧")


def test_large():
    tasks = [i % 50 + 1 for i in range(20000)]
    got = lpt_finish(tasks, 64)
    lower = max(max(tasks), (sum(tasks) + 63) // 64)
    check(got >= lower, True, "不快过下界")
    check(got <= lower * 1.1, True, "两万个任务时接近最优")
