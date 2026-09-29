from checker import check, check_close, raises
from solution import fastest_plan, max_safe_unavailable, rollout


def close_all(got, want, what, **kw):
    """逐个比较（不依赖 numpy）"""
    got, want = list(got), list(want)
    check(len(got), len(want), f"{what}：个数")
    for i, (g, w) in enumerate(zip(got, want)):
        check_close(g, w, what=f"{what}（第 {i + 1} 个）", **kw)


def test_example():
    check_close(rollout(16, 0.8, 180, 120, 1, 0)["minutes"], 80.0, what="逐个替换")
    check(rollout(16, 0.8, 180, 120, 4, 0)["overloaded"], True, "一次下线 4 个")


def test_strategies():
    r = rollout(16, 0.8, 180, 120, 1, 0)
    check((r["rounds"], r["overloaded"]), (16, False), "逐个替换：16 轮、不过载")
    check_close(r["min_capacity"], 15 / 16, what="最低容量 94%")
    r = rollout(16, 0.8, 180, 120, 0, 2)
    check((r["rounds"], r["overloaded"]), (8, False), "先加后减：8 轮")
    close_all((r["minutes"], r["min_capacity"], r["extra_gpu"]), (40.0, 1.0, 0.125), what="先加后减")
    r = rollout(16, 0.8, 180, 120, 0, 16)
    close_all((r["rounds"], r["minutes"], r["extra_gpu"]), (1, 5.0, 1.0), what="蓝绿")
    r = rollout(10, 0.5, 60, 60, 3, 0)
    check((r["rounds"], r["overloaded"]), (4, False), "10 个实例、每轮 3 个：向上取整成 4 轮")
    check_close(r["minutes"], 8.0, what="4 轮 × 2 分钟")
    with raises(ValueError, "每轮一个都不换"):
        rollout(16, 0.8, 180, 120, 0, 0)


def test_safe_and_fastest():
    check(max_safe_unavailable(16, 0.8), 3, "13/16 = 81% ≥ 80%")
    check(max_safe_unavailable(16, 0.95), 0, "流量 95%：一个都不能下线")
    check(max_safe_unavailable(10, 0.5), 5, "正好 50% 也算不过载")
    check(max_safe_unavailable(4, 0.0), 4, "没有流量时可以全下线")
    check(fastest_plan(16, 0.8, 180, 120, 0), (3, 0), "没有额外的卡：每轮下线 3 个")
    check(fastest_plan(16, 0.8, 180, 120, 2), (3, 1), "2 个实例的额外卡：3 + 1 已经是 4 轮，再多拉起一个还是 4 轮")
    check(fastest_plan(16, 0.95, 180, 120, 0), None, "流量太高又没有额外的卡：没有可行方案")
    check(fastest_plan(16, 0.95, 180, 120, 4), (0, 4), "只能先加后减")
    check(fastest_plan(8, 0.5, 60, 60, 4), (4, 4), "下线 4 个 + 额外 4 个：1 轮做完")
    check(fastest_plan(8, 0.5, 60, 60, 3), (4, 0), "额外的卡不够一轮做完：多用卡并不能更快，选额外卡少的")
