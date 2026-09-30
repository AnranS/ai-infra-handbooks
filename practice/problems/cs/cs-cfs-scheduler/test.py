from checker import check
from solution import WEIGHT, simulate


def test_example_weights():
    check(simulate([("a", 0, 0), ("b", 5, 0)], 1000), {"a": 753, "b": 247}, "nice 0 和 nice 5 按 1024 : 335 分配")


def test_example_late_arrival():
    check(simulate([("a", 0, 0), ("b", 0, 500)], 1000), {"a": 750, "b": 250}, "晚到的 b 只和 a 平分后面的时间")


def test_equal_share():
    got = simulate([("x", 0, 0), ("y", 0, 0), ("z", 0, 0)], 300)
    check(got, {"x": 100, "y": 100, "z": 100}, "三个同样 nice 的任务平分")


def test_proportional():
    got = simulate([("hi", -5, 0), ("mid", 0, 0), ("low", 10, 0)], 1000)
    total = WEIGHT[-5] + WEIGHT[0] + WEIGHT[10]
    for name, nice in [("hi", -5), ("mid", 0), ("low", 10)]:
        want = 1000 * WEIGHT[nice] / total
        check(abs(got[name] - want) <= 2, True, f"{name} 应得约 {want:.1f} ms，实际 {got[name]}")


def test_idle_and_late():
    got = simulate([("late", 0, 300)], 1000)
    check(got, {"late": 700}, "前 300 ms 没有任务，CPU 空闲")


def test_late_heavy_does_not_starve():
    got = simulate([("a", 0, 0), ("b", 0, 0), ("c", -10, 600)], 1000)
    check(got["a"] >= 300 and got["b"] >= 300, True, f"c 晚到之后 a、b 不应被饿住：{got}")
    check(got["c"] > 300, True, f"c 权重大，到达后应占大部分 CPU：{got}")
