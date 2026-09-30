from checker import check
from solution import run


def sleeper(s=0.1):
    yield ("sleep", s)


def blocker(s=0.3):
    yield ("cpu", s)


def test_example_overlap():
    check(run({"a": sleeper(), "b": sleeper()}), {"a": 0.1, "b": 0.1}, "两个等待重叠进行")


def test_example_blocking():
    got = run({"bad": blocker(), "a": sleeper(), "b": sleeper()})
    check(got, {"bad": 0.3, "a": 0.4, "b": 0.4}, "阻塞调用让所有人都晚了 0.3 秒")


def test_many_sleepers():
    tasks = {f"t{i}": sleeper(0.1) for i in range(1000)}
    got = run(tasks)
    check(set(got.values()), {0.1}, "一千个等待也只要 0.1 秒")


def test_interleaving():
    order = []

    def worker(name, naps):
        for n in naps:
            order.append((name, "start"))
            yield ("sleep", n)
        order.append((name, "end"))

    got = run({"x": worker("x", [0.2, 0.2]), "y": worker("y", [0.1, 0.1, 0.1])})
    check(got, {"x": 0.4, "y": 0.3}, "各自的完成时刻")
    check(order[:2], [("x", "start"), ("y", "start")], "开始时按字典顺序各跑一段")


def test_cpu_then_sleep():
    def mixed():
        yield ("cpu", 0.05)
        yield ("sleep", 0.1)
        yield ("cpu", 0.05)

    got = run({"m": mixed(), "s": sleeper(0.12)})
    check(got, {"m": 0.2, "s": 0.2}, "s 0.17 就到期了，但事件循环被 m 的第二段计算占着，0.2 才轮到它；时钟不会倒退")
