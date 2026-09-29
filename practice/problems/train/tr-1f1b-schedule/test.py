from checker import check, check_close, raises
from solution import peak_activations, schedule_1f1b, simulate


def gpipe(p, m):
    return [[("F", i) for i in range(m)] + [("B", i) for i in range(m)] for _ in range(p)]


def test_example():
    check(schedule_1f1b(3, 4)[0], [("F", 0), ("F", 1), ("F", 2), ("B", 0), ("F", 3), ("B", 1), ("B", 2), ("B", 3)], "p=3、m=4 的 stage 0")
    check_close(simulate(4, 8, schedule_1f1b(4, 8)), 33.0, what="p=4、m=8")


def test_schedule():
    plans = schedule_1f1b(3, 4)
    check(plans[2], [("F", 0), ("B", 0), ("F", 1), ("B", 1), ("F", 2), ("B", 2), ("F", 3), ("B", 3)], "最后一个 stage 没有预热")
    check(plans[1], [("F", 0), ("F", 1), ("B", 0), ("F", 2), ("B", 1), ("F", 3), ("B", 2), ("B", 3)], "stage 1")
    check(schedule_1f1b(4, 2)[0], [("F", 0), ("F", 1), ("B", 0), ("B", 1)], "micro-batch 比 stage 少")
    for p, m in [(2, 3), (5, 5), (8, 16)]:
        for s, ops in enumerate(schedule_1f1b(p, m)):
            check(sorted(ops), sorted([("F", i) for i in range(m)] + [("B", i) for i in range(m)]), f"p={p}、m={m} 的 stage {s} 每个操作恰好一次")


def test_makespan():
    for p, m in [(4, 8), (4, 4), (4, 2), (8, 32), (1, 5), (3, 7)]:
        check_close(simulate(p, m, schedule_1f1b(p, m)), (m + p - 1) * 3.0, what=f"1F1B，p={p}、m={m}")
        check_close(simulate(p, m, gpipe(p, m)), (m + p - 1) * 3.0, what=f"GPipe，p={p}、m={m}")
    check_close(simulate(4, 8, schedule_1f1b(4, 8), tf=1.0, tb=1.0), 22.0, what="前向反向一样长")


def test_memory():
    check(peak_activations(schedule_1f1b(4, 8)), [4, 3, 2, 1], "1F1B：stage s 最多 p - s 个")
    check(peak_activations(schedule_1f1b(4, 2)), [2, 2, 2, 1], "m < p 时不超过 m")
    check(peak_activations(gpipe(4, 8)), [8, 8, 8, 8], "GPipe 要保存全部 micro-batch")


def test_deadlock():
    bad = [[("F", 0), ("B", 0)], [("B", 0), ("F", 0)]]     # stage 1 先等自己的反向，永远等不到
    with raises(RuntimeError):
        simulate(2, 1, bad)
