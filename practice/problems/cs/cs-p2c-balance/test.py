import random

from checker import check, check_close
from solution import least_conn, p2c, round_robin, simulate


def test_example():
    waits = simulate("rr", [5, 5, 5, 5], n=2, seed=0)
    check(len(waits), 4, "每个请求一个等待时间")
    for got, want in zip(waits, [0.0, 0.0, 3.0, 3.0]):
        check_close(got, want, rtol=1e-9, what="两台轮流，第三、四个请求各等 3 ms")


def test_pickers():
    check(round_robin(7, 3), 1, "轮询")
    check(least_conn([2, 0, 0, 3]), 1, "并列时选下标小的")
    rng = random.Random(0)
    a, b = random.Random(0).randrange(4), random.Random(0).randrange(4)
    picked = p2c(rng, [5, 5, 5, 5])
    check(picked, random.Random(0).randrange(4), "全都相等时返回第一次随机到的那台")


def test_least_conn_beats_round_robin():
    costs = [50.0 if i % 20 == 0 else 1.0 for i in range(400)]       # 每 20 个请求里有一个很慢
    rr = sorted(simulate("rr", costs, n=4))
    lc = sorted(simulate("lc", costs, n=4))
    p99 = int(len(costs) * 0.99)
    check(lc[p99] < rr[p99], True, f"最少连接的 p99 更好（{lc[p99]:.1f} < {rr[p99]:.1f}）")


def test_p2c_close_to_least_conn():
    costs = [50.0 if i % 20 == 0 else 1.0 for i in range(400)]
    lc = sorted(simulate("lc", costs, n=4))
    p2 = sorted(simulate("p2c", costs, n=4, seed=7))
    rr = sorted(simulate("rr", costs, n=4))
    p99 = int(len(costs) * 0.99)
    check(p2[p99] <= rr[p99], True, "二选一不差于轮询")
    check(p2[p99] >= lc[p99] * 0.5, True, "二选一与最少连接同一量级")


def test_no_queue_when_fast():
    waits = simulate("lc", [0.5] * 50, n=4)
    check(max(waits), 0.0, "处理得比到达快时不会排队")


def test_deterministic():
    costs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    check(simulate("p2c", costs, n=3, seed=42) == simulate("p2c", costs, n=3, seed=42), True,
          "同样的种子结果可复现")
