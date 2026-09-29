import numpy as np

from checker import check, check_close
from solution import ring_allreduce


def make(n, N, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(N) for _ in range(n)]


def test_example():
    data = make(4, 12)
    res, sent = ring_allreduce(data)
    total = sum(data)
    for r in range(4):
        check_close(res[r], total, rtol=1e-12, atol=1e-12, what=f"卡 {r} 的结果")
    check(sent, [18, 18, 18, 18], "每张卡发送的元素数：2 × 3 步 × 3 个元素")


def test_uneven_chunks():
    data = make(3, 10, seed=1)
    res, sent = ring_allreduce(data)
    for r in range(3):
        check_close(res[r], sum(data), rtol=1e-12, atol=1e-12, what=f"卡 {r}（10 个元素切成 4/3/3）")
    check(sum(sent), 2 * 2 * 10, "总发送量 = 2(n-1)/n × N × n")


def test_single_and_two():
    data = make(1, 5)
    res, sent = ring_allreduce(data)
    check_close(res[0], data[0], what="只有一张卡")
    check(sent, [0], "一张卡不需要通信")
    data = make(2, 6, seed=2)
    res, sent = ring_allreduce(data)
    check_close(res[1], data[0] + data[1], what="两张卡")
    check(sent, [6, 6], "两张卡各发送 N 个元素")


def test_simultaneous_semantics():
    """8 张卡，每张卡的值都不同"""
    data = [np.full(8, float(r + 1)) for r in range(8)]
    res, _ = ring_allreduce(data)
    for r in range(8):
        check_close(res[r], np.full(8, 36.0), what=f"卡 {r}")


def test_inputs_unchanged():
    data = make(4, 9, seed=3)
    before = [x.copy() for x in data]
    ring_allreduce(data)
    for a, b in zip(data, before):
        check_close(a, b, what="输入数组不应该被修改")
