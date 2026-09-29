import random

from checker import check
from solution import ring_allreduce


def test_example():
    check(ring_allreduce([[1, 2], [3, 4]]), ([[4, 6], [4, 6]], [2, 2]), "两个 rank")


def test_random():
    rng = random.Random(0)
    for n in range(2, 9):
        L = n * rng.randint(1, 4)
        data = [[rng.randint(-50, 50) for _ in range(L)] for _ in range(n)]
        copy = [list(x) for x in data]
        result, sent = ring_allreduce(data)
        total = [sum(col) for col in zip(*copy)]
        check(result, [total] * n, f"n={n}、L={L} 时每个 rank 都应得到总和")
        check(sent, [2 * (n - 1) * L // n] * n, f"n={n}、L={L} 时每个 rank 发送 2(n-1)/n·L 个元素")
        check(data, copy, "不要修改输入")


def test_single_rank():
    check(ring_allreduce([[5, 6, 7]]), ([[5, 6, 7]], [0]), "只有一个 rank 时不需要通信")


def test_simultaneous():
    # 如果不是"先取出再统一应用"，同一步里后处理的 rank 会把已经加过的块再发出去
    result, _ = ring_allreduce([[1, 0, 0], [0, 10, 0], [0, 0, 100]])
    check(result, [[1, 10, 100]] * 3, "三个 rank、每块一个元素")
