import math

from checker import check, check_close, raises
from solution import allreduce_time, best_algo, busbw, crossover


def test_example():
    check(best_algo(16 * 1024, 8, 1.5e-6, 450e9), "one-shot", "16 KiB")
    check(best_algo(8 * 2**20, 8, 1.5e-6, 450e9), "two-shot", "8 MiB")
    check_close(crossover(8, 1.5e-6, 450e9), 2 * 1.5e-6 * 450e9 / 5.25, what="8 卡的分界点")


def test_time():
    check_close(allreduce_time("ring", 2**20, 8, 1.5e-6, 450e9), 14 * 1.5e-6 + 1.75 * 2**20 / 450e9, what="ring")
    check_close(allreduce_time("one-shot", 2**20, 8, 1.5e-6, 450e9), 3e-6 + 7 * 2**20 / 450e9, what="one-shot")
    check_close(allreduce_time("two-shot", 2**20, 8, 1.5e-6, 450e9), 6e-6 + 1.75 * 2**20 / 450e9, what="two-shot")
    with raises(ValueError):
        allreduce_time("tree", 1, 8, 1e-6, 1e9)


def test_crossover():
    S = crossover(8, 1.5e-6, 450e9)
    check_close(allreduce_time("one-shot", S, 8, 1.5e-6, 450e9), allreduce_time("two-shot", S, 8, 1.5e-6, 450e9), what="分界点上两者相等")
    check(best_algo(S * 0.9, 8, 1.5e-6, 450e9), "one-shot", "分界点以下")
    check(best_algo(S * 1.1, 8, 1.5e-6, 450e9), "two-shot", "分界点以上")
    check(crossover(2, 1.5e-6, 450e9), math.inf, "两张卡时 one-shot 永远不慢")
    assert crossover(4, 1.5e-6, 450e9) > crossover(8, 1.5e-6, 450e9), "卡越少，one-shot 适用的范围越大"


def test_busbw():
    check_close(busbw(2**30, 4.2e-3, 8), 2**30 / 4.2e-3 * 1.75, what="8 卡 all-reduce 的 busbw")
    check_close(busbw(100, 1.0, 2), 100.0, what="两张卡时系数为 1")
