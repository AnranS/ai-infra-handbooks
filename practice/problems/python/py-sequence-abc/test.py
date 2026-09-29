from collections.abc import Sequence

from checker import check, raises
from solution import ArithSeq


def test_example():
    s = ArithSeq(0, 1, 0.25)
    check(list(s), [0, 0.25, 0.5, 0.75], "list(s)")
    check(s[-1], 0.75, "s[-1]")
    check(s[1:3], ArithSeq(0.25, 0.75, 0.25), "s[1:3]")
    check((0.5 in s, s.index(0.5), s.count(0.5)), (True, 2, 1), "in / index / count")
    check(list(reversed(ArithSeq(5, 0, -2))), [1, 3, 5], "reversed")


def test_matches_range():
    for args in [(0, 10), (0, 10, 3), (10, 0, -3), (5, 5), (5, 2), (-4, 7, 2), (3, -8, -4)]:
        r, s = range(*args), ArithSeq(*args)
        check(len(s), len(r), f"len(ArithSeq{args})")
        check(list(s), list(r), f"list(ArithSeq{args})")
        for sl in [slice(None, None, -1), slice(1, None, 2), slice(-3, None), slice(None, 2), slice(5, 1, -2)]:
            check(list(s[sl]), list(r[sl]), f"ArithSeq{args}[{sl.start}:{sl.stop}:{sl.step}]")


def test_index_errors():
    s = ArithSeq(0, 3)
    with raises(IndexError, "s[3]"):
        s[3]
    with raises(IndexError, "s[-4]"):
        s[-4]
    with raises(TypeError, 's["1"]'):
        s["1"]
    with raises(ValueError, "step=0"):
        ArithSeq(0, 1, 0)


def test_is_sequence_and_no_accumulated_error():
    assert isinstance(ArithSeq(0, 1), Sequence)
    s = ArithSeq(0, 1, 0.1)
    check(len(s), 10, "len(ArithSeq(0, 1, 0.1))")
    check(s[7], 0 + 7 * 0.1, "s[7] 用乘法计算")


def test_only_two_methods_implemented():
    """in / 迭代 / index / count 应该继承自 Sequence"""
    for name in ["__contains__", "__iter__", "__reversed__", "index", "count"]:
        assert name not in ArithSeq.__dict__, f"不需要自己实现 {name}：继承 Sequence 就有了"
