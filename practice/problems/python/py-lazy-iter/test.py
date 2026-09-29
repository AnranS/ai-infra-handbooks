import itertools

from checker import check, raises
from solution import chunked, merge_sorted, sliding_window


def test_example():
    check(list(chunked(range(7), 3)), [(0, 1, 2), (3, 4, 5), (6,)], "chunked(range(7), 3)")
    check(list(sliding_window("abcd", 2)), [("a", "b"), ("b", "c"), ("c", "d")], 'sliding_window("abcd", 2)')
    check(list(merge_sorted([1, 4, 9], [2, 3], [5])), [1, 2, 3, 4, 5, 9], "merge_sorted")


def test_edge_cases():
    check(list(chunked([], 3)), [], "chunked([], 3)")
    check(list(chunked([1, 2], 5)), [(1, 2)], "chunked([1, 2], 5)")
    check(list(sliding_window([1, 2], 3)), [], "输入不足 n 个")
    check(list(sliding_window([1, 2, 3], 3)), [(1, 2, 3)], "正好 n 个")
    check(list(merge_sorted()), [], "merge_sorted()")
    check(list(merge_sorted([], [1], [])), [1], "有空输入")


def test_lazy_on_infinite_input():
    """输入是无限的 itertools.count()"""
    check(list(itertools.islice(chunked(itertools.count(), 2), 3)), [(0, 1), (2, 3), (4, 5)], "无限输入上的 chunked")
    check(next(sliding_window(itertools.count(), 3)), (0, 1, 2), "无限输入上的 sliding_window")
    evens, odds = itertools.count(0, 2), itertools.count(1, 2)
    check(list(itertools.islice(merge_sorted(evens, odds), 6)), [0, 1, 2, 3, 4, 5], "两个无限序列归并")


def test_pulls_only_what_is_needed():
    pulled = []

    def source():
        for i in range(100):
            pulled.append(i)
            yield i

    g = chunked(source(), 4)
    next(g)
    assert len(pulled) <= 5, f"只取第一组，却从输入里拿了 {len(pulled)} 个元素"


def test_validation_at_call_time():
    with raises(ValueError, "chunked(x, 0)"):
        chunked([1, 2, 3], 0)
    with raises(ValueError, "sliding_window(x, -1)"):
        sliding_window([1, 2, 3], -1)


def test_merge_equal_keys_and_unorderable_iterators():
    check(list(merge_sorted([1, 1, 3], [1, 2])), [1, 1, 1, 2, 3], "有重复值")
    check(list(merge_sorted(iter([1, 2]), iter([1, 2]))), [1, 1, 2, 2], "迭代器之间不能比较大小")
