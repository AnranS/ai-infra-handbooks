from checker import check
from solution import max_concurrent, merge, min_remove


def test_example():
    check(merge([[1, 3], [2, 6], [8, 10]]), [[1, 6], [8, 10]], "合并")
    check(min_remove([[1, 2], [2, 3], [3, 4], [1, 3]]), 1, "删一个")
    check(max_concurrent([[1, 4], [2, 5], [5, 6]]), 2, "最大并发")


def test_merge_edges():
    check(merge([]), [], "空列表")
    check(merge([[1, 4]]), [[1, 4]], "单个区间")
    check(merge([[1, 4], [2, 3]]), [[1, 4]], "被包含的区间")
    check(merge([[1, 2], [3, 4]]), [[1, 2], [3, 4]], "相邻但不重叠")
    check(merge([[4, 5], [1, 2]]), [[1, 2], [4, 5]], "输入无序")
    check(merge([[1, 4], [4, 5]]), [[1, 5]], "端点相接算重叠")


def test_remove_edges():
    check(min_remove([]), 0, "空列表")
    check(min_remove([[1, 2]]), 0, "单个")
    check(min_remove([[1, 2], [1, 2], [1, 2]]), 2, "三个完全相同")
    check(min_remove([[1, 2], [2, 3]]), 0, "端点相接不算重叠")


def test_remove_long_interval():
    # 一个长区间和三个短区间：按起点排序会保留长的，只能留 1 个；按终点排序能留 3 个
    check(min_remove([[1, 100], [2, 3], [4, 5], [6, 7]]), 1, "长区间要被删掉")


def test_concurrent_edges():
    check(max_concurrent([]), 0, "空列表")
    check(max_concurrent([[1, 2]]), 1, "单个")
    check(max_concurrent([[1, 2], [2, 3]]), 1, "接续不算同时")
    check(max_concurrent([[1, 5], [2, 4], [3, 6]]), 3, "三个重叠")


def test_large():
    n = 20000
    nested = [[i, 2 * n - i] for i in range(n)]            # 层层嵌套
    check(max_concurrent(nested), n, "全部嵌套时并发数等于个数")
    check(merge(nested), [[0, 2 * n]], "合并成一个")
    disjoint = [[3 * i, 3 * i + 1] for i in range(n)]
    check(min_remove(disjoint), 0, "互不重叠时一个都不用删")
