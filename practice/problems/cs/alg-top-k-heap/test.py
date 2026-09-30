from checker import check
from solution import kth_largest, top_k_heap


def test_example():
    check(top_k_heap([3, 2, 1, 5, 6, 4], 2), [6, 5], "前两大")
    check(kth_largest([3, 2, 1, 5, 6, 4], 2), 5, "第二大")
    check(kth_largest([1], 1), 1, "单元素")


def test_edges():
    check(top_k_heap([], 3), [], "空数组")
    check(top_k_heap([1, 2], 0), [], "k 为 0")
    check(top_k_heap([1, 2], -1), [], "k 为负")
    check(top_k_heap([1, 2], 5), [2, 1], "k 超过长度")
    check(kth_largest([], 1), None, "空数组")
    check(kth_largest([1, 2], 3), None, "k 越界")
    check(kth_largest([1, 2], 0), None, "k 从 1 开始")


def test_duplicates():
    check(top_k_heap([1, 1, 1, 1], 2), [1, 1], "全相同")
    check(kth_largest([3, 3, 3], 2), 3, "重复元素")
    check(kth_largest([1, 2, 2, 3], 2), 2, "第二大是 2")
    check(kth_largest([1, 2, 2, 3], 3), 2, "第三大还是 2")


def test_negative():
    check(top_k_heap([-5, -1, -3], 2), [-1, -3], "全负数")
    check(kth_largest([-5, -1, -3], 1), -1, "最大的负数")


def test_consistency():
    import random
    rng = random.Random(3)
    for _ in range(200):
        a = [rng.randrange(20) for _ in range(rng.randrange(1, 15))]
        k = rng.randrange(1, len(a) + 1)
        want = sorted(a, reverse=True)
        check(top_k_heap(a, k), want[:k], f"与排序一致：{a}, k={k}")
        check(kth_largest(a, k), want[k - 1], f"第 k 大一致：{a}, k={k}")


def test_large():
    n = 200000
    a = list(range(n))
    check(top_k_heap(a, 3), [n - 1, n - 2, n - 3], "二十万个元素取前三")
    check(kth_largest(a, 1000), n - 1000, "第 1000 大")
