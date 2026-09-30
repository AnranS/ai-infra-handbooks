from checker import check, check_close
from solution import KthLargest, MedianFinder


def test_example():
    kth = KthLargest(3, [4, 5, 8, 2])
    check(kth.add(3), 4, "第三大")
    check(kth.add(5), 5, "加入 5 之后")
    m = MedianFinder()
    m.add(1)
    m.add(2)
    check_close(m.median(), 1.5, rtol=1e-9, what="两个数的中位数")


def test_kth_edges():
    kth = KthLargest(2, [])
    check(kth.add(1), None, "不足 k 个")
    check(kth.add(5), 1, "正好 k 个")
    check(kth.add(0), 1, "更小的不影响")
    check(kth.add(9), 5, "更大的挤掉最小的")


def test_kth_duplicates():
    kth = KthLargest(2, [3, 3, 3])
    check(kth.add(3), 3, "全相同")
    check(kth.add(4), 3, "第二大还是 3")
    check(kth.add(5), 4, "现在是 4")


def test_median_edges():
    m = MedianFinder()
    check(m.median(), None, "空")
    m.add(5)
    check_close(m.median(), 5.0, rtol=1e-9, what="单个元素")
    m.add(5)
    check_close(m.median(), 5.0, rtol=1e-9, what="两个相同")


def test_median_order():
    m = MedianFinder()
    for x in (6, 10, 2, 6, 5, 0):
        m.add(x)
    check_close(m.median(), 5.5, rtol=1e-9, what="乱序插入")


def test_median_against_sorted():
    import random
    rng = random.Random(11)
    m = MedianFinder()
    ref = []
    for _ in range(2000):
        x = rng.randrange(1000)
        m.add(x)
        ref.append(x)
        ref.sort()
        n = len(ref)
        want = ref[n // 2] if n % 2 else (ref[n // 2 - 1] + ref[n // 2]) / 2
        check_close(m.median(), want, rtol=1e-9, what="始终与排序结果一致")


def test_large():
    kth = KthLargest(5)
    for i in range(100000):
        kth.add(i)
    check(kth.add(-1), 99995, "十万个元素的第五大")
