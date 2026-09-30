from checker import check, check_close
from solution import batch_waste, sorted_waste, total_waste, waste_ratio


def test_example():
    check(batch_waste([500, 20]), 480, "两个请求")
    check(total_waste([500, 20, 480, 30], 2), 930, "按到达顺序")
    check(sorted_waste([500, 20, 480, 30], 2), 30, "按长度排序后")
    check_close(waste_ratio([500, 20, 480, 30], 2), 0.475, rtol=0.01, what="浪费占比")


def test_edges():
    check(batch_waste([]), 0, "空批次")
    check(batch_waste([100]), 0, "一个请求没有浪费")
    check(batch_waste([50, 50, 50]), 0, "长度相同没有浪费")
    check(total_waste([], 4), 0, "没有请求")
    check(waste_ratio([], 4), 0.0, "没有请求时比例为 0")


def test_partial_batch():
    check(total_waste([10, 20, 30], 2), 10, "最后一批只有一个请求")
    check(total_waste([10, 20, 30], 5), 30, "一批装下全部")


def test_sorting_helps():
    reqs = [1000, 10, 900, 20, 800, 30]
    check(sorted_waste(reqs, 2) < total_waste(reqs, 2), True, "排序后浪费更少")
    check(sorted_waste(reqs, 2), 880, "排序后仍有一批跨越长短（30 和 800）")


def test_batch_size_one():
    check(total_waste([1, 2, 3], 1), 0, "每批一个请求时没有浪费")
    check(waste_ratio([1, 2, 3], 1), 0.0, "浪费比例为 0")


def test_large():
    reqs = [(i * 37) % 2000 + 1 for i in range(4096)]
    unsorted_ratio = waste_ratio(reqs, 64)
    sorted_ratio = waste_ratio(reqs, 64, sort_first=True)
    check(sorted_ratio < unsorted_ratio, True, f"排序后浪费比例更低（{sorted_ratio:.3f} < {unsorted_ratio:.3f}）")
    check(sorted_ratio < 0.05, True, "按长度分组后浪费很小")
