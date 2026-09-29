from checker import check, check_close, raises
from solution import bubble_fraction, bubble_overhead, min_microbatches, peak_inflight


def test_example():
    check_close(bubble_fraction(8, 32), 7 / 39, what="p=8、m=32")
    check_close(bubble_fraction(8, 32, v=2), 7 / 71, what="交错式，v=2")
    check(min_microbatches(8, 0.1), 63, "气泡 ≤ 10%")


def test_overhead():
    check_close(bubble_overhead(8, 32), 7 / 32, what="开销")
    check_close(bubble_overhead(4, 16, v=4), 3 / 64, what="交错式开销")
    check_close(bubble_fraction(1, 5), 0.0, atol=1e-12, what="不切流水线就没有气泡")


def test_min_microbatches():
    check(min_microbatches(8, 0.1, v=2), 32, "交错式只要一半左右的 micro-batch")
    check(min_microbatches(4, 0.5), 3, "p=4、气泡 ≤ 50%")
    for p in (2, 4, 16):
        m = min_microbatches(p, 0.05)
        assert bubble_fraction(p, m) <= 0.05 < bubble_fraction(p, m - 1), f"p={p} 时 m={m} 应该恰好满足"


def test_memory():
    check(peak_inflight(8, 32, "gpipe"), 32, "GPipe 保存全部 micro-batch 的激活")
    check(peak_inflight(8, 32, "1f1b"), 8, "1F1B 最多保存 p 个")
    check(peak_inflight(8, 4, "1f1b"), 4, "m < p 时")
    with raises(ValueError, "未知调度"):
        peak_inflight(8, 4, "zb")
