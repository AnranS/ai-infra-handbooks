from checker import check
from solution import split_va, tlb_stats


def test_example_split():
    check(split_va(0x7F3A1C2D5E6F), (254, 232, 225, 213, 0xE6F), "四级下标和偏移")


def test_example_lru():
    check(tlb_stats([0, 4096, 0, 8192, 4096], 4096, 2), (1, 4), "LRU：第 0 页刚用过，淘汰的是第 1 页")


def test_split_edges():
    check(split_va(0), (0, 0, 0, 0, 0), "地址 0")
    check(split_va((1 << 48) - 1), (511, 511, 511, 511, 4095), "最大的 48 位地址")
    check(split_va(0x200000), (0, 0, 1, 0, 0), "2 MiB 正好是第 2 级下标 1")


def test_hot_page_survives():
    addrs = []
    for i in range(1, 50):                      # 第 0 页一直在用，其他页只用一次
        addrs += [0, i * 4096]
    hits, misses = tlb_stats(addrs, 4096, 4)
    check((hits, misses), (48, 50), "热页不会被 LRU 淘汰")


def test_huge_pages():
    scan = list(range(0, 8 << 20, 64))          # 按缓存行顺序扫描 8 MiB
    check(tlb_stats(scan, 4096, 64), (len(scan) - 2048, 2048), "4 KiB 页：每页缺失一次")
    check(tlb_stats(scan, 2 << 20, 64), (len(scan) - 4, 4), "2 MiB 页：只缺失 4 次")
