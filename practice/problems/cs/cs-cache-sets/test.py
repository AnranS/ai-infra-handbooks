from checker import check
from solution import cache_stats, pad_floats


def test_example_stats():
    check(cache_stats([0, 64, 0, 128], 128, 1), (1, 3), "2 组 1 路：地址 0 和 128 争同一组")


def test_example_pad():
    check(pad_floats(512, 512, 48 * 1024, 12), 16, "48 KB、12 路、64 组：每行补 16 个 float")


def test_same_line():
    check(cache_stats([0, 8, 16, 63], 4096, 4), (3, 1), "同一条缓存行里的四个地址")


def test_sequential():
    addrs = list(range(0, 4096, 8))                      # 顺序扫描 64 条缓存行
    check(cache_stats(addrs, 4096, 4), (len(addrs) - 64, 64), "顺序扫描：每条行缺失一次")


def test_lru_within_set():
    addrs = [0, 512, 0, 1024, 512]                       # 1 KB、2 路、8 组：三个地址都落进第 0 组
    check(cache_stats(addrs, 1024, 2), (1, 4), "组内 LRU：0 刚用过，被淘汰的是 512")


def test_column_walk():
    rows, cols = 64, 1024
    walk = [r * cols * 4 + k * 4 for k in range(cols) for r in range(rows)]
    hits, misses = cache_stats(walk, 48 * 1024, 12)
    check(misses, len(walk), "行距 4096 字节：64 行全落进同一组，每次都缺失")
    pad = pad_floats(rows, cols, 48 * 1024, 12)
    padded = [r * (cols + pad) * 4 + k * 4 for k in range(cols) for r in range(rows)]
    check(cache_stats(padded, 48 * 1024, 12)[1] < misses / 10, True, "补齐之后大部分访问命中")


def test_pad_edges():
    check(pad_floats(64, 1024, 48 * 1024, 12), 16, "行距 4096 = 64 条行，全撞在一组")
    check(pad_floats(64, 1008, 48 * 1024, 12), 0, "行距 4032 = 63 条行，与 64 互质，本来就分散")
    check(pad_floats(2, 512, 48 * 1024, 12), 0, "只有 2 行时，步长 32 给的 2 个组就够了")
