from checker import check
from solution import PageCache


def test_example():
    c = PageCache(2)
    c.write(1)
    c.write(1)
    c.write(2)
    c.fsync()
    c.read(3)
    check(c.stats(), {"hits": 0, "disk_reads": 1, "disk_writes": 2, "dirty": 0}, "写合并、fsync 只写脏页、淘汰干净页不用写")


def test_evict_dirty_writes_back():
    c = PageCache(2)
    c.write(1)
    c.read(2)
    c.read(3)                                   # 淘汰脏页 1：写回一次
    check(c.stats(), {"hits": 0, "disk_reads": 2, "disk_writes": 1, "dirty": 0}, "淘汰脏页要写回")


def test_lru_order():
    c = PageCache(2)
    c.read(1)
    c.read(2)
    c.read(1)                                   # 1 最近用过
    c.read(3)                                   # 淘汰的是 2
    c.read(1)
    c.read(2)
    check(c.stats(), {"hits": 2, "disk_reads": 4, "disk_writes": 0, "dirty": 0}, "LRU：刚用过的页留下")


def test_fsync_twice():
    c = PageCache(4)
    for p in range(3):
        c.write(p)
    c.fsync()
    c.fsync()                                   # 没有新的脏页，第二次什么都不写
    c.write(1)
    c.fsync()
    check(c.stats()["disk_writes"], 4, "第一次写 3 页，之后只写重新变脏的 1 页")


def test_write_hit_counts():
    c = PageCache(3)
    c.write(7)
    c.read(7)
    check(c.stats(), {"hits": 1, "disk_reads": 0, "disk_writes": 0, "dirty": 1}, "写过的页再读是命中")
