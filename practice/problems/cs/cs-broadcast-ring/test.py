from checker import check
from solution import Ring


def test_example():
    r = Ring(n_readers=2, n_chunks=2)
    check((r.try_write("a"), r.try_write("b")), (True, True), "两块都空着")
    check(r.try_write("c"), False, "第 0 块还有读者没读，不能覆盖")
    check((r.try_read(0), r.try_read(1)), ("a", "a"), "两个读者都读到 a")
    check(r.try_write("c"), True, "第 0 块读完了，可以覆盖")


def test_order_of_flags():
    r = Ring(n_readers=3, n_chunks=2)
    r.try_write("x")
    r.log.clear()
    for k in range(3):
        r.try_read(k)
    r.try_write("y")
    r.try_write("z")                                # 覆盖第 0 块：此时读标志都是 1
    writes = r.log[3:]                              # 跳过三个读者置读标志的记录
    last_flags = [(i, v) for i, v in writes if i < 4]   # 第 0 块的四个标志
    check(last_flags[-1], (0, 1), "最后一步必须是把写标志置 1")
    check(all(v == 0 for i, v in last_flags[:-1] if i != 0), True, "读标志要在写标志置 1 之前清零")


def test_slow_reader_blocks_writer():
    r = Ring(n_readers=2, n_chunks=3)
    for m in "abc":
        check(r.try_write(m), True, f"写 {m}")
    for _ in range(3):
        r.try_read(0)                               # 读者 0 读完了，读者 1 一条都没读
    check(r.try_write("d"), False, "最慢的读者没读，写者只能等")
    check(r.try_read(1), "a", "读者 1 读到第一条")
    check(r.try_write("d"), True, "现在第 0 块可以覆盖了")
    check([r.try_read(1), r.try_read(1), r.try_read(1)], ["b", "c", "d"], "读者 1 按顺序读完")
    check(r.try_read(0), "d", "读者 0 读到新写的 d")


def test_no_double_read():
    r = Ring(n_readers=1, n_chunks=4)
    r.try_write(1)
    check(r.try_read(0), 1, "第一次读")
    check(r.try_read(0), None, "没有新数据时返回 None，不会重复读")


def test_many_rounds():
    r = Ring(n_readers=2, n_chunks=3)
    got = {0: [], 1: []}
    sent = 0
    for step in range(200):
        if sent < 50 and r.try_write(sent):
            sent += 1
        reader = step % 2
        if step % 5 != 0:                            # 读者偶尔落后
            v = r.try_read(reader)
            if v is not None:
                got[reader].append(v)
    for reader in (0, 1):
        while (v := r.try_read(reader)) is not None:
            got[reader].append(v)
    check(sent, 50, "全部写出")
    check(got, {0: list(range(50)), 1: list(range(50))}, "每个读者都按顺序收到全部 50 条")
