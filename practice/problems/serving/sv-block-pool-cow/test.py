from checker import check, raises
from solution import BlockPool, Sequence


def test_example():
    pool = BlockPool(8)
    a = Sequence(pool, 4)
    a.append(6)                                   # 2 个块，最后一块写了 2 个
    check(a.block_table, [0, 1], "a 的块表")
    b = a.fork()
    check((b.num_tokens, b.block_table), (6, [0, 1]), "fork 出来的序列")
    check(pool.ref_cnt[:2], [2, 2], "共享后的引用计数")
    b.append(1)                                   # 写进共享的、没写满的块 1 -> 写时复制
    check(pool.copies, [(1, 2)], "写时复制记录")
    check((b.block_table, a.block_table), ([0, 2], [0, 1]), "复制后两个块表")
    check(pool.ref_cnt[:3], [2, 1, 1], "引用计数")


def test_no_copy_when_last_block_full():
    pool = BlockPool(6)
    a = Sequence(pool, 4)
    a.append(8)
    b = a.fork()
    b.append(3)
    check(pool.copies, [], "最后一块写满时不需要复制")
    check(b.block_table, [0, 1, 2], "b 的块表")
    a.append(1)
    check(a.block_table, [0, 1, 3], "a 的块表")


def test_slot_mapping():
    pool = BlockPool(10)
    pool.allocate(3)                              # 先占掉 0、1、2
    s = Sequence(pool, 4)
    s.append(7)
    check(s.block_table, [3, 4], "块表")
    check(s.slot_mapping([0, 3, 4, 6]), [12, 15, 16, 18], "slot mapping")


def test_out_of_memory_keeps_state():
    pool = BlockPool(3)
    a = Sequence(pool, 4)
    a.append(5)
    b = a.fork()
    b.append(3)                                   # 复制 1 块，此时已经用满 3 块
    snapshot = (list(pool.ref_cnt), pool.num_free(), list(b.block_table), b.num_tokens, list(pool.copies))
    with raises(MemoryError, "空间不够时 append"):
        b.append(10)
    check((list(pool.ref_cnt), pool.num_free(), list(b.block_table), b.num_tokens, list(pool.copies)), snapshot,
          "失败后状态不变")


def test_free_and_reuse():
    pool = BlockPool(4)
    a = Sequence(pool, 2)
    a.append(3)
    b = a.fork()
    c = b.fork()
    a.free()
    check(pool.num_free(), 2, "a 释放后，共享的块还被 b、c 引用")
    b.free()
    c.free()
    check(pool.num_free(), 4, "全部释放")
    check(pool.ref_cnt, [0, 0, 0, 0], "引用计数")
    check(a.num_tokens, 0, "释放后 num_tokens")


def test_beam_like_many_forks():
    pool = BlockPool(20)
    root = Sequence(pool, 4)
    root.append(5)
    kids = [root.fork() for _ in range(3)]
    for k in kids:
        k.append(1)
    check(len(pool.copies), 3, "三个分支各复制一次共享的末尾块")
    check(pool.ref_cnt[root.block_table[0]], 4, "第一块被 4 个序列共享")
    check(pool.ref_cnt[root.block_table[1]], 1, "root 的第二块只剩它自己")
