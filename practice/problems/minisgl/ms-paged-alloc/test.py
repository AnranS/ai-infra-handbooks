import numpy as np

from checker import check, raises
from solution import PageAllocator


def test_example():
    a = PageAllocator(num_pages=8, page_size=4, max_rows=2, max_len=16)
    a.allocate_paged([(0, 0, 6), (1, 0, 3)])            # 请求 0 要 2 页，请求 1 要 1 页
    check(a.page_table[0, :8].tolist(), [0, 1, 2, 3, 4, 5, 6, 7], "请求 0 的 page table")
    check(a.page_table[1, :4].tolist(), [8, 9, 10, 11], "请求 1 的 page table")
    check(a.out_loc([(0, 0, 6), (1, 0, 3)]).tolist(), [0, 1, 2, 3, 4, 5, 8, 9, 10], "out_loc")
    check(a.free_slots, [12, 16, 20, 24, 28], "剩下的空闲页")


def test_decode_reuses_partial_page():
    a = PageAllocator(8, 4, 1, 32)
    a.allocate_paged([(0, 0, 6)])
    a.allocate_paged([(0, 6, 7)])                       # 第 7 个 token 还在第 2 页里：不分配
    check(len(a.free_slots), 6, "没有新分配")
    a.allocate_paged([(0, 7, 9)])                       # 跨进第 3 页
    check(a.page_table[0, 8:12].tolist(), [8, 9, 10, 11], "第 3 页")
    check(a.out_loc([(0, 7, 9)]).tolist(), [7, 8], "out_loc 跨页")


def test_page_size_one():
    a = PageAllocator(5, 1, 2, 8)
    a.allocate_paged([(1, 0, 2), (0, 0, 3)])
    check(a.page_table[:, :3].tolist(), [[2, 3, 4], [0, 1, -1]], "page_size=1")


def test_out_of_memory():
    a = PageAllocator(3, 2, 2, 8)
    a.allocate_paged([(0, 0, 4)])
    before = (list(a.free_slots), a.page_table.copy())
    with raises(MemoryError, "只剩 1 页却要 2 页"):
        a.allocate_paged([(1, 0, 1), (0, 4, 6)])
    check((a.free_slots, a.page_table.tolist()), (before[0], before[1].tolist()), "失败后状态不变")


def test_free_and_lazy():
    a = PageAllocator(4, 2, 2, 8)
    a.allocate_paged([(0, 0, 8)])
    a.free(a.page_table[0, 2:6])                        # 释放第 2、3 页（位置 2..5）
    check(a.free_slots, [2, 4], "free 追加页首位置")
    with a.lazy_free():
        a.free(a.page_table[0, 6:8])
        a.free(a.page_table[0, 0:2])
        check(a.free_slots, [2, 4], "lazy 区域内还没有真正释放")
    check(a.free_slots, [2, 4, 6, 0], "退出时按调用顺序追加")
