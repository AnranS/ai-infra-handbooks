from checker import check, check_close, raises
from solution import comm_bytes, model_state_bytes, partition


def test_example():
    check(partition([3, 5, 2], 4), (3, [[(0, 0, 3)], [(1, 0, 3)], [(1, 3, 5), (2, 0, 1)], [(2, 1, 2)]]), "题目中的例子")


def test_partition():
    check(partition([4, 4], 2), (4, [[(0, 0, 4)], [(1, 0, 4)]]), "恰好整除")
    check(partition([10], 3), (4, [[(0, 0, 4)], [(0, 4, 8)], [(0, 8, 10)]]), "一个参数跨三个 rank")
    check(partition([1], 4), (1, [[(0, 0, 1)], [], [], []]), "大部分 rank 只有补零")
    per, segs = partition([7, 1, 9, 3, 11], 5)
    covered = sorted((i, a, b) for s in segs for (i, a, b) in s)
    for i, n in enumerate([7, 1, 9, 3, 11]):
        parts = [(a, b) for (j, a, b) in covered if j == i]
        check(sum(b - a for a, b in parts), n, f"第 {i} 个参数的每个元素都要恰好被一个 rank 负责")
    for r, s in enumerate(segs):
        assert sum(b - a for _, a, b in s) <= per, f"rank {r} 负责的元素超过了 per"


def test_memory():
    G = 2**30
    check_close(model_state_bytes(70e9, 64, 0) / G, 16 * 70e9 / G, what="不切分：每参数 16 字节")
    check_close(model_state_bytes(70e9, 64, 1) / G, (4 * 70e9 + 12 * 70e9 / 64) / G, what="ZeRO-1")
    check_close(model_state_bytes(70e9, 64, 2) / G, (2 * 70e9 + 14 * 70e9 / 64) / G, what="ZeRO-2")
    check_close(model_state_bytes(70e9, 64, 3) / G, 16 * 70e9 / 64 / G, what="ZeRO-3")
    with raises(ValueError):
        model_state_bytes(1e9, 8, 4)


def test_comm():
    check_close(comm_bytes(1e9, 8, 0), 2 * 7 / 8 * 2e9, what="数据并行：一次 all-reduce")
    check_close(comm_bytes(1e9, 8, 2), comm_bytes(1e9, 8, 0), what="ZeRO-2 与数据并行相同")
    check_close(comm_bytes(1e9, 8, 3) / comm_bytes(1e9, 8, 1), 1.5, what="ZeRO-3 多 50%")
    with raises(ValueError):
        comm_bytes(1e9, 8, -1)
