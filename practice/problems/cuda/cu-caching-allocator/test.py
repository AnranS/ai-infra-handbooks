import random

from checker import check, raises
from solution import CachingAllocator

KB, MB = 1 << 10, 1 << 20


def test_example():
    al = CachingAllocator(64 * MB)
    a = al.malloc(1000)
    check((a.addr, a.size), (0, 1024), "取整到 1024，小池的第一段从地址 0 开始")
    b = al.malloc(5 * MB)
    check((b.addr, b.size), (2 * MB, 5 * MB), "大池新段 [2 MiB, 22 MiB)")
    check((al.memory_allocated(), al.memory_reserved()), (1024 + 5 * MB, 22 * MB), "统计")


def test_small_pool_split_and_best_fit():
    al = CachingAllocator(64 * MB)
    a = al.malloc(1000)
    b = al.malloc(1)
    check((b.addr, b.size), (1024, 512), "同一段里切下一块，至少 512")
    al.free(a)
    c = al.malloc(600)
    check((c.addr, c.size), (0, 1024), "最佳适配：选 [0, 1024) 这块；剩余 0 字节不切")
    d = al.malloc(256)
    check(d.addr, 1536, "剩下的空闲块从 1536 开始")
    al.free(c)
    e = al.malloc(300)
    check((e.addr, e.size), (0, 512), "从 1024 的空闲块里切出 512，剩 512 仍然切下来")
    check(al.snapshot()[0][3][:4], [(0, 512, True), (512, 512, False), (1024, 512, True), (1536, 512, True)], "小池段里的块")


def test_large_pool_rules():
    al = CachingAllocator(200 * MB)
    a = al.malloc(5 * MB)
    b = al.malloc(int(14.5 * MB))
    check((b.addr, b.size), (5 * MB, 15 * MB), "剩余 0.5 MiB 不超过 1 MiB：整块 15 MiB 都给它")
    check(al.memory_allocated(), 20 * MB, "统计按整块算")
    c = al.malloc(12 * MB + 1)
    check((c.addr, c.size, al.memory_reserved()), (20 * MB, 12 * MB + 512, 34 * MB), "不小于 10 MiB 的请求：段按 2 MiB 取整（14 MiB）")
    d = al.malloc(1 * MB)
    check(d.addr, 34 * MB, "1 MiB 走小池，开一个新的 2 MiB 段")
    e = al.malloc(1 * MB + 512)
    check((e.addr, e.size), (20 * MB + 12 * MB + 512, 2 * MB - 512), "用大池 14 MiB 段切剩的 (2 MiB - 512) 那块；再切的话剩余不到 1 MiB，所以整块给它")


def test_merge_and_empty_cache():
    al = CachingAllocator(100 * MB)
    x = [al.malloc(4 * MB) for _ in range(4)]          # 一个 20 MiB 段：4 × 4 MiB + 剩余 4 MiB
    check(len(al.snapshot()), 1, "4 个请求在同一段里")
    al.free(x[1])
    al.free(x[2])
    blocks = al.snapshot()[0][3]
    check(blocks, [(0, 4 * MB, True), (4 * MB, 8 * MB, False), (12 * MB, 4 * MB, True), (16 * MB, 4 * MB, False)], "中间两块合并")
    y = al.malloc(7 * MB)
    check((y.addr, y.size), (4 * MB, 8 * MB), "合并后的 8 MiB 块被最佳适配选中（剩 1 MiB 不切）")
    al.free(y)
    al.free(x[3])
    al.free(x[0])
    check(al.snapshot()[0][3], [(0, 20 * MB, False)], "全部释放后合并成一整块")
    check(al.memory_reserved(), 20 * MB, "释放后显存仍然被缓存着")
    al.empty_cache()
    check((al.memory_reserved(), al.snapshot()), (0, []), "empty_cache 把整段还给驱动")
    with raises(ValueError, "重复释放"):
        al.free(x[0])


def test_streams():
    al = CachingAllocator(100 * MB)
    a = al.malloc(4096, stream=0)
    al.free(a)
    b = al.malloc(4096, stream=1)
    check(b.addr, 2 * MB, "stream 1 不能用 stream 0 缓存的块")
    c = al.malloc(4096, stream=0)
    check(c.addr, 0, "stream 0 的请求复用自己的缓存")


def test_fragmentation_oom():
    al = CachingAllocator(40 * MB)
    A, B, C, D = (al.malloc(8 * MB) for _ in range(4))
    check([blk.addr for blk in (A, B, C, D)], [0, 8 * MB, 20 * MB, 28 * MB], "两段各放两块")
    al.free(A)
    al.free(C)
    check(al.memory_reserved() - al.memory_allocated(), 24 * MB, "保留但空闲的显存有 24 MiB")
    with raises(MemoryError, "空闲的显存不连续：分不出 12 MiB"):
        al.malloc(12 * MB)
    al.free(B)
    al.free(D)
    big = al.malloc(30 * MB)                           # 超出上限：先释放两个完全空闲的段，再向驱动要
    check((big.addr, al.memory_reserved()), (40 * MB, 30 * MB), "empty_cache 之后分配成功")


def test_random_invariants():
    rng = random.Random(0)
    al = CachingAllocator(256 * MB)
    live = []
    for step in range(1500):
        if live and rng.random() < 0.45:
            al.free(live.pop(rng.randrange(len(live))))
        else:
            size = rng.choice([rng.randint(1, 4 * KB), rng.randint(1, 900 * KB), rng.randint(1 * MB, 12 * MB)])
            try:
                live.append(al.malloc(size, stream=rng.randint(0, 1)))
            except MemoryError:
                pass
        if step % 50 == 0 or step > 1450:
            snap = al.snapshot()
            reserved = sum(s[1] for s in snap)
            check(reserved, al.memory_reserved(), "memory_reserved 等于各段之和")
            check(reserved <= 256 * MB, True, "不超过上限")
            alloc = 0
            for base, size, stream, blocks in snap:
                check(sum(b[1] for b in blocks), size, "段里的块正好铺满整段")
                pos = base
                for i, (addr, bsize, used) in enumerate(blocks):
                    check(addr, pos, "块按地址连续排列")
                    pos += bsize
                    alloc += bsize if used else 0
                    if i and not used:
                        check(blocks[i - 1][2], True, "不能有两个相邻的空闲块（应该合并）")
            check(alloc, al.memory_allocated(), "memory_allocated 等于已分配块之和")
            check(sorted((b.addr, b.size) for b in live), sorted((a, s) for _, _, _, bl in snap for a, s, u in bl if u),
                  "已分配的块就是还没释放的那些")
