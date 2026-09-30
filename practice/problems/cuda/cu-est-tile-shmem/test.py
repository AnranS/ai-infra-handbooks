from checker import check
from solution import accum_regs, blocks_per_sm, is_feasible, shmem_bytes


def test_example():
    check(shmem_bytes(128, 128, 64, 3), 98304, "三级流水的 128x128x64")
    check(blocks_per_sm(128, 128, 64, 3), 2, "H100 上能放两个")
    check(accum_regs(128, 128, 256), 64, "每线程 64 个累加器寄存器")
    check(is_feasible(128, 128, 64, 3, 256), True, "可行")
    check(is_feasible(256, 256, 64, 4, 128), False, "放不下")


def test_stages_scale():
    two = shmem_bytes(128, 128, 64, 2)
    four = shmem_bytes(128, 128, 64, 4)
    check(four, two * 2, "级数翻倍，共享内存翻倍")
    check(blocks_per_sm(128, 128, 64, 2), 3, "两级流水能放三个 block")
    check(blocks_per_sm(128, 128, 64, 6), 1, "六级流水只能放一个")


def test_dtype():
    check(shmem_bytes(64, 64, 32, 2, dtype_bytes=1), shmem_bytes(64, 64, 32, 2, dtype_bytes=2) // 2,
          "FP8 的共享内存减半")


def test_regs():
    check(accum_regs(256, 256, 128), 512, "超过每线程 255 的上限")
    check(is_feasible(256, 256, 32, 1, 128), False, "寄存器放不下累加器")
    check(is_feasible(256, 256, 32, 1, 512), True, "线程多了就放得下")


def test_too_big():
    check(blocks_per_sm(256, 256, 128, 4), 0, "共享内存不够，一个都放不下")
    check(is_feasible(256, 256, 128, 4, 512), False, "不可行")


def test_search_space():
    # 在一组候选里筛出可行的配置——这就是自动调优的搜索空间
    candidates = [(bm, bn, 64, st, 256)
                  for bm in (64, 128, 256) for bn in (64, 128, 256) for st in (2, 3, 4)]
    feasible = [c for c in candidates if is_feasible(*c)]
    check(len(feasible) > 5, True, "有多个可行配置")
    check(all(blocks_per_sm(*c[:4]) >= 1 for c in feasible), True, "每个都至少放得下一个 block")
