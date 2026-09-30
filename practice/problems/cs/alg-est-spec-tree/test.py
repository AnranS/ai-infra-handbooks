from checker import check, check_close
from solution import accept_expect, best_k, speedup


def test_example():
    check_close(accept_expect(0.8, 4), 3.362, rtol=0.01, what="期望产出")
    check_close(speedup(0.8, 4, 0.1, 1.0), 2.401, rtol=0.01, what="加速比")
    check(best_k(0.8, 0.1, 1.0), 6, "接受率高时可以猜多一点")
    check(best_k(0.3, 0.1, 1.0), 1, "接受率低时只猜一个")


def test_expect_edges():
    check_close(accept_expect(0.5, 0), 1.0, rtol=1e-9, what="不猜时只有目标模型的一个")
    check_close(accept_expect(0.0, 5), 1.0, rtol=1e-9, what="全都猜不中")
    check_close(accept_expect(1.0, 5), 6.0, rtol=1e-9, what="全都猜中")


def test_speedup_edges():
    check_close(speedup(0.5, 0, 0.1, 1.0), 1.0, rtol=1e-9, what="k=0 时就是不用投机")
    check(speedup(0.0, 4, 0.1, 1.0) < 1.0, True, "全猜不中时反而更慢")
    check(speedup(0.95, 4, 0.01, 1.0) > 4.0, True, "接受率高、草稿便宜时加速明显")


def test_best_k_monotone():
    ks = [best_k(p, 0.1, 1.0) for p in (0.2, 0.5, 0.8, 0.95)]
    check(ks == sorted(ks), True, f"接受率越高，最优 k 越大：{ks}")


def test_expensive_draft():
    check(best_k(0.8, 0.9, 1.0), 0, "草稿模型太贵时干脆别猜")
    check(best_k(0.8, 2.0, 1.0), 0, "草稿比目标还贵就别用了")


def test_large_k():
    check(best_k(0.99, 0.01, 1.0, max_k=64) > 10, True, "接受率极高时可以猜很长")
    check_close(accept_expect(0.9, 10), sum(0.9 ** i for i in range(11)), rtol=1e-9, what="公式一致")
