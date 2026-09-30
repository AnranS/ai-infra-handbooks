from checker import check, check_close
from solution import hit_rate_needed, matched_len, prefill_ms, saved_ms


def test_example():
    check(matched_len((1, 2, 3, 4), (1, 2, 9)), 2, "公共前缀")
    check_close(prefill_ms(4096), 614.4, rtol=1e-9, what="4096 个 token")
    check_close(saved_ms((1, 2, 3), (1, 2, 9)), 0.3, rtol=1e-9, what="省下两个 token")
    check_close(hit_rate_needed(200, 4096), 0.707, rtol=0.01, what="需要七成命中")


def test_matched_edges():
    check(matched_len((), (1, 2)), 0, "空缓存")
    check(matched_len((1, 2), ()), 0, "空请求")
    check(matched_len((1, 2), (1, 2)), 2, "完全相同")
    check(matched_len((1, 2), (1, 2, 3)), 2, "请求更长")
    check(matched_len((9, 1, 2), (1, 2, 3)), 0, "顺序不同就不算命中")


def test_order_matters():
    check(matched_len((1, 2, 3), (3, 2, 1)), 0, "同样的 token 但顺序相反")
    check(matched_len((1, 2, 3), (1, 3, 2)), 1, "只有第一个相同")


def test_hit_rate_edges():
    check(hit_rate_needed(20, 4096), 1.0, "预算正好被固定开销吃掉")
    check(hit_rate_needed(10, 4096), 1.0, "目标比固定开销还小")
    check(hit_rate_needed(10000, 4096), 0.0, "预算充足，不需要命中")
    check_close(hit_rate_needed(1000, 4096), 0.0, rtol=1e-9, what="4096 个 token 只要 614 ms")


def test_long_prompt():
    check_close(hit_rate_needed(500, 32768), 0.902, rtol=0.01, what="32K 的 prompt 需要九成命中")
    check_close(prefill_ms(32768), 4915.2, rtol=1e-9, what="不命中时要 4.9 秒")


def test_saved():
    system = tuple(range(1000))
    req = system + (9999,)
    check_close(saved_ms(system, req), 150.0, rtol=1e-9, what="1000 个 token 的系统提示词省 150 ms")
    check_close(saved_ms((9999,) + system, req), 0.0, rtol=1e-9, what="开头变了就一点也省不到")
