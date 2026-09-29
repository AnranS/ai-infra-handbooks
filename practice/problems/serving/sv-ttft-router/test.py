from checker import check, check_close
from solution import Router


def close_tuple(actual, expected, what):
    check(actual is None, expected is None, f"{what}：是否拒绝")
    if expected is not None:
        check(actual[:2], expected[:2], f"{what}：实例与命中")
        check_close(actual[2], expected[2], what=f"{what}：TTFT")


def test_example():
    r = Router(2, speed=1000, slo=1.0, block=64)
    close_tuple(r.route(0.0, ["a", "b"], 200), (0, 0, 0.2), "第一个请求")
    close_tuple(r.route(0.0, ["a", "b"], 200), (1, 0, 0.2), "第二个请求去空闲的实例")
    close_tuple(r.route(0.5, ["a", "b", "c"], 300), (0, 128, 0.172), "命中两块")


def test_cache_beats_queue():
    r = Router(2, speed=1000, slo=10.0, block=100)
    r.route(0.0, ["s1", "s2", "s3"], 400)            # 实例 0：排到 0.4
    close_tuple(r.route(0.1, ["s1", "s2", "s3", "x"], 500), (0, 300, 0.5), "排队 0.3 秒但只要算 200 个 token，比去空闲实例算 500 个快")
    close_tuple(r.route(0.1, ["y"], 500), (1, 0, 0.5), "没有缓存可用时去空闲实例")


def test_reject():
    r = Router(1, speed=100, slo=1.0, block=10)
    close_tuple(r.route(0.0, [], 90), (0, 0, 0.9), "刚好达标")
    close_tuple(r.route(0.0, [], 50), None, "预计 1.4 秒，拒绝")
    check(r.busy[0], 0.9, "拒绝不改变状态")
    close_tuple(r.route(1.0, [], 50), (0, 0, 0.5), "队列清空后可以接收")


def test_prefix_must_be_contiguous():
    r = Router(1, speed=1000, slo=5.0, block=64)
    r.route(0.0, ["a", "b", "c"], 192)
    close_tuple(r.route(1.0, ["z", "b", "c"], 192), (0, 0, 0.192), "第一块不同，后面的块不能用")
