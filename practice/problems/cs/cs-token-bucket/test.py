from checker import check, check_close
from solution import TokenBucket


def test_example():
    b = TokenBucket(rate=10, capacity=10)
    check(b.allow(0.0, 10), True, "桶是满的")
    check(b.allow(0.0, 1), False, "一个令牌都不剩")
    check_close(b.wait_time(0.0, 5), 0.5, rtol=1e-9, what="攒 5 个令牌要 0.5 秒")
    check(b.allow(1.0, 5), True, "一秒后补满，够了")


def test_reject_does_not_consume():
    b = TokenBucket(rate=10, capacity=10)
    check(b.allow(0.0, 8), True, "先花掉 8 个")
    check(b.allow(0.0, 5), False, "只剩 2 个，拒绝")
    check(b.allow(0.0, 2), True, "拒绝不扣令牌，所以这 2 个还在")


def test_capacity_cap():
    b = TokenBucket(rate=10, capacity=10)
    check(b.allow(0.0, 10), True, "先清空")
    check(b.allow(100.0, 10), True, "空闲很久后最多攒满一桶")
    check(b.allow(100.0, 1), False, "不会攒出 1000 个令牌")


def test_wait_time_is_pure():
    b = TokenBucket(rate=100, capacity=100)
    b.allow(0.0, 100)
    check_close(b.wait_time(0.0, 50), 0.5, rtol=1e-9, what="等 0.5 秒")
    check_close(b.wait_time(0.0, 50), 0.5, rtol=1e-9, what="查询不改变状态，再查一次还是 0.5")
    check(b.allow(0.5, 50), True, "0.5 秒后正好够")


def test_steady_rate():
    b = TokenBucket(rate=1000, capacity=1000)          # 每秒 1000 个 token
    passed = sum(b.allow(i * 0.001, 2) for i in range(2000))   # 2 秒里每毫秒来一个 2 token 的请求
    # 可用令牌 = 初始 1000 + 这 2 秒生成的 2000，每个请求花 2 个，所以约 1500 个能通过
    check(1450 <= passed <= 1500, True, f"长期通过率被速率限制住（实际 {passed}）")
