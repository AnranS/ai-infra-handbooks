import asyncio

from checker import check, raises
from solution import gather_limited


class Probe:
    def __init__(self):
        self.running = 0
        self.peak = 0
        self.started = []
        self.finished = []

    def make(self, i, delay=0.01, fail=False):
        async def job():
            self.started.append(i)
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                await asyncio.sleep(delay)
                if fail:
                    raise RuntimeError(f"job {i} failed")
                return i * i
            finally:
                self.running -= 1
                self.finished.append(i)

        return job


async def test_example():
    p = Probe()
    got = await gather_limited([p.make(i) for i in range(10)], limit=3)
    check(got, [i * i for i in range(10)], "结果（按输入顺序）")
    check(p.peak, 3, "最大并发数")


async def test_order_not_completion_order():
    p = Probe()
    delays = [0.05, 0.01, 0.03, 0.02]
    got = await gather_limited([p.make(i, d) for i, d in enumerate(delays)], limit=4)
    check(got, [0, 1, 4, 9], "结果顺序与输入一致")


async def test_lazy_creation():
    """还没轮到的工厂不会被提前调用"""
    p = Probe()
    task = asyncio.ensure_future(gather_limited([p.make(i, 0.05) for i in range(6)], limit=2))
    await asyncio.sleep(0.02)
    check(sorted(p.started), [0, 1], "开始 0.02 秒时只启动了前 2 个")
    await task
    check(len(p.finished), 6, "全部完成")


async def test_failure_cancels_others():
    p = Probe()
    jobs = [p.make(0, 0.2), p.make(1, 0.01, fail=True), p.make(2, 0.2), p.make(3, 0.2)]
    with raises(RuntimeError, "有任务失败时"):
        await gather_limited(jobs, limit=2)
    await asyncio.sleep(0.01)
    check(p.running, 0, "失败后还在运行的任务数")
    assert 3 not in p.started, "失败之后不应该再启动新任务"


async def test_limit_one_and_empty():
    p = Probe()
    check(await gather_limited([p.make(i, 0.001) for i in range(5)], limit=1), [0, 1, 4, 9, 16], "limit=1")
    check(p.peak, 1, "limit=1 时的最大并发")
    check(await gather_limited([], limit=3), [], "空列表")
    with raises(ValueError, "limit=0"):
        await gather_limited([p.make(0)], limit=0)
