import asyncio

from checker import check, raises
from solution import DynamicBatcher


def make_model(delay=0.0, log=None):
    async def model(items):
        if log is not None:
            log.append(list(items))
        await asyncio.sleep(delay)
        return [x * 10 for x in items]

    return model


async def test_example_full_batches():
    """同时到达 10 个请求，max_batch_size=4：分成 4 + 4 + 2"""
    b = DynamicBatcher(make_model(0.005), max_batch_size=4, max_wait=0.05)
    got = await asyncio.gather(*(b.submit(i) for i in range(10)))
    check(list(got), [i * 10 for i in range(10)], "每个请求拿到自己的结果")
    check(b.batch_sizes, [4, 4, 2], "每批的大小")
    await b.close()


async def test_wait_timeout():
    """只有 1 个请求时，等 max_wait 之后单独处理"""
    b = DynamicBatcher(make_model(), max_batch_size=8, max_wait=0.05)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    check(await b.submit(3), 30, "结果")
    used = loop.time() - t0
    assert 0.04 <= used < 0.5, f"应该等大约 max_wait=0.05 秒再处理，实际 {used:.3f} 秒"
    check(b.batch_sizes, [1], "每批的大小")
    await b.close()


async def test_requests_during_processing_go_to_next_batch():
    log = []
    b = DynamicBatcher(make_model(0.05, log), max_batch_size=8, max_wait=0.01)
    first = asyncio.ensure_future(asyncio.gather(*(b.submit(i) for i in range(3))))
    await asyncio.sleep(0.03)          # 第一批正在处理
    second = asyncio.ensure_future(asyncio.gather(*(b.submit(i) for i in range(3, 5))))
    await first
    await second
    check(log, [[0, 1, 2], [3, 4]], "处理期间到达的请求进入下一批")
    await b.close()


async def test_staggered_arrivals():
    """第一个请求到达后 max_wait 内陆续到达的请求并入同一批"""
    log = []
    b = DynamicBatcher(make_model(0, log), max_batch_size=10, max_wait=0.08)

    async def later(i, d):
        await asyncio.sleep(d)
        return await b.submit(i)

    got = await asyncio.gather(later(0, 0), later(1, 0.02), later(2, 0.04), later(3, 0.2))
    check(list(got), [0, 10, 20, 30], "结果")
    check(log, [[0, 1, 2], [3]], "批次划分")
    await b.close()


async def test_model_error_propagates_and_recovers():
    calls = {"n": 0}

    async def flaky(items):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("GPU OOM")
        return [x + 1 for x in items]

    b = DynamicBatcher(flaky, max_batch_size=4, max_wait=0.01)
    res = await asyncio.gather(*(b.submit(i) for i in range(2)), return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in res), f"这一批的每个请求都应该收到 ValueError，实际：{res}"
    check(await b.submit(5), 6, "出错之后批处理器继续工作")
    await b.close()


async def test_close():
    b = DynamicBatcher(make_model(0.01), max_batch_size=4, max_wait=0.5)
    pending = asyncio.ensure_future(asyncio.gather(*(b.submit(i) for i in range(2))))
    await asyncio.sleep(0.01)
    await b.close()
    check(list(await pending), [0, 10], "close() 之前提交的请求都要处理完")
    with raises(RuntimeError, "close() 之后 submit"):
        await b.submit(1)
