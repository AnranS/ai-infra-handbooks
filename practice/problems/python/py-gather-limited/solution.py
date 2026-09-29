import asyncio


async def gather_limited(factories, limit):
    if limit < 1:
        raise ValueError("limit 必须 >= 1")
    results = [None] * len(factories)
    it = iter(enumerate(factories))

    async def worker():
        for i, factory in it:
            results[i] = await factory()

    workers = [asyncio.ensure_future(worker()) for _ in range(min(limit, len(factories)))]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    return results
