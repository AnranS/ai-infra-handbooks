import asyncio


async def gather_limited(factories, limit):
    return await asyncio.gather(*(f() for f in factories))   # 没有限制并发
