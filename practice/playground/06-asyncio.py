"""asyncio：100 个并发请求与限流"""
# Playground 支持顶层 await：浏览器里的事件循环已经在运行，直接 await 就行（不要用 asyncio.run）。
import asyncio
import time


async def call_model(i, sem):
    async with sem:                                  # 同时最多 LIMIT 个请求在"推理"
        await asyncio.sleep(0.05)                    # 假装一次请求要 50 毫秒
        return i * i


for limit in (5, 20, 100):
    sem = asyncio.Semaphore(limit)
    start = time.perf_counter()
    results = await asyncio.gather(*(call_model(i, sem) for i in range(100)))
    print(f"并发上限 {limit:>3}：100 个请求用时 {time.perf_counter() - start:.2f} 秒，结果之和 {sum(results)}")
