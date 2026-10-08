import asyncio
import time


async def handle(i):
    await asyncio.sleep(0.1)                        # 等待期间让出事件循环，别的协程接着跑
    return i


async def main():
    loop = asyncio.get_running_loop()
    print("事件循环底下的选择器：", type(loop._selector).__name__)
    t = time.perf_counter()
    done = await asyncio.gather(*(handle(i) for i in range(1000)))
    print("1000 个各等 0.1 秒的请求，总共不到 0.5 秒：", len(done) == 1000 and time.perf_counter() - t < 0.5)

    t = time.perf_counter()
    blocker = loop.run_in_executor(None, time.sleep, 0.3)   # 阻塞的工作放进线程池
    await asyncio.gather(blocker, *(handle(i) for i in range(10)))
    print("阻塞调用放进线程池后，其他请求照常进行：", time.perf_counter() - t < 0.45)

    t = time.perf_counter()

    async def bad():
        time.sleep(0.3)                             # 在协程里直接做阻塞调用：整个事件循环停住

    await asyncio.gather(bad(), *(handle(i) for i in range(10)))
    print("协程里直接阻塞 0.3 秒，其他请求被拖到 0.4 秒以后：", time.perf_counter() - t >= 0.4)


asyncio.run(main())
