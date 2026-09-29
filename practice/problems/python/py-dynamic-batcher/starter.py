import asyncio


class DynamicBatcher:
    def __init__(self, model, max_batch_size: int, max_wait: float):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait
        self.batch_sizes = []

    async def submit(self, item):
        # 没有攒批：每个请求单独处理
        self.batch_sizes.append(1)
        return (await self.model([item]))[0]

    async def close(self):
        pass
