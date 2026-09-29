import asyncio

_STOP = object()


class DynamicBatcher:
    def __init__(self, model, max_batch_size: int, max_wait: float):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait
        self.batch_sizes = []
        self._queue = None
        self._task = None
        self._closed = False

    def _ensure_started(self):
        if self._task is None:
            self._queue = asyncio.Queue()
            self._task = asyncio.ensure_future(self._loop())

    async def submit(self, item):
        if self._closed:
            raise RuntimeError("批处理器已经关闭")
        self._ensure_started()
        fut = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((item, fut))
        return await fut

    async def _run_batch(self, batch):
        self.batch_sizes.append(len(batch))
        try:
            outputs = await self.model([item for item, _ in batch])
            if len(outputs) != len(batch):
                raise ValueError(f"model 返回了 {len(outputs)} 个结果，期望 {len(batch)} 个")
        except Exception as e:  # noqa: BLE001
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
            return
        for (_, fut), out in zip(batch, outputs):
            if not fut.done():
                fut.set_result(out)

    async def _loop(self):
        loop = asyncio.get_running_loop()
        stopping = False
        while not stopping:
            first = await self._queue.get()
            if first is _STOP:
                return
            batch = [first]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout)
                except asyncio.TimeoutError:
                    break
                if nxt is _STOP:
                    stopping = True
                    break
                batch.append(nxt)
            await self._run_batch(batch)

    async def close(self):
        self._closed = True
        if self._task is not None:
            self._queue.put_nowait(_STOP)
            await self._task
