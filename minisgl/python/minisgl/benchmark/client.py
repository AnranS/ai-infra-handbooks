"""在线压测客户端：按泊松过程（或一次性）发请求，流式接收，统计 TTFT、TPOT、端到端延迟。

    python -m minisgl.benchmark.client --port 1919 --num-requests 64 --rate 4

TTFT：发出请求到收到第一个 token 的时间；TPOT：之后平均每个 token 的间隔；
所有指标都给出均值、中位数和 P99。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import httpx


@dataclass
class RequestResult:
    ttft: float = 0.0
    latency: float = 0.0
    chunk_times: List[float] = field(default_factory=list)
    output_chars: int = 0

    @property
    def tpot(self) -> float:
        if len(self.chunk_times) < 2:
            return 0.0
        return (self.chunk_times[-1] - self.chunk_times[0]) / (len(self.chunk_times) - 1)


async def one_request(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int) -> RequestResult:
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    res = RequestResult()
    start = time.perf_counter()
    async with client.stream("POST", url, json=body) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                now = time.perf_counter() - start
                if not res.chunk_times:
                    res.ttft = now
                res.chunk_times.append(now)
                res.output_chars += len(delta)
    res.latency = time.perf_counter() - start
    return res


def _summary(name: str, xs: List[float]) -> str:
    xs = sorted(xs)
    p99 = xs[min(len(xs) - 1, int(0.99 * len(xs)))]
    return (f"{name:<8} mean {statistics.mean(xs) * 1e3:8.1f} ms   median "
            f"{statistics.median(xs) * 1e3:8.1f} ms   P99 {p99 * 1e3:8.1f} ms")


async def run(host: str, port: int, num_requests: int, rate: float, max_tokens: int) -> List[RequestResult]:
    url = f"http://{host}:{port}/v1/chat/completions"
    rng = random.Random(0)
    prompts = [f"Request {i}: write a story about the number {rng.randint(0, 999)}."
               for i in range(num_requests)]
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = []
        for p in prompts:
            tasks.append(asyncio.create_task(one_request(client, url, p, max_tokens)))
            if rate > 0:  # 泊松过程：相邻请求的间隔服从指数分布
                await asyncio.sleep(rng.expovariate(rate))
        return await asyncio.gather(*tasks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--rate", type=float, default=0.0, help="每秒请求数，0 表示一次全部发出")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()
    t = time.perf_counter()
    results = asyncio.run(run(args.host, args.port, args.num_requests, args.rate, args.max_tokens))
    t = time.perf_counter() - t
    print(f"{len(results)} requests in {t:.2f}s")
    print(_summary("TTFT", [r.ttft for r in results]))
    print(_summary("TPOT", [r.tpot for r in results]))
    print(_summary("E2E", [r.latency for r in results]))


if __name__ == "__main__":
    main()
