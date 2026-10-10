"""阶段三的里程碑：阶段二的 LLM.generate 变成了一个多进程的 OpenAI 兼容服务。

起服务、看进程、一次非流式、一次流式（量首 token 延迟）、8 个并发，最后打印一条
curl——这一页跑完，任何 OpenAI 客户端都能连上你写的引擎。
"""

import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import psutil
from openai import OpenAI

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
proc = subprocess.Popen(
    [sys.executable, "-m", "minisgl", "--model", "models/Qwen3-0.6B", "--dtype", "float32",
     "--port", str(port), "--num-pages", "4096", "--max-seq-len-override", "1024"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
try:
    t = time.time()
    while True:                                                   # 轮询 /v1/models，直到服务起来
        try:
            if httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    workers = [c for c in psutil.Process(proc.pid).children(recursive=True)
               if "multiprocessing" in " ".join(c.cmdline())]
    print(f"服务就绪（{time.time() - t:.0f} 秒）：1 个 API Server 进程 + {len(workers)} 个工作进程（调度器、tokenizer），用 ZMQ 传消息")
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")

    r = client.chat.completions.create(
        model="qwen3", messages=[{"role": "user", "content": "The capital of France is? Answer in one word. /no_think"}],
        max_tokens=16, temperature=0)
    print(f"非流式：{r.choices[0].message.content!r}")

    t0 = time.perf_counter()
    first = None
    chunks = []
    for c in client.chat.completions.create(
            model="qwen3", messages=[{"role": "user", "content": "Count from 1 to 10. /no_think"}],
            max_tokens=32, temperature=0, stream=True):
        if c.choices[0].delta.content:
            first = first or time.perf_counter() - t0
            chunks.append(c.choices[0].delta.content)
    print(f"流式：{len(chunks)} 个片段，首个片段 {first * 1000:.0f} ms 后到达，全部 {(time.perf_counter() - t0) * 1000:.0f} ms")

    def ask(i: int) -> float:
        t0 = time.perf_counter()
        client.chat.completions.create(model="qwen3", messages=[{"role": "user", "content": f"{i}+{i}=?"}],
                                       max_tokens=16, temperature=0, extra_body={"ignore_eos": True})
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        lat = list(ex.map(ask, range(8)))
    total = time.perf_counter() - t0
    print(f"8 个并发请求 x 16 个 token：总用时 {total:.2f} s（{8 * 16 / total:.1f} tokens/s），"
          f"每个请求 {min(lat):.2f}～{max(lat):.2f} s——它们在同一个 batch 里，没有排队")
    print("自己试：curl http://127.0.0.1:端口/v1/chat/completions -H 'Content-Type: application/json' "
          "-d '{\"model\":\"qwen3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'")
finally:
    os.killpg(proc.pid, signal.SIGKILL)
