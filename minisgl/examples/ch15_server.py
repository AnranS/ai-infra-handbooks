"""第 15 章：启动完整的服务，用 OpenAI 客户端访问。"""

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
    while True:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    children = psutil.Process(proc.pid).children(recursive=True)
    workers = [c for c in children if "multiprocessing-fork" in " ".join(c.cmdline())]
    print(f"服务就绪（约 {time.time() - t:.0f} 秒）：主进程（API Server）+ {len(workers)} 个工作进程"
          f"（调度器、tokenizer）")
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")

    r = client.chat.completions.create(
        model="qwen3", messages=[{"role": "user", "content": "用一句话介绍北京。/no_think"}],
        max_tokens=48, temperature=0)
    print("非流式:", repr(r.choices[0].message.content))

    stream = client.chat.completions.create(
        model="qwen3", messages=[{"role": "user", "content": "Count from 1 to 5. /no_think"}],
        max_tokens=24, temperature=0, stream=True)
    chunks = [c.choices[0].delta.content for c in stream if c.choices[0].delta.content]
    print(f"流式: 收到 {len(chunks)} 个片段:", chunks[:12], "...")

    def ask(i: int) -> float:
        t0 = time.perf_counter()
        client.chat.completions.create(model="qwen3", messages=[{"role": "user", "content": f"{i}+{i}=?"}],
                                       max_tokens=16, temperature=0, extra_body={"ignore_eos": True})
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        lat = list(ex.map(ask, range(8)))
    print(f"8 个并发请求（各 16 个 token）总用时 {time.perf_counter() - t0:.2f}s，单个 {min(lat):.2f}～{max(lat):.2f}s")
finally:
    os.killpg(proc.pid, signal.SIGKILL)
