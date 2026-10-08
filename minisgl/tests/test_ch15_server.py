import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from openai import OpenAI

from conftest import QWEN3, ROOT, free_port, hf_greedy


def start_server(*extra: str, log=subprocess.DEVNULL):
    port = free_port()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
    proc = subprocess.Popen([sys.executable, "-m", "minisgl", "--model", QWEN3, "--dtype", "float32",
                             "--port", str(port), "--num-pages", "4096",
                             "--max-seq-len-override", "1024",
                             "--attention-backend", "torch",    # FlashInfer 没有 fp32 的注意力 kernel
                             *extra],
                            env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1).status_code == 200:
                return proc, port
        except httpx.HTTPError:
            time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGKILL)
    raise RuntimeError("server did not start")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    with open(log_path, "w") as log:
        proc, port = start_server(log=log)
        yield port, log_path
    os.killpg(proc.pid, signal.SIGKILL)


def test_openai_non_stream_matches_hf(server):
    from transformers import AutoTokenizer

    port, _ = server
    prompt = "The capital of France is"
    resp = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"model": "m", "prompt": prompt, "max_tokens": 12, "temperature": 0,
                            "ignore_eos": True}, timeout=120).json()
    ref = AutoTokenizer.from_pretrained(QWEN3).decode(hf_greedy(QWEN3, [prompt], 12)[0])
    assert resp["choices"][0]["message"]["content"] == ref


def test_openai_streaming_and_concurrency(server):
    port, _ = server
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")
    stream = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "Say hi."}], max_tokens=8, temperature=0,
        stream=True, extra_body={"ignore_eos": True})
    chunks = [c.choices[0].delta.content for c in stream if c.choices[0].delta.content]
    assert len(chunks) >= 4  # 流式：多个增量片段

    def ask(i):
        return client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": f"What is {i} + {i}?"}],
            max_tokens=6, temperature=0).choices[0].message.content

    with ThreadPoolExecutor(6) as ex:
        answers = list(ex.map(ask, range(6)))
    assert all(answers)


def test_client_disconnect_aborts_request(server):
    """客户端中途断开：服务端中止请求并释放资源，之后的请求照常工作。"""
    port, log_path = server
    with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"model": "m", "prompt": "Tell me a long story.", "max_tokens": 500,
                            "stream": True, "ignore_eos": True}, timeout=60) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i >= 3:
                break  # 读了几段就断开
    time.sleep(1.0)
    resp = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"model": "m", "prompt": "1 + 1 =", "max_tokens": 4}, timeout=60)
    assert resp.status_code == 200
    assert "Aborting request" in log_path.read_text()
