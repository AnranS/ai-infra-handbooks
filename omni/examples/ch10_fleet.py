"""两个假 worker + 真的 Rust router：看轮询、健康检查，以及 worker 挂掉的那一刻会发生什么。"""
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROUTER = os.path.join(os.environ["CARGO_TARGET_DIR"], "debug", "sgl-omni-router")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_worker(name: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            msg = json.loads(body)["messages"][0]["content"]
            out = json.dumps({"worker": name, "echo": msg}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def worker_toml(name: str, port: int) -> str:
    return f'''
[[workers]]
worker_id = "{name}"
base_url = "http://127.0.0.1:{port}/"
trust_domain = "local"
default_model_id = "toy-model"
health_path = "/health"

[[workers.service_profiles]]
service = "generation_http"
model_ids = ["toy-model"]
message_content_forms = ["string"]
media_placements = ["top_level"]
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming"]
'''


def chat(router_port: int, text: str) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{router_port}/v1/chat/completions",
        data=json.dumps({"model": "toy-model", "messages": [{"role": "user", "content": text}]}).encode(),
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())["worker"]
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {json.loads(exc.read()).get('error', {}).get('code')}"


def wait_ready(port: int) -> None:
    for _ in range(200):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.05)
    raise TimeoutError("router not ready")


ports = {"w1": free_port(), "w2": free_port()}
servers = {name: fake_worker(name, port) for name, port in ports.items()}
router_port = free_port()
config = f'''schema_version = 1
[server]
listen = "127.0.0.1:{router_port}"
max_connections = 64
[shutdown]
drain_timeout_ms = 1000
[logging]
format = "compact"
filter = "error"
[router]
strategy = "round_robin"
[admission]
global = 16
generation_http = 16
[health]
interval_ms = 1000
timeout_ms = 500
success_threshold = 1
failure_threshold = 1
[http]
buffered_request_total_bytes = 16777216
connect_timeout_ms = 1000
pool_idle_timeout_ms = 90000
pool_max_idle_per_host = 8
[http_generation]
trust_domain = "local"
buffered_request_max_bytes = 1048576
streamed_request_max_bytes = 1048576
request_timeout_ms = 10000
''' + "".join(worker_toml(n, p) for n, p in ports.items())
with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
    f.write(config)
router = subprocess.Popen([ROUTER, "--config", f.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    wait_ready(router_port)
    print("轮询四次：", [chat(router_port, f"q{i}") for i in range(4)])
    servers["w2"].shutdown()
    servers["w2"].server_close()                      # w2 下线，但 router 还不知道
    print("w2 刚下线时：", [chat(router_port, f"q{i}") for i in range(4, 8)])
    time.sleep(2.5)                                   # 等健康检查把 w2 标成不健康
    print("健康检查之后：", [chat(router_port, f"q{i}") for i in range(8, 12)])
    with urllib.request.urlopen(f"http://127.0.0.1:{router_port}/diagnostics") as resp:
        diag = json.loads(resp.read())
    print("diagnostics 里的 worker 健康：", {w["worker_id"]: w["health"] for w in diag["workers"]})
finally:
    router.terminate()
    router.wait(timeout=30)
