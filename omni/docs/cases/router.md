# 案例三：Rust router——多副本前面的那扇门

<p class="lead">一个 omni 服务进程就是一条完整的流水线。要更大的吞吐，就起多个副本，前面放一个路由器：它对外是一个 OpenAI 兼容的地址，对内把请求分给健康的、能处理这类请求的副本，并且控制住同时在途的请求数。omni 的路由器最早用 Python（FastAPI + httpx）写，2026 年 8 月开始用 Rust 重写成一个独立的进程 <code>sgl-omni-router</code>，到基准提交已经有两万七千行。这一章读它的结构和一条请求的路径，然后在开发机上把它真的跑起来：两个假 worker，一个真 router，看轮询、看健康检查，再看一个 worker 突然下线的那一刻发生了什么——那一刻正是第十二章要做的那个 PR 的起点。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么要用 Rust 重写一个"只是转发请求"的路由器？
    2. router 怎么知道一个 worker 能处理某个请求？"直接路径"和"分类路径"有什么区别？
    3. `round_robin` 在 router 里是怎么实现的？为什么不是简单的取模？
    4. 一个 worker 突然下线，健康检查还没发现，这时恰好分给它的请求会怎样？
    5. 给这个 router 提 PR，CI 至少要过哪几道检查？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 路由器的成本是"每个请求固定的那一份"：Python 版里大约四分之三是 asyncio、httpx、FastAPI 的开销，不管请求带 4 KB 还是 4 MB 都要付。分成多个进程只是把同样昂贵的单位复制 N 份。Rust 版（tokio + hyper + reqwest）直接把每请求的成本降下来（RFC #1623 的论证）。
    2. 每个 worker 在配置里声明 service profile（服务类型、模型、输入输出模态、流式模式、音频格式……）。如果同一信任域里所有合格副本的默认模型和 profile 都一样（同质），走直接路径：不读请求体，边收边转发；否则走分类路径：在有界的字节预算里把请求体读一遍，解析出模型、模态、格式等事实，再按事实挑副本，原始字节原样转发。
    3. 一个"最近使用顺序"数组：从前往后找第一个合格的 worker，选中后把它挪到所有其他 worker 的后面。这样候选子集变化时（某个 worker 不健康了）不会和取模的相位纠缠在一起。
    4. 连接被拒绝，请求得到 `502 upstream_protocol_error`，同时触发一次立即健康探测，探测失败后这个 worker 被标成不健康，后续请求不再分给它。当前的 router 不重试——这正是 RFC #1623 里"连接失败时带抖动重试"那一项要解决的问题。
    5. `cargo fmt --check`、MSRV（1.90）上的 `cargo check`、`cargo clippy -D warnings`、`cargo test`、`cargo doc -D warnings`、release 构建，以及用 release 二进制校验所有示例配置。

## 为什么要一个独立的 Rust 进程

RFC #1623（"Rust Router for SGLang-Omni — Phased Roadmap"）的核心论证是：Python 路由器先做了"不缓冲响应"（Phase 0），再拆成一个控制面加 N 个无状态的转发进程（Phase 1），两步都有帮助，但都没碰到真正决定上限的东西——**每个请求转发一次的固定成本**。于是数据面先换成 Rust，控制面以后再说。

crate 的模块文档一句话概括了它的职责：

```rust title="sglang_omni_router/rust/src/lib.rs @ 921ea2c8 L1-5"
//! Standalone SGLang-Omni Rust router.
//!
//! This crate owns strict startup configuration, a static worker pool, bounded
//! routing and health, byte-preserving chat and media HTTP relays, route-aware
//! readiness, and joined process shutdown.
```

| 路径 | 职责 |
| --- | --- |
| `config.rs` | 严格的 TOML 配置和跨字段校验 |
| `server.rs` | 组装运行时、路由表、监听器、关停 |
| `worker_pool/` | 准入、健康检查、profile、选择策略、worker 负载 |
| `http_relay/` | 共享的 HTTP 客户端、缓冲、请求 / 响应体适配、转发 |
| `http_generation/` | 聊天请求的校验和分类 |
| `http_media/` | 语音、批量语音、转录、翻译、音色上传 |
| `websocket/` | 语音和 realtime 会话的建立与转发 |
| `operations.rs` | `/v1/models`、`/metrics`、`/diagnostics` |

![图：Rust router 里一条请求的路径](../assets/figures/omni-router.svg){.aig-svg}

依赖都锁死了确切版本（`tokio = "=1.51.4"`、`axum = "=0.8.9"`……），`Cargo.toml` 的 lint 段禁止 `unwrap`、`expect`、`panic`、`todo`、`unsafe`——这是一个"不允许在请求路径上崩溃"的服务该有的样子。

## 一条请求的路径

### 直接路径和分类路径

```text title="docs/basic_usage/omni_router.md @ 921ea2c8 L199-216"
### Request paths

The direct path is available when every eligible replica in a trust-scoped
cohort has the same concrete default model and compatible profile contract.
Heterogeneous pools use the classified path. The optional
`x-sglang-omni-route-model` and `x-sglang-omni-route-stream` headers are checked
against the classified body and are not forwarded upstream. The router relays
a direct request body as a backpressured stream.

Requests that require body-owned routing facts reserve aggregate byte capacity,
read the body once, and classify the model, content forms, media placement,
input and output modalities, response format, and stream mode. Classification
runs on Tokio's blocking pool with execution limited to the available CPU
parallelism, while the aggregate buffered-byte budget bounds concurrent
classifier memory. The original bytes are forwarded without reconstructing
JSON or multipart content.

The direct path is bounded by each route's `streamed_request_max_bytes`. The
```

同质的副本池（最常见的情况：N 个一模一样的 omni 进程）不需要读请求体，直接以流的方式转发；异构的池子（比如有的副本能出音频、有的不能）才需要先读一遍请求体判断"这个请求需要什么"。分类在 tokio 的阻塞线程池里做，并发数和缓冲的总字节数都有上限——**路由器不能因为分类而无限占内存**。

### 选择

`round_robin` 的实现只有十几行：

```rust title="sglang_omni_router/rust/src/worker_pool/selection.rs @ 921ea2c8 L49-62"
impl SelectorGuard<'_> {
    pub(super) fn select(&mut self, candidates: &[bool; MAX_WORKERS]) -> Option<usize> {
        let position = self.recency.ordinals[..self.recency.worker_count]
            .iter()
            .position(|ordinal| candidates[*ordinal])?;
        let selected = self.recency.ordinals[position];

        // Moving the selected ordinal behind every other worker prevents
        // changing candidate subsets from sharing a modulo phase.
        let worker_count = self.recency.worker_count;
        self.recency.ordinals[position..worker_count].rotate_left(1);
        Some(selected)
    }
}
```

`ordinals` 是一个"最近使用顺序"：从前往后找第一个合格的（健康且 profile 兼容的）worker，选中后把它挪到所有其他 worker 后面。注释说明了为什么不用取模：候选子集会变（有 worker 不健康了、某类请求只有部分 worker 能处理），取模的相位会和子集的变化纠缠在一起，导致某些 worker 被连续选中。`least_requests` 比较在途请求数，平局时用同样的顺序轮转。

### 准入与背压

```text title="docs/basic_usage/omni_router.md @ 921ea2c8 L238-254"
### Admission and backpressure

Global admission counts request envelopes. Per-service admission counts
requests or sessions, except speech-batch admission, which counts batch items.
Admission is fail-fast, and its permits and worker-load guard remain held until
response EOF, upstream error, or downstream cancellation. The worker remains
responsible for its execution and queue capacity.

The router sends one upstream request through a shared HTTP/1.1 connection
pool. Redirects, ambient proxies, retries, and automatic decompression are
disabled. Request and response bodies use direct backpressure without a body
pump, application queue, or extra relay task.

The request deadline covers upload, connection establishment, and upstream
response headers. A final worker response received before the upload completes
is rejected as an upstream protocol error and is not committed downstream.
After response headers are committed, the response has no total wall-clock
```

准入是"快速失败"：全局一个信号量、每类服务一个信号量，拿不到就立刻拒绝，而不是排队；许可一直持有到响应结束（或上游出错、客户端断开）。注意倒数第二段：**重定向、环境代理、重试、自动解压都被关掉了**——这是"当前行为"，下一节的实验会看到它的后果。

### 转发：一次 send，错误怎么归类

```rust title="sglang_omni_router/rust/src/http_relay/mod.rs @ 921ea2c8 L188-209"
        let sent = tokio::select! {
            biased;
            result = request.send() => result,
            () = tokio::time::sleep_until(deadline) => {
                let fault = deadline_fault(outgoing.upload.as_ref());
                if fault == HttpFault::UpstreamTimeout {
                    lease.request_immediate_probe();
                }
                return Err(fault);
            }
        };
        let response = match sent {
            Ok(response) => response,
            Err(_source) => {
                let fault = upload_fault(outgoing.upload.as_ref())?
                    .unwrap_or(HttpFault::UpstreamProtocolError);
                if fault == HttpFault::UpstreamProtocolError {
                    lease.request_immediate_probe();
                }
                return Err(fault);
            }
        };
```

`request.send()` 和请求的截止时间赛跑：超时 → `UpstreamTimeout`（504）并请求一次立即探测；发送失败 → 默认归成 `UpstreamProtocolError`（502），同样请求立即探测。注意 `Err(_source)`：底层的 reqwest 错误被丢掉了——连接被拒绝（请求根本没发出去）和"连上了但对方中途断开"（请求可能已经被处理了一半）在这里没有区分。错误到状态码的映射：

```rust title="sglang_omni_router/rust/src/error.rs @ 921ea2c8 L190-197"
            Self::ExpectationFailed => StatusCode::EXPECTATION_FAILED,
            Self::NoCompatibleWorker => StatusCode::UNPROCESSABLE_ENTITY,
            Self::RouterOverloaded => StatusCode::TOO_MANY_REQUESTS,
            Self::InternalError => StatusCode::INTERNAL_SERVER_ERROR,
            Self::UpstreamProtocolError => StatusCode::BAD_GATEWAY,
            Self::RouterUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::UpstreamTimeout => StatusCode::GATEWAY_TIMEOUT,
            Self::HttpVersionNotSupported => StatusCode::HTTP_VERSION_NOT_SUPPORTED,
```

### 健康

```text title="docs/basic_usage/omni_router.md @ 921ea2c8 L298-307"
## Health and Readiness

Workers start with unknown health. Each worker has one serial probe loop that
applies the configured consecutive success and failure thresholds.
Transport and upstream protocol failures can request an immediate coalesced
probe. Application responses do not directly change worker health.

`GET /ready` returns `200` while the process is serving and every enabled
generation, media, and WebSocket service has a compatible healthy worker.
Readiness also requires the configured uploaded-voice owner to be healthy and
```

每个 worker 一个串行的探测循环，连续成功 / 失败若干次才切换状态；传输层和协议层的失败可以请求一次"立即探测"（多个请求同时失败时合并成一次）。

## 动手：两个假 worker，一个真 router

本书的环境里装了 Rust 工具链（版本由仓库的 `rust-toolchain.toml` 锁定），先构建 router：

```bash title="ch10_build.sh"
cd "$OMNI_TREE/sglang_omni_router/rust"
CARGO=$(command -v cargo || echo "$HOME/.cargo/bin/cargo")
"$CARGO" build --locked -q 2>/dev/null
"$CARGO_TARGET_DIR/debug/sgl-omni-router" --version
```

```text title="输出"
sgl-omni-router 0.1.0
```

再用 Python 写两个假 worker（`/health` 返回 200，`/v1/chat/completions` 把自己的名字写进响应），生成一份配置，启动 router：

```python title="ch10_fleet.py" ci="loose"
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
```

```text title="输出"
轮询四次： ['w1', 'w2', 'w1', 'w2']
w2 刚下线时： ['w1', 'HTTP 502 upstream_protocol_error', 'w1', 'w1']
健康检查之后： ['w1', 'w1', 'w1', 'w1']
diagnostics 里的 worker 健康： {'w1': 'healthy', 'w2': 'unhealthy'}
```

逐行读：

1. **轮询**：两个健康的 worker 交替被选中；
2. **w2 刚下线**：router 还认为 w2 健康，下一个轮到 w2 的请求连接被拒绝——用户拿到了一个 `502 upstream_protocol_error`。这个请求其实**根本没有到达任何 worker**，换一个 worker 完全可以成功；但 router 不重试。同时它触发了一次立即探测，w2 很快被标成不健康，后面两个请求都去了 w1；
3. **健康检查之后**：全部去 w1，`/diagnostics` 里 w2 是 `unhealthy`。

第 2 行就是 RFC #1623 第一阶段清单里"Retry with jitter on upstream connect failure (new)"要解决的问题：对"连接没有建立"这一类失败，换一个健康的 worker 重发是安全的（一个字节都没发出去，不存在重复执行的风险）。在 worker 滚动重启、扩缩容、某个副本崩溃时，这能把一批 502 变成正常的响应。第十二章会从这里出发，走完"读代码 → 定范围 → 写测试 → 写 PR"的全过程。

## 测试：真实的 TCP

router 的单测写在实现旁边，集成测试在 `tests/` 下，用真实的回环 socket 起假 worker：

```bash title="ch10_tests.sh"
cd "$OMNI_TREE/sglang_omni_router/rust"
CARGO=$(command -v cargo || echo "$HOME/.cargo/bin/cargo")
"$CARGO" test --locked -q 2>/dev/null | grep -E '^test result' | awk '{p += $4; f += $6} END {printf "通过 %d，失败 %d\n", p, f}'
echo "集成测试文件："
ls tests/ | sed 's/^/  /'
```

```text title="输出"
通过 243，失败 0
集成测试文件：
  chat_http.rs
  config.rs
  media_http.rs
  process.rs
  voice_state.rs
  websocket.rs
```

官方文档列出了 CI 用的全部质量门禁：

````text title="docs/basic_usage/omni_router.md @ 921ea2c8 L415-429"
### Quality gates

Run the same checks used by CI:

```console
cargo fmt --all -- --check
cargo +1.90.0 check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked -- --test-threads=1
RUSTDOCFLAGS="-D warnings" \
  cargo doc --workspace --all-features --no-deps --locked
cargo build --release --workspace --all-features --locked
```

Validate all tracked deployment examples with the release binary:
````

提 PR 之前，这几条要在本地全部跑过——包括用最低支持版本 1.90 做一次 `cargo check`（`rust-toolchain.toml` 平时用的是 1.97.1）。

## 练习

**1. 换成 `least_requests`。** 把实验里的策略改成 `least_requests`，并让 w1 的处理慢一点（`time.sleep(0.5)`），并发发送 6 个请求，观察分配。

??? success "参考思路"
    用 `concurrent.futures.ThreadPoolExecutor` 并发调用 `chat`。`least_requests` 选在途请求最少的 worker，w1 慢、在途数高，更多请求会被分给 w2；`round_robin` 则不管负载严格交替。这就是文档说"按目标并发下的完整语料测量来选策略"的原因。

**2. 看一眼 `/metrics`。** 在实验里 w2 下线之后，抓 `/metrics`，找出和 router 生成的错误、worker 探测结果相关的指标。

??? success "参考思路"
    `urllib.request.urlopen(f"http://127.0.0.1:{router_port}/metrics").read().decode()`，grep `fault`、`probe`。标签用固定词表（错误种类、探测结果），不带 worker id、模型名这类高基数的值——文档专门强调过这一点。

**3. 区分两种失败。** 在 `http_relay/mod.rs` 的 `send` 里，`Err(_source)` 的 `_source` 是 `reqwest::Error`。查 reqwest 的文档，找出能区分"连接失败"和"其他错误"的方法，并说明为什么这个区分是安全重试的前提。

??? success "参考思路"
    `reqwest::Error::is_connect()`：连接阶段失败（拒绝连接、DNS、TLS 握手等），请求体一个字节都没发出去。其他错误（`is_request`、`is_body`、读响应时断开）可能发生在 worker 已经开始处理之后，对生成类的 POST 重发会导致重复执行。第十二章的重试只对 `is_connect()` 生效。

!!! interview "怎么讲清楚"
    讲多副本推理服务的路由器，先讲为什么独立成一个 Rust 进程：瓶颈是每请求的固定转发成本，Python 拆多进程只是复制成本。再讲路径：同质池直接流式转发，异构池在有界预算里读一次请求体做分类；选择策略是"最近使用顺序"的轮询或最少在途；准入是全局 + 每类服务的信号量，快速失败，许可持有到响应结束；健康是每 worker 一个串行探测循环，失败可以触发立即探测。最后讲一个真实的缺口：连接失败目前直接 502，对"没发出去的请求"换一个 worker 重试是安全的，这是路线图上的一项。

## 小结

- [x] Rust router 是独立进程（tokio + hyper + reqwest，依赖锁死、禁止 unwrap / panic），动机是降低每请求的固定转发成本（RFC #1623）。
- [x] 同质池走直接路径（流式转发、不读请求体），异构池走分类路径（有界预算内读一次、原样转发）。
- [x] 选择：`round_robin` 是"最近使用顺序"，`least_requests` 比在途数；准入是信号量快速失败；健康是串行探测 + 立即探测。
- [x] 实验：worker 刚下线时，分给它的请求得到 `502 upstream_protocol_error`，虽然请求从未到达任何 worker——连接失败重试的动机。
- [x] 质量门禁：fmt、MSRV check、clippy `-D warnings`、test、doc、release 构建、校验示例配置。
