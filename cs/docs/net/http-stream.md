# HTTP 与流式输出：一个 token 怎么到达浏览器

<p class="lead">推理服务对外的样子就是一个 HTTP 接口：`POST /v1/chat/completions`，`stream=true`，然后 token 一个一个蹦出来。这套机制底下是 HTTP 的分块传输和 SSE；中间只要有一个环节做了缓冲，"逐字输出"就会变成"等半天一次性吐出来"。这一章从零写一个 SSE 服务端和客户端，看清每个 token 是怎么变成网络上的字节的，再看代理缓冲、HTTP/1.1 的队头阻塞、HTTP/2 与 gRPC 的区别，以及流式接口该怎么处理断线、取消和超时。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 服务端在响应还没生成完的时候就开始发送，HTTP 协议上靠什么实现？
    2. SSE 和 WebSocket 有什么区别？为什么 OpenAI 风格的流式接口用 SSE？
    3. 服务端明明每 20 ms 发一个 token，用户却看到"卡住几秒、然后一次性出现"。可能是哪几个环节的问题？
    4. HTTP/1.1 的队头阻塞是什么？浏览器和 HTTP/2 各自怎么绕开它？
    5. 用户关掉页面之后，服务端怎么知道该停止生成、释放 KV Cache？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 分块传输编码（`Transfer-Encoding: chunked`）：响应头里不写 `Content-Length`，之后每次发一个"十六进制长度 + 数据"的块，最后发一个长度为 0 的块表示结束。HTTP/2 不需要它，因为帧本身就带长度。
    2. SSE 是单向的（服务端到客户端）、基于普通的 HTTP 响应，浏览器有原生的 `EventSource`，能自动重连，穿过代理和防火墙不需要协议升级；WebSocket 是全双工的独立协议，要先 `Upgrade` 握手。生成式接口只需要服务端往下推文本，SSE 足够，并且和现有的 HTTP 基础设施（鉴权、负载均衡、限流）兼容。
    3. 常见的三处：（1）服务端没开 `TCP_NODELAY` 或没刷新缓冲区，token 被攒起来；（2）中间的代理（Nginx 的 `proxy_buffering`、CDN、API 网关）缓冲了响应；（3）客户端框架把响应当成完整的 body 来读。本章的实验里，一个攒够 4 KB 才转发的代理，把首个事件从 0.3 ms 拖到了 405 ms。
    4. HTTP/1.1 一条连接上同一时刻只能有一个请求在传（即使用流水线，响应也必须按序返回），前面的慢请求会堵住后面的。浏览器的办法是对同一个域名开 6 条连接；HTTP/2 的办法是把请求拆成带流 ID 的帧，在一条连接上交错传输。
    5. 客户端断开后，服务端向这条连接写数据会失败（`EPIPE` / `ECONNRESET`），或者框架把请求标记为已断开（ASGI 里的 `http.disconnect`、`await request.is_disconnected()`）。推理引擎要把这个信号一路传到调度器：中止这个请求的生成、释放它占的 KV 块。不处理的话，用户关掉页面后 GPU 还在为它算，浪费的是最贵的资源。

## 从一个 token 到一段 HTTP 响应

![图：流式输出的链路和三个常见的缓冲点](../assets/figures/sse-stream.svg){.aig-svg}

OpenAI 风格的流式响应长这样：响应头里声明 `Content-Type: text/event-stream`，然后每个 token 发一条 `data: {...}` 事件，最后一条是 `data: [DONE]`。下面这个程序把服务端和客户端都写出来，并记录每个事件到达的时刻：

```python title="sse.py"
# 一个最小的 SSE（Server-Sent Events）服务端 + 客户端：分块传输，每生成一个 token 就刷一次
import socket
import threading
import time

HOST, TOKENS = "127.0.0.1", 8
HEAD = (b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n")


def chunk(payload: bytes) -> bytes:
    return b"%x\r\n" % len(payload) + payload + b"\r\n"           # 分块传输：十六进制长度 + 数据


def serve(srv):
    conn, _ = srv.accept()
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 不开的话 token 会被 Nagle 攒起来
        while b"\r\n\r\n" not in conn.recv(4096):                   # 读完请求头
            pass
        conn.sendall(HEAD)
        time.sleep(0.05)                                            # 假装在做 prefill
        for i in range(TOKENS):
            conn.sendall(chunk(b'data: {"token": "t%d"}\n\n' % i))
            time.sleep(0.02)                                        # 每 20 ms 产出一个 token
        conn.sendall(chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")     # 长度为 0 的块表示结束


srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, 0))
srv.listen(1)
threading.Thread(target=serve, args=(srv,), daemon=True).start()

c = socket.create_connection(srv.getsockname())
c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
t0 = time.perf_counter()
c.sendall(b"GET /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAccept: text/event-stream\r\n\r\n")

buf, head_at, arrivals = b"", None, []
while b"\r\n\r\n" not in buf:                                       # 先收响应头
    buf += c.recv(4096)
head_at = time.perf_counter() - t0
buf = buf.split(b"\r\n\r\n", 1)[1]
while True:                                                         # 再按块收事件
    while b"\r\n" not in buf:
        buf += c.recv(4096)
    size_line, rest = buf.split(b"\r\n", 1)
    size = int(size_line, 16)
    if size == 0:
        break
    while len(rest) < size + 2:
        rest += c.recv(4096)
    arrivals.append((time.perf_counter() - t0, rest[:size]))
    buf = rest[size + 2:]
c.close()
srv.close()

print(f"响应头到达（可以开始渲染）：{head_at * 1e3:4.0f} ms")
for at, payload in arrivals[:3]:
    print(f"事件到达 {at * 1e3:4.0f} ms：{payload.strip().decode()}")
print(f"……最后一个事件：{arrivals[-1][0] * 1e3:.0f} ms，共 {len(arrivals)} 个事件，"
      f"平均间隔 {(arrivals[-1][0] - arrivals[0][0]) / (len(arrivals) - 1) * 1e3:.0f} ms")
```

```text title="输出（本机示例）"
响应头到达（可以开始渲染）：   0 ms
事件到达   50 ms：data: {"token": "t0"}
事件到达   71 ms：data: {"token": "t1"}
事件到达   91 ms：data: {"token": "t2"}
……最后一个事件：212 ms，共 9 个事件，平均间隔 20 ms
```

几处值得注意：

- **响应头先发**。服务端一确定要开始生成就把头发出去，浏览器据此进入"流式渲染"模式。这也是为什么 TTFT 里"看到第一个字"比"请求完成"重要得多。
- **每个事件一个块**。分块传输编码的每块是"十六进制长度 + `\r\n` + 数据 + `\r\n`"，最后一个长度为 0 的块表示结束。`data:` 后面是 SSE 的载荷，两个换行表示一条事件结束。
- **每块都要立刻发出去**。这里显式打开了 `TCP_NODELAY`；用框架时对应的是"每写一段就 flush"（Python 的 `StreamingResponse`、Go 的 `http.Flusher`）。忘了这一步，token 会被攒在一起（见上一章的 [Nagle 算法](tcp.md#nagle-算法与-40-毫秒)）。

SSE 的其他字段也有用：`event:` 给事件分类（比如把 `usage` 统计和正文分开）、`id:` 给事件编号，客户端断线重连时会带上 `Last-Event-ID`，服务端可以据此续传（推理服务一般不续传，直接重新生成）；`retry:` 告诉客户端重连前等多久。为了穿过那些会因为"长时间没有数据"就断开连接的中间设备，长时间没有 token 时要定期发一个注释行（`: keep-alive`）当心跳。

## 缓冲是流式输出的头号敌人

链路上任何一个环节攒数据，流式就没了。用一个最小的代理来演示：一个直通转发，一个攒够 4 KB 再转发。

```python title="proxy_buffer.py"
# 中间加一个代理：直通 vs 攒够 4 KB 再转发。后者会把流式输出变成"一次性吐出来"
import socket
import threading
import time

HOST, TOKENS, GAP = "127.0.0.1", 40, 0.01
TOKEN = b"data: " + b"x" * 40 + b"\n\n"                     # 每个事件约 50 字节


def upstream(srv):
    conn, _ = srv.accept()
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.recv(4096)
        for _ in range(TOKENS):
            conn.sendall(TOKEN)
            time.sleep(GAP)


def proxy(listen, upstream_addr, buffer_size):
    conn, _ = listen.accept()
    up = socket.create_connection(upstream_addr)
    up.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    up.sendall(conn.recv(4096))
    pending = b""
    while True:
        data = up.recv(4096)
        if not data:
            break
        pending += data
        if len(pending) >= buffer_size:                     # 攒够一个缓冲区才转发
            conn.sendall(pending)
            pending = b""
    if pending:
        conn.sendall(pending)
    conn.close(), up.close()


def run(buffer_size):
    up_srv = socket.socket()
    up_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    up_srv.bind((HOST, 0))
    up_srv.listen(1)
    px_srv = socket.socket()
    px_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    px_srv.bind((HOST, 0))
    px_srv.listen(1)
    threading.Thread(target=upstream, args=(up_srv,), daemon=True).start()
    threading.Thread(target=proxy, args=(px_srv, up_srv.getsockname(), buffer_size), daemon=True).start()
    c = socket.create_connection(px_srv.getsockname())
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t0 = time.perf_counter()
    c.sendall(b"GET / HTTP/1.1\r\n\r\n")
    first, got = None, 0
    while got < TOKENS * len(TOKEN):
        data = c.recv(65536)
        if not data:
            break
        first = first if first is not None else time.perf_counter() - t0
        got += len(data)
    last = time.perf_counter() - t0
    c.close(), up_srv.close(), px_srv.close()
    return first * 1e3, last * 1e3


for name, size in [("直通（收到就转发）", 1), ("攒够 4 KB 再转发", 4096)]:
    first, last = run(size)
    print(f"{name}：第一个事件 {first:6.1f} ms，全部收完 {last:6.1f} ms")
```

```text title="输出（本机示例）"
直通（收到就转发）：第一个事件    0.3 ms，全部收完  395.0 ms
攒够 4 KB 再转发：第一个事件  405.1 ms，全部收完  405.1 ms
```

直通时第一个事件立刻到达，总时长就是生成时长；攒够 4 KB 才转发时，用户在最后一刻才一次性收到全部内容——体验从"逐字出现"变成"转圈几百毫秒然后刷屏"，而总时长几乎没变。压测报表上的"平均延迟"也看不出差别，只有 TTFT 指标能抓到它。

实际部署里要逐个检查的环节：

| 环节 | 要做的事 |
| --- | --- |
| 应用框架 | 用流式响应对象，每写一段就 flush；不要用会缓冲的中间件（比如统一的响应压缩、JSON 包装） |
| Nginx / OpenResty | `proxy_buffering off;`、`proxy_cache off;`、`chunked_transfer_encoding on;`、`gzip off`（或对 `text/event-stream` 关掉压缩） |
| 云上的负载均衡与 API 网关 | 确认支持流式响应；有的网关默认会缓冲整个响应、或者有响应体大小上限 |
| CDN | 对流式接口直接绕开，或者确认它支持流式回源 |
| 客户端 | 用支持流式的读法（`fetch` 的 `ReadableStream`、`requests` 的 `stream=True`、`httpx` 的 `iter_lines`），不要 `response.json()` |

压缩要单独说一句：`gzip` 为了压缩率也会攒数据，对流式响应要么关掉，要么开启"每次刷新都结束一个压缩块"的模式。而 SSE 的文本本来就重复度高，网关那层的收益有限，实践中多数直接关掉。

## HTTP/1.1、HTTP/2 与 gRPC

一条 HTTP/1.1 连接同一时刻只能处理一个请求，前面的慢请求会堵住后面的——这就是**队头阻塞**：

```python title="multiplex.py"
# 同一条连接上并发发 N 个请求：HTTP/1.1 只能一个接一个（队头阻塞），HTTP/2 可以交错
def http11(reqs, conns):
    """conns 条连接，每条串行处理排给它的请求。reqs 是每个请求的服务端耗时（ms）"""
    queues = [0.0] * conns
    done = []
    for i, cost in enumerate(reqs):
        k = i % conns                          # 轮流放进各条连接
        queues[k] += cost
        done.append(queues[k])
    return done


def http2(reqs):
    """一条连接，所有请求同时开始（服务端并发处理），各自在自己的耗时后完成"""
    return list(reqs)


slow, fast = 500.0, 20.0
reqs = [slow] + [fast] * 5                     # 一个慢请求排在前面，后面五个很快
for name, done in [("HTTP/1.1，1 条连接", http11(reqs, 1)),
                   ("HTTP/1.1，6 条连接", http11(reqs, 6)),
                   ("HTTP/2，1 条连接", http2(reqs))]:
    print(f"{name:20s} 快请求的完成时间：{[round(t) for t in done[1:]]} ms，"
          f"最后一个 {max(done):.0f} ms")
```

```text title="输出"
HTTP/1.1，1 条连接       快请求的完成时间：[520, 540, 560, 580, 600] ms，最后一个 600 ms
HTTP/1.1，6 条连接       快请求的完成时间：[20, 20, 20, 20, 20] ms，最后一个 500 ms
HTTP/2，1 条连接         快请求的完成时间：[20, 20, 20, 20, 20] ms，最后一个 500 ms
```

浏览器的解法是对同一个域名开 6 条连接（也就是上面的第二行）；HTTP/2 的解法是把请求拆成带流 ID 的帧，在一条连接上交错传输，一条连接就能并发很多个请求。

对推理服务来说：

- **面向浏览器的接口**用 HTTP/1.1 + SSE 就够：一个对话就是一个长响应，浏览器不会在同一条连接上再塞别的请求。
- **服务之间**（网关到推理实例、推理实例之间）用 HTTP/2 或 gRPC 更合适：一条长连接上并发很多个请求，省掉反复握手，还有流控和优先级。gRPC 的服务端流式（server streaming）天然适合逐 token 返回。注意 HTTP/2 有自己的流控窗口，默认 64 KB，传大请求体（比如长上下文或者多模态的图片）时要调大。
- **HTTP/3 / QUIC** 走 UDP，解决的是 TCP 层的队头阻塞（一个丢包会卡住这条连接上的所有流）。对移动端和弱网有意义，数据中心内部收益不大。

一个容易忽略的细节：HTTP/2 的多路复用是在**一条 TCP 连接**上做的，所以一条连接被负载均衡器分到哪个后端，这条连接上的所有请求就都去那个后端。四层负载均衡加 HTTP/2 长连接，很容易造成后端负载不均（下一章会展开）。

## 流式接口的工程细节

**取消与断连。** 用户关掉页面、客户端超时、网关断开，都要让服务端停止生成。链路是：框架检测到断开（ASGI 的 `http.disconnect`、写操作返回 `EPIPE`）→ 取消这个请求的协程 → 通知调度器中止 → 释放 KV 块。少了最后一步，GPU 会继续为一个没人看的请求生成到 `max_tokens`。vLLM 和 SGLang 都有对应的中止路径（`abort` 请求），自己写服务时要确认这条链路是通的。

**超时要分成两种。** 一种是"多久还没开始输出"（TTFT 超时），一种是"两个 token 之间最多能隔多久"（空闲超时）。总时长超时对长回答不适用：一个 4000 token 的回答本来就要一分多钟。网关、负载均衡、客户端三处的超时都要按这个口径设，并且保证上游的超时比下游长。

**错误发生在中途怎么办。** 响应头已经发出去了（`200 OK`），中途出错没法再改状态码。约定的做法是发一条错误事件（`event: error` 或载荷里带 `error` 字段）然后结束流。客户端要能识别，重试逻辑也要知道"已经吐了一半"的请求不能简单地整个重试。

**统计信息。** token 用量要在流的最后一条事件里带上（OpenAI 的 `stream_options: {"include_usage": true}`），或者在服务端记日志；不要指望客户端自己数 token。

**鉴权与限流放在流开始之前**。一旦开始流式输出，再想拒绝就晚了。限流要按"并发的流"计数，而不是按"每秒请求数"——一个流可能持续几分钟。

!!! interview "面试怎么答"
    被问流式输出怎么实现：响应头声明 `text/event-stream` + 分块传输编码，每个 token 发一条 `data:` 事件，最后 `data: [DONE]`；每写一段都要 flush，并且开 `TCP_NODELAY`。接着讲最常见的故障：链路上任何一处缓冲（框架中间件、Nginx 的 `proxy_buffering`、网关、CDN、gzip、客户端用了非流式读法）都会把逐字输出变成一次性吐出，只有 TTFT 指标能发现。协议选择：面向浏览器用 HTTP/1.1 + SSE，服务之间用 HTTP/2 或 gRPC（多路复用避免队头阻塞，但一条连接绑定一个后端，四层负载均衡下容易不均）。最后是工程细节：断连要一路传到调度器中止请求并释放 KV；超时分 TTFT 超时和 token 间空闲超时；中途出错只能发错误事件；限流按并发流计数。

## 练习

**1. 找出缓冲。** 上线后用户反馈"要等好几秒才开始出字，然后哗一下全出来"。服务端日志显示第一个 token 在 200 ms 就生成了。给出一个排查顺序，每一步用什么命令或方法验证。

??? success "参考答案"
    从里往外逐段验证，每段都用 `curl -N`（不缓冲）加时间戳看首个事件何时到达：

    1. 直接请求推理实例（绕开所有中间层）：`curl -N -w '%{time_starttransfer}' http://实例:端口/v1/...`。如果这里就慢，问题在应用层（没 flush、中间件缓冲、没开 `TCP_NODELAY`）；
    2. 经过 Nginx / 网关请求：慢了就是代理缓冲（检查 `proxy_buffering`、`gzip`、网关的流式支持）；
    3. 经过 CDN / 公网入口：慢了就是最外层；
    4. 客户端：换 `curl -N` 能正常，就是客户端代码用了非流式读法。

    另外用 `tcpdump` 看服务端发出的包是不是一小段一小段的，可以区分"服务端没发"和"中间攒着"。

**2. 队头阻塞的账。** 一个网关用一条 HTTP/1.1 长连接把请求转发给推理实例。某个请求要生成 2000 个 token（40 秒）。后面排队的 5 个短请求会怎样？换成 HTTP/2 呢？换成 6 条 HTTP/1.1 连接呢？

??? success "参考答案"
    HTTP/1.1 单连接：那 5 个请求必须等前面的流结束，各自多等 40 秒——对流式接口这是灾难，因为一个流可以持续很久。HTTP/2：5 个请求和长流交错传输，互不影响（只要没触发流控）。6 条连接：能并发 6 个，但第 7 个仍然要等，而且长流会长期占着连接。

    结论：网关到推理实例之间如果要复用连接，必须用 HTTP/2 或 gRPC；用 HTTP/1.1 就得为每个流单独开连接，并控制连接数。

**3. 断连处理。** 写出从"用户关闭浏览器"到"GPU 停止为它计算"的完整链路，并说明每一环少做了会怎样。

??? success "参考答案"
    浏览器关闭 → TCP 连接被关闭（服务端读到 EOF，或写时收到 RST）→ 框架产生断开事件 / 写操作抛异常 → 请求处理协程被取消 → 调用引擎的 abort 接口 → 调度器把这个请求移出运行队列 → 释放它占的 KV 块和前缀缓存引用。

    少了"取消协程"：这个请求会一直生成到 `max_tokens`，白占 GPU；少了"通知调度器"：协程退了但引擎还在算；少了"释放 KV"：显存泄漏，可用并发越来越少，最终排队变长甚至 OOM。生产上要在监控里看"中止请求数"和"KV 使用率"，两者对不上就是这条链路有洞。

**4. 超时怎么设。** 一个服务最长回答 4000 个 token，正常情况下 TTFT 在 2 秒以内、每个 token 间隔 30 ms。给网关、客户端各设一组超时，并说明理由。

??? success "参考答案"
    网关：读响应头超时（TTFT）设 10 秒左右（留出排队和 prefill 的余量），token 间空闲超时设 30～60 秒（远大于正常的 30 ms，但能及时发现卡死），不设或设一个很大的总时长超时（4000 × 30 ms = 2 分钟，总超时至少要 5 分钟）。

    客户端：TTFT 超时可以更短（比如 15 秒）以便快速重试，空闲超时 60 秒，总超时 5 分钟以上。原则是**上游的超时比下游长**，否则下游还在正常输出，上游先断开，既浪费算力又让用户看到截断。

## 小结

- [x] 流式输出 = 分块传输编码 + SSE 事件；每写一段都要 flush，并开 `TCP_NODELAY`。
- [x] 链路上任何一处缓冲（中间件、`proxy_buffering`、网关、CDN、gzip、客户端读法）都会毁掉流式，只有 TTFT 指标能发现。
- [x] HTTP/1.1 一条连接同时只能跑一个请求；浏览器开多条，HTTP/2 用帧交错，但一条连接绑定一个后端。
- [x] 服务之间用 HTTP/2 或 gRPC；注意 HTTP/2 默认 64 KB 的流控窗口。
- [x] 断连要一路传到调度器中止请求并释放 KV；超时分 TTFT 与 token 间空闲两种，上游要比下游长；中途出错只能发错误事件。
