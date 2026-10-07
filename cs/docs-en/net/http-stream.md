# HTTP and streaming output: how a token reaches the browser

<p class="lead">From the outside an inference service is an HTTP endpoint: `POST /v1/chat/completions`, `stream=true`, and then the tokens pop out one at a time. Underneath that are HTTP's chunked transfer encoding and server-sent events; let any one link in the chain buffer, and the word-by-word output turns into a long wait followed by everything at once. This chapter writes a server-sent-events server and client from scratch to see exactly how each token becomes bytes on the network, then looks at proxy buffering, HTTP/1.1's head-of-line blocking, the difference between HTTP/2 and gRPC, and how a streaming endpoint should handle disconnection, cancellation and timeouts.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What in the HTTP protocol lets the server start sending before the response has finished generating?
    2. What is the difference between server-sent events and WebSocket? Why do OpenAI-style streaming endpoints use server-sent events?
    3. The server sends a token every 20 ms, yet the user sees nothing for several seconds and then everything at once. Which links in the chain could be at fault?
    4. What is HTTP/1.1's head-of-line blocking? How do browsers and HTTP/2 each get around it?
    5. After the user closes the page, how does the server know to stop generating and free the KV cache?

??? success "Answers for the self-test (answer first, then open this)"
    1. Chunked transfer encoding (`Transfer-Encoding: chunked`): the response headers carry no `Content-Length`, and after them each send is a chunk of "hexadecimal length plus data", ending with a chunk of length 0. HTTP/2 does not need it, because its frames carry their own length.
    2. Server-sent events are one-way (server to client) and ride on an ordinary HTTP response; browsers have a native `EventSource` that reconnects automatically, and it passes through proxies and firewalls without a protocol upgrade. WebSocket is a full-duplex protocol of its own that needs an `Upgrade` handshake first. A generative endpoint only has to push text downward, so server-sent events suffice and stay compatible with the existing HTTP infrastructure (authentication, load balancing, rate limiting).
    3. Three common places: (1) the server has no `TCP_NODELAY` or does not flush, so the tokens accumulate; (2) a proxy in the middle buffers the response (Nginx's `proxy_buffering`, a CDN, an API gateway); (3) the client framework reads the response as a complete body. In this chapter's experiment, a proxy that waits for 4 KB before forwarding pushed the first event from 0.3 ms to 405 ms.
    4. An HTTP/1.1 connection can only carry one request at a time (even with pipelining, the responses have to come back in order), so a slow request in front blocks those behind it. The browser's answer is 6 connections per domain; HTTP/2's answer is to split requests into frames with a stream id and interleave them on one connection.
    5. After the client disconnects, writing to that connection fails (`EPIPE` / `ECONNRESET`), or the framework marks the request as disconnected (`http.disconnect` in ASGI, `await request.is_disconnected()`). An inference engine has to carry that signal all the way to the scheduler: abort this request's generation and free the KV blocks it holds. Without that, the GPU keeps computing for a page the user has closed, wasting the most expensive resource there is.

## From one token to a piece of HTTP response {#从一个-token-到一段-http-响应}

![Figure: the path of a streamed response and the three common buffering points](../assets/figures/sse-stream.svg){.aig-svg}

An OpenAI-style streamed response looks like this: the headers declare `Content-Type: text/event-stream`, then each token goes out as one `data: {...}` event, and the last one is `data: [DONE]`. The program below writes both the server and the client and records when each event arrives:

```python title="sse.py"
# a minimal server-sent-events server and client: chunked transfer, flushing once per generated token
import socket
import threading
import time

HOST, TOKENS = "127.0.0.1", 8
HEAD = (b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n")


def chunk(payload: bytes) -> bytes:
    return b"%x\r\n" % len(payload) + payload + b"\r\n"           # chunked transfer: the hexadecimal length plus the data


def serve(srv):
    conn, _ = srv.accept()
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # without this, Nagle accumulates the tokens
        while b"\r\n\r\n" not in conn.recv(4096):                   # read the request headers to the end
            pass
        conn.sendall(HEAD)
        time.sleep(0.05)                                            # pretend to be doing the prefill
        for i in range(TOKENS):
            conn.sendall(chunk(b'data: {"token": "t%d"}\n\n' % i))
            time.sleep(0.02)                                        # one token every 20 ms
        conn.sendall(chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")     # a chunk of length 0 means the end


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
while b"\r\n\r\n" not in buf:                                       # receive the response headers first
    buf += c.recv(4096)
head_at = time.perf_counter() - t0
buf = buf.split(b"\r\n\r\n", 1)[1]
while True:                                                         # then receive the events chunk by chunk
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

```text title="output (on this machine)"
响应头到达（可以开始渲染）：   0 ms
事件到达   50 ms：data: {"token": "t0"}
事件到达   71 ms：data: {"token": "t1"}
事件到达   91 ms：data: {"token": "t2"}
……最后一个事件：212 ms，共 9 个事件，平均间隔 20 ms
```

A few things worth noting:

- **The headers go first**. The server sends them as soon as it has decided to start generating, and the browser enters streaming-render mode on that basis. This is also why, within the time to first token, seeing the first character matters far more than the request completing.
- **One chunk per event**. Each chunk of the chunked transfer encoding is "hexadecimal length, `\r\n`, data, `\r\n`", and a final chunk of length 0 ends it. What follows `data:` is the server-sent-events payload, and two newlines end one event.
- **Every chunk has to go out immediately**. `TCP_NODELAY` is turned on explicitly here; with a framework the equivalent is flushing after every write (Python's `StreamingResponse`, Go's `http.Flusher`). Forget this step and the tokens accumulate (see [Nagle's algorithm](tcp.md#nagle-算法与-40-毫秒) in the previous chapter).

The other fields of server-sent events are useful too: `event:` classifies an event (for instance to separate `usage` statistics from the text), `id:` numbers the events, and a client reconnecting after a drop sends `Last-Event-ID` so the server can resume from there (inference services generally do not resume; they regenerate). `retry:` tells the client how long to wait before reconnecting. To get through the middle devices that drop a connection after a long silence, send a comment line (`: keep-alive`) periodically as a heartbeat when no token has come for a while.

## Buffering is the number one enemy of streaming {#缓冲是流式输出的头号敌人}

If any link in the chain accumulates data, the streaming is gone. A minimal proxy demonstrates it: one that forwards straight through, and one that waits for 4 KB.

```python title="proxy_buffer.py"
# a proxy in the middle: straight through against waiting for 4 KB. The latter turns streamed output into one burst
import socket
import threading
import time

HOST, TOKENS, GAP = "127.0.0.1", 40, 0.01
TOKEN = b"data: " + b"x" * 40 + b"\n\n"                     # about 50 bytes per event


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
        if len(pending) >= buffer_size:                     # forward only once a buffer's worth has accumulated
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

```text title="output (on this machine)"
直通（收到就转发）：第一个事件    0.3 ms，全部收完  395.0 ms
攒够 4 KB 再转发：第一个事件  405.1 ms，全部收完  405.1 ms
```

Straight through, the first event arrives immediately and the total time is the generation time. Waiting for 4 KB, the user receives everything in one go at the last moment. The experience goes from characters appearing one by one to a spinner for a few hundred milliseconds and then a flood, while the total time barely changes. The average latency on a load-test report shows no difference either; only a time-to-first-token metric catches it.

The links to check one by one in a real deployment:

| Link | What to do |
| --- | --- |
| The application framework | use a streaming response object and flush after every write; avoid middleware that buffers (blanket response compression, JSON wrapping) |
| Nginx / OpenResty | `proxy_buffering off;`, `proxy_cache off;`, `chunked_transfer_encoding on;`, `gzip off` (or compression off for `text/event-stream`) |
| A cloud load balancer or API gateway | confirm it supports streamed responses; some gateways buffer the whole response by default, or cap the response body's size |
| A CDN | bypass it for streaming endpoints, or confirm it supports streaming from the origin |
| The client | use a streaming read (`fetch`'s `ReadableStream`, `stream=True` with `requests`, `iter_lines` with `httpx`), not `response.json()` |

Compression deserves its own sentence: `gzip` also accumulates data for the sake of the ratio, so for a streamed response either turn it off or enable the mode that ends a compression block on every flush. Server-sent-events text is highly repetitive to begin with, so the gain at the gateway layer is limited and in practice most deployments simply turn it off.

## HTTP/1.1, HTTP/2 and gRPC {#http11http2-与-grpc}

One HTTP/1.1 connection handles one request at a time, so a slow request in front blocks those behind it. This is **head-of-line blocking**:

```python title="multiplex.py"
# N concurrent requests on one connection: HTTP/1.1 can only do them one after another (head-of-line blocking), HTTP/2 can interleave
def http11(reqs, conns):
    """conns 条连接，每条串行处理排给它的请求。reqs 是每个请求的服务端耗时（ms）"""
    queues = [0.0] * conns
    done = []
    for i, cost in enumerate(reqs):
        k = i % conns                          # deal them round-robin into the connections
        queues[k] += cost
        done.append(queues[k])
    return done


def http2(reqs):
    """一条连接，所有请求同时开始（服务端并发处理），各自在自己的耗时后完成"""
    return list(reqs)


slow, fast = 500.0, 20.0
reqs = [slow] + [fast] * 5                     # one slow request in front and five fast ones behind
for name, done in [("HTTP/1.1，1 条连接", http11(reqs, 1)),
                   ("HTTP/1.1，6 条连接", http11(reqs, 6)),
                   ("HTTP/2，1 条连接", http2(reqs))]:
    print(f"{name:20s} 快请求的完成时间：{[round(t) for t in done[1:]]} ms，"
          f"最后一个 {max(done):.0f} ms")
```

```text title="output"
HTTP/1.1，1 条连接       快请求的完成时间：[520, 540, 560, 580, 600] ms，最后一个 600 ms
HTTP/1.1，6 条连接       快请求的完成时间：[20, 20, 20, 20, 20] ms，最后一个 500 ms
HTTP/2，1 条连接         快请求的完成时间：[20, 20, 20, 20, 20] ms，最后一个 500 ms
```

The browser's answer is 6 connections per domain (the second row above); HTTP/2's answer is to split requests into frames with a stream id and interleave them on one connection, so one connection carries many concurrent requests.

For an inference service:

- **A browser-facing endpoint** needs nothing more than HTTP/1.1 and server-sent events: one conversation is one long response, and the browser will not put other requests on that connection.
- **Between services** (the gateway to an inference instance, or instances to each other) HTTP/2 or gRPC fits better: many concurrent requests on one long-lived connection, no repeated handshakes, plus flow control and priorities. gRPC's server streaming is a natural fit for returning tokens one at a time. Note that HTTP/2 has a flow-control window of its own, 64 KB by default, which has to be raised for a large request body (a long context, or images in a multimodal request).
- **HTTP/3 / QUIC** runs over UDP and solves head-of-line blocking at the TCP layer (where one lost packet stalls every stream on the connection). It matters for mobile and poor networks; inside a data centre the gain is small.

One detail that is easy to miss: HTTP/2's multiplexing happens on **one TCP connection**, so whichever back end a load balancer sends that connection to is where every request on it goes. A layer-4 load balancer plus long-lived HTTP/2 connections easily produces uneven back-end load (the next chapter covers this).

## The engineering details of a streaming endpoint {#流式接口的工程细节}

**Cancellation and disconnection.** A user closing the page, a client timing out, the gateway dropping the connection all have to stop the server generating. The chain is: the framework detects the disconnection (ASGI's `http.disconnect`, or a write returning `EPIPE`), the request's coroutine is cancelled, the scheduler is told to abort, and the KV blocks are freed. Without the last step, the GPU keeps generating to `max_tokens` for a request nobody is watching. vLLM and SGLang both have an abort path for this; when writing a service yourself, confirm the chain is unbroken.

**Timeouts come in two kinds.** One is how long until output starts (the time-to-first-token timeout), the other is the most time allowed between two tokens (the idle timeout). A total-duration timeout does not work for a long answer: a 4000-token answer takes over a minute by its nature. Set the timeouts at all three places, the gateway, the load balancer and the client, in these terms, and make sure the upstream timeout is longer than the downstream one.

**What to do when an error happens mid-stream.** The headers are already out (`200 OK`), so an error part-way through cannot change the status code. The convention is to send an error event (`event: error`, or an `error` field in the payload) and then end the stream. The client has to recognise it, and the retry logic has to know that a request which already emitted half its answer cannot simply be retried whole.

**Statistics.** Token usage goes in the last event of the stream (OpenAI's `stream_options: {"include_usage": true}`) or into a server-side log; do not expect the client to count tokens itself.

**Authentication and rate limiting go before the stream starts.** Once streaming output has begun it is too late to reject. Rate limiting has to count concurrent streams rather than requests per second, since one stream may last minutes.

!!! interview "How to answer in an interview"
    Asked how streaming output works: headers declaring `text/event-stream` plus chunked transfer encoding, one `data:` event per token, `data: [DONE]` at the end; flush after every write and turn on `TCP_NODELAY`. Then the most common failure: buffering anywhere in the chain (framework middleware, Nginx's `proxy_buffering`, a gateway, a CDN, gzip, a client using a non-streaming read) turns word-by-word output into one burst, and only a time-to-first-token metric finds it. On protocol choice: HTTP/1.1 with server-sent events facing browsers, HTTP/2 or gRPC between services (multiplexing avoids head-of-line blocking, but one connection is pinned to one back end, which easily goes uneven behind a layer-4 load balancer). Finish with the engineering details: a disconnection has to reach the scheduler to abort the request and free the KV; timeouts split into a time-to-first-token one and an inter-token idle one; an error mid-stream can only be an error event; and rate limiting counts concurrent streams.

## Exercises {#练习}

**1. Find the buffering.** After launch, users report waiting several seconds before any text appears and then all of it at once. The server log shows the first token was generated at 200 ms. Give an order of investigation, and what command or method verifies each step.

??? success "Answer"
    Verify segment by segment from the inside out, using `curl -N` (no buffering) with a timestamp each time to see when the first event arrives:

    1. Request the inference instance directly, bypassing every middle layer: `curl -N -w '%{time_starttransfer}' http://instance:port/v1/...`. If it is already slow here, the problem is in the application (no flush, buffering middleware, no `TCP_NODELAY`).
    2. Request through Nginx or the gateway: slower means proxy buffering (check `proxy_buffering`, `gzip`, and the gateway's streaming support).
    3. Request through the CDN or the public entry point: slower means the outermost layer.
    4. The client: if `curl -N` works fine, the client code is using a non-streaming read.

    Also use `tcpdump` to see whether the server's packets go out in small pieces, which distinguishes "the server is not sending" from "something in the middle is accumulating".

**2. The head-of-line blocking account.** A gateway forwards requests to an inference instance over one long-lived HTTP/1.1 connection. One request has to generate 2000 tokens (40 seconds). What happens to the 5 short requests queued behind it? And with HTTP/2? And with 6 HTTP/1.1 connections?

??? success "Answer"
    HTTP/1.1 on one connection: those 5 requests have to wait for the stream in front to end, 40 extra seconds each. For a streaming endpoint this is a disaster, because one stream can last a very long time. HTTP/2: the 5 requests interleave with the long stream and do not affect each other (as long as flow control is not hit). 6 connections: 6 can run at once, but the 7th still waits, and the long stream holds a connection indefinitely.

    The conclusion: if the gateway reuses connections to the inference instance, it has to be HTTP/2 or gRPC; with HTTP/1.1 you need a separate connection per stream and a cap on the connection count.

**3. Handling a disconnection.** Write out the full chain from "the user closes the browser" to "the GPU stops computing for it", and say what happens if any link is left out.

??? success "Answer"
    The browser closes, the TCP connection is closed (the server reads EOF, or gets an RST on a write), the framework raises a disconnect event or the write throws, the request-handling coroutine is cancelled, the engine's abort interface is called, the scheduler takes the request out of the running queue, and the KV blocks and prefix-cache references it holds are freed.

    Without the cancellation: the request generates all the way to `max_tokens`, occupying the GPU for nothing. Without telling the scheduler: the coroutine is gone but the engine is still computing. Without freeing the KV: device memory leaks, the usable concurrency keeps shrinking, and eventually the queue grows or you run out of memory. In production, watch the aborted-request count and the KV utilisation in monitoring; when the two do not line up, there is a hole in this chain.

**4. Setting the timeouts.** A service answers up to 4000 tokens, normally with a time to first token under 2 seconds and 30 ms between tokens. Set a group of timeouts for the gateway and for the client, with your reasoning.

??? success "Answer"
    The gateway: a response-header (time-to-first-token) timeout of about 10 seconds, leaving room for queueing and prefill; an inter-token idle timeout of 30 to 60 seconds, far above the normal 30 ms but still quick to catch a hang; and either no total-duration timeout or a very large one (4000 x 30 ms is 2 minutes, so a total timeout has to be at least 5).

    The client: the time-to-first-token timeout can be shorter (say 15 seconds) to retry quickly, an idle timeout of 60 seconds, and a total above 5 minutes. The principle is that **the upstream timeout is longer than the downstream one**, otherwise the downstream is still producing output when the upstream cuts it off, which wastes compute and shows the user a truncated answer.

## Summary {#小结}

- [x] Streaming output is chunked transfer encoding plus server-sent events; flush after every write and turn on `TCP_NODELAY`.
- [x] Buffering anywhere in the chain (middleware, `proxy_buffering`, a gateway, a CDN, gzip, the client's read) destroys the streaming, and only a time-to-first-token metric finds it.
- [x] An HTTP/1.1 connection runs one request at a time; browsers open several and HTTP/2 interleaves frames, but one connection is pinned to one back end.
- [x] Use HTTP/2 or gRPC between services, and watch HTTP/2's default 64 KB flow-control window.
- [x] A disconnection has to reach the scheduler to abort the request and free the KV; timeouts split into time-to-first-token and inter-token idle, with the upstream longer than the downstream; an error mid-stream can only be an error event.
