# TCP: one request's journey across the network

<p class="lead">The network leg is often left out of an inference service's latency account: why is the first byte 100 milliseconds slower for a user in another city? Why does a client that reads slowly bring the GPU to a halt? Why does a health check that sends a few dozen bytes occasionally take 40 milliseconds? This chapter follows a TCP connection from its opening to its close, using socket programs run on this machine to see the round-trip account of the handshake, the 40-millisecond delay that Nagle's algorithm causes, the back pressure that a full buffer creates, and why the bandwidth-delay product keeps a long-distance transfer from filling the link.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How many round trips does an HTTPS request take at minimum before the first response byte arrives? Which of them does connection reuse save?
    2. What is Nagle's algorithm? What happens when it collides with delayed acknowledgement?
    3. Once `send()` has returned on the sending side, has the data reached the other end? What if the other end never reads?
    4. On a connection with a 30 ms round-trip time and a 256 KB window, what is the most bandwidth you can get?
    5. Why does the side that closes a connection actively enter TIME_WAIT? What is going on when a load test runs out of ports?

??? success "Answers for the self-test (answer first, then open this)"
    1. At least 4: one for the DNS lookup (when it is not cached), one for the TCP three-way handshake, one for the TLS 1.3 handshake (TLS 1.2 takes 2), and one from sending the request to receiving the first byte. Reusing a connection saves the handshake and the TLS, a cached DNS saves that too, and only the last round trip is left. That is why every client should use a connection pool.
    2. Nagle's algorithm: while a connection still has unacknowledged small packets, new small data is held back until an ACK comes back or a full MSS has accumulated. The other end's delayed acknowledgement then holds the ACK for tens of milliseconds (up to 40 ms on Linux). So the write-the-header, write-the-body, wait-for-the-reply pattern reliably costs tens of extra milliseconds. This chapter measures 41 ms, dropping to 0.02 ms with `TCP_NODELAY` on.
    3. No. `send()` returns as soon as the data has been copied into the kernel's send buffer. If the other end does not read, its receive buffer fills, TCP's flow control squeezes the sender's window down to 0, the send buffer fills in turn, and a further `send()` blocks (a non-blocking socket returns `EAGAIN`). That is back pressure: a slow client pushes the pressure all the way back into the application.
    4. Throughput is about window / round-trip time = 256 KB / 30 ms, about 8.7 MB/s, about 0.07 Gb/s. To fill a 10 Gb/s link with a 1 ms round-trip time the window has to be at least 1.2 MB. That is the bandwidth-delay product.
    5. TIME_WAIT absorbs duplicate segments that may still be late in the network and makes sure the final ACK can be retransmitted if it is lost. It lasts twice the maximum segment lifetime (60 seconds on Linux). In a load test that opens a new connection per request, tens of thousands of TIME_WAIT entries fill the local port range and you get `Cannot assign requested address`. The fix is to reuse connections (a pool, HTTP keep-alive), not to tune the risky kernel parameters.

## How many round trips a request takes {#一个请求要花几个往返}

Start with the arithmetic. The round-trip time is the basic unit of network latency: a fraction of a millisecond within one data centre, a few milliseconds within a city, tens of milliseconds between cities, around two hundred between continents.

```python title="handshake.py"
# how many round trips does a request take on the network? (the RTT is taken as 30 ms, between cities)
rtt = 30
steps = [
    (1, "DNS 查询", "有缓存时省掉"),
    (1, "TCP 三次握手", "连接复用时省掉"),
    (1, "TLS 1.3 握手", "会话复用或 0-RTT 时省掉；TLS 1.2 要 2 个往返"),
    (1, "发请求、等第一个响应字节", "这一段里还包含服务端的处理时间"),
]
print(f"第一次请求（RTT = {rtt} ms）：")
for n, name, note in steps:
    print(f"  {n * rtt:3d} ms  {name}（{note}）")
print(f"  {sum(n for n, _, _ in steps) * rtt:3d} ms  合计，还没算服务端生成第一个 token 的时间")
print(f"\n连接复用 + DNS 缓存之后只剩最后一个往返：{rtt} ms\n")
print("推理服务的 TTFT 里网络占多少（服务端按 200 ms 算）：")
for name, r in [("同机房", 0.5), ("同城", 5), ("跨城", 30), ("跨洲", 200)]:
    print(f"  {name:4s}RTT {r:5.1f} ms：新建连接要 {4 * r:6.1f} ms，复用连接 {r:5.1f} ms，"
          f"网络占总 TTFT 的 {r / (r + 200):4.1%}")
```

```text title="output"
第一次请求（RTT = 30 ms）：
   30 ms  DNS 查询（有缓存时省掉）
   30 ms  TCP 三次握手（连接复用时省掉）
   30 ms  TLS 1.3 握手（会话复用或 0-RTT 时省掉；TLS 1.2 要 2 个往返）
   30 ms  发请求、等第一个响应字节（这一段里还包含服务端的处理时间）
  120 ms  合计，还没算服务端生成第一个 token 的时间

连接复用 + DNS 缓存之后只剩最后一个往返：30 ms

推理服务的 TTFT 里网络占多少（服务端按 200 ms 算）：
  同机房 RTT   0.5 ms：新建连接要    2.0 ms，复用连接   0.5 ms，网络占总 TTFT 的 0.2%
  同城  RTT   5.0 ms：新建连接要   20.0 ms，复用连接   5.0 ms，网络占总 TTFT 的 2.4%
  跨城  RTT  30.0 ms：新建连接要  120.0 ms，复用连接  30.0 ms，网络占总 TTFT 的 13.0%
  跨洲  RTT 200.0 ms：新建连接要  800.0 ms，复用连接 200.0 ms，网络占总 TTFT 的 50.0%
```

![Figure: the round trips a new connection takes against a reused one](../assets/figures/rtt-ladder.svg){.aig-svg}

Two conclusions:

- **Connection reuse is the best value in network optimisation**: it saves three round trips, which is 90 ms between cities. An inference client, and a gateway talking to an inference instance, should both use long-lived connections (HTTP keep-alive, HTTP/2, gRPC).
- **Where you deploy decides the network's share of the latency**: within one data centre it is negligible; between continents it is as large as the whole time to first token. A latency-sensitive service has to be deployed close to its users (see [Production deployment and operations](serving://ops/deploy/)).

## Nagle's algorithm and 40 milliseconds {#nagle-算法与-40-毫秒}

TCP wants to send fewer small packets: every packet carries 40 bytes of IP and TCP header, so sending one byte of data costs 41. **Nagle's algorithm** is the rule that while a connection still has unacknowledged data, new small chunks are held back until an ACK comes back or a full MSS (usually 1460 bytes) has accumulated.

On the other side, the receiver has **delayed acknowledgement**: it does not ACK immediately on receiving data, but waits a little to see whether the ACK can ride along with data of its own, up to 40 ms on Linux.

When the two collide, things go wrong. The program below models the most common pattern, writing the request header, then the body, then waiting for the reply:

```python title="nagle.py"
# a request written in two calls (the header, then the body), with the server replying only once it has everything. Nagle's algorithm holds the second write until the first one's ACK
import socket
import threading
import time

HOST, N = "127.0.0.1", 50


def server(sock):
    conn, _ = sock.accept()
    with conn:
        while True:
            head = conn.recv(4)                       # the header: 4 bytes
            if not head:
                return
            body = b""
            while len(body) < 8:
                body += conn.recv(8 - len(body))      # the body: 8 bytes
            conn.sendall(b"ok")                       # reply only once it is all here


def round_trips(nodelay):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, 0))
    srv.listen(1)
    t = threading.Thread(target=server, args=(srv,), daemon=True)
    t.start()
    c = socket.create_connection(srv.getsockname())
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 if nodelay else 0)
    for _ in range(3):                                # warm up
        c.sendall(b"head"), c.sendall(b"12345678"), c.recv(2)
    t0 = time.perf_counter()
    for _ in range(N):
        c.sendall(b"head")                            # the first write: goes out at once
        c.sendall(b"12345678")                        # the second write: Nagle waits for the previous packet's ACK
        c.recv(2)
    dt = (time.perf_counter() - t0) / N * 1e3
    c.close()
    srv.close()
    t.join(timeout=1)
    return dt


print(f"开着 Nagle（默认）：每次往返 {round_trips(False):.3f} ms")
print(f"TCP_NODELAY：      每次往返 {round_trips(True):.3f} ms")
```

```text title="output (on this machine)"
开着 Nagle（默认）：每次往返 41.059 ms
TCP_NODELAY：      每次往返 0.021 ms
```

The second write is held by Nagle, waiting for the first packet's ACK, while the other end waits to have everything before replying, so the ACK only goes out when the delayed-acknowledgement timer expires. Every request reliably costs 40 ms more, dropping to 0.02 ms once `TCP_NODELAY` turns Nagle off.

The rules in practice:

- **Turn `TCP_NODELAY` on for every interactive, request-response connection**. Nginx, gRPC and Redis clients all have it on by default; remember to set it when you write socket code yourself.
- **The deeper fix is to write it all at once**: join the header and the body into one `send`, or submit several segments in one `writev` / `sendmsg`. Accumulating before sending is what Nagle was trying to do in the first place.

In an inference service the easiest place to hit this is streaming output: each generated token writes a small piece (a dozen-odd bytes), and without `TCP_NODELAY` the tokens get held together and arrive in bursts, so the user sees stuttering instead of a word-by-word flow.

## Sent is not delivered: buffers and back pressure {#发送不等于送达缓冲区与背压}

A `send()` that returns only means the data was copied into the kernel's send buffer. Where it actually goes is limited by two things: this end's send buffer (`SO_SNDBUF`), and the **receive window** the other end advertises in its ACKs, which is the space left in its buffer. If the other end does not read, the window falls to 0 and the sender can only stop.

How much a sender can write when the receiver reads nothing at all:

```python title="backpressure.py"
# how many bytes can the sender write when the receiver does not read? The send buffer plus the peer's receive window, and a further write blocks
import socket
import threading

HOST = "127.0.0.1"


def probe(sndbuf, rcvbuf):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)   # the kernel actually allocates about twice the value set
    srv.bind((HOST, 0))
    srv.listen(1)
    box = {}
    threading.Thread(target=lambda: box.setdefault("conn", srv.accept()[0]), daemon=True).start()
    c = socket.create_connection(srv.getsockname())
    c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
    c.setblocking(False)
    written = 0
    chunk = b"x" * 8192
    while True:
        try:
            written += c.send(chunk)                              # the server reads not one byte
        except BlockingIOError:
            break
    real_snd = c.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    c.close()
    if "conn" in box:
        box["conn"].close()
    srv.close()
    return written, real_snd


for snd, rcv in [(16 * 1024, 16 * 1024), (64 * 1024, 64 * 1024), (256 * 1024, 256 * 1024)]:
    written, real = probe(snd, rcv)
    print(f"SO_SNDBUF 设为 {snd // 1024:3d} KB（内核给 {real // 1024:3d} KB）、"
          f"对端 SO_RCVBUF {rcv // 1024:3d} KB：阻塞前写进去 {written / 1024:6.0f} KB")
```

```text title="output (on this machine)"
SO_SNDBUF 设为  16 KB（内核给  32 KB）、对端 SO_RCVBUF  16 KB：阻塞前写进去     40 KB
SO_SNDBUF 设为  64 KB（内核给 128 KB）、对端 SO_RCVBUF  64 KB：阻塞前写进去    160 KB
SO_SNDBUF 设为 256 KB（内核给 512 KB）、对端 SO_RCVBUF 256 KB：阻塞前写进去    616 KB
```

What goes in is about the send buffer plus the peer's receive buffer (the kernel usually allocates twice what `setsockopt` asked for, to cover protocol overhead). Once it is full, a blocking socket stalls in `send()`, a non-blocking one returns `EAGAIN`, and `epoll` stops reporting it writable (see [epoll and io_uring](../os/io-models.md)).

That chain is **back pressure**: the client reads slowly, the peer's receive buffer fills, the send window goes to 0, this end's send buffer fills, the write blocks or returns `EAGAIN`, and the application has to stop. For an inference service this means:

- A client with a bad network, or one that simply does not read the response, stalls the coroutine or thread that writes results back. If that thread also does other work, such as the scheduling loop, it slows the whole engine down. This is why inference engines separate generating from sending: results go into a queue first and a dedicated I/O task writes them out.
- The queue cannot grow without bound either. The sensible approach is a cap on each request's pending queue, disconnecting or dropping the request when it is exceeded (a slow client should not take the service down), and monitoring that distinguishes "generating slowly" from "cannot send".
- Many slow clients also hold memory: a few hundred kilobytes of send and receive buffer per connection is a few gigabytes over ten thousand connections.

## The bandwidth-delay product: the window decides the throughput {#带宽延迟积窗口决定吞吐}

A connection's throughput ceiling is **the window divided by the round-trip time**: after sending a window's worth of data you have to wait for the ACK, so one round trip carries at most one window.

```python title="bdp.py"
# the bandwidth-delay product: one TCP connection's throughput is about the window / the RTT. With too small a window, no amount of bandwidth gets filled
def throughput_gbps(window_kb, rtt_ms):
    return window_kb * 1024 * 8 / (rtt_ms * 1e-3) / 1e9


print("窗口 \\ RTT" + "".join(f"{r:>10} ms" for r in (0.1, 1, 10, 50)))
for kb in (64, 256, 1024, 4096):
    row = "".join(f"{throughput_gbps(kb, r):10.2f} Gb/s" for r in (0.1, 1, 10, 50))
    print(f"{kb:6d} KB" + row)
print()
print("要跑满一条链路，窗口至少要有带宽 × RTT：")
for gbps, rtt in [(10, 0.1), (10, 1), (25, 10), (100, 50)]:
    need = gbps * 1e9 / 8 * (rtt * 1e-3)
    print(f"  {gbps:3d} Gb/s、RTT {rtt:4.1f} ms：{need / 1024:8.0f} KB")
print()
print("慢启动：初始窗口 10 个 MSS（约 14 KB），每个 RTT 翻倍，要几个 RTT 才能发完一个响应？")
for kb in (14, 100, 1024, 10240):
    sent, win, rtts = 0, 14, 0
    while sent < kb:
        sent += win
        win *= 2
        rtts += 1
    print(f"  {kb:6d} KB 的响应：{rtts} 个 RTT")
```

```text title="output"
窗口 \ RTT       0.1 ms         1 ms        10 ms        50 ms
    64 KB      5.24 Gb/s      0.52 Gb/s      0.05 Gb/s      0.01 Gb/s
   256 KB     20.97 Gb/s      2.10 Gb/s      0.21 Gb/s      0.04 Gb/s
  1024 KB     83.89 Gb/s      8.39 Gb/s      0.84 Gb/s      0.17 Gb/s
  4096 KB    335.54 Gb/s     33.55 Gb/s      3.36 Gb/s      0.67 Gb/s

要跑满一条链路，窗口至少要有带宽 × RTT：
   10 Gb/s、RTT  0.1 ms：     122 KB
   10 Gb/s、RTT  1.0 ms：    1221 KB
   25 Gb/s、RTT 10.0 ms：   30518 KB
  100 Gb/s、RTT 50.0 ms：  610352 KB

慢启动：初始窗口 10 个 MSS（约 14 KB），每个 RTT 翻倍，要几个 RTT 才能发完一个响应？
      14 KB 的响应：1 个 RTT
     100 KB 的响应：4 个 RTT
    1024 KB 的响应：7 个 RTT
   10240 KB 的响应：10 个 RTT
```

The key points:

- **A transfer across regions has to grow the window**. Linux's receive window auto-tunes (`net.ipv4.tcp_rmem`) with a ceiling usually in the megabytes. When moving a large file between data centres (model weights, a KV-cache snapshot) and the throughput sticks at tens of MB/s, compute the bandwidth-delay product first, then check the window ceiling and whether window scaling is on.
- **Slow start penalises small responses**. A freshly opened connection has a congestion window of only 10 MSS (about 14 KB), doubling every round trip. A 1 MB response takes 7 round trips to send, over 200 ms between cities, during which the link is nowhere near full. The other benefit of reusing a connection is exactly this: the window has already grown.
- **Streaming output is not subject to this limit**: each send is a few dozen bytes, the window is never the bottleneck, and the latency is set by the round-trip time and the server's generation speed.

The congestion-control algorithm (CUBIC by default on Linux, replaceable with BBR) decides how the window grows and how it backs off on loss. Inside a data centre, schemes like ECN and DCTCP are also common, letting the switch signal the host before it actually drops anything. RDMA takes another route: it puts reliable transport into the network card and bypasses the kernel stack (see [Pinned memory, DMA and NUMA](../os/pinned-numa.md) and [The RDMA programming model](serving://comm/rdma/) in the inference handbook).

## Closing a connection, and TIME_WAIT {#连接的关闭与-time_wait}

Closing takes a four-way exchange: each side sends a FIN and each side ACKs. **The side that closes actively** enters TIME_WAIT and waits twice the maximum segment lifetime (60 seconds on Linux) before really releasing, so as to absorb late duplicate segments and to guarantee the final ACK can be retransmitted if lost.

```python title="reuse.py"
# a new connection per request against reusing one: the difference over loopback on this machine, and how many TIME_WAIT entries are left
import socket
import subprocess
import threading
import time

HOST, N = "127.0.0.1", 300


def serve(srv, stop):
    while not stop.is_set():
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def handle(conn):
    with conn:
        while True:
            data = conn.recv(64)
            if not data:
                return
            conn.sendall(b"ok")


def time_waits(port):
    out = subprocess.run(["ss", "-tan", "state", "time-wait"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if f":{port} " in line or line.rstrip().endswith(f":{port}"))


srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, 0))
srv.listen(128)
port = srv.getsockname()[1]
stop = threading.Event()
threading.Thread(target=serve, args=(srv, stop), daemon=True).start()

before = time_waits(port)
t0 = time.perf_counter()
for _ in range(N):
    c = socket.create_connection((HOST, port))         # a new connection per request
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.sendall(b"ping")
    c.recv(2)
    c.close()                                          # the side that closes actively enters TIME_WAIT
new_conn = (time.perf_counter() - t0) / N * 1e3
left = time_waits(port) - before

c = socket.create_connection((HOST, port))
c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
t0 = time.perf_counter()
for _ in range(N):
    c.sendall(b"ping")
    c.recv(2)
reused = (time.perf_counter() - t0) / N * 1e3
c.close()
stop.set()
srv.close()

print(f"每次新建连接：每个请求 {new_conn:.3f} ms，结束后留下 {left} 个 TIME_WAIT")
print(f"复用一条连接：每个请求 {reused:.3f} ms，快 {new_conn / reused:.0f} 倍")
```

```text title="output (on this machine)"
每次新建连接：每个请求 0.166 ms，结束后留下 300 个 TIME_WAIT
复用一条连接：每个请求 0.016 ms，快 10 倍
```

Over loopback on this machine, opening a connection costs 10 times what reusing one does (the gap is larger on a real network, which adds a round trip), and every short-lived connection leaves a TIME_WAIT behind. The local port range (`net.ipv4.ip_local_port_range`, usually a little over 30,000) runs out quickly under short-lived connections at high QPS, showing up as `Cannot assign requested address`.

The right answer is **to reuse connections**: a pool and HTTP keep-alive on the client, long-lived connections from the gateway to the back end as well. On the server side, let the client close first so the server does not pile up TIME_WAIT of its own. The folk remedies of setting `tcp_tw_reuse` or shortening `tcp_fin_timeout` relieve the symptom but are risky behind NAT, so get connection reuse right first.

A few other common connection-layer problems:

| Symptom | Common cause |
| --- | --- |
| The connection is occasionally reset (`Connection reset by peer`) | the peer process crashed, or called `close` while data was still in the buffer; it can also be a timeout in a load balancer in the middle |
| The first request after an idle period fails | a device in the middle (a load balancer, NAT) quietly reclaimed the idle connection. Turn on TCP keepalive or an application-level heartbeat, and set the client's idle timeout shorter than the middle device's |
| Many connections stuck in `SYN_RECV` under high concurrency | the `listen` backlog is too small or accept is too slow, and the handshake queue overflows (check `Send-Q` and `Recv-Q` with `ss -lnt`) |
| Occasional jitter of a few hundred milliseconds | loss and retransmission: the minimum interval for one timeout retransmission is already on the order of 200 ms |

The tools to investigate with: `ss -tan` (the distribution of connection states), `ss -lnt` (the listen queues), `netstat -s` or `nstat` (retransmission and loss counters), `tcpdump` to capture packets, and `ping` and `mtr` for round-trip time and loss.

!!! interview "How to explain it"
    To explain what happens to a request on its way from a client to a server: DNS, the TCP three-way handshake, the TLS handshake, sending the request, waiting for the first byte, 30 ms per round trip between cities, and connection reuse saves the first three. Then three places things go wrong. (1) Nagle colliding with delayed acknowledgement reliably costs 40 ms more, so interactive connections need `TCP_NODELAY`, streaming output especially. (2) A `send` that returns is not delivery, and a peer that does not read pushes the pressure back through the window, which is back pressure, so an inference engine separates generating from sending and caps the pending queue. (3) Throughput is the window divided by the round-trip time, a large transfer across regions needs a bigger window, and slow start costs a small response a few extra round trips. Finish with connection management: the side that closes actively enters TIME_WAIT, a load test over short-lived connections exhausts the ports, and the answer is a connection pool and keep-alive.

## Exercises {#练习}

**1. Count the round trips.** A user in Shanghai, an inference service deployed in Beijing (30 ms round-trip time), HTTPS with TLS 1.3, and a new connection per request. The server's time to first token is 150 ms, after which it produces a token every 20 ms, for a 200-token answer. When does the user see the first character? How long does the whole answer take? And with a reused connection?

??? success "Answer"
    A new connection: DNS (cached, so 0) + 30 for the handshake + 30 for TLS + 30 from the request to the first byte + 150 on the server = 240 ms to the first character; then 200 tokens x 20 ms = 4000 ms, plus the one-way transfer of the last piece, about 4.2 s in total.

    A reused connection: 30 + 150 = 180 ms to the first character, 60 ms saved; the total is almost unchanged (the later tokens of a stream are no longer affected by the handshake). The time to first token improves by 25%, which matters far more to how it feels than the total does.

**2. Diagnose 40 milliseconds.** An internal service has a p50 latency of 0.5 ms, but about 15% of requests land at almost exactly 40 ms, retries included. What could it be? How would you confirm it? How would you fix it?

??? success "Answer"
    The classic Nagle plus delayed acknowledgement: the client splits a request into two `write` calls (a length prefix, then the message body), the second write is held by Nagle, and it waits out the other end's 40 ms delayed-acknowledgement timer. Only some requests are hit because Nagle only applies while the previous packet is still unacknowledged.

    To confirm: capture with `tcpdump` and look for a 40 ms ACK between the two data packets; or change the client to write once and see whether the anomaly disappears. To fix: turn on `TCP_NODELAY` and combine the header and the body into one send (or use `writev`).

**3. Where the back pressure goes.** A streaming inference service has a client that opens a connection and then reads nothing at all. Based on this chapter's experiment, roughly how much can the server write before it blocks? If the server writes blockingly and the write sits inside the generation loop, what happens? Give two fixes.

??? success "Answer"
    Roughly the server's send buffer plus the client's receive buffer, tens to hundreds of kilobytes, which at a dozen-odd bytes per token is a few thousand to a few tens of thousands of tokens.

    With the write inside the generation loop, that request's write blocks the whole loop: no other request's tokens go out either, the GPU idles, and everyone's latency degrades together. The fixes: (1) non-blocking writes with a pending queue per connection, sent by an I/O thread or an event loop, disconnecting the connection when its queue exceeds a cap; (2) a timeout on the write, cancelling the request and freeing the KV cache it holds when it expires. The two are usually used together.

**4. Moving KV between data centres.** Two data centres with a 10 ms round-trip time and a 25 Gb/s link. A 4 GB KV cache has to move from A to B. How long does it take over one TCP connection with a 4 MB window ceiling? What would fill the link?

??? success "Answer"
    One connection's throughput ceiling is 4 MB / 10 ms = 400 MB/s, about 3.4 Gb/s, so 4 GB takes about 10 s and uses 13% of the link.

    To fill it: (1) raise the window above the bandwidth-delay product (25 Gb/s x 10 ms is about 31 MB, which needs window scaling and a high enough `tcp_rmem` ceiling); (2) or open several connections in parallel (8 connections with a 4 MB window suffice), which is what transfer tools normally do; (3) in production, KV transfer usually goes over RDMA, handing reliable transport to the network card and bypassing both the kernel stack and the window limit (see [The KV transfer engine](serving://comm/kv-storage/) in the inference handbook).

## Summary {#小结}

- [x] An HTTPS request on a new connection takes at least 4 round trips (DNS, handshake, TLS, request); reuse saves the first three and is the best value in network optimisation.
- [x] Nagle meeting delayed acknowledgement reliably costs 40 ms more: turn on `TCP_NODELAY` for interactive connections and write it all at once where you can.
- [x] A `send` that returns is not delivery; a peer that does not read pushes the pressure back through the window (back pressure), so an inference engine separates generating from sending and caps the pending queue.
- [x] Throughput is the window divided by the round-trip time; a large transfer across regions needs a bigger window or several connections, and slow start costs a small response a few extra round trips.
- [x] The side that closes actively enters TIME_WAIT, and a load test over short-lived connections exhausts the ports; investigate with `ss`, `nstat` and `tcpdump`.
