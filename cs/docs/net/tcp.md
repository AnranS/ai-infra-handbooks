# TCP：一个请求的网络之旅

<p class="lead">推理服务的延迟账里，网络那一段常常被忽略：为什么同样的服务，跨城访问的首包慢了 100 毫秒？为什么客户端读得慢会让 GPU 停下来？为什么一个只发几十字节的健康检查偶尔要 40 毫秒？这一章从一条 TCP 连接的建立讲到关闭，用本机实跑的 socket 程序看清握手的往返账、Nagle 算法造成的 40 毫秒延迟、缓冲区满带来的背压，以及带宽延迟积为什么让长距离传输跑不满带宽。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个 HTTPS 请求在拿到第一个响应字节之前，至少要经历几个往返？连接复用能省掉哪几个？
    2. 什么是 Nagle 算法？它和延迟确认（delayed ACK）撞在一起会发生什么？
    3. 发送端调用 `send()` 返回之后，数据到对方了吗？对方一直不读会怎样？
    4. 一条 RTT 为 30 ms 的连接，窗口 256 KB，最多能跑多少带宽？
    5. 主动关闭连接的一方为什么会进入 TIME_WAIT？压测时端口耗尽是怎么回事？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 至少 4 个：DNS 查询 1 个（没缓存时）、TCP 三次握手 1 个、TLS 1.3 握手 1 个（TLS 1.2 要 2 个）、发请求到收到第一个字节 1 个。复用连接省掉握手和 TLS，DNS 有缓存也省掉，只剩最后一个往返。这就是所有客户端都要用连接池的原因。
    2. Nagle 算法：一条连接上还有没被确认的小包时，新的小数据先攒着，等 ACK 回来或者攒够一个 MSS 再发。对端的延迟确认又会把 ACK 拖上几十毫秒（Linux 最多 40 ms）。于是"先写头、再写体、等回应"这种写法会稳定地多花几十毫秒——本章实测 41 ms，打开 `TCP_NODELAY` 后降到 0.02 ms。
    3. 没有。`send()` 只是把数据拷进内核的发送缓冲区就返回。对方不读，它的接收缓冲区会满，通过 TCP 的流量控制把发送端的窗口压到 0，发送缓冲区也随之填满，再 `send()` 就会阻塞（非阻塞 socket 返回 `EAGAIN`）。这就是背压：慢客户端会一路把压力顶回到应用层。
    4. 吞吐 ≈ 窗口 ÷ RTT = 256 KB / 30 ms ≈ 8.7 MB/s ≈ 0.07 Gb/s。要跑满 10 Gb/s、RTT 1 ms 的链路，窗口至少要 1.2 MB。这就是带宽延迟积（BDP）。
    5. TIME_WAIT 是为了吸收网络里可能迟到的重复报文，并确保最后一个 ACK 丢了还能重传，持续 2 倍报文最大生存时间（Linux 上 60 秒）。压测时每个请求新建一条连接，几万个 TIME_WAIT 会占满本地端口范围，出现 `Cannot assign requested address`。解决办法是复用连接（连接池、HTTP keep-alive），而不是去调那些有风险的内核参数。

## 一个请求要花几个往返

先算账。RTT（往返时延）是网络延迟的基本单位：同机房零点几毫秒，同城几毫秒，跨城几十毫秒，跨洲两百毫秒左右。

```python title="handshake.py"
# 一个请求在网络上要花几个往返？（RTT 按跨城 30 ms 算）
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

```text title="输出"
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

![图：新建连接与复用连接各要几个往返](../assets/figures/rtt-ladder.svg){.aig-svg}

两个结论：

- **连接复用是网络优化里性价比最高的一件事**：省掉三个往返，跨城就是 90 ms。推理客户端、网关到推理实例之间，都应该用长连接（HTTP keep-alive、HTTP/2、gRPC）。
- **部署位置决定了网络在延迟里的占比**：同机房时网络可以忽略不计，跨洲时它和整个 TTFT 一样大。对延迟敏感的服务要就近部署（见[生产部署与运维](serving://ops/deploy/)）。

## Nagle 算法与 40 毫秒

TCP 想少发小包：每个包都有 40 字节的 IP + TCP 头，发一个字节的数据要付 41 字节的代价。**Nagle 算法**的规则是——连接上还有没被确认的数据时，新的小块数据先攒着，等 ACK 回来或者攒够一个 MSS（通常 1460 字节）再发。

另一边，接收方有**延迟确认**：收到数据不马上回 ACK，等一小会儿，看看能不能和自己要发的数据一起捎带出去，Linux 上最多等 40 ms。

两者撞在一起就出事。下面这个程序模拟最常见的写法——先写请求头，再写请求体，然后等回应：

```python title="nagle.py"
# 一个请求分两次写（先写头、再写体），服务端收齐才回。Nagle 算法会把第二次写压到第一次的 ACK 之后
import socket
import threading
import time

HOST, N = "127.0.0.1", 50


def server(sock):
    conn, _ = sock.accept()
    with conn:
        while True:
            head = conn.recv(4)                       # 头：4 字节
            if not head:
                return
            body = b""
            while len(body) < 8:
                body += conn.recv(8 - len(body))      # 体：8 字节
            conn.sendall(b"ok")                       # 收齐才回


def round_trips(nodelay):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, 0))
    srv.listen(1)
    t = threading.Thread(target=server, args=(srv,), daemon=True)
    t.start()
    c = socket.create_connection(srv.getsockname())
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 if nodelay else 0)
    for _ in range(3):                                # 预热
        c.sendall(b"head"), c.sendall(b"12345678"), c.recv(2)
    t0 = time.perf_counter()
    for _ in range(N):
        c.sendall(b"head")                            # 第一次写：立刻发出
        c.sendall(b"12345678")                        # 第二次写：Nagle 会等前一个包的 ACK
        c.recv(2)
    dt = (time.perf_counter() - t0) / N * 1e3
    c.close()
    srv.close()
    t.join(timeout=1)
    return dt


print(f"开着 Nagle（默认）：每次往返 {round_trips(False):.3f} ms")
print(f"TCP_NODELAY：      每次往返 {round_trips(True):.3f} ms")
```

```text title="输出（本机示例）"
开着 Nagle（默认）：每次往返 41.059 ms
TCP_NODELAY：      每次往返 0.021 ms
```

第二次写被 Nagle 压住，要等第一个包的 ACK；而对方在等收齐了再回应，于是只有延迟确认超时才发 ACK。每个请求稳定地多花 40 ms，打开 `TCP_NODELAY`（关掉 Nagle）之后降到 0.02 ms。

实践中的规则：

- **交互式的、请求-响应式的连接一律打开 `TCP_NODELAY`**。Nginx、gRPC、Redis 客户端默认都是开的；自己写 socket 代码时要记得设。
- **更根本的做法是一次写完**：把头和体拼成一次 `send`，或者用 `writev` / `sendmsg` 一次提交多段数据。攒够再发，本来就是 Nagle 想做的事。

推理服务里最容易踩到这个坑的地方是流式输出：每生成一个 token 就写一小段（十几个字节），如果没开 `TCP_NODELAY`，token 会被攒在一起、一顿一顿地到达，用户看到的不是"逐字蹦出"而是"一卡一卡"。

## 发送不等于送达：缓冲区与背压

`send()` 返回只意味着"数据拷进了内核的发送缓冲区"。数据真正的去向受两个东西限制：本端的发送缓冲区（`SO_SNDBUF`），和对端通过 ACK 报文告知的**接收窗口**——对端缓冲区里还剩多少空位。对端不读，窗口就会降到 0，发送端只能停下。

看看接收端完全不读时，发送端最多能写进去多少：

```python title="backpressure.py"
# 接收端不读时，发送端最多能写进去多少字节？——发送缓冲区 + 对端接收窗口，再写就阻塞
import socket
import threading

HOST = "127.0.0.1"


def probe(sndbuf, rcvbuf):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)   # 内核实际分配约为设定值的两倍
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
            written += c.send(chunk)                              # 服务端一个字节都不读
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

```text title="输出（本机示例）"
SO_SNDBUF 设为  16 KB（内核给  32 KB）、对端 SO_RCVBUF  16 KB：阻塞前写进去     40 KB
SO_SNDBUF 设为  64 KB（内核给 128 KB）、对端 SO_RCVBUF  64 KB：阻塞前写进去    160 KB
SO_SNDBUF 设为 256 KB（内核给 512 KB）、对端 SO_RCVBUF 256 KB：阻塞前写进去    616 KB
```

写进去的量大约是"发送缓冲区 + 对端接收缓冲区"（内核实际分配的通常是 `setsockopt` 设定值的两倍，用于协议开销）。写满之后，阻塞式 socket 会卡在 `send()` 上，非阻塞 socket 返回 `EAGAIN`，`epoll` 不再报告可写（见 [epoll 与 io_uring](../os/io-models.md)）。

这条链路就是**背压**：客户端读得慢 → 对端接收缓冲区满 → 发送窗口为 0 → 本端发送缓冲区满 → 写操作阻塞或返回 `EAGAIN` → 应用层必须停下来。对推理服务来说，这意味着：

- 一个网络很差、或者干脆不读响应的客户端，会让负责写回结果的那个协程/线程卡住。如果这个线程还兼着别的活（比如调度循环），整个引擎都会被它拖慢——所以推理引擎把"生成"和"发送"分开：结果先进队列，由专门的 I/O 任务负责写出去；
- 队列也不能无限堆。合理的做法是给每个请求的待发队列设上限，超过就断开或丢弃这个请求（慢客户端不应该拖垮服务），并且在监控里区分"生成慢"和"发不出去"；
- 大量慢客户端还会占着内存：每条连接的收发缓冲区加起来几百 KB，一万条连接就是几个 GB。

## 带宽延迟积：窗口决定吞吐

一条连接的吞吐上限是**窗口 ÷ RTT**：发出一窗数据后必须等 ACK 回来才能继续，一个 RTT 最多发一个窗口。

```python title="bdp.py"
# 带宽延迟积：一条 TCP 连接的吞吐 ≈ 窗口大小 ÷ RTT。窗口不够大时，带宽再高也跑不满
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

```text title="输出"
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

要点：

- **跨地域传输必须放大窗口**。Linux 的接收窗口默认会自动调节（`net.ipv4.tcp_rmem`），上限通常几 MB；跨机房搬大文件（模型权重、KV Cache 快照）时，如果吞吐卡在几十 MB/s，先算一下 BDP，再检查窗口上限和是否开了窗口缩放。
- **慢启动让小响应吃亏**。连接刚建立时的拥塞窗口只有 10 个 MSS（约 14 KB），每个 RTT 翻一倍。一个 1 MB 的响应要 7 个 RTT 才发完，跨城就是 200 多毫秒——而这段时间链路根本没跑满。复用连接的另一个好处正是：窗口已经涨上去了。
- **流式输出反而不受这个限制**：每次只发几十字节，窗口从来不是瓶颈，延迟由 RTT 和服务端的生成速度决定。

拥塞控制算法（Linux 默认 CUBIC，也可以换成 BBR）决定窗口怎么涨、丢包时怎么退。数据中心内部还常用 ECN、DCTCP 这类方案，让交换机在真正丢包之前就给主机发信号。RDMA 走的是另一条路：把可靠传输做进网卡，绕开内核协议栈（见[锁页内存、DMA 与 NUMA](../os/pinned-numa.md)和推理手册的 [RDMA 编程模型](serving://comm/rdma/)）。

## 连接的关闭与 TIME_WAIT

关闭要四次挥手：各自发 FIN、各自回 ACK。**主动关闭的一方**进入 TIME_WAIT，等 2 倍报文最大生存时间（Linux 上 60 秒）才真正释放，为的是吸收迟到的重复报文、并保证最后一个 ACK 丢了还能重传。

```python title="reuse.py"
# 每个请求都新建连接 vs 复用一条连接：本机 loopback 上的差别，以及留下多少 TIME_WAIT
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
    c = socket.create_connection((HOST, port))         # 每个请求一条新连接
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.sendall(b"ping")
    c.recv(2)
    c.close()                                          # 主动关闭的一方进入 TIME_WAIT
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

```text title="输出（本机示例）"
每次新建连接：每个请求 0.166 ms，结束后留下 300 个 TIME_WAIT
复用一条连接：每个请求 0.016 ms，快 10 倍
```

本机 loopback 上，新建连接的开销是复用的 10 倍（真实网络上差距更大，因为还多一个 RTT），而且每个短连接都会留下一个 TIME_WAIT。本地端口范围（`net.ipv4.ip_local_port_range`，通常 3 万多个）在高 QPS 的短连接下很快就会用完，表现为 `Cannot assign requested address`。

正确的做法是**复用连接**：客户端用连接池、开 HTTP keep-alive，网关到后端也保持长连接。服务端侧要注意让客户端先关（避免服务端自己积累 TIME_WAIT）。那些"调 `tcp_tw_reuse`、缩短 `tcp_fin_timeout`"的偏方能缓解症状，但在 NAT 环境下有风险，不如先把连接复用做对。

另外几个常见的连接层问题：

| 现象 | 常见原因 |
| --- | --- |
| 连接偶尔被重置（`Connection reset by peer`） | 对端进程崩溃、或者在缓冲区里还有数据时调用了 `close`；也可能是中间的负载均衡器超时 |
| 空闲一段时间后第一个请求失败 | 中间设备（负载均衡器、NAT）悄悄回收了空闲连接。开 TCP keepalive 或应用层心跳，并把客户端的空闲超时设得比中间设备短 |
| 高并发时大量连接卡在 `SYN_RECV` | `listen` 的 backlog 太小或 accept 太慢，握手队列溢出（`ss -lnt` 看 `Send-Q`、`Recv-Q`） |
| 偶发的几百毫秒抖动 | 丢包重传：一次超时重传的最小间隔就有 200 ms 级别 |

排查用的工具：`ss -tan`（连接状态分布）、`ss -lnt`（监听队列）、`netstat -s` 或 `nstat`（重传、丢包计数）、`tcpdump` 抓包、`ping` 和 `mtr` 看 RTT 与丢包。

!!! interview "怎么讲清楚"
    讲"一个请求从客户端到服务端经历了什么"：DNS → TCP 三次握手 → TLS 握手 → 发请求 → 等第一个字节，跨城时每个往返 30 ms，连接复用能省掉前三个。再讲三个容易出问题的点：（1）Nagle 和延迟确认撞在一起会稳定多花 40 ms，交互式连接要开 `TCP_NODELAY`，流式输出尤其要注意；（2）`send` 返回不代表送达，对端不读会通过窗口把压力顶回来，这就是背压，所以推理引擎要把生成和发送分开、给待发队列设上限；（3）吞吐 = 窗口 ÷ RTT，跨地域传大文件要放大窗口，慢启动让小响应多花几个 RTT。最后是连接管理：主动关闭的一方进 TIME_WAIT，短连接压测会端口耗尽，解法是连接池和 keep-alive。

## 练习

**1. 算往返。** 一个用户在上海，推理服务部署在北京（RTT 30 ms），用 HTTPS、TLS 1.3、每次请求新建连接。服务端的 TTFT 是 150 ms，之后每 20 ms 产出一个 token，回答 200 个 token。用户看到第一个字的时间是多少？整个回答花多久？如果改成复用连接呢？

??? success "参考答案"
    新建连接：DNS（假设有缓存，0）+ 握手 30 + TLS 30 + 请求到首字节 30 + 服务端 150 = 240 ms 看到第一个字；之后 200 个 token × 20 ms = 4000 ms，加上最后一段的单程传输，总共约 4.2 s。

    复用连接：30 + 150 = 180 ms 看到第一个字，省了 60 ms；总时长几乎不变（流式输出的后续 token 不再受握手影响）。TTFT 改善 25%，对体感的影响比总时长大得多。

**2. 诊断 40 毫秒。** 一个内部服务的 p50 延迟 0.5 ms，但有大约 15% 的请求恰好在 40 ms 左右，重试也是。可能是什么原因？怎么验证？怎么修？

??? success "参考答案"
    典型的 Nagle + 延迟确认：客户端把一个请求分成两次 `write`（比如先写长度前缀、再写消息体），第二次写被 Nagle 压住，等对方延迟确认的 40 ms 超时。之所以只有一部分请求中招，是因为只有"上一个包还没被确认"时 Nagle 才生效。

    验证：`tcpdump` 抓包看两个数据包之间是否隔着一个 40 ms 的 ACK；或者把客户端改成一次写完，看异常是否消失。修法：打开 `TCP_NODELAY`，并且把头和体合成一次发送（或用 `writev`）。

**3. 背压的去向。** 一个流式推理服务，某个客户端建立连接后完全不读数据。按本章的实验，服务端大约能写进去多少数据之后开始阻塞？如果服务端用阻塞式写、并且写操作就在生成循环里，会发生什么？给出两种修法。

??? success "参考答案"
    大约是"服务端发送缓冲区 + 客户端接收缓冲区"，几十 KB 到几百 KB，按每个 token 十几字节算，几千到几万个 token 就写满了。

    如果写在生成循环里，这个请求的写操作会阻塞整个循环：其他请求的 token 也发不出去，GPU 空转，所有人的延迟一起变差。修法：（1）改成非阻塞写 + 每个连接一个待发队列，由 I/O 线程或事件循环负责发送，队列超过上限就断开这个连接；（2）给写操作设超时，超时就取消这个请求并释放它占的 KV Cache。两者通常一起用。

**4. 跨机房传 KV。** 两个机房之间 RTT 10 ms、链路 25 Gb/s。要把一个 4 GB 的 KV Cache 从 A 搬到 B。单条 TCP 连接、窗口上限 4 MB 时要多久？怎样才能用满带宽？

??? success "参考答案"
    单条连接的吞吐上限 = 4 MB / 10 ms = 400 MB/s ≈ 3.4 Gb/s，4 GB 要约 10 s，只用了链路的 13%。

    要用满：（1）把窗口开大到 BDP 以上（25 Gb/s × 10 ms ≈ 31 MB，需要窗口缩放和足够的 `tcp_rmem` 上限）；（2）或者开多条并行连接（8 条 4 MB 窗口的连接就够），这也是各种传输工具的常规做法；（3）生产上的 KV 传输一般直接走 RDMA，把可靠传输交给网卡，绕开内核协议栈和窗口限制（见推理手册的 [KV 传输引擎](serving://comm/kv-storage/)）。

## 小结

- [x] 一个新连接的 HTTPS 请求至少 4 个往返（DNS、握手、TLS、请求）；连接复用省掉前三个，是性价比最高的网络优化。
- [x] Nagle 遇上延迟确认会稳定多花 40 ms：交互式连接开 `TCP_NODELAY`，并且尽量一次写完。
- [x] `send` 返回不等于送达；对端不读会通过窗口把压力顶回来（背压），推理引擎要把生成与发送分开、给待发队列设上限。
- [x] 吞吐 = 窗口 ÷ RTT；跨地域传大数据要放大窗口或开多条连接，慢启动让小响应多花几个 RTT。
- [x] 主动关闭的一方进入 TIME_WAIT，短连接压测会端口耗尽；排查用 `ss`、`nstat`、`tcpdump`。
