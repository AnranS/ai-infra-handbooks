---
title: 估算：网络往返与流式延迟
chapter: net/tcp.md
difficulty: 简单
tags: [估算, RTT, TTFT, 带宽延迟积]
---
把一次流式请求的网络账算清楚。实现：

1. `first_byte_ms(rtt_ms, server_ms, reuse=False, tls=True)`：用户看到第一个字的时间。新建连接要 TCP 握手 1 个 RTT、TLS 1.3 握手 1 个 RTT（`tls=False` 时没有），然后是发请求到第一个字节的 1 个 RTT，服务端处理时间 `server_ms` 包含在最后这一段里（即最后一段是 `rtt_ms + server_ms`）；`reuse=True` 时省掉握手和 TLS；
2. `stream_total_ms(rtt_ms, server_ms, tokens, per_token_ms, reuse=False, tls=True)`：整个回答收完的时间（首字时间 + 之后每个 token 的间隔）；
3. `throughput_mbps(window_kb, rtt_ms)`：一条 TCP 连接的吞吐上限（MB/s，1 MB = 1e6 字节）；
4. `window_for_mbps(target_mbps, rtt_ms)`：要达到目标吞吐，窗口至少要多大（KB，向上取整）；
5. `slow_start_rtts(response_kb, init_window_kb=14)`：慢启动下发完这么大的响应要几个 RTT（初始窗口 `init_window_kb`，每个 RTT 翻倍）。

```python
first_byte_ms(30, 150)                     # 240.0：握手 30 + TLS 30 + (30 + 150)
first_byte_ms(30, 150, reuse=True)         # 180.0
round(throughput_mbps(256, 30), 2)         # 8.74
window_for_mbps(1000, 1)                   # 977
slow_start_rtts(1024)                      # 7
```

<!-- 题解 -->
第 1、2 题是 TTFT 的账：新建连接比复用多两个 RTT，跨城就是 60 ms。流式输出的总时长几乎只由 token 数决定，所以握手开销对总时长影响很小，对**首字时间**影响很大——而首字时间才是用户的体感。

第 3、4 题是带宽延迟积：一条连接每个 RTT 最多发一个窗口，吞吐 = 窗口 ÷ RTT。跨机房搬 KV Cache 时吞吐卡住，先算这个数，再去查窗口上限有没有开够。

第 5 题是慢启动：初始窗口约 10 个 MSS（14 KB），每个 RTT 翻倍，1 MB 的响应要 7 个 RTT 才发完。这也是复用连接的另一个好处：窗口已经涨上去了。
