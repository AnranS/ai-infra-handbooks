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
