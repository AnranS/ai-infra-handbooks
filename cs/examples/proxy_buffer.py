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
