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
