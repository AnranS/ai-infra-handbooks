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
