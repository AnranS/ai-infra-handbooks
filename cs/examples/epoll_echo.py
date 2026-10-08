import os
import select
import socket
import sys

N = 200
srv = socket.socket()
srv.bind(("127.0.0.1", 0))
srv.listen(1024)
srv.setblocking(False)
port = srv.getsockname()[1]

sys.stdout.flush()
pid = os.fork()
if pid == 0:                                        # 子进程当客户端：同时开 200 个连接
    socks = [socket.create_connection(("127.0.0.1", port)) for _ in range(N)]
    for i, s in enumerate(socks):
        s.sendall(f"hello {i}\n".encode())
    ok = sum(s.recv(64) == f"hello {i}\n".encode() for i, s in enumerate(socks))
    print(f"客户端：{N} 个连接，收到 {ok} 条正确的回声", flush=True)
    os._exit(0)

ep = select.epoll()                                 # 服务端：一个线程、一个 epoll
ep.register(srv.fileno(), select.EPOLLIN)
conns, echoed = {}, 0
while echoed < N:
    for fd, _ in ep.poll():                         # 只返回就绪的连接
        if fd == srv.fileno():
            while True:                             # 可能一次来了好几个新连接，接到没有为止
                try:
                    c, _ = srv.accept()
                except BlockingIOError:
                    break
                c.setblocking(False)
                ep.register(c.fileno(), select.EPOLLIN)
                conns[c.fileno()] = c
        else:
            data = conns[fd].recv(64)
            if data:
                conns[fd].sendall(data)
                echoed += 1
os.waitpid(pid, 0)
print(f"服务端：一个线程处理了 {len(conns)} 个连接，回声 {echoed} 条")
