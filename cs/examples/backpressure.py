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
