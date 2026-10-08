import os
import socket
import sys


def ns(kind):
    return os.readlink(f"/proc/self/ns/{kind}")    # 形如 net:[4026531840]，方括号里是命名空间的编号


parent_net = ns("net")
r, w = os.pipe()
sys.stdout.flush()
if os.fork() == 0:
    same = ns("net") == parent_net
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)  # 新建一个用户命名空间（不需要 root）和一个网络命名空间
    nics = [name for _, name in socket.if_nameindex()]
    os.write(w, f"{same}|{ns('net') != parent_net}|{nics}".encode())
    os._exit(0)
os.wait()
same, differ, nics = os.read(r, 4096).decode().split("|")
print("fork 出的子进程默认和父进程在同一个网络命名空间：", same)
print("unshare 之后换了一个新的网络命名空间：", differ)
print("新的网络命名空间里只有这些网卡：", nics)
