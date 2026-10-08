import os
import sys

import zmq

ADDR = f"ipc:///tmp/cs-handbook-{os.getpid()}.ipc"       # 带上 pid：两个人同时跑、或者 CI 和本地同时跑也不会撞到同一个套接字文件
N = 10000

sys.stdout.flush()
pid = os.fork()
if pid == 0:                                        # 子进程：PUSH 端，先 connect，这时对面还没有 bind
    sock = zmq.Context().socket(zmq.PUSH)
    sock.connect(ADDR)
    for i in range(N):
        sock.send(i.to_bytes(4, "little"))          # 发出去的消息先排在本地队列里，连上以后自动送达
    sock.close(linger=-1)                           # 关闭前等队列里的消息都发完
    os._exit(0)

pull = zmq.Context().socket(zmq.PULL)               # 父进程：PULL 端，晚一点才 bind
pull.bind(ADDR)
got = [int.from_bytes(pull.recv(), "little") for _ in range(N)]
os.waitpid(pid, 0)
print(f"收到 {len(got)} 条消息，顺序和内容都对：", got == list(range(N)))
print("先 connect、后 bind 也能工作：ZMQ 在后台自动建立（和重建）连接")
