"""第 12 章：一条消息从对象到字节再回到对象；ZMQ 的 PUSH/PULL 与 PUB/SUB。"""

import threading

import msgpack
import torch
from minisgl.core import SamplingParams
from minisgl.message import BaseBackendMsg, BatchBackendMsg, UserMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue

msg = BatchBackendMsg(data=[
    UserMsg(uid=7, input_ids=torch.tensor([9707, 11, 1879], dtype=torch.int32),
            sampling_params=SamplingParams(temperature=0.6, max_tokens=64)),
])
d = msg.encoder()
print("encoder() 得到的字典:")
print(" ", {k: v for k, v in d.items() if k != "data"}, "data[0] =")
inner = dict(d["data"][0])
print("   ", inner)
raw = msgpack.packb(d, use_bin_type=True)
print(f"msgpack 之后 {len(raw)} 字节")
back = BaseBackendMsg.decoder(msgpack.unpackb(raw, raw=False))
print("解码回来:", back.data[0].uid, back.data[0].input_ids, back.data[0].sampling_params)

# PUSH/PULL：一对一的队列；PUB/SUB：一对多的广播
addr = "ipc:///tmp/minisgl_example_pushpull"
pull = ZmqPullQueue(addr, create=True, decoder=BaseBackendMsg.decoder)
push = ZmqPushQueue(addr, create=False, encoder=BaseBackendMsg.encoder)
threading.Thread(target=lambda: [push.put(UserMsg(uid=i, input_ids=torch.tensor([i], dtype=torch.int32),
                                                  sampling_params=SamplingParams())) for i in range(3)]).start()
print("PULL 收到 uid:", [pull.get().uid for _ in range(3)], " 队列已空:", pull.empty())

paddr = "ipc:///tmp/minisgl_example_pubsub"
pub = ZmqPubQueue(paddr, create=True, encoder=BaseBackendMsg.encoder)
subs = [ZmqSubQueue(paddr, create=False, decoder=BaseBackendMsg.decoder) for _ in range(2)]
pub.wait_for_subscribers(2)  # 订阅生效之前发布的消息会被丢弃，所以先等两个订阅者到齐
pub.put_raw(raw)  # rank 0 把收到的原始字节原样转发，不用重新编码
print("两个 SUB 都收到了:", [s.get().data[0].uid for s in subs])
for q in [pull, push, pub, *subs]:
    q.stop()
