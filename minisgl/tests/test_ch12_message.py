import threading

import torch
from minisgl.core import SamplingParams
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    BaseTokenizerMsg,
    TokenizeMsg,
    UserMsg,
)
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue


def test_backend_message_roundtrip_with_tensor_and_nesting():
    msg = BatchBackendMsg(data=[
        UserMsg(uid=3, input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
                sampling_params=SamplingParams(temperature=0.5, max_tokens=7)),
        UserMsg(uid=4, input_ids=torch.tensor([9], dtype=torch.int32), sampling_params=SamplingParams()),
    ])
    encoded = msg.encoder()
    assert encoded["__type__"] == "BatchBackendMsg"
    assert encoded["data"][0]["input_ids"]["__type__"] == "Tensor"
    decoded = BaseBackendMsg.decoder(encoded)
    assert isinstance(decoded, BatchBackendMsg) and decoded.data[0].sampling_params.max_tokens == 7
    assert torch.equal(decoded.data[0].input_ids, msg.data[0].input_ids)


def test_zmq_push_pull_over_ipc(ipc_dir):
    addr = f"ipc://{ipc_dir}/q"
    pull = ZmqPullQueue(addr, create=True, decoder=BaseTokenizerMsg.decoder)
    push = ZmqPushQueue(addr, create=False, encoder=BaseTokenizerMsg.encoder)
    msgs = [TokenizeMsg(uid=1, text=[{"role": "user", "content": "你好"}], sampling_params=SamplingParams()),
            DetokenizeMsg(uid=1, next_token=42, finished=True)]
    threading.Thread(target=lambda: [push.put(m) for m in msgs]).start()
    got = [pull.get(), pull.get()]
    assert got == msgs and pull.empty()
    push.stop()
    pull.stop()


def test_pub_waits_for_subscribers_before_first_message(ipc_dir):
    """XPUB 等订阅者到齐后立即发布：不 sleep 也不会丢第一条消息（慢订阅者问题）。"""
    addr = f"ipc://{ipc_dir}/pub"
    for _ in range(20):
        pub = ZmqPubQueue(addr, create=True, encoder=BaseTokenizerMsg.encoder)
        subs = [ZmqSubQueue(addr, create=False, decoder=BaseTokenizerMsg.decoder) for _ in range(3)]
        pub.wait_for_subscribers(3)
        pub.put(DetokenizeMsg(uid=5, next_token=1, finished=False))
        assert [s.get().uid for s in subs] == [5, 5, 5]
        for q in [pub, *subs]:
            q.stop()
