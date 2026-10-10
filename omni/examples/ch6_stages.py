"""src 产生一个 4 MB 的张量和一两段流，分别发给同进程的 near 和另一个进程的 far。"""
import queue

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler


class SrcScheduler:
    def __init__(self, big_chunk: bool):
        self.inbox, self.outbox, self.running = queue.Queue(), queue.Queue(), False
        self.big_chunk = big_chunk

    def warm_up_serving_thread(self):
        pass

    def start(self):
        self.running = True
        while self.running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg.type != "new_request":
                continue
            chunks = [torch.arange(8)]                                   # 64 字节
            if self.big_chunk:
                chunks.append(torch.zeros(16384, dtype=torch.float32))   # 64 KB
            for chunk in chunks:
                self.outbox.put(OutgoingMessage(msg.request_id, "stream", data=chunk))
            payload = msg.data
            payload.data = {"features": torch.ones(1024, 1024)}   # 4 MB 的"编码器输出"
            self.outbox.put(OutgoingMessage(msg.request_id, "result", data=payload))

    def stop(self):
        self.running = False

    def abort(self, request_id):
        pass


def create_src(big_chunk: bool = False):
    return SrcScheduler(big_chunk)


def project(payload: StagePayload) -> StagePayload:
    return StagePayload(payload.request_id, payload.request, {"features": payload.data["features"]})


def create_near():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"near_sum": int(payload.data["features"].sum())}
        return payload
    return SimpleScheduler(fn)


class FarScheduler(StreamingSimpleScheduler):
    def __init__(self):
        super().__init__(compute_fn=None)
        self.sizes: dict[str, list[int]] = {}

    def is_streaming_payload(self, payload):
        return True

    def on_stream_chunk(self, request_id, item):
        self.sizes.setdefault(request_id, []).append(item.data.numel())
        return []

    def on_stream_done(self, request_id):
        payload = self.stream_payloads[request_id]
        payload.data = {"far_chunks": self.sizes.pop(request_id, []),
                        "far_sum": int(payload.data["features"].sum())}
        return [OutgoingMessage(request_id, "result", data=payload)]


def create_far():
    return FarScheduler()
