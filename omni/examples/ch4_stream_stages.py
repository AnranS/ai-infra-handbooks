"""一个"会说话"的玩具：talker 每步吐一个码（流式），vocoder 边收边"解码"。"""
import queue

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler


class ToyTalkerScheduler:
    """自己实现 StageScheduler 协议：inbox / outbox / start / stop / abort。"""

    def __init__(self, steps: int = 3):
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.steps = steps
        self.running = False

    def warm_up_serving_thread(self) -> None:
        pass

    def start(self) -> None:
        self.running = True
        while self.running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg.type != "new_request":
                continue
            payload: StagePayload = msg.data
            for step in range(self.steps):           # 自回归：一步一个码，立刻往下游推
                codes = torch.tensor([step * 10 + 1, step * 10 + 2])
                self.outbox.put(OutgoingMessage(msg.request_id, "stream", data=codes,
                                                metadata={"step": step}))
            payload.data = {"num_steps": self.steps}
            self.outbox.put(OutgoingMessage(msg.request_id, "result", data=payload))

    def stop(self) -> None:
        self.running = False

    def abort(self, request_id: str) -> None:
        pass


def create_talker():
    return ToyTalkerScheduler()


class ToyVocoderScheduler(StreamingSimpleScheduler):
    """收到一个码就"解码"成一段音频（这里是码乘 100），流式发给客户端。"""

    def __init__(self):
        super().__init__(compute_fn=None)
        self.decoded: dict[str, list[int]] = {}

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return True

    def on_stream_chunk(self, request_id, item):
        samples = (item.data * 100).tolist()
        self.decoded.setdefault(request_id, []).extend(samples)
        return [OutgoingMessage(request_id, "stream", data={"samples": samples},
                                metadata={"modality": "audio"})]

    def on_stream_done(self, request_id):
        payload = self.stream_payloads[request_id]
        payload.data = {"total_samples": len(self.decoded.pop(request_id, []))}
        return [OutgoingMessage(request_id, "result", data=payload)]


def create_vocoder():
    return ToyVocoderScheduler()
