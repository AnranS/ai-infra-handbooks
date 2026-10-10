"""一个会攒批的玩具编码器：batch_compute_fn 一次处理一批，并把批大小写进结果。"""
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_encoder(max_batch_wait_ms: float = 200, batch_wait_when_idle: bool = True):
    def one(payload: StagePayload) -> StagePayload:
        payload.data = {"batch_size": 1}
        return payload

    def batch(payloads: list[StagePayload]) -> list[StagePayload]:
        for p in payloads:
            p.data = {"batch_size": len(payloads)}
        return payloads

    return SimpleScheduler(one, batch_compute_fn=batch, max_batch_size=8,
                           max_batch_wait_ms=max_batch_wait_ms, batch_wait_when_idle=batch_wait_when_idle)
