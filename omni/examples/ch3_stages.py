"""第三章的玩具 stage：每个 stage 都是一个 SimpleScheduler 包着一个普通函数。"""
import time

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_split():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"text": payload.data["raw_inputs"]}
        return payload
    return SimpleScheduler(fn)


def create_text():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"upper": payload.data["text"].upper()}
        return payload
    return SimpleScheduler(fn)


def create_audio():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"samples": len(payload.data["text"]) * 100}
        return payload
    return SimpleScheduler(fn)


def route_split(request_id, output):
    """动态路由：只有请求要了音频才发给 audio。"""
    return ["text", "audio"] if output.request.params.get("want_audio") else ["text"]


def resolve_terminals(request):
    """按请求决定 Coordinator 要等哪几个终点。"""
    return ["text", "audio"] if request.params.get("want_audio") else ["text"]


def create_slow():
    def fn(payload: StagePayload) -> StagePayload:
        time.sleep(3)
        return payload
    return SimpleScheduler(fn)
