"""扇出 + 扇入：split 把一份输出分给 upper 和 count，join 等两边都到齐再合并。"""
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_split():
    def fn(payload: StagePayload) -> StagePayload:
        text = payload.data["raw_inputs"]
        payload.data = {"text": text, "words": text.split()}
        return payload
    return SimpleScheduler(fn)


def project_to_upper(payload: StagePayload) -> StagePayload:
    """投影：upper 只需要原文。"""
    return StagePayload(payload.request_id, payload.request, {"text": payload.data["text"]})


def project_to_count(payload: StagePayload) -> StagePayload:
    """投影：count 只需要分好的词。"""
    return StagePayload(payload.request_id, payload.request, {"words": list(payload.data["words"])})


def create_upper():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"upper": payload.data["text"].upper(), "upper_saw": sorted(payload.data)}
        return payload
    return SimpleScheduler(fn)


def create_count():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"count": len(payload.data["words"]), "count_saw": sorted(payload.data)}
        return payload
    return SimpleScheduler(fn)


def merge(inputs: dict[str, StagePayload]) -> StagePayload:
    """inputs 是 {来源 stage 名: payload}；到达顺序不固定，所以按名字排序再合并。"""
    first = next(iter(inputs.values()))
    data = {"sources": sorted(inputs)}
    for name in sorted(inputs):
        data.update(inputs[name].data)
    return StagePayload(first.request_id, first.request, data)


def create_join():
    return SimpleScheduler(lambda payload: payload)
