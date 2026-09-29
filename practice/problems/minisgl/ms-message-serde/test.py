from dataclasses import dataclass, field

import numpy as np

from checker import check, raises
from solution import deserialize, serialize


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 16


@dataclass
class UserMsg:
    uid: int
    input_ids: np.ndarray
    sampling_params: SamplingParams


@dataclass
class DetokenizeMsg:
    uid: int
    next_token: int
    finished: bool


@dataclass
class BatchMsg:
    data: list
    meta: dict = field(default_factory=dict)
    pair: tuple = ()


CLS = {"UserMsg": UserMsg, "SamplingParams": SamplingParams, "DetokenizeMsg": DetokenizeMsg, "BatchMsg": BatchMsg}


def only_basic(x):
    if isinstance(x, dict):
        return all(isinstance(k, str) for k in x) and all(only_basic(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(only_basic(v) for v in x)
    return isinstance(x, (int, float, str, type(None), bool, bytes))


def test_example():
    msg = UserMsg(7, np.array([1, 2, 3], dtype=np.int32), SamplingParams(0.7, 128))
    data = serialize(msg)
    assert only_basic(data), f"序列化结果里还有非基本类型：{data}"
    check(data["__type__"], "UserMsg", "__type__")
    back = deserialize(CLS, data)
    check((back.uid, back.sampling_params), (7, SamplingParams(0.7, 128)), "还原的字段")
    check((back.input_ids.tolist(), back.input_ids.dtype), ([1, 2, 3], np.dtype(np.int32)), "还原的数组")


def test_nested_containers():
    msg = BatchMsg([DetokenizeMsg(1, 42, False), DetokenizeMsg(2, 7, True)], {"k": [1, 2.5, None, "x"]},
                   (np.arange(4, dtype=np.float32), b"raw"))
    data = serialize(msg)
    assert only_basic(data)
    check(type(data["pair"]), tuple, "元组保持元组")
    back = deserialize(CLS, data)
    check(back.data, [DetokenizeMsg(1, 42, False), DetokenizeMsg(2, 7, True)], "列表里的 dataclass")
    check(back.meta, {"k": [1, 2.5, None, "x"]}, "普通字典")
    check(back.pair[0].tolist(), [0.0, 1.0, 2.0, 3.0], "元组里的数组")
    check(back.pair[1], b"raw", "bytes")


def test_array_is_writable_copy():
    data = serialize(UserMsg(1, np.array([5, 6], dtype=np.int64), SamplingParams()))
    data["input_ids"]["buffer"] = bytes(data["input_ids"]["buffer"])       # 模拟 msgpack 解出的只读 bytes
    back = deserialize(CLS, data)
    back.input_ids[0] = 9
    check(back.input_ids.tolist(), [9, 6], "还原的数组可以写")


def test_reject_2d():
    with raises(ValueError, "二维数组"):
        serialize(UserMsg(1, np.zeros((2, 2)), SamplingParams()))
