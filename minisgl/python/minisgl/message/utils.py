"""消息的序列化：dataclass ↔ 由基本类型组成的 dict，再交给 msgpack 变成字节。

每个对象序列化成 {"__type__": 类名, 字段...}；1 维张量存成原始字节加 dtype。
反序列化时按类名在给定的类表（通常是定义消息的模块的 globals()）里找到类，递归还原字段。
"""

from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    if isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    return serialize_type(value)


def serialize_type(obj: Any) -> Dict:
    if isinstance(obj, torch.Tensor):
        assert obj.dim() == 1, "only 1D tensors are supported"
        return {"__type__": "Tensor", "buffer": obj.numpy().tobytes(), "dtype": str(obj.dtype)}
    result: Dict[str, Any] = {"__type__": type(obj).__name__}
    for k, v in obj.__dict__.items():
        result[k] = _serialize_any(v)
    return result


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    return data


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    if type_name == "Tensor":
        np_dtype = getattr(np, data["dtype"].replace("torch.", ""))
        return torch.from_numpy(np.frombuffer(data["buffer"], dtype=np_dtype).copy())
    kwargs = {k: _deserialize_any(cls_map, v) for k, v in data.items() if k != "__type__"}
    return cls_map[type_name](**kwargs)
