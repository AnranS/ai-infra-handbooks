import numpy as np

BASIC = (int, float, str, type(None), bool, bytes)


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(serialize(v) for v in obj)
    if isinstance(obj, BASIC):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.ndim != 1:
            raise ValueError("只支持一维数组")
        return {"__type__": "Tensor", "buffer": obj.tobytes(), "dtype": str(obj.dtype)}
    result = {"__type__": type(obj).__name__}
    for k, v in obj.__dict__.items():
        result[k] = serialize(v)
    return result


def deserialize(cls_map, data):
    if isinstance(data, dict):
        if "__type__" not in data:
            return {k: deserialize(cls_map, v) for k, v in data.items()}
        name = data["__type__"]
        if name == "Tensor":
            return np.frombuffer(data["buffer"], dtype=np.dtype(data["dtype"])).copy()
        kwargs = {k: deserialize(cls_map, v) for k, v in data.items() if k != "__type__"}
        return cls_map[name](**kwargs)
    if isinstance(data, (list, tuple)):
        return type(data)(deserialize(cls_map, v) for v in data)
    return data
