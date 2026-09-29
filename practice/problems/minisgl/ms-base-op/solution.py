import numpy as np


def _join(prefix, name):
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    def _children(self):
        for name, value in self.__dict__.items():
            if not name.startswith("_"):
                yield name, value

    def state_dict(self, prefix="", result=None):
        result = {} if result is None else result
        for name, value in self._children():
            if isinstance(value, np.ndarray):
                result[_join(prefix, name)] = value
            elif isinstance(value, BaseOP):
                value.state_dict(_join(prefix, name), result)
        return result

    def _load(self, sd, prefix):
        for name, value in self._children():
            key = _join(prefix, name)
            if isinstance(value, np.ndarray):
                if key not in sd:
                    raise KeyError(f"state_dict 里缺少 {key}")
                item = sd.pop(key)
                if item.shape != value.shape or item.dtype != value.dtype:
                    raise ValueError(f"{key}：期望 {value.shape} {value.dtype}，实际 {item.shape} {item.dtype}")
                setattr(self, name, item)
            elif isinstance(value, BaseOP):
                value._load(sd, key)

    def load_state_dict(self, state_dict):
        sd = dict(state_dict)
        self._load(sd, "")
        if sd:
            raise KeyError(f"state_dict 里有多余的键：{sorted(sd)}")


class OPList(BaseOP):
    def __init__(self, ops):
        self.op_list = list(ops)

    def state_dict(self, prefix="", result=None):
        result = {} if result is None else result
        for i, op in enumerate(self.op_list):
            op.state_dict(_join(prefix, str(i)), result)
        return result

    def _load(self, sd, prefix):
        for i, op in enumerate(self.op_list):
            op._load(sd, _join(prefix, str(i)))
