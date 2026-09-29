import numpy as np


def _join(prefix, name):
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    def state_dict(self, prefix=""):
        return {_join(prefix, k): v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}   # 没有递归

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)


class OPList(BaseOP):
    def __init__(self, ops):
        self.op_list = list(ops)
