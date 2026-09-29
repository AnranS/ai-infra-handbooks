import numpy as np


def serialize(obj):
    return obj.__dict__            # 没有递归，也没有处理数组


def deserialize(cls_map, data):
    pass
