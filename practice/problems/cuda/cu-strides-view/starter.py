import math


def is_contiguous(shape, stride):
    expected = 1
    for n, s in zip(reversed(shape), reversed(stride)):
        if s != expected:                              # 没有跳过长度为 1 的维度
            return False
        expected *= n
    return True


def view_strides(shape, stride, new_shape):
    if not is_contiguous(shape, stride):               # 太保守：很多不连续的张量也能 view
        return None
    new_stride, acc = [], 1
    for n in reversed(new_shape):
        new_stride.append(acc)
        acc *= n
    return tuple(reversed(new_stride))


def expand_strides(shape, stride, new_shape):
    pass
