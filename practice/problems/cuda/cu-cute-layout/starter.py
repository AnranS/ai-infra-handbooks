def crd2idx(crd, shape, stride):
    """坐标 → 下标。crd 可以是整数，也可以是和 shape 结构对应的（嵌套）元组"""
    if isinstance(crd, tuple):
        return sum(c * d for c, d in zip(crd, stride))   # 没有处理嵌套的维，也没有处理整数坐标
    return crd * stride


def coalesce(shape, stride):
    pass


def complement(shape, stride, max_idx):
    pass


def composition(a, b):
    pass
