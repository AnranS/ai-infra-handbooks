import math


def is_contiguous(shape, stride):
    if math.prod(shape) == 0:
        return True
    expected = 1
    for n, s in zip(reversed(shape), reversed(stride)):
        if n == 1:                                   # 长度为 1 的维度，stride 是多少都不影响布局
            continue
        if s != expected:
            return False
        expected *= n
    return True


def _infer(numel, new_shape):
    new_shape = list(new_shape)
    if new_shape.count(-1) > 1:
        raise ValueError("最多只能有一个 -1")
    if -1 in new_shape:
        known = math.prod(n for n in new_shape if n != -1)
        if known == 0 or numel % known:
            raise ValueError(f"{numel} 个元素不能排成 {tuple(new_shape)}")
        new_shape[new_shape.index(-1)] = numel // known
    if math.prod(new_shape) != numel:
        raise ValueError(f"{numel} 个元素不能排成 {tuple(new_shape)}")
    return new_shape


def view_strides(shape, stride, new_shape):
    """PyTorch 的 computeStride：把原张量按"内存上连续的块"分组，新形状的每一维必须落在某一块里"""
    new_shape = _infer(math.prod(shape), new_shape)
    if not shape:
        return tuple(1 for _ in new_shape)
    new_stride = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    chunk_base = stride[-1]                          # 当前这块最内层的 stride
    tensor_numel = view_numel = 1
    for d in range(len(shape) - 1, -1, -1):
        tensor_numel *= shape[d]
        # 走到一块的开头：d 是第 0 维，或者第 d-1 维和第 d 维在内存上接不上
        if d == 0 or (shape[d - 1] != 1 and stride[d - 1] != tensor_numel * chunk_base):
            while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
                new_stride[view_d] = view_numel * chunk_base
                view_numel *= new_shape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:           # 新形状的某一维跨过了两块：做不成视图
                return None
            if d > 0:
                chunk_base = stride[d - 1]
                tensor_numel = view_numel = 1
    if view_d != -1:
        return None
    return tuple(new_stride)


def expand_strides(shape, stride, new_shape):
    if len(new_shape) < len(shape):
        raise ValueError("expand 不能减少维数")
    lead = len(new_shape) - len(shape)
    out = []
    for i, n in enumerate(new_shape):
        if i < lead:                                 # 新加在前面的维度：stride 为 0，所有下标指向同一份数据
            if n < 0:
                raise ValueError("新加的维度不能是 -1")
            out.append(0)
            continue
        old_n, old_s = shape[i - lead], stride[i - lead]
        if n == -1 or n == old_n:
            out.append(old_s)
        elif old_n == 1:
            out.append(0)                            # 长度为 1 的维度广播：stride 置 0
        else:
            raise ValueError(f"第 {i} 维长度 {old_n} 不能扩展成 {n}")
    return tuple(out)
