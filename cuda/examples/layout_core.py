"""layout_core.py —— CuTe 的布局：形状 + 步长，坐标到下标的映射，以及合并（coalesce）。

布局 = 形状（Shape）: 步长（Stride），两者都可以是嵌套的元组。布局是一个函数：把坐标映射成一维下标。
一维坐标按"第 0 维变化最快"（colexicographic，和列优先一致）拆成多维坐标，这是 CuTe 的约定。
打印格式和 CuTe 的 print 相同（只是不区分编译期常量）。
"""

from math import prod


def is_tuple(x):
    return isinstance(x, tuple)


def flatten(t):
    return sum((flatten(x) for x in t), ()) if is_tuple(t) else (t,)


def size(shape):
    return prod(flatten(shape))


def unflatten(flat, like):
    """按 like 的嵌套结构把扁平的元组重新分组，返回 (结果, 剩下的元素)"""
    if not is_tuple(like):
        return flat[0], flat[1:]
    out = []
    for x in like:
        y, flat = unflatten(flat, x)
        out.append(y)
    return tuple(out), flat


def crd2idx(crd, shape, stride):
    if is_tuple(crd):                                         # 元组坐标：逐维映射再相加
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    if is_tuple(shape):                                       # 整数坐标落在多维的一维上：先拆开（第 0 维变化最快）
        idx = 0
        for s, d in zip(shape[:-1], stride[:-1]):
            idx += crd2idx(crd % size(s), s, d)
            crd //= size(s)
        return idx + crd2idx(crd, shape[-1], stride[-1])     # 最后一维不取余：超出范围的坐标沿最后一维延伸
    return crd * stride


def fmt(t):
    return "(" + ",".join(fmt(x) for x in t) + ")" if is_tuple(t) else str(t)


class Layout:
    def __init__(self, shape, stride=None):
        if stride is None:                                    # 默认步长：列优先的紧凑布局
            flat, acc = [], 1
            for n in flatten(shape):
                flat.append(acc)
                acc *= n
            stride = unflatten(tuple(flat), shape)[0]
        self.shape, self.stride = shape, stride

    def __call__(self, coord):
        """坐标 → 下标。coord 可以是整数（一维坐标），也可以是和形状对应的（嵌套）元组"""
        return crd2idx(coord, self.shape, self.stride)

    def size(self):                                           # 定义域的大小：有多少个坐标
        return size(self.shape)

    def cosize(self):                                         # 值域的大小：最大下标 + 1
        return self(self.size() - 1) + 1

    def __len__(self):                                        # 秩：顶层有几维（mode）
        return len(self.shape) if is_tuple(self.shape) else 1

    def __getitem__(self, i):                                 # 第 i 维单独拿出来，也是一个布局
        return Layout(self.shape[i], self.stride[i]) if is_tuple(self.shape) else self

    def __eq__(self, other):
        return (self.shape, self.stride) == (other.shape, other.stride)

    def __repr__(self):
        return f"{fmt(self.shape)}:{fmt(self.stride)}"


def make_layout(*layouts):
    """把几个布局并排成一个多维布局"""
    return Layout(tuple(l.shape for l in layouts), tuple(l.stride for l in layouts))


def coalesce(layout):
    """合并：把"接得上"的相邻维（形状 × 步长 = 下一维的步长）并成一维，去掉长度为 1 的维；一维坐标上的函数不变"""
    shape, stride = [1], [0]
    for n, d in zip(flatten(layout.shape), flatten(layout.stride)):
        if n == 1:
            continue
        if shape[-1] == 1:
            shape[-1], stride[-1] = n, d
        elif shape[-1] * stride[-1] == d:
            shape[-1] *= n
        else:
            shape.append(n)
            stride.append(d)
    return Layout(shape[0], stride[0]) if len(shape) == 1 else Layout(tuple(shape), tuple(stride))
