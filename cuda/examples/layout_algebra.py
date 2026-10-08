"""layout_algebra.py —— 布局代数：复合、补集、逻辑划分。算法与 CUTLASS 自带的 pycute 一致，另外像 CuTe 一样检查整除条件。"""

from layout_core import Layout, coalesce, flatten, is_tuple, make_layout


def composition(a, b):
    """复合：R(c) = A(B(c))，R 的形状就是 B 的形状。B 也可以是逐维作用的元组（每一维一个布局）"""
    if isinstance(b, tuple):                                  # 逐维复合：A 的第 i 维和 B 的第 i 维复合
        return make_layout(*[composition(a[i], b[i]) for i in range(len(b))], *[a[i] for i in range(len(b), len(a))])
    if is_tuple(b.shape):                                     # B 有多个维：每一维分别和 A 复合
        return make_layout(*[composition(a, b[i]) for i in range(len(b))])
    if b.stride == 0:
        return Layout(b.shape, 0)
    rest_n, rest_d = b.shape, b.stride                        # B = n:d —— 在 A 里每隔 d 个取一个，一共取 n 个
    shape, stride = [], []
    fa = coalesce(a)
    fn, fd = flatten(fa.shape), flatten(fa.stride)
    for n, d in zip(fn[:-1], fd[:-1]):                        # 逐维"先跳过 d，再取出 n"
        assert n % rest_d == 0 or rest_d % n == 0, "步长不整除，复合没有定义"
        take = min(max(1, n // rest_d), rest_n)
        assert rest_n % take == 0, "形状不整除，复合没有定义"
        if take != 1:
            shape.append(take)
            stride.append(rest_d * d)
        rest_n //= take
        rest_d = -(-rest_d // n)
    if rest_n != 1 or not shape:                              # A 的最后一维可以无限延伸
        shape.append(rest_n)
        stride.append(rest_d * fd[-1])
    return Layout(shape[0], stride[0]) if len(shape) == 1 else Layout(tuple(shape), tuple(stride))


def complement(layout, max_idx=1):
    """补集：A 没有覆盖的下标怎么排。(A, 补集) 合起来恰好把 [0, max_idx) 每个下标覆盖一次"""
    shape, stride, cur = [], [], 1
    for d, n in sorted(zip(flatten(layout.stride), flatten(layout.shape))):
        if d == 0 or n == 1:
            continue
        assert d % cur == 0, "A 的各维互相交叠，补集没有定义"
        shape.append(d // cur)
        stride.append(cur)
        cur = n * d
    shape.append(-(-max_idx // cur))
    stride.append(cur)
    return coalesce(Layout(tuple(shape), tuple(stride)))


def logical_divide(a, b):
    """逻辑划分：按 B 把 A 切成块。第 0 维是块内（B 选中的元素），第 1 维是第几块。B 可以是逐维作用的元组"""
    if isinstance(b, tuple):
        return make_layout(*[logical_divide(a[i], b[i]) for i in range(len(b))], *[a[i] for i in range(len(b), len(a))])
    return composition(a, make_layout(b, complement(b, a.size())))


def zipped_divide(a, tiler):
    """逐维划分之后重新分组：((块内的各维), (块号的各维))"""
    d = logical_divide(a, tiler)
    return make_layout(make_layout(*[d[i][0] for i in range(len(tiler))]),
                       make_layout(*[d[i][1] for i in range(len(tiler))]))
