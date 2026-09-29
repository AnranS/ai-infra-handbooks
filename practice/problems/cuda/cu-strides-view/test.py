import itertools
import math
import random

from checker import check, raises
from solution import expand_strides, is_contiguous, view_strides


def addrs(shape, stride):
    return [sum(i * s for i, s in zip(idx, stride)) for idx in itertools.product(*[range(n) for n in shape])]


def check_view(shape, stride, new_shape, got, what):
    """got 为 None 时确认确实做不成视图；否则确认新 stride 描述的内存布局和原来一样"""
    new_shape = list(new_shape)
    if -1 in new_shape:
        new_shape[new_shape.index(-1)] = math.prod(shape) // -math.prod(new_shape)
    old = addrs(shape, stride)
    if got is None:
        cand = [old[math.prod(new_shape[d + 1:])] - old[0] if n > 1 else 0 for d, n in enumerate(new_shape)]
        if addrs(new_shape, cand) == old:
            raise AssertionError(f"{what}：返回了 None，但 stride {tuple(cand)} 就能做成视图")
        return
    if len(got) != len(new_shape):
        raise AssertionError(f"{what}：stride 的维数不对：{got}")
    if addrs(new_shape, got) != old:
        raise AssertionError(f"{what}：stride {got} 描述的内存布局和原张量不同")


def same_ignoring_ones(got, want, new_shape):
    if got is None or want is None:
        return got == want
    return all(n == 1 or g == w for g, w, n in zip(got, want, new_shape))


def test_example():
    check(view_strides((3, 4), (4, 1), (2, 6)), (6, 1), "连续张量拆成 (2, 6)")
    check(view_strides((4, 3), (1, 4), (12,)), None, "转置之后不能展平")
    check(view_strides((3, 4), (0, 1), (3, 2, 2)), (0, 2, 1), "expand 出来的维度也能拆")
    check(expand_strides((3, 1), (1, 1), (2, 3, 4)), (0, 1, 0), "expand")


def test_is_contiguous():
    check(is_contiguous((3, 4), (4, 1)), True, "行主序")
    check(is_contiguous((4, 3), (1, 4)), False, "转置")
    check(is_contiguous((3, 2), (4, 1)), False, "切片 x[:, 1:3]")
    check(is_contiguous((3, 1, 4), (4, 99, 1)), True, "长度为 1 的维度不看 stride")
    check(is_contiguous((1, 5), (1, 1)), True, "第 0 维长度为 1")
    check(is_contiguous((2, 3), (0, 1)), False, "expand 出来的（stride 为 0）不连续")
    check(is_contiguous((7,), (2,)), False, "步长为 2 的切片")


def test_view_cases():
    cases = [
        ((3, 4), (4, 1), (12,), (1,)),
        ((3, 4), (4, 1), (4, 3), (3, 1)),
        ((3, 4), (4, 1), (-1, 2), (2, 1)),
        ((3, 2), (4, 1), (6,), None),                  # 切片 x[:, 1:3] 之后不能展平
        ((3, 2), (4, 1), (3, 2, 1), (4, 1, 1)),
        ((3, 2), (4, 1), (3, 1, 2), (4, 1, 1)),
        ((2, 4, 3), (12, 1, 4), (8, 3), None),         # permute(0, 2, 1) 之后后两维接不上
        ((2, 4, 3), (12, 1, 4), (1, 2, 4, 3), (24, 12, 1, 4)),
        ((2, 3, 4), (12, 4, 1), (6, 4), (4, 1)),       # 前两维合并
        ((2, 3, 4), (1, 8, 2), (2, 12), (1, 2)),       # 后两维接得上（8 = 4 × 2），合并成步长 2 的一块
        ((2, 3, 4), (1, 8, 2), (6, 4), None),          # 第 0 维和后面接不上
        ((4, 6), (12, 1), (4, 2, 3), (12, 3, 1)),      # 每行隔 12 个元素（从更宽的矩阵里切出来的）
        ((4, 6), (12, 1), (8, 3), None),               # 每行 6 个拆成 2×3 可以，但 8 = 4×2 要跨行合并，行间接不上
        ((3, 4), (0, 1), (12,), None),                 # expand 的维度不能和别的维度合并
        ((5,), (3,), (5, 1), (3, 1)),
    ]
    for shape, stride, new, want in cases:
        got = view_strides(shape, stride, new)
        check_view(shape, stride, new, got, f"shape={shape} stride={stride} → {new}")
        if not same_ignoring_ones(got, want, [n if n != -1 else 2 for n in new]):
            raise AssertionError(f"shape={shape} stride={stride} → {new}：期望 {want}，实际 {got}")


def test_view_errors():
    with raises(ValueError, "元素个数对不上"):
        view_strides((3, 4), (4, 1), (5, 2))
    with raises(ValueError, "-1 推不出整数"):
        view_strides((3, 4), (4, 1), (5, -1))
    with raises(ValueError, "两个 -1"):
        view_strides((3, 4), (4, 1), (-1, -1))


def random_tensor(rng):
    nd = rng.randint(1, 4)
    shape = [rng.randint(1, 4) for _ in range(nd)]
    kind = rng.random()
    if kind < 0.4:                                     # 连续张量再 permute
        stride, acc = [0] * nd, 1
        order = list(range(nd))
        rng.shuffle(order)
        for d in reversed(order):
            stride[d] = acc
            acc *= shape[d]
    else:                                              # 任意的 stride（包括 0、切片出来的间隔）
        stride = [rng.choice([0, 1, 2, 3, 4, 6, 8, 12, 16, 24]) for _ in range(nd)]
    return tuple(shape), tuple(stride)


def random_new_shape(rng, numel):
    factors, n = [], numel
    for _ in range(rng.randint(0, 3)):
        f = rng.choice([f for f in range(1, n + 1) if n % f == 0])
        factors.append(f)
        n //= f
    factors.append(n)
    rng.shuffle(factors)
    return factors


def test_view_random():
    rng = random.Random(0)
    n_view = n_none = 0
    for trial in range(600):
        shape, stride = random_tensor(rng)
        new = random_new_shape(rng, math.prod(shape))
        got = view_strides(shape, stride, new)
        check_view(shape, stride, new, got, f"第 {trial} 组 shape={shape} stride={stride} → {tuple(new)}")
        n_view += got is not None
        n_none += got is None
        want = addrs(shape, stride) == list(range(math.prod(shape)))
        check(is_contiguous(shape, stride), want, f"第 {trial} 组：shape={shape} stride={stride} 是否连续")
    check(n_view > 100 and n_none > 100, True, "随机用例里两种情况都有")


def test_expand():
    check(expand_strides((1,), (1,), (5,)), (0,), "长度 1 扩展")
    check(expand_strides((2, 3), (3, 1), (-1, 3)), (3, 1), "-1 表示不变")
    check(expand_strides((4, 1, 3), (3, 3, 1), (2, 4, 6, 3)), (0, 3, 0, 1), "加维度并扩展中间一维")
    with raises(ValueError, "长度不为 1 的维度不能扩展"):
        expand_strides((2, 3), (3, 1), (4, 3))
    with raises(ValueError, "不能减少维数"):
        expand_strides((2, 3), (3, 1), (6,))
