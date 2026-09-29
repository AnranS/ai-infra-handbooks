import random
from math import prod

from checker import check
from solution import coalesce, complement, composition, crd2idx


def flat(t):
    return sum((flat(x) for x in t), ()) if isinstance(t, tuple) else (t,)


def ref_idx(crd, shape, stride):
    """参考实现：只在测试里用来算期望值"""
    if isinstance(crd, tuple):
        return sum(ref_idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    if isinstance(shape, tuple):
        idx = 0
        for s, d in zip(shape[:-1], stride[:-1]):
            idx += ref_idx(crd % prod(flat(s)), s, d)
            crd //= prod(flat(s))
        return idx + ref_idx(crd, shape[-1], stride[-1])
    return crd * stride


def ref_coalesce(shape, stride):
    out_n, out_d = [1], [0]
    for n, d in zip(flat(shape), flat(stride)):
        if n == 1:
            continue
        if out_n[-1] == 1:
            out_n[-1], out_d[-1] = n, d
        elif out_n[-1] * out_d[-1] == d:
            out_n[-1] *= n
        else:
            out_n.append(n)
            out_d.append(d)
    return tuple(out_n), tuple(out_d)


def size(shape):
    return prod(flat(shape))


def rand_layout(rng, depth=0):
    """随机的（可能嵌套的）布局；步长可以是 0"""
    r = rng.randint(1, 3)
    shape, stride = [], []
    for _ in range(r):
        if depth == 0 and rng.random() < 0.3:
            s, d = rand_layout(rng, 1)
            shape.append(s if isinstance(s, tuple) else (s,))
            stride.append(d if isinstance(d, tuple) else (d,))
        else:
            shape.append(rng.choice([1, 2, 3, 4, 6]))
            stride.append(rng.choice([0, 1, 2, 3, 4, 8, 12]))
    return (tuple(shape), tuple(stride)) if r > 1 else (shape[0], stride[0])


def test_example():
    check(crd2idx((1, 5), (4, (2, 4)), (2, (1, 8))), 19, "crd2idx((1, 5)) —— 嵌套的维上用整数坐标")
    check(crd2idx(5, (4, (2, 4)), (2, (1, 8))), 3, "crd2idx(5) —— 一维坐标")
    check(coalesce((2, (1, 6)), (1, (6, 2))), (12, 1), "coalesce((2,(1,6)):(1,(6,2)))")
    check(complement(4, 2, 24), ((2, 3), (1, 8)), "complement(4:2, 24)")
    r = composition(((6, 2), (8, 2)), (4, 3))
    check([ref_idx(i, *r) for i in range(4)], [0, 24, 2, 26], "composition((6,2):(8,2), 4:3) 的各个值")


def test_crd2idx():
    L = ((4, (2, 4)), (2, (1, 8)))
    check([crd2idx(i, *L) for i in range(8)], [0, 2, 4, 6, 1, 3, 5, 7], "一维坐标 0..7：第 0 维变化最快")
    check(crd2idx((3, (1, 3)), *L), 31, "完全展开的坐标 (3, (1, 3))")
    check(crd2idx((2, 7), (4, 8), (8, 1)), 23, "行优先")
    check(crd2idx(7, 8, 3), 21, "一维布局")
    check(crd2idx(9, (2, 3), (1, 2)), 9, "超出 size 的坐标沿最后一维延伸（CuTe 的约定）")
    rng = random.Random(0)
    for _ in range(300):
        shape, stride = rand_layout(rng)
        got = [crd2idx(i, shape, stride) for i in range(size(shape))]
        check(got, [ref_idx(i, shape, stride) for i in range(size(shape))], f"布局 {shape}:{stride} 的所有一维坐标")


def test_coalesce():
    cases = [
        (((4, 8), (1, 4)), (32, 1)),
        (((4, 8), (8, 1)), ((4, 8), (8, 1))),
        (((2, (1, 6)), (1, (6, 2))), (12, 1)),
        (((1, 1), (5, 7)), (1, 0)),
        (((2, 1, 3), (1, 9, 2)), (6, 1)),
        (((3, 2, 2), (0, 0, 0)), (12, 0)),                         # 步长全为 0：3 × 0 = 0 也"接得上"
        ((((2, 2), (2, 2)), ((1, 2), (4, 8))), (16, 1)),
    ]
    for (shape, stride), want in cases:
        check(coalesce(shape, stride), want, f"coalesce({shape}:{stride})")
    rng = random.Random(1)
    for _ in range(300):
        shape, stride = rand_layout(rng)
        cs, cd = coalesce(shape, stride)
        n = size(shape)
        check([ref_idx(i, cs, cd) for i in range(n)], [ref_idx(i, shape, stride) for i in range(n)],
              f"coalesce({shape}:{stride}) = {cs}:{cd} 必须是同一个函数")
        fs, fd = flat(cs), flat(cd)
        assert not isinstance(cs, tuple) or all(isinstance(x, int) for x in cs), f"coalesce 的结果应该是扁平的：{cs}"
        assert all(n > 1 for n in fs) or (fs, fd) == ((1,), (0,)), f"coalesce({shape}:{stride}) 还有长度为 1 的维：{cs}:{cd}"
        for k in range(len(fs) - 1):
            assert fs[k] * fd[k] != fd[k + 1], f"coalesce({shape}:{stride}) = {cs}:{cd}，第 {k} 维和第 {k + 1} 维还能合并"


def rand_injective(rng):
    """每一维的步长都是前面各维覆盖范围的整数倍：单射，补集有定义"""
    modes, cur = [], 1
    for _ in range(rng.randint(1, 3)):
        d = cur * rng.choice([1, 1, 2, 3])
        n = rng.choice([2, 3, 4])
        modes.append((n, d))
        cur = n * d
    rng.shuffle(modes)
    shape, stride = tuple(m[0] for m in modes), tuple(m[1] for m in modes)
    return (shape, stride, cur) if len(modes) > 1 else (shape[0], stride[0], cur)


def test_complement():
    check(complement((2, 2), (1, 6), 24), ((3, 2), (2, 12)), "complement((2,2):(1,6), 24)")
    check(complement(4, 1, 24), (6, 4), "complement(4:1, 24)")
    check(complement(4, 1, 4), (1, 0), "A 已经覆盖了全部：补集是 1:0")
    check(complement((3, 4), (4, 1), 12), (1, 0), "(3,4):(4,1) 覆盖了 0..11")
    rng = random.Random(2)
    for _ in range(300):
        shape, stride, cover = rand_injective(rng)
        m = cover * rng.choice([1, 2, 3])
        cs, cd = complement(shape, stride, m)
        both = ((shape if isinstance(shape, tuple) else (shape,)) + (cs if isinstance(cs, tuple) else (cs,)),
                (stride if isinstance(stride, tuple) else (stride,)) + (cd if isinstance(cd, tuple) else (cd,)))
        n = size(both[0])
        check(n, m, f"(A, complement(A, {m})) 的 size，A = {shape}:{stride}，补集 = {cs}:{cd}")
        check(sorted(ref_idx(i, *both) for i in range(n)), list(range(m)),
              f"(A, complement(A, {m})) 要恰好覆盖 0..{m - 1}，A = {shape}:{stride}，补集 = {cs}:{cd}")
        check((cs, cd), coalesce(cs, cd), f"complement 的结果要先合并：{cs}:{cd}")


def test_composition():
    cases = [
        (((6, 2), (8, 2)), (3, 1), 3),
        (((4, 8), (8, 1)), (8, 4), 8),
        (((8, 8), (8, 1)), (4, 16), 4),
        ((24, 1), (4, 2), 4),
        ((4, 1), (4, 2), 4),                                       # 超出 A 的范围：沿最后一维延伸
        (((2, 3), (1, 2)), (5, 0), 5),                             # 步长为 0：广播
        ((((2, 2), 6), ((1, 12), 2)), (6, 2), 6),
    ]
    for a, b, n in cases:
        r = composition(a, b)
        check(size(r[0]), n, f"composition({a[0]}:{a[1]}, {b[0]}:{b[1]}) 的 size")
        check([ref_idx(i, *r) for i in range(n)], [ref_idx(i * b[1], *a) for i in range(n)],
              f"composition({a[0]}:{a[1]}, {b[0]}:{b[1]}) = {r[0]}:{r[1]}，要满足 R(i) == A(B(i))")
    rng = random.Random(3)
    tried = 0
    while tried < 300:
        shape = rng.choice([(4, 8), (6, 2), (2, (2, 3)), 24, (8, 4), ((2, 2), 6), (3, 4, 2), (2, 2, 2, 2)])
        a = (shape, rng.choice([None, "rev"]))
        # 紧凑的列优先或行优先布局
        fs = flat(shape)
        strides, acc = [], 1
        order = list(range(len(fs))) if a[1] is None else list(reversed(range(len(fs))))
        st = [0] * len(fs)
        for k in order:
            st[k] = acc
            acc *= fs[k]
        def rebuild(like, it):
            return tuple(rebuild(x, it) for x in like) if isinstance(like, tuple) else next(it)
        stride = rebuild(shape, iter(st))
        n, d = rng.choice([2, 3, 4, 6]), rng.choice([1, 2, 3, 4, 6])
        ok, rest_n, rest_d = True, n, d                            # 只测复合有定义的情形（两个整除条件）
        for s in flat(ref_coalesce(shape, stride)[0])[:-1]:
            if not (s % rest_d == 0 or rest_d % s == 0):
                ok = False
                break
            take = min(max(1, s // rest_d), rest_n)
            if rest_n % take:
                ok = False
                break
            rest_n //= take
            rest_d = -(-rest_d // s)
        if not ok:
            continue
        tried += 1
        r = composition((shape, stride), (n, d))
        check(size(r[0]), n, f"composition({shape}:{stride}, {n}:{d}) 的 size")
        check([ref_idx(i, *r) for i in range(n)], [ref_idx(i * d, shape, stride) for i in range(n)],
              f"composition({shape}:{stride}, {n}:{d}) = {r[0]}:{r[1]}，要满足 R(i) == A(B(i))")
