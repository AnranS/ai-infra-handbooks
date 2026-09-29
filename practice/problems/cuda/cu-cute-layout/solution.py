from math import prod


def _flatten(t):
    return sum((_flatten(x) for x in t), ()) if isinstance(t, tuple) else (t,)


def _size(shape):
    return prod(_flatten(shape))


def crd2idx(crd, shape, stride):
    if isinstance(crd, tuple):
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    if isinstance(shape, tuple):
        idx = 0
        for s, d in zip(shape[:-1], stride[:-1]):
            idx += crd2idx(crd % _size(s), s, d)
            crd //= _size(s)
        return idx + crd2idx(crd, shape[-1], stride[-1])
    return crd * stride


def coalesce(shape, stride):
    out_n, out_d = [1], [0]
    for n, d in zip(_flatten(shape), _flatten(stride)):
        if n == 1:
            continue
        if out_n[-1] == 1:
            out_n[-1], out_d[-1] = n, d
        elif out_n[-1] * out_d[-1] == d:
            out_n[-1] *= n
        else:
            out_n.append(n)
            out_d.append(d)
    if len(out_n) == 1:
        return out_n[0], out_d[0]
    return tuple(out_n), tuple(out_d)


def complement(shape, stride, max_idx):
    out_n, out_d, cur = [], [], 1
    for d, n in sorted(zip(_flatten(stride), _flatten(shape))):
        if d == 0 or n == 1:
            continue
        out_n.append(d // cur)
        out_d.append(cur)
        cur = n * d
    out_n.append(-(-max_idx // cur))
    out_d.append(cur)
    return coalesce(tuple(out_n), tuple(out_d))


def composition(a, b):
    n, d = b
    if d == 0:
        return n, 0
    shape, stride = coalesce(*a)
    fn, fd = _flatten(shape), _flatten(stride)
    out_n, out_d = [], []
    for s, t in zip(fn[:-1], fd[:-1]):
        take = min(max(1, s // d), n)
        if take != 1:
            out_n.append(take)
            out_d.append(d * t)
        n //= take
        d = -(-d // s)
    if n != 1 or not out_n:
        out_n.append(n)
        out_d.append(d * fd[-1])
    if len(out_n) == 1:
        return out_n[0], out_d[0]
    return tuple(out_n), tuple(out_d)
