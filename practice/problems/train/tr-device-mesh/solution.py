import itertools


def _strides(sizes, order):
    strides, acc = {}, 1
    for name in reversed(order):
        strides[name] = acc
        acc *= sizes[name]
    return strides


def mesh_groups(sizes, order):
    strides = _strides(sizes, order)
    out = {}
    for dim in order:
        others = [d for d in order if d != dim]
        groups = []
        for c in itertools.product(*[range(sizes[d]) for d in others]):
            base = sum(ci * strides[d] for ci, d in zip(c, others))
            groups.append([base + i * strides[dim] for i in range(sizes[dim])])
        out[dim] = sorted(groups, key=lambda g: g[0])
    return out


def coords(rank, sizes, order):
    out = {}
    for name in reversed(order):
        out[name] = rank % sizes[name]
        rank //= sizes[name]
    return {name: out[name] for name in order}


def megatron_order(spec, sizes):
    return [d for d in reversed(spec.split("-")) if d in sizes]


def intra_node(groups, gpus_per_node):
    return all(len({r // gpus_per_node for r in g}) == 1 for g in groups)
