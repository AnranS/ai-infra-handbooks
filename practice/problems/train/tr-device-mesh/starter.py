def mesh_groups(sizes, order):
    out, n = {}, 1
    for s in sizes.values():
        n *= s
    for dim in order:                                    # 只切出了连续的一段一段：对最内层的维度才对
        k = sizes[dim]
        out[dim] = [list(range(i, i + k)) for i in range(0, n, k)]
    return out


def coords(rank, sizes, order):
    pass


def megatron_order(spec, sizes):
    pass


def intra_node(groups, gpus_per_node):
    pass
