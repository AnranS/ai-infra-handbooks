def first_divergence(ours, ref, tol=1e-3):
    if len(ours) != len(ref):
        raise ValueError(f"层数不同：{len(ours)} 与 {len(ref)}")
    for layer, (a, b) in enumerate(zip(ours, ref)):
        if len(a) != len(b):
            raise ValueError(f"第 {layer} 层的位置数不同")
        for pos, (x, y) in enumerate(zip(a, b)):
            if len(x) != len(y):
                raise ValueError(f"第 {layer} 层第 {pos} 个位置的维度不同")
            if any(abs(u - v) > tol for u, v in zip(x, y)):
                return layer, pos
    return None


def guess_cause(position, boundaries):
    if position == 0:
        return "all_positions"
    for name, length in boundaries.items():
        if position == length:
            return name
    return "position_dependent"
