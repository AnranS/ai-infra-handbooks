def partition(sizes, world):
    total = sum(sizes)
    per = -(-total // world)                        # 向上取整：补零到 world 的整数倍
    segments = [[] for _ in range(world)]
    offset = 0                                       # 当前参数在一维缓冲区里的起点
    for i, n in enumerate(sizes):
        pos = offset
        while pos < offset + n:
            r = pos // per
            end = min(offset + n, (r + 1) * per)
            segments[r].append((i, pos - offset, end - offset))
            pos = end
        offset += n
    return per, segments


def model_state_bytes(psi, world, stage):
    table = {0: 16 * psi, 1: 4 * psi + 12 * psi / world, 2: 2 * psi + 14 * psi / world, 3: 16 * psi / world}
    if stage not in table:
        raise ValueError(f"未知的 ZeRO 级别：{stage}")
    return table[stage]


def comm_bytes(psi, world, stage):
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"未知的 ZeRO 级别：{stage}")
    ring = (world - 1) / world * 2 * psi             # 一次 reduce-scatter 或 all-gather（bf16）
    return (3 if stage == 3 else 2) * ring
