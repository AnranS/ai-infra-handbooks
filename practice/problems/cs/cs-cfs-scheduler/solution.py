# 内核的 sched_prio_to_weight：nice 从 -20 到 19 对应的权重，nice 0 是 1024
WEIGHT = dict(zip(range(-20, 20), [
    88761, 71755, 56483, 46273, 36291, 29154, 23254, 18705, 14949, 11916,
    9548, 7620, 6100, 4904, 3906, 3121, 2501, 1991, 1586, 1277,
    1024, 820, 655, 526, 423, 335, 272, 215, 172, 137,
    110, 87, 70, 56, 45, 36, 29, 23, 18, 15]))


def simulate(tasks, total_ms):
    nice_of = {name: nice for name, nice, _ in tasks}
    vruntime, ran = {}, {name: 0 for name, _, _ in tasks}
    for t in range(total_ms):
        for name, _, arrive in tasks:
            if arrive == t:                     # 新到的任务放到当前最小值上，不会独占 CPU
                vruntime[name] = min(vruntime.values()) if vruntime else 0.0
        if not vruntime:
            continue                            # 这一毫秒 CPU 空闲
        name = min(vruntime, key=lambda n: (vruntime[n], n))
        ran[name] += 1
        vruntime[name] += 1024 / WEIGHT[nice_of[name]]
    return ran
