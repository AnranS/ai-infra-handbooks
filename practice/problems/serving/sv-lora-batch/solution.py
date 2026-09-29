import numpy as np


def lora_delta(x, adapter_ids, A, B, scale):
    ids = np.asarray(adapter_ids)
    out = np.zeros((x.shape[0], B[0].shape[1]))
    order = np.argsort(ids, kind="stable")                 # 用同一个适配器的 token 排到一起
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    for s, e in zip(starts, ends):
        a = int(sorted_ids[s])
        if a < 0:
            continue
        rows = order[s:e]
        out[rows] = scale[a] * (x[rows] @ A[a]) @ B[a]     # 一段做一次矩阵乘：先降到秩 r_a，再升回 d_out
    return out


def schedule(requests, max_loras, gpu_slots, max_batch):
    pending = sorted(requests, key=lambda r: (r[2], r[0]))
    left = {r[0]: r[3] for r in requests}
    adapter_of = {r[0]: r[1] for r in requests}
    running, resident, last_used = [], set(), {}
    batches, loads, t = [], 0, 0
    while pending or running:
        active = {adapter_of[r] for r in running} - {-1}
        still = []
        for req in pending:                                # 准入：按到达顺序扫描，放不下的跳过
            rid, a, arrival, _ = req
            if arrival > t or len(running) >= max_batch:
                still.append(req)
                continue
            if a == -1 or a in active or len(active) < max_loras:
                running.append(rid)
                if a != -1:
                    active.add(a)
            else:
                still.append(req)
        pending = still
        for a in sorted(active - resident):                # 显存槽位：缺的加载，满了换出最久没用、且没在用的
            if len(resident) >= gpu_slots:
                victim = min(resident - active, key=lambda x: (last_used.get(x, -1), x))
                resident.remove(victim)
            resident.add(a)
            loads += 1
        batches.append(sorted(running))
        for a in active:
            last_used[a] = t
        for rid in running:
            left[rid] -= 1
        running = [r for r in running if left[r] > 0]
        t += 1
    return batches, loads
