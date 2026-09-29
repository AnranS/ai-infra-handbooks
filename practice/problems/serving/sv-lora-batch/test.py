import numpy as np

from checker import check, check_close
from solution import lora_delta, schedule


def test_example():
    reqs = [(0, 1, 0, 2), (1, 2, 0, 1), (2, 3, 0, 1), (3, 1, 0, 1)]
    check(schedule(reqs, max_loras=2, gpu_slots=2, max_batch=4), ([[0, 1, 3], [0, 2]], 3), "题目里的例子")


def _reference(x, ids, A, B, scale):
    out = np.zeros((x.shape[0], B[0].shape[1]))
    for t, a in enumerate(ids):
        if a >= 0:
            out[t] = scale[a] * (x[t] @ A[a]) @ B[a]
    return out


def test_lora_delta():
    rng = np.random.default_rng(0)
    d_in, d_out, ranks = 24, 20, [4, 16, 8, 1]
    A = [rng.standard_normal((d_in, r)) for r in ranks]
    B = [rng.standard_normal((r, d_out)) for r in ranks]
    scale = [2.0, 0.5, 1.0, 4.0]
    for trial in range(20):
        n = int(rng.integers(1, 40))
        ids = rng.integers(-1, len(ranks), n).tolist()
        x = rng.standard_normal((n, d_in))
        got = lora_delta(x, ids, A, B, scale)
        check(got.shape, (n, d_out), "输出形状")
        check_close(got, _reference(x, ids, A, B, scale), rtol=1e-10, atol=1e-10, what=f"第 {trial} 组：与逐 token 计算一致")
    x = rng.standard_normal((3, d_in))
    check(bool(np.all(lora_delta(x, [-1, -1, -1], A, B, scale) == 0)), True, "全部是 -1 时增量为 0")


def test_base_requests_and_batch_limit():
    reqs = [(0, -1, 0, 1), (1, 5, 0, 1), (2, -1, 0, 1), (3, 6, 0, 1), (4, 5, 0, 1)]
    check(schedule(reqs, max_loras=1, gpu_slots=1, max_batch=3), ([[0, 1, 2], [3], [4]], 3),
          "-1 不占适配器名额；batch 满了就停止扫描；第 1 步只能有一个适配器；适配器 5 第二次要用时已被换出，重新加载")
    check(schedule(reqs, max_loras=2, gpu_slots=4, max_batch=8), ([[0, 1, 2, 3, 4]], 2), "名额够时一步全部跑完")


def test_lru_eviction():
    reqs = [(0, 1, 0, 3), (1, 2, 0, 1), (2, 3, 1, 1), (3, 2, 2, 1), (4, 4, 3, 1)]
    # 第 0 步：{1, 2}；第 1 步：请求 2（适配器 3）加入，槽位满，换出没在用的 2；第 2 步：适配器 2 又要用，换出 3（1 在用）；
    # 第 3 步：请求 0 已结束，1 和 2 都在第 2 步用过，适配器 4 换出编号小的 1
    check(schedule(reqs, max_loras=2, gpu_slots=2, max_batch=4), ([[0, 1], [0, 2], [0, 3], [4]], 5), "LRU 换出")
    check(schedule(reqs, max_loras=2, gpu_slots=4, max_batch=4), ([[0, 1], [0, 2], [0, 3], [4]], 4), "槽位充足时每个适配器只加载一次")


def test_idle_and_arrivals():
    reqs = [(7, 1, 2, 2), (3, 1, 2, 1)]
    check(schedule(reqs, max_loras=1, gpu_slots=1, max_batch=2), ([[], [], [3, 7], [7]], 1), "前两步没有请求到达")
    check(schedule([], 1, 1, 1), ([], 0), "没有请求")


def test_invariants():
    rng = np.random.default_rng(2)
    for trial in range(30):
        n = int(rng.integers(1, 25))
        reqs = [(i, int(rng.integers(-1, 6)), int(rng.integers(0, 6)), int(rng.integers(1, 5))) for i in range(n)]
        max_loras = int(rng.integers(1, 4))
        slots = max_loras + int(rng.integers(0, 3))
        max_batch = int(rng.integers(1, 6))
        batches, loads = schedule(reqs, max_loras, slots, max_batch)
        adapter = {r[0]: r[1] for r in reqs}
        runs = {r[0]: 0 for r in reqs}
        for t, b in enumerate(batches):
            check(len(b) <= max_batch, True, f"第 {trial} 组第 {t} 步：batch 不超过 max_batch")
            check(len({adapter[r] for r in b} - {-1}) <= max_loras, True, f"第 {trial} 组第 {t} 步：适配器数不超过 max_loras")
            for r in b:
                runs[r] += 1
        check(runs, {r[0]: r[3] for r in reqs}, f"第 {trial} 组：每个请求恰好跑了 steps 步")
        used = {a for b in batches for r in b for a in [adapter[r]]} - {-1}
        check(loads >= len(used), True, f"第 {trial} 组：每个用到的适配器至少加载一次")
