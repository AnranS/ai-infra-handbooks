import random

from checker import check
from solution import BlockPool, Request, Scheduler, Status


def run(workload, num_blocks, block_size, budget, max_seqs, chunked, max_steps=2000):
    pool = BlockPool(num_blocks)
    sch = Scheduler(pool, block_size, budget, max_seqs, chunked)
    reqs = [Request(str(i), [1] * plen, mt) for i, (plen, mt) in enumerate(workload)]
    for r in reqs:
        sch.add_request(r)
    trace = []
    for _ in range(max_steps):
        if not sch.has_unfinished():
            break
        out = sch.schedule()
        trace.append(([(r.request_id, n) for r, n in out.scheduled], out.num_preempted))
        for r, n in out.scheduled:
            r.num_computed += n
        for r, n in out.scheduled:
            if r.num_computed == r.num_tokens and r.status is Status.RUNNING:
                r.output_ids.append(7)
                if len(r.output_ids) >= r.max_tokens:
                    sch.finish(r)
        if not out.scheduled:
            trace.append("stuck")
            break
    return trace, reqs, pool


def ref_schedule(sch):
    budget, scheduled, pre, i = sch.max_num_batched_tokens, [], 0, 0
    while i < len(sch.running) and budget > 0:
        req = sch.running[i]
        n = min(req.num_tokens - req.num_computed, budget)
        ok = True
        while not sch._allocate(req, req.num_computed + n):
            victim = sch.running.pop()
            sch._preempt(victim)
            pre += 1
            if victim is req:
                ok = False
                break
        if not ok:
            break
        scheduled.append((req, n))
        budget -= n
        i += 1
    while sch.waiting and budget > 0 and not pre and len(sch.running) < sch.max_num_seqs:
        req = sch.waiting[0]
        n = req.num_tokens - req.num_computed
        if not sch.enable_chunked_prefill and n > budget:
            break
        n = min(n, budget)
        if not sch._allocate(req, req.num_computed + n):
            break
        sch.waiting.popleft()
        req.status = Status.RUNNING
        sch.running.append(req)
        scheduled.append((req, n))
        budget -= n
    from solution import SchedulerOutput
    return SchedulerOutput(scheduled, pre)


def ref_run(*args, **kw):
    orig = Scheduler.schedule
    Scheduler.schedule = ref_schedule
    try:
        return run(*args, **kw)
    finally:
        Scheduler.schedule = orig


def test_example_chunked_prefill():
    """预算 8：长提示词被切成几段，decode 请求每步 1 个 token"""
    trace, reqs, _ = run([(3, 2), (13, 1)], num_blocks=32, block_size=4, budget=8, max_seqs=4, chunked=True)
    check(trace[0], ([("0", 3), ("1", 5)], 0), "第 1 步：请求 0 的 3 个 token + 请求 1 的前 5 个")
    check(trace[1], ([("0", 1), ("1", 7)], 0), "第 2 步：请求 0 decode 1 个，请求 1 继续 7 个")
    check(trace[2], ([("1", 1)], 0), "第 3 步：请求 1 算完最后 1 个")
    check([len(r.output_ids) for r in reqs], [2, 1], "生成的 token 数")


def test_no_chunking():
    trace, _, _ = run([(3, 2), (13, 1)], num_blocks=32, block_size=4, budget=8, max_seqs=4, chunked=False)
    check(trace[0], ([("0", 3)], 0), "不分块时，13 个 token 的请求放不进剩余预算 5")


def test_preemption_basic():
    """块不够时从末尾抢占，被抢占的请求回到等待队列最前面、从头重算"""
    trace, reqs, pool = run([(4, 6), (4, 6)], num_blocks=3, block_size=4, budget=16, max_seqs=4, chunked=True)
    pre_steps = [t for t in trace if t[1] > 0]
    assert pre_steps, "块只有 3 个，两个请求都要长到 10 个 token，应该发生抢占"
    assert all(len(r.output_ids) == 6 for r in reqs), "所有请求最终都要完成"
    check(pool.num_free(), 3, "结束后块全部归还")
    check(trace, ref_run([(4, 6), (4, 6)], num_blocks=3, block_size=4, budget=16, max_seqs=4, chunked=True)[0],
          "完整的调度轨迹")


def test_max_num_seqs():
    trace, _, _ = run([(2, 3)] * 5, num_blocks=64, block_size=4, budget=64, max_seqs=2, chunked=True)
    check(len(trace[0][0]), 2, "第一步最多接收 2 个请求")


def test_random_workloads_match_reference():
    rng = random.Random(0)
    for trial in range(40):
        wl = [(rng.randint(1, 30), rng.randint(1, 12)) for _ in range(rng.randint(1, 8))]
        bs = rng.choice([2, 4, 8])
        need = max((p + m + bs - 1) // bs for p, m in wl)
        cfg = dict(num_blocks=rng.randint(need, need * 4), block_size=bs, budget=rng.choice([4, 8, 16, 64]),
                   max_seqs=rng.randint(1, 6), chunked=rng.random() < 0.7)
        want, _, _ = ref_run(wl, **cfg)
        got, _, _ = run(wl, **cfg)
        check(got, want, f"随机负载 {trial}（{cfg}）的调度轨迹")
