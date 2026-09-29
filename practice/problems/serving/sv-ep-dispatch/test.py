import numpy as np

from checker import check, check_close
from solution import ep_moe


def make_experts(E, d, seed=0, log=None):
    rng = np.random.default_rng(seed)
    Ws = [rng.standard_normal((d, d)) * 0.3 for _ in range(E)]

    def mk(e):
        def f(v):
            if log is not None:
                log.append((e, v.shape[0]))
            return np.tanh(v @ Ws[e])
        return f

    return [mk(e) for e in range(E)], Ws


def setup(n=2, E=4, k=2, d=6, tokens=(3, 4), seed=0):
    rng = np.random.default_rng(seed)
    x = [rng.standard_normal((t, d)) for t in tokens]
    ids = [np.array([rng.choice(E, k, replace=False) for _ in range(t)], dtype=np.int64).reshape(t, k) for t in tokens]
    w = [rng.dirichlet(np.ones(k), t).reshape(t, k) for t in tokens]
    return x, ids, w


def reference(x, ids, w, Ws):
    outs = []
    for r in range(len(x)):
        o = np.zeros_like(x[r])
        for i in range(len(x[r])):
            for j in range(ids[r].shape[1]):
                o[i] += w[r][i, j] * np.tanh(x[r][i] @ Ws[ids[r][i, j]])
        outs.append(o)
    return outs


def test_example():
    x, ids, w = setup()
    experts, Ws = make_experts(4, 6)
    out, counts = ep_moe(x, ids, w, experts, 2)
    ref = reference(x, ids, w, Ws)
    for r in range(2):
        check_close(out[r], ref[r], rtol=1e-10, atol=1e-12, what=f"rank {r} 的输出")
    want = [[sum(int(e) // 2 == s for e in ids[r].ravel()) for s in range(2)] for r in range(2)]
    check(counts, want, "send_counts")


def test_grouped_calls():
    x, ids, w = setup(n=4, E=8, k=2, tokens=(5, 6, 0, 7), seed=1)
    log = []
    experts, Ws = make_experts(8, 6, seed=1, log=log)
    out, counts = ep_moe(x, ids, w, experts, 4)
    ref = reference(x, ids, w, Ws)
    for r in range(4):
        check_close(out[r], ref[r], rtol=1e-10, atol=1e-12, what=f"rank {r}（rank 2 没有 token）")
    called = [e for e, _ in log]
    check(len(called), len(set(called)), "每个专家最多调用一次")
    total = {e: 0 for e in range(8)}
    for r in range(4):
        for e in ids[r].ravel():
            total[int(e)] += 1
    check(dict(log), {e: c for e, c in total.items() if c}, "每个专家一次处理的 token 数")
    check(sum(map(sum, counts)), sum(t * 2 for t in (5, 6, 0, 7)), "发送的 (token, 专家) 对总数")


def test_top1_and_many_ranks():
    x, ids, w = setup(n=4, E=4, k=1, tokens=(4, 4, 4, 4), seed=2)
    experts, Ws = make_experts(4, 6, seed=2)
    out, _ = ep_moe(x, ids, w, experts, 4)
    ref = reference(x, ids, w, Ws)
    for r in range(4):
        check_close(out[r], ref[r], rtol=1e-10, atol=1e-12, what=f"top-1，rank {r}")
