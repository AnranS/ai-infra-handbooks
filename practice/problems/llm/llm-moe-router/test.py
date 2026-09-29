import numpy as np

import solution
from checker import check, check_close
from solution import moe_forward


def make(N=12, d=8, E=6, dff=10, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, d))
    router = rng.standard_normal((d, E))
    experts = [(rng.standard_normal((d, dff)) * 0.3, rng.standard_normal((d, dff)) * 0.3,
                rng.standard_normal((dff, d)) * 0.3) for _ in range(E)]
    return x, router, experts


def silu_ffn(v, g, u, dn):
    a = v @ g
    return (a / (1 + np.exp(-a)) * (v @ u)) @ dn


def ref(x, router, experts, k, norm):
    N, E = x.shape[0], router.shape[1]
    out = np.zeros_like(x)
    ids, ws = [], []
    counts = np.zeros(E)
    probs_all = []
    for i in range(N):
        lg = x[i] @ router
        p = np.exp(lg - lg.max())
        p /= p.sum()
        probs_all.append(p)
        order = sorted(range(E), key=lambda e: (-p[e], e))[:k]
        w = np.array([p[e] for e in order])
        if norm:
            w = w / w.sum()
        ids.append(order)
        ws.append(w)
        for e, we in zip(order, w):
            out[i] += we * silu_ffn(x[i], *experts[e])
            counts[e] += 1
    f = counts / (N * k)
    P = np.mean(probs_all, axis=0)
    return out, np.array(ids), np.array(ws), E * float(np.sum(f * P))


def test_example():
    x, r, ex = make()
    res = moe_forward(x, r, ex, top_k=2)
    out, ids, ws, aux = ref(x, r, ex, 2, True)
    check(res["topk_ids"], ids, "topk_ids")
    check_close(res["topk_weights"], ws, rtol=1e-10, what="topk_weights")
    check_close(res["out"], out, rtol=1e-9, atol=1e-10, what="输出")
    check_close(res["aux_loss"], aux, rtol=1e-10, what="aux_loss")


def test_no_renormalize():
    x, r, ex = make(seed=1)
    res = moe_forward(x, r, ex, top_k=3, norm_topk_prob=False)
    out, ids, ws, _ = ref(x, r, ex, 3, False)
    check_close(res["topk_weights"], ws, rtol=1e-10, what="不重新归一化的权重")
    check_close(res["out"], out, rtol=1e-9, atol=1e-10, what="输出")


def test_tokens_per_expert():
    x, r, ex = make(N=20, seed=2)
    res = moe_forward(x, r, ex, top_k=2)
    tpe = res["tokens_per_expert"]
    check(len(tpe), 6, "专家数")
    check(sum(len(t) for t in tpe), 40, "路由次数之和 = N·k")
    for e, toks in enumerate(tpe):
        want = sorted(i for i in range(20) if e in res["topk_ids"][i])
        check(list(toks), want, f"专家 {e} 的 token")


def test_ties_prefer_lower_index():
    x = np.ones((2, 2))
    router = np.zeros((2, 4))                     # 所有专家打分相同
    ex = [(np.eye(2, 3), np.eye(2, 3), np.eye(3, 2)) for _ in range(4)]
    res = moe_forward(x, router, ex, top_k=2)
    check(res["topk_ids"].tolist(), [[0, 1], [0, 1]], "并列时选下标小的")
    check_close(res["aux_loss"], 4 * (0.5 * 0.25 + 0.5 * 0.25), what="aux_loss")


def test_grouped_by_expert():
    """每个专家只调用一次，而且只处理分给它的 token"""
    x, r, ex = make(N=30, seed=3)
    calls = []
    original = solution.swiglu

    def counting(v, *w):
        calls.append(v.shape[0])
        return original(v, *w)

    solution.swiglu = counting
    try:
        res = moe_forward(x, r, ex, top_k=2)
    finally:
        solution.swiglu = original
    used = [len(t) for t in res["tokens_per_expert"] if t]
    check(sorted(calls), sorted(used), "每次调用专家时处理的 token 数")
