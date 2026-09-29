import numpy as np

from checker import check, time_limit
from solution import sample_batch


def ref_one(logits, t, k, p, u):
    if t <= 0 or (k == 1 and p == 1.0):
        return int(np.argmax(logits))
    z = logits / t
    probs = np.exp(z - z.max())
    probs /= probs.sum()
    order = np.lexsort((np.arange(len(probs)), -probs))
    if k > 0:
        probs[order[k:]] = 0
    if p < 1.0:
        c = np.cumsum(probs[order])
        keep = int(np.searchsorted(c, p - 1e-12)) + 1
        probs[order[keep:]] = 0
    probs /= probs.sum()
    idx = int(np.searchsorted(np.cumsum(probs), u, side="right"))
    return idx if idx < len(probs) else int(np.nonzero(probs)[0][-1])


def batch(B, V, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((B, V)) * rng.uniform(0.5, 4, (B, 1))
    t = rng.choice([0.0, 0.3, 0.7, 1.0, 1.5], B)
    k = rng.choice([-1, 1, 5, 20], B)
    p = rng.choice([1.0, 0.95, 0.8, 0.3], B)
    u = rng.random(B)
    return logits, t, k, p, u


def test_example():
    logits = np.log(np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [5.0, 1.0, 5.0, 0.0]]))
    got = sample_batch(logits, np.array([1.0, 1.0, 0.0]), np.array([-1, 2, -1]), np.array([1.0, 1.0, 1.0]),
                       np.array([0.05, 0.05, 0.9]))
    check(got.tolist(), [0, 2, 0], "三个请求：普通采样、top-k=2、贪心（并列取下标小的）")


def test_matches_reference():
    for seed in range(5):
        logits, t, k, p, u = batch(64, 50, seed)
        want = [ref_one(logits[i], t[i], k[i], p[i], u[i]) for i in range(64)]
        check(sample_batch(logits, t, k, p, u).tolist(), want, f"随机 batch {seed}")


def test_ties_in_top_k():
    logits = np.zeros((2, 6))
    got = sample_batch(logits, np.array([1.0, 1.0]), np.array([3, 3]), np.array([1.0, 1.0]), np.array([0.0, 0.99]))
    check(got.tolist(), [0, 2], "全部并列时 top-3 保留下标 0、1、2")


def test_speed():
    logits, t, k, p, u = batch(256, 5000, 9)
    with time_limit(1.0, "B=256, V=5000 的批量采样"):
        got = sample_batch(logits, t, k, p, u)
    for i in [0, 17, 255]:
        check(int(got[i]), ref_one(logits[i], t[i], k[i], p[i], u[i]), f"第 {i} 行")
