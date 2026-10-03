import numpy as np

from checker import check
from solution import sample


def ref(logits, u, temperature=1.0, top_k=-1, top_p=1.0):
    logits = [float(x) for x in logits]
    n = len(logits)
    if temperature == 0:
        return max(range(n), key=lambda i: (logits[i], -i))
    z = [x / temperature for x in logits]
    m = max(z)
    e = [np.exp(x - m) for x in z]
    s = sum(e)
    p = [x / s for x in e]
    order = sorted(range(n), key=lambda i: (-p[i], i))
    keep = set(order if top_k <= 0 else order[:top_k])
    if top_p < 1.0:
        acc, kept = 0.0, set()
        for i in order:
            if i not in keep:
                continue
            kept.add(i)
            acc += p[i]
            if acc >= top_p - 1e-12:
                break
        keep = kept
    total = sum(p[i] for i in keep)
    acc = 0.0
    for i in range(n):
        if i in keep:
            acc += p[i] / total
            if acc > u:
                return i
    return max(keep)


L = np.log(np.array([0.1, 0.2, 0.3, 0.4]))


def test_example():
    check(sample(L, u=0.05), 0, "sample(u=0.05)")
    check(sample(L, u=0.05, top_k=2), 2, "top_k=2")
    check(sample(L, u=0.99, top_p=0.5), 3, "top_p=0.5, u=0.99")
    check(sample(L, u=0.5, temperature=0), 3, "temperature=0")


def test_inverse_cdf_boundaries():
    check([sample(L, u) for u in [0.0, 0.0999, 0.1001, 0.29, 0.3001, 0.6001, 0.999]], [0, 0, 1, 1, 2, 3, 3], "累计和两侧")


def test_top_p_details():
    check(sample(L, 0.0, top_p=0.4), 3, "top_p=0.4 正好等于最大概率：只保留 1 个")
    check(sample(L, 0.0, top_p=0.41), 2, "top_p=0.41：保留 {3, 2}，按下标累加先遇到 2")
    check(sample(L, 0.99, top_p=0.95, top_k=3), 3, "top_k=3 后 top_p=0.95")


def test_temperature():
    logits = np.array([1.0, 2.0, 3.0])
    check(sample(logits, 0.5, temperature=1e-4), 2, "温度趋近 0 时接近贪心")
    check(sample(logits, 0.5, temperature=100.0), 1, "温度很高时接近均匀分布（u=0.5 落在中间）")


def test_greedy_ties():
    check(sample(np.array([1.0, 5.0, 5.0]), 0.9, temperature=0), 1, "并列最大值取下标小的")


def test_random_against_reference():
    rng = np.random.default_rng(0)
    for i in range(300):
        n = int(rng.integers(1, 20))
        logits = rng.standard_normal(n) * rng.uniform(0.1, 5)
        if rng.random() < 0.2:
            logits[rng.integers(0, n)] = logits.max()       # 制造并列
        kw = dict(temperature=float(rng.choice([0.0, 0.5, 1.0, 1.7])), top_k=int(rng.choice([-1, 1, 2, 5])),
                  top_p=float(rng.choice([1.0, 0.9, 0.5, 0.1])))
        u = float(rng.random())
        check(sample(logits, u, **kw), ref(logits, u, **kw), f"随机用例 {i}（{kw}, u={u:.3f}）")
