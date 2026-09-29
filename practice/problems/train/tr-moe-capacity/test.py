import numpy as np

from checker import check, check_close
from solution import aux_loss, capacity, combine, dispatch

IDX = np.array([[0, 1], [0, 2], [0, 1]])
W = np.array([[0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])


def test_example():
    check(np.asarray(dispatch(IDX, W, E=3, cap=2)).tolist(), [[0, 0], [1, 0], [-1, 1]], "按 token 顺序：第 3 个 token 在专家 0 上被丢")
    check(np.asarray(dispatch(IDX, W, E=3, cap=2, policy="score")).tolist(), [[-1, 0], [1, 0], [0, 1]],
          "按分数：专家 0 丢掉权重最小的那份（0.6）")


def test_capacity():
    check(capacity(4096, 8, 256, 1.0), 128, "每个专家平均 128 份")
    check(capacity(10, 2, 3, 1.0), 7, "20 / 3 向上取整")
    check(capacity(100, 2, 8, 1.25), 32, "容量因子 1.25")


def test_dispatch_random():
    rng = np.random.default_rng(0)
    T, k, E = 64, 2, 8
    for policy in ("order", "score"):
        idx = np.array([rng.choice(E, k, replace=False) for _ in range(T)])
        w = rng.random((T, k))
        cap = capacity(T, k, E, 1.0)
        slot = np.asarray(dispatch(idx, w, E, cap, policy))
        for e in range(E):
            s = slot[idx == e]
            kept = np.sort(s[s >= 0])
            check(kept.tolist(), list(range(min(cap, (idx == e).sum()))), f"{policy}：专家 {e} 的缓冲区位置是 0..n-1")
            if policy == "score" and (s < 0).any():
                check(bool(w[idx == e][s < 0].max() <= w[idx == e][s >= 0].min()), True, f"专家 {e}：丢掉的份权重都不大于留下的")
        dropped = (slot < 0).mean()
        slot2 = np.asarray(dispatch(idx, w, E, capacity(T, k, E, 2.0), policy))
        check(bool((slot2 < 0).mean() <= dropped), True, "容量因子越大，丢得越少")


def test_combine():
    rng = np.random.default_rng(1)
    T, k, E, d = 20, 2, 4, 3
    idx = np.array([rng.choice(E, k, replace=False) for _ in range(T)])
    w = rng.random((T, k))
    cap = 8
    slot = np.asarray(dispatch(idx, w, E, cap))
    x = rng.standard_normal((T, d))
    buf = np.zeros((E, cap, d))
    for t in range(T):
        for j in range(k):
            if slot[t, j] >= 0:
                buf[idx[t, j], slot[t, j]] = x[t] * (idx[t, j] + 1)       # 专家 e 把输入乘以 e + 1
    got = combine(buf, idx, w, slot)
    want = np.array([sum((w[t, j] * x[t] * (idx[t, j] + 1) for j in range(k) if slot[t, j] >= 0), np.zeros(d)) for t in range(T)])
    check_close(got, want, rtol=1e-12, atol=1e-12, what="按 slot 取回、按门控权重加权求和")


def test_aux_loss():
    T, E = 8, 4
    idx = np.array([[t % E] for t in range(T)])
    loss, _ = aux_loss(np.zeros((T, E)), idx)
    check_close(loss, 1.0, what="完全均衡时损失为 1")
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((16, 6)) * 2
    idx = np.argsort(-logits, axis=1)[:, :2]
    loss, grad = aux_loss(logits, idx)
    num = np.zeros_like(logits)
    for i in np.ndindex(logits.shape):
        e = np.zeros_like(logits)
        e[i] = 1e-6
        num[i] = (aux_loss(logits + e, idx)[0] - aux_loss(logits - e, idx)[0]) / 2e-6
    check_close(grad, num, rtol=1e-5, atol=1e-8, what="梯度与数值差分一致（f 当作常数）")
    skew = np.array([[0, 1]] * 16)
    loss_skew, _ = aux_loss(np.tile([5.0, 5.0, 0, 0, 0, 0], (16, 1)), skew)
    check(bool(loss_skew > 2.5), True, f"全部挤在两个专家上时损失很大（{loss_skew:.2f}）")
