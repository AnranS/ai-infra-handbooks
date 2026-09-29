import numpy as np

from checker import check, check_close
from solution import forward_backward


def loss_fn(H, W, b, y):
    Z = H @ W + b
    Z = Z - Z.max(1, keepdims=True)
    return float(-(Z[np.arange(len(y)), y] - np.log(np.exp(Z).sum(1))).mean())


def numgrad(f, X, eps=1e-6):
    g = np.zeros_like(X)
    it = np.nditer(X, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = X[i]
        X[i] = old + eps
        a = f()
        X[i] = old - eps
        c = f()
        X[i] = old
        g[i] = (a - c) / (2 * eps)
    return g


def setup(N=5, d=4, V=7, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, d)), rng.standard_normal((d, V)), rng.standard_normal(V), rng.integers(0, V, N)


def test_example_loss():
    H, W, b, y = setup()
    loss, dH, dW, db = forward_backward(H, W, b, y)
    check_close(loss, loss_fn(H, W, b, y), rtol=1e-10, what="loss")
    check((dH.shape, dW.shape, db.shape), (H.shape, W.shape, b.shape), "梯度的形状")


def test_gradients_match_numeric():
    H, W, b, y = setup(seed=1)
    _, dH, dW, db = forward_backward(H.copy(), W.copy(), b.copy(), y)
    f = lambda: loss_fn(H, W, b, y)  # noqa: E731
    check_close(dW, numgrad(f, W), rtol=1e-5, atol=1e-8, what="dW")
    check_close(db, numgrad(f, b), rtol=1e-5, atol=1e-8, what="db")
    check_close(dH, numgrad(f, H), rtol=1e-5, atol=1e-8, what="dH")


def test_does_not_modify_inputs():
    H, W, b, y = setup(seed=2)
    H0, W0, b0 = H.copy(), W.copy(), b.copy()
    forward_backward(H, W, b, y)
    check_close(H, H0, what="H 不应被修改")
    check_close(W, W0, what="W 不应被修改")
    check_close(b, b0, what="b 不应被修改")


def test_large_logits_stable():
    H, W, b, y = setup(seed=3)
    loss, dH, dW, db = forward_backward(H * 100, W * 100, b, y)
    assert np.isfinite(loss) and np.isfinite(dW).all() and np.isfinite(dH).all(), "大 logits 时也不能出现 inf/nan"


def test_bigger_batch_no_loops():
    H, W, b, y = setup(N=2000, d=64, V=300, seed=4)
    loss, dH, dW, db = forward_backward(H, W, b, y)
    check_close(loss, loss_fn(H, W, b, y), rtol=1e-9, what="大批量的 loss")
    check_close(db.sum(), 0.0, atol=1e-9, what="db 的元素和（每行 softmax - onehot 的和为 0）")
