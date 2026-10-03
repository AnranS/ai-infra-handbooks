import math
import random

from checker import check_close
from solution import Value


def numgrad(f, xs, i, eps=1e-6):
    a = list(xs)
    b = list(xs)
    a[i] += eps
    b[i] -= eps
    return (f(*a) - f(*b)) / (2 * eps)


def test_example():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a.exp()
    c.backward()
    check_close(c.data, -6 + math.exp(2), what="c.data")
    check_close(a.grad, -3 + math.exp(2), what="a.grad")
    check_close(b.grad, 2.0, what="b.grad")


def test_reuse_accumulates():
    x = Value(3.0)
    y = x * x + x          # dy/dx = 2x + 1 = 7
    y.backward()
    check_close(x.grad, 7.0, what="x 被使用多次时梯度累加")


def test_mixed_with_numbers():
    x = Value(4.0)
    y = 2 * x + 1 - x / 2 + 3 / x - (1 - x) + x ** 2
    y.backward()
    f = lambda v: 2 * v + 1 - v / 2 + 3 / v - (1 - v) + v ** 2  # noqa: E731
    check_close(y.data, f(4.0), what="y.data")
    check_close(x.grad, numgrad(f, [4.0], 0), rtol=1e-5, what="x.grad（与数值梯度比较）")


def test_functions():
    fs = {
        "exp": (lambda v: v.exp(), math.exp),
        "log": (lambda v: v.log(), math.log),
        "tanh": (lambda v: v.tanh(), math.tanh),
        "relu": (lambda v: v.relu(), lambda z: max(z, 0.0)),
    }
    for name, (fv, fn) in fs.items():
        for z in [0.3, 1.7, -0.8]:
            if name == "log" and z <= 0:
                continue
            x = Value(z)
            y = fv(x)
            y.backward()
            check_close(y.data, fn(z), what=f"{name}({z})")
            check_close(x.grad, numgrad(fn, [z], 0), rtol=1e-5, atol=1e-7, what=f"{name}'({z})")


def test_tiny_mlp():
    """两层 MLP 的梯度与数值梯度一致"""
    rng = random.Random(0)
    w = [rng.uniform(-1, 1) for _ in range(9)]

    def forward(*ws):
        x = [0.5, -1.2]
        h1 = (ws[0] * x[0] + ws[1] * x[1] + ws[2])
        h2 = (ws[3] * x[0] + ws[4] * x[1] + ws[5])
        if isinstance(h1, Value):
            h1, h2 = h1.tanh(), h2.relu()
        else:
            h1, h2 = math.tanh(h1), max(h2, 0.0)
        out = ws[6] * h1 + ws[7] * h2 + ws[8]
        return (out - 1.0) ** 2

    vs = [Value(x) for x in w]
    loss = forward(*vs)
    loss.backward()
    for i, v in enumerate(vs):
        check_close(v.grad, numgrad(forward, w, i), rtol=1e-4, atol=1e-7, what=f"第 {i} 个参数的梯度")


def test_deep_graph():
    """5000 层深的计算图：不能用递归"""
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.0001 + 0.0
    y.backward()
    check_close(x.grad, 1.0001 ** 5000, rtol=1e-9, what="深图的梯度")


def test_backward_twice_accumulates_like_torch():
    x = Value(2.0)
    y = x * 3
    y.backward()
    y.backward()
    check_close(x.grad, 6.0, what="调用两次 backward，梯度累加（和 PyTorch 一样）")
