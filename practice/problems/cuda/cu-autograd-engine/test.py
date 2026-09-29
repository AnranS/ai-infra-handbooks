import numpy as np

import solution
from checker import check, check_close
from solution import Tensor, unbroadcast


def test_example():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)
    (x * y + x).sum().backward()
    check_close(x.grad, [4.0, 5.0], what="x.grad = y + 1")
    check_close(y.grad, [1.0, 2.0], what="y.grad = x")


def test_unbroadcast():
    g = np.arange(24.0).reshape(2, 3, 4)
    check(unbroadcast(g, (2, 3, 4)).shape, (2, 3, 4), "形状相同时不变")
    check_close(unbroadcast(g, (4,)), g.sum((0, 1)), what="(2, 3, 4) → (4,)")
    check_close(unbroadcast(g, (3, 1)), g.sum(0).sum(1, keepdims=True), what="(2, 3, 4) → (3, 1)")
    check_close(unbroadcast(g, (2, 1, 4)), g.sum(1, keepdims=True), what="(2, 3, 4) → (2, 1, 4)")
    check_close(unbroadcast(g, ()), g.sum(), what="(2, 3, 4) → 标量")
    check_close(unbroadcast(g, (1, 1, 1)), g.sum(keepdims=True), what="(2, 3, 4) → (1, 1, 1)")


def test_broadcast_grads():
    rng = np.random.default_rng(0)
    a = Tensor(rng.standard_normal((3, 1)), requires_grad=True)
    b = Tensor(rng.standard_normal(4), requires_grad=True)
    c = Tensor(2.0, requires_grad=True)
    ((a * b + c) * (a * b + c)).sum().backward()
    ab = a.data * b.data
    s = ab + c.data
    check(a.grad.shape, (3, 1), "a.grad 的形状和 a 相同")
    check(b.grad.shape, (4,), "b.grad 的形状和 b 相同")
    check(c.grad.shape, (), "标量的梯度还是标量")
    check_close(a.grad, (2 * s * b.data).sum(1, keepdims=True), what="a.grad")
    check_close(b.grad, (2 * s * a.data).sum(0), what="b.grad")
    check_close(c.grad, (2 * s).sum(), what="c.grad")


def test_matmul_mlp_vs_numeric():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 4))
    W1 = Tensor(rng.standard_normal((4, 6)), requires_grad=True)
    b1 = Tensor(rng.standard_normal(6), requires_grad=True)
    W2 = Tensor(rng.standard_normal((6, 3)), requires_grad=True)

    def loss(w1, bb, w2):
        h = (Tensor(X) @ w1 + bb).relu()
        out = h @ w2
        return (out * out).sum()

    loss(W1, b1, W2).backward()
    for name, p in (("W1", W1), ("b1", b1), ("W2", W2)):
        num = np.zeros_like(p.data)
        for idx in np.ndindex(p.shape):
            old = p.data[idx]
            p.data[idx] = old + 1e-6
            up = loss(Tensor(W1.data), Tensor(b1.data), Tensor(W2.data)).data
            p.data[idx] = old - 1e-6
            down = loss(Tensor(W1.data), Tensor(b1.data), Tensor(W2.data)).data
            p.data[idx] = old
            num[idx] = (up - down) / 2e-6
        check_close(p.grad, num, rtol=1e-5, atol=1e-6, what=f"{name} 的梯度与数值差分一致")


def test_leaf_rules():
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    k = Tensor([2.0, 2.0, 2.0])                        # 不需要梯度
    h = x * k
    y = (h * h).sum()                                  # 同一个中间结果在一个节点里用了两次
    y.backward()
    check_close(x.grad, 8 * x.data, what="y = Σ(2x)² 的梯度 8x")
    check(h.grad is None, True, "中间结果的 .grad 保持 None")
    check(k.grad is None, True, "requires_grad=False 的张量不接收梯度")
    y2 = (x * x).sum()                                 # 同一个叶子在一个节点里用了两次
    y2.backward()
    check_close(x.grad, 8 * x.data + 2 * x.data, what="第二次 backward 累加到 .grad 上")


def _count_calls(limit):
    classes = [solution.AddBackward, solution.MulBackward, solution.MatMulBackward, solution.SumBackward, solution.ReluBackward]
    originals = {c: c.__dict__["backward"] for c in classes}
    counter = {"n": 0}

    def wrap(f):
        def w(self, grad):
            counter["n"] += 1
            if counter["n"] > limit:
                raise AssertionError(f"反向节点一共被调用了超过 {limit} 次（图里只有 {limit} 个节点）：有节点被重复调用了")
            return f(self, grad)
        return w

    for c in classes:
        c.backward = wrap(originals[c])
    return counter, lambda: [setattr(c, "backward", originals[c]) for c in classes]


def test_each_node_once():
    x = Tensor(1.5, requires_grad=True)
    y = x
    for _ in range(40):                                 # 40 层"菱形"：递归写法要调用 2^40 次
        y = y * 0.5 + y * 0.5
    counter, restore = _count_calls(limit=40 * 3)
    try:
        y.backward()
    finally:
        restore()
    check(counter["n"], 120, "每个节点恰好调用一次")
    check_close(x.grad, 1.0, what="每层都是 y → y，梯度为 1")


def test_shared_subgraph_grads():
    rng = np.random.default_rng(2)
    a = Tensor(rng.standard_normal((2, 2)), requires_grad=True)
    h = a @ a                                            # h 被下面两处用到
    z = (h.relu() * h + h).sum()
    counter, restore = _count_calls(limit=6)
    try:
        z.backward()
    finally:
        restore()
    hd = a.data @ a.data
    gh = np.where(hd > 0, 2 * hd, 0) + 1                 # d/dh [relu(h)·h + h]
    check_close(a.grad, gh @ a.data.T + a.data.T @ gh, what="共享子图上的梯度")
