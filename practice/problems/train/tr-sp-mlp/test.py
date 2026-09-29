import numpy as np

from checker import check, check_close
from solution import Comm, act, act_grad, sp_backward, sp_forward


def serial(x, g, W1, W2, dout, eps=1e-6):
    rms = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    xh = x / rms
    y = xh * g
    h = y @ W1.T
    a = act(h)
    out = x + a @ W2.T
    dW2 = dout.T @ a
    dh = (dout @ W2) * act_grad(h)
    dW1 = dh.T @ y
    dy = dh @ W1
    dg = (dy * xh).sum(0)
    dxh = dy * g
    dx = dout + (dxh - xh * (dxh * xh).mean(-1, keepdims=True)) / rms
    return out, dx, dg, dW1, dW2


def run(t, s=8, h=6, f=8, seed=0):
    rng = np.random.default_rng(seed)
    x, dout = rng.standard_normal((s, h)), rng.standard_normal((s, h))
    g = rng.random(h) + 0.5
    W1, W2 = rng.standard_normal((f, h)) / 3, rng.standard_normal((h, f)) / 3
    comm = Comm(t)
    out, cache = sp_forward(np.split(x, t), g, np.split(W1, t, axis=0), np.split(W2, t, axis=1), comm)
    fwd_log = list(comm.log)
    dx, dg, dW1, dW2 = sp_backward(np.split(dout, t), cache, comm)
    return (x, g, W1, W2, dout), (out, dx, dg, dW1, dW2), fwd_log, comm.log[len(fwd_log):]


def test_example():
    (x, g, W1, W2, dout), (out, dx, dg, dW1, dW2), _, _ = run(t=2)
    ref = serial(x, g, W1, W2, dout)
    check_close(np.concatenate(out), ref[0], rtol=1e-10, atol=1e-10, what="前向输出（按序列拼起来）")


def test_backward_matches_serial():
    for t in (1, 2, 4):
        (x, g, W1, W2, dout), (out, dx, dg, dW1, dW2), _, _ = run(t=t, seed=t)
        ref_out, ref_dx, ref_dg, ref_dW1, ref_dW2 = serial(x, g, W1, W2, dout)
        check_close(np.concatenate(out), ref_out, rtol=1e-10, atol=1e-10, what=f"t={t}：前向")
        check_close(np.concatenate(dx), ref_dx, rtol=1e-9, atol=1e-10, what=f"t={t}：dx")
        check_close(dg, ref_dg, rtol=1e-9, atol=1e-10, what=f"t={t}：RMSNorm 权重的梯度 dg")
        check_close(np.concatenate(dW1, axis=0), ref_dW1, rtol=1e-9, atol=1e-10, what=f"t={t}：dW1（按行拼起来）")
        check_close(np.concatenate(dW2, axis=1), ref_dW2, rtol=1e-9, atol=1e-10, what=f"t={t}：dW2（按列拼起来）")


def test_communication_pattern():
    s, h = 8, 6
    _, _, fwd, bwd = run(t=4, s=s, h=h)
    check(fwd, [("all_gather", s * h), ("reduce_scatter", s * h)], "前向：一次 all_gather + 一次 reduce_scatter")
    check(sorted(bwd), sorted([("all_gather", s * h), ("reduce_scatter", s * h), ("all_reduce", h)]),
          "反向：all_gather（reduce_scatter 的反向）+ reduce_scatter（all_gather 的反向）+ dg 的 all_reduce")


def test_numeric_gradient():
    rng = np.random.default_rng(9)
    s, h, f, t = 4, 3, 4, 2
    x, g = rng.standard_normal((s, h)), rng.random(h) + 0.5
    W1, W2 = rng.standard_normal((f, h)), rng.standard_normal((h, f))
    w = rng.standard_normal((s, h))                            # loss = Σ w ⊙ out

    def loss(xx, gg):
        out, _ = sp_forward(np.split(xx, t), gg, np.split(W1, t, axis=0), np.split(W2, t, axis=1), Comm(t))
        return float((np.concatenate(out) * w).sum())

    out, cache = sp_forward(np.split(x, t), g, np.split(W1, t, axis=0), np.split(W2, t, axis=1), Comm(t))
    dx, dg, _, _ = sp_backward(np.split(w, t), cache, Comm(t))
    num = np.zeros(h)
    for i in range(h):
        e = np.zeros(h)
        e[i] = 1e-6
        num[i] = (loss(x, g + e) - loss(x, g - e)) / 2e-6
    check_close(dg, num, rtol=1e-5, atol=1e-7, what="dg 与数值差分一致")
