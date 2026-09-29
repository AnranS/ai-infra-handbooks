import numpy as np

from checker import check, check_close
from solution import kd_grad, kd_loss, log_softmax, reverse_kl, topk_kd_loss


def naive_kl(p, q):
    return float((p * np.log(p / q)).sum(-1).mean())


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_example():
    s, t = np.array([[1.0, 2.0, 3.0]]), np.array([[3.0, 2.0, 1.0]])
    check_close(kd_loss(s, s, 2.0), 0.0, atol=1e-12, what="学生和老师相同时损失为 0")
    check_close(kd_loss(s, t, 1.0), naive_kl(softmax(t), softmax(s)), rtol=1e-9, what="T=1 时就是 KL(p‖q)")


def test_log_softmax_and_stability():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4, 10)) * 3
    check_close(log_softmax(z), np.log(softmax(z)), rtol=1e-10, atol=1e-12, what="log_softmax")
    check_close(log_softmax(z, T=4.0), np.log(softmax(z / 4)), rtol=1e-10, atol=1e-12, what="带温度")
    big = np.array([[1e4, 0.0, -1e4], [5e3, 5e3, 0.0]])
    out = log_softmax(big)
    check(bool(np.isfinite(out[:, :2]).all()), True, "logits 很大时不溢出")
    check_close(out[1, :2], [np.log(0.5)] * 2, rtol=1e-12, what="两个一样大的 logit 各占一半")
    s2, t2 = big + 1.0, big[::-1].copy()
    for f in (lambda: kd_loss(s2, t2, 1.0), lambda: reverse_kl(s2, t2), lambda: kd_loss(s2, t2, 3.0)):
        check(bool(np.isfinite(f())), True, "极端 logits 下损失仍然有限")


def test_kd_loss_and_grad():
    rng = np.random.default_rng(1)
    s, t = rng.standard_normal((5, 7)) * 2, rng.standard_normal((5, 7)) * 2
    for T in (1.0, 2.0, 5.0):
        check_close(kd_loss(s, t, T), T * T * naive_kl(softmax(t / T), softmax(s / T)), rtol=1e-9, what=f"T={T} 的损失")
        g = kd_grad(s, t, T)
        num = np.zeros_like(s)
        for i in np.ndindex(s.shape):
            e = np.zeros_like(s)
            e[i] = 1e-6
            num[i] = (kd_loss(s + e, t, T) - kd_loss(s - e, t, T)) / 2e-6
        check_close(g, num, rtol=1e-5, atol=1e-8, what=f"T={T} 的梯度与数值差分一致")


def test_high_temperature_limit():
    rng = np.random.default_rng(2)
    s, t = rng.standard_normal((3, 6)), rng.standard_normal((3, 6))
    d = s - t
    want = (d - d.mean(-1, keepdims=True)) / (6 * 3)
    check_close(kd_grad(s, t, 1e4), want, rtol=5e-3, atol=1e-7, what="高温时梯度趋向 (s - t 去均值) / (V·N)，与 T 无关")


def test_forward_vs_reverse():
    teacher = np.array([[10.0, 10.0, 0.0, 0.0]])        # 两个峰
    cover = np.array([[2.0, 2.0, 1.0, 1.0]])            # 覆盖两个峰，但在其他地方也放了不少概率
    seek = np.array([[10.0, 0.0, 0.0, 0.0]])            # 只抓住一个峰
    check(bool(kd_loss(cover, teacher, 1.0) < kd_loss(seek, teacher, 1.0)), True, "正向 KL 偏爱覆盖所有峰的学生")
    check(bool(reverse_kl(seek, teacher) < reverse_kl(cover, teacher)), True, "反向 KL 偏爱只抓一个峰的学生")
    check_close(reverse_kl(seek, teacher), np.log(2), rtol=1e-3, what="只抓一个峰：反向 KL 约为 log 2")


def test_topk():
    rng = np.random.default_rng(3)
    N, V, k = 4, 50, 5
    student, teacher = rng.standard_normal((N, V)), rng.standard_normal((N, V)) * 3
    lt = log_softmax(teacher)
    ids = np.argsort(-teacher, axis=1)[:, :k]
    got = topk_kd_loss(student, ids, np.take_along_axis(lt, ids, 1))
    pt = softmax(np.take_along_axis(teacher, ids, 1))
    qs = np.take_along_axis(softmax(student), ids, 1)
    check_close(got, float((pt * np.log(pt / qs)).sum(-1).mean()), rtol=1e-9, what="top-k 蒸馏损失")
    full_ids = np.argsort(-teacher, axis=1)
    check_close(topk_kd_loss(student, full_ids, np.take_along_axis(lt, full_ids, 1)), kd_loss(student, teacher, 1.0),
                rtol=1e-9, what="k = V 时等于完整的 KL")
    check(bool(got >= 0), True, "损失非负")
