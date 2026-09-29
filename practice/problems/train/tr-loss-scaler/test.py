import numpy as np

from checker import check, check_close
from solution import GradScaler, fp16_backward, underflow_fraction


def test_example():
    g = [np.array([1e-8, 1e-6, 1e-3])]
    check_close(underflow_fraction(g, 1.0), 1 / 3, what="不缩放时 1e-8 下溢成 0")
    check_close(underflow_fraction(g, 2.0**16), 0.0, what="乘 2^16 之后都能表示")


def test_unscale_recovers_small_grads():
    s = GradScaler()
    true = [np.array([1e-8, 1e-6, 2e-5, -3e-7], dtype=np.float32)]
    grads, found_inf = s.unscale(fp16_backward(true, s.scale))
    check(found_inf, False, "没有溢出")
    check(grads[0].dtype, np.dtype(np.float32), "反缩放后是 fp32")
    check_close(grads[0], true[0], rtol=2e-3, atol=0, what="小梯度被找回来了（fp16 的相对精度约 1e-3）")
    check(bool((fp16_backward(true, 1.0)[0][0]) == 0), True, "（对照）不缩放时 1e-8 变成 0")


def test_overflow_skips_and_backs_off():
    s = GradScaler(init_scale=2.0**16)
    p = [np.ones(3, dtype=np.float32)]
    g16 = fp16_backward([np.array([1.0, 0.5, 1.5])], s.scale)       # 1.0 × 65536 > 65504：溢出
    check(bool(np.isinf(g16[0]).any()), True, "（前提）fp16 里出现了 inf")
    check(s.step(p, g16, lr=0.1), False, "有 inf 时跳过这一步")
    check(p[0].tolist(), [1.0, 1.0, 1.0], "参数不变")
    check(s.scale, 2.0**15, "scale 减半")
    g16 = fp16_backward([np.array([1.0, 0.5, 1.5])], s.scale)
    check(s.step(p, g16, lr=0.1), True, "缩放减半后不再溢出")
    check_close(p[0], [0.9, 0.95, 0.85], rtol=1e-6, what="按真实梯度更新")


def test_growth_interval():
    s = GradScaler(init_scale=8.0, growth_interval=3)
    p = [np.zeros(2, dtype=np.float32)]
    for i in range(3):
        s.step(p, fp16_backward([np.array([0.1, 0.2])], s.scale), lr=1.0)
    check(s.scale, 16.0, "连续 3 步正常之后翻倍")
    s.step(p, fp16_backward([np.array([0.1, 0.2])], s.scale), lr=1.0)
    s.step(p, fp16_backward([np.array([np.inf, 0.2])], s.scale), lr=1.0)
    check(s.scale, 8.0, "溢出时减半")
    for i in range(2):
        s.step(p, fp16_backward([np.array([0.1, 0.2])], s.scale), lr=1.0)
    check(s.scale, 8.0, "溢出之后计数清零：再正常 2 步还不会翻倍")
    s.step(p, fp16_backward([np.array([0.1, 0.2])], s.scale), lr=1.0)
    check(s.scale, 16.0, "第 3 步正常之后翻倍")


def test_dynamic_equilibrium():
    rng = np.random.default_rng(0)
    s = GradScaler(init_scale=2.0**20, growth_interval=10)
    p = [np.zeros(100, dtype=np.float32)]
    skipped = 0
    for step in range(100):
        g = rng.uniform(-1, 1, 100)
        g[0] = 10.0                                                  # 最大的梯度是 10
        if not s.step(p, fp16_backward([g], s.scale), lr=1e-3):
            skipped += 1
    check(skipped, 16, "先连续溢出 8 次降到 4096，之后每 11 步（10 步正常 + 翻倍后溢出 1 次）跳过一步")
    check(s.scale, 4096.0, "最后停在不溢出的最大缩放附近")
