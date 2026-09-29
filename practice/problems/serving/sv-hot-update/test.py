import numpy as np

from checker import check, check_close, raises
from solution import hot_update, pause_costs, quantize


class Layer:
    def __init__(self, w):
        self.qweight, self.scale = quantize(w)

    def __call__(self, x):
        return x @ (self.qweight.astype(np.float32) * self.scale).T


def test_example():
    rng = np.random.default_rng(0)
    old, new = rng.standard_normal((16, 32)) * 0.05, rng.standard_normal((16, 32)) * 0.05
    layer = Layer(old)
    q_obj, s_obj = layer.qweight, layer.scale
    hot_update(layer, new)
    check(layer.qweight is q_obj and layer.scale is s_obj, True, "还是原来那两个数组")
    check(layer.qweight, quantize(new)[0], "内容是新权重的量化结果")
    check_close(pause_costs("wait", [10, 20], [100, 300], 2048, 25.0)["wait_s"], 7.5, what="等最长的请求")


def test_quantize():
    w = np.array([[0.5, -1.0, 0.25], [0.0, 0.0, 0.0], [2.0, 1.0, -0.5]], dtype=np.float32)
    q, s = quantize(w)
    check(q.dtype, np.dtype(np.int8), "qweight 的类型")
    check(s.dtype, np.dtype(np.float32), "scale 的类型")
    check(s.shape, (3, 1), "scale 的形状")
    check(q, np.array([[64, -127, 32], [0, 0, 0], [127, 64, -32]], dtype=np.int8), "量化结果（半数取偶）")
    check_close(s[:, 0], np.array([1 / 127, 1.0, 2 / 127], dtype=np.float32), what="缩放系数；全零行取 1")


def test_in_place():
    rng = np.random.default_rng(1)
    layer = Layer(rng.standard_normal((64, 64)) * 0.05)
    q_buf, s_buf = layer.qweight, layer.scale
    q_ptr = layer.qweight.__array_interface__["data"][0]
    new = rng.standard_normal((64, 64)) * 0.05
    hot_update(layer, new)
    check(layer.qweight.__array_interface__["data"][0], q_ptr, "存储地址没变")
    check(np.shares_memory(layer.scale, s_buf), True, "scale 也是原地更新")
    x = rng.standard_normal((8, 64)).astype(np.float32)
    ref = x @ new.T
    err = np.linalg.norm(layer(x) - ref) / np.linalg.norm(ref)
    check(bool(err < 0.02), True, f"更新后的输出和新权重一致（相对误差 {err:.4f}）")
    check(bool(np.array_equal(q_buf, quantize(new)[0])), True, "旧的引用看到的也是新值（就像 CUDA Graph）")


def test_shape_mismatch():
    rng = np.random.default_rng(2)
    layer = Layer(rng.standard_normal((8, 16)))
    before_q, before_s = layer.qweight.copy(), layer.scale.copy()
    with raises(ValueError, "形状对不上"):
        hot_update(layer, rng.standard_normal((8, 32)))
    with raises(ValueError, "能广播的形状也要拒绝"):
        hot_update(layer, rng.standard_normal((1, 16)))
    check(layer.qweight, before_q, "出错时 qweight 不变")
    check(layer.scale, before_s, "出错时 scale 不变")


def test_pause_costs():
    done, left = [100, 200, 300, 400], [50, 5000, 10, 800]
    check(pause_costs("abort", done, left, 1000, 20.0),
          {"wait_s": 0.0, "redecode": 1000, "reprefill": 4000, "mixed": 0, "kv_fresh": True}, "abort")
    got = pause_costs("wait", done, left, 1000, 20.0)
    check_close(got["wait_s"], 100.0, what="wait：最长的请求还要 5000 步")
    check((got["redecode"], got["reprefill"], got["mixed"], got["kv_fresh"]), (0, 0, 0, True), "wait 的其他代价")
    check(pause_costs("keep", done, left, 1000, 20.0),
          {"wait_s": 0.0, "redecode": 0, "reprefill": 0, "mixed": 4, "kv_fresh": False}, "keep")
    check(pause_costs("retract", done, left, 1000, 20.0),
          {"wait_s": 0.0, "redecode": 0, "reprefill": 5000, "mixed": 4, "kv_fresh": True}, "retract")
    with raises(ValueError, "不认识的方式"):
        pause_costs("drop", done, left, 1000, 20.0)
