import random

from checker import check, raises
from solution import first_divergence, guess_cause


def make(layers, positions, dim, seed=0):
    rng = random.Random(seed)
    return [[[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(positions)] for _ in range(layers)]


def test_example():
    ours = [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [5.0, 1.0]]]
    ref = [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]
    check(first_divergence(ours, ref), (1, 1), "第 1 层第 1 个位置")
    check(guess_cause(512, {"sliding_window": 512}), "sliding_window", "从 512 开始错")
    check(guess_cause(1, {"sliding_window": 512}), "position_dependent", "位置 0 正常、从 1 开始错")


def test_divergence():
    ref = make(6, 20, 8)
    check(first_divergence([[row[:] for row in layer] for layer in ref], ref), None, "完全相同")
    ours = [[[v + 1e-5 for v in row] for row in layer] for layer in ref]
    check(first_divergence(ours, ref), None, "浮点误差在阈值内")
    check(first_divergence(ours, ref, tol=1e-6), (0, 0), "阈值更严时第 0 层第 0 个位置就算错")
    ours = [[row[:] for row in layer] for layer in ref]
    for layer in range(3, 6):                   # 第 3 层起、从第 12 个位置开始错，后面的层也跟着错
        for pos in range(12, 20):
            ours[layer][pos][5] += 0.1
    ours[5][2][0] += 1.0                        # 更靠后的层里即使有更早的位置出错，也不是"第一处"
    check(first_divergence(ours, ref), (3, 12), "第一处对不上的层和位置")


def test_shapes_and_causes():
    ref = make(4, 5, 3)
    with raises(ValueError, "层数不同"):
        first_divergence(ref[:3], ref)
    with raises(ValueError, "位置数不同"):
        first_divergence([layer[:4] for layer in ref], ref)
    bounds = {"sliding_window": 512, "chunk": 2048, "page": 16, "block": 16}
    check(guess_cause(0, bounds), "all_positions", "位置 0 就错")
    check(guess_cause(2048, bounds), "chunk", "分块 prefill 的边界")
    check(guess_cause(16, bounds), "page", "多个边界相等时取靠前的")
    check(guess_cause(300, bounds), "position_dependent", "不在任何边界上")
