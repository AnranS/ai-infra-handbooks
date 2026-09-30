from checker import check
from solution import eager_bytes, fused_bytes, fusion_groups, plan_bytes

SMALL = [("x", "input", [], 64), ("sq", "pointwise", ["x"], 64), ("s", "reduce", ["sq"], 4),
         ("y", "pointwise", ["x", "s"], 64)]

# 正文 fx_fusion.py 里的 RMSNorm + SiLU + 乘法：x、up 是 [8, 4096] 的 fp32，w 是 [4096]
RMS = [("x", "input", [], 131072), ("w", "input", [], 16384), ("up", "input", [], 131072),
       ("pow", "pointwise", ["x"], 131072), ("mean", "reduce", ["pow"], 32), ("add", "pointwise", ["mean"], 32),
       ("rsqrt", "pointwise", ["add"], 32), ("mul1", "pointwise", ["x", "rsqrt"], 131072),
       ("mul2", "pointwise", ["mul1", "w"], 131072), ("silu", "pointwise", ["mul2"], 131072),
       ("mul3", "pointwise", ["silu", "up"], 131072)]


def test_example():
    check(eager_bytes(SMALL), 328, "小例子：逐个执行")
    check(fused_bytes(SMALL, {"sq", "s", "y"}), 128, "小例子：融合")


def test_rmsnorm():
    check(eager_bytes(RMS), 1589440, "正文：8 个算子逐个执行")
    fused = [n for n, k, *_ in RMS if k != "input"]
    check(fused_bytes(RMS, set(fused)), 409600, "正文：融合成一个 kernel")
    check(fusion_groups(RMS), [fused], "整条链是一组")
    check(plan_bytes(RMS), 409600, "按分组融合后的整张图")


def test_partial_and_shared():
    check(fused_bytes(SMALL, {"sq", "s"}), 64 + 4, "只融合前两个：读 x、写 s")
    shared = SMALL + [("z", "pointwise", ["s"], 4)]
    check(fused_bytes(shared, {"sq", "s", "y"}), 64 + 4 + 64, "s 还被组外的 z 使用：s 也要写回显存")
    check(fused_bytes(SMALL, {"y"}), 68 + 64, "单个节点的组就是 eager 的账")


def test_groups_with_matmul():
    g = [("x", "input", [], 100), ("W", "input", [], 1000), ("h", "matmul", ["x", "W"], 200),
         ("b", "input", [], 20), ("hb", "pointwise", ["h", "b"], 200), ("act", "pointwise", ["hb"], 200),
         ("W2", "input", [], 1000), ("o", "matmul", ["act", "W2"], 100), ("r", "pointwise", ["o", "x"], 100),
         ("n", "reduce", ["r"], 4), ("out", "pointwise", ["r", "n"], 100)]
    check(fusion_groups(g), [["hb", "act"], ["r", "n", "out"]], "矩阵乘把图分成两组")
    eager = eager_bytes(g)
    check(eager, (1100 + 200) + (220 + 200) + (200 + 200) + (1200 + 100) + (200 + 100) + (100 + 4) + (104 + 100), "eager")
    check(plan_bytes(g), (1100 + 200) + (220 + 200) + (1200 + 100) + (200 + 100), "融合后：两个矩阵乘 + 两组")
    two = [("a", "input", [], 8), ("p", "pointwise", ["a"], 8), ("b", "input", [], 8), ("q", "pointwise", ["b"], 8)]
    check(fusion_groups(two), [["p"], ["q"]], "没有边相连的逐元素运算各自成组")
