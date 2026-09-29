from checker import check, check_close, raises
from solution import max_batch_by_kv, plan, step_ms

ARGS = (141e9, 327680, 8, 80, 3350)


def test_example():
    r = plan(200, 2000, 500, 50, *ARGS)
    check(r["batch"], 546, "并发数受 KV 容量限制")
    check_close(r["step_ms"], (141e9 + 546 * 2250 * 327680) / (8 * 3350e9) * 1e3, rtol=1e-12, what="单步时间")
    check_close(r["rps_per_instance"], 546 / (500 * r["step_ms"] / 1e3), rtol=1e-12, what="Little 定律")
    check((r["instances"], r["gpus"]), (6, 48), "实例数与卡数")


def test_helpers():
    check(max_batch_by_kv(403e9, 2250, 327680), 546, "KV 预算")
    check(max_batch_by_kv(-1e9, 2250, 327680), 0, "权重就放不下时为 0")
    check_close(step_ms(0, 141e9, 327680, 2250, 26800), 141e9 / 26800e9 * 1e3, what="只读权重")
    check_close(step_ms(10, 1e9, 1000, 100, 1000, overhead_ms=2), 1.0 + 0.001 + 2, rtol=1e-9, what="加上固定开销")


def test_tpot_bound():
    r = plan(200, 2000, 500, 10, *ARGS)
    limit = int((10 / 1e3 * 8 * 3350e9 - 141e9) // (2250 * 327680))
    check(r["batch"], limit, "TPOT 很严时并发数受带宽限制")
    assert r["step_ms"] <= 10 + 1e-9, "单步时间不能超过 TPOT"
    assert r["gpus"] > plan(200, 2000, 500, 50, *ARGS)["gpus"], "TPOT 更严，需要更多卡"


def test_fp8_kv_helps():
    bf16 = plan(200, 2000, 500, 50, *ARGS)
    fp8 = plan(200, 2000, 500, 50, 141e9, 163840, 8, 80, 3350)
    assert fp8["batch"] > 1.9 * bf16["batch"], "FP8 KV 让并发数翻倍"
    assert fp8["rps_per_instance"] > 1.3 * bf16["rps_per_instance"], "单实例吞吐明显上升"


def test_infeasible():
    with raises(ValueError, "TPOT 低于读一遍权重的时间"):
        plan(10, 2000, 500, 4, *ARGS)
    with raises(ValueError, "单卡放不下 70B 的权重"):
        plan(10, 2000, 500, 50, 141e9, 327680, 1, 80, 3350)
