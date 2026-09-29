from checker import check, raises
from solution import choose, divide_sm, prefill_layers_per_step, stream_groups

H100 = [(132, 0), (112, 20), (104, 28), (96, 36), (88, 44), (80, 52), (72, 60), (0, 132)]


def test_example():
    check(stream_groups(132, 9), H100, "H100（132 个 SM，sm_90）的分组")
    check(choose(H100, 32, True), 5, "32 个 decode 请求 + 有 prefill")


def test_divide():
    check(divide_sm(108, 8, 6), [(84, 24), (78, 30), (72, 36), (66, 42), (60, 48), (54, 54)], "A100：粒度 2")
    check(divide_sm(78, 9, 6), [(56, 22), (48, 30), (40, 38)], "H20：候选不够 6 个时全部返回")
    check(divide_sm(132, 9, 3), [(104, 28), (88, 44), (72, 60)], "候选 6 个取 3 个：步长 2")
    check(divide_sm(132, 9, 4), [(96, 36), (88, 44), (80, 52), (72, 60)], "候选 6 个取 4 个：步长 1，取前 4 个")
    check(divide_sm(80, 7, 2), [(52, 28), (40, 40)], "sm_7x：13 个候选、步长 6")
    check(divide_sm(40, 6, 30)[:3], [(24, 16), (23, 17), (22, 18)], "sm_6x：粒度 1")
    with raises(ValueError, "SM 太少，decode 分不到 16 个"):
        divide_sm(24, 9, 6)
    with raises(ValueError, "不支持的计算能力"):
        divide_sm(132, 10, 6)


def test_groups():
    g = stream_groups(78, 9)
    check(g, [(78, 0), (56, 22), (48, 30), (40, 38), (0, 78)], "H20 一共 5 组")
    check(stream_groups(132, 9, sm_group_num=4), [(132, 0), (96, 36), (72, 60), (0, 132)], "只切 2 档：步长 3")


def test_choose():
    got = [choose(H100, bs, True) for bs in (1, 6, 11, 12, 18, 24, 30, 35, 36, 64)]
    check(got, [1, 1, 1, 2, 3, 4, 5, 5, 6, 6], "按 decode 请求数线性选档")
    check(choose(H100, 32, False), 7, "只有 decode：整卡给 decode")
    check(choose(H100, 0, True), 0, "只有 prefill：整卡给 prefill")
    check(choose(H100, 0, False), 0, "空闲")
    check(choose(H100, 24, True, decode_bs_divisor=24), 6, "divisor 改成 24")
    g5 = stream_groups(78, 9)
    check([choose(g5, bs, True) for bs in (5, 12, 24, 40)], [1, 1, 2, 3], "H20：只有 3 档")
    th = [0, 8, 16, 24, 32, 48]
    check([choose(H100, bs, True, thresholds=th) for bs in (1, 8, 20, 47, 48, 100)], [1, 2, 3, 5, 6, 6], "手工阈值")
    check(choose(H100, 3, True, thresholds=[4, 8, 16, 24, 32, 48]), 1, "低于所有阈值时用第 1 组")


def test_layers():
    check([prefill_layers_per_step(t, 36) for t in (512, 8192, 32768, 131072)], [36, 8, 2, 1], "每轮的层数")
    check(prefill_layers_per_step(8192, 36, done_layers=32), 4, "最后一轮只剩 4 层")
    check(prefill_layers_per_step(0, 36, done_layers=10), 26, "没有新 token 时一次算完剩下的")
    check(prefill_layers_per_step(1000, 28, token_budget=10000), 10, "换一个预算")
