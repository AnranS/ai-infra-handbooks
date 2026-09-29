from checker import check, check_close, raises
from solution import colocated_s, disaggregated_s, max_instances


def close_all(got, want, what, **kw):
    """逐个比较（不依赖 numpy）"""
    got, want = list(got), list(want)
    check(len(got), len(want), f"{what}：个数")
    for i, (g, w) in enumerate(zip(got, want)):
        check_close(g, w, what=f"{what}（第 {i + 1} 个）", **kw)

W, NIC, NVLINK = 1e12, 50e9, 450e9


def test_example():
    check_close(colocated_s(W, 8, NVLINK), 1e12 / 8 / 450e9, what="共置：每卡 1/8，走 NVLink")
    check_close(disaggregated_s(W, 64, NIC, 8, "relay"), 2.625, what="64 个实例、流水接力")


def test_modes():
    close_all([disaggregated_s(W, n, NIC, 8, "naive") for n in (1, 8, 64)], [20.0, 160.0, 1280.0], what="朴素")
    close_all([disaggregated_s(W, n, NIC, 8, "parallel") for n in (1, 8, 64)], [2.5, 20.0, 160.0], what="8 卡并行")
    close_all([disaggregated_s(W, n, NIC, 8, "relay") for n in (1, 8, 64)], [2.5, 2.625, 2.625], what="接力")
    check_close(disaggregated_s(2e11, 4, 25e9, 4, "parallel"), 8.0, what="换一组参数")
    with raises(ValueError, "不认识的方式"):
        disaggregated_s(W, 8, NIC, 8, "broadcast")


def test_max_instances():
    check(max_instances(60, W, NIC, 8, "naive"), 3, "朴素：60 秒只够发 3 个实例")
    check(max_instances(60, W, NIC, 8, "parallel"), 24, "8 卡并行：60 / 2.5 = 24")
    check(max_instances(20, W, NIC, 8, "parallel"), 8, "正好整除时也算")
    check(max_instances(60, W, NIC, 8, "relay"), None, "接力：没有上限")
    check(max_instances(2.55, W, NIC, 8, "relay"), 1, "接力的 5% 开销放不下时只能带一个")
    check(max_instances(1.0, W, NIC, 8, "relay"), 0, "预算连一个实例都不够")
    check(max_instances(10, W, NIC, 8, "naive"), 0, "朴素：10 秒一个都发不完")
