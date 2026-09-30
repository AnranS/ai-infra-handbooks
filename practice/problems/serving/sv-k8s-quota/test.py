from checker import check
from solution import admit, gpu_hours, preempt


def test_example():
    check(admit({"gpu": 8}, {"gpu": 6}, {"gpu": 1}), (True, ""), "配额内")
    check(admit({"gpu": 8}, {"gpu": 6}, {"gpu": 4}), (False, "超配额"), "超了")
    check(preempt([{"name": "b1", "priority": 100, "gpu": 2}], 2, 1000), ["b1"], "抢占低优先级")


def test_admit_missing_requests():
    check(admit({"cpu": 10, "gpu": 4}, {"cpu": 0, "gpu": 0}, {"gpu": 1}),
          (False, "缺少 requests"), "开了配额就必须写全")


def test_admit_unknown():
    check(admit({"gpu": 4}, {"gpu": 0}, {"gpu": 1, "tpu": 1}), (False, "未知资源"), "配额里没有这项")


def test_admit_exact():
    check(admit({"gpu": 4}, {"gpu": 3}, {"gpu": 1}), (True, ""), "正好用满")
    check(admit({"gpu": 4}, {"gpu": 4}, {"gpu": 1}), (False, "超配额"), "已经满了")


def test_preempt_priority():
    pods = [{"name": "low", "priority": 100, "gpu": 1},
            {"name": "mid", "priority": 500, "gpu": 4},
            {"name": "high", "priority": 2000, "gpu": 8}]
    check(preempt(pods, 1, 1000), ["low"], "先抢优先级最低的")
    check(preempt(pods, 4, 1000), ["low", "mid"], "不够就继续往上抢")
    check(preempt(pods, 8, 1000), None, "比自己优先级高的抢不动")


def test_preempt_fewest_victims():
    pods = [{"name": "a", "priority": 100, "gpu": 1},
            {"name": "b", "priority": 100, "gpu": 4}]
    check(preempt(pods, 4, 1000), ["b"], "同优先级时先抢占卡多的，少动几个 Pod")


def test_preempt_nothing_to_take():
    check(preempt([], 1, 1000), None, "没有可抢的")
    check(preempt([{"name": "x", "priority": 1000, "gpu": 8}], 1, 1000), None, "同优先级不能抢")


def test_gpu_hours():
    pods = [{"name": "a", "priority": 1000, "gpu": 8},
            {"name": "b", "priority": 1000, "gpu": 4},
            {"name": "c", "priority": 100, "gpu": 2}]
    check(gpu_hours(pods, 24), {1000: 288, 100: 48}, "同优先级要累加")
    check(gpu_hours([], 10), {}, "空集群")
