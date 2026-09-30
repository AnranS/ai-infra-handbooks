from checker import check, check_close
from solution import (capacity_loss, extra_gpus, rollout_batches, rollout_seconds,
                      scale_lag_seconds)


def test_example():
    check(rollout_batches(12, 2, 1), 4, "每批换 3 个")
    check(rollout_seconds(12, 150, 60, 2, 1), 600, "4 批 × 150 秒")
    check(extra_gpus(12, 8, 2), 16, "多两个副本的卡")
    check_close(capacity_loss(12, 1), 0.917, rtol=0.01, what="最少可用 11/12")
    check(scale_lag_seconds(150, image_pull_s=120, node_provision_s=180), 450, "三段都要算")


def test_batches_edges():
    check(rollout_batches(1, 1, 0), 1, "单副本")
    check(rollout_batches(10, 0, 0), 10, "两个参数都是 0 时每批至少换一个")
    check(rollout_batches(7, 1, 1), 4, "不满一批也算一批")
    check(rollout_batches(0, 1, 1), 0, "没有副本")


def test_strategy_tradeoff():
    conservative = rollout_seconds(6, 60, 30, 1, 0)      # 一个一个换
    balanced = rollout_seconds(6, 60, 30, 2, 1)
    blue_green = rollout_seconds(6, 60, 30, 6, 0)
    check(conservative > balanced > blue_green, True, "surge 越大发布越快")
    check(extra_gpus(6, 8, 6) > extra_gpus(6, 8, 1), True, "但需要的额外 GPU 也越多")


def test_capacity():
    check_close(capacity_loss(10, 0), 1.0, rtol=1e-9, what="maxUnavailable=0 时容量不降")
    check_close(capacity_loss(10, 5), 0.5, rtol=1e-9, what="一半")
    check_close(capacity_loss(3, 5), 0.0, rtol=1e-9, what="上限超过副本数时全部可能不可用")
    check(capacity_loss(0, 1), 0.0, "没有副本")


def test_ready_time_dominates():
    fast = rollout_seconds(12, 10, 30, 2, 1)
    slow = rollout_seconds(12, 300, 30, 2, 1)
    check(slow / fast, 10.0, "启动时间直接乘进发布耗时")


def test_scale_lag():
    check(scale_lag_seconds(60), 60, "只有启动")
    check(scale_lag_seconds(60, image_pull_s=240), 300, "加上拉镜像")
    check(scale_lag_seconds(60, 240, 300) > 9 * 60, True, "要等开机器时延迟接近十分钟")
