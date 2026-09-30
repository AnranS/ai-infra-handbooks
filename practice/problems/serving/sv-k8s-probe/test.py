from checker import check
from solution import restart_loop, shutdown_seconds, startup_budget, truncated


def test_example():
    check(startup_budget(10, 30), 300, "10 秒一次、允许失败 30 次")
    check(restart_loop(180, 20, 3, has_startup_probe=False), True, "60 秒预算撑不住 180 秒加载")
    check(restart_loop(180, 20, 3, has_startup_probe=True), False, "有 startupProbe 就没事")
    check(shutdown_seconds(5, 120, 60), 65, "宽限期到了就强杀")
    check(truncated(5, 120, 60), True, "会截断")


def test_budget():
    check(startup_budget(5, 60), 300, "5 秒一次、60 次")
    check(startup_budget(1, 1), 1, "最小情形")


def test_restart_loop_边界():
    check(restart_loop(60, 20, 3, has_startup_probe=False), False, "正好等于预算不算超")
    check(restart_loop(61, 20, 3, has_startup_probe=False), True, "超一秒就被杀")
    check(restart_loop(300, 20, 3, True, startup_period_s=10, startup_failures=30), False, "正好用满 startup 预算")
    check(restart_loop(301, 20, 3, True, startup_period_s=10, startup_failures=30), True, "startup 预算也有上限")


def test_shutdown():
    check(shutdown_seconds(0, 10, 60), 10, "请求很快做完")
    check(shutdown_seconds(5, 10, 60), 15, "preStop 不占用宽限期")
    check(shutdown_seconds(5, 600, 60), 65, "长回答被强杀")
    check(truncated(5, 10, 60), False, "做得完就不会截断")


def test_realistic():
    # 权重加载 4 分钟、最长回答 3 分钟的推理服务
    check(restart_loop(240, 20, 3, has_startup_probe=True, startup_period_s=10, startup_failures=30),
          False, "startup 预算 300 秒够用")
    check(truncated(5, 180, 60), True, "宽限期 60 秒不够一个长回答")
    check(truncated(5, 180, 240), False, "宽限期设成 240 秒就够了")
