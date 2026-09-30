from checker import check, check_close
from solution import erlang_c, max_utilization, servers_needed, wait_ms


def test_example():
    check_close(erlang_c(4, 0.7), 0.429, rtol=0.01, what="4 台、70% 利用率时的排队概率")
    check_close(wait_ms(4, 0.7, 10), 3.57, rtol=0.02, what="平均排队 3.6 ms")
    check_close(max_utilization(16, 10, 5), 0.918, rtol=0.01, what="16 台、排队预算 5 ms")
    check(servers_needed(200, 50, 10), 13, "200 请求/秒、服务 50 ms、排队 10 ms 以内")


def test_single_server():
    # M/M/1：排队概率就是利用率，平均排队 = rho/(1-rho) * 服务时间
    check_close(erlang_c(1, 0.8), 0.8, rtol=1e-6, what="单台的排队概率等于利用率")
    check_close(wait_ms(1, 0.8, 10), 40.0, rtol=1e-6, what="单台 80% 利用率排队 40 ms")
    check_close(wait_ms(1, 0.5, 10), 10.0, rtol=1e-6, what="单台 50% 利用率排队 10 ms")


def test_nonlinear_growth():
    w = [wait_ms(4, r, 10) for r in (0.5, 0.7, 0.9, 0.95)]
    check(all(b > a for a, b in zip(w, w[1:])), True, "利用率越高排队越久")
    check(w[3] / w[0] > 20, True, f"50% 到 95% 涨了 {w[3] / w[0]:.0f} 倍")


def test_bigger_pool_is_better():
    ws = [wait_ms(c, 0.9, 10) for c in (1, 4, 16)]
    check(all(b < a for a, b in zip(ws, ws[1:])), True, "同样利用率，机器越多排队越短")
    check_close(ws[0], 90.0, rtol=0.01, what="1 台 90% 利用率排队 90 ms")
    check_close(ws[2], 3.7, rtol=0.05, what="16 台只要 3.7 ms")


def test_max_utilization_monotone():
    us = [max_utilization(c, 10, 5) for c in (1, 4, 16, 64)]
    check(all(b > a for a, b in zip(us, us[1:])), True, "池子越大，能跑的利用率越高")
    check(us[0] < 0.5, True, "单台要把排队压到 5 ms，利用率只能到三分之一左右")


def test_servers_needed_edges():
    check(servers_needed(10, 50, 100), 1, "负载很轻时一台就够")
    check(servers_needed(200, 50, 1000) < servers_needed(200, 50, 1), True, "预算越紧需要的机器越多")
    check(servers_needed(200, 50, 50) >= 11, True, "至少要 10 台才跟得上到达率")
