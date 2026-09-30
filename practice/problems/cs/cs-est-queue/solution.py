from math import factorial


def erlang_c(c, rho):
    a = c * rho
    top = a ** c / factorial(c) / (1 - rho)
    return top / (sum(a ** k / factorial(k) for k in range(c)) + top)


def wait_ms(c, rho, service_ms):
    return erlang_c(c, rho) * service_ms / (c * (1 - rho))


def max_utilization(c, service_ms, budget_ms):
    lo, hi = 0.0, 1.0
    if wait_ms(c, 0.001, service_ms) > budget_ms:
        return 0.0
    for _ in range(40):                                 # 二分：排队时间随利用率单调递增
        mid = (lo + hi) / 2
        if wait_ms(c, mid, service_ms) <= budget_ms:
            lo = mid
        else:
            hi = mid
    return round(lo, 3)


def servers_needed(rps, service_ms, budget_ms):
    load = rps * service_ms / 1000                      # 需要的"服务器·秒"，也就是最少服务器数
    c = max(1, int(load) + 1)
    while True:
        rho = load / c
        if rho < 1 and wait_ms(c, rho, service_ms) <= budget_ms:
            return c
        c += 1
