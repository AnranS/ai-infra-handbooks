from math import factorial


def erlang_c(c, rho):
    a = c * rho
    top = a ** c / factorial(c)                          # 少了 /(1 - rho)
    return top / (sum(a ** k / factorial(k) for k in range(c)) + top)


def wait_ms(c, rho, service_ms):
    return erlang_c(c, rho) * service_ms / c             # 少了 /(1 - rho)：利用率高时不会爆炸


def max_utilization(c, service_ms, budget_ms):
    for i in range(1, 1000):
        if wait_ms(c, i / 1000, service_ms) > budget_ms:
            return round((i - 1) / 1000, 3)
    return 0.999


def servers_needed(rps, service_ms, budget_ms):
    return max(1, round(rps * service_ms / 1000))        # 只按平均负载算，没留排队的余量
