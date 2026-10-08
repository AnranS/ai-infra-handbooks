# 排队论：利用率越接近 1，排队时间涨得越快。M/M/c 的等待概率用 Erlang C 公式
from math import factorial


def erlang_c(c, rho):
    """c 台服务器、利用率 rho 时，一个请求到达就要排队的概率"""
    a = c * rho                                        # 到达率 ÷ 单台服务率
    top = a ** c / factorial(c) / (1 - rho)
    bottom = sum(a ** k / factorial(k) for k in range(c)) + top
    return top / bottom


def wait_ms(c, rho, service_ms):
    """平均排队时间 = 排队概率 × 服务时间 / (c × (1 - 利用率))"""
    return erlang_c(c, rho) * service_ms / (c * (1 - rho))


print("平均服务时间 10 ms 时，不同利用率下的平均排队时间：")
print("利用率   1 台      4 台      16 台")
for rho in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    row = "".join(f"{wait_ms(c, rho, 10):8.1f} ms" for c in (1, 4, 16))
    print(f"{rho:5.0%} {row}")
print()
print("同样 90% 的利用率，机器越多排队越短（小池子更怕长尾）：")
for c in (1, 2, 4, 8, 16, 32):
    print(f"  {c:2d} 台：排队 {wait_ms(c, 0.9, 10):6.2f} ms，排队概率 {erlang_c(c, 0.9):4.0%}")
print()
print("要把排队压到 5 ms 以内，利用率最高能到多少：")
for c in (1, 4, 16, 64):
    hi = max((r for r in (i / 1000 for i in range(1, 1000)) if wait_ms(c, r, 10) <= 5), default=0)
    print(f"  {c:2d} 台：{hi:.1%}")
