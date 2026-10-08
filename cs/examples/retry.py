# 后端短暂变慢时，重试策略会把负载放大多少
from collections import defaultdict

RPS, CAPACITY, SECONDS, OUTAGE = 1000, 1200, 60, range(10, 20)
MAX_ATTEMPTS, BUDGET = 3, 0.1              # 最多重试 2 次；重试预算：每秒重试量不超过正常流量的 10%


def simulate(policy):
    """返回（峰值负载, 总请求数, 故障期间成功的请求数, 恢复到正常负载用了几秒）"""
    queued = defaultdict(lambda: defaultdict(int))     # 秒 -> {第几次尝试: 请求数}
    peak = total = served_in_outage = 0
    back_to_normal = None
    for sec in range(SECONDS):
        arrivals = dict(queued.pop(sec, {}))
        retries = sum(arrivals.values())
        if policy.endswith("重试预算"):                # 超出预算的重试直接放弃（快速失败）
            allowed = int(RPS * BUDGET)
            for attempt in sorted(arrivals, reverse=True):
                take = min(arrivals[attempt], allowed)
                arrivals[attempt], allowed = take, allowed - take
            retries = sum(arrivals.values())
        arrivals[0] = arrivals.get(0, 0) + RPS         # 这一秒新到的请求
        load = sum(arrivals.values())
        cap = CAPACITY if sec not in OUTAGE else CAPACITY // 10
        peak, total = max(peak, load), total + load
        if sec in OUTAGE:
            served_in_outage += min(load, cap)
        elif sec > OUTAGE[-1] and back_to_normal is None and load <= RPS * 1.05:
            back_to_normal = sec - OUTAGE[-1]
        if load <= cap or policy == "不重试":
            continue
        share = (load - cap) / load                    # 按比例分摊失败
        for attempt, n in list(arrivals.items()):
            failed = int(n * share)
            if not failed or attempt + 1 >= MAX_ATTEMPTS:
                continue
            if policy == "立即重试":
                queued[sec + 1][attempt + 1] += failed
            else:                                      # 指数退避 + 抖动：等 2^(k+1) 秒，并摊平到这段窗口里
                window = 2 ** (attempt + 1)
                for d in range(window):
                    queued[sec + window + d][attempt + 1] += failed // window
    return peak, total, served_in_outage, back_to_normal


print(f"正常 {RPS} 请求/秒、容量 {CAPACITY}，第 {OUTAGE[0]}～{OUTAGE[-1] + 1} 秒容量掉到 {CAPACITY // 10}：")
print("策略                 峰值负载  打到后端的总量  故障期间成功  恢复用时")
for policy in ["不重试", "立即重试", "指数退避 + 抖动", "退避 + 抖动 + 重试预算"]:
    peak, total, served, back = simulate(policy)
    print(f"{policy:20s} {peak:6d}/s {total:11d} {served:12d} {back if back is not None else '-':>7} 秒")
