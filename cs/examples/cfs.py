# 内核里 nice 值到权重的对照表（kernel/sched/core.c 的 sched_prio_to_weight）：nice 0 是 1024，每差一级约差 1.25 倍
WEIGHT = {-5: 3121, 0: 1024, 5: 335, 10: 110}
tasks = {"调度主循环": -5, "tokenizer": 0, "指标上报": 10}

vruntime = {name: 0.0 for name in tasks}
ran = {name: 0 for name in tasks}
for _ in range(1000):                                    # 模拟 1 秒，每次运行 1 ms
    name = min(vruntime, key=lambda n: (vruntime[n], n))  # 总是挑虚拟运行时间最小的任务
    ran[name] += 1
    vruntime[name] += 1024 / WEIGHT[tasks[name]]         # 权重越大，虚拟时间走得越慢，被挑中的次数越多

total = sum(WEIGHT[n] for n in tasks.values())
for name, nice in tasks.items():
    print(f"{name}（nice {nice:+d}）：运行了 {ran[name]} ms，按权重应得 {1000 * WEIGHT[nice] / total:.1f} ms")
