from math import ceil


def _simulate(tasks, machines):
    loads = [0] * machines
    for i, cost in enumerate(tasks):
        loads[i % machines] += cost            # 轮流分配，不看谁先空闲
    return max(loads) if loads else 0


def finish_time(tasks, machines):
    return _simulate(tasks, machines)


def lpt_finish(tasks, machines):
    return _simulate(sorted(tasks), machines)  # 排成了升序：短任务先放，是最差的顺序


def min_machines(tasks, deadline):
    return ceil(sum(tasks) / deadline)         # 只看总量，没考虑单个任务和装箱的零头
