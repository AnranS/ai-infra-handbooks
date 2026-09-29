import math


def rollout(n, load, cold_s, drain_s, max_unavailable, max_surge):
    rounds = n // max_unavailable                     # 没算额外拉起的实例，轮数也没向上取整
    return {"rounds": rounds, "minutes": rounds * cold_s / 60, "min_capacity": 1.0,
            "extra_gpu": 0.0, "overloaded": False}


def max_safe_unavailable(n, load):
    pass


def fastest_plan(n, load, cold_s, drain_s, extra_budget):
    pass
