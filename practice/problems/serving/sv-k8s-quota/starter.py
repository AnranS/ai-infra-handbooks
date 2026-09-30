def admit(quota, used, request):
    for key in quota:
        if used.get(key, 0) + request.get(key, 0) > quota[key]:   # 缺 requests 时当成 0，直接放行
            return False, "超配额"
    return True, ""


def preempt(pods, need_gpu, my_priority):
    victims = sorted(pods, key=lambda p: p["priority"])            # 没有排除同级和更高优先级的
    freed, chosen = 0, []
    for pod in victims:
        chosen.append(pod["name"])
        freed += pod["gpu"]
        if freed >= need_gpu:
            return chosen
    return chosen                                                  # 凑不够也返回


def gpu_hours(pods, hours):
    return {p["priority"]: p["gpu"] * hours for p in pods}         # 同优先级的互相覆盖
