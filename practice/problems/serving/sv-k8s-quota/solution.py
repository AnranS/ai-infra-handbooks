def admit(quota, used, request):
    for key in request:
        if key not in quota:
            return False, "未知资源"
    for key in quota:
        if key not in request:
            return False, "缺少 requests"
        if used.get(key, 0) + request[key] > quota[key]:
            return False, "超配额"
    return True, ""


def preempt(pods, need_gpu, my_priority):
    victims = sorted((p for p in pods if p["priority"] < my_priority),
                     key=lambda p: (p["priority"], -p["gpu"], p["name"]))
    freed, chosen = 0, []
    for pod in victims:
        if freed >= need_gpu:
            break
        chosen.append(pod["name"])
        freed += pod["gpu"]
    return chosen if freed >= need_gpu else None


def gpu_hours(pods, hours):
    out = {}
    for pod in pods:
        out[pod["priority"]] = out.get(pod["priority"], 0) + pod["gpu"] * hours
    return out
