def filter_nodes(nodes, pod):
    feasible, reasons = [], {}
    for node in nodes:
        if pod["gpu"] > node["gpu"] - node["used"]["gpu"]:     # 只看了 GPU
            reasons[node["name"]] = "资源不足"
            continue
        feasible.append(node)                                  # 标签和污点都没检查
    return feasible, reasons


def score(node, pod, strategy="spread"):
    used = node["used"]
    free = (node["gpu"] - used["gpu"]) / max(1, node["gpu"])   # 只按 GPU 打分，也没算上这个 Pod
    return free * 100 if strategy == "spread" else 100 - free * 100


def schedule(nodes, pod, strategy="spread"):
    feasible, _ = filter_nodes(nodes, pod)
    if not feasible:
        return None
    best = max(feasible, key=lambda n: score(n, pod, strategy))
    for key in ("cpu", "mem", "gpu"):
        best["used"][key] += pod[key]
    return best["name"]
