def filter_nodes(nodes, pod):
    feasible, reasons = [], {}
    for node in nodes:
        used = node["used"]
        if (pod["cpu"] > node["cpu"] - used["cpu"] or pod["mem"] > node["mem"] - used["mem"]
                or pod["gpu"] > node["gpu"] - used["gpu"]):
            reasons[node["name"]] = "资源不足"
            continue
        if any(node["labels"].get(k) != v for k, v in pod.get("selector", {}).items()):
            reasons[node["name"]] = "标签不匹配"
            continue
        if set(node.get("taints", [])) - set(pod.get("tolerations", [])):
            reasons[node["name"]] = "污点未容忍"
            continue
        feasible.append(node)
    return feasible, reasons


def score(node, pod, strategy="spread"):
    used = node["used"]
    parts = [(node["cpu"] - used["cpu"] - pod["cpu"]) / node["cpu"],
             (node["mem"] - used["mem"] - pod["mem"]) / node["mem"]]
    if node["gpu"]:
        parts.append((node["gpu"] - used["gpu"] - pod["gpu"]) / node["gpu"])
    spread = sum(parts) / len(parts) * 100
    return spread if strategy == "spread" else 100 - spread


def schedule(nodes, pod, strategy="spread"):
    feasible, _ = filter_nodes(nodes, pod)
    if not feasible:
        return None
    best = max(feasible, key=lambda n: (score(n, pod, strategy), [-ord(c) for c in n["name"]]))
    for key in ("cpu", "mem", "gpu"):
        best["used"][key] += pod[key]
    return best["name"]
