def _fresh(state):
    image = state["spec"]["image"]
    return [n for n, p in state["pods"].items() if p["image"] == image]


def _stale(state):
    image = state["spec"]["image"]
    return sorted(n for n, p in state["pods"].items() if p["image"] != image)


def _available(state):
    image = state["spec"]["image"]
    return sum(1 for p in state["pods"].values() if p["ready"] and p["image"] == image)


def reconcile(state, max_surge=1, max_unavailable=1):
    replicas, image = state["spec"]["replicas"], state["spec"]["image"]
    total, stale = len(state["pods"]), _stale(state)
    if _available(state) < replicas - max_unavailable and total < replicas + max_surge:
        return "create", image
    if total > replicas + max_surge or (total >= replicas and stale):
        return "delete", stale[0] if stale else sorted(state["pods"])[-1]
    if total < replicas:
        return "create", image
    if total > replicas:
        return "delete", stale[0] if stale else sorted(state["pods"])[-1]
    return None


def run(state, max_steps=100, **kw):
    actions, seq = [], 0
    for _ in range(max_steps):
        for pod in state["pods"].values():          # 上一轮创建的 Pod 到这一轮就绪
            pod["ready"] = True
        action = reconcile(state, **kw)
        if action is None:
            return actions
        actions.append(action)
        if action[0] == "create":
            seq += 1
            state["pods"][f"pod-{seq}"] = {"image": action[1], "ready": False}
        else:
            state["pods"].pop(action[1], None)
    return actions
