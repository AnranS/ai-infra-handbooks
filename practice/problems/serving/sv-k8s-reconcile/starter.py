def reconcile(state, max_surge=1, max_unavailable=1):
    replicas, image = state["spec"]["replicas"], state["spec"]["image"]
    total = len(state["pods"])
    available = sum(1 for p in state["pods"].values() if p["ready"])   # 没看镜像是不是最新的
    if total < replicas:
        return "create", image
    if total > replicas:
        return "delete", sorted(state["pods"])[-1]
    return None                                     # 镜像变了也不换：永远不会滚动更新


def run(state, max_steps=100, **kw):
    actions, seq = [], 0
    for _ in range(max_steps):
        action = reconcile(state, **kw)             # 忘了把 Pod 置为 ready
        if action is None:
            return actions
        actions.append(action)
        if action[0] == "create":
            seq += 1
            state["pods"][f"pod-{seq}"] = {"image": action[1], "ready": False}
        else:
            state["pods"].pop(action[1], None)
    return actions
