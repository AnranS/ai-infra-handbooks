"""一个 InferenceService Operator 的 reconcile 逻辑（与 controller-runtime 的 Reconcile 同构）。

真实 Operator 里这段函数的输入是 API server 的对象，输出是对子对象的增删改；
这里把集群抽象成字典，方便把"该建什么、该改什么、状态怎么写回"这套逻辑单独跑通。
"""


def desired_children(spec):
    """从自定义资源的 spec 推导出应该存在的子对象"""
    name, replicas = spec["name"], spec["replicas"]
    return {
        f"deploy/{name}": {"replicas": replicas, "image": spec["image"],
                           "gpus": spec["gpusPerReplica"]},
        f"svc/{name}": {"selector": name, "port": 8000},
        f"hpa/{name}": {"min": max(1, replicas // 2), "max": replicas * 3,
                        "metric": "num_requests_waiting"},
    }


def reconcile(spec, actual):
    """比较期望与实际，返回要执行的动作列表和新的 status（幂等：只看当前状态）"""
    want = desired_children(spec)
    actions = []
    for key, cfg in want.items():
        if key not in actual:
            actions.append(("create", key, cfg))
        elif any(actual[key].get(k) != v for k, v in cfg.items()):
            actions.append(("update", key, cfg))     # 只比较自己管的字段，别碰子控制器写的状态
    for key in actual:                                   # 多余的子对象要删掉（比如副本数改成 0）
        if key not in want:
            actions.append(("delete", key, None))
    ready = actual.get(f"deploy/{spec['name']}", {}).get("ready", 0)
    phase = "Ready" if ready >= spec["replicas"] and not actions else \
            "Progressing" if actions or ready < spec["replicas"] else "Unknown"
    return actions, {"readyReplicas": ready, "phase": phase}


def apply(actions, actual):
    """把动作施加到"集群"上，模拟子控制器随后把 Pod 拉起来"""
    for op, key, cfg in actions:
        if op == "delete":
            actual.pop(key, None)
        else:
            actual[key] = dict(cfg)
    for key, obj in actual.items():
        if key.startswith("deploy/"):
            obj["ready"] = obj["replicas"]               # 假设 Pod 都能就绪
    return actual
