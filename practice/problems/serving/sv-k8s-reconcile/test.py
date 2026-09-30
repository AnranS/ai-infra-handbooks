from checker import check
from solution import reconcile, run


def make(replicas, image, pods=None):
    return {"spec": {"replicas": replicas, "image": image},
            "pods": {n: dict(p) for n, p in (pods or {}).items()}}


def test_example():
    s = make(2, "v1")
    check(run(s), [("create", "v1"), ("create", "v1")], "从零扩到两个")
    check(len(s["pods"]), 2, "最终两个 Pod")


def test_idempotent():
    s = make(2, "v1", {"a": {"image": "v1", "ready": True}, "b": {"image": "v1", "ready": True}})
    check(reconcile(s), None, "已经收敛就不该有动作")
    check(run(s), [], "再跑一遍什么也不做")


def test_scale_down():
    s = make(1, "v1", {"a": {"image": "v1", "ready": True}, "b": {"image": "v1", "ready": True}})
    actions = run(s)
    check(len(actions), 1, "只需要删一个")
    check(actions[0][0], "delete", "动作是删除")
    check(len(s["pods"]), 1, "剩一个")


def test_rolling_update():
    pods = {f"p{i}": {"image": "v1", "ready": True} for i in range(3)}
    s = make(3, "v2", pods)
    actions = run(s)
    check(all(p["image"] == "v2" for p in s["pods"].values()), True, "全部换成新镜像")
    check(len(s["pods"]), 3, "副本数不变")
    check(any(a[0] == "create" for a in actions) and any(a[0] == "delete" for a in actions),
          True, "既创建也删除")


def test_surge_limit():
    pods = {f"p{i}": {"image": "v1", "ready": True} for i in range(3)}
    s = make(3, "v2", pods)
    seen_max, seq = 0, 0
    for _ in range(50):
        for pod in s["pods"].values():
            pod["ready"] = True
        action = reconcile(s, max_surge=1, max_unavailable=0)
        if action is None:
            break
        if action[0] == "create":
            seq += 1
            s["pods"][f"new-{seq}"] = {"image": "v2", "ready": False}
        else:
            s["pods"].pop(action[1])
        seen_max = max(seen_max, len(s["pods"]))
    check(seen_max <= 4, True, f"总数不超过 replicas + max_surge（实际峰值 {seen_max}）")
    check(all(p["image"] == "v2" for p in s["pods"].values()), True, "最终全部是新镜像")


def test_self_heal():
    s = make(3, "v1", {f"p{i}": {"image": "v1", "ready": True} for i in range(3)})
    s["pods"].pop("p0")                              # 有人手动删了一个
    actions = run(s)
    check(actions, [("create", "v1")], "自己补回来")
    check(len(s["pods"]), 3, "恢复到三个")


def test_scale_to_zero():
    s = make(0, "v1", {f"p{i}": {"image": "v1", "ready": True} for i in range(2)})
    run(s)
    check(len(s["pods"]), 0, "缩到零")
