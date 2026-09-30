from checker import check
from solution import filter_nodes, schedule, score


def node(name, cpu, mem, gpu, labels=None, taints=()):
    return {"name": name, "cpu": cpu, "mem": mem, "gpu": gpu, "labels": labels or {},
            "taints": list(taints), "used": {"cpu": 0, "mem": 0, "gpu": 0}}


def cluster():
    return [node("gpu-a", 96, 512, 8, {"gpu": "h100"}),
            node("gpu-b", 96, 512, 8, {"gpu": "h100"}),
            node("gpu-c", 64, 256, 4, {"gpu": "a100"}, taints=["maintenance"]),
            node("cpu-d", 64, 256, 0)]


def gpu_pod(gpu=1, **kw):
    p = {"cpu": 8, "mem": 64, "gpu": gpu, "selector": {"gpu": "h100"}, "tolerations": []}
    p.update(kw)
    return p


def test_example():
    nodes = cluster()
    check(schedule(nodes, gpu_pod()), "gpu-a", "分数相同时取名字小的")
    check(nodes[0]["used"], {"cpu": 8, "mem": 64, "gpu": 1}, "资源被记账")


def test_filter_reasons():
    nodes = cluster()
    feasible, reasons = filter_nodes(nodes, gpu_pod())
    check(sorted(n["name"] for n in feasible), ["gpu-a", "gpu-b"], "只有两个节点可行")
    check(reasons["gpu-c"], "标签不匹配", "a100 节点标签不对")
    check(reasons["cpu-d"], "资源不足", "没有 GPU")


def test_taint():
    nodes = cluster()
    pod = gpu_pod(selector={"gpu": "a100"})
    feasible, reasons = filter_nodes(nodes, pod)
    check(feasible, [], "污点没被容忍")
    check(reasons["gpu-c"], "污点未容忍", "原因正确")
    pod["tolerations"] = ["maintenance"]
    check(schedule(nodes, pod), "gpu-c", "容忍之后可以调度")


def test_no_feasible_node():
    nodes = cluster()
    check(schedule(nodes, gpu_pod(gpu=16)), None, "没有节点放得下")


def test_spread_vs_binpack():
    for strategy, want in (("spread", {"gpu-a": 3, "gpu-b": 3}), ("binpack", {"gpu-a": 6, "gpu-b": 0})):
        nodes = cluster()
        for _ in range(6):
            schedule(nodes, gpu_pod(), strategy)
        got = {n["name"]: n["used"]["gpu"] for n in nodes if n["name"].startswith("gpu-") and n["gpu"] == 8}
        check(got, want, f"{strategy} 策略的分布")


def test_score_accounts_for_pod():
    n = node("x", 10, 100, 4)
    s_empty = score(n, {"cpu": 0, "mem": 0, "gpu": 0})
    s_big = score(n, {"cpu": 5, "mem": 50, "gpu": 2})
    check(s_empty > s_big, True, "打分要算上这个 Pod 占用之后的剩余")


def test_cpu_only_node():
    nodes = [node("cpu-only", 16, 64, 0)]
    check(schedule(nodes, {"cpu": 4, "mem": 8, "gpu": 0}), "cpu-only", "没有 GPU 的节点也能调度")
