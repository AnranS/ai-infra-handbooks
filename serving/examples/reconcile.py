# 控制器模式：读"期望状态"和"实际状态"，算出差异，只做把两者拉近的那一步动作
from collections import deque


class Cluster:
    """极简的 API server：只存对象，不做任何决策"""

    def __init__(self):
        self.deployments = {}                  # 名字 -> {"replicas": 期望副本数, "image": 版本}
        self.pods = {}                          # 名字 -> {"owner": ..., "image": ..., "phase": ...}
        self.events = []
        self.seq = 0

    def create_pod(self, owner, image):
        self.seq += 1
        name = f"{owner}-{self.seq}"
        self.pods[name] = {"owner": owner, "image": image, "phase": "Pending"}
        self.events.append(f"create {name}")
        return name

    def delete_pod(self, name):
        self.pods.pop(name, None)
        self.events.append(f"delete {name}")

    def owned(self, owner):
        return {n: p for n, p in self.pods.items() if p["owner"] == owner}


def reconcile(cluster, name):
    """一次 reconcile：只看当前状态，不依赖"上次做了什么"（幂等）"""
    spec = cluster.deployments[name]
    pods = cluster.owned(name)
    stale = [n for n, p in pods.items() if p["image"] != spec["image"]]
    if len(pods) < spec["replicas"]:
        cluster.create_pod(name, spec["image"])            # 少了就补
    elif len(pods) > spec["replicas"]:
        cluster.delete_pod(sorted(pods)[-1])               # 多了就删
    elif stale:
        cluster.delete_pod(sorted(stale)[0])               # 数量够了但版本旧：换掉一个
    else:
        return False                                        # 已经收敛，不需要动作
    return True


def run(cluster, name, max_steps=50):
    """控制循环：反复 reconcile 直到不再产生动作"""
    steps = 0
    while steps < max_steps and reconcile(cluster, name):
        for pod in cluster.pods.values():                  # 模拟 kubelet：Pending 的 Pod 过一会变 Running
            if pod["phase"] == "Pending":
                pod["phase"] = "Running"
        steps += 1
    return steps
