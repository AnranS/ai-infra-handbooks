"""一个迷你 Kubernetes 调度器：过滤（predicate）+ 打分（score）+ 绑定，外加 gang 调度。

和真实调度器的对应：filter 对应 NodeResourcesFit、NodeAffinity、TaintToleration 等插件；
score 对应 NodeResourcesFit（LeastAllocated / MostAllocated）、ImageLocality、PodTopologySpread 等。
"""


class Node:
    def __init__(self, name, cpu, mem_gb, gpus, labels=None, taints=()):
        self.name, self.cpu, self.mem, self.gpus = name, cpu, mem_gb, gpus
        self.labels = labels or {}
        self.taints = set(taints)
        self.used_cpu = self.used_mem = self.used_gpu = 0

    def free(self):
        return self.cpu - self.used_cpu, self.mem - self.used_mem, self.gpus - self.used_gpu

    def place(self, pod):
        self.used_cpu += pod["cpu"]
        self.used_mem += pod["mem"]
        self.used_gpu += pod["gpu"]

    def release(self, pod):
        self.used_cpu -= pod["cpu"]
        self.used_mem -= pod["mem"]
        self.used_gpu -= pod["gpu"]


def filters(node, pod):
    """预选：任何一条不满足就淘汰这个节点，返回 (是否可行, 原因)"""
    cpu, mem, gpu = node.free()
    if pod["cpu"] > cpu or pod["mem"] > mem or pod["gpu"] > gpu:
        return False, "资源不足"
    for key, value in pod.get("selector", {}).items():          # nodeSelector / nodeAffinity
        if node.labels.get(key) != value:
            return False, f"标签不匹配 {key}={value}"
    if node.taints - set(pod.get("tolerations", ())):            # 污点与容忍
        return False, "有未被容忍的污点"
    return True, ""


def score_least_allocated(node, pod):
    """默认策略：剩余比例越高分越高（把负载摊开）"""
    cpu, mem, gpu = node.free()
    parts = [(cpu - pod["cpu"]) / node.cpu, (mem - pod["mem"]) / node.mem]
    if node.gpus:
        parts.append((gpu - pod["gpu"]) / node.gpus)
    return sum(parts) / len(parts) * 100


def score_most_allocated(node, pod):
    """装箱策略：剩余比例越低分越高（把碎片攒到一起，利于腾出整机）"""
    return 100 - score_least_allocated(node, pod)


def schedule(nodes, pod, score=score_least_allocated):
    """返回 (选中的节点, 各节点的分数或淘汰原因)"""
    feasible, report = [], {}
    for node in nodes:
        ok, why = filters(node, pod)
        if ok:
            feasible.append(node)
            report[node.name] = round(score(node, pod), 1)
        else:
            report[node.name] = why
    if not feasible:
        return None, report
    best = max(feasible, key=lambda n: (report[n.name], n.name))
    best.place(pod)
    return best, report


def schedule_gang(nodes, pods, score=score_least_allocated):
    """gang 调度：要么全部放下，要么一个都不放（避免多机多卡任务占着资源互相等待）"""
    placed = []
    for pod in pods:
        node, _ = schedule(nodes, pod, score)
        if node is None:
            for n, p in placed:                                  # 回滚已经放下的
                n.release(p)
            return None
        placed.append((node, pod))
    return [(n.name, p["name"]) for n, p in placed]


def fragmentation(nodes, gpus_per_pod):
    """碎片：加起来还剩很多卡，但没有一个节点能放下一个完整的任务"""
    free_total = sum(n.free()[2] for n in nodes)
    placeable = sum(n.free()[2] // gpus_per_pod for n in nodes)
    return free_total, placeable * gpus_per_pod
