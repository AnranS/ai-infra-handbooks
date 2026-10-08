"""GPU 分配：拓扑感知的选卡、MIG 切分与时间片共享的账。

一台 8 卡机的 NVLink 拓扑（简化成"两组四卡"的常见形态）：组内直连，跨组要绕。
真实拓扑用 nvidia-smi topo -m 看，Kubernetes 侧由 device plugin 的拓扑感知分配或 DRA 处理。
"""
from itertools import combinations

GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7)]            # 同组内 NVLink 直连


def link_cost(gpus):
    """一组卡之间的通信代价：同组算 1，跨组算 4（数值只表示量级）"""
    cost = 0
    for a, b in combinations(sorted(gpus), 2):
        same = any(a in g and b in g for g in GROUPS)
        cost += 1 if same else 4
    return cost


def pick_gpus(free, n, topology_aware=True):
    """从空闲卡里挑 n 张：拓扑感知时选通信代价最小的一组，否则按编号先到先得"""
    free = sorted(free)
    if len(free) < n:
        return None
    if not topology_aware:
        return free[:n]
    return min(combinations(free, n), key=lambda c: (link_cost(c), c))


MIG_PROFILES = {                                  # H100 80GB 的部分 MIG 档位：(份数, 显存 GB, SM 占比)
    "1g.10gb": (1, 10, 1 / 7),
    "2g.20gb": (2, 20, 2 / 7),
    "3g.40gb": (3, 40, 3 / 7),
    "7g.80gb": (7, 80, 1.0),
}


def mig_layout(profile, card_mem_gb=80):
    """一张卡切成这个档位能得到几个实例、每个实例多少显存和多少算力比例"""
    slices, mem, share = MIG_PROFILES[profile]
    return 7 // slices, mem, share


def decode_ms(weight_gb, bandwidth_gbs, share=1.0):
    """batch=1 的 decode 下限：读一遍权重。MIG 按份额切带宽，时间片共享则是排队"""
    return weight_gb / (bandwidth_gbs * share) * 1000


def timeslice_latency(base_ms, tenants):
    """时间片共享：n 个租户轮流用整张卡，单个请求的等待时间线性变差（显存还互相挤）"""
    return base_ms * tenants
