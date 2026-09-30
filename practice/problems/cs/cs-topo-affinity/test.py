from checker import check
from solution import all_nvlink, link_rank, pick_nics


def test_example():
    topo = {"GPU0": {"NIC0": "PIX", "NIC1": "SYS"}, "GPU1": {"NIC0": "PIX", "NIC1": "NODE"}}
    check(pick_nics(topo, ["GPU0", "GPU1"], ["NIC0", "NIC1"]), {"GPU0": "NIC0", "GPU1": "NIC1"},
          "GPU0 只有 NIC0 是近的")


def test_rank_order():
    ranks = [link_rank(x) for x in ["NV18", "NV2", "PIX", "PXB", "PHB", "NODE", "SYS", "X"]]
    check(ranks == sorted(ranks), True, "从快到慢")
    check(link_rank("NV18") < link_rank("NV2"), True, "链路越多越快")
    check(link_rank("nv4") < link_rank("PIX"), True, "小写也要认；NVLink 快过 PIX")


def test_greedy_would_fail():
    # GPU0 抢先占走 NIC0 的话，GPU1 只能走 SYS；最优是 GPU0 用 NIC1
    topo = {"GPU0": {"NIC0": "PIX", "NIC1": "PXB"}, "GPU1": {"NIC0": "PIX", "NIC1": "SYS"}}
    check(pick_nics(topo, ["GPU0", "GPU1"], ["NIC0", "NIC1"]), {"GPU0": "NIC1", "GPU1": "NIC0"},
          "总代价最小，而不是先来先得")


def test_eight_gpus():
    topo = {}
    for i in range(8):                             # 0~3 挂在 0 号交换芯片下，4~7 挂在 1 号
        topo[f"GPU{i}"] = {f"NIC{j}": ("PIX" if i == j else ("NODE" if i // 4 == j // 4 else "SYS"))
                           for j in range(8)}
    got = pick_nics(topo, [f"GPU{i}" for i in range(8)], [f"NIC{j}" for j in range(8)])
    check(got, {f"GPU{i}": f"NIC{i}" for i in range(8)}, "每张卡配自己交换芯片下的网卡")


def test_all_nvlink():
    nv = {f"GPU{i}": {f"GPU{j}": ("X" if i == j else "NV18") for j in range(4)} for i in range(4)}
    check(all_nvlink(nv, ["GPU0", "GPU1", "GPU2", "GPU3"]), True, "8 卡机内全互联")
    nv["GPU0"]["GPU3"] = "SYS"
    check(all_nvlink(nv, ["GPU0", "GPU1", "GPU2", "GPU3"]), False, "有一对不是 NVLink")
    check(all_nvlink(nv, ["GPU0", "GPU1"]), True, "只看这两张卡")
