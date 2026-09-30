ORDER = ["PIX", "PXB", "PHB", "NODE", "SYS"]


def link_rank(link):
    link = (link or "").strip().upper()
    if link.startswith("NV"):
        return 0                                   # 所有 NVLink 一样快？链路数没用上
    return ORDER.index(link) + 1 if link in ORDER else len(ORDER) + 1


def pick_nics(topo, gpus, nics):
    taken, out = set(), {}
    for g in gpus:                                 # 贪心：每张 GPU 各自挑最好的，先来先得
        free = [n for n in nics if n not in taken]
        pick = min(free, key=lambda n: link_rank(topo[g].get(n)))
        out[g], _ = pick, taken.add(pick)
    return out


def all_nvlink(topo, gpus):
    return all(str(topo[a].get(b, "")).upper().startswith("NV")
               for a in gpus for b in gpus if a != b)
