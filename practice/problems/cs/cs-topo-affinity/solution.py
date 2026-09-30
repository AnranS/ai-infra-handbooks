from itertools import permutations

ORDER = ["PIX", "PXB", "PHB", "NODE", "SYS"]


def link_rank(link):
    link = (link or "").strip().upper()
    if link.startswith("NV"):
        try:
            n = int(link[2:])
        except ValueError:
            n = 1
        return -n                                  # NVLink 排在所有其他类型之前，链路多的更快
    return ORDER.index(link) + 1 if link in ORDER else len(ORDER) + 1


def pick_nics(topo, gpus, nics):
    best, best_cost = None, None
    for perm in permutations(nics, len(gpus)):
        cost = sum(link_rank(topo[g].get(n)) for g, n in zip(gpus, perm))
        if best_cost is None or cost < best_cost:
            best, best_cost = perm, cost
    return dict(zip(gpus, best))


def all_nvlink(topo, gpus):
    return all(str(topo[a].get(b, "")).upper().startswith("NV")
               for a in gpus for b in gpus if a != b)
