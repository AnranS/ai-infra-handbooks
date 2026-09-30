def place_group(nodes, members, gpus_each):
    placed = []
    for _ in range(members):
        candidates = [n for n in nodes if n["free"] >= gpus_each]
        if not candidates:
            for node in nodes:                       # 整组回滚
                node["free"] += sum(g for name, g in placed if name == node["name"])
            return None
        best = min(candidates, key=lambda n: (n["free"], n["name"]))
        best["free"] -= gpus_each
        placed.append((best["name"], gpus_each))
    return placed


def fragmentation(nodes, gpus_each):
    total = sum(n["free"] for n in nodes)
    usable = sum(n["free"] // gpus_each * gpus_each for n in nodes)
    return total, usable


def defrag_gain(nodes, gpus_each):
    total, usable = fragmentation(nodes, gpus_each)
    return total // gpus_each - usable // gpus_each
