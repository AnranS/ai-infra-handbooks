def place_group(nodes, members, gpus_each):
    placed = []
    for _ in range(members):
        candidates = [n for n in nodes if n["free"] >= gpus_each]
        if not candidates:
            return None                              # 没有回滚：前面的成员还占着卡
        best = max(candidates, key=lambda n: n["free"])   # 挑最空的：把整机打散
        best["free"] -= gpus_each
        placed.append((best["name"], gpus_each))
    return placed


def fragmentation(nodes, gpus_each):
    total = sum(n["free"] for n in nodes)
    return total, total                              # 以为空闲卡都能用上


def defrag_gain(nodes, gpus_each):
    return 0
