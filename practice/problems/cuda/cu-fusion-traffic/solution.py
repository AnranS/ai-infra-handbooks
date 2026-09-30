FUSIBLE = {"pointwise", "reduce"}


def _index(graph):
    return {name: (kind, inputs, nbytes) for name, kind, inputs, nbytes in graph}


def _users(graph):
    users = {name: [] for name, *_ in graph}
    for name, _, inputs, _ in graph:
        for i in inputs:
            users[i].append(name)
    return users


def eager_bytes(graph):
    idx = _index(graph)
    return sum(sum(idx[i][2] for i in inputs) + nbytes for name, kind, inputs, nbytes in graph if kind != "input")


def fused_bytes(graph, group):
    idx, users = _index(graph), _users(graph)
    group = set(group)
    ext = {i for n in group for i in idx[n][1] if i not in group}
    outs = [n for n in group if not users[n] or any(u not in group for u in users[n])]
    return sum(idx[i][2] for i in ext) + sum(idx[o][2] for o in outs)


def fusion_groups(graph):
    order = {name: k for k, (name, *_) in enumerate(graph)}
    idx = _index(graph)
    parent = {n: n for n, kind, *_ in graph if kind in FUSIBLE}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for name, kind, inputs, _ in graph:
        if kind not in FUSIBLE:
            continue
        for i in inputs:
            if i in parent:
                parent[find(i)] = find(name)
    groups = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    out = [sorted(g, key=order.get) for g in groups.values()]
    return sorted(out, key=lambda g: order[g[0]])


def plan_bytes(graph):
    idx = _index(graph)
    total = sum(fused_bytes(graph, g) for g in fusion_groups(graph))
    total += sum(sum(idx[i][2] for i in inputs) + nbytes for name, kind, inputs, nbytes in graph if kind == "matmul")
    return total
