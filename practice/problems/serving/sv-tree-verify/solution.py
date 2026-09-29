import numpy as np


def tree_mask(tree):
    n = len(tree)
    mask = np.zeros((n, n), dtype=bool)
    for i, (parent, _) in enumerate(tree):
        if parent >= 0:
            mask[i] = mask[parent]
        mask[i, i] = True
    return mask


def tree_positions(tree, base):
    depth = []
    for parent, _ in tree:
        depth.append(1 if parent < 0 else depth[parent] + 1)
    return [base + d for d in depth]


def accept(tree, root_next, node_next):
    children = {}
    for i, (parent, _) in enumerate(tree):
        children.setdefault(parent, []).append(i)
    cur, want = -1, root_next
    nodes, tokens = [], []
    while True:
        nxt = next((c for c in children.get(cur, []) if tree[c][1] == want), None)
        if nxt is None:
            break
        nodes.append(nxt)
        tokens.append(tree[nxt][1])
        cur, want = nxt, node_next[nxt]
    return nodes, tokens + [want]
