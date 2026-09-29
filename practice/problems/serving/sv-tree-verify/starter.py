import numpy as np


def tree_mask(tree):
    n = len(tree)
    return np.tril(np.ones((n, n), dtype=bool))     # 当成一条链处理：错误


def tree_positions(tree, base):
    return [base + i + 1 for i in range(len(tree))]


def accept(tree, root_next, node_next):
    pass
