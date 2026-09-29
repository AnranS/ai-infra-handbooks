import random

import numpy as np

from checker import check

TREE = [(-1, 10), (-1, 11), (0, 20), (0, 21), (2, 30)]


def test_example():
    check(__import__("solution").accept(TREE, 10, [21, 99, 30, 7, 8]), ([0, 3], [10, 21, 7]), "accept")


def test_mask_and_positions():
    from solution import tree_mask, tree_positions

    m = tree_mask(TREE)
    want = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=bool)
    check(m, want, "树形注意力掩码")
    check(tree_positions(TREE, 100), [101, 101, 102, 102, 103], "位置")


def test_accept_cases():
    from solution import accept

    check(accept(TREE, 55, [0] * 5), ([], [55]), "树根的预测不在孩子里：只输出一个 token")
    check(accept(TREE, 10, [20, 0, 30, 0, 42]), ([0, 2, 4], [10, 20, 30, 42]), "走到叶子")
    dup = [(-1, 5), (-1, 5), (1, 6)]
    check(accept(dup, 5, [6, 6, 9]), ([0], [5, 6]), "兄弟节点 token 相同时取编号小的")


def test_random_trees():
    from solution import tree_mask

    rng = random.Random(0)
    for trial in range(50):
        n = rng.randint(1, 30)
        tree = [(rng.randint(-1, i - 1), rng.randint(0, 5)) for i in range(n)]
        m = tree_mask(tree)
        for i in range(n):
            anc, j = {i}, tree[i][0]
            while j >= 0:
                anc.add(j)
                j = tree[j][0]
            check(set(np.nonzero(m[i])[0].tolist()), anc, f"随机树 {trial} 第 {i} 行")
