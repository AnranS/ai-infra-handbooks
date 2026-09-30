class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def lowest_ancestor(root, p, q):
    if root is None:
        return None
    if root.val == p.val or root.val == q.val:     # 比较值而不是节点：值重复时会错
        return root
    left = lowest_ancestor(root.left, p, q)
    right = lowest_ancestor(root.right, p, q)
    if left and right:
        return root
    return left or right


def max_depth(root):
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))   # 深树会 RecursionError
