class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def lowest_ancestor(root, p, q):
    if root is None or root is p or root is q:
        return root
    left = lowest_ancestor(root.left, p, q)
    right = lowest_ancestor(root.right, p, q)
    if left and right:                         # 两边各找到一个：当前节点就是答案
        return root
    return left or right


def max_depth(root):
    depth, stack = 0, [(root, 1)] if root else []
    while stack:                               # 迭代写法，避免深树爆栈
        node, d = stack.pop()
        depth = max(depth, d)
        if node.left:
            stack.append((node.left, d + 1))
        if node.right:
            stack.append((node.right, d + 1))
    return depth
