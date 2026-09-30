from collections import deque


class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def preorder(root):
    out, stack = [], [root] if root else []
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.right:
            stack.append(node.right)           # 先压右，弹出时左先
        if node.left:
            stack.append(node.left)
    return out


def inorder(root):
    out, stack, node = [], [], root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def level_order(root):
    out, q = [], deque([root] if root else [])
    while q:
        level = []
        for _ in range(len(q)):                # 先记下这一层有几个
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        out.append(level)
    return out
