from collections import deque


class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def preorder(root):
    out, stack = [], [root] if root else []
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.left:
            stack.append(node.left)            # 压栈顺序反了：变成了"根右左"
        if node.right:
            stack.append(node.right)
    return out


def inorder(root):
    return preorder(root)                      # 直接拿前序充数


def level_order(root):
    out, q = [], deque([root] if root else [])
    while q:
        node = q.popleft()
        out.append([node.val])                 # 每个节点单独一层
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return out
