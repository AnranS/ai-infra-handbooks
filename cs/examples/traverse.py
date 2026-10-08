# 树的四种遍历，全部写成迭代（Python 的递归深度只有 1000，退化成链的树会崩）
from collections import deque


class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


root = T(1, T(2, T(4), T(5)), T(3, None, T(6)))


def preorder(node):
    out, stack = [], [node] if node else []
    while stack:
        cur = stack.pop()
        out.append(cur.val)
        if cur.right:
            stack.append(cur.right)            # 先压右，后压左：弹出时左先
        if cur.left:
            stack.append(cur.left)
    return out


def inorder(node):
    out, stack = [], []
    while node or stack:
        while node:                            # 一路向左
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def postorder(node):
    out, stack = [], [node] if node else []
    while stack:                               # 按"根右左"遍历再整体反转，就是"左右根"
        cur = stack.pop()
        out.append(cur.val)
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.right)
    return out[::-1]


def level_order(node):
    out, q = [], deque([node] if node else [])
    while q:
        level = []
        for _ in range(len(q)):                # 一次处理一整层
            cur = q.popleft()
            level.append(cur.val)
            q.extend(x for x in (cur.left, cur.right) if x)
        out.append(level)
    return out


print("       1")
print("      / \\")
print("     2   3")
print("    / \\   \\")
print("   4   5   6")
print("前序（根左右）：", preorder(root))
print("中序（左根右）：", inorder(root))
print("后序（左右根）：", postorder(root))
print("层序：", level_order(root))
print()
deep = T(0)
cur = deep
for i in range(1, 20000):                      # 退化成链的树：递归一定爆栈
    cur.right = T(i)
    cur = cur.right
print("两万层的退化树，迭代前序遍历的前几个：", preorder(deep)[:5], "…… 共", len(preorder(deep)), "个节点")
