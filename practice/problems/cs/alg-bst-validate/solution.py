class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def is_bst(root):
    stack = [(root, float("-inf"), float("inf"))]
    while stack:
        node, low, high = stack.pop()
        if node is None:
            continue
        if not (low < node.val < high):        # 严格：等于也不行
            return False
        stack.append((node.left, low, node.val))
        stack.append((node.right, node.val, high))
    return True


def kth_smallest(root, k):
    if k <= 0:
        return None
    stack, node, seen = [], root, 0
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        seen += 1
        if seen == k:
            return node.val                    # 数到了就停，不用遍历完
        node = node.right
    return None
