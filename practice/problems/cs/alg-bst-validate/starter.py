class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def is_bst(root):
    if root is None:
        return True
    if root.left and root.left.val >= root.val:    # 只比较父子：子树里的值没检查
        return False
    if root.right and root.right.val <= root.val:
        return False
    return is_bst(root.left) and is_bst(root.right)


def kth_smallest(root, k):
    values = []

    def inorder(node):
        if node:
            inorder(node.left)
            values.append(node.val)
            inorder(node.right)

    inorder(root)                              # 深树会 RecursionError
    return values[k - 1] if 0 < k <= len(values) else None
