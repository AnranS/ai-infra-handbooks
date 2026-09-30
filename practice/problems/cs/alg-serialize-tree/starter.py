from collections import deque


class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def serialize(root):
    if root is None:
        return ""
    out, q = [], deque([root])
    while q:
        node = q.popleft()
        if node is None:
            out.append("#")
            continue
        out.append(str(node.val))
        q.append(node.left)
        q.append(node.right)
    return ",".join(out)                       # 没有裁掉末尾的 #


def deserialize(data):
    parts = data.split(",")                    # 空串会得到 [""]
    root = T(int(parts[0]))
    q = deque([root])
    i = 1
    while q:
        node = q.popleft()
        if parts[i] != "#":                    # 没检查是否读完：会越界
            node.left = T(int(parts[i]))
            q.append(node.left)
        i += 1
        if parts[i] != "#":
            node.right = T(int(parts[i]))
            q.append(node.right)
        i += 1
    return root
