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
    while out and out[-1] == "#":              # 裁掉末尾多余的占位符
        out.pop()
    return ",".join(out)


def deserialize(data):
    if not data:
        return None
    parts = data.split(",")
    root = T(int(parts[0]))
    q = deque([root])
    i = 1
    while q and i < len(parts):
        node = q.popleft()
        if i < len(parts) and parts[i] != "#":
            node.left = T(int(parts[i]))
            q.append(node.left)
        i += 1
        if i < len(parts) and parts[i] != "#":
            node.right = T(int(parts[i]))
            q.append(node.right)
        i += 1
    return root
