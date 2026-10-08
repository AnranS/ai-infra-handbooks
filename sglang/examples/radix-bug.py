"""复现 #7 修掉的匹配 bug：树里有 "Hello_L.A.!" → "world"，查 "Hello_world"。"""
from collections import defaultdict


class Node:
    def __init__(self, value=""):
        self.children, self.value = {}, value


def match(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def insert(node, key):
    for ck, child in node.children.items():
        n = match(ck, key)
        if n == len(ck):
            return insert(child, key[n:]) if n < len(key) else None
        if n:
            mid = Node(ck[:n]); mid.children[ck[n:]] = child; child.value = ck[n:]
            del node.children[ck]; node.children[ck[:n]] = mid
            return insert(mid, key[n:])
    if key:
        node.children[key] = Node(key)


def match_prefix(node, key, buggy):
    out = []
    for ck, child in node.children.items():
        n = match(ck, key)
        if n:
            if (n == len(key) and n != len(ck)) if buggy else (n < len(ck)):
                out.append(ck[:n])                      # 分裂（这里只取前半段，省略真正的分裂）
            else:
                out.append(ck[:n] if buggy else ck)
                out += match_prefix(child, key[n:], buggy)
            break
    return out


root = Node()
insert(root, "Hello_L.A.!")
insert(root, "Hello_L.A.!world")
for buggy in (True, False):
    hit = "".join(match_prefix(root, "Hello_world", buggy))
    print(f"{'修复前' if buggy else '修复后'}：命中 {hit!r}（{len(hit)} 个 token）")
