"""radix.py —— SGLang 式的基数树前缀缓存（token 粒度）。

树的每条边保存一段 token 序列（key）和这些 token 的 KV 槽位（value）。从根到某个节点的路径拼起来，
就是一个被缓存过的前缀。匹配时沿树向下走，必要时把一条边劈成两段；淘汰时按最近访问时间从叶子开始删除，
正在被请求使用的节点通过 lock_ref 保护起来，不会被淘汰。
"""

import heapq
import itertools


class Node:
    def __init__(self, key=(), value=(), parent=None):
        self.children: dict[int, Node] = {}   # 子节点，以边上第一个 token 为键
        self.key, self.value = list(key), list(value)
        self.parent = parent
        self.lock_ref = 0
        self.last_access = 0


def _common_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class RadixCache:
    def __init__(self):
        self.root = Node()
        self.root.lock_ref = 1                # 根永远不被淘汰
        self._clock = itertools.count(1)
        self.evictable_size = 0               # 未被锁住、可以淘汰的 token 数
        self.protected_size = 0               # 被正在运行的请求锁住的 token 数

    def match_prefix(self, key: list[int]) -> tuple[list[int], Node]:
        """返回最长匹配前缀的 KV 槽位，以及匹配到的最后一个节点（用于加锁）。"""
        node, value = self.root, []
        now = next(self._clock)
        while key:
            child = node.children.get(key[0])
            if child is None:
                break
            n = _common_len(child.key, key)
            if n < len(child.key):
                child = self._split(child, n)   # 只匹配了半条边：把边劈开，让匹配终点落在节点上
            child.last_access = now
            value += child.value
            node, key = child, key[n:]
        return value, node

    def insert(self, key: list[int], value: list[int]) -> int:
        """插入一条序列，返回其中已经在树里的前缀长度（这部分的 value 是重复的，调用方应释放）。"""
        node, matched = self.root, 0
        now = next(self._clock)
        while key:
            child = node.children.get(key[0])
            if child is None:
                new = Node(key, value, node)
                new.last_access = now
                node.children[key[0]] = new
                self.evictable_size += len(key)
                return matched
            n = _common_len(child.key, key)
            if n < len(child.key):
                child = self._split(child, n)
            child.last_access = now
            matched += n
            node, key, value = child, key[n:], value[n:]
        return matched

    def _split(self, child: Node, n: int) -> Node:
        """把 child 的边从第 n 个 token 处劈开，返回新的中间节点。"""
        mid = Node(child.key[:n], child.value[:n], child.parent)
        mid.lock_ref, mid.last_access = child.lock_ref, child.last_access
        child.parent.children[mid.key[0]] = mid
        child.key, child.value, child.parent = child.key[n:], child.value[n:], mid
        mid.children[child.key[0]] = child
        return mid

    def inc_lock_ref(self, node: Node) -> None:
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable_size -= len(node.key)
                self.protected_size += len(node.key)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node: Node) -> None:
        while node is not self.root:
            node.lock_ref -= 1
            if node.lock_ref == 0:
                self.evictable_size += len(node.key)
                self.protected_size -= len(node.key)
            node = node.parent

    def evict(self, num_tokens: int) -> list[int]:
        """按 LRU 从叶子开始淘汰至少 num_tokens 个 token，返回释放出来的 KV 槽位。"""
        leaves = [(n.last_access, id(n), n) for n in self._nodes() if not n.children and n.lock_ref == 0]
        heapq.heapify(leaves)
        freed = []
        while leaves and len(freed) < num_tokens:
            _, _, node = heapq.heappop(leaves)
            freed += node.value
            parent = node.parent
            del parent.children[node.key[0]]
            self.evictable_size -= len(node.key)
            if parent is not self.root and not parent.children and parent.lock_ref == 0:
                heapq.heappush(leaves, (parent.last_access, id(parent), parent))
        return freed

    def _nodes(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root:
                yield n
            stack.extend(n.children.values())

    def total_size(self) -> int:
        return sum(len(n.key) for n in self._nodes())
