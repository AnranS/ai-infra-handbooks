import heapq
import itertools


class Node:
    _ids = itertools.count()

    def __init__(self, key, value, timestamp):
        self.key, self.value = list(key), list(value)
        self.children = {}
        self.parent = None
        self.ref_count = 0
        self.timestamp = timestamp
        self.uid = next(Node._ids)


class Handle:
    def __init__(self, cached_len, node):
        self.cached_len, self.node = cached_len, node


class RadixCache:
    def __init__(self):
        self.clock = 0
        self.root = Node([], [], 0)
        self.root.ref_count = 1
        self.evictable_size = 0
        self.protected_size = 0

    @staticmethod
    def _set_parent(node, parent):
        node.parent = parent
        parent.children[node.key[0]] = node

    def _split(self, node, pos):
        front = Node(node.key[:pos], node.value[:pos], node.timestamp)
        front.ref_count = node.ref_count
        self._set_parent(front, node.parent)
        node.key, node.value = node.key[pos:], node.value[pos:]
        self._set_parent(node, front)
        return front

    def _walk(self, ids):
        self.clock += 1
        tic = self.clock
        node, pos = self.root, 0
        while pos < len(ids):
            child = node.children.get(ids[pos])
            if child is None:
                break
            m = 0
            while m < len(child.key) and pos + m < len(ids) and child.key[m] == ids[pos + m]:
                m += 1
            pos += m
            if m < len(child.key):
                child = self._split(child, m)
                child.timestamp = tic
                return child, pos, tic
            child.timestamp = tic
            node = child
        return node, pos, tic

    def match_prefix(self, ids):
        node, n, _ = self._walk(list(ids))
        return Handle(n, node), n

    def insert_prefix(self, ids, indices):
        ids, indices = list(ids), list(indices)
        node, n, tic = self._walk(ids)
        if n < len(ids):
            leaf = Node(ids[n:], indices[n:], tic)
            self._set_parent(leaf, node)
            self.evictable_size += len(leaf.key)
            node = leaf
        return n, Handle(len(ids), node)

    def lock(self, handle):
        node = handle.node
        while node is not self.root:
            if node.ref_count == 0:
                self.evictable_size -= len(node.key)
                self.protected_size += len(node.key)
            node.ref_count += 1
            node = node.parent

    def unlock(self, handle):
        node = handle.node
        while node is not self.root:
            node.ref_count -= 1
            if node.ref_count == 0:
                self.evictable_size += len(node.key)
                self.protected_size -= len(node.key)
            node = node.parent

    def evict(self, size):
        if size > self.evictable_size:
            raise ValueError(f"最多只能淘汰 {self.evictable_size} 个，要求 {size} 个")
        heap, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if not node.children:
                if node.ref_count == 0:
                    heap.append((node.timestamp, node.uid, node))
            else:
                stack.extend(node.children.values())
        heapq.heapify(heap)
        out, freed = [], 0
        while freed < size:
            _, _, node = heapq.heappop(heap)
            out += node.value
            freed += len(node.key)
            self.evictable_size -= len(node.key)
            parent = node.parent
            del parent.children[node.key[0]]
            if parent is not self.root and not parent.children and parent.ref_count == 0:
                heapq.heappush(heap, (parent.timestamp, parent.uid, parent))
        return out

    def matched_indices(self, handle):
        parts, node = [], handle.node
        while node is not self.root:
            parts.append(node.value)
            node = node.parent
        return [x for p in reversed(parts) for x in p]
