# LRU 缓存：哈希表定位 + 双向链表维护顺序。这里用 OrderedDict，再手写一遍对照
from collections import OrderedDict


class LRUEasy:
    def __init__(self, capacity):
        self.cap, self.d = capacity, OrderedDict()

    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)                # 变成最近使用
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)         # 弹出最久没用的


class LRU:
    """手写版：双向链表的每个节点是 [key, value, prev, next]，头是最久没用的"""

    def __init__(self, capacity):
        self.cap, self.map = capacity, {}
        self.head, self.tail = ["#head", None, None, None], ["#tail", None, None, None]
        self.head[3], self.tail[2] = self.tail, self.head

    def _unlink(self, node):
        node[2][3], node[3][2] = node[3], node[2]

    def _append(self, node):                   # 接到尾部（最近使用）
        node[2], node[3] = self.tail[2], self.tail
        self.tail[2][3] = node
        self.tail[2] = node

    def get(self, key):
        node = self.map.get(key)
        if node is None:
            return -1
        self._unlink(node)
        self._append(node)
        return node[1]

    def put(self, key, value):
        node = self.map.get(key)
        if node is not None:
            node[1] = value
            self._unlink(node)
            self._append(node)
            return
        node = [key, value, None, None]
        self.map[key] = node
        self._append(node)
        if len(self.map) > self.cap:
            oldest = self.head[3]
            self._unlink(oldest)
            del self.map[oldest[0]]


for cls in (LRUEasy, LRU):
    c = cls(2)
    c.put(1, 1)
    c.put(2, 2)
    got = [c.get(1)]                           # 访问 1，让 2 变成最久没用的
    c.put(3, 3)                                # 淘汰 2
    got += [c.get(2), c.get(3)]
    c.put(4, 4)                                # 淘汰 1
    got += [c.get(1), c.get(3), c.get(4)]
    print(f"{cls.__name__:8s} 依次得到 {got}（-1 表示已被淘汰）")
print()
print("两种写法都是 O(1)：哈希表负责定位，双向链表负责维护顺序。")
print("推理引擎的 KV 块池用的是同一套结构：块哈希 -> 块，空闲块按 LRU 排队等待复用。")
