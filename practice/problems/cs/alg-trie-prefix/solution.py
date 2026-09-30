class Trie:
    END = "#end"
    COUNT = "#count"

    def __init__(self):
        self.root = {}

    def insert(self, seq):
        node = self.root
        for ch in seq:
            node = node.setdefault(ch, {})
            node[self.COUNT] = node.get(self.COUNT, 0) + 1
        node[self.END] = node.get(self.END, 0) + 1

    def _walk(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node:
                return None
            node = node[ch]
        return node

    def contains(self, seq):
        node = self._walk(seq)
        return node is not None and node.get(self.END, 0) > 0

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def count_prefix(self, prefix):
        node = self._walk(prefix)
        return 0 if node is None else node.get(self.COUNT, 0)

    def longest_match(self, seq):
        node, matched = self.root, 0
        for ch in seq:
            if ch not in node:
                break
            node = node[ch]
            matched += 1
        return matched
