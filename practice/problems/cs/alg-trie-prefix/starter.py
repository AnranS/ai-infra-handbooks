class Trie:
    def __init__(self):
        self.words = []                        # 用列表存所有插入过的序列

    def insert(self, seq):
        self.words.append(seq)

    def contains(self, seq):
        return seq in self.words

    def starts_with(self, prefix):
        return any(w[:len(prefix)] == prefix for w in self.words)   # O(词数 × 前缀长度)

    def count_prefix(self, prefix):
        return sum(1 for w in self.words if w[:len(prefix)] == prefix)

    def longest_match(self, seq):
        best = 0
        for w in self.words:
            i = 0
            while i < min(len(w), len(seq)) and w[i] == seq[i]:
                i += 1
            best = max(best, i)
        return best
