# 前缀树：推理引擎的 Radix Cache 就是它的压缩版（把只有一个孩子的链压成一条边）
class Trie:
    def __init__(self):
        self.root = {}                         # 每个节点是一个字典，特殊键 "#" 表示这里是一个词的结尾

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
            node.setdefault("#count", 0)
            node["#count"] += 1                # 经过这个前缀的词有几个
        node["#"] = True

    def contains(self, word):
        node = self.walk(word)
        return node is not None and "#" in node

    def walk(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node:
                return None
            node = node[ch]
        return node

    def count_prefix(self, prefix):
        node = self.walk(prefix)
        return 0 if node is None else node.get("#count", 0)

    def longest_match(self, text):
        """text 与树里任何一条路径的最长公共前缀长度——这正是前缀缓存的匹配"""
        node, matched = self.root, 0
        for ch in text:
            if ch not in node:
                break
            node = node[ch]
            matched += 1
        return matched


t = Trie()
for w in ["cat", "car", "card", "care", "dog"]:
    t.insert(w)
print("包含 'car'：", t.contains("car"), "；包含 'ca'：", t.contains("ca"))
print("以 'ca' 开头的词有", t.count_prefix("ca"), "个；以 'car' 开头的有", t.count_prefix("car"), "个")
print("'cards' 与已有内容的最长公共前缀长度：", t.longest_match("cards"))
print("'dogs' ：", t.longest_match("dogs"), "；'xyz'：", t.longest_match("xyz"))
print()

# 换成 token 序列：多轮对话共享同一个系统提示词时，能复用多少前缀
cache = Trie()
system = [101, 102, 103, 104, 105]
cache.insert(tuple(system + [201, 202]))
for name, req in [("同一个系统提示词 + 新问题", system + [301]),
                  ("换了系统提示词", [999] + system),
                  ("完全相同的请求", system + [201, 202])]:
    matched = cache.longest_match(tuple(req))
    print(f"{name}：可复用前缀 {matched} 个 token，需要重算 {len(req) - matched} 个")
