---
title: 前缀树与最长前缀匹配
chapter: algo/tree-graph.md
difficulty: 中等
tags: [前缀树,字典树,前缀缓存]
---
实现一个前缀树（字典树）：

- `insert(seq)`：插入一个序列（字符串或 token 元组）；
- `contains(seq)`：这个序列是否被完整插入过；
- `starts_with(prefix)`：有没有以 `prefix` 开头的序列；
- `count_prefix(prefix)`：有多少个插入的序列以 `prefix` 开头（同一个序列插入两次算两个）；
- `longest_match(seq)`：`seq` 与树中任意路径的**最长公共前缀长度**——这就是前缀缓存的匹配逻辑。

```python
t = Trie()
for w in ["cat", "car", "card"]:
    t.insert(w)
t.contains("car")          # True
t.contains("ca")           # False
t.starts_with("ca")        # True
t.count_prefix("car")      # 2
t.longest_match("cards")   # 4
```

<!-- 题解 -->
每个节点是一个字典，键是下一个字符/token。`insert` 沿路创建节点并给每个节点的计数加一；`contains` 走到头之后检查结束标记；`count_prefix` 直接读节点上的计数。

`longest_match` 是推理引擎最关心的操作：新请求的 token 序列能和缓存里的内容匹配多长，就能跳过多少 prefill。注意它**不要求走到某个词的结尾**，匹配到哪算哪。

真实的 Radix Cache 在这个基础上做了两件事：把只有一个孩子的链**压缩成一条边**（省内存、少跳转），以及给每个节点挂 KV 块 + 引用计数 + LRU（正在用的不淘汰）。把 `insert` 的键从字符换成 token、再按固定大小分块，就是 vLLM 的哈希块方案。见[前缀缓存](serving://engine/prefix-cache/)。
