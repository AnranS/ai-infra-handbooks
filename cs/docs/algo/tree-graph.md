# 树与图：遍历、拓扑序、并查集与前缀树

<p class="lead">推理系统里到处是树和图：计算图要排出合法的执行顺序，前缀缓存是一棵按 token 分叉的树，投机解码的草稿是一棵待验证的树，分布式作业的依赖关系是一张有向无环图。面试里这一类题也最能看出基本功——遍历能不能写成迭代、BFS 和 DFS 分别适合什么、拓扑排序怎么判环、并查集的两个优化各解决什么。这一章把这些模板写清楚，并用计算图和 Radix Cache 作为例子。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 二叉树的前、中、后、层序遍历，各用什么数据结构写成迭代？
    2. 为什么在 Python 里树和图的题最好写迭代？
    3. BFS 求出的"层数"是最短路。为什么它不等于计算图里"最早能执行的时刻"？
    4. 拓扑排序怎么判断图里有环？
    5. 并查集的两个优化是什么？各解决什么问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 前序和后序用栈（后序可以按"根右左"遍历再整体反转）；中序用栈加"一路向左"的写法；层序用队列，每轮先记下当前队列长度就能按层切分。
    2. 默认递归深度只有 1000。退化成链的二叉树、长链表、深度很大的图都会 `RecursionError`；而且函数调用开销不小。面试时可以说"我写迭代，避免爆栈"，这本身就是一个加分点。
    3. BFS 的最短路是"最少经过几条边"，而计算图里一个算子要等**所有**前驱都完成，对应的是**最长路径**（关键路径）。所以排执行顺序要用拓扑排序加动态规划求最长路，不是 BFS 的最短路。
    4. 入度为 0 的节点逐个出队、把后继的入度减一；最后如果出队的节点数少于总节点数，剩下的节点一定在环里。DFS 写法则是用三色标记，遇到"正在访问中"的节点说明有环。
    5. **路径压缩**（find 时把沿途节点直接挂到根上）和**按大小/秩合并**（把小树挂到大树下）。前者让树变平，后者避免树长歪；两者合用后单次操作近似 O(1)（反阿克曼函数）。

## 树的遍历：全部写成迭代

```python title="traverse.py"
# 树的四种遍历，全部写成迭代（Python 的递归深度只有 1000，退化成链的树会崩）
from collections import deque


class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


root = T(1, T(2, T(4), T(5)), T(3, None, T(6)))


def preorder(node):
    out, stack = [], [node] if node else []
    while stack:
        cur = stack.pop()
        out.append(cur.val)
        if cur.right:
            stack.append(cur.right)            # 先压右，后压左：弹出时左先
        if cur.left:
            stack.append(cur.left)
    return out


def inorder(node):
    out, stack = [], []
    while node or stack:
        while node:                            # 一路向左
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def postorder(node):
    out, stack = [], [node] if node else []
    while stack:                               # 按"根右左"遍历再整体反转，就是"左右根"
        cur = stack.pop()
        out.append(cur.val)
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.right)
    return out[::-1]


def level_order(node):
    out, q = [], deque([node] if node else [])
    while q:
        level = []
        for _ in range(len(q)):                # 一次处理一整层
            cur = q.popleft()
            level.append(cur.val)
            q.extend(x for x in (cur.left, cur.right) if x)
        out.append(level)
    return out


print("       1")
print("      / \\")
print("     2   3")
print("    / \\   \\")
print("   4   5   6")
print("前序（根左右）：", preorder(root))
print("中序（左根右）：", inorder(root))
print("后序（左右根）：", postorder(root))
print("层序：", level_order(root))
print()
deep = T(0)
cur = deep
for i in range(1, 20000):                      # 退化成链的树：递归一定爆栈
    cur.right = T(i)
    cur = cur.right
print("两万层的退化树，迭代前序遍历的前几个：", preorder(deep)[:5], "…… 共", len(preorder(deep)), "个节点")
```

```text title="输出"
       1
      / \
     2   3
    / \   \
   4   5   6
前序（根左右）： [1, 2, 4, 5, 3, 6]
中序（左根右）： [4, 2, 5, 1, 3, 6]
后序（左右根）： [4, 5, 2, 6, 3, 1]
层序： [[1], [2, 3], [4, 5, 6]]

两万层的退化树，迭代前序遍历的前几个： [0, 1, 2, 3, 4] …… 共 20000 个节点
```

最后一行是重点：**两万层的退化树**用递归一定爆栈，迭代没问题。二叉搜索树按顺序插入有序数据就会退化成这样，所以这不是人为构造的极端情况。

几个常用结论：

- **中序遍历二叉搜索树得到升序序列**，所以"验证 BST""找第 k 小"都可以用中序；
- **层序遍历**天然适合"每层做一件事"的题（右视图、每层最大值、之字形遍历）；
- **后序遍历**适合"需要先知道子树结果"的题（求高度、判平衡、树形 DP、最近公共祖先）。递归写起来更自然时，面试里可以先写递归、再说"数据大时改成迭代或显式栈"。

## 图：BFS、DFS 与拓扑排序

```python title="graph.py"
# 图的三个基本算法：BFS 最短路（无权）、拓扑排序、并查集
from collections import defaultdict, deque

# 一个小的依赖图：算子 -> 依赖它的算子（计算图的样子）
edges = [("embed", "attn"), ("attn", "norm1"), ("norm1", "mlp"), ("mlp", "norm2"),
         ("embed", "residual"), ("residual", "norm2"), ("norm2", "logits")]
graph = defaultdict(list)
indeg = defaultdict(int)
nodes = set()
for a, b in edges:
    graph[a].append(b)
    indeg[b] += 1
    nodes |= {a, b}


def bfs_dist(start):
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in dist:                # 第一次访问就是最短距离
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def topo_sort():
    deg = {n: indeg[n] for n in nodes}
    q = deque(sorted(n for n in nodes if deg[n] == 0))   # 排序只为输出稳定
    out = []
    while q:
        cur = q.popleft()
        out.append(cur)
        for nxt in graph[cur]:
            deg[nxt] -= 1
            if deg[nxt] == 0:
                q.append(nxt)
    return out if len(out) == len(nodes) else None       # 少了节点说明有环


class DSU:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:                  # 按大小合并
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


print("从 embed 出发的层数：", dict(sorted(bfs_dist("embed").items())))
print("拓扑序（一种合法的执行顺序）：", topo_sort())

graph["logits"].append("embed")                # 加一条回边，制造环
indeg["embed"] += 1
print("加一条 logits -> embed 之后：", topo_sort(), "（None 表示有环，计算图非法）")
graph["logits"].pop()
indeg["embed"] -= 1

dsu = DSU(nodes)
merged = [dsu.union(a, b) for a, b in edges]
print("\n并查集：把所有有依赖关系的算子并起来后，连通块个数 =",
      len({dsu.find(n) for n in nodes}), "，合并成功的边数 =", sum(merged), "（其余是成环的边）")
```

```text title="输出"
从 embed 出发的层数： {'attn': 1, 'embed': 0, 'logits': 3, 'mlp': 3, 'norm1': 2, 'norm2': 2, 'residual': 1}
拓扑序（一种合法的执行顺序）： ['embed', 'attn', 'residual', 'norm1', 'mlp', 'norm2', 'logits']
加一条 logits -> embed 之后： None （None 表示有环，计算图非法）

并查集：把所有有依赖关系的算子并起来后，连通块个数 = 1 ，合并成功的边数 = 6 （其余是成环的边）
```

几点说明：

- **BFS 给的是最短路（边数最少）**，DFS 给的是"能不能到达"和连通性。网格题（岛屿数量、最短路径）里 BFS 用队列、DFS 用栈或递归，关键都是**访问标记要在入队/入栈时打**，不是出队时才打，否则同一个节点会被重复加入。
- **拓扑排序**用入度表 + 队列（Kahn 算法）：入度为 0 的先出队，出队的节点把后继入度减一。最后出队的节点数少于总数，就说明有环。计算图、任务依赖、课程表都是这一类。
- 上面的例子里，`norm2` 的 BFS 层数是 2，但它要等 `mlp`（层数 3）完成才能执行。**排执行顺序要用拓扑序上的最长路（关键路径），不是 BFS 的最短路**。真实的算子调度还要考虑并行度和显存，见[计算图与编译](serving://engine/graphs-compile/)。
- **并查集**（DSU）解决的是"这两个点连通吗"：路径压缩 + 按大小合并之后，单次操作近似 O(1)。典型用途：判断无向图连通分量、Kruskal 最小生成树、检测加边是否成环、"账户合并"这类等价类归并。

带权最短路要换算法：**Dijkstra**（非负权，用堆，O((V+E) log V)）、**Bellman-Ford**（可以有负权，O(VE)）、**Floyd**（所有点对，O(V³)）。集群里"选一条延迟最低的路由路径"就是 Dijkstra。

## 前缀树：从字典树到 Radix Cache

![图：基数树——共享前缀的请求共享同一条路径](../assets/figures/radix-tree.svg){.aig-svg}

```python title="trie.py"
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
```

```text title="输出"
包含 'car'： True ；包含 'ca'： False
以 'ca' 开头的词有 4 个；以 'car' 开头的有 3 个
'cards' 与已有内容的最长公共前缀长度： 4
'dogs' ： 3 ；'xyz'： 0

同一个系统提示词 + 新问题：可复用前缀 5 个 token，需要重算 1 个
换了系统提示词：可复用前缀 0 个 token，需要重算 6 个
完全相同的请求：可复用前缀 7 个 token，需要重算 0 个
```

前缀树的每个节点代表一个前缀，边是一个字符（或一个 token）。它的价值是**共享前缀只存一份**，并且能在 O(前缀长度) 时间里回答"有没有这个前缀""以它开头的有几个""最长能匹配多少"。

推理引擎的 **Radix Cache** 就是它的压缩版：只有一个孩子的链被压成一条边（省内存、减少指针跳转），每个节点挂着对应的 KV 块，再加上引用计数和 LRU 淘汰。上面最后几行模拟的正是它的核心问题——**一个新请求能复用多少前缀**：同一个系统提示词换个问题，前缀全命中，只需要算新的那部分；系统提示词一变，从第一个 token 就不匹配。这也解释了为什么"把系统提示词放在最前面、把变化的内容放在后面"能大幅提高缓存命中率（见[前缀缓存](serving://engine/prefix-cache/)）。

## 这一章的题在考什么

| 看到这些字眼 | 往哪个模板想 |
| --- | --- |
| 二叉树、深度、路径、子树 | 递归/后序（数据大时改迭代） |
| 每一层、右视图、之字形 | 层序（队列） |
| 二叉搜索树、第 k 小、验证 | 中序 |
| 最短步数、最少操作、网格 | BFS |
| 连通、是否存在路径、岛屿 | DFS 或并查集 |
| 依赖、课程表、执行顺序、判环 | 拓扑排序 |
| 合并集合、是否同一组、加边成环 | 并查集 |
| 前缀、自动补全、公共前缀 | 前缀树 |
| 带权最短路 | Dijkstra（堆） |

!!! interview "怎么讲清楚"
    讲树先说遍历方式和为什么："这道题要先知道子树的结果，所以用后序"；并主动说"Python 递归深度只有 1000，我写迭代/显式栈"。讲图先说清楚**建图方式**（邻接表、入度表）和**访问标记打在哪**（入队时打，避免重复入队），再给复杂度 O(V+E)。拓扑排序要说明判环方式（出队数少于节点数）。并查集要说出两个优化（路径压缩、按大小合并）和近似 O(1)。能把题目映射到工程场景会加分：拓扑排序对应计算图的执行顺序（注意是最长路而不是最短路）、前缀树对应 Radix Cache 的前缀复用、并查集对应连通性判断。

## 小结

- [x] 四种遍历都能写成迭代：前/后序用栈、中序"一路向左"、层序用队列；退化成链的树会让递归爆栈。
- [x] BFS 求最短路、DFS 判连通；访问标记要在入队时打。
- [x] 拓扑排序用入度表判环；计算图的执行顺序要用拓扑序上的最长路，不是 BFS 的最短路。
- [x] 并查集 = 路径压缩 + 按大小合并，单次近似 O(1)。
- [x] 前缀树让共享前缀只存一份；Radix Cache 是它的压缩版，决定了一个请求能复用多少前缀。
