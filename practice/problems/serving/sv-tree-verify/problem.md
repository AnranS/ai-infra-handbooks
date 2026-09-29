---
title: 树形草稿：注意力掩码与最长接受路径
chapter: topics/speculative.md
difficulty: 中等
tags: [投机解码, 树形注意力, EAGLE]
---
EAGLE、Medusa 让草稿模型一次提出一棵**候选树**，目标模型用一次前向验证整棵树：把树上所有节点拼成一个序列，用特殊的注意力掩码让每个节点只看到自己的祖先。

树用列表描述：`tree = [(parent, token), ...]`，节点编号是列表下标，`parent = -1` 表示父节点是树根（树根是已经确定的最后一个 token，不在列表里）。父节点的编号总是小于子节点。实现：

1. `tree_mask(tree)`：`(n, n)` 的 bool 数组，`mask[i][j]` 为真当且仅当 `j` 是 `i` 自己或 `i` 的祖先；
2. `tree_positions(tree, base)`：每个节点的位置 = `base + 深度`（树根的直接孩子深度为 1，所以位置是 `base + 1`）；
3. `accept(tree, root_next, node_next)`：目标模型的贪心预测：`root_next` 是它在树根之后预测的 token，`node_next[i]` 是它在节点 `i` 之后预测的 token。
   从树根出发，每一步在当前节点的孩子里找 token 等于"当前节点的预测"的那个（有多个时取编号最小的），沿着走下去；走不下去时停止。
   返回 `(accepted_nodes, tokens)`：接受的节点编号列表，以及输出的 token 列表（接受的节点的 token，再加上最后一个节点的预测作为奖励 token）。

```python
tree = [(-1, 10), (-1, 11), (0, 20), (0, 21), (2, 30)]
accept(tree, root_next=10, node_next=[21, 99, 30, 7, 8])   # ([0, 3], [10, 21, 7])
```

<!-- 题解 -->
掩码：`mask[i] = mask[parent[i]]`（父节点的祖先集合）再加上自己；因为父节点编号更小，按编号顺序一遍就能算完。深度同理：`depth[i] = depth[parent] + 1`。

接受：维护当前节点 `cur`（树根记为 -1）和它的预测 `want`，在 `children[cur]` 里找 token 等于 `want` 的节点。
一次前向验证了整棵树，平均能接受的 token 比链式草稿多——树的分支覆盖了草稿模型不确定的地方。
