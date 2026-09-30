---
title: 二叉树的迭代遍历
chapter: algo/tree-graph.md
difficulty: 简单
tags: [二叉树,迭代,栈]
---
用**迭代**（不能递归）实现二叉树的三种遍历，节点类：

```python
class T:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right
```

1. `preorder(root)`：前序（根左右）；
2. `inorder(root)`：中序（左根右）；
3. `level_order(root)`：层序，返回**每层一个列表**。

```python
# 树：1(2(4,5), 3(None,6))
preorder(root)      # [1, 2, 4, 5, 3, 6]
inorder(root)       # [4, 2, 5, 1, 3, 6]
level_order(root)   # [[1], [2, 3], [4, 5, 6]]
```

<!-- 题解 -->
前序用栈，**先压右孩子再压左孩子**（栈是后进先出，这样弹出时左先）。中序用"一路向左"的写法：把沿途节点压栈，走到头就弹出一个记录值，再转向它的右子树。层序用队列，每轮开始先记下 `len(q)`，只处理这么多个节点，就能按层切开。

必须写迭代的原因：Python 默认递归深度 1000，而二叉搜索树按有序数据插入会退化成一条链，深度等于节点数。测试里有一棵两万层的退化树专门检查这一点。

后序可以按"根右左"遍历再整体反转（等价于"左右根"），这是最好写的迭代后序。
