---
title: 反转链表
chapter: algo/linked-stack-hash.md
difficulty: 简单
tags: [链表,双指针,迭代]
---
反转一个单链表，返回新的头节点。节点定义（测试里会用这个类）：

```python
class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt
```

要求**迭代**实现（Python 的递归深度只有 1000，长链表会崩），O(1) 额外空间。同时实现 `reverse_between(head, left, right)`：只反转第 `left` 到第 `right` 个节点（从 1 开始计数，闭区间），其余不动。

```python
# 1->2->3->4->5  反转成  5->4->3->2->1
# reverse_between(1->2->3->4->5, 2, 4) 得到 1->4->3->2->5
```

<!-- 题解 -->
反转的核心是三个指针：`prev`（已反转部分的头）、`cur`（当前节点）、`nxt`（先存下来，否则改完指针就找不到了）。循环体固定四步：存后继、改指向、prev 前移、cur 前移。

局部反转用**虚拟头节点**：先走到第 `left - 1` 个节点（记为 `pre`），然后用"头插法"——每次把 `cur` 的下一个节点摘下来插到 `pre` 后面，做 `right - left` 次。这样不用先断开再接回，边界也不容易错。

面试时可以主动提一句递归写法（`new_head = reverse(head.next); head.next.next = head; head.next = None`）并说明为什么在 Python 里不用它。
