---
title: 用两个栈实现队列
chapter: algo/linked-stack-hash.md
difficulty: 简单
tags: [栈,队列,摊还分析]
---
只用两个栈实现先进先出的队列：`push(x)`、`pop()`（弹出并返回队首）、`peek()`（返回队首）、`empty()`。空队列时 `pop` 和 `peek` 返回 `None`。要求**摊还** O(1)。

```python
q = MyQueue()
q.push(1); q.push(2)
q.peek()     # 1
q.pop()      # 1
q.empty()    # False
```

<!-- 题解 -->
两个栈：`in_stack` 负责入队，`out_stack` 负责出队。出队时如果 `out_stack` 空了，就把 `in_stack` **整个倒过去**（顺序正好反过来，变成先进先出），否则直接弹 `out_stack`。

关键是**只在 `out_stack` 为空时才倒**：每个元素一生只会被"压入 in、弹出 in、压入 out、弹出 out"各一次，共 4 次操作，所以摊还是 O(1)。如果每次出队都倒来倒去，就退化成 O(n)。

这是**摊还分析**的经典例子，面试时要能说清"单次最坏 O(n)，但 n 次操作总共 O(n)，所以摊还 O(1)"。同类：用队列实现栈（每次 push 后把前面的元素轮转到后面）、动态数组的扩容。
