---
title: 修掉五个经典的坑
chapter: practice/pitfalls.md
difficulty: 简单
tags: [修 bug, 可变默认参数, 浅拷贝]
---
模板里的五个函数各有一个 Python 新手常踩的坑，找出并修好它们（函数签名不变）：

1. `add_request(req, queue=[])`：把请求加进队列并返回队列。不传 `queue` 时每次都应该是新的空队列。
2. `remove_finished(reqs)`：**原地**删掉 `reqs` 里 `finished` 为 `True` 的请求（`reqs` 是字典列表），返回 `None`。
3. `make_grid(rows, cols)`：返回 `rows × cols` 的二维列表，初始全是 0，修改一个格子不应该影响其他行。
4. `same_model(a, b)`：两个模型名字符串相同就返回 `True`。
5. `total_cost(prices)`：把一组浮点数价格加起来，返回四舍五入到分（2 位小数）的结果，并且要保证 `total_cost([0.1, 0.2]) == 0.3`。

<!-- 题解 -->
1. 默认参数只在函数定义时求值一次，多次调用共享同一个列表：改成 `queue=None`，函数里 `if queue is None: queue = []`；
2. 遍历列表时删除元素会跳过下一个元素：用切片赋值 `reqs[:] = [r for r in reqs if not r["finished"]]`；
3. `[[0] * cols] * rows` 里每一行都是同一个列表对象：改成 `[[0] * cols for _ in range(rows)]`；
4. `is` 比较的是对象身份，字符串要用 `==`（短字符串碰巧被驻留时 `is` 也为真，所以这个 bug 很隐蔽）；
5. 浮点数有表示误差，`0.1 + 0.2` 是 `0.30000000000000004`：结果用 `round(sum(prices), 2)`，或者用 `decimal.Decimal` / 整数分来算。
