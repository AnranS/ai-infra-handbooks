---
title: 结构体布局：对齐、填充与伪共享
chapter: memory/layout.md
difficulty: 简单
tags: [对齐, 内存布局, 缓存, 伪共享]
---
跨语言、跨设备共享的结构体（比如传给 GPU kernel 的参数块、RDMA 里的消息头）必须精确知道每个字段的偏移。按 C/C++ 的规则实现：

- `constexpr std::size_t align_up(std::size_t n, std::size_t a)`：把 `n` 向上取到 `a` 的倍数（`a` 是 2 的幂）；
- `Layout layout_of(const std::vector<Field>& fields)`：`Field` 是 `{size, align}`，按顺序摆放：每个字段放在不小于当前偏移、且是自身对齐的倍数的位置；结构体的对齐是各字段对齐的最大值；总大小向上取到结构体对齐的倍数。返回 `{offsets, size, align}`。没有字段时大小和对齐都是 1（C++ 的空结构体）；
- `std::vector<std::size_t> best_order(const std::vector<Field>& fields)`：返回一个字段的排列（下标列表），让 `layout_of` 得到的大小最小：按对齐从大到小排，对齐相同的保持原来的顺序；
- `PaddedCounter`：里面有一个 `std::atomic<std::uint64_t> value{0};`，并且每个 `PaddedCounter` 独占一个 64 字节的缓存行（`alignof` 和 `sizeof` 都是 64），多个线程各自累加数组里的一个计数器时不会伪共享。

`Field`、`Layout` 已经在模板里定义好。

```cpp
layout_of({{1, 1}, {8, 8}, {4, 4}});   // struct { bool; double; int; }：偏移 {0, 8, 16}，大小 24
best_order({{1, 1}, {8, 8}, {4, 4}});  // {1, 2, 0}：double、int、bool，大小 16
```

<!-- 题解 -->
`align_up(n, a) = (n + a - 1) & ~(a - 1)`。`layout_of` 维护当前偏移：每个字段先对齐、再加上大小；最后把总大小对齐到最大对齐——这就是 `sizeof` 里那些"看不见的字节"的来源，也是数组里下一个元素仍然对齐的保证。

按对齐从大到小排能消掉所有内部填充：C 的基本类型大小都是对齐的倍数，对齐大的字段放完之后，偏移仍是后面每个更小对齐的倍数；剩下的只是末尾的填充，而任何顺序的大小都不会小于"所有字段的大小之和向上取到最大对齐"，所以这个顺序是最优的。

`struct alignas(64) PaddedCounter` 让每个计数器独占一个缓存行。不加对齐时，8 个 8 字节的计数器挤在同一个缓存行里，不同的核心写不同的计数器也要来回争抢这一行（伪共享），多线程计数反而比单线程慢。
