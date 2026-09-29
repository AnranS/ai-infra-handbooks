---
title: 每一步的临时内存：arena 分配器
chapter: memory/allocators.md
difficulty: 中等
tags: [内存池, 对齐, arena]
---
推理引擎每一步都要准备一批临时数组（位置、槽位映射、序列长度……），用完就扔。实现一个 arena（线性分配器）`Arena`：

- `explicit Arena(std::size_t capacity)`：持有一块 `capacity` 字节的内存（可以假设 `capacity` 是 4096 的倍数）；
- `void* allocate(std::size_t n, std::size_t align = alignof(std::max_align_t))`：返回一块 `n` 字节、**地址**按 `align` 对齐的内存（`align` 是 2 的幂，最大 4096）；空间不够时抛出 `std::bad_alloc`；`n` 为 0 时也返回一个有效（对齐的）地址；
- `void reset()`：一次性回收全部内存，之后的分配从头开始（同样的请求序列应该得到同样的地址）；
- `std::size_t used() const`：从缓冲区起点到当前分配末尾的字节数（包括对齐造成的空隙）；`std::size_t capacity() const`。

析构时释放缓冲区。测试会检查对齐、分配之间互不重叠、越界时抛异常、`reset` 之后复用。

<!-- 题解 -->
缓冲区用 `std::aligned_alloc(4096, capacity)` 分配，或者对**绝对地址**做对齐：`p = (base + off + align - 1) & ~(align - 1)`。
只对偏移量对齐是常见的错误：缓冲区起点只保证 16 字节对齐时，偏移量按 256 对齐并不能得到按 256 对齐的地址。见 [分配器与内存池](cpp://memory/allocators/) 的 arena 一节。
