---
title: KV 块池：引用计数与写时复制
chapter: memory/allocators.md
difficulty: 中等
tags: [块分配器, 引用计数, 写时复制]
---
实现分页 KV Cache 的块分配器 `BlockPool`，管理编号 `0` 到 `n-1` 的块：

- `explicit BlockPool(int num_blocks)`：所有块都空闲；
- `std::optional<std::vector<int>> allocate(int n)`：分配 `n` 个块，**要么全部成功，要么一个都不分配**（返回 `std::nullopt`）；成功时引用计数都为 1。分配顺序：**最近释放的块最先被再次分配**（空闲栈），初始时按编号从小到大分配；
- `void retain(int b)`：共享一个已分配的块，引用计数加一；对空闲块调用抛出 `std::logic_error`；
- `void release(int b)`：引用计数减一，减到 0 时回到空闲栈；对空闲块调用（重复释放）抛出 `std::logic_error`；
- `int make_writable(int b)`：准备往块 `b` 里写。如果 `b` 只被一个序列持有，直接返回 `b`；如果被共享，分配一个新块、释放对 `b` 的这一份引用，返回新块号（写时复制；数据的复制由调用者负责）；没有空闲块时返回 `-1` 且不改变任何状态；
- `int num_free() const`、`int refcount(int b) const`。

```cpp
BlockPool pool(4);
auto a = pool.allocate(2);         // {0, 1}
pool.release((*a)[1]);             // 块 1 回到空闲栈
pool.allocate(1);                  // {1}：最近释放的最先分配
```

<!-- 题解 -->
空闲块用一个 `vector<int>` 当栈：构造时按 `n-1, …, 1, 0` 的顺序压入，栈顶就是 0。`allocate` 先检查 `n <= 空闲数`，再依次弹出；`release` 减到 0 时压回栈顶。
写时复制只发生在共享块上：先分配新块（失败就原样返回 `-1`），成功后再 `release(b)`，顺序反过来会在失败时丢掉一份引用。见 [分配器与内存池](cpp://memory/allocators/) 的块分配器一节，以及推理系统手册的 [分页 KV Cache](serving://engine/paged-kv/)。
