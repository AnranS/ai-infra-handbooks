---
title: 实现线程安全引用计数的 shared_ptr
chapter: basics/ownership.md
difficulty: 困难
tags: [智能指针, 原子操作, 内存序]
sanitize: thread
---
实现一个简化的 `SharedPtr<T>`：多个 `SharedPtr` 共享一个对象，最后一个被销毁（或 `reset`）时删除对象。

- 控制块里放一个 `std::atomic<long>` 引用计数和对象指针；
- `SharedPtr()`：空；`explicit SharedPtr(T* p)`：接管 `p`（`p` 为空时也是空的 `SharedPtr`）；
- 拷贝构造、拷贝赋值：计数加一；移动构造、移动赋值：转移，不改计数，源对象变空；赋值要正确处理自我赋值；
- 析构和 `reset()`：计数减一，减到 0 时删除对象和控制块；
- `T* get() const`、`T& operator*() const`、`T* operator->() const`、`long use_count() const`（空时为 0）、`explicit operator bool() const`。

测试会让多个线程同时拷贝、读取、销毁指向同一个对象的 `SharedPtr`，并在 **ThreadSanitizer** 下运行：计数的增减必须是原子的，而且"最后一个减到 0 的线程删除对象"必须能看到其他线程对对象的全部访问——想想减计数应该用什么内存序。

<!-- 题解 -->
加一可以用 `relaxed`：拿到一个新副本的线程本来就已经通过某个现有的副本"看到"了对象。
减一必须用 `acq_rel`（或者 `release` 减一，再在减到 0 的那个分支里加一个 `acquire` 栅栏）：release 保证每个线程在减计数之前对对象的读写都"发布"出去，acquire 保证删除对象的那个线程看到所有这些读写之后才删除。
只用 `relaxed` 减计数时，TSan 会报告"删除对象"与"其他线程读取对象"之间的数据竞争。见 [atomic 与内存序](cpp://concurrency/atomics/)。
