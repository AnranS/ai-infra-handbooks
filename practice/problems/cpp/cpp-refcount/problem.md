---
title: 侵入式引用计数：什么时候可以用 relaxed
chapter: concurrency/atomics.md
difficulty: 中等
tags: [原子操作, 内存序, 引用计数, 智能指针]
sanitize: thread
---
实现一个侵入式引用计数（计数放在对象里，类似 `boost::intrusive_ptr`，推理引擎里常用来管理 KV 块、请求对象这类被多处共享的东西）：

- `RefCounted`：基类，计数是一个 `std::atomic<int>`，新建的对象计数为 1。`retain()` 加一；`release()` 减一，减到 0 时 `delete this` 并返回 `true`；`use_count()` 返回当前计数；
- `RefPtr<T>`：`RefPtr(T*)` 接管一个已有的引用（不加计数）；拷贝构造 / 拷贝赋值加计数，移动构造 / 移动赋值转移所有权（源对象变空）；析构和 `reset()` 释放引用；`get()`、`->`、`*`、`explicit operator bool`；
- `make_ref<T>(args...)`：创建对象并返回持有它的 `RefPtr<T>`。

测试在 **ThreadSanitizer** 下运行：8 个线程同时持有同一个对象，每个线程写自己的一个槽位后释放引用，最后一个释放的线程负责析构，析构函数读所有槽位。计数要对（恰好析构一次），而且不能有数据竞争——想想 `retain` 和 `release` 各自需要什么内存序。模板里的计数全用了 `relaxed`，移动和赋值也没写完。

<!-- 题解 -->
- **加计数可以用 `relaxed`**：能调用 `retain()` 的线程手里已经有一个引用，对象不可能在这时被销毁；加计数也不需要把任何数据"发布"给别人；
- **减计数至少要 `release`**：本线程在放手之前对对象的写入（测试里的槽位）必须在计数变小之前完成、并被最后一个线程看到；
- **最后一个线程要 `acquire`**：它要看到其他所有线程的写入才能安全析构。写法一是 `fetch_sub(1, acq_rel)`；写法二是 `fetch_sub(1, release)`，只在减到 0 时再加一个 `std::atomic_thread_fence(std::memory_order_acquire)`，省掉其他线程上的 acquire（`std::shared_ptr` 的常见实现就是这样）。GCC 的 ThreadSanitizer 不认识独立的 fence，会误报，所以这里用 `acq_rel`；
- **赋值先加后减**：`p = p` 时如果先释放旧的，计数可能先减到 0 把对象销毁了。

`relaxed` 版本在大多数时候"看起来能跑"，x86 上甚至很难观察到错误（x86 的 RMW 指令本身就是全屏障），但在 ARM 上、或者编译器重排之后，析构函数可能读到别的线程还没写完的数据——TSan 按 happens-before 关系检查，不依赖运气，能稳定地报出来。
