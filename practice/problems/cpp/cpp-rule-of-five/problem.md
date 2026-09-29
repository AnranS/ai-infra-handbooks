---
title: 五法则：一维张量类
chapter: basics/move.md
difficulty: 中等
tags: [五法则, 移动语义, 异常安全]
---
实现一个用裸数组保存数据的一维张量 `Tensor1D`（练习五法则；真实代码里直接用 `std::vector<float>` 当成员就好）：

- `Tensor1D()`：空张量（`size() == 0`，`data() == nullptr`）；`Tensor1D(std::size_t n, float value)`：`n` 个元素都初始化为 `value`；
- 拷贝构造、拷贝赋值：**深拷贝**，拷贝赋值要正确处理自我赋值，并且在分配失败时保持原对象不变（强异常安全保证，推荐"拷贝再交换"）；
- 移动构造、移动赋值：转移所有权，源对象变成空张量；都标 `noexcept`；
- 析构函数释放内存；
- `std::size_t size() const`、`float* data()`、`const float* data() const`、`float& operator[](std::size_t)`、`float operator[](std::size_t) const`。

测试会把它放进 `std::vector` 并反复扩容：移动构造没有 `noexcept` 时，vector 会退回到拷贝。所有测试在 AddressSanitizer 下运行，泄漏、重复释放都算失败。

<!-- 题解 -->
五个特殊成员函数的分工见 [移动语义与完美转发](cpp://basics/move/) 的练习 1。几个容易错的点：

- 移动构造用 `std::exchange` 把源对象的指针和长度置空，析构时对空指针 `delete[]` 是安全的；
- 拷贝赋值写成"先拷贝到临时对象、再 `swap`"：分配可能抛异常，但那时 `*this` 还没动；
- 移动赋值要先释放自己原来的数组，否则泄漏；
- 不写 `noexcept`，`vector` 扩容时会调用拷贝构造，测试里会统计出多余的深拷贝。
