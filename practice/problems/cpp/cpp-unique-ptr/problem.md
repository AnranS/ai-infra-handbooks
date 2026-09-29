---
title: 实现 unique_ptr
chapter: basics/ownership.md
difficulty: 中等
tags: [智能指针, 移动语义, 模板]
---
实现一个简化的 `UniquePtr<T, D = DefaultDelete<T>>`（模板里已经给出了 `DefaultDelete`）：

- `UniquePtr()`、`UniquePtr(std::nullptr_t)`：空指针；`explicit UniquePtr(T* p, D d = D())`：接管 `p`，并保存删除器；
- 析构时如果非空，调用删除器 `d(p)`；
- **不能拷贝，可以移动**（都标 `noexcept`），移动时删除器也一起移动；移动赋值要先释放自己原来持有的对象；
- `T* get() const`、`T& operator*() const`、`T* operator->() const`、`explicit operator bool() const`；
- `T* release()`：放弃所有权并返回指针，不调用删除器；`void reset(T* p = nullptr)`：释放原来的，接管 `p`；
- `D& get_deleter()`；
- 用默认删除器（没有成员的类型）时，`sizeof(UniquePtr<T>)` 必须等于 `sizeof(T*)`——提示：C++20 的 `[[no_unique_address]]`。

<!-- 题解 -->
和 [智能指针与所有权](cpp://basics/ownership/) 里的讲解一致：`unique_ptr` 就是"一个指针 + 一个删除器"的 RAII 包装。
没有成员的删除器类型本来也要占 1 个字节（再加上对齐就是 8 个字节），`[[no_unique_address]]` 允许编译器让它和别的成员共用地址，于是 `sizeof` 等于一个指针——标准库里的实现用"空基类优化"达到同样的效果。
`reset` 要先把新指针存进去、再删除旧的对象：旧对象的析构函数可能反过来访问这个 `UniquePtr`，先更新状态更安全。
