---
title: RAII 文件描述符
chapter: basics/value-raii.md
difficulty: 简单
tags: [RAII, 移动语义, 资源管理]
---
实现一个独占文件描述符的 RAII 类 `UniqueFd`。测试程序提供了一对假的系统调用：`int fake_open()` 返回一个新的描述符（非负整数），`void fake_close(int fd)` 关闭它——重复关闭、关闭一个没打开的描述符都会被记为错误，结束时还开着的描述符算泄漏。

要求：

- `UniqueFd()` 构造一个空对象（持有 `-1`）；`explicit UniqueFd(int fd)` 接管 `fd`；
- 析构时关闭持有的描述符，空对象什么都不做；
- **不能拷贝，可以移动**：移动后源对象变空；移动赋值要先关闭自己原来持有的；自我移动赋值不能出错；移动操作标 `noexcept`；
- `int get() const`；`explicit operator bool() const`（非空时为真）；
- `int release()`：放弃所有权并返回描述符，**不关闭**，之后对象为空；
- `void reset(int fd = -1)`：关闭原来持有的（如果有），再接管 `fd`。

```cpp
{
  UniqueFd f(fake_open());
  UniqueFd g = std::move(f);   // f 变空，g 持有
}                              // g 析构时关闭，没有泄漏
```

<!-- 题解 -->
这是 [值语义与 RAII](cpp://basics/value-raii/) 和 [移动语义](cpp://basics/move/) 两章的综合：析构负责释放，拷贝删除，移动用 `std::exchange` 把源对象置成 `-1`。
移动赋值和 `reset` 的共同点是"先释放自己的，再接管新的"，可以让移动赋值直接调用 `reset(other.release())`——它天然处理了自我赋值：`release()` 先把自己置空并返回原值，`reset` 再把同一个值接管回来，中间不会关闭任何东西。
