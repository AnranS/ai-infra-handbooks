---
title: 编译期派发：头维度与数据类型
chapter: basics/templates.md
difficulty: 中等
tags: [模板, 编译期常量, 派发]
---
注意力 kernel 需要把 `head_dim` 和数据类型作为**编译期常量**（模板参数），而它们在运行时才从模型配置里读出来。实现两个派发函数：

1. `template <class F> decltype(auto) dispatch_head_dim(int head_dim, F&& f)`：支持 64、96、128、256，调用 `f(std::integral_constant<int, D>{})` 并返回它的结果；其他值抛出 `std::invalid_argument`；
2. `template <class F> decltype(auto) dispatch_dtype(DType t, F&& f)`：`DType::F32` 调用 `f(type_tag<float>{})`，`DType::BF16` 调用 `f(type_tag<bf16>{})`，`DType::F16` 调用 `f(type_tag<f16>{})`，返回结果。

`DType`、`type_tag`、`bf16`、`f16` 已经在模板里给出。测试会在回调里用 `decltype(d)::value` 当数组大小（必须是真正的编译期常量），并嵌套两层派发。

```cpp
int d = dispatch_head_dim(128, [](auto hd) { return decltype(hd)::value; });   // 128
std::size_t s = dispatch_dtype(DType::BF16, [](auto tag) { return sizeof(typename decltype(tag)::type); });   // 2
```

<!-- 题解 -->
一个 `switch`，每个 `case` 返回 `f(std::integral_constant<int, D>{})`。`std::integral_constant` 把一个值编码进**类型**里，回调是泛型 lambda，每种类型实例化一次，`decltype(hd)::value` 就是编译期常量。
这正是推理库里 `DISPATCH_HEAD_DIM` 一类宏展开后的样子，见 [模板、concepts 与 constexpr](cpp://basics/templates/) 的"派发"一节。
