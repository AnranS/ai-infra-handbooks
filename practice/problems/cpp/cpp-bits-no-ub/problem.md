---
title: 没有未定义行为的位操作：bf16 舍入、溢出检查与未对齐读取
chapter: basics/compile-ub.md
difficulty: 中等
tags: [未定义行为, UBSan, bit_cast, bf16]
---
写 kernel 和数值代码时常用的几个小工具，要求**没有任何未定义行为**（测试在 AddressSanitizer + UBSan 下运行，UBSan 报告即失败）：

1. `float_bits(float) -> uint32_t`、`bits_float(uint32_t) -> float`：查看 / 构造 float 的位模式；
2. `f32_to_bf16(float) -> uint16_t`：fp32 转 bf16，取高 16 位，按**就近舍入、正好一半时取偶数**（round-to-nearest-even）；`inf` 保持 `inf`，最大的有限 float 舍入后变成 `inf`；NaN 必须仍是 NaN（包括尾数只在低 16 位有 1 的那种）。`bf16_to_f32(uint16_t) -> float` 做反向转换；
3. `checked_add(int64_t a, int64_t b, int64_t* out) -> bool`：不溢出时写 `*out = a + b` 并返回 `true`；溢出时返回 `false` 且不写 `out`；
4. `midpoint(int32_t a, int32_t b) -> int32_t`：$\lfloor (a + b) / 2 \rfloor$（向负无穷取整），对任何输入都不能溢出；
5. `load_u32(const unsigned char* p) -> uint32_t`：从**任意地址**（不一定按 4 字节对齐）读 4 个字节，按机器字节序解释。

模板是"看起来能跑"的写法：`reinterpret_cast` 读 float 的位、先加再检查溢出、`(a + b) / 2`、强转指针做未对齐的读取——每一个都是未定义行为或者算错了。

<!-- 题解 -->
- **严格别名**：通过 `uint32_t*` 读一个 `float` 对象是未定义行为，优化器可以假设两种类型的指针不指向同一块内存。合法的写法是 `std::bit_cast`（C++20）或 `std::memcpy`，编译器会把它优化成一条寄存器移动；sanitizer 查不出这一类，要靠自己遵守；
- **bf16 舍入**：`u + 0x7FFF + ((u >> 16) & 1)` 再右移 16 位——低 16 位大于一半时进位，正好一半时只有保留部分是奇数才进位。进位可能一路进到指数（这正是正确的结果），最大的有限值会进成 `inf`。NaN 要单独处理：直接截断时，尾数只在低 16 位的 NaN 会变成 `inf`，所以要置上静默位（`| 0x0040`）。PyTorch 的 `c10::BFloat16`、CUDA 的 `__float2bfloat16_rn` 都是这个算法；
- **有符号溢出**是未定义行为，"先加再看符号"的检查可能被编译器整个删掉（它有权假设溢出不会发生）。先和 `INT64_MAX - b` 比较，或者用 `__builtin_add_overflow`；
- **中点**：换成 64 位再算最简单；C++20 的 `std::midpoint` 也不会溢出，但它向 `a` 取整，和这里的要求不同；
- **未对齐的访问**在 x86 上"碰巧能跑"，在 ARM 的某些指令、GPU 上会出错，而且是未定义行为（UBSan 的 `alignment` 检查会报）。`memcpy` 到局部变量是标准写法；GPU kernel 里用向量化加载（`float4`）时，也必须保证地址按 16 字节对齐。
