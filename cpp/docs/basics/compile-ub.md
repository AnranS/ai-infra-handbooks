# 编译模型与未定义行为

<p class="lead">读大型 C++ 仓库，先要知道代码是怎样被编译、链接起来的：为什么头文件里的函数要写 <code>inline</code>，为什么改一个头文件要重编半个仓库。写 C++，先要知道哪些写法是"未定义行为"：它们不一定当场崩溃，却会在优化、换编译器、换数据之后突然出错。这一章把两件事讲清楚，并把 sanitizer 变成日常工具。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 头文件里定义一个普通（非 `inline`）函数，被两个 `.cpp` 包含，会发生什么？
    2. 头文件里写 `static int counter = 0;`，程序里有几个 `counter`？
    3. `int` 加法溢出和 `unsigned` 加法溢出，哪个是未定义行为？
    4. 用 `reinterpret_cast<uint32_t*>(&f)` 读取一个 `float` 的位模式，有什么问题？应该怎么写？
    5. `std::vector` 的 `push_back` 之后，之前拿到的元素引用还能用吗？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 两个翻译单元各有一份这个函数的定义，链接时报"多重定义"（违反单一定义规则）。要加 `inline`，或者只在头文件里声明、在一个 `.cpp` 里定义。
    2. 每个包含了这个头文件的 `.cpp` 各有一个独立的 `counter`（`static` 意味着内部链接），互不相干——通常不是想要的结果。全程序共享一个要用 `inline int counter = 0;`。
    3. `int`（有符号）溢出是未定义行为；`unsigned` 按 $2^n$ 取模回绕，是定义良好的。
    4. 违反严格别名规则：通过 `uint32_t*` 读一个 `float` 对象是未定义行为，编译器可能按"两种类型的指针不会指向同一块内存"来优化。应该用 `std::bit_cast<uint32_t>(f)`（C++20）或者 `std::memcpy` 到一个 `uint32_t` 变量里。
    5. 不一定：如果 `push_back` 引起了扩容，元素被搬到新的内存，之前的引用、指针和迭代器都悬垂了。事先 `reserve` 足够的容量，或者用下标代替引用。

## 从源码到可执行文件

一个 `.cpp` 文件经过四步变成程序：

1. **预处理**：展开 `#include`（就是把头文件的文本原样粘进来）、宏和条件编译；
2. **编译**：把预处理后的整个文本——一个**翻译单元**（translation unit）——编译成汇编；
3. **汇编**：得到目标文件（`.o`），里面是机器码和一张符号表："我定义了哪些函数和变量，我引用了哪些别人的"；
4. **链接**：把所有 `.o` 和库合在一起，把每个"引用"对上某个"定义"。

几个直接的推论：

- 每个 `.cpp` **独立编译**，编译器看不到别的 `.cpp` 里的内容。一个函数要在多个 `.cpp` 里调用，就把**声明**放进头文件，**定义**放进某一个 `.cpp`；
- 头文件被多少个 `.cpp` 包含，就被编译多少遍。模板必须把定义写在头文件里（每个用到它的翻译单元都要看到完整定义才能实例化），这就是重模板的库编译慢的原因：一个按数据类型 × 头维度 × 页大小展开的注意力 kernel，实例化几百份是常事；
- 链接器只认符号名。两个 `.cpp` 定义了同名的普通函数，链接时报"重复定义"；只声明没定义，报"未定义的引用"。

### 头文件里能放什么：`inline` 与单一定义规则

**单一定义规则**（One Definition Rule，ODR）：一个普通函数或全局变量在整个程序里只能有一个定义。但头文件里的定义会出现在每个包含它的翻译单元里，于是 C++ 给了一个例外：标了 `inline` 的函数和变量（C++17 起），允许在多个翻译单元里各有一份**完全相同**的定义，链接器只保留一份。

类内定义的成员函数、模板、`constexpr` 函数都隐式是 `inline` 的，所以它们可以放心写在头文件里。

`static` 的含义则完全不同：它让名字只在本翻译单元内可见（内部链接），于是**每个包含它的 `.cpp` 都有自己独立的一份**。下面的例子把两者放在一起：

```cpp title="counter.hpp"
#pragma once

static int static_counter = 0;   // 内部链接：每个翻译单元一份
inline int inline_counter = 0;   // C++17 inline 变量：整个程序一份
```

```cpp title="a.cpp" lib="yes"
#include "counter.hpp"

void bump_in_a() {
  ++static_counter;
  ++inline_counter;
}
```

```cpp title="b.cpp" lib="yes"
#include "counter.hpp"

void bump_in_b() {
  ++static_counter;
  ++inline_counter;
}
```

```cpp title="counter_main.cpp" with="a.cpp b.cpp"
// g++ -std=c++20 counter_main.cpp a.cpp b.cpp && ./a.out
#include <cstdio>

#include "counter.hpp"

void bump_in_a();
void bump_in_b();

int main() {
  bump_in_a();
  bump_in_a();
  bump_in_b();
  std::printf("main 看到的 static_counter = %d\n", static_counter);
  std::printf("inline_counter = %d\n", inline_counter);
}
```

```text title="输出"
main 看到的 static_counter = 0
inline_counter = 3
```

`a.cpp`、`b.cpp`、`main` 各自改的是自己那份 `static_counter`，所以 `main` 看到的还是 0；`inline_counter` 全程序只有一份，被加了 3 次。
想在头文件里放一个全局共享的变量（比如一个全局的配置、日志级别），用 `inline`；想要"每个文件私有"的，放进 `.cpp` 里的匿名命名空间，不要在头文件里写 `static`。

!!! warning "违反 ODR 通常不会报错"
    如果两个 `.cpp` 里各有一个 `inline int block_size() { return 16; }` 和 `inline int block_size() { return 32; }`（比如两个文件用不同的宏展开了同一个头文件），
    链接器不会报错，只会**随便保留一份**，另一个文件里的调用悄悄地拿到错误的值。大型仓库里"只在某种编译选项组合下出错"的 bug，不少是这样来的。
    教训：头文件的内容不要依赖包含它的文件事先定义了什么宏。

### 名字修饰与 `extern "C"`

C++ 支持重载，所以同名函数在符号表里要加上参数类型才能区分，这叫**名字修饰**（name mangling）：`void launch(float*, int)` 在符号表里是 `_Z6launchPfi`（用 `c++filt _Z6launchPfi` 可以还原）。
C 语言没有重载，也没有修饰。要让 C 代码、Python 的 `ctypes` 或别的语言按名字找到一个 C++ 函数，就用 `extern "C"` 关掉修饰：

```cpp
extern "C" int kv_store_put(const char* key, const void* data, size_t len);   // 符号名就是 kv_store_put
```

推理系统里跨语言的边界（C API、`dlopen` 加载的插件、Rust 调 C++ 的 FFI）几乎都长这样。

## 未定义行为

C++ 标准对某些操作说："程序这样做了，结果是**未定义的**"（undefined behavior，UB）。这不是"结果不确定"，而是**标准对整个程序的行为不再做任何保证**。编译器可以假设 UB 永远不会发生，并据此优化——这是 C++ 快的原因之一，也是它危险的原因。

一个经典的例子：

```cpp
bool will_overflow(int x) { return x + 1 < x; }   // 想检查 x + 1 会不会溢出
```

有符号整数溢出是 UB，所以编译器可以认为 `x + 1` 一定不溢出，于是 `x + 1 < x` 永远为假。`g++ -O2` 会把整个函数编译成 `return false;`。
检查溢出要在做运算**之前**比较：`x == INT_MAX`，或者用 `__builtin_add_overflow`。

推理系统代码里最常见的几类 UB：

| 类别 | 例子 | 怎么发现 |
| --- | --- | --- |
| 越界访问 | 下标越界、`memcpy` 长度算错 | ASan |
| 悬垂的指针和引用 | 容器扩容后继续用旧引用、返回局部变量的引用、`delete` 之后再用 | ASan |
| 有符号整数溢出 | `int` 算的 token 数、字节数超过 $2^{31}$ | UBSan |
| 违反严格别名、未对齐访问 | 用 `reinterpret_cast` 把字节缓冲区当成 `float*` / `uint32_t*` 读 | UBSan（部分） |
| 未初始化的读取 | 局部变量、`new` 出来的数组没初始化就用 | MSan（clang）、Valgrind |
| 数据竞争 | 两个线程无同步地读写同一个变量 | TSan |

下面挨个看 sanitizer 怎么把它们抓出来。

### 容器扩容让引用悬垂

这是写调度器、KV 管理代码时最容易犯的错。**这个程序有 bug：**

```cpp title="dangling.cpp" expect="fail"
#include <cstdio>
#include <vector>

int main() {
  std::vector<int> blocks = {1, 2, 3};
  int& first = blocks[0];                       // 引用指向 vector 内部的缓冲区
  for (int i = 0; i < 100; ++i) blocks.push_back(i);   // 扩容：缓冲区搬家，旧的被释放
  std::printf("%d\n", first);                   // 读已经释放的内存
}
```

```text title="ASan 的报告（节选）"
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x502000000010
READ of size 4 at 0x502000000010 thread T0
    #0 in main dangling.cpp:8
freed by thread T0 here:
    #0 in operator delete(void*, unsigned long)
    #1 in std::vector<int>::_M_realloc_insert ...
```

不开 sanitizer 时，这个程序多半会"正常"打印一个数——有时是 1，有时是垃圾值，取决于被释放的内存有没有被别人重用。这正是 UB 可怕的地方。
修法：保存**下标**而不是引用；或者事先 `reserve` 足够的容量；或者换成扩容时不搬家的容器（`std::deque` 的元素地址在两端插入时不变）。
`std::unordered_map` 的 rehash、`std::string` 的增长也有同样的问题。

### 有符号整数溢出

**这个程序有 bug：**

```cpp title="overflow.cpp" expect="fail"
#include <climits>
#include <cstdio>

int main(int argc, char**) {
  int tokens = INT_MAX - 1 + argc;   // 运行时才知道的值：argc 为 1 时就是 INT_MAX
  int next = tokens + 1;             // 有符号溢出：未定义行为
  std::printf("%d\n", next);
}
```

```text title="UBSan 的报告"
overflow.cpp:6:7: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
```

在推理系统里，"token 数 × 隐藏维度 × 字节数"这样的乘积很容易超过 $2^{31}$：128K token × 8192 维 × 2 字节就是 $2^{31}$。字节数、偏移量一律用 `size_t` 或 `int64_t`。
无符号整数的溢出是**有定义**的（按 $2^{n}$ 取模回绕），但回绕通常也是 bug——`size_t` 的 `a - b` 在 `a < b` 时会变成一个巨大的正数。

### 严格别名与 `std::bit_cast`

C++ 规定：一般只能通过对象本身的类型（或 `char` / `unsigned char` / `std::byte`）去访问它。用 `reinterpret_cast<uint32_t*>(&x)` 去读一个 `float` 违反了**严格别名规则**，是 UB，优化后可能读到旧值。
要看一个浮点数的位模式——比如做 bf16 转换——用 C++20 的 `std::bit_cast`，或者用 `memcpy`（编译器会把它优化成一条寄存器移动）：

```cpp title="bits.cpp"
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>

// bf16 就是 float 的高 16 位（这里直接截断；正确的做法是按"舍入到最近的偶数"进位）
std::uint16_t to_bf16_truncate(float x) { return std::bit_cast<std::uint32_t>(x) >> 16; }

int main() {
  float x = 1.0f;
  std::printf("1.0f 的位模式：0x%08x\n", std::bit_cast<std::uint32_t>(x));
  std::uint32_t u;
  std::memcpy(&u, &x, sizeof u);   // C++20 之前的写法，同样正确
  std::printf("memcpy 得到：0x%08x\n", u);
  std::printf("3.14159f 截断成 bf16：0x%04x\n", to_bf16_truncate(3.14159f));
}
```

```text title="输出"
1.0f 的位模式：0x3f800000
memcpy 得到：0x3f800000
3.14159f 截断成 bf16：0x4049
```

### 未对齐的访问

从网络、文件、共享内存里读出来的是字节缓冲区，里面的字段不一定按类型对齐。**这个程序有 bug：**

```cpp title="misaligned.cpp" expect="fail"
#include <cstdint>
#include <cstdio>

int main() {
  alignas(4) unsigned char buf[8] = {0, 1, 0, 0, 0, 2, 0, 0};
  // 从偏移 1 读一个 uint32：地址没有按 4 字节对齐，同时也违反了严格别名
  auto* p = reinterpret_cast<const std::uint32_t*>(buf + 1);
  std::printf("%u\n", *p);
}
```

```text title="UBSan 的报告"
misaligned.cpp:8:15: runtime error: load of misaligned address 0x7ffd... for type 'const uint32_t', which requires 4 byte alignment
```

在 x86 上未对齐的读取通常"碰巧能用"，在 ARM 或 GPU 上可能直接出错，编译器生成向量化指令时也可能出错。正确的写法同样是 `memcpy`：

```cpp title="aligned_read.cpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

std::uint32_t read_u32(const unsigned char* p) {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof v);   // 对任意地址都正确，编译器会生成一条普通的读指令
  return v;
}

int main() {
  unsigned char buf[8] = {0, 1, 0, 0, 0, 2, 0, 0};
  std::printf("%u %u\n", read_u32(buf + 1), read_u32(buf + 4));
}
```

```text title="输出"
1 512
```

（x86 是小端序：`buf[1..4]` 是 `01 00 00 00`，读出来是 1；`buf[4..7]` 是 `00 02 00 00`，是 $2 \times 256 = 512$。）

## 把 sanitizer 变成默认

| 工具 | 编译参数 | 抓什么 | 代价 |
| --- | --- | --- | --- |
| AddressSanitizer | `-fsanitize=address` | 越界、释放后使用、重复释放、泄漏 | 约 2 倍慢、2～3 倍内存 |
| UndefinedBehaviorSanitizer | `-fsanitize=undefined` | 有符号溢出、未对齐、空指针、非法移位等 | 很小 |
| ThreadSanitizer | `-fsanitize=thread` | 数据竞争、锁顺序反转 | 5～15 倍慢，不能和 ASan 同时用 |
| libstdc++ 断言 | `-D_GLIBCXX_ASSERTIONS` | 标准容器的下标越界、空容器取 `front()` 等 | 很小，可以在测试构建里常开 |

几条经验：

- 开发和测试构建默认带 `-fsanitize=address,undefined -g -fno-omit-frame-pointer`，加 `-g` 报告里才有行号；
- UBSan 默认只打印不退出，测试里加 `UBSAN_OPTIONS=halt_on_error=1`（或编译时 `-fno-sanitize-recover=all`），让它变成失败；
- 并发代码单独用 TSan 构建跑一遍（见[线程、锁与条件变量](../concurrency/threads.md)）；
- GPU 代码对应的工具是 `compute-sanitizer`（`memcheck`、`racecheck`、`initcheck`），思路完全一样，见 CUDA 手册的[性能分析](cuda://tools/profiling/)。

libstdc++ 断言的例子。**这个程序有 bug**（下标等于长度）：

```cpp title="assertions.cpp" expect="fail" sanitize="none" flags="-D_GLIBCXX_ASSERTIONS"
#include <vector>

int main() {
  std::vector<float> logits(128);
  int token = 128;                 // 词表大小是 128，合法下标是 0～127
  return logits[token] > 0;
}
```

```text title="运行结果"
/usr/include/c++/12/bits/stl_vector.h:1123: ... Assertion '__n < this->size()' failed.
Aborted
```

!!! interview "面试怎么答"
    编译模型题：每个 `.cpp` 是一个翻译单元，头文件只是被粘贴的文本，链接器按符号名把它们接起来；头文件里的普通函数定义被两个 `.cpp` 包含会重复定义，要写成 `inline`（全程序一份）或模板；头文件里的 `static` 变量每个翻译单元各一份。未定义行为题要讲清"编译器假设它不发生并据此优化"，所以错误可能出现在离 bug 很远的地方；常见的有越界、`vector` 扩容后的悬垂引用、有符号溢出（无符号溢出是定义好的回绕）、违反严格别名（看位模式用 `std::bit_cast` 或 `memcpy`）、数据竞争。开发构建默认开 ASan + UBSan，并发代码跑 TSan。

## 练习

1. 下面哪些是未定义行为？
    1. `uint32_t a = 0; a - 1;`
    2. `int32_t n = 1 << 31;`
    3. `std::vector<int> v; v.reserve(10); v[3] = 1;`
    4. 在 `for (auto& r : requests)` 循环里对 `requests` 调用 `push_back`
    5. `float f; std::memcpy(&f, bytes, 4);`（`bytes` 是任意地址的 `unsigned char*`）

??? success "参考答案"
    1. 不是。无符号运算按 $2^{32}$ 取模，结果是 4294967295（但这往往是 bug）。
    2. 是（C++20 之前）。`1 << 31` 超出了 `int` 能表示的范围；C++20 起有符号左移的结果按补码定义，这一条不再是 UB，但写成 `1u << 31` 或 `INT32_MIN` 更清楚。
    3. 是。`reserve` 只分配容量，不改变 `size()`，`v[3]` 访问的是不存在的元素（ASan 查不出，因为内存确实分配了；`-D_GLIBCXX_ASSERTIONS` 能查出）。要用 `resize`。
    4. 是。`push_back` 可能扩容，范围 for 循环里保存的迭代器随之失效。
    5. 不是。`memcpy` 对任意地址、任意类型都合法，这正是推荐的写法。

2. 下面的调度器代码想把新到的请求加入队列，并给"当前正在处理的请求"计数，但它有 bug。找出来并修好，要求修改后在 ASan 下运行，输出 `processed=3 queue=6`。

    ```cpp
    #include <cstdio>
    #include <vector>

    struct Request { int id; int steps = 0; };

    int main() {
      std::vector<Request> queue = {{1}, {2}, {3}};
      Request& current = queue[0];
      int processed = 0;
      for (int i = 0; i < 3; ++i) {
        queue.push_back({10 + i});      // 新请求到达
        current.steps++;                // 给当前请求记一步
        processed++;
      }
      std::printf("processed=%d queue=%zu steps=%d\n", processed, queue.size(), current.steps);
    }
    ```

??? success "参考答案"
    `current` 是指向 `queue` 内部元素的引用，`push_back` 扩容后它就悬垂了。保存下标，每次用的时候再取：

    ```cpp title="scheduler_fixed.cpp"
    #include <cstdio>
    #include <vector>

    struct Request {
      int id;
      int steps = 0;
    };

    int main() {
      std::vector<Request> queue = {{1}, {2}, {3}};
      std::size_t current = 0;         // 保存下标，而不是引用
      int processed = 0;
      for (int i = 0; i < 3; ++i) {
        queue.push_back({10 + i});
        queue[current].steps++;
        processed++;
      }
      std::printf("processed=%d queue=%zu\n", processed, queue.size());
    }
    ```

    ```text title="输出"
    processed=3 queue=6
    ```

    另一种修法是在循环前 `queue.reserve(...)`，但它依赖"不会超过预留容量"这个隐含前提，将来有人改了循环次数就又出错，不如保存下标稳妥。

## 小结

- [x] 每个 `.cpp` 独立编译成一个翻译单元，链接器按符号名把它们接起来；头文件只是被粘贴的文本。
- [x] 头文件里的函数和变量定义要么是 `inline`（全程序一份），要么是模板或类内定义；不要在头文件里写 `static` 变量。
- [x] 未定义行为不等于"结果随机"：编译器会假设它不发生并据此优化，出错的位置可能离真正的 bug 很远。
- [x] 常见的 UB：越界、悬垂引用（容器扩容）、有符号溢出、违反严格别名和对齐、未初始化读取、数据竞争。看位模式用 `std::bit_cast` 或 `memcpy`。
- [x] 开发构建默认开 ASan + UBSan，并发代码单独跑 TSan；它们是写 C++ 的基本工具，而不是出事后才用的调试器。
