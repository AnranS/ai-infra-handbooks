# 移动语义与完美转发

<p class="lead">值语义让代码好推理，但拷贝可能很贵，而独占的资源（显存、文件、通信句柄）根本不能拷贝。移动语义解决的就是"把东西交出去而不复制"：转移所有权，只搬几个指针。这一章讲清楚右值引用、<code>std::move</code> 到底做了什么、为什么移动构造要标 <code>noexcept</code>，以及模板里的完美转发。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `std::move(x)` 会移动任何东西吗？
    2. 一个对象被移动之后，还能做什么、不能做什么？
    3. 为什么 `std::vector` 扩容时，元素类型的移动构造函数没标 `noexcept` 就会退回到拷贝？
    4. `return std::move(local);` 有什么问题？
    5. `template <class T> void f(T&& x)` 里的 `T&&` 和 `void g(std::string&& x)` 里的 `&&` 有什么区别？为什么要写 `std::forward<T>(x)`？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 不会：它只是把实参转换成右值引用类型，真正转移资源的是被调用的移动构造函数或移动赋值运算符（如果有的话）。
    2. 只能析构它，或者给它重新赋值；标准库类型处于"有效但未指定"的状态，不要依赖它的值（除非类型有明确的规定，比如 `unique_ptr` 移动后为空）。
    3. 扩容要把旧元素搬到新内存，如果移动到一半抛了异常，已经移走的元素没法恢复，破坏了 `push_back` 的强异常保证；拷贝不会破坏原来的元素。所以只有移动构造是 `noexcept` 的（或者类型不能拷贝）时才移动。
    4. 它阻止了返回值优化（NRVO）：本来可以直接在调用方的位置构造、连移动都不需要，写了 `std::move` 反而强制一次移动；而且返回局部变量时编译器本来就会自动把它当作右值。
    5. 模板里的 `T&&` 是转发引用：传左值时 `T` 推导成左值引用，折叠后是左值引用；传右值时才是右值引用。`std::string&&` 只能绑定右值。有名字的参数本身是左值，`std::forward<T>(x)` 按 `T` 恢复它原来的值类别，原样转发给下一个函数。

## 左值、右值与右值引用

粗略地说：

- **左值**是有名字、可以取地址的东西：变量、`v[i]`、`*p`；
- **右值**是马上就要消失的临时值：`f()` 的返回值、`std::string("tmp")`、`a + b`。

右值马上就要被销毁，它持有的资源没人会再用——那就可以直接"偷"过来，而不必复制。**右值引用** `T&&` 就是专门绑定到右值上的引用，重载一个参数为 `T&&` 的构造函数，就能在"源对象是临时的"时候走偷资源的路径：

```cpp
std::vector<int> a = make_tokens();   // make_tokens() 是右值：移动（实际上会被省略）
std::vector<int> b = a;               // a 是左值：拷贝，a 以后还要用
std::vector<int> c = std::move(a);    // 明确表示"a 我不要了"：移动
```

**`std::move` 本身什么都不移动**，它只是一个类型转换：把左值转成右值引用（`static_cast<T&&>(x)`），告诉重载决议"可以偷它"。真正干活的是被选中的**移动构造函数**或**移动赋值运算符**。

## 给资源类写上移动

接着上一章的 `DeviceBuffer`：它禁止了拷贝，现在加上移动。移动构造把源对象的指针拿过来，再把源对象置空，让它的析构函数什么都不做：

```cpp title="move_buffer.cpp"
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static int live = 0;

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : ptr_(std::malloc(bytes)), bytes_(bytes) { ++live; }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();                                 // 先释放自己原来的
      ptr_ = std::exchange(o.ptr_, nullptr);
      bytes_ = std::exchange(o.bytes_, 0);
    }
    return *this;
  }

  std::size_t size() const { return bytes_; }

 private:
  void reset() {
    if (ptr_) {
      std::free(ptr_);
      --live;
      ptr_ = nullptr;
    }
  }
  void* ptr_;
  std::size_t bytes_;
};

DeviceBuffer allocate_kv(std::size_t blocks) { return DeviceBuffer(blocks * 4096); }

int main() {
  DeviceBuffer a = allocate_kv(16);   // 返回值直接构造到 a，既不拷贝也不移动
  DeviceBuffer b = std::move(a);      // 移动：b 接管内存，a 变成空壳
  std::printf("a.size=%zu b.size=%zu live=%d\n", a.size(), b.size(), live);

  std::vector<DeviceBuffer> pool;
  pool.push_back(std::move(b));       // 移动进 vector
  pool.emplace_back(8192);            // 直接在 vector 的内存里构造
  std::printf("pool=%zu live=%d\n", pool.size(), live);

  a = DeviceBuffer(100);              // 移动赋值：a 重新持有一块资源
  std::printf("a.size=%zu live=%d\n", a.size(), live);
}
```

```text title="输出"
a.size=0 b.size=65536 live=1
pool=2 live=2
a.size=100 live=3
```

几个要点：

- `std::exchange(x, v)` 把 `x` 设为 `v` 并返回旧值，写移动操作时非常顺手；
- 析构函数要能处理"已经被移走"的空对象（`reset()` 里判空）；
- 移动赋值要先释放自己原来持有的资源，并处理 `a = std::move(a)` 这种自我赋值；
- 移动操作只搬指针，不会失败，所以标上 `noexcept`——下一节说明这为什么重要。

### 被移动之后的对象

标准库对"被移走的对象"的承诺是：**处于有效但未指定的状态**。可以对它做不依赖其值的操作——析构、重新赋值、`clear()`——但不要读它的内容。自己写的类最好让被移走的对象处于一个明确的"空"状态（像上面的 `size() == 0`）。

## `noexcept` 与 `vector` 扩容

`std::vector` 扩容时要把旧缓冲区里的元素搬到新缓冲区。它有一个很强的承诺：**如果搬的过程中抛出异常，vector 保持原样**（强异常安全保证）。
用拷贝搬，失败了旧缓冲区还完好，可以回滚；用移动搬，搬到一半失败时，一部分元素已经被掏空，回不去了。所以 vector 的规则是：**只有移动构造函数保证不抛异常（`noexcept`）时才用移动，否则退回到拷贝**。

```cpp title="noexcept_realloc.cpp"
#include <cstdio>
#include <vector>

struct Counts {
  int copies = 0, moves = 0;
};

template <bool Noexcept>
struct Item {
  static inline Counts c;
  Item() = default;
  Item(const Item&) { ++c.copies; }
  Item(Item&&) noexcept(Noexcept) { ++c.moves; }
};

template <bool N>
void run(const char* name) {
  std::vector<Item<N>> v;
  for (int i = 0; i < 5; ++i) v.push_back(Item<N>{});   // 容量 1 → 2 → 4 → 8，扩容 3 次
  std::printf("%s：copies=%d moves=%d\n", name, Item<N>::c.copies, Item<N>::c.moves);
}

int main() {
  run<false>("移动构造没有 noexcept");
  run<true>("移动构造标了 noexcept");
}
```

```text title="输出"
移动构造没有 noexcept：copies=7 moves=5
移动构造标了 noexcept：copies=0 moves=12
```

5 次 `push_back` 本身各移动一次；3 次扩容一共搬了 $1 + 2 + 4 = 7$ 个元素——没有 `noexcept` 时这 7 次全是拷贝。元素如果是一个装着几 MB 数据的对象，差别就是几十 MB 的内存复制。
**规则：移动构造和移动赋值只要不会失败，就标 `noexcept`。**编译器生成的移动操作会自动推导 `noexcept`（所有成员的移动都是 `noexcept` 时它就是）。

## 按值传参再移动

一个函数要**保存**参数的一份拷贝（构造函数把参数存成成员是最常见的情况），最简单又高效的写法是按值接收、再移动进去：

```cpp
class Request {
 public:
  Request(std::string prompt, std::vector<int> tokens)
      : prompt_(std::move(prompt)), tokens_(std::move(tokens)) {}
 private:
  std::string prompt_;
  std::vector<int> tokens_;
};

Request r1(prompt, tokens);                          // 调用者还要用：各拷贝一次
Request r2(std::move(prompt), tokenize(text));       // 调用者不要了：全程只有移动
```

这样一个构造函数就同时覆盖了"调用者传左值"和"传右值"两种情况，不需要为 `const T&` 和 `T&&` 各写一个重载。

!!! warning "不要写 `return std::move(local);`"
    返回局部变量时，编译器会先尝试**省略拷贝**（NRVO，直接把局部变量构造在返回值的位置），做不到时也会自动把它当右值移动。
    手动写 `std::move` 反而**阻止了省略拷贝**，`g++ -Wall` 会给出 `-Wpessimizing-move` 警告。直接 `return local;`。

## 完美转发

写一个通用的包装函数——比如对象池的 `acquire(args...)`、`vector::emplace_back`、`std::make_unique`——要把参数**原样**传给另一个函数：调用者传左值，就按左值传下去；传右值，就按右值传下去。

模板参数里的 `T&&` 是一个特殊情况，叫**转发引用**：传左值时 `T` 被推导成 `U&`，`T&&` 折叠成 `U&`；传右值时 `T` 是 `U`，`T&&` 就是 `U&&`。
但在函数体里，参数 `x` 有名字，它**总是左值**。要恢复调用者原来的值类别，就用 `std::forward<T>(x)`：

```cpp title="forwarding.cpp"
#include <cstdio>
#include <string>
#include <utility>

void consume(const std::string&) { std::printf("  拷贝进来（左值）\n"); }
void consume(std::string&&) { std::printf("  移动进来（右值）\n"); }

template <class T>
void wrong(T&& x) { consume(x); }                    // x 有名字，是左值：永远走拷贝

template <class T>
void right(T&& x) { consume(std::forward<T>(x)); }   // 保持调用者的值类别

int main() {
  std::string s = "prompt";
  std::printf("wrong(s)：\n");
  wrong(s);
  std::printf("wrong(临时对象)：\n");
  wrong(std::string("tmp"));
  std::printf("right(s)：\n");
  right(s);
  std::printf("right(临时对象)：\n");
  right(std::string("tmp"));
}
```

```text title="输出"
wrong(s)：
  拷贝进来（左值）
wrong(临时对象)：
  拷贝进来（左值）
right(s)：
  拷贝进来（左值）
right(临时对象)：
  移动进来（右值）
```

可变参数版本的写法（`emplace_back` 就是这样实现的）：

```cpp
template <class T, class... Args>
T* construct_at_slot(void* slot, Args&&... args) {
  return new (slot) T(std::forward<Args>(args)...);   // 在已有内存上原地构造，参数原样转发
}
```

`emplace_back(8192)` 比 `push_back(DeviceBuffer(8192))` 少一次移动：它把参数转发给构造函数，直接在 vector 的内存里构造元素。

## 练习

1. 下面的 `BlockTable` 用裸数组保存一个请求的 KV 块号。按五法则补全它：拷贝要深拷贝，移动要转移所有权并标 `noexcept`，赋值要正确处理自我赋值。要求在 ASan 下运行，输出和参考答案一致。

    ```cpp
    class BlockTable {
     public:
      explicit BlockTable(std::size_t n);
      ~BlockTable();
      // 补全：拷贝构造、拷贝赋值、移动构造、移动赋值
      int& operator[](std::size_t i);
      std::size_t size() const;
     private:
      std::size_t n_;
      int* ids_;
    };
    ```

??? success "参考答案"
    拷贝赋值用"拷贝再交换"的写法：先把 `o` 拷贝到一个临时对象（可能抛异常，但此时 `*this` 还没被改动），再用不会失败的 `swap` 交换，天然处理了自我赋值，也给出了强异常安全保证。

    ```cpp title="block_table.cpp"
    #include <algorithm>
    #include <cstdio>
    #include <utility>
    #include <vector>

    class BlockTable {
     public:
      explicit BlockTable(std::size_t n) : n_(n), ids_(new int[n]) { std::fill(ids_, ids_ + n_, -1); }
      ~BlockTable() { delete[] ids_; }

      BlockTable(const BlockTable& o) : n_(o.n_), ids_(new int[o.n_]) { std::copy(o.ids_, o.ids_ + n_, ids_); }
      BlockTable& operator=(const BlockTable& o) {
        BlockTable tmp(o);   // 可能抛异常，但 *this 还没动
        swap(tmp);           // 不会失败
        return *this;        // tmp 带着旧数据析构
      }
      BlockTable(BlockTable&& o) noexcept : n_(std::exchange(o.n_, 0)), ids_(std::exchange(o.ids_, nullptr)) {}
      BlockTable& operator=(BlockTable&& o) noexcept {
        if (this != &o) {
          delete[] ids_;
          n_ = std::exchange(o.n_, 0);
          ids_ = std::exchange(o.ids_, nullptr);
        }
        return *this;
      }
      void swap(BlockTable& o) noexcept {
        std::swap(n_, o.n_);
        std::swap(ids_, o.ids_);
      }

      int& operator[](std::size_t i) { return ids_[i]; }
      std::size_t size() const { return n_; }

     private:
      std::size_t n_;
      int* ids_;
    };

    int main() {
      BlockTable a(4);
      a[0] = 7;
      BlockTable b = a;                // 深拷贝
      b[0] = 9;
      std::printf("拷贝：a[0]=%d b[0]=%d\n", a[0], b[0]);

      BlockTable& same = b;
      b = same;                        // 自我赋值
      std::printf("自我赋值后：b[0]=%d\n", b[0]);

      BlockTable c = std::move(a);     // 移动
      std::printf("移动：a.size=%zu c.size=%zu c[0]=%d\n", a.size(), c.size(), c[0]);

      std::vector<BlockTable> tables;
      for (int i = 0; i < 5; ++i) tables.emplace_back(2);   // 扩容时走 noexcept 的移动
      std::printf("tables=%zu\n", tables.size());
    }
    ```

    ```text title="输出"
    拷贝：a[0]=7 b[0]=9
    自我赋值后：b[0]=9
    移动：a.size=0 c.size=4 c[0]=7
    tables=5
    ```

    当然，真实代码里直接用 `std::vector<int>` 当成员（零法则）就够了。手写五法则的价值在于理解标准库容器和智能指针内部在做什么。

2. 下面的代码在把请求加入批次时想省掉一次拷贝，但统计出来的 token 数总是 0。为什么？

    ```cpp
    batch.tokens.insert(batch.tokens.end(),
                        std::make_move_iterator(req.tokens.begin()), std::make_move_iterator(req.tokens.end()));
    batch.requests.push_back(std::move(req));
    stats.total_tokens += req.tokens.size();
    ```

??? success "参考答案"
    `req` 在第二行已经被移动进了 `batch.requests`，第三行读的是一个被移走的对象，`req.tokens` 通常已经是空的（标准只保证"有效但未指定"）。
    被移动之后只能析构或重新赋值，不能再读。把统计挪到移动之前，或者改成从 `batch.requests.back()` 读。另外，`int` 这种基础类型"移动"和拷贝一样，第一行的 `make_move_iterator` 没有意义。

## 小结

- [x] `std::move` 只是类型转换，真正转移资源的是移动构造函数和移动赋值运算符。
- [x] 资源类的移动操作：接管指针、把源对象置空、析构时判空；移动赋值先释放自己原有的资源。
- [x] 被移动的对象处于"有效但未指定"的状态，只能析构或重新赋值。
- [x] 移动操作标 `noexcept`，否则 `vector` 扩容时会退回到拷贝。
- [x] 要保存参数时按值接收再移动；返回局部变量不要写 `std::move`。
- [x] 模板里的 `T&&` 是转发引用，配合 `std::forward<T>` 原样转发参数；`emplace_back` 直接在容器里构造元素。
