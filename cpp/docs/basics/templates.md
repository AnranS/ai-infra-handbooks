# 模板、concepts 与 constexpr

<p class="lead">打开任何一个推理算子库，最先劝退人的往往是模板：<code>template &lt;typename DTypeQ, typename DTypeKV, uint32_t HEAD_DIM, PosEncodingMode POS, bool USE_SLIDING_WINDOW&gt;</code>，外面再套几层 <code>DISPATCH_*</code> 宏。这一章讲清楚它们在做什么：把运行时才知道的参数变成编译期常量，让编译器生成最快的代码；再用 <code>constexpr</code>、<code>static_assert</code> 和 concepts 把错误挡在编译期。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 模板函数被 3 种类型调用，最终的二进制里有几份代码？
    2. 为什么注意力 kernel 要把 `head_dim` 写成模板参数，而不是普通的函数参数？
    3. 运行时读到的 `head_dim = 128`，怎样变成 `kernel<128>` 的调用？
    4. `if constexpr` 和普通的 `if` 有什么区别？
    5. concepts 解决了模板的什么问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 三份：每组不同的模板参数都实例化出一份独立的代码。
    2. 作为模板参数它是编译期常量：循环次数、数组大小、寄存器分配都能在编译期确定，编译器可以完全展开循环、把数据放进寄存器；作为普通参数这些都做不到。
    3. 写一个派发：`switch (head_dim) { case 64: return kernel<64>(...); case 128: return kernel<128>(...); ... }`，把每个支持的运行时取值映射到一个实例化（常用宏生成这些分支），不支持的取值报错。
    4. `if constexpr` 的条件在编译期求值，没选中的分支根本不会被实例化，所以里面可以写只对某些类型成立的代码；普通 `if` 的两个分支都要能编译。
    5. 在模板的入口写明对类型参数的要求：不满足时，报错直接指出哪个约束不满足，而不是在模板深处报出几百行难懂的错误；还能按约束选择重载。

## 模板就是代码生成器

模板本身不是代码，是生成代码的配方。每用一组新的模板参数，编译器就**实例化**出一份独立的函数或类：

```cpp title="template_basics.cpp"
#include <array>
#include <cstdio>

template <class T>
T dot(const T* a, const T* b, int n) {
  T s{};
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// HEAD_DIM 是编译期常量：数组大小固定、循环次数已知，编译器可以完全展开、向量化
template <int HEAD_DIM>
float qk_score(const float* q, const float* k) {
  std::array<float, HEAD_DIM> prod{};
  for (int i = 0; i < HEAD_DIM; ++i) prod[i] = q[i] * k[i];
  float s = 0;
  for (float p : prod) s += p;
  return s;
}

int main() {
  float q[128], k[128];
  double qd[4] = {1, 2, 3, 4};
  for (int i = 0; i < 128; ++i) {
    q[i] = 1.0f;
    k[i] = 0.5f;
  }
  std::printf("dot<float>=%.1f dot<double>=%.1f\n", dot(q, k, 128), dot(qd, qd, 4));
  std::printf("qk_score<64>=%.1f qk_score<128>=%.1f\n", qk_score<64>(q, k), qk_score<128>(q, k));
}
```

```text title="输出"
dot<float>=64.0 dot<double>=30.0
qk_score<64>=32.0 qk_score<128>=64.0
```

`dot<float>`、`dot<double>`、`qk_score<64>`、`qk_score<128>` 是四个互不相关的函数。两个直接后果：

- **性能**：`HEAD_DIM` 是常量，`std::array<float, HEAD_DIM>` 可以放在寄存器里，循环可以完全展开。GPU kernel 更依赖这一点——寄存器数组的大小必须在编译期确定，否则只能放进慢得多的本地内存；
- **代价**：每多一个取值就多一份代码。数据类型 × 头维度 × 是否因果 × 位置编码……组合起来就是几百份实例，编译几十分钟、二进制几百 MB。这就是 FlashInfer 这类库改用 **JIT**（运行时按需编译用到的那几种组合）的原因。

## 运行时参数 → 编译期常量：派发

模型的 `head_dim`、数据类型是运行时从配置里读出来的，而 kernel 需要编译期常量。办法是写一个 `switch`，把每个支持的取值映射到一个实例化——这就是各种 `DISPATCH_HEAD_DIM`、`AT_DISPATCH_FLOATING_TYPES` 宏做的事。用 C++20 的泛型 lambda 可以不用宏：

```cpp title="dispatch.cpp"
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

struct bf16 {
  std::uint16_t bits;
};
inline float to_float(float x) { return x; }
inline float to_float(bf16 x) { return std::bit_cast<float>(std::uint32_t(x.bits) << 16); }

enum class DType { F32, BF16 };

template <class T>
struct type_tag {
  using type = T;
};

// 一个按数据类型和头维度特化的 "kernel"：对一个头做归约
template <class T, int D>
float reduce_head(const void* x) {
  const T* p = static_cast<const T*>(x);
  float s = 0;
  for (int i = 0; i < D; ++i) s += to_float(p[i]);
  return s;
}

template <class F>
decltype(auto) dispatch_dtype(DType t, F&& f) {
  switch (t) {
    case DType::F32: return f(type_tag<float>{});
    case DType::BF16: return f(type_tag<bf16>{});
  }
  throw std::invalid_argument("未知的数据类型");
}

template <class F>
decltype(auto) dispatch_head_dim(int head_dim, F&& f) {
  switch (head_dim) {
    case 64: return f(std::integral_constant<int, 64>{});
    case 128: return f(std::integral_constant<int, 128>{});
    case 256: return f(std::integral_constant<int, 256>{});
    default: throw std::invalid_argument("不支持的 head_dim：" + std::to_string(head_dim));
  }
}

float run(DType dtype, int head_dim, const void* x) {
  return dispatch_dtype(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return dispatch_head_dim(head_dim, [&](auto hd) {
      constexpr int D = decltype(hd)::value;   // 在这里 D 是真正的编译期常量
      return reduce_head<T, D>(x);
    });
  });
}

int main() {
  float xf[256];
  bf16 xb[256];
  for (int i = 0; i < 256; ++i) {
    xf[i] = 0.5f;
    xb[i] = bf16{0x3f80};   // bf16 的 1.0
  }
  std::printf("f32  head_dim=128 → %.1f\n", run(DType::F32, 128, xf));
  std::printf("bf16 head_dim=64  → %.1f\n", run(DType::BF16, 64, xb));
  try {
    run(DType::F32, 96, xf);
  } catch (const std::invalid_argument& e) {
    std::printf("%s\n", e.what());
  }
}
```

```text title="输出"
f32  head_dim=128 → 64.0
bf16 head_dim=64  → 64.0
不支持的 head_dim：96
```

两层派发实例化了 $2 \times 3 = 6$ 份 `reduce_head`。读推理库源码时，看到一串嵌套的 `DISPATCH_*` 宏或 lambda，就知道它们在做同样的事：把运行时的配置一层层变成模板参数，最里面调用一个完全特化的 kernel。
"不支持的 head_dim" 这个报错你大概在实际使用中也见过——某个模型用了库没有实例化的维度，就会在这里失败。

## `constexpr`：让计算发生在编译期

`constexpr` 函数可以在编译期求值（参数是常量时），`static_assert` 在编译期检查条件。两者配合，可以把"这个配置合不合法"变成编译错误，而不是运行时的神秘崩溃：

```cpp title="constexpr_config.cpp"
#include <cstddef>
#include <cstdio>

struct TileConfig {
  int bm, bn, bk, stages;
};

constexpr std::size_t smem_bytes(TileConfig c, std::size_t elem) {
  return (std::size_t(c.bm) * c.bk + std::size_t(c.bk) * c.bn) * elem * c.stages;
}
constexpr std::size_t kSmemPerSm = 228 * 1024;   // H100 每个 SM 的共享内存

template <class... Dims>
constexpr std::size_t numel(Dims... d) { return (std::size_t(d) * ... * 1); }   // 折叠表达式

template <TileConfig C, class T>   // C++20：结构体也可以当非类型模板参数
struct GemmKernel {
  static constexpr std::size_t smem = smem_bytes(C, sizeof(T));
  static_assert(smem <= kSmemPerSm, "tile 太大：共享内存放不下");
  static_assert(C.bm % 16 == 0 && C.bn % 16 == 0, "tile 边长必须是 16 的倍数（Tensor Core 的要求）");
  static constexpr int blocks_per_sm = int(kSmemPerSm / (smem + 1024));
};

int main() {
  using K1 = GemmKernel<TileConfig{128, 128, 64, 3}, short>;
  using K2 = GemmKernel<TileConfig{128, 256, 64, 4}, short>;
  static_assert(numel(2, 3, 4) == 24);
  std::printf("K1：共享内存 %zu KiB，每个 SM %d 个块\n", K1::smem / 1024, K1::blocks_per_sm);
  std::printf("K2：共享内存 %zu KiB，每个 SM %d 个块\n", K2::smem / 1024, K2::blocks_per_sm);
}
```

```text title="输出"
K1：共享内存 96 KiB，每个 SM 2 个块
K2：共享内存 192 KiB，每个 SM 1 个块
```

如果有人把配置改成放不下的大小，编译直接失败，报错里就是你写的那句话：

```cpp title="too_big.cpp" expect="compile-error" error="tile 太大"
#include <cstddef>

struct TileConfig {
  int bm, bn, bk, stages;
};
constexpr std::size_t smem_bytes(TileConfig c, std::size_t elem) {
  return (std::size_t(c.bm) * c.bk + std::size_t(c.bk) * c.bn) * elem * c.stages;
}

template <TileConfig C, class T>
struct GemmKernel {
  static_assert(smem_bytes(C, sizeof(T)) <= 228 * 1024, "tile 太大：共享内存放不下");
};

int main() { GemmKernel<TileConfig{256, 256, 64, 4}, float> k; (void)k; }
```

```text title="编译器的报错（节选）"
too_big.cpp:12:42: error: static assertion failed: tile 太大：共享内存放不下
```

`consteval`（C++20）更进一步：函数**必须**在编译期求值。CUTLASS、CuTe 这类库把大量的布局计算放在编译期，就是靠 `constexpr` 函数和模板元编程。

## `if constexpr`：编译期分支

普通的 `if` 两个分支都要能编译；`if constexpr` 的条件在编译期求值，**没选中的分支不会被实例化**。常用来按类型或常量走不同的实现：

```cpp title="if_constexpr.cpp"
#include <bit>
#include <concepts>
#include <cstdint>
#include <cstdio>

struct bf16 {
  std::uint16_t bits;
};

template <class T>
concept Scalar = std::floating_point<T> || std::same_as<T, bf16>;

template <Scalar T, int D>
float sum_head(const T* x) {
  float s = 0;
  if constexpr (std::same_as<T, bf16>) {
    for (int i = 0; i < D; ++i) s += std::bit_cast<float>(std::uint32_t(x[i].bits) << 16);   // 先转 float 再累加
  } else {
    for (int i = 0; i < D; ++i) s += static_cast<float>(x[i]);
  }
  if constexpr (D % 8 != 0) {
    std::printf("（D=%d 不是 8 的倍数：真实 kernel 在这里走不了 128 位的向量化加载）\n", D);
  }
  return s;
}

int main() {
  float f[96];
  bf16 b[64];
  for (auto& v : f) v = 0.25f;
  for (auto& v : b) v = bf16{0x4000};   // bf16 的 2.0
  std::printf("%.1f\n", sum_head<float, 96>(f));
  std::printf("%.1f\n", sum_head<bf16, 64>(b));
  std::printf("%.1f\n", sum_head<float, 20>(f));
}
```

```text title="输出"
24.0
128.0
（D=20 不是 8 的倍数：真实 kernel 在这里走不了 128 位的向量化加载）
5.0
```

## concepts：给模板参数写"接口"

没有约束的模板，传错类型时报错会出现在模板**内部**深处，动辄几百行。C++20 的 **concepts** 让你在模板的"入口"写明要求：

```cpp
template <class T>
concept Scalar = std::floating_point<T> || std::same_as<T, bf16>;

template <class A>
concept BlockAllocator = requires(A a, int n, int block) {
  { a.allocate(n) } -> std::same_as<std::vector<int>>;   // 必须有这些成员函数，返回值满足这些约束
  a.free(block);
  { a.num_free() } -> std::convertible_to<int>;
};

template <BlockAllocator A>
class Scheduler { /* ... */ };
```

不满足要求时，编译器直接告诉你是哪一条不满足：

```cpp title="concept_error.cpp" expect="compile-error" error="constraints not satisfied"
#include <concepts>
#include <cstdint>

struct bf16 {
  std::uint16_t bits;
};
template <class T>
concept Scalar = std::floating_point<T> || std::same_as<T, bf16>;

template <Scalar T>
float first(const T* x) { return static_cast<float>(x[0]); }

int main() {
  int tokens[4] = {1, 2, 3, 4};
  return static_cast<int>(first(tokens));   // int 不是 Scalar
}
```

```text title="编译器的报错（节选）"
error: no matching function for call to 'first(int [4])'
note:   constraints not satisfied
note: the expression 'std::floating_point<T> || std::same_as<T, bf16> [with T = int]' evaluated to 'false'
```

## 模板还是虚函数

C++ 有两种多态：

| | 虚函数（运行时多态） | 模板（编译期多态） |
| --- | --- | --- |
| 选择实现的时机 | 运行时，通过虚函数表间接调用 | 编译期，直接调用甚至内联 |
| 代码份数 | 一份 | 每种参数一份 |
| 适合 | 粗粒度、调用次数少的接口：注意力后端的选择、调度策略、存储后端 | 细粒度、在热循环里的操作：kernel 内部、每个元素都要调用的函数 |

推理框架通常两者结合：用虚函数（或者 `std::function`、函数指针表）在启动时选定"用哪个注意力后端"，而后端内部的 kernel 全是模板。
一次前向只调用几十次的接口，虚函数那一次间接跳转的开销可以忽略；每个元素都要调用的函数，就必须让编译器能内联。

## 练习

1. 给上面的 `dispatch_head_dim` 加上 96 的支持，并让 `reduce_head` 在编译期检查 `D` 是 8 的倍数（向量化加载的要求）。用 `static_assert` 实现，然后说明：如果有人加上 `case 100:`，会在什么时候、以什么形式发现错误？

??? success "参考答案"
    ```cpp title="dispatch96.cpp"
    #include <cstdio>
    #include <stdexcept>
    #include <type_traits>

    template <int D>
    float reduce_head(const float* x) {
      static_assert(D % 8 == 0, "head_dim 必须是 8 的倍数：kernel 按 8 个元素一组做向量化加载");
      float s = 0;
      for (int i = 0; i < D; ++i) s += x[i];
      return s;
    }

    template <class F>
    decltype(auto) dispatch_head_dim(int head_dim, F&& f) {
      switch (head_dim) {
        case 64: return f(std::integral_constant<int, 64>{});
        case 96: return f(std::integral_constant<int, 96>{});
        case 128: return f(std::integral_constant<int, 128>{});
        case 256: return f(std::integral_constant<int, 256>{});
        default: throw std::invalid_argument("不支持的 head_dim");
      }
    }

    int main() {
      float x[256];
      for (auto& v : x) v = 1.0f;
      for (int hd : {64, 96, 128}) {
        float r = dispatch_head_dim(hd, [&](auto d) { return reduce_head<decltype(d)::value>(x); });
        std::printf("head_dim=%d → %.0f\n", hd, r);
      }
    }
    ```

    ```text title="输出"
    head_dim=64 → 64
    head_dim=96 → 96
    head_dim=128 → 128
    ```

    加上 `case 100:` 之后，`reduce_head<100>` 被实例化，`static_assert` 在**编译期**失败，报错信息就是你写的那句话——而不是等到某个用户加载了一个 head_dim=100 的模型，才在线上得到错误的结果。

2. 一个注意力库要支持 3 种数据类型、4 种 head_dim、是否因果（2 种）、3 种位置编码。全部预先实例化需要多少份 kernel？有哪些办法控制编译时间和二进制大小？

??? success "参考答案"
    $3 \times 4 \times 2 \times 3 = 72$ 份，每份还可能按 tile 配置再分几种。常见的办法：
    只实例化真正会用到的组合（按支持的模型列表生成）；把不影响内层循环的参数改成运行时参数；把实例化分散到多个 `.cpp` 里并行编译；
    或者像 FlashInfer 那样用 JIT，第一次遇到某个组合时才编译并缓存。

## 小结

- [x] 模板是代码生成器：每组模板参数实例化一份独立的代码，换来编译期已知的常量和彻底的优化。
- [x] 运行时配置到编译期常量靠派发：一个 `switch` 把每个支持的取值映射到一个实例化，嵌套几层就得到完全特化的 kernel。
- [x] `constexpr` + `static_assert` 把配置错误变成编译错误；`if constexpr` 按类型或常量选择实现，没选中的分支不会被实例化。
- [x] concepts 在模板入口写明对参数的要求，报错清楚得多。
- [x] 粗粒度的"选哪个实现"用虚函数，热循环里的操作用模板。
