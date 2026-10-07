# Templates, concepts and constexpr

<p class="lead">Open any inference operator library and what puts people off first is usually the templates: <code>template &lt;typename DTypeQ, typename DTypeKV, uint32_t HEAD_DIM, PosEncodingMode POS, bool USE_SLIDING_WINDOW&gt;</code>, wrapped in several layers of <code>DISPATCH_*</code> macros. This chapter explains what they are doing: turning parameters known only at run time into compile-time constants, so the compiler can generate the fastest code; and then using <code>constexpr</code>, <code>static_assert</code> and concepts to keep errors at compile time.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. A template function is called with 3 types. How many copies of the code are in the final binary?
    2. Why is `head_dim` a template parameter of an attention kernel rather than an ordinary function parameter?
    3. How does a `head_dim = 128` read at run time become a call to `kernel<128>`?
    4. How does `if constexpr` differ from an ordinary `if`?
    5. What problem with templates do concepts solve?

??? success "Answers (try it yourself first, then expand)"
    1. Three: every distinct set of template arguments instantiates its own copy of the code.
    2. As a template parameter it is a compile-time constant: the trip count, the array sizes and the register allocation can all be settled at compile time, so the compiler can unroll the loop completely and keep the data in registers; as an ordinary parameter none of that is possible.
    3. Write a dispatch: `switch (head_dim) { case 64: return kernel<64>(...); case 128: return kernel<128>(...); ... }`, mapping every supported run-time value to one instantiation (macros usually generate these branches), and raising an error for unsupported values.
    4. `if constexpr`'s condition is evaluated at compile time and the branch not taken is never instantiated, so it can contain code that is only valid for certain types; both branches of an ordinary `if` have to compile.
    5. They state the requirements on the type parameters at the template's entrance: when they are not met, the error says which constraint failed instead of hundreds of impenetrable lines from deep inside the template; and overloads can be selected by constraint.

## A template is a code generator {#模板就是代码生成器}

A template is not code but a recipe for generating code. For each new set of template arguments the compiler **instantiates** its own function or class:

```cpp title="template_basics.cpp"
#include <array>
#include <cstdio>

template <class T>
T dot(const T* a, const T* b, int n) {
  T s{};
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// HEAD_DIM is a compile-time constant: a fixed array size and a known trip count, so the compiler can unroll and vectorize completely
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

```text title="output"
dot<float>=64.0 dot<double>=30.0
qk_score<64>=32.0 qk_score<128>=64.0
```

`dot<float>`, `dot<double>`, `qk_score<64>` and `qk_score<128>` are four unrelated functions. Two immediate consequences:

- **performance**: `HEAD_DIM` is a constant, so `std::array<float, HEAD_DIM>` can live in registers and the loop can be unrolled completely. A GPU kernel depends on this even more, since a register array's size has to be known at compile time or it goes to much slower local memory;
- **the cost**: every extra value is another copy of the code. Data type × head dimension × causal or not × position encoding and so on combine into hundreds of instantiations, tens of minutes of compilation and hundreds of MB of binary. That is why libraries like FlashInfer moved to **JIT** (compiling only the combinations actually used, at run time).

## A run-time parameter into a compile-time constant: dispatch {#运行时参数--编译期常量派发}

![Figure: a run-time parameter becoming a compile-time constant - one switch, and the template instance in each branch treats it as a constant](../assets/figures/dispatch-compile-time.svg){.aig-svg}

A model's `head_dim` and data type are read from a configuration at run time, while the kernel needs compile-time constants. The way out is a `switch` mapping every supported value to an instantiation, which is what the various `DISPATCH_HEAD_DIM` and `AT_DISPATCH_FLOATING_TYPES` macros do. A C++20 generic lambda does it without macros:

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

// a "kernel" specialized by data type and head dimension: reducing one head
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
      constexpr int D = decltype(hd)::value;   // D is a genuine compile-time constant here
      return reduce_head<T, D>(x);
    });
  });
}

int main() {
  float xf[256];
  bf16 xb[256];
  for (int i = 0; i < 256; ++i) {
    xf[i] = 0.5f;
    xb[i] = bf16{0x3f80};   // 1.0 in bf16
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

```text title="output"
f32  head_dim=128 → 64.0
bf16 head_dim=64  → 64.0
不支持的 head_dim：96
```

The two levels of dispatch instantiated $2 \times 3 = 6$ copies of `reduce_head`. Reading an inference library's source, a chain of nested `DISPATCH_*` macros or lambdas is doing the same thing: turning a run-time configuration into template arguments level by level and calling a fully specialized kernel at the bottom.
The "unsupported head_dim" error is one you have probably met in practice: a model using a dimension the library never instantiated fails right here.

## `constexpr`: computing at compile time {#constexpr让计算发生在编译期}

A `constexpr` function can be evaluated at compile time (when its arguments are constants), and `static_assert` checks a condition at compile time. Together they turn "is this configuration valid" into a compilation error rather than a mysterious crash at run time:

```cpp title="constexpr_config.cpp"
#include <cstddef>
#include <cstdio>

struct TileConfig {
  int bm, bn, bk, stages;
};

constexpr std::size_t smem_bytes(TileConfig c, std::size_t elem) {
  return (std::size_t(c.bm) * c.bk + std::size_t(c.bk) * c.bn) * elem * c.stages;
}
constexpr std::size_t kSmemPerSm = 228 * 1024;   // an H100's shared memory per SM

template <class... Dims>
constexpr std::size_t numel(Dims... d) { return (std::size_t(d) * ... * 1); }   // a fold expression

template <TileConfig C, class T>   // C++20: a struct can be a non-type template parameter too
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

```text title="output"
K1：共享内存 96 KiB，每个 SM 2 个块
K2：共享内存 192 KiB，每个 SM 1 个块
```

If someone changes the configuration to a size that does not fit, the build fails outright with the message you wrote:

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

```text title="the compiler's error (excerpt)"
too_big.cpp:12:42: error: static assertion failed: tile 太大：共享内存放不下
```

`consteval` (C++20) goes further: the function **must** be evaluated at compile time. Libraries like CUTLASS and CuTe do a great deal of layout computation at compile time exactly through `constexpr` functions and template metaprogramming.

## `if constexpr`: a compile-time branch {#if-constexpr编译期分支}

Both branches of an ordinary `if` have to compile; `if constexpr`'s condition is evaluated at compile time and **the branch not taken is never instantiated**. It is commonly used to take a different implementation per type or constant:

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
    for (int i = 0; i < D; ++i) s += std::bit_cast<float>(std::uint32_t(x[i].bits) << 16);   // convert to float first, then accumulate
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
  for (auto& v : b) v = bf16{0x4000};   // 2.0 in bf16
  std::printf("%.1f\n", sum_head<float, 96>(f));
  std::printf("%.1f\n", sum_head<bf16, 64>(b));
  std::printf("%.1f\n", sum_head<float, 20>(f));
}
```

```text title="output"
24.0
128.0
（D=20 不是 8 的倍数：真实 kernel 在这里走不了 128 位的向量化加载）
5.0
```

## concepts: an "interface" for template parameters {#concepts给模板参数写接口}

With an unconstrained template, passing the wrong type produces an error deep **inside** the template, routinely hundreds of lines. C++20's **concepts** let you state the requirements at the template's entrance:

```cpp
template <class T>
concept Scalar = std::floating_point<T> || std::same_as<T, bf16>;

template <class A>
concept BlockAllocator = requires(A a, int n, int block) {
  { a.allocate(n) } -> std::same_as<std::vector<int>>;   // these member functions must exist and their return types must satisfy these constraints
  a.free(block);
  { a.num_free() } -> std::convertible_to<int>;
};

template <BlockAllocator A>
class Scheduler { /* ... */ };
```

When a requirement is not met, the compiler says which one:

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
  return static_cast<int>(first(tokens));   // an int is not a Scalar
}
```

```text title="the compiler's error (excerpt)"
error: no matching function for call to 'first(int [4])'
note:   constraints not satisfied
note: the expression 'std::floating_point<T> || std::same_as<T, bf16> [with T = int]' evaluated to 'false'
```

## A template or a virtual function {#模板还是虚函数}

C++ has two kinds of polymorphism:

| | virtual functions (run-time polymorphism) | templates (compile-time polymorphism) |
| --- | --- | --- |
| When the implementation is chosen | at run time, through an indirect call via the vtable | at compile time, called directly or even inlined |
| Copies of the code | one | one per set of arguments |
| Suits | coarse-grained interfaces called rarely: choosing an attention back end, a scheduling policy, a storage back end | fine-grained operations inside a hot loop: a kernel's internals, a function called for every element |

An inference framework usually combines them: a virtual function (or a `std::function` or a table of function pointers) picks "which attention back end" at startup, and the kernels inside that back end are all templates.
For an interface called a few dozen times per forward pass, the one indirect jump of a virtual call is negligible; for a function called for every element, the compiler has to be able to inline it.

!!! interview "Answering in an interview"
    Bring the template question down to inference kernels: a template is a code generator and each set of arguments instantiates its own copy; making `head_dim` and the tile size template parameters lets the compiler unroll loops, keep arrays in registers and compute the shared memory size, none of which a run-time parameter allows. A configuration read at run time becomes a compile-time constant through dispatch: a `switch` (or the dispatch macros in vLLM and FlashInfer) maps every supported value to one instantiation, at the cost of compilation time and binary size. `if constexpr`'s branch is chosen at compile time and the branch not taken is never instantiated; `constexpr` + `static_assert` turns a configuration error into a compilation error; concepts state the requirements on the parameters at the template's entrance. Coarse-grained "which implementation" goes to virtual functions, and a hot loop goes to templates.

## Exercises {#练习}

1. Add support for 96 to the `dispatch_head_dim` above, and have `reduce_head` check at compile time that `D` is a multiple of 8 (what a vectorized load requires). Implement it with `static_assert`, then say when and in what form an error would surface if someone added a `case 100:`.

??? success "Answer"
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

    ```text title="output"
    head_dim=64 → 64
    head_dim=96 → 96
    head_dim=128 → 128
    ```

    With `case 100:` added, `reduce_head<100>` is instantiated and the `static_assert` fails **at compile time** with exactly the message you wrote, rather than waiting for some user to load a head_dim=100 model and get wrong results in production.

2. An attention library has to support 3 data types, 4 head dimensions, causal or not (2) and 3 position encodings. How many kernels does instantiating all of them ahead of time take? What are the ways to control the compilation time and the binary size?

??? success "Answer"
    $3 \times 4 \times 2 \times 3 = 72$, each possibly split further by tile configuration. The common approaches:
    instantiate only the combinations actually used (generated from the list of supported models); turn the parameters that do not affect the inner loop into run-time parameters; spread the instantiations over several `.cpp` files and compile them in parallel;
    or do what FlashInfer does and use JIT, compiling a combination the first time it is met and caching it.

## Summary {#小结}

- [x] A template is a code generator: each set of arguments instantiates its own copy of the code, bought with compile-time constants and thorough optimization.
- [x] A run-time configuration becomes a compile-time constant through dispatch: a `switch` maps every supported value to an instantiation, and a few nested levels give a fully specialized kernel.
- [x] `constexpr` + `static_assert` turns a configuration error into a compilation error; `if constexpr` picks an implementation by type or constant, and the branch not taken is never instantiated.
- [x] concepts state the requirements on the parameters at the template's entrance, with far clearer errors.
- [x] Coarse-grained "which implementation" goes to virtual functions, and operations in a hot loop go to templates.
